from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dotgat_lrmc_agents import (
    AgentMatrixReconstructionDataset,
    MultiHeadDotGAT,
    evaluate,
    spectral_penalty,
    train,
)
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(scope="module")
def setup_dataset():
    """
    Create a small dataset for consistent testing.
    """
    dataset = AgentMatrixReconstructionDataset(
        num_graphs=4,
        n=8,
        m=8,
        r=2,
        num_agents=4,
        view_mode='sparse', 
        density=0.5,
        sigma=0.01
    )
    return dataset

def test_dataset_shape_and_consistency(setup_dataset):
    """
    Ensure data dimensions are correct, masks align with target, and agent input shapes are expected.
    """
    print("\n[TEST] Verifying dataset shapes and agent distribution...")
    sample = setup_dataset[0].to(device)
    assert sample.x.shape == (4, 64)  # 4 agents, flattened 8x8 input
    assert sample.y.shape == (64,)
    assert sample.mask.shape == (64,)
    assert sample.x.dtype == torch.float
    assert sample.y.dtype == torch.float
    assert sample.mask.dtype == torch.bool
    print("PASSED: Shapes and types are correct.")

    # Check that agent views are masked correctly (sparse mode)
    zeros_per_agent = [(agent == 0).sum().item() for agent in sample.x]
    assert all(z > 0 for z in zeros_per_agent), "Each agent should have some zero-masked entries"
    print("PASSED: Agents receive partial masked views as expected.")

def test_projection_mode_distribution():
    """
    Validate that each agent gets a different projection in projection mode.
    """
    print("\n[TEST] Validating projection mode produces unique per-agent projections...")
    dataset = AgentMatrixReconstructionDataset(
        num_graphs=1, n=8, m=8, r=2,
        num_agents=4, view_mode='project', density=0.5, sigma=0.01
    )
    sample = dataset[0].to(device)
    differences = [(sample.x[i] - sample.x[j]).abs().sum().item() for i in range(4) for j in range(i+1, 4)]
    assert all(d > 0 for d in differences), "Each agent projection should be distinct"
    print("PASSED: All agent projections are distinct in projection mode.")

def test_model_forward_pass(setup_dataset):
    """
    Ensure the GAT model forward pass produces expected output dimensions.
    """
    print("\n[TEST] Verifying model forward pass output shape...")
    first_sample = setup_dataset[0]
    model = MultiHeadDotGAT(
        in_features=first_sample.x.shape[1],
        hidden_features=32,
        output_dim=first_sample.y.shape[0],  # inferred directly
        num_heads=2,
        dropout=0.1,
        agg_mode='concat'
    ).to(device)
    model.eval()
    sample = setup_dataset[0].to(device)
    out = model(sample.x)
    assert out.shape == (4, 64)
    print("PASSED: Model forward pass outputs correct shape.")

def test_spectral_penalty_behavior():
    """
    Confirm spectral penalty returns valid quantities and enforces correct conditions.
    """
    print("\n[TEST] Evaluating spectral penalty structure and stability...")
    mat = torch.randn(5, 5)
    penalty, first_sv, gap = spectral_penalty(mat)
    assert penalty >= 0
    assert gap >= 0
    assert isinstance(penalty, torch.Tensor) or isinstance(penalty, float)
    print(f"Returned spectral gap: {gap:.4f}, First singular value: {first_sv:.4f}")
    print("PASSED: Spectral penalty outputs are valid.")

def test_training_and_eval_loop(setup_dataset):
    """
    Run one epoch of training and evaluation to verify full cycle works end-to-end.
    """
    print("\n[TEST] Running training and evaluation loop...")
    loader = DataLoader(setup_dataset, batch_size=2)
    model = MultiHeadDotGAT(
        in_features=setup_dataset.input_dim,
        hidden_features=32,
        output_dim=setup_dataset.n * setup_dataset.m,
        num_heads=2,
        dropout=0.1,
        agg_mode='concat'
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    loss = train(model, loader, optimizer, theta=0.95, criterion=criterion, device=device)
    assert loss > 0
    val_known, val_unknown, _, _, _ = evaluate(model, loader, criterion, 8, 8)
    print(f"Train Loss: {loss:.4f} | Eval Known: {val_known:.4f} | Eval Unknown: {val_unknown:.4f}")
    assert val_known >= 0 and val_unknown >= 0
    print("PASSED: Training and evaluation cycle completed correctly.")

def test_agent_view_diversity_sparse(setup_dataset):
    """
    Ensure that agents receive non-identical views in sparse mode.
    """
    print("\n[TEST] Ensuring agent view diversity in sparse mode...")
    sample = setup_dataset[0]
    diversity = any((sample.x[i] != sample.x[j]).any().item() for i in range(4) for j in range(i+1, 4))
    assert diversity
    print("PASSED: Agent views are diverse in sparse mode.")

def test_agent_with_no_entries():
    """
    Force an agent to receive zero entries and ensure no crash occurs.
    """
    print("\n[TEST] Handling agents with zero entries...")
    dataset = AgentMatrixReconstructionDataset(
        num_graphs=1, n=8, m=8, r=2, num_agents=4, view_mode='sparse', density=0.01, sigma=0.0
    )
    sample = dataset[0].to(device)
    zero_agents = [(agent == 0).all().item() for agent in sample.x]
    assert any(zero_agents)

    model = MultiHeadDotGAT(
        in_features=dataset.input_dim,
        hidden_features=16, 
        output_dim=dataset.n * dataset.m,
        num_heads=2, dropout=0.0).to(device)
    try:
        model(sample.x)
        print("PASSED: Model handles zero-entry agents without crashing.")
    except Exception as e:
        pytest.fail(f"Model failed on agent with no entries: {e}")

def test_message_passing_effect(setup_dataset):
    """
    Verify that the model modifies inputs through message passing.
    """
    print("\n[TEST] Verifying message passing alters agent representations (is not a no-op)...")
    model = MultiHeadDotGAT(
        in_features=setup_dataset.input_dim,
        hidden_features=32, 
        output_dim=setup_dataset.n * setup_dataset.m,
        num_heads=2, dropout=0.0
    ).to(device)
    model.eval()
    sample = setup_dataset[0].to(device)
    out = model(sample.x)
    assert not torch.allclose(out, sample.x)
    print("PASSED: Message passing modifies inputs as expected.")

def test_spectral_penalty_reduces():
    """
    Verify that spectral penalty reduces over multiple training epochs.
    """
    print("\n[TEST] Validating spectral penalty reduces over training...")
    dataset = AgentMatrixReconstructionDataset(
        num_graphs=4, n=8, m=8, r=2, num_agents=4
    )
    loader = DataLoader(dataset, batch_size=2)
    first_sample = dataset[0]
    model = MultiHeadDotGAT(
        in_features=first_sample.x.shape[1],
        hidden_features=32,
        output_dim=first_sample.y.shape[0],  # inferred directly
        num_heads=2,
        dropout=0.1,
        agg_mode='concat'
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    norms = []
    for _ in range(5):
        train(model, loader, optimizer, theta=0.5, criterion=criterion, device=device)
        _, _, nuclear, _, _ = evaluate(model, loader, criterion, 8, 8)
        norms.append(nuclear)
    assert norms[-1] <= norms[0] or abs(norms[-1] - norms[0]) < 1e-3
    print("PASSED: Spectral penalty reduces over training.")

def test_wide_matrix_size():
    """
    Verify that the model handles wide matrices (e.g., 4x64) without crashing.
    """
    print("\n[TEST] Checking wide matrix size and agent robustness...")
    dataset = AgentMatrixReconstructionDataset(
        num_graphs=2, n=4, m=64, r=3, num_agents=4
    ).to(device)
    model = MultiHeadDotGAT(
        in_features=dataset.input_dim, 
        hidden_features=64, 
        output_dim=dataset.n * dataset.m, 
        num_heads=2, dropout=0.1
    ).to(device)
    out = model(dataset[0].x)
    assert out.shape == (dataset.num_agents, dataset.n * dataset.m)
    print("PASSED: Model handles wide matrices (e.g., 4x64) correctly.")

def test_agent_ensemble_aggregation_behavior():
    """
    Verify that the ensemble prediction is at least as good as the worst individual agent.
    """
    print("\n[TEST] Verifying aggregated prediction vs. individual agents...")
    dataset = AgentMatrixReconstructionDataset(
        num_graphs=1, n=8, m=8, r=2, num_agents=4
    )
    sample = dataset[0].to(device)
    model = MultiHeadDotGAT(
        in_features=dataset.input_dim,
        hidden_features=32, 
        output_dim=dataset.n * dataset.m, 
        num_heads=2, dropout=0.0
    ).to(device)
    model.eval()
    out = model(sample.x)
    ensemble = out.mean(dim=0)
    individual_losses = [F.mse_loss(out[i], sample.y).item() for i in range(out.size(0))]
    ensemble_loss = F.mse_loss(ensemble, sample.y).item()
    assert ensemble_loss <= max(individual_losses)
    print("PASSED: Ensemble prediction is at least as good as worst individual agent.")
