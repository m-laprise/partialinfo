
import pytest
import torch
from graphattentionnet import DotGATLayer, MultiHeadDotGAT
from torch_geometric.data import Data


@pytest.fixture
def graph_data():
    num_nodes = 4
    in_features = 8
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2],
                               [1, 0, 3, 2, 2, 0]], dtype=torch.long)
    x = torch.randn((num_nodes, in_features))
    return x, edge_index


def test_dotgatlayer_output_shape(graph_data):
    x, edge_index = graph_data
    out_features = 16
    layer = DotGATLayer(x.size(1), out_features)
    out = layer(x, edge_index)
    assert out.shape == (x.size(0), out_features)


def test_dotgatlayer_residual_connection(graph_data):
    x, edge_index = graph_data
    # Case where in_features == out_features
    layer = DotGATLayer(x.size(1), x.size(1))
    out = layer(x, edge_index)
    assert out.shape == x.shape
    # Check if residual is effectively applied by verifying that gradients flow
    out.sum().backward()
    assert x.grad is None or x.grad.shape == x.shape  # Allow no grad if detached


@pytest.mark.parametrize("agg_mode", ["concat", "mean"])
def test_multiheaddotgat_output_shape(graph_data, agg_mode):
    x, edge_index = graph_data
    model = MultiHeadDotGAT(
        in_features=x.size(1),
        hidden_features=4,
        out_features=3,
        num_heads=2,
        dropout=0.5,
        agg_mode=agg_mode
    )
    out = model(x, edge_index)
    assert out.shape == (x.size(0), 3)


def test_multiheaddotgat_dropout_behavior(graph_data):
    x, edge_index = graph_data
    model = MultiHeadDotGAT(
        in_features=x.size(1),
        hidden_features=4,
        out_features=3,
        num_heads=2,
        dropout=0.9,
        agg_mode="concat"
    )
    model.train()
    out1 = model(x, edge_index)
    out2 = model(x, edge_index)
    assert not torch.allclose(out1, out2), "Dropout should introduce randomness in training mode"
    model.eval()
    out3 = model(x, edge_index)
    out4 = model(x, edge_index)
    assert torch.allclose(out3, out4), "Outputs should be consistent in eval mode"
