
"""
Test harness for SensingMasks.

This script:
  * Generates dummy TemporalData.
  * Constructs SensingMasks instances for different N, G, rho, gamma.
  * Checks that empirical endowments match the intended μ(ρ) behavior:
        μ(ρ) = (1 - ρ) * G / N + ρ * G
    where G is the number of globally known entries, N the number of agents.

In particular, we verify (among other things) that:
  * For approximately constant G, increasing N decreases the mean entries per agent.
  * For fixed N and G, increasing ρ increases the mean entries per agent toward G.
  * For fixed N, G, and ρ, increasing γ increases inequality (variance of endowments).
"""

import numpy as np
import torch

from datautils.sensing import SensingMasks

# ---------- Utilities ---------- #

def reset_seed(seed: int = 12345):
    """Reset both NumPy and Torch random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


class DummyTemporalData:
    """
    Minimal dummy dataset that mimics TemporalData interface used by SensingMasks:
      * attributes .t, .m
      * __len__ implemented

    We use t=1, m=G, density=1.0 so that:
      * total_entries = m = G
      * global_known_count = density * total_entries = G
      * hence the size of the global known set is exactly G.
    """
    def __init__(self, t: int, m: int):
        self.t = t
        self.m = m

    def __len__(self):
        return 1


def build_sensing_masks(
    G: int,
    N: int,
    rho: float,
    gamma: float = 0.0,
    rank: int = 1,
    density: float = 1.0,
    hide_future: bool = False,
    verbose: bool = False,
):
    """
    Helper to instantiate SensingMasks with:
      * t = 1, m = G
      * density set so that the global known set size is exactly G
        (because total_entries = t * m = G, density=1.0 => G entries)
    """
    assert density == 1.0, "This helper assumes density=1.0 so that G = m."
    t = 1
    m = G
    dataset = DummyTemporalData(t=t, m=m)

    masks = SensingMasks(
        TemporalData=dataset,
        rank=rank,
        num_agents=N,
        density=density,
        hide_future=hide_future,
        rho=rho,
        gamma=gamma,
        ndim=1,
        verbose=verbose,
    )
    return masks


def get_endowments(masks: SensingMasks) -> np.ndarray:
    """
    Return per-agent endowments as a NumPy array.
    """
    endowments = masks.stats["agent_endowments"]
    if isinstance(endowments, torch.Tensor):
        endowments = endowments.cpu().numpy()
    else:
        endowments = np.asarray(endowments)
    return endowments


def theoretical_mu(G: int, N: int, rho: float) -> float:
    """
    μ(ρ) = (1 - ρ) * G / N + ρ * G.
    """
    return (1.0 - rho) * (G / float(N)) + rho * float(G)


# ---------- Tests ---------- #

def test_mean_endowment_matches_mu_for_various_rho_and_N():
    """
    For fixed G and varying N and ρ, check that the empirical mean matches
    the theoretical μ(ρ) within a relative tolerance.

    This directly tests the core mapping:
        μ(ρ) = (1 - ρ) * G / N + ρ * G
    implemented in _agent_samplesizes.
    """
    reset_seed(0)
    G = 1000  # number of globally known entries
    rho_values = [0.0, 0.25, 0.5, 0.9]
    N_values = [5, 10, 20, 40]

    rtol = 0.05  # 5% relative tolerance

    for rho in rho_values:
        for N in N_values:
            masks = build_sensing_masks(
                G=G,
                N=N,
                rho=rho,
                gamma=0.0,      # near-equal endowments
                density=1.0,
                hide_future=False,
                verbose=False,
            )
            endowments = get_endowments(masks)
            empirical_mean = float(endowments.mean())
            mu_th = theoretical_mu(G, N, rho)

            # Relative error check
            rel_err = abs(empirical_mean - mu_th) / mu_th
            assert rel_err < rtol, (
                f"Mean endowment mismatch for G={G}, N={N}, rho={rho}: "
                f"empirical={empirical_mean:.2f}, theoretical={mu_th:.2f}, "
                f"rel_err={rel_err:.3f}"
            )


def test_avg_entries_per_agent_decreases_with_N_for_fixed_G_and_rho():
    """
    As G remains constant and N increases, the average entries per agent
    should decrease appropriately (roughly like 1/N when rho=0, and
    approach rho * G as N grows when rho>0).

    Here we explicitly check monotonic decrease for rho=0.
    """
    reset_seed(1)
    G = 1000
    rho = 0.0
    N_values = [5, 10, 20, 40]  # widely spaced to make the effect large

    means = []
    for N in N_values:
        masks = build_sensing_masks(
            G=G,
            N=N,
            rho=rho,
            gamma=0.0,
            density=1.0,
            hide_future=False,
            verbose=False,
        )
        endowments = get_endowments(masks)
        empirical_mean = float(endowments.mean())
        means.append(empirical_mean)

    # We only enforce a coarse monotonic decrease to avoid flakiness.
    # Given how large the differences are (200 → 25 for G=1000, N=5→40),
    # this should be robust.
    for i in range(len(N_values) - 1):
        assert means[i] > means[i + 1], (
            f"Expected mean entries per agent to decrease as N increases. "
            f"N={N_values[i]} → mean={means[i]:.2f}, "
            f"N={N_values[i+1]} → mean={means[i+1]:.2f}"
        )


def test_mean_endowment_increases_with_G_for_fixed_N_and_rho():
    """
    For fixed N and rho, increasing G (number of globally known entries)
    should increase μ(ρ) and hence the mean per-agent endowment.

    Since we set t=1, m=G, density=1, we control G directly via m.
    """
    reset_seed(2)
    rho = 0.5
    N = 10
    G_values = [500, 1000, 2000]

    means = []
    for G in G_values:
        masks = build_sensing_masks(
            G=G,
            N=N,
            rho=rho,
            gamma=0.0,
            density=1.0,
            hide_future=False,
            verbose=False,
        )
        endowments = get_endowments(masks)
        empirical_mean = float(endowments.mean())
        means.append(empirical_mean)

    for i in range(len(G_values) - 1):
        assert means[i] < means[i + 1], (
            f"Expected mean entries per agent to increase as G increases. "
            f"G={G_values[i]} → mean={means[i]:.2f}, "
            f"G={G_values[i+1]} → mean={means[i+1]:.2f}"
        )


def test_mean_endowment_monotone_in_rho_for_fixed_N_and_G():
    """
    For fixed N and G, μ(ρ) is increasing in ρ, from G/N to G.
    We check that the empirical mean endowment increases with ρ.
    """
    reset_seed(3)
    G = 1000
    N = 20
    rho_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    means = []
    for rho in rho_values:
        masks = build_sensing_masks(
            G=G,
            N=N,
            rho=rho,
            gamma=0.0,
            density=1.0,
            hide_future=False,
            verbose=False,
        )
        endowments = get_endowments(masks)
        empirical_mean = float(endowments.mean())
        means.append(empirical_mean)

    for i in range(len(rho_values) - 1):
        assert means[i] < means[i + 1], (
            f"Expected mean entries per agent to increase with rho. "
            f"rho={rho_values[i]} → mean={means[i]:.2f}, "
            f"rho={rho_values[i+1]} → mean={means[i+1]:.2f}"
        )


def test_gamma_controls_endowment_variance():
    """
    For fixed N, G, and rho, increasing gamma should increase inequality
    among agents (higher variance of s_i).

    We compare variance at gamma=0.0 (nearly-equal) and gamma=1.0 (heavy-tailed).
    """
    reset_seed(4)
    G = 1000
    N = 50
    rho = 0.5

    # Low-inequality regime
    masks_low = build_sensing_masks(
        G=G,
        N=N,
        rho=rho,
        gamma=0.0,
        density=1.0,
        hide_future=False,
        verbose=False,
    )
    endowments_low = get_endowments(masks_low)
    var_low = float(endowments_low.var())

    # High-inequality regime
    masks_high = build_sensing_masks(
        G=G,
        N=N,
        rho=rho,
        gamma=1.0,
        density=1.0,
        hide_future=False,
        verbose=False,
    )
    endowments_high = get_endowments(masks_high)
    var_high = float(endowments_high.var())

    assert var_high > var_low, (
        f"Expected higher endowment variance at gamma=1.0 than at gamma=0.0. "
        f"var_low={var_low:.2f}, var_high={var_high:.2f}"
    )


def test_fraction_clipped_grows_near_rho_one():
    """
    When rho is close to 1, mu approaches G, and the raw Dirichlet samples
    will often request more than G entries for some agents, leading to clipping.

    At rho=0, the total_target is G, so oversampling should be rare or zero.
    At rho=1, the total_target is N*G, and oversampling should be common.
    """
    reset_seed(5)
    G = 500
    N = 30

    # rho = 0
    masks_rho0 = build_sensing_masks(
        G=G,
        N=N,
        rho=0.0,
        gamma=0.5,
        density=1.0,
        hide_future=False,
        verbose=False,
    )
    frac_clipped_rho0 = masks_rho0.stats["fraction_clipped"]

    # rho = 1
    masks_rho1 = build_sensing_masks(
        G=G,
        N=N,
        rho=1.0,
        gamma=0.5,
        density=1.0,
        hide_future=False,
        verbose=False,
    )
    frac_clipped_rho1 = masks_rho1.stats["fraction_clipped"]

    assert frac_clipped_rho1 > frac_clipped_rho0, (
        f"Expected higher fraction_clipped at rho=1.0 than at rho=0.0. "
        f"frac_clipped_rho0={frac_clipped_rho0:.3f}, "
        f"frac_clipped_rho1={frac_clipped_rho1:.3f}"
    )


# ---------- Simple runner ---------- #

if __name__ == "__main__":
    # Run all tests sequentially if the script is executed directly.
    test_mean_endowment_matches_mu_for_various_rho_and_N()
    test_avg_entries_per_agent_decreases_with_N_for_fixed_G_and_rho()
    test_mean_endowment_increases_with_G_for_fixed_N_and_rho()
    test_mean_endowment_monotone_in_rho_for_fixed_N_and_G()
    test_gamma_controls_endowment_variance()
    test_fraction_clipped_grows_near_rho_one()
    print("All SensingMasks tests passed.")