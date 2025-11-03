import numpy as np
from src.channel import Channel
from src.report import EmptyReporter


def test_channel_fading_applied_deterministic():
    """Check per-symbol uniform fading [0.5,0.9] is applied deterministically when RNG is seeded.
    We pass a very large Eb/N0 so noise is negligible, then compare the output to the
    input scaled by the expected fades (computed with a separate RNG seeded the same).
    """
    seed = 42
    rng_for_channel = np.random.default_rng(seed)
    rng_for_expected = np.random.default_rng(seed)

    num_sym = 8
    # Each symbol is a 2-d vector set to ones -> after fading each row should equal fade value
    sym = np.ones((num_sym, 2))

    ch = Channel(eb_n0_db=1000.0, with_fading=True, rng=rng_for_channel)
    reporter = EmptyReporter()
    out = ch.encode(sym, reporter)

    # Compute expected fades using a rng with the same seed
    expected_fades = rng_for_expected.uniform(0.5, 0.9, size=num_sym)
    expected = sym * expected_fades.reshape((num_sym, 1))

    assert out.shape == sym.shape
    assert np.allclose(out, expected, atol=1e-6)


def test_channel_awgn_variance_matches_theory():
    """Check AWGN variance approximately equals theoretical sigma^2 for given Eb/N0.
    Use many samples and a fixed RNG to get stable statistics.
    """
    seed = 1234
    rng = np.random.default_rng(seed)

    # Choose Eb/N0 in dB and compute theoretical sigma^2 = N0/2 = 1/(2 * 10^(Eb/N0_dB/10))
    eb_n0_db = 10.0
    ebn0_lin = 10 ** (eb_n0_db / 10.0)
    theoretical_var = 1.0 / (2.0 * ebn0_lin)

    num_samples = 20000
    # Use scalar symbols (dimension 1) set to zero so output variance is just noise variance
    sym = np.zeros((num_samples, 1))

    ch = Channel(eb_n0_db=eb_n0_db, with_fading=False, rng=rng)
    reporter = EmptyReporter()
    out = ch.encode(sym, reporter)

    # Flatten and compute empirical variance
    emp_var = np.var(out.flatten(), ddof=0)

    # Allow some tolerance due to finite-sample (~5%)
    rel_error = abs(emp_var - theoretical_var) / theoretical_var
    assert rel_error < 0.05, f"emp_var={emp_var:.4e}, theory={theoretical_var:.4e}, rel_err={rel_error:.3f}"
