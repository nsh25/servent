import numpy as np

from ldpc_is.decoder_interface import DummyThresholdDecoder
from ldpc_is.estimators import estimate_error_rates_is


def test_estimators_finite_outputs():
    x0 = np.ones(8)
    centers = np.stack([x0, x0 - 0.2])
    probs = np.array([0.5, 0.5])
    dec = DummyThresholdDecoder(fail_threshold=-100.0)  # always converged/no FE
    rng = np.random.default_rng(123)
    out = estimate_error_rates_is(
        x0=x0,
        sigma=1.0,
        decoder=dec,
        centers=centers,
        mixture_probs=probs,
        num_samples=200,
        batch_size=64,
        rng=rng,
    )
    assert out["weights_finite"]
    assert out["FER_hat"] == 0.0
    assert out["BER_hat"] == 0.0
