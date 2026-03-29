import numpy as np

from ldpc_is.boundary_search import find_failure_boundary
from ldpc_is.decoder_interface import DummyThresholdDecoder


def test_boundary_search_finds_threshold_region():
    n = 4
    x0 = np.ones(n)
    d = np.array([1.0, 0.0, 0.0, 0.0])
    d = d / np.linalg.norm(d)
    sigma = 1.0
    dec = DummyThresholdDecoder(fail_threshold=1.0)
    res = find_failure_boundary(d, x0, sigma, dec, low=0.0, high=1.0, tol=1e-3, max_iter=30)
    assert res["found_failure"]
    assert 0.45 <= res["alpha_star"] <= 0.55
