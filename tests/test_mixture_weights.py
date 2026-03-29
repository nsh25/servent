import numpy as np

from ldpc_is.mixture_is import choose_mixture_weights, importance_weight


def test_choose_mixture_weights_sum_to_one():
    alphas = np.array([0.3, 1.0, 2.0])
    p = choose_mixture_weights(alphas, mode="distance_based", beta=1.2)
    np.testing.assert_allclose(p.sum(), 1.0)
    assert np.all(p > 0)


def test_importance_weight_unity_when_target_equals_proposal():
    y = np.array([[0.2], [-1.1], [3.3]])
    x0 = np.array([0.0])
    centers = np.array([[0.0]])
    probs = np.array([1.0])
    w = importance_weight(y, x0, centers, probs, sigma=1.0)
    np.testing.assert_allclose(w, np.ones_like(w), rtol=1e-12, atol=1e-12)
