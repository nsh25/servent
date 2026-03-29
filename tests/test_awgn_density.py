import numpy as np

from ldpc_is.mixture_is import log_gaussian_pdf_diag, log_proposal_density


def test_log_gaussian_scalar_known_value():
    y = np.array([[0.0]])
    mu = np.array([0.0])
    sigma = 1.0
    val = log_gaussian_pdf_diag(y, mu, sigma)[0]
    np.testing.assert_allclose(val, -0.5 * np.log(2.0 * np.pi), rtol=1e-12)


def test_mixture_density_symmetric_two_components():
    y = np.array([[0.0]])
    centers = np.array([[-1.0], [1.0]])
    probs = np.array([0.5, 0.5])
    sigma = 1.0
    qlog = log_proposal_density(y, centers, probs, sigma)[0]
    assert np.isfinite(qlog)
