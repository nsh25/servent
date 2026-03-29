import numpy as np

from ldpc_is.awgn import sigma_from_ebn0_db
from ldpc_is.modulation import bpsk_map


def test_bpsk_map_basic():
    bits = np.array([0, 1, 0, 1], dtype=int)
    out = bpsk_map(bits)
    np.testing.assert_allclose(out, np.array([1.0, -1.0, 1.0, -1.0]))


def test_sigma_from_ebn0_db_rate1_0db():
    sigma = sigma_from_ebn0_db(0.0, rate=1.0)
    np.testing.assert_allclose(sigma, np.sqrt(0.5), rtol=1e-12)
