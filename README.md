# LDPC Trapping-Set Importance Sampling (FER/BER Error Floor)

This package implements trapping-set-based importance sampling (IS) for binary LDPC codes over BPSK/AWGN.

## Mathematical idea
For received vector `y \in R^n`:
- Target channel density: `p(y) = N(y; x0, sigma^2 I)` where `x0 = +1^n` (all-zero codeword mapped to BPSK).
- Proposal mixture: `q(y) = sum_m pi_m N(y; mu_m, sigma^2 I)`.
- IS weight: `w(y)=p(y)/q(y)`.
- Estimators:
  - `FER_hat = (1/N) sum I_f(y_i) w(y_i)`
  - `BER_hat = (1/(nN)) sum b(y_i) w(y_i)`

Boundary search finds `alpha_m` where decoding of `y(alpha)=x0-alpha d_m` first fails for trapping-set direction `d_m`. Then mixture centers are `mu_m = x0 - c * alpha_m d_m`.

## Features
- Generic decoder interface (`LDPCDecoderInterface` protocol).
- Dummy decoder for runnable demos/tests.
- `.trap`, JSON, and CSV trapping-set loading.
- Stable log-density computations with `scipy.special.logsumexp`.
- IS and MC estimators with CIs/ESS/weight diagnostics.
- Plot helpers and CLI demo.

## Trapping-set input formats
### Preferred `.trap` format (supported directly)
One set per line:

```text
(a, b) v1 v2 ... va
```

- `(a,b)` = TS label.
- `v_i` are **1-based** variable-node indices.
- Parser converts to 0-based internally.
- Validation checks line syntax, index count, and index bounds.

Example:
```text
(3, 3) 1 261 354
(5, 3) 1 202 261 316 354
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo run
```bash
python -m ldpc_is.run_is_demo \
  --n 16 --k 8 \
  --ts-path examples/example.trap \
  --ebn0-db 2.5 \
  --num-is-samples 4000 \
  --num-mc-frames 2000 \
  --top-m 3 \
  --seed 42 \
  --output-dir outputs/demo
```

## Plug in your real LDPC decoder
Replace the demo decoder in `ldpc_is/run_is_demo.py`:

```python
from ldpc_is.decoder_interface import ExternalDecoderAdapter

decoder = ExternalDecoderAdapter(your_decoder_callable)
```

Your callable should accept `llr: np.ndarray` and return either:
1. `DecoderResult`, or
2. a dict containing `hard_bits`, `num_bit_errors`, `frame_error`, `converged`, optional metadata, or
3. hard-bit `np.ndarray` (adapter infers error counts).

## Common failure modes
- No bracketing failure in boundary search: SNR too high, weak TS list, or decoder robustness.
- ESS collapse / huge weight CV: centers too aggressive, too few components, or poor mixture probabilities.
- Dominance by one component: extend TS list or tune `center_scaling` and distance-based mixture beta.

## Tests
```bash
pytest -q
```
