# Normalization: implementation-depth revision

Canonical ID: `batch-layer-group-rms-normalization`.

The existing full program already constructs centered normalization, RMS, grouping and train/eval BatchNorm state from tensor primitives and compares library modules. Retain it. The new **Implement the derivative and connect it to the module** section adds manual NumPy vector-Jacobian products, explains affine parameter-sharing axes and links the existing forward owner rather than claiming it was missing.

New canonical program: `public/learn-assets/batch-layer-group-rms-normalization/normalization-backward.py`. It compares manual input derivatives with BN/LN/GN/IN/RMS modules, then affine LayerNorm gradients and independent finite differences. The generator renders the complete new prose and a lazy code disclosure. A downloadable execution record is provided. Existing plots, real-data fits and recorded outputs are untouched.

The mechanism is linear in the number of tensor elements; it computes VJPs directly without a quadratic dense Jacobian. It uses array primitives and supported finite float64 fixtures, not a fused mixed-precision/GPU kernel. Epsilon, variance conventions, parameter axes and training/evaluation state remain explicit. The page distinguishes the training derivative through computed statistics from the evaluation derivative through fixed stored statistics. BatchNorm buffer-update ownership remains in `normalization-experiments.py`.

The new practice manually updates affine parameters and compares with a library module, then adapts the operation to RMS. It gives a hint, reasoned solution and reuse link to the preceding autodiff lesson. Forward/state parity and the original CPU experiment remain valid evidence for unchanged sources; the new derivative code has its own native and independent review record plus targeted production checks.

Primary contracts consulted 21 September 2026: [BatchNorm2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html), [RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html). Finite-difference checks establish local numerical agreement, not correctness on every conceivable input or hardware platform.
