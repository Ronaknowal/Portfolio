# Transfer Learning — executable implementation depth

Updated 22 September 2026. Scope: the user-requested audit of scratch and normal-library routes. Original source fits, eighteen target fits, validation selection and the one selected test output are preserved.

## Coverage and ownership

| Mechanism | Executable route and ownership | Audit result |
| --- | --- | --- |
| Backbone/head reuse; probe, full and discriminative updates | Existing complete `transfer-experiments.py` defines models, copies initial states, selects trainable parameters and builds explicit optimizer groups | Already substantive normal-library code; preserved |
| LoRA factor forward pass and merge | Existing `LowRankLinear` defines both thin projections and merge with ordinary PyTorch operations | New NumPy routine independently implements the forward function, mean-MSE derivatives for A/B/input and a simultaneous SGD update |
| Frozen parameters versus differentiable input | Existing framework fixture and live lab distinguish gradient flags, optimizer membership and mode | New bridge checks base `.grad is None` and exact unchanged base while matching the manual input derivative |
| Zero-update initialization | Existing hand calculation and library fixture | New executed controls distinguish zero B, both-zero factors, zero bottleneck measurement and zero learning rate; changed two-row/nonzero-B fixture exercises A's gradient |
| Bottleneck adapter | Existing readable `BottleneckAdapter` explicitly computes `h + up(tanh(down(h)))`, initializes the up path to zero and trains within the full experiment | Reuses Backpropagation's general matrix/tanh pullbacks; another autodiff engine or copy of those derivatives would add duplication |
| Data and selection protocol | Existing complete real-digit split/training/validation/selection/checkpoint program | Preserved. The mechanism bridge does not claim new generalization results |
| Larger-model PEFT branches | Existing explanatory LoRA+/DoRA/QLoRA/ULMFiT scope with primary sources | No claim that the local two-factor routine implements quantization, distributed state, or those complete systems |

The added mechanism program is standalone: no CSV, hidden model or experiment module import. NumPy computes the rule; normal PyTorch `nn.Linear`, `nn.Parameter`, `F.linear`, mean MSE and SGD provide the counterpart. This is a readable implementation of LoRA using standard tensor/library primitives, not a claim to reproduce a third-party PEFT package's configuration and checkpoint format.

## Learner route and implementation quality

`#transfer-code-route` follows the first factor-update calculation in §4. It states matrix shapes and mean reduction before the runnable source and printed outputs, connects to the deeper derivative equations in §6, and explains each control. A batch-duplication exercise tests the difference between parameter and input derivatives, with a separate solution.

The NumPy routine uses dense matrix products and computes thin factor paths directly; it does not construct a dense BA during training. It rejects accidental target broadcasting and incompatible/empty matrix shapes with one short contract check. For X:(N,k), W:(d,k), A:(r,k), B:(d,r), the factor arithmetic is O(Nkr+Nrd), in addition to O(Nkd) for the base path. Bottleneck/output/input-gradient storage is O(Nr+Nd+Nk); factors and their gradients use O(r(k+d)) storage. The frozen O(kd) base remains in memory. Compatible inference merging deliberately constructs a dense d-by-k matrix once. These are explicit costs, not an unmeasured speed or universal optimality claim.

Supported inputs are dense real finite NumPy matrices and a scalar scale under the stated shapes; the checked route uses float64. Matrix multiplication and loss can still overflow for sufficiently large values. No mixed-precision, quantized, stochastic, distributed or mutation-tracking behavior is implied. The code exposes the mechanism while keeping verification machinery outside the learner program.

## Evidence and review

- [New program](../../../public/learn-assets/transfer-learning/lora-mechanism-bridge.py), [existing full experiment](../../../src/learn/assets/transfer-learning/transfer-experiments.py), [learner body](../../../src/learn/data/topics/transfer-learning-fine-tuning-strategies.jsx).
- [Native verifier](../../../scripts/verify-neuron-implementation-depth.py) actually executes the program and records [source hashes/stdout](../evidence/neuron-implementation-depth.json), using Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu.
- A changed 4-row, 3-input, 5-output, rank-2 fixture matches the framework output, loss and A/B/input gradients. Every element of all three derivative arrays is independently checked by central differences. Batch duplication, reciprocal factor rescaling and accidental-target-shape cases probe relationships beyond the original hand example.
- In the printed single-row fixture loss is 2.5→2.025; both-zero/null-measurement/zero-rate controls preserve their respective losses. The two-row nonzero-B fixture gives 2.375→1.517758789. All base-parameter changes are exactly zero; merging agrees within floating-point tolerance.
- [Browser evidence](../evidence/neuron-implementation-depth-browser.json) separately covers readable deferred source, recorded output, downloads, keyboard access and desktop/phone containment.

## Final closure — 22 September 2026

The author browser verifier passed all seven groups across the three affected topics against the final production build at `http://127.0.0.1:4194`, after the per-topic metadata split. This topic's complete displayed source/output and downloaded bytes agree with the executed program; closed views fetch no source, reopening reuses it, and keyboard scrolling works. The shared failure/retry case also passes. All six final desktop/320-pixel captures were visually inspected for readable controls, dark code panels, visible focus and locally scrolling code. No material finding remains in this added route.

The [browser receipt](../evidence/neuron-implementation-depth-browser.json) binds the final body, shared UI and [topic-only output metadata](../../../src/learn/data/transfer-learning-mechanism-program.js) by SHA-256. The [separate review receipt](../evidence/neuron-implementation-depth-independent.json) records the integration owner's scalar-coordinate LoRA oracle and factor-rescaling checks. Native verification passed 48 groups, including all five executed LoRA controls. Original source/target fits and selected test evidence are unchanged. These additions are complete within the stated bounds. Broader integration and user acceptance remain distinct.
