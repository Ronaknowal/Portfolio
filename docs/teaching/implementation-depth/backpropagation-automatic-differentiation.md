# Backpropagation — executable implementation depth

Updated 22 September 2026. Scope: the user-requested audit of scratch and normal-library routes. Existing XOR/digit fits, native outputs and the original engine remain unchanged.

## Coverage and ownership

| Mechanism | Existing executable implementation | Added correspondence |
| --- | --- | --- |
| Reverse topological traversal, repeated operands and reused results | Complete NumPy `Tensor` engine in `teaching-autodiff.py`: reachable-node traversal, saved forward values and additive pullbacks | The same engine is imported by the new bridge, never reimplemented |
| Broadcast reduction, two-matrix matmul, tanh/ReLU, sum/mean, exp/log | Explicit NumPy primitives and pullbacks in the same file | The bridge matches a three-row 2→3→2 network against `nn.Linear` / `nn.Tanh`; prior primitive checks are retained |
| Stable mean cross-entropy and full classifier gradients | Engine loss rule and complete training loop, taught with the full matrix reverse pass | Same initial parameters, data, labels, float64 and reduction in `F.cross_entropy`; all four parameter gradients are compared after transposing library weights |
| Optimizer versus backward pass | Explicit simultaneous parameter update from old gradients | `torch.optim.SGD` with matching rate, zero momentum/decay; .2 and zero-rate cases |
| Framework behavior and deeper AD | Existing executed accumulation, finite differences, JVP/VJP, Hessian-vector, custom-operation, microbatch and checkpoint files | Kept in their existing sections; no second library or duplicate implementation added |
| Real-data training | Existing engine XOR and digit programs and actual observations | Preserved; a different earlier activation experiment is not presented as engine parity |

This lesson is the prerequisite owner of the general reverse-mode engine. Perceptrons owns activation forward/local-slope rules; Loss Functions owns objective choice; Transfer Learning owns which parameters change and its factor-specific paths. Later lessons should reuse this engine/shape explanation instead of copying it.

## Learner route and quality

The added `#backprop-code-route` begins §5. The runnable bridge needs the supplied `teaching-autodiff.py` beside it; both downloads and exact instructions are present. It imports only definitions, so it does not run the earlier long training experiments. It explains why comparing weights/gradients requires transposes, why accumulation must be cleared, and why a mean/sum mismatch changes a three-row objective and gradients by three. A changed-reduction diagnostic has an independently openable solution.

The existing engine performs real matrix products and sums broadcast uses correctly. Graph traversal work is O(V+E), with O(V) traversal bookkeeping beyond the graph's stored edges and tensor arrays. For the dense N-by-D, H-hidden, C-output classifier, numerical work is O(NDH+NHC); inputs, parameters and principal batch intermediates occupy O(ND+DH+HC+NH+NC) values. This is a conventional efficient algorithm for the supported dense computation, not a claim about the fastest possible autodiff runtime.

The engine's two-dimensional matmul, recursive small-graph traversal, immutable float64 forward values and fresh gradients per backward call are explicit contracts. Device execution, arbitrary batched matmul, sparse/complex derivatives, mutation tracking and higher-order differentiation are not implemented by it; framework examples cover the separately named higher-order capabilities. Those limits are visible to the learner. No intentionally incorrect or artificially slow engine is introduced for contrast.

## Execution and review

- [Bridge](../../../public/learn-assets/backpropagation/engine-library-bridge.py), [reused engine](../../../public/learn-assets/backpropagation/teaching-autodiff.py), [learner body](../../../src/learn/data/topics/backprop.jsx).
- [Native evidence](../evidence/neuron-implementation-depth.json) records exact program stdout and both source hashes. It was executed with the retained Python 3.12.14 / NumPy 2.3.5 / PyTorch 2.14.0+cpu environment.
- The .2 step changes both losses from 0.840015027835 to 0.774784850062; zero rate preserves 0.840015027835. Every gradient array differs by less than 7e−17 in the printed fixture. The original engine's broader primitive/finite-difference evidence remains applicable to its unchanged source.
- [PyTorch SGD API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html) was checked for update conventions. No momentum equivalence or optimizer ranking is claimed.
- [Affected browser receipt](../evidence/neuron-implementation-depth-browser.json) records exact displayed source/output, deferred fetch, byte-matched download, keyboard access and desktop/phone containment, separately from numerical evidence.

## Final closure — 22 September 2026

The author browser verifier passed all seven groups across the three affected topics against the final production build at `http://127.0.0.1:4194`, after the per-topic metadata split. This topic's complete displayed source/output and downloaded bytes agree with the executed bridge; closed views fetch no source, reopening reuses it, and keyboard scrolling works. The shared failure/retry case also passes. All six final desktop/320-pixel captures were visually inspected: the download and disclosure remain readable, code uses the intended dark theme and mono font, focus is visible, and long lines stay in their local scroller. No material finding remains in this added route.

The [browser receipt](../evidence/neuron-implementation-depth-browser.json) binds the final body, shared UI and [topic-only output metadata](../../../src/learn/data/backprop-mechanism-program.js) by SHA-256. Native verification passed 48 groups across the three programs, including both complete bridge runs. The existing engine and prior fits are unchanged; their earlier broader evidence was reused. The integration owner separately reviewed the bridge's orientation/update mapping. These additions are complete within the stated bounds. Broader integration and user acceptance remain distinct.
