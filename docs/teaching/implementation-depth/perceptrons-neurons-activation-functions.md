# Perceptrons — executable implementation depth

Updated 22 September 2026. Scope: the user-requested audit of scratch and normal-library routes. This augments the reviewed lesson; it preserves the prepared packet and all eighteen recorded digit fits.

## Coverage and ownership

| Mechanism | Executable scratch route | Normal-library correspondence |
| --- | --- | --- |
| Perceptron mistake update, bias augmentation, zero-margin policy | Existing complete NumPy program in `perceptron-examples.js`, taught in §2 | New bridge runs `sklearn.linear_model.Perceptron` on the same ordered AND/XOR rows with fixed step, no penalty or shuffling, and twelve passes. It reproduces (3,2,−4) for AND and the zero-coefficient XOR cycle. The early-stop policy difference is explained. |
| Affine score and bias broadcasting | `X @ W.T + b` with explicit nonsymmetric weights and two examples | Values copied to `nn.Linear`; identical output-by-input layout and float64 inputs |
| Sigmoid, tanh, ReLU and leaky ReLU values/slopes | Reusable shape-preserving NumPy `activation(name, z)` | Corresponding `nn` modules and autograd on identical inputs, including declared zero conventions |
| ELU, exact/approximate GELU, SiLU and Mish | Explicit activation and derivative formulas; numerical primitives are NumPy operations and vectorized SciPy `erfc` | Explicit module parameters preserve ELU alpha, leaky slope and GELU approximation choice |
| Incoming weight versus local slope | The product `weight * slope` at a common operating point | Differentiation through `torch.sigmoid(weight * input)` at weights 0, .5 and 4 |
| SwiGLU projections and multiplication | Three explicit matrix products and the SiLU gate on nonzero data | Three copied `nn.Linear` layers, `nn.SiLU` and pointwise multiplication |
| Learning on actual handwriting | Existing complete six-activation, three-seed PyTorch experiment | Preserved actual CSV, splits, metrics and sample correspondence; no retraining in this audit |

The lesson owns neuron forward rules and local sensitivities. [Backpropagation](../../../src/learn/data/topics/backprop.jsx) owns the reusable autodiff engine, full classifier reverse pass and derivative checking; there is no second engine here. Loss selection and normalization retain their separate owners. Approximation-theorem discussion does not imply that this program proves training convergence.

## Learner route and quality

The added `#perceptron-code-route` appears next to the local-sensitivity investigation. It explains stable calculations, shapes, mapping, commands, actual output, matching conventions and a changed negative-slope exercise with solution. The complete program is both downloadable and readable in a keyboard-accessible disclosure; source is fetched only when opened. The core route starts with sigmoid/tanh/ReLU and labels §7 functions as a later branch.

The scratch implementation avoids unnecessary sigmoid evaluation in unrelated activation branches. Sigmoid uses `exp(-abs(z))`; its derivative avoids subtracting a value rounded to one. Softplus uses a stable form, ELU avoids evaluating a growing exponential on its positive branch, and exact GELU uses vectorized `erfc` to retain negative-tail mass. Some library tail derivatives round differently; checks use stated numerical tolerances. These are algorithmic/numerical choices, not a claim of globally optimal performance.

For K values, activation work and temporary storage are O(K). Dense N-by-D to H projection work is O(NDH) with O(NH) output storage, excluding already-owned inputs/weights. The functions preserve scalar/array shape and are assessed as float64 dense CPU operations on the stated finite operating range, including −40 to 40. Cubic/squared intermediates are not claimed safe at every finite float64 magnitude. Production device kernels, mixed precision, quantization and extreme-value approximation contracts remain outside this teaching implementation.

## Sources, execution and review

- [Program](../../../public/learn-assets/perceptrons/activation-mechanisms.py), [learner body](../../../src/learn/data/topics/perceptrons-neurons-activation-functions.jsx).
- [Native verifier](../../../scripts/verify-neuron-implementation-depth.py) executed the complete file in Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1 and PyTorch 2.14.0+cpu. Setup is shown; fresh-environment installation was not performed.
- [Native evidence](../evidence/neuron-implementation-depth.json) preserves exact stdout and source hashes. Activation coverage includes 1,601 scores per function, output/slopes, shape permutation, large sigmoid inputs, negative GELU tail and positive sigmoid derivative tail. Assertions stay outside learner code.
- Official [Perceptron API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) was reviewed for shuffle, step, stopping and binary decision contracts. Prior PyTorch activation references remain in the lesson. Function and derivative formulas were checked against independent installed library implementations, not browser JavaScript.
- [Affected browser checks](../../../scripts/verify-neuron-implementation-depth-browser.cjs) cover deferred source loading, exact code/output, downloads, keyboard disclosure/scrolling, retry and desktop/phone containment. Their result and final UI source hashes live in [the browser receipt](../evidence/neuron-implementation-depth-browser.json).

## Final closure — 22 September 2026

The author browser verifier passed all seven groups across the three affected topics against the final production build at `http://127.0.0.1:4194`, after the per-topic metadata split. This topic's complete displayed source/output and downloaded bytes agree with the executed files; closed views fetch no source, reopening reuses it, keyboard scrolling works, and the shared failure/retry case recovers. The six final desktop/320-pixel captures were visually inspected for readable gold controls, dark code panels, focus outlines and local horizontal scrolling. No material finding remains in this added route. The integration owner's bounded repair gives existing long Perceptron display equations local scrolling; final page containment passes.

The [browser receipt](../evidence/neuron-implementation-depth-browser.json) binds the final topic, shared UI, scoped CSS and [topic-only output metadata](../../../src/learn/data/perceptron-mechanism-program.js) by SHA-256. The [separate review receipt](../evidence/neuron-implementation-depth-independent.json) records the integration owner's complementary finite-difference slope checks. Native verification passed 48 groups, including execution of all three complete programs; the original training runs remain preserved. These additions are complete within the stated bounds. Broader integration and user acceptance remain distinct.
