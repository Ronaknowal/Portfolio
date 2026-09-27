# Backpropagation: concept-level intuition review

26 September 2026. Read all eight production sections, every advanced branch, code block and practice solution, and the existing figures/labs. Full-mode manuscript remains src/learn/data/topics/backprop.jsx. Independent and rendered review pending.

| Transition / location | Disposition and learner benefit |
| --- | --- |
| Opening: derivative versus update | Removed the control catalogue that introduced technical terms before the motivating question. The existing reverse-question introduction and first-pass route are clearer. |
| §1 local sensitivity, chain rule, finite update | Retain the complete scalar forward/backward trace, numerical table and live line fit. They already connect units, signs, products and step size. |
| §2 graph reuse, repeated operand, reverse scheduling | Retain two explicit accumulation causes, editable cancellation example and dependency-order explanation. Local primitive table supplies saved values as well as formulas. |
| §3 broadcasting | Existing three-row fan-out/fan-in diagram already shows why to sum rather than average; retained. |
| §3 matrix pullbacks | Added a two-output routing diagram and separate shared-weight table. The output-column sum for an input contrasts with the example-row sum for a weight. The transposes now connect to concrete effects, not only indices and shapes. |
| §3 softmax cross-entropy signal | Added probability allocation strip, p-minus-target table, descent directions and zero-sum/common-shift reasoning before the complete reverse pass. |
| §3 full network, reductions, measured training | Existing shape lanes and exact NumPy run/download already connect all operations; retained without a new training claim. |
| §4 finite differences, cancellation and corners | Existing perturbation/offset lab, paired values and corner examples suffice. Stable cross-entropy versus clipped objective distinction retained. |
| §5 scratch/library alignment | Complete engine bridge and explained API shape/reduction conventions retained. |
| §5 grad buffer, retain_graph, create_graph | Added three-object distinction with x² at 3: recipe, accumulated number, and differentiable derivative recipe. Existing freeze/no_grad/detach/eval table and frozen-input example retained. |
| §6 engine operations / graph traversal | Complete scratch engine, multiply and matrix closures remain. Earlier diagram grounds the matrix rule; no code duplication. |
| §6 JVP / VJP | Existing Jacobian, signed products and exact coordinate arithmetic already distinguish direction from output weighting. |
| §6 Hessian-vector product | Added motive (change of gradient advice) and actual small input move (.3,.7)→(.31,.72), gradients (1.3,4.5)→(1.34,4.63), scaled change (4,13). Exact quadratic versus local smooth approximation stated. |
| §6 custom primitive, hard sigmoid, surrogate derivative | Existing local-rule code, corner policy and true versus substituted derivative distinction retained. |
| §6 microbatch accumulation | Added equal-example coefficient diagram contrasting full mean, unweighted group means and corrected group-size weights. Makes unequal weighting visible, not just a mismatch in final gradients. |
| §6 checkpointing | Existing retained/recomputed chain and K+L/K model already explain cost tradeoff and limits; retained. |
| §7–8 practice, resources and next topic | All eight changed cases, solutions, references and progression retained. No prediction-entry requirement added. |

## Research and choices

Read Stanford's original CS231n Backpropagation Intuitions notes (https://cs231n.github.io/optimization-2/), emphasizing local circuit rules and vectorized gradients. Used the input-perturbation interpretation and explicit collection of paths; original examples here are different. Do not import the notes' informal max-tie rule as a unique mathematical derivative.

Consulted 3Blue1Brown's original text companion (https://www.3blue1brown.com/lessons/backpropagation/) for linking a desired output change to upstream changes. No video-watch claim. Consulted PyTorch's double-backward tutorial (https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html) for what recording the backward graph means. These supplement the existing AD survey and versioned API sources rather than replace the implementation contracts.

## Visual and source contract

Three static, topic-owned native figures encode different relationships: a branch and accumulation flow, shared-total probability strip, and five equal-width coefficient columns with a common scale. Values and their meaning remain in text/tables. No imitation controls; original live labs remain. CSS stacks branches on narrow screens. Browser checks must inspect wrapping, column labels and the smallest probability segment.

The old build-backprop-lesson importer consumes an earlier manuscript that predates the current scratch/library bridge. It now checks the ledger's canonical manuscript before any write and refuses to overwrite subsequent revisions. Do not regenerate from that historical packet. Existing numerical inputs, engines, downloadable programs and measured outcomes are preserved; this revision adds explanation and independently checked constructed examples.

## Evidence / follow-through

Author checks: scripts/verify-backprop-intuition.mjs covers parsed source, independently computed route sensitivities, cross-entropy finite differences, equal-example microbatch coefficients, Hessian small move and unchanged legacy numerical sources. Its receipt binds actual bytes. Independent learning/correctness review and desktop/narrow rendered checks remain pending.
