# Independent review: backpropagation and automatic differentiation

Reviewer: classical_representation_intuition, 26 September 2026; not the author. Read the complete `backprop.jsx`, all eight practice solutions, all added figures/CSS, source ownership guard and author's map/checks. Independent source and learning review; no learner trial or browser certification.

## Correctness and coverage

No blocking source finding. The new matrix figure correctly distinguishes output-column collection for an input from example-row collection for a shared parameter. Its arrows explicitly carry sensitivities. Cross-entropy signals refer to logits, not direct parameter updates. Forward recipe, accumulated buffer and derivative recipe remain different objects. The Hessian table is exact for its quadratic and explicitly local for general functions. Unequal microbatch coefficients expose both changed relative weighting and changed total scale. Full scalar graph, repeated operands, broadcast, finite differences, code bridge, JVP/VJP, custom derivatives and checkpointing remain intact.

Complementary checks use a new reused branch that cancels exactly, a changed nonuniform microbatch partition with a per-example linear objective, and a Hessian-direction symmetry identity. This checks graph/weighting interpretations without rerunning digit fits or duplicating the author’s full engine suite.

## Learning-experience checklist

1. Route: §1–5 constitute a complete path; §6 labels the engine and higher-order return visit.
2. Cautions: derivative correctness versus optimizer step, nonsmooth checks and mixed microbatch state each attach to the mechanism they limit. No caution-only opening remains.
3. Real question: adjusting erroneous predictions returns in both exact scalar loss reduction and real digit/XOR experiments; those tasks are explicitly distinct.
4. Labs: existing live step-size, shared-branch and finite-difference investigations expose parameters beyond presets and null/large-step cases. Browser replay remains root's responsibility.
5. Figures: local routing, shared probability and equal-example coefficient columns encode different relationships; exact sums accompany visuals. CSS includes narrow branch stacking. Actual visibility remains browser-pending.
6. Connections: backpropagate the same forward equations, then bridge scratch to transposed library weights. JVP/VJP duality and Hessian gradient movement are explicitly reconciled.
7. Code: complete bounded teaching engine and normal PyTorch update route remain linked and explained; excerpts label their owner rather than masquerading as standalone programs.
8. Practice: eight changed graphs, broadcasts, step sizes, corner checks, directions and a new primitive include reasoned outcomes; no lab output is gated by practice.
9. Screenshots: not inspected by this reviewer; do not infer rendering from source parsing.
10. Buildup throughout: each middle/advanced transition explains its question, needed forward values, operation and result; newly concrete matrix contractions and derivative recipes address genuine former jumps.

Result: independent source acceptance, with browser/integration pending separately. Unchanged native programs and measured results retain their existing evidence.
