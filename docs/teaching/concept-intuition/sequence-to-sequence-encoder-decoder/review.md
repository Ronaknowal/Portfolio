# Sequence-to-sequence: concept-level author review

Full production lesson read, including all optional §9 branches, complete scratch protocol/native bridge, eight worked exercises and resource annotations. Canonical manuscript and generator updated together; model, dataset, learned outcomes and existing four interactive investigations retained.

| Location / transition | Disposition and support |
| --- | --- |
| Opening / §1 conditional task and two timelines | Retained: actual inflection example, request-token necessity and source/target BOS/EOS tracks. |
| §2 embeddings, support, source packing / target mask / shift | Retained explicit address interpretation, shape table, ate alignment and live correction investigation. |
| §3 state interface, context, shared/different weights | Retained scalar forward arithmetic, context intervention and live bridge; no new generic diagram needed. |
| §4 autoregressive likelihood / EOS / denominator | Retained product/log bridge; fixed token-vs-sequence weighting with two- vs six-token arithmetic and per-token weights. |
| §4 teacher forcing and joint encoder credit | Retained causal prefix interpretation, exact scalar derivative disclosure and no-gradient detachment distinction. |
| §5 greedy/beam, candidate ownership, stopping/caps | Retained complete probability tree, live search, deterministic tie convention and actual model generation. |
| §5 length score | Retained worked negative-score reversal and precise EOS length convention. |
| §6 data grouping, simple rules, outcome types and interpretation | Retained all measured results, actual failure and label/split caveats; no new training. |
| §7 full code and inference route | Retained complete standalone source/inference, bounded browser model and execution contract. |
| §8 task / fit / generalization / search diagnoses | Retained four distinct questions and context/prefix live interventions. |
| Scratch/library route and changed-code task | Retained cell owner reuse, explicit batching, ended state and beam-parent gather requirements. |
| §9 context shapes / repeated summary vs attention | Retained complete state handoff and learned projection examples. |
| §9 source reversal | Fixed historical statement with explicit aligned A/B/C→X/Y/Z path table: 3,3,3 becomes 1,3,5, explaining who benefits and who does not. |
| §9 scheduled sampling | Fixed abstract inconsistency claim with original two-bit joint-mass visual: dependent diagonal distribution vs independent replaced-prefix distribution. |
| §9 sequence objective vs label smoothing | Fixed name-only transition with exact two-answer expected reward and derivative; large-space estimation limits stated. |
| §9 search bounds / deployment / task metric | Retained contracts and metric caveats; fixed score-bound distinction with concrete normalized-score counterexample. |
| §10–11 practices / next lesson / alternatives | Retained all tasks, primary sources, lecture/notes alternative and progression to attention. |

## Research

Read [Huszár, §4–4.1](https://arxiv.org/pdf/1511.05101), the original-vs-replaced-prefix objective and two-symbol conditional/marginal derivation. The new fair-bit graphic is independently enumerated and deliberately restricted to the fully replaced limit. The lesson's existing original scheduled-sampling source and annotation remain. No video viewed during this review. Existing source-reversal and GNMT citations retained; their added path/score counterexamples are constructed arithmetic, not new measurements or paper reproductions.

## Visual / actual checks

`ScheduledPrefixFigure` uses two labelled probability tables, fixed corresponding cells, non-color labels and natural-height panels stacking below 650px. No new interactive controls: the existing timeline, alignment, joint-gradient, search and fitted-model interfaces already serve the core mechanisms.

`node scripts/verify-structure-learning-intuition.mjs sequence-to-sequence-encoder-decoder` checks five finite arithmetic groups, parses all changed JSX and binds four original runtime/result identities. The first regeneration caught an existing CRLF-vs-LF comparison in the generator; comparison now normalizes line endings without changing source semantics, and identical packaged JSON/source bytes are preserved. Generator then completed successfully. No training or browser execution is implied. Independent review, actual viewport checks and shared production build remain pending with root.
