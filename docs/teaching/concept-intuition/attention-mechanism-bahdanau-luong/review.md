# Recurrent attention: whole-lesson concept-transition review

26 September 2026. Read the complete learner JSX, including every advanced branch, source/code explanation and all eight practices. Revision 4 already provides extensive local intuition. Preserve it where sufficient rather than inserting a second introductory template.

## Canonical ownership

The active manuscript is this directory's `lesson.md`; `scripts/generate-recurrent-attention-lesson.mjs` now renders it into `src/learn/data/topics/attention.jsx`. The revision-4 manuscript and its receipts remain unchanged historical evidence. The active copy retains every original code fence. Existing numerical programs, trained weights, measured results and live engines are unchanged; generation republishes the same numerical artifacts, not a fresh experiment.

| Concept / exact manuscript location | Finding and action | Representation and transfer evidence |
| --- | --- | --- |
| §1 one summary and saved memories | Sufficient spelling-prefix need, source-state shelf and final-state distinction; retained. | Recall contrast, contextualized two occurrences of a, bidirectional non-streaming boundary. |
| §2 query, key and value | Sufficient lookup analogy progresses to concrete vectors with distinct roles; retained. | Lookup bridge and three-position memory shelf. |
| §2 score → share → read | Sufficient exact scores, exponentials, weighted vectors and convex geometry; retained. | Softmax steps, worked read and directly playable read lab. |
| §3 additive, dot and general scores | Sufficient dimensions, parameter counts, nonlinear comparison and shape restrictions; retained. | Additive steps and explicit local matrix shapes. |
| §3 decoder timing and input feeding | Sufficient two executable timelines distinguish old/new states and fed context; retained. | Synchronized schedules and practice 3. |
| §3 question cancellation | Sufficient linear concatenation cancellation proof and null case; retained. | Cancellation lab and practice 2. |
| §4 tensor shapes and two normalizations | Sufficient B/S/T roles and attention versus vocabulary distribution distinction; retained. | Shape table and full scratch read. |
| §4 source, target, EOS and padding | Sufficient 3→2 denominator counterexample, backward-state contamination and target loss ownership; retained. | Padding-share diagram, fitted mask lab, practice 4. |
| §5 training the read | Sufficient output loss to signed value credit to score/query gradient derivation; retained. | Exact gradient figures and immediate one-update/null-case lab. |
| §6 data and measured architecture comparison | Sufficient grouped lemma split, target overlap, seed variability and confounding limits; retained. | Actual full run checkpoints and rule baseline that wins. |
| §6 complete implementation | Full training program, saved-model invocation and explained generation/data pipeline retained. | Downloadable code and measured source provenance; no imported-library-only substitution. |
| §7 alignment and ambiguity | Sufficient full worked generation, EOS nuance and identical-context/different-weight construction; retained. | Alignment inspection and ambiguity figure. |
| Implementation pass | Sufficient scratch read composed into normal trainable PyTorch modules; retained. | Temperature customization with expected exact contexts and unchanged masking contract. |
| §8 global, local, monotonic and Gaussian windows | Sufficient hard support versus soft multiplier distinction and unnormalized local-p sum; retained. | Window lab, explicit normalizing counterexample, practice 7. |
| §8 copying | Sufficient repeated source words accumulate before mixing with vocabulary route; retained. | Ada/met/Ada mass flow and practice 6. |
| §8 speech/location/LAS | Sufficient previous-row convolution and repeated-vowel disambiguation, pyramidal reduction and bidirectional non-streaming limits; retained. | Location-flow figure and practice 8. |
| §8 coverage | Gap: formula was named without showing accumulated mass or overlap. Added original three-position, three-read running tally and overlap calculation. | Semantic table shows total mass 1→2→3, overlap 0.7, zero first-step penalty, and why coverage differs from last-row location. |
| §8 caching, costs and scaled scores | Sufficient what remains fixed, S×T interactions, decoder/beam state and variance assumptions; retained. | Separate cache/per-read/output-history counts. |
| §9–10 practice and onward reading | Eight full solutions and annotated primary/alternative resources remain. | New coverage probe supports mechanism understanding without forcing a prediction or unlocking output. |

## Research actually inspected

- [See et al., Get To The Point](https://arxiv.org/pdf/1704.04368), §2.3, equations 10–12: accumulated attention, score feedback and bounded overlap loss. The new table is an original calculation; it does not claim a measured summarization result or factual coverage.
- Existing Distill, D2L, 3Blue1Brown, Stanford and original Bahdanau/Luong annotations remain. No new video viewing or native fitting is claimed during this increment.

## Representation and checks

A row-aligned table is the useful representation here: it makes three distinct objects—one read, running total and overlap—comparable in the same source columns. Existing topic-specific live labs already expose the changing read, scorer, gradient, masks and windows. No duplicate generic lab is needed for simple addition.

The scoped verifier computes all coverage rows and overlap, the first-step zero case, and mass accumulation; parses the JSX; compares all manuscript code fences with the preserved revision; and confirms deterministic generator agreement. Source hashes bind the manuscript, generator, output, this map and verifier. Independent learning review and browser/integration review remain pending separately.
