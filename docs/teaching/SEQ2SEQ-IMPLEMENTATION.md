# Sequence-to-sequence implementation and review handoff

22 September 2026. Stable ID `sequence-to-sequence-encoder-decoder`, Deep Learning position 15. The complete prepared revision-3 manuscript/specifications are the content source. Those frozen inputs and their phase checkpoints remain unchanged. Root owns final shared integration and ledger closure.

## Teaching and implementation

The body preserves twelve major sections: the input/output task, shifted training tracks, state bridge, likelihood and gradient, greedy/beam/length scoring, real inflection experiment, complete runnable programs, diagnosis, scratch/library ownership, deeper branches, eight independent exercises, and annotated references including the official Stanford lecture/notes. The changed-code batch/beam exercise remains with its hint and solution. Negative measured results and the stronger rule baseline are retained.

The 11.5 KB complete training program is bound byte-for-byte to its canonical download and displayed on demand; the smaller standalone fixed-weight inference example remains inline and is also downloadable. The prepared RNN owner link now points to the actual published/downloadable recurrent implementation being completed earlier in this same module increment. There is no eager 5 MB experiment report or weights import: inline plots use a 3.2 KB derived measurement file, and the 762 KB seed-one JSON loads only when the fitted investigation opens.

Four different live investigations serve separate learning hurdles:

- **Target tracks:** edit the actual reference string, inspect previous-input/current-target positions, add ignored storage and contrast an explicitly incorrect unshifted mode. Editing the target generates all cards together instead of asking learners to solve a drag arrangement before seeing output. The small mask calculator uses explicitly constructed equal correct-token probabilities, with a separate labeled native fitted-model mask result; it does not present toy loss as fitted NLL.
- **Scalar state bridge:** change source values, encoder weight and update rate; inspect encoder/decoder states, probabilities, per-token costs, full signed joint derivative and proposed update. Detaching context changes this gradient path while preserving the forward calculation.
- **Probability tree:** change actual normalized conditional rows, beam width and displayed step; see every complete leaf, retained and pruned frontier, EOS and winner immediately. A separate length-ranking pair makes the sign/convention visible.
- **Saved fitted model:** edit a short source/request, forced prefix, context intervention and cap. Real GRU calculations update encoder/decoder state traces, supported-token probability bars/tables, output and termination. Another disclosure shows actual width-1/2/3 candidate-owned beam results under a clearly separate current-source/no-intervention policy.

The source/output timeline diagram and all-seed measured curves complement those labs. Every measured curve point comes from the retained experiment; connecting lines are labeled as eye guides. Hidden coordinates have no invented linguistic names. Original source label, constructed query, forced output and natural EOS/cap status are distinct.

## Owned files and boundaries

- `src/learn/data/topics/sequence-to-sequence-encoder-decoder.jsx`.
- `src/learn/components/lesson-labs/Seq2SeqLabs.jsx`, `seq2seq-labs.css`.
- `src/learn/data/seq2seq-models.js`, `seq2seq-measurements.json`.
- `src/learn/data/curriculum/blueprints/sequence-to-sequence-encoder-decoder.js`.
- `public/learn-code/sequence-to-sequence-encoder-decoder/`: canonical programs, real attributed CSV, provenance, unchanged measurements/mechanics, source-only example and derived seed-one weights.
- `scripts/generate-seq2seq-lesson.mjs`, `verify-seq2seq-models.mjs`, `verify-seq2seq-native.py`, `verify-seq2seq-browser.cjs`.

The small JS gate helper mirrors the prepared manual-trace primitive so this lesson does not import a different lesson's analytical models or examples. The learner-facing scratch gate derivation and matched `nn.GRU` convention are owned by the linked recurrent lesson; this topic owns the encoding/decoding/teacher-forcing/search protocol. No shared registry, manifest, generated navigation or central ledger edits were made by this author.

## Actual verification

`node scripts/verify-seq2seq-models.mjs` passes seven groups. It checks shifted targets and masks; scalar derivatives against the saved native/autograd calculation and independent central differences across changed values; search normalization/frontier history and score conventions; all seven complete saved source/prefix/context trajectories; replay/null/cap cases; 27 actual native beam fixtures; and JSX/content/deferred-resource contracts. JS state/probability agreement against independent NumPy uses tolerance 1e−10; observed maximum probability difference is 2.054e−15. The 27 beam comparisons take roughly 143 ms combined in the Node author run, not a browser/device latency promise. Evidence and source hashes: `docs/teaching/evidence/seq2seq-author.json`.

`scratch/lesson-tools/Scripts/python.exe -X utf8 -B scripts/verify-seq2seq-native.py` passes seven groups on Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu. It verifies source/data bytes, actual split/shape/parameter/metric counts, reproduces all 447 seed-one development token sequences, executes 27 native beams, reruns the complete mechanism program with exact JSON agreement, executes the standalone inference program, and checks batch/source ownership with capped generation. Evidence: `docs/teaching/evidence/seq2seq-native.json`.

Original three final-split training runs remain unchanged source-bound evidence. They were not refit for presentation changes. The obsolete author checker binds the pre-rewrite historical body hash, so it is not rerun as an implementation checker; the new native verifier reestablishes its substantive current contracts without mutating prepared files or pretending that old-source assertion remains applicable.

Primary-source refresh rechecked PyTorch 2.14's GRU convention, current Transformers generation length/stopping descriptions and the original seq2seq source record. Existing detailed source reading and annotated resources remain, with no claim of newly watched videos or pretrained experiments.

## Independent and browser review

Independent review by `convnext_finish` checked the full manuscript, specifications, source and implementation; its full assessment is in [SEQ2SEQ-INDEPENDENT-REVIEW.md](SEQ2SEQ-INDEPENDENT-REVIEW.md). A fresh unequal-length native batch covers four records and 32 decoder positions; all 3,328 compared state/probability values agree with the browser model, maximum difference 2.776e−15. The batch includes minimum/maximum allowed source lengths and every ignored padded position. Deliberately incorrect, empty and nonfinite fixtures fail the comparison guard. Evidence: `docs/teaching/evidence/seq2seq-independent.json` and `seq2seq-independent-native.json`; the independent verifier is separate from this author's tests.

Review corrected three concrete issues before completion: continuous sliders now preserve precisely typed values such as 0.123 instead of silently displaying a rounded step; visible inputs/selects/buttons have a 44 px minimum target; and the padded teacher-forcing track now shifts the entire padded target exactly as the canonical batch does, preserving EOS as the input at the first ignored target. Chart axes and labels explicitly use neutral colors rather than inheriting green-tinted defaults.

The author browser suite passed eight groups on development port 4195 and again on the final production build at port 4194 after these fixes. It exercises all nine continuous controls with 0.123 numeric/range equality, pointer and keyboard sliders, signed/zero values, actual ignored-token alignment, independent native beam output, caps/EOS, source/prefix/context edits, lazy resources, failure/retry, neutral theme and 1366/390/320 geometry. `development-report.json` records the earlier checkpoint; `docs/teaching/evidence/seq2seq-browser/report.json` binds the final production check to the current topic/component/model/styles.

All twelve retained production captures were opened and inspected: scalar bridge, tree and fitted-model labs at 1366/390 px; alignment and fitted-model labs at 320 px; measured curves and source/output timelines at 1366/320 px. Token wrapping, signed values, measured axes, state traces and live controls remain readable without page overflow or overlapping lesson elements. Wide comparison tables retain contained horizontal scrolling. Screenshot-only styling hides unrelated fixed global navigation/skip-link chrome during tall element captures so that it does not obscure the lesson; it does not alter the product or its lesson content. No unresolved author or independent finding remains. Root still owns shared final integration and the central ledger.

Regions: `seq2seq-alignment`, `seq2seq-bridge`, `seq2seq-tree`, `seq2seq-fitted`. Exercise current values on load, numeric blanks/invalid retention, actual slider keyboard/pointer, source/prefix quick edits and reset, caps 1 and 16, detached gradient/zero-rate, probability extremes, unchanged future-prefix positions, EOS preservation, optional code/weight failures and retry, and navigation during an in-flight resource fetch. Fetches cancel when the resource closes/unmounts. Retained old data is not labeled as a new response.

Neutral surfaces override the older shared green lab background locally; no generic unscoped SVG sizing is added. Root should retain the historical title/publication mapping, register the semantic blueprint, and link the final source-bound integration report before marking the central implementation phase complete.
