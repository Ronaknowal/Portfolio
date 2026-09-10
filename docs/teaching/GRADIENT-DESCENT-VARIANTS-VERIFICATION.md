# Gradient Descent Variants — completed author verification

10 September 2026. Mathematics position 9; stable route `gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars`. The authored lesson is complete and its six semantic source files are frozen at **11:00:14 UTC**. This is author and independent review, not user acceptance. Root owns the subsequent production build, publication/loading checks and shared progress ledger.

## What changed and what was preserved

The previous published body was read in full and saved in `scratch/gradient-variants-authoring/original-lesson.jsx`. The title/ID and useful original coverage remain, including the scalar momentum trace 5 → 4 → 2.3 → .31. The former incomplete trust-ratio fragment is replaced by a complete LARS/LAMB implementation with explicit conventions. The lesson now derives sampling, memory, coordinate scaling, bias correction, coupled versus direct decay, layer-relative movement and exact numerical continuation. It does not present optimizer names as universal guarantees or a synthetic workload as a neural-network benchmark.

The final route has nine core sections, five distinct interactive investigations, two inline figures, nine complete NumPy programs and six independent practice groups with hints and explained acceptance checks. Counts follow learning needs. The [individual design](GRADIENT-DESCENT-VARIANTS-DESIGN.md) records assumptions, retained coverage, source review and the [destination note for Learning Rate Schedules](topic-notes/learning-rate-schedules-cosine-warmup-onecyclelr.md). That is the next catalogue topic, irrespective of publication status.

## Native and mathematical checks

Command: `node scripts/verify-gradient-variants.mjs`, invoking `scripts/verify-gradient-variants-native.py` through `scratch/lesson-tools/Scripts/python.exe`. Final mathematical run **10:55:31 UTC**, Python **3.12.14**, NumPy **2.3.5**. [Actual result](../../scratch/gradient-variants-verification/results.json) is at repository-relative `scratch/gradient-variants-verification/results.json`; generated cases/programs and model-boundary evidence are alongside it.

Passed:

- All **9 complete programs** ran independently and matched their displayed stdout.
- **540 batch cases:** exact subset enumeration, objective evaluation, means and variances.
- **5,179 momentum frames:** independent coordinate matrix powers, including lookahead and bounded divergent traces.
- **6,912 adaptive frames:** direct geometric weighted-history sums, separately correcting raw moments; no recurrence copied as its own oracle.
- **1,080 decay frames:** coupled gradient histories versus direct parameter shrinkage.
- **1,944 layer blocks:** independent vector norms and the stated ratio/fallback rules.
- **2,100 updates of the actual native helper** on unseen five-dimensional histories, and **200 actual block updates** with arbitrary seven-dimensional prior states.
- **20 native boundary/failure cases**, **23 model boundary groups**, small nonzero formatting and six independent practice groups. Six extreme-gradient cases verify that a failed float64 calculation leaves the persistent helper's parameters, both histories and counter unchanged.

The complete helper definitions embedded in separate runnable programs are AST-identical. The class commits a candidate finite state only after arithmetic succeeds. The stateless block example remains an educational float64 implementation with explicit finite-input/shape/state validation, not an arbitrary-range numerical library. Expected outputs were first captured once, then verified independently; they were not regenerated to make later failures pass.

The [independent finite review](GRADIENT-VARIANTS-INDEPENDENT-REVIEW.md) found no actionable correctness defect. Complementary evidence under `scratch/gradient-variants-independent-review/` includes **324 exact-rational Nesterov coordinate states**, **768 exhaustive noise paths** against finite-time moment identities, and **120 native block-vector cases**. Its scope included sampling assumptions, stationary noise, update units, LARS rate placement and LAMB raw/corrected state, but not a full audit of every cited proof or framework implementation.

## Actual browser and visual review

Command: `node scripts/review-gradient-variants.cjs`. Final full interaction run **10:58:02 UTC** on local port5173 with headless Microsoft Edge at **1440×1000** and **390×1000**. At each width:

- **195 exercised states:** 47 sampled-gradient, 30 momentum/lookahead, 98 adaptive-history, 10 decay and 10 layer states.
- Every method/profile/preset, all sliders, correction toggle, step/rewind/restart controls, zero cases and finite divergence behavior checked. Removing the last selected observation shows an alert and preserves the valid state.
- Native keyboard slider movement, Enter/Space step controls and independent-practice disclosure checked. There are 36 lab focus targets; this count alone is not an accessibility certification.
- Nine real in-page anchors reach their sections. All nine complete program strings and outputs render. All six practice groups start with solutions closed. Twelve annotated reference/alternative anchors render through the actual Sources component.
- No page errors, KaTeX errors or document horizontal overflow.

Command: `node scripts/review-gradient-variants-reading.cjs`. Final normal-reading/geometry check after the source freeze covers **1440, 390 and 320px**: all nine reading sections, both inline figures and eleven displayed equations. No equation overflow or SVG text outside the view box. Three locally wide tables at390 and four at320 were focused and scrolled with actual ArrowRight keys. Tables keep readable text with local scrolling; they do not widen the document.

The author actually opened screenshots for all five teaching mechanisms: the sampled-gradient bowl; momentum/lookahead parameter and loss plots; sparse coordinate history; coupled/AdamW arithmetic; and shared-scale layer bars. Also opened both inline figures, sections1/3/4/5/6/7/8/9 in ordinary reading and the full references at390. Final mobile screenshots confirm complete select labels after the width fix. Evidence lives in `scratch/gradient-variants-browser/`, with `results.json`, `reading-results.json`, `batch-default-*`, `momentum-*`, `adaptive-sparse-*`, `decay-zero-*`, `layer-lamb-*`, `inline-*`, `reading-*` and `sources-*`.

Plot provenance is explicit. The momentum plane uses the same pixels per coordinate unit and exact quadratic level sets; arrows show actual and plain-gradient displacements with the same rate. The loss is computed from every displayed iterate. Adaptive traces replay stated synthetic gradients rather than claim shared-objective training performance. Layer bars use one linear norm scale across both blocks, so a smaller absolute update is actually drawn smaller. Tables provide exact rounded values and zero-norm interpretations.

## Resolved findings and limits

Visual review corrected clipped mobile select values by using a full-width control row and split the noisy-iteration recurrence to fit320. It also moved setup before the first program and removed two references to the authoring history from learner-facing momentum prose. An earlier geometry pass aligned x/y units and gave both layer bars a shared norm scale. Source formatting preserved normalized AST, JSX and template values, recorded in `scratch/gradient-variants-verification/formatting-results.json`.

Two initial browser failures were harness assumptions: Playwright's `uncheck()` expects a last-checkbox removal that this UI correctly rejects, and an `output` has an implicit status role alongside the explicit status paragraph. The harness now clicks the rejected action and targets the intended paragraph. A first practice-oracle exact-float comparison was replaced with exact rational arithmetic. These were not product defects. The320 equation overflow was a real presentation finding and was fixed before the final reading pass.

Primary papers and versioned PyTorch2.14 documentation were actually read at the sections recorded in the design. This lesson's executable examples are NumPy; PyTorch execution is not claimed. The direct Stanford video is curated using its official listing/description and selected companion-slide text. No full video playback, transcript review or full PDF visual review is claimed. Historic defaults and reported distributed speedups are not adopted as universal guidance.

No live learner study, assistive-technology study, production GPU/distributed benchmark, arbitrary-hardware reproducibility guarantee or full optimization convergence proof is claimed. These checks establish the lesson's declared examples/models and tested local reading/interaction behavior. Production integration remains root's separate responsibility.

Final semantic SHA-256 fingerprints and freeze timestamp: `scratch/gradient-variants-verification/final-source-hashes.json`. Title-only/example-heading polish after the full numerical run changes no executable Python code or formula; the final ordinary-reading pass covers it. The independent review preceded the additional failure-atomic helper guard; that guard has separate native failure cases above. No catalogue/shared manifest or existing lesson identity was changed by this author.
