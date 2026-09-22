# Implementation depth: build, use and extend

User-requested review, 21 September 2026. This is an assessment of how lessons teach implementation, not a new curriculum queue or a claim that a finite curriculum guarantees every research/engineering task. The [canonical standard](../../LESSON-TEACHING-STANDARD.md#build-the-mechanism-then-control-the-library) and [design ownership map](TOPIC-DESIGN-BRIEF.md#4-design-complete-examples-and-practice) now require both the mechanism and ordinary tool route, a bridge between them, meaningful customization practice and a high-quality, efficient reference implementation with stated limits.

## What prompted the change

The earlier standard required runnable code and visible mechanisms but did not explicitly check these as separate learning outcomes. Some lessons already have substantial handwritten algorithms, while others explain formulas and then jump to a library fit. Browser-model source, a derivation, a downloaded experiment and an independently implementable algorithm are different kinds of evidence. An audit must inspect the actual learner-facing source and its explanation.

The scratch boundary depends on the subject. Loss code may use NumPy primitives; a new neural layer may reuse the earlier autodiff engine. Numerical mathematics also needs assumptions and justified derivation; operational topics need an inspectable reproducible workflow rather than a reimplementation of the entire OS. No artificial identical page template is prescribed.

## Coverage and records

- [Foundations](implementation-depth/FOUNDATIONS.md): Programming, DSA and Mathematics; exact per-topic evidence and scope limits.
- [Classical ML](implementation-depth/CLASSICAL-ML.md): source-located review of both implementation routes.
- [Prepared Deep Learning](implementation-depth/PREPARED-DEEP-LEARNING.md): manuscript/program inspection, distinct from implemented or freshly executed lessons.
- The first five implemented DL topics have individual records in `implementation-depth/<stable-id>.md`. The Loss/Normalization additions have a separate independent review in that directory.

Read each row's disposition. **Breadth inspection is not a fresh full correctness, performance or mastery review of every program.** Unchecked mechanisms and recorded gaps remain visible. A missing per-topic record means no standalone depth-review receipt, not that the topic is complete or deficient. Historical content/implementation completion stays in the phase ledger; changed lessons receive a new reviewed revision. The new requirements do not silently certify unchanged lessons.

## Continuation rules

The normal topic preflight now returns `implementationDepth` for every topic, including existing prepared packets, with the standard, this review and any per-topic record. Future authors must map each core mechanism to its local implementation or actual prerequisite owner, inspect the cited coverage, and resolve gaps before completing the authorized phase. The relevant module report and saved destination notes supply known findings. Do not simply add links or imports and mark a gap closed.

Content-first authors provide complete explained scratch/library code and the comparison/extension plan, with truthful execution status. Finish authors implement the promised route, execute relevant code, verify mappings and discoverability, and retain browser quality. Existing baseline experiments and unchanged labs do not require repeat campaigns merely because a new small comparison was added. Recheck affected claims and source paths, keep evidence bounded, and record unresolved work honestly.

High quality includes appropriate algorithms, numerical stability, reusable interfaces, meaningful edge checks and resource reasoning. A pedagogical trace may be small; the final owned routine must implement the promised contract competently. Do not claim universal optimality or GPU/library throughput from an unmeasured reference example. Optimize for the stated workload without compromising the explanation or correctness.

## Focused changes delivered

Seven lessons received substantive additions: NumPy's scalar-to-vectorized mechanism; Perceptrons' stable activation/affine/library route; Backpropagation's same-state training-step bridge; Loss Functions' explicit objectives, gradients and update; Normalization's manual VJPs; Transfer Learning's manual LoRA gradients/update/merge; and Feature Scaling's fitted-state scratch/library comparison. Each adds interpretation and changed-input implementation practice. Existing training experiments are preserved. New full programs load on demand, use topic-owned data, and remain directly downloadable. A production phone check also found and repaired Perceptrons' previously uncontained display equations with a lesson-scoped math scroller; no mathematical SVG geometry rule was changed.

The independent checks cover the changed code: Loss/Normalization's 55 complementary checks and repaired tiny-vector/default and corner-derivative conventions; scalar-coordinate and factor-rescaling LoRA oracles; finite-difference activation slopes; and NumPy/Feature Scaling permutation, translation, ownership and extension checks. Author checks cover the broader API fixtures. These are bounded correctness checks, not benchmarks or proof that every program is optimal.

Final production checks use desktop, 390px and 320px views, actual code open/close and download behavior, keyboard focus/scroll, accurate source/output, lazy requests, page containment and math/runtime error checks. Their retained receipts are `evidence/implementation-depth-browser.json` and `evidence/neuron-implementation-depth-browser.json`. Native/independent receipts are linked from the seven topic/module records. The build retains its existing large-catalogue-chunk warning; this scope does not claim to have eliminated that unrelated bundle cost.

The subsequent [35-topic remediation](implementation-depth/REMEDIATION.md) closes all eleven foundation library gaps, nineteen deeper-review dispositions and three Classical ML specialist bridges, plus focused Logistic Regression/GP convention comparisons. Their independent reviews and final production checks are complete. The subsequent [prepared writing revision](implementation-depth/PREPARED-WRITING-REVISION.md) now resolves five specialist-route notes and broader ownership checks in the written content of all 37 prepared packets. Their new content revisions are tracked separately; website implementation remains deferred. This closes the recorded implemented-lesson findings, without claiming every possible secondary mechanism or performance workload has been certified.

## Revision boundary

Closed 22 September 2026 (local date). [The final reconciliation](evidence/implementation-depth-integration.json) records seven completed revision-3 checkpoints, preserves their prior revisions and proves the other 170 phase rows unchanged. The inventory still reports 177 complete content checkpoints, 140 complete implementations, 37 prepared implementations, 231 published lessons and 1,460 stable topic IDs. Curriculum, generated-artifact, import-boundary, runtime-organization and phase checks pass. The production build passes with its pre-existing catalogue-size warning. The seven-topic production receipt has 33 affected check groups and 21 captures; the complementary three-topic program receipt has seven groups and six inspected captures. These counts describe the tests actually run, not a curriculum-wide correctness guarantee.

`evidence/implementation-depth-baseline.json` preserves the prior phase rows and manifest identity before the new revisions. The existing published IDs, module order and user progress are conserved. Final checks and changed-source reconciliation are recorded in the implementation-depth evidence/records; unmodified historical verification is reused only for unchanged sources. User acceptance remains separate.
