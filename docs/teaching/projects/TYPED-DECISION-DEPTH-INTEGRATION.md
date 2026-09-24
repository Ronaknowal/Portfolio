# Typed decision project — depth revision 2 integration

Completed 22 September 2026. Authorized scope: deepen the existing project and link the relevant detailed topics at their point of use. All eight stable stage IDs are preserved. This is project revision 2, not completion of its planned curriculum companion or a new lesson queue.

## Delivered learning flow

The guide follows a concrete request through IDs and tokens, candidate ordering, two kinds of padding masks, embedding/attention/FFN/gather shapes, the shared scorer, cross-entropy and parameter gradients, clipping/AdamW, calibration and cost decisions, evaluation and saved inference. It includes worked numbers, canonical source explanations, solved exercises and contextual links. The completed source/link review establishes 31 contextual links to 24 distinct topics; three targets are explicitly planned outlines. Source views use the actual downloadable programs and fetch only when opened.

Three new live investigations supplement the existing probability/cost explorer: actual tokenizer/marker construction, an explicitly constructed attention mixture, and a frozen-feature shared-head SGD update. They expose different mechanisms with immediate changes, reset controls and no prediction-entry feature. The main native model uses AdamW and trains the whole encoder; the small gradient lab clearly explains its narrower scope.

The new research tool imports the unchanged core, emits an exact trace, and calculates diagnostic tables from saved artifacts. The Transformers adapter provides a runnable tokenizer/encoder/head pipeline, training, calibration, saving and prediction. It was tested with tiny random local encoders, including full and frozen training flows; no pretrained weights were downloaded. Meaningful pretrained accuracy, RL, real support data and a network service remain extensions.

The guide preserves observed failure rather than promising a successful research outcome: the lexical baseline wins on the constructed test fixture, the temperature fit worsens test loss, action policies incur more cost than review-all under the stated fixture assumptions, and candidate ordering changes probabilities substantially. These outcomes have executable diagnostic support.

## Verification and final paint

- [Author checks](evidence/typed-decision-depth-author.json): six substantive groups cover parsing, catalogue IDs, canonical excerpts, native/browser encoding parity, attention arithmetic, finite-difference shared-head gradients and worked numbers.
- [Native record](typed-decision-model-native-depth.md) and [execution evidence](evidence/typed-decision-native-depth.json): exact trace, real checkpoint diagnostics, offline library adapter and artifact restoration. The original small-model source and measured report remain unchanged.
- [Independent depth review](typed-decision-model-depth-review.md): separate numeric constructions, masking and ID invariance, calibration/cost denominators, derivative checks, artifact parity, full prose and final source-link review. No unresolved material blocker in the authorized scope.
- Production build passed with `node node_modules/vite/bin/vite.js build --logLevel warn`. The existing shared catalogue chunk warning remains; this revision does not introduce model weights or eager project bodies into discovery pages.
- `node scripts/verify-typed-decision-depth-browser.cjs`: [28 integration groups](evidence/depth-revision/depth-browser.json), all 17 canonical source excerpts, actual input/reorder/checkbox changes, keyboard and pointer slider input, zero-rate behavior and reset, and every stage at 1366, 390 and 320 pixels. No uncaught browser errors or page-width overflow.
- `node scripts/verify-learning-workspace.cjs`: [39 scoped workspace checks](evidence/depth-revision/workspace-browser-review.json), preserving routing, progress isolation/persistence, lazy loading, failed-load recovery and the original probability/cost controls. Revision-1 evidence was retained separately.
- Final visual inspection covered the encoding, attention and training desktop captures, training on phone, workspace desktop, and original cost explorer at desktop/phone. A phone table initially hid important gradient columns behind horizontal scrolling; live tables now stack into labeled values on small screens. Final screenshots confirm every value remains readable. Theme remains neutral charcoal/black and amber.
- Project metadata/registration validation passed for all eight stages. Generated-artifact validation reports 1,461 topics, 29 modules, 231 publication mappings, 486 separate outlines, and zero stale generated files. No lesson phase checkpoint changed.

The browser evidence lives in `evidence/depth-revision/`. It was produced from the final source after the mobile repair and the final clarification distinguishing the 24-token native trace row from the 28-token diagram row. Mathematical/native files were unchanged by that clarification; their passing checks were reused. The independent review records final integration hashes separately.

## Handoff and retention

[Project ledger revision 2](project-delivery-progress.json) records content, implementation, independent review and browser integration complete; user review remains pending. Revision 1 is preserved in its history. The project authoring rules in `docs/engineering/LEARNING-WORKSPACE.md` now require locally explained operations, relevant links with reasons, runnable library bridges and topic-specific investigations, without enforcing one visual template.

Only the task-owned temporary dependency and native-run directories were removed after retaining evidence. Existing tools, unrelated scratch work and user changes were preserved. No deployment or commit was made. Continue only under a new scoped request; do not reopen unchanged evidence or infer completion of the planned companion lesson.
