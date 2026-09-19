# Semi-supervised learning — prepared-content implementation

19 September 2026. This record continues the complete [manuscript](lesson.md), [design/research](design.md) and [visual specifications](visual-specifications.md) without changing their phase-one checkpoint. Author: the semi-supervised implementation agent. Independent review and final shared integration belong to the increment owner and are separate from the author checks below. User acceptance remains separate.

## Coverage and ownership

The stable title, topic ID and Classical ML order are preserved. The published body contains the full ten-section prepared lesson, including the local probability/matrix refreshers, graph derivations, electrical and random-walk interpretations, hard versus soft normalization, final-refit self-training, categorical recipient-side co-training, the banknote experiment, cost analysis, optional deeper families, all eight changed practices with separately closed hints/solutions, and annotated alternate resources. No section was reduced to an outline. The earlier packet's canonical-reference and original-conservation maps remain the scope authority.

Production ownership:

- `src/learn/data/topics/semi-supervised-learning-label-propagation-self-training-co-training.jsx`: complete static reader body; integrated route and explicit reading/practice estimate.
- `src/learn/data/semi-supervised-models.js`: bounded pure graph, prototype and categorical models, parsers and provenance.
- `src/learn/components/lesson-labs/SemiSupervisedFigures.jsx`: seven concept-specific inline figures plus their accessible tables.
- `src/learn/components/lesson-labs/SemiSupervisedLabs.jsx` and `semi-supervised.css`: three editable prediction-gated investigations and responsive presentation.
- `src/learn/data/semi-supervised-examples.js`: exact executed code and stdout for the three complete programs.
- `public/learn-assets/semi-supervised-learning/`: unchanged licensed CSV/provenance and the three runnable downloads.
- `src/learn/data/curriculum/blueprints/semi-supervised-learning-label-propagation-self-training-co-training.js`: topic-owned outcomes, sequence, practice, scope and references. Shared registration is performed by the parent.
- `scripts/generate-semi-supervised-lesson.mjs`: constrained topic-specific author-time conversion from the complete retained manuscript to static JSX. It verifies the graph and banknote code blocks match their downloadable programs; no Markdown parser or preparation file enters the browser.

## Specification disposition and implementation decisions

| Contract | Implementation |
| --- | --- |
| F1 label access | Separate fit/select/report compartments; six observed labels, 314 hidden training targets and 160 evaluation labels remain explicit |
| F2 same inputs/different targets | Two aligned scatterplots with identical coordinates, shape-coded labels, exact-coordinate table and authored-example caption |
| F3 hard-clamped graph | Fixed endpoints, synchronous intermediate/equilibrium table, weighted edge key and the same circuit interpretation |
| F4 raw versus normalized | Separate numerically scaled evidence/readout bars, explicit classes and S row sums; zero evidence is unavailable |
| F5 prototype provenance | Observed/proposed/refitted lanes with exact first-round and final boundary values |
| F6 paired views | Donor rule → shared row → recipient feature/rule flow, with the red/square conflict explained |
| F7 measured audit | Correct/wrong stacked counts and separate cumulative counts with exact numbers, zero baseline and no invented accuracy curve |
| I1 graph | Arbitrary 2–10 named nodes, edge edits, 0/1/unknown labels, target selection, hard/soft modes, alpha, step/replay/reset, direct solve and residual, reachable/unanchored support, full numerical contribution/readout tables |
| I2 feedback | Editable observed and unlabeled coordinates, threshold/query, keyboard and pointer range controls for each unlabeled point, synchronous offers, old means/probabilities and final refits, separate class origins and exact null/tie/reversed-order behavior |
| I3 co-training | Arbitrary 2–20 paired category rows, editable observed labels, target row/view, five transfer subphases, distinct recipient arrays/rules, conflicts, complete donor provenance, duplicate-view/no-anchor nulls |

The graph prediction asks whether the edited target's equilibrium score is below/above/tied at 0.5 or unavailable, using the specification's absolute-outcome alternative. This avoids a hidden moving comparison baseline when learners add arbitrary nodes. The worked chain and preset contrasts remain next to the lab. Learner-authored changes are available through the actual node/edge editors; the transfer prompt explicitly requests a new case and explanation rather than treating a preset click as independent mastery. No pass badge certifies completion of that open-ended task.

Numeric coordinate text plus per-point range controls replace an in-chart drag handle: they edit the same actual points with equal pointer/keyboard access. Tables retain coincident point identities and reversed class ordering. Model computation occurs only on an explicit prediction commit. Changing any mathematical input clears prediction, result, feedback and execution state; replay preserves a prediction for unchanged inputs. All-zero evidence, unknown labels, conflicts and score ties have explicit answer options.

The graph uses curved long edges in the four-node layout: the first screenshot inspection showed a straight B–D shortcut passing visually through C, which could imply a false connection. The curved edge preserves the data while exposing the actual shortcut. Short names/numeric node keys keep arbitrary names readable. Narrow numerical score cells preserve decimal integrity. Explanatory labels and axis ticks use wrapping HTML outside quantitative geometry.

Categorical provenance is a DAG. The model retains exact supporting events; its text view lists each unique event once with the supporting row IDs. It does not recursively duplicate every path into an exponentially large rendered tree. Model limits are 10 graph nodes/5,000 update steps, 20 unlabeled points/20 self-training rounds and 20 paired rows/eight co-training rounds. No heavy ML library, network fitting, animation loop or corpus download enters the browser.

## Author correctness and native verification

- `node scripts/verify-semi-supervised-models.mjs`: **12 grouped checks pass**, recorded in [model evidence](../../evidence/semi-supervised-models.json). Independent references are the retained NumPy solves and Python co-training trace, plus analytically derived changed examples. Coverage includes synchronous graph updates, shortcut/separation, all raw/normalized soft fixtures, islands/isolates/no anchors, one class, conflicting anchors, alpha-zero/high-alpha behavior, ties, parser bounds, prototype contrast/null/equality/overlap/reversal/final-refit, practice answers, categorical conflicts, no-anchor/duplicate-view nulls and order-independent proposals.
- `scratch/lesson-tools/Scripts/python.exe scripts/verify-semi-supervised-native.py`: **all three exact downloadable programs execute successfully**, recorded in [native evidence](../../evidence/semi-supervised-native.json). Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1. Native graph/categorical outputs and all seven banknote candidates agree with the packet. Baseline development/test 72/80; threshold 0.8 accepts 252 guesses, 73 wrong, leaving 62 unknown; threshold 0.95 accepts none.
- Babel JSX parsing passes for the full reader and both component files. The generation script verifies the complete displayed graph and banknote programs equal the copied runnable files.

The full existing source/research packet is retained. No empirical claim is generalized beyond the declared banknote subset/protocol. Hidden targets remain absent from promotion decisions and are opened only for retrospective audit. Deep consistency methods remain optional explained connections, not claimed reproduced runs. The original video recommendation retains its provenance distinction: associated slides and official linkage were reviewed during preparation; no claim of watching the video is added.

## Author learning-experience review

The first-pass route is sections 1–7 followed by practice; deeper connections are explicitly optional. The lesson starts with the label bottleneck and introduces the vocabulary before the matrix detail. Every title mechanism has intermediate states, a meaningful changed case, an appropriate null and independent practice. The empirical result is allowed to favor the baseline. Exact code outputs are visible, with the full categorical source behind an optional expansion. The user can learn the mechanisms without operating a lab; the inline figures carry the initial relationships.

Cautions stay with their relevant assumptions rather than being appended to every result. Model confidence, observed labels and retrospective truth have separate terminology throughout. The alternatives retain purpose/level annotations and direct source links. Scope/title remain appropriate; no new destination note is needed from this implementation.

Browser/visual evidence and independent-review closure are recorded below as those stages finish. Shared publication/navigation, production loading/build, final ledger hashes and integration are the parent's responsibility; this record does not mark those complete on the author's behalf.

## Author browser and visual closure

`PLAYWRIGHT_PACKAGE=<bundled playwright> LEARNING_BASE_URL=http://127.0.0.1:4184 node scripts/review-semi-supervised-learning.cjs` passes **13 grouped browser checks** across 1366, 390 and 320 px. [Browser evidence](../../evidence/semi-supervised-browser.json) binds the actual lesson, models, examples, component/CSS and verifier bytes. The script checks all three labs' prediction gates, meaningful contrasts and nulls, step/replay/reset, stale-result invalidation, malformed input, keyboard commit, closed practice disclosures, complete recorded code outputs, seven inline figures, downloadable inputs/programs, no KaTeX error and no document overflow. No page errors were observed. Sixteen selected captures are retained under `scratch/semi-supervised-review/` and individually hashed in that evidence.

The author inspected every desktop static figure and the informative narrow lab states (shortcut reversal, changed prototype/query and recipient transfer). The numerical strips, class identities, curves, provenance and counts are readable and correspond to their explanations. The B–D edge-routing repair and decimal-cell reflow above were verified in the final narrow capture. Long element screenshots can include the site's fixed navigation strip at the current viewport boundary; that is capture composition, not a label/edge collision inside the figure.

`audit-lesson-visual-layout.cjs --topics semi-supervised-learning-label-propagation-self-training-co-training --widths 1366,390,320 --output docs/teaching/evidence/semi-supervised-layout.json` completed without tool errors. Its three flagged figures are the same chain at three widths: it reports straight graph edges beneath A–D labels. Source/rendered inspection dispositions these as **occluded-line false positives**: opaque node circles are painted over those edges before the centered labels, with no group opacity. No label is visibly crossed. The auditor records a stale pre-existing dist manifest because this was a development-server review; it is not production-build evidence. The browser evidence's narrow, explicit source hashes identify the tested implementation.

Initial separate Vite processes contended during dependency discovery; the author's unused 4186 server was stopped. All passing browser evidence uses the parent's shared 4184 server. An early selector check exposed verbose implicit select names; matching explicit accessible names were added, and all keyboard cases then passed. These are implementation corrections, not rewrites of the prepared lesson.

Author stage is complete. The separate [independent review](../../SEMI-SUPERVISED-INDEPENDENT-REVIEW.md) then corrected tiny-conductance normalization, a conflicting-category rule count and exact prototype tick alignment; it records the complementary checks and refreshed author browser/model evidence. Independent review is complete; shared production/inventory/phase-ledger integration is owned by the parent increment record.
