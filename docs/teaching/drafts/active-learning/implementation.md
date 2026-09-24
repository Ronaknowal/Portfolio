# Active Learning — prepared-content implementation

19 September 2026. This continues the complete retained [manuscript](lesson.md), [research/design](design.md) and [visual specifications](visual-specifications.md). The root agent authored the implementation; a different agent performs the independent review. The original content-first checkpoint is unchanged. User acceptance and final shared integration are separate.

## Coverage and representation

The full ten-section manuscript is implemented as static JSX, including the information boundary, finite threshold family, three uncertainty scores, committee decomposition, batch coverage, complete executed Banknote experiment, annotation workflow, optional deeper connections, eight changed practice tasks with sixteen separately closed hints/solutions, and annotated references. The stable ID, catalogue identity and module sequence are retained. Nothing substitutes an outline for the prepared prose.

Seven inline figures teach distinct mechanisms: a label-access flow, aligned threshold rulers, probability strips, an entropy decomposition, equal-scale coordinate geometry, measured acquisition curves, and an annotation-state timeline. Three separate investigations let the learner edit hypotheses/observations, member distributions, and candidate coordinates/probabilities. These are different learning activities, not one reused parameter panel. Captions distinguish exact authored calculations, conceptual diagrams, and measured results.

The recorded Banknote experiment has 31 actual budget checkpoints for each of five paired initial-label sets and four strategies. The reader can inspect means or individual runs, an exact table, and the acquired-label replay. The replay exposes an oracle answer only after acquisition. The plot does not invent confidence intervals or interpolate unmeasured checkpoints; its cropped vertical scale is explicit. Browser execution uses retained measured output and bounded small teaching models, not scikit-learn or background fitting.

The threshold lab separates setup from prediction, retains hypothesis identities across elimination, supports a genuinely uninformative query and inconsistent oracle evidence, and exposes weighted expectations only after commitment. The committee lab handles shared ambiguity, opposing confidence, identical members, normalization and stated vote ties. The geometric lab compares the learner's batch with entropy and farthest-first selections using the same inputs/budget, exact distance tables, step traces and an all-coincident null. Prediction inputs start unset; edits invalidate results. Same-input replay is distinct from a fresh prediction.

## Ownership

- `src/learn/data/topics/active-learning.jsx`: complete static reader.
- `src/learn/data/active-learning-models.js`: bounded pure teaching calculations.
- `src/learn/data/active-learning-experiment.json`: actual measured curves/traces from the packet.
- `src/learn/data/active-learning-examples.js`: exact executed downloadable program as source text.
- `src/learn/components/lesson-labs/ActiveLearningFigures.jsx`, `ActiveLearningLabs.jsx`, `active-learning.css`: topic-specific graphics, interactions and scoped responsive styling.
- `src/learn/data/curriculum/blueprints/active-learning.js`: topic-owned pedagogical metadata.
- `public/learn/downloads/active-learning/`: unchanged program, licensed dataset and provenance.
- `scripts/build-active-learning-lesson.py`: author-time manuscript compiler using Mistune; no Markdown parser ships in the browser. Complete programs/data are copied reproducibly, not abbreviated for display.

## Correctness checks

- [Native execution](../../evidence/active-learning-native.json): both complete recorded experiment invocations and the packet's smaller author calculations reproduced; **5,981 numerical leaves** compared. Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1. The exact displayed program equals the downloadable source, and all three download files retain packet bytes. Temporary execution directories clean themselves up.
- [Model/content checks](../../evidence/active-learning-models.json): **552 independent numerical comparisons**, using explicit possible-world averages, the KL-divergence identity for committee disagreement and separate squared-distance/brute-force geometry references. Includes weighted hypotheses, entropy grids, ties/nulls, contradictions, invalid inputs, approximation bounds, complete ten-section structure, all sixteen practice disclosures, source escaping and exact code identity.
- [Browser checks](../../evidence/active-learning-browser.json): **21 grouped checks** at 1440, 780, 390 and 320 px, with informative desktop/phone captures. Keyboard commitment, predictions, sequential acquisition, contrast/null grading, edit invalidation, resets, query masking, replay visibility, full section route, math rendering, page bounds and increased text at an intermediate width are exercised. Actual painted SVG label fill and minimum probability-input width are asserted; the DOM alone did not expose those presentation defects.

The author inspected the seven inline figure types and informative states for all three investigations at desktop and phone widths. This inspection caught black axis labels inside the lab (the original CSS reached only inline figures), a probability input falling into a narrow identifier column, and fixed-navigation strips obscuring tall screenshot captures. The production CSS now covers both visual contexts and gives probability input its own wide column; screenshot-only styling hides navigation during isolated captures. Long equations have keyboard-focusable local scroll regions and no document overflow. Labels, keys and tables preserve coincident point identity without moving data marks.

## Learning-experience and research review

The reader encounters purpose and tangible inputs before formal selection objectives. The worked examples remain understandable without manipulating the labs, while the labs expose counterexamples and intermediate states. Practice asks for changed calculations, explanations, protocol repairs and deployment reasoning; optional depth does not block the first pass. No finite experiment is presented as proof that active learning always helps.

Existing prepared research is reused for unchanged claims. During implementation, the official [CMU 10-601 course page](https://www.cs.cmu.edu/~ninamf/courses/601sp15/lectures.shtml) and [1 April Active Learning slides](https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/20_al_4-1-2015.pdf) were opened; the pool/stream, threshold and sampling-bias sections were inspected. The companion slides are now directly linked beside the alternate resource. The video remains a course-linked resource, without a claim of full playback. Dasgupta's linked survey introduction/query-selection discussion was also inspected as a primary conceptual source. The lesson remains self-contained.

Scope/title remain suitable. No new destination-topic note was required. Broader Bayesian acquisition and calibration are linked as subsequent depth rather than silently added to the current prerequisites.

## Continuation

Author implementation is complete. The [independent review](../../ACTIVE-LEARNING-INDEPENDENT-REVIEW.md) identified two reachable edge cases: selecting an already seeded question at startup, and floating-point noise defeating the stated identifier tie order for symmetric binary entropies. Startup now selects only unused questions (or the explicit exhausted state); entropy differences within 10⁻¹² nats use the visibly documented tie rule. The reviewer checked the changed cases, exhausted-query null and additional weighted/multiclass/geometric examples. The affected author model/browser campaigns passed again; exact native programs and their evidence are unchanged. Final production loading checks and the central phase-ledger checkpoint remain with the increment owner. Reuse unchanged evidence rather than restarting prior stages.
