# Sampling, Measurement & Experimental Design — author verification

Author-frozen at **2026-09-11T04:51:45.306991+00:00**, Mathematics49, stable ID `sampling-measurement-experimental-design`. [Exact source/evidence packet](evidence/sampling-measurement-author-review.json). Independent review and production integration belong to the parent; this is not user acceptance.

## Scope, continuity and teaching contracts

The original entry was planned, without a previous body or runnable program. Its exact inventory is preserved in [the original-plan JSON](evidence/sampling-measurement-original-plan.json); no older content was deleted. Title, stable identity and module position remain unchanged. The sole required topic is Random Variables48, whose final author-frozen source/evidence was read. Next in the actual sequence is Ordinary Differential Equations & Linear Systems50.

The [design](SAMPLING-MEASUREMENT-LESSON-DESIGN.md) records the complete scope/research rationale. Ten sections introduce the estimand and collection/measurement unit before the sampling and assignment formulas. Five distinct investigations expose finite samples, inclusion-weight contributions, independent units/repeated readings, observable potential outcomes and blocked allocation laws, and a missing factorial cell. Five inline figures connect selection stages, calibration, strata/clusters, experimental replication and bounded missing outcomes. Twelve complete programs, two early checkpoints and eleven changed practice tasks supply worked feedback. These counts describe the implemented learning work, not a reusable quota.

The core distinctions stay beside the claims: first-order inclusion is not whole-subset probability; fixed-N HT is not a realized normalized-ratio mean; a fixed offset does not become an independently redrawn error; an observed table does not contain both potential outcomes; blocking is not universally variance reducing; and bounds on absent assigned outcomes are not causal counterfactual bounds. The SRS and finite-randomization variance derivations state their assumptions. The capstone records assignment units, repeated readings, target weighting, exclusions/missingness and a reproducible synthetic procedure.

## Native and independent numerical checks

`node scripts/verify-sampling-measurement.mjs` passed **2026-09-11T01:43:58.750Z**, using `scratch/lesson-tools/Scripts/python.exe` (Python3.12.14) and the paired topic-specific Python verifier. The learner programs use only the standard library. Data: [native results](../../scratch/sampling-measurement-verification/results.json).

- 649 finite-sample/frame states against independent bit-mask subset enumeration and Fraction moments.
- 243 unequal-inclusion designs against exact joint subset laws, inclusion marginals and both estimators.
- 1,875 unit/repeat states against a full covariance-matrix sum, rather than the simplified implemented formula.
- 1,458 assignment states against Boolean allocation enumeration and exact observed outcomes; complete-design variance independently agrees with the Neyman expression.
- 65 factorial states and 256 missing-outcome cases against direct response-function and extreme-corner calculations.
- All 12 actual stored programs reproduce their captured stdout; 206 changed calls to their actual helpers agree with independent permutations and finite-population oracles.
- 25 invalid/range calls reject, including explicit nonzero-variance/squared-bias underflow cases. Two small positive accepted controls remain nonzero. Rounded displays are not arbitrary-precision guarantees.

Thus the main model suite covers **4,546 states**, separately from changed native helpers and rejection cases. [The focused weighted-protocol oracle](../../scratch/sampling-measurement-verification/protocol-results.json), run with `scripts/verify-sampling-weighted-protocol.py` at **2026-09-11T01:48:21.358631+00:00**, checks 27 heterogeneous potential-outcome protocols under all eight paired allocations. It confirms the student-weighted target independently of the JavaScript models. The actual Python seed50 choices are C2/C4/C6, contrasts5/5/5 and estimate5; the constructed constant effect remains3. The concrete alternative uses class sizes10/10/20/20/30/30 and pair weights1/6,1/3,1/2, with the required equal sizes within each pair.

## Actual browser, accessibility and ordinary reading

`node scripts/review-sampling-measurement-lesson.cjs` passed **2026-09-11T01:43:36.244Z** in Edge with actual Space Grotesk fonts at1440/390/320. Each width checks149 selected control states, including coverage failure, exact-zero variance, all permitted sample/allocation choices, reveals/resets and factorial mixtures. It verifies actual keyboard sliders/buttons/check boxes and local code scrolling, all ten anchors, all twelve complete displayed code/output/question triples, both revealed checkpoint explanations and all eleven hints/explained solutions. Probability masses and model-derived bar geometry agree with state. Nine equations fit; SVG labels stay inside their viewBoxes; no lesson console/page errors, KaTeX errors or document overflow remain.

The final narrow amendments add the concrete capstone answer, rounded-readout wording, a gap between the Reading label and its bar, and phone table-scroll help. `node scripts/review-sampling-measurement-final-reading.cjs` passed **2026-09-11T04:48:22.222Z** after resuming the interrupted author turn. It verifies all three widths, real fonts, the final weighted protocol, label clearance, zero-variance interpretation, all nine equations, thirteen direct reference links, and actual keyboard scrolling to the previously offscreen observed-outcome column. This focused pass closes those changes without pretending the earlier full suite occurred after them. [Final reading payload](../../scratch/sampling-measurement-browser/final-reading-results.json); [full behavior payload](../../scratch/sampling-measurement-browser/results.json).

The author actually opened **23 distinct saved captures** recorded by hash in the packet. These include ordinary reading, all five inline forms, finite-target bias, contribution expansion, potential-outcome observability, blocked and factorial states, references, the final zero/variance states, narrow equations, protocol output and both capstone interpretations. Full-behavior captures are explicitly labeled as preceding the final small display amendments; the eight final-reading captures were reopened after the resumed final pass. Wide tables/code deliberately scroll locally with keyboard access; prose and displayed formulas fit the page.

## Defects found and resolved

Browser execution identified literal-set JSX notation and a decoded raw comparison that static prose inspection missed; explicit text expressions now preserve the intended mathematics. The owned formatter verifies normalized Babel/PostCSS AST conservation and zero JSX transform warnings. Actual bounds exposed two clipped SVG axis labels and several wide phone formulas; the final visual layout was measured and opened. A final targeted test initially used an incorrect aria-label string; correcting that test locator did not alter production semantics. Accepted-input underflow cases now fail explicitly instead of displaying an impossible exact zero. The record distinguishes these source repairs from test-locator mistakes.

## Research, incoming note and limits

The primary-source ledger records Statistics Canada sampling/weighting sections, NIST measurement/blocks/whole-plot/interaction guidance, selected Peng Ding finite-randomization proofs, and Python3.12 API/reproducibility contracts. Official MIT14.310x video pages and selected transcripts were inspected; the annotated direct player links supplement self-contained teaching. No full-video viewing is claimed. Failed Penn State retrievals and an unresolved embedded YouTube target were not presented as inspected resources.

The [incoming Random Variables note](topic-notes/sampling-measurement-experimental-design.md) is resolved through the explicit unit/repeat covariance model and calibrated A/B estimand distinction. Its final implemented origin was read, rather than relying on design-stage claims. Domain-specific GPU, neuroscience and evaluation protocols remain with their named owners; this scoped authoring did not audit or rewrite the catalogue.

No real subjects were recruited, contacted or randomized. Finite synthetic checks are not empirical effectiveness evidence or a beginner usability study. Independent review may still request a narrow amendment; preserve this frozen record when doing so. No shared manifest/index/ledger/generated artifacts were edited by this author.

## Frozen production identity

| Source | SHA256 |
| --- | --- |
| `src/learn/data/topics/sampling-measurement-experimental-design.jsx` | `0f81b304040c9268cb97e3eabdcc87433c0ca3987f49ac656baa076d481deff8` |
| `src/learn/data/sampling-measurement-models.js` | `894a4d835f9e0b5870cf611545c4405fa779c2df79cba536dc509a7f222083d4` |
| `src/learn/data/sampling-measurement-examples.js` | `be66c6e3d3d3e6ec71a972c913dbc28e8e4e045d22698a011febf652df531079` |
| `src/learn/components/lesson-labs/SamplingMeasurementLabs.jsx` | `ed734ed23b69ceafa30123cba6c0a3f6ece04d30983c047c8ee9b515826a0c41` |
| `src/learn/components/lesson-labs/sampling-measurement-labs.css` | `ec2ae36bb19a748383a3a252b362e7adf477002efcc68b8cc315dde70b65b740` |
| `src/learn/data/curriculum/blueprints/sampling-measurement-experimental-design.js` | `38640ba0ba4db563b8d72d7f2c8a66cf0dc04a2bd233a70153593f7404a722a2` |
