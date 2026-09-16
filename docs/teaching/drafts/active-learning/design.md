# Active learning — prepared content design and author record

Topic `active-learning`; content-only assignment, batch14 of the root's next30. Content prepared after author reread; implementation not started. No production module, generated inventory, shared ledger or browser surface changed. Phase two consumes this packet and performs implementation and independent checks. This is not an independent review record.

## Source and scope decisions

Full original `src/learn/data/topics/active-learning.jsx` read across contiguous ranges0–219,220–449,450–689,690–939 and870–end to fill truncated reference output. Total993 lines; baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`, source SHA256 `921c3a231c216945e6ebd0f31e52c6a3f0aed13a84cdb4f44b095b493bb722bd`. Scoped inventory `--topic active-learning --work content` and topic-note ownership checked; no own unresolved note. Predecessor is semi-supervised label propagation/self-training/co-training; next is Evaluation Metrics. Runtime order unchanged.

Keep canonical title and ID. The authored display heading adds an explanatory subtitle; scope remains active learning. Include annotation/oracle boundaries, model uncertainty versus disagreement, geometric batch diversity, matched-budget evaluation, and sampling bias locally; these are necessary to understand the method. Detailed calibration, PAC/VC theory, full GP conditioning, supervised task formulation and temporal validation remain their owners' topics. Refresh the exact operations needed here rather than require later catalogue entries first. No new topic or unrelated inventory audit.

Conservation of existing useful content:

| Original content | Prepared treatment |
|---|---|
| Motivation, history, learning curves | Ground-up label budget/oracle problem and actual measured curves; references retained where useful without speculative ROI/history digression |
| Uncertainty/QBC/k-center intuition and equations | §§2–5 exact threshold, multiclass scores, committee entropy, nearest-center geometry with comparisons/nulls |
| Complete random/entropy/margin/committee/diversity Python examples | Complete local four-strategy real-data program; binary margin equivalence explicitly derived rather than redundant benchmark strategy; all original query mechanisms remain taught |
| modAL/library and MC-dropout introduction | Optional current scikit-activeml docs; actual trained committee experiment; MC dropout defined with actual trained-model requirement, no fake runnable simulation |
| Query traces, entropy map, benchmark plots | Checked threshold trace/probability strips/committee decomposition/geometry/31-step measured traces replace unsupported aesthetic values |
| Decision matrix and scale/cost | Criteria distinguished in flow; annotation unit/budget/queue/abstention/stopping and computation costs §§7–8 |
| Calibration, redundancy, imbalance, noisy oracles, cold start/drift | Consolidated contextual cautions; balanced-seed assumption, no hidden true-class quotas; binary-temperature ranking null; no universal stop threshold |
| Advanced BALD/core-set/BADGE | Correct primary-source equations, limited guarantees, last-layer predicted-label blocks and model assumptions |
| Six practice prompts and resources | Eight independent tasks with closed hints/reasoned solutions, executable experiment and annotated article/video/toolbox alternatives |

Corrections investigated instead of simplifying into falsehood: no guaranteed label multiplier/ROI; high-confidence wrong labels can yield large gradients; entropy nats boundedlnC; binary entropy/margin/LC same ordering; finite committee indices distinguish members/classes; all evaluated predictors use all observed data; final acquisition receives a fit; selected IDs masked even at coincident positions; geometric2-approx is not accuracy guarantee; BADGE uses one predicted label and last-layer blocks, not all hypothetical-label gradients; unrelated NumPy randomness is not MC dropout; BALD originates Houlsby2011, not Gal2017; original Gal5%-error table attribution295 toBALD is wrong (335BALD,295variation ratios,835random in primary table), so the lesson avoids importing those historical numbers. Bootstrap label support guaranteed by stated stratification, not assumed. Calibration not advertised as guaranteed acquisition improvement; rare classes cannot be selected via hidden true labels. No unsupported systems timings/ranks or LLM-oracle cost/quality promises.

## Learning-experience plan

Core outcome: operate and critique a budgeted label-acquisition process. Opening route: §1 setup → §2 exact question selection → §3 one-model uncertainty → §4 committee disagreement → §5 batch geometry → §6 executable measured comparison → §7 annotation/evaluation decisions. Optional§8 derives related objectives, GP variance reduction, BADGE, importance weights and temperature-ranking null. §9 has changed independent exercises; §10 bridges to metrics without skipping curriculum sequence.

| Learner hurdle | Teaching move | Observable evidence |
|---|---|---|
| Confusing pseudo-labeling with oracle acquisition | Queue/data-access diagram + retained oracle boundary code | Explain which answer was read and when; count seed/new/evaluation labels |
| Why one question beats another | Eight-hypothesis ruler board with expected survivors | Choose balanced query, trace survivors, identify legal null/contradiction |
| Equating uncertainty and usefulness | Multiclass strip ranking and confident-wrong gradient | Compute scores, distinguish criterion from downstream benefit |
| Equating predictive entropy and reducible uncertainty | Editable committee rows with same-mean contrast | Explain entropy decomposition and shared-model limitation |
| Repeating near-duplicate batch queries | Coordinate map with nearest-center segments | Choose distinct IDs, compute radius, separate geometry from label accuracy |
| Trusting fabricated curves or one lucky seed | Offline UCI data, complete program, paired repeated starts | Match actual31 checkpoints, final fit36 labels, dev selection/test lock |
| Evaluating on selected labels | Fixed-loss sampling toy and independent audit lanes | Distinguish selected query loss.8 from target risk.5 |

Three focused investigations plus seven inline figures and an optional evidence reader. No arbitrary one-lab rule; no tool manufacture of a fourth prediction exercise solely for consistency. Advanced branches use equations/tables because a control panel would add little. Learner changes meaningful hypotheses, committee rows and point coordinates/probabilities. All investigations record initially unset prediction, require commitment before reveal, and invalidate on every active input edit. Exact contracts in visual-specifications.md.

## Research: actual retrieval and canonical coverage

Retrieved2026-09-12. Sources used for technical facts are primary papers, author-hosted surveys/course materials, official datasets and official library documentation. Author wording/examples are newly composed; do not copy paragraphs or adopt unsupported claims from a primary source merely because it is primary.

| Source and actual material inspected | Claims / implementation decisions supported |
|---|---|
| [Settles survey](https://burrsettles.com/pub/settles.activelearning.pdf),67-page PDF, full contents and selected actual text §§2.1–2.3,3.1–3.6,6.1–6.3,6.7 opening; updatedJan26,2010/TR1648 cited2009 | Settings, acquisition taxonomy, annotation constraints; canonical map below. No imported empirical speedup, cost or label-multiplier claim |
| [Dasgupta Two Faces](https://cseweb.ucsd.edu/~dasgupta/papers/twoface.pdf),20-page PDF, abstract,§1.2 sampling-bias example,§1.3 framework,§2 threshold/version-space opening and§2.7 noise discussion | Sampling bias, confidently wrong neglected regions, finite noiseless example limitations; our thresholds/calculations independently authored |
| [Houlsby et al2011](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf),17-page PDF,§2 expected parameter entropy and equivalent output entropy identity,§3 opening,§4 contrast | BALD origin/conditionalMI/posterior assumption and myopic objective |
| [Gal et al2017 paper](https://proceedings.mlr.press/v70/gal17a.html) and linked10-page PDF,abstract,§3 dropout approximation,§4 acquisition opening,§5 protocol andTable1 | Real stochastic forward-pass requirement, approximate model-posterior interpretation; corrected prior result attribution; no dropout run claimed |
| [Sener/Savarese](https://arxiv.org/pdf/1708.00489),13-page PDF,abstract,§4.3 Eq5/Algorithm1/2OPT and robust-MIP extension | Fixed-center metric objective, distinct selected points, bound on radius only; no empirical runtime/ranking imported |
| [BADGE](https://arxiv.org/pdf/1906.03671),26-page PDF,§2 notation,§3 construction/Algorithm1/Prop1 discussion | Single predicted label, last-layer blocks, k-means++ rather than deterministic farthest-first; numeric gradient independently derived |
| [IWAL2009](https://cseweb.ucsd.edu/~dasgupta/papers/iwal-icml.pdf),8-page PDF,§§2–3 sampling skeleton andTheorem1 proof/conditions;§4 opening | Importance weights, positive support and variance limits; fixed-model toy not offered as full adaptive-training guarantee |
| [UCI Banknote](https://archive.ics.uci.edu/dataset/267/banknote+authentication), dataset metadata/license/features and actual source archive extracted in memory |480-row actual measured dataset/provenance; codes retained without unverified semantics |
| [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html),§1.16.3.4 temperature definition | Positive global temperature; binary ranking-invariance proof is original derivation |
| [scikit-activeml official docs](https://scikit-activeml.github.io/latest/index.html),actual pool loop snippet and stream introduction/import example | Optional library route; not installed/executed or advertised as equivalent outputs |
| [CMU official10-6012015 schedule](https://www.cs.cmu.edu/~ninamf/courses/601sp15/lectures.shtml),Apr1ActiveLearning entry and its official [lecture20YouTube link](https://www.youtube.com/watch?v=2BZhsEakEH8) | Video link/topic verified through official course and opened video metadata. No playback, transcript or full video inspection claimed. Matching slides failed retrieval; not cited as read. CMU2018recitation PDF opened but sparse parsing insufficient, not used substantively |

Canonical survey contents were cross-checked against the lesson rather than copying their order:

- §2 settings → manuscript§1; all three present with eligible-query implications.
- §3.1uncertainty/§3.2committee → §§3–4; complete equations, contrasting and null fixtures.
- §3.3modelchange/§3.4futureerror/§3.5variance → optional§8 equations, calculated contrasts, fixed-hyperparameterGP; no “one objective is universally best” wording.
- §3.6density → §3coverage/exploration motivation and§5geometry; no arbitrary density formula required without a chosen distribution/metric.
- §4empirical/theory → exact§2 finite assumptions and§6 actual held-out experiment; PAC/VC/general bounds reserved for their own upcoming topics.
- §5structured and related tasks → §7sentence/span/token cost and measurement applications; feature acquisition/class selection/clustering mentioned by canonical map but not misrepresented as the same labeled-instance action. Detailed methods stay adjacent specialists.
- §6batch/noisyoracle/variablecost/alternativequery/stop → §§5,7; query status, abstention, cost ratios, fixed-budget/test protocol. Multitask/model-class changing nuances only where they affect representation/interpretation, not a full new subject.
- §7related semi-supervision → opening/local bridge; RL/submodularity/equivalence queries not prerequisites and not silently claimed covered. Information-based objectives shown locally; full theories retain their owning lessons.

Substantive manuscript claim locators: oracle and costs§1/§6/§7; thresholdassumptions§2; LC/margin/entropy§3; committee/BALD§4; radiusbound§5; actualbenchmark§6; samplebias/stopping§7; EGL/GP/BADGE/IWAL/calibration§8; resource verification§10. All displayed numeric claims map to `checked-results.json.examples` or `.banknotes`, except direct simple counts from the CSV and code/version metadata in provenance.

## Author calculations and checks actually performed

- `author-calculations.py` executed on the stated shared read-only runtime. Threshold survival and legal null assertions; exact multiclass scores/entropy decomposition; masked farthest-first normal/changed/coincident cases; fixed-risk importance arithmetic; BADGE blocks/norm; advanced/exercise arithmetic and temperature ordering checked.
- Full real-data four-strategy experiment executed: each of5paired runs has6seed labels,30 distinct queries,31 fitted dev counts and final36labels. Development means at acquisition0/5/10/20/30 match manuscript. Selected entropy final test[79,79,79,80,79]/80. No hidden test evaluation of other strategies in code.
- Generated the complete learner program `banknote-active-learning.py` from the same reviewed benchmark body with a standalone run/print/save entry point; executed it and compared the **entire** result object to retained benchmark data, not only final scores. Exact equality held. Removed only that redundant generated JSON after checking its exact resolved parent and filename; retained durable checked-results.
- Supplemental arithmetic probe executed without rerunning the unchanged benchmark: expected survivors4/6.25/8 and2.5/2/4; contradictionempty; H(.2).500402423538; logisticgradient1.98/.02/expected.0396; utility ratios.12/.045; expectedrisks.2/.155; GPvariance.2/null0; binary logits[-4,−1,.5] preserve uncertainty order[2,1,0] atT.5/1/2.
- A console word-count attempt after successful equality comparison hit Windows default cp1252 decoding for Markdown; no calculation failed or manuscript changed. Subsequent reads explicitly useUTF8. No package install, browser, full production check, benchmark performance comparison, or independent content review performed.
- Full manuscript reread completed in contiguous ranges0–169,170–359,360–end, plus the full visual contracts. Corrected the signed-gradient sentence to gradient magnitude; supplied exact Windows/macOS/Linux environment-interpreter commands; added an explicit committee inline-figure placement. A final localized reread checked those edits.
- Added learner CLI `--budget`, `--strategies`, and `--development-only` because the independent exercise must actually avoid test prediction. Replayed the changed default learner program; entire benchmark result still exactly equals the retained baseline. Executed the15-query random/entropy development-only exercise:16checkpoints per run,15distinct queries,empty final_test. Retained its actual result under `practice_development_only`; no new final-test inspection. Code parameterization changes loop bound and final refit together.
- Closing author checklist completed: beginner route and local prerequisites; exact numeric units/evidence; initially unset prediction and edit invalidation; meaningful changed/null entities;16closed hint/solution panels; all local lesson download links resolve;480unique source rows and exactCSVchecksum; truthful resources; next-topic bridge and explicit phase-two boundaries. No independent or UI review claimed.

## Phase-two handoff

Implement the authored prose without reducing mechanism depth or replacing the topic-specific visuals with generic controls. Render inline figures where introduced; implement3investigations against exact fixtures and input-signature contracts; optional stored-evidence reader must never imply browser retraining. Provide genuine local downloads for the CSV and full program and accurate source attribution. Resolve relative lesson links through the site's canonical download/lesson routing conventions. Keep optional video learner-initiated.

Phase two must verify its math helpers against these fixtures, execute meaningful browser/native interactions and narrow/accessibility checks, independently inspect content/math and code, check lazy loading/resource bounds and navigation, and update implementation ledger only after those are complete. Do not mark implementation complete from this manuscript. No production code was added in phase one.
