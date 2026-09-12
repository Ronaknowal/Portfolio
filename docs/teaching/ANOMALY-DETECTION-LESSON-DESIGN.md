# Anomaly Detection — content revision 1

Date: 12 September 2026, local authoring date. Scope: research and write only, stages 1–2. The existing publication remains unchanged. This design is not a production verification record.

Stable topic ID: anomaly-outlier-detection-isolation-forest-one-class-svm-lof. Catalogue title retained exactly: **Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF)**. Position 15 of Classical ML follows DBSCAN & Density-Based Clustering and precedes Gaussian Mixture Models (GMM) & EM Algorithm. H1 uses the catalogue title; no rename, reorder, publication or shared prerequisite edit is proposed.

## Authoritative inputs and ownership

Read the current AGENTS, LESSON-AUTHORING-HANDOFF, full updated LESSON-TEACHING-STANDARD, ML domain playbook, topic-design brief, learning code standard and topic-note policy. Ran:

    node scripts/build-curriculum-inventory.mjs --topic anomaly-outlier-detection-isolation-forest-one-class-svm-lof --work content

The initial plan identified an existing published body, no individual authored brief and no incoming canonical note. UNASSIGNED contained no relevant unresolved obligation. Read the entire existing topic source, including its scratch programs, comparisons and six exercises. Root preserved its baseline; SHA256 b8aab5255ab5bc1fb6d3a8bfb002cc640e705a459c2a9f319fe8452d667aea8e. A byte copy is also retained in this draft packet. No runtime source or shared registry changed.

Owned deliverables live in [the draft directory](drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/). The [learner manuscript](drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/lesson.md) supplies actual explanations and solutions; [visual specifications](drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/visual-specifications.md) supply buildable contracts rather than built visuals. The delivery ledger remains root-owned.

## Learner task, outcomes and prerequisite decisions

The motivating task is an industrial-temperature monitoring investigation: what is unusual relative to an early reference, what workload follows a threshold, and what can the supplied event annotations actually establish? Small exact examples teach each mechanism before the real chronology.

Learners should be able to:

1. Declare an observation unit, reference population, fitting mode and intended response; distinguish point/contextual/collective patterns.
2. Trace isolation cuts, count depth and leaf correction, and explain the path normalizer without turning it into a probability.
3. Calculate neighbor radius, directed reach, local density proxy and LOF on changed inputs; keep training and new-query neighborhoods separate.
4. Interpret the RBF similarity sum and offset, including disconnected input-space support; explain what nu controls under exact optimization.
5. Orient library scores consistently, distinguish model offset from external threshold, and quantify prevalence/budget effects.
6. Reproduce the supplied data transformation and chronological protocol without future leakage, distinguish row alerts from window hits, and write a qualified result report.
7. Diagnose representation, scale, masking, drift and resource limits before proposing a more elaborate algorithm.

No new hard prerequisite is proposed for this content revision. Locally introduce averages, reciprocal/ratio units, exponential rescaling, kernel similarity, calibration split and score direction. The optional dual branch assumes familiarity with norm/quadratic objectives but explicitly derives the equations it uses. Existing SVM and optimization lessons can be further-reading links in implementation, not silent blockers. Prior DBSCAN optional OPTICS/HDBSCAN material is explicitly not required. The next GMM lesson owns EM and probabilistic mixture fitting, not this lesson.

## Original coverage: retain, repair, extend

| Original area | Disposition and reason |
|---|---|
| Isolation intuition, random partition tree, subsamples, terminal depth and c(n), score, scratch forest | Retain mechanism and executable construction. Replace decorative or unsupported output with exact one-dimensional interval calculations and a compact seeded forest. Keep actual sample size, harmonic cases and duplicates visible. Explain multidimensional axis cuts rather than making a lengthy generic tree framework the beginner's entry point. |
| LOF kth radius, reachability, lrd, ratio, brute-force implementation | Retain every computational stage. Add ID-based self exclusion, tied-neighborhood convention and positive-reach assumption. The original first-five-points description and reported outputs are not inherited as proof. |
| OCSVM primal, dual, kernel decision and library use | Retain complete local derivation and a full executable boundary example; the real protocol executes the library. A two-anchor analytic solution exposes gamma geometry without an opaque external QP solver dominating the explanation. |
| Comparisons, scaling, dimensionality, streaming, tuning and evaluation | Retain useful distinctions, state cost assumptions and remove universal row/time cutoffs or unsupported “best drop-in” claims. Tie decisions to the same available reference information and a real simple baseline. |
| Original exercises: parameter/path calculation, nu, LOF, diagnostics and implementation | Preserve their learning intentions in changed tasks A–J, with intermediate reasoning and hints. Repair the original path-score arithmetic and add budget, observability and event-matching transfer. No original false result retained solely for byte continuity. |

Material original claims repaired:

* One-Class SVM is not generically an interchangeable smallest sphere/halfspace description. The origin-separating formulation is taught; sphere equivalence is not asserted without conditions.
* Nu is not contamination, true prevalence, exact fitted negative fraction or a guaranteed future false-positive rate. State exact optimization, nonzero rho and strict margin violations; distinguish numerical boundary counts.
* An RBF input-space region need not be connected. Increasing nu is not taught through an unsupported universally “looser boundary” rule.
* Gamma='scale' is a variance-based parameter rule, not per-feature standardization, dimensionality reduction or sigma itself.
* IsolationForest score_samples is the negative normalized isolation score, not negative average path length. Special harmonic cases and actual fitted psi matter.
* LOF self exclusion is by row identity. Training-array query scoring differs from training factors even with the same coordinates. Original tied-neighborhood and fixed-k library definitions need separate contracts.
* Library epsilon stabilization is an arithmetic convention, not evidence that duplicate points are distinguishable. The small scratch program declares its distinct-coordinate domain.
* Contamination changes selected offset, not truth. Strict threshold/ties, future shift, ranking metrics and budget are explained separately.
* A random train/test row split can leak temporal events or groups. Scores and unreviewed events do not provide fault labels.
* Dataset descriptions do not justify universal fraud prevalence, speed benchmarks, claimed optimal subsample sizes or guaranteed convergence with a fixed forest size. No invented measured benchmark survives.

## Learning route and representations

The first-pass route follows task → reference/score/action → isolation → local density → novelty contract → kernel boundary → method choice → workload → real chronology. Optimization and resource branches remain optional but complete.

| Hurdle | Representation and learner action | Changed proof of understanding |
|---|---|---|
| Unusual versus bad; context hidden by features | Point/context/sequence examples and observation→reference diagram | Explain scheduled shutdown and construct a stuck-sensor feature |
| Random path statistic obscured by the score formula | One-dimensional cut intervals linked to a path tree and terminal population | Compute changed first-cut probabilities; correct a reachable truncated leaf |
| LOF neighbors are themselves measured objects | Linked line/radius/floor/ration table using two unequal-density groups | Calculate query 4 versus query 17 and changed three-neighbor reaches |
| Same coordinate does not imply same ID/query contract | Frozen-reference/new-query toggle with explicit selected IDs | Explain training4/3 versus query7/8 without alleging a bug |
| Kernel boundary is not a connected input blob | Two similarity curves, weighted sum/rho and signed decision | Change anchor spacing and compensate gamma algebraically |
| Good rate but unaffordable alerts | Population counts and score threshold ruler | Calculate changed prevalence/budget result and undefined precision case |
| One event-hit metric hides repeated work | Real temperature/annotation/alert trace plus row/event counts | Compare matched denominators and propose a locked new experiment |

All interactive contracts are in the linked specification. Prediction begins unset; user-edited entities and parameters must be meaningfully editable, not just preset replay. Disclosures hide practice answers on the published page. No representation count is used as a completion target.

## Exact data and numerical fixture decisions

### Real data

Use pinned NAB industrial-machine temperature and combined event-window JSON, offline. Current pinned repository license is MIT; preserve its notice rather than repeating historical AGPL claims. [Dataset provenance](drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/dataset-provenance.md) and machine-readable download record identify commit, bytes and hashes.

The exact series has 22,695 raw rows, 22,683 timestamps and twelve duplicate excess rows. Aggregate duplicates explicitly by timestamp mean, retaining raw bytes. Use exact one-hour timestamp lookup for v(t)−v(t−1h), drop the twelve missing-lag rows, and do not infer regular spacing. Current-timestamp aggregation assumes those observations are available together; late duplicates need a deployment delay/revision policy.

Temperature units and timezone are unspecified. Four annotation windows are not pointwise fault truth or exact onset times. Label metrics accordingly. All learned methods get identical two-feature reference data/scaler; the level-only robust baseline's representation difference remains explicit.

Fit before 2013-12-06 (885 feature rows), calibrate through before 2013-12-10 (1,152), and inspect later 20,634 rows. Fixed settings and q=.95/.99 are declared, not selected to make a method win. A bounded author probe found four window hits for every method/quantile with sharply different outside-window row counts. This is a useful metric null paired with a workload contrast. Do not claim official NAB scoring, a prospective early-warning study, fault precision, significance, independent observations or timing results.

### Tiny checks already reasoned or executed

* Exact first-cut interval probabilities and Fraction-integrated depth-capped 1D forest: [0,1,2,3,12], regular endpoint variant and all-equal null.
* LOF exact-k neighbors/radii/reaches and library comparison for k=2,3,5; query 6 and17; separate query 4=35/24; actual same-coordinate/new-query discrepancy.
* Analytic two-anchor normalized OCSVM at gamma .1,.5,1,2 with fixed nu .5; midpoint sign change and exact anchor-zero null.
* Hypothetical prevalence counts and changed review-budget arithmetic.
* Reachable practice B was corrected during the parent's continuity read: original supplied depth2/leaf3/fitted4 state was impossible. Final [0,1,2,3] cuts .5 then1.5 leave [2,3], h=3 and score2^(-18/13)=.3829915893.

Author probe outputs are [author-calculations.json](drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/author-calculations.json) and [manuscript-calculations.json](drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/manuscript-calculations.json). They are content reasoning, not formal model/native/browser certification. Four compact manuscript programs were executed to substantiate shown arithmetic. The full real-data presentation program was initially syntax-checked with the existing fixed probe reused; it was subsequently executed once to supply actual row-level visual inputs and an unsolved q=.975 case. That separate purpose/output/source hash is retained in [visual-input-calculation.json](drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/visual-input-calculation.json). The old scope statements remain accurate for their timestamps rather than being retroactively relabeled. Phase two still owns independent and displayed-runtime verification and production contracts.

## Canonical-reference coverage check

Primary canonical map: Chandola, Banerjee and Kumar's 2009 survey, especially its problem taxonomy and family organization. Read taxonomy/organization and relevant passages, not every proof in 58 pages.

| Canonical area | Treatment and owner |
|---|---|
| Data nature, unit, point/contextual/collective anomaly | Core §1, temporal feature §10, changed taskJ |
| Labels, supervised/semi/unsupervised and score/label outputs | Outlier/novelty §2 and score/action §8; fully supervised anomaly classification is a supervised-learning continuation, not a fourth full algorithm here |
| Classification-based methods | Deep kernel support formulation and exact nu proof, §§6/9 |
| Nearest-neighbor methods | Deep LOF mechanism, identities, ties, bounds, §§4/5/11 |
| Clustering-based detection | DBSCAN noise distinction and frozen-reference scope; no duplication of clustering algorithms |
| Statistical methods | Robust baseline in §10; [next GMM note](topic-notes/gaussian-mixture-models-gmm-em-algorithm.md) owns responsibilities/density/threshold and EM |
| Information-theoretic and spectral families | Family coverage acknowledged here; full entropy/spectral/PCA mechanisms have separate curriculum owners. They are not prerequisites or claimed covered end-to-end by this three-method title |
| Contextual/collective and time-series challenges | Local causal feature/stuck sensor/event metrics/drift policy; advanced sequence detectors and streaming algorithms are further owners, not unsupported algorithm-name lists |
| Applications, validation and limitations | Real licensed chronology, changed budget/report tasks and representation-dependent applications |

Scope adds the missing observation/decision and evaluated-real-data layers, while retaining the three algorithm owners in the title. It does not claim every anomaly family or statistical guarantee is taught.

## Source and claim ledger

Primary sources were accessed on 12 September 2026 local time. No long verbatim passage is copied.

| Source | Material used | Actual review bound |
|---|---|---|
| [Chandola et al. survey](https://arindam.cs.illinois.edu/papers/09/anomaly.pdf) | Taxonomy, family/label/output distinctions | Organization §1.5, taxonomy §2 and relevant family passages; not full-proof review |
| [Liu et al. ICDM2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) | Trees, path correction, normalization, subsampling/masking | Construction and formulas §§2–4; experimental choices treated as empirical |
| [Breunig et al. SIGMOD2000](https://sigmodrecord.org/publications/sigmodRecord/0006/pdfs/LOF_%20Identifying%20Density-Based%20Local%20Outliers.pdf) | k-distance/ties, reach/lrd/LOF, duplicate assumption/local bounds | Definitions §§3–5.1 inspected; original paper's tie rule explicitly differs from fixed-k teaching rule |
| [Schölkopf et al. technical report](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-87.pdf) | Primal/dual, nu assumptions and proof | Formulation, Proposition4 and appendix proof inspected; no finite future-FPR inference |
| [scikit-learn guide](https://scikit-learn.org/stable/modules/outlier_detection.html) and linked three APIs | Versioned fitting/query/output, offsets, parameter contracts | Current 1.9.1 guide/API text and substantive LOF novelty example inspected; imprecise probability shorthand is qualified using original theorem |
| [Goix companion](https://ngoix.github.io/nicolas_goix_osi_presentation.pdf), [recording](https://webcast.in2p3.fr/video/anomaly_detection_algorithms_in_scikitlearn) | Optional alternate explanation of modes and isolation | All 15 slides' extracted substantive text read, author listing and working recording page checked. No full video watch, transcript, or slide-image inspection claimed. Clearly historical 2015 APIs |
| [Pinned NAB data README](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/README.md), labels, license, issue376 | Real series, event windows, reuse terms and duplicate issue | Exact bytes downloaded and hashed; duplicate counts calculated. No inferred missing units/timezone/point truth |

The source-selection goal is faithful explanation and authoritative contracts, not reproduction of another site's article structure. Main derivations use original tiny fixtures with visibly connected substitutions.

## Routed discoveries and neighboring ownership

The [GMM destination note](topic-notes/gaussian-mixture-models-gmm-em-algorithm.md) was saved centrally after scoped source inspection. Link it for the future author: relative responsibility can be decisive while total density is small; neither automatically determines a useful alert. Existing GMM likelihood/spherical/hard-EM/clipping concerns are recorded there, not patched in this task.

DBSCAN author confirmed the stable route and boundary: noise means no attachment under the selected geometry, membership strength is not anomaly probability, and there is no built-in predict contract. A short optional radius correspondence prevents confusing LOF neighbor-radius with OPTICS source-radius or HDBSCAN both-radius formulas; no optional prior branch becomes required.

No unrelated notes, GMM body, catalogue, runtime, shared manifest or blueprint index were edited.

## Author learning-experience check and phase-two handoff

Completed content checks:

* First encounter begins with an industrial reading and an action question, before terms/formulas.
* Each method has a distinct causal representation, exact concrete example and meaningful null; none is a reused generic matrix/grid.
* Every symbol needed for the first-pass calculations is introduced locally; optional optimization is visibly separated.
* Score/probability caution is centered in §2, nu in §6/9, and annotation caution beside the real report. Later practice uses those distinctions rather than adding repeated warning boxes.
* The false starting state in practice B was repaired with an explicit reachable path; changed tasks include arithmetic, debugging, classification of evidence and a worked report.
* Real data are openly reusable, supplied offline, audited for timestamps and explicitly bounded in their interpretation.
* Four small-program runs support their printed outcomes; the later real presentation-program run is recorded separately with its row-level input artifact. All five manuscript programs have been executed for these bounded content purposes, without labeling that formal independent/runtime verification.
* Setup appears before the first program. Resources explain what to read/watch and honestly delimit what the author inspected.
* Parent continuity suggestions incorporated: module-context links, actual file links, original title retained, radius-count correspondence and unsolved editable-threshold spec.

Deferred by explicit user mode: actual diagrams/labs, interface/backend models, complete formal executable example suite, independent mathematical review, geometry/perceptibility measurement, keyboard/screen-reader and 1440/390/320 browser checks, dependency/performance/build checks and production integration. No learner study or user acceptance is claimed.

Next implementation task: run the topic command with --work finish against root's saved content checkpoint; read the entire manuscript, specs, provenance, numerical inputs and this design. Retain the existing stable topic body destination under src/learn/data/topics. Proposed new owned support files are src/learn/data/anomaly-detection-models.js, src/learn/data/anomaly-detection-examples.js, src/learn/components/lesson-labs/AnomalyDetectionLabs.jsx and a matching anomaly-detection-labs.css; first inspect current ownership and component conventions before creating them. An individual stable-ID blueprint belongs in the existing data/curriculum/blueprints directory when centrally registered. Runtime datasets must remain topic-owned and lazily loaded; don't pull the real series into the global catalogue.

Retain the original archive and stable identity. Verify predictions, changed inputs, exact code/output, accessibility and real-series calculations before publication. If implementation reveals a better explanation or numerical boundary, correct content and refresh the revision rather than silently certifying changed bytes. The proposed filenames are a compatible handoff, not files created or registered in this content-only task.
