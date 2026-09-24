> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Semi-Supervised Learning — prepared content, design and evidence

Mode: research and write only. Revision1, 2026-09-12. Author classical_probabilistic_content. Content prepared for root checkpoint; implementation not started. This is an author record, not a claim of independent review or browser validation.

Packet: [manuscript](lesson.md), [visual contracts](visual-specifications.md), [data](banknote-subset.csv), [provenance](data-provenance.md), [source receipt](data-source.json), [author calculations](author-calculations.py), [checked results](checked-results.json), [categorical program](cotrain-categories.py), [graph program](graph-solutions.py), [banknote program](banknote-experiment.py).

## Identity, sequence, learner and scope

Stable ID semi-supervised-learning-label-propagation-self-training-co-training. Display title retained in substance: Semi-Supervised Learning: Label Propagation, Self-Training & Co-Training. Punctuation can follow catalogue style; no requested runtime title/ID change. The three named mechanisms remain substantial and distinct. Optional generative/low-density/deep extensions belong here as connections; they do not require expanding the title into a technique list.

Module order: Gaussian Processes → this topic → Active Learning. Intro connects covariance-based similarity to graph relationships; closing separates automatic pseudo-labels from querying an oracle. Local weighted-average, confidence, one-hot-vector, graph/matrix and evaluation-role refreshers avoid requiring the later Evaluation Metrics, Calibration/Conformal, PAC or Rademacher lessons. Full mixture fitting stays in the earlier EM topic; this lesson explains why a conditional pseudo-label loop is not automatically EM.

Observable outcomes: identify an unlabeled-data assumption; compute harmonic scores and a shortcut-induced reversal; distinguish hard clamping, symmetric normalization and soft evidence; trace pseudo-label origin and final refit; transfer a label through the recipient's representation; run a complete measured comparison that may reject SSL; select a practical diagnostic; construct a changed graph/point/view case and explain it.

Core route: introduction and §§1–7, then practice. §8 is an explicit optional branch. Rough editorial reading estimate50–65 minutes; programs, three focused investigations and changed practice can support90–150 minutes. These are not measured user times or quotas.

### Topic-specific scope decisions

| Idea | Decision / reason |
| --- | --- |
| Inductive versus transductive evaluation | Teach early because graph fitting can see inputs even without targets; this distinction affects validity of every later example |
| Harmonic voltage/random-walk correspondence | Develop as a second interpretation of the exact same graph, not unrelated trivia |
| Hard propagation versus soft spreading | Distinct full derivations, traces and program; correct the original α=0 confusion |
| Self-training confirmation bias | Transparent editable prototype feedback plus real logistic experiment where later accepted batches deteriorate |
| Co-training implementation | Complete categorical program and paired-view transfer, not a mislabeled generic ensemble; retain assumptions/theory context without unsupported universal sample bound |
| Fair empirical evaluation | Six fitting labels plus160 evaluation labels explicitly counted; shared label-free preprocessing acknowledged; keep measured failure |
| Semi-supervised generative/TSVM/entropy/manifold methods | Optional connection map with mechanisms/objective, not a claim to teach full solvers here |
| FixMatch/Noisy Student | Short verified weak/strong-target and teacher/student mechanisms; full deep training belongs with later architectures/optimization |
| Positive-unlabeled, missing-not-at-random label mechanisms and structured SSL | Related specialist settings, not implied by hiding labels in this binary training pool. Survey points to them; no new catalogue topic invented during this packet |
| Active labeling policy | Immediate next owner in this range; graph island/conflict motivates the next question without claiming uncertainty is always best |

No unhandled incoming own-topic note was found in topic inventory. No note to another topic was needed: the deferred items above have established broader owners, and this packet does not discover a currently orphaned mechanism requiring a new future instruction.

## Canonical section-list coverage map

Read the final2006 book's official contents, not only a generic search summary: Chapelle/Schölkopf/Zien, MIT Press, https://academic.oup.com/mit-press-scholarship-online/book/41571, contents lines724–835. The publisher metadata has some odd chapter numbering; use chapter names/families rather than treating that numbering as verified print pagination. An older104-page2005 draft was also inspected, but its incomplete/stub chapters are not represented as the final book.

Also read Zhu's2008 survey full contents and targeted FAQ/self-training paragraphs. This is an additional coverage map, not a claim to have read all60 pages.

| Canonical family / section list | Treatment here |
| --- | --- |
| Introduction: assumptions, inductive/transductive distinction | Core§1, same-input/different-target counterexample and split ledger |
| Generative models: taxonomy, EM text, degradation risk, constrained clustering | EM contrast§4; optional mechanism/risk§8; earlier EM lesson owns complete mixture fitting |
| Low-density methods: TSVM, SDP, null-category GP, entropy, data-dependent regularization | Basic TSVM and entropy mechanism§8; full SDP/null-category solvers excluded from introductory outcomes, canonical reference retained |
| Graph: quadratic criterion, geometric regularization, discrete regularization, harmonic mixing | Core§§2–3 exact graph/harmonic/spreading; manifold objective§8; variant-specific proofs remain reference depth |
| Representation: spectral graph kernels, dimensionality reduction, modified distances | Graph scaling/neighbor construction§2; kernel/representation relationship§8; earlier kernels/manifold lessons supply main machinery |
| Practice: large-scale algorithms, protein examples, benchmark analysis | Concrete measured banknotes§6, annotated biological candidate-use§7, dense/sparse dimension cost§7 |
| Perspectives: augmented PAC, metric methods, transduction discussion | Same-marginal ambiguity§1/8; upcoming PAC/generalization lessons own formal guarantees |
| Zhu survey's self/co-training and graph sections | All three named title mechanisms taught in depth with independent authored fixtures |
| Zhu survey's class proportions, structured output, related PU/active/representation/label-sampling topics | Imbalance/selection cautions§§4/7; explicit active next link and optional representation bridge; related specialist settings not falsely claimed mastered |
| Modern primary consistency/teacher-student additions | FixMatch equation and Noisy Student process§8, dated sources rather than current leaderboard claims |

## Original conservation and correctness corrections

Original source src/learn/data/topics/semi-supervised-learning-label-propagation-self-training-co-training.jsx, all924 lines read in contiguous ranges including the end's references/practice. SHA256 a52e8c886ea30fee8a3c4759ff49bb8bd1ab647c310b5d9b8fcba1ad22d4d961, baseline8c5da59f18516be77c29d5aeeafca3decca4f738.

| Original material | Conservation / change |
| --- | --- |
| History, Scudder/Blum/Zhu/Zhou/FixMatch/NoisyStudent and motivations | Keep relevant mechanisms and dated primary references; remove unsupported ranking, citation-count and universal-gain claims |
| Cluster/manifold intuition and three-family trace | Rebuild as explicit assumptions, same-input counterexample and three different investigations |
| Gaussian graph, Laplacian, spreading equation | Preserve and derive; S is symmetric adjacency, not stochastic P; unlabeled Y rows are zero |
| Claimed spreading objective/fixed point | Correct relative factor through α tr(Fᵀ(I−S)F)+(1−α)||F−Y||²; derive stationary equation |
| α=0/hard clamping, convergence comments | α0 is no propagation in this spreading equation; hard clamping separate; error factorα, not1−α; exact.99 contraction arithmetic |
| From-scratch synthetic graph/self/co-training programs | Preserve inspectable algorithmic learning value through checked graph solve, self-training loop, complete categorical transfer and real-data experiment; no unverified moon accuracy claims |
| Co-training shared pools/unresolved donations | Separate recipient arrays, synchronous offers, explicit conflicts, final rules and duplicate-view null |
| EM and theoretical guarantee statements | Conditional unlabeled likelihood sums to1; no automatic ascent; qualified co-training compatibility/conditional independence/noise learnability/weak usefulness |
| sklearn examples and “fully supervised upper bound” | Real measured protocol and named underlying baselines; fully labeled linear model is not a mathematical upper bound for a different nonlinear method |
| Invented heatmap, accuracy curves and timings | Remove; replace exact arithmetic and executed promotion audit. No unsupported interpolation or fixed method ranking |
| Cost analysis | n² dense storage with actual bytes; sparse indices, symmetrization, C columns and graph construction acknowledged |
| Failure cases and six exercises | Consolidate interpretation cautions, add changed graph/centroid/view/EM/budget questions with closed hint and reasoned solution; independent investigation has no fabricated expected numerical result |

Further corrected original pitfalls: co-training's conditional independence is not tested by running a model on the other view's coordinates; overall agreement/error correlation is not conditional independence; confidence thresholds do not certify correctness; class quotas are not a universal fix; augmentation/noise does not prove semantic understanding; final self-training model must include the last promotion; deep methods do not remove label-mismatch risks.

## Hurdle, representation and evidence map

| Hurdle | Mechanism | Representation | Learner evidence |
| --- | --- | --- | --- |
| Hidden labels mistaken for observed labels | Separate L/U/dev/test access and provenance | F1 | Budget/protocol repair exerciseF |
| Data density mistaken for target semantics | Same input distribution, different target rules | F2 | Assumption explanation and EM exerciseG |
| Repeated averaging looks magical | Four-node equations, shortcut, harmonic solve | F3/I1 | Changed weighted graph exerciseA and edited graph |
| Similar normalization names blur meanings | P versus S; raw F versus normalized readout | F4/I1 | Zero-evidence/null interpretation, exerciseB |
| More pseudo-labels assumed better | Prototype boundary feedback plus actual wrong-label batches | F5/I2/F7 | Changed point exerciseC and data-result exerciseE |
| Two models assumed independent | Recipient-feature transfer, conflict and duplicated views | F6/I3 | Broken bridge exerciseD, authored paired rows |
| Toy success mistaken for empirical evidence | Fixed licensed subset, controlled baseline and locked test | Actual tables/F7 | Development-only independent experimentH |
| Scale and advanced families become an unconnected list | Dimensional compute costs and changed objective/assumptions | Equations and practical comparison | Explain why the method can or cannot use the available structure |

## Claim and retrieval record

All retrievals2026-09-12. New explanations, numerical fixtures and prose; no copied quotations. “Read” below refers to actual text/PDF/slides inspected, not merely a returned search title.

| Primary source | Locator actually inspected | Use |
| --- | --- | --- |
| https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf | §2 harmonic equations/partition solution eq5; §3.1 random walk and electrical interpretation | Hard graph semantics; authored small graphs solved independently |
| https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf | §2 algorithm, zero Y, symmetric S and fixed point; §3 objective/gradient eq4/6; §4 setup | Spreading normalization/objective, distinction from P and hard anchors |
| https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf | Extended10-page version: introduction/§2 compatibility, bipartite setup; §5 weak usefulness and theorem1, including class-noise discussion | Qualified theoretical setting and two-view mechanism; no universal epsilon label bound |
| https://scikit-learn.org/stable/modules/semi_supervised.html | Main guide on self-training, label propagation/spreading, kernels, −1 labels and calibration recommendation | Library vocabulary; guide's α/hard-clamp language not conflated with our spreading equation |
| https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html | Parameters; fit/transduction_; predict inductive operation and predict_proba normalized readout | Real API protocol, versions verified by execution |
| https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_self_training_varying_threshold.html | Main example text and full displayed source, labels/calibratedSVC/manualCV/threshold sweep | Alternate worked code inspected, not executed; no imported threshold optimum or runtime claim |
| https://academic.oup.com/mit-press-scholarship-online/book/41571 | Official final2006 publication metadata/abstract and all contents families | Canonical coverage map; full-text book not read |
| https://www.cs.cmu.edu/~zhuxj/tmp/book.pdf | Contents of104-page2005 draft | Historical draft crosscheck only, not final-source coverage authority |
| https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf | VersionJuly19,2008, full contents, FAQ assumptions/failure, §3 self-training | Broad alternate route and family check, not modern completeness |
| https://www.cs.cmu.edu/~wcohen/10-605/notes/graph-ssl.pdf | §§1–3, graph/network/coupling examples, reset walk and optimization opening | Lecture-note alternative; different reset convention kept separate from our harmonic walk |
| https://www.cs.cmu.edu/~ninamf/courses/601sp15/lectures.shtml | Mar30semi-supervised entry links exact lecture19 slides/video | Verified association of resource topic |
| https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/19_ssl_03-30-2015.pdf | Slides21–32 co-training/assumptions and36–44 graph family/construction | Companion slides actually read, supporting video recommendation |
| https://www.youtube.com/watch?v=gnNLjX50F7U | Lecture19 destination metadata opened from official course page | Link/topic verified; video/audio not watched and transcript not inspected; no time-specific or spoken-claim assertion |
| https://arxiv.org/pdf/2001.07685 | FixMatch§§2.1–2.4, eq4, weak/strong augmentation and implementation factors | Optional mechanism only; not a reproduced training run or leaderboard claim |
| https://arxiv.org/pdf/1911.04252 | NoisyStudent§2, algorithm1 and teacher/student/noise description | Optional method connection; no universal gain asserted |
| https://arxiv.org/pdf/1804.09170 | Oliveretal§2P1–P6, §4validation discussion and§5 recommendations | Evaluation-label cost, underlying-model comparability and mismatch cautions |
| UCI source and download in data-provenance.md | Dataset metadata/license, actual archive member/data extraction | Four-feature real observations; label semantics not invented |

The CMU lecture video is offered because official course linkage and actual matching slides were verified. We do not claim to have watched it. The downloadable survey/notes and API references provide complete alternate text routes without requiring video access. The sklearn threshold example was inspected but not chosen as the primary empirical story; its dataset/optimum is not transplanted.

## Author calculations, reread and phase-two boundary

Executed author-calculations.py in the existing shared read-only Python3.12.14/NumPy2.3.5/SciPy1.18.1/sklearn1.9.1 environment. No package installation or environment mutation. It checks hard line/shortcut/separated systems, normalized row sums, soft direct solve versus200 iterations for α0/.2/.8, contraction arithmetic, unanchored zero forcing, real data candidates and final selected test, and complete centroid/co-training contrasting/null traces.

The corrected pseudo_wrong count excludes still-unlabeled rows:73 errors among252 accepted labels;62 remaining are not counted as incorrect pseudo-labels. This was an author-stage bookkeeping repair before packet freeze, and all printed manuscript counts agree with the corrected record.

Executed the exact complete graph and banknote Python blocks extracted from the manuscript into named topic-owned downloadable programs. The categorical program was executed directly. Graph results matched to printed precision; banknote development[72,63,61,59,65,59,72]/80 and selectedbaseline test72/80 matched; per-round accepted/wrong lists matched. Compared custom self-training's final development predictions with sklearn SelfTrainingClassifier at both thresholds; equal for this data/version. Our comparison uses≥ while sklearn's threshold boundary can differ; no data point lies on the threshold in these fixtures, and no general byte-for-byte equivalence claim is made.

Mathematically derived practice: weighted graph B1/5,C3/5; changed centroid boundary11/12; broken categorical bridge's unreachable rules; conditional marginal log1=0; dense matrix80GB. These are worked derivations, not fabricated benchmark outcomes. The optional deeper algorithms were researched and explained, not trained.

Learning-experience checklist: purpose and outcomes stated; core route before optional branches; local prerequisite refreshers; all title mechanisms have intermediate steps; static figures placed where concepts appear; three investigations match graph/feedback/paired-view structures; each has editable unsolved inputs, committed prediction, checked contrast/null, captions/text accessibility and bounds; real measured failure remains visible; programs include setup/run/results; eight changed practices have closed hints/solutions; next topic is the actual next module entry. Author final reread and link/fixture checks are recorded at completion.

Completion record: reread the entire final manuscript in three contiguous ranges and the complete visual specifications. Fixed two instructions that previously asked learners to edit after committing a prediction; edits now precede the new prediction. Made restoration of the original chain explicit before the B–C removal fixture, avoiding an accidental retained-shortcut contradiction. Qualified probabilistic-classifier wording, used approximate equality for rounded scores, clarified conditional independence as observing features, included the O(nC) score-update cost and allowed equal-capacity Noisy Students. The final graph/banknote program blocks equal their executed downloadable files; all manuscript local links resolve;16 hint/solution blocks are balanced and closed; pseudo-label totals agree across manuscript/results. No browser, build or independent-review checks were run.

Deferred phase two: production components, static code-native figures, lab model ports, formula rendering, downloads/relative-link integration, responsive and keyboard experience, browser performance, final independent technical/pedagogical review, build/runtime checks and implementation ledger checkpoint. Those checks are not represented as completed.

Retention: this packet's numerical data, named example programs, provenance and author checks support future implementation and are required, not disposable scratch. No own temporary research archives or screenshots remain. Do not delete another author's work or shared runtime.
