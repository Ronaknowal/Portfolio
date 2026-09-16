# Gaussian Mixture Models & EM — content design and handoff

**Visual-layout follow-up, 14 September 2026:** the user's screenshot exposed a collision between F2's left update and right-component heading. [The repair record](LESSON-VISUAL-LAYOUT-REVIEW.md) replaces that fixed text stack with wrapping HTML stages and equal-scale before/after strips that stack on phones. All allocation shares and fitted statistics are preserved. Mean-guide and covariance-label clearance were also repaired. A final independent review caught coincident lane names in the EM identical-components preset; a wrapping color/dash key now carries both names outside the plot. The current [browser verifier](evidence/gmm-browser.json) passes 14 cases, including five allocation widths, enlarged text, and all four EM presets before/after a full cycle at desktop/320px. Actual screenshots were inspected. This presentation repair leaves the prepared manuscript/specifications, examples and model unchanged and supersedes earlier selected-image claims for the affected visuals.

Updated 12 September 2026. Canonical ID `gaussian-mixture-models-gmm-em-algorithm`; Classical ML position 16 of the declared module order. Title retained: **Gaussian Mixture Models (GMM) & EM Algorithm**. No broader title is needed for conditional prediction and sampling applications of the same model.

## Current scope and phase

The user requested the next three lessons in **content-first** mode with parallel authors. This author owns only the GMM draft packet, this record and the GMM destination note. Root owns the delivery ledger, current handoff, shared records and neighboring notes. Complete stages1–2: full learner manuscript, actionable visual/investigation specifications, real offline data, numerical author calculations, practice with hints/solutions and annotated sources. No runtime, lesson JSX, blueprint, registry, navigation or publication changes are authorized in this phase.

* **Content:** complete after the author checks recorded below; checkpoint final hashes are collected by root.
* **Implementation:** complete, 13 September 2026; see the phase-two section at the end of this record.
* **Computational verification:** bounded author calculations passed; this is not the phase-two full displayed-program/native/model campaign.
* **Browser/visual/accessibility review:** run against a production build at 1366, 390 and 320 px; evidence in `docs/teaching/evidence/gmm-browser.json` with screenshots.
* **Independent phase-two review:** complete; see [GMM-INDEPENDENT-REVIEW.md](GMM-INDEPENDENT-REVIEW.md) and the disposition at the end of this record. During the content phase root had performed a separate content-sequence reconciliation read, returning four focused writing findings resolved below.
* **User acceptance:** not assessed.
* **Next action:** root registers this exact packet as the content checkpoint. A later authorized finish request runs `node scripts/build-curriculum-inventory.mjs --topic gaussian-mixture-models-gmm-em-algorithm --work finish`, reads the complete packet and performs stages3–6.

Authoring inputs read: nested `AGENTS.md`, `LESSON-AUTHORING-HANDOFF.md`, complete current `LESSON-TEACHING-STANDARD.md`, ML-specific `DOMAIN-PLAYBOOK.md`, `TOPIC-DESIGN-BRIEF.md`, `LEARNING-CODE-STANDARD.md`, topic-note policy and the topic inventory with `--work content`. The returned GMM destination note and resolved UNASSIGNED inbox were read. Relevant predecessor/prerequisite sources were inspected for Bayes' rule/density, likelihood, weighted covariance/PCA and the existing anomaly/evaluation bridge. Historical superpowers plans were not used as teaching instructions.

## Exact original-source baseline and conservation

The entire original872-line `src/learn/data/topics/gaussian-mixture-models-gmm-em-algorithm.jsx` was read before writing. It is recoverable at git commit **8c5da59f18516be77c29d5aeeafca3decca4f738**; original file SHA256 **13848098891a40626de8bcadf0bb3dffd3a627e22d0d54cd2afd3d160bc3e895**. Initial worktree was clean; later parallel changes belong to their other authors. No original copy is needed because the git object preserves it. The original production lesson was not edited.

| Original useful coverage / weakness | Current treatment |
| --- | --- |
| Generative hidden-selector intuition, Gaussian density and responsibility equations | Retained, with density/probability distinction, one observation held fixed in every denominator term and a complete numerical example in §§2–3 |
| Full EM weighted parameter formulas | Retained in §§3 and5, with weighted second-moment calculation, new-mean covariance and maximum-likelihood denominator explained |
| Large NumPy2D example and claimed verbatim outputs | Replaced by transparent4-row1D EM code and a real2D library fit. The vector covariance update and Cholesky evaluation mechanism remain explicit; historical output is not reused as current evidence. This reduces display overhead while retaining the algorithm's full-covariance reasoning |
| ELBO derivation, general EM and convergence | Retained in deeper§9; Q and Q+entropy separated, old touching equality explicit, exact nondecrease separated from boundedness/parameter convergence/stationarity/global optimum |
| Covariance types, parameter counts, model selection | Retained and expanded with exact matrix/ellipse contracts and a real selection disagreement; library spherical permits a variance per component |
| BayesianGaussianMixture branch | Retained with both prior families/truncation cap and actual prior/active-threshold sensitivity; no automatic true-K claim |
| Anomaly scoring, sampling and alternatives | Retained; score/action bridge developed with a worked hypothetical device example, sampling/total variance derived, conditional GMM prediction added as useful optional transfer |
| Complexity, memory, initialization and label switching | Retained with factorization cost, decimal-byte versus GiB distinction, accurate API names and explicit label-invariance reasoning |
| Historical opener and claimed83-year unsolved gap/citation counts | Removed unsupported historical story and moving citation counts. Short supported note on1977 generalization and prior special cases remains in§9. Original bibliography is recoverable; no priority claim is needed to teach the model |
| Heatmap row[0,0.35,1] sums1.35; arbitrary “450-point” plotted subsets and traces | Replaced by source-derived exact row-normalized responsibilities, consistent data IDs and declared analytic/observed provenance |
| Unbounded likelihood called bounded; strict growth except local maximum | Repaired with explicit covariance-collapse counterexample, symmetric stationary null, a declared positive variance-floor objective and qualified theorem discussion |
| Underflow denominator clipping claimed uniform | Rejected: it can produce all-zero rows. Stable log-sum-exp worked case and concise code replace it |
| k-means equated with spherical+0.5 threshold | Rejected; shared isotropic covariance/equal fixed weights/hard training updates and the small-variance limit are treated separately |
| Categorical Naive Bayes called diagonal Gaussian mixture; PPCA/VQ-VAE conflated with Gaussian mixtures | Removed incorrect equivalences. Scoped source-type note is communicated to root; the manuscript gives only accurate local/next-topic connections |
| Arbitrary d<20/30/50 cutoffs, universal50–200-iteration/rate/speed claims | Removed unsupported fixed thresholds/benchmarks, retaining operation counts and conditional practical reasoning |

## Outcomes, sequence and prerequisite continuity

Reader progression stays Anomaly15 → GMM16 → t-SNE/UMAP17 → ICA18 → NMF19. No reordering or prerequisite injection is proposed. The manuscript's next-topic link preserves `?module=classical-ml`; foundational review links use the actual `math-foundations` module. It introduces a new dataset explicitly rather than reusing an unexplained row identity.

| Learner outcome / hurdle | Local bridge and explanation | Complete evidence opportunity | Representation |
| --- | --- | --- | --- |
| Distinguish component choice, density and alert | Hidden selector; probability is area; Bayes reverses the selector; responsibilities compare components | x0/2/8 table, unequal weight midpoint variation, device-score application and practice1 | F1 generative branching plus I1 linked heights/sum/allocation |
| Compute and interpret one EM cycle | Effective count, weighted moments, new-mean scatter; fixed data versus inferred membership | Four-row fully worked E/M; changed D=3; standalone small program; practice2 | F2 fractional mass lanes and I2 half-step geometric fit |
| State the actual optimization guarantee | Declare unconstrained failure; derive scalar variance-floor maximizer; distinguish objectives | Collapse sequence, unchanged identical-components null, practice3/6 | F3 spike/local zoom; exact F6 bound chain |
| Read covariance geometry and model restrictions | Squared standardized displacement, determinant/volume and PCA eigenvectors | Equal-distance contrast, null/reversal, parameter counts and changed practice4 | I3 covariance ellipse/eigenprojection; F4 restrictions gallery |
| Fit and assess a real density model without target leakage | Label-blind split, training transform, K1 baseline, declared validation rule | Offline150-row Iris program,16 candidates, frozen test result, ARI and reduced-grid practice5 | F5 actual score comparison and narrow-variance diagnosis |
| Explain deeper connections precisely | Shared variance/hard versus soft training; conditional Gaussian refresher; model-prior distinction | k-means limit, conditional mixture mean/variance and changed practice7/8, Bayesian weights | Compact equations and exact tables; no extra lab merely to increase count |

First-pass route is explicit immediately after the introduction: §§1–8 with three investigations and two programs, core practices1–5. §9 proof and §10 connections say they are deeper branches; practices6–8 and their readiness criteria belong there. Reading/practice estimates are separated and not a length target. The text supplies a local refresher at each new mathematical step, rather than relying on linked lesson titles as proof of readiness.

## Canonical-reference coverage check and topic ownership

Canonical treatment: Bishop, *Pattern Recognition and Machine Learning*, chapter9, using the legitimately free Microsoft-hosted complete PDF. The chapter's section list was read:9.1 k-means/image segmentation;9.2 Gaussian mixtures/maximum likelihood/EM;9.3 alternate latent-variable view/Gaussians/k-means/Bernoulli/Bayesian linear regression;9.4 general EM. The next chapter's variational-mixture headings were checked as the boundary of the optional Bayesian introduction.

| Canonical idea / discovered connection | Decision and reason |
| --- | --- |
| Singular likelihood, label permutations and local optimization | Core. These are headline facts required to repair the former guarantee and interpret fits |
| Exact latent-variable/complete-data view and lower bound | Full deeper derivation retained; essential nondecrease conditions remain visible in core§4 |
| Precise relation to k-means | Deeper§10 with exact finite-soft versus explicit-hard versus limiting distinction; inaccurate API shortcut removed |
| Image segmentation | No additional application: earlier k-means already owns pixel grouping and this lesson's real-data/scoring/conditional applications have more distinct teaching value |
| Bernoulli mixtures and Bayesian linear-regression EM | No full derivation here; they require other observation models. The core explains general EM without equating categorical Naive Bayes with a Gaussian density |
| Generalized EM, MAP/variational bound distinction, incremental sufficient statistics | Brief deeper/generalization connection, retaining objective conditions. Full approximate-inference derivation belongs to `variational-inference`; no new external code dependency or invented performance ranking |
| Conditional mixture prediction and total variance | Included here because the same selector/density mechanism directly explains both; complete small cases teach transfer beyond clustering |
| GMM versus ICA/NMF meanings of hidden component | Short sequence bridge. Root saved the destination-specific [NMF note](topic-notes/non-negative-matrix-factorization-nmf.md) about additive latent factors, nonuniqueness and objective monotonicity. GMM selects one component per observation; ICA/NMF combine contributions. No next-topic implementation performed |

No remaining scope/title expansion is needed. Source breadth was increased where the original contained concrete inaccuracies; unsupported catalogues of applications and arbitrary thresholds were removed rather than adding more caution paragraphs.

## Research and claim ledger

All research dates below are12September2026. Documentation identified itself as scikit-learn1.9.1; actual numerical environment matched1.9.1. Sources were assessed for their stated purpose, not blanket-endorsed. Formula derivations and all teaching prose are original; no long source passages were copied into the packet.

| Claim / source | What was actually reviewed and resolved | Scope / limit |
| --- | --- | --- |
| [Bishop PRML official full PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), chapter9 | Contents plus visible singularity discussion on printed p.434, general-bound/GEM discussion on pp.453–455 and relevant exercises on pp.456–457. Some other retrieved page output was truncated and is not claimed as fully reviewed; the complete Gaussian update walkthrough was read in the Stanford notes. The browser rejected the18MB file; read-only Python retrieval with `pypdf` supplied text. Output encoding/truncation was diagnosed; only actually accessible passages are claimed | Used for canonical coverage, collapse, weighted updates, k-means connection, Q/ELBO and extension ownership. Detailed conditional prediction formulas independently checked from Gaussian conditioning and the exercise prompt; full book not claimed read |
| [Stanford Ng Gaussian-mixture notes](https://cs229.stanford.edu/notes_archive/cs229-notes7b.pdf), pp1–4 | All four pages: hidden selector, fixed-observation Bayes denominator, weighted parameter updates, initialization sensitivity | Compact alternate after hand example; informal “guess” wording is refined to conditional expectations in our text |
| [Ma/Ng EM notes](https://cs229.stanford.edu/notes-spring2019/cs229-notes8.pdf), Jensen and EM sections | The concavity construction, equality at posterior q and full nondecrease chain, approximatelypp1–8 | Confirms bound reasoning; not used to infer bounded unconstrained likelihood or a general parameter-convergence theorem |
| [Dempster/Laird/Rubin1977 DOI](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x) | Publisher bibliographic record and abstract describe the general incomplete-data algorithm and applications | Full publisher paper unavailable; exact theorem support is through inspected canonical derivations. No citation-count or first-mixture-fit priority claims |
| [Wu1983 DOI](https://doi.org/10.1214/aos/1176346060) and indexed original abstract | Abstract explicitly distinguishes stationary limit points and whole-sequence convergence, giving separate conditional results | Full-text attempts failed (publisher response/mirror certificate mismatch); do not claim to have read theorem proofs. Text uses the conservative distinction and our own bounded scalar proof, not an unchecked theorem number |
| [GaussianMixture API](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) | Covariance shapes, separate spherical variances, `init_params`, `n_init`, `tol`, `max_iter`, `reg_covar`, `score`, `score_samples`, `predict`, `predict_proba`, `converged_` | `predict_proba` API wording calls its output density; normalized responsibility behavior was independently checked in actual1.9.1 fitting and via formulas. Guide's broad convergence phrasing is qualified by canonical analysis |
| [Mixture user guide](https://scikit-learn.org/stable/modules/mixture.html) | Singularity, initialization, covariance families, model-selection and variational sections | Source for implementation overview; no unqualified “always local optimum” or “automatic true component count” claim imported |
| [BayesianGaussianMixture API](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html) | Prior family, concentration, finite cap, output weights and variational lower-bound contract | Supplementary fit uses concentrations0.01/1/10 and declared active-weight thresholds. No effective-K truth claim |
| [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), [load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) | Dataset question/context,150×4 measurements, cm units, balanced species, historical correction note, CC BY4.0. Exported actual scikit-learn data and explicitly inspected corrected rows35/38 | Exact provided snapshot differs from historical UCI text file; no broad ecological sampling claim. `data-provenance.md` supplies attribution and transform details |
| [Stanford Lecture12 video page](https://see.stanford.edu/Course/CS229/43), [transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture12.html) | Video page/bookmarks, transcript's density→latent labels→soft weighted updates passage (roughly transcriptparagraphs43–80), companion Gaussian notes; substantive content was reviewed | Video not watched. Transcript has speech-recognition errors; notes supply exact formulas. Direct lecture URL was verified; initial guessed `/39` was wrong and corrected to`/43`. No video-only technical claims or unverified timestamps published |

Moving URLs remain links to checked versions as of this date, not promises of permanent current behavior. The selected alternatives are annotated in the manuscript with fit, level, prerequisites/access and actual review mode. No need to collect more links once the actual questions were resolved.

## Real data and author calculation evidence

Packet: `docs/teaching/drafts/gaussian-mixture-models-gmm-em-algorithm/` contains `lesson.md`, `visual-specifications.md`, `iris.csv`, `data-provenance.md`, `author-calculations.py`, `checked-results.json`. These are necessary pending handoff inputs. No duplicate source archive, per-file report or checkpoint-manifest variant is retained.

The shared existing runtime `scratch/lesson-tools/Scripts/python.exe` was reused without modification or recursive scratch scanning. Exact environment: Python3.12.14, NumPy2.3.5, SciPy1.18.1, scikit-learn1.9.1. Author calculations performed:

* Analytic four-row E-step, first weighted mean/variance,20-step constrained trace and independently checked first-step mean via hyperbolic-tangent identity. Row sums and nondecrease checked on the ordinary, identical-start, changed-point, asymmetric-initialization and active-floor fixtures.
* Base density/responsibility contrast, unequal-prior midpoint reversal, variance-change contrast and identical-component null; log-sum-exp underflow example.
* Mahalanobis geometry at rho±0.75/0 plus center case, using SciPy multivariate-normal density as an independent numerical reference. Exact inverse/eigenvector reasoning is in the text. The changed practice uses rho±0.5 and points(2,±1); formula checks are recorded in supplemental results.
* Covariance-collapse path with a fixed broad component and shrinking standard deviation; all stated finite values recomputed.
* Sixteen fixed Iris fits: K1–4×four covariance types, n_init5, seed16, reg_covar1e−4, tol1e−6, max_iter500. Every candidate converged. Training/validation/test IDs and transform values are retained. Validation picked fullK2, training BIC fullK4; fullK4's thin component has width variance1e−4. Selected test mean log-density−3.011223 versus baseline−2.887249; test ARI0.433446. Retained the unfavorable test result rather than changing seed/split/selection rule.
* Three Bayesian-prior sensitivity fits on the same training rows; all converged; active counts depend on both concentration and a declared threshold.
* Supplementary calculation of exact Q/entropy/bound chain, small-variance probabilities, changed-practice answers, BIC/AIC example, conditional-mixture moments and ellipse probability. Completed grid evidence was reused rather than rerunning it for these additions.

Reproduction: from the repository, run `& 'scratch/lesson-tools/Scripts/python.exe' 'docs/teaching/drafts/gaussian-mixture-models-gmm-em-algorithm/author-calculations.py'`. The author initially ran the main calculation, then `--supplement` to extend necessary proof/practice quantities while reusing the grid. The finalized default command produces both sections. This script is an author arithmetic input; phase two implements independent browser model checks and verifies displayed teaching programs in their final form. It does not represent a browser or production test.

## Author learning-experience checklist and final disposition

This is an **author heuristic assessment of complete written content and specifications**, not an observed beginner session or implemented visual review.

1. **Route:** core/deeper route after the introduction, separate time estimates, deeper section labels. Root noticed readiness had accidentally required the skipped proof; moved proof repair explicitly into deeper readiness.
2. **Cautions:** density/probability/interpretation lives in§2; exact/constrained/unconstrained optimization in§4; model-selection assumptions in§6; dataset sampling in§7. Removed repeated convergence warnings from every example. The device threshold references the established score/decision boundary. Programs print results, not disclaimers.
3. **Real question:** introduction asks whether a mixture improves unseen sepal-density prediction overK1. Actual held-out result returns to that question and retains the baseline's better test performance. The dataset's balanced historical sampling is stated and prevents a population-prevalence interpretation.
4. **Investigations:** I1 edits x,weight,variance; I2 edits actual observations and initial means; I3 edits correlation and point coordinates. Each saves an initially unset prediction against exact inputs and compares it with computed output. Family contrast/null fixtures were numerically checked; final supported input-boundary/browser behavior stays deferred.
5. **Figures:** the specifications explicitly repair three known perceptibility risks: tiny tail density gets a log readout, collapsing spikes get a local zoom with explicit scale, and training-score comparisons includeK1 and actual fitted narrow-variance geometry. Desktop/phone perceptibility has not been visually verified because implementation is deferred.
6. **Connections:** the later row34 responsibility example names its link to§2; scalar weighted variance and full covariance share a derivation; covariance eigenvectors connect to PCA; boundQ+entropy is tied to the earlier EM cycle; core versus deeper k-means restrictions are explicit. Canonical headline omissions are resolved in the coverage table above.
7. **Code:** standalone small EM exposes log-sum-exp and weighted statistics directly. Real-data code supplies CSV reader, split, training-only scaling, baseline, selection and result interpretation. A compact pinned install command was added after root's setup finding. No expansive guard framework obscures the update.
8. **Practice:** eight tasks cover altered weights, a genuinely changed M-step, numerical diagnosis, changed covariance geometry, reduced-grid real data, proof repair, changed small-variance example and a conditional-prediction variation. Root caught practice7 copying the worked x1; changed it to x0.5 and verified answers. Practice4 now changes both point pair and correlation magnitude. Practice5 gives exact reproducible output after a declared independent variation.
9. **Screenshots:** not captured or claimed. Specs list informative states, including wrong prediction feedback, symmetric null, active variance floor, correlation reversal, spike zoom, full desktop score figure and bound equality. Phase two must inspect these, not only initial screens.

An inline-reading pass placed F1 at hidden selection, F2 at fractional weighted fitting, F3 at collapse, I3's initial ellipse beside the covariance mechanism, F4 at model restrictions, F5 at real selection and F6 beside the proof. It identified the risk of plain-parenthesis inline LaTeX; inline mathematical expressions were normalized to explicit delimiters during final authoring. Root then caught a syntactically valid but wrongly segmented p(x) table heading; a focused surrounding-prose/function-call pass repaired that heading, remaining shape/count expressions and operator notation. Syntax parsing alone was not treated as proof of correct surrounding prose. New diagrams are specified rather than implemented, with all labels, formulas, states and verification expectations provided.

Final author format/input checks: all 232 final inline/display expressions parsed with the installed KaTeX engine; all three Python code blocks compiled; corrected Iris rows35/38, the CSV SHA256 and150 disjoint split IDs matched the retained evidence. These are manuscript/input checks, not rendered-figure or full native-output verification. The floor-bound clarification and the repaired surrounding function/shape notation received the final232-expression check; all seven internal topic links resolve to existing publication identities.

The destination note is resolved for the content phase, with implementation explicitly pending. Root receives final hashes and owns the phase-ledger update. No runtime build, browser pass, publication or deployment was run or inferred from content completion.

---

# Phase two: implementation, 13 September 2026

The content packet above was implemented without changing its claims. Where the published page differs from the manuscript, the difference is recorded here.

## What was built

| File | What it owns |
|---|---|
| `src/learn/data/gmm-models.js` | The mathematics: `logNormal`, `mixtureAt` (log-sum-exp), `expectation`, the constrained `maximization`, `emCycle` and `emTrace`, `boundDecomposition` into Q and entropy, `collapseLogLikelihood`, the plane geometry with `symmetricEigenpairs`, `parameterCount`, `criteria`, `sphericalResponsibility`, `conditionalMixture` and `mixtureMoments` |
| `src/learn/data/gmm-iris-data.js` | The generated real data: 150 rows with their split and species, the frozen scaler, all 16 candidates, the selected model's parameters, the 30 reserved rows with their densities and responsibilities, the narrow component and the three Dirichlet-process fits |
| `src/learn/data/gmm-examples.js` | The three displayed programs with their executed output |
| `src/learn/components/lesson-labs/GmmShared.jsx` | `Investigation`, `Field`, `NumberField`, `Table`, `useInvestigation`, `Prediction`, `Plot` |
| `src/learn/components/lesson-labs/GmmLabs.jsx` | The three investigations |
| `src/learn/components/lesson-labs/GmmFigures.jsx` | The six inline figures |
| `src/learn/data/curriculum/blueprints/gaussian-mixture-models-gmm-em-algorithm.js` | The authoring blueprint, registered centrally by title |
| `public/learn-assets/gmm/iris.csv` | The same 150 rows the program reads, served for download |

## The interaction contract the specification asked for

The three investigations do not recompute as the learner types. Each keeps a draft and an active state: editing any field clears the result and returns the prediction to unset with a polite status, and one action commits the draft and records the prediction against it. The answer is computed from the committed draft rather than from whatever was on screen beforehand, so a prediction can never be graded against inputs the learner has already changed. A second action calculates without recording a prediction, for a learner who wants to look first.

The EM investigation adds an explicit half-step machine: apply the setup, record the prediction, compute the E-step, apply the M-step, or step back. The responsibility matrix the M-step used is kept and labelled as such, rather than being overwritten by the fresh one computed afterwards, and an effective count below 1e-12 stops with a specific message instead of silently reseeding a component.

## Departures from the manuscript

* Sixteen displayed formulas were re-set to fit a 320 px column without a horizontal scrollbar, nine of them by splitting across lines and five by naming a subexpression: the residual `d_i` in the covariance M-step, `a_ik` and `1_ik` in the complete-data log-likelihood, `p_ik` in the Jensen step, and `B_k` for the conditional regression block. Five inline formulas became displayed blocks or plain prose for the same reason. No claim changed.
* The manuscript's investigation I1 asked for an optional two-point synthesis. It is implemented as a saved pair on a frozen model, and the target it checks is stated on the page: both responsibilities above 0.99 and negative log-densities at least 5 apart. Changing the model clears the pair.
* The Bayesian weight block is presented as a third runnable program rather than a bare appendix, executed in the Iris program's namespace exactly as the lesson instructs.
* `symmetricEigenpairs` lives in the model layer rather than the figure, because a drawn contour is a mathematical claim and the verifier now rebuilds each matrix from its own eigendecomposition.
* Two manuscript tables moved into figures, because each was an argument about a picture: section 3's four-row responsibility table sits inside the allocation figure, and section 7's sixteen-candidate table inside the candidate figure, in a scroll-capped region with the selected row marked.
* Section 12 gained a readiness-check table, matching the neighbouring lessons.
* Section 6's AIC and BIC example, section 10's small-variance table and section 10's Dirichlet-process weights are computed or read from the verified model and the generated data rather than retyped, so a change in either cannot leave the prose behind.
* `RunnableExample`, a component every lesson shares, now prints the filename each program asks to be saved as. The anomaly lesson's review had raised its absence.
* The manuscript's section 9 sentence "Zero terms are handled by their limiting values where the support permits" is not on the page; the bound figure and the surrounding derivation carry the same restriction implicitly, and the sentence did not survive the split of that formula across lines.
* Three specification items are not implemented as drawn: F4's per-cell links from matrix entries to semiaxis lengths (the matrices are printed beside each panel instead), F3's common-axis inset of the broad component (a shared window carries the contrast instead), and F5's third panel shows the fitted contour rather than a full covariance overlay in both panels.

## Verification

| Check | Command | Result |
|---|---|---|
| Browser models against the content-phase native probes, the manuscript fixtures and the generated module | `node scripts/verify-gmm-models.mjs` | 80 grouped checks |
| The three displayed programs executed | `scratch/lesson-tools/Scripts/python.exe scripts/verify-gmm-examples.py` | 3 programs, 18 oracle assertions |
| Every candidate refitted from the supplied CSV and matched against the packet | `scratch/lesson-tools/Scripts/python.exe scripts/verify-gmm-iris-data.py` | 150 rows, 16 candidates |
| The production build in Edge, at 1366, 390 and 320 px | `node scripts/verify-gmm-browser.cjs` | 10 cases |

Evidence: `docs/teaching/evidence/gmm-models.json`, `gmm-native.json`, `gmm-iris-data.json`, `gmm-browser.json`, and the screenshots under `docs/teaching/evidence/screenshots/gmm-*.png`.

Independent checks worth naming, because they are the ones that would catch a plausible-looking error: each Gaussian is integrated numerically to 1, every eigenpair rebuilds its own matrix, the constrained variance maximizer is confirmed by scanning the objective, twelve consecutive cycles from three different starts never decrease the objective, the bound stays below the objective for allocations that are not the posterior, and the frozen scaler is recomputed from the 90 training rows rather than trusted.

## Visual repairs found by inspecting the screenshots

Every assertion passed while four figures were still wrong. The covariance gallery drew the diagonal and spherical families as straight lines, because the eigenvector helper returned the same direction twice whenever the cross term was zero; that was a real mathematical defect in a figure, and it now lives in the model layer with a verifier check. The gallery also used a vertical scale half the horizontal one, so every ellipse leaned wrongly. The selector figure clipped its own legend and axis labels, the allocation figure printed its caption on top of the mass columns, and the bound chain overlapped three labels. The panel grid was widened so that panel text is not scaled down to eight pixels, and the narrow-component band is drawn at its true height rather than padded to be visible.

## What is still not claimed

No learner study, no user acceptance, and no claim that a mixture is the better model for these flowers. The real-data result is one fixed split of one curated balanced collection under one rule declared in advance, and on the reserved rows the single-Gaussian baseline scored 0.123975 nats per row higher than the selected mixture. That outcome is reported as it came out.

## Disposition of the independent review

The [independent review](GMM-INDEPENDENT-REVIEW.md) recomputed every stated number from first principles, at fifty digits where it mattered, importing nothing from this lesson's own code, and refitted the whole Iris pipeline from the served CSV. It found **no numerical disagreement anywhere**, and confirmed the served bytes against `load_iris`, UCI `iris.data` and `bezdekIris.data`. It raised two blocking findings, fourteen to fix and twenty-one observations.

| Finding | What was wrong | Resolution |
|---|---|---|
| B1 | The handoff and `AGENTS.md` claimed a closed ledger and a review that did not yet exist, and this record's header still said implementation had not started | The review exists, the header is corrected above, and the ledger's implementation phase is closed below with final hashes |
| B2 | A sentence printed twice mid-derivation, and all three programs rendered "Before running:" twice | The duplicate is gone; the prefix now lives only in the `Program` wrapper, and a browser case counts it |
| S1 | Practice 1 sent the learner to a lab that had no means control and drew no tie | The responsibility lab takes both means and marks the tie location on its axis, computed from the crossing of the two weighted densities |
| S2 | The version caveat was dropped from section 7 | Restored beside the candidate figure, naming full K = 4 as the delicate fit |
| S3 | The intro promised more than the page keeps | Reworded to what it does: a prediction is asked for before an answer, and retired when an input changes |
| S4 | The EM lab kept a selection across a graded cycle and left a verdict for an undone one | A graded cycle clears the choice and the reason; stepping back retires the verdict and re-enables recording |
| S5 | The collapse panels were drawn in three windows and looked identical, and two required elements were missing | One shared window now carries the width contrast, and the objective strip is drawn against a logarithmic sigma |
| S6 | The covariance gallery showed no matrix to a sighted reader | Each panel prints its matrices, with the spherical scalars as badges and tied marked as shared |
| S7 | The covariance plots never reached the specification's type size | The panels stack full width and their viewBox matches the rendered width, so the labels land near 14 px and the point IDs are legible on a phone |
| S8 | The lab computed the opening allocation, the old parameters and the unclipped scatter, then discarded them | All three are shown: the opening responsibilities, a before-and-after parameter table, and the scatter the floor acts on |
| S9 | Observation IDs were struck through by the mean stems and collided in the repeated preset | Duplicate IDs share one label, and the stems start above the axis and are named |
| S10 | The row 35 and 38 corrections were cited to a page documenting neither | Attributed to the UCI record, which states both rows verbatim |
| S11 | The departures list was incomplete | Extended above |
| S12 | The objective history printed its labels outside the frame | Both labels are placed from the plot's own scale, beside their points |
| S13 | The narrow component was drawn as a line, and the in-strip rows were unnamed | Its one-deviation contour is drawn in centimetres, and the accessible description names the rows inside the strip |
| S14 | No investigation offered the explanation field the contract asks for twice | An optional, never-graded reason field is committed with the prediction and echoed beside the verdict |

Of the observations, six were acted on: table columns keep their decimal places, so the selected candidate reads −2.668800 as the prose quotes it (O1); the Dirichlet-process weights are rendered as the manuscript's table (O3); the figure comment naming the wrong section is corrected (O6); a rounded negative zero is published as zero (O7); the stale hedge about Wu's availability is reworded (O8); the split seed is stated in prose (O11); the saved two-point pair is discarded when the model changes rather than filtered (O15); the intro no longer says 150 rows are fitted (O16); component identity carries a dash pattern and a fill pattern as well as a hue (O13); the allocation bar shows percentages (O14); and the dead exports the page never used are removed (O2).

The rest are recorded and left: the "Before running" question still precedes the program's own heading (O5), the sixteen-row candidate table is still scroll-capped (O17), the scikit-learn citations still pin a version against a moving URL (O18), control ranges are stated in the responsibility lab and as validation messages elsewhere (O19), the ARI rests on six setosa rows (O10), and the baseline carries the same regularization as every other candidate (O12), which is what makes it a fair comparison rather than a different model.

## Prepared-content implementation audit, 14 September 2026

The user requested verification of PCA through GMM, rather than a new lesson revision. The coordinator compared the complete saved GMM manuscript/specifications with the production body, all figures/investigations, examples and data; an independent reviewer checked the corrected source, mathematics and learning experience. See [the substantive comparison](GMM-CONTENT-IMPLEMENTATION-AUDIT.md) and [five-topic integration record](PCA-THROUGH-GMM-IMPLEMENTATION-AUDIT.md).

This pass restored the explicit distance-softmax and conditional mixing-weight formulas, the hand M-step weighted numerator/second moment and the Jensen support qualification. It added no prerequisite barrier to the first-pass route: these details remain next to their existing worked examples or deeper branches. The saved phase-one packet is retained unchanged as the input against which omissions were identified. The implementation now supplies the missing steps; it was not reduced to match an incomplete UI.

The allocation lab now distinguishes finite crossings, no crossing and equal allocations everywhere; it offers both quadratic roots and lets the learner use the exact changed-practice tie. The log-density polynomial is tested independently. Density peaks use the plotted grid as well as the query value. Covariance results do not leak before a prediction, and the selected point's projections onto the covariance axes are shown with exact squared-coordinate/eigenvalue contributions. The covariance SVG has readable light labels and a bounded width. Recorded answers/reasons remain fixed; changed inputs require a fresh result.

The EM lab distinguishes separated numerical convergence from an identical-component fixed point. Pending inputs cannot advance the old model. Its 50-cycle cap bounds retained history; six ticks at that limit preserve all 51 plotted states without label crowding. The collapse figure's analytic maximum now includes the narrowest Gaussian, so its peak stays inside the shared window. The real-data panel discloses every one of the 150 observation IDs, measurements and split roles. It retains the unfavorable reserved-test comparison and the narrow-component diagnosis.

Primary-source follow-up checked the support/bound interpretation against [Stanford CS229 EM notes](https://cs229.stanford.edu/notes-spring2019/cs229-notes8.pdf) and the covariance/mixture distinctions against [the official scikit-learn mixture guide](https://scikit-learn.org/stable/modules/mixture.html). A Microsoft-hosted Bishop PDF did not load during this follow-up; its earlier recorded research was not relabeled as a fresh full-text reading.

Verification: 81 grouped model checks and 12 production browser cases passed on the corrected source, including the exact practice tie, both roots, hidden outcomes, covariance decomposition, numerical plateau and full 50-cycle state. Independent calculations separately covered seven crossing regimes, 25 projection cases, conditional weights, EM arithmetic and the collapse scale. Exact-source execution evidence for the three unchanged displayed programs and the 16 real-data fits remains applicable. Actual screenshot inspection caught the initially black covariance labels and a narrow-screen formula overflow; both were corrected before closure. The final coordinator record names the inspected captures and source applicability, rather than claiming that automated assertions establish visual quality.

This corrective review preserves the current revision and earlier phase completion timestamps; the phase ledger re-binds the appended design and corrected implementation after review. User acceptance remains separate. The next module topic is t-SNE, UMAP & Manifold Learning, whose prepared content still requires an authorized finish request.
