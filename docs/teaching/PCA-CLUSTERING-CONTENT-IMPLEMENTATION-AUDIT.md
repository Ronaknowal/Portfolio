# PCA and Clustering Evaluation: prepared-content implementation audit

Reviewed 14 September 2026 for the user's PCA-through-GMM verification. Scope: `pca-dimensionality-reduction` and `clustering-evaluation-validation-silhouette-ari-nmi`. An independent reviewer compared each complete prepared manuscript and visual specification with its complete implemented lesson, displayed examples, models, figures, labs and styles. The actual topic inventory commands and topic notes were read. This is a source-based review with complementary numerical checks, not a beginner user study or a fresh browser certification.

No omitted core explanation, derivation, displayed program, exercise or resource was found. The implementation retains the prepared depth and topic-specific mechanisms. The concrete defects below were repaired; the integrating agent owns the remaining browser/build/source-checkpoint closure. Earlier review reports are historical evidence, not substitutes for this comparison.

## Coverage map

| Prepared content | Implemented coverage and disposition |
| --- | --- |
| PCA sections 1–3: Wine question, four-point geometry, mean/direction/score/reconstruction, covariance and variance-loss identity | All present. The shadow figure introduces the mechanism before the projection lab; the energy partition and numeric denominators connect retained variance, SSE, per-row and per-entry error. Sign ambiguity is explained. |
| PCA sections 4–6: SVD fit/transform/inverse, new rows, scale/ties, Wine analysis, validation error budget | All present, including row/column conventions, the mean-only baseline, 133/45 training/validation split and inverse scaling. The scale lab and budget investigation implement different mechanisms. |
| PCA sections 7–8: loadings/correlations, biplots, distance and task information | All present, including the square-root-of-0.9 coefficient example, dot-product biplot interpretation, full orthogonal distance preservation and high-variance class collision. |
| PCA sections 9–11: changed practice, deeper branches, next step and references | Eight exercises, mini-project, exact six-percent budget target, eigen/Lagrange/SVD/Eckart–Young, rank/whitening, Gaussian null/denoising, prediction baseline, applications, computational costs and alternatives retained. Book/article/MIT video-and-slides annotations and data attribution retained. |
| PCA visual specifications | Seven figure families and four investigations are realized: shadows, conservation, matrix dimensions, Wine variance/score/coefficient views, biplot, Gaussian reference, residual alarm; projection, metric, budget and task-information labs. Documented optional specimen selection is handled by the budget lab. An exact numeric editor is an allowed alternative to dragging. No missing essential transformation was found. |
| Clustering sections 1–4: purpose/population/identity, silhouette, geometry, internal scores | All present, including the six-point calculation, negative and singleton cases, undefined domains, rescore versus refit, full-PCA invariance, ring counterexample, CH/DB/Dunn formulas and numeric examples. |
| Clustering sections 5–7: pairs, contingency, RI/ARI, entropy/MI/NMI/AMI | All present, including fixed-margin expectation, exact finite chance experiment, normalizer distinctions, constant conventions, purity/homogeneity/completeness/V-measure/Fowlkes–Mallows, label matching and variation of information. |
| Clustering sections 8–11: real Iris, rejection, stability and broader validation | Eight fitted Iris candidates, frozen-fit metric rescoring, coverage/intersection, fixed probes, exact weighted solver, trivial k=1 stability, resampling identity, hierarchy assignment, co-sampled denominators, gap/one-SE, cophenetic calculation and complexity/sampling distinctions retained. |
| Clustering section 12 and references | Frozen 90/30/30 pipeline, one-center baseline, eight changed exercises and exact changed-location result (cost 4 to 36), museum application and DBSCAN next step retained. Six programs and annotated primary/article/video resources remain present. |
| Clustering visual specifications | Seven figure families and five investigations are realized: identity alignment, silhouette distances/bars, rings, information/pair representations, Iris candidates, rejection lanes and fit/probe flow; silhouette, pair board, exact chance experiment, Iris weights and multiplicity/center ruler. The spec's attainable-score distribution is represented by an exact-value table plus overlap-count histogram. This preserves the finite-null mechanism and supplies readable exact values. |

All eight PCA and six Clustering Python programs match the corresponding manuscript programs after removing comments/blank lines and normalizing surrounding indentation. All PCA expected-output blocks are preserved; Clustering's resampling example expands the draft's narrative centers/labels/ARI result into its complete printed output. No learner program was changed by this audit.

## Correctness findings and repairs

| Location | Defect and repair | Evidence/closure |
| --- | --- | --- |
| `pca-models.js`, `rectangleMetric` | Legal b=4, multiplier=10 formed coordinates of ±40 but a shared ±20 guard threw before Apply. The validated rectangle now uses the same private centered covariance solver; the public free-point guard remains unchanged. | Fresh model run passed 34 grouped checks, including 27 boundary/interior combinations in raw and standardized modes and rejection of a free point at 21. Raw axis variances were checked against 4a²/3 and 4(mb)²/3. |
| `PcaLabs.jsx`, projection history | Add selected a fifth point; Back restored four points without restoring selection and could dereference a missing point. History now restores the selected index. | Integrating browser check: Add → Back, then edit/select. |
| `PcaLabs.jsx`, Fit best direction | Whole-degree rounding contradicted the exact fitted direction. The proposed angle now keeps fitted precision; only its label is rounded. | Changed D=(5,2.5) gives 43.339262118326985° and SSE 2.5819714051996776; rounding gave the larger 2.5824630807226803. Browser closure pending. |
| `PcaLabs.jsx`, task-information feedback | Two same-class collisions were described as four distinct retained coordinates. Feedback now correctly says that each retained location contains one class. | PC2 with y-based labels has two class-pure locations; labels remain distinguishable. |
| `ClusteringEvaluationLabs.jsx`, chance preset | “Match U exactly” copied the default U instead of the edited U. It now copies the current reference. | Added browser regression edits A in U, copies U, and expects NMI=AMI=1. |
| `ClusteringEvaluationLabs.jsx`, contingency and null display | Constant-candidate data had one column but a four-column layout; first-seen V=1 also made the highlighted cell disagree with the null's smaller-V-label definition. Column count and highlighted event now derive from the actual table. | Added browser assertions for three grid columns and `U1 and V0: 0 observations`. Independent changed-reference enumeration: 56 assignments; overlap counts 1,15,30,10; mean AMI approximately 1.7e-16. |
| `ClusteringEvaluationLabs.jsx`, resampling caption | The caption ambiguously scaled cost and said probes never move despite an explicit location editor. It now distinguishes multiplicity changes from shared coordinate changes and says tripling coordinates multiplies cost by nine. | Matches the existing exact weighted-solver behavior and changed exercise. |

## Learning-experience review

The nine teaching-standard checks were considered separately from numerical correctness:

1. **Route:** both lessons give a first-pass route and label deeper branches. PCA's route incorrectly promised four labs in sections 1–6; it now says three, with the fourth in section 8. Teaching successors remain Clustering Evaluation after PCA and DBSCAN after Clustering Evaluation.
2. **Cautions:** caveats have substantive homes (metric choice, task versus variance, population/coverage/chance and stability). Displayed programs concentrate on their mechanism rather than printing teaching-policy cautions. No recurring warning block required another rewrite.
3. **Real question:** Wine measurement compression and Iris grouping return to inspectable results, with actual data, declared protocols and baseline comparisons. The real-data code and calculations are preserved.
4. **Labs as investigations:** controls expose genuine changes: point geometry, units, reconstruction budgets and information loss; assignment, chance margins, frozen metrics and sample multiplicities. A shared defect allowed predictions to be edited after their answers were revealed. Compared selections are now frozen until inputs change or reset. PCA also hid neither the new budget crossing nor its new answer after a budget edit; those are now hidden until a new commitment. Root must execute the changed controls in the browser.
5. **Figures:** source review confirms the entities, intermediate operations, equal-unit geometry, baselines and exact-value alternatives. A complementary numeric bound check put all Wine and Iris scatter coordinates inside their declared SVG view boxes; this is not a pixel/contrast check. Rendered desktop/phone perceptibility is explicitly reserved for the root browser pass.
6. **Connections:** PCA links geometry to covariance/SVD and conserved energy to reconstruction; Clustering links pairs to contingency, entropy to chance-adjustment, and fitted representatives to frozen probe assignments. The canonical headline concepts are included or routed into labeled deeper branches. No unexplained omission was found.
7. **Code:** all 14 displayed programs preserve the prepared mechanism and reproducible outputs. Independent source/verifier identities still match the recorded successful native execution; no new execution is claimed for unchanged programs.
8. **Practice:** both lessons include changed-number/context exercises, hints and worked reveals after an independent attempt, exact numeric transfer targets and an applied mini-project/protocol. Practice was not replaced by the visual controls.
9. **Screenshots:** this reviewer did not operate the browser or inspect new rendered screenshots. The integrating root agent owns contrast states, post-prediction feedback, desktop figures and mobile layout. Existing reports do not certify these changed source versions.

The inline-visual reading pass found the representation choices appropriate to each mechanism: projected shadows and residuals, array shapes, empirical curves and class collisions for PCA; pair decisions, contingency cells, a finite distribution, silhouette distances and movable fitted representatives for Clustering. The fixes preserve those topic-specific forms. This assessment is an independent pedagogical heuristic, not evidence from first-time learner testing.

## Verification and remaining closure

Executed by this reviewer: the affected PCA model suite (34 grouped checks), source/expected-output comparison for 14 programs, native evidence source-and-verifier identity checks, changed-reference null arithmetic, and numeric plot-coordinate bounds. The existing PCA and Clustering browser verifiers were extended with targeted regressions; those new assertions are not counted as executed here. Root reported that the original 11-case PCA browser suite passed against the final production build before the complementary cases were added. The first Clustering run exposed a too-strict test locator for a wrapping label; the locator was corrected to the established non-exact semantic-label matching without changing runtime code. Both verifiers pass syntax checks. Root will run the expanded suites.

Reused native evidence remains source-bound: PCA examples SHA-256 `166065296261d699661c0b48dd89e853882a084d2b6fe272fc0102a5a046265b`, native verifier `ff32863ccb24924d150dc9c8f8abd0dd6795a0dd11690f84775070bb589cafd8`; Clustering examples `b068474e41d0befab400dc751797e2046964af1463927813bf7ea85e5537b2f1`, native verifier `04801d0d1f1328f13268267118dc21b79b73cb6666cd72061a2485bf94c87e41`. The native evidence records eight and six successful displayed-program runs respectively. These are retained prior executions, not reruns on 14 September.

Before closing the overall verification, root must run the topic/browser regressions against the final build, inspect the informative rendered states, and reconcile the ledger/evidence checkpoints. No curriculum order, publication status, prepared manuscript or specification was altered here. Source hashes below describe the inspected post-fix checkpoint and become historical if the integrating agent changes these files.

## Inspected source checkpoint (SHA-256)

| File | SHA-256 |
| --- | --- |
| `docs/teaching/drafts/pca-dimensionality-reduction/lesson.md` | `779d461fda06484aa620aa28d67267665b96ccb83bf3fcb281c54d491cdf6148` |
| `docs/teaching/drafts/pca-dimensionality-reduction/visual-specifications.md` | `20033e9b558b4aa5f8d6402485d1b967e4bbf70cb52a82cf70b2ce3c02c404bd` |
| `docs/teaching/drafts/clustering-evaluation-validation-silhouette-ari-nmi/lesson.md` | `4f35ac5d1c7ea4bdfffbf974e52517183a6e5f543359651ff0cd0f10244400ff` |
| `docs/teaching/drafts/clustering-evaluation-validation-silhouette-ari-nmi/visual-specifications.md` | `6546b674535f7ea43a6bab30df77cefb2caf9ef733fc3c0fb54211b78e52ed5f` |
| `src/learn/data/topics/pca-dimensionality-reduction.jsx` | `e242fc5a5d5300441abc797705a85fdceeb1997d02094be3c7d382b59e25b403` |
| `src/learn/data/pca-models.js` | `68fe8a36470a7351c7d0e73bc84339502912eaa8f405892d4b1b709fdedf1b21` |
| `src/learn/data/pca-examples.js` | `166065296261d699661c0b48dd89e853882a084d2b6fe272fc0102a5a046265b` |
| `src/learn/data/pca-wine-data.js` | `d5964935fb9bd4af7e4164fb1275fcdee339b5720267ad6e7cdf7ba85e141e84` |
| `src/learn/components/lesson-labs/PcaLabs.jsx` | `f7f6077243b3b20858c3e7e618bb07f4d2ab2174907cf2b491808f89f8710d2c` |
| `src/learn/components/lesson-labs/PcaFigures.jsx` | `a0c2d03847f7b55c5250bf77ccf2e150aeaaa946fd920aec4b883645fba54a79` |
| `src/learn/components/lesson-labs/pca-labs.css` | `6b4044f60f0dab5aae4b4ec7c95b88e3d07724b31d209c8797a92e2e2c0caea1` |
| `src/learn/data/topics/clustering-evaluation-validation-silhouette-ari-nmi.jsx` | `7030b810c96280027569f8b3a93f50a7f40981057dd8c891c88d82c5a59e60ac` |
| `src/learn/data/clustering-evaluation-models.js` | `efd46924ee804a0b28c61bb605a013f9b2b60122aac3307afc091f32a496a95e` |
| `src/learn/data/clustering-evaluation-examples.js` | `b068474e41d0befab400dc751797e2046964af1463927813bf7ea85e5537b2f1` |
| `src/learn/data/clustering-evaluation-data.js` | `e0568b62afc3f72ea2f1ca587db59554ccecec45769484222a2138bb14c51b07` |
| `src/learn/components/lesson-labs/ClusteringEvaluationLabs.jsx` | `1ec16771da4357c07c81f42134d2e0f5925d820aeaa44ec42fd02b9d49e0be14` |
| `src/learn/components/lesson-labs/ClusteringEvaluationFigures.jsx` | `c67d8127b522428b91a7ab4c3808ee4e34488bbabb105e3aa10a39eca0117e5c` |
| `src/learn/components/lesson-labs/clustering-evaluation-labs.css` | `b9f1a76d7109f1e8f4ada1bc09457651bb8e4d40cfb7904ae3038aeeb3aaac15` |
| `scripts/verify-pca-models.mjs` | `29a0abfdfc98fbb3d6aea8d8896b9fd095705331012d9b9f8329dcd73bc38cda` |
| `scripts/verify-clustering-evaluation-browser.cjs` | `09535ba361bf0d5ff4ae73d2fa7db5420eab27742e0c5fab57899d96f0f30724` |
| `docs/teaching/evidence/pca-models.json` | `70feb5136b29c239eb35625f7291ea2aecb3f8bcdcd94333369e0e9b353948b2` |
| `docs/teaching/evidence/pca-native.json` | `e88a6b6318e418ef36f74de202eb8464165cef59434411df701448b5a6992ba6` |
| `docs/teaching/evidence/clustering-evaluation-native.json` | `da309d5efb48aa3b3242e89d29226420ad9036205aee1c38debf28dfb3895a10` |
| `scripts/verify-pca-browser.cjs` | `3cb3a8b0b76ee440dd735757a95cba50d4116bf53be8294234b4ae5363931e6b` |

## Coordinator browser closure, 14 September 2026

The reviewer-only browser-pending statements above describe the state and responsibility at the time of that review. The coordinator has now completed the affected production browser checks and inspected selected informative captures; the [five-topic closure](PCA-THROUGH-GMM-IMPLEMENTATION-AUDIT.md) names the cases, actual visual observations, evidence reuse and limitations. Its source-comparison evidence binds the final repaired files. This closure does not attribute the coordinator's browser work to the independent reviewer.
