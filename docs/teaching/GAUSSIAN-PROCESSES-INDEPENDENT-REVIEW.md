# Gaussian Processes — independent implementation review

19 September 2026. Reviewer: `semi_supervised_implementation`, who did not author the Gaussian-process implementation. Scope: prepared-content phase two for `gaussian-processes-gp`. Disposition: **passed after one material figure correction; production integration remains the increment owner's separate check.** This is a source/code and learning-experience review, not a novice user study.

## Inputs and scope

Read the complete prepared manuscript and visual contracts, design/conservation record, incoming RKHS destination note, implemented narrative, pure numerical model, figure/lab components, styles and author implementation record. Compared the implemented structure with the six figure mechanisms, three investigation contracts, complete native examples, practice and references. The first investigation's direct and spatial interfaces are a justified split, yielding four mounted investigations without dropping a contract.

Reused the author's independently executed native programs, model fixtures and broad browser evidence rather than repeating their complete runs. Checked that **all** hashes in [the author receipt](evidence/gaussian-process-author-review.json) matched files after the correction. The current [browser receipt](evidence/gaussian-process-browser.json) passes 91 grouped checks with 18 capture digests; [model evidence](evidence/gaussian-process-model.json) contains 1,196 comparisons; [content evidence](evidence/gaussian-process-content.json) covers strict formula rendering, headings, examples and closed practice disclosures. These counts describe author checks, not additional independent executions by this reviewer.

The independent checks below target different examples and failure mechanisms. No central ledger, authored component, checkpoint-bound packet or destination note was changed by this reviewer.

## Correctness review

- Finite-dimensional Gaussian definition, covariance versus correlation, PSD restriction and random-line example are consistent. The plotted joining lines are expressly a convention rather than a claim about continuous sample paths.
- Conditioning uses observation noise in the training system and distinguishes latent variance from a new independent measurement's variance. Cholesky solves implement the stated formulas. Caching depends on inputs, noise and kernel parameters; changing observed values does not incorrectly invalidate or alter covariance.
- The one-observation update scales to the matrix calculation. Opposite observations can cancel their mean contributions without canceling information. The text scopes variance monotonicity to a fixed model, preserving the distinction from hyperparameter refitting.
- Kernel assumptions are explicit. Sum/product kernels are not confused with a product of Gaussian sample functions. Matérn differentiability, RBF length versus periodicity, and ARD versus causal importance are distinguished.
- The historical experiment separates 1990–95 training, 1996–97 development and 1998–99 test; the chosen family is refitted through 1997. The browser exploration freezes recorded kernel settings, centers only the selected prefix, and identifies itself as a different experiment. The noise assumption is not represented as NOAA measurement uncertainty.
- The poor test interval result, 13/24, remains visible alongside improved MAE. Pointwise model-conditional intervals are not sold as simultaneous bands or unconditional calibration; serial dependence rules out interpreting those 24 points as independent binomial trials.
- Probe selection uses **current posterior covariance to the selected target**, squared and divided by candidate predictive observation variance. It is not a rule to maximize the candidate's own variance. Negative covariance can provide information; zero covariance gives a genuine null. New candidate measurement values are not needed or fabricated.
- Advanced claims preserve their boundaries: GP classification requires non-Gaussian inference; mean versus variance costs differ; inducing methods are approximations; random features stay fixed between training and prediction.
- The incoming RKHS note is fully carried into section 9: average-loss KRR matches the mean when `sigma² = n lambda`; deterministic ridge alone supplies no posterior intervals; the Brownian RKHS is anchored and requires square-integrable derivatives, whereas its sample paths are almost surely outside; the finite-rank exception is retained. Feature-array construction and dense feature-regression solve cost are separated. The integration owner can close the destination note after its normal final checks.

### Complementary numerical cases actually executed

Used direct Node imports of `gaussian-process-model.js` and `gaussian-process-data.js`, with `node:assert/strict`. These were read-only executions and did not regenerate the author's receipts.

1. **Repeated independent noisy readings.** At the same input, readings `[2, -1]`, unit prior variance and noise variance `0.25` have closed-form posterior mean `1 / 2.25 = 0.444444444444…` and variance `1 / 9 = 0.111111111111…`. Browser-model results matched. Changing only both values to `[4, 4]` retained covariance and one cached factorization.
2. **Negative covariance remains informative.** With the lesson's observations `(0,1), (2,-1)`, RBF length 1 and noise variance 0.25, target `-0.5` and candidate `1.5` give posterior covariance `-0.05846869814254746`, candidate latent variance `0.3344696269695664`, target variance `0.37480327218414633` and reduction `0.005849044167118611`. The remaining variance is `0.3689542280170277`. This checks the sign-sensitive connection against the sign-insensitive information gain.
3. **Changed targets and candidates.** Crossed targets `[-0.5, 0.5, 1, 1.5, 3.5]` with candidates `[-0.5, 0.5, 1.5, 3, 4]`. All 25 gains were nonnegative and remaining target variances were nonnegative within `1e-10` numerical tolerance.
4. **Exact independent-kernel null.** Without existing observations, target 1 and candidates 1 and 3 with noise variance 0.25 gave gains `[0.8, 0]`.
5. **Forecast indexing.** A frozen composite forecast with cutoff 95 and horizon 1 starts at `1997-12`; a noninteger cutoff 95.5 is rejected, rather than silently slicing at another month.

### Independent browser check actually executed

Used headless Edge via the bundled Playwright runtime, shared dev server `http://127.0.0.1:4184`, viewport 390 × 900 and reduced motion. The served route was `/learn/path/full-curriculum/gaussian-processes-gp?module=classical-ml`.

- Set probe target to `-0.5` and candidate 1 position to `1.5`. Confirmed no result before commitment. Predicted candidate 1 with a written reason. The result correctly recognized a maximum, displayed covariance `-0.058469` and gain `0.005849`, and painted the negative connection with a dashed SVG path.
- Changed reading 1's value to 3. The old result and selected prediction cleared. After a new commitment, the complete gain result text was unchanged, as fixed-model covariance requires.
- Changed direct covariance from `+0.5` to `-0.5` against the saved baseline. A “Mean only” prediction was correctly graded; the squared covariance reduction stayed fixed.
- Asserted the corrected density-slice tick coordinates in the rendered DOM: x labels `[-3, 0, 3]` at `[74, 200, 326]`; y labels `[-3, 3]` at baselines `[285, 57]` (`mapY(value) + 5`). The observation-2 slice is at x=284.
- No page exceptions or page-level horizontal overflow occurred in this independently exercised state.

## Finding and verified closure

**P2 — density-slice labels did not share the plotted coordinate transformation.** Contours used `mapX(x)=200+42x`, but the original `−3/+3` labels were at 52/348 instead of 74/326. The y-axis labels similarly did not track `mapY(y)=166−38y`. This was a quantitative figure accuracy issue, not merely spacing.

Reported the defect to the author. The author now derives ticks and labels from the same maps as contours and the observed-value slice, using an intentional 5-unit text baseline offset on y labels. Independently verified live coordinates as listed above and inspected the updated desktop density-slice capture. The revised labels correctly agree with the contour geometry. **Resolved.** No other material correctness or implementation gaps were found in the reviewed scope.

## Separate learning-experience review

The lesson has a meaningful first-pass route before its deeper branches. It begins with deciding where another pipe measurement would help, introduces function values and covariance before matrix conditioning, then returns to that same mechanism for prospective measurement selection. The NOAA experiment demonstrates an actual failure of interval performance rather than supplying only a flattering model result.

The figures use different mechanisms for different ideas: vector-to-function correspondence, density slicing, signed mean contributions, aligned prior draws/covariance maps, temporal forecasts with misses and residuals, and target/candidate information links. Labs expose real problem inputs; none substitutes a decorative generic score panel. Prediction choices are initially unselected, reasons are required, and edits invalidate answers. Solved examples elsewhere in the reader are clearly distinct from each investigation's pre-answer state.

Inspected the retained final captures `slice-1366.png`, `kernels-390.png`, `probes-390.png`, `forecast-390.png`, `matrix-1366.png` and `conditioning-result-390.png` under `scratch/gaussian-process-implementation/`. The review checked common quantitative axes, interval bands actually painted, chart/data/legend agreement, the visible 13/24 failure, numerical tables, readable mobile stacking and informative changed-input results. In particular, rounded covariance-map zeros are qualified by exact/scientific-notation examples; forecast misses use shape as well as color; candidate links state that their two lanes separate roles rather than dimensions. These informative screenshots supplement, rather than replace, the live interaction checks.

Practice changes inputs or assumptions and asks learners to diagnose claims, propose a kernel and transfer the KRR connection; separate closed hints and solutions preserve the opportunity to attempt each task. References describe why and when each resource helps. The next-topic bridge distinguishes Gaussian prediction at unlabeled locations from the assumptions used in semi-supervised classification.

No new learner study, screen-reader session, third-party video viewing or complete production-performance run is claimed. The independent browser pass is complementary, not an exhaustive re-execution of the author's 91 groups. The increment owner still owns route metadata, final production loading/build and ledger completion.

## Reviewed source binding

SHA-256 values after the correction:

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/gaussian-processes-gp.jsx` | `9fcb1beae5e2261240b3f43b60a334ba60633f33fab83ee65fbe4924647a3446` |
| `src/learn/data/gaussian-process-model.js` | `5af96779860ae275b9458044cda19b95e8af7017e4226832063c89a0255667bf` |
| `src/learn/data/gaussian-process-data.js` | `c144e5d09b730b480aed9fb454bdeda58de63a7ab4df2da3d548227b5a2a9b7f` |
| `src/learn/data/gaussian-process-examples.js` | `e2effc3ffb00e2091ff7c5d03e3cf24681f317bd037891988b753af3c84cf178` |
| `src/learn/components/lesson-labs/GaussianProcessFigures.jsx` | `6c746b14b9ed32ab73b81f0c69710ac32a37f979b2149f591127be04c6632b4d` |
| `src/learn/components/lesson-labs/GaussianProcessLabs.jsx` | `eaf567f102f8fcdd380adc9b1fd6cf6fab02c1430d1bad1842c33bc6d5e4d7df` |
| `src/learn/components/lesson-labs/gaussian-process.css` | `e31c28a6a18a9971eaa9a9a4a74383d8b9d49cc5c24f5606b3f34a5d6361a0db` |
| `src/learn/data/curriculum/blueprints/gaussian-processes-gp.js` | `c5fe7e03eeb8561b490a4781c5446d4e99bca7a275d7560cfacf54aa86d592ce` |
| `docs/teaching/drafts/gaussian-processes-gp/implementation.md` | `c339b9d63bb2803e3fd353d1f5468fcacd43fe03e91c821a42f5e3cf590784ff` |

The linked author receipt also binds the native programs, CSV, provenance and verification scripts. Reopen affected review checks if these implementation files change in a way relevant to the conclusions above; a later administrative completion update to the author's status is not itself a new technical implementation.
