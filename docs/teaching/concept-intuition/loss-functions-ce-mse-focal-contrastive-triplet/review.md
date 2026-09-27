# Loss functions: concept-level intuition review

26 September 2026. Read the complete production lesson, all practice solutions, displayed code and existing LossFunctionsLabs representations. Full-mode manuscript is the topic JSX; this record specifies the new visuals. Author pass only: independent and browser review pending.

| Location / transition | Assessment and disposition |
| --- | --- |
| Opening and loss→gradient→update | Removed an early catalogue of not-yet-defined controls. Retained the three different application failures, previous-lesson connection and exact prediction/update figure. |
| Regression penalties and influence | Existing moving-observation lab shows residual, influence and fitted constants for squared, absolute and Huber objectives. Retained the mean/median/Huber comparison and corner conventions. |
| Conditional mean and likelihood | Existing variance decomposition and Gaussian/Laplace assumptions distinguish an estimand from a noise model. Retained with locally defined variables. |
| Quantile asymmetry | Added five-measurement slope-balance diagram at estimates 3.5 and 4.5, q=.75. Opposite aggregate signs explain the optimum at 4; arrows explicitly encode descent rather than physical point positions. |
| Binary / categorical CE | Existing hard-label penalty values, logits and stable formula suffice. Previous backprop lesson now owns the probability-minus-target signal diagram; no duplicate copy. |
| Entropy / KL decomposition | Added a population with positive prevalence .25: correct expected loss .56234 versus .69315 for .5, separating unavoidable uncertainty from the extra mismatch penalty .13081. |
| Stable computation / target formats | Existing extreme-logit API example and independent-versus-exclusive label distinction retained. |
| Smoothing | Added actual gradients against hard and smoothed three-class targets, including an overconfident model whose target-class gradient reverses. This grounds 'changed goal' before weighting. |
| Positive weighting, threshold and resampling | Existing .1 prevalence / weight 9 calculation explains changed odds; live decision lab connects scores to threshold costs. Retained. |
| Focal modulation and gradients | Existing loss/gradient distinction, product derivative, many-easy/one-hard contribution lab and label-error boundary already supply intuition; retained. |
| Real-data experiment / metrics | All runs, baselines, split and uncertainty boundaries retained. Probability and ranking metrics continue to answer different questions. |
| Pair and triplet geometry / mining | Existing coordinates, active inequalities, draggable geometry, gradients and collapse counterexample already expose the mechanism. Retained with API distance distinction. |
| Candidate CE, masks, temperature, false negatives | Existing live competition and both exact candidate-mask tables already teach these; deliberately did not add a duplicate matrix figure. |
| Multiple positives | Added two probability allocations with equal total positive mass, contrasting log-of-sum with mean-of-log. Shows why the latter rewards support for each positive. |
| Unit-vector distance, temperature limit, information bound | Existing identities and stated candidate/sampling assumptions retained; downstream evaluation remains the interpretation boundary. |
| Angular margin | Added actual unit-circle scoring geometry and logit values at 30° versus 40°. The feature stays fixed; only its true-class training score changes. Marks the decreasing-cosine branch and does not claim a complete boundary-policy implementation. |
| Reduction denominators, costs, mining scale | Existing short/long sequence table and area-correct score-storage squares already expose these mechanisms. Retained. |
| Scratch / library / modification | Complete NumPy objective and derivative program, matched SGD update, PyTorch mapping and smoothing extension preserved. All caveats remain connected to their relevant computation. |
| Practice / references / progression | All changed-input exercises, answers and annotated alternative sources retained. |

## Research used

Reviewed the information-theory/softmax passages of Stanford CS231n's original linear-classification notes: https://cs231n.github.io/linear-classify/ . Added an original population example to distinguish a one-observation one-hot target from an intrinsically uncertain outcome distribution.

Reviewed SimCLR §2 and Algorithm 1 at https://arxiv.org/html/2002.05709v3 . Confirmed the existing lab already supplies the paired and two-view masks; no new duplicate teaching block was justified.

Reviewed ArcFace's geometric interpretation and angular-margin formula at https://arxiv.org/html/1801.07698v3 . Added an original small-angle example rather than reproducing its figure or benchmark claims. These sources already appear in the lesson. No video playback or new model experiment claimed.

## Visual contract and checks

Quantile arrows encode direction, with numerical slope magnitudes beside them. They are explicitly not a spatial number line. Angular geometry is calculated from trigonometric coordinates, with a solid feature ray and a dashed scoring ray; labels and all values have text equivalents. Topic-owned charcoal/amber CSS, readable labels and narrow-screen layout. Neither static figure pretends to be a control. Existing live labs and all native programs remain unchanged.

Run scripts/verify-loss-intuition.mjs for the new arithmetic, source parsing and legacy model/program identity. Independent concept/correctness review and rendered checks remain pending.
