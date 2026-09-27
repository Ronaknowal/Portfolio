# Linear & Logistic Regression: concept-by-concept reading review

26 September 2026. Full-mode revision of the existing production lesson. The JSX is the authoritative manuscript; no generator owns this topic body. Author review is separate from root independent and browser closure.

## Complete concept map and dispositions

| Location / new conceptual transition | Existing support assessed | Gap and disposition |
| --- | --- | --- |
| §1 observation → available input → training/inference | Shipment timing diagram, feature/target definitions, split table and baseline examples | Retain: information availability and separate output meanings are explicit. |
| §2 line → residual → squared loss → units | Residual lab displays exact same four rows and adjustable line | Retain; worked MSE and RMSE distinction already supplies the bridge. |
| §2 best constant → mean identity → conditional mean | Correct derivation but immediate algebra | Add the balancing-pulls explanation before expansion; explain irreducible spread versus displacement cost. Population extension reuses this identity with its finite-moment condition. |
| §3 scalar optimum → matrix rows/columns | Complete numerator, denominator, design matrix and predictions | Retain; introduce columns as allowed coordinated changes before normal equations. |
| §3 normal equations → projection → rank | Formula and duplicate-column executable example | Add signed contribution diagram: each feature direction's residual pull cancels. This teaches perpendicularity in observation space without pretending a two-dimensional feature plot represents four-dimensional outcomes. Duplicate-column example remains the concrete nonidentification case. |
| §4 gradient → update → curvature → step limit | Live landscape, exact first update and eigenvalue derivation | Add crossing/shrinking/amplifying intuition before eigenvector recurrence; keep full stability bound and scaling caveat. |
| §5 binary outcome → log odds → probability → threshold | Score/odds lab, numerical example and odds-ratio explanation | Retain; each representation is already tied to the same score. |
| §6 likelihood → loss → stable calculation | Observed-outcome probabilities, extreme-score explanation and full programs | Retain. Add one-row forward/backward numerical flow at gradient derivation: belief → signed score pull → feature-weighted parameter pull. |
| §6 convexity, penalty normalization and library parity | PSD expression, eight-row scratch/library equality, changed-sample-count practice | Retain; the new flow introduces the gradient before this depth. Preserve full parity code. |
| §7 probability → action cost → class weighting | Confusion lab and expected-cost derivation | Retain; distinct fixed-probability threshold and changed-objective effects are explicit. |
| §8 separation → penalty → ridge/lasso/elastic net | Live diverging-weight example and correct scalar solutions | Add matched-axis coefficient-response plots, same λ and same evidence a; explain marginal penalty charge to account for lasso's exact-zero interval. Elastic net continues from the two ingredients. |
| §9 feature map → nonlinear boundary → capacity | Feature-map figure and runnable polynomial example | Retain; input-space distinction, feature count and validation reason are explicit. |
| §10 complete workflow and refit | Full program, small observed gain interpreted, leakage boundary | Retain complete program/results; no rerun of unchanged fit. |
| §11 fitting → inference assumptions → causality | Gaussian/Laplace likelihood and Gauss–Markov conditions | Retain statements, add leverage story before the hat-matrix definition. Add repeated-estimate versus fresh-outcome intuition before uncertainty formulas. Existing interval diagram supports the two variance terms. |
| §12 solver choice → sparse/streaming costs | Task table, operation/storage calculation and complete stream example | Retain: work, memory, convergence and objective compatibility differentiated. |
| §13–14 independent transfer and connections | Twelve changed tasks, explained solutions, complete alternate capstone, next-topic bridge | Retain all; new diagrams support existing gradient, rank, penalty and uncertainty tasks. |

## New representation contracts

`NormalEquationBalanceFigure` belongs directly before normal equations. Four shipment residuals are 0.1, 0.2, −0.7, 0.4; the intercept contribution is each residual and slope contribution is x times it. Both sums vanish. Amber positive/right and neutral negative/left bars share one signed scale. Text gives exact rounded arithmetic and the coordinate interpretation. Mobile stacks the two independent directions; no simulated interaction is claimed.

`LogisticLearningFlowFigure` belongs after the gradient equation. A constructed row x=2,y=1 under b=−1,w=1 gives z=1, p=σ(1), score derivative p−1, weight derivative 2(p−1), and rate-0.1 weight 1.053788… . The intercept's separate contribution and batch/penalty distinction are explicit. Semantic ordered HTML reflows at narrow widths.

`PenaltyResponseFigure` belongs after scalar ridge/lasso solutions. Three actual response curves use a∈[−2,2], λ=0.5; horizontal and vertical scales match in each panel and across panels. The white point shows a=0.3. No training benchmark or universal multivariate coefficient path is implied. Small multiples stack instead of shrinking a wide multi-panel SVG.

## Research actually consulted

- [Dive into Deep Learning, Linear Regression](https://d2l.ai/chapter_linear-regression/linear-regression.html), written chapter: task/model/loss/optimization progression. Used to assess the handoff from error to parameter updates; original shipment examples and new diagrams were derived locally.
- [Seeing Theory, Regression Analysis](https://seeing-theory.brown.edu/regression-analysis/index.html), archived creator page: read OLS interaction description, mean/slope/SSE explanation and dataset-selection structure. Did not claim to operate its live simulation. Existing residual lab already supplies that active manipulation; the newly identified gaps were later mechanisms rather than another intro lab.
- Existing CS229, NumPy, scikit-learn and Penn State references remain attached for derivation/API/uncertainty scope. Their previously recorded version-bound native evidence is reused only for unchanged programs.

## Author checks and boundaries

Read complete production lesson, including later inference/solver sections and all practice. Checked new worked arithmetic independently and parsed changed JSX with esbuild; details and source hashes are in `author-checks.json`. Canonical example programs, numeric engines, recorded outputs, prior practice, metadata and publication IDs are preserved. No new API or performance claim. Root owns browser and independent review; neither is certified by the author record.
