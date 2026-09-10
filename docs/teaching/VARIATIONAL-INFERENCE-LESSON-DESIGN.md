# Variational Inference — scoped lesson design

Author implementation and native/browser verification completed10 September2026; see [actual evidence](VARIATIONAL-INFERENCE-VERIFICATION.md). Parent integration and user acceptance remain separate. Stable ID `variational-inference`, mathematics position21. Title, identity and module order are preserved. This topic follows Monte Carlo/MCMC and precedes Exponential Families & Sufficient Statistics.

## Learning contract and existing coverage

The learner can follow a posterior, an expectation and an ordinary gradient. Bayesian Inference and MCMC supply those foundations; this lesson refreshes density versus distribution parameters, normalization, expectations and the gradient actually being optimized. KL, entropy, score and natural parameters are introduced locally where required, without presuming the later information-theory or exponential-family lessons.

The practical question is: can a distribution that is cheap to evaluate and sample answer the posterior question we care about? The finish line is a complete checked approximation, not a minimized loss alone. Learners will derive the ELBO, calculate it for a small target, distinguish family/optimization/Monte Carlo error, carry out coordinate updates and stochastic gradients, scale minibatches correctly, explain amortization and assess consequential missing dependence or modes.

The original lesson has useful ELBO/mean-field/reverse-KL explanations, the symmetric two-normal mixture example, coordinate-ascent/reparameterization/minibatch mentions, amortization and an MCMC-comparison workflow. Preserve all. Its full source is archived at `scratch/variational-inference-native-verification/original-lesson.jsx`, SHA256 `68695b35e219bfae9a451c412f9c86c0a1e9d6ee759eab905ca4611d846aff76`. Its Python example was independently executed: `(0,3)→.877`, `(3,1)→.689`, `(-3,1)→.689`. The previous body has no mechanism diagrams, real optimization program, gradient derivation, annotated resources or independent explained practice.

## Scope and ownership decisions

| Idea | Existing evidence | Decision and owner |
| --- | --- | --- |
| ELBO constant, family comparability and three errors | Open incoming MCMC note; older sections3/6 are imprecise | Include exact fixed-joint algebra here. Unknown evidence prevents evaluating absolute KL but does not prevent dropping the constant for optimization. Exact same-model ELBOs across families are comparable. |
| Mean-field uncertainty | Existing factorization but no calculation | Derive correlated-Gaussian optimum and contrast two linear quantities. Do not claim every variance is underestimated. |
| Coordinate-ascent inference | Mention only | Derive normalized exponential-of-expected-log-joint factor update and execute a two-factor example. |
| Score/pathwise gradients | Reparameterization mention | Derive and independently check both, support/regularity limits and an instructive counterexample to universal variance rankings. Execute a complete optimizer. |
| Stochastic VI and minibatches | Mention only | Teach likelihood scaling with exact all-batch oracle; distinguish posterior draws from data batches and original global/local SVI from general stochastic gradients. |
| Amortization | VAE mention | Locally complete Normal example shows shared mapping versus separate distribution parameters. Full VAE training belongs to its generative-model owner; flows get a capability/limitation bridge only. |
| KL/entropy theory and exponential-family natural gradients | Later dedicated owners exist | Introduce only what the current derivations use; connect rather than forward-require those lessons. |
| Applications | Earlier industry list | Use correlated uncertainty of sum/difference, multimodal sign decisions, repeated sensor observations and large-data posterior gradients. Each changes a concrete inference, without an application quota. |

The title already encompasses these methods; no rename is needed. The unrelated bit-manipulation inbox does not apply.

## Concept and reading sequence

1. A distribution is the optimization result: latent variable versus variational parameters; a known finite posterior separates approximation from numerical integration.
2. Derive KL and ELBO from Bayes' rule; expand expected log likelihood minus prior KL; explain normalization/support and differential-entropy caveats.
3. Restricting a family changes the answer: factorization versus actual independence, correlated Gaussian mean-field optimum, contrasting linear decisions.
4. Derive and perform coordinate ascent; visible update order and a monotone objective are distinct from eliminating the family gap.
5. Preserve and deepen the two-mode example: density overlay, actual KL quadrature and sign probability; compare candidates without pretending they are a global optimizer.
6. Differentiate an expectation: pathwise and score estimates, analytical entropy, fixed noise and a full seeded optimizer; valid support/derivative interchange.
7. Scale data contributions and share inference computation: unbiased minibatches, global/local SVI context and an exact amortized Normal example.
8. Validate the quantity that matters: three errors, predictive uncertainty, multiple starts, richer families, constrained transformations, independently diagnosed MCMC and changed-condition practice.

Every worked program includes imports, data, algorithm, exact executed output and interpretation. Practice covers algebra/support, family-induced error, an independent coordinate update, missing-mode decisions, gradients, minibatch bias and amortization/predictive uncertainty; hints and explained solutions stay separately accessible.

## Representation contracts

| Representation and location | Concrete model and encoding | Learning action and independent check |
| --- | --- | --- |
| ELBO balance, beside its derivation | Prior1/3, observed likelihood(.24,.60,.36), joint(.08,.20,.12), evidence.4, posterior(.2,.5,.3). Same-state bars compare p/q; component table uses natural logs. | Change qA and B's share of remaining mass or restrict qB=qC. Compare gap with exact log evidence. Zero-q terms use their limiting value; support mismatch gets infinite KL. Exact arithmetic and independent summation check. |
| Gaussian uncertainty geometry | Target covariance[[1,rho],[rho,1]], mean0. Compare exact joint, reverse-KL mean-field diag(1−rho²), and product of true marginals diag1. Equal-coordinate ellipse geometry represents Mahalanobis radius1, containing39.35% in2D, not68%. | Change correlation/family and observe variance of sum and difference. Independent matrix KL/covariance oracle; avoid blanket underconfidence claim. |
| Coordinate-ascent path | Target mean(1,−1), covariance as above; optimal diagonal variance fixed; q means start(−2,2). Plot axes are variational means, not latent observations. | Alternate exact factor updates; inspect state/history and remaining family gap. Analytic quadratic objective, independently solved first-order equations, and each update's nondecrease. |
| Two-mode density projection | p=.5N(−3,1)+.5N(3,1), q=N(m,s²). Density axes and finite display window; reverse KL computed in noise coordinates with stated numerical quadrature. | Compare original three candidates and changed shapes. Inspect q(theta>0), true.5. Independent adaptive quadrature and grid refinement; candidate comparisons are not global-optimality claims. |
| Noise-to-gradient investigation | Normal target and q=N(m,exp(2a)); seeded standard-normal noise maps to draws and per-draw pathwise/score contributions. | Move q, change sample budget/seed, compare Monte Carlo and analytic gradients. Show p=q counterexample: normalized score estimate is zero but the chosen pathwise estimator still varies. Finite-difference/integration/variance checks. |
| Amortized sensor diagram | PriorN(0,1), noise variance1, observations−2,0,2; posterior meansx/2 and variance.5. | Static split between independent parameters and shared mapping; no controls because relationship is fixed. Exact conjugate posterior and predictive variances checked. |

All quantitative graphics are model-calculated, not benchmarks. Readable labels, local tables, keyboard controls, explicit/resettable seeds, bounded input and no timers. Read the page without operating controls as well as testing interactions; meaningful initial diagrams must teach the relationship immediately. Mobile geometry must retain readable effective labels and no page overflow.

## Research and claim ledger

- Blei, Kucukelbir and McAuliffe review, `https://arxiv.org/html/1601.00670v9`: inspected ELBO, mean-field, coordinate-ascent and stochastic-VI sections. The2017 Columbia PDF timed out; do not claim that full PDF was reviewed. Exact same-joint comparability follows the stated KL identity. Gaussian counterexample qualifies the review's broad variance description.
- Ranganath, Gerrish and Blei2014, `https://proceedings.mlr.press/v33/ranganath14.pdf`: inspected score-gradient equation2, control-variate and minibatch sections. Baselines in the implementation must be constant/independent where unbiasedness is claimed; same-sample fitting is not silently treated as exact.
- Hoffman et al2013, `https://www.jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf`: primary source for original conjugate global/local SVI; exact inspected locators and depth will be recorded with final evidence.
- Stan Reference Manual2.39, `https://mc-stan.org/docs/reference-manual/variational.html`: current family and stochastic-optimization description inspected; distinctions between transformed-space q, constraints, adaptation and approximate ELBO stopping will be checked before publication.
- Stanford Online, Chelsea Finn, CS3302022 lecture11, `https://www.youtube.com/watch?v=iL1c1KmYPM0`: verified institutional title/description covering latent-variable, variational and amortized models. Full playback not claimed. Its deep-learning framing is an alternate later bridge, not a prerequisite.
- The official NIPS2016 tutorial page's old Channel9 video link redirects to a generic Microsoft shows page. Do not offer that dead video as a verified resource. Author tutorial slides can still be independently assessed.

## Verification plan and current limits

Pure finite/analytic models: independent finite sums, closed Gaussian matrix calculations, exact conditional optimization, independent adaptive quadrature, finite differences and all-minibatch enumeration. Native examples run in isolated Python3.12 processes with exact stdout. Check numerical convergence and state immutability/invalid inputs. Report finite empirical checks as empirical, not proofs.

Browser:1440/390/320; full first-pass route, headings and references, model states/controls/reset/errors, native programs, narrow equations, keyboard focus, readable SVG labels/text equivalents and independently opened screenshots all passed as recorded in the verification file. Parent owns registration and production integration. The implemented source is frozen for independent integration review; user acceptance remains pending.
