# Concept-level author review: Dropout, DropPath and Stochastic Depth

Scope: the complete current lesson, all ten numbered sections, library route and seven worked practice problems were read. Production JSX is the current content checkpoint; the manuscript and renderer agree. Browser and independent review remain pending root.

## Concept map and decisions

| Location / transition | Assessment and action |
| --- | --- |
| Opening, §1 availability → masked forward → shared mask backward → update | Retained the complete two-feature numerical example and live update lab; both losses and gradients already explain the mechanism. |
| §2 survival scaling → expectation → variance → nonlinear outputs | Retained four-mask enumeration, endpoints, mean/variance lab, ReLU counterexample and expected-loss boundary. No additional generic figure needed. |
| §3 named axes → shared decisions → covariance | Existing geometry lab and two-channel calculations adequately expose which positions share randomness. Retained. |
| §4 residual branch bit → shortcut derivative → depth schedule | Existing branch and depth investigations support every transition, including skip versus whole-output masking and expected active blocks versus actual compute. Retained. |
| §5 train/eval versus gradient tracking → BN variance → selective MC | Existing mode lab and numerical variance mismatch explain why evaluation controls cannot be interchanged. Retained. |
| Library route | Full scratch mask and normal torch/torchvision paths, axis contracts, locked-feature changed case and endpoint checks retained. |
| §6 noisy objective → measured model choice | Retained actual digit fits, seed comparisons, correct counts/CE and no-universal-benefit conclusion. No manufactured new experiment. |
| §7 probability averaging → entropy disagreement → regression uncertainty | Classification examples and live MC view sufficient. Added exact two-component regression mixture: variance of means 1 plus conditional noise .25 = 1.25, with denominator and Monte Carlo noise distinction. |
| §8 spatial versus individual masking | Replaced short list with subsections. New 5×5 shape figure holds nine removed positions fixed while changing spatial contiguity. New effective-matrix figure distinguishes a column mask from independent connection decisions. |
| §8 temporal state and branch mixing | Retained Zoneout's old/proposed state example. Added Shake-Shake forward .25/.75 and independently sampled backward .75/.25, explaining why ordinary differentiation of the forward expression is insufficient. ShakeDrop remains an explicitly linked separate recipe. |
| §8 Gaussian/variational/local noise | Added two-route moment figure showing N(0,.08) marginal equivalence, independent weights, non-identical draws, expected per-example loss and joint-covariance boundary. |
| Attention preview, §9 practice, §10 sources | Existing attention row-sum counterexample, all seven changed-input tasks/solutions, and prerequisite reuse remain sufficient and preserved. |

## Research actually read

- Wan et al., DropConnect, §2.2–3: https://proceedings.mlr.press/v28/wan13.pdf — connection masks; used to distinguish original convention from our explicitly inverted teaching example.
- Ghiasi et al., DropBlock, Figure 1 and §3 Algorithm 1: https://proceedings.neurips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf — spatial redundancy, center expansion, overlap and normalization; linked at point of use.
- Gastaldi, Shake-Shake, §§1.2–1.3: https://arxiv.org/pdf/1705.07485 — distinct forward/backward random mixture, test coefficients; original arithmetic added rather than reproduced wording.
- Kingma et al., local reparameterization, §2.3 equation 6: https://arxiv.org/pdf/1506.02557 — Gaussian moments and per-example local noise; precise caveat avoids claiming identical full-batch joint samples.

## Checks and limits

Changed JSX parses. Independent arithmetic checks cover both matrix products, mask counts, local Gaussian moments, Shake-Shake example and total variance. Generator reproduces the exact topic source. Existing native training outputs and full programs were preserved, not needlessly rerun. Topic-scoped charcoal/amber CSS uses wrapping DOM panels and grids, no SVG text collisions, fake handles or global style changes. Rendered behavior is not marked passed until root checks it.
