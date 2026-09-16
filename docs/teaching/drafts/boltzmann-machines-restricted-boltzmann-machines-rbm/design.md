# Boltzmann machines/RBM: author design and phase-two handoff

Stable ID boltzmann-machines-restricted-boltzmann-machines-rbm, Deep Learning Fundamentals position31,13September2026. Root author; research/write only. All7files are required prepared-content inputs, with runtime implementation, publication, formal independent review and browser/integration checks deferred.

## Inputs, conservation and scope

Read current handoff, complete teaching standard/design brief, model/systems playbook and code/retention instructions. Actual `node scripts/build-curriculum-inventory.mjs --topic boltzmann-machines-restricted-boltzmann-machines-rbm --work content` preflight consumed. It reports no bespoke blueprint or prerequisite record, no destination note; the unrelated resolved DSA bit note is not an instruction for this topic.

Full published original src/learn/data/topics/boltzmann-machines-restricted-boltzmann-machines-rbm.jsx consumed sequentially0–240,240–520,520–785,785–end. Original SHA-25691a9d5150f96b66a1cefa045b9f62709325f632f972f896f46c3699dd7e32274; baseline commit8c5da59f18516be77c29d5aeeafca3decca4f738. Original source untouched. No runtime/catalogue/title/route/inventory/prerequisite mutation by this packet. Keep title and stable ID; all added depth belongs to the existing probability-model scope. No orphan discovery requires an inbox note.

Sequence: predecessor Graph Transformers/Geometric Deep Learning; successor Spectral Normalization/Gradient Penalty; later Modern Hopfield link clarifies the separate energy-memory connection. These are actual stable-ID module routes. No claim that the next published lesson is the teaching successor. First-pass path §§1–7/practice1–5; advanced routes §8/practice6–8. Local refreshers cover weighted sums, probability, expectation, conditional independence, natural logs and gradients. Physical temperature/metaphors do not introduce a physics prerequisite.

## Teaching design

Begin with assigning probability to plausible images, then two visible switches, then one hidden switch and exact mass accounting. A shared three-switch fixture connects graph, conditionals, marginal dependence, free energy, normalization, positive/negative statistics, one full update, exact transition distribution, CD bias, probability-versus-state distinction and clamped inference. It is small enough to derive every number without a black box.

The real task deliberately uses64visible/8hidden binary units so256hidden configurations support exact Z and exact model expectations. This makes approximate-gradient claims testable rather than asking the beginner to trust an inaccessible normalizer. Real evidence continues the retained handwritten-digit source with a declared different binary transformation; duplicate audit happens after that transformation. Nine actual fits and independent-pixel baseline report all outcomes and multiple metrics without selecting a winner after the fact.

Visual forms follow the mechanism: switch/energy table, state-marginal bars, two-phase co-occurrence ledgers, exact probability-flow graph versus sampled particle, reconstruction versus probability-budget comparison, signed learned filters, independent sample contact sheet and masked-pixel conditional probability image. Five detailed specification families contain separate purposeful investigations; no single-lab quota or generic repeated text box. Predictions are input-bound and initially null; actual state/weight/mask edits, invalidation/reset, meaningful nulls and accessible numerical alternatives are explicit. Full executable code and actual inputs/results retained; no fabricated timing or loss plot.

## Canonical agenda audit and primary research

Canonical source: [Hinton, A Practical Guide to Training RBMs](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf),2010. Actually read its complete17-section contents and selected body §§2–8.1 and13–17 on13September2026. This is an actual section-list audit, not a claim to have read every page of every reference. Relevant contents/disposition:

| Guide agenda | Current lesson disposition |
| --- | --- |
| 1 Introduction;2 RBMs/CD overview | §§1–5, rebuilt from explicit tiny probability model |
| 3 Statistics: hidden states, visible states, collection, CD1 recipe | Exact positive expectations, sampled binary transitions, explicit mean-replacement counterexample and complete code |
| 4 Minibatches | §6 protocol/code; partial positive batch and persistent negative pool normalized separately |
| 5 Progress/reconstruction monitoring;6 Overfitting | Exact NLL, finite reconstruction counterexample, per-role gaps and held-out protocol |
| 7 Learning rate;8 Initialization | Declared fixed ascent rate, simultaneous step, smoothed visible logits and nonzero small random interactions; no universal recommended rate |
| 9 Momentum;10 Weight decay;11 Sparsity | Named optional update/regularization choices in diagnosis; the full machinery belongs to earlier optimization/regularization lessons. No unsupported claim these are mandatory for a valid RBM |
| 12 Hidden-unit count | Exact-enumeration/capacity tradeoff, actual8unit evidence, changed8-versus12design practice |
| 13 Other unit types: categorical/multinomial, Gaussian visible/both, binomial, rectified | §8 categorical/count/continuous support contract, explicit Gaussian–Bernoulli energy and normalizability caveat. General exponential-family construction and specialized count/rectified models are deeper guide reading; do not substitute a neural activation and silently retain Bernoulli likelihood |
| 14 CD varieties/PCD/fast weights | §§5–6 sampled CD/PCD mechanics/actual comparison; fast-weight overlay omitted as a nonessential specialized variant rather than conflated with persistence |
| 15 Visual monitoring | Shared-scale learned weights/activation probabilities, actual model outputs and diagnostic table |
| 16 Discrimination/free energy | §3 marginal derivation; §8 feature classifier, joint-label cancellation versus separate class normalizers |
| 17 Missing values | Exact masked hidden-state sum, actual completion/placeholder null and distinction from tied user-specific recommender models |

Additional primary reading actually inspected: [Hinton PoE technical report](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf),2000 version, abstract/§§1–3 selected probability product/normalizer/contrast rationale; [Tieleman2008](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Tieleman.2008.pdf),§§2–4 conditional gradient/PCD initialization, smaller-layer exact likelihood and empirical design; [Sutskever/Tieleman2010](https://proceedings.mlr.press/v9/sutskever10a.html),primary abstract precise non-gradient/convergence result, not a full proof-reading claim; [Salakhutdinov/Murray2008](https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf),§§2–3.2 normalization/importance assumptions/AIS bridge and estimator; [DBN2006](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf),abstract/§1/tied-network opening describing undirected top/direct lower model and historical initialization; [DBM2009](https://proceedings.mlr.press/v5/salakhutdinov09a.html),primary abstract on multilayer undirected inference; [collaborative filtering2007](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf),§§2–2.3 categorical ratings/tied user-specific model/missing pattern/conditional prediction. Brief historical source summaries support original worked derivations and real author measurements, rather than copying source wording or figures.

Current scikit-learn1.9.1 BernoulliRBM API inspected for PCD/SML, component orientation, ignored fit labels, binary score_samples input and random-bit pseudo-likelihood metric. Current UCI primary license/data page inspected. Current creator [Hinton lecture index](https://www.cs.toronto.edu/~hinton/coursera_lectures.html) fully read, lectures11e/12a–e/14a–e linked as alternate spoken learning. Creator Tieleman page also lists a2008recorded talk but direct Videolectures returned anti-bot page; it is not the learner's sole video resource, and no full-video viewing claimed. Earlier incorrect PDF URL and Deep Learning book endpoint failed; no claims rely on those unavailable fetches. A combined DBN tool output was truncated, then its relevant beginning was reopened and read separately.

## Original-source corrections and coverage

| Original issue | Prepared resolution |
| --- | --- |
| General BM necessarily fully connected; no-hidden BM factorizes | Allow sparse undirected edges and explicitly calculate dependent visible-only two-switch example |
| Energy treated like absolute likelihood/physical score | Dimensionless convention, normalizer, ratio and common-offset null |
| Conditional independence treated as general independence | Marginal dependence computed from complete2visible/1hidden table |
| Exact Z requires all2^(D+H)joint states; sampling needs Z | Analytic factorization plus smaller-layer enumeration; conditionals and within-model ratios need no Z |
| Sampling requires temperature annealing; fixed100×gain | Fixed temperature-one Gibbs/finite-state mixing conditions; no universal speed ratio |
| CD positive hidden probability called mean-field approximation | It is exact conditional expectation; averaging through later nonlinearities is different |
| Mean visible states called exact Gibbs/CD chain | Actual binary draws in training and an explicit.831036vs.809375counterexample |
| CD generally unbiased/stable/sharpens/no need caveat | Exact finite transition calculation distinguishes bias and sampling variance; precise2010non-gradient result and no universal convergence |
| log1p(exp(x)) called overflow-safe | logaddexp/expit/logsumexp in full executable program |
| Reconstruction/free-energy noise gap validates true likelihood | Exact finite reconstruction counterexample, normalized NLL and same-version normalizer cancellation limits |
| Unreproducible5000MNIST/CUDA12second/3.35pointDBNwin/PCDcurves | Replace with licensed398unique-binary real images,9declaredCPUfits, full weights/provenance and no timing leaderboard |
| DefaultPCD/currentlibrary claims and score treated as LL | Current official SML/PCD and random-bit pseudo-likelihood contract |
| Large weights automatically NaN; absent momentum inherently invalid; fixed size/cost cutoffs | Distinguish arithmetic stability, mixing, learning rate, optional penalties and exponential state count |
| Largest filter norm means most active/interpretable | Shared-scale signed filters plus actual input-dependent hidden probabilities; no invented semantic labels |
| Gaussian/activation substitution universally handles real data | Explicit support/energy/conditional/normalizability consistency |
| RBM stack/DBN/DBM/autoencoder conflated | Separate generative graphs/objectives and properly bounded historical statements |
| All modern models exact likelihood; all ML requires MCMC; RBMs universally dead/best | Distinguish metrics and objectives; relevant current-family ties without ungrounded popularity/production claims |
| AIS log estimate bound/unbiasedness conflated | Ratio estimator versus logarithm distinction and explicit support/transition/initialization assumptions |
| Missing values replaced by zeros or ratings masking equated with marginalization | Exact hidden posterior, observed/missing roles, true placeholder null and tied-family distinction |
| Open answers/generic lab and data-free practice | Eight changed problems with closed hints/solutions, multiple data/entity investigations and complete inference/training path |

## Author checks and closure

Nine fixed fits executed once. A later constructed reconstruction counterexample added to exact_checks and only that function rerun; real results/protocol unchanged. Three-switch direct joint enumeration matches hidden-marginal Z; all tiny analytic derivatives agree with independent central differences to1.9729e−11; exact transition stationary invariant checked; conditional5/7/energy-offset invariance checked. Binary duplicate handling found398unique and split238/80/80 before fitting. All9ignored-placeholder effects below1e−12, all metrics finite, all outcomes retained. The inline full program matches its companion file. Numerical author verification is not formal independent phase-two review.

Final author reread replaced three lab prediction defaults that repeated visible worked answers with fresh hidden-bias, input00 metric and observed-pixel18 interventions; the existing computed examples remain explanatory baselines. Executed investigation_checks only to calculate these fresh fixtures and updated the inline full program. No fits or experimental choices changed.

Final author reread checks manuscript explanations/equations/real tables against results, complete runnable code, first-pass/deeper route, inline visual placements, accurate metric names, actual sequence links and all17closed disclosures. Specs include actual parameter/state/mask inputs, meaningful nulls, bound/reset/invalidation, numerical contracts and deferred checks. Source conservation and local file links validated at content checkpoint. No unsupported performance graph, copied illustration or temporary source dump retained.

Content packet is ready for research/write completion. Phase two must consume all7files, implement topic-owned lazy-loaded visual forms and bounded numerical interactions, independently verify translated inference/metrics, run the applicable content/learning/integration reviews and browser/accessibility checks, then update implementation status. This packet is retained work, not disposable scratch.
