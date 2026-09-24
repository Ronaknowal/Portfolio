> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Conditional Random Fields — content design and continuation

Stable ID `conditional-random-fields-crf`; Classical ML, Probabilistic & Graphical Models; authorized batch position 11. Mode: **research and write only**. Content complete after the author read and calculations below; root owns checkpoint registration. Implementation, formal independent review, rendered/keyboard review and publication are **not started**. User acceptance is separate. Prepared 12 September 2026.

## Source and preserved learning scope

Read the complete original `src/learn/data/topics/conditional-random-fields-crf.jsx` in bounded sections. Baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; source SHA-256 `5d52e547b3dff2e128d6ed6dc000b3c3f33e50cbd459b4223c085ebc214ffb9d`. The original runtime was not edited. Current source is recoverable from the baseline; do not create a duplicate source archive. Topic inventory `--topic conditional-random-fields-crf --work content` was read; no destination note existed. The only inbox item was the resolved unrelated bitwise branch.

Retain the title: this page teaches the general CRF idea through a complete first-order chain, with enough advanced scope to distinguish it from all graphical models. No identity, order or title migration is needed. The real data changes from invented repeated NER sentences to actual coarse POS tagging; BIO remains a supported explanatory example for constraints, not an unmeasured NER benchmark.

| Existing material | Retention / correction |
| --- | --- |
| Generative/conditional contrast, rich features, HMM bridge | §§1–2 preserve the mechanism and local input-availability contract. Remove claims that HMM observations must be single categorical tokens or that one model always dominates. |
| Log-linear distribution, partition and feature functions | §§2–4 retain equations and add complete four-path arithmetic, global score-shift invariance and the exact independent-classifier special case. |
| Forward, backward, Viterbi, edge marginals | §3 explicitly separates summing prefixes from maximizing one. Original “forward trace” used max scores and unsupported numbers; new exact table is independently enumerated. |
| Conditional training, L2, from-scratch code | §4 plus §7 supply complete code, shapes, output and regularization convention. Correct regularized moment matching, convexity conditions and optimizer convergence claims. |
| Library workflow and feature inspection | §7 points to inspected CRFsuite tutorial/API after an executed SciPy optimization route. Remove fabricated perfect predictions, repeated-sentence “realistic data” and fabricated feature weights. |
| Neural output layer | §8 retains tensor flow, masks, transition conventions, encoder gradients and an annotated complete official PyTorch route. Original incomplete training sketch is replaced by a fully explained architecture and direct runnable alternative, since implementing a second neural framework is outside this scoped CRF core. |
| Label bias and legal BIO outputs | §§5–6 give separate constructed examples. Correct local-normalization generalizations and the false claim that `all_possible_transitions=False` masks illegal BIO edges. |
| Complexity, feature explosion, alternatives | §§8–9 retain dense chain cost, sparse/higher-order mechanisms and general inference. Remove invented throughput, fixed sample-count thresholds and universal library rankings. Feature vocabulary is training-derived; regularization and feature choice are assessed through development experiments. |
| General/latent/semi-supervised scope | §9 develops chain versus loopy inference, segment models, partial labels and why ordinary unlabeled conditional likelihood is zero. Correct “cannot generate sequences”: conditional label sampling is derived, while input modeling remains absent. |
| Reference history / literature | Retain substantive canonical and neural sources. Drop unsupported historical supremacy and score-gain anecdotes. Correct Lample architecture to character BiLSTM rather than character CNN. |
| Exercises | Five changed calculation, gradient, diagnosis and real experiment tasks with closed hints and explained solutions. No copying the core arithmetic inputs. |

## Learning contract and hurdle map

Intended learner: familiar with basic arrays, sums/products, probability normalization and the preceding HMM/Bayesian graph chapters; no CRF knowledge assumed. Refresh each random variable, score, marginal, feature and recurrence locally. The module order is unchanged. Link to exact prior HMM/Bayesian-network routes and next GP route; content is self-contained enough to use the local algebra if a prerequisite lesson is only planned at runtime.

| Outcome / hurdle | Teaching mechanism and example | Representation / evidence | Route |
| --- | --- | --- | --- |
| Identify observed and predicted variables | Words fixed, labels uncertain; whole input can feed local output factors | Input strip and factor graph F1; information-contract practice4 | Core |
| Normalize a structured score | Four path masses3/24/1/2, Z30 | F2 path bars and grouped marginals; changed practice1 | Core |
| Distinguish sum, max and marginal decisions | Forward4/26 versus best3/24; marginal-mode counterexample .35/.34/.01/.30 | I1 editable trellis, F3 join; practice3 | Core |
| Explain one learning update | Observed-minus-expected AB1−.8=.2, correct NLL sign and L2 | Count balance F3; changed practice2 | Core |
| Locate label bias | Two branches/private denominators cancel later compatibility | I2 local/global normalization, unequal/equal factors | Core |
| Enforce a real output rule | BIO legality mask, finite penalty counterexample | F4 automaton; practice4 | Core |
| Evaluate a real fit | Training-only DictVectorizer; fixed independent/chain comparison; development selection | F5 actual EWT errors/counts; independent practice5 | Core/application |
| Connect neural and graphical extensions | Shapes, masks, gradients, higher-order states, segments and loops | F6 architecture/scope; explained deeper readiness | Deeper |
| Distinguish output sampling from input generation | Backward conditional sampling from messages | Derivation §9; prerequisite for advanced extension | Deeper |

First pass directly after introduction: §§1–6 and practices1–4, then real program§7 and practice5; §§8–9 explicitly deeper. Primary anchor is span tagging, with simplified A/B factors for arithmetic and real coarse POS for evaluation. Original form-processing scenario illustrates statistical preference versus hard rule; it is labeled a teaching scenario, not a deployed application claim.

## Canonical-reference coverage check

Canonical source: Sutton & McCallum, *An Introduction to Conditional Random Fields*, freely hosted author PDF. Read the complete contents list, modeling introductions and §§2.1–2.3 relevant definitions, §4.1 recurrence passages, §5.1 likelihood/gradient passages, and §§6.1.3/6.2 distinctions. This is a scoped content review, not a claim to have read every line of the109-page tutorial.

| Canonical sections / headline | Decision and learner location |
| --- | --- |
| §2 graphical modeling, generative/discriminative, linear chains, general CRFs, features | Core §§1–4; general factor scope in§9. Exact output graph versus observed input dependence is explicit. |
| §2 examples/applications/terminology | Core span example, actual POS tagging, deeper segments and repeated mentions. No unsourced industry list. |
| §3 algorithm overview; §4.1 forward/backward/Viterbi | Full core derivation and code, including choice-of-loss distinction. |
| §4.2 general inference; §4.3 implementation concerns | Log stability core; trees/loops/treewidth and approximate methods deeper. No generic exact-inference promise. |
| §5.1 maximum likelihood; §5.2 stochastic methods; §5.3 parallelism | Full batch optimizer route and gradient; stochastic/parallel implementation engineering remains optional reference because the scoped program is small. The linearity/covariance argument is given, with neural/latent exceptions. |
| §5.4 approximate training; §5.5 scaling | Deeper explanation that approximate inference and training interact; higher-order/sparse state complexity. No benchmark plots without measurements. |
| §6.1 structured learning, neural nets, MEMMs | Label bias gets its own investigated counterexample; neural flow and structured-SVM distinction are retained. |
| §6.2 Bayesian, semi-supervised, structure learning | Explain parameter integration and ordinary-unlabeled objective limitation; partial/latent labels included. Full Bayesian integration and learned graph structures remain canonical-reference branches rather than an unimplemented advanced library project. |

## Research and claim locators

All retrievals below occurred 12 September 2026. Read primary content through the web tool. No video was watched or endorsed from metadata alone. Useful written alternatives are included in the learner-facing references with level and purpose; there is no required video quota.

| Claim / resource | URL and exact locator | Actual assessment |
| --- | --- | --- |
| CRF factorization, arbitrary observed features, chain inference/training | https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf — contents pp1–2; modeling §§2.1–2.3; inference§4.1; estimation§5.1 | Definitions and recurrences inspected; our numeric fixtures/code are original calculations. |
| Label bias information path and feature caveat | Same PDF§6.1.3, printed pp357–359, backward MEMM recurrence6.5 | Supports local cancellation counterexample and explains why the directed/undirected contrast alone is insufficient. |
| Latent/neural nonconvexity, Bayesian and semi-supervised distinctions | Same PDF§6.1.2 and§6.2, printed pp356/359–361 | Scoped research supports corrected boundaries; no claim of implementing these variants. |
| Alternative full tutorial chapter | https://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf — introduction, §1.2 and model factorization | Older author chapter inspected for scope/cross-check, not used as duplicate learner link. |
| Transition feature generation versus legality | https://sklearn-crfsuite.readthedocs.io/en/latest/api.html — `all_possible_transitions`, `all_possible_states`, `c1`, `c2`, `predict_marginals` | Docs say feature generation, not BIO legality. Documentation header0.3; absent local package, no invented installed-version claim. |
| Library feature workflow | https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html — dictionary extractor, fit, evaluation and weight inspection | Substantive page read; its dataset and outputs are not copied into our experiment. |
| Complete alternate neural code | https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html — CRF explanation and `BiLSTM_CRF` implementation | Written code reviewed; tutorial has historical last-update/verification dates even though current site chrome reports2.14.0+cu130. Neural execution deferred. |
| Neural character representation | https://aclanthology.org/N16-1030.pdf — §2 chain architecture, §4.1 and Figure4 character BiLSTM | Corrects old CNN attribution; no paper benchmark number reused. |
| EWT release, annotation and attribution | https://universaldependencies.org/treebanks/en_ewt/index.html; https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.16/README.md — description, License/Copyright, metadata | Exact release extracted offline; underlying-text rights notice preserved in provenance rather than claiming broader ownership. |

The old founding-paper URL at `cis.upenn.edu/~pereira/papers/crf.pdf` failed retrieval. It is not used to claim a newly verified historical theorem or six-page article length. The canonical author tutorial resolves the necessary model claims. A failed ACL dataset-paper lookup was replaced by the maintained treebank and exact release README; no source was fabricated.

## Author checks and full learning-experience pass

Executed `scratch/lesson-tools/Scripts/python.exe docs/teaching/drafts/conditional-random-fields-crf/author-calculations.py` in the existing read-only environment: Python3.12.14, NumPy2.3.5, SciPy1.18.1, sklearn1.9.1. The retained JSON records full results. This is bounded research arithmetic/model evidence, not formal phase-two implementation verification.

- Exact four-path enumeration equals the forward partition30; node sums equal1; AB logZ derivative from a centered finite difference is.79999999980096 versus edge marginal.8, absolute error below1e−8.
- An additional bounded probe compared unary second-A factor1,6,12 under zero and learned pair interactions: winners agree/disagree/agree as specified; Z pairs12/30,32/50,56/74. Equal-factor and changed-prior label-bias nulls are exact algebraic substitutions. Changed practice masses8/4/6/12 and updates were hand-derived.
- Real independent/chain fits both converged; fixed development selection favors chain, and final selected test output is293/370tokens and7/40sentences. Full confusion matrices, IDs and probabilities retained. No extra seed search or feature tuning was done to make a preferred method win.
- Read the **complete new manuscript** in three bounded chunks. Checked beginner sequence, displayed equations/program continuity, changed practice and answers. One reconciliation removed a proposed input-logit display because that data was not retained; the real figure now uses actual stored marginals and pair scores only. Removed authoring-only video prose from the learner conclusion. Fixed dense spacing in the visual handoff to keep future implementation legible.

Learning-experience findings, distinct from numerical correctness:

1. Route is directly after intro and deeper labels/readiness match the optional sections.
2. Cautions have one home: input availability§1, normalization/decision type§§2–3, convexity§4, label bias§5, legality§6, small-data/split protocol§7. No code prints disclaimers.
3. A person-span question introduces why output labels interact; real web annotations provide a concrete held-out result and errors.
4. Two investigations have different learning jobs. I1 edits actual unary/pair factors; I2 edits route compatibility and prior. Predictions are initially unset/input-bound; fixtures include actual reversals and meaningful nulls. No fixed lab quota.
5. Static diagrams appear at input/factor introduction, path normalization, messages/counts, legality, real errors and neural flow. Quantitative bar scales and zero baselines are specified; actual desktop/mobile visibility is deferred, not marked passed.
6. Grouped path marginal and forward/backward result are explicitly connected; independent softmax is derived as a special case. Canonical scope decisions above retain the field's main distinctions.
7. Complete instructional programs expose inference/training and use one small real-data file. Pathological constraints are explained outside the displayed core. A small inference/data interface difference in author code is disclosed in provenance.
8. Practice changes numbers, gold path and context. The real exercise requires an unsolved feature ablation, stated prediction, counts and interpretable changed errors, with an exact baseline to reproduce.
9. Screenshots, browser operation and rendered states do not exist in content-only mode. They are explicit finish requirements.

No material content question remains for this checkpoint. Formal independent review may still identify corrections in phase two; the author reread is not independent review. Root's cross-topic reconciliation may return a content finding before checkpoint freeze.

## Files, retention and exact next action

Retain `lesson.md`, `visual-specifications.md`, `design.md`, `ewt-sequences.json`, `data-provenance.md`, `data-sources.json`, `author-calculations.py` and `checked-results.json` as the content packet. Remove only the local import cache if produced; no full corpus, temporary script, screenshots or installed package were retained. Runtime sources remain unchanged.

When a finish request is authorized, run the stable-ID `--work finish` preflight and read this complete packet. Build topic-owned figures/labs/models and the real reader lesson, preserving current route/module context. Execute assembled displayed programs and any chosen library alternative, complete legality/numerical/model checks and formal independent correctness plus learning-experience review, then actual responsive/keyboard/browser closure and integration. Preserve or refresh changed content hashes rather than treating this author's result as publication approval. Until then, implementation stays not started.
