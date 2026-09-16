# Neural ODE content design and continuation

Stable ID: neural-ode-continuous-depth-models. Deep Learning Fundamentals & Architectures position 38. Research/write only, 13 September 2026; implementation is not started. Root assumed this untouched packet from the probabilistic author by explicit coordination.

The actual content preflight returned no destination note; the unrelated resolved inbox did not apply. The entire 1,052-line published source was read, recovering truncated output in contiguous follow-up reads. Original SHA256: f022724f84307be8059de01876003b5e8b4ce91329aa21410b5f6d9071e73cee. Baseline HEAD: 8c5da59f18516be77c29d5aeeafca3decca4f738. Preserve the current publication and stable identity.

## Scope, teaching route and title decision

Keep the existing title and topic identity. This is the natural home for learned vector fields, numerical realization, continuous/discrete sensitivity, topology and augmentation, plus an actual continuous-depth learning example. Preserve the original bridges into irregular observations, continuous density models, flow matching and structured scientific models. Dedicated normalizing-flow, rectified-flow and PINN topics own their complete specialized training pipelines; this lesson supplies meaningful mechanisms and direct links without pretending to master every application at once.

The learner's initial problem is a point following local arrows. Progression: state/derivative and initial conditions → residual/Euler connection → solver stages and error → adaptive decisions and stiffness → learning through a solver → exact-flow constraints and augmentation → actual Iris study → observation versus query → optional density/flow-matching/physics branches → changed practice. Twelve inline figure anchors support the explanation where needed. Six investigations use different geometric/time/state structures and have different fresh inputs from the solved exposition. The number follows the hurdles, not a quota.

Do not make a generic slider dashboard for every branch. The first uses a phase plane; the second an accepted/rejected timeline; the third a derivative comparison; the fourth a geometric lift; the fifth complete learned-state and probability traces; the sixth observation jumps and availability. CNF patches and paired velocity paths work as inline explanations without compulsory extra labs.

## Experimental declaration recorded before fitting

Use the existing corrected 150-row Iris CSV from the GMM packet, with UCI/CC BY 4 attribution and inherited provenance. Keep complete source rows. Collapse identical four-feature vectors for role assignment/scoring, verify consistent labels and record duplicate IDs. Fixed NumPy seed 926 stratifies unique rows per class: first 30 fit, next 10 validation, remainder assessment. Fit standardization mean/population SD only on 90 fitting rows. Measurements are not physical-time trajectories: integration time 0…1 is learned representation depth.

Compare four declared models: linear softmax; four distinct residual vector-field blocks; one four-dimensional field integrated by four classical RK4 steps; a six-dimensional zero-augmented field integrated the same way. Fields concatenate depth t, use one 16-unit tanh layer and a linear derivative output; classify with a linear three-output head. Residual blocks use h=0.25 and distinct parameters. No input embedding obscures augmentation. Report actual parameter counts, not matched-capacity claims.

Three seeds 13/37/61 per model, 12 fits. Full fitting batch; 300 AdamW updates, learning rate 0.01, weight decay 0.001, default betas 0.9/0.999; float64 CPU. Evaluate fit/validation at update one and every 25 updates; retain minimum-validation-CE checkpoint per run. Assess selected checkpoints only. No tuning grid, post-assessment rescue, dropout, BatchNorm, data augmentation or timing benchmark. Retain all curves and full selected weights. Diagnose input edits and solver changes with saved weights, without retraining. Keep uniform/constant-class and linear baselines.

## Conservation of the original learning agenda

| Original useful scope | Manuscript destination and repair |
| --- | --- |
| Residual networks, continuous depth and motivation | §§1–2: consistently parameterized field/refinement, shared weights and finite evaluation cost |
| Euler, RK4, adaptive steps and solver code | §§2–3: full stage equations, exact analytic comparisons and complete executable teaching integrators |
| Reverse-mode continuous adjoint and custom autograd | §4: derivation, correct parameter sign, discrete chain rule and exact-versus-numerical comparison; remove fragile parameter-mutating custom wrapper |
| Forward/backward memory and compute | §4/§11: components, ill-conditioned reconstruction, checkpoint alternatives and actual cost factors |
| Trajectory uniqueness and augmentation | §5: local Lipschitz/existence interval and linear-readout assumptions, constructive lift, coarse-order counterexample and finite-sample gaps |
| Neural-field training, spiral classification and curves | §6 replaces unsupported toy counts/curves with complete real-data 12-fit study, saved weights, baselines, all outcomes and native parity |
| Irregular times, ODE-RNN and latent ODE | §7: observation assimilation versus continuous propagation, encoder/posterior, reconstruction/prior loss, causal information boundary and timing-process extension |
| CNF divergence and trace estimation | §8: independent linear determinant example, signed trace probes, probe consistency and stochastic/numerical distinction |
| Flow matching and generation | §9: directly calculated conditional targets, marginal mean field, different objectives, endpoint-noise qualification and next-owner link |
| Scientific, Hamiltonian, PINN and event applications | §10: concrete mechanisms and what changes in their assumptions; full specialist pipelines linked |
| Failure modes, learning resources and practice | §11 consolidated diagnostic table; §12 nine changed problems with closed hints/solutions; §13 primary/article/documentation/video routes |

Consequential repairs: rotation speed is one radian/unit, not 90 degrees/unit. Fourth-order error is polynomial, not exponential. RK4 does not generally conserve energy exactly. A solver's local tolerance is not a final statistical accuracy guarantee. Stiffness is not simply high velocity or “difficult input.” The continuous parameter gradient has a positive forward-time integral; a backward accumulator uses a negative derivative with reversed limits. Continuous and discrete gradients need not match at finite resolution. Reversing a stable forward flow may be ill-conditioned.

Remove universal 5–10× runtime, 20–200 NFE, 2× backward-cost and total-constant-memory claims. Eliminate incomplete code with undefined helper functions, detached/freed graphs, dtype mistakes and parameter mutation. Preserve the ability to implement and learn the mechanism through complete programs. No source's empirical speed/quality comparison becomes a universal architecture ranking. No unsupported clinical deployment or modern-model recipe is retained.

The original Onken/Ruthotto arXiv ID 2005.13420 was investigated and is correct; do not list it as a repaired incorrect ID. The retained historical reference needed its actual numerical argument, not removal.

## Canonical-reference coverage check

The actual full main/appendix agenda of Neural ODE v5 was inspected: introduction; reverse-mode sensitivities; supervised classification/software/architecture/error-control/depth; CNFs and density matching/likelihood; latent time-series training/Poisson/irregular sampling; limitations; related work; conclusion; appendices on instantaneous density change, modern adjoints, full adjoint algorithm, autograd implementation, latent training and figures.

Disposition: §§1–7 cover the core architecture, solver, sensitivity and sequence machinery. §8 covers density change with a complete original linear proof/check and trace estimator. §11 covers limitations. Appendix sensitivity ideas receive an original product-rule derivation, not an unverified copied custom backward implementation. Full stochastic likelihood/VAE training and every original benchmark/appendix proof remain specialist source branches. The lesson's real classifier is an executed instructional experiment, not a paper replication.

The Augmented Neural ODE main agenda was also inspected: background → one-dimensional obstruction → flow limitations and NFE → augmentation/toy/image comparisons → scope/conclusion; appendices on nonintersection, one-dimensional proof, homeomorphism, region obstruction and experiment settings. We read the main representational argument, toy/image discussion and key uniqueness proofs. §5 preserves the necessary assumptions and uses an independent exactly solvable construction. Learned lift success and lower NFE are empirical possibilities, not theorems asserted for all models.

For other primary sources, only the declared sections below were consumed. Their entire paper was not implicitly reviewed because a link was opened. Coverage is supplemented with direct calculations and the existing topic's complete agenda; no content is dropped merely because a compact framework fails to mention it.

## Research record and actual extent

Retrieved/read 13 September 2026. Durable sources and claim locators:

| Source | Actual material read | Contribution and limit |
| --- | --- | --- |
| [Neural ODE v5](https://arxiv.org/html/1806.07366v5) | Full table of contents including appendices; introduction and reverse-mode algorithm; classification/error/depth; CNF core and density applications; latent timeseries and limitations/related work | Core framing/ownership. Appendix proof texts were not all read. Own calculations replace numerical examples. |
| [Augmented Neural ODE v1](https://arxiv.org/html/1904.01681v1) | Full agenda, main introduction through augmentation, toy/image/generalization discussion; Appendices A/B/C uniqueness/order/homeomorphism and beginning D region proof | Conditions and augmentation. No replication of image experiments or complete Appendix D/E review. |
| [Onken/Ruthotto v2](https://arxiv.org/html/2005.13420v2) | Introduction/background, §3 full gradient comparison, §4.1 numerical/extrapolation discussion through §4.2 entrance | Discrete objective, reconstruction, rediscretization and extrapolation. Its speed ratios are not reused as general claims. |
| [Latent ODE primary PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf) | Abstract/introduction; model background; §§3.1–3.5 including ODE-RNN, latent ELBO, observation-time process, batching and selection; §4.1/4.2 setup and beginning §4.3 | Observation/update/inference distinction and actual time-aware baselines. No clinical deployment or whole experiment replication claimed. |
| [FFJORD v2](https://arxiv.org/html/1810.01367v2) | Introduction, generative/CNF/adjoint background, scalable trace/bottleneck method, algorithm and experiments through VAE discussion | Fixed-probe stochastic trace reasoning. No appendix proof or external training execution claim. |
| [Flow Matching v2](https://arxiv.org/html/2210.02747v2) | §§2–3, conditional/marginal definitions and theorem statements; §4 Gaussian paths and diffusion/OT examples through related-work entrance | Direct regression targets and conditional versus marginal trajectories. Appendix proofs and all benchmark results not read. |
| [torchdiffeq README](https://raw.githubusercontent.com/rtqichen/torchdiffeq/master/README.md), [FAQ](https://raw.githubusercontent.com/rtqichen/torchdiffeq/master/FAQ.md) | README API/method/adjoint/event material through references; full short FAQ | Current callable order, solver options, 3/8-rule RK4 and practical caveats. No installation, official example run or full repository source review. |
| [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/) | Substantive documentation for checkpoint, forward, implicit and backsolve methods | Current discrete/continuous differentiation alternatives. No JAX execution or library ranking. |
| [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) | Method, output times, event, tolerance, Jacobian and vectorization documentation | API used in actual local SciPy1.18.1 calculation; web reference version can differ slightly. |
| [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) | Variables, rows/classes, correction notes, citation and license | Real-data attribution and scope; inherited CSV hash and local role audit are separately recorded. |
| [CVPR2020 organizer talk page](https://anucvml.github.io/ddn-cvprw2020/talk2.html) | Complete short title/speaker/abstract/video page | Annotated alternate video route. Recording not watched; no invented timestamps or claim of reviewed technical narration. |

The author also inspected Kidger's neural differential equations thesis abstract as a discovery lead; its full chapters were not used as verified coverage or quoted material. All prose, calculations, exercise variants and diagrams in this packet are original instructional constructions, not copied source figures or text.

## Executed evidence and continuation

neural_ode_study.py executed all 12 declared fits once. The unique-data split is 90/30/29; actual parameter counts are 15/671/179/251. Every nonlinear run selected update50, linear runs update300. Strong linear results, seed variation and imperfect augmented results remain visible. There is no timing benchmark.

ode_calculations.py reuses selected weights without training. It computes exact/finite rotation, adaptive attempted-step histories, an analytic slow trajectory with stiffness, exact/discrete/backsolve scalar gradients, inverse conditioning, augmentation, observation jumps, trace probes, conditional-velocity averaging and full real-input interventions. Independent NumPy reproduces saved PyTorch logits/states within8.9e−16; scalar and actual input derivatives agree with central differences within2e−8.

check_author_packet.py supplies final source/data binding, full saved-model metric reconciliation without refitting, inline-program execution, fresh/null/practice arithmetic and manuscript/specification structure checks. Its results are author evidence only. A full author reread and learning-experience assessment are recorded at closure after those checks.

Phase two must consume the entire packet, implement topic-specific figures and six investigations, preserve exact operators and state, extract compact lazy model/data assets, execute final displayed programs, perform independent correctness and learning-experience review, check browser/mobile/keyboard/accessibility/loading/error/performance behavior and integrate downloads/routes. A current content checkpoint is required before starting. No runtime publication, React component, browser campaign, production build or formal phase-two review occurred here.

## Completed author closure — 13 September 2026

The author reread the complete learner prose in bounded contiguous reads, the complete embedded study program against its executed source, the entire visual contract and the design/provenance. The final prose has a concrete opening, local prerequisites, a first-pass route and optional deeper branches. Nine practice problems use changed inputs, with 18 initially closed hint/solution blocks. The adjoint derivation and complete training program are separately closed optional details.

The reread corrected the PINN destination module to its actual frontier-research membership, clarified local versus global numerical error, and made the specification's later investigations/provenance readable rather than compressed notes. The full training program was added as an exact source-bound download/accordion, with setup and saved-weight reuse instructions. These were content/interface-specification corrections; no fit was rerun.

check_author_packet.py passed: original/data hashes; 90/30/29 disjoint unique roles; all 12 saved models' parameter counts, selected steps and fit/validation/assessment metrics with maximum discrepancy zero; exact inline RK4 operation parity; full displayed-program binding; 12 figure and six investigation anchors; 20 closed details; changed practice arithmetic; zero-state, constant-field and future-observation nulls. The earlier independent full-model and gradient calculations remain passing. Author evidence is saved in author-checks.json.

Learning-experience checklist: each core concept begins with a reason and local explanation; the solver's actual stages and errors are visible; adjoint objectives are named; fresh labs edit mathematical or real input entities and bind predictions to them; small probability differences receive a dedicated residual view; nulls explain invariants; the real study preserves strong simple baselines and unfavorable outcomes; no graph invents measured performance; applications include substantive conditional-density, observation, energy and event mechanisms; primary sources and alternate video/documentation routes have honest inspection limits. No outstanding content-author finding remains.

The packet is ready for a content checkpoint and a later authorized finish request. No runtime files, published lesson, catalogue identity, browser state, build or implementation-review status changed. No disposable topic scratch artifacts were created; necessary pending source data, weights, programs and manuscripts are retained.
