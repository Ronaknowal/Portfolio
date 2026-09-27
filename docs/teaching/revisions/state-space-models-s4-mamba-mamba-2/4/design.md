# State Space Models — teaching revision 4

Topic: `state-space-models-s4-mamba-mamba-2`, Deep Learning position 18. This revision responds to the user's report that the implemented lessons lacked intuition, buildup and readability. It changes the teaching route throughout, while keeping the tested numerical operators, complete programs, fitted parameters and real data unchanged. Original prepared material and completed revision-3 records are historical evidence, not overwritten sources for this revision.

## Diagnosis and intended learner experience

The earlier version defined a state and quickly introduced matrices. Correct plots often followed their derivations; optional HiPPO/S4 details still interrupted the main reading sequence. A reader could verify equations without having a clear picture of why the next mechanism was needed. The revised lesson repeatedly follows problem → inadequate simple choice → inspectable operation → worked values → terminology/equations → useful experiment.

The sensor example uses deviations `[5,0,0]` and explicitly treats zero as a real observation. Latest-only storage, complete history and a running mean are legitimate alternatives with different information tradeoffs; the lesson does not falsely say a running mean requires growing storage. The retention rule is a designed illustration, not fitted sensor evidence. The learner manipulates retained fraction before seeing the recurrence symbols.

The first-pass route identifies actual sections and labs, followed by an implementation pass. Singular/bilinear discretization, continuous impulse response and HiPPO/DPLR are complete expandable branches. These branches retain their numerical assumptions; essential state-update conventions, ZOH meaning and Mamba's distinct injection remain in the visible core.

## Revised route and conservation map

| Conceptual hurdle | Revised teaching decision | Retained depth and evidence |
| --- | --- | --- |
| What does a state number mean? | Inspect retained/new contributions before naming h and u; work 1,.8,.64 and compare three summaries | General vector shapes, separate feedthrough, update-before-read, LTI definition and finite-memory tradeoff |
| Why distinguish a rate from a step? | Start with approaching a held reading; show exact curves and samples before ODE; connect .8 retention to −ln(.8) | Singular-safe block exponential, expm1, bilinear comparison and stability conditions in a return branch |
| How can convolution equal recurrence? | Add a simpler impulse-trail view, sum selected columns, then derive kernel taps | Two-mode original example, nonzero initial response, continuous distribution/feedthrough convention, independent direct/FFT algorithms and padding |
| Why several state coordinates? | Fast disturbance and slow background; decay trails before half-life; rotating real pair before complex modes | Complete S4D-style parameterization/conjugate factor; polynomial reconstruction, exact 1/t HiPPO, normal correction, finite kernel and Woodbury/Cauchy derivation retained |
| Why make the update content dependent? | Marked calibration values versus routine messages; compare marked memory, a fixed delay and constant smoothing | Fixed-delay counterexample, supplied versus learned gates, exact scalar ZOH identity versus Mamba ΔB injection, full block, associative scan and fusion |
| What is SSD doing? | Multiply a 2×2 retained matrix, write an outer product and add rows before naming N,P and operator equality | All four worked updates, signed/nonsoftmax influence, shared scalar restriction, four chunk stages, uneven/changing chunk investigation |
| How does any of this learn? | Return to temporal order in real movement; connect data shapes, loss arrows and code terms to prior diagrams | Complete duplicate-safe study, all runs/baseline, real curves/confusions, native parity, frozen validation interventions and training program |
| What should I implement myself? | Explicit mechanism → trainable scaffold → maintained operator route, with code-reading passes | All three canonical programs, gradient/output comparison contract, CUDA execution limitation and changed-code exercise |
| How do newer variants connect? | S5 changes sharing; Mamba-3 changes endpoints, rotation and write rank | All original equations, assumptions, examples, scope limits, nine end exercises and annotated resources retained |

No model was retrained to improve the narrative. No comparison result, held-out role, or deep mechanism was removed to shorten the page. The original canonical Python code and public model/data assets are reused, not duplicated into this packet.

## Research actually reviewed for this revision — 26 September 2026

These are teaching influences and verification sources, not copied text or diagrams. Previous canonical technical research remains recorded in the original design and revision-3 implementation record. New toy examples and visual geometry are authored locally.

- [Rush and Karamcheti, The Annotated S4](https://srush.github.io/annotated-s4/), reviewed through its [published literate source](https://raw.githubusercontent.com/srush/annotated-s4/main/s4/s4.py), opening explanation, contents and initial discrete formulation. The HTML fetch exceeded the browser tool's size limit. Its separation of basic mechanisms from advanced structured-kernel implementation informed the return branches and code-reading route. We did not run its JAX/Flax code or claim a full tutorial execution. Unlike that source's equation-first opening, this lesson starts with visible retention arithmetic.
- [Mamba paper, §§3.1–3.2 and selection discussion](https://arxiv.org/html/2312.00752v2), read the compression motivation, fixed-spacing versus selective-copy distinction, input-dependent coefficients and gate interpretation. This motivates contrasting “last marked value” with “two steps ago.” Our calibration stream is an original teaching fixture, not the paper's measured benchmark or a deployed system. A nonlinear stack is not conflated with one fixed LTI operator.
- [Dao and Gu, SSD Part I](https://goombalab.github.io/blog/2024/mamba2-part1-model/), read the model/framework/algorithm distinction, scalar-identity restriction, state/head dimensions, exact duality and architecture separation. This informed the explanation of why sharing a decay enables a different computation. The lesson demonstrates a write/read by hand before introducing the attention-like equality; it does not treat scalar restriction as universal quality superiority.
- [Dao, SSD Part III](https://tridao.me/blog/2024/mamba2-part3-algorithm/), read the four chunk stages, explanatory code, interchunk dense shortcut, cumulative-product/log-sum stability and discrete-parameter discussion. This informed the local-versus-incoming interpretation and first-pass guidance. The article's displayed nested-exponential expression in its discretization discussion is not copied; the lesson retains the paper/reference-verified exp(ΔA) convention. No hardware throughput claim is imported into the CPU reference.

Each new research summary is intentionally bounded. The new lesson derives its original small examples locally; the full existing mathematical and empirical coverage keeps its prior primary citations and evidence.

## Delivery contract

Current prose: `lesson.md`; current visual decisions: `visual-specifications.md`. The generator reads this revision explicitly. Runtime additions use semantic topic-scoped files `StateSpaceIntuition.jsx` and `state-space-intuition.js`; existing operators remain in `state-space-models.js`. New arithmetic, conservation and source-identity checks write to this revision's `evidence/`, never into historical receipts.

Independent review must critique the learning route beyond correctness: does each new mechanism have a reason to exist, a visible operation, named quantities, an interpreted result and a useful next action? Browser review must inspect the new visuals and their surrounding prose at desktop, 760px and 320px, including functional retention/column controls, branch disclosures, local table overflow and existing lab access. Record completion and any limits in `implementation.md`. Parent owns shared ledger/build integration.
