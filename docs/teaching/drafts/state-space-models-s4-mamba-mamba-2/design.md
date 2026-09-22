# State Space Models — design, research and author record

Canonical ID: state-space-models-s4-mamba-mamba-2. Module position 18; preceding Long-Context Sequence Models, next RWKV & Linear Attention Models. Research/write packet prepared; implementation has not started. Root owns the shared ledger and checkpoint.

## Scope, baseline and ownership

Read the current authoring entry point, teaching standard, design brief, ML domain guidance, ownership/coordination/retention rules and this increment's scope. The actual scoped command was:

```text
node scripts/build-curriculum-inventory.mjs --topic state-space-models-s4-mamba-mamba-2 --work content
```

It returned the existing published lesson, in-progress content phase and the ODE destination note. Read the complete 1,220-line original in four untruncated chunks. Baseline commit: 8c5da59f18516be77c29d5aeeafca3decca4f738. Original source: src/learn/data/topics/state-space-models-s4-mamba-mamba-2.jsx. SHA256: 790a636f3c00b6eac5d9009c071c82f6caed6e66a658be9a36e3e8707c98177d. Production source remains unchanged.

The current official Mamba repository identified the March 2026 Mamba-3 paper. A scoped search of topic titles, curriculum documents and notes found no separate Mamba-3 owner. Root approved a proportionate extension within this topic. Suggested display title: **State Space Models: S4 and the Mamba Family**. Keep the stable ID; no runtime title/catalogue mutation was performed. The extension explains three mechanisms and their boundaries, without turning the packet into a second full model implementation.

The existing ODE note is accepted: correct the double feedthrough, include initial-state response, and use a singular-safe held-input integral. Its disposition is content addressed / phase-two integration pending. Neural ODEs will own nonlinear continuous-depth solvers and their training; this packet owns the linear continuous/discrete sequence bridge and model-specific conventions. Self-attention owns row-softmax attention; this packet's SSD duality is explicitly a different, unnormalized structured bilinear operator. RWKV owns its own weighted-memory recurrence. Jamba/hybrid architecture decisions remain with the later dedicated topic.

## Learner, outcome and teaching route

A beginner at this point in the module has encountered recurrent models and attention but has not yet studied the later detailed transformer lessons. Local refreshers define state, matrix shapes, write/retain/read paths, softplus/sigmoid, outer products, logits/loss and per-step versus whole-record outputs. The linear-algebra-heavy HiPPO/DPLR and newer-family material is clearly optional.

Core route: sensor smoothing → state paths → continuous sampled dynamics → recurrence/convolution equality → useful memory modes → content selection → SSD matrix/chunk computation → complete real-data training → practical diagnosis → changed practice. The running practical task is a movement trajectory whose temporal order matters. Exact small examples explain each operation before the real classifier combines them.

| Hurdle | Teaching response | Observable check |
|---|---|---|
| State is an unexplained black box | Four update/read paths and scalar calculation | Identify old-state, write and feedthrough contributions |
| Continuous A confused with discrete Ad | Held interval, exact integral, bilinear comparison and integrator | Compute both coefficients; A=0 remains valid |
| Convolution seems like a different model | Unroll recurrence and align impulse contributions | Recurrent/direct/FFT agree, including initial response |
| Long memory means one almost-unit decay | Several timescales and rotating real coordinate pair | Explain a decaying oscillation and a memory coefficient |
| HiPPO means universally optimal memory | Explicit measure/basis projection and 1/t dynamics | Reconstruct a linear function; state the approximation criterion |
| LTI allegedly cannot copy | Exact shift-register counterexample | Fixed delay works; marker-dependent gap is a different task |
| All gates/discretizations are conflated | Exact scalar gate versus actual ΔB scan and full block | Calculate a substantial injection difference |
| SSD is called ordinary attention | Signed influence matrix and exact state/matrix equality | Negative coefficient and no softmax normalization |
| Chunking loses history | Four-stage local/carry decomposition | Uneven and changed chunk sizes preserve the output |
| New model supposedly wins by default | Predeclared small fits and ordered baseline | Report weaker neural outcomes without tuning them away |
| Diagram is mistaken for measured speed | Explicit bytes and actual loss/error data | Distinguish count, equation illustration and measurement |

The visual specification provides 24 inline visual entries and four distinct investigations. Each has editable native entities, a recorded current live comparison bound to current inputs, checked contrasting and null fixtures, reset/invalidation, numerical contracts, accessible text/mobile alternatives and bounded phase-two work. First three lab defaults differ from worked examples. The real-data exercise edits actual coordinates/order. Nine changed practice problems have eighteen initially closed hint/solution disclosures.

## Original-to-new conservation

| Original useful coverage | New treatment and rationale |
|---|---|
| Continuous/discrete equations and kernels | §§1–3 retain and derive them with one indexing convention, initial response and feedthrough exactly once |
| ZOH inverse and sampling | Exact integral/block exponential valid for singular A; original S4 bilinear choice retained distinctly |
| S4/HiPPO history, matrix structure, DPLR/Cauchy | §4 projection example, normal-plus-low-rank basis, finite generating function and rank-one Woodbury derivation |
| Oscillatory LTI and FFT program | Complete state_space_mechanisms.py with recurrent/direct/FFT equality and explicit padding |
| S4D-lite program | Complete trainable diagonal complex-mode classifier with exact Bd/conjugate read; honestly labeled S4D-style, not full DPLR S4 |
| Selective-copy learning motivation | Fixed-delay counterexample plus editable selective-memory challenge and real trained selective mixer |
| Mamba block and reference use | §5 actual scan parameterization/full branch diagram; official code and environment boundary in §8 |
| Mamba-2 and duality | §6 shape-correct outer writes, signed influence matrix, exact four-step trace and chunk decomposition |
| S5 | §9 shared MIMO-state/scan bridge, scoped as a distinct family |
| Performance/memory discussion | Exact specified-array byte counts and actual measured small-fit losses/errors; discard unsourced speed/ranking curves |
| Gotchas | Compact causal/index/state/numerical/evaluation diagnostic table |
| Original five exercises/resources | Nine independently changed problems and annotated primary/creator/spoken alternatives |
| Newly relevant Mamba-3 | Optional two-endpoint, complex rotation and rank-R write/read extension, with dated provenance and scoped guarantees |

Removed or corrected claims include double Du, implicit zero initial state, singular A inverse, ZOH presented as S4's universal choice, fixed-delay copying said to be impossible for LTI, nonlinear whole stacks equated with a single LTI layer, unspecified conjugate/Bd details, malformed SSD values, invented speed curves/current universal rankings, incomplete byte accounting and unconditional dtype/initialization claims. No useful topic was deleted to shorten the article.

## Actual research retrieval and canonical coverage — 13 September 2026

These entries state what was actually read. No embedded recording was watched, no large checkpoint was downloaded, and no CUDA kernel was executed.

- [S4](https://arxiv.org/pdf/2111.00396): background §§2.1–2.4, especially the bilinear formulas and explicit h[−1]=0; full relevant mechanism §§3.1–3.4, including nonnormal conditioning, unitary normal-plus-low-rank representation, DPLR/Woodbury/Cauchy kernel construction, channel mixing and nonlinear stack. Experiment organization and selected context inspected; the full appendix/benchmark suite was not replicated.
- [HiPPO](https://arxiv.org/pdf/2008.07669): §2 approximation, measure and basis; §2.3 translated measures and §2.4 discretization; §3 scaled Legendre theorem with 1/t factors and its approximation/timescale qualifications. §4 experiment organization and reconstruction/character-trajectory context inspected. The 47-page proof collection was not read in full. Local polynomial and normal-matrix identities were independently calculated.
- [S4D](https://arxiv.org/pdf/2206.11893): §3 discretization, Vandermonde computation, stability and conjugate-symmetry choices; §4 normal-HiPPO/diagonal approximation and the large-state limit distinction. The small program's chosen linear-frequency initialization is stated directly, rather than claimed to reproduce every paper setting.
- [S5](https://arxiv.org/pdf/2208.04933): §§3.1–3.4 cover SISO banks versus shared MIMO state, normal-HiPPO parameterization, parallel scan/irregular intervals and computational comparison. The manuscript retains the mechanism without a first-ever scan or universal speed attribution.
- [Mamba](https://arxiv.org/pdf/2312.00752): §2 continuous/discrete parameterization; §§3.1–3.6 selection motivation, fixed versus variable-spacing copy, algorithms/shapes, fusion/scan/recomputation, block, scalar gating and initialization discussion. Appendix C's exact gate derivation read. Experiments/limitations headings and selected mechanism-ablation context inspected; no universal 2026 quality claim imported.
- [SSD/Mamba-2](https://arxiv.org/pdf/2405.21060): canonical section structure; §§5.1–5.3 scalar identity, structured masked attention and duality; §6 block algorithm; §§7.1–7.2 projection/normalization and head patterns; §9 experiment organization. The packet develops the exact core operator and algorithm; specialized distributed implementations and the full proof collection remain advanced source follow-up.
- Creator [SSD Part I](https://tridao.me/blog/2024/mamba2-part1-model/): scalar transition, state/head shapes and training/inference tradeoffs. [Part III](https://tridao.me/blog/2024/mamba2-part3-algorithm/): full four-step code explanation, the minimal dense interchunk shortcut, cumulative-product ratios, log-sum cancellation, stable segment sums and discrete-parameter discussion. A displayed formula in the article's discretization section appears to have an extra nested exponential; the manuscript uses paper/reference-verified exp(ΔA), not a copied typo.
- [Official repository](https://github.com/state-spaces/mamba): relevant README installation/model/pretrained/precision sections, inspected on main on 13 September 2026. Core installation versus opt-in compiled scan support is date-located, not falsely release-pinned. [selective_scan_interface.py](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py), reference function lines119–183: float32 handling, exp(ΔA), ΔBu, recurrence/read, Du once and optional SiLU gate. [mamba_simple.py](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba_simple.py): initialization, projections, convolution, forward and step boundary. [mamba2.py](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba2.py): parallel z/x/B/C/dt projections and grouped shapes.
- [Mamba-3](https://arxiv.org/pdf/2603.15569): methodology §§3.1–3.4, including exponential-adjusted integral/two-endpoint coefficients, second-order conditions, complex-to-real rotation and accumulated write/read rotations, MIMO state/write/read dimensions, arithmetic intensity, training/chunking mechanism and architecture changes. Experiment section organization and stated comparison settings inspected. The manuscript includes no new claimed benchmark reproduction, no guarantee for arbitrary learned λ, and no false statement that every real-valued matrix cannot rotate.
- [UCI Libras](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement): current license and dataset metadata read, along with the full original names file. Byte-original files were reused from the self-attention author's documented official download; see data-provenance.md. Exact duplicates, roles and results were computed locally.
- [Albert Gu's hosted conversation](https://www.cognitiverevolution.ai/the-state-space-model-revolution-with-albert-gu/): verified host title/date, embedded-video link, chapter list and transcript. Read the historical introduction and detailed state, memory-hierarchy, training/inference and Mamba-2 comparison transcript around 29:30–50:03. The learner resource points to the host and relevant chapters. The recording itself was not played.
- Annotated S4/S4D website attempts returned reader errors; they are not claimed as read. A CMU seminar listing was read but its recording link did not resolve usefully, and a candidate YouTube page failed retrieval. These attempts were replaced by the verified creator articles and hosted transcript; no broken direct video URL was added to the lesson.

Canonical coverage decisions are explicit: teach the core mathematical mechanisms and exact small calculations fully; retain source paths for longer proof collections and hardware implementations; distinguish those from the executable small learning study. No canonical model family or mechanism named in the original scope was dropped.

## Author checks and retained evidence

Executed state_space_mechanisms.py using the shared read-only Python runtime with NumPy/SciPy. Retained mechanism-results.json checks:
- Correct ZOH recurrence/direct/FFT agreement; singular integrator, direct-only path and nonzero initial-state response.
- Damped rotation, N=4 HiPPO normal correction, degree-one reconstruction, fixed-delay shift register and exact sigmoid/ZOH identity.
- Selective versus constant gates and zero-input null.
- SSD recurrent/matrix/chunk equality for q=1,2,3,4,8, future edit, zero writes and reset-before-row.
- Mamba-3 two-endpoint 4 versus 6.5, parity rotation, and specified state/cache bytes.

Executed trajectory_state_models.py with four complete small CPU fits. Seeds and budgets were predeclared; no retraining to obtain a desired ranking. Diagonal seeds17/41 selected epoch100; selective selected58/68. Assessment errors30/60 and28/60 for both kinds, baseline22/60. Saved all histories, roles, confusion matrices, selected weights and logits. Corrected one metadata label after fitting: a repeated selective serial forward is not an independent FFT comparison. This edit did not alter weights or metrics.

Executed author_calculations.py to verify the readout gradient using autograd, independent finite generating-function and Woodbury equalities, conjugate read/rank fixtures, changed unsolved lab defaults, coordinate edits/reversal/reset and causal-prefix checks on frozen fits. A temporary local function-name shadowing error in that probe was fixed before its successful run. This did not affect model training. Finite-kernel discrepancy1.11e−16 and Woodbury6.21e−17 in their tiny complex examples; diagonal full-model FFT/recurrence discrepancy below4.8e−6. Complete arrays are in investigation-checks.json.

Performed a full author reread of the manuscript in sequence and the complete visual contract. The reread corrected a subtle wording defect: input-dependent nonlinear dynamics can remain time-shift-equivariant, so input dependence must not be described as automatically breaking time invariance. Added the actual polynomial integral, finite kernel/Woodbury mechanism and local activation explanations. Checked independent practice arithmetic and reconciled every displayed training count with retained results. Verified local download targets, lesson routes, data/source hashes, record dimensions, duplicate roles, finite outputs and closed disclosure pairs.

## Learning-experience closure and deferred work

- Ground-up intuition precedes notation; every key operation has a local mechanism explanation and a visual form that matches it.
- Worked examples progress to changed learner inputs and changed practice, not answer replay. Real-data evidence is a substantive supervised task with objective, update, baseline, split and complete program.
- Core versus optional depth is explicit; S4/HiPPO, S5 and Mamba-3 detail is preserved without becoming a prerequisite blockade.
- Accuracy nuances are taught where needed: initial conditions, feedthrough, exact versus reference discretization, nonlinear/time-invariant distinction, signed SSD coefficients, row-level data claims and measured versus hypothetical behavior.
- Reset/null/contrasting inputs, meaningful live controls, shape labels, text alternatives and bounded evaluation are specified for every investigation.
- The immediate Long-Context→SSM→RWKV route uses actual local stable-ID links; later attention and hybrid ownership links supplement rather than replace module sequence.
- Primary papers, creator articles, source code and a verified hosted video/transcript provide alternative learning routes with honest retrieval notes.

Phase two must implement/export the figures and investigations, verify exact browser forward parity and all live result/reset/input lifecycle paths, check current optional-kernel setup if that route is included, perform independent content/code review, accessibility/responsive/performance/native integration checks, and update runtime publication/status. These tasks have not been performed or mislabeled complete. Necessary source data, learned arrays, manuscripts, specs and evidence are retained; only disposable own cache may be removed. No shared ledger, production code or curriculum files were changed.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Change a state-space write and read. Edit tiny system coefficients, impulse inputs, selective writes, distraction sequence, chunk boundaries and supported real trajectories. Show impulse response, carried state, input-conditioned updates, SSD matrix entries and chunk equivalence live. Stepping exposes current recurrence arithmetic. Decide which information needs selection or state carry and distinguish a mathematically equal scan from a different update rule.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Connect the recurrence to the maintained scan and complete block” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| LTI discretization, recurrence/convolution and learned diagonal mixer | `state_space_mechanisms.py` and `trajectory_state_models.py:DiagonalMixer`; exact published ODE/LA reuse | SciPy expm/solve, NumPy/Torch FFT and actual Torch learning loop | Existing singular/initial/direct-state fixtures and derivatives; full S4 DPLR kernel not advertised |
| Selective recurrence and ordinary complete Mamba blocks | `trajectory_state_models.py:SelectiveMixer`; explicit exp(ΔA), ΔBu and C readout | `state_space_library_bridge.py`; same scan values/gradients and Mamba/Mamba2 one-update workflow | Input/layout/softplus/last-state-gradient mapping; GPU route fully written but unexecuted |
| SSD recurrence, matrix and chunk decomposition | `state_space_mechanisms.py:ssd_recurrent, decay_matrix, ssd_chunked` | Dense algebra as independent comparison; Mamba2 ordinary block is different full model | Nonzero initial-state length7/chunk1,3,8 extension and exact carry solution |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.
