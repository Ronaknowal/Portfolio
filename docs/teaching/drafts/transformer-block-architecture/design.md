# Transformer Block Architecture — research/write design

Research/write only, 13 September 2026. Stable identity and title retained. Actual content preflight completed; no bespoke blueprint/destination note was returned. Entire original `src/learn/data/topics/transformer-block-architecture.jsx` was read at baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`, SHA-256 `8261cf488a1fc48c09c3bac9c867d79c0b4545643f5e959aee6c8456ea446f69`. Original publication remains untouched. Shared ledger/checkpoint belongs to root.

## Declared experiment, before fitting

Question: how do complete, otherwise matched pre-norm and post-norm blocks behave on real ordered hand-movement classification, and what exactly does a normalization/gradient diagnostic measure? This is not a large language-model architecture ranking. Reuse the previous self-attention packet's licensed Libras input, exact deduplication and seed-73 220/50/60 row split to make the learning sequence intelligible. Keep all 15 classes. Raw source records are 45 ordered x/y coordinate pairs, not linguistic sentences; performer/session IDs are absent.

Two independently initialized blocks, width 24, two heads, FFN width 48, exact GELU, biased dense maps and two affine LayerNorms per block. Add a fixed normalized time coordinate (-1 to 1) to each transformed x/y pair before the shared 3-to-24 stem. This locally defined time tag supplies order; detailed position methods belong to the next topic. Both comparison models use final LayerNorm(24) and a mean-pooled 24-to-15 head, deliberately controlling the final readout convention. Only placement inside both blocks changes. This is a matched architectural experiment, not a faithful original-2017 reproduction. Dropout zero, Adam .003, no decay, 180 full-batch epochs. Seeds 101/102/103; each seed gives identical initial parameter tensors across placements. Select each checkpoint using validation macro F1, then lower validation cross entropy, then earliest exact tie. Test once per declared fit after selection. Preserve all six outcomes, including failures; no result-driven retuning.

Illustrate seed 101 in both placements and the first test example of class 4 (source row 77), fixed before fitting. Store actual parameters and complete stage vectors. Interventions: reverse x/y while keeping time slots; jointly reverse x/y/time (permutation-invariant pooled null); reflect frame 23 x coordinate; append five (.75,.75) records with fixed original times and zero pad time, comparing correct padding-mask/valid-only-pooling to no mask. No browser training: phase two loads fixed actual weights and performs bounded inference for genuinely edited inputs.

Exact fixtures separately check one-token LN sum-loss null and nonconstant probe, LN/RMS shift behavior, two-token whole-block trace with editable actual matrices and zero-branch control, same-parameter public API agreement and initial VJP diagnostics at depths 1/4/12. Final affine-free LN is shared in the depth probe and the loss is a unit-length fixed random projection, not `output.sum()`. Random seed 53 for blocks,97 for inputs/probe. These diagnostics are observations at initialization, not proofs of trainability. Full final learning-experience review follows completion of the manuscript/specifications.

## Scope, sequence and conservation

Retain the stable title and identity. This is the best home for whole-block wiring, positionwise feature processing, gated FFNs, normalization placement, meaningful gradient diagnostics, task assembly and component accounting. The first-pass route is §§1–6 and exercises 1, 2 and 4; later derivatives, variant taxonomy and systems details are deeper branches.

The actual predecessor is Self-Attention & Multi-Head Attention. Its complete current packet provides Q/K/V, masks, heads, permutation and cache foundations; §1 refreshes the needed mechanism locally. The actual next topic is Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi). Fixed movement time tags are defined here before use; detailed rotations, positional biases and extrapolation belong next. Both bridges use stable curriculum links with module context.

The earlier normalization manuscript was read at its feature-vector, axes and derivative sections; its epsilon, positive-scale, common-offset and affine conventions match this packet. The earlier residual manuscript's identity, cancellation, shape/order and pre/post sections were also read. Those packets already state the relevant limitations correctly and remain frozen.

| Original publication area | New disposition |
|---|---|
| Historical architecture and model-family evolution | §§1–2 preserve original ReLU/post-norm and block/stack/task distinctions. §8 gives version-specific parallel and branch-output normalization alternatives instead of a universal current recipe. |
| Attention versus FFN, expansion and nonlinearities | §§1 and 4 include explicit feature-response/write-direction algebra, ReLU/GELU/SiLU/SwiGLU, editable matrices and changed practice. Attention is acknowledged as nonlinear; no collapse theorem or automatic neuron meaning is asserted. |
| Residual and normalization formulas, final norm, variance | §§2–3 and 7 give exact axes, epsilon, RMS versus L2, affine parameters, zero-branch controls, derivatives and covariance. Final normalization is a declared architecture choice. |
| Complete pre/post code and native API | §5 provides an executed same-parameter comparison for both placements. Defaults, cloning versus sharing, redundant normalization and dropout sites are corrected. |
| Gradient-depth experiment | Replace the invalid sum-after-LN diagnostic and typed curve with the exact null proof, a meaningful nonconstant probe at depths 1/4/12 and actual supervised fits. |
| Activation trace and heatmap | Preserve the visual intent using actual junction vectors and real two-block traces. Branch inputs, updates, sums and carried states remain separate; no fabricated intensity matrix. |
| Synthetic copying experiment | §8 retains exact source/target indexing and the fresh-data loss floor, with a changed independent exercise. Old unretained accuracy/loss numbers are not claimed reproduced. A full actually executed real-study program now supplies the complete training example. |
| Llama-like block, RoPE and gating | RMSNorm/SwiGLU receive complete executed component code and precise parameter accounting. The full block/model programs cover assembly. RoPE conventions and GQA/MLA cache details belong to the following owners rather than a misleading faithful-model mini example. |
| Latency/fusion/FP8 and width recommendations | §9 replaces universal speed/quality claims with exact MAC/FLOP/storage assumptions and measurement guidance. Rounded gated widths are not exactly parameter matched; no invented timing curve remains. |
| Depth/width, checkpointing and parallelism | §9 explains parameter sharing, attention matrix versus cache, recomputation and explicit up-column/down-row partitioning with a partial-output sum. Later Ring Attention owns detailed position-partition communication. |
| Failure cases and QAs | Cautions live at their mechanisms. Eight changed exercises have closed hints and explained solutions; code does not print disclaimer text. |

### Canonical-reference section coverage

The actual D2L 1.0.3 §11.7 section list, exercise list and relevant substantive body/code were inspected. Its sections are Model; Positionwise Feed-Forward Networks; Residual Connection and Layer Normalization; Encoder; Decoder; Training; Summary; Exercises.

| Canonical section/question | Local coverage or actual boundary |
|---|---|
| Model | §1 block/stack/task and §8 three information-boundary layouts. |
| Positionwise FFN | §4 algebra, feature circuit, gating and §5 code. |
| Residual and normalization | §§2–3 placement and exact operation; §7 derivatives and diagnostics. |
| Encoder | §5 complete tested block; §6 complete two-block real classifier. |
| Decoder | §8 causal target shift, cross-attention sources and cache consistency. Earlier sequence-to-sequence/Bahdanau lessons own full translation training; later Interleaved/Cross-Attention owns specialized arrangements. |
| Training | §6 real inputs, full program, optimizer, selection rule and held-out results. The book's helper-specific translation program is an annotated alternative rather than another required campaign. |
| Summary | Explicit first-pass route and immediate positional-encoding bridge. |
| Exercise: deeper models | §7 actual declared depth diagnostics and §6 matched fits, without turning one measurement into a theorem. |
| Exercise: additive versus dot-product scoring | Earlier Bahdanau/Luong and immediate self-attention owners provide detailed scoring; §1 supplies a local refresh. |
| Exercise: language-model layout | §8 causal layouts, target shift, copying entropy and changed practice 6. |
| Exercises: long-sequence limitations and efficiency | §9 counts, cache, materialization, checkpointing and partitioning with later owners named. |

Vaswani §3's actual list was read for the preceding topic and reused: Encoder and Decoder Stacks; Attention (scaled dot product, multi-head and applications); Positionwise Feed-Forward Networks; Embeddings and Softmax; Positional Encoding. Stacks, FFNs and task readouts are local. Attention is the immediate predecessor and full positional methods are next. Embedding/output-head sharing is included in count/readout discussion. This conservation decision does not authorize a catalogue audit.

## Learning hurdles and representations

| Learner question | Local bridge | Representation/activity |
|---|---|---|
| What do all the boxes do? | Position rows and feature columns retain the block interface | F1 communication lanes and dependency table |
| Why does moving a norm change the function? | Follow the exact bypass and branch inputs | F2 paired circuits and I2 actual junction vectors/zero control |
| What is normalized? | Feature spread versus distance from zero | I1 offset/scale ruler and a learner-built zero-mean null |
| Why an FFN after attention? | Contextual features become responses and signed updates | F3/I2 matrix edits, separate-token control and gated multiplier |
| How is this a predictor? | Explicit time tag, stack, pool, classifier and loss | Full program plus I3 actual movement failure and genuine edits |
| What does the gradient measure? | A scalar probe must be able to change | I4 analytic/finite-difference contrast and F5 declared VJPs |
| Why can the stream grow? | Covariance and branch/state distinction | Worked independent/correlated updates and actual RMS chart |
| Are all Transformers the same layout? | Task boundary and versioned wiring | F6 target shift and F7 actual alternative circuits |
| What costs computation or memory? | Tensor shapes, sharing and units | F8 derived counts, separate cache and partial-output sums |

Static explanations and four genuinely editable investigations have distinct jobs. No word or lab quota was used. Essential reasoning is in the manuscript as well as the future visuals; new edits compute the actual mechanism.

## Substantive research record

Research inspected 13 September 2026. This table records actual material read. Sources already read substantively for the immediately preceding packet are marked reused. The numerical teaching fixtures and real experiment are original local work; source wording and structure are not reproduced.

| Resource and locator | Actual reading and use |
|---|---|
| [Vaswani](https://arxiv.org/pdf/1706.03762), §§3.1–3.5 and 4–5 | Reused substantive architecture, attention, FFN, embeddings, position and training reading. Establishes historical ReLU/post-norm and interfaces. |
| [D2L 1.0.3 Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html), §11.7 | Read the actual complete section and exercise lists plus relevant Model, FFN, AddNorm, encoder/decoder PyTorch code, cache/source attention and training body. Canonical coverage is mapped above. Translation helper code and scores are alternate learning material, not our experiment. |
| [LayerNorm](https://arxiv.org/pdf/1607.06450), §§3 and 5.1 | Read feature statistics, independence between examples, affine parameters and invariances. Local epsilon and positive-scale qualifications prevent overgeneralizing ideal equations. |
| [RMSNorm](https://arxiv.org/pdf/1910.07467), §§3–4.2 | Read the formula, invariance analysis and derivative setup. Partial RMS in §5 was also inspected; its sampling approximation is intentionally excluded from this whole-block lesson. No published speed ratio is generalized. |
| [Xiong et al.](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf), §§3–4 | Read exact pre/post equations, final norm and loss placement, initialization assumptions, Theorem 1 and Lemmas 2–3. The uniform-attention/Gaussian setup and gradients near the output remain explicit; our local diagnostics do not claim to replicate the large experiment or theorem. |
| [Shazeer GLU](https://arxiv.org/html/2002.05202v1), §§1–3.2 | Read all mechanism equations, the two-thirds gated-width comparison, T5 setup and reported table. Supports parameter accounting and component implementation, without claiming an unrun universal quality improvement. |
| [Geva et al.](https://aclanthology.org/2021.emnlp-main.446.pdf), §§1–3 | Read persistent key/value algebra, unnormalized activation coefficients and the initial trained-pattern inspection method. The learner interpretation does not assume one neuron equals one fact. |
| [3Blue1Brown MLP creator notes/video page](https://www.3blue1brown.com/lessons/mlp/) | Read assumptions, matrix/bias/activation/down-projection explanation and the Superposition/A Quick Demo correction. Video was not independently watched. Useful matrix geometry is annotated alongside the creator's correction to the near-orthogonality demonstration; speculative model dimensions/capacity numbers are not endorsed. |
| [PaLM](https://arxiv.org/html/2204.02311v5), §2 | Read parallel attention/MLP branches and nearby gate, bias, head and embedding choices. The local sequential equation includes both residual updates; it does not copy a source abbreviation that could omit attention from the final sum. No reported speed is generalized. |
| [Swin V2](https://arxiv.org/html/2111.09883v2), §3.2 and Figure 1 description | Read normalization on the branch output, scaled cosine attention and additional main-branch normalization in the largest configuration. Detailed vision/position methods remain later owners. |
| [Gemma 2](https://arxiv.org/html/2408.00118v3), §2 | Read RMSNorm at sublayer inputs and outputs and nearby attention/logit controls. A concrete counterexample to a universal pre-only recipe. |
| [DeepNet](https://arxiv.org/pdf/2203.00555), §§2 and 4.1 | Read scaled residual, initialization table and architecture equation as a combined design. No independent 1,000-layer replication or unrestricted stability theorem is claimed. |
| [PyTorch 2.14 EncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html), [Encoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html) | Stable web URLs returned redirects without substantive bodies. Inspected the installed 2.14 signature, docstring and forward source instead, including defaults, clone warning and both routes; executed same-parameter float64 comparisons. |
| [FlashAttention](https://arxiv.org/pdf/2205.14135), §§2–3.1 | Reused substantive hardware/materialization/tiling/recomputation reading from self-attention. Mathematical exactness is distinguished from bitwise equality and arbitrary device performance. |
| [UCI Libras](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement) and original metadata | Reused checked source/license/acquisition reading and exact original bytes. The data provenance records rights, duplicates, split and limitations. No new download was claimed. |

## Consequential repairs

The old `output.sum()` after final default-affine LayerNorm is constant with respect to upstream inputs, so its typed gradient curve was not evidence of post-norm trainability. This discovery was sent to root. The new one-vector proof and actual nonconstant probe replace it. The earlier residual/normalization packets already contain the correct principles and remain frozen.

Other repairs are explicit in the conservation table and learner explanations: LayerNorm's Jacobian need not contract; identity contributions can cancel; residual variance depends on covariance; final normalization does not guarantee every softmax outcome; post-norm and branch-output normalization are valid design alternatives; GELU does not zero every negative input; attention already has nonlinearity; gated widths require exact rounding/bias accounting; reference defaults are ReLU and fixed FFN width 2048; and MAC/FLOP/memory counts do not establish hardware speed.

## Actual calculations and final author review

The six declared fits completed. Test counts are pre-norm 49/49/51 and post-norm 52/51/47 out of 60. Source row 77 is wrong for both seed-101 display models. Joint point/time permutation and correct padding are numerical nulls; reversing coordinates against fixed time tags and reflecting one real coordinate change outputs materially. The tagged input removes the earlier model's structural order blindness, but comparing whole models does not isolate every changed ingredient. No result-driven retuning followed.

The three complete displayed Python programs were extracted and executed: both reference comparisons return True at shape (2,4,8); sum-probe and contrast gradients match; RMSNorm/SwiGLU returns (2,3,12) and 1,152 FFN parameters. Independent NumPy analytic and central-difference gradients agree within 8.37e−12. Changed practice vectors, matrix outputs, parameter counts, entropy denominators and MAC values match their saved calculations.

The full manuscript was reread in sequence, followed by the entire visual specification. This reread added a complete numerical stage table so the whole-block example is self-contained without the future lab, moved gradient practice off the beginner route, corrected the exact double-normalization rounding and clarified that the FFN example's input is a generic representation rather than a claimed unit-normalized vector. A stray empty disclosure was removed. Spacing and handoff wording were edited for readability. Each constructive visual challenge now includes an explicit initially closed hint and worked solution while accepting other verified answers.

Final scoped author checks: all draft-relative learner links resolve; all three JSON artifacts parse; the 16 hint/solution disclosures are balanced and initially closed; the original published JSX still has its baseline SHA-256; all 11 required packet files exist. Scoped `git diff --check` produced no errors. The displayed programs were unchanged by the final prose/specification edits, so their passing execution evidence was reused. No consequential content or specification gap remains from this author reread. These checks do not substitute for later formal independent review or browser verification.

## Author learning-experience checklist

1. **Route:** first pass is §§1–6 and exercises 1, 2 and 4. Gradient, systems and variant branches are explicitly deeper; no hidden prerequisite for positional encoding.
2. **Cautions:** numerical normalization conditions live in §3, API/dropout conventions in §5, real data/evaluation boundaries in §6, gradient interpretation in §7 and cost scope in §9. Code prints useful outputs, not disclaimer strings.
3. **Real question and data:** the learner follows recorded ordered hand movements through a complete classifier, with a preserved error and exact duplicate/split boundaries. The dataset is not presented as full sign-language understanding.
4. **Investigations:** real vector, matrix, point/time and output-probe edits are defined. Predictions begin unset, bind inputs, invalidate on meaningful changes and have checked contrasts/nulls. There is no preset-only or stepper-only substitute.
5. **Figures:** dependency circuits, signed write directions, normalization rulers, true trajectories, actual stage vectors, gradient probes and derived cost charts have distinct jobs. Axes and accessible alternatives are specified; rendered perceptibility remains phase two.
6. **Connections:** attention to local FFNs, FFNs to persistent learned memories, placement to exact derivatives, time tags to permutation behavior, and tensor shape to cost are explicit. Canonical omissions have actual adjacent owners.
7. **Code:** displayed programs focus on the block, its API comparison and core components. The complete offline study separately owns fitting and provenance. No native training is proposed for the browser.
8. **Practice:** eight independent changed questions have initially closed hints and explained solutions. New vectors, matrix edits, alphabet/source length, parameter dimensions and covariance assumptions test transfer beyond the worked examples.
9. **Screenshots:** none taken under content-only scope. Specifications name informative wrong-prediction, edited-input, failure, null, desktop and narrow-screen states for later independent implementation review.

## Ready handoff and next action

Required files: `lesson.md`, `visual-specifications.md`, `design.md`, `data-provenance.md`, `movement_libras.data`, `movement_libras.names`, `author-calculations.py`, `author-results.json`, `block-models.json`, `additional-calculations.py`, `additional-fixtures.json`. Keep all as pending phase-two inputs. No disposable scratch was created. This packet is frozen for root's content reconciliation and checkpoint. Root owns checkpoint hashes and phase statuses. The author proceeds to Positional Encodings; runtime, browser, formal independent phase-two review and publication remain deferred.
