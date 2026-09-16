# Deep-learning architectures: content scope and handoff

Authorized 13 September 2026: continue research and writing only for the next 30 topics in actual module order, after Convolution, Pooling & Receptive Fields. The scope is Deep Learning Fundamentals & Architectures positions 10–39. It starts with Landmark Architectures and ends with Hybrid SSM–Transformer Architectures (Jamba). No module boundary is crossed in this request.

The central [delivery ledger](lesson-delivery-progress.json) owns the current content and implementation phases. This document records the fixed scope, source conservation and cross-topic decisions; it is not a second mutable status queue.

## Exact sequence and disjoint ownership

| Scope position | Module position | Topic | Packet | Author |
| --- | --- | --- | --- | --- |
| 1 | 10 | Landmark Architectures (LeNet → AlexNet → VGG → ResNet → EfficientNet) | [Design](drafts/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/design.md) | deep_foundations_content |
| 2 | 11 | Depthwise Separable & Dilated Convolutions | [Design](drafts/depthwise-separable-dilated-convolutions/design.md) | deep_foundations_content |
| 3 | 12 | ConvNeXt & Modern CNN Designs | [Design](drafts/convnext-modern-cnn-designs/design.md) | deep_foundations_content |
| 4 | 13 | Capsule Networks | [Design](drafts/capsule-networks/design.md) | deep_foundations_content |
| 5 | 14 | RNNs, LSTMs & GRUs | [Design](drafts/rnns-lstms-grus/design.md) | deep_foundations_content |
| 6 | 15 | Sequence-to-Sequence & Encoder-Decoder | [Design](drafts/sequence-to-sequence-encoder-decoder/design.md) | deep_foundations_content |
| 7 | 16 | Attention Mechanism (Bahdanau, Luong) | [Design](drafts/attention-mechanism-bahdanau-luong/design.md) | deep_foundations_content |
| 8 | 17 | Long-Context Sequence Models (Transformer-XL, Griffin, Perceiver) | [Design](drafts/long-context-sequence-models-transformer-xl-griffin-perceiver/design.md) | classical_probabilistic_content |
| 9 | 18 | State Space Models (S4, Mamba, Mamba-2) | [Design](drafts/state-space-models-s4-mamba-mamba-2/design.md) | classical_probabilistic_content |
| 10 | 19 | RWKV & Linear Attention Models | [Design](drafts/rwkv-linear-attention-models/design.md) | classical_probabilistic_content |
| 11 | 20 | Self-Attention & Multi-Head Attention | [Design](drafts/self-attention-multi-head-attention/design.md) | classical_feature_content |
| 12 | 21 | Transformer Block Architecture | [Design](drafts/transformer-block-architecture/design.md) | classical_feature_content |
| 13 | 22 | Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi) | [Design](drafts/positional-encodings-sinusoidal-learned-rope-alibi/design.md) | classical_feature_content |
| 14 | 23 | Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) | [Design](drafts/grouped-query-attention-gqa-multi-query-attention-mqa/design.md) | classical_feature_content |
| 15 | 24 | Multi-Head Latent Attention (MLA) | [Design](drafts/multi-head-latent-attention-mla/design.md) | classical_feature_content |
| 16 | 25 | Sparse & Linear Attention Variants | [Design](drafts/sparse-linear-attention-variants/design.md) | classical_feature_content |
| 17 | 26 | Vision Transformers (ViT, DeiT, Swin, DiNOv2) | [Design](drafts/vision-transformers-vit-deit-swin-dinov2/design.md) | classical_feature_content |
| 18 | 27 | Mixture-of-Experts Transformers (MoE) | [Design](drafts/mixture-of-experts-transformers-moe/design.md) | root |
| 19 | 28 | Interleaved / Cross-Attention Architectures | [Design](drafts/interleaved-cross-attention-architectures/design.md) | root |
| 20 | 29 | Message Passing & Graph Convolutions (GCN, GAT, GraphSAGE) | [Design](drafts/message-passing-graph-convolutions-gcn-gat-graphsage/design.md) | root |
| 21 | 30 | Graph Transformers & Geometric Deep Learning | [Design](drafts/graph-transformers-geometric-deep-learning/design.md) | root |
| 22 | 31 | Boltzmann Machines & Restricted Boltzmann Machines (RBM) | [Design](drafts/boltzmann-machines-restricted-boltzmann-machines-rbm/design.md) | root |
| 23 | 32 | Spectral Normalization & Gradient Penalty | [Design](drafts/spectral-normalization-gradient-penalty/design.md) | root |
| 24 | 33 | Modern Hopfield Networks | [Design](drafts/modern-hopfield-networks/design.md) | classical_probabilistic_content |
| 25 | 34 | xLSTM (Extended LSTM) | [Design](drafts/xlstm-extended-lstm/design.md) | classical_probabilistic_content |
| 26 | 35 | Hyena & Long Convolution Models | [Design](drafts/hyena-long-convolution-models/design.md) | classical_probabilistic_content |
| 27 | 36 | Ring Attention & Sequence Parallelism | [Design](drafts/ring-attention-sequence-parallelism/design.md) | root |
| 28 | 37 | Advanced Optimizers (Lion, Sophia, Prodigy, Schedule-Free) | [Design](drafts/advanced-optimizers-lion-sophia-prodigy-schedule-free/design.md) | root |
| 29 | 38 | Neural ODE & Continuous-Depth Models | [Design](drafts/neural-ode-continuous-depth-models/design.md) | root |
| 30 | 39 | Hybrid SSM-Transformer Architectures (Jamba) | [Design](drafts/hybrid-ssm-transformer-architectures-jamba/design.md) | classical_probabilistic_content |

Predecessor: [Convolution, Pooling & Receptive Fields](drafts/convolution-pooling-receptive-fields/lesson.md), already content-complete. Next unrequested topic: Titans (Multi-Memory Architecture) (titans-multi-memory-architecture). Mini-Batches, Training Loops & Gradient Accumulation and Neural Training Diagnostics also remain outside this increment. Do not skip ahead or reorder topics by prerequisites.

## Authoring and phase boundary

The user continued the earlier parallel research/write workflow. Four authors, including root, own the disjoint topics above. Root also owns this scope, current handoff, shared ledger, inventory regeneration and cross-range reconciliation. Authors write only their assigned stable-ID draft directories and the disposition of notes addressed to those topics; send proposals for other destinations to root with exact ID, reason and evidence. Do not change another author’s files without explicit coordination.

For every topic run its actual --work content inventory preflight, read all returned relevant destination notes and the complete existing published lesson (or full planned blueprint), and follow the current teaching standard, domain playbook, design brief and code/retention policy. Preserve the important original depth while repairing unsupported claims, hidden prerequisites and weak examples. Read a canonical reference’s real section list and record coverage decisions; research substantive current claims and useful alternate learning resources, including videos or creator notes where they help. Record what was actually read, watched or executed.

Produce a complete learner manuscript in lesson.md, actionable visual-specifications.md and a full design.md under docs/teaching/drafts/<stable-id>/. Retain required small offline licensed data, provenance, author programs and measured/derived results in that same topic-owned directory. No partial outline or instructions for another agent to write the missing teaching count as complete content. A convenient prior dataset is a candidate, not a compulsory experiment for every architecture.

Teach a concrete problem, local intuition and prerequisites, explicit shapes/state dependencies, mechanism and formal reasoning, complete worked program or calculation, interpreted results and failure case, changed independent practice with initially closed hints/solutions, and appropriate deeper connections. State a first-pass route; consolidate repeated cautions in their natural homes. Prefer different useful forms for spatial kernels, recurrent memory, attention, graph neighborhoods, energy landscapes, communication and optimizer updates. Neither lab count nor word count establishes quality.

Every interactive specification needs an initially unanswered input-bound prediction, genuine edits to meaningful entities, accurate feedback/invalidation/reset, actual checked contrasts and nulls, readable labels and accessible/mobile composition. Static figures have their own placements and explanatory purpose. Distinguish exact fixtures, simulations, real measurements and hypothetical comparisons. Do not invent benchmark rankings, new hidden states, probabilities or hardware timings absent from retained evidence.

Content first ends before website implementation: no JSX/lab/model/runtime/manifest/catalogue/navigation edits, no rendered/browser or full integration campaign, and no formal phase-two independent review. Small bounded computations and complete instructional programs that substantiate written claims are authorized. Reuse scratch/lesson-tools/Scripts/python.exe read-only; do not install into that shared runtime. Keep native dependencies optional when absent, with complete code and honest deferred execution rather than fabricated output.

Each author rereads its whole completed manuscript and specifications, records the author learning-experience checklist with rendered checks explicitly deferred, verifies material example reasoning/data boundaries, and notifies root of a stable packet. Root performs focused content and sequence reconciliation and hashes all required inputs before marking content complete. Freeze a checkpointed packet unless a concrete edit is coordinated and its affected evidence is updated. Continue all assigned topics without new permission gates.

## Sequence bridges requiring deliberate treatment

- Earlier sequence models precede the dedicated self-attention topic. Define any necessary attention/position/masking mechanism locally before relying on it, then link the later full treatment. A prerequisite link is not a substitute for the local explanation needed to follow this lesson.
- The CNN sequence builds on the completed convolution, initialization, normalization and residual packets. Reuse their actual definitions; distinguish historical reported results from newly run small pedagogical comparisons.
- S4/Mamba, RWKV, sparse/linear attention, xLSTM, Hyena, long-context systems and Jamba overlap. Each lesson must identify its exact recurrence/kernel/state, version and training-versus-inference contract. Root reconciles boundaries; do not silently repeat the same generic linear-attention chapter.
- GQA/MQA and MLA distinguish head sharing, latent compression, positional components and actual cache computation. Parameter/memory arithmetic is not a hardware benchmark.
- Graph methods distinguish permutation-equivariant computation, graph structure, geometry/symmetry and claimed expressivity. The energy-model/Hopfield connection needs matching normalization and update assumptions.
- Ring/sequence parallelism separates algebraic exactness, floating-point order, device communication and actual measured overlap. CPU traces may teach the protocol without claiming distributed GPU performance.
- Advanced optimizers and continuous-depth models need locally stated objectives, state and differentiation assumptions, plus practical failure diagnostics; do not promise universal superiority.

## Original-source baseline

Starting commit: 8c5da59f18516be77c29d5aeeafca3decca4f738. Existing uncommitted/untracked work includes the previous 30 content packets, GMM/manifold/ICA and shared handoff/ledger notes; preserve it. Starting catalogue: 1218 topics, 28 modules, 228 publications and 7 guided paths. Original sources below remain untouched phase-one inputs, not endorsed correctness. Recover them from this commit instead of duplicating source archives.

| Stable topic ID | Original publication source | SHA-256 |
| --- | --- | --- |
| landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet | src/learn/data/topics/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet.jsx | e6a5fe7df260309646198710a61c2b36f4bc23fa1b72baf1967732aaf93dbe0e |
| depthwise-separable-dilated-convolutions | src/learn/data/topics/depthwise-separable-dilated-convolutions.jsx | 9d2d40d607268706bd3b0536e184876c1b15809954a830494aac32e5af3b065b |
| convnext-modern-cnn-designs | src/learn/data/topics/convnext-modern-cnn-designs.jsx | 996fbe6ba48f0ca90b434785fbc7e10b908c2dad37e830b1cb08f71365ab7b0a |
| capsule-networks | src/learn/data/topics/capsule-networks.jsx | f980679c6504889399b8316e83880fe58c58038af7857dfcbc6a8587bf556a58 |
| rnns-lstms-grus | src/learn/data/topics/rnns-lstms-grus.jsx | 7fd7f02af2f2eca7aceb5b34422793e571c186f8d8525138bfdf2c913ce0f2cf |
| sequence-to-sequence-encoder-decoder | src/learn/data/topics/sequence-to-sequence-encoder-decoder.jsx | 7ad7db0588632290b7b78a6d628f97c76cb685876b601357f551fe716bbe6005 |
| attention-mechanism-bahdanau-luong | src/learn/data/topics/attention.jsx | eb481258c4aaf13bd9f1a3d697802190174b7e1f55c6ff62b47f825255fd5e36 |
| long-context-sequence-models-transformer-xl-griffin-perceiver | Planned title/level and inherited model guidance; preflight confirms no bespoke blueprint, so individual design is required | — |
| state-space-models-s4-mamba-mamba-2 | src/learn/data/topics/state-space-models-s4-mamba-mamba-2.jsx | 790a636f3c00b6eac5d9009c071c82f6caed6e66a658be9a36e3e8707c98177d |
| rwkv-linear-attention-models | src/learn/data/topics/rwkv-linear-attention-models.jsx | acccaa77eeefb66aeaacbe7b0b7fe7491028d1853910b65cf627cb09cfe498b0 |
| self-attention-multi-head-attention | src/learn/data/topics/self-attention-multi-head-attention.jsx | 31c623b5015af4ba80b58085603f263aac2d71089455d897a4e52b6964de94e3 |
| transformer-block-architecture | src/learn/data/topics/transformer-block-architecture.jsx | 8261cf488a1fc48c09c3bac9c867d79c0b4545643f5e959aee6c8456ea446f69 |
| positional-encodings-sinusoidal-learned-rope-alibi | src/learn/data/topics/positional-encodings-sinusoidal-learned-rope-alibi.jsx | 403cc0706f77768d87421ed5d671e7897f92d387639201e3033041e3cd072ff6 |
| grouped-query-attention-gqa-multi-query-attention-mqa | src/learn/data/topics/grouped-query-attention-gqa-multi-query-attention-mqa.jsx | 423821bbbbb9f33cf2c0a73c8261f2bab3b7de354e118eadd9237118564bebcf |
| multi-head-latent-attention-mla | src/learn/data/topics/multi-head-latent-attention-mla.jsx | 2b8ca2d831ebd2704cbbde34de240151ddd2ba971613d73aefd1fe7f871ad732 |
| sparse-linear-attention-variants | src/learn/data/topics/sparse-linear-attention-variants.jsx | d043d75790b4a29bed9300aa16eb39e57a92b9fd3030d9d9ea38e4c8412d6261 |
| vision-transformers-vit-deit-swin-dinov2 | src/learn/data/topics/vision-transformers-vit-deit-swin-dinov2.jsx | d70f71f77a07b0946e11bfbded97c1fca41e29d4f2cb1816c9cbe4d4b4457b0b |
| mixture-of-experts-transformers-moe | src/learn/data/topics/mixture-of-experts-transformers-moe.jsx | fad5a8e9f48f6d51a7a94825aa09ded73e7ef84443253762e4659d6e64d0f230 |
| interleaved-cross-attention-architectures | src/learn/data/topics/interleaved-cross-attention-architectures.jsx | 56b34fe167f6b46645c084df8cd09112da75a739ba1dc171fc890d33a232a6b1 |
| message-passing-graph-convolutions-gcn-gat-graphsage | src/learn/data/topics/message-passing-graph-convolutions-gcn-gat-graphsage.jsx | cfc63c969032ce8a17c991bd42e15bce7fdf00a695850d4bbdb575045d25c925 |
| graph-transformers-geometric-deep-learning | src/learn/data/topics/graph-transformers-geometric-deep-learning.jsx | f110437d9e968702028b3f785ff28853f3f93474233aa496d77c2c80e47f20b8 |
| boltzmann-machines-restricted-boltzmann-machines-rbm | src/learn/data/topics/boltzmann-machines-restricted-boltzmann-machines-rbm.jsx | 91a9d5150f96b66a1cefa045b9f62709325f632f972f896f46c3699dd7e32274 |
| spectral-normalization-gradient-penalty | src/learn/data/topics/spectral-normalization-gradient-penalty.jsx | 79209e97bfdb4e7d03f676aa5bd60cc3b0b0240eb095d95793549c25896dc068 |
| modern-hopfield-networks | src/learn/data/topics/modern-hopfield-networks.jsx | d17566f43c90d83fbb24f0bb9ba51f6779a930a6b6cac366279a00d68fbdbcf5 |
| xlstm-extended-lstm | src/learn/data/topics/xlstm-extended-lstm.jsx | d9594e806e11b12e8eedbd763c141da9b36bf9067a4ef82e986e80ee80f9b995 |
| hyena-long-convolution-models | src/learn/data/topics/hyena-long-convolution-models.jsx | 621f9daadcbdebb6a38eaf8035daf0518b5026e996c5eb997086fd554cacb93c |
| ring-attention-sequence-parallelism | src/learn/data/topics/ring-attention-sequence-parallelism.jsx | bcd2f2b55ef4c16c93b74892df46d6c05075a682db3cea5e575c25bf32b16e7a |
| advanced-optimizers-lion-sophia-prodigy-schedule-free | src/learn/data/topics/advanced-optimizers-lion-sophia-prodigy-schedule-free.jsx | 179254cc0ec5b8c9232b6a3ad7862cca77b618383bac7020ea47dbf81c20539c |
| neural-ode-continuous-depth-models | src/learn/data/topics/neural-ode-continuous-depth-models.jsx | f022724f84307be8059de01876003b5e8b4ce91329aa21410b5f6d9071e73cee |
| hybrid-ssm-transformer-architectures-jamba | src/learn/data/topics/hybrid-ssm-transformer-architectures-jamba.jsx | e71d445f9077cc2dcf9f265b749d47a631a5022e29f3a93f696e87a8a5d457d8 |

## Reconciliation, evidence and next action

### Cross-topic decisions and learning continuity

- **Keep the actual syllabus order.** The recurrent/long-context lessons introduce the local attention, masking and state concepts needed before the later dedicated Transformer sequence. The attention manuscript continues to Long-Context Sequence Models; it does not jump to the later Self-Attention publication. Neural ODEs distinguish continuous depth from sequence time before the Jamba bridge. The next unprepared entry after this scope is Titans.
- **Conserve depth while correcting the old claims.** Each design maps the complete original lesson, or the planned long-context brief, to retained, corrected, expanded and deliberately linked coverage. Each also records a canonical reference's actual section agenda and disposition. Stable titles/identities and existing runtime sources remain unchanged; clearer learner headings do not require catalogue renames.
- **Use different representations for different mechanisms.** Kernel footprints and patch grids, recurrent state traces, decoder timelines, head/cache layouts, graph neighborhoods, energy landscapes, normalization geometry, ring communication, optimizer state and numerical trajectories have distinct visual and investigation contracts. A static explanation is not counted as an independent lab. Predictions begin unanswered; fresh inputs, meaningful edits, actual contrast, null cases and reset/invalidation are specified per investigation.
- **Separate architectural choices that are easy to conflate.** Full/selected router normalization is stated per model; the general MoE lesson and Jamba retain their respective conventions. Head sharing, latent cache compression, sparse pair selection, kernel approximation and shared cross-attention memory are separate operations. Weight sharing alone does not establish cache sharing. Causal layer/position dependencies are explicit.
- **Reuse data only when the question benefits.** The recurrent encoder–decoder and attention packets share the exact final grouped inflection extract and source-bound baseline, without refitting the predecessor. Other packets use appropriate image, handwriting-trajectory, movement, graph, tabular or DNA evidence, or exact mathematical examples where a data fit would not answer the mechanism question. License, source identity, extraction, split and limitations are recorded locally.
- **Preserve the actual outcomes.** Native studies retain their declared conditions, inspected split roles and unfavorable comparisons. A stronger baseline is not removed to make the focal method win. A changed probability, state or logit can be important even when the winning class is unchanged. Resource equations, exact arithmetic and measured model results are labelled separately; no invented timing/Pareto curve substitutes for a benchmark.
- **Make the second phase concrete.** Packets retain full manuscripts, changed practice and closed solutions, annotated primary/alternate resources, necessary offline programs/data/results and precise figure/lab specifications. Model-specific optional programs that require unavailable packages, checkpoints or suitable hardware are explicitly marked unexecuted. Browser models, UI states, accessibility/perceptibility, payload measurements, independent correctness/learning review, publication and integration remain for an authorized finish request.

Current extensions better owned elsewhere are saved as OPEN destination notes: [causal-encoder cross-attention memory and index reuse](topic-notes/interleaved-cross-attention-architectures.md), [matrix-gradient geometry versus curvature, including Muon](topic-notes/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient.md), and [the full self-distillation comparison through DINOv3 and dense-feature objectives](topic-notes/self-distillation-byol-dino-dinov2.md). These notes carry reasons, source extents and boundaries; they do not authorize reopening a frozen packet or implemented lesson. Future authors must assess them in their actual topic preflight.

Full research, author rereads, focused corrections, actual numerical checks and remaining native limitations belong in each topic's design/provenance. Root's cross-topic reconciliation is part of phase-one authorship, not formal independent phase-two review or a learner study.

### Final content checkpoint — 13 September 2026

All **30 complete manuscripts/specification packets** are closed in the central ledger, with **367 required files bound by SHA-256**. Each topic's actual `--work finish` preflight returned content `complete`, implementation `not-started` and `canFinish: true`. Every file in each topic directory belongs to its saved checkpoint; no unexpected nested temporary artifacts remain. These hashes identify completed author inputs, not reviewed website implementations.

The final bounded reconciliation verified the exact thirty identities against actual module positions 10–39, all **29 unchanged original published-source hashes**, and the one original planned entry's continued unpublished state. It checked **136 lesson routes** against actual topic/module membership and **223 relative links** in the packets. All JSON inputs parse. The **previous thirty packets and all 241 of their checkpointed files remain unchanged**, with their implementation phases still not started.

`node scripts/verify-lesson-delivery.mjs` passed all eight behavior groups with 174 tracked topic revisions. A single final inventory regeneration retained **1,218 topics, 28 modules, 228 publications, 355 topic briefs, 383 recorded prerequisite reviews and seven guided paths**. Current effective content completion increased from 143 to **173**; current implementation completion stayed **109**. The existing separately recorded K-Means proposal was not resolved or silently approved by this work. All 43 relative links in this scope and its three new destination notes resolve, and the final scoped `git diff --check` passed.

Per-topic author checks and final rereads are recorded in their designs. Examples of focused final corrections include ViT's PCA percentage sum, Jamba's affine state-update wording and an answer-bearing prediction prompt, and numerical nulls where the leading class remains unchanged. These repairs reused unchanged fits and calculations. Optional foundation-model programs and unavailable native/distributed execution retain their explicit unexecuted status; no new measurement was fabricated to fill that gap.

Necessary manuscripts, specifications, source/provenance, licensed offline data, programs, fitted arrays and author results remain for the implementation handoff. Disposable topic-owned retrieval extracts/helpers and Python caches were removed by their authors; no recursive scratch audit or deletion of prior pending work/shared tools was performed. No website runtime, live catalogue, publication manifest, lab component or browser implementation was changed. Application build, browser/accessibility review, formal independent phase-two review, deployment and commits were not performed for this content-only request.

**Next action:** follow the user's next scope. A finish request consumes a selected packet through its `--work finish` preflight and completes implementation, independent review, applicable corrections/checks and integration. A new sequential content request begins with **Titans (Multi-Memory Architecture)**, stable ID `titans-multi-memory-architecture`, module position 40. This completed scope does not authorize starting it automatically. The current handoff and AGENTS entry now point to this boundary; the earlier Landmark boundary is explicitly historical.
