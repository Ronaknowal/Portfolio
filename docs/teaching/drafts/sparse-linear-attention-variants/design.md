# Sparse and Linear Attention — authoring record

Research/write packet, 13 September 2026; implementation not started. Author completion checks are recorded at the end. Actual `node scripts/build-curriculum-inventory.mjs --topic sparse-linear-attention-variants --work content` preflight consumed; no individual note/blueprint exists. The resolved unrelated bit-manipulation inbox does not add scope. Full original published JSX read, including all 1,097 lines of code, visual matrices, claims and practice; SHA-256 `d043d75790b4a29bed9300aa16eb39e57a92b9fd3030d9d9ea38e4c8412d6261`. Preserve that runtime source. Root owns shared phase status and exact checkpoint hashes.

## Learning contract, scope and sequence

Stable ID `sparse-linear-attention-variants`, Deep Learning Fundamentals & Architectures module position25. Retain the title **Sparse & Linear Attention Variants**: it covers the family comparison, exact-versus-approximate distinctions and modern selection methods without promising a single implementation or exhaustive historical catalogue.

The learner has actual prepared self-attention, Transformer block, positional encoding, GQA/MQA and MLA packets. Earlier RWKV/SSM packets locally teach recurrent reads and distinguish signed/normalized operators. Refresh Q/K/V, row-softmax, causal legality, matrix shapes, outer products, state and metric definitions locally. Gaussian moment identity and pseudoinverse appear only with explanation in the deeper branch, not as hidden entry prerequisites. The immediate predecessor is MLA; the next topic is ViT/DeiT/Swin/DINOv2, followed by MoE. Use real full-curriculum links with module context, preserving order.

Observable outcomes: calculate a sparse row and its removed-mass effect; trace a causal input route across layers; update/read/evict a feature memory; distinguish an exact changed kernel from a softmax approximation; diagnose future leakage in sequence compression; distinguish token sparsity, tile work and persistent state; interpret a controlled real model comparison without promoting it to universal performance evidence. The first-pass route appears immediately after the introduction and ends in practice1–5; random-feature theory, additional families, gradients and current systems are deeper branches with their own transfer practice.

### Scope and coverage decisions

| Idea | Evidence/current coverage | Decision and destination |
| --- | --- | --- |
| Sparse graph, local/dilated/strided/global/random access | Original family breadth useful, causal diagrams/cost claims unreliable | Teach masks, directed layer paths and independent output calculation in§2; current-owner depth |
| Hashing and clustering | Original hash definition mixed different LSH schemes | Correct actual Reformer hash and causal original-position mask; bounded Routing Transformer mechanism/cost, no universal empirical ranking |
| Feature kernels and Gaussian random features | Original conflated ELU+1 with softmax, ratio with unbiased kernel, fixed-radius ORF with Gaussian | Main mechanisms§3–4 with original hand derivations, full programs, actual seed variation and correct distribution distinction |
| Sequence compression and Nyström | Linformer causality claim too absolute; Nyström only named | Teach full projection leakage and explicitly changed causal prefix construction; bounded landmark/pseudoinverse mechanism§5 |
| Native sparse kernels and current indexers | Old benchmark/production assertions unsupported | Tile occupancy investigation and current FlexAttention/NSA/DSA/CSA2 mechanisms§6; never infer seconds from counts |
| Exact tiled attention and cache compression | Actual earlier packets own foundations | Short refresher/cost contrast, not duplicate GQA/MLA chapters; later Ring owner handles distributed communication |
| RetNet/GLA/delta/SSM/mLSTM families | Earlier RWKV/SSM and later xLSTM packets explicitly coordinate exact operators | Local decay/delta read proof and links; no claim all are softmax approximations or all have identical state size |
| CED/CSA2 cross-layer memory source | Current V4.1 report reveals separate representation and selection reuse | Open [destination note](../../topic-notes/interleaved-cross-attention-architectures.md), saved with root authorization; future scoped extension, not reopened frozen packet |
| Axial/shifted spatial windows | Adjacent ViT/Swin naturally owns image geometry | Bridge at lesson end; assess axial comparison during that next owned packet, no new catalogue topic |
| Sparse expert computation | Adjacent MoE owns expert routing rather than attention-edge selection | Explicit distinction and next-owner bridge; do not duplicate expert balancing here |

### Canonical reference coverage check

Canonical family map: Tay, Dehghani, Bahri and Metzler, [Efficient Transformers: A Survey, v3](https://arxiv.org/html/2009.06732v3). Actual full section list read, plus background/taxonomy and§4 evaluation/design discussion. It is a historical map, not the authority for every current implementation or formula. The inspected list is: §1 introduction/version notes; §2 background with multi-head attention, FFN, composition, complexity, encoder/decoder/encoder–decoder and applications; §3 taxonomy and detailed reviews of Memory Compressed, Image, Set, Sparse, Axial, Longformer, ETC, BigBird, Routing, Reformer, Sinkhorn, Linformer, Performer, Linear, Synthesizer, Transformer-XL, Compressive and Sparse/MoE models; §4 evaluation, model-design trends, orthogonal efficiency efforts and retrospective/future discussion; §5 conclusion.

Disposition of ideas beyond this manuscript's deep treatment:

- Background full-block/encoder–decoder mechanics are already taught in their actual preceding packets; local refreshers supply the used equations. No redundant full-block derivation.
- Memory Compressed and Compressive Transformer are represented by the compression-versus-individual-memory distinction; detailed recurrent segment memory belongs to the actual Long-Context packet. A new historical reproduction would not improve the current key contrasts.
- Image/Axial attention belongs with image geometry in the next ViT/Swin packet; Set/inducing-point methods relate to the already prepared Perceiver bottleneck and later cross-attention owner. They are not silently claimed as fully taught here.
- ETC's document structure is developed as an alternate-resource connection; generic local/global graph mechanism is taught here. A separate model-specific training recipe is omitted because it repeats that mechanism while adding unrelated objectives.
- Sinkhorn's learned block ordering and Synthesizer's alternative score generation are not developed into new full subsections: this lesson teaches candidate routing, low-rank summary formation and kernel reassociation; those historical models add architecture-specific learning objectives rather than a missing core dependency. This is deliberate scope, not a claim of exhaustive model coverage.
- Weight sharing, mixed precision, pruning, distillation, NAS and adapters are orthogonal efficiency tools with existing dedicated curriculum owners. The local cost table makes their distinct effect clear; their full methods do not belong in an attention-operator chapter.
- Reversible residuals and FFN chunking are memory-execution choices distinct from graph/kernel changes. The exact-attention discussion records tiling/recomputation/checkpointing; full reversible-network training is outside the operator scope. Original sparse family depth, actual equations and whole-model memory distinction are conserved without a second residual-network lesson.
- Current NSA, V3.2 DSA, V4.1 CSA2 and FlexAttention postdate the survey and were researched directly. Their source-specific asymptotic qualifications prevent the historical map becoming a stale “sparse attention is obsolete” conclusion.

## Conceptual hurdle and evidence map

| Hurdle | Local bridge and mechanism | Representation and evidence | Core/deeper |
| --- | --- | --- | --- |
| One word “efficient” hides different costs | Count actual legal pairs, retained tensors and whole-model work | F1 resource map; exact allocation example; practice3/5 | Core |
| Deleting a key changes normalization | Renormalized probabilities and own removed-mass bound | F2, I1 numerical tab, changed practice | Core |
| Graph reach depends on time and depth | Row receiver/column source, Boolean path composition, causal hubs | F3 and I1 fresh hub4 versus hub0 | Core |
| A state can answer queries without retaining individual records | Outer-product writes and separate normalizer | F4, I2 edits/eviction/constant null; displayed full program | Core |
| Changing a kernel versus approximating softmax | ELU+1 defined operator, Gaussian identity, ratio counterexample | F5 and I3 seeded exact/approximate contrast | Basic distinction core; derivation deeper |
| Compressed summaries may already contain the future | Explicit length projection versus causal prefix sums | F6, I4 future and zero-coefficient controls | Core worked example; variants deeper |
| Fewer edges need not mean proportionally less hardware work | Occupied tiles and selection overhead | I5 counts, F7 current architecture cost arrows | Deeper practical |
| Actual learned response differs from structural reach | Same data, initial weights, budget and baselines; no manufactured winner | F8/I6 remote null and near contrast; all declared outcomes | Core |
| Training/inference use the same mathematical operator | Recurrence, quotient gradients, reverse-scan influence | Float64 all-parameter comparison and bounded gradient derivation | Deeper |

## Original conservation and corrections

The starting source is recoverable at commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; no duplicate source archive is needed. Read all original sections, including complete code and JSX visual data. Preserve family breadth, cost reasoning, application intent, implementation learning and self-check function, but replace unsupported outputs with executed evidence.

| Original region | Conserved value | Correction or replacement |
| --- | --- | --- |
| Opening/history | Long sequence motivation and cost pressure | Remove unsupported historical/budget claims and wrong GPT-2 context limit; exact attention does not require materializing L² scores |
| Intuition and math | Sparse versus low-rank/kernel families | Separate graph changes, sequence summaries, exact chosen kernels and stochastic softmax approximation; supply explicit dimensions and local examples |
| Local/global/BigBird | Paths and sparse counts | Window counts include self; causal hub0 cannot gather future; two-layer paths are not simultaneous-head paths; theorem existence is not a fixed-budget equality guarantee |
| Reformer/Routing | Content-aware candidate discovery | Correct signed-axis argmax hash versus hyperplane-sign collision formula; bounded chunks trade coverage, while uncapped buckets change worst-case work; include routing assignment cost |
| Linformer | Sequence projection | Full projection can leak; a separately defined prefix-summary construction is causal and linear for fixed rank, so “no causal route exists” is rejected |
| Performer code/theory | Positive features and reassociation | Gaussian scale on both Q/K, unbiased unnormalized estimator versus biased ratio, actual ORF marginal radii, stable common key units, no feature resampling inside cached inference |
| Claimed GPU stdout/scaling plots | A need for empirical evidence | Remove unverified hardware numbers and invented quality frontiers; replace with actual small CPU study, stored model weights, seed distribution, explicit state counts and tile occupancy |
| Production advice | Choosing methods and testing failure modes | Remove universal thresholds, model-family blanket claims and “sparse obsolete” conclusion; current primary indexer/kernel evidence with exact cost boundaries |
| Old examples/practice | Hands-on learning and transfer | Fresh independent practice/hints/solutions and six mechanism-specific investigations; worked examples explicitly ungraded |

## Research and substantive read record

All checks below occurred 13 September 2026. Papers are primary sources; their own broad empirical wording is not copied as a universal statement. The mathematical examples and controls are original local calculations. No source metadata/abstract is presented as a full paper/video review.

| Source / locator | Substantive material actually inspected | Claim use and boundary |
| --- | --- | --- |
| [Linear Transformers](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf),§3.1–3.3.2 | General kernel operator, ELU+1, causal state equations, numerator backward recurrence | Exact chosen-kernel reassociation and causal gradients; do not repeat outdated dense-materialization necessity |
| [Performer v4](https://arxiv.org/html/2009.14794v4),§2.1–2.4,§3 variance formulas and Gaussian/regularized appendix material | Positive Gaussian identity, ORF, SMREG/fixed-radius distinction, expectation/MSE reasoning | Gaussian-marginal versus fixed-radius sampler, pair variance; own ratio counterexample rejects sloppy normalized-unbiased claim |
| [Longformer v2](https://arxiv.org/html/2004.05150v2),§3.1–3.2 and beginning§4 | Window/dilation/global projections, implementations, autoregressive versus encoder setting | Graph mechanisms, not universal retained attention mass or one implementation for every mode |
| [BigBird v2](https://arxiv.org/html/2007.14062v2),§2,§3.2 theorem and proof sketch; application description | Local/global/random graph, star condition, fixed-domain approximation scope, genomics motivation | No random-edge-count theorem invented; no clinical/scientific validity inferred from graph alone |
| [Sparse Transformer v1](https://arxiv.org/html/1904.10509v1),§4.1–4.3,§5.1–5.2 | Exact local/strided/fixed definitions, sequential/merged/head alternatives, recomputation | Graph routes and L√L construction; avoid copying source shorthand/off-by-one ambiguity |
| [Reformer v2](https://arxiv.org/html/2001.04451v2),§2 through multi-round/causal masking,§3 reversible/FFN chunking | Actual hash, shared QK, original-position masking, bounded sorted chunks, memory mechanisms | Model-family breadth and scoped limitations; appendix multi-round implementation was not fully reviewed or reproduced |
| [Routing Transformer](https://aclanthology.org/2021.tacl-1.4.pdf),§4/4.1 including algorithm and cost | Normalized query/key routing, online centroids, balanced budget, assignment+pair costs | Mechanism/cost only; no paper leaderboard imported |
| [Linformer v3](https://arxiv.org/html/2006.04768v3),§3 and§4 model/proof statement | Empirical attention-map spectra, approximation context, learned length projections | Own explicit matrix orientation and future-leak/prefix construction; no universal key/value low-rank assumption |
| [Nyströmformer v3](https://arxiv.org/html/2102.03902v3),§2–3 through landmarks/inverse/algorithm | Three row-softmax factors, pseudoinverse, iterative approximation, segment means | Small own fixture uses numerical pseudoinverse, no unexecuted iterative convergence claim |
| [NSA](https://arxiv.org/html/2502.11089v1),§3.1–3.3.3 | Three branches, compressed-score selection, grouped-head aggregation, local branch | Separate branch normalization/gates and residual compression cost, no imported throughput |
| [V3.2-Exp report](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf),physical pages1–4 | Full indexer formula, warmup/teacher objective, sparse training, explicit quadratic indexer versus Lk main read | Fetched/read in memory with pypdf after web MIME parser failed; no retained report archive; no architecture training reproduced |
| [V3.2-Exp repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp),README | Indexer implementation context and November2025 RoPE-layout bugfix | Version caution; do not interchange indexer and MLA positional layouts |
| [V4.1 report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf),full TOC andphysical pages9–12 (§2.2–2.3),model-card introduction | Causal encoder–decoder global-memory source, local-window independence, Full/Reindex/Reuse, hierarchical candidate pool | Open destination note; §3.2.2 bounded replay body was not inspected, so do not claim exact cache replay or reproduce benchmark numbers |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA),full current README | Current dense/sparse support, architecture-specific record formats, separate V3/V4 paths | Sparse methods remain active; parameter/cache arithmetic is not measured speed |
| [FlexAttention blog](https://pytorch.org/blog/flexattention/),full score/mask/packing/API discussion;[current API](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html) | Mask modifier and BlockMask semantics, compiler role, skipping empty tiles | Current production pathway; no GPU execution claim |
| [Performer author article](https://research.google/blog/rethinking-attention-with-performers/),full substantive body | Factorization/prefix visuals and protein application | Annotated alternate learning resource; kernel-versus-ratio qualification and historical hardware scope supplied locally |
| [Sparse-attention author article](https://research.google/blog/constructing-transformers-for-longer-sequences-with-sparse-attention-methods/),full substantive body | Graphs, ETC sentence/paragraph structure, BigBird, blockification, genomics | Useful alternative representation, with historical hardware/performance caveats; no video-watching claim |

FlashAttention's exact tiled-operator distinction reuses the substantive primary-source research from the earlier owned self-attention/Transformer packets and the current FlexAttention source. No need to reopen frozen lessons. Current source reads identify more model names than this chapter teaches; the title does not promise a list of every paper published by the retrieval date.

## Applications and practice rationale

The main real anchor asks what three operators retain and how an older coordinate edit affects a next-point forecast. It is deliberately modest, openly licensed and offline. Exact duplicate grouping, whole-trajectory split and unavailable performer IDs are explained once in the data-method home. Persistence and affine baselines prevent a sophisticated operator from being credited for trivial smoothness. The real comparison preserves a near-tie and small sensitivity changes instead of tuning a dramatic winner.

Constructed examples supply properties the real task alone cannot establish: causal absence of a path, removed-mass renormalization, exact summary collisions, future leakage and tile occupancy. Current indexers teach why discovering a subset is part of its cost. DNA/protein context connects graph locality to nonlocal scientific structure, with explicit task mapping and no implied scientific validation. These add distinct learning rather than a list of industries.

Ten changed practice questions cover calculation, diagnosis, counterexample, state cost and experimental design. Optional hints/solutions are initially closed. Six investigations use separately declared fresh configurations with visible default results; all include genuine edits and checked nulls/contrasts. The I6 boundary control was computed from saved weights, not a second training campaign. Mechanism/program/provenance files are retained so another author can implement the actual systems without inventing data.

## Experiment declaration before fitting

Use byte-exact, verified CC BY 4.0 UCI Libras Movement originals, 330 unique trajectories after label-consistent exact duplicate grouping, and the prior seed-73 classwise whole-trajectory split of 220/50/60. Reuse fixed `2*x-1` scaling and causal next-coordinate task: positions 0–43 predict 1–44. Labels only stratify; performer/session IDs are unavailable, so this is a row-level mechanism diagnostic, not new-person generalization or a production long-context benchmark. Retain persistence and a six-parameter training-fitted affine baseline.

Train three declared small models from the same seed-137 initial parameters: full causal softmax; causal window softmax with **five total legal keys including self**; normalized ELU+1 feature-kernel attention. Each has a 2-to-24 biased stem, fixed ordinary sinusoidal position addition of width 24, one pre-norm block, three heads of width 8, bias-free Q/K/V/output maps, LayerNorm epsilon 1e-5, biased GELU FFN 24→48→24, final LayerNorm and biased two-coordinate forecast. No dropout or weight decay. Use 160 full-batch Adam updates at 0.003; select each model's minimum validation MSE, earliest exact tie. Report all declared outcomes, including budget-endpoint selections or simple-baseline wins. These are equal-budget local runs, not matched best possible architectures or replicated statistical evidence.

The linear model uses features `elu(Q/d**.25)+1`, `elu(K/d**.25)+1`, positive normalization and causal prefix outer-product/normalizer state. This is a different defined kernel, not an approximation to softmax. Compare its quadratic feature-kernel reference, state accumulation and full-network incremental computation with identical weights. Check full-parameter float64 gradient equivalence for the two feature-kernel evaluation orders on a small input. The small full-batch teaching implementation may store prefix states for autograd; do not label its training memory constant merely because streaming inference is.

Predeclare source row 77 and its first 32 points for inspection. Save all three models and actual forecasts, attention/state traces, source-point edit at frame 23, strict earlier-output causal null, correct incremental equivalence and actual cache/state counts. Do not force a dramatic remote effect or a preferred winning operator.

Separately evaluate positive IID Gaussian random-feature approximations to the **selected dense model's actual head-0 Q/K/V** on this prefix. Use nested feature counts 16, 64 and 256 and seeds 0–7, retaining every outcome. Fold softmax scale into Q/K via d^(-1/4); never resample features inside a single inference. Report operator output error against the same softmax reference, not task-trained Performer quality or a worst-case theorem. The primary iid mechanism and a independently constructed Gaussian-marginal orthogonal-feature sampler will be illustrated with exact distribution requirements; no fixed-radius sampler is described as Gaussian.

Independent hand fixtures cover sparse gathered-vs-masked equality, causal graph reach through layers, a hub at position zero that cannot gather later information, removed-mass error, block tile occupancy, linear-state arithmetic and destructive collisions, Linformer future leakage versus a causal prefix-summary construction, and actual random-feature convergence variability. No fabricated density-quality or GPU-latency plots. Complete manuscript, representations, changed practice, research/conservation records and author reread/checklist are required before freezing.

## Author completion and bounded checks

**Content ready for root reconciliation, 13 September 2026.** Full completed manuscript and visual specifications reread from beginning to end. The local formulas, actual figures, program outputs, practice and source caveats form a continuous route; no material writing gap remains. This is the author's review, not formal independent review or implementation completion.

- `author-calculations.py` executed the declared three fits once, saving every declared outcome, actual weights/traces, 24 random-feature evaluations and whole-network derivative comparison. Results and limitations are recorded in provenance; no later tuning campaign.
- `mechanism-calculations.py` executed float64 sparse gathered/masked equality, causal hub paths, feature-state arithmetic, eviction/null/collision cases, future projection leakage, Nyström factors, Gaussian-marginal orthogonal rows, tile counts and changed practice. A later comment was corrected to avoid falsely labelling its particular Nyström matrix singular; calculations are unchanged.
- `author-checks.py` executed saved-weight reproduction, fresh near-window contrast/earlier null, manual random/sparse/projection controls and the displayed NumPy program. The fresh random-feature problem actually worsens when increasing 8→64 features; the contract preserves this counterexample.
- Scoped packet checks passed: relative links, real module-sequence route IDs, balanced inline/display math delimiters, closed details/hints/solutions, readable Unicode, and unchanged original JSX hash. A first route check incorrectly required the next unwritten topic's draft directory to exist; it was corrected to use the actual scope/catalogue sequence, with no route or curriculum change. Root owns the separate shared ledger validator.
- Drafting exposed lost inline math delimiters from command-string escaping; the complete manuscript's formulas were repaired and reread, with balanced delimiter checks. Dense/linear head dimensions, source metadata, chosen validation endpoints and fresh/default distinction remained consistent. Final spec readability was cleaned without changing numerical contracts.

### Author learning-experience checklist

| Check | Concrete finding and disposition |
| --- | --- |
| Route | First-pass route after introduction; content-based family comparison, random-feature details and hardware/current systems visibly marked deeper. Core continues to the real trajectory study and practice1–5. Readiness does not require memorizing family names. |
| Cautions/hedging | Whole-manuscript pass retained operator conditions at their home and removed redundant positive-map warning, internal missing-helper comparison and irrelevant video-review sentence from learner prose. Recorded signal: 12 selected hedging phrases across147 paragraph blocks. Code prints numerical output/PASS, not caveats. |
| Real question | Introduction's selective-read versus aggregate-memory question returns in a real trajectory forecast with persistence/affine baselines, the same declared budget and actual small effects. Duplicate/entity and familiar-dataset boundaries appear in§7. |
| Investigations | All six have fresh inputs, visible default results, actual entity edits, immediate recomputation/reset and checked contrasts/nulls. Worked arrays remain ungraded. The fresh random-feature result worsens with more features; remote versus near edits prevent a false universal window lesson. |
| Inline figures | Full prose-only reading pass verified placements for graph routes, normalization, outer products, memory collisions, projection leakage, landmark factors, tiled work, current indexers and real state/trajectory comparison. Different mechanisms use different forms. Rendered desktop/phone visibility remains explicitly deferred. |
| Connections | Sparse gathered and masked reference; kernel pair and recurrent reference; full-prefix and incremental model; gradient routes are connected in prose. Canonical omissions have explicit local/adjacent-owner decisions. Actual MLA→Sparse→ViT route preserved. |
| Code | Displayed compact NumPy program exposes the recurrence and normalizer with only local validation; it was executed as displayed. Full downloadable study includes all helpers and data boundaries. Browser models/labs remain unimplemented. |
| Practice | Ten changed problems include exact new values, diagnostics, counterexamples and experimental design. Twelve hint/solution disclosures start closed. Exact numerical answers and open-task reasoning criteria supplied. |
| Screenshots | Not taken: content-first mode defers implementation/rendering. Specs name informative desktop/phone states, including the small forecast changes, random-feature reversals and checked changed-input states. No unrendered figure is called visually verified. |

### Final handoff

Preserve all **14 files** in this stable-ID directory: lesson.md, visual-specifications.md, design.md, provenance.md, original data/metadata, three complete author programs and five necessary observed/calculated result/model files. They total about1.83MB because the pending handoff includes full saved traces; phase two derives compact lazy assets rather than eagerly importing the packet. No scratch cleanup remains for this topic.

Root may now reconcile and bind all required files to the content checkpoint; implementation stays **not started**. A later authorized `--work finish` consumes the full packet, implements semantic topic-owned figures/models/labs and reproduction downloads, checks actual rendered/browser/keyboard behavior and numerical portability, resolves independent review and integrates publication. Do not reopen frozen earlier packets, retrain the study merely on resume, or infer GPU speed from the saved CPU/operator results. Continue the separately authorized next topic, Vision Transformers, in module order.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Expose the approximation and legal path. Edit sparse edges, feature-memory writes/evictions, random-feature settings, compression coefficients, block layout and supported real trajectories. Show removed mass, reachability, normalized summaries, approximation error, future influence and tile occupancy live under a fixed random draw. Choose sparsity or approximation by accessible information, numerical error and actual block work, not a single sparsity percentage.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Implementation ownership and completed writing — 22 September 2026

**Phase: content-prepared; implementation not started.** The following map is the current scope decision under the scratch-and-library rule. Actual code and explanatory teaching are already supplied in the manuscript/packet. Finishing executes, reviews, corrects and integrates these artifacts; it is not assigned to invent the missing core mechanism. The original title, sequence, data and retained experiments are conserved.

| Outcome | Scratch owner | Ordinary tool/library | Bridge | Changed-constraint practice | Scope decision |
| --- | --- | --- | --- | --- | --- |
| Sparse legal edges and efficient local gather | mechanism-calculations.py::selected_attention; attention_compression_bridges.py::gathered_window | SDPA explicit dense mask as independent semantics oracle | Identical legal windows/normalization; dense comparison not sparse execution | Width1 null, tail positions, distant edited value | Local O(LW) gathered reference; hardware blocking later |
| Chosen-kernel recurrence and positive random features | lesson §7 complete recurrence; mechanism-calculations.py Gaussian/orthogonal sampler; AttentionForecaster | Torch tensor/autograd model and ordinary optimizer | Dense versus recurrent all-parameter gradients, exact chosen kernel versus approximate softmax | Zero overlap, cached feature resampling, common-key stabilization and nondivisible chunks | Local; no universal error monotonicity or unbiased normalized-ratio claim |
| Linformer length projection and Nyström landmarks | attention_compression_bridges.py::linformer, segment_landmarks, nystrom | SDPA compressed slots; torch.linalg.pinv for small middle system | Forward/all Linformer gradients; all-landmark identity; tolerance/rank contract | Length11/four landmarks; nearly duplicate landmark threshold | Local complete deeper operators, noncausal summaries explicitly stated |
| Content routing and modern sparse families | lesson §§2/6 explained LSH/cluster/indexer mechanism and cost boundaries | Prepared RWKV owns gated/delta memory; planned GPU attention kernel/serving owners own hardware systems | Candidate discovery differs from attention; selected edges versus occupied blocks | Indexer cost and sparse tile exercises | Family interpretation, not full Reformer/Routing/DeepSeek reproduction promise |
| Real causal forecast and quality comparison | author-calculations.py complete three-model data/train/inference/gradient paths | PyTorch optimizer/state_dict and retained results | Frozen protocol, cache prefixes, known future exclusion | Near versus remote edit, constant-value null, honest baseline | Local retained experiment, no retraining for a preferred result |

**Reuse status:** references to other packets in Deep Learning are prepared manuscripts, not newly published implementations. The existing published historical article at the same route does not certify the replacement code. Implemented mathematics/normalization owners are linked as prerequisites; this packet owns only its new operations. Preserve the complete prerequisite program and required file links when packaging the download.

**Authoring checks and evidence:** [the writing report](../../implementation-depth/PREPARED-ATTENTION-WRITING.md) and [source-bound manifest](../../implementation-depth/prepared-attention-writing.json) distinguish source review, small new arithmetic/API probes and unexecuted optional routes. Existing fit outputs keep their original dates and evidence; no new large fit, GPU benchmark or full scientific replication is claimed.

**Deferred phase-two work:** execute the supplied native code under the recorded/compatible versions, compare independent references on meaningful changed cases, review every claimed output and derivative, integrate accessible topic-specific visuals and exact matching code downloads, then perform browser/layout/performance checks. Resolve failures by updating the content checkpoint; never mark these written additions as implementation complete without that evidence.
