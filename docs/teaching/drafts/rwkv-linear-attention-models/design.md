# RWKV & Linear Attention Models — content design and continuation

Prepared13September2026 by classical_probabilistic_content. **Research/writing complete; implementation not started.** Root owns the central checkpoint and shared scope/ledger. This packet preserves the stable topic ID and published runtime source; no website code, catalogue, manifest, navigation or lab was changed.

## Preflight, baseline and scope

Actual command run before drafting: node scripts/build-curriculum-inventory.mjs --topic rwkv-linear-attention-models --work content. The topic is published historically, content in progress, implementation not started; module position19, Recurrent & Sequence Models, frontier. Individual design required. No destination note exists for this topic; the unassigned note did not contain relevant unresolved scope.

The complete original source, all1,238 lines of src/learn/data/topics/rwkv-linear-attention-models.jsx, was read in contiguous ranges. SHA256:
acccaa77eeefb66aeaacbe7b0b7fe7491028d1853910b65cf627cb09cfe498b0.
The final author check matches that same source. Baseline commit8c5da59f18516be77c29d5aeeafca3decca4f738. Existing source is not automatically approved; it contains useful breadth and several consequential unsupported claims.

**Title decision:** keep “RWKV & Linear Attention Models.” The broad existing title already owns the weighted-summary/matrix/correction progression. Current experimental RWKV-8 directions receive a proportionate scoped paragraph with honest external-storage accounting. They do not justify inventing a stable released architecture or replacing the core4–7 teaching with a release catalogue.

**Sequence:** preceding SSM→this RWKV/linear-memory lesson→Self-Attention & Multi-Head Attention. The module places these alternatives before the dedicated transformer fundamentals, so§1 refreshes query/key/value, causal softmax, projections and shapes locally. Actual local route links use module=deep-learning-fundamentals. No reader navigation is reordered.

**Ownership coordination:** classical_feature_content owns later Sparse & Linear Attention Variants, including full approximation/sparse/low-rank comparisons. This packet teaches the recurrence and necessary performer/linformer distinctions plus concise exact RetNet/GLA/delta bridges. Parent received the RWKV-7 theorem-condition finding. The later attention owner received current FLA discoveries (KDA/Gated DeltaNet2/preconditioned variants) for scoped judgment; no other draft was edited. Root owns shared routing. No broad catalogue audit.

## Learning design

The running question is whether a stream needs a weighted summary, a growing transcript or a mapping that can revise a particular address. SensorA/B writes tie the kernels to a concrete retrieval requirement. A real hand-trajectory task then forces the projections/readout to be learned and exposes state-continuation mistakes.

Core route after the introduction:§§1–5, investigationsA/B, real-data§8 and early changed practice. Advanced branches: positive random-feature identity, version5–7 matrix updates, theorem conditions, RetNet/GLA/hardware choices. The lecture remains self-contained for the core task. Later links supplement rather than replace necessary explanations.

| Learner hurdle | Mechanism/explanation | Representation and transfer |
| --- | --- | --- |
| Query/key/value sounds like terminology without operations | One head, dot scores, normalization and weighted vector sum | Causal transcript diagram and numerical4.25 sum |
| A fixed summary seems magical | Derive outer-product S and weight total z by distributivity | State grid, contribution ledger, editable kernel investigation |
| “Linear attention equals softmax” | Exact chosen-kernel regrouping versus different ELU kernel and random approximation | Matched weight bars;8.807971 versus6.666667 |
| Causality disappears during chunking | Incoming state plus masked local contribution before normalization | Two-source chunk diagram and remainder-size checks |
| Current bonus becomes permanent | Separate read-before-write equations and decay indexing | Two-stage circuit and bonus/state invariance |
| Numerical stabilization seems to clip information | Common log scale preserves a signed numerator/positive denominator ratio | Scale ruler and+1000key-offset test |
| Central state is mistaken for the complete model | Expose token-shift slots, channel branch and readout accumulation | Complete block diagram and real split/carry/reset |
| Repeated key should mean replacement | Derive delta gradient, then demonstrate correlated-key interference | Key compass, residual tile and changed-address practice |
| Version names hide changed mathematical objects | Compare vector summary, matrix rows and rank-one correction with explicit orientations | Structural version map and exact2×2update |
| A stability or speed headline becomes universal | Separate theorem conditions, persistent bytes, full training memory and workload timing | Conditional theorem branch, exact inventory curves, critical reading exercise |
| A mathematically correct computation is assumed accurate | Real trained fits include errors and unfavorable seed outcomes | Measured data table, continuation equality separate from class correctness |

## Conservation and corrections

| Original useful coverage | Disposition in prepared content |
| --- | --- |
| History and model-family overview |§§1/6/7 explain the progression and genuine distinctions; remove unsupported claims that every family is the same kernel or has the same state size. |
| Normalized kernel reassociation, S/z state and feature choices |Preserved and fully derived in§2; causal chunk program in§3. |
| Performer and Linformer |Preserved as scoped mechanism bridges with actual primary sources and later owner link. Correct positive Gaussian features and sequence-axis compression. |
| RWKV4 token shift, weighted recurrence and stable state |Preserved and expanded with read/write index proof, signed values, scale invariance, complete trainable block and program. |
| Versions5/6/7 |Preserved with actual matrix orientation, current-token terms, dynamic token shifting/retention, correction update and block components. |
| RetNet, GLA, SSD connections |Preserved with exact core equations, fixed versus input-dependent retention and full-block qualifications. |
| Naive/stable comparison and chunk computation |Replaced with complete original NumPy programs, exact numerical fixtures and real trained state-continuation checks. |
| Full RWKV4-like block code |Complete readable original instructional PyTorch program, with disclosed small configuration and actual fitting. |
| Pretrained/inference/FLA integration examples |Version-aware checkpoint/tokenizer/state guidance and verified maintained project resources; no unsupported current-class compatibility or fabricated local checkpoint outputs. |
| Timing, memory and decay plots |Memory becomes exact formula inventory; decay becomes calculated half-life; claimed timing rankings removed unless an actual workload supports them. |
| Practical decision criteria, failure modes and practice |Rewritten around observable state/error conditions,10 changed problems with20 closed hint/solution disclosures, and four input-driven investigations. |

Specific corrected original claims:
- Softmax does not require storing the full quadratic score matrix; exact FlashAttention changes memory traffic/storage while dense arithmetic remains quadratic over a full sequence. Cached decode per step grows linearly with history, not quadratically per step.
- No universal width-based runtime crossover, unsupported8×speed claim, fixed performance gap, “best on phones” or inevitable failure wall at a named context length.
- ELU+1 is a chosen kernel, not exact softmax. FAVOR+ uses positive random features; generic sine/cosine features are not that construction. Unbiased kernel estimates do not imply an unbiased normalized ratio.
- Nonnegative features alone can give zero total weight. Causal prefix masks are structural; no meaningless mask/feature-map “if and only if commutation” claim.
- RWKV4 current bonus enters its read, not stored write. Raw exponentials overflow with large keys; permitted decays need not overflow merely from age.
- An identical ε added before/after rescaling changes the operator differently. Stable numerator may be negative; log scale tracks weights.
- Original RWKV4 CUDA executes a serial time loop per parallel batch/channel element; “parallel training” does not certify a time-parallel scan implementation.
- Versions5/6 retain special current-token terms and change state constants. Version7 state is transposed relative to5/6; generalized removal/write is not universally ordinary SGD on one shared local loss.
- RWKV7 AppendixC product bound assumes time-independent a. AppendixD's strong constructions introducec=2 and boundary values, unlike releasedc=1. Do not transplant these into unconditional trained-model guarantees.
- Constrained retention cannot drift into positive growth merely because its raw parameter is positive. Out-of-vocabulary inputs are not a generic explanation for NaNs.
- Persistent state information capacity is not a bound on the entropy of the original history. State acceptance length is not perfect recall length.
- Streaming preserves the complete state and weighted readout, not just the central recurrence tensor.

## Canonical coverage and actual research

Research date13September2026. Papers/official source used for technical claims; resource discovery articles are annotated alternate learning routes, not sole proof. These locators record what was actually read, not a claim to have reproduced every paper or watched a recording.

| Canonical source and actual reading | Coverage decision / lesson location |
| --- | --- |
| [Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf),§§3.1–3.4, experiment organization§4 |Feature-map operator, causal state, memory-conscious backward discussion, recurrent interpretation→§§2–3. Images/ASR experiments establish context only; do not copy source's overly broad uncached-inference wording. |
| [RWKV4](https://aclanthology.org/2023.findings-emnlp.936.pdf),§§2–4 architecture/training, full AppendixD, initialization context and evaluation organization |Ground-up recurrent block, token shift, WKV, stable representation and five vector slots→§§4–5/7. Large model scores are not reproduced as our evidence. |
| [Official RWKV4 forward CUDA](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu),full125-line file, especially forward loop and launcher |Read/write scales, no arbitrary ε, serial time loop plus batch/channel parallel work. Kernel source inspected, not executed on GPU. |
| [RWKV5/6](https://arxiv.org/pdf/2404.05892),TOC,§§3–4.2.2,context§2 and tokenizer boundary |Matrix state, per-head norm/SiLU, current bonus, static/dynamic retention, low-rank dynamic shift→§6. Full world-tokenizer study stays with tokenization owner; task only needs compatible tokenizer/version. |
| [RWKV7](https://arxiv.org/pdf/2503.14456),TOC,§§2–4.2,§§5–6 context, AppendixC theorem/proof and D.1/D.2 construction introduction |Generalized delta mechanism, parameter roles, value residual, readout bonus, feed-forward change→§6; exact conditional stability/expressivity qualifications. Not a claim to verify every appendix proof or all benchmark rows. |
| [RetNet](https://arxiv.org/pdf/2307.08621),§2 recurrent/parallel/chunk and gated multiscale retention |Scalar decay per head, positional/norm/gate context→§7; sufficient recurrence bridge, not full checkpoint replication. |
| [GLA](https://arxiv.org/pdf/2312.06635),§2.1–2.2,§3 hardware/materialization discussion,§4.1–4.3 including gate parameterization table |Row gate versus general matrix gate, recurrence/chunk split, log-product instability and hardware motivation→§§3/7. No benchmark numbers transplanted. |
| [Performer](https://arxiv.org/pdf/2009.14794),§§2.3–2.4,positive-feature lemma and orthogonal sampling |Original expectation derivation and distinction from finite normalized estimator→optional§2; later sparse/linear owner covers deeper approximation analysis. |
| [Linformer](https://arxiv.org/pdf/2006.04768),§3 spectrum/theorem context,§4 projection mechanism |Sequence-axis learned compression and causal warning→§7; source theorem is not restated as universal exact low rank. |
| [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) README/current repository, and [project architecture history](https://wiki.rwkv.com/basic/architecture.html),version4/7 core example andversion8 DeepEmbed/ROSA sections |Current implementation discovery and proportionate experimental extension; reject marketing of external RAM/SSD lookup as literally free storage. |
| [FLA](https://github.com/fla-org/flash-linear-attention),current News/Models/Installation/Usage/Generation/Hybrid sections |Actual model/kernel integration resource and current family breadth. Not installed/executed; APIs/backend must be pinned in future use. |
| [UCI181](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement),current page and full original names read during same-day preceding packet; byte-original source reuse |Licensed real data and explicit role protocol→§8/provenance. No new network download falsely claimed. |
| [Oxen illustrated article with embedded discussion](https://www.oxen.ai/blog/how-rwkv-7-goose-works-notes-from-the-author),metadata/introduction,memory motivation,delta/update,parallelism/limitations/demo sections |Annotated alternative involving RWKV contributor Eugene Cheah. Its attention-complexity shorthand is explicitly qualified. Embedded YouTube identity4Bdty7GOrbw obtained from host link; direct watch/embed retrieval failed, recording not watched. Host article remains useful and accessible. |

Canonical section-list check: the central sources collectively cover input preparation, query/read, write/decay/erase, normalization, recurrence/parallel/chunk computation, outer objective/training, versions, capacity/stability, empirical evidence and deployment. Every core mechanism is taught here. Detailed favorable random-feature variance bounds, Linformer approximation proofs, GPU kernels and newer delta variants stay at the later attention/GPU owners; the local bridges make their relationship explicit. No missing source section is silently treated as already taught. A literature history is proportionate, not an exhaustive release tracker.

Failed/limited resource retrievals: guessed ChatRWKV raw model and RWKV_v7_demo URLs did not resolve; they are not learner links or implementation evidence. A guessed wiki pip route failed; lesson instead links the verified architecture page and maintained repository. The video direct URL failed while the host article and embedded identity were readable; no “watched” assertion. No large checkpoint/download campaign or source-license uncertainty was introduced.

## Programs, data and author calculations

- linear_memory_mechanisms.py: complete NumPy operators and exact fixtures. Direct versus recurrent/chunk agreement, ELU/softmax contrast, signed/stable RWKV, additive/delta update, RWKV7 matrices, gated-memory outputs, cache inventory. Writes mechanism-results.json.
- trajectory_memory_models.py: complete PyTorch/NumPy/sklearn data preparation, two small model kinds, actual supervised fitting, best-validation checkpoint selection, metrics, chunk continuation and safe numerical-array export. Four fits at seeds17/41; original ordered baseline, not a paper checkpoint.
- author_calculations.py: actual frozen real-input edits/carry/reset/reversal/future-prefix/null fixtures and autograd delta-gradient. Source7 is worked; source20 is fresh default. Corrected a helper variable-shadowing metadata error before freeze without changing trained fits.
- The two displayed Python blocks were extracted and executed: complete chunk program prints[2,4,2.5,3]; displayed training step ran on five fitting examples with a fresh model/seed93, finite loss3.0247814655. This bounded step is separate from the four retained predeclared fits.
- author-checks.json: original source hash, five actual route IDs, all relative learner downloads,20 closed practice disclosures, source-data hashes,330 unique rows/30 duplicate copies,220/50/60 disjoint roles, finite arrays, consistent confusion counts and explicit unperformed checks.
- Numerical evidence is exact derived or executed small arithmetic/fits as labeled. No speed experiment, full language benchmark or accelerator result is implied.

## Author reread and learning-experience checklist

The author reread the entire9,000+word manuscript in three contiguous views and the complete visual specification. The reread tightened numerical prose spacing and confirmed:
- The learner's concrete problem precedes vocabulary; the local attention and matrix-orientation refreshers make the actual sequence readable.
- A visible core route avoids forcing every advanced family/theorem detail on the first pass. Essential equations and worked reasoning remain visible.
-23 inline representations are chosen for distinct mechanisms; four investigations arise from four actual questions, not a quota.
- All four investigations have fresh unanswered inputs, input-bound initially unset predictions, genuine sequence/key/coordinate edits, contrasting and null fixtures, explanations, reset/invalidation, mobile/accessibility/bounds and explicit phase-two work.
- The real investigation uses a meaningful failure case: fresh source20 is initially misclassified, carrying preserves computation, resetting can change confidence without flipping the RWKV class. The stronger source7 class-flip contrast remains available.
- Frozen model outputs/curves are clearly separated from exact toy calculations and formula storage curves. Unfavorable seed outcomes are retained.
-10 changed problems have separately closed hints and explained solutions; they assess transfer, geometry, initialization, nonunit keys, weighted pooling, storage and evidence.
- Cautions are consolidated in one diagnostic table and necessary scope paragraphs, not repeated beneath every worked result.
- Complete programs, data, setup/run/results, annotated alternatives and actual next-module link are present; no required concept is replaced by a resource link.
- No formal phase-two independent review, browser inspection, generated visual/lab or production implementation has been claimed.

## Freeze and next action

Content packet ready for root's scoped reconciliation and exact-source checkpoint. Preserve all pending draft/data/program/evidence files. Root will bind content completion; implementation stays not started. Do not retrain these fits merely to get a better result or rerun unrelated historical checks.

A future authorized finish request must use --work finish, consume this packet, implement topic-owned figures/labs with lazy loading and the stated state/prediction contracts, port only necessary frozen data, independently verify the new shipped model, execute any newly added displayed code and complete correctness/learning/browser/accessibility/performance/integration review. Reconcile any content corrections against these exact sources and keep implementation status separate.

