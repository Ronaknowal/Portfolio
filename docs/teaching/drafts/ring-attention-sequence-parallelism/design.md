# Ring Attention: author design and continuation

Stable ID `ring-attention-sequence-parallelism`, Deep Learning Fundamentals position 36. Root author, 13 September 2026. Research and writing only. The manuscript and detailed specifications are ready for later implementation once root binds the completed checkpoint. Runtime, publication, browser labs, distributed execution, formal independent review and integration remain deferred.

## Scope and learner route

Consumed the current teaching standard, domain/design workflow, code/retention requirements and actual `node scripts/build-curriculum-inventory.mjs --topic ring-attention-sequence-parallelism --work content` output. No topic-specific destination note or bespoke brief was returned. The GPU/systems playbook is appropriate here: begin with a correct computation and explicit ownership, then distinguish arithmetic, communication and performance.

Read the complete original source in consecutive ranges 0–235, 235–485, 485–745 and 745–end; later rechecked its actual heading list. Original `src/learn/data/topics/ring-attention-sequence-parallelism.jsx` SHA256 is `bcd2f2b55ef4c16c93b74892df46d6c05075a682db3cea5e575c25bf32b16e7a`, at baseline `8c5da59f18516be77c29d5aeeafca3decca4f738`. Preserve this runtime source in content-only mode.

Actual sequence is Hyena → Ring Attention → Advanced Optimizers. Earlier Self-Attention, GQA and Positional Encodings supply direct prerequisite links, with local refreshers rather than a forced route detour. Keep the stable ID and catalogue title; no rename or extra topic is necessary. The additional mask, target, gradient and decode details are directly owned by this systems topic. Full CUDA kernels, cluster deployment and serving scheduler design remain deeper implementation/specialist work, not omitted prerequisites to the explanation.

Learning outcomes: explain why blocks need global normalization; trace exact token identity through a device ring; implement the small CPU reference; distinguish useful pairs from executed tiles; calculate bytes and an explicitly hypothetical overlap model; compare sequence/head reshards; identify backward and target-boundary dependencies; interpret fixed-model equivalence without treating it as accuracy or speed evidence.

Start with four devices reading one sequence. Progress from one query's four scalar values to labeled packets, causal grids, timing dependencies, alternative tensor ownership and gradients. The real example preserves a previously trained attention layer while changing only its execution plan. Two incorrect predictions remain visible. This systems topic does not need a new model fitting campaign to establish its defining claim.

Seven representation families serve distinct mechanisms; five can support meaningful fresh-input investigations. Static or stepped explanations remain valid where a full lab adds little. All graded states require an unset prediction, exact input/model revision, actual computation, changed examples, null controls, reset/invalidation and accessible alternatives. No browser components were built.

## Original coverage and consequential corrections

| Original agenda | Retained/repaired content |
| --- | --- |
| 1 Why it exists; 2.1 memory ceiling | Explicit dimensions and bytes replace unconditional single-GPU impossibility. Linear activation storage is separate from model/optimizer/workspace memory. |
| 2.2 ring; 3.3 mapping; 4.1 loop; 6.1 rotation | Labeled KV movement, P computations/P−1 necessary forward transfers, uneven ownership, both directions and logical output ordering. |
| 2.3 overlap; 3.4/4.2 accounting; 4.3; 6.2/6.3 | Separate payload bytes, GEMM FLOPs, latency, double buffers, accumulator dtype, aliases and explicit hypothetical timelines. No invented device benchmark or guaranteed overlap. |
| 2.4 Ulysses; 5 landscape; 7 decision matrix | Exact sequence/head resharding, retained 1/P payload and head/GQA constraints; remove false unchanged single-device memory ceiling and arbitrary method ranking. |
| 2.5/4.4 striping; 6.4; 8.4 | Derive actual contiguous/striped counts and zigzag alternative. Contiguous ratio tends to 2P−1 for large c, not P. No invented sqrt(P) striped imbalance law. Kernel tile granularity is explicit. |
| 3.1 tile sums; 3.2 online accumulator; 9.1 | Derive stable numerator/denominator updates, initial/empty states, common-score and arrival-order nulls. Do not average normalized block outputs or confuse rescaling with duplicate correction. |
| 5.1 open kernels; 5.3 DTensor; 9.2/9.3 | Current maintained sources and installed API contract; no nonexistent async_op argument, automatic DTensor ring claim or blocking cyclic-send pattern. |
| 5.2 frontier deployments; 8.5 mesh | Remove unsupported closed-vendor architecture claims and fabricated Llama configuration/MFU. Explain verified public framework concepts and orthogonal mesh ownership instead. |
| 5.4 inference | Distinguish prefill from single-query decode; derive stationary-KV summary merging and separate paging/cache scheduling. |
| 8.1–8.3 scaling | Distinguish fixed global L, fixed local c and fixed dataset tokens. Linear context capacity is not fixed duration or linear total attention work. |
| 9.4 rank ordering; 9.5 imbalance; 9.6 mask; 9.7 drift | Reverse circulation is valid with identities preserved. Global positions/document membership, target shift, weighted loss and dropout recomputation prevent semantic drift. Exact arithmetic is not bitwise equivalence. |
| 10 references; 11 self-check | Verified annotated primary/alternate resources and eight changed exercises with closed hints/solutions; add backward ownership, real entity edits and honest output evidence. |

No topic was removed. Incorrect original metadata about Ring's venue, Striped authors and claimed benchmark configuration are not carried forward. The old synthetic “peak memory” and wall-clock graphs are replaced by named analytical inventories/timelines, with no claim of measured values. Retained performance questions are answered through assumptions and dependencies rather than unsupported universal winners.

## Canonical source agenda and reviewed extent

Canonical source: [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/html/2310.01889v4), Hao Liu, Matei Zaharia and Pieter Abbeel. Read the actual complete heading agenda. Read introduction through section 5.3, section 5.4, related work/conclusion, Appendix A's forward/backward code, B's stated experiment settings, C's inference argument and D's fixed-dataset FLOP analysis. This is a coverage reading, not replication of the paper's training, performance tables or every citation.

| Actual canonical section | Disposition in this lesson |
| --- | --- |
| 1 Introduction | Long-context ownership motivation with scoped resource claims. |
| 2 Large Context Memory Constraint | Separate stored score matrix, linear tensors and unsharded model state. |
| 3 Ring Attention with Blockwise Parallel Transformers | Own complete stable merging, ring schedule, masks, overlap and backward derivation. Explain that positionwise FFN can remain sequence-sharded. |
| 4 Experimental Setting | Source context for empirical claims; no transplantation of its hardware constants into our calculations. |
| 5.1 Maximum Context Length | Capacity versus learned capability; no “near-infinite” literal promise. |
| 5.2 Model FLOPs Utilization | Timing/performance measurements require an actual setup; our cost model is not MFU. |
| 5.3 In-context RL; 5.4 LLM Performance | Explain long trajectory/document applications and limits of task-specific evaluation. Do not reproduce benchmark results as our own. |
| 6 Related Work; 7 Conclusion | Locate FlashAttention, Ulysses and other mesh dimensions. Current alternatives are treated on their actual contracts. |
| Appendix A Code | Read JAX collective permutation and custom backward ownership. Own CPU code is a different transparent teaching implementation, not copied distributed code. |
| Appendix B.1/B.2/B.3 experiment details | Preserve distinction among capacity, utilization and line-retrieval settings. Do not infer all experiments use the same precision or workload. |
| Appendix C Inference requirement | Read critically: its bandwidth/FLOP unit argument is not a universal decoding overlap proof. Derive the separate one-query stationary-KV alternative. |
| Appendix D Training FLOPs Scaling | Explain fixed-token dataset versus fixed-sequence scaling with independent elementary arithmetic. Do not repeat source table typos or extrapolation as measured runs. |

The source uses claims such as fully hidden communication under its overlap premise; this lesson makes the premise visible. Appendix A executes P permutations for its convenient carry pattern, while our explicit forward schedule needs P−1 transfers to visit all KV blocks. These are different schedules, not contradictory arithmetic. The CPU reference uses complete local score blocks rather than an on-chip FlashAttention kernel.

## Supporting primary sources and alternatives

- [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867): read section 3, Algorithm 3, its induction and section 3.1's parallel merge. Retain scalar/vector-summary reasoning and exact nulls; no benchmark reproduction. Our examples and calculations are independently constructed.
- [FlashAttention](https://arxiv.org/pdf/2205.14135): read relevant section 3/Algorithm 1 and Appendix B.2–B.4 derivation, complete displayed forward/backward algorithms, RNG-state and recomputation details. No claim of reading all 34 pages or implementing its CUDA kernel. Scope distinguishes stored probability tiles from global row statistics and dtype-dependent equality.
- [Striped Attention](https://arxiv.org/html/2311.09431v1): actual authors are William Brandon, Aniruddha Nrusimha, Kevin Qian, Zachary Ankner, Tian Jin, Zhiye Song and Jonathan Ragan-Kelley. Read actual agenda (1 Introduction; 2 Background/causal/Ring/tiling; 3 Evaluation; 4 Discussion; 5 Related Work; 6 Conclusion; 7 Acknowledgments; A full results). Read body through section 4 and its visible results tables, with selected related-work context. Preserve tile/round caveats. Its A100 3B 256k speedup of 1.45 uses TP2/SP4; this is not the original lesson's invented N16 general result and is not needed as a learner performance promise. Use global k≤q independently instead of blindly copying ambiguous pseudocode inequalities.
- [DeepSpeed-Ulysses](https://arxiv.org/html/2309.14509v2): read abstract/background and sections 3.1–3.4, 4.1–4.2. Actual method gathers all positions for fewer heads, so the old claim that memory remains unsharded is false. Our per-rank nonlocal-byte count has explicit assumptions and is not the paper's physical per-link topology model.
- [Megatron context parallelism overview](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html): read overview, benefits and enablement with its TP/CP communication diagram description. Also inspected official parallelism guide and shared-parameter gradient notes. Current documentation can evolve; no library deployment here.
- [PyTorch context-parallel tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html): complete instructional body, setup, sharding/unsharding example, rotation modes and limitations read from 2.14 documentation. Public experimental context and private helper imports are version-sensitive. GPU example unexecuted.
- [TorchTitan author explanation](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082): read introduction, integration, mesh composition and implementation through tensor sharding/rotation. Author-origin technical post supports paired early/late chunks; no claim to read every later benchmark/comment.
- [DeepSpeed current HF integration](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/): actual agenda and body through weighted loss, target shifting and tiled-loss entry read. Compare with the older Megatron-DeepSpeed tutorial, whose old compatibility/version statements should not be treated as universal current restrictions. No model download or GPU training.
- [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention): README variants, adapter example, benchmark methodology and known limitations read. README's numeric results are not ours; unsupported dropout/window settings and extra FP32 buffers matter. Do not infer all advertised variants support every argument.
- Installed Torch 2.14 `batch_isend_irecv` signature and docstring inspected directly. It accepts a list of P2POp records, returns requests and documents ordering/device/group caveats. The web distributed URL redirected without useful body; local API evidence is primary for the exact signature. No collective was launched.
- [Stanford CS336 2025](https://cs336.stanford.edu/spring2025/): actual syllabus, prerequisites, assignments and parallelism lecture entries read. Official Stanford Online indexed [Lecture 7 video](https://www.youtube.com/watch?v=l1RJcDjzK8M) identity/title/publisher verified; direct playback-page retrieval failed. Full video unwatched, no timestamps claimed. Annotate as broader systems background rather than a Ring-specific demonstration.
- UCI data attribution and inherited frozen fit are in data-provenance.md. Required actual source and frozen model definitions were read from the Self-Attention packet; its completed training/checkpoint was reused without reopening the previous campaign.

## Evidence and author closure

Three semantic Python programs are complete and executed. The reference's forward demo passed. The partition study calculated scalar/fresh/offset traces, dense/causal/packed/empty-row identities, reversal and rank-renaming nulls, direct/autograd/finite-difference derivatives, missing-metadata/gradient contrasts and frozen real input edits. The systems program calculated ownership/tile/scheduling/byte/target/loss/rotary cases; its one later rerun added the explicit rotary inset without repeating model work.

Manuscript reread completely in consecutive ranges, including the full displayed code and all solutions. Full visual specifications were rewritten into readable handoff prose after the first draft's compressed notation made them harder to consume. Numerical contracts, meaningful input controls, fresh cases and deferred checks were preserved. This revision was for implementer clarity, not a new calculation campaign. The displayed program is the complete executable reference, with minimal validation and mechanism occupying most lines.

Author reread checks: first-pass intuition before notation; same scalar example through stable arithmetic; dimensions/units and conditions local to their claims; real input/output preserved with actual mistakes; different visual forms for different questions; independent changed practice; useful source annotations with honest review extents; appropriate same-module sequence links. Eight paired hint/solution disclosures plus one initial reveal are closed by default. No training plots or hardware timings were invented.

Bounded structural/source checks passed: all three Python programs parse; all JSON files parse with nonfinite constants rejected; the full displayed Python code exactly equals its executed reference file; 17 disclosures are closed; original runtime source SHA256 is unchanged. The 7,548-word manuscript was complete at the numerical check; subsequent spacing repairs do not change its instructional content. Selected NumPy probabilities agree with the inherited float32 probe to 4.86410143e−8. The readable specification was also fully reread. This own reread is not formal independent phase-two review. Exact checkpoint hashes in the central ledger remain authoritative; do not infer implementation completion from the presence of runnable author programs.

## Later authorized finish

Run the actual `--work finish` preflight and read the entire packet. Implement the specified figures and investigations, extract only required on-demand model/data assets, render formulas and closed disclosures, check native/browser parity and genuine entity edits, and complete independent content/correctness/learning review plus mobile, keyboard, accessibility, performance and failure-recovery checks. Keep the semantic runtime topic filename and compatible stable route. The implementer may improve the representations with a reason, preserving scope and correctness.

Retain the 11 packet files (manuscript, specifications, design, provenance, three programs, two result files, data and model) until their handoff is consumed. Do not eagerly import the full author JSON or dataset into navigation. No new disposable scratch captures or copied media were created. Root owns shared ledger/handoff/inventory changes.
