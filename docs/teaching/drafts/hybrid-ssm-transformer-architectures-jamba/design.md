# Hybrid SSM–Transformer Architectures (Jamba): author design and evidence

Status: content-first packet complete; final author reread and bounded evidence checks finished on 13 September 2026. Implementation has not started. Root owns the shared delivery ledger, hashes, scope and checkpoint. This file records this topic's decisions and evidence rather than creating another status queue.

Stable ID: hybrid-ssm-transformer-architectures-jamba. Module: Deep Learning Fundamentals & Architectures, position39; final topic of the current next-30 content-only scope. Previous: neural-ode-continuous-depth-models. Next unrequested: titans-multi-memory-architecture. No runtime sequence or catalogue edits.

## Preflight, source baseline and scope

Actual command run: node scripts/build-curriculum-inventory.mjs --topic hybrid-ssm-transformer-architectures-jamba --work content. It reported frontier level, individual design/prerequisite review required, no explicit prerequisite IDs, content in progress and implementation not started. No topic-specific destination note exists. The unassigned DSA note was already resolved and is unrelated.

Read Portfolio/AGENTS.md, current LESSON-AUTHORING-HANDOFF.md and docs/teaching/DEEP-LEARNING-ARCHITECTURES-CONTENT.md, with the current teaching standard, topic-design brief, domain and code/ownership/retention instructions from this active authoring assignment. Historical plans do not override them.

The full original JSX lesson was read, including its from-scratch training program, commented outputs, deployment snippets, all diagrams/plots, failure modes, references and open-answer exercises:
src/learn/data/topics/hybrid-ssm-transformer-architectures-jamba.jsx.
Baseline SHA-256: e71d445f9077cc2dcf9f265b749d47a631a5022e29f3a93f696e87a8a5d457d8.
Recover original from commit8c5da59f18516be77c29d5aeeafca3decca4f738 rather than retaining another archive copy. Source remains untouched.

Suggested display title: **Hybrid SSM–Transformer Architectures: Jamba and Complementary Memory**. The existing stable ID remains appropriate. This is an explanatory subtitle proposal only; phase one does not rename the catalogue.

Primary scope: combining recurrent state and address-sensitive attention across depth, Jamba's separate MoE dimension, actual release shapes, correct cache/weight/arithmetic accounting, training/evaluation and stateful inference. Proportionate advanced branches compare Zamba, Zamba2, Samba, Hymba and a dated Granite4 hybrid example. These are architectural alternatives, not a new model leaderboard or a replacement chapter for each family.

Local refreshers teach normalized recency summaries, finite softmax reads, causal masking, two residual updates, RMSNorm, selective state shapes, short convolution, GQA and top-k routing before using them. Dedicated SSM, attention, position and MoE lessons remain their detailed owners. Unlike the earlier long-context module entry, this topic follows those dedicated chapters, but does not require the reader to remember every equation.

## What a learner should be able to do

By the end, the learner can distinguish input-position and network-depth axes; explain a concrete state collision and why it is not universal; trace both residual additions; identify the recurrent matrix, short-history and K/V objects; count their bytes with explicit dimensions/dtypes; separate stored versus active expert parameters; recognize Jamba's retained router probabilities; reproduce a small real-data comparison; diagnose a cache fault that preserves the final argmax; and design a workload-based comparison without fabricating an architecture ranking.

Core route is §§1–8 plus practice1–6. Optional arithmetic, alternative architectures and deployment-design branches add depth. The practical study appears after the mechanism and accounting, so the learner can read its actual computation. Practice gives changed scenarios with closed hints and full solutions. No first-pass text depends on a live foundation-model server.

## Hurdles and representations

| Hurdle | Teaching choice | Evidence/representation |
|---|---|---|
| “Attention retrieves exactly” | A finite matching score produces29/11 rather than stored2. | Exact contribution strip and scalar calculation. |
| “All recurrent memory fails copying” | Exhibit a collision for one stated two-number recurrence, then limit the conclusion. | Two paths converge to identical state while keyed reads differ. |
| Confusing layers with tokens | Put depth and position on different axes; inspect actual offset4. | Eight-row mixer/FFN ladder. |
| Dropping one residual path | Name the intermediate u and add the second branch to it. | Two junctions and scalar repair practice. |
| Forgetting convolution state | Show it beside H, then clear only this history in a real fitted model. | Request timeline and actual logit differences. |
| Confusing GQA heads/storage | Four queries share one K/V bank; eight KV heads determine original cache bytes. | Fan-in diagram and memory formula. |
| “Hybrid memory is constant” | Add the remaining full-attention growth term; compare a changed windowed operator. | Stacked exact bands and length-zero null. |
| “Top-2 means normalized selected weights” | Compare retained full-softmax mass to selected renormalization. | Editable score/output router; unselected-score intervention. |
| “A hybrid must win” | Retain all six real fits, including the all-attention assessment advantage. | Actual curves, selected epochs and all errors. |
| “Same class means same function” | Worked cache faults leave class0 while changing logits; fresh K/V fault changes1→2. | Probability vectors, prefix differences and null boundaries. |
| “Shared weights imply one cache” | A two-vector identity-projection counterexample. | Separate parameter ties and explicit cache-sharing arrows. |
| “Fits cache therefore fits GPU” | Count all approximate52B BF16 weights before request/workspace costs. | Separate memory categories and release identities. |

Four investigations have distinct jobs, not a quota: JA memory information loss, JC resource accounting, JD router convention, JB real input/cache behavior. Eighteen figures support their concepts inline. Specifications require unset input-bound results, genuine editable entities, checked contrasting/null fixtures, correct reset/invalidation, accessibility, narrow layout, bounded computation and deferred phase-two verification.

## Original-source conservation and corrections

| Original material | Disposition in the new manuscript |
|---|---|
| Motivation through long-context memory and recall | Preserved, grounded in concrete paired questions; removed universal quality rankings and invented “90%” claims. |
| Intuitive mixed-layer stack | Preserved with two axes, actual offset4 and orthogonal FFN choices. Removed lossless-cache/DRAM analogy and invented depth-specific induction rule. |
| Formal hybrid/SSM/attention/MoE equations | Replaced incorrect missing residual, unqualified timestep update, RoPE/head configuration and two-matrix SwiGLU counts. |
| Eight-layer copy-task program and claimed11/49/98% outcomes | Replaced with a complete actually executed real pen-sequence study and exact recurrence collision. The original comments are not evidence of execution or a capacity theorem. No CPU-versus-CUDA speed ranking is retained. |
| Memory calculator | Preserved and extended with batch, GQA, independent dtypes, short-convolution state, GiB/GB and fixed versus growing terms. |
| Placement discussion | Preserved as an experimental choice. Removed unsupported universal end/start failures and alleged uniform-spacing ablation not present in the cited original main paper. |
| HF/vLLM deployment examples | Replaced with a complete clearly unexecuted revision-explicit generator and an evidence-based serving protocol. Removed independent prompts falsely treated as a conversation and unverified latency/GPU promises. |
| Family comparisons and fine-tuning | Retained proportionate primary-source distinctions, actual Large72layers, Zamba depth sharing, Samba local attention and Hymba parallel heads. Removed Granite3/Falcon-Mamba hybrid conflations and universal LoRA/LR/clipping advice. |
| Visual walkthrough, architecture heatmap, memory graph | Replaced by18 mechanism-specific figures and four interactive contracts. Actual shape/count data conserved. |
| Invented Pareto plot “adjusted for2026” | Removed. Its points have no reproducible common evaluation protocol. Actual learning histories and scalar arithmetic provide honest quantitative visuals. |
| Decision matrix and scaling section | Rewritten as workload/evidence questions; no unconditional context-length winner or assertion that full attention is infeasible beyond a fixed length. |
| Failure modes | Distributed into their natural mechanisms/serving home without repeated warning lists. Kept state identity, precision, kernel compatibility and long-context training concerns. |
| Sources and exercises | Retained useful canonical families, added actual implementation/data/course/serving references, replaced open answers with ten original changed practice questions and complete closed hints/solutions. |

Additional corrections found during research: Jamba current source does not renormalize selected router mass; B/C and low-rank timestep features have internal RMSNorm; the original release has no explicit position encoding; 1.5 Large uses Mamba-1 after scoped comparisons. A model-card example can itself contain inconsistent identifier order. The manuscript asks the reader to inspect the resolvable identifier rather than silently copying it.

## Canonical section coverage

The actual section list of Lieber et al., arXiv2403.19887v1 was read, and the main paper through its conclusion was inspected. It is the canonical organizing reference, not a template copied into the lesson.

| Canonical section | Ownership and disposition |
|---|---|
| 1 Introduction | §§1–2 explain the engineering problem; claims are interpreted in their historical setting. |
| 2 Architecture | §§2–5 teach exact mixer/FFN separation, residuals, Mamba state, GQA and sparse experts with local prerequisites. |
| 3 Reaping the Benefits, 3.1 single80GB configuration, 3.2 throughput | §§4/6/8 separate formula bytes, quantization, measured hardware conditions and workload testing. Historical throughput is not extrapolated to current hardware. |
| 4 Training Infrastructure and Dataset | §6 and §8 explain training/serving dependencies; proprietary dataset details are not invented or replicated. |
| 5 Evaluation; 5.1 academic, 5.2 long context, 5.2.1 needle, 5.2.2 naturalistic | §8 teaches complementary evaluation conditions; §7 supplies a real local controlled protocol. Original synthetic and naturalistic scopes remain distinct. |
| 6.1 Attention/Mamba ratios | §2 explains the actual1.3B/250B ratio comparison, with no universal placement claim. |
| 6.2 Why the combination works | §1 distinguishes representational examples from hypothesis; §8 includes formatting/missing-evidence and joint reasoning tests. Anecdotal induction heads are not a theorem. |
| 6.3 MoE | §5 teaches stored/active counts and operator convention; dedicated MoE owner handles full routing training/distributed mechanisms. |
| 6.4 normalization; 6.5 explicit positions | §3 and §6 explain exact source operations, no-RoPE result, numerical range and the study model's deliberate position-tag difference. |
| 7 Conclusion and References | §9–12 give scope-limited alternatives, application reasoning, annotated references and the actual next-topic bridge. |

Jamba-1.5 section list was also read in full:1Introduction,2Architecture,3Serving(ExpertsInt8,ActivationLoss),4Throughput/Latency,5Training(infrastructure/data,stages,post-training with table/documentQA/tool use/steerability,observations),6Evaluation(academic,chatbot,long-context RULER/Infinite-Bench,multilingual),7Alignment/Safety,8Conclusion. Main text was read through §8, including its reported training/metric conditions. §§2–3 own the release/precision extension, §§5–6 inform practical evaluation, while detailed commercial alignment policy and individual leaderboard entries are not recreated in this architecture lesson.

## Research record and claim locators

Retrieved13September2026. Papers and public docs were read through web tools; targeted full primary HTML/model-code downloads provided contiguous source reading where web windows truncated text. Downloaded source extracts are disposable scratch, not a second permanent source archive. The retained implementation source hash below identifies what was inspected, without claiming a released version from current main.

| Source and actual reading extent | Claim use |
|---|---|
| https://arxiv.org/html/2403.19887v1 — section list and full main§§1–7, including§6.1–6.5 and table conditions. | Original architecture, ratio comparison, hypotheses, normalization, positional result. Source table4 and figure5 caption disagree on one ratio label; lesson uses explicit table/text1:3 vs1:7, not the caption typo. |
| https://arxiv.org/html/2408.12570v1 — full main§§1–8 and section list; extracted source hash6050c02932e13ce1ddf7b20e7e5d7aa8e1668f8cbb33714e604742bc6114c991 for raw downloadedHTML. | Large72layers/8192/64queryheads, ExpertsInt8, activation penalty, Mamba-1choice and evaluation/training constraints. |
| https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/config.json — entire config fields. | Offset4/period8, oddexpertlayers, width4096, 32query/8KV, state16,conv4,expand2,FFN14336. |
| https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/jamba/modeling_jamba.py — complete relevant MambaMixer,MLP,Experts,SparseMoeBlock and both decoder-layer implementations; model cache setup inspected. RawSHA d042581162b28edae2635e955f117de878138d635ea5a021b38077916bbf83ea. | Actual internal norms, cache objects, two residuals, SwiGLU matrices and non-renormalized selected weights. Historicalv4.40.1 retrieval failed; do not claim that tag was verified. |
| https://huggingface.co/docs/transformers/model_doc/jamba — configuration API, generation examples and kernel/quantization notes read. | Optional production path and need to match installed software. A documentation wording typo about latency is not copied. |
| https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.7 — public model information, details and usage through Transformers notes. | Dated successor context, BF16 residency and model-ID inconsistency; no access agreement accepted or weights downloaded. |
| https://arxiv.org/html/2405.16712v1 — introduction,§Ioverview/contributions and§IIarchitecture through start ofII-B; remaining paper is not claimed fully read. | Repeated shared module, embedding concatenation and limited parameter-sharing claim. Do not infer identical cache contents from tied projection weights. |
| https://huggingface.co/Zyphra/Zamba2-7B — full public model card; https://www.zyphra.com/our-work/the-zyphra-training-cookbook — architecture-rationale excerpt. | Mamba2, two shared modules, depth LoRA and RoPE; explanatory alternative rather than competitive ranking. |
| https://arxiv.org/html/2406.07522v1 — introduction and full§2methodology,§3opening; other sections not claimed complete. | Layerwise Mamba/SWA/MLP and fixed-window read boundary. |
| https://arxiv.org/html/2411.13676v1 — introduction and full§2.1–2.3, figure/table conditions and start of following discussion; appendix locator list inspected. | Parallel heads, normalization/rescaling, explicit cache sharing and meta-token initialization. No novel guarantee about human memory or unlimited perfect recall. |
| https://www.ibm.com/granite/docs/models/granite4-0 — overview and model-family table. | Correct distinction between named hybrid and traditional variants, with dense/MoE independent. No vendor speed multiplier repeated without conditions. |
| https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ — definitions, allocation cases, especially case4, prefix caching cases and implementation overview through end. | Per-layer allocation and state/prefix rules; page declares revision458e74 and unfinished features, so no universal current-support assertion. Direct download403; web text read successfully. |
| UCI81 current page license/details plus full byte-original pendigits.names. | Real spatially resampled pen inputs,44writers,30/14officialboundary,CCBY4. See data-provenance.md. |
| https://www.deeplearning.ai/courses/build-long-context-ai-apps-with-jamba/ — official indexed course listing with instructors,9video lessons,4codeexamples and subject description. Direct page requests returned403. | Annotated alternate video-course pointer. Videos were not watched; notebooks not executed; no claim of full course review. |

Broader search results were used only to locate these primary resources. Third-party promotional comparisons were not adopted. A Databricks talk search did not establish a verified direct creator-video URL, so it is not included as a supposedly watched resource.

## Actual calculations and limitations

All six declared PenDigits fits executed once. Data is unique and the official writer-disjoint assessment set is preserved. Model/program/protocol details and exact source hashes are in data-provenance.md. All 300 validation logits from every saved fit reload with maximum difference zero. All neural validation prefixes were compared between full, token and carried-chunk paths; the largest float32 output discrepancy is 1.2516975402832031e−5.

An independent two-sequence float64 MAM calculation compares full and incremental outputs, input gradients and every parameter gradient: output difference 1.4210854715202004e−14, input-gradient difference 3.2751579226442118e−15 and parameter-gradient difference 3.094746681142624e−15. Editing future coordinates leaves the checked earlier outputs unchanged; the shortened-prefix difference is 4.884981308350689e−15. These are author checks of the small instructional model, not native Jamba or browser integration tests.

hybrid_mechanisms.py derives the read/collision, cache, operation-count and router fixtures. author_calculations.py derives the actual edited-stroke and cache-fault fixtures, including unchanged final classes, a changed fresh class and degenerate input. No rendered heatmap, graph or guessed parameter state stands in for these quantities. Definitions and tolerances are in visual-specifications.md.

The selected study checkpoints are not a claim of state-of-the-art digit recognition. The tiny models have unequal counts, modest data and one or two seeds. Their eight-point input is not a context-scaling experiment. Assessment reversals and both MAM seeds remain printed. There was no quality-driven refit after these results.

deployment_example.py is complete but unexecuted; it requires a user-selected exact model revision and suitably provisioned compatible environment. Foundation-model weights, CUDA kernels, vLLM serving, adapter tuning, website rendering, accessibility/browser performance and formal implementation review remain deferred.

## Coordination and closure

Parent root already reconciled the full manuscript, numerical/specification content and provenance, finding only an article typo and joined words in the first specification draft. The typo is fixed and the specification was rewritten in normal prose. This is focused content reconciliation during phase one, not a claim of independent phase-two completion. MoE's own packet explicitly permits different normalization conventions; this lesson's local actual-Jamba convention fits that boundary.

No cross-topic notes or runtime changes are needed. Previous own packets remain frozen. Root's NeuralODE predecessor is linked correctly; Titans is the next unrequested destination. Finish only this packet, notify root, then freeze its files for shared checkpointing.

## Final author reread and learning-experience closure

The author reread the entire final manuscript in contiguous sections, the entire rewritten visual specification, this design/source record and the data provenance. This was a full reading of the explanation and learning flow, not a heading or word-count scan. The final pass corrected the conditional state update to “affine,” removed an answer from JD's initial prediction prompt and retained the complete denominator explanation in feedback. Root's article/spacing findings were also resolved. No fitted result changed. **Historical interaction record:** the earlier prediction/reveal behavior described here is superseded by the 21 September live-exploration contract; it is not a phase-two implementation requirement. Preserve the recorded mathematical checks and fixtures.

Completed checklist:

- The introduction starts with two concrete questions about the same records. Local operations are defined before the architectural vocabulary depends on them. Sequence position, network depth and the two meanings of hybrid remain distinguishable.
- The core route has observable outcomes, explicit prerequisite refreshers, step-by-step equations, changed examples and real applications. Optional scaling/deployment/family comparisons deepen rather than block that route. The actual previous and next module links are preserved.
- All eighteen inline visual specifications explain a specific mechanism. The four investigations use distinct representations, meaningful editable inputs and initially unset input-bound results. Contrasts include a real cache fault that changes the predicted class and faults that do not. Boundary nulls are explicitly defined against matching carry computations.
- Ten changed practice questions retain twenty initially closed hint/solution disclosures. The explanations connect the calculation to its interpretation rather than stopping at an answer.
- Full source conservation, actual primary-resource reading extents, exact dataset attribution, reported versus computed quantities and scope-limited alternative resources are recorded. The course listing is verified without claiming its videos were watched.
- Complete small-model programs, exact split IDs, all six outcomes and all measured failures remain reproducible. The larger deployment program is clearly unexecuted. Models are not ranked by fabricated performance figures.
- Accuracy, accessibility, narrow-screen behavior, keyboard edits, reset/invalidation, lazy delivery and computation limits have concrete phase-two contracts. No website/browser or native Jamba implementation check is claimed complete.

The final bounded command was `scratch/lesson-tools/Scripts/python.exe -X utf8 docs/teaching/drafts/hybrid-ssm-transformer-architectures-jamba/packet_checks.py`. It syntax-parsed all six Python programs without executing deployment; verified the unchanged original lesson hash, six actual curriculum destinations, three relative download links, eighteen figure/spec matches and twenty closed disclosures; executed recurrence/router boundary identities and changed practice counts; and compared worked/fresh boundary-zero and boundary-eight faults with matching carry branches. All four null differences are exactly zero. The saved report is packet-checks.json. Its first link check incorrectly assumed every planned topic had a source filename; this was corrected to check actual catalogue IDs, with no curriculum or lesson-link change. The original three data hashes were separately rechecked and match provenance. The scoped `git diff --check` passed.

Only exact disposable Jamba source extracts, the retrieval helper and this packet's Python cache are removed at closure. All nineteen substantive packet files remain for the phase-two handoff. No other author's files or shared runtime are removed. Root may now bind the content checkpoint; implementation remains not started.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Compare sequence memory and expert routes. Edit record keys/values, decay, score gap, cache budget, expert probabilities/capacity and supported stroke inputs. Show retained state versus explicit memory read, probability mass, exact request memory and continued frozen-model outputs. Choose a hybrid arrangement by memory retention and routing costs; named architecture examples do not imply identical mechanisms.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Implementation ownership and content-depth revision — 22 September 2026

Delivery remains **content-first**. The complete computational teaching route is part of this prepared packet now; phase two receives written code, explanations, mapped state/settings and closed practice, rather than an instruction to invent the missing mechanism. Earlier authoring records remain dated evidence; this section supersedes their incomplete depth handoffs. The title and stable ID are retained because the new material fulfills the existing scope.

| Advertised computational outcome | Scratch/source owner | Ordinary tool route | Matching bridge | Independent practice | Scope boundary |
| --- | --- | --- | --- | --- | --- |
| Selective recurrence, causal convolution and attention cache | stroke_models.py::SelectiveMixer/AttentionMixer | nn.Linear/Conv1d and explicit Torch tensor operations | full-versus-stream same weights, state and global offsets | Practice 1–3,5,7,9; request-swap exercise | Small complete hybrid; fused accelerator kernel engineering is beyond this lesson |
| Residual hybrid training and deployment | stroke_models.py::Layer/StrokeModel/train | deployment_example.py::main uses Transformers AutoModelForCausalLM | task/shape/state contract versus distinct full checkpoint config | Practice 6,10 | Provisioned CUDA deployment example unexecuted; not local-model parity |
| Experts and state/resource decisions | hybrid_mechanisms.py::route/cache_bytes; ../mixture-of-experts-transformers-moe/moe_study.py | Prepared MoE tensor dispatch owner; actual released model adapter | selected mass/renormalization and active-versus-stored state | Practice 4,8,9 | MoE owner prepared not yet implemented; local stroke FFN intentionally dense |

All local source owners above were inspected at their actual function/class definitions. Full model fitting, data/provenance and existing worked results are retained. Reused actual prerequisite code is named explicitly in the manuscript; prepared owners are not described as already published updated instruction. Whole-family releases mentioned for context do not expand the promised executable outcome into every checkpoint or every GPU kernel.

The teaching sequence is construct → explain the state/update → normal tool use → compare the same contract → changed-constraint practice, inserted where the relevant mechanism is explained. Original mechanism programs remain canonical; new programs depend on them only where the import is explicit. No browser program, published lesson, manifest or curriculum sequence is changed by this revision.

Author checks for this revision: source/API-contract reading, Python syntax parsing, matching embedded/downloadable source and local links, and scoped arithmetic probes where recorded in the specialist-writing report. These are content-authoring checks. Earlier fit outputs remain their original evidence; new multi-process/GPU/specialist-package execution, formal independent implementation review, rendered diagrams/labs and browser/accessibility/integration checks are **deferred**, with exact targets in the current visual specifications and specialist report.

Next action: after the root records the new content checkpoint, consume the full current packet for an authorized finish request, execute the relevant new programs and capture honest outputs, build the specified topic-owned views, independently check the translated models and integrate them. No core scratch/library manuscript writing is left as a finish-only TODO.
