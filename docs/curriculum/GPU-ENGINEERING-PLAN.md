# GPU engineering curriculum and authoring plan

Planning revision: 9 September 2026. Scope: an end-to-end GPU software engineering route, with explicit branches into distributed systems, scientific computing, graphics, portable execution and architecture research. This document and the associated data are **curriculum plans, not completed or verified lesson implementations**.

The authoritative individual briefs are in [gpu-expansion.js](../../src/learn/data/curriculum/gpu-expansion.js). That file adds 65 topics, gives individual briefs to the 32 pre-existing hardware topics, and exports exact-title prerequisites for the existing topics. The existing `hardware-systems` track remains the topic home. A GPU Engineering guided path should reuse those topics; the LLM Engineer path should reference the relevant shared portions. Do not create a second copy of CUDA, tensor parallelism or FlashAttention under LLM engineering.

Read [the teaching standard](../../LESSON-TEACHING-STANDARD.md) first. The Linux lesson accepted by the user establishes the expected mechanism visibility: separate visuals for separate learning hurdles, worked examples, predictions, hints, independent transfer and honest model boundaries. Neither this plan nor its singular `visual` planning field limits a lesson to one diagram or one lab.

## What completion means

A learner completing the core and appropriate specialist branches should be able to take a defined computation from a CPU reference to a correct, measured GPU implementation; choose libraries versus custom kernels; debug memory, ordering and numerical failures; integrate with frameworks; reason about multiple devices; deploy, operate and maintain the application; and investigate unfamiliar hardware or APIs from primary documentation.

No finite catalogue can promise that its reader knows every future product, proprietary implementation, application domain, scientific model or engineering contingency. The completeness target is observable capability coverage with visible boundaries. Teach how to verify unfamiliar assumptions, investigate documentation changes and extend the curriculum. Do not turn “end to end” into an unsupported claim of universal mastery.

The architecture/RTL branch is a bounded introduction to open GPU hardware research. It does not claim to replace a full digital design, VLSI, physical implementation, formal verification and semiconductor curriculum. If that specialization becomes an explicit goal, give its prerequisites and projects separate modules rather than burying them inside the introductory RTL lesson.

## What was missing in the original track

The original 32 topics already included architecture, CUDA, tiling, warp collectives, custom ML kernels, FlashAttention, compiler optimization, distributed training and accelerator case studies. Those topics are retained by exact title and URL.

The expansion fills several previously implicit transitions:

| Missing capability | Coverage now planned | Why it matters |
| --- | --- | --- |
| Write and build native code before CUDA | C/C++ ownership and arrays; CMake/linking; toolchain compatibility | Python familiarity alone does not teach buffer lifetime, pointers, compile errors or binary targets. |
| Design parallel work | Work/span, dependency graphs, CPU–GPU partitioning, break-even analysis | A kernel launch is not a parallel algorithm design. |
| Establish correctness | Indexing/strides/tails, memory ordering, barriers, streams/events, buffer lifetime, sanitizer/debugging | Many GPU failures are intermittent or reported after their cause. |
| Establish numerical trust | Floating-point representation, stable reductions, low precision, RNG, solvers/refinement | Matching one ordinary random input does not validate a numerical kernel. |
| Measure the application and kernel | Benchmarking, Nsight Systems, Nsight Compute, occupancy and cache studies | Local speedups can disappear behind transfers, launch costs or CPU work. |
| Build reusable primitives | Reductions, scans, histograms, GEMM, stencils, sparse operations, sort/top-k, graph work | These give learners transfer beyond a single matrix-multiply tutorial. |
| Use advanced execution deliberately | Matrix instructions, async copy/TMA, clusters, persistent kernels, CuTe, advanced Triton, autotuning | Low-level features need visible participation, ownership and version conditions. |
| Integrate with a real stack | PTX/native instructions, linking/runtime contexts, custom operators/autograd, allocation strategy, graphs | A standalone fast kernel is only one part of a useful application. |
| Scale and recover | Peer/NUMA topology, NCCL diagnosis, overlap, MPI/NVSHMEM, consistent checkpoints | Correct communication, progress and failure handling are essential engineering work. |
| Operate sustainably | Containers, schedulers/sharing, health diagnostics, CI, energy/cost, security and storage | Production work includes reliability and deployment constraints. |
| Work outside one vendor or ML | HIP/ROCm, AMD profiling, SYCL/OpenMP/OpenACC, WebGPU, graphics, scientific simulation | GPU engineering is broader than NVIDIA training and inference. |
| Demonstrate synthesis | Kernel library, multi-GPU application and inference capstones | Completion needs observable independent work. |

## Routes and prerequisite behavior

The catalogue groups related material for navigation. **Section order is not the dependency graph.** Several foundational new topics must precede pre-existing lessons. Resolve the exact prerequisite graph when constructing a guided route. The exported `gpuPrerequisites` map exists to put missing foundations before those earlier entry topics. A learner may test out of a prerequisite by demonstrating its outcome; a beginner should receive a refresher or a link, not an unexplained leap.

Use a short orientation with four realistic destinations: GPU application developer, kernel/performance engineer, distributed GPU engineer, and accelerator/graphics/scientific specialist. They share foundations and branch intentionally. Do not present all 97 topics as equally mandatory for every learner.

### Shared foundations

1. Reuse the existing Python, Git, Linux, testing, NumPy, basic linear algebra and complexity lessons as needed.
2. Learn **C & C++ Foundations for GPU Programming** and **CMake, Linking & Reproducible Native Builds**. The C/C++ bridge begins with a familiar array transform, then makes pointers, ownership and lifetime visible. Deeper template machinery belongs where CuTe or native-library design requires it.
3. Learn **Parallel Algorithm Design: Work, Span & Decomposition**, CPU architecture, GPU architecture and memory hierarchy.
4. Complete **GPU Toolchains, Drivers & Compatibility**, then the existing **CUDA Programming Model** with a working, verified small kernel.
5. Learn indexing, floating-point representation, atomics/visibility, synchronization, streams/events and transfer lifetimes. The first runnable kernel does not require all these topics at once; subsequent lessons refine the initial bounded model.
6. Establish reliable benchmarking and the first application timeline. Introduce the roofline as a scoped prediction model, then learn detailed counters and resource limits when an experiment needs them.

Readiness gate: the learner can build a CPU reference, launch a correctly indexed operation, explain when the output becomes available, test tails and exceptional values, identify a failure’s cause versus reporting point, and report a defensible timing.

### Application developer route

After the shared foundations, use CUDA Libraries; unified memory and explicit transfer choices; input pipelines; containers/compatibility; correctness CI; and a small domain application. Integrate through existing PyTorch, JAX or other framework material when relevant. A library-first route is a valid destination: no requirement to hand-code GEMM before building useful GPU applications.

Application mastery check: choose an appropriate library, validate its shape/layout and numerical contract, keep lifetimes correct, locate the end-to-end bottleneck and package a reproducible result.

### Kernel and performance engineer route

Proceed through tiling and warp participation; reductions and scan; histograms, GEMM and stencils; sparse, sorting and irregular algorithms as domain branches. Pair each primitive with its local prediction, correctness and benchmark task.

Then study occupancy/locality, matrix instruction layouts, low-precision error budgets, async pipelines, persistent kernels and optional clusters. Compare CUTLASS/CuTe and advanced Triton after the underlying layouts and data movement are understood. Add PTX/native inspection, runtime/device linking, custom framework operators, allocators, graphs and autotuning. Complete **GPU Engineering Capstone: Verified Kernel Library**.

Kernel mastery check: defend both correctness and optimization using a CPU/library reference, adversarial inputs, a relevant tool run, a numerical rationale and comparable measurement artifacts. Explain why a candidate optimization failed as confidently as why another succeeded.

### Distributed GPU engineer route

After the shared core, study interconnects and node topology before collectives. Then learn collective semantics, NCCL diagnosis, DDP equivalence, communication overlap, tensor/pipeline/FSDP/ZeRO partitioning and hybrid rank groups. Add expert routing when the learner understands MoE mathematics and token bookkeeping; add CUDA-aware MPI/NVSHMEM for scientific/remote-memory applications.

Follow with checkpoint consistency and recovery, scheduler placement/sharing, health diagnostics, storage, deployment, security and energy/cost. Complete **GPU Engineering Capstone: Multi-GPU Application**. A two-device bounded project is sufficient to demonstrate mechanisms; it must not be advertised as validation of a large cluster.

Distributed mastery check: trace one piece of data through ranks and collectives; predict each buffer’s owner and lifetime; show where the critical path lies; reproduce a consistent restart; explain scaling and resource limits using measured evidence.

### LLM engineer connection

The LLM path reuses GPU architecture, memory, precision, profiling and selected distributed lessons. Its specialized destination joins existing attention, KV-cache, continuous batching, inference architecture and serving lessons with the new allocator, graph and custom-operator topics.

Use **GPU Engineering Capstone: Profiled Inference Engine** for synthesis. Keep model mathematics and serving policy in their existing homes. GPU lessons explain execution, memory and measurement; cross-link without restating all of transformer training or request routing.

For a real implementation, add or deepen relevant submodules inside existing lessons as evidence demands: ragged and grouped GEMM; MoE token dispatch; decode versus prefill kernels; normalization; gather/scatter; low-bit packing and scaling; paged layouts; attention masks; sampling; communication overlap. These are mechanisms belonging to existing topic scopes, not a reason to invent dozens of near-duplicate pages now.

### Scientific, graphics and portability branches

- **Scientific computing:** numerical methods and linear algebra → stencils/sparse kernels → FFTs or solvers as needed → RNG/Monte Carlo where relevant → domain decomposition → scientific simulation. Validate mathematical convergence and physical invariants as well as code agreement.
- **AMD/portable engineering:** HIP and actual ROCm compatibility → AMD profiling → named architecture tuning. Add SYCL or directive offload as an alternative model. Distinguish source portability, correctness portability and performance portability.
- **Browser/graphics:** WebGPU/WGSL → graphics pipeline and compute/resource interoperation → a small visible compute-to-render project. Browser limits and device-loss handling belong in the practical route.
- **Architecture research:** existing hardware foundation → PTX/counters → simulator validation → a bounded open RTL experiment. Add formal digital-hardware prerequisite modules if the user later requests full GPU chip-design specialization.

## How to write these lessons at the accepted quality level

Every individual topic has a `blueprint` with an intended summary, two observable outcomes, exact-title prerequisites, five topic-specific teaching stages, a visual question and interaction, a practical task and success criterion, misconceptions, primary source URLs, depth and review focus. These briefs are minimum planning anchors, not finished lesson prose or a mechanical heading template.

Before implementation, expand the brief into a concept-hurdle table. For every hard transition identify: what the learner already knows, the small example, the unknown relationship, the visual representation, a prediction, the actual mechanism, the check and an independent variation. Keep the continuing objects, coordinates and values consistent across explanations and code. Introduce each symbol, unit and shape at first use.

The primary visual listed in a brief is the first identified hurdle. Add independent diagrams and labs for other mechanisms. For example:

| Topic | Distinct representation jobs |
| --- | --- |
| Streams/events | Host submission versus completion timeline; a dependency graph; a buffer-reuse/lifetime view if transfers are involved. |
| Tiling | Lane-to-address transaction map; shared-bank map; input/output tile correspondence. |
| Synchronization | Participant/arrival gates; visibility/dependency arrows; deadlock schedule showing why progress stops. |
| GEMM | One output dot product; tile reuse map; register/shared/global movement; measured roofline comparison. |
| Atomics | Lost-update interleaving; indivisible read-modify-write; release/acquire publication with explicit scope. |
| FSDP/ZeRO | Rank ownership; tensor lifetime timeline; peak-memory accounting. |
| GPU profiling | Full application timeline first; source-to-counter evidence second; a controlled before/after experiment. |
| Low precision | Representable-value grid; multiply/accumulate trace; application error distribution. |

A changing number or table is insufficient when the concept is ownership, participation, routing or causality. Draw and label the connections. Tables remain useful as exact-value and accessible companions. Use color consistently and never as the only indication of state. Provide useful default data, reset, keyboard operation, explanation and reduced-motion behavior. A complex lesson may have several small labs; a short concept may need only a static diagram.

The preferred GPU teaching loop is:

1. Solve an inspectable version on the CPU or by hand.
2. Draw the data, dependencies, ownership and lifetime before showing the implementation.
3. Predict a result or failure.
4. Run or step the mechanism and explain every relevant changed state.
5. Validate against an independent reference and a stated numerical/semantic contract.
6. Measure the defined workload and test one performance hypothesis.
7. Apply the idea to a changed shape, layout, distribution, device or failure case.

Do not force numerical analysis or architecture history into the same order as a command-oriented operational lesson. Derivations may precede a lab; a failure story may introduce a debugging lesson; a reference table may be the right tool for a library contract. Preserve the question-to-mechanism-to-evidence philosophy while choosing a subject-appropriate teaching form.

## Examples, laboratories and assessment

Every implemented runnable example must supply all required files, build/run commands, imports, fixtures, environment assumptions, expected output and explanation. Start with tiny deterministic values where possible, then cover realistic scale. Avoid unexplained downloads or unprovided training scripts.

The core CUDA sequence uses a small invented sensor-array workload: transform values, combine statistics, filter or compact readings and visualize a spatial field. Matrix and attention lessons may switch to clearly introduced data with declared shapes. Never keep an analogy after it obscures the actual mathematics or data structure.

Practice should reduce support: a prediction; a fully worked case; a partially completed variation; an independent changed-input task; and an error diagnosis or synthesis where useful. These are types of support, not quotas. Substantial exercises need an optional hint and a full reasoned solution. Open-ended optimization tasks need criteria and a valid example analysis, not a single magic tile size.

Use adversarial cases appropriate to the operator: zero/one sizes and tails; strides and noncontiguity; integer index overflow; empty segments; repeated keys; heavily skewed destinations; cancellation; extreme magnitudes; NaN/infinity policy; wrong masks; incomplete barrier participation; asynchronous reuse; mismatched collective order; and interrupted checkpoints. Do not include irrelevant cases solely to lengthen a test list.

The final readiness check asks the learner to explain or produce evidence. A navigation completion checkbox is not evidence of competence.

## Validation and hardware honesty

Separate these evidence levels in lesson reports:

- **Mathematical/reference check:** independent hand calculation or trusted CPU/library result with justified error criteria.
- **Teaching-model check:** tests that verify the browser diagram’s own state and simplified rules.
- **Real backend execution:** compiled GPU code on a named device, toolkit, driver and compiler; relevant sanitizer/debugger output.
- **Performance evidence:** defined workload, comparable baseline, sample distribution, timing boundary, warmup/compilation treatment and environment.
- **Browser/accessibility check:** desktop/mobile layout, keyboard controls, labels, focus, overflow, sensible initial state and error recovery.

A browser scheduler, memory-bank illustration or simulated cluster is not a CUDA/ROCm runtime or a cycle-accurate hardware model. Say so where it affects interpretation. It may still teach the concept without GPU access. Provide a no-GPU route using honest traces, CPU references and conceptual investigations. Never report a GPU benchmark, sanitizer pass, device compatibility or physical conservation test that was not actually performed.

Performance is sensitive to shape, dtype, layout, clock state, contention, cache state, tool overhead, input distribution and version. A speedup claim should be a bounded result, not a universal recommendation. Review correctness again after every material optimization.

## Research record and future verification

Primary references were consulted during the 9 September 2026 planning session. The 55-entry `gpuSources` registry contains actual reference URLs. Most are live documentation, not immutable release snapshots. A page’s retrieval time, footer, search-engine publication label or `/latest` URL does not alone prove that every described capability shipped by a particular historical date. Pin the chosen documentation/repository revision when writing a lesson, and record the actual tested environment.

The source families inform coverage, while the explanations, examples, visual questions and project designs are original:

- The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) anchors execution, memory, synchronization and feature scopes. The reviewed guide has a broad core plus feature-specific and technical-reference routes; this curriculum uses its own learner examples and prerequisite sequence.
- The [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) supports measuring the application, selecting worthwhile acceleration and checking correctness during optimization.
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/), [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) and [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) anchor the distinct timeline, counter and correctness workflows.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/), [CUTLASS](https://docs.nvidia.com/cutlass/latest/overview.html), [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) and [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) provide the primary implementation references for advanced kernels. Pin exact versions and supported architectures; do not copy marketing performance claims into the course.
- [Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/), [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/), [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/), [cuFFT](https://docs.nvidia.com/cuda/cufft/) and [cuRAND](https://docs.nvidia.com/cuda/curand/) anchor numerical and domain-library contracts.
- [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/), official PyTorch distributed references and [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) anchor distributed communication and training state. The original [FlashAttention paper](https://arxiv.org/abs/2205.14135) anchors the algorithmic derivation; implementation generations need separate review.
- [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html), [AMD GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html) and [ROCprofiler-SDK](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html) prevent an NVIDIA-only mental model. Do not transplant counter names or warp assumptions across vendors.
- [SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html), [OpenMP](https://www.openmp.org/specifications/), [OpenACC](https://www.openacc.org/specification), [WebGPU](https://gpuweb.github.io/gpuweb/), [WGSL](https://gpuweb.github.io/gpuweb/wgsl/), [Vulkan](https://docs.vulkan.org/spec/latest/index.html) and [Metal](https://developer.apple.com/metal/) provide standards/vendor references for portable and graphics branches. The WebGPU material reviewed included an editor’s draft; feature support still needs browser and adapter checks.
- [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/), [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/), [Slurm GRES](https://slurm.schedmd.com/gres.html), [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/), [MPS](https://docs.nvidia.com/deploy/mps/latest/index.html), [DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html), [Xid guidance](https://docs.nvidia.com/deploy/xid-errors/latest/index.html) and [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/) support operations coverage.
- [Accel-Sim](https://accel-sim.github.io/) and [Vortex](https://github.com/vortexgpgpu/vortex) are primary project references for bounded architecture/simulation and open RTL exercises. Research models are not vendor-silicon validation.

Research limitations are explicit. The NVSHMEM endpoint returned no extractable text in this pass; the topic is planned, but its current API contracts require a successful primary-manual review before runnable authoring. Several accelerator landing pages and PyTorch URLs provided redirects or limited text; detailed SDK/API pages must be read at implementation time. Legacy Graphcore, MI300/MI350 and other product names remain existing titles for continuity; their current lifecycle and support must be researched before making recommendations. No current price, market availability or universal newest-release claim is made here.

## Maintenance and handoff

For a future authorized topic, read the teaching standard, its `blueprint`, actual existing lesson/components, and the relevant reference pages. Reuse useful work. Make the concept-hurdle and validation plan before implementing. Expand the brief when a topic-specific difficulty requires more stages or a different representation, and record why. Do not remove scope-critical content to fit a word count or an arbitrary lab count.

When a source changes, determine whether it alters a factual claim, supported setup, API, visual model, exercise result or prerequisite. Update affected parts together. Date a research claim, date a tested environment and date learner/user approval separately.

This planning increment validates module import, all 65 new titles for uniqueness, all 97 blueprint shapes, exact prerequisite resolution against the catalogue, coverage of the 32 old hardware topics and cycles within the GPU prerequisite graph. The application integrator must additionally validate the final combined catalogue and guided paths. No GPU lesson body, CUDA/HIP kernel, benchmark or visual lab was implemented or executed in this planning increment.
