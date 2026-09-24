# Prepared Deep Learning content: live exploration revision

Updated 21 September 2026. Scope: the 42 Deep Learning packets that were content-complete and implementation-not-started in the [captured baseline](evidence/live-exploration-baseline.json). All 42 remain prepared-only; this review does not claim shipped labs or new browser verification.

The user explicitly removed learner-prediction features, including optional ones. The four current teaching/engineering policies now require useful default outputs, meaningful input controls, synchronized mechanism and output views, bounded computation, reset, null/invalid cases and a practical decision connection. This revision replaces the conflicting lab instructions inside each packet, rather than relying on a policy paragraph to override them. Model predictions, attention/memory gates, causal masks and separate written practice are retained.

The revision changes 126 authoring files: lesson.md, visual-specifications.md and design.md for each packet. Each packet has an individual live route below and matching detailed contracts at its visual placements. Historical author-review notes that describe older prediction interactions are explicitly marked superseded; historical evidence JSON and native calculation files were not rewritten.

## Scope and evidence

- [Machine-readable disposition and hashes](evidence/live-exploration-prepared-content.json) records each changed file, prior checkpoint hash, retained supporting artifact and authoring checks.
- [Original authoring text archive](evidence/live-exploration-prepared-originals.json) preserves the exact original text of the 126 files. It is historical input, not a teaching instruction.
- Exact fenced programs and original external reference URLs are retained in all 126 documents. Details/code fences are balanced. The numeric values in every original visual specification are retained; specific fixture details embedded in old prediction instructions were carried into explicit reference-fixture paragraphs.
- All 346 ledger-listed supporting files outside these 126 documents remain byte-identical to the captured content checkpoint. No data, native calculations, measurement records or model parameters were changed.
- No per-ID authored blueprint exists for these 42 topics under src/learn/data/curriculum/blueprints. None was created. The inspected domain guidance uses model prediction as subject matter and separate written practice; it does not require a lab prediction gate.
- These are conservation and authoring-contract checks, not independent re-verification of all scientific claims. No training campaigns, new research claims, runtime publication, React implementation, browser accessibility or rendering checks occurred for these prepared packets. Existing numeric evidence applies only to its original unchanged quantities.

## Phase-two obligations

Read the complete manuscript, specification, design and retained data/code. Implement the topic-owned forms below; do not replace them with a repeated generic output box. Show the initial valid result immediately and recompute the linked views on valid edits, with accessible numeric/keyboard alternatives and bounded work. Retain exact reference parity checks, units, causal/data boundaries and measured-versus-simulated labels. Verify successive edits, null/invalid/extreme cases, reset and rapid edits; ensure stale outputs never appear current. Capture informative desktop/phone states and complete the ordinary independent content, code and learning-experience reviews before marking implementation complete. The shared ledger must refresh content hashes while retaining implementation-not-started.

## Per-packet disposition

### 1. Move a point and its weighted evidence

Packet: [perceptrons-neurons-activation-functions](drafts/perceptrons-neurons-activation-functions/design.md). **Updated; implementation not started.**

**Controls:** Edit input coordinates, weights, bias and common scale; move the activation operating point and incoming weight; edit XOR hidden bias and output coefficient.
**Visible consequence:** Synchronize contribution bars, boundary distance, hard/smooth outputs, activation value/slope and all four XOR rows. Compare a shared coefficient rescaling with a moved input, and a repaired corner with the remaining corners.
**Decision:** Choose whether the decision boundary, smooth confidence or local sensitivity needs to change; a one-row repair need not solve the whole task.

### 2. Trace credit and the effect of a step

Packet: [backpropagation-automatic-differentiation](drafts/backpropagation-automatic-differentiation/design.md). **Updated; implementation not started.**

**Controls:** Edit the two-example fit, learning rate, repeated-path coefficient and finite-difference step/offset.
**Visible consequence:** Show fitted line, residuals, derivative contributions and before/after loss immediately; step the backward accumulation without hiding the current total. The finite-difference panel displays both errors and the analytic derivative.
**Decision:** Distinguish a correct gradient from a useful step size, and truncation/cancellation from a faulty derivative.

### 3. Watch which errors receive influence

Packet: [loss-functions-ce-mse-focal-contrastive-triplet](drafts/loss-functions-ce-mse-focal-contrastive-triplet/design.md). **Updated; implementation not started.**

**Controls:** Move observations, change the loss and focal gamma, drag decision thresholds, edit pair/triplet coordinates and change InfoNCE temperature.
**Visible consequence:** Update loss, signed gradients, fitted location, confusion counts, eligible negatives and candidate probabilities together. Keep score-based metrics distinct from threshold decisions.
**Decision:** Choose an objective or operating threshold from the error tradeoff rather than from a single loss number.

### 4. Change the group that defines the ruler

Packet: [batch-layer-group-rms-normalization](drafts/batch-layer-group-rms-normalization/design.md). **Updated; implementation not started.**

**Controls:** Edit tensor cells, normalization method/group count, offset/scale, epsilon, affine values, mode and running-statistic parameters.
**Visible consequence:** Highlight each statistic membership set and show all affected outputs, means/variances and running buffers immediately. Compare changing another example with changing a member of the same group.
**Decision:** Choose a normalizer and mode by its information dependencies, batch sensitivity and inference state.

### 5. Inspect what freezing actually freezes

Packet: [transfer-learning-fine-tuning-strategies](drafts/transfer-learning-fine-tuning-strategies/design.md). **Updated; implementation not started.**

**Controls:** Toggle parameter updates, gradient recording and module mode; edit LoRA factors/rate; change parameter budgets over saved validation candidates.
**Visible consequence:** Show parameters, buffers, gradients and before/after function values separately. Display eligible candidates and the validation-selected winner live; the one retained test result remains clearly identified as previously observed.
**Decision:** Select an adaptation strategy under a real budget and avoid confusing frozen weights with frozen behavior or repeated inspection with a fresh test.

### 6. Explore scale and directional transmission

Packet: [weight-initialization-xavier-kaiming-p](drafts/weight-initialization-xavier-kaiming-p/design.md). **Updated; implementation not started.**

**Controls:** Inspect saved initialization/seed traces; edit four activation values, singular directions, depth and width/rate scaling.
**Visible consequence:** Show forward/backward second moments, means/variance, directional gain and shape/update formulas together. Continuous tiny models are distinct from selectors over measured training records.
**Decision:** Choose an initialization/parameterization by the signal and update behavior it preserves, without treating average scale as every-direction stability.

### 7. Follow the direct and residual paths

Packet: [residual-connections-skip-connections](drafts/residual-connections-skip-connections/design.md). **Updated; implementation not started.**

**Controls:** Edit branch weights, scalar depth/gain, normalization placement, projection entries and Euler step; inspect recorded block omissions.
**Visible consequence:** Show correction contributions, current output/loss, both derivative paths and shape compatibility as inputs change. Display every intermediate gain and genuine before/after ablation record.
**Decision:** Recognize cancellation, shape mismatch and step-size instability despite the presence of an identity path.

### 8. Play with mask geometry and mode

Packet: [dropout-droppath-stochastic-depth](drafts/dropout-droppath-stochastic-depth/design.md). **Updated; implementation not started.**

**Controls:** Edit features/weights, probability, survivor scale, mask grouping, branch position, per-block rates and train/eval mode; inspect retained Monte Carlo prefixes.
**Visible consequence:** Update weighted outcome means/variances, gradient routes, call counts, state buffers and saved prediction distributions immediately. Keep the sampled mask fixed while comparing a parameter, with resampling a separate action.
**Decision:** Choose masking scope and evaluation behavior from their actual effects; distinguish expected active depth from work that was really skipped.

### 9. Move a window and see every dependency

Packet: [convolution-pooling-receptive-fields](drafts/convolution-pooling-receptive-fields/design.md). **Updated; implementation not started.**

**Controls:** Edit image/kernel cells, stride/dilation/padding, pooling inputs, shared-weight targets/rate and receptive-field threshold.
**Visible consequence:** Synchronize the selected patch, products, output map, transpose contributions, gradient accumulation and ancestry paths. Geometry edits visibly change output size, alignment and holes rather than only a summary label.
**Decision:** Choose window geometry and pooling from reach, alignment, information loss and update behavior.

### 10. Compare architecture operations and budgets

Packet: [landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet](drafts/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet/design.md). **Updated; implementation not started.**

**Controls:** Edit head dimensions, channel-context cells, scaling allocations, deployment budgets and signed score-map weights.
**Visible consequence:** Show exact parameter/MAC counts, gate contributions, candidate eligibility and current CAM/logit arithmetic live. Recorded model/seed selectors display existing evidence immediately.
**Decision:** Identify which operation consumes the budget, what information a head discards and why a smaller model is not automatically better.

### 11. Construct channel routes and sampling coverage

Packet: [depthwise-separable-dilated-convolutions](drafts/depthwise-separable-dilated-convolutions/design.md). **Updated; implementation not started.**

**Controls:** Edit depthwise/pointwise filter entries, channel cells, stencil dilation/offsets, serial rates and parallel branch choices; manipulate retained digit inputs where weights are available.
**Visible consequence:** Update output contributions, rank restrictions, visited lattice sites, branch union and exact frozen-model outputs. Keep coverage geometry separate from learned influence.
**Decision:** Choose separability, dilation or parallel context from expressiveness, blind spots and the measured budget.

### 12. Inspect modern convolution blocks

Packet: [convnext-modern-cnn-designs](drafts/convnext-modern-cnn-designs/design.md). **Updated; implementation not started.**

**Controls:** Change stage dimensions, normalization groups, GRN feature cells, valid visible-patch selections and branch-folding coefficients.
**Visible consequence:** Show parameter counts, shared GRN denominator, changed feature maps, reconstruction consequences and folded-kernel equality immediately. Preserve image masking as the learning objective, not UI answer hiding.
**Decision:** Separate architecture from recipe, global channel context from local normalization, and valid reparameterization from a changed function.

### 13. Manipulate votes and vector geometry

Packet: [capsule-networks](drafts/capsule-networks/design.md). **Updated; implementation not started.**

**Controls:** Edit capsule votes, routing iterations, vector magnitude/direction and supported retained image/latent coordinates.
**Visible consequence:** Show coupling rows, vote contributions, squash length/direction, current parent vectors and saved/frozen-model outputs. Step routing to inspect its computation, with all current outputs visible.
**Decision:** Distinguish agreement from activation magnitude, pose changes from class evidence and a model intervention from a new empirical result.

### 14. Carry and edit recurrent state

Packet: [rnns-lstms-grus](drafts/rnns-lstms-grus/design.md). **Updated; implementation not started.**

**Controls:** Edit sequence entries, recurrent weights, LSTM gates, GRU reset placement and supported pen-trajectory coordinates.
**Visible consequence:** Update state trajectories, retained/injected terms, shared-weight credit and exact learned outputs. Step, rewind and reset state explicitly; padding and request boundaries remain visible.
**Decision:** Choose what must persist or reset, and diagnose saturation, reset-order differences and accidental cross-sequence leakage.

### 15. Explore source, prefix and decoding paths

Packet: [sequence-to-sequence-encoder-decoder](drafts/sequence-to-sequence-encoder-decoder/design.md). **Updated; implementation not started.**

**Controls:** Edit source/target shifts, bridge weights/rate, tiny probability trees, beam width and supported fitted source/prefix inputs.
**Visible consequence:** Show aligned timelines, dependency paths, sequence probabilities and bounded beam candidates live. Keep teacher-forced versus generated inputs explicit at every step.
**Decision:** Distinguish model probability from a decoding decision and identify when a prefix or alignment changes the actual task.

### 16. Move a memory contribution

Packet: [attention-mechanism-bahdanau-luong](drafts/attention-mechanism-bahdanau-luong/design.md). **Updated; implementation not started.**

**Controls:** Edit queries, keys/values, scorer parameters, padding validity, local-window placement and supported real source/prefix inputs.
**Visible consequence:** Synchronize scores, normalized weights, weighted values, context and decoder output; keep masked rows and missing legal donors explicit. Retain saved observations as records.
**Decision:** Choose or diagnose a scorer/read window from the dependencies it creates, without reading attention weight as a complete causal explanation.

### 17. Compare memory retained by different mechanisms

Packet: [long-context-sequence-models-transformer-xl-griffin-perceiver](drafts/long-context-sequence-models-transformer-xl-griffin-perceiver/design.md). **Updated; implementation not started.**

**Controls:** Edit sequence records, segment and memory lengths, recurrence/input gates, latent queries and supported trajectory coordinates.
**Visible consequence:** Show legal donors, cache contents, retained/injected state, latent mixtures and exact frozen-model readouts as controls change. Compare reordering whole records with changing their positions.
**Decision:** Choose a memory/compression scheme by what it preserves and loses, and separate context reach from usable information.

### 18. Change a state-space write and read

Packet: [state-space-models-s4-mamba-mamba-2](drafts/state-space-models-s4-mamba-mamba-2/design.md). **Updated; implementation not started.**

**Controls:** Edit tiny system coefficients, impulse inputs, selective writes, distraction sequence, chunk boundaries and supported real trajectories.
**Visible consequence:** Show impulse response, carried state, input-conditioned updates, SSD matrix entries and chunk equivalence live. Stepping exposes current recurrence arithmetic.
**Decision:** Decide which information needs selection or state carry and distinguish a mathematically equal scan from a different update rule.

### 19. Read and write a compact memory

Packet: [rwkv-linear-attention-models](drafts/rwkv-linear-attention-models/design.md). **Updated; implementation not started.**

**Controls:** Edit keys/queries/values, decay, current-token bonus, memory write/correction inputs and real stream interruptions.
**Visible consequence:** Update summary matrices/denominators, present output versus stored history and chronological state together. Compare a continued stream with an explicit reset.
**Decision:** Choose the correct current-read and future-memory semantics and see what compact state cannot retain.

### 20. Build an attention read

Packet: [self-attention-multi-head-attention](drafts/self-attention-multi-head-attention/design.md). **Updated; implementation not started.**

**Controls:** Drag/edit vectors, manipulate legal communication edges, change head projections/temperature and supported trajectory points.
**Visible consequence:** Show scores, weights, mixture point, per-head outputs, mask legality and sensitivity immediately. A direct vector edit reaches every linked output and accessible table.
**Decision:** Distinguish compatibility, value content and legal access; choose head/mask/scale settings by those separate effects.

### 21. Follow a complete block

Packet: [transformer-block-architecture](drafts/transformer-block-architecture/design.md). **Updated; implementation not started.**

**Controls:** Edit features, normalizer offset/scale, FFN matrices, branch multiplier, pre/post placement and supported retained trajectory inputs.
**Visible consequence:** Update normalization reference sets, feature writes, residual state, block output and derivative paths together. Show exact zero-branch and common-shift cases.
**Decision:** Reason about placement and branch scaling using their actual function and gradient effects rather than an architecture slogan.

### 22. Move a position without confusing it with content

Packet: [positional-encodings-sinusoidal-learned-rope-alibi](drafts/positional-encodings-sinusoidal-learned-rope-alibi/design.md). **Updated; implementation not started.**

**Controls:** Edit position IDs, frequencies, RoPE vectors, ALiBi slopes/scores, cache identities and supported trajectory coordinates.
**Visible consequence:** Synchronize phase geometry, relative-score changes, distance penalty, legal cache relations and final mixture. Show whole-record reorder and ID-only edit as different operations.
**Decision:** Choose and troubleshoot positional mechanisms by relative/absolute behavior and cache consistency; do not infer long-context quality from a toy phase plot.

### 23. Share memory across query heads

Packet: [grouped-query-attention-gqa-multi-query-attention-mqa](drafts/grouped-query-attention-gqa-multi-query-attention-mqa/design.md). **Updated; implementation not started.**

**Controls:** Edit Q/K/V, query-to-KV grouping, cache dimensions, offset masks and supported causal input prefixes.
**Visible consequence:** Show each reader, shared K/V record, weighted sum, exact byte/MAC budgets and compact cache outputs immediately. Compare equal-head versus unequal-head regrouping.
**Decision:** Choose grouping by memory and functional tradeoffs, keeping payload arithmetic separate from measured latency and model quality.

### 24. Inspect a latent cache and its alternatives

Packet: [multi-head-latent-attention-mla](drafts/multi-head-latent-attention-mla/design.md). **Updated; implementation not started.**

**Controls:** Edit latent vectors/projections, rotation, retained rank, payload dimensions and supported frozen-model prefixes.
**Visible consequence:** Show expanded and absorbed paths, commutation residuals, singular-direction effects, bytes/arithmetic and resulting outputs together.
**Decision:** Choose compression by the function and input directions it preserves; distinguish a low parameter error from low task error.

### 25. Expose the approximation and legal path

Packet: [sparse-linear-attention-variants](drafts/sparse-linear-attention-variants/design.md). **Updated; implementation not started.**

**Controls:** Edit sparse edges, feature-memory writes/evictions, random-feature settings, compression coefficients, block layout and supported real trajectories.
**Visible consequence:** Show removed mass, reachability, normalized summaries, approximation error, future influence and tile occupancy live under a fixed random draw.
**Decision:** Choose sparsity or approximation by accessible information, numerical error and actual block work, not a single sparsity percentage.

### 26. Change pixels, positions and feature relationships

Packet: [vision-transformers-vit-deit-swin-dinov2](drafts/vision-transformers-vit-deit-swin-dinov2/design.md). **Updated; implementation not started.**

**Controls:** Edit patch pixels/projection coefficients, patch-position swaps, window/shift geometry, teacher/student logits and feature angles.
**Visible consequence:** Update patch contributions, actual frozen-model logits/maps, spatial dependency sets, teacher targets/gradients and relational loss immediately.
**Decision:** Separate image edits from position changes, direct neighbors from multi-hop reach and target sharpening from learned quality.

### 27. Route tokens through experts and capacity

Packet: [mixture-of-experts-transformers-moe](drafts/mixture-of-experts-transformers-moe/design.md). **Updated; implementation not started.**

**Controls:** Edit token/router scores, expert values, capacity, grouping and budget dimensions; manipulate supported retained images.
**Visible consequence:** Show selected experts, discarded probability mass, overflow/drop routes, recombined outputs, balance terms and active/total resource counts together.
**Decision:** Choose routing/capacity policies from missing contributions and resource tradeoffs; auxiliary balance does not establish task quality.

### 28. Arrange readers, memory and availability

Packet: [interleaved-cross-attention-architectures](drafts/interleaved-cross-attention-architectures/design.md). **Updated; implementation not started.**

**Controls:** Edit rectangular Q/K/V cells, shape assignments, input order/availability, compressor entries and gate parameters.
**Visible consequence:** Show current read contributions, dependency legality, compression collisions and gradient routes. Saved image/question selectors reveal the actual retained outputs immediately.
**Decision:** Choose a cross-attention arrangement from what can be read, when it is available and what compression removes.

### 29. Edit a graph and follow the message

Packet: [message-passing-graph-convolutions-gcn-gat-graphsage](drafts/message-passing-graph-convolutions-gcn-gat-graphsage/design.md). **Updated; implementation not started.**

**Controls:** Change directed edges/node features, aggregation/normalization, attention scores, masks and supported fitted-graph inputs.
**Visible consequence:** Update adjacency, degree factors, synchronized round states, reachability, information boundaries and fitted outputs together.
**Decision:** Choose aggregation and sampling from information flow, normalization and expressiveness while avoiding evaluation leakage.

### 30. Transform geometry and test the same rule

Packet: [graph-transformers-geometric-deep-learning](drafts/graph-transformers-geometric-deep-learning/design.md). **Updated; implementation not started.**

**Controls:** Edit graph structure/coordinates, structural encodings, transformation action, message parameters and intentionally broken symmetry modes.
**Visible consequence:** Show invariant scalars, equivariant vectors, neighborhood membership and transformed outputs side by side. Distinguish a relabeled graph from a physically moved geometry.
**Decision:** Choose operations compatible with the required symmetry and find informative asymmetric counterexamples.

### 31. Move energy and follow probability mass

Packet: [boltzmann-machines-restricted-boltzmann-machines-rbm](drafts/boltzmann-machines-restricted-boltzmann-machines-rbm/design.md). **Updated; implementation not started.**

**Controls:** Edit small-model biases/interactions, data counts, transition/sampling settings and supported retained digit states.
**Visible consequence:** Show normalized joint/marginal probabilities, data-model statistics, exact transition mass and sampled chain trajectories simultaneously.
**Decision:** Distinguish energy from normalized likelihood, reconstruction from probability and finite mixing behavior from an equilibrium claim.

### 32. Control stretch and inspect where a penalty acts

Packet: [spectral-normalization-gradient-penalty](drafts/spectral-normalization-gradient-penalty/design.md). **Updated; implementation not started.**

**Controls:** Edit matrix entries, normalization method, power-iteration steps, critic/input values, interpolation points and margin geometry.
**Visible consequence:** Update singular stretch, effective matrix, derivative paths, sampled gradient penalties and local decision distances live.
**Decision:** Choose or diagnose a constraint by what it actually bounds and where it was evaluated; a sampled penalty is not a global guarantee.

### 33. Shape an associative memory

Packet: [modern-hopfield-networks](drafts/modern-hopfield-networks/design.md). **Updated; implementation not started.**

**Controls:** Flip cue bits and visit order; edit continuous memory vectors, temperature, keys/queries/values and real handwriting pixels.
**Visible consequence:** Show current energy, attractor steps, retrieval weights, payload and classifier/reconstruction outputs. Step iteration without hiding its current state.
**Decision:** See how ambiguity, scale and address/content choices change retrieval and when classification and reconstruction objectives diverge.

### 34. Inspect scalar and matrix recurrent memory

Packet: [xlstm-extended-lstm](drafts/xlstm-extended-lstm/design.md). **Updated; implementation not started.**

**Controls:** Edit evidence gates/values, address vectors, chunk boundaries/state carry and supported digit pixels.
**Visible consequence:** Update stabilized scalar numerator/denominator, matrix-address contributions, causal chunk states and exact model outputs together.
**Decision:** Distinguish a probability normalization from signed matrix addressing and identify what state must cross a chunk boundary.

### 35. Change a filter, gate or sequence symbol

Packet: [hyena-long-convolution-models](drafts/hyena-long-convolution-models/design.md). **Updated; implementation not started.**

**Controls:** Edit convolution signals/kernels, gate vectors, causal/circular mode, recurrence poles/residues, chunk cut and supported biological-sequence symbols.
**Visible consequence:** Show dependency ranges, computed outputs, gated operator entries, retained state and exact frozen-model responses as edits apply.
**Decision:** Diagnose future leakage, choose effective context and distinguish exact recurrence carry from truncating the filter.

### 36. Move blocks without changing the attention result

Packet: [ring-attention-sequence-parallelism](drafts/ring-attention-sequence-parallelism/design.md). **Updated; implementation not started.**

**Controls:** Edit tiny Q/K/V and block ownership, merger order, causal positions, communication budgets and supported trajectory points.
**Visible consequence:** Show stable summary accumulation, legal work grid, circulating ownership, payload timeline and current output/error against the dense reference.
**Decision:** Separate mathematical equivalence from communication cost and identify invalid identity, masking or state-carry changes.

### 37. Make an optimizer decision visible

Packet: [advanced-optimizers-lion-sophia-prodigy-schedule-free](drafts/advanced-optimizers-lion-sophia-prodigy-schedule-free/design.md). **Updated; implementation not started.**

**Controls:** Edit gradient/momentum, curvature estimate, step scale, averaging state and supported real digit input for one update.
**Visible consequence:** Show the exact update vector and state changes for Lion, Sophia, Prodigy and Schedule-Free, including where gradients and evaluation occur.
**Decision:** Choose a debugging question from sign, curvature, scale and averaging effects rather than extrapolating a universal optimizer ranking.

### 38. Follow the field and the numerical solver

Packet: [neural-ode-continuous-depth-models](drafts/neural-ode-continuous-depth-models/design.md). **Updated; implementation not started.**

**Controls:** Edit vector-field parameters, initial state, step/tolerance, differentiation route, augmentation and supported real measurements.
**Visible consequence:** Show field arrows, accepted/rejected solver stages, current trajectory/error, derivative target and class output immediately or through bounded process steps.
**Decision:** Choose a solver/tolerance or representation from error, work and topology, distinguishing numerical approximation from the continuous equation.

### 39. Compare sequence memory and expert routes

Packet: [hybrid-ssm-transformer-architectures-jamba](drafts/hybrid-ssm-transformer-architectures-jamba/design.md). **Updated; implementation not started.**

**Controls:** Edit record keys/values, decay, score gap, cache budget, expert probabilities/capacity and supported stroke inputs.
**Visible consequence:** Show retained state versus explicit memory read, probability mass, exact request memory and continued frozen-model outputs.
**Decision:** Choose a hybrid arrangement by memory retention and routing costs; named architecture examples do not imply identical mechanisms.

### 40. Write into memory and inspect later reads

Packet: [titans-multi-memory-architecture](drafts/titans-multi-memory-architecture/design.md). **Updated; implementation not started.**

**Controls:** Edit key/query/value cards, rate, momentum, decay, request tokens and chunk size; step bounded writes or continue/reset request state.
**Visible consequence:** Show weight/update state, residual and gradient terms, current query output, gated topology and anchor/current-gradient comparison.
**Decision:** Choose write timing, state isolation and chunk semantics from the outputs they can affect, rather than from the word memory alone.

### 41. Separate examples, gradient buffers and updates

Packet: [mini-batches-training-loops-gradient-accumulation](drafts/mini-batches-training-loops-gradient-accumulation/design.md). **Updated; implementation not started.**

**Controls:** Edit rows, microbatch boundaries, learning rate, clearing/step policy, target weights and normalization groups.
**Visible consequence:** Populate row model outputs/errors/gradients immediately; step backward and optimizer events with distinct clocks and buffers. Compare final policies using the same rows and initial state.
**Decision:** Decide when accumulation is equivalent, what receives influence and why partition-sensitive operations change the computation.

### 42. Design a useful diagnostic and reproduce an operation

Packet: [neural-training-diagnostics-reproducible-experiments](drafts/neural-training-diagnostics-reproducible-experiments/design.md). **Updated; implementation not started.**

**Controls:** Edit tiny training rows/step settings, train/eval and graph modes, experimental evidence choices and restored checkpoint fields.
**Visible consequence:** Show gradient versus parameter movement, statistic buffers, comparable measured outcomes and the exact next operation under restored/missing state.
**Decision:** Choose a check that distinguishes a real failure from a null example and preserve the state required for a meaningful replay.

