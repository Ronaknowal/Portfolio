# RNNs, LSTMs & GRUs — content-first design and continuation

Prepared 13 September 2026. Stable ID `rnns-lstms-grus`; Deep Learning Fundamentals & Architectures position 14, current thirty-topic batch position 5. **Content authored; implementation not started.** Root owns final shared-ledger checkpointing and reconciliation.

## Authorized scope and source baseline

Actual preflight executed:

```text
node scripts/build-curriculum-inventory.mjs --topic rnns-lstms-grus --work content
```

It returned this foundation topic in Recurrent & Sequence Models, content in progress revision 1, implementation not started, no destination-topic note requiring disposition. The resolved bit-operation note was unrelated. Read the live published source completely in bounded chunks: lines 0–145, 145–355, 355–580, 580–820 and 820–end; a first overlarge truncated tool response was replaced with those complete reads.

Baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`. Published source `src/learn/data/topics/rnns-lstms-grus.jsx` SHA-256 **`7fd7f02af2f2eca7aceb5b34422793e571c186f8d8525138bfdf2c913ce0f2cf`**. It remains unchanged. This packet does not update title/slug, runtime, manifest, navigation, blueprint or publication.

Retain the established title: it covers the standard cells, their learning and practical use. Expanded scope adds an honest real completed-trajectory task, state ownership, causal preprocessing, full-versus-direct derivatives and framework reset conventions at their useful local home. It does not absorb full encoder–decoder training, attention, long-context benchmarks, state-space-model internals or later xLSTM variants.

## Learning design and sequence

Entrance: a numbered pen trajectory, ordered vector and orderless statistics make the input question visible before a recurrent equation. A beginner can understand what order preserves without knowing sequence-model terminology. Explain weights versus state and batch/position/coordinate axes locally.

First-pass route is immediately after introduction. Sequence: input representation → scalar recurrence → one shared-weight BPTT update → LSTM signed retain/write/read → compounded direct retention → GRU candidate/reset/blend → full real CPU experiment → fixed-weight point edits → state ownership and detach → packing/directions → optional derivative/cost/variants → changed practice.

Hurdle map:

| Likely learning obstacle | Local teaching response | Representation / learner action |
| --- | --- | --- |
| “Only an RNN can see order” | Ordered logistic baseline and explicit position columns | Point path versus flattened coordinates versus invariant statistics |
| New state mistaken for new weights | Fixed shared-weight legend, distinct states at each position | Edit one observation and identify unaffected prefix |
| BPTT feels like a different calculus | Scalar three-step chain, shared-weight contributions and actual update | Backward signed credit arrows and contribution table |
| LSTM as four unexplained boxes | Retain/write/add/expose arithmetic before gate equations | Signed cell accounting; output-only intervention |
| Positive forget bias guarantees long memory | Exact fixed-factor decay/half-life and separate full-state Jacobian | Design retained fraction/horizon; optional derivative block |
| GRU reset confused with clearing memory | Candidate path separate from z-retain blend | Before/after matrix mixing with computed counterexample |
| High confidence implies correct/robust | Actual development probabilities and reversal/swap probes | Edit real points with fixed model, Calculate and explain probability/class separately |
| Chunking, detaching and resetting collapsed | Separate forward-state and backward-credit connections | Boundary intervention and exact forward/null gradient checks |
| Padding zero treated as nonexistent | Valid-position ownership, causal versus backward path | Editable padding, packed/individual comparison, true final states |
| Cell names substituted for deployment reasoning | Completed-trace availability and actual data split | Changed live-prefix study-design exercise |

Readiness requires core cell/state/gradient/task understanding. Full block-Jacobian products, unusual variants and hardware optimization are optional branches. Scope boundaries occur once at their relevant mechanism or data home; programs print numeric results, not repeated disclaimers.

The previous Capsule lesson's routing is per-input iteration, not temporal state. Next is `sequence-to-sequence-encoder-decoder`, followed by `attention-mechanism-bahdanau-luong`. Actual full-curriculum links retain `module=deep-learning-fundamentals`. State-space/self-attention owners were told that pen data uses completed, spatially resampled traces, so no cross-dataset architecture ranking or live-prefix claim should be inferred. Their Libras sequence tasks remain independent.

## Original depth conservation and repairs

| Original material | Disposition |
| --- | --- |
| Motivation, unrolled graph, basic RNN/state, many-to-one/many-to-many | Preserved with a real pen trace, ordered baseline, local axes and explicit output arrangements |
| RNN/LSTM/GRU equations, from-scratch cells and native reference | Preserved in learner equations and complete independent NumPy `manual_sequence` plus native parity with nonzero state and biases |
| BPTT, vanishing/exploding gradients, clipping | Expanded to a complete scalar gradient/update, reliable norm-bound language, actual saturation and matrix-product counterexamples |
| LSTM cell/hidden distinction and gate controls | Preserved; three sigmoid gates plus candidate, signed write, hidden-output scope and local half-life arithmetic |
| GRU parameter comparison and formulas | Preserved with bias-convention counts; corrected reset-before/after and hidden-bias placement, z=retain convention, no literal “restricted LSTM” claim |
| Character-language-model loop | Replaced as the main applied program by full real pen classification; old loop carried h across unrelated random segments and its tiny repeated string was not real-data evidence. Next-token objective remains in the output-arrangement bridge; full generated-sequence program belongs next |
| Copy-memory/gradient heatmaps and performance diagrams | Replaced with executed fits and exact declared formulas. Old curves/heatmap values lacked reproducible source, copy chance/length claims were questionable, and h-only LSTM derivative hid the c path |
| Stacks, bidirectionality, padding, TBPTT, practical snippets | Preserved with complete packing program and actual native comparisons; last-index backward state corrected; detach separated from reset; state follows stream identity |
| Initialization and regularization | Preserved with effective one-sided forget bias, layer-dropout semantics and orthogonal-initialization limits |
| LSTM variants, alternatives, scaling, real uses | Preserved at accurate optional depth; projections/peepholes local, later special architectures linked by ownership |
| Failure modes and exercises | Integrated at causal home and eight changed questions with separate closed hints/solutions |
| Resources | Verified primary papers/docs/data plus annotated author article and official Stanford video/slides |

Removed unsupported historical/current rankings and numeric speed promises: no “only serious architecture,” universal T thresholds, fixed maximum RNN stack depth, A100 multiplier, automatic LSTM long-memory guarantee, universal fused-kernel speedup, or statement that every selective state-space model is a stationary convolution.

Technical repairs explicitly include:

- A feedforward classifier can retain sequence order through position-specific input columns.
- \(\|W_h\|>1\) does not guarantee exploding gradients; saturation and Jacobian directions matter. Individual time-varying spectral radii do not characterize the product.
- A non-peephole direct \(\partial c_t/\partial c_{t-1}\) entry is f when h is held fixed; total recurrent derivative uses (h,c).
- The original 1997 LSTM, its architecture-specific truncation, the later forget gate and modern full automatic differentiation are distinct.
- Effective PyTorch forget bias is the sum of input-side and hidden-side slices. Use one side1/other0; no `.data` mutation recommendation.
- Native GRU candidate reset is after the recurrent affine, including its hidden bias.
- Native `batch_first` does not change hidden-state axis order. Bidirectional final states differ from concatenating both components at the last sequence index.
- Right-padding can change final state; backward valid outputs can see padding. Packing and valid-target loss masking are separate.
- A state carried between optimizer updates is not necessarily the exact prefix state under the newly updated model.
- Irregular time and completed-trace normalization must be part of the task definition.

## Canonical reference agenda and coverage decisions

Canonical pedagogical reference: **Dive into Deep Learning 1.0.3**, actual chapter-index agendas for [chapter 9](https://d2l.ai/chapter_recurrent-neural-networks/index.html) and [chapter 10](https://d2l.ai/chapter_recurrent-modern/index.html), read from their published section lists before scope closure. This audit records scope, not a claim of reading every notebook backend.

| Actual reference section | Coverage decision |
| --- | --- |
| 9.1 Working with Sequences | Local ordered input/state/output intuition and causal availability |
| 9.2 Converting Raw Text into Sequence Data | Brief next-symbol bridge; a full text/token pipeline is owned by the next encoder–decoder lesson and existing tokenizer lesson. Real pen data supplies the complete local input route |
| 9.3 Language Models | Local next-symbol output arrangement; conditional sequence factorization/teacher forcing/generation owned next |
| 9.4 Recurrent Neural Networks | Full local recurrence, weight sharing, dimensions, output head and actual scalar trace |
| 9.5 Recurrent Neural Network Implementation from Scratch | Independent NumPy implementation of all three cells with gate traces and native parity; not the reference's Time Machine notebook |
| 9.6 Concise Implementation of RNNs | Full native CPU fitting program and shape explanations |
| 9.7 Backpropagation Through Time | Scalar/shared gradients, full chain and actual update; truncation and clipping; full-state sensitivity advanced |
| 9.7.1 full/truncated/randomized truncation, comparison | Full and ordinary truncated core; randomized unbiased truncation is not required for this first recurrence lesson and is retained as a linked deeper source rather than presented as the normal implementation |
| 10.1 Long Short-Term Memory | Full local modern gates, cell/hidden, fixed-path retention, two-bias implementation and limitations |
| 10.2 Gated Recurrent Units | Full local gate/blend, original versus PyTorch reset placement, actual fit |
| 10.3 Deep Recurrent Neural Networks | Stacking, changed layer input width, dropout placement and depth-versus-time distinction |
| 10.4 Bidirectional Recurrent Neural Networks | Offline/causal boundary, correct true final states, packing |
| 10.5 Machine Translation and the Dataset | Next topic owns its own actual source/target data and split; no borrowed scores |
| 10.6 Encoder–Decoder Architecture | Deliberate next-topic bridge, not a duplicate implementation |
| 10.7 Sequence to Sequence Learning | Next-topic owner |
| 10.8 Beam Search | Next-topic owner; not a recurrent-cell readiness gate |

D2L gate diagrams guide the mechanism questions, not copied visual assets/wording. Its broad statements about constant memory and practical full BPTT are narrowed to the explicit mathematical and task conditions here.

## Research record and actual review extent

All links checked/retrieved 13 September 2026. Only primary technical sources and original educational authors are used for consequential claims.

- **UCI Pen-Based Digits:** read current source schema/description/license and full original `pendigits.names` text in the public ZIP; source hashes, row counts, exact duplicate audit and licensing in `data-provenance.md`. Actual original writer pools documented, individual IDs unavailable.
- **D2L 1.0.3:** read chapter 9/10 actual agendas; 9.4 entrance/MLP contrast/recurrence/weight-sharing body; 10.1 gated memory-cell subsections through hidden-state mechanism; 10.2 reset/candidate/blend and beginning of scratch implementation; 9.7 unrolling/gradient analysis/full/truncated/randomized/comparison text through that comparison's start. Did not claim all backend notebooks or chapter 10.5–10.8 read for this packet.
- **[Pascanu et al. 2013](https://proceedings.mlr.press/v28/pascanu13.pdf):** read §1.1 and §2.1 temporal products/singular-norm bounds, portions of §2.3 geometry, §3.1 alternatives and §3.2 Algorithm1/threshold discussion, beginning of §3.3. No numerical benchmark is copied. Did not read every later experiment/supplement.
- **[Hochreiter & Schmidhuber 1997](https://www.bioinf.jku.at/publications/older/2604.pdf):** read abstract/introduction/outline, prior work and gradient problem, §4 input/output gates and constant-error cell equations/topology/truncation, experimental section agenda/summary tables and selected temporal-order discussion, and §6 limitation passage distinguishing full gradient from the original efficient truncated rule. Did not claim all 32 pages/appendix derivations were fully reviewed.
- **[Gers et al. 2000](https://pubmed.ncbi.nlm.nih.gov/11032042/):** verified paper metadata and abstract identifying adaptive forget-gate contribution. Full paper not read, no detailed experimental reproduction claim.
- **[Cho et al. 2014](https://arxiv.org/pdf/1406.1078):** read §2.2 encoder–decoder factorization/conditioning and §2.3 reset/update equations and explanation, plus boundary into the SMT use. Next packet will inspect its fuller encoder–decoder context. No original corpus scores reused here.
- **PyTorch 2.14.0:** GRU main API equations, gate order, shapes, biases and intentional reset-placement note read; LSTM equations, projection/input/output shapes, bias/stack/dropout semantics and final-state distinction consulted; packing main API and CPU lengths/enforce_sorted behavior read; clipping main API norm/return/nonfinite behavior read. Local installed versions independently recorded and bounded native programs executed. No GPU throughput measured.
- **[Olah 2015](https://colah.github.io/posts/2015-08-Understanding-LSTMs/):** read article mechanism text from long-dependency motivation through gates, cell update, output and variants. Used as an annotated alternate visual explanation; strong default-memory/general success rhetoric is qualified locally. Did not copy diagrams.
- **[Stanford CS231n 2017 syllabus](https://cs231n.stanford.edu/2017/syllabus):** verified Lecture10 ownership, agenda and official video; [video metadata/description](https://www.youtube.com/watch?v=6niqTuYFZLQ) identifies the university channel and recurrent/captioning/attention scope. Read extracted slide text on gradient flow, gates, variants and summary (approximately slides93–104). Video not watched in full; no timestamps claimed. Slide shorthand “singular value>1 means explosion” is explicitly not used as a theorem in this lesson.
- Deep Learning Book web chapter retrieval exceeded provider content limits; it was not counted as a reviewed canonical source. The directly accessible D2L agendas supply the actual canonical coverage audit.

## Programs, arithmetic, outputs and review boundary

Retained source/data packet:

- `lesson.md` full learner manuscript with a complete nine-fit Python program and complete independent packing example.
- `visual-specifications.md` six mechanism-specific forms, including real point edits, explicit input/output consistency/result checks and computed contrasts/nulls.
- `prepare-pen-data.py` / `pen-trajectories.csv` / `data-extraction.json` / `data-provenance.md` reproduce and explain the openly licensed real inputs.
- `pen-sequence-learning.py` / `calculated-inputs.json` retain all actual fits, baselines, fixed-model probes and weights.
- `recurrent-mechanics.py` / `mechanics-results.json` provide independent NumPy gates/traces, actual scalar/manual backward check, state/padding fixtures, Jacobian and mathematical contrasts.
- `author-checks.py` / `author-check-results.json` check changed practice arithmetic, exact disclosed-program/source match, execute the small packing program, check 27 saved result probability/count pairs, source/data hashes, closed practice disclosures and independent finite-difference Jacobian.

Executed in read-only shared Python runtime, one CPU thread, no installs. Nine fits ran once; later checks did not repeat them. Source/data byte identity verified. Float64 nonzero-state/bias native parity ≤2.23e−16; saved fitted float32 versus independent NumPy probability error ≤5.97e−7, hidden error ≤3.29e−7. Full-state Jacobian finite-difference error 7.77e−12. The scalar updated loss and changed practice values are actual arithmetic, not inferred from a plot.

The point3 x edit's small numerical effect is intentionally retained instead of seeking a dramatic flip. Whole-development reversed/swapped outcomes come from the same actual fitted weights. All fits are descriptive development evidence, no untouched final test or architecture speed ranking. The A/B stream-assignment fixture was also executed: wrong state ownership changes hidden outputs by up to 0.0426162211, while matched input/state reordering is an exact null. Its actual inputs, prefix states and weights are retained.

Author closure: full manuscript and visual-specification reread, equation/variable/shape/sequence/resource review, source-conservation and learning-experience checks are required before freeze. Record completed dispositions below when that reread finishes. Publication, runtime implementation, full native campaign, independent phase-two correctness/learning review and browser/accessibility/integration checks remain deferred.


## Completed author closure

Read the full written manuscript in three bounded file reads and the entire visual specification after writing. The reread corrected a changed-practice arithmetic approximation to 0.733564, made the elementwise product notation explicit, aligned the fitted-path prediction question with the selected-class grading contract, widened the manual forget-factor bound to include every derived target/horizon pair, and retained exact native boundary/padding weights for portable implementation. It also motivated the bounded A/B ownership contrast above. No model fits were repeated. **Historical interaction record:** the earlier prediction/reveal behavior described here is superseded by the 21 September live-exploration contract; it is not a phase-two implementation requirement. Preserve the recorded mathematical checks and fixtures.

Learning-experience checklist completed: intuition before terms; explicit first-pass route; local notation and tensor axes; prior/next links in actual order; full mechanism and runnable real-data program; computed visual forms matched to each hurdle; distinct exact-formula and empirical evidence; useful application and availability limits; changed independent practice with 16 closed hint/solution blocks; annotated alternate resources; deeper material separated from readiness; browser compute bounded and lazy-load handoff explicit. Data and published-source hashes still match. Native and author checks passed after the final affected calculation changes. This is author closure of content, not independent phase-two review, publication or user acceptance.

## Focused reconciliation: fresh investigation defaults

13 September 2026: root identified that B/C/D and E reused immediately worked examples as gated defaults. The manuscript and all nine fits remain unchanged. The visual specification now explicitly separates A's ungraded representation diagram from E's computed real-input investigation and B's scalar credit investigation from F's later state-boundary investigation. B uses fresh inputs/weights/target, C fresh cell/gate and retention questions, D a fresh matrix/reset problem, E a different real specimen (pendigits.tes:6, actual1), and F a fresh boundary3 plus lengths5/4/2 padding problem. Existing worked values remain explanatory fixtures; all fresh answers are author-only until live comparison.

Executed fresh-investigation-fixtures.py once with the existing NumPy/PyTorch CPU runtime and fixed saved models. Retained the full fresh inputs, actual original/changed pen-state traces, scalar losses/gradients, gate/retention curves, GRU placement values, state gradients/ownership and padding checks in fresh-investigation-fixtures.json. Verified target-only, zero-rate, zero-input-gate, reset-ones/diagonal, repeated-input, unchanged-prefix, reverse-twice, carry/detach, owned-state reorder and ignored-padding nulls. No data change, repeated fit, browser work or phase-two review. Reread the changed specification sections and checked whitespace. Original author checks/results remain valid for the unchanged manuscript/model experiments; the new file records the additional bounded calculations.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Carry and edit recurrent state. Edit sequence entries, recurrent weights, LSTM gates, GRU reset placement and supported pen-trajectory coordinates. Update state trajectories, retained/injected terms, shared-weight credit and exact learned outputs. Step, rewind and reset state explicitly; padding and request boundaries remain visible. Choose what must persist or reset, and diagnose saturation, reset-order differences and accidental cross-sequence leakage.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Translate gate equations into a reusable recurrent implementation” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| RNN/LSTM/GRU mechanisms, states and gate conventions | `recurrent-mechanics.py:manual_sequence, scalar_credit`; NumPy exact native-weight route | `pen-sequence-learning.py:PenClassifier`; nn.RNN/LSTM/GRU actual training | Chunk5 full/short-tail/detach extension with solution; exact gate/bias mapping |
| Padding, batch identity, clipping and gradient memory | `state_and_padding`, author-checks and current complete model loop | pack_padded_sequence and clip_grad_norm_ used with stated semantics | Existing reorder/reset/pad practice retained; general AD is actual earlier owner |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.
