# Modern Hopfield Networks — content design and continuation

Prepared 13 September 2026 by classical_probabilistic_content. **Research and writing complete; implementation not started.** Root owns the central checkpoint. This packet contains a complete manuscript, visual/investigation contracts, instructional programs and retained data/evidence. It does not change the published lesson, catalogue, navigation, manifests or browser components.

## Preflight, original source and scope

Actual preflight: `node scripts/build-curriculum-inventory.mjs --topic modern-hopfield-networks --work content`. Advanced topic, module position 33, Alternative Architectures & Historical Models; historically published, current content in progress, implementation not started, individual design required. No destination note exists for this ID. No relevant unresolved unassigned note was found.

The complete original `src/learn/data/topics/modern-hopfield-networks.jsx`, all 1,025 lines, was read in contiguous ranges, with follow-up reads of the initially truncated output. Its SHA256 is `d17566f43c90d83fbb24f0bb9ba51f6779a930a6b6cac366279a00d68fbdbcf5`. Baseline commit: `8c5da59f18516be77c29d5aeeafca3decca4f738`. The final bounded author check confirms the same source. Existing useful coverage is conserved below; historical publication is not evidence of technical review.

Keep the title **Modern Hopfield Networks** and stable ID. The topic owns classical associative-memory prerequisites, the continuous softmax energy, attention correspondence, learned memory geometry, realistic retrieval assessment and proportionate higher-order/current energy-derived extensions. Hopular and DeepRC are applications of the mechanism, not new unrequested full lessons. Energy Transformer belongs here as a short, carefully distinguished advanced connection; ordinary transformer blocks do not inherit its energy construction.

The module predecessor is Spectral Normalization & Gradient Penalty; the next topic is xLSTM. Local bridges refresh signs, dot products, weighted sums, softmax, tanh, gradients and matrix shapes. Prerequisite links supplement those explanations. Actual route links preserve module context; the packet never jumps navigation to a later published topic. Parent owns Boltzmann Machines: this lesson distinguishes deterministic state descent from stochastic conditional sampling and partition-function learning without duplicating that lesson. Parent's attention integration packet and the attention author's self-attention lesson own transformer details. The ODE link makes energy descent a future connection, not an assumed prerequisite.

## Intended learning experience

Running question: how can a damaged cue retrieve useful stored information, and what does a successful read actually mean? Start with a four-feature repair that a learner can calculate, move to a continuous weighted read whose geometry is visible, separate keys from payloads, then fit the similarity function on real handwriting. The practical task deliberately measures recognition and pixel reconstruction separately.

The first-pass route is §§1–6, investigations and exercises 1–7. Optional §7 branches cover margins, capacity definitions, higher-order memory and causal extensions. §8 applications explain the supervision/data flow, with a final changed experiment-design exercise. No fixed lab or figure quota shaped the design.

| Learner hurdle | Explanation and observable outcome | Representation / transfer |
| --- | --- | --- |
| “Memory” sounds like an indexed array | Distinguish address lookup from cue-based association and distinguish recall, reconstruction and classification | Lookup diagram, three evaluation branches |
| Similarity seems identical to distance | Unequal vector norms can reverse a dot-product preference | Exact two-memory norm trap with distances and scores |
| Hebbian weights seem arbitrary | Build signed pairwise products, remove self-connections, sum current neighbor votes | Signed edge diagram and selected matrix row |
| Updating every bit at once seems harmless | Derive one-coordinate energy difference; demonstrate a synchronous two-cycle | Coordinate trace, stepwise energy and changed visit order |
| A modern read must return one stored row | Compute opposing-memory softmax and repeat tanh | Weight bars, convex hull, cobweb plot and editable cue |
| Stored patterns and fixed points are confused | At finite β the stable point differs from the stored coordinate; symmetry can remain fixed | Correct saddle and stable-point labels |
| An energy gradient is treated as a solved state | Derive fixed-point equation and one-step majorization inequality | Fixed-X contour plus measured energy strip |
| Attention “equivalence” is taken too far | Separate key-space refinement from independent value-space payload | Two lanes and a payload edit that leaves weights unchanged |
| Learning an association is confused with iterating a state | Derive class-mass gradient, then show parameter fitting through both projection uses | Loss/gradient before–after and training graph |
| A clean demonstration proves generalization | Fixed memory, fit-query, validation and separate-writer test roles | Data boundary diagram and two-seed outcome table |
| A sharper reconstruction means a better decision | A damaged 1 becomes a plausible 0; MSE and class mass answer different questions | Real 8×8 pixels, weighted images and top-reference thumbnails |
| Exponential storage is read as free compute or a universal count | State theorem conditions, derive a gap bound, calculate explicit bytes and dense arithmetic | Competitor-mass diagram and separate storage/work axes |
| An interesting application is only a name drop | Explain variable-size bags, label supervision, feature/sample reads and changing-key energy | DeepRC/Hopular data-flow diagrams and scoped ET comparison |

## Conservation and consequential corrections

| Original useful coverage | Prepared disposition |
| --- | --- |
| Classical history, Hebbian storage, energy, recall and familiar capacity claim | §§1–2 retain the mechanism with a complete asynchronous trace, tie rule, two-cycle counterexample and measured finite-bank scan. Distinguish exact fixed states, damaged-cue recall and the particular asymptotic error criterion. |
| Continuous memories, β, log-sum-exp, fixed points, CCCP and retrieval regimes | §3 derives all operations and the majorization bound. Two-memory geometry exposes global averaging, separated attractors, saddle symmetry and finite-temperature displacement. |
| Attention relationship and learned associations | §4 supplies exact shapes, the K=V energy boundary, independent values, trainable queries/keys and class-mass gradient; executed PyTorch scaled-dot-product parity. |
| Classical capacity, Gaussian-memory demonstrations and iterative retrieval code | Replaced by complete original programs with all inputs, emitted results and interpretation; no fabricated large-memory success benchmark. |
| Modern Hopfield library, static lookup/pooling and integration examples | Three layer forms are explained with input/parameter ownership; official maintained source is annotated, and legacy environment requirements are disclosed rather than copied as current installation guarantees. |
| DeepRC, Hopular, few-shot recognition, drug/retrieval ideas | DeepRC and Hopular receive concrete mechanism/supervision detail; few-shot support sets connect directly to the executed handwriting task. Unsupported universal production-use claims are removed. |
| Higher-order memory and activation/prototype connection | §7 preserves the mechanism and introduces an exact parity example showing why an order change matters. |
| Capacity/latency/memory figures, attention maps and practical decision criteria | Calculated geometry, measured eight-bank scan, explicit byte/work accounting and real memory-weight images replace unsupported timing or capacity curves. |
| Failure modes, questions and references | Consolidated mechanism-based cautions, ten changed questions with closed hints/solutions, four distinct investigations and annotated primary/alternate learning routes. |

Specific repairs to the original reasoning:

- Asynchronous coordinate descent with zero diagonal and ties preserved has a direct energy argument. Parallel sign updates can cycle. Self-connections do not universally force an all-positive state.
- A memory row, a stationary point, a stable attractor and a global minimum are distinct. The fixed-point equation is not a closed-form solution. One update may retrieve approximately; it is not always exact convergence.
- The negative log-sum-exp term is concave. The majorization update lowers the same fixed-bank energy; changing keys, arbitrary value maps, residuals and a full transformer stack require new analysis.
- Capacity depends on the success definition and pattern distribution. Roughly 0.138d is not a universal finite hard limit or the criterion for every pattern to be an exact fixed point. The conditional random-sphere theorem is not an unconditional exp(d/2) count for learned keys.
- An explicit bank still occupies P×d storage and a dense read costs work proportional to P×d. Efficient attention avoids score materialization without eliminating dense pair arithmetic. Mamba is not merely a softmax approximation.
- Association mass can be useful for labels without being calibrated confidence. Learned class geometry need not optimize pixel reconstruction or every corruption distribution.
- A majority of successful examples or a favorable seed cannot erase failures. Both predefined fits and unfavorable occlusion outcomes remain visible; no universal architecture ranking or GPU timing is claimed.
- In a causal age-bias bridge, define logits directly as β times similarity minus γ times age. The source appendix has inconsistent βγ placement in adjoining formulas; the manuscript states its own unambiguous operator instead of copying that notation.

## Actual research and claim locators

Research date: 13 September 2026. Technical claims use primary papers, official author resources or explicit local derivations. This list records the material actually read; it does not claim every appendix or linked video was consumed.

| Source / actual reading | Claims and manuscript placement |
| --- | --- |
| [Ramsauer et al., Hopfield Networks is All You Need, v3](https://arxiv.org/html/2008.02217v3), full main §§1–4 (paper pp.1–10), full appendix section list, A.1.1–A.1.2 opening energy argument and A.1.8 forgetting/causality | Canonical energy Eq.2, update Eq.3, Theorems 1–5 and capacity Eq.4, layer forms and attention Eq.10 → §§3–4/7. Main theorem statements and assumptions were inspected; not every 94-page proof was read. Our one-step inequality and current-cue gap bound are separately derived and checked. |
| [Krotov and Hopfield, Dense Associative Memory, v2](https://arxiv.org/html/1606.01164v2), §§1–4 and §5 introduction, source section list | Higher-order energy/update, parity limitation and feature-to-prototype mechanism → §7. The constructed four-pattern parity calculation is original and exhaustively checked for its eight assignments. |
| [Hopfield 1982](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/), abstract, metadata and scanned article index | Historical attribution. Full scanned pages were not read; the binary mechanism also rests on the canonical modern source and an explicit derived local-energy proof. |
| [McEliece et al. 1987](https://authors.library.caltech.edu/records/q92rz-95p89), author-institution abstract and publication metadata | Most-versus-all exact-recall scales n/(2 log n) and n/(4 log n) → §7. The 22-page proof was not read or reproduced. |
| [JKU author tutorial](https://ml-jku.github.io/hopfield-layers/), classical, modern, attention, layer types, DeepRC and materials sections | Alternate ground-up route and implementation orientation; corroborates three input/parameter configurations and points to the recording. Read as an article, not a claim of watching embedded media. |
| [Official Hopfield layers repository](https://github.com/ml-jku/hopfield-layers), README installation, usage and examples | Verified library and notebook discovery → references. Original documented Python 3.8.3/PyTorch 1.6.0 context is old; package/notebooks were not installed or run. |
| [DeepRC](https://arxiv.org/pdf/2007.13505), introduction, Figure 1, supervision and Deep Repertoire Classification formulation/algorithm sections | Variable-size repertoire bags, encoder, learned-query pooling and bag-level loss → §8. No new clinical validity or reproduced benchmark claim. |
| [Hopular](https://arxiv.org/pdf/2206.00664), introduction, §2 context and §3 architecture including Hs/Hf, masking and refinement | Separate sample and feature memories, target masking and residual refinement → §8. No current benchmark dominance claim. |
| [Energy Transformer](https://arxiv.org/html/2302.07253v1), section list, §§1–2 and §§3–4 introductory task descriptions | Energy built over changing token variables includes key as well as query derivatives, tied structure and normalized dynamical update → optional §7. This is not the energy of an arbitrary ordinary transformer. |
| [UCI Optical Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), full metadata/license page and complete original optdigits.names | Input acquisition, source roles, separate writers, class/features and permission → §5 and data-provenance.md. Byte-original data downloaded from the official ZIP. |
| [Yannic Kilcher recording](https://www.youtube.com/watch?v=nv6oFDp6rNQ), title/search metadata and JKU author-page link | Optional alternate learning resource only. Direct video retrieval failed; neither the recording nor a transcript was watched/read. The lesson states that limit and recommends the paper for exact assumptions. |

### Canonical section-list coverage decision

The Ramsauer main paper supplies the central objects, update, guarantees, layer forms and application families. Appendix A.1's energy/global/local/fixed-point/storage/association/forgetting/spurious-state organization was explicitly checked against the manuscript. Energy/update and a sufficient descent argument are core; local Jacobian and conditional capacity are optional; association training is concrete; causal forgetting is a scoped bridge; mixture and unwanted states are visible in the two-memory geometry. Infinite-memory proofs and all probabilistic capacity lemmas are optional primary reading, not prerequisites silently omitted from a claimed full proof.

A.2's log-sum-exp, softmax and Lambert-W tools are refreshed at their point of use; exhaustive Legendre-transform details would not help the core retrieval task. A.3 higher-order binary memory has a worked logical example. A.4 attention has an exact operator/dimension bridge. A.5's BERT-head, MIL, UCI and drug experiments provide application context; the lesson substitutes an openly reproducible handwriting experiment for copied benchmark tables and does not assert reproduction of those studies. A.6 usage is covered through three layer forms and the verified official library; backend integration is phase two when needed, not a speculative install in this content phase.

Failed retrievals were handled without inventing evidence: arXiv v4 did not exist for the selected canonical paper, web PDF exceeded the parser limit, and OpenReview presented a challenge. The actual v3 PDF was downloaded for local text extraction; its v3 HTML later worked. The temporary extraction was read and then removed after recording locators. Energy Transformer PDF endpoints failed before the HTML succeeded. Video access remained limited as disclosed. No failed URL was used as an implementation oracle.

## Programs, data and executed author evidence

The packet retains complete downloadable programs rather than isolated pseudo-implementations:

- `associative_memory.py` writes `mechanism-results.json`: binary storage/coordinate recall, eight random-bank scans, continuous iterations/energy, exact attention equivalence, gradient step and polynomial parity.
- `digit_memory.py` writes `digit-results.json` and `digit-memory-fits.npz`: all data roles, baseline grid, two selected projections, curves, confusion matrices, association weights, clean/occluded metrics and source IDs. Both fits were actually executed.
- `author_calculations.py` writes `investigation-checks.json`: fresh/changed/null visual fixtures, real edited pixels and complete reconstructed arrays, gradient finite differences and association/payload edits.
- `check_author_packet.py` writes `author-checks.json`: executes the actual inline manuscript function against both saved-model reads, verifies the changed exercises, source hash, disjoint roles, unique data, archive finiteness, confusion totals, download/route existence and disclosure/visual counts. This is an author check, not an independent formal review.

Executed environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, single-threaded small author runs in the existing shared runtime. No package installs, pretrained models or GPU campaign. Setup/download/run instructions appear with the practical program. Optional official-library notebooks remain unexecuted.

Key evidence reconciled with prose/specs:

- Binary worked energy [0,0,−1.5,−1.5,−1.5]; changed third-coordinate exercise field .75 and energy change −1.5; synchronous two-cycle retained.
- Continuous β=2 read [.3799489623,0], then [.6410168327,0]; β=.5 averages toward zero. Negative fresh cue, symmetric null, high-temperature wrong-side choice and duplicate-memory effect are separately checked.
- Fixed-X energy gradient maximum finite-difference discrepancy 1.16e−11; PyTorch attention parity discrepancy below 3e−17; class-mass analytic/autograd gradient discrepancy below 1e−16.
- Exact unique features: all 5,620 images distinct. Internal source IDs disjoint across 200 memory, 800 fitting-query and 300 validation rows. Original 1,797-image separate-writer test preserved. Full provenance and hashes are in data-provenance.md.
- Selected pixel β=64 has 136 clean / 667 occluded test errors. Seed 17 has 100 / 564; seed 41 has 113 / 740. The unfavorable seed-41 occlusion result remains. These are this small protocol's outcomes, not architecture rankings.
- The fresh digit investigation uses validation source 3748, different from worked source 2946. It includes a failed central-column occlusion, a single editable pixel, success source 3052, restoration identity and an all-blank all-class tie. The UI must not call index-order argmax a confident prediction for that tie.
- The inline read matches both saved-model functions exactly in the current CPU check. The tight changed gap bound is approximately .667614375 for target mass and .664771250 for distance. Stored one-million-by-64 float32 memory is 256,000,000 bytes, not millions of free memories.

Validation/selection uses clean data by a predefined rule. The baseline script saves test reports for every predefined β while its selection expression uses validation alone; the candidate grid was not tuned after seeing assessment. Two fit seeds were predefined, seed 17 is the demonstration default, and occlusion was fixed. The experiment has no per-writer internal validation grouping because the files lack those identifiers. Recognition loss does not directly optimize reconstruction. Limitations are taught where they affect interpretation, not accumulated as generic warnings.

## Author reread and learning checklist

Completed full own-manuscript reread from the opening through every exercise, solution, application and reference; completed full visual-specification reread, including a separate re-read of the initially truncated figure table. Inspected the source/provenance, instructional program logic and actual emitted results. This is one author's final reconciliation, not a claim of independent review.

- The opening supplies a practical reason to care and explains association before terminology.
- Every core symbol and operation receives a local explanation; projection Wₑ is distinct from memory count P, and theorem scalar K is distinct from key matrix K.
- Each core mechanism has an inline explanatory representation at its introduction, before requiring lab interaction.
- The four investigations use different meaningful entities: signed bits/order, geometric memories/cue, paired keys/values and real handwriting pixels. All fresh Current results are visible on opening. and are bound to active inputs.
- Contrast and null fixtures are computed, including failures, ties, payload changes and restore behavior; results are not hardcoded responses to presets.
- Core and optional routes are explicit; no undocumented transformer prerequisite or forced source-paper section template.
- Exercises change inputs, criteria or constraints; all ten hints/solutions are initially closed. Practical outcomes include interpretation, failure cases and a fair extension plan.
- Canonical research coverage was considered beyond the source's opening equations. Nuanced claims, data permission and numerical examples were investigated rather than simplified into false statements.
- Real training has explicit fitting/selection/assessment roles and retained licensed offline inputs. The reader can reproduce the small experiment from the complete program.
- Resources are direct, annotated and honest about what was read; no video viewing, full-paper reproduction or library execution is fabricated.
- Topic predecessor/next links and related links are real stable IDs with module context. The manuscript and specs are ready for root's source-bound content checkpoint.

## Deferred implementation work

Phase two implements all 21 inline figures and four investigations from visual-specifications.md, derives compact attributed lazy digit/model assets and retains the complete downloadable packet. It must validate pure browser numerical functions against the saved fixtures and full validation arrays, preserve blank/tied outcomes and score/value distinctions, and test input-bound live recomputation, keyboard access, narrow layouts, reduced motion, alt/table alternatives and reset behavior. No browser training or eager loading of all models/data.

Phase two also handles production lesson structure, download URLs, approved title/catalogue decisions if any, source-bound implementation review, runtime/build/navigation checks and browser performance/accessibility evidence. It must not mark this content checkpoint as implementation completion. Root performs the shared ledger/checkpoint updates and any cross-packet review. No open author content finding remains; future review can still correct substantive issues rather than treating the freeze as a ban on evidence-based changes.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Shape an associative memory. Flip cue bits and visit order; edit continuous memory vectors, temperature, keys/queries/values and real handwriting pixels. Show current energy, attractor steps, retrieval weights, payload and classifier/reconstruction outputs. Step iteration without hiding its current state. See how ambiguity, scale and address/content choices change retrieval and when classification and reconstruction objectives diverge.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Implementation ownership and content-depth revision — 22 September 2026

Delivery remains **content-first**. The complete computational teaching route is part of this prepared packet now; phase two receives written code, explanations, mapped state/settings and closed practice, rather than an instruction to invent the missing mechanism. Earlier authoring records remain dated evidence; this section supersedes their incomplete depth handoffs. The title and stable ID are retained because the new material fulfills the existing scope.

| Advertised computational outcome | Scratch/source owner | Ordinary tool route | Matching bridge | Independent practice | Scope boundary |
| --- | --- | --- | --- | --- | --- |
| Classical binary storage, asynchronous retrieval and energy | associative_memory.py::store_binary/binary_recall/binary_energy | NumPy arrays and indexed coordinate updates | explicit zero diagonal, tie-preserving asynchronous state | Practice 1,9; order-dependent examples | Finite binary network; not universal capacity guarantee |
| Stable modern retrieval and attention relation | associative_memory.py::retrieve/iterate/energy | torch.nn.functional.scaled_dot_product_attention in main | matched Q/K/V/scale; new query-gradient exercise | Practice 2–4,8; distinct-value gradient extension | Energy interpretation requires its fixed-bank assumptions |
| Learned bank and class-mass read | digit_memory.py::read_memory/main | nn.Linear, log_softmax/logsumexp, nll_loss, Adam | projection parameters and log class-mass learning | Practice 5–7,10 | Explicit tensor composition is ordinary route; optional hflayers APIs are not required outcomes |

All local source owners above were inspected at their actual function/class definitions. Full model fitting, data/provenance and existing worked results are retained. Reused actual prerequisite code is named explicitly in the manuscript; prepared owners are not described as already published updated instruction. Whole-family releases mentioned for context do not expand the promised executable outcome into every checkpoint or every GPU kernel.

The teaching sequence is construct → explain the state/update → normal tool use → compare the same contract → changed-constraint practice, inserted where the relevant mechanism is explained. Original mechanism programs remain canonical; new programs depend on them only where the import is explicit. No browser program, published lesson, manifest or curriculum sequence is changed by this revision.

Author checks for this revision: source/API-contract reading, Python syntax parsing, matching embedded/downloadable source and local links, and scoped arithmetic probes where recorded in the specialist-writing report. These are content-authoring checks. Earlier fit outputs remain their original evidence; new multi-process/GPU/specialist-package execution, formal independent implementation review, rendered diagrams/labs and browser/accessibility/integration checks are **deferred**, with exact targets in the current visual specifications and specialist report.

Next action: after the root records the new content checkpoint, consume the full current packet for an authorized finish request, execute the relevant new programs and capture honest outputs, build the specified topic-owned views, independently check the translated models and integrate them. No core scratch/library manuscript writing is left as a finish-only TODO.
