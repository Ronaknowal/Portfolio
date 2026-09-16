# Hidden Markov Models — content design and handoff

Stable ID: hidden-markov-models-hmm. Classical ML position 27, batch position 9. Research/write complete; implementation not started. Root took ownership before any packet files were written; the feature author confirmed no prior HMM work. This packet is part of the authorized 30-topic content-only scope.

## Entry and source conservation

Ran the exact topic inventory with --work content. No incoming HMM note exists; the resolved bit-manipulation item is unrelated. Current teaching policies were read and reused within this ongoing request. Read the complete original src/learn/data/topics/hidden-markov-models-hmm.jsx, including all examples, algorithms, library programs, visual values, claims, comparisons and six exercises. Baseline commit 8c5da59f18516be77c29d5aeeafca3decca4f738; original SHA256 9f99c9c7edab75df56ec98b2eb628a0cb97a19beeabb7e4e47a7d7e22200614c. No runtime source is edited.

Predecessor AutoML concerns selecting procedures; HMM now introduces a specific sequence-generating model. Refresh discrete probabilities, matrix rows and dynamic programming locally. Earlier GMM provides a nonadjacent latent-mixture/EM connection, not a false “previous lesson.” Next is Bayesian Networks, then CRF. Keep that actual order and distinguish joint generation, posterior queries and discriminative sequence labeling.

Title and identity remain appropriate. Preserve evaluation, decoding, learning, log-space implementation, backward/gamma/xi, categorical/Gaussian library examples, sequence comparisons, scaling, failure diagnosis and deeper graphical connections. Expand important missing outcomes: filtering versus smoothing versus forecast timing, marginal decoding versus legal joint paths, sequence-boundary counts, hard structural zeros versus smoothing, geometric dwell times, known-state supervision versus latent-state EM, and a real attributed sequence dataset.

## Issues found in the old source

- Weather/activity is falsely attributed to Rabiner's exact tutorial; the tutorial's introduction identifies coin-tossing and urn examples. Do not repeat that attribution or claim an inferred weather label is verified truth.
- The same Viterbi trellis has inconsistent predecessor and numerical values between displayed code and visual steps. Derive one reference and use its actual values.
- Baum–Welch guarantees nondecrease for the exact matching objective, not strict increase, global recovery or necessarily better hidden labels. A flattened curve does not prove global convergence. Report actual model history at the same parameter version as outputs.
- Label permutation alone does not explain arbitrary fitted parameter differences. State indices, semantic identifiability, sample error and poor local fits are separate.
- Smoothing uses future observations and is not available to an online decision at that time. It cannot be called universally more accurate on every case.
- The original plot describes increasing negative log-likelihood values as a downward curve; axes and named quantity must agree.
- Float64 underflow assertions are numerically false (0.3^100 is representable). All-zero evidence can also be a genuine impossible event. Normalized scaling with retained factors is a valid exact alternative to log-space; a generic epsilon changes structural zeros.
- Dense recurrences have TN² transition terms, not an exact hardware operation/time guarantee. Sparse edges and beam candidates must account for successor expansion; a beam is not generally O(TK). Categorical emission counts can be scattered by observed symbols instead of requiring TNM work. Do not inherit unsupported state/dimension/device ceilings.
- More states permit at least the same optimum only under a nested model construction; a finite local fit can worsen. Equal emission means do not prove duplicate states when variances/dynamics differ. AIC/BIC need parameter/observation conventions and regularity caveats.
- Correlated observations are allowed marginally; it is conditional factorization that is constrained. Delta features do not magically restore conditional independence. CRFs need not use only local input windows, neural encoders can retain CRF output transitions, and no universal data-count/model/GPU winner rule is justified.

## Research read (12 September 2026)

- Jurafsky and Martin, [SLP Appendix A](https://web.stanford.edu/~jurafsky/slp3/A.pdf), current fetched draft **19 August 2026**, 17 pages. Read A.1 Markov Chains; A.2 HMM; A.3 Forward; A.4 Viterbi; A.5 Forward–Backward/EM; A.6 summary and historical notes/references. Pages 14–17 were additionally extracted directly in memory to cover the web extract's omitted tail. This is the canonical self-contained chapter: conserve its whole learning progression with independently authored examples. It includes prose slips about joint versus conditional likelihood; use the explicit correct recurrences rather than copying prose mechanically.
- Rabiner1989, [30-page tutorial](https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf). Actual section list retrieved across all pages: I Introduction; II discrete chains/hidden extension/elements/three problems; III solutions; IV types (continuous, autoregressive, null/tied structures, explicit duration, ML/MMI/MDI, model comparison); V implementation (scaling, multiple sequences, initialization, insufficient data, model choice); VI isolated-word system (features/quantization/segmentation/duration/results); VII connected words; VIII large vocabulary and limitations; IX summary. Read introduction's examples/progression and selected IV duration plus V scaling/multiple-sequence text. Full paper not yet claimed read. Columbia copy returned an error; usable university mirror succeeded. Historical speech benchmarks are not current performance guarantees.
- [hmmlearn 0.3.3 tutorial](https://hmmlearn.readthedocs.io/en/0.3.3/tutorial.html), official overview/model table, training/decoding, monitoring and sequence lengths; official API CategoricalHMM constructor/implementation modes; official Multinomial example's count-vector versus category distinction. The tutorial calls EM gradient-based; this is not a generally correct characterization and will not be repeated. Native API execution is deferred explicitly in the final packet.

## Intended learning forms and data

Use an unrolled graph and a trellis whose edges carry actual multiplied contributions, not a text-only step box. Separate sum, maximum and posterior marginals; let a changed observation alter earlier smoothing while leaving a past filtered belief unchanged. A second investigation edits a legal path/topology and diagnoses disagreement with pointwise decoding. A third edits observed sequence entities/boundaries and inspects expected counts and one EM update. Place dwell-time, numerical-scale and real sequence comparisons beside their explanations; do not impose a lab quota.

The real branch reuses the existing attributed English EWT r2.16 short-sentence extract from the CRF packet without changing it: 120 train, 40 development and 40 reserved test sentences, 1,188/341/370 tokens. A supervised count-based HMM and simple lexical baseline will expose transition usefulness and unknown-word failures. This is supervised parameter learning with hidden labels at prediction time, not proof that unsupervised states recover grammatical categories. Preserve original UPOS, lossy coarse mapping, sentence IDs, exact source/license notices and declared educational sampling limitations.


## Final content completion and canonical conservation

Research/write complete12 September2026; implementation not started. Manuscript and complete visual contracts were reread in full during authoring/reconciliation, alongside program and saved numerical values. Reconciliation corrected an ordinal/time-index mismatch in the scaling and investigation prose, clarified observed-only emission denominators when reports are missing, and replaced ambiguous “four binary states” with four time steps/two states.

| Canonical SLP Appendix A section | Local conservation |
|---|---|
| A.1 Markov chains | §1 transition rows, conditional meaning, generation and time assumptions |
| A.2 HMMs and their questions | §§1–2 joint factorization, probability queries, three primary problems plus online timing |
| A.3 Forward | §3 full four-step arithmetic, filtering/forecast and dynamic-program state |
| A.4 Viterbi | §5 exact recurrence, predecessor proof, backtracking and complete path probability |
| A.5 Forward–backward/EM | §§4/6 full backward, gamma/xi, independent-sequence sufficient counts and one update |
| A.6 Summary/history | Explicit next graphical/conditional links and annotated historical source; no copied historical performance claims |

Rabiner's additional section-list audit is resolved: continuous/autoregressive models, null/tied topology and explicit duration appear in §§9–10; scaling/multiple sequences/initialization/data sufficiency/model choice are §§6–9. Historical recognition systems and duration applications are summarized in §10. MMI/MDI alternatives are not silently equated with likelihood EM: the lesson distinguishes joint versus conditional objectives and points to the immediate CRF owner. Detailed historical LPC/quantization/recognition experiments remain optional source study, not prerequisite material or a claim of current end-to-end speech coverage.

Additional primary reading: hmmlearn0.3.3 base.py actual decoder/monitor implementation and relevant categorical/Gaussian API declarations; official count-vector example; EWT r2.16 README summary/license/citation/version metadata; Eisner author bundle and ACL2002 metadata/abstract (not video/spreadsheet execution); HMMER project/profile-model guide description (historical model topology only); Ghahramani/Jordan1997 author-hosted abstract for factorial hidden states and exact-inference difficulty (not full paper). No unverified paper URL retained.

## Author evidence and learning-experience checklist

- Full prerequisite-to-mastery prose, ten changed problems with20 closed hint/solution disclosures, complete NumPy program, optional complete native program and six annotated alternative resource entries. First-pass route distinguishes core inference/one update/real comparison from optional advanced models.
- At least three genuinely different investigations: edit future evidence and construct a posterior contrast; choose/repair a path in a constrained graph; reconstruct sequence boundaries and fractional event counts. Each starts unsolved, uses actual calculations, has an executed contrast/null, and records inputs/reset/answer invalidation. Real labeled text adds a separate inspection task. Eight visual homes adapt to the concept rather than one uniform shell.
- All probability quantities name joint, marginal, conditional or density semantics. Toy zero-based indexing is consistent after repair. Viterbi's third-column Rainy predecessor, MAP path illegality, missing-step versus deletion and EM objective direction were explicitly checked.
- Author program executed:16path enumeration/parity, pair margins, changed-future filter null, log/scaled rare-event agreement, true impossible event, independent counts/one M-step, three40-step starts plus symmetric3-step start, supervised real train/dev counts and four decoder configurations. Extra duplicate-dataset and length-one probes ran without repeating real fitting; their assertions are now preserved in the program.
- Data identity/source hashes,200distinct sentenceIDs, role counts and retained source attribution checked. Real HMM improves ten/breaks six matching lexical decisions; full-sentence and token counts are not conflated.140unknown development tokens explain a real limitation. No reserve scoring, fake native results, latency chart or “all models improve” outcome.
- Two Python files AST-parsed. Native hmmlearn dependency is absent: that displayed optional program remains explicitly unexecuted, with actual execution/parity required before publishing it.
- Full author reading checked mechanism-first flow, once-stated substantive cautions, meaningful figures, exact numeric examples, independent practice, alternate-learning annotations, honest empirical limits and canonical coverage. This is author content work; formal independent review, runtime code and browser/build verification are deferred.

Next action is an explicitly authorized finish request using the complete current content checkpoint. Implement the specified topic-owned visual forms and labs, execute optional native examples, independently review correctness and learning experience, check browser/accessibility/performance and integrate. Preserve original stable ID, module order and the unchanged original JSX until that phase.

## Phase two, part A: implementation and numerical verification (16 September 2026)

Implementation of the frozen content packet. Manuscript, visual specifications, `calculated-inputs.json`,
provenance and data were consumed unchanged; this section is the only edit to a content-checkpoint file.
Browser verification is not in this part: the blueprint is not registered yet, and evidence is captured against
a registered page in part C.

### Source conservation

The published body this replaces was preserved to `scratch/hmm-phase-two/original-hidden-markov-models-hmm.jsx`
before anything was written. Its SHA-256 is `9f99c9c7edab75df56ec98b2eb628a0cb97a19beeabb7e4e47a7d7e22200614c`,
63,607 bytes, byte-identical to `git show HEAD:src/learn/data/topics/hidden-markov-models-hmm.jsx` and to the
original hash this design record already carried. A copy of the git-HEAD version is retained beside it.

### What was built

| File | What it holds |
| --- | --- |
| `src/learn/data/hmm-models.js` | The whole verified model layer, 1,076 lines. Inference by retained scaling, the raw trellis under both operators, enumeration, expected counts, one EM step, duration, numerical-scale facts, the quantities the figures draw, and the rules the investigations grade with. |
| `src/learn/data/hmm-data.js` | Generated. Provenance, the 146-symbol vocabulary, 40 development sentences, two fitted models serving four decoder configurations with per-token beliefs, the decision changes, the exact-arithmetic tie audit and the EM track. |
| `src/learn/data/hmm-examples.js` | Generated. Both displayed programs, extracted verbatim and executed. |
| `src/learn/components/lesson-labs/HmmShared.jsx` | Controls, the prediction and construction contracts, the probability-row editor, and the shared drawings: trellis, state graph, belief bars. |
| `src/learn/components/lesson-labs/HmmFigures.jsx` | Nine inline figures. |
| `src/learn/components/lesson-labs/HmmLabs.jsx` | Six investigations. |
| `src/learn/components/lesson-labs/hmm-labs.css` | Lesson styling, with the SVG type rule scoped at the lesson root in longhands. |
| `src/learn/data/topics/hidden-markov-models-hmm.jsx` | The lesson body: thirteen sections, ten practice problems, the readiness table and the references. |
| `src/learn/data/curriculum/blueprints/hidden-markov-models-hmm.js` | The authored plan. **Not registered**; registration is the integration owner's. |
| `public/learn-assets/hmm/` | The lesson's own copies of `ewt-sequences.json`, both programs and `ATTRIBUTION.txt`. |
| `scripts/verify-hmm-{models.mjs,examples.py,data.py}` | Three verifiers, each writing evidence to `docs/teaching/evidence/`. |

Six investigations, one per conceptual hurdle rather than to a quota: one complete path's product (§1), evidence
arriving later with separate filtering and smoothing commitments (§4), repairing a path in a constrained graph
(§5), moving a recording boundary (§6), the real tagging comparison (§8) and dwell time (§9). Nine inline
figures: the graph and its unrolling, the forward trellis, the two beliefs, the Viterbi trellis with its path
share, the pointwise-versus-legal graph, the fractional count flow, the EM objective on two windows, the
numerical-scale tables and the profile/factorial topologies.

### Verification actually run

| Verifier | Result |
| --- | --- |
| `verify-hmm-models.mjs` | PASS. 518 grouped checks across 66 groups. |
| `verify-hmm-examples.py` | PASS. 2 of 2 displayed programs executed, 65 oracle assertions, 13,243 recorded leaves reproduced by a fresh run of the complete program. |
| `verify-hmm-data.py` | PASS. 2,665 checks across 481 groups; module regeneration byte-identical. |
| Escape audit | CLEAN over 13 files including all three verifiers, and 122 displayed LaTeX expressions. |
| Render smoke test | All 16 components and the whole lesson render server-side without error. |
| Falsification harness | 19 of 19 breakages caught, all 19 citing the guard they were aimed at. |

**Trust-root coverage: 13,246 of 13,247 scalar leaves of `calculated-inputs.json` re-derived, 99.99%,
measured as asserted leaf paths rather than a counter.** The single uncovered leaf is
`additional_author_checks.native_example_ast_parsed_not_executed`, which records the state of the author's own
process rather than a calculation, and which phase two supersedes by executing that program for real.

Independence is real rather than nominal. The browser model infers by **retained scaling** while the packet's
author program works in **log space** throughout; `verify-hmm-data.py` re-derives the packet through a third
implementation that imports nothing from `hmm-experiments.py`; the sixteen-path enumeration and a brute-force
search over legal paths supply routes that use no recurrence at all; and section 8's smoothing is applied to
integer counts through the manuscript's formulas rather than by accumulating onto pre-seeded float arrays.

Every rule an investigation grades with lives in the verified model layer, not inside a component, and is
exercised over the **whole grid its control can reach**: all sixteen paths of five models including one where
every path ties and one where every path is impossible; all 256 four-step recordings at every query time, with
filtering's independence of everything after the queried time asserted by exact equality; all 231 priors on a
twentieths grid of the three-simplex; every composition of up to eight reports into up to four recordings under
three symbol patterns; every single-report replacement; and every self-transition probability on a hundredths
grid against elapsed times up to twenty.

### What was done about `hmmlearn`

It was **executed for real**, not displayed as unexecuted. `hmmlearn` was not installed into the shared lesson
runtime; an isolated environment was created at `scratch/hmm-optional/` and resolved to **hmmlearn 0.3.3, NumPy
2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 on Python 3.12.14**. The shared runtime's NumPy 2.3.5 is therefore
untouched and no other lesson's recorded output can have moved. The program's stdout is deterministic across
repeated runs and is displayed on the page as real output, with the environment named beside it.

Its numbers are checked against oracles derived from the model rather than from its own output: the observation
log probability against log 0.00933936, the Viterbi score against the log of the best path's joint mass, all
eight smoothed marginals against this lesson's own posterior, and the independent-sequence fit against
2 log 0.5. The manuscript's version-specific warning is now executed rather than asserted — the MAP decoder
returns 2.940021586061572, which is matched against the sum of the four smoothed maxima and separately shown to
be positive, which no log probability of a probability can be. The warning the library prints to stderr is
checked too: its count of 7 free scalar parameters is exactly what section 9's own formula gives for two states
and three symbols.

### Departures from the packet, each with its reason

1. **The complete NumPy program is displayed as four contiguous excerpts plus its run, not as one block.** The
   manuscript links `hmm-experiments.py` as a file; inlining 356 lines would be four times the longest displayed
   program in any completed lesson and would bury the mechanism. Each excerpt is a contiguous slice of the frozen
   file taken through Python's AST with its line numbers recorded, the complete file is served for download at
   `/learn-assets/hmm/hmm-experiments.py`, and the whole program is executed. The verifier asserts that each
   excerpt is a literal substring of the frozen file, that its recorded hash matches, and that the four excerpts
   together cover exactly the functions they name and no others.
2. **Programs are extracted verbatim from the packet's frozen `.py` files, not from manuscript code fences.**
   The manuscript contains exactly one fenced block, a bash run command, which is extracted and pinned by the
   manuscript hash. The Python lives in packet files, so those are pinned by SHA-256 instead — the same
   guarantee by the only available route.
3. **A thirteenth section, "Readiness and the next lesson", was added.** The manuscript's readiness sentence sits
   after its references; the reader route in this project puts a readiness table and the next-lesson link before
   them, as the preceding lessons do.
4. **Two size envelopes rather than one.** The visual contract bounds toy exploration at 12 steps, 4 states and 4
   symbols; the recorded tagger needs 146 symbols and 15-token sentences. Passing the wrong envelope is a
   RangeError rather than a silent widening.
5. **The two fitted models are stored once and referenced by index**, rather than once per configuration. Two
   decoders read the same fitted counts, and this makes that true in the data as well as the prose; it also
   halves the emission payload.
6. **The EM objective is plotted on two windows.** One window cannot hold an initial score near −566 and also
   separate final scores 0.4 apart: at the overview's range the closest pair is 0.20 px apart, and at the detail
   window's range it is 5.57 px. Both numbers are asserted, so "the curves differ" is a measured claim.
7. **`verify-hmm-data.py` reproduces the declared program's log-space `argmax` for the tagger's Viterbi only**,
   while everything else in that file is independent. See the tie finding below: at three sentences the decoded
   path is settled by arithmetic, and reproducing which member the packet recorded requires reproducing its
   arithmetic. The independent guarantee is supplied instead by exact rational arithmetic, which proves that
   whatever is reported is a true maximiser.

### Disagreements with the packet, recorded rather than fixed

1. **Three development sentences have two complete paths of exactly equal probability.** Established in exact
   rational arithmetic, not floating point. Each contains adjacent tokens that both map to the unknown symbol, so
   exchanging their two states permutes the same multiset of factors. Every path the packet stores is verified to
   be an exact maximiser, so the packet is **correct**; but the recorded token counts are one member of a band.
   At smoothing 1.0 the same fitted model reports 269, 270 or 272 depending on the tie rule, and at smoothing 0.1
   it reports 267, 268 or 270. The packet's 270 and 268 sit inside those bands. The selection of the winning
   configuration and the sign of the HMM's gain over the lexical baseline survive every tie rule; the exact
   four-token margin the manuscript states does not. The lesson now teaches this rather than hiding it, the tie
   audit is served in `hmm-data.js`, and the models verifier pins all six band members.
2. **A one-unit-in-the-last-place difference in the emission table changes which member of a tie is reported.**
   Computing the smoothed counts from integer counts through the manuscript's formulas, rather than by
   accumulating onto pre-seeded float arrays as the author program does, differs by at most 1.1e-16 — and that
   one bit flips development sentences 1 and 2 at smoothing 0.1, moving the recorded count from 268 to 267. Both
   routes are computed and asserted to agree to 4e-16; the published model uses the declared program's
   accumulation order, because that is what makes the published table reproducible.
3. **`hmm-experiments.py` does not write `additional_author_checks`.** A fresh run reproduces all 13,243 leaves
   the packet shares with it, exactly, and writes no `additional_author_checks` block. Those four booleans were
   appended by the author. Three of them are re-derived here from the declared settings; the fourth is a
   statement about the author's process and is superseded by the execution above.
4. **The manuscript states that all three full EM trajectories are retained, including the model after each
   update.** `calculated-inputs.json` retains the 41-point objective history and the final parameters only. Phase
   two re-derives the initial and first-update parameters from the declared seeds and serves three named
   checkpoints per seed, which is what a page can display usefully.
5. **The manuscript's paragraph on the EM histories was written here as "all three fits exceed the generating
   score", not "a fitted model can exceed it".** All three do: −390.253507, −390.630410 and −392.284393 against
   −393.562196. The symmetric start stays below it, so the contrast is not vacuous.

### Defects found and fixed during implementation

Four were found by the render smoke test and the falsification harness rather than by any assertion written in
advance, which is the argument for having both:

- Rounding the 146-column emission rows to nine decimal places accumulated about 4e-9 of error and failed the
  browser model's own row-sum guard on first paint. Probability rows are now stored at full precision, and the
  data verifier now asserts that every serialised probability row is still a distribution.
- `infer` dropped its size envelope when delegating to `forwardScaled`, `trellis` and `pathJointOf`, so the
  146-symbol tagger was validated against the four-symbol toy bound and threw on first paint. No assertion about
  a returned value could have caught this.
- The drafted trellis and the duration strip displayed graded quantities before the prediction was committed.
  Both now hide them until commitment; the trellis keeps its structure and loses only its cell values.
- Four of the verifier's own guards could not fail: the path-rank order had no fixture where every path is
  impossible, the row-sum tolerance could be loosened a hundred-million-fold without any refused case noticing,
  the tie band could be narrowed freely, and the unvisited-row guard was masked by an earlier assertion. All four
  are closed, and the harness now trips each of them.

### Remaining for part C, and for the reviewer

Browser verification against a production build at 1366, 390 and 320 px, with screenshots opened and looked at —
particularly the trellis at 320 px, the two EM panels, the count-flow bars and the token chips. KaTeX display
width is unmeasured on a real page; 122 expressions were audited for structure and for `\widehat`, which none
uses, but no rendered width has been measured. `scripts/verify-hmm-browser.cjs` is not written yet and will
carry `sampleCurvesThroughLabels` copied from `verify-bias-variance-browser.cjs`. The blueprint is unregistered
by design. No build directory was created and no preview was started, so there is nothing to tear down.

## Phase two, part C: browser review against a production build (16 September 2026)

The blueprint is registered and the lesson builds into the production bundle, so this part captures evidence
against the page a reader actually meets. `scripts/verify-hmm-browser.cjs` runs at 1366, 900, 390 and 320 px
against `npx vite preview --outDir dist-hmm --port 4190` and writes `docs/teaching/evidence/hmm-browser.json`.

**Result: passed, 21 cases, 55 screenshots**, each with its own path, its own content hash and a recorded
`{file, digest, bytes}` triple; 8.6 MB in total. Every capture was opened and looked at. The three phase-A
verifiers, the escape audit, the render smoke test and the falsification harness were all re-run after the last
change and remain green: 518 model checks, 2,665 data checks at 99.99 % trust-root coverage, 65 native oracles,
19 of 19 breakages caught.

### The clarification the coordinator asked for, restated precisely

On the exact-tie finding, the distinction is this, and none of the three parts implies another:

- **The packet is correct.** Every path it stores is proved, in exact rational arithmetic, to attain the exact
  optimal joint probability. Nothing recorded is a mistake.
- **The margin is fragile.** Three of the forty development sentences have two paths of exactly equal
  probability, so the reported token count is one member of a band: 269, 270 or 272 at smoothing 1.0 and 267,
  268 or 270 at smoothing 0.1, selected by a tie rule nobody declared. A one-unit-in-the-last-place change in a
  single emission entry moves it. The four-token margin the manuscript states therefore is not a property of the
  model alone.
- **The conclusion is robust.** Under every admissible tie rule the HMM beats its matching lexical baseline, and
  smoothing 1.0 beats 0.1, so the selected configuration and the sign of the gain do not depend on the tie.

### What the browser verifier checks that the others cannot

Visible content and the served assets with their hashes; all six investigations driven through their declared
contrasts, their nulls and both branches of every construction grader; figure geometry read back from the paint
rather than the DOM; label collisions including curve-shaped ones, sampled along each curve's own geometry
because the shared inspector reads straight lines only; every SVG label's size in **rendered pixels**, measured
as computed font size times the element-to-viewBox scale; the two objective windows measured as a rendered
separation on screen; horizontal table overflow at desktop in both the fixture state and the committed state;
KaTeX display width at 390 and 320; keyboard reach; the loading closure; completion, sequence and recovery.

Two numbers worth recording. Rendered label sizes stay between **9.81 px and 14.82 px** at all three widths, and
the reading genuinely differs between them, which is what distinguishes measuring pixels from measuring user
units. The objective figure's two windows separate their four final scores by **0.27 px** and **7.45 px**
respectively — the claim that one window cannot resolve what the other can, measured in the image rather than
asserted from the model.

### Defects found by looking, and fixed

Every one of these passed the model, data and native verifiers.

1. **The edge-width encoding was invisible.** `.hmm-edge { stroke-width: 1 }` outranked the presentation
   attribute the component computed, so all twelve trellis edges rendered at 1 px while carrying the correct
   width in the DOM — the exact encoding figure 2's caption promises. The stylesheet no longer sets a width on a
   quantitative edge, and a new guard compares every `stroke-width` attribute on the page with its painted value.
2. **The emission connector ran through the bottom row's cell values.** Every Sunny-row value in three trellises
   was struck through by the dashed connector. Values moved inside their nodes and the connector became a
   column-level bus, which also stopped it reading as "only the bottom state emits".
3. **The state graph's two cross arcs were unlabelled by direction.** With no arrowheads and both arcs joining
   the same pair of nodes, a bare number on each could not say which was Rainy to Sunny. Each label now names its
   own direction.
4. **The factorial diagram's upper chain sent its connector through the lower chain's node**, which read as the
   two chains being wired to each other. It now bows around it.
5. **A self-loop arc travelled along a third of its own label.** Only the curve sampler could see this; the
   straight-line inspector reads `<line>` elements. The labels moved above the arcs' apex.
6. **Three tables hid a comparison column at desktop width** behind an overlay scrollbar, including the
   original-versus-changed table whose whole purpose is that comparison. Column headers now wrap, two panel rows
   stack instead of pairing, and two over-long cells moved their prose into footnotes.
7. **A table cell was cut mid-entry in the committed state** of the boundary investigation, which the
   fixture-state sweep could not see. The long list moved to the last column, which is the one allowed to wrap,
   and the verifier now sweeps both states.
8. **Labels rendered at 8.35 px in the topology figure and 6.9 px in a trellis on a phone.** Inflating the font
   fixed the size and broke the layout, because the coordinates around the text did not grow with it. Each
   drawing now sits in a frame that caps it on a wide screen and scrolls at its natural size on a phone, with a
   visible affordance, and a four-column trellis shows state initials there while the exact values are read from
   the table beside it.
9. **The document scrolled sideways by 141 px at 320 px**, caused by state strips whose long applied-state
   strings could not break.
10. **A correctness defect in prose**: figure 5 said A→A has probability "undefined" and then that its joint
    probability is exactly 0. Those are different things and the lesson exists to distinguish them; it now says
    exactly 0 and explains why that is not an undefined quantity.
11. **A factual error in a figure heading**: "Only three transitions exist" over a graph with four. Both
    verifiers now pin that count to the model.
12. **The identification task's inputs sat below its own submit button**, so a learner met the action before the
    fields it grades. They moved above it. Its two defaults also pre-answered half the task and were changed to a
    position that is neither a repair nor a break.
13. Smaller: a stale caption after the values moved inside the nodes; the reference line and one data track
    sharing a dash pattern; a caption promising three settings over a five-row table; a literal `2^1000`; a
    missing space after a bold caption lead; a scroll hint that claimed things untrue of a schematic topology;
    and a caption quoting "under a quarter of a pixel" where the rendered separation measures 0.27.

One capture problem was a harness defect rather than a page defect: the site navigation is `position: fixed` and
painted across narrow-width screenshots. Captures now hide it for the shot, which changes no layout because a
fixed element occupies no flow space.

### Known and accepted

At 900 px — between the desktop layout and the 560 px stacking breakpoint — one table scrolls horizontally by
42 px inside its own stable-gutter scroll box. It is recorded as an inspected intermediate width rather than
fixed, because the box is the intended behaviour there and the table stacks completely below 560 px.

The uniform-start track in the objective figure's detail window spans only three updates, so it is a short
segment near the left edge. That is what the recorded run contains; the legend names it and the table gives its
values.

`dist-hmm` was deleted and the preview stopped at the end of this part.

## Phase two, part D: disposition of the independent review (16 September 2026)

`docs/teaching/HMM-INDEPENDENT-REVIEW.md` re-derived this lesson's mechanism by a third numerical route — exact
rational arithmetic, against the browser's retained scaling and the packet program's log space — and found no
disagreement in any displayed number. It raised one blocking finding, nine should-fix findings and seven
observations. Every one of them is dispositioned below, and every repaired guard was broken on purpose and
watched go red.

**Final state: all four verifiers green after the last change.**

| Verifier | Result |
| --- | --- |
| `verify-hmm-models.mjs` | PASS. **18,328 executed assertions** in 68 recorded groups, including 1,024 belief-move verdicts over all 256 four-step recordings, 231 priors, 486 recording compositions and 1,515 distinct hazard settings. |
| `verify-hmm-data.py` | PASS. 2,668 data checks across 481 groups; 13,246 of 13,247 trust-root scalar leaves re-derived (99.99 %); module regeneration byte-identical. |
| `verify-hmm-examples.py` | PASS. 2 of 2 displayed programs executed, 66 oracle assertions, 13,243 recorded leaves reproduced by a fresh run. |
| `verify-hmm-browser.cjs` | PASS. 21 cases, **56 screenshots**, each with its own path, its own digest and a recorded `{file, digest, bytes}` triple. |
| Escape audit | CLEAN over **15** files — now including `verify-hmm-browser.cjs` and the falsification harness itself — and 122 displayed LaTeX expressions. |
| Falsification harness | **28 of 28 executed breakages caught, all 28 citing the guard they were aimed at**, across five runners: models, data, examples, server render and **browser**. One further breakage is applied and restored without being executed, and is reported as recorded rather than caught. |

The escape audit's file list is itself the reason one defect survived a whole phase: `verify-hmm-browser.cjs` had
never been in it, so a regex written through a heredoc whose `\b` became a literal backspace byte was reported
CLEAN. The audit now fails if any owned verifier is missing from its own list.

### Finding by finding

| Finding | What was wrong | Resolution |
| --- | --- | --- |
| **B1** (blocking) | The opening paragraph attributed the exact-tie finding to "three of the forty **test** sentences". The computation used the **development** split, and the page says three separate times that the reserved forty are never scored — so the first paragraph asserted a result on the held-out data the rest of the lesson promises never to touch. | The sentence now reads its count and its split size out of `tieAudit` and `provenance`, and states that the reserved sentences are not decoded anywhere in this lesson. A new browser guard reads the **rendered** introduction and requires that it name the development split with the size the data records, that it contain none of "test sentences have", "reserved sentences have" or "held-out sentences have", that every tied sentence be one of the development sentences the page serves, and that `provenance.reservedScored` be false. |
| **S1** | The six quantities in the section-2 question table were plain strings, so `P(z_t \| o_{0:t})` reached the reader as literal braces and underscores — and the subscript is exactly the distinction sections 3 and 4 are about. | Every quantity cell is a `<Math>`. Looking at the rendered table then showed KaTeX breaking `arg max_z P(z \| o)` across two lines at the relation, so that expression is now a single group, which removes the top-level break opportunity. No table's fit changed: at 900 px exactly one table still scrolls, still by 42 px. |
| **S2** | The Clean-forecast sentence opened with a transition entry times an emission entry — the very multiplication the surrounding paragraph warns against. | Rewritten to start from the predicted state row: each state's Clean probability weighted by how likely that state now is, 0.46(0.5) + 0.54(0.1) = 0.284. Read back in the rendered page. |
| **S3** | The page told the learner the served program is 332 lines; it is 331. The count was newline characters plus one, which is right only for a file with no trailing newline. | `len(source.splitlines())`. |
| **S4** | A trellis edge carrying **nothing** painted at 1.095 px, wider than one carrying almost nothing at 0.985 px: `edgeWidth(0)` returned `null`, the component emitted no attribute, and the line took the SVG initial 1 px. This lesson's own zero-versus-tiny distinction, inverted on screen. The override guard could not see it, because it compares attributes with painted values and an element with no attribute was not a failing case but an unexamined one. | `edgeWidth` is now total and monotone non-decreasing across its whole domain, returning a declared zero width of 0.5 below the positive minimum of 0.9. `trellisEdges` separates an edge the model **forbids** from a permitted edge that **carries nothing**, and the second is drawn thinner, dimmer and dashed in its own style. Every drawn edge, forbidden ones included, now sets an explicit `stroke-width`, and the guard treats a missing attribute as a failure rather than a skip. Because no edge carries zero mass on the page as first painted — which is why nothing caught this — the browser verifier now drives the third investigation's free-exploration panel to a Rainy emission row of `0, .5, .5` and asserts, in that reachable state, that every edge sets a width, that at least two edges carry nothing and at least two carry mass, and that the widest empty edge is thinner than the thinnest carrying one. |
| **S5** | Figure 7's caption said "the four final scores land less than a pixel apart" where they span 3.36 rendered px; only the closest pair is under a pixel. | The caption now says the closest pair lands under half a pixel apart and all four span barely three, and that the same closest pair separates by several pixels on the detail window. Checked against the rendered figure, not against the numbers. |
| **S6** | Three trust-root leaves were marked re-derived while nothing read their values; flipping one to false in the packet left the run green at 99.99 %. | The three booleans are asserted before `cover.mark` is reached. |
| **S7** | Four Phase A record claims contradict the tree. | `design.md` is append-only, so they are corrected here, and this paragraph supersedes the Phase A table where they conflict: `hmm-models.js` is **1,136** lines, not 1,076; inlining `hmm-experiments.py` would be **331** lines, not 356; the blueprint **is registered**, by the integration owner, at `blueprints/index.js:26`; and `scripts/verify-hmm-browser.cjs` **exists and runs**. |
| **S8** | The same two window separations are recorded as "px" with two different values. | Corrected here: Phase A's **0.20 and 5.57 are viewBox user units**, computed by `separationInPixels` against `objectiveWindows`; Phase C's **0.27 and 7.45 are rendered CSS pixels**, at the 1.338 element-to-viewBox scale that relates them. Both are right in their own frame; the unit labels were not, and the difference between a number the model asserts and a number measured in the image is the whole point of part C. |
| **S9** | A fifth, sixth and seventh guard that cannot fail, plus one block of dead code: (a) two of the three per-edge assertions in the ninety-edge sweep compared the same IEEE multiply of the same stored operands with itself, and `forbidden` was re-asserted against its own definition; (b) 606 of the advertised 2,121 hazard settings compared a function with itself, because the inner loop began at `late = early`; (c) several construction verdicts were re-read with the formula that produced them; (d) `readTrellisGeometry`'s second call was assigned and never referenced. | (a) The sweep's expectations now come from `recomputeColumns`, an independent forward/max recursion written inside the verifier, and from the model's own matrices — cell values, destination aggregates, source masses and transitions are all compared against arithmetic this file performs, never against the object under test. (b) The hazard grid pairs **distinct** elapsed times only, 101 × 15 = 1,515, and each is additionally compared against 1 − a computed in the verifier. (c) Construction verdicts are compared against `pathLegal` and `pathJointOf` recomputed from the model. (d) The discarded readback is gone. The headline number is now **executed assertions**, incremented inside `ok`, `equals` and `close` themselves, so it cannot drift from what ran. |
| **O1** | "518 model checks" was a marker count, not an assertion count. | Replaced by 18,328 executed assertions, counted by the helpers. The three other headline numbers the reviewer checked were already genuine. |
| **O2** | Nothing falsified the browser layer — including the guard written to catch the browser-layer defect — and two sweeps asserted `.every()` over sets that could be empty. | The harness has a **browser runner**: it rebuilds the production bundle over the running preview, which serves from disk per request, and runs the real twenty-one-case verifier, not a reduced copy. Three browser breakages are in it. Every sweep now has a non-empty floor stating how many subjects it had. |
| **O3** | The edge-width readback compared sorted multisets, so any permutation of widths across the twelve edges passed. | Each drawn edge is paired, in DOM order, with the model edge at the same position; both its attribute and its painted value are compared, and the twelve widths must be twelve distinct values. |
| **O4** | Loose floors where an exact count was available. | Exact counts: 71 trellis nodes and cards, 12 distinct widths, exactly one table scrolling at 900 px and by 42 px. One floor was worse than loose — the curve sweep demanded 20 curves on a page that draws 15 — which running it immediately exposed; it is now the measured 15 and 150. |
| **O5** | Smoothing rendered as "1" in one sentence and "1.0" everywhere else. | `.toFixed(1)` in that sentence. |
| **O6** | `design.md` calls the 2-simplex a "three-simplex". | Corrected here: the 231 priors lie on the **probability simplex on three states**, which is the 2-simplex. The verifier's own wording was corrected in place. |
| **O7** | Where the lesson is better than its brief. | Recorded with thanks; no change. |

### Evidence that each repaired guard now fails when it should

Every row below was produced by applying the breakage to a real source file, running the verifier that is
supposed to catch it, and restoring the file under a SHA-256 check.

| Repaired guard | Breakage applied | What the verifier said |
| --- | --- | --- |
| S4, the zero width (models) | `edgeWidth(0)` returns the widest stroke | `a zero share takes the declared zero width: 4.4 versus 0.5` |
| S4, one notch off (models) | `edgeWidth(0)` returns the smallest **positive** width | `a zero share takes the declared zero width: 0.9 versus 0.5` |
| S4, the empty-edge case (models) | `carriesNothing: false` | `edges that exist and carry nothing: the subject set is empty, so nothing was checked` |
| S9(a), the per-edge sweep (models) | a trellis cell multiplied by the wrong state's emission, leaving every stored edge field mutually consistent | `forward column 1[0]: 0.0414 versus 0.0552`. The verifier aborts at its first failure and a stored-column comparison reaches this defect earlier, so the citation is that guard; re-run with assertions collected instead of thrown, the breakage trips 160 assertions, **42 of them the sweep's own** `matches the independent recursion in sum mode`. |
| S9(b), the hazard grid (models) | a hazard that differs between elapsed times but agrees with itself, which the old `late = early` diagonal could not see | `the hazard at a = 0 is the same after 0 and 4 steps` |
| S6, the packet booleans (data) | the expectation flipped to `is False`, the only way to reach a value that lives in the frozen packet | `trust root - the packet records duplicated_dataset_parameter_null as true` |
| S3, the displayed line count (examples) | the repaired expression shifted by one | `src/learn/data/hmm-examples.js is stale or missing` |
| The attribute-versus-paint guard (browser) | `.hmm-edge { stroke-width: 1 }` in the stylesheet, which is the original defect exactly | `an edge carrying nothing paints at 1px, and the thinnest edge carrying anything at 1px` |
| S4 in the paint (browser) | `edgeWidth(0)` returns `null` again, so a zero-mass edge takes the SVG initial 1 px | `every edge in the reachable zero-mass state sets a stroke-width attribute (hmm-edge is-empty does not)` |
| B1, the intro and the split (browser) | the introduction names the test split | `the introduction names the development split and its size` |
| O4, the node count | the exact 71 relaxed back to a floor of 30 | Applied and restored, **not executed**: re-running the twenty-one-case suite for it buys nothing the three browser breakages above do not already buy. It is reported as recorded, not as caught. |

This is also the answer to why the harness matters more than its pass rate. Before this part it was 19 of 19,
and two of the defects the reviewer found — a stylesheet beating a presentation attribute, and a claim in prose
disagreeing with the computation — were in layers it could not reach at all. The first browser run of the new
breakages found a third: `edgeWidth(0)` returning `null` **survived the whole browser suite**, because no edge
carries zero mass on the page as first painted. That survivor is what produced the reachable-state check, and it
is the clearest evidence in this record that a guard's domain matters as much as its condition.

### Observations acted on, and what was left

All seven observations were acted on except O7, which records where the lesson exceeds its brief and asks for
nothing. O1, O3, O4 and O5 were straightforward repairs to counting, pairing, exactness and formatting. O2 was
not: it required a new runner, and the run it enabled found a defect that had survived everything else. O6 and
the two record findings, S7 and S8, cannot be repaired where they were written, because this file is appended to
and never edited; they are corrected in the table above, and a reader who finds the Phase A table asserting 1,076
lines, 356 lines, an unregistered blueprint or an unwritten browser verifier should take this section as
superseding it.

Two things were deliberately not changed. The 900 px table that scrolls horizontally by 42 px inside its own
scroll box stays as it is: the reviewer examined it (A13) and accepted the judgement, the box is the intended
behaviour at that width, and the table stacks completely below 560 px. And the tie finding itself is unchanged in
substance, because the review confirmed all three of its parts — the packet is correct, the margin is fragile,
the conclusion is robust. What did change is the reason given for the third part. The reviewer supplied a
stronger argument than the one written: the **same three sentences are tied at both smoothing strengths**, with
the same two candidate paths and the same correct-token counts, so any one consistent tie rule contributes the
same amount to both totals and smoothing 1.0 beats 0.1 by exactly 2 under every rule. That is a statement about
the rule cancelling, not about two bands failing to overlap, and it is now what the page says.

`dist-hmm` was deleted and the preview stopped at the end of this part.
