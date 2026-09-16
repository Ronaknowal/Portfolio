# Hidden Markov Models — independent phase-two review

Reviewed 16 September 2026 against the working tree plus the uncommitted HMM implementation. The reviewer authored
neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or
`docs/teaching/drafts/` was edited; this review document is the only write outside `scratch/`. Throwaway scripts live
under `scratch/hmm-independent/`. The four verifiers were re-run, which rewrites `docs/teaching/evidence/hmm-*.json`;
those files were copied to `scratch/hmm-independent/evidence-backup/` first and the regenerated ones are byte-identical
to the committed ones apart from their timestamp field (checked, see A12). `dist-hmm` was built, driven and deleted;
the preview was stopped. No state-changing git command was run.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/hmm-independent/`, importing neither `hmm-models.js` nor `hmm-data.js` nor `hmm-experiments.py` nor any `verify-hmm-*` script. A hand-written exact-rational HMM engine (`exact.py`: `fractions.Fraction`, no scaling, no logarithms, no NumPy) supplying a **third** inference route independent of both the builder's retained scaling and the packet's log space; a 60-digit `mpmath` re-derivation of the EM trajectories; a 146-symbol exact-rational Viterbi with full tie-set enumeration over the forty development sentences. Separately: the optional program executed in the isolated venv, all four verifiers and the falsification harness re-run, and my own Playwright sweep (Edge, 1366/900/390/320 px) comparing every SVG presentation attribute against its painted value. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `data-sources.json`); the lesson body, model layer, generated data and examples modules, all six investigations, the shared scaffolding, the figures module and the CSS; all four verifiers; the falsification harness; `LESSON-TEACHING-STANDARD.md`; the served assets and their attribution. |
| **Reused (declared)** | NumPy/mpmath as *general* libraries, not lesson code; `numpy.random.default_rng` to regenerate the seeded sample, because the seeded stream **is** the data and a different stream would be different data, not a different route. Two subagents for fan-out: one reading `design.md` + the teaching standard, one statically auditing the four verifiers. Every conclusion below that rests on their output I re-derived myself — I re-read and quoted each cited line of `verify-hmm-models.mjs`, `verify-hmm-data.py` and `verify-hmm-browser.cjs`, I opened the images, and I drove the page myself. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/hidden-markov-models-hmm.jsx` | `31ffa1b38d03a0726d119bde4b4cb745b34077a9743201a173d4e490236a0f7c` |
| `src/learn/data/hmm-models.js` | `38db9f7bee727c5c064a39b7f8c56b3a37977f3b4404d905a40146cf93abaf77` |
| `src/learn/data/hmm-data.js` | `8af3a3bcd2cfcdc280102ded1fc2dc3d24c3da60cf7079e0e65bf49dae3b837b` |
| `src/learn/data/hmm-examples.js` | `962b0913f0a4d9caf18a0bbe2427df91a6f4ffe5729ab30444394468e3862868` |
| `src/learn/components/lesson-labs/HmmShared.jsx` | `e70ae260fa6bc742a93846e7ef890d5ab1a1221b9545aee12c8d555376d48851` |
| `src/learn/components/lesson-labs/HmmLabs.jsx` | `6d397e81562872b5cdd384ba7405526b11cf2046150ecdefa926719ca3b67e6d` |
| `src/learn/components/lesson-labs/HmmFigures.jsx` | `229de399ab82f7343082cab0266d9bca0111b2e3e21101f345fbf767f5960e03` |
| `src/learn/components/lesson-labs/hmm-labs.css` | `25ebf4575ee77dcaa4482393072710cdd0ca4d7b6767bfa495e99248bd7a1401` |
| `src/learn/data/curriculum/blueprints/hidden-markov-models-hmm.js` | `405c2ae4424e47b94fca1ee404535aef4b4c142297954d8116f5e390f479e839` |
| `docs/teaching/drafts/.../lesson.md` | `d79e5e4a20eb6aea4247cdabe805ce1c018d3849cdf9ef19953119fbfe5a28fb` |
| `docs/teaching/drafts/.../design.md` | `53fc6098f288234da14b2aee15732268a42166c5cc854ce3a2d1a8cbcf259fc0` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `d05a1926402b07a36bb994ac9c0614df417c4bccfc35c8902b8438fec61268f9` |

---

# Part A — correctness

## A1. Exact recomputation of the whole mechanism — **no disagreement**

`scratch/hmm-independent/exact.py`, `mech.py`, `mech2.py`, `extra.py`. Pure sum-product and max-product over
`fractions.Fraction`: no scaling, no logarithms, no tolerance anywhere a rational answer exists. This is a third route —
the browser infers by retained scaling, the packet in log space, and neither is what I ran.

Every number sections 1–5 state is exact:

- **Forward masses** 3/50, 6/25; 69/1250, 243/5000; 363/62500, 3429/125000; 9399/1250000, 711/390625 — i.e.
  0.06/0.24, 0.0552/0.0486, 0.005808/0.027432, 0.0075192/0.00182016. Column totals 0.3, 0.1038, 0.03324, 0.00933936.
  Evidence **58371/6250000 = 0.00933936**, reproduced a second way by summing all sixteen enumerated paths.
- **Filtered Rainy** 0.2, 0.531792, 0.174729, 0.805109. **Smoothed Rainy** 0.194943, 0.433828, 0.236316, 0.805109, and
  the last row agrees with filtering exactly, not to tolerance.
- **Backward** β₂ = 19/50 and 13/50 — the 0.38 and 0.26 the prose computes by hand.
- **All three pair blocks**, with row sums equal to γₜ and column sums equal to γₜ₊₁, exactly.
- **Viterbi** δ = 0.06/0.24, 0.0384/0.0432, 0.002688/0.015552, 0.0031104/0.00093312; predecessors Sunny/Sunny,
  **Rainy**/Sunny, Sunny/Sunny; path Sunny→Sunny→Sunny→Rainy; joint **243/78125 = 0.0031104**; posterior share
  0.333042; 66.695791 % of the mass elsewhere. The section-5 "the best Rainy predecessor is Rainy" arithmetic
  (0.02688 beats 0.01728) holds.
- **Forecast** [0.46, 0.54]; P(Clean) 0.284; P(Shop) 0.346; the collapsed-to-Sunny 0.34.
- **Scale factors** 0.3, 0.346, 0.320231, 0.280968, product exactly the evidence.
- **Changed future** [0,1,0,0]: the three earlier filtered rows are bit-identical, smoothed t=1 becomes 0.397624,
  and the best path becomes all Sunny.
- **Missing vs deleted**: 0.802780 against 0.795320.
- **Changed Rainy emission row** to .2/.3/.5 leaves the best path's joint mass at exactly 243/78125.
- **The three-state constrained model**: marginals 0.40/0.35/0.25 and 0.60/0.20/0.20; modes A→A; Viterbi B→A at
  exactly 7/20; expected correct 1.00 against 0.95; the changed prior [.2,.55,.25] moves both decoders to B→A. The
  joint table ([[0,1/5,1/5],[7/20,0,0],[1/4,0,0]]) matches the figure cell for cell.
- **Section 6**: initial counts [0.481478, 1.518522]; edge counts [[0.408329, 0.073150],[0.933322, 0.585199]] summing
  to 2; symbol counts summing to 4; updated π, A and B rows all as printed; log-likelihood −4.728043 → −3.529108.
  The duplication null is exact equality of the updated parameters, the doubled counts are exactly twice, and a
  length-one recording retains its transition rows identically. Joining the boundary gives 1 start and 3 transitions
  and moves the Rainy→Rainy count from 0.408329 to 0.506913 — the number the page prints.
- **Duration**: a = 0.7 → mean 10/3 and the eight bars; a = 0.95 → mean 20; a = 0.8 → P(D=3) = 0.128; a = 0 absorbing
  at the other end. **Underflow**: 0.3¹⁰⁰ = 5.153775e−53 representable, 0.01⁴⁰⁰ = 0.0, 400 log 0.01 = −1842.068074.
- **Parameter count** (2−1) + 2(2−1) + 2(3−1) = 7.

**Zero disagreements.** Every displayed digit of the constructed mechanism is exactly right.

## A2. The EM trajectories at 60 digits — **no disagreement**

`scratch/hmm-independent/em.py`, `mpmath` at `dps=60`, my own forward/backward with no scaling and no logarithms;
only the seeded RNG draws are shared, for the reason declared above.

The twelve length-30 sequences reproduce the stored ones exactly. Generating-model score **−393.562196**. Seed 3
−566.599897 → **−390.253507**; seed 7 −410.236336 → **−390.630410**; seed 19 −459.466552 → **−392.284393**; every one
of the 120 updates non-decreasing at 60 digits. Uniform start: −395.500424 → **−395.091859** and flat thereafter, with
the two emission rows identically equal and equal to the empirical frequencies [0.333333, 0.352778, 0.313889]. Seed
19's final start vector matches `hmm-data.js` to all stored digits.

The builder's declared departure — writing "all three fits exceed the generating score" where the manuscript wrote
"a fitted model can exceed that score" — is **correct and an improvement**: all three do exceed it and the symmetric
start stays below, so the contrast the sentence draws is real rather than hypothetical.

## A3. The supervised tagging experiment — **no disagreement**

`scratch/hmm-independent/tag.py`, `rb.py`, `rb2.py`. Exact-rational fit and exact-rational Viterbi over the full
146-symbol emission table; smoothing entered as `Fraction(1,10)` and `Fraction(1)`, never as a float.

120/40/40 sentences, 1188/341/370 tokens, 200 distinct ids, every sentence of length 3–15, vocabulary **146**,
**140** unknown development tokens, majority-Other **216**. Four configurations: 0.1 lexical **266** (8 sentences),
0.1 HMM **268** (9), 1.0 lexical **266** (8), 1.0 HMM **270** (9). The declared tie rule is load-bearing exactly as
the page claims — the two lexical rows really do tie at 266.

I independently recomputed the decision changes and got **all sixteen positions** the served data records, in both
lists and in order: repairs at (1,3) (1,7) (8,7) (12,0) (14,10) (15,0) (16,11) (20,0) (27,0) (34,0); breaks at (3,3)
(4,0) (11,0) (14,5) (16,5) (39,4). Ten repairs, six breaks, net four. Both named specimens reproduce: "Dear Nina ,"
lexical Noun–Noun–Other against HMM Other–Noun–Other matching the reference, and "Read the entire article ; …" where
context turns "article" from a correct Noun into Other.

## A4. The tie finding — adjudicated in exact arithmetic. **The lesson's teaching is accurate.**

This is the thing the brief asked to be settled, so it gets its own treatment. I found the ties myself, from my own
exact-rational Viterbi, by enumerating the full set of optimal paths per sentence rather than backtracking a single
one.

**Part one — the ties are real, and there are exactly three.** At both smoothing strengths, exactly three of the forty
development sentences carry two complete paths of exactly equal probability, and they are the same three:
dev 1 `…gettingpolitical…-0005` ("A la guerre c'est comme a la guerre !"), dev 2 `…juancole…20040114…-0005`
("The US troops fired into the hostile crowd , killing 4 ."), dev 39 `…tacitusproject…-0024` ("That 's a Senate term
-- particularly on good judges ."). These are the three the served `tieAudit` names, and the `adjacentUnknownPairs`
it lists (5, 5 and 4 pairs) are exactly the adjacent unknown-symbol pairs I compute from the encoded sentences.

**Part two — the mechanism the page states is exactly right, and stronger than it needs to be.** The page says the
two paths "permute the same multiset of transition and emission factors and leave the product identical." I checked
this literally. For dev 2 the two optimal paths are `[2,0,2,0,0,…]` and `[2,0,0,2,0,…]`; their transition multisets
are both `{(0,0)×3, (0,2)×4, (2,0)×4}` and their emission multisets are both
`{(0,0)×7, (2,0)×1, (2,5)×1, (2,7)×1, (2,118)×2}`. Identical multisets, so the two joint probabilities are the same
product in a different order. The consequence — which the page does not spell out but which supports rather than
undermines it — is that **this tie cannot be broken by perturbing any parameter of the model by any amount**. It is
structural, not numerical.

**Part three — the band.** My exact computation, with an explicit tie policy rather than a backtrack:

| | lowest-index rule | highest-index rule | packet's float log-space route |
| --- | ---: | ---: | ---: |
| smoothing 0.1 | **267** | **270** | 268 |
| smoothing 1.0 | **269** | **272** | 270 |

The page says: "the reported 270 is one member of a band: taking the lowest-indexed state at every tie gives 269,
taking the highest gives 272", and "the same fitted model reports anywhere from 269 to 272". Both are exactly right,
and the served `tieAudit.configurations[*].tokenTotals` — `{first: 267, last: 270, stored: 268}` and
`{first: 269, last: 272, stored: 270}` — match my numbers on all six values. Since the three tied sentences are
independent, every intermediate total (270 and 271 at smoothing 1.0; 268 and 269 at 0.1) is also attainable, so
"anywhere from 269 to 272" is the correct form of the claim, not an over-statement.

**Part four — robustness. The claim holds, and I can give it a stronger proof than the record does.** The three tied
sentences, their two candidate paths, and those paths' correct-token counts are *identical* at both smoothing
strengths. Therefore any single consistent tie rule contributes the same amount at both, and the difference
1.0 − 0.1 is invariably **exactly 2**, whatever rule is chosen. Likewise the HMM's minimum over the band (269 at
smoothing 1.0, 267 at 0.1) exceeds the lexical 266 under every rule. So both halves of the page's robustness sentence
are true, and the stated four-token margin genuinely is not robust — the band admits +3 to +6. **Accurate, neither
over- nor under-claimed.**

One record nuance, not an error: `design.md:194–199` says a one-unit-in-the-last-place difference in the emission
table "flips development sentences 1 and 2 at smoothing 0.1, moving the recorded count from 268 to 267". That is a
statement about which tied path a *floating-point* `argmax` selects, not about the tie itself, and as such it is
correct — the exact tie is unbreakable, which is precisely why the reported member is an arithmetic artefact. The
wording is careful enough that I do not think it misleads.

## A5. `hmmlearn` — **no disagreement**

The shared lesson runtime `scratch/lesson-tools` resolves to Python 3.12.14, **NumPy 2.3.5, SciPy 1.18.1,
scikit-learn 1.9.1, pandas 3.0.1, and no hmmlearn** — exactly what `hmm-native.json` records, so the claim that
resolving hmmlearn elsewhere left it untouched holds. The isolated venv `scratch/hmm-optional` resolves to
hmmlearn 0.3.3, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 on Python 3.12.14 — the five versions the page names.

I executed `hmmlearn-examples.py` in that venv myself. Its stdout is **byte-for-byte** the `expected` string in
`hmm-examples.js`, including the Gaussian block, and its stderr is exactly the displayed warning.

Every claim the page makes about it checks out:

- score −4.673517551573099 = log 0.00933936 ✔
- Viterbi −5.773003943698154 = log(243/78125), path [1 1 1 0] = Sunny Sunny Sunny Rainy ✔
- the eight `predict_proba` values are my exact smoothed rows to every printed digit ✔
- the MAP decoder's **2.940021586061572** against the sum of the four smoothed maxima: exactly **57204/19457 =
  2.9400215860615715** in rationals, which is the same double to within its last bit. The page's "no log probability
  of a probability can be positive, and the value is exactly the sum of the four smoothed maxima" is correct ✔
- the fit score −1.3862943611198906 is exactly `2 log 0.5` ✔
- the stderr warning's **7 free scalar parameters** equals the lesson's own (N−1) + N(N−1) + N(M−1) at N=2, M=3 ✔

## A6. The trust root — **honest count, real mechanism, justified exclusion** (but see S6)

I counted the scalar leaves of `calculated-inputs.json` myself with my own walker: **13,247** total, of which
**4** are under `additional_author_checks`, leaving **13,243** written by the program. The packet hash matches the
one recorded in three places. 13,243 + 4 = 13,247 and 13,243 + 3 = 13,246, so the arithmetic behind
"13,246 of 13,247, 99.99 %" is internally consistent and honest.

The mechanism is **not** a tautology. `verify-hmm-examples.py` re-runs the frozen program in a fresh process and
compares all 13,243 leaves of the fresh run against the packet's, and separately asserts that the only extra leaf
paths in the packet are those four booleans. `verify-hmm-data.py` re-derives the packet through a genuinely separate
implementation. The one exclusion, `native_example_ast_parsed_not_executed`, records that hmmlearn was absent while
the content was written; that is a fact about the author's process, it is not re-derivable, and phase two genuinely
supersedes it by executing the program — which I independently confirmed in A5. **The exclusion is justified.**

Where the claim overstates is narrower and is recorded as S6.

## A7. Served data, programs and attribution — **no disagreement**

`public/learn-assets/hmm/ewt-sequences.json`, `hmm-experiments.py` and `hmmlearn-examples.py` are **byte-identical**
to the packet copies (`diff` clean), and all four served hashes plus `data-sources.json` and the preserved original
`.jsx` match every hash recorded in the evidence files and in `design.md` (including 63,607 bytes and
`9f99c9c7…0614c`).

`ATTRIBUTION.txt` is accurate on every checkable point: UD English EWT r2.16, CC BY-SA 4.0 for the annotations,
Stanford copyright 2013–2021, the underlying-text notices, 200 records, 120/40/40, 1188/341/370, distinct ids, length
3–15, integer-id rows only, unchanged served file with original UPOS preserved, and the statement that the 40
reserved sentences are never scored. I verified every one of those counts from the file itself. The three source
`.conllu` hashes are declared as coming from the earlier CRF retrieval, which `data-provenance.md` states plainly
rather than implying a fresh download — the right way to record a reused extract.

## A8. The displayed excerpts and their seams — faithful; the departure is justified

All four excerpt ranges are exact AST spans of the functions they name: 9–71 (`log_values` + `infer`), 74–86
(`forward_scaled`), 89–118 (`expected_counts` + `normalize_counts` + `em_step`), 233–300 (`real_tagging`). I parsed
the frozen file myself and confirmed each range starts and ends on the right line and that the four excerpts cover
those functions and no others. Each excerpt is a literal substring of the frozen file.

The seams are honest: every excerpt is headed "Lines X–Y of `hmm-experiments.py`, reproduced exactly", so the gaps
(1–8 imports, 119–232 `fit_em`/`sample_sequences`/`mechanism_examples`, 301–331 `serializable`/`main`) are visible
from the line numbers, and the complete file is a real download that I verified byte-identical. Showing 153 of 331
lines rather than inlining the whole program is a reasonable departure and is declared. The one thing wrong with it
is the count in the sentence that justifies it — S3.

## A9. The investigation contract — met

Checked in code and then driven on the page.

- **Learner's own inputs.** Six investigations, each with real entity-level controls: a state per time step; every
  report editable plus add/remove/missing plus a freely typed Rainy emission row; a state per time in the constrained
  graph plus an editable initial row; recording contents and the boundary position; sentence and smoothing plus a
  free respelling; the self-transition probability. None is a two-preset toggle.
- **Recorded prediction before anything is shown or graded.** `Prediction` starts with no radio selected, `Apply` is
  disabled until every group is answered, and the verdict, the describe line and every consequent table are inside
  `{shown && …}`. I confirmed on first paint at 1366 px that the tagging lab shows no tags, no counts and no
  aggregate table. Radios and the numeric field are disabled after reveal, so a prediction cannot be edited into
  correctness.
- **Graded against the committed draft.** `check` computes `answerFor(draft, active)` and freezes
  `key: describeKey(draft)`; the verdict prints "Graded against the committed state: …". Not live state.
- **Verdict retired on any input change.** `edit` and `suggest` both call `retire()`, which clears the result and the
  recorded choices. A retired verdict is deliberately **not** shown while the next prediction is open — the code
  comment gives the right reason (for a null, the previous outcome is the answer).
- **Fixture / null cases.** Real ones, and they are exact: the filtering null under a changed future (bit-identical
  prefix), the respelling null (identical symbols so nothing downstream can move), the dataset-duplication null
  (exactly equal updated parameters), the relabelling null, and the tie note itself.
- **Construction graded from the computed consequence**, never from which preset was pressed —
  `gradeSmoothedConstruction`, `gradeLegalPathConstruction` and `gradeBoundaryConstruction` all re-run inference or
  re-count events on the submitted draft.
- **Ordering.** The prose that resolves the tagging lab's default fixture ("Dear Nina ,") sits at line 278, *after*
  `<RealTaggingLab />` at line 277. The lab is met before its answer.

## A10. The graded rules at degenerate inputs — **no disagreement**

`scratch/hmm-independent/rules.mjs`. I drove the exported rules directly, at the inputs most likely to break them,
and compared each against what I hold to be the right answer.

- **Exact ties at the maximum.** A model where all sixteen paths tie, and a flanked-swap model with eight tied
  optima: `pathRankOutcome` returns `largest` for **every** tied maximiser. That is correct — they all are — and it
  is the case this lesson actually contains.
- **Every path impossible.** All paths return `zero`, not `largest`. The branch order is load-bearing and right: a
  probability-zero story is never reported as the most likely one merely because nothing beats it.
- **State-independent emissions**, where no edit can move either belief: both questions return `unchanged`,
  correctly, rather than manufacturing a direction. The prediction options include "does not move" and "has no value
  at all", so a learner can record the right answer in that degenerate world.
- **Exactly at a threshold.** `gradeLegalPathConstruction` at `target === joint` returns solved (the comparison is
  `>=`, matching the task's "at least"), and fails one ulp above. Correct.
- **Both ends of the hazard family.** `stay = 0` (absorbing exit) and `stay = 1` (never leaves) both return `same`,
  which is the claim.
- **The identification task** accepts *any* of the ten repairs and *any* of the six breaks (`.some`), so a correct
  answer cannot be graded wrong for picking a different valid position. Its defaults (sentence 0, token 0) are
  neither a repair nor a break — I checked against the served lists.
- **The 1-and-3 decoy** is rejected by structure while matching all three totals, which is the point.

I could not construct a case where the page marks a correct answer wrong. The one asymmetry I found —
`gradeBoundaryConstruction` rejecting a [2,2] split that contains a retained *missing* report, because its emission
total is 3 rather than 4 — is correct behaviour under a task that explicitly says "the four reports arrived
concatenated", and the verdict text names the shortfall precisely rather than just failing.

## A11. The paint — **no CSS-over-attribute defect found anywhere**

`scratch/hmm-independent/paint2.cjs`, my own sweep, independent of the builder's guard. At 1366, 900, 390 and 320 px
I compared **every** SVG element's `stroke-width`, `fill`, `stroke`, `stroke-dasharray`, `opacity`, `font-size` and
`r` attribute against its computed value (numerics at relative tolerance; `none`-vs-painted for the colour
properties, which is the `fill="none"` failure mode). **Zero mismatches at all four widths.**

The edge-width encoding genuinely reaches the paint: 30 distinct rendered stroke widths spanning 1.095–4.819 px at
1366 px, and I verified per figure that each edge's attribute survives — figure 2's twelve carrying edges at twelve
distinct widths, figure 4's chosen-versus-rejected split, figure 5's four permitted edges (three at the 4.4 maximum,
one rejected at 3.4 encoding a 0.714 share) and its five forbidden edges at the distinct 1 px dotted style. I opened
the images and confirmed the thickness variation is visible, not merely present in the DOM.

The two perceptibility claims are measurements, and I reproduced them independently:

- **Rendered label sizes.** 209 labels; 10.32–14.82 px at 1366, 10.95–14.82 at 900, 10.00–12.35 at 390, 9.81–12.35 at
  320. The recorded 9.81–14.82 bound is exactly the min and max across the three declared widths.
- **The two objective windows.** Smallest rendered separation of the four final scores: **0.272 px** in the overview
  and **7.448 px** in the detail window, matching the recorded 0.27 and 7.45 to the digit.

The document does not scroll horizontally at any of the four widths, so the 141 px overflow the record reports
fixing is genuinely fixed.

## A12. Verifiers, evidence and the falsification harness — all green, and reproducible

Re-run from a clean build: `verify-hmm-models.mjs` **PASS, 518 grouped checks across 66 groups**;
`verify-hmm-data.py` **PASS, 2,665 checks across 481 groups, 13,246 of 13,247 leaves (99.99 %)**;
`verify-hmm-examples.py` **PASS, 2 of 2 programs executed, 65 oracles, 13,243 leaves reproduced**;
`verify-hmm-browser.cjs` **passed, 21 cases, 55 screenshots**. The regenerated `hmm-models.json`, `hmm-data.json`
and `hmm-native.json` are byte-identical to the committed ones once the timestamp field is removed — the numbers are
genuinely deterministic, not re-fitted to the record. All 55 screenshot digests are distinct.
`scratch/hmm-phase-two/falsify.mjs`: **19 of 19 caught, 19 citing the guard they were aimed at**, including the four
guards the record admits previously could not fail.

What they do *not* assert is S9 and O1–O2.

## A13. The accepted-not-fixed 900 px item — recorded exactly, and I accept the judgement

I measured it myself: at 900 px exactly one `.hmm-table-scroll` overflows, by **exactly 42 px**, and no table
overflows at 1366, 390 or 320. It is figure 8's three-column comparison table, inside a scroll box with
`scrollbar-gutter: stable`, and it stacks fully below 560 px. That is the recorded claim, to the pixel. Forty-two
pixels on a three-column table inside a box that visibly is a scroll box, at one intermediate width, with a stacked
layout on either side of it, is a reasonable thing to record rather than chase. **Accept.** The only thing I would
tighten is the guard, which passes anything under 60 px (O5).

---

# Part B — findings

## Blocking

### B1. The lesson's opening promise states the tie finding on the **reserved test split**

`src/learn/data/topics/hidden-markov-models-hmm.jsx:103`, inside `<LessonIntro>`. Rendered text, verified in the
paint:

> … and then read 270 of 341 real tokens tagged from 120 real sentences — and find that **three of the forty test
> sentences** have two complete paths of *exactly* equal probability.

Two things are wrong with it, and the second is worse than the first.

**It is factually false.** The ties are in the forty **development** sentences. I found them myself, in exact
rational arithmetic, and they are dev indices 1, 2 and 39 (A4). The reserved split is a different forty sentences
and nothing in this lesson decodes them at all — `real_tagging` filters to `split == "dev"` and the packet records
`reserved_test_scored: False`.

**It contradicts the lesson's own integrity claim, three times over.** The same page says the reserved sentences
"stay unscored here" (line 284), "remain unscored, precisely so that a frozen procedure could be evaluated on an
appropriate independent set" (line 375), and are "never predicted or scored" (line 413). The opening sentence tells
the reader that a property of those forty sentences was computed. A learner who reads the introduction and the
provenance note has been told both that the reserved set was touched and that it was not.

This is the same defect class as the Phase C figure-5 repair — a prose statement that collapses a distinction the
lesson exists to teach — and here the distinction is split discipline, which section 8 spends three paragraphs on.
It is one word, and it is in the first paragraph a reader meets.

**How verified.** Recomputed the tied sentences from the extract in exact rationals and confirmed all three are in
the `dev` split; read the served `tieAudit` (which is built from the development configurations); read the rendered
page text at 1366 px through Playwright.

## Should-fix

### S1. The section-2 question table renders raw LaTeX subscripts as literal text

`hidden-markov-models-hmm.jsx:136–141`. The six rows are plain strings, so the page renders:

```
How well does the model explain the reports?            P(o₀…o_{T−1})
What is the current state after this report?            P(z_t | o₀…o_t)
What was an earlier state, using the later reports too? P(z_t | o₀…o_{T−1})
What is the next state likely to be?                    P(z_{t+1} | o₀…o_t)
What single whole path has greatest probability?        argmax_z P(z | o)
```

The braces and underscores are visible on screen. The same cell mixes a correctly typeset Unicode subscript (`o₀`)
with an untypeset one (`o_{T−1}`), so it reads as a rendering failure rather than a convention. Every other formula
on the page goes through `<Math>`/KaTeX, and the manuscript writes these as proper LaTeX. This table is the one
place the notation leaked.

It also sits outside the escape audit: that audit covers the 122 displayed LaTeX expressions, and these strings are
not LaTeX expressions as far as the tooling is concerned, so nothing can catch it but looking.

**Why it matters.** This is the table that introduces the lesson's six queries, and the distinction between
`P(z_t | o_{0:t})` and `P(z_t | o_{0:T−1})` is the whole of sections 3–4. The two are told apart by their
subscripts, and the subscripts are the part that did not render.

**How verified.** Read `document.body.innerText` from the rendered page at 1366 px.

### S2. The Clean-forecast sentence opens with a number that is not a forecast of anything

`hidden-markov-models-hmm.jsx:157`. Rendered:

> The predicted probability of the next report being Clean is **0.7(0.5)**… more precisely 0.46(0.5) + 0.54(0.1) =
> 0.284.

`0.7` is bound to `weather.transition[0][0]` — P(Rainy→Rainy). It is not a predicted state probability, and
0.7 × 0.5 = 0.35, which is neither the answer (0.284) nor the wrong answer the next sentence warns against
(collapsing to Sunny gives 0.284's Shop analogue, 0.34). It reads as a first approximation that is then corrected,
but it is not labelled as one and it is not a quantity any reader can reconstruct. Almost certainly the binding
should be `forecast.nextState[0]` (0.46), which would make the sentence "is 0.46(0.5)… more precisely …" —
redundant but coherent.

**Why it matters.** This paragraph's entire job is to say that you must propagate the whole filtered vector rather
than jump from one state. Leading it with an unexplained product of a transition entry and an emission entry models
the mistake the paragraph is warning against, without saying so.

**How verified.** Read the binding in the JSX; read the rendered sentence from the page; recomputed 0.46/0.54 and
0.284 exactly.

### S3. The page tells the learner the program is 332 lines; it is 331

`hidden-markov-models-hmm.jsx:221` prints `hmmExamples.experiments.lineCount`, which `verify-hmm-examples.py:363`
computes as `source.count(NL) + 1`. `hmm-experiments.py` ends with a newline, so it has 331 newline characters and
331 lines; `count(NL) + 1` returns 332. `wc -l` and `len(text.splitlines())` both give 331.

The number is load-bearing in that sentence — it is the justification for showing four excerpts instead of the whole
file — and it is the only quantity on the page that is not derived from a verified model.

Separately, `design.md:156` states the same file is **356 lines**, which is wrong by 25 and matches neither the file
nor the page. Both numbers should be 331.

**How verified.** Counted the file three ways; read the generator expression; read the rendered sentence.

### S4. A trellis edge that carries **nothing** is painted wider than one that carries almost nothing

`hmm-models.js:905–909` and `HmmShared.jsx:614`. `edgeWidth(0)` returns `null`, so the component emits no
`stroke-width` attribute and the line paints at the SVG initial value, **1 px**. The smallest positive share paints
at `edgeWidthRange.minimum` = **0.9 px**. The encoding is therefore non-monotone at exactly zero: an edge carrying no
mass is drawn *thicker* than an edge carrying an arbitrarily small positive mass.

The doc comment immediately above says this cannot happen:

> A zero share returns `null`, which the drawing renders as a distinctly styled forbidden edge instead of a hairline.

It does not. The distinct forbidden style is gated on `edge.forbidden`, which `trellisEdges` (`hmm-models.js:929`)
sets from `edge.transition === 0`. A share of zero arising from a zero *source mass* — a state the evidence has ruled
out — has a positive transition, so `forbidden` is false and the edge is drawn in the ordinary solid `is-carrying`
green.

**It is reachable on the page, and I reached it.** In the section-4 investigation: open "Free exploration: edit the
model as well", press "Unlock the model", type `0, 0.5, 0.5` into the Rainy emission row, apply. The row is accepted
(it is a distribution), and the drafted trellis then contains edges with `stroke-width` attribute `null`, painted at
1 px / **1.095 rendered px**, in the same green solid style as the contributing edges beside them — measured in the
live page and captured at `scratch/hmm-independent/shots/zero-emission-trellis.png`.

**Why it matters.** This lesson's recurring thesis is that zero and vanishingly small are different things — figure
5 exists to say so, figure 8 is titled "Three different things that all look like a very small number", and practice
6 turns on it. The trellis draws them the wrong way round, with zero rendered as the larger of the two.

The builder's new stroke-width guard cannot see this: it iterates `.hmm-lesson svg [stroke-width]`, and these
elements have no attribute to compare. Neither can my sweep in A11, for the same reason — I found it from the model
layer and then reproduced it in the browser.

**How verified.** `scratch/hmm-independent/zeroshare.mjs` (four such edges in a four-step recording) and
`zerolab3.cjs` (driven on the built page, widths read from `getComputedStyle`).

### S5. Figure 7's caption still over-states what the overview window cannot resolve

`HmmFigures.jsx:390`:

> at that range the four final scores land less than a pixel apart, which is why the second panel exists

I measured the rendered last points of the four tracks at 1366 px. The **closest pair** is 0.272 px apart — that
part is fine, and it is what the verifier records as `overviewSmallestSeparationPx`. But the **four scores** span
**3.36 rendered px** end to end (gaps 0.272, 1.148, 1.940). Read as written — the four land within a pixel of one
another — the sentence is false by more than threefold, and 3.36 px is a separation a reader can see.

This is the second iteration of this caption. The first said "under a quarter of a pixel" where the measurement is
0.27; the replacement fixed the bound but kept the subject as all four scores, which was the other half of the
problem. The true and defensible sentence is about the closest pair.

I also confirmed the claim is directionally sound: the detail window separates the same closest pair by 7.448 px, so
the second panel is genuinely earning its place. Only the quantifier is wrong.

**How verified.** `scratch/hmm-independent/paint2.cjs` reads each `polyline.hmm-track`'s final point and scales by
the SVG's own element-to-viewBox factor; cross-checked against `separationInPixels` and `objectiveWindows` by hand
(overview −570…−385 over 96 units gives 0.196 viewBox px for the closest pair, ×1.338 render scale = 0.26).

### S6. Three trust-root leaves are counted as re-derived without any assertion reading their values

`verify-hmm-data.py:697–699`. The verifier re-derives the three properties honestly — it recomputes the
duplication null, the doubled counts and the length-one retention — and then marks the corresponding packet leaves
covered:

```python
cover.mark("additional_author_checks.duplicated_dataset_parameter_null",
           "additional_author_checks.duplicated_dataset_counts_double",
           "additional_author_checks.length_one_transition_rows_retained")
```

`Coverage.mark` (lines 462–471) only resolves the path and checks it exists in the packet. Nothing anywhere compares
those three stored booleans to `True`. I grepped both Python verifiers and the Node verifier: the only other
reference is `verify-hmm-examples.py:218`, which asserts the packet has exactly those four *extra leaf paths*, not
their values. Flipping any of the three to `false` in `calculated-inputs.json` would leave the run green and
coverage at 99.99 %.

The file does it correctly three hundred lines later — `expect(block["reserved_test_scored"] is False, …)` at line
1126, *then* `cover.mark(...)` — so the pattern exists; it was just not applied here.

**Why it matters.** `design.md:117` states coverage is "measured as asserted leaf paths rather than a counter",
which is the sentence that makes the 99.99 % meaningful. For 3 of the 13,246 it is measured as *declared* leaf
paths. The overstatement is small in magnitude and precise in location, and the underlying mathematics really is
re-derived — but the mechanism is weaker than the sentence describing it, and this is a guard that exists and cannot
fire.

**How verified.** Read `Coverage.mark` and all 31 `cover.mark` call sites; grepped every reference to the four
`additional_author_checks` keys across all four verifiers; counted the packet's leaves myself (13,247 / 4 / 13,243).

### S7. Four Phase A record claims contradict the tree, and Phase C only partly supersedes them

`design.md` carries both appended sections, and a reader of the Phase A table is told four things that are not true
of the tree I reviewed:

| `design.md` | Claim | Tree |
| --- | --- | --- |
| L86 | `hmm-models.js` is "the whole verified model layer, **1,076 lines**" | **1,136 lines** |
| L156 | inlining `hmm-experiments.py` would be "**356 lines**" | **331 lines** |
| L94, L236 | the blueprint is "**Not registered**" / "unregistered by design" | **registered** — `src/learn/data/curriculum/blueprints/index.js:26` imports it |
| L235 | `scripts/verify-hmm-browser.cjs` "**is not written yet**" | exists, 67 KB, and I ran it |

Phase C corrects two of these in passing (L241 says the blueprint is registered and the browser verifier runs), but
the Phase A table and its departure list were never updated, so the same file asserts both states. The line counts
are corrected nowhere. This is the failure mode the brief names — a record asserting an implementation state that
contradicts reality — in its mildest form: the numbers are stale rather than fabricated, and one of them (356) is
also the number displayed to learners as 332 (S3). Two of the three conflicting pairs are resolvable only by
checking the tree, which is what a record is supposed to save a reader from doing.

**How verified.** `wc -l` and `splitlines()` on both files; `grep` for the blueprint registration; ran the browser
verifier.

### S8. The same two window separations are recorded as "px" with two different values

`design.md:176` (Phase A): "at the overview's range the closest pair is **0.20 px** apart, and at the detail
window's range it is **5.57 px**. Both numbers are asserted."
`design.md:278` (Phase C): the two windows "separate their four final scores by **0.27 px** and **7.45 px**".

Same figure, same quantity, both described as asserted/measured. I resolved it: Phase A's numbers are `viewBox` user
units, computed by `separationInPixels` against `objectiveWindows.*.pixels`; Phase C's are rendered CSS pixels. The
ratio is 0.27/0.20 = 1.35 and 7.45/5.57 = 1.337 — the same render scale (1.338 at 1366 px), which is what confirms
the reading. Both numbers are right in their own frame.

But the record calls both "px", which is exactly the confusion the Phase C work exists to dispel — the difference
between a number the model asserts and a number measured in the image. A reader reconciling the two sections cannot
tell that without re-deriving the scale factor, as I did. Label the Phase A pair as viewBox units, or restate it.

**How verified.** Computed `separationInPixels` by hand from `objectiveWindows` and the four final scores
(0.196 and 5.567 viewBox units); measured 0.272 and 7.448 rendered px in the browser; took the ratio.

### S9. A fifth, sixth and seventh guard that cannot fail — and one block of dead code

The record says four un-failable guards were found and closed, and the harness now trips each. There are more, and
they sit inside the advertised grids.

**(a) Two of the three per-edge assertions in the "90 drawn trellis edges" sweep are tautologies.**
`verify-hmm-models.mjs:455–458`:

```js
assert(edge.forbidden === (edge.transition === 0), …);
close(edge.carried, edge.previousValue * edge.transition, …, 1e-15);
```

`trellisEdges` (`hmm-models.js:917–930`) copies `previousValue`, `transition` and `carried` verbatim off the object
`trellis` built at lines 248–252, where `carried: mass * model.transition[origin][destination]` and
`previousValue: mass`. The second assertion is therefore the same IEEE multiply of the same two stored operands
compared against itself; the first is `(t === 0) === (t === 0)` because `forbidden` is *defined* as
`edge.transition === 0` at line 929. Neither can fail for any model, any observation sequence or any mutation of the
surrounding code. They run 90 times each — two thirds of the assertions in that grid.

The companion at line 440, `close(total, cell.aggregate, …)`, has the same shape: `total` is a left-to-right reduce
of `edge.carried` and `aggregate` is `sum(carried)` over the same array in the same order, so the two are bitwise
identical by construction.

**(b) 606 of the 2,121 "hazard settings" compare a function with itself.** `verify-hmm-models.mjs:810–820` runs
`for (let late = early; …)`, so on the first iteration of every inner loop `late === early` and both assertions
reduce to `f(s,e)` against `f(s,e)`. 101 values of `stay` × 6 such pairs = **606**, or 28.6 % of the advertised
total. The remaining 1,515 are a real memorylessness property and breakage #6 confirms they trip. (Two smaller
notes: `exitAfter` ignores `elapsed` entirely, and the comment says "every pair of elapsed times up to twenty" where
the loop steps by 4 and visits six values.)

**(c) Assertions implied by the line above them.** `verify-hmm-models.mjs:571–573` and `672–677` re-assert
conditions that the preceding `assert.equal` has already established, because the `expected` value on the line above
is built from the same conjuncts. Likewise `verify-hmm-models.mjs:702`, eighteen times:
`assert.equal(verdict.solved, verdict.legal && verdict.joint >= target)` re-reads the same object's own fields with
the formula that produced `solved`.

**(d) Dead code carrying the rationale for the check that replaced it.** `verify-hmm-browser.cjs:149–169` defines
`readTrellisGeometry`, documented as *"the only way to know that the width the model computed is the width the
reader sees, rather than one a stylesheet overrode."* It is called twice (lines 642 and 644). The first result feeds
a single non-null assertion; **`forwardTrellis` on line 644 is never referenced again anywhere in the file**. Every
node centre, card box, edge endpoint, stroke width and label it reads is computed and discarded. The real
stroke-width verification is a separate inline evaluate at lines 701–724.

**Why it matters.** Not because the lesson is wrong — I found no numerical defect in anything these guards cover —
but because the four grids in question are the ones the record cites as evidence of breadth, and the advertised
counts include roughly 750 assertions that no change to the code under test could break. The harness's own stated
principle is that a guard which has never been seen to fail is not evidence.

**How verified.** Read and quoted each cited line; traced `carried`/`forbidden`/`aggregate` back to their
definitions in `hmm-models.js`; computed the hazard loop's 2,121 = 101 × 21 and its 606 diagonal pairs by hand;
grepped `forwardTrellis` across the whole verifier (one occurrence, the assignment).

## Observations

### O1. "518 model checks" is a marker count; three of the four headline numbers are genuine

`verify-hmm-models.mjs` totals hand-placed `record(name)` calls, not executed assertions. Thirty-three of the 66
groups fire exactly once regardless of work done — the 2,121 hazard checks contribute 1, the 1,024 belief-move
verdicts contribute 1, the 231 priors contribute 1 — while **200 of the 518** come from a single 201-point grid of a
linear function and 37 more from refusal cases. The console string says "grouped … checks", which is honest;
`design.md:110` and the Phase C summary render it as "518 model checks", which reads as an assertion tally.

By contrast **2,665 data checks** and **65 native oracles** are true runtime tallies (both `expect()` helpers
increment on every call), and **55 screenshots** is the best-evidenced number in the suite — each path is pushed only
after a successful capture, and distinct paths *and* distinct SHA-256 digests are asserted. I confirmed all 55
digests are unique.

### O2. Nothing falsifies the browser layer, including the guard written to catch the browser-layer defect

All 19 breakages touch `hmm-models.js`, `hmm-data.js` and `hmm-examples.js` only. No breakage touches
`hmm-labs.css`, `HmmFigures.jsx`, `HmmShared.jsx`, the topic `.jsx` or `verify-hmm-browser.cjs`. So "19 of 19
breakages caught" says nothing about the 21 browser cases.

That matters most for the stroke-width guard, which I examined closely and judge **genuine**: it compares
`node.getAttribute('stroke-width')` against `getComputedStyle(node).strokeWidth` (not computed against computed), it
runs over a real subject set (65 elements, recorded in the evidence), and it carries an explicit `>= 40` non-empty
floor — the guard that two other checks in the same file lack. But it exists precisely because a CSS declaration can
outrank a presentation attribute, and the one-line breakage that would prove it works (append
`.hmm-edge { stroke-width: 2px }` to the stylesheet) is not in the harness. It is also run once, in the reset state,
at one width, so figures that only exist after a prediction is committed are outside it.

Two smaller gaps in the same file: `verify-hmm-browser.cjs:469–472` asserts `.every()` over a possibly-empty
`text.hmm-muted` set with no non-empty floor, so a class rename would make "no path score is shown before the
prediction is recorded" pass with nothing inspected; and the curve sampler at line 776 has no floor either, where
every other geometry check in the file has one.

### O3. The edge-width readback compares a sorted multiset

`verify-hmm-browser.cjs:660–670` sorts both the expected and the actual widths before comparing, so it establishes
that the twelve widths are present, not that each edge received its own. Any permutation of widths across the twelve
edges passes. The distinctness floor and the 1e-3 tolerance are both good; only the pairing is unchecked. I verified
the pairing myself in A11 (per-edge attribute against per-edge paint) and it is correct.

### O4. Loose floors where an exact count was available

`assert(drawnEdges >= 60)` where the value is deterministically 90; `assert(hazardChecks > 1000)` where it is 2,121;
`assert.ok(nodes.length >= 30)` where it is 71; and the 900 px table guard accepts anything under 60 px where the
measured value is 42. Each would let a whole fixture stop being inspected, or a regression grow by 40 %, without
tripping.

### O5. Smoothing renders as "1" in one sentence and "1.0" everywhere else

`hidden-markov-models-hmm.jsx:282` renders "smoothing 1 beats 0.1" because `selected.smoothing` is the number `1`,
while the tagging table, the lab's selector and the identification task all say "1.0". Cosmetic, one word, but it is
in the sentence that states the robustness conclusion.

### O6. `design.md` calls the 2-simplex a "three-simplex"

Line 131: "231 priors on a twentieths grid of the three-simplex". 231 = C(22,2) is the count of lattice points on
the twentieths grid of the **2**-simplex — the probability simplex on three states. The count is right and the sweep
is real; only the name is loose.

### O7. Where the lesson is notably better than its brief

Three things are worth recording as strengths rather than findings, because each is a place a lesson of this kind
usually fails.

The **tie treatment** is the best-handled piece of numerical honesty I have reviewed in this series. The lesson does
not bury a floating-point artefact behind a single number; it finds an exact structural degeneracy, teaches the
mechanism correctly (identical multisets of factors), reports a band with both endpoints, separates the robust
conclusion from the fragile margin, and carries that qualification into the practice solution and the lab's own
sentence-level note. My independent exact arithmetic agreed with all six band members, the three sentence
identities, the fourteen adjacent unknown pairs and the robustness claim.

The **prediction contract** is implemented once, in shared scaffolding, with the hard cases handled: a null whose
previous verdict would be the answer is withheld rather than shown; a numeric guess against an undefined quantity is
told it is "not a near miss, and not zero" rather than scored; and the move tolerance (1e-12) is deliberately tied
to the printed precision (12 places), so "does not move" is a claim a learner can read off two printed numbers.

The **grading rules live in the verified model layer**, not in render functions, which is what made A10 possible at
all — I could exercise every graded rule over degenerate inputs directly, without driving a UI.

---

# Verdict

Recomputing the entire lesson from first principles — an exact-rational engine that uses neither scaling nor
logarithms, a 60-digit EM re-derivation, a 146-symbol exact Viterbi with full tie-set enumeration, an independent
leaf count of the trust root, a real execution of the optional program in the isolated venv, and my own
attribute-versus-paint sweep at four widths — **I found no disagreement with any published number**. Not in the
manuscript, the lesson body, the model layer, the generated data, the figures, the served assets, the attribution,
or the packet's own trust root. The tie finding, which was the thing to adjudicate, is **correct in all three of its
parts**, and the robustness half of it is true for a stronger reason than the record gives. The verifiers are green,
reproducible and, where I could check them, honest about their own limits; the accepted 900 px item is recorded to
the pixel and I accept the judgement.

The defects are in prose and in one encoding. **B1** is a single word that asserts a result on the held-out split, in
a lesson that makes split discipline a teaching point and repeats the opposite three times — fix that before
anything else. **S1–S3** are three more places where what reaches the reader is not what the model computed: raw
LaTeX in the query table, a forecast sentence that opens with a number that forecasts nothing, and a line count that
is off by one on the page and off by twenty-five in the record. **S4** is the one drawing defect, and it is the
lesson's own theme inverted: a trellis edge that carries nothing is painted wider than one that carries almost
nothing, reachable in four interactions, and invisible to every guard including the new one. **S5** is the third
draft of a caption that still quantifies the wrong thing. **S6–S9** are the record and the harness: three leaves
counted as re-derived without being asserted, four stale Phase A claims, two unit-mislabelled measurements, and
roughly 750 assertions inside the advertised grids that nothing could break.

Not complete. Fix B1, then S1–S5; S6–S9 are corrections to the record and the harness rather than to the lesson.
