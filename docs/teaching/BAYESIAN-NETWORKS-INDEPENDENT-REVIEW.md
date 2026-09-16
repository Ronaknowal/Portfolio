# Bayesian Networks & Causal Graphical Models — independent phase-two review

Reviewed 16 September 2026 against the working tree plus the uncommitted Bayesian-networks implementation. The
reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/`,
`docs/teaching/drafts/` or `docs/teaching/topic-notes/` was edited; this review document is the only write outside
`scratch/`. Throwaway scripts live under `scratch/bn-independent/` and `scratch/bn-reviewer2/`. No state-changing git
command was run. The build directory `dist-bn/` was deleted and the preview on 4191 stopped at the end.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/bn-independent/`, importing neither `bayesnet-models.js` nor `bayesnet-data.js` nor `bayesnet-examples.js` nor any `verify-bayesnet-*` script nor the packet's `network-experiments.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.1, scikit-learn 1.9.1, mpmath 1.3.0). **94 exact-rational checks** of every finite constructed number the manuscript, the visual specification, the practice solutions and the lesson body state, in `fractions.Fraction` throughout — zero failures. **357,634 d-separation cases** decided three independent ways — zero disagreements. A full independent refit of the wine pipeline from the **served** CSV. A reviewer-written Playwright drive (`drive.cjs`, `drive2.cjs`, `geom2.cjs`, `play.cjs`, `play2.cjs`, `i3text.cjs`, `text*.cjs`) sharing no code with `verify-bayesnet-browser.cjs`, at 1366, 768, 390 and 320 px on Edge 153. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `calculated-inputs.json`, `network-experiments.py`, `pgmpy-example.py`); the destination note; the lesson body, model layer, both generated modules, all three lab/figure/shared components and the CSS; all four verifiers line by line; the blueprint; `LESSON-TEACHING-STANDARD.md`; the three served assets and the attribution. |
| **Reused (declared)** | scikit-learn's `train_test_split` at the declared seeds, because the protocol *is* defined as that call — as the builder states. NumPy/pandas as general libraries. Two subagents for fan-out: verifier re-runs plus environment inspection, and a verifier-internals audit. **Every finding below that a subagent surfaced, I re-derived myself before reporting it** — I read each cited line in the source, and where the claim was empirical I measured it in the live page (S2 in particular). I also corrected one framing a subagent got wrong: see “A note on the 12-versus-13 count”. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx` | `ac5b590f0b228a564dc9cda00a3a328ab291b16bc20176c48000fd30f7c9cf9b` |
| `src/learn/data/bayesnet-models.js` | `13ab059bd83b60fe53b63d5731819ab803b42eed5c041f23c6bb8fc48e6a63ea` |
| `src/learn/data/bayesnet-data.js` | `e96a76129b32af838638a55bc9f6583134478b075e369f51e508e83c5bff40cc` |
| `src/learn/data/bayesnet-examples.js` | `0742ff92b949cb6b6a35cb9ef43cd1ae2eb7026a4d04f591d977372ffad114fe` |
| `src/learn/components/lesson-labs/BayesNetShared.jsx` | `9b716e3273ae21f81c963b21f7cbd927b10f708443df4b29df3ed1973c07a467` |
| `src/learn/components/lesson-labs/BayesNetLabs.jsx` | `003851e93b0981a9f74822b920a0a75f481e36f61e8dbbfb064f4253a8d1d8a6` |
| `src/learn/components/lesson-labs/BayesNetFigures.jsx` | `b62daebd52b3c7ad82b1fda8be93963251fc478d9f8f078e5b24cc56f8df5d9e` |
| `src/learn/components/lesson-labs/bayesnet-labs.css` | `db6c6706572f21428d02516fabfaa9f4cfa00c1f5408674eaedf6d459da9e5d1` |
| `docs/teaching/drafts/.../lesson.md` | `6a475928fa46848bc0c427344440995165d0bb764943218fd90a8d166b8922c4` |
| `docs/teaching/drafts/.../design.md` | `1272621682c206955a4fbeea921c73f63804024f56020b40085ff1da3a7c1f6a` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `774d8b14ae9de8bdea51f18a8456c408dc98c768320d45307dd71e0f932f07b6` |
| `docs/teaching/topic-notes/bayesian-networks-causal-graphical-models.md` | `b3add0ce2703b678c9b3018b7c1237576ecc612bcaf5d6074cc36369840ec79a` |

---

# Part A — correctness

## A1. Exact recomputation of every finite construction — **no disagreement**

`scratch/bn-independent/exact.py`. Exact rational arithmetic throughout; nothing compared at floating tolerance where
a rational answer exists. **94 checks, zero failures.**

- **§1.** The assembled world is exactly `5910156/10^10`. The parameter counts are exact: 32 joint entries, 31 free,
  20 stored, 10 free, and the general `q_i(r_i−1)` rule reproduces each row.
- **§2, all seven published evidence sets.** Posteriors, evidence probabilities and compatible-world counts, checked
  twice — once by enumerating 32 worlds, once against the closed rational form. The numerator is exactly
  `59224259/10^11`, the denominator exactly `2084100239/10^12`. Compatible-world counts are exactly `2^(5−|e|)` for
  every one of the seven. The screening-off claim is **exact**: `P(B=1|A=1)` and `P(B=1|A=1,J=1)` are the same
  rational number, not two numbers that happen to round alike.
- **§3 elimination.** `g(B,A)` = 499211/500000, 789/500000, 2999/50000, 47001/50000 — the four printed values
  exactly. `h(1)` = `59224259/10^11`, `h(0)` = `1491857649/10^12`, and `h(1)/(h(0)+h(1))` is **the same rational** as
  the enumeration posterior. The claim that elimination only reorders the same sums is demonstrably exact.
- **§6 service model.** Observed 0.04 and 0.1625, interventional 0.055 and 0.125, association 0.1225, causal 0.07,
  bias exactly 0.0525; serviced high-load share exactly 3/4, unserviced exactly 1/3. Randomised [.4,.4] makes
  observed and causal *exactly equal*. Changed response: causal 0.05, observed 0.1125, intervention 0.105,
  observed-serviced 0.1525. No overlap: observed risks exactly [0.01, 0.20], association 0.19, causal still 0.07.
- **§7 frontdoor.** `P(X=0)=P(X=1)=1/2`; the four `P(Y=1|M,X)` cells exactly 12/100, 33/100, 58/100, 82/100; inner
  averages exactly 0.225 and 0.700; mediator weights exactly [.9,.1] and [.1,.9]; estimates exactly 0.2725 and
  0.6525. **The frontdoor formula and the truncated factorisation are equal as rationals**, not to 1e-12 — the
  stronger form of the claim the figure makes. Observational 0.166 and 0.771 exactly, so the page's "gap of 0.605
  against the causal 0.38" is exact.
- **§7 counterfactuals.** Both models' column averages exactly 1/2; model A's rows (0,0) and (1,1), B's (0,1) and
  (1,0); both abduction/action/prediction pairs check out.
- **§8 query families.** MPE at Q=1 with mass 0.39 against a marginal-MAP of Q=0 at 0.60; the practice variation's
  MPE (1,0) and marginal-MAP Q=1 at 0.55 versus 0.45. Rejection retention exactly 0.002084100239 — 208.41 of 100,000.
- **Practices.** Practice 1's two nulls are *exact*. Practice 3's 14/24/23. Practice 4's exact `74/113`. Practice 7's
  0.105/0.055/0.05 and 0.1525/0.04/0.1125.
- **Investigation fixtures.** The false-call edit gives 0.1654409766448569; the unused-row null with `A` fixed at 1 is
  *exactly* unchanged; equal John rows return exactly the prior; zero John rows give an undefined posterior.

I found **no disagreement with any number** in the manuscript, the visual specification, the practice solutions or
the rendered lesson body. Every value I read off the rendered page matched too.

## A2. The graphical claims, by three independent routes — **no disagreement**

`scratch/bn-independent/dsep.py`. The builder verifies by ancestral moralisation because that never enumerates a
path. I used **three** routes, all written from the definitions: (1) explicit simple-path enumeration with
path-local collider classification and the `An(Z) ∪ Z` rule; (2) **Bayes-ball / Shachter reachability**, a two-phase
(node, direction) BFS that never materialises a path at all; (3) ancestral moralisation.

Applied to all nine graphs the page can draw (526 cases across every endpoint pair and every subset of the remaining
nodes) and to **900 generated random DAGs with randomised topological labelling (357,108 further cases)**:
**357,634 cases, zero disagreements between the three routes.** Colliders, descendants of colliders and the empty
conditioning set were all exercised — on the alarm graph the empty set is the *only* set that separates `B` and `E`,
while `A`, `J` and `M` each open it, exactly as the body claims.

The specific claims check out individually: `B–A–E` blocked with nothing observed, open under `A`, open under `J`
alone by the collider-descendant rule; in the second-route graph `{K}` blocks both routes, `{J}` leaves both active,
`{K,J}` leaves exactly the collider route active — the three contrasts §4 promises; in the collider-chain graph `R`
and `T` are separated with nothing observed and open under `W`, `S` and `{S,W}`, which is practice 2's answer. The
**Markov blanket of `B` is `{A,E}`**, independently computed.

**The frontdoor conditions check out on all three graphs.** In the base graph the only directed `X`→`Y` path is
`X→M→Y`; the only `X`-to-`M` backdoor path is `X←U→Y←M`, blocked at the unobserved collider `Y`; the only `M`-to-`Y`
backdoor path is `M←X←U→Y`, blocked at `X`. Adding `U→M` opens exactly **one** backdoor path; adding `X→Y` leaves
the mediator on exactly **one of two** directed paths. Those are the counts the body and practice 9 print.

## A3. The wine pipeline, refit from the served CSV — **no disagreement**

`scratch/bn-independent/wine.py`, written from the manuscript's stated protocol, reading
`public/learn-assets/bayesian-networks/wine.csv` (not the packet copy) and reusing only `train_test_split`.

Reproduced exactly: 178 rows × 15 columns, all distinct, class counts 59/71/48; splits 106/36/36, disjoint; training
medians `[13.05, 1.90, 2.11, 4.75]`; all six conditional-mutual-information weights; the maximum-weight spanning
tree, giving **alcohol as the feature parent of all three other measurements**; validation NB 34/36 at
**0.16292244894704044** and TAN 34/36 at **0.18523288599124255**; validation prior 14/36 at 1.0898523145570962; the
selection of NB; the refit test score **33/36 at 0.2709439464482761**; and the development prior 14/36 at
1.0896162218694347. Figure 4's two column means reproduce the two recorded validation log losses to fifteen decimals,
and I confirmed independently that **both models misclassify the same two specimens** (ids 26 and 155), so the page's
"accuracy ties and log loss does not" framing is accurate rather than convenient.

Specimen 156's five published posterior rows match the visual specification and the page exactly, as do specimen 80's
three. The three alcohol edits behave exactly as promised: crossing the 13.05 median moves the posterior; an edit
within the same bin is an **exact** null; editing a hidden value is an exact null. With nothing revealed the
posterior is exactly the class prior.

**Test isolation is real.** All 36 specimens exported to the browser are in the validation split, none in the test
split, and the three splits are disjoint. The page's claim that the reserved test specimens are "never reachable from
any investigation" is true as implemented.

## A4. The destination note's resolution — **correct, and the body does state it**

I re-derived the note's claim from my own path enumerator rather than accepting either the note or the builder.

In `S→E, S→Y, E→T, T→Y` there are exactly **two** `T`–`Y` paths: `T→Y` and `T←E←S→Y`. Only the second is a backdoor
path. Its interior nodes `E` (chain) and `S` (fork) are both non-colliders; `T`'s only descendant is `Y`, the query
outcome. Therefore:

| Set | descendant violation | open backdoor path | valid |
| --- | --- | --- | --- |
| ∅ | none | `T–E–S–Y` | **no** |
| {E} | none | none | **yes** |
| {S} | none | none | **yes** |
| {E,S} | none | none | **yes** |

Exactly the resolution's claim. With `E→Y` added, a second backdoor route `T←E→Y` appears which `S` does not touch,
so `{S}` becomes invalid while `{E}` and `{E,S}` remain valid — practice 5's answer, also confirmed.

**The body does state it, not merely imply it.** I read the rendered page: §6 prints a four-row table whose first row
is labelled "the empty set" with "Backdoor path blocked? no / Valid: no", and rows for `{E}`, `{S}` and `{E, S}` all
"yes". The prose adds that conditioning on `E` opens nothing because the graph's only collider, `Y`, is an endpoint
rather than an interior node, and that "being earlier in the drawing cannot" distinguish valid sets. **The resolution
does not overstate what the body does.**

**The §9.3 qualification is also correct.** The claim "no observed set can block those paths" is true exactly when
the unobserved variable is a *direct* common parent: the path `X←U→Y` then has one interior node, the unobservable
fork `U`, which no observed set can block. Routed through an observed non-collider it is blockable, so the original
sentence was indeed too broad. The note's second concern was aimed slightly wide, as recorded. The replacement
callout is on the rendered page and states the corrected claim in both directions. I agree with the disposition as
written.

## A5. The trust root — the mechanism is real, not a tautology

The claim is 1,236 of 1,236 leaf paths, "measured as paths its comparison actually visited, with the run failing
unless the covered set equals the complete set". I read the code rather than the claim.

`scripts/verify-bayesnet-data.py:96` derives `all_leaves` by walking **the packet file**. `covered` is populated in
exactly one place, line 130, at the point where a **recomputed** scalar is about to be compared with the packet's —
inside `compare_tree`, reachable only with a value the verifier produced itself, and which bails out without
recording on any key-set or length mismatch. Line 551 fails the run unless `covered == all_leaves`. **This is a
genuine measurement, not a walk of the same tree twice.** Re-running reproduces 1,270 checks and 1,236/1,236, and an
injected perturbation produces exactly one new failure, so the comparison is discriminating.

The second separation algorithm is likewise real: `scripts/verify-bayesnet-models.mjs:72` is textbook ancestral
moralisation and shares no code with `simplePaths`/`pathStatus` (it borrows only `ancestorsOf`, which the path
enumerator does not use — that uses `descendantsOf`). Mutation-testing it confirms it discriminates: removing the
collider-descendant clause from the module produces 9 disagreements in 454 cases, and removing the collider rule
entirely produces 19.

**What remains genuinely unchecked.** The four declared exclusions — browser rendering/layout/keyboard, the split
itself, per-path blocked flags, and investigation 3's tree-augmented-only sweep — are all real and correctly
characterised. I add five the record does not name, each detailed below:

- The generated-graph sweep does **not** require both verdicts; only the 1,004 preset cases do (O3).
- Edge clearance is asserted on a curve the browser does not draw, at a threshold below the router's own, and against
  node circles rather than the larger drawn badges (S3).
- The investigation-3 leak guard cannot fire in its text half (S2).
- The escape auditor is not in the repository, so four of its claims cannot be re-run at all (S1).
- "Nothing revealed on first paint" tests only for the absence of the verdict banner (B1).

With those five added, the exclusion list would be honest and complete. As written it is honest but not complete.

## A6. Served data and programs — **no disagreement**

- `wine.csv` is **byte-identical** to the packet's copy (`cmp` clean; both SHA-256 `34ced17c…e818be`, 12,100 bytes),
  matching the value printed on the page and in `ATTRIBUTION.txt`.
- The attribution's every factual claim checks out against the file I parsed: 178 data rows, 15 columns, all rows
  distinct, cultivar counts 59/71/48, CRLF line endings (179 CRLF, 0 bare LF), and the four quoted ranges
  (alcohol 11.03–14.83, malic acid 0.74–5.80, flavanoids 0.34–5.08, colour intensity 1.28–13.00) are exactly the
  minima and maxima. Source, DOI, creator and CC BY 4.0 are stated, and the honest limit — "no fresh byte comparison
  against the original UCI archive is claimed" — is stated too.
- Provenance matches what the pipeline does: seeds, sizes, strictly-above-median binarisation, one pseudo-count per
  CPT state including the class prior, the MI/MST/feature-0-root construction, the selection rule and the single test
  assessment are all exactly what I reproduced.
- **The served `network-experiments.py` is byte-identical to the packet's and to the code the examples module pins
  and executed.** I verified all three program records by parsing `bayesnet-examples.js` as text: `enumerate` is
  byte-identical to the manuscript's own fenced block, `pgmpy` to the packet's `pgmpy-example.py`, and `experiment`
  to both the packet's and the served `network-experiments.py`. The pinning is genuine.

## A7. The pgmpy environment claims — **true**

The shared runtime is undisturbed: `scratch/lesson-tools` resolves Python 3.12.14, **NumPy 2.3.5, pandas 3.0.1,
scikit-learn 1.9.1, SciPy 1.18.1**, and `import pgmpy` fails there. The isolated environment
`scratch/bayesnet-optional/` exists, is built with `include-system-site-packages = false`, and carries
**pgmpy 1.1.2, NumPy 2.5.3, pandas 3.0.5**, networkx 3.6.1, SciPy 1.18.1, scikit-learn 1.9.1 — matching
`docs/teaching/evidence/bayesnet-native.json` and the versions the page names. The divergence the page cites as its
reason for isolating is real.

The displayed output `[0.71582816 0.28417184]` is what that program prints, and its second state agrees with my
exact-rational posterior `0.28417183536439294` to the last digit NumPy shows. The page's sentence is accurate, the
manuscript's "has not been executed in the shared environment" remains literally true, and the declared departure is
justified and correctly reported. A missing isolated environment is a hard failure in the verifier, not a skip.

---

# Part B — the rendered page

Built with `npx vite build --outDir dist-bn`, previewed on `127.0.0.1:4191`, driven with my own Playwright scripts on
Edge 153 at 1366, 768, 390 and 320 px, with every screenshot opened and looked at.

**The page renders, with zero console errors and zero page errors** across every drive. Phase C's defect 1 — the
lesson not rendering at all — is genuinely fixed, and I could not provoke a render failure through any control,
including impossible evidence, an empty treatment group, a cycle-creating edge and removing a query endpoint.

**Phase C's other five defects are confirmed fixed in the images**, not merely in the code:

- *Defect 2 (edge through two nodes).* Figure 5's `U→Y` edge now bows clear of `X` and `M`; at 6× magnification it
  passes outside both circles and outside both node labels. (It does clip `X`'s endpoint badge — S3.)
- *Defect 3 (24 px sideways scroll at 768).* `scrollWidth === clientWidth` at all four widths; zero elements extend
  past the viewport once the hidden KaTeX MathML mirror and declared scroll boxes are excluded.
- *Defect 4 (two equations overflowing at 320).* Zero `.katex-display` blocks overflow their own box at any width.
- *Defect 5 (frontdoor outer tray clipped).* The outer tray spans the full figure; both identification columns —
  "frontdoor formula" 0.272500 and "truncated factorisation" 0.272500 — are visible side by side.
- *Defect 6 (paired-outcome column clipped).* Figure 6's "Y if X = 1" column is fully visible in both panels; rows
  (0,0)/(1,1) and (0,1)/(1,0) with both averages 0.5.
- *Defect 7 (investigation 3 prefilling hidden values).* Fixed in the page: on first paint all four number inputs
  have `value === ""` and all four cards read "not revealed". The `blind` field still publishes edits, so the
  hidden-edit null works. (The *guard* written to keep it fixed is half dead — S2.)

Departure 3 is confirmed in the image: figure 3's right panel draws **both** dashed fill arcs, `A–C` over one span
and `A–D` over three, at different heights and clearly separable, captioned "Fill edges: A–C, A–D". I re-derived the
induced widths independently — endpoint-first width 1 (4 binary cells), middle-first width 2 (8 cells), fill edges
`A–C` then `A–D` — and they match.

No SVG label renders under 8 CSS pixels at any width; no scrollable table clips a column at 1366 px.

**Grading is correct at every degenerate input I drove.** Screening-off (`A=1` then `J=1`) graded "exactly
unchanged" with both sides printing 0.37355123; impossible evidence graded "no posterior at all" with "the evidence
probability is exactly 0"; the unused-row null graded exactly unchanged; the collider-descendant case graded
"dependence possible" with the right reason; investigation 4's randomised, no-overlap and changed-response presets
graded falls/rises/falls on the observed difference (0.1225 → 0.07, 0.19, 0.1125) and "exactly unchanged" on the
causal difference, matching my exact arithmetic; investigation 3's purchase comparison returned 0.894394 nats for
alcohol against 0.812855 for flavanoids, which I reproduced independently to nine decimals. **I did not find a single
case where the page grades a correct answer wrong.**

---

# Blocking

## B1 — Three of the four investigations print the graded quantity before the prediction is recorded, and one does it on first paint

**Where.** `src/learn/components/lesson-labs/BayesNetLabs.jsx:164` and `:179` (investigation 1), `:322` (investigation
2), `:635` and `:689` (investigation 4). The guard meant to catch this is `scripts/verify-bayesnet-browser.cjs:315-316`.

**What is wrong.** The lesson's own contract, printed on screen inside every `Prediction`, is *"The answer appears
once a prediction is recorded."* Investigation 2's prompt goes further: *"Before revealing the verdict…"*. Both
sentences are false.

- **Investigation 2 is the clearest case, and it leaks on first paint.** `report` is computed from the **draft**
  (`:225`) and rendered by `<PathList report={report} />` at `:322`, *above* the prediction fieldset. I loaded the
  page and read it before touching anything:

  > × B–A–E **blocked** — the collider A is unobserved and no descendant of it is observed.

  and the drawing's caption underneath: *"The thick trail follows B–A–E, which is blocked."* The prediction below
  then offers "the graph guarantees independence — **every path is blocked**" against "the graph leaves dependence
  possible — at least one path stays active". The correct option is selectable by matching one word. Ticking the "A"
  observation box before predicting flips the list to "→ B–A–E **active** — the collider A is observed, which opens
  it", again before any prediction. Screenshots: `scratch/bn-independent/shots/i2-first-paint.png` and
  `i2-after-observe-A-before-predict.png`.

- **Investigation 1** computes `shown` from the draft at `:60` and prints it twice before grading: a three-row mass
  table at `:164` and a closing caption at `:179`. After clicking the suggested setup "Reveal that there was an
  earthquake" and before recording anything, the caption read:

  > Posterior for the state in the fields: **0.003262**. Currently applied: **0.284172**.

  The prediction asks whether the posterior will be higher, lower, exactly unchanged, or undefined *relative to the
  applied state*. Both numbers are on screen.

- **Investigation 4** computes `result` from the draft at `:571`, prints the table at `:635`, and prints the applied
  values at `:689`. After the "Assign the procedure at random" preset and before predicting, the table showed
  observed difference **0.070000** and causal **0.070000** while the caption showed "Applied state: observed
  difference **0.1225**, causal difference **0.07**". Both graded questions — falls, and exactly unchanged — are
  answered in adjacent text. Screenshot: `i4-after-setup-before-predict.png`.

Investigation 3 is clean for its purchase question: the distribution shown is for the currently revealed set, not for
either candidate, and the blind fields work.

**Why it matters.** `LESSON-TEACHING-STANDARD.md:316` requires that "a learner should have a chance to try before
seeing the answer", and `:234` requires the prediction to be recorded *before acting*. Here the recorded prediction
is a formality: the page has computed and displayed the answer next to the control that asks for it. This is the
dominant defect class for this curriculum, and investigation 2 — the one the design record singles out as "the reason
this file exists", because "a lab that grades correct reasoning wrong is the easiest defect to ship here" — is the
worst affected.

**Why the guard did not fire.** `verify-bayesnet-browser.cjs:315-316` is `assert.equal(await
page.locator('.bn-verdict').count(), 0, …)`, which asserts only that no *verdict banner* has rendered. It says
nothing about whether the graded quantity is displayed elsewhere, and it runs only at first paint, so it would not
catch investigations 1 and 4 even if broadened. The case is nevertheless recorded as "First paint reveals no verdict
and no history…" (`:323`) and design.md reports it as "nothing is revealed on first paint". The record is stronger
than the check.

**How I verified it.** Reviewer-written Playwright drive (`scratch/bn-independent/drive.cjs`), reading each
investigation's rendered text and screenshotting it before any prediction was recorded; then reading the three source
sites; then re-reading the browser verifier's guard. The leak is in visible text, not only in an `aria-label`.

**Scope note.** I am not asking for the workbench numbers to disappear. The contract is satisfied by gating the
*draft* readouts behind the commit — the labs already keep `active` for exactly this purpose — while continuing to
show the applied state; or, for investigation 2, by showing the path skeleton and collider annotations without the
per-path blocked/active verdict until Apply.

---

# Should-fix

## S1 — The "escape auditor" is reported as a passing verifier but does not exist in the repository

**Where.** `design.md:266` and `:444` report it as a PASS row in both verification tables ("14 files … now also
refusing CRLF and dead conditionals"); `:345` states that the dead-conditional guard "was confirmed to fire on
exactly the expression that was removed" — which is the thirteenth of the "13 of 13" falsification breakages.

**What is wrong.** There is no such script. `scripts/` contains only the four `verify-bayesnet-*` files; the three
`scripts/audit-*` scripts are lesson-structure, visual-layout and working-artifacts tools that mention neither
escapes nor CRLF; and `verify-bayesnet-models.mjs` performs no escape, KaTeX, CRLF or conditional auditing (its file
list at `:1230-1242` is twelve sources, used only for existence and hashing). Ad-hoc escape auditors from *other*
lessons do survive in `scratch/` (`scratch/bias-variance/escape-audit.mjs`, `scratch/hmm-phase-two/escape-audit.py`,
`scratch/imb-audit.mjs`), and none of them carries either of the two new guards — so the bayesnet auditor was written
and discarded. The "Files created" table correctly omits it, but a table row reading **PASS** beside four runnable
verifiers implies a re-runnable check, and one of its guards is counted in the headline falsification result.

**Why it matters.** Four claims — 14 files clean, every display-math block wrapped, CRLF refused, dead conditionals
refused — rest on an artifact nobody can execute, and the "13 of 13" count cannot be independently reproduced.

**What I could confirm, and what I could not.** I wrote my own scan for JSX conditionals with identical branches
across all eight lesson sources and found **zero**, so the defect the guard was written for is genuinely gone. I also
confirmed the CRLF claim is a non-issue: this repository has `core.autocrlf=true`, every long-tracked sibling file
(`bias-variance-models.js`, `BiasVarianceLabs.jsx`, …) is equally CRLF in the working tree, and the convention is
about stored bytes. The underlying *states* are fine. What I could not do is confirm either guard **fires**, which is
what the brief asked and what the record asserts.

## S2 — The guard protecting investigation 3 from re-leaking its measurements cannot fire in its text half

**Where.** `scripts/verify-bayesnet-browser.cjs:389-390`.

**What is wrong.** The builder found, in Phase C, that investigation 3 prefilled each hidden measurement's edit box
with its own value — its own instance of the dominant defect class. The guard added to stop it recurring is:

```js
assert.ok(!inputValues.includes(printed), `… is prefilled into an edit control`);
assert.ok(!visibleText.includes(` ${printed} `), `… is printed in the panel`);
```

The second assertion searches for the value **surrounded by spaces**. The card that would print it renders as
adjacent inline elements with no separating text (`BayesNetLabs.jsx:422-431`), so JSX emits no space on either side.
I measured this in the live page: with alcohol revealed, the card's `textContent` is

```
"alcohol13.17training median 13.05 — this specimen is above itreveal this measurementedit alcohol"
```

so `raw.includes("13.17")` is **true** while `normalised.includes(" 13.17 ")` is **false**. The guard passes today
for the right reason — nothing is printed while hidden — but it would *also* pass if the value were printed. The text
half cannot fire.

The `inputValues` half **is** live: `NumberField` renders `shown = draft === null ? (blind ? '' : String(value)) :
draft` (`BayesNetShared.jsx:110`), and the browser compares the identical `String(measured)`, so removing `blind`
would be caught. So the regression is half-guarded, not unguarded.

**Why it matters.** This is the specific defect this lesson already shipped once, and the browser case that claims to
prevent it — "Investigation 3 reveals none of the four measured values before they are bought" — is written into the
evidence file as passing. Dropping the spaces (or matching on the card's own `<b>` text) restores it.

**How I verified it.** Read the assertion; then reproduced the DOM text in the live page with
`scratch/bn-independent/i3text.cjs`, revealing a measurement and printing both `includes` results.

## S3 — The edge-clearance guard samples a curve the browser does not draw, at a threshold below the router's own, against a shape smaller than the one drawn

**Where.** `scripts/verify-bayesnet-models.mjs:1084-1092`; `src/learn/data/bayesnet-models.js:963`, `:976-977`,
`:1031-1043`; `src/learn/components/lesson-labs/BayesNetShared.jsx:378`, `:383`.

Three distinct gaps compound in the one guard that exists because Phase C found an edge drawn straight through two
nodes and their labels:

1. **It samples the wrong curve.** The verifier measures `route.trace`, which is the true sub-arc of the quadratic
   over `[startT, endT]`. The browser renders `route.path` (`BayesNetShared.jsx:378`), which is
   `M start Q control.controlX control.controlY base`. But `controlX`/`controlY` in `point(bow, t)`
   (`bayesnet-models.js:976-977`) **do not depend on `t`** — they are the control point of the *full* curve, reused
   for the *trimmed* span. A trimmed quadratic has a different control point, so the drawn curve and the traced curve
   are different curves. Measured divergence reaches **9.0 px**, over half a node radius.
2. **The threshold is below the router's own.** The assertion is `nearest >= layout.radius`, while `edgeRoute`
   selects its bow to achieve `radius + clearance` with `clearance = 3` (`:963`). The check accepts 3 px tighter than
   the router is designed to produce.
3. **It measures against the node circle, not the drawn shape.** The failure message says "it would be drawn through
   that node's label", but the query-endpoint badge is drawn from `point.x − drawnRadius − 5` with width
   `2·drawnRadius + 10` (`BayesNetShared.jsx:383`), and the "observed" tag sits at `y + drawnRadius + 13`. Both are
   outside the guaranteed envelope.

**This bites once today.** In figure 5 the `U→Y` edge passes 25.98 units from `X`'s centre — clear of the circle
(radius 17) by 8.98, so the defect-2 fix is working — but **inside `X`'s endpoint badge**, which it crosses near the
top-left and bottom-left corners. It is visible at normal magnification: the grey dashed arc cuts through the amber
dashed rectangle.

**Why it matters.** No current drawing is unreadable, and the tightest *drawn* clearance I measured is 8.98 px. But
graph diagrams are the most exposed drawing in the hub to the class where every assertion passes and the picture is
still wrong, and design.md states the fix more broadly than it holds: "bow an edge around every node it does not
join", and "the verifier now asserts the clearance on every edge of every drawable graph at five widths". Sampling
`route.path`, raising the threshold to `radius + clearance`, and routing against the drawn badge extent would close
all three.

**How I verified it.** Read `edgeRoute` and the assertion; then measured the **actually rendered** geometry with
`scratch/bn-independent/geom2.cjs`, which samples every `g.bn-edge path` at 200 points via `getPointAtLength`,
identifies each edge's own endpoints by nearest node and excludes them, and computes minimum distance to every other
node's circle, badge rect and label box across all twelve SVGs — one real collision, which I then captured at 6×
(`frontdoor-zoom.png`) and looked at.

## S4 — Investigation 2's "adjustment graph of practice 5" preset has an invariant verdict and cannot check the claim it is offered for

**Where.** `src/learn/components/lesson-labs/BayesNetLabs.jsx:250`, and the §6 cross-reference: *"You can check either
graph yourself in investigation 2 — the adjustment graph is one of its presets."*

**What is wrong.** The preset sets endpoints `T` and `Y` on `S→E, S→Y, E→T, T→Y, E→Y`. `T→Y` is a single edge, and a
path with no interior node can never be blocked. So **all four** reachable observation sets — ∅, {E}, {S}, {E,S} —
return "the graph leaves dependence possible". The same holds for the four-edge graph. The preset cannot exhibit a
contrast, which is the one thing a d-separation preset exists to do; every other preset has both verdicts available
(alarm 8 sets → both, second route 16 → both, collider chain 8 → both).

Worse, the sentence that sends a learner there is about the **backdoor criterion**, and investigation 2 grades
**d-separation**, which includes the causal path the backdoor criterion deliberately excludes. A learner who has read
§6's table saying `{E}` is valid, follows the cross-reference, selects `E`, and predicts "the graph guarantees
independence" is marked **≠**. I drove exactly that: the page replies "You recorded the graph guarantees
independence…; the calculation gives the graph leaves dependence possible… 3 of 3 paths stay active".

**Why it matters.** The mildest form of the dominant defect class: not a wrong grader, but a page that invites a
reasonable learner into a question it does not answer and then marks their reasoning wrong. The verdict is
mathematically right, which is why no verifier caught it — the sweep asserts agreement between two algorithms, not
that a preset can teach anything.

**How I verified it.** Enumerated every observation set for every preset with my own path enumerator
(`scratch/bn-independent/dsep.py`), finding the two education graphs invariant and the other three not; then drove
the sequence in the browser (`scratch/bn-independent/play.cjs`).

## S5 — "about" followed by nine to eleven significant figures

**Where.** `src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx:252` and `:726`.

**What is wrong.** Both sites wrap a percentage in the `num()` helper, which rounds to nine **decimals** — nine to
eleven significant figures for a percentage. The rendered page reads:

> …so the posterior is 0.284171835, about **28.417183536%**.

> …rejection sampling retains about **0.208410024%** of prior samples on average: roughly 208 of 100,000.

The manuscript wrote "about 28.42%" and "about 0.2084%". The second is worse: the sentence's whole point is an order
of magnitude, and it immediately rounds to "roughly 208".

**Why it matters.** A hedge word followed by eleven digits reads as a formatting accident and undercuts a lesson that
is otherwise careful about which digits are meaningful. Purely presentational — the values are exactly right.

**How I verified it.** Read the rendered text off the previewed page; grepped the three `about {num(…)}` sites.

---

# Observations

**O1 — Several assertions in the verifiers cannot fail.** None changes a conclusion, but together they inflate the
headline counts. The clearest, each of which I read in the source:

- `verify-bayesnet-examples.py:205` — `frozen` is read at `:201`, *after* every subprocess has run, and line 205
  re-reads the same file and compares it to that snapshot. "Nothing inside the frozen packet directory was written"
  is asserted as `x == x`, after the only thing that could write it.
- `verify-bayesnet-models.mjs:397-401`, `:403`, `:406-407`, `:464` — restatements of the module's own object literal
  (`bayesnet-models.js:266-269` defines `separated: active.length === 0` and `verdict:` from the same boolean),
  evaluated 1,004 + 124,232 times.
- `verify-bayesnet-models.mjs:363` — `close(2 ** 31 * 8 / 2 ** 30, 16, …)` contains no symbol from the module.
- `verify-bayesnet-models.mjs:1082` — `assert(spread > 4)` where `spread` is identically `arrowWidth = 5.5`, a
  default parameter. The tangent direction its comment describes is not checked anywhere.
- `verify-bayesnet-models.mjs:284` — `nullMargin / unchangedTolerance > 1e6` is bit-for-bit equivalent to line 282's
  `nullMargin > 1e-6`, since `unchangedTolerance` is `1e-12`.
- `verify-bayesnet-examples.py:189` — entailed by the two oracles above it, which already pin the sum to 1 and the
  burglary state to 0.28417183536.

**O2 — Counters are reported but never floored.** `verify-bayesnet-models.mjs`'s `record()`/`counts` (504 checks, 58
groups), `verify-bayesnet-data.py:75,558` (`checks["count"]`, `root_checks` = 25 property statements) and
`verify-bayesnet-examples.py:91` (`oracle_count`) are all printed and written to evidence with no minimum asserted.
Deleting every `record()` call, or all 25 `root(...)` statements, still prints PASS. The counts are honest today —
I re-ran all three and reproduced them exactly — but nothing defends them.

**O3 — The generated-graph sweep does not require both verdicts.** `verify-bayesnet-models.mjs:412` increments
`separatedCases`/`connectedCases` inside the **presets** loop only; the generated loop (`:450-469`) increments only
`generatedCases`. Line 477's `assert(separatedCases > 50 && connectedCases > 50)` therefore constrains the 1,004
preset cases alone, and a change that made every one of the 124,232 generated cases come out "separated" would still
pass. Both verdicts do occur on generated graphs in fact; nothing asserts it. The console line's "(240 separated, 764
not among the presets)" is correctly scoped to the presets, so the report is accurate even though the guard is not.

**O4 — Two skip branches in the browser verifier reference classes that exist nowhere.** `bn-halo`
(`verify-bayesnet-browser.cjs:72`, `:96`) has exactly one occurrence in the whole tree — that line itself; the prefix
appears to be carried over from the AutoML lesson's `am-halo`. `HALO_ALLOWANCE = 0.02` and its documented "a
backplate may be crossed briefly" exemption are therefore dead. `bn-grid` (`:77`) is styled at
`bayesnet-labs.css:54` but applied to no element, so that skip is dead too. Both make the guard *stricter* rather
than weaker, so nothing is wrong on the page — but the documented exemption does not exist. The third skip,
`bn-underlay`, is real (`BayesNetShared.jsx:375`) and is the one the underlay-order check earns.

**O5 — The keyboard check computes an outline style and never asserts it.** `verify-bayesnet-browser.cjs:462-468`
returns `{ tag, outline: style.outlineStyle }` and then asserts only `assert.ok(focusVisible, 'tabbing moves focus
off the body')`. An `outlineStyle` of `'none'` passes, and line 469 then writes "the focused control has a visible
outline style" into the evidence file. The record claims more than the check tests. (Focus is in fact visible on the
page — I tabbed through it — so this is a record-accuracy point, not a page defect.)

**O6 — The typeable-value sweep checks four values against a control that does not exist.**
`verify-bayesnet-models.mjs:1009` adds all four alarm-CPT rows to the sweep as "an alarm row" at `decimals: 3`, but
the page has no alarm-row editor: the twelve `NumberField`s in `BayesNetLabs.jsx` are two priors (4 dp), four caller
rows (3 dp), four measurement edits (2 dp, 0–100) and six service fields (3 dp). `withAlarmRow` and `withRootChance`
are exported by the model and called only by the verifier's own `refuses(...)` cases. More generally every
`decimals`/`min`/`max` in `typeable` is hard-coded in the verifier rather than read from the component, so a control
whose `decimals` changed would not be caught by the check written to catch exactly that.

**O7 — Investigation 3 lets a learner pick the same measurement as both candidates, producing two identical
options.** `BayesNetLabs.jsx:481` and `:484` both offer the full `hiddenCandidates` list. The prediction then renders
`["alcohol leaves less uncertainty", "alcohol leaves less uncertainty", "they leave the same uncertainty"]`. The
grader is right — `purchaseComparison` short-circuits to `equal` and explains "The same measurement was chosen twice,
so the two options are identical by construction" — so no correct answer is marked wrong. But two indistinguishable
radio labels is a state the UI should not permit; excluding the first select's value from the second's options would
remove it. Verified by driving it. Relatedly, the 864-case purchase sweep never produces the `equal` outcome
(measured `{first: 396, second: 468, equal: 0}`); line 997 covers `equal` separately with an
equal-by-construction pair.

**O8 — There is an exact tie in the tree's edge weights, benign but unexercised.** On the 106 training specimens,
`I(alcohol; flavanoids | C)` and `I(flavanoids; colour intensity | C)` are equal to every printed digit and their
float difference is exactly `0.0`, although their exact rational term multisets differ — a genuine floating-point
tie, not an algebraic identity. Kruskal's declared `(−weight, a, b)` ordering resolves it deterministically, and the
outcome does not depend on it because the loser would close a cycle regardless. Worth recording because the
provenance promises "deterministic weight/index order" and nothing exercises a tie that *would* change the tree.

**O9 — `verify-bayesnet-models.mjs` writes its evidence file unconditionally.** Line 1298 writes
`docs/teaching/evidence/bayesnet-models.json` with no `--write` gate, unlike both Python verifiers, which are
read-only by default (`verify-bayesnet-data.py:433`, `verify-bayesnet-examples.py:90`). An independent reviewer
re-running the models verifier mutates a record. The file is untracked, so nothing tracked changed.

**O10 — Investigation 1's optional numeric tolerance accepts 0 for the 0.001 prior.** `BayesNetLabs.jsx:141` sets
`tolerance: 0.005` for a probability. When the graded posterior is the prior 0.001, a learner who types `0` is told
they were "within 0.005". That sits awkwardly beside §2's insistence that a posterior of zero and an undefined
posterior are different things, and beside the same lab's impossible-evidence branch, which handles the undefined
case carefully and correctly. Optional and clearly labelled, so minor.

**O11 — The SVG text alternative is `role="img"` + `aria-label`, not `<title>`/`<desc>`.** The visual specification
asks for "accessible titles/descriptions"; all twelve SVGs carry `role="img"` with a full sentence in `aria-label`
and none carries `<title>` or `<desc>`. Functionally equivalent, and each drawing also has an adjacent text
equivalent as specified — but it is an undeclared deviation from the specification's wording, absent from the
fourteen-item departure list.

---

# Part C — the records against the tree

I read the appended Phase A and Phase C sections of `design.md` and checked every claim I could.

**Accurate.** The baseline preservation (`scratch/bayesnet-baseline/` holds both the working copy and a
`git show HEAD:` extract); the files-created table; the four verifier counts, which I re-ran and which reproduce
exactly (504 grouped checks / 58 groups, 1,004 preset plus 124,232 generated separation cases on 742 graphs, 26
backdoor sets, 864 measurement comparisons, 960 geometry checks; 1,270 data checks with 1,236/1,236 leaves; 3
programs pinned and executed with 17 oracle assertions and a byte-for-byte reproduction of `calculated-inputs.json`);
the browser evidence file's 17 records and 35 screenshots with unique digests; the pgmpy environment versions; the
numerical nuance about `0.37355122828183607` versus `0.373551228281836` and the measured null margin; and all
fourteen declared departures, each of which I checked and each of which I judge justified. Departures 3, 8, 11, 12
and 13 are genuine improvements on the specification rather than conveniences, and departure 1 (running pgmpy) is the
right call, correctly reported on the page.

**The stated limits are honest.** No beginner walkthrough, so the learning-experience assessment is the author's own;
accessibility covering only focus visibility, label association, text equivalents and stacking with no screen-reader
pass; narrow-width inspection covering 768, 390 and 320 px and the named figures rather than every figure at every
width. All three are accurate and none is understated. I did not run a screen reader either, so I can neither confirm
nor extend that one.

**Four claims are stronger than the tree supports**, all raised above: the escape auditor row (S1), "nothing is
revealed on first paint" (B1), "Investigation 3 reveals none of the four measured values before they are bought"
(S2), and "the focused control has a visible outline style" (O5). Two are narrower than they read: the edge-clearance
assertion (S3) and the both-verdicts requirement (O3).

**A note on the 12-versus-13 count.** Phase A says "Twelve breakages" and "12 of 12"; Phase C's summary says "13 of
13". This looks like a contradiction, and one of my subagents reported it as one. It is not: Phase C adds the
dead-conditional guard and records that it "was confirmed to fire on exactly the expression that was removed", which
is the thirteenth. The narrative and the totals are consistent once both phases are read together. I re-derived this
myself rather than accept the subagent's framing. (The CRLF guard, added in the same sentence, is *not* claimed to
have been falsification-tested, and the count is right not to include it.)

---

# Summary

This is a strong implementation. The mathematics is exact and I could not break it: 94 exact-rational checks and
357,634 d-separation cases across three independent algorithms produced **zero** disagreements; the wine pipeline
reproduces from the served CSV to fifteen decimal places; all three programs are byte-identical to their frozen
sources; the pgmpy isolation is real and honestly described; the trust-root coverage mechanism is a genuine
measurement rather than a tautology, and mutation-testing confirms its separation oracle discriminates; the
destination note's resolution is correct and the body genuinely states it; and every grading verdict I could provoke
at a degenerate input was right. Phase C's seven defects are all genuinely fixed, confirmed in the images.

The one blocking finding is a contract failure, not an arithmetic one: three of the four investigations print the
quantity they are about to grade, and investigation 2 prints its verdict before the learner has touched anything,
while the page and the record both assert the opposite. Alongside it, the guard written to stop investigation 3
re-leaking its measurements cannot fire in its text half, and the guard written to stop an edge being drawn through a
node samples a curve the browser does not draw. The arithmetic here is beyond reproach; it is the guards around the
teaching contract, and the records describing them, that need the attention.

**Blocking:** B1. **Should-fix:** S1–S5. **Observations:** O1–O11.
