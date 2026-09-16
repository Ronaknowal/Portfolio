# Imbalanced Learning (SMOTE, Cost-Sensitive Learning) — independent phase-two review

Reviewed 16 September 2026 against the working tree plus the uncommitted imbalanced-learning implementation. The
reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or
`docs/teaching/drafts/` was edited; this review document is the only write outside `scratch/`. Throwaway scripts live
under `scratch/imb-independent/` (and a subagent's transcripts under `scratch/imb-review-agent/`,
`scratch/imb-links-agent/`). No state-changing git command was run. The build directory `dist-imb` was deleted and the
preview server stopped on completion.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/imb-independent/`, importing neither `imbalance-models.js` nor `imbalance-data.js` nor `imbalance-examples.js` nor any `verify-imbalance-*` script nor `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.1, scikit-learn 1.9.1, mpmath 1.3.0, imbalanced-learn 0.14.2). **≈33,285,000 independent constructed checks**: 25,351,101 exact-rational evaluations of investigation 2's decision rule over its entire enterable control grid; 7,920,000 exact-rational evaluations of investigation 3's forward and inverse verdicts over theirs; a from-scratch refit of all five procedures by my own damped Newton on the analytic Hessian, polished to **50 decimal digits** with `mpmath`, and every published study quantity recomputed twice — once from the packet's L-BFGS-B parameters and once from my own optimum; all 1,000 threshold-sweep candidates × 8 recorded fields recomputed; both 200-value score arrays per procedure; 73 exact-rational (`fractions`) checks of every constructed number the manuscript, the practices and the figures state; and a by-hand parse and identity repair of the served `yeast.data`. The generated data module was never imported — the study was rebuilt from the served bytes and compared against the packet, and the module was read as text. |
| **Read** | The frozen packet (`lesson.md`, `visual-specifications.md`, `design.md` including its appended Phase A and Phase C sections, `data-provenance.md`, `calculated-inputs.json`); the lesson body, model layer, generated data and examples modules, all five investigations, the shared scaffolding, the figures module and the CSS; all four verifiers; `LESSON-TEACHING-STANDARD.md`; the served asset and its attribution; the blueprint and its registration. |
| **Drove** | The real production page at `127.0.0.1:4188` in Microsoft Edge via Playwright at 1366, 390 and 320 px, across three scripted passes (`drive.cjs`, `drive2.cjs`, `drive3.cjs`) plus a figure-4 geometry probe. **I opened the images.** |
| **Reused (declared)** | scikit-learn/NumPy/SciPy as *general* libraries — `train_test_split` to reproduce the documented seeded split, and `average_precision_score`/`roc_auc_score` purely as a second opinion against my own exact-rational and pairwise implementations. Not lesson code. Two subagents for fan-out: verifier re-runs, and external-link fetching. Every conclusion resting on their output I re-derived or re-checked myself — I re-ran nothing blind, and the one substantive thing the link agent found (the `EXC` count) I had already found independently from the served file. |

## Source versions reviewed (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/imbalanced-learning-smote-cost-sensitive-learning.jsx` | `5c10bc50794b982aaf4714d122592c5318a04ed4e85940ebced0c72fcf3e9705` |
| `src/learn/data/imbalance-models.js` | `4b2699b7bdc0db33fc13909cbffa56290c764397da4e1939d2c3fab60ac9ddef` |
| `src/learn/data/imbalance-data.js` | `c45a564a2d1f2d135f07ac4d0e9b977e320488ea6579e0c99b9f0ad0c775146f` |
| `src/learn/data/imbalance-examples.js` | `59345b538c693efed7b5950cbfadd2a15e38884a91e76f5a18430f1ad331813d` |
| `src/learn/components/lesson-labs/ImbalanceShared.jsx` | `2a8094bf6ca1c03097cadd023318c29406fe920d55c40fd6077062966fd9fc3f` |
| `src/learn/components/lesson-labs/ImbalanceLabs.jsx` | `ee5c74bffd562887148dbbe4434f23bdacf5b3b86c6d14f5041b2a90c7e58d0f` |
| `src/learn/components/lesson-labs/ImbalanceFigures.jsx` | `7b5015d779d1db8a71fbb5bba8e88eddad1feb8196641c8aeeebad89fbc4be7e` |
| `src/learn/components/lesson-labs/imbalance-labs.css` | `3f6621ef3c42b7c5e3e946390e6497c337d752a6a19f3709a13fa4d6d5f72937` |
| `docs/teaching/drafts/.../lesson.md` | `6ea57a4ce4be9a7fa6c7e07cdfb10439b928a3868da0e8bef633fd6be95d4bae` |
| `docs/teaching/drafts/.../design.md` | `be297708cb48b2c30985f3d0e4fbc36ab111bfffbfe6f7a66444d22e2df05a60` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `1e48991ae587815a31535b9e4e27be678d64a42bc98b26b729df5ed9199fbfc3` |

---

# Part A — correctness

## A1. The identity arrangement holds — **no disagreement**

The brief asked me to verify rather than report the title split. It holds, exactly.

- The catalogue title is unchanged: `blueprints/index.js:150` registers the blueprint under
  `'Imbalanced Learning (SMOTE, Cost-Sensitive Learning)'`, and `docs/curriculum/curriculum-inventory.json` carries the
  same string with `id: imbalanced-learning-smote-cost-sensitive-learning`.
- The stable ID is unchanged and resolves: `lesson-manifest.json:149` maps it to the topic file.
- The module route is unchanged; the live URL
  `/learn/path/full-curriculum/imbalanced-learning-smote-cost-sensitive-learning?module=classical-ml` renders the
  lesson, and the page's own `h1` reads **“Imbalanced Learning: SMOTE, Cost-Sensitive Learning & Rare-Event
  Decisions.”** So the expanded display title is present on the page while the identity is untouched.
- `docs/teaching/classical-ml-supervised-progress.json` contains no mention of this topic; nothing about progress was
  changed for it.
- The blueprint is genuinely registered (imported at `blueprints/index.js:23`, exported in the map at `:150`), as the
  record claims.

## A2. The served data and its provenance — **no disagreement**

`scratch/imb-independent/step1_data.py`, parsing the **served** bytes by hand.

- `public/learn-assets/imbalanced-learning/yeast.data` is **byte-identical** to the packet copy (`cmp` clean), 94,976
  bytes, SHA-256 `7cf61776…218258`, matching the module header, `ATTRIBUTION.txt` and `data-provenance.md`.
- The physical description is accurate on every detail I checked: **pure LF** (zero CR bytes anywhere), 1,484 LF
  terminators, file ends with LF, no tabs, 1,484 non-empty lines each with exactly 10 whitespace-separated fields.
- Identity repair reproduces by a plain first-occurrence scan: exactly 22 repeated identifiers, each occurring exactly
  twice, **no conflicting repeat**, all 22 belonging to CYT (19) or NUC (3), leaving 1,462 distinct proteins and all 51
  ME2 positives. The lesson reuses one variable for both “the source has 51 ME2 rows” and “that leaves … 51 positives”;
  I checked that those two numbers are genuinely equal here, and they are.
- The four roles reproduce **exactly** from the documented seeds 61/62/63 — my independent `train_test_split` rerun
  returns the stored `development`, `reserve`, `fitting`, `tuning` and `inspection` ID lists element for element, with
  positives 21/7/7/16, disjoint, and covering every retained protein.
- **This lesson serves its own bytes.** No source file under the lesson references another lesson's asset directory;
  the only other-lesson path anywhere is a *negative* assertion in `verify-imbalance-browser.cjs:705`. Driving the real
  page and recording every request, **zero** requests touched any `learn-assets/` path other than
  `learn-assets/imbalanced-learning/`.
- **No reserved identity is published anywhere.** I extracted every source ID in the generated module (400 tuning +
  inspection IDs, 12 distinct top-ten IDs) and intersected with the 462 reserve IDs: empty, and every published ID is a
  tuning or inspection member. The lesson's strongest structural claim is true.
- Licence CC BY 4.0, DOI `10.24432/C5KG68`, creator and 1,484-instance count all confirmed against the live UCI page.

One number in the frozen packet is wrong — see **S3**.

## A3. The five weighted logistic fits, at 50 digits — **no disagreement**, and the 7.2e-07 question answered

`scratch/imb-independent/step2_fits.py`. My own standardisation (population sd), my own training matrices assembled
from the stored lineage, my own damped Newton on the analytic Hessian, then an 8-iteration `mpmath` polish at
`mp.dps = 50`.

| procedure | rows | my Newton ‖g‖∞ | 50-digit ‖g‖∞ | max &#124;50-digit − packet&#124; |
| --- | ---: | ---: | ---: | ---: |
| original | 600 | 1.7e−17 | 2.9e−52 | 3.92e−08 |
| balanced_weight | 600 | 1.6e−17 | 4.6e−52 | 3.73e−07 |
| random_over | 1158 | 4.1e−17 | 3.4e−51 | 5.22e−08 |
| random_under | 42 | 3.2e−17 | 4.1e−52 | 7.21e−07 |
| smote | 1158 | 2.5e−17 | 7.9e−52 | 1.86e−07 |

**The builder's 7.2e-07 figure is independently confirmed, and it is entirely L-BFGS-B's early stop, not Newton's
error.** My 50-digit optimum has a gradient infinity norm below 1e−51, so it *is* the minimiser; the packet's published
parameters sit up to 7.21e−07 away from it, on `random_under` — the 42-row fit, as one would expect.

**Is that tolerance tight enough for every published digit?** I answered this directly rather than by argument: I
recomputed the entire downstream study from both parameter sets and compared every quantity the page prints.

| procedure | params | selected threshold | TP/FP/FN/TN | cost | AP | ROC-AUC | Brier | top-10 |
| --- | --- | ---: | :--: | ---: | ---: | ---: | ---: | ---: |
| original | packet | 0.145005741 | 2/12/5/181 | 72 | 0.173459 | 0.797187 | 0.033475 | 2 |
| original | true optimum | 0.145005741 | 2/12/5/181 | 72 | 0.173459 | 0.797187 | 0.033475 | 2 |
| balanced_weight | packet | 0.740872551 | 4/15/3/178 | 51 | 0.273317 | 0.821614 | 0.131518 | 1 |
| balanced_weight | true optimum | 0.740872628 | 4/15/3/178 | 51 | 0.273317 | 0.821614 | 0.131518 | 1 |
| random_over | packet | 0.706115888 | 4/17/3/176 | 53 | 0.278679 | 0.832717 | 0.136743 | 2 |
| random_over | true optimum | 0.706115871 | 4/17/3/176 | 53 | 0.278679 | 0.832717 | 0.136743 | 2 |
| random_under | packet | 0.772568773 | 4/19/3/174 | 55 | 0.193550 | 0.816432 | 0.162806 | 2 |
| random_under | true optimum | 0.772568862 | 4/19/3/174 | 55 | 0.193550 | 0.816432 | 0.162806 | 2 |
| smote | packet | 0.759271675 | 4/13/3/180 | 49 | 0.272615 | 0.819393 | 0.121856 | 1 |
| smote | true optimum | 0.759271698 | 4/13/3/180 | 49 | 0.272615 | 0.819393 | 0.121856 | 1 |

**No published display quantity differs** — not one confusion count, not one cost, not one metric at its printed six
places, not one top-ten identity list, and not one selected threshold at the six places the page shows. The selected
thresholds differ in the eighth decimal, which is precisely the digit that is not shown. So the answer to the question
the brief poses is: **yes, 7.2e-07 is tight enough for every published digit**, and I verified that rather than
assuming it. See **O2** for the one residue.

Every recorded study quantity also reproduces exactly against the packet from the packet's own parameters: thresholds
to 1e−15, AP/ROC-AUC/Brier to 1e−12, all counts exactly, all top-ten identity lists exactly. My exact-rational
non-interpolated AP and my pairwise ROC-AUC agree with scikit-learn's to twelve decimals on all five procedures. The
scaler reproduces to 6.1e−16.

**The full threshold sweep.** I recomputed all 1,000 candidates (5 procedures × 200) and every one of the 8 recorded
fields per candidate — threshold token, TP, FP, FN, TN, precision (with `None` where nothing is selected), recall,
cost, alerts. **Zero mismatches.** The declared tie rule holds on every procedure: `original` and `random_over` each
have two candidates at the minimum cost and the higher threshold is the one selected; the other three have a unique
minimum.

**SMOTE lineage.** All 558 stored triples check out: every anchor is one of the 21 fitting-role minority proteins,
every neighbour is inside that anchor's **k = 3** nearest minority neighbours in standardised space, no self-neighbour,
every fraction in [0, 1]. The reconstructed synthetic vectors match the packet's first-ten standardised and
source-scale records to 1.2e−14 and 1.1e−16. I also tested the *one scalar u for the whole vector* claim directly: the
per-coordinate implied fraction within each generated row varies by at most 2.1e−15, so the construction really is a
line segment and not a per-coordinate draw. The resampling seeds 64/65/66 are independently re-derived by the data
verifier and agree with what I rebuilt from the stored lineage.

## A4. Every constructed number, in exact rational arithmetic — **no disagreement**

`scratch/imb-independent/step4_exact.py`. 73 checks with `fractions.Fraction`, or 50-digit `mpmath` where a log or exp
appears. Zero failures. (One line reported FAIL — my own test asserted `(1 − 0.9)² == 0.01` in binary floating mpmath;
in `Fraction` it is exactly 1/100. My test's fault, not the lesson's.)

- **§1.** 1,000 cases; baseline 980/1000 = 98%; model 976 correct and strictly less accurate; precision exactly
  14/32 = 7/16 = .4375; recall exactly 7/10; specificity 962/980; FPR exactly 1 − specificity; balanced accuracy the
  exact mean of recall and specificity; F1 exactly 28/52.
- **§2 ladder.** Precision exactly 2/3, 1/2, 0 and recall exactly 1, 1/2, 0, and precision *strictly* falls at each
  step — the counterexample is real, not approximate.
- **§2 average precision.** Exactly 5/6, and exactly equal to (1/2)(1) + (1/2)(2/3) as the prose factors it.
- **§2 flows.** Flow A exactly 100 positives, 80 detected, 99 false alarms of 9,900, precision exactly 80/179; the
  prevalence identity reproduces the same rational, so the flow and the formula are one claim. Flow B precision exactly
  80/1079, and its false-alarm count is exactly 99.9 — genuinely fractional, which is why the figure must not call it a
  census, and it does not.
- **§3.** Cutoff exactly 1/13; at p = .1 risks exactly 9/10 and 6/5. **At exactly p = cutoff the two risks are exactly
  equal**, which is the tie the model's convention resolves to *select*.
- **§4.** Total weight exactly 4; intercept contributions exactly 1/2 and −3/2; gradients exactly −1/4 and −3/4; a step
  of .4 gives exactly .1 and .3; new scores σ(.1) and σ(.7) to 12 places. The λ term is exactly zero at w = 0, so the
  arithmetic the figure draws really is just the two weighted sums. Balanced weights exactly 600/42 and 600/1158, and
  each class carries exactly n/K.
- **§4 optimum.** q* at (p = .1, 9, 1) is exactly 1/2; the equivalent probability cutoff is exactly 1/10; the
  doubled-weights null (18, 2) gives the *identical* rational; practice 4's (p = .2, 4, 1) is exactly 1/2 and inverts
  to exactly 1/5; tripling both weights leaves it unchanged.
- **§5.** A's nearest minority neighbour is B (squared distances exactly 4 and 25/4); the generated point is exactly
  (1, 0); majority M sits exactly on it. With the y-divisor 10 the nearest neighbour becomes C (1/16 against 4), so the
  metric contrast is real. Practice 5's point is exactly (2, 5/2).
- **§8 focal.** Modulators exactly 1/100 and 16/25; CE totals 1053.6051565783 and 16.0943791243 at 50 digits, focal
  10.5360515658 and 10.3004026396; the difficult group is **1.505%** of the cross-entropy mass and **49.435%** of the
  focal mass, so the figure's printed 98.5% / 50.6% shares are right and `design.md`'s “1.5%” is right.
- **§8 prior shift.** Sampled odds exactly 4, multiplier exactly 1/99, deployment probability exactly 4/103 and under
  4%. Practice 8: odds exactly 1, multiplier exactly 4/49, probability exactly 4/53.
- **§8 expansion.** 99:1 expands 100 → 198, factor exactly 1.98 and strictly below 2. Practice 9: 1,920 rows from
  1,000, factor exactly 1.92, undersample exactly 80.
- **Practices 1 and 2.** Both matrices give exactly 1188/1200 with recalls exactly 0 and 1; the tied queue at .8 gives
  exactly TP 1, FP 2, FN 1, TN 0, precision exactly 1/3, recall exactly 1/2, and selects three records — so the
  two-record budget really is a different policy.

Every one of these also matches the corresponding entry in `calculated-inputs.json`.

## A5. The drawn rule against the applied rule, over the entire enterable grid — **no disagreement**

This is the check the brief singled out, and it is the one I pushed hardest.

**Investigation 2.** The controls are a posterior on a 0.01 step in [0, 1] and two costs on a 0.1 step in [0, 50], and
values are published as `Number(x.toFixed(d))` — so the enterable grid is exactly 101 × 501 × 501 =
**25,351,101 points**. I built every point twice: once in exact rational arithmetic (`Fraction(k,100)`,
`Fraction(j,10)`) evaluating the rule the lesson *states* — select when (1−p)c_FP ≤ p c_FN, equivalently p ≥
c_FP/(c_FP+c_FN) — and once in the double arithmetic `actionRisks()` actually performs, tie band included.

| test | result |
| --- | ---: |
| grid points where the page grades against the exact stated rule | **0** |
| “exactly equal” announced where the exact risks are *not* equal | **0** |
| exact ties the page fails to announce as ties | **0** |
| printed cutoff fraction that is not the exact cutoff | **0** |

The tie band `1e-12 × max(1, R_sel, R_skip)` is exactly calibrated: it fires on precisely the 1,457 grid points where
the exact risks tie and floating point disagrees, and on no other. My first pass flagged 275 “disagreements”; opening
them showed every one was an exact tie at p = cutoff where the page correctly prints *“action select (by the
equal-cost convention)”* and the correct fraction — my paraphrase of the drawn rule was wrong, not the page. I record
that because it is the trap this check exists to avoid. **No value the control accepts grades correct reasoning as
wrong in investigation 2.**

**Investigation 3.** Forward grid 99 probabilities × 200 × 200 weights and the inverse grid the same shape —
**7,920,000 points**, again exact-rational against the implemented below/equal/above verdict with its ±1e−12 band.
**Zero disagreements in either mode**, including all 1,180 grid points where the exact answer is *exactly one half*.

Investigation 5 is where this class of defect does bite — but through a preset label rather than through arithmetic.
See **B1**.

## A6. External links and cross-references — **no disagreement**

All sixteen distinct external URLs resolve and support the specific claim attached to them. The two that most invited
doubt hold up:

- The **imbalanced-learn** sample-generation page states `x_new = x_i + λ(x_zi − x_i)` with λ a *single* random number
  in [0,1] — the lesson's one-scalar-u convention exactly.
- **Chawla et al.**'s own pseudocode redraws the gap *inside* the per-attribute loop, i.e. coordinate-wise. The lesson
  is the party that flags this, in its own Sources entry: *“Its pseudocode's placement of the random gap can be read
  coordinate-wise; this lesson explicitly uses one scalar fraction for a whole vector.”* That is accurate and correctly
  caveated — a strength, not a defect.
- The **NearMiss-1/2/3** table matches the current under-sampling documentation verbatim, including the
  nearest/farthest direction, which is the detail most often got backwards.
- scikit-learn's `average_precision_score` page states the non-interpolated convention and the exact Σ(Rₙ−Rₙ₋₁)Pₙ
  formula the lesson prints, and explicitly warns that the trapezoidal reading differs.
- The threshold-tuning guide does state that tuning leaves `predict_proba`, ROC and PR unchanged.

All internal links point at real stable IDs.

## A7. Practice instructions against the real controls — one failure

Practices 1–4 and 6–10 are all reachable and correct. Practice 2's four records sit on the 0.01 score step; practice
3's costs 2 and 7 sit on the 0.1 cost step and p = .2 on the 0.01 step; practice 4's weights 4 and 1 sit inside
[0.1, 20] on the 0.1 step. The three “Practice N” presets named in the prose (2, 3, 4) exist and load exactly the
records the prose states. Practice 5 does not — see **S1**.

## A8. The investigation contract — met in structure, broken in two places

Read in the code and driven on the real page.

`useInvestigation` (`ImbalanceShared.jsx:191–238`) holds `draft`, `active` and `previous` separately; `check` sets
`previous = active` before advancing and grades `answerFor(draft, active)` — **against the state committed, not live
state** — and stamps the result with `describeKey(draft)` so a verdict can never be shown beside inputs it was not
computed from. `edit` retires the verdict and clears both the radio and the numeric guess. `Prediction` disables Apply
until a choice is recorded and disables every radio once a verdict exists. Driving the real page confirmed all of it:
first paint has **five prediction gates, zero verdicts, zero checked radios, zero history notes**, and no revealed
bins, curves or record tables; the “Graded against the committed state: p = 0.1, c_FP = 3, c_FN = 36” caption prints
the committed inputs; the mismatch branch renders correctly; a preset retires the choice and the verdict. The
`role` badge switches correctly between tuning and exploratory, and the reserve is stated as unscored in both modes.

Two things break out of it: the retained history (**B2**) and one undefined-value path (**B3**).

## A9. Environment integrity — **no disagreement**

`pip check` reports no broken requirements. Directory timestamps in `scratch/lesson-tools/Lib/site-packages/` show the
22:05 install on 15 September created **exactly** `imbalanced_learn-0.14.2` and `sklearn_compat-0.1.6` and nothing
else. `scipy` dates to 10 September, `scikit_learn` to 11 September, and `shap`/`numba`/`llvmlite`/`slicer` to 01:33 on
15 September — an earlier lesson, not this one. NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1 and SciPy 1.18.1 all
import at the versions other lessons' recorded outputs depend on. **The shared runtime is undisturbed and the record's
claim about it is exactly accurate.**

---

# Part B — findings

## Blocking

### B1. The preset labelled “Exact null” grades the correct null answer as wrong

`src/learn/components/lesson-labs/ImbalanceLabs.jsx:887` (the preset) against `:905–917` (`answerFor`).

```js
{ key: 'double', label: 'Exact null: double both costs, holding the gate', ...tuningBaseline, costFP: 2, costFN: 24 },
```

The graded question in investigation 5 is *“Against the state currently applied, what happens to the **total declared
cost** on these 200 records?”* Doubling both costs doubles the total cost. From the baseline (Original, gate ≥ .5,
costs 1/12) the cost is 1×0 + 12×6 = **72**; after the preset it is 2×0 + 24×6 = **144**. The verdict is *“It rises.”*

Driven on the real page, recording the answer the button's own label asserts:

> **≠ You recorded Exactly unchanged; the calculation gives It rises.** At the applied gate the counts are TP 1, FP 0,
> FN 6, TN 193, for a total of 144 units.

(`scratch/imb-independent/shots/i5-null-graded-wrong.png`.) This is exactly the failure the brief names: a learner who
reasons correctly — “this is the null, so nothing moves” — is marked wrong by the page.

The genuine invariance is real but is a *different quantity*: I confirmed the **selected gate** is 0.14500574144149464
under both cost pairs, and the reveal below does print it. The button names a null for the quantity it does not grade.

**The verifier encodes the contradiction and passes.** `scripts/verify-imbalance-browser.cjs:481–485`:

```js
// The exact cost-scaling null: the same records, twice the cost.
await tuning.getByRole('button', { name: /Exact null: double both costs/ }).click();
await tuning.getByLabel('It rises', { exact: true }).check();
await tuning.getByRole('button', { name: 'Apply and check' }).click();
await checkText(tuning.locator('.imb-verdict'), /Your prediction matches: It rises\./);
```

The comment says “null”, the assertion says “It rises”, and the case summary at `:501` calls it “the exact
cost-scaling null”. `design.md`'s own investigation table lists I5's fixture as a “cost-scaling null”. The record, the
verifier and the button all assert a null that the graded question does not provide.

**Why it matters.** Three of the five investigations are built on nulls; this is the one whose null is false as
stated, and it is false in the direction that punishes correct understanding. Everything else about I5 is sound.

**Fix shape** (not applied): either retitle the preset to what it actually demonstrates — the cost *scales* while the
selected gate does not move — or add a fourth graded question in I5 about the selected gate, which is the quantity the
null is genuinely about and which the reveal already computes.

### B2. The retained verdict leaks the answer to the three nulls it was introduced for, and hides itself exactly when it would serve its stated purpose

`src/learn/components/lesson-labs/ImbalanceShared.jsx:306–311`:

```jsx
{!shown && state.history.length > 0 && <p className="imb-history">
  {historyLabel ?? 'A previous attempt, no longer current'}: you recorded {label(state.history.at(-1).choice)} and
  the calculation gave {label(state.history.at(-1).answer.outcome)} for the inputs applied then. …
</p>}
```

The brief asked me to judge whether this departure is sound or whether it leaks. **It leaks, and the render condition
is inverted relative to its own justification.**

The render guard is `!shown` — the history is on screen **only while the next prediction is being recorded**, and it
disappears the instant the new verdict arrives. That is precisely backwards from the stated reason for keeping it
(*“erasing the previous result destroys the comparison the null is about”*): the comparison can only be made once both
verdicts exist, and at that moment the earlier one is gone.

Whether it leaks depends on whether the graded outcome is absolute or relative. I checked all five:

| | `answerFor` signature | graded outcome | leaks on its null? |
| --- | --- | --- | --- |
| I1 score queue | `proposed` only (`:128`) | direction of precision, a property of the committed draft alone | **yes** |
| I2 action costs | `proposed` only (`:275`) | which action is cheaper, absolute | **yes** |
| I3 weighted score | `proposed` only (`:402`) | below / equal / above one half, absolute | **yes** |
| I4 SMOTE | `(proposed, current)` (`:705`) | `majorityMove`/`neighbour` compare before with after | no |
| I5 tuning queue | `(proposed, current)` (`:905`) | direction of change against the applied state | no |

The three that leak are exactly the three the builder names as the justification — the common-cost-factor null, the
doubled-weights null, and (for I1) the display-order null. Driven on the real page:

*Investigation 2, after applying the declared setup and then clicking “Null: multiply both costs by 3”:*

> A previous cost setting, no longer applied: you recorded Selecting and **the calculation gave Selecting** for the
> inputs applied then. The inputs have changed since, so that verdict no longer describes what is on screen; record a
> new prediction.

The correct answer to the question now on screen is *Selecting*. (`scratch/imb-independent/shots/i2-history-leak.png`.)
Applying it then produced *“Your prediction matches: Selecting”* and the history vanished
(`i2-after-null.png`).

*Investigation 1, after applying the tied setup and then clicking “Null: the same tied records, displayed in the other
order”:*

> A previous gate, no longer applied: you recorded Exactly unchanged and **the calculation gave Exactly unchanged** for
> the inputs applied then. …

Again the answer to the pending question, printed above the radios. (`i1-reorder-history.png`.)

The same applies to I3's doubled-weights null, whose graded outcome is “Exactly one half” before and after.

`LESSON-TEACHING-STANDARD.md`'s contract is that nothing is revealed before a prediction is recorded. Here the page
reveals the previous *outcome label* — which for a null is the current answer — in the moment the prediction is being
made.

**The browser verifier positively asserts the leak rather than guarding against it**:
`scripts/verify-imbalance-browser.cjs:267` is `await checkText(queue.locator('.imb-history'), /no longer applied/);`.

**Why it matters more than a wording issue.** The retained history was introduced as a deliberate departure from the
shared contract of the earlier lessons. The departure's premise is sound — a null is about two results, and you need
both. The implementation delivers the opposite: it shows the old result when it is a hint and withholds it when it is
evidence.

**Fix shape** (not applied): render the retired verdict **alongside the new one**, after `shown` becomes truthy, as a
labelled two-row comparison — which is what the justification actually asks for — and show nothing but the “inputs have
changed, record a new prediction” notice while the gate is open. That both closes the leak and delivers the comparison.

### B3. Investigation 1 prints “the calculation gives —” for an undefined precision, and the guard written to stop exactly this never fires

`ImbalanceLabs.jsx:128–131` returns `value: result.after.precision ?? Number.NaN`; `ImbalanceShared.jsx:303` renders
`round(shown.answer.value, …)` and `ImbalanceShared.jsx:28` turns `NaN` into `'—'`.

Driven on the real page — the first-class preset **“Above every score: select nothing”**, the correct radio, and any
number in the optional guess field:

> = Your prediction matches: It becomes undefined. You wrote 0.5 for the new precision; **the calculation gives —,
> outside 0.000001.** Recall moved decrease, from 1 to 0.

(`scratch/imb-independent/shots/i1-undefined-guess.png`.) Two things are wrong, and both are the mistake this lesson
exists to prevent:

- The em dash reads as a *formatted number*, not as “undefined”. §2's own paragraph is: *“When nothing is selected,
  precision's denominator is zero. That mathematical quantity is undefined… Show the counts and the convention rather
  than silently adding a tiny denominator and pretending the number has its ordinary interpretation.”* Everywhere else
  the lesson is scrupulous about this — `Undefined`, `ratio()`, `CountStrip` and the cost investigation's
  “cutoff **undefined** — both costs are zero” all name the vanished denominator. This one path does not.
- It then grades the learner's number as **“outside 0.000001”**, as though a correct number existed and theirs merely
  missed it. The learner answered the categorical question *correctly*.

**This is the defect Phase C recorded as fixed, surviving in a different investigation.** Phase C defect 2:
*“Investigation 3 printed ‘the formula gives —’ … A new check rejects any revealed text of the form ‘gives/is/are/equals
—’.”* I applied that exact pattern to the string above: `/(gives|is|are|equals)\s+—/` **matches**. The guard exists,
the guard would catch this, and the guard never sees it — because the browser verifier never combines a filled numeric
guess with an operating point whose precision is undefined. It fills a guess only in the “grading contract's misses”
section, at a gate where precision is defined.

I checked the other four investigations for the same shape: I2's difference, I3's optimum and inverse, I4's generated
x and I5's cost are all finite over their whole control ranges, so **I1 is the only path**. I confirmed I2's
both-costs-zero corner behaves correctly (“the calculation gives 0” — genuinely zero — and “cutoff undefined — both
costs are zero”).

### B4. Figure 4's caption claims “no arrow reaches the reserved proteins at all” — the diagram has no arrows, and a connector terminates inside the reserved box

`src/learn/components/lesson-labs/ImbalanceFigures.jsx:304–307` (figcaption), the in-diagram label at `:358`,
and the geometry at `:320–322`:

```jsx
{lanes.map(lane => <g key={lane.key}>
  <line className="imb-flow" x1={26} x2={26} y1={52} y2={lane.y + 14} />
  <rect className={lane.className} x={26} y={lane.y} width={104} height={40} rx="3" />
```

`lanes` includes `reserve` at `y: 252`, so the reserved box spans x 26–130, y 252–292 and its connector runs from
(26, 52) to (26, **266**) — **inside the box**. Measured on the live page rather than read off the source: figure 4
contains **zero** `<marker>` definitions, zero `marker-start`/`marker-end`, and computed `marker-end: none` on all
thirteen `line.imb-flow` elements; exactly one connector terminates within the reserved box's rectangle
(`scratch/imb-independent/fig4.cjs` output).

Against that drawing the page makes three statements:

| where | text |
| --- | --- |
| figcaption | “*No arrow carries a sampler into an assessment branch*, and **no arrow reaches the reserved proteins at all**.” |
| in-diagram `<text>` | “**no arrow reaches the reserve**” |
| `aria-label` | “Reserved, 462 proteins with 16 positives, has **no arrow into prediction**.” |

The aria description is correct and carefully qualified. The two visible statements drop the qualifier and become
false in two different ways at once: there are no arrows anywhere in the figure to reason about, and the one connector
class the figure does use *does* reach the reserve — as it must, since the reserve is created by the same split. What
is actually true, and what the aria text says, is that no connector leaves the reserve into a `transform`/`predict`
step.

**Why it matters.** §6 is the assessment-boundary section and this is its figure. A learner who does what the caption
invites — look at the picture and check that nothing reaches the reserve — finds a line reaching it and must conclude
either that the caption is wrong or that they have misread the diagram. And sighted and screen-reader learners are
given materially different claims about the same drawing.

This also settles the third of the builder's “could not be fixed” items (see **A10**/Part C below): omitting
arrowheads is a defensible drawing decision, but it is not compatible with four uses of the word *arrow* in the prose
that describes the drawing. Either the markers or the wording has to give.

## Should-fix

### S1. Practice 5's preset loads two minority points, not the three its own solution describes

`ImbalanceLabs.jsx:573–581` (the preset object begins at `:574`):

```js
{ key: 'practice', label: 'Practice 5: anchor (1, 2), neighbour (5, 4), fraction .25', …
  points: [
    { id: 'A', cls: 'minority', x: 1, y: 2 },
    { id: 'B', cls: 'minority', x: 5, y: 4 },
    { id: 'M', cls: 'majority', x: 3, y: 3 },
  ] },
```

The practice question (topic file line 338) posits *“only three minority observations”*; the solution (line 340) says
*“There are at most **two** other minority observations, so five-neighbour SMOTE is undefined in that fold… The
‘Practice 5’ setup in the SMOTE investigation loads **this geometry**; the investigation refuses k = 5 **with three
minority points** rather than silently reducing it.”*

The preset loads **two** minority points. Driven on the real page after clicking it: the k field's `max` is **1**, and
typing 5 produces the field error **“Keep it between 1 and 1.”** (`scratch/imb-independent/shots/i4-practice5-k5.png`).

So a learner who follows the instruction to check the solution's arithmetic in the lab is shown a limit of one, not
two, and the configuration the sentence describes — three minority points — is never created. The behavioural claim
(“refuses … rather than silently reducing it”) is itself satisfied: the `NumberField` rejects the out-of-range value
with a named error and publishes nothing. It is the geometry that does not match. Adding a third minority point to the
preset fixes both the count and the quoted “at most two”.

*A related but separate note:* `repairCloud` (`:863–872`, the clamp at `:870`) **does** silently clamp `k` when an edit lowers the minority
count — `Math.min(Math.max(1, Math.round(inputs.k)), available)`. That is defensible (the repaired value is what the
state strip displays, so the learner sees it), but it sits under prose that says the investigation does not silently
reduce k, so the sentence would read better scoped to the control rather than to the investigation.

### S2. The record's “Files created” line counts are stale by up to 112 lines after Phase C

`design.md`, Phase A “Files created” table, against `wc -l`:

| file | record | actual | delta |
| --- | ---: | ---: | ---: |
| `ImbalanceLabs.jsx` | 934 | **1046** | **+112** |
| `ImbalanceFigures.jsx` | 530 | **553** | **+23** |
| `imbalance-labs.css` | 293 | **319** | **+26** |
| `ImbalanceShared.jsx` | 415 | 417 | +2 |
| the other seven | — | — | ±1 (a trailing-newline counting convention) |

The ±1 rows are a counting convention and not worth touching. The first three are Phase C's own edits — the defects it
fixed in §“Defects the verifier found” and §“Defects found only by opening the screenshots” — never written back into
the Phase A table. Phase C adds no table of its own, so the only line counts anywhere in the record are wrong for the
three files Phase C changed most. This is the record-versus-tree class, in its mildest form: the table is explicitly a
Phase A snapshot, but nothing tells a reader that, and `ImbalanceLabs.jsx` is understated by 12%.

Everything else in the record's file table checks out: the blueprint is registered, the preserved original is exactly
63,689 bytes and byte-identical to `git show HEAD:`, and the **139** KaTeX expressions the record counts are exactly
139 in the tree.

### S3. The frozen packet's `data-provenance.md` states a class count that is wrong and does not sum to its own row total

`docs/teaching/drafts/imbalanced-learning-smote-cost-sensitive-learning/data-provenance.md`, “What the columns mean”:

> The original location counts are CYT463, NUC429, MIT244, ME3163, ME251, ME144, **EXC37**, VAC30, POX20, ERL5.

Counting the served file myself: **EXC = 35**. The stated ten counts sum to **1,486**, contradicting the same
document's own “1,484 rows” two paragraphs earlier. The true distribution sums to 1,484, and both
`calculated-inputs.json` (`data.class_counts.EXC = 35`) and `src/learn/data/imbalance-data.js`
(`provenance.classCounts.EXC: 35`) carry the correct value, so **nothing the learner sees is wrong**.

I raise it because `data-provenance.md` is the document a future maintainer consults to re-derive the study, it is the
one place in the packet whose numbers no verifier reads, and it is internally inconsistent. The implementation is not
at fault — this is a content-phase transcription error in a frozen file — but the builder re-derived the class counts
in Phase A and had the correct value in hand, and a one-line correction note would have closed it.

### S4. Investigation 4's two relative questions grade trivially “unchanged” when nothing has been moved

`ImbalanceLabs.jsx:705–716`. For `majorityMove` and `neighbour`, `answerFor(proposed, current)` compares the
construction built from the draft with the one built from the applied state. The question itself is part of the draft,
and selecting it is an `edit`. So a learner who picks *“Does moving a majority point change it?”* and presses Apply
**without moving anything** gets `after === before` and the verdict *“No, it stays exactly where it was”* — correct by
construction, and evidence of nothing.

The prompt text does instruct (*“You are about to move only a majority point”*), and the `Exact null` preset does move
M, so the intended path is well signposted. But Apply is enabled whenever a radio is set, including when
`state.pending` is false, so the degenerate path is one click away and indistinguishable in its output from the real
null. Requiring a pending edit before Apply on these two questions — or saying in the verdict that nothing changed
because no input changed — would close it.

## Observations

### O1. Trust-root coverage: 47.9% of scalar leaves are re-derived; the uncovered 52.1% is almost all one unconsumed block

The brief asked for an explicit fraction. `calculated-inputs.json` holds **18,239 scalar leaves**. Enumerating what any
verifier touches:

| block | leaves | re-derived by a verifier? |
| --- | ---: | --- |
| `data` (kept IDs, removed IDs, duplicate names, class counts) | 1,532 | yes — `verify-imbalance-data.py`, by an independent parse and first-occurrence scan |
| `split` ID lists and `positive_counts` | 2,466 | yes — re-run of the seeded stratified splits |
| `split.tuning_labels` / `inspection_labels` | 400 | **no** (the verifier uses its own labels, never comparing the stored copies) |
| `scaler` | 12 | yes |
| `resampling` (duplication, undersampling, minority, 558 synthesis triples, first-ten vectors) | 2,415 | yes — seeds 64/65/66 re-drawn independently |
| `methods.parameters` | 35 | yes — Newton against L-BFGS-B at 2e−6 |
| `methods.tuning_scores` / `inspection_scores` | 2,000 | yes — recomputed from the verifier's own parameters at 5e−6 |
| `methods` thresholds, count blocks, AP/AUC/Brier, top-ten, fixtures | 195 | yes (the `threshold`, `precision` and `recall` keys inside the count dicts are not compared: ~70 leaves) |
| **`methods.tuning_sweep`** | **9,000** | **no** |
| `methods.convergence` | 20 | **no** |
| `versions`, `inspection_case_ids` | 6 | **no** |
| `constructed` | 73 | yes — **28 explicit `root()` re-derivations** |

**Covered ≈ 47.9%; uncovered ≈ 52.1%, of which 9,000 of 9,496 leaves are the single `tuning_sweep` block.** That block
is also consumed by nothing: `verify-imbalance-models.mjs` reads only `recorded.constructed`,
`recorded.methods[].tuning_investigation_fixture` and the per-method summary fields, and the generated data module
publishes no sweep. So the uncovered majority is inert rather than load-bearing.

This is **much better than the comparable scope in the previous lesson**, where the trust root was checked by nothing
at all. Here the data verifier genuinely re-derives the split, the scaler, the resampling RNG draws and all five fits
through a different implementation, and the 28 `root()` checks cover the whole `constructed` section independently. Two
structural residues remain:

- `calculated-inputs.json` **is hashed** (`verify-imbalance-data.py:272`) but the hash is only *recorded* into the
  evidence file (`:761`) — it is never asserted against a pinned value. A hand-edit changes the recorded hash silently.
- Everything downstream of the fits is deliberately computed from the packet's **saved scores** rather than the
  verifier's own (the code says so in a comment). That is the right choice — the lesson publishes those scores — but it
  means the counts, thresholds and metrics are checked for internal consistency with the saved scores, not against an
  end-to-end independent refit.

**I closed both gaps for the current bytes**: I recomputed all 1,000 sweep rows × 8 fields from my own refit (zero
mismatches), verified the 400 stored labels against the served file, and recomputed every downstream quantity from
parameters I derived myself (A3). No value in the trust root is wrong today.

### O2. The published `parameters` carry sixteen digits of which about seven are established

`imbalance-data.js` prints e.g. `-4.260539934573494`. My 50-digit optimum says the true minimiser differs from these in
the seventh or eighth significant figure. Nothing the page shows is affected (A3), and the arrays are not rendered
anywhere in the lesson, so this is not a defect — but a maintainer who reuses those coefficients as a reference should
know they are the L-BFGS-B stopping point, not the optimum. One sentence in the module header would say so.

### O3. What the verifiers do not assert

Re-run confirms every count in the record, exactly and verbatim:

- `verify-imbalance-data.py` — `PASS: 1,442 data checks, including 28 re-derivations of the packet's own trust root;
  five fits reproduced by Newton iteration within 7.21e-07 of the packet's L-BFGS-B parameters; module byte-identical
  (62 KB).`
- `verify-imbalance-examples.py` — `PASS: 3 displayed programs extracted verbatim and executed, 58 oracle assertions.`
- `verify-imbalance-models.mjs` — `PASS: 386 grouped imbalanced-learning model checks across 58 groups, including 1,000
  swept threshold candidates and 168 weighted-optimum grid points.`
- `imbalance-browser.json` — 20 case records, status `passed`, **46 screenshots**: I confirmed all 46 paths are
  distinct, all 46 files exist on disk, and all 46 have **distinct content hashes**. No path is written twice and no
  two files share bytes. (This is the failure the previous review found; it is genuinely absent here.)

All four exit 0 and the generated data module is byte-identical after a re-run. The new guards the brief asked me to
test are **real, not decorative**:

- The `fill="none"` rule (`verify-imbalance-browser.cjs:601–615`) walks every `[fill], [stroke]` in every lesson SVG and
  compares the declared attribute with the computed style. I ran my own equivalent sweep across *six* presentation
  attributes on the live page and found **zero divergences** — the figure-1 class of defect is genuinely gone, and the
  band assertions that follow (four painted parts, four distinct colours, unfilled outline, three in-band labels) all
  hold in the image I opened.
- The tie null carries an explicit non-vacuity assertion (`:299–300`): *“the reordered display really did change the listed
  order, so this null is not vacuous.”*
- The type-size check uses genuinely different thresholds for primary (10 px) and secondary (8.4 px) text, and my own
  measurement found **zero** SVG text under 9.5 px at either 390 or 320 px.

The substantive blind spots beyond **B1**–**B3**:

- **The numeric-guess path is never exercised against a null-valued answer**, which is what lets B3 stand behind a
  green verifier and a guard that matches the string.
- **The retired-history element is asserted to exist** (`:267`) and never asserted to be absent while a gate is open,
  so B2 is enshrined rather than guarded.
- **No assertion compares a figure's visible caption with its `aria-label`.** Had one existed, B4 would have surfaced:
  the two say different things about the same drawing.
- **Nothing tests a preset's label against its own graded outcome.** B1 is a one-line check away: for every preset
  whose label contains “null”, assert the verdict is the unchanged branch.
- The curve-through-label sweep is real and finds real things, but it tests straight `line` elements and drawn paths
  against label rects; it has no notion of a label being *far from the thing it names* (see O6).

### O4. Five line-through-label crossings survive, all cosmetic

My own geometry probe — sample every `polyline`/`path`/`line` by `getTotalLength()`/`getPointAtLength()`, transform
through `getScreenCTM()`, test containment in every `<text>` client rect, with all `<details>` forced open — found at
1366 px:

| shape | label | samples inside |
| --- | --- | ---: |
| `line.imb-grid` ×3 | `both populations: FPR 0.01, TPR 0.8` (figure 2's ROC inset) | 10.5% each |
| `line.imb-axis` | `Random oversampling` (figure 5's cost panel) | 5.0% |
| `line.imb-axis` | `Random undersampling` | 5.0% |

At 390 and 320 px a fourth gridline joins the first group and `Balanced weights` joins the second. I opened both
figures: the gridlines are faint and pass *behind* legible text, and the two group headings overrun the zero axis by
about fifteen pixels with the text painted on top. Neither obscures a value. I record them because the shared
inspector does not see them and because the class is the one that produced a blocking finding in the previous lesson;
here it did not.

### O5. Figure 1 puts the negatives band's overflow label at the opposite end of the track from its segment

`ImbalanceFigures.jsx:51–55`. A band part narrower than 46 units gets its label outside the track, always at
`x={right}` with `textAnchor="end"`. In the positives band both parts are wide, so this never fires. In the negatives
band the narrow part is **false alarm 18** — 1.8% of the class, about six units wide, drawn at the **left** edge — and
its label *“false alarm 18 (1.8% of this class)”* is printed at the far **right**, roughly 300 px away, with no leader
line. I confirmed it in the rendered image (`scratch/imb-independent/shots/b1366-00.png`).

Relatedly, the two bands order their parts inconsistently: positives run *detected, missed* (error on the right),
negatives run *false alarm, cleared* (error on the left), so the eye cannot use position to find the errors. Neither is
wrong; both cost a little of the figure's readability. Anchoring the overflow label to the side its segment is on would
fix the first for the cost of one conditional.

### O6. Figure 5's cost panel is the one place where the drawn segments and their arithmetic are both perfect

Recorded because I went looking for a defect and did not find one. Every bar in “Realised cost at 0.5 and at the
selected threshold” prints its own decomposition — `85 = 1 + 12×7`, `72 = 12 + 12×5`, `71 = 35 + 12×3`, `51 = 15 +
12×3`, `70 = 34 + 12×3`, `53 = 17 + 12×3`, `65 = 29 + 12×3`, `49 = 13 + 12×3` — and all eight reproduce exactly from my
own refit, as do the dashed baseline at 84 and every entry of both tables beneath (thresholds to six places, all forty
confusion counts, AP, ROC-AUC, Brier, top-ten counts and the top-ten identity lists). The three-winners sentence is
true: SMOTE is uniquely lowest on cost (49), random oversampling uniquely highest on AP (0.278679), and original,
random oversampling and random undersampling tie at two positives in their top ten while the other two find one.

### O7. Investigation 1's card rail orders the cards by selection before anything is revealed

`ImbalanceLabs.jsx:103` renders `point.selectedIds.concat(point.unselectedIds)` with `point` built from the **draft**,
so before the prediction is recorded the card order already encodes the `score ≥ threshold` partition, even though each
card reads “awaiting the gate”. The truths are learner-owned and visible in the controls, so this is at most a small
shortcut to an answer the learner could compute from inputs they chose, and there is no divider marking the split.
Recording it because the contract phrase is “nothing revealed on first paint” and a computed partition is, strictly,
encoded there.

### O8. The topic note's closing sentence is now false

`docs/teaching/topic-notes/imbalanced-learning-smote-cost-sensitive-learning.md` (modified 12 September, i.e. by the
content phase, not the builder) ends: *“the original published body is still unchanged.”* The body was replaced in
Phase A. The file is outside the builder's declared file list, so this is not the builder's error, but it is the last
record in the chain that still asserts the pre-implementation state.

### O9. Things the record gets right that were worth checking

Most of the record holds up, and several claims that invited doubt are accurate:

- **All eleven Phase A departures are real and justified as described.** The `imbalanced-learn` program genuinely runs
  and its three fold APs (`0.49462759 0.36721427 0.37700713`, mean `0.41294966368063185`) coincide with **none** of the
  five inspection APs — I checked all fifteen pairs at 1e−6, which is the claim the manuscript makes about it.
  Departure 6's argument is sound: I confirmed the loss at the optimum varies by orders of magnitude across
  w ∈ [0.1, 20], so a pinned axis is genuinely impossible and printing both losses is the right substitute.
  Departure 8's ROC zoom is necessary and the axis label names it. Departure 9's two-panel choice is necessary: the
  difficult group really is 1.505% of the cross-entropy mass.
- **Both Phase C departures I was asked to judge are justified.** Removing the two in-plot value labels from the cost
  investigation is correct — at the case posterior the two lines are at their closest, so any label there sits in the
  worst possible place, and both risks are printed to six places in the strip directly above, which is what the
  specification asks for. The corner-clearance search is also correct and its stated reason is verifiable: with costs
  1 against 12 the selecting line runs along the bottom of the box for its whole width, and with them reversed it does
  not, so no fixed offset can be safe. I watched the annotation move as I changed the costs.
- **Two of the three “could not be fixed” items are honestly scoped.** The small cost crossing at 1 against 12 is a
  property of the quantities on a shared axis, and the compensations claimed (both risks, the cutoff as a fraction and
  a decimal, and the chosen action printed above the plot) are all genuinely present — I read them off the rendered
  page. Investigation 5's sampled candidate table is the right call; the sweep really does traverse all 200 candidates
  and only the display samples, the footnote says so and gives the full count, and I verified the full sweep
  independently. The third item — no arrowheads — is where the drawing and the prose part company: see **B4**.
- The first Phase A defect is real and well diagnosed: `Math` is shadowed by the KaTeX import in the topic module, and
  the `largest`/`smallest` reducers at `:55–56` carry a comment naming the hazard. The stylesheet-scope defect is real
  too — `.imb-lesson svg text` with longhands does cover diagrams inside investigation stage wrappers.
- Every Phase C “found by looking” fix is present **in the images**, which I opened: figure 1's four bands paint in
  four distinct colours with their in-band count labels (the template defect is gone); figure 5's identity column is
  last and unclipped; the metric columns carry fixed decimals (`0.193550` beside `0.273317`); the headers are
  shortened; the slider and its number field share a grid cell; entity rows align to their entity's column count;
  investigation 5 prints six places with the note that the decision uses the saved score exactly; and the SMOTE point
  labels are offset perpendicular to the segment with G taking the opposite side.
- 139 KaTeX expressions, zero accents, and **no horizontal overflow at either 390 or 320 px** (`scrollWidth ===
  innerWidth` at both), with **zero** SVG text under 9.5 px. 173 focusable controls inside the lesson and **zero**
  enabled controls with a zero-size box.

---

## Closure

The mathematics and the data handling of this lesson are excellent. Across roughly 33.3 million independent constructed
checks — 25,351,101 exact-rational evaluations of investigation 2's rule and 7,920,000 of investigation 3's over their
complete enterable grids, a from-scratch refit of all five procedures polished to 50 decimal digits, all 1,000 sweep
candidates and both 200-value score arrays per procedure recomputed, 73 exact-rational checks of every constructed
number, and a by-hand parse and identity repair of the served bytes — **I found no disagreement with any published
number**, in the manuscript, the lesson body, the figures, the practice solutions, the generated data module, or the
packet's trust root. The served data is byte-identical and honestly described, this lesson serves its own copy and
requests nothing else, no reserved identity is published anywhere, every external link resolves and supports its claim,
the title/ID/route/module arrangement is exactly as approved, the shared Python runtime is undisturbed, and the
7.2e-07 optimiser gap moves not one digit the page prints — which I established rather than assumed. The trust root is
covered far better than in the previous lesson of this scope.

The defects are in the teaching surface, not the arithmetic. **B1** grades correct reasoning as wrong, through a
preset whose own label, the design record and the browser verifier all call a null that the graded question is not.
**B2** is a deliberate departure implemented inside out: it shows the previous answer while the learner is predicting
and hides it once the comparison could be made, leaking the answer for exactly the three nulls it was introduced to
serve. **B3** is the “gives —” defect Phase C fixed in one investigation and left standing in another, behind a guard
that matches the string but is never pointed at it. **B4** is a caption asserting a property of arrows in a diagram
that has none, contradicted by the connector the figure actually draws, and disagreeing with the figure's own aria
description.

Two of the four are visible only by driving the page into a state the verifiers never enter, and the other two only by
reading a label against the rule it names. That is the shape of what is left to find here.

Not complete. Fix B1–B4, then S1–S4.
