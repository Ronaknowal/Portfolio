# Anomaly & Outlier Detection — independent phase-two review

Reviewed 13 September 2026 against the working tree at commit `8c5da59f18516be77c29d5aeeafca3decca4f738` plus the uncommitted anomaly-detection implementation. Reviewer did not author the packet or the implementation. No file under `src/`, `public/`, `scripts/` or `docs/teaching/drafts/` was edited; this review document is the only write. No state-changing git command was run.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under the session scratchpad, written from the manuscript's definitions and importing neither the lesson's models nor the author's verifiers, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1). They recomputed `c(m)` for m = 0…6, the exact five-position cut integration at depth cap 3 for `[0,1,2,3,12]` and `[0,1,2,3,4]`, `2^(-18/13)`, the full k = 2 and k = 3 LOF state on `[0,1,2,20,24,28]` with queries 3/4/6/17/25, every training-versus-query agreement case over 6 coordinates × k = 1…5, the kernel rho/midpoint/anchor values and the gamma = 1 zero crossing by `brentq`, practices A, B, C, E, F, H and I, the §8 population arithmetic, and the **complete temperature pipeline refitted from the raw CSV** (row counts, median/MAD, scaler, all four detectors, all eight published rows, the four `0.975` rows, three quantile indices × four methods, three direct sweep levels, and per-row alert counts inside all four window detail blocks). The five displayed programs were extracted from `anomaly-detection-examples.js` and executed against the served CSV/JSON. `node` was used to dump `anomaly-temperature-data.js` and to evaluate the JSX template literal in the setup code block. `sha256sum` was run on every reviewed file. |
| **Read in full** | `LESSON-TEACHING-STANDARD.md`, `lesson.md` (802 lines), `visual-specifications.md`, `dataset-provenance.md`, the phase-two section of `ANOMALY-DETECTION-LESSON-DESIGN.md`, the published `.jsx` body, `AnomalyDetectionLabs.jsx`, `AnomalyTemperatureLab.jsx`, `AnomalyDetectionShared.jsx`, `AnomalyDetectionFigures.jsx`, `anomaly-detection-labs.css`, `anomaly-detection-models.js`, `anomaly-detection-examples.js`, `RunnableExample.jsx`, the blueprint, the three served asset files, the four evidence JSONs, the ledger entry and the anomaly section of `LESSON-AUTHORING-HANDOFF.md`. Five of the eleven evidence screenshots were opened and looked at (`temperature-desktop`, `kernel-desktop`, `lof-desktop`, `first-cut-mobile`, `threshold-ruler-mobile`). |
| **Reused, not rerun** | `evidence/anomaly-detection-models.json` (63 grouped browser-model checks) and `evidence/anomaly-detection-browser.json` (13 Playwright cases). Their recorded source hashes equal the hashes below for every file they name, so the evidence binds to the bytes reviewed. The production build, the Playwright run and `verify-curriculum.mjs` were not repeated. |
| **Delegated** | Every external URL in the Sources block was fetched and checked against the sentence that cites it, including the pinned NAB commit, its `LICENSE.txt`, `data/README.md`, `labels/combined_windows.json` and issue 376; the four PDFs were downloaded and their section structure extracted with `pdftotext` rather than trusted to a summariser. Findings 3, 4 and observations O4–O5 come from that pass. |
| **Not done** | No beginner walkthrough (this is a reviewer's heuristic learning-experience assessment). No production build, no browser session, no keyboard pass: the leak in finding S1 and the table clipping in O2 were established by reading the components and one screenshot, not by operating the page. Six screenshots were not opened. |

## Source versions reviewed (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/anomaly-outlier-detection-isolation-forest-one-class-svm-lof.jsx` | `11e2d319f8b8c4d2c251bc4f3967416358a6dcee3aab2246f963bab367ca84d2` |
| `src/learn/data/anomaly-detection-models.js` | `7d8a2dd0225426f507a2e57ff7d8dcf2c1f3e6233c7097b0b6e9d7bafa782d67` |
| `src/learn/data/anomaly-temperature-data.js` | `750d462024c0321687bd1b98c8f0755616351f60d5b9465d1f6dfc59d53625d4` |
| `src/learn/data/anomaly-detection-examples.js` | `472d6bfbc707a848fb77795444ab6f0666639e44c75d009797c8ef781a8066b2` |
| `src/learn/components/lesson-labs/AnomalyDetectionLabs.jsx` | `c9da69f2823d32ca87937471c6b88369bf009e7893710825b0ff19d687d8a4dd` |
| `src/learn/components/lesson-labs/AnomalyTemperatureLab.jsx` | `478a0127a38ade6e2a73391f889d9a481f68b1a528cc61a391e059c90d49fb5b` |
| `src/learn/components/lesson-labs/AnomalyDetectionShared.jsx` | `7678139fc40508674b417f1b62629a29c6687c8545cbcfe0fbcc33a3012e8143` |
| `src/learn/components/lesson-labs/AnomalyDetectionFigures.jsx` | `0ea3774ef1dfe62f1437eb28d81bd5cf72d9c7312cd123cf957f2c8695c2733c` |
| `src/learn/components/lesson-labs/anomaly-detection-labs.css` | `4f066dc8dad28ce9921e36f1085555514f10066a35b156d739fe1f3d5f86b164` |
| `src/learn/data/curriculum/blueprints/anomaly-outlier-detection-isolation-forest-one-class-svm-lof.js` | `82ad9dc1d288ea831868aa682a386c5e36ab3e8e2d4722c8daad5985f63e51d8` |
| `public/learn-assets/anomaly-detection/machine_temperature_system_failure.csv` | `92bf5b87fc7f9bba8ca0b7ec63ccaac8cb4a1371a258e8c29a10ae9c018d82a4` |
| `public/learn-assets/anomaly-detection/nab-event-windows.json` | `1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1` |
| `public/learn-assets/anomaly-detection/NAB-LICENSE.txt` | `0a0b4d0b10cb1f7ed9ab2993ef93defc03447e6eba9daca1315dd32dae4877e3` |
| `docs/teaching/drafts/.../lesson.md` | `de1530bae9ddb5e9a8ec8b3d5d52f77e5c244340599bc8bd472dd42d47faba02` |
| `docs/teaching/drafts/.../visual-specifications.md` | `00f022a527237228e5063ef1512b4e4cc4daf7bfb1ea8f4ec79531460620447b` |
| `docs/teaching/drafts/.../author-calculations.json` | `72907abc78905ad719ff5df25b65d769c8bd8b55019c2da4f6c06e7ab2c69721` |
| `docs/teaching/drafts/.../dataset-provenance.md` | `cbe2729d2b85fee81598e7374a3a442cc5d33f788c82894d9ce5eafc3dcf821e` |
| `docs/teaching/drafts/.../content-author-checkpoint.json` | `fa33528a8ac88f7729fbb821929fa14008896a21e461063eeb666b87e7874b15` |
| `docs/teaching/ANOMALY-DETECTION-LESSON-DESIGN.md` | `75c855b3d3c48d6e9fb4a532316118f8f0ac545fdafe6d1f901574313b09d325` |
| `scripts/verify-anomaly-detection-models.mjs` (read only) | `bc48a6a12d621758df27f1bec30c3bc527f474d90426078cd9497593677d92e6` |
| `scripts/verify-anomaly-detection-examples.py` (read only) | `2c000b5843b0d0dcdadc94dacc58ec6ec6666f0e196d22589b82c5f7aff089db` |
| `scripts/verify-anomaly-temperature-data.py` (hash only) | `c41716204a49fb8e85ccd1437d75b08344a084f401f35ace083975f9c0d7d84c` |
| `scripts/verify-anomaly-detection-browser.cjs` (hash only) | `bc1e96aa62b91151d8b91f4710df3ca15f0da5737aa3f3bead2c5de04befec24` |

A hash proves identity, not correctness. The three evidence files that record source hashes (`anomaly-detection-models.json`, `anomaly-detection-browser.json`, `anomaly-native.json`) name exactly these bytes, so their recorded results apply to the version reviewed here.

## Part A — correctness

### A1. Manuscript sections, claims, cautions and tasks preserved?

The published body carries all thirteen manuscript sections, the first-pass route, the three-kind table, both deeper branches, all five programs in the manuscript's stated order, all four honesty caveats, and practices A–J with separate closed hint and solution disclosures. Coverage checked sentence by sentence; the only omissions are ones that moved into a figure. Specifically:

- **Annotations are not fault labels** — §10 first paragraph ("not verified fault labels for every row, and there is no documented one-to-one mapping between the windows and that prose description") and again after the program.
- **Outside-window alerts are unmatched workload** — §10 paragraph after the program, and in the temperature lab's caption.
- **Nu is not prevalence** — §6 sub-head "Nu is not tomorrow's fault prevalence", kept verbatim in substance.
- **Scores are not probabilities** — the §1 Callout ("It is not a probability, not a distance in the original units and not a verdict"), §2's decision-function paragraph, §3's "0.8 does **not** mean an 80% chance of failure", and the isolation lab's own caption.

Deviations found, each judged:

| Location | Deviation from manuscript / specification | Judgement |
| --- | --- | --- |
| §4 Step 1 (`.jsx` L127–128) | The manuscript's six-row `r₂` table is not in the prose. | Acceptable: `ReachFloorFigure` (`AnomalyDetectionFigures.jsx` L108–114) carries a superset — row, both neighbours, radius, lrd and factor — one element later. Not recorded as a departure. |
| §8 (`.jsx` L206) | The manuscript's "sorted calibration scores 1, 1, 2, 4, 4" paragraph is not in the prose. | Acceptable: `ThresholdRulerFigure` carries it with the same numbers, and its aria-label states the same three facts. Not recorded as a departure. |
| §3 heading (`.jsx` L13) | Manuscript: "an empty gap **can** make a point easy to separate". JSX: "an empty gap **makes** a point easy to separate". | Dropped qualifier. Finding S8. |
| §4 opening (`.jsx` L125) | Manuscript: "A global nearest-distance rule **can** penalize it". JSX: "penalises it". | Dropped qualifier. Finding S8. |
| §13 (`.jsx` L298–305) | A six-row "Readiness check" table not present in the manuscript. | Good addition — it is exactly the standard's readiness check. Not recorded as a departure. |
| §3 program (`anomaly-detection-examples.js` L9) | The published program prints two extra lines (`Fitted sample size`, `Scores`) that the manuscript block does not. | Improvement: the fitted-size line is what the surrounding prose asks the learner to look at. Not recorded as a departure. |
| Section titles 3, 5 and 13 | Reworded from the manuscript. | No consequence. |
| `LESSON-AUTHORING-HANDOFF.md` / ledger | Records disagree with each other and with reality. | Finding B2. |

The three departures the design record does list (six formulas re-set for 320 px, a second checkpoint before the real-data program, the 0.975 row added to the published table) were all confirmed present and accurate. The second checkpoint's numbers — 1,548 later alerts for Isolation Forest (7.50 % of 20,634) and 9,232 for One-Class SVM (44.7 %) — are correct.

### A2. Independent recomputation (reviewer's own scripts, no lesson code)

| Claim | Independent result | Agrees? |
| --- | --- | --- |
| `c(5) = 77/30 ≈ 2.5667`; `c(2) = 1`, `c(4) = 13/6` | `[0, 0, 1, 5/3, 13/6, 77/30, 29/10]` for m = 0…6 | Yes |
| Five expected corrected paths, depth cap 3 | `31/12, 73/22, 17/5, 17/6, 841/660` | Yes, exactly |
| The five rounded scores | `0.497755, 0.408159, 0.399239, 0.465258, 0.708845` | Yes, to all six digits |
| "Replace 12 with 4 and the endpoints tie at about 0.569715" | paths `25/12, 17/6, 3, 17/6, 25/12`; scores `0.569715, 0.465258, 0.444782, 0.465258, 0.569715` | Yes, and the middle value matches the specification's `.444782` |
| First-cut probabilities 3/4 and 1/12 | `9/12 = 3/4`, `1/12` | Yes |
| Practice B: `2^(-18/13) ≈ 0.3830` | `2^(-3/(13/6)) = 0.38299158933399…` | Yes |
| LOF radii, lrd and factors on `[0,1,2,20,24,28]`, k = 2 | radii `[2,1,2,8,4,8]`; lrd `[2/3,1/2,2/3,1/6,1/8,1/6]`; factors `[7/8,4/3,7/8,7/8,4/3,7/8]` | Yes |
| `LOF(4) = 35/24`, reaches 2 and 3, lrd 2/5, mean neighbour density 7/12 | identical | Yes |
| `LOF(17) = 35/32`, reaches max(3,8) = 8 and max(7,4) = 7, lrd 2/15, mean neighbour density 7/48 | identical | Yes |
| `LOF(6) = 21/8 = 2.625` (program output) | identical | Yes |
| Kernel: rho and midpoint at gamma 0.1 and 1 | rho `0.835160023` / `0.509157819`; g(0) `+0.069677` / `−0.141278`; g(±1) exactly 0 | Yes |
| Practice E: anchors ±2, gamma 0.25 | rho `= (1+e⁻⁴)/2 = 0.5091578194`, midpoint sum `e⁻¹`, g(0) `−0.14127837827` | Yes — identical to anchors ±1 at gamma 1 |
| Gamma = 1 accepted region is disconnected | zero crossing at `0.9145419060` (brentq); g(0.95) `+0.00075`, g(0.9) `−0.00061` | Yes — the lab's "−1 to −0.9145, 0.9145 to 1" is right |
| §8 population arithmetic | 100 faults, 80 flagged; 99,900 non-faults, 999 flagged; 1,079 total; `80/1079 = 7.4143 %` | Yes |
| Practice F | 100 faults → 90; 49,900 non-faults → 249.5; total **339.5**; `90/339.5 = 26.5096 %` | Yes |
| Practice C | reaches `0.4, 0.8, 1.0`, mean `11/15`, density `15/11`, factor `1.5/(15/11) = 1.1` | Yes |
| Practice H | `7913/445 = 17.7820` | Yes |
| Practice I | A: 2/2 windows, 6 rows, 0 outside; B: 2/2, 3 rows, 1 outside | Yes |
| **CSV facts** | 22,695 raw rows; 22,683 unique timestamps; 22,671 feature rows after dropping 12 with no exact one-hour lag; the twelve dropped are the first hour | Yes |
| **Uniform spacing** (the lab derives every displayed timestamp as `start + index × 5 min`) | all 22,682 level gaps and all 22,670 feature gaps are exactly `00:05:00`; first feature timestamp `2013-12-02 22:15:00` = `seriesStart` | Yes — the derivation is safe, not an assumption |
| Split **885 / 1,152 / 20,634** | identical | Yes |
| **2,268 inside windows, 18,366 outside** | identical | Yes |
| Reference median 81.8620 and MAD 4.42426 | `81.862044` / `4.424260` | Yes |
| **All eight published quantile rows** (thresholds, test alerts, inside, outside, 4/4 hits) | every one of the 8 × 5 numbers reproduced from the raw CSV | Yes |
| "11 calibration alerts out of 1,152" at q = 0.99 | 11 for every method (57 at q = 0.95) | Yes |
| The specification's four `0.975` rows | `4.566579723695049 / 28 / 2170 / 1036`, `0.5555652392489324 / 28 / 5632 / 4263`, `−6.415545934270259 / 28 / 9661 / 8318`, `1.4936764075602469 / 28 / 8191 / 7002`, all 4/4 | Yes |
| `quantileOutcomes` covers every reachable index | 116 contiguous rows, 1036…1151 per method; `ceil(0.9 × 1151) = 1036`, `ceil(1.0 × 1151) = 1151` — no lookup can miss | Yes |
| Spot-check of the shipped ladder at indices 1036, 1123, 1151 × 4 methods | all 12 rows equal the native refit, **including q = 1 for Isolation Forest, where the hit count drops to 2/4** (packed `10`) | Yes |
| Direct-mode sweep levels (3 baseline + 1 isolation spot-checks) | thresholds, calibration alerts, test alerts, inside/outside and hits all reproduce | Yes |
| The `exceeds` rank encoding in `windowDetail` | for 4 methods × 3 thresholds × 4 window blocks, the lab's `exceeds > thresholdRank` rule gives per-block alert counts **identical** to `native_score > threshold` on the same 711 rows (48/48 exact) | Yes |
| 358 offerable thresholds per method contain no duplicates | confirmed, so `findIndex` is unambiguous | Yes |

Nothing recomputed disagreed with the manuscript, the JSX, the generated module or the evidence. The numerical core of this lesson is sound.

**Could not verify numerically:** the seeded compact-forest scores (`[0.4996 0.4135 0.3998 0.4613 0.7097]`) are a property of NumPy's `default_rng(17)` draw sequence, so they were confirmed by execution (A3) rather than derived; and the 1e-9 agreement claim against the packet's 21,786-row `nab-derived-scores.csv` was not re-run — I refitted from the CSV instead, which is the stronger check.

### A3. Displayed programs versus their published output

All five programs were extracted, written to the filenames the lesson names, and executed beside the served CSV and JSON. **Every one reproduces its displayed "Expected result" block byte for byte**, modulo the absent trailing newline in the stored string. That includes the seeded forest, both LOF programs, the kernel table and the full nine-line temperature output. No program prints an explanatory or cautionary sentence; the closest is `raise ValueError("This baseline requires a positive reference MAD.")`, which is an error path, not a disclaimer.

### A4. Model and lab logic read for defects

- `Exact` is a reduced BigInt rational with a positive denominator; `Exact.from` rejects more than the declared decimals. `35/24`, `841/660` and `77/30` really are exact on screen, and the "identical records ⇒ every score 0.5" null is an exact statement, not a rounding artefact.
- `isolationExpectations` memoises on `(from, to, depth)` over contiguous sorted runs — correct, because a threshold cut always yields contiguous runs. My independent integrator agrees on every probed state.
- `isolationPath` reports an unusable cut rather than silently ignoring it, and the leaf badge says "N rows remain; depth d" instead of claiming isolation. Matches the specification.
- `lofState` excludes the training row **by id**, not by coordinate, and a query excludes nothing — exactly the §5 contract. Duplicate references are refused with a reason rather than stabilised.
- `kernelBoundary` finds the positive runs by a 4,096-point scan plus 60 bisection steps per edge, and classifies `|g| ≤ 1e-12` as boundary, which absorbs the floating-point difference between `-gamma*(2a)²` and `-4*gamma*a*a`. A reference anchor therefore reads "exactly on the boundary" at every admitted gamma, as the prose promises.
- `quantileIndex = ceil(q(n−1))` matches NumPy's `method="higher"`; the strict `>` comparison is used everywhere, including the calibration counts.
- The `exceeds` device is the one piece of real engineering here, and it is correct: because no offerable threshold coincides with a score, `score > threshold_i ⇔ exceeds > i` holds even when two lists contribute equal threshold values. I verified the equivalence against native scores on 48 independent cases rather than trusting the generator's own assertion.
- `usePrediction` stores the input snapshot with the commitment and retires the answer when the snapshot changes — implemented in all six investigations, and the retirement message names the retired choice. This is the strongest part of the interaction design.
- Defects found by reading: the answer leak in the temperature lab's detail panel (S1), the always-false agreement sentence in `LofModeLab` (S2), and the k = 2-only prose in `LofNeighbourhoodLab` (S5).

## Part B — learning-experience checklist (reviewer's heuristic run)

| # | Item | Finding |
| --- | --- | --- |
| 1 | **Route** | Present: the `ad-route` paragraph immediately after the intro names sections 1–8, the §10 investigation and practices A–F, sends 9 and 11 with practices G–J to a later pass, and gives times. Both deeper sections carry "Deeper branch" in their `H2`. |
| 2 | **Cautions** | The §1 Callout is a genuine single home and says so ("we will not repeat this caution after every result"), and it is honoured for "score", "outlier" and "alert". The annotation caution is not: it appears in §10 prose and again, near-verbatim, in the lab caption two elements later (O1). Hedging density is roughly one phrase per 3.7 prose blocks — above the standard's 1-in-4-to-5 signal, though many instances are mechanism statements rather than disclaimers. No code prints a caution. |
| 3 | **Real question** | Opens on a machine running hot and returns to it in §10 with 22,671 real NAB rows, a downloadable CSV and annotation JSON, a pinned commit, a verbatim MIT notice, and a result the reader can judge (four window hits against 445 to 8,808 unmatched alerts). Real data present, provenance complete. |
| 4 | **Labs as investigations** | **Isolation**: prediction committed before reveal; five freely editable positions plus a depth cap; three fixtures (gap / regular / identical) whose contrasts I independently confirmed — 12 wins, endpoints tie, all scores 0.5 with no ranking. **LOF neighbourhood**: two predictions, free query, k and all six references; the k = 3 null (every query factor exactly 1) is real. **Fitting mode**: prediction committed; the agreeing and disagreeing coordinates both exist; but the explanation of agreement is wrong (S2). **Kernel**: prediction committed; anchor, gamma and query all free; the disconnected region at gamma 1 and the single interval at gamma 0.1 are both real. **Review queue**: prediction committed; five free inputs; but only half the question is recordable (S6). **Temperature**: two predictions committed together, 358 thresholds per method reachable, staleness on method or threshold change — and the hit count really does fall to 2/4 at q = 1, so the prediction is not a formality. Its detail panel, however, shows the alerting rows and their count before the commitment (S1). |
| 5 | **Figures** | Desktop (761 px): the kernel lab's two panels separate `+0.0697` from `−0.1413` clearly and the two positive slivers are visible as distinct bars; the temperature overview's window bands, threshold line and alert marks are all legible; the reach-floor and provenance figures are clean. Phone (350 px): the first-cut and threshold-ruler figures are legible, with stacked tied dots distinguishable. Baselines are present (the ε-equivalent threshold line, the zero line under `g`, the full temperature envelope). No caption apologises for its figure. SVG text is 11 units in a 340-unit viewBox, ≈ 10.3 px at phone width (O3). |
| 6 | **Connections** | `c(5) = 77/30` recurs as normaliser, leaf correction and lab readout with the link stated; `7/8` and `4/3` are tied across §4, §5, the fitting-mode lab and practice D; the reachability floor is explicitly contrasted with OPTICS, HDBSCAN and DBSCAN's `min_samples`; practice E's "why do they agree?" is the same exponent by two routes, and the kernel lab's caption repeats the link. Canonical facts present and correct: `c(n) = 2H(n−1) − 2(n−1)/n`, `s = 2^(−E[h]/c(ψ))`, depth limit `⌈log₂ψ⌉`, the tie-inclusive original LOF neighbourhood, the nu upper/lower bounds, `gamma='scale'` ≠ standardisation. The *citations* for two of them point at the wrong sections (S3, S4). |
| 7 | **Code** | Mechanism dominates in four of five programs. The isolation program spends 6 of ~55 lines on three `raise` blocks, two of which are `type(x) is not int` checks (O15). The LOF program has one guard line, the temperature program one. No printed disclaimers. |
| 8 | **Practice** | Ten tasks, each changing both numbers and context, with hints separate from solutions. Exact reproducible values in A, B, C, E, F, H and I — all recomputed and all correct. F and A explicitly send the learner into a lab to reproduce the figure, and F's four inputs are all within the lab's admitted ranges and decimal limits (checked). |
| 9 | **Screenshots** | Eleven captures, and the informative states were reached: a revealed temperature threshold with its verdict line, a *missed* LOF prediction with its explanation, the disconnected kernel region, the review queue. Gaps: no desktop capture of the first-cut or threshold-ruler figures, no phone capture of the isolation, LOF or fitting-mode labs, and — most relevant here — no capture of the temperature lab in its **uncommitted** state, which is exactly where S1 would have shown up (O12). |

## Ranked actionable findings

Severity: **blocking** = wrong on the page, or a record that asserts something untrue; **should-fix** = an inaccuracy, a defeated teaching mechanism or a mis-citation with a bounded fix; **observation** = recorded for the ledger.

### Blocking

**B1. The PowerShell activation line is destroyed by JavaScript escape processing and renders as a command that cannot work.**
Location: `src/learn/data/topics/anomaly-outlier-detection-isolation-forest-one-class-svm-lof.jsx` L113, inside `<CodeBlock language="bash">{'python -m venv .venv\n# Windows PowerShell:\n.\.venv\Scripts\Activate.ps1\n…'}`.
Evidence: in a single-quoted JS string `\.` → `.`, `\S` → `S`, `\A` → `A`. Evaluating the literal with `node` prints:

```
python -m venv .venv
# Windows PowerShell:
..venvScriptsActivate.ps1
# macOS or Linux:
# source .venv/bin/activate
```

The manuscript (`lesson.md` L124) has the correct `.\.venv\Scripts\Activate.ps1`. Every reader on Windows — the environment the project itself uses — is handed a broken first step, and the standard requires complete, working run instructions.
Fix: double the backslashes (`'.\\\\.venv\\\\Scripts\\\\Activate.ps1'` in the source literal, i.e. `\\` per rendered backslash) or move the block to `String.raw`. Then re-read the rendered block, because the same hazard applies anywhere else a path appears in a JS string literal — I checked the rest of this file and found no second instance.

**B2. The handoff asserts an independent review that had not happened, and contradicts the phase ledger, which still says implementation has not started.**
Locations: `LESSON-AUTHORING-HANDOFF.md` L85 (table row: "complete, independently reviewed, integrated", linking `docs/teaching/ANOMALY-DETECTION-INDEPENDENT-REVIEW.md`), L93 and L97 ("the disposition of the independent review"; "The phase ledger records implementation complete with the final source hashes"); `docs/teaching/lesson-delivery-progress.json`, topic entry.
Evidence: `docs/teaching/ANOMALY-DETECTION-INDEPENDENT-REVIEW.md` did not exist when this review began — the handoff's link was dead and its claim unsupported. The ledger entry reads `"implementation": {"status": "not-started"}` with `nextAction: "… Implementation has not started."`, and `git diff docs/teaching/lesson-delivery-progress.json` contains **no** line mentioning this topic, so it was never updated. The ledger's content checkpoint also still binds `ANOMALY-DETECTION-LESSON-DESIGN.md` at `d0d9acd7c82b0eaab…`, while the file is now `75c855b3d3c48d6e…` after the phase-two append. The standard is explicit that "a source comment claiming checks were run is not a fresh verification", that unknown states are recorded as unknown, and that an author's own reading is not independent review.
Fix: (a) update the ledger entry to `implementation: complete` with the final source hashes and a next action, and refresh the design-record hash in the content checkpoint; (b) restate the handoff row so it matches — it may now legitimately cite this document, but the sentence claiming the ledger records completion must not precede the ledger actually recording it.

### Should-fix

**S1. The temperature lab shows the alerting rows, and their count, before the learner commits a prediction.**
Location: `src/learn/components/lesson-labs/AnomalyTemperatureLab.jsx` L84–86 and L239 — `alertRowCount` and `visibleRows` are derived from `detailAlert(...)` with no `shown` guard, and `onlyAlerts` defaults to `true`; the table caption prints `Alerting rows in this block: ${alertRowCount}` unconditionally.
Evidence: opening the "Inspect the actual rows around one annotated window" disclosure before committing shows a table filtered to exactly the alerting rows, captioned with their count, for each of the four windows in turn — which settles the window-hit prediction outright and brackets the workload prediction. Only the `alerts` column is masked ("hidden until revealed") and only the figure's marks are gated by `shown`. This contradicts the visual specification ("new threshold-specific alert markers/counts are hidden until commitment", `visual-specifications.md` L179) and the lesson's own promise in the intro that "Every lab records a prediction before it reveals a result".
Fix: gate the filter and the count on `shown` — show every row in the block with the score column visible and the alert column masked until the reveal, then switch to the filtered view. One condition on `visibleRows` and one on the caption.

**S2. `LofModeLab` explains every agreeing coordinate with a statement that is never true.**
Location: `src/learn/components/lesson-labs/AnomalyDetectionLabs.jsx` L279–281: `` `Both give ${…} here. The two contracts still differ; this coordinate happens to keep the same neighbour set.` ``
Evidence: a new query always selects the reference standing on its own coordinate at distance 0, while the training row excludes that identity, so the neighbour sets can never coincide. Exhaustively over the six offered coordinates × k = 1…5, the factors agree in nine cases (k = 1 at all six coordinates; k = 2 at 0, 2, 20, 28; k = 5 at 0 and 28) and the neighbour sets are different in **all nine**. Concretely at coordinate 0, k = 2: the training row uses {P1, P2} and the query uses {P0, P1}, both giving 7/8. The sentence therefore teaches the opposite of §5's point, on the one screen built to make that point, and the specification had flagged exactly this risk ("Equality of one number must not be presented as proof that the APIs are interchangeable").
Fix: replace with something like "Both give 7/8 here, from *different* neighbour sets — the training row uses {P1, P2}, the query uses {P0, P1}. Two contracts can land on one number." The component already has both neighbour lists to hand.

**S3. The LOF paper's section range is wrong.**
Locations: `.jsx` L313 and `lesson.md` L798 — "sections 3 to 5 define reachability, tied neighbourhoods and the local bounds used in section 11".
Evidence: Breunig, Kriegel, Ng and Sander, SIGMOD 2000, actual structure — §1 Introduction, §2 Related Work, **§3 Problems of Existing (Non-Local) Approaches**, **§4 Formal Definition of Local Outliers**, **§5 Properties of Local Outliers**, §6 MinPts, §7 Experiments. `k-distance`, `N_k(p)` and `reach-dist_k(p,o) = max{k-distance(o), d(p,o)}` are Definitions 3–5 in **§4**; the upper/lower bound the lesson's §11 mirrors is Theorem 1 in **§5.2**. §3 is the `DB(pct, dmin)` critique and defines none of them. (The tie claim itself is correct and verbatim in the paper: "the cardinality of N_4(p) can be greater than 4, in this case 6".)
Fix: "sections 4 and 5". Same edit in the manuscript.

**S4. The Isolation Forest section pointers send the reader to the wrong place for two of the three things named.**
Locations: `.jsx` L312 and `lesson.md` L797 — "read tree construction and path normalisation in sections 2 and 3, then the subsampling discussion".
Evidence: Liu, Ting and Zhou, ICDM 2008 — §2 *Isolation and Isolation Trees* contains both the construction and the normalisation (Eq. 1 `c(n) = 2H(n−1) − 2(n−1)/n` and Eq. 2 `s(x,n) = 2^(−E(h(x))/c(n))`). §3 is *Characteristic of Isolation Trees*, i.e. swamping and masking. Sub-sampling and the depth limit the lesson quotes (`l = ceiling(log₂ ψ)`, `ψ = 256` default) are in **§4.1**. All three formulas the lesson states are verbatim correct; only the signposting is off.
Fix: "read tree construction and path normalisation in section 2, the swamping and masking discussion in section 3, and sub-sampling with the height limit in section 4.1."

**S5. The LOF neighbourhood lab's pair prediction is worded for k = 2 and the default references, but stays live when either changes.**
Location: `AnomalyDetectionLabs.jsx` L172 (prompt: "Query 4 sits 2 units from its nearest reference; query 17 sits 3 units from its nearest"), L175 (describe: "the tight group's radii are small, so query 4 looks under-supported beside it, while the loose group's radii absorb query 17's larger distances"), L177 and L179 (table captions "its two chosen neighbours" and "farther from its nearest reference, yet lower").
Evidence: at k = 3 the two factors are **exactly equal** — `LOF(4) = LOF(17) = 1`, which I recomputed independently — so `pairAnswer` correctly becomes "They tie" while the describe text still asserts the asymmetry and the second caption still says "yet lower"; "its two chosen neighbours" is wrong for any k ≠ 2. Editing any reference coordinate invalidates the "2 units"/"3 units" figures in the prompt. The snapshot retires the *answer* but not the *wording*.
Fix: interpolate the numbers (`k`, the two nearest distances) into the prompt and captions, and branch the describe text on `pairAnswer === 'equal'` — at k = 3 the true explanation is that every reference row's own neighbourhood now crosses the gap, equalising the lrd within each group, which is a better lesson than the one currently printed.

**S6. The review-queue lab asks two questions and can record only one.**
Location: `AnomalyDetectionLabs.jsx` L388–389 — prompt "What fraction of the alerts will be faults, **and will the budget cover them?**" with options covering only the fraction band.
Evidence: `answer` is derived from `counts.precision` alone; budget sufficiency appears only in the post-reveal `describe`. The specification (`visual-specifications.md` L141) required both: "expected useful-alert fraction bucket … **and whether B suffices**. Commit before displaying derived answer." A learner who commits "between 10 % and half" is graded on half of what they were asked.
Fix: either add a second committed yes/no field for the budget (the temperature lab already implements a two-field commit that can be copied), or shorten the prompt to the fraction alone and leave the budget as commentary.

**S7. The design record's "Departures from the manuscript" list is incomplete.**
Location: `docs/teaching/ANOMALY-DETECTION-LESSON-DESIGN.md`, phase-two section.
Evidence: three departures are recorded; at least these are not — (a) §4's Step-1 radius table moved into `ReachFloorFigure`; (b) §8's tie example moved into `ThresholdRulerFigure`; (c) the readiness-check table added in §13; (d) the isolation program gained two printed lines; (e) sections 3, 5 and 13 retitled; (f) two "can" qualifiers dropped (S8); (g) three specification items not implemented — the recorded budget prediction (S6), the practice-B four-row terminal fixture as a reachable lab state (O14), and the optional "change units ×10" button; (h) `RunnableExample` does not display `example.file`, although the lesson instructs the learner to save the blocks under five specific names (O7).
Fix: add them. Most are improvements or harmless; the record is what lets the next author tell a deliberate choice from a slip.

**S8. Two dropped qualifiers turn conditional claims into unconditional ones.**
Locations: `.jsx` L13 (heading 3, "an empty gap makes a point easy to separate") and L125 ("A global nearest-distance rule penalises it even though its spacing is internally consistent").
Evidence: `lesson.md` L72 and L205 both write "can". The second is the more consequential: whether a global rule penalises the loose group depends on where the global threshold sits.
Fix: restore "can" in both.

### Observation

**O1. The annotation caution is stated twice, two elements apart.** `.jsx` L249 ("Outside-window alerts are unmatched workload, not verified false positives. Inside-window rows are not individually confirmed faults either.") and `AnomalyTemperatureLab.jsx` L205 ("Alerts outside a window are unmatched workload, not verified false positives, and rows inside one are not individually confirmed faults."). Keep the prose sentence; shorten the lab caption to the early-alert point it alone makes.

**O2. The temperature lab's comparison tables are wider than the reader column.** In `anomaly-temperature-desktop.png` (761 px) the "outside windows" and "windows hit" columns are clipped and reachable only by scrolling the region — and "windows hit" is half the lesson's point. They are in a keyboard-operable `role="region"` with `tabIndex=0`, so this is friction rather than inaccessibility. Consider shortening the headings ("outside", "hit") or dropping the calibration-alerts column, which the readout already states.

**O3. Phone-width SVG labels are ≈ 10.3 px.** `anomaly-detection-labs.css` sets `.ad-plot svg text, .ad-figure svg text { font: 11px … }` in 340-unit viewBoxes; `.ad-figure svg` caps at 620 px so desktop is comfortable, but `.ad-plot` svgs scale to the viewport and land near 10 px at 350 px. The two mobile figures I opened are legible; the sibling DBSCAN lesson adopted a 12 px floor, so this is worth a deliberate decision rather than drift.

**O4. The NAB licence pin is load-bearing and should be documented as such.** `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140` is literally the relicensing commit — its message is `chore: MIT License (#408)`, dated 3 December 2024 — and its parent `5a98e4f1` carries **AGPL-3.0**. The MIT claim in the lesson and in `NAB-LICENSE.txt` is correct *at this SHA* and false for essentially all of the repository's history. `dataset-provenance.md` already warns "Do not replace this pinned license with an older tutorial's AGPL description"; adding the reason (the pin *is* the relicensing commit; never loosen it to a tag or branch) would make that warning self-explaining. Separately, `raw.githubusercontent.com/.../LICENSE` 404s at that commit — the file is `LICENSE.txt`, which is what the packet records.

**O5. NAB's own `data/README.md` describes three anomalies while `combined_windows.json` lists four windows.** This is an upstream inconsistency, and the lesson already handles it correctly by declining to map the prose onto the windows. Recorded so a future author does not "fix" the four into a three.

**O6. `nab-event-windows.json` is the whole 58-key `combined_windows.json`.** §10 calls it "annotations [that] contain four time windows" — true of the key the program selects, loose about the file. `dataset-provenance.md` states it precisely ("retained as downloaded, including unrelated dataset keys"). Optional one-clause fix in the lesson.

**O7. Program filenames are stored but never rendered.** `anomaly-detection-examples.js` carries `file` for each program and the lesson instructs "Save each Python block in its own file, named `isolation_example.py`, …", but `RunnableExample.jsx` renders only title, code and expected output. The blocks do appear in the stated order, so a careful reader can match them; printing `example.file` beside the title would remove the counting.

**O8. The "Before running" question precedes the heading it belongs to.** `Program` (`.jsx` L28) emits a `<Prose>` before `<RunnableExample>`, which emits the `<H3>` title. Reading order is question → title → code.

**O9. Dead call.** `AnomalyDetectionLabs.jsx` L33: `prediction.setChoice(prediction.choice)` inside `setValue` is a no-op.

**O10. Inconsistent "not sure" affordance.** The shared `Prediction` component marks an unsure commitment with `?`; the temperature lab's hand-rolled verdict (`AnomalyTemperatureLab.jsx` L136–137) marks it `≠` with the unsure styling.

**O11. Evidence disagrees with the module on one count.** `evidence/anomaly-temperature-data.json` says "240 direct thresholds per method"; the module ships 242 sweep levels and the design record says 242. The two extra are the out-of-range endpoints the lab uses only to bound its input, so both statements are defensible — but they should not differ.

**O12. Screenshot gaps.** No desktop capture of `FirstCutFigure` or `ThresholdRulerFigure`; no phone capture of the isolation, LOF-neighbourhood or fitting-mode labs; and no capture of the temperature lab *before* commitment, which is the state where S1 is visible. Structural assertions exist for those states; visual inspection does not.

**O13. The retrieval date is in the packet, not on the page.** `dataset-provenance.md` records 12 September 2026; the lesson gives the pinned commit instead, which is a stronger identifier. Recorded only.

**O14. Practice B's terminal state is not reachable in the isolation lab.** `isolationLimits.points` admits 2–6 positions and `isolationPath` handles the four-row case (the verifier exercises it), but `IsolationLab` renders a fixed five `NumberField`s with no add/remove control, so the `c(4) = 13/6` normaliser and `2^(-18/13)` cannot be produced on the page. Practice B does not claim otherwise — unlike practice A, which correctly says "Set these five positions in the isolation lab" — so this is a specification item silently dropped rather than a false statement.

**O15. Validation in the displayed isolation program.** `fit_forest` carries three `raise` blocks (6 of ~55 lines), two of them `type(x) is not int` checks that a learner is unlikely to trip. The standard prefers a named helper or a single finiteness check; the mechanism still dominates, so this is a preference, not a defect.

## Closure

The mathematics, the data handling and the numerical contracts of this lesson are in excellent shape. Every number I recomputed from first principles agrees — `c(5) = 77/30`, all five exact expectations and scores, `2^(-18/13)`, the whole LOF state including `35/24` against `35/32`, both kernel gammas and the `0.9145` crossing, the population arithmetic including practice F's `339.5` and `26.51 %`, and the entire temperature pipeline refitted from the raw CSV down to `885 / 1,152 / 20,634`, `2,268 / 18,366` and all eight published rows. The five displayed programs reproduce their output byte for byte. The `exceeds` rank encoding that lets the browser answer 358 thresholds per method without refitting is not just plausible but verified against native scores on 48 independent cases. The five-minute spacing that every displayed timestamp depends on was checked, not assumed, and holds.

What needs work is smaller but real: one displayed command that cannot run (B1), a record that claims a review which had not happened and a ledger that still says the work has not started (B2), one investigation that hands over its answer before the prediction (S1), one explanation that is false every time it appears (S2), two mis-cited section ranges (S3, S4), and prose in two labs that is hard-wired to a setting the learner is invited to change (S5, S6).

Recheck plan after the fixes: B1 needs the rendered block read in a build, nothing else. B2, S7 and the observations are documentation. S1, S2, S5 and S6 are component changes — after them, rerun `node scripts/verify-anomaly-detection-models.mjs` (it covers `lofModeComparison` and `alarmCounts`) and `node scripts/verify-anomaly-detection-browser.cjs`, and capture the temperature lab in its uncommitted state and the LOF lab at k = 3, which are the two pictures that would have caught S1 and S5. S3, S4 and S8 are prose edits in both the JSX and `lesson.md` and need no verifier rerun. No numerical verifier needs to be repeated for any finding in this review.
