# DBSCAN & Density-Based Clustering — independent phase-two review

Reviewed 12 September 2026 against the working tree at commit `6316d31cbd58010a633c74b1c5551ad849979a72` plus the uncommitted DBSCAN implementation. Reviewer did not author the packet or the implementation. No file under `src/`, `scripts/` or `public/` was edited; no state-changing git command was run.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | `scratch/dbscan-review/independent_checks.py` (disposable reviewer script, own first-principles DBSCAN; does not import the lesson's models or the author's scripts) with the project's `scratch/lesson-tools` Python: NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1. It recomputed the trail fixture at five radii and under reversal, the float exactness of every promised tie, `c₄`, the Iris chain from `load_iris` for four settings, the ε = 0.8 species cross-tab, the CSV against `load_iris`, the L4 analytic interval against the actual graph for all 1,536 grid combinations, practices D, E, H and the four-corner halving null, the ring gaps, the K-Means centers/ARI, the haversine distances and the memory arithmetic. `sha256sum` was run on every reviewed file. |
| **Read in full** | `LESSON-TEACHING-STANDARD.md` (sections cited in the brief), `DBSCAN-LESSON-DESIGN.md` including the phase-two section, `lesson.md`, `visual-specifications.md`, `data-provenance.md`, the structure and Iris/street entries of `author-calculations.json` and `supplementary-author-calculations.json`, all seven implemented source files, `dbscan-labs.css`, `RunnableExample.jsx`, the blueprint, the CSV header, and the three verifier scripts' assertion lists. All twelve evidence screenshots were opened and looked at. |
| **Reused, not rerun** | `evidence/dbscan-models.json` (23 grouped browser-model checks), `evidence/dbscan-native.json` (nine programs executed natively, 14 oracles), `evidence/dbscan-browser.json` (11 Playwright cases). Their recorded source hashes equal the hashes below for every file they name, so the evidence binds to the bytes reviewed. The build, the Playwright run and `verify-curriculum.mjs` were not repeated. |
| **Not done** | No beginner walkthrough (this is a reviewer's heuristic learning-experience assessment). The contrib `hdbscan` package is not installed, so the sklearn-versus-contrib `min_samples` convention was checked only against the cited sklearn documentation, not by execution. |

## Source versions reviewed (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/dbscan-density-based-clustering.jsx` | `e83b9cab482c79b8d31c3e514b3be61fa8575e182573d9adfaa882bf22c545e6` |
| `src/learn/components/lesson-labs/DbscanLabs.jsx` | `c16f79f096d5dbb8d4a97365672be036bee4bcd910fc17f498ebcb44b0a8fc9a` |
| `src/learn/components/lesson-labs/DbscanFigures.jsx` | `0a32bb4689a9994d902c98704ea1b40964f52cf9af2223d77f89c877864f27ba` |
| `src/learn/components/lesson-labs/dbscan-labs.css` | `2f237beccadd9b73cd5a161ba3e794e2add49643f84eef2bf52ae1e52c425901` |
| `src/learn/data/dbscan-models.js` | `15d7cdf4c507a9d2bcf256dde810bd8a87557f4450eb468d3e88ad7969b72ad4` |
| `src/learn/data/dbscan-iris-data.js` | `4361a654de47ef3f8b97643f7339accf15af8489dfdb238633c8a6b214c1cfae` |
| `src/learn/data/dbscan-examples.js` | `b27124dbfa1154b62879418e3d178e64201771b5b0e3b4597b8c764f0aa0ac2e` |
| `src/learn/data/curriculum/blueprints/dbscan-density-based-clustering.js` | `dab9435c28905f188596e21ea81e3b34d54cac50cc8548a360971bb277562784` |
| `public/learn-assets/dbscan/iris.csv` | `387c9585511b1d4ab2513075377d2f07c5cbf05e7a04fd934fcb254290a7a651` |
| `docs/teaching/drafts/dbscan-density-based-clustering/lesson.md` | `8e884b5d3177f098f44c7ab8732e227da3fa5fce048e71475d5ee4698d46e7f0` |
| `docs/teaching/drafts/dbscan-density-based-clustering/visual-specifications.md` | `b089b3176ed3b74e9b9b14087e0097623cc697460a3d93e4472ae2fcaa27813a` |
| `docs/teaching/drafts/dbscan-density-based-clustering/data-provenance.md` | `f631945dfa2a61411c6b4550be24ce82df4da14331235d969039e3fa2e31767c` |
| `docs/teaching/drafts/dbscan-density-based-clustering/author-calculations.json` | `58fca03512d78077d789bbb073258015fe1c20778c4a4f54fd8cedc1f01d57a6` |
| `docs/teaching/DBSCAN-LESSON-DESIGN.md` | `b6593d6a832e15e5ebdc6ff85a98887bb1a7b2c354851b4e24414291a55f2001` |
| `scripts/verify-dbscan-examples.py` (read only) | `e367fc8aee18630680db06dd74958ec364b9c2f98a480aa1579b686436159247` |
| `scripts/verify-dbscan-models.mjs` (read only) | `e32ada102f90175e9064dad508b098efcde12dc35585d61848c46dc37073d239` |
| `scripts/verify-dbscan-browser.cjs` (hash only) | `b8b5a54a45a4b2e66c8a2fcf505fcf68b6b433dfa714542df9d981249239a81c` |

A hash proves identity, not correctness.

## Part A — correctness

### A1. Manuscript explanations and visual contracts preserved?

Section by section, the JSX carries the manuscript's fourteen sections, the first-pass route, the nine programs at their manuscript positions, the twelve practice tasks with hints and separate solutions, the seven inline figures and the four investigations. The following are the only deviations found; each is judged.

| Location | Deviation from manuscript / specification | Judgement |
| --- | --- | --- |
| `DbscanLabs.jsx` L2 (`DbscanMetricLab`) | Specification L2 allows "learner-selected row coordinate edits on a .125 lattice". Implemented controls are the four/five-row toggle, per-axis factors from a fixed list, both radii, m and the cited pair; no coordinate editing. Not listed in the design record's "Specification deviations". | Acceptable: factors × radius already give outcomes the prose has not resolved (e.g. x × 2, y × 0.25). Should be recorded as a deviation. |
| All four labs | Shared contract step 2 asks for "a prediction plus a short reason". No lab has a reason field. Not recorded as a deviation. | Acceptable: a free-text reason cannot be compared; the recorded, compared prediction is what the standard checks. Record it. |
| `DbscanFigures.jsx` F4 | Specification asks for "a numeric table [with] each group's points, threshold and resulting types" and "two stacked detail panels if needed" or a magnified inset. The figure has the interval bars and two strips but no table or inset; the adjacent lab supplies a group/requirement table and a per-row readout. | Acceptable in substance because the lab sits directly below; see finding 3 on perceptibility of the strips. |
| `dbscan-density-based-clustering.jsx` §6 | Manuscript: "independently standardizing the axes **can** distort physical proximity". JSX: "independently standardizing the axes distorts physical proximity". | Dropped qualifier that strengthens a conditional claim into an unconditional one (two axes with equal spread are barely distorted). Restore "can". Finding 5. |
| `dbscan-density-based-clustering.jsx` practice G | Solution adds "the Iris lab's snapshot comparison does exactly that" after "evaluate both settings on their common retained IDs if comparing conditional agreement". The lab's comparison table reports the intersection size and the **agreement between the two snapshots' partitions** on common rows (`compareReports.ariCommon`); it does not recompute either setting's species ARI or silhouette on the common rows, which is what "conditional agreement" means in §7. | Inaccurate description of the lab. Finding 2. |
| `dbscan-density-based-clustering.jsx` §10 | Manuscript sentence "It is a teaching step toward the hierarchy, not a full HDBSCAN implementation" is dropped. | Justified: the program title "Mutual reachability and a minimum spanning tree by hand" carries it and one hedge is removed. |
| `DbscanFigures.jsx` F1 caption | Roster printed in index order "D, E, I"; manuscript writes "D, I, E". | No consequence. |
| `dbscan-examples.js` Program 1 | Manuscript output block has the typo `F1 core`; the executed block prints `F 1 core`. | Correct in implementation. |
| `DBSCAN-LESSON-DESIGN.md` phase two | Says the blueprint is "awaiting registration in the shared blueprint index". `src/learn/data/curriculum/blueprints/index.js` already imports and maps it (lines 13, 128), and `generated/navigation.js` lists the topic at order 70 with the correct prerequisite. | Record is stale, not a defect. |

Everything else checked line by line against `lesson.md` — the §1 counts, the three-type table, §2 components and distances, §3 vocabulary, §5 radius table and `c₄` vector, §6 four-corner null and haversine constant, §7 grid, Program 3 table, §8 fixture and decision table, §9 OPTICS rule, §10 mutual-reachability examples and stability arithmetic, §11 memory arithmetic, §12 policy, all twelve practice statements and solutions, §14 references — is a faithful transcription. The §1 callout is a legitimate single home for the noise/border caution and correctly adds the border half.

### A2. Independent recomputation (own script, no lesson code)

| Claim | Independent result | Agrees? |
| --- | --- | --- |
| Trail counts at ε = 1, m = 4 | `[4,4,4,5,5,4,4,4,3,1]`; A–H core, I border, J noise | Yes |
| Trail table at ε = 0.125 / 0.5 / 0.75 / 1 / 1.25 (core, border, noise, components) | (0,0,10,0) (4,4,2,2) (8,0,2,2) (8,1,1,2) (9,0,1,1); I is noise → noise → noise → border → core | Yes; matches §5 table and the L1 caption |
| Promised exact ties are exact in binary floating point | `d(D,I)==1.0`, `d(I,E)==1.0`, `d(I,C)==1.25`, `d(I,F)==1.25`, `d(A,C)==0.5`, `d(A,D)==0.75` all `True`; moved I: `hypot(1, 0.125) = 1.00778 > 1` | Yes — all coordinates are dyadic, so the closed-boundary merges at 0.375 and 1.25 are real ties |
| `c₄` in A–J order | `[0.75,0.5,0.5,0.75,0.75,0.5,0.5,0.75,1.25,2.75]` | Yes |
| Reversal at ε = 1 | Types identical; I with D's component forward, with E's component reversed | Yes |
| sklearn `DBSCAN(eps=1, min_samples=4)` on the trail | labels `[0,0,0,0,1,1,1,1,0,-1]` | Yes (self-counting, closed boundary) |
| Iris, `load_iris` → `StandardScaler` → `DBSCAN(0.5, 5)` | 2 clusters, sizes 71/45, 116 assigned, 34 noise, silhouette 0.656, ARI all 0.442, ARI assigned 0.631; noise IDs equal the Program 3 output list | Yes |
| Iris standardized ε = 0.5, m = 10 (practice G) | 3 clusters (37/14/10), 61 assigned, 89 noise, ARI assigned 1.000, ARI all 0.279; 61/150 = 40.7 % | Yes |
| Iris raw cm ε = 0.5, m = 5 (practice K) | 2 clusters 84/49, 133 assigned, 17 noise, silhouette 0.735, ARI all 0.521, ARI assigned 0.607 | Yes |
| Iris standardized ε = 0.8, "a group related to setosa" (§7 prose, not covered by any verifier) | cross-tab cluster 0 = 49 setosa / 0 / 0; cluster 1 = 0 / 50 versicolor / 47 virginica | Yes — the prose is exact |
| `iris.csv` vs `load_iris` | SHA256 `387c95…a651` as recorded; header `row_id,sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,species`; 150 rows; IDs 0–149 in order; features and species byte-equal to the loader; row 34 `[4.9,3.1,1.5,0.2]`, row 37 `[4.9,3.6,1.4,0.1]`; LF only | Yes |
| **L4 analytic interval formula** `exists ⇔ max(0.125, spacing) < offset − 0.375` — the author checked four offset/spacing pairs; I brute-forced all 12 offsets × 8 spacings × 16 radii = 1,536 combinations from first principles and also required that the three components be exactly the three intended groups | 0 mismatches | Yes — the formula is correct over the whole control range, including the offset 0.625 edge (gap 0.25) |
| Practice D | m = 3: core, core, core, noise; deduplicated unweighted: noise, noise; m = 1: labels `[0,0,0,1]` | Yes |
| Practice E five rows | Original: T noise; y × 100 and ε × 100: T core and in P–Q's component; four corners under the same faulty conversion: `[0,0,1,1]` unchanged | Yes |
| §6 four corners, y × 0.5, ε = 1 | one component | Yes |
| Practice H via `cluster_optics_dbscan` | reach `[∞, .9, .5]`, core `[.3, .4, .5]`, ε = .6 → `[0,1,1]` (second row starts cluster); core `.8` → `[0,-1,0]` (noise, next row joins the earlier cluster) | Yes, including the "depends on ordering state" remark |
| Rings | inner gap 0.5176, outer 0.5229, minimum inter-ring distance 2.0; sklearn DBSCAN 48 core, ARI 1.0; `KMeans(2, n_init=10, random_state=0)` centers ±(0.412869, 1.540847), symmetric about the origin, ARI −0.016216 — equal to the embedded `ringKmeans` | Yes |
| Haversine example | adjacent 1.112 km, endpoints 2.224 km, ε = 2/6371 = 0.000313922 rad | Yes |
| §11 / practice J arithmetic | 2.5 × 10⁹ entries, 2 × 10¹⁰ bytes = 18.63 GiB; 10¹⁰ × 8 = 8 × 10¹⁰ bytes | Yes |

Hedging note relevant to A2: none of the recomputed numbers disagreed with the manuscript, the JSX, the examples file or the embedded native data.

### A3. Displayed programs versus manuscript output blocks

All nine `dbscan-examples.js` `expected` blocks equal the native stdout recorded in `dbscan-native.json` (same `sourceHash`), and each equals the manuscript's derived value: Program 1 (with the manuscript's `F1` typo corrected), Program 2 (`c₄` vector, sorted vector, 8), geographic `[0,0,0,-1]`, Program 3 (the four table rows and the noise-ID lists; the 0.3 row's `19 11` core/border sums to the 30 assigned flowers the prose cites), rings (`48`, `1.0`, `-0.016`), OPTICS (ordering, reachability with `inf`, labels), MST (`c₄`, 9, `0.75 1.25`), HDBSCAN (labels, probabilities with I at `0.6`), duplicates (`[0,0,0,-1]`, `[-1,-1]`). `RunnableExample` prints only the title, the code and an "Expected result" block; no program prints an explanatory or cautionary sentence.

### A4. Model and lab logic read for defects

- `dbscan()` in `dbscan-models.js`: closed neighbourhoods, self-counting, weights as multiplicities, only core rows pushed to `pending`, `eligible` lists every core neighbour's component for a border row. Matches the manuscript's algorithm; my first-principles results equal its recorded outputs at every checked setting.
- `standardize()` uses population scale (ddof 0), as `StandardScaler` does; confirmed indirectly by the exact Iris agreement.
- `intervalFixture()` bounds and formula: verified exhaustively (A2).
- `OpticsOrderingFigure` start rule `reach undefined or > ε, and core ≤ ε` and table role text match `cluster_optics_dbscan`.
- L1 `lattice()` snaps to 0.125 multiples (exact); radius sliders step 0.125 (exact). L3 typed radius allows 0.01 steps — harmless, as no tie is promised there.
- L4 "recovers all three" is defined as three components and no noise; on this line with m = 3 that is equivalent to the intended groups (confirmed by the 1,536-case brute force with the stricter group test).
- L3 comparison: `compareReports()` computes agreement between the two snapshot partitions on common rows; it does not compute species ARI on common rows (see finding 2). For the practice G pair the two happen to coincide numerically (0.848) because the m = 10 clusters are pure species subsets — a learner cannot tell that from the label.

## Part B — learning-experience checklist (reviewer's heuristic run, with evidence)

| # | Item | Finding |
| --- | --- | --- |
| 1 | **Route** | Present: `db-route` paragraph immediately after the intro names sections 1–8, the first lab, Programs 1–3, practice A–E and the Iris report; sections 9–12 carry "Deeper branch" in their `H2`; the rings comparison is a collapsed `details`. |
| 2 | **Cautions** | One §1 callout for noise/border meaning; metric reporting in §7; suitability in §8's table; library conventions in §10; new rows in §12. Hedge count over 99 `Prose` blocks: 1 "does not prove", 1 "alone does not", 1 "not a claim", 2 "does not establish", 6 "is not a …", 1 "not a proof", 1 "not a verdict" — about one per 7–8 paragraphs, under the standard's threshold. One cluster remains: the optional rings block says the same thing three times ("not a proof that DBSCAN is preferable for every task", "not a verdict that one method is better in general", "not general scientific quality") and the closing paragraph restates all fixture disclaimers. No program prints a caution. |
| 3 | **Real question** | The flower-trail survey opens the lesson; the same flower question returns in §7 with the real 150-row corrected Iris copy, a working download, coverage over 150, and the species cross-tab claim that I verified. Real data is present and provenance is in the Sources. |
| 4 | **Labs as investigations** | **L1**: two predictions (component count, selected row's type) recorded via `select`, compared after commit, feedback names the count and eligible components; result hidden until commit; stale invalidation works (screenshot). Free controls: any row's x/y, ε, m, order, first row — well beyond presets. Fixture contrasts and nulls (0.75/1/1.25, reversal, moved I, duplicates, m = 1) confirmed by my own recomputation. **L2**: yes/no prediction recorded and compared with the cited pair's before/after distances; factors and radii free; buttons only set controls; five-row contrast versus four-corner accidental null both confirmed. **L3**: group count and coverage band recorded and compared; species hidden until a reveal after commit; any ε (slider or typed), m, raw/standardized; m = 5 vs m = 10 contrast confirmed; the snapshot comparison shows the common-ID intersection but not conditional scores on it (finding 2). **L4**: existence and tested-radius predictions recorded and compared; offset, spacing and radius free; analytic interval matches the actual graph for every grid combination; the "I need to inspect" option is always scored "Not this time" (finding 6). |
| 5 | **Figures** | Desktop (761 px captures): F2's two components, dotted attachments and struck arrow are clear; F3's four tied 0.5 values, four 0.75, the 1.25 step and J's 2.75 are all distinguishable on one truthful axis with the ε = 1 line; F5's E bar rising above the cut with its 0.75 core mark and "core start" label is readable, though the y-tick "1.5" abuts the "core start" label at A; F7's rings, edges, centers and bisector are clear. Phone (350 px captures): F1 scrolls horizontally with 13 px text but the in-SVG strip captions are clipped ("→ 5 ro…") until scrolled; F4 and F6 render with labels of roughly 7–9 px (CSS `8.5px` in a 340-unit viewBox at ≈ 320–360 px width; the lab's interval SVG uses `7.5px` in a 320-unit viewBox), below the specification's 12 px floor (finding 1). In F4 and in the L4 square plot the four rows of each dense group overlap into one blob at 5 px (figure) and 3.7 px (lab) spacing; the dense-versus-sparse contrast is visible, the row count is not (finding 3). Baselines present (ε line, 0 origin, both requirement intervals). No caption apologizes for a figure. |
| 6 | **Connections** | `c₄` is linked across the count test, the sorted curve, OPTICS core distances and mutual reachability in prose and in the MST program's note; the core-graph proof is tied to the expansion algorithm; the closed boundary is named as the reason for the exact 0.375 merge in figure, lab and practice F. Canonical facts (closed ≤, self count, DBSCAN*, border-order dependence, sklearn-vs-contrib convention, no `predict`) are present. |
| 7 | **Code** | Program 1: 15 of ~30 lines are the mechanism, no guards; Program 3: one scaler line, one loop, no validation; MST: construction dominates. The browser model's `checkPoints` lives in one helper. No printed disclaimers. |
| 8 | **Practice** | Twelve tasks with changed numbers and contexts. Exact reproducible values in A, D, E, F, G, I, J, K; all of those I recomputed agree. K asks for an independent radius with a stated rubric. |
| 9 | **Screenshots** | Informative states were captured for L1 (after a matching prediction, roster and table), L3 (species revealed), L4 (baseline after commit), F2/F3/F5/F7 at desktop, and F1/F4/F6 plus L2 at phone width. Gaps: L2 is captured only in its pre-commit state (the five-row faulty conversion result is asserted by Playwright but no picture of it was looked at); no phone capture of L1 or L3, the two most crowded labs; no desktop capture of F1, F4 or F6. |

## Ranked actionable findings

Severity: **material** = affects correctness or the learner's ability to use the teaching surface as promised; **minor** = inaccuracy or friction with a small fix; **none** = recorded for the ledger only. Fixes are the least intrusive that do not degrade the teaching surface.

### Material

**1. Phone-width labels in the mid-width figures and the interval-lab SVG render at ~7–9 px, below the specification's 12 px floor.**
Files/locations: `src/learn/components/lesson-labs/dbscan-labs.css` rules `.db-figure > svg.db-svg-mid text { font-size: 8.5px }` and `.db-interval-svg text { font: 7.5px … }`; consumers `CoreRadiusFigure` (F3), `IncompatibleIntervalFigure` (F4), `OpticsOrderingFigure` (F5), `StabilityTreeFigure` (F6) in `DbscanFigures.jsx`, and the interval SVG in `DbscanIntervalLab` (`DbscanLabs.jsx` ~line 320).
Evidence: viewBox widths 320–340 units scale to ≈ 290–360 px on phones, giving 7.2–9 px text; `dbscan-interval-figure-mobile.png` and `dbscan-stability-mobile.png` show it. The implementer fixed exactly this problem for F1/F2 (13 px in a `db-scroll` region) but did not apply it to the other four figures or the lab SVG.
Fix: wrap those SVGs in the existing `.db-scroll` region (`min-width: 560px` already defined) so the scale factor is ≥ 1.65 and text ≥ 14 px, or raise the two font-size rules to ≈ 12–13 px viewBox units and re-check F5's "restart"/"core start" and the interval-lab bar captions for collisions. Re-capture the four figures at 320 px afterwards.

### Minor

**2. Practice G claims the Iris lab evaluates both settings on common rows; the lab shows inter-snapshot agreement instead.**
Files/locations: `dbscan-density-based-clustering.jsx` practice G solution ("the Iris lab's snapshot comparison does exactly that"); `DbscanLabs.jsx` `DbscanIrisLab` comparison table row `groups … partition agreement (ARI) on common rows`; `dbscan-models.js` `compareReports`.
Evidence: for standardized ε = 0.5, the 61 rows assigned at m = 10 are a subset of the 116 at m = 5; species ARI on those 61 rows is 0.848 for m = 5 and 1.000 for m = 10 (silhouette 0.690 / 0.710). The lab shows only "0.848" labelled as agreement between the two snapshots.
Fix (either): add one table row "ARI vs species on the common rows" computed as `adjustedRandIndex(common.map(i => irisSpecies[i]), common.map(i => report.fit.labels[i]))` for each snapshot, shown only after species are revealed (check values 0.848 / 1.000 for the practice G pair); or change the practice G sentence to "the Iris lab's snapshot comparison shows the common retained rows and how far the two partitions agree on them."

**3. Dense-group rows are not countable in F4's strips or the L4 square plot.**
Files/locations: `DbscanFigures.jsx` `IncompatibleIntervalFigure` (`px = 24 + v·40`, glyph radius 5 at 5 px spacing); `DbscanLabs.jsx` `DbscanIntervalLab` `SquarePlot` (≈ 3.7 px spacing, 12 collinear rows in a square).
Evidence: `dbscan-interval-desktop.png` and `dbscan-interval-figure-mobile.png` show each dense group as one pill-shaped blob; the specification anticipated this ("if a magnified inset is used, give its own axis").
Fix: add a small magnified inset (0 to 1.25 m, own axis labels) beneath strip 1 and strip 3 of F4, drawing the eight dense rows at ≈ 30 px spacing with their types; optionally replace the L4 square plot with a one-row strip plus the same inset. The numeric tables already carry the exact values, so this is a perceptibility improvement, not a correctness repair.

**4. F1 strip captions live inside the scrolling SVG and are clipped on phones.**
Files/locations: `DbscanFigures.jsx` `TrailStrip` `<text x="8" y={y − 34}>{caption}</text>`.
Evidence: `dbscan-roster-mobile.png` shows "D selected: interval [−2, 0] holds A, B, C, D, I → 5 ro" cut at the region edge.
Fix: render the two captions as HTML `<p>` elements above the scroll region (they are already available as strings) and keep only the interval endpoints inside the SVG.

**5. Dropped qualifier in §6.**
File/location: `dbscan-density-based-clustering.jsx`, §6 paragraph ending "independently standardizing the axes distorts physical proximity. Choose deliberately."
Fix: restore the manuscript's "can distort".

**6. L4's "I need to inspect the intervals" option is always scored as a miss.**
File/location: `DbscanLabs.jsx` `DbscanIntervalLab` prediction field `exists`, option `inspect`; shared `Prediction` component marks any non-matching value "Not this time".
Fix: treat `inspect` as neither match nor miss in the feedback sentence (e.g. "You chose to inspect first: the answer is …"), or drop the option. Small `Prediction` change or a per-field `neutral` value.

**7. Iris slider label says "scaled feature space" even when raw centimetres are selected.**
File/location: `DbscanLabs.jsx` `DbscanIrisLab`, `Field label="Radius ε in the scaled feature space"`.
Fix: label it "Radius ε in the fitted feature space" and let the representation select say which units apply, or switch the label text on `pending.representation`.

**8. Triple hedge in the optional rings block.**
Files/locations: `dbscan-density-based-clustering.jsx` rings `details` paragraph 2 last sentence; `DbscanFigures.jsx` `RingsFigure` closing sentence; rings `Program` note.
Fix: keep the sentence in the prose paragraph, delete the figure's "it is not a verdict …" clause and shorten the program note to "These score the constructed ring identities."

**9. Undocumented specification deviations.**
File/location: `docs/teaching/DBSCAN-LESSON-DESIGN.md`, "Specification deviations" list.
Fix: add (a) L2 has no per-row coordinate editing, (b) no lab has a free-text "reason" field, (c) F4's per-row table is provided by the adjacent lab rather than the figure; and correct "awaiting registration" — `blueprints/index.js` already registers the blueprint.

### None (ledger only)

- F1 roster prints "D, E, I" (index order) where the manuscript writes "D, I, E".
- F5: the "1.5" axis tick abuts the "core start" label at A on desktop; legible, cosmetic.
- F7 prints the literal "ring ARI 1.0" while the K-Means ARI comes from embedded data; the value is natively verified.
- Evidence gaps for the ledger (Part B item 9): no post-commit picture of L2, no phone capture of L1/L3, no desktop capture of F1/F4/F6. Structural assertions exist for those states; visual inspection does not.
- The sklearn-versus-contrib `min_samples` convention was read against the sklearn 1.9.1 HDBSCAN Notes, not executed (contrib not installed); the lesson attributes it to that source correctly.

## Closure

Correctness: every recomputed number agrees with the manuscript, the JSX, the executed outputs and the embedded native data; the analytic interval formula holds over the entire control grid; the CSV is byte-equal to `load_iris`; every promised tie is exact. The one material finding is a rendering contract (phone label size) that the implementer already solved for two figures and can extend to the rest with the same pattern. After finding 1 is fixed, re-capture the four figures at 320 px and the interval lab at 390 px; after finding 2, run `node scripts/verify-dbscan-models.mjs` (the comparison helper is covered there) and re-read practice G. No other verifier needs to be rerun for the minor findings, which are prose, label and layout changes.
