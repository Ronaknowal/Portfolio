# Clustering Evaluation & Validation — independent review

12 September 2026. Topic `clustering-evaluation-validation-silhouette-ari-nmi`. Reviewer did not author the manuscript, specifications, models, labs or verifiers. No file under `src/`, `scripts/` or `public/` was changed; no state-changing git command was run.

## Reviewer statement

| Activity | What was done |
| --- | --- |
| Read | `LESSON-TEACHING-STANDARD.md` (sections 2, 5, 6, 11 and the learning-experience checklist); the design record including its phase-two section; `lesson.md`, `visual-specifications.md`, `data-provenance.md`; the topic JSX, both lab/figure components, `clustering-evaluation-models.js`, `clustering-evaluation-data.js` (structure and stored constants; the 150 rows were checked by hash and by comparison with `load_iris`, not read line by line), `clustering-evaluation-examples.js`, the blueprint, the CSV header, the three evidence JSON records, and the `sourceHashes` and import lines of the three verifiers. |
| Executed | One Python script (Python 3.12, NumPy 2.3.5, scikit-learn 1.9.1 from `scratch/lesson-tools`; written to `scratch/` for the run and deleted afterwards) that recomputed every bounded value listed in Part A from definitions, without importing the lesson's models; one Node probe of the split solver's tie comparator, which imports `clustering-evaluation-models.js` only to exercise that comparator on random supported inputs; `sha256sum` over the reviewed files. |
| Reused, not rerun | `evidence/clustering-evaluation-models.json` (26 grouped model checks), `evidence/clustering-evaluation-native.json` (six displayed programs executed, 11 oracles), `evidence/clustering-evaluation-browser.json` (12 Playwright cases) and the twelve screenshots, each opened and inspected. The build, browser suite and curriculum verifier were not rerun. |
| Not done | No beginner walkthrough; Part B is a reviewer's heuristic assessment with evidence, not a learner study. No live rendering beyond the author's screenshots. |

Author calculation JSONs (`author-calculations.json`, `visual-input-calculations.json`, `report-author-calculation.json`) were read for their values and compared with my recomputation; they were not treated as oracles.

## Reviewed source versions (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/clustering-evaluation-validation-silhouette-ari-nmi.jsx` | `64cba704985b012e596d3c238c80252ed06a47cbd74108a4c59f54569f3ba28b` |
| `src/learn/components/lesson-labs/ClusteringEvaluationLabs.jsx` | `3bc40673e6dda100b7b4a20d72acc09dab33738f333b78ce1971d761512e07ae` |
| `src/learn/components/lesson-labs/ClusteringEvaluationFigures.jsx` | `6dffd2e164e1bbf73043f98d5194d1e9b3eb5e0a9b69ddc29bfd947c66555bc1` |
| `src/learn/components/lesson-labs/clustering-evaluation-labs.css` | `b9f1a76d7109f1e8f4ada1bc09457651bb8e4d40cfb7904ae3038aeeb3aaac15` |
| `src/learn/data/clustering-evaluation-models.js` | `72542b8687be0fcdbdebddba730cf7064a5f3711696c3d0abe0dbde0b5ded9d1` |
| `src/learn/data/clustering-evaluation-data.js` | `e0568b62afc3f72ea2f1ca587db59554ccecec45769484222a2138bb14c51b07` |
| `src/learn/data/clustering-evaluation-examples.js` | `0395823733b0bf64c8684e25b059f526064dd5fd41609ec3d31b3fd526e2ac13` |
| `src/learn/data/curriculum/blueprints/clustering-evaluation-validation-silhouette-ari-nmi.js` | `bc58927e0091123a854906b1ece87e81a0097ff0452c5dd4332fabf0d68cfaba` |
| `public/learn-assets/clustering-evaluation/iris.csv` | `13c9255444bce09fd9df60cccfec34d5b0a0eeebc06008fb4e2b836aa2caf78a` |
| `docs/teaching/drafts/.../lesson.md` | `4f35ac5d1c7ea4bdfffbf974e52517183a6e5f543359651ff0cd0f10244400ff` |
| `docs/teaching/drafts/.../visual-specifications.md` | `6546b674535f7ea43a6bab30df77cefb2caf9ef733fc3c0fb54211b78e52ed5f` |
| `scripts/verify-clustering-evaluation-examples.py` | `aa7b7862e5546fee556cc765fde09f922e9b6b9cf61d2feb517bfb7dd3ee0f9e` |
| `scripts/verify-clustering-evaluation-models.mjs` | `26fa11cf2fd08c86654d13951e72b3d4c1589a82bd0531e0ad135f74eef89a97` |
| `scripts/verify-clustering-evaluation-browser.cjs` | `af6ec921e48d23403229045bf3542bc2eba5fea55b8c0fcbda3d0eacc5eae7b3` |

The browser record's `sourceHashes` match all nine of its listed files at these versions. The model record's hashes for the two lab/figure components and the CSS are older (`439caf…`, `0ed888…`, `0d1730…`) because those files changed in the screenshot-defect repair after the model verifier ran; the model verifier imports only `clustering-evaluation-models.js` and `clustering-evaluation-data.js`, whose hashes match, so its result still applies to the models (see finding M-4).

## Part A — correctness

### A1. Manuscript explanations and visual contracts preserved?

Read in order against `lesson.md` and `visual-specifications.md`.

- All twelve sections, headings, formulas, tables, the caution callout, the first-pass route, six programs, eight practice tasks with hint-before-answer, the references and the DBSCAN successor link are present in the JSX. Formula transcriptions (a/b/s, E[RI], ARI, MI, arithmetic NMI, AMI, hypergeometric P(R = r), VI, W/B, CH, Gap) match the manuscript; the split `\begin{gathered}` layouts are equivalent.
- The seven figures and five labs exist where the specifications place them. V1–V4 precede their abstractions; V5 follows Program 4; V6 sits in section 9; V7 after the fit/selection/report explanation.
- Deliberate drops, all justified by "say each caution once": the ring section's closing "A positive silhouette is not enough to declare the ring slices … correct"; section 8's sentence telling the reader to freeze the partition and use a common multiplier (the lab question and caption now carry it); section 11's "Benchmark actual data sizes only when timing is the question" and "Do not label it exact silhouette". None removes a condition that matters.
- The manuscript's a/b/s table in section 3 moved into V2's accessible table; the section 9 numbers (0.469841, 0.838314, 5/6) moved into V6's titles and table. Both remain visible in ordinary reading.
- Recorded specification deviations (V2 location labels instead of tick numerals, V6 ✕ chip with screen-reader text, shorter L4 weight labels and mode option, `select` + button predictions, typed coordinates only) are all permitted by the specifications or are neutral for teaching; I agree with the reasons.
- Contracts that the specifications made explicit and the implementation honours: predictions start unset, are committed with a draft, are compared after Apply and go stale on a further edit (all five labs); undefined silhouette states keep the points and name the reason (L1); constant labelings show a degenerate null instead of a zero mean (L3); the L3 histogram recomputes for unbalanced margins (56 assignments checked in the browser record); L4 rescoring never refits and marks ARI/AMI "same memberships"; L5 solves exactly, shows the alternatives table and allows k = 1.
- Not implemented and acceptable: the optional 36-entry distance table in L1; L2's separate "display-name" field (the rename button demonstrates the invariant, and colours follow first-observed order so a rename does not shuffle them); the optional common-ID lane in V6; the compare mode of L4 carries no prediction control (the specifications describe it as an inspected comparison state, not the investigation).

### A2. Independent recomputation (no lesson model called)

| Claim | Where | Independent result | Agrees |
| --- | --- | --- | --- |
| Six-point silhouettes and mean | §3, V2, Program 1 | [0.8125, 0.857143, 0.75, 0.75, 0.857143, 0.8125], mean 0.806548 | yes |
| C moved to R | §3, L1 | s(C) = −0.75, vector [0.846154, 0.818182, −0.75, 0.589744, 0.644444, 0.607843], mean 0.459394 | yes |
| C at 3, original groups | Checkpoint | a = 2.5, b = 5, s = 0.5 | yes |
| Common scale ×2; all-zero coordinates; one group | L1 fixture table | unchanged; all 0; undefined | yes |
| Singleton state `000112` | L1 fixture table | [0.8, 0.846154, 0.727273, 0.5, 0, 0]; the two singletons are 0 by convention | yes |
| −1 as a group / retained five | V6 | 0.469841 / 0.838314, coverage 5/6 | yes |
| Ring vs slice means | V3 | 0.0745943140 / 0.3206129287 | yes |
| CH, DB, Dunn for six points | §4 deeper branch | 73.5, 4/21 = 0.190476, 2.5 | yes |
| D/E swap: S, A, B, M, RI, ARI, E[RI] | §5, Program 2 | 6, 12, 12, 28, 4/7, 1/8, 25/49 | yes |
| Cross `00110011`: RI, ARI, NMI, AMI | L2, practice 2 | 3/7, −1/6, 0, −0.129745 | yes |
| Refine `00112233`: RI, ARI, NMI, geometric NMI, AMI, H(V|U) | L2, V4, §7 | 5/7, 4/11, 2/3, 0.707107, 0.533333, 1 bit | yes |
| Singletons: purity 1, H(V) = 3, I = 1, NMI, ARI, AMI | practice 3 | NMI 0.5, ARI 0, AMI 0 exactly (E[MI] = 1) | yes |
| Constant candidate vs `00001111` | L3 convention | NMI 0, AMI 0, ARI 0 | yes |
| Permutation invariance | §2, Program 2 | rename V → {9, 4}: ARI and NMI identical; joint random permutation of the eight IDs: ARI identical | yes |
| Chance level, 70 balanced assignments | §6, Program 3, L3 | overlaps {0:1, 1:16, 2:36, 3:16, 4:1}; mean ARI 0, mean AMI 0 (to 1e-12), mean NMI 0.114844286; enumerated mean MI equals hypergeometric E[MI] 0.1148442860 | yes |
| Chance level, 3/5 margins (56 assignments) | L3 preset | mean ARI 0, mean AMI 0; enumerated mean MI 0.1202235121 = hypergeometric E[MI] | yes |
| Independent-assignment pair agreement at k = 5 | §5 | 0.68 | yes |
| Cophenetic correlation for 0, 1, 3 | §11 | 0.8660254 = √3/2 | yes |
| 8n² bytes at n = 100 000 | §11 | 74.5058 GiB | yes |
| Gap-rule teaching table | §11 | picks k = 2 | yes |
| FM for D/E swap | §7 | 0.5 | yes |
| L5 defaults and probe ARI | §10, Program 5 | centers (0.5, 6.5) and (2.5, 8.5), both cost 18.5; labels 001111 / 000011; ARI −1/14 | yes |
| Practice 7 | §12 | centers (1, 9) cost 4; tripled (3, 27) cost 36 | yes |
| Iris CSV vs `load_iris` | data provenance | 150 rows, IDs 0–149 contiguous, all four features and species identical to `load_iris` (scikit-learn 1.9.1); file SHA256 `13c9…af78a` | yes |
| Iris raw4 chain from `load_iris`, KMeans(n_init=20, random_state=17) | §1, §8, V5 | k = 2: silhouette 0.681046, ARI 0.539922, AMI 0.653838 (NMI 0.656519), sizes 53/97, negatives 0; k = 3: 0.552819, 0.730238, 0.755119 (NMI 0.758176), sizes 50/62/38, negatives 0 | yes |
| V5 caption "k = 2 merges versicolor and virginica; k = 3 splits them" | V5 | k = 2 contingency [[50,0],[3,47],[0,50]]; k = 3 [[50,0,0],[0,48,2],[0,14,36]] | yes |
| Stored scaler mean/scale, PCA components and explained variances | `clustering-evaluation-data.js` | identical to `StandardScaler` and `PCA(2, svd_solver="full")` on standardized `load_iris` | yes |
| scaled4 vs pca2 k = 3 identity | §8 | ARI 1 between the two fitted label arrays; silhouettes 0.459948 vs 0.509168 | yes |
| Frozen scaled4 k = 3 rescoring | L4 | weights 1: 0.459948239 (2 negative); petal width ¼: 0.454404648 (5); petal width 4: 0.471241835 (7); all 4: 0.459948239 (2) | yes |
| Full-rank unwhitened PCA invariance; whitening = divide by √explained variance | §4, models `irisRepresentation` | both confirmed with scikit-learn | yes |

Claims the author verifiers do not check (the model verifier calls `centroidIndices`, `expectedRI`, `FM`, `nmiGeometric`, `homogeneity` and `vi`, so those are covered): the cophenetic √3/2 example, the 74.5 GiB figure, the gap-rule table outcome, the k = 5 independent-model probability, the V5 contingency claim, the CSV-versus-`load_iris` identity, and the stored scaler/PCA constants. All verified above.

### A3. Displayed programs versus manuscript output blocks

The six `code` strings in `clustering-evaluation-examples.js` are character-for-character the manuscript's blocks (Programs 1–6), and each `expected` string equals the manuscript's output block. The native record's stdout for all six equals `expected`. My direct `load_iris` chain reproduces Program 4's raw4 lines to six decimals, and my exact split reproduces Program 5's lines. No printed line contains a disclaimer or cautionary sentence; every `print` is a labelled value.

### A4. JSX prose read for transcription errors, dropped qualifiers and reference models

- Reference models are stated where each score is introduced: ARI subtracts the fixed-margin permutation expectation E[S] = AB/M (§5); AMI subtracts E[MI] under the same fixed margins, hypergeometric per cell (§6); NMI is named arithmetic, "the current scikit-learn default", with the `min` normalizer contrast (§6); Program 4's `adjusted_mutual_info_score` default is arithmetic in 1.9.1, so the printed AMI matches the prose convention.
- Constant-labeling conventions (both constant → 1, one constant → 0) match scikit-learn 1.9.1 for NMI and AMI, and the ARI convention (identical partitions → 1, one constant vs nontrivial → 0) matches `adjusted_rand_score`.
- Numbers in prose and captions (0.681046/0.552819, 0.539922/0.730238, 25/49, 1/8, 2/3, 1/√2, 0.114844, −0.13, 0.454/0.471, −1/14, 4.67/1.38, 0.628523) all match the recomputation.
- One transcription defect: Program 1's `question` already begins "Before running:" and the `Program` wrapper prefixes a bold "Before running:", so the page renders "Before running: Before running: after C moves …" (finding m-1). The other five questions have no prefix.
- One dropped qualifier worth restoring cheaply: the manuscript's L4 instruction to "choose your own positive weight for petal width … then use a common multiplier … as the null control" is now only implied by the caption's last sentence; the preset buttons carry the two contrasts but the prose no longer tells the reader to enter a weight of their own (finding m-5).
- Minor cosmetic: L3's readout prints "H(U) = 1 bits" (plural for 1).

### A5. Model code read for behaviour the verifiers do not exercise

- `pairAgreement` degenerate branch: the only solutions of (A + B)/2 = AB/M are A = B ∈ {0, M}, so the branch fires exactly for both-constant and both-all-singletons; `A === B` there is always true. Correct.
- `informationMeasures` clamps a rounding-negative MI to 0 and handles both-constant/one-constant before dividing. AMI with a vanishing denominator uses 1/0 rather than scikit-learn's ±eps clamp; unreachable for the lab's eight-ID inputs except the constant cases already handled.
- `fixedMarginNull` enumerates distinct assignments of V's label multiset (70 for 4/4, 56 for 3/5); overlap r counts U's first group receiving V's smallest label, as the caption says.
- `exactLineCenters` orders tied optima by `centers.join(',').localeCompare(…, { numeric: true })`. Numeric collation compares digit runs, not real numbers: `"2.5,8.5"` sorts before `"2.25,9"`, and a leading minus sign is punctuation, so `[-3,27]` sorts after `[-1,20]`. A random search of 40 000 supported inputs (integer locations in [−40, 40], weights 1–5, k ≤ 3) found tied optima in about 0.25 % of cases, and in about half of those the kept tuple was not the numerically smaller one that Program 5's tuple comparison would keep; e.g. locations [−40, −35, −24, −13, −9, 7], weights [4, 3, 1, 2, 5, 2], k = 3 keeps centers (−36.125, −10.143, 7) where the lexicographically smaller tie is (−37.857, −11.875, 7), and the two tied optima give probe C different labels. The readout then says "the smaller ordered centers were kept", which is false for that input (finding m-2). Defaults, presets and every manuscript fixture are untied and unaffected.
- Prediction records in L1, L2, L4 and L5 key staleness on the draft only. After a comparison, changing the selected observation (L1), pair (L2, L5) or specimen (L4) re-labels the standing feedback with the new selection while the compared value belongs to the old one (L1 shows "s(D) is now −0.75" after C was compared and D selected). Finding m-3.

## Part B — learning-experience checklist (reviewer's heuristic run)

| # | Item | Finding with evidence |
| --- | --- | --- |
| 1 | Route | A first-pass route paragraph (`ce-route`) follows the intro and names sections 1–7, Programs 1 and 2, sections 8–10 with one of Programs 4/5, and section 12; section 11 and the three "Deeper branch" H3s (CH/DB/Dunn, E[MI] without enumeration, matching/VI) announce themselves as optional where they begin. Pass. |
| 2 | Cautions | One `Callout` in section 1 states the population/representation/comparison-rule condition and says the lesson will not repeat it. Across 118 `Prose` paragraphs the standard's hedge phrases occur zero times ("does not prove", "not a certificate", "not automatically", "alone does not"); "does not" appears 10 times, almost all as technical statements (e.g. "FM does not subtract a chance baseline"). Later limitations (rejection denominator, trivial stability, leakage) appear once each beside their mechanism. No displayed program prints a cautionary sentence (all 22 `print` labels are values). Pass. |
| 3 | Real question | The lesson opens with the raw-Iris two-versus-three disagreement, shows it in V5 with a contingency table, returns to it in section 8 prose and L4 ("Return to the opening question…"), and closes with the report paragraph and Program 6's frozen protocol on the same specimens. Real data: the licensed CSV is downloadable from sections 3 and 8, identical to `load_iris`, with provenance and license in the references. Pass. |
| 4 | Labs as investigations | **L1**: prediction recorded (select + "Apply and compare"), compared, stale on edit (screenshot shows a matched negative-sign prediction); free controls: six coordinates and six memberships plus selection; fixture contrasts confirmed independently (s(C) −0.75, mean 0.4594; doubling leaves every s unchanged — the null; all-zero and one-group states handled). **L2**: prediction on ARI direction or a chosen pair's status; free controls: any of eight candidate labels 0–7 and the reference; rename is the null (ARI, tiles unchanged), cross −1/6 and refine 4/11 are the contrasts, all recomputed here. **L3**: prediction on AMI sign or null-mean sign; free controls: sixteen binary labels; contrasts: overlap 2 gives NMI 0 with AMI −0.1297 while the null mean NMI is 0.1148 (screenshot); null: constant candidate reports a degenerate null, not zero. **L4**: prediction (rescore mode) on mean direction or a row's sign; free controls: four weights in [0.25, 4] and any specimen; contrast: petal-width ¼ → 0.4544 and 4 → 0.4712 while ARI 0.6201/AMI 0.6552 stay fixed (screenshot); null: common weight 4 leaves every bar in place. Compare mode has no prediction control, which the specification allows. **L5**: prediction on a pair under fit B or ARI direction; free controls: twelve multiplicities, k, six locations; contrast: default emphasis flips C and D with ARI −1/14 (screenshot); null: k = 1 gives ARI 1; the practice-7 state (1, 9 / cost 4 → 3, 27 / cost 36) is reachable by two buttons. All five pass the three requirements; residual: m-2 (tie ordering) and m-3 (feedback relabelling after a selection change). |
| 5 | Figures | V2 at 320 px: arcs, own/foreign distinction, location labels and the six bars with the 0.807 mean marker are all legible (fan-mobile). V3 at desktop: the two panels share axes and units; the ring panel's negative inner bars and 0.075 marker against the slice panel's 0.321 marker are perceptible, though the 32 compact bars are about 1.4 units tall (readable as a distribution, not per point, which the caption does not promise). V4 at 320 px: bar lengths 1 bit / 2 bits / 1 bit are exact and labelled. V5 has no screenshot but is two tables. V6 at 320 px: both bar sets and both means are readable; the accessible table's third column is clipped into a scroll region so the means are read from the titles above (minor, noted). L4 at desktop: 150 compact bars in three sorted bands with the dashed 0.471 mean and the two negative tails are visible. Baselines present: the six-point original mean in V6's table, one-center baseline in V7, the applied mean in every direction prompt. No caption apologizes for a figure. Pass, with the V6 clipping noted. |
| 6 | Connections | Pair board S = TP tiles (L2 caption); refinement ties H(U|V), homogeneity, completeness, arithmetic NMI and V-measure (V4, §7); PCA/standardization change silhouette but not memberships (§8 prose and L4, ARI 1 between scaled4 and pca2); common scaling links Program 1's invariance, L1's doubling, practice 7 and L5's tripling; the report protocol reuses the PCA lesson's train-only discipline. Canonical Zaki & Meira breadth is retained or routed with reasons in the design record. Pass. |
| 7 | Code | Program 1: one two-line guard, mechanism otherwise; Program 2: no guards; Program 3: none; Program 4: none; Program 5: one comment; Program 6: the eligibility rule is the mechanism being taught. Displayed teaching code shows the mechanism in most lines; validation is separate and minimal. Pass. |
| 8 | Practice | Eight tasks change numbers and context (foreign-group minimum with a misleading nearest point, crossed labels, singletons, 90/10 weighting, rejection, bootstrap alignment, changed probe locations with scaling, museum capstone). Exact check values present and verified: s = 0.5; RI 3/7, ARI −1/6, NMI 0, AMI −0.129745; NMI 0.5, ARI 0, AMI 0; 0.68 vs 0.2; centers 1/9 cost 4 → 3/27 cost 36. Practice 7 is reproducible in L5 and is an independent variation. Pass. |
| 9 | Screenshots | Twelve captures opened. Informative states present: L1 after a matched prediction with the moved-C fixture; L2 after a matched "ARI lower" prediction on the crossed partition; L3 with the 70-assignment histogram and attainable-value table; L4 after a matched "mean higher" rescoring with petal-width weight 4; L5 default contrast (prediction unset); V3 at full desktop width; V7 desktop; V2, V4, V6, L2 and L5 at 320 px. Chance and resample captures show unset predictions; the browser record states their predictions were compared in the run, so the compared state is exercised but not pictured. Pass, with that note. |

## Ranked findings

No material findings. All recomputed values, program outputs, reference-model statements and visual contracts check out. The items below are minor and cosmetic; none degrades the teaching surface, and each fix is local.

| Rank | Severity | File · location | Finding | Least-intrusive fix |
| --- | --- | --- | --- | --- |
| m-1 | minor (visible text defect) | `src/learn/data/clustering-evaluation-examples.js` line 6, `silhouette.question`; rendered by `Program` in the topic JSX line 27 | The question string begins "Before running:" and the wrapper prefixes the same words, so Program 1 renders "Before running: Before running: after C moves…". The generator `scripts/verify-clustering-evaluation-examples.py` line 31 carries the same string. | Delete the leading "Before running: " from that one question in the generator and regenerated examples file, and capitalize "After". No numerical change; rerun only the examples verifier (its `sourceHash` will change). |
| m-2 | minor (tie-break claim can be false) | `src/learn/data/clustering-evaluation-models.js` line 191 (`alternatives.sort` comparator); readout text in `ClusteringEvaluationLabs.jsx` line 321 | String numeric collation does not order real numbers (decimals, negatives), so on the rare tied-optimum inputs the lab may keep a tuple that is not the lexicographically smaller one while saying it was; the two tied optima can also assign a probe differently. Defaults, presets and all manuscript fixtures are untied. | Replace the comparator's second key with an element-wise numeric comparison of `centers` (first differing element decides), and compare costs with the same 1e-9 tolerance used for `ties`. Add the tied example above to the model verifier's split checks. |
| m-3 | minor (feedback relabelling) | `ClusteringEvaluationLabs.jsx`: L1 line 92 (`key`), L2 line 152, L4 line 239, L5 line 297 | Staleness is keyed on the draft only, so changing the selected observation/pair/specimen after a comparison re-labels the standing feedback with the new selection while the compared value belongs to the old one. | Include the selection (`selected`, `pair`, `specimen`) in each lab's `key` so the feedback goes stale and asks for a new prediction; alternatively store the selection label in `committed` and render it from there. |
| m-4 | minor (evidence bookkeeping) | `docs/teaching/evidence/clustering-evaluation-models.json` `sourceHashes` | Hashes for `ClusteringEvaluationLabs.jsx`, `ClusteringEvaluationFigures.jsx` and the CSS predate the screenshot-defect repair; the verifier does not import them, so the result stands, but the record misstates the version set. | Rerun `node scripts/verify-clustering-evaluation-models.mjs` once after m-2 (which changes `models.js` anyway); the record then names the current versions. |
| m-5 | minor (dropped instruction) | Topic JSX, section 8 prose after `CeIrisLab` (line 182) or the L4 `question` | The manuscript told the reader to enter a petal-width weight of their own and use a common multiplier as the null control; the page now only offers preset buttons and a caption sentence. | Add one sentence to the L4 `question` or the following paragraph: "Type a petal-width weight of your own before using the presets; the common-weight button is the control that should leave every bar in place." |
| — | none (cosmetic) | `ClusteringEvaluationLabs.jsx` line 213; V6 table at 320 px; V7 figcaption wrap | "H(U) = 1 bits"; the V6 accessible table's mean column is only reachable by scrolling on phones (means are also in the bar titles); "90 / 30 / 30" wraps to a second line on desktop. | Optional: pluralize by value; add `white-space: nowrap` to the last V6 column or move the three means before the label column; use a non-breaking "90/30/30". |

## Closure checks for the fixes

- m-1: examples verifier passes and the rendered Program 1 heading text contains exactly one "Before running:" (browser case 1 already reads the program blocks).
- m-2: model verifier's split section asserts that for the tied example the kept centers equal the tuple Python's `(cost, centers)` comparison keeps, and that all existing untied fixtures are unchanged.
- m-3: browser lab cases add one step per affected lab: change the selection after a comparison and assert the stale message appears.
- m-4/m-5: documentation-only apart from the rerun; no rebuild needed for m-4, a rebuild plus the existing browser case for m-5.

## Statement

Correctness: the implementation preserves the manuscript's explanations, formulas, reference models and numerical claims; every recomputed value agrees; the six displayed programs and their output blocks are the manuscript's and reproduce natively. Learning experience: the nine checklist items pass on the evidence above with the minor notes recorded. The topic is ready for integration once the integrating agent handles the shared blueprint index and reruns the browser suite against the shared `dist`, as the design record already requires; the findings above can be closed in that same pass.
