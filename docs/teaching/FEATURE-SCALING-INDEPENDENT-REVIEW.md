# Feature Scaling, Encoding & Imputation — independent phase-two review

Reviewed 14 September 2026 against the working tree plus the uncommitted feature-preparation implementation. The reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or `docs/teaching/drafts/` was edited; this review document is the only write outside `scratch/feature-scaling-review/`. No verifier was re-run (see below for why), and no state-changing git command was executed.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/feature-scaling-review/`, written from the manuscript's own arithmetic and importing neither `scaling-models.js` nor any `verify-scaling-*` script nor `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1, SciPy 1.18.1). **The whole penguin preparation was refitted from the served CSV by hand**: the stratified split, the four training medians, the post-imputation means and population standard deviations, the min/max and the linear-convention quartiles, the fitted category vocabulary, all 86 × 7 transformed held-out rows, a hand-written 5-nearest-neighbour vote (not `KNeighborsClassifier`) for all four preparations, the majority baseline and all four confusion matrices. Also recomputed independently: the five-value scaler fixture and the later 150, the Q/A/B distances with the 4,400 g contrast and the doubled-divisor null, the one-hot geometry with and without a dropped reference, the three overlap distances and every donor contrast/null, the cross-fitted encoding with its own-target null and both practice-7 cells, the rank and log coordinates, the Yeo–Johnson branches at λ = 0, 1 and 2 on both signs, Rubin pooling for both examples, and every practice answer 1–9. **Both displayed programs were extracted from `scaling-examples.js` with `node`, written to the filenames the lesson names, and executed beside a copy of the served CSV.** The JS model layer was probed with `node` and compared value-by-value against the Python. `sha256sum` was run on every reviewed file, and the served CSV was fetched from the upstream project URL and compared byte for byte. |
| **Read in full** | `lesson.md` (525 lines), `visual-specifications.md` (80 lines), `data-provenance.md`, `design.md` including its phase-two append, the published `.jsx` body (287 lines), `ScalingShared.jsx`, `ScalingFigures.jsx`, `ScalingLabs.jsx`, `scaling-labs.css`, `scaling-models.js`, `scaling-examples.js`, the generated `scaling-data.js` header/tail plus a full programmatic dump, the blueprint and the `blueprints/index.js` diff, `RunnableExample.jsx`, `LessonElements.jsx`, the four evidence JSONs, `scripts/verify-scaling-examples.py` in full and `verify-scaling-models.mjs` in part, the feature-scaling entries in `lesson-delivery-progress.json` and `curriculum-inventory.json`, and the relevant sections of `LESSON-AUTHORING-HANDOFF.md` and `LESSON-TEACHING-STANDARD.md`. |
| **Looked at** | Nine figure captures and eighteen isolated figure SVGs taken by the reviewer at 1366 px, plus the I4 donor graph after a commit. Findings **S1**, **S2**, **S4** and **S7** come from that pass. The implementer's 29 committed screenshots were **not** opened; the reviewer re-captured instead, so the observations below describe the live page rather than the record. |
| **Operated** | The live page was driven with Playwright (msedge channel) against the running `dist-scaling` preview at `http://127.0.0.1:4183`, at 1366, 390 and 320 px. All four investigations were exercised: unset-prediction state, commit, edit-and-retire, every checked contrast and every checked null in the specification, the no-overlap fallback, the absent-target-column refusal, the zero-smoothing prior fallback, the empty-fold refusal, all three category states, keyboard-only prediction and commit, out-of-range and zero-divisor entry, and the explore path. Rendered font sizes, table clipping and document width were measured in the DOM at three widths. Findings **B1**, **S3**, **S5** and observations **O2**, **O3**, **O4**, **O5**, **O12** were confirmed there, not merely inferred from source. |
| **Reused, not rerun** | `evidence/scaling-models.json` (27 grouped checks), `scaling-native.json` (2 programs, 24 oracles), `scaling-data.json` and `scaling-browser.json` (14 cases, 29 screenshots). Every source hash they record equals the hash below for the same file, so their recorded results bind to the bytes reviewed here. **The verifiers were deliberately not re-run**: all four rewrite their evidence JSON unconditionally, `verify-scaling-examples.py --write` rewrites `src/learn/data/scaling-examples.js`, and `verify-scaling-data.py` rewrites `src/learn/data/scaling-data.js`, so running any of them would have violated the review's no-write boundary (observation **O1**). Their results were instead confirmed by the independent recomputation above, which agreed on every number. |
| **Delegated** | Every external URL in the Sources block and in the provenance file was fetched and checked against the sentence that cites it — ten in all, including the feature-hashing PDF (title, authors and Definition 1 extracted from the PDF itself, not trusted to a summariser), and the upstream `penguins.csv`, whose bytes were downloaded and hashed. Everything resolved and supported its claim; see A6. |
| **Not done** | No screen-reader session. No other browser engine. No 200 % zoom pass. The implementer's 29 committed screenshots were not opened. `verify-scaling-browser.cjs` was not run. Version-independence of the real-data result was not tested on any library version other than the pinned one. The neighbour classifier was re-implemented by hand but not cross-checked against a second library. |

## Preview-build currency (checked before any page observation was trusted)

The brief warns that a stale preview of the old `dist` was found squatting on port 4183. The following was established before any live observation was used:

- `curl http://127.0.0.1:4183/` returns a document **byte-identical to `dist-scaling/index.html`** (`diff` clean), not to `dist/index.html`.
- The served page renders the header "Feature Scaling, Encoding & Imputation", "20 of 39 topics on this route", nine `.sc-figure`, four `.sc-investigation`, nine `.sc-practice` and two `.python-example` elements.
- Strings that exist only in the current sources are present in the served page and in `dist-scaling/assets/feature-scaling-encoding-imputation-GjbHVTnr.js`: "executed verbatim against the same served CSV", "edited copy — not a measured specimen", "no edge leaves row", "Figure 4c", "Figure 5b", "Readiness check".
- `dist-scaling/assets/feature-scaling-encoding-imputation-jM-vzt-8.css` contains `display:contents` (the Figure 6 bar-grid repair) and the `font:11px var(--font-mono,monospace)` rule, so the CSS in the build is the CSS on disk.
- **One caveat.** `dist-scaling/index.html` was built at 18:23:52; `src/learn/data/scaling-data.js` and `public/learn-assets/feature-scaling/penguins.csv` were written at 18:28:26 by `verify-scaling-data.py`, and `src/learn/data/curriculum/blueprints/index.js` at 18:30:07. The data rewrite is idempotent — every fitted statistic, row and count embedded in the built chunk equals the current module byte for byte, and the CSV hash is unchanged — so the build is current for the lesson body. **It is not current for the blueprint**: no chunk in `dist-scaling` references `feature-scaling-encoding-imputation.js` under `blueprints/`, so the registration was never in the build the browser evidence was taken against. See **B2**.

## Source versions reviewed (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` | `2fea8f9e8152f764e89f94e464ed7d9132616864e08c1ced3efd0d831511864d` |
| `src/learn/data/scaling-models.js` | `2016b0426dad0a3c76da8a04dd88ccee56a78c9f75bf81b029007552d56c9d2c` |
| `src/learn/data/scaling-data.js` | `1f0bc9e1ae81883aa0bccbd0ee3daa1140ce9f87b5847203f1da82efdbf93c04` |
| `src/learn/data/scaling-examples.js` | `0ef4de579fe7b3a38a15d10dbc8e6f8265081d6f13a82894cd5d91140aefd2ef` |
| `src/learn/components/lesson-labs/ScalingShared.jsx` | `62889bcf4de73be3adea32d55286b88ce73e54a9481f85ab97b82e7f2bf41448` |
| `src/learn/components/lesson-labs/ScalingLabs.jsx` | `7d3dc5f21495294ba730d485e8e4d72eb512990c0aaec310cede1c6c4f9e0516` |
| `src/learn/components/lesson-labs/ScalingFigures.jsx` | `caa34de92c6a0d4340f38580c2e037faaba42506adbbb10fb4b415e6aa2db162` |
| `src/learn/components/lesson-labs/scaling-labs.css` | `757920fad31c092e8f599df8355f4990917177b4e60df55aedd20e2c22354b2b` |
| `src/learn/data/curriculum/blueprints/feature-scaling-encoding-imputation.js` | `8b87cdc8f5f49cdd7036c63f7f56016e97dabfd2006413e77151010b061310b6` |
| `public/learn-assets/feature-scaling/penguins.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
| `docs/teaching/drafts/.../penguins.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
| `docs/teaching/drafts/.../lesson.md` | `389fc4e93f958e0ca819a73ae36bb5e5fb09aa11ce73072572329b2eb07182da` |
| `docs/teaching/drafts/.../visual-specifications.md` | `3729dfb734d3b4a929d7600f5c8db237b2b0e8543e665eb2dea1f9187cc9e91c` |
| `docs/teaching/drafts/.../calculated-inputs.json` | `f89cf22494ef0a47f0d3539cb29448baba9e7cffabc38e5fd34393a19f16e70f` |
| `docs/teaching/drafts/.../data-provenance.md` | `233f38872a40b907011c6010010a876151d825a43174cbd375593f83d7d667fd` |
| `docs/teaching/drafts/.../author-calculations.py` | `a81c1a030838a87a4d6993ca842a753e5bc21a79066e54c0741797fdb2317a1b` |
| `docs/teaching/drafts/.../design.md` | `cde0c09eec8adf8f1ea941acddb79fcdf6c03ecd46d5cc0c60cba2bfbe83a150` |
| `scripts/verify-scaling-models.mjs` | `24c9f425d50c935f48148814b1f3a2c5f7ff345c238f5287495d461d80b9d743` |
| `scripts/verify-scaling-examples.py` | `bf074aa2a638213f406e5b4243cc162dc131ac7bb9c7adfbcb8829d4c83cc045` |
| `scripts/verify-scaling-data.py` | `fc17c2c19aea0e36457b9079c6818117c2ea91cd466d46b04a79637758fc55d9` |
| `scripts/verify-scaling-browser.cjs` | `5ee18d31a316b2f4a505c37ed4b0ee479e9ac56b62d411c0e4e4915f904f49c5` |

A hash proves identity, not correctness. Every hash in the design record's own table matches the value above. All four evidence files record source hashes, and every one equals the value above for the same file, so their recorded results bind to exactly the bytes reviewed here. The ledger's content checkpoint still binds `design.md` at `9ad40bc1…`, which is stale after the phase-two append (**B2**); all six other packet files still match their recorded hashes exactly.

**The served CSV is byte-identical to the packet copy, and both are byte-identical to the file the provenance URL serves today.** `https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv` returns 15,241 bytes hashing to `f204db2c…`; so do `public/learn-assets/feature-scaling/penguins.csv`, the packet copy, and the bytes actually served over HTTP by the preview. **The cross-validation lesson serves its own copy at `public/learn-assets/cross-validation/penguins.csv`; it is byte-identical to this one, so neither topic has edited the other's file.**

## Part A — correctness

### A1. Manuscript sections, claims, cautions and tasks preserved?

All eleven manuscript sections are present in the published body, in order and under the manuscript's own titles (`.jsx` L17–29), with the declared first-pass route (L47), both deeper-branch section titles carrying "Deeper branch", every manuscript table, both programs in the manuscript's order, practices 1–9 with separate closed hint and solution disclosures, the readiness paragraph and the full reference list. `headingId` (L30) derives anchors with the same slug rule `H2` uses, and all eleven route links resolve.

A mechanical sentence-by-sentence comparison of the manuscript prose against the rendered page text (`textContent`, so closed `<details>` are included; math and code stripped) found **370 manuscript prose sentences, of which every one is present verbatim except those altered by a declared departure, a MathML/KaTeX rendering difference, or a list-to-paragraph reflow.** The genuinely altered ones are:

| Manuscript | Page | Judgement |
| --- | --- | --- |
| Intro's "We will use real observations, inspect individual transformations, and keep a small set of rows aside…" and the first-pass paragraph | Rewritten into `LessonIntro` and the `sc-route` paragraph, which name the three core investigations, the program and the practice split explicitly | Sound; more specific than the manuscript |
| §3 "One-hot encoding assigns one coordinate to each known category:" + the 3 × 3 table | Sentence merged with the next; table rendered live inside `CategoryFigure` | Declared departure; every row present and computed |
| §8 "Here is an exact six-row example, with smoothing 2:" + the six-row table + the fold-0/fold-1 arithmetic paragraph | Rendered live inside `EncodingFigure`, which reproduces the paragraph with the same numbers | Declared departure; every row present and computed |
| §5 the Fit/Transform bullet pair | One paragraph | Cosmetic |
| §6 "The following complete experiment was executed during authoring through an equivalent calculation script" | "The block below is executed verbatim against the same served CSV before this page is published, and the output shown underneath it is that run's own output, not a transcription" | **Sound, and verified true.** See A3 |
| §6 "standard scaling misclassified two rows here, whereas the raw model misclassified nineteen" | Same sentence with the counts computed live from `comparison` ("2", "19") | Sound |
| §8 "This bounded calculation was executed during authoring." | Dropped, not replaced | Observation **O7** |
| Practice 1 solution | Gains "Investigation 1 accepts these exact coordinates and divisors if you want to watch the contributions." | Finding **B1** — the sentence is false |
| Practice 5 solution | Gains "Both edits are available in Investigation 2." | **True**; I performed both edits on the live page |
| Sources: "Our attached CSV preserves the public file; the accompanying provenance records its exact hash and use." | Replaced by a sentence naming the byte count, the SHA-256, the row count and the `NA` convention, all computed from the module | An improvement |

**No invented claim was found in the lesson prose apart from B1.** Every figure and lab caption traces to a computed value or to a specification line.

The five characteristic honesty caveats named in the brief were checked sentence by sentence and are all present, each with a genuine single home:

- **A transform fitted on all the rows leaks.** §5 in full (`.jsx` L145–150): "A new row should not redefine the ruler"; "Split rows before learning preprocessing statistics"; "An estimate of future performance is compromised when the fitting procedure gets information it would not have at prediction time"; and, crucially, "Looking at held-out feature distributions to choose a transformation can also make an experiment adaptive, even without reading labels." Reinforced as mechanism rather than repeated warning in Figure 4's crossed backward arrow, in I3's standing note ("…were fitted on the 258 training rows and never change here, however you edit the record"), in the closing provenance paragraph (L283), and in the model layer itself (A5).
- **An imputed value is a fabricated value, not a measurement.** §4 L121–122 ("the two missing measurements have not been discovered"; "it does not certify that an estimated value was physically measured"), the checkpoint's closing sentence (L173), practice 6's "It is not a recovered observation", Figure 1's second stage ("inserted estimate", drawn with a different outline and said so in the caption), I2's closing caption ("An imputed cell is a supplied model input, not a discovered measurement"), and I3's `is-imputed` coordinate marking.
- **One-hot columns are linearly dependent with an intercept.** §3 L107: "If all three columns are kept, their sum is the intercept column, so its coefficients are not uniquely identified. Predictions can still be fitted with a suitable numerical least-squares solver." Both halves survive — the second is the repair the design record's disposition table claims, and it is on the page.
- **A distance changes meaning when a unit changes.** §2 in full, and I1's standing note: "Changing a divisor expresses a choice about which differences should count; nothing moves in the physical world." Also I1's two-panel design, which draws distance circles only on the dimensionless plane and says on the physical plot that "a round circle here would assert a comparison the units do not support".
- **Any penguin result is one dataset under one protocol.** §6 L153 ("does not establish performance on every future population or collection protocol"), L164 ("One extra correct row does not establish that min–max or robust scaling is generally superior… selecting a procedure from these results would require a separate evaluation plan"), Figure 4c's paragraph, and the closing paragraph L283 ("None of them is a benchmark or a claim about any future dataset").

Also preserved and checked: "It **does not** turn a skewed distribution into a Gaussian distribution"; "the distribution of a feature and the distribution of a model's errors are different objects"; "Constant columns may be removed, but a value changing after training can also be a useful data-quality signal"; "'Always drop the first category' is not a universal preparation rule"; "This avoids an exception, but it does not teach the model how `sensor_C` behaves"; "An ordinal scale alone does not justify those gaps"; "Adding an indicator does not, by itself, solve MNAR"; "Observed data alone generally cannot distinguish MAR from every MNAR alternative"; "A 'Gaussian-looking' histogram can therefore hide an important loss of magnitude information"; "that expectation is not a guarantee for every pair under a fixed small hash table"; "this small arithmetic example is not an automatic validity certificate"; and "It does not repair a feature that already leaks the target, a split that puts repeated subjects on both sides, or a manually fitted transformer created before the split."

**The unflattering result is published as it came out.** Standard scaling — the default a reader expects to win — is beaten by one row by both min–max and robust, and the page says so three times (the table, Figure 4c's bars on a zero-origin axis, and the paragraph refusing to draw a winner). Nothing anywhere implies a recommendation.

### A2. Independent recomputation (reviewer's own scripts, no lesson code)

Every numeric claim in the manuscript, in the published body, in the figures, in the labs and in the generated data module was recomputed from first principles. **Nothing disagreed. Not one value.**

| Claim | Independent result | Agrees? |
| --- | --- | --- |
| Served CSV: 344 rows, 15,241 bytes, missing counts 2/2/2/2 numeric and 11 sex | identical; header and `NA` convention as the provenance states | Yes |
| **Split: 258 training / 86 held out**, first held-out source row 309 | `train_test_split(test_size=0.25, random_state=20, stratify=species)` → 258/86, first index 309 | Yes |
| **The whole 86-row `heldOutOrder`** | element-for-element identical | Yes |
| The 344-row `rows` table (source index, species code, four measurements with `null` for absent, sex code, split flag) | regenerated from the CSV; **zero mismatches across all 344 × 8 cells** | Yes |
| **Fitted medians `[45.0, 17.3, 197.0, 4000.0]`** | `[45.0, 17.3, 197.0, 4000.0]`, by my own linear-percentile median on the training rows only | Yes |
| **Fitted means `[43.93062015503876, 17.09922480620155, 200.65891472868216, 4190.988372093023]`** | identical to the last digit | Yes |
| **Fitted scales `[5.400031618043809, 1.9387119564900213, 14.230655916521705, 806.6875829303058]`** | identical to the last digit, with divisor *n* not *n − 1* | Yes |
| Min–max centre `[33.1, 13.1, 172, 2700]`, range `[26.5, 8.4, 59, 3350]`, max `[59.6, 21.5, 231, 6050]` | identical | Yes |
| Robust centre `[45, 17.3, 197, 4000]`, IQR `[9.05, 3.075, 23, 1200]`, quartiles as stored | identical | Yes |
| Fitted vocabulary `['female','male','not_recorded']` in that order | identical (sorted, as `OneHotEncoder` sorts) | Yes |
| **All 86 × 7 transformed held-out rows** | regenerated from my own fit; **max absolute difference 0.000e+00 across all 602 values** | Yes |
| **Source row 309 → `[1.3091367505, 0.8772706993, 0.1645100047, −0.1127925780, 0, 1, 0]`** | `1.3091367504848355, 0.877270699293386, 0.16451000467236732, −0.11279257796742845` and `[0,1,0]` | Yes, every digit |
| **Correct counts 38 / 67 / 84 / 85 / 85 out of 86** | my own hand-written 5-NN vote: majority 38, raw 67, standard 84, minmax 85, robust 85 | Yes |
| All four confusion matrices | `raw [[32,1,5],[9,7,1],[3,0,28]]`, `standard [[37,1,0],[0,17,0],[0,1,30]]`, `minmax` and `robust` both `[[38,0,0],[0,17,0],[1,0,30]]` | Yes |
| `heldOutTruth` species codes | identical | Yes |
| **Checkpoint / practice 6: `(4000 − 4190.988372093023) ÷ 806.6875829303058 ≈ −0.2368`** | `−0.236756305829395` | Yes |
| **Scale fixture**: mean 22, sd 39.01281840626232; rows −0.5383/0/−1, −0.5127/0.0101/−0.5, −0.4870/0.0202/0, −0.4614/0.0303/0.5, 1.9993/1/48.5; later 150 → 3.2810/1.5051/73.5 | every value reproduced to 13 significant figures | Yes |
| **I1 raw totals 10,001 and 10 → B; scaled 2 and 9.0001 → A; A mass 4,400 g → 17 → B; both divisors doubled → 0.5 and 2.250025, ranking still A** | identical, and confirmed on the live page | Yes |
| **One-hot geometry**: all three pairs √2; after dropping red, red–green 1, red–blue 1, green–blue √2, unknown–red 0, unknown–green 1 | identical | Yes |
| **Donor distances 7.5, 3, 12; k = 2 estimate 200** | `1.5 × (1 + 4) = 7.5`, `3 × 1 = 3`, `3 × 4 = 12`; D2 and D1 → 200 | Yes |
| Donor contrasts: remove D2.c → 300; D1.c 100 → 140 → 220; D3.c 500 → 900 → still 200; empty overlap → labelled fallback mean 300; whole target column absent → "cannot estimate" | every one reproduced in the model layer **and on the live page** | Yes |
| **Cross-fitted encoding `[5/9, 7/9, 2/9, 4/9, 2/9, 7/9]`; row 0's target 1 → 0 gives `[5/9, 2/9, 2/9, 2/9, 2/9, 5/9]`** | identical; priors 1/3 and 2/3 | Yes |
| **Practice 7: row 3's target 0 → 1 makes row 0 equal 7/9 while row 3 stays 4/9** | identical, and reproduced on the live page in I4 | Yes |
| Rank coordinates `[0, .25, .5, .75, 1]`; logs `0, 0.693147, 1.098612, 1.386294, 4.605170`; 100 → 1,000 leaves the rank at 1 and moves the log to 6.907755 | identical | Yes |
| **Yeo–Johnson**: λ = 1 gives 3 and −3; λ = 0 gives `log1p(3) = 1.386294` and the negative quadratic branch −7.5; λ = 2 gives 7.5 and `−log(4) = −1.386294`, a logarithm and not a square root | identical, and each branch label matches the formula the page prints | Yes |
| **Rubin pooling**: `[9,10,11]` with `U = 4` → θ̄ 10, B 1, correction 4/3, T 16/3, SE 2.309401 | identical | Yes |
| **Practice 9**: `[8,10,10,12]` with `U = 1` → B 8/3, T 13/3 ≈ 4.3333, SE 2.081666 | identical | Yes |
| Practice 1: raw 3,604 vs 125 → B; divisors `[1,30]` → 8 vs 25.111… → A | identical | Yes |
| Practice 2: mean 4, sd √(8/3), new 8 → √6 ≈ 2.4495, min–max 1.5 | identical | Yes |
| Practice 3: `[2,1,2]` and `[6,3,6]` both → `[2/3, 1/3, 2/3]`, norms 3 and 9 | identical | Yes |
| Practice 4: unknown-to-small 1, small-to-medium √2 | identical | Yes |
| Signed hashing: `apple:3, pear:1, banana:2` → `[2,2]`; `apple:2, banana:2` → `[2,2]` | identical | Yes |
| Circular coordinates: 1° = 0.017453 rad, 359° = 6.265732 rad, both `cos` 0.999848, `sin` ±0.017452, separation 0.034905 | identical | Yes |

**Could not verify numerically:** the reading-time estimates; the MCAR/MAR/MNAR definitions and the identifiability statement of §4, which are definitional rather than numeric; the sparse-storage and tree-invariance statements of §2, which are qualitative; and version-independence of the real result — every number is reproduced in the pinned environment, which is also the environment the author used, and no other library version was tested. The page states the pinned versions and the design record states the limit.

### A3. Displayed programs versus their published output

Both programs were extracted from `scaling-examples.js` with `node`, written to `penguins_prepare.py` and `cross_fit.py` beside a copy of the served CSV, and executed with the pinned interpreter. **Both reproduce their stored `expected` block exactly**, line for line (the only diff is CRLF from the shell redirect).

A three-way comparison confirms the manuscript's fenced block, the `code` string in `scaling-examples.js`, and the copy embedded in `scripts/verify-scaling-examples.py` are **byte-identical for both programs**.

The §6 departure — "The block below is executed verbatim against the same served CSV before this page is published, and the output shown underneath it is that run's own output, not a transcription" — is therefore **true and stronger than the manuscript's claim**, subject to one condition the reader cannot see: it holds only if `verify-scaling-examples.py` is actually run before publication. The verifier executes its own copy of the code inside `contextlib.chdir(ASSETS)` with `__file__` set to the served asset directory, so `Path(__file__).with_name("penguins.csv")` resolves to exactly the file a learner downloads; without `--write` it asserts that both the recorded `code` and the recorded `expected` equal a fresh run. This is a genuine gate, and the design record's reason for the departure is accurate. It is the one departure this class of lesson usually gets wrong in the other direction, and here it was strengthened rather than softened.

Neither program prints an explanatory or cautionary sentence. Displayed code is 61 and 18 lines. The `pip install` pin is a separate `bash` block, as the manuscript specifies.

### A4. Model and lab logic read for defects

- `scaling-models.js` keeps the discipline its header claims. **Every fitting function returns an explicit fitted object and every applying function takes one**: `fitScaler`/`applyScaler`, `fitMedianImputer`/`applyMedianImputer`, `fitOneHot`/`encodeCategory`, `fitPreparation`/`transformRecord`. `transformRecord` reads `fitted` and never writes to it; I confirmed by diffing the fitted object before and after transforming all 86 rows.
- Validation refuses rather than substitutes: a zero or negative divisor (`positive`, L29–32), an empty training column (L77), a query whose target cell is *present* (L267), a non-integer or sub-1 neighbour count (L268), a missing category with no declared fill (L176), a non-binary target (L363), fewer than two non-empty folds (L366), a fold with no donors outside it (L368), and a negative variance (L458). `knnImpute` reports `no-target-column` with a `null` estimate rather than inventing a zero (L282), and `overlapDistance` returns `squared: null` for an empty overlap rather than 0 (L255).
- `safeScale` (L58) reproduces sklearn's `_handle_zeros_in_scale` and the comment states the teaching consequence correctly.
- `percentile` (L61–69) implements the linear convention the manuscript declares; I checked it against my own implementation on both the five-value fixture and all four penguin columns.
- `useInvestigationWithHistory` (`ScalingLabs.jsx` L650–682) is the shared contract. `edit` retires the standing result, promotes it to `previous` and clears the choice; `advance()` promotes `draft` → `active` at the moment of commit. **This is the repair the design record lists as its ninth finding, and it is real** — see A7.
- Defects found by reading and then confirmed on the live page: Figure 4's misplaced forbidden label and orphaned answers rail (**S1**); I4's missing prior edges (**S2**); the explore button left enabled after a graded check in all four investigations (**S3**); `.sc-point-id` defeated by the stylesheet's `font` shorthand (**S4**). Dead exports are **O10**.

### A5. The leakage claim, specifically

The implementer asserts that the model layer enforces the boundary: that a frozen training-only fit reproduces all 86 recorded held-out rows, and that a fit over all 344 rows gives a different centre and a different median. **Both halves are true, and I verified each independently.**

- `scripts/verify-scaling-models.mjs` L226–241 contains exactly those two assertions, including `assert(Math.abs(leaked.scalers[3].center − preparation.scalers[3].center) > 1)` and `assert.notDeepEqual(leaked.imputer.medians, preparation.imputer.medians)`.
- My own recomputation, with no lesson code:

  | | training-only (258 rows) | all 344 rows |
  | --- | --- | --- |
  | body-mass median | **4000.0** | **4050.0** |
  | body-mass mean | **4190.988372093023** | **4200.872093023256** |
  | body-mass sd | 806.6875829303058 | 798.5333364699052 |
  | bill-length median | 45.0 | 44.45 |
  | bill-length mean | 43.93062015503876 | 43.925000000000004 |

  The mass centre moves by 9.884 (the assertion's threshold is 1) and two of the four medians move. The contrast is real, not a tautology.
- The frozen fit reproduces all 86 held-out rows to 0.000e+00, which I verified against my own fit rather than against the module.

**The page never quietly fits on everything, anywhere.** Every `fit*` call reachable from the UI was enumerated:

| Call site | Rows it fits on |
| --- | --- |
| `ScalingFigures.jsx` L217–222 `fitPreparation` | `rows.filter(row => row[7] === 0)` — the 258 training rows only. This one module-level object is the *only* preparation on the page, and both `PipelineFigure` and `PipelineLab` import it |
| `ScalingFigures.jsx` L102 `scalerComparison([1,2,3,4,100], [150])` | the constructed training column only; 150 is passed as a later value and is never in the fit |
| `ScalingFigures.jsx` L144 `fitOneHot(['red','green','blue'])` | the constructed vocabulary |
| `ScalingFigures.jsx` L442 / `ScalingLabs.jsx` L507 `crossFitEncoding` | donors from the other fold only, prior included |
| `ScalingLabs.jsx` L193 `knnImpute` | donors only; the query's target cell must be absent or the call throws |

**Inside the investigations, nothing refits.** I3 lets the learner change the record, clear a measurement, and enter a category the fit never saw; I confirmed on the live page that the fitted median, centre, scale and vocabulary shown in the step table are unchanged across all of those edits, and that clearing the body mass changes exactly one of the seven coordinates (`−0.1127925780` → `−0.2367563058`) while the other six are bit-identical. That is the specification's stated null, and it holds.

### A6. Links, references and provenance

Every external URL was fetched and checked against the sentence that cites it. **All ten resolve and all ten support their claim.**

| URL | Checked against | Result |
| --- | --- | --- |
| `scikit-learn.org/stable/modules/preprocessing.html` | the Yeo–Johnson four-branch definition and `PowerTransformer` behaviour | The published definition is `[(x+1)^λ−1]/λ`, `ln(x+1)`, `−[(−x+1)^(2−λ)−1]/(2−λ)`, `−ln(−x+1)` — **identical to the lesson's, including the `2 − λ` exponent and the λ = 2 logarithm.** λ by maximum likelihood and `standardize=True` by default both confirmed |
| `…/preprocessing.html#non-linear-transformation` | anchor exists | Yes, section 8.3.2 |
| `scikit-learn.org/stable/modules/impute.html` | simple/iterative/neighbour/indicator behaviour and empty columns | Confirmed, including "the training set average for that feature is used" — the fallback the lesson names |
| `…/impute.html#nearest-neighbors-imputation` | the feature-by-feature donor rule | Confirmed: "Each missing feature is imputed using values from `n_neighbors` nearest neighbors **that have a value for the feature**" |
| `…/nan_euclidean_distances` (reached from the imputation guide) | the `m/q` weight | The published formula is `weight = Total # of coordinates / # of present coordinates` with the worked example `√((4/2)((3−1)² + (6−5)²))` — **exactly the lesson's rule** |
| `…/generated/sklearn.preprocessing.TargetEncoder.html` | the 1.9 API claims | Confirmed on the 1.9.1 page: unseen categories → `target_mean_`; "`fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a cross fitting scheme is used"; "Changed in version 1.9: Cross-validation generators and iterables can also be passed as `cv`"; **both `shuffle` and `random_state` deprecated since 1.9**. The lesson's sentence is accurate and current |
| `…/auto_examples/preprocessing/plot_target_encoder_cross_val.html` | "near-unique categories can overfit without cross-fitting" | Confirmed: a deliberately near-unique feature, 0.8585 train / 0.6338 test without CF against 0.8000 / 0.7928 with it |
| `…/auto_examples/preprocessing/plot_all_scaling.html` | "full-range and magnified views of actual housing data", robust vs quantile | Confirmed: California Housing, full and 99 %-trimmed views, explicit Robust-vs-Quantile discussion |
| `allisonhorst.github.io/palmerpenguins/` and `…/reference/penguins.html` | licence, attribution, 344 rows, units | Confirmed: "Data are available by CC-0 license…", Horst/Hill/Gorman, 344 observations, bill and flipper in millimetres, mass in grams |
| `stefvanbuuren.name/fimd/sec-MCAR.html` | MCAR/MAR/MNAR definitions | Confirmed verbatim. The page is cited for developing the assumptions, which it does; the lesson's separate non-identifiability sentence stands on its own and is not attributed to a quotation |
| `amices.org/mice/reference/pool.html` | the pooling rule and small-sample treatment | Confirmed: `ubar + (1 + 1/m) * b`, the Barnard–Rubin adjustment, and "A common error is to reverse steps 2 and 3, i.e., to pool the multiply-imputed data instead of the estimates" — the exact error Figure 6's crossed shortcut draws |
| `arxiv.org/pdf/0902.2206` | title, authors, the signed map | Text extracted from the PDF itself: *Feature Hashing for Large Scale Multitask Learning*, Weinberger, Dasgupta, Attenberg, Langford, Smola. Definition 1 gives `φ_i(x) = Σ_{j:h(j)=i} ξ(j)x_j`, and the paper says the signed sum "leads to an **unbiased** estimate" — which is precisely the lesson's "preserves inner products in expectation" and its caveat that expectation is not a per-pair guarantee |

**Licence and attribution are correct and complete.** The Sources bullet names CC0, all three creators, the collecting researchers, the byte count, the SHA-256 and the row count, and links the served file. The provenance document's derivation (download URL, date, hash, row count, missing counts, split rule) matches the tree exactly.

### A7. The late staleness repair, verified rather than trusted

The design record (L133) says three investigations "computed their displayed detail from `active` while grading from `draft`, and never promoted `draft` to `active`, so the detail tables showed the initial fixture after every commit", and that they now promote on commit. **I verified the repair on the live page rather than in the code, in every investigation, and it holds. I could not find any remaining case where a displayed number does not belong to the committed inputs.**

| Investigation | What I did | What was displayed |
| --- | --- | --- |
| **I1** | committed with the default divisors, then pressed "Declared: 1 mm, 100 g" and committed again, then changed A's mass to 4,400 g and committed again | the contribution table read "under divisors 1 and 1 … A 10001, B 10", then "under divisors 1 and 100 … A 2, B 9.0001", then "… A 17, B 9.0001". The bars, the readout, the scaled-plane aria text ("A sits at 1, 1 and B at 3, 0.01 … radii 1.4142 and 3") and the physical-plot aria text all moved with it |
| **I2** | committed the base table, then cleared D2's `c` and committed again | overlap table went from "D2 … yes (300)" to "D2 … **no**", selected donors from "D2 and D1 → 200" to "D1 and D3 → 300", and the readout's "supply 300 and 100" to "supply 100 and 500" |
| **I3** | committed row 309, then cleared the body mass and committed again, then switched the inspected coordinate, then cycled female / juvenile / absent | the step table's body-mass row went from "4100 g … −0.1127925780" to "**absent** … 4000 (training median) … −0.2367563058"; the other three rows did not move a digit; the category readout tracked `[1,0,0]`, `[0,0,0]`, `[0,0,1]` correctly with the right explanatory clause for each of the three states |
| **I4** | committed row 0, then flipped row 0's target and committed, then inspected row 1, then ran practice 7 | the six-row table and the donor graph both tracked every edit; the own-target null held (5/9 unchanged) and the row-1 contrast appeared (7/9 → 2/9) |

In every case an edit cleared the standing result, replaced it with "Inputs changed; record a new prediction", and left no result table, graph or readout on screen. **Before any commit, none of the four investigations shows a result**: measured on first load, I1 has no table and no SVG, I2 has only its editable source table, and I3 and I4 have neither. That is the specification's "Reveal the computed outcome and reasoning only after Apply", and it is honoured. The only remaining gap in the contract is that the *ungraded* path stays open after a graded one (**S3**).

I also confirmed that the two investigation-level explanatory clauses that are chosen by branch rather than computed — I2's three `mode` readouts and I3's three category-state clauses — are selected from the *computed outcome*, not from the mode the learner picked. This is the defect class that produced ICA's B1 and S2; it is not present here.

## Part B — learning-experience checklist (reviewer's heuristic run)

| # | Item | Finding |
| --- | --- | --- |
| 1 | **Route** | Present and declared (`.jsx` L47): sections 1–6, the three core investigations named individually, the program, practices 1–6; sections 7–9 marked as deeper branches in their own titles with practices 7–9 assigned to them. Both time bands given. The `readTime` metadata string adds "~95 min complete read", which the route paragraph does not support (**O11**) |
| 2 | **Cautions** | Each of the five characteristic caveats has one home and recurs only as mechanism (A1). No caution is restated as a warning box. Neither program prints a disclaimer. The pinned-version caveat is present in the prose and in the evidence limits |
| 3 | **Real question** | Opens on a single row whose three cells need three different treatments and returns in §6 with 344 real penguin observations, a downloadable CSV under a correctly stated CC0 licence with its hash on the page, a stratified split declared before the rule, and an outcome that went against the default method — kept, with the numbers, in three places |
| 4 | **Labs as investigations** | All four: prediction unset, retired on any edit, graded against the committed snapshot, with a real null and a real contrast each. **I2 is the best of the set** — it reaches four distinct states (neighbours, labelled column-mean fallback, "cannot estimate", and ineligible-donor) and refuses to invent a value in the one case where the manuscript says an implementation must choose a policy. I3's "edited copy — not a measured specimen" badge is correct and switches the moment a cell differs from the file. I4's own-target null is the specification's own checked case and it works. Defects: **S2** (the graph draws two of three prior edges) and **S3** (the ungraded button stays live) |
| 5 | **Figures** | Nine figures against six contracts plus three declared additions. Figures 2, 4b, 4c, 5, 5b and 6 are correct and legible at desktop, and Figure 2 is the best of the set — four rows each with its own labelled ticks, a bracketed magnified interval, the later 150 as a hollow marker kept on the full line, and an exact table underneath. Figure 4 has a misplaced forbidden label and an orphaned rail (**S1**); Figure 3 draws "red" as a blue dot and "blue" as a green dot (**S7**); the numbering runs 4 → 4c → 4b in reading order and Figure 1's caption points at the wrong one (**S6**); all figure type is small at 320 px and the record overstates how small (**S5**). No caption apologises for its figure, and every figure's `aria-label` carries the numbers a sighted reader takes from it — I checked all eighteen |
| 6 | **Connections** | Strong. The `[1,2,3,4,100]` column is carried through §2, Figure 2, §7 and Figure 5 without being re-typed. The Q/A/B fixture is shared by §2, I1 and practice 1. The donor table is shared by §4, I2 and practice 5. The six-row encoding table is shared by §8, Figure 5b, I4 and practice 7. The penguin fit is shared by §6, Figure 4b, I3, the checkpoint and practice 6. NMF is named as the preceding representation constraint and cross-validation as the next owner of the boundary |
| 7 | **Code** | Two complete workflows, both executed, both reproducing their output byte for byte, both byte-identical to the manuscript. The `pip` pin is one `bash` block. `Path(__file__).with_name("penguins.csv")` runs as written because the verifier executes it beside the served file |
| 8 | **Practice** | Nine tasks, each changing numbers or the decision; hints and solutions in separate disclosures, **all 14 `<details>` closed on load** (measured); every stated value reproduced independently. Practice 5's pointer into I2 is correct and reachable — I performed both edits. **Practice 1's pointer into I1 is not** (**B1**) |
| 9 | **Accessibility** | Keyboard-only prediction and commit verified in I1 (arrow keys move the radio, two Tabs reach "Check prediction", Enter commits and the verdict appears). All 12 scrollable table regions carry `role="region"`, an `aria-label` and `tabIndex={0}`. Every control is labelled. Out-of-range entry sets `aria-invalid`, shows a linked `sc-field-error`, disables nothing silently, and reverts to the last good value on blur — the model is never handed a substitute |

## Ranked actionable findings

Severity: **blocking** = wrong on the published page, or a record that asserts something untrue; **should-fix** = an inaccuracy, a defeated teaching mechanism, or a dropped specification item with a bounded fix; **observation** = recorded for the ledger.

### Blocking

**B1. Practice 1 tells the learner that Investigation 1 accepts coordinates it refuses. Six of the eight numbers it names are outside the lab's accepted ranges.**

Location: `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` L235 (last sentence); `src/learn/components/lesson-labs/ScalingLabs.jsx` L60–71; `src/learn/data/scaling-models.js` L43–46.

The added sentence — not in the manuscript — reads:

> Investigation 1 accepts these exact coordinates and divisors if you want to watch the contributions.

Practice 1's coordinates are query `[0,0]`, A `[2,60]`, B `[5,10]`, divisors `[1,30]`. Investigation 1's six coordinate fields are bounded:

```jsx
<NumberField label="Q bill length (mm)" … min={20} max={80} …
<NumberField label="Q body mass (g)"   … min={2000} max={7000} …
```

Confirmed on the live page at 1366 px: typing `0` into "Q bill length (mm)" produces **"Keep it between 20 and 80."**; typing `0` into "Q body mass (g)" produces **"Keep it between 2000 and 7000."** Every bill value in practice 1 (0, 2, 5) is below 20 and every mass value (0, 60, 10) is below 2,000. The two divisors are the only part of the sentence that is true.

The lab degrades safely — the refusal is explicit, `aria-invalid` is set, and the field reverts to the last good value on blur rather than handing the model a substitute — so nothing false is *computed*. But the page makes a checkable factual promise to a learner who is at that moment trying to reconcile two calculations, and the promise is false. A learner who follows it will conclude either that they mistyped or that the investigation is broken.

Practice 5's analogous sentence ("Both edits are available in Investigation 2") **is** true; I performed both edits. So this is a single unchecked addition, not a pattern.

Fix, either: (a) delete the sentence; (b) restate it honestly ("Investigation 1 uses penguin-scale measurements, so try the same contrast there with the query at 40 mm / 4,000 g"); or (c) widen `limits.billLength`/`limits.mass` — but that weakens the lab's own domain plausibility, so (a) or (b) is better. Then add a browser assertion that any practice sentence naming a lab's inputs is reachable in that lab.

**B2. Four records assert that this lesson is unimplemented, the design record asserts that `blueprints/index.js` was not edited when the tree shows it edited, the ledger's design-record hash is stale, and the blueprint registration is absent from the build the browser evidence was taken against.**

Locations: `docs/teaching/lesson-delivery-progress.json` (topic `feature-scaling-encoding-imputation`); `docs/curriculum/curriculum-inventory.json` (same topic); `docs/curriculum/CURRICULUM-INVENTORY.md` L157; `LESSON-AUTHORING-HANDOFF.md` (no entry at all); `docs/teaching/drafts/feature-scaling-encoding-imputation/design.md` L87 and L142; `src/learn/data/curriculum/blueprints/index.js` L20 and L143.

Evidence:

- The ledger entry reads `"implementation": {"status": "not-started"}` with `nextAction: "On an explicit finish request, pass --work finish … Implement the lesson and visuals/labs, execute final displayed programs, complete independent correctness and learning-experience review, applicable fixes/checks and integration. Content is complete; **implementation has not started**."` Every one of those actions has in fact been performed in the working tree, including this review.
- The generated `curriculum-inventory.json` carries the same state forward: `"implementation": "not-started"`, `"recordedImplementation": "not-started"`, `"designStatus": "individual-design-required"`, `"teachingReview": "individual-review-required"`, and the same `nextAction` string. `CURRICULUM-INVENTORY.md` L157 renders the row as `| Feature Scaling, Encoding & Imputation | foundation | published | design needed |`.
- The ledger's content checkpoint binds `design.md` at `9ad40bc10fcd55598f415244fbdf3d4540d25bbb9377d5a0e91160c53aaa2536`; the file is now `cde0c09eec8adf8f1ea941acddb79fcdf6c03ecd46d5cc0c60cba2bfbe83a150` after the phase-two append. **I confirmed the other six bound files still match their recorded hashes exactly, so this is the only stale binding.**
- `LESSON-AUTHORING-HANDOFF.md` contains **no occurrence of "scaling", "imputation" or "penguin" anywhere.** Its "Latest completed scope" is still t-SNE/UMAP (position 17) and it states "Next eligible prepared topic is Independent Component Analysis (ICA), position 18." A next agent reading the handoff and the ledger together would conclude that positions 18, 19 and 20 are unimplemented.
- `design.md` L87 says "Registration in `blueprints/index.js` is the integration owner's edit, not this author's", and L142 repeats "The `docs/teaching/lesson-delivery-progress.json` ledger and `blueprints/index.js` were **not edited**". `git diff src/learn/data/curriculum/blueprints/index.js` shows `import featureScalingBlueprint from './feature-scaling-encoding-imputation.js';` at L20 and `'Feature Scaling, Encoding & Imputation': featureScalingBlueprint,` at L143, uncommitted, alongside seven sibling registrations. The file's mtime is **18:30:07**, after `design.md`'s 18:27:43. Whoever made the edit, the record now asserts something untrue about the tree it describes. This is the same self-contradiction ICA's review raised as its B2(b).
- The registration is **not in `dist-scaling`**: no chunk in the build references the blueprint module, because the build finished at 18:23:52 and the browser run ended at 18:29:13, both before the 18:30:07 edit. The lesson declares `hasIntegratedGuide: true` (`.jsx` L42), so the guide's rendered effect is unverified by any evidence in this increment. This is ICA's S8 recurring.

Mitigating, and worth recording in the disposition: `LESSON-TEACHING-STANDARD.md` L38 does place "Update the existing ledger" in stage 6 with the increment owner, and the design record explicitly disclaims closing the phase. Declining to close the ledger is therefore defensible. The blocking part is the *contents* of the `nextAction` string, the stale hash, the absent handoff entry, and the two statements about `index.js` that are no longer true of the tree.

Fix: (a) correct `design.md` L87 and L142 to record that the blueprint is registered, or, if the implementer genuinely did not make that edit, say who did and when; (b) the integration owner updates the ledger entry to `implementation: complete` with the final source hashes, refreshes the `design.md` hash in the content checkpoint, and rewrites `nextAction`; (c) regenerate the inventory; (d) add a feature-scaling entry to the handoff linking the design record's phase-two section, its evidence and this review; (e) rebuild `dist-scaling` **after** the registration and re-run `verify-scaling-browser.cjs`, so the integrated guide is covered by at least one case.

### Should-fix

**S1. Figure 4 — the one diagram whose whole subject is the fit/transform boundary — labels a permitted flow as the forbidden one and draws the withheld-answers rail starting from the forbidden marker, attached to nothing.**

Location: `src/learn/components/lesson-labs/ScalingFigures.jsx` L236–239 and L247, L257.

```jsx
<path className="sc-flow is-forbidden" d="M192,29 L152,29" strokeDasharray="5 3" />
<line className="sc-flow is-forbidden" x1="165" y1="21" x2="179" y2="37" />
<line className="sc-flow is-forbidden" x1="179" y1="21" x2="165" y2="37" />
<text x="198" y="62" style={{ fontSize: 9.5 }}>fitting on all rows</text>
…
<path className="sc-flow" d="M263,48 L263,146" />          {/* the legitimate apply arm */}
<path className="sc-flow is-target" d="M176,48 L176,216" /> {/* the answers rail */}
```

Two problems, both visible in a capture of the rendered figure:

1. **The forbidden label sits across a legitimate line.** The crossed marker is centred at (172, 29), but its label is placed at (198, 62) — below the held-out box and to the right of the marker — and the *permitted* "apply, frozen" arm runs vertically through x = 263, which passes straight through the word "all". On screen the sentence "fitting on all rows" is struck by a green connector that means the opposite thing. The nearest line to the label is the one it contradicts.
2. **The answers rail begins in empty space, directly below the crossed marker.** The training box spans x 6–148 and the held-out box x 192–334; x = 176 belongs to neither, and y = 48 is the bottom edge of both. The rail therefore emerges 4 units right of the X's centre and 11 below its lowest point, and reads as a continuation of the forbidden arrow down into "final comparison". The figure's own `aria-label` says "The held-out answers travel their own rail and join only the final comparison" — the rail lands on the comparison box correctly, but it never touches the held-out box it is supposed to leave.

Separately, **no figure in this lesson draws an arrowhead**: there is no `<defs>`, no `marker-end` and no arrow polygon anywhere in `ScalingFigures.jsx`, `ScalingLabs.jsx` or `ScalingShared.jsx`. Direction is carried entirely by small text labels. The manuscript specifies "A separate forbidden **backward arrow**" and the specification says evaluation rows apply the object "without an arrow into fitting"; a diagram whose caption is "observations do not cross **backward**" cannot express backwardness without a direction marker.

This is the same figure the design record's repair 3 says was redrawn because "Figure 4's 'apply, frozen' arrow pointed into empty space and the final-comparison box was fed by arrows that contradicted its caption". The redraw fixed the comparison box; it left an orphaned start and moved the mislabelling.

Fix: start the rail at the held-out box (`M263,48` branching, or `M230,48 L230,216` with the branch drawn), move "fitting on all rows" to sit directly under the crossed marker clear of x = 263, and add a single shared `<marker>` arrowhead used by `.sc-flow`. Then re-capture the Figure 4 screenshots.

**S2. Investigation 4's donor graph draws only the non-matching donors into the fold prior, so a learner who traces the edges computes the wrong prior — and the figure's own accessible text says otherwise.**

Location: `src/learn/components/lesson-labs/ScalingLabs.jsx` L597, with the legend at L606 and the `aria-label` at L587.

```jsx
{donor && <path className={`sc-flow${same ? '' : ' is-target'}`} d={`M126,${place(index)} L206,${same ? 74 : 140}`} />}
```

Each donor gets exactly **one** edge: a same-category donor goes to the "same category" box and nothing else; a non-matching donor goes to the "fold prior" box. In the base fixture with row 0 inspected, that means row 1 (A, y = 1) has a solid edge to "sum 1 / n 1" and **no edge at all into the prior**, while rows 3 and 5 have dashed edges into the prior. Measured on the live page and confirmed in a capture: two dashed edges reach the "fold prior 1/3" box, and the two targets on them are 0 and 0. A learner reading the drawing gets `(0 + 0)/2 = 0`; the box says 1/3, and the arithmetic table beside it correctly says the prior comes from rows 1, 3 and 5.

The specification is explicit about the thing that is missing: "Different labels/styles distinguish same-category contributions to the numerator from **all-donor contributions to the prior**." The `aria-label` already says it too: "Rows 1 share its category and enter the numerator; **all donors enter the fold prior**." So a screen-reader user is told the truth and a sighted user is shown something else.

This matters because the fold-specific prior is the whole point of §8's repair — the manuscript's own emphasis is "learn category sums, counts, **and the prior mean** from the other folds" — and I4 is the only interactive demonstration of it.

Fix: draw two edges for a same-category donor (one solid to the numerator box, one dashed to the prior box), or draw every donor's dashed prior edge first and overlay the solid numerator edges. Then re-capture the I4 screenshots.

**S3. "Calculate without recording a prediction" stays enabled after a graded check in all four investigations, and one click destroys the verdict — with no history kept in two of them.**

Locations: `src/learn/components/lesson-labs/ScalingShared.jsx` L187–189 (I1); `src/learn/components/lesson-labs/ScalingLabs.jsx` L279–283 (I2), L432–434 (I3), L567–569 (I4).

```jsx
<button type="button" className="is-primary" disabled={state.choice === '' || Boolean(shown)}
  onClick={() => state.check(answerFor, label)}>Check prediction</button>
<button type="button" onClick={() => state.explore(answerFor, label)}>{exploreLabel}</button>
```

"Check prediction" is correctly disabled once a result stands. Its neighbour is not. Since `draft` and `active` are equal at that moment, the explore click recomputes exactly the same answer and replaces the graded verdict with an ungraded restatement. Confirmed live:

- **I1**: "= Your prediction matches: B is nearer. The deciding contribution is the mass term." → one click → "· Calculated without a recorded prediction: B is nearer. The deciding contribution is the mass term." (The retired trial survives in the "Previous trial" note, so the loss is recoverable here.)
- **I3**: "= Your prediction matches. You recorded 1.31; the calculation gives 1.3091367505, a deviation of −0.000863…" → one click → "· Calculated without a recorded prediction: 1.3091367505." **I3 and I4 keep no previous-trial element at all, so the record that the learner predicted anything is simply gone.**

I2's and I4's buttons carry only `disabled={!valid(state.draft)}` or nothing. This is the same defect NMF's review raised as its S3, and the fix is identical.

Fix: add `disabled={Boolean(shown)}` to the explore button in `ScalingShared.jsx` L189 and `disabled={Boolean(result)}` to the three hand-rolled ones. One attribute each; no verifier rerun beyond a browser re-capture.

**S4. `.sc-point-id` is inert. The stylesheet's `font` shorthand and its higher specificity strip the observation identifiers of the weight and colour that make them identifiers — the same class of defect the record says was repaired.**

Locations: `src/learn/components/lesson-labs/scaling-labs.css` L70 and L81.

```css
.sc-plot svg text, .sc-line svg text, .sc-figure svg text, .sc-panel svg text { fill: #cfd7d2; font: 11px var(--font-mono, monospace); }
…
.sc-point-id { font-weight: 700; fill: #f0e6c8; }
```

`.sc-line svg text` has specificity (0,1,2); `.sc-point-id` has (0,1,0). The `font` shorthand also resets `font-weight` to `normal`. So both declarations in the `.sc-point-id` rule lose. Measured on the live page at 1366, 390 and 320 px, every element carrying the class reports `font-weight: 400` and `fill: rgb(207, 215, 210)` — the generic body colour, not `#f0e6c8`.

Affected: the `a`, `b`, `c`, `d`, `e`, `new` and `a–d` markers on all eight number lines in Figure 2, and the `100` and `1–4` callouts in Figure 5. These are exactly what the F2 contract asks for — "A source-row ID/shape follows each point across views" — and what makes the later `150` distinguishable from the fitted column at a glance. They currently look like axis text.

The design record's repair 1 is that "`.sc-figure svg text` sets the `font` shorthand, and a CSS rule beats an SVG presentation attribute", fixed by moving sizes to inline styles. That fix is real and complete for sizes — I confirmed there is not one surviving `fontSize=` presentation attribute in either figure file, and the inline styles do win. But the same shorthand still silently defeats the one class whose whole job is emphasis, and nothing in the repair list covers it.

Fix: raise the rule's specificity (`.sc-figure svg text.sc-point-id`, etc.) or move the `.sc-point-id` declarations into the same block with `!important`-free ordering. Better: replace the `font` shorthand at L70 with `font-size` + `font-family` so it stops resetting weight, style and line-height for every `<text>` in the topic.

**S5. At 320 px, half the figure text renders below 9 px and a quarter of it below 8 px; the design record claims the smallest is "near 9 px".**

Locations: `docs/teaching/drafts/feature-scaling-encoding-imputation/design.md` L131; `src/learn/components/lesson-labs/ScalingFigures.jsx` (the `fontSize: 9.5` labels throughout) and `ScalingShared.jsx` L260.

Measured in the DOM at a 320 px viewport, computing each `<text>` node's rendered size as `computedFontSize × (renderedWidth ÷ viewBoxWidth)` across all nine figures:

| | count |
| --- | --- |
| figure `<text>` nodes | 181 |
| below 9 px | **93** |
| below 8 px | **42** |
| smallest | **6.82 px** — "expanded below", the label that connects each full-range line to its magnified view |

Representative 7.82 px labels: Figure 1's `sex_female` / `sex_male` / `sex_not_recorded` output coordinates and its `training median` / `inserted estimate` cell — the two things that stage exists to name; Figure 4's `features and answers`, `features only, for now`, `medians · centres · scales`, `apply, frozen` and `fitting on all rows`. At 390 px the smallest is 8.68 px and Figure 1's output names are 9.78 px.

The record states: "At 320 px the diagram text was about 7 px. The two small sizes were raised by roughly 20 % … the smallest label now renders near 9 px at 320 px." The raise happened — the sizes in source are 9.5 and 10.5 rather than 8 and 9 — but the outcome claim is wrong by about 2 px at the bottom of the distribution, and the record does not mention that "expanded below" is set at 8 units and was never raised.

Nothing else about narrow widths is broken: the document does not scroll horizontally at 1366, 390 or 320 px, and all ten to twelve clipped tables are inside `overflow-x: auto` regions that carry `role="region"`, an `aria-label` and `tabIndex={0}`, so they are reachable by keyboard.

Fix: raise `ScalingShared.jsx` L260's 8 to at least 9.5, raise the 9.5-unit labels to 10.5 where the box allows, or reduce the widest figures' `viewBox` widths so the scale factor at 320 px is nearer 1. Then correct `design.md` L131 to the measured value.

**S6. The figures are numbered out of reading order — 4, then 4c, then 4b — and Figure 1's caption sends the reader to the wrong one.**

Locations: `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` L163 (`<ComparisonFigure />`) and L169 (`<PipelineFigure />`); `src/learn/components/lesson-labs/ScalingFigures.jsx` L55–56.

Reading the rendered page top to bottom, the nine captions are: Figure 1, Figure 2, Figure 3, Figure 4, **Figure 4c**, **Figure 4b**, Figure 5, Figure 5b, Figure 6. `ComparisonFigure` is placed immediately after the results table and `PipelineFigure` twelve lines later, so the "b" variant follows the "c" variant.

Compounding it, Figure 1's caption says:

> the actual fitted statistics arrive in **Figure 4** and Investigation 3.

Figure 4 is the boundary schematic; it contains no fitted statistic. The medians, centres and scales are in Figure **4b**, which the reader now meets *after* Figure 4c.

Also in this family: Figure 5b is a §8 figure named as a variant of §7's Figure 5, which it has nothing to do with.

Fix: renumber the §6 pair so the boundary diagram, the fitted bundle and the comparison read 4, 4b, 4c in page order (swap the two components, or rename); correct Figure 1's caption to "Figure 4b"; and rename Figure 5b to a §8 number.

**S7. Figure 3, whose subject is the categories red, green and blue, draws "red" as a blue dot and "blue" as a green dot.**

Location: `src/learn/components/lesson-labs/ScalingFigures.jsx` L181–183, with `scaling-labs.css` L82–83.

```jsx
<circle className="sc-mark-b" cx={place(0)} cy={lift(0)} r="6" />   {/* red (0,0)  → #91aecf, blue */}
<circle className="sc-mark-a" cx={place(1)} cy={lift(0)} r="6" />   {/* green (1,0) → #8eb9a5, green */}
<circle className="sc-mark-a" cx={place(0)} cy={lift(1)} r="6" />   {/* blue (0,1)  → #8eb9a5, green */}
```

The colour choice is a role distinction — the dropped reference corner against the two kept categories — and the specification does allow "Color distinguishes roles but labels, shapes, and connectors carry the same meaning", which they do here (every point is labelled with its name and coordinates, and the exhaustive pairwise-distance table is underneath). But this is the one lesson in the module where the categories *are* colours, and a reader glancing at the plane sees a blue marker captioned "red (0,0)". The confusion is gratuitous and costs nothing to remove.

Fix: give the reference corner a distinct *shape* (a hollow square) rather than a distinct hue, or use a neutral grey for all three markers.

### Observations

**O1. None of the four verifiers can be re-run without writing to the tree, and one of them rewrote a source module after the build the browser evidence used.** All four write their `docs/teaching/evidence/*.json` unconditionally; `verify-scaling-examples.py --write` rewrites `src/learn/data/scaling-examples.js`; `verify-scaling-data.py` rewrites `src/learn/data/scaling-data.js` (mtime 18:28:26, after the 18:23:52 build). The rewrite happened to be idempotent — I confirmed every statistic, row and count in the built chunk equals the current module — but an independent reviewer cannot re-run any verifier without violating a no-write boundary, and an implementer cannot re-run the data verifier without invalidating the build. Consider a `--check` mode that asserts without writing, and ordering the build after the data regeneration.

**O2. Figure 2's min–max panel states the fitted minimum and range where the contract asks for the minimum and the maximum.** `ScalingFigures.jsx` L114 renders "Fitted minimum 1, fitted range 99"; the F2 contract says "min 1/max 100". The maximum is recoverable from the two and the prose below names it, so nothing is lost, but the contract item is not literally met.

**O3. Investigation 3 grades a numeric prediction to ±0.05 while asking for four decimals.** `ScalingLabs.jsx` L378 and L426. Entering `1.3` against `1.3091367505` is reported as "Your prediction matches"; the verdict does say "That is within the 0.05 tolerance this investigation grades on", so the tolerance is disclosed. For a coordinate whose whole point is an exact arithmetic trace, ±0.05 is loose — practice 6 asks the learner to compute `−0.2368`, and `−0.20` would pass.

**O4. Investigation 2's verdict lists the predicted donors sorted and the computed donors in distance order**, so a correct prediction can read "You recorded D1 and D2 supplying 200. The calculation uses D2 and D1 and gives 200." (`ScalingLabs.jsx` L288–290.) Print both in the same order.

**O5. Investigation 3's category readout stutters in the absent case**: "The categorical branch received **an absent value**, which is **absent**, so the fitted not_recorded category absorbs it" (`ScalingLabs.jsx` L465–468). The three clauses are chosen from the computed state, which is right; only the wording repeats.

**O6. Figure 4c's bars carry hard-coded hex fills on a class with no stylesheet rule.** `ScalingFigures.jsx` L337–338 sets `fill={row.method === 'majority' ? '#746138' : '#8eb9a5'}` on `<rect className="sc-bar">`, and `scaling-labs.css` defines `.sc-bars`, `.sc-bar-row`, `.sc-bar-track`, `.sc-bar-fill` and `.sc-bar-value` but never `.sc-bar`. The two colours are the only place in the topic where a figure colour bypasses the stylesheet, and the baseline's different colour is never explained in the caption.

**O7. The manuscript's execution claim for the second program was dropped and not replaced.** §8's "This bounded calculation was executed during authoring" is absent from the page, and `RunnableExample` labels its output only "Expected result". The §6 sentence that replaced the equivalent §6 claim is scoped to "The block below". In fact `verify-scaling-examples.py` executes *both* programs and pins both stdouts, so the honest sentence is available and stronger; it is simply not on the page for the §8 block.

**O8. Figure 6 renders `θ̂` and `θ̄` as `θ^` and `θ‾` in the monospace SVG font** (`ScalingFigures.jsx` L479, L484). The combining marks do not compose. The HTML table and prose around it are fine.

**O9. The served asset folder carries no provenance file.** `public/learn-assets/feature-scaling/` contains only `penguins.csv`, where the ICA and cross-validation topics each serve a `data-provenance.md` beside their data. The licence, attribution, byte count and SHA-256 are on the lesson page itself, which is adequate and arguably better, but a learner who downloads the CSV alone gets no attribution with it.

**O10. Dead exports.** `ScalingShared.jsx` L123–159 exports a `useInvestigation` hook that this topic never uses (all four labs use `useInvestigationWithHistory` in `ScalingLabs.jsx` L650), and L300 exports `curve`, which is never called. In `scaling-models.js`, `boxCox`, `yeoJohnson`, `signedHash`, `l2Normalize`, `rankCoordinates`' sibling helpers and the exported guards are verified by `verify-scaling-models.mjs` but never reached from the UI — in particular the §8 signed-hash table (`.jsx` L217–220) is hand-written literal strings (`'3 − 1 = 2'`, `'[2,2]'`) although `signedHash` exists and is verified. The departures table's justification for moving two tables into figures — "computed live from the model layer, so they cannot drift from the arithmetic beside them" — is not applied to this third table.

**O11. The `readTime` metadata string does not match the route paragraph.** `.jsx` L41 reads "~50 min first pass · ~95 min complete read + 50–80 min code and practice"; the route paragraph (L47) offers "roughly 50 minutes for the core reading and another 50 to 80 for calculation and code" and says nothing about 95.

**O12. Investigation 1's caption promises a refusal message the field does not give.** The caption (`ScalingLabs.jsx` L77–79) says "A divisor must be greater than zero: dividing by nothing is not a ruler, so the field refuses it instead of quietly substituting 1." Typing `0` produces "Keep it between 0.000001 and 1000000." The refusal is real and the substitution warning is honoured; only the message is generic, and the actual lower bound (1e−6) is not "greater than zero".

**O13. The `result.key` fingerprint is computed and stored but never read.** `ScalingLabs.jsx` L213, L367, L519. Staleness is prevented instead by clearing `result` inside every `edit`, which works — I could not defeat it — but the unused key is a maintenance hazard: a future edit path that forgets to clear `result` would fail silently, where a key check on render would catch it. ICA's `useIcaInvestigation` re-checks the key on every render; this one does not.

**O14. Investigation 1 offers no point movement.** The I1 contract says learners can edit "by number input **or** point movement"; only number inputs are implemented, which satisfies the "or". Recorded because the plane is drawn and would support dragging.

## Disposition

The numerical and provenance core of this lesson is the strongest I have reviewed in this series. **Every one of the roughly sixty distinct numeric claims — the 258/86 split, all four fitted rulers, all 602 transformed held-out values, all five correct counts, all four confusion matrices, every constructed fixture, both program outputs and all nine practice answers — was reproduced independently, from the served CSV and the manuscript's own arithmetic, with zero disagreements.** The served CSV is byte-identical to the file the provenance URL serves today and to the cross-validation topic's copy; neither topic has touched the other's. Every external URL resolves and supports its sentence, including the primary hashing paper checked from the PDF itself. The leakage claim is true in both halves and is enforced in the model layer, and no figure or investigation refits on anything it should not. The three-investigation staleness bug the implementer found late is genuinely repaired, and I could not find a surviving case anywhere on the page where a displayed number does not belong to the committed inputs.

The two blocking findings are one false sentence added to a practice solution (**B1**) and the record set (**B2**), which repeats the pattern the sibling reviews found. Five of the seven should-fix findings are figure defects — one misleading diagram (**S1**), one incomplete graph (**S2**), one inert CSS class (**S4**), one overstated legibility claim (**S5**) and one numbering slip (**S6**) — and the remaining two are a one-attribute interaction leak (**S3**) and a colour choice (**S7**). None of them touches a number.

Recommended order: **B1** (one sentence), **S3** (four attributes), **S2** and **S1** (the two figures that teach a mechanism), **S4**, **S6**, then **S5** and **S7**. **B2** belongs to the integration owner, together with a rebuild of `dist-scaling` after the blueprint registration so the integrated guide is covered by at least one browser case.
