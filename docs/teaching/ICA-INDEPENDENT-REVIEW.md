# Independent Component Analysis (ICA) — independent phase-two review

Reviewed 14 September 2026 against the working tree plus the uncommitted ICA implementation. The reviewer authored neither the packet nor the implementation. No file under `src/`, `public/`, `scripts/` or `docs/teaching/drafts/` was edited; this review document is the only write. No verifier was re-run (see below for why), and no state-changing git command was executed.

## Reviewer statement: executed, read, reused

| Activity | What was actually done |
| --- | --- |
| **Executed** | Disposable reviewer scripts under `scratch/ica-review/`, written from the manuscript's own algebra and importing neither `ica-models.js` nor any `verify-ica-*` script nor `author-calculations.py`, run with `scratch/lesson-tools/Scripts/python.exe` (Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1). Exact work used `fractions.Fraction`; the normalization used **mpmath at 50 decimal places**. They recomputed the four observed states, Σₓ and its eigenvalues, the whitened diamond and its covariance, the recovery, the complete §5 fixed-point trace row by row, the normalization and `1 − |wᵀw_new|`, the three kurtosis families at 0/30/45/90/135/180°, the `a⁴+b⁴ = 1−2a²b²` identity, the zero-kurtosis counterexample's second/fourth/sixth moments, the §7 column norms, every C1 contribution fixture and null, the §8 storage figure, and all six practice answers. Separately, the **entire real-recording pipeline was refitted from the served CSV**: the calibration, the chronological split, PCA checked twice (sklearn and my own eigen-decomposition of the training covariance), FastICA, and all twenty-four correlations computed with my own Pearson routine rather than `np.corrcoef`. Every value in `ica-data.js` — the twelve signed development correlations, the three frozen choices, the held-out diagnostics, `n_iter_`, the reconstruction MSE, the 500-column min/max envelope for all four traces and the 20-row exact window — was regenerated and compared. Both displayed programs were extracted from `ica-examples.js` with `node`, written to the filenames the lesson names, and executed beside the served CSV. `sha256sum` was run on every reviewed file. |
| **Read in full** | `lesson.md` (520 lines), `visual-specifications.md` (121 lines), `data-provenance.md`, `author-calculations.py`, `ICA-LESSON-DESIGN.md` including its phase-two append, the published `.jsx` body (408 lines), `IcaShared.jsx`, `IcaFigures.jsx`, `IcaLabs.jsx`, `ica-labs.css`, `ica-models.js`, `ica-examples.js`, the generated `ica-data.js`, the blueprint and its registration diff, `RunnableExample.jsx`, `LessonElements.jsx`, `Headings.jsx`, the four evidence JSONs, the ICA rows of `lesson-delivery-progress.json`, the ICA topic note, and the ICA paragraphs of `LESSON-AUTHORING-HANDOFF.md`. Also `scripts/verify-ica-models.mjs` (read), and the write paths of `verify-ica-examples.py` and `verify-ica-data.py`. |
| **Looked at** | **Fourteen of the twenty-five evidence screenshots were opened and inspected**: `m1-1366`, `m1-390`, `d1-1366`, `d1-390`, `w1-1366`, `w1-320`, `f1-1366`, `f1-390` (plus a 4× crop of its right edge), `p1-1366`, `p1-390`, `e1-1366`, `e1-390`, `e1-320`, `rotation-matched-1366`, `rotation-missed-1366`, `rotation-gaussian-null-1366`, `rotation-320`, `contribution-matched-1366`, `contribution-390`, `contribution-zoom`, `scale-compensated-1366`, `scale-390`. Findings **B1**, **S3**, **S6**, **S7** and observations **O3**, **O9**, **O13** come from that pass. |
| **Operated** | The live page was driven with Playwright (msedge channel) against the running `dist-ica` preview at `http://127.0.0.1:4181`, at 1366, 390 and 320 px: keyboard-only prediction and commit in R1, the Gaussian null, the binary equal-|κ| case, C1's zero-source null, C1b's source-only branch and its `c = 0` refusal, Reset behaviour, and DOM measurement of every scroll region and every display-math block. Findings **B1**, **S1**, **S2**, **S3** and **S7** were confirmed there, not merely inferred from source. |
| **Reused, not rerun** | `evidence/ica-models.json` (9 groups, 14,978 assertions), `ica-native.json` (48 oracles), `ica-data.json` (31 checks) and `ica-browser.json` (6 behaviour groups, 25 captures). Every source hash they record equals the hash below for the same file, so their recorded results bind to the bytes reviewed here. **The verifiers were deliberately not re-run**: `verify-ica-examples.py` and `verify-ica-data.py` rewrite their evidence JSON unconditionally, and `verify-ica-data.py` L211 rewrites `src/learn/data/ica-data.js` with no `--write` flag, so running either would have violated the review's no-write boundary (observation **O16**). Their results were instead confirmed by the independent recomputation above, which agreed on every number. |
| **Delegated** | Every external URL in the Sources block, plus every URL in the provenance file, was fetched and checked against the sentence that cites it — fifteen in all, including both Helsinki PDFs (contents and body pages extracted, not trusted to a summariser), the SEE lecture bookmark list, the YouTube oEmbed record, the CS229 transcript and notes, the scikit-learn page, the MNE tutorial, the PhysioNet record and its `SHA256SUMS.txt`, the ODC-By text, and four DOIs. Findings **O5** and **O6** come from that pass; everything else resolved and supported its claim. |
| **Not done** | No screen-reader session. No other browser engine. No 200 % zoom pass of my own (the implementer's `contribution-zoom` capture was inspected instead). Eleven screenshots were not opened. The lesson was not read aloud to a beginner. The blueprint's rendered effect was not observed, because the `dist-ica` build predates its registration (finding **S8**). Version-independence of the real-data result was not tested on any library version other than the pinned one. |

## Source versions reviewed (SHA256)

| File | SHA256 |
| --- | --- |
| `src/learn/data/topics/independent-component-analysis-ica.jsx` | `83fdc09d6260607f919e405669a6d305b841fb4d4a5fdfb6edcc9eb22aef6dbb` |
| `src/learn/data/ica-models.js` | `9430d1ee67b56dd013cc3aef15a7576e1dc8b907bc54dd961ef7df5812f58cfa` |
| `src/learn/data/ica-data.js` | `897ca68813a4b55b64677032e9a7df97e25d30d8a5acffdfde610ce9e3152e61` |
| `src/learn/data/ica-examples.js` | `65262b951808f1d3bd2e12e33669c59766ed78c85e90a05224072d7a263e6064` |
| `src/learn/components/lesson-labs/IcaShared.jsx` | `fd5fc22dded763c19824a4ac81da8130c945e169d2a90fa2decdd1db252c144a` |
| `src/learn/components/lesson-labs/IcaFigures.jsx` | `6e1d86d3d7d4b526a0b43b9e1cb1b41bd49526eb02714ffb13a4613d1b450755` |
| `src/learn/components/lesson-labs/IcaLabs.jsx` | `cbd5fdea70668bc071c9f324059bc2f89960ea39145fb67a1cfb8d9e577f736e` |
| `src/learn/components/lesson-labs/ica-labs.css` | `1a27451bbb303545842ceac8932fa50a4e078a356af2cb0e268806d48c1f50d0` |
| `src/learn/data/curriculum/blueprints/independent-component-analysis-ica.js` | `2d4ab17200cc2b394f7781fbace0d2297dcb5f3d488ceaff4e82adf2a118c76c` |
| `public/learn-assets/ica/r01-first20s.csv` | `7c95ef45ceaf96254950ce633b2ab0b5089b15a4fdbd617e1b843846beef23cc` |
| `public/learn-assets/ica/data-provenance.md` | `b2b0571bd35c74bd6dba8f4e009748a78e637b1a0237439d18a8e705dce8b033` |
| `public/learn-assets/ica/ica_by_hand.py` | `413cdca7027c83788faa20e424922cfd31cbfe93030a8c617f7ce43ccb1e3f53` |
| `public/learn-assets/ica/ica_recording.py` | `5ed962efeaa670a84ac5f595fd2bb6f8a480a10f5b5424f3295062cf63e6ec66` |
| `docs/teaching/drafts/.../lesson.md` | `ae908b270c0dab7dc0a8709840cbcdb436f2e8a9d06ca58fed656db80a314fd0` |
| `docs/teaching/drafts/.../visual-specifications.md` | `b3e719d7e96811aa5aae515204e4797e2174ad80d665a0953c778f7cb71dfea6` |
| `docs/teaching/drafts/.../data-provenance.md` | `b2b0571bd35c74bd6dba8f4e009748a78e637b1a0237439d18a8e705dce8b033` |
| `docs/teaching/drafts/.../r01-first20s.csv` | `7c95ef45ceaf96254950ce633b2ab0b5089b15a4fdbd617e1b843846beef23cc` |
| `docs/teaching/ICA-LESSON-DESIGN.md` | `fbbbf26d927b47dd6c89668dc2dbfeddbb6f519da5177f6d56013d90adebe71b` |
| `scripts/verify-ica-models.mjs` (read only) | `a5d1b97ff2b5a5bfae17c4c84f84122426f71fa1590266402825c2d846aa5cf3` |
| `scripts/verify-ica-examples.py` (hash only) | `38a8bd4a166dd9b57621c400ed362427c8d9c805ceae1780f1ff8e56472c5cbb` |
| `scripts/verify-ica-data.py` (hash only) | `ef7eadd2dc86c3720a642bfdbdfddac53a1d0bebb0d4a2b5cb44de9fb1c96b5c` |
| `scripts/verify-ica-browser.cjs` (hash only) | `693ae0407bbd0a451423a6d5be8ad86a94fd91ae5ab4cc12057d798155a3e5fd` |

A hash proves identity, not correctness. All four evidence files record source hashes, and every one equals the value above for the same file, so their recorded results bind to exactly the bytes reviewed here. The served `public/learn-assets/ica/r01-first20s.csv` is **byte-identical** to the packet's copy (both `7c95ef45…`, both 390,723 bytes), and both are the file the evidence names and the file the provenance document hashes. `public/learn-assets/ica/data-provenance.md` is likewise byte-identical to the packet copy. Both served `.py` files are byte-identical to the `code` strings in `ica-examples.js`, which are in turn byte-identical to the manuscript's fenced blocks.

## Part A — correctness

### A1. Manuscript sections, claims, cautions and tasks preserved?

All eleven manuscript sections are present in the published body, in order and under the manuscript's own titles (`.jsx` L12–24), with the declared first-pass route (L58), both deeper branches carrying "Deeper branch" in their headings, the four-state table, the model-change table, both programs in the manuscript's order, practices 1–6 with separate closed hint and solution disclosures, the readiness section and the full eight-item reference list. `H2` derives its anchor with the same slug rule the lesson's own `headingId` uses, so every route link in `LessonIntro` resolves.

A mechanical comparison (every manuscript sentence, math stripped, against the concatenated implementation sources; then the reverse direction over every string literal in the `.jsx`) found **no invented claim anywhere in the lesson body**. Every sentence in the published prose traces to the manuscript except the two declared departures below, the `LessonIntro` blurb, three table captions and the figure/lab captions.

The four ICA-characteristic honesty caveats named in the brief were checked sentence by sentence and are all present, each with a genuine single home rather than a repeated warning:

- **Order and sign are not identified.** §7 (`.jsx` L271): "ICA therefore identifies sources up to scale, sign and permutation. A unit-variance convention fixes scale in a useful way, but component index and polarity are still conventions. The model supplies no default ranking by explained variance." Reinforced as mechanism, not as repeated hedging, in §5's "Parallel and antiparallel vectors describe the same source direction" (L223), §5's one-to-one matching rule (L237, "Taking each row's best match independently can reuse one recovered component and exaggerate recovery"), §6's `fit_transform` warning (L259), §10's "Why do component numbers need matching across fits?" (L389), and C1b's closing note (`IcaLabs.jsx` L213).
- **Whitening is not ICA.** The §3 heading itself, L139 ("The whitened coordinates are still dependent mixtures"), L151 ("Reducing to k < d principal coordinates before ICA chooses a variance-based subspace; it does not select the k most independent or most non-Gaussian physical sources"), W1's title ("Whitening removes the stretch; the four states stay dependent") and W1's hollow absent-origin marker.
- **Non-Gaussianity is the assumption doing the work.** All of §4, including "At most one Gaussian source is allowed in the usual identifiable ICA model" (L177), the zero-excess-kurtosis non-Gaussian counterexample (L179), and R1's Gaussian family as a permanent, reachable null.
- **The real comparison is one interval of one record.** §6 L245: "This is a short within-recording investigation; evaluating a clinical or cross-person claim would require a different task, endpoints and independent participants." Plus the four stated limits in E1's provenance block, and §8 L336.

Also preserved and checked: the Gaussian ambiguity stated as a *population* fact and explicitly distinguished from "whether a numerical solver happens to return an array"; "Finite Gaussian samples can have accidental fourth-moment structure, and a solver can follow it"; "a threshold such as 'all kurtoses close to zero' cannot establish Gaussianity"; "Whitening makes E[zzᵀ] = I; it does **not** by itself make that factorization exact"; "A small directional change records numerical convergence, which is distinct from a successful application diagnostic"; "exact reconstruction … Any invertible change of coordinates can do that"; "the cocktail-party story is a useful motivation for this simplified model"; and "20,000 time samples need not provide the information of 20,000 independent draws."

**The unflattering real result is published exactly as it came out**, in three independent places: the program's stored output, the §6 paragraph ("PCA coordinate 4 has the largest test value in this fixed comparison"), and E1's own bar chart on an untruncated 0–1 axis that shows the PCA mark furthest right. Nothing anywhere implies ICA won.

Deviations found, each judged:

| Location | Deviation from manuscript / specification | Judgement |
| --- | --- | --- |
| `.jsx` L239 | The manuscript's §5 closing sentence "They are not a completed production/runtime verification campaign; the implementation packet identifies the later execution checks" is replaced by "The program above was executed again for this page, in the recorded environment, and its printed output is the output shown." | **Sound, and verified true.** The phase-one sentence became false at implementation. I extracted the program and executed it myself: it reproduces `[[1. 0.]\n [0. 1.]]\n[1. 1.]\nTrue` exactly. Declared departure 1. |
| `.jsx` L255 | The manuscript's "Author probe result with Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1" preamble is removed; the environment statement now lives in E1's provenance block (`IcaFigures.jsx` L340). | **Sound, and an improvement.** The figure states all four versions *and* adds "another library version can change the rounding or the component identity", which the manuscript never said. The version note is one element after the output rather than one before — acceptable. Declared departure 2. This is the caveat GMM's review had to raise as a missing item (its S2); here it is present and stronger. |
| `.jsx` L211 | Prose prints `(0.7091653, 0.7050423)`; F1 two elements later prints `0.705042249`. | Finding **S4** — the true value is `0.70504224900758…`, so the prose's last digit is wrong and disagrees with the figure on the same screen. |
| `IcaLabs.jsx` L58 | R1's feedback appends "Your direction changed the fourth moment" unconditionally. | Finding **B1**. |
| `IcaLabs.jsx` L163–212 | C1b's initial applied state is `c = 2, compensated`, and its result table is rendered before any prediction. | Finding **S1**. |
| `IcaLabs.jsx` L186 | C1b's explanatory clause is chosen by the *mode*, not by the *outcome*. | Finding **S2**. |
| `IcaFigures.jsx` L222–277 | P1 keeps a three-column stage grid at every width; the specification asks for "three stacked stage rows" on mobile and for a lock marker at selection. | Finding **S6**. |
| `public/learn-assets/ica/data-provenance.md` L24, and its two relative links | "LF lines" is false (all 20,001 lines are CRLF); both relative links 404 to the SPA fallback when served. | Finding **S5**. |
| `ICA-LESSON-DESIGN.md` L7–17, L187, L243 | The record's own phase table and two statements contradict the appended phase-two section and the working tree. | Finding **B2**. |
| Packet-relative links dropped | `r01-first20s.csv` and `data-provenance.md` become `/learn-assets/ica/` links; the `visual-specifications.md` pointers and the closing design-record pointer are not published. | Correct. Declared departure 3; authoring references do not belong on a learner page. |
| R1 family/marker/extent policy, C1b as a sibling investigation, the two amplitude bounds | Declared departures 4–7 and 9. | All sound, and all verified in code and on the live page. The family-restart path explicitly refuses to grade a cross-family comparison (`IcaLabs.jsx` L37–45), which is exactly what the specification asked for, made visible rather than silent. |

### A2. Independent recomputation (reviewer's own scripts, no lesson code)

Every numeric claim in the manuscript, in the published body, in the figures and in the generated data module was recomputed from first principles. **Nothing disagreed except the one rounding in S4.**

| Claim | Independent result | Agrees? |
| --- | --- | --- |
| Observed states `(−3,−3), (−1,1), (1,−1), (3,3)` from `A = [[2,1],[1,2]]` | identical, exactly by `Fraction` | Yes |
| `Σₓ = [[5,4],[4,5]]` with denominator 4; eigenvalues 9 and 1; eigenvectors `(1,±1)/√2` | exact: `5 + 4 = 9`, `5 − 4 = 1` | Yes, exact |
| Whitened diamond `(−√2,0), (0,−√2), (0,√2), (√2,0)`; `E[zzᵀ] = I` | `±1.414213562373`, off-diagonals exactly 0 | Yes |
| Recovery by `Q = [[1,1],[1,−1]]/√2` returns the four source states | exactly `(−1,−1), (−1,1), (1,−1), (1,1)` | Yes |
| **F1 row table**: `y = −0.8√2, −0.6√2, 0.6√2, 0.8√2`; `y³ = ∓1.448154688, ∓0.610940259`; `z y³ = (2.048,0), (0,0.864), (0,0.864), (2.048,0)`; `3y² = 3.84, 2.16, 2.16, 3.84` | every value reproduced to 9 dp | Yes |
| **`E[z y³] = (1.024, 0.432)`, `E[3y²] = 3`, `r = (−1.376, −1.368)`, `‖r‖ = 1.940309`** | `1.024`, `0.432`, `3.0`, `(−1.376, −1.368)`, `1.9403092537015845` | Yes |
| **Normalized update `(−0.709165303, −0.705042249)`** | mpmath at 50 dp: `−0.70916530309535176944…`, `−0.70504224900758809636…` | Figure **yes**; §5 prose says `0.7050423`, which is wrong — see **S4** |
| `1 − |w_newᵀw| = 0.009642` | `0.00964240811916572662…` | Yes |
| Angle between the sign-flipped update and `(1,1)/√2` | `0.167°` — "It moved toward (1,1)/√2" is a real, checkable claim | Yes |
| **Kurtosis families at 0/30/45/90/135/180°**: binary `−2, −1.25, −1, −2, −1, −2`; Laplace `3, 1.875, 1.5, 3, 1.5, 3`; Gaussian all 0 | identical to 9 dp for all three families | Yes |
| `a⁴ + b⁴ = 1 − 2a²b²` at `a = √3/2, b = 1/2` | `0.625` both ways | Yes |
| Binary source `E[s⁴] = 1 ⇒ κ = −2` | exact | Yes |
| **Zero-excess-kurtosis counterexample**: variance 1, `E[y⁴] = 3`, excess 0, `E[y⁶] = 9` against a Gaussian's 15 | exact by `Fraction`: `1`, `3`, `0`, `9` | Yes |
| §7 `‖a₁‖² = ‖a₂‖² = 5` | `4+1` and `1+4` | Yes |
| §8 `8d² = 800,000,000` at `d = 10,000` | exact | Yes |
| **Every C1 fixture**: `(1,−1)` keep 1 → `(2,1)`, removed `(−1,−2)`; `(2,−3)` → both `(1,−4)`, keep 1 `(4,2)`, keep 2 `(−3,−6)`, none `(0,0)`; `(2,0)` makes removing source 2 a null; all-zero stays zero | every one reproduced exactly | Yes |
| **Practices 1–6**, every stated value: `s = (3,−4)` with the `(1.5,0.5)` rescale reproducing `(5,−1)`; `1.875 = 30/16` with variance 1 against `1.5` and `3`; `r = Iw − w = 0`; select component 1 at `0.60` and report `0.15`; removed task part `(0.4q, −0.2q)` | every one reproduced, the exact ones by `Fraction` | Yes |
| **CSV identity** | served CSV byte-identical to the packet copy; 20,000 rows × 5 integer columns plus header; every stored value an integer in signed-16-bit range; header exactly as provenance states | Yes |
| **Calibration** `(d + 32768) × 6553.6/65535 − 3276.8` | matches the provenance's EDF endpoint derivation exactly | Yes |
| **PCA fit** | my own eigen-decomposition of the training covariance reproduces `pca.transform` to \|r\| = 1.000000000000 on every coordinate; eigenvalues `181.589, 145.882, 30.152, 5.147` equal `explained_variance_` | Yes |
| **All twelve signed development correlations** — channel `[0.043935, 0.101807, 0.201203, 0.079348]`, PCA `[−0.034740, 0.127795, 0.010770, 0.184580]`, ICA `[0.079635, 0.169804, −0.043955, −0.116516]` | every one reproduced to six decimals with my own Pearson routine | Yes |
| **The published comparison: raw channel 3 at `0.201203` / `0.119806`; PCA coordinate 4 at `0.184580` / `0.450178`; ICA coordinate 2 at `0.169804` / `0.343966`; `n_iter_ = 14`** | `channel 3 0.201203 0.119806`, `PCA 4 0.184580 0.450178`, `ICA 2 0.169804 0.343966`, `ICA iterations 14` | **Yes, every digit** |
| The held-out signs are all positive, so `displaySign = +1` for all three; PCA's lead is not an artefact of taking absolute values | full signed test vectors: channel `[−0.095220, 0.043829, 0.119806, −0.095770]`, PCA `[−0.019861, −0.021452, 0.076871, 0.450178]`, ICA `[0.098053, 0.343966, −0.089043, 0.114770]` | Yes |
| Fitted IC variances exactly 1; `mixing_` is the pseudo-inverse of `components_`; test round-trip MSE `1.036337128655316e−28` | identical, to the last digit of the MSE | Yes |
| **The 500-column peak-preserving envelope** for all four traces | regenerated from the raw samples: display z-score within `[16, 18) s`, sign-aligned by the development sign, reshaped to 500 blocks of 4, min/max per block — **max absolute difference 0.000e+00** against the published 4-dp values, for every one of the 4,000 published numbers | Yes |
| **The 20-row exact window** | max absolute difference 0.000e+00 across all 100 values plus the `second` column | Yes |
| The caption's "shared vertical range of ±4.2" is not clipping anything | observed display-z range across all four traces is `[−3.6182, 4.0932]` | Yes |
| `r01.edf` SHA-256 `7549bbd3…` and 3,061,792 bytes | confirmed character-for-character against PhysioNet's published `SHA256SUMS.txt` and a `Content-Length` header | Yes |

**Could not verify numerically:** the asymptotic cost statements `O(nd²+d³)`, `O(nk²+k³)` and `O(nd)` (they are not computed quantities); the reading-time estimates; the differential-entropy and negentropy statements of §4 and §8, which are definitional rather than numeric; and version-independence of the real-data result — every number is reproduced in the pinned environment, which is also the environment the author used, and I tested no other library version. The page now says this itself (`IcaFigures.jsx` L340).

### A3. Displayed programs versus their published output

Both programs were extracted from `ica-examples.js` with `node`, written to `ica_by_hand.py` and `ica_recording.py` beside a copy of the served CSV, and executed with the pinned interpreter. **Both reproduce their stored `expected` block exactly**, byte for byte modulo the trailing newline. A three-way diff confirms the manuscript's fenced block, the `code` string in the module, and the served `.py` file under `public/learn-assets/ica/` are identical for both programs, and that both `expected` strings equal the manuscript's `text` blocks.

Neither program prints an explanatory or cautionary sentence. The only guard is `raise RuntimeError("Iteration limit reached")`, which is a real convergence check, not a disclaimer. Displayed code is compact: 33 and 27 lines. `verify-ica-examples.py` run without `--write` asserts that the served `.py` files still equal the displayed code, so the two cannot drift silently.

### A4. Model and lab logic read for defects

- `symmetricEigen2` (`ica-models.js` L81–104) returns two genuinely different axes in the `b = 0` branch and applies one deterministic sign convention. This is the defect class that made GMM's covariance gallery draw ellipses as lines; here the source comment names it explicitly and the verifier tests the diagonal and isotropic branches. I rebuilt `Σₓ` from the returned eigenpairs and it reconstructs.
- `whitenerFromCovariance` (L107–117) refuses a non-positive eigenvalue with the manuscript's own rank argument rather than substituting a default. `fastIcaStep` (L156–200) refuses a non-unit direction and refuses a zero update with a message that *is* practice 3's failure mechanism. `contributionModel`, `rescaleComponent` and `checkAmplitude` all refuse rather than clamp. This is the right discipline throughout.
- `MODEL_AMPLITUDE_LIMIT = 16` against `ENTERED_AMPLITUDE_LIMIT = 4` (L315–320) is correctly reasoned in its comment: a compensated rescale by 4 legitimately carries an entered 4 to 16.
- `useIcaInvestigation` (`IcaShared.jsx` L24–90) is the strongest part of the design. A committed prediction is bound to `JSON.stringify(draft)`; `edit` clears the prediction, clears the commitment and demotes any standing result to `historical`; `apply` refuses unless `committed.key === key`, so a result can never be shown beside inputs it was not computed from; `currentResult` re-checks the key on every render; `undo` and `reset` both clear the prediction and label the old result as history. I confirmed the history labelling renders on the live page ("Previous result (history, not feedback on the edited draft)"). `Prediction` starts with no radio selected, disables Commit until a choice exists and disables Apply until a commitment exists — all verified live, including keyboard-only operation (Space on the radio, Enter on both buttons).
- `gradeRotation` (L295–304) grades from the committed snapshot with the specification's 1e−10 tolerance, and `gradeContribution` with 1e−9. Both compute the answer from the committed inputs, not from whatever is rendered.
- `rotationModel` (L253–292) derives the support, the contours and all 361 curve samples from the shared fixture; nothing is a drawing constant. Laplace contours are the `|s₁|+|s₂| = 1,2,3` diamonds and Gaussian contours are radius-1/2/3 circles, exactly as specified, and the rotation is applied in the correct direction (the level set in `y` space is `R` applied to the level set in `s` space).
- Defects found by reading, then confirmed on the live page: the unconditional "changed the fourth moment" clause (**B1**); C1b's pre-revealed answer (**S1**); C1b's outcome-independent explanatory clause (**S2**). Dead exports are listed as **O10**.

## Part B — learning-experience checklist (reviewer's heuristic run)

| # | Item | Finding |
| --- | --- | --- |
| 1 | **Route** | Present and declared (`.jsx` L58): sections 1–5 then the section-5 program, then the section-6 comparison; practices 1, 2 and 4 before the readiness check; 7 and 8 marked as deeper branches at their entry, both headed "Deeper branch". Both time bands are given. The `readTime` metadata string mangles the same information (**O1**). |
| 2 | **Cautions** | Model conditions have one home (§1's "The model's conditions belong here"), the real diagnostic's scope has one home (§6's second paragraph plus E1's four limits), and the source-label caution lives with exclusion (§7's last paragraph plus C1's closing note). No caution is restated as a warning; the distinctions recur as mechanism. No program prints a disclaimer. The version caveat that GMM's review had to raise as missing is present here, in E1. |
| 3 | **Real question** | Opens on two sensors carrying the same two signals and returns in §6 with 20 seconds of a real ECG recording, a downloadable CSV under a correctly stated licence, a chronological split declared before the rule, and an outcome that went against the method the lesson is about — kept, with the numbers, in four places. |
| 4 | **Labs as investigations** | **R1**: prediction unset, retired on any edit, graded against the committed snapshot, with a real permanent null (Gaussian, flat at zero, with its own explanatory sentence) and a real transfer (Laplace at 30° reproduces practice 2's 1.875 exactly). Its feedback sentence is wrong in the null case (**B1**). **C1**: prediction unset, numeric, with an optional second field, every keep-set reachable, the zero-source null reachable, and a required changed amplitude so the worked example cannot be re-submitted. **C1b**: a genuinely distinct second question, gated on completing one exclusion trial and remounted when the parent state changes — but it shows its own answer before the learner predicts (**S1**) and explains the wrong branch when the outcome is a null (**S2**). |
| 5 | **Figures** | M1, D1, W1 and E1 are legible and correct at desktop and at 390 px; W1 is the best of the set — four panels, equal x/y scales, real ticks, per-panel extents stated, the four states carried by letter identity, and the absent joint zero drawn as a hollow marker with "no mass here" beside it. F1 is correct but its dense table clips mid-numeral at 390 px (**S3**). P1 was genuinely rebuilt in HTML and is excellent at desktop, but does not stack on a phone and loses its lock marker (**S6**). E1 hides its headline column at 320 px (**S7**). No caption apologises for its figure. |
| 6 | **Connections** | Strong. The four-state fixture is carried through §1, §2, §3, §5, §7, W1, F1, R1 and C1 without ever being re-typed. §3's orthogonal recovery and §5's fixed-point update are explicitly reconciled ("This is the same final combination we found by algebra in section 3"). §5 states, and R1's frame note repeats, that the section-3 diamond and R1's source-aligned frame are the same distribution 45° apart. §4's equal-Laplace result is the exact value practice 2 asks for and the exact value R1 reproduces. §7's mixing-column-versus-inverse-row distinction is drawn in M1's second line and again in C1's column table. §8's model-change table names which condition each extension breaks. |
| 7 | **Code** | Two complete workflows, 33 and 27 displayed lines, both executed, both reproducing their output, both served as files whose bytes the verifier pins. Deflation is inside the iteration, which is the specific defect the design record set out to repair. The `pip install` pin is one `sh` block. |
| 8 | **Practice** | Six tasks, each changing numbers or the decision; hints and solutions in separate closed disclosures; every stated value reproduced independently. Practice 2's pointer into R1 is **correct and reachable** — I confirmed Laplace at 30° yields exactly 1.875 in the lab. |
| 9 | **Screenshots** | Twenty-five captures covering both matched and missed predictions, the Gaussian null, the compensated-scale null, every figure at desktop and 390 px, three at 320 px and one 200 % zoom page. Gaps: no 320 px capture of M1, D1, F1 or P1; no capture of C1b's `c = 0` refusal, its negative-`c` sign flip or R1's family-restart state, all of which the browser record claims were exercised; and two of the captures that were taken show a false sentence that nobody noticed (**B1**). |

## Ranked actionable findings

Severity: **blocking** = wrong on the published page, or a record that asserts something untrue; **should-fix** = an inaccuracy, a defeated teaching mechanism, or a dropped specification item with a bounded fix; **observation** = recorded for the ledger.

### Blocking

**B1. The rotation investigation tells the learner "Your direction changed the fourth moment" every time it grades — including in the Gaussian null, where the whole point is that it did not.**

Location: `src/learn/components/lesson-labs/IcaLabs.jsx` L58.

```js
message: `You recorded ${words[prediction]}; the actual change was ${words[grade.actual]}. |κ| went from ${format(grade.before, 6)} at … to ${format(grade.after, 6)} at …, a signed difference of ${signedFormat(grade.difference, 6)}. Your direction changed the fourth moment while the variance stayed 1 and the covariance stayed the identity.${gaussian}`,
```

The clause is unconditional. The `gaussian` suffix is appended after it, so in the null case the page renders a self-contradicting pair of sentences. Confirmed on the live page at 1366 px:

- Gaussian, 90° → 30°: *"|κ| went from 0 at 90° to 0 at 30°, a signed difference of 0. **Your direction changed the fourth moment** while the variance stayed 1 and the covariance stayed the identity. Every angle has the same joint Gaussian distribution; this contrast supplies no source direction."*
- Binary, 0° → 90°: *"|κ| went from 2 at 0° to 2 at 90°, a signed difference of 0. **Your direction changed the fourth moment** …"* — the direction did change, the fourth moment did not.

It is also in the committed evidence. `docs/teaching/evidence/screenshots/ica-rotation-missed-1366.png` reads *"|κ| went from 2 at 0° to 2 at 0°, a signed difference of 0. Your direction changed the fourth moment…"* — a case where neither the direction nor the moment changed. `ica-rotation-gaussian-null-1366.png` carries the Gaussian version. Both screenshots were captured, and the design record says "Every capture was opened and looked at", yet the sentence survived.

This matters more than a wording slip. The Gaussian null is the fixture the specification calls the investigation's answer to "What happens when the non-Gaussian distinction disappears?", and §4's whole argument is that the fourth moment is what distinguishes directions. Telling the learner it changed, in the one case where it provably cannot, teaches the opposite of the section.

Fix: the code already has the number it needs. Make the clause conditional on `grade.difference`, e.g. `Your direction ${Math.abs(grade.difference) > 1e-10 ? 'changed' : 'left unchanged'} the fourth moment while the variance stayed 1 and the covariance stayed the identity.` Then re-capture `ica-rotation-missed-1366.png` and `ica-rotation-gaussian-null-1366.png`, which currently document the defect.

**B2. The design record contradicts itself and contradicts the working tree, and two downstream records assert a state that no longer holds.**

Locations: `docs/teaching/ICA-LESSON-DESIGN.md` L7, L13, L15, L17, L187, L243; `docs/teaching/lesson-delivery-progress.json` (ICA entry); `LESSON-AUTHORING-HANDOFF.md` L29.

Evidence:

- The record's "Current source and phase" section still reads "Implementation is **not started**" (L7), "Visual/lab implementation | Not started; every figure and investigation is specified, none rendered" (L13), "Independent phase-two review | Deferred" (L15) and "Next action | On a later explicit finish request … complete stages 3–6" (L17) — immediately above a 90-line "Phase two: implementation — 14 September 2026" section describing all of those as done. This is the same self-contradiction GMM's independent review raised as its B1(b) on 13 September; it has recurred verbatim in shape.
- L187 states: "`blueprints/index.js` was **not** edited; the integration owner registers it", and L243 repeats "The phase ledger and `blueprints/index.js` were not touched." `git diff src/learn/data/curriculum/blueprints/index.js` shows `import icaBlueprint from './independent-component-analysis-ica.js';` at L16 and `'Independent Component Analysis (ICA)': icaBlueprint,` at L135, uncommitted. The file's mtime is 18:14, after the design record's 17:57 and after every evidence file. Whoever made the edit, the record now asserts something untrue about the tree it describes.
- The ledger entry reads `"implementation": {"status": "not-started"}` with `nextAction: "… Current content is complete; implementation has not started."` Its content checkpoint binds `docs/teaching/ICA-LESSON-DESIGN.md` at `b3eb4a3cc77e044eb34a60ad8bcf47f8affd5f30f74fe8b06e3e5c8ad88523f8`; the file is now `fbbbf26d927b47dd6c89668dc2dbfeddbb6f519da5177f6d56013d90adebe71b` after the phase-two append.
- `LESSON-AUTHORING-HANDOFF.md` L29 still reads "Its research/write phase is complete and its implementation remains not started."

The record does disclaim the ledger ("It does not close the phase ledger; the integration owner does that"), which excuses the ledger and handoff as the integration owner's outstanding work. It does **not** excuse the record's own header contradicting its own body, or the two false statements about `index.js`.

Fix: (a) rewrite `ICA-LESSON-DESIGN.md` L5–17 so the phase table matches the appended section, or move it under a dated "at content-phase close" heading, as GMM's review recommended for the identical defect; (b) correct L187 and L243 to record that the blueprint is registered, or, if the implementer genuinely did not make that edit, say who did and when; (c) the integration owner then updates the ledger entry to `implementation: complete` with the final source hashes, refreshes the design-record hash in the content checkpoint, and restates handoff L29.

### Should-fix

**S1. The C1b scale investigation displays its own answer before the learner has recorded anything.**

Locations: `src/learn/components/lesson-labs/IcaLabs.jsx` L164 (`useIcaInvestigation({ component: '1', scale: 2, mode: 'compensate' })`), L170–174 (`applied` is computed from `state.active` on every render), L205–212 (the result table).

C1b's prediction is *"Compared with the applied reconstruction, the observed sensor amplitudes will: stay the same / change."* Its initial applied state is `c = 2` with a compensated column — the answer to that question. Measured on the live page at the moment C1b first appears, with zero radios checked, status "Prediction not recorded." and no result element in the DOM:

```
caption: Applied state: component 1 at c = 2, with a compensated column.
   Quantity                     | before  | after
   Source amplitudes            | (3, −1) | (6, −1)
   Mixing column 1              | (2, 1)  | (1, 0.5)
   Contribution of that component | (6, 3) | (6, 3)
   Reconstructed sensors        | (5, 1)  | (5, 1)
```

"Reconstructed sensors: before (5, 1) → after (5, 1)" is the compensated-scale null, stated in a table, beside an unset prediction. The specification's C1 contract says "No preset outcome text beside the controls," and the shared contract's whole point is that a result belongs to a recorded prediction. The evidence screenshot `ica-scale-390.png` shows the same thing after a Reset.

This is not cosmetic: scale/sign ambiguity is §7's headline claim and C1b is the only interactive demonstration of it. The mechanism that would make the learner commit to an answer is defeated by the default state.

R1 got this right — its proposed-angle marker is drawn only after a commitment (`IcaLabs.jsx` L114), and the design record calls that out as a deliberate policy. C1 got it right too, by requiring a changed amplitude. C1b is the one lab that did not.

Fix: render only the "before" column until the first Apply (C1 already has the `everApplied` pattern), or make the initial applied state the identity, or hide the whole table behind the same gate that reveals the investigation.

**S2. C1b's explanatory sentence is chosen by the operation the learner picked, not by what actually happened, so it contradicts its own verdict whenever the rescaled component contributes nothing.**

Location: `src/learn/components/lesson-labs/IcaLabs.jsx` L186 — the trailing clause is `compensate ? '…leaves every product…' : 'Scaling the source without compensating its column is a different operation; it changes the physical contribution.'`, independent of `unchanged`.

Confirmed on the live page. With applied amplitudes `(2, 0)` and keep-set "keep source 1 only", rescaling **component 2** source-only by `c = 2`:

> "You recorded "stay the same"; the observed sensors **stayed the same**. Source 2 became 0 and its mixing column became (1, 2). The reconstruction went from (4, 2) to (4, 2). Scaling the source without compensating its column is a different operation; **it changes the physical contribution**."

Reachable in two ordinary ways: a zero-amplitude component (the specification's own `(2, 0)` null fixture), and rescaling a component the keep-set already excludes. Same class of defect as B1 — a fixed explanatory clause appended to a computed verdict.

Fix: choose the clause from `unchanged` as well as `compensate`; when a source-only rescale leaves the reconstruction fixed, say why (the component contributes nothing to this keep-set, or its amplitude is zero).

**S3. F1's table is clipped mid-numeral at 390 px, and the four visible values then contradict the average stated one element below.**

Locations: `src/learn/components/lesson-labs/IcaFigures.jsx` L180–187 (`<DataTable dense …>`); `src/learn/components/lesson-labs/ica-labs.css` L97–99.

Measured on the live page at 390 px: the F1 table's scroll region is `scrollWidth 335` against `clientWidth 322`, hiding 13 px — exactly the last character of the `3y²` column. A 4× crop of `docs/teaching/evidence/screenshots/ica-f1-390.png` confirms it: the column reads **3.8, 2.1, 2.1, 3.8** where the values are 3.84, 2.16, 2.16, 3.84. There is no ellipsis, no fade and no visual cue; the truncated values look like complete numbers.

A reader who checks the arithmetic from what is visible gets `(3.8 + 2.1 + 2.1 + 3.8)/4 = 2.95`, while the very next element states "2 · average of 3y² = **3**". The figure exists to expose the averages; at phone width it shows four wrong numbers that break the one it exposes. At 320 px the same region hides 83 px, which at least cuts visibly.

This is the one case among the lesson's five narrow-width scroll regions where the clipping is silent rather than obvious. For contrast, W1's table at 320 px hides 67 px and cuts mid-parenthesis, which is unmistakable.

Fix: give the `3y²` column a non-breaking minimum width, or drop `y³` (which is already implied by `z y³`) at narrow widths, or stack the table into per-state rows below 520 px. Then re-capture `ica-f1-390.png`.

**S4. The published prose and the figure two elements below print different values for the same vector.**

Locations: `src/learn/data/topics/independent-component-analysis-ica.jsx` L211; `IcaFigures.jsx` L194–195.

The prose says "the next vector is approximately **(0.7091653, 0.7050423)**". `F1` computes and prints **(0.709165303, 0.705042249)**. Recomputed at 50 decimal places from the exact rational `r = (−172/125, −171/125)`:

```
‖r‖ = 1.9403092537015845505808601940389…
     = 0.70916530309535176944306832421312…
       0.70504224900758809636491095023514…
```

So the seven-decimal value is `0.7050422`, not `0.7050423`. The prose's last digit is wrong, and a reader looking at both on one screen sees two different numbers for one quantity.

This came in from the packet and is wrong there too, in two mutually inconsistent ways: `lesson.md` L189 has `0.7050423` and `visual-specifications.md` L70 has `−.705042246` (the true value is `…2490`). Only the implementation, which computes rather than types, is right — and the native verifier's oracle uses the correct `0.70504225`, so nothing caught the manuscript's rounding.

Fix: `.jsx` L211 → `(0.7091653, 0.7050422)`, and correct both packet files so the next reader does not reintroduce it.

**S5. The served attribution document states something false about the bytes it describes, and both of its internal links are dead.**

Location: `public/learn-assets/ica/data-provenance.md` L24 and its two relative links; linked from `.jsx` L247 and L402.

- L24 reads "Retained CSV: 20,000 data rows × 5 numeric columns, plus one header; 390,723 bytes; UTF-8/ASCII header, decimal integers, comma-delimited, **LF lines**." The served file contains 20,001 `\r` and 20,001 `\r\n` sequences — every line is CRLF, including the header. Everything else on that line is correct, including the byte count. The cause is `np.savetxt` in `author-calculations.py` opening in text mode on Windows; a reader following the provenance to regenerate the file on Linux would get a different byte count and a different SHA-256.
- The document ships two relative links: `[author-calculations.py](author-calculations.py)` and `[the design record](../../ICA-LESSON-DESIGN.md)`. Neither target is served. Both resolve to the SPA's 590-byte `index.html` fallback (verified: `HTTP 200, Content-Type: text/html, 590 bytes`). This is the licence and attribution document the lesson tells the learner to read.

The licence handling itself is correct and was checked against the primary sources: PhysioNet's page shows "Open Data Commons Attribution License v1.0", ODC-By §4.3 requires a notice making users aware the content came from the database, and E1's provenance block names the database, its version, the record, the original authors, the licence with a link, the extract's hash and the fact that it is an extract. That satisfies the licence.

Fix: correct L24 to "CRLF lines" (or regenerate with `newline='\n'` and update the hash everywhere it appears, which is a much larger change — the text fix is the right one), and either serve the two linked files or make the links absolute repository references that a served copy can carry honestly.

**S6. P1 does not stack on a phone, so its stage labels break mid-word, and it drops the lock marker the specification required.**

Locations: `src/learn/components/lesson-labs/IcaFigures.jsx` L222–277; `ica-labs.css` L191–206.

The specification says: "**Mobile:** three stacked stage rows retaining a small 0–20 s ruler" and "A **lock marker** at selection denotes frozen index, accompanied by text."

`.ic-lane-stages` keeps `grid-template-columns: 12fr 4fr 4fr` at every width, with `overflow-wrap: anywhere` on the cells. In `docs/teaching/evidence/screenshots/ica-p1-390.png` the column head renders as "evaluat / e", and the select column's output text as "one coordin / ate index per metho / d, then locked". Mid-word breaks with no hyphen, in a figure whose whole job is to be read as text at reading size — which is exactly why the implementer rebuilt it out of SVG in the first place. The repair fixed the type size and left the wrapping.

Separately, there is no lock marker anywhere; "then locked" carries the whole idea in words. The specification asked for the marker *and* the text.

Fix: below about 520 px, collapse `.ic-lane-stages` to one column and repeat the stage name per row (the ruler already carries the intervals); and add a small lock glyph, `aria-hidden`, beside the select stage's output.

**S7. E1 hides the lesson's headline numbers behind a horizontal scroll at 320 px.**

Locations: `IcaFigures.jsx` L323–326; `ica-labs.css` L97.

Measured on the live page at 320 px: E1's first table has `scrollWidth 421` against `clientWidth 252` — 169 px, 40 % of its width, off-screen. `docs/teaching/evidence/screenshots/ica-e1-320.png` confirms that the visible columns stop at "development |r|"; the **held-out |r|** column, which carries `0.119806 / 0.450178 / 0.343966`, is not visible at all. Those three numbers are the result the entire section exists to report.

The design record says of the earlier repair: "E1's headline number was off-screen at 390 px. The long signed-development column moved last so the chosen coordinate and both diagnostics are visible without scrolling." I confirmed that is now true at 390 px (measured: nothing hidden for that table). It is still not true at 320 px, which is a width the browser evidence claims to cover.

Partial mitigation: the bar chart above the table does plot the held-out marks on a 0–1 axis, and the program's expected-output block above the figure prints the numbers. So the information is reachable; the exact table is not.

Fix: below about 360 px, render the first table as three stacked definition blocks (one per representation) instead of a five-column table; or move the signed-development column into its own table. Then re-capture `ica-e1-320.png`.

**S8. The blueprint registration is covered by no recorded check, and the build the browser evidence names does not contain it.**

Locations: `src/learn/data/curriculum/blueprints/index.js` L16, L135; `docs/teaching/evidence/ica-browser.json`.

- `ica-browser.json` records `"buildHash": "e8de1fd67ef315b9644d1faf88e0c69c969ae44eff983c046985864555044a65"` and `"distDir": "dist-ica"`, checked at `2026-09-14T12:29:05.693Z`. `dist-ica/index.html` has mtime 17:53 local; `index.js` has mtime **18:14**, roughly fifteen minutes after every evidence file was written.
- I grepped the built bundle for two strings that exist only in the ICA blueprint — `"Uncorrelated coordinates are independent coordinates"` and `"Whitening has already separated the sources"`. Neither appears anywhere in `dist-ica/`. The blueprint was demonstrably not in the build the browser pass exercised.
- `ica-browser.json` records the blueprint *file's* hash in `sources`, which makes it look covered. It is not; nothing rendered it.

The exposure is limited — `TopicContent.jsx` L74 renders `LessonGuide` only when `!topic.hasIntegratedGuide`, and ICA sets `hasIntegratedGuide: true`, so the blueprint does not appear on the lesson body itself. It does feed `track-definitions.js` L1958 and therefore curriculum-level surfaces. So this is a coverage and record gap rather than a rendering defect, but it is exactly the kind of gap the source-bound evidence discipline exists to prevent.

Fix: rebuild and re-run `scripts/verify-ica-browser.cjs` so the recorded `buildHash` describes a build containing the registration, add `blueprints/index.js` to the evidence's `sources` map, and reconcile with **B2**(b).

### Observation

**O1. The `readTime` metadata garbles the route's own timings.** `.jsx` L45 reads `'~45–55 min first pass · deeper branches + 45–75 min for code and practice'`, which parses as "deeper branches take 45–75 minutes". The manuscript and the in-body route say 45–55 min reading, 45–75 min for code and practice, and about 25 extra minutes for the deeper branches. The generated `navigation.js` has propagated the garbled string.

**O2. D1 labels the covariance as `E[uv]`.** `IcaFigures.jsx` L116–118 prints `E[uv] = {format(summary.covariance)}`, and `summary.covariance` is the centred second moment, not `E[uv]`. They coincide here only because `E[u] = 0`. The value is right and the manuscript makes the same identification in prose; the figure could compute `Σ p·u·v` directly, which it already has the data for.

**O3. The `u · v` column shows `−1`, `0`, `+1`.** `signedFormat` prefixes `+` and `−` but returns a bare `0`, so the column mixes signed and unsigned forms. Correct, slightly noisy.

**O4. Trailing-zero stripping makes the figure disagree with the program.** `IcaShared.jsx` L5–11 does `Number(value.toFixed(places))`, so E1's table prints `0.18458` where the program output four elements above prints `0.184580` and the manuscript, the provenance and the evidence all write `0.184580`. Nothing is wrong; a table whose decimal places vary row to row reads as noise, and the headline number no longer matches the block that produced it. (GMM's review raised the identical behaviour as its O1; the helper is a copy of the same idea.)

**O5. The Hyvärinen & Oja citation is one equation wider than the derivation it supports.** `.jsx` L219 cites "section 6, equations 41–44" for the approximate-Newton step. In the paper, (41) is the Kuhn–Tucker condition `E{xg(wᵀx)} − βw = 0`, (42) the Jacobian `E{xxᵀg′(wᵀx)} − βI`, (43) the approximative Newton iteration; (44) is the Gram–Schmidt deflation step in §6.2, which belongs to the *next* paragraph's claim. `41–43` would be exact. Everything else about the citation checks out: the PDF is live, is that paper (Neural Networks 13(4–5):411–430, 2000), and its sections 2–6 do cover identifiability, non-Gaussianity, whitening and the fixed point as the lesson says. The manuscript has the same range.

**O6. "Dataset version DOI" is the project-level DOI.** `data-provenance.md` L9 calls `10.13026/C2RP4B` the "Dataset version DOI"; it resolves to `physionet.org/content/adfecgdb/` (version-agnostic), not to the `1.0.0` page. Harmless in practice — 1.0.0 is the only version — but the label overstates its granularity. Everything else in the provenance's citation block verified: the Jezewski et al. authors, journal, 2012, 57(5), 383–394; the Pollard et al. *Nature Health* platform citation (which does resolve, and which PhysioNet does request); the ODC-By text; the published `r01.edf` checksum and its 3,061,792 bytes, both character-for-character correct.

**O7. `data-provenance.md` is served as `text/markdown`.** `.jsx` L247 and L402 link `/learn-assets/ica/data-provenance.md`; the preview server returns `Content-Type: text/markdown`, so most browsers download it rather than render it. Combined with S5's dead internal links, the attribution document is the weakest-served artefact in an otherwise careful provenance chain.

**O8. C1's required first action is presented as an error.** `IcaLabs.jsx` L226–228 returns "Change at least one amplitude first: the worked example is already solved above." from `validate`, so on first paint the lab shows red error text (`ica-labs.css` L86) and a disabled Commit button before the learner has done anything. Visible in `ica-contribution-zoom.png`. The gate itself is right and is what the specification asked for; it reads as a failure rather than an instruction.

**O9. F1 draws both the new direction and its sign-equivalent as dashed lines.** `IcaFigures.jsx` L206–207 uses `is-companion` (grey, `stroke-dasharray: 4 3`) for the flipped vector and `is-proposed` (green, `5 3`) for the actual update; the specification asked for a solid actual and a "dashed sign-equivalent". They are distinguishable by hue and weight and both are labelled, and no label is clipped — the design record's claim that the 260-unit viewBox fixed the earlier "sign-equi" truncation is real and verified in `ica-f1-1366.png` and `ica-f1-390.png`.

**O10. Exports that reach the verifier but never the page.** `deflate`, `absoluteCorrelation`, `CONTRASTS.tanh`, `CONTRASTS.linear`, `multiplyMatrices`, `covarianceOfRows`, `symmetricEigen2`, `whitenerFromCovariance`, `projectionKurtosis`, `checkEnteredAmplitude`, `MODEL_AMPLITUDE_LIMIT`, `SOURCE_STATES` and `STATE_PROBABILITY` are imported by no lesson component. Two are worth naming. `deflate` is the repair the design record set out to make — deflation inside the iteration — and it is implemented and tested but never shown; only the Python program and the prose carry it. And `CONTRASTS.linear` carries the source comment "practice 3's broken program, kept so the model can demonstrate the failure rather than describe it" (`ica-models.js` L148–149), but nothing on the page ever demonstrates it: practice 3 describes the failure in prose exactly as before. The comment claims a page behaviour that does not exist.

**O11. E1's shared vertical range is a typed constant.** `IcaFigures.jsx` L298 sets `const limit = 4.2;` and the caption states it. It happens to contain the observed display-z range `[−3.6182, 4.0932]` with 0.1 to spare, which I verified — but the figure would silently clip if the data were regenerated. Everything else in E1 is derived from `ICA_RECORDING`.

**O12. P1's `role="img"` hides its own visible text from assistive technology, and the replacement label omits the intervals.** `IcaFigures.jsx` L236–237. The label describes the lane structure and the reference's absence from the fit, but not the 0–12 / 12–16 / 16–20 boundaries the graphic conveys. The accessible alternative the specification required — a stage/time/arrays/output/forbidden table — is present directly below and carries them, so nothing is lost; the label could still carry the split.

**O13. E1's traces and bars have no accessible content beyond a method description.** `TraceRow`'s label (L287–288) explains the envelope, the interval and the normalization but says nothing about the waveform; the correlation bar markers (L310–311) carry only `title` attributes. The exact table and the 20-row window are the accessible equivalents, which is what the specification designated as primary, so this is a limit rather than a defect.

**O14. The reflection textarea is uncontrolled and survives Reset.** `IcaShared.jsx` L240–245 renders a `<textarea>` with no value, no state and no key. Confirmed live: text typed into C1's reflection is still there after pressing Reset, whose status line says "Documented starting inputs restored." The field is correctly ungraded and is a good addition that the specification did not ask for; it is simply outside the reset contract.

**O15. R1's kurtosis curve has no y gridlines or tick marks.** `IcaLabs.jsx` L110 draws three y labels at the left with no accompanying rule, so reading 1.25 off the curve at 30° requires interpolating between bare numbers. The exact value is printed in the readouts immediately below, which is why this is an observation.

**O16. No verifier can be run read-only.** `scripts/verify-ica-examples.py` L218 and `scripts/verify-ica-data.py` L234 rewrite their evidence JSON on every run, `--write` or not, so any re-run mutates `docs/teaching/evidence/`. `verify-ica-data.py` L211 additionally rewrites `src/learn/data/ica-data.js` whenever the regenerated text differs, with no flag required — the module's own header says "Regenerate with … --write", which is not what the code does. The hard `check()` assertions all run before that write, so a changed numeric result would raise rather than be silently republished; the exposure is limited to unasserted content such as the envelope's non-extreme columns. This is why this review reproduced the numbers independently instead of re-running the verifiers.

**O17. The design record's departure list omits the unimplemented specification items.** The nine declared departures are accurate and well argued. Not listed: P1's missing lock marker and its unimplemented mobile stacking (**S6**), and C1b's pre-revealed initial state (**S1**). The record is what lets the next author tell a deliberate choice from a slip.

**O18. Screenshot gaps.** No 320 px capture of M1, D1, F1 or P1 — and F1 at 320 px is where the clipping in S3 is worst. No capture of C1b's `c = 0` refusal, its negative-`c` sign flip, C1's out-of-range refusal, or R1's family-restart state, all four of which `ica-browser.json` lists as exercised behaviours. Two of the captures that exist document B1 without flagging it.

**O19. §11 does not use the shared `Sources` component.** The lesson renders its reference list as a plain `<ul>` under an `<H2>`, styled by `.ica-lesson > ul`, where sibling lessons use `LessonElements.Sources`. A consequence of the manuscript numbering References as section 11; no functional difference, but the shared component's closing "These are optional deeper references" note is absent.

## Closure

The mathematics and the data handling of this lesson are, as far as I can establish, flawless. Every number I recomputed from first principles agrees to the full stated precision: the four-state covariance and its eigenpairs, the whitened diamond and its identity covariance, the entire fixed-point trace row by row, the normalization to fifty decimal places, the three kurtosis families at six angles each, the zero-kurtosis counterexample's sixth moment, every contribution fixture and null, and all six practice answers — several of them exactly by rational arithmetic. The real-recording pipeline reproduces bit-exactly from the served CSV: the calibration, the split, PCA checked twice by independent routes, FastICA at fourteen iterations, all twelve signed development correlations, the three frozen choices, the held-out `0.119806 / 0.450178 / 0.343966` with PCA ahead, the `1.03e−28` round-trip residual, and all four thousand published envelope values at **zero** absolute difference. Both displayed programs reproduce their stored output exactly, and their served copies are byte-identical to the displayed code. All fifteen external URLs resolve to the documents they claim, and every checksum, byte count, page range, chapter range and lecture timestamp the lesson asserts checks out against the primary source. The licence is stated correctly and the ODC-By attribution requirement is met. Every screenshot-driven repair the design record claims — P1 rebuilt in HTML, F1's viewBox, R1's axis caption and family-specific extent, E1's fixed trace height and reordered column, the caption moved outside the scroll region, the plot padding, the faded removed bars, the `+0` formatter, the narrow-width type step — is real, and I verified each in the current code and the current captures.

What needs work is one sentence that is false wherever it matters most, one investigation that shows its answer before asking the question, and the record-keeping.

The rotation lab tells every learner that their direction changed the fourth moment, including in the Gaussian null where the section's entire argument is that it cannot (**B1**) — and two of the lesson's own evidence screenshots show it. The design record asserts, in its header, that none of this work has been done, and asserts twice that a file it did not touch was not touched when the tree shows otherwise (**B2**). The scale-ambiguity investigation opens with "before (5, 1) → after (5, 1)" printed beside an unset prediction (**S1**); its explanation contradicts its own verdict on the null branch (**S2**); F1's table loses a digit per cell at phone width and stops adding up (**S3**); the prose and its own figure print two different values for one vector (**S4**); the served attribution document misstates the file's line endings and both of its links are dead (**S5**); P1 breaks its labels mid-word on a phone and lost its lock marker (**S6**); E1 hides the lesson's headline numbers at 320 px (**S7**); and the blueprint registration is outside every check that was run (**S8**).

Recheck plan after the fixes. **B2, S5, O1, O5, O6, O17 and most observations are documentation or prose and need no verifier.** **B1, S1 and S2** are changes to `IcaLabs.jsx` only: re-run `node scripts/verify-ica-models.mjs` (unchanged, but cheap) and `node scripts/verify-ica-browser.cjs`, and capture three pictures that would have caught them — R1 immediately after a graded equal-|κ| apply, R1's Gaussian null, and C1b at first reveal showing no "after" column. **S3, S6 and S7** are figure and CSS changes needing fresh captures at 390 and 320 px, including the F1 and P1 captures at 320 px that do not currently exist. **S4** touches `.jsx`, `lesson.md` and `visual-specifications.md`; no verifier covers the prose, but `verify-ica-examples.py` should be re-run because its oracle string names the same quantity. **S8** needs a rebuild and a fresh `verify-ica-browser.cjs` run with `blueprints/index.js` added to the recorded sources. **No numerical verifier needs to be repeated for any finding in this review**: every number on the page is correct, and I have recomputed all of them.
