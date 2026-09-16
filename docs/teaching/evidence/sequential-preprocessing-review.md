# Sequential preprocessing review — 16 September 2026

Scope: Classical ML position 20 only, `feature-scaling-encoding-imputation`. Reviewed the complete prepared lesson and visual specifications, design/disposition and prior independent review, current reader, four investigations, nine figures, shared controls, models and styles. Production changes and final execution belong to the integration owner; this reviewer changed only this record and the authorized browser regressions. No later topic was audited.

## Coverage reconciliation

No prepared section, substantive claim, program, figure, practice problem or investigation was omitted. The eleven sections, nine figures, four investigations, two complete programs and nine practices preserve the core/deeper split. Coverage includes units as metric weights; fitted standard/min–max/robust scaling and out-of-range values; constant features, row normalization and sparse-centering constraints; full/dropped/unknown categorical geometry; missingness mechanisms and indicators; overlap-aware donor eligibility and fallback; the frozen training boundary; the real 258/86 penguin experiment and all seven coordinates; nonlinear transforms and their domain/branch qualifications; cross-fitting with donor-only priors, signed hashing, and Rubin pooling of analysis estimates.

The reader retains the 38/67/84/85/85 correct counts, source row identity, the unchanged downloadable CSV, genuine measured versus edited/imputed states, and the warning that one extra correct row establishes no generally superior scaler. Code/output identity is preserved. Hints and solutions stay separately disclosed, and first-pass readiness does not require the deeper branches.

The interaction review checked committed versus edited state, result retirement, prediction gating, reset, ties and missing/unknown values. Revealed output tables and graphs follow committed inputs; ungraded calculation has explicit wording and cannot replace an existing grade. Input worksheets needed to calculate a prediction are intentional scaffolding. Figures keep identified observations, real coordinate scales where claimed, labelled schematic category geometry, the frozen bundle's permitted/forbidden directions, rank versus log contrasts, and comparable variance bars. Existing source-bound visual review is reused, with fresh browser geometry/paint checks owned by the integration owner.

## Findings and repairs

All five findings were repaired in the bounded production delta and independently inspected in source.

1. **P20-1 — I2 could reject every enterable answer to a repeating mean (P2).** `Predicted estimate` permits four decimals, while the old grader required error below `1e-9`. Set donor c values to `[0,0,1]` and k to 3: the true mean is 1/3, but entering 0.3333 necessarily failed. The repair advertises four-decimal rounding and accepts error at most 0.00005 with floating-point slack. Regression: 0.3333 passes, 0.3334 fails; a two-donor mean 0.00005 accepts both 0 and 0.0001 but rejects 0.0002. Donor-set correctness remains separately required.
2. **P20-2 — I3 supported measurements with no enterable correct prediction (P2).** Non-mass fields accept up to 300, but the prediction field stopped at ±20. Bill depth 300 produces approximately 145.92202532. The repair derives one fixed symmetric prediction bound from every permitted numeric endpoint and saved median. It covers all records/coordinates and does not change with the current answer. Regression checks both endpoints of all four input fields, including the large coordinate, and unchanged prediction limits.
3. **P20-3 — I1's translation preset bypassed its own measurement bounds (P2).** From reset, repeated +5 mm/+500 g clicks could push A above 7,000 g and later B above 80 mm, although typing those values was refused. Translation now disables before any next value would exceed either bound and explains how to continue. Regressions independently reach the mass and bill limits. The related zero-total bar denominator now safely falls back to one; the zero-width painted-bar check is defensive verification, not a separate claimed visual defect.
4. **P20-4 — Figure 1 converted a known category into missing data (P2).** In “the same row with one measurement absent,” only mass was absent and recorded sex still read female, but the downstream category box read `not_recorded`. It now correctly reads female / own indicator. The regression checks the absent-mass stage's retained observed category and rejects `not_recorded` there.
5. **P20-5 — inclusive numeric tolerance rejected its boundary (P2).** I3/I4 used strict `< 0.005`; I3 feedback described answers within 0.005 as acceptable. Both now advertise the tolerance before commitment and use a shared inclusive comparison with `4 * Number.EPSILON * max(1, abs(value), abs(expected))` slack. This covers subtraction dust without meaningfully widening the teaching tolerance. At I4, smoothing 0 and changing row 3's category to A yield exactly 0.5 for row 0: predictions 0.495 and 0.505 pass, while 0.4949 and 0.5051 fail. The integration owner separately evaluated the shared helper boundaries, including I3's use of that same helper.

## Evidence and limits

Fresh baseline: integration owner reports 27 model groups and 17 browser cases passing. Production rebuild after these repairs passed. The amended browser script adds one grouped regression and three captures; its syntax check passes. Final production browser `--write` passed **18 cases with 32 captures** after all repairs. The owner inspected the repaired Figure 1 and large-coordinate 145.922 feedback captures and confirmed both are readable and correct. All five findings are closed; position 20 is ready for checkpoint closure by the owner.

Native evidence is reused: current `scaling-examples.js` and its verifier match `scaling-native.json`, and this reviewer independently matched both displayed code hashes and both expected outputs exactly to the recorded executed programs (24 native oracles). Current model/data/CSV sources match the prior reviewed identity. The data evidence records the refit of all four preparations, 86 held-out transformations, practice cases and constructed fixtures against the supplied calculated inputs. It lacks a complete source-hash map, so it is not described as having one; identity is supported by the current data/CSV and prior review checkpoint, not by invented evidence fields. No fresh native refit is necessary for these UI-only repairs.

No new external factual claim or later-topic assessment was introduced. Browser assertions support interaction and computed geometry; they do not substitute for screenshot inspection or an assistive-technology session.

## Source identity

SHA-256 after the bounded repairs (paths relative to repository):

| Source | SHA-256 |
| --- | --- |
| `docs/teaching/drafts/feature-scaling-encoding-imputation/lesson.md` | `389fc4e93f958e0ca819a73ae36bb5e5fb09aa11ce73072572329b2eb07182da` |
| `docs/teaching/drafts/feature-scaling-encoding-imputation/visual-specifications.md` | `3729dfb734d3b4a929d7600f5c8db237b2b0e8543e665eb2dea1f9187cc9e91c` |
| `docs/teaching/drafts/feature-scaling-encoding-imputation/design.md` | `c7a4f15a216cc140408f559e4325ae3d8f6962bfddaf7ddcd26218e84cb209e9` |
| `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` | `22b7fc178c0664a7d87d881d42997f908470c67dffbd25eb3e62f5db8cc43dcb` |
| `src/learn/data/scaling-models.js` | `2016b0426dad0a3c76da8a04dd88ccee56a78c9f75bf81b029007552d56c9d2c` |
| `src/learn/data/scaling-data.js` | `1f0bc9e1ae81883aa0bccbd0ee3daa1140ce9f87b5847203f1da82efdbf93c04` |
| `src/learn/data/scaling-examples.js` | `0ef4de579fe7b3a38a15d10dbc8e6f8265081d6f13a82894cd5d91140aefd2ef` |
| `src/learn/components/lesson-labs/ScalingShared.jsx` | `1bf1c4ad6bced33d90ec3c6136ef03ef79311dccf0bce8f7949fcc0c445e8af3` |
| `src/learn/components/lesson-labs/ScalingLabs.jsx` | `9ae5d49c4eb0a73d720e19c2dda3dfec0bc76e41348acf2d9362bb46a03ba9be` |
| `src/learn/components/lesson-labs/ScalingFigures.jsx` | `c4ba8c21fb3d9f90c0f7e9db6d73db2844023138193ffacc5e4eba4b55cccb19` |
| `src/learn/components/lesson-labs/scaling-labs.css` | `aa9ad22ed790347b211c38f8690416dc91c02afc82e5fc686786abce835b6a22` |
| `scripts/verify-scaling-browser.cjs` | `d44b7b53042909d79474acb15d957e9a8fc395f90440c95765c6a5678a061d15` |
| `scripts/verify-scaling-examples.py` | `55ad5ebc51eaa9b08325ac27e7d73b96f507c55cbede14bafbe2b7be5ab31921` |
| `scripts/verify-scaling-data.py` | `1364b4e0d5232086b08a3dad0d599388111a6ce63493e622e31f246705b7bc3d` |
| `public/learn-assets/feature-scaling/penguins.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
