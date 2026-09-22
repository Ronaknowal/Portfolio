# Independent review: typed decision model project

Review date: 22 September 2026. Scope: the eight-stage project, executable Python reference, pure browser investigation model, source-excerpt mechanism, metadata, README, measured report and author handoff. This review does not complete the separately planned depth topic. It does not claim pretrained-model training, a Jev/Laya reproduction, a production service or browser verification.

## Status

**Closed for this content/native scope.** Scientific mechanism and reported experiment were independently checked. Both actionable source-contract issues were repaired by the author and independently rechecked below. No remaining finding was identified in this bounded review. The integration owner retains final build, loading, keyboard, responsive and visual checks.

## Reviewed sources

- `src/learn/data/projects/typed-decision-model/content.jsx`
- `src/learn/data/projects/typed-decision-model/metadata.js`
- `src/learn/data/projects/typed-decision-model/decision-model.js`
- `src/learn/data/projects/typed-decision-model/project.css`
- `public/learn-projects/typed-decision-model/typed_decision.py`
- `public/learn-projects/typed-decision-model/request.json`
- `public/learn-projects/typed-decision-model/verified-report.json`
- `public/learn-projects/typed-decision-model/README.md`
- `docs/teaching/projects/typed-decision-model.md`

The Python reference and every project stage were read. The review also loaded the author's existing 600-step checkpoint read-only from `scratch/typed-decision-project-check-long` to independently reconstruct results. It did not replace or retrain that artifact, alter the reported experiment, or install packages. Native checks used the existing lesson-tools Python, `-X utf8 -B`, PyTorch `2.14.0+cpu` and one thread. The final report is the durable public evidence; temporary weights are not a website dependency.

## Findings and disposition

### R1 — Floating-point equality violates the stated tie policy

Initial state: `expected_cost_decision([0.9, 0.1], 10, 1)` returned `act`, with estimated cost `0.9999999999999998`, although the lesson and README promise review on equal expected costs. This is a real boundary case in the taught threshold `p > 0.9`, not a demand to compare arbitrary floating-point quantities exactly.

Requested repair: adopt and explain a small numerical tie convention consistently in Python and the browser model; verify the nominal tie and meaningfully separated cases on both sides. Preserve the expected cost rather than silently altering the probability. Status: **closed**. Both implementations now act only when expected cost is below review cost minus `1e-12 * max(1, abs(actCost), abs(reviewCost))`; guide and README state the convention. Independent post-repair Python and JavaScript executions return review at `p = 0.9`, review at `p = 0.899999` and act at `p = 0.900001`. Expected costs are retained unchanged.

### R2 — Unsupported text can silently become no evidence

Initial state: the reference tokenizer uses `[a-z]+`. Numeric-only and non-Latin strings pass the nonempty-string schema checks but produce zero word tokens (`words('12345') == []`; `words('नमस्ते') == []`). The page did not state the ASCII English word scope, and “unknown words map to unknown” does not explain characters dropped before vocabulary lookup.

Requested repair: state the intentionally limited tokenizer contract in the guide/README and reject required textual fields/candidate descriptions with no supported word tokens. Keep application IDs separate: IDs need not become English model input. Document that punctuation, digits and unsupported scripts are not a general-purpose language interface. Status: **closed**. Define/serving stages and README state the toy tokenizer scope. Independent post-repair checks reject both numeric-only and non-Latin-only inputs in each of question, state and description (six cases). A valid request with a non-Latin application ID remains accepted, preserving the intended separation. Mixed inputs still retain only supported word runs, as documented; this is an intentionally limited fixture tokenizer, not multilingual support.

## Independent numerical evidence

Using the saved model, the reviewer computed logits and then calculated probabilities, log loss and multiclass Brier sum in scalar Python `math` arithmetic, without calling the reference `metrics` or loss implementation. The scalar log normalizer used max subtraction. Results agree with the public report within `2e-5` (which accommodates its float32 calculation and six-decimal output).

| Evidence | Accuracy | NLL | Brier sum |
| --- | --- | --- | --- |
| Test lexical baseline | 18/18 | 0.551444713932 | 0.269515343071 |
| Test model, raw | 12/18 | 2.047912000169 | 0.548923128574 |
| Test model, fitted temperature | 12/18 | 39.975477866691 | 0.657156710756 |
| Stress model, fitted temperature | 1/3 | 84.989664912543 | 1.332842398096 |

Independent scalar evaluation of all 121 log-grid temperatures selects `exp(-3) = 0.049787068367863944`, the lower boundary, matching the saved artifact. The calibration set has 18/18 correct raw classifications. This supports the guide's explanation of over-sharpening on easy calibration examples followed by severely worsened test probability loss. It does not demonstrate good deployment calibration or general language understanding.

Further bounded checks:

- Confirmed 36 training, 18 calibration, 18 test and three stress cases. All pairwise split sentence sets are disjoint. Templates still deliberately share task keywords; this is correctly disclosed rather than treated as group-independent real-world evidence.
- Compared scratch attention and SDPA on a new seed-41, float64, three-example fixture. Maximum forward-score error and maximum error across every parameter gradient were both `2.220446049250313e-16`. This checks the actual mask/projection path and gradients, beyond merely asking whether gradients are finite.
- Renamed only application IDs and proved encoded token sequences/marker positions are unchanged, supporting the proposed ID-versus-description exercise.
- Evaluated 162 combinations spanning three/four candidates, low/default/high temperatures, score extremes and cost endpoints in the browser's pure model. All returned probabilities were finite, bounded and summed to one within `1e-12`; entropy concentration remained in range within rounding tolerance. This is model arithmetic evidence, not proof that sliders move in the browser.
- Resolved every one of the eight displayed source-excerpt start/end markers against the canonical downloadable Python file. Each region is nonempty and in order. The source UI fetches the canonical file on expansion instead of maintaining copied variants; integration still needs to confirm the browser fetch/render path.

## Teaching and evidence assessment

The project clearly distinguishes a tiny original choice-model implementation from upstream commercial/open checkpoints. It teaches the shared option scorer, attention masks, candidate identity, stable cross-entropy, an ordinary SDPA library route, calibration and expected-cost action before optional research extensions. A full pretrained encoder, ordinal/boolean variants, policy-gradient objective and web service are correctly described as extensions rather than completed experiments.

The lexical baseline's victory and the calibration failure are visible in the guide and measured report. The 120-step pilot and 600-step follow-up are disclosed; the final fixture is an investigated teaching example, not a blind benchmark evaluation. Future model selection needs the separate development split described in the guide. Do not rebrand this observed sequence as a preregistered confirmatory experiment.

The live explorer is explicitly illustrative arithmetic and provides immediate output without a prediction gate. Candidate/temperature changes affect the distribution; cost changes affect the action. Source styles use scoped selectors, wrap token lanes and stack the two-column layout on narrow screens, with deliberate table/code scrolling. Those source properties reduce foreseeable layout risk but do not replace screenshots, keyboard operation, loading-error checks or actual rendered sizing by the integration owner.

The core meets its bounded educational purpose with both contract issues resolved. Its results justify learning and further investigation, not deployment of the example model.

## Source identity at closure

SHA-256 values below identify the reviewed sources after the two repairs. Later relevant edits require affected review, not automatic invalidation of unchanged mathematical evidence.

| Source relative to repository | SHA-256 |
| --- | --- |
| `src/learn/data/projects/typed-decision-model/content.jsx` | `9997f2394bb52358309d2c2254c4ea8fc9253d79c5fc201120d7e31bacc56c6d` |
| `src/learn/data/projects/typed-decision-model/metadata.js` | `e887a038b752a38eea892527ce9f1e05d57084501f95f84ca01689277794fd1c` |
| `src/learn/data/projects/typed-decision-model/decision-model.js` | `b5f050529f6fd0ae9c9becafa9a6074611792d76d735814a0cb14b2985094232` |
| `src/learn/data/projects/typed-decision-model/project.css` | `b6a8d152de9382400dffd7a243c151fa0556321c8ba2dadc5e7ca434a6099f30` |
| `public/learn-projects/typed-decision-model/typed_decision.py` | `edaf1c9e3c7b867a0e982e70b13970be615e180fce7693872ca6d373ad4d779b` |
| `public/learn-projects/typed-decision-model/request.json` | `2d0ae1b74e05a749434b2453a785276efb74915df45d1f16c7d6455023d88382` |
| `public/learn-projects/typed-decision-model/verified-report.json` | `057dd59310798378580f8d03b6706b7d9e9f822a5451d8969f8f2a514a0b0b80` |
| `public/learn-projects/typed-decision-model/README.md` | `3612fd6d4bedfe27fe995c30ad5beb5d43334f587963079493dc3c7d31a0a405` |

## Scoped metadata revision — project preview

Reviewed the added declarative `preview` on 22 September 2026. Current `metadata.js` SHA-256 is `d08244b77e2200d987f3885a0f7459bd9fb59f691c9283e8df4bb74f810b89c1`, superseding that file's original closure hash above. The remaining reviewed sources are unchanged.

The preview names all three conditioning inputs (state, question and candidate descriptions), the decision encoder/shared scorer and the probability-to-action flow. Its accessible description correctly attributes the act/review decision to an explicit cost policy rather than to the encoder alone. `p(y | x)` is appropriate compact notation for the distribution conditioned on the grouped inputs, not a claim of measured correctness or complete uncertainty calibration. The caption invites training and investigation without claiming a successful deployment or upstream model reproduction. No new content finding was identified. The rendered diagram should preserve the stated distinction between model probabilities and cost-policy action; actual layout and accessibility remain part of integration. No unchanged numerical checks were rerun for this metadata-only addition.
