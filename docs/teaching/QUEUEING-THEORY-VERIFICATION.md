# Queueing Theory — author verification

Author-reviewed implementation of mathematics position 35, **Queueing Theory (M/M/1, M/G/1, Little's Law)**. Stable ID `queueing-theory-m-m-1-m-g-1-little-s-law` and existing legacy `littles-law.jsx` publication filename are preserved. This record does not assert parent integration, independent review or user acceptance.

The final source identity, full numerical/browser payloads and hashes of twelve actually opened final screenshots are in [the durable author record](evidence/queueing-author-review.json). The [design and research ledger](QUEUEING-THEORY-LESSON-DESIGN.md) records retained scope, assumptions, representation choices and source review bounds.

## What changed and what was conserved

The original seven-section formula-oriented body was read in full and archived with its exact source and four code/output blocks in [queueing-original-content.json](evidence/queueing-original-content.json). Both original programs and stdout are byte-conserved in the new ten-program collection. Original units, M/M/1 means, Little's Law, the M/G/1 variance calculation, multiple workers, retries/batches/downstream limits, measurement advice and the 90/100 practice scenario remain and have been developed with explanations or answered practice. The incorrect universal one-server stability wording is now qualified by model; finite-buffer and deterministic critical-load counterexamples explain why that matters.

The new route begins with an actual FCFS trace and exact occupancy areas. It derives stationary geometric probabilities, PASTA's sampling role, queue versus total-time tails and the zero-wait atom, length-biased residual service and P–K, pooled versus separately routed capacity, and finite-buffer admission. Heavy-tail truncation and independent repeated vacations are deeper branches with their own assumptions. Ten changed tasks have hints and explained solutions; two early checkpoints test concepts before advanced formulas. All ten programs have visible prediction questions and setup appears before the first program.

Six investigations address different questions: finite occupancy/censoring, M/M/1 stationary load, wait/total survival, service inspection/residual work, pooling, and finite admission. Three inline figures introduce the complete job timeline, birth/death count flow and stranded-versus-pooled capacity. On mobile the entire four-job timeline and occupancy axis remain visible, and the two residual triangles stack with a common scale. Larger probability plots and tables retain local scrolling with keyboard access; the page itself does not overflow. These counts are descriptive, not a fixed authoring quota.

## Native and independent numerical checks

Commands executed from the repository:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 scripts/generate-queueing-examples.py
node scripts/verify-queueing-models.mjs
scratch/lesson-tools/Scripts/python.exe -X utf8 scripts/verify-queueing-native.py
```

The final native result at **2026-09-10T19:47:40.536552+00:00** is saved in `scratch/queueing-verification/native-results.json` and embedded durably in the author record. Python 3.12.14 executed the actual displayed programs. Checks passed:

| Independent formulation | Actual covered cases |
| --- | ---: |
| Complete programs compared to displayed stdout | 10 |
| Priority-event dispatcher versus JS and native FCFS recurrence, including empty/tied/zero-service traces | 244 |
| Exact Fraction midpoint integration versus clipped residence, with both system and waiting boundaries | 2,431 windows |
| SciPy geometric law, Erlang-mixture survival with an explicit omitted-tail bound, and native quantiles | 96 stationary states |
| Residual triangle/continuous exponential quadrature versus model moments | 20 states |
| Changed rational discrete mixtures versus actual native helper | 24 |
| Erlang-B recurrence converted to Erlang C, versus pool model and native implementation | 240 |
| Finite generators, 90-digit Decimal probabilities and actual native helpers, including equal rates, overload and extreme supported ratios | 148 |
| Capped Pareto moments from direct survival integrals | 14 integrals |
| Changed repeated-vacation residual integrations and rational congestion terms | 18 |
| Invalid native/model inputs explicitly rejected | 14 native / 18 model |
| Original code/output blocks independently conserved against archive | 4 |

The browser trace model rejects a positive service duration that is lost at its chosen floating-point time origin; the displayed Python trace has the matching guard. Pending residence is summed directly rather than subtracting two nearly equal totals. The vacation helper rejects a nonfinite computed result. Supported positive rates, capacities and trace bounds are explicit in the models. These finite checks do not certify every possible floating-point input or arbitrary production queue.

The simulation uses eight independent seeded runs, each with 5,000 warm-up arrivals and 50,000 complete measured arrivals. Its reported mean is 0.500580 s and estimated Monte Carlo standard error 0.005147 s against stationary M/M/1 mean 0.5 s. This is a reproducible finite experiment with correlated customers within a run, not a proof that warm-up eliminates transient bias, a production benchmark or an empirical validation of exponential service.

## Actual browser and visual verification

```text
node scripts/review-queueing-lesson.cjs
node scripts/review-queueing-reading.cjs
```

Headless Edge used the local Vite route and the website's normal public font assets at **1440, 390 and 320px**, with reduced motion and settled instant scrolling for ordinary-reading captures. The complete interaction payload is `scratch/queueing-browser/results.json`; the final focused reading/static-geometry payload is `scratch/queueing-reading-final/results.json`. Both are embedded in the durable record.

At every width the full suite passed 68 changed model/control states, ten anchors, ten complete rendered code/output pairs with their preceding questions, twelve checkpoints/practice sections with functioning explanations, all 13 KaTeX expressions, every lab control's focus, keyboard slider/reset behavior and horizontal scrolling where needed. It verified stable/unstable load states, queue quantiles inside the zero atom, all service presets, pooled capacity failures, finite K=1 and overload, selected job rows and resets. No page/console/request errors or document overflow were reported. Ordinary paragraphs remain separate valid HTML paragraphs.

Initial browser passes found long equations at 390/320px. They were split into readable derivation lines rather than reduced to tiny text. Opened captures prompted compact aligned timelines, stacked residual triangles, a connected narrow birth/death row and explicit scroll guidance only where the larger axis may extend. The final focused pass also verified the singular `1 job` label after a grammar-only amendment; the full behavior result precedes only that label change, and the final reading result covers the final source.

Twelve **final** files were actually opened, beyond earlier iterative captures:

- `timeline-320.png`, `birth-death-390.png`, `censored-area-390.png`, `rare-long-320.png`;
- `reading-6-1440.png`, `reading-8-320.png`, `reading-10-390.png`, `sources-390.png`;
- `tail-derivation-320.png`, `pk-derivation-320.png`, `occupancy-equation-320.png`, `finite-buffer-320.png`.

All are under `scratch/queueing-reading-final/`; exact hashes are in the durable record. The review inspected actual geometry, reading flow, labels, units, table clipping/scrolling and equation legibility, not only selectors or screenshot creation. Other generated captures were not all opened and are not represented as individually visually reviewed. Focus rings visible in a few captures are deliberate evidence of keyboard focus.

## Source and learning-resource limits

Sigman's pathwise Little and renewal-reward notes, the MIT birth/death and Modiano M/G/1/vacation slides, Gallager's associated lecture notes, and Harchol-Balter's selected inspection/architecture slides were read as documented in the design ledger. The direct Gallager YouTube recording was verified through its official course entry; associated substantive notes were inspected, not the full video or transcript. Historical systems examples were not imported as current hardware measurements. References are annotated and supplement complete local teaching.

The [incoming Stochastic Processes note](topic-notes/queueing-theory-m-m-1-m-g-1-little-s-law.md) is resolved in this authored scope. The [outgoing LLM-serving note](topic-notes/queueing-theory-for-llm-serving.md) remains open with concrete reasons and corrected tail values for its future owner. No catalogue audit or downstream body rewrite was performed.

The formatting pass used normalized JS/JSX and CSS AST equality, preserving literal values and JSX text; its record is `scratch/queueing-authoring/format-conservation.json`. Later narrow wording/style changes are covered by the final rendered checks. Parent-owned registration, order, ledger, production build and integration remain outside this author record. User approval is still pending.

## Post-freeze numerical amendment — 2026-09-10T20:53:15.304113+00:00

The independent reviewer found an accepted simulation case where calendar timestamps erase a short service: arrival rate 10⁻⁶, service rate 10⁶, seed 17, 100,000 warm-up customers and 100 measured customers. The old reported mean total was 1.52587890625×10⁻⁷ although the same sampled services average 1.217625453856014×10⁻⁶ and every measured wait is zero. The original failure is preserved in the independent record and this author record’s amendment.

The displayed helper now computes remaining workload from the previous total duration and next interarrival gap, then adds service. This is the lesson’s existing Lindley recurrence, applied without a large common calendar origin. Random draw order is preserved. A visible paragraph explains why the arithmetic matters. All ten displayed outputs and the other nine example records are unchanged; the two original programs/outputs remain byte-conserved. Only the body and example module changed among the six production files.

`scripts/verify-queueing-simulation-precision.py` passed six changed rate/seed/cohort cases at 2026-09-10T20:47:23.213174+00:00. It independently uses exact Fraction arrival/start/finish calendar arithmetic on identical sampled floats. Both reported extreme cases now match exactly; ordinary, near-saturation, fast and slow cases also agree. The full native suite reran successfully at 2026-09-10T20:51:27.402492+00:00 with the same counts shown above. The independent reviewer’s distinct workload oracle also passed.

`scripts/review-queueing-simulation-precision.cjs` passed at 2026-09-10T20:51:16.083Z with actual project fonts at 1440/390/320. It checked actual current code, prediction question, complete output and the visible explanation, with no page/console errors or document overflow. The desktop program and the 390/320 explanation screenshots were actually opened. Two initial browser attempts used incorrect code-block selectors; they failed before assertions and are not counted as successful evidence. The successful check uses the actual rendered example container.

The complete earlier browser suite remains evidence for unchanged lab behavior; the focused final suite covers this narrow amendment. Exact prior/new production fingerprints, previous/final native evidence and the three opened images are retained in the durable author JSON. Final body SHA-256: `fb98824bffab7c7b316939d5e146e6f1b7c2c3cdc2effc52840123016d9f3101`; example SHA-256: `62d095639fa892b58ad50e203fd61db232567a05024dbfec83014c5e31ed2196`. Independent review, integration and user acceptance remain separate.
