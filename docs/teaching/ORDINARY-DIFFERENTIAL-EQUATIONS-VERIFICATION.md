# Ordinary Differential Equations & Linear Systems — author verification

Stable ID: `ordinary-differential-equations-linear-systems`, Mathematical & Statistical Foundations position 50. The lesson is implemented and author-verified. The exact **seven-file production freeze is 2026-09-11T05:23:15.161196+00:00**, recorded in [the durable source/evidence packet](evidence/ordinary-differential-equations-author-review.json). Independent mathematical review, integrated production checks and user acceptance remain separate; registration is not evidence of those outcomes.

## What was built and preserved

The [assessed design](ORDINARY-DIFFERENTIAL-EQUATIONS-LESSON-DESIGN.md) is realized as fourteen sections with a navigable first-pass route, four optional depth disclosures, thirteen complete Python programs, fourteen independent practice groups with optional hints and explained answers, and eighteen annotated source/alternate-resource links. There are **eight browser investigations and four inline explanatory figures**. These counts describe the actual choices; they are not an authoring quota.

The topic was planned at the scoped inventory check, with no old lesson body or executable program. The [exact original-plan archive](evidence/ordinary-differential-equations-original-plan.json) preserves that baseline. Cooling, scalar IVPs, coupled state, eigenvalue interpretation and Euler comparison remain, with complete reasoning and assessment added. No older program was removed and no nonexistent original runtime was claimed. The provisional [design arithmetic](evidence/ordinary-differential-equations-design-checks.json) is retained separately from the final production checks.

The learner starts with joules, watts and the minutes-to-seconds conversion in a thermal balance, then learns to read a direction field and distinguish a solution, an initial state and its time domain. Local existence/uniqueness hypotheses, a nonunique waiting family and finite-time blow-up prevent the visual intuition from implying universal solvability. Scalar separation and integrating factors lead into full position/velocity states, evolving basis columns, matrix exponentials, defective modes, initial versus forced response and the order of changing systems. The exponential series convergence and differentiation argument is local and self-contained; it does not assume the later Real Analysis lesson.

Numerical teaching includes actual Euler, midpoint and RK4 probes, a derived accumulated Euler error bound, equal final horizons with shortened last steps, sign/stability/accuracy distinctions, and executed SciPy stiffness and event workflows. The BVP branch changes the question explicitly: exact endpoint maps can produce one, many or no solutions. Optional Picard, Bernoulli, exact-equation and ordinary-point series branches provide their needed local definitions. Practice changes parameters, forcing schedules, initial states and structural conditions instead of merely repeating a worked calculation.

## Representation contracts and placement

| Hurdle and location | Representation and tested contract |
| --- | --- |
| Balance/units before the first ODE, §1 | Energy-flow figure distinguishes stored energy from power; minutes multiply the seconds-based rate by 60. The loss-area figure uses the actual analytical cooling loss rather than a decorative region. |
| Local rate versus whole solution, §2 | Field investigation changes cooling/logistic initial values and compares the corresponding analytical trajectory. Equilibria and axes are explicit. Tiny valid populations are checked numerically even though UI presets are bounded. |
| A formula's existence/uniqueness/domain, §3 | Waiting-time control exposes different solutions through the same initial point; a separate blow-up figure shows the excluded time and explains the curve's displayed sampling interval. |
| Position alone does not determine motion, §5 | Signed spring-force diagram precedes the state equation. Paired time and phase curves share an inspection instant, with actual velocity and energy-rate readouts. Damping does not imply a position coordinate decreases monotonically. |
| A solution operator acts on coordinates, §§6–7 | Fundamental-column investigation evolves both basis vectors and the selected initial combination. Matrix categories include stable, defective and transient cases; a category alone does not establish growth for every initial state. |
| Initial condition versus input timing, §8 | Input-schedule investigation separates initial, first-window and second-window contributions. Actual exponential integrals produce the readouts; this is a deterministic teaching model, not an empirical system benchmark. |
| Matrix products record chronological order, §9 | Two shear-stage operations expose both orders and their different final states. No commuting assumption is silently introduced. |
| Numerical trial probes versus accepted trajectory, §10 | The selected scheme displays its actual stage times, trial states, derivatives, weighted slope and accepted update; the history reports its status and final error against the analytical solution. Instability and a shortened final step are deliberately exercised. |
| Initial data versus endpoint conditions, §12 | Boundary investigation classifies symbolic endpoint cases and plots the chosen family member. It does not decide mathematical resonance by testing whether a floating approximation to sin(π) equals zero. |

Plots identify coordinates, units where meaningful, sampled analytical/computed provenance and relevant limits. Endpoint markers are not cut in half by the curve clip. The native solver example is presented as an executed program rather than a browser component pretending to run SciPy.

## Executed model and native checks

Final model check: **2026-09-11T05:10:34.528Z**. Final native check: **2026-09-11T05:10:36.406734+00:00**. The durable packet embeds both results and their file hashes. Actual environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and mpmath 1.3.0.

Commands:

```text
node scripts/format-ordinary-differential-equations-source.cjs
node scripts/verify-ordinary-differential-equations-models.mjs
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-ordinary-differential-equations-native.py
```

The generator executes all thirteen displayed programs, and the final native suite independently executes their exact strings and compares all displayed stdout. Stored program prompts are also rendered explicitly before their code through the topic-owned wrapper. The native suite then checks changed cases through the actual displayed helper functions, rather than only inspecting an independently written demonstration.

| Evidence | Actual checked scope |
| --- | --- |
| High-precision scalar states | 64 thermal states and 216 logistic states, including subnormal initial populations; 16 waiting-family equation/join cases. |
| Independent linear algebra | 210 matrix exponentials against SciPy, 105 oscillator/energy states including near-critical cases, and 6 time-ordered matrix cases. Maximum observed matrix absolute difference was 2.3305801732931286e−11; the test uses scale-aware tolerances, not a universal absolute-error theorem. |
| Independent response integrals | 96 forcing states against quadrature and 32 changed calls to the actual held-input helper, including zero/singular matrices and small intervals. |
| Actual numerical mechanisms | 27 stability-polynomial cases, 12 equal-horizon partial-step products, 9 nonautonomous refinement pairs and 18 checks of the stated Euler accumulated-error bound. |
| Additional changed native calls | 51 exact Fraction series identities and 9 near-critical oscillator comparisons with independent numerical integration. |
| Source/input checks | All 23 display formulas parse; 12 malformed or out-of-contract model calls are rejected. The formatter conserved normalized ASTs for five JS/JSX/brief files and the scoped stylesheet's parsed structure. Actual Python examples are conventionally formatted by their generator. |

The root-owned independent reviewer may use complementary inputs and proof arguments. This author's independent numerical oracles do not replace that separate source review.

## Actual browser, keyboard and reading review

Final actual-font browser result: **2026-09-11T05:11:52.350Z**, Edge through Playwright against the actual module route on the shared development server. Google Fonts were loaded through the authorized browser check; the final evidence uses Space Grotesk, JetBrains Mono and KaTeX rather than a fallback-only approximation.

```text
node scripts/review-ordinary-differential-equations-lesson.cjs
```

At **1440, 390 and 320 pixels**, each run passed 26 changed investigation states, 36 keyboard checks, all 14 actual section-anchor arrivals, all 13 visible prompt/program/output comparisons, all 23 display-equation fits and the 18 source-link attributes. It exercises native selects/ranges, resets, optional-depth disclosures and practice hints/solutions. There are no reported page/request/runtime errors, document overflow or clipped SVG text in the tested states. Code and dense comparison tables retain local horizontal scrolling where needed; no claim is made that a long code line becomes readable without scrolling on a phone.

The script saved 204 screenshots. **The author actually opened and inspected 36 final screenshots**, with exact paths and SHA-256 hashes in the durable packet; saved images are not automatically counted as viewed. These include all 23 narrow display equations and surrounding prose, the thermal balance and blow-up figures, paired position/phase reading, fundamental columns, input timing, the swapped-stage state, actual RK4 arithmetic, unstable Euler, the endpoint classification, a changed independent exercise, complete event code, the changed capstone output and alternate-learning references.

Prefinal findings are retained transparently: two equations did not fit at 390 pixels and five further cases did not fit at 320 pixels. They were split along meaningful mathematical steps, with a defined RK4 weighted average rather than a clipped formula. Actual screenshot inspection also caught half-clipped endpoint markers; those markers now sit outside the line-path clip group. The full final browser run and reopened final images include those repairs. The durable packet retains the last prefinal failure record as historical evidence, not a final failure. The valid tiny-population logistic branch was also checked against high precision before final verification.

## Sources, ownership and limits

The design's source table states the exact inspected scope: selected LSU-hosted existence/Picard/Gronwall definitions and proofs, MIT linear-system/matrix-exponential notes, Lebl scalar/system/forcing/numerical sections, and SciPy API parameters, status, tolerances and event limitations. Additional implemented exact-equation and ordinary-point series branches use inspected Lebl derivations. The current API pages identified SciPy 1.18.0; the actual executed installed runtime is 1.18.1, as recorded above.

Two MIT video resources are annotated alternatives. Their official title/embed metadata and associated written-session scope were inspected; full playback or uninspected timestamps are not claimed. The Frobenius chapter is a verified further-reading destination whose full specialist derivations were not reviewed. No failed PDF/playlist retrieval is presented as a read source. Explanations, programs, numbers and diagrams are original to this lesson.

The incoming Dynamics bridge is [resolved with actual implementation evidence](topic-notes/ordinary-differential-equations-linear-systems.md). That same note preserves reasoned open follow-up for full Frobenius, differential-algebraic and nonsmooth-solution courses: the current local caveats and links do not imply those fields are completely taught. The boundary-spectrum/compatibility discovery is delivered to [the actual later PDE owner](topic-notes/partial-differential-equations-conservation-boundary-conditions.md). The earlier [S4 direct-term/initial-state discovery](topic-notes/state-space-models-s4-mamba-mamba-2.md) remains open for that owner; no unrelated body was rewritten.

The completed source is bounded teaching code, not a certified solver for arbitrary floating inputs or a benchmark ranking numerical libraries. Event sign changes can miss interior crossings; local error tolerances are not automatic global guarantees. The lesson closes with the actual next module entry, **Complex Numbers, Fourier & Laplace Transforms**, and does not skip to whichever unrelated lesson is published. Shared catalogue/order/loader files and integrated build evidence remain root-owned.

The frozen production files are the semantic topic, model, examples, labs, standalone figures, scoped CSS and individual blueprint. All seven full hashes and all verification-script hashes are in the durable packet. Any later amendment must retain this identity and state which affected checks were rerun rather than silently attaching these timestamps to changed source.
