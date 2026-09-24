# Python foundations reimplementation

Date: 9 September 2026. Stable ID: `python-basics-types-control-flow-functions-modules`.
Authorized scope: first five lessons in the recorded curriculum route. This is topic one, for a learner with no programming prerequisite. This document is an implementation/design record, subordinate to the current teaching standard.

## Learning contract and design

The old page contained useful runnable examples and accurate reference detail, but asked beginners to absorb syntax inventories before seeing state. A single reference trace did not explain branches or the separation between call results and output. Preserve its complete examples and nuanced optional details; replace the primary route with an explicit temperature-report investigation.

| Outcome / hurdle | Mechanism and representation | Practice evidence | Depth |
| --- | --- | --- | --- |
| Run a script and distinguish code from output | File → interpreter → printed result static flow; tiny full script | Change input, predict result, run fresh | Core |
| Choose numeric/text/missing values | Text-to-number stages; exact type/operation examples | Diagnose string concatenation and preserve zero | Core |
| Explain a name vs an object | Separate names, arrows and list elements; switch alias/copy and append/rebind; a separate static table teaches indices | Predict both observed lists; transfer to nested shallow copy | Core plus deeper |
| Follow a decision and a loop | Input cursor → missing gate → threshold gate → selected container, beside active code | Compare thresholds including zero; explain continue vs break | Core |
| Distinguish calls, local state, print and return | Caller waiting → local frame → return path vs output channel | Predict result and console under return/print; change temperature | Core |
| Reject invalid input and handle errors intentionally | Complete parser with typed failures, empty-data and finite-number check | Malformed, blanks-only, zero and nonfinite cases | Core |
| Build a reusable two-file program | Separate module definition/import from guarded entry point | Run and import without unintended report output | Core |
| Transfer workflow to another domain | Independent highest-score selection from a name/score dictionary with an explicit tie policy | Empty input, negative scores, first-on-tie behavior, unchanged input and a last-on-tie variation | Applied |
| Retain precise Python detail without blocking core | Optional collection, scope/default, numeric and iteration branches | Repair mutable default / shallow copy errors | Deeper |

Finish line: write, trace and explain a small program that turns text readings into a validated summary, without relying on notebook state. NumPy arrays, broadcasting and vectorization follows in the opening curriculum sequence: the bridge is lists, zero-based/sliced selection, loops, function results and imports. Other guided routes can have a different immediate successor; the reader's named Next link follows that route. OOP, generators, decorators, full environments and testing have their own later lessons.

## Visual contracts

1. **Names and objects.** Start before assignment, with two name slots and no lists. Predict whether append affects readings when backup is an alias or a copy. Step changes arrows or list slots; rebinding creates a distinct list. Labels A/B/C are teaching identities, not memory addresses. Previous/Next/Reset, native selects and textual explanation accompany the diagram. Unreferenced objects are hidden; garbage collection and full Python evaluation are outside this model. Separate nested-copy runnable example establishes the shallow-copy limit.
2. **Route each reading.** Fixed input `[18, None, 25, 31, 0]`; choose threshold 0, 20 or 30. Predict the final selected values before stepping. An input cursor and labeled gates expose missing/compare/add/back-to-loop states; selected values remain visible. Back/Reset reconstruct exact prior state. Text labels supplement color. Model only supports finite numbers and the explicitly shown loop, not arbitrary Python code. Zero threshold tests the missing-vs-false misconception.
3. **Call frame and output.** Choose 0, 20 or 100 degrees Celsius and return or print. Predict the caller's result, then step function definition, paused call, local calculation, return/output, assignment and final print. The diagram distinguishes the result channel from console output. Every preset resets, and Back/Reset recover states. Frame is a conceptual name scope, not an implementation of CPython stack memory; recursive calls, exceptions and closures are outside this explorer.
4. **Program/module flow.** Static local-file diagram places reusable readings.py beside report.py and labels imported function / input / returned dict / formatting. A paired direct-run vs import example demonstrates the main guard. No additional control is needed for this relationship.

## Claim ledger

Primary Python documentation reviewed 9 September 2026, showing Python 3.14 documentation at retrieval. Displayed exercises use longstanding Python 3 features; runtime evidence is the installed Python 3.11.7, not a latest-version claim. Original teaching prose/diagrams; no copied educational structure.

| Claim / decision | Primary source and locator | Verified nuance |
| --- | --- | --- |
| Assignment, numeric/string operations and slicing | https://docs.python.org/3/tutorial/introduction.html sections 3.1.1–3.1.3 | Text digits remain strings; ordinary number division, zero-based indexes, excluded slice stop; list assignment does not copy |
| Branches, loops, continue, return, parameters and default evaluation | https://docs.python.org/3/tutorial/controlflow.html sections 4.1–4.9 | First matching branch; range stop excluded; implicit None return; mutable defaults created at definition; keyword-only parameters |
| Collections and shallow copying | https://docs.python.org/3/tutorial/datastructures.html sections 5.1–5.6 | Mutating list methods return None; comprehension; tuple mutability nuance; dictionary insertion order; set uniqueness |
| Truthiness, identities and sequence semantics | https://docs.python.org/3/library/stdtypes.html truth-value testing, comparisons and sequence types | Zero and None are both false but distinct; and/or return operands; str counts Unicode code points rather than displayed glyphs |
| Modules and import boundary | https://docs.python.org/3/tutorial/modules.html sections 6 and 6.1.1 | Normal imports cached by module name in a process; direct execution uses __main__; imports execute top-level code |
| Specific errors and exception handling | https://docs.python.org/3/tutorial/errors.html sections 8.1–8.6 | Syntax vs runtime errors; traceback diagnosis; targeted except; raise and chaining |
| Floating-point representation and finite validation | https://docs.python.org/3/tutorial/floatingpoint.html and https://docs.python.org/3/library/math.html#math.isfinite | Many decimals approximate; tolerance context; parsing nan/inf succeeds, so parser explicitly rejects nonfinite values |

## Evidence status

- Implementation: complete. The stable lesson now has three separate mechanism investigations plus static program and module flow diagrams. New examples include a finite-validated reusable parser, explicit returned summary, import behavior and changed inputs. Existing collection, numeric, function/default and loop detail is retained in optional deeper branches. No old useful topic or route was removed.
- Runtime/model verification: `scripts/python-foundations-verify.mjs` passed with Python 3.11.7. It executes 22 displayed new/retained examples, checks 260 model states/outcomes against independent Python execution or selection comprehensions, and validates seven report contract cases, including zero, blanks, malformed text, NaN/±infinity and input preservation. Independent winner task variation and revised tie policy are also executed. Evidence: `scratch/python-foundations/runtime-results.json`.
- Browser/keyboard/mobile/visual: `scripts/python-foundations-review.cjs` passed at 1440 and 390 pixels. It exercises all three labs, keyboard Enter stepping, back/reset, control resets, changed-copy/threshold/temperature/mode outcomes, solutions, all six lesson navigation anchors, and absence of page errors and document overflow. Screenshots of all three labs at both widths are saved under `scratch/python-foundations/`; reference, flow and call screenshots were visually inspected. These tall element captures can include fixed navigation across the image; ordinary viewport/interaction checks passed. The final batch's scoped data-lab captures exclude that bar during capture only. See [the first-five integration record](../../FIRST-FIVE-REIMPLEMENTATION.md) for final build, cross-page and integrated visual evidence.
- User review: pending. Linux remains the only user-approved quality reference.
- Learner evaluation: author heuristic only; no observed novice trial.

## Files and reproduction

- Lesson: `src/learn/data/topics/python-basics-types-control-flow-functions-modules.jsx`.
- Isolated model: `src/learn/data/python-foundations-model.js`.
- New runnable fixtures: `src/learn/data/python-foundations-examples.js`; retained examples are imported from the existing batch-one file without changing it.
- UI: `src/learn/components/lesson-labs/python-foundations-labs.jsx` and `python-foundations.css`.
- Computational check: set `PYTHON_EXECUTABLE` to the installed absolute interpreter path when PATH-based child spawning is restricted, then run `node scripts/python-foundations-verify.mjs`. The verified local path is `C:/msys64/ucrt64/bin/python.exe`.
- Browser check: point `PLAYWRIGHT_PACKAGE` to the available Playwright package, run the local site at port 5173, then `node scripts/python-foundations-review.cjs`. It uses headless Edge. Selectors: `[data-pyf-lab="references"]`, `[data-pyf-lab="flow"]`, `[data-pyf-lab="calls"]`.
- `scripts/review-programming-batch-one.cjs` now delegates current Python/OOP coverage to their dedicated browser reviews and retains the iterator lesson's original checks. That public entry point passed end-to-end at 1440 and 390 pixels. The legacy native script retains old fixtures as regression evidence, with live source coverage limited to the unchanged iterator lesson; it is not a substitute for the current Python model/example verifier.

Remaining boundaries: the diagrams model fixed examples rather than arbitrary Python; the parser assumes a list of strings and modest finite Celsius values, not a production sensor/large-number validator. No novice study or claim of universal mastery is made. User acceptance is still pending.

## Visual follow-up · 10 September 2026

The [Python Foundations representation review](PYTHON-VISUAL-REVIEW.md) adds an immediately visible assignment-versus-copy reference figure before the interactive investigation. It records the gap, retained labs, precise visual contract and fresh Python/browser evidence. Use that review with this design when continuing; the title and existing outcomes are retained.
