# Programming and DSA control review — 21 September 2026

This bounded follow-up checks the 17 reviewed Programming & Scientific Computing lessons and 22 reviewed Data Structures & Algorithms lessons for the interaction/affordance defects reported in the residual and calibration figures. It does not reopen their completed mathematical or curriculum reviews, and it is not a claim that every possible input or future lesson is perfect.

## Findings and repairs

The inspected diagrams use actual selectors, array/bit/grid buttons and explicit process steppers. The native ranges have change handlers. Static graph nodes and data marks remain illustrations; their surrounding controls identify how to edit the model. No second instance of the residual figure's fake slider was confirmed in this scope. Existing shared dark/amber control styling remains applicable.

Eleven lab or inline-figure instructions still asked the learner to predict a result before using an already playable mechanism. They now direct the learner to move, step, inspect or compare it instead:

| Lesson | Corrected interaction copy |
| --- | --- |
| Bash Scripting & Command-Line Automation | Follow expansion into argument boundaries. |
| Arrays, Strings & Hash Maps | Watch which occupied bit is removed. |
| Disjoint Sets / Union-Find | Open the ring's center and inspect connectivity. |
| Binary Search, Sorting & Two-Pointer Patterns | Compare a chosen processing rate with its slot budget. |
| Greedy Algorithms & Exchange Arguments | Compare individual completion times and maximum lateness after an exchange. |
| Dynamic Programming | Follow the changed route after opening a blocked exit. |
| Segment Trees, Fenwick Trees & Range Queries | Follow the computed prefix-block trace. |
| Shortest Paths, Spanning Trees & Topological Ordering | Follow the extraction after changing a weight. |
| String Matching | Follow the next prefix candidate during fallback. |
| Network Flow | Inspect the displayed bottleneck before advancing the algorithm. |
| Computational Geometry | Use the coordinate sliders and watch the intersection classification. |

These are copy-only corrections in eleven topic-owned JSX files. Separate independent practice and technical uses of “prediction” remain intact; no learner-prediction form or reveal gate is introduced. Selected-topic preflights were read; existing resolved Arrays/DP/Range Query destination notes remain resolved.

The additional 320px check found two real layout defects that the earlier 390px survey did not expose. The **Algorithm Correctness** invariant selector retained its intrinsic minimum width and extended to 329.8px; its grid select now has `min-width: 0` and `width: 100%`. **Backtracking & Divide and Conquer** contained an unbroken plain-text recurrence that made the page 337px wide; spaces at the addition operators provide mathematically natural line breaks. Its locally scrolling diagrams were already correctly contained. These repairs do not alter any model, data, calculation, initial value, practice or reference.

After correcting the pointer test's scrolling race, the complete 320px pass exposed two further narrow layouts: the **Complexity Analysis** recurrence selector needed its label and select to shrink below their intrinsic widths, and **Ordered Patterns** needed a local scroll region around its interval-union SVG. The latter retains its minimum drawing width for legible labels; it now scrolls with pointer or keyboard inside an explicitly named region instead of escaping the page. An independent layout-only traversal reached all 39 pages without stopping at interaction failures and found no remaining candidate after applying these exact two proposed repairs in the browser. Final production verification checks their compiled implementation.

## Evidence and limits

The baseline loaded all 39 routes, inspected 1,099 visible controls and exercised 158 controls, including every one of the 18 initially rendered native ranges by keyboard. It found no unlabelled visible field, page error or escaped content at desktop and 390px. Of those representative actions, 125 changed non-control text or graphics. The remaining cases included re-applying unchanged drafts, selecting which endpoint to edit and choosing a key for an explicitly staged B-tree operation; those are not counted as proof of a broken control or as successful output-change tests.

The final regression checker, `scripts/verify-programming-dsa-control-review.cjs`, passed against the frozen production build: **39 routes**, **18 actual native pointer drags and Home/End keyboard edits**, and **94 process actions with 94 changes to non-control text or graphics**. Desktop/320px layout, revised displayed copy and page-error checks pass. All six retained narrow-screen captures were inspected. The [final receipt](evidence/programming-dsa-control-review.json) binds the source hashes and production manifest. Backtracking's hierarchy and Ordered Patterns' interval diagram pan with the keyboard inside their named local regions; the hierarchy's rightmost leaf can be focused and activated. External Memory's explicitly staged key/range controls produce the expected present/absent searches and exact returned records after advancing their operations.

All **20 visible checkbox labels** were clicked to toggle and restore their associated inputs. Nineteen labels are at least 24px tall at 320px. The String Matching fault label is a compact 254×21.59px line, separated by 12px from its nearest other control, and the full label works; its 18px native glyph is not the entire clickable target. This measured disposition avoids treating every small glyph or locally scrolled control as a defect.

An initial pointer failure was a harness coordinate race during smooth scrolling, confirmed with real pointer/input event telemetry: after deterministic scrolling, the same untouched range emits every intervening value from 8 to 2. The checker now uses reduced-motion preference and waits two animation frames before measuring. Its Union-Find selector is exact, and each route restores its own viewport even after a prior failure. A final External Memory test initially matched both the native `output` status role and its result paragraph; the corrected paragraph selector passed in a [single-topic follow-up](evidence/programming-dsa-control-review-focused.json). The 38 passing route results were reused, and the follow-up was merged only after checking identical production-manifest and matching source hashes. No runtime change was needed for these harness corrections.

The [source/baseline receipt](evidence/programming-dsa-control-changes.json) checks 296 scoped dependencies and records the eleven copy corrections plus the four responsive repairs across fourteen topics and fourteen runtime files. Unchanged numerical/algorithm reviews can therefore be reused. This review does not assert exhaustive behavior for every hidden disclosure, every button, all arbitrary user data or topics outside the 39-topic scope.
