# Complexity Analysis & Recursion — lesson design

10 September 2026. Stable ID `complexity-analysis-recursion`; Algorithmic Patterns in DSA. The topic-plan command and current handoff, teaching/code/DSA standards, domain playbook and design brief were read. No destination note existed. The bit-manipulation inbox was assessed: representation and width matter locally, but teaching masks/XOR is not required here and remains with the unresolved owner. The title is retained because resource analysis and recursive decomposition are the two connected promises.

## Scope and progression

This was a planned title without individual teaching content. A newcomer needs a cost model before shorthand bounds, and actual suspended calls before recurrence algebra. Earlier tree/graph examples are useful revisits; their prior exposure is not assumed to have taught formal analysis. Python loops/functions/lists are required; powers, logarithms and induction receive local bridges so the later mathematics route is supporting review, not a hidden detour.

| Outcome / hurdle | Treatment and evidence | Placement |
| --- | --- | --- |
| Specify size, operation and case | Inspect a record scan and unordered-pair count; distinguish item count, independent inputs, output and integer bit length | Core sections 1–3, counted programs and comparison/claim repair exercises |
| Derive rather than guess a bound | Lattice of actual iteration pairs; triangular sum, doubling loop, O/Ω/Θ witnesses and counterexamples | Work-lattice investigation, bounds example, independent loop task |
| Distinguish case and probability from a bound | Search best/worst and explicitly uniform successful expectation; deterministic doubling aggregate versus expected hashing | Exact search and append-budget programs; diagnosis exercise |
| Follow and justify recursion | One array, index contracts, base case, remaining-length progress, pending additions and return value | Inline call/return ladder; frame investigation; exact native trace and iterative counterpart |
| Relate recurrence, total work and live depth | Per-level node count × local cost for five recurrences; leaves count, two children versus one child, constant combine versus linear combine | Level accountant and balanced sum/power programs |
| Account for real space and arithmetic | Input/output/auxiliary distinction, retained slices, frame depth, arbitrary-precision numbers, repeated subproblems | Inline slice staircase, Fibonacci and reversal programs; bit/space transfer |
| Make practical choices | Original record-pair comparison, bounded sensor subtotal, power reuse, branching dependency calculation, resize budget; separate observations from proofs | Worked mechanisms plus own benchmark protocol; local capstone |

The future Binary Search/Sorting/Two-Pointer lesson owns full search boundary/sort/window algorithms. Backtracking/Divide-and-Conquer owns path restoration and larger decompositions; DP owns state design and reuse beyond the explained Fibonacci bridge. Correctness/Termination owns more general proof systems. No catalogue reorder, extra topic or expanded title is needed. The present recurrence and amortization examples fully explain their local claims; they are not placeholders for those later lessons.

## Representation contracts

1. **Iteration lattice:** n=6 initially; exact (outer, inner) executions for square, triangle j<i, and repeated doubling j=1,2,4,…<n. Cells are work events, not timing pixels. The learner changes n=0…16 or loop, predicts growth, selects a row and sees exact totals and a 2n comparison. Same model drives active cells and counts. Inputs apply immediately; Reset restores n=6/triangle/row0. n=0 has no row/work. Display uses local scrolling where needed and labels plus exact table, never color alone. Independently compare combinatorial counts, powers and real Python output.
2. **Call/return ladder (inline):** original [3,1,4], suffix index0→3, base0 and returning4→5→8; arrows distinguish call descent and value return. Layout is correspondence, not time duration. Text captions and responsive rows retain all values.
3. **Pending-frame investigation:** sum over bounded entered integer list, default3,1,4; explicit Apply rejects malformed/overlong input atomically. Each native-equivalent enter/call/base/return snapshot exposes source index, locals, waiting addition and current return. Frames are shown newest-first; source array stays one shared object. Back/reset deterministic; empty input demonstrates base only. No arbitrary Python execution; native example independently verifies event sequence and every snapshot result.
4. **Recurrence level accountant:** n=8, two half-sized children with local size work; choose five decreasing recurrences and power-of-two n=1…32. At each depth display count, size, local work/node and total work; bar length is linearly scaled exact selected work, and blocks encode child counts (bounded32). Leaves cost1. Work sums and maximum sequential call depth are separate. Recursion tree is a work accounting device, not simultaneous CPU execution or a memory map. Independent explicit traversal and closed forms verify sums. Changing parameters resets highlighted level; no invented benchmark comparison.
5. **Retained-slice staircase (inline):** n=4 originals versus length3,2,1,0 copied lists while parents wait. Width encodes copied reference slots; original list separate. Sum6 extra slots, not bytes/deep copies. Compare index-only recursion and iterative auxiliary storage; textual totals accessible.

## Source/claim and alternative-resource ledger

Inspected 10 September 2026. All lesson prose, examples, diagrams and exercises are original. No measured timing curve is published.

| Source | Actual reviewed scope / supported use | Limits |
| --- | --- | --- |
| https://algs4.cs.princeton.edu/14analysis/ | Cost-model, order-of-growth, experimental versus theoretical and amortized sections | Java memory/timing constants are not transferred to Python |
| https://introcs.cs.princeton.edu/python/23recursion/ | Call traces, induction, repeated calls and recursion pitfalls | Uses older examples; no source code copied; Python limits checked in current docs |
| https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/c6d8f06c6f11e3342633dec85498f551_MIT6_006S20_r01.pdf | Asymptotic definitions and model-distinction sections in the seven-page PDF | No video viewing claimed |
| https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/1869dbf640ded6b31f1bd369d2001ef5_MIT6_006S20_r03.pdf | Recurrence expansion/master theorem and algorithm cost discussion | Source code/solutions not reproduced; local examples checked independently |
| https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-1-algorithmic-thinking-peak-finding/ | Video resource page and the 1-D peak/smaller-subproblem sections of its six-page lecture PDF | Recording not watched end-to-end; older course, linked for intuition, not current Python semantics |
| https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/pages/readings/python-cost-model/ | List concatenation and size-dependent primitive cost discussion | Historical measured coefficients deliberately excluded |
| https://docs.python.org/3/library/sys.html#sys.getrecursionlimit | Current recursion-limit contract | No fixed limit or safe instruction to raise it |
| https://docs.python.org/3/library/timeit.html | Repeat/setup/GC and timing interpretation | No timing asserted as measured here |
| Seven official LeetCode statements in the topic practice dataset | IDs344/509/1342/896/50/104/70: displayed title/difficulty, constraints, accessible statement reviewed | No submissions/editorials tested. 104 is an intentional analysis-focused revisit;70 is optional DP transfer |

## Verification plan and status

Exact native strings/stdout, independently built combinatorial/iterative oracles, model states and meaningful invalid inputs passed. The 1440px/390px browser review passed all three investigations, changed/empty inputs, Back/Reset, invalid atomic rejection, keyboard, exact values, overflow, anchors, code/output and practice. Screenshots were opened and ordinary reading figures inspected separately. See [the final evidence and actual counts](COMPLEXITY-RECURSION-VERIFICATION.md). User review/beginner study remain pending. Root owns registration and final application integration.
