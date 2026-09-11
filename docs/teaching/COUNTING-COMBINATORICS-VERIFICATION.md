# Counting, Combinatorics & Mathematical Induction — author verification

Stable ID: `counting-combinatorics-mathematical-induction`, mathematics position 46. Author verification completed 11 September 2026 UTC. The exact freeze time, **seven production source fingerprints**, test records and actually opened image fingerprints are in [the durable author packet](evidence/counting-combinatorics-author-review.json). Independent review, production integration and user acceptance remain separate.

## What was implemented and preserved

The [approved design](COUNTING-COMBINATORICS-LESSON-DESIGN.md) was implemented as a complete lesson. The title, identity, sole prerequisite (Sets, Logic, Relations & Proof Techniques) and actual module order remain intact. The root agent owns their registration. [Original-plan evidence](evidence/counting-original-plan.json) preserves the pre-authoring blueprint; the topic had no previous body, code or lab, so no legacy program was removed or replaced.

The lesson has a mechanism-first progression: outcome identity, disjoint cases and constant branch counts; equal-fiber division and its failure under repeated descriptions; selection and multiset/multinomial counts; resource bounds; double counting and induction; inclusion–exclusion; pigeonhole and compression; tiling and partition recurrences; adequate induction base coverage; Catalan decomposition/reflection; and signposted coefficient and cyclic-symmetry extensions. Twelve changed practice tasks have hints before full explanations. Thirteen complete standard-library Python programs have visible questions and actual stdout; setup appears before the first one. The closing route correctly names Single-Variable Calculus, rather than jumping to a later published lesson.

The prior Sets44 induction material is reused as a proof contract, followed by new counting identities, constructive base ranges and structural decompositions. The Backtracking lesson's explicit promise of Catalan counts is fulfilled by both the convolution recurrence and a complete reflection bijection with its inverse. These are not claims to exhaust specialist combinatorics.

## Representation review

| Representation and placement | Actual contract and reviewed teaching purpose |
| --- | --- |
| Unequal branching, section 1 | Explicit allowed lead/helper leaves distinguish varying branch counts from merely prefix-dependent labels. All five assignments are visible; the sum is 1+2+2. |
| Outcome fibers, section 2 | Select real groups and inspect all descriptions. Repetition changes constant-size groups to sizes 1,2,1. Empty choices and impossible descriptions are separate. This is an exact finite partition, not an implied probability distribution. |
| Tagged repeated symbols, section 3 | Four labeled tokens map to the NOON positions, explaining the common 2!×2! multiplicity. Subscripts stay attached to their symbols. |
| Resource containers and star/bar word, section 4 | Token movements update an exact tuple and reversible word. Minimum/capacity presets recompute feasibility and counts; empty runs are marked with an explicitly nonliteral italic zero. The empty feasible tuple and no feasible tuple are distinguished. |
| A chaired committee, section 5 | Both counting orders produce one visible marked committee. The object and its reversible descriptions explain the equality. |
| Per-object overlap ledger, section 6 | Each contribution is derived from the actual membership board. Object 12 has weights 3,0,1 through the correction stages; the complete union has size eight. Empty memberships work. No area or frequency is implied by the layout. |
| Compression capacity, section 7 | Eight concrete three-bit inputs and seven complete shorter outputs expose the injection obstruction, including the empty output. Source assumptions and side-information limits are explained. |
| Tiling strips and partition blocks, section 8 | Added after the ordinary-reading pass identified a useful visual gap. All five length-four tilings split by first tile; actual tile spans encode lengths. A second figure shows singleton insertion versus either of two existing blocks. It explicitly says one smaller partition is shown and all three contribute. |
| Induction coverage, section 9 | Removing a base certificate affects proof support, while independent arithmetic still displays a valid representation. Both 4/7 and changed 3/5 constructions are checked. A finite chain illustrates, and does not replace, the arbitrary induction proof. |
| Balanced paths, section 10 | Prefix position and unmatched-open height are actual coordinates. The first negative boundary, reflected endpoint +2 and inverse first +1 are exact. The valid-word mode reports the unique first-return decomposition; the empty word is a separate base object. |
| Coefficient construction, section 11 | Degree positions and integer counts come from the actual polynomial convolution. Changed capacities, impossible targets and the empty product are handled. The new station's exact contributions remain visible. |
| Rotational orbits, section 12 | Actual binary patterns, distinct images, stabilizing shifts and fixed-word totals agree. Marking the start changes the outcome definition. Reflections are explicitly not identified. |

The final implementation has **seven interactive investigations and six inline figures**, selected for different conceptual hurdles. The extra two recurrence figures are in `CountingRecurrenceFigures.jsx`, a small topic-owned module, not a shared all-topic bundle. Reusable controls do not impose identical mathematical representations. All numerical readouts are exact bounded constructions; there are no illustrative benchmark curves or measured-data claims.

## Executed model and native evidence

Commands run from the application repository:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/generate-counting-combinatorics-examples.py
node scripts/verify-counting-combinatorics-models.mjs
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-counting-combinatorics-native.py
```

The final model record at `scratch/counting-combinatorics-verification/model-results.json` passed at 00:48:56 UTC. It exports actual JS states for a separate Python oracle, checks 23 malformed/range/sparse-input cases and parses all 14 actual formula templates with KaTeX error throwing. Mathematical formulas that admit large counts use `BigInt`; the native reference uses exact Python integers. Enumeration has explicit bounds rather than silently truncating a count.

The native record at `scratch/counting-combinatorics-verification/native-results.json` passed at 00:48:57 UTC:

- All 13 actual displayed programs reproduce their stored stdout.
- 60 choice states compare independent Cartesian-product enumeration and finite fibers.
- 1,282 bounded-allocation states compare direct tuple enumeration with both the actual JS model and the displayed Python inclusion–exclusion helper.
- 4,096 finite membership boards compare union rosters and stage weights with independent per-object binomial multiplicities.
- 1,352 induction states compare backward witness chains with forward closure from enabled certificates and a separate integer-coin search.
- 1,275 parenthesis words compare prefix heights, a ballot-prefix counting DP, first-return pieces and the exact reflection/inverse target family.
- 500 coefficient rows compare with the distribution of sums over direct choices, rather than another convolution implementation alone.
- Eight ring lengths compare fixed counts against permutation cycles; every orbit partitions the strings and satisfies its orbit–stabilizer identity.
- 609 binomial cases compare with Python integers, including counts beyond safe floating-point integer precision.
- 35 actual Stirling calls compare with independently canonicalized restricted-growth partitions; eight derangement sizes compare with direct permutations.
- All 12 changed practice tasks' numerical promises are independently checked, including seven shifted/capped allocations, the 29=8×3+5 witness, reflection of `())(`, five-site rotation count, and combined counts 84 and 54. The impossible trained-team extension returns zero.

These are finite verification results. The body separately supplies the general division, induction, recurrence, inclusion–exclusion, reflection and orbit-counting arguments. A large case count does not establish those proofs or substitute for independent mathematical review.

## Actual browser, reading and accessibility evidence

`node scripts/review-counting-combinatorics-lesson.cjs` passed at **00:49:02 UTC** using Edge, the real `5173` route and intended fonts at **1440, 390 and 320 pixels**. Each width includes 14 genuine anchor arrivals, 41 recorded state scenarios, 35 keyboard actions, all 13 question/code/stdout checks, the 12 initially hidden solution disclosures, 14 equations, source-link behavior and ordinary reading captures. Buttons are activated with keyboard focus and Enter; a native select is also changed with ArrowDown/Enter. Reset, previous/next, zero and impossible states, unsupported proofs, reflection/inverse, changed coefficients and marked/unmarked rings are exercised. No page/console errors, math parse errors, SVG label clipping or document overflow remain. Native program blocks intentionally preserve their own horizontal scrolling.

`node scripts/review-counting-recurrence-figures.cjs` then passed on the **final two inline additions** at all three widths. It checks all five exact tilings, their CSS unit spans and actual pixel widths; the actual partition memberships and insertion cases; normal reading and caption captures; all unchanged 14 equation fits; and no page/document overflow. It changes no mathematical model, existing interactive control, program or solution. Its exact timestamp is in `scratch/counting-combinatorics-browser/final-recurrence-results.json`. The earlier comprehensive run is not relabeled as a run of the later additions.

Forty final images were actually opened and read, rather than merely generated: all original inline figures; all distinct changed investigations; all 14 equations at 320; six views of the final recurrence additions; ordinary desktop/mobile prose; changed practice; an actual program; accepted capstone output; and resources. Their exact names and hashes are recorded in the durable packet. The pixel review checked text, equations, group membership and diagram meaning as well as fit.

### Findings repaired before freeze

1. Initial shell/string escaping damaged the LaTeX templates. The first browser pass exposed 11 parse errors and the rendered command loss. The actual 14 templates were rewritten with literal backslashes, parsed with error throwing and visually reviewed. This did not change the exact counting models or program outputs. The initial failure packet is preserved rather than reported as a pass.
2. The Stirling base-condition row was seven pixels wider than its 320-pixel reading area. Breaking the base conditions onto separate mathematical lines repaired the fit without changing the formula.
3. Opened mobile pictures showed that path/ring labels would benefit from larger text. Their final topic-specific label sizes and shorter ring caption keep the entire small geometric object readable without panning. Tagged letters now remain single labeled tokens; singular token text and coefficient superscripts are also clear.
4. Ordinary reading showed that recurrence decomposition deserved actual objects as well as formulas. The final tile and partition figures were added and independently checked as described above.
5. The focused figure harness initially inspected `gridColumnEnd`, while the authored `gridColumn: span n` correctly puts its span in `gridColumnStart`. The assertion was corrected and complemented by independent measured-width checks. This was a harness issue, not a production geometry defect.

`scripts/format-counting-combinatorics-source.cjs` records normalized AST conservation for the original five formatted sources; the final recurrence module and its scoped CSS/body insertions are intentional later additions, not represented as formatting-only changes. The exact final seven source identities are authoritative.

## Research scope and practical limits

The design records the inspected primary/authoritative written sections: MIT's counting/division, overlap/pigeonhole and strong-induction notes; selected formal-series sections; Levin's encodings and set-partition recurrence; Sagan's lattice reflection and orbit-counting arguments; and versioned Python standard-library contracts. MIT's official video titles/resource pages and matching notes were inspected. Full video playback, exact timestamps and entire-book reading are not claimed. Alternate routes are annotated and open separately so the learner retains the lesson.

Stable core scope is complete under the design, including the local prerequisites for optional branches. General group actions, analytic/exponential generating functions, full integer-partition theory and extremal combinatorics remain specialist continuations; the lesson does not pretend finite examples exhaust them. The interview bit-manipulation ownership note remains outside this topic. No shared route reordering, catalogue audit, deployment or full production build was performed by this author.
