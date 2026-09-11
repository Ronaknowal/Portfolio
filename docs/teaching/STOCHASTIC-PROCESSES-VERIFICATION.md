# Stochastic Processes — author verification

Mathematics position33, stable ID stochastic-processes-markov-chains-brownian-motion-poisson. The title, memberships, progress and reader order are preserved. The actual next topic remains Random Matrix Theory; the Queueing and Itô branches do not replace it.

This record distinguishes authored source, author checks, subsequent independent review, production integration and user acceptance. The final freeze and exact six source fingerprints are recorded in [the author evidence](evidence/stochastic-processes-author-review.json). Root owns shared publication metadata, the rollout ledger and integration.

## Original coverage and resulting learning flow

The exact old body is preserved in scratch/stochastic-processes-design/original-lesson.jsx, SHA256 7854bb01f1edd11bab8f4be5e8df9c87db688b233f1694946e68a3ccbbba1676. Original-preservation.json beside it records the three original programs actually executed before replacement. The final native verifier also compares their code strings to the archive: all three are preserved, including comments and outputs.

The old introduction becomes section1's path/time-slice distinction and three processes with identical marginals. Its weather calculation remains in section2, followed by explicit row orientation, path-versus-endpoint probabilities and a hidden-state counterexample. Section3 separates stationary laws, marginal convergence and time occupation using periodic, reducible and sticky chains. The old Poisson calculation remains in section5, now connected to waiting times, cumulative event epochs and exposure units. Section6 adds independently marked routing, its deterministic counterexample and a piecewise intensity clock. The old Brownian sample remains in section8, with the standard-deviation argument explained before the nonunit-grid example.

The previous modeling cautions and support-ticket question are retained and developed in section10: separate trajectories, unidentified departure rows, censored observation windows, held-out diagnostics and schedule-versus-burst mechanisms. New first-passage, finite continuous-time clocks, covariance, bridge/barrier and bounded-martingale explanations make the named families usable. Formal infinite-state stability belongs to Queueing; stochastic integration and general SDE accuracy belong to Itô. No topic rename or catalogue restructuring was required.

There are twelve complete visible programs, five investigations and five inline mechanism figures. Those counts describe the finished teaching choices; they were not quotas. Thirteen independent tasks have separate hints and explained solutions, including a changed literal-trace audit with reference answers and acceptance checks. Program questions render immediately before each program through a topic-owned wrapper.

## Visual and mathematical contracts

| Representation | What is computed or illustrated | Limits and independent evidence |
| --- | --- | --- |
| Path-family strips | Exact finite laws for independent fair redraws, a frozen fair initial bit and an alternating fair initial phase | Four-time strips distinguish two displayed iid examples from sixteen equally likely paths. Exact enumeration checks marginal and adjacent-equality probabilities. |
| Hidden-state cycle | A→B→C→A, with A/B both reported as0 | Reported-history cases demonstrate lost information. The upper return connector is directed. This is a deterministic construction, not a fitted hidden Markov model. |
| Markov probability flow | Applied two-state matrix, four contributions, destination sums and distribution through60 steps | Presets/input changes recompute linked views; exact0/1 transitions are meaningful. Other probabilities stay at least10⁻⁶ from endpoints. NumPy powers and exact small-path enumeration check values. |
| Class-structure figures | Absorption, periodic irreducibility and disconnected closed classes | These separate sufficient conditions from necessity, existence from uniqueness, and convergence from occupation. Finite-state qualifications accompany claims. |
| Reserve investigation | Probability on states0…B, accumulated absorption, first-hit mass, survival and solved h/t equations | B=3…8, upward probability .1… .9, at most60 steps. Starting at a boundary has hitting time0. Fraction closed forms, independently enumerated stopped paths and linear algebra check solutions and fundamental visits. |
| Arrival clock | One event stream linked to routing lanes, right-continuous count steps, integrated intensity and a selected interval | Rates0 or .01…6/minute, UI horizon3, cap120. Clock/marking draws are separate. Alternating marks do not move arrival times. Counts after a capped window are unknown. Poisson/binomial laws and independent rate integrals check results. |
| Holding-time clock | Actual On/Off durations, departures, generator and separate exact P(On at t) | Rates .05…6/hour, UI horizons1/4/12, cap120. Time and jump-visit fractions differ. Final holds are censored. Independent matrix exponentials/integration check the law and expected exposure. |
| Brownian shared increments | Shared and new increments explain covariance σ²min(s,t) | Values are dependent; disjoint increments are independent. Geometry conveys sharing, not measured covariance. |
| Coupled Brownian investigation | Eight seeded finite-grid paths, highlighted increment, pointwise normal intervals and actual squared-increment sums | Finest grid256; coarser views retain its points. Model moments differ from realized sums; no monotonic convergence claim. Independent basis covariance maps and Gaussian quadrature verify claims. |
| Bridge/barrier figure | Fixed endpoints below a threshold, a schematic possible crossing, exact conditional moments and crossing probability | Curved line is explicitly schematic. No unlabeled variance segment remains. Formula is exact for constant drift/scale Brownian bridges. Gaussian conditioning and reflected-density ratios check values. |

Every plot has a visible ordinate/time label or equivalent labeled state geometry. SVG text stays inside its viewBox; larger reserve diagrams and data tables have keyboard-accessible local scrolling. Required explanations precede optional table/deeper disclosures. Color and hover are never the sole explanation.

## Accuracy decisions

- πP=π does not imply convergence from every start. The finite irreducible aperiodic sufficient condition is separate from the finite irreducible occupation theorem. Reducible convergent chains are shown. Detailed balance is sufficient, not necessary.
- The absorbing calculation uses an actual transient system with accessible absorbing boundaries. Having an absorbing state somewhere is insufficient; mean first-passage time can fail outside these conditions.
- Poisson cumulative counts share increments. Independently marked outputs are unconditionally independent, whereas conditioning on their total couples them. Alternating marks and copied streams do not inherit ordinary Poisson-process claims.
- Deterministic nonhomogeneous intensity keeps independent increments but loses stationary increments. A zero-rate interval stays flat.
- Positive tiny interval means use interval overlaps instead of subtracting large accumulated intensities. The regression [2,2+2⁻⁵¹] after earlier intensity12 retains mean .01×2⁻⁵¹. Unrepresentable positive rate products and nonzero decimal literals parsing as0 are rejected; deliberate zeros remain meaningful.
- Poisson probabilities above40 are summed directly through160 instead of subtracting a rounded CDF from1. At the maximum bounded mean36, the omitted tail above160 is negligible relative to the displayed quantity; the tail is labeled approximate. Positive-tail underflow is disclosed.
- Brownian increments use standard deviation σ√Δt. Covariance and squared-increment moments include scale, drift and nonunit time. Quadratic variation is proved in mean square on deterministic equal partitions, not inferred from a picture or unrestricted path-dependent partitions.
- Exported bridge fractions are exact0/1 or at least10⁻⁶ from endpoints. Underflowed barrier probabilities retain a finite log probability and explicit indicator; observed endpoint crossings stay probability1.
- Native event/holding simulators reject exhausted caps rather than silently returning incomplete observations. An unrepresentable nonpositive time advance also fails explicitly. These are educational computations, not production arbitrary-precision software.

## Executed native verification

Run node scripts/verify-stochastic-processes-models.mjs. It exports actual model fixtures and actual displayed program strings, then runs scripts/verify-stochastic-processes-native.py using scratch/lesson-tools/Scripts/python.exe.

The final native record is scratch/stochastic-processes-native-verification/results.json; a durable copy is linked from the author evidence. It includes:

- 12 complete stdout comparisons and exact preservation of all3 original programs.
- 12 finite joint laws,196 Markov states and195 absorbing systems.
- 170 arrival states,1,080 splitting laws and97 continuous-time clock states.
- 144 coupled Brownian states,48 independent basis/covariance maps and576 conditional bridges.
- 26 changed actual-native inputs,24 invalid native inputs,2 native cap rejections and34 invalid browser-model inputs.
- 30,000 repeatable strictly interior uniform draws as an RNG contract check, not proof of ideal randomness.
- 160,451 numerical comparisons after the independent-review conditioning amendment; largest absolute discrepancy about2.91×10⁻¹¹ within the stated tolerances.

Independent methods include Fraction enumeration/closed forms, NumPy matrix powers/inverses, SciPy distributions, finite matrix exponentials, exposure integrals, Gaussian second/fourth-moment quadrature, covariance transforms and reflected endpoint densities. Snapshot trees are recursively frozen and checked. Monte Carlo agreement is not used to establish theorems.

Scripts/format-stochastic-processes.cjs preserves normalized JavaScript ASTs, literal/JSX strings and CSS meaning while formatting only owned sources. Its conservation record is saved beside native evidence. Curriculum validation passed; parent runs production build/loading/order integration.

## Browser review

- scripts/review-stochastic-processes-lesson.cjs checks actual controls, changed inputs, invalid-draft preservation, presets, interval choices, coupled refinement, keyboard resets, all anchors, visible program questions, practice disclosures, geometry and clean logs.
- scripts/review-stochastic-processes-final-reading.cjs checks actual-font reading/equations/inline figures at1440/390/320 and the precision-boundary regressions in the loaded browser module.

Raw screenshots and JSON are in scratch/stochastic-processes-browser/. Final author evidence records actual opened screenshots and passing timestamps. Shared Vite HMR is intentionally disconnected during snapshots; no mocked fonts or network-error suppression replaces real public font loading.

Early review caught and repaired four malformed TeX closing escapes, eleven equations needing shorter phone lines, an eight-pixel overflow from a long covariance expression in an expanded answer, an overlapping hidden-state return label, an unlabeled bridge segment and a table caption forcing unnecessary width. A test's ambiguous native-select label/value match was repaired; it was not a lesson defect. Root restored a temporary missing shared Vite listener. Failed attempts remain distinct from final evidence.

The independent reviewer then identified two splitting issues. The worked paragraph now uses mean3, matching its actual program and joint probability .0967860609071275. Conditional dependence is qualified by positive total and nondegenerate routing. For zero mean and a positive requested total, the exported helper returns conditional:null and conditioningEventPossible:false while retaining the formal binomial marking kernel and the actual zero joint probability. Mean-zero/total-zero conditioning is valid and degenerate. All1,080 split states were rechecked independently; the final browser closure checks routing0/1 and the impossible-conditioning case. No reviewer found another material mathematical defect before author freeze.

The main behavioral pass completed at20:42:52 UTC with39 named changed states per width, 10 anchors, 12 visible program questions/code/output pairs and13 independent practice solutions. Its snapshots precede the small reviewer amendment. Final reading/boundary snapshots close that amendment and additionally verify the compact Brownian raw/centered comparison and wrapping button labels. The final evidence preserves both runs, rather than claiming an earlier snapshot already contained a later correction.

Author verification does not certify untested browsers, screen-reader speech output, arbitrary numerical inputs, population model fit or infinite-dimensional sample-path theorems. Bounded arithmetic, exact/synthetic/schematic distinctions and local scrolling are intentional and visible.

## Resources and destination notes

The accepted [individual design](STOCHASTIC-PROCESSES-LESSON-DESIGN.md) records actual primary-source reads and selection rationale. The lesson annotates Weber, Gallager, Cambridge continuous-time notes, Sigman, Haugh and MIT reflection/Brownian notes. Official video descriptions and lecture identities were verified; no linked recording was watched end to end.

The Queueing note uses catalogue ID queueing-theory-m-m-1-m-g-1-little-s-law, while the legacy source mapping stays unchanged. Its author resolved routing, occupancy and censoring bridges in Math35 and linked the evidence. The [Itô note](topic-notes/ito-calculus-stochastic-differential-equations.md) remains for its owner to assess; it preserves deterministic-partition, adapted-information and exact-versus-approximate bridge distinctions. The unrelated bit-manipulation inbox was not forced into this lesson.
