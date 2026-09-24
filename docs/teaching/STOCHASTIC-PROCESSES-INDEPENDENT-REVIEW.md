# Stochastic Processes — bounded independent review

Independent reviewer: `testing_documentation_completion`. Topic: mathematics 33, `stochastic-processes-markov-chains-brownian-motion-poisson`.

The complete installed body, models, all lab rendering logic and scoped CSS, twelve actual displayed programs, the full individual design and six-section original archive were read. This is a complementary mathematical/source review with independently formulated numerical checks and actual inspection of selected author screenshots. It is not the author's full browser suite repeated or relabeled as independent work. No production source was edited by this reviewer.

**Disposition: closed with no unresolved material finding in the bounded scope.** Both reported splitting issues are repaired and complementary execution passes. All six production hashes match the author freeze at **2026-09-10T20:54:38.276130+00:00**. Exact source, numerical and image evidence is in [the durable independent record](evidence/stochastic-processes-independent-review.json).

## Findings and their resolution

1. **Conditional dependence needed its nondegenerate assumptions.** The text and arrival lab originally said that conditioning on the total makes split counts dependent, including the selectable routing probabilities 0 and 1. For a fixed total n, the conditional covariance is −nr(1−r): dependence requires n>0 and 0<r<1. At either routing endpoint, or total zero, both counts are deterministic after conditioning. The author qualified both the body and UI. The exported `splitCountLaw` also now distinguishes the formal binomial mark kernel from a conditional probability when mean=0 and the requested total is positive: the latter is `null`, with `conditioningEventPossible:false`, while joint/product probabilities correctly remain zero. This is a probability-zero conditioning event, not an extra observed count law.
2. **One worked number disagreed with its complete program.** Section 6 stated μ=2.5 while reporting the joint split probability 0.096786 for r=.4,m=1,n=2. That value belongs to μ=3, which the displayed `arrivals` program already used. The author corrected the prose to μ=3. The conditional probability .432 remains unchanged; μ=2.5 would instead give approximately .092345623 for the unconditional joint probability.

The reviewer reported both issues before author freeze. The final source read confirmed the corrections, and changed-input checks include zero intensity, zero total, routing endpoints and nondegenerate marks. Neither the factorization proof nor the original program's output needed changing.

## Mathematical and teaching scope inspected

- **A law versus a path:** redraw, frozen and alternating processes share their one-time fair marginal but not joint laws or occupation behavior. Strict stationarity is defined for shifted finite collections and separated from stationary increments and ergodic time averages. Poisson and Brownian families are correctly identified as Markov as well.
- **State adequacy and finite chains:** row orientation, homogeneity, path versus endpoint probabilities, Chapman–Kolmogorov and hidden-state aggregation are explained locally. Existence, uniqueness, marginal convergence and occupation averages have distinct hypotheses and counterexamples. The positive-power/minorization argument establishes the claimed finite mixing result; the return-cycle argument for occupation does not require aperiodicity. Countable-state positive recurrence is not inferred from finite-state arguments.
- **Absorption:** the hitting-time definition includes time zero; first return does not. The boundary equations, finite mean under access to absorption, fundamental matrix and censored survival are coherent. The fair and biased examples distinguish time to either boundary from success. The deeper formula's near-fair cancellation warning is appropriate; the browser solves the finite equations.
- **Point processes:** increment intervals use (s,t], while cumulative levels share earlier increments. Counts, exponential waits and Erlang epochs are connected through the same event. Independent marking, conditioning and superposition have separate assumptions; alternating marks do not produce ordinary Poisson outputs. Deterministic accumulated intensity and its zero-rate pieces are distinct from random or self-exciting intensity. Work caps do not silently turn incomplete traces into completed observation windows.
- **Continuous-time chains:** generator entries are rates, rows sum to zero, and the matrix exponential is distinguished from entrywise exponentiation and invalid large-step Euler probabilities. The elapsed-time law, finite expected exposure and holding-time-weighted stationary occupation agree. The embedded two-state chain alternates, so its visit fractions differ from time fractions.
- **Brownian motion:** continuity belongs to the process definition; exact finite-grid Gaussian sampling does not make straight interpolation an actual Brownian path. Variance versus standard deviation, drift/scale units, covariance of shared increments, pointwise versus pathwise bands, and coupled refinement are explicit. The bridge Gaussian conditioning and reflection argument use constant coefficients. The quadratic-variation proof is for deterministic equal partitions in mean square and does not claim convergence for arbitrary path-dependent partitions. The bounded-stopping proof states the fixed finite bound and integrability required.
- **Practical learning:** all thirteen changed practice solutions were read, including transformed units, a three-state transition law, boundary censoring, conditional covariance, a changed Brownian bridge and temporal data splitting. The fitting examples preserve separate trajectories and complete exposure. References supplement local reasoning and clearly state that recordings were not fully watched.

The original weather, Poisson and Brownian programs and outputs were independently compared with the archive. All three remain conserved apart from the wrapper's trailing newline. Original ideas about model assumptions, hidden observations, time resolution, temporal diagnostics and support-ticket seasonality/bursts are retained and explained more fully.

## Complementary execution

Command:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 scripts/verify-stochastic-processes-independent.py
```

The final execution at **2026-09-10T20:55:27.469011+00:00** passed against the exact frozen source and repaired split helper. Its result is `scratch/stochastic-processes-independent-review/results.json`. Python 3.12.14 executes the actual imported program strings; NumPy/SciPy support independent oracles.

| Independent calculation | Covered cases |
| --- | ---: |
| Actual complete programs and displayed stdout | 12 |
| Original programs and outputs conserved | 3 pairs |
| Exact two-state path enumeration under changed laws | 40 endpoint laws |
| Changed three-state native propagation versus exact path weights | 6 endpoint laws |
| Changed reserve problems and actual native rational helper | 90 states |
| Exact stopped eight-coin histories, first hits and drift-martingale identities | 23,040 histories |
| Poisson uniformization of CTMC laws and integrated exposure | 12 parameter states |
| Gaussian conditional covariance and independently integrated killed heat kernels | 9 bridge cases |
| Exact degree-four Gaussian quadrature, including cross-resolution QV covariance | 162 quadrature states |
| Nested path identity across four resolutions and three path choices | 12 path views |
| Changed splitting probabilities and impossible-event contracts | 80 cases |
| Changed scheduled clocks, intensity inversion and censored caps | 27 cases |

Maximum numerical discrepancy was about 8×10⁻¹⁵. The oracle formulations have different failure modes from the implementations: exact stopped path enumeration checks first-hit flows and accumulated survival; CTMC uniformization uses Poisson mixtures of a discrete opportunity chain; the bridge check integrates products of absorbing heat kernels; Gauss–Hermite quadrature verifies fourth moments and covariance across coupled resolutions. These finite computations complement, rather than prove, the stated general process theorems.

The reviewer directly read [Weber's Markov notes](https://www.statslab.cam.ac.uk/~rrw1/markov/M.pdf), sections 9.3 and 10.1, to distinguish convergence assumptions from occupation averages. [MIT's Brownian-motion reflection notes](https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/aca1518a09539a09ddd37428ab0d0268_MIT15_070JF13_Lec7.pdf), the reflection principle and joint maximum/endpoint argument, support the barrier calculation. These were scoped source checks; no full textbook proof or full video playback audit is claimed.

## Actual visual inspection and limits

Nine author-generated images were independently opened and read under `scratch/stochastic-processes-browser/`:

- `final-inline-0-390.png` and `final-inline-1-320.png`: aligned process-law slices and the hidden-state memory counterexample.
- `reading-section-8-390.png` and `reading-section-7-320.png`: ordinary Brownian and continuous-time-chain reading flow, units and equations.
- `final-inline-4-390.png`: a correctly labeled schematic between-observation crossing.
- `final-reading-poisson-splitting-320.png`: repaired dependence/zero-event conditions in actual narrow reading.
- `final-reading-markov-definition-1440.png` and `final-reading-variation-390.png`: readable matrix setup and deterministic-partition derivation.
- `final-brownian-comparison-320.png`: the final compact table distinguishes realized versus expected raw/centered quadratic variation with readable columns.

A separate very tall Brownian capture was opened but resized too much for detailed inspection; it is deliberately not counted as a reviewed readable screenshot. This reviewer did not rerun the author's broad browser/keyboard suite. The author’s complete behavior run at20:42:52 UTC precedes the splitting amendment; its focused final actual-font1440/390/320 reading and boundary checks at20:51:45 UTC close that change. These executions remain attributed to the author. The reviewer independently matched their final six-file source identity.

No additional material issue was found in this bounded scope. This does not establish arbitrary floating-point safety, validity for real traffic/physical data, stochastic theorem coverage beyond the stated hypotheses, integrated production approval or user acceptance. No shared curriculum or publication files were edited.

## Final production identity

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/stochastic-processes-markov-chains-brownian-motion-poisson.jsx` | `7133383f1b1b15adb1b16b029cacbf12966c18b7fde10549832fbca8a2556c06` |
| `src/learn/data/stochastic-processes-models.js` | `909cc8d2678d709558627f777bcf9f64f35c85226d9e1c933227c7fdecebc921` |
| `src/learn/data/stochastic-processes-examples.js` | `5582eee8b842bfbf6f4926a2b7bdd5df828e31a96d039976752bf9b6d051f482` |
| `src/learn/components/lesson-labs/StochasticProcessesLabs.jsx` | `c1e74207f7d366b032099bcbbf593ac354b50fc0e174af1f2c187b7ce0e11f7b` |
| `src/learn/components/lesson-labs/stochastic-processes-labs.css` | `05d2936b0cbf7ccfe2a15b897c5a8545062970e791265690e784235bc8c1a6fc` |
| `src/learn/data/curriculum/blueprints/stochastic-processes-markov-chains-brownian-motion-poisson.js` | `5fcd97a6314528bc8977b4fe0ce6c82ec7dd586148e60c3d69427d74915af018` |
