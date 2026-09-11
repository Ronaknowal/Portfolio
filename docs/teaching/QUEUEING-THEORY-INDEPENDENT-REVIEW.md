# Queueing Theory — bounded independent review

Mathematics position 35, `queueing-theory-m-m-1-m-g-1-little-s-law`. This review is independent of the topic author's numerical and browser runs. It reads the complete actual lesson, all ten displayed programs, models, labs, design and verification record. It does not certify arbitrary queueing systems, every floating-point input, integration or user acceptance.

The complementary source and numerical checks pass after one concrete simulation repair. This review closes against all six author-frozen production hashes at **2026-09-10T20:53:15.304113+00:00**. No unresolved material finding remains. The [durable independent record](evidence/queueing-independent-review.json) preserves the pre-fix failure, final passing run, exact identities and nine images actually opened here. The [author record](QUEUEING-THEORY-VERIFICATION.md) owns its broader native/browser evidence.

## Source, mathematics and learning flow

The complete read covered the actual FCFS trace and Lindley recurrence; clipped residence versus completed cohorts; Little's limiting conditions and PASTA; geometric normalization and stationary FCFS tails; residual service, P–K, heavy tails and repeated vacations; pooled capacity, finite buffers, measurement and all changed practice solutions.

No mathematical claim blocker was found. The lesson keeps the following distinctions explicit:

- An exact finite occupancy identity includes unfinished residence. A completion-only mean does not automatically satisfy that finite-window equality. Little's long-run statement has compatible populations and limiting assumptions.
- Poisson arrivals sample a nonanticipating state under the stated assumptions. PASTA does not establish independent successive delays or justify a state-dependent routed arrival stream.
- The FCFS M/M/1 total time is exponential; queue waiting instead includes an atom at zero. The inverse correctly returns zero inside that atom. Infinite-buffer stability, finite-buffer stationarity and a deterministic critical-load counterexample are distinguished.
- Busy-time sampling is length biased. Its conditional residual differs from a residual averaged over idle periods. The P–K decomposition uses independent, unrevealed waiting customers' own service requirements; it is not asserted for arbitrary scheduling.
- A stable heavy-tailed queue can have infinite mean wait. Capped-service comparison supplies a lower-bound argument. The repeated-vacation derivation states independent repeated vacations and exhaustive service; its low-positive-arrival limit is not an observed mean for an empty zero-arrival population.
- Pooling adds simultaneous service; a faster server changes each job's duration. Finite capacity includes service, and admitted rather than offered throughput determines the admitted-customer mean.

The ten changed practice tasks have meaningful hints and explained answers. The original 90/100 capacity scenario remains; both original Python programs and their printed outputs are byte-conserved. Prediction questions and execution setup occur before the programs. The progression from four physical job intervals to area, probabilistic assumptions and deeper consequences is coherent; the optional tail/vacation branches keep their assumptions visible.

The independent research check read Sigman's pathwise conditions and clipped-area argument in [Notes on Little's Law](https://www.columbia.edu/~ks20/stochastic-I/stochastic-I-LL.pdf), pages 1–4. It also read the independent-service, residual and repeated-vacation derivations in [MIT Modiano lectures 8–9](https://ocw.mit.edu/courses/6-263j-data-communication-networks-fall-2002/65d9ab519ec4851af812f4b89ffaeedc_Lectures8_9.pdf), slides 2–9. The lesson's notation and stated conditions agree with those inspected portions. This reviewer did not watch the linked video; the author's resource ledger accurately distinguishes verified recording identity and inspected associated notes from full-video viewing.

## Complementary checks and resolved finding

Run from the repository with the existing scientific Python environment:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-queueing-independent.py
```

The script executes the actual exported lesson programs and imports the actual browser models. It does not invoke the author's model verifier or regenerate production data.

| Independent check | Covered cases |
| --- | ---: |
| Complete actual program stdout | 10 |
| Conserved original code/output blocks, plus archive source hash | 4 blocks |
| Admission-conditioned exponential stage counts, independently of Little's Law, including equal rates and extreme supported ratios | 20 |
| Tail inverse identities, zero-atom branch and displayed mass plus tail | 20 |
| Erlang-B recursion converted to pooled waiting probability | 12 |
| Exact rational event rectangles, tied arrival, zero service and censoring | 16 |
| Integrated residual survival, idle weighting and repeated-vacation decomposition | 9 |
| Changed capped Pareto survival integrals | 12 |
| Paired seeded simulation draws with a relative-workload oracle | 4 |

The original simulation accepted arrival rate `1e-6`, service rate `1e6`, seed 17, 100,000 warm-up customers and 100 measured customers. It accumulated arrivals near a very large absolute time and added small service intervals to that clock. Floating-point rounding erased positive services. It returned mean total time `1.52587890625e-7`, whereas the same sampled services and a relative-workload calculation give `1.217625453856014e-6`, with no waiting. A second independently selected extreme case also failed. These were errors in the actual displayed helper, beyond ordinary Monte Carlo fluctuation.

The author changed the helper to the duration recurrence `W_i=max(0,T_(i−1)−gap_i)`, `T_i=W_i+service_i`, preserving the random draw order and avoiding large calendar timestamps. The repaired cases match the paired oracle; two ordinary changed cases also pass. All ten displayed outputs and original bytes remain conserved. The example now explains this arithmetic choice. The initial failure is preserved separately from the final corrected run rather than overwritten.

## Visual and evidence boundaries

This reviewer actually opened seven author captures under `scratch/queueing-reading-final/`: `timeline-320.png`, `censored-area-390.png`, `rare-long-320.png`, `tail-derivation-320.png`, `finite-buffer-320.png`, `reading-8-320.png` and `sources-390.png`. The time axis matches the event data; censored areas visibly separate completed and unfinished contributions; equal-scale residual triangles explain the second moment; the tail derivation fits; finite-buffer states and admission labels are meaningful. The ordinary pooling paragraph and source annotations remain readable. The wider residence table uses local scrolling, as documented by the author, rather than claiming all columns fit simultaneously.

The author's full 1440/390/320 interaction, keyboard, reset, anchors, code/output and font-enabled geometry results were read as evidence; they are not represented as independently rerun here. This review adds mathematical/source checks, actual screenshot inspection and the complementary executable cases above. The author's targeted final simulation check passed at 1440/390/320 with exact code, question, output and visible explanation, normal fonts and no document overflow or errors. This reviewer additionally opened `scratch/queueing-simulation-precision/program-390.png` and `explanation-320.png`; the recurrence and ordinary prose are readable. The author added a complementary exact-Fraction calendar-time oracle for six changed simulation cases; that evidence was inspected and is separately attributed in the durable record. The final complementary rerun matches all six amended production hashes. Parent registration and integrated builds remain separate.
