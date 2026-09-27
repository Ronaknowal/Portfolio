# Titans: source-bound implementation design

The full prepared twelve-section manuscript, eight changed practice cases and its
complete program explanations remain in the production source. Authoring prose is
compiled once; no Markdown parser is shipped. The five investigations address
different mechanisms, with no prediction-entry or answer-unlock flow.

## Concept-to-representation map

| Learning transition | Concrete intuition and mechanism | Runtime representation |
| --- | --- | --- |
| Three memory lifetimes | Moving last-two window versus compressed learned rule versus session-fixed prefix | `TitansLifetimesFigure`: seven record cells, changing brackets, contextual weight grid, separate prefix dependency |
| Associative write and interference | A key coordinate selects the weight a residual can change; overlap selects the later affected read | `TitansWriteFigure`: key→weight coordinate edges and read ruler; orthogonal/correlated vector planes; `TitansWriteLab` actual key/query/target edits, current state and next-write preview |
| Loss is not gradient or semantic importance | Zero key retains loss but removes the parameter path; changing residual direction can change Jacobian sensitivity | §3 equations and zero-key example, live zero-key preset, separate prewrite loss/gradient/update readouts |
| Momentum, forgetting and reset | New movement adds remembered movement and old-weight shrinkage | Signed decomposition bars, scalar worked update, complete-reset state categories; explicit live zero-new-gradient and full-decay presets |
| Stability and normalization | The fixed-key error multiplier can contract, alternate or amplify | Four exact residual trajectories in `TitansStabilityFigure`, manuscript derivation, projection diagram shows normalization only on q/k |
| Nonlinear memory | Hidden nonlinear features remove an affine XOR contradiction | Connected shape-aware network with backward parameter path; signed XOR corners, full explained `neural_memory.py` |
| Learned views and local history | A single observed token supplies distinct read/write roles | `TitansProjectionFigure` shows Q/K/V learned views, causal convolution and query/key normalization; slow projection weights distinguished from request history |
| MAC / MAG / MAL | The retrieved representation can join input rows, an output gate, or a previous layer | `TitansJunctionsFigure`: explicit current-segment bypass, historical read, attend→write→postwrite-read MAC route, parallel MAG branches, serial MAL route, state continuation arrows |
| Causal state ownership | Continuing a request carries both learned state and its recent rows | `TitansGatedLab`: actual softmax rows/weights, W/S matrices, long read and coordinate product, full output table, carry/fresh suffix and independent request B |
| Inner versus outer learning | A write can itself be trained using its effect on a later query | Full dependency diagram, scalar derivative in prose, live nonlinear retained/detached graph comparison with identical values and separate finite functional sensitivity |
| Real observation availability | Forecast first; the target may update memory only when it arrives | Actual date timeline, all three-seed paired MAE panels and baselines, real rental replay with one editable arrived count; selected-date presence/availability table |
| Sequential versus anchored chunk gradient | The second gradient may see either a changed weight or one shared anchor | `TitansChunkLab` dependency geometry and complete two-step table; editable targets, rate, start, chunk1/2; highlight selects inspection, never access to an answer |
| Affine scan | Compose already chosen updates without silently changing gradient dependencies | Actual .5/2 then .25/−1 maps and combined .125/−.5 map in `TitansScanFigure` |
| State cost | Per-request fast weights, momentum and K/V scale separately from model weights/workspace | Additive equal-scale .75 GiB bands, exact BigInt byte table including full48 GiB cache; omitted categories explicitly unspecified |
| Research transfer | Change an objective or use an available auxiliary target, then name what was tested | Half-square versus Huber derivative curves (illustrative); rotated-shape auxiliary target graph; original benchmark/context distinctions remain prose, no fabricated benchmark curve |
| Code customization and changed practice | Translate state and graph contracts into complete runnable functions | Four complete on-demand programs; local explanations preserved; graph-retention task and eight worked practice solutions in closed details |

## Numerical and data evidence

`check-titans-memory-native.py` executed the complete prepared author checker in a
temporary copied packet, preserving historical outputs. It replayed all three
saved initial parameter sets through 366 days each; no new fits or selection were
performed. Native CPU float64, one thread. It also evaluated 24 NumPy linear writes,
18 MAG sequences, nine nonlinear dimension/seed combinations, four retained versus
detached outer derivatives, and three changed arrival histories. The downloadable
programs are exact copies of the canonical sources.

`check-titans-memory-models.mjs` independently ports the analytic linear update,
two-layer SiLU derivatives, softmax gate, arbitrary scalar chunk rules and native
rental replay. 11,995 scalar comparisons pass; maximum absolute error is
9.94176962976212e-13. Checks include complete final parameter/momentum arrays,
full-state continuation, no caller mutation, future exclusion, zero-update equality,
closed polynomial chunk formulas and exact integer payload arithmetic. The
browser's real input change is an arrived observation, not an editable forecast;
the unchanged same-day prediction is a tested causal null.

The 460,722-byte full result is a learner download, not an eager import. A small
lossless display module owns the 731 dates/counts, all metric summaries and one
2→3→1 initialization. Only the selected seed's model/trace is fetched when its lab
intersects the viewport. Seed changes and retries preserve the edited date/count
and inspected date above the loading boundary. Cleanup aborts stale requests.
Downloads use an explicit ten-file allowlist; no manuscript/spec/design/review
documents are deployed.

## Author learning-experience checklist

1. **Need before naming:** maintenance-log and storage-lifetime problem precedes the
   architecture names; each later subsection retains its prepared motivation.
2. **Intermediate reasoning:** read/write values, residual/Jacobian, momentum sum,
   nonlinear hidden shapes, gating terms and chunk evaluation weights remain visible.
3. **Meaningful investigation:** actual keys/targets/tokens/arrived counts are editable;
   controls alter the relevant mathematical entities immediately.
4. **Comparison and null:** orthogonal/correlated, zero key, momentum-only, full decay,
   prefix removal, no writes, carry/fresh, independent request, chunk1/rate0, graph
   detach and unchanged same-day forecast are concrete checked cases.
5. **Fixture quality:** defaults show nonzero first-write movement, nontrivial second
   token prefix effect, distinct chunk outputs, a nonzero outer derivative, and all
   real seeds including seed7's failure to beat the simple assessment baseline.
6. **Figure perceptibility:** signed coordinate axes and native-size focusable SVG
   containers, exact tables, shared paired-MAE axes and enlarged rental-difference
   plot. Painted browser observations remain the integration owner's responsibility.
7. **Real evidence:** full recorded experiment retained; no smoothed or invented
   metrics. New-arrival edits are sensitivity comparisons and do not claim accuracy.
8. **Complete code and practice:** scratch recurrence, full autograd memory, full
   real study and reproduction checker on demand; all original code explanations and
   changed practice remain. Display source never includes test harness scaffolding.
9. **Usability:** stable numeric controls, local table/chart scrolling, keyboardable
   controls, no animation loop, immediate outputs; friendly load/retry and preserved
   edits. SSR passes; actual keyboard/mobile/invalid-input/reset remain root checks.
10. **Boundaries:** no Titans pretrained checkpoint or language benchmark is executed;
    MAG is the explicitly declared teaching specialization and rental adaptation is
    supervised after observation. Browser and independent review remain separate.

## Primary sources revisited for implementation

- [Titans v1](https://arxiv.org/html/2501.00663v1), §§3.1–3.3 and4.1–4.4: precise
  update, anchored chunk gradients, persistent prefix and architecture data paths.
  The manuscript deliberately qualifies the informal full-forgetting and surprise
  descriptions using their actual equations.
- [PyTorch2.14 autograd.grad](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html):
  create_graph, allow_unused and disconnected graph behavior; verified natively.
- [Google Titans/MIRAS overview](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/),
  dated4December2025: four design axes. The objective and vision illustrations are
  labeled conceptual transfers, without claiming those experiments were run here.

The primary excerpts were inspected during this implementation. Earlier full
prepared research/attribution records remain authoritative for their dated scope.
