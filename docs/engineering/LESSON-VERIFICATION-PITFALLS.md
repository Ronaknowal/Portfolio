# Verification lessons worth carrying forward

Condensed on 24 September 2026 from the completed lesson batches formerly repeated
in the main authoring handoff. This is a reference for the relevant implementation
or verifier change, not an instruction to reopen all completed lessons. The
[teaching standard](../../LESSON-TEACHING-STANDARD.md) and
[code standard](LEARNING-CODE-STANDARD.md) remain authoritative.

## Check the rendered mechanism, not only the numbers

- Correct DOM, formulas and offline calculations can coexist with incorrect paint.
  Compare computed CSS and actual rendering with the intended attributes. CSS can
  override SVG stroke widths, fill or font sizes; a missing presentation attribute
  is a case to inspect, not a reason to skip the element.
- Never apply broad descendant `svg { height: auto }` rules under a root containing
  KaTeX. They can collapse radicals/accents and change the mathematical meaning.
  Scope figure geometry with a dedicated class. Inspect both dimensions and painted
  notation; zero `.katex-error` elements proves neither geometry nor legibility.
- Exercise actual reachable states, including zero mass, zero step, exact ties and
  limits. Displayed rounding, control increments and the applied decision rule must
  agree. A fix at one example is insufficient when the same discrepancy is reachable
  elsewhere. Assert the relationship across the relevant domain.
- Trace what the browser really paints: `route.path` may differ from `route.trace`.
  Include the actual shape types in collision checks. Curves need sampling beyond
  straight-line intersections; the existing bias-variance verifier has a scoped
  `sampleCurvesThroughLabels` implementation. Promoting a stricter shared check is
  separate scoped work, not a reason to rerun every closed lesson now.
- Verify captions, legends, units, axis visibility, real versus constructed data,
  and practice inputs against the actual artifact. A named hatch must be drawn; a
  practice value must be enterable; constructed cases are not validation evidence
  from a real dataset. Keep an honest real-data investigation where the topic needs it.

## Make the verification independent and capable of failing

Use a different derivation, representation or theorem when possible, rather than
another call to the same implementation. Check the trust-root data that other
checks consume. State which values were independently re-derived and which were
pinned to an existing reference. Treat ties, numerical margins and robustness of
the conclusion separately; do not rewrite a preserved correct packet merely to
hide a legitimate alternative tie resolution or documented rounding difference.

Test new substantive guards against representative deliberate faults in an isolated,
restorable setting. Known traps include a heredoc turning regex `\b` into a literal
backspace, an inequality that accepts its own regression, wrong units, an empty
subject set, a DOM pattern never produced, a state never reached, checking only
width for a collapsed shape, comparing against the inset being guarded, guessed
thresholds, enumerating success markup instead of the failing property, and checking
text nodes separately when React splits one visible expression across nodes.
Leak checks need the page's actual numeric range and text formatting. Widening a
guard may create a new blind spot; check both shapes. A fault must remove the
property, not just one of its equivalent implementations.

Assert that the subject set is nonempty and report real coverage: failable assertions,
scan passes, checked leaf paths and exclusions. Do not count `check(true)` or
self-comparisons as verification. Commit a reusable check before claiming future
agents can run it. Diagnose process/runtime failures with an experiment before
stating an explanation. This guidance is proportional to the substantive change,
not a mandate to add mutation campaigns for cosmetic edits or cleanup.

## Protect evidence and the shared workspace

A source-mutating falsification harness needs a durable recovery sidecar, an actual
single-instance lock, unique mutation anchors, narrowly owned file targets, and
restoration of evidence as well as source bytes. On Windows a forced kill may deliver
no catchable signal; a handler alone is insufficient. Refuse ambiguous anchors.
Counting text occurrences before/after is unreliable when a mutation wraps its
find-string. Recovery state should say whether a case is running, whether the
process is alive, and whether sources **and the build directory** were restored.

Write provisional failing evidence first; mark success only after every required
check. Support a non-overwriting/no-evidence mode for investigation. Narrow reruns
must not silently replace broader receipts or delete another lesson's captures.
Record hashes and byte sizes for important screenshots, not only filenames.
Corrections to past claims should be explicit; do not falsify frozen evidence to
make it look as though a different check ran.

Isolate optional Python libraries when their resolved versions differ from the
shared teaching environment. Record actual versions and distinguish tested adapters
from unexecuted experiments. Never upgrade a shared runtime to get one lesson to
pass. Under Windows, MSYS/Git Bash PIDs, `kill -0`, and shell path conversion can
misrepresent native process liveness. Use a reliable native check such as PowerShell
`Get-CimInstance Win32_Process` or `spawnSync('tasklist', args)` without `shell: true`.
Do not assume a stale-looking lock permits recovery over a live run. Coordinate
shared builds and parse the lesson's owned JSX before integration.

Older records discuss prediction-entry and grading gates. Those lab designs are
superseded: current labs provide immediate live exploration, with **no learner-
prediction feature, even optional**. Preserve scientific model predictions and
separate practice, but do not revive obsolete answer-unlock UI while reusing a check.

## Original evidence

The topic ledger links the full designs, retained source and independent reviews.
Useful concrete cases include the [final Classical ML audit](../teaching/CLASSICAL-FINAL-TOPICS-AUDIT.md),
[Feature Selection review](../teaching/FEATURE-SELECTION-INDEPENDENT-REVIEW.md),
[Bias-Variance review](../teaching/BIAS-VARIANCE-INDEPENDENT-REVIEW.md),
[HMM review](../teaching/HMM-INDEPENDENT-REVIEW.md) and
[Bayesian Networks review](../teaching/BAYESIAN-NETWORKS-INDEPENDENT-REVIEW.md).
Consult the current record before treating an old finding as still open.
