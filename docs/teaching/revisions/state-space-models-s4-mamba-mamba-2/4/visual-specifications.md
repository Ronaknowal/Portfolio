# State Space revision 4 — visual and live-learning specifications

This packet preserves the implemented revision-3 visual mechanisms and four investigations. The original draft's obsolete “run/prediction” wording does not apply: valid edits show results immediately, no learner guess entry exists. Existing canonical programs, model weights and data remain source-owned by their public assets; browser model inference is unchanged.

## New representations before formalism

| Visual / placement | Entities and operation | Exact contract / learner action |
| --- | --- | --- |
| Retain/write, §1 before h/u | Observations [5,0,0], zero initial summary; split one update into retained old value and weighted current observation; two bars share scale 0–5; trace compares latest-only, full-history mean and fading summary | Slider + numeric retained fraction [0,1], step .01; three selected-update buttons; reset .8/step1. Default states [1,.8,.64]. Fraction0 returns inputs; fraction1 from zero stays zero. Mean [5,2.5,5/3] is a separate task, not a fitted reference. No measured accuracy claim |
| Overlapping impulse trails, §3 before kernel equations | Birth-time rows; time columns; same lag weights [1,.5,.25,.125]; inputs [2,0,1,0]; bars represent contribution with common 0–2 scale | Output-column buttons highlight corresponding cells and their exact sum. Outputs [2,1,1.5,.75]. Cells before input birth say “not yet”; actual zero contributions say 0. Horizontal scrolling stays inside keyboard-focusable table region |
| Marked memory, §5 before selective equation | Same observed values [4,9,−7,6]; first/last events explicitly marked. Compare content-based replacement, fixed two-step delay and half-retention update | Marked state [4,4,4,6]; fixed delay [0,0,4,9]; constant smoothing [2,5.5,−.75,2.625]. Static event strip and exact table teach distinct questions; the later existing selection lab supplies free controls. Markers/gates are supplied, not learned |
| Matrix write/read, §6 before SSD symbols | Old [[2,1],[0,0]], shared decay .5; new value [3,−1], write weights [0,1]; retain + outer write; read weights [1,1] add rows | Retained [[1,.5],[0,0]], write [[0,0],[3,−1]], state [[1,.5],[3,−1]], output [4,−.5]. Memory rows/value-feature columns labeled; values remain readable as small semantic matrices at 320px. This is exactly step1 of the retained four-step example |

Each representation has preceding “what to inspect” text and following interpretation. Fractions/bars do not imply probability certainty or training success. Numeric outputs are calculated from `state-space-intuition.js`, with independent arithmetic checks. No timer, new dependency, remote request or large allocation is needed for these small views.

## Existing figures placed more usefully

- Four-path state picture now follows numerical write/retain/read/feedthrough arithmetic and precedes the general matrix equation.
- Continuous decay/held-input plots precede ODE notation. Core prose tells the learner to inspect solid exact curves first; the approximation is explained in the optional discretization branch.
- Existing two-mode contribution ledger follows the new simple impulse ledger as a deliberate extension, retaining the original initial-response and FFT details.
- Timescale and equal-axis rotating-pair figures now precede half-life/complex notation. Real coordinates remain the bridge to complex storage.
- Polynomial reconstruction and DPLR stay inside a clearly labeled return branch, with purpose explained before the algebra.
- Fixed delay, supplied selective gates, full Mamba branch diagram, signed SSD influence/chunk displays, all actual trajectories, training-flow/curves/confusions, byte counts and three Mamba-3 comparisons remain available.

## Preserved live investigations and programs

1. System laboratory: current input heights, continuous rates/B/C/D/Δ/initial state, singular and direct-path cases; independently implemented recurrence/direct/FFT with exact current state and contributions.
2. Selective memory: available values and markers, gates, constant comparison, and independent retention versus write. Current prose asks learners to change an actual distractor before altering its gate.
3. SSD workshop: b/c/v/decay/initial state, chunk size and length; current prose isolates future-value causality, same-operator regrouping, and write/read differences.
4. Fitted trajectory workbench: 50 validation rows, two seed17 models, actual point dragging/numeric coordinates, order interventions, internal response and all15 class probabilities. Edits infer immediately without retraining. Underlying test evidence and source IDs are unchanged.

The three on-demand full-program readers and canonical downloads remain: `state_space_mechanisms.py`, `trajectory_state_models.py`, `state_space_library_bridge.py`. CUDA package route stays explicitly source-checked/unexecuted on GPU. Existing native evidence is reused only for unchanged files verified by hash.

## Rendering and verification

Black/charcoal with amber, neutral labels and existing signed-quantity palette. Controls have at least44px height, labels, visible focus and no optional prediction fields. Marks are accompanied by text, not color-only meaning. No new animation or drag surface is introduced. Tables may scroll locally; the page may not overflow. Event cards become two columns at narrow article widths. The matrix decomposition stacks while its individual2×2 arrays remain legible.

Test default/fraction0/fraction1/reset and keyboard step selection; exact contribution columns; static marked traces and matrix outputs; open/close all three deeper branches; retention of all old lab and code routes. Inspect desktop, intermediate sidebar-constrained width and320px. New arithmetic/conservation/source bindings live in `evidence/teaching-checks.json`; authored browser/reviewer findings live in `implementation.md` after execution.
