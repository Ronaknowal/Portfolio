# State Space Models — revision 4 implementation record

Date: 26 September 2026. Topic ID: `state-space-models-s4-mamba-mamba-2`. Requested mode: complete teaching revision in response to learner feedback. Both revision-4 phases are complete; production-browser integration review is recorded below. The revision-3 implementation record, original drafts and prior evidence are preserved unchanged.

## What changed and why

The revision rebuilds the route throughout the lesson. It begins with a short observed stream and compares reasonable memory choices, then lets the reader manipulate a retained/new contribution before introducing h/u notation. A simpler impulse-trail ledger now precedes convolution, fast/slow summaries precede timescale equations, and a rotating real pair precedes complex modes. Mamba starts with a marked-event requirement and SSD starts with an actual two-by-two write/read. The measured classifier then closes the loop between a designed memory rule, a learned task, code and evaluation.

The first-pass route identifies specific sections, labs and programs. Three complete deeper derivations are expandable: robust discretization, continuous impulse response and HiPPO/S4 kernel construction. Core assumptions remain visible. Numerical experiments, complete canonical programs, all previous exercises and primary resources were retained; the authored tutorial resource is added with a precise statement of what was reviewed.

## Source ownership

- Current complete manuscript/design/specifications: this revision directory.
- Current generator: `scripts/generate-state-space-lesson.mjs`, explicitly reading revision4. Removed obsolete repairs for historical prediction prompts because the new manuscript carries the actual current copy.
- Generated lesson: `src/learn/data/topics/state-space-models-s4-mamba-mamba-2.jsx`.
- New small figures: `StateSpaceIntuition.jsx` and `state-space-intuition.js`, with scoped styles in `state-space-labs.css`.
- Unchanged numerical model, prior figure calculations, four main labs, public Python programs, fitted weights, source data and measured outcomes keep their original owners. No duplicate trained artifacts or new training run was created.

## Verification performed

`node scripts/generate-state-space-lesson.mjs` and `node scripts/verify-state-space-teaching.mjs` pass. The new source-bound receipt is `evidence/teaching-checks.json`. It checks:

1. Default and endpoint retention, arithmetic decomposition, whole-history means and101 fractions against independently expanded signed-input sums.
2. Impulse trails, future absence versus actual zeros, column sums and separate recurrent evaluation.
3. Marker retention, delay, constant smoothing, variable distraction gaps and a no-marker null.
4. The outer-product/state/read fixture against hand arithmetic and the previously verified SSD operator; .8 retention versus continuous step.
5. JSX parsing, the complete12-section route, all19 standalone visual groups, four existing investigations, three complete code readers, all original displayed code blocks, nine end exercises plus changed-code task, technical boundaries and external references.
6. Exact identity of the previous model/lab/data/weights/Python sources before reusing their old numerical/native evidence. Old checks are not misrepresented as review of the new prose or layout. No CUDA execution is claimed.

The conservation checks compare practice content after whitespace normalization only; they do not validate a summary instead of the full solutions. The new arithmetic checks exercise operations, not merely asserted source strings. Structural counts are supplemental and do not stand in for a teaching review.

## Independent reading review

The complementary LongContext author read the entire revised manuscript and new figure/data files, explicitly assessing learner flow through every architecture section. Their source-bound receipt is `evidence/independent-review.json`. They independently checked the new worked arithmetic and confirmed the retained technical distinctions. One minor caption issue (“Both matrices” despite showing more than two) was corrected to “Every matrix.” No substantive open issue remained within that review's scope. This does not claim an independent complete browser or native-training rerun.

## Production-browser review

The author initially had no available CUA surface. The integration owner subsequently inspected the current production build through its working browser bridge at port 4194. All four new figures were inspected at desktop width; the write/read matrix stacked at 760px; retention, matrix and bounded figure widths were checked at 320px. Wide tables keep local scrolling without widening the page.

At step 1, changing retention from .8 to .5 changed state from .8 to 1.25; retention 1 preserved the zero initial state; reset restored .8. ArrowRight changed .8 to .81 and produced .7695. Selecting impulse column 3 displayed .25 + 0 + .5 + 0 = .75. The continuous-impulse-response disclosure opened and closed, and existing lab access remained present. No new native training or GPU execution is implied by these UI checks. See the [root browser receipt](../../../evidence/attention-memory-intuition/browser-review.json) and [shared integration record](../../../ATTENTION-MEMORY-INTUITION-REVISION.md).
