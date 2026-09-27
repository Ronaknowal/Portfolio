# Recurrent attention implementation

Completed 26 September 2026. This finishes revision 3 of
`attention-mechanism-bahdanau-luong`; its original delivery mode remains
`content-first`. The complete [prepared packet](drafts/attention-mechanism-bahdanau-luong/design.md)
and its content hashes remain intact. This is implementation review, not user acceptance.

## Teaching and code delivered

The complete manuscript becomes eleven navigable sections. Its route moves from
a revisitable source memory to query/key/value arithmetic, scorer versus decoder
schedule, masks, gradients, real inflection experiments, alignment interpretation,
implementation and deeper local/copy/location mechanisms. All eight separate
practice hint/solution pairs and the changed-temperature implementation task remain.
Annotated papers, articles, documentation and alternate explanations are retained.

Eight inline figure families show the source shelf, value geometry and signed
contributions, decoder timelines, gradient path, measured learning curves,
recorded alignment, ambiguous contexts and location features. Seven investigations
provide immediate controls: memory editing, scorer cancellation, padding faults,
learning updates, full fitted decoding, local windows and vocabulary/copy mixing.
There is no learner prediction form or answer gate.

The self-contained NumPy route implements GRU gates, source projection/cache,
stable masked softmax, the two decoder orders and full generation using the same
saved weights as the PyTorch route. NumPy supplies array operations, not a model.
The maintained-library route owns its attention operations and composes them with
PyTorch embeddings, GRU, loss and optimizer. Prior recurrent gate derivations are
linked instead of duplicated. The original complete training program, analytic
gradient/window/copy calculations, data, parameters and extraction provenance are
downloadable. Displayed complete programs fetch those exact canonical files.

Model weights load only when their fitted investigation is opened. Only the chosen
architecture loads; encoding and projected keys are memoized, and prefix edits
replay only the bounded decoder. The source accepts 3–8 lowercase characters,
at most two PAD slots and sixteen output steps. No training runs in the browser.
The full 12.9 MB measurement archive is a download, not a lesson import. Topic
code remains a separate lazy chunk. Inputs retain the last valid calculation
while an invalid edit is explained; empty legal masks are prevented explicitly.

## Corrections made during implementation and review

- Removed two obsolete learner-prediction prompts from production prose.
- Replaced the incomplete attention-work expression with source projection,
  query projection, scoring, value-mixing and cache costs, stating excluded work.
- Added the complete self-contained NumPy saved-model inference route, with its
  mechanism explained beside the code and a checked native counterpart.
- An independent reviewer found that a faulty padded run can outlast its clean
  reference. The inspector now states that the correctly masked run already ended.
  It never formats that absent comparison as an undefined number.
- Numbered dependency lists now retain their explicit sequence after intervening
  paragraphs. A focused renderer regression check covers that case.
- Phone review simplified window-card labels, prevented checkbox shrinking and
  increased scratch-explanation spacing. Container queries orient dependency
  arrows vertically when the actual article is narrow, including a 760px viewport
  with the sidebar still present.
- The published provenance's split-integrity link now points to a retained local
  copy of the real preceding audit. Frozen scientific draft and data bytes remain
  unchanged; the generator records this publication-only link adaptation.

## Executed correctness checks

`node scripts/verify-recurrent-attention-models.mjs` passes five groups: all ten
calculated reads/gradients/updates; local-window boundaries and normalization,
copying and cancellation; sixteen complete saved traces; manuscript/practice
integration; and ordered-list continuation. Maximum saved NumPy/JavaScript
trace difference is 2.8311e−15. Additional source/prefix/PAD cases use `prone`,
`third_person`, a forced `ab`, output cap one and masked storage.

`scratch/lesson-tools/Scripts/python.exe -X utf8 -B scripts/verify-recurrent-attention-native.py`
passes four groups under Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu.
All six fitted models reproduce their 447 development outputs and reported
parameter/error counts. Fourteen fresh scratch/native cases have maximum output
probability difference 2.29991e−7. Native autograd and finite differences check
the analytic derivative. Both learner entry programs actually print `lactated True`.
The frozen fits are reused; no new training or model selection was performed.

Evidence: [model checks](evidence/recurrent-attention-models.json),
[native checks](evidence/recurrent-attention-native.json),
[browser observations](evidence/recurrent-attention-browser.json), and
[independent review](ATTENTION-INDEPENDENT-REVIEW.md).
The independent reviewer checked the full manuscript and implementation, fresh
finite-difference/permutation/prefix cases, the primary Luong equations and the
actual padding defect in the built UI. Final source hashes bind that separate review.
The implementation refresh also checked the official PyTorch 2.14 GRU contract;
resource metadata is not represented as a watched video.

## Rendered and learning-experience review

The production preview was inspected at 1280×720, 320×844 and 760×900. All eight
figure families were inspected on desktop and phone; changed states and dense
tables/matrices were checked as well. The document stayed within its viewport.
Wide matrices and code scroll locally; eleven section links resolve and KaTeX
reported no errors. Actual keyboard, numeric, checkbox, selector and pointer
interactions were used, including a phone drag of the copy-mixture slider.
The seven labs expose their computed starting state, a meaningful changed case,
an appropriate null and deterministic reset. The browser record retains concrete
inputs and observed results. Screenshots were inspected inline in the tool session;
no nonexistent PNG archive is claimed.

The author learning review found a coherent first-pass question, terms defined
before use, local diagrams at representation changes, complete explained code and
transfer practice. Real measured curves retain unfavorable results and their rule
baseline. Equal-value and same-context cases make the limits of heatmap interpretation
visible. Cautions have specific homes rather than replacing the main explanation.
This is an author/reviewer heuristic review, not an observed novice or assistive-
technology study. Network failures were not injected; abort/retry handling was
source-reviewed and successful resource loading was exercised.

The initial development server stalled independently on the home route and failed
module fetches. It supplied no verification evidence. The working production
preview on port 4196 supplied the checks above. No production-preview console error
was observed; earlier development errors remain distinguishable in the session log.

## Continue

Both phases for this revision are complete after the shared integration checkpoint.
The next teaching topic is Long-Context Sequence Models, then State Space Models.
After this three-topic request, continue with RWKV in actual module order only
when the user requests more implementation. Pending packets remain preserved.
