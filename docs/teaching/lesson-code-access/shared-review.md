# Shared code access: independent source review

28 September 2026. Read-only review of `content/Code.jsx`, `RemoteCodeBlock.jsx`, `LessonCodeDownloads.jsx`, `code-downloads.js`, `lesson-code.css`, `TopicContent.jsx`, `RunnableExample.jsx` and `MechanismProgram.jsx`. Supporting topic/component sources were read only to establish concrete cases. No runtime, source, ledger or test files were edited in this review.

## Findings sent to the integration owner

1. **P2 — canonical Gaussian Process assets are omitted from the common index.** `content/code-downloads.js:24` admits `/learn-code/` and `/learn-assets/` but excludes `/learn/examples/`. The four existing links in `data/topics/gaussian-processes-gp.jsx:345` point to `gp_conditioning.py`, `co2_gp.py`, `mauna-loa-monthly.csv` and `README.md` under that prefix; all four files exist in `public/learn/examples/gaussian-processes-gp/`. A direct call to `localCodeAsset` with the first URL returns `null`. Include this existing canonical namespace while retaining same-origin and extension restrictions.

2. **P2 — in-body solution downloads appear outside the practice group.** `content/Code.jsx:26`, `RemoteCodeBlock.jsx:10` and `LessonCodeDownloads.jsx:50` recognize final exercise/section classes, `lesson-check` and `neural-practice`, but omit native practice disclosures without those ancestors. `arrays-strings-hash-maps.jsx:102` renders a `PythonExample` inside “Complete solution and why it works”; `linked-lists-stacks-queues.jsx:90` and `:98` do likewise inside their in-body solutions. Their code registers with `practice: false`, so the common index bypasses the answer's disclosure. Preserve these authored native feedback boundaries through an explicit context or a structural rule compatible with the now-visible teaching sections; do not derive practice from prose labels.

3. **P2 — Python example download names disagree with its save/run instructions.** `lesson-labs/PythonExample.jsx:7` knows the filenames in `example.files`, and `:10` has `example.filename`, but neither passes that value to `CodeBlock`. New downloads therefore receive generated snippet names even when the component tells the learner to save and run a particular program. This also breaks a multi-file example if its imported companion module is downloaded under a generated name. Pass existing canonical filename metadata to the common code block; retain generated names only when no real filename was authored. This component belongs to the parallel component owner.

## Bounded review result

The reviewed core does not execute displayed code: copy reads its exact string or rendered text, snippet download creates a text Blob, and canonical download links retain their source URL. `RemoteCodeBlock` resets its inner component by source, aborts outstanding fetches on cleanup, rejects failed/HTML responses, exposes retry, and loads near the viewport. Stable registration callbacks and separate registration/entry contexts avoid an evident render/effect loop. Output blocks are excluded from the default index, and provider state resets with topic identity.

No additional actionable issue was found in that bounded lifecycle/CSS review. Browser clipboard, downloaded-byte, fetch-error and navigation checks are owned by the integration tests; this report does not claim those executions. The known fallback filename duplication outside the active lesson provider is outside this published-lesson scope.

## Resolution check

All three findings are resolved in the reviewed source:

- `localCodeAsset` now accepts `/learn/examples/`. A focused assertion accepts all four existing Gaussian Process asset URLs and still rejects a cross-origin URL.
- `CodeBlock`, `RemoteCodeBlock` and the contextual asset scanner now recognize native `details` ancestors as well as the explicit practice markers. The three in-body solution examples therefore retain their disclosure boundary in the index. The index labels this group “Practice and optional files”, which also describes retained optional disclosures without inferring their role from prose.
- `PythonExample` forwards each companion filename and its main `example.filename || example.file`; `TerminalExample` forwards the same main-file metadata. Both derive separate output filenames while output remains outside the default file index. Existing save/run instructions and program filenames now agree.

No unresolved findings remain in this bounded shared-runtime source review. This resolution check inspected the final source and exercised the canonical URL helper; the integration receipt owns the browser, download-byte, retry and production-build results.
