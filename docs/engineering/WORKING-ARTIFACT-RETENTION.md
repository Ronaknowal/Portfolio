# Working files: keep useful evidence, remove finished working material

The user's 11 September 2026 cleanup request applies to `scratch/` and temporary artifacts elsewhere in the workspace. Cleanup is part of finishing work, not a new recurring audit of every lesson.

## What belongs where

| Material | Location and lifetime |
| --- | --- |
| Lesson content, figures used by the site, models and examples | Topic-owned `src/` files and actual runtime assets under `public/`; keep them. |
| Reusable checks and example generators | Semantic files under `scripts/`; keep their documented inputs, avoid one script per tiny correction. |
| Content-first manuscript and visual/lab specifications awaiting phase two | `docs/teaching/drafts/<stable-topic-id>/`; keep while referenced by the delivery ledger. They are required handoff inputs even when another agent/session will implement them. |
| Current drafts, datasets and temporary execution outputs | One topic-owned directory under `scratch/`; keep while the active task needs them. Remove installed duplicate drafts and obsolete patch/install scripts. |
| Final numerical evidence, review summaries, source hashes and original baselines | Durable records under `docs/teaching/`; retain the necessary original inputs and linked final attachments. |
| Browser screenshots | Capture a useful default and selected changed/edge states at relevant widths. Keep images supporting the final visual review and unresolved findings. Remove unused intermediate, failed, duplicate and superseded captures after closure. Do not retain every button-click screenshot automatically. |
| Library logs, caches and build outputs | Use `scratch/<topic>/` or the tool's ignored output directory. Disable unnecessary training-file output where supported. Remove obsolete outputs when no live process or preview uses them; preserve installed dependencies and shared environments still in use. |

## At the end of a topic or bounded change

1. Record the requested phase's result and actual source version in the existing evidence record and central delivery ledger. Content-first completion does not finish the topic: preserve the pending manuscript/specifications and deferred-work instructions. Retire them only after integration and an updated checkpoint to retained equivalent sources or a useful archive. Keep the next action concise; completed checks are not a future task queue.
2. Decide which exact screenshots, native outputs and original inputs support that result. Prefer a small representative set with clear learning/verification purpose, without an arbitrary numerical quota. Keep additional states only when they establish a distinct claim or unresolved finding.
3. Check for active owners/processes, runtime imports, script inputs, documentation/evidence references and user-authored work before removal. An old timestamp, temporary filename, lack of Git tracking or lack of a direct reference alone does not prove a file is disposable. Dynamic paths need contextual inspection.
4. Remove confirmed disposable files. On Windows, resolve every target within the explicitly intended workspace directory, reject reparse points/path escapes, and use native literal-path file operations. Never delete a whole scratch tree or dependency environment by a broad wildcard.
5. Keep required attachments at their recorded paths. If retiring an obsolete attachment that has a historical reference, record its retirement or an exact accessible archive mapping; do not pretend a deleted attachment still exists. Do not rewrite frozen evidence to claim a new run.
6. Archive only material worth recovering; do not perpetually move all junk into a new folder. A compact recovery archive is appropriate for obsolete editing utilities when their history may matter. Generated unused screenshots usually need no archive once their final review evidence is retained.
7. Verify selected files are gone and protected files remain. Documentation/cleanup changes need these checks, not a new application build or all historical lesson tests. Remove the cleanup's own temporary inventory afterward and resume the authorized lesson task.

## Prevent obsolete instructions and mutation helpers from accumulating

Keep the active authoring handoff focused on current phase counts, the next eligible
work and links to canonical standards. Do not prepend a full completion narrative
after every batch. Record that narrative once in the topic/integration record;
distill a useful general lesson into the relevant standard or focused engineering
reference. Dated history must not require every future author to read it again.

Retire an unreferenced one-off script after its patch, source freeze or ledger update
is recorded. Especially remove helpers with fixed dates/status text or scripts that
would overwrite newer source, archives or handoffs. Reusable verifiers, required
inputs and recovery sidecars have a different lifetime: keep them while their
recorded purpose remains. A cache or duplicate working manuscript is not a pending
content packet merely because its filename says “draft”; use the ledger and retained
production owner to establish the difference before removal.

For a cleanup, record exact removed paths/reasons and any duplicate's retained
counterpart, then prove that runtime source, pending manuscript/specification packets
and ledgers were unchanged. Keep the small final record, not the temporary broad
inventory. Lack of a filename reference alone still does not authorize deletion.

## Optional inspection tool

`node scripts/audit-working-artifacts.mjs [output-json-path] [--protect scratch-folder-prefix ...]` produces a **read-only candidate report** for old scratch images, including direct references and byte-identical duplicates. Shared runtimes and dependency/cache directories are always excluded. Supply each active work prefix with `--protect`; the helper no longer assumes a historical Classical ML batch is still active. It does not approve deletion, interpret every dynamically constructed path or run any lesson checks. Use it only for an actual cleanup request or a concrete accumulation problem, not at every session start.

The teaching standard's bounded verification policy still applies: reuse passing evidence for unchanged code, rerun affected checks after relevant changes, and keep completed modules closed. Repository history and archived scripts are not authoring instructions.
