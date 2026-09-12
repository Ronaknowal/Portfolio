# Tracking content and implementation separately

Updated 12 September 2026. The [teaching standard](../../LESSON-TEACHING-STANDARD.md#delivery-modes-and-stopping-boundaries) owns the two delivery modes and quality requirements. This document owns the data contract and handoff procedure. It does not add another review campaign.

## One current phase ledger

[`lesson-delivery-progress.json`](lesson-delivery-progress.json) is keyed by stable topic ID. An absent topic means both phases are not started under this workflow; it does not mean the website has no older published article. The old module/increment ledgers retain their evidence, reviewed-source bindings and historical completion. They are not a separate current phase queue.

The initial migration records **107 previously completed topic revisions** with both phases complete. This is a migration of existing review records, not a fresh review or retrospective claim of checks never performed. `legacyReview: true` delegates source currency to the existing inventory review logic. The currently proposed K-Means revision is explicitly pending; its previous completed revision is preserved, and changed current source is reported as stale. Do not mark its new bytes reviewed just to make the migration counts uniform.

Each current entry has:

| Field | Meaning |
| --- | --- |
| `revision` | Positive integer for the current requested revision. Increment when starting a substantive new version. |
| `deliveryMode` | `full` or `content-first`. A later finish request continues that revision's original mode. |
| `record` | Repository-relative design/handoff record with scope, sources, checks, unresolved work and next action. |
| `content.status` | `not-started`, `in-progress` or `complete`. |
| `content.manuscript`, `content.visualSpecifications` | Paths identifying the actual complete written lesson and specification document. Both must be present in `content.files` for a new completed checkpoint. A generic blueprint/reference file alone is insufficient. |
| `content.files` | Nonempty repository-relative path → SHA256 map for the complete manuscript/specifications or full-mode content checkpoint. Required for new content-complete entries. |
| `implementation.status` | `not-started`, `in-progress` or `complete`. Anything beyond not-started requires content complete. |
| `implementation.reviewedFiles` | Nonempty path → SHA256 map of actual final lesson, examples, models, visual/lab files and relevant metadata. Required for new implementation-complete entries. |
| `nextAction` | Concrete continuation, including deferred implementation/review when only content was requested. |
| `content.completedAt`, `implementation.completedAt` | Optional actual completion timestamps. Do not invent historical dates during migration. |
| `previousRevisions` | Preserve earlier completed entries/evidence when opening a new revision. Keep history flat, without recursively nesting histories. |
| `legacyReview` | Migration marker only. Never use it to bypass checkpoints on new work. |
| `pendingRevision` | An explicit existing proposal or decision outside the historical reviewed revision, when relevant. |

User acceptance remains in the relevant user-review record; writing `complete` into either phase does not assert user approval. The ledger stores no student completion/progress and does not change the website's completion controls.

## Commands and validation

The usual topic command now returns `topic.delivery` with both effective phase states, recorded phase states, record/files and `canFinish`:

```sh
node scripts/build-curriculum-inventory.mjs --topic "pca-dimensionality-reduction"
node scripts/build-curriculum-inventory.mjs --topic "pca-dimensionality-reduction" --work content
node scripts/build-curriculum-inventory.mjs --topic "pca-dimensionality-reduction" --work full
node scripts/build-curriculum-inventory.mjs --topic "pca-dimensionality-reduction" --work finish
```

`--work` is a read-only preflight reflecting the user's requested action. It does not authorize work, create a draft or advance a status. Full implementation is one mode; content then finish is the other. Without `--work`, inspection remains read-only and reports missing prerequisites instead of throwing a finish-only error.

The finish preflight rejects missing, incomplete or changed content checkpoints. Schema validation also rejects implementation in progress/complete when content is not complete. If saved files differ from their hashes, the effective state is `stale`, while recorded historical completion is preserved. Do not repair that warning by blindly rehashing files: assess the actual change, reconcile the content handoff and run the affected verification before updating completion.

The inventory distinguishes current authoring revision from publication. While a new content draft awaits implementation, an unchanged old published lesson can still have its historical teaching-review status; `topic.delivery` makes the new pending work explicit. New implementation-complete entries contribute teaching review only when their content and reviewed-source checkpoints match. Generated delivery counts refer to current checkpoints, so they may differ from the count of historically completed revisions.

## Updating a topic

1. **Start from the user's scope.** Retrieve the topic and notes. Record full or content-first mode. For a new revision, preserve the earlier entry/evidence in `previousRevisions`, remove the migration marker from the new current entry, set content in progress and implementation not started. Do not create a new revision merely to resume an unfinished one.
2. **Complete content.** Write the actual lesson and precise visual/lab specifications. Record references, actual author checks, uncertainties, deferred runtime verification and intended production destinations in the handoff. Identify `content.manuscript` and `content.visualSpecifications` and save their SHA256 hashes, plus every other required handoff input, in `content.files`. Only then mark content complete. A missing core section, unresolved material contradiction or outline alone leaves it in progress.
3. **Honor the stopping point.** Content-first delivery now stops and reports “content complete; implementation not started.” Preserve its pending manuscript/specifications. Full delivery continues without a new permission question.
4. **Start implementation.** A later finish request must pass the preflight and consume the full manuscript, specifications and notes. Mark implementation in progress. Reuse research/evidence still valid for the relevant source; fill real gaps and correct the manuscript when necessary. Do not replay all prior research or silently drop specified learning outcomes.
5. **Close implementation.** Build the lesson and visuals/labs, complete the canonical review/correction/integration requirements, record actual final source hashes and mark implementation complete. Update the existing record and handoff. Preserve independent-review attribution and user-acceptance distinctions.
6. **Refresh status.** Run `node scripts/verify-lesson-delivery.mjs` and regenerate the inventory. Its `--topic` output must show the intended phases. These ledger checks do not replace the lesson checks required in phase two. Documentation/ledger-only changes do not require a production build or browser campaign.

Another agent needs the same complete packet, not a copy of the original conversation. The implementer may improve representation, terminology, structure or explanation where the topic benefits, recording substantive reasons and keeping content/checkpoints consistent. Completing phase one does not freeze incorrect material or forbid further research.

After successful integration, pending drafts can be retired only when the ledger points to retained equivalent content/specification sources or an intentionally retained archive. Preserve evidence required to interpret earlier checks. Never delete the only complete handoff because it lives outside `src/`.

## Minimal new entry

Before content is complete, an entry may be as small as:

```json
{
  "revision": 1,
  "deliveryMode": "content-first",
  "record": "docs/teaching/PCA-LESSON-DESIGN.md",
  "content": { "status": "in-progress" },
  "implementation": { "status": "not-started" },
  "nextAction": "Complete the manuscript and visual specifications; defer implementation."
}
```

This is a schema example, not an active PCA request or a claim that this example record exists. Use the actual topic's existing design filename. At completion, identify the manuscript/specification paths, add their real `content.files` hashes and change only that phase to complete. After phase two, add actual `implementation.reviewedFiles` and its completed status. Current source identity is required; copying another topic's hashes is invalid evidence. Schema/hash validation establishes a declared checkpoint's completeness and identity, not the teaching quality of its prose; the author must assess that substance.

## Migration verification, 12 September 2026

`node scripts/verify-lesson-delivery.mjs` passed eight behavior groups covering missing/incomplete content, valid handoffs, changed/missing specifications, separate implementation identity, historical completion, malformed records and protection against inheriting an old review during a new rewrite. The actual topic CLI allowed a PCA content request and rejected finishing it without content; Linux retained both completed phases; the changed K-Means working revision reported stale while preserving both historical completion fields.

Inventory regeneration passed with all 1,218 topic IDs, 28 modules, 228 publication entries and seven paths retained. At migration, the ledger contains 107 historically complete revisions and the effective inventory has 106 current complete checkpoints because of the separately recorded K-Means proposal. This is a dated result, not a fixed future target. No lesson, lab, live catalogue or publication mapping was changed by this workflow migration. Application build/browser checks were not rerun for these authoring-only changes.
