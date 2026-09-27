# Documentation organization

Completed 28 September 2026. This was a documentation and authoring-path cleanup, not a lesson rewrite or a new scientific review. The user requested that detailed Markdown files leave the repository root and allowed removal of obsolete material already represented by the skill.

## Current owners

The root keeps `README.md`, `AGENTS.md` and the conventional hidden `.impeccable.md` design context. [The documentation directory](../README.md) indexes the current handoff, curriculum, project guidance, engineering contracts and historical records. Reusable lesson instructions stay in [the skill](../../skills/lesson-authoring/SKILL.md). The installed personal skill links to this versioned package.

## Relocated documents

These files retain their substantive content. Relative links were rebased; the portfolio design reference now explicitly identifies its portfolio-only scope. Completed rollout reports are history, not future task queues.

| Former repository-root path | Current location |
| --- | --- |
| `LESSON-AUTHORING-HANDOFF.md` | [LESSON-AUTHORING-HANDOFF.md](../teaching/LESSON-AUTHORING-HANDOFF.md) |
| `LEARNING-CURRICULUM-PLAN.md` | [LEARNING-CURRICULUM-PLAN.md](../curriculum/LEARNING-CURRICULUM-PLAN.md) |
| `PROJECT-AUTHORING-STANDARD.md` | [PROJECT-AUTHORING-STANDARD.md](../teaching/projects/PROJECT-AUTHORING-STANDARD.md) |
| `DESIGN REFERENCE.md` | [PORTFOLIO-DESIGN-REFERENCE.md](PORTFOLIO-DESIGN-REFERENCE.md) |
| `CLASSICAL-ML-SUPERVISED-IMPLEMENTATION.md` | [CLASSICAL-ML-SUPERVISED-IMPLEMENTATION.md](../archive/lesson-rollout/CLASSICAL-ML-SUPERVISED-IMPLEMENTATION.md) |
| `CURRICULUM-EXPANSION-VERIFICATION.md` | [CURRICULUM-EXPANSION-VERIFICATION.md](../archive/lesson-rollout/CURRICULUM-EXPANSION-VERIFICATION.md) |
| `DSA-CORE-STRUCTURES-IMPLEMENTATION.md` | [DSA-CORE-STRUCTURES-IMPLEMENTATION.md](../archive/lesson-rollout/DSA-CORE-STRUCTURES-IMPLEMENTATION.md) |
| `DSA-MATH-FOUNDATIONS-IMPLEMENTATION.md` | [DSA-MATH-FOUNDATIONS-IMPLEMENTATION.md](../archive/lesson-rollout/DSA-MATH-FOUNDATIONS-IMPLEMENTATION.md) |
| `FIRST-FIVE-REIMPLEMENTATION.md` | [FIRST-FIVE-REIMPLEMENTATION.md](../archive/lesson-rollout/FIRST-FIVE-REIMPLEMENTATION.md) |
| `NEXT-THREE-REIMPLEMENTATION.md` | [NEXT-THREE-REIMPLEMENTATION.md](../archive/lesson-rollout/NEXT-THREE-REIMPLEMENTATION.md) |
| `PROGRAMMING-MODULE-COMPLETION.md` | [PROGRAMMING-MODULE-COMPLETION.md](../archive/lesson-rollout/PROGRAMMING-MODULE-COMPLETION.md) |
| `PROGRAMMING-REWRITE-LINUX.md` | [PROGRAMMING-REWRITE-LINUX.md](../archive/lesson-rollout/PROGRAMMING-REWRITE-LINUX.md) |
| `SYSTEMS-STRUCTURES-IMPLEMENTATION.md` | [SYSTEMS-STRUCTURES-IMPLEMENTATION.md](../archive/lesson-rollout/SYSTEMS-STRUCTURES-IMPLEMENTATION.md) |
| `VISUAL-TEACHING-REVIEW.md` | [VISUAL-TEACHING-REVIEW.md](../archive/lesson-rollout/VISUAL-TEACHING-REVIEW.md) |

## Removed material

Seven obsolete forwarding pages were removed after updating active links. Their destinations retain the useful material:

| Removed forwarding page | Retained owner |
| --- | --- |
| `LESSON-CONTINUATION-REVIEW.md` | [LESSON-CONTINUATION-REVIEW.md](../archive/lesson-rollout/LESSON-CONTINUATION-REVIEW.md) |
| `LESSON-TEACHING-PILOT.md` | [LESSON-TEACHING-PILOT.md](../archive/lesson-rollout/LESSON-TEACHING-PILOT.md) |
| `PROGRAMMING-REWRITE-BATCH-01.md` | [PROGRAMMING-REWRITE-BATCH-01.md](../archive/lesson-rollout/PROGRAMMING-REWRITE-BATCH-01.md) |
| `PROGRAMMING-REWRITE-BATCH-02.md` | [PROGRAMMING-REWRITE-BATCH-02.md](../archive/lesson-rollout/PROGRAMMING-REWRITE-BATCH-02.md) |
| `PROGRAMMING-REWRITE-BATCH-03.md` | [PROGRAMMING-REWRITE-BATCH-03.md](../archive/lesson-rollout/PROGRAMMING-REWRITE-BATCH-03.md) |
| `PROGRAMMING-REWRITE-PANDAS.md` | [PROGRAMMING-REWRITE-PANDAS.md](../archive/lesson-rollout/PROGRAMMING-REWRITE-PANDAS.md) |
| `LESSON-TEACHING-STANDARD.md` | [SKILL.md](../../skills/lesson-authoring/SKILL.md) |

The unreferenced `scripts/update-foundations-completion-handoff.mjs` was also removed. It was a completed one-use mutation script with fixed old status text and assumptions; running it again would try to overwrite later handoffs. The DSA/mathematics completion report, ledger and integration evidence remain. Its original implementation is recoverable in Git history.

## Frozen receipts and source identity

Historical JSON receipts, phase ledgers, source-bound records and prepared manuscripts retain their original bytes. They may name a former root path from the review date; use the relocation table above to locate the current report, and Git history when the exact old revision is needed. A current destination is not a claim that its bytes match an older receipt. No historical hash was refreshed to hide this relocation.

Active instructions, ordinary documentation links and the authoring inventory generator use current destinations. The outer workspace entry point and installed skill adapter were updated too. The remaining topic-design/domain-playbook indexes under `docs/teaching/` serve existing consumers; detailed teaching policy continues to have one owner in the skill.

## Verification

- The scoped conservation check found **4,733 protected files byte-identical**, including runtime/content files, JSON ledgers and evidence, prepared packets and frozen source-bound records. All **177** topic delivery states are unchanged; no completion or source hash was promoted.
- Regenerated inventory preserves **1,461 topics, 29 modules, 234 publications, 10 guided paths**, complete route/topic order and all topic metadata except relocated report links. Effective content/implementation counts remain **81/81**; the other historical rows retain their existing state.
- **792 relative documentation links** were checked with no missing targets; redirected skill-section anchors resolve. Five pre-existing broken links in the touched reports were repaired or explicitly labeled as retired source locations with current ownership links.
- `node scripts/verify-lesson-delivery.mjs` passed all eight cases for 177 tracked topics. `node scripts/verify-repository-files.mjs` passed. The Linux topic preflight uses the current handoff and skill guidance. The skill validator passed, and the personal installation still points to the versioned package.

The temporary inventory, baselines, patch helper and verification output created
for this cleanup were removed after these results were retained here. No
production build or browser review was needed for this documentation-only change.
