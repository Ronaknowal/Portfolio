# Documentation directory

Start with the guide for the work you are doing. Reusable lesson teaching rules
live in the [lesson-authoring skill](../skills/lesson-authoring/SKILL.md); the
repository keeps the actual curriculum, manuscripts, phase state and evidence.

| Work | Start here | What belongs here |
| --- | --- | --- |
| Website structure and development | [Repository contract](engineering/REPOSITORY-STRUCTURE.md), [site architecture](engineering/SITE-ARCHITECTURE.md) | `engineering/`: ownership, runtime contracts, deployment and engineering evidence |
| Research, write, implement or review a lesson | [Current handoff](teaching/LESSON-AUTHORING-HANDOFF.md), [skill and modes](../skills/lesson-authoring/SKILL.md) | `teaching/`: selected topic records, drafts, reviews and two-phase delivery ledger |
| Curriculum scope, sequence and new subjects | [Curriculum plan](curriculum/LEARNING-CURRICULUM-PLAN.md), [current inventory](curriculum/CURRICULUM-INVENTORY.md) | `curriculum/`: coverage plans, syllabi and generated topic metadata |
| End-to-end guided projects | [Project authoring standard](teaching/projects/PROJECT-AUTHORING-STANDARD.md) | `teaching/projects/`: project-specific explanation, stage plans, evidence and separate delivery ledger |
| Articles | [Article authoring](writing/ARTICLE-AUTHORING.md) | `writing/`: publishing guidance; manuscripts live in `content/articles/` |
| Earlier implementation decisions | [Historical rollout index](archive/lesson-rollout/README.md) | `archive/`: useful completed history, consulted for a specific question |

Keep `README.md`, `AGENTS.md` and the hidden `.impeccable.md` design context at the
repository root. Add detailed documents to the relevant directory above, with
semantic names. Update an existing record when it owns the work; do not create
another instructions file or report for every small change.

Prepared content and review evidence are not disposable merely because teaching
instructions moved into a skill. Follow the [retention rules](engineering/WORKING-ARTIFACT-RETENTION.md)
for cleanup. The [documentation relocation record](engineering/DOCUMENTATION-ORGANIZATION.md)
maps old root names retained inside historical receipts to current locations.
