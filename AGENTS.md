# Working in this repository

This is the ronak.ai personal site, with Home, Portfolio, Learn and Articles.
Read [README.md](README.md) for commands and [the repository contract](docs/engineering/REPOSITORY-STRUCTURE.md) for ownership, imports, publishing and deployment.

## Choose the task's instructions

- Site UI/routing: [SITE-ARCHITECTURE.md](docs/engineering/SITE-ARCHITECTURE.md) and [.impeccable.md](.impeccable.md).
- Article writing/publishing: [ARTICLE-AUTHORING.md](docs/writing/ARTICLE-AUTHORING.md).
- Educational content: start at [LESSON-AUTHORING-HANDOFF.md](LESSON-AUTHORING-HANDOFF.md), then the applicable teaching standard and scoped ledger record. [Preserved workspace checkpoints](docs/teaching/WORKSPACE-INSTRUCTIONS.md) are historical context to consult only for a specific question, not a required second handoff. The teaching ledger owns current phases; historical snapshots are not new queues.
- Educational runtime: also [LEARNING-CODE-STANDARD.md](docs/engineering/LEARNING-CODE-STANDARD.md) and [LEARNING-WORKSPACE.md](docs/engineering/LEARNING-WORKSPACE.md).
- Guided Learn projects: [PROJECT-AUTHORING-STANDARD.md](PROJECT-AUTHORING-STANDARD.md).
- Independent deployed apps: [apps/README.md](apps/README.md).

Read the documents relevant to the authorized task. Site architecture work does not authorize lesson rewrites or a curriculum rollout. Do not reopen old lesson checks or recursively inspect scratch merely to orient a task. The current handoff summarizes live work; historical records retain completed batches without becoming future instructions.

## Common engineering rules

Preserve existing uncommitted/untracked work. Use semantic domain and purpose names, never temporal batch names. Keep stable URLs, lesson IDs, publication mappings, phase ledgers and learner progress. The app composes lazy sections; shared layout and Home must not import their bodies. Keep educational source paths stable rather than moving thousands of files cosmetically.

Preserve the neutral black/charcoal and amber theme; the user rejects green/olive decoration. Use the shared section switcher and separate local navigation. Test new navigation at narrow widths; preserve accessible links, focus, keyboard behavior and reduced motion. Do not fabricate publications, metrics or deployed apps.

Required source, manuscripts, generated source, docs and public runtime assets must remain trackable. Root build/scratch rules are anchored in .gitignore; do not add blanket Markdown/JSON/content exclusions. Run npm run check:repository after ignore changes. Public files are deployed even when ignored by Git: never leave caches, private drafts or test fixtures there.

After applicable changes run scoped checks, the production build and relevant browser verification. Article/build checks are documented in README. Reuse unchanged educational numerical evidence for site-only changes. Record actual results and limits, not unverified completeness claims. Remove only your disposable artifacts after retaining useful final evidence. Do not commit, push or deploy unless requested.
