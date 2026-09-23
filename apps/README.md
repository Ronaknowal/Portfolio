# Independently deployed applications

Use `apps/<descriptive-app-id>/` when a real project needs a separate runtime,
backend, dependency graph or release process. A small browser-only utility can
instead be a lazy feature under `src/tools/`. Do not create empty apps to reserve names.

Each independent app needs its own README, dependencies/lockfile or deliberately
configured workspace, tests, redacted environment example and deployment instructions.
Root `npm run build` builds the personal site only. Linking a project does not deploy it.

When a working app exists, expose its public URL through an appropriate Tools or
Portfolio entry. A Learn build guide may explain the same system separately.
See [the repository contract](../docs/engineering/REPOSITORY-STRUCTURE.md).
