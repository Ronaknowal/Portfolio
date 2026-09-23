# Learning workspace: concepts, paths and research projects

Updated 22 September 2026. Current information architecture for the researcher-oriented learning section. Read with the teaching and learning-code standards; lesson depth, accuracy, delivery phases and curriculum sequence remain authoritative.

Site-level navigation now follows [SITE-ARCHITECTURE.md](SITE-ARCHITECTURE.md) (23 September): `/` is the personal hub, `/portfolio` contains the portfolio, and all `/learn` URLs remain stable. `LearningNav` supplies learning-local links to the shared `SiteHeader`; the section switcher connects Home, Portfolio and Learn. Research notes, articles and standalone tools can become separate site sections when real content is published.

## Product decisions

The user asked for the breadth of fanout.sh with an original UI and stronger research orientation. The public [directory](https://fanout.sh/), [AI route](https://fanout.sh/ai) and [labs directory](https://fanout.sh/labs) were inspected for scope/navigation. Preserve our dark/amber identity, readable editorial hierarchy and concept-specific visuals; avoid copying its presentation or creating interchangeable course cards.

| Entrance | Learner question | Route and responsibility |
| --- | --- | --- |
| Explore | Where does my question belong? | `/learn`: orientation, a real featured project, availability and grouped fields; `/learn/modules`: complete module outlines |
| Paths | What should I study, in what order? | `/learn/paths`: all guided paths and complete syllabus; preserve existing reader routes |
| Projects | How do these ideas become a working system? | `/learn/projects` and `/learn/projects/:projectId/:stageId?`: build guides with linked stages and deliverables |
| Search | Where is this exact concept? | `/learn/catalogue`: topic/subtopic search with module, publication and difficulty filters |

Projects connect knowledge across modules; they do not become another giant module. Link exact prerequisite owners and a related depth topic where useful. A project and its companion topic have independent publication states. Preserve stable IDs, progress identities and teaching order. Derive counts from the existing catalogue and route resolvers.

Theme: use the shared `--learn-*` tokens in `learning-workspace.css`: near-black, neutral charcoal, readable neutral text and amber accents. The user rejected green/olive workspace and project colours on 22 September. Do not reintroduce them for decoration or progress. New project styles must inherit these tokens instead of inventing their own palette. Data-series colours in an existing scientific figure have a different semantic purpose; do not indiscriminately recolour its evidence.

## Research project authoring

Follow [PROJECT-AUTHORING-STANDARD.md](../../PROJECT-AUTHORING-STANDARD.md), the dedicated end-to-end project teaching and delivery manual. The earlier authoring rules from this section are preserved and expanded there, so future changes have one project-authoring owner rather than two competing checklists. This file continues to own information architecture, runtime boundaries and shared workspace integration.

The user selected the finished Typed Decision Model depth revision 2 as the teaching reference. Preserve locally explained mechanisms, actual scratch/tool implementations, contextual concept links, distinct live investigations without learner-prediction gates, and evidence-based interpretation. Adapt stages and visual forms to the project. Read the manual for planning, complete content-first packets, review, phase tracking and handoff; do not treat the reference's stage/lab counts as quotas.

## Source ownership and loading

- `components/LearningNav.jsx`: local learning navigation inside the shared site header. Add learning destinations here only when useful content exists; site sections belong in `src/app/navigation.js`.
- `LearnHub.jsx`, `learning-workspace.css`: discovery views; no lesson or project bodies in metadata imports.
- `ProjectReader.jsx`, `project-reader.css`: stages, prerequisites, accessible loading/recovery, milestone checklist, Previous/Next and unknown-address recovery.
- `data/projects/<stable-project-id>/metadata.js`: compact ID, title, description, kind, level, honest availability, prerequisite/related topic IDs, stage IDs/titles/summaries/deliverables and optional declarative preview. No React, weights, whole programs or manuscripts.
- `data/projects/<stable-project-id>/content.jsx`: default stage-ID-to-component mapping, with local model helpers and scoped CSS. Loaded only on the selected project route.
- `data/projects/catalogue.js` and `loader.js`: metadata and explicit dynamic-import registration. Add both together. Cache success, evict rejection and ignore stale asynchronous loads; keep errors distinct from missing content.
- `public/learn-projects/<stable-project-id>/`: canonical programs, sample inputs and selected measured reports. Fetch large source only when opened. Never place model weights in the hub or ordinary lesson bundle.
- `docs/teaching/projects/<stable-project-id>.md`: author scope, source review, computational evidence and limitations; independent review remains separate.
- `docs/teaching/projects/project-delivery-progress.json`: project authoring phase record, separate from topic delivery. The central lesson ledger must not imply a full topic was implemented because a project guide was added.

Learner progress uses `learning-project-progress-v1`, keyed by `projectId/stageId`. Never write the lesson key `kd-progress`. Stable stage IDs survive wording changes; unknown IDs must not inflate counts. A local milestone means the learner checked a deliverable, not that an experiment succeeded. Handle unavailable storage without crashing and explain the persistence limit. Browser progress is not cloud sync or certification.

## Future expansion

The project contract supports model builds, paper reproductions, systems builds and experimental studies. Add filters and project search as the collection grows. Index only compact metadata. Preview diagrams must belong to the specific project; other projects receive an appropriate custom preview or a stage-sequence fallback.

Later add papers, standalone labs, reference tools, reading collections and research notes as real resource types with stable IDs, compact metadata, explicit availability, concept/project links and on-demand detail routes. Keep a common navigation/visual language and let content structure vary. Do not add dead tabs, fabricated counts or a pile of disabled 'coming soon' cards. Shared labs retain a canonical model owner.

## Verification and continuation

Verify catalogue conservation, prerequisite links, stage registration, local progress isolation/persistence, invalid routes, all stage bodies, source disclosures, mobile/keyboard behavior, loading failures and production import boundaries. Inspect actual screenshots as well as overflow bounds. Reuse unchanged numerical lesson evidence; a navigation change does not call for retraining every earlier example.

Run `node scripts/verify-learning-projects.mjs` for metadata/registration checks. Run `scripts/verify-learning-workspace.cjs` against a fresh production build with `LEARNING_BASE_URL` and the installed `PLAYWRIGHT_PACKAGE`; it uses fresh contexts and retains a bounded report and three screenshot candidates. Inspect the images before recording visual acceptance. See [the completed integration](../teaching/projects/WORKSPACE-INTEGRATION.md) for actual evidence and performance limits.

The project ledger records content, implementation, independent review, browser integration and user acceptance separately. Only complete phases supported by their linked evidence; preserve earlier records when revising a project. A completed project must not advance its companion topic's lesson phases.

The first project is `typed-decision-model`; its in-depth companion `typed-decision-models-calibrated-neural-decision-systems` remains planned. Read its author/reviewer records and phase checkpoint for future work. This redesign does not authorize an additional lesson or project queue.
