# Learning workspace integration — 22 September 2026

The research-oriented workspace and first project are implemented locally. This record closes the integration handoff in the author and independent-review records. It does not mark the planned companion lesson complete and does not certify the example model for deployment.

## Delivered structure

- Explore (`/learn`): orientation, a real featured build and grouped fields.
- Paths (`/learn/paths`): all guided paths and the complete curriculum.
- Modules (`/learn/modules`): existing module outlines and field filters.
- Search (`/learn/catalogue`): searchable topics/subtopics and availability.
- Projects (`/learn/projects`): a separate collection of end-to-end guides.
- `/learn/projects/typed-decision-model/:stageId?`: eight linked stages, prerequisites, immediate live controls, canonical source disclosures and separate local milestones.

All 1,460 prior topic IDs and their relative section/topic order were preserved; one planned Typed Decision Models companion was added to Large Language Models, yielding 1,461 topics. It is included in the complete-curriculum and LLM-engineer routes. Publication remains 231 lessons; the lesson ledger remains 177 complete content checkpoints, 140 complete implementations and 37 prepared implementations. Project delivery is tracked in [its separate ledger](project-delivery-progress.json).

## Final integration evidence

The production Vite build passed after the final theme repair. The package-manager shim is broken on this host, so the equivalent installed CLI was used: `node node_modules/vite/bin/vite.js build --logLevel warn`.

Passed curriculum verification/inventory generation, learning-artifact freshness and import-boundary checks. The final artifact check confirmed all 1,461 records, 29 modules, 231 publication mappings and guided/module routes match authoring sources. The project verifier confirmed the registry, eight ordered exports, prerequisite/companion links, metadata isolation and invalid-ID rejection.

The [production browser report](evidence/workspace-browser-review.json) records 39 passing cases:

- Five discovery views at 1,366, 390 and 320 pixels; all eight project stages at desktop and 320 pixels.
- Search to the planned companion and its reciprocal project link.
- Temperature keyboard changes, candidate addition, cost-dependent action and reset.
- Milestone persistence across reload, isolation from lesson completion and resume at the first unfinished stage.
- On-demand canonical Python source rendering.
- Unknown project/stage and prototype-name recovery.
- Failed project import, disabled milestone completion and explicit reload recovery.
- Shared navigation on the existing Linear & Logistic Regression lesson at 320 pixels.
- No page errors in the normal flow and no unrelated lesson/project bodies fetched by the fresh hub.

The independent UI review also checked unavailable-storage memory fallback, listener cleanup, cross-tab/clear synchronization and progress isolation. This is local browser progress, not a backend account or certification.

The retained [hub](evidence/workspace-desktop.png), [desktop lab](evidence/decision-lab-desktop.png) and [phone lab](evidence/decision-lab-mobile.png) screenshots were actually inspected after the tests. The Codex browser additionally confirmed the final hub with loaded web fonts. Text, controls and diagrams are readable; narrow layouts stack and deliberately scroll code/tables instead of clipping controls.

## Theme correction and source scope

The user rejected the green/olive palette during integration. New workspace, navigation and project surfaces now use a shared `--learn-*` palette: neutral near-black/charcoal, neutral text and amber controls/accents. Removed the conflicting green-progress direction from `.impeccable.md`; the engineering contract records the explicit preference. The reciprocal project banner uses the shared stylesheet instead of hard-coded inline colours.

Observed final hub backgrounds in the actual browser were RGB (12,12,12) for the page/nav, (20,20,20) for the feature and (31,31,31) for the preview. They contain no green tint. This was a scoped palette correction, not an indiscriminate change to semantic colours in existing scientific data.

The browser report binds the core UI and model source hashes. Additional final source identities:

| Source | SHA-256 |
| --- | --- |
| `src/learn/data/projects/typed-decision-model/project.css` | `6f192ed4e30498f4322344c2bce4597d34cac26e6a51a5eedd8a236f3e123958` |
| `src/learn/components/TopicContent.jsx` | `2240ec2ae10c7ecef6860086af2df707606444e7aa3d25f4d288184737cc9a4d` |
| `src/learn/data/projects/typed-decision-model/metadata.js` | `d08244b77e2200d987f3885a0f7459bd9fb59f691c9283e8df4bb74f810b89c1` |

The project stylesheet hash supersedes the independent review's earlier CSS hash. Its changes are palette substitutions; the project manuscript, browser arithmetic, Python program and measured report remain identical to the independently reviewed sources. Unchanged numerical experiments were not retrained for a colour change.

## Performance and limits

Project teaching is dynamically imported when its route is opened. The Python program is fetched only when a source disclosure opens; model weights are never shipped in the hub. The fresh-hub network check found no unrelated lesson/lab chunks.

Vite still reports a large shared catalogue chunk: approximately 889 kB decoded / 129 kB encoded in this local browser report. This is shared curriculum metadata, not the project manuscript or model weights. The threshold was not raised or suppressed. No before/after performance improvement or exhaustive site audit is claimed; future catalogue optimization can be scoped separately.

The executed model remains a 30,977-parameter CPU teaching system. The lexical baseline outperforms it on the constructed held-out examples, and temperature fitting worsens test NLL. These findings remain visible. Pretrained training, RL, additional question types and a network service are labelled unexecuted extensions.

No deployment or commit was performed. User review remains pending. Future additions follow [the workspace/project contract](../../engineering/LEARNING-WORKSPACE.md); no further lesson queue is authorized by this completion.
