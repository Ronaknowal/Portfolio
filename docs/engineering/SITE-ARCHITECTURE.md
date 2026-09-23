# Personal hub and site navigation

Updated 23 September 2026. This is the site-level information architecture. The user chose a **personal hub with Portfolio and Learn**, then requested Articles as another section. Preserve the black, neutral-charcoal and amber identity; do not introduce green/olive decoration. The current learning and project authoring standards continue to own their content.

## Published structure

| Destination | Route | What belongs here |
| --- | --- | --- |
| Home | `/` | A brief introduction, clear entrances, and a small editorial selection of actual work and learning material. |
| Portfolio | `/portfolio` | Personal work, career experience, project showcases, and contact. Existing portfolio content is preserved. |
| Learn | `/learn` | Concepts, interactive teaching, guided paths and end-to-end educational builds. Existing nested reader and project URLs are unchanged. |
| Articles | `/articles`, `/articles/:slug` | Standalone writing with dates, subjects, references and on-demand Markdown bodies. The explicitly requested index is available before the first article; no posts were fabricated. |

Home is a useful starting point, not a full catalogue. Its selected entries are links to existing content, not a second publication ledger or manually maintained total. Avoid invented statistics, unpublished project demos, stale “latest” claims, dead links and disabled coming-soon cards.

## Home presentation: personal introduction and directory

The user liked the initial hub idea but asked for inspiration from [Aleksa Gordić's homepage](https://www.aleksagordic.com/). The rendered reference was inspected on 23 September: a personal introduction, concrete work context, a compact page directory and social links inside a terminal-styled presentation. Borrow the directness, personality and compact organization. Do not reproduce its name art, biography, green palette, terminal window or command interface.

The refined home leads with **Ronak Sharma**, the existing professional role, and a short first-person introduction grounded in the portfolio. A compact directory sits beside it on desktop and follows it on mobile. Portfolio, Learn and Articles are the main destinations, with small nested shortcuts for guided paths, builds and concept search. The selected-entry list below connects to actual work. Directory-style monospace labels and amber path separators supply a restrained technical character; paragraphs retain readable proportional type. All navigation remains ordinary accessible links.

Keep this as a personal space rather than a promotional landing page: avoid oversized generic slogans, unsupported current-employer claims, fictional live status, fake shell controls and unnecessary metric panels. Future published notes/articles/tools can extend the directory without turning it into a wall of cards. The original global routing, stable learning URLs, shared header and lazy-loading boundaries are preserved.

The shared header has a home link at the brand. Inside a section, the section name opens a **Browse site** disclosure with Home, Portfolio, Learn and Articles. The right side contains local navigation: Explore/Paths/Projects/Search in Learn and section anchors in Portfolio. This keeps the learning workflow intact instead of cramming every destination into the reader header. The disclosure supports keyboard access, Escape with focus return, outside click, focus leaving it, and closing on navigation. Do not implement it as an ARIA menu unless full menu keyboard behavior is also implemented.

Portfolio now lives at `/portfolio`. Preserve its section IDs. Existing root links such as `/#projects` and `/#contact` redirect with replacement to the matching portfolio address; unrelated home anchors must continue to work normally. Direct anchors wait for the lazy section to mount. New page routes start at the top. Do not change lesson progress keys, curriculum order, project stage IDs or topic availability to implement site navigation.

## Growth model

These are reserved information-architecture directions, **not currently published routes**. Introduce a destination when it has useful real content and a working index/detail view:

| Future section | Suggested routes | Content contract |
| --- | --- | --- |
| Research notes | `/notes`, `/notes/:slug` | Working questions, paper annotations, experimental observations and evidence. Show dates, sources, revision state and uncertainty; link relevant concepts and builds. |
| Tools | `/tools`, `/tools/:slug` | Usable utilities, interactive applications and deployed experiments. Explain purpose, inputs, limits, privacy and availability; provide a real launch link. |

Maintain useful distinctions: a **portfolio project** demonstrates work; a **Learn project** teaches a complete build; a **tool** is something someone can actually use. One underlying project can have all three views with cross-links. Do not duplicate a teaching manuscript into the portfolio. Research notes are provisional records; articles are finished pieces. Avoid creating a vague “Other” destination.

Add a published section to `src/app/navigation.js` and its lazy route to `src/app/SiteRoutes.jsx`. The section switcher is generated from the registry. Adapt the home page's entrance hierarchy when a substantial section ships; do not turn every new entry into an identical card. Keep the global section list short. At larger scale, add section-local filters/search first; a future shared search can index compact metadata across content types without importing their bodies.

Each new collection needs stable IDs, compact metadata, honest availability, a meaningful index, on-demand detail loading, unknown-address recovery and contextual links. Do not add notes/articles/tools to the lesson curriculum or lesson-completion storage. Each content type owns its evidence and progress, if it needs progress at all.

## Source and runtime ownership

- `src/main.jsx`: React/router mount and minimal global reset.
- `src/app/SiteRoutes.jsx`: route boundaries, independent lazy section imports, root-anchor compatibility, scroll handling, unknown-address and load-failure recovery.
- `src/app/navigation.js`: compact live navigation labels, destinations and descriptions. Legacy portfolio anchors belong to `src/portfolio/navigation.js`. No curriculum or body imports.
- `src/shared/layout/SiteHeader.jsx`, `site-shell.css`: shared header, section switcher, focus affordances, neutral/amber shell tokens and recovery UI.
- `src/home/Home.jsx`, `home.css`: personal hub and curated entrances. Editorial text must reflect actual linked content. No dynamic lesson imports, portfolio animation imports, synthetic charts, publication totals or background render loops.
- `src/portfolio/Portfolio.jsx`: lazy portfolio body and its own local navigation. Preserve its existing work separately from the home.
- `src/learn/components/LearningNav.jsx`: supplies local learning navigation to the shared header. `learning-base.css` holds the existing learning/reader styles formerly bundled globally; `learning-workspace.css` owns workspace presentation.
- `index.html`: one shared font stylesheet with `display=swap`. Do not inject duplicates from individual pages.

Keep styles scoped. Preserve the reader's 57px header contract unless all affected offsets are deliberately updated together. At narrow widths keep Learn's compact navigation; Portfolio's longer section list can scroll within its own labeled navigation row. Site controls must stay visible and usable at 320px. Use text and real links for orientation rather than relying on illustration or hover alone. Maintain reduced-motion behavior, visible focus, skip links and readable contrast.

## Verification

Build with `node node_modules/vite/bin/vite.js build --logLevel warn`. Test the fresh production build using `scripts/verify-site-shell.cjs` with `SITE_BASE_URL` and `PLAYWRIGHT_PACKAGE` set for the local environment. It checks the homepage payload boundary, entrance and highlight navigation, legacy anchors, section-switcher interaction, unknown-address recovery, learning/project deep links, progress preservation and responsive header bounds. Inspect its retained desktop/mobile screenshots before calling the presentation accepted. User acceptance is separate from these checks.

This is a navigation task: reuse existing computational lesson evidence. Do not rerun numerical training or rewrite manuscripts. Keep only the final bounded browser report and reviewed screenshots in `docs/engineering/evidence/site-shell/`; discard task-specific scratch captures after review. Preserve unrelated evidence and uncommitted work.

## Repository and Articles ownership

[REPOSITORY-STRUCTURE.md](REPOSITORY-STRUCTURE.md) owns the broader folder/dependency contract and future independent apps. [ARTICLE-AUTHORING.md](../writing/ARTICLE-AUTHORING.md) owns Markdown publishing. Article UI lives in `src/articles/`, manuscripts in `content/articles/`, and metadata/import registries in `src/articles/generated/`. Only explicitly published bodies are emitted into production. Root AGENTS.md is now a short task entry; its complete former teaching instructions are preserved in `docs/teaching/WORKSPACE-INSTRUCTIONS.md`. This structural change updates no lesson completion state.
