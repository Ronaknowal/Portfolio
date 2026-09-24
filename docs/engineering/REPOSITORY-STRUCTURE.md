# Repository structure and extension contract

Updated 25 September 2026. The repository is a personal publishing and learning
site, with room for independently deployed applications. The npm package is
`ronak-site`; the checkout directory is unchanged. The intended custom domain is
`ronak.sh`; [DEPLOYMENT.md](DEPLOYMENT.md) owns its domain and HTTPS setup.

## Boundaries

Keep one application while sections share a deployment and runtime. Separate by
purpose, not by the order in which a task happened. Do not create batches named
`next-three`, a giant `pages` file, eager all-content barrels, or a monorepo package
for every navigation item. Introduce workspaces only when a real independent app
or shared package justifies a separate dependency/build boundary.

| Owner | Responsibility | Must not own |
| --- | --- | --- |
| `src/app` | Route composition, failure boundaries, scroll/legacy routing, compact global navigation | Article manuscripts, curriculum bodies, feature implementations |
| `src/shared/layout` | Theme tokens, global header, site switcher, recovery presentation | Lessons, article bodies, portfolio content |
| `src/home` | Introduction and a few curated, real links | Imports of section bodies or generated curriculum |
| `src/portfolio` | Existing work, experience and project showcases | Entire site's routing |
| `src/articles` | Index, filters, reader, generated publication metadata/imports | Handwritten manuscripts or educational progress |
| `content/articles` | Stable-ID manuscript folders | Executable application code, credentials |
| `src/learn` | Established educational system | Articles/notes appended as fake curriculum topics |
| `apps/<app-id>` | A genuinely independent application's own source, dependencies, tests and deployment instructions | An automatically imported bundle in the personal site |
| `public` | Explicitly public, deployed assets | Drafts, source backups, logs from test runs, secrets |

`src/main.jsx` remains the small mount; `src/index.css` remains the global reset.
Root teaching documents and established lesson internals keep their stable paths.
Do not rearrange thousands of lesson files for cosmetic symmetry. Domain CSS is
scoped; the common header retains its 57px contract and neutral/amber palette.

Dependency direction: the app composes section entry points; sections use shared
layout; shared layout uses the lightweight app navigation registry only. That
registry must never import section code. Cross-section discovery uses stable
URLs or compact metadata, not eager content imports. No section imports the
application router. Prefer direct descriptive imports over generic barrel files.

## Add a section

1. Give it a clear job and stable route: notes are evidence/working observations;
   articles are finished writing; tools are usable applications.
2. Create `src/<section>/` with its index, detail view where applicable, scoped CSS,
   loading/unknown/error states and its own data contract. Use the shared header
   with meaningful local navigation.
3. Register a lazy section entry in `src/app/SiteRoutes.jsx`. Add the compact entry
   in `src/app/navigation.js` once the section is ready to expose. The user
   explicitly requested the empty Articles destination; this does not authorize
   a row of placeholder future sections.
4. Add an appropriate entrance to Home. Keep it a compact directory. Test all
   header links at 320px after adding destinations; use the existing section
   switcher rather than overflowing local navigation.
5. Keep large documents/data/visualizations behind detail-level imports. Generate
   indexes from source metadata; do not manually synchronize counts or duplicate
   body text into search bundles. Only load heavy libraries on their feature route.
6. Update this contract and the route table; verify direct loads, navigation,
   unknown addresses, mobile layout, focus and actual production imports.

Reuse the article pipeline's pattern for future notes, but keep different
schemas if revision state, experiment artifacts or publication semantics differ.
Extract a common collection engine only after there are real shared requirements.
Never rename established slugs without retaining a redirect.

## Articles

`scripts/lib/article-content.mjs` owns validation and metadata generation.
`content/articles/<slug>/metadata.json` and `body.md` are source of truth.
`src/articles/generated/catalogue.js` contains only published metadata; `loaders.js`
contains explicit dynamic imports for those published bodies. Keep generated files
in Git, regenerate rather than hand-edit, and never add an eager/raw glob over all
drafts. Build/dev regenerate them before source compilation. Invalid metadata
fails the build with the article ID. Development watches creation, deletion and edits.

The index imports metadata only. The reader route loads
[react-markdown](https://github.com/remarkjs/react-markdown) and
[remark-gfm](https://github.com/remarkjs/remark-gfm), then the requested body. Raw
HTML is skipped, unsafe protocols are filtered by the renderer, and tables/code
have bounded horizontal scrolling. React escapes textual metadata. Do not enable
raw HTML or arbitrary MDX evaluation as a convenience for an isolated article.
Reader state resets between articles and provides retry/error recovery.

No articles were invented during this structural change. The index has a deliberate
empty state until the owner supplies a real piece. This collection is ready to
publish without changing JSX for each article.

## Independently deployed projects

A small tool that shares this site can live under `src/tools/<tool-id>` and load
lazily. An app with its own server, credentials, storage, release cadence or large
dependency tree belongs in `apps/<app-id>` (or a separate repository). Its own
README must define development, tests, environment **names only**, deployment,
public URL, data handling and ownership. Keep its build separate from `npm run build`
at the root. Register a tool listing/launch link only when there is a working
destination. Do not fake a deployment by linking an unserved source directory.

Shared tooling or npm workspaces can be added when a real app needs them; this
change does not create empty packages, backend stubs, hosting accounts or deployments.

## Hosting

The deployment target is Cloudflare Workers Static Assets with root-relative URLs.
`.github/workflows/deploy-cloudflare.yml` builds and validates `dist/`, then deploys
it using `wrangler.jsonc`. The initial configuration enables `workers.dev`; attaching
`ronak.sh` is a separate migration step after the build and DNS are verified.
`assets.not_found_handling` is explicitly `single-page-application`, so unmatched
page URLs receive the root HTML with HTTP 200 before React resolves the route.

`writeSiteEntries` still emits entry files for published site sections, Learn
discovery pages and every published article, with escaped page metadata. Existing
files take precedence over the SPA fallback. `.nojekyll`, `404.html` and
`public/CNAME` are retained for the existing GitHub Pages deployment during migration;
they do not configure a Cloudflare custom domain. Pages settings control its domain,
and Pages fallback-only URLs still return HTTP 404 before client-side recovery.

For indexing/social previews requiring fully rendered article bodies, introduce
tested static prerendering or an SSR-capable host; do not claim that this client
renderer already provides it. If hosting changes, preserve stable URLs and replace
the fallback with the host's rewrite/real entry strategy. Run output checks after
changing build output, base URL or route naming.

## Ignore rules, checks and cleanup

The `.gitignore` excludes dependencies, local environments, root build/scratch
outputs and explicitly named independent-app build outputs. Root-only patterns
are anchored: `/scratch/` must not swallow `docs/archive/scratch/` evidence. Logs
used as published examples remain trackable; disposable logs belong in scratch.
Never blanket-ignore Markdown, JSON, generated source, runtime assets, `content/`,
or public Python examples. `.env` files remain ignored; redacted `.env.example`
and `.env.<environment>.example` templates can be committed.

`npm run check:repository` tests current source/docs/assets and representative future
paths against repository ignore rules while also checking that caches/secrets stay
ignored. It deliberately excludes machine-global Git rules; inspect those separately
if a developer still cannot add a required file. `npm run check:content` validates
all articles and exercises schema, draft exclusion, safe Markdown and creation.
`npm run check:site-build` checks the built route/body loading boundaries and static
entry files. CI runs these around the normal build.

Keep only purposeful evidence and reproducible scripts. Use temporary OS directories
for self-contained test fixtures and clean the exact directory created, in `finally`.
Temporary browser fixtures must never remain in `content/` or `public/`. Do not
delete unrelated uncommitted work. Vite copies **everything** in `public/`, even Git-
ignored files: never run Python/build tools there in a way that leaves caches.
See the learning code standard for its existing evidence retention contract.
