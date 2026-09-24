# ronak.sh

A personal site with a home, portfolio, learning workspace and articles. This is
one React/Vite application with independently loaded sections, not a portfolio
component that owns the rest of the site.

## Development

Use Node.js 20.19+ (or a supported newer LTS) and npm.

```sh
npm ci
npm run dev
npm run check:repository
npm run check:content
npm run build
npm run check:site-build
npm run preview -- --host 127.0.0.1 --port 4194
```

Build and development startup regenerate the learning metadata and article
catalogue. Article additions, edits, publication changes and removals refresh
the development server. On Windows, if the npm PowerShell shim is broken, run
the installed `npm-cli.js` with Node; do not create an `npm` file in the repository.

## Where things belong

```text
src/app/             route composition and published navigation
src/shared/layout/   shared header, section switcher and shell tokens
src/home/            personal introduction and curated entrances
src/portfolio/       portfolio presentation and local navigation
src/articles/        article discovery, reader and generated metadata
src/learn/           existing curriculum, lesson, lab and guided-build system
content/articles/    Markdown manuscripts and publication metadata
public/              assets intentionally copied into the deployed website
apps/                independently built/deployed applications, when added
scripts/             generators, authoring tools and verification
docs/engineering/    architecture, implementation contracts and evidence
docs/writing/        article publishing guidance
docs/teaching/       lesson/project records, drafts, ledgers and evidence
scratch/             disposable local experiments; never an authoring source
```

The existing teaching manuals remain at their established root paths so their
links and authoring workflow keep working. Start learning work at
[LESSON-AUTHORING-HANDOFF.md](LESSON-AUTHORING-HANDOFF.md); start repository work at
[the structure guide](docs/engineering/REPOSITORY-STRUCTURE.md).

## Write an article

```sh
npm run article:new -- --slug a-stable-address --title "An article title"
```

Edit `content/articles/a-stable-address/metadata.json` and `body.md`. New entries
are drafts. The [authoring guide](docs/writing/ARTICLE-AUTHORING.md) explains
publication, Markdown, images, dates, references and verification. Draft source
is versioned but excluded from production imports; it is **not secret storage**.

## Deployment

The Cloudflare workflow (`.github/workflows/deploy-cloudflare.yml`) installs from
the lockfile, verifies content and ignore rules, builds, checks output boundaries,
and deploys `dist/` to Workers Static Assets. Add the `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID` repository Actions secrets before publishing to `main`.
`wrangler.jsonc` initially enables a `workers.dev` address for verification; the
intended custom domain is `ronak.sh`. See [domain and HTTPS setup](docs/engineering/DEPLOYMENT.md)
for the remaining account, DNS and redirect steps.

The existing GitHub Pages workflow is retained during migration. Top-level sections
and published articles receive real HTML entry files, while Cloudflare's explicit
SPA fallback serves other nested URLs. `public/CNAME` and `404.html` remain for
Pages compatibility. Rendering still uses React; this is not full server rendering.
See [hosting details](docs/engineering/REPOSITORY-STRUCTURE.md#hosting).

Do not deploy backend services by putting their source in `public/`. A usable
tool, a portfolio showcase and a Learn build guide can link to the same project,
but they have different jobs and do not share a completion ledger.
