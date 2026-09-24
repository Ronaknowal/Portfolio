# Writing and publishing articles

Articles are standalone writing, not lessons with a mandatory lesson template.
Preserve the author's voice. Establish the question, develop a clear argument or
explanation, use examples/evidence where they help, and distinguish observation,
interpretation and uncertainty. Research technical or current claims using primary
sources; link directly to the material supporting them. Never fabricate an author's
experience, experiments, citations, article or publication date.

## Start a draft

```sh
npm run article:new -- --slug a-stable-address --title "A meaningful title"
```

The command creates `content/articles/a-stable-address/metadata.json` and `body.md`.
It rejects invalid/reserved slugs and refuses to overwrite an existing folder.
Slugs use lowercase letters/digits and single hyphens. Keep a published slug stable
even if the title improves; a future rename requires an explicit redirect.

```json
{
  "title": "A meaningful title",
  "summary": "A concise description of the question and what the reader gains.",
  "status": "draft",
  "tags": ["Engineering"]
}
```

`title`, `summary`, `status` and `tags` are required. Tags are a unique array of
nonempty strings, with consistent capitalization across the collection. Supported
optional fields are `author` (defaults to Ronak Sharma), `publishedAt` and `updatedAt`.
Dates are real `YYYY-MM-DD` dates. A published article requires `publishedAt` and a
nonempty manuscript; `updatedAt` cannot precede its publication date. Unknown fields
fail validation rather than silently accepting a misspelling. Dates are displayed
in UTC to prevent a reader's time zone moving the calendar day.

`status: "published"` publishes at the **next build**, regardless of its date.
This is not a scheduling system. Do not mark a future piece published until it is
ready to appear. Draft titles and bodies are excluded from the production import
graph. Drafts are still in Git and may be visible in a public source repository;
keep private writing and secrets outside this repository. There is no hidden admin
or unauthenticated draft-preview URL.

## Manuscript and assets

Write standard Markdown/GitHub-flavored Markdown in `body.md`. The metadata title
already supplies the page's H1; start manuscript sections at `##`. Tables, fenced
code, lists, blockquotes, links, footnotes and images are supported. Do not depend
on arbitrary HTML, JavaScript, MDX or automatic math rendering: these are not
enabled. Add a narrowly scoped rendering feature only when a real article needs
it, with accessibility, correctness and payload checks.

Use ordinary links, including `/learn/...` for detailed topics and `/articles/<slug>`
for another published article. Links to draft/unimplemented pages need an honest
explanation, not an assertion that their content is available. Match the site's
amber link theme. Include references close to the claims they support, not just a
pile of unrelated links. Describe why a recommended alternative resource is useful.

Place final, optimized images in `public/articles/<slug>/` and reference them as
`![Meaningful description](/articles/<slug>/figure.svg)`. Use meaningful file names,
accurate labels, legible mobile text, source/provenance and captions where needed.
Check long code and wide tables scroll within their own frame without stretching
the page. All public assets are copied into deployments, even if the article is a
draft or Git ignores the file. Keep private/draft image work outside `public/` until
it is safe to serve. Use lazy images and appropriate file sizes. For interactive
experiments, link a dedicated tool or reviewed feature rather than inserting an
unbounded animation loop into a prose page.

## Preview and publish

1. Finish the content and sources; review facts, dates, code, figures and authorship.
2. Run `npm run check:content`. This validates source and the publishing pipeline.
3. For a **local** reader preview, temporarily set the article to `published` with
   a valid date and use `npm run dev`. Undo that status before committing a piece
   that should remain a draft. Development reacts to file creation/deletion/edits.
   Do not push a temporary publication state: the existing main-branch workflow deploys.
4. Check `/articles`, search and subject filters, the full reader, references,
   code/tables, unknown-slug recovery and a narrow phone width. There must be a
   useful reading experience, not merely valid Markdown.
5. When publication is authorized, set the actual publication date/status. Run
   `npm run content:generate`, `npm run check:repository`, `npm run check:content`,
   `npm run build`, then `npm run check:site-build`. Preview the production output.
6. Commit the manuscript, metadata, needed public assets and regenerated catalogue/
   loaders together. Deployment is a separate authorized action. Do not hand-edit
   generated files or add draft imports to fix a preview.

For a substantive update preserve `publishedAt`, add/update `updatedAt`, and explain
important corrections in the text. Removing a published piece or changing its slug
affects external links; decide redirects/replacement guidance explicitly.

## Handoff and quality

State what was written, what sources were checked, what code/experiments actually
ran, which UI paths were verified, and any remaining limitation. Preserve drafts
for another author without calling them published. Use source-bound evidence for
substantial claims/experiments; don't rerun unrelated lesson checks. Clean only your
temporary fixtures, screenshots and logs. The adjacent architecture guide owns
folder/dependency boundaries; the lesson teaching standard applies only when the
task is actually educational lesson authoring.
