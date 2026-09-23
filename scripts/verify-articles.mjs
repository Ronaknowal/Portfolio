import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { createArticle, generateArticles, readArticles } from "./lib/article-content.mjs";
import { writeSiteEntries } from "./lib/site-build.mjs";

const repositoryRoot = fileURLToPath(new URL("..", import.meta.url));
const fixtureRoot = await mkdtemp(path.join(tmpdir(), "ronak-article-tests-"));
const contentRoot = path.join(fixtureRoot, "content/articles");
await mkdir(contentRoot, { recursive: true });
async function save(slug, overrides = {}, body = "A source-backed explanation.") {
  const directory = path.join(contentRoot, slug);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "metadata.json"), JSON.stringify({ title: "A test article", summary: "A test summary", tags: ["Engineering"], status: "published", publishedAt: "2026-09-01", ...overrides }));
  await writeFile(path.join(directory, "body.md"), body);
}
try {
  await save("older-entry");
  await save("newer-entry", { publishedAt: "2026-09-02" });
  await save("private-draft", { status: "draft", title: "DO_NOT_SHIP_DRAFT_TITLE" }, "DO_NOT_SHIP_DRAFT_BODY");
  const published = await generateArticles(fixtureRoot);
  assert.deepEqual(published.map(item => item.slug), ["newer-entry", "older-entry"]);
  const catalogue = await readFile(path.join(fixtureRoot, "src/articles/generated/catalogue.js"), "utf8");
  const loaders = await readFile(path.join(fixtureRoot, "src/articles/generated/loaders.js"), "utf8");
  assert.ok(!catalogue.includes("private-draft") && !catalogue.includes("DO_NOT_SHIP"));
  assert.ok(!loaders.includes("private-draft") && !loaders.includes("DO_NOT_SHIP"));
  assert.equal((loaders.match(/=> import\(/g) || []).length, 2);
  assert.ok(loaders.includes("../../../content/articles/newer-entry/body.md?raw"));
  assert.equal(published[0].readingMinutes, 1);
  for (const [metadata, body, pattern] of [
    [{ publishedAt: "2026-02-30" }, "Content", /real YYYY-MM-DD/],
    [{ publishedAt: undefined }, "Content", /require publishedAt/],
    [{ updatedAt: "2025-01-01" }, "Content", /cannot precede/],
    [{ status: "publishd" }, "Content", /status must/],
    [{ tags: ["Code", "code"] }, "Content", /tags must be unique/],
    [{ publishAt: "2026-09-01" }, "Content", /Unknown metadata/],
    [{}, "  ", /nonempty body/],
  ]) {
    await save("invalid-entry", metadata, body);
    await assert.rejects(() => readArticles(fixtureRoot), pattern);
  }
  await rm(path.join(contentRoot, "invalid-entry"), { recursive: true });
  await createArticle(fixtureRoot, "new-draft", "A new draft");
  await assert.rejects(() => createArticle(fixtureRoot, "new-draft", "Overwrite"), { code: "EEXIST" });
  await assert.rejects(() => createArticle(fixtureRoot, "../escape", "Bad slug"), /Invalid article slug/);
  await assert.rejects(() => createArticle(fixtureRoot, "con", "Reserved name"), /Invalid article slug/);
  assert.equal((await readArticles(fixtureRoot)).length, 2);
  await save("escaped-metadata", { title: 'Prices: $` <model> & "data"', summary: 'Quotes " and <tags> must stay text.' });
  const outputRoot = path.join(fixtureRoot, "dist");
  await mkdir(outputRoot);
  await writeFile(path.join(outputRoot, "index.html"), '<html><head><title>Original</title><meta name="description" content="Original" /></head><body><div id="root"></div></body></html>');
  await writeSiteEntries(fixtureRoot, outputRoot);
  const entry = await readFile(path.join(outputRoot, "articles/escaped-metadata/index.html"), "utf8");
  assert.ok(entry.includes('<title>Prices: $` &lt;model&gt; &amp; &quot;data&quot; · ronak.ai</title>'));
  assert.ok(entry.includes('content="Quotes &quot; and &lt;tags&gt; must stay text."'));
  await assert.rejects(() => readFile(path.join(outputRoot, "articles/private-draft/index.html")), { code: "ENOENT" });
  assert.equal(await readFile(path.join(outputRoot, "404.html"), "utf8"), await readFile(path.join(outputRoot, "index.html"), "utf8"));
  const markup = renderToStaticMarkup(createElement(Markdown, { remarkPlugins: [remarkGfm], skipHtml: true }, "## A heading\n\n[unsafe](javascript:alert%281%29)\n\n<script>window.bad = true;</script>\n\n| A | B |\n| - | - |\n| one | two |\n\n```js\nconst answer = 42;\n```"));
  assert.ok(markup.includes("<table>") && markup.includes("<h2>A heading</h2>") && markup.includes("<pre><code"));
  assert.ok(!markup.includes("javascript:") && !markup.includes("<script>"));
  const current = await readArticles(repositoryRoot);
  console.log(`Article checks passed: ordering, dates/schema, draft exclusion, lazy imports, safe Markdown/GFM, creation/no-overwrite, escaped static entries and fallback. ${current.length} real published articles validated.`);
} finally {
  // Only the exact directory created by this test is disposable.
  await rm(fixtureRoot, { recursive: true, force: true });
}
