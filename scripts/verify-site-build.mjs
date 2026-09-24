import assert from "node:assert/strict";
import { readFile, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readArticles } from "./lib/article-content.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const manifest = JSON.parse(await readFile(path.join(root, "dist/.vite/manifest.json"), "utf8"));
function staticClosure(entry, found = new Set()) {
  if (found.has(entry)) return found;
  assert.ok(manifest[entry], `Missing manifest entry: ${entry}`);
  found.add(entry);
  for (const dependency of manifest[entry].imports || []) staticClosure(dependency, found);
  return found;
}
const home = staticClosure("index.html");
const index = staticClosure("src/articles/ArticlesIndex.jsx");
const reader = "src/articles/ArticleReader.jsx";
for (const entry of [reader, "src/articles/ArticlesIndex.jsx", "src/portfolio/Portfolio.jsx", "src/learn/LearnHub.jsx"]) {
  assert.ok(manifest[entry]?.isDynamicEntry, `${entry} must be a lazy route`);
  assert.ok(!home.has(entry), `Home eagerly imports ${entry}`);
}
assert.ok(!index.has(reader), "Articles index eagerly imports Markdown reader");
for (const entry of [...home, ...index]) assert.ok(!entry.startsWith("content/"), `${entry} is eagerly loaded`);
const published = await readArticles(root);
const allowedBodies = new Set(published.map(article => `content/articles/${article.slug}/body.md?raw`));
for (const key of Object.keys(manifest).filter(key => key.startsWith("content/articles/"))) {
  assert.ok(allowedBodies.has(key), `An unpublished article entered the build: ${key}`);
  assert.ok(manifest[key].isDynamicEntry, `Article body is not loaded on demand: ${key}`);
}
for (const key of allowedBodies) assert.ok(manifest[key], `Missing published body: ${key}`);
for (const file of ["articles/index.html", "portfolio/index.html", "learn/index.html", "404.html", ".nojekyll", ...published.map(article => `articles/${article.slug}/index.html`)]) await access(path.join(root, "dist", file));
for (const article of published) {
  const html = await readFile(path.join(root, "dist/articles", article.slug, "index.html"), "utf8");
  assert.ok(!html.includes("<title>Ronak Sharma — Build"), "Article entry needs its own metadata");
}
console.log(`Build checks passed: lazy section boundaries, isolated Markdown reader, ${published.length} on-demand published bodies, no draft chunks, static route entry files and Pages fallback.`);
