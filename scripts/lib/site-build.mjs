import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { readArticles } from "./article-content.mjs";
import { siteSections } from "../../src/app/navigation.js";

const escapeHtml = value => value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);

// GitHub Pages has no SPA rewrite rules. Give published destinations real entry
// files, plus a fallback for existing nested lesson links and unknown addresses.
export async function writeSiteEntries(repositoryRoot, outputRoot) {
  const html = await readFile(path.join(outputRoot, "index.html"), "utf8");
  const articles = await readArticles(repositoryRoot);
  const pages = [
    ...siteSections.filter(section => section.href !== "/").map(section => ({ route: section.href, title: `${section.label} · ronak.ai`, summary: section.description })),
    ...["paths", "modules", "catalogue", "projects"].map(view => ({ route: `/learn/${view}`, title: "Learn · ronak.ai", summary: "Deep concepts and guided builds." })),
    ...articles.map(article => ({ route: `/articles/${article.slug}`, title: `${article.title} · ronak.ai`, summary: article.summary })),
  ];
  for (const page of pages) {
    const directory = path.join(outputRoot, page.route.slice(1));
    await mkdir(directory, { recursive: true });
    const content = html.replace(/<title>.*?<\/title>/s, () => `<title>${escapeHtml(page.title)}</title>`)
      .replace(/<meta name="description" content="[^"]*"\s*\/>/, () => `<meta name="description" content="${escapeHtml(page.summary)}" />`);
    await writeFile(path.join(directory, "index.html"), content);
  }
  await writeFile(path.join(outputRoot, "404.html"), html);
  await writeFile(path.join(outputRoot, ".nojekyll"), "");
}
