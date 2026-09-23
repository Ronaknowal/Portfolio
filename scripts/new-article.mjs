import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { createArticle } from "./lib/article-content.mjs";

try {
  const { values } = parseArgs({ options: { slug: { type: "string" }, title: { type: "string" } } });
  if (!values.slug || !values.title) throw new Error('Usage: npm run article:new -- --slug my-article --title "My article"');
  const directory = await createArticle(fileURLToPath(new URL("..", import.meta.url)), values.slug, values.title);
  console.log(`Draft created: ${directory}\nRead docs/writing/ARTICLE-AUTHORING.md before publishing.`);
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
