import { fileURLToPath } from "node:url";
import { generateArticles } from "./lib/article-content.mjs";

const articles = await generateArticles(fileURLToPath(new URL("..", import.meta.url)));
console.log(`Articles: ${articles.length} published; drafts excluded from the site.`);
