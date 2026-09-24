import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { generateArticles } from "./scripts/lib/article-content.mjs";
import { writeSiteEntries } from "./scripts/lib/site-build.mjs";

const repositoryRoot = fileURLToPath(new URL(".", import.meta.url));
function generateLearningArtifacts() {
  // A fresh process avoids retaining stale imports when an author edits a plan.
  execFileSync(process.execPath, ["scripts/generate-learning-artifacts.mjs"], { cwd: repositoryRoot, stdio: "inherit" });
}

function learningArtifacts() {
  return {
    name: "learning-artifacts",
    buildStart: generateLearningArtifacts,
    handleHotUpdate({ file }) {
      const normalized = file.replaceAll("\\", "/");
      if (/\/src\/learn\/data\/(curriculum\/|track-definitions\.js$|topic-id\.js$|lesson-manifest\.json$|topics\/[^/]+\.jsx$)/.test(normalized)) {
        generateLearningArtifacts();
      }
    },
  };
}

function articlePublishing() {
  let outputRoot;
  return {
    name: "article-publishing",
    enforce: "post",
    configResolved(config) { outputRoot = path.resolve(config.root, config.build.outDir); },
    buildStart() { return generateArticles(repositoryRoot); },
    configureServer(server) {
      const contentRoot = path.join(repositoryRoot, "content/articles");
      let timer;
      server.watcher.add(contentRoot);
      const onContentChange = (_event, file) => {
        if (!path.resolve(file).startsWith(contentRoot + path.sep)) return;
        clearTimeout(timer);
        timer = setTimeout(() => {
          try {
            execFileSync(process.execPath, ["scripts/generate-articles.mjs"], { cwd: repositoryRoot, stdio: "pipe" });
            server.ws.send({ type: "full-reload" });
          } catch (error) {
            const message = error.stderr?.toString() || error.message;
            server.config.logger.error(message);
            server.ws.send({ type: "error", err: { message, stack: "", plugin: "article-publishing" } });
          }
        }, 80);
      };
      server.watcher.on("all", onContentChange);
      server.httpServer?.once("close", () => { clearTimeout(timer); server.watcher.off("all", onContentChange); });
    },
    writeBundle() { return writeSiteEntries(repositoryRoot, outputRoot); },
  };
}

export default defineConfig({
  plugins: [learningArtifacts(), articlePublishing(), react()],
  base: "/",
  build: { manifest: true },
});
