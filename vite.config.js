import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

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

export default defineConfig({
  plugins: [learningArtifacts(), react()],
  base: "/",
  build: { manifest: true },
});
