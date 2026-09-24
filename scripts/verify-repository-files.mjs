import assert from "node:assert/strict";
import { readdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repositoryRoot = fileURLToPath(new URL("..", import.meta.url));
const excludedDirectories = new Set(["node_modules", "__pycache__", ".venv", "venv", ".git"]);
const machineArtifact = file => /(?:^|\/)\.env(?:\.|$)/.test(file) && !/\.example$/.test(file) || /\.(?:pyc|pyo|pyd|swp)$/.test(file) || /(?:^|\/)(?:\.DS_Store|Thumbs\.db|desktop\.ini)$/.test(file);
async function collect(directory) {
  const files = [];
  for (const entry of await readdir(path.join(repositoryRoot, directory), { withFileTypes: true })) {
    if (entry.isSymbolicLink()) continue;
    const file = `${directory}/${entry.name}`;
    if (entry.isDirectory()) {
      if (/^apps\/[^/]+$/.test(directory) && ["dist", "build", ".next", ".vite", "playwright-report", "test-results"].includes(entry.name)) continue;
      if (!excludedDirectories.has(entry.name) && !entry.name.endsWith(".egg-info")) files.push(...await collect(file));
    } else if (!machineArtifact(file)) files.push(file);
  }
  return files;
}
function ignored(files) {
  // Check repository rules deterministically, independent of a machine's global excludes.
  const globalExcludes = process.platform === "win32" ? "NUL" : "/dev/null";
  const result = spawnSync("git", ["-c", `core.excludesFile=${globalExcludes}`, "check-ignore", "--no-index", "--stdin", "-z"], {
    cwd: repositoryRoot, input: files.join("\0") + "\0", encoding: "utf8", maxBuffer: 16 * 1024 * 1024,
  });
  if (![0, 1].includes(result.status)) throw new Error(result.stderr || result.error?.message || "git check-ignore failed");
  return result.stdout.split("\0").filter(Boolean);
}
const roots = ["src", "public", "content", "apps", "scripts", "docs", ".github"];
const required = (await Promise.all(roots.map(collect))).flat();
for (const entry of await readdir(repositoryRoot, { withFileTypes: true })) {
  if (entry.isFile() && (/\.md$/i.test(entry.name) || ["package.json", "package-lock.json", "vite.config.js", "index.html", ".gitignore", ".env.example"].includes(entry.name))) required.push(entry.name);
}
const futureFiles = ["content/articles/my-article/body.md", "content/articles/my-article/metadata.json", "public/articles/my-article/build.log", "public/learn-assets/example.py", "src/articles/generated/catalogue.js", "docs/archive/scratch/evidence.json", "docs/archive/dist/example.md", "apps/example/src/main.ts", ".env.production.example"];
assert.deepEqual(ignored([...required, ...futureFiles]), [], "Required source, documentation or public assets are ignored by repository rules");
const disposable = ["node_modules/example/index.js", "dist/index.html", "scratch/report.json", "scratch_local.py", "apps/example/dist/index.html", ".env", ".env.production", "apps/example/.env.local", "public/learn-assets/__pycache__/example.cpython-312.pyc"];
assert.deepEqual(new Set(ignored(disposable)), new Set(disposable), "Caches, builds or secrets are no longer ignored");
async function verifyPublicDirectory(directory) {
  for (const entry of await readdir(path.join(repositoryRoot, directory), { withFileTypes: true })) {
    const file = `${directory}/${entry.name}`;
    assert.ok(!entry.isSymbolicLink(), `Public assets must not link outside their tree: ${file}`);
    assert.ok(!excludedDirectories.has(entry.name) && !machineArtifact(file), `Vite would deploy a cache or private machine file: ${file}`);
    if (entry.isDirectory()) await verifyPublicDirectory(file);
  }
}
await verifyPublicDirectory("public");
console.log(`Repository ignore checks passed for ${required.length} current files and ${futureFiles.length} future-path probes; caches and secrets remain excluded. Machine-global excludes are outside this check.`);
