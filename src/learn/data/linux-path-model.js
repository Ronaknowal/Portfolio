// A fixed, permission-granted tree without symlinks. Walk components before
// simplifying: missing/.. and file/.. must fail, not cancel lexically.
export const linuxPathEntries = Object.freeze([
  { path: "/", name: "/", kind: "directory", parent: null },
  { path: "/project", name: "project", kind: "directory", parent: "/" },
  { path: "/project/data", name: "data", kind: "directory", parent: "/project" },
  { path: "/project/data/raw", name: "raw", kind: "directory", parent: "/project/data" },
  { path: "/project/data/raw/run 1.csv", name: "run 1.csv", kind: "file", parent: "/project/data/raw" },
  { path: "/project/reports", name: "reports", kind: "directory", parent: "/project" },
  { path: "/project/reports/summary.txt", name: "summary.txt", kind: "file", parent: "/project/reports" },
]);

const entries = new Map(linuxPathEntries.map(entry => [entry.path, entry]));

export const linuxPathPresets = [
  { id: "up-and-across", label: "Reach a sibling branch", cwd: "/project/data/raw", path: "../../reports", operation: "cd" },
  { id: "absolute", label: "Start from the root", cwd: "/project/data/raw", path: "/project/reports", operation: "cd" },
  { id: "wrong-start", label: "Try the wrong starting point", cwd: "/project/data/raw", path: "reports", operation: "cd" },
  { id: "dot-file", label: "Locate a file beside you", cwd: "/project/data/raw", path: "./run 1.csv", operation: "locate" },
  { id: "file-cd", label: "Try to enter a file", cwd: "/project/data/raw", path: "./run 1.csv", operation: "cd" },
];

export function resolveLinuxPath({ cwd = "/project/data/raw", path = "../../reports", operation = "cd" } = {}) {
  if (entries.get(cwd)?.kind !== "directory") throw new Error("Choose a starting directory in the lab tree.");
  if (typeof path !== "string") throw new TypeError("Path must be a string.");
  if (!["cd", "locate"].includes(operation)) throw new Error("Unsupported path operation.");

  const absolute = path.startsWith("/");
  const segments = path.split("/").filter(Boolean);
  let location = absolute ? "/" : cwd;
  const steps = [{
    kind: "start", location, cwd, segmentIndex: -1,
    title: absolute ? "Begin at /" : "Begin at the working directory",
    explanation: absolute
      ? "The leading slash chooses the root as the lookup starting point. It does not move the shell."
      : `There is no leading slash. Begin looking inside ${cwd}.`,
  }];

  function fail(title, explanation, segmentIndex, code) {
    steps.push({ kind: "error", location, cwd, title, explanation, segmentIndex, code });
    return { absolute, segments, steps, ok: false, result: null, cwd, error: code, prediction: "error" };
  }

  if (!path.length) return fail("An empty path has no destination", "Supply a pathname. This input is one explicit argument, so an empty string is not the same as running cd with no argument.", -1, "ENOENT");
  if (path.includes("\0")) return fail("A pathname cannot contain a NUL character", "The NUL byte terminates path strings in the operating-system interface.", -1, "EINVAL");

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    if (entries.get(location).kind !== "directory") {
      return fail("A file cannot contain the next component", `${location} is a file. There is no directory inside it in which to look up ${segment}.`, index, "ENOTDIR");
    }
    const from = location;
    if (segment === ".") {
      steps.push({ kind: "stay", location, from, cwd, segmentIndex: index, title: ". means stay here", explanation: `Keep looking in ${location}. Dot names this directory itself.` });
      continue;
    }
    if (segment === "..") {
      location = entries.get(location).parent || "/";
      steps.push({ kind: "move", location, from, cwd, segmentIndex: index, title: ".. means one parent", explanation: from === "/" ? "The root has no higher directory. /.. stays at /." : `Move the lookup marker from ${from} to its parent, ${location}.` });
      continue;
    }
    const candidate = `${location === "/" ? "" : location}/${segment}`;
    if (!entries.has(candidate)) {
      return fail(`No entry named ${segment} here`, `Looked inside ${location}; that exact name is absent. Lookup stops here, even if a later component is "..". The shell stays in ${cwd}.`, index, "ENOENT");
    }
    location = candidate;
    steps.push({ kind: "move", location, from, cwd, segmentIndex: index, title: `Found ${segment}`, explanation: `${segment} is a ${entries.get(location).kind} directly inside ${from}. Follow that one branch.` });
  }

  const kind = entries.get(location).kind;
  if (kind === "file" && (operation === "cd" || path.endsWith("/"))) {
    return fail("Found a file, but a directory is required", operation === "cd"
      ? `The path exists, but cd can only enter directories. ${location} is a regular file, so the working directory stays ${cwd}.`
      : "The trailing slash requires a directory. This entry is a regular file, so lookup fails.", segments.length - 1, "ENOTDIR");
  }

  const nextCwd = operation === "cd" ? location : cwd;
  steps.push({ kind: "finish", location, cwd: nextCwd, segmentIndex: segments.length, title: operation === "cd" ? "cd succeeds: now move the shell" : `Located the ${kind}`, explanation: operation === "cd"
    ? `All components resolved to a directory. Only now does the shell's working directory change to ${location}.`
    : `The complete path identifies ${location}. Looking up an entry does not change the shell's working directory.` });
  return { absolute, segments, steps, ok: true, result: location, kind, cwd: nextCwd, prediction: kind };
}
