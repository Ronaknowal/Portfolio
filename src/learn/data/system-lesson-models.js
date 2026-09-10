export const gitTrace = [
  { command: ": # initial committed version", versions: [1, 1, 1], status: "clean", note: "HEAD, index and working file all contain version 1." },
  { command: "printf 'version 2\\n' > report.txt", versions: [1, 1, 2], status: " M report.txt", note: "Editing changes only the working file. The index still contains version 1." },
  { command: "git add report.txt", versions: [1, 2, 2], status: "M  report.txt", note: "Add copies the current file content into the index. It does not commit it." },
  { command: "printf 'version 3\\n' > report.txt", versions: [1, 2, 3], status: "MM report.txt", note: "There are now two comparisons: index differs from HEAD, and worktree differs from index." },
  { command: 'git commit -qm "Record staged version two"', versions: [2, 2, 3], status: " M report.txt", note: "Commit records the staged version 2. The newer working edit remains unstaged." },
];

export function permissionModel(mode, subject, directory) {
  const bits = Number(mode[{ owner: 0, group: 1, other: 2 }[subject]]);
  const read = Boolean(bits & 4), write = Boolean(bits & 2), execute = Boolean(bits & 1);
  return directory ? [
    ["List entry names", read, "Read permission; opening an entry also needs traversal."],
    ["Traverse a known entry", execute, "Search/execute permission on this directory."],
    ["Create or remove entries", write && execute, "Both write and search; sticky-bit rules can further restrict deletion."],
  ] : [
    ["Read file contents", read, "Assumes its parent directories can be traversed."],
    ["Modify file contents", write, "Does not decide whether its directory entry can be deleted."],
    ["Attempt program execution", execute, "A valid executable/interpreter and mount policy are still required."],
  ];
}
