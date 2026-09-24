// Authoring blueprint: Git, GitHub & Collaborative Version Control.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Make a coherent change by tracing stored snapshots, branch ancestry, conflict decisions and local knowledge of a remote, then locate an introduced regression.",
  "outcomes": [
    "Predict which file version an ordinary commit records and choose the intended undo layer",
    "Trace branch pointers, fast-forward and merge ancestry",
    "Separate fetch from integration and resolve a conflict while checking its meaning",
    "Review a pull-request workflow and independently commit only one intended file",
    "Use a reproducible good/bad predicate to locate a regression in disposable history"
  ],
  "prerequisites": [],
  "sequence": [
    "Start with a shared report problem and distinguish Git from a hosting service",
    "Introduce the working tree, index, commit and branch reference",
    "Explain the few Bash primitives and initialize each complete example in a fresh folder",
    "Trace edit, stage, edit again and commit across three stored versions",
    "Choose restore, revert or private-history rewrite by intended layer",
    "Watch branch creation, divergence, fast-forward refusal and a two-parent merge",
    "Compare base and both edits, resolve text, stage the decision and commit the merge",
    "Inspect ignored files, secrets and recovery boundaries without assuming Git backs up everything",
    "Trace colleague push, local work, fetch and explicit fast-forward integration",
    "Connect a branch, review, tests and hosting permissions in a pull request",
    "Explore rebase and bisect as deeper branches with explicit assumptions",
    "Independently commit a note while leaving a changed result unstaged"
  ],
  "visual": {
    "type": "Three stored snapshots with transfer arrows",
    "question": "Will committing record version 2 or the version 3 currently in the editor?",
    "interaction": "Step editing and staging; choose whether to stage again; compare exact HEAD/index/worktree contents and both status columns."
  },
  "visuals": [
    {
      "type": "Commit ancestry graph",
      "question": "Why does one merge only move a name while another creates a commit?",
      "interaction": "Change whether main diverges; step parent edges, HEAD, branch references and checked-out files."
    },
    {
      "type": "Local versus remote knowledge",
      "question": "Did fetch change the files or update what this clone knows?",
      "interaction": "Advance shared, tracking and local references separately; add local divergence and inspect fast-forward refusal."
    },
    {
      "type": "Three-way textual decision",
      "question": "Can a syntactically resolved conflict still lose intended meaning?",
      "interaction": "Compare base/ours/theirs, choose a resolution, step edited/unmerged/staged/committed state and inspect omitted intent."
    }
  ],
  "practice": {
    "task": "Perform a complete change in a disposable repository, locate a regression with a binary predicate, then stage only a unit note while leaving the result edit outside the commit.",
    "success": "Eleven complete Bash examples and both alternatives in all four investigations agree with native Git. The independent task proves committed and working contents differ as intended; regression search returns the first bad commit and restores main."
  },
  "misconceptions": [
    "Stage captures content now; it does not subscribe to future edits",
    "Branch creation does not duplicate every file",
    "Fetch does not automatically merge into the checked-out branch",
    "Removing conflict markers does not verify the intended result",
    "A clean status does not prove correctness or a remote backup",
    "Bisect needs a meaningful reproducible classification; flaky or nonmonotonic histories need care"
  ],
  "sources": [
    "https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository",
    "https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell",
    "https://git-scm.com/docs/git-restore",
    "https://git-scm.com/docs/git-merge",
    "https://git-scm.com/docs/git-fetch",
    "https://git-scm.com/docs/git-bisect",
    "https://missing.csail.mit.edu/2026/version-control/"
  ],
  "depth": "core",
  "reviewFocus": "Linux comes next, so explain required Bash syntax here. Use disposable local repositories and local identity; remote tests stay local. Verify real object/reference/file states and distinguish GitHub instructions from executed hosting actions. Curate MIT notes/video and Pro Git articles; no full-video-viewing claim.",
  "designRecord": "docs/teaching/next-three-reimplementation.md"
};
