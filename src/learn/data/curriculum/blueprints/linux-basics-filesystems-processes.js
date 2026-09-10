// Authoring blueprint: Linux Basics, Filesystems & Processes.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Locate experiment files, separate results from diagnostics, explain access failures and manage the correct child process using a visible model of what Linux does.",
  "outcomes": [
    "Trace an absolute or relative path and explain when the shell's location changes",
    "Predict stdout/stderr destinations and the first failed permission check",
    "Distinguish a paused, running or exited child and retrieve its status",
    "Complete and explain a file-and-process investigation with changed inputs"
  ],
  "prerequisites": [],
  "sequence": [
    "Separate terminal, shell and kernel; explain disposable practice setup",
    "Trace paths through a tree before using navigation and inspection commands",
    "Route outputs and errors through visible connections",
    "Apply path traversal and file access as successive permission gates",
    "Connect inherited environment and parent/child lifecycle to commands",
    "Combine the mechanisms in an investigation; use links and diagnostics as deeper branches"
  ],
  "visual": {
    "type": "Stepped path tree",
    "question": "Which directory does this relative path reach, and when does the shell move?",
    "interaction": "Predict, step through components, inspect a missing entry or file target, then reset and compare an absolute path."
  },
  "additionalVisuals": [
    {
      "type": "Stream routing lab",
      "question": "Where do the result and warning go?",
      "interaction": "Predict destinations, change routes/status, reveal exact output and explain descriptor ordering in optional depth."
    },
    {
      "type": "Permission gates lab",
      "question": "Which access check fails first?",
      "interaction": "Toggle directory listing/search and file-read bits, then compare listing a directory with accessing a known filename."
    },
    {
      "type": "Process lifecycle lab",
      "question": "Is this child stopped, alive or finished?",
      "interaction": "Start, pause, continue, terminate and collect status with the parent still visible."
    },
    {
      "type": "Static environment and link diagrams",
      "question": "What is copied or shared?",
      "interaction": "Trace parent/child environment values and the difference between hard-link identity and a symbolic link's stored path."
    }
  ],
  "practice": {
    "task": "Find the data from reports, preview it, save warning lines with original line numbers, count results and interpret an absent-match status; then change the input.",
    "success": "The independently verified output matches the fixtures, the learner explains destinations and status, and the transfer distinguishes rebuilding a report from appending duplicate results."
  },
  "misconceptions": [
    "A terminal, a shell and the Linux kernel are different layers",
    "Directory read and search permissions answer different questions",
    "Stopped is not exited; Bash may already have reaped a child before wait retrieves its stored status",
    "A browser teaching model is not a general shell runtime"
  ],
  "sources": [
    "https://man7.org/linux/man-pages/man7/path_resolution.7.html",
    "https://www.gnu.org/software/bash/manual/html_node/Redirections.html",
    "https://www.gnu.org/software/coreutils/manual/html_node/Mode-Structure.html",
    "https://man7.org/linux/man-pages/man7/signal.7.html"
  ],
  "depth": "core",
  "reviewFocus": "User-approved on 2026-09-09. Keep platform and model limits, verified native fixtures, four distinct learning questions and independent transfer. See PROGRAMMING-REWRITE-LINUX.md for recorded evidence."
};
