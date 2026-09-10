// Authoring blueprint: Bash Scripting & Command-Line Automation.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Coordinate a complete report job through explicit argument, status, process-scope and publication contracts.",
  "outcomes": [
    "Predict exact arguments after expansion, splitting and globbing",
    "Preserve filenames and empties through arrays and function arguments",
    "Explain default pipeline status, pipefail, PIPESTATUS and conditional errexit behavior",
    "Distinguish parent state from exported child or pipeline-loop state",
    "Stage a complete report and retain previous output on failure",
    "Independently collect validated reports in supplied order with cleanup and repeat-run checks"
  ],
  "prerequisites": [
    "Linux Basics, Filesystems & Processes",
    "Python Basics: Types, Control Flow, Functions & Modules",
    "Scientific File Formats, Schemas & Reliable Data I/O"
  ],
  "sequence": [
    "Motivate a repeatable report and distinguish Bash, terminal, Linux, sh and PowerShell",
    "Explain explicit Bash invocation, shebang, PATH and text-file assumptions",
    "Trace quoted/unquoted values into exact arguments with spaces, patterns and empty input",
    "Separate code from data and option parsing from argument boundaries",
    "Use required/default parameters, arrays, functions, local scope and quoted forwarding",
    "Contrast child environment, source, current directory and command-substitution newline behavior",
    "Separate stdout/stderr from status and contrast pipeline status with component statuses",
    "Distinguish grep no-match from errors and show the conditional-function errexit counterexample",
    "Visit filenames using globs and quoted arrays, compare piped and redirected loops",
    "Trace staged versus public content through successful or failed production",
    "Run a complete Python CSV-to-JSON worker and Bash wrapper with all fixtures",
    "Verify success, empty/zero data, malformed input, missing files and repeated publication",
    "Independently collect several reports, preserving input order and rejecting partial collections",
    "Transfer same-file publication to manifests, then bridge to OS and Threads"
  ],
  "visual": {
    "type": "Expansion to argument boundaries",
    "question": "How does one filename become two arguments?",
    "interaction": "Select spaces, wildcard or empty value and quoted/unquoted expansion; step parse, expand, split/glob and launch."
  },
  "visuals": [
    {
      "type": "Pipeline output/status channels",
      "question": "Can a producer emit data and still fail?",
      "interaction": "Change producer and consumer statuses and toggle pipefail; inspect the parent-visible status."
    },
    {
      "type": "Staging versus public destination",
      "question": "What does a reader see after a failed replacement?",
      "interaction": "Step partial production, success/failure, rename and cleanup while the previous complete public report remains visible."
    }
  ],
  "practice": {
    "task": "Build collect-reports.sh with an output and one or more input reports. Validate JSON count/mean contracts in Python, preserve spaces and order, publish a single complete array or retain previous output.",
    "success": "Native Bash and Python verify valid/empty/malformed/missing/zero cases, repeated-run identity, reversed input order, schema rejections, discarded partial worker output and cleanup after an owned TERM interruption."
  },
  "misconceptions": [
    "Quoting and option parsing are different boundaries",
    "Variable expansion does not generally reparse its value as shell operators",
    "A command string is not an argument array",
    "Printed output does not prove status success",
    "pipefail does not roll back output",
    "set -e does not stop every failing command",
    "Export does not send child assignments back to the parent",
    "No glob match differs from a literal pattern under nullglob",
    "A successful rename gives file visibility, not crash durability or multi-file atomicity"
  ],
  "sources": [
    "https://www.gnu.org/software/bash/manual/bash.html",
    "https://www.shellcheck.net/wiki/SC2086",
    "https://missing.csail.mit.edu/2020/shell-tools/",
    "https://docs.python.org/3/library/csv.html",
    "https://docs.python.org/3/library/json.html",
    "https://man7.org/linux/man-pages/man2/rename.2.html"
  ],
  "depth": "core",
  "reviewFocus": "Keep explicit failure policy and complete worker fixtures. Native Bash evidence must cover quoted argv, conditional -e behavior, pipeline status and staging cleanup. Unix/Linux assumptions and ordinary local publication boundaries remain explicit; title and progress IDs are retained. Video/notes are optional supplements, not a shell specification.",
  "designRecord": "docs/teaching/bash-completion-verification.md"
};
