// A bounded teaching model of these six commands, not a shell interpreter.
// Semantics: GNU Bash manual, Redirections and Pipelines (reviewed 2026-09-09).
export const LINUX_STREAM_ROUTES = [
  { id: "terminal", label: "Both to terminal", suffix: "", stdout: "terminal", stderr: "terminal" },
  { id: "stdout-file", label: "Save results", suffix: "> result.txt", stdout: "result.txt", stderr: "terminal" },
  { id: "split", label: "Separate files", suffix: "> result.txt 2> errors.txt", stdout: "result.txt", stderr: "errors.txt" },
  { id: "pipe", label: "Count result lines", suffix: "| wc -l", stdout: "wc -l", stderr: "terminal" },
  { id: "merge", label: "File, then duplicate", suffix: "> combined.txt 2>&1", stdout: "combined.txt", stderr: "combined.txt", advanced: true },
  { id: "reverse", label: "Duplicate, then file", suffix: "2>&1 > combined.txt", stdout: "combined.txt", stderr: "terminal", advanced: true },
];

const explanations = {
  terminal: "No outer redirection changes either destination. The metric and warning both reach the terminal, but they remain separate streams.",
  "stdout-file": "The unnumbered > redirects descriptor 1 (stdout). Descriptor 2 is untouched, so the warning still reaches the terminal.",
  split: "Each numbered stream has its own destination. The result file contains the metric; the error file contains the warning. This command prints neither to the terminal.",
  pipe: "The pipe connects stdout to wc's stdin (descriptor 0). wc receives one newline and prints 1. The warning bypasses wc and reaches the terminal through stderr.",
  merge: "First > combined.txt points stdout at the file. Then 2>&1 gives stderr that same destination. Both writes enter the file, metric first in this sequential program.",
  reverse: "First 2>&1 copies stdout's current destination: the terminal. Then > combined.txt changes stdout alone. stderr keeps the terminal destination; it does not follow later changes to stdout.",
};

export function linuxStreamModel(routeId = "stdout-file", producerStatus = 0) {
  const route = LINUX_STREAM_ROUTES.find(candidate => candidate.id === routeId);
  if (!route || ![0, 7].includes(producerStatus)) throw new RangeError("Choose a supported routing preset and exit status.");
  const stdout = "metric=18\n";
  const stderr = "warning: tiny sample\n";
  const files = {};
  for (const [destination, content] of [[route.stdout, stdout], [route.stderr, stderr]]) {
    if (destination !== "terminal" && destination !== "wc -l") files[destination] = (files[destination] || "") + content;
  }
  const terminal = [];
  if (route.stdout === "terminal") terminal.push({ stream: "stdout", content: stdout });
  if (route.stderr === "terminal") terminal.push({ stream: "stderr", content: stderr });
  if (route.id === "pipe") terminal.push({ stream: "wc stdout", content: "1\n" });
  return {
    ...route,
    stdoutText: stdout,
    stderrText: stderr,
    files,
    terminal,
    producerStatus,
    shellStatus: route.id === "pipe" ? 0 : producerStatus,
    count: route.id === "pipe" ? 1 : null,
    explanation: explanations[route.id],
    command: `bash --noprofile --norc -c '\n  printf "metric=18\\n"\n  printf "warning: tiny sample\\n" >&2\n  exit ${producerStatus}\n'${route.suffix ? ` ${route.suffix}` : ""}`,
    summary: `stdout goes to ${route.stdout}; stderr goes to ${route.stderr}.${route.id === "pipe" ? " wc counts one newline and sends 1 to the terminal." : ""}`,
  };
}
