# Linux lesson improvement

9 September 2026 implementation record. Status: implemented, technically verified, and approved by the user as the current quality reference. Original scope: **Linux Basics, Filesystems & Processes only**. The original programming sequence in this historical implementation was Git → Linux → Bash; the live curriculum now places OS after Linux. The targeted 10 September resource/bridge update is recorded below and in [the current increment](SYSTEMS-STRUCTURES-IMPLEMENTATION.md).

## Review the result

[Open the Linux lesson](http://127.0.0.1:5173/learn/topic/linux-basics-filesystems-processes). The local development server must be running; from this project use `npm.cmd run dev -- --host 127.0.0.1`. Nothing was deployed.

The [lesson source](src/learn/data/topics/linux-basics-filesystems-processes.jsx) keeps its existing URL, eight runnable examples, useful conditions and failure explanations. The improvements implement [the teaching standard](LESSON-TEACHING-STANDARD.md) with several representations serving distinct learning questions.

## What was weak and what changed

| Learning hurdle | Previous limitation | Improvement and useful investigation |
| --- | --- | --- |
| Path lookup versus shell location | Prose and terminal output required learners to picture the tree themselves. | A connected tree with separate lookup and shell markers. Walk `../../reports`, compare an absolute path, and try to enter a file. The shell moves only after successful `cd`. |
| stdout, stderr and pipelines | A table described channels, but did not show their connections. | Routing diagram with predictions and reveal, exact destination contents, six routes and a failure-status option. Compare saving stdout alone with separate files; then inspect why a warning bypasses `wc`. Full Bash syntax is optional until wanted. |
| Permissions on a whole path | One octal table listed operations without showing where a read was blocked. | Sequential permission gates plus independently controlled directory read/search and file read. Try reading a known file without being allowed to list names. Letter meanings precede octal notation; the earlier mode table remains in optional depth. |
| Process state | Commands and final output did not expose paused versus exited state. | Parent/child lifecycle explorer: start, pause, continue, terminate, collect status. The parent remains visible and state feedback explains what each action changes. |
| Inherited environment and links | Two additional mechanisms depended mainly on prose. | Separate static diagrams for copied child environment and hard-link identity versus a symlink's stored path. |
| First-time pacing | Common setup syntax and advanced details appeared early with limited scaffolding. | Explained temporary-directory setup before use, plain-language bridges before commands, an explicit first-pass route, and optional branches for links, diagnostics, remote machines, and advanced redirection. |
| Independent application | Practice was mostly individual prompts with immediate full-answer reveals. | A complete project investigation with setup, tasks, a hint, runnable solution, expected output, changed-input transfer, and readiness questions. |

All four labs have different jobs; their number is a consequence of the concepts, not a template quota. They are bounded browser models, not shell interpreters, and clearly state their limits. The permission fixture uses a simpler path than the path lab; both contain invented measurement data. Each shell example creates its own disposable project.

## Coverage and intentional depth

Core outcomes: locate and inspect files; explain absolute/relative paths and quoting; route results and diagnostics; identify the first failed permission check; explain child environment inheritance; distinguish paused, alive and exited processes; complete an evidence-based investigation.

Optional depth preserves hard/symbolic links, storage/resource diagnosis, shared-machine practices, ordinary octal modes and class selection, redirection ordering, and pipeline status. Networking administration, full shell scripting, service management, ACL configuration, scheduler operations and security policy remain beyond this lesson.

The investigation starts inside `reports`, asks learners to find and preview a CSV, save warning lines with their original line numbers, count the result and interpret an absent-match exit status. The transfer task adds a warning and tests the difference between rebuilding a report and appending duplicated results. It also reconnects file access to the permission gates.

Technical refinements include explaining that Bash may reap a child asynchronously before `wait` retrieves its stored termination status. The process model distinguishes “alive and not stopped” from actively consuming CPU; its `sleep` process can be waiting on a timer. The guided lifecycle resumes a stopped child before TERM, and states that real timing and signal handlers have additional cases.

## Verification completed

| Check | Result |
| --- | --- |
| `node scripts/verify-linux-lesson.mjs` | Passed: 11 path cases, all 8 permission gate combinations, class-selection cases, process transitions, and 12 stream/status combinations. Exports native fixtures; no browser data is treated as independent proof of OS behavior. |
| `scripts/verify-linux-native.py` in Ubuntu/WSL as unprivileged `nobody` | Passed: all 8 existing examples, complete investigation and changed-input variant; all 6 stream routes at exit statuses 0 and 7; actual path resolution; 8 permission gate combinations and owner no-fallback; owned child STOP/CONT/TERM lifecycle. |
| `scripts/review-linux-lesson.cjs` | Passed at 1440 px and 390 px: four labs, path failure/success, reset state, stream routes/status, all permission gate states, mode class selection, lifecycle transitions, answer reveals, nine visible example outputs, navigation targets, keyboard activation/focus movement, no page errors and no document/lab overflow. |
| Visual review | Inspected fresh desktop and phone screenshots. Tree/lookup relationship, stdout/stderr routes, permission gates and parent/child state are represented directly. Exact tables and output remain available. |
| `npm.cmd run build` | Passed. Existing unrelated JSX warnings in the Bayesian-networks topic and the large shared-content chunk warning remain. Build output is saved under `scratch/linux-lesson-review/build.log`. |

Native tests use temporary Linux files, restore permission fixtures before cleanup and signal only the child they create. Permission-gate behavior is tested with the applicable bits as fixture owner, not by creating privileged accounts; owner no-fallback is separately exercised with mode `047`. No elevated Linux user is used. The initial WSL approval-review service failure was resolved before the successful native run; no verification blocker remains.

One test correction mattered: after `2>&1 > combined.txt`, the producer's stderr message travels to the outer stdout destination. The UI groups terminal messages by their producer stream, while the native harness captures outer stdout/stderr separately. The corrected test verifies this distinction against actual Bash.

Screenshots are under [scratch/linux-lesson-review](scratch/linux-lesson-review): `paths`, `streams`, `permissions`, and `process`, each at 1440 and 390 px. Browser checks use an existing Playwright package selected via `PLAYWRIGHT_PACKAGE`; no website dependency was added. The older batch-four browser script was adapted to the additional Linux labs and investigation instead of assuming exactly one lab/eight outputs.

The adapted `scripts/review-programming-batch-four.cjs` also passed its existing Git/Linux regression checks at both widths; Git content was not edited in this increment.

This is technical and author visual review, not a study with actual first-time learners or a complete assistive-technology audit. The user subsequently approved this output as the current quality reference. That acceptance does not establish results from an actual novice study.

## Research used

Original prose, diagrams, controls and exercises are used. Primary references checked for the new explanations/models:

- [Linux pathname resolution](https://man7.org/linux/man-pages/man7/path_resolution.7.html): component traversal, absolute/relative starts, directory search, final-entry requirements and failures.
- [GNU permission structure](https://www.gnu.org/software/coreutils/manual/html_node/Mode-Structure.html): owner/group/other selection and permission meaning.
- [GNU Bash redirections](https://www.gnu.org/software/bash/manual/html_node/Redirections.html) and [pipelines](https://www.gnu.org/software/bash/manual/html_node/Pipelines.html): descriptor routing, order and pipeline status.
- [Linux signals](https://man7.org/linux/man-pages/man7/signal.7.html): stopping, continuing and termination behavior. Bash-specific status behavior is also checked by the runnable examples.

## Continuation

The original Linux implementation is complete, with the above model/native/browser evidence and user approval. On 10 September its four labs and existing programs were retained; annotated Missing Semester notes/video were added and the closing bridge was corrected to the live OS next position. The Linux model and desktop/mobile browser checks passed again; unchanged native examples retain the earlier explicitly dated evidence. [The current handoff](LESSON-AUTHORING-HANDOFF.md) owns active scope and the continuation queue; this report does not instruct a Bash rewrite.
