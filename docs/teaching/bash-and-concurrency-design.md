# Bash and concurrency: design and source ledger

Designed 10 September 2026 for the authorized completion of Programming & Scientific Computing. Teaching policy: LESSON-TEACHING-STANDARD.md. Linux remains the user-approved reference; these lessons require author verification and later user acceptance.

## Bash Scripting & Command-Line Automation

Scope review: retain the title. Its useful missing coverage belongs here: argument boundaries, expansion, status versus output, explicit failure handling, working-directory/environment scope, bounded orchestration and artifact publication. Preserve earlier variables/defaults, loops/functions, redirection/pipelines and automation coverage; replace the missing train.py with complete fixtures. Do not promise `set -e` stops on every failure. Linux owns basic files/processes; data I/O owns CSV parsing/schema validation; concurrency owns synchronization.

Outcomes: predict exact argv; keep data separate from executable syntax; handle success/no-match/error distinctly; explain pipeline status; pass parameters through functions/arrays; stage a complete artifact and expose only successful results; diagnose an interrupted or failed run.

Flow: motivating repeatable report → Bash execution/environment → expansion and argument lab → variables/arguments/arrays/functions → status/pipeline lab → file iteration and process boundaries → staged report lifecycle lab → complete runnable report and failure case → independent changed-input report → limitations, debugging, resources and OS bridge. Three labs address different mechanisms; none is a real shell emulator. Each exposes its model boundary and asks for predictions and transfers. Plain shell examples run in a disposable directory under native Bash; browser models must be compared with native argv/status/lifecycle results.

Independent practice: summarize several named reports while preserving spaces, distinguish missing input from empty valid input, never publish a partial summary, verify repeat-run behavior and retain useful diagnostics. Provide hints before a complete solution. Interesting applications: filenames as argument-injection bugs; staging a generated experiment manifest or static-site artifact with same-filesystem rename. Explain the mechanism and durability/last-writer limits rather than presenting a magical safe-automation recipe.

Sources checked: MIT Missing Semester 2020 Shell Tools and Scripting notes (including the linked video identity), ShellCheck SC2086, GNU Bash manual indexed results. Individual GNU HTML sections and full manual timed out through the web reader; native `help set`, `help trap`, parameter expansion and pipeline probes will provide runtime evidence. Annotate the video as an older alternate walkthrough, not watched in full. Use explicit status logic rather than copying the lecture's treatment of every grep nonzero as no-match.

## Threads, Concurrency, Locks & Deadlocks

Scope review: retain title and existing brief's producer/consumer and two-lock goals. Add result/exception collection, bounded queue lifecycle, cooperative cancellation and GIL/free-threading nuance because correctness includes termination. OS owns address spaces/scheduling; GPU modules own device synchronization/memory ordering. Avoid silently importing the whole asyncio/distributed-systems curriculum.

Outcomes: distinguish shared objects from private execution state; exhibit a lost update and protect the entire invariant; explain wait/recheck/reacquire; draw a wait-for cycle and apply one lock order; finish bounded concurrent work with explicit result, error and shutdown channels. Flow: two workers one total → thread/process map → race and lock lab → deterministic native race/repair → condition/predicate lab → producer/consumer code → deadlock and ordering lab → ordered native transfers → futures, cancellation, bounded queue practice → runtime caveats and next-module bridge.

Three distinct causal investigations: freely schedule reads/writes and blocked lock attempts; notify a waiting consumer while the predicate is false or stolen; acquire opposite versus shared lock orders and trace owners/waiters. Text state, disabled invalid actions, keyboard controls, reset and narrow layouts are required. These are legal abstract interleavings, not timing benchmarks or CPython bytecode simulations.

Native verification must force the lost-update schedule with a Barrier, not depend on chance or sleep. No actual unbounded deadlock. Verify ordered transfers, preserved totals, exception retrieval, condition prepublication/wait behavior and bounded queue completion under subprocess timeouts. Each displayed program is an executable fixture. Independent practice changes the data/problem, with explicit acceptance cases and a complete solution.

Sources reviewed: Python current threading documentation (condition wait/reacquire/notify), concurrent.futures (exception and cancellation semantics, nested-future deadlocks), free-threading how-to; OSTEP Locks, Condition Variables and Concurrency Bugs chapters. Version assumptions must distinguish the tested ordinary CPython 3.12 runtime from current optional free-threaded builds. Video alternative selection uses an official conference description/recording identity; no claim of full viewing.

## Evidence

Implementation and verification finished. [The module-completion record](../../PROGRAMMING-MODULE-COMPLETION.md) owns the aggregate evidence; [the Bash record](bash-completion-verification.md) records its native Bash/WSL and browser results.

Threads: all six displayed Python programs passed on 3.12.14, including a deliberately forced race and repaired counter, condition early/late publication, ordered transfers, future error collection, cooperative cancellation and bounded-queue practice. Eighteen changed queue cases across one/two/three workers match the sequential contract with no leaked workers or unfinished items. Exhaustive small-model exploration visits 73 distinct states and witnesses the unsafe race/deadlock while validating protected terminal invariants. Scripts: verify-thread-completion.mjs/.py.

Independent browser review passes all three thread labs at desktop and 390px, six rendered programs, all controls/resets, focus, disclosures, anchors and overflow checks. The wait-for graph was enlarged for legible mobile labels and controls have 44px touch targets. All three mobile labs and the desktop graph were visually inspected. Evidence is under scratch/thread-completion-review/browser; review-thread-completion.cjs repeats the checks. The condition snapshots explicitly show lock ownership at the predicate check, not a claim that notification itself transfers ownership. User acceptance remains pending.

The official PyCon 2017 Amber Brown talk description supports selecting the alternate CPU/OS/shared-state explanation; the direct recording was resolved as https://www.youtube.com/watch?v=31fXwpb0P9c. No full-video viewing was performed. Current Python docs control free-threaded runtime details, not the older recording.

Scoped discovery: [neural firmware buffer policy](topic-notes/embedded-processing-fpga-pipelines-and-real-time-neural-firmware.md) refines that existing topic's planned buffer/timing investigation. It distinguishes blocking, loss and latest-value policies under requirements; it is not a new topic or immediate rewrite request.
