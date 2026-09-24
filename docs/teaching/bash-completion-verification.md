# Bash implementation and verification

10 September 2026. Completes the authorized Bash rewrite within Programming & Scientific Computing. The parent [Bash/concurrency design](bash-and-concurrency-design.md) owns the initial hurdle map; this is its scoped implementation/evidence supplement. The root handoff owns current rollout status.

## Coverage and title review

Ran `node scripts/build-curriculum-inventory.mjs --topic bash-scripting-command-line-automation`; no incoming destination note and no unresolved inbox entry. Existing title retained. Earlier lesson offered one generic strict-mode snippet depending on missing train.py and incorrectly declared the module complete. Useful variables/defaults, quoting, orchestration, stdout and repeat-run intent are retained and taught through complete files instead.

New core: Bash versus terminal/Linux/sh/PowerShell, interpreter selection, exact argument boundaries, arrays and empty values, option boundaries, data versus reparsed code, local/export/source/subshell/current-directory scope, command substitution, status/stream separation, PIPESTATUS and grep status policy, conditional errexit counterexample, nullglob and pipeline loops, staged publication/cleanup, full worker/wrapper and independent collector. Optional branches explain early consumers/SIGPIPE and transferring one-file publication to generated indexes/manifests. OS then Threads is the next module sequence; no publication-based skip.

The parent supplies three bounded investigations and fixture sources. This lesson connects their predictions, controls and consequences to actual Bash/Python execution. Exact native output is displayed for all five standalone Bash programs; the multi-file report and collection also show expected JSON and separate status/diagnostic meaning. Every required input file and full solution is available. The independent collector distinguishes missing input from valid empty data and requires finite means/nonnegative counts while preserving order, filename boundaries and previous output.

Publication is explicitly local, same-filesystem successful rename of one file. The fixture is not a concurrent transaction system or a crash-durability demonstration. The CSV worker is deliberately bounded, uses the default standard-library dialect, skips blank records and stores values for fsum. General schema/data-ingestion depth belongs to Scientific File Formats. No new unrelated destination expansion was warranted.

During verification design, the prepared worker was corrected by the parent to catch csv.Error and label enumeration diagnostics as records rather than physical lines. CSV records can span physical lines. A second source-level correction escaped Bash parameter expansion inside JavaScript template literals. These are implementation repairs, not title or curriculum changes.

## Source ledger

- [ShellCheck SC2086](https://www.shellcheck.net/wiki/SC2086): substantive quoted/unquoted argument examples, array alternatives and glob/splitting order reviewed. This is a targeted alternate explanation and static-analysis rationale, not a guarantee a script is correct.
- [Python csv](https://docs.python.org/3/library/csv.html): DictReader field handling, extra fields, newline convention, blank records and dialect/record-versus-line boundaries reviewed.
- [Python json](https://docs.python.org/3/library/json.html): parent reviewed parsing/encoding and allow_nan. The collector validates its application schema explicitly; json.load by itself is insufficient for finite/count contracts.
- [Linux rename](https://man7.org/linux/man-pages/man2/rename.2.html): local destination replacement and existing open-file behavior reviewed; native observations test the same specified fixture behavior without claiming power-failure durability.
- [GNU Bash manual](https://www.gnu.org/software/bash/manual/bash.html): current source URL identified, but full/individual sections timed out in the web reader in both parent and this task. Native Bash help set/trap/read/source/local and parameter/status probes therefore supply installed-version evidence. Do not claim the inaccessible full manual was read.
- [MIT Missing Semester Shell Tools/Scripting notes](https://missing.csail.mit.edu/2020/shell-tools/) and [official linked video](https://www.youtube.com/watch?v=kgII-YWo3Zw): parent inspected written material and verified video identity. The recording was not watched in full. Annotated as a 2020 Unix/Bash alternative; current lesson explicitly distinguishes grep no-match from actual errors instead of generalizing all nonzero statuses.

## Verification status

Implementation complete; native and browser/visual review passed. User acceptance remains pending. Parent owns final application build, module sequence conservation and handoff updates.

Prepared evidence commands:

- `node scripts/export-bash-completion.mjs`: read exact displayed outputs and export five Bash scripts, report/collector fixtures and sixteen browser-model configurations.
- Run `scripts/verify-bash-completion.py` under ordinary-user native Linux with the exported fixtures path. It creates one owned Linux temporary root, checks syntax and exact outputs, real argument/status semantics, schema/empty/missing cases, ordered collector behavior, staged visibility, preserved prior output, discarded partial worker output and TERM cleanup of only its own fixture process group. Cleanup verifies the root remains the specific created temp directory.
- `node scripts/review-bash-completion.cjs` passed on Edge/Chromium at 1440 × 1000 and 390 × 1000: all sixteen model configurations, their available forward/back/reset interactions, native select/checkbox controls, keyboard reset/toggle/hint activation, valid section anchors, five standalone programs, five report files, collector solution and exact next topic OS. No page errors, horizontal overflow or clipped lab states/controls. Three mobile and two desktop lab screenshots were visually inspected; controls and state labels remain readable.

Actual native run passed in Ubuntu WSL, ordinary `nobody` UID 65534: Bash 5.2.21(1)-release, Python 3.12.3, Linux 6.6.87.2 WSL2. It checked five exact outputs, sixteen model configurations, defaults/expansion/data nonexecution, source/subshell semantics, five report failure cases, nine invalid collection cases, empty and zero data, argument order, repeated content, rename visibility to an already-open reader, and old output retained after both a partial worker failure and an owned process-group TERM interruption. All temporary staging was removed; the verifier removed only its own Linux temporary root.

Evidence artifacts: `scratch/bash-completion-review/native-results.json`, `browser-results.json`, exported `fixtures.json`, and `arguments/statuses/publication-390.png` / `-1440.png`. The native JSON records the actual command output. Browser screenshots and their inspection establish author review, not an observed novice study. No SIGKILL durability, network-filesystem, concurrent-publisher, arbitrary-filename-stream or large-data-memory guarantee was tested.
