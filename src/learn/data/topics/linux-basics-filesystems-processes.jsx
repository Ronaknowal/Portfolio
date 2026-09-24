import { Code, CodeBlock, H2, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import TerminalExample from "../../components/lesson-labs/TerminalExample";
import PermissionLab from "../../components/lesson-labs/PermissionLab";
import LinuxPathLab from "../../components/lesson-labs/LinuxPathLab";
import LinuxStreamsLab from "../../components/lesson-labs/LinuxStreamsLab";
import LinuxProcessLab, { LinuxEnvironmentDiagram, LinuxLinksDiagram } from "../../components/lesson-labs/LinuxProcessLab";
import { linuxExamples } from "../linux-command-examples.js";
import { investigationSetup, investigation } from "../linux-practice-examples";
import "../../components/lesson-labs/linux-lesson.css";

export default {
  title: "Linux Basics, Filesystems & Processes",
  readTime: "~40 min read + 75 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot linux-lesson">
    <LessonIntro exampleKind="Shell" prerequisites="No Linux experience required. Use a Linux terminal with Bash and ordinary user permissions, not an administrator shell."
      sections={[["1-separate-the-terminal-shell-and-operating-system", "Start here"], ["2-navigate-by-path-not-by-guessing", "Paths & files"], ["5-keep-standard-output-and-errors-distinct", "Output & errors"], ["6-understand-permissions-before-changing-them", "Permissions"], ["8-know-what-a-child-process-inherits", "Processes"], ["12-practise-a-small-investigation", "Put it together"]]}>
      An experiment produced a file, a warning and a process that seems stuck. Learn to locate the file, separate results from diagnostics, explain a permission failure and manage the correct process. Four focused explorers and a complete investigation connect commands to what Linux actually does.
    </LessonIntro>
    <Prose><strong>Your first pass:</strong> follow sections 1–6, then 8–9 and the investigation in 12. Return to links (7), resource diagnosis (10) and shared machines (11) as deeper branches. You should finish able to explain a command's effect before running it, trace a file path, and distinguish a stopped process from one that has ended.</Prose>
    <H2>1. Separate the terminal, shell and operating system</H2>
    <Prose>Linux is the operating-system foundation used on many servers and research machines. You can work with it by typing commands instead of opening folders and clicking menus. That makes a workflow easy to record and repeat—but first you need to know which program receives each instruction.</Prose>
    <Prose>The terminal displays text and sends keystrokes. A shell such as Bash interprets command lines and starts programs. The Linux kernel manages processes, memory, devices and filesystem access. These are different layers: changing your terminal theme does not change permissions, and installing Bash on Windows does not turn the Windows filesystem into a Linux one.</Prose>
    <Prose><strong>Choose your practice environment:</strong> use an ordinary-user Linux terminal. On Windows, an existing Ubuntu session in WSL is suitable; PowerShell interprets a different command language. You can complete the browser explorers before setting up Linux. Their diagrams simulate small examples and never change your files or processes.</Prose>
    <details className="linux-deeper"><summary>Tested environment and platform differences</summary><Prose>The original examples were recorded in Ubuntu under WSL as a non-root user, with Bash 5.2.21, GNU coreutils 9.4 and procps-ng 4.0.4. They use a Linux temporary filesystem so permission and link behaviour is real. Git Bash/MSYS, macOS and files under Windows-mounted paths can differ in commands, flags and filesystem semantics. These are tested versions, not a requirement to install those exact releases.</Prose></details>
    <Prose>Each complete runnable example starts in a fresh temporary directory. <Code>mktemp -d</Code> creates that directory and prints its path. The setup <Code>{'lab=$(mktemp -d)'}</Code> stores that printed path under the name <Code>lab</Code>; <Code>{'cd "$lab"'}</Code> uses the stored value to enter it. Quotes keep a path together as one argument. <Code>|| exit 1</Code> stops the script if the preceding setup command fails, rather than continuing in the wrong directory.</Prose>
    <Prose>All example data is invented; no sudo is needed. Type commands without an extra prompt symbol. You may inspect <Code>$lab</Code> afterwards; temporary storage is not a place to keep work you need permanently. The diagnostic command list in section 10 is a separate set of read-only recipes for your own machine.</Prose>
    <LessonTable caption="Read a command line" headers={["Part", "Example", "Meaning"]} rows={[
      ["Program / builtin", "ls, grep, cd", "The operation to run; cd must change the current shell's directory."],
      ["Option", "-l, --help", "Changes the program's behaviour."],
      ["Argument", "data/run.csv", "A value such as a path."],
      ["Quoting", '"run 1.csv"', "Keeps the space inside one argument."],
      ["Option terminator", "--", "For supporting commands, subsequent arguments are not options."],
    ]} />
    <Prose>Use command --help or man command to learn its actual options; press q to leave many manual pagers. Type or command -v can reveal whether a name resolves to a builtin, alias or executable. PATH is the ordered list used to search for programs; it is not the current working directory. Avoid adding writable or untrusted directories at the front of it.</Prose>

    <H2>2. Navigate by path, not by guessing</H2>
    <Prose>A directory holds names of files and other directories. A <strong>path</strong> is a route through those names. Your shell has a <strong>current working directory</strong>: the starting point for a relative route. <Code>pwd</Code> prints that starting point, <Code>ls</Code> lists entries, and <Code>cd</Code> changes it. Reading a file with <Code>head</Code> does not move your shell.</Prose>
    <Prose>A leading <Code>/</Code> starts at the root. Otherwise start where you are. A <Code>.</Code> segment stays in the same directory; <Code>..</Code> goes to its parent. Change the example path below to inspect its destination or first failure, then walk through how each segment was resolved.</Prose>
    <LinuxPathLab />
    <Prose>The explorer uses a small invented project. The runnable version below creates its project in a fresh temporary directory, shown as <Code>LAB</Code> in the output. The same relative-path rules apply; the absolute prefix is different.</Prose>
    <TerminalExample example={linuxExamples.paths}><Prose>The dot-prefixed note appears because ls uses -A. After moving into data/raw, pwd identifies the location; the output substitutes LAB for the random temporary prefix. The space-containing filename remains one argument. Two parent steps return to the project root before find searches data.</Prose></TerminalExample>
    <Prose>An absolute path starts at /, the filesystem root. A relative path starts from the current working directory. Dot means the current directory and dot-dot means its parent. A user's home is commonly /home/name; root's home is usually /root, which is not the same thing as /. Use pwd before a command whose destination matters.</Prose>
    <LessonTable caption="Common locations and their role" headers={["Location", "Typical purpose", "Practical consequence"]} rows={[
      ["/home", "Ordinary user homes", "Keep personal work within your permitted area."],
      ["/etc", "System configuration", "Do not edit it casually to fix a project-local issue."],
      ["/tmp", "Temporary files", "Retention is not guaranteed; use for disposable work."],
      ["/var", "Changing system data, including many logs", "Access and retention depend on the service."],
      ["/usr", "Installed programs and shared resources", "Package management normally owns these files."],
      ["/proc", "Virtual process/kernel information", "These are not ordinary persistent disk files."],
      ["/mnt, /media", "Common mount locations", "Different filesystems may have different semantics and capacity."],
    ]} />
    <Prose>Linux paths are typically case-sensitive: Data and data need not be the same directory. A leading dot is a naming convention that normal ls hides, not encryption. Quote expanded paths as <Code>{'"$HOME/project notes"'}</Code>. Tilde expansion has special shell rules; a quoted literal "~" is not a reliable substitute for your home path. With symlinks, logical and physical paths can differ; pwd -P shows the physical directory resolution.</Prose>

    <H2>3. Copy, move and remove deliberately</H2>
    <TerminalExample example={linuxExamples.files}><Prose>Cp creates a separate copy, so editing it leaves source unchanged. Mv renames the copy within this filesystem. Rm removes only the named disposable copy. The checks confirm that the original still exists. These steps are safe because the target was created in this example's new directory.</Prose></TerminalExample>
    <Prose>Existing destinations can be overwritten by ordinary copy/move operations. Inspect both source and destination first; an interactive confirmation option may help during manual work, but is not a substitute for correct paths. Moving across filesystems can require copying and then removing the source, so it is not necessarily an instantaneous atomic rename.</Prose>
    <Prose>Rm is not a guaranteed desktop recycle-bin operation. Recursive deletion expands the scope enormously; avoid it until the resolved directory and its contents have been checked and you explicitly intend the removal. Do not build deletion lists by parsing ls. Spaces, newlines, wildcard expansion, symlinks and option-like names make “clever” cleanup commands dangerous.</Prose>
    <Prose>Touch creates an empty file if absent or updates timestamps if present; it does not empty existing content. Mkdir -p creates missing parents and tolerates an already-existing directory. File extensions are conventions: a .csv suffix does not prove a file is valid CSV or safe to parse.</Prose>

    <H2>4. Find files and read evidence</H2>
    <Prose>Ask two different questions: <strong>where is the file?</strong> Use <Code>find</Code>. <strong>Which lines inside it matter?</strong> Use <Code>grep</Code>. In the next example, first notice the two WARN lines. The <Code>|</Code> symbol then passes those matching lines to a counting program; section 5 makes that connection visible.</Prose>
    <TerminalExample example={linuxExamples.search}><Prose>Grep -n prints original line numbers, so WARN appears at lines 2 and 4. -F treats WARN as literal text rather than a regular expression. The pipe sends those two lines into wc -l. A quiet search for SUCCESS produces exit status 1 because there is no match; that is different from a read or syntax error.</Prose></TerminalExample>
    <Prose>Find searches filesystem entries; grep searches content. Quote '*.log' so find receives the pattern instead of the shell expanding it first in the current directory. Add -type f when you want regular files. For large source trees, rg is often convenient when installed; understand ignore and hidden-file defaults rather than assuming every search tool examines the same files.</Prose>
    <LessonTable caption="Inspect without loading everything at once" headers={["Tool", "Use", "Caution"]} rows={[
      ["ls -lah / stat", "Metadata, sizes and permissions", "Human-oriented listings are not a safe filename parser."],
      ["head / tail", "Beginning or end of a text file", "A preview does not validate the entire dataset."],
      ["less", "Page through large text", "Use / to search and q to leave; binary data may be unsuitable."],
      ["tail -f", "Follow appended log lines", "Ctrl-C stops the follower, not the service writing the log."],
      ["file", "Inspect likely content type", "Identification is a heuristic, not a security guarantee."],
      ["wc -l", "Count newline characters", "An unterminated final line changes what this count means."],
    ]} />
    <Prose>Exit status is a small integer returned by a command. Zero usually means success; nonzero values have command-specific meanings. Inspect $? immediately if using it, because the next command replaces it. The if statement in the example handles grep's result without mistaking “no match” for a successful discovery. Larger error-handling patterns belong in the separate Bash lesson.</Prose>

    <H2>5. Keep standard output and errors distinct</H2>
    <Prose>A program can send useful results down one channel and diagnostic messages down another. Both usually appear in your terminal, which can make them look like one thing. <strong>Standard output</strong> (stdout) is the normal-results channel; <strong>standard error</strong> (stderr) is the diagnostics channel. A message on stderr is not itself proof of failure: the program also returns an exit status.</Prose>
    <LinuxStreamsLab />
    <TerminalExample example={linuxExamples.streams}><Prose>The first subprocess writes a metric to standard output and a warning to standard error. Separate redirects capture each stream. Append adds a second result line; the count becomes two. The final command combines both streams in one file in the order this small program emits them.</Prose></TerminalExample>
    <LessonTable caption="Three standard file descriptors" headers={["Descriptor", "Default role", "Shell operation"]} rows={[
      ["0: stdin", "Input", "< file supplies file contents as input."],
      ["1: stdout", "Normal results", "> file creates/truncates; >> file appends."],
      ["2: stderr", "Diagnostics", "2> file redirects errors separately."],
      ["pipe |", "Connect stdout to another command's stdin", "Stderr does not automatically travel through this pipe."],
    ]} />
    <Prose>A <Code>{">"}</Code> redirect can empty an existing output file before the program even starts. Use <Code>{">>"}</Code> when you intend to append; never use the same file as both input and truncating output. Keep separate error logs when diagnostics must not corrupt machine-readable metrics.</Prose>
    <details className="linux-deeper"><summary>Deeper: redirection order and pipeline status</summary><Prose>Redirections are processed left to right. <Code>{"> combined.txt 2>&1"}</Code> first redirects stdout, then makes stderr refer to that destination. Reversing them can leave stderr pointing at the terminal. The duplication copies the current destination; it does not create a promise to follow later changes.</Prose><Prose>A pipeline normally reports the last command's status in Bash. An earlier stage can fail while the last stage succeeds; pipefail changes that behaviour. Avoid trusting a result solely because the final wc or writer succeeded. The Bash lesson develops deliberate error handling further.</Prose></details>

    <H2>6. Understand permissions before changing them</H2>
    <Prose>Knowing a filename does not automatically give a process access to its contents. Linux first has to follow the path through its directories, then check the requested operation on the file. A directory's <strong>search</strong> permission lets you reach an entry by name; the file's <strong>read</strong> permission lets you read its bytes. These are separate checks.</Prose>
    <PermissionLab />
    <Prose>Ordinary mode bits have three classes: owner, matching group, and other. Within a class, read is 4, write 2, execute/search 1; add them to form the octal digit. Mode 640 therefore means owner read/write, group read, other none. The long form -rw-r----- begins with a file-type character, then three groups of three permission positions.</Prose>
    <TerminalExample example={linuxExamples.permissions}><Prose>The regular file has mode 640. A directory with mode 600 lets its owner read entry names and has a write bit, but lacks search permission, so opening value.txt through it fails. Mode 700 restores traversal. In the subshell, umask 077 removes group/other permissions from ordinary new-file and directory creation, producing 600 and 700.</Prose></TerminalExample>
    <Prose>Directory execute means search/traverse, not “run this directory.” Reading file contents also requires traversal of the relevant parent directories. Creating or removing entries normally needs write and search permissions on the containing directory. Thus a read-only file can sometimes be deleted by someone permitted to modify its directory; the file's write bit is not a deletion lock. Sticky directories such as shared temporary directories add restrictions.</Prose>
    <Prose>Umask removes permission bits from the mode a program requests; it is not decimal subtraction. Ordinary file creation typically requests no execute bits, so a permissive umask does not make every new file executable. Default ACLs and filesystem policies can affect real results. Chmod changes mode bits; chown changes ownership and often requires privileges. Neither is a universal repair for a path or mount problem.</Prose>
    <Prose>Do not respond to every permission error with sudo or mode 777. Identify who you are with id, inspect ownership and every relevant parent path, and request only the access required. Root/capabilities, ACLs, security modules and mount options extend this basic model. A script also needs an appropriate interpreter or executable format; the x bit alone does not make arbitrary bytes runnable.</Prose>
    <Checkpoint prompt="A file is readable, but its parent directory denies search permission. Why can cat still fail?">
      <Prose>The kernel must resolve the pathname through the parent first. File read permission alone does not grant that traversal. Check directory search bits and other applicable restrictions rather than broadening the file mode blindly.</Prose>
    </Checkpoint>

    <H2>7. Distinguish file content from directory entries</H2>
    <details className="linux-deeper"><summary>Deeper branch: hard links, symbolic links and disk space</summary>
    <LinuxLinksDiagram />
    <TerminalExample example={linuxExamples.links}><Prose>The hard link is another name for the same underlying file, confirmed by -ef. The symbolic link stores the pathname record.txt. Renaming that path leaves the hard link usable but the symbolic link dangling. A relative symlink target is interpreted from the link's containing directory, not the caller's working directory.</Prose></TerminalExample>
    <Prose>An inode identifies filesystem metadata and data references within one filesystem; directory entries associate names with inodes. Ordinary hard links cannot cross filesystems, and directory hard linking is restricted. A symbolic link can name a target on another filesystem or a target that does not currently exist. Tools differ in whether they follow links; inspect their options before copying, searching or changing permissions recursively.</Prose>
    <Prose>Removing one hard-link name does not delete the contents while another link still exists. Open file descriptors can also keep unlinked data alive until closed. This helps explain why deleting a large open log might not immediately free space—investigate the owning service rather than repeatedly deleting more files.</Prose>
    </details>

    <H2>8. Know what a child process inherits</H2>
    <Prose>A program file is a set of instructions stored on disk. A <strong>process</strong> is a running instance with its own identity and state. When your shell starts another program, we call the shell its parent and the new process its child. Two launches can run the same program while keeping separate state.</Prose>
    <Prose>A shell variable is a named value. <Code>export</Code> makes a value part of the environment passed to subsequently launched programs. It does not turn the value into a global variable shared by every process.</Prose>
    <LinuxEnvironmentDiagram />
    <TerminalExample example={linuxExamples.environment}><Prose>A non-exported shell variable is absent in the child. Export makes it available to later children. A child changing its own value does not update the parent. A one-command assignment supplies once only to that process; the parent's local value remains local.</Prose></TerminalExample>
    <Prose>A process starts with a working directory and environment supplied by its parent. A Python program, notebook kernel and shell may therefore resolve the same relative filename or program name differently. Inspect the actual environment where the failing command runs. An environment variable is a string convention interpreted by a program, not a typed setting enforced by Linux.</Prose>
    <Prose>Do not dump the entire environment into a public bug report: it can contain secrets. Prefer targeted checks that avoid printing credential values. Changing PATH or activating an environment changes command selection in that shell; it does not automatically replace an already-running notebook kernel.</Prose>

    <H2>9. Identify and manage your own process</H2>
    <Prose>Linux assigns a process ID, or <strong>PID</strong>, to identify a process. Putting <Code>&amp;</Code> after a command lets the shell accept another command while the child is still alive. A <strong>signal</strong> is a request or notification sent to a process. Pausing, continuing and terminating are different operations.</Prose>
    <LinuxProcessLab />
    <Checkpoint prompt="Pause the child in the explorer. Has it finished? Continue it, then end it: what extra information does wait retrieve?">
      <Prose>A paused child still exists. Continuing resumes the same process. After this sleep process terminates, wait retrieves its termination status; it does not restart it or undo anything. A program that writes files can leave those changes behind even after termination.</Prose>
    </Checkpoint>
    <TerminalExample example={linuxExamples.process}><Prose>Ampersand starts sleep in the background; $! captures that child's PID. Ps confirms the command name. TERM requests termination, and wait retrieves the child's termination status. Bash may already have reaped the operating-system process and kept that status for you. Bash reports 143 for termination by signal 15 here: 128 + 15. That encoding is shell-specific context, not a universal application error number.</Prose></TerminalExample>
    <Prose>A PID identifies a process at a point in time and can later be reused. Check the owner, command, start time and context before acting on a remembered PID. On a shared server, do not terminate by broad name matching just because a process contains “python.” A training run, notebook kernel and service can all share that executable name.</Prose>
    <LessonTable caption="Common process controls" headers={["Action", "Meaning", "Boundary"]} rows={[
      ["Ctrl-C", "Usually sends SIGINT to the terminal's foreground process group", "The program can handle it; it is not an undo operation."],
      ["Ctrl-Z", "Usually stops the foreground job", "It remains present; stopped does not mean finished."],
      ["jobs / fg / bg", "Inspect or resume this shell's jobs", "Job numbers are not global PIDs."],
      ["kill -TERM PID", "Request termination", "Allows handling/cleanup if the program implements it."],
      ["SIGKILL", "Forced termination without a catchable handler", "Last resort after identification; no application cleanup opportunity."],
      ["wait PID", "Wait for a child if needed and retrieve its status", "The shell must own the child; not a general wait for any PID."],
    ]} />
    <Prose>Stopping a process does not roll back files or database writes. Prefer a program's supported shutdown or scheduler cancellation mechanism. A background ampersand is not a guarantee of surviving logout; use the approved job scheduler, service manager or persistent session workflow for long-running work. Zombies have exited but await reaping by a parent; sending them more kill signals does not perform that reaping.</Prose>

    <H2>10. Diagnose resource problems methodically</H2>
    <details className="linux-deeper"><summary>Deeper branch: investigate a slow job or a full disk</summary>
    <Prose>“The job is slow” can mean CPU saturation, memory pressure, storage latency, waiting for a network service or a blocked process. Begin with read-only observations. The following commands are inspection recipes, not fixed-output demonstrations: device names, PIDs and measurements depend on your machine.</Prose>
    <CodeBlock language="bash">{`df -h .
df -i .
du -sh ./outputs
free -h
ps -u "$USER" -o pid,ppid,stat,%cpu,%mem,etime,args
ss -ltn`}</CodeBlock>
    <LessonTable caption="Interpret resource evidence" headers={["Tool", "Question", "Common misreading"]} rows={[
      ["df -h", "How full is the filesystem containing this path?", "Not the size of only your project."],
      ["df -i", "Are inode slots exhausted?", "Free bytes alone may not allow more tiny files."],
      ["du -sh", "How much storage is attributed to this directory?", "Sparse files, hard links, permissions and open unlinked files affect comparisons."],
      ["free -h", "How is system memory being used?", "Low free memory is not identical to low available memory; caches can be reclaimed."],
      ["ps / top", "Which processes exist and how are resources used?", "A process's virtual address size is not all resident RAM; CPU measures have sampling context."],
      ["ss -ltn", "Which TCP sockets are listening?", "A listening port does not prove application health or external reachability."],
    ]} />
    <Prose>Du may itself traverse a large tree, so scope it to the area you need. If a filesystem is full, inspect ownership and retention policy before deleting anything. If a process is waiting, read its relevant logs before starting duplicate jobs. Service status and logs may be available through systemctl or journalctl on systemd systems, but those tools are not universal in containers or every WSL setup.</Prose>
    </details>

    <H2>11. Work safely on remote and shared machines</H2>
    <details className="linux-deeper"><summary>Deeper branch: remote sessions, packages and shared infrastructure</summary>
    <Prose>An SSH session starts a shell on another machine. Confirm host and working directory before acting; the same-looking prompt can hide a different filesystem. Verify a new or changed host-key fingerprint through a trusted channel rather than disabling host checking. Protect private keys and use the organisation's authentication method.</Prose>
    <Prose>Use the machine's package manager and administrator process for system software; project environments are better for project-specific Python dependencies. Do not pipe an unknown downloaded script directly into a privileged shell. A container shares a host kernel and has its own namespaces/mounts; it is not simply a folder or a full virtual machine, and a mounted host path can expose real host data.</Prose>
    <Prose>When seeking help, share the exact command, relevant path context, expected behaviour, exit status and a minimal redacted error. Avoid uploading full private logs or credentials. Check permissions and resource observations first; a reproducible, focused report is more useful than a list of destructive commands already attempted.</Prose>
    </details>

    <H2>12. Practise a small investigation</H2>
    <section className="linux-mission" aria-label="Linux project investigation">
      <h3>Find the measurements and explain the warning report</h3>
      <p>You are working inside a project's <Code>reports</Code> directory. Your teammate needs the first reading from a CSV and a report of the warning lines. Create this disposable project, then solve the tasks without opening the solution.</p>
      <CodeBlock language="bash">{investigationSetup}</CodeBlock>
      <ol>
        <li>Find the CSV under the sibling <Code>data</Code> directory. Read its header and first measurement, keeping the space-containing filename as one argument.</li>
        <li>Save WARN lines from <Code>../logs/run.log</Code>, with original line numbers, to <Code>warnings.txt</Code> in the current directory. Keep the original log unchanged.</li>
        <li>Count the saved warning lines. Search for SUCCESS and interpret the status when it is absent.</li>
        <li>Explain which commands read data, which create output, and whether any of them change your current directory.</li>
      </ol>
      <details><summary>Hint: choose a tool for each question</summary><p>From reports, <Code>..</Code> reaches the project directory. Use find for filenames, head for a preview, grep -n -F for matching lines, and a redirect for the report. A redirect writes a file; it does not change directories. Check a command's status before running another command.</p></details>
      <details><summary>Show the complete solution and expected output</summary>
        <p>This independent solution includes a fresh copy of the setup so you can run the whole block.</p>
        <TerminalExample example={investigation}><Prose>The two warning lines retain positions 2 and 4 from the original log. Only the report is written; find, head and grep read their inputs. The final search status 1 means no matching SUCCESS line, not that the file was unreadable. The shell stays in reports after setup.</Prose></TerminalExample>
      </details>
      <details><summary>Transfer task: a third warning and a blocked path</summary><p>In the disposable project, append <Code>WARN reviewed</Code> followed by a newline to the log. Predict the new line numbers and count, then rerun your report commands. Separately use the permission explorer to explain why a readable CSV can still be unreachable. Which path check would you investigate before changing its file permissions?</p><details><summary>Check your reasoning</summary><p>The report contains lines 2, 4 and 5 and counts 3. Rebuilding it with a truncating redirect avoids duplicating the previous report; appending an entire regenerated report would count old matches again. A missing directory search permission can block resolution before file read permission is checked.</p></details></details>
    </section>
    <Checkpoint prompt="The file is named run 1.csv. Why can head run 1.csv fail even when that file exists?">
      <Prose>The shell passes run and 1.csv as separate arguments. Quote the complete path: head -n 2 "run 1.csv". The example demonstrates the correct argument boundary.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Choose directory mode 750 in the explorer. What may a matching-group user do?">
      <Prose>Group digit 5 means read + search: list entry names and traverse known entries, but not create/delete entries because write is absent. Opening a child file still depends on that file's permissions and the rest of the path.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Your captured metrics contain warning text. How would you separate them without hiding failures?">
      <Prose>Have the program write metrics to stdout and warnings to stderr, then redirect them to separate files as in the stream example. Check the program's exit status and inspect the error log. Redirection cannot fix a program that puts every message on the same stream without changing its logging policy.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Add another WARN line to the search fixture and confirm the result without editing the original log in a real project.">
      <Prose>Use the fresh practice directory, append WARN reviewed to its fixture and rerun grep. It appears on line 5 and the matching-line count becomes 3. Find should still return one .log file. This distinguishes number of matching lines from number of files.</Prose>
    </Checkpoint>
    <Prose><strong>Before moving on:</strong> explain an absolute versus relative path, predict where stdout and stderr go, identify a blocked permission check, and distinguish a stopped child from an exited one. Revisit the relevant explorer if any explanation still depends on guessing.</Prose>
    <Prose>Next in this module, <a href="/learn/topic/bash-scripting-command-line-automation">Bash Scripting &amp; Command-Line Automation</a> combines these operations into repeatable programs, with explicit arguments and failure handling. <a href="/learn/topic/os-processes-virtual-memory-isolation">OS processes and virtual memory</a> then explains scheduling and memory isolation beneath those programs. Connect back to <a href="/learn/topic/git-github-collaborative-version-control">Git</a>: its working tree is a directory on this filesystem, and shell redirects can change files that Git later reports as modified. Networking administration, service operations, security policy and cluster scheduling need deeper modules.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://missing.csail.mit.edu/2026/course-shell/">MIT Missing Semester: Course Overview and the Shell — current notes and lecture</a>. Another guided introduction to navigation, arguments, streams and composing tools. Work through its exercises in a disposable directory after the path and stream labs.</li>
      <li><a href="https://www.youtube.com/watch?v=Z56Jmr9Z34Q">Missing Semester's 2020 shell lecture on YouTube</a> with <a href="https://missing.csail.mit.edu/2020/course-shell/">matching written notes</a>. A useful older recording for seeing commands evolve at a terminal. Core shell mechanisms remain relevant; platform-specific setup and interfaces can differ from current Linux/WSL.</li>
    </LearningResources>}>
      <li><a href="https://man7.org/linux/man-pages/man7/path_resolution.7.html">Linux pathname resolution and traversal</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/ls.1.html">GNU ls options</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/find.1.html">Find patterns and filesystem traversal</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/grep.1.html">Grep matches, fixed strings and exit status</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/bash.1.html">Bash quoting, streams, environment and jobs</a></li>
      <li><a href="https://www.gnu.org/software/bash/manual/html_node/Redirections.html">GNU Bash: redirection order and file descriptors</a></li>
      <li><a href="https://www.gnu.org/software/bash/manual/html_node/Environment.html">GNU Bash: exported variables and child environments</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/chmod.1.html">Mode bits and symbolic permissions</a></li>
      <li><a href="https://man7.org/linux/man-pages/man7/signal.7.html">Linux signals and process handling</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/ps.1.html">Process inspection</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/df.1.html">Filesystem capacity</a></li>
      <li><a href="https://man7.org/linux/man-pages/man1/du.1.html">Directory storage accounting</a></li>
    </Sources>
  </div>,
};
