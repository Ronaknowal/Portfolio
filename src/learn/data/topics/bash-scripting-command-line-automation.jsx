import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { BashArgumentsLab, BashStatusLab, BashPublicationLab } from "../../components/lesson-labs/BashWorkflowLabs.jsx";


import { bashExamples, reportFiles, summaryPractice } from "../bash-workflow-examples.js";

export const bashObservedOutputs={
  arguments:`count=1
<run alpha.csv>
count=2
<run>
<alpha.csv>
count=1
<*.csv>
count=2
<a.csv>
<run alpha.csv>
count=1
<>
count=0`,
  scope:`function=function; arguments=2
item=<run alpha.csv>
item=<>
child=child
parent=parent
captured=<two lines>`,
  statuses:`default=0
pipefail=4
individual=4,0
valid search: no match`,
  errexit:`continued inside conditional function
function reported success from its last command
an expected false condition is handled here`,
  files:`matched=2
path=<./beta.csv>
path=<./run alpha.csv>
missing=0
parent count after pipeline=0
parent count after redirected loop=2`,
};
function Example({name,children}){
  const example=bashExamples[name];
  return <div className="bash-complete-example" data-bash-example={name}><H3>{example.title}</H3><p className="lesson-note">Save as <strong>{name}.sh</strong>. Run <strong>bash {name}.sh</strong> from your Bash terminal.</p><CodeBlock language="bash">{example.code}</CodeBlock><p className="lesson-note">Expected stdout</p><CodeBlock language="output">{bashObservedOutputs[name]}</CodeBlock>{children}</div>;
}
function File({name}){return <div data-report-file={name}><p className="lesson-note">Save as <strong>{name}</strong> in the same new exercise folder.</p><CodeBlock language={name.endsWith('.py')?'python':name.endsWith('.sh')?'bash':'csv'}>{reportFiles[name]}</CodeBlock></div>;}

export default {
 title:'Bash Scripting & Command-Line Automation',readTime:'~38 min read + 80 min practice',hasIntegratedGuide:true,
 content:()=> <div className="lesson-pilot">
  <LessonIntro exampleKind="Bash and Python" prerequisites="Linux Basics: paths, streams, permissions and process status. Python functions, CSV/JSON and basic file handling are refreshed in the complete report example; the Scientific File Formats lesson supplies their deeper contracts."
   sections={[["1-turn-a-command-history-into-a-repeatable-job","Start a script"],["2-preserve-argument-boundaries","Arguments"],["3-pass-values-without-building-command-strings","Values and scope"],["4-treat-output-and-status-as-separate-results","Failures"],["5-visit-files-and-keep-state-in-the-right-process","Files and loops"],["6-publish-only-a-complete-report","Publication"],["7-build-the-complete-report-job","Run the project"],["8-independent-investigation-collect-several-reports","Independent practice"]]}>
   Build an automation you can inspect: preserve each filename as one argument, decide what a failure means, and replace a public report only after a complete result exists. Trace the shell's hidden expansion and process boundaries before relying on a script.
  </LessonIntro>

  <H2>1. Turn a command history into a repeatable job</H2>
  <Prose>You have a CSV of readings and a Python program that summarizes it. Each run needs the same coordination: select input, choose an output path, run the worker, examine its result and tell the caller what happened. A script records those decisions so you can repeat them and diagnose a failed run. Bash is useful for connecting existing programs; the numerical formula and CSV parser can stay in Python.</Prose>
  <Prose><strong>Bash</strong> is a particular shell and scripting language. A terminal is the window hosting it; Linux is the operating system beneath it. PowerShell has different syntax. The name <Code>sh</Code> selects a shell interface that need not support Bash arrays or <Code>[[ ... ]]</Code>. Run these examples in Bash on Linux, including an Ubuntu terminal in WSL on Windows. The commands here use Bash, GNU tools and Python 3's standard library; no package installation or remote data is required.</Prose>
  <Prose>The complete scripts were verified in Ubuntu on WSL using Bash 5.2.21 and Python 3.12.3. Browser labs model the specific examples; they do not execute arbitrary shell commands or change your files.</Prose>
  <CodeBlock language="bash">{`bash --version
python3 --version`}</CodeBlock>
  <Prose>Save scripts with ordinary UTF-8 text and LF line endings. <Code>bash arguments.sh</Code> explicitly starts Bash and reads that file; executable permission is unnecessary for this form. If you instead make the file executable and run <Code>./arguments.sh</Code>, Linux uses its first line, the <strong>shebang</strong>, to select the interpreter. <Code>#!/usr/bin/env bash</Code> finds Bash through PATH. Calling <Code>sh arguments.sh</Code> explicitly selects sh, irrespective of the shebang.</Prose>
  <Prose>A simple command has a program or function name followed by separate arguments. The shell interprets quoting, expands values and builds those arguments before launching the command. The launched program receives arguments, not your original source-code quotes. Understanding that translation prevents many failures that otherwise look like broken filenames.</Prose>

  <H2>2. Preserve argument boundaries</H2>
  <Prose>Suppose <Code>value='run alpha.csv'</Code>. The stored value contains a space; its quote marks were shell syntax and are not stored. In an ordinary command, unquoted <Code>$value</Code> can be split on the shell's field separators, then treated as filename patterns. With the default separators, one filename becomes two arguments. Double-quoting the expansion keeps it as one argument, even when it is an empty string.</Prose>
  <BashArgumentsLab/>
  <Prose>The next script creates its own temporary directory and two invented files. <Code>mktemp -d</Code> chooses and creates that directory; the <Code>|| exit 1</Code> branch stops if creation fails. Only after success does <Code>trap</Code> register removal of this owned directory at script exit. We will study traps in the publication section. <Code>LC_ALL=C</Code> makes the displayed filename order deterministic. <Code>show_args</Code> prints the number of received arguments, excluding the command name, and surrounds each with angle brackets so empty and split values are visible.</Prose>
  <Prose><Code>{'show_args() { ...; }'}</Code> defines a Bash function. Its <Code>for arg in "$@"; do ...; done</Code> loop visits each received argument once, assigning it to arg while running the body. <Code>$#</Code> is the argument count. Unlike Python, Bash closes these compound constructs with explicit braces or keywords rather than relying on indentation.</Prose>
  <Example name="arguments"><Prose>The six argument counts are 1, 2, 1, 2, 1 and 0. Quoted <Code>"*.csv"</Code> reaches the function literally; the unquoted pattern expands to matching filenames. The matched filename containing a space remains one glob result: the shell does not repeat field splitting after glob expansion. An unquoted empty expansion disappears; a quoted empty expansion creates one empty argument.</Prose></Example>
  <LessonTable caption="Quoting controls what the shell interprets" headers={['Form','What it does','Use here']} rows={[["'literal text'",'Preserves literal characters; no parameter or command substitution','A known fixture or code passed intentionally to bash -c'],['"$value"','Expands the variable but prevents splitting and glob expansion of its value','One filename or other data argument'],['"${args[@]}"','Expands each array element as its own argument, preserving empties','A prepared argument list'],['--','Many commands recognize this as end of options','Protect a following filename beginning with a dash; check the command supports it']]} />
  <Prose><strong>Data is not automatically reparsed as shell code.</strong> If a variable contains literal characters such as <Code>$(...)</Code> or a semicolon, expanding it as <Code>"$value"</Code> passes those characters as data; it does not execute them. Unquoted expansion creates splitting and globbing problems, but still does not generally reparse the result as new shell operators. An explicit second interpreter such as <Code>eval</Code> or interpolated <Code>bash -c</Code> changes that boundary. Keep executable code fixed and pass variable data as arguments.</Prose>
  <Checkpoint prompt="A path starts with a dash. Does quoting alone guarantee that a command treats it as a filename?"><Prose>No. Quoting preserves one argument; the program may still interpret that argument as an option. For tools that support it, place -- before path operands, or use an explicit path such as ./-report.csv. Argument boundaries and the receiving program's option syntax are different contracts.</Prose></Checkpoint>

  <H2>3. Pass values without building command strings</H2>
  <Prose>Assignment has no spaces around the equals sign: <Code>label=parent</Code>. Reading a value uses <Code>$label</Code>. Script arguments are <Code>$1</Code>, <Code>$2</Code> and so on; <Code>$#</Code> counts them, <Code>$0</Code> identifies the invocation, and <Code>"$@"</Code> preserves all positional arguments as separate values. Test the count before accessing required parameters when <Code>set -u</Code> is enabled.</Prose>
  <LessonTable caption="Give each parameter an explicit policy" headers={['Expression or operation','Meaning','Decision to make']} rows={[["seed=${1:-7}",'Use 7 when parameter 1 is unset or empty','Is empty equivalent to absent?'],["seed=${1-7}",'Use 7 only when parameter 1 is unset','Is empty a meaningful value?'],['shift','Remove the first positional argument; the rest move left','Validate that an argument exists first'],['return 4 / exit 4','Return status 4 from a function / terminate the script with 4','A Bash function return is a status, not arbitrary data'],['printf \'%s\\n\' "$value"','Print data with a fixed format string','Keep the format separate from data containing percent signs']]} />
  <Prose>An array stores separate arguments directly: <Code>args=('run alpha.csv' '')</Code> contains a filename and an empty argument. A string containing a command and all its options loses these boundaries. Use <Code>{'command "${args[@]}"'}</Code> to pass the array instead of assembling text for eval. Functions receive their own positional arguments; <Code>local</Code> gives a function a scoped variable so a temporary label does not replace its caller's label.</Prose>
  <Example name="scope"><Prose>The function receives two values, including the empty one, and its local label disappears when it returns. <Code>export label</Code> makes the outer label available in subsequently launched child processes. The child changes its own environment copy; that does not reassign the parent's variable. The two newline characters at the end of command-substitution output are removed by <Code>$(...)</Code>, explaining why captured is just “two lines.” Internal newlines would remain.</Prose></Example>
  <Prose>Changes to the current directory also belong to a process. A parent script can use <Code>cd</Code> to change where later relative paths resolve; a subshell created with parentheses or command substitution cannot change its parent's directory. <Code>source settings.sh</Code> runs that file's commands in the current shell, so assignments and cd can affect the caller. It is executing code, not merely reading a data format. Prefer explicit arguments or a data parser when those are the intended interfaces.</Prose>

  <H2>4. Treat output and status as separate results</H2>
  <Prose>A command can return bytes on <strong>stdout</strong>, diagnostic bytes on <strong>stderr</strong>, and a small integer <strong>exit status</strong>. Status 0 conventionally means success; nonzero meanings belong to the command. Seeing output does not prove success: a producer might write several rows, then fail. Save <Code>$?</Code> immediately after the command whose status you need, because the next command replaces it.</Prose>
  <LessonTable caption="Connect the right channels" headers={['Syntax','Effect','Consequence']} rows={[["command > result.txt",'Open/truncate result.txt and route stdout there','The destination changes before command success is known'],['command 2> errors.txt','Route stderr to its own file','Keep machine-readable output separate from diagnostics'],['command >> log.txt','Append stdout','Repeated runs accumulate; this is not a fresh complete report'],['producer | consumer','Feed producer stdout into consumer stdin','Stderr stays separate unless explicitly redirected'],['command > all.txt 2>&1','Make stderr follow the already redirected stdout','Order matters: 2>&1 > result.txt routes differently']]} />
  <BashStatusLab/>
  <Example name="statuses"><Prose>The producer writes a row but returns 4. Normally the pipeline reports cat's successful 0; with pipefail it reports 4. If several pipeline commands fail, pipefail selects the rightmost nonzero status, not the first failure in time. <Code>PIPESTATUS</Code> gives each command's status; copying the entire array immediately preserves it before another command overwrites that information.</Prose><Prose>grep status 1 means a valid search found no matching lines; a larger status means an error. The example handles no match as an expected result and routes other failures to stderr. This policy is part of the script's meaning, not an automatic property of nonzero.</Prose></Example>
  <Prose><Code>if command; then ...; else ...; fi</Code> chooses a branch from the command's status. <Code>command || exit 1</Code> handles a failure by terminating. <Code>command &amp;&amp; next_command</Code> runs the second command only after success. These are status decisions; do not confuse them with whether printed text is empty.</Prose>
  <H3>What set -e does—and why it is not a failure policy</H3>
  <Prose><Code>set -e</Code>, often called errexit, exits on some unhandled failures. It has context-sensitive exceptions: conditions, portions of AND/OR lists, pipelines and functions invoked from tested contexts can behave differently. It does not roll back output, inspect whether JSON is valid, or decide whether grep's no-match status should be acceptable. <Code>set -u</Code> detects many unset-variable expansions; pipefail changes a pipeline's status. Combining these flags is not a proof that every failed action terminates correctly.</Prose>
  <Example name="errexit"><Prose>probe is the condition of if, so its internal false does not stop execution under -e. The following printf succeeds, becoming the function's return status; the outer branch sees success. A tempting “strict mode makes failure impossible to miss” assumption is therefore false. Use deliberate checks around steps whose failure must block publication, as the report wrapper below does.</Prose></Example>
  <details><summary>Early consumers and expected broken pipes</summary><Prose>A consumer such as head can deliberately stop after enough input. An upstream writer may then receive SIGPIPE because no reader remains. With pipefail, that can make the whole pipeline nonzero even if the first few lines were exactly what you wanted. Define whether early termination is permitted; do not erase every nonzero status with <Code>|| true</Code>. This lesson's report worker runs to completion before publication.</Prose></details>

  <H2>5. Visit files and keep state in the right process</H2>
  <Prose>A filename is an argument-sized value, not a line from a human-readable listing. Do not split the output of ls to drive a loop. In Bash, collect a glob's matches into an array and iterate over quoted elements. With <Code>shopt -s nullglob</Code>, a pattern with no matches produces no elements instead of leaving the literal pattern in the list. By default, <Code>./*.csv</Code> excludes names starting with a dot.</Prose>
  <Example name="files"><Prose>The array contains exactly two filenames, so the space in run alpha.csv causes no split. No JSON files exist, so the second array is empty. The first while loop is on the right of a pipeline; under default Bash settings it runs in a subshell, and its counter changes do not reach the parent. The second loop receives its text through a here-document and runs in the current shell, so the final count is 2.</Prose></Example>
  <Prose><Code>IFS= read -r line</Code> reads a text line without trimming it through field splitting or interpreting backslashes as escapes. A quoted here-document delimiter, as in <Code>&lt;&lt;'ROWS'</Code>, prevents parameter and command expansion in its body. Reading arbitrary path streams needs a delimiter that cannot appear in filenames, usually NUL with suitable tools; ordinary newline records are fine for this explicitly line-based fixture.</Prose>
  <Checkpoint prompt="Your pipeline loop finds ten files, but its counter is still zero afterward. Does adding export to the counter fix the process boundary?"><Prose>No. Export passes a value into a child environment; it does not send the child's later assignments back. Restructure the input redirection so the loop runs in the parent, or make the child return its result explicitly. Bash's lastpipe option changes some pipeline execution cases; it is not enabled in this fixture.</Prose></Checkpoint>

  <H2>6. Publish only a complete report</H2>
  <Prose>A report's public filename is an interface for another process. If you redirect the worker straight into that filename, the shell truncates the old report before the worker runs. A worker failure can leave the reader with a partial file. Instead, write to a private staging file, inspect the worker's status, and publish only on success.</Prose>
  <BashPublicationLab/>
  <Prose>Place staging in the destination's directory so the final move can use a same-filesystem rename. With the successful local rename assumed here, a reader opening the public name gets the old complete file or the new complete file, rather than an intermediate write through that name. A reader already holding the old file open may continue reading it. This property concerns visibility of one filename, not durability after power loss or a transaction covering many outputs.</Prose>
  <Prose><Code>trap cleanup EXIT</Code> registers a function to run when the shell exits; it removes the temporary directory owned by this run. Separate INT and TERM traps turn common interruptions into nonzero exits, which then trigger cleanup. They do not catch SIGKILL or a power failure. Concurrent successful publishers can still replace each other's outputs; use separate run names or a coordination protocol when that is not the intended policy.</Prose>

  <H2>7. Build the complete report job</H2>
  <Prose>Create a new exercise directory, and save all five files below there. The Python worker reads one column named value, converts finite numbers and produces one JSON object. A header-only file is valid empty data: count 0 and mean null. Blank CSV records are skipped by the reader. CSV parsing uses the standard-library default dialect; this is a deliberately small numerical report, not a general schema or streaming-memory engine.</Prose>
  <File name="report.py"/>
  <Prose>DictReader supplies strings; float converts each reading, and isfinite rejects NaN and infinities. math.fsum reduces avoidable summation error, then division gives the mean. The code collects its finite fixture values in a list; it does not claim constant memory. json.dumps with allow_nan=False prevents publication of nonstandard NaN/Infinity values. Diagnostics go to stderr, while stdout contains only the result.</Prose>
  <Prose>In the wrapper, <Code>(( $# != 2 ))</Code> is a numeric condition: it succeeds when the argument count is not two, selecting the usage branch. <Code>[[ ... ]]</Code> groups Bash tests; <Code>-f</Code> asks whether a path is a regular file, <Code>-d</Code> whether it is a directory, <Code>!</Code> negates a test, and <Code>||</Code> means either condition suffices. A satisfied condition has status 0 even though ordinary arithmetic uses zero to represent false.</Prose>
  <File name="run-report.sh"/>
  <Prose>The wrapper validates exactly two arguments before expanding them. Paths are relative to the caller's working directory; script_dir separately locates report.py beside the wrapper, even if you invoke the wrapper from another directory. Its CDPATH setting prevents cd from emitting an unexpected directory announcement into the captured path. A successful worker exit allows mv; failure preserves the old destination and returns the worker's status. The wrapper itself reports publication on stderr, so its stdout remains empty.</Prose>
  {['run alpha.csv','empty.csv','invalid.csv'].map(name=><File key={name} name={name}/>)}
  <H3>Run a successful job, then a failed replacement</H3>
  <CodeBlock language="bash">{`bash run-report.sh 'run alpha.csv' 'outputs/report.json'
cat -- 'outputs/report.json'
bash run-report.sh 'invalid.csv' 'outputs/report.json'
status=$?
printf 'status=%s\\n' "$status"
cat -- 'outputs/report.json'
bash run-report.sh 'empty.csv' 'outputs/empty.json'
cat -- 'outputs/empty.json'`}</CodeBlock>
  <CodeBlock language="output">{`First report: {"count": 3, "mean": 4.0}
Failed replacement status: 4
Report after failure: {"count": 3, "mean": 4.0}
Empty report: {"count": 0, "mean": null}`}</CodeBlock>
  <Prose>The labeled lines above summarize the results to compare; actual cat prints only the JSON and printf prints status=4. stderr also explains the nonfinite input and successful publication. The failed invocation must not replace the earlier count 3 report. Running the successful job again replaces the same output rather than appending duplicate JSON.</Prose>
  <LessonTable caption="Diagnose the boundary that failed" headers={['Change','Expected behavior','Reason']} rows={[["Omit an argument",'Status 2; usage on stderr','The wrapper rejects the invocation before reading $2'],['Missing input path','Status 2; old output retained','Precondition fails before staging'],['Wrong header, extra column or nonfinite reading','Worker failure; old output retained','Input must satisfy the report contract'],['Valid header with no readings','Status 0; count 0, mean null','Empty data is different from a missing file'],['Output destination is a directory','Status 2; no report published','A file destination is required']]} />
  <details><summary>Transfer the publication mechanism to generated artifacts</summary><Prose>A static-site index, experiment manifest or local cache file has the same reader problem: a filename can exist while its producer is still writing. Stage the complete representation, validate it, then rename into place. For our manifest, “valid” includes JSON schema and numerical constraints; for a site index it might include internal link checks. If a bundle needs several files to agree, a single file rename is insufficient: publish a versioned directory and switch a small version pointer through a deliberately designed protocol.</Prose></details>

  <H2>8. Independent investigation: collect several reports</H2>
  <Prose>Write <Code>collect-reports.sh OUTPUT.json INPUT.json...</Code>. It should validate and combine one or more reports into a JSON array in argument order. Each entry records source, count and mean. Preserve filenames with spaces. Require a nonnegative integer count, mean null exactly when count is zero, and a finite numeric mean otherwise. Publish nothing if any input fails; retain an existing output. An empty report is valid, while zero supplied input paths is an invocation error.</Prose>
  <CodeBlock language="bash">{`bash collect-reports.sh 'outputs/collection.json' 'outputs/report.json' 'outputs/empty.json'
cat -- 'outputs/collection.json'`}</CodeBlock>
  <CodeBlock language="json">{`[{"count": 3, "mean": 4.0, "source": "outputs/report.json"}, {"count": 0, "mean": null, "source": "outputs/empty.json"}]`}</CodeBlock>
  <LessonTable caption="Independent acceptance cases" headers={['Case','What must hold']} rows={[["Two valid reports",'Two objects, matching argument order and source spelling'],['A valid empty report','Kept with count 0 and null mean'],['Missing, invalid JSON, count true, or nonfinite mean','Nonzero status; preexisting collection unchanged'],['No input paths','Status 2; usage diagnostic'],['Reverse the input order','Reverse the output entries; no sorting shortcut'],['Run twice with the same inputs','Same JSON content; one complete array, no duplicates from appending']]} />
  <details><summary>Hint: separate argument policy, data policy and publication</summary><Prose>Take output from $1, then shift and pass the remaining <Code>"$@"</Code> unchanged. Have Python parse JSON and write only a complete valid array into staging. With <Code>python3 - "$@" &lt;&lt;'PY'</Code>, the fixed Python program comes from stdin while paths arrive as separate arguments. The quoted delimiter stops Bash from expanding the Python source. Reuse the report's success/rename/cleanup lifecycle.</Prose></details>
  <details><summary>Complete solution: collect-reports.sh</summary><div data-bash-practice><CodeBlock language="bash">{summaryPractice}</CodeBlock></div><Prose>shift changes only this script's positional parameters. Python receives every remaining path, including spaces or an empty argument, as one value. The strict type check rejects JSON true as a count even though Python bool is a subclass of int. Missing files and invalid values reach stderr and status 4; a staged result replaces the destination only after all inputs validate. Additional JSON fields are ignored by this collector. It reads local inputs in order, not a simultaneous snapshot of files another process may be changing.</Prose></details>
  <Checkpoint prompt="One input has count 0 and mean 0. Another has count 2 and mean null. Should either be accepted? What if the second input fails after the first was read?"><Prose>Neither satisfies the declared schema: empty requires null, while a nonempty mean must be finite numeric data. No collection is published after the later failure, so the first successfully read input cannot leave a partially updated public array. The old destination remains available.</Prose></Checkpoint>
  <H3>Make the next failure explainable</H3>
  <Prose><Code>bash -n script.sh</Code> checks syntax without running the script; it does not prove filenames, statuses or algorithms correct. <Code>bash -x script.sh ...</Code> executes while showing expanded commands; use the disposable examples and remember that command arguments appear in that diagnostic output. ShellCheck can identify likely shell mistakes such as unintended splitting. Native changed-input tests still establish the workflow's actual behavior.</Prose>
  <Prose>You are ready to move on when you can explain each argument boundary, distinguish no match from failed execution, identify which process owns a variable and justify why a failed producer cannot replace this report. Next in this module is <a href="/learn/topic/os-processes-virtual-memory-isolation">OS Processes, Virtual Memory &amp; Isolation</a>, then <a href="/learn/topic/threads-concurrency-locks-deadlocks">Threads, Concurrency, Locks &amp; Deadlocks</a>. Those lessons explain the execution and coordination mechanisms beneath these scripts.</Prose>
  <Sources alternatives={<LearningResources>
   <li><a href="https://www.youtube.com/watch?v=kgII-YWo3Zw">MIT Missing Semester: Shell Tools and Scripting video</a> and <a href="https://missing.csail.mit.edu/2020/shell-tools/">written lesson and exercises</a> — a guided Bash workflow from the 2020 course. Use its quoting, variables and functions sections alongside this page; its tool demonstrations assume a Unix environment. Keep this lesson's explicit distinction between grep no-match and error statuses.</li>
   <li><a href="https://www.shellcheck.net/wiki/SC2086">ShellCheck SC2086: double quoting and argument boundaries</a> — a focused explanation with repaired examples and array alternatives. Useful when a script works for simple filenames but fails for spaces, empty values or patterns.</li>
  </LearningResources>}>
   <li><a href="https://www.gnu.org/software/bash/manual/bash.html">GNU Bash reference manual</a> — shell expansion, parameters, arrays, pipeline status, set and trap. The installed Bash also provides <Code>help set</Code>, <Code>help trap</Code> and <Code>help read</Code> for its own version.</li>
   <li><a href="https://docs.python.org/3/library/csv.html">Python csv documentation</a> — DictReader fields, newline handling and dialect scope.</li>
   <li><a href="https://docs.python.org/3/library/json.html">Python json documentation</a> — parsing, encoding and allow_nan behavior.</li>
   <li><a href="https://man7.org/linux/man-pages/man2/rename.2.html">Linux rename manual</a> — destination replacement, existing open descriptors and filesystem boundaries.</li>
  </Sources>
 </div>,
};
