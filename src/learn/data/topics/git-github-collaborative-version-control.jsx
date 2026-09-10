import { Code, CodeBlock, H2, H3, Prose } from "../../components/content";
import { Checkpoint, LessonIntro, LessonTable, Sources } from "../../components/lesson-labs/LessonElements";
import TerminalExample from "../../components/lesson-labs/TerminalExample";
import { GitStagingLab, GitBranchLab, GitConflictLab, GitRemoteLab } from "../../components/lesson-labs/GitFoundationsLabs";
import { LearningResources } from "../../components/lesson-labs/LessonInvestigation.jsx";
import { gitExamples } from "../git-examples.js";
import { gitPracticeExamples } from "../git-practice-examples.js";

export default {
  title: "Git, GitHub & Collaborative Version Control",
  readTime: "~45 min read + 100 min practice",
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot">
    <LessonIntro exampleKind="Shell" prerequisites="Comfort editing a text file. No earlier Git or Linux course is assumed. Commands use Bash; the few required shell operations are explained here."
      sections={[["1-know-what-git-records", "Mental model"], ["2-make-a-small-local-commit", "First commit"], ["3-stage-content-not-a-promise-to-save-later", "Staging"], ["6-resolve-a-real-merge-conflict", "Conflicts"], ["9-fetch-before-deciding-how-to-integrate", "Remotes"], ["10-turn-a-branch-into-a-reviewable-pull-request", "GitHub"], ["12-practise-a-complete-change", "Practise"]]}>
      Record a coherent change, review precisely what will be committed, combine work without losing it, and understand what happens before publishing to a shared repository. Four investigations and complete disposable examples expose the stored state and history.
    </LessonIntro>
    <H2>1. Know what Git records</H2>
    <Prose>You improve a report, a teammate changes its description, and you need to keep both contributions. Numbered folders such as project-final-2 cannot explain ancestry or isolate a regression reliably. Git records snapshots and their relationships, lets you compare them, and helps integrate separate lines of work. GitHub is a hosting and collaboration service around repositories; local commits do not require it or an internet connection.</Prose>
    <LessonTable caption="The places you must distinguish" headers={["Place", "Meaning", "Useful inspection"]} rows={[
      ["Working tree", "Files currently on disk that you edit", "git status; git diff"],
      ["Index / staging area", "The proposed snapshot for the next ordinary commit", "git diff --staged"],
      ["HEAD commit", "The currently checked-out commit, usually through a branch", "git show HEAD; git log"],
      ["Local branch", "A movable reference to a commit", "git branch; git branch --show-current"],
      ["Remote-tracking reference", "Your locally recorded view, such as origin/main", "git branch -r; git fetch"],
    ]} />
    <Prose>A commit records a tree snapshot, parent reference(s), author/committer information and a message. It is not just a text diff; Git computes comparisons between snapshots. A branch is a reference, not a separate folder copied for each change. Normally HEAD names your current branch; committing advances that branch. In detached HEAD state, HEAD points directly to a commit—create a branch if you want to keep developing that line of work.</Prose>
    <Prose>A clean status does not mean the program is correct, tested or backed up remotely. Git does not automatically capture your environment, ignored data or files you never staged. Record which inputs and environment produced an analysis; the later <a href="/learn/topic/reproducible-notebooks-experiment-structure">reproducible-notebook lesson</a> develops that manifest. Here we first learn what is actually in a commit.</Prose>

    <H2>2. Make a small local commit</H2>
    <Prose>A terminal runs commands through a shell. Open Bash: for example Git Bash on Windows, or a Bash session on macOS/Linux. These blocks use Bash syntax, so do not paste them directly into PowerShell. Check <Code>git --version</Code> first; if it is unavailable, follow the <a href="https://git-scm.com/downloads">official installation instructions</a>. The examples create fresh temporary folders and configure identity only there. Their current verification uses Git 2.48.1.windows.1 and Git Bash; this is a tested snapshot, not a newest-version claim.</Prose>
    <LessonTable caption="The small shell vocabulary used below" headers={['Syntax','Meaning','Why it is here']} rows={[["name=value; $name",'Store a shell variable; read its value','Remember the disposable practice folder'],['$(command)','Use the command’s output as text','Capture the directory chosen by mktemp'],['cd folder','Change the current directory','Tell later Git commands which repository to inspect'],['printf ... > file','Write formatted text, replacing the file','Create invented versions in the practice folder'],['cat file','Print a file’s contents','Check what is actually on disk'],['command || exit 1','Stop this shell script if that command fails','Avoid continuing in the wrong directory'],['"$lab/repo"','Quote a path so it is one argument','Keep spaces from splitting the path']]}/>
    <Prose>In printf, %s inserts a text argument and \n ends the line; the quoted format keeps those characters together. Enter a complete worked block together, or save it as a .sh file and run it with bash. The guard's exit stops that script; if run directly in an interactive terminal, it can close that shell session. Each block initializes its own folder, so repeating one does not depend on earlier examples. The main branch name is chosen explicitly instead of relying on a system default.</Prose>
    <Prose>Mktemp -d chooses an unused directory. The cd guard stops if entry fails, so later commands do not act on your current project. -q reduces incidental progress output. This is practice identity using a reserved invalid domain, not account authentication. For real commits, choose the identity and email privacy settings you intend to publish. The examples leave disposable files in their printed-or-inspectable $lab location; they never push to the internet.</Prose>
    <TerminalExample example={gitExamples.first}><Prose>Question marks mean untracked. After add, A in the first status column means the new file is staged. Commit records it, then log prints the message and branch reports main. Successful commands can be silent: lack of output is not evidence that they did nothing.</Prose></TerminalExample>
    <Prose>Init creates a repository in a new location; clone copies an existing repository and configures a remote. Do not initialise a second repository inside an existing project by accident. A real first workflow is: inspect status, edit one coherent change, review the diff, stage selected paths, review the staged diff, run relevant tests, then commit with a message explaining the outcome.</Prose>
    <Prose>Use add with explicit paths while learning. Add -p lets you review and stage individual hunks. A coherent commit might include an implementation, its tests and documentation together; “one file per commit” is not the goal. Commit -a stages changes to already tracked files but does not include new untracked files, and can sweep in unrelated tracked edits.</Prose>

    <H2>3. Stage content, not a promise to save later</H2>
    <GitStagingLab />
    <TerminalExample example={gitExamples.staging}><Prose>Read the three version lines as HEAD=1, index=2, working file=3. Commit records version 2 because that is what was staged. Afterwards HEAD is 2 but the working file is still 3, shown by the remaining unstaged M. Add does not subscribe the index to future edits.</Prose></TerminalExample>
    <LessonTable caption="Read the two short-status columns" headers={["Status", "Index versus HEAD", "Working tree versus index"]} rows={[
      ["M followed by a space", "Modified and staged", "No later tracked edit."],
      ["Space followed by M", "No staged change", "Modified but unstaged."],
      ["MM", "Modified and staged", "Also modified after staging."],
      ["??", "Untracked path", "Not included until staged."],
      ["UU", "Unmerged path", "Both sides changed it in this conflict example."],
    ]} />
    <Prose>Git diff compares working content with the index; git diff --staged compares index with HEAD; git diff HEAD compares working content with HEAD for tracked paths. Ordinary diff does not include the contents of untracked files. Git show HEAD:report.txt reads the committed blob; git show :report.txt reads the indexed blob. Those colon forms are inspection tools, not different disk filenames.</Prose>
    <Checkpoint prompt="You stage a correct function, then break it while experimenting. A normal git commit succeeds. Which version did it record?">
      <Prose>The staged version. Your working file can remain broken. Conversely, tests can pass on unstaged corrections while the staged snapshot still contains a bug. Review the staged diff and test the intended commit contents, not just whatever happens to be on disk.</Prose>
    </Checkpoint>

    <H2>4. Undo the correct layer</H2>
    <TerminalExample example={gitExamples.unstage}><Prose>Restore --staged resets this file's index entry from HEAD while leaving version 2 on disk. The later plain restore replaces that unstaged working edit from the index, returning version 1. The second action discards data; use it only for edits you explicitly do not want.</Prose></TerminalExample>
    <LessonTable caption="Choose by what you intend to change" headers={["Intent", "Tool", "Risk / boundary"]} rows={[
      ["Remove a path from the proposed commit", "git restore --staged -- path", "Keeps working edits; default source is HEAD."],
      ["Discard an unwanted unstaged edit", "git restore -- path", "Replaces working content from the index by default; inspect and back up first."],
      ["Undo a published ordinary commit", "git revert COMMIT", "Creates a new inverse change; may conflict."],
      ["Change a private latest commit", "git commit --amend", "Creates a replacement commit ID; coordinate if already shared."],
      ["Move a local branch reference", "git reset", "Mode determines what happens to index/worktree; not a generic undo button."],
    ]} />
    <Prose>Reset --soft moves HEAD while keeping index and worktree; mixed reset also resets the index; hard reset can discard tracked working edits and remove obstructing untracked paths. Do not copy a hard-reset or git clean deletion command to make a confusing status disappear. First identify and preserve the changes you need. Git usually cannot recover text that was never committed or otherwise saved.</Prose>
    <Prose>Reflog records local reference movements and may help locate an abandoned commit. Inspect it, identify the right commit, and create a rescue branch before further changes. Reflogs expire and unreachable objects can be collected; they are not a permanent backup. Tag an important version deliberately, and maintain an appropriate backup or trusted remote.</Prose>

    <H2>5. Work on a branch and integrate deliberately</H2>
    <GitBranchLab />
    <TerminalExample example={gitExamples.branch}><Prose>The note exists on add-note after its commit, but not on main until integration. Because main has not moved independently, --ff-only can advance it to the feature commit without a merge commit. The history contains two commits, not two disconnected copies of the project.</Prose></TerminalExample>
    <CodeBlock language="text">{`Before integration:
A  main
 \\
  B  add-note

After fast-forward:
A -- B  main, add-note`}</CodeBlock>
    <Prose>Switch -c creates and checks out a branch; switch alone selects an existing one. Inspect status before switching: compatible uncommitted changes can follow you, while changes that would be overwritten cause Git to refuse. A branch is not a container for uncommitted edits. Commit, deliberately stash, or use a separate worktree when you need a clean parallel workspace.</Prose>
    <Prose>If both branches have progressed, a normal merge can create a commit with two parents. Rebase instead replays commits on a new base, producing new IDs for rewritten commits. Agree on merge versus rebase conventions; do not casually rebase history other people already depend on. A fast-forward-only update fails on divergence, giving you a chance to inspect rather than silently choosing a history policy.</Prose>

    <H2>6. Resolve a real merge conflict</H2>
    <Prose>A conflict means Git cannot automatically combine edits—not that one contributor is wrong. This example changes the same line on two branches. The initial title is deliberately simple; the resolution is a decision that the report should describe both model and region. In real code, a textually clean merge can still contain a semantic bug.</Prose>
    <GitConflictLab />
    <TerminalExample example={gitExamples.conflict}><Prose>UU marks the unresolved path. Index stage 1 is the common base, stage 2 is our main-side content, and stage 3 is the incoming branch. After writing the intended resolution, add marks the path resolved; commit completes integration. Four commits remain: base, two branch changes and the merge.</Prose></TerminalExample>
    <Prose>Typical conflict markers separate HEAD-side and incoming content. Read both plus the base, edit the final intended file, remove marker text, run tests and inspect git diff --cached before committing. Diff --check helps detect whitespace errors and conflict-marker mistakes; it cannot prove the resolution is logically correct.</Prose>
    <Prose>If the merge is not what you intended, git merge --abort attempts to return to the pre-merge state. Starting with unrelated uncommitted changes can make recovery harder, so prepare first. Rebase has its own --continue and --abort workflow. During rebase, the interpretation of “ours” and “theirs” differs from the intuitive feature-versus-main reading; inspect actual contents rather than selecting a side by label alone.</Prose>
    <Checkpoint prompt="Both people edited different lines, and Git merged without a conflict. Can you skip tests?">
      <Prose>No. One change might rename a function while another adds a call to its old name elsewhere. Textual merge success says nothing about that dependency. Run tests and review the combined behaviour.</Prose>
    </Checkpoint>

    <H2>7. Reverse a committed change without hiding history</H2>
    <TerminalExample example={gitExamples.revert}><Prose>The wrong-units commit remains in history, followed by a revert that restores version 1. The count is three commits. This preserves an audit trail and is usually easier to coordinate for shared history than making the bad commit disappear from a branch.</Prose></TerminalExample>
    <Prose>Revert computes the inverse patch relative to the target commit; later edits can make that inverse conflict. It is not always equivalent to restoring an entire old snapshot. Reverting merge commits requires choosing a mainline parent and understanding future merge effects; do not apply the ordinary one-parent recipe blindly to a merge.</Prose>

    <H2>8. Pause work and ignore the right files</H2>
    <TerminalExample example={gitExamples.stash}><Prose>The -u option includes the untracked draft. The clean-status call prints nothing, and the tracked report returns to version 1. Apply restores the draft but keeps the stash entry, so it can be inspected before deliberate removal. Stashes are local and can conflict when applied elsewhere.</Prose></TerminalExample>
    <Prose>Default stash does not include untracked files; -u includes them but not ignored files. Pop applies and, on success, removes the stash; apply is easier to verify before dropping it. Restoring the previous staged/unstaged arrangement can require --index. A stash is temporary storage, not a substitute for a meaningful commit or backup.</Prose>
    <TerminalExample example={gitExamples.ignore}><Prose>Run.log is ignored, but the already-tracked report still appears modified even though .gitignore names it. Ignore rules affect untracked discovery; they do not erase tracked history or protect secrets retroactively. The .gitignore file itself is a new untracked file until you stage it.</Prose></TerminalExample>
    <Prose>Keep environment files containing credentials, caches and generated model checkpoints out of ordinary commits. Track configuration templates without credentials. Git LFS or an artifact store may suit large files, depending on the project. If a secret has been committed or published, rotate/revoke it and follow a coordinated cleanup process; adding its filename to .gitignore is not remediation.</Prose>
    <Prose>Line endings can create noisy diffs across operating systems. Agree on text rules with .gitattributes rather than changing repository-wide conversion settings impulsively. These fixtures set core.autocrlf false locally to keep their tiny text examples predictable.</Prose>

    <H2>9. Fetch before deciding how to integrate</H2>
    <GitRemoteLab />
    <Prose>The next example simulates collaboration entirely on disk. A bare repository stores shared Git data without a normal checked-out working tree. A second clone represents a colleague. None of the push commands sends files to GitHub or another network service.</Prose>
    <TerminalExample example={gitExamples.remote}><Prose>After fetch, our working report still says version 1 while origin/main contains version 2. The two counts are 0 commits unique to HEAD and 1 unique to origin/main. Only the subsequent fast-forward merge updates our working report. Fetch refreshes knowledge; it does not integrate into the checked-out branch.</Prose></TerminalExample>
    <Prose>Origin is a conventional remote name, not a special server. Origin/main is a local remote-tracking reference, not a live query. Push publishes selected objects and updates references when the remote permits it. The -u option establishes an upstream association for the branch; it does not grant permissions or make every future branch track automatically.</Prose>
    <Prose>Pull combines fetching with an integration strategy, commonly merge or rebase according to options/configuration. “Pull before starting” is incomplete advice: inspect local work first, fetch, understand divergence, then integrate using the team's policy. For a straightforward no-divergence update, an explicit fast-forward-only strategy makes refusal preferable to an unexpected merge.</Prose>
    <Prose>If a push is rejected because the remote advanced, fetch and compare the histories. Do not reflexively force-push. Force-with-lease adds an expected-reference check but still rewrites a shared reference when it succeeds; it requires an agreed workflow and is not permission to overwrite other work. Protected branch rules may forbid the update entirely.</Prose>

    <H2>10. Turn a branch into a reviewable pull request</H2>
    <Prose>On GitHub, a pull request proposes changes from a head branch into a base branch. Choose both deliberately: a correct patch against the wrong base is still the wrong proposal. A fork is a separate hosted repository under another owner; a clone is a local repository. Use a fork when the contribution workflow requires it rather than assuming you can push to the upstream project.</Prose>
    <CodeBlock language="text">{`Example PR description
Purpose: make the report's measurement units explicit.
Change: add a units note and a regression check.
Evidence: report tests pass; output units are unchanged.
Scope: no data files, credentials or model artifacts included.
Review question: is the chosen terminology clear to new readers?`}</CodeBlock>
    <Prose>Authenticate using the organisation's supported HTTPS credential or SSH workflow. Keep tokens and private keys out of repository files, shell history and remote URLs you share. Review the destination and staged content before publishing. Local author identity is metadata; it is not proof of authenticated access.</Prose>
    <Prose>Before requesting review, read the complete diff as the reviewer will, run checks and explain risks. Address comments with follow-up changes and discussion. Passing CI is evidence about configured checks, not automatic proof of correctness. Branch rules may require approvals or checks. Merge, squash and rebase-based PR completion leave different histories, so follow the repository convention rather than choosing an option by its label alone.</Prose>

    <H2>11. Inspect history to answer a question</H2>
    <LessonTable caption="Read-only tools for understanding a project" headers={["Question", "Tool", "Interpretation"]} rows={[
      ["Which commits led here?", "git log --oneline --graph --decorate --all", "A compact ancestry view; labels identify references."],
      ["What changed in one commit?", "git show COMMIT", "Metadata plus a patch view; inspect the surrounding context too."],
      ["When did a path change?", "git log -- path", "Path-focused history; renames may require further tracing."],
      ["Where did this line last change?", "git blame path", "Last recorded line change, not moral responsibility or full origin."],
      ["Which commit introduced a reproducible failure?", "git bisect", "Binary search over good/bad commits with a trustworthy test."],
    ]} />
    <Prose>Place -- before a path when it could otherwise look like an option or revision. Short commit IDs and dates differ between repositories; the examples print stable messages and counts instead of pretending everyone will get the same hash. Hooks, submodules, worktrees, signing and advanced history repair are useful deeper topics, not prerequisites for this first collaborative change.</Prose>

    <H2>12. Practise a complete change</H2>
    <details className="nt-deeper"><summary>Use history as an experiment: find the first failing commit</summary>
      <Prose>Version history can answer a technical question, not just recover old text. Suppose the intended report value must stay below 50. Six commits contain 10, 11, 12, 99, 99 and 99. You know the oldest passes and the newest fails. Instead of manually reading every change, bisect checks intermediate snapshots and narrows the first good-to-bad transition using a repeatable test.</Prose>
      <div className="nt-flow"><span>Known good: 10<small>Passed the same test</small></span><span>Test a middle snapshot<small>Pass narrows toward newer; fail toward older</small></span><span>First bad: 99<small>Investigate this change and its context</small></span></div>
      <Prose>The test command returns exit status 0 for good and 1 for bad. Here test compares the number from cat with 50 using -lt, meaning less than. In real use, the predicate might verify units, a serialization round trip or a deterministic regression test. The test script lives outside the checked-out repository so changing snapshots cannot remove it.</Prose>
      <TerminalExample example={gitPracticeExamples.bisect}><Prose>The for loop writes each listed value, then creates a commit. --allow-empty keeps the last two history steps even though their file contents are unchanged; ordinary Git would refuse those empty changes. HEAD~5 follows five first-parent steps to the initial snapshot. Tags preserve the chosen endpoints. Bisect changes the checkout while testing; reset restores the original branch. Redirecting incidental output to a temporary log leaves only the discovered commit message, its value and the restored branch name.</Prose></TerminalExample>
      <Checkpoint prompt="Could an intermittently failing test, or a bug that is introduced and later fixed, mislead this investigation?">
        <Prose>Yes. The result depends on reliable classifications and a suitable good-to-bad transition. Reproduce failures, check environment/data compatibility across snapshots, and use skip for an untestable revision rather than guessing. Several skipped commits can leave the first culprit ambiguous. A detected commit localizes evidence; it does not prove who is responsible or explain the root cause.</Prose>
      </Checkpoint>
    </details>
    <Checkpoint prompt="Repeat the staging example but run git add again after writing version 3. What changes?">
      <Prose>The index becomes version 3, so the commit records 3. With no other changes, status becomes clean. Compare HEAD, index and working content again to verify all three agree.</Prose>
    </Checkpoint>
    <Checkpoint prompt="In the local remote example, make a local commit before fetching the colleague's commit. Will --ff-only still work?">
      <Prose>No: the branches diverge, with one unique commit on each side in this setup. Preserve both, inspect the changes and deliberately merge or rebase under an agreed policy. A force-push is not the default solution.</Prose>
    </Checkpoint>
    <Checkpoint prompt="You added a credentials file to .gitignore after committing it. Is it now absent from history?">
      <Prose>No. It remains tracked and in prior commits. Treat a real exposed credential as compromised, rotate it and coordinate cleanup. A disposable practice repository should contain only dummy text, never a real secret.</Prose>
    </Checkpoint>
    <Checkpoint prompt="Complete a private practice change: add a report note on a branch, review it, integrate it and explain how to reverse it.">
      <Prose>Start from the first fixture, switch -c to a topic branch, create the note, inspect status/diff, add that path, inspect --staged and commit. Switch main and use --ff-only when main has not diverged. Revert the ordinary note commit if you need an auditable reversal. Verify the note's presence/absence and log rather than trusting a command's exit alone.</Prose>
    </Checkpoint>
    <H3>Independent task: commit the explanation, retain the experiment</H3>
    <Prose>Begin with tracked result.txt containing “value 1” and note.txt containing “draft”. On an explain-units branch, change result.txt to “value 2” and note.txt to “units: ms”. Commit only the note. Predict the committed contents of both files and the remaining short status. Check the snapshots directly; a commit message alone is not evidence.</Prose>
    <details><summary>Hint: stage a path, then inspect that snapshot</summary><Prose>Stage note.txt explicitly. After committing, git show HEAD:result.txt and git show HEAD:note.txt read committed contents; cat result.txt reads the later working edit. Do not use commit -a for this task.</Prose></details>
    <details><summary>Worked solution and exact result</summary><TerminalExample example={gitPracticeExamples.transfer}/><Prose>The branch commits its staged snapshot, so result.txt remains version 1 in history while your version 2 experiment stays on disk. The leading space before M means a working-tree modification relative to the index.</Prose></details>
    <Checkpoint prompt="Now deliberately include the result change in a second commit. Which command sequence and observations demonstrate that you succeeded?">
      <Prose>Review git diff, add result.txt, inspect git diff --staged, run the relevant check, then commit. HEAD:result.txt now reads value 2; note.txt still records units: ms, and status is clean if nothing else changed. Two commits explain two coherent steps without discarding the experiment.</Prose>
    </Checkpoint>
    <Prose>In the opening curriculum, next <a href="/learn/topic/linux-basics-filesystems-processes">learn the filesystem and process model</a> behind these commands. The reader's named Next link follows your selected route. Bash automation is a later lesson; first understand the operations being automated.</Prose>
    <Sources alternatives={<LearningResources>
      <li><a href="https://www.youtube.com/watch?v=9K8lB61dl3Y">MIT Missing Semester — Version Control and Git (2026)</a> · Video lecture with <a href="https://missing.csail.mit.edu/2026/version-control/">written notes and exercises</a>. Use its data-model section after the branch investigation and its command-line section for review. The notes give a second route through snapshots, graphs and collaboration.</li>
      <li><a href="https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell">Pro Git — Branches in a Nutshell</a> · An illustrated written explanation of movable references and HEAD. Read alongside the branch lab; connect each pointer movement to a command you already tried.</li>
    </LearningResources>}>
      <li><a href="https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository">Git: tracked files, status and snapshots</a></li>
      <li><a href="https://git-scm.com/docs/git-restore">Restore sources and target locations</a></li>
      <li><a href="https://git-scm.com/docs/git-reset">Reset modes and risks</a></li>
      <li><a href="https://git-scm.com/docs/git-merge">Merge, conflicts and abort behaviour</a></li>
      <li><a href="https://git-scm.com/docs/git-rebase">Rebase and rewritten history</a></li>
      <li><a href="https://git-scm.com/docs/git-revert">Revert and merge-parent caveats</a></li>
      <li><a href="https://git-scm.com/docs/git-stash">Stash options and restoration</a></li>
      <li><a href="https://git-scm.com/docs/gitignore">Ignore rules and already-tracked files</a></li>
      <li><a href="https://git-scm.com/docs/git-fetch">Fetch and remote-tracking references</a></li>
      <li><a href="https://git-scm.com/docs/git-bisect">Bisect: predicates, skipped revisions and restoring the checkout</a></li>
      <li><a href="https://docs.github.com/en/pull-requests/reference/pull-requests">GitHub: pull-request concepts</a></li>
    </Sources>
  </div>,
};
