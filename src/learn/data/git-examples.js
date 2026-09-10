const gitStart = `lab=$(mktemp -d) || exit 1
git init -q -b main "$lab/repo" || exit 1
cd "$lab/repo" || exit 1
git config user.name "Practice Learner"
git config user.email "learner@example.invalid"
git config core.autocrlf false
`;

const trackedStart = gitStart + `printf 'version 1\\n' > report.txt
git add report.txt
git commit -qm "Add initial report"
`;

const git = (code, output, base = trackedStart) => ({ code: base + code, output, language: "bash" });

export const gitExamples = {
  first: git(`printf 'version 1\\n' > report.txt
git status --short
git add report.txt
git status --short
git commit -qm "Add initial report"
git log -1 --format=%s
git branch --show-current`, "?? report.txt\nA  report.txt\nAdd initial report\nmain", gitStart),
  staging: git(`printf 'version 2\\n' > report.txt
git add report.txt
printf 'version 3\\n' > report.txt
git status --short
git show HEAD:report.txt
git show :report.txt
cat report.txt
git commit -qm "Record staged version two"
git show HEAD:report.txt
git status --short`, "MM report.txt\nversion 1\nversion 2\nversion 3\nversion 2\n M report.txt"),
  unstage: git(`printf 'version 2\\n' > report.txt
git add report.txt
git restore --staged -- report.txt
git status --short
cat report.txt
# Discard only this disposable example's unstaged edit.
git restore -- report.txt
cat report.txt
git diff --exit-code
printf 'tracked files match the index\\n'`, " M report.txt\nversion 2\nversion 1\ntracked files match the index"),
  branch: git(`git switch -qc add-note
printf 'Check measurement units.\\n' > note.txt
git add note.txt
git commit -qm "Document measurement units"
git switch -q main
if test -e note.txt; then
    printf 'unexpected: note exists on main\\n'
    exit 1
else
    printf 'main has no note yet\\n'
fi
git merge --ff-only -q add-note
cat note.txt
git log --format=%s --reverse`, "main has no note yet\nCheck measurement units.\nAdd initial report\nDocument measurement units"),
  conflict: git(`git switch -qc by-region
printf 'title=Latency by region\\n' > report.txt
git add report.txt
git commit -qm "Describe regional report"
git switch -q main
printf 'title=Latency by model\\n' > report.txt
git add report.txt
git commit -qm "Describe model report"
if git merge --no-edit by-region > "$lab/merge.log" 2>&1; then
    printf 'unexpected: conflict was not raised\\n'
    exit 1
else
    printf 'merge needs a decision\\n'
fi
git status --short
git show :1:report.txt
git show :2:report.txt
git show :3:report.txt
# Decision: the document should describe both dimensions.
printf 'title=Latency by model and region\\n' > report.txt
git add report.txt
git diff --cached --check
git commit -qm "Combine model and region descriptions"
cat report.txt
git rev-list --count HEAD`, "merge needs a decision\nUU report.txt\nversion 1\ntitle=Latency by model\ntitle=Latency by region\ntitle=Latency by model and region\n4"),
  revert: git(`printf 'wrong units\\n' > report.txt
git add report.txt
git commit -qm "Change report units"
git revert --no-edit HEAD > "$lab/revert.log"
cat report.txt
git rev-list --count HEAD
git log -1 --format=%s`, 'version 1\n3\nRevert "Change report units"'),
  stash: git(`printf 'unfinished revision\\n' > report.txt
printf 'draft note\\n' > draft.txt
git stash push -qu -m "Pause report draft"
git status --short
cat report.txt
git stash list --format=%s
git stash apply -q
cat report.txt
git status --short
git stash list --format=%s`, "version 1\nOn main: Pause report draft\nunfinished revision\n M report.txt\n?? draft.txt\nOn main: Pause report draft"),
  remote: git(`git init --bare -q -b main "$lab/remote.git"
git remote add origin "$lab/remote.git"
git push -qu origin main
git clone -q "$lab/remote.git" "$lab/colleague"
git -C "$lab/colleague" config user.name "Practice Colleague"
git -C "$lab/colleague" config user.email "colleague@example.invalid"
printf 'version 2\\n' > "$lab/colleague/report.txt"
git -C "$lab/colleague" add report.txt
git -C "$lab/colleague" commit -qm "Update shared report"
git -C "$lab/colleague" push -q
git fetch -q origin
cat report.txt
git show origin/main:report.txt
git rev-list --left-right --count HEAD...origin/main
git merge --ff-only -q origin/main
cat report.txt`, "version 1\nversion 2\n0\t1\nversion 2"),
  ignore: git(`printf '*.log\\nreport.txt\\n' > .gitignore
printf 'diagnostic only\\n' > run.log
printf 'version 2\\n' > report.txt
git check-ignore run.log
git status --short
git ls-files report.txt`, "run.log\n M report.txt\n?? .gitignore\nreport.txt"),
};
