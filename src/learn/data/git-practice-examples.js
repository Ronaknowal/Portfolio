const practiceGit=`lab=$(mktemp -d) || exit 1
git init -q -b main "$lab/repo" || exit 1
cd "$lab/repo" || exit 1
git config user.name "Practice Learner"
git config user.email "learner@example.invalid"
git config core.autocrlf false
`;

export const gitPracticeExamples={
  bisect:{language:'bash',code:practiceGit+`for value in 10 11 12 99 99 99
do
  printf '%s\\n' "$value" > value.txt
  git add -- value.txt
  # Keep all six history steps, including two with unchanged file contents.
  git commit --allow-empty -qm "Set value $value"
done
git tag known-good HEAD~5
git tag known-bad HEAD
# The test lives outside the checkout so older snapshots cannot remove it.
printf '%s\\n' 'test "$(cat value.txt)" -lt 50' > "$lab/check.sh"
git bisect start known-bad known-good > /dev/null
git bisect run bash "$lab/check.sh" > "$lab/bisect-output.txt" 2>&1
git log -1 --format=%s
cat value.txt
git bisect reset > /dev/null 2>&1
git branch --show-current`,output:`Set value 99
99
main`},
  transfer:{language:'bash',code:practiceGit+`printf 'value 1\\n' > result.txt
printf 'draft\\n' > note.txt
git add -- result.txt note.txt
git commit -qm "Start report"
git switch -qc explain-units
printf 'value 2\\n' > result.txt
printf 'units: ms\\n' > note.txt
git add -- note.txt
git commit -qm "Explain units"
printf 'committed result: '
git show HEAD:result.txt
printf 'committed note: '
git show HEAD:note.txt
printf 'working result: '
cat result.txt
git status --short`,output:`committed result: value 1
committed note: units: ms
working result: value 2
 M result.txt`},
};
