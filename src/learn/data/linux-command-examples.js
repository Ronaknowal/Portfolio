const linux = (code, output) => ({ code: `lab=$(mktemp -d) || exit 1
cd "$lab" || exit 1
` + code, output, language: "bash" });

export const linuxExamples = {
  paths: linux(`mkdir -p data/raw outputs
printf 'latency_ms\\n10\\n20\\n' > "data/raw/run 1.csv"
printf 'internal note\\n' > .note
ls -A1
cd data/raw || exit 1
pwd | sed "s|$lab|LAB|"
head -n 2 "run 1.csv"
cd ../.. || exit 1
find data -type f -name '*.csv'`, ".note\ndata\noutputs\nLAB/data/raw\nlatency_ms\n10\ndata/raw/run 1.csv"),
  files: linux(`printf 'original\\n' > source.txt
cp -- source.txt copy.txt
printf 'changed copy\\n' > copy.txt
cat source.txt
mv -- copy.txt renamed.txt
cat renamed.txt
# Remove only the named disposable copy, never the source.
rm -- renamed.txt
test -f source.txt && printf 'source retained\\n'
test ! -e renamed.txt && printf 'copy removed\\n'`, "original\nchanged copy\nsource retained\ncopy removed"),
  search: linux(`mkdir logs
printf 'INFO start\\nWARN slow\\nERROR timeout\\nWARN retry\\n' > logs/run.log
grep -n -F 'WARN' logs/run.log
grep -F 'WARN' logs/run.log | wc -l
if grep -q -F 'SUCCESS' logs/run.log; then
    printf 'success marker present\\n'
else
    status=$?
    printf 'grep status: %s\\n' "$status"
fi
find logs -type f -name '*.log'`, "2:WARN slow\n4:WARN retry\n2\ngrep status: 1\nlogs/run.log"),
  streams: linux(`bash --noprofile --norc -c 'printf "metric=18\\n"; printf "warning: tiny sample\\n" >&2' > result.txt 2> errors.txt
cat result.txt
cat errors.txt
printf 'second run\\n' >> result.txt
wc -l < result.txt
bash --noprofile --norc -c 'printf "out\\n"; printf "err\\n" >&2' > combined.txt 2>&1
cat combined.txt`, "metric=18\nwarning: tiny sample\n2\nout\nerr"),
  permissions: linux(`printf 'measurement\\n' > data.txt
chmod 640 data.txt
stat -c '%a %A' data.txt
mkdir private
printf 'inside\\n' > private/value.txt
chmod 600 private
if cat private/value.txt > "$lab/check.out" 2> "$lab/check.err"; then
    printf 'unexpected: traversal succeeded\\n'
    exit 1
else
    printf 'directory traversal denied\\n'
fi
chmod 700 private
cat private/value.txt
(umask 077; touch new.txt; mkdir new-dir)
stat -c '%a %n' new.txt new-dir`, "640 -rw-r-----\ndirectory traversal denied\ninside\n600 new.txt\n700 new-dir"),
  links: linux(`printf 'record\\n' > record.txt
ln record.txt hard.txt
ln -s record.txt shortcut.txt
test record.txt -ef hard.txt && printf 'hard links share one file\\n'
readlink shortcut.txt
mv -- record.txt renamed.txt
cat hard.txt
if test -L shortcut.txt && test ! -e shortcut.txt; then
    printf 'symbolic link is now dangling\\n'
fi`, "hard links share one file\nrecord.txt\nrecord\nsymbolic link is now dangling"),
  environment: linux(`unset LESSON_MODE
LESSON_MODE=local
bash --noprofile --norc -c 'printf "child before export: %s\\n" "\${LESSON_MODE-unset}"'
export LESSON_MODE
bash --noprofile --norc -c 'printf "child after export: %s\\n" "$LESSON_MODE"; LESSON_MODE=child'
printf 'parent still: %s\\n' "$LESSON_MODE"
LESSON_MODE=once bash --noprofile --norc -c 'printf "one command: %s\\n" "$LESSON_MODE"'
printf 'parent afterwards: %s\\n' "$LESSON_MODE"`, "child before export: unset\nchild after export: local\nparent still: local\none command: once\nparent afterwards: local"),
  process: linux(`sleep 60 &
pid=$!
# This PID belongs to the sleep started immediately above.
ps -p "$pid" -o comm=
kill -TERM "$pid"
if wait "$pid" 2>/dev/null; then
    printf 'child completed normally\\n'
else
    status=$?
    printf 'wait status: %s\\n' "$status"
fi
printf 'owned child has been reaped\\n'`, "sleep\nwait status: 143\nowned child has been reaped"),
};
