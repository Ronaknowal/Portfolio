export const investigationSetup = `lab=$(mktemp -d) || exit 1
cd "$lab" || exit 1
mkdir -p data/raw logs reports
printf 'latency_ms\\n10\\n20\\n' > "data/raw/run 1.csv"
printf 'INFO start\\nWARN slow\\nERROR timeout\\nWARN retry\\n' > logs/run.log
cd reports || exit 1`;

export const investigation = {
  language: "bash",
  code: investigationSetup + `
find ../data -type f -name '*.csv'
head -n 2 "../data/raw/run 1.csv"
grep -n -F 'WARN' ../logs/run.log > warnings.txt
cat warnings.txt
printf 'warning lines: '
wc -l < warnings.txt
if grep -q -F 'SUCCESS' ../logs/run.log; then
    printf 'success marker found\\n'
else
    status=$?
    printf 'success search status: %s\\n' "$status"
fi`,
  output: "../data/raw/run 1.csv\nlatency_ms\n10\n2:WARN slow\n4:WARN retry\nwarning lines: 2\nsuccess search status: 1",
};
