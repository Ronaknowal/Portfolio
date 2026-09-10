export const bashExamples = {
  arguments: {title:'See the actual arguments',language:'bash',code:`#!/usr/bin/env bash
set -u
export LC_ALL=C
work=$(mktemp -d) || exit 1
trap 'rm -rf -- "$work"' EXIT
cd -- "$work" || exit 1
touch -- a.csv 'run alpha.csv'
show_args() {
    printf 'count=%s\\n' "$#"
    for arg in "$@"; do printf '<%s>\\n' "$arg"; done
}
value='run alpha.csv'
show_args "$value"
show_args $value  # Deliberate bug: compare argument boundaries.
value='*.csv'
show_args "$value"
show_args $value  # Deliberate filename expansion.
value=''
show_args "$value"
show_args $value`,expected:'counts: 1, 2, 1, 2, 1, 0; quoted *.csv is literal; unquoted matches a.csv and run alpha.csv.'},
  scope: {title:'Pass values through functions and processes',language:'bash',code:`#!/usr/bin/env bash
set -u
label=parent
describe() {
    local label=function
    printf 'function=%s; arguments=%s\\n' "$label" "$#"
    for item in "$@"; do printf 'item=<%s>\\n' "$item"; done
}
args=('run alpha.csv' '')
describe "\${args[@]}"
export label
bash -c 'label=child; printf "child=%s\\n" "$label"'
printf 'parent=%s\\n' "$label"
captured=$(printf 'two lines\\n\\n')
printf 'captured=<%s>\\n' "$captured"`,expected:'function=function; arguments=2, two distinct items including empty; child=child; parent=parent; captured=<two lines>.'},
  statuses: {title:'Keep exit status separate from output',language:'bash',code:`#!/usr/bin/env bash
producer() { printf 'partial row\\n'; return 4; }
producer | cat > /dev/null
printf 'default=%s\\n' "$?"
set -o pipefail
producer | cat > /dev/null
printf 'pipefail=%s\\n' "$?"
producer | cat > /dev/null
parts=("\${PIPESTATUS[@]}")
printf 'individual=%s,%s\\n' "\${parts[0]}" "\${parts[1]}"
if printf 'alpha\\n' | grep -q 'beta'; then
    printf 'found\\n'
else
    status=$?
    case "$status" in
        1) printf 'valid search: no match\\n' ;;
        *) printf 'search failed: %s\\n' "$status" >&2; exit "$status" ;;
    esac
fi`,expected:'default=0; pipefail=4; individual=4,0; valid search: no match.'},
  errexit: {title:'Why explicit failure policy matters',language:'bash',code:`#!/usr/bin/env bash
set -e
probe() {
    false
    printf 'continued inside conditional function\\n'
}
if probe; then
    printf 'function reported success from its last command\\n'
fi
if false; then
    printf 'unreachable\\n'
else
    printf 'an expected false condition is handled here\\n'
fi`,expected:'Three lines: continued inside conditional function; function reported success from its last command; an expected false condition is handled here.'},
  files: {title:'Visit filenames without parsing a listing',language:'bash',code:`#!/usr/bin/env bash
set -u
export LC_ALL=C
work=$(mktemp -d) || exit 1
trap 'rm -rf -- "$work"' EXIT
cd -- "$work" || exit 1
touch -- 'run alpha.csv' 'beta.csv'
shopt -s nullglob
files=(./*.csv)
printf 'matched=%s\\n' "\${#files[@]}"
for path in "\${files[@]}"; do
    printf 'path=<%s>\\n' "$path"
done
missing=(./*.json)
printf 'missing=%s\\n' "\${#missing[@]}"
count=0
printf 'a\\nb\\n' | while IFS= read -r line; do count=$((count + 1)); done
printf 'parent count after pipeline=%s\\n' "$count"
while IFS= read -r line; do count=$((count + 1)); done <<'ROWS'
a
b
ROWS
printf 'parent count after redirected loop=%s\\n' "$count"`,expected:'matched=2; ./beta.csv and ./run alpha.csv remain distinct arguments; missing=0; parent count after pipeline=0 (default Bash lastpipe disabled); redirected loop count=2.'},
};

export const reportFiles = {
 'report.py': `import csv
import json
import math
import sys

def summarize(path):
    values = []
    with open(path, newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["value"]:
            raise ValueError("expected one column named value")
        for number, row in enumerate(rows, start=2):
            if None in row:
                raise ValueError(f"extra columns at record {number}")
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite value at record {number}")
            values.append(value)
    return {"count": len(values), "mean": math.fsum(values) / len(values) if values else None}

if __name__ == "__main__":
    try:
        result = summarize(sys.argv[1])
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, TypeError, OverflowError, csv.Error) as error:
        print(f"report failed: {error}", file=sys.stderr)
        raise SystemExit(4)
`,
 'run-report.sh': `#!/usr/bin/env bash
set -u
set -o pipefail
if (( $# != 2 )); then
    printf 'usage: bash run-report.sh INPUT.csv OUTPUT.json\\n' >&2
    exit 2
fi
input=$1
output=$2
if [[ ! -f "$input" || -d "$output" ]]; then
    printf 'input must be a file; output must not be a directory\\n' >&2
    exit 2
fi
outdir=$(dirname -- "$output") || exit 1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
mkdir -p -- "$outdir" || exit 1
work=$(mktemp -d "$outdir/.report.XXXXXX") || exit 1
cleanup() { rm -rf -- "$work"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if python3 "$script_dir/report.py" "$input" > "$work/result.json"; then
    mv -f -- "$work/result.json" "$output" || exit 1
    printf 'published: %s\\n' "$output" >&2
else
    status=$?
    printf 'not published; worker status=%s\\n' "$status" >&2
    exit "$status"
fi
`,
 'run alpha.csv':'value\n2\n4\n6\n',
 'empty.csv':'value\n',
 'invalid.csv':'value\n2\nNaN\n',
};

export const summaryPractice = `#!/usr/bin/env bash
# Usage: bash collect-reports.sh OUTPUT.json INPUT.json...
set -u
if (( $# < 2 )); then printf 'need output and at least one input\\n' >&2; exit 2; fi
output=$1
shift
if [[ -d "$output" ]]; then printf 'output is a directory\\n' >&2; exit 2; fi
outdir=$(dirname -- "$output") || exit 1
mkdir -p -- "$outdir" || exit 1
work=$(mktemp -d "$outdir/.collection.XXXXXX") || exit 1
trap 'rm -rf -- "$work"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if python3 - "$@" > "$work/summary.json" <<'PY'
import json
import math
import sys
try:
    results = []
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        count, mean = report["count"], report["mean"]
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        if count == 0:
            if mean is not None: raise ValueError("empty report must have null mean")
        elif type(mean) not in (int, float) or not math.isfinite(mean):
            raise ValueError("nonempty mean must be finite")
        results.append({"source": path, "count": count, "mean": mean})
    print(json.dumps(results, sort_keys=True, allow_nan=False))
except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
    print(f"collection failed: {error}", file=sys.stderr)
    raise SystemExit(4)
PY
then
    mv -f -- "$work/summary.json" "$output" || exit 1
    printf 'collection published\\n' >&2
else
    status=$?
    exit "$status"
fi
`;
