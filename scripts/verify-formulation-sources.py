"""Source-hygiene audit of every file the problem-formulation lesson owns, its verifiers included.

This exists because of a specific, repeated failure in this repository: a shell
heredoc turns a backslash sequence into the byte it names. That has shipped a
broken virtual-environment path, a lost LaTeX control word, and -- worst -- a
regex whose word-boundary escape became a literal backspace byte INSIDE A
VERIFIER, so the guard written to catch a defect could never fire. A checker
that audits only the lesson sources and not the checkers would repeat exactly
that, so the five verifiers and the falsification harness are in the list below.

Eight classes of defect, each of which has shipped here before:

  1. **A shell ate an escape.** Both halves are checked: no raw C0 control byte
     survives anywhere, and every escape that remains is one the file's language
     defines.
  2. **A KaTeX sequence went missing**, silently changing what a formula says.
     The sequences this lesson depends on are named and required.
  3. **A display block was left as one long line**, which overflows a 320 px
     column. Every block must be short or explicitly wrapped.
  4. **A conditional whose branches are identical** -- the prose equivalent of
     an assertion that cannot fail.
  5. **`Math` is shadowed in the lesson body.** The topic module imports KaTeX's
     `Math` component, so `Math.round` there resolves to a React component and
     yields `undefined` rather than a number, silently.
  6. **A bare descendant `svg` selector.** `.formulation-lesson svg { height:
     auto }` also matches KaTeX's radical SVGs, whose height comes from
     `height: inherit`; `auto` leaves them with no intrinsic height and every
     square root on the page collapses. A collapsed radical is a DIFFERENT
     FORMULA, not a blemish, and no DOM check can see it.
  7. **An untagged SVG.** Every drawing must go through `Drawing`, which applies
     the class the layout rule is scoped to. A figure that wrote its own
     `<svg>` element would fall out of the rule.
  8. **Borrowing another lesson's asset directory.** This lesson serves its own
     copy of the dataset and must reference no other lesson's.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-formulation-sources.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/teaching/evidence/formulation-sources.json"

FILES = [
    "src/learn/data/formulation-models.js",
    "src/learn/data/formulation-data.js",
    "src/learn/data/formulation-examples.js",
    "src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx",
    "src/learn/data/curriculum/blueprints/ml-problem-formulation-baselines-data-leakage.js",
    "src/learn/components/lesson-labs/FormulationShared.jsx",
    "src/learn/components/lesson-labs/FormulationLabs.jsx",
    "src/learn/components/lesson-labs/FormulationFigures.jsx",
    "src/learn/components/lesson-labs/formulation-labs.css",
    "scripts/verify-formulation-models.mjs",
    "scripts/verify-formulation-data.py",
    "scripts/verify-formulation-examples.py",
    "scripts/verify-formulation-browser.cjs",
    "scripts/verify-formulation-sources.py",
    "scripts/falsify-formulation.mjs",
    "public/learn-assets/problem-formulation/ATTRIBUTION.txt",
]

# Everything this lesson owns that a parser can read. The four JSX components
# and the topic body go through the repository's own esbuild; the verifiers and
# the harness go through `node --check`. The CSS and the attribution text have
# no parser here and are covered by the byte and regex scans only.
ESBUILD = Path(__file__).resolve().parents[1] / "node_modules" / ".bin" / (
    "esbuild.cmd" if os.name == "nt" else "esbuild")
PARSEABLE = [
    "src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx",
    "src/learn/components/lesson-labs/FormulationShared.jsx",
    "src/learn/components/lesson-labs/FormulationLabs.jsx",
    "src/learn/components/lesson-labs/FormulationFigures.jsx",
    "src/learn/data/formulation-models.js",
    "src/learn/data/formulation-data.js",
    "src/learn/data/formulation-examples.js",
    "src/learn/data/curriculum/blueprints/ml-problem-formulation-baselines-data-leakage.js",
    "scripts/verify-formulation-models.mjs",
    "scripts/verify-formulation-browser.cjs",
    "scripts/falsify-formulation.mjs",
]

TOPIC = "src/learn/data/topics/ml-problem-formulation-baselines-data-leakage.jsx"
CSS = "src/learn/components/lesson-labs/formulation-labs.css"
COMPONENTS = [
    TOPIC,
    "src/learn/components/lesson-labs/FormulationLabs.jsx",
    "src/learn/components/lesson-labs/FormulationFigures.jsx",
    "src/learn/components/lesson-labs/FormulationShared.jsx",
]
DRAWERS = [
    "src/learn/components/lesson-labs/FormulationLabs.jsx",
    "src/learn/components/lesson-labs/FormulationFigures.jsx",
]
SHARED = "src/learn/components/lesson-labs/FormulationShared.jsx"

REQUIRED_KATEX = [
    r"\\frac", r"\\tfrac", r"\\sum", r"\\bar y", r"\\rm FP", r"\\rm FN", r"\\tau",
    r"\\bigl", r"\\bigr", r"\\begin{gathered}", r"\\end{gathered}", r"\\mathrm",
]

# Other lessons' asset directories. This lesson serves its own copy of its data
# and must never reference theirs; a path that works today would silently follow
# someone else's edit tomorrow.
FOREIGN_ASSET = re.compile(r"learn-assets/(?!problem-formulation)[a-z0-9-]+")

# Files that name a foreign asset path on purpose: the two verifiers that assert
# the page never requests one, and the harness that falsifies that assertion.
PATH_NAMING_ALLOWED = {
    "scripts/verify-formulation-sources.py",
    "scripts/verify-formulation-browser.cjs",
    "scripts/falsify-formulation.mjs",
}

# Members of the global Math object. In the topic module `Math` is the KaTeX
# component, so any of these is a silent undefined.
MATH_MEMBERS = [
    "round", "sqrt", "min", "max", "abs", "log", "log2", "log10", "exp", "floor",
    "ceil", "pow", "hypot", "sign", "trunc", "random", "E", "PI", "LN2", "LN10",
]

keep_evidence = "--no-evidence" not in sys.argv

problems: list[str] = []
checks = 0     # failable assertions
scans = 0      # scan passes, which report by appending to `problems`


def write_provisional():
    """A record stamped `passed: false`, written before the first check runs.

    Writing evidence only at the end looks safe and is not: a run that fails
    leaves the PREVIOUS file on disk, still saying `passed: true`, describing a
    source version that no longer exists. Anyone reading the directory then
    sees a green record for a red tree.
    """
    if not keep_evidence:
        return
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-formulation-sources.py",
        "status": "running",
        "note": "Provisional record written before the first check. If this is what is on disk, the run did "
                "not reach its end: it raised, or it was killed.",
        "passed": False,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)


def scanned(before, label, subjects=None):
    """Record that one scan pass ran, and return whether it was clean.

    A scan reports by APPENDING to `problems`; it does not evaluate a
    condition. Counting each pass as a "check" and printing one combined total
    overstated what the suite asserts: of an advertised 107, only 49 were
    failable `check()` calls and 58 were scan passes, and this function's
    return value was discarded at every call site. So the two are counted
    separately now and both are printed. A scan pass can still fail the run --
    through the problems it appends -- but it is not an assertion and is no
    longer reported as one.

    `subjects`, where a scan's subject set must not be empty, is asserted as a
    real check: a pattern that matches nothing otherwise increments a counter
    while looking at nothing, which is the shape of every inert guard this
    effort has found.
    """
    global scans
    scans += 1
    if subjects is not None:
        check(subjects > 0, f"{label}: the scan had no subjects, so it asserted nothing")
    return len(problems) == before


def strip_comments(text, kind):
    """Remove comments while preserving line counts, so a reported line number
    stays right. A file's own explanation of a hazard names the hazard, and a
    scan that flagged the explanation is a scan nobody can keep green."""
    def blank(match):
        return "\n" * match.group(0).count("\n")
    if kind in ("js", "css"):
        text = re.sub(r"/\*[\s\S]*?\*/", blank, text)
    if kind == "js":
        text = re.sub(r"(?m)^\s*//.*$", "", text)
    if kind == "py":
        text = re.sub(r'(?s)"""[\s\S]*?"""', blank, text)
        text = re.sub(r"(?m)^\s*#.*$", "", text)
    return text


def identical_ternary_branches(body):
    """Every `? A : B` in `body` whose two branches are the same expression.

    A balanced scan, not a literal match. The first version of this required
    both branches to be QUOTED STRING LITERALS, so the one genuine instance in
    this lesson -- `Number.isInteger(value) ? sign(String(value)) :
    sign(String(value))` in FormulationShared.jsx -- was invisible to it, in a
    file this very scan reads. Worse, the falsification case injected a
    string-literal instance, so the guard collected a green tick on a shape it
    could not catch in real code. A guard's domain has to be the construct, not
    one spelling of it.

    Branch extent is found by walking forward at bracket depth zero: the
    consequent ends at `:`, the alternate at the first `,`, `;` or newline.
    Quotes and template literals are skipped, so a colon inside a string cannot
    end a branch. A nested ternary makes the extent ambiguous, so those sites
    are left unreported rather than reported wrongly -- under-reporting is a
    known limit, guessing would be a wrong finding.
    """
    findings = []
    openers = "([{"
    closers = ")]}"

    def scan(start, stops):
        depth = 0
        index = start
        while index < len(body):
            char = body[index]
            if char in "\"'`":
                quote = char
                index += 1
                while index < len(body) and body[index] != quote:
                    index += 2 if body[index] == "\\" else 1
                index += 1
                continue
            if char in openers:
                depth += 1
            elif char in closers:
                if depth == 0:
                    return index
                depth -= 1
            elif depth == 0 and char in stops:
                return index
            index += 1
        return -1

    for match in re.finditer(re.escape("?"), body):
        at = match.start()
        # Optional chaining and nullish coalescing are not conditionals.
        if body[at:at + 2] in ("?.", "??") or (at and body[at - 1] == "?"):
            continue
        mid = scan(at + 1, ":")
        if mid < 0 or body[mid] != ":":
            continue
        end = scan(mid + 1, ",;" + chr(10))
        if end < 0:
            continue
        consequent = body[at + 1:mid]
        alternate = body[mid + 1:end]
        if "?" in consequent or "?" in alternate:
            continue
        left = " ".join(consequent.split())
        right = " ".join(alternate.split())
        if left and left == right:
            findings.append((body[:at].count(chr(10)) + 1, left))
    return findings


def main():
    write_provisional()
    audited = 0
    blocks: list[str] = []
    for relative in FILES:
        path = ROOT / relative
        if not path.exists():
            problems.append(f"{relative}: MISSING")
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        audited += 1

        # 1a. No raw control byte survived a shell.
        mark = len(problems)
        for index, byte in enumerate(raw):
            if byte < 0x20 and byte not in (0x09, 0x0A, 0x0D):
                line = raw[:index].count(b"\n") + 1
                problems.append(
                    f"{relative}:{line}: raw control byte 0x{byte:02x} "
                    f"({unicodedata.name(chr(byte), 'unnamed')}) - a shell ate an escape here")
        scanned(mark, f"{relative}: control bytes", subjects=len(raw))

        # 1b. Every surviving escape is one the language defines.
        mark = len(problems)
        # The allow-list is explicit rather than ending in `isalnum()`, which
        # would admit every letter and make no letter escape reportable.
        # Escapes that survive as TEXT are legitimate; what must never survive
        # is the BYTE a shell would leave if it ate one, and that is scan 1a's
        # job. (This comment deliberately spells no escape out.)
        allowed = set("nrtbfv0\\'\"`$/aeux<>whsSdDwWbBAZzpPkGQE.^[]()|*+?{}-,:;=!&@#%~ \n")
        latex_words = {
            "frac", "tfrac", "sum", "bar", "rm", "tau", "bigl", "bigr", "begin", "end",
            "gathered", "mathrm", "text", "le", "ge", "ne", "cdot", "quad", "approx",
            "left", "right", "sqrt", "varepsilon", "delta", "theta", "infty", "log", "ln",
        }
        for match in re.finditer(r"\\(.)", text):
            following = match.group(1)
            if following in allowed or following.isdigit():
                continue
            word = re.match(r"[A-Za-z]+", text[match.start() + 1:])
            if word and word.group(0) in latex_words:
                continue
            line = text[:match.start()].count("\n") + 1
            problems.append(f"{relative}:{line}: unusual escape \\{following!r}")
        scanned(mark, f"{relative}: escapes")

        # 8. No borrowing of another lesson's asset directory.
        mark = len(problems)
        if relative not in PATH_NAMING_ALLOWED:
            for match in FOREIGN_ASSET.finditer(text):
                line = text[:match.start()].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: references {match.group(0)}, which belongs to another lesson. "
                    "This lesson serves its own copy under public/learn-assets/problem-formulation/.")
        scanned(mark, f"{relative}: own assets only")

    # 2. The KaTeX sequences the lesson depends on are still there.
    katex = (ROOT / TOPIC).read_text(encoding="utf-8")
    for token in REQUIRED_KATEX:
        check(token in katex, f"the KaTeX sequence {token} is missing from the lesson body")

    # 3. Every display block is short or wrapped rather than left as one long line.
    blocks = re.findall(r"<MathBlock>\{'(.*?)'\}</MathBlock>", katex, re.S)
    blocks += [match.replace("'\n      + '", "")
               for match in re.findall(r"<MathBlock>\{'(.*?)'\}</MathBlock>", katex, re.S)]
    blocks = re.findall(r"<MathBlock>\{('(?:[^']|\\')*'(?:\s*\+\s*'(?:[^']|\\')*')*)\}</MathBlock>", katex, re.S)
    blocks = [re.sub(r"'\s*\+\s*'", "", block).strip("'") for block in blocks]
    check(len(blocks) >= 5, f"only {len(blocks)} display blocks were found to check")
    # Two signals, because either alone lets a formula through. Stripping every
    # backslash command makes a formula of nothing but commands look short; the
    # raw length is the blunter, more honest proxy. Neither measures the
    # rendered width, which is what verify-formulation-browser.cjs does at 390
    # and 320 px.
    for block in blocks:
        visible = re.sub(r"\\[a-zA-Z]+\{?|[{}\\]", "", block)
        wrapped = "\\\\\\\\" in block or "gathered" in block
        check(wrapped or (len(visible) <= 46 and len(block) <= 70),
              f"an unwrapped display block of {len(visible)} visible and {len(block)} raw "
              f"characters: {block[:70]}")

    # 4. No conditional whose branches are the same.
    for relative in COMPONENTS:
        body = strip_comments((ROOT / relative).read_text(encoding="utf-8"), "js")
        mark = len(problems)
        for line, branch in identical_ternary_branches(body):
            problems.append(
                f"{relative}:{line}: a conditional whose branches are identical ({branch[:60]}) - "
                "the test decides nothing, so either a branch is missing or the conditional is")
        scanned(mark, f"{relative}: dead conditionals")

    # 5. `Math` is the KaTeX component in the lesson body.
    check("from '../../components/content/Math.jsx'" in katex,
          "the lesson body imports the KaTeX Math component, which is what shadows the global")
    katex_code = strip_comments(katex, "js")
    mark = len(problems)
    for member in MATH_MEMBERS:
        for match in re.finditer(r"\bMath\." + member + r"\b", katex_code):
            line = katex_code[:match.start()].count("\n") + 1
            problems.append(
                f"{TOPIC}:{line}: Math.{member} in a module where `Math` is the KaTeX component. "
                "It resolves to that component and yields undefined, silently. Use a helper from "
                "formulation-models.js or FormulationShared.jsx instead.")
    scanned(mark, "no shadowed global Math member access in the lesson body")

    # 6. The layout rule is scoped by class and cannot reach KaTeX's radicals.
    css = (ROOT / CSS).read_text(encoding="utf-8")
    rules = strip_comments(css, "css")
    mark = len(problems)
    for match in re.finditer(r"(?m)(^|[\s,>+~])svg\s*(\{|,)", rules):
        line = rules[:match.start()].count("\n") + 1
        problems.append(
            f"{CSS}:{line}: a bare svg selector. It also matches KaTeX's radical SVGs, whose height comes "
            "from `height: inherit`; a height rule there collapses every square root on the page, which "
            "changes what the formulas say and leaves the DOM correct.")
    scanned(mark, "no bare svg selector")
    check("svg.form-svg" in rules, "the layout rule is scoped to this lesson's own SVG class")
    check(re.search(r"[^.]svg\.form-svg\s*\{", rules) is not None,
          "and attaches that class to the svg element rather than to a container")

    # 7. Every SVG comes from the shared Drawing component. Comments are
    #    stripped first: this wrapper's own explanation of the rule quotes the
    #    element it forbids elsewhere.
    shared_raw = (ROOT / SHARED).read_text(encoding="utf-8")
    shared = strip_comments(shared_raw, "js")
    mark = len(problems)
    for match in re.finditer(r"<svg\b([^>]*)", shared, re.S):
        line = shared[:match.start()].count("\n") + 1
        if 'className={`form-svg' not in match.group(1) and 'className="form-svg' not in match.group(1):
            problems.append(f"{SHARED}:{line}: the shared drawing wrapper writes an <svg> without the "
                            "form-svg class the layout rule is scoped to.")
    scanned(mark, "the shared drawing wrapper tags its svg", subjects=len(re.findall(r"<svg", shared)))
    found_svgs = len(re.findall(r"<svg\b", shared))
    check(found_svgs == 1, f"the shared wrapper contains exactly one <svg> element, found {found_svgs}")
    for relative in DRAWERS:
        body = strip_comments((ROOT / relative).read_text(encoding="utf-8"), "js")
        mark = len(problems)
        for match in re.finditer(r"<svg\b", body):
            line = body[:match.start()].count("\n") + 1
            problems.append(
                f"{relative}:{line}: an <svg> element written outside the shared Drawing wrapper. "
                "A figure that writes its own svg can omit the class the layout rule is scoped to; use "
                "Drawing from FormulationShared.jsx instead.")
        scanned(mark, f"{relative}: no hand-written svg", subjects=len(re.findall(r"<Drawing", body)))
        check("<Drawing" in body, f"{relative} draws through the shared wrapper")

    # A guard on the guards: the verifiers must contain assertions, and the
    # browser verifier must actually look for the leak this topic is exposed to.
    models = (ROOT / "scripts/verify-formulation-models.mjs").read_text(encoding="utf-8")
    check(models.count("assert") >= 150, "the models verifier still carries its assertions")
    browser = (ROOT / "scripts/verify-formulation-browser.cjs").read_text(encoding="utf-8")
    check("sampleCurvesThroughLabels" in browser,
          "the browser verifier carries its own curve-through-label sampler")
    check("toFixed(6)" in browser,
          "and pins the graded quantity numerically rather than by marker class alone")
    check(".katex svg" in browser and "getBoundingClientRect" in browser,
          "and measures KaTeX's own drawn SVGs on the rendered page, which is the only way a collapsed "
          "radical is visible at all")
    harness = (ROOT / "scripts/falsify-formulation.mjs").read_text(encoding="utf-8")
    check("SIDECAR" in harness and "LOCK" in harness,
          "the falsification harness keeps a sidecar and a lock, so a kill cannot leave a breakage applied")

    # The six-decimal convention this topic's leak pin depends on: `fixed` is
    # the graded form and nothing else may produce six decimals.
    labs = (ROOT / "src/learn/components/lesson-labs/FormulationLabs.jsx").read_text(encoding="utf-8")
    check("probabilityText" in labs,
          "the ranked scores are printed through probabilityText, not through the six-decimal graded form")
    check("toFixed(4)" in shared,
          "and probabilityText really is four decimals, so it cannot collide with the graded pin")
    check(shared.count("toFixed(digits)") >= 1, "while `fixed` prints every decimal place it is asked for")

    # The two generated modules must still say they are generated, and must not
    # have been hand-edited into something their generator would not produce.
    for relative, generator in (
            ("src/learn/data/formulation-data.js", "verify-formulation-data.py"),
            ("src/learn/data/formulation-examples.js", "verify-formulation-examples.py")):
        body = (ROOT / relative).read_text(encoding="utf-8")
        check(body.startswith(f"/* GENERATED by scripts/{generator}"),
              f"{relative} still declares which script generates it")
        check(body.rstrip().endswith("export default " + Path(relative).stem.replace("-", "_")
                                     .replace("formulation_data", "formulationData")
                                     .replace("formulation_examples", "formulationExamples") + ";"),
              f"{relative} still ends with its default export")

    # Every verifier honours --no-evidence, so an independent reviewer can rerun
    # one without writing to the record they are reviewing, and writes only to
    # this lesson's own evidence files.
    for relative in ("scripts/verify-formulation-models.mjs", "scripts/verify-formulation-data.py",
                     "scripts/verify-formulation-examples.py", "scripts/verify-formulation-sources.py"):
        body = (ROOT / relative).read_text(encoding="utf-8")
        check("--no-evidence" in body, f"{relative} accepts --no-evidence")
        check("docs/teaching/evidence/formulation-" in body,
              f"{relative} writes only to this lesson's own evidence file")

    # The lesson body imports nothing from another topic's modules.
    mark = len(problems)
    for match in re.finditer(r"from '(\.\./[^']+)'", katex):
        target = match.group(1)
        if "formulation" in target or "/components/" in target:
            continue
        line = katex[:match.start()].count("\n") + 1
        problems.append(f"{TOPIC}:{line}: imports {target}, which is not one of this lesson's own modules")
    scanned(mark, "the lesson body imports only its own data and shared components",
            subjects=len(re.findall(r"from '", katex)))

    # Accessibility affordances the labs promise.
    check('role="status"' in labs, "the investigations announce their results politely")
    check("prefers-reduced-motion" in css, "the stylesheet honours a reduced-motion preference")
    check("aria-labelledby" in shared_raw and "aria-describedby" in shared_raw,
          "and every drawing carries a title and a description")

    # 9. Every owned source actually PARSES.
    #
    #    Nothing in this lesson's verifier set used to parse the JSX it owns:
    #    byte scans, escape regexes and comment-stripped matching all pass a
    #    stray brace, an unclosed tag or an unescaped apostrophe inside a
    #    single-quoted string. That is not hypothetical -- a one-character JSX
    #    error in a sibling's file broke `vite build` repo-wide and was
    #    invisible to every offline verifier here, which is why this lesson's
    #    falsification record sat understated until the sibling's fix landed.
    #    A parse failure in a file this checker owns must fail this checker.
    parsed = 0
    for relative in PARSEABLE:
        path = ROOT / relative
        if not path.exists():
            continue
        if path.suffix in (".mjs", ".cjs"):
            command = ["node", "--check", str(path)]
        else:
            command = [str(ESBUILD), str(path), f"--loader:{path.suffix}={'jsx' if path.suffix == '.jsx' else 'js'}",
                       "--outfile=" + os.devnull, "--log-level=error"]
        result = subprocess.run(command, capture_output=True, text=True)
        check(result.returncode == 0,
              f"{relative}: does not parse - {(result.stderr or result.stdout).strip()[:300]}")
        if result.returncode == 0:
            parsed += 1
    check(parsed == len(PARSEABLE), f"only {parsed} of {len(PARSEABLE)} owned sources were parsed")

    # Floors, close to the real numbers.
    check(checks >= 60, f"only {checks} failable assertions ran; the suite has lost coverage")
    check(scans >= 55, f"only {scans} scan passes ran; the suite has lost coverage")
    check(audited == len(FILES), f"only {audited} of {len(FILES)} declared files were audited")

    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-formulation-sources.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filesAudited": audited,
        "filesDeclared": len(FILES),
        "failableAssertions": checks,
        "scanPasses": scans,
        "countingNote": "Assertions and scan passes are counted separately. A scan reports by appending to the problem list rather than by evaluating a condition, so it can fail the run but is not an assertion; reporting one combined total overstated what the suite asserts.",
        "sourcesParsed": len(PARSEABLE),
        "displayBlocksChecked": len(blocks),
        "requiredKatexSequences": len(REQUIRED_KATEX),
        "fileHashes": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                       for relative in FILES if (ROOT / relative).exists()},
        "scope": "Raw control bytes and undefined backslash escapes in every file this lesson owns, its four "
                 "verifiers and its falsification harness included; the KaTeX sequences the body depends on; "
                 "that every display block is short or wrapped; that no JSX conditional has identical "
                 "branches; that the lesson body never reaches for a member of the global Math object in a "
                 "module where `Math` is the KaTeX component; that the stylesheet contains no bare svg "
                 "selector and scopes its layout rule to this lesson's own class; that no figure or lab "
                 "writes an svg element of its own, so the class cannot be forgotten; that the six-decimal "
                 "graded form is not used for model scores; and that no owned file references another "
                 "lesson's asset directory.",
        "limitations": [
            "Text-level hygiene only. It does not parse JSX or KaTeX, and a wrapped block can still overflow: "
            "the rendered width is measured by scripts/verify-formulation-browser.cjs.",
            "The escape allow-list is deliberately broad; it catches a byte a shell replaced, not every "
            "questionable escape.",
            "Comments are stripped before the selector, dead-conditional and shadowed-Math scans, because "
            "each of those hazards is explained in a comment in the file it guards. A defect written inside "
            "a comment is therefore not reported, which is the right trade: a comment renders nothing.",
        ],
        "passed": not problems,
    }, indent=2) + "\n"
    if keep_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(evidence, encoding="utf-8", newline="\n")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(problems)} source-hygiene problems across {audited} files")

    print(f"PASS: {checks} failable assertions and {scans} scan passes over {audited} files; {len(PARSEABLE)} sources parsed - no eaten escape, no raw control byte, "
          f"every required KaTeX sequence present, all {len(blocks)} display blocks short or wrapped, "
          f"no conditional with identical branches, no shadowed Math member, no bare svg selector, no "
          f"hand-written svg outside the shared wrapper, no other lesson's asset path.")


if __name__ == "__main__":
    main()
