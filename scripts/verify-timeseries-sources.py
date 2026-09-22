"""Source-hygiene audit of every file the forecasting lesson owns, its verifiers included.

This exists because of a specific, repeated failure in this repository: a shell
heredoc turns a backslash sequence into the byte it names. That has shipped a
broken virtual-environment path, a lost LaTeX command, and -- worst -- a regex
whose word-boundary escape became a literal control byte INSIDE A VERIFIER, so
the guard written to catch a defect could never fire. A checker that audits only
the lesson sources and not the checkers would repeat exactly that.

Eight classes of defect, each of which has shipped here before:

  1. **A shell ate an escape.** Both halves are checked: no raw C0 control byte
     survives anywhere, and every escape that remains is one the file's language
     defines.
  2. **A bare `svg` selector.** `.lesson svg { height: auto }` also matches
     KaTeX's own radical SVGs, whose height comes from `height: inherit`; `auto`
     leaves them no intrinsic height and every square root on the page collapses
     to nothing. In one lesson that turned a radius into the radius SQUARED --
     a different quantity -- with correct DOM, zero katex-error nodes and four
     green offline verifiers. Every `svg` token in this lesson's stylesheet must
     be immediately qualified by `.ts-diagram`.
  3. **An untagged SVG element.** The scoping in (2) only helps if every drawn
     SVG carries the class, so only ONE file is allowed to open an svg tag --
     the shared component that applies the class itself -- and its tag must
     carry it. A figure therefore cannot omit it.
  4. **A KaTeX sequence went missing**, silently changing what a formula says.
     The sequences this lesson depends on are named and required.
  5. **A display block was left as one long line**, which overflows a 320 px
     column. Every block must be short or explicitly wrapped.
  6. **A conditional whose branches are identical** -- the prose equivalent of
     an assertion that cannot fail.
  7. **`Math` is shadowed in the lesson body.** The topic module imports KaTeX's
     `Math` component, so `Math.round` there resolves to a React component and
     yields `undefined` rather than a number, silently. No global-`Math` member
     access is allowed in that file.
  8. **Borrowing a sibling lesson's dataset or component family.** This lesson
     owns its copy of the daily CSV and its own `ts-` component family; a path
     into another lesson's directory would work today and silently follow
     someone else's edit tomorrow.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-timeseries-sources.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = "node"
EVIDENCE = ROOT / "docs/teaching/evidence/timeseries-sources.json"
NEWLINE = chr(10)

FILES = [
    "src/learn/data/timeseries-models.js",
    "src/learn/data/timeseries-data.js",
    "src/learn/data/timeseries-examples.js",
    "src/learn/data/topics/time-series-validation-forecasting-baselines.jsx",
    "src/learn/data/curriculum/blueprints/time-series-validation-forecasting-baselines.js",
    "src/learn/components/lesson-labs/TimeSeriesShared.jsx",
    "src/learn/components/lesson-labs/TimeSeriesLabs.jsx",
    "src/learn/components/lesson-labs/TimeSeriesFigures.jsx",
    "src/learn/components/lesson-labs/timeseries-labs.css",
    "scripts/verify-timeseries-models.mjs",
    "scripts/verify-timeseries-examples.py",
    "scripts/verify-timeseries-data.py",
    "scripts/verify-timeseries-browser.cjs",
    "scripts/verify-timeseries-render.cjs",
    "scripts/verify-timeseries-sources.py",
    "scripts/falsify-timeseries.mjs",
    "public/learn-assets/time-series-validation/ATTRIBUTION.txt",
]

TOPIC = "src/learn/data/topics/time-series-validation-forecasting-baselines.jsx"
CSS = "src/learn/components/lesson-labs/timeseries-labs.css"
SHARED = "src/learn/components/lesson-labs/TimeSeriesShared.jsx"
FIGURES = "src/learn/components/lesson-labs/TimeSeriesFigures.jsx"
LABS = "src/learn/components/lesson-labs/TimeSeriesLabs.jsx"
COMPONENTS = [TOPIC, LABS, FIGURES, SHARED]

REQUIRED_KATEX = [
    r"\\hat y", r"\\mid", r"\\frac", r"\\sqrt", r"\\sum", r"\\operatorname", r"\\bmod",
    r"\\le", r"\\ge", r"\\text", r"\\begin{gathered}", r"\\end{gathered}",
    r"\\mathcal T", r"\\sigma", r"\\mu", r"\\log", r"\\exp", r"\\approx", r"\\ldots",
]

# Directories belonging to other lessons. This lesson serves its own copy of the
# daily CSV and owns its own ts- component family.
FORBIDDEN_PATHS = [
    "learn-assets/pac-learning",
    "learn-assets/evaluation-metrics",
    "learn-assets/regularization",
    "learn-assets/semi-supervised-learning",
    "pac-models.js",
    "PacShared.jsx",
    "bias-variance-models.js",
]

# These files name the forbidden paths on purpose; see the check below.
PATH_NAMING_ALLOWED = {
    "scripts/verify-timeseries-sources.py",
    "scripts/verify-timeseries-browser.cjs",
    "scripts/falsify-timeseries.mjs",
}

# Members of the global Math object. In the topic module `Math` is the KaTeX
# component, so any of these is a silent undefined.
MATH_MEMBERS = [
    "round", "sqrt", "min", "max", "abs", "log", "log2", "log10", "exp", "floor",
    "ceil", "pow", "hypot", "sign", "trunc", "random", "E", "PI", "LN2", "LN10",
]

# The only layout classes the stylesheet is scoped to. Applied by the shared
# Diagram component and nowhere else.
DIAGRAM_CLASS = "ts-diagram"

keep_evidence = "--no-evidence" not in sys.argv

problems: list[str] = []
checks = 0


def strip_comments(text):
    """Blank out comments while preserving line numbers.

    A scan for an opening svg tag that reads comments would flag this file's own
    explanation of why such a tag is forbidden, and the figure file's too -- a
    guard nobody can keep green, which is how a guard gets deleted rather than
    fixed. A double slash is only treated as a comment when it does not follow a
    colon, so a URL scheme survives.
    """
    without_blocks = re.sub(r"/\*.*?\*/", lambda match: NEWLINE * match.group(0).count(NEWLINE),
                            text, flags=re.S)
    return re.sub(r"(?<!:)//[^" + NEWLINE + r"]*", "", without_blocks)


def parse_check(paths):
    """Every owned JS and JSX file must actually parse.

    A JSX syntax error is invisible to every other check in this suite: the
    model and data verifiers import plain JS, the scans above read text, and
    nothing parses the component files. It surfaces only in a production build
    or a browser run.

    This is not hypothetical. During phase C a sibling lesson's unescaped
    apostrophe inside a single-quoted string broke the shared `vite build` for
    every lesson at once, and no offline verifier in any of the three lessons
    saw it. The check is cheap, it cannot go inert -- a file that does not parse
    cannot be made to look as though it does -- and it has now fired in reality.

    The blueprint is included deliberately. Nothing imports it at runtime, so a
    syntax error there would reach the integration owner rather than this suite.
    """
    targets = [name for name in paths if name.endswith((".js", ".jsx", ".mjs", ".cjs"))]
    if not targets:
        problems.append("the parse check found no JS or JSX files to parse, so it asserted nothing")
        return 0
    script = (
        "const {transformSync}=require('esbuild');const fs=require('fs');const bad=[];"
        "for(const f of process.argv.slice(1)){try{transformSync(fs.readFileSync(f,'utf8'),"
        "{loader:f.endsWith('.jsx')?'jsx':'js',sourcefile:f});}"
        "catch(e){bad.push(f+': '+((e.errors&&e.errors[0]&&e.errors[0].text)||e.message));}}"
        "if(bad.length){console.log(bad.join('\\n'));process.exit(1);}"
    )
    result = subprocess.run([NODE, "-e", script, *targets], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        for line in (result.stdout or result.stderr).strip().split(NEWLINE):
            if line.strip():
                problems.append(f"does not parse: {line.strip()}")
    return len(targets)


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)
    return bool(condition)


def scanned(before, label):
    """Turn a scan that reports by appending into a real, counted assertion.

    A scan reports by appending to `problems`, which does not touch the counter,
    so a headline count that included those scans would be a claim about
    coverage that nothing established. Each one now evaluates whether its own
    scan found anything. The specific problem is already recorded by the scan,
    so this does not append a second time; it exists to make the count true.
    """
    global checks
    checks += 1
    return len(problems) == before


def write_evidence(payload):
    if not keep_evidence:
        return
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + NEWLINE, encoding="utf-8", newline=NEWLINE)


def main():
    started = datetime.now(timezone.utc).isoformat()
    write_evidence({
        "checkedAt": started, "verifier": "scripts/verify-timeseries-sources.py",
        "status": "in progress: this record is provisional and is rewritten only after the final assertion",
        "passed": False,
    })

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
        scanned(mark, f"{relative}: control bytes")

        # 1b. Every surviving escape is one the language defines.
        #
        # Escapes that survive as TEXT are legitimate here -- a regex word
        # boundary is spelled with a backslash and a letter. What must never
        # survive is the BYTE a shell would leave if it ate one, and that is
        # scan 1a's job, not this one. The allow-list is explicit rather than
        # ending in a catch-all for letters, because a catch-all admits every
        # letter escape and can then never flag one.
        mark = len(problems)
        allowed = set("nrtbfv0\\'\"`$/aeux<>whsSdDwWbBAZzpPkGQE.^[]()|*+?{}-,:;=!&@#%~ " + NEWLINE)
        latex_words = {
            "hat", "mid", "frac", "tfrac", "sqrt", "sum", "operatorname", "bmod", "le", "ge", "ne",
            "text", "begin", "end", "gathered", "mathcal", "sigma", "mu", "log", "exp", "approx",
            "ldots", "cdots", "dots", "quad", "qquad", "bigl", "bigr", "left", "right", "times",
            "sim", "in", "to", "min", "max", "ln", "pi", "theta", "delta", "varepsilon", "infty",
        }
        for match in re.finditer(r"\\(.)", text):
            following = match.group(1)
            if following in allowed or following.isdigit():
                continue
            word = re.match(r"[A-Za-z]+", text[match.start() + 1:])
            if word and word.group(0) in latex_words:
                continue
            line = text[:match.start()].count(NEWLINE) + 1
            problems.append(f"{relative}:{line}: unusual escape backslash-{following!r}")
        scanned(mark, f"{relative}: escapes")

        # 8. No borrowing of a sibling lesson's dataset or component family.
        mark = len(problems)
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text and relative not in PATH_NAMING_ALLOWED:
                line = text[:text.index(forbidden)].count(NEWLINE) + 1
                problems.append(
                    f"{relative}:{line}: references {forbidden}, which belongs to another lesson. "
                    "This lesson serves its own copy under public/learn-assets/time-series-validation/ "
                    "and owns its own ts- component family.")
        scanned(mark, f"{relative}: own assets only")

    topic = (ROOT / TOPIC).read_text(encoding="utf-8")

    # 4. The KaTeX sequences the lesson depends on are still there.
    for token in REQUIRED_KATEX:
        check(token in topic, f"the KaTeX sequence {token} is missing from the lesson body")

    # 5. Every display block is short or wrapped rather than left as one long line.
    blocks = re.findall(r"<MathBlock>\{'(.*?)'\}</MathBlock>", topic, re.S)
    check(len(blocks) >= 4, f"only {len(blocks)} display blocks were found to check")
    # Two signals, because the first alone let a formula through that then
    # overflowed at 320 px on the real page: stripping every backslash command
    # made a long expression look short, when each of those commands renders as
    # a glyph. The raw length is the blunter, more honest proxy. Neither
    # measures the rendered width; that is what the browser verifier does at
    # 390 and 320 px.
    for block in blocks:
        visible = re.sub(r"\\[a-zA-Z]+\{?|[{}\\]", "", block)
        wrapped = ("\\\\" + "\\\\") in block or "gathered" in block
        check(wrapped or (len(visible) <= 46 and len(block) <= 70),
              f"an unwrapped display block of {len(visible)} visible and {len(block)} raw "
              f"characters: {block[:70]}")

    # 6. No conditional whose branches are the same.
    for relative in COMPONENTS:
        body = (ROOT / relative).read_text(encoding="utf-8")
        mark = len(problems)
        for match in re.finditer(r"\?\s*('[^']*'|\"[^\"]*\")\s*:\s*('[^']*'|\"[^\"]*\")", body):
            if match.group(1) == match.group(2):
                line = body[:match.start()].count(NEWLINE) + 1
                problems.append(
                    f"{relative}:{line}: a conditional whose branches are identical ({match.group(1)}) - "
                    "it can never render anything, so the phrase it was meant to carry is missing")
        scanned(mark, f"{relative}: dead conditionals")

    # 7. `Math` is the KaTeX component in the lesson body; global members there
    #    are silent undefined.
    check("from '../../components/content/Math.jsx'" in topic,
          "the lesson body imports the KaTeX Math component, which is what shadows the global")
    # Block comments are stripped first, with their line count preserved so the
    # reported line number stays right. This file's own explanation of the
    # hazard names Math.round inside a comment, and a scan that flagged the
    # explanation would be a scan nobody could keep green -- which is how a
    # guard gets deleted rather than fixed.
    topic_code = strip_comments(topic)
    mark = len(problems)
    for member in MATH_MEMBERS:
        for match in re.finditer(r"\bMath\." + member + r"\b", topic_code):
            line = topic_code[:match.start()].count(NEWLINE) + 1
            problems.append(
                f"{TOPIC}:{line}: Math.{member} in a module where `Math` is the KaTeX component. "
                "It resolves to that component and yields undefined, silently. Use a helper from "
                "timeseries-models.js or TimeSeriesShared.jsx instead.")
    scanned(mark, "no shadowed global Math member access in the lesson body")

    # 3. ONE file may open an svg tag, and its tag must carry the layout class.
    #    This is stricter than requiring every svg to be tagged: a new figure
    #    cannot omit a class it never writes.
    shared = strip_comments((ROOT / SHARED).read_text(encoding="utf-8"))
    for relative in (FIGURES, LABS, TOPIC):
        body = strip_comments((ROOT / relative).read_text(encoding="utf-8"))
        mark = len(problems)
        for match in re.finditer(r"<svg\b", body):
            line = body[:match.start()].count(NEWLINE) + 1
            problems.append(
                f"{relative}:{line}: opens an <svg> tag. Only {SHARED} may do that, because it applies the "
                f"{DIAGRAM_CLASS} class the stylesheet is scoped to. Use the shared Diagram component.")
        scanned(mark, f"{relative}: opens no svg tag of its own")
    svg_tags = re.findall(r"<svg\b[^>]*", shared, re.S)
    check(len(svg_tags) == 1,
          f"{SHARED} opens {len(svg_tags)} svg tags; exactly one shared wrapper is expected")
    mark = len(problems)
    for tag in svg_tags:
        if DIAGRAM_CLASS not in tag:
            problems.append(f"{SHARED}: the shared svg wrapper does not apply the {DIAGRAM_CLASS} class, "
                            "so nothing on this page would receive the scoped layout rule")
    scanned(mark, f"{SHARED}: the shared wrapper applies the layout class")

    # 2. NO BARE `svg` SELECTOR. Every occurrence of the svg type selector in
    #    this lesson's stylesheet must be immediately qualified by the layout
    #    class, so no rule here can reach KaTeX's radical SVGs.
    css = (ROOT / CSS).read_text(encoding="utf-8")
    css_without_comments = strip_comments(css)
    mark = len(problems)
    for match in re.finditer(r"(?<![\w.#-])svg(?![\w-])", css_without_comments):
        following = css_without_comments[match.end():match.end() + len(DIAGRAM_CLASS) + 1]
        if not following.startswith("." + DIAGRAM_CLASS):
            line = css_without_comments[:match.start()].count(NEWLINE) + 1
            context = css_without_comments[match.start():match.start() + 70].split(NEWLINE)[0]
            problems.append(
                f"{CSS}:{line}: a bare `svg` selector: {context!r}. It also matches KaTeX's own radical "
                f"SVGs and would collapse every square root on the page. Write `svg.{DIAGRAM_CLASS}`.")
    scanned(mark, f"{CSS}: no bare svg selector")
    # O3: the comment-stripped copy, as the bare-selector scan above already
    # uses. Against the raw text a commented-out rule would satisfy both.
    check(f"svg.{DIAGRAM_CLASS}" in css_without_comments,
          "the stylesheet carries at least one correctly scoped svg rule")
    # And the rule must actually set the property that caused the defect, or
    # scoping it correctly protects nothing.
    check(re.search(r"svg\.ts-diagram\s*\{[^}]*height:\s*auto", css_without_comments, re.S) is not None,
          "the scoped layout rule does not set height: auto, so it is not the rule this guard is about")

    # A guard on the guards: the verifiers must contain assertions, and the
    # browser verifier must actually look for the defects this topic is exposed
    # to.
    models = (ROOT / "scripts/verify-timeseries-models.mjs").read_text(encoding="utf-8")
    # O4: actual assertion CALLS, not the substring "assert", which the word
    # "assertion" in a comment also satisfies.
    model_assertions = len(re.findall(r"\bassert(?:\.\w+)?\(", models))
    check(model_assertions >= 150,
          f"the models verifier carries only {model_assertions} assertion calls")
    check("informationSetAudit" in models,
          "the models verifier still asserts the no-future-observation invariant")
    check("eligibleBySets" in models,
          "and still checks eligibility by an independent availability-set route")
    browser = (ROOT / "scripts/verify-timeseries-browser.cjs").read_text(encoding="utf-8")
    check("sampleCurvesThroughLabels" in browser,
          "the browser verifier carries its own curve-through-label sampler")
    check("polyline" in browser,
          "and that sampler queries the shape this lesson actually draws")
    check("toFixed(6)" in browser or "fixedText" in browser,
          "and pins the graded quantity numerically rather than by marker class alone")
    check(".katex" in browser and "sqrt" in browser,
          "and measures the KaTeX radicals the scoped CSS rule exists to protect")
    data_verifier = (ROOT / "scripts/verify-timeseries-data.py").read_text(encoding="utf-8")
    check("--write" in data_verifier and "--no-evidence" in data_verifier,
          "the data verifier is read-only by default and can skip its evidence write")
    # The render verifier is the only check that executes the React tree. A
    # sibling lesson did not render at all while three offline verifiers stayed
    # green, so this one must actually call the body rather than parse it.
    render = (ROOT / "scripts/verify-timeseries-render.cjs").read_text(encoding="utf-8")
    check("renderToStaticMarkup" in render and "lesson.content()" in render,
          "the render verifier does not actually execute the lesson body")
    check("ts-verdict" in render and "ts-reveal" in render,
          "and does not check the first-paint investigation contract")
    # O2: this file is one of the five subjects, and both search literals appear
    # in this loop's own source -- so for itself the test was satisfied by the
    # check rather than by the behaviour, and deleting the behaviour would not
    # have made it fire. The loop's own region is excised before searching self,
    # between the two sentinels below, so the substrings must be found somewhere
    # that actually implements them.
    # --- self-exclusion region start
    for name in ("scripts/verify-timeseries-models.mjs", "scripts/verify-timeseries-examples.py",
                 "scripts/verify-timeseries-data.py", "scripts/verify-timeseries-sources.py",
                 "scripts/verify-timeseries-render.cjs"):
        body = (ROOT / name).read_text(encoding="utf-8")
        if name == "scripts/verify-timeseries-sources.py":
            start = body.find("# --- self-exclusion region start")
            end = body.find("# --- self-exclusion region end")
            if start == -1 or end == -1:
                problems.append("the self-exclusion sentinels are missing; this check cannot verify itself")
                continue
            body = body[:start] + body[end:]
        check("--no-evidence" in body,
              f"{name} has no --no-evidence switch, so re-running it overwrites the record it documents")
        check('"passed": False' in body or "passed: false" in body,
              f"{name} does not write a provisional failing record before it starts")
    # --- self-exclusion region end

    # Every owned JS and JSX file must parse. See parse_check above.
    mark = len(problems)
    parsed = parse_check(FILES)
    scanned(mark, "every owned JS and JSX file parses")
    # O4: EXACT, derived from the declared list rather than a number typed
    # here. A floor set at today's count can only notice a shrink; this
    # notices a file added to FILES and never parsed, too.
    expected_parsed = len([name for name in FILES if name.endswith((".js", ".jsx", ".mjs", ".cjs"))])
    check(parsed == expected_parsed,
          f"{parsed} files were parsed but {expected_parsed} of the declared files are JS or JSX")

    # A floor, close to the real number. Without one, deleting whole scans would
    # shrink the headline silently.
    # O4: read into a variable first. `check(checks >= 90, ...)` evaluates its
    # condition before `check` increments, so it silently compared a count one
    # short of the total it printed.
    ran = checks
    check(ran >= 100, f"only {ran} hygiene checks ran before this floor; the suite has lost coverage")
    check(audited == len(FILES), f"only {audited} of {len(FILES)} declared files were audited")

    write_evidence({
        "checkedAt": started,
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-timeseries-sources.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filesAudited": audited,
        "filesDeclared": len(FILES),
        "checks": checks,
        "filesParsed": parsed,
        "displayBlocksChecked": len(blocks),
        "requiredKatexSequences": len(REQUIRED_KATEX),
        "fileHashes": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                       for relative in FILES if (ROOT / relative).exists()},
        "svgScoping": {
            "layoutClass": DIAGRAM_CLASS,
            "filesAllowedToOpenAnSvgTag": [SHARED],
            "svgTagsInSharedWrapper": len(svg_tags),
            "rule": "Every `svg` type selector in the stylesheet must be immediately qualified by the layout "
                    "class, and only the shared wrapper may open an svg tag. Together these mean a new figure "
                    "cannot omit the class and no rule here can reach KaTeX's radical SVGs.",
        },
        "scope": "Raw control bytes and undefined backslash escapes in every file this lesson owns, its own "
                 "five verifiers and its falsification harness included; the KaTeX sequences the body depends "
                 "on; that every display block is short or wrapped; that no JSX conditional has identical "
                 "branches; that the lesson body never reaches for a member of the global Math object in a "
                 "module where `Math` is the KaTeX component; that no bare `svg` selector exists and no file "
                 "but the shared wrapper opens an svg tag; and that no owned file references another lesson's "
                 "dataset directory or component family.",
        "limitations": [
            "Text-level hygiene only. It does not parse JSX, CSS or KaTeX, and a wrapped block can still "
            "overflow: the rendered width is measured by scripts/verify-timeseries-browser.cjs.",
            "The escape allow-list is deliberately broad; it catches a byte a shell replaced, not every "
            "questionable escape.",
            "The bare-selector scan reads the stylesheet as text. A rule injected from another file, or an "
            "inline style attribute, is outside its reach; the browser verifier measures the radicals "
            "themselves, which is the property that actually matters.",
            "The CRLF question is deliberately not checked: this repository sets core.autocrlf=true and the "
            "convention is about stored bytes, so such a guard would test nothing.",
        ],
        "passed": not problems,
    })

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(problems)} source-hygiene problems across {audited} files")

    print(f"PASS: {checks} source-hygiene checks over {audited} files - no eaten escape, no raw control byte, "
          f"every required KaTeX sequence present, all {len(blocks)} display blocks short or wrapped, "
          f"no conditional with identical branches, no shadowed Math member, no bare svg selector, "
          f"exactly one file opening an svg tag and it applies the layout class, no sibling-lesson path, "
          f"and all {parsed} JS/JSX files parse.")


if __name__ == "__main__":
    main()
