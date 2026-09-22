"""Source-hygiene audit of every file the PAC/VC lesson owns, its verifiers included.

This exists because of a specific, repeated failure in this repository: a shell
heredoc turns a backslash sequence into the byte it names. That has shipped a
broken virtual-environment path, a lost `\\gamma`, and -- worst -- a regex whose
`\\b` became a literal backspace byte INSIDE A VERIFIER, so the guard written to
catch a defect could never fire. A checker that audits only the lesson sources
and not the checkers would repeat exactly that.

Six classes of defect, each of which has shipped here before:

  1. **A shell ate an escape.** Both halves are checked: no raw C0 control byte
     survives anywhere, and every escape that remains is one the file's language
     defines.
  2. **A KaTeX sequence went missing**, silently changing what a formula says.
     The sequences this lesson depends on are named and required.
  3. **A display block was left as one long line**, which overflows a 320 px
     column. Every block must be short or explicitly wrapped.
  4. **A conditional whose branches are identical** -- the prose equivalent of
     an assertion that cannot fail. `{x.executed ? '' : ''}` shipped in another
     lesson's body: an intended phrase that got lost.
  5. **`Math` is shadowed in the lesson body.** The topic module imports KaTeX's
     `Math` component, so `Math.sqrt` there resolves to a React component and
     yields `undefined` rather than a number, silently. No global-`Math` member
     access is allowed in that file.
  6. **Borrowing a sibling lesson's dataset.** Two other lessons serve the same
     banknote extract. This one owns its copy and must never reference theirs;
     a path that works today would silently follow someone else's edit tomorrow.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-pac-sources.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/teaching/evidence/pac-sources.json"

FILES = [
    "src/learn/data/pac-models.js",
    "src/learn/data/pac-data.js",
    "src/learn/data/pac-examples.js",
    "src/learn/data/topics/pac-learning-vc-dimension.jsx",
    "src/learn/data/curriculum/blueprints/pac-learning-vc-dimension.js",
    "src/learn/components/lesson-labs/PacShared.jsx",
    "src/learn/components/lesson-labs/PacLabs.jsx",
    "src/learn/components/lesson-labs/PacFigures.jsx",
    "src/learn/components/lesson-labs/pac-labs.css",
    "scripts/verify-pac-models.mjs",
    "scripts/verify-pac-examples.py",
    "scripts/verify-pac-data.py",
    "scripts/verify-pac-browser.cjs",
    "scripts/verify-pac-sources.py",
    "scripts/falsify-pac-guards.mjs",
    "public/learn-assets/pac-learning/ATTRIBUTION.txt",
]

TOPIC = "src/learn/data/topics/pac-learning-vc-dimension.jsx"
COMPONENTS = [
    TOPIC,
    "src/learn/components/lesson-labs/PacLabs.jsx",
    "src/learn/components/lesson-labs/PacFigures.jsx",
    "src/learn/components/lesson-labs/PacShared.jsx",
]

REQUIRED_KATEX = [
    r"\\Pr", r"\\varepsilon", r"\\delta", r"\\frac", r"\\sqrt", r"\\sum", r"\\binom",
    r"\\inf_", r"\\hat R", r"\\mathbf1", r"\\le", r"\\ge", r"\\Pi_H", r"\\log_2",
    r"\\theta", r"\\pi", r"\\infty", r"\\begin{gathered}", r"\\end{gathered}",
]

# Paths that belong to other lessons serving the same extract.
FORBIDDEN_PATHS = [
    "learn-assets/evaluation-metrics",
    "learn-assets/semi-supervised-learning",
]

# These four files name the forbidden paths on purpose; see the check below.
PATH_NAMING_ALLOWED = {
    "scripts/verify-pac-data.py",
    "scripts/verify-pac-sources.py",
    "scripts/verify-pac-browser.cjs",
    "scripts/falsify-pac-guards.mjs",
    "public/learn-assets/pac-learning/ATTRIBUTION.txt",
}

# Members of the global Math object. In the topic module `Math` is the KaTeX
# component, so any of these is a silent undefined.
MATH_MEMBERS = [
    "round", "sqrt", "min", "max", "abs", "log", "log2", "log10", "exp", "floor",
    "ceil", "pow", "hypot", "sign", "trunc", "random", "E", "PI", "LN2", "LN10",
]

keep_evidence = "--no-evidence" not in sys.argv

problems: list[str] = []
checks = 0


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)


def scanned(before, label):
    """Turn a scan that reports by appending into a real, counted assertion.

    S4: six call sites used to pass the literal `True`, so 55 of 96 "checks"
    evaluated nothing -- the detection logic behind them is real, but it reports
    by appending to `problems`, which does not touch the counter. A number in a
    summary line is a claim about coverage, so each of those now evaluates
    whether its own scan found anything. The specific problem is already
    recorded by the scan, so this does not append a second time; it exists to
    make the count true.
    """
    global checks
    checks += 1
    return len(problems) == before


def main():
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
        mark = len(problems)
        # S4: the allow-list used to end in `or following.isalnum()`, which
        # admits every letter and digit, so no letter escape could ever be
        # flagged. The list is now explicit: string escapes, regex classes and
        # the LaTeX control words this lesson actually uses. A letter outside
        # both lists is now reported for a human to read.
        #
        # Escapes that survive as TEXT are legitimate here -- a regex word
        # boundary is spelled with a backslash and a b. What must never survive
        # is the BYTE a shell would leave if it ate one, and that is scan 1a's
        # job, not this one. (This comment deliberately spells no escape out;
        # the first draft of it tripped this very scan.)
        allowed = set("nrtbfv0\\'\"`$/aeux<>whsSdDwWbBAZzpPkGQE.^[]()|*+?{}-,:;=!&@#%~ \n")
        latex_words = {
            "Pr", "varepsilon", "delta", "frac", "tfrac", "sqrt", "sum", "binom", "inf", "hat", "widehat",
            "mathbf", "mathrm", "le", "ge", "ne", "in", "to", "cdots", "min", "max", "log", "ln", "exp",
            "Pi", "theta", "pi", "infty", "begin", "end", "gathered", "text", "bigl", "bigr", "Bigl",
            "Bigr", "left", "right", "quad", "approx", "times", "sim", "forall", "dots", "ldots",
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

        # 6. No borrowing of a sibling lesson's copy of the same dataset.
        #    Four files may name those paths, because naming them is their job:
        #    the two verifiers that assert the page never requests them, the
        #    harness that falsifies that assertion, and the attribution file,
        #    which documents that sibling copies exist and are not read.
        mark = len(problems)
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text and relative not in PATH_NAMING_ALLOWED:
                line = text[:text.index(forbidden)].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: references {forbidden}, which belongs to another lesson. "
                    "This lesson serves its own copy under public/learn-assets/pac-learning/.")
        scanned(mark, f"{relative}: own dataset only")

    # 2. The KaTeX sequences the lesson depends on are still there.
    katex = (ROOT / TOPIC).read_text(encoding="utf-8")
    for token in REQUIRED_KATEX:
        check(token in katex, f"the KaTeX sequence {token} is missing from the lesson body")

    # 3. Every display block is short or wrapped rather than left as one long line.
    blocks = re.findall(r"<MathBlock>\{'(.*?)'\}</MathBlock>", katex, re.S)
    check(len(blocks) >= 12, f"only {len(blocks)} display blocks were found to check")
    # Two signals, because the first one alone let a formula through that then
    # overflowed at 320 px on the real page: stripping every backslash command
    # made `\max\left\{\frac4\varepsilon\log_2\frac2\delta, ...\}` look short,
    # when each of those commands renders as a glyph. The raw length is the
    # blunter, more honest proxy. Neither measures the rendered width; that is
    # what scripts/verify-pac-browser.cjs does at 390 and 320 px.
    for block in blocks:
        visible = re.sub(r"\\[a-zA-Z]+\{?|[{}\\]", "", block)
        wrapped = "\\\\\\\\" in block or "gathered" in block
        check(wrapped or (len(visible) <= 46 and len(block) <= 70),
              f"an unwrapped display block of {len(visible)} visible and {len(block)} raw "
              f"characters: {block[:70]}")

    # 4. No conditional whose branches are the same.
    for relative in COMPONENTS:
        body = (ROOT / relative).read_text(encoding="utf-8")
        mark = len(problems)
        for match in re.finditer(r"\?\s*('[^']*'|\"[^\"]*\")\s*:\s*('[^']*'|\"[^\"]*\")", body):
            if match.group(1) == match.group(2):
                line = body[:match.start()].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: a conditional whose branches are identical ({match.group(1)}) - "
                    "it can never render anything, so the phrase it was meant to carry is missing")
        scanned(mark, f"{relative}: dead conditionals")

    # 5. `Math` is the KaTeX component in the lesson body; global members there
    #    are silent undefined.
    check("from '../../components/content/Math.jsx'" in katex,
          "the lesson body imports the KaTeX Math component, which is what shadows the global")
    # Block comments are stripped first, with their line count preserved so the
    # reported line number stays right. This file's own explanation of the
    # hazard names Math.sqrt inside a comment, and a scan that flagged the
    # explanation would be a scan nobody could keep green -- which is how a
    # guard gets deleted rather than fixed.
    katex_code = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), katex, flags=re.S)
    mark = len(problems)
    for member in MATH_MEMBERS:
        for match in re.finditer(r"\bMath\." + member + r"\b", katex_code):
            line = katex_code[:match.start()].count("\n") + 1
            problems.append(
                f"{TOPIC}:{line}: Math.{member} in a module where `Math` is the KaTeX component. "
                "It resolves to that component and yields undefined, silently. Use a helper from "
                "pac-models.js or PacShared.jsx instead.")
    scanned(mark, "no shadowed global Math member access in the lesson body")

    # 7. Every SVG this lesson renders carries one of its own layout classes.
    #    The layout rule is scoped to those classes because an unscoped
    #    `.pac-lesson svg` rule also matched KaTeX's radical SVGs and collapsed
    #    every square root on the page to nothing. An untagged SVG would fall
    #    out of the rule instead and render at its intrinsic size.
    for relative in ("src/learn/components/lesson-labs/PacShared.jsx",
                     "src/learn/components/lesson-labs/PacFigures.jsx"):
        body = (ROOT / relative).read_text(encoding="utf-8")
        mark = len(problems)
        for match in re.finditer(r"<svg\b([^>]*)>", body, re.S):
            attributes = match.group(1)
            line = body[:match.start()].count("\n") + 1
            if not re.search(r'className="[^"]*\b(pac-line|pac-plot|pac-panel)\b', attributes):
                problems.append(
                    f"{relative}:{line}: an <svg> with no pac-line, pac-plot or pac-panel class. "
                    "The layout rule is scoped to those classes, so this one would not get it.")
        scanned(mark, f"{relative}: every svg is tagged for the layout rule")
    css = (ROOT / "src/learn/components/lesson-labs/pac-labs.css").read_text(encoding="utf-8")
    check("svg:is(.pac-line, .pac-plot, .pac-panel)" in css,
          "the SVG layout rule is scoped by class and cannot reach KaTeX's radical SVGs")

    # A guard on the guards: the verifiers must contain assertions, and the
    # browser verifier must actually look for the leak this topic is exposed to.
    models = (ROOT / "scripts/verify-pac-models.mjs").read_text(encoding="utf-8")
    check(models.count("assert") >= 200, "the models verifier still carries its assertions")
    browser = (ROOT / "scripts/verify-pac-browser.cjs").read_text(encoding="utf-8")
    check("sampleCurvesThroughLabels" in browser,
          "the browser verifier carries its own curve-through-label sampler")
    check("toFixed(6)" in browser or "fixed(" in browser,
          "and pins the graded quantity numerically rather than by marker class alone")

    # S5: a floor, close to the real number. This verifier had none at all, so
    # deleting whole scans would have shrunk the headline silently.
    check(checks >= 90, f"only {checks} hygiene checks ran; the suite has lost coverage")
    check(audited == len(FILES), f"only {audited} of {len(FILES)} declared files were audited")

    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-pac-sources.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filesAudited": audited,
        "filesDeclared": len(FILES),
        "checks": checks,
        "displayBlocksChecked": len(blocks),
        "requiredKatexSequences": len(REQUIRED_KATEX),
        "fileHashes": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                       for relative in FILES if (ROOT / relative).exists()},
        "scope": "Raw control bytes and undefined backslash escapes in every file this lesson owns, its own "
                 "five verifiers and its falsification harness included; the KaTeX sequences the body depends "
                 "on; that every display block is short or wrapped; that no JSX conditional has identical "
                 "branches; that the lesson body never reaches for a member of the global Math object in a "
                 "module where `Math` is the KaTeX component; and that no owned file references the two "
                 "sibling lessons' copies of the same banknote extract.",
        "limitations": [
            "Text-level hygiene only. It does not parse JSX or KaTeX, and a wrapped block can still overflow: "
            "the rendered width is measured by scripts/verify-pac-browser.cjs.",
            "The escape allow-list is deliberately broad; it catches a byte a shell replaced, not every "
            "questionable escape.",
            "The CRLF question is deliberately not checked: this repository sets core.autocrlf=true and the "
            "convention is about stored bytes, so such a guard would test nothing.",
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

    print(f"PASS: {checks} source-hygiene checks over {audited} files - no eaten escape, no raw control byte, "
          f"every required KaTeX sequence present, all {len(blocks)} display blocks short or wrapped, "
          f"no conditional with identical branches, no shadowed Math member, no sibling-lesson asset path.")


if __name__ == "__main__":
    main()
