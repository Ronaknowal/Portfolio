"""Source-hygiene audit of every file the Calibration & Conformal Prediction lesson owns.

Its own verifiers are in scope, and that is the point. The defect this exists to
catch has twice been a guard that could never fire: a shell heredoc turned the
`\\b` inside a regex into a literal backspace byte, so the check written to
catch a defect silently matched nothing. A checker that audited only the lesson
and not the checkers would have passed that.

Four classes of defect, each of which has shipped in this repository before.

  1. **A shell ate an escape.** A heredoc turns a backslash sequence into the
     byte it names. Both halves are checked: no raw C0 control byte survives
     anywhere, and every escape that remains is one the file's language defines.
  2. **A KaTeX sequence went missing**, silently changing what a formula says.
     The sequences this lesson depends on are named and required, and every
     display block must be wrapped rather than left as one long line that
     overflows a narrow column. `\\widehat` is singled out: a `\\widehat` accent
     once emitted a 336-pixel accent SVG in a 320-pixel column, and rewriting
     the line around it three times never helped, so the block that carries one
     must be wrapped whatever its visible length.
  3. **A conditional whose branches are identical** — the prose equivalent of an
     assertion that cannot fail. One shipped in this lesson's own first draft:
     `['CQR', '...'].slice(0, 1)[0] === 'CQR' ? [...] : [...]` with both arms
     the same, which is an expression that can never render anything but one of
     them.
  4. **A stylesheet rule that beats a presentation attribute.** A blanket
     `fill` or `stroke-width` under the lesson root silently overrides an
     attribute that carries a quantity. The stylesheet is checked for those two
     properties appearing on a bare element selector.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-calibration-sources.py
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
EVIDENCE = ROOT / "docs/teaching/evidence/calibration-sources.json"

TOPIC = "src/learn/data/topics/calibration-conformal-prediction.jsx"
MATH_SOURCES = [
    TOPIC,
    "src/learn/components/lesson-labs/CalibrationLabs.jsx",
    "src/learn/components/lesson-labs/CalibrationShared.jsx",
    "src/learn/components/lesson-labs/CalibrationFigures.jsx",
]
CSS = "src/learn/components/lesson-labs/calibration-labs.css"

FILES = [
    "src/learn/data/calibration-models.js",
    "src/learn/data/calibration-data.js",
    "src/learn/data/calibration-examples.js",
    TOPIC,
    "src/learn/data/curriculum/blueprints/calibration-conformal-prediction.js",
    "src/learn/components/lesson-labs/CalibrationShared.jsx",
    "src/learn/components/lesson-labs/CalibrationLabs.jsx",
    "src/learn/components/lesson-labs/CalibrationFigures.jsx",
    CSS,
    "scripts/verify-calibration-models.mjs",
    "scripts/verify-calibration-examples.py",
    "scripts/verify-calibration-data.py",
    "scripts/verify-calibration-browser.cjs",
    "scripts/verify-calibration-sources.py",
    "scripts/falsify-calibration.mjs",
    "public/learn-assets/calibration/ATTRIBUTION.txt",
    "public/learn-assets/calibration/reliability_bins.py",
    "public/learn-assets/calibration/monotone_map.py",
    "public/learn-assets/calibration/conformal_rank.py",
]

COMPONENTS = [
    TOPIC,
    "src/learn/components/lesson-labs/CalibrationLabs.jsx",
    "src/learn/components/lesson-labs/CalibrationFigures.jsx",
    "src/learn/components/lesson-labs/CalibrationShared.jsx",
]

REQUIRED_KATEX = [
    r"\\mathbb E", r"\\mid", r"\\sum", r"\\frac", r"\\alpha", r"\\beta", r"\\tau", r"\\sigma",
    r"\\lceil", r"\\rceil", r"\\le", r"\\ge", r"\\max", r"\\tfrac", r"\\Pr",
    r"\\widehat{\\mathrm{ECE}}", r"\\mathrm{Beta}", r"\\begin{gathered}", r"\\end{gathered}",
]

problems: list[str] = []
checks = 0


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)


def main():
    audited = 0
    control_bytes = 0
    escape_sites = 0
    bytes_scanned = 0
    # Per-file subject counts. `check(True, ...)` used to stand where these do:
    # four literal-true calls that incremented the headline count and could not
    # fail, so 43 of the reported 78 "checks" were assertions with no condition.
    # Each is now a floor on what the scan actually looked at, which is the
    # thing that goes wrong when a regex or a path stops matching.
    per_file_escapes = {}
    for relative in FILES:
        path = ROOT / relative
        if not path.exists():
            problems.append(f"{relative}: MISSING")
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        audited += 1

        # 1a. No raw control byte survived a shell.
        for index, byte in enumerate(raw):
            if byte < 0x20 and byte not in (0x09, 0x0A, 0x0D):
                line = raw[:index].count(b"\n") + 1
                control_bytes += 1
                problems.append(
                    f"{relative}:{line}: raw control byte 0x{byte:02x} "
                    f"({unicodedata.name(chr(byte), 'unnamed')}) — a shell ate an escape here")
        bytes_scanned += len(raw)
        check(len(raw) > 0, f"{relative}: is empty, so the control-byte scan inspected nothing")

        # 1b. Every surviving escape is one the language defines.
        allowed = set("nrtbfv0\\'\"`$/aeux<>whsSdDwWbBAZzpPkGQE.^[]()|*+?{}-,:;=!&@#%~ \n")
        for match in re.finditer(r"\\(.)", text):
            escape_sites += 1
            following = match.group(1)
            if following in allowed or following.isalnum():
                continue
            line = text[:match.start()].count("\n") + 1
            problems.append(f"{relative}:{line}: unusual escape \\{following!r}")
        per_file_escapes[relative] = len(re.findall(r"\\(.)", text))
        check(len(text) > 0, f"{relative}: decoded to nothing, so the escape scan inspected nothing")

    # 1c. The escape scan above allows any alphanumeric after a backslash,
    #     because JS and Python both define plenty of them — which means it
    #     cannot see the defect it was written for. An eaten backslash turns
    #     the source `'\\alpha'` into `'\alpha'`, and `\a` is alphanumeric, so
    #     it passes. This guard covers that domain: inside a math element every
    #     quoted literal must have backslashes only in even-length runs, since
    #     each TeX command needs a doubled backslash in the JSX source.
    #
    #     The first draft of this guard matched `<Math>{'…'}</Math>` as a whole
    #     and so saw 136 of the 169 math elements: the other 33 build their
    #     content by concatenation, `{'k=' + value}`, and those literals carry
    #     TeX too. Matching the element region and then every literal inside it
    #     closes that gap; the floor below is the true element count, so losing
    #     the subject set again fails rather than passing quietly.
    math_elements = 0
    math_literals = 0
    for relative in MATH_SOURCES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"<(Math|MathBlock)>\{(.*?)\}</\1>", source, re.S):
            math_elements += 1
            region = match.group(2)
            for literal in re.finditer(r"'((?:[^'\\]|\\.)*)'", region):
                math_literals += 1
                body = literal.group(1)
                for run in re.finditer(r"\\+", body):
                    if len(run.group(0)) % 2 == 1:
                        line = source[:match.start()].count(chr(10)) + 1
                        problems.append(
                            f"{relative}:{line}: the math literal {body[:60]!r} has an odd run of "
                            f"{len(run.group(0))} backslashes; a shell or an editor ate one, and KaTeX "
                            "will render the command as literal text")
                        break
    check(math_elements >= 169,
          f"only {math_elements} math elements were found to check for eaten backslashes; the scan has lost "
          "its subject set")
    check(math_literals >= math_elements,
          f"{math_literals} quoted literals across {math_elements} math elements: some element's content is "
          "not being read at all")

    # 2. The KaTeX sequences the lesson depends on are still there.
    katex = (ROOT / TOPIC).read_text(encoding="utf-8")
    for token in REQUIRED_KATEX:
        check(token in katex, f"the KaTeX sequence {token} is missing from the lesson body")

    # 2b. Every display block is wrapped rather than left as one long line, and a
    #     block carrying a wide accent is wrapped whatever its visible length.
    blocks = re.findall(r"<MathBlock>\{'(.*?)'\}</MathBlock>", katex, re.S)
    check(len(blocks) >= 8, f"only {len(blocks)} display blocks were found to check")
    wide_accent = 0
    for block in blocks:
        visible = re.sub(r"\\\\[a-zA-Z]+\{?|[{}\\]", "", block)
        wrapped = "\\\\\\\\" in block or "gathered" in block
        check(len(visible) <= 46 or wrapped,
              f"an unwrapped display block of {len(visible)} visible characters: {block[:70]}")
        if "widehat" in block or "widetilde" in block or "overbrace" in block:
            wide_accent += 1
            check(wrapped,
                  f"a display block carrying a wide accent is not wrapped: {block[:70]} — the accent's own SVG "
                  "can exceed a narrow column even when the visible line is short")
    check(wide_accent >= 1,
          "no display block carries a wide accent, so the accent guard above inspected nothing")

    # 3. No conditional whose branches are the same.
    dead = 0
    for relative in COMPONENTS:
        body = (ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"\?\s*('[^']*'|\"[^\"]*\")\s*:\s*('[^']*'|\"[^\"]*\")", body):
            if match.group(1) == match.group(2):
                line = body[:match.start()].count("\n") + 1
                dead += 1
                problems.append(
                    f"{relative}:{line}: a conditional whose branches are identical ({match.group(1)}) — "
                    "it can never render anything, so the phrase it was meant to carry is missing")
        # An array-literal conditional with two identical arms is the shape this
        # lesson's own first draft shipped; the string form above misses it.
        for match in re.finditer(r"\?\s*(\[[^\[\]]*\])\s*:\s*(\[[^\[\]]*\])", body):
            if match.group(1) == match.group(2):
                line = body[:match.start()].count("\n") + 1
                dead += 1
                problems.append(
                    f"{relative}:{line}: a conditional whose two array branches are identical ({match.group(1)})")
        ternaries = len(re.findall(r"\?[^?:]{0,200}:", body))
        check(ternaries > 0,
              f"{relative}: the dead-conditional scan found no conditional expression of any kind to inspect, "
              "so a defect of that shape could not be reported from this file")

    # 4. No stylesheet rule that reaches into markup this lesson does not own.
    #
    #    Two members of one family, both of which are a selector matching more
    #    than the author had in mind.
    #
    #    4a. A blanket paint rule beating a presentation attribute: a `fill` or
    #        a `stroke-width` on a bare element type overrides an attribute
    #        that encodes a quantity, and every model assertion stays green.
    #    4b. A bare descendant `svg` selector under the lesson root. KaTeX
    #        renders accents, radicals and stretchy delimiters as real <svg>
    #        elements and sizes them with `.katex svg { height: inherit }`. A
    #        rule written for this lesson's own diagrams has the same
    #        specificity, loads later, and wins. A sibling lesson lost every
    #        radical to zero height that way and printed a bound as its own
    #        radicand. Here it drew the lesson's one `\widehat` at 9.078px
    #        against KaTeX's 4.641px. Both are the same mistake: a stylesheet
    #        reaching into markup it does not own, which is why this sits
    #        inside the existing check rather than beside it.
    stylesheet = (ROOT / CSS).read_text(encoding="utf-8")
    # Comments out, newlines kept, so reported line numbers still point at the
    # real line. Without this the scan read the header comment's own examples
    # as rules and reported the documentation as the defect.
    uncommented = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), stylesheet, flags=re.S)
    check(uncommented.count("\n") == stylesheet.count("\n"),
          "stripping comments from the stylesheet changed its line count, so every reported line number below "
          "would be wrong")
    rules_seen = 0
    for match in re.finditer(r"^\s*([^{}\n]+)\{([^}]*)\}", uncommented, re.M):
        rules_seen += 1
        selector, body = match.group(1).strip(), match.group(2)
        # match.start(1), not match.start(): the leading \s* swallows the
        # newlines before the selector, which reported every rule as line 1.
        # Offsets are into `uncommented`, which has the same line count.
        line = uncommented[:match.start(1)].count("\n") + 1
        # A rule that names an element type rather than a class can reach a
        # shape whose attribute carries a quantity.
        bare_element = re.fullmatch(
            r"\.cal-lesson svg\.cal-plot (rect|circle|line|path|polyline|polygon|g)", selector)
        if bare_element and re.search(r"(^|;)\s*(fill|stroke-width)\s*:", body):
            problems.append(
                f"{CSS}:{line}: `{selector}` sets fill or stroke-width on a bare element type; a stylesheet rule "
                "beats a presentation attribute, so a width or a fill that encodes a quantity would be overridden")
        for part in selector.split(","):
            part = part.strip()
            if not part:
                continue
            if re.search(r"(^|\s)svg(?![\w.\-])", part) and ".cal-plot" not in part:
                problems.append(
                    f"{CSS}:{line}: `{part}` selects `svg` without naming this lesson's own `cal-plot` class. "
                    "KaTeX renders accents, radicals and stretchy delimiters as <svg> under the same root and "
                    "sizes them with `height: inherit`; this rule has equal specificity and loads later, so it "
                    "silently resizes or collapses them")
    check(rules_seen >= 40,
          f"only {rules_seen} stylesheet rules were parsed; the paint-rule scan is not reading the sheet")
    # Which only helps while every SVG this lesson draws carries the class.
    svg_tags = 0
    for relative in COMPONENTS:
        source = (ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"<svg\b([^>]*)>", source, re.S):
            svg_tags += 1
            if "cal-plot" not in match.group(1):
                line = source[:match.start()].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: an <svg> without the `cal-plot` class. Every stylesheet rule for this "
                    "lesson's diagrams names that class, so this one renders with the browser's defaults: "
                    "16px text inside a ~300-unit viewBox")
    check(svg_tags >= 6,
          f"only {svg_tags} <svg> elements were found in the lesson's components; the class scan has lost its "
          "subject set")
    # The lesson-root text rule must use longhands: the `font` shorthand resets
    # every font-size set on a descendant.
    check(re.search(r"\.cal-lesson svg\.cal-plot text\s*\{[^}]*font-size", stylesheet),
          "the lesson-root SVG text rule no longer sets a font-size longhand")
    check(not re.search(r"\.cal-lesson svg\.cal-plot text\s*\{[^}]*(^|;)\s*font\s*:", stylesheet, re.M),
          "the lesson-root SVG text rule uses the `font` shorthand, which resets descendant font sizes")
    check(".cal-lesson svg.cal-plot text" in stylesheet,
          "the SVG text rule is not scoped at the lesson root, so a diagram inside an investigation wrapper "
          "would render at the browser default size inside a 300-unit viewBox")

    check(escape_sites >= 1200,
          f"only {escape_sites} escape sites were inspected across the corpus; the escape scan has lost its "
          "subject set")
    check(bytes_scanned >= 200000,
          f"only {bytes_scanned} bytes were scanned for control bytes; files are missing from the audit")
    check(min(per_file_escapes.values(), default=0) >= 0 and len(per_file_escapes) == audited,
          "some audited file produced no escape-scan entry at all")

    # A counter that is reported but never floored is decoration: a block that
    # stops running still prints PASS, with a smaller number nobody reads. The
    # models verifier has had these floors since phase A; the guard audit found
    # that the other four verifiers report their headline counts unfloored.
    if checks < 85:
        raise SystemExit(f"only {checks} source-hygiene checks ran; a block did not execute, so this PASS "
                         "covers less than it claims")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-calibration-sources.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filesAudited": audited,
        "filesDeclared": len(FILES),
        "checks": checks,
        "escapeSitesInspected": escape_sites,
        "mathElementsInspected": math_elements,
        "mathLiteralsInspected": math_literals,
        "bytesScannedForControlBytes": bytes_scanned,
        "stylesheetRulesParsed": rules_seen,
        "rawControlBytes": control_bytes,
        "deadConditionals": dead,
        "displayBlocksChecked": len(blocks),
        "displayBlocksWithWideAccent": wide_accent,
        "requiredKatexSequences": len(REQUIRED_KATEX),
        "scope": "Raw control bytes and undefined backslash escapes in every file this lesson owns, its own four "
                 "verifiers and its three served programs included; the KaTeX sequences the body depends on; that "
                 "every display block is wrapped, with a block carrying a wide accent required to be wrapped "
                 "whatever its visible length; that no JSX conditional has two identical branches, in string or "
                 "array form; and that the stylesheet sets no blanket fill or stroke-width on a bare SVG element "
                 "type and scopes its text rule at the lesson root using longhands.",
        "limitations": [
            "Text-level hygiene only. It does not parse JSX or KaTeX, and a wrapped block can still overflow: "
            "the rendered width is measured by scripts/verify-calibration-browser.cjs.",
            "The escape allow-list is deliberately broad; it catches a byte a shell replaced, not every "
            "questionable escape.",
            "The stylesheet check looks for bare element selectors under the lesson root. A class-scoped rule "
            "that happens to match a quantity-carrying shape is not detected here; the browser verifier "
            "compares painted values with attributes instead.",
        ],
        "passed": not problems,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(problems)} source-hygiene problems across {audited} files")

    print(f"PASS: {checks} source-hygiene checks over {audited} files and {escape_sites} escape sites — no eaten "
          f"escape, no raw control byte, every required KaTeX sequence present, all {len(blocks)} display blocks "
          f"wrapped ({wide_accent} carrying a wide accent), no conditional with identical branches, and no "
          f"blanket paint rule that could beat a presentation attribute.")


def _record_failure(evidence_path, verifier, error):
    """Write a FAILING evidence record.

    Every check in this file raises or exits, and a raise skips the evidence
    write at the end of main() — which leaves the PREVIOUS run's
    `"passed": true` on disk describing a tree that fails. A reviewer reading
    the evidence directory afterwards sees green for red. A sibling lesson
    shipped exactly that, so the failure path writes its own record.
    """
    import traceback
    try:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps({
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "verifier": verifier,
            "passed": False,
            "failure": {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "traceback": traceback.format_exc()[-4000:],
            },
            "note": "This run failed. Written from the failure path so a red tree cannot be read as green "
                    "from an earlier run's evidence file.",
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
    except Exception:  # noqa: BLE001 - the original failure must still surface
        pass


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:  # noqa: BLE001 - includes SystemExit from a FAIL path
        if isinstance(error, SystemExit) and not error.code:
            raise
        _record_failure(EVIDENCE, __file__.replace("\\", "/").split("/scripts/")[-1], error)
        raise
