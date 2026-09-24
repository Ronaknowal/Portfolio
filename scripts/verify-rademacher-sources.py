"""Source-hygiene audit of every file the Rademacher lesson owns.

This scans the lesson sources AND its own verifiers. Six classes of defect, each
of which has shipped in this repository before:

  1. **A shell ate an escape.** A heredoc turns a backslash sequence into the
     byte it names. This has shipped a broken venv path, a lost `\\gamma`, and a
     regex whose `\\b` became a literal backspace *inside a verifier*, so the
     guard written to catch a defect could never fire. Both halves are checked:
     no raw C0 control byte survives anywhere, and every escape that remains is
     one the file's language defines.
  2. **A KaTeX sequence went missing**, silently changing what a formula says.
     The sequences this lesson depends on are named and required, and every
     display block must be wrapped rather than left as one long line that
     overflows a narrow column.
  3. **A conditional whose branches are identical** -- the prose equivalent of
     an assertion that cannot fail. `{x.executed ? '' : ''}` shipped in a lesson
     body: an intended phrase that got lost, leaving an expression that can
     never render anything.
  4. **A backslash-u escape written in JSX TEXT.** JSX text is not a string
     literal, so `\\u201c` there renders as six visible characters rather than a
     quotation mark. This is specific to writing components with a tool that
     escapes non-ASCII by habit.
  5. **A figure caption inside its own scroll box.** Below the narrow
     breakpoint a table blockifies and a `<caption>` collapses to the width of
     its longest word. This lesson's tables must use a sibling paragraph.
  6. **An SVG text rule scoped to a figure class rather than the lesson root**,
     which misses drawings inside investigation wrappers and leaves them at the
     browser default 16px inside a ~320-unit viewBox; and a `font` shorthand,
     which silently resets every font-size set on a descendant.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-rademacher-sources.py
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
EVIDENCE = ROOT / "docs/teaching/evidence/rademacher-sources.json"

TOPIC = "src/learn/data/topics/rademacher-complexity-generalization-bounds.jsx"
CSS = "src/learn/components/lesson-labs/rademacher-labs.css"

FILES = [
    "src/learn/data/rademacher-models.js",
    "src/learn/data/rademacher-data.js",
    "src/learn/data/rademacher-examples.js",
    TOPIC,
    "src/learn/data/curriculum/blueprints/rademacher-complexity-generalization-bounds.js",
    "src/learn/components/lesson-labs/RademacherShared.jsx",
    "src/learn/components/lesson-labs/RademacherLabs.jsx",
    "src/learn/components/lesson-labs/RademacherFigures.jsx",
    CSS,
    "scripts/verify-rademacher-models.mjs",
    "scripts/verify-rademacher-examples.py",
    "scripts/verify-rademacher-data.py",
    "scripts/verify-rademacher-browser.cjs",
    "scripts/verify-rademacher-sources.py",
    "scripts/falsify-rademacher.mjs",
    "public/learn-assets/rademacher/ATTRIBUTION.txt",
]

COMPONENTS = [
    TOPIC,
    "src/learn/components/lesson-labs/RademacherLabs.jsx",
    "src/learn/components/lesson-labs/RademacherFigures.jsx",
    "src/learn/components/lesson-labs/RademacherShared.jsx",
]

# Sequences the lesson's formulas depend on. A lost backslash here changes what
# a formula says without changing whether it renders.
REQUIRED_KATEX = [
    r"\\widehat{\\mathfrak R}", r"\\mathfrak R_n", r"\\sup", r"\\sum", r"\\frac", r"\\sqrt",
    r"\\mathbb E", r"\\sigma", r"\\delta", r"\\rho", r"\\ell", r"\\varphi", r"\\phi_\\rho",
    r"\\|w\\|_2", r"\\|v\\|_\\infty", r"\\operatorname{tr}", r"\\langle", r"\\begin{cases}",
    r"\\begin{gathered}", r"\\end{gathered}", r"\\underbrace", r"\\leq", r"\\geq", r"\\neq",
    r"\\eta_t", r"\\beta", r"\\lambda", r"\\varepsilon",
]

problems: list[str] = []
checks = 0


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)


def main():
    # The falsification harness drives this verifier with a deliberate defect
    # injected; `--no-evidence` keeps that result off the record.
    no_evidence = "--no-evidence" in sys.argv
    audited = 0
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
                problems.append(
                    f"{relative}:{line}: raw control byte 0x{byte:02x} "
                    f"({unicodedata.name(chr(byte), 'unnamed')}) -- a shell ate an escape here")
        check(True, f"{relative}: control bytes")

        # 1b. Every surviving escape is one the language defines.
        allowed = set("nrtbfv0\\'\"`$/aeux<>whsSdDwWbBAZzpPkGQE.^[]()|*+?{}-,:;=!&@#%~ \n")
        for match in re.finditer(r"\\(.)", text):
            following = match.group(1)
            if following in allowed or following.isalnum():
                continue
            line = text[:match.start()].count("\n") + 1
            problems.append(f"{relative}:{line}: unusual escape \\{following!r}")
        check(True, f"{relative}: escapes")

    # 2. The KaTeX sequences the lesson depends on are still there.
    katex = (ROOT / TOPIC).read_text(encoding="utf-8")
    for token in REQUIRED_KATEX:
        check(token in katex, f"the KaTeX sequence {token} is missing from the lesson body")

    # 2b. Every display block is wrapped rather than left as one long line.
    #     A `\widehat` over a single letter is fine; a long unwrapped line is
    #     what overflows 320px, and the browser verifier measures the result.
    blocks = re.findall(r"<MathBlock>\{'(.*?)'\}</MathBlock>", katex, re.S)
    check(len(blocks) >= 14, f"only {len(blocks)} display blocks were found to check")
    for block in blocks:
        visible = re.sub(r"\\\\[a-zA-Z]+\{?|[{}\\]", "", block)
        wrapped = "\\\\\\\\" in block or "gathered" in block or "cases" in block
        check(len(visible) <= 46 or wrapped,
              f"an unwrapped display block of {len(visible)} visible characters: {block[:70]}")

    # 3. No conditional whose branches are the same.
    for relative in COMPONENTS:
        body = (ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"\?\s*('[^']*'|\"[^\"]*\")\s*:\s*('[^']*'|\"[^\"]*\")", body):
            if match.group(1) == match.group(2):
                line = body[:match.start()].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: a conditional whose branches are identical ({match.group(1)}) -- "
                    "it can never render anything, so the phrase it was meant to carry is missing")
        check(True, f"{relative}: dead conditionals")

    # 4. A backslash-u escape sitting in JSX TEXT rather than in a string.
    #
    #    An earlier version stripped string literals first and then looked at
    #    what was left. That was INERT: a single apostrophe in a prose comment
    #    made the single-quote regex swallow the whole region after it,
    #    including the line a defect was injected into, so the guard could not
    #    fire. The falsification harness is what exposed that, and it is exactly
    #    the "a guard's domain is where defects hide" failure.
    #
    #    This looks directly at what JSX text IS: the characters between a `>`
    #    and the next `<`. Attribute values live inside the angle brackets and
    #    are excluded by construction, with no string parsing at all. Block
    #    comments are blanked first (preserving line breaks, so line numbers
    #    stay true) because a comment can sit between two tags.
    for relative in COMPONENTS:
        body = (ROOT / relative).read_text(encoding="utf-8")
        body = re.sub(r"/\*.*?\*/", lambda match: re.sub(r"[^\n]", " ", match.group(0)), body, flags=re.S)
        for segment in re.finditer(r">([^<>]*)<", body, re.S):
            for escape in re.finditer(r"\\u[0-9a-fA-F]{4}", segment.group(1)):
                line = body[:segment.start(1) + escape.start()].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: {escape.group(0)} in JSX text -- JSX text is not a string literal, so "
                    "this renders as six visible characters rather than the glyph it names")
        check(True, f"{relative}: JSX-text escapes")

    # 5. No <caption> inside a table this lesson renders. Comments are stripped
    #    first: the shared table component EXPLAINS in a comment why it does not
    #    use one, and a guard that fires on its own rationale is a guard nobody
    #    will keep.
    for relative in sorted(set(COMPONENTS)):
        body = (ROOT / relative).read_text(encoding="utf-8")
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        body = re.sub(r"//[^\n]*", "", body)
        check("<caption" not in body,
              f"{relative}: a <caption> inside a table collapses to its longest word when the table blockifies; "
              "this lesson's captions are sibling paragraphs")

    # 6. The SVG text rule is scoped at the lesson root and uses longhands.
    css = (ROOT / CSS).read_text(encoding="utf-8")
    # 6a. NO bare descendant `svg` selector under the lesson root.
    #
    #     `.rad-lesson svg { height: auto }` also matches KaTeX's own inline
    #     SVGs, which take `height: inherit`. It collapsed all 24 radical signs
    #     on this page to under a pixel, so a square root simply vanished and
    #     the formula said the radius squared where it meant the radius; nine
    #     stretchy accents went with them, turning empirical complexity into the
    #     population quantity. This checker previously reported "the SVG text
    #     rule scoped at the lesson root" as PASSING while that defect was live,
    #     because its domain was the text rule and the geometry rule beside it
    #     was never examined.
    css_body = re.sub(r"/\*.*?\*/", lambda match: re.sub(r"[^\n]", " ", match.group(0)), css, flags=re.S)
    for match in re.finditer(r"\.rad-lesson\s+svg(?!\s*\.rad-drawing)(?![\w-])", css_body):
        line = css_body[:match.start()].count("\n") + 1
        problems.append(
            f"{CSS}:{line}: a bare descendant `svg` selector under the lesson root also matches KaTeX's "
            "radical and accent SVGs. Target `svg.rad-drawing`, the class the shared Drawing component applies.")
    check(True, f"{CSS}: no bare descendant svg selector")
    check(".rad-lesson svg.rad-drawing text {" in css,
          "the SVG text rule must be scoped at the lesson root AND to this lesson's own drawings, or drawings "
          "inside investigation wrappers render at the browser default size inside a ~320-unit viewBox")
    # 6c. Every SVG this lesson renders is tagged, and only through `Drawing`.
    shared = (ROOT / "src/learn/components/lesson-labs/RademacherShared.jsx").read_text(encoding="utf-8")
    check("className={`rad-drawing${className ? ` ${className}` : ''}`}" in shared,
          "the shared Drawing component must apply rad-drawing itself, so a new figure cannot forget it")
    for relative in ("src/learn/components/lesson-labs/RademacherFigures.jsx",
                     "src/learn/components/lesson-labs/RademacherLabs.jsx"):
        component = (ROOT / relative).read_text(encoding="utf-8")
        component = re.sub(r"/\*.*?\*/", " ", component, flags=re.S)
        check("<svg" not in component,
              f"{relative}: SVGs must be created through the shared Drawing component, which tags them; a raw "
              "<svg> here would escape the rad-drawing scoping")
    check("font-size:" in css and "font-family:" in css,
          "the SVG text rule must use font longhands")
    for match in re.finditer(r"^\s*font:\s*[^;]+;", css, re.M):
        # A `font` shorthand is allowed on HTML controls, never on the SVG text
        # rule, because it resets every font-size set on a descendant.
        context = css[max(0, match.start() - 200):match.start()]
        if "svg text" in context:
            problems.append("the SVG text rule uses a `font` shorthand, which resets descendant font sizes")
    check(True, "font shorthand placement")
    # And no blanket fill on shapes, which would beat a fill="none" attribute.
    for match in re.finditer(r"^\.rad-lesson svg\s*\{[^}]*\bfill\s*:", css, re.M | re.S):
        problems.append("a blanket fill on `.rad-lesson svg` would beat every fill=\"none\" presentation attribute")
    check(True, "blanket fill")

    # 6b. NO hand-written numeric tolerance may appear in the investigations.
    #
    #     The independent review found the lesson's own named defect class
    #     reappearing one field over: the numeric answer field had been fixed to
    #     accept half a unit in the last printed place, while the CATEGORICAL
    #     comparison beside it kept a 1e-12 threshold, so a learner one
    #     keystroke from a shipped preset was told their category was wrong by a
    #     verdict that printed both "different" values as 0.707107.
    #
    #     Fixing the site would leave the class open. Every graded comparison
    #     now routes through `gapOutcome`, which grades at the precision the
    #     operands are printed with, and feasibility slack is a named import —
    #     so no tolerance literal needs to exist in this file at all, and the
    #     ban can be absolute rather than an allow-list of blessed exceptions.
    #     Comments are stripped first: a guard that fires on the comment
    #     explaining it is a guard the next person deletes.
    labs = "src/learn/components/lesson-labs/RademacherLabs.jsx"
    body = (ROOT / labs).read_text(encoding="utf-8")
    body = re.sub(r"/\*.*?\*/", lambda match: re.sub(r"[^\n]", " ", match.group(0)), body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    for match in re.finditer(r"\b\d(?:\.\d+)?e-\d+\b", body):
        line = body[:match.start()].count("\n") + 1
        problems.append(
            f"{labs}:{line}: the numeric tolerance {match.group(0)} is written by hand. Graded comparisons must "
            "use gapOutcome/displayValue so the threshold is the precision the page prints; feasibility checks "
            "must use the exported feasibilitySlack.")
    check(True, f"{labs}: no hand-written tolerance")
    check("gapOutcome" in body, f"{labs}: the display-precision comparator is actually used")

    # 7. The verifiers themselves must not contain an assertion that cannot
    #    fail. A comparison of a value with itself is the canonical form.
    for relative in ("scripts/verify-rademacher-models.mjs",):
        body = (ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"close\(\s*([A-Za-z0-9_.\[\]()' ]+?)\s*,\s*\1\s*,", body):
            line = body[:match.start()].count("\n") + 1
            problems.append(f"{relative}:{line}: a value compared with itself asserts nothing")
        check(True, f"{relative}: self-comparisons")

    def build_evidence():
        return json.dumps({
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "verifier": "scripts/verify-rademacher-sources.py",
            "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "filesAudited": audited,
            "filesDeclared": len(FILES),
            "checks": checks,
            "displayBlocksChecked": len(blocks),
            "requiredKatexSequences": len(REQUIRED_KATEX),
            "scope": "Raw control bytes and undefined backslash escapes in every file this lesson owns, its own "
                     "verifiers included; the KaTeX sequences the body depends on; that every display block is "
                     "wrapped rather than left as one long line; that no JSX conditional has identical branches; "
                     "that no backslash-u escape sits in JSX text, where it would render as six visible characters; "
                     "that no table caption sits inside its own scroll box; that the SVG text rule is scoped at the "
                     "lesson root and uses font longhands with no blanket fill; and that the models verifier contains "
                     "no comparison of a value with itself.",
            "limitations": [
                "Text-level hygiene only. It does not parse JSX or KaTeX, and a wrapped block can still overflow: "
                "the rendered width is measured by scripts/verify-rademacher-browser.cjs.",
                "The escape allow-list is deliberately broad; it catches a byte a shell replaced, not every "
                "questionable escape.",
                "The self-comparison scan matches the `close(a, a, ...)` form only. A comparison laundered through a "
                "variable is not caught here; the falsification harness is what establishes that a guard can fire.",
            ],
        "passed": not problems,
        "failureNotes": problems,
    }, indent=2) + "\n"

    # `--no-evidence` is honoured here because the falsification harness drives
    # this verifier with a deliberate defect injected into a lesson source. Left
    # unguarded, a harness run ended with `rademacher-sources.json` recording
    # "passed": false, derived from a defect the harness itself put there — the
    # same class the builder had already closed for the browser artefacts,
    # surviving in the one writer it did not cover.
    if not no_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(build_evidence(), encoding="utf-8", newline="\n")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(problems)} source-hygiene problems across {audited} files")

    print(f"PASS: {checks} source-hygiene checks over {audited} files -- no eaten escape, no raw control byte, "
          f"no backslash-u in JSX text, every required KaTeX sequence present, all {len(blocks)} display blocks "
          f"wrapped, no conditional with identical branches, no caption inside a scroll box, and the SVG text rule "
          f"scoped at the lesson root.")


if __name__ == "__main__":
    main()
