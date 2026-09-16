"""Source-hygiene audit of every file the Bayesian-networks lesson owns.

This ran as a session-local tool during phases A and C and was reported in the
design record as a passing verifier. It was not in the repository, so four of
its claims could not be re-run by anyone else and one of its guards was counted
in a falsification total nobody could reproduce. It is committed here so the
record is true; the independent review's S1 is the reason it exists as a file.

Three classes of defect, each of which has shipped in this repository before:

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
     an assertion that cannot fail. `{x.executed ? '' : ''}` shipped in the
     lesson body: an intended phrase that got lost, leaving an expression that
     can never render anything.

One guard was deliberately dropped. An earlier version refused CRLF line
endings in the working tree. That was a misreading: this repository sets
`core.autocrlf=true`, every long-tracked sibling file is equally CRLF in the
working tree, and the convention is about stored bytes. The guard tested
nothing and would have failed on files that are correct.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bayesnet-escapes.py
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
EVIDENCE = ROOT / "docs/teaching/evidence/bayesnet-sources.json"

FILES = [
    "src/learn/data/bayesnet-models.js",
    "src/learn/data/bayesnet-data.js",
    "src/learn/data/bayesnet-examples.js",
    "src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx",
    "src/learn/data/curriculum/blueprints/bayesian-networks-causal-graphical-models.js",
    "src/learn/components/lesson-labs/BayesNetShared.jsx",
    "src/learn/components/lesson-labs/BayesNetLabs.jsx",
    "src/learn/components/lesson-labs/BayesNetFigures.jsx",
    "src/learn/components/lesson-labs/bayesnet-labs.css",
    "scripts/verify-bayesnet-models.mjs",
    "scripts/verify-bayesnet-examples.py",
    "scripts/verify-bayesnet-data.py",
    "scripts/verify-bayesnet-browser.cjs",
    "scripts/verify-bayesnet-escapes.py",
    "public/learn-assets/bayesian-networks/ATTRIBUTION.txt",
]

TOPIC = "src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx"
COMPONENTS = [
    TOPIC,
    "src/learn/components/lesson-labs/BayesNetLabs.jsx",
    "src/learn/components/lesson-labs/BayesNetFigures.jsx",
    "src/learn/components/lesson-labs/BayesNetShared.jsx",
]

REQUIRED_KATEX = [
    r"\\mid", r"\\sum", r"\\frac", r"\\to", r"\\leftarrow", r"\\alpha", r"\\theta",
    r"\\hat I", r"\\perp", r"\\bar X", r"\\underline Z",
    r"\\begin{gathered}", r"\\end{gathered}", r"\\mathbin{\\mathrm{XOR}}",
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
                    f"({unicodedata.name(chr(byte), 'unnamed')}) — a shell ate an escape here")
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
    blocks = re.findall(r"<MathBlock>\{'(.*?)'\}</MathBlock>", katex, re.S)
    check(len(blocks) >= 10, f"only {len(blocks)} display blocks were found to check")
    for block in blocks:
        visible = re.sub(r"\\\\[a-zA-Z]+\{?|[{}\\]", "", block)
        wrapped = "\\\\\\\\" in block or "gathered" in block
        check(len(visible) <= 46 or wrapped,
              f"an unwrapped display block of {len(visible)} visible characters: {block[:70]}")

    # 3. No conditional whose branches are the same.
    for relative in COMPONENTS:
        body = (ROOT / relative).read_text(encoding="utf-8")
        for match in re.finditer(r"\?\s*('[^']*'|\"[^\"]*\")\s*:\s*('[^']*'|\"[^\"]*\")", body):
            if match.group(1) == match.group(2):
                line = body[:match.start()].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: a conditional whose branches are identical ({match.group(1)}) — "
                    "it can never render anything, so the phrase it was meant to carry is missing")
        check(True, f"{relative}: dead conditionals")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-bayesnet-escapes.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filesAudited": audited,
        "filesDeclared": len(FILES),
        "checks": checks,
        "displayBlocksChecked": len(blocks),
        "requiredKatexSequences": len(REQUIRED_KATEX),
        "scope": "Raw control bytes and undefined backslash escapes in every file this lesson owns, including "
                 "its own verifiers; the KaTeX sequences the body depends on; that every display block is "
                 "wrapped rather than left as one long line; and that no JSX conditional has identical "
                 "branches. The CRLF guard an earlier version carried was dropped: this repository sets "
                 "core.autocrlf=true and the convention is about stored bytes, so it tested nothing.",
        "limitations": [
            "Text-level hygiene only. It does not parse JSX or KaTeX, and a wrapped block can still overflow: "
            "the rendered width is measured by scripts/verify-bayesnet-browser.cjs.",
            "The escape allow-list is deliberately broad; it catches a byte a shell replaced, not every "
            "questionable escape.",
        ],
        "passed": not problems,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(problems)} source-hygiene problems across {audited} files")

    print(f"PASS: {checks} source-hygiene checks over {audited} files — no eaten escape, no raw control byte, "
          f"every required KaTeX sequence present, all {len(blocks)} display blocks wrapped, "
          f"no conditional with identical branches.")


if __name__ == "__main__":
    main()
