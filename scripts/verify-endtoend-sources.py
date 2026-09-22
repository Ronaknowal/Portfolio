"""Source-hygiene audit of every file the end-to-end lesson owns, its verifiers included.

This exists because of specific, repeated failures in this repository.

  1. **A shell ate an escape.** A heredoc turns a backslash sequence into the byte
     it names. That has shipped a broken virtual-environment path, a lost LaTeX
     control word, and -- worst -- a regex whose word-boundary escape became a
     literal control byte INSIDE A VERIFIER, so the guard written to catch a
     defect could never fire. A checker that audits only the lesson sources and
     not the checkers would repeat exactly that, so the verifiers and the
     falsification harness are audited here too.
  2. **A KaTeX sequence went missing**, silently changing what a formula says.
     The sequences this lesson depends on are named and required.
  3. **A display block was left as one long line**, which overflows a 320 px
     column. Every block must be short or explicitly wrapped.
  4. **A conditional whose branches are identical** -- the prose equivalent of an
     assertion that cannot fail.
  5. **`Math` is shadowed in the lesson body.** The topic module imports KaTeX's
     `Math` component, so `Math.sqrt` there resolves to a React component and
     yields `undefined` rather than a number, silently.
  6. **A bare descendant `svg` selector under the lesson root.** This is the
     worst defect this effort has produced: `.lesson svg { height: auto }` also
     matches KaTeX's own SVGs, whose height comes from `height: inherit`, and
     collapses every radical on the page. This page renders three KaTeX SVGs --
     one radical in the Wilson formula and two stretchy delimiters in the
     practice matrix -- so the hazard is live here. The rule is refused at the
     selector level, not merely checked for at the file level.
  7. **Borrowing a sibling lesson's dataset.** Three other lessons serve Wine
     measurements. This one owns its copy and must never reference theirs.

And two that belong to this topic in particular.

  8. **A score printed without its role.** Every number this page shows is
     training, validation, selection or held-out evidence. A six-decimal
     literal typed into a component is a score nobody labelled, so they are
     refused: the numbers come from the data module and reach the page through
     the two components that require a role.
  9. **A held-out quantity hidden rather than withheld.** The gate must not
     render its children at all. A CSS rule that hides an element leaves its
     text in the document, in the accessibility tree and in a copy-paste, so any
     attempt to gate the report with `display: none` is refused here.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-endtoend-sources.py
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
EVIDENCE = ROOT / "docs/teaching/evidence/endtoend-sources.json"

FILES = [
    "src/learn/data/endtoend-models.js",
    "src/learn/data/endtoend-data.js",
    "src/learn/data/endtoend-examples.js",
    "src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx",
    "src/learn/data/curriculum/blueprints/end-to-end-supervised-learning-error-analysis.js",
    "src/learn/components/lesson-labs/EndToEndShared.jsx",
    "src/learn/components/lesson-labs/EndToEndLabs.jsx",
    "src/learn/components/lesson-labs/EndToEndFigures.jsx",
    "src/learn/components/lesson-labs/endtoend-labs.css",
    "scripts/verify-endtoend-models.mjs",
    "scripts/verify-endtoend-examples.py",
    "scripts/verify-endtoend-data.py",
    "scripts/verify-endtoend-browser.cjs",
    "scripts/verify-endtoend-sources.py",
    "scripts/falsify-endtoend.mjs",
    "public/learn-assets/end-to-end/ATTRIBUTION.txt",
]

TOPIC = "src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx"
CSS = "src/learn/components/lesson-labs/endtoend-labs.css"
COMPONENTS = [
    TOPIC,
    "src/learn/components/lesson-labs/EndToEndLabs.jsx",
    "src/learn/components/lesson-labs/EndToEndFigures.jsx",
    "src/learn/components/lesson-labs/EndToEndShared.jsx",
]
DRAWING = [
    "src/learn/components/lesson-labs/EndToEndShared.jsx",
    "src/learn/components/lesson-labs/EndToEndFigures.jsx",
    "src/learn/components/lesson-labs/EndToEndLabs.jsx",
]
LAYOUT_CLASSES = ("ete-plot", "ete-strip", "ete-lanes", "ete-rail", "ete-bars")

REQUIRED_KATEX = [
    r"\\frac", r"\\sqrt", r"\\sum", r"\\mu_", r"\\mathrm", r"\\log",
    r"\\begin{gathered}", r"\\end{gathered}", r"\\begin{pmatrix}", r"\\end{pmatrix}",
    r"\\pm", r"\\hat p",
]

# Paths that belong to other lessons serving Wine measurements.
FORBIDDEN_PATHS = [
    "learn-assets/pca",
    "learn-assets/bayesian-networks",
    "learn-assets/feature-selection",
]

# These files name the forbidden paths on purpose; see the check below.
PATH_NAMING_ALLOWED = {
    "scripts/verify-endtoend-sources.py",
    "scripts/verify-endtoend-browser.cjs",
    "scripts/falsify-endtoend.mjs",
    "public/learn-assets/end-to-end/ATTRIBUTION.txt",
}

# Members of the global Math object. In the topic module `Math` is the KaTeX
# component, so any of these is a silent undefined.
MATH_MEMBERS = [
    "round", "sqrt", "min", "max", "abs", "log", "log2", "log10", "exp", "floor",
    "ceil", "pow", "hypot", "sign", "trunc", "random", "E", "PI", "LN2", "LN10",
]

# The held-out quantities. None of them may be typed into a component: they
# come from the data module, and the gate decides whether they are rendered.
HELD_OUT_LITERALS = ["0.966667", "0.972222", "0.128708", "0.9666666", "0.9722222"]

keep_evidence = "--no-evidence" not in sys.argv

problems: list[str] = []
checks = 0


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)
    return bool(condition)


def scanned(before, label):
    """Turn a scan that reports by appending into a real, counted assertion.

    A scan reports by appending to `problems`, which does not touch the counter,
    so a headline count that included them would be counting scans nobody ran.
    Each of these evaluates whether its own scan found anything; the specific
    problem is already recorded, so this does not append a second time.
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
        #
        # Escapes that survive as TEXT are legitimate -- a regex word boundary is
        # spelled with a backslash and a b. What must never survive is the BYTE a
        # shell would leave if it ate one, and that is scan 1a's job. (This
        # comment deliberately spells no escape out; a draft of it tripped this
        # very scan.)
        mark = len(problems)
        allowed = set("nrtbfv0\\'\"`$/aeux<>whsSdDwWbBAZzpPkGQE.^[]()|*+?{}-,:;=!&@#%~ \n")
        latex_words = {
            "frac", "tfrac", "sqrt", "sum", "mu", "mathrm", "mathbf", "log", "ln", "exp", "hat",
            "widehat", "le", "ge", "ne", "in", "to", "pm", "cdot", "cdots", "begin", "end",
            "gathered", "pmatrix", "text", "left", "right", "quad", "approx", "times", "dots",
            "ldots", "varepsilon", "delta", "theta", "pi", "infty", "min", "max", "sim",
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

        # 7. No borrowing of a sibling lesson's copy of the same measurements.
        mark = len(problems)
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text and relative not in PATH_NAMING_ALLOWED:
                line = text[:text.index(forbidden)].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: references {forbidden}, which belongs to another lesson. "
                    "This lesson serves its own copy under public/learn-assets/end-to-end/.")
        scanned(mark, f"{relative}: own dataset only")

    topic = (ROOT / TOPIC).read_text(encoding="utf-8")

    # 2. The KaTeX sequences the lesson depends on are still there.
    for token in REQUIRED_KATEX:
        check(token in topic, f"the KaTeX sequence {token} is missing from the lesson body")

    # 3. Every display block is short or wrapped rather than left as one long line.
    #
    # A block's source may be one literal or several concatenated across lines,
    # so the expression between the tags is captured first and its string
    # literals joined. Matching `'(.*?)'` directly ran across the gap between two
    # blocks and reported a 4,451-character formula that does not exist.
    blocks = []
    for expression in re.findall(r"<MathBlock>\{(.*?)\}</MathBlock>", topic, re.S):
        blocks.append("".join(re.findall(r"'([^']*)'", expression)))
    check(len(blocks) >= 3, f"only {len(blocks)} display blocks were found to check")
    # Two signals, because the first alone let a formula through that then
    # overflowed at 320 px on the real page: stripping every backslash command
    # makes a formula full of commands look short, when each renders as a glyph.
    # The raw length is the blunter, more honest proxy. Neither measures the
    # rendered width; that is what scripts/verify-endtoend-browser.cjs does.
    for block in blocks:
        visible = re.sub(r"\\[a-zA-Z]+\{?|[{}\\]", "", block)
        wrapped = "\\\\\\\\" in block or "gathered" in block or "pmatrix" in block
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

    # 5. `Math` is the KaTeX component in the lesson body.
    check("from '../../components/content/Math.jsx'" in topic,
          "the lesson body imports the KaTeX Math component, which is what shadows the global")
    # Block comments are stripped first, with their line count preserved so the
    # reported line number stays right. This file's own explanation of the
    # hazard names a Math member inside a comment, and a scan that flagged the
    # explanation would be a scan nobody could keep green.
    topic_code = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), topic, flags=re.S)
    mark = len(problems)
    for member in MATH_MEMBERS:
        for match in re.finditer(r"\bMath\." + member + r"\b", topic_code):
            line = topic_code[:match.start()].count("\n") + 1
            problems.append(
                f"{TOPIC}:{line}: Math.{member} in a module where `Math` is the KaTeX component. "
                "It resolves to that component and yields undefined, silently. Use a helper from "
                "endtoend-models.js instead.")
    scanned(mark, "no shadowed global Math member access in the lesson body")

    # 6a. Every SVG this lesson renders carries one of its own layout classes.
    for relative in DRAWING:
        body = (ROOT / relative).read_text(encoding="utf-8")
        # Block comments are stripped first, with their line count preserved so
        # the reported line number stays right. These files explain the tagging
        # rule in a comment that names an svg tag, and a scan that flagged its
        # own explanation would be a scan nobody could keep green -- which is
        # how a guard gets deleted rather than fixed.
        body = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), body, flags=re.S)
        mark = len(problems)
        for match in re.finditer(r"<svg\b([^>]*)>", body, re.S):
            attributes = match.group(1)
            line = body[:match.start()].count("\n") + 1
            tagged = re.search(r'className=\{?["`][^"`]*\b(' + "|".join(LAYOUT_CLASSES) + r")\b", attributes)
            variable = re.search(r"className=\{className", attributes)
            if not tagged and not variable:
                problems.append(
                    f"{relative}:{line}: an <svg> with no {' or '.join(LAYOUT_CLASSES)} class. "
                    "The layout rule is scoped to those classes, so this one would not get it.")
        scanned(mark, f"{relative}: every svg is tagged for the layout rule")

    # 6a-ii. The two frames take their class from a prop, so the prop's VALUE is
    # what decides whether the layout rule reaches the element. Every call site
    # must pass one of the layout classes, and the default must be one too;
    # otherwise the element-level scan above passes an svg that the stylesheet
    # never touches.
    shared_for_default = (ROOT / "src/learn/components/lesson-labs/EndToEndShared.jsx").read_text(
        encoding="utf-8")
    default = re.search(r"className\s*=\s*'([^']+)'\s*,?\s*\n?\s*xTickText", shared_for_default) \
        or re.search(r"className\s*=\s*'(ete-[a-z]+)'", shared_for_default)
    check(default is not None and default.group(1) in LAYOUT_CLASSES,
          "the PlotFrame default class is one of this lesson's layout classes")
    for relative in DRAWING:
        body = (ROOT / relative).read_text(encoding="utf-8")
        body = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), body, flags=re.S)
        mark = len(problems)
        for match in re.finditer(r"<(Canvas|PlotFrame)\b((?:[^<>]|\{[^{}]*\})*?)(?:/?>)", body, re.S):
            given = re.search(r'className="([^"]*)"', match.group(2))
            if given and given.group(1) not in LAYOUT_CLASSES:
                line = body[:match.start()].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: <{match.group(1)}> is given className=\"{given.group(1)}\", which is "
                    f"not one of {', '.join(LAYOUT_CLASSES)}. The stylesheet's layout rule is scoped to those "
                    "classes, so this figure would render at its intrinsic size.")
        scanned(mark, f"{relative}: every figure frame is given a layout class")

    css = (ROOT / CSS).read_text(encoding="utf-8")
    check("svg:is(.ete-plot, .ete-strip, .ete-lanes, .ete-rail, .ete-bars)" in css,
          "the SVG layout rule is scoped by class and cannot reach KaTeX's own SVGs")

    # 6b. NO BARE DESCENDANT `svg` SELECTOR, at the selector level.
    #
    # This is the strong form of the rule. Comments are stripped first, then
    # every selector is examined: a compound that is exactly the element `svg`,
    # with no class or attribute qualifying it, is refused outright. A rule such
    # as `.endtoend-lesson svg text` is fine -- its last compound is `text` --
    # and `svg:is(...)` is fine because the element is qualified.
    css_code = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), css, flags=re.S)
    mark = len(problems)
    for match in re.finditer(r"([^{}]+)\{", css_code):
        selector_text = match.group(1).strip()
        if not selector_text or selector_text.startswith("@"):
            continue
        for selector in selector_text.split(","):
            compounds = selector.strip().split()
            if compounds and compounds[-1] == "svg":
                line = css_code[:match.start()].count("\n") + 1
                problems.append(
                    f"{CSS}:{line}: the selector `{selector.strip()}` ends in a bare `svg` compound. "
                    "Under a root where KaTeX renders, that also matches KaTeX's own SVGs, whose height "
                    "comes from `height: inherit`; a height rule there collapses every radical on the page "
                    "to nothing and silently changes what a formula says. Qualify the element with one of "
                    f"this lesson's layout classes: {', '.join(LAYOUT_CLASSES)}.")
    scanned(mark, f"{CSS}: no bare descendant svg selector")

    # 8. Every score the page shows carries its role.
    #
    # Two halves. First, the components that print a score require a role and
    # refuse anything else -- checked by requiring the guard to be present.
    # Second, no six-decimal literal is typed into a component at all, because
    # such a literal is a number nobody labelled.
    shared = (ROOT / "src/learn/components/lesson-labs/EndToEndShared.jsx").read_text(encoding="utf-8")
    check("SCORE_ROLES.includes(record.role)" in shared,
          "the Score component refuses a record without one of the four roles")
    check("SCORE_ROLES.includes(role)" in shared,
          "the Count component refuses a count without one of the four roles")
    check('data-role={record.role}' in shared,
          "the printed score carries its role as a data attribute, so the browser verifier can assert it")
    models = (ROOT / "src/learn/data/endtoend-models.js").read_text(encoding="utf-8")
    check("throw new RangeError" in models and "is not one of the four score roles" in models,
          "the model layer refuses to build a score record with an unknown role")

    for relative in COMPONENTS:
        body = (ROOT / relative).read_text(encoding="utf-8")
        code = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), body, flags=re.S)
        code = re.sub(r"//[^\n]*", "", code)
        mark = len(problems)
        for match in re.finditer(r"(?<![\w.])\d\.\d{4,}", code):
            line = code[:match.start()].count("\n") + 1
            problems.append(
                f"{relative}:{line}: the literal {match.group(0)} looks like a score typed into a component. "
                "Scores come from endtoend-data.js and reach the page through Score or Count, which require "
                "a role; a typed one carries none.")
        scanned(mark, f"{relative}: no score typed in without a role")
        mark = len(problems)
        for literal in HELD_OUT_LITERALS:
            if literal in code:
                line = code[:code.index(literal)].count("\n") + 1
                problems.append(
                    f"{relative}:{line}: the held-out quantity {literal} is typed into a component. "
                    "A held-out number must come from the data module so that the gate decides whether it "
                    "is rendered at all.")
        scanned(mark, f"{relative}: no held-out quantity typed in")

    # 9. The gate withholds rather than hides.
    check("if (earned) return children;" in shared,
          "HeldOutOnly returns its children only when earned, rather than rendering and hiding them")
    check("ete-sealed" in shared and "ete-sealed" in css,
          "the closed state renders its own placeholder element, which is what a reader sees instead")
    mark = len(problems)
    for match in re.finditer(r"\.ete-sealed[^{]*\{([^}]*)\}", css_code):
        if "display: none" in match.group(1) or "visibility: hidden" in match.group(1):
            line = css_code[:match.start()].count("\n") + 1
            problems.append(
                f"{CSS}:{line}: the sealed placeholder is hidden by CSS. The gate's claim is that the "
                "held-out text is NOT IN THE DOCUMENT; hiding an element leaves its text in the "
                "accessibility tree and in a copy-paste.")
    scanned(mark, "the gate is not a CSS rule")
    check(topic.count("<HeldOutOnly") >= 3,
          f"the lesson body wraps only {topic.count('<HeldOutOnly')} regions in the gate; the held-out "
          "report, the final report paragraph and the Wilson scale check all quote held-out quantities")
    check("<HeldOutProvider>" in topic, "the lesson body installs the gate around its whole content")

    # 10. EVERY OWNED .jsx ACTUALLY PARSES.
    #
    # This one has already fired in reality, which is more than most of what
    # this file asserts can say. An unescaped apostrophe inside a single-quoted
    # string reached `EndToEndLabs.jsx` and broke `npx vite build` REPO-WIDE:
    # a sibling lesson discovered it because it could not regenerate its own
    # evidence, and had to report a falsification record that understated what
    # it had proved.
    #
    # Nothing else here could have caught it. The model and data verifiers
    # import plain `.js` and never touch a component; the text scans below are
    # regexes over source and do not parse. Only a build or a browser run sees
    # it — and both are far away from the edit that causes it. A parse is cheap,
    # runs offline in about a second, and a broken component in a file this
    # lesson owns must fail this lesson's own checker.
    parsed = 0
    for relative in COMPONENTS:
        path = ROOT / relative
        result = subprocess.run(
            ["npx", "esbuild", str(path), "--loader:.jsx=jsx", "--jsx=automatic",
             *(["--outfile=NUL"] if os.name == "nt" else ["--outfile=/dev/null"])],
            capture_output=True, text=True, shell=(os.name == "nt"), cwd=ROOT, timeout=180)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            problems.append(
                f"{relative}: does not parse as JSX, so `npx vite build` fails for the WHOLE repository "
                f"and every sibling lesson's build with it — {' | '.join(detail[:3])}")
        else:
            parsed += 1
    check(parsed == len(COMPONENTS),
          f"only {parsed} of {len(COMPONENTS)} owned JSX files parsed")

    # 11. THE HARNESS REFUSES TO RECOVER OVER A LIVE RUN — exercised, not read.
    #
    # A lock plus a `.orig` sidecar cannot distinguish "crashed" from "case in
    # progress", and that ambiguity has already cost a falsification run: two
    # independent readers drew OPPOSITE wrong conclusions from exactly those
    # two files within the same hour, and `--recover` was run against a live
    # harness, un-mutating a file it had deliberately broken.
    #
    # The lock now carries a heartbeat and `--recover` refuses while it is
    # fresh. That refusal is asserted here by performing it: a lock naming a
    # live pid with a current heartbeat is written, `--recover` is run, and the
    # refusal is required. Asserting the presence of the code that refuses
    # would be a text match; this is the behaviour.
    #
    # A THROWAWAY lock path, so this never touches the real one. This verifier
    # is itself run by the harness, which holds the real lock; an earlier draft
    # skipped whenever a lock existed, which meant it skipped exactly when the
    # harness ran it and its falsification case was inert. The harness reads
    # ENDTOEND_LOCK, so the refusal can be exercised in isolation instead.
    lock = ROOT / "scratch/endtoend-recovery-gate-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.write_text(json.dumps({
            "pid": os.getpid(),
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "heartbeatAt": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n", encoding="utf-8")
        attempt = subprocess.run(
            ["node", "scripts/falsify-endtoend.mjs", "--recover"],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
            shell=(os.name == "nt"), env={**os.environ, "ENDTOEND_LOCK": str(lock)})
        refused = "refusing to recover" in (attempt.stderr + attempt.stdout)
        check(refused,
              "the harness recovered over a LIVE run instead of refusing: --recover must not un-mutate a "
              "file a running harness deliberately broke, because the run's result then describes a tree "
              f"that no longer exists. It said: {(attempt.stderr or attempt.stdout)[:200]}")
        check(attempt.returncode != 0, "the refusal exits non-zero so a script cannot ignore it")
        recovery_gate = "exercised: --recover refused against a live heartbeat"
    finally:
        lock.unlink(missing_ok=True)

    # 12. THE HEARTBEAT ADVANCES ACROSS A BLOCKING CHILD — exercised.
    #
    # The first heartbeat used `setInterval`, and measured against a live run it
    # never advanced at all: a timer cannot fire in a process that blocks its
    # event loop with `spawnSync`, which is how the harness drives every child.
    # It was inert exactly while a case was running. The probe below makes one
    # short blocking call and requires the recorded heartbeat to move.
    probe = subprocess.run(
        ["node", "scripts/falsify-endtoend.mjs", "--heartbeat-probe"],
        capture_output=True, text=True, cwd=ROOT, timeout=180, shell=(os.name == "nt"),
        env={**os.environ, "ENDTOEND_LOCK": str(ROOT / "scratch/endtoend-heartbeat-probe.lock")})
    advanced = '"advanced":true' in probe.stdout.replace(" ", "")
    check(advanced,
          "the lock's heartbeat did not advance across a blocking child, so it records only the moment the "
          f"run started and says nothing about whether it is still going. Probe said: {probe.stdout[:160]}")

    # 13. THE LIVENESS PROBE ANSWERS CORRECTLY — exercised, on real pids.
    #
    # This decides whether --recover may touch anything, so it is the most
    # load-bearing line in the harness. Adding `shell: true` to it makes
    # `tasklist /FI` reach the MSYS layer, which rewrites the switch into a path;
    # the call then errors and the probe answers "dead" for a live process --
    # the direction that invites recovery over a live run.
    live_child = subprocess.Popen(["node", "-e", "setTimeout(()=>{},8000)"], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        def ask(pid):
            return subprocess.run(["node", "scripts/falsify-endtoend.mjs", "--is-alive", str(pid)],
                                  capture_output=True, text=True, cwd=ROOT, timeout=120,
                                  shell=(os.name == "nt")).stdout.strip()
        check(ask(live_child.pid) == "alive",
              f"the liveness probe reported a RUNNING process (pid {live_child.pid}) as not running. That is "
              "the dangerous direction: --recover would then restore over a live run.")
        check(ask(999999) == "dead", "the liveness probe reported a nonexistent pid as running")
    finally:
        live_child.kill()
        live_child.wait(timeout=30)

    # A guard on the guards: the verifiers must contain assertions, and the
    # browser verifier must actually look for what this topic is exposed to.
    models_verifier = (ROOT / "scripts/verify-endtoend-models.mjs").read_text(encoding="utf-8")
    check(models_verifier.count("assert") >= 150, "the models verifier still carries its assertions")
    check("passed: false" in models_verifier,
          "the models verifier writes a provisional failing record before its first assertion")
    browser = (ROOT / "scripts/verify-endtoend-browser.cjs").read_text(encoding="utf-8")
    check("sampleCurvesThroughLabels" in browser,
          "the browser verifier carries its own curve-through-label sampler")
    check(".katex svg" in browser or "katex .sqrt" in browser,
          "and measures KaTeX's own SVGs on the rendered page, which is the only thing standing between a "
          "stylesheet and a wrong formula")
    check("toFixed(6)" in browser or "fixed(" in browser,
          "and pins the graded quantity numerically rather than by marker class alone")
    check("data-role" in browser,
          "and asserts that every printed score carries its role in the rendered DOM")

    # A floor, close to the real number. A verifier with none would shrink its
    # own headline silently when a scan was deleted.
    check(checks >= 95, f"only {checks} hygiene checks ran; the suite has lost coverage")
    check(audited == len(FILES), f"only {audited} of {len(FILES)} declared files were audited")

    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-endtoend-sources.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filesAudited": audited,
        "filesDeclared": len(FILES),
        "checks": checks,
        "displayBlocksChecked": len(blocks),
        "requiredKatexSequences": len(REQUIRED_KATEX),
        "gatedRegionsInLessonBody": topic.count("<HeldOutOnly"),
        "harnessRecoveryGate": recovery_gate,
        "fileHashes": {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                       for relative in FILES if (ROOT / relative).exists()},
        "scope": "Raw control bytes and undefined backslash escapes in every file this lesson owns, its five "
                 "verifiers and its falsification harness included; the KaTeX sequences the body depends on; "
                 "that every display block is short or wrapped; that no JSX conditional has identical "
                 "branches; that the lesson body never reaches for a member of the global Math object in a "
                 "module where `Math` is the KaTeX component; that no selector in this lesson's stylesheet "
                 "ends in a bare `svg` compound and that every svg the lesson renders carries a layout class; "
                 "that no score or held-out quantity is typed into a component rather than coming from the "
                 "data module through a component that requires a role; that the held-out gate withholds its "
                 "children rather than hiding them with CSS; and that no owned file references another "
                 "lesson's copy of the Wine measurements.",
        "limitations": [
            "Text-level hygiene only. It does not parse JSX, CSS or KaTeX, and a wrapped block can still "
            "overflow: the rendered width is measured by scripts/verify-endtoend-browser.cjs.",
            "The bare-`svg` scan splits selectors on whitespace and commas. It catches the descendant form "
            "that has caused this defect; an exotic selector written to evade it would not be caught here, "
            "and the rendered KaTeX measurement in the browser verifier is the backstop.",
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
          f"no conditional with identical branches, no shadowed Math member, no bare `svg` selector, "
          f"no score or held-out quantity typed in without a role, "
          f"{topic.count('<HeldOutOnly')} gated regions that withhold rather than hide, "
          f"and no sibling-lesson asset path.")


if __name__ == "__main__":
    main()
