"""Compile this prepared manuscript into static JSX; no Markdown parser ships to browsers.

Run with scratch/lesson-tools/Scripts/python.exe (Mistune 3.3.4).
The source packet is retained; topic-specific teaching components replace its design markers.
"""
from pathlib import Path
import json
import re
import shutil
import mistune

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/active-learning"
parse = mistune.create_markdown(renderer="ast", plugins=["table"])
opening_paragraphs = {
    "The core route is sections 1–7. Section 8 provides deeper connections to gradients, Gaussian processes, and importance weighting. You can complete the practical experiment before taking that branch.": "route",
}
literal = lambda value: "{" + json.dumps(value, ensure_ascii=False) + "}"
library_bridge = '      <H3>Map the same acquisition loop to a specialist query API</H3>\n      <Prose>The full experiment already implements useful query policies with a scikit-learn learner. <a href="https://scikit-activeml.github.io/latest/generated/api/skactiveml.pool.UncertaintySampling.html">scikit-activeml\'s uncertainty sampler</a> offers a maintained query interface. This bridge keeps the same responsibilities: the controller owns IDs and budget, the strategy sees only acquired labels, and the oracle reveals the selected answer. Its invented one-dimensional pool checks this contract; it is not another accuracy benchmark.</Prose>\n      <Prose>Unlabeled entries are <Code>np.nan</Code>, not a third class. <Code>SklearnClassifier</Code> adapts the learner to that convention. Fit on acquired labels, then use <Code>query(..., fit_clf=False)</Code> so the manual entropy calculation and package utilities share exactly the same fitted probabilities. Candidate IDs index the original pool, not a shortened array, and the returned utility row uses those pool indices. Ties may be broken randomly: compare the maximizing set and utilities rather than expecting every seed to choose the same tied ID.</Prose>\n      <CodeBlock language="sh">{\'python -m pip install numpy==2.3.5 scikit-learn==1.9.1 scikit-activeml==1.0.0\\npython query-library-bridge.py\'}</CodeBlock>\n      <MechanismProgram {...activeMechanismProgram} title="Read the complete query API and budget bridge" />\n      <Prose>Every successful query adds one acquired label, spends one unit of budget and refits immediately—including the final acquisition. Zero budget or an empty eligible pool returns before touching the learner, strategy or oracle. Entropy uses 0 log 0 = 0. Scoring m candidates over K classes costs O(mK) after inference; model fitting has its own cost. Cached scores become stale when the fitted model or eligible pool changes.</Prose>\n      <details className="lesson-solution"><summary>Implementation practice: add annotation eligibility</summary><Prose>Rows 3 and 5 cannot be labeled today. Add an availability mask, retain original pool IDs, and prove neither blocked row reaches the oracle. Repeat with all remaining rows blocked.</Prose><details><summary>Hint and reasoned solution</summary><Prose>Use <Code>np.flatnonzero(np.isnan(known) &amp; available)</Code>, and apply the empty-pool return to that intersection. The same IDs select probabilities and index utilities. Assert that every recorded oracle call was eligible when selected. No eligible rows means no query and no budget spent. Do not fill missing labels with guessed classes: that changes acquisition into pseudo-labeling.</Prose></details></details>\n'

figure_components = {
    "target-directed variance reduction": "TargetVarianceFigure",
    "gradient direction and batch redundancy": "GradientDiversityFigure",
    "an annotation queue": "AnnotationBoundaryFigure",
    "eight threshold rulers": "ThresholdFigure",
    "three probability-strip candidates": "UncertaintyFigure",
    "where committee uncertainty comes from": "CommitteeFigure",
    "coverage before and after selection": "CoverageFigure",
    "measured acquisition curves": "AcquisitionCurves",
    "an annotation record timeline": "AnnotationTimeline",
}
used_figures = []
headings = []
current_section = 0
table_index = 0


def inline(nodes):
    output = []
    for node in nodes:
        kind = node["type"]
        if kind == "text":
            output.append(literal(node["raw"]))
        elif kind in ("softbreak", "linebreak"):
            output.append(literal(" ") if kind == "softbreak" else "<br />")
        elif kind in ("strong", "emphasis", "codespan"):
            tag = {"strong": "strong", "emphasis": "em", "codespan": "Code"}[kind]
            body = literal(node["raw"]) if kind == "codespan" else inline(node["children"])
            output.append(f"<{tag}>{body}</{tag}>")
        elif kind == "link":
            url = node["attrs"]["url"]
            download = not re.match(r"https?://|#|/", url)
            if download:
                url = "/learn/downloads/active-learning/" + url
            extra = " download" if download else ""
            output.append(f"<a href={literal(url)}{extra}>{inline(node['children'])}</a>")
        else:
            raise ValueError(f"Unsupported inline token: {node}")
    return "".join(output)


def render(nodes):
    global current_section, table_index
    output = []
    for node in nodes:
        kind = node["type"]
        if kind == "blank_line":
            continue
        if kind == "heading":
            level = node["attrs"]["level"]
            title = "".join(child.get("raw", "") for child in node["children"])
            if level == 1:
                continue
            if level == 2:
                if current_section in (2, 4, 5):
                    output.append({2: "<ThresholdInvestigation />", 4: "<CommitteeInvestigation />", 5: "<BatchInvestigation />"}[current_section])
                if current_section == 6:
                    output.append("<ExperimentDownloads />")
                if current_section == 7:
                    output.append(library_bridge)
                current_section = int(title.split(".", 1)[0])
                if current_section == 1:
                    output.append("<ActiveLearningRoute />")
                headings.append((f"active-section-{current_section}", title))
                output.append(f'<div id="active-section-{current_section}" className="active-section-anchor"><H2>{inline(node["children"])}</H2></div>')
            else:
                output.append(f'<H3>{inline(node["children"])}</H3>')
        elif kind in ("paragraph", "block_text"):
            text = "".join(child.get("raw", "") or "".join(n.get("raw", "") for n in child.get("children", [])) for child in node["children"])
            if text.startswith("[Inline figure:"):
                name = re.match(r"\[Inline figure: (.+?)\.\]", text)[1]
                if name not in figure_components:
                    raise ValueError(f"Unknown visual placement: {name}")
                used_figures.append(name)
                output.append(f"<{figure_components[name]} />")
            else:
                body = inline(node["children"])
                opening = f' opening="{opening_paragraphs[text]}"' if text in opening_paragraphs else ""
                output.append(f"<Prose{opening}>{body}</Prose>" if kind == "paragraph" else body)
                if text.startswith("For another explanation of the core idea"):
                    output.append('<Prose>Read the <a href="https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/20_al_4-1-2015.pdf">companion lecture slides</a>, especially the query settings, threshold-search example and sampling-bias discussion. Those slide sections were inspected for this lesson; the video is a course-linked alternative, not a claim of full playback review.</Prose>')
        elif kind == "block_code":
            language = node.get("attrs", {}).get("info", "")
            body = literal(node["raw"].rstrip())
            output.append(f'<div className="active-equation" tabIndex={{0}} role="region" aria-label="Mathematical equation"><MathBlock>{body}</MathBlock></div>' if language == "math" else f'<CodeBlock language={literal(language)}>{body}</CodeBlock>')
        elif kind == "table":
            table_index += 1
            rows = []
            headers = node["children"][0]["children"]
            head = "".join(f'<th scope="col">{inline(cell["children"])}</th>' for cell in headers)
            for row in node["children"][1]["children"]:
                cells = []
                for index, cell in enumerate(row["children"]):
                    tag = 'th scope="row"' if index == 0 else 'td'
                    cells.append(f'<{tag}>{inline(cell["children"])}</{tag.split()[0]}>')
                rows.append("<tr>" + "".join(cells) + "</tr>")
            caption = f"Section {current_section}: " + " / ".join("".join(item.get("raw", "") for item in cell["children"]) for cell in headers[:2])
            output.append(f'<div className="lesson-table-wrap" role="region" tabIndex={{0}} aria-label={literal(caption)}><table><caption>{literal(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
        elif kind == "list":
            tag = "ol" if node["attrs"]["ordered"] else "ul"
            output.append(f'<{tag} className="active-list">' + "".join(f"<li>{render(item['children'])}</li>" for item in node["children"]) + f"</{tag}>")
        elif kind == "block_html":
            matches = list(re.finditer(r"<details><summary>(.*?)</summary>(.*?)</details>", node["raw"], re.S))
            if not matches or re.sub(r"<details><summary>.*?</summary>.*?</details>", "", node["raw"], flags=re.S).strip():
                raise ValueError(f"Unexpected authored HTML: {node['raw']}")
            for match in matches:
                output.append(f'<details className="active-answer"><summary>{literal(match[1])}</summary>{render(parse(match[2]))}</details>')
        else:
            raise ValueError(f"Unsupported block token: {node}")
    return "\n      ".join(output)


manuscript = (PACKET / "lesson.md").read_text(encoding="utf-8")
manuscript = re.sub(r"(?m)^\\\[\s*$", "```math", manuscript)
manuscript = re.sub(r"(?m)^\\\]\s*$", "```", manuscript)
body = render(parse(manuscript))
assert set(used_figures) == set(figure_components) and len(used_figures) == 9
assert len(headings) == 10
source = '''import MechanismProgram from '../../components/lesson-labs/MechanismProgram.jsx';
import activeMechanismProgram from '../active-mechanism-program.js';
import { TargetVarianceFigure, GradientDiversityFigure } from '../../components/lesson-labs/ActiveLearningIntuition.jsx';
import { Prose, H2, H3, Code, CodeBlock } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { AnnotationBoundaryFigure, ThresholdFigure, UncertaintyFigure, CommitteeFigure,
  CoverageFigure, AcquisitionCurves, AnnotationTimeline, ExperimentDownloads } from '../../components/lesson-labs/ActiveLearningFigures.jsx';
import { ThresholdInvestigation, CommitteeInvestigation, BatchInvestigation } from '../../components/lesson-labs/ActiveLearningLabs.jsx';
import '../../components/lesson-labs/active-learning.css';

// Compiled from the complete retained packet by scripts/build-active-learning-lesson.py.
// Text is emitted as escaped JS strings, preserving mathematical backslashes and source wording.
function ActiveLearningRoute() {
  return <LessonIntro prerequisites={<>Probability, logistic regression and the previous semi-supervised lesson. Gaussian processes are used only in the optional deeper branch.</>}
    sections={SECTIONS} exampleKind="Python">
    Follow sections 1–7 for the first pass, then attempt the practice. Section 8 is optional depth.
    Three investigations let you spend labels, separate committee uncertainty, and design a geometric batch.
  </LessonIntro>;
}

export default {
  title: 'Active Learning: Choosing Which Examples to Label',
  readTime: '~55 min read + 90 min practice',
  hasIntegratedGuide: true,
  content: () => <div className="active-learning-lesson">
      BODY
      <Prose><a href="/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml">Continue to Evaluation Metrics: connect each score to the decision it supports.</a></Prose>
    </div>,
};
'''.replace("SECTIONS", json.dumps(headings, ensure_ascii=False)).replace("BODY", body)
(ROOT / "src/learn/data/topics/active-learning.jsx").write_text(source, encoding="utf-8")
downloads = ROOT / "public/learn/downloads/active-learning"
downloads.mkdir(parents=True, exist_ok=True)
for name in ("banknote-subset.csv", "banknote-active-learning.py", "data-provenance.md"):
    shutil.copyfile(PACKET / name, downloads / name)
program = (PACKET / "banknote-active-learning.py").read_text(encoding="utf-8")
(ROOT / "src/learn/data/active-learning-examples.js").write_text(
    "// Exact complete downloadable program; verified by scripts/verify-active-learning-native.py.\n"
    + "export const activeLearningProgram = " + json.dumps(program, ensure_ascii=False) + ";\n", encoding="utf-8")
results = json.loads((PACKET / "checked-results.json").read_text(encoding="utf-8"))["banknotes"]
(ROOT / "src/learn/data/active-learning-experiment.json").write_text(json.dumps(results, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
print(f"Compiled all 10 sections, {table_index} tables and 9 inline visual placements; copied 3 downloadable resources.")
