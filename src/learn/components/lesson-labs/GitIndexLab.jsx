import { useState } from "react";
import { LessonTable } from "./LessonElements";
import { gitTrace } from "../../data/system-lesson-models";
import "./python-trace.css";

export default function GitIndexLab() {
  const [index, setIndex] = useState(0);
  const step = gitTrace[index];
  return <section className="lesson-lab python-trace" aria-label="Git staging explorer">
    <h3>Which version will the commit contain?</h3>
    <p>A five-step trace for one tracked file, checked against real Git. Predict the next state before advancing. This does not operate on your repository.</p>
    <div className="lesson-controls">
      <button disabled={index === 0} onClick={() => setIndex(i => i - 1)}>Previous step</button>
      <button disabled={index === gitTrace.length - 1} onClick={() => setIndex(i => i + 1)}>Next step</button>
      <button onClick={() => setIndex(0)}>Reset staging</button>
    </div>
    <div className="lesson-results" aria-live="polite" aria-atomic="true">
      <p>Step {index + 1} of {gitTrace.length}</p>
      <pre className="python-trace__code">{step.command}</pre>
      <LessonTable caption="report.txt after this step" headers={["Location", "Stored content"]} rows={["HEAD commit", "Index (staging)", "Working file"].map((label, i) => [label, "version " + step.versions[i]])} />
      <p>Status (two columns before the path):</p><pre className="python-trace__code">{step.status}</pre>
      <p>{step.note}</p>
    </div>
  </section>;
}
