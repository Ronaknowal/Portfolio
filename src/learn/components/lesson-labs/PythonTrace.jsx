import { useState } from "react";
import { LessonTable } from "./LessonElements";
import "./python-trace.css";

export const traces = {
  references: {
    title: "Watch names point to objects",
    question: "Before each step, predict which names will see a changed list. A, B and C below are object labels, not memory addresses.",
    steps: [
      { code: "a = [18, 21]", rows: [["a", "List A", "[18, 21]"]], note: "Create a list, then bind the name a to it." },
      { code: "b = a", rows: [["a", "List A", "[18, 21]"], ["b", "List A", "[18, 21]"]], note: "Assignment adds a name for the same list. It does not copy the list." },
      { code: "c = a.copy()", rows: [["a", "List A", "[18, 21]"], ["b", "List A", "[18, 21]"], ["c", "List B", "[18, 21]"]], note: "A shallow copy creates a new outer list. These elements are immutable numbers." },
      { code: "b.append(24)", rows: [["a", "List A", "[18, 21, 24]"], ["b", "List A", "[18, 21, 24]"], ["c", "List B", "[18, 21]"]], note: "Mutation changes List A. Both a and b see it; c does not." },
      { code: "b = [0]", rows: [["a", "List A", "[18, 21, 24]"], ["b", "List C", "[0]"], ["c", "List B", "[18, 21]"]], note: "Rebinding b does not change List A or the name a." },
    ],
  },
  instances: {
    title: "Follow the state of two objects",
    question: "Both logs use the same class definition. Predict which instance each call changes; self refers to that instance.",
    steps: [
      { code: 'morning = ReadingLog("morning")', rows: [["morning", "Log A → values list A", "[]"]], note: "Initialisation creates a fresh list for this instance." },
      { code: 'evening = ReadingLog("evening")', rows: [["morning", "Log A → values list A", "[]"], ["evening", "Log B → values list B", "[]"]], note: "A second constructor call creates a separate log and a separate list." },
      { code: "morning.add(18)", rows: [["morning", "Log A → values list A", "[18]"], ["evening", "Log B → values list B", "[]"]], note: "Python supplies morning as self. Only morning.values changes." },
      { code: "ReadingLog.add(morning, 24)", rows: [["morning", "Log A → values list A", "[18, 24]"], ["evening", "Log B → values list B", "[]"]], note: "This is the explicit form of morning.add(24), not a second object." },
      { code: "alias = morning; alias.add(30)", rows: [["morning", "Log A → values list A", "[18, 24, 30]"], ["alias", "Log A → values list A", "[18, 24, 30]"], ["evening", "Log B → values list B", "[]"]], note: "Another name is not another instance. The alias mutates the morning log." },
    ],
  },
  generator: {
    title: "Advance a generator one request at a time",
    question: "The function in the worked example prints before and between yields. Predict when those messages appear. Creating the generator does not run its body.",
    steps: [
      { code: "stream = readings()", rows: [["stream", "Created", "Body has not started"]], output: "", note: "The generator exists, but has not produced any readings." },
      { code: 'print("created")', rows: [["stream", "Created", "Body has not started"]], output: "created", note: "This print belongs to the caller, not the generator." },
      { code: "print(next(stream))", rows: [["stream", "Suspended after yield 18", "Next request resumes here"]], output: "created\nstart\n18", note: "Start the body, print start, then yield 18. The caller prints the returned item." },
      { code: "print(next(stream))", rows: [["stream", "Suspended after yield 24", "Next request resumes here"]], output: "created\nstart\n18\nresume\n24", note: "Resume after the first yield. Locals and execution position were retained." },
      { code: 'print(next(stream, "done"))', rows: [["stream", "Exhausted", "No next value"]], output: "created\nstart\n18\nresume\n24\nfinish\ndone", note: "The body finishes. next uses the supplied default when iteration stops." },
      { code: 'print(next(stream, "done"))', rows: [["stream", "Exhausted", "Still no next value"]], output: "created\nstart\n18\nresume\n24\nfinish\ndone\ndone", note: "Exhaustion is permanent for this generator. The body does not restart." },
    ],
  },
};

export default function PythonTrace({ kind, trace: suppliedTrace }) {
  const trace = suppliedTrace || traces[kind];
  const [index, setIndex] = useState(0);
  const step = trace.steps[index];
  return <section className="lesson-lab python-trace" aria-label={trace.title}>
    <h3>{trace.title}</h3>
    <p>{trace.question}</p>
    <p className="lesson-note">Illustrated Python execution trace; this is not an embedded Python interpreter.</p>
    <div className="lesson-controls">
      <button type="button" disabled={index === 0} onClick={() => setIndex(i => i - 1)}>Previous step</button>
      <button type="button" disabled={index === trace.steps.length - 1} onClick={() => setIndex(i => i + 1)}>Next step</button>
      <button type="button" onClick={() => setIndex(0)}>Reset trace</button>
    </div>
    <div className="python-trace__state" aria-live="polite" aria-atomic="true">
      <p><strong>Step {index + 1} of {trace.steps.length}</strong></p>
      <pre className="python-trace__code"><code>{step.code}</code></pre>
      <LessonTable caption="State after this step" headers={trace.headers || ["Name", kind === "generator" ? "Execution state" : "Refers to", kind === "generator" ? "What happens next" : "Stored readings"]} rows={step.rows} />
      {step.output !== undefined && <div><p>Output so far</p><pre className="python-trace__code">{step.output || "(nothing yet)"}</pre></div>}
      <p>{step.note}</p>
    </div>
  </section>;
}
