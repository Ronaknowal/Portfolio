import { useState } from "react";
import { LessonTable } from "./LessonElements";

export default function NotebookStateLab() {
  const [rate, setRate] = useState(null);
  const [cost, setCost] = useState(null);
  const [message, setMessage] = useState("Fresh kernel: no variables or output yet.");
  return <section className="lesson-lab" aria-label="Notebook state explorer">
    <h3>Saved output is not a live formula</h3>
    <p>A bounded simulation of three Python cells. Try setting rate to 2, computing cost, then setting rate to 3 and displaying cost again.</p>
    <div className="lesson-controls">
      <button onClick={() => { setRate(2); setMessage("rate = 2 ran. cost was not recalculated."); }}>Set rate to 2</button>
      <button onClick={() => { setRate(3); setMessage("rate = 3 ran. cost was not recalculated."); }}>Set rate to 3</button>
      <button onClick={() => { if (rate === null) setMessage("NameError: rate is not defined."); else { setCost(rate * 10); setMessage("cost = rate * 10 ran using rate " + rate + "."); } }}>Compute cost</button>
      <button onClick={() => setMessage(cost === null ? "NameError: cost is not defined." : "print(cost) → " + cost)}>Display cost</button>
      <button onClick={() => { setRate(null); setCost(null); setMessage("Fresh kernel: variables cleared. Saved notebook output, if any, would still need rerunning."); }}>Restart kernel</button>
    </div>
    <LessonTable caption="Current simulated kernel memory" headers={["Variable", "Value"]} rows={[["rate", rate ?? "undefined"], ["cost", cost ?? "undefined"]]} />
    <div className="lesson-results" aria-live="polite">{message}</div>
    <p className="lesson-note">A real restart clears Python memory, not files, database changes or external services. This explorer does not execute arbitrary code.</p>
  </section>;
}
