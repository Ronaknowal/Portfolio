import { useState } from "react";
import { LessonTable } from "./LessonElements";
import "./python-trace.css";

export const broadcastCases = {
  feature: { shape: "(2,)", values: [20, 50], aligned: [[20, 50], [20, 50], [20, 50]] },
  sample: { shape: "(3, 1)", values: [[1], [2], [3]], aligned: [[1, 1], [2, 2], [3, 3]] },
  invalid: { shape: "(3,)", values: [1, 2, 3], aligned: null },
};
const X = [[18, 40], [24, 50], [30, 60]];

export default function BroadcastLab() {
  const [mode, setMode] = useState("feature");
  const [cell, setCell] = useState(0);
  const selected = broadcastCases[mode];
  const row = Math.floor(cell / 2), column = cell % 2;
  return <section className="lesson-lab broadcast-lab" aria-label="NumPy broadcasting lab">
    <h3>Choose which axis receives an offset</h3>
    <p>X has shape (3, 2): three observations and two features. Predict the result shape before changing the offset. This is an illustrated model of broadcasting, not a Python runtime.</p>
    <div className="lesson-controls">
      <label>Offset shape<select aria-label="Offset shape" value={mode} onChange={e => setMode(e.target.value)}>
        <option value="feature">Per feature: (2,)</option>
        <option value="sample">Per observation: (3, 1)</option>
        <option value="invalid">Incompatible: (3,)</option>
      </select></label>
      <button type="button" onClick={() => { setMode("feature"); setCell(0); }}>Reset broadcasting</button>
    </div>
    <LessonTable caption="Input X, shape (3, 2)" headers={["Observation", "Feature 0", "Feature 1"]} rows={X.map((values, i) => [i, ...values])} />
    <div className="lesson-results" aria-live="polite">
      <p>Offset shape <strong>{selected.shape}</strong>: <code>{JSON.stringify(selected.values)}</code></p>
      {selected.aligned ? <p>Compatible → result shape <strong>(3, 2)</strong>. {mode === "feature" ? "Align (2,) as (1, 2); use each column's offset on every row." : "The singleton column expands; use each row's offset on both columns."}</p> : <p><strong>ValueError: incompatible shapes.</strong> Compare the rightmost dimensions first: 2 and 3 are unequal and neither is 1. NumPy does not infer that you intended one value per row.</p>}
    </div>
    {selected.aligned && <>
      <LessonTable caption="Conceptual offsets at each position (no tiling required in NumPy)" headers={["Observation", "Feature 0", "Feature 1"]} rows={selected.aligned.map((values, i) => [i, ...values])} />
      <p>Inspect a result cell:</p>
      <div className="broadcast-cells">
        {X.flatMap((values, i) => values.map((v, j) => <button type="button" key={i + "-" + j} aria-label={"Inspect row " + i + " column " + j} aria-pressed={cell === i * 2 + j} onClick={() => setCell(i * 2 + j)}>
          <span>[{i}, {j}]</span><strong>{v - selected.aligned[i][j]}</strong>
        </button>))}
      </div>
      <p className="broadcast-explanation" aria-live="polite">Result[{row}, {column}] = X[{row}, {column}] − offset = {X[row][column]} − {selected.aligned[row][column]} = <strong>{X[row][column] - selected.aligned[row][column]}</strong>.</p>
    </>}
  </section>;
}
