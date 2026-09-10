import { useState } from "react";
import { processActions, transitionProcess } from "../../data/linux-process-model";
import "./linux-lesson.css";

const stateNames = { ready: "Not started", running: "Alive · not stopped", stopped: "Stopped · still exists", exited: "Exited · status available", collected: "Status collected" };

export default function LinuxProcessLab() {
  const [state, setState] = useState("ready");
  const [history, setHistory] = useState([]);
  const [last, setLast] = useState(null);
  function act(action) {
    setState(previous => transitionProcess(previous, action));
    setHistory(previous => [...previous, action]);
    setLast(action);
  }
  return <section className="lesson-lab linux-process" aria-label="Linux process lifecycle explorer">
    <h3>Paused is not finished</h3>
    <p>Start a child, pause it, then continue and end it. Predict: does pausing erase the child's identity? Which action lets its parent collect the result?</p>
    <p className="lesson-note">A manual model of one owned <code>sleep</code> process. It advances only when you act; no command runs and no real process is signaled. PIDs below are illustrative.</p>
    <div className="linux-process__relationship">
      <div className="linux-process__parent"><span className="linux-diagram-label">PARENT · PID 4200</span><strong>Bash shell</strong><span>Accepts your commands</span></div>
      <div className="linux-process__connection" aria-hidden="true"><span className="linux-flow-wide">{state === "collected" ? "← exit status 143" : state === "exited" ? "← result available" : "starts this child →"}</span><span className="linux-flow-narrow">{state === "collected" ? "↑ exit status 143 to parent" : state === "exited" ? "↑ result available to parent" : "↓ starts this child"}</span></div>
      <div className={`linux-process__child linux-process__child--${state}`}>
        <span className="linux-diagram-label">{state === "ready" ? "NO CHILD YET" : "CHILD · EXAMPLE PID 4201"}</span>
        <strong>{state === "ready" ? "Waiting to start" : "sleep 60"}</strong>
        <span>{stateNames[state]}</span>
      </div>
    </div>
    <div className="lesson-controls">{Object.entries(processActions).map(([key, action]) => <button key={key} type="button" disabled={!action.from.includes(state)} onClick={() => act(key)}>{action.label}</button>)}<button type="button" onClick={() => { setState("ready"); setHistory([]); setLast(null); }}>Reset process</button></div>
    <div className="lesson-results" aria-live="polite" aria-atomic="true">
      <p><strong>{stateNames[state]}</strong></p>
      <pre className="linux-command"><code>{last ? processActions[last].command : "# Start the child to begin"}</code></pre>
      <p>{last ? processActions[last].note : "The diagram will keep the parent and child separate. Only the child's state changes in this investigation."}</p>
    </div>
    {history.length > 0 && <ol className="linux-process__history" aria-label="Actions in this investigation">{history.map((action, index) => <li key={index}>{processActions[action].label}</li>)}</ol>}
    <details className="linux-deeper"><summary>Why is End unavailable while stopped?</summary><p>A stopped process may leave a termination signal pending until it is continued. This guided route resumes before sending TERM. Real signal handling, blocking and timing have additional cases. This model also omits the timer expiring naturally.</p></details>
  </section>;
}

export function LinuxEnvironmentDiagram() {
  return <figure className="linux-static-diagram" aria-labelledby="linux-environment-caption">
    <figcaption id="linux-environment-caption"><strong>Inheritance is a starting copy, not a live connection</strong></figcaption>
    <div className="linux-inheritance">
      <div><span className="linux-diagram-label">PARENT SHELL</span><code>export LESSON_MODE=local</code><p>Working directory: <code>/project</code></p><p>Later, the parent's value stays <strong>local</strong>.</p></div>
      <div className="linux-inheritance__arrow" aria-hidden="true"><span className="linux-flow-wide">start child →</span><span className="linux-flow-narrow">↓ start child</span><small>copy exported values<br />and working directory</small></div>
      <div><span className="linux-diagram-label">CHILD SHELL</span><code>LESSON_MODE=local</code><p>Starts in <code>/project</code>.</p><p>Changing its own value to <strong>child</strong> changes only this process.</p></div>
    </div>
    <p className="lesson-note">The child starts with the exported value. Later assignments do not travel back into its parent. Each process also has its own current working directory; a child changing directories does not move the parent.</p>
  </figure>;
}

export function LinuxLinksDiagram() {
  return <figure className="linux-static-diagram" aria-labelledby="linux-links-caption">
    <figcaption id="linux-links-caption"><strong>Two names for one file; one shortcut to a name</strong></figcaption>
    <div className="linux-links-map">
      <div className="linux-links-map__names"><code>record.txt</code><code>hard.txt</code></div>
      <div className="linux-links-map__arrow" aria-hidden="true">↘<br />↗</div>
      <div className="linux-links-map__file"><strong>One underlying file</strong><span>Contents: <code>record</code></span><span>Both names refer here.</span></div>
    </div>
    <p className="linux-links-map__shortcut"><code>shortcut.txt</code><span aria-hidden="true"> → </span>stores the path <code>record.txt</code></p>
    <p>Rename <code>record.txt</code> to <code>renamed.txt</code>: the file still has the names <code>renamed.txt</code> and <code>hard.txt</code>. The shortcut still stores the old path, so it can no longer reach the target.</p>
  </figure>;
}
