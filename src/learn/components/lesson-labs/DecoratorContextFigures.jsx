import { useId } from "react";
import "./python-mechanism-figures.css";

function ArrowMarker({ id }) {
  return <defs><marker id={id} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor" /></marker></defs>;
}

export function DecoratorBindingFigure() {
  return <figure className="pmf-figure" data-python-figure="decorator-binding">
    <figcaption><strong>The public name moves; the original function stays reachable.</strong><span>Definition time and call time are different events.</span></figcaption>
    <div className="pmf-decoration">
      <div className="pmf-decoration-before"><small>BEFORE @logged</small><code>mean</code><span className="pmf-link">↓ refers to</span><div className="pmf-function"><strong>original mean</strong><code>return sum(values) / len(values)</code></div></div>
      <div className="pmf-decoration-after"><small>AFTER mean = logged(mean)</small><code>mean</code><span className="pmf-link">↓ now refers to</span><div className="pmf-wrapper"><strong>wrapper</strong><p>Retained reference:</p><code>function</code><span className="pmf-link">↓</span><div className="pmf-function"><strong>same original mean</strong><code>return sum(values) / len(values)</code></div></div></div>
    </div>
    <div className="pmf-call-ribbon"><code>mean([18, 24])</code><span>→ enter wrapper → call retained function → 21.0 returns outward</span></div>
    <p className="pmf-caption">The inner box marks a retained reference, not copied source code or memory layout. In the logger below, finally prints “leave” before either the returned value or the exception reaches the caller.</p>
  </figure>;
}

export function ContextRouteMap({ state, path, suppress }) {
  const marker = `pmf-context-${useId().replace(/:/g, "")}`;
  const failedEntry = path === "enter-fails", caught = failedEntry || path === "body-fails" && !suppress;
  const active = phase => state.phase === phase ? "pmf-route-active" : "";
  return <div className="pmf-context-route" data-python-figure="context-route">
    <p className="pmf-caption">Current step: <strong>{state.phase === "before" ? "before entry" : state.phase}</strong></p>
    <svg viewBox="0 0 370 361" role="img" aria-label={`Selected control route: ${failedEntry ? "enter fails, skip body and exit, reach outer exception handler" : `enter succeeds, run body, exit closes resource, ${caught ? "propagate to outer handler" : "continue after with"}`}. Current phase: ${state.phase}. Resource: ${state.resource}. Exception: ${state.error || "none"}.`}>
      <ArrowMarker id={marker}/>
      <g className={active("enter")}><rect className="pmf-route-node" x="5" y="18" width="180" height="61" rx="3"/><text x="17" y="43">__enter__</text><text className="pmf-svg-label" x="17" y="65">{failedEntry ? "raise error" : "open + return"}</text></g>
      <path className={`pmf-arrow ${failedEntry ? "pmf-route-skipped" : ""}`} d="M95,80 L95,118" markerEnd={`url(#${marker})`}/>
      {!failedEntry && <text x="105" y="103" className="pmf-svg-label">successful entry</text>}
      <g className={`${failedEntry ? "pmf-route-skipped" : ""} ${active("body")}`}><rect className="pmf-route-node" x="5" y="124" width="180" height="62" rx="3"/><text x="17" y="150">with body</text><text className="pmf-svg-label" x="17" y="173">{failedEntry ? "SKIPPED" : path === "body-fails" ? "ValueError" : "reads one line"}</text></g>
      <path className={`pmf-arrow ${failedEntry ? "pmf-route-skipped" : ""}`} d="M95,187 L95,223" markerEnd={`url(#${marker})`}/>
      <g className={`${failedEntry ? "pmf-route-skipped" : ""} ${active("exit")}`}><rect className="pmf-route-node" x="5" y="230" width="180" height="62" rx="3"/><text x="17" y="254">__exit__</text><text className="pmf-svg-label" x="17" y="276">{failedEntry ? "NOT CALLED" : "close file"}</text></g>
      {failedEntry ? <><path className="pmf-arrow pmf-route-error" d="M187,46 H321 V311" markerEnd={`url(#${marker})`}/><text x="202" y="109" className="pmf-svg-label">entry error</text><text x="202" y="131" className="pmf-svg-label">bypasses</text><text x="202" y="153" className="pmf-svg-label">body + exit</text></> : <><path className="pmf-arrow" d="M187,260 H274 V311" markerEnd={`url(#${marker})`}/><text x="202" y="199" className="pmf-svg-label">{path === "body-fails" ? suppress ? "True: suppress" : "False: propagate" : "normal completion"}</text><text x="202" y="221" className="pmf-svg-label">after cleanup</text></>}
      <g className={active("outside")}><rect className="pmf-route-node" x="161" y="318" width="204" height="36" rx="3"/><text x="263" y="341" textAnchor="middle" className="pmf-svg-label">{caught ? "OUTER HANDLER" : "AFTER with"}</text></g>
    </svg>
    <div className="pmf-resource-state"><span><strong>File:</strong> {state.resource}</span><span><strong>Exception:</strong> {state.error || "none active"}</span></div>
    <p className="pmf-caption">Nodes name operations on the selected route, including steps not reached yet. The outline marks the current step; the file/exception readout shows its state now. Suppression continues after with, never inside the interrupted body. Spacing does not measure time.</p>
  </div>;
}
