import "./workflow-figures.css";

export function BashArgumentPicture({ trace, quoted, step }) {
  return <figure className="wv-figure wv-lab-figure" data-visual="bash-argument-boundaries">
    <div className="wv-command-source"><span>Written command</span><code>{quoted ? 'show_args "$value"' : 'show_args $value'}</code></div>
    <div className="wv-expansion-source"><span>One stored string</span><code>{JSON.stringify(trace.value)}</code></div>
    <div className="wv-argument-transfer">↓ {step < 2 ? 'Follow the expansion into argument boundaries' : quoted ? 'Keep one quoted word' : 'Split fields, then expand filename patterns'}</div>
    <div className="wv-argument-receiver"><strong>{step < 3 ? 'Words prepared for the call' : `show_args receives ${trace.argv.length} positional argument${trace.argv.length === 1 ? '' : 's'}`}</strong>
      {step < 2 ? <p className="wv-awaiting">Advance to expansion to reveal the words.</p> : <ol className="wv-arguments">{trace.argv.length ? trace.argv.map((value, index) => <li key={index}><small>argument {index + 1}</small><code>{JSON.stringify(value)}</code>{value === '' && <span>One empty argument</span>}</li>) : <li className="wv-no-arguments">No arguments · the empty unquoted expansion disappeared</li>}</ol>}
    </div>
    <figcaption>Each outlined item is one received string, not one word as a human might read it. Spaces inside a quoted value stay inside the same boundary. Quotes shown around the values are notation for strings; they are not characters added to the argument.</figcaption>
  </figure>;
}

export function BashPipelinePicture({ upstream, downstream, strict, status }) {
  return <figure className="wv-figure wv-lab-figure" data-visual="bash-output-status-channels">
    <div className="wv-pipe-flow"><div className="wv-process"><strong>Producer</strong><code>stdout: row 1<br/>row 2</code></div><div className="wv-pipe"><span>stdout → stdin</span><b aria-hidden="true">→ → →</b><small>bytes flow through the pipe</small></div><div className="wv-process"><strong>Consumer</strong><span>Reads the emitted rows</span></div></div>
    <div className="wv-status-flow"><div><span>Producer exits</span><output>{upstream}</output></div><div className="wv-status-rule"><span>Parent chooses pipeline status</span><code>{strict ? 'rightmost nonzero, or 0' : 'last command only'}</code></div><div><span>Consumer exits</span><output>{downstream}</output></div></div>
    <div className="wv-pipeline-result">↘ <code>[{upstream}, {downstream}]</code> → <strong>{status}</strong> ↙</div>
    <figcaption>Read across the upper channel for data; read the exit codes below for success or failure. Switching pipefail changes how the parent combines statuses. It does not retrieve bytes already sent or turn a partial result into a complete one. This fixed example emits the two rows before returning either selected producer status.</figcaption>
  </figure>;
}
