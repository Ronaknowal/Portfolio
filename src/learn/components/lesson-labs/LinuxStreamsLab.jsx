import { useId, useState } from "react";
import { LINUX_STREAM_ROUTES, linuxStreamModel } from "../../data/linux-stream-model";
import "./linux-streams-lab.css";
function FlowArrow({
  label
}) {
  return <div className="linux-stream-arrow" aria-hidden="true"><span>{label}</span><i /></div>;
}
function StreamRoute({
  stream,
  destination,
  text,
  pipe = false
}) {
  return <div className={`linux-stream-route linux-stream-route--${stream}`}>
    <div className="linux-stream-port">
      <span className="linux-stream-small">Program writes to</span>
      <strong>{stream} <span>({stream === "stdout" ? "1" : "2"})</span></strong>
      <code>{text.trim()}</code>
    </div>
    <FlowArrow label={"goes to"} />
    <div className="linux-stream-destination">
      <>
        <span className="linux-stream-small">{pipe ? "Next program's input" : destination === "terminal" ? "On screen" : "Saved file"}</span>
        <strong>{pipe ? "wc -l · stdin (0)" : destination === "terminal" ? "Your terminal" : destination}</strong>
        <span>{pipe ? "Receives 1 newline" : destination === "terminal" ? "Visible immediately" : "Receives this stream"}</span>
      </>
    </div>
    {pipe && true && <div className="linux-stream-pipe-result"><FlowArrow label="stdout (1)" /><div className="linux-stream-destination"><span className="linux-stream-small">On screen</span><strong>Your terminal</strong><code>1</code></div></div>}
  </div>;
}
export default function LinuxStreamsLab() {
  const id = useId();
  const [routeId, setRouteId] = useState("stdout-file");
  const [fails, setFails] = useState(false);
  const model = linuxStreamModel(routeId, fails ? 7 : 0);
  const chooseRoute = next => {
    setRouteId(next);
  };
  const reset = () => {
    chooseRoute("stdout-file");
    setFails(false);
  };
  return <section className="lesson-lab linux-stream-lab" data-live-exploration aria-labelledby={`${id}-title`}>
    <span className="lesson-eyebrow">STREAM ROUTING LAB</span>
    <h3 id={`${id}-title`}>Two outputs. Choose where each goes.</h3>
    <p>A program has two output channels: it writes one metric to <strong>stdout</strong> and one warning to <strong>stderr</strong>. Keep the messages fixed and change the connections the shell makes.</p>
    <p className="lesson-note">This is a browser model of six fixed commands. It does not run a shell or write files; each trace starts with fresh modeled destinations.</p>
    <fieldset className="linux-stream-presets">
      <legend>Choose a routing instruction</legend>
      <div>{LINUX_STREAM_ROUTES.filter(route => !route.advanced).map(route => <button key={route.id} type="button" aria-pressed={routeId === route.id} onClick={() => chooseRoute(route.id)}>{route.label}<code>{route.suffix || "no redirection"}</code></button>)}</div>
    </fieldset>

    <details className="linux-stream-deeper">
      <summary>Deeper: why redirection order matters</summary>
      <p><code>2&gt;&amp;1</code> gives stderr the destination stdout has <em>at that moment</em>. Compare these two instructions from left to right.</p>
      <div className="linux-stream-order">{LINUX_STREAM_ROUTES.filter(route => route.advanced).map(route => <button key={route.id} type="button" aria-pressed={routeId === route.id} onClick={() => chooseRoute(route.id)}>{route.label}<code>{route.suffix}</code></button>)}</div>
    </details>

    <details className="linux-stream-deeper">
    <summary>See the complete Bash command</summary>
    <p className="linux-stream-command-label">The complete command · <strong>{model.label}</strong></p>
    <pre className="linux-stream-command"><code>{model.command}</code></pre>
    <p className="lesson-note">Inside this small program, <code>&gt;&amp;2</code> sends the warning to stderr. The instruction after the closing quote sets the destinations for the whole program.</p>
    </details>

    
    <div className="linux-stream-actions"><button type="button" onClick={reset}>Reset streams</button></div>

    <figure className="linux-stream-diagram" aria-label={model.summary}>
      <div className="linux-stream-source"><strong>One program</strong><span>Two separately routable streams</span></div>
      <StreamRoute stream="stdout" text={model.stdoutText} destination={model.stdout} pipe={model.id === "pipe"} />
      <StreamRoute stream="stderr" text={model.stderrText} destination={model.stderr} />
      <figcaption>Solid gold = stdout; dashed pale blue = stderr. Arrows show connections, not elapsed time.</figcaption>
    </figure>

    <div className="linux-stream-feedback" aria-live="polite" aria-atomic="true"><p><strong>{"Follow descriptor 2. "}</strong>{model.explanation}</p></div>

    <div className="linux-stream-results">
      <h4>What arrives at each destination</h4>
      <div className="linux-stream-output-grid">
        <div className="linux-stream-output"><strong>Your terminal</strong>{model.terminal.length ? model.terminal.map(item => <div key={item.stream}><span className="linux-stream-small">From {item.stream}</span><pre><code>{item.content}</code></pre></div>) : <p>No output from this command.</p>}</div>
        {Object.entries(model.files).map(([filename, content]) => <div className="linux-stream-output" key={filename}><strong>{filename}</strong><pre><code>{content}</code></pre></div>)}
      </div>
      {model.id === "pipe" && <p className="lesson-note">The terminal receives a warning and the count. Their on-screen order can vary because the two programs run concurrently; they are grouped by source here. Only the metric enters wc.</p>}
      <div className="linux-stream-status"><span>Program exit status <strong>{model.producerStatus}</strong></span><span>{model.id === "pipe" ? "Pipeline status" : "Shell's command status"} <strong>{model.shellStatus}</strong></span></div>
      <p>{fails ? model.id === "pipe" ? "The producer exits 7, but wc succeeds. With Bash's pipefail disabled, the pipeline reports the last command's 0. A successful count does not prove the producer succeeded." : "The program exits 7 even though it produced output. Changing the destination of a message does not repair a failure." : "The warning is diagnostic text. This program deliberately exits 0, so the warning alone does not mean the command failed."}</p>
    </div>

    <details className="linux-stream-deeper">
      <summary>Deeper: can useful output come from a failed command?</summary>
      <p>Try “Count result lines”, then make the producer fail after printing. Watch the count and reported status as you change the exit behavior.</p>
      <label className="linux-stream-failure"><input type="checkbox" checked={fails} onChange={event => {
          setFails(event.target.checked);
        }} />Exit 7 after printing the same messages</label>
      <p className="lesson-note">This lab assumes successful file access, a successful wc, and Bash with pipefail disabled. A real redirection or program can fail for other reasons. pipefail and error handling are explored further in the Bash lesson.</p>
    </details>

    <details className="linux-stream-deeper">
      <summary>Try it in your own terminal</summary>
      <p>This browser lab models six fixed commands; it does not execute a shell or write files. To reproduce the selected command, start a disposable directory in a Linux Bash terminal:</p>
      <pre className="linux-stream-command"><code>{'lab=$(mktemp -d)\ncd "$lab" || exit 1\nset +o pipefail'}</code></pre>
      <p>Run the complete command shown above. Immediately run <code>{'printf "status: %s\\n" "$?"'}</code> to inspect its status. Use <code>cat</code> with the output filename to inspect any file it created.</p>
      <p><code>&gt;</code> creates or truncates its destination before the program runs. Use this new directory so existing work is not overwritten. Each browser trace starts with fresh modeled files.</p>
    </details>
    <p className="lesson-note">Transfer: you need a machine-readable metric file and a separate warning log. Choose a route, then explain why plain <code>| wc -l</code> would leave the warning visible. Compare the routes directly in the diagram.</p>
  </section>;
}
