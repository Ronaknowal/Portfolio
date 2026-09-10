import "./workflow-figures.css";

export function ApiCheckingPicture() {
  return <figure className="wv-figure" data-visual="api-checking-paths">
    <div className="wv-call-source"><span>One source file</span><code>def double(value: float) -&gt; float:<br/>&nbsp;&nbsp;&nbsp;&nbsp;return value * 2<br/><br/>double("ha")</code></div>
    <div className="wv-check-fork"><div><span className="wv-branch-label">↙ Inspect source with a type checker</span><strong>Does the call match the annotation?</strong><div className="wv-type-match"><code>"ha" : str</code><b>≠</b><code>expected float</code></div><p>Report an incompatible argument.</p></div><div><span className="wv-branch-label">↘ Execute source with ordinary Python</span><strong>What does the operation actually do?</strong><div className="wv-type-match"><code>"ha" * 2</code><b>→</b><code>"haha"</code></div><p>Repeat the string; no automatic annotation check.</p></div></div>
    <figcaption>These are separate ways to use the same file, not two automatic gates in a Python call. A project may require a checker to pass before running; Python itself still needs explicit validation for rules such as finite values or milliseconds.</figcaption>
  </figure>;
}
