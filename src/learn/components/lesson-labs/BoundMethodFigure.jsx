import "./python-mechanism-figures.css";

export function BoundMethodRetentionFigure() {
  return <figure className="pmf-figure" data-python-figure="bound-method">
    <figcaption><strong>A saved method keeps its receiver when a name moves.</strong><span>State after add_to_morning = morning.add, then morning = evening.</span></figcaption>
    <div className="pmf-bound-map">
      <div className="pmf-bound-pair"><code>add_to_morning</code><span className="pmf-link">↓ refers to a bound method</span><dl><dt>Function · __func__</dt><dd><code>ReadingLog.add</code></dd><dt>Receiver · __self__</dt><dd><strong>↓ object A</strong></dd></dl><div className="pmf-log"><small>ORIGINAL MORNING LOG · A</small><code>values → []</code></div></div>
      <div className="pmf-bound-current"><code>morning</code><code>evening</code><span className="pmf-link">↓ both now refer to</span><div className="pmf-log"><small>EVENING LOG · B</small><code>values → []</code></div></div>
    </div>
    <p className="pmf-caption"><code>add_to_morning(18)</code> supplies <strong>A</strong> as self, so A's list becomes [18]. B's list remains empty. The saved method keeps an object reference, not an instruction to look up the name morning again.</p>
  </figure>;
}
