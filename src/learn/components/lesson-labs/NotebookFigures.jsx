import "./workflow-figures.css";

export function NotebookStatePicture() {
  return <figure className="wv-figure" data-visual="notebook-three-stores">
    <div className="wv-notebook-picture">
      <div className="wv-notebook-document">
        <strong>Notebook document · after an edit</strong>
        <div className="wv-notebook-cell"><span>Input cell</span><code>values = [10, 20, 30]<br/>offset = <del>2</del> <mark>5</mark></code><small>Edited, not executed</small></div>
        <div className="wv-notebook-cell"><span>Calculation cell</span><code>mean = sum(values) / len(values) - offset</code></div>
        <div className="wv-saved-output"><span>Displayed output from the earlier run</span><output>18.0 ms</output></div>
        <div className="wv-save-bracket">Save .ipynb → source + retained output</div>
      </div>
      <div className="wv-kernel-picture">
        <strong>Live kernel · last executed values</strong>
        <dl><dt>offset</dt><dd>2</dd><dt>mean</dt><dd>18.0</dd></dl>
        <p>Editing has sent no new instruction to this process.</p>
        <div className="wv-kernel-boundary">Live objects are outside the saved notebook.</div>
      </div>
    </div>
    <figcaption>Same example, three states: the source now says 5, the kernel still stores offset 2, and the old output still says 18.0. Saving can preserve this disagreement. To obtain 15.0, execute the new input, recalculate, then display. The picture shows a notebook that retains outputs; saving is not a snapshot of all Python memory.</figcaption>
  </figure>;
}
