import { sensorValues, selectionResult } from "../../data/numpy-foundations-model";
import './scientific-concept-visuals.css';

function ReadingCells({ source, columns }) {
  return <div className="sci-reading-cells" style={{ '--sci-columns': columns }}>{source.map(index => <span key={index} className={index % 2 ? 'sci-sensor-b' : 'sci-sensor-a'}><strong>{sensorValues[index]}</strong><small>{index % 2 ? 'B' : 'A'} · time {Math.floor(index / 2)}</small></span>)}</div>;
}

export function NumpyShapeComparison() {
  return <figure className="sci-visual sci-shape-comparison" aria-label="Same readings, two different coordinate maps">
    <figcaption><strong>Same (2, 3) shape. Different neighbours.</strong><span>Every value keeps its original sensor and time label below.</span></figcaption>
    <div className="sci-shape-source"><code>X · (3, 2)</code><ReadingCells source={[0, 1, 2, 3, 4, 5]} columns={2} /></div>
    <div className="sci-shape-branches"><div><p><span aria-hidden="true">↙ </span><code>X.T</code> · swap axes</p><ReadingCells source={selectionResult('transpose').source} columns={3} /><p>Each row follows <strong>one sensor</strong> through times 0, 1 and 2.</p></div><div><p><span aria-hidden="true">↘ </span><code>X.reshape(2, 3)</code></p><ReadingCells source={selectionResult('reshape').source} columns={3} /><p>C-order regroups the row-by-row sequence. The new rows <strong>mix sensors</strong>.</p></div></div>
    <p className="sci-caption">Follow 24 °C: it becomes [0, 1] after transpose and [0, 2] after reshape. This drawing maps logical coordinates, not physical memory movement.</p>
  </figure>;
}
