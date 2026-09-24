import { csvBoundarySource, csvBoundaryFields } from '../../data/csv-parsing-model.js';
import './scientific-concept-visuals.css';

export function CsvBoundaryDiagram() {
  const split = csvBoundarySource.split(',');
  return <figure className="sci-visual sci-csv-boundaries" aria-label="A quoted comma belongs inside the site field">
    <figcaption><strong>A comma can be data or a boundary</strong><span>The same source record, interpreted two ways.</span></figcaption>
    <div className="sci-csv-source"><code>001<span>,</span>18.5<span>,</span>C<span>,</span><mark>"room,north"</mark></code></div>
    <div className="sci-field-row"><strong>CSV reader · 4 fields</strong><div>{csvBoundaryFields.map((field, i) => <span key={i}><small>{['sample_id', 'temperature', 'unit', 'site'][i]} · string</small><code>{field}</code></span>)}</div></div>
    <div className="sci-field-row sci-field-row--wrong"><strong>split(",") · 5 pieces</strong><div>{split.map((field, i) => <span key={i}><small>piece {i + 1} · string</small><code>{field}</code></span>)}</div></div>
    <p className="sci-caption">The quotes group the room name for the CSV parser. The parser removes those enclosing quotes and retains the inner comma. The naive split has no quoting rule; it also leaves fragments of the quote characters. Both outputs still contain strings, including "18.5".</p>
  </figure>;
}
