import { useId, useState } from 'react';
import { applyMatrix, dot, formatVectorNumber as number, linearMapPresets, matrixProduct, measurementTensor, parseSmallMatrix, productCell, projectVector, reduceMeasurementTensor, tensorAxisNames } from '../../data/vector-tensor-models.js';
import './vector-tensor-labs.css';
const pair = vector => `(${vector.map(number).join(', ')})`;
const point = ([x, y], [lower, upper] = [-6, 6]) => {
  const scale = 264 / (upper - lower);
  return [38 + (x - lower) * scale, 302 - (y - lower) * scale];
};
function Investigation({
  name,
  eyebrow,
  title,
  children
}) {
  return <section className="vector-investigation" data-lab={name} aria-label={title}>
    <p className="vector-eyebrow">{eyebrow}</p><h3>{title}</h3>{children}
  </section>;
}
function CoordinatePlane({
  label,
  vectors = [],
  segments = [],
  polygon,
  domain = [-6, 6],
  children
}) {
  const markerId = useId().replaceAll(':', '');
  const position = vector => point(vector, domain);
  const origin = position([0, 0]);
  const step = domain[1] - domain[0] > 8 ? 2 : 1;
  const ticks = [];
  for (let value = Math.ceil(domain[0] / step) * step; value <= domain[1]; value += step) ticks.push(value);
  return <svg className="vector-coordinate-plane" viewBox="0 0 340 340" role="img" aria-label={label} data-domain={domain.join(',')}>
    <title>{label}</title>
    <defs><marker id={markerId} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0,0 L7,3.5 L0,7" fill="context-stroke" /></marker></defs>
    {ticks.map(value => <g key={value} className="vector-grid-line">
      <line x1={position([value, 0])[0]} y1="38" x2={position([value, 0])[0]} y2="302" />
      <line x1="38" y1={position([0, value])[1]} x2="302" y2={position([0, value])[1]} />
    </g>)}
    <path d={`M25,${origin[1]} H318 M${origin[0]},315 V25`} className="vector-coordinate-axis" />
    <text x="317" y={origin[1] - 12}>x</text><text x={origin[0] + 11} y="29">y</text><text x={origin[0] - 16} y={origin[1] + 18}>0</text>
    {ticks.filter(value => value !== 0).map(value => <g key={value} className="vector-tick-label"><text x={position([value, 0])[0]} y={origin[1] + 18} textAnchor="middle">{value}</text><text x={origin[0] - 17} y={position([0, value])[1] + 5} textAnchor="end">{value}</text></g>)}
    {polygon && <polygon className="vector-unit-region" points={polygon.map(vector => position(vector).join(',')).join(' ')} />}
    {segments.map((segment, index) => <line key={index} x1={position(segment.from)[0]} y1={position(segment.from)[1]} x2={position(segment.to)[0]} y2={position(segment.to)[1]} className={`vector-arrow ${segment.kind}`} />)}
    {vectors.map((vector, index) => {
      const start = position(vector.from || [0, 0]);
      const end = position(vector.to);
      return <g key={index}>
        <line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} markerEnd={start[0] === end[0] && start[1] === end[1] ? undefined : `url(#${markerId})`} className={`vector-arrow ${vector.kind}`} />
        <circle cx={end[0]} cy={end[1]} r={vector.kind === 'original' ? 4 : 3} className={`vector-endpoint ${vector.kind}`} />
      </g>;
    })}
    {children}
  </svg>;
}
function NumericMatrix({
  values,
  caption,
  selectedRow,
  selectedColumn,
  selectedCell,
  onSelect,
  sourceKeys
}) {
  return <div className="vector-matrix-scroll" tabIndex={0} role="region" aria-label={caption}>
    <table className="vector-value-matrix"><caption>{caption}</caption><tbody>{values.map((row, rowIndex) => <tr key={rowIndex}>{row.map((value, columnIndex) => {
            const selected = selectedRow === rowIndex || selectedColumn === columnIndex || selectedCell?.[0] === rowIndex && selectedCell?.[1] === columnIndex;
            return <td key={columnIndex} className={selected ? 'is-selected' : ''} data-source={sourceKeys?.[rowIndex]?.[columnIndex]}>
        {onSelect ? <button type="button" aria-label={`Output row ${rowIndex}, column ${columnIndex}: ${number(value)}`} aria-pressed={selected} onClick={() => onSelect(rowIndex, columnIndex)}>{number(value)}</button> : number(value)}
      </td>;
          })}</tr>)}</tbody></table>
  </div>;
}
export function VectorCombinationFigure() {
  return <figure className="vector-inline-figure">
    <figcaption>Two moves become one displacement</figcaption>
    <CoordinatePlane domain={[-2, 4]} label="Move from (0,0) to (2,1), then add (-1,2) to arrive at (1,3)." vectors={[{
      to: [2, 1],
      kind: 'basis-one'
    }, {
      from: [2, 1],
      to: [1, 3],
      kind: 'basis-two'
    }, {
      to: [1, 3],
      kind: 'original'
    }]} />
    <p><span className="vector-key basis-one">Solid green: u=(2,1)</span> → <span className="vector-key basis-two">dashed blue: v=(−1,2), moved to u's tip</span> → <span className="vector-key original">gold: u+v=(1,3)</span>. The second arrow still means left one/up two; moving its tail does not change that displacement. Axes use the same distance unit and scale.</p>
  </figure>;
}
export function VectorProjectionLab() {
  const controlsId = useId();
  const [horizontal, setHorizontal] = useState(1);
  const [vertical, setVertical] = useState(4);
  const [directionKey, setDirectionKey] = useState('diagonal');
  const vector = [horizontal, vertical];
  const direction = {
    diagonal: [2, 1],
    horizontal: [1, 0],
    zero: [0, 0]
  }[directionKey];
  const result = projectVector(vector, direction);
  const reset = () => {
    setHorizontal(1);
    setVertical(4);
    setDirectionKey('diagonal');
  };
  return <Investigation name="vector-projection" eyebrow="ALONG A DIRECTION, THEN WHAT REMAINS" title="A dot product locates a projection">
    <p>Now start with v=(1,4), which makes the perpendicular remainder easier to see. Inspect its signed contribution along u while changing it. Coordinates apply immediately. The dotted connector is v−p; set v=(3,2) to revisit the hand calculation above.</p>
    <div className="vector-controls">
      <label htmlFor={`${controlsId}-v-horizontal`}>v horizontal coordinate <output>{horizontal}</output><input id={`${controlsId}-v-horizontal`} aria-label="v horizontal coordinate" type="range" min="-4" max="4" step="1" value={horizontal} onChange={event => setHorizontal(Number(event.target.value))} /></label>
      <label htmlFor={`${controlsId}-v-vertical`}>v vertical coordinate <output>{vertical}</output><input id={`${controlsId}-v-vertical`} aria-label="v vertical coordinate" type="range" min="-4" max="4" step="1" value={vertical} onChange={event => setVertical(Number(event.target.value))} /></label>
      <label>Direction u<select value={directionKey} onChange={event => setDirectionKey(event.target.value)}><option value="diagonal">u=(2,1)</option><option value="horizontal">u=(1,0)</option><option value="zero">u=(0,0): test the boundary</option></select></label>
    </div>
    <div className="vector-geometric-reading"><CoordinatePlane label={`v=${pair(vector)}, direction u=${pair(direction)}${result.projection ? `, projection p=${pair(result.projection)}, residual=${pair(result.residual)}` : ', projection undefined'}`} vectors={[{
        to: direction,
        kind: 'basis-two'
      }, ...(result.projection ? [{
        to: result.projection,
        kind: 'basis-one'
      }] : []), {
        to: vector,
        kind: 'original'
      }]} segments={result.projection ? [{
        from: result.projection,
        to: vector,
        kind: 'residual'
      }] : []} />
      <div className="vector-result" aria-live="polite">
        <p className="vector-key original">Gold v={pair(vector)}</p><p className="vector-key basis-two">Dashed blue u={pair(direction)}</p>
        <p data-result="dot">v·u = {horizontal}×{direction[0]} + {vertical}×{direction[1]} = <strong>{number(result.product)}</strong></p>
        {result.projection ? <>
          <p>Coefficient = {number(result.product)} / {number(dot(direction, direction))} = {number(result.coefficient)}</p>
          <p className="vector-key basis-one" data-result="projection">Solid green p = {pair(result.projection)}</p>
          <p data-result="residual">Dotted residual v−p = {pair(result.residual)}</p>
          <p>Residual · u = {number(dot(result.residual, direction))}</p>
          <p>Cosine: {result.cosine === null ? 'undefined because v is zero' : number(result.cosine)}</p>
        </> : <p role="status">There is no direction to project onto: u·u=0, so division would be undefined. The dot product is still zero.</p>}
      </div>
    </div>
    <p className="vector-note">Equal coordinate scales; positions are exact calculations, displayed decimals rounded to three places. Coincident or zero arrows may overlap; the labelled coordinates remain available. A negative coefficient places p on the opposite half of u's line.</p>
    <button type="button" onClick={reset}>Reset projection</button>
    <details><summary>Try a changed case</summary><p>Use v=(−1,2), u=(2,1): dot=0, p=(0,0), and the entire v is perpendicular residual. Change v to (−2,−1): p=v and the coefficient is −1. Zero residual means v lies on the line, including its negative direction.</p></details>
  </Investigation>;
}
export function LinearMapLab() {
  const controlsId = useId();
  const [preset, setPreset] = useState('shear');
  const [horizontal, setHorizontal] = useState(2);
  const [vertical, setVertical] = useState(1);
  const matrix = linearMapPresets[preset].matrix;
  const vector = [horizontal, vertical];
  const output = applyMatrix(matrix, vector);
  const first = applyMatrix(matrix, [1, 0]);
  const second = applyMatrix(matrix, [0, 1]);
  const contribution = first.map(value => value * horizontal);
  const square = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const plottedCoordinates = [...vector, ...output, ...first, ...second, ...contribution, ...square.flat(), ...square.flatMap(corner => applyMatrix(matrix, corner))];
  const domain = [Math.min(-1, Math.floor(Math.min(...plottedCoordinates)) - 1), Math.max(2, Math.ceil(Math.max(...plottedCoordinates)) + 1)];
  const reset = () => {
    setPreset('shear');
    setHorizontal(2);
    setVertical(1);
  };
  return <Investigation name="linear-map" eyebrow="THE COLUMNS TELL YOU WHERE UNIT STEPS GO" title="Transform a whole shape using two column images">
    <p>Inspect the destination of x=(2,1) under the shear. Then try a map that collapses the square: can you still recover every original input? Each selection applies immediately; both plots use the same scale.</p>
    <div className="vector-controls">
      <label>Linear map<select value={preset} onChange={event => setPreset(event.target.value)}>{Object.entries(linearMapPresets).map(([key, value]) => <option key={key} value={key}>{value.name}</option>)}</select></label>
      <label htmlFor={`${controlsId}-x-first`}>x first coordinate <output>{horizontal}</output><input id={`${controlsId}-x-first`} aria-label="x first coordinate" type="range" min="-3" max="3" value={horizontal} onChange={event => setHorizontal(Number(event.target.value))} /></label>
      <label htmlFor={`${controlsId}-x-second`}>x second coordinate <output>{vertical}</output><input id={`${controlsId}-x-second`} aria-label="x second coordinate" type="range" min="-3" max="3" value={vertical} onChange={event => setVertical(Number(event.target.value))} /></label>
    </div>
    <div className="vector-two-planes"><div><h4>Input coordinates</h4><CoordinatePlane domain={domain} label={`Unit square and input x=${pair(vector)}`} polygon={square} vectors={[{
          to: [1, 0],
          kind: 'basis-one'
        }, {
          to: [0, 1],
          kind: 'basis-two'
        }, {
          to: vector,
          kind: 'original'
        }]} /></div><div><h4>Output coordinates</h4><CoordinatePlane domain={domain} label={`Transformed unit square, A e1=${pair(first)}, A e2=${pair(second)}, Ax=${pair(output)}`} polygon={square.map(corner => applyMatrix(matrix, corner))} vectors={[{
          to: first,
          kind: 'basis-one'
        }, {
          to: second,
          kind: 'basis-two'
        }, {
          to: output,
          kind: 'original'
        }]} segments={[{
          from: [0, 0],
          to: contribution,
          kind: 'residual'
        }, {
          from: contribution,
          to: output,
          kind: 'residual'
        }]} /></div></div>
    <div className="vector-matrix-equation"><NumericMatrix values={matrix} caption="A · rows are output coordinates" /><div className="vector-result" aria-live="polite">
      <p className="vector-key basis-one">Column 1 = A e₁ = {pair(first)}</p><p className="vector-key basis-two">Column 2 = A e₂ = {pair(second)}</p>
      <p data-result="map-output"><strong>Ax = {horizontal}×{pair(first)} + {vertical}×{pair(second)} = {pair(output)}</strong></p>
      <p>Row 1: {matrix[0][0]}×{horizontal} + {matrix[0][1]}×{vertical} = {number(output[0])}<br />Row 2: {matrix[1][0]}×{horizontal} + {matrix[1][1]}×{vertical} = {number(output[1])}</p>
    </div></div>
    <p className="vector-note">Green solid and blue dashed arrows track unit directions; gold tracks x and Ax. The shaded region is the unit square's image. Both plots share equal x/y scale and the same range; changing inputs may refit both ranges together. Dotted segments add scaled columns head to tail. Collapse to a line or point has zero shaded area; exact coordinates distinguish coincident arrows.</p>
    <button type="button" onClick={reset}>Reset map</button>
    <details><summary>What does collapse discard?</summary><p>For A=[[1,1],[0,0]], both (1,0) and (0,1) become (1,0). The whole direction (1,−1) becomes zero. No operation on the output alone can recover which of these inputs you supplied. The zero map loses every direction.</p></details>
  </Investigation>;
}
export function MatrixProductLab() {
  const defaults = {
    left: [[2, 1], [0, 3], [1, 2]],
    right: [[3, 2], [5, 4]]
  };
  const [leftText, setLeftText] = useState('2,1;0,3;1,2');
  const [rightText, setRightText] = useState('3,2;5,4');
  const [active, setActive] = useState(defaults);
  const [selection, setSelection] = useState([0, 0]);
  const [terms, setTerms] = useState(0);
  const [error, setError] = useState('');
  const result = productCell(active.left, active.right, selection[0], selection[1], terms);
  const apply = event => {
    event.preventDefault();
    try {
      const left = parseSmallMatrix(leftText);
      const right = parseSmallMatrix(rightText);
      matrixProduct(left, right);
      setActive({
        left,
        right
      });
      setSelection([0, 0]);
      setTerms(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  };
  const reset = () => {
    setLeftText('2,1;0,3;1,2');
    setRightText('3,2;5,4');
    setActive(defaults);
    setSelection([0, 0]);
    setTerms(0);
    setError('');
  };
  return <Investigation name="matrix-product" eyebrow="ONE OUTPUT, ONE ROW, ONE COLUMN" title="Watch the shared index get summed away">
    <p>Select an output cell, predict its value and add its contributions one at a time. The starting left matrix is orders×products; the right is products×resources. Indices shown here start at zero.</p>
    <form className="vector-controls" onSubmit={apply}>
      <label>Left matrix · commas within rows; semicolons between rows<input value={leftText} onChange={event => setLeftText(event.target.value)} /></label>
      <label>Right matrix<input value={rightText} onChange={event => setRightText(event.target.value)} /></label>
      <button type="submit">Apply matrices</button>
    </form>
    {error && <p className="vector-error" role="alert">{error} The active matrices and trace were kept.</p>}
    <p className="vector-note">Draft text changes only after Apply matrices. Maximum 3×3, values −20…20. Custom entries are numerical examples; assign their own axis meanings before interpreting them as orders.</p>
    <div className="vector-product-grid">
      <NumericMatrix values={active.left} selectedRow={selection[0]} caption={`Left (${active.left.length},${active.left[0].length}) · selected row ${selection[0]}`} />
      <span className="vector-operation-symbol">×</span>
      <NumericMatrix values={active.right} selectedColumn={selection[1]} caption={`Right (${active.right.length},${active.right[0].length}) · selected column ${selection[1]}`} />
      <span className="vector-operation-symbol">=</span>
      <NumericMatrix values={result.result} selectedCell={selection} caption="Full product · select a cell to inspect" onSelect={(row, column) => {
        setSelection([row, column]);
        setTerms(0);
      }} />
    </div>
    <ol className="vector-product-terms">{result.contributions.map(term => <li key={term.index} className={term.index < terms ? 'is-included' : ''}>k={term.index}: {number(term.left)} × {number(term.right)} = {number(term.product)} <span>{term.index < terms ? 'included' : 'not added yet'}</span></li>)}</ol>
    <p className="vector-result" aria-live="polite" data-result="partial">Partial sum: <strong>{number(result.partial)}</strong> · {terms} of {active.right.length} contributions added. {terms === active.right.length ? 'This now equals the selected full-product cell.' : 'The full product above is the destination, not the current partial sum.'}</p>
    <div className="vector-step-controls"><button type="button" disabled={terms === 0} onClick={() => setTerms(value => value - 1)}>Remove last term</button><button type="button" disabled={terms === active.right.length} onClick={() => setTerms(value => value + 1)}>Add next term</button><button type="button" onClick={reset}>Reset product</button></div>
    <details><summary>Try a compatible shape with a different meaning</summary><p>Use left 1,2,3 and right 1;0;−1. The product has shape (1,1) and value −2. Elementwise multiplication cannot replace this weighted sum: it would preserve or broadcast positions instead of summing the shared three-coordinate index.</p></details>
  </Investigation>;
}
export function TensorReductionLab() {
  const [dataset, setDataset] = useState('ramp');
  const [axis, setAxis] = useState(1);
  const [selection, setSelection] = useState([0, 0]);
  const tensor = measurementTensor(dataset);
  const reduction = reduceMeasurementTensor(tensor, axis);
  const selected = reduction.cells[selection[0]][selection[1]];
  const sources = new Set(selected.sources.map(source => source.coordinate.join(',')));
  const reset = () => {
    setDataset('ramp');
    setAxis(1);
    setSelection([0, 0]);
  };
  return <Investigation name="tensor-reduction" eyebrow="HOLD THE SURVIVING INDICES FIXED" title="An average removes a named axis">
    <p>This invented measurement array has shape (2 sessions, 2 times, 3 channels), all in the same arbitrary unit. Inspect which values contribute to the highlighted output. Changing an axis or dataset applies immediately and selects the first output cell.</p>
    <div className="vector-controls"><label>Dataset<select value={dataset} onChange={event => {
          setDataset(event.target.value);
          setSelection([0, 0]);
        }}><option value="ramp">Distinct values 0…11</option><option value="repeat">Repeat the same session</option><option value="impulse">One reading of 12, all others zero</option></select></label><label>Average over<select value={axis} onChange={event => {
          setAxis(Number(event.target.value));
          setSelection([0, 0]);
        }}>{tensorAxisNames.map((name, index) => <option key={name} value={index}>axis {index}: {name}</option>)}</select></label></div>
    <div className="vector-tensor-slices">{tensor.map((session, sessionIndex) => <div key={sessionIndex}><h4>Session {sessionIndex}</h4><p className="vector-note">rows: time 0,1 · columns: channel 0,1,2</p><table className="vector-tensor-source"><tbody>{session.map((row, time) => <tr key={time}>{row.map((value, channel) => <td key={channel} className={sources.has([sessionIndex, time, channel].join(',')) ? 'is-selected' : ''} data-coordinate={[sessionIndex, time, channel].join(',')}><strong>{value}</strong><small>({sessionIndex},{time},{channel})</small></td>)}</tr>)}</tbody></table></div>)}</div>
    <NumericMatrix values={reduction.cells.map(row => row.map(cell => cell.mean))} selectedCell={selection} onSelect={(row, column) => setSelection([row, column])} caption={`Mean shape (${reduction.shape.join(',')}) · rows: ${reduction.axisNames[0]}, columns: ${reduction.axisNames[1]}`} />
    <p className="vector-result" aria-live="polite" data-result="reduction">For output ({selection.join(',')}): ({selected.sources.map(source => source.value).join(' + ')}) / {selected.sources.length} = <strong>{number(selected.mean)}</strong>. Only the {tensorAxisNames[axis]} index varies.</p>
    <p className="vector-note">Outlined source cells supply that one mean; parenthesized labels are (session,time,channel), not values. Removing the session axis combines sessions. Removing time keeps sessions separate. Removing channel mixes channels and is meaningful only if the channel-average answers your question.</p>
    <button type="button" onClick={reset}>Reset tensor</button>
    <details><summary>Try a changed dataset</summary><p>Choose the single reading of 12. Averaging time divides it by 2; averaging channels divides it by 3. Averaging sessions also divides it by 2, but its surviving labels are time/channel instead of session/channel. Equal output sizes do not imply equal questions.</p></details>
  </Investigation>;
}
export function TensorReindexFigure() {
  return <figure className="vector-inline-figure"><figcaption>Same six values, two different rearrangements</figcaption>
    <div className="vector-reindex-grid">
      <NumericMatrix values={[[0, 1, 2], [3, 4, 5]]} caption="Original (2,3)" />
      <NumericMatrix values={[[0, 3], [1, 4], [2, 5]]} caption="Transpose (3,2) · swap row and column" />
      <NumericMatrix values={[[0, 1], [2, 3], [4, 5]]} caption="C-order reshape (3,2) · regroup the row-wise sequence" />
    </div><p>Follow value 3: (row 1,column 0) → (row 0,column 1) under transpose; → (row 1,column 1) under reshape. Both results have shape (3,2). Their coordinate meanings differ. Reshape's order here is NumPy's default C index order; it is not a claim about physical storage.</p>
  </figure>;
}
export function CompositionFigure() {
  return <figure className="vector-inline-figure"><figcaption>Changing the order changes the destination</figcaption>
    <div className="vector-composition-lanes"><p><strong>Start (1,1)</strong><span>stretch horizontal ×2 → (2,1)</span><span>quarter-turn → <strong>(−1,2)</strong></span></p><p><strong>Start (1,1)</strong><span>quarter-turn → (−1,1)</span><span>stretch horizontal ×2 → <strong>(−2,1)</strong></span></p></div>
    <p>Both chains use the same two maps. The stretch acts along the fixed horizontal axis, so rotating before it changes which component is doubled. For column vectors, the rightmost matrix acts first.</p>
  </figure>;
}
