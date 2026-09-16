import { useState } from 'react';
import { samplingPopulation, finiteSampleState, inclusionDesignState, groupedMeasurementState, assignmentState, factorialState, boundedMissingMean, formatSampling as f } from '../../data/sampling-measurement-models.js';
import './sampling-measurement-labs.css';
function Slider({
  label,
  value,
  minimum,
  maximum,
  step = 1,
  onChange
}) {
  return <label>{label}: {value}<input aria-label={label} type="range" min={minimum} max={maximum} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function ScrollTable({
  label,
  headers,
  rows
}) {
  return <div className="sampling-table-scroll" role="region" tabIndex={0} aria-label={label}><table><caption>{label}</caption><thead><tr>{headers.map(header => <th key={header}>{header}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function Distribution({
  values,
  target,
  expectation,
  selected,
  label,
  limits
}) {
  const buckets = new Map();
  values.forEach(value => {
    const key = value.toFixed(10);
    const previous = buckets.get(key);
    buckets.set(key, {
      value,
      count: (previous?.count || 0) + 1
    });
  });
  const groups = [...buckets.values()].sort((a, b) => a.value - b.value);
  const minimum = limits?.[0] ?? Math.floor(Math.min(...values, target) - 1);
  const maximum = limits?.[1] ?? Math.ceil(Math.max(...values, target) + 1);
  const maximumProbability = Math.max(...groups.map(group => group.count / values.length));
  const x = value => 48 + 260 * (value - minimum) / (maximum - minimum);
  return <div className="sampling-distribution"><svg viewBox="0 0 340 235" role="img" aria-label={label}>
    <text x="48" y="22">Probability</text>
    <path className="axis" d="M48 45V169H313" />
    <text x="42" y="55" textAnchor="end" className="small">{f(maximumProbability, 2)}</text>
    <text x="42" y="170" textAnchor="end" className="small">0</text>
    {groups.map(group => <rect key={group.value} data-value={group.value} data-probability={group.count / values.length} x={x(group.value) - 4} y={169 - 115 * group.count / values.length / maximumProbability} width="8" height={115 * group.count / values.length / maximumProbability} className={Math.abs(group.value - selected) < 1e-9 ? 'selected-mass' : 'mass'} />)}
    <line className="target" x1={x(target)} x2={x(target)} y1="39" y2="175" />
    <circle className="expectation" cx={x(expectation)} cy="177" r="5" />
    {[minimum, (minimum + maximum) / 2, maximum].map(tick => <text key={tick} x={x(tick)} y="200" textAnchor="middle">{f(tick, 2)}</text>)}

  </svg><p className="sampling-axis-label">Horizontal axis: estimate</p><p className="sampling-legend"><span className="rose-key">Dashed: target</span><span className="blue-key">Blue dot: expectation</span><span className="gold-key">Gold: selected estimate</span></p></div>;
}
export function CollectionDesignFigure() {
  return <figure className="sampling-figure collection-map"><div className="collection-stages">
    <div><strong>Target population</strong><div className="unit-strip">{Array.from({
            length: 8
          }, (_, i) => <span key={i}>U{i + 1}</span>)}</div><p>Who should the answer describe?</p></div>
    <div><span className="stage-arrow" aria-hidden="true">↓</span><strong>Available frame</strong><div className="unit-strip">{[1, 2, 3, 4, 5, 6].map(i => <span key={i}>U{i}</span>)}</div><p>U7 and U8 cannot be selected here.</p></div>
    <div><span className="stage-arrow" aria-hidden="true">↓</span><strong>Selected units</strong><div className="unit-strip"><span>U1</span><span>U3</span><span>U4</span><span>U6</span></div><p>A sampling rule chooses who enters.</p></div>
  </div><div className="assignment-branches"><div><strong>Assigned A</strong><span>U1 · U4</span><span>↓ recorded outcomes</span></div><div><strong>Assigned B</strong><span>U3 · U6</span><span>↓ recorded outcomes</span></div></div><figcaption>Assignment operates on the selected units. It cannot restore the two units absent from the frame. A measurement rule then determines what each recorded outcome means. IDs connect the same units across stages; this schematic is not a claim about a real study.</figcaption></figure>;
}
export function MeasurementBiasFigure() {
  const readings = [11, 12, 13, 11, 12, 13];
  const x = value => 45 + (value - 8) * 48;
  return <figure className="sampling-figure"><svg viewBox="0 0 340 210" role="img" aria-label="Six readings from eleven to thirteen average twelve, while the reference value is ten">
    <path className="axis" d="M45 142H307" />
    {[8, 9, 10, 11, 12, 13].map(value => <text key={value} x={x(value)} y="168" textAnchor="middle">{value}</text>)}
    <line className="target" x1={x(10)} x2={x(10)} y1="40" y2="149" />
    {readings.map((value, index) => <circle className="reading" key={index} cx={x(value)} cy={111 - Math.floor(index / 3) * 21} r="7" />)}
    <path className="gold-line" d={`M${x(10)} 44H${x(12)}m-5-4 5 4-5 4`} />
    <text x="183" y="29" textAnchor="middle">offset: +2</text><text x="178" y="200" textAnchor="middle">Measured quantity (mV)</text>
  </svg><figcaption>Reference 10 mV; average of these declared readings 12 mV. Repeatability describes scatter under specified conditions. The 2 mV reference gap is a different question. These six synthetic points illustrate the distinction; they do not establish a real instrument's bias distribution.</figcaption></figure>;
}
export function FiniteSamplesLab() {
  const [frame, setFrame] = useState('complete');
  const [size, setSize] = useState(2);
  const [selected, setSelected] = useState(0);
  const state = finiteSampleState(samplingPopulation, frame === 'complete' ? null : [0, 1, 2, 3], size);
  const sample = state.samples[Math.min(selected, state.samples.length - 1)];
  const reset = () => {
    setFrame('complete');
    setSize(2);
    setSelected(0);
  };
  return <section className="sampling-lab" aria-label="Finite sampling investigation"><h3>Which samples can actually happen?</h3><p>Predict what happens to the center when you take a census of an incomplete frame. The target stays the mean of all eight units: 9.</p>
    <div className="sampling-controls"><label>Available frame<select aria-label="Available frame" value={frame} onChange={event => {
          setFrame(event.target.value);
          setSize(Math.min(size, event.target.value === 'complete' ? 8 : 4));
          setSelected(0);
        }}><option value="complete">All eight units</option><option value="partial">Only U1–U4</option></select></label><Slider label="Sample size" value={size} minimum={1} maximum={state.eligible.length} onChange={value => {
        setSize(value);
        setSelected(0);
      }} /></div>
    <div className="population-strip">{state.population.map((value, index) => <div key={index} className={`${!state.eligible.includes(index) ? 'excluded' : ''} ${sample.indices.includes(index) ? 'sampled' : ''}`}><span>U{index + 1}</span><strong>{value}</strong><small>{!state.eligible.includes(index) ? 'outside frame' : sample.indices.includes(index) ? 'selected' : 'eligible'}</small></div>)}</div>
    <div className="sampling-controls"><Slider label="Selected subset" value={selected + 1} minimum={1} maximum={state.samples.length} onChange={value => setSelected(value - 1)} /></div>
    <div className="sampling-readout" aria-live="polite">Subset {selected + 1} of {state.samples.length}: {sample.indices.map(index => `U${index + 1}`).join(', ')}<br />Selected mean = {f(sample.mean)}; probability of this subset = 1/{state.samples.length}<br />Design expectation = {f(state.expectation)}; target = {f(state.target)}<br />Variance = {f(state.formulaVariance)}; bias = {f(state.bias)}; MSE = {f(state.mse)}</div>
    <Distribution values={state.samples.map(item => item.mean)} target={state.target} expectation={state.expectation} selected={sample.mean} limits={[0, 18]} label="Exact distribution of the sample mean over every allowed equally probable subset" />
    <p>Bars combine subsets with the same mean. Their heights are probabilities, with the vertical scale shown. {frame === 'partial' ? 'Even at n=4, the frame mean stays 5. The zero sampling variance of that census cannot remove its −4 target bias.' : 'Increasing n reduces the exact design variance here. Individual values stay fixed; the selected subset is the source of randomness.'}</p><button onClick={reset}>Reset sampling</button>
  </section>;
}
export function InclusionWeightsLab() {
  const [mode, setMode] = useState('unequal');
  const [selected, setSelected] = useState(0);
  const state = inclusionDesignState(mode);
  const sample = state.samples[selected];
  return <section className="sampling-lab" aria-label="Inclusion weighting investigation"><h3>Follow the contribution of a selected unit</h3><p>Before changing the denominator, predict whether an unbiased total remains an unbiased mean. Here N=4 is known before sampling.</p>
    <div className="sampling-controls"><label>Selection design<select aria-label="Selection design" value={mode} onChange={event => {
          setMode(event.target.value);
          setSelected(0);
        }}><option value="unequal">Unequal subset probabilities</option><option value="equal">Equal subset probabilities</option><option value="uncovered">Only U1 and U2 can appear</option></select></label><label>Inspect subset<select aria-label="Inspect subset" value={selected} onChange={event => setSelected(Number(event.target.value))}>{state.samples.map((item, index) => <option key={index} value={index} disabled={item.probability === 0}>{item.indices.map(unit => `U${unit + 1}`).join(' + ')}</option>)}</select></label></div>
    <ScrollTable label="Each unit's inclusion and selected contribution" headers={['Unit', 'Value y', 'P(included)', 'Selected y / π']} rows={state.values.map((value, unit) => [`U${unit + 1}`, value, f(state.inclusion[unit]), sample.indices.includes(unit) ? f(value / state.inclusion[unit]) : '—'])} />
    <p>Chosen subset probability: {f(sample.probability)}. This is the probability of the whole pair, not the inclusion probability of either unit.</p>
    {state.covered ? <><div className="sampling-contributions">{sample.indices.map((unit, index) => <span key={unit}>U{unit + 1}: {f(sample.contributions[index])}</span>)}<strong>sum = {f(sample.contributions.reduce((sum, value) => sum + value, 0))}</strong></div><div className="sampling-readout" aria-live="polite">Raw mean: {f(sample.raw)}<br />HT mean: sum / fixed N = {f(sample.ht)}<br />Ratio mean: sum / selected inverse-weight sum = {f(sample.ratio)}</div>
      <ScrollTable label="Exact averages over the sampling design" headers={['Rule', 'Expectation', 'Variance']} rows={[["Raw mean", f(state.raw.mean), f(state.raw.variance)], ['Fixed-N HT mean', f(state.ht.mean), f(state.ht.variance)], ['Normalized ratio', f(state.ratio.mean), f(state.ratio.variance)], ['Target', f(state.target), 'fixed']]} />
      <p>The normalized denominator is random unless this design makes it constant. A smaller variance does not establish unbiasedness, and unbiasedness alone does not minimize squared error.</p></> : <div className="sampling-warning" role="status">U3 and U4 have zero inclusion probability. Their unknown contributions cannot be recovered by these design weights. The complete target mean is shown only because this teaching population is synthetic; HT and ratio claims are withheld.</div>}
    <details><summary>Inspect all six sample probabilities</summary><ScrollTable label="The complete subset law" headers={['Subset', 'Probability']} rows={state.samples.map(item => [item.indices.map(unit => `U${unit + 1}`).join(', '), f(item.probability)])} /></details><button onClick={() => {
      setMode('unequal');
      setSelected(0);
    }}>Reset weights</button>
  </section>;
}
export function GroupingDesignFigure() {
  return <figure className="sampling-figure"><div className="sampling-grouping"><div><strong>Strata: some from each</strong><div className="group-line"><b className="chosen">1</b><b>2</b><b className="chosen">3</b><b>4</b></div><div className="group-line"><b>5</b><b className="chosen">6</b><b>7</b><b className="chosen">8</b></div></div><div><strong>Clusters: some whole groups</strong><div className="group-line"><b className="chosen">1</b><b className="chosen">2</b><b className="chosen">3</b><b className="chosen">4</b></div><div className="group-line"><b>5</b><b>6</b><b>7</b><b>8</b></div></div></div><figcaption>Gold means selected. Both pictures have four recorded units. In the first, both groups supply information directly. In the second, selection acts on a whole group. These are example selections, not a claim that the two designs have the same probability law or uncertainty.</figcaption></figure>;
}
export function UnitsRepeatsLab() {
  const [units, setUnits] = useState(4);
  const [repeats, setRepeats] = useState(4);
  const [unitVariance, setUnitVariance] = useState(4);
  const [readingVariance, setReadingVariance] = useState(1);
  const [bias, setBias] = useState(0);
  const state = groupedMeasurementState(units, repeats, unitVariance, readingVariance, bias);
  return <section className="sampling-lab" aria-label="Independent units and repeated readings investigation"><h3>Buy another unit, or repeat the same measurement?</h3><p>Hold the model fixed: a unit effect is shared by its readings; different units have independent effects. Predict what changing only the repeat count can reduce.</p>
    <div className="sampling-controls"><Slider label="Independent units G" value={units} minimum={1} maximum={16} onChange={setUnits} /><Slider label="Readings per unit m" value={repeats} minimum={1} maximum={16} onChange={setRepeats} /><Slider label="Unit variance" value={unitVariance} minimum={0} maximum={9} onChange={setUnitVariance} /><Slider label="Reading variance" value={readingVariance} minimum={0} maximum={9} onChange={setReadingVariance} /><Slider label="Fixed offset b" value={bias} minimum={-3} maximum={3} step={.5} onChange={setBias} /></div>
    <div className="reading-groups">{Array.from({
        length: units
      }, (_, unit) => <div key={unit}><strong>Unit {unit + 1}</strong><div>{Array.from({
            length: repeats
          }, (_, reading) => <i key={reading} aria-hidden="true" />)}</div><span>{repeats} readings share U{unit + 1}</span></div>)}</div>
    <p>Each small stroke is one reading, not a realized value. Unit grouping records the dependence structure.</p>
    <svg viewBox="0 0 340 190" role="img" aria-label={`Variance contributions: unit ${f(state.unitComponent)}, reading ${f(state.readingComponent)}; fixed scale zero to nine`}>
      <text x="30" y="25">Variance contributions</text><text x="20" y="59">Unit</text><rect className="selected-mass" x="108" y="41" width={200 * state.unitComponent / 9} height="22" />
      <text x="20" y="103">Reading</text><rect className="mass" x="108" y="85" width={200 * state.readingComponent / 9} height="22" /><path className="axis" d="M108 127H308" />
      <text x="108" y="152" textAnchor="middle">0</text><text x="208" y="152" textAnchor="middle">4.5</text><text x="308" y="152" textAnchor="middle">9</text><text x="208" y="177" textAnchor="middle">Squared score units</text>
    </svg>
    <div className="sampling-readout" aria-live="polite">{state.readings} readings from {units} independent units<br />Unit contribution = {unitVariance}/{units} = {f(state.unitComponent)}<br />Reading contribution = {readingVariance}/({units}×{repeats}) = {f(state.readingComponent)}<br />Variance = {f(state.variance)}; SD of the mean = {f(state.standardError)}<br />Fixed bias = {f(bias)}; MSE = variance + b² = {f(state.mse)}<br />Within-unit correlation = {f(state.correlation)}; design effect = {f(state.designEffect)}</div>
    <p>The design effect compares this equally weighted, balanced mean with {state.readings} independent readings having the same marginal variance. {state.designEffect === null ? 'Both variances vanish, so the variance ratio and correlation are undefined.' : 'It is specific to this model; it is not a universal conversion from rows to independent subjects.'} All bars use a fixed 0–9 variance scale. Numerical values remain in the text; noninteger readouts are rounded.</p><button onClick={() => {
      setUnits(4);
      setRepeats(4);
      setUnitVariance(4);
      setReadingVariance(1);
      setBias(0);
    }}>Reset readings</button>
  </section>;
}
export function AssignmentInvestigation() {
  const [design, setDesign] = useState('complete');
  const [effect, setEffect] = useState('constant');
  const [selected, setSelected] = useState(0);
  const [reveal, setReveal] = useState(false);
  const state = assignmentState(design, effect);
  const allocation = state.states[selected];
  return <section className="sampling-lab" aria-label="Random assignment investigation"><h3>Change the allocation, not the enrolled units</h3><p>Predict whether one balanced assignment must recover the target effect. Then change which pairs may be split between treatment and control.</p>
    <div className="sampling-controls"><label>Assignment rule<select aria-label="Assignment rule" value={design} onChange={event => {
          setDesign(event.target.value);
          setSelected(0);
        }}><option value="complete">Any three of six</option><option value="prognostic">Pair similar baselines</option><option value="mixed">Pair unlike baselines</option></select></label><label>Individual effects<select aria-label="Individual effects" value={effect} onChange={event => {
          setEffect(event.target.value);
          setSelected(0);
        }}><option value="constant">Every effect = 2</option><option value="heterogeneous">Effects: 0, 2, 4, 0, 2, 4</option></select></label><Slider label="Allocation number" value={selected + 1} minimum={1} maximum={state.states.length} onChange={value => setSelected(value - 1)} /></div>
    {state.blocks && <p>Fixed pairs: {state.blocks.map(pair => pair.map(unit => `U${unit + 1}`).join('–')).join('; ')}. One treated unit per pair.</p>}
    <label className="sampling-toggle"><input type="checkbox" checked={reveal} onChange={event => setReveal(event.target.checked)} /> Reveal the synthetic potential-outcome table</label>
    <p className="sampling-scroll-hint">Swipe the table sideways, or focus it and use the arrow keys, to see every column.</p>
    <ScrollTable label={reveal ? 'Synthetic science table: both potential outcomes are specified' : 'Observed data: one potential outcome per enrolled unit'} headers={['Unit', 'Assigned', 'Y(0)', 'Y(1)', 'Observed']} rows={state.baseline.map((value, unit) => {
      const treated = allocation.indices.includes(unit);
      return [`U${unit + 1}`, treated ? '1: treated' : '0: control', reveal || !treated ? value : 'unobserved', reveal || treated ? state.treated[unit] : 'unobserved', allocation.observed[unit]];
    })} />
    <div className="sampling-readout" aria-live="polite">Allocation {selected + 1} of {state.states.length}<br />Observed treated mean = {f(allocation.treatedMean)}; control mean = {f(allocation.controlMean)}<br />Realized contrast = {f(allocation.difference)}<br />Fixed enrolled-unit target = {f(state.target)}; design expectation = {f(state.expectation)}<br />Exact assignment variance = {f(state.variance)}</div>
    <Distribution values={state.states.map(item => item.difference)} selected={allocation.difference} target={state.target} expectation={state.expectation} label="Exact assignment distribution over every permitted allocation" />
    <p>Each permitted allocation is equally likely. These hypothetical contrasts require the full constructed science table; they cannot all be calculated from one real experiment's observed outcomes without extra assumptions. {design === 'mixed' ? 'Unlike pairing makes uncertainty larger here: the two members have very different untreated outcomes.' : design === 'prognostic' ? 'Similar baseline pairs remove much between-pair variation from each comparison.' : 'Balanced group sizes do not guarantee equal baseline means in one allocation.'}</p><button onClick={() => {
      setDesign('complete');
      setEffect('constant');
      setSelected(0);
      setReveal(false);
    }}>Reset assignment</button>
  </section>;
}
export function FactorialInvestigation() {
  const [interaction, setInteraction] = useState(4);
  const [share, setShare] = useState(.5);
  const [reveal, setReveal] = useState(false);
  const state = factorialState(interaction, share);
  return <section className="sampling-lab" aria-label="Factorial interaction investigation"><h3>Find the combination an OFAT path never visited</h3><p>The first three cells stay fixed. Before revealing the fourth, decide whether they determine A's effect when B=1.</p>
    <div className="factorial-board"><span /><strong>A=0</strong><strong>A=1</strong><strong>B=0</strong><div>10</div><div>12</div><strong>B=1</strong><div>9</div><div className={reveal ? 'fourth-revealed' : ''}>{reveal ? f(state.cells[1][1]) : '?'}</div></div>
    <div className="sampling-controls"><Slider label="Interaction contrast" value={interaction} minimum={-6} maximum={6} onChange={setInteraction} /><Slider label="Share with B at one" value={share} minimum={0} maximum={1} step={.25} onChange={setShare} /></div><button onClick={() => setReveal(!reveal)}>{reveal ? 'Hide fourth cell' : 'Reveal fourth cell'}</button>
    {reveal ? <div className="sampling-readout" aria-live="polite">A effect at B=0: 12−10 = 2<br />A effect at B=1: {f(state.cells[1][1])}−9 = {f(state.highBEffect)}<br />Interaction: {f(state.highBEffect)}−2 = {f(state.interaction)}<br />A effect in the declared B mixture: {f(state.averageAEffect)}</div> : <p role="status">The fourth cell is hidden. Its value can change while the three observed cells remain identical. The contrast is therefore not identified by those three cells.</p>}
    <p>The cells are declared synthetic means in score units. The population mixture weights the two conditional effects; it does not change the cell means. Replicated randomized observations would be needed to estimate these means and their uncertainty in a real experiment.</p><button onClick={() => {
      setInteraction(4);
      setShare(.5);
      setReveal(false);
    }}>Reset factorial</button>
  </section>;
}
export function ExperimentalUnitFigure() {
  return <figure className="sampling-figure"><div className="experimental-units"><div><strong>Class C1 → method A</strong><span>quiz 1 · quiz 2 · quiz 3 · quiz 4</span></div><div><strong>Class C2 → method B</strong><span>quiz 1 · quiz 2 · quiz 3 · quiz 4</span></div></div><figcaption>The treatment was assigned twice, at the class level. Eight quiz rows do not create eight independent method assignments. If each method is used in only one class, class and method effects cannot be separated by treating quizzes as new classes.</figcaption></figure>;
}
export function MissingOutcomeFigure() {
  const A = boundedMissingMean([8, 9, null, null]);
  const B = boundedMissingMean([4, 5, 6, 7]);
  return <figure className="sampling-figure"><ScrollTable label="Keep every assigned outcome in the denominator" headers={['Assigned group', 'Recorded outcomes', 'Possible full mean']} rows={[["A", '8, 9, missing, missing', `${f(A.lower)} to ${f(A.upper)}`], ['B', '4, 5, 6, 7', f(B.lower)]]} /><figcaption>Scores are known to lie in [0, 10]. Setting both missing A outcomes to 0 gives the lower mean; setting both to 10 gives the upper. The full assigned-group contrast can range from {f(A.lower - B.upper)} to {f(A.upper - B.lower)}. This is missing-data uncertainty about assigned outcomes, not a claim that both counterfactual outcomes were observed.</figcaption></figure>;
}
