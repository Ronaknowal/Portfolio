import { useMemo, useState } from 'react';
import { Investigation } from './LessonInvestigation.jsx';
import { formatRandomMatrix as format, gaussianSpectrumBound, marchenkoPastur, matrixNullCalibration, randomMatrixBins, randomMatrixSpectrum, spikeLimits, twoLevelSpectrum, wignerSpectrum } from '../../data/random-matrix-models.js';
import './random-matrix-labs.css';
function Select({
  label,
  value,
  onChange,
  options
}) {
  return <label className="rm-control"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, title]) => <option key={key} value={key}>{title}</option>)}</select></label>;
}
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = .1
}) {
  return <label className="rm-control"><span>{label}: <strong>{format(value)}</strong></span><input aria-label={label} type="range" value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function SampleButtons({
  seed,
  setSeed,
  reset
}) {
  return <div className="rm-buttons"><button type="button" onClick={() => setSeed(seed + 1)}>Draw another sample</button><span>Seed {seed}</span><button type="button" onClick={reset}>Reset experiment</button></div>;
}
function SmallMatrix({
  values,
  caption
}) {
  return <div className="rm-matrix" role="region" aria-label={caption} tabIndex={0}><table><caption>{caption}</caption><tbody>{values.map((row, index) => <tr key={index}>{row.map((value, column) => <td key={column} className={value > 0 ? 'positive' : value < 0 ? 'negative' : ''}>{format(value, 2)}</td>)}</tr>)}</tbody></table></div>;
}
function Legend({
  first = 'Finite sample',
  second = 'Limiting reference'
}) {
  return <div className="rm-legend"><span className="rm-blue">{first}</span><span className="rm-gold">{second}</span></div>;
}
function Plot({
  label,
  children
}) {
  return <svg className="rm-plot" viewBox="-8 -12 376 260" role="img" aria-label={label}>{children}</svg>;
}
function Axes({
  xLabel,
  yLabel,
  maximum = 1,
  left = 0,
  right = 1
}) {
  return <><path d="M48,25 V190 H338" fill="none" stroke="var(--rm-line)" /><text x="48" y="17">{yLabel}</text><text x="190" y="238" textAnchor="middle">{xLabel}</text><text x="40" y="195" textAnchor="end">0</text><text x="40" y="45" textAnchor="end">{format(maximum, 2)}</text><text x="48" y="214" textAnchor="start">{format(left, 2)}</text><text x="338" y="214" textAnchor="end">{format(right, 2)}</text></>;
}
export function NoiseSampleFigure() {
  return <figure className="rm-inline"><div className="rm-counted"><div><strong>Eight observed rows</strong><table><thead><tr><th>First</th><th>Second</th><th>Count</th></tr></thead><tbody>{[[1, 1, 3], [-1, -1, 3], [1, -1, 1], [-1, 1, 1]].map((row, index) => <tr key={index}>{row.map((value, column) => <td key={column}>{value}</td>)}</tr>)}</tbody></table></div><SmallMatrix values={[[1, .5], [.5, 1]]} caption="XᵀX / 8" /></div><Plot label="Observed principal directions: same-sign direction has scale 1.5; opposite-sign direction has scale 0.5"><path d="M50,115 H310 M180,205 V25" stroke="var(--rm-line)" /><path d="M105,190 L255,40" stroke="var(--rm-gold)" strokeWidth="4" /><path d="M155,90 L205,140" stroke="var(--rm-blue)" strokeWidth="4" /><text x="240" y="30" fill="var(--rm-gold)">1.5</text><text x="218" y="145" fill="var(--rm-blue)">0.5</text><text x="295" y="108">x₁</text><text x="188" y="35">x₂</text></Plot><figcaption>This is a possible sample under independent population coordinates. Its observed counts favor matching signs. The gold and blue lines show eigenvector directions, with lengths proportional to eigenvalues; they are not confidence intervals.</figcaption></figure>;
}
export function CenteringRankFigure() {
  return <figure className="rm-inline"><div className="rm-rank">{[[5, 'At most five before centering', 'At least 3 zero eigenvalues'], [4, 'At most four after centering', 'At least 4 zero eigenvalues']].map(([rank, title, note]) => <div key={title}><strong>{title}</strong><div className="rm-dots" aria-label={rank + ' possible nonzero directions out of eight'}>{Array.from({
            length: 8
          }, (_, index) => <span className={index < rank ? 'active' : ''} key={index}>{index < rank ? '•' : '0'}</span>)}</div><p>{note}</p></div>)}</div><figcaption>Five observations, eight features. Subtracting the sample mean imposes a sum-to-zero relation on the rows. These are maximum ranks; additional dependence can remove more directions.</figcaption></figure>;
}
function MassChart({
  bins
}) {
  const maximum = Math.max(...bins.flatMap(bin => [bin.observed, bin.theoretical])) * 1.15;
  const width = 290 / bins.length;
  return <Plot label="Observed eigenvalue mass and integrated Marchenko–Pastur mass in matching bins"><Axes xLabel="Eigenvalue" yLabel="Mass per bin" maximum={maximum} right={bins.at(-1).upper} />{bins.map((bin, index) => <g key={index}><rect x={48 + index * width + 1} y={190 - 150 * bin.observed / maximum} width={width * .43} height={150 * bin.observed / maximum} fill="var(--rm-blue)" /><rect x={48 + index * width + width * .5} y={190 - 150 * bin.theoretical / maximum} width={width * .43} height={150 * bin.theoretical / maximum} fill="var(--rm-gold)" /></g>)}</Plot>;
}
export function CovarianceSpectrumLab() {
  const [shape, setShape] = useState('64,16');
  const [law, setLaw] = useState('gaussian');
  const [centered, setCentered] = useState(false);
  const [seed, setSeed] = useState(7);
  const [rows, columns] = shape.split(',').map(Number);
  const result = useMemo(() => randomMatrixSpectrum({
    rows,
    columns,
    law,
    centered,
    seed
  }), [rows, columns, law, centered, seed]);
  const reference = marchenkoPastur(result.gamma);
  const bins = randomMatrixBins(result.values, result.gamma);
  function reset() {
    setShape('64,16');
    setLaw('gaussian');
    setCentered(false);
    setSeed(7);
  }
  return <Investigation id="random-covariance" kicker="COMPUTED SPECTRUM · SHAPE AND PREPROCESSING" title="A noise sample still has principal directions">
    <p>Predict what happens to the zeros when features outnumber observations. Then change the shape, keeping the entry variance at one.</p>
    <div className="rm-controls"><Select label="Matrix shape" value={shape} onChange={setShape} options={[['32,8', '32 rows × 8 features'], ['64,16', '64 rows × 16 features'], ['48,48', '48 rows × 48 features'], ['24,48', '24 rows × 48 features']]} /><Select label="Entry law" value={law} onChange={setLaw} options={[['gaussian', 'Gaussian: mean 0, variance 1'], ['sign', 'Independent signs: −1 or +1']]} /><label className="rm-check"><input type="checkbox" checked={centered} onChange={event => setCentered(event.target.checked)} />Subtract sample means</label></div>
    <SampleButtons seed={seed} setSeed={setSeed} reset={reset} /><Legend /><MassChart bins={bins} />
    <div className="rm-zero"><strong>Mass exactly at zero, shown separately</strong><p>Numerical sample: {result.zeroCount}/{columns} = {format(result.zeroCount / columns)}. Limiting atom: {format(reference.atom)}.</p><div className="rm-mass-track"><span style={{
          width: 100 * result.zeroCount / columns + '%'
        }} /></div></div>
    <p aria-live="polite">Divide by {result.denominator}; effective ratio {format(result.gamma)}. Mean eigenvalue {format(result.meanEigenvalue)}, largest {format(result.largest)}. Continuous reference support [{format(reference.lower)}, {format(reference.upper)}].</p>
    <p className="rm-note">Bars measure probability mass, not density height. Each gold bar integrates the formula across its whole bin, including the integrable singularity at ratio 1. Bars plus the separate zero atom have total mass one. A finite sample may put eigenvalues outside the limiting support.</p>
    <p className="rm-note">{centered ? 'For Gaussian entries, rotating the centered rows gives an exact n−1-row Gaussian representation. Centered signs do not become independent rows exactly; this is a limiting comparison.' : 'The population mean is known to be zero here. A generated sample need not have exactly zero column means.'} Values within 10⁻⁹ × max(1, largest eigenvalue) of zero are displayed as numerical zeros.</p>
    <details><summary>Inspect the actual entries, eigenvalues and bin masses</summary><SmallMatrix values={result.data.slice(0, 4).map(row => row.slice(0, 4))} caption="First four raw rows and columns" /><p className="rm-values">{result.values.map(value => format(value)).join(', ')}</p><div className="rm-table-scroll" tabIndex={0}><table><caption>Matching intervals, mass divided by all features</caption><thead><tr><th>Interval</th><th>Sample</th><th>Reference</th></tr></thead><tbody>{bins.map((bin, index) => <tr key={index}><th scope="row">{format(bin.lower, 2)}–{format(bin.upper, 2)}</th><td>{format(bin.observed)}</td><td>{format(bin.theoretical)}</td></tr>)}</tbody></table></div></details>
    <p><strong>Transfer:</strong> compare 32×8 with 64×16. Which reference stays fixed, and which sample details still fluctuate?</p>
  </Investigation>;
}
export function FiniteSpectrumLab() {
  const [nullModel, setNullModel] = useState('iid');
  const [observedModel, setObservedModel] = useState('spike');
  const result = useMemo(() => matrixNullCalibration(nullModel, observedModel), [nullModel, observedModel]);
  const maximum = Math.max(result.bulkEdge, result.observed, ...result.maxima) * 1.15;
  const x = value => 48 + 290 * value / maximum;
  const bound = gaussianSpectrumBound(40, 12, .05);
  return <Investigation id="random-finite-calibration" kicker="FINITE NULL · MODEL CHECK" title="Compare a largest eigenvalue with a finite null">
    <p>Each dot is the largest eigenvalue of a separately seeded 40×12 simulated matrix. The green triangle is one observation; the dashed gold line is the iid limiting bulk edge.</p>
    <div className="rm-controls"><Select label="Null generation" value={nullModel} onChange={setNullModel} options={[['iid', 'Independent Gaussian columns'], ['duplicate', 'Last column duplicates the first']]} /><Select label="Observed model" value={observedModel} onChange={setObservedModel} options={[['iid', 'Independent Gaussian columns'], ['duplicate', 'One duplicated column'], ['spike', 'Population spike of 3']]} /></div>
    <Plot label="Fifty-nine finite null maxima and the observed maximum"><path d="M48,190 H338" stroke="var(--rm-line)" /><text x="48" y="20">59 null maxima</text><text x="48" y="210">0</text><text x="338" y="210" textAnchor="end">{format(maximum, 2)}</text><text x="190" y="230" textAnchor="middle">Largest eigenvalue</text><line x1={x(result.bulkEdge)} x2={x(result.bulkEdge)} y1="30" y2="185" stroke="var(--rm-gold)" strokeDasharray="5 4" />{result.maxima.map((value, index) => <circle key={index} cx={x(value)} cy={55 + index % 7 * 13} r="3.5" fill="var(--rm-blue)" />)}<path d={'M' + x(result.observed) + ',172 l-7,-11 h14z'} fill="var(--rm-green)" /></Plot>
    <p aria-live="polite">Observed maximum {format(result.observed)}. Null values at least this large: {result.exceedances}/59. Conservative rank score: (1 + {result.exceedances})/60 = {format(result.rankScore)}.</p>
    <p>That score has a null probability interpretation when the observation and null draws are exchangeable under the fully specified null. A deliberately different observed model need not satisfy it. Choosing a null after looking at many scores also changes the procedure. The fixed seeds make this teaching comparison reproducible; they do not validate the null.</p>
    <p className="rm-note">Resolution is 1/60: zero exceedances do not establish zero tail probability. For comparison, the finite iid Gaussian 95% simultaneous bound is [{format(bound.lower)}, {format(bound.upper)}]. It does not apply to duplicated columns. Dot height separates marks and encodes no probability.</p>
    <button type="button" onClick={() => {
      setNullModel('iid');
      setObservedModel('spike');
    }}>Reset comparison</button>
    <details><summary>Inspect all finite null maxima</summary><p className="rm-values">{result.maxima.map(value => format(value)).join(', ')}</p></details>
  </Investigation>;
}
export function SpikeAlignmentLab() {
  const [population, setPopulation] = useState(2);
  const [columns, setColumns] = useState(16);
  const [seed, setSeed] = useState(7);
  const sample = useMemo(() => randomMatrixSpectrum({
    rows: 64,
    columns,
    seed,
    spike: population
  }), [columns, seed, population]);
  const limit = spikeLimits(columns / 64, population);
  const curve = Array.from({
    length: 81
  }, (_, index) => {
    const strength = 1 + index / 20;
    return [strength, spikeLimits(columns / 64, strength).sample];
  });
  const maximum = Math.max(sample.largest, ...curve.map(point => point[1])) * 1.15;
  const x = value => 48 + (value - 1) / 4 * 290;
  const y = value => 190 - value / maximum * 150;
  return <Investigation id="random-spike" kicker="POPULATION → SAMPLE → ALIGNMENT" title="A population spike is not its sample eigenvalue">
    <p>One population direction has variance ℓ; all others have variance 1. Holding the seed fixed changes that direction's strength without redrawing the base noise.</p>
    <div className="rm-controls"><Range label="Population eigenvalue ℓ" value={population} onChange={setPopulation} min={1} max={5} /><Select label="Features with 64 observations" value={columns} onChange={value => setColumns(Number(value))} options={[[16, '16 features: ratio 0.25'], [32, '32 features: ratio 0.5']]} /></div>
    <SampleButtons seed={seed} setSeed={setSeed} reset={() => {
      setPopulation(2);
      setColumns(16);
      setSeed(7);
    }} />
    <Legend /><Plot label="Limiting sample eigenvalue curve and one actual sample eigenvalue"><Axes xLabel="Population eigenvalue ℓ" yLabel="Sample eigenvalue" maximum={maximum} left={1} right={5} /><polyline points={curve.map(point => x(point[0]) + ',' + y(point[1])).join(' ')} fill="none" stroke="var(--rm-gold)" strokeWidth="3" /><line x1={x(limit.threshold)} x2={x(limit.threshold)} y1="30" y2="190" stroke="var(--rm-line)" strokeDasharray="4 4" /><circle cx={x(population)} cy={y(limit.sample)} r="5" fill="var(--rm-gold)" /><circle cx={x(population)} cy={y(sample.largest)} r="5" fill="var(--rm-blue)" /></Plot>
    <div className="rm-overlap">{[[limit.alignment, 'Limiting squared overlap', 'gold'], [sample.alignment, 'Finite squared overlap', 'blue']].map(([value, title, color]) => <div key={title}><span>{title}: {value === null ? 'top direction numerically unresolved' : format(value)}</span><div className="rm-mass-track"><span className={color} style={{
            width: 100 * (value ?? 0) + '%'
          }} /></div></div>)}</div>
    <p aria-live="polite">Threshold {format(limit.threshold)}. Population ℓ = {format(population)}; limiting sample value {format(limit.sample)}; actual sample value {format(sample.largest)}. Squared overlap ranges from 0 to 1 and ignores the eigenvector's arbitrary sign.</p>
    <p className="rm-note">The gold curve is the large-dimension, single-spike Gaussian result with fixed ratio below 1. The blue point comes from 64 actual simulated rows. A positive finite overlap below threshold does not contradict a limiting overlap of zero. The dashed line marks a population threshold, not a finite significance cutoff.</p>
  </Investigation>;
}
export function InverseGainFigure() {
  return <figure className="rm-inline"><div className="rm-gains">{[[.01, 100, 1 / .11], [2, .5, 1 / 2.1]].map(([eigenvalue, before, after]) => <div key={eigenvalue}><strong>Direction with λ = {eigenvalue}</strong><p>Input perturbation: 0.02</p><p>Inverse gain {format(before)} → change {format(.02 * before, 5)}</p><p className="rm-green">Add α = 0.1: gain {format(after)} → change {format(.02 * after, 5)}</p></div>)}</div><figcaption>Adding αI changes every inverse gain from 1/λ to 1/(λ + α). This controls amplification in the weak direction, while also changing the estimation target.</figcaption></figure>;
}
export function WignerSpectrumLab() {
  const [size, setSize] = useState(32);
  const [seed, setSeed] = useState(7);
  const [law, setLaw] = useState('gaussian');
  const [scaled, setScaled] = useState(true);
  const result = useMemo(() => wignerSpectrum(size, seed, law), [size, seed, law]);
  const factor = scaled ? 1 : Math.sqrt(size);
  const extent = Math.max(2.15, ...result.values.map(Math.abs)) * 1.05;
  const width = 2 * extent / 12;
  const bins = Array.from({
    length: 12
  }, (_, index) => ({
    left: -extent + index * width,
    density: result.values.filter(value => value >= -extent + index * width && value < -extent + (index + 1) * width).length / size / width
  }));
  const maximum = Math.max(1 / Math.PI, ...bins.map(bin => bin.density)) * 1.15;
  const x = value => 48 + (value + extent) / (2 * extent) * 290;
  const y = value => 190 - value / maximum * 150;
  const curve = Array.from({
    length: 81
  }, (_, index) => {
    const value = -2 + index / 20;
    return [value, Math.sqrt(Math.max(0, 4 - value * value)) / (2 * Math.PI)];
  });
  return <Investigation id="random-wigner" kicker="MIRRORED ENTRIES · SPECTRAL SCALE" title="Reflect the entries, then inspect the spectrum">
    <p>Covariance multiplies a matrix by its transpose. This model instead mirrors independent entries across a diagonal. Predict whether negative eigenvalues are possible.</p>
    <div className="rm-controls"><Select label="Symmetric matrix size" value={size} onChange={value => setSize(Number(value))} options={[[12, '12 × 12'], [32, '32 × 32'], [48, '48 × 48']]} /><Select label="Symmetric entry law" value={law} onChange={setLaw} options={[['gaussian', 'GOE: Gaussian upper triangle'], ['sign', 'Signs off diagonal; zero diagonal']]} /><label className="rm-check"><input type="checkbox" checked={scaled} onChange={event => setScaled(event.target.checked)} />Divide matrix by square root of size</label></div>
    <SampleButtons seed={seed} setSeed={setSeed} reset={() => {
      setSize(32);
      setSeed(7);
      setLaw('gaussian');
      setScaled(true);
    }} />
    <SmallMatrix values={result.matrix.slice(0, 4).map(row => row.slice(0, 4).map(value => value * factor))} caption="Actual upper-left 4 × 4 block" />
    <Legend /><Plot label="Symmetric matrix eigenvalue density and a scaled semicircle reference"><Axes xLabel="Eigenvalue" yLabel="Density" maximum={maximum / factor} left={-extent * factor} right={extent * factor} />{bins.map((bin, index) => <rect key={index} x={x(bin.left) + 1} y={y(bin.density)} width={290 / 12 - 2} height={190 - y(bin.density)} fill="var(--rm-blue)" opacity=".8" />)}<polyline points={curve.map(point => x(point[0]) + ',' + y(point[1])).join(' ')} fill="none" stroke="var(--rm-gold)" strokeWidth="3" /></Plot>
    <p aria-live="polite">Mean squared eigenvalue: {format(result.secondMoment * factor * factor)}. {law === 'gaussian' ? 'Expected' : 'Exact'} finite value: {format((law === 'gaussian' ? 1 + 1 / size : 1 - 1 / size) * factor * factor)}.</p>
    <p className="rm-note">Histogram area is one. Removing the division stretches the horizontal axis by √d and lowers density by the same factor; the matrix and eigenvalues actually rescale. GOE has diagonal variance twice its off-diagonal variance. The sign model has a zero diagonal. Both converge to the displayed semicircle under their stated scaling; their finite spectra need not match.</p>
    <details><summary>Inspect every eigenvalue</summary><p className="rm-values">{result.values.map(value => format(value * factor)).join(', ')}</p></details>
  </Investigation>;
}
export function AvoidedCrossingLab() {
  const [difference, setDifference] = useState(0);
  const [coupling, setCoupling] = useState(.5);
  const result = twoLevelSpectrum(0, difference, coupling);
  const curve = Array.from({
    length: 81
  }, (_, index) => {
    const value = -2 + index / 20;
    return [value, twoLevelSpectrum(0, value, coupling)];
  });
  const x = value => 48 + (value + 2) / 4 * 290;
  const y = value => 115 - value * 28;
  return <Investigation id="random-level-gap" kicker="EXACT 2 × 2 MODEL · AVOIDED CROSSING" title="Change coupling and watch a crossing open">
    <p>The diagonal entries are d and −d; both off-diagonal entries are c. At c = 0 the two eigenvalue branches meet. Test what changes when c is nonzero.</p>
    <div className="rm-controls"><Range label="Diagonal difference d" value={difference} onChange={setDifference} min={-2} max={2} /><Range label="Coupling c" value={coupling} onChange={setCoupling} min={0} max={1.5} /></div>
    <Plot label="Exact eigenvalue branches of a coupled two-level symmetric matrix"><path d="M48,30 V200 H338 M48,115 H338" fill="none" stroke="var(--rm-line)" /><text x="48" y="20">Eigenvalue</text><text x="185" y="225">d</text><text x="47" y="48" textAnchor="end">2.5</text><text x="47" y="119" textAnchor="end">0</text><text x="47" y="189" textAnchor="end">−2.5</text><text x="48" y="216">−2</text><text x="328" y="216">2</text>{['upper', 'lower'].map((key, index) => <polyline key={key} points={curve.map(point => x(point[0]) + ',' + y(point[1][key])).join(' ')} fill="none" stroke={index ? 'var(--rm-blue)' : 'var(--rm-gold)'} strokeWidth="3" />)}<line x1={x(difference)} x2={x(difference)} y1={y(result.lower)} y2={y(result.upper)} stroke="var(--rm-green)" strokeWidth="4" /></Plot>
    <p aria-live="polite">Eigenvalues {format(result.lower)}, {format(result.upper)}. Current gap {format(result.gap)}; minimum possible gap over d is 2|c| = {format(2 * Math.abs(coupling))}.</p>
    <button type="button" onClick={() => {
      setDifference(0);
      setCoupling(.5);
    }}>Reset levels</button><p className="rm-note">These are exact calculated branches of the displayed matrix. They illustrate a mechanism behind level repulsion; they are not measurements of a physical system or a universal spacing law.</p>
  </Investigation>;
}
