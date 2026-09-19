import { useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { gaussianProcessData as data } from '../../data/gaussian-process-data.js';
import { createConditioner, measurementGains, initialObservations, normal95, spatialKernel } from '../../data/gaussian-process-model.js';
import './gaussian-process.css';

export const gpNumber = value => Math.abs(value) < 5e-10 ? '0' : Number(value.toFixed(6)).toString();
const covarianceNumber = value => value !== 0 && Math.abs(value) < 1e-5 ? value.toExponential(2) : gpNumber(value);
const colors = ['#f5ce70', '#77c8d0', '#c2a7e4'];
const path = (points, x, y) => points.map((point, i) => `${i ? 'L' : 'M'}${x(point[0])},${y(point[1])}`).join(' ');

/** Model coordinates stay in the plot; prose/legends have their own flowing space. */
export function GPChart({ label, series = [], points = [], bands = [], xLabel = 'position x', yLabel = 'function value', xDomain, yDomain, xTicks, xFormat = gpNumber, verticals = [] }) {
  const all = [...series.flatMap(s => s.values), ...points.map(p => [p.x, p.y]), ...bands.flatMap(b => b.x.flatMap((x, i) => [[x, b.lower[i]], [x, b.upper[i]]]))];
  const xd = xDomain || [Math.min(...all.map(p => p[0])), Math.max(...all.map(p => p[0]))];
  const limits = yDomain || [Math.min(...all.map(p => p[1])), Math.max(...all.map(p => p[1]))];
  const padding = (limits[1] - limits[0] || 1) * 0.08;
  const yd = yDomain || [limits[0] - padding, limits[1] + padding];
  const x = value => 80 + (value - xd[0]) / (xd[1] - xd[0] || 1) * 292;
  const y = value => 224 - (value - yd[0]) / (yd[1] - yd[0] || 1) * 183;
  const ticks = xTicks || [xd[0], (xd[0] + xd[1]) / 2, xd[1]];
  return <div className="gp-chart-wrap">
    <p className="gp-axis-title">{yLabel}</p>
    <svg viewBox="0 0 410 270" role="img" aria-label={label} className="gp-chart">
      <title>{label}</title>
      {[yd[0], (yd[0] + yd[1]) / 2, yd[1]].map((value, i) => <g key={i}>
        <path d={`M80 ${y(value)} H372`} className="gp-grid" />
        <text x="72" y={y(value) + 5} textAnchor="end">{Number(value.toFixed(Math.abs(value) < 10 ? 2 : 1))}</text>
      </g>)}
      {bands.map((band, i) => <g key={i}>
        {band.type !== 'observation' && <path className="gp-band" d={`${path(band.x.map((v, j) => [v, band.upper[j]]), x, y)} ${path(band.x.map((v, j) => [v, band.lower[j]]).reverse(), x, y).replace('M', 'L')} Z`} />}
        {band.type === 'observation' && ['lower', 'upper'].map(side => <path key={side} className="gp-observation-band" d={path(band.x.map((v, j) => [v, band[side][j]]), x, y)} />)}
      </g>)}
      {verticals.map((v, i) => <path key={i} d={`M${x(v)} 41 V224`} className="gp-boundary" />)}
      {series.map((s, i) => <path key={i} d={path(s.values, x, y)} fill="none" stroke={s.color || colors[i % 3]} strokeWidth="2.5" strokeDasharray={s.dash} />)}
      {points.map((p, i) => p.target
        ? <path key={i} d={`M${x(p.x)} ${y(p.y) - 7} l7 7 -7 7 -7 -7 Z`} fill="#f5ce70" stroke="#111" />
        : <circle key={i} cx={x(p.x)} cy={y(p.y)} r={p.missed ? 4.8 : 3.6} fill={p.missed ? 'none' : '#e2e4de'} stroke={p.missed ? '#f49d89' : '#101819'} strokeWidth={p.missed ? 2 : 1} />)}
      {ticks.map((value, i) => <text key={i} x={x(value)} y="249" textAnchor={i === 0 ? 'start' : i === ticks.length - 1 ? 'end' : 'middle'}>{xFormat(value)}</text>)}
    </svg>
    <p className="gp-axis-title gp-x-label">{xLabel}</p>
  </div>;
}

export function GPBands({ x, mean, latentSD, observationSD, points = [], label, ...props }) {
  const bands = [
    { x, lower: mean.map((v, i) => v - normal95 * latentSD[i]), upper: mean.map((v, i) => v + normal95 * latentSD[i]) },
    { type: 'observation', x, lower: mean.map((v, i) => v - normal95 * observationSD[i]), upper: mean.map((v, i) => v + normal95 * observationSD[i]) },
  ];
  return <><GPChart label={label} series={[{ values: x.map((v, i) => [v, mean[i]]) }]} bands={bands} points={points} {...props} />
    <p className="gp-key"><span>━ mean</span><span>▰ latent 95% band</span><span>┄ new-observation 95% limits</span><span>● observed / ◇ target</span></p>
  </>;
}

export function VectorFunctionFigure() {
  const [draw, setDraw] = useState(0);
  const sample = data.tiny.prior_samples['1.0'];
  const values = [16, 32, 48].map(index => sample.y[draw][index]);
  return <figure className="gp-figure" data-gp-figure="vector">
    <figcaption><strong>One draw, three coordinates, three plotted values</strong></figcaption>
    <label>Finite joint draw <select value={draw} onChange={event => setDraw(Number(event.target.value))}>{[0, 1, 2].map(i => <option key={i} value={i}>Draw {i + 1}</option>)}</select></label>
    <div className="gp-vector">{values.map((v, i) => <div key={i}><span>slot {i + 1} → x = {i}</span><strong>{gpNumber(v)}</strong></div>)}</div>
    <GPChart label="Three coordinates from one Gaussian draw placed at inputs zero, one and two" series={[{ values: values.map((v, i) => [i, v]) }]} points={values.map((v, i) => ({ x: i, y: v }))} xTicks={[0, 1, 2]} />
    <figcaption>Connecting finite plotted values. These values come from one joint draw (seed 23, length 1), not three independent marginal draws. The joining lines are a drawing convention, not the complete continuous realization.</figcaption>
  </figure>;
}

function SlicePanel({ rho }) {
  const determinant = 1.25 - rho * rho;
  const mean = rho * 2 / 1.25;
  const variance = determinant / 1.25;
  const mapX = x => 200 + x * 42;
  const mapY = y => 166 - y * 38;
  const contours = [0.7, 1.5, 2.5].map(radius => Array.from({ length: 101 }, (_, i) => {
    const angle = 2 * Math.PI * i / 100;
    const a = Math.sqrt(1.25) * radius * Math.cos(angle);
    const b = rho / Math.sqrt(1.25) * radius * Math.cos(angle) + Math.sqrt(variance) * radius * Math.sin(angle);
    return [a, b];
  }));
  const grid = Array.from({ length: 81 }, (_, i) => -4 + i / 10);
  const density = (v, mu, v2) => Math.exp(-0.5 * (v - mu) ** 2 / v2) / Math.sqrt(2 * Math.PI * v2);
  return <div className="gp-slice">
    <h4>Covariance ρ = {rho}</h4>
    <p>Joint-density contours: y₀ across, f* up</p>
    <svg viewBox="0 0 410 310" className="gp-chart" role="img" aria-label={`Joint Gaussian contours, covariance ${rho}, sliced at observation 2`}>
      <path d="M46 166 H354 M200 43 V289" className="gp-grid" />
      {contours.map((points, i) => <path key={i} d={path(points, mapX, mapY)} fill="none" stroke="#77c8d0" strokeWidth="1.8" />)}
      <path d={`M${mapX(2)} 43 V289`} className="gp-slice-line" />
      <text x={mapX(2)} y="30" textAnchor="middle">slice y₀ = 2</text>
      {[-3, 0, 3].map(value => <g key={`x-${value}`}><path d={`M${mapX(value)} 162 V170`} className="gp-grid" /><text x={mapX(value)} y="303" textAnchor="middle">{value}</text></g>)}
      {[-3, 3].map(value => <g key={`y-${value}`}><path d={`M196 ${mapY(value)} H204`} className="gp-grid" /><text x="28" y={mapY(value) + 5} textAnchor="end">{value}</text></g>)}
    </svg>
    <GPChart label={`Marginal and conditional density for covariance ${rho}`} xDomain={[-4, 4]} yDomain={[0, 0.5]} xLabel="latent value f*" yLabel="probability density" series={[
      { values: grid.map(v => [v, density(v, 0, 1)]), color: '#c2a7e4', dash: '6 4' },
      { values: grid.map(v => [v, density(v, mean, variance)]), color: '#f5ce70' },
    ]} />
    <p>Dashed: marginal N(0, 1). Solid: conditional N({gpNumber(mean)}, {gpNumber(variance)}). Its standard deviation is {gpNumber(Math.sqrt(variance))}.</p>
  </div>;
}

export function ConditioningSliceFigure() {
  return <figure className="gp-figure" data-gp-figure="slice"><figcaption><strong>Normalize a vertical slice to get the conditional density</strong></figcaption>
    <div className="gp-panels"><SlicePanel rho={0.5} /><SlicePanel rho={0} /></div>
    <figcaption>The ellipses are equal joint density, not sampled curves. In the independent case the conditional and marginal curves overlap exactly. Covariance matrices have diagonal (1.25, 1); their determinants are 1 and 1.25, respectively.</figcaption>
  </figure>;
}

export function ObservationMatrixFigure() {
  const result = createConditioner().predict({ x: [0, 2], y: [1, -1], targets: [0, 1, 2, 4] });
  const cross = Math.exp(-0.5);
  return <figure className="gp-figure" data-gp-figure="matrix"><figcaption><strong>Opposite mean contributions; shared information</strong></figcaption>
    <div className="gp-vector">{[0, 1].map(i => <div key={i}><span>Observation {i + 1}: x = {i * 2}</span><strong>y = {i ? '−1' : '1'}</strong><span>cross-covariance → {gpNumber(cross)}</span><span>mean contribution → {gpNumber(cross * result.weights[i])}</span></div>)}</div>
    <LessonTable caption="Observation covariance C (rows and columns are observations)" headers={['row', 'x = 0', 'x = 2']} rows={result.observationMatrix.map((row, i) => [`x = ${i * 2}`, ...row.map(gpNumber)])} />
    <p>At target x = 1: <strong>mean = 0</strong>, but <strong>variance = {gpNumber(result.variance[1])}</strong>. The signed mean contributions cancel; the variance subtraction is {gpNumber(1 - result.variance[1])}.</p>
    <LessonTable caption="Every checked target" headers={['target', 'mean', 'latent variance', 'new-observation variance']} rows={[0, 1, 2, 4].map((x, i) => [x, gpNumber(result.mean[i]), gpNumber(result.variance[i]), gpNumber(result.variance[i] + 0.25)])} />
  </figure>;
}

export function KernelGeometryFigure() {
  const extrema = Object.values(data.tiny.prior_samples).flatMap(sample => sample.y.flat());
  const yDomain = [Math.min(...extrema) - 0.2, Math.max(...extrema) + 0.2];
  const positions = [-1, 0, 1, 2, 3, 4];
  return <figure className="gp-figure" data-gp-figure="kernels"><figcaption><strong>The same normal draws, three covariance assumptions</strong></figcaption>
    {[0.3, 1, 3].map(length => {
      const sample = data.tiny.prior_samples[length.toFixed(1)];
      return <div className="gp-kernel-row" key={length}><div><h4>RBF length ℓ = {length}</h4><GPChart label={`Three prior function draws at length ${length}`} yDomain={yDomain} xDomain={[-1, 4]} series={sample.y.map((values, i) => ({ values: sample.x.map((x, j) => [x, values[j]]), dash: ['', '7 4', '2 4'][i] }))} /></div>
        <div className="gp-matrix-wrap" tabIndex="0" role="region" aria-label={`Covariance matrix for length ${length}`}><table className="gp-covariance"><caption>Covariance k(x, z); shade 0 → 1</caption><thead><tr><th>x ∖ z</th>{positions.map(x => <th key={x}>{x}</th>)}</tr></thead><tbody>{positions.map(x => <tr key={x}><th>{x}</th>{positions.map(z => { const value = spatialKernel('rbf', length)(x, z); return <td key={z} style={{ backgroundColor: `rgba(119,200,208,${value * 0.36})` }}>{value.toFixed(2)}</td>; })}</tr>)}</tbody></table><p>Distance 0: 1; distance 1: {covarianceNumber(spatialKernel('rbf', length)(0, 1))}; distance 2: {covarianceNumber(spatialKernel('rbf', length)(0, 2))}. Matrix entries are rounded; 0.00 need not mean exact independence.</p></div>
      </div>;
    })}
    <figcaption>Common output axis and covariance shade scale. Solid, dashed and dotted paths reuse the same three base normal vectors (seed 23). Finite draws use 81 positions and 10⁻¹⁰ diagonal stabilization. Greater length connects distant values more strongly; it is not a period.</figcaption>
  </figure>;
}

export function ForecastTable({ rows, mean, latentSD, observationSD }) {
  return <details><summary>Monthly values and interval checks</summary><LessonTable caption="Historical forecast details (ppm)" headers={['month', 'actual', 'mean', 'latent SD', 'observation SD', 'inside 95%?']} rows={rows.map((row, i) => [row.month, row.co2, gpNumber(mean[i]), gpNumber(latentSD[i]), gpNumber(observationSD[i]), Math.abs(row.co2 - mean[i]) <= normal95 * observationSD[i] ? 'yes' : 'no'])} /></details>;
}

export function ForecastResult({ forecast, cutoff, title, showPrefix = false, yDomain }) {
  const rows = data.observations.slice(cutoff, cutoff + forecast.mean.length);
  const latentSD = forecast.latent_sd || forecast.latentSD;
  const observationSD = forecast.observation_sd || forecast.observationSD;
  const prefix = showPrefix ? data.observations.slice(Math.max(0, cutoff - 24), cutoff) : [];
  const points = [...prefix.map(row => ({ x: 1990 + row.x, y: row.co2 })), ...rows.map((row, i) => ({ x: 1990 + row.x, y: row.co2, missed: Math.abs(row.co2 - forecast.mean[i]) > normal95 * observationSD[i] }))];
  return <section className="gp-forecast-result"><h4>{title}</h4><GPBands label={title} x={rows.map(row => 1990 + row.x)} mean={forecast.mean} latentSD={latentSD} observationSD={observationSD} points={points} xLabel="calendar year" yLabel="monthly CO₂ (ppm)" xFormat={v => Number(v.toFixed(1))} verticals={showPrefix ? [1990 + cutoff / 12] : []} yDomain={yDomain} />
    <p>● measured monthly mean; coral open rings fall outside the observation limits. {showPrefix && 'Dashed boundary ends the conditioning prefix.'}</p>
    <GPChart label={`${title}: residuals actual minus forecast`} yLabel="actual − forecast (ppm)" xLabel="calendar year" xFormat={v => Number(v.toFixed(1))} series={[{ values: rows.map((row, i) => [1990 + row.x, row.co2 - forecast.mean[i]]) }, { values: rows.map(row => [1990 + row.x, 0]), color: '#b4b8b0', dash: '4 3' }]} />
    <ForecastTable rows={rows} mean={forecast.mean} latentSD={latentSD} observationSD={observationSD} />
  </section>;
}

export function HistoricalForecastFigure() {
  return <figure className="gp-figure" data-gp-figure="forecast"><figcaption><strong>Accurate levels can coexist with poor intervals</strong></figcaption>
    <p>First train: January 1990–December 1995 → development: 1996–97. Refit: January 1990–December 1997 → final test: 1998–99. The two segments are separate fitted models.</p>
    <GPChart label="All 120 NOAA monthly observations and the two chronological boundaries" series={[{ values: data.observations.map(row => [1990 + row.x, row.co2]) }]} xDomain={[1990, 2000]} xTicks={[1990, 1996, 2000]} xLabel="calendar year; boundaries at 1996 and 1998" yLabel="observed monthly CO₂ (ppm)" verticals={[1996, 1998]} />
    <div className="gp-panels"><ForecastResult forecast={data.real.development.trend_periodic} cutoff={72} title="Development forecast · 1996–97" showPrefix /><ForecastResult forecast={data.real.final_test} cutoff={96} title="Final test forecast · 1998–99" showPrefix /></div>
    <p>Final GP MAE <strong>1.233775 ppm</strong>; seasonal-naive MAE <strong>3.813333 ppm</strong>. Only <strong>13/24</strong> monthly observations enter the nominal 95% observation intervals.</p>
    <figcaption>NOAA GML observations, historical subset retrieved 12 September 2026. These correlated monthly outcomes are one forecast period; the count is not a binomial calibration estimate from independent cases. All bands are pointwise and conditional on the fitted model.</figcaption>
  </figure>;
}

export function ProbeResults({ rows }) {
  const maximum = Math.max(...rows.map(row => row.reduction), 1e-15);
  return <div className="gp-probe-results">{rows.map((row, i) => <div className="gp-probe-result" key={row.id || i}>
    <h4>Candidate {i + 1} · x = {row.x}</h4>
    <p>◇ target at x = {row.target}; □ candidate at x = {row.x}</p>
    <svg viewBox="0 0 410 150" className="gp-chart" role="img" aria-label={`Covariance connection from target ${row.target} to candidate ${row.x}`}>
      <path d="M45 38 H365 M45 88 H365" className="gp-grid" />
      {row.covariance !== 0 && <path d={`M${45 + (row.target + 1) * 64} 38 L${45 + (row.x + 1) * 64} 88`} fill="none" stroke="#77c8d0" strokeWidth={8 * Math.abs(row.covariance)} strokeDasharray={row.covariance < 0 ? '4 3' : undefined} />}
      <path d={`M${45 + (row.target + 1) * 64} 30 l8 8 -8 8 -8 -8 Z`} fill="#f5ce70" />
      <rect x={37 + (row.x + 1) * 64} y="80" width="16" height="16" fill="#bda2df" />
      <text x="45" y="132">−1</text><text x="205" y="132" textAnchor="middle">position x</text><text x="365" y="132" textAnchor="end">4</text>
    </svg>
    <p>Connection width encodes |covariance|; a dashed link is negative. Zero covariance draws no link. The two lanes separate roles, not physical dimensions.</p>
    <p>Covariance to target: <strong>{gpNumber(row.covariance)}</strong>; own latent variance: {gpNumber(row.ownVariance)}.</p>
    <div className="gp-gain-track" aria-hidden="true"><div style={{ width: `${100 * row.reduction / maximum}%` }} /></div>
    <p>Target variance: {gpNumber(row.currentVariance)} − <strong>{gpNumber(row.reduction)}</strong> = {gpNumber(row.remainingVariance)}.</p>
  </div>)}</div>;
}

export function ProbeChoiceFigure() {
  const rows = measurementGains({ observations: initialObservations, target: 1, candidates: [{ x: 1, noise: 0.25 }, { x: 4, noise: 0.25 }] });
  return <figure className="gp-figure" data-gp-figure="probes"><figcaption><strong>Which reading clarifies target x = 1?</strong></figcaption><ProbeResults rows={rows} /><figcaption>Bars compare variance reduction at this target, with a shared zero and scale. The reading at x = 4 has greater uncertainty about its own value but a weaker covariance connection to the target. No future observed value is needed to compute the gain.</figcaption></figure>;
}
