import { useId, useState } from 'react';
import {
  eventWindows, methodLabels, methodOrder, overviewBins, publishedOutcomes, quantileOutcomes,
  rowCounts, seriesStart, sourceFacts, stepMinutes, sweepOutcomes, windowDetail,
} from '../../data/anomaly-temperature-data';
import { formatTimestamp, quantileIndex, rowTimestamp, windowHits } from '../../data/anomaly-detection-models';
import { Field, Investigation, NumberField, Table, integer, round } from './AnomalyDetectionShared.jsx';
import './anomaly-detection-labs.css';

const OUTCOME = { threshold: 0, calibrationAlerts: 1, testAlerts: 2, inside: 3, outside: 4, hits: 5 };
const workloadBands = [
  ['none', 'No unmatched alerts', outside => outside === 0],
  ['small', 'Up to 500 unmatched alerts', outside => outside > 0 && outside <= 500],
  ['medium', '501 to 2,000', outside => outside > 500 && outside <= 2000],
  ['large', '2,001 to 5,000', outside => outside > 2000 && outside <= 5000],
  ['huge', 'More than 5,000', outside => outside > 5000],
];
const bandOf = outside => workloadBands.find(([, , test]) => test(outside))[0];
const time = index => formatTimestamp(rowTimestamp(seriesStart, index, stepMinutes));

/** §10. The real chronology: choose a threshold, then meet its workload. */
export function TemperatureThresholdLab() {
  const [method, setMethod] = useState('isolation');
  const [mode, setMode] = useState('quantile');
  const [quantile, setQuantile] = useState(0.975);
  const [direct, setDirect] = useState(0.6);
  const [window, setWindow] = useState(0);
  const [onlyAlerts, setOnlyAlerts] = useState(true);
  const [hitChoice, setHitChoice] = useState('');
  const [bandChoice, setBandChoice] = useState('');
  const [committed, setCommitted] = useState(null);
  const [revealed, setRevealed] = useState(false);
  const [rowPage, setRowPage] = useState(0);
  const hitId = useId();
  const bandId = useId();

  const sweep = sweepOutcomes[method];
  const snapshot = JSON.stringify([method, mode, mode === 'quantile' ? quantile : direct]);
  const outcome = mode === 'quantile'
    ? quantileOutcomes[method].find(row => row[0] === quantileIndex(quantile, rowCounts.calibration)).slice(1)
    : sweep.reduce((best, row) => (Math.abs(row[0] - direct) < Math.abs(best[0] - direct) ? row : best), sweep[0]);
  const threshold = outcome[OUTCOME.threshold];
  const hits = windowHits(outcome[OUTCOME.hits]);
  const hitCount = hits.filter(Boolean).length;
  const stale = committed !== null && committed.snapshot !== snapshot;
  const shown = revealed && !stale;

  const commit = () => {
    setCommitted({ snapshot, hit: hitChoice, band: bandChoice });
    setRevealed(false);
  };
  const reset = () => {
    setMethod('isolation'); setMode('quantile'); setQuantile(0.975); setDirect(0.6); setWindow(0);
    setOnlyAlerts(true); setHitChoice(''); setBandChoice(''); setCommitted(null); setRevealed(false); setRowPage(0);
  };

  // Overview geometry. Bins keep the level extremes, so the band is the real
  // envelope rather than a smoothed curve.
  const levels = overviewBins.flatMap(bin => [bin[2], bin[3]]);
  const lowLevel = Math.min(...levels);
  const highLevel = Math.max(...levels);
  const scaleX = row => 36 + 288 * row / rowCounts.total;
  const scaleLevel = value => 96 - 76 * (value - lowLevel) / (highLevel - lowLevel);
  const scoreColumn = 5 + methodOrder.indexOf(method);
  const scored = overviewBins.filter(bin => bin[scoreColumn] !== null).map(bin => bin[scoreColumn]);
  const lowScore = Math.min(...scored, threshold);
  const highScore = Math.max(...scored, threshold);
  const scaleScore = value => 78 - 50 * (value - lowScore) / (highScore - lowScore || 1);
  const windowRows = eventWindows.map((pair, index) => {
    const detail = windowDetail[index];
    return { index, from: detail.firstRow + detail.insideFrom, to: detail.firstRow + detail.insideTo, pair };
  });

  const detail = windowDetail[window];
  const detailScores = detail.score[method];
  const detailExceeds = detail.exceeds[method];
  const thresholdRank = (() => {
    // `exceeds` counts thresholds below a row's score in the ascending order of
    // the quantile and sweep tables combined, so find this threshold's place in
    // that same order.
    const all = [...quantileOutcomes[method].map(row => row[1]), ...sweep.map(row => row[0])].sort((left, right) => left - right);
    return all.findIndex(value => value === threshold);
  })();
  const detailAlert = index => detailExceeds[index] > thresholdRank;
  const detailRows = detail.level.map((level, index) => ({ index, level, change: detail.change[index], score: detailScores[index], alert: detailAlert(index), inside: index >= detail.insideFrom && index <= detail.insideTo }));
  // Filtering to the alerting rows, or counting them, would answer the
  // prediction before it is made, so both wait for the reveal.
  const filtered = onlyAlerts && shown;
  const inspectedRows = filtered ? detailRows.filter(row => row.alert) : detailRows;
  const pageSize = 60;
  const lastPage = Math.max(0, Math.ceil(inspectedRows.length / pageSize) - 1);
  const currentPage = Math.min(rowPage, lastPage);
  const visibleRows = inspectedRows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const alertRowCount = detailRows.filter(row => row.alert).length;

  return <Investigation
    title="Choose a threshold on the real series, then meet its workload"
    question={`The detectors are already fitted on the reference period and every later row is already scored. What is left is the decision: where to put the threshold, and what that costs. ${integer(rowCounts.calibration)} calibration rows set it; ${integer(rowCounts.test)} later rows pay for it.`}
    onReset={reset}>
    <div className="ad-controls">
      <Field label="Method">
        <select value={method} onChange={event => setMethod(event.target.value)}>
          {methodOrder.map(key => <option key={key} value={key}>{methodLabels[key]}</option>)}
        </select>
      </Field>
      <Field label="How the threshold is set">
        <select value={mode} onChange={event => setMode(event.target.value)}>
          <option value="quantile">A quantile of the calibration scores</option>
          <option value="direct">A score typed on this method's own scale</option>
        </select>
      </Field>
      {mode === 'quantile'
        ? <NumberField label="Calibration quantile q" value={quantile} min={0.9} max={1} step="0.005" decimals={3} onChange={setQuantile} />
        : <NumberField label="Threshold on the score scale" value={direct} min={Math.floor(sweep[0][0] * 100) / 100} max={Math.ceil(sweep.at(-1)[0] * 100) / 100} step="0.05" decimals={3} onChange={setDirect} />}
    </div>
    {mode === 'direct' && <p className="ad-caption">
      This method's scores run from about {round(sweep[1][0], 3)} to {round(sweep.at(-2)[0], 3)}; a One-Class SVM score is negative throughout, which is a scale, not an error.
      Typed values snap to the nearest of {sweep.length} precomputed levels, and the level actually used is named with the result.
    </p>}

    <div className="ad-prediction">
      <p><strong>Predict first.</strong> Commit both answers for the settings above, then reveal them together.</p>
      <div className="ad-controls">
        <label className="ad-field" htmlFor={hitId}><span>How many of the four annotated windows contain at least one alert?</span>
          <select id={hitId} value={hitChoice} onChange={event => setHitChoice(event.target.value)}>
            <option value="">Choose a prediction</option>
            {[0, 1, 2, 3, 4].map(value => <option key={value} value={String(value)}>{value} of 4</option>)}
            <option value="unsure">I am not sure</option>
          </select>
        </label>
        <label className="ad-field" htmlFor={bandId}><span>How many alerts land outside every window?</span>
          <select id={bandId} value={bandChoice} onChange={event => setBandChoice(event.target.value)}>
            <option value="">Choose a prediction</option>
            {workloadBands.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
            <option value="unsure">I am not sure</option>
          </select>
        </label>
      </div>
      <div className="ad-buttons">
        <button type="button" disabled={!hitChoice || !bandChoice || (committed !== null && !stale)} onClick={commit}>Commit both predictions</button>
        <button type="button" className="is-primary" disabled={committed === null || stale || revealed} onClick={() => setRevealed(true)}>Reveal this threshold's result</button>
      </div>
      {stale && <p className="ad-stale" role="status">The method or threshold changed after your last prediction, so that answer is retired. Commit again for the settings now on screen.</p>}
      {shown && <p className={`ad-verdict ${committed.hit === String(hitCount) && committed.band === bandOf(outcome[OUTCOME.outside]) ? 'is-match' : committed.hit === 'unsure' || committed.band === 'unsure' ? 'is-unsure' : 'is-miss'}`} role="status">
        <span className="ad-verdict-mark" aria-hidden="true">{committed.hit === String(hitCount) && committed.band === bandOf(outcome[OUTCOME.outside]) ? '=' : committed.hit === 'unsure' || committed.band === 'unsure' ? '?' : '≠'}</span>
        Windows with an alert: {hitCount} of 4{committed.hit === 'unsure' ? '' : `, you said ${committed.hit}`}. Alerts outside every window: {integer(outcome[OUTCOME.outside])}
        {committed.band === 'unsure' ? '' : `, you said ${workloadBands.find(([key]) => key === committed.band)[1].toLowerCase()}`}.
      </p>}
    </div>

    <figure className="ad-plot">
      <figcaption>The whole series: recorded temperature, the three periods, and the four annotated windows</figcaption>
      <svg viewBox="0 0 340 222" role="img" aria-label={`Temperature from ${time(0)} to ${time(rowCounts.total - 1)}, drawn as ${overviewBins.length} bins that keep each bin's lowest and highest reading. The first ${integer(rowCounts.reference)} rows are the reference period, the next ${integer(rowCounts.calibration)} the calibration period, and the remaining ${integer(rowCounts.test)} the inspected period. Four annotated windows are shaded. ${shown ? `At the chosen threshold ${integer(outcome[OUTCOME.testAlerts])} later rows alert, ${integer(outcome[OUTCOME.inside])} of them inside a window.` : 'Alert marks appear once a prediction is committed and revealed.'}`}>
        <rect className="ad-span" x={scaleX(0)} y="16" width={scaleX(rowCounts.reference) - scaleX(0)} height="80" fillOpacity=".1" />
        <rect className="ad-span" x={scaleX(rowCounts.reference)} y="16" width={scaleX(rowCounts.reference + rowCounts.calibration) - scaleX(rowCounts.reference)} height="80" fillOpacity=".22" />
        {windowRows.map(entry => (
          <rect key={entry.index} className="ad-span is-selected" x={scaleX(entry.from)} y="16"
            width={Math.max(1.5, scaleX(entry.to) - scaleX(entry.from))} height="80" />
        ))}
        <path fill="#8eb9a5" fillOpacity=".55" stroke="none"
          d={`${overviewBins.map(bin => `${bin === overviewBins[0] ? 'M' : 'L'}${scaleX(bin[0])},${scaleLevel(bin[3])}`).join(' ')} ${[...overviewBins].reverse().map(bin => `L${scaleX(bin[0])},${scaleLevel(bin[2])}`).join(' ')} Z`} />
        <line className="ad-axis" x1="36" x2="324" y1="96" y2="96" />
        <text x="36" y="12">recorded temperature</text>
        <text x="30" y={scaleLevel(highLevel) + 4} textAnchor="end">{round(highLevel, 0)}</text>
        <text x="30" y={scaleLevel(lowLevel) + 4} textAnchor="end">{round(lowLevel, 0)}</text>
        {[0, 0.5, 1].map(fraction => {
          const row = Math.round(fraction * (rowCounts.total - 1));
          return <text key={fraction} x={scaleX(row)} y="108" textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}>{time(row).slice(0, 10)}</text>;
        })}
        <g transform="translate(0 118)">
          <text x="36" y="6">{methodLabels[method]}</text>
          <text x="36" y="18">highest score in each bin{shown ? ', with the threshold' : ''}</text>
          <polyline className="ad-curve" stroke="#91aecf" strokeWidth="1.4"
            points={overviewBins.filter(bin => bin[scoreColumn] !== null).map(bin => `${scaleX(bin[0])},${scaleScore(bin[scoreColumn])}`).join(' ')} />
          {shown && <>
            <line className="ad-threshold" x1="36" x2="316" y1={scaleScore(threshold)} y2={scaleScore(threshold)} />
            <text x="320" y={scaleScore(threshold) + 4}>tau</text>
            {overviewBins.filter(bin => bin[scoreColumn] !== null && bin[scoreColumn] > threshold).map(bin => (
              <rect key={bin[0]} x={scaleX(bin[0])} y="84" width="1.4" height="7" fill="#e7b94a" />
            ))}
            <text x="36" y="101">each mark is a bin with an alerting row</text>
          </>}
        </g>
      </svg>
    </figure>

    {shown && <>
      <p className="ad-readout" aria-live="polite">
        {mode === 'quantile'
          ? `Quantile ${round(quantile, 3)} of the ${integer(rowCounts.calibration)} calibration scores is ${round(threshold, 6)}.`
          : `Your ${round(direct, 3)} snapped to the nearest available level, ${round(threshold, 6)}.`}
        {' '}It leaves {integer(outcome[OUTCOME.calibrationAlerts])} calibration alerts out of {integer(rowCounts.calibration)},
        and {integer(outcome[OUTCOME.testAlerts])} alerts out of {integer(rowCounts.test)} later rows:
        {' '}{integer(outcome[OUTCOME.inside])} inside a window and {integer(outcome[OUTCOME.outside])} outside every window.
        Windows with at least one alert: {hits.map((hit, index) => (hit ? index + 1 : null)).filter(Boolean).join(', ') || 'none'}.
      </p>
      <Table caption={mode === 'quantile'
        ? 'The same calibration quantile, method by method, so the workload can be compared on equal footing'
        : 'A typed score belongs to one method’s scale, so the others cannot be read at the same number'}
        headings={['method', 'threshold', 'calibration alerts', 'test alerts', 'inside windows', 'outside windows', 'windows hit']}
        rows={methodOrder.map(key => {
          const row = mode === 'quantile'
            ? quantileOutcomes[key].find(entry => entry[0] === quantileIndex(quantile, rowCounts.calibration)).slice(1)
            : null;
          return row === null
            ? [methodLabels[key], key === method ? round(threshold, 6) : 'other scale', '—', '—', '—', '—', '—']
            : [methodLabels[key], round(row[OUTCOME.threshold], 6), integer(row[OUTCOME.calibrationAlerts]),
              integer(row[OUTCOME.testAlerts]), integer(row[OUTCOME.inside]), integer(row[OUTCOME.outside]),
              `${windowHits(row[OUTCOME.hits]).filter(Boolean).length}/4`];
        })}
        rowClass={index => (methodOrder[index] === method ? 'is-selected' : undefined)} />
      <p className="ad-caption">
        A first alert near a window's start is early relative to an annotation boundary, not evidence of warning before a fault began.
      </p>
    </>}

    <details>
      <summary>Inspect the actual rows around one annotated window</summary>
      <div className="ad-buttons">
        <button type="button" disabled={window === 0} onClick={() => { setWindow(window - 1); setRowPage(0); }}>Previous window</button>
        <button type="button" disabled={window === eventWindows.length - 1} onClick={() => { setWindow(window + 1); setRowPage(0); }}>Next window</button>
        <span>Window {window + 1} of {eventWindows.length}: {eventWindows[window][0].slice(0, 16)} to {eventWindows[window][1].slice(0, 16)}</span>
        <button type="button" disabled={!shown} onClick={() => { setOnlyAlerts(!onlyAlerts); setRowPage(0); }}>{onlyAlerts ? 'Show every row in this block' : 'Show only alerting rows'}</button>
        {!shown && <span>Every row is shown until you commit and reveal; filtering to the alerting rows would answer the question for you.</span>}
      </div>
      <figure className="ad-plot">
        <figcaption>Window {window + 1} with six hours of context on each side{shown ? `, and the ${integer(alertRowCount)} rows that alert at this threshold` : ''}</figcaption>
        <svg viewBox="0 0 340 110" role="img" aria-label={`Temperature across ${detail.level.length} rows from ${time(detail.firstRow)} to ${time(detail.firstRow + detail.level.length - 1)}, with the annotated window shaded.${shown ? ` ${integer(alertRowCount)} of these rows alert.` : ''}`}>
          {(() => {
            const low = Math.min(...detail.level);
            const high = Math.max(...detail.level);
            const x = index => 36 + 288 * index / (detail.level.length - 1);
            const y = value => 84 - 64 * (value - low) / (high - low || 1);
            return <>
              <rect className="ad-span is-selected" x={x(detail.insideFrom)} y="16" width={Math.max(1, x(detail.insideTo) - x(detail.insideFrom))} height="68" />
              <polyline className="ad-curve" stroke="#d9d3bf" strokeWidth="1.3" points={detail.level.map((value, index) => `${x(index)},${y(value)}`).join(' ')} />
              {shown && detailRows.filter(row => row.alert).map(row => <rect key={row.index} x={x(row.index)} y="86" width="1.3" height="7" fill="#e7b94a" />)}
              <line className="ad-axis" x1="36" x2="324" y1="84" y2="84" />
              <text x="30" y="20" textAnchor="end">{round(high, 1)}</text>
              <text x="30" y="88" textAnchor="end">{round(low, 1)}</text>
              <text x="36" y="104">{time(detail.firstRow)}</text>
              <text x="324" y="104" textAnchor="end">{time(detail.firstRow + detail.level.length - 1)}</text>
              {shown && <text x="36" y="12">each mark below the axis is one alerting row</text>}
            </>;
          })()}
        </svg>
      </figure>
      <div className="ad-buttons" aria-label="Inspect rows in this window">
        <button type="button" disabled={currentPage === 0} onClick={() => setRowPage(currentPage - 1)}>Previous rows</button>
        <span role="status">{inspectedRows.length === 0 ? 'No rows match this filter' : `Rows ${currentPage * pageSize + 1}–${currentPage * pageSize + visibleRows.length} of ${integer(inspectedRows.length)}`}</span>
        <button type="button" disabled={currentPage === lastPage} onClick={() => setRowPage(currentPage + 1)}>Next rows</button>
      </div>
      <Table scroll caption={`${filtered ? `Alerting rows in this block: ${integer(alertRowCount)}` : `Rows in this block: ${integer(detailRows.length)}`}, ${visibleRows.length} shown on this page. Levels, changes and scores are rounded for display; the alert mark uses the exact comparison.`}
        headings={['timestamp', 'level', 'one-hour change', `${methodLabels[method]} score`, 'in window', 'alerts']}
        rows={visibleRows.map(row => [
          time(detail.firstRow + row.index), round(row.level, 2), round(row.change, 2), round(row.score, 4),
          row.inside ? `window ${window + 1}` : 'outside', shown ? (row.alert ? 'yes' : 'no') : 'hidden until revealed',
        ])}
        rowClass={index => (visibleRows[index].alert && shown ? 'is-alert' : undefined)} />
      {!shown && <p className="ad-caption">The score column is shown throughout; whether each row clears the threshold appears once you reveal.</p>}
    </details>

    <details>
      <summary>The worked comparison the lesson already ran, at 0.95 and 0.99</summary>
      <Table caption="Every method at both published quantiles. All four windows are hit in all eight rows; the workload is what moves."
        headings={['method', 'q', 'threshold', 'test alerts', 'inside windows', 'outside windows', 'windows hit']}
        rows={methodOrder.flatMap(key => ['0.95', '0.99'].map(q => {
          const row = publishedOutcomes[key][q];
          return [methodLabels[key], q, round(row[OUTCOME.threshold], 6), integer(row[OUTCOME.testAlerts]),
            integer(row[OUTCOME.inside]), integer(row[OUTCOME.outside]), `${windowHits(row[OUTCOME.hits]).filter(Boolean).length}/4`];
        }))} />
      <p className="ad-caption">
        Moving from the 95th to the 99th calibration percentile keeps every window hit and removes thousands of alerts for some methods and almost none for others. An event metric that only asks “did any alert land inside” cannot see that difference, which is why the row counts sit beside it.
      </p>
    </details>

    <p className="ad-caption">
      The series holds {integer(sourceFacts.rawRows)} raw rows at {integer(sourceFacts.uniqueTimestamps)} distinct timestamps; readings sharing a timestamp were averaged, and the {sourceFacts.droppedMissingLag} rows with no reading exactly one hour earlier were dropped rather than filled. Temperature units and timezone are not stated by the source, so neither is invented here.
    </p>
  </Investigation>;
}
