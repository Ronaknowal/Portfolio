import './timeseries-intuition.css';

export function PathCoverageFigure() {
  const scenarios = [
    { name: 'Misses spread across paths', misses: (path,day) => path === day, complete: 3 },
    { name: 'Misses coincide on one path', misses: path => path === 0, complete: 9 },
  ];
  return <figure className="ts-intuition" data-figure="daily-versus-path-coverage">
    <div className="ts-coverage-panels">{scenarios.map(scenario=><div key={scenario.name}><strong>{scenario.name}</strong><table><caption>Ten constructed seven-day paths</caption><thead><tr><th>Path</th>{[1,2,3,4,5,6,7].map(day=><th key={day}>D{day}</th>)}</tr></thead><tbody>{Array.from({length:10},(_,path)=><tr key={path}><th>{path+1}</th>{Array.from({length:7},(_,day)=>{const missed=scenario.misses(path,day); return <td key={day} className={missed?'ts-coverage-miss':''} aria-label={missed?'outside interval':'inside interval'}>{missed?'×':'·'}</td>;})}</tr>)}</tbody></table><p>Every day: 9/10 inside<br/>Entire path: {scenario.complete}/10 inside</p></div>)}</div>
    <figcaption>Dot = inside that day's interval; × = outside. Each column has the same 90% empirical coverage in both constructions. Requiring all seven entries in a row to be inside leaves three complete paths when misses are spread across paths, and nine when misses coincide on one path. Daily coverage alone does not determine their joint coverage. These are counting examples, not fitted interval results or an independence assumption.</figcaption>
  </figure>;
}
