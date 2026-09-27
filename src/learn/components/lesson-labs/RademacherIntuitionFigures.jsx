import './rademacher-intuition.css';

export function ChainingCorrectionsFigure() {
  const rows = [{label:'coarse', value:.5}, {label:'finer', value:.75}, {label:'finer again', value:.625}];
  const x = value => 30 + value * 290;
  return <figure className="rademacher-intuition" data-figure="chaining-corrections"><svg viewBox="0 0 350 258" role="img" aria-label="Three approximations to target value .68: .5, .75 and .625. Each row uses the same zero-to-one scale. Dashed line is the target; amber dot is the approximation.">
    <text x="20" y="20">Same target t = .68, finer grids</text>
    {rows.map(({label,value},i)=><g key={label}><text x="20" y={48+i*60}>{label}: {value}</text><path d={`M30 ${66+i*60}H320`} stroke="#777"/><path d={`M${x(.68)} ${55+i*60}V${77+i*60}`} stroke="#ddd" strokeDasharray="3 3"/><path d={`M${x(value)} ${66+i*60}H${x(.68)}`} stroke="#e8b44a" strokeWidth="3"/><circle cx={x(value)} cy={66+i*60} r="5" fill="#e8b44a"/></g>)}
    {[0,.5,1].map(value=><text key={value} x={x(value)} y="221" textAnchor="middle">{value}</text>)}
    <text x="175" y="248" textAnchor="middle">prediction at each input</text>
  </svg><figcaption>Constructed constant prediction vectors fₜ=(t,t), with t in [0,1]. Grids spaced .5, .25 and .125 cover this class in empirical RMS distance with radii .25, .125 and .0625. The highlighted target .68 is represented as .5 + (.75−.5) + (.625−.75) + (.68−.625). Corrections can point either way; their permitted magnitude shrinks with scale. This is a decomposition to understand chaining, not a computed generalization bound.</figcaption></figure>;
}
