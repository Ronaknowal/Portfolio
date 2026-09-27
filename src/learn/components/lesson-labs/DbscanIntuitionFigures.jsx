import './dbscan-intuition.css';

export function DensityPersistenceAreaFigure() {
  return <figure className="db-intuition" data-figure="density-persistence-area"><svg viewBox="0 0 325 235" role="img" aria-label="An abstract condensed branch has three rows from lambda two to three and two rows from lambda three to five. The area is three times one plus two times two, equal to seven.">
    <text x="20" y="18">rows remaining in this branch</text>
    <path d="M45 40V180H295" fill="none" stroke="#888" />
    <path d="M70 180V54H135V96H265V180Z" fill="#493a1b" stroke="#e8b44a" strokeWidth="2" />
    {[1, 2, 3].map(n => <text key={n} x="31" y={185-n*42} textAnchor="end">{n}</text>)}
    {[2, 3, 4, 5].map(n => <text key={n} x={70+(n-2)*65} y="203" textAnchor="middle">{n}</text>)}
    <text x="167" y="228" textAnchor="middle">λ = 1 / radius</text><text x="100" y="145" textAnchor="middle">3×1</text><text x="199" y="145" textAnchor="middle">2×2</text>
  </svg><figcaption>A declared finite condensed branch with minimum cluster size 2 is born at λ=2. One row leaves at 3 and the remaining two leave at 5. Sum lifetimes: (3−2)+(5−2)+(5−2)=7. Or sum the shaded count-by-density areas: 3×1+2×2=7. The area counts retained rows through density levels; it is not geographic area or a probability. This tree is illustrative, not an Iris fit.</figcaption></figure>;
}

export function NewRowBridgeFigure() {
  const nodes = [[55, 'D: −1'], [162, 'I: 0'], [270, 'E: 1']];
  return <figure className="db-intuition" data-figure="new-row-core-bridge"><svg viewBox="0 0 325 270" role="img" aria-label="Before adding another observation at zero, I has three neighbors and is border between D and E. After adding I prime at zero, both zero rows have four neighbors, are core and connect the former components.">
    <text x="16" y="18">Before: I cannot transmit</text>
    <path d="M55 61H270" stroke="#888" strokeDasharray="5 5" />
    {nodes.map(([x, label], i) => <g key={label}><circle cx={x} cy="61" r="7" fill={i===1?'#101010':'#e8b44a'} stroke="#e8b44a" strokeWidth="2" /><text x={x} y="90" textAnchor="middle">{label}</text></g>)}
    <text x="162" y="114" textAnchor="middle">I's neighbors: D, I, E</text>
    <text x="16" y="150">Refit: both zero rows are core</text>
    <path d="M55 203L162 182L270 203L162 224Z M162 182V224" stroke="#e8b44a" fill="none" strokeWidth="2" />
    {[[55,203],[162,182],[162,224],[270,203]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="6" fill="#e8b44a" />)}
    <text x="55" y="253" textAnchor="middle">D</text><text x="270" y="253" textAnchor="middle">E</text><text x="176" y="178">I</text><text x="176" y="237">I′</text>
  </svg><figcaption>Use the original trail, ε=1 and m=4. Add a distinct observation I′ at the same position as I. Each zero row now counts D, E, I and I′ and becomes core. The graph acquires a path between the former components. I and I′ are offset vertically only to show two row identities; both physical positions are x=0. Other original core neighbors of D and E are omitted from this graph excerpt; J remains isolated.</figcaption></figure>;
}
