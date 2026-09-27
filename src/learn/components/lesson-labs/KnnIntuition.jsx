import './knn-intuition.css';

export function NeighborBallBoundFigure() {
  return <figure className="knn-intuition" data-concept="ball-pruning-bound">
    <figcaption><strong>You can rule out an entire group before measuring every member.</strong></figcaption>
    <svg viewBox="0 0 280 210" role="img" aria-label="Query at zero, current neighbor three units away, ball centered six units away with radius two; all ball members are at least four units away">
      <circle cx="190" cy="100" r="50" fill="#1d1d1d" stroke="#bbb" strokeWidth="2" />
      <line x1="40" y1="100" x2="240" y2="100" stroke="#666" />
      <line x1="40" y1="132" x2="140" y2="132" stroke="#e8b84b" strokeWidth="3" />
      <line x1="190" y1="100" x2="240" y2="100" stroke="#ccc" strokeWidth="3" />
      <circle cx="40" cy="100" r="5" fill="#eee" />
      <circle cx="115" cy="100" r="5" fill="#e8b84b" />
      <circle cx="190" cy="100" r="3" fill="#eee" />
      <text x="40" y="83" textAnchor="middle">q</text>
      <text x="115" y="83" textAnchor="middle">best</text>
      <text x="190" y="83" textAnchor="middle">c</text>
      <text x="40" y="180" textAnchor="middle">0</text>
      <text x="115" y="180" textAnchor="middle">3</text>
      <text x="140" y="180" textAnchor="middle">4</text>
      <text x="190" y="180" textAnchor="middle">6</text>
      <text x="240" y="180" textAnchor="middle">8</text>
      <text x="140" y="203" textAnchor="middle">distance along the center line</text>
    </svg>
    <p>Center distance = 6; group radius = 2. Even the closest possible point in the ball is <strong>6 − 2 = 4</strong> away. We already have a neighbor at distance 3, so this group cannot improve a 1NN result. The amber segment is the lower bound, not a measured group member.</p>
    <p>If the current best were 5 away, the bound would no longer exclude a better member. Search the group; a loose bound is permission to inspect, not proof that a better member exists.</p>
  </figure>;
}

export function NeighborNoiseFigure() {
  return <figure className="knn-intuition" data-concept="neighbor-label-noise">
    <figcaption><strong>Perfectly matching inputs can still produce different outcomes.</strong></figcaption>
    <p>Two independent labels, each positive with probability 0.8. Column widths represent the neighbor's label probabilities; row heights represent the new outcome's. Cell area is their product.</p>
    <svg viewBox="0 0 270 280" role="img" aria-label="Independent neighbor and query outcomes: both positive 64 percent, neighbor positive query negative 16 percent, neighbor negative query positive 16 percent, both negative 4 percent">
      <rect x="20" y="36" width="160" height="160" fill="#252525" stroke="#888" />
      <rect x="180" y="36" width="40" height="160" fill="#3e3015" stroke="#e8b84b" />
      <rect x="20" y="196" width="160" height="40" fill="#3e3015" stroke="#e8b84b" />
      <rect x="180" y="196" width="40" height="40" fill="#252525" stroke="#888" />
      <text x="100" y="23" textAnchor="middle">neighbor +</text>
      <text x="206" y="23" textAnchor="middle">−</text>
      <text x="100" y="120" textAnchor="middle">64%</text>
      <text x="200" y="120" textAnchor="middle">16%</text>
      <text x="100" y="221" textAnchor="middle">16%</text>
      <text x="200" y="221" textAnchor="middle">4%</text>
      <text x="232" y="120">+</text><text x="232" y="221">−</text>
      <text x="120" y="263" textAnchor="middle">new label: + top, − below</text>
    </svg>
    <p>Amber cells disagree: <strong>16% + 16% = 32%</strong>. Always choosing the more likely class at this same input makes only 20% errors. A neighbor supplies a sample of the local outcome distribution; one sample is noisier than knowing its majority.</p>
  </figure>;
}
