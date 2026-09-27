import './decision-tree-intuition.css';

export function GiniPairFigure() {
  const labels = [0, 0, 0, 1];
  return <figure className="tree-intuition" data-concept="gini-pairs">
    <figcaption><strong>Impurity is the chance that two independent labels disagree.</strong></figcaption>
    <p>Constructed leaf: three class-0 rows and one class-1 row. Choose a row, put it back, then choose again. Each of the 16 ordered pairs is equally likely.</p>
    <div className="tree-pair-grid" role="table" aria-label="Six of sixteen ordered pairs have different labels">
      <div className="tree-pair-row" role="row"><span role="columnheader">first ↓<br />second →</span>{labels.map((label, index) => <span role="columnheader" key={`head-${index}`}>{String.fromCharCode(65 + index)}: {label}</span>)}</div>
      {labels.map((first, row) => <div className="tree-pair-row" role="row" key={row}>
        <span role="rowheader">{String.fromCharCode(65 + row)}: {first}</span>
        {labels.map((second, column) => <span className={first !== second ? 'tree-pair-different' : ''} role="cell" key={column}>{first !== second ? '≠' : '='}</span>)}
      </div>)}
    </div>
    <p>Six amber cells disagree: Gini = 6/16 = <strong>0.375</strong>. A majority-class predictor makes one error in four, or <strong>0.25</strong>. These quantities ask different questions, so they need not match.</p>
  </figure>;
}

export function TreePrefixScanFigure() {
  const labels = [0, 0, 1, 1];
  const cuts = [1, 2, 3];
  return <figure className="tree-intuition" data-concept="prefix-scan">
    <figcaption><strong>A threshold moves the boundary; the counts move one row at a time.</strong></figcaption>
    <p>Constructed sorted inputs x = [1, 2, 3, 4], labels = [0, 0, 1, 1]. Parent Gini is 0.5. Every arrow marks a candidate cut between distinct inputs.</p>
    {cuts.map(cut => {
      const left = labels.slice(0, cut);
      const right = labels.slice(cut);
      const gini = rows => { const fraction = rows.reduce((total, label) => total + label, 0) / rows.length; return 2 * fraction * (1 - fraction); };
      const gain = 0.5 - (left.length * gini(left) + right.length * gini(right)) / labels.length;
      return <section className="tree-scan-state" key={cut}>
        <div className="tree-scan-strip">{labels.map((label, index) => <span key={index} className={index === cut - 1 ? 'tree-scan-cut' : ''}><small>x={index + 1}</small><strong>{label}</strong>{index === cut - 1 && <i aria-label="cut">↓</i>}</span>)}</div>
        <p>t = {cut + 0.5}: left counts [{left.filter(value => value === 0).length}, {left.filter(value => value === 1).length}], right counts [{right.filter(value => value === 0).length}, {right.filter(value => value === 1).length}]<br />Gain = <strong>{gain.toFixed(6)}</strong></p>
      </section>;
    })}
    <p>Counts are [class 0, class 1]. Moving from t=1.5 to t=2.5 transfers one class-0 row from right to left. The middle cut makes both children pure. A prefix scan updates counts without rebuilding each child list; this drawing shows its states, not a measured speedup.</p>
  </figure>;
}

export function ForestProximityFigure() {
  const observations = [{ id: 'A', leaves: [1, 2] }, { id: 'B', leaves: [1, 3] }, { id: 'C', leaves: [4, 3] }];
  return <figure className="tree-intuition" data-concept="forest-proximity">
    <figcaption><strong>Similarity can mean “the fitted trees make the same distinctions.”</strong></figcaption>
    <div className="tree-leaf-signatures">{observations.map(row => <div key={row.id}><strong>{row.id}</strong><span>Tree 1 → leaf {row.leaves[0]}</span><span>Tree 2 → leaf {row.leaves[1]}</span></div>)}</div>
    <p>A–B: share Tree 1 only → 1/2. B–C: share Tree 2 only → 1/2. A–C: share neither → 0.</p>
    <p>Each tree supplies a separate set of leaf slots. Put a 1 in the reached slot and 0 elsewhere. Two observations contribute 1 to their dot product for each shared slot; dividing by two gives the shared-leaf fraction. Leaf 3 in one tree is never confused with leaf 3 in another tree.</p>
  </figure>;
}
