import './counting-combinatorics-labs.css';

const tilingCases = [
  { first: 1, remaining: 3, tilings: [[1, 1, 1, 1], [1, 1, 2], [1, 2, 1]] },
  { first: 2, remaining: 2, tilings: [[2, 1, 1], [2, 2]] },
];

export function TilingDecompositionFigure() {
  return <figure className="counting-figure" aria-label="Five length-four tilings split by first tile">
    <div className="counting-tiling-cases">{tilingCases.map(branch => <div key={branch.first}>
      <strong>First tile has length {branch.first}</strong>
      <p>Remaining strip: length {branch.remaining}</p>
      {branch.tilings.map(tiling => <div className="counting-tile-strip" key={tiling.join('-')} aria-label={`Tile lengths ${tiling.join(', ')}`}>
        {tiling.map((length, index) => <span key={index} style={{ gridColumn: `span ${length}` }} className={index === 0 ? 'counting-first-tile' : ''}>{length}</span>)}
      </div>)}
      <p>{branch.tilings.length} complete tilings in this case</p>
    </div>)}</div>
    <figcaption>Each row fills four unit positions; a length-two tile spans two positions. Amber marks the fixed first tile. Remove it to recover a smaller tiling, or put it back to reverse the construction. The first-tile cases contain three and two outcomes, so T₄ = T₃ + T₂ = 5.</figcaption>
  </figure>;
}

function Partition({ blocks }) {
  return <div className="counting-partition" aria-label={`Blocks ${blocks.map(block => block.join('')).join(' and ')}`}>
    {blocks.map(block => <span className="counting-partition-block" key={block.join('')}>{block.map(item => <strong className={item === 'D' ? 'counting-last-item' : ''} key={item}>{item}</strong>)}</span>)}
  </div>;
}

export function PartitionDecompositionFigure() {
  return <figure className="counting-figure" aria-label="Insert D into a new or existing set-partition block">
    <div className="counting-partition-cases">
      <div><strong>D makes its own block</strong><Partition blocks={['ABC'.split('')]} /><span className="counting-branch-arrow">↓ add a new singleton</span><Partition blocks={['ABC'.split(''), ['D']]} /><p>One smaller one-block partition gives one outcome.</p></div>
      <div><strong>D joins an existing block</strong><Partition blocks={['AB'.split(''), ['C']]} /><span className="counting-branch-arrow">↓ choose which of these two blocks receives D</span><Partition blocks={['ABD'.split(''), ['C']]} /><span className="counting-branch-arrow">or</span><Partition blocks={['AB'.split(''), 'CD'.split('')]} /><p>Shown: one smaller two-block partition gives two outcomes. Each of the three smaller two-block partitions does the same.</p></div>
    </div>
    <figcaption>Blocks have no room names, but the block containing A and B differs from the block containing C. Removing D recovers the earlier partition and reveals which case was used. In total, S(4,2) = 1 + 2×3 = 7.</figcaption>
  </figure>;
}
