import './dropout-noise-intuition.css';

const scattered = new Set([0, 2, 4, 6, 8, 12, 16, 18, 24]);
const block = new Set([6, 7, 8, 11, 12, 13, 16, 17, 18]);

export function DropBlockFootprintFigure() {
  return <figure className="drop-noise-intuition">
    <h4>The same number of zeros can remove different evidence</h4>
    <p>Constructed 5 × 5 feature map masks: amber means retained; × means removed. Both remove 9 of 25 positions.</p>
    <div className="drop-noise-panels">{[['Scattered positions', scattered], ['A contiguous 3 × 3 region', block]].map(([name, removed]) => <section key={name}><h5>{name}</h5><div className="drop-noise-grid" role="img" aria-label={`${name}; removed row-major positions ${[...removed].map(i => i + 1).join(', ')}`}>
      {Array.from({ length: 25 }, (_, i) => <span key={i} className={removed.has(i) ? 'drop-noise-removed' : ''} aria-hidden="true">{removed.has(i) ? '×' : '•'}</span>)}
    </div></section>)}</div>
    <figcaption>A local detector can often use neighboring retained activations around scattered gaps. A block removes an entire neighborhood. These are chosen mask shapes, not samples or measured results from the DropBlock algorithm; its center sampling, overlap and rescaling still matter.</figcaption>
  </figure>;
}

export function DropConnectTopologyFigure() {
  const cases = [
    { title: 'One activation decision', matrix: [[2, 0], [2, 0]], output: [2, 2], description: 'Mask [1, 0] on h. Both recipients lose input 2.' },
    { title: 'One decision per connection', matrix: [[2, 0], [0, -2]], output: [2, -4], description: 'Keep only the two diagonal weights. The recipients lose different inputs.' },
  ];
  return <figure className="drop-noise-intuition">
    <h4>Which edges disappear together?</h4>
    <p>Input h = [1, 2]; weight rows [1, 1] and [1, −1]; output is Wh. For this comparison, use inverted scaling with keep probability 0.5.</p>
    <div className="drop-noise-panels">{cases.map(({ title, matrix, output, description }) => <section key={title}><h5>{title}</h5><p>{description}</p><div className="drop-noise-matrix" aria-label={`Effective weight rows ${matrix.map(r => r.join(', ')).join('; ')}`}>{matrix.flat().map((value, i) => <span key={i} className={value === 0 ? 'drop-noise-removed' : ''}>{value}</span>)}</div><p>Recipient 1: {matrix[0][0]} × 1 + {matrix[0][1]} × 2 = <strong>{output[0]}</strong><br />Recipient 2: {matrix[1][0]} × 1 + {matrix[1][1]} × 2 = <strong>{output[1]}</strong></p></section>)}</div>
    <figcaption>Rows are recipients; columns are inputs. Activation masking removes a whole column of the effective weight matrix. Connection masking can retain part of that column. This example applies our inverted convention to both operations; it does not assert that the original DropConnect paper used that convention.</figcaption>
  </figure>;
}

export function LocalNoiseFigure() {
  return <figure className="drop-noise-intuition">
    <h4>Move the random draw to the quantity the loss actually uses</h4>
    <p>Fixed input [1, 2]. Independent Gaussian weights have means [0.5, −0.25] and variances [0.04, 0.01].</p>
    <div className="drop-noise-panels"><section><h5>Draw weights, then sum</h5><p>w₁ = 0.5 + 0.2ε₁<br />w₂ = −0.25 + 0.1ε₂</p><p>z = w₁ + 2w₂<br /><strong>z = 0.2ε₁ + 0.2ε₂</strong></p></section><section><h5>Compute moments, then draw z</h5><p>Mean = 1(0.5) + 2(−0.25) = 0<br />Variance = 1²(0.04) + 2²(0.01) = 0.08</p><p><strong>z = √0.08 ε</strong></p></section></div>
    <figcaption>Each ε is standard normal. Both routes give the same single-example distribution N(0, 0.08), not identical draws. Independent local draws across examples preserve the sum of expected per-example losses; they do not preserve the joint covariance produced by one shared weight draw. Batch-coupled computations need additional analysis.</figcaption>
  </figure>;
}
