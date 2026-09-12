import { sixPoints, sixLabels, sixNames, silhouetteSamples, pairAgreement, informationMeasures, ringContrast, irisFit } from '../../data/clustering-evaluation-models';
import { irisSpecies, irisSpeciesNames, reportSplit } from '../../data/clustering-evaluation-data';
import { SilhouetteBars, Table, number } from './ClusteringEvaluationLabs.jsx';
import './clustering-evaluation-labs.css';

const ids8 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
const Strip = ({ name, labels, marks = [] }) => <div className="ce-strip"><span>{name}</span>{labels.map((label, index) => <span key={index} className={`ce-chip ${marks.includes(index) ? 'is-selected' : ''}`}>{label}</span>)}</div>;

/** V1: identity survives a rename; a membership change does not. */
export function CeIdentityFigure() {
  const u = [0, 0, 0, 0, 1, 1, 1, 1], renamed = [7, 7, 7, 7, 3, 3, 3, 3], changed = [0, 0, 1, 1, 0, 0, 1, 1];
  return <figure className="ce-figure">
    <figcaption><strong>Observation identity survives a rename</strong> · the same eight specimens under three label rows</figcaption>
    <Strip name="ID" labels={ids8} marks={[0, 2, 4]} />
    <Strip name="U" labels={u} marks={[0, 2, 4]} />
    <Strip name="renamed" labels={renamed} marks={[0, 2, 4]} />
    <Strip name="changed" labels={changed} marks={[0, 2, 4]} />
    <div className="ce-ledger">
      <div><strong>A–C</strong>together in U · together when renamed · apart when changed</div>
      <div><strong>A–E</strong>apart in U · apart when renamed · together when changed</div>
      <div><strong>raw label equality</strong>U versus renamed: 0 of 8 positions match</div>
      <div><strong>partition agreement</strong>U versus renamed: unchanged (ARI {number(pairAgreement(u, renamed).ARI)}); U versus changed: ARI {number(pairAgreement(u, changed).ARI)}</div>
    </div>
    <p>Columns keep the specimens in place; only the label values change between rows. The renamed row shares every together and apart decision with U although no label value matches, which is why comparing label arrays position by position answers the wrong question. The changed row keeps the same label values as U and alters the memberships: A and E now sit together, A and C no longer do.</p>
  </figure>;
}

/** V2: C's distance fan and the six genuine silhouette bars. */
export function CeDistanceFanFigure() {
  const result = silhouetteSamples(sixPoints, sixLabels);
  const x = value => 30 + 26 * value;
  return <figure className="ce-figure ce-standalone">
    <figcaption><strong>A distance fan and genuine silhouette bars</strong> · six locations, two groups, C selected</figcaption>
    <svg viewBox="0 40 300 92" role="img" aria-label="Number line 0 to 9 with A, B, C in group L and D, E, F in group R. From C, two solid arcs to A and B have lengths 2 and 1, averaging 1.5; three dashed arcs to D, E, F have lengths 5, 6, 7, averaging 6.">
      <line x1="30" x2="264" y1="80" y2="80" className="ce-grid" strokeWidth="1.5" />
      {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(value => <line key={value} x1={x(value)} x2={x(value)} y1="77" y2="83" className="ce-grid" />)}
      {[0, 1].map(j => <path key={j} className="ce-fan-own" fill="none" d={`M${x(2)},80 Q${(x(2) + x(j)) / 2},${80 - 18 - (2 - j) * 6} ${x(j)},80`} />)}
      {[7, 8, 9].map(j => <path key={j} className="ce-fan-foreign" fill="none" d={`M${x(2)},80 Q${(x(2) + x(j)) / 2},${80 + 14 + (j - 2) * 3} ${x(j)},80`} />)}
      {sixPoints.map((point, index) => <g key={index}><circle cx={x(point[0])} cy="80" r="5" className={`ce-point ${index === 2 ? 'is-selected' : ''}`} /><text x={x(point[0])} y="112" textAnchor="middle">{sixNames[index]}·{sixLabels[index]}</text><text x={x(point[0])} y="126" textAnchor="middle" className="ce-muted">{point[0]}</text></g>)}
    </svg>
    <p className="ce-legend">Locations are printed under each name. Solid arcs to own group L: distances 2 and 1, so a(C) = 1.5. Dashed arcs to group R: distances 5, 6, 7, so b(C) = 6. Then s(C) = (6 − 1.5) / 6 = 0.75.</p>
    <SilhouetteBars labels={[...sixLabels]} values={result.values} names={[...sixNames]} mean={result.mean} title="The six silhouettes, sorted within each group; every bar has equal height" />
    <Table caption="Inputs to each score, in the line's distance units" headings={['ID', 'a', 'b', 's']} rows={sixNames.map((name, index) => [name, number(result.details[index].a), number(result.details[index].b), number(result.details[index].s)])} />
    <p>b is the smallest of the average distances to each other group, not the distance to the nearest single point. The mean marker at {number(result.mean)} averages all six observations equally; a mean of the two group means would be a different summary and must be named as such.</p>
  </figure>;
}

/** V3: the same 32 points under ring and left/right memberships. */
export function CeRingFigure() {
  const rings = ringContrast();
  const x = value => 80 + 32 * value, y = value => 80 - 32 * value;
  const Panel = ({ title, labels, result, groups }) => <div>
    <p>{title}</p>
    <svg viewBox="4 4 152 152" role="img" aria-label={`${title}: thirty-two points on two concentric rings. Mean silhouette ${number(result.mean)}.`}>
      <circle cx="80" cy="80" r="32" fill="none" className="ce-grid" /><circle cx="80" cy="80" r="64" fill="none" className="ce-grid" />
      <line x1="80" x2="80" y1="10" y2="150" className="ce-grid" /><line x1="10" x2="150" y1="80" y2="80" className="ce-grid" />
      {rings.points.map((point, index) => { const cls = `ce-bar-group-${groups.indexOf(labels[index]) % 8}`; return groups.indexOf(labels[index]) === 0 ? <circle key={index} cx={x(point[0])} cy={y(point[1])} r="3.2" className={cls} /> : <rect key={index} x={x(point[0]) - 3} y={y(point[1]) - 3} width="6" height="6" className={cls} />; })}
    </svg>
    <SilhouetteBars labels={labels} values={result.values} mean={result.mean} compact title={`mean silhouette ${number(result.mean)}`} />
  </div>;
  return <figure className="ce-figure">
    <figcaption><strong>A shape mismatch on identical distances</strong> · constructed points, supplied memberships, not fitted results</figcaption>
    <div className="ce-figure-pair">
      <Panel title="Grouped by ring (inner circles, outer squares)" labels={rings.ringLabels} result={rings.ring} groups={['inner', 'outer']} />
      <Panel title="Grouped by side (left circles, right squares)" labels={rings.sliceLabels} result={rings.slice} groups={['left', 'right']} />
    </div>
    <p>Both panels use exactly the same 32 coordinates, drawn with equal units on both axes, and Euclidean distances. Grouping by ring scores about {number(rings.ring.mean, 3)} because points across a ring are far from each other; slicing into left and right scores about {number(rings.slice.mean, 3)} because each half is compact. Silhouette measures average pairwise compactness and separation. A question about connected dense structures needs a different criterion; neither grouping here was discovered by an algorithm.</p>
  </figure>;
}

/** V4: a strict refinement keeps U's one bit and adds a second. */
export function CeInformationFigure() {
  const u = [0, 0, 0, 0, 1, 1, 1, 1], v = [0, 0, 1, 1, 2, 2, 3, 3];
  const info = informationMeasures(u, v);
  return <figure className="ce-figure">
    <figcaption><strong>Information retained versus extra distinctions</strong> · U splits eight specimens in half; V splits each half again</figcaption>
    <Strip name="ID" labels={ids8} /><Strip name="U" labels={u} /><Strip name="V" labels={v} />
    <div className="ce-bars" role="img" aria-label={`Entropy bars: H(U) ${number(info.HU)} bit, H(V) ${number(info.HV)} bits, mutual information ${number(info.I)} bit.`}>
      <div className="ce-bar-row"><span>H(U)</span><span className="ce-bar-track"><span className="ce-bar-fill" style={{ width: '50%' }} /></span><span className="ce-bar-value">{number(info.HU)} bit</span></div>
      <div className="ce-bar-row"><span>H(V)</span><span className="ce-bar-track"><span className="ce-bar-fill" style={{ width: '50%' }} /><span className="ce-bar-fill is-second" style={{ left: '50%', width: '50%' }} /></span><span className="ce-bar-value">{number(info.HV)} bits: 1 shared + 1 extra</span></div>
      <div className="ce-bar-row"><span>I(U;V)</span><span className="ce-bar-track"><span className="ce-bar-fill" style={{ width: '50%' }} /></span><span className="ce-bar-value">{number(info.I)} bit</span></div>
    </div>
    <div className="ce-ledger">
      <div><strong>V determines U</strong>H(U | V) = {number(info.HUgivenV)}; homogeneity {number(info.homogeneity)}</div>
      <div><strong>U leaves two V choices</strong>H(V | U) = {number(info.HVgivenU)} bit; completeness {number(info.completeness)}</div>
      <div><strong>arithmetic NMI</strong>2I / (H(U) + H(V)) = {number(info.nmi)}</div>
      <div><strong>geometric NMI</strong>I / √(H(U)·H(V)) = {number(info.nmiGeometric)}</div>
    </div>
    <Table caption="Occupied contingency cells" headings={['U', 'V', 'IDs']} rows={info.table.rows.flatMap((row, r) => info.table.columns.map((column, c) => [row, column, info.table.cells[r][c].map(i => ids8[i]).join('')]).filter(entry => entry[2]))} />
    <p>The gold length is the one bit U and V share; the green length is the extra bit V adds. Every quantity here comes from this finite table with equal cells, which is why the bar lengths are exact. A `min` normalizer would give this strict refinement a score of 1, so the normalizer must be named with the number.</p>
  </figure>;
}

/** V5: the opening two-versus-three disagreement on raw Iris measurements. */
export function CeIrisSnapshotFigure() {
  const two = irisFit('raw4', 2), three = irisFit('raw4', 3);
  const clusters = [0, 1, 2];
  return <figure className="ce-figure">
    <figcaption><strong>Two groups versus three, raw centimetres, all 150 specimens</strong> · KMeans(n_init=20, random_state=17), scikit-learn 1.9.1</figcaption>
    <Table caption="Same data, two evaluation questions" headings={['fit', 'mean silhouette', 'species ARI', 'species AMI', 'group sizes']} rows={[['k = 2', number(two.silhouette.mean), number(two.ari), number(two.ami), two.sizes.join(' / ')], ['k = 3', number(three.silhouette.mean), number(three.ari), number(three.ami), three.sizes.join(' / ')]]} />
    <Table caption="k = 3: species rows against cluster columns" headings={['species', ...clusters.map(cluster => `cluster ${cluster}`)]} rows={irisSpeciesNames.map((name, species) => [name, ...clusters.map(cluster => three.labels.filter((label, i) => label === cluster && irisSpecies[i] === species).length)])} />
    <p>Two groups are more compact and separated by the silhouette criterion; three groups agree more closely with the recorded species, splitting versicolor and virginica that k = 2 merges. Neither number is wrong. The silhouette column is a geometric question about the four-dimensional distances; the ARI and AMI columns compare two partitions of the same 150 IDs. No single bar should combine them.</p>
  </figure>;
}

/** V6: rejecting one observation changes the denominator. */
export function CeRejectionFigure() {
  const all = silhouetteSamples(sixPoints, ['0', '0', '-1', '1', '1', '1']);
  const retainedPoints = sixPoints.filter((_, i) => i !== 2), retainedLabels = ['0', '0', '1', '1', '1'], retainedNames = sixNames.filter((_, i) => i !== 2);
  const retained = silhouetteSamples(retainedPoints, retainedLabels);
  const base = silhouetteSamples(sixPoints, sixLabels);
  return <figure className="ce-figure">
    <figcaption><strong>The denominator changes when a case is rejected</strong> · labels [0, 0, −1, 1, 1, 1] on the six line points</figcaption>
    <div className="ce-strip"><span>lane 1</span>{sixNames.map((name, index) => <span key={name} className="ce-chip">{name}·{['0', '0', '−1', '1', '1', '1'][index]}</span>)}<span /><span /></div>
    <SilhouetteBars labels={['0', '0', '-1', '1', '1', '1']} values={all.values} names={[...sixNames]} mean={all.mean} title={`All six IDs, −1 treated as an ordinary singleton group: mean ${number(all.mean)}, population 6`} />
    <div className="ce-strip"><span>lane 2</span>{sixNames.map((name, index) => <span key={name} className={`ce-chip ${index === 2 ? 'is-rejected' : ''}`}>{index === 2 ? <><span aria-hidden="true">{name} ✕</span><span className="ce-sr-only">{name} rejected</span></> : name}</span>)}<span /><span /></div>
    <SilhouetteBars labels={retainedLabels} values={retained.values} names={retainedNames} mean={retained.mean} title={`Retained A, B, D, E, F, recomputed: mean ${number(retained.mean)}, coverage 5/6 = 83.33%`} />
    <Table caption="Three different questions about the same six points" headings={['population', 'labels', 'mean silhouette']} rows={[['six, original two groups', 'L L L R R R', number(base.mean)], ['six, −1 as a singleton group', '0 0 −1 1 1 1', number(all.mean)], ['five retained after rejecting C', '0 0 · 1 1 1', number(retained.mean)]]} />
    <p>C's rejection is drawn as a dashed chip marked ✕, not an invisible gap, and the retained bars are recomputed on five observations: A and B now average their distance to R over the same three points but lose C from their own group. The conditional mean is legitimately higher, on fewer observations. It is not a like-for-like improvement over the six-point scores, and the −1 encoding is yet another question.</p>
  </figure>;
}

/** V7: the report protocol as data ownership and information flow. */
export function CeReportFlowFigure() {
  return <figure className="ce-figure">
    <figcaption><strong>The final report has no backward arrow</strong> · 150 specimen IDs permuted by default_rng(23) and split 90 / 30 / 30</figcaption>
    <div className="ce-flow">
      <div className="ce-flow-box"><strong>fit · {reportSplit.fit.length} IDs</strong>learns standardization (mean, scale) and k-means centers for k = 2, 3, 4<span className="ce-flow-arrow">→ scales, centers</span></div>
      <div className="ce-flow-box"><strong>selection · {reportSplit.selection.length} IDs</strong>assigned with the fitted centers; each candidate scored by silhouette on these rows<span className="ce-flow-arrow">→ chosen k = {reportSplit.selectedK} (silhouette {number(reportSplit.selectionSilhouette)})</span></div>
      <div className="ce-flow-box"><strong>report · {reportSplit.report.length} IDs</strong>assigned once by the frozen pipeline; distortion {number(reportSplit.reportDistortion)} against one-center baseline {number(reportSplit.reportBaselineDistortion)}; species ARI {number(reportSplit.reportSpeciesAri)} read afterwards</div>
    </div>
    <Table caption="Candidate comparison on the selection rows" headings={['k', 'training sizes', 'eligible (min size ≥ 10)', 'selection silhouette']} rows={reportSplit.candidates.map(candidate => [candidate.k, candidate.sizes.join(' / '), candidate.eligible ? 'yes' : 'no', number(candidate.silhouette)])} highlight={index => reportSplit.candidates[index].k === reportSplit.selectedK} />
    <details><summary>The exact row IDs in each role</summary><Table caption="Disjoint specimen IDs" headings={['role', 'row IDs']} rows={[['fit', reportSplit.fit.join(', ')], ['selection', reportSplit.selection.join(', ')], ['report', reportSplit.report.join(', ')]]} /></details>
    <p>Objects travel forward only: fitted scales and centers from the fit rows, a chosen k from the selection rows, predicted labels onto the report rows. Species labels bypass fitting and selection and join only the final comparison. Nothing learned from the report rows flows back; if the rule had produced no eligible candidate, that would be the reported outcome.</p>
  </figure>;
}
