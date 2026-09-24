import { useState } from "react";
import { bridgeSpectrum, graphEnergy, multiply, dot, formatSpectral as fmt } from '../../data/spectral-graph-models.js';
import { SpectralNodeGraph } from './SpectralGraphLabs.jsx';
import { LessonTable } from "./LessonElements";
export default function SpectralLab() {
  const [weight, setWeight] = useState(.2);
  const [mode, setMode] = useState(1);
  const graph = bridgeSpectrum(weight);
  const vector = graph.vectors[mode];
  const energy = graphEnergy(graph, vector);
  const residual = multiply(graph.laplacian, vector).map((value, i) => value - graph.values[mode] * vector[i]);
  return <section className="lesson-lab" aria-label="Graph spectrum lab">
    <h3>Two groups, one adjustable bridge</h3>
    <p>Each triangle has internal edge weight 1. Change the C–D bridge and inspect a Laplacian eigenvector. Node colours and signed numbers show its values; positions only organise the drawing.</p>
    <div className="lesson-controls">
      <label>Bridge weight: {weight.toFixed(2)}<input aria-label="Bridge weight" type="range" min="0" max="2" step=".05" value={weight} onChange={e => setWeight(+e.target.value)} /></label>
      <label>Eigenvector<select aria-label="Eigenvector" value={mode} onChange={e => setMode(+e.target.value)}>{graph.values.map((v, i) => <option value={i} key={i}>u{i + 1}: λ = {v.toFixed(3)}{i === 1 ? " (Fiedler)" : ""}</option>)}</select></label>
    </div>
    <SpectralNodeGraph graph={graph} signal={vector} label={`Two triangles with bridge weight ${weight}; eigenvector u${mode + 1}`} />
    <div className="lesson-results" aria-live="polite"><strong>λ₂ = {graph.values[1].toFixed(4)}</strong> · {weight === 0 ? "Two disconnected components: two zero eigenvalues. The zero-eigenspace basis is not unique; do not interpret this arbitrary second vector as a unique partition." : "One connected component. At a weak bridge, the Fiedler values split the two triangles by sign."}</div>
    <p className="lesson-note">Select u₁ to see a constant signal when connected, then u₂ to see the slow variation across the bridge. Higher modes vary more across edges. Eigenvector signs may be flipped without changing their meaning; repeated eigenvalues also allow rotations of the basis.</p>
    <p>Selected unit mode: edge energy {fmt(energy.total, 6)}; squared length {fmt(dot(vector, vector), 6)}; residual length {fmt(Math.sqrt(dot(residual, residual)), 6)}. The energy equals its eigenvalue up to floating-point error.</p>
    <p className="lesson-note">Near-zero eigenvalues in the four-decimal table are rounded numerical results. Components are counted from positive edges, not a rounded display. Line width is 1+2×weight in drawing units; lengths only arrange nodes. On narrow screens the same triangles stack vertically so both groups remain visible.</p>
    <div className="lesson-controls"><button type="button" onClick={() => {
        setWeight(0.2);
        setMode(1);
      }}>Reset spectrum</button></div>
    <LessonTable caption="Where the selected mode spends its energy" headers={['Edge', 'Weight', 'Difference', 'Weighted squared difference']} rows={energy.terms.map(term => [`${'ABCDEF'[term.i]}–${'ABCDEF'[term.j]}`, fmt(term.weight), fmt(term.difference), fmt(term.energy, 6)])} />
    <LessonTable caption="Eigenvalues, ordered by signal energy" headers={["Mode", "Eigenvalue λ"]} rows={graph.values.map((v, i) => [`u${i + 1}`, v.toFixed(4)])} />
    <LessonTable caption={`Node values in u${mode + 1}`} headers={["Node", "Signal value"]} rows={vector.map((v, i) => ["ABCDEF"[i], v.toFixed(4)])} />
  </section>;
}
