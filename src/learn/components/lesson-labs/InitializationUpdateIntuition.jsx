import './initialization-update-intuition.css';

export function CorrelatedUpdateFigure() {
  const input = [1, -1, 1, -1];
  const rate = 0.01;
  return <figure className="init-update-intuition">
    <h4>Gradient contributions can add even when the inputs alternate signs</h4>
    <p>One linear output z = Σwᵢxᵢ starts with w = 0, target y = 1 and loss ½(z − 1)². One SGD step uses rate η = 0.01.</p>
    <div className="init-update-terms">{input.map((value, index) => <section key={index}><h5>Term {index + 1}</h5><p>xᵢ = {value}</p><p>Gradient = −xᵢ<br />Δwᵢ = {rate * value}</p><p className="init-update-product">Δwᵢxᵢ = +{rate.toFixed(2)}</p></section>)}</div>
    <p>The four changes sum to Δz = <strong>0.04</strong>. Each update learned its sign from the same input later used in the sum.</p>
    <div className="init-update-table" role="region" tabIndex={0} aria-label="Random and aligned update scales"><table><caption>Same per-weight change magnitude η; different dependence on the input</caption><thead><tr><th scope="col">Number of ±1 inputs</th><th scope="col">Independent random ±η changes: output RMS</th><th scope="col">This gradient update: output change</th></tr></thead><tbody>{[4, 16, 64].map(width => <tr key={width}><th scope="row">{width}</th><td>η√n = {(rate * Math.sqrt(width)).toFixed(2)}</td><td>ηn = {(rate * width).toFixed(2)}</td></tr>)}</tbody></table></div>
    <figcaption>Random-change RMS is an expectation over independent sign draws; gradient change is exact for this constructed state. Neither is a measured deep-network training curve. This SGD example explains why independence cannot be reused after an update; it does not derive the later architecture-specific Adam μP recipe.</figcaption>
  </figure>;
}
