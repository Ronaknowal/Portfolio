import { useId, useState } from "react";
import { algebraNumber as fmt, parseEquationDraft, equationState, functionProbeState, compositionState, quadraticState, growthState, logarithmState } from '../../data/algebra-functions-models.js';
import { LessonTable } from './LessonElements';
import './algebra-functions-labs.css';
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  return <label className="algebra-control"><span>{label}: <strong>{fmt(value)}</strong></span><input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Plot({
  label,
  xRange,
  yRange,
  curves,
  points = [],
  logY = false,
  xLabel = "input x",
  yLabel = "output y"
}) {
  const clip = useId().replaceAll(":", "");
  const x = value => 44 + (value - xRange[0]) / (xRange[1] - xRange[0]) * 252;
  const transform = value => logY ? Math.log10(value) : value;
  const low = transform(yRange[0]);
  const high = transform(yRange[1]);
  const y = value => 216 - (transform(value) - low) / (high - low) * 168;
  const ticks = logY ? [0.1, 1, 10, 100, 1000, 10000, 100000].filter(v => v >= yRange[0] && v <= yRange[1]) : [yRange[0], (yRange[0] + yRange[1]) / 2, yRange[1]];
  return <svg className="algebra-plot" viewBox="0 0 330 275" role="img" aria-label={label}>
    <defs><clipPath id={clip}><rect x="44" y="45" width="252" height="174" /></clipPath></defs>
    <text x="44" y="24" className="axis-title">{yLabel}</text>
    {ticks.map(t => <g key={t}><line x1="44" x2="296" y1={y(t)} y2={y(t)} className="grid" /><text x="37" y={y(t) + 5} textAnchor="end">{t >= 1000 ? `${t / 1000}k` : fmt(t)}</text></g>)}
    {[xRange[0], (xRange[0] + xRange[1]) / 2, xRange[1]].map((t, index) => <g key={t}><line x1={x(t)} x2={x(t)} y1="48" y2="220" className="grid" /><text x={x(t)} y="243" textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}>{fmt(t)}</text></g>)}
    {xRange[0] <= 0 && xRange[1] >= 0 && <line x1={x(0)} x2={x(0)} y1="48" y2="216" className="axis" />}
    {!logY && yRange[0] <= 0 && yRange[1] >= 0 && <line x1="44" x2="296" y1={y(0)} y2={y(0)} className="axis" />}
    <g clipPath={`url(#${clip})`}>{curves.map((curve, i) => <polyline key={i} fill="none" className={`curve ${curve.style || ""}`} points={curve.values.filter(pair => pair.every(Number.isFinite) && (!logY || pair[1] > 0)).map(pair => `${x(pair[0])},${y(pair[1])}`).join(" ")} />)}
      {points.map((p, i) => <circle key={i} cx={x(p.x)} cy={y(p.y)} r="5.5" className={p.hollow ? "point hollow" : "point"} />)}</g>
    <text x="296" y="265" textAnchor="end" className="axis-title">{xLabel}</text>
  </svg>;
}
export function FractionReflectionFigure() {
  return <figure className="algebra-figure"><div className="algebra-fraction" aria-label="Three quarters plus one half equals five quarters: one whole and one quarter"><div><strong>¾ litre</strong><span>{[0, 1, 2, 3].map(i => <i className={i < 3 ? "filled" : ""} key={i} />)}</span></div><b>+</b><div><strong>½ = ²⁄₄ litre</strong><span>{[0, 1, 2, 3].map(i => <i className={i < 2 ? "filled" : ""} key={i} />)}</span></div><b>=</b><div><strong>⁵⁄₄ litre</strong><span>{[0, 1, 2, 3, 4].map(i => <i className="filled" key={i} />)}</span></div></div>
    <svg viewBox="0 0 330 145" role="img" aria-label="Before reflection −3 is left of 2. Multiplying by −1 sends −3 to 3 and 2 to −2, reversing order."><line x1="20" x2="310" y1="70" y2="70" className="axis" />{[-3, -2, 0, 2, 3].map(n => <g key={n}><line x1={165 + n * 40} x2={165 + n * 40} y1="65" y2="75" className="axis" /><text x={165 + n * 40} y="98" textAnchor="middle">{n}</text></g>)}<path d="M45 60 Q165 -10 285 60 M245 60 Q165 10 85 60" fill="none" className="curve" /><text x="165" y="133" textAnchor="middle">−3 &lt; 2 becomes 3 &gt; −2</text></svg>
    <figcaption>Equal-size quarter pieces can be counted together. On the number line, multiplication by −1 reflects positions about zero, reversing their order. Piece lengths encode quarters; curved arrows show correspondence, not travelled distance.</figcaption></figure>;
}
export function EquationStepsLab() {
  const [draft, setDraft] = useState({
    a: "3",
    b: "6",
    c: "21"
  });
  const [state, setState] = useState(equationState(3, 6, 21));
  const [step, setStep] = useState(0);
  const [error, setError] = useState("");
  function apply(next = draft) {
    try {
      const model = parseEquationDraft(next);
      setState(model);
      setDraft(next);
      setStep(0);
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }
  const frame = state.frames[step];
  return <section className="algebra-lab" aria-label="Equation steps investigation"><h3>Do the same reversible operation to both sides</h3><p>Inspect the answer as you step. Draft edits take effect only after Apply equation. The active equation stays intact after an invalid edit.</p>
    <form onSubmit={event => {
      event.preventDefault();
      apply();
    }} className="algebra-controls">{["a", "b", "c"].map(key => <label key={key}>Coefficient {key}<input aria-label={`Coefficient ${key}`} inputMode="numeric" value={draft[key]} onChange={event => setDraft({
          ...draft,
          [key]: event.target.value
        })} /></label>)}<button>Apply equation</button></form>
    <div className="algebra-buttons">{[["Ordinary", 3, 6, 21], ["Negative slope", -2, 4, 10], ["Every x", 0, 7, 7], ["No x", 0, 7, 8]].map(([name, a, b, c]) => <button type="button" key={name} onClick={() => apply({
        a: String(a),
        b: String(b),
        c: String(c)
      })}>{name}</button>)}</div>
    {error && <p role="alert">{error}</p>}
    <div className="algebra-equation" aria-live="polite"><span>{frame.left}</span><b>=</b><span>{frame.right}</span></div><p className="algebra-readout">Step {step + 1} of 3. {frame.action}</p>
    <div className="algebra-buttons"><button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous step</button><button disabled={step === 2} onClick={() => setStep(step + 1)}>Next step</button><button onClick={() => setStep(0)}>Restart steps</button></div>
    {step === 2 && <p className="algebra-result">{state.kind === "one" ? `One solution: x ≈ ${fmt(state.solution)}. Check the exact fraction in ${state.a}x + (${state.b}) = ${state.c}.` : state.kind === "all" ? "All real x solve the original equation." : "No real x solves the original equation."}</p>}
    <p className="lesson-note">The numerator and denominator remain exact integers; a displayed decimal is rounded. Transfer: with a=−3, b=5, c=−7, what changes and what stays reversible?</p>
  </section>;
}
export function FunctionProbeLab() {
  const [kind, setKind] = useState("square");
  const [input, setInput] = useState(2);
  const [restricted, setRestricted] = useState(false);
  const state = functionProbeState(kind, input, restricted);
  const values = Array.from({
    length: 161
  }, (_, i) => -4 + i / 20);
  const evaluate = x => kind === "affine" ? 2 * x + 1 : kind === "square" ? x * x : 1 / x;
  const groups = kind === "reciprocal" ? [values.filter(x => x < 0), values.filter(x => x > 0)] : [values.filter(x => kind !== "square" || !restricted || x >= 0)];
  return <section className="algebra-lab" aria-label="Function and inverse investigation"><h3>A whole rule, one queried value</h3><p>Move the input and trace its output. With the square rule, predict whether the output identifies just one input.</p><div className="algebra-controls"><label>Rule<select aria-label="Function rule" value={kind} onChange={event => setKind(event.target.value)}><option value="square">Square: x²</option><option value="affine">Affine: 2x + 1</option><option value="reciprocal">Reciprocal: 1/x</option></select></label><Range label="Function input" value={input} onChange={setInput} min={-4} max={4} step={0.25} />{kind === "square" && <label><input type="checkbox" checked={restricted} onChange={event => setRestricted(event.target.checked)} /> Restrict square domain to x ≥ 0</label>}</div>
    <Plot label="Calculated function curve and selected input-output point" xRange={[-4, 4]} yRange={kind === "square" ? [-2, 18] : [-10, 10]} curves={groups.map(group => ({
      values: group.map(x => [x, evaluate(x)])
    }))} points={state.allowed ? [{
      x: input,
      y: state.y
    }] : []} />
    <p className="algebra-result" aria-live="polite">{state.allowed ? `f(${fmt(input)}) = ${fmt(state.y)}. Inputs returning this output: ${state.preimages.map(fmt).join(" and ")}.` : "This input is outside the selected domain. There is no output; it is not zero."}</p>
    <p>{kind === "square" ? restricted ? "The allowed branch has x ≥ 0, range y ≥ 0, and inverse √y. The input control still lets you test excluded negative queries." : "The whole square function has domain all real x and range y ≥ 0. Equal heights at x and −x prevent a single-valued inverse." : kind === "affine" ? "Domain and range are all real numbers. The inverse sends y to (y−1)/2." : "Domain excludes x=0; range excludes y=0. The two branches are intentionally not joined across zero."}</p><p className="lesson-note">Calculated samples of the stated formula; the viewing window is not the full domain. Transfer: explain why x² is a function even when it has two preimages for one output.</p>
  </section>;
}
export function CompositionLab() {
  const [input, setInput] = useState(2);
  const state = compositionState(input);
  return <section className="algebra-lab" aria-label="Function composition investigation"><h3>Feed the output into the next rule</h3><p>Here A(x)=2x+1 and S(x)=x² are dimensionless rules. Inspect which ordering gives the larger result; the answer can change with x.</p><Range label="Composition input" value={input} onChange={setInput} min={-4} max={4} step={0.25} />
    <div className="algebra-pipeline"><strong>S(A(x))</strong><span>{fmt(input)}</span><b>→ A →</b><span>{fmt(state.affine)}</span><b>→ S →</b><span>{fmt(state.squareAfterAffine)}</span></div>
    <div className="algebra-pipeline second"><strong>A(S(x))</strong><span>{fmt(input)}</span><b>→ S →</b><span>{fmt(state.square)}</span><b>→ A →</b><span>{fmt(state.affineAfterSquare)}</span></div>
    <p className="algebra-result" aria-live="polite">Square after affine: {fmt(state.squareAfterAffine)}. Affine after square: {fmt(state.affineAfterSquare)}.</p><p>Undo just A in reverse order: {fmt(state.affine)} → subtract 1 → {fmt(state.affine - 1)} → divide by 2 → {fmt(state.recovered)}. Reversing the entire square pipeline also needs a valid square-root branch.</p><p className="lesson-note">Every intermediate is evaluated from the same input. Transfer: test x=−1 and x=0; equality at a few inputs does not make two functions identical.</p>
  </section>;
}
export function QuadraticLab() {
  const [h, setH] = useState(3);
  const [k, setK] = useState(-4);
  const state = quadraticState(h, k);
  const values = Array.from({
    length: 181
  }, (_, i) => -6 + i / 12).map(x => [x, (x - h) ** 2 + k]);
  return <section className="algebra-lab" aria-label="Quadratic roots investigation"><h3>Move the vertex; watch the roots appear or disappear</h3><p>The curve y=(x−h)²+k always opens upward. Explore how many times it meets y=0 while changing the height k.</p><div className="algebra-controls"><Range label="Vertex h" value={h} onChange={setH} min={-2} max={4} step={0.5} /><Range label="Vertex k" value={k} onChange={setK} min={-9} max={4} step={0.5} /></div>
    <Plot label="Parabola with its vertex and real zero crossings" xRange={[-6, 9]} yRange={[-10, 16]} curves={[{
      values
    }]} points={[{
      x: h,
      y: k
    }, ...state.roots.map(x => ({
      x,
      y: 0
    }))]} />
    <p className="algebra-result">y=(x−({fmt(h)}))²+({fmt(k)})<br />Expanded: y=x²+({fmt(state.b)})x+({fmt(state.c)})</p><p aria-live="polite">Vertex ({fmt(h)}, {fmt(k)}). {k > 0 ? "No real roots: a nonnegative square cannot equal a negative number." : k === 0 ? `One root x=${fmt(h)}, repeated twice algebraically.` : `Two roots h ± √(−k): approximately ${state.roots.map(fmt).join(" and ")}.`}</p><p className="lesson-note">Exact formula, sampled curve and rounded coordinates. Transfer: make the minimum zero, then positive. A clipped curve outside this window still follows the same formula.</p>
  </section>;
}
export function RationalHoleFigure() {
  return <figure className="algebra-figure"><Plot label="The rational function equals x+1 except for a hole at the excluded input x=1" xRange={[-2, 4]} yRange={[-1, 5]} curves={[{
      values: [[-2, -1], [4, 5]]
    }]} points={[{
      x: 1,
      y: 2,
      hollow: true
    }]} /><figcaption>For (x²−1)/(x−1), the open circle at (1,2) is missing: the original denominator is zero there. Simplifying to x+1 changes the expression’s form, not its inherited domain.</figcaption></figure>;
}
export function GrowthScaleLab() {
  const [rate, setRate] = useState(0.2);
  const [step, setStep] = useState(4);
  const [factor, setFactor] = useState(3);
  const [logY, setLogY] = useState(false);
  const state = growthState(rate, step, factor);
  const maximum = Math.max(260, state.rows[8].growth);
  const high = 10 ** Math.ceil(Math.log10(maximum));
  return <section className="algebra-lab" aria-label="Growth and scale investigation"><h3>Add the same amount, or multiply by the same factor?</h3><p>Both models start at 100 units. Amber multiplies each period; blue adds 20 units. Changing the axis changes spacing, not any value.</p><div className="algebra-controls"><label>Fractional change per period<select aria-label="Growth rate" value={rate} onChange={event => setRate(Number(event.target.value))}>{[-0.5, -0.2, 0, 0.1, 0.2, 0.5, 1].map(value => <option key={value} value={value}>{value * 100}% per period</option>)}</select></label><Range label="Observation period" value={step} onChange={setStep} min={0} max={8} /><label><input type="checkbox" checked={logY} onChange={event => setLogY(event.target.checked)} /> Use logarithmic quantity axis</label><label>Target / initial quantity<select aria-label="Growth target factor" value={factor} onChange={event => setFactor(Number(event.target.value))}>{[0.125, 0.5, 1, 2, 3, 8].map(value => <option key={value} value={value}>{value}× initial</option>)}</select></label></div>
    <Plot label="Calculated additive and repeated-multiplier models on the chosen axis" xRange={[0, 8]} yRange={logY ? [0.1, high] : [0, high]} logY={logY} xLabel="time (periods)" yLabel="quantity (units)" curves={[{
      values: Array.from({
        length: 81
      }, (_, i) => [i / 10, 100 * state.multiplier ** (i / 10)])
    }, {
      style: "second",
      values: Array.from({ length: 81 }, (_, index) => [index / 10, 100 + 2 * index])
    }]} points={[{
      x: step,
      y: state.active.growth
    }]} />
    <p className="algebra-result" aria-live="polite">Period {step}: repeated multiplier {fmt(state.multiplier)} gives {fmt(state.active.growth)} units; fixed addition gives {fmt(state.active.additive)} units.</p>
    <p>{state.crossing === null ? "A constant 100 never reaches this different target." : rate === 0 ? "Every time has the target value 100; zero is the first observed period." : `The continuous model equals the target at t ≈ ${fmt(state.crossing)} periods. ${state.future ? "This is a nonnegative crossing; whole-period decisions need a direct value check." : "This crossing is before t=0, so it is not a future equality."}`}</p>
    <LessonTable caption="Same values under either axis choice" headers={["Period", "Add 20", `Multiply ${fmt(state.multiplier)}`, "Last change"]} rows={state.rows.map(row => [row.t, fmt(row.additive), fmt(row.growth), row.change === null ? "Initial" : fmt(row.change)])} />
    <p className="lesson-note">Calculated models, not measurements. Curves interpolate by the stated real-power rule; actual discrete observations are the integer rows. Transfer: a −20% change multiplies by 0.8, not −0.2.</p>
  </section>;
}
export function LogRulerLab() {
  const [base, setBase] = useState(2);
  const [exponent, setExponent] = useState(1.5);
  const state = logarithmState(base, exponent);
  return <section className="algebra-lab" aria-label="Logarithmic ruler investigation"><h3>Use the exponent as the position</h3><p>The quantity is a positive ratio q/q₀ to a reference q₀. Equal gaps on this ruler mean equal multiplications, even when the quantity labels differ greatly.</p><div className="algebra-controls"><label>Base<select aria-label="Logarithm base" value={base} onChange={event => setBase(Number(event.target.value))}><option value="2">2</option><option value="10">10</option><option value="0.5">0.5</option></select></label><Range label="Exponent position" value={exponent} onChange={setExponent} min={-3} max={3} step={0.25} /></div>
    <svg viewBox="0 0 330 175" className="algebra-ruler" role="img" aria-label={`Exponent ruler, base${base}; ratio${fmt(state.value)} at position${fmt(exponent)}`}><text x="18" y="24">Top: positive ratio q/q₀</text><line x1="25" x2="305" y1="80" y2="80" className="axis" />{state.ticks.filter(t => t.power % 2 !== 0).map(t => <g key={t.power}><line x1={165 + t.power * 46} x2={165 + t.power * 46} y1="71" y2="90" className="axis" /><text x={165 + t.power * 46} y="57" textAnchor="middle">{fmt(t.value)}</text><text x={165 + t.power * 46} y="115" textAnchor="middle">{t.power}</text></g>)}<circle cx={165 + exponent * 46} cy="80" r="6" className="point" /><text x="18" y="156">Bottom: log base {base} of the ratio</text></svg>
    <p className="algebra-result" aria-live="polite">{base}<sup>{fmt(exponent)}</sup> ≈ {fmt(state.value)}<br />log base {base} of {fmt(state.value)} ≈ {fmt(exponent)}</p><p>{state.increasing ? "Larger ratios lie farther right: this base is greater than 1." : "Larger ratios lie farther left: a base between 0 and 1 reverses order."} Ratio 1 is at exponent 0, halfway along this viewing window.</p><p className="lesson-note">Tick spacing represents exponents. Decimal values are rounded; the exact value is base raised to the shown exponent. Transfer: move from exponent −1 to 1 and name the multiplication factor.</p>
  </section>;
}
