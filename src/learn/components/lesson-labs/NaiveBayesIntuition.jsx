import './naive-bayes-intuition.css';

export function ConditionalMixtureFigure() {
  return <figure className="nb-intuition" data-concept="conditional-mixture">
    <figcaption><strong>Independent within each group does not mean independent after mixing groups.</strong></figcaption>
    <p>Constructed equal-size classes. Two binary features A and B are generated independently after the class is chosen.</p>
    <div className="nb-mixture-branches">{[0.2, 0.8].map((probability, index) => <section key={probability}>
      <strong>Class {index}: half the population</strong>
      <div className="nb-mixture-fork"><span>A is on<br />p={probability}</span><span>B is on<br />p={probability}</span></div>
      <p>Both on: {probability} × {probability} = {(probability * probability).toFixed(2)}</p>
    </section>)}</div>
    <p>After pooling: P(A=1)=P(B=1)=0.5, but P(A=1,B=1)=0.5×0.04+0.5×0.64=<strong>0.34</strong>, rather than 0.5×0.5=0.25.</p>
    <p>Seeing A on makes class 1 more plausible; class 1 also makes B on more likely. The hidden class links the pooled observations. Naive Bayes assumes a factorization within each class, not across this mixture.</p>
  </figure>;
}

export function AbsentVersusMissingFigure() {
  return <figure className="nb-intuition" data-concept="absent-versus-missing">
    <figcaption><strong>“The alarm is off” supplies evidence; “the alarm was not measured” may not.</strong></figcaption>
    <p>Constructed population: fault prior 0.2; alarm-positive rates 0.8 under fault and 0.4 under normal. Assume measurement failure itself supplies no extra class information.</p>
    <div className="nb-observation-branches">
      <section><strong>Observed off</strong><span>Fault likelihood: 1−0.8=0.2</span><span>Normal likelihood: 1−0.4=0.6</span><span>Fault weight: 0.2×0.2=0.04</span><span>Normal weight: 0.8×0.6=0.48</span><b>Fault posterior: 0.04/0.52 = 1/13</b></section>
      <section><strong>Unobserved alarm</strong><span>Fault likelihood: 0.8+0.2=1</span><span>Normal likelihood: 0.4+0.6=1</span><span>Fault weight: 0.2×1=0.2</span><span>Normal weight: 0.8×1=0.8</span><b>Fault posterior: unchanged at 0.2</b></section>
    </div>
    <p>Marginalizing means allowing both possible alarm values and adding their probabilities. Encoding an unobserved alarm as zero would silently take the observed-off branch even though no off observation occurred.</p>
  </figure>;
}

function Composition({ firstCount, secondCount, label }) {
  return <div className="nb-composition"><span>{label}</span><div aria-label={`${firstCount} A units and ${secondCount} B units`}>{Array.from({ length: firstCount }, (_, index) => <b key={`a-${index}`}>A</b>)}{Array.from({ length: secondCount }, (_, index) => <i key={`b-${index}`}>B</i>)}</div></div>;
}

export function IntegratedTokenFigure() {
  return <figure className="nb-intuition" data-concept="integrated-token-prediction">
    <figcaption><strong>Two future tokens share the same uncertain vocabulary composition.</strong></figcaption>
    <p>Posterior shapes (4,1) are represented by four A units and one B unit. They are parameters of uncertainty, not five new observed test tokens.</p>
    <div className="nb-observation-branches">
      <section><strong>Freeze the mean table</strong><Composition firstCount={4} secondCount={1} label="First A: 4/5" /><Composition firstCount={4} secondCount={1} label="Second A: still 4/5" /><b>P(AA) = 16/25 = 0.64</b></section>
      <section><strong>Integrate shared uncertainty</strong><Composition firstCount={4} secondCount={1} label="First A: 4/5" /><Composition firstCount={5} secondCount={1} label="Given the first is A: 5/6" /><b>P(AA) = (4/5)(5/6) = 2/3</b></section>
    </div>
    <p>The integrated second step is a conditional factorization of the joint future event AA. It accounts for what the first hypothetical A tells us about the shared composition. It does not use a hidden class label or refit on an observed test answer. At fixed composition the draws remain independent; averaging over an uncertain shared composition creates predictive dependence.</p>
  </figure>;
}
