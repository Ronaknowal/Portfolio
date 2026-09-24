import './decision-theory-labs.css';
export function ItemDecisionFigure() {
  return <figure className="decision-figure" aria-label="Two available actions and their uncertain consequences">
    <div className="decision-flow"><strong>Observe item information</strong><span>→</span><strong>Choose an action</strong></div>
    <div className="decision-paths"><div><h4>Release</h4><p>↳ Sound: chance .92 → loss 0</p><p>↳ Faulty: chance .08 → loss 80</p><p>Weighted contributions: 0 + 6.4</p></div>
      <div><h4>Quarantine / rework</h4><p>↳ Sound: chance .92 → loss 10</p><p>↳ Faulty: chance .08 → loss 10</p><p>Weighted contributions: 9.2 + .8</p></div></div>
    <figcaption>The state is an already manufactured condition. The action changes what happens to the item and its loss, while the same state probabilities label both action branches. All numbers are stipulated teaching inputs.</figcaption>
  </figure>;
}
export function InterventionBenefitFigure() {
  return <figure className="decision-figure" aria-label="Compare untreated and treated outcomes for two synthetic groups">
    <div className="decision-paths"><div><h4>Group A: high untreated risk</h4><p>No action → failure probability .80</p><p>Intervene → failure probability .75</p><p>Reduction .05 × avoided loss 50 = 2.5</p><p>After action cost 6: <strong>net benefit −3.5</strong></p></div>
      <div><h4>Group B: lower untreated risk</h4><p>No action → failure probability .30</p><p>Intervene → failure probability .05</p><p>Reduction .25 × avoided loss 50 = 12.5</p><p>After action cost 6: <strong>net benefit +6.5</strong></p></div></div>
    <figcaption>These are declared intervention probabilities for otherwise comparable members of each group. An observed treated group's outcome rate would not, by itself, establish either arrow. Highest baseline risk and highest expected benefit are different rankings.</figcaption>
  </figure>;
}
