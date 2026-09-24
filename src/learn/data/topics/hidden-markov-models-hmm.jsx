import { Callout, H2, H3, Prose, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  BoundaryCountLab, DurationLab, EvidenceLaterLab, LegalPathLab, PathProductLab, RealTaggingLab,
} from '../../components/lesson-labs/HmmLabs.jsx';
import {
  BeliefFigure, CountFlowFigure, DataProvenanceNote, EmHistoryFigure, ForwardTrellisFigure,
  GraphUnrollFigure, NumericScaleFigure, PointwiseFigure, TopologyFigure, ViterbiTrellisFigure,
} from '../../components/lesson-labs/HmmFigures.jsx';
import { hmmExamples } from '../hmm-examples.js';
import {
  configurations, decisionChanges, developmentSentences, emTrack, majority, provenance,
  selectedConfigurationIndex, tieAudit, unknownDevelopmentTokens, vocabulary,
} from '../hmm-data.js';
import {
  countFlow, durationModel, emStep, enumeratePaths, fixtures, infer, parameterCount,
  predictNext, repeatedProduct, trellis, withRainyEmission, withStart,
} from '../hmm-models.js';

/** Print a computed number with a typographic minus sign and no float dust. */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');
/* `Math` in this module is the KaTeX component imported above, not the global
   object, so `Math.max` here resolves to that component and throws on first
   paint. These two helpers keep the extremes explicit and shadow-proof. */
const largest = values => values.reduce((best, value) => (value > best ? value : best));
const smallest = values => values.reduce((best, value) => (value < best ? value : best));

const weather = fixtures.weather;
const reports = fixtures.reports;
const forward = trellis(weather, reports, 'sum');
const best = trellis(weather, reports, 'max');
const main = infer(weather, reports);
const corrected = infer(weather, fixtures.correctedFinal);
const missing = infer(weather, fixtures.missingMiddle);
const deleted = infer(weather, fixtures.deletedMiddle);
const altered = infer(withRainyEmission(fixtures.changedRainyEmission), reports);
const enumerated = enumeratePaths(weather, reports);
const forecast = predictNext(weather, infer(weather, [0]).filtered[0]);
const split = countFlow(weather, fixtures.splitRecordings);
const joined = countFlow(weather, fixtures.joinedRecording);
const updated = emStep(weather, fixtures.splitRecordings);
const constrained = fixtures.constrained;
const pointwise = infer(constrained, [0, 0]);
const changedPrior = infer(withStart(constrained, fixtures.changedConstrainedStart), [0, 0]);
const rare = repeatedProduct(fixtures.rareFactor, fixtures.rareCount);
const representable = repeatedProduct(fixtures.representableFactor, fixtures.representableCount);
const duration = durationModel(fixtures.durationDefault);
const practiceDuration = durationModel(fixtures.durationPractice);

const selected = configurations[selectedConfigurationIndex];
const matchedLexical = configurations[2];
const repairs = decisionChanges.repairs.length;
const breaks = decisionChanges.breaks.length;
const tieBand = tieAudit.configurations[1].tokenTotals;
const nina = developmentSentences[27];
const article = developmentSentences[3];

const headings = [
  '1. Separate the thing you see from the state you infer',
  '2. The same observations support different questions',
  '3. Forward: add the paths without listing them',
  '4. Backward: later evidence changes an earlier belief',
  '5. Viterbi: keep the best path, not the sum',
  '6. Learning: replace invisible counts with expected counts',
  '7. Make the computation reliable and reproducible',
  '8. A real sequence: infer grammatical roles in short sentences',
  '9. Choose a model that matches how the sequence behaves',
  '10. Where this idea leads',
  '11. Cost follows the allowed edges',
  '12. Practice: change the evidence, keep the question precise',
  '13. Readiness and the next lesson',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Excerpt({ excerpt }) {
  return <section className="hmm-excerpt">
    <h4>{excerpt.title}</h4>
    <p className="lesson-note">
      Lines {excerpt.lines[0]}&ndash;{excerpt.lines[1]} of <Code>{hmmExamples.experiments.file}</Code>,
      reproduced exactly.
    </p>
    <CodeBlock language="python">{excerpt.code}</CodeBlock>
    <Prose>{excerpt.guidance}</Prose>
  </section>;
}
function Practice({ title, question, hint, revealLabel = 'Show the explained solution', children }) {
  return <section className="hmm-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>{revealLabel}</summary>{children}</details>
  </section>;
}

const hiddenMarkovModelsContent = {
  title: 'Hidden Markov Models: Infer the Process Behind a Sequence',
  readTime: '~55 min first pass · ~105 min complete read + 60–90 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot hmm-lesson">
    <LessonIntro prerequisites={<>Discrete probability, conditional probability and a matrix row that sums to one. The latent-variable and EM ideas from <a href="/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml">Gaussian Mixture Models &amp; EM</a> are a useful companion but not required; everything this lesson needs about dynamic programming is built here. The preceding <a href="/learn/path/full-curriculum/automl-neural-architecture-search-nas?module=classical-ml">AutoML &amp; Neural Architecture Search</a> lesson asked how to <em>select</em> a procedure. This one examines one particular model closely.</>} sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      A machine changes between operating conditions, and its sensor never says which one. A low reading is evidence, not a label. You will multiply one complete story through the model and inspect its computed probability, watch a later report move a belief about an earlier time while another belief provably cannot move, follow a Viterbi predecessor that is <em>not</em> the largest cell in the previous column, join the two most probable states at two times into a route the model forbids, restore a recording boundary that changes the expected counts rather than merely storing them, and then read {selected.correct} of {selected.tokens} real tokens tagged from {provenance.sentences.train} real sentences &mdash; and find that {tieAudit.configurations[1].tiedSentences.length} of the {provenance.sentences.development} <strong>development</strong> sentences have two complete paths of <em>exactly</em> equal probability. The {provenance.sentences.reserved} reserved sentences are not decoded anywhere in this lesson. Every investigation updates its topic-specific results from valid control changes, with no expected-answer input.
    </LessonIntro>
    <div className="hmm-route"><Prose><strong>First pass.</strong> Read sections 1&ndash;5 and run the four small investigations there: one path&rsquo;s product, the two beliefs at one time, the constrained path, and the trellis figures between them. Then follow the single expected-count update in section 6 and the real tagging comparison in section 8. Sections 7, 9, 10 and 11 &mdash; numerical scale, duration and Gaussian emissions, where the idea leads, and cost &mdash; are deeper branches to return to. Practice 1&ndash;5 belong to the first pass; 6&ndash;10 transfer the deeper ideas.</Prose></div>

    <Prose>A machine changes between operating conditions, but its sensor readings do not announce the condition directly. A low reading is evidence, not a label. Several readings in order can tell you more than any one reading alone.</Prose>
    <Prose>A <strong>hidden Markov model</strong>, or <strong>HMM</strong>, describes that situation using two connected sequences: a hidden state that evolves, and an observation produced at each step. It lets us ask how likely the observations are, what states could explain them, and how to learn the model&rsquo;s probabilities. The key picture is a row of hidden-state nodes with observations hanging underneath. The key computation is a <strong>trellis</strong>: all possible states laid out across time, with shared partial calculations replacing an enormous list of complete paths.</Prose>
    <Prose>The earlier <a href="/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml">GMM lesson</a> is the useful connection: a mixture assigns a latent component to each observation independently; an HMM makes those assignments depend on each other across time. That one change is what the whole of this lesson follows.</Prose>

    <H2>{headings[0]}</H2>
    <Prose>For a small constructed example, imagine receiving a friend&rsquo;s daily activity report while not seeing the weather. Use two hidden states, <strong>Rainy</strong> and <strong>Sunny</strong>, and three observations, <strong>Walk</strong>, <strong>Shop</strong> and <strong>Clean</strong>. These are deliberately simplified probabilities, not measured weather data, and the story is an original teaching construction rather than an example from any particular paper.</Prose>
    <LessonTable caption="The three tables that define the model: an initial row, one transition row per state, and one emission row per state" headers={['Table', 'Row', 'Entries']} rows={[
      ['Initial state', '—', `Rainy ${weather.start[0]}, Sunny ${weather.start[1]}`],
      ['Transition, from Rainy', 'Rainy → …', `Rainy ${weather.transition[0][0]}, Sunny ${weather.transition[0][1]}`],
      ['Transition, from Sunny', 'Sunny → …', `Rainy ${weather.transition[1][0]}, Sunny ${weather.transition[1][1]}`],
      ['Emission, in Rainy', 'Rainy emits …', `Walk ${weather.emission[0][0]}, Shop ${weather.emission[0][1]}, Clean ${weather.emission[0][2]}`],
      ['Emission, in Sunny', 'Sunny emits …', `Walk ${weather.emission[1][0]}, Shop ${weather.emission[1][1]}, Clean ${weather.emission[1][2]}`],
    ]} />
    <Prose>Each row is a separate distribution and sums to one. Rainy→Sunny is a transition between states. Rainy→Clean is an emission: an observation conditional on a state. Neither arrow gives the reverse probability. In particular, <Math>{'P(\\mathrm{Clean}\\mid\\mathrm{Rainy})=0.5'}</Math> does not imply <Math>{'P(\\mathrm{Rainy}\\mid\\mathrm{Clean})=0.5'}</Math>.</Prose>
    <GraphUnrollFigure />
    <Prose>Generate a sequence by drawing the initial state, drawing its activity, moving to a new state using the current state&rsquo;s transition row, and repeating. The observer sees the activity row of the story; inference reasons about the hidden row.</Prose>
    <Prose>The <strong>first-order Markov assumption</strong> says that, given the current hidden state, the next state does not additionally depend on earlier states. A separate <strong>emission assumption</strong> says observations factor independently once the entire state sequence is fixed. Observations can still be correlated marginally: a persistent hidden state can produce a run of similar readings.</Prose>
    <Callout title="Correlated readings are not by themselves a violation">
      This distinction matters in sensor data. Consecutive readings being correlated is expected under the model, because their states are related. It is correlation that <em>remains after conditioning on the modelled states</em> that reveals a missing dependency, an inadequate state representation or an unsuitable emission family. Every later caution about assumptions in this lesson refers back to this one.
    </Callout>
    <Prose>Let <Math>{'z_t'}</Math> be the hidden state and <Math>{'o_t'}</Math> the observed symbol at time <Math>{'t'}</Math>, starting at <Math>{'t=0'}</Math>. Write the initial probabilities as <Math>{'\\pi'}</Math>, transitions as <Math>{'A'}</Math>, and emission probabilities as <Math>{'B'}</Math>. For <Math>{'N'}</Math> states and <Math>{'M'}</Math> symbols, their shapes are <Math>{'N'}</Math>, <Math>{'N\\times N'}</Math>, and <Math>{'N\\times M'}</Math>. The probability of one complete hidden path together with its observations is</Prose>
    <MathBlock>{'\\begin{gathered}P(z_{0:T-1},o_{0:T-1})\\\\[4pt]=\\pi_{z_0}B_{z_0,o_0}\\\\[4pt]\\times\\prod_{t=1}^{T-1}A_{z_{t-1},z_t}B_{z_t,o_t}.\\end{gathered}'}</MathBlock>
    <Prose>Read this as &ldquo;start, emit, transition, emit, transition, emit.&rdquo; Multiplication follows one possible story; adding over different stories accounts for uncertainty. Parameters are fixed throughout these inference calculations; learning them is a later operation. A timeline makes the count visible: four observations contain <strong>three</strong> within-sequence transitions, not four.</Prose>
    <PathProductLab />

    <H2>{headings[1]}</H2>
    <Prose>Suppose the reports are <strong>Walk → Shop → Walk → Clean</strong>. Before calculating, choose the question:</Prose>
    {/* The quantity column goes through KaTeX like every other formula on the
        page. As plain strings the subscripts that tell filtering from smoothing
        rendered as literal braces and underscores, which is the one distinction
        sections 3 and 4 are about. */}
    <LessonTable caption="Six different questions about one recording, and the information each is allowed to use" headers={['Question', 'Quantity', 'Available observations']} rows={[
      ['How well does the model explain the reports?', <Math key="q1">{'P(o_{0:T-1})'}</Math>, 'The specified sequence'],
      ['What is the current state after this report?', <Math key="q2">{'P(z_t\\mid o_{0:t})'}</Math>, 'Past and present: filtering'],
      ['What was an earlier state, using the later reports too?', <Math key="q3">{'P(z_t\\mid o_{0:T-1})'}</Math>, 'Entire sequence: smoothing'],
      ['What is the next state likely to be?', <Math key="q4">{'P(z_{t+1}\\mid o_{0:t})'}</Math>, 'Past and present: prediction'],
      ['What single whole path has greatest probability?', <Math key="q5">{'{\\arg\\max_z P(z\\mid o)}'}</Math>, 'Entire sequence: Viterbi decoding'],
      ['What probabilities should the model use?', <Math key="q6">{'\\pi, A, B'}</Math>, 'Training sequences: learning'],
    ]} />
    <Prose>Filtering and smoothing are not interchangeable in a live system. A dashboard operating on Tuesday cannot use a Thursday reading; a retrospective analyst can. A more informed posterior also does not promise a more accurate label on every individual example.</Prose>
    <Prose>For a fixed observed sequence with positive probability, maximising <Math>{'P(z\\mid o)'}</Math> is equivalent to maximising the joint <Math>{'P(z,o)'}</Math>, because all paths share the denominator <Math>{'P(o)'}</Math>. Maximising <Math>{'P(o\\mid z)'}</Math> alone would omit the path prior and can choose a different answer.</Prose>

    <H2>{headings[2]}</H2>
    <Prose>Four time steps with two possible states each give <Math>{'2^4=16'}</Math> possible paths &mdash; the sixteen the investigation above lists. A thousand steps would give <Math>{'2^{1000}'}</Math>. We need to share work. Define</Prose>
    <MathBlock>{'\\alpha_t(j)=P(o_0,\\ldots,o_t,z_t=j).'}</MathBlock>
    <Prose>This is <strong>joint probability mass</strong>, not yet a normalised posterior over states. At the first Walk report,</Prose>
    <MathBlock>{'\\begin{gathered}\\alpha_0(R)=0.6(0.1)=0.06,\\\\[6pt]\\alpha_0(S)=0.4(0.6)=0.24.\\end{gathered}'}</MathBlock>
    <Prose>The observation has probability {num(forward.columnTotals[0])}. Dividing by that total gives the filtered belief: {num(100 * main.filtered[0][0])}% Rainy, {num(100 * main.filtered[0][1])}% Sunny. At the next report, Shop, Rainy can be reached from either earlier state:</Prose>
    <MathBlock>{'\\begin{gathered}\\alpha_1(R)\\\\[4pt]=[0.06(0.7)+0.24(0.4)]\\times 0.4\\\\[4pt]=0.0552.\\end{gathered}'}</MathBlock>
    <Prose>The bracket adds the mass arriving along both arrows; the last factor accounts for the activity at the destination. Similarly <Math>{'\\alpha_1(S)=0.0486'}</Math>. The general recurrence repeats this operation:</Prose>
    <MathBlock>{'\\begin{gathered}\\alpha_0(j)=\\pi_jB_{j,o_0},\\\\[6pt]\\alpha_t(j)=B_{j,o_t}\\sum_i\\alpha_{t-1}(i)A_{ij}.\\end{gathered}'}</MathBlock>
    <ForwardTrellisFigure />
    <Prose>Summing the last two masses gives <Math>{'P(\\mathrm{Walk,Shop,Walk,Clean})='}</Math>{num(forward.final)}, the same number the sixteen enumerated paths add to. The Rainy mass rises at the last step even though the <em>total</em> prefix probability falls: contributions have moved between states, and no individual cell has to shrink monotonically.</Prose>

    <H3>Filtering and forecasting without retaining the whole past</H3>
    <Prose>Let <Math>{'f_t'}</Math> be the normalised filtered row vector. First predict the next state, <Math>{'q_{t+1}=f_tA'}</Math>. Then, when the new report arrives, multiply <Math>{'q_{t+1}'}</Math> by its emission column and normalise. After the first Walk, <Math>{'f_0=[0.2,0.8]'}</Math>, so the next-state prediction is [{num(forecast.nextState[0])}, {num(forecast.nextState[1])}].</Prose>
    <Prose>The predicted probability of the next report being Clean weights each state&rsquo;s Clean probability by how likely that state now is: {num(forecast.nextState[0])}({weather.emission[0][2]}) + {num(forecast.nextState[1])}({weather.emission[1][2]}) = {num(forecast.nextObservation[2])}. Forecasting an observation requires both the state transition and the emission distribution. Jumping straight from the current most likely state discards that uncertainty: collapsing the belief to Sunny first would give a Shop probability of {num(forecast.collapsedObservation[1])} instead of {num(forecast.nextObservation[1])}.</Prose>
    <Prose>For a fixed model, the current filtered vector summarises everything about the observation history that the next filtering update needs. Storing every earlier forward vector is unnecessary if this is the only query.</Prose>

    <H2>{headings[3]}</H2>
    <Prose>At time 1, filtering slightly favours Rainy: {num(main.filtered[1][0])}. But if we later see Walk and Clean, the smoothed Rainy probability is {num(main.smoothed[1][0])}. The later observations changed the earlier conclusion. Define a backward likelihood,</Prose>
    <MathBlock>{'\\begin{gathered}\\beta_t(i)\\\\[4pt]=P(o_{t+1},\\ldots,o_{T-1}\\mid z_t=i).\\end{gathered}'}</MathBlock>
    <Prose>At the last time there are no later reports, so <Math>{'\\beta_{T-1}(i)=1'}</Math>. Step backward using</Prose>
    <MathBlock>{'\\beta_t(i)=\\sum_j A_{ij}B_{j,o_{t+1}}\\beta_{t+1}(j).'}</MathBlock>
    <Prose>At time 2 the only future report is Clean, so <Math>{'\\beta_2(R)=0.7(0.5)+0.3(0.1)=0.38'}</Math> and <Math>{'\\beta_2(S)=0.4(0.5)+0.6(0.1)=0.26'}</Math>. Multiply the evidence from the left and the right, then normalise:</Prose>
    <MathBlock>{'\\begin{gathered}\\gamma_t(i)=P(z_t=i\\mid o_{0:T-1})\\\\[6pt]=\\frac{\\alpha_t(i)\\beta_t(i)}{P(o_{0:T-1})}.\\end{gathered}'}</MathBlock>
    <BeliefFigure />
    <Prose>The last row agrees exactly because no later observations remain. Smoothing uses the whole sequence, but it does not reveal a verified hidden truth: both columns are probabilities under this model.</Prose>
    <EvidenceLaterLab />
    <Prose>A missing report is another useful contrast. At a retained time step with no observation, summing over all possible symbols gives emission likelihood one, so the state still transitions. Removing the entire step instead changes elapsed model time and the number of transitions. Replacing Shop with a missing report gives a final Rainy probability of {num(missing.smoothed[3][0])}; deleting that step gives {num(deleted.smoothed[2][0])}. Treating missingness this way assumes the fact of missingness itself supplies no additional state evidence.</Prose>

    <H2>{headings[4]}</H2>
    <Prose>The forward algorithm combines every path into a cell. Viterbi retains the greatest joint probability among paths ending there:</Prose>
    <MathBlock>{'\\begin{gathered}\\delta_t(j)=\\max_{z_0,\\ldots,z_{t-1}}\\\\[4pt]P(z_0,\\ldots,z_t=j,o_0,\\ldots,o_t).\\end{gathered}'}</MathBlock>
    <Prose>Replace the sum with a maximum, and store the maximising predecessor for each destination:</Prose>
    <MathBlock>{'\\begin{gathered}\\delta_0(j)=\\pi_jB_{j,o_0},\\\\[6pt]\\delta_t(j)=B_{j,o_t}\\max_i[\\delta_{t-1}(i)A_{ij}].\\end{gathered}'}</MathBlock>
    <Prose>Why can other prefixes be discarded? Two prefixes ending in the same current state have the same possible future factors. Multiplying both by the same nonnegative suffix cannot make the lower-probability prefix strictly better. The current state is the boundary that makes dynamic programming valid.</Prose>
    <ViterbiTrellisFigure />
    <Prose>At the Shop step the best Rainy prefix comes from Sunny, giving {num(best.columns[1].cells[0].value)}, and the best Sunny prefix is {num(best.columns[1].cells[1].value)}. At the following Walk step, however, the best <strong>Rainy</strong> predecessor is Rainy: {num(best.columns[1].cells[0].value)}(0.7) = {num(best.columns[1].cells[0].value * weather.transition[0][0])} beats {num(best.columns[1].cells[1].value)}(0.4) = {num(best.columns[1].cells[1].value * weather.transition[1][0])}. Selecting the largest cell in the previous column without accounting for the particular transition would get this predecessor wrong.</Prose>
    <Prose>Starting from the largest final cell and following the stored predecessors backward gives <strong>{main.path.map(state => weather.stateNames[state]).join(' → ')}</strong>, with joint probability {num(main.pathJoint)} and posterior probability {num(main.pathJoint)}/{num(main.evidence)} &asymp; {num(main.pathPosterior)}. &ldquo;Most likely&rdquo; does not mean &ldquo;nearly certain&rdquo;: about {num(100 * (1 - main.pathPosterior))}% of the posterior mass belongs to other paths collectively.</Prose>

    <H3>The most probable state at each time can form an impossible path</H3>
    <Prose>Pointwise decoding chooses <Math>{'\\arg\\max_i\\gamma_t(i)'}</Math> separately at each time. It minimises expected total per-position mistakes when predictions are unconstrained. Viterbi minimises the chance of getting the entire sequence wrong under a whole-sequence 0&ndash;1 loss. They optimise different objectives.</Prose>
    <Prose>Consider a separate two-step, three-state model whose only observation symbol has probability one in every state, so the reports carry no information at all. Its initial probabilities are [{constrained.start.join(', ')}], and the only permitted transitions are A→B, A→C, B→A and C→A. At the first time, A has the greatest marginal probability, {num(pointwise.smoothed[0][0])}. At the second, A again leads with {num(pointwise.smoothed[1][0])}. Pointwise modes therefore return <strong>A→A</strong>, a transition that has probability exactly zero.</Prose>
    <PointwiseFigure />
    <Prose>Viterbi returns <strong>{pointwise.path.map(state => constrained.stateNames[state]).join('→')}</strong>, with probability {num(pointwise.pathJoint)}. Its expected number of correct positions is {num(pointwise.pathExpectedCorrect)}, compared with {num(pointwise.modesExpectedCorrect)} for the unconstrained but impossible pointwise output. Better expected position count and validity as a joint path are separate properties.</Prose>
    <LegalPathLab />
    <Prose>Changing the starting probabilities to [{fixtures.changedConstrainedStart.join(', ')}] while keeping transitions fixed makes both decoders choose {changedPrior.path.map(state => constrained.stateNames[state]).join('→')}, at probability {num(changedPrior.pathJoint)}. Total evidence stays exactly {num(changedPrior.evidence)} because the observations remain uninformative: a changed prior moved the path probabilities without moving the evidence at all. If validity and expected position accuracy are both requirements, a constrained minimum-risk decoder can maximise <Math>{'\\sum_t\\gamma_t(z_t)'}</Math> over legal paths by another dynamic program. That is still a different objective from Viterbi&rsquo;s product of model factors.</Prose>

    <H2>{headings[5]}</H2>
    <Prose>If training states are known, estimate transitions by counting adjacent state pairs and emissions by counting symbols within each state. Sentence or recording boundaries matter: the final state of one recording is not followed by the first state of an unrelated recording.</Prose>
    <Prose>If states are not labelled, we cannot count one true path. <strong>Baum&ndash;Welch</strong> uses expectation&ndash;maximisation: infer distributions over paths using current parameters, compute expected counts, then fit new probabilities from those counts. The state posterior <Math>{'\\gamma_t(i)'}</Math> contributes a fractional occupancy count. A transition requires a <em>joint pair posterior</em>, not the product of two marginal posteriors:</Prose>
    <MathBlock>{'\\begin{gathered}\\xi_t(i,j)=P(z_t=i,z_{t+1}=j\\mid o)\\\\[6pt]=\\frac{\\alpha_t(i)A_{ij}B_{j,o_{t+1}}\\beta_{t+1}(j)}{P(o)}.\\end{gathered}'}</MathBlock>
    <Prose>For each edge time, <Math>{'\\sum_{ij}\\xi_t(i,j)=1'}</Math>; its row sums equal <Math>{'\\gamma_t(i)'}</Math> and its column sums equal <Math>{'\\gamma_{t+1}(j)'}</Math>. Those relationships are useful numerical checks and they are what makes a count-flow diagram interpretable. For <Math>{'K'}</Math> independent sequences with lengths <Math>{'T_k'}</Math>, the updates are</Prose>
    <MathBlock>{'\\begin{gathered}\\pi_i^{\\mathrm{new}}=\\frac1K\\sum_k\\gamma^{(k)}_0(i),\\\\[8pt]A_{ij}^{\\mathrm{new}}=\\frac{\\sum_k\\sum_{t}\\xi^{(k)}_t(i,j)}{\\sum_k\\sum_{t}\\gamma^{(k)}_t(i)},\\end{gathered}'}</MathBlock>
    <MathBlock>{'B_{i,v}^{\\mathrm{new}}=\\frac{\\sum_{k,t}\\gamma^{(k)}_t(i)\\mathbf1[o_t^{(k)}=v]}{\\sum_{k,t}\\gamma^{(k)}_t(i)}.'}</MathBlock>
    <Prose>The transition sums run to <Math>{'T_k-2'}</Math>, excluding each sequence&rsquo;s last position because it has no outgoing within-sequence transition; the emission sums run to <Math>{'T_k-1'}</Math> and include it. Initial-state counts are divided by the number of sequences, not the total number of positions. When missing reports carry no state information and are marginalised out, update emissions using only observed positions in both numerator and denominator; transitions still include the retained time steps.</Prose>

    <H3>One complete update, with a boundary you can move</H3>
    <Prose>Treat <strong>Walk→Shop</strong> and <strong>Walk→Clean</strong> as two independent recordings. Under the original parameters the expected transition counts are [[{num(split.edges[0][0].mass)}, {num(split.edges[0][1].mass)}], [{num(split.edges[1][0].mass)}, {num(split.edges[1][1].mass)}]]. They sum to {num(split.totals.edgeMass)}, because each recording contains one transition. Row-normalising gives the updated transition rows [{updated.model.transition[0].map(value => num(value)).join(', ')}] and [{updated.model.transition[1].map(value => num(value)).join(', ')}].</Prose>
    <CountFlowFigure />
    <Prose>Expected initial-state counts are [{split.start.map(entry => num(entry.mass)).join(', ')}], so <Math>{'\\pi^{\\mathrm{new}}'}</Math> &asymp; [{updated.model.start.map(value => num(value)).join(', ')}]. The joint training log-likelihood of the two recordings increases from {num(updated.logLikelihoodBefore)} to {num(updated.logLikelihoodAfter)} after this one update. Removing the boundary creates a <em>different</em> data model, with one start and {joined.totals.transitions} transitions, and changes the expected counts &mdash; the Rainy→Rainy count alone moves from {num(split.edges[0][0].mass)} to {num(joined.edges[0][0].mass)}. It is not merely a storage optimisation.</Prose>
    <BoundaryCountLab />

    <H3>What EM does and does not guarantee</H3>
    <Prose>An exact E-step and matching exact M-step make the observed training log-likelihood nondecreasing in exact arithmetic. The increase can be zero. This property does not promise a global maximum, semantic recovery, better development performance, or a useful state count. Approximate updates, added penalties, changed objectives and finite-precision computations each require their own analysis.</Prose>
    <Prose>The complete program generates {emTrack.recordings} independent length-{emTrack.length} sequences from the constructed model, using data seed {emTrack.dataSeed}. Three declared starting seeds each receive forty EM updates.</Prose>
    <EmHistoryFigure />
    <Prose>The generating parameters score {num(emTrack.generatingLogLikelihood)} on this particular finite sample. All three fits exceed that score by adapting to sample variation, while the symmetric start stays below it; higher training likelihood is not proof of recovering the generating parameters. With uniform state initialisation, both emission rows remain identical to each other for ever: after the first update they equal the empirical symbol frequencies, and subsequent iterations stay at approximately {num(emTrack.uniformStart.logLikelihood[1])}. Symmetry is not broken by simply iterating longer.</Prose>
    <Prose>State labels can also be permuted without changing observation likelihood: permute <Math>{'\\pi'}</Math>, both axes of <Math>{'A'}</Math>, and the rows of <Math>{'B'}</Math> consistently. That explains index ambiguity. It does not explain every poor fit, and it does not justify naming an unsupervised state &ldquo;Rainy&rdquo; just because its index is zero. If a state has zero expected occupancy, its unconstrained emission row is unidentified; the program retains that row rather than dividing by zero.</Prose>

    <H2>{headings[6]}</H2>
    <Prose>The complete offline program implements forward&ndash;backward, filtering, Viterbi with backtracking, expected counts, independent-sequence Baum&ndash;Welch, scaled forward inference and the real tagging experiment below. Python and NumPy are its only computational dependencies. Save <a href={hmmExamples.experiments.download} download>{hmmExamples.experiments.file}</a> beside <a href={hmmExamples.experiments.dataFile} download>the supplied sequence data</a>, then run it:</Prose>
    <RunnableExample example={hmmExamples.experiments}>
      <Prose>
        It writes <Code>{hmmExamples.experiments.writes}</Code>, holding the actual trellis, posterior rows, count
        updates, training histories and real predictions. That file is the reference this page is checked against:
        every number above and below was recomputed independently and matched against it. No fitted state or timing
        curve is invented for display. The whole program is {hmmExamples.experiments.lineCount} lines, so the four
        excerpts below carry its mechanism; each is an exact slice of the file you can download, not a retyping.
      </Prose>
    </RunnableExample>
    {hmmExamples.experiments.excerpts.map(excerpt => <Excerpt key={excerpt.key} excerpt={excerpt} />)}
    <Prose>Start by changing one observation and inspecting one computed row. Understanding that row is more useful than memorising the whole program.</Prose>

    <H3>Tiny probability or genuinely impossible event?</H3>
    <Prose>Repeated multiplication can underflow even when the mathematical probability is positive. In float64, <Math>{'0.3^{100}'}</Math> &asymp; {num(representable.ordinary)} is still perfectly representable &mdash; underflow is not a property of small exponents in general. But <Math>{'0.01^{400}=10^{-800}'}</Math> rounds to zero, while its log probability, {num(rare.logValue)}, remains manageable.</Prose>
    <Prose>In log-space, products become sums, and sums of probabilities use log-sum-exp:</Prose>
    <MathBlock>{'\\begin{gathered}\\log\\sum_i e^{z_i}=m+\\log\\sum_i e^{z_i-m},\\\\[6pt]m=\\max_i z_i.\\end{gathered}'}</MathBlock>
    <Prose>Handle an all-negative-infinity row explicitly: it represents a zero sum, and subtracting negative infinity from itself would create a NaN. A structural zero transition stays negative infinity in log-space. Adding an arbitrary epsilon would change which paths the model permits.</Prose>
    <Prose>The alternative is normalised scaling. Write <Math>{'b_j(o_t)=B_{j,o_t}'}</Math>. If <Math>{'f_{t-1}'}</Math> is the filtered distribution, compute</Prose>
    <MathBlock>{'\\begin{gathered}u_t(j)=b_j(o_t)\\sum_i f_{t-1}(i)a_{ij},\\\\[6pt]c_t=\\sum_j u_t(j),\\qquad f_t(j)=\\frac{u_t(j)}{c_t}.\\end{gathered}'}</MathBlock>
    <Prose>At the first observation use <Math>{'\\pi_jb_j(o_0)'}</Math>. Here <Math>{'c_t=P(o_t\\mid o_{0:t-1})'}</Math> for <Math>{'t\\geq1'}</Math>, and <Math>{'c_0=P(o_0)'}</Math>, so <Math>{'\\log P(o)=\\sum_t\\log c_t'}</Math>. Our four activities give factors {main.factors.map(factor => num(factor)).join(', ')}, whose product is {num(main.evidence)}. Some texts define the scale factor as the reciprocal; their final log formula then carries a minus sign. Compare definitions before comparing code.</Prose>
    <NumericScaleFigure />
    <Prose>If every state assigns zero probability to an observed symbol, the observation sequence really is impossible under the model. Neither scaling nor log-space should convert it into a valid posterior. Check whether a hard constraint was intended, or whether a training vocabulary needs a deliberately learned unknown-symbol category. That is a modelling decision.</Prose>

    <H3>Use the library without changing the question</H3>
    <Prose>The optional program below targets the <a href="https://hmmlearn.readthedocs.io/en/0.3.3/tutorial.html">hmmlearn 0.3.3 API</a> and shows a fixed categorical model, independent-sequence fitting and a Gaussian-emission model. <strong>The content phase could not execute it</strong>, because the dependency was unavailable; this implementation installed it in an isolated environment and ran it, so the output below is real. The environment is deliberately separate from the one the program above ran in: resolving hmmlearn moves NumPy, and other lessons&rsquo; recorded outputs depend on the exact versions there. It resolved to hmmlearn {hmmExamples.hmmlearn.environment.hmmlearn}, NumPy {hmmExamples.hmmlearn.environment.numpy}, SciPy {hmmExamples.hmmlearn.environment.scipy} and scikit-learn {hmmExamples.hmmlearn.environment['scikit-learn']} on Python {hmmExamples.hmmlearn.environment.python}.</Prose>
    <CodeBlock language="bash">{hmmExamples.hmmlearn.setup}</CodeBlock>
    <RunnableExample example={hmmExamples.hmmlearn}>
      <Prose>
        It also printed one warning to standard error: <Code>{hmmExamples.hmmlearn.warning}</Code> Both halves of
        that sentence check out. Section 9 counts the free parameters of a two-state, three-symbol categorical model
        as {parameterCount(2, 3)}, which is the number the library reports; and the two two-step recordings really do
        supply only four observations. The fit it warns about reaches a log score of −1.386294, which is exactly
        2&thinsp;log&thinsp;0.5: it has explained each recording with probability one half by making its states
        deterministic. A warning naming a degenerate solution is worth more than a convergence flag.
      </Prose>
    </RunnableExample>
    <Prose>A categorical observation is an integer symbol, stored in a <Math>{'T\\times1'}</Math> array. A multinomial observation is a vector of category counts for one observation; these are different sample spaces. A Gaussian observation is a real-valued feature vector, stored in a <Math>{'T\\times D'}</Math> array. If several independent sequences share one storage array, pass their lengths: a boundary is not an observed transition.</Prose>
    <LessonTable caption="What each returned quantity means, and what the executed output shows" headers={['Call', 'What it returns', 'In the run above']} rows={[
      ['score', 'log probability or log density of the observations, summed over hidden paths', '−4.673518, which is log 0.00933936'],
      ['decode with algorithm="viterbi"', 'highest-scoring joint path and its joint log score', '−5.773004 with path Sunny Sunny Sunny Rainy'],
      ['predict_proba', 'smoothed state marginals for the complete supplied sequence', 'the same four rows this page computes'],
      ['decode with algorithm="map"', 'a marginal mode at each time, which can violate structural constraints', '2.940022 — see below'],
    ]} />
    <Callout title="One version-specific trap, now executed rather than asserted">
      In hmmlearn 0.3.3 the MAP decoder returns the <em>sum of the selected marginal probabilities</em> as its score, despite a general return description calling the value a log probability. The run above returns 2.940022. No log probability of a probability can be positive, and the value is exactly the sum of the four smoothed maxima. It is the expected number of correct states for that pointwise decision. Inspect the <a href="https://github.com/hmmlearn/hmmlearn/blob/0.3.3/src/hmmlearn/base.py">actual decoder</a> when interpreting such a result. Likewise, a convergence monitor can stop because the iteration budget is exhausted: read the objective history and the stopping condition rather than a boolean flag.
    </Callout>

    <H2>{headings[7]}</H2>
    <Prose>Words make the value and the limitations of context visible. &ldquo;Read the entire article&rdquo; has a sequence of grammatical roles, even though an isolated word can be ambiguous.</Prose>
    <DataProvenanceNote />
    <Prose>These are the first eligible short sentences from each official split, not a representative random sample. For a readable first model, map NOUN and PROPN to <strong>Noun</strong>, VERB and AUX to <strong>Verb</strong>, and all other original UPOS labels to <strong>Other</strong>. The original labels are preserved in the served data. Other is a deliberately broad category, so the task is easier and less linguistically complete than full part-of-speech tagging.</Prose>
    <Prose>Here the training states are labelled. We estimate initial, transition and emission probabilities from actual counts rather than running latent-state EM; the labels are hidden only when predicting a new sentence. This is supervised HMM fitting, not a claim that unsupervised states recover grammatical categories.</Prose>

    <H3>Fit without learning from the answer sheet</H3>
    <Prose>Lowercase the training words and retain those occurring at least twice; all remaining words map to a single unknown symbol. That makes {vocabulary.length} emission symbols. Development words do not alter that vocabulary, and {unknownDevelopmentTokens} of the {selected.tokens} development tokens map to unknown. For smoothing strength <Math>{'\\alpha'}</Math>, use</Prose>
    <MathBlock>{'\\begin{gathered}\\hat a_{ij}=\\frac{C_{ij}+\\alpha}{\\sum_kC_{ik}+3\\alpha},\\\\[8pt]\\hat b_i(w)=\\frac{C_{iw}+\\alpha}{\\sum_vC_{iv}+146\\alpha}.\\end{gathered}'}</MathBlock>
    <Prose>Smooth the three initial-state counts similarly. This experiment has no declared impossible tag transitions, so adding pseudocounts to every transition is intentional. In a topology with genuinely forbidden edges, smooth only the allowed events and renormalise there. Compare two decision rules from the same fitted counts: a <strong>lexical baseline</strong>, which at each token chooses the state maximising its training frequency times the state&rsquo;s emission probability and ignores neighbouring tags; and the <strong>HMM</strong>, which uses the learned start and transition probabilities to select the Viterbi path for the sentence. Try only the two declared smoothing strengths, {configurations.map(entry => entry.smoothing).filter((value, index, all) => all.indexOf(value) === index).join(' and ')}, yielding two probability fits and four decoder configurations.</Prose>
    <LessonTable caption="Four decoder configurations on the forty development sentences" headers={['Smoothing', 'Decoder', 'Correct tokens / 341', 'Entire sentences correct / 40']} rows={configurations.map(entry => [
      String(entry.smoothing), entry.label, String(entry.correct), String(entry.sentencesCorrect),
    ])} />
    <Prose>The majority-Other baseline gets {majority.correct} tokens correct. Development token accuracy selects the HMM with smoothing {selected.smoothing} among these candidates; the declared tie rule, recorded in the program, prefers the lexical decoder and then the smaller strength, and the two lexical rows do in fact tie, so the rule is not decorative. These are development findings from a small educational extract, not final held-out estimates or a claim about modern taggers.</Prose>
    <RealTaggingLab />
    <Prose>Compared with its matching lexical baseline, the selected HMM repairs {repairs} token decisions and breaks {breaks}, for a net gain of {repairs - breaks}. In <strong>&ldquo;{nina.tokens.join(' ')}&rdquo;</strong> the lexical predictions are Noun&ndash;Noun&ndash;Other while the HMM predicts Other&ndash;Noun&ndash;Other, matching the coarse reference labels. In <strong>&ldquo;{article.tokens.join(' ')}&rdquo;</strong>, context changes &ldquo;article&rdquo; from a correct Noun into Other. The model&rsquo;s preference for a common transition pattern can override useful lexical evidence. Inspect both before celebrating the aggregate gain.</Prose>

    <H3>How firm is that gain? An exact tie says how firm</H3>
    <Prose>Those token counts are the output of one decoder implementation, and one thing worth knowing about them is not visible from the table. Checked in exact rational arithmetic rather than in floating point, <strong>{tieAudit.configurations[1].tiedSentences.length} of the forty development sentences have two complete paths of exactly equal probability</strong>. Each of them contains adjacent tokens that both map to the unknown symbol, so exchanging those two states permutes the same multiset of transition and emission factors and leaves the product identical. The model cannot prefer either, and which one a decoder reports is settled by its arithmetic.</Prose>
    <Prose>Every path this page displays is verified to be an exact maximiser. But the reported {selected.correct} is one member of a band: taking the lowest-indexed state at every tie gives {tieBand.first}, taking the highest gives {tieBand.last}. The comparison the section is about survives all of it, and for a sharper reason than a band overlap: the same three sentences are tied at <em>both</em> smoothing strengths, with the same two candidate paths and the same correct-token counts, so any one consistent tie rule contributes the same amount to both totals and smoothing {selected.smoothing.toFixed(1)} beats {configurations[1].smoothing.toFixed(1)} by exactly {selected.correct - configurations[1].correct} whichever rule is chosen. The HMM also beats its matching lexical baseline under every rule, since even the band&rsquo;s lowest member exceeds {matchedLexical.correct}. What does not survive is the exact margin of {repairs - breaks} tokens. This is what the lesson&rsquo;s own point about unknown-word collapse looks like when it reaches the arithmetic, and it is a good reason to report a comparison rather than a single number.</Prose>
    <Checkpoint prompt={`A colleague reports "our HMM tagger reaches ${selected.correct}/${selected.tokens} on development, a ${repairs - breaks}-token gain over the lexical baseline" and proposes that as the headline result. What would you add before that sentence leaves the room?`}>
      <Prose>Three things, in order of how much they matter. First, the {tieBand.last - tieBand.first}-token tie band: {tieAudit.configurations[1].tiedSentences.length} sentences have exactly tied optimal paths, so the same fitted model reports anywhere from {tieBand.first} to {tieBand.last} depending on a tie rule nobody declared. The direction of the gain is robust; its size is not. Second, this is a development number used to <em>select</em> among four candidates, so it is not an estimate of performance on unseen text; the {provenance.sentences.reserved} reserved sentences exist precisely so that a fresh protocol remains possible, and they stay unscored here. Third, the task is a deliberately coarse three-way mapping on short sentences chosen first from each split, and {unknownDevelopmentTokens} of the {selected.tokens} tokens carry no lexical information at all. None of that makes the comparison worthless; it makes the honest sentence a comparison rather than a number.</Prose>
    </Checkpoint>
    <Prose>For words sharing the unknown category, the model cannot use their distinct spellings as evidence &mdash; which is why replacing one unknown spelling with another, in the investigation above, changes nothing at all. A future suffix or character feature could help, but it must be defined from training data and evaluated as a new procedure. The next conditional-model lessons explain a more flexible way to use such input features. A high posterior is confidence under the chosen model: coarse labels, misspecified independence, unknown-word collapse and limited training data can still make a confident prediction wrong.</Prose>

    <H2>{headings[8]}</H2>
    <H3>Duration is a hidden assumption you can see</H3>
    <Prose>A self-transition lets a state persist, but it imposes a particular duration distribution. If its self-transition probability is <Math>{'a'}</Math>, then for <Math>{'0\\leq a<1'}</Math>,</Prose>
    <MathBlock>{'\\begin{gathered}P(D=d)=a^{d-1}(1-a),\\\\[6pt]d=1,2,\\ldots,\\qquad E[D]=\\frac1{1-a}.\\end{gathered}'}</MathBlock>
    <Prose>A state must remain for <Math>{'d-1'}</Math> transitions and then leave. With <Math>{'a=0.7'}</Math> the mean dwell time is {num(duration.mean)} steps; with <Math>{'a=0.95'}</Math> it is {num(durationModel(0.95).mean)}. The chance of leaving next is still <Math>{'1-a'}</Math>, however long the state has already lasted. That constant hazard applies when the conditioning history has positive probability. At a = 0 the state always leaves on its first transition, so asking about its departure after ten survived steps conditions on an impossible history; the answer is undefined, not 1. This boundary matters when applying the geometric distribution&rsquo;s memorylessness.</Prose>
    <DurationLab />
    <Prose>This can suit a simple regime model, but not a process that becomes progressively more likely to end after a characteristic duration. An explicit-duration or hidden semi-Markov model adds a duration model. An absorbing state with <Math>{'a=1'}</Math> never leaves; do not draw it as a finite-mean geometric curve. The time step also has meaning: a transition matrix fitted per minute is not automatically a per-second matrix. If <Math>{'k'}</Math> unobserved equal time steps pass under the same homogeneous model, propagation uses <Math>{'A^k'}</Math>. Keeping a missing observation as an unobserved time step is different from deleting that time step.</Prose>

    <H3>Real-valued measurements: Gaussian emissions</H3>
    <Prose>For a sensor vector <Math>{'x_t\\in\\mathbb R^D'}</Math>, replace the categorical probability with a density, <Math>{'b_i(x_t)=\\mathcal N(x_t;\\mu_i,\\Sigma_i)'}</Math>. The forward and backward structure is unchanged. A density can exceed one and changes with measurement units; it is not a probability assigned to one exact real-valued point. With responsibilities <Math>{'\\gamma_t(i)'}</Math>, the maximum-likelihood M-step is</Prose>
    <MathBlock>{'\\mu_i=\\frac{\\sum_t\\gamma_t(i)x_t}{\\sum_t\\gamma_t(i)},'}</MathBlock>
    <MathBlock>{'\\Sigma_i=\\frac{\\sum_t\\gamma_t(i)(x_t-\\mu_i)(x_t-\\mu_i)^\\top}{\\sum_t\\gamma_t(i)}.'}</MathBlock>
    <Prose>Sum across independent sequences too. A diagonal covariance models within-time features without off-diagonal covariance; a full covariance permits within-time correlations. Neither change removes the standard HMM&rsquo;s across-time conditional emission factorisation. An autoregressive emission model can instead condition a current observation on previous observations as well as the state. Adding overlapping windows or delta features to an ordinary emission vector may be useful engineering, but it does not make those windows conditionally independent.</Prose>
    <Prose>Very small state occupancy and collapsed Gaussian covariances can make likelihood fitting unstable or degenerate &mdash; which is exactly what the library warning above was about, in its categorical form. Appropriate covariance constraints, priors, training evidence and development checks matter. Two states with equal means can still have different variances or transition roles; equality of means alone does not prove redundancy.</Prose>

    <H3>How many states?</H3>
    <Prose>For a fully free categorical HMM with <Math>{'N'}</Math> states and <Math>{'M'}</Math> symbols, the parameter count is <Math>{'p=(N-1)+N(N-1)+N(M-1)'}</Math>. Each probability row sums to one. Our two-state, three-symbol model has {parameterCount(2, 3)} free parameters. Structural zeros, tied rows, fixed parameters and other emission families change that count.</Prose>
    <Prose>Adding states can increase expressive capacity, yet a particular locally optimised fit can have worse likelihood than a smaller model &mdash; the three seeds in figure 7 are that phenomenon at one fixed state count. Compare several starts and the query you care about, using independent sequences or a justified temporal split. AIC and BIC can be useful selection heuristics, but latent models can be nonregular and correlated observations complicate a casual choice of sample size. State the likelihood, the free-parameter count and the sample-size convention instead of treating a formula as an automatic answer. For a sensor deployed online, evaluate filtering or forecasting at the actual decision time: a beautiful smoothed reconstruction obtained after the whole recording arrives answers a different operational question.</Prose>

    <H2>{headings[9]}</H2>
    <TopologyFigure />
    <Prose><strong>Biological sequences.</strong> A profile HMM represents a sequence family with positions that can match, insert or skip. Match states emit aligned residues; insert states emit additional residues; delete states advance the profile without emitting one. A silent delete transition therefore differs from a missing measurement at a real time step. The <a href="https://hmmer.org/">HMMER project</a> uses profile HMMs for biological sequence analysis; its <a href="https://eddylab.org/software/hmmer/2.3.1/Userguide.pdf">historical model guide</a> explains this topology. The interesting connection is alignment as inference through an allowed graph, rather than forcing every sequence to have identical length.</Prose>
    <Prose><strong>Speech as a sequence of submodels.</strong> Historical speech recognisers combined state models for sound or word segments. Compare observation likelihoods under candidate models, account for class priors, or compose models and search over legal concatenations. The practical questions include feature modelling, duration and boundaries, not simply &ldquo;run Viterbi.&rdquo; Rabiner&rsquo;s tutorial develops that application in its historical setting; its hardware timings are not current benchmarks.</Prose>
    <Prose><strong>Several hidden causes at once.</strong> A household power signal can reflect several devices whose states evolve separately. A factorial HMM uses multiple hidden chains with shared observations, as the right-hand drawing above sketches. Independent prior transitions do not make posterior inference independent once those causes explain the same measurement. A joint representation grows rapidly, motivating structured approximations. This is the extension developed by <a href="https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf">Ghahramani and Jordan</a>.</Prose>
    <Prose><strong>Continuous hidden states.</strong> Position and velocity are naturally real-valued. A linear Gaussian state-space model replaces discrete state probabilities with Gaussian beliefs and leads to Kalman filtering and smoothing. The common pattern is prediction through a state transition followed by correction from evidence; different assumptions change the representation and computation.</Prose>
    <Prose><strong>Conditional sequence prediction.</strong> HMMs model a joint distribution over observations and states. A CRF models the label sequence conditional on an observed input and can use rich features of that input, including future context when the application permits it. A neural encoder can provide scores to a CRF; these ideas are compatible. Parameter count, available labels, inference constraints and the actual task determine whether one model is useful. There is no universal sequence-length or data-count threshold that selects a winner.</Prose>

    <H2>{headings[10]}</H2>
    <Prose>For <Math>{'T'}</Math> time steps and <Math>{'N'}</Math> states, a dense forward or Viterbi pass evaluates <Math>{'O(TN^2)'}</Math> transition contributions, plus emission evaluation &mdash; our four-step, two-state trellis evaluates {forward.transitionContributions} of them. Forward combines them with sums; Viterbi uses maxima. A topology with <Math>{'E'}</Math> allowed edges can reduce the transition work to <Math>{'O(TE)'}</Math>, provided the representation and implementation actually exploit sparsity.</Prose>
    <Prose>Filtering needs only the current and previous <Math>{'N'}</Math>-state rows. Retaining a full trellis or Viterbi backpointers costs <Math>{'O(TN)'}</Math> memory. A full array of pairwise posteriors costs <Math>{'O(TN^2)'}</Math>; EM can instead accumulate sufficient statistics without retaining every pair row. The teaching program deliberately retains the small pair arrays for inspection, so it is not a memory-minimal implementation.</Prose>
    <Prose>For categorical emissions, add expected counts directly into the observed-symbol column. That avoids looping over all <Math>{'M'}</Math> symbols at every time step; dense emission tables still cost <Math>{'O(NM)'}</Math>. Gaussian diagonal densities cost roughly <Math>{'O(TND)'}</Math>; dense full-covariance densities involve matrix factorisations and roughly <Math>{'O(TND^2)'}</Math> quadratic-form work after factorisation.</Prose>
    <Prose>A beam limits the active candidates, but work also depends on their outgoing edges and duplicate successors, so &ldquo;beam width <Math>{'K'}</Math> means <Math>{'O(TK)'}</Math>&rdquo; is incomplete for a dense state graph. Low-rank matrix multiplication can accelerate sums in some models; replacing the sum with a maximum does not preserve the same algebra automatically. Specialised parallel methods and sparse structures are options to measure, not reasons to invent hardware-independent speed ratios.</Prose>

    <H2>{headings[11]}</H2>
    <Prose>Try each before opening its hint or solution. Questions 1&ndash;5 use the first-pass route; 6&ndash;10 transfer the deeper ideas. A calculator or a short script is welcome; the target is a defensible explanation, not mental arithmetic speed.</Prose>

    <Practice title="1. Forecast another activity"
      question="After observing Walk, what is the probability of Shop next? Why not simply choose the currently most likely state and use its emission row?"
      hint="Propagate the complete filtered distribution [0.2, 0.8] through A before applying the Shop probabilities.">
      <Prose>The next-state probabilities are [{num(forecast.nextState[0])}, {num(forecast.nextState[1])}]. Shop has probability {num(forecast.nextState[0])}&times;{weather.emission[0][1]} + {num(forecast.nextState[1])}&times;{weather.emission[1][1]} = {num(forecast.nextObservation[1])}. Collapsing the current belief to Sunny would produce next-state probabilities [{weather.transition[1][0]}, {weather.transition[1][1]}] and a Shop probability of {num(forecast.collapsedObservation[1])}. It discards uncertainty before prediction. Reproduce both numbers with the one-step forecast calculation in section 3, first propagating the full filtered row and then replacing it with a point mass on Sunny.</Prose>
    </Practice>

    <Practice title="2. Repair a misleading backpointer"
      question={<>At the third observation, Walk, which previous state gives the best path ending in Rainy? Use the preceding Viterbi scores {num(best.columns[1].cells[0].value)} and {num(best.columns[1].cells[1].value)}.</>}
      hint="Compare scores after multiplying by the appropriate incoming transition, before multiplying by the shared destination emission.">
      <Prose>Rainy contributes {num(best.columns[1].cells[0].value)}&times;{weather.transition[0][0]} = {num(best.columns[1].cells[0].value * weather.transition[0][0])}; Sunny contributes {num(best.columns[1].cells[1].value)}&times;{weather.transition[1][0]} = {num(best.columns[1].cells[1].value * weather.transition[1][0])}. The Rainy predecessor wins even though its previous score is smaller. Multiplying by the Rainy Walk emission gives {num(best.columns[2].cells[0].value)}. The final best path need not pass through this cell &mdash; and here it does not.</Prose>
    </Practice>

    <Practice title="3. A later observation is corrected"
      question="The last Clean observation becomes Walk. Should the probability of Rainy immediately after processing the second observation change? Should the probability of Rainy at that same time, after seeing the whole corrected recording, change?"
      hint="Name which observations each query conditions on.">
      <Prose>The filtered value stays {num(main.filtered[1][0])}, bit for bit, because its prefix is unchanged. The smoothed value changes from {num(main.smoothed[1][0])} to {num(corrected.smoothed[1][0])}. The best complete path becomes {corrected.path.map(state => weather.stateNames[state]).join(' → ')}. These are different queries, not inconsistent answers. The &ldquo;correct the final report to Walk&rdquo; setup in the investigation above loads exactly this comparison.</Prose>
    </Practice>

    <Practice title="4. A popular state sequence is impossible"
      question="In the three-state example, the marginal modes are A then A, but A cannot transition to A. What are that path's joint probability and the Viterbi path?"
      hint="Marginal modes optimise expected position-wise correctness without enforcing path validity.">
      <Prose>A→A has probability exactly {num(pointwise.modesJoint)}. {pointwise.path.map(state => constrained.stateNames[state]).join('→')} has joint probability {num(pointwise.pathJoint)} and is the Viterbi path. Pointwise modes have expected correct-state count {num(pointwise.modesExpectedCorrect)}, versus {num(pointwise.pathExpectedCorrect)} for {pointwise.path.map(state => constrained.stateNames[state]).join('→')}, but they optimise a different loss over a larger decision set. To require legal paths while optimising position-wise correctness, solve a constrained max-sum problem using marginal rewards.</Prose>
    </Practice>

    <Practice title="5. Count independent recordings"
      question={<>Three recordings have lengths {fixtures.practiceLengths.join(', ')}. How many expected starts, emissions and within-recording transitions should the E-step counts sum to? What changes if you silently concatenate them?</>}
      hint="A length-one recording still has a start and an emission, but no transition.">
      <Prose>There are {fixtures.practiceLengths.length} starts, {fixtures.practiceLengths.reduce((total, value) => total + value, 0)} emissions and ({fixtures.practiceLengths.map(value => `${value}−1`).join(') + (')}) = {fixtures.practiceLengths.reduce((total, value) => total + value - 1, 0)} transitions. Concatenation changes this to 1 start, {fixtures.practiceLengths.reduce((total, value) => total + value, 0)} emissions and {fixtures.practiceLengths.reduce((total, value) => total + value, 0) - 1} transitions. The two extra transitions are invented boundaries, and the posterior fractions of the genuine transitions change too. Build all three of these in the boundary investigation and read the totals off the applied state.</Prose>
    </Practice>

    <Practice title="6. A zero that logarithms cannot rescue — deeper"
      question="Both states emit only symbol 0, but the recording contains symbol 1. A colleague adds a tiny epsilon to every entry. Is this numerical stabilisation alone?"
      hint="Distinguish a positive number too small to represent from a genuinely zero probability.">
      <Prose>No. The original event is impossible: its evidence is exactly zero and its posterior has no value at all, which is different from the {num(rare.logValue)} that {fixtures.rareCount} genuinely rare reports produce. Epsilon introduces previously forbidden observations and requires row normalisation, creating a different model. If the zeros reflect insufficient data rather than hard constraints, an explicit smoothing model may be sensible; it must be stated and fitted accordingly, exactly as the {vocabulary.length}-symbol vocabulary in section 8 does with its deliberate unknown category.</Prose>
    </Practice>

    <Practice title="7. Design a five-step mean duration — deeper"
      question="Choose the self-transition probability for mean duration five. Find the probability of duration exactly three. After ten steps already spent in the state, what is the chance of leaving next?"
      hint="Use the geometric duration and its constant exit probability.">
      <Prose>a = {fixtures.durationPractice}. P(D = 3) = {fixtures.durationPractice}² &times; {num(practiceDuration.exitProbability)} = {num(practiceDuration.probabilities[2])}. The next-step exit probability remains {num(practiceDuration.exitProbability)}, conditional on still being in the state, however long it has lasted. An age-dependent departure process needs a richer duration model. Set a = {fixtures.durationPractice} in the dwell-time investigation and read all three numbers off the applied state.</Prose>
    </Practice>

    <Practice title="8. The EM curve decreases — deeper"
      question="An implementation shows a substantial log-likelihood decrease after an alleged EM iteration. List checks that could explain it before concluding EM's theorem is false."
      hint="The theorem concerns a matching exact objective and a complete update.">
      <Prose>Check whether the plotted quantity is log-likelihood or negative log-likelihood; whether scores and parameters refer to the same iteration; whether the sequences and their boundaries changed between the two evaluations; whether the probability rows still normalise; whether an approximate E- or M-step, clipping, prior or penalty changed the objective; and whether zeros or floating arithmetic broke the calculation. Tiny roundoff differences are distinct from substantive decreases &mdash; the program in section 7 refuses a decrease beyond 10⁻⁸ and would have raised rather than plotted one.</Prose>
    </Practice>

    <Practice title="9. Interpret the real improvement — deeper"
      question={<>The HMM repairs {repairs} lexical predictions but breaks {breaks}. What is its net improvement? Does this establish that contextual models always help, or that development accuracy is a final test estimate?</>}
      hint="The unit being counted is a token, the development set helped select the procedure, and three sentences have exactly tied optimal paths.">
      <Prose>{repairs - breaks} additional tokens are correct, from {matchedLexical.correct} to {selected.correct} out of {selected.tokens}. Three qualifications belong with that number. It is a small, selected development comparison on a coarse task. Its exact size depends on a tie rule: the same fitted model reports between {tieBand.first} and {tieBand.last} depending on which member of three exact ties is reported, though the HMM beats the lexical baseline under every one of them. And the {provenance.sentences.reserved} reserved sentences remain unscored, precisely so that a frozen procedure could be evaluated on an appropriate independent set before generalisation is claimed. Do not add features based on that final set&rsquo;s mistakes and still call it untouched.</Prose>
    </Practice>

    <Practice title="10. Equal means, different states — deeper"
      question="Two Gaussian states have the same mean but different covariances and self-transition probabilities. Must they be merged?"
      hint="A state's role includes its observation distribution and its dynamics.">
      <Prose>No. They can distinguish low-variance from high-variance regimes, or short from persistent episodes: with a = 0.7 the mean dwell time is {num(duration.mean)} steps and with a = 0.95 it is {num(durationModel(0.95).mean)}, which is a substantive difference in behaviour at identical means. Whether both are useful requires model and task evaluation. Equal means alone establish neither identical distributions nor redundant sequence behaviour.</Prose>
    </Practice>

    <H2>{headings[12]}</H2>
    <Prose>You are ready to continue when you can name the conditioning information in each query, compute one sum and one maximum trellis update, explain a backward correction, preserve sequence boundaries during learning, and recognise when the model&rsquo;s assumptions are doing more work than its data.</Prose>
    <LessonTable caption="Readiness check" headers={['You should be able to', 'Where it was taught']} rows={[
      ['Multiply one complete story through the model and say how many transitions it has', 'Section 1, figure 1, the path investigation'],
      ['Name what each of six queries is allowed to condition on', 'Section 2'],
      ['Carry out one forward step and say why a cell value is joint mass, not a belief', 'Section 3, figure 2'],
      ['Forecast the next observation without collapsing the current belief', 'Section 3, practice 1'],
      ['Say which belief a later report can move and which it provably cannot', 'Section 4, figure 3, the evidence investigation, practice 3'],
      ['Choose a Viterbi predecessor after multiplying by its transition, not before', 'Section 5, figure 4, practice 2'],
      ['Recognise pointwise modes that form a path of probability zero', 'Section 5, figure 5, the path investigation, practice 4'],
      ['Count starts, emissions and transitions across recording boundaries', 'Section 6, figure 6, the boundary investigation, practice 5'],
      ['Read an EM objective history without treating a flat curve as a global optimum', 'Section 6, figure 7, practice 8'],
      ['Separate underflow, log-space and a genuinely impossible event', 'Section 7, figure 8, practice 6'],
      ['Read a library return value against what it actually computes', 'Section 7, the executed hmmlearn run'],
      ['Compare two decision rules on real tokens and report a band rather than a number', 'Section 8, the tagging investigation, practice 9'],
      ['Turn a self-transition into a duration assumption and back', 'Section 9, the dwell-time investigation, practice 7'],
      ['Count free parameters and say what would change that count', 'Section 9, figure 9'],
    ]} />
    <Prose>The next lesson, <a href="/learn/path/full-curriculum/bayesian-networks-causal-graphical-models?module=classical-ml">Bayesian Networks &amp; Causal Graphical Models</a>, makes the graph and its conditional independences explicit: this lesson used one particular chain structure, and that lesson asks what any graph implies. After it, <a href="/learn/path/full-curriculum/conditional-random-fields-crf?module=classical-ml">Conditional Random Fields</a> revisits sequence labelling from the conditional perspective, and returns to exactly the tagging task in section 8 with a model that can use features of the input the emission table here could not.</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; these offer a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://web.stanford.edu/~jurafsky/slp3/A.pdf">Jurafsky and Martin &mdash; Speech and Language Processing, Appendix A</a>. Markov chains, HMMs, forward inference, Viterbi and EM in one complete accessible progression, and the closest canonical treatment of this lesson&rsquo;s material. The draft fetched during authoring is dated 19 August 2026; notation and chapter numbering may change as the book develops.</li>
      <li><a href="https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf">Rabiner &mdash; A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, 1989</a>. Read section III for the three problems, section IV for duration and continuous emissions, and section V for scaling and multiple sequences. Its speech systems supply historical context, not current performance guidance.</li>
      <li><a href="https://www.cs.jhu.edu/~jason/papers/#eisner-2002-tnlp">Jason Eisner&rsquo;s HMM teaching resources</a>, accompanying <a href="https://aclanthology.org/W02-0102/">An Interactive Spreadsheet for Teaching the Forward-Backward Algorithm</a>. A small model you manipulate by hand, which is a good second route to section 4 in particular. The paper metadata and resource descriptions were checked during authoring; the video was not watched and the spreadsheet was not executed.</li>
    </ul></>}>
      <li><a href="https://hmmlearn.readthedocs.io/en/0.3.3/tutorial.html">hmmlearn 0.3.3 tutorial</a> and its <a href="https://github.com/hmmlearn/hmmlearn/blob/0.3.3/src/hmmlearn/base.py">0.3.3 decoder source</a> &mdash; observation shapes, sequence lengths, initialisation and decoding. Pair the prose with the versioned source when interpreting returned scores; the MAP decoder&rsquo;s return value is the concrete reason why.</li>
      <li><a href={provenance.page}>Universal Dependencies English EWT</a> and its <a href={provenance.readme}>{provenance.release} source and rights notice</a> &mdash; the actual annotations. This page serves <a href={provenance.file} download>its own unchanged copy</a> of the extract, {provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <Code>{provenance.sha256}</Code>, beside its <a href={provenance.attribution}>attribution</a>, which records the extraction, the original sentence identifiers, the licence and the coarse-label transformation.</li>
      <li><a href="https://hmmer.org/">HMMER</a> for profile models of biological sequences, and <a href="https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf">Ghahramani and Jordan &mdash; Factorial Hidden Markov Models</a> for several interacting hidden explanations. Both extend the state structure rather than adding iterations to the same model.</li>
    </Sources>
    <Prose>The Rainy/Sunny model, its four activity reports, the three-state constrained graph, the two recordings, the duration family and every practice value are explicitly <strong>constructed calculations</strong>, not measurements, and the weather story is an original teaching construction rather than an example taken from any particular paper. The EM histories are measured calculations on {emTrack.recordings} sequences sampled from that constructed model with seed {emTrack.dataSeed}. The tagging results are calculations on the identified real extract under one declared protocol, with the {provenance.sentences.reserved} reserved sentences never predicted or scored. None of them is a benchmark or a claim about any future dataset.</Prose>
  </div>,
};

export default hiddenMarkovModelsContent;
