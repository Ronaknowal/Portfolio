import { Callout, H2, H3, Prose, Code } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import {
  EvidenceLab, InterventionLab, MeasurementLab, PathLab,
} from '../../components/lesson-labs/BayesNetLabs.jsx';
import {
  CounterfactualFigure, EliminationGeometryFigure, EquivalenceFigure, FactorWorkbenchFigure,
  FrontdoorFigure, MeasuredContrastFigure, WorldAssemblyFigure, measuredContrast,
} from '../../components/lesson-labs/BayesNetFigures.jsx';
import { bayesnetExamples } from '../bayesnet-examples.js';
import { conditionalInformation, protocol, provenance, scores, trainingModels } from '../bayesnet-data.js';
import {
  alarmNetwork, backdoorCriterion, counterfactualOfUnit, counterfactualPair, dSeparation, eliminationRun,
  fixtures, freeParameters, frontdoorConditions, frontdoorModel, jointProbability, markovBlanket,
  queryFamilies, queryPosterior, serviceModel, withCallerRow,
} from '../bayesnet-models.js';

/** Print a computed number with a typographic minus sign and no float dust. */
const num = value => String(Number(value.toFixed(9))).replace('-', '−');
/* `Math` in this module is the KaTeX component imported above, not the global
   object: writing `Math.round` here resolves to that component and silently
   yields `undefined` rather than a number. Anything that would reach for the
   global `Math` gets an explicit, shadow-proof helper instead. */
const whole = value => Number(value.toFixed(0));
/** A percentage in prose, at the precision the sentence around it claims.
 *  `num` rounds to nine DECIMALS, which on a percentage is nine to eleven
 *  significant figures -- "about 28.417183536%" after the word "about". */
const percent = (value, digits = 2) => String(Number((100 * value).toFixed(digits)));

const assembled = jointProbability(alarmNetwork, fixtures.assembledWorld);
const posteriors = fixtures.publishedEvidence.map(entry => ({
  ...entry, ...queryPosterior(alarmNetwork, entry.evidence),
}));
const bothCalls = posteriors.find(entry => entry.label === 'John and Mary call');
const alarmKnown = posteriors.find(entry => entry.label === 'Alarm definitely sounds');
const alarmAndJohn = posteriors.find(entry => entry.label === 'Alarm sounds; John calls');
const withEarthquake = posteriors.find(entry => entry.label === 'Alarm sounds; earthquake occurs');
const johnOnly = posteriors.find(entry => entry.label === 'John calls');

const elimination = eliminationRun(alarmNetwork, fixtures.eliminationEvidence, fixtures.eliminationOrder);
const earthquakeStep = elimination.steps.find(step => step.variable === 'E');
const alarmParameters = freeParameters(fixtures.alarmParameters);
const practiceParameters = freeParameters(fixtures.practiceParameters);
const naiveParameters = freeParameters(fixtures.naiveBayesParameters);
const treeParameters = freeParameters(fixtures.treeAugmentedParameters);

const blanket = markovBlanket(fixtures.alarmEdges, 'B');
const colliderClosed = dSeparation(fixtures.alarmEdges, 'B', 'E', []);
const colliderOpen = dSeparation(fixtures.alarmEdges, 'B', 'E', ['A']);
const colliderByDescendant = dSeparation(fixtures.alarmEdges, 'B', 'E', ['J']);

const educationSets = fixtures.educationCandidateSets.map(set => ({
  set, ...backdoorCriterion(fixtures.educationEdges, 'T', 'Y', set),
}));
const educationWithDirect = fixtures.educationCandidateSets.slice(1).map(set => ({
  set, ...backdoorCriterion(fixtures.educationWithDirectEffect, 'T', 'Y', set),
}));
const soleBackdoorPath = educationSets[0].backdoorPaths[0];

const service = serviceModel(fixtures.service);
const randomised = serviceModel(fixtures.serviceRandomised);
const changedResponse = serviceModel(fixtures.serviceChangedResponse);
const noOverlap = serviceModel(fixtures.serviceNoOverlap);
const servicedHighLoad = service.lanes[1].observedShares[1];

const frontdoor = frontdoorModel();
const frontdoorOk = frontdoorConditions(fixtures.frontdoorEdges, 'X', 'M', 'Y');
const frontdoorLatentMediator = frontdoorConditions(fixtures.frontdoorWithLatentMediator, 'X', 'M', 'Y');
const frontdoorDirect = frontdoorConditions(fixtures.frontdoorWithDirectEffect, 'X', 'M', 'Y');

const counterfactuals = counterfactualPair();
const observedUnit = counterfactualOfUnit(
  fixtures.counterfactualObservation.treatment,
  fixtures.counterfactualObservation.outcome,
  fixtures.counterfactualObservation.changedTreatment);
const practiceUnit = counterfactualOfUnit(
  fixtures.practiceCounterfactualObservation.treatment,
  fixtures.practiceCounterfactualObservation.outcome,
  fixtures.practiceCounterfactualObservation.changedTreatment);

const mapQuery = queryFamilies(fixtures.queryMasses);
const practiceMap = queryFamilies(fixtures.practiceQueryMasses);

const uninformativeMary = queryPosterior(
  withCallerRow(withCallerRow(alarmNetwork, 'M', 0, fixtures.uninformativeCallRow), 'M', 1, fixtures.uninformativeCallRow),
  { J: 1, M: 1 });
const uninformativeMaryAlone = queryPosterior(
  withCallerRow(withCallerRow(alarmNetwork, 'M', 0, fixtures.uninformativeCallRow), 'M', 1, fixtures.uninformativeCallRow),
  { M: 1 });

const contrast = measuredContrast();
const treeParentNames = trainingModels.treeAugmented.parents
  .map((parent, index) => (parent === null ? null : `${protocol.featureLabels[index]} ← ${protocol.featureLabels[parent]}`))
  .filter(Boolean);
const strongestPair = conditionalInformation.reduce((best, entry) =>
  (entry.informationNats > best.informationNats ? entry : best));
const retainedFraction = bothCalls.evidenceProbability;

const headings = [
  '1. One world is a product of local choices',
  '2. Infer a cause by adding the worlds that fit',
  '3. Reuse the arithmetic with variable elimination',
  '4. Read independence off the paths',
  '5. Learn the tables, then question a real specimen',
  '6. Changing a mechanism is not selecting evidence',
  '7. Deeper: identify an effect, then ask whose counterfactual it is',
  '8. Deeper: larger networks, structure search and the alternatives',
  '9. Practice: explain the changed case before calculating',
  '10. What you can now do, and where it goes next',
];
const headingId = heading => heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function Program({ example, children }) {
  return <section>
    <Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example}>{children}</RunnableExample>
  </section>;
}

function Practice({ title, question, hint, revealLabel = 'Show the explained solution', children }) {
  return <section className="bn-practice">
    <H3>{title}</H3>
    <Prose>{question}</Prose>
    {hint && <details><summary>Get a hint</summary><Prose>{hint}</Prose></details>}
    <details><summary>{revealLabel}</summary>{children}</details>
  </section>;
}

const bayesianNetworksContent = {
  title: 'Bayesian Networks & Causal Graphical Models',
  readTime: '~60 min first pass · ~115 min complete read + 60–100 min code and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot bn-lesson">
    <LessonIntro
      prerequisites={<>Multiplication, weighted averages and conditional probability. <Math>{'P(B=1\\mid J=1)'}</Math> means
        “among outcomes in which John called, what fraction have a burglary?”. Uppercase letters name variables and
        lowercase letters their selected values. The preceding <a href="/learn/path/full-curriculum/hidden-markov-models-hmm?module=classical-ml">Hidden
        Markov Models</a> lesson supplies a chain of repeated local factors; this one generalises that to an arbitrary
        acyclic graph. Directed acyclic graphs, conditional probability tables, d-separation and the do operator are all
        introduced here.</>}
      sections={headings.map(heading => [headingId(heading), heading.replace(/^\d+\. /, '')])}>
      Two people call to report that your house alarm is sounding. Is there a burglary? An earthquake can also set the
      alarm off, and a caller sometimes reports a sound incorrectly. You will build the five-variable model, add up the
      {' '}{bothCalls.compatibleWorlds} worlds that fit both calls to get a burglary posterior
      of {num(bothCalls.posterior)}, watch that number collapse to {num(withEarthquake.posterior)} when an earthquake
      turns up, find a clue that changes it by exactly nothing, decide by hand which observation sets a graph
      guarantees independence for, and then take four chemical measurements on {provenance.specimens} real wine
      specimens and ask which one is worth buying. Every investigation asks for a recorded prediction before it
      calculates anything, and retires that prediction the moment an input changes.
    </LessonIntro>

    <div className="bn-route"><Prose><strong>First pass.</strong> Read sections 1–6 and do practice 1–6. That route
      gets you a working network, a posterior you calculated yourself, a path you can read, a real fitted classifier,
      and the difference between observing something and doing it. Run the two programs on the way. Sections 7 and 8
      are deeper branches: 7 develops identification, do-calculus and counterfactuals, and 8 covers structure search,
      junction trees, sampling and the neighbouring model families. Come back to them once the small sums feel
      natural.</Prose></div>

    <Prose>A <strong>Bayesian network</strong> describes a joint probability distribution using small local
      probability models connected by arrows. Its first job is bookkeeping: state which variables each local model
      depends on, then combine those models consistently. Once the joint distribution is defined you can ask many
      questions of the same model, including questions in which most of the variables are unobserved.</Prose>
    <Prose>There is a second, stronger use for a graph. If its arrows describe how a system is <em>generated</em>,
      under suitable causal assumptions, it can help predict what happens when a mechanism is changed. Observing that
      an alarm is sounding and deliberately switching it on are different events. The first may tell you about a
      burglary; the second need not.</Prose>

    <Callout title="Which numbers here are measurements">
      All alarm, maintenance and mediation probabilities in this lesson are <strong>constructed teaching
      models</strong>. They are not measured crime statistics, engineering reliability data or clinical results.
      The Wine experiment in section 5 is the one place real observations appear, and it is attributed there.
      This caution is stated once; the rest of the lesson refers back to it rather than repeating it.
    </Callout>

    {/* ============================================================ §1 */}
    <H2>{headings[0]}</H2>
    <Prose>Name five binary variables. <Math>{'B'}</Math> is a burglary, <Math>{'E'}</Math> an
      earthquake, <Math>{'A'}</Math> the alarm sounding, and <Math>{'J'}</Math> and <Math>{'M'}</Math> John
      and Mary calling. Draw <Math>{'B\\to A\\leftarrow E'}</Math>, with <Math>{'A\\to J'}</Math> and <Math>{'A\\to M'}</Math>.</Prose>
    <Prose>The drawing contains no directed cycle: following arrows can never bring you back to where you started.
      Such a graph is a <strong>directed acyclic graph</strong>, or DAG. A <strong>parent</strong> points directly
      into a node; an <strong>ancestor</strong> reaches it through one or more arrows, and a{' '}
      <strong>descendant</strong> is reachable by following arrows away from it.</Prose>

    <LessonTable caption="The five variables, what state 1 means, and which variables each one listens to"
      headers={['Variable', 'State 1 means', 'Parents']}
      rows={alarmNetwork.nodes.map(node => [
        node, alarmNetwork.labels[node], alarmNetwork.parents[node].join(', ') || 'none',
      ])} />

    <Prose>The root probabilities are <Math>{'P(B=1)=0.001'}</Math> and <Math>{'P(E=1)=0.002'}</Math>.
      The alarm table gives a complete distribution for each of the four parent settings. This is
      a <strong>conditional probability table</strong>, or CPT.</Prose>
    <LessonTable caption="The alarm's conditional probability table. Each row is a distribution and sums to one."
      headers={['B', 'E', 'P(A = 1 | B, E)', 'P(A = 0 | B, E)']}
      rows={Object.entries(alarmNetwork.chance.A).map(([key, chance]) => [
        key.split(',')[0], key.split(',')[1], num(chance), num(1 - chance),
      ])} />
    <Prose>John calls with probability {num(alarmNetwork.chance.J[1])} if the alarm sounds
      and {num(alarmNetwork.chance.J[0])} otherwise. Mary's corresponding probabilities
      are {num(alarmNetwork.chance.M[1])} and {num(alarmNetwork.chance.M[0])}.</Prose>

    <Prose>The model's factorisation is the product of those five local models:</Prose>
    <MathBlock>{'\\begin{gathered}P(b,e,a,j,m)=\\\\P(b)\\,P(e)\\,P(a\\mid b,e)\\\\\\times P(j\\mid a)\\,P(m\\mid a).\\end{gathered}'}</MathBlock>
    <Prose>Take the world <Math>{'B=1,E=0,A=1,J=1,M=1'}</Math>. Each node contributes the entry its own parents
      select:</Prose>
    <MathBlock>{'\\begin{gathered}(.001)(.998)(.94)(.90)(.70)\\\\=' + num(assembled) + '.\\end{gathered}'}</MathBlock>
    <Prose>Multiply, rather than adding five confidence scores. Change a state in the figure below and watch which
      row of each table is read.</Prose>

    <WorldAssemblyFigure />

    <Prose>That is one world, not the probability of a burglary. Other worlds also produce two calls, and section 2
      adds them up.</Prose>

    <H3>Why this product is already a distribution</H3>
    <Prose>Imagine generating the variables in parent-before-child order: choose <Math>{'B'}</Math> and <Math>{'E'}</Math>,
      then <Math>{'A'}</Math>, then <Math>{'J'}</Math> and <Math>{'M'}</Math>. Each choice distributes the probability
      mass arriving at a node among that node's states. Equivalently, sum the joint over <Math>{'J'}</Math> and{' '}
      <Math>{'M'}</Math>: their conditional distributions each contribute one. Sum over <Math>{'A'}</Math>, then the
      roots, and you are left with one. Cyclic systems can have probability models too, but it is the DAG construction
      that makes normalisation automatic here.</Prose>

    <Prose>The graph also makes specific independence assumptions. John and Mary may be associated before you know
      whether the alarm sounded, because both respond to it. Once <Math>{'A'}</Math> is known, this model treats their
      remaining reporting randomness as independent. If they telephone one another, that assumption fails, and a
      beautifully drawn graph cannot repair a missing dependency.</Prose>
    <Prose>More generally the <strong>local Markov property</strong> says that a node is independent of its
      nondescendants, other than its parents, once its parents are given. That is exactly what lets each local CPT
      leave earlier variables out of the full chain rule. Section 4 turns the property into a test for harder
      queries.</Prose>

    <H3>What the assumptions buy</H3>
    <Prose>For five unconstrained binary variables a joint table
      has {alarmParameters.jointEntries} entries and {alarmParameters.jointFree} free numbers: one entry is fixed by
      the sum-to-one constraint. This network stores {alarmParameters.stored} CPT entries but
      only <strong>{alarmParameters.free} free numbers</strong> — one for each root, four for the alarm, two for each
      caller. The reduction comes from assumptions, not from compression without consequences. In general a
      node <Math>{'i'}</Math> with <Math>{'r_i'}</Math> states and <Math>{'q_i'}</Math> parent configurations
      contributes <Math>{'q_i(r_i-1)'}</Math> free parameters.</Prose>
    <Prose>One naming point: “Bayesian” here does not require Bayesian parameter estimation. The graph can be fitted
      by maximum likelihood, as section 5 does. Putting a prior distribution over its parameters is an additional
      modelling choice, which section 5 also makes, separately and on purpose.</Prose>

    {/* ============================================================ §2 */}
    <H2>{headings[1]}</H2>
    <Prose>With both calls observed, the quantity we want is</Prose>
    <MathBlock>{'\\begin{gathered}P(B=1\\mid J=1,M=1)\\\\[4pt]=\\frac{\\sum_{e,a}P(1,e,a,1,1)}{\\sum_{b,e,a}P(b,e,a,1,1)}.\\end{gathered}'}</MathBlock>
    <Prose>There are two operations to recognise. <strong>Marginalisation</strong> sums over unobserved
      alternatives. <strong>Conditioning</strong> keeps the outcomes compatible with the evidence and renormalises
      what is left. Neither one selects a single most likely hidden explanation.</Prose>
    <Prose>The numerator is the burglary half of that sum, {num(bothCalls.mass[1])}, and the
      denominator is the whole of it, {num(bothCalls.evidenceProbability)}, so the posterior
      is <strong>{num(bothCalls.posterior)}</strong>, about {percent(bothCalls.posterior)}%. Two calls are
      substantial evidence against a {num(100 * posteriors[0].posterior)}% prior, and the model still considers the
      no-burglary outcomes collectively more likely.</Prose>

    <LessonTable caption="Burglary posterior under seven evidence sets, from enumerating all 32 worlds"
      headers={['Evidence', 'Compatible worlds', 'Burglary posterior']}
      rows={posteriors.map(entry => [entry.label, String(entry.compatibleWorlds), num(entry.posterior)])} />

    <Prose>The last three rows carry two separate ideas. Discovering an earthquake supplies an alternative
      explanation for the alarm, and in this particular model it drops the burglary posterior
      from {num(alarmKnown.posterior)} to {num(withEarthquake.posterior)}. That is <strong>explaining away</strong>.
      In the other direction, once the alarm state is already known, John's report adds nothing at all about a
      burglary: the posterior stays at {num(alarmAndJohn.posterior)} because the intervening alarm state screens off
      that information. “More evidence always increases the posterior” is not a rule, and neither is “more evidence
      always changes it”.</Prose>

    <EvidenceLab />

    <H3>A short executable calculation</H3>
    <Prose>Every loop below corresponds to one compatible world. It uses only the standard library.</Prose>
    <Program example={bayesnetExamples.enumerate}>
      <Prose>The printed number is the same {num(bothCalls.posterior)} the table gives, to the last digit the machine
        can carry. This is a fine way to answer a question about a tiny network. Enumerating every world costs{' '}
        <Math>{'2^n'}</Math> operations, so the same program on thirty binary variables would have a billion worlds to
        walk, and section 3 fixes that.</Prose>
    </Program>

    {/* ============================================================ §3 */}
    <H2>{headings[2]}</H2>
    <Prose>A <strong>factor</strong> is a table indexed by some variables. It need not itself sum to one. You can
      multiply factors that share variables, then sum a variable out of their product; the resulting smaller table
      summarises exactly what the eliminated variable contributed.</Prose>
    <Prose>With <Math>{'J=M=1'}</Math>, first combine the earthquake prior and the alarm CPT into</Prose>
    <MathBlock>{'g(b,a)=\\sum_e P(e)\\,P(a\\mid b,e).'}</MathBlock>
    <LessonTable caption="The factor left after summing the earthquake out. It is a table over B and A, and it does not sum to one."
      headers={['B', 'A', 'g(B, A)']}
      rows={earthquakeStep.cells.map(cell => [
        String(cell.states.B), String(cell.states.A), num(cell.value),
      ])} />
    <Prose>Then calculate <Math>{'h(b)=P(b)\\sum_a g(b,a)P(J=1\\mid a)P(M=1\\mid a)'}</Math>.
      For <Math>{'b=1'}</Math>:</Prose>
    <MathBlock>{'\\begin{gathered}h(1)=.001\\bigl[(' + num(earthquakeStep.cells[2].value) + ')(.05)(.01)\\\\+(' + num(earthquakeStep.cells[3].value) + ')(.90)(.70)\\bigr]\\\\=' + num(elimination.mass[1]) + '.\\end{gathered}'}</MathBlock>
    <Prose>And <Math>{'h(0)=' + num(elimination.mass[0])}</Math>. Dividing <Math>{'h(1)'}</Math> by the sum
      gives {num(elimination.posterior)} — the same answer enumeration gave, with the intermediate results reused
      instead of recomputed. <strong>The two routes agree because elimination only reorders the same sums and
      products.</strong> That correspondence is worth holding onto: it is why a faster algorithm here is not an
      approximation.</Prose>

    <FactorWorkbenchFigure />

    <H3>The four steps, and the one mistake</H3>
    <Prose>For a general elimination step: collect every current factor containing the variable; multiply those
      factors, aligning shared states; sum over the variable; put the resulting factor back alongside the untouched
      ones. Do not sum a variable out of one factor while leaving another occurrence elsewhere — that breaks the
      dependency the two factors share. Evidence can simplify factors before elimination, and ancestors irrelevant to
      the query can often be pruned entirely.</Prose>

    <Prose>The order changes the cost, even though correct arithmetic gives the same marginal. Eliminating a highly
      connected node can create a large factor coupling all of its neighbours.</Prose>

    <EliminationGeometryFigure />

    <Prose>For an order with <strong>induced width</strong> <Math>{'w'}</Math>, the largest ordinary dense
      intermediate over binary variables can have <Math>{'2^{w+1}'}</Math> entries; with equal <Math>{'r'}</Math>-state
      variables replace 2 by <Math>{'r'}</Math>. Width is computed on the undirected interaction graph the factors
      induce, not by counting arrows. For a Bayesian network, connect every pair of parents of the same child and then
      drop the arrow directions: that is its <strong>moral graph</strong>, in which each CPT's variables form a
      connected clique. The best achievable induced width over all orders is the graph's <strong>treewidth</strong>.
      A DAG with one child of many parents can have a huge family factor while looking shallow.</Prose>
    <Prose>At width 30, a 31-variable binary float64 factor alone takes <Math>{'2^{31}\\times8=16'}</Math> GiB, before
      any extra copies or other factors. That is an exact storage calculation, not a measured library threshold and
      not a claim that every network of that width is infeasible.</Prose>

    {/* ============================================================ §4 */}
    <H2>{headings[3]}</H2>
    <Prose>The arrows do two jobs: they locate the CPTs, and they encode conditional-independence
      guarantees. <strong>D-separation</strong> is the graphical test for the second job. A path may follow edges in
      either direction; it is not restricted to directed ancestry.</Prose>
    <Prose>Inspect the middle node of each three-node segment:</Prose>
    <LessonTable caption="The three segment patterns, and what observing the middle node does to each"
      headers={['Pattern', 'Middle node unobserved', 'Middle node observed']}
      rows={[
        ['Chain X → Z → Y, or reversed', 'can carry dependence', 'blocks this path'],
        ['Fork X ← Z → Y', 'can carry dependence', 'blocks this path'],
        ['Collider X → Z ← Y', 'blocks, unless a descendant of Z is observed', 'opens this part of the path'],
      ]} />
    <Prose>A path is <strong>active</strong> when every noncollider along it is unobserved and every collider is
      either observed or has an observed descendant. Two variables are d-separated by an observation set
      when <strong>all</strong> connecting paths are blocked. One blocked route does not cancel another active one.</Prose>
    <Prose>In the alarm graph the single path between <Math>{'B'}</Math> and <Math>{'E'}</Math> is{' '}
      <Code>{colliderClosed.paths[0].path.join('–')}</Code>. With nothing observed it is blocked: {colliderClosed.paths[0].reason}.
      Observe <Math>{'A'}</Math> and it opens. Observe only <Math>{'J'}</Math> — a descendant of the alarm, not the
      alarm itself — and it opens too: {colliderByDescendant.paths[0].reason}. A checker that tests only whether the
      collider itself is observed misses that case entirely.</Prose>

    <PathLab />

    <Callout title="An active path is not a correlation">
      For DAG-factorising distributions, d-separation <em>guarantees</em> conditional independence wherever the
      conditioning event has support. The converse does not hold: an active path says only that the graph does not
      guarantee independence, and particular numerical parameters can still cancel the dependence.{' '}
      <strong>Faithfulness</strong> is the additional assumption that no such extra independence occurs. Setting every
      alarm row to the same number makes the alarm independent of its stated parents in that distribution while the
      graph still shows an active conditioned collider path; the independence came from the numbers, not from a new
      d-separation. <a href="https://ermongroup.github.io/cs228-notes/representation/directed/">Stanford's
      directed-model notes</a> develop the distinction.
    </Callout>

    <H3>The Markov blanket</H3>
    <Prose>For a node <Math>{'Y'}</Math>, its parents, its children and its children's other parents form
      a <strong>Markov blanket</strong>: once those variables are known, the graph guarantees that no other node adds
      information about <Math>{'Y'}</Math>. For <Math>{'B'}</Math> the blanket
      is <Math>{'\\{' + blanket.blanket.join(',') + '\\}'}</Math>. The earthquake belongs there precisely because
      learning the common effect can associate its possible causes — the same explaining-away mechanism, now read off
      the picture.</Prose>
    <Prose>This connects to <a href="/learn/path/full-curriculum/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml">Feature
      Selection &amp; Importance</a>: a sufficient blanket can make every other input redundant for a specified
      distribution. Marginal mutual information and pairwise correlations generally cannot identify that set on their
      own, and degenerate distributions may admit smaller blankets.</Prose>

    {/* ============================================================ §5 */}
    <H2>{headings[4]}</H2>
    <Prose>If a node's parent setting occurs in 10 complete training rows and the child is 1 in three of them, its
      maximum-likelihood estimate for that entry is <Math>{'3/10'}</Math>. The same count is repeated for each parent
      setting. The log joint likelihood separates into sums of these local count terms, so fixed-graph, fully observed,
      discrete parameter learning reduces to estimating multinomial tables. A parent setting that never occurs has no
      empirical distribution to estimate at all.</Prose>
    <Prose>One response is a Dirichlet prior. For child-state counts <Math>{'N_k'}</Math> and positive prior
      pseudo-counts <Math>{'\\alpha_k'}</Math>, the posterior predictive probability for the next case is</Prose>
    <MathBlock>{'\\begin{gathered}P(X_{\\rm new}=k\\mid\\text{setting},D)\\\\[4pt]=\\frac{N_k+\\alpha_k}{N+\\sum_j\\alpha_j}.\\end{gathered}'}</MathBlock>
    <Prose>For counts <Math>{'[9,1]'}</Math> with <Math>{'\\alpha=[1,1]'}</Math>, the probability of state 1
      is <Math>{'2/12'}</Math> rather than <Math>{'1/10'}</Math>, and an empty binary row
      gets <Math>{'[1/2,1/2]'}</Math>. These are posterior means and predictive probabilities, not generally MAP
      estimates: where an interior mode exists, the MAP estimate subtracts one from each posterior Dirichlet parameter
      before normalising. A prior states an assumption and can remove fragile zeros; it does not make the resulting
      probabilities accurate. <a href="https://pgmpy.org/examples/Parameter_Learning_Discrete_BN.html">pgmpy's
      parameter-learning guide</a> separates complete-data estimation, prior-based estimation and latent-variable EM.</Prose>

    <H3>Does modelling extra dependencies improve cultivar probabilities?</H3>
    <Prose>The <a href={provenance.doi}>UCI Wine dataset</a> holds {provenance.specimens} specimens from{' '}
      {provenance.cultivars} cultivars grown in one region of Italy, with {provenance.measurementsInFile} chemical
      measurements each. This page serves <a href={provenance.file} download>its own unchanged
      copy</a>, {provenance.bytes.toLocaleString('en-US')} bytes, SHA-256 <Code>{provenance.sha256}</Code>, beside
      its <a href={provenance.attribution}>attribution</a>; it is licensed{' '}
      <a href={provenance.licenseUrl}>{provenance.license}</a>, created by {provenance.creator}. This is cultivar
      recognition, not wine quality prediction.</Prose>
    <Prose>We use four measurements: {protocol.featureLabels.join(', ')}. Each is deliberately turned into a binary
      indicator — is it strictly above that feature's <strong>training median</strong>? That makes the learned CPTs
      small enough to inspect, at the cost of losing detail. Median thresholds are a data-processing choice, not a
      chemical law.</Prose>

    <Prose>Two recipes are compared. <strong>Naive Bayes</strong> makes the cultivar <Math>{'C'}</Math> the parent of
      each measurement. <strong>Tree-augmented naive Bayes</strong> also allows each measurement at most one
      measurement parent; it picks a maximum-weight spanning tree using conditional mutual
      information <Math>{'I(X_i;X_j\\mid C)'}</Math>, orients that tree outward from the first feature and
      adds <Math>{'C\\to X_i'}</Math>. The extra arrows model residual associations between measurements within a
      cultivar group; they are not discovered chemical causes.{' '}
      <a href="https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf">Friedman, Geiger and
      Goldszmidt, section 4</a> establishes the tree construction.</Prose>
    <MathBlock>{'\\begin{gathered}\\hat I(X_i;X_j\\mid C)=\\\\\\sum_{c,a,b}\\hat P(c,a,b)\\log\\frac{\\hat P(a,b\\mid c)}{\\hat P(a\\mid c)\\hat P(b\\mid c)}.\\end{gathered}'}</MathBlock>
    <Prose>Empty joint cells contribute zero. The score asks whether knowing one measurement helps predict another
      after the cultivar is accounted for. On the training specimens the strongest pair
      is {protocol.featureLabels[strongestPair.left]} with {protocol.featureLabels[strongestPair.right]},
      at {num(strongestPair.informationNats)} nats. With {provenance.cultivars} classes and four binary measurements,
      naive Bayes has {naiveParameters.free} free parameters and the tree-augmented model {treeParameters.free}; more
      expressive tables also divide the observations into smaller groups.</Prose>

    <Callout title="The protocol, fixed before any score was read">
      Split into {protocol.trainSize} training, {protocol.validationSize} validation
      and {protocol.testSize} test specimens, stratified by cultivar with seeds {protocol.testSeed} and{' '}
      {protocol.validationSeed}. Fit medians, tree structure and CPTs on the training specimens only. Both models use
      one pseudo-count per state and the same four measurements. Choose whichever has the lower validation mean log
      loss, breaking an exact tie in favour of naive Bayes. Refit the chosen recipe on
      the {protocol.trainSize + protocol.validationSize} development specimens and assess it once on the
      reserved {protocol.testSize}.
    </Callout>

    <Prose>Log loss is the average <Math>{'-\\log p'}</Math> assigned to the observed class; lower is better. A
      confident wrong prediction costs more than a tentative one. Accuracy counts only which class has the largest
      probability.</Prose>
    <LessonTable caption="Training-only models on the validation specimens. The declared criterion is the log-loss column."
      headers={['Model', 'Validation correct', 'Validation log loss']}
      rows={[
        ['Prior only', `${scores.priorValidation.correct}/${scores.priorValidation.rows}`, num(scores.priorValidation.logLoss)],
        ['Naive Bayes', `${scores.naiveBayesValidation.correct}/${scores.naiveBayesValidation.rows}`, num(scores.naiveBayesValidation.logLoss)],
        ['Tree-augmented', `${scores.treeAugmentedValidation.correct}/${scores.treeAugmentedValidation.rows}`, num(scores.treeAugmentedValidation.logLoss)],
      ]} />
    <Prose>Naive Bayes wins the declared criterion despite the tree's extra connections. Refit on the development
      specimens it scores {scores.selectedTest.correct}/{scores.selectedTest.rows} correct
      and {num(scores.selectedTest.logLoss)} log loss on the reserved test specimens; the development prior alone
      scores {scores.priorTest.correct}/{scores.priorTest.rows} and {num(scores.priorTest.logLoss)}. One small single
      split cannot establish a universal ordering between the two recipes, and no collection dates or grouping
      metadata accompany these specimens to support a claim about other regions or later vintages.</Prose>

    <MeasuredContrastFigure />

    <H3>Missing a measurement means summing possibilities</H3>
    <Prose>The training medians are <Math>{'[' + trainingModels.treeAugmented.medians.map(m => num(m)).join(',') + ']'}</Math>.
      The learned tree gives {treeParentNames.length} of the four measurements a measurement parent, and in every case
      that parent is {protocol.featureLabels[0]}, in addition to the cultivar. To calculate a posterior from a subset
      of measurements, sum the joint over both states of every hidden measurement, then normalise across the
      cultivars. Do not fill an unknown measurement with its more likely state: that throws away part of the joint
      probability mass.</Prose>

    <MeasurementLab />

    <Prose>This is a demonstration of conditional inference under a fixed fitted model, not a validated
      measurement-purchasing policy. Missingness that is caused by the unobserved value itself may require modelling
      the missingness mechanism; nothing in the visible-subset calculations above establishes that ignoring that
      mechanism is appropriate here.</Prose>

    <Prose>The complete experiment — counts, tree selection, exact missing-measurement marginalisation, the split
      protocol and every saved numerical example — is one
      file: <a href="/learn-assets/bayesian-networks/network-experiments.py" download>network-experiments.py</a>. Save
      it beside the CSV above and run it with Python, NumPy and scikit-learn. Its inference mechanism in words you can
      map to the code: enumerate the 16 possible binary measurement states, discard only the assignments that
      contradict a visible value, multiply the class prior by four CPT entries for each cultivar, add the compatible
      assignments, then normalise. With nothing visible, all the conditional tables sum away and the answer is exactly
      the class prior.</Prose>

    {/* ============================================================ §6 */}
    <H2>{headings[5]}</H2>
    <Prose>Now add causal meaning explicitly. Consider a constructed maintenance
      model <Math>{'Z\\to X\\to Y'}</Math> with <Math>{'Z\\to Y'}</Math>, where <Math>{'Z=1'}</Math> is high
      load, <Math>{'X=1'}</Math> a particular service procedure and <Math>{'Y=1'}</Math> a later failure. Half the
      systems run at high load. The procedure is used with probability {num(fixtures.service.assignment[0])} at low
      load and {num(fixtures.service.assignment[1])} at high load.</Prose>
    <LessonTable caption="Failure probabilities. These numbers intentionally describe a harmful procedure."
      headers={['Load Z', 'Failure without the procedure', 'Failure with the procedure']}
      rows={[0, 1].map(z => [String(z), num(fixtures.service.outcome[0][z]), num(fixtures.service.outcome[1][z])])} />
    <Prose>A name such as “treatment” or “service” does not make an intervention beneficial.</Prose>

    <Prose>Among serviced systems, {num(100 * servicedHighLoad)}% run at high load, because the assignment mechanism
      selected them that way. Their failure probability is {num(service.lanes[1].observedRisk)}. Among unserviced
      systems the high-load share is only {num(service.lanes[0].observedShares[1])}, giving failure
      probability {num(service.lanes[0].observedRisk)}. The observed difference
      is {num(service.associationDifference)}.</Prose>
    <Prose>Suppose we instead assign <strong>every</strong> system the procedure, leaving the load and the failure
      mechanism unchanged. The high-load share stays at one half, so</Prose>
    <MathBlock>{'\\begin{gathered}P(Y=1\\mid do(X=1))\\\\=.5(.05)+.5(.20)=' + num(service.lanes[1].interventionRisk) + '.\\end{gathered}'}</MathBlock>
    <Prose>Assigning no system the procedure gives {num(service.lanes[0].interventionRisk)}. The causal risk
      difference is <strong>{num(service.causalDifference)}</strong>. The observational difference overstates it
      by {num(service.bias)} — which is not the same as the difference between the two treated probabilities.</Prose>

    <Prose>The <strong>do operator</strong> denotes the intervention that replaces the mechanism
      assigning <Math>{'X'}</Math> with a fixed value. In a causally interpreted, fully observed DAG with independent
      external disturbances, the post-intervention joint is the old factor product
      with <Math>{'P(x\\mid pa_X)'}</Math> removed, <Math>{'X'}</Math> fixed, and every other mechanism retained.
      That is the <strong>truncated factorisation</strong>. Conditioning on <Math>{'X=x'}</Math> instead keeps the
      assignment factor and renormalises, which changes the composition of the group you are looking at.</Prose>

    <InterventionLab />

    <Callout title="A causal graph is an assumption, not a fitted result">
      Fitting a Bayesian network to observations does not license reading its arrows as mechanisms. A randomised
      experiment can justify an assignment mechanism; domain knowledge, temporal constraints and scientific arguments
      support other arrows or exclusions. Observational fit alone never establishes that replacing a table describes
      a real intervention.
    </Callout>

    <H3>Adjustment: block the paths that enter the treatment</H3>
    <Prose>In the service model, <Math>{'X\\leftarrow Z\\to Y'}</Math> is a <strong>backdoor path</strong>. Comparing
      procedure groups within load strata and averaging with the target population's load distribution gives</Prose>
    <MathBlock>{'\\begin{gathered}P(y\\mid do(x))\\\\[4pt]=\\sum_z P(y\\mid x,z)\\,P(z).\\end{gathered}'}</MathBlock>
    <Prose>The sufficient <strong>backdoor criterion</strong> selects a set that contains no descendant
      of <Math>{'X'}</Math> and blocks every path from <Math>{'X'}</Math> to <Math>{'Y'}</Math> whose first arrow
      points into <Math>{'X'}</Math>. You also need the required treatment levels to occur within the strata being
      averaged — <strong>positivity</strong> — and measurements and causal assumptions appropriate to the
      question.</Prose>
    <Prose>Set the two assignment probabilities to 0 and 1 in the investigation above and procedure status identifies
      load perfectly. The fully specified teaching model still returns a causal difference
      of {num(noOverlap.causalDifference)}, but observed data alone never reveal the missing
      low-load-with-procedure and high-load-without-procedure response cells. Do not present the adjustment
      calculation as estimated from those data. Knowing a generative model and identifying a quantity from available
      observations are different things.</Prose>

    <H3>More than one set can be valid</H3>
    <Prose>Take the graph <Math>{'S\\to E,\\ S\\to Y,\\ E\\to T,\\ T\\to Y'}</Math>, and ask for the effect
      of <Math>{'T'}</Math> on <Math>{'Y'}</Math>. There are exactly two paths between them: the direct
      edge <Math>{'T\\to Y'}</Math>, which is the causal path we are trying to measure, and{' '}
      <Code>{soleBackdoorPath.path.join('–')}</Code>, whose first arrow points into <Math>{'T'}</Math>. That second
      path is the only backdoor route, and both of its interior nodes are noncolliders.</Prose>
    <LessonTable caption="Every candidate adjustment set for the effect of T on Y in that graph, checked against both parts of the criterion"
      headers={['Set', 'Contains a descendant of T?', 'Backdoor path blocked?', 'Valid']}
      rows={educationSets.map(entry => [
        entry.set.length ? `{${entry.set.join(', ')}}` : 'the empty set',
        entry.descendantViolations.length ? entry.descendantViolations.join(', ') : 'no',
        entry.openPaths.length ? 'no' : 'yes',
        entry.valid ? 'yes' : 'no',
      ])} />
    <Prose>So <Math>{'\\{E\\}'}</Math> <strong>and</strong> <Math>{'\\{S\\}'}</Math> both work, and so does their
      union, subject to the data actually supporting the strata. Conditioning on <Math>{'E'}</Math> creates no new
      path here, because the only collider in this graph is <Math>{'Y'}</Math> itself, which is an endpoint of the
      query rather than an interior node of any <Math>{'T'}</Math>-to-<Math>{'Y'}</Math> path. The graph does not
      declare the more upstream variable universally better. Measurement quality, cost, support and statistical
      efficiency can distinguish valid choices; being earlier in the drawing cannot.</Prose>
    <Prose>Add one arrow, <Math>{'E\\to Y'}</Math>, and that changes. Practice 5 asks you to work it out; the answer
      is that <Math>{'\\{S\\}'}</Math> stops being valid while <Math>{'\\{E\\}'}</Math> and <Math>{'\\{E,S\\}'}</Math> remain
      so. Investigation 2 grades a different question — d-separation over the whole graph, which counts the
      direct causal path that the backdoor criterion deliberately sets aside — so it will not confirm this table.
      The practice-5 graph is one of its presets under a question it <em>can</em> settle: whether status and
      training can be separated at all.</Prose>

    <Checkpoint prompt={'A colleague proposes adjusting for every variable that was measured, on the grounds that '
      + 'more control is safer. Name three separate ways that can go wrong, using the vocabulary of this section.'}>
      <Prose>A collider can <em>open</em> a route that was closed: conditioning on a common effect of the treatment
        and the outcome creates an association where the graph had none. A mediator removes part of the total effect,
        because blocking a directed causal path answers a different question from the one asked. And additional
        variables can worsen overlap: the more finely the population is stratified, the more likely some stratum
        contains only one treatment level, at which point positivity fails and the adjustment formula has an empty
        denominator rather than a small one. First name the causal quantity you want, then inspect the paths.{' '}
        <a href="https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf">Pearl's causal-inference paper, printed
        pages 2517–2519</a>, develops the criterion and its assumptions.</Prose>
    </Checkpoint>

    {/* ============================================================ §7 */}
    <H2>{headings[6]}</H2>
    <Prose><strong>This section is a deeper branch.</strong> It assumes sections 1–6 and develops identification,
      the three do-calculus rules and individual counterfactuals.</Prose>

    <H3>A measured mediator can sometimes bypass an unmeasured cause</H3>
    <Prose>Suppose an unobserved <Math>{'U'}</Math> influences both <Math>{'X'}</Math> and <Math>{'Y'}</Math>, while{' '}
      <Math>{'X\\to M\\to Y'}</Math>, with no direct <Math>{'X\\to Y'}</Math> arrow. Under this graph
      the <strong>frontdoor criterion</strong> can identify the total effect using only the
      observed <Math>{'X,M,Y'}</Math>. It has three conditions, and they hold here:</Prose>
    <LessonTable caption="The three frontdoor conditions, checked against the stated graph rather than asserted"
      headers={['Condition', 'Holds here', 'Why']}
      rows={[
        ['M intercepts every directed path from X to Y', frontdoorOk.intercepts ? 'yes' : 'no',
          `The directed paths are ${frontdoorOk.directedPaths.map(path => path.join('→')).join(' and ')}.`],
        ['No backdoor path from X to M is open', frontdoorOk.treatmentToMediatorOpen.length ? 'no' : 'yes',
          'Any route back out of X reaches M only through the collider at Y, which is unobserved.'],
        ['Conditioning on X blocks every backdoor path from M to Y', frontdoorOk.mediatorToOutcomeOpen.length ? 'no' : 'yes',
          'The route M ← X ← U → Y is blocked at the noncollider X once X is conditioned on.'],
      ]} />
    <Prose>With the required support, the result is</Prose>
    <MathBlock>{'\\begin{gathered}P(y\\mid do(x))\\\\[4pt]=\\sum_m P(m\\mid x)\\,q(m),\\\\[6pt]q(m)=\\sum_{x\'}P(y\\mid m,x\')\\,P(x\').\\end{gathered}'}</MathBlock>
    <Prose>Written as two steps, because that is how it is computed. The inner
      average <Math>{'q(m)'}</Math> estimates the response to setting the mediator to <Math>{'m'}</Math>, corrected
      for the mediator's association with the upstream treatment. The outer sum then combines those two responses
      using the mediator distribution that setting <Math>{'X=x'}</Math> induces. Figure 5 below is those two steps as
      two trays. Simply conditioning on the mediator is a different calculation.
      These conditions and the formula appear in <a href="https://ftp.cs.ucla.edu/pub/stat_ser/uai12-mohan-pearl.pdf">Mohan
      and Pearl's graphical-model tutorial</a>.</Prose>

    <FrontdoorFigure />

    <Callout title="An unobserved common cause does not settle the question">
      Lack of a valid backdoor set is not the same as nonidentification. An unobserved common cause blocks <em>ordinary
      adjustment</em> only on the paths where no observed noncollider lies: if a backdoor route runs through an
      observed noncollider elsewhere, conditioning on that variable blocks it, and the frontdoor construction above
      identifies an effect in a graph where no adjustment set exists at all. The reverse also holds: observing a
      mediator does not by itself license the frontdoor formula. Check the exact graph and name which criterion you
      are appealing to.
    </Callout>

    <Prose>Two small changes break it, in two different ways. Adding <Math>{'U\\to M'}</Math> opens{' '}
      <Math>{'X\\leftarrow U\\to M'}</Math>, so the second condition fails
      ({frontdoorLatentMediator.treatmentToMediatorOpen.length} backdoor path from <Math>{'X'}</Math> to{' '}
      <Math>{'M'}</Math> is then open). Adding <Math>{'X\\to Y'}</Math> instead leaves the mediator intercepting only{' '}
      {frontdoorDirect.directedPaths.filter(path => path.includes('M')).length} of{' '}
      {frontdoorDirect.directedPaths.length} directed paths, so the first condition fails. Neither change licenses the
      same formula, and neither failure proves the effect is unidentifiable by any route.</Prose>

    <H3>The three do-calculus rules, with their graph operations</H3>
    <Prose>For disjoint variable sets <Math>{'X,Y,Z,W'}</Math>, let <Math>{'G_{\\bar X}'}</Math> remove the arrows
      entering <Math>{'X'}</Math>, and <Math>{'G_{\\underline Z}'}</Math> remove the arrows
      leaving <Math>{'Z'}</Math>. Combined subscripts apply both. Each rule transforms an expression when the
      indicated d-separation holds <em>in its own modified graph</em>; ordinary d-separation in the original graph
      cannot substitute for these tests.</Prose>
    <Prose><strong>Rule 1 — drop an irrelevant observation.</strong> If <Math>{'Y\\perp Z\\mid X,W'}</Math> in{' '}
      <Math>{'G_{\\bar X}'}</Math>:</Prose>
    <MathBlock>{'\\begin{gathered}P(y\\mid do(x),z,w)\\\\=P(y\\mid do(x),w).\\end{gathered}'}</MathBlock>
    <Prose><strong>Rule 2 — exchange an action for an observation.</strong> If <Math>{'Y\\perp Z\\mid X,W'}</Math> in{' '}
      <Math>{'G_{\\bar X,\\underline Z}'}</Math>:</Prose>
    <MathBlock>{'\\begin{gathered}P(y\\mid do(x),do(z),w)\\\\=P(y\\mid do(x),z,w).\\end{gathered}'}</MathBlock>
    <Prose><strong>Rule 3 — drop an irrelevant action.</strong> If <Math>{'Y\\perp Z\\mid X,W'}</Math> in{' '}
      <Math>{'G_{\\bar X,\\overline{Z(W)}}'}</Math>:</Prose>
    <MathBlock>{'\\begin{gathered}P(y\\mid do(x),do(z),w)\\\\=P(y\\mid do(x),w).\\end{gathered}'}</MathBlock>
    <Prose>Here <Math>{'Z(W)'}</Math> contains the <Math>{'Z'}</Math>-nodes that are <em>not</em> ancestors of
      any <Math>{'W'}</Math>-node in <Math>{'G_{\\bar X}'}</Math>; when <Math>{'W'}</Math> is empty it is all
      of <Math>{'Z'}</Math>.</Prose>
    <Prose>A worked case: in a causally sufficient <Math>{'X\\to Y'}</Math> model with no other paths, delete the
      outgoing arrow from <Math>{'X'}</Math>. Now <Math>{'X'}</Math> and <Math>{'Y'}</Math> are separated in that
      modified graph, so rule 2 justifies <Math>{'P(y\\mid do(x))=P(y\\mid x)'}</Math>. Add an unobserved common
      cause, and the remaining backdoor route prevents exactly that argument.{' '}
      <a href="https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf">Pearl's rule statements</a> give the formal
      conditions.</Prose>
    <Prose><strong>Identification</strong> asks whether every causal model satisfying the assumptions and yielding the
      same observational distribution agrees on the causal query. <strong>Estimation</strong> asks how to approximate
      an identified quantity from finite observations. Do-calculus plus ordinary probability operations is complete
      for the relevant interventional identification problems in the standard acyclic framework with latent variables
      allowed; that is not a promise that every effect, every counterfactual or every feedback system is identified.</Prose>

    <H3>Counterfactuals need the same unit in two worlds</H3>
    <Prose>A <strong>structural causal model</strong> writes variables as assignments
      such as <Math>{'Y=f_Y(X,U_Y)'}</Math>, with external variables <Math>{'U'}</Math> describing the variation not
      otherwise represented. Independent external noises in an acyclic, fully observed model yield the familiar causal
      DAG factorisation; dependent external variables require representing that latent dependence rather than
      silently multiplying independent noise distributions.</Prose>
    <Prose>An individual counterfactual takes three steps: infer the external state from what
      happened (<strong>abduction</strong>), replace the specified assignment (<strong>action</strong>), and run the
      changed model using that same inferred state (<strong>prediction</strong>). A population intervention averages
      over the population's external states instead.</Prose>
    <Prose>Consider randomised binary <Math>{'X'}</Math> and an independent fair binary <Math>{'U'}</Math>, and
      compare model A, <Math>{'Y=U'}</Math>, with model B, <Math>{'Y=X\\mathbin{\\mathrm{XOR}}U'}</Math>, where XOR is
      1 exactly when its inputs differ.</Prose>

    <CounterfactualFigure />

    <Prose>Both models
      have <Math>{'P(Y=1\\mid X=x)=P(Y=1\\mid do(X=x))=1/2'}</Math> for either <Math>{'x'}</Math>, so they agree on
      every observational and population-interventional distribution of <Math>{'X,Y'}</Math>. Yet for a unit observed
      with <Math>{'X=0,Y=0'}</Math>, both infer <Math>{'U=' + observedUnit[0].inferredExternalState}</Math>; changing
      that unit's <Math>{'X'}</Math> to 1 yields <Math>{'Y=' + observedUnit[0].prediction}</Math> in
      A and <Math>{'Y=' + observedUnit[1].prediction}</Math> in B.</Prose>
    <Prose>The distinction is the coupling of the same unit's outcomes across settings, and a conditional probability
      table alone did not determine it. One may include an unused <Math>{'X\\to Y'}</Math> parent in model A if using a
      common permissive graph; it then exhibits an extra independence rather than faithfulness. Some particular
      counterfactual quantities can be identified under weaker assumptions than a fully numerically specified
      structural model, but the population tables in this example are not enough.</Prose>

    {/* ============================================================ §8 */}
    <H2>{headings[7]}</H2>
    <Prose><strong>This section is a deeper branch.</strong> It covers structure search, exact compilation,
      approximate inference, query families and the neighbouring model classes.</Prose>

    <H3>A graph can predict well without revealing a unique direction</H3>
    <Prose>With complete discrete training data, fixed-graph likelihood is easy to evaluate. Structure search can add,
      remove or reverse an edge while preserving acyclicity, scoring fit against complexity. Under the usual
      regular-model approximation, a BIC score to <strong>maximise</strong>
      is <Math>{'\\ell(\\hat\\theta)-(k/2)\\log n'}</Math>; equivalently minimise <Math>{'-2\\ell+k\\log n'}</Math>,
      using the actual CPT parameter count for <Math>{'k'}</Math>. That is a statistical approximation with
      assumptions, not a universal penalty for arbitrary latent or singular models.</Prose>
    <Prose>Searching many DAGs is expensive and greedy hill climbing can stop at a local optimum; parent limits,
      domain constraints and score caching help. Conditional-independence approaches such as PC instead remove
      adjacencies through independence tests and orient only what the resulting constraints justify. Under their
      standard correctness conditions they rely on assumptions including causal sufficiency, Markovness, faithfulness
      and suitable independence information, and finite samples can make the tests unstable. Latent-variable
      approaches such as FCI address a different assumption set and represent partially determined structure.</Prose>
    <Prose>For ordinary DAG independence models, <strong>Markov-equivalent</strong> graphs share the same skeleton and
      the same unshielded colliders. The skeleton forgets arrowheads; an unshielded
      collider <Math>{'X\\to Z\\leftarrow Y'}</Math> has no <Math>{'X{-}Y'}</Math> edge.</Prose>

    <EquivalenceFigure />

    <Prose>Pure observational independence information cannot choose between the three non-collider orientations.
      Additional functional assumptions or actual interventions may identify more, so do not claim that observational
      orientation is always impossible under every model class.</Prose>
    <Prose>If values or variables are missing during fitting, local complete-data counts no longer suffice. EM
      alternates posterior expected sufficient counts under current parameters with parameter updates; exact E-steps
      can be costly, and local optima and unidentifiability remain possible. That connects directly to
      the <a href="/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml">Gaussian
      mixture</a> lesson, where the hidden quantity was a component membership rather than a graph node.</Prose>

    <H3>Compile repeated exact queries, or approximate deliberately</H3>
    <Prose>A <strong>junction tree</strong> groups interacting variables into clusters connected as a tree, and
      neighbouring clusters exchange functions over their shared variables. Every variable must appear in a connected
      set of clusters — the <strong>running-intersection property</strong> — so that messages can summarise the
      excluded subproblems consistently. Building such clusters usually involves moralisation and triangulation: add
      fill edges to remove chordless cycles of length four or more, then organise the cliques. Triangulation does not
      remove every cycle from the graph, and its large clusters are where treewidth reappears.{' '}
      <a href="https://ermongroup.github.io/cs228-notes/inference/jt/">Stanford's junction-tree chapter</a> is a
      useful derivation after the factor workbench in section 3.</Prose>
    <Prose>Sum-product messages are exact on an appropriate tree of factors or clusters. Loopy belief propagation
      applies similar local updates to a graph with loops, where convergence and exactness are no longer
      automatic.</Prose>
    <LessonTable caption="Approximate choices for large discrete models, and a failure worth checking for each"
      headers={['Method', 'Main operation', 'A failure worth checking']}
      rows={[
        ['Ancestral sampling', 'Sample each node after its parents', 'Rare evidence makes rejection discard almost everything'],
        ['Likelihood weighting', 'Fix evidence; weight samples by its likelihood', 'A few samples may carry almost all the weight'],
        ['Gibbs sampling', 'Resample each variable given its blanket', 'Strong or deterministic constraints can prevent useful movement'],
        ['Variational inference', 'Optimise a tractable approximating distribution', 'The family and objective can miss important dependence or modes'],
        ['Loopy belief propagation', 'Iterate local messages', 'Oscillation, or a stable but inaccurate fixed point'],
      ]} />
    <Prose>For the alarm's two-call evidence, rejection sampling retains
      about {percent(retainedFraction, 4)}% of prior samples on average: roughly{' '}
      {whole(fixtures.rejectionSampleSize * retainedFraction)} of{' '}
      {fixtures.rejectionSampleSize.toLocaleString('en-US')}. Weighting avoids literal rejection but can still
      concentrate weight severely. For correlated Monte Carlo draws the raw draw count overstates precision, so assess
      effective sample size and exploration; a Gibbs chain constrained to keep two binary variables equal may be
      unable to move either coordinate alone, and blocked updates or another sampler may be needed.{' '}
      <a href="https://ermongroup.github.io/cs228-notes/inference/sampling/">Stanford's sampling chapter</a> supplies
      the mechanisms.</Prose>

    <H3>Three query families that are not interchangeable</H3>
    <Prose>A <strong>marginal</strong> asks for <Math>{'P(Y\\mid e)'}</Math>. <strong>MPE</strong> selects the
      highest-probability assignment to <em>all</em> remaining hidden variables. A <strong>marginal-MAP</strong> query
      maximises over a selected set after summing the others out. Max and sum generally cannot swap, and a single
      example settles it.</Prose>
    <LessonTable caption="Four joint masses over a query variable Q and a hidden variable H"
      headers={['Q', 'H', 'mass']}
      rows={mapQuery.entries.map(entry => [String(entry.query), String(entry.hidden), num(entry.value)])} />
    <Prose>The highest-probability world here has <Math>{'Q=' + mapQuery.mostProbableWorld.query}</Math>, at
      mass {num(mapQuery.mostProbableWorld.value)}. The marginal-MAP answer
      is <Math>{'Q=' + mapQuery.marginalMapQuery}</Math>, because its rows combine
      to {num(mapQuery.marginalMapMass)} against {num(mapQuery.rowMasses[1 - mapQuery.marginalMapQuery])}. The two
      questions have different answers on the same distribution.</Prose>

    <H3>Choosing a representation for the question</H3>
    <Prose>A hidden Markov model is a time-unrolled directed model with repeated local transition and emission rules,
      and its forward algorithm is specialised variable elimination. The next lesson, Conditional Random Fields,
      instead models <Math>{'P(\\text{labels}\\mid\\text{observations})'}</Math> directly and need not supply a
      generative distribution for the observations at all; that changes which quantities the model can
      answer.</Prose>
    <Prose>An undirected Markov random field uses compatibility factors and a partition function. Neither directed nor
      undirected graphical independence models universally contains the other: a nonchordal undirected cycle and a
      directed collider illustrate why their conditional-independence semantics differ. Linear-Gaussian networks
      replace discrete CPTs with local linear regressions and Gaussian disturbances and can retain analytic Gaussian
      inference; nonlinear or neural conditional distributions broaden expressive power but may require different
      inference methods. Probabilistic programming can express far richer hierarchical and latent models, but the
      inference engine still has requirements — gradient-based HMC is designed for suitable continuous latent spaces,
      not raw discrete state jumps, so discrete enumeration, marginalisation, custom samplers or variational
      approximations may be needed.</Prose>
    <Prose>A useful application beyond classification is <strong>diagnosis with selectively missing sensors</strong>:
      maintain a distribution over underlying faults while measurements arrive, exactly as investigation 3 does with
      chemical measurements. A separate utility model is needed to compare the expected value of a test against its
      cost; probability alone does not decide what action is worthwhile. Reliability planning similarly distinguishes
      observing a component failure from replacing a component's failure mechanism. Genomic and neural applications
      can use graphs to state competing explanations, provided correlated signals and selection effects are not
      relabelled as causal connections.</Prose>

    <H3>A current library route</H3>
    <Prose>The program below builds the same five-node network through pgmpy, with the state order and the
      parent-column order written out explicitly. The content phase left it written but unexecuted. This
      implementation installed <strong>pgmpy {bayesnetExamples.pgmpy.environment.pgmpy}</strong> into an isolated
      environment and ran it there, so the output shown below is what it actually printed. The isolation was not
      ceremony: resolving pgmpy moved NumPy
      to {bayesnetExamples.pgmpy.environment.numpy} and pandas to {bayesnetExamples.pgmpy.environment.pandas},
      against the {bayesnetExamples.enumerate.environment.numpy} and {bayesnetExamples.enumerate.environment.pandas} the
      other programs on this page ran on, and every recorded output in this curriculum depends on those exact
      versions. The code follows the currently
      inspected <a href="https://pgmpy.org/api/generated/models/pgmpy.models.DiscreteBayesianNetwork.html">DiscreteBayesianNetwork</a> and{' '}
      <a href="https://pgmpy.org/api/generated/inference/pgmpy.inference.VariableElimination.html">VariableElimination</a> APIs.</Prose>
    <Program example={bayesnetExamples.pgmpy}>
      <Prose>The second state is the burglary probability, and it agrees with the enumeration in section 2 to the last
        digit NumPy prints. Do not infer a fixed-value intervention from a method's name alone: inspect whether an API
        cuts incoming edges, changes a CPT, sets a particular state or samples an intervention. State ordering and
        evidence-column ordering also matter, because a normalised table can still encode the wrong parent
        configuration.</Prose>
    </Program>

    {/* ============================================================ §9 */}
    <H2>{headings[8]}</H2>
    <Prose>Work through 1–6 before the deeper problems. Each one changes a mechanism or an assumption rather than
      asking you to repeat the preceding trace.</Prose>

    <Practice title="1. A different caller"
      question={<>Replace Mary's table by <Math>{'P(M=1\\mid A=0)=P(M=1\\mid A=1)=0.4'}</Math>. With only Mary
        calling, what is the burglary posterior? With both John and Mary calling, which earlier posterior should
        reappear?</>}
      hint={<>The factor for Mary's evidence is now the same constant in every remaining world.</>}>
      <Prose>It cancels between numerator and denominator. Mary alone gives back the
        prior, {num(uninformativeMaryAlone.posterior)}. Both calls give John's posterior,
        approximately {num(uninformativeMary.posterior)} — the same {num(johnOnly.posterior)} as before. Equal
        conditional rows remove Mary's information from this distribution, even though the drawn graph still shows the
        arrow. Investigation 1 has both of Mary's rows as editable fields and a “Practice 1” setup that applies this
        exact change, so you can reproduce both numbers there.</Prose>
    </Practice>

    <Practice title="2. A descendant opens a path"
      question={<>Draw <Math>{'R\\to S\\leftarrow T'}</Math> and <Math>{'S\\to V\\to W'}</Math>.
        Are <Math>{'R'}</Math> and <Math>{'T'}</Math> d-separated with no observations, with <Math>{'W'}</Math> observed,
        and with both <Math>{'S'}</Math> and <Math>{'W'}</Math> observed? Does the graph specify a negative
        correlation in the latter two cases?</>}
      hint={<>Ask whether the collider has an observed descendant; then separate “active” from the sign of any
        association.</>}>
      <Prose>{dSeparation(fixtures.colliderChainEdges, 'R', 'T', []).summary} With <Math>{'W'}</Math> observed
        the path opens, because {dSeparation(fixtures.colliderChainEdges, 'R', 'T', ['W']).paths[0].reason}. With
        both <Math>{'S'}</Math> and <Math>{'W'}</Math> observed it is open
        too: {dSeparation(fixtures.colliderChainEdges, 'R', 'T', ['S', 'W']).paths[0].reason}. The graph supplies no
        numerical sign or strength at all. Explaining away in section 2 was a consequence of the alarm's particular
        probabilities, not a universal negative-correlation theorem. This graph is a preset in investigation 2.</Prose>
    </Practice>

    <Practice title="3. Count a different network"
      question={<>Let a three-state <Math>{'C'}</Math> parent three binary
        features <Math>{'F_1,F_2,F_3'}</Math>, and add <Math>{'F_1\\to F_2'}</Math>. How many free CPT parameters are
        there? Compare with a fully unrestricted joint over the four variables.</>}
      hint={<>Count each node's parent configurations, then multiply by its number of states minus one.</>}>
      <Prose>{practiceParameters.rows.map(row =>
        `${row.name} contributes ${row.configurations} × ${row.states - 1} = ${row.free}`).join('; ')}, for a total
        of {practiceParameters.free}. The unrestricted joint has {practiceParameters.jointEntries} entries
        and {practiceParameters.jointFree} free parameters. These count probabilities, not data rows and not
        bytes.</Prose>
    </Practice>

    <Practice title="4. A probability query with one hidden variable"
      question={<>You have <Math>{'P(C=1)=0.4'}</Math>, <Math>{'P(F=1\\mid C=0)=0.2'}</Math> and <Math>{'P(F=1\\mid C=1)=0.8'}</Math>.
        A downstream measurement <Math>{'G'}</Math> depends only on <Math>{'F'}</Math>,
        with <Math>{'P(G=1\\mid F=0)=0.1'}</Math> and <Math>{'P(G=1\\mid F=1)=0.9'}</Math>.
        Calculate <Math>{'P(C=1\\mid G=1)'}</Math> by summing over <Math>{'F'}</Math>.</>}
      hint={<>First obtain the two likelihoods <Math>{'P(G=1\\mid C)'}</Math>.</>}>
      <Prose>They are <Math>{'0.8(0.1)+0.2(0.9)=0.26'}</Math> for <Math>{'C=0'}</Math>
        and <Math>{'0.2(0.1)+0.8(0.9)=0.74'}</Math> for <Math>{'C=1'}</Math>. So the answer
        is <Math>{'0.4(0.74)/[0.6(0.26)+0.4(0.74)]=0.296/0.452\\approx0.654867'}</Math>. Choosing the most
        likely <Math>{'F'}</Math> first and then conditioning on it would solve a different problem, and would give a
        different number.</Prose>
    </Practice>

    <Practice title="5. Valid adjustment sets, changed graph"
      question={<>For <Math>{'S\\to E,\\ S\\to Y,\\ E\\to T,\\ T\\to Y'}</Math>, check <Math>{'\\{E\\}'}</Math> and{' '}
        <Math>{'\\{S\\}'}</Math>. Now add <Math>{'E\\to Y'}</Math>. Which of <Math>{'\\{E\\},\\{S\\},\\{E,S\\}'}</Math> still
        satisfy the backdoor criterion for the total effect of <Math>{'T'}</Math> on <Math>{'Y'}</Math>?</>}
      hint={<>List the new route that begins <Math>{'T\\leftarrow E'}</Math>.</>}>
      <Prose>Originally both singleton sets work, as the table in section 6 shows. Adding <Math>{'E\\to Y'}</Math> creates
        a second backdoor route, <Code>{educationWithDirect[0].backdoorPaths.find(entry => entry.path.length === 3)?.path.join('–')}</Code>,
        which <Math>{'S'}</Math> does not touch.</Prose>
      <LessonTable caption="The same three candidate sets after the arrow from E to Y is added"
        headers={['Set', 'Backdoor paths open', 'Valid']}
        rows={educationWithDirect.map(entry => [
          `{${entry.set.join(', ')}}`,
          entry.openPaths.length ? entry.openPaths.map(path => path.path.join('–')).join(', ') : 'none',
          entry.valid ? 'yes' : 'no',
        ])} />
      <Prose>So <Math>{'\\{E\\}'}</Math> and <Math>{'\\{E,S\\}'}</Math> work
        while <Math>{'\\{S\\}'}</Math> does not. Required support and correct graph assumptions still apply. Being
        earlier in the graph was never sufficient to be a valid adjustment variable, and this is the counterexample
        that shows why.</Prose>
    </Practice>

    <Practice title="6. A more cautious real-data claim"
      question={<>The two Wine recipes classify the same number of validation specimens correctly, but naive Bayes has
        the lower log loss. Explain how that can happen. Would reporting the tree-augmented model's test result after
        seeing naive Bayes's test loss preserve the declared selection protocol? And in investigation 3, why must
        changing a <em>hidden</em> alcohol value leave the posterior unchanged?</>}
      hint={<>Distinguish the largest-probability label from the whole probability vector, and distinguish information
        available to a query from a value stored in a record.</>}>
      <Prose>Two models can differ in confidence while agreeing on the label, and can even make different errors with
        the same total count. Log loss reads the true-class probability on every row; accuracy reads only which entry
        is largest. In figure 4 both models misclassify the same number of specimens, and the difference between{' '}
        {num(scores.naiveBayesValidation.logLoss)} and {num(scores.treeAugmentedValidation.logLoss)} is assembled
        entirely from confidence.</Prose>
      <Prose>No, it would not preserve the protocol. Choosing or emphasising a second model after inspecting the
        held-out result reuses test information for selection; a new selection needs new evaluation, or a
        transparently exploratory report that says the assessment is no longer independent.</Prose>
      <Prose>A hidden measurement is marginalised over: the query sums across both of its states and the stored value
        never enters the calculation. It cannot be read until it becomes evidence. That is why the “edit a hidden
        value” case in investigation 3 is an exact null rather than a small change.</Prose>
    </Practice>

    <Practice title="7. Change a causal response, not its assignment"
      question={<>In the service example, change the low-load procedure failure probability
        from {num(fixtures.service.outcome[1][0])} to {num(fixtures.serviceChangedResponse.outcome[1][0])}, leaving
        everything else fixed. What is the new causal difference? Under the original assignment probabilities, what is
        the observed difference?</>}
      hint={<>Use population weights <Math>{'[1/2,1/2]'}</Math> for the intervention and serviced-group
        weights <Math>{'[1/4,3/4]'}</Math> for the observation.</>}>
      <Prose>The intervention probability with the procedure
        becomes <Math>{'0.5(0.01)+0.5(0.20)=' + num(changedResponse.lanes[1].interventionRisk)}</Math>.
        Without it, {num(changedResponse.lanes[0].interventionRisk)} is unchanged, so the causal difference
        is {num(changedResponse.causalDifference)}. The observed serviced risk
        becomes {num(changedResponse.lanes[1].observedRisk)}; subtracting the
        unchanged {num(changedResponse.lanes[0].observedRisk)} gives {num(changedResponse.associationDifference)}.
        Changing one response cell moves the two averages by different amounts, because their weights differ —
        {' '}<Math>{'[1/2,1/2]'}</Math> against <Math>{'[1/4,3/4]'}</Math>. Investigation 4 has this exact edit as a
        suggested setup.</Prose>
    </Practice>

    <Practice title="8. Marginal answer versus best world"
      question={<>Change the four masses in section 8
        to <Math>{'[' + practiceMap.entries.map(entry => num(entry.value)).join(',') + ']'}</Math> in the
        same <Math>{'(Q,H)'}</Math> order. Find both answers. Why does their matching here not justify replacing sum
        with max in general?</>}
      hint={<>Compare the largest single cell with each row sum.</>}>
      <Prose>The most likely world
        is <Math>{'(' + practiceMap.mostProbableWorld.query + ',' + practiceMap.mostProbableWorld.hidden + ')'}</Math>,
        and the marginal-MAP value is also <Math>{'Q=' + practiceMap.marginalMapQuery}</Math>, whose mass
        is {num(practiceMap.marginalMapMass)} against {num(practiceMap.rowMasses[1 - practiceMap.marginalMapQuery])}.
        They coincide here. Section 8's original table is a counterexample to the proposed general shortcut, and one
        coincidence cannot establish an algebraic identity.</Prose>
    </Practice>

    <Practice title="9. Break a frontdoor assumption"
      question={<>In section 7's graph, add <Math>{'U\\to M'}</Math>. Is the displayed frontdoor formula still
        justified by that criterion? What if instead you add only <Math>{'X\\to Y'}</Math>?</>}
      hint={<>Check the backdoor paths from the treatment to the mediator, and the set of directed treatment-to-outcome
        paths, separately.</>}>
      <Prose>The first addition opens <Math>{'X\\leftarrow U\\to M'}</Math>, so the second condition fails: the graph
        now has {frontdoorLatentMediator.treatmentToMediatorOpen.length} open backdoor path
        from <Math>{'X'}</Math> to <Math>{'M'}</Math>, against {frontdoorOk.treatmentToMediatorOpen.length} before.
        The direct <Math>{'X\\to Y'}</Math> addition instead violates the interception condition: there are
        now {frontdoorDirect.directedPaths.length} directed paths from <Math>{'X'}</Math> to <Math>{'Y'}</Math> and
        the mediator lies on only {frontdoorDirect.directedPaths.filter(path => path.includes('M')).length} of them.
        Neither change licenses the same formula. Failure of a sufficient criterion does not by itself prove the
        effect unidentifiable; reassess the graph and the assumptions you are prepared to make.</Prose>
    </Practice>

    <Practice title="10. Find what the probability tables leave unspecified"
      question={<>For the two structural models in section 7, observe <Math>{'X=1,Y=0'}</Math>.
        Infer <Math>{'U'}</Math>, then set <Math>{'X=0'}</Math>. What does each model predict? Which population
        quantities remain equal?</>}
      hint={<>Keep each model's inferred external state fixed while changing <Math>{'X'}</Math>.</>}>
      <Prose>Model A infers <Math>{'U=' + practiceUnit[0].inferredExternalState}</Math> and still
        predicts <Math>{'Y=' + practiceUnit[0].prediction}</Math>. Model B
        infers <Math>{'U=' + practiceUnit[1].inferredExternalState}</Math> and
        predicts <Math>{'Y=' + practiceUnit[1].prediction}</Math> after the change. Both models still give a
        fair <Math>{'Y'}</Math> under either intervention — the column averages in figure 6
        are {num(counterfactuals.models[0].averageUnderZero)} and {num(counterfactuals.models[0].averageUnderOne)} in
        both — and the same observational distribution over <Math>{'X,Y'}</Math>. They disagree about paired,
        unit-level outcomes, which those distributions never specified.</Prose>
    </Practice>

    {/* ============================================================ §10 */}
    <H2>{headings[9]}</H2>
    <Prose>You are ready to move on when you can explain why an unobserved variable is summed out rather than filled
      in, identify a collider's observed descendant, distinguish a causal query from a predictive one, and justify a
      small adjustment set by naming the path it blocks. For the deeper route, add distinguishing identification from
      finite-sample estimation, and distinguishing a population intervention from an individual counterfactual.</Prose>

    <LessonTable caption="Readiness check"
      headers={['you should be able to', 'where it was taught']}
      rows={[
        ['Assemble one world as a product of five local table entries', 'Section 1, figure 1'],
        ['Add the compatible worlds and renormalise to get a posterior', 'Section 2, investigation 1, practice 4'],
        ['Say when extra evidence changes nothing, and when a posterior has no value at all', 'Section 2, investigation 1, practice 1'],
        ['Eliminate a variable and explain why the answer does not change', 'Section 3, figure 2'],
        ['Read an elimination order off the moral graph and count its largest factor', 'Section 3, figure 3'],
        ['Decide whether an observation set d-separates two variables, including a collider descendant', 'Section 4, investigation 2, practice 2'],
        ['Name a Markov blanket and say what it makes redundant', 'Section 4'],
        ['Fit a discrete CPT by counting, and say what a Dirichlet pseudo-count changes', 'Section 5'],
        ['Read a real two-recipe comparison where accuracy ties and log loss does not', 'Section 5, figure 4, practice 6'],
        ['Answer a query with measurements missing, and say why a hidden edit is a null', 'Section 5, investigation 3'],
        ['Separate an observed difference from a causal one and name the bias', 'Section 6, investigation 4, practice 7'],
        ['Check the backdoor criterion on a stated graph and find every valid set', 'Section 6, practice 5'],
        ['Check the three frontdoor conditions, and break them two different ways', 'Section 7, figure 5, practice 9'],
        ['Run abduction, action and prediction on one unit', 'Section 7, figure 6, practice 10'],
        ['Tell a most probable world from a marginal-MAP answer', 'Section 8, practice 8'],
      ]} />

    <Prose>The next topic in this module
      is <a href="/learn/path/full-curriculum/conditional-random-fields-crf?module=classical-ml">Conditional Random
      Fields</a>. Carry forward factors and normalisation, then ask what changes when only the conditional label
      distribution is modelled and the observations get no generative model at all. The
      separate <a href="/learn/path/full-curriculum/causal-inference-do-calculus?module=math-foundations">causal-inference</a>,{' '}
      <a href="/learn/path/full-curriculum/monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts?module=math-foundations">Monte
      Carlo</a> and <a href="/learn/path/full-curriculum/variational-inference?module=math-foundations">variational-inference</a> lessons
      extend the deeper branches; none of them is a reason to skip this module's reading sequence.</Prose>

    <Sources alternatives={<><Prose>Use these after the core route. The lesson is self-contained; each of these offers
      a second explanation or a fuller reference.</Prose><ul>
      <li><a href="https://ermongroup.github.io/cs228-notes/">Stanford CS228 notes</a>, Volodymyr Kuleshov and Stefano
        Ermon with course staff. Begin with directed representation, then variable elimination and junction trees, and
        use the sampling chapter after exact sums feel familiar. The course also covers undirected models,
        latent-variable learning and variational inference. Read the formulas critically: these are evolving notes
        that acknowledge possible errors, and several sections are marked under construction.</li>
      <li><a href="https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf">Bayesian Network
        Classifiers</a>, Friedman, Geiger and Goldszmidt, 1997. Section 4 explains the conditional-information tree
        and why a constrained graph search is tractable. Our discretisation, split, smoothing choice and choice of
        measurements are an original small experiment here, not a reproduction of the paper's benchmark.</li>
      <li><a href="https://ftp.cs.ucla.edu/pub/stat_ser/uai12-mohan-pearl.pdf">Graphical Models for Causal
        Inference</a>, Mohan and Pearl — a slide-form walkthrough. Use its d-separation, intervention and frontdoor
        diagrams as an alternative visual explanation; the presentation is more formal than this lesson's first pass.</li>
    </ul></>}>
      <li><a href="https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf">The Mathematics of Causal Inference</a>,
        Pearl — the structural-model setup, and printed pages 2517–2519 for d-separation, identification, the three
        do-calculus rules and the backdoor criterion. The later mediation and transport sections go beyond this
        lesson's core route.</li>
      <li><a href="https://pgmpy.org/examples/Inference_Discrete_BN.html">pgmpy discrete inference examples</a> and{' '}
        <a href="https://pgmpy.org/examples/Parameter_Learning_Discrete_BN.html">parameter-learning examples</a> —
        a reusable API beside our small explicit sums. Documentation inspected September 2026; the version this page
        actually executed is named beside the program in section 8.</li>
      <li><a href="https://ermongroup.github.io/cs228-notes/inference/jt/">CS228 — junction trees</a> and{' '}
        <a href="https://ermongroup.github.io/cs228-notes/inference/sampling/">CS228 — sampling</a>, the two chapters
        section 8 refers to directly.</li>
      <li><a href={provenance.doi}>UCI Wine</a>, {provenance.creator}, licensed{' '}
        <a href={provenance.licenseUrl}>{provenance.license}</a> — the actual observations. This page serves{' '}
        <a href={provenance.file} download>its own unchanged copy</a>, {provenance.bytes.toLocaleString('en-US')} bytes,
        SHA-256 <Code>{provenance.sha256}</Code>, beside its <a href={provenance.attribution}>attribution</a>, which
        records the extraction, the column meanings and what the source record does not supply. Reuse the measurements
        to inspect probabilities, not to infer chemical causality.</li>
    </Sources>

    <Prose>The alarm network, the maintenance service model, the mediation example, the two structural causal models
      and every practice matrix on this page are <strong>constructed calculations</strong>, not measurements. The
      failure probabilities of {num(fixtures.service.outcome[1][1])} and {num(fixtures.service.outcome[0][1])} are a
      declared teaching assumption describing a deliberately harmful procedure, not an engineering figure. The Wine
      results are calculations on the identified real dataset under one declared protocol, with
      the {protocol.testSize} reserved test specimens never reachable from any investigation on this page and scored
      exactly once. The {contrast.length} rows of figure 4 are recomputed in your browser from the saved fitted tables
      and reproduce the two recorded log losses exactly; they are not a second experiment. None of this is a benchmark
      or a claim about any future dataset.</Prose>
  </div>,
};

export default bayesianNetworksContent;
