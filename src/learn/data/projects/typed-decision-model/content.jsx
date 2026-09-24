import { useState } from 'react';
import { Link } from 'react-router-dom';
import { exploreDecision } from './decision-model.js';
import './project.css';
import { Commands, Source } from './project-elements.jsx';
import { ProblemWalkthrough, DataWalkthrough, BaselineWalkthrough, ArchitectureWalkthrough } from './mechanism-depth.jsx';
import { TrainingWalkthrough, CalibrationWalkthrough, EvaluationWalkthrough, ServingWalkthrough } from './research-depth.jsx';
import { ScorerTrainingLab } from './mechanism-labs.jsx';

const assets = '/learn-projects/typed-decision-model';
const programUrl = `${assets}/typed_decision.py`;

function Reference({ href, children }) {
  return <a href={href} target="_blank" rel="noreferrer">{children}<span className="tdp-sr-only"> (opens in a new tab)</span></a>;
}

function Readiness({ children }) {
  return <aside className="tdp-deliverable"><span className="tdp-eyebrow">Evidence to carry forward</span>{children}</aside>;
}

function RangeControl({ id, title, value, onChange, min, max, step = 0.1, suffix = '' }) {
  return <label className="tdp-range" htmlFor={id}>
    <span>{title}<output htmlFor={id}>{Number(value).toFixed(step < 1 ? 1 : 0)}{suffix}</output></span>
    <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}

function DecisionExplorer() {
  const [logits, setLogits] = useState([2.2, 0.4, -0.5, 0]);
  const [temperature, setTemperature] = useState(1);
  const [wrongCost, setWrongCost] = useState(10);
  const [reviewCost, setReviewCost] = useState(1);
  const [includeOther, setIncludeOther] = useState(false);
  const names = ['Billing', 'Account access', 'Delivery', 'Other'];
  const count = includeOther ? 4 : 3;
  const result = exploreDecision({ logits: logits.slice(0, count), temperature, wrongCost, reviewCost });
  function reset() {
    setLogits([2.2, 0.4, -0.5, 0]);
    setTemperature(1);
    setWrongCost(10);
    setReviewCost(1);
    setIncludeOther(false);
  }
  return <section className="tdp-explorer" aria-label="Explore probabilities and decision costs">
    <div className="tdp-explorer-heading"><div><span className="tdp-eyebrow">Live investigation</span><h3>Same evidence. Different decisions.</h3></div><button type="button" onClick={reset}>Reset</button></div>
    <p className="tdp-small">These are editable illustrative scores, not outputs of a trained language model. Change any control; every result updates immediately.</p>
    <div className="tdp-explorer-grid">
      <div>
        <h4>1 / Evidence scores</h4>
        {names.slice(0, count).map((name, index) => <RangeControl key={name} id={`tdp-score-${index}`} title={name} value={logits[index]} min={-4} max={4} onChange={value => setLogits(current => current.map((score, position) => position === index ? value : score))} />)}
        <label className="tdp-check"><input type="checkbox" checked={includeOther} onChange={event => setIncludeOther(event.target.checked)} />Add an “Other” candidate</label>
        <RangeControl id="tdp-temperature" title="Temperature" value={temperature} min={0.2} max={3} onChange={setTemperature} />
        <p className="tdp-small">Higher temperature flattens the distribution. It preserves the winning candidate for these fixed scores.</p>
      </div>
      <div>
        <h4>2 / Candidate probabilities</h4>
        <div className="tdp-probabilities">
          {result.probabilities.map((probability, index) => <div className="tdp-probability" key={names[index]}>
            <div><span>{names[index]}</span><strong>{(100 * probability).toFixed(1)}%</strong></div>
            <div className="tdp-bar-track"><span style={{ width: `${100 * probability}%` }} /></div>
          </div>)}
        </div>
        <p className="tdp-small">All bars sum to 100%. Adding a candidate changes the denominator, even if the original scores stay fixed.</p>
        <dl className="tdp-readouts"><div><dt>Largest probability</dt><dd>{(result.probabilities[result.bestIndex] * 100).toFixed(1)}%</dd></div><div><dt>Entropy concentration</dt><dd>{result.concentration.toFixed(3)}</dd></div></dl>
        <p className="tdp-small">Concentration is 1 − H(p)/log K. It describes the distribution’s shape; it is not a measured correctness rate.</p>
      </div>
    </div>
    <div className="tdp-policy-grid">
      <div><h4>3 / Costs in arbitrary units</h4><RangeControl id="tdp-wrong-cost" title="Cost of a wrong action" value={wrongCost} min={1} max={30} step={1} onChange={setWrongCost} /><RangeControl id="tdp-review-cost" title="Cost of review" value={reviewCost} min={0} max={10} onChange={setReviewCost} /></div>
      <div className="tdp-policy-result"><span className="tdp-eyebrow">Policy chooses</span><strong>{result.action === 'act' ? `Act: ${names[result.bestIndex]}` : 'Send for review'}</strong><p>Act cost: {wrongCost} × (1 − {result.probabilities[result.bestIndex].toFixed(3)}) = <b>{result.expectedActCost.toFixed(2)}</b><br />Review cost: <b>{reviewCost.toFixed(2)}</b></p><p className="tdp-small">Choose the lower expected cost; ties go to review. This simplified policy assumes review resolves the case and a correct action costs zero.</p></div>
    </div>
  </section>;
}

function DefineStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">Build a model that answers a question with a probability distribution over choices you supply. Then build the system that decides whether those probabilities justify taking an action.</p>
    <p>Our continuing example routes a support request to billing, account access, or delivery. The same design can rank hypotheses in an annotation workflow or decide which specialist should inspect a research document. Those are possible applications; this project’s measured results come only from its small constructed routing dataset.</p>
    <div className="tdp-contract"><div><span className="tdp-eyebrow">Input</span><p>State + question + candidate descriptions</p><code>“I need help with a refund.”</code></div><span aria-hidden="true">→</span><div><span className="tdp-eyebrow">Model</span><p>A probability for every candidate</p><code>billing: 0.8 · access: 0.1 · delivery: 0.1</code></div><span aria-hidden="true">→</span><div><span className="tdp-eyebrow">Policy</span><p>Act or request review</p><code>Choose using probability and cost</code></div></div>
    <p>The numbers above are an example, not a reported experiment. A <strong>typed answer</strong> means the response obeys an agreed format; it can still choose the wrong team. A <strong>calibrated probability</strong> describes agreement with observed frequencies across comparable cases. A <strong>decision policy</strong> uses those probabilities and the consequences of mistakes.</p>
    <DecisionExplorer />
    <h3>Your build contract</h3>
    <p>The executable core handles a single <code>choice</code> question, 2–8 described options, and up to 128 tokens. Its deliberately small tokenizer reads a–z word runs: numerals and non-Latin text are outside this toy model’s scope. It includes a tiny transformer trained from random initialization, a lexical baseline, held-out calibration, evaluation, and JSON inference. The browser investigation above runs arithmetic only; it never downloads weights or launches training.</p>
    <p>Jev motivates the structured interface. Laya provides public evidence for an encoder-and-candidate-marker approach. This project is an original small implementation of the broader idea. Exact Jev internals are not publicly established by the material reviewed here. <Reference href="https://docs.typesafe.ai/introduction">Compare Jev’s documented question types</Reference>.</p>
    <ProblemWalkthrough />
    <Readiness><p>Write the input schema, what each label means, which mistakes cost most, and what review accomplishes. Your first finish line is a reproducible small model—not a claim that it is ready for real customer traffic.</p></Readiness>
  </article>;
}

function DataStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">The unit of data is a decision, not just a sentence: a state, a question, a candidate set, and a target tied to that set.</p>
    <p>The <a href={`${assets}/request.json`} download>example request</a> supplies stable IDs such as <code>billing</code> and separate natural-language descriptions. The model reads descriptions; the application uses IDs. Reordering candidates must reorder the target index with them. Otherwise training silently rewards the wrong answer.</p>
    <Source start="def fixture_data" end="def validate_request" title="Inspect the complete constructed dataset" />
    <div className="tdp-split-flow" aria-label="Distinct roles for data splits">
      <div><b>Training · 36 cases</b><p>Learn weights. Build the vocabulary here. Shuffle candidate order during training.</p></div>
      <div><b>Calibration · 18 cases</b><p>Freeze weights. Fit the temperature using different sentence templates.</p></div>
      <div><b>Test · 18 cases</b><p>Report the fixed procedure once. Do not select a better model using these results.</p></div>
      <div><b>Stress · 3 cases</b><p>Try unseen paraphrases that remove the shared vocabulary cues.</p></div>
    </div>
    <p>These splits have distinct sentences and template families. They deliberately reuse simple task vocabulary, so they test the pipeline’s mechanics rather than broad language understanding. The stress set exposes this distinction: “money left my bank twice” has no literal match with the billing description.</p>
    <h3>Replace fixtures with real evidence</h3>
    <ol><li>Collect permissioned, de-identified cases from the actual workflow. Record provenance, source group, time, question, options and label rationale.</li><li>Write a labeling guide. Include competing requests, missing evidence and an explicit review/unknown outcome when the task needs one. Two annotators’ disagreement is evidence to inspect, not something to erase.</li><li>Split by the entity that can leak: conversation, customer, document family, or time period. Near duplicates and later conversation turns stay with their group.</li><li>Reserve a development split for model choices, an independent calibration split for temperatures, and a final test split for the locked system.</li></ol>
    <p>If multiple answers can be simultaneously correct, change the contract to a multi-label model. A softmax forces competition and cannot express two independent “yes” answers. If the correct team is missing from the candidates, normalized probabilities still sum to one; that is a reason to test candidate coverage explicitly.</p>
    <DataWalkthrough />
    <Readiness><p>Keep a data manifest with split rules and counts. Confirm no group crosses a boundary, all targets point to an existing option, and ambiguous cases have a documented resolution. A real-data extension is complete only after these checks, not after replacing a filename.</p></Readiness>
  </article>;
}

function BaselineStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">Give the transformer an honest competitor. If matching a few words solves the dataset, a successful neural run has not yet justified its complexity.</p>
    <p>Download <a href={programUrl} download>typed_decision.py</a> and <a href={`${assets}/request.json`} download>request.json</a> into one folder. Use Python 3.10 or newer with a CPU installation of PyTorch 2.6 or newer. The reference uses no network requests, pretrained checkpoints, or downloaded training data. Check the <Reference href="https://pytorch.org/get-started/locally/">official installation selector</Reference> if your platform needs a different wheel.</p>
    <Commands>{`python -m venv .venv
# Activate .venv using your shell's normal command, then:
python -m pip install "torch>=2.6,<3"
python typed_decision.py verify
python typed_decision.py train --steps 600 --seed 7 --output decision-artifact`}</Commands>
    <p>The training command writes <code>weights.pt</code>, <code>config.json</code>, <code>fixture-data.json</code> and <code>report.json</code> into the output folder. Record your installed PyTorch version and seed with the report. Different versions and hardware can change floating-point results; the downloaded <a href={`${assets}/verified-report.json`}>author’s CPU run</a> is a comparison record, not a required score to imitate.</p>
    <h3>What the baseline computes</h3>
    <p>For each candidate, count distinct words shared by the state and its description. Convert those counts to probabilities with softmax. This intentionally simple scoring rule can solve the fixture’s keyword cases. Its limitations make the next experiment clearer: paraphrases, negation and the relationship between two clauses require more than overlap.</p>
    <Source start="def lexical_logits" end="def metrics" title="Read the lexical baseline" />
    <p>The report uses <strong>accuracy</strong> for the top choice, <strong>negative log-likelihood</strong> for the probability assigned to the observed answer, and the <strong>multiclass Brier sum</strong> for squared probability error across all candidates. Smaller loss scores are better; accuracy alone cannot distinguish a cautious error from a confident one.</p>
    <BaselineWalkthrough />
    <Readiness><p>Run verification before training. Preserve the baseline even if the neural model loses. Explain which result tests plumbing, which tests generalization, and what evidence you would need before paying for a larger encoder.</p></Readiness>
  </article>;
}

function ArchitectureStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">A fixed classifier has one learned output slot per label. Here, every candidate gets a representation inside the input, and one shared scorer evaluates all of them.</p>
    <div className="tdp-token-sequence" aria-label="Input token sequence"><span className="tdp-special">CLS</span><span>question</span><span className="tdp-special">SEP</span><span className="tdp-special">OPTION</span><span>billing description</span><span className="tdp-special">SEP</span><span className="tdp-special">OPTION</span><span>access description</span><span>…</span><span className="tdp-special">SEP</span><span>state</span><span className="tdp-special">SEP</span></div>
    <p>The marked positions are not answers by themselves. Bidirectional attention lets each marker gather information from its candidate description, the question, the state and the other candidates. A single scalar head scores every marker. Softmax turns those scores into a distribution over the current options.</p>
    <div className="tdp-table-wrap" tabIndex={0} aria-label="Tensor shape table"><table><thead><tr><th>Step</th><th>Shape</th><th>Meaning</th></tr></thead><tbody><tr><td>Token IDs</td><td>B × L</td><td>Batch size and padded sequence length</td></tr><tr><td>Representations</td><td>B × L × 32</td><td>Token plus learned position embedding</td></tr><tr><td>Attention per block</td><td>B × 4 × L × L</td><td>Four heads; each query reads valid keys</td></tr><tr><td>Gathered markers</td><td>B × K × 32</td><td>K padded candidate positions</td></tr><tr><td>Scores / probabilities</td><td>B × K</td><td>Invalid candidate slots receive zero probability</td></tr></tbody></table></div>
    <Source start="class AttentionBlock" end="def negative_log_likelihood" title="Read the complete attention blocks and candidate head" />
    <h3>Build it, then control the library</h3>
    <p>The source explicitly projects Q, K and V, scales dot products by the square root of head dimension, masks padding, normalizes attention, combines values, and adds residual/feed-forward updates. Two blocks provide the encoder. PyTorch supplies tensor operations, normalization and automatic differentiation; the attention mechanism and candidate head remain visible.</p>
    <p>Set <code>--library-attention</code> to use <code>torch.nn.functional.scaled_dot_product_attention</code> with identical learned projections. The verifier compares this route against manual attention. The boolean mask here uses <code>true</code> for allowed keys; other attention APIs can use the opposite convention. <Reference href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">Read the exact SDPA contract</Reference>.</p>
    <p>Padding tokens must not contribute as keys. Padded candidate slots must not enter the normalizing denominator. Absolute positions can introduce candidate-order sensitivity; shuffling during training is useful, but invariance still needs evaluation. Full attention materializes a matrix proportional to L² per head, which is why this reference rejects inputs above 128 tokens.</p>
    <p>For comparison, Laya’s public implementation uses an existing bidirectional encoder, a question-type embedding, additional transformer layers, and scores option markers. Its auxiliary act/escalate head is separate from the main candidate scorer. Our small model uses the transparent cost policy instead. <Reference href="https://github.com/NandhaKishorM/laya/blob/main/laya/common.py">Inspect the public Laya architecture</Reference>.</p>
    <ArchitectureWalkthrough />
    <Readiness><p>Trace one request through every shape. Run a mixed batch with two and three candidates; verify the padded candidate has probability zero. Explain why this model can accept new descriptions without proving that it understands new tasks.</p></Readiness>
  </article>;
}

function TrainingStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">Start with the simplest objective that answers the research question: place probability on the labeled answer. More elaborate training should earn its place through a controlled comparison.</p>
    <p>For the correct candidate y, cross-entropy is <code>logsumexp(scores) − score[y]</code>. Subtracting the chosen score from the stable log normalizer teaches both discrimination and probability allocation. Its gradient with respect to each score is the reported probability minus the target probability.</p>
    <Source start="def negative_log_likelihood" end="def lexical_logits" title="Read stable cross-entropy from the executable source" />
    <Source start="def train" end="def predict" title="Inspect the complete training and artifact-writing loop" />
    <p>Each step shuffles candidate order and remaps the answer index. AdamW updates the weights; gradient clipping bounds the gradient norm before AdamW, not the parameter-step norm. The fixed step budget avoids choosing a checkpoint from test performance. Save the training trace, seed, data and environment alongside weights. A decreasing training loss shows optimization progress; use independent examples to investigate what was learned.</p>
    <TrainingWalkthrough />
    <ScorerTrainingLab />
    <h3>The scoring-rule research extension</h3>
    <p>A proper scoring rule rewards reporting the underlying distribution honestly in expectation. Our supervised log loss already has this foundation. For soft labels, replace the selected-label term with a target-weighted sum. You can compare it with a spherical reward, <code>q[y] / ||q||₂</code>, or their weighted combination under the same data and compute budget.</p>
    <p>For ordered categories, a ranked probability score compares cumulative probabilities at each cut point. This distinguishes being one level away from being four levels away. It requires an ordinal meaning; it does not belong on unordered billing/access/delivery categories.</p>
    <p>Only then investigate a stochastic probability-reporting policy: sample logit perturbations, score the resulting distributions, subtract a baseline, and optimize the log-probability of those samples using detached advantages. Compare this with directly differentiating the scoring objective. Record the sampling distribution, baseline, estimator and regularization rather than labeling every grouped update “PPO”. Laya’s public notebook supplies one concrete implementation to inspect. <Reference href="https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb">Read its fine-tuning procedure</Reference>.</p>
    <Readiness><p>The core deliverable is the supervised checkpoint and training trace. The reinforcement-learning comparison is a research extension, not an executed result in this guide. State the hypothesis and competing objective before running it; report failures and uncertainty as well as gains.</p></Readiness>
  </article>;
}

function CalibrationStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">A model can choose the right answer and still attach unreliable probabilities. Freeze its weights, then fit the probability transformation using separate cases.</p>
    <p>Temperature scaling divides every candidate score by the same positive number T before softmax. The reference searches a bounded logarithmic grid using calibration-set negative log-likelihood and saves the selected T. Including T = 1 gives the unchanged model a place in the comparison.</p>
    <Source start="def fit_temperature" end="def expected_cost_decision" title="Read the held-out temperature fit" />
    <p>Positive temperature preserves score ordering and therefore top-1 accuracy. It changes sharpness. A search-boundary optimum tells you to inspect the fit, not to claim that the boundary is universally correct. In the recorded 600-step CPU run, the easy calibration set selected T ≈ 0.0498 at the lower boundary. Test negative log-likelihood worsened from 2.048 to 39.975. This is a failed calibration generalization result to investigate, not a temperature to recommend for deployment.</p>
    <h3>Convert a probability into a decision</h3>
    <p>Suppose a wrong automated action costs 10 units, a correct one costs zero, and a definitive human review costs 1. If the winning probability is p, acting costs <code>10 × (1 − p)</code> in expectation. Act only when this is below 1: <code>p &gt; 0.9</code>. For p = 0.8, review is cheaper even though billing remains the most likely class.</p>
    <p>For unequal mistake costs, use a full loss matrix: multiply the loss of each possible action under each true class by that class probability, then add. Review can also have errors, delays and finite capacity; incorporate those costs when moving beyond the simplified policy.</p>
    <Source start="def expected_cost_decision" end="def permute_candidates" title="Read the explicit act-or-review policy" />
    <p className="tdp-small">The implementation sends numerically equal costs to review using a tolerance of 10⁻¹² times the larger of 1 and the two cost magnitudes. This prevents floating-point rounding from turning the exact p = 0.9 boundary into an action.</p>
    <p>Revisit the live controls in the first stage and change only cost. The probabilities stay fixed while the action changes. Change only temperature and the top label stays fixed while both expected cost and the chosen action can change. This is why classification, calibration and policy deserve separate checks.</p>
    <CalibrationWalkthrough />
    <Readiness><p>Save the calibration split identity, objective, temperature, boundary flag and policy assumptions. Report test losses before and after calibration. For real deployment, evaluate calibration by relevant language, candidate count and task groups, with uncertainty appropriate to each group’s sample size.</p></Readiness>
  </article>;
}

function EvaluationStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">Try to falsify the useful claim: this model makes better decisions on the cases that matter. A clean API and low training loss are the starting point.</p>
    <p>Open <code>report.json</code>. Compare lexical and neural accuracy, then raw and calibrated probability losses. The stress cases contain unseen wording; this small vocabulary-based model is expected to struggle there. A favorable result on the easy fixtures should not erase that finding.</p>
    <div className="tdp-table-wrap" tabIndex={0} aria-label="Recorded CPU results"><table><thead><tr><th>Recorded run · seed 7, 600 steps</th><th>Test accuracy</th><th>Test log loss</th></tr></thead><tbody><tr><td>Lexical baseline</td><td>18 / 18</td><td>0.551</td></tr><tr><td>Tiny transformer, raw</td><td>12 / 18</td><td>2.048</td></tr><tr><td>With fitted temperature</td><td>12 / 18</td><td>39.975</td></tr></tbody></table></div>
    <p>The model fits training examples but loses to the simpler baseline and becomes overconfident after calibration. An earlier 120-step run stayed near chance; the longer run was used to expose learning and its generalization failure, not to claim a benchmark win. The <a href={`${assets}/verified-report.json`}>full measured report</a> includes the training trace and stress results. Keep this unfavorable evidence as the starting research question.</p>
    <div className="tdp-table-wrap" tabIndex={0} aria-label="Research evaluation matrix"><table><thead><tr><th>Change</th><th>Keep fixed</th><th>Inspect</th></tr></thead><tbody><tr><td>Reorder options</td><td>Descriptions and correct semantic ID</td><td>Probability drift after restoring original ID order</td></tr><tr><td>Add a plausible distractor</td><td>Question, state and original candidates</td><td>Accuracy, probability dilution, and review rate</td></tr><tr><td>Paraphrase the state</td><td>Meaning and intended answer</td><td>OOV tokens, errors, and confidence in errors</td></tr><tr><td>Remove positional embeddings or one block</td><td>Splits, seeds and training budget</td><td>Quality/compute tradeoff across repeated runs</td></tr><tr><td>Switch attention implementation</td><td>Weights, inputs and masks</td><td>Numerical agreement and measured latency</td></tr><tr><td>Use a pretrained encoder</td><td>Task data and evaluation protocol</td><td>Generalization, calibration and serving cost</td></tr></tbody></table></div>
    <h3>Measure the system, not just the winner</h3>
    <p>The diagnostic program below computes a confusion matrix, reliability bins, coverage, selective risk and realized policy cost. Extend those reports with relevant subgroup comparisons and plots. Include confidence intervals or repeated splits where the design supports them. With small samples, report counts; a precise-looking percentage can obscure a single case.</p>
    <p>Measure latency after warmup with actual batch sizes, input lengths and candidate counts. Record device, dtype, thread count, library versions and whether tokenization is included. GPU timing needs synchronization. Do not compare a local toy forward pass with a remote service’s end-to-end request and call the difference an architectural speedup.</p>
    <details className="tdp-practice"><summary>Research exercise: can changing an ID change the answer?</summary><p>Keep candidate descriptions fixed and rename only their application IDs. The reference tokenizes descriptions, so probabilities should be identical and output keys should follow the new IDs. Changing a description is a different experiment because it changes the model’s evidence. Run the diagnostics tool’s ID-only check and inspect its zero logit drift, then test description paraphrases separately.</p></details>
    <EvaluationWalkthrough />
    <Readiness><p>Write a short release decision: which distribution was tested, which baseline was beaten or retained, where the model fails, and whether the expected cost meets the stated requirement. The tiny fixture project’s justified conclusion is that the mechanism runs and can be investigated; a real release requires real task evidence.</p></Readiness>
  </article>;
}

function ServingStage() {
  return <article className="tdp-stage">
    <p className="tdp-lead">The research artifact is more than weights. Ship the vocabulary, schema, calibration settings, checks and the evidence needed to interpret its output.</p>
    <Commands>{`python typed_decision.py predict --artifact decision-artifact --input request.json --wrong-cost 10 --review-cost 1`}</Commands>
    <p>This loads the saved model in evaluation mode, uses inference-only execution and library attention, and returns JSON with the chosen ID, all candidate probabilities, policy decision and estimated cost. It also accepts one request from standard input. It is a local inference program, not a deployed web service.</p>
    <Source start="def predict" end="def verify" title="Inspect artifact loading and validated JSON inference" />
    <p>The interface rejects duplicate IDs, fewer than two or more than eight candidates, missing text, text fields with no supported a–z words, and over-budget token sequences. Rejection makes lost evidence visible. If you later support truncation, report what was removed and measure the resulting errors. A tokenizer, model and calibration version must travel together.</p>
    <ServingWalkthrough />
    <h3>Turn the build into research</h3>
    <ul><li><strong>New tasks:</strong> hold out entire question families and candidate descriptions to distinguish task transfer from familiar-label classification.</li><li><strong>Ordered answers:</strong> define explicit rubric levels, train distributions over them, and inspect both expected level and tail probability.</li><li><strong>Boolean questions:</strong> use false/true candidates and return P(true); independent boolean questions need separate distributions.</li><li><strong>Selective routing:</strong> compare this explicit cost policy with a learned escalation head using observed review outcomes and capacity constraints.</li></ul>
    <p>For a service, add bounded request sizes, batching, timeouts, authentication, drift monitoring and a rollbackable artifact version. Log enough to diagnose failures while respecting data retention and privacy requirements. These operational extensions depend on the intended deployment; this guide does not claim they have been deployed.</p>
    <Readiness><p>Your final bundle contains the runnable source, environment versions, data provenance and split rules, model/configuration, calibration evidence, tests, error analysis and a decision policy. Another researcher should be able to reproduce the run, change one assumption, and explain why the output changes.</p></Readiness>
    <div className="tdp-next-reading"><span className="tdp-eyebrow">Keep building understanding</span><Link to="/learn/path/full-curriculum/typed-decision-models-calibrated-neural-decision-systems?module=large-language-models">Typed Decision Models & Calibrated Neural Decision Systems — planned depth companion →</Link></div>
  </article>;
}

export default {
  define: DefineStage,
  data: DataStage,
  baseline: BaselineStage,
  architecture: ArchitectureStage,
  training: TrainingStage,
  calibration: CalibrationStage,
  evaluation: EvaluationStage,
  serving: ServingStage,
};
