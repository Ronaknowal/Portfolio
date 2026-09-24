import { Commands, ConceptLinks, Practice, Source } from './project-elements.jsx';

const assets = '/learn-projects/typed-decision-model';

function Reference({ href, children }) {
  return <a href={href} target="_blank" rel="noreferrer">{children}<span className="tdp-sr-only"> (opens in a new tab)</span></a>;
}

function ComparisonTable({ label, headings, rows }) {
  return <div className="tdp-table-wrap" tabIndex={0} aria-label={label}>
    <table>
      <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{rows.map(([name, ...cells]) => <tr key={name}><th scope="row">{name}</th>{cells.map((cell, index) => <td key={index}>{cell}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}

export function TrainingWalkthrough() {
  return <section className="tdp-depth" aria-label="Training mechanism explained">
    <h3>Follow one error back into the model</h3>
    <p>Suppose the candidates are billing, access and delivery, in that order. A request about a refund has target index <code>0</code>. For this worked example, use scores <code>[2, 1, 0]</code>. These are chosen numbers that isolate the mathematics, not recorded outputs from the trained checkpoint.</p>
    <ol>
      <li><strong>Normalize the scores.</strong> Subtract the largest score before exponentiation: <code>exp([0, −1, −2])</code>. Divide each result by their sum. The probabilities are approximately <code>[0.665241, 0.244728, 0.090031]</code>. Adding the same constant to every score cannot change these ratios.</li>
      <li><strong>Measure the mistake.</strong> The label asks for probability on billing. Its negative log probability is <code>−log(0.665241) = 0.407606</code>. The same result is <code>logsumexp([2, 1, 0]) − 2</code>. Computing log-sum-exp directly avoids an unnecessary softmax-then-log operation that can underflow for a badly wrong answer.</li>
      <li><strong>Find a direction.</strong> The derivative with respect to the scores is <code>p − one_hot(y)</code>: <code>[−0.334759, 0.244728, 0.090031]</code>. Subtracting this gradient raises the correct score and lowers the competing scores. The entries sum to zero, consistent with the loss being insensitive to a common score shift.</li>
      <li><strong>Pass that signal through the scorer.</strong> If the normalized candidate vector is <code>hᵢ</code> and its score is <code>w · hᵢ + b</code>, then this example contributes <code>Σᵢ (pᵢ − yᵢ)hᵢ</code> to the gradient of <code>w</code>. Each candidate also sends <code>(pᵢ − yᵢ)w</code> backwards through normalization and the encoder. The shared bias adds the same offset to every valid candidate; by itself it cannot change their softmax probabilities.</li>
      <li><strong>Update the parameters that created the scores.</strong> Autograd follows the operations through the gather, both transformer blocks and the embeddings. The integer token IDs and marker indices are addresses, so they are not differentiated. The embedding vectors selected by those addresses are trainable.</li>
    </ol>
    <p>A direct gradient step on these three illustrative scores with learning rate <code>0.1</code> produces <code>[2.033476, 0.975527, −0.009003]</code>; billing rises to about <code>0.677106</code> and the loss falls to <code>0.389928</code>. The actual program updates shared weights with AdamW. Its next score change need not equal this independent-score example because one weight influences many tokens, candidates and training requests.</p>
    <p>Download <a href={`${assets}/research_tools.py`} download>research_tools.py</a> beside the core program to inspect the complete numerical path. This trace starts from seeded random weights, uses float64 for inspectable arithmetic, and takes one illustrative SGD update. It is deliberately separate from the retained 600-step AdamW experiment.</p>
    <Commands>{`python research_tools.py verify
python research_tools.py trace --output trace.json`}</Commands>
    <p>In <code>trace.json</code>, follow <code>tokens</code> and <code>markers</code> into <code>batch</code> and <code>shapes</code>, then inspect <code>one_model_update</code>. It records scores and loss before and after a whole-model update, plus the scorer’s gradient and weight-change norms. The separate <code>analytic_logit_update</code> field reproduces the three-score calculation above. A JSON <code>null</code> in a padded score slot represents masked −∞, because standard JSON cannot encode infinity.</p>
    <p>The native trace’s second row removes Delivery and also shortens the state to “refund”: 24 real tokens plus 11 padding positions. The data-stage diagram instead keeps the full request text, giving 28 real tokens plus 7 padding positions. Both use the same batching rule; only the input length differs.</p>
    <Source file="research_tools.py" start="# BEGIN trace" end="# END trace" title="Trace the actual tensors and verify one weight update" />
    <ConceptLinks items={[
      { id: 'loss-functions-ce-mse-focal-contrastive-triplet', reason: 'Derive cross-entropy, stable implementation and the difference between label indices and probability targets.' },
      { id: 'backpropagation-automatic-differentiation', reason: 'Rebuild the chain-rule mechanism behind backward(), including shared parameters and gradient checks.' },
    ]} />

    <h3>Read one training iteration as a dataflow</h3>
    <p>The core training set has 36 requests and three candidates per request. Every step uses all 36 requests, with newly shuffled candidate orders. Let <code>L</code> be the longest encoded request in that batch. Read these operations alongside <code>train</code> in the downloadable source.</p>
    <ComparisonTable label="Training iteration operations and tensor shapes" headings={['Operation', 'What flows through it', 'Why it is here']} rows={[
      ['permute_candidates', '36 decision records; each target index follows its semantic candidate.', 'Discourage a shortcut such as “the first candidate is usually correct.” Shuffling does not prove order invariance.'],
      ['batch_rows', 'IDs: 36 × L; marker positions and candidate mask: 36 × 3.', 'Pad token sequences into one tensor while keeping real positions distinguishable from padding.'],
      ['model(...)', 'IDs → 36 × L × 32 hidden states → 36 × 3 scores.', 'Compute one scalar per candidate using the same scorer. Keep scores attached to the computation graph.'],
      ['negative_log_likelihood', '36 per-request losses → one mean scalar.', 'Each request contributes equally, regardless of its token length. Padded candidate scores are excluded by −∞ masking.'],
      ['zero_grad(set_to_none=True)', 'Remove gradients left by the preceding update.', 'PyTorch accumulates into parameter gradients. Clearing once per update prevents accidental accumulation.'],
      ['loss.backward()', 'One scalar → gradients with the shape of each parameter.', 'Apply the chain rule. This computes derivatives; it does not change weights.'],
      ['clip_grad_norm_(..., 1.0)', 'All parameter gradients considered as one long vector.', 'Scale down a gradient whose global L2 norm exceeds 1; retain its direction.'],
      ['optimizer.step()', 'Parameters plus AdamW moving averages → updated parameters.', 'Use the clipped gradient, optimizer history and decoupled weight decay to choose the next weights.'],
    ]} />
    <p>The <code>train_nll</code> entry is the loss from the forward pass <em>before</em> that iteration’s update. The source logs selected steps, not every step. It calls <code>loss.item()</code> only for reporting; converting the loss to a Python number before <code>backward()</code> would break the training graph. <Reference href="https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html">PyTorch’s autograd walkthrough</Reference> explains graph construction and gradient accumulation with a smaller runnable example.</p>

    <h3>Clipping and AdamW solve different problems</h3>
    <p>If the combined gradient norm is 5, a clipping threshold of 1 multiplies its components by approximately <code>1/5</code>. It does not set each component independently to ±1. AdamW then maintains moving averages of the gradient and its square, corrects their initial bias, and scales each coordinate using the square-root second moment plus a small numerical stabilizer. Its weight decay shrinks parameters separately from that gradient normalization.</p>
    <p>Consequently, a gradient norm of at most 1 does <strong>not</strong> imply an AdamW parameter-update norm of at most 1, or at most the learning rate. Momentum, coordinate scaling and weight decay all matter. The reference uses learning rate <code>0.003</code>, weight decay <code>0.01</code>, and the library’s remaining defaults. Those are transparent settings for this fixture, not universally optimal transformer settings. It applies one parameter group; excluding biases and normalization parameters from decay would be a different, testable configuration. See the <Reference href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html">AdamW update contract</Reference> and <Reference href="https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html">global gradient-norm clipping contract</Reference>.</p>
    <ConceptLinks items={[
      { id: 'gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars', reason: 'Follow the scratch optimizer implementations before changing momentum, adaptive scaling or decoupled decay.' },
      { id: 'learning-rate-schedules-cosine-warmup-onecyclelr', reason: 'Add scheduling only with a declared update budget; a schedule counts optimizer steps, not examples.' },
    ]} />

    <h3>Understand the controls before making the model larger</h3>
    <ComparisonTable label="Model and training controls" headings={['Control in the core', 'What changing it changes', 'What to measure']} rows={[
      ['Hidden width 32; 4 heads', 'Each head receives 8 features. Width must divide evenly by head count in this implementation. Wider projections increase parameters and computation; more heads at fixed width partition the same width.', 'Development loss, parameter count, memory and latency. Head count alone does not guarantee more capacity.'],
      ['2 blocks; feed-forward width 128', 'More blocks add sequential transformations; a larger feed-forward layer changes per-token capacity. Neither creates additional labeled evidence.', 'Training versus held-out curves, errors by task family, and cost under the same experiment budget.'],
      ['Pre-normalization and residual paths', 'LayerNorm prepares inputs to each sublayer; the residual path carries the existing representation through the block.', 'Finite gradients and agreement between attention routes. Do not interpret normalization as data leakage from the test set.'],
      ['36 examples per update; 600 updates', 'The reference is full-batch over a tiny dataset. Candidate permutations change between updates even though the 36 underlying requests repeat.', 'Training fit and template-family generalization, not just the last training scalar.'],
      ['128-token maximum; no dropout', 'Longer attention grows quadratically in sequence length in the manual path. The core has no dropout to keep comparisons transparent.', 'Memory versus sequence length and evidence lost at input boundaries. Dropout is a new experiment, not an inference toggle to add blindly.'],
    ]} />
    <p>To move to larger data, use a dataset, a shuffled data loader and a collator that calls the same batching contract. A <strong>minibatch</strong> update sees only part of the training set; an <strong>epoch</strong> sees each training item once. With 36 rows and batch size 12, one epoch has three updates, so “600 steps” and “600 epochs” are different budgets. If accumulating gradients over several minibatches, divide appropriately for the intended example mean, clear once at the start, then clip and step only after the whole accumulated batch. Unequal final batch sizes need example weighting rather than an unweighted mean of batch means.</p>
    <p>Use a separate development set for selecting a learning rate, architecture or stopping rule. Keep calibration for the fitted probability transformation, and the final test for evaluation. This fixture has no development split because its recorded run uses a fixed declared training procedure. Repeatedly inspecting its test score and choosing a new architecture turns that test into development evidence.</p>
    <ConceptLinks items={[
      { id: 'batch-layer-group-rms-normalization', reason: 'See exactly which axes LayerNorm uses and why it differs from batch statistics.' },
      { id: 'cross-validation-hyperparameter-tuning', reason: 'Design development comparisons without selecting an architecture on the final test set.' },
      { id: 'bias-variance-tradeoff-learning-curves', reason: 'Interpret tiny training loss together with poor held-out performance.' },
    ]} />
    <Practice title="Practice: rebuild a stable loss and explain one update">
      <p>Implement the loss for a two-row tensor: scores <code>[[2, 1, 0], [0, 2, −∞]]</code>, targets <code>[0, 1]</code>. Use log-sum-exp and gathering, compare with <code>F.cross_entropy</code>, and confirm every gradient is finite and the padded score has zero gradient. Then exchange the first row’s first and third candidates while remapping its target to index 2.</p>
      <details><summary>Check the reasoning and result</summary><p>The per-row losses are approximately <code>0.407606</code> and <code>0.126928</code>, so the mean is <code>0.267267</code>. Each row’s score gradient is its <code>p − one_hot(y)</code> divided by 2 because the batch loss is a mean. The padded option has probability and gradient zero. Exchanging both scores and target preserves the loss. Exchanging only scores changes which answer is rewarded and is a label bug, even if training still runs.</p><p>The original <code>negative_log_likelihood</code> is the implementation to compare with after trying the exercise. On the whole model, compare manual attention and library attention from the same state dictionary before concluding that an optimizer difference is architectural.</p></details>
    </Practice>
  </section>;
}

export function CalibrationWalkthrough() {
  return <section className="tdp-depth" aria-label="Calibration and decision mechanism explained">
    <h3>Work through a temperature before fitting one</h3>
    <p>Take the same illustrative scores <code>[2, 1, 0]</code>. Temperature changes score gaps: the billing-to-delivery odds are <code>exp((2 − 0)/T)</code>. At <code>T = 1</code> they are about 7.39; at <code>T = 0.5</code> they are about 54.60. The model has not gained new evidence. We have made the existing preference more emphatic.</p>
    <ComparisonTable label="Worked temperature example" headings={['Temperature', 'Billing / access / delivery', 'Loss if billing is true', 'Loss if delivery is true']} rows={[
      ['0.5', '0.866813 / 0.117310 / 0.015876', '0.142932', '4.142932'],
      ['1', '0.665241 / 0.244728 / 0.090031', '0.407606', '2.407606'],
      ['2', '0.506480 / 0.307196 / 0.186324', '0.680270', '1.680270'],
    ]} />
    <p>Sharper probabilities improve log loss on this correct example and worsen it if the observed answer is delivery. Fitting temperature trades these effects across the <em>whole calibration set</em>. It cannot rescue a wrong ranking because dividing by a positive scalar preserves ordering. Even a well-fitted temperature is only one shared sharpness correction; it cannot independently repair different classes or input subgroups.</p>

    <h3>What the calibration code fits, step by step</h3>
    <ol>
      <li><strong>Freeze the trained model.</strong> Run the separate calibration rows with evaluation behavior and without constructing a gradient graph. Save their candidate scores and true answer indices.</li>
      <li><strong>Create candidate temperatures.</strong> The core uses <code>exp(−3 + 6i/120)</code> for integer <code>i</code> from 0 through 120, and explicitly includes 1. The endpoints are about 0.0498 and 20.0855. Equal steps in log-temperature examine multiplicative changes rather than wasting nearly all resolution on the large end.</li>
      <li><strong>Evaluate the same objective for each temperature.</strong> Divide every calibration score by that candidate temperature and average stable cross-entropy over the calibration rows. There is no retraining of the encoder.</li>
      <li><strong>Select and preserve the procedure.</strong> Choose the lowest calibration loss on that finite grid. Save both the temperature and whether the selected point is at a search boundary. The grid optimum is not a proof of the continuous optimum.</li>
      <li><strong>Measure transfer.</strong> Apply the selected temperature unchanged to test scores. Report raw and scaled loss on those same test rows, plus policy outcomes. This is where an apparently successful fit can fail.</li>
    </ol>
    <p>The scalar optimization does not need to touch model weights. For a smoother library route, optimize a scalar <code>log_T</code> with a numerical optimizer on cached calibration logits, using <code>T = exp(log_T)</code> to enforce positivity. Keep the same objective and held-out boundary, compare the result against the grid, and inspect extreme solutions. A different optimizer does not create more representative calibration data.</p>
    <p>In the retained experiment, the sharp lower-boundary temperature makes the six wrong test choices much more expensive under log loss. The evidence supports “the fitted temperature transferred badly,” not “temperature scaling never works.” Redesign calibration coverage using development evidence, then assess a newly locked procedure on genuinely unused cases. <Reference href="https://proceedings.mlr.press/v70/guo17a.html">Guo and colleagues’ calibration paper</Reference> explains the method and its empirical motivation; its results do not guarantee improvement on this small constructed distribution.</p>
    <ConceptLinks items={[
      { id: 'calibration-conformal-prediction', reason: 'Study calibration diagnostics, split roles and why temperature scaling is different from a conformal coverage guarantee.' },
      { id: 'ml-problem-formulation-baselines-data-leakage', reason: 'Decide what counts as genuinely unused evidence when repairing a failed calibration procedure.' },
    ]} />

    <h3>Probability quality is a measurable property</h3>
    <p>For the single billing example at <code>T = 1</code>, the multiclass Brier sum is <code>(0.665241 − 1)² + 0.244728² + 0.090031² ≈ 0.180061</code>. It penalizes the full distribution. The core averages this sum over requests; it does not divide by the number of classes. State that convention when comparing a library metric because “Brier score” can refer to different binary or multiclass normalizations.</p>
    <p>A reliability diagram asks a different question: among predictions assigned similar confidence, how often was the top choice correct? A bin containing two predictions cannot justify a strong statement about an 80% success frequency. Report its count, average confidence and empirical accuracy together. Good top-label calibration also does not prove that every candidate probability is calibrated conditionally on every language, candidate count or user group.</p>

    <h3>Generalize the action rule without mixing it into training</h3>
    <p>Let <code>C(a, y)</code> be the cost of taking action <code>a</code> when the true class is <code>y</code>. The expected cost of that action is <code>Σᵧ C(a, y)p(y | x)</code>. Compute this for each allowed action, including review, and choose the minimum. In the existing equal-error-cost rule, the best automatic action is the largest-probability candidate. With unequal costs it need not be.</p>
    <p>For example, suppose probabilities for billing, access and delivery are <code>[0.6, 0.3, 0.1]</code>. Routing to billing costs 10 when access was right and 2 when delivery was right, so its expected cost is <code>10×0.3 + 2×0.1 = 3.2</code>. A definitive review costing 1 wins. A real review action may itself be wrong or delayed, so its row in the cost matrix should reflect that evidence rather than assume perfection. The policy is only as useful as its probabilities and cost assumptions.</p>
    <ConceptLinks items={[
      { id: 'decision-theory-risk-cost-sensitive-decisions', reason: 'Implement full loss matrices, compare actions and understand where probabilities end and policy begins.' },
    ]} />
    <Practice title="Practice: the cheapest action can change without changing the model">
      <p>Use probabilities <code>[0.8, 0.15, 0.05]</code>, a wrong-action cost of 10, and review cost 1. Then change only the wrong-action cost to 3. Explain the action at both settings and what a temperature change could affect.</p>
      <details><summary>Check the decision</summary><p>The estimated automatic cost first is <code>10×0.2 = 2</code>, so choose review. It then becomes <code>3×0.2 = 0.6</code>, so act on billing. The model scores and probabilities have not changed; the consequences have. Temperature can change the probabilities and therefore expected cost, but cannot change the top candidate for a fixed set of finite scores and positive temperature. Never tune a cost merely to make the resulting action look more decisive.</p></details>
    </Practice>
  </section>;
}

export function EvaluationWalkthrough() {
  return <section className="tdp-depth" aria-label="Evaluation mechanisms and interpretation">
    <h3>Read the retained experiment as evidence</h3>
    <p>A three-way uniform prediction has log loss <code>log(3) ≈ 1.098612</code>. The recorded training trace stays near that value for many updates, then falls to roughly 0.000104 at step 600. That is evidence of fitting the repeated training cases. The test result of 12 correct cases out of 18, versus the lexical baseline’s 18 out of 18, does not justify replacing the baseline on this task.</p>
    <p>There is a useful difference between <strong>being wrong</strong> and <strong>being confidently wrong</strong>. Raw and temperature-scaled scores choose the same six wrong test answers, so both report 12/18. Scaling increases test NLL from about 2.048 to 39.975 because some true answers receive extremely small probability. Reporting only accuracy would hide this large deterioration in the probabilities that drive the cost policy.</p>
    <p>The three stress examples produce one correct top choice for both models. That fraction has a denominator of three, and the examples were constructed to expose vocabulary weaknesses. Treat them as named failure probes. They do not estimate a population-level success rate, and a confidence interval cannot turn them into a representative sample.</p>

    <h3>Run the implemented diagnostics on your checkpoint</h3>
    <p>Keep <a href={`${assets}/research_tools.py`} download>research_tools.py</a> beside <code>typed_decision.py</code>, then point it at the artifact produced by training. This command reads the saved vocabulary, temperature, fixture split and weights. It does not retrain the model or refit temperature on test labels.</p>
    <Commands>{`python research_tools.py evaluate --artifact decision-artifact --output diagnostics.json
# Optional, narrowly scoped CPU timing:
python research_tools.py evaluate --artifact decision-artifact --output diagnostics-with-timing.json --latency-runs 50`}</Commands>
    <p>Open <code>splits.test.raw</code> and <code>splits.test.calibrated</code> side by side. Both contain metrics, a labeled confusion matrix, five equal-width reliability bins, and a sweep of the review-cost policy. Repeat with <code>splits.stress</code>. The matrix’s rows are true IDs and its columns chosen IDs; these fixed fixture labels are billing, access and delivery.</p>
    <p>Each <code>examples</code> entry keeps the row ID, state, target and chosen semantic IDs, every candidate probability, and counts of state words and unknown state words. For example, unknown-token fraction is <code>state_unknown_word_count / state_word_count</code>; these counts exclude the question, option descriptions and inserted structural tokens. Use this trace to locate a failure before assigning it an explanation.</p>
    <p>In <code>reliability.bins</code>, empty bins have a count of zero and null averages. <code>ece</code> is the count-weighted absolute gap between bin accuracy and mean confidence; its value depends on the binning and sample. In <code>policy_sweep</code>, wrong-action cost stays 10 while review cost varies through 0, 0.25, 0.5, 1, 2, 5 and 10. These are counterfactual diagnostics on fixed cases, not permission to select a winning policy on the test set.</p>
    <p>The <code>option_order</code> section compares all six permutations of three candidates, including the original order, for every case. It reports the largest probability change after realigning semantic IDs, plus the fraction of choices that change. <code>id_renaming.max_logit_change</code> must be zero: arbitrary application identifiers never enter the encoder. The file also records the weight-file hash, so diagnostics stay attributable to a particular checkpoint.</p>
    <Source file="research_tools.py" start="# BEGIN diagnostics" end="# END diagnostics" title="Read the implemented reliability, policy and perturbation diagnostics" />
    <p>The optional latency section records ten warmups followed by the requested number of batch-size-one CPU forward passes through library attention. Its median and nearest-rank p95 exclude tokenization, model loading, JSON, network and queueing. Re-run on your machine and label that scope; it cannot answer how fast a deployed service will respond.</p>

    <h3>What the deeper checks reveal in this run</h3>
    <p>The <a href={`${assets}/verified-diagnostics.json`}>retained diagnostic report</a> records the cases, bins, costs and perturbations below. These are measured results from the fixed constructed dataset, not universal properties of decision transformers.</p>
    <p>The raw-model confusion matrix places all six mistakes in the billing column: three access requests and three delivery requests are routed there. This is more informative than “66.7% accurate,” but it still does not prove a cause. Inspect those rows, then test a specific hypothesis such as dependence on a sentence template or token position.</p>
    <p>The raw model’s 0.8–1.0 confidence bin contains 16 cases with mean confidence about 0.9885, yet only 12 of those 16 choices are correct. That is a visible mismatch between apparent certainty and observed performance. With wrong-action cost 10 and review cost 1, the downstream consequence is:</p>
    <ComparisonTable label="Measured policy consequences on eighteen constructed test cases" headings={['Policy input', 'Acted / reviewed', 'Errors among acted', 'Realized cost per request']} rows={[
      ['Raw probabilities', '15 / 3', '4 / 15', '(4 × 10 + 3 × 1) / 18 = 2.388889'],
      ['Fitted-temperature probabilities', '18 / 0', '6 / 18', '(6 × 10) / 18 = 3.333333'],
      ['Review every request', '0 / 18', 'Not applicable', '(18 × 1) / 18 = 1'],
    ]} />
    <p>Here, sharper probabilities invite more automated actions and make the system worse under its own cost assumptions. The review-everything comparison assumes the same perfect, fixed-cost review used by the policy. It would need revision for a capacity-limited or error-prone reviewer.</p>
    <p>Candidate-order diagnostics also detect probability changes as large as about 0.9989 on the test fixtures after restoring ID order. This result comes from the evaluated checkpoint, despite candidate shuffling during training. A low fraction of changed top choices can coexist with a large change in confidence, so testing only the winning ID misses an important policy failure mode.</p>

    <h3>Turn diagnostic numbers into decisions</h3>
    <ComparisonTable label="How to interpret model diagnostics" headings={['Diagnostic', 'How it is constructed', 'What an unfavorable result suggests']} rows={[
      ['Confusion matrix', 'For every row, increment the cell at [true semantic ID, chosen semantic ID]. Keep the ID order explicit.', 'Look for a systematic pair of confused teams. Inspect the actual requests before assigning a cause.'],
      ['Reliability bins', 'Group by maximum probability; compare mean confidence with the fraction of correct top choices and report each count.', 'Overconfidence means the probability-driven action rule may underestimate error cost. Sparse bins mean limited evidence.'],
      ['Coverage and selective risk', 'At a confidence threshold, coverage is acted cases / all cases; selective risk is errors / acted cases.', 'Rejecting most traffic can make selected accuracy look good while review cost dominates. Risk is undefined if no cases are acted on.'],
      ['Realized policy cost', 'For each labeled case, charge review cost if reviewed, zero for a correct action, and wrong-action cost for a wrong action. Average over all cases.', 'Compare to review-everything and the baseline under the same costs, not just to the model’s own estimated cost.'],
      ['Candidate-order drift', 'Permute candidates and restore resulting probabilities to their original semantic IDs before subtracting.', 'Large drift reveals a representation/position sensitivity. A probability-vector difference in raw slot order is not this test.'],
      ['Unknown-token rate', 'Count tokens mapped to <unk>, using the saved training vocabulary. Keep denominator and special-token treatment explicit.', 'A sentence can be accepted by validation yet lose informative words in tokenization. Inspect which words were lost.'],
    ]} />
    <p>For a small policy example, suppose the three largest probabilities are <code>[0.95, 0.80, 0.60]</code> and the corresponding top choices are <code>[correct, wrong, correct]</code>. At threshold 0.9, one case is acted on: coverage 1/3 and selective risk 0/1. At threshold 0.7, two are acted on: coverage 2/3 and selective risk 1/2. These are descriptive results for these three cases. They do not establish that raising a threshold will always improve realized risk on every finite sample.</p>
    <p>Keep semantic IDs and row IDs in every diagnostic output. If the candidate set differs across examples, a matrix of fixed slot numbers mixes unrelated meanings. If the largest probabilities tie, record the tie policy: this core selects the first largest value, so exact ties can legitimately change the chosen ID after reordering even when the aligned probability distribution is unchanged.</p>
    <ConceptLinks items={[
      { id: 'evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae', reason: 'Choose metrics for the decision being made and verify averaging, denominators and undefined cases.' },
      { id: 'end-to-end-supervised-learning-error-analysis', reason: 'Convert individual failures into controlled changes without repeatedly consuming the final test.' },
      { id: 'hypothesis-testing-confidence-intervals', reason: 'Understand paired comparisons and uncertainty when a future evaluation uses an appropriate sample.' },
    ]} />

    <h3>Distinguish an invariant from a research hypothesis</h3>
    <p>Adding padding or switching manual attention to its equivalent library operation should preserve the same model output within numerical tolerance. That is an implementation contract. Reordering candidates, paraphrasing a description or inserting a distractor is a stronger behavioral hypothesis. The model is not mathematically guaranteed to satisfy it.</p>
    <p>When adding a distractor to the real encoder, it is not enough to explain a changed result by the softmax denominator. The new description participates in attention and shifts token positions, so the original candidate scores can change too. The first-stage lab deliberately holds those scores fixed to isolate normalization. Compare that controlled arithmetic with the complete-model diagnostic to identify which mechanism changed.</p>
    <p>For an architectural ablation, train each variant from initialization under a declared comparable budget. Removing a trained block only at evaluation asks about damage to a trained network, not how a shallower model would have learned. Record which budget is equalized—updates, examples processed or wall-clock compute—and keep split identities and seed handling explicit. One budget cannot automatically equalize the others.</p>
    <Practice title="Practice: design an experiment that can falsify a shortcut">
      <p>Your model routes “refund” correctly but fails “money left my bank twice.” Choose between adding more copies of the original sentence, holding out whole paraphrase families, and selecting the best seed by test accuracy. State the useful experiment, its control and the evidence you would keep.</p>
      <details><summary>A defensible experiment</summary><p>Hold out semantic paraphrase families or collect representative independent cases with different wording. Keep a keyword baseline, the label contract and a fixed development protocol. Inspect unknown-token rates and case-level errors; distinguish missing vocabulary from failures to combine known words. More copies of the same sentence may strengthen the shortcut. Selecting seeds on test accuracy leaks selection into the reported result. Report all prespecified runs, counts and paired differences; do not discard a seed because it loses.</p></details>
    </Practice>
  </section>;
}

export function ServingWalkthrough() {
  return <section className="tdp-depth" aria-label="Packaging, inference and library implementation explained">
    <h3>Follow one request through the saved artifact</h3>
    <ol>
      <li><strong>Recreate the architecture.</strong> Read <code>config.json</code>, construct the tiny model using its vocabulary size, and load the matching state dictionary. Weights alone do not contain the Python forward-pass definition. This core’s architecture name identifies its fixed two-block, width-32 contract.</li>
      <li><strong>Use the saved vocabulary.</strong> Map the request with the training vocabulary, including unknown words. Never rebuild a vocabulary from the new request: the same integer could then point to a different embedding meaning.</li>
      <li><strong>Prepare one batch.</strong> Validation, encoding and batching produce IDs, marker positions and candidate masks with batch size 1. Inference uses the same ordering and input contract as training.</li>
      <li><strong>Compute scores without training work.</strong> <code>model.eval()</code> sets evaluation behavior for layers such as dropout if present. <code>torch.inference_mode()</code> disables gradient recording and associated overhead. They have different jobs; neither replaces the other. The tiny model itself has no dropout.</li>
      <li><strong>Apply calibration and policy.</strong> Divide scores by the saved temperature, compute a distribution over valid candidates, and evaluate the explicit costs. A top choice still exists when the policy returns <code>review</code>; it is the model’s suggestion, not permission to execute it.</li>
      <li><strong>Restore the interface meaning.</strong> Zip each probability with the corresponding application ID, not with a hard-coded class name. Return probabilities, top choice, policy decision, costs and the artifact’s architecture identifier.</li>
    </ol>
    <p>The current checkpoint is an <strong>inference artifact</strong>. It does not save AdamW moments or random-generator state, so loading it is not an exact continuation of the previous training trajectory. To support resumable training, save the optimizer state, step, data ordering/RNG state and any scheduler or mixed-precision scaler as well. A service should additionally use a unique version or content hash; the architecture string alone cannot distinguish two sets of weights.</p>
    <ComparisonTable label="Artifact responsibilities" headings={['File', 'What it preserves', 'What it does not establish']} rows={[
      ['typed_decision.py', 'The executable input, forward-pass, training, calibration and policy definitions.', 'A source file alone does not show that a particular run passed.'],
      ['weights.pt', 'The learned tensor values keyed by parameter name.', 'The tokenizer, cost assumptions, dataset provenance or training optimizer state.'],
      ['config.json', 'Vocabulary, fitted temperature, architecture identifier, token budget, question type and seed.', 'That the fitted temperature improves future data or that this seed is superior.'],
      ['fixture-data.json and report.json', 'The constructed examples and selected measured outputs of this run.', 'Real-customer representativeness or a guarantee that the same scores recur on every library/hardware version.'],
    ]} />

    <h3>The library bridge preserves the mechanism</h3>
    <p>A pretrained encoder supplies learned language representations. It does not automatically supply our variable-candidate decision head. Retain the semantic request contract and the marker-gather/scorer mechanism, while replacing the tokenizer and encoder together. In the library route, the encoder returns <code>last_hidden_state</code> with shape <code>B × L × D</code>; <code>D</code> comes from its configuration, not from the tiny model’s constant 32.</p>
    <ComparisonTable label="Scratch to Transformers implementation correspondence" headings={['Tiny implementation', 'Ordinary library route', 'Contract to preserve']} rows={[
      ['ASCII word lookup plus learned token and position embeddings', 'A checkpoint-matched AutoTokenizer and AutoModel.', 'Token IDs, padding IDs and embedding rows must belong to the same tokenizer/model pair.'],
      ['Explicit <option> positions from encode()', 'Register a dedicated special marker, then record its positions in actual token IDs.', 'One unambiguous marker per option; reject user text that would inject structural markers.'],
      ['Two explicit attention blocks', 'The pretrained bidirectional encoder produces last_hidden_state.', 'Pass a correct attention mask and stay within the checked sequence budget.'],
      ['gather(...), LayerNorm and Linear(D, 1)', 'The same shared candidate scorer on gathered encoder states.', 'One score per supplied candidate rather than a fixed num_labels classifier.'],
      ['Scratch log-sum-exp cross-entropy', 'The same loss or equivalent F.cross_entropy on raw candidate logits.', 'Mask invalid candidate slots; never pass already normalized probabilities as logits.'],
      ['Tiny vocabulary and weights saved together', 'Save the tokenizer, encoder and custom head with compatible metadata.', 'Reload the exact marker ID, head width, calibration and request formatting.'],
    ]} />
    <p>When adding a special token, resize the encoder’s token embeddings <em>before</em> constructing the optimizer. The new row starts without task knowledge and needs training. If you freeze every encoder parameter, you also freeze that new row; calling a head-only run “learning the marker embedding” would be incorrect. Full fine-tuning exposes more capacity but needs a suitable learning rate, development protocol and memory budget. Freezing, gradual unfreezing and parameter-efficient adaptation answer different experimental questions.</p>
    <p>Do not put an ordinary string separator into the text and assume it occupies one tokenizer position. A subword tokenizer may split it. Do not silently truncate the end of the request either: the state or a candidate can disappear while the JSON still looks valid. Construct the token sequence, locate markers, verify counts and reject over-budget input before the forward pass. These checks connect the high-level interface to the actual tensors the model reads.</p>

    <h3>Run the library adapter, then inspect its four boundaries</h3>
    <p>Download <a href={`${assets}/pretrained_decision.py`} download>pretrained_decision.py</a> beside <code>typed_decision.py</code>. It provides a real Transformers implementation, with separate batch construction, encoder adaptation, training, and artifact loading. The adapter intentionally supports a BERT-style bidirectional encoder with CLS, SEP and PAD token IDs, retains a 128-token project budget, and rejects decoder/encoder-decoder configurations. It is not a universal adapter for every <code>AutoModel</code> checkpoint.</p>
    <Commands>{`python -m pip install "torch>=2.6,<3" "transformers>=4.48,<5"
python pretrained_decision.py verify --output adapter-checks.json`}</Commands>
    <p>The verifier builds a tiny randomly initialized BERT through the actual Transformers API and a local tokenizer. It exercises marker registration, embedding resizing, padded batches, gradients, one optimizer step, a saved-and-reloaded encoder/tokenizer/head, and frozen-encoder behavior. It requires no downloaded weights. This establishes those implementation checks in the recorded environment; it is not evidence that a pretrained ModernBERT has been fine-tuned or evaluated for routing.</p>
    <Source file="pretrained_decision.py" start="# BEGIN batch" end="# END batch" title="Library route: construct tokens, marker positions and masks" />
    <p><code>encode_request</code> builds the sequence from token IDs. It inserts the registered <code>&lt;decision_option&gt;</code> ID and records the sequence length immediately before each insertion. <code>batch_requests</code> right-pads with the tokenizer’s own pad ID, builds a token attention mask and a separate candidate mask. These masks describe different things: tokens the encoder can read, and options the final softmax may consider.</p>
    <Source file="pretrained_decision.py" start="# BEGIN adapter" end="# END adapter" title="Library route: load the encoder and gather candidate representations" />
    <p><code>load_backbone</code> loads the tokenizer and encoder from the same checkpoint/revision, registers the marker, and then constructs the decision model. <code>EncoderDecisionModel.forward</code> takes the encoder’s final hidden states, expands the marker-position tensor across the hidden dimension, gathers candidate vectors, and applies the shared scorer. In a frozen-encoder run, its <code>train()</code> override keeps the encoder in evaluation mode while the head trains, so dropout does not randomly change the frozen features.</p>
    <Source file="pretrained_decision.py" start="# BEGIN train" end="# END train" title="Library route: train, calibrate and evaluate with the shared fixture" />
    <p>The adapter trains on the same constructed decision rows and candidate permutations. It uses <code>F.cross_entropy</code> for the already-verified loss, clips trainable gradients, and fits temperature only after training. Its AdamW parameter groups apply decay to matrices and omit one-dimensional bias/normalization parameters. This is an explicit difference from the tiny core’s one-group setup; do not attribute a changed result solely to pretraining when several settings changed.</p>
    <p>The command below runs a <strong>new optional experiment</strong> using a pretrained checkpoint. It can download a substantially larger model; the project’s recorded checks do not include that training run. By default the adapter requires locally available model files. The explicit <code>--allow-download</code> flag enables Hub access for this example.</p>
    <Commands>{`python pretrained_decision.py train --checkpoint answerdotai/ModernBERT-base --allow-download --steps 60 --learning-rate 0.00002 --output pretrained-artifact
python pretrained_decision.py predict --artifact pretrained-artifact --input request.json --wrong-cost 10 --review-cost 1`}</Commands>
    <p>For an already downloaded checkpoint, replace the checkpoint argument with its local directory and omit <code>--allow-download</code>. For a reproducible Hub experiment, set <code>--revision</code> to the reviewed immutable commit; the artifact records the requested revision and resolved commit where available. Sixty steps is a smoke-experiment budget, not a convergence recommendation. This script uses CPU float32 and full-batch fixtures to keep the execution path inspectable. A GPU/minibatch extension must move the encoder, head, input tensors and targets together and verify numerical agreement before adding mixed precision or different attention kernels.</p>
    <p>Compare full fine-tuning with <code>--freeze-encoder</code> under a declared protocol. In the latter, the new marker embedding is frozen too; only the shared scorer adapts. A useful next variant trains the marker embedding plus the head, but that requires an explicit selective-gradient implementation rather than assuming the freeze flag does it. Replace fixture data only after creating a real-data manifest and independent development/calibration/test roles.</p>
    <Source file="pretrained_decision.py" start="# BEGIN save_load" end="# END save_load" title="Library route: save and reload all parts of the decision model" />
    <p>The output has an <code>encoder/</code> directory, <code>tokenizer/</code> directory, <code>scorer.pt</code>, <code>decision-config.json</code> and a measured <code>report.json</code>. Loading validates the artifact format, marker identity, positive temperature and vocabulary/embedding size agreement. Keeping only the encoder would discard the learned decision head; keeping only the head would lose the representations it was trained to read.</p>
    <ConceptLinks items={[
      { id: 'transfer-learning-fine-tuning-strategies', reason: 'Choose trainable parameter groups, preserve optimizer state semantics and compare adaptation strategies.' },
      { id: 'byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram', reason: 'Understand why tokenizer positions and special-token registration must be checked rather than guessed from characters.' },
      { id: 'contextual-embeddings-elmo-bert-variants', reason: 'This planned topic owns deeper contextual-encoder comparisons; the local explanation above is sufficient for this build.' },
    ]} />
    <p><Reference href="https://huggingface.co/docs/transformers/model_doc/modernbert">ModernBERT’s official model documentation</Reference> specifies its encoder outputs and mask interface. <Reference href="https://huggingface.co/docs/transformers/main_classes/tokenizer">The tokenizer API documentation</Reference> specifies token registration and embedding resizing. Use the tested adapter’s declared dependency range and record exact installed versions; a generic API description is not evidence that an arbitrary model checkpoint is compatible.</p>

    <h3>Make the next experiment reproducible</h3>
    <p>Store the source/checkpoint revision, tokenization format, split manifest, training options, calibration procedure and cost assumptions alongside the output. Keep full per-case results for error analysis, not only an aggregate score. If the deployment receives longer requests, additional languages or more candidate types, evaluate those regimes explicitly rather than assuming pretrained language knowledge covers the entire decision contract.</p>
    <p>For a local service, load one immutable artifact per worker and reuse it across requests; do not rebuild the encoder for every call. Bound queue length and request size, batch only under a latency budget, and measure tokenization plus model plus policy time. Treat review as a first-class response the application must honor. Monitor input/score changes, but use subsequently available labels to measure actual errors and calibration drift. A shifted input histogram alone cannot tell you whether accuracy improved or deteriorated.</p>
    <ConceptLinks items={[
      { id: 'model-serving-api-frameworks', reason: 'Planned continuation for request schemas, batching, timeouts and deployment interfaces; the current project provides local JSON inference.' },
      { id: 'model-monitoring-drift-detection', reason: 'Planned continuation for distribution shift, delayed labels and release monitoring.' },
    ]} />
    <Practice title="Practice: prove that the exported model means the same thing">
      <p>Save and reload a trained artifact in a fresh process. Run the same request before and after reload, then rename only its candidate IDs. Identify which fields should match numerically and which should change.</p>
      <details><summary>Acceptance criteria</summary><p>Before and after reload, compare logits and probabilities within a stated floating-point tolerance using the same inference route and configuration. Candidate-ID renaming preserves these values because IDs are not encoded; the output keys and winning ID follow the renaming. Policy decisions also remain unchanged for the same costs, except that numerical boundary cases require the documented tolerance. Candidate-description edits are a different test because they alter input evidence.</p><p>Next change only temperature and confirm the logits remain identical while probabilities can change. Finally change only costs and confirm both logits and probabilities remain identical while the action can change. These three separations make future failures easier to locate.</p></details>
    </Practice>
  </section>;
}
