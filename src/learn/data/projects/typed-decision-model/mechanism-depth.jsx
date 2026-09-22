import { Commands, ConceptLinks, Practice, Source, projectAssets } from './project-elements.jsx';
import { AttentionLab, EncodingLab } from './mechanism-labs.jsx';

export function ProblemWalkthrough() {
  return <>
    <h3>What exactly are we learning to build?</h3>
    <p>A language model that writes an answer chooses a sequence of tokens. Our system makes a different, smaller decision: given a request and descriptions of the permitted answers, assign one score to each answer. The application keeps the IDs and turns the resulting distribution into an action. Descriptions are inputs, so the last layer does not need a permanently assigned “billing output neuron.” This is useful when options differ between requests, but changing the options changes the statistical problem too.</p>
    <p>Keep three objects separate. The <strong>state</strong> is the evidence, such as “please help with my refund.” The <strong>question</strong> specifies the decision, such as which team should handle it. The <strong>candidate set</strong> says which answers are available now. If “refund approval” rather than “routing” becomes the question, the same evidence needs a different labeling policy and often different training data. A flexible input schema alone does not teach a new task.</p>
    <div className="tdp-table-wrap" tabIndex={0} aria-label="Three contracts to test separately"><table><thead><tr><th>Contract</th><th>Example test</th><th>What passing establishes</th></tr></thead><tbody>
      <tr><td>Representation</td><td>Every output ID is among the supplied unique candidates; probabilities sum to one.</td><td>The program obeys the format.</td></tr>
      <tr><td>Statistical</td><td>On held-out requests, high-confidence decisions are usually correct at the stated rate.</td><td>Evidence about this distribution, with sampling uncertainty.</td></tr>
      <tr><td>Operational</td><td>The cost of errors plus review meets the workflow’s requirements.</td><td>Whether using the model is worthwhile under those assumptions.</td></tr>
    </tbody></table></div>
    <p>A valid distribution can be wrong with probability 0.99 attached to it. Conversely, a modest probability can support an action when mistakes are inexpensive. The project therefore implements a scorer, a probability transformation and a decision policy as separate pieces. You can inspect or replace one without pretending that the others have been validated automatically.</p>
    <h3>The route from a request to a research artifact</h3>
    <ol className="tdp-reading-route"><li><b>Define and collect:</b> establish labels, costs and split boundaries before choosing the architecture.</li><li><b>Represent and compare:</b> turn text into token IDs and establish a lexical baseline.</li><li><b>Encode and score:</b> contextualize token vectors, gather option markers and share one scoring head.</li><li><b>Learn and calibrate:</b> fit weights on training examples, then fit temperature on separate evidence.</li><li><b>Investigate and package:</b> inspect errors, changed candidate sets and measured costs; preserve everything inference needs.</li></ol>
    <p>Build the small program first so every tensor and file has an understandable job. Then use the supplied Transformers adapter to learn the ordinary pretrained-encoder integration. Both routes share the candidate-scoring idea; they do not have identical tokenizers, architectures or learned knowledge. The code and explanation here stand on their own; deeper links unpack reusable foundations without forcing you to leave in the middle of a missing project step.</p>
    <ConceptLinks items={[
      { id: 'ml-problem-formulation-baselines-data-leakage', reason: 'Define the target, comparison baseline and split unit before optimizing a model.' },
      { id: 'decision-theory-risk-cost-sensitive-decisions', reason: 'Derive why the most likely answer and the best action can differ.' },
    ]} />
    <Practice title="Practice: a request fits two teams — worked reasoning">
      <p>“I was charged, but my parcel never arrived” can legitimately involve two teams. If the contract asks for the first team to investigate, supply a labeling rule that resolves the priority. If the contract asks for every relevant team, a single-choice softmax is the wrong output contract: use independent labels or a multi-stage workflow. Do not silently label such requests with whichever word the baseline matches first. The representation, target and evaluation must change together.</p>
    </Practice>
  </>;
}

export function DataWalkthrough() {
  return <>
    <h3>Follow the refund request all the way to a batch</h3>
    <p>In Python, a request is a dictionary. <code>target = 0</code> means the first option in this particular row is correct, not that the number zero intrinsically means Billing. The stable application ID is <code>billing</code>. A training row contains the target; an inference request does not need one.</p>
    <Commands label="Worked request with a training-only target">{`{
  "question": "Which team should handle this request?",
  "state": "please help with my refund",
  "options": [
    {"id": "billing", "description": "payment invoice charge refund billing"},
    {"id": "access", "description": "login password account access locked"},
    {"id": "delivery", "description": "shipping delivery parcel tracking delayed"}
  ],
  "target": 0
}`}</Commands>
    <p><code>make_vocabulary</code> gathers words from training questions, states and candidate descriptions, then sorts them for a deterministic mapping. It reserves IDs 0–4 for PAD, UNK, CLS, SEP and OPTION. An ID is a lookup address, not an amount: token 40 is not “twice as meaningful” as token 20. Learning assigns a vector to each address. Fitting the vocabulary on evaluation text would change what the system knows before the experiment begins.</p>
    <Source start="def make_vocabulary" end="class AttentionBlock" title="Read vocabulary construction, encoding and batch assembly" />
    <p><code>encode</code> first validates the request, lowercases text and extracts a–z words. It places CLS at the beginning, separates fields with SEP and records the position immediately before adding each OPTION token. It places the state after the candidate descriptions. Because attention is bidirectional, a marker can read that later state. There is no causal next-token mask.</p>
    <EncodingLab />
    <p>At the default input, the six question words occupy positions 1–6. Position 7 is SEP; the three OPTION markers are at 8, 15 and 22. The state occupies 29–33, and the final SEP is 34. The resulting length is 35. These are sequence positions, distinct from vocabulary IDs. The source uses <code>len(ids)</code> rather than hard-coded offsets so a longer description moves all later markers correctly.</p>
    <h3>Two masks, two different problems</h3>
    <p>A batch makes several differently sized examples rectangular. Remove Delivery from a second copy of this request: its sequence has 28 tokens and two candidates. The first copy still has 35 tokens and three candidates. We pad the shorter row to the longest length in this batch and separately pad its candidate-position list.</p>
    <div className="tdp-batch-diagram" aria-label="Batch padding and candidate masking">
      <div><b>Token IDs · shape 2 × 35</b><p>Row A: 35 real tokens</p><div className="tdp-batch-lane"><span style={{ flex: 35 }}>35 valid</span></div><p>Row B: 28 real tokens + 7 PAD</p><div className="tdp-batch-lane"><span style={{ flex: 28 }}>28 valid</span><span className="tdp-pad-region" style={{ flex: 7 }}>7 PAD</span></div></div>
      <div><b>Candidate positions · shape 2 × 3</b><p><code>A: [8, 15, 22] · [true, true, true]</code></p><p><code>B: [8, 15, 0] · [true, true, false]</code></p><p className="tdp-small">The last 0 is a storage placeholder, not a third answer.</p></div>
    </div>
    <p>The <strong>token mask</strong>, computed as <code>ids != 0</code>, excludes padding keys from attention. The <strong>candidate mask</strong> replaces a nonexistent candidate’s final score with negative infinity, so softmax assigns it exactly zero probability. Masking tokens alone cannot remove a fake output option. Masking candidates alone cannot stop attention reading padding.</p>
    <p>The filler position 0 temporarily gathers the CLS vector for a missing candidate; the later candidate mask removes that score from the objective. Padding query vectors may themselves be nonzero because position embeddings and projections still operate on them. That is harmless here because they never become allowed keys or valid candidate outputs in later blocks. Keep that reasoning explicit when adapting the architecture.</p>
    <Source start="def validate_request" end="def make_vocabulary" title="Inspect the input contract before tensors are allocated" />
    <p>The deliberately small tokenizer exposes a limitation worth seeing. Unknown words all become UNK: “money” and “bank” can lose their separate identities. A numerals-only or non-Latin field is rejected; numerals embedded in otherwise supported text are still dropped. This representation cannot safely distinguish invoice numbers or amounts. A subword tokenizer is a substantial change to the inputs and requires retraining, not a drop-in replacement for the saved vocabulary.</p>
    <ConceptLinks items={[
      { id: 'byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram', reason: 'Understand how subword tokenization avoids this word-vocabulary failure and why marker IDs depend on the tokenizer.' },
      { id: 'vectors-matrices-tensor-operations', reason: 'Read batch, sequence and feature axes correctly before reshaping or gathering them.' },
      { id: 'cross-validation-hyperparameter-tuning', reason: 'Separate model selection from the calibration and final-test roles used here.' },
    ]} />
    <Practice title="Practice: repair a shuffled target — solution and failure signature">
      <p>Reorder [billing, access, delivery] to [delivery, billing, access]. The old target was 0; the new target must be 1. Keep a semantic ID in your dataset and derive its current index after every permutation. If you forget this, a model can reduce training loss by learning the wrong associations. A tensor-shape check will still pass, so verify ID-to-index correspondence as a separate invariant.</p>
    </Practice>
  </>;
}

export function BaselineWalkthrough() {
  return <>
    <h3>Make the reference run reproducible</h3>
    <p>Keep the project files together. <code>typed_decision.py</code> is the canonical small model and owns its tokenizer, dataset, attention, loss and saved-artifact format. <a href={`${projectAssets}/research_tools.py`} download>research_tools.py</a> imports those implementations for tracing and diagnostics. <a href={`${projectAssets}/pretrained_decision.py`} download>pretrained_decision.py</a> adds the ordinary Transformers route. The latter two reuse the first file; renaming it breaks their imports.</p>
    <Commands label="Project folder and generated artifacts">{`decision-project/
  typed_decision.py        # canonical small model
  research_tools.py        # trace and inspect the same implementation
  pretrained_decision.py  # optional Transformers encoder route
  request.json            # inference input
  decision-artifact/      # produced by the core train command
    weights.pt
    config.json
    fixture-data.json
    report.json`}</Commands>
    <p>Use the virtual environment’s interpreter for installation and execution. On Windows it is <code>.venv\\Scripts\\python.exe</code>; on macOS/Linux it is <code>.venv/bin/python</code>. Activating the environment makes <code>python</code> resolve to that interpreter. If importing torch fails after installation, check <code>python -c "import sys; print(sys.executable)"</code> before reinstalling packages into another environment.</p>
    <p><code>verify</code> exercises mathematical invariants before an expensive run. A passing result establishes those contracts on the supplied fixtures; it does not establish task accuracy. Training then writes a fresh output directory. Use a different directory for each changed seed or objective so comparison evidence is not overwritten. Record command, source revision, package versions and data identity beside each report.</p>
    <h3>Calculate the baseline once by hand</h3>
    <p>For “please help with my refund,” only Billing’s description shares the distinct word <code>refund</code>. The overlap scores are [1, 0, 0]. Softmax gives approximately [0.5761, 0.2119, 0.2119]. The selected answer is Billing; its negative log-likelihood is −log(0.5761) ≈ 0.5514. A correct label does not make the log loss zero: the score still assigned probability to alternatives.</p>
    <p>The multiclass Brier sum for this row is (0.5761 − 1)² + 0.2119² + 0.2119² ≈ 0.2695. This project sums across classes, then averages across rows; another library may use a different normalization. State the convention before comparing numbers. With no matching words, [0, 0, 0] gives a uniform distribution. The code’s first-index tie break is deterministic bookkeeping, not evidence that the first team is preferable.</p>
    <p>Here the baseline intentionally ignores the question and word order. “Do not send this billing question to billing” can score well by overlap while having the wrong meaning. This makes it a useful control: a neural model must demonstrate a benefit on such cases rather than receiving credit merely for being more complex. The shared keyword vocabulary explains why the simple baseline wins the recorded easy test.</p>
    <Source start="def metrics" end="def fit_temperature" title="Inspect exactly how accuracy, log loss and Brier sum are measured" />
    <ConceptLinks items={[
      { id: 'testing-debugging-dependency-management', reason: 'Use executable invariants and isolated dependencies to distinguish environment failures from model defects.' },
      { id: 'reproducible-notebooks-experiment-structure', reason: 'Keep source, configuration, data identity and outputs together across controlled experiments.' },
      { id: 'evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae', reason: 'Choose metrics that answer the task question and inspect class-specific errors instead of one headline score.' },
    ]} />
    <Practice title="Practice: does repeating refund improve this baseline?">
      <p>No. The implementation intersects sets of words, so “refund refund refund” has the same overlap count as “refund.” The neural tokenizer retains repetitions and changes sequence length and positions. This gives a concrete experiment where the two representations differ even when they choose the same top label.</p>
    </Practice>
  </>;
}

export function ArchitectureWalkthrough() {
  return <>
    <h3>Read the forward pass as a sequence of transformations</h3>
    <p>The embedding lookup turns each integer into a learned vector of 32 numbers. Adding a learned position vector lets equal words at different positions be represented differently. At the start of training these vectors are random; an embedding table does not arrive with an understanding of refunds. Token embeddings are shared across occurrences, while position embeddings are shared across examples at the same sequence offset.</p>
    <ol className="tdp-reading-route"><li><b>Normalize each token’s features.</b> The block uses LayerNorm before attention. It normalizes across the 32 features of a token, not across examples, and includes learned scale and shift. This is a pre-normalized residual block.</li>
      <li><b>Project three views.</b> One linear map produces 96 numbers per token, split into query, key and value. Four heads each receive eight coordinates. Queries describe what a position seeks, keys determine compatibility, and values are the information to mix. These are learned mathematical roles rather than literal language questions.</li>
      <li><b>Read valid positions.</b> Compute QKᵀ / √8 for each head, mask invalid keys, then normalize each query’s row over keys. Each row’s weights sum to one. Multiply by V to get a weighted mixture.</li>
      <li><b>Combine heads and preserve a residual route.</b> Restore the 32-feature axis, apply the output projection and add the incoming hidden vector. A second LayerNorm and 32 → 128 → 32 feed-forward network with GELU produce another residual update. The feed-forward network processes each token independently; attention is the cross-token communication.</li>
      <li><b>Repeat and gather.</b> After two blocks, read only each OPTION position’s contextual vector and apply the same LayerNorm/linear scalar scorer to every candidate. Mask nonexistent candidates, then normalize the remaining scores.</li></ol>
    <ConceptLinks items={[
      { id: 'self-attention-multi-head-attention', reason: 'Derive Q/K/V, scaled compatibility and why each query normalizes over keys.' },
      { id: 'batch-layer-group-rms-normalization', reason: 'Understand which axes LayerNorm normalizes and why pre-normalization changes gradient flow.' },
      { id: 'transformer-block-architecture', reason: 'Connect residual paths, attention and token-wise feed-forward updates into a complete block.' },
    ]} />
    <AttentionLab />
    <h3>Why these reshapes are necessary</h3>
    <p>For the two-row batch from the data stage, the combined projection has shape [2, 35, 96]. <code>view(2, 35, 3, 4, 8)</code> exposes Q/K/V, head and head-feature axes without changing the number of elements. <code>permute(2, 0, 3, 1, 4)</code> reorders those axes to [3, 2, 4, 35, 8], and <code>unbind(0)</code> yields three tensors of shape [2, 4, 35, 8]. Mixing up a sequence and head axis can produce a legal tensor operation with a completely wrong mechanism.</p>
    <p>The token mask starts as [2, 35]. Adding singleton axes gives [2, 1, 1, 35], which broadcasts the same allowed-key decision across four heads and every query. It does not impose causal order. After attention, transpose [2, 4, 35, 8] to [2, 35, 4, 8] and reshape to [2, 35, 32]. Using <code>reshape</code> after the transpose permits a needed contiguous copy; assuming that every transposed tensor can be viewed without copying is unsafe.</p>
    <h3>A candidate gets a contextual vector, not a fixed class weight</h3>
    <p>The gather index has shape [B, K, 32]. It repeats each marker’s sequence position for all 32 feature coordinates; <code>hidden.gather(1, index)</code> reads along the sequence axis. For Billing at position 8, it copies <code>hidden[b, 8, :]</code>. The shared head maps that 32-vector to one scalar. It then applies the same parameters to Access and Delivery. The descriptions, state and positions made those vectors different; the final head does not own three separate label slots.</p>
    <p><code>expand</code> creates a broadcasted index view instead of materializing 32 independent copies of every position. <code>gather</code> is differentiable with respect to the hidden values, but not the integer positions. During backpropagation, selected marker vectors receive gradients through the shared head; attention carries those gradients back to other relevant token representations and shared parameters. The <a href="https://docs.pytorch.org/docs/2.14/generated/torch.gather.html" target="_blank" rel="noreferrer">PyTorch gather contract</a> spells out its index/output shape requirements.</p>
    <p>The scalar head’s final bias is shared across all candidates. Adding that common constant changes neither softmax nor the loss, so its discriminative gradient cancels. The useful discrimination comes from different contextual candidate vectors and the shared projection. This is a useful check on the computation, not a reason to claim all trained parameters have equal influence.</p>
    <h3>Control implementation cost without hiding the mechanism</h3>
    <p>For each block, manual attention forms B × H × L² scores; doubling sequence length roughly quadruples that score storage. The projections and feed-forward layers also cost work, so a short-input runtime cannot be inferred from the L² term alone. We batch matrix multiplications rather than looping over token pairs in Python. The SDPA route can use more efficient kernels, depending on device, dtype and masks, while preserving the intended operation within numerical tolerance.</p>
    <p>The reference uses four heads because 32 divides into four groups of eight; it is an inspectable CPU-sized choice, not an experimentally proven optimum. Two blocks allow a second round of contextual interaction. Changing width also changes projection, embedding and scorer shapes; changing heads requires divisibility; changing maximum length changes the position table. Record these in a new architecture/configuration and rerun shape/parity checks before interpreting quality results.</p>
    <ConceptLinks items={[
      { id: 'tensor-algebra-einsum-notation', reason: 'Track contractions, singleton broadcasting and axis permutations in attention without guessing from shapes.' },
      { id: 'backpropagation-automatic-differentiation', reason: 'Follow the gradient through gather, shared parameters and residual paths before changing the training loop.' },
    ]} />
    <Practice title="Practice: can this model ignore candidate order automatically?">
      <p>No. The scorer is shared, but the encoder uses absolute positions and contextualizes candidates jointly. Moving a candidate changes its position and the surrounding input. Training-time shuffling discourages easy position shortcuts, but you must align output probabilities by semantic ID and measure their drift. A good permutation test distinguishes a desired property from one the architecture actually guarantees.</p>
    </Practice>
  </>;
}
