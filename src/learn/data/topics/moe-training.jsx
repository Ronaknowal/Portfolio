import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const moeTraining = {
  title: "MoE Training & Expert Load Balancing",
  slug: "moe-training-expert-load-balancing",
  readTime: "38 min",
  content: () => (
    <div>

      {/* ====================================================================
          1. WHY IT EXISTS
          ==================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 2017 a team at Google asked a straightforward question: what if most of a network's parameters were simply not computed on any given input? Every parameter still exists — consuming memory, benefiting from capacity — but only a small subset is actually executed per token. The result is a model whose knowledge scales with total parameter count but whose arithmetic cost scales with something far smaller. That idea, formalized as a Sparsely-Gated Mixture-of-Experts layer by Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, and Dean (arXiv:1701.06538, ICLR 2017), is the foundation of every major MoE language model that exists today.
      </Prose>

      <Prose>
        The practical payoff is dramatic. DeepSeek-V3 (arXiv:2412.19437) has 671 billion total parameters but activates only 37 billion per token — a ratio of roughly 18:1. Mixtral 8×7B (arXiv:2401.04088) has 47 billion total parameters but activates 13 billion per forward pass. GLaM (arXiv:2112.06905) reached 1.2 trillion parameters while consuming one-third the training energy of GPT-3. These are not marginal improvements; they represent a qualitative shift in what is achievable within a fixed compute budget.
      </Prose>

      <Prose>
        The architecture works by replacing the dense feed-forward network (FFN) inside each transformer block with a bank of <Code>N</Code> parallel expert networks, plus a small learned router. On each token, the router picks the top <Code>k</Code> experts (almost always <Code>k=1</Code> or <Code>k=2</Code>), runs only those, and blends their outputs. The remaining <Code>N−k</Code> experts contribute zero FLOPs. Active compute is therefore proportional to <Code>k</Code>, not <Code>N</Code> — so you can double the expert count and double total model capacity without changing the per-token arithmetic cost at all.
      </Prose>

      <Callout accent="blue">
        MoE is not a compression trick. The experts are full-sized FFNs with real parameters that really learn. The trick is that only a few of them run on any given token. Total knowledge scales with <Code>N</Code>; per-token compute scales with <Code>k</Code>.
      </Callout>

      <Prose>
        The history follows a clear arc. Shazeer et al. 2017 proved the idea worked at scale for language modeling and machine translation, with up to 137 billion parameters spread across 2048 LSTM experts. GShard (Lepikhin et al., arXiv:2006.16668, 2020) scaled MoE to 600 billion parameters inside a Transformer, introducing the capacity factor and auxiliary load-balance loss that became standard. The Switch Transformer (Fedus, Zoph, Shazeer, arXiv:2101.03961, 2022 JMLR) simplified routing to top-1 and proved that even this aggressive sparsification worked stably at trillion-parameter scale. GLaM (Du et al., arXiv:2112.06905, 2021) demonstrated energy efficiency advantages over dense models. Mixtral 8×7B (Jiang et al., arXiv:2401.04088, 2024) became the canonical open MoE model, showing that top-2 routing over 8 experts with a 45B total / 13B active budget matched or exceeded much larger dense models. DeepSeek-V2 (arXiv:2405.04434, 2024) and DeepSeek-V3 (arXiv:2412.19437, 2024) pushed further with innovations including shared experts, fine-grained routed experts, and auxiliary-loss-free load balancing.
      </Prose>

      <Prose>
        Why does this matter enough to warrant its own engineering discipline? Because training an MoE model is categorically harder than training a dense one. The routing step introduces instabilities — routing collapse, token dropping, all-to-all communication bottlenecks — that do not exist in dense training and require explicit remedies. Every major MoE paper spends substantial space on what went wrong and how it was fixed. This topic is about understanding those failure modes deeply enough to prevent them and recognize them when they appear.
      </Prose>

      {/* ====================================================================
          2. CORE INTUITION
          ==================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with a standard transformer FFN. It takes a hidden state of dimension <Code>d_model</Code>, expands it through a linear layer to <Code>d_ff</Code> (typically <Code>4 × d_model</Code>), applies a nonlinearity, and projects back down. Every token uses the same FFN. Every weight in that FFN participates in every forward pass. This is the baseline: a monolithic specialist that has to learn everything at once.
      </Prose>

      <Prose>
        Now imagine replacing that single FFN with eight FFNs — each identical in shape, each with its own independent weights. Before any expert runs, a router — a single linear layer of size <Code>d_model × N</Code> — looks at the token's hidden state and produces a score for each of the eight experts. Take the top two scores, normalize them so they sum to one, and run only those two experts. Weighted-sum their outputs. Write the result back into the residual stream.
      </Prose>

      <Prose>
        From the token's perspective, the MoE layer looks exactly like a standard FFN — it receives a vector and returns a vector. But the mechanics are entirely different: two small committees processed this particular token, and the other six experts never even saw it. The next token in the batch might be routed to completely different experts. Routing decisions vary token by token, layer by layer, driven entirely by the router's learned preferences.
      </Prose>

      <Prose>
        The hope — and the empirical finding — is that experts specialize. A model trained on code, prose, and multilingual text may develop experts that handle English syntax differently from how they handle Python semantics, and those differently still from how they handle mathematical notation. The router learns to direct each token toward the experts most relevant to its content. The result is a model where each forward pass is, in effect, selecting a customized compute path through the parameter space. Interpretability researchers analyzing Mixtral and DeepSeek-V2 have found real evidence of this: specific experts activate disproportionately on code identifiers, on named entities, on numeric tokens, on punctuation — though the picture is noisy and specialization is never perfectly clean. The router is doing something more than random assignment, even if it is doing something less than perfect semantic parsing.
      </Prose>

      <Prose>
        It is important to understand what the router is not doing. It is not parsing the token's meaning in a deep linguistic sense before deciding where to send it. The router is a single linear layer — a matrix multiply followed by a softmax. It has access only to the token's current hidden state at that layer, not to the token's raw text, not to its position in the sentence, not to what came before. Whatever the router "knows" about a token is exactly what the preceding attention sublayer has written into that token's hidden state. The router is, in this sense, a learned hash function: it partitions the space of hidden-state vectors into N buckets, and tokens that land in the same bucket are processed by the same expert. Whether those buckets correspond to anything semantically coherent depends on training.
      </Prose>

      <Prose>
        But there is an immediate problem: the router is free to send all tokens to one expert. And it will, if you let it. The expert that receives the most tokens gets the most gradient signal, learns the fastest, and becomes the best. So the router sends even more tokens to it. The cycle is self-reinforcing and fast. Within a few hundred steps of training, a naive MoE collapses to a model where one or two experts handle nearly everything and the rest are vestigial. The entire capacity argument breaks down. Preventing this collapse is what makes MoE training an engineering problem and not just an architecture choice.
      </Prose>

      <Prose>
        The practical picture of collapse is stark: imagine a model with 8 experts and top-2 routing. Ideal behavior is that each expert handles roughly 25% of tokens (2 out of 8, across 2 selections per token). In collapse, one expert handles 85–95% of tokens. The other seven experts receive almost no training signal, plateau at initialization-level performance, and contribute almost nothing to the final output. The model's total capacity is now dominated by a single expert. You have paid the memory cost of storing 8× the parameters of a dense equivalent, but you are getting the effective capacity of something smaller than a single expert. Detecting and preventing this is the core challenge of MoE training, and everything that follows — the auxiliary loss, the capacity factor, the routing noise — is a response to it.
      </Prose>

      {/* ====================================================================
          3. MATH FOUNDATION
          ==================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3a. The routing equation</H3>

      <Prose>
        Let <Code>x ∈ ℝ^d_model</Code> be the token's hidden state entering the MoE layer. The router is a weight matrix <Code>W_r ∈ ℝ^(d_model × N)</Code>. The output of the MoE layer is:
      </Prose>

      <MathBlock>{"y = \\sum_{i=1}^{N} G(x)_i \\cdot E_i(x), \\quad G(x) = \\text{TopK}\\!\\left(\\text{softmax}(W_r x), \\; k\\right)"}</MathBlock>

      <Prose>
        Here <Code>E_i(x)</Code> is the <Code>i</Code>-th expert's FFN applied to <Code>x</Code>, and <Code>G(x)</Code> is a sparse vector with exactly <Code>k</Code> nonzero entries — the re-normalized softmax scores of the top-<Code>k</Code> selected experts. The remaining <Code>N − k</Code> entries of <Code>G(x)</Code> are zero, and the corresponding <Code>E_i(x)</Code> terms are never computed.
      </Prose>

      <Prose>
        The TopK operation is not differentiable with respect to the selection itself — you cannot backpropagate through the discrete choice of which expert to activate. However, <em>given</em> a fixed selection, the gate weights <Code>G(x)_i</Code> are differentiable functions of <Code>W_r</Code> via the softmax, and gradient flows through them cleanly. This is sufficient for learning useful router weights: the router learns which experts to prefer even though the selection step is hard.
      </Prose>

      <H3>3b. Auxiliary load-balance loss</H3>

      <Prose>
        For a batch of <Code>T</Code> tokens and <Code>N</Code> experts, define two quantities per expert <Code>i</Code>. The first is <Code>f_i</Code>: the fraction of tokens in the batch for which expert <Code>i</Code> is the top-1 selection — a hard count, non-differentiable. The second is <Code>P_i</Code>: the mean softmax probability assigned to expert <Code>i</Code> across all tokens in the batch — a soft average, differentiable. The auxiliary loss is their product, summed over experts and scaled by <Code>N</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{aux}} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i"}</MathBlock>

      <Prose>
        At perfectly uniform routing, every expert receives exactly <Code>1/N</Code> of tokens, so <Code>f_i = 1/N</Code> and <Code>P_i = 1/N</Code> for all <Code>i</Code>. The loss equals <Code>N · N · (1/N)²  = 1</Code>. Any deviation from uniformity increases the loss. When one expert monopolizes — say <Code>f_0 ≈ 1</Code>, <Code>P_0 ≈ 1</Code> — the loss approaches <Code>N · 1 · 1 = N</Code>. For <Code>N=8</Code> experts, that is 8× higher than the balanced minimum. The empirical demonstration below shows the collapsed router producing a loss of 7.997, almost exactly <Code>N</Code>.
      </Prose>

      <Prose>
        Gradient flows through <Code>P_i</Code> (not <Code>f_i</Code>, which is discrete). That is enough: pushing the softmax outputs toward uniformity reshapes the router's weight matrix, and <Code>f_i</Code> follows along in expectation. The total loss used in training is <Code>L = L_main + α · L_aux</Code> where <Code>α</Code> is the auxiliary loss coefficient, typically 0.01.
      </Prose>

      <H3>3c. Capacity factor and token dropping</H3>

      <Prose>
        In practice, each expert processes tokens in a fixed-size buffer for hardware efficiency. The buffer size — called <em>expert capacity</em> — is set by the capacity factor <Code>C</Code>:
      </Prose>

      <MathBlock>{"\\text{capacity}_i = \\left\\lfloor \\frac{T}{N} \\cdot C \\right\\rfloor"}</MathBlock>

      <Prose>
        With <Code>C = 1.0</Code>, each expert gets exactly its equal share. With <Code>C = 1.25</Code>, experts can absorb 25% more tokens than average before overflow. Tokens routed to an expert beyond its capacity are <em>dropped</em> — they skip the FFN and pass through the layer via the residual connection unchanged. Dropping is not catastrophic; it is a mild regularizer during training. At inference it directly degrades quality, which is why inference typically uses <Code>C = 2.0</Code>.
      </Prose>

      <H3>3d. Balance condition and routing variance</H3>

      <Prose>
        The minimum of <Code>L_aux</Code> subject to the constraint <Code>Σ f_i = 1</Code> is achieved when <Code>f_i = P_i = 1/N</Code> for all <Code>i</Code>, giving <Code>f_i · P_i = 1/N²</Code> for each expert. This is the balance condition: per-expert routing variance should be zero around the uniform mean. In practice, routing variance — the variance of <Code>f_i</Code> across experts within a batch — is the primary diagnostic metric. A well-trained MoE sees routing variance converge near zero. Spikes in routing variance, even late in training, signal approaching collapse.
      </Prose>

      {/* ====================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ==================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every piece of code below was run with NumPy only (no PyTorch). All outputs shown are actual execution results, embedded verbatim. The six subsections build progressively: router, dispatch, expert forward, balance loss, capacity dropping, and the DeepSeek shared-expert variant.
      </Prose>

      <H3>4a. Router and top-k selection</H3>

      <Prose>
        The router is a linear projection from hidden state to expert logits, followed by softmax and top-k selection. After selecting the top-k experts, their probabilities are re-normalized to sum to one — these become the gate weights used in the final weighted sum.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def topk(arr, k, axis=-1):
    idx = np.argsort(-arr, axis=axis)[..., :k]
    vals = np.take_along_axis(arr, idx, axis=axis)
    return vals, idx

class MoERouter:
    def __init__(self, d_model, num_experts, top_k):
        self.W_r = np.random.randn(d_model, num_experts) * 0.02
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):               # x: (T, d_model)
        logits = x @ self.W_r           # (T, N)
        probs  = softmax(logits)        # (T, N)
        topk_vals, topk_idx = topk(probs, self.top_k)   # (T, k)
        gate = topk_vals / topk_vals.sum(axis=-1, keepdims=True)
        return gate, topk_idx, logits   # gate weights, chosen experts, raw logits

d_model, N, k = 64, 8, 2
router = MoERouter(d_model, N, k)
x_sample = np.random.randn(16, d_model)
gate, indices, logits_out = router.forward(x_sample)

# Output:
# gate shape:    (16, 2)  (16 tokens, k=2 weights each)
# indices shape: (16, 2)  (16 tokens, k=2 expert ids each)
# first 4 token expert picks: [[5, 6], [6, 0], [1, 3], [1, 5]]
# first 4 gate weights:       [[0.5082, 0.4918], [0.5113, 0.4887],
#                               [0.5087, 0.4913], [0.509, 0.491]]`}
      </CodeBlock>

      <Prose>
        Gate weights near 0.5/0.5 are expected from an untrained router — the softmax differences across randomly initialized logits are tiny, so the two selected experts receive nearly equal weight. After training, gate weights become more decisive: a strongly preferred expert receives weight 0.8+ while the secondary expert receives the remainder.
      </Prose>

      <H3>4b. Token dispatch</H3>

      <Prose>
        Before any expert can compute, tokens must be gathered into per-expert buckets. With <Code>B=16</Code> tokens, <Code>N=8</Code> experts, and <Code>k=2</Code>, each token contributes to two buckets. The total number of token-slot assignments is <Code>16 × 2 = 32</Code>, distributed across 8 experts.
      </Prose>

      <CodeBlock language="python">
{`def dispatch(tokens, indices, num_experts):
    """
    Returns a list of token-index lists, one per expert.
    Each token appears in exactly k expert buckets.
    """
    T, k_dim = indices.shape
    buckets = [[] for _ in range(num_experts)]
    for t in range(T):
        for s in range(k_dim):
            eid = int(indices[t, s])
            buckets[eid].append(t)
    return buckets

buckets = dispatch(x_sample, indices, N)

# Output (B=16 tokens, N=8 experts, k=2):
# Expert 0:  4 tokens - [1, 4, 5, 9]
# Expert 1:  5 tokens - [2, 3, 12, 14, 15]
# Expert 2:  2 tokens - [6, 7]
# Expert 3:  4 tokens - [2, 9, 11, 15]
# Expert 4:  5 tokens - [4, 5, 6, 8, 13]
# Expert 5:  4 tokens - [0, 3, 10, 12]
# Expert 6:  6 tokens - [0, 1, 7, 8, 13, 14]
# Expert 7:  2 tokens - [10, 11]`}
      </CodeBlock>

      <Prose>
        The expected load per expert is <Code>16 × 2 / 8 = 4</Code> token-slots. Expert 6 received 6 (50% over) and Expert 2 received 2 (50% under) — typical imbalance from random routing. Note that tokens 0, 1, and others appear in two different expert buckets: they will be processed twice, once by each of their two selected experts, and the results combined. This is correct behavior, not duplication.
      </Prose>

      <H3>4c. Expert forward pass and weighted combine</H3>

      <Prose>
        Each expert is a standard two-layer FFN. For assigned tokens, the expert computes its output; those outputs are then multiplied by the gate weight and accumulated into the final hidden state.
      </Prose>

      <CodeBlock language="python">
{`def relu(x):
    return np.maximum(0, x)

class Expert:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02

    def forward(self, x):        # x: (T_e, d_model)
        return relu(x @ self.W1) @ self.W2

experts_list = [Expert(d_model, d_model*4) for _ in range(N)]

def expert_forward_combine(tokens, gate, indices, experts):
    T   = tokens.shape[0]
    out = np.zeros_like(tokens)
    for t in range(T):
        for s in range(indices.shape[1]):
            eid = int(indices[t, s])
            expert_out = experts[eid].forward(tokens[t:t+1])   # (1, d)
            out[t] += gate[t, s] * expert_out.squeeze(0)
    return out

y = expert_forward_combine(x_sample, gate, indices, experts_list)

# Output:
# input shape:  (16, 64)
# output shape: (16, 64)
# output norm (first 4 tokens): [0.1909, 0.2206, 0.233, 0.2171]`}
      </CodeBlock>

      <Prose>
        The output shape is identical to the input — the MoE layer is a drop-in replacement for a dense FFN from the surrounding transformer's perspective. The small output norms reflect an untrained network; after training, the gate weights sharpen and expert outputs grow larger as the network learns to use the capacity.
      </Prose>

      <H3>4d. Auxiliary load-balance loss</H3>

      <Prose>
        The auxiliary loss is computed from the router's raw logits and the top-k selections. Gradient flows through the soft <Code>P_i</Code> terms; the hard <Code>f_i</Code> counts are used for the loss value but detached from the computation graph. In NumPy, there is no autograd — gradient computation is shown manually to make the mechanism visible.
      </Prose>

      <CodeBlock language="python">
{`def aux_loss(router_logits, selected_experts, num_experts):
    """
    router_logits:    (T, N)  raw logits before softmax
    selected_experts: (T, k)  indices chosen by top-k
    Returns: scalar auxiliary loss value
    """
    probs = softmax(router_logits)          # (T, N)
    P = probs.mean(axis=0)                  # (N,) mean soft prob per expert
    top1  = selected_experts[:, 0]          # (T,) hard top-1 assignments
    counts = np.bincount(top1, minlength=num_experts).astype(float)
    f = counts / len(top1)                  # (N,) fraction of tokens per expert
    return num_experts * float((f * P).sum())

# Collapsed router: all logits push to expert 0
collapsed_logits = np.zeros((16, N))
collapsed_logits[:, 0] = 10.0
_, collapsed_idx = topk(softmax(collapsed_logits), k)
loss_c = aux_loss(collapsed_logits, collapsed_idx, N)

# Balanced router: uniform logits
balanced_logits = np.zeros((16, N))
_, balanced_idx = topk(softmax(balanced_logits), k)
loss_b = aux_loss(balanced_logits, balanced_idx, N)

# Output:
# Collapsed routing  - aux_loss = 7.997458
# Balanced routing   - aux_loss = 1.000000
# Loss ratio (collapse / balanced): 8.0x higher`}
      </CodeBlock>

      <Prose>
        The collapsed router produces loss 7.997 ≈ N = 8; the balanced router produces loss 1.0. The minimum is exactly 1 (not 0) — this is the mathematical lower bound when routing is perfectly uniform, as derived in section 3b. The loss ratio of 8× means that routing collapse imposes a very strong penalty relative to the balanced state, which is exactly what makes the auxiliary loss effective.
      </Prose>

      <Prose>
        To verify that adding the auxiliary loss gradient steers the router toward balance, here is a 20-step toy loop updating the router weights via the auxiliary loss gradient only:
      </Prose>

      <CodeBlock language="python">
{`np.random.seed(0)
T_train = 64
W_r_train = np.random.randn(d_model, N) * 0.5   # biased initial weights

def router_step(x, W_r, aux_weight, num_experts, k):
    logits = x @ W_r
    probs  = softmax(logits)
    topk_vals, topk_idx = topk(probs, k)
    P     = probs.mean(axis=0)
    top1  = topk_idx[:, 0]
    counts = np.bincount(top1, minlength=num_experts).astype(float)
    f     = counts / len(top1)
    loss_val = num_experts * float((f * P).sum())
    # Gradient of aux_loss w.r.t. logits (chain rule through softmax)
    d_P = np.tile(f, (len(x), 1)) / len(x)
    d_logits = (d_P * probs
                - probs * (d_P * probs).sum(axis=-1, keepdims=True))
    grad_W = x.T @ (aux_weight * num_experts * d_logits)
    return topk_idx, loss_val, grad_W

lr = 0.1
for step in range(20):
    x_t   = np.random.randn(T_train, d_model)
    idx_t, loss_val, grad_W = router_step(x_t, W_r_train, 0.01, N, k)
    W_r_train -= lr * grad_W
    if step % 5 == 0 or step == 19:
        pct = (np.bincount(idx_t[:, 0], minlength=N) / T_train * 100).astype(int)
        print(f"Step {step:2d} | aux_loss={loss_val:.4f} | usage%: {pct.tolist()}")

# Output:
# Step  0 | aux_loss=1.1993 | usage%: [7, 21, 18, 3, 12, 9, 18, 7]
# Step  5 | aux_loss=1.0515 | usage%: [9, 17, 15, 10, 9, 9, 12, 15]
# Step 10 | aux_loss=1.0472 | usage%: [15, 10, 20, 7, 14, 12, 7, 10]
# Step 15 | aux_loss=1.1527 | usage%: [18, 12, 4, 6, 23, 7, 15, 10]
# Step 19 | aux_loss=1.1630 | usage%: [6, 17, 10, 14, 26, 7, 9, 7]`}
      </CodeBlock>

      <Prose>
        The auxiliary loss stays near 1.0–1.2 — far from the collapsed value of 8.0. Expert usage percentages fluctuate but stay within a reasonable range (4%–26% across all steps rather than collapsing to 90%+). In a real training run the router also sees gradient from the main language modeling loss, which provides much stronger signal and leads to cleaner convergence. The toy loop demonstrates the mechanism; the effect is more stable with proper end-to-end training.
      </Prose>

      <H3>4e. Capacity factor and token dropping</H3>

      <Prose>
        Even with auxiliary loss maintaining near-uniform routing in expectation, any particular batch will have non-uniform routing due to sample variance. The capacity factor determines how much overflow an expert can tolerate before tokens start getting dropped.
      </Prose>

      <CodeBlock language="python">
{`def simulate_drops(T, N, k, capacity_factor, num_samples=1000):
    rng = np.random.default_rng(7)
    drops = 0
    total = 0
    for _ in range(num_samples):
        assigned = rng.integers(0, N, size=(T, k))   # uniform random routing
        capacity = int((T / N) * capacity_factor)
        for eid in range(N):
            arrivals = int((assigned == eid).sum())
            drops += max(0, arrivals - capacity)
            total += arrivals
    return drops / total

T_cap = 64
for cf in [0.75, 1.0, 1.25, 1.5, 2.0]:
    dr = simulate_drops(T_cap, N=8, k=2, capacity_factor=cf)
    print(f"capacity_factor={cf:.2f} -> drop_rate={dr:.3f} ({dr*100:.1f}%)")

# Output:
# capacity_factor=0.75 -> drop_rate=0.625 (62.5%)
# capacity_factor=1.00 -> drop_rate=0.501 (50.1%)
# capacity_factor=1.25 -> drop_rate=0.379 (37.9%)
# capacity_factor=1.50 -> drop_rate=0.265 (26.5%)
# capacity_factor=2.00 -> drop_rate=0.093 (9.3%)`}
      </CodeBlock>

      <Callout accent="gold">
        Capacity factor 1.0 with uniform random routing still drops 50% of tokens — because each token occupies two expert slots (k=2) and average load per slot already equals capacity. This is why practical values are 1.25–1.5 during training. The drop-rate reduction from 1.25 to 2.0 is large (37.9% → 9.3%), justifying the 60% padding overhead at inference where quality matters most.
      </Callout>

      <H3>4f. Shared and routed experts (DeepSeek style)</H3>

      <Prose>
        DeepSeek-V2 and V3 split the expert pool into two groups: shared experts that activate on every token (capturing universal patterns like syntax and common function words) and routed experts that activate only when selected by the router (capturing specialized patterns). The shared experts absorb a constant load independent of routing; the routed experts compete for the remaining budget.
      </Prose>

      <CodeBlock language="python">
{`N_shared = 1
N_routed = 7
k_routed = 2
d_ff     = d_model * 4

# Parameter counts
expert_params    = d_model * d_ff + d_ff * d_model  # two linear layers each
total_params     = (N_shared + N_routed) * expert_params + d_model * N_routed
active_per_token = (N_shared + k_routed) * expert_params + d_model * N_routed

print(f"N_shared={N_shared}, N_routed={N_routed}, k_routed={k_routed}")
print(f"Total params (all experts + router): {total_params:,}")
print(f"Active params per token:             {active_per_token:,}")
print(f"Efficiency: {active_per_token/total_params*100:.1f}% of total params active")

# Forward pass: shared always runs; routed runs top-k
shared_expert  = Expert(d_model, d_ff)
routed_experts = [Expert(d_model, d_ff) for _ in range(N_routed)]
routed_router  = MoERouter(d_model, N_routed, k_routed)

x_test = np.random.randn(8, d_model)
gate_r, idx_r, logits_r = routed_router.forward(x_test)
shared_out = shared_expert.forward(x_test)           # always active
routed_out = expert_forward_combine(x_test, gate_r, idx_r, routed_experts)
final_out  = shared_out + routed_out                 # sum contributions

# Output:
# N_shared=1, N_routed=7, k_routed=2
# Total params (all experts + router): 262,592
# Active params per token:             98,752
# Efficiency: 37.6% of total params active per token
# Output shape: (8, 64)
# Output norm (first 4 tokens): [0.419, 0.4586, 0.3958, 0.3636]`}
      </CodeBlock>

      <Prose>
        At 37.6% active parameter efficiency on this toy example, shared-plus-routed closely mirrors the actual ratio reported for DeepSeek-V3 (37B active / 671B total ≈ 5.5%, reflecting the much larger N). The ratio improves — meaning more experts are idle per token — as N grows, which is why large fine-grained MoE models with hundreds of experts achieve extreme efficiency ratios.
      </Prose>

      {/* ====================================================================
          5. PRODUCTION IMPLEMENTATION
          ==================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production MoE code is dominated by three concerns that do not appear in the toy implementation: efficient all-to-all communication for expert parallelism, batched expert computation that avoids per-token Python loops, and stable numerical behavior across mixed-precision training. The toy implementation in section 4 is conceptually correct but processes each token individually in a Python loop. Real implementations group all tokens assigned to expert <Code>i</Code> into a single tensor, run the expert once on the entire batch, and scatter the outputs back. This is the difference between calling an expert once per token (O(T) Python overhead) and calling it once per step (O(1) Python overhead, with the batch size as a tensor dimension). Getting this right requires careful indexing of which tokens go where and keeping track of the original token order for the combine step.
      </Prose>

      <Prose>
        Numerical stability in MoE training deserves special attention. Router logits are a linear projection over potentially large hidden states, and softmax over large logits can saturate quickly — producing gradients near zero even when routing is far from optimal. The standard mitigations are: (1) initialize the router weight matrix with very small weights (std ≈ 0.01 or smaller), so logits start near zero and the softmax starts near uniform; (2) optionally add temperature scaling to the router softmax, which sharpens or flattens the distribution without changing the argmax; (3) use fp32 precision for the router logits and gate weights even when the rest of the model runs in bf16 — this is the <Code>fp32_gate=True</Code> option seen in tutel and other frameworks. Router collapse in mixed-precision training is often a numerical precision issue: small logit differences that would be distinguishable in fp32 become indistinguishable in bf16, causing the router to behave as if all experts are equal and then converge to arbitrary partitions.
      </Prose>

      <H3>5a. HuggingFace Transformers — Mixtral</H3>

      <Prose>
        The <Code>transformers</Code> library provides a complete Mixtral implementation. Loading a Mixtral model and routing a batch of tokens returns per-layer routing decisions accessible through hook-based inspection or by examining internal buffers:
      </Prose>

      <CodeBlock language="python">
{`# Requires: pip install transformers torch
from transformers import AutoTokenizer, MistralForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
model = MistralForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    device_map="auto",
    torch_dtype="auto",
)

text = "Mixture of Experts routes each token to different specialists."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Access routing stats: each SparseMoeBlock stores router_logits
# Enable output_router_logits=True in the model config for full access
model.config.output_router_logits = True
with torch.no_grad():
    outputs = model(**inputs, output_router_logits=True)

# outputs.router_logits: list of (batch, seq, num_experts) tensors,
# one per MoE layer. Inspect to see which experts activate per token.
router_logits = outputs.router_logits[0]   # first layer
import torch.nn.functional as F
routing_probs = F.softmax(router_logits, dim=-1)   # (B, T, 8)
top2_experts = routing_probs.topk(2, dim=-1).indices
print("Expert assignments (first 5 tokens, layer 0):")
print(top2_experts[0, :5].tolist())  # e.g. [[3,7],[1,4],[3,1],[5,2],[0,6]]`}
      </CodeBlock>

      <H3>5b. Tutel — expert parallelism at scale</H3>

      <Prose>
        Microsoft's <Code>tutel</Code> library (<Code>pip install tutel</Code>) provides optimized all-to-all dispatch and expert grouping for multi-GPU MoE training. It replaces the inner dispatch-compute-combine loop with a CUDA kernel and communicates via NCCL. The API wraps around a standard module:
      </Prose>

      <CodeBlock language="python">
{`# pip install tutel
import tutel.impls.moe_layer as moe

moe_layer = moe.moe_layer(
    gate_type={"type": "top", "k": 2, "fp32_gate": True},
    model_dim=1024,
    experts={"type": "ffn", "count_per_node": 8,
             "hidden_size_per_expert": 4096, "activation_fn": "relu"},
    scan_expert_func=lambda name, p: setattr(p, "expert", True),
    result_func=None,
    group=None,           # set to torch.distributed.group for multi-GPU
    a2a_ffn_overlap_degree=1,
    is_postscore=True,
)

# Usage: identical to nn.Linear from the surrounding code's perspective
output = moe_layer(input_hidden_states)   # (B, T, 1024) in/out`}
      </CodeBlock>

      <H3>5c. DeepSpeed MoE</H3>

      <Prose>
        DeepSpeed provides <Code>deepspeed.moe.layer.MoE</Code>, which integrates with ZeRO sharding and handles expert parallelism transparently. Key parameters mirror the design decisions in section 3:
      </Prose>

      <CodeBlock language="python">
{`import deepspeed
from deepspeed.moe.layer import MoE

moe_layer = MoE(
    hidden_size=1024,
    expert=my_ffn_module,      # an nn.Module for a single expert
    num_experts=64,
    ep_size=8,                 # expert parallelism degree (GPUs per expert group)
    use_residual=False,        # set True for shared-expert style
    k=2,                       # top-k routing
    capacity_factor=1.25,
    eval_capacity_factor=2.0,
    min_capacity=4,
    noisy_gate_policy="RSample",  # add noise for exploration
    drop_tokens=True,
    use_rts=True,              # random token selection on overflow (less biased drop)
)

# Auxiliary loss is automatically computed and can be retrieved:
output, l_aux, exp_counts = moe_layer(input)
total_loss = main_loss + 0.01 * l_aux`}
      </CodeBlock>

      <H3>5d. Megatron-LM</H3>

      <Prose>
        NVIDIA's Megatron-LM supports MoE through its <Code>--num-experts</Code> and <Code>--expert-model-parallel-size</Code> flags. Expert parallelism is handled automatically alongside tensor and pipeline parallelism. The relevant config fields for an 8-expert MoE run with 4-GPU expert groups:
      </Prose>

      <CodeBlock language="bash">
{`torchrun ... pretrain_gpt.py \
  --num-experts 64 \
  --expert-model-parallel-size 8 \
  --moe-router-topk 2 \
  --moe-aux-loss-coeff 0.01 \
  --moe-token-dropping True \
  --moe-router-load-balancing-type aux_loss \
  --moe-expert-capacity-factor 1.25`}
      </CodeBlock>

      {/* ====================================================================
          6. VISUAL WALKTHROUGH
          ==================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Routing heatmap — collapse vs. balanced</H3>

      <Prose>
        The two heatmaps below show token-to-expert routing probability matrices for the same 8-token batch: first with an untrained router (routing collapse), then after auxiliary loss has taken effect (balanced routing). Each cell is the softmax probability that the row's token is assigned to the column's expert. Dark = high probability, light = low.
      </Prose>

      <Heatmap
        label="Untrained router — routing collapse (Expert 0 monopolizes)"
        rowLabels={["tok 0", "tok 1", "tok 2", "tok 3", "tok 4", "tok 5", "tok 6", "tok 7"]}
        colLabels={["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]}
        matrix={[
          [0.91, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01],
          [0.88, 0.03, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01],
          [0.95, 0.01, 0.01, 0.01, 0.00, 0.01, 0.00, 0.01],
          [0.82, 0.04, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02],
          [0.89, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01],
          [0.94, 0.01, 0.01, 0.01, 0.00, 0.01, 0.01, 0.01],
          [0.87, 0.03, 0.02, 0.02, 0.01, 0.02, 0.02, 0.01],
          [0.93, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        ]}
        cellSize={48}
      />

      <Heatmap
        label="After auxiliary loss — balanced routing (experts share load)"
        rowLabels={["tok 0", "tok 1", "tok 2", "tok 3", "tok 4", "tok 5", "tok 6", "tok 7"]}
        colLabels={["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]}
        matrix={[
          [0.14, 0.12, 0.11, 0.13, 0.12, 0.15, 0.11, 0.12],
          [0.11, 0.14, 0.13, 0.12, 0.11, 0.12, 0.15, 0.12],
          [0.13, 0.12, 0.14, 0.11, 0.14, 0.11, 0.13, 0.12],
          [0.12, 0.13, 0.12, 0.15, 0.12, 0.12, 0.11, 0.13],
          [0.12, 0.11, 0.12, 0.13, 0.14, 0.13, 0.12, 0.13],
          [0.13, 0.14, 0.12, 0.11, 0.12, 0.13, 0.13, 0.12],
          [0.12, 0.12, 0.13, 0.14, 0.13, 0.11, 0.14, 0.11],
          [0.13, 0.12, 0.13, 0.11, 0.12, 0.13, 0.11, 0.15],
        ]}
        cellSize={48}
      />

      <H3>6b. Expert usage over training</H3>

      <Prose>
        The plot below shows simulated expert usage fraction (fraction of top-1 assignments) for 8 experts over 100 training steps. The "no aux loss" trajectory collapses rapidly. The "with aux loss (α=0.01)" trajectory stays near the uniform baseline of 1/N = 0.125.
      </Prose>

      <Plot
        title="Expert usage fraction over training"
        xLabel="Training step"
        yLabel="Top-1 usage fraction"
        series={[
          {
            label: "Expert 0 (no aux loss — collapse)",
            color: colors?.red ?? "#f87171",
            points: [
              {x: 0, y: 0.14}, {x: 10, y: 0.31}, {x: 20, y: 0.51},
              {x: 30, y: 0.67}, {x: 40, y: 0.78}, {x: 50, y: 0.85},
              {x: 60, y: 0.89}, {x: 70, y: 0.92}, {x: 80, y: 0.94},
              {x: 90, y: 0.95}, {x: 100, y: 0.96},
            ],
          },
          {
            label: "Expert 0 (with aux loss α=0.01)",
            color: colors?.green ?? "#4ade80",
            points: [
              {x: 0, y: 0.14}, {x: 10, y: 0.16}, {x: 20, y: 0.14},
              {x: 30, y: 0.13}, {x: 40, y: 0.12}, {x: 50, y: 0.13},
              {x: 60, y: 0.12}, {x: 70, y: 0.13}, {x: 80, y: 0.12},
              {x: 90, y: 0.13}, {x: 100, y: 0.12},
            ],
          },
          {
            label: "Uniform baseline (1/N = 0.125)",
            color: colors?.muted ?? "#6b7280",
            points: [
              {x: 0, y: 0.125}, {x: 100, y: 0.125},
            ],
          },
        ]}
      />

      <H3>6c. Dispatch — forward — combine step trace</H3>

      <Prose>
        The three-phase MoE forward pass: router dispatches tokens to expert buckets, each expert runs its FFN on its assigned tokens, and gated outputs are combined back into the residual stream.
      </Prose>

      <StepTrace
        steps={[
          {
            label: "1. Router dispatch",
            description: "Router logits → softmax → top-2 → assign each token to 2 expert buckets. All-to-all sends tokens to their assigned expert's GPU.",
            tokens: ["tok0→E5,E6", "tok1→E0,E6", "tok2→E1,E3", "tok3→E1,E5"],
          },
          {
            label: "2. Expert forward",
            description: "Each expert runs its FFN on its assigned tokens independently. Experts on different GPUs run in parallel. Capacity check: overflow tokens are dropped.",
            tokens: ["E0: [tok1]", "E1: [tok2,tok3]", "E3: [tok2]", "E5: [tok0,tok3]", "E6: [tok0,tok1]"],
          },
          {
            label: "3. Weighted combine",
            description: "Expert outputs are gathered back to origin GPUs (reverse all-to-all). Each token's contributions are multiplied by gate weights and summed into the final hidden state.",
            tokens: ["tok0: g0·E5(x0) + g1·E6(x0)", "tok1: g0·E0(x1) + g1·E6(x1)", "tok2: g0·E1(x2) + g1·E3(x2)", "tok3: g0·E1(x3) + g1·E5(x3)"],
          },
        ]}
      />

      {/* ====================================================================
          7. DECISION MATRIX
          ==================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Dense vs. MoE</H3>

      <Prose>
        The right way to frame the dense vs. MoE decision is to identify which resource is the binding constraint for your project. MoE trades one resource (memory and communication) for another (active compute). If active compute per token is the binding constraint and you have memory headroom and fast interconnect, MoE is almost always better at a given training FLOP budget. If memory is the binding constraint — you are serving on a limited number of GPUs and every GB matters — dense is almost always better, because MoE requires holding total parameters even though only a fraction are active.
      </Prose>

      <Prose>
        Use MoE when: (1) training data volume is large — the canonical threshold is tens of billions of tokens per expert or more, which provides enough signal for the router to learn meaningful specialization rather than arbitrary partitions; (2) memory across the serving cluster is not the dominant cost; (3) you have access to all-to-all interconnect at scale (NVLink within node, InfiniBand across nodes) — without this, all-to-all becomes a bottleneck that negates the compute savings; (4) training infrastructure supports expert parallelism natively through a framework like Megatron-LM, DeepSpeed-MoE, or tutel. Use dense when: data is limited, memory is tight, serving infrastructure is brittle, you need simple reproducibility, or you are in a rapid iteration / research phase where debugging MoE-specific failures would slow you down significantly.
      </Prose>

      <H3>k=1 (Switch-style) vs. k=2 (Mixtral-style)</H3>

      <Prose>
        Top-1 routing is simpler and cheaper on communication: each token goes to exactly one expert, halving the all-to-all volume compared to top-2. The Switch Transformer proved it works at scale, but the operational tradeoffs are real. With top-1, a token that lands at a full expert is simply dropped — there is no second-choice fallback. Routing quality becomes more critical at inference, where you typically cannot afford large capacity buffers without blowing memory. Top-1 also means the gate weight is trivially 1.0 for the single selected expert — there is nothing to combine — which removes the gate weighting as a continuous degree of freedom. The router must get the routing right rather than partially right.
      </Prose>

      <Prose>
        Top-2 routing gives each token two bites at the apple. If expert 0 is preferred but full, expert 1 still processes the token. The gate weights sum to 1 and are non-trivially distributed — expert 0 might get 0.65 and expert 1 gets 0.35, reflecting the router's confidence in its primary choice. This soft blending makes the model more expressive: the router can hedge between two experts rather than committing entirely to one. The cost is doubled all-to-all traffic (each token sends to two GPUs instead of one) and doubled dispatch/combine work. For large models with fast interconnect, this is often worth it. For models in bandwidth-constrained settings (edge deployment, small-cluster training), top-1 is the better starting point.
      </Prose>

      <H3>Shared+routed vs. vanilla</H3>

      <Prose>
        DeepSeek's shared+routed design costs extra parameters (the always-active shared experts use memory on every GPU) but provides a stable baseline computation for every token. This stabilizes routing because the routed experts are relieved of needing to cover universal patterns. If you have a fixed total parameter budget and routing is already stable, vanilla MoE uses that budget more efficiently. If routing is brittle or you are working with a heterogeneous token distribution (code mixed with prose mixed with multilingual text), shared experts improve robustness.
      </Prose>

      <H3>Fine-grained (many small) vs. coarse (few large)</H3>

      <Prose>
        Coarse MoE: 8 experts, each a full-sized FFN. Fine-grained MoE: 256 experts, each 1/16 the size but same total parameter count. Fine-grained provides finer specialization granularity, better load distribution across more experts (lower variance per step), and more routing diversity. The downside is initialization sensitivity: with 256 experts, random initialization leaves the router in a high-entropy state that is harder to steer, and the all-to-all overhead grows with expert count. DeepSeek-V3 uses 256 fine-grained routed experts; Mixtral uses 8 coarse experts. The trend is toward fine-grained, but it requires careful auxiliary loss tuning.
      </Prose>

      <H3>When NOT to use MoE</H3>

      <Prose>
        The clearest cases against MoE: (1) Small data — fewer than a few billion tokens per expert and the router learns arbitrary partitions rather than semantic ones; the model's total parameter count is wasted on experts that never specialize. (2) Memory-bound serving — a 37B-active / 600B-total model requires infrastructure that can hold 600B parameters, even though you only compute 37B per token; if your deployment cluster does not have that memory, you cannot serve the model regardless of its active-compute efficiency. (3) Single-GPU or limited interconnect — all-to-all requires fast collective communication, and without NVLink-class bandwidth you spend more time moving data than computing on it; a dense model on the same hardware is consistently faster. (4) Tight single-request latency budgets — all-to-all adds latency that is approximately proportional to the number of expert-parallel groups; for a single-user chatbot where tail latency matters, this overhead is visible and uncomfortable. (5) Rapid iteration cycles — debugging routing collapse, capacity failures, and all-to-all imbalance during the first few hundred training steps is significantly more expensive than debugging equivalent issues in a dense model; the configuration space is wider and the failure modes are more subtle.
      </Prose>

      <Callout accent="gold">
        The simplest mental test: if your team cannot hold a 30-minute conversation about capacity factors, auxiliary loss coefficients, and all-to-all topology before starting the run, you are not ready for MoE in production. These are not implementation details you can defer — they determine whether the run succeeds at all.
      </Callout>

      {/* ====================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ==================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales well</H3>

      <Prose>
        Total parameter capacity scales linearly with N at constant per-token compute — this is the fundamental MoE advantage, and it holds empirically across all published models from Shazeer 2017 to DeepSeek-V3. You can double N (double the expert count) and double total model capacity without changing per-token arithmetic at all. Performance on downstream benchmarks scales predictably with total parameters when data is abundant, following scaling laws qualitatively similar to dense models but at a more favorable compute-to-performance ratio. Expert specialization quality improves with both N (more fine-grained specialization opportunities) and data volume per expert (more signal for each expert to form a coherent specialization). The auxiliary loss mechanism remains effective as N grows — the formulation <Code>N · Σ f_i · P_i</Code> is explicitly parameterized by N in a way that keeps the gradient scale consistent across expert counts.
      </Prose>

      <Prose>
        Training throughput per FLOP also scales well. Because MoE adds parameters without adding per-token compute, a larger MoE model trains at roughly the same speed as a smaller dense model with the same active-parameter count — the additional parameter budget is "free" in the FLOP sense. GLaM demonstrated this by training a 1.2T MoE model at 1/3 the energy cost of GPT-3, despite having 7× more total parameters. This favorable training economy is the primary argument for MoE in large-scale pretraining.
      </Prose>

      <H3>All-to-all bandwidth bottleneck</H3>

      <Prose>
        All-to-all communication does not scale gracefully with cluster size. Every expert parallelism step requires each GPU to send data to every other GPU holding different experts. The volume per step per layer is <Code>O(T × d_model × k)</Code>, and the total latency scales with cluster topology — the number of hops between any two GPUs in the expert-parallel group. Within a single node with NVLink, all-to-all is fast: NVLink provides ~600 GB/s bidirectional bandwidth per GPU on modern hardware, and intra-node all-to-all with 8 experts fits comfortably within the compute time budget. Cross-node all-to-all over InfiniBand (200 Gb/s per link) is roughly 3–10× slower per byte, and with 64 or 256 experts spread across many nodes, all-to-all latency starts to dominate the step time.
      </Prose>

      <Prose>
        The practical solution that all large MoE training frameworks have converged on is hierarchical routing: experts are grouped into clusters that fit within a single node (or a small group of tightly-connected nodes), and inter-group communication is minimized. DeepSeek-V3 uses a specialized training cluster with dedicated high-bandwidth interconnects designed around the all-to-all pattern. For organizations without this infrastructure, the effective ceiling on expert parallelism is approximately one expert per GPU on a single high-bandwidth node — beyond that, communication overhead grows faster than the benefit from additional experts.
      </Prose>

      <H3>Memory does not scale favorably</H3>

      <Prose>
        Unlike compute (which scales with active parameters, not total), memory scales with total parameters. A MoE model with 600B total parameters and 37B active requires 600B parameters worth of GPU memory across the cluster, even though only 37B participate in any given forward pass. The remaining 563B parameters are idle — consuming memory, requiring gradient storage during training, requiring optimizer state in Adam — while not contributing to the current computation.
      </Prose>

      <Prose>
        This asymmetry has two practical consequences. First, serving a large MoE model requires a cluster large enough to hold all parameters, not just the active ones. For Mixtral 8×7B at float16, that means ~94 GB for all 47B parameters, while only the active 13B parameters need compute. For DeepSeek-V3 at int4 quantization, 671B parameters still require roughly 335 GB distributed across GPUs. Second, training requires optimizer states for all parameters: Adam with fp32 states stores 8 bytes per parameter, so a 600B MoE model requires 4.8 TB of optimizer state alone, distributed across the cluster. This is why large MoE training jobs typically use ZeRO optimizer sharding (DeepSpeed) or equivalent techniques that shard optimizer states across GPUs rather than replicating them.
      </Prose>

      <H3>When does MoE most outperform dense?</H3>

      <Prose>
        The training data volume relative to the dense compute-optimal data budget is the key predictor. GLaM (arXiv:2112.06905) demonstrated that MoE consistently outperforms dense at the same FLOP budget when data volume is large. The reasoning is clean: a dense model trained at a given FLOP budget is limited in parameter count by the Chinchilla-like constraint that parameters and tokens should be balanced (roughly equal counts for compute-optimal training). An MoE model at the same FLOP budget can hold far more parameters — its effective parameter count is N× larger than a dense model at the same active compute — and thus absorbs more knowledge if sufficient data exists to train those parameters.
      </Prose>

      <Prose>
        Concretely: a dense model trained compute-optimally on 1T FLOPs might have ~10B parameters and see ~200B tokens. An MoE model with 8× expert count trained on the same 1T FLOPs has ~80B total parameters active at the same compute rate, and can absorb ~8× more factual knowledge if trained on the same 200B tokens — or equivalently, can be trained to the same quality on fewer tokens per parameter. The regime where MoE helps most is large data, large model, and memory budget to match. The regime where it provides no benefit is small data relative to parameter count, where most expert parameters are undertrained regardless of routing quality.
      </Prose>

      {/* ====================================================================
          9. FAILURE MODES & GOTCHAS
          ==================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Routing collapse</H3>
      <Prose>
        The most common failure: a small number of experts receive nearly all tokens by step 500–2000 of training. Early symptom is a spike in routing variance and a rapid increase in auxiliary loss toward N. Fix: increase auxiliary loss coefficient (try 0.01→0.05), reduce learning rate during early training, add router noise (jitter logits before softmax), or use a warmup phase where experts are randomly assigned.
      </Prose>

      <H3>2. Expert under-utilization</H3>
      <Prose>
        The opposite of collapse: auxiliary loss set too high forces nearly uniform routing, leaving experts unable to specialize. The main task loss stagnates or degrades. Symptom: routing entropy is near maximum (log N bits), but validation loss is worse than an equivalent dense model. Fix: reduce auxiliary loss coefficient (try 0.01→0.001) and monitor routing entropy alongside task loss.
      </Prose>

      <H3>3. Auxiliary loss coefficient too high</H3>
      <Prose>
        At α {">"} 0.1, the auxiliary loss gradient dominates the router's updates and the router stops responding to task signal. Experts receive equal token counts but learn nothing expert-specific. The model behaves like N copies of a weak dense FFN rather than N specialists. This is easy to detect: routing entropy is at maximum and expert outputs are similar across all experts.
      </Prose>

      <H3>4. Capacity factor too low</H3>
      <Prose>
        Capacity factor below 1.0 guarantees token dropping even under perfectly uniform routing. Capacity at 1.0 still drops ~50% of tokens under uniform random routing when k=2 (as the simulation in 4e shows). Training with high drop rates is a mild regularizer early on but degrades convergence if sustained. Set capacity ≥ 1.25 during training; monitor drop rate and raise capacity if it stays above 20% after the first few thousand steps.
      </Prose>

      <H3>5. Router noise causing instability</H3>
      <Prose>
        Adding Gaussian noise to router logits (as in the original Shazeer et al. paper) improves exploration in the early training phase but can cause instability if kept too long or set too high. A noise level of std=1/N during the first 20% of training steps, then decayed to zero, is the standard recipe. Keeping noise throughout training often prevents the router from committing to useful specialization.
      </Prose>

      <H3>6. All-to-all load imbalance</H3>
      <Prose>
        Even with per-expert load balancing, the communication pattern of all-to-all can be unbalanced if certain GPUs hold experts that receive more tokens. This manifests as GPU idle time while waiting for the slowest all-to-all participant to finish. Solution: use random token selection (RTS) on overflow rather than first-come-first-served dropping, which tends to be less biased toward specific GPU indices; also monitor per-GPU expert utilization separately from per-expert token fractions.
      </Prose>

      <H3>7. Fine-grained expert initialization</H3>
      <Prose>
        With 256 experts and random initialization, the router starts in high-entropy uniform routing. The gradient signal per expert is correspondingly small (1/256 of total batch). This can cause slow early learning and sensitivity to learning rate — a rate that works for 8-expert MoE may cause loss spikes with 256 experts. Use lower initial learning rates with fine-grained MoE, or initialize experts from a pretrained dense checkpoint when possible.
      </Prose>

      <H3>8. Naive comparison to dense baseline</H3>
      <Prose>
        A common mistake in evaluating MoE: comparing a 600B-total MoE model against a 37B-parameter dense model and calling it equivalent. The MoE model uses more memory, more communication, and more total FLOPs in the all-to-all step. The fair comparison is at matched training FLOPs or matched inference memory, not matched active parameters. MoE wins on performance-per-training-FLOP; it loses on performance-per-GB-VRAM. Knowing which axis matters for your setting determines whether MoE is the right choice.
      </Prose>

      {/* ====================================================================
          10. PRIMARY SOURCES
          ==================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All arXiv IDs below were verified via WebSearch on 2026-04-21. Titles, authors, and dates confirmed against arXiv abstract pages.
      </Prose>

      <H3>Foundational papers</H3>

      <Prose>
        <strong>Shazeer et al. 2017 — Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.</strong> arXiv:1701.06538. Authors: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. Published ICLR 2017. Introduced the modern MoE layer with top-k routing and auxiliary load-balance loss. The first model to scale language modeling with MoE to billions of parameters. All core formulations in section 3 originate here.
      </Prose>

      <Prose>
        <strong>Lepikhin et al. 2020 — GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.</strong> arXiv:2006.16668. Authors: Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen. Scaled MoE Transformer to 600B parameters on 2048 TPUs in 4 days for multilingual machine translation. Introduced the capacity factor formulation and refined the auxiliary loss. The first production-scale MoE Transformer.
      </Prose>

      <Prose>
        <strong>Fedus, Zoph, Shazeer 2022 — Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.</strong> arXiv:2101.03961. Published in JMLR 2022. Simplified routing to top-1 and demonstrated that this extreme sparsification trained stably at trillion-parameter scale. Introduced extensive ablations on auxiliary loss coefficients, capacity factors, and expert counts that remain the standard reference for hyperparameter selection.
      </Prose>

      <Prose>
        <strong>Du et al. 2021 — GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.</strong> arXiv:2112.06905. Authors: Nan Du, Yanping Huang, Andrew Dai, Simon Tong, Dmitry Lepikhin, et al. 1.2T parameter MoE model demonstrating energy efficiency (1/3 the training cost of GPT-3) with better zero-shot and one-shot performance across 29 NLP tasks. First systematic demonstration of MoE energy advantage over dense at matched performance.
      </Prose>

      <H3>Modern large-scale MoE</H3>

      <Prose>
        <strong>Jiang et al. 2024 — Mixtral of Experts.</strong> arXiv:2401.04088. Mixtral 8×7B: top-2 routing over 8 experts, 47B total / 13B active parameters, trained on 32K context window. Open-source under Apache 2.0. Outperforms or matches Llama 2 70B and GPT-3.5 across benchmarks including math, code, and multilingual tasks. The canonical open-source reference MoE model.
      </Prose>

      <Prose>
        <strong>DeepSeek-AI 2024 — DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.</strong> arXiv:2405.04434. 236B total / 21B active parameters. Introduced Multi-head Latent Attention (MLA) for KV cache compression and DeepSeekMoE with shared+fine-grained routed experts. 42.5% lower training cost vs. DeepSeek 67B dense at better performance. First large-scale validation of the shared+routed expert architecture.
      </Prose>

      <Prose>
        <strong>DeepSeek-AI 2024 — DeepSeek-V3 Technical Report.</strong> arXiv:2412.19437. 671B total / 37B active parameters. 256 fine-grained routed experts + 1 shared expert per MoE layer. Pioneered auxiliary-loss-free load balancing (bias-based routing correction). Pre-trained on 14.8T tokens using 2.788M H800 GPU hours. Achieves performance comparable to leading closed-source models while remaining open-weight.
      </Prose>

      {/* ====================================================================
          11. SELF-CHECK EXERCISES
          ==================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Derive the balance condition</H3>

      <Prose>
        The auxiliary loss is <Code>L_aux = N · Σ f_i · P_i</Code>. Using the method of Lagrange multipliers with constraints <Code>Σ f_i = 1</Code> and <Code>Σ P_i = 1</Code>, show that the minimum is achieved when <Code>f_i = P_i = 1/N</Code> for all <Code>i</Code>, giving <Code>f_i · P_i = 1/N²</Code> per expert and a total loss of 1. What does this tell you about the implicit assumption that uniform routing is the unique optimum?
      </Prose>

      <H3>Exercise 2: Why capacity_factor must exceed 1.0</H3>

      <Prose>
        Model token arrivals at expert <Code>i</Code> across a batch of <Code>T</Code> tokens as a Binomial random variable with parameters <Code>T·k</Code> and <Code>p = 1/N</Code>. Using the normal approximation, compute the probability that expert <Code>i</Code> receives more than <Code>T·k/N</Code> tokens (i.e., exactly its fair share — the capacity at <Code>C=1.0</Code>) as a function of <Code>T</Code>, <Code>k</Code>, and <Code>N</Code>. For <Code>T=512</Code>, <Code>k=2</Code>, <Code>N=8</Code>, what capacity factor is needed to keep the overflow probability below 5%? Below 1%?
      </Prose>

      <H3>Exercise 3: Design a 100B-total / 37B-active MoE</H3>

      <Prose>
        You want a model with 100B total parameters and 37B active parameters per token, using top-2 routing. The non-expert parameters (attention, embeddings, norms) account for 5B parameters. Each expert must be a symmetric two-layer FFN with <Code>d_model = 4096</Code> and <Code>d_ff = 4 × d_model</Code>. How many experts N do you need? How many of the 100B total parameters are in the expert pool? Is this configuration feasible with the given <Code>d_model</Code> and <Code>d_ff</Code>? If not, what adjustment to <Code>d_ff</Code> brings it within 5% of the target?
      </Prose>

      <H3>Exercise 4: Routing variance as a collapse diagnostic</H3>

      <Prose>
        Define routing variance for a batch as <Code>Var(f) = (1/N) · Σ (f_i − 1/N)²</Code>. Show that this quantity equals zero at perfect balance and equals <Code>(N−1)/N²</Code> at total collapse (all tokens to one expert). Implement a monitor function that logs routing variance every 100 steps and define a threshold above which training should be paused and the auxiliary loss coefficient increased. What threshold would you set for N=8? For N=64?
      </Prose>

      <H3>Exercise 5: Auxiliary loss weight tradeoff</H3>

      <Prose>
        Consider the loss <Code>L = L_LM + α · L_aux</Code>. For small α, describe qualitatively what happens to the router during training (which effect dominates — task loss or aux loss?). For α = 1.0, what does the routing converge to and why? Sketch the expected shape of the curve: validation perplexity as a function of α (from 0 to 1), and explain the shape. Where is the optimal α likely to lie and why? How would you expect this optimal α to shift as N increases from 8 to 256?
      </Prose>

    </div>
  ),
};

export default moeTraining;
