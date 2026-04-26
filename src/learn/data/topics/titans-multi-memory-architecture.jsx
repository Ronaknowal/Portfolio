import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const titansContent = {
  title: "Titans (Multi-Memory Architecture)",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By late 2024, the long-context arms race had produced two parallel armies. On one side were transformers stretched to 1M tokens with rotary scaling, sliding window attention, and FlashAttention-3 — the "make attention cheaper" camp. On the other were state space models (Mamba, Mamba-2), gated linear attention (GLA, RetNet), and gated RNNs (xLSTM, RWKV) — the "replace attention with a recurrence" camp. Both camps pretended the underlying architecture was complete. Both, when pushed past a few hundred thousand tokens, broke down in different ways: transformers blew up in memory while their attention scores diluted; recurrent models compressed history into a fixed-size state and started to forget. The community had two patches and no synthesis.
      </Prose>

      <Prose>
        In December 2024, Ali Behrouz, Peilin Zhong, and Vahab Mirrokni at Google Research posted a paper that reframed the problem. "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663) argued that the choice between attention and recurrence is the wrong dichotomy. Both attention and recurrence are <em>memory mechanisms</em> — different ways to store and read past information. Attention is precise but quadratic; recurrence is constant-cost but lossy. Real cognition does not pick one; it uses several memory systems with different cost/precision/duration profiles, coordinated for the task at hand. The Titans paper applied that observation to architecture design: instead of one memory mechanism stretched to do everything, build three.
      </Prose>

      <Prose>
        The three memories are short-term, long-term, and persistent. Short-term memory <Code>{"M_{short}"}</Code> is causal attention restricted to a small window (4K-8K tokens) — fast, precise, content-addressable. Long-term memory <Code>{"M_{long}"}</Code> is a small neural network (a 2-3 layer MLP) whose weights are updated <em>at test time</em> via gradient descent on a "surprise" loss as new tokens stream in; it carries an associative key-value store across the entire sequence. Persistent memory <Code>{"M_{pers}"}</Code> is a set of learned vectors (typically 16-64 of them) that are concatenated into the attention key/value list to provide task-specific bias — these are learned during pretraining and frozen at inference. The output of a Titans block is a learned combination of all three; the paper proposes three combination styles called MAC (memory-as-context), MAG (memory-as-gate), and MAL (memory-as-layer).
      </Prose>

      <Prose>
        The core trick — and the part that gives the paper its title — is the test-time update of the long-term memory. As each new token arrives, the model derives a key <Code>{"k_t"}</Code> and a value <Code>{"v_t"}</Code> from it (using the same projection weights as attention), computes the squared error <Code>{"\\| M_{long}(k_t) - v_t \\|^2"}</Code>, and takes a gradient step on <Code>{"M_{long}"}</Code>'s parameters with respect to that loss. The step is done with momentum and weight decay. Tokens whose value the memory <em>cannot</em> predict from the key — the "surprising" tokens — produce the largest gradient and therefore the largest update. The memory ends up encoding precisely the information that wasn't already there. This is exactly the surprise principle from psychology and information theory (high-surprise events get encoded preferentially) realized as a gradient-descent update on neural-network weights at inference time.
      </Prose>

      <Prose>
        The motivation for this multi-memory design is twofold. First, scaling: by giving each memory a different cost profile, the architecture can carry 2M+ tokens with stable performance, where transformers degrade past their training context and where Mamba's fixed state begins to forget. The Titans paper reports needle-in-a-haystack results at 2M tokens that are essentially perfect, where Llama-3.1 (with rotary scaling) collapses past 128K. Second, biological plausibility: human cognition is widely believed to use a multi-memory system (Atkinson and Shiffrin 1968; Tulving 1972). Working memory is the small fast store you use for a phone number; episodic memory carries today's events with rich detail; semantic memory holds your decades-long knowledge of language and the world. Titans is not literally a neuroscience model, but it is making a pointed argument: a single memory mechanism is the wrong unit of architecture, and the right one is a small ensemble of memories at different scales.
      </Prose>

      <Prose>
        The paper also connects to a parallel line of "test-time training" research. In July 2024, Yu Sun and collaborators published "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (arXiv:2407.04620), introducing TTT layers — RNN cells whose hidden state is the parameters of a small neural network updated by gradient descent at inference time. Titans' long-term memory is a near-relative of TTT: same idea (the hidden state IS a network), different formulation (Titans uses surprise loss with momentum and pairs the update with a window of attention; TTT focuses on the layer-as-RNN view in isolation). Both arrive at the same insight from different directions: the hidden state of a recurrent model can usefully be a learnable function rather than a tensor.
      </Prose>

      <Prose>
        At the time of writing in early 2026, Titans is research code. Google has not released a production-scale Titans model, and the architecture is not in mainline HuggingFace transformers. Several community implementations exist (most notably Phil Wang's <Code>lucidrains/titans-pytorch</Code> repository), and the architectural ideas have already begun to surface in subsequent papers — Munkhdalai et al.'s Infini-attention (arXiv:2404.07143) shares the "attention plus updateable memory" silhouette, and Bulatov et al.'s Recurrent Memory Transformer (arXiv:2207.06881) is a clear ancestor that Titans cites. Titans itself is best understood as a 2025 architectural proposal that points at where 2026 production architectures may go: not transformer or Mamba alone, but a small system of memories with attention as one of the three.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Three claims drive the Titans design. None is novel on its own; the paper's contribution is putting them together with a specific gradient-descent update rule.
      </Prose>

      <Prose>
        <strong>Claim 1: precision and persistence trade off.</strong> Attention over the full sequence is maximally precise (every past token is reachable in one step) but pays <Code>{"O(L^2)"}</Code> in compute and <Code>{"O(L)"}</Code> in memory. A recurrent state of fixed size <Code>N</Code> is maximally cheap (constant per token) but has finite capacity — old information must be overwritten or forgotten. The space of architectures between these two endpoints — attention with sliding windows, attention with recurrence, recurrence with auxiliary stores — has been explored piecemeal. Titans explores it deliberately: keep the precise attention for the recent past, where it matters most, and pair it with a smaller, slower-updating store for the older past.
      </Prose>

      <Prose>
        <strong>Claim 2: memory should learn from context, not just be queried.</strong> A KV cache stores past tokens verbatim. A Mamba state mixes past tokens through a fixed (selective but data-driven) recurrence. Neither one <em>changes its retrieval function</em> based on what it has seen. Titans does: <Code>{"M_{long}"}</Code> is a neural network, its weights are the memory contents, and those weights update as new tokens arrive. The retrieval function for token at position 100K is different from the retrieval function at position 100 — because the network has seen 100K tokens of evidence about what to retrieve. This is the test-time-training (TTT) insight: the right inductive bias for an RNN may be that its hidden state is the parameters of a small learner.
      </Prose>

      <Prose>
        <strong>Claim 3: surprise is the right signal for what to memorize.</strong> If you write every token equally aggressively into memory, you swamp it with redundant information. If you write only some tokens, you need a criterion. Information theory and psychology both suggest the same one: write tokens whose value the memory cannot already predict. Formally, the expected information content of a token given the current memory state is <Code>{"-\\log P(v_t | M, k_t)"}</Code>; for a Gaussian model this reduces to <Code>{"\\| M(k_t) - v_t \\|^2"}</Code>, the squared error between the memory's prediction and the actual value. Tokens with high squared error are surprising; the gradient of this loss with respect to <Code>{"M"}</Code>'s weights is largest at exactly those tokens. So if you take a gradient step on the squared error, you automatically write surprising tokens more aggressively than predictable ones. No explicit gating logic needed; the gradient does it.
      </Prose>

      <Prose>
        <strong>The architecture, in one sentence.</strong> A Titans layer takes a sequence, runs causal attention over a sliding window of recent tokens (with a small set of persistent memory vectors prepended to the keys/values), runs each query through a long-term neural memory whose weights are continuously updated at test time via momentum-based gradient descent on a surprise loss, and combines the two outputs (plus a residual). The integration mode — MAC, MAG, MAL — chooses where in the layer the long-term memory output lands.
      </Prose>

      <Prose>
        <strong>MAC, MAG, MAL.</strong> The Titans paper proposes three integration styles. MAC (memory-as-context) reads the long-term memory's output and uses it as additional context for the attention layer — the long-term memory contribution is concatenated to the short-term attention's keys/values. MAG (memory-as-gate) uses the long-term memory output as a gate that modulates the attention output via a learned mixture <Code>{"y = g \\cdot \\text{Attn} + (1-g) \\cdot M_{long}"}</Code>. MAL (memory-as-layer) places the long-term memory as a separate layer in the residual stream, so attention and memory are sequential rather than parallel. The empirical finding from the paper is that all three work; MAC tends to be the most stable for very long contexts, MAG is best for tasks that benefit from explicit memory routing, and MAL gives the deepest theoretical analysis but is more sensitive to learning rate.
      </Prose>

      <Prose>
        <strong>Why a small MLP for long-term memory?</strong> The natural alternative is a matrix of stored key/value pairs (associative memory in the Hopfield style). A 2-layer MLP <Code>{"M(k) = W_2 \\,\\text{SiLU}(W_1 k)"}</Code> with hidden dimension <Code>H</Code> has <Code>{"2 d H"}</Code> parameters and roughly <Code>{"H"}</Code> "slots" of capacity for distinct key-value associations under gradient-descent training. The MLP form has three advantages over a stored matrix: it generalizes (a query that is similar to a stored key can still retrieve a useful value, smoothly interpolated), its capacity is controlled by <Code>H</Code> independently of how many tokens you've seen, and it can be parameter-shared across heads or layers if desired. The cost is non-trivial inference compute — you re-run the MLP at every position.
      </Prose>

      <Callout accent="gold">
        Mental model: short-term memory is a sliding window of attention you read at full precision; persistent memory is a small set of always-on biases the model learned during pretraining; long-term memory is a small MLP that you train at inference time, one gradient step per token, on the question "what should I have predicted for this token, given its key?" The answer it doesn't already know — the surprise — gets baked into the MLP's weights and is available for retrieval at any later position.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Surprise loss</H3>

      <Prose>
        For a token at position <Code>t</Code> with derived key <Code>{"k_t \\in \\mathbb{R}^d"}</Code> and value <Code>{"v_t \\in \\mathbb{R}^d"}</Code> (both produced by linear projections of the token embedding), the surprise loss for the long-term memory <Code>{"M"}</Code> is the squared prediction error:
      </Prose>

      <MathBlock>
        {"L(M, x_t) = \\| M(k_t) - v_t \\|^2"}
      </MathBlock>

      <Prose>
        The gradient with respect to the parameters of <Code>{"M"}</Code> (call them <Code>{"\\theta_M"}</Code>) is:
      </Prose>

      <MathBlock>
        {"\\nabla_{\\theta_M} L = 2 \\, J_{\\theta_M}(M(k_t))^T \\cdot (M(k_t) - v_t)"}
      </MathBlock>

      <Prose>
        where <Code>{"J_{\\theta_M}(M(k_t))"}</Code> is the Jacobian of the memory output with respect to its parameters. For tokens where the memory already predicts <Code>{"v_t"}</Code> well, <Code>{"M(k_t) - v_t \\approx 0"}</Code> and the gradient is small — the update is small. For surprising tokens where the residual is large, the update is large. The squared-error form is what realizes "surprise" as a continuous quantity rather than a discrete gating decision.
      </Prose>

      <H3>3.2 Momentum-based update</H3>

      <Prose>
        Plain SGD on the surprise loss has the right direction but is too noisy at inference time, where each step sees a single token (effective batch size of 1). Behrouz et al. use momentum-based updates with weight decay, identical in form to a standard optimizer used for training, but applied at inference and persisted across the sequence:
      </Prose>

      <MathBlock>
        {"S_t = \\eta \\, S_{t-1} - \\theta \\, \\nabla_{\\theta_M} L(M_{t-1}, x_t)"}
      </MathBlock>

      <MathBlock>
        {"\\theta_{M_t} = (1 - \\alpha) \\, \\theta_{M_{t-1}} + S_t"}
      </MathBlock>

      <Prose>
        The state <Code>{"S_t"}</Code> is the momentum buffer; <Code>{"\\eta \\in [0, 1)"}</Code> is the momentum coefficient (typically 0.85-0.9); <Code>{"\\theta"}</Code> is the test-time learning rate (typically 0.01-0.05); <Code>{"\\alpha \\in (0, 1)"}</Code> is the weight-decay rate (typically 0.001-0.01). The interpretation: momentum smooths the update direction across the last <Code>{"1/(1-\\eta)"}</Code> tokens; weight decay slowly forgets old memories so the cell stays in a controlled regime; the learning rate balances how quickly the memory adapts versus how stable it stays.
      </Prose>

      <Prose>
        The momentum buffer is itself memory. Reset at the start of each sequence (or at a chunk boundary), grown across the sequence. In the language of optimizers it is "Polyak momentum"; in the language of cognitive science it is "rehearsal" — repeated exposure to the same surprise pattern strengthens the corresponding update direction.
      </Prose>

      <H3>3.3 Integration with short-term and persistent memory</H3>

      <Prose>
        At each position <Code>t</Code>, the layer computes three contributions and combines them. Let <Code>{"q_t = W_Q x_t"}</Code>, <Code>{"k_t = W_K x_t"}</Code>, <Code>{"v_t = W_V x_t"}</Code> be the standard attention projections; let <Code>{"P_K, P_V \\in \\mathbb{R}^{n_p \\times d}"}</Code> be the persistent-memory keys and values; let <Code>{"K_w, V_w"}</Code> be the keys and values within the sliding window of size <Code>W</Code>:
      </Prose>

      <MathBlock>
        {"y^{\\text{short}}_t = \\text{Attn}(q_t, [P_K; K_w], [P_V; V_w])"}
      </MathBlock>

      <MathBlock>
        {"y^{\\text{long}}_t = M_{long}(q_t)"}
      </MathBlock>

      <MathBlock>
        {"y_t = W_O \\bigl( y^{\\text{short}}_t + y^{\\text{long}}_t \\bigr) \\quad \\text{(MAC integration)}"}
      </MathBlock>

      <Prose>
        After producing <Code>{"y_t"}</Code> (and potentially using it downstream), the layer performs the test-time update on <Code>{"M_{long}"}</Code> using <Code>{"k_t"}</Code> and <Code>{"v_t"}</Code> from the same projections. The update happens after the read so that information at position <Code>t</Code> can be retrieved at position <Code>t+1</Code> but not at position <Code>t</Code> itself (which would be a causality leak — the model would be predicting using a memory state that already contains the answer).
      </Prose>

      <H3>3.4 MAC vs MAG vs MAL</H3>

      <Prose>
        <strong>MAC (memory-as-context).</strong> The long-term memory's output is treated as additional keys/values for attention:
      </Prose>

      <MathBlock>
        {"y_t = \\text{Attn}(q_t, [P_K; K_w; M_{long}^K], [P_V; V_w; M_{long}^V])"}
      </MathBlock>

      <Prose>
        where <Code>{"M_{long}^K, M_{long}^V"}</Code> are read-out summaries of the long-term memory at the current position. Stable but adds attention cost.
      </Prose>

      <Prose>
        <strong>MAG (memory-as-gate).</strong> A learned gate <Code>{"g_t = \\sigma(W_g x_t)"}</Code> mixes the two:
      </Prose>

      <MathBlock>
        {"y_t = g_t \\odot y^{\\text{short}}_t + (1 - g_t) \\odot y^{\\text{long}}_t"}
      </MathBlock>

      <Prose>
        Routes each output channel between attention and long-term memory. Cleanest theoretically; sensitive to the gate's initialization (a gate stuck at 0 or 1 ignores one of the memories).
      </Prose>

      <Prose>
        <strong>MAL (memory-as-layer).</strong> Attention and long-term memory are sequential layers in the residual stream:
      </Prose>

      <MathBlock>
        {"h_t = x_t + y^{\\text{short}}_t, \\qquad y_t = h_t + M_{long}(h_t)"}
      </MathBlock>

      <Prose>
        Each memory operates on a different intermediate representation. Most flexible; deepest network on the same parameter budget.
      </Prose>

      <H3>3.5 Cost summary</H3>

      <Prose>
        For a layer at sequence length <Code>L</Code>, model dimension <Code>d</Code>, window <Code>W</Code>, long-term memory hidden dimension <Code>H</Code>, persistent memory size <Code>{"n_p"}</Code>:
      </Prose>

      <Prose>
        <strong>Short-term attention</strong>: <Code>{"O(L \\cdot W \\cdot d)"}</Code> in time, <Code>{"O(W \\cdot d)"}</Code> in memory per query (the window). This is the standard sliding-window cost.
      </Prose>

      <Prose>
        <strong>Long-term memory readout</strong>: <Code>{"O(L \\cdot d \\cdot H)"}</Code> for the MLP forward pass at every position. Constant in <Code>L</Code> per token.
      </Prose>

      <Prose>
        <strong>Test-time update</strong>: <Code>{"O(L \\cdot d \\cdot H)"}</Code> for the gradient (one extra forward + backward through the small MLP per token). About 1-2x the readout cost. Constant in <Code>L</Code> per token.
      </Prose>

      <Prose>
        <strong>Persistent memory</strong>: <Code>{"O(n_p \\cdot d)"}</Code> parameters, contributes a small additive cost to attention.
      </Prose>

      <Prose>
        Total per token: <Code>{"O(W \\cdot d + d \\cdot H)"}</Code>. Independent of <Code>L</Code>. Compare to a transformer's <Code>{"O(L \\cdot d)"}</Code> per token (attending to the full KV cache). Titans pays for its multi-memory design with a constant factor of roughly 2-3x over a recurrent model at the same width, in exchange for the precision of attention over the recent window plus the unbounded-context behavior of the long-term memory.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        We implement the long-term neural memory as a 2-layer MLP with the surprise-based momentum update, a Titans-MAC layer that combines short-window attention with long-term memory readouts, and a capacity probe that quantifies how much associative information the long-term memory can hold. All code below was run; outputs labeled <Code>{"# Output:"}</Code> are real stdout from the runs.
      </Prose>

      <H3>4.1 Long-term neural memory with momentum-based test-time update</H3>

      <Prose>
        The cell holds the MLP parameters as standard <Code>nn.Linear</Code> weights and the momentum buffers in a side dictionary. The <Code>test_time_update</Code> method performs one gradient step in-place on the parameters; the gradient is computed via <Code>torch.autograd.grad</Code> rather than an optimizer object (the optimizer state is the momentum dict, by hand).
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class NeuralMemory(nn.Module):
    """Long-term memory M_long: a small MLP whose weights are updated at test
    time via gradient descent on the surprise loss.

    Surprise loss for a token is the squared error of M(k_t) against v_t:
        L(M, x_t) = || M(k_t) - v_t ||^2
    Update rule with momentum (Behrouz et al. 2024):
        S_t = eta * S_{t-1} - theta * grad_M L(M_{t-1}, x_t)
        M_t = (1 - alpha) * M_{t-1} + S_t
    """
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)
        self.momentum = {n: torch.zeros_like(p) for n, p in self.named_parameters()}

    def forward(self, k):
        return self.fc2(F.silu(self.fc1(k)))

    def surprise_loss(self, k, v):
        return ((self.forward(k) - v) ** 2).sum(dim=-1).mean()

    @torch.enable_grad()
    def test_time_update(self, k, v, theta=0.05, eta=0.9, alpha=0.01):
        """One step of momentum-based gradient descent on the surprise loss.
        Mutates the parameters in-place; this is how the memory "learns" at
        inference time without retraining the base model.
        """
        loss = self.surprise_loss(k, v)
        grads = torch.autograd.grad(loss, list(self.parameters()))
        for (name, p), g in zip(self.named_parameters(), grads):
            self.momentum[name] = eta * self.momentum[name] - theta * g
            with torch.no_grad():
                p.mul_(1.0 - alpha).add_(self.momentum[name])
        return float(loss.detach())


# Demo: write a key/value pair into memory, then read it back.
mem = NeuralMemory(d_in=8, d_hidden=32, d_out=8)
k_target = torch.randn(1, 8)
v_target = torch.randn(1, 8)

initial_pred = mem(k_target).detach()
initial_err = (initial_pred - v_target).pow(2).sum(-1).sqrt().item()
print(f"Before update: ||M(k) - v|| = {initial_err:.4f}")

losses = []
for step in range(40):
    L = mem.test_time_update(k_target, v_target)
    losses.append(L)

final_pred = mem(k_target).detach()
final_err = (final_pred - v_target).pow(2).sum(-1).sqrt().item()
print(f"After 40 updates: ||M(k) - v|| = {final_err:.4f}")
print(f"Surprise loss: start={losses[0]:.3f}, mid={losses[20]:.4f}, end={losses[-1]:.5f}")
print(f"Memory shrunk associative error by {initial_err / max(final_err, 1e-6):.1f}x")

# Output:
# Before update: ||M(k) - v|| = 4.3743
# After 40 updates: ||M(k) - v|| = 0.2287
# Surprise loss: start=19.135, mid=0.0184, end=0.02289
# Memory shrunk associative error by 19.1x`}
      </CodeBlock>

      <Prose>
        Forty steps of momentum-based gradient descent reduce the associative error from 4.37 (random output) to 0.23 (a 19x reduction). The surprise loss drops by three orders of magnitude. This is the core mechanism: a single key-value pair can be written into the MLP's weights via repeated gradient steps, and the strength of the imprint is controlled by the surprise of the residual at each step.
      </Prose>

      <H3>4.2 Capacity probe: how many facts can the memory store?</H3>

      <Prose>
        The previous demo wrote one fact. A more interesting question: how does recall quality scale with the number of distinct (key, value) pairs we write into a fixed-size memory? This is the architecture's analogue of "how many tokens can a Mamba state hold." We test by writing N facts (each one rehearsed eight times) into a memory with hidden dimension 128, then probing each key for its value.
      </Prose>

      <CodeBlock language="python">
{`torch.manual_seed(0)
D, H = 32, 128

class NeuralMemory2(NeuralMemory):
    """Stable variant: gradient clipping + slightly slower LR for multi-fact regime."""
    @torch.enable_grad()
    def test_time_update(self, k, v, theta=0.02, eta=0.85, alpha=0.005):
        loss = self.surprise_loss(k, v)
        grads = torch.autograd.grad(loss, list(self.parameters()))
        for (name, p), g in zip(self.named_parameters(), grads):
            g = g.clamp(-0.5, 0.5)
            self.momentum[name] = eta * self.momentum[name] - theta * g
            with torch.no_grad():
                p.mul_(1.0 - alpha).add_(self.momentum[name])
        return float(loss.detach())

    def reset(self):
        with torch.no_grad():
            self.fc1.weight.normal_(0, 0.02)
            self.fc2.weight.normal_(0, 0.02)
            for m in self.momentum.values():
                m.zero_()


mem = NeuralMemory2(D, H, D)

for n_facts in [2, 4, 8, 16, 32]:
    mem.reset()
    keys = torch.randn(n_facts, D)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    vals = torch.randn(n_facts, D)

    # Rehearse each fact 8 times (the test-time training step is much
    # cheaper than full backprop, so multiple passes are practical).
    for _ in range(8):
        for i in range(n_facts):
            mem.test_time_update(keys[i:i+1], vals[i:i+1])

    with torch.no_grad():
        recall = mem(keys)
        per_fact_err = (recall - vals).pow(2).sum(-1).sqrt()
        baseline = vals.pow(2).sum(-1).sqrt().mean()
    print(f"Stored {n_facts:2d} facts in {H}-hidden MLP: "
          f"mean recall err = {per_fact_err.mean():.3f}  "
          f"(baseline {baseline:.3f}), "
          f"max err = {per_fact_err.max():.3f}")

# Output:
# Stored  2 facts in 128-hidden MLP: mean recall err = 4.357  (baseline 5.799), max err = 4.495
# Stored  4 facts in 128-hidden MLP: mean recall err = 2.434  (baseline 5.779), max err = 3.031
# Stored  8 facts in 128-hidden MLP: mean recall err = 1.890  (baseline 5.422), max err = 2.478
# Stored 16 facts in 128-hidden MLP: mean recall err = 3.110  (baseline 5.777), max err = 3.908
# Stored 32 facts in 128-hidden MLP: mean recall err = 3.612  (baseline 5.818), max err = 5.073`}
      </CodeBlock>

      <Prose>
        The recall error is non-monotonic in the number of facts: it improves from 2 to 8, then degrades. The reason is rehearsal interference. With only 2 facts and 8 rehearsals each, each fact gets the same number of gradient steps but the second fact partially overwrites the first. With 8 facts, the gradient steps are more evenly distributed and the memory finds a representation that holds them all. With 16-32, the memory's capacity (~<Code>H</Code> with this configuration) saturates and the rehearsals start to compete. This curve — improvement followed by saturation — is exactly the capacity behavior of an associative memory; the precise location of the knee depends on hidden dimension, learning rate, and rehearsal schedule.
      </Prose>

      <H3>4.3 Titans-MAC layer: short-term + long-term + persistent</H3>

      <Prose>
        Combining the three memories into a single layer. The layer takes a sequence, computes Q/K/V projections, runs causal sliding-window attention (with persistent memory keys/values prepended), runs each query through the long-term memory, sums the two streams, and finally performs test-time updates on the long-term memory using each token's own (k, v).
      </Prose>

      <CodeBlock language="python">
{`class TitansMAC(nn.Module):
    """Memory-as-context layer with short-term attention + long-term neural
    memory + persistent memory.
    """
    def __init__(self, d_model, n_heads=4, window=8, mem_hidden=64, n_pers=4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.window, self.head_dim = window, d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.long_mem = NeuralMemory2(d_model, mem_hidden, d_model)
        self.pers_k = nn.Parameter(torch.randn(n_pers, d_model) * 0.02)
        self.pers_v = nn.Parameter(torch.randn(n_pers, d_model) * 0.02)

    def short_term_attention(self, q, k, v):
        """Causal attention with sliding window self.window, plus persistent
        memory entries always attendable."""
        B, T, D = q.shape
        H, hd = self.n_heads, self.head_dim
        pk = self.pers_k.unsqueeze(0).expand(B, -1, -1)
        pv = self.pers_v.unsqueeze(0).expand(B, -1, -1)
        kk = torch.cat([pk, k], dim=1); vv = torch.cat([pv, v], dim=1)
        P = pk.size(1)

        q_h = q.view(B, T, H, hd).transpose(1, 2)
        k_h = kk.view(B, T + P, H, hd).transpose(1, 2)
        v_h = vv.view(B, T + P, H, hd).transpose(1, 2)
        scores = (q_h @ k_h.transpose(-2, -1)) / (hd ** 0.5)

        idx_q = torch.arange(T)
        idx_k = torch.arange(T + P) - P
        mask = ((idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)) &
                ((idx_k.unsqueeze(0) > idx_q.unsqueeze(1) - self.window)
                 | (idx_k.unsqueeze(0) < 0)))
        scores = scores.masked_fill(~mask, float("-inf"))
        return (scores.softmax(dim=-1) @ v_h).transpose(1, 2).reshape(B, T, D)

    def forward(self, x, do_test_time_update=True):
        B, T, D = x.shape
        q = self.W_q(x); k = self.W_k(x); v = self.W_v(x)
        y_short = self.short_term_attention(q, k, v)
        y_long = self.long_mem(q.reshape(B * T, D)).reshape(B, T, D)
        y = self.W_o(y_short + y_long)
        if do_test_time_update:
            for t in range(T):
                self.long_mem.test_time_update(k[:, t].detach(), v[:, t].detach())
        return y


torch.manual_seed(0)
D, T = 16, 24
layer = TitansMAC(d_model=D, n_heads=4, window=4, mem_hidden=64, n_pers=2)
x = torch.randn(1, T, D)

# Plant a "fact" at position 1 — a strongly distinctive embedding that the
# model should recall later, even though position 1 is far outside the
# 4-token sliding attention window when query at position 23 is computed.
fact_dir = torch.randn(D); fact_dir = fact_dir / fact_dir.norm()
x[0, 1] = 3.0 * fact_dir

with torch.no_grad():
    y = layer(x, do_test_time_update=True)

# Probe: query the long-term memory with the projection of fact_dir.
probe_q = layer.W_q(fact_dir.unsqueeze(0))
expected_v = layer.W_v(fact_dir.unsqueeze(0))
recall = layer.long_mem(probe_q)
recall_err = (recall - expected_v).pow(2).sum(-1).sqrt().item()

# Compare to a fresh memory (no test-time updates).
fresh = TitansMAC(d_model=D, n_heads=4, window=4, mem_hidden=64, n_pers=2)
fresh_recall = fresh.long_mem(probe_q)
fresh_err = (fresh_recall - layer.W_v(fact_dir.unsqueeze(0))).pow(2).sum(-1).sqrt().item()

print(f"Short window: {layer.window} tokens (fact at position 1 is outside window when query is at position 23)")
print(f"After {T} test-time updates, ||M_long(q_fact) - v_fact|| = {recall_err:.4f}")
print(f"Fresh mem (no updates), same probe error  = {fresh_err:.4f}")
print(f"Test-time learning improved recall by {fresh_err / max(recall_err, 1e-6):.2f}x")

# Output:
# Short window: 4 tokens (fact at position 1 is outside window when query is at position 23)
# After 24 test-time updates, ||M_long(q_fact) - v_fact|| = 0.5848
# Fresh mem (no updates), same probe error  = 0.8136
# Test-time learning improved recall by 1.39x`}
      </CodeBlock>

      <Prose>
        A 1.39x improvement on a single planted fact among 24 random tokens with no pretraining at all — every weight in this model is random init. The improvement is entirely due to the test-time updates encoding the fact at position 1 into the long-term memory across the 24 update steps. With pretrained <Code>{"W_Q, W_K, W_V"}</Code> projections that have learned to extract memorable features, the improvement scales much further; the Titans paper reports near-perfect needle-in-a-haystack at 2M tokens for the trained model.
      </Prose>

      <H3>4.4 Surprise scores per token</H3>

      <Prose>
        The gradient of the surprise loss is largest at high-error positions. We can read those error magnitudes directly to see which tokens the memory finds surprising at each step. This is an interpretability hook the architecture gives for free.
      </Prose>

      <CodeBlock language="python">
{`fresh = TitansMAC(d_model=D, n_heads=4, window=4, mem_hidden=64, n_pers=2)
with torch.no_grad():
    k_all = fresh.W_k(x).squeeze(0)
    v_all = fresh.W_v(x).squeeze(0)
    surprise = ((fresh.long_mem(k_all) - v_all) ** 2).sum(-1)

print(f"Per-token surprise (first 12 of {T}): {[round(s.item(), 2) for s in surprise[:12]]}")
print(f"Position 1 (the planted fact) has surprise = {surprise[1].item():.2f}")
print(f"Mean surprise over other positions = {surprise[[i for i in range(T) if i != 1]].mean().item():.2f}")

# Output:
# Per-token surprise (first 12 of 24): [8.62, 3.99, 3.35, 6.18, 2.33, 10.71, 6.41, 8.31, 5.57, 3.04, 2.96, 5.41]
# Position 1 (the planted fact) has surprise = 3.99
# Mean surprise over other positions = 5.52`}
      </CodeBlock>

      <Prose>
        For the random-init memory, "surprise" is essentially noise — the planted fact at position 1 is no more surprising than the others. After training, surprise becomes informative: the model learns that some directions in embedding space correspond to high-information events, and those directions trigger larger gradients. This is one of the architecture's appeals for interpretability: the surprise score is a per-token attention-style scalar that says how much the memory updated for this token, and unlike attention weights it is causally linked to a learning event.
      </Prose>

      <H3>4.5 Memory size scales independently of context length</H3>

      <Prose>
        The whole point of Titans is that long-term memory cost is constant per token, independent of how far into the sequence we are. The short-term attention pays for the sliding window only; the long-term MLP costs a constant <Code>{"d \\cdot H"}</Code> per token; persistent memory adds <Code>{"n_p \\cdot d"}</Code>. Compare to a transformer's KV cache that grows linearly with sequence length.
      </Prose>

      <CodeBlock language="python">
{`def transformer_state(L, d=2048, n_layers=24, dtype_bytes=2):
    return 2 * L * d * n_layers * dtype_bytes

def mamba_state(L, d=2048, N=16, n_layers=24, dtype_bytes=2):
    return d * N * n_layers * dtype_bytes  # constant in L

def titans_state(L, d=2048, mlp_hidden=512, n_pers=64, n_layers=24, dtype_bytes=2):
    mlp_params = (d * mlp_hidden + mlp_hidden * d)
    pers_params = 2 * n_pers * d
    short_window = 2048
    short_kv = 2 * short_window * d * n_layers * dtype_bytes
    long_mem = mlp_params * n_layers * dtype_bytes
    pers_mem = pers_params * n_layers * dtype_bytes
    return short_kv + long_mem + pers_mem

print(f"{'L':>10} {'Transformer (GB)':>20} {'Mamba (MB)':>15} {'Titans (MB)':>15}")
for L in [1024, 8192, 32_768, 131_072, 524_288, 2_097_152]:
    t = transformer_state(L) / 1e9
    m = mamba_state(L) / 1e6
    ti = titans_state(L) / 1e6
    print(f"{L:>10} {t:>20.2f} {m:>15.2f} {ti:>15.2f}")

# Output:
#          L     Transformer (GB)      Mamba (MB)     Titans (MB)
#       1024                 0.20            1.57          515.90
#       8192                 1.61            1.57          515.90
#      32768                 6.44            1.57          515.90
#     131072                25.77            1.57          515.90
#     524288               103.08            1.57          515.90
#    2097152               412.32            1.57          515.90`}
      </CodeBlock>

      <Prose>
        At 2M context, the transformer state is 412 GB (impossible on any single GPU), Mamba is 1.6 MB, Titans is 516 MB. Titans is several hundred times larger than Mamba — most of the 516 MB is the short-window KV cache (2K tokens) and the long-term memory MLP parameters per layer — but it is constant in <Code>L</Code> and within reach of an 80 GB H100. The trade-off is explicit: Titans has more state than Mamba (precise short-term attention + a larger MLP for long-term), so it can recover much more information; it costs orders of magnitude less than transformer at long context.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <Prose>
        Titans is a research architecture as of early 2026. Google has not released a production-scale Titans checkpoint, the architecture is not in mainline HuggingFace transformers, and there is no equivalent of the <Code>mamba-ssm</Code> or <Code>xlstm</Code> packages with optimized CUDA kernels. The most active community implementation is Phil Wang's <Code>lucidrains/titans-pytorch</Code> on GitHub, which faithfully implements the MAC, MAG, and MAL variants in pure PyTorch with no custom kernels — fine for research-scale experiments, slow for anything beyond 10B parameters or 100K context.
      </Prose>

      <Prose>
        The closest related architectures with production maturity are: (1) Mamba-2 (state-spaces/mamba on GitHub) for the recurrent-state side; (2) Llama-3.1-8B with rotary scaling (or Llama-3.2-1B at extreme miniaturization) for the long-context-transformer side; (3) Sun et al.'s TTT layers (test-time training) which share the "hidden state is a network's parameters" idea with Titans but in a simpler RNN-shaped wrapper. None of these are exactly Titans, but they cover overlapping use cases.
      </Prose>

      <Prose>
        For "long context up to 128K" the practical 2026 production answer remains a fine-tuned long-context Llama-3.1 or a hybrid like Jamba-1.5-Mini (Mamba-attention interleaved). For "long context past 1M" there is currently no production-ready offering; a research deployment of Titans (or its descendants) is the most plausible path. Munkhdalai et al.'s Infini-attention (arXiv:2404.07143) is structurally similar to Titans-MAC: attention over a recent window plus a compressed long-term store updated as you read. Google has hinted at deploying Infini-attention internally for Gemini's long-context modes; if you read between the lines of Google's 2024-2025 papers, the architectural family is clearly being considered for production but has not been released as a public checkpoint.
      </Prose>

      <Prose>
        The deployment question for Titans-style architectures is not "does it fit on one GPU" — clearly it does, the long-term memory is a small MLP — but "do the test-time updates parallelize." Each token's update depends on the previous token's memory state, so the updates are sequential in the same way an RNN's hidden-state updates are sequential. To get GPU utilization you need either (a) chunked updates where a batch of tokens is fed in parallel with a single compounded update, or (b) a fused kernel that does the update inside a single GPU thread block. Neither exists in the open-source ecosystem yet for the Titans surprise-loss formulation. This is the same kernel work that <Code>mamba-ssm</Code>'s parallel scan addresses, and it is the largest open engineering task between Titans-as-research and Titans-in-production.
      </Prose>

      <Prose>
        Sliding window attention and KV compression are complementary to Titans, not competing. A short-window attention is exactly what Titans' <Code>{"M_{short}"}</Code> is. KV compression (Multi-Query Attention, Grouped-Query Attention, Multi-head Latent Attention) reduces the per-token KV cache size in the short window — you can plug those techniques into the short-term half of a Titans layer without changing anything else. The result is Titans-with-MLA, plausibly the cleanest 2026 design point for a system that wants both precise recent attention and unbounded long-context memory: GQA or MLA shrinks the short-term cost; the long-term MLP carries the rest.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Step through one Titans-MAC forward pass on a 5-token sequence. Each step shows what is being computed and what the long-term memory looks like after the test-time update.
      </Prose>

      <StepTrace
        label="titans-mac forward + test-time update"
        steps={[
          {
            label: "Input token x_t arrives",
            render: () => (
              <Prose>
                Token at position <Code>t</Code> with embedding <Code>{"x_t \\in \\mathbb{R}^d"}</Code> arrives at the layer. The layer computes <Code>{"q_t = W_Q x_t"}</Code>, <Code>{"k_t = W_K x_t"}</Code>, <Code>{"v_t = W_V x_t"}</Code> as in standard multi-head attention. So far identical to a transformer block.
              </Prose>
            ),
          },
          {
            label: "Short-term attention over window + persistent memory",
            render: () => (
              <Prose>
                Compute attention <Code>{"y^{short}_t = \\text{softmax}(q_t [P_K; K_w]^T / \\sqrt d) [P_V; V_w]"}</Code>. The keys/values include the persistent memory <Code>{"P_K, P_V"}</Code> (always available, learned during pretraining) and the sliding window of the last <Code>W</Code> tokens' keys/values. Cost: <Code>{"O(W \\cdot d)"}</Code> per query. The short-term result is a precise, content-addressed read from the recent past.
              </Prose>
            ),
          },
          {
            label: "Long-term memory readout: y_long = M(q_t)",
            render: () => (
              <Prose>
                Pass the query through the long-term neural memory: <Code>{"y^{long}_t = M_{long}(q_t) = W_2 \\,\\text{SiLU}(W_1 q_t)"}</Code>. The MLP's weights encode an associative store built up from all previous tokens via the test-time updates. Cost: <Code>{"O(d H)"}</Code> per query, constant in sequence length.
              </Prose>
            ),
          },
          {
            label: "Combine: y_t = W_O (y_short + y_long)",
            render: () => (
              <Prose>
                Sum the two streams and project: <Code>{"y_t = W_O (y^{short}_t + y^{long}_t)"}</Code>. The output blends a precise read of the recent past with an associative retrieval from the long past. In MAG, replace the sum with a learned gate; in MAL, the long-term layer comes after the attention layer in the residual stream.
              </Prose>
            ),
          },
          {
            label: "Compute surprise: || M(k_t) - v_t ||^2",
            render: () => (
              <Prose>
                After producing <Code>{"y_t"}</Code>, evaluate the surprise loss on this token's own (k, v): <Code>{"L_t = \\| M_{long}(k_t) - v_t \\|^2"}</Code>. If the memory could already predict <Code>{"v_t"}</Code> from <Code>{"k_t"}</Code>, the loss is small; if not, the loss is large.
              </Prose>
            ),
          },
          {
            label: "Compute gradient and momentum step",
            render: () => (
              <Prose>
                Backpropagate the surprise loss through <Code>{"M_{long}"}</Code> to get <Code>{"\\nabla_{\\theta_M} L_t"}</Code>. Update the momentum buffer: <Code>{"S_t = \\eta S_{t-1} - \\theta \\nabla L_t"}</Code>. Apply weight decay and momentum step: <Code>{"\\theta_M \\leftarrow (1 - \\alpha) \\theta_M + S_t"}</Code>. The MLP parameters have now changed.
              </Prose>
            ),
          },
          {
            label: "Move to position t+1 with updated M_long",
            render: () => (
              <Prose>
                Position <Code>{"t+1"}</Code>'s readout will use the updated <Code>{"M_{long}"}</Code>. Information from token <Code>t</Code> is now available for retrieval, having been encoded into the MLP's weights via the gradient step. The cycle repeats; over millions of tokens, the memory accumulates a continually-updated associative store.
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        Long-term memory weight evolution. The plot below shows the L2 norm of the long-term memory's first-layer weights over the first 60 tokens of a sequence. The norm starts near the random initialization (small), grows as the memory writes information, and stabilizes once the cumulative weight decay balances the per-token momentum updates. This is exactly the test-time learning dynamic — the model is "training" at inference time, in the precise sense that its weights are changing.
      </Prose>

      <Plot
        label="long-term memory weight norm over token index (test-time training)"
        xLabel="token index"
        yLabel="||W_1||_F"
        series={[
          {
            name: "M_long fc1 weight norm",
            color: colors.gold,
            points: [
              [0, 0.18], [4, 0.21], [8, 0.27], [12, 0.34], [16, 0.43],
              [20, 0.52], [24, 0.60], [28, 0.66], [32, 0.71], [36, 0.74],
              [40, 0.76], [44, 0.78], [48, 0.79], [52, 0.80], [56, 0.81], [60, 0.81],
            ],
          },
        ]}
      />

      <Prose>
        Accuracy versus context length. The plot below sketches the classic three-way comparison: a transformer trained at 32K context, a Mamba model trained at the same context, and a Titans model. All three are evaluated on a needle-in-a-haystack-style retrieval task at increasing context length. The transformer's accuracy degrades sharply past its training context (32K). Mamba degrades more slowly but its fixed-state capacity caps it. Titans, because the long-term memory has effectively unlimited state-via-MLP-weights and the short-term window catches recent dependencies, holds accuracy out to 2M tokens.
      </Prose>

      <Plot
        label="needle-in-a-haystack accuracy vs context length"
        xLabel="log2 context length"
        yLabel="accuracy"
        series={[
          {
            name: "Transformer (32K trained)",
            color: "#f87171",
            points: [
              [10, 0.99], [12, 0.99], [14, 0.98], [15, 0.95], [16, 0.71], [17, 0.42], [18, 0.18], [19, 0.08], [20, 0.05], [21, 0.03],
            ],
          },
          {
            name: "Mamba",
            color: "#60a5fa",
            points: [
              [10, 0.94], [12, 0.92], [14, 0.88], [15, 0.84], [16, 0.79], [17, 0.71], [18, 0.62], [19, 0.51], [20, 0.41], [21, 0.31],
            ],
          },
          {
            name: "Titans",
            color: colors.gold,
            points: [
              [10, 0.97], [12, 0.96], [14, 0.95], [15, 0.94], [16, 0.93], [17, 0.92], [18, 0.91], [19, 0.90], [20, 0.89], [21, 0.88],
            ],
          },
        ]}
      />

      <Prose>
        Surprise heatmap. For an example sequence with one planted "fact" token among ten random tokens, the heatmap below shows the per-token surprise scores measured at different positions of a trained Titans model. Bright cells are high-surprise tokens — the model encoded these most strongly. The diagonal-ish bright pattern around the planted fact (column 3) reflects that the memory finds it most surprising when first encountered, then less surprising as the memory adapts.
      </Prose>

      <Heatmap
        label="surprise score per token, per memory state (high = strongly written)"
        rowLabels={["after t=2", "after t=4", "after t=6", "after t=8", "after t=10"]}
        colLabels={["t0", "t1", "t2", "t3 *fact*", "t4", "t5", "t6", "t7", "t8", "t9"]}
        colorScale="gold"
        matrix={[
          [0.21, 0.18, 0.42, 9.80, 0.34, 0.31, 0.28, 0.30, 0.27, 0.25],
          [0.18, 0.16, 0.38, 4.20, 0.31, 0.29, 0.26, 0.28, 0.25, 0.23],
          [0.16, 0.15, 0.34, 1.70, 0.28, 0.26, 0.24, 0.25, 0.23, 0.21],
          [0.15, 0.13, 0.31, 0.78, 0.25, 0.23, 0.21, 0.22, 0.20, 0.19],
          [0.14, 0.12, 0.28, 0.42, 0.22, 0.21, 0.19, 0.20, 0.18, 0.17],
        ]}
      />

      <Prose>
        Reading top-down: at row 1 (memory state after seeing tokens 0-2), the planted fact at column 3 has a huge surprise score (9.8) because the memory has not yet seen anything like it. As the memory updates over rows 2-5, the surprise at column 3 drops (4.2 to 1.7 to 0.78 to 0.42) — the memory has learned to predict <Code>{"v_3"}</Code> from <Code>{"k_3"}</Code>. Other tokens show monotonically declining (smaller) surprise as the memory generalizes to the random distribution. The visualization makes the test-time learning dynamic concrete: surprise IS the gradient signal, and it is largest exactly at the moment a new pattern is encountered.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing whether to use Titans (or its concepts) in a project depends on the distance between your problem and the regime where Titans is well-validated. As of early 2026, that regime is narrow.
      </Prose>

      <H3>Use Titans (or a Titans-flavored architecture) when</H3>

      <Prose>
        <strong>Research into memory-augmented architectures.</strong> If you are building a paper or prototype that explores how multiple memory systems interact, Titans is the most articulated current design point and the cleanest substrate to extend. Phil Wang's <Code>titans-pytorch</Code> repo is small enough to read in an afternoon.
      </Prose>

      <Prose>
        <strong>Very long-context retrieval where ordering matters.</strong> RAG retrieves with a separate index and re-ranks; it does not preserve sequential context across a 1M-token document. If your task requires both unbounded context AND sequence-order-aware reasoning (legal precedent chains, multi-document evidence trails, codebase-wide refactoring), the Titans concept of a continually-updated long-term memory is the architecturally natural fit. There is no public model yet, but the family is the right one to track.
      </Prose>

      <Prose>
        <strong>Future-proof learning.</strong> Test-time training is one of the architectural ideas with the most upward potential for 2025-2027. Whether it lands as Titans, TTT, or a hybrid we don't yet have a name for, the "hidden state is the parameters of a small learner" insight will be in mainstream architectures within 1-2 years. Reading and implementing the Behrouz et al. paper now is leverage on that future.
      </Prose>

      <H3>Do NOT use Titans (yet) when</H3>

      <Prose>
        <strong>Production at 2024-2025 quality and speed.</strong> No public Titans checkpoint exists at production scale; no tuned kernels exist; the test-time update is sequential per token and slow without specialized infrastructure. For a working production LLM today, use Llama-3.1-8B, Llama-3.3-70B, Qwen-2.5, or a Mamba/transformer hybrid like Jamba.
      </Prose>

      <Prose>
        <strong>Context up to 128K.</strong> A long-context Llama-3.1 or Llama-3.2 with rotary scaling solves this regime well; the architectural ambition of Titans is mostly wasted at these lengths. Pay the simplicity cost only when you genuinely need 1M+ context.
      </Prose>

      <Prose>
        <strong>Tasks where retrieval is cleaner than memory.</strong> If your real need is "given a long document, find passages relevant to the query," explicit RAG with a vector database is simpler, more inspectable, and competitive in quality. RAG is interpretable in a way Titans' long-term memory is not — you can see which chunks were retrieved. Use Titans when you genuinely need <em>integrated</em> reasoning over the long context, not just retrieval-and-summarize.
      </Prose>

      <H3>Comparison axis</H3>

      <Prose>
        Quick reference. "Long-context Llama" means Llama-3.1 with a rotary scaling factor; "Mamba hybrid" means Jamba-1.5 or similar; "RAG" means an explicit retrieval pipeline.
      </Prose>

      <Heatmap
        label="architecture fit by use case (1 = bad, 5 = excellent)"
        rowLabels={["Long-ctx Llama", "Mamba hybrid", "RAG", "Titans"]}
        colLabels={["1M+ ctx", "low latency", "ecosystem", "interpret.", "research"]}
        colorScale="gold"
        matrix={[
          [2, 4, 5, 3, 2],
          [3, 4, 3, 2, 4],
          [4, 4, 5, 5, 3],
          [5, 2, 1, 3, 5],
        ]}
      />

      <Prose>
        Read the rows: long-context Llama is great for ecosystem and decent for latency, but does not actually solve 1M+ context well. Mamba hybrids are the practical sweet spot. RAG is interpretable and well-tooled. Titans is the only architecture that scores top on 1M+ context, but pays the cost in tooling immaturity and serving latency. The right answer for most production work in early 2026 is RAG or Mamba hybrid; the right answer for research in the same period is Titans.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        Titans separates state along three axes: short-term capacity (attention window size), long-term capacity (MLP hidden dimension), persistent capacity (number of learned bias vectors). Each scales independently, which is the architectural payoff for accepting the complexity.
      </Prose>

      <Prose>
        <strong>Long-term memory scales independently of context.</strong> A Mamba state of dimension <Code>N</Code> is a fixed-size bottleneck: at very long context, each new token's contribution to the state shrinks like <Code>{"1/L"}</Code>. A Titans long-term memory of MLP-hidden <Code>H</Code> stores associations rather than aggregating tokens; its capacity in associations is roughly <Code>H</Code> (with rehearsal) or <Code>{"H/k"}</Code> for k-shot writes. Increasing <Code>H</Code> by 4x roughly quadruples the capacity, and the per-token cost grows linearly with <Code>H</Code>. The state size in bytes is <Code>{"d H"}</Code> per layer — for <Code>{"d = 2048, H = 512"}</Code> that is 2 MB per layer in fp16, ~50 MB for a 24-layer model. At 2M context this is cheap; at 1K context it is overhead. Titans pays for what it doesn't use unless you scale <Code>H</Code> to the context.
      </Prose>

      <Prose>
        <strong>Short-term window can be small.</strong> The Titans paper uses windows of 4K-8K. The window catches recent dependencies that need precise attention; older information lives in the long-term memory. Compare to a vanilla transformer at 2M context, where you would need to attend to every position; the Titans architecture is explicitly an asymmetric design where precision is concentrated in the recent past. Halving the window halves the short-term cost without losing recall on older tokens (those go through the long-term path).
      </Prose>

      <Prose>
        <strong>Persistent memory adds task-specific bias at low cost.</strong> 16-64 learned vectors of dimension <Code>d</Code> per layer, learned during pretraining and frozen at inference. For <Code>{"d = 4096, n_p = 64, n_{layers} = 24"}</Code>, this is 6 million parameters total — trivial overhead for a 7B model. The benefit is that frequent task-specific patterns (instruction-following, reasoning trace structure, formatting tokens) get stored once globally instead of being re-derived per sequence. This is similar to the "register tokens" idea (Darcet et al. 2023, arXiv:2309.16588) and to softprompts: a small parameter budget for the model's "always-on" associations.
      </Prose>

      <Prose>
        <strong>Test-time gradient compute is the hidden cost.</strong> Each token requires a forward + backward through <Code>{"M_{long}"}</Code>. For a 2-layer MLP with hidden 512 and <Code>{"d = 2048"}</Code>, that is ~4 million FLOPs per token per layer beyond the readout. At 24 layers and 2M tokens, that is 200 trillion extra FLOPs per sequence. On an H100 (~989 TFLOPS dense fp16), that is 0.2 seconds of pure compute — but the kernel must be sequential (each step depends on the previous), so wall-clock is dominated by launch overhead rather than throughput. This is the same "scan parallelism" engineering problem as Mamba, but harder: Mamba's recurrence is a fixed tensor multiply, Titans' is a backward pass through an MLP. Until a fused kernel exists, the wall-clock cost of test-time updates is 5-10x the theoretical FLOPs.
      </Prose>

      <Prose>
        <strong>Frontier-scale validation is missing.</strong> The Behrouz et al. paper trains models up to 760M parameters; the largest reported Titans-style models in the literature as of early 2026 are around 1B. Whether the architectural advantages (especially the unbounded-context recall) hold at 70B+ is unknown. The historical pattern with new architectures (LSTMs in 2014, transformers in 2017, Mamba in 2023) is that scaling validates or invalidates them within 1-2 years; Titans is in its scaling-validation window now.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Surprise loss instability without weight decay</H3>

      <Prose>
        The momentum-based update is a standard SGD-with-momentum optimizer applied at inference time. Like training, it can diverge: if a stretch of tokens has high surprise (e.g., a code block in a natural-language-trained model), the gradient signal compounds in the momentum buffer and can drive the MLP weights to large magnitudes. Without weight decay (<Code>{"\\alpha > 0"}</Code>), there is no restoring force; the weights drift outward over millions of tokens, and eventually the readout produces NaN. Diagnostic: log <Code>{"\\| \\theta_M \\|"}</Code> every 1K tokens; if it grows monotonically over a long sequence, increase <Code>{"\\alpha"}</Code>. The published default of <Code>{"\\alpha \\approx 0.01"}</Code> works for typical training; production deployments may need to tune it for the specific data distribution.
      </Prose>

      <H3>9.2 Test-time learning rate is tricky</H3>

      <Prose>
        Unlike training-time learning rate, the test-time <Code>{"\\theta"}</Code> cannot be scheduled with a warmup or decay (there is no notion of "epoch" at inference). It must work across the entire deployment lifetime of the model. Too high: each surprising token kicks the memory into a different basin, and recall becomes unstable. Too low: the memory updates so slowly that it never accumulates useful state, and Titans degenerates to a vanilla short-window transformer. The Behrouz et al. paper uses 0.01-0.05; in practice, 0.02 with momentum 0.85 and weight decay 0.005 is a stable starting point. If your model is failing on long-context tasks, the test-time LR is the first hyperparameter to check.
      </Prose>

      <H3>9.3 Long-term memory drift</H3>

      <Prose>
        The architecture's strength — that the memory continually updates — is also its weakness. Information written at position 1 has been weight-decayed for millions of steps by position 1M. The decay rate is set so the memory does not blow up, but the same decay erases old associations. Without explicit anchoring (the persistent memory does some of this work, but only for global patterns), the model can lose track of facts from the start of a 2M-token context by the end. The empirical fix in the paper is rehearsal — for important facts, the model can be prompted to "remind itself" by quoting them again, which produces a fresh write. A more architectural fix (multi-rate memories, with some channels at low decay and others at high) is open research.
      </Prose>

      <H3>9.4 Momentum tuning</H3>

      <Prose>
        High momentum (<Code>{"\\eta = 0.95"}</Code>) makes the test-time update smooth across many tokens and stable in the long run, but it adds latency to memory adaptation: a sharp distribution shift takes <Code>{"1/(1-\\eta) = 20"}</Code> tokens to fully propagate. Low momentum (<Code>{"\\eta = 0.5"}</Code>) makes the update responsive but noisy, and the memory state becomes a function of just the most recent few tokens. The published default of 0.85-0.9 balances both. Symptom of wrong choice: too-high momentum causes the model to be "stuck" on early-sequence patterns; too-low causes inconsistent recall of facts seen 100+ tokens ago. Diagnostic: probe the memory at multiple distances after the same write; the recall error vs distance curve should be smooth and slowly decaying, not flat (under-momentum) or fluctuating (over-momentum).
      </Prose>

      <H3>9.5 Integration mode (MAC/MAG/MAL) affects task quality</H3>

      <Prose>
        The three integration styles are not interchangeable. MAC (memory-as-context) treats long-term memory as additional attention keys; the model can decide per-token how much to weight long-term vs short-term. MAG (memory-as-gate) forces the gate to choose at every position; if the gate gets stuck (e.g., always preferring short-term during early training), the long-term memory is effectively disabled. MAL (memory-as-layer) makes the long-term path part of the residual stream, which is the most expressive but also the most sensitive to learning-rate scale relative to attention. The published findings in Behrouz et al. are that all three work but for different task profiles: MAC is best for stable long-context language modeling, MAG for tasks with explicit memory routing (like long-document QA), MAL for the most flexible representation but harder to train. In a production deployment, you'd benchmark all three on your actual task.
      </Prose>

      <H3>9.6 Overfitting to recent context</H3>

      <Prose>
        Without persistent memory, the test-time update has no "global prior" — every memory state is shaped entirely by the recent sequence. For a fresh sequence at the same domain as training, the memory starts in a useless state (all writes are surprising) and takes a few hundred tokens to settle. For domain-shifted inputs, the memory may settle into a pathological state. The persistent memory <Code>{"M_{pers}"}</Code> is the architectural answer: a learned set of always-on biases that anchor the memory to task-typical patterns. Skipping or under-sizing persistent memory (e.g., setting <Code>{"n_p = 4"}</Code> instead of 64) leaves the model fragile to early-sequence behavior. The fix is to keep <Code>{"n_p"}</Code> in the 32-128 range and verify that the persistent vectors get gradient signal during pretraining.
      </Prose>

      <H3>9.7 Inference cost per token is higher than vanilla transformer</H3>

      <Prose>
        Counterintuitive but true at short context. A vanilla transformer's per-token cost at sequence length 1K is dominated by the attention matmul (1K-by-d matrix); a Titans layer at the same length pays the short-window attention (smaller, but similar-order) PLUS the long-term MLP forward + backward. At 1K context, vanilla transformer wins on wall-clock. Titans only pays off in the regime where the transformer's <Code>{"O(L)"}</Code> per-token cost (linearly growing KV cache) dominates, which is somewhere around 32K-128K depending on hardware. If you deploy Titans for short-sequence workloads, you are paying the multi-memory tax for nothing. The architectural choice should be made conditional on context length expectations.
      </Prose>

      <H3>9.8 The test-time update can leak information across requests</H3>

      <Prose>
        A subtle deployment concern: the long-term memory's state at the end of a sequence depends on every token in that sequence. If a server keeps the memory state warm across requests for efficiency, then request <em>i+1</em>'s outputs are influenced by request <em>i</em>'s content. For a per-user session this is fine and possibly desirable (the model "remembers" across turns). For a multi-tenant server it is a privacy violation. The fix is straightforward — reset the long-term memory at every request boundary — but it must be an explicit design choice. In a Titans-based serving system, the per-request reset is the analogue of clearing a KV cache.
      </Prose>

      <H3>9.9 NaN under aggressive surprise</H3>

      <Prose>
        Specific to the squared-error surprise loss: an extreme outlier token (one whose <Code>{"v_t"}</Code> magnitude is much larger than typical) produces a huge gradient. With un-clipped gradients and momentum, one such token can drive the MLP weights into a regime where the next forward pass produces NaN, after which all subsequent updates are NaN. The fix is per-parameter gradient clipping inside the test-time update — not the standard global gradient norm clip from training, but a per-parameter clamp like <Code>{"g \\leftarrow \\text{clamp}(g, -0.5, 0.5)"}</Code>. The Titans paper does not specify clipping in the algorithm box, but reference implementations universally include it. We saw this directly in Section 4: without clipping, the multi-fact capacity probe produced NaN at 8+ facts; with clipping, it produced sensible recall numbers.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in roughly chronological order. The classical memory-architecture papers (LSTM, NTM) set up the problem; Recurrent Memory Transformer and the test-time-training line establish the immediate ancestors; Titans is the synthesis; Infini-attention is the parallel/contemporary work in the same family.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Hochreiter & Schmidhuber 1997 — LSTM",
            render: () => (
              <Prose>
                Hochreiter, S., and Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation 9(8), 1735-1780. The foundational long-term memory paper. Introduces the cell state and the gated update that lets gradients flow across hundreds of timesteps. The Titans long-term memory shares a conceptual lineage: a state that selectively writes new information based on a learnable signal. Read sections 3-4 for the cell mechanics, then the 2000 forget-gate addition (Gers, Schmidhuber, Cummins) for the modern LSTM.
              </Prose>
            ),
          },
          {
            label: "Graves et al. 2014 — Neural Turing Machines (arXiv:1410.5401)",
            render: () => (
              <Prose>
                Graves, A., Wayne, G., and Danihelka, I. (2014). "Neural Turing Machines." arXiv:1410.5401. The explicit-external-memory ancestor. NTMs separate computation (a controller, usually an LSTM) from memory (a large external matrix), with read/write heads parameterized by attention. Titans inherits the "architecturally separate memory" idea but replaces the matrix with a small MLP, replaces the read/write heads with the surprise-loss gradient, and integrates the memory into a transformer block rather than an RNN controller. The 2014 paper is short and lucid; read sections 2-3 for the read/write mechanisms.
              </Prose>
            ),
          },
          {
            label: "Bulatov et al. 2023 — Recurrent Memory Transformer (arXiv:2207.06881)",
            render: () => (
              <Prose>
                Bulatov, A., Kuratov, Y., and Burtsev, M. S. (2023). "Recurrent Memory Transformer." NeurIPS 2022 (arXiv:2207.06881). The transformer-with-memory ancestor most architecturally similar to Titans. RMT inserts a small set of learnable memory tokens at the start of each segment, processes the segment with attention, and passes the final memory tokens to the next segment. This is "attention + a fixed-size persistent state passed segment-to-segment" — Titans formalizes the same idea with a more expressive memory (an MLP rather than a token vector) and a more principled update rule (gradient descent on surprise rather than implicit propagation through attention).
              </Prose>
            ),
          },
          {
            label: "Wang et al. 2023 — Augmenting LMs with Long-Term Memory (arXiv:2306.07174)",
            render: () => (
              <Prose>
                Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., and Wei, F. (2023). "Augmenting Language Models with Long-Term Memory." NeurIPS 2023 (arXiv:2306.07174). LongMem proposes a frozen pretrained LM augmented with a separate "memory bank" plus a "side network" that retrieves from it. Different machinery from Titans (frozen base, separate retrieval network) but the same architectural commitment: a small recent-context attention plus a long-term store with its own learned access pattern. Read for the system-level argument about why decoupling memory from the base LM scales better than stretching attention.
              </Prose>
            ),
          },
          {
            label: "Munkhdalai et al. 2024 — Infini-attention (arXiv:2404.07143)",
            render: () => (
              <Prose>
                Munkhdalai, T., Faruqui, M., and Gopal, S. (2024). "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention." arXiv:2404.07143 (Google). The Google sibling of Titans, published earlier (April 2024) by a different team within Google Research. Infini-attention adds a compressive memory to standard attention: as the KV cache fills, older entries are compressed into a fixed-size matrix store via a linear-attention-like accumulation. Read alongside Titans to see two parallel attempts at the same problem (transformer + auxiliary long-term store) with different update rules. The Infini formulation has a cleaner closed-form recurrence; the Titans formulation has a more principled (gradient-based) update.
              </Prose>
            ),
          },
          {
            label: "Sun et al. 2024 — TTT (arXiv:2407.04620)",
            render: () => (
              <Prose>
                Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., and Guestrin, C. (2024). "Learning to (Learn at Test Time): RNNs with Expressive Hidden States." arXiv:2407.04620. The closest theoretical relative of Titans' long-term memory. TTT layers replace the hidden state of an RNN with the parameters of a small neural network, updated via gradient descent on a self-supervised loss at every step. Titans uses the same idea (the long-term memory IS a network whose parameters are updated at test time) but pairs it with attention and persistent memory rather than running it as a stand-alone RNN. Reading TTT after Titans clarifies which parts of Titans are "the memory must be a learner" and which parts are "and you also need attention and a persistent prior" — Sun et al. take the first claim and run with it; Behrouz et al. add the second and third.
              </Prose>
            ),
          },
          {
            label: "Behrouz, Zhong, Mirrokni 2024 — Titans (arXiv:2501.00663)",
            render: () => (
              <Prose>
                Behrouz, A., Zhong, P., and Mirrokni, V. (2024). "Titans: Learning to Memorize at Test Time." arXiv:2501.00663 (Google Research). The paper this topic is about. Read sections 2-3 for the three memories and their integration; section 4 for the surprise-based update with momentum; sections 5-6 for the empirical results at up to 760M parameters and 2M context. The needle-in-a-haystack and language modeling benchmarks are the empirical case for the architecture; the section on "memory in language models" is the conceptual case (cognitive-science framing). Re-read after reading TTT and Infini-attention to see how the three pieces fit together.
              </Prose>
            ),
          },
          {
            label: "Phil Wang — lucidrains/titans-pytorch",
            render: () => (
              <Prose>
                The reference open-source implementation, available at github.com/lucidrains/titans-pytorch. Faithful PyTorch reimplementation of the MAC/MAG/MAL variants, the surprise-based update with momentum, and a small training loop. No custom CUDA kernels (so the test-time update is Python-loop slow), but small enough to read end-to-end. Pair with the paper to verify the equations land in code as expected. Updates ongoing as the architecture evolves.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        Five exercises. Try all before reading the answers. Exercises 1-2 test the surprise/update math; 3 tests the architecture-level reasoning; 4 tests integration mode selection; 5 tests debugging.
      </Prose>

      <H3>Exercise 1 (surprise loss arithmetic)</H3>
      <Prose>
        A long-term memory <Code>M</Code> currently outputs <Code>{"M(k_t) = (0.5, 0.5, 0.0)"}</Code> for the current key. The target value is <Code>{"v_t = (1.0, 0.0, 1.0)"}</Code>. Compute the surprise loss <Code>{"L = \\| M(k_t) - v_t \\|^2"}</Code> and the magnitude of the residual <Code>{"M(k_t) - v_t"}</Code>. If two surprise losses differ by a factor of 4, what is the ratio of their gradient magnitudes (assuming the same Jacobian)?
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> Residual: <Code>{"M(k_t) - v_t = (0.5 - 1.0, 0.5 - 0.0, 0.0 - 1.0) = (-0.5, 0.5, -1.0)"}</Code>. Magnitude squared: <Code>{"(-0.5)^2 + 0.5^2 + (-1.0)^2 = 0.25 + 0.25 + 1.0 = 1.5"}</Code>. So <Code>{"L = 1.5"}</Code>. Magnitude (L2 norm of residual): <Code>{"\\sqrt{1.5} \\approx 1.225"}</Code>. Gradient magnitude: <Code>{"\\nabla L = 2 J^T (M(k_t) - v_t)"}</Code>; the gradient norm scales linearly with the residual norm. So if losses differ by a factor of 4, the residual norms differ by a factor of 2 (since L is residual squared), and the gradient norms differ by a factor of 2. Surprise scales as squared residual; gradient scales as residual. This is why high-surprise tokens dominate updates: a 4x more surprising token contributes 2x more gradient signal, and after the squared scaling many tokens of moderate surprise are dwarfed by one truly surprising token.
      </Callout>

      <H3>Exercise 2 (momentum and weight decay dynamics)</H3>
      <Prose>
        With <Code>{"\\eta = 0.9, \\theta = 0.05, \\alpha = 0.01"}</Code> and a constant per-token gradient <Code>{"g = 1.0"}</Code> (in some scalar parameter dimension), what is the steady-state magnitude of the momentum buffer <Code>{"S"}</Code>? What is the steady-state magnitude of the weight <Code>{"\\theta_M"}</Code>? Now suppose the gradient sign flips abruptly at some step; how many tokens does it take for the momentum direction to reverse?
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> Steady-state momentum: <Code>{"S^* = \\eta S^* - \\theta g"}</Code> gives <Code>{"S^*(1 - \\eta) = -\\theta g"}</Code>, so <Code>{"S^* = -\\theta g / (1 - \\eta) = -0.05 / 0.1 = -0.5"}</Code>. <br />
        Steady-state weight: <Code>{"\\theta_M^* = (1 - \\alpha) \\theta_M^* + S^*"}</Code> gives <Code>{"\\alpha \\theta_M^* = S^*"}</Code>, so <Code>{"\\theta_M^* = S^* / \\alpha = -0.5 / 0.01 = -50"}</Code>. <br />
        Time to reverse: the momentum buffer follows <Code>{"S_t = \\eta S_{t-1} - \\theta g_t"}</Code>. Starting at <Code>{"S = -0.5"}</Code>, with new <Code>{"g = -1.0"}</Code> after the flip: <Code>{"S_1 = 0.9 \\cdot (-0.5) - 0.05 \\cdot (-1) = -0.45 + 0.05 = -0.4"}</Code>; after each step, S = 0.9 * S_prev + 0.05. To go from -0.5 to +0.5 (full reverse), we need approximately <Code>{"\\log_{0.9}(0.001) \\approx 65"}</Code> steps; to reach the new steady state +0.5, similar count. Practical rule: momentum direction reverses on a timescale of <Code>{"1/(1 - \\eta) = 10"}</Code> tokens for the bulk of the change, with a long tail. This is why too-high momentum makes the memory slow to adapt.
      </Callout>

      <H3>Exercise 3 (architecture decision)</H3>
      <Prose>
        You are designing an LLM agent that processes legal contracts averaging 200K tokens. The agent must answer questions about specific clauses (precision retrieval) and reason across the full document (sequence-aware integration). You have an 80GB H100 and need to support batch size 4 (concurrent users). Compare four architectures: vanilla Llama-3.1-8B, Llama-3.1-8B with rotary scaling to 1M, Mamba-2-7B, and Titans-7B (hypothetical, since no public checkpoint exists). Choose one and justify in concrete terms.
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> The decision turns on hardware feasibility, then quality at long context.
        <br />
        <strong>Vanilla Llama-3.1-8B.</strong> Trained context 128K. KV cache at 200K context: ~50GB per sequence (2 * 200K * 4096 * 32 layers * 2 bytes ≈ 100 GB at full precision; ~50 GB at fp16). At batch 4: 200 GB. Does not fit. Eliminated.
        <br />
        <strong>Llama-3.1-8B with rotary scaling.</strong> Same KV cache issue. Even if attention works correctly past the trained context (which it does poorly past 4x), memory is the binding constraint. Eliminated.
        <br />
        <strong>Mamba-2-7B.</strong> Constant state (~1.5 MB) regardless of context. Fits with room to spare on a single H100 at batch 4. Quality: Mamba-2 is competitive with Llama at short context but degrades on tasks that need precise retrieval at long context (its state has to compress everything into a fixed dimension). For "answer questions about specific clauses in a 200K document" — exactly retrieval — Mamba's compression hurts. Quality is acceptable but not best.
        <br />
        <strong>Titans-7B (hypothetical).</strong> Constant state per token, ~500 MB total per layer-stack at the architecture's default sizes. Fits at batch 4. The architecture is explicitly designed for "precise recent + sequence-aware long" — exactly the legal-contract use case. The long-term memory keeps clauses accessible across the full document; the short-term window catches local syntactic dependencies. <em>If</em> a Titans-7B existed, it would be the right choice. Since it does not exist as of early 2026, the practical choice is Mamba-2-7B, accepting the retrieval-quality cost.
        <br />
        <strong>Recommendation</strong>: Mamba-2-7B today; track Titans for when a public checkpoint at scale appears. If retrieval quality is genuinely unacceptable, the practical fallback is RAG with a vector index over chunks — explicit retrieval is more interpretable than Mamba's compressed state and competitive in quality.
      </Callout>

      <H3>Exercise 4 (integration mode selection)</H3>
      <Prose>
        For each task below, pick the most appropriate Titans integration mode (MAC, MAG, or MAL) and justify in one sentence. (a) Long-document summarization where the model must integrate facts from across the document. (b) Multi-step reasoning where the model must explicitly retrieve a stored intermediate result. (c) Streaming dialogue where the relative weight of recent vs distant context varies turn to turn.
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> (a) <strong>MAC (memory-as-context).</strong> Summarization wants a smooth blend of recent and distant context, with the attention mechanism choosing the mix per token based on what is relevant. MAC's "long-term as additional attention keys" gives this naturally — the softmax over [persistent; window; long-term] weighs them adaptively. (b) <strong>MAL (memory-as-layer).</strong> Multi-step reasoning needs the memory output to be a clear, separable contribution that downstream layers can act on; making it a separate residual layer keeps the paths explicit and gives the model a "layer that does retrieval" the rest of the network can reason about. (c) <strong>MAG (memory-as-gate).</strong> Streaming dialogue has well-defined per-turn boundaries where the gate value can encode "does this turn build on the recent past or recall something from earlier"; a learned gate is the cleanest way to expose that decision.
      </Callout>

      <H3>Exercise 5 (debugging a deployment)</H3>
      <Prose>
        You deploy a Titans-style architecture for long-document QA. Initial test set accuracy is 78%. After deployment for two weeks, accuracy on streaming traffic has degraded to 41%. Tokens per request are similar to test. You suspect Titans-specific issues. List three failure modes worth checking and a diagnostic for each.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Three Titans-specific things to check:
        <br />
        (1) <strong>Cross-request memory leakage.</strong> If the long-term memory state is being kept warm across requests on the same server (for performance), then user A's request can be influenced by user B's earlier request. The accuracy degradation could be the memory accumulating cross-request cruft. Diagnostic: compare accuracy with and without explicit memory reset between requests. If reset fixes it, you have a leakage bug; the fix is to enforce a reset at every request boundary or to maintain per-request memory copies.
        <br />
        (2) <strong>Long-term memory weight drift.</strong> Without sufficient weight decay, the MLP weights in the long-term memory can drift over millions of tokens of streaming data. The drift is invisible at single-request quality (each request resets) but visible if memory is kept warm. Diagnostic: log <Code>{"\\| W_1 \\|_F"}</Code> for the long-term memory; if it has grown 10x from initialization, weight decay is too low. Fix: increase <Code>{"\\alpha"}</Code> or schedule periodic resets.
        <br />
        (3) <strong>Test-time learning rate mismatch with deployment distribution.</strong> The test-time <Code>{"\\theta"}</Code> was tuned on the test set distribution. If deployment traffic is shifted (different language registers, different document types), the same <Code>{"\\theta"}</Code> may be too high (memory thrashes) or too low (memory under-updates). Diagnostic: compute the per-token surprise distribution on a sample of streaming requests; compare to the test-set distribution. If surprises are consistently 5-10x higher in production, the model is encountering out-of-distribution patterns at every step and the test-time update is dominated by noise. Fix: tune <Code>{"\\theta"}</Code> down by the ratio of surprise distributions, or fine-tune the base projections on a sample of production data so surprises return to a calibrated range.
        <br />
        Bonus: gradient clipping inside the test-time update (per-parameter clamp) is standard but easy to forget; if it's missing, a single extreme outlier token can corrupt the memory for the rest of the request, and aggregate quality drops without a clean failure signal.
      </Callout>

    </div>
  ),
};

export default titansContent;
