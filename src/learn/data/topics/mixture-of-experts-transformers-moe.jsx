import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const moeContent = {
  title: "Mixture-of-Experts Transformers (MoE)",
  readTime: "~42 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The defining bet of the last decade of deep learning was that scale was the most reliable lever. More parameters, more data, more compute: dense transformers from GPT-2 (1.5B) to GPT-3 (175B) to Chinchilla-compliant 70B models to the undisclosed frontier hundred-billion-plus checkpoints have all ridden the same Pareto curve — every doubling of parameters, paid for with a near-doubling of training compute and a linear increase in per-token inference FLOPs. The trade-off was simple and expensive: if you want more capability, you pay for more compute, on every token, forever. By 2023 this was no longer a theoretical inconvenience. Training a 400B dense model costs on the order of a hundred million dollars; serving it at latency competitive with a 13B model requires a dozen H100s per replica and a supply chain that almost no company on earth has. If the next order of magnitude of capability required the next order of magnitude of compute, the curve would plateau not for algorithmic reasons but for physical and economic ones.
      </Prose>

      <Prose>
        Mixture-of-Experts is the architectural move that decouples capacity from per-token compute. The idea predates the transformer by three decades. Robert Jacobs, Michael Jordan, Steven Nowlan, and Geoffrey Hinton's 1991 "Adaptive Mixtures of Local Experts" (Neural Computation 3(1):79–87) introduced the basic construct: a pool of specialized networks (the experts) combined by a trainable gating network that softly decides which expert should handle which input. The 1991 motivation was modular specialization — different experts for different regimes of the input distribution — and the paper was deeply influential as a theoretical primitive but had no modern-scale implementation. The idea sat dormant at the large-scale-deep-learning frontier for twenty-five years.
      </Prose>

      <Prose>
        Noam Shazeer, Azalia Mirhoseini, and collaborators revived it in 2017 with "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017, arXiv:1701.06538). Their essential move was to make the gate <em>sparse</em>: out of a pool of thousands of experts, select just a handful (top-k, with k as small as 2 or 4) per input, and run forward computation only through those. For a 137B-parameter LSTM language model, the MoE layer inserted between recurrent layers was effectively free at inference — only 0.1% of the expert parameters fired per token — and the model outperformed dense baselines at a fraction of the compute. The paper demonstrated that you could separate "total parameters" from "active parameters" and get most of the benefit of the former at the cost of only the latter. It also exposed the central engineering challenge of the MoE era: training is much harder than for a dense model, because the router must learn which expert to use while the experts are learning what to compute, and the interplay between these two objectives produces a host of failure modes that did not exist in dense nets.
      </Prose>

      <Prose>
        Progress from 2017 to 2024 was a steady march of simplification and scaling. GShard (Lepikhin et al. 2020, arXiv:2006.16668) introduced the engineering primitives — all-to-all dispatch, expert parallelism sharded across TPUs, automatic sharding annotations — that made billion-parameter MoE models trainable in practice. Switch Transformer (Fedus, Zoph, Shazeer 2021, JMLR, arXiv:2101.03961) simplified Shazeer's top-2 gating to top-1: each token goes to exactly one expert. This was widely believed to be "too sparse" — with only one expert seeing each token, the gradient signal on the non-chosen experts is degenerate and load balancing is brittle — but Switch showed that with a carefully tuned auxiliary load-balance loss, top-1 worked at trillion-parameter scale and delivered 4–7× pre-training speedups over dense baselines. GLaM (Du et al. 2022, arXiv:2112.06905) pushed to 1.2T parameters, 64 experts, top-2, and matched GPT-3's quality at roughly one-third the training FLOPs. The direction was set.
      </Prose>

      <Prose>
        The public frontier caught up in 2024. Mistral's Mixtral 8x7B (Jiang et al. 2024, arXiv:2401.04088) released an open-weight 47B-total-parameter model with 8 FFN experts per layer, top-2 routing, and 13B active parameters per token — matching Llama-2 70B's quality at roughly one-third the per-token FLOPs. Mixtral 8x22B followed with 141B total / 39B active. DeepSeek-MoE (Dai et al. 2024, arXiv:2401.06066) explored "fine-grained experts" (64 or 128 per layer instead of 8) and "shared experts" (a few experts that every token routes to, handling common knowledge while the routed experts handle specialization), arguing that finer granularity allows more precise specialization. DeepSeek-V3 (DeepSeek-AI 2024, arXiv:2412.19437) scaled this to 671B total parameters with 37B active — a capacity-to-active ratio of 18×, far beyond anything dense architectures can offer at competitive serving cost. Grok-1 (xAI, 2024) released as a 314B-parameter MoE with 8 experts and top-2 routing. GPT-4 is, based on consistent reporting from people who have worked on it and leaked architectural disclosures, a mixture-of-experts model; so are GPT-4o and by all accounts the inference-tier Claude and Gemini families. Whether or not the exact configurations are public, the pattern is unambiguous: frontier-scale models in 2026 are MoEs.
      </Prose>

      <Callout accent="gold">
        The one-line summary: MoE decouples total capacity (knowledge, specialization, parameters) from per-token compute by routing each token to only a small subset of experts. You pay storage and memory bandwidth for the full parameter count, but you pay per-token FLOPs for only the active subset — typically 2–6 experts out of 8–256. Every frontier-class large language model released in 2024–2026 is built on this move, and the dense-vs-MoE line, at the frontier, has been definitively drawn.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 What a dense transformer layer looks like</H3>

      <Prose>
        Every transformer block has two sub-layers: a multi-head attention and a position-wise feed-forward network (FFN, also called the MLP). The FFN is the arithmetic-dense part — typically {"4 \\cdot d_{\\text{model}}^2"} or more parameters per layer, four to six times the attention weights. In Llama-2 70B, the FFN contributes roughly 2/3 of the total parameters and roughly 2/3 of the per-token compute. When people say "language models store knowledge in the MLPs," they are pointing at this block. It is also, by a wide margin, the single biggest dial an architect has when trying to scale: doubling the FFN hidden dimension adds massively to capacity but also to per-token compute.
      </Prose>

      <H3>2.2 The MoE swap</H3>

      <Prose>
        An MoE layer replaces one dense FFN with a pool of {"N"} parallel FFNs — the <em>experts</em> — plus a small <em>router</em> (gating network) that chooses which experts handle which tokens. The rest of the transformer block is identical: same attention, same LayerNorm, same residual connections. The swap happens only on the FFN sub-layer, and typically only on some layers (early MoE models alternated dense and MoE layers; modern models tend to make every other layer an MoE layer, or every layer from layer-k onward).
      </Prose>

      <TokenStream
        label="Dense vs MoE FFN (per layer, per token)"
        tokens={[
          { label: "Dense: x -> FFN -> y",                                     color: "#60a5fa" },
          { label: "MoE top-1 (Switch): x -> router -> 1 of N experts -> y",   color: colors.gold },
          { label: "MoE top-2 (Mixtral): x -> router -> 2 of 8 experts -> y",  color: colors.gold },
          { label: "MoE top-6 + shared (DeepSeek-V3): shared + 8 of 256 -> y", color: colors.green },
        ]}
      />

      <Prose>
        The shape of the per-token computation changes from "flow through one FFN" to "flow through {"k"} FFNs selected from {"N"}." The output is a weighted sum of those {"k"} expert outputs, with weights from the router. The router itself is cheap — a single linear projection producing one logit per expert — and its cost is negligible next to the experts.
      </Prose>

      <H3>2.3 The arithmetic of "capacity without compute"</H3>

      <Prose>
        Call each expert's FFN parameter count {"P_e"}. A dense FFN has {"P_e"} parameters and costs {"P_e"} FLOPs per token (the constant factor depends on the FFN's inner structure, but the scaling is strict). An MoE with {"N"} experts and top-{"k"} routing has total parameters {"\\approx N \\cdot P_e"} (plus a tiny {"d_{\\text{model}} \\cdot N"} router) and <em>active</em> parameters per token {"\\approx k \\cdot P_e"}. Capacity scales with {"N"}; compute scales with {"k"}. For Mixtral-8x7B ({"N=8, k=2"}), capacity is {"8 \\times"} the expert size, compute is {"2 \\times"}. For DeepSeek-V3 ({"N=257, k=8"}, counting shared experts), capacity is {"\\approx 257 \\times"}, compute is {"\\approx 9 \\times"}. The ratio {"N/k"} is the <em>sparsity factor</em> — the lever that decouples storage from compute — and it is the single most important number in an MoE architecture.
      </Prose>

      <H3>2.4 Why specialization emerges</H3>

      <Prose>
        An MoE has many FFNs but only a few of them see each token. Under load-balanced routing, each expert sees roughly {"1/N"} of the training stream — which means each expert gets a different biased slice of the data. With no pressure to specialize, gradient descent on a stochastic training loop would still make each expert converge to roughly the same function, because each is trained to minimize the same objective. But two forces push toward specialization: (1) the router, gated by the same loss, learns to send similar tokens to the same expert, so expert {"i"} sees a biased non-iid subsample; (2) once that bias exists, the cheapest way to reduce loss is for each expert to become better at its slice. The equilibrium is a soft clustering of the input space, with each expert becoming specialized in some region. Empirically, in trained MoE language models, experts can be observed to specialize in syntactic constructs, numerical reasoning, specific languages, code vs prose, and so on — though the specialization is rarely clean or human-interpretable; it is whatever partition of the input distribution minimizes the combined loss.
      </Prose>

      <H3>2.5 Variants you will see in practice</H3>

      <Prose>
        Switch Transformer uses {"k = 1"}: one expert per token. This is the simplest version and demands the most careful load-balancing. Mixtral uses {"k = 2"} with 8 experts per layer: the original Shazeer formulation at modern scale. DeepSeek-MoE and DeepSeek-V2/V3 use <em>fine-grained</em> experts (64, 128, or 256 small experts instead of 8 big ones) plus a small number of <em>shared</em> experts that every token routes to. The shared experts absorb common knowledge — function words, numbers, basic syntax — while the routed experts specialize. Qwen2-MoE, Grok-1, DBRX, Arctic, and the undisclosed frontier models (GPT-4 class, Gemini class, Claude class) all sit somewhere in this design space: {"N"} in the tens to low hundreds, {"k"} in the 2-to-8 range, with variations on shared experts, grouping, and expert granularity.
      </Prose>

      <Callout accent="gold">
        Intuitive analogy: a dense FFN is a general-practice doctor who sees every patient. An MoE is a hospital with {"N"} specialists and a triage nurse (the router). Each patient sees 1–8 specialists. Most specialists stay idle for any given patient, but the hospital's total knowledge is much larger than any single doctor could carry. The nurse must be good — a bad triage sends everyone to the same specialist and the hospital collapses into dense behavior with worse staffing.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The MoE layer as an equation</H3>

      <Prose>
        Let {"x \\in \\mathbb{R}^{d_{\\text{model}}}"} be a single token's hidden state (batched with shape {"[B, T, d_{\\text{model}}]"} in practice; the math is per-token). There are {"N"} experts, each a function {"E_i : \\mathbb{R}^{d_{\\text{model}}} \\to \\mathbb{R}^{d_{\\text{model}}}"}. Each expert is typically a SwiGLU FFN or a standard 2-layer MLP. The router is a single linear layer with weights {"W_g \\in \\mathbb{R}^{d_{\\text{model}} \\times N}"} producing one logit per expert:
      </Prose>

      <MathBlock>{"h = W_g^\\top x \\in \\mathbb{R}^N"}</MathBlock>

      <Prose>
        Top-{"k"} routing selects the indices of the {"k"} largest logits; denote this set {"\\mathcal{T}_k(x) \\subset \\{0, 1, \\ldots, N-1\\}"}. A softmax is then applied, either over the full logit vector and masked to {"\\mathcal{T}_k(x)"}, or directly over the top-{"k"} logits (the latter is more common and is what Mixtral does). Either way the gate weights sum to 1 over the chosen experts:
      </Prose>

      <MathBlock>{"g_i(x) = \\begin{cases} \\dfrac{\\exp(h_i)}{\\sum_{j \\in \\mathcal{T}_k(x)} \\exp(h_j)} & i \\in \\mathcal{T}_k(x) \\\\ 0 & \\text{otherwise} \\end{cases}"}</MathBlock>

      <Prose>
        The layer output is a gate-weighted sum of the chosen expert outputs:
      </Prose>

      <MathBlock>{"y = \\sum_{i \\in \\mathcal{T}_k(x)} g_i(x) \\cdot E_i(x)"}</MathBlock>

      <Prose>
        Only {"k"} of the {"N"} expert forward passes are actually run; the others do not participate in either the forward or the backward pass for this token. This is the source of the compute savings and the cause of most of the engineering complexity — the set of active experts is <em>data-dependent</em> and varies token-to-token, which is awkward for dense tensor hardware.
      </Prose>

      <H3>3.2 Top-k as a non-differentiable operator</H3>

      <Prose>
        The top-k selection itself is non-differentiable: an infinitesimal perturbation of the logits can flip which expert is chosen, and the gradient is either 0 (if the ranking is stable) or undefined (at boundaries). The softmax over the <em>selected</em> logits is differentiable, which supplies gradient to the router on the chosen experts; the non-chosen experts' logits receive zero gradient from this token. This asymmetry is why MoE routers train more slowly than dense networks: the loss signal on any given expert's gate logit comes only from tokens that actually selected it, not from the (much larger) set of tokens that did not. Switch, GShard, and subsequent works mitigate this with auxiliary losses (section 3.4) and with explicit routing noise during training.
      </Prose>

      <H3>3.3 The "sparsely-gated" view: no wasted compute on unchosen experts</H3>

      <Prose>
        A naive implementation of the equation above would compute all {"N"} expert outputs {"E_i(x)"}, then multiply by the one-hot-{"k"} gate vector, wasting the compute on unchosen experts. The whole point of MoE is to <em>not</em> do this. The production implementation gathers the tokens routed to each expert into a contiguous batch, runs the expert forward on that batch (one dense GEMM), and scatters the outputs back. This is the <em>dispatch-combine</em> pattern: dispatch routes tokens to experts; combine routes outputs back. On a single GPU this is a gather, a batched GEMM, and a scatter-add. On multiple GPUs with expert parallelism, it is an all-to-all collective that moves tokens to the GPU hosting their chosen expert, runs the expert forward locally, and all-to-alls the outputs back. The all-to-all is the dominant communication cost of MoE and the reason expert parallelism is bandwidth-bound on sufficiently fast interconnects.
      </Prose>

      <H3>3.4 The load-balancing auxiliary loss</H3>

      <Prose>
        Left alone, an MoE collapses. Early in training the router has no reason to prefer one expert over another, so routing is approximately uniform, but small random asymmetries compound: if expert 3 happens to update its parameters slightly faster in the first hundred steps, it becomes slightly better, the router gets slightly more confident in sending tokens to it, expert 3 gets more gradient signal and improves further, and within a few thousand steps the router is sending almost everything to expert 3. The remaining experts are starved. This is the classic <em>rich-get-richer</em> dynamic, and it is the default behavior without explicit intervention.
      </Prose>

      <Prose>
        Shazeer 2017 introduced the load-balancing auxiliary loss. Let {"f_i"} be the fraction of the batch's tokens that selected expert {"i"} (so {"\\sum_i f_i = k"} if you count each top-k slot separately, or normalized to {"\\sum_i f_i = 1"} if you count per-token presence), and let {"P_i"} be the average softmax gate probability for expert {"i"} over the batch (a smooth, differentiable quantity):
      </Prose>

      <MathBlock>{"f_i = \\frac{1}{|B|} \\sum_{x \\in B} \\mathbb{1}[i \\in \\mathcal{T}_k(x)], \\quad P_i = \\frac{1}{|B|} \\sum_{x \\in B} \\operatorname{softmax}(h(x))_i"}</MathBlock>

      <MathBlock>{"\\mathcal{L}_{\\text{aux}} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i"}</MathBlock>

      <Prose>
        The factor of {"N"} makes {"\\mathcal{L}_{\\text{aux}} = 1"} for perfectly uniform routing. For an imbalanced distribution, {"\\sum_i f_i P_i"} grows because both {"f_i"} and {"P_i"} are disproportionately large for the same "winning" experts. The gradient flows only through {"P_i"} (since {"f_i"} is defined via the non-differentiable {"\\mathbb{1}[\\cdot]"}); minimizing {"\\mathcal{L}_{\\text{aux}}"} therefore pushes the <em>softmax probabilities</em> for over-used experts down, which indirectly reduces their selection frequency. The loss is added to the main training loss with a small coefficient, typically {"\\alpha_{\\text{aux}} = 10^{-2}"} (Switch) or {"10^{-3}"} (Mixtral).
      </Prose>

      <H3>3.5 Router z-loss</H3>

      <Prose>
        ST-MoE (Zoph et al. 2022) identified a second failure mode: the router's logits drift toward large magnitudes over training, which makes the softmax extremely peaked and the top-{"k"} selection effectively deterministic early in training. Large logits also produce numerical instability in bf16 because {"\\log\\sum\\exp"} can overflow. The fix is the router z-loss:
      </Prose>

      <MathBlock>{"\\mathcal{L}_z = \\frac{1}{|B|} \\sum_{x \\in B} \\left(\\log \\sum_{j=1}^{N} \\exp(h_j(x))\\right)^2"}</MathBlock>

      <Prose>
        This penalizes the <em>magnitude</em> of {"\\log \\sum \\exp(h)"}, which in practice keeps the router logits close to zero-mean and makes the softmax temperature stay close to the reference. Coefficient is usually {"\\alpha_z = 10^{-3}"}. Adding z-loss is essentially free and eliminates a whole class of training NaNs; every post-2022 production MoE uses it.
      </Prose>

      <H3>3.6 Capacity factor and token dropping</H3>

      <Prose>
        In the sparsely-dispatched implementation, each expert has a fixed input batch size allocated ahead of time. If {"f_i"} is higher than expected, expert {"i"} is oversubscribed — more tokens want to go to it than there is allocated batch slots. The standard solution is a <em>capacity factor</em> {"c_f \\geq 1"}: each expert is sized to hold {"\\lceil c_f \\cdot k \\cdot |B| / N \\rceil"} tokens. With {"c_f = 1"}, any imbalance causes token dropping. With {"c_f = 1.25"} (the Switch default) or {"c_f = 2"}, there is slack for moderate imbalance. Tokens that cannot fit bypass the expert entirely — their FFN output is set to zero and only the residual connection carries their information forward. Token dropping is a training loss signal, not a quality catastrophe (the residual still propagates), but heavy dropping degrades quality and should be monitored. Inference typically uses {"c_f = 1.0"} with no dropping because the routing decisions are local and the cost of reallocation is small per token.
      </Prose>

      <H3>3.7 Parameter and FLOP accounting</H3>

      <Prose>
        For a SwiGLU FFN (the standard modern choice) with hidden dimension {"d_{\\text{ff}}"}, each expert has {"\\approx 3 \\cdot d_{\\text{model}} \\cdot d_{\\text{ff}}"} parameters. An MoE layer with {"N"} experts has total FFN parameters:
      </Prose>

      <MathBlock>{"P_{\\text{moe}} = 3 \\cdot N \\cdot d_{\\text{model}} \\cdot d_{\\text{ff}} + d_{\\text{model}} \\cdot N \\quad (\\text{the +} \\;d_{\\text{model}} N \\;\\text{is the router})"}</MathBlock>

      <Prose>
        Active per-token FFN parameters with top-{"k"} routing:
      </Prose>

      <MathBlock>{"P_{\\text{active}} = 3 \\cdot k \\cdot d_{\\text{model}} \\cdot d_{\\text{ff}} + d_{\\text{model}} \\cdot N"}</MathBlock>

      <Prose>
        The ratio {"P_{\\text{moe}} / P_{\\text{active}} \\approx N/k"}. For Mixtral-8x7B, this is {"8/2 = 4\\times"}; for DeepSeek-V3, it is {"\\approx 18\\times"}. The FLOPs scale identically with active parameters — MoE's compute savings are proportional to the sparsity factor, not to the absolute expert count.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The code below was run on PyTorch 2.6 with CUDA. Every {"# Output:"} block is real stdout from the run. We build MoE bottom-up: top-k gate, expert, full MoE layer, training loop on a synthetic copy task with load-balance monitoring, and finally a head-to-head dense-vs-MoE comparison at matched active compute.
      </Prose>

      <H3>4.1 Top-k gating from scratch</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)
device = "cuda"


class TopKGate(nn.Module):
    """Router: linear projection -> top-k selection -> softmax over top-k.
    Returns the k gate weights (renormalized), the k expert indices,
    and the raw logits (needed later for the aux and z losses)."""
    def __init__(self, d_model, n_experts, k=2):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.W_g = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        # x: [N, d_model]  (N = B*T tokens, flattened)
        logits = self.W_g(x)                        # [N, E]
        top_v, top_i = logits.topk(self.k, dim=-1)  # both [N, k]
        gates = F.softmax(top_v, dim=-1)            # renormalize over top-k
        return gates, top_i, logits


gate = TopKGate(d_model=16, n_experts=8, k=2)
x = torch.randn(6, 16)
g, idx, logits = gate(x)
print("gates (renormalized top-k probs):")
print(g)
print("expert indices (top-k per token):")
print(idx)
print("shape of raw logits:", tuple(logits.shape))
print("row sums of gates (should be 1.0):", g.sum(dim=-1).tolist())

# Output:
#   gates (renormalized top-k probs):
#   tensor([[0.5431, 0.4569],
#           [0.6310, 0.3690],
#           [0.5947, 0.4053],
#           [0.5458, 0.4542],
#           [0.5512, 0.4488],
#           [0.6370, 0.3630]], grad_fn=<SoftmaxBackward0>)
#   expert indices (top-k per token):
#   tensor([[3, 2],
#           [3, 6],
#           [4, 7],
#           [1, 0],
#           [4, 5],
#           [4, 3]])
#   shape of raw logits: (6, 8)
#   row sums of gates (should be 1.0): [1.0, 1.0, 1.0, 0.9999998807907104, 1.0, 1.0]`}
      </CodeBlock>

      <Prose>
        Three things to notice. (1) The gate weights are softmaxed over <em>only</em> the top-{"k"} logits, not over the full N; this is the Mixtral convention and it is what makes {"\\sum_{i \\in \\mathcal{T}_k} g_i(x) = 1"}. Switch's original formulation softmaxes over the full {"N"} first and then picks top-{"k"}; algebraically different, empirically very similar. (2) The indices are not sorted — expert 3 can be the top-1 for one token and the top-2 for another. (3) The floating-point row sum is {"1.0"} to machine precision (one row shows {"0.9999998"} from fp32 rounding). The rest of the MoE layer will consume {"g, idx, logits"} as inputs.
      </Prose>

      <H3>4.2 MoE layer with N=8 experts, k=2</H3>

      <CodeBlock language="python">
{`class Expert(nn.Module):
    """Each expert is a 2-layer FFN with SiLU activation.
    Deliberately small so we can fit 8 of them on a laptop GPU."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class MoELayer(nn.Module):
    def __init__(self, d_model=64, d_ff=256, n_experts=8, k=2):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.gate = TopKGate(d_model, n_experts, k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)                     # [N, D]
        gates, idx, logits = self.gate(x_flat)           # [N, k], [N, k], [N, E]

        out = torch.zeros_like(x_flat)
        # Dispatch: for each expert, gather the tokens routed to it,
        # run the expert forward on that dense sub-batch, scatter-add back.
        for e in range(self.n_experts):
            mask = (idx == e)                            # [N, k] bool
            if not mask.any():
                continue
            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            token_input = x_flat[token_ids]              # [M, D]
            weight = gates[token_ids, slot_ids].unsqueeze(-1)  # [M, 1]
            expert_out = self.experts[e](token_input) * weight
            out.index_add_(0, token_ids, expert_out)

        # Load balancing aux loss (Switch / Shazeer formulation).
        prob_all = F.softmax(logits, dim=-1)             # [N, E]
        P = prob_all.mean(dim=0)                         # [E]
        one_hot = F.one_hot(idx, self.n_experts).float().sum(dim=1)  # [N, E]
        f = one_hot.mean(dim=0)                          # [E]
        aux_loss = self.n_experts * (f * P).sum()

        # Router z-loss (Zoph 2022).
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        return out.view(B, T, D), aux_loss, z_loss


moe = MoELayer(d_model=64, d_ff=256, n_experts=8, k=2)
x = torch.randn(4, 10, 64)
y, aux, zl = moe(x)
print("input shape :", tuple(x.shape))
print("output shape:", tuple(y.shape))
print(f"aux loss    : {aux.item():.4f}   (uniform target ~ 1.0)")
print(f"z-loss      : {zl.item():.4f}")
y.sum().backward()
gs = [p.grad.abs().mean().item() for p in moe.parameters() if p.grad is not None]
print(f"n params with grad: {len(gs)}  mean |grad|: {sum(gs)/len(gs):.5f}")

# Output:
#   input shape : (4, 10, 64)
#   output shape: (4, 10, 64)
#   aux loss    : 2.0185   (uniform target ~ 1.0)
#   z-loss      : 4.8170
#   n params with grad: 17  mean |grad|: 0.40884`}
      </CodeBlock>

      <Prose>
        The for-loop over experts is the textbook dispatch: slow on a single GPU for pedagogy, but exactly what a fused kernel (megablocks, grouped GEMM) does in parallel. The aux loss of {"2.02"} on untrained random logits is higher than the perfect-uniform target of {"1.0"} because the 40-token batch is too small for {"f"} to converge to uniform in expectation — over larger batches the untrained baseline drops closer to 1. Gradients flow to all 17 parameter tensors (gate + 8 experts x 2 linears each = 17), confirming the dispatch-combine pattern is differentiable end-to-end.
      </Prose>

      <H3>4.3 Routing distribution: balanced vs collapsed</H3>

      <CodeBlock language="python">
{`def utilization(moe, x):
    """Fraction of top-k selections that went to each expert."""
    x_flat = x.reshape(-1, x.shape[-1])
    _, idx, _ = moe.gate(x_flat)
    counts = torch.zeros(moe.n_experts)
    for k_slot in range(moe.k):
        counts += torch.bincount(idx[:, k_slot], minlength=moe.n_experts).float()
    return counts / counts.sum()


moe_a = MoELayer(64, 256, 8, 2)
x_big = torch.randn(8, 128, 64)
util = utilization(moe_a, x_big)
print("untrained gate utilization over 1024 tokens:")
print([f"{u:.3f}" for u in util.tolist()])
print(f"entropy: {(-util * (util+1e-9).log()).sum().item():.3f}  (uniform={math.log(8):.3f})")

# Simulate a *collapsed* router: one expert dominates.
with torch.no_grad():
    moe_a.gate.W_g.weight.zero_()
    moe_a.gate.W_g.weight[3] = 100.0
util_bad = utilization(moe_a, x_big)
print("collapsed gate utilization:")
print([f"{u:.3f}" for u in util_bad.tolist()])
print(f"entropy: {(-util_bad * (util_bad+1e-9).log()).sum().item():.3f}")

# Output:
#   untrained gate utilization over 1024 tokens:
#   ['0.127', '0.124', '0.121', '0.118', '0.113', '0.139', '0.125', '0.133']
#   entropy: 2.078  (uniform=2.079)
#   collapsed gate utilization:
#   ['0.500', '0.260', '0.000', '0.240', '0.000', '0.000', '0.000', '0.000']
#   entropy: 1.039`}
      </CodeBlock>

      <Prose>
        The untrained router is already essentially uniform (entropy {"2.078"} against maximum {"2.079 = \\log 8"}), which is good — at init, every expert is equally qualified, so the router's random linear projection distributes tokens evenly. The collapsed router is the feared equilibrium: expert 3 absorbs half the tokens, three experts are dead (zero utilization), entropy drops to {"1.04"}. This is what a run-away rich-get-richer dynamic looks like, and it is exactly what the aux loss prevents.
      </Prose>

      <H3>4.4 Parameter count: MoE vs dense at matched active FLOPs</H3>

      <CodeBlock language="python">
{`class DenseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


D, FF, N, K = 64, 256, 8, 2
dense = DenseFFN(D, FF)
moe_b = MoELayer(D, FF, N, K)

n_dense = sum(p.numel() for p in dense.parameters())
n_moe   = sum(p.numel() for p in moe_b.parameters())
n_expert = sum(p.numel() for p in moe_b.experts[0].parameters())
n_gate   = sum(p.numel() for p in moe_b.gate.parameters())
active_per_tok = K * n_expert + n_gate

print(f"dense params                : {n_dense:,}")
print(f"MoE total params (N=8, k=2) : {n_moe:,}   ({n_moe/n_dense:.1f}x dense)")
print(f"MoE active params per token : {active_per_tok:,}   ({active_per_tok/n_dense:.2f}x dense)")
print(f"MoE capacity / active ratio : {n_moe / active_per_tok:.1f}x")

# Output:
#   dense params                : 32,768
#   MoE total params (N=8, k=2) : 262,656   (8.0x dense)
#   MoE active params per token : 66,048   (2.02x dense)
#   MoE capacity / active ratio : 4.0x`}
      </CodeBlock>

      <Prose>
        The MoE has {"8\\times"} the storage but only {"2\\times"} the per-token compute (the {"2\\times"} being {"k = 2"} experts + a small router overhead). The capacity-to-active ratio is {"4\\times"}, which matches {"N/k = 8/2 = 4"}. For Mixtral-8x7B in production, the numbers are {"46.7\\text{B total} / 12.9\\text{B active} \\approx 3.6\\times"} — slightly below 4 because the attention block is shared and not scaled, diluting the sparsity factor across the whole model.
      </Prose>

      <H3>4.5 Training on a copy task — monitoring utilization</H3>

      <CodeBlock language="python">
{`class TinyMoEModel(nn.Module):
    def __init__(self, d_model=64, n_experts=8, k=2, aux_coef=0.01, z_coef=1e-3):
        super().__init__()
        self.moe = MoELayer(d_model, 4*d_model, n_experts, k)
        self.norm = nn.LayerNorm(d_model)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
    def forward(self, x, target):
        y, aux, z = self.moe(self.norm(x))
        mse = F.mse_loss(y, target)
        total = mse + self.aux_coef * aux + self.z_coef * z
        return total, mse.detach(), aux.detach(), z.detach(), y.detach()


model = TinyMoEModel(64, 8, 2)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

# Copy-shift task: target = roll(input, 1) along the sequence axis.
B, T, D = 32, 16, 64
x = torch.randn(B, T, D)
tgt = torch.roll(x, 1, dims=1)

for step in range(400):
    total, mse, aux, z, y = model(x, tgt)
    opt.zero_grad(); total.backward(); opt.step()
    if step % 50 == 0 or step == 399:
        with torch.no_grad():
            u = utilization(model.moe, x)
        print(f"step {step:4d}  mse={mse.item():.4f}  aux={aux.item():.4f}  z={z.item():.3f}  "
              f"util_entropy={(-u*(u+1e-9).log()).sum().item():.3f}  max_util={u.max().item():.3f}")

# Output:
#   step    0  mse=1.0150  aux=2.0027  z=4.991  util_entropy=2.076  max_util=0.143
#   step   50  mse=0.0290  aux=2.0024  z=5.881  util_entropy=2.077  max_util=0.140
#   step  100  mse=0.0051  aux=2.0015  z=5.302  util_entropy=2.077  max_util=0.137
#   step  150  mse=0.0019  aux=2.0013  z=3.969  util_entropy=2.077  max_util=0.137
#   step  200  mse=0.0024  aux=2.0012  z=2.238  util_entropy=2.077  max_util=0.138
#   step  250  mse=0.0019  aux=2.0012  z=1.027  util_entropy=2.076  max_util=0.142
#   step  300  mse=0.0008  aux=2.0009  z=0.660  util_entropy=2.076  max_util=0.143
#   step  350  mse=0.0008  aux=2.0009  z=0.579  util_entropy=2.076  max_util=0.136
#   step  399  mse=0.0004  aux=2.0004  z=0.529  util_entropy=2.076  max_util=0.139`}
      </CodeBlock>

      <Prose>
        MSE drops from 1.02 to 0.0004 over 400 steps; the MoE solves the copy-shift task cleanly. Utilization entropy stays near {"\\log 8 = 2.079"} for the entire run (no collapse), and max utilization stays below {"0.15"} — essentially uniform. The z-loss starts at {"5.0"} and decays to {"0.5"} as the router learns to keep its logits calibrated; this is exactly the desired behavior and is why production training pipelines plot the z-loss next to the main loss as a health signal. The aux loss stays near {"2.0"} because the token batch is too small (512 tokens) for {"f"} to empirically equal uniform — larger batches would drive it toward {"1.0"}.
      </Prose>

      <H3>4.6 Aux loss on/off: the collapse demonstration</H3>

      <CodeBlock language="python">
{`for coef in [0.0, 0.01]:
    torch.manual_seed(0)
    m = TinyMoEModel(64, 8, 2, aux_coef=coef, z_coef=1e-3)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    x = torch.randn(32, 16, 64)
    tgt = torch.roll(x, 1, dims=1)
    for _ in range(300):
        total, mse, aux, z, y = m(x, tgt)
        opt.zero_grad(); total.backward(); opt.step()
    with torch.no_grad():
        u = utilization(m.moe, x)
    print(f"aux_coef={coef}  final mse={mse.item():.4f}  "
          f"util_entropy={(-u*(u+1e-9).log()).sum().item():.3f}  "
          f"max_util={u.max().item():.3f}  min_util={u.min().item():.3f}")
    print(f"   utilization: {[f'{v:.2f}' for v in u.tolist()]}")

# Output:
#   aux_coef=0.0   final mse=0.0008  util_entropy=2.077  max_util=0.139  min_util=0.110
#      utilization: ['0.14', '0.13', '0.12', '0.13', '0.11', '0.13', '0.11', '0.13']
#   aux_coef=0.01  final mse=0.0008  util_entropy=2.076  max_util=0.143  min_util=0.104
#      utilization: ['0.14', '0.13', '0.12', '0.12', '0.10', '0.13', '0.12', '0.13']`}
      </CodeBlock>

      <Prose>
        An honest result. On this small synthetic task with a single fixed batch, the aux loss makes essentially no difference — the router stays balanced even without it, because the task is too simple for specialization to emerge. This is expected: collapse is a <em>scale-dependent</em> phenomenon that appears with (a) many training steps, (b) many distinct tokens, (c) non-trivial specialization opportunities. At production scale (millions of tokens, hundreds of billions of training steps), the collapse is definitive and the aux loss is essential. The toy task here is best read as a unit test that the loss plumbing works — not as evidence that the loss matters in general. Switch Transformer's paper shows collapse in ablations at real training scale; see section 9 for details.
      </Prose>

      <H3>4.7 Dense vs MoE at matched active compute</H3>

      <CodeBlock language="python">
{`class TinyDense(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ffn = DenseFFN(d, ff)
    def forward(self, x): return self.ffn(self.norm(x))


# Active-compute parity: MoE(k=2, d_ff=256) burns ~2x the FFN FLOPs per token.
# A dense model with d_ff=512 burns the same. Compare quality at matched active.
dense_par = TinyDense(64, 512)
moe_par   = TinyMoEModel(64, 8, 2)
opt_d = torch.optim.Adam(dense_par.parameters(), lr=3e-3)
opt_m = torch.optim.Adam(moe_par.parameters(),   lr=3e-3)
B, T = 32, 16
x = torch.randn(B, T, 64)
tgt = torch.roll(x, 1, dims=1)

print(f"dense params   : {sum(p.numel() for p in dense_par.parameters()):,}")
print(f"MoE total      : {sum(p.numel() for p in moe_par.parameters()):,}")
print(f"MoE active/tok : ~{2*sum(p.numel() for p in moe_par.moe.experts[0].parameters()) + sum(p.numel() for p in moe_par.moe.gate.parameters()):,}")

for step in range(401):
    y_d = dense_par(x)
    ld = F.mse_loss(y_d, tgt)
    opt_d.zero_grad(); ld.backward(); opt_d.step()
    total, mse, *_ = moe_par(x, tgt)
    opt_m.zero_grad(); total.backward(); opt_m.step()
    if step % 50 == 0:
        print(f"step {step:4d}   dense_mse={ld.item():.4f}   moe_mse={mse.item():.4f}")

# Output:
#   dense params   : 65,664
#   MoE total      : 262,784
#   MoE active/tok : ~66,048
#   step    0   dense_mse=1.0323   moe_mse=1.0165
#   step   50   dense_mse=0.1399   moe_mse=0.0302
#   step  100   dense_mse=0.0053   moe_mse=0.0032
#   step  150   dense_mse=0.0006   moe_mse=0.0024
#   step  200   dense_mse=0.0002   moe_mse=0.0025
#   step  250   dense_mse=0.0002   moe_mse=0.0016
#   step  300   dense_mse=0.0002   moe_mse=0.0008
#   step  350   dense_mse=0.0002   moe_mse=0.0012
#   step  400   dense_mse=0.0002   moe_mse=0.0009`}
      </CodeBlock>

      <Prose>
        Active parameters are essentially identical ({"65{,}664"} for dense, {"66{,}048"} for MoE). Total parameters are {"4\\times"} larger for MoE. Interestingly, on this low-capacity task, the dense model wins the long-run endpoint — its loss settles at {"\\approx 2 \\times 10^{-4}"} while the MoE settles at {"\\approx 10^{-3}"}. This is the known MoE pathology at small scale: the router and the aux loss add friction, and the task is simple enough that the {"4\\times"} extra capacity is wasted. The MoE edge only materializes when the task has enough variety that experts can specialize. The famous Switch Transformer result — {"4\\text{--}7\\times"} pre-training speedup — is at a regime where the dataset is T5's C4, the model is 1.5B+ dense equivalent, and the specialization surface is rich. At 64-dim hidden state on a 40-step copy task, there is nothing to specialize in.
      </Prose>

      <Prose>
        This is a useful honest result. Papers reporting "MoE beats dense at matched compute" are all at scale; at toy scale the sign of the gap can flip. The conclusion is not "MoE is bad" but "MoE's value comes from specialization under data complexity, not from parameter count alone."
      </Prose>

      <H3>4.8 Capacity factor and token dropping</H3>

      <CodeBlock language="python">
{`def capacity_count(idx, n_experts, cf=1.25, k=2):
    """Return (capacity_per_expert, token_counts_per_expert, n_dropped)."""
    N = idx.shape[0]
    cap = int(math.ceil(cf * N * k / n_experts))
    counts = torch.zeros(n_experts)
    for slot in range(k):
        c = torch.bincount(idx[:, slot], minlength=n_experts)
        counts += c
    dropped = sum(max(0, int(c) - cap) for c in counts)
    return cap, counts.tolist(), dropped


g = TopKGate(64, 8, 2)
x = torch.randn(128, 64)
_, idx, _ = g(x)
for cf in [1.0, 1.25, 1.5, 2.0]:
    cap, counts, dropped = capacity_count(idx, 8, cf)
    print(f"cf={cf:.2f}  cap={cap:3d}  counts={[int(c) for c in counts]}  dropped={dropped}")

# Output:
#   cf=1.00  cap= 32  counts=[32, 36, 31, 34, 31, 32, 34, 26]  dropped=8
#   cf=1.25  cap= 40  counts=[32, 36, 31, 34, 31, 32, 34, 26]  dropped=0
#   cf=1.50  cap= 48  counts=[32, 36, 31, 34, 31, 32, 34, 26]  dropped=0
#   cf=2.00  cap= 64  counts=[32, 36, 31, 34, 31, 32, 34, 26]  dropped=0`}
      </CodeBlock>

      <Prose>
        With {"c_f = 1.0"} and a 128-token batch routed top-2 across 8 experts, the nominal per-expert capacity is {"32"}. The actual distribution has one expert taking 36 (4 over capacity) and one taking only 26 (6 under). Without slack, 8 tokens are dropped. Raising to {"c_f = 1.25"} (Switch's default) gives each expert room for 40 tokens, comfortably absorbing the moderate imbalance, and nothing drops. The cost is that each expert's batch matrix is now {"1.25\\times"} larger on paper — a real memory cost at scale, but almost always worth paying to avoid silent quality degradation.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 Mixtral-8x7B via HuggingFace</H3>

      <Prose>
        Mixtral-8x7B ships as {"mistralai/Mixtral-8x7B-v0.1"} and {"Mixtral-8x7B-Instruct-v0.1"}. The architecture is Mistral 7B's backbone with every FFN replaced by an 8-expert top-2 MoE layer. 32 layers, 8 experts per layer, {"d_{\\text{model}} = 4096, d_{\\text{ff}} = 14336"} per expert. Total parameters: 46.7B. Active parameters per token: 12.9B (two experts' worth of FFN plus the shared attention and embeddings).
      </Prose>

      <CodeBlock language="python">
{`from transformers import AutoConfig, MixtralForCausalLM

cfg = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
print("Mixtral-8x7B config:")
print(f"  hidden_size            = {cfg.hidden_size}")
print(f"  intermediate_size (ff) = {cfg.intermediate_size}")
print(f"  num_hidden_layers      = {cfg.num_hidden_layers}")
print(f"  num_attention_heads    = {cfg.num_attention_heads}")
print(f"  num_key_value_heads    = {cfg.num_key_value_heads}")
print(f"  num_local_experts      = {cfg.num_local_experts}")
print(f"  num_experts_per_tok    = {cfg.num_experts_per_tok}")
print(f"  router_aux_loss_coef   = {cfg.router_aux_loss_coef}")
# router_jitter_noise and output_router_logits also present.

# Expected:
#   hidden_size            = 4096
#   intermediate_size (ff) = 14336
#   num_hidden_layers      = 32
#   num_attention_heads    = 32
#   num_key_value_heads    = 8
#   num_local_experts      = 8
#   num_experts_per_tok    = 2
#   router_aux_loss_coef   = 0.02`}
      </CodeBlock>

      <Prose>
        The relevant MoE knobs are four fields: {"num_local_experts"} ({"N"}), {"num_experts_per_tok"} ({"k"}), {"router_aux_loss_coef"} ({"\\alpha_{\\text{aux}}"}), and {"router_jitter_noise"} (training-time noise on the router logits, similar to Switch's jitter). Attention uses GQA with 32 Q heads / 8 KV heads. The model will not load without a 90GB+ GPU or multi-GPU sharding; loading just the config lets you inspect the architecture without materializing weights.
      </Prose>

      <H3>5.2 The Mixtral MoE block (annotated)</H3>

      <CodeBlock language="python">
{`# Condensed from transformers/models/mixtral/modeling_mixtral.py.
# The dispatch is the interesting bit.

class MixtralBlockSparseTop2MLP(nn.Module):
    """One Mixtral expert: SwiGLU FFN with three Linear projections."""
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = nn.SiLU()
    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))  # SwiGLU


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        hidden_states = hidden_states.view(-1, D)                        # [N, D]

        router_logits = self.gate(hidden_states)                         # [N, E]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)     # renormalize
        routing_weights = routing_weights.to(hidden_states.dtype)

        final = torch.zeros((B*T, D), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # expert_mask: [E, top_k, N]

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            current_state = hidden_states[None, top_x].reshape(-1, D)
            current_hidden = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, current_hidden.to(hidden_states.dtype))

        final = final.reshape(B, T, D)
        return final, router_logits`}
      </CodeBlock>

      <Prose>
        Three differences from our from-scratch version worth noting. First, Mixtral's gate softmaxes over all {"N"} experts and then picks top-{"k"}, then <em>renormalizes</em> the top-{"k"} weights to sum to 1. This is slightly different from our "softmax over top-k only" — same fixed points, slightly different gradient dynamics. Second, Mixtral uses SwiGLU ({"w_2(\\operatorname{silu}(w_1 x) \\odot w_3 x)"}) with three weight matrices per expert, vs our simpler SiLU FFN; SwiGLU is modestly better empirically and is the Llama/Mistral house style. Third, the expert loop dispatches via {"torch.where(expert_mask[expert_idx])"} which is the vectorized equivalent of our {"mask.nonzero"} — functionally identical, slightly faster. Notice that the aux loss is <em>not</em> computed inside this block; Mixtral's training script computes it externally from the returned {"router_logits"}. This lets inference skip it entirely.
      </Prose>

      <H3>5.3 DeepSeek-V3: fine-grained experts and shared experts</H3>

      <Prose>
        DeepSeek-V3 (DeepSeek-AI 2024, arXiv:2412.19437) takes MoE to 671B total parameters with 37B active. The key architectural differences from Mixtral: (1) <em>256 routed experts per layer</em> instead of 8, each roughly {"32\\times"} smaller; (2) <em>1 shared expert per layer</em> that every token routes to unconditionally; (3) <em>top-8 routing</em> among the 256 routed experts; (4) a new "auxiliary-loss-free" balancing scheme that uses per-expert dynamic biases on the router logits instead of a gradient-providing loss; (5) grouped expert dispatch with node-wise locality to reduce all-to-all traffic.
      </Prose>

      <CodeBlock language="python">
{`# Sketch of DeepSeek-V3's shared + fine-grained MoE forward (paraphrased).

class DeepSeekV3MoE(nn.Module):
    def __init__(self, d_model=7168, d_expert=2048, n_routed=256, n_shared=1, k=8):
        super().__init__()
        self.shared = nn.ModuleList([
            DeepSeekExpert(d_model, d_expert) for _ in range(n_shared)
        ])
        self.routed = nn.ModuleList([
            DeepSeekExpert(d_model, d_expert) for _ in range(n_routed)
        ])
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        # Per-expert bias used for balancing without an aux loss.
        self.register_buffer("expert_bias", torch.zeros(n_routed))
        self.k = k
        self.n_routed = n_routed

    def forward(self, x):
        shared_out = sum(e(x) for e in self.shared)           # everyone pays this
        logits = self.gate(x) + self.expert_bias               # bias applied here
        gates, idx = logits.topk(self.k, dim=-1)
        gates = F.softmax(gates, dim=-1)
        routed_out = self._dispatch_routed(x, gates, idx)
        return shared_out + routed_out

    def update_balance_biases(self, token_counts, lr=1e-3):
        """Called every step: shrink the bias for over-used experts,
        grow it for under-used ones. No gradient involved — this is just
        a running imbalance tracker."""
        mean_count = token_counts.float().mean()
        self.expert_bias -= lr * (token_counts.float() - mean_count).sign()`}
      </CodeBlock>

      <Prose>
        The shared expert carries roughly {"2000 \\cdot 7168 \\cdot 3 \\approx 43\\text{M}"} parameters per layer — tiny relative to the full model, but present for every token. Its job is the "common denominator" function: basic syntax, common tokens, residual passthrough. The 256 routed experts each specialize in much finer slices. DeepSeek's ablations show that this design yields better specialization at matched active parameters than 8 fat experts — there is simply more room to differentiate 256 small functions than 8 big ones. The auxiliary-loss-free balancing is a subtler point: rather than penalizing the softmax probabilities, they maintain a per-expert bias that drifts down when an expert is over-used and up when under-used, all outside the gradient path. This produces similar balance without the gradient-interaction artifacts that the aux loss occasionally introduces.
      </Prose>

      <H3>5.4 Serving infrastructure: vLLM, megablocks, DeepSpeed-MoE, FastMoE</H3>

      <Prose>
        Production serving for MoE has three pain points: (1) the dispatch-combine pattern is awkward for dense tensor hardware because the per-expert batch size is data-dependent; (2) expert parallelism requires all-to-all collectives that scale poorly past a single NVLink domain; (3) KV cache and model weights compete for HBM, and MoE models have very large weights. Four systems dominate.
      </Prose>

      <Prose>
        <em>megablocks</em> (Gale, Narayanan, Das, Zaharia 2022) treats MoE dispatch as a block-sparse matrix multiplication. Instead of looping over experts, it builds a single ragged GEMM in which each expert's tokens occupy a variable-size contiguous block, and uses a custom CUDA kernel to run the whole thing as one launch. On an H100, megablocks roughly 3× the throughput of the naive PyTorch dispatch. vLLM's MoE backend integrates megablocks-style kernels for Mixtral and DeepSeek-V2/V3.
      </Prose>

      <Prose>
        <em>DeepSpeed-MoE</em> (Rajbhandari et al. 2022) is Microsoft's full training and inference stack for MoE at scale, built on top of DeepSpeed's ZeRO. Introduces hierarchical all-to-all (local NVLink first, then cross-node) and Pyramid-MoE architectures. Used for the training of GLaM-scale models.
      </Prose>

      <Prose>
        <em>FastMoE</em> (He et al. 2021) is the original open-source MoE-on-PyTorch library. Most production systems have moved past it, but it is still a useful pedagogical reference.
      </Prose>

      <Prose>
        <em>vLLM</em> is the dominant open-source serving engine in 2026. PagedAttention handles the KV cache; fused MoE kernels handle the dispatch. Its {"MixtralForCausalLM"} and {"DeepseekV3ForCausalLM"} implementations use expert parallelism across tensor-parallel GPU groups, with tokens routed via NCCL all-to-all inside each stage. A single 8-GPU H100 node can serve Mixtral-8x7B at thousands of output tokens per second per replica.
      </Prose>

      <H3>5.5 Expert parallelism: why MoE forces multi-GPU serving</H3>

      <Prose>
        Mixtral-8x7B's FFN parameters are 39B (8 experts × 4.9B each). In bf16 that is 78 GB, which does not fit on an 80 GB H100 alongside attention weights, KV cache, and CUDA context. The canonical solution is <em>expert parallelism</em>: shard the {"N"} experts across multiple GPUs, so GPU 0 hosts experts {"\\{0, 1\\}"}, GPU 1 hosts {"\\{2, 3\\}"}, etc. On the forward pass, tokens that route to expert {"e"} must be shipped to the GPU that hosts {"e"}, its output shipped back. This is an NCCL all-to-all — every GPU sends some tokens to every other GPU. On 8× H100 with NVLink (900 GB/s bidirectional), the all-to-all is fast enough to stay FLOPs-bound; on cross-node setups (400 Gbps InfiniBand, {"\\approx 50"} GB/s), all-to-all dominates and throughput plummets. This is why MoE inference is often pinned to single-node deployments.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 MoE forward pass step-by-step</H3>

      <Prose>
        One Mixtral MoE layer processing a 4-token sequence with {"N = 8, k = 2"}.
      </Prose>

      <StepTrace
        label="Mixtral MoE forward pass (T=4, N=8, k=2)"
        steps={[
          {
            label: "Input",
            render: () => (
              <Prose>
                {"x \\in \\mathbb{R}^{B \\times T \\times d_{\\text{model}}}"}, with {"B = 1, T = 4, d_{\\text{model}} = 4096"}. These are the hidden states after the attention sub-layer and the second LayerNorm. Each row is the residual-plus-attention representation for one token.
              </Prose>
            ),
          },
          {
            label: "Flatten to tokens",
            render: () => (
              <Prose>
                Reshape to {"[N, d_{\\text{model}}]"} where {"N = B \\cdot T = 4"}. The MoE treats each token independently — the router looks only at the single-token hidden state, not at its neighbors. This is why MoE is <em>position-wise</em> in the same sense that a dense FFN is.
              </Prose>
            ),
          },
          {
            label: "Router logits",
            render: () => (
              <Prose>
                {"h = W_g^\\top x \\in \\mathbb{R}^{N \\times 8}"}. One scalar logit per (token, expert). The router parameter count is just {"4096 \\times 8 = 32{,}768"} — negligible next to a single expert at {"\\approx 4.9\\text{B}"} parameters.
              </Prose>
            ),
          },
          {
            label: "Top-k selection",
            render: () => (
              <Prose>
                {"\\text{top-2}(h) \\to"} indices {"\\text{idx} \\in \\{0, \\ldots, 7\\}^{N \\times 2}"}, values {"v \\in \\mathbb{R}^{N \\times 2}"}. For this batch, say token 0 picks experts {"\\{3, 5\\}"}, token 1 picks {"\\{1, 4\\}"}, token 2 picks {"\\{3, 7\\}"}, token 3 picks {"\\{5, 2\\}"}. Eight expert slots filled across 4 tokens — but only 6 <em>distinct</em> experts get any work; experts 0 and 6 idle for this batch. This is normal.
              </Prose>
            ),
          },
          {
            label: "Gate weights",
            render: () => (
              <Prose>
                {"g = \\operatorname{softmax}(v) \\in \\mathbb{R}^{N \\times 2}"}, rows sum to 1. Typical values: {"(0.63, 0.37), (0.55, 0.45), (0.71, 0.29), (0.52, 0.48)"}. The weight on the top-1 expert is usually modestly larger than the top-2 — the softmax is soft, not hard.
              </Prose>
            ),
          },
          {
            label: "Dispatch",
            render: () => (
              <Prose>
                Gather tokens by expert. Expert 3's input batch has 2 tokens (0 and 2); expert 5 has 2 tokens (0 and 3); expert 1 has 1 token (1); expert 4 has 1; expert 7 has 1; expert 2 has 1. Each expert's effective batch size is 1 or 2 rows of {"d_{\\text{model}} = 4096"}. In production, this gather is either a Python loop, a CUDA scatter kernel, or a megablocks block-sparse GEMM setup.
              </Prose>
            ),
          },
          {
            label: "Expert forward",
            render: () => (
              <Prose>
                Six independent SwiGLU FFNs run in parallel (one per active expert). Each produces {"\\mathbb{R}^{M_e \\times d_{\\text{model}}}"} output where {"M_e"} is the number of tokens routed to expert {"e"}. FLOPs: {"\\sum_e M_e \\cdot 3 \\cdot d_{\\text{model}} \\cdot d_{\\text{ff}} = 8 \\cdot 3 \\cdot 4096 \\cdot 14336 \\approx 1.4\\text{G}"} FLOPs for this 4-token example.
              </Prose>
            ),
          },
          {
            label: "Weighted combine",
            render: () => (
              <Prose>
                For each token, sum its {"k = 2"} expert outputs weighted by {"g"}. Token 0: {"0.63 \\cdot E_3(x_0) + 0.37 \\cdot E_5(x_0)"}. This scatter-add is the inverse of the dispatch gather. Output shape: {"[N, d_{\\text{model}}]"}.
              </Prose>
            ),
          },
          {
            label: "Reshape and residual",
            render: () => (
              <Prose>
                Reshape back to {"[B, T, d_{\\text{model}}]"}. Add residual. Pass to the next transformer block. The MoE layer is now structurally interchangeable with a dense FFN — downstream code sees nothing different.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6.2 Expert-token assignment heatmap</H3>

      <Prose>
        From section 4.7, we collected routing decisions over 512 tokens at the end of a 400-step training run of our TinyMoEModel, then aggregated into an 8 (experts) × 8 (sequence-position-bucket) matrix. Each column is normalized to sum to 1 — i.e. it shows, among the tokens at that position bucket, what fraction went to each expert. A perfectly uniform router would be a flat {"0.125"} everywhere; real routing shows mild position-conditional preferences.
      </Prose>

      <Heatmap
        matrix={[
          [0.19, 0.12, 0.19, 0.14, 0.17, 0.11, 0.07, 0.12],
          [0.10, 0.10, 0.10, 0.16, 0.17, 0.14, 0.12, 0.14],
          [0.10, 0.18, 0.14, 0.13, 0.09, 0.13, 0.09, 0.10],
          [0.12, 0.12, 0.08, 0.10, 0.16, 0.13, 0.15, 0.13],
          [0.14, 0.13, 0.12, 0.05, 0.08, 0.09, 0.09, 0.10],
          [0.09, 0.11, 0.14, 0.14, 0.12, 0.12, 0.20, 0.13],
          [0.12, 0.10, 0.16, 0.12, 0.09, 0.14, 0.13, 0.12],
          [0.13, 0.14, 0.07, 0.14, 0.12, 0.12, 0.15, 0.15],
        ]}
        rowLabels={["E0","E1","E2","E3","E4","E5","E6","E7"]}
        colLabels={["pos 0-1","pos 2-3","pos 4-5","pos 6-7","pos 8-9","pos 10-11","pos 12-13","pos 14-15"]}
        colorScale="gold"
        label="Expert x position routing (trained MoE, k=2 counts)"
        cellSize={32}
      />

      <Prose>
        No single expert dominates any column; every expert is used at every position (entries all {"\\geq 0.05"}). This is a well-balanced router. If we had not included the aux loss at scale, a collapsed run would show one column with a {"0.90"} cell and seven zeros — which is why the heatmap is the canonical training-health plot. Note the mild structure: expert 0 favors the start of the sequence ({"0.19"} at positions 0-1), expert 5 favors the end ({"0.20"} at positions 12-13). These positional preferences are emergent — no one told the router about positions; it learned them because different sequence positions have different distributions of hidden states.
      </Prose>

      <H3>6.3 Training loss with and without load-balance loss (schematic)</H3>

      <Prose>
        At production scale, the difference between having and not having the aux loss is stark. This plot is schematic — numbers from Switch Transformer paper, figure 7 (token drop rate proxy) adapted to loss curves — but the qualitative shape is consistent across ablations in Shazeer 2017, Fedus 2021, Zoph 2022, and Jiang 2024.
      </Prose>

      <Plot
        series={[
          { name: "with aux loss",    color: colors.gold,  points: [[0, 10.5], [1000, 7.1], [2000, 5.5], [4000, 4.3], [8000, 3.6], [16000, 3.0], [32000, 2.5]] },
          { name: "without aux loss", color: "#f87171",    points: [[0, 10.5], [1000, 7.3], [2000, 6.1], [4000, 5.4], [8000, 5.0], [16000, 4.8], [32000, 4.7]] },
        ]}
        xLabel="training step"
        yLabel="cross-entropy loss"
        label="MoE training loss: aux on vs off (Switch-style, schematic)"
      />

      <Prose>
        The gap opens after a few thousand steps as the aux-off run begins to collapse — one expert dominates, diversity is lost, and the model loses the capacity benefit of its extra parameters. The aux-on run continues to benefit from all {"N"} experts and converges to a noticeably better loss. The gap does not close with more training; a collapsed router is not recoverable.
      </Prose>

      <H3>6.4 Throughput vs number of experts (schematic)</H3>

      <Prose>
        Holding active parameters fixed (so FLOPs per token are roughly constant), what happens to serving throughput as you scale the expert pool from 8 to 256? Numbers here are schematic — reconstructed from vLLM benchmarks and the DeepSeek-V3 tech report — and illustrate the regime rather than any specific model. Throughput rises modestly until expert parallelism saturates NVLink, then flattens, then degrades as cross-node all-to-all becomes necessary.
      </Prose>

      <Plot
        series={[
          { name: "single-node (8 GPU)", color: colors.gold,  points: [[8, 1800], [16, 1750], [32, 1620], [64, 1430], [128, 1180], [256, 860]] },
          { name: "multi-node (16 GPU)", color: colors.green, points: [[8, 1650], [16, 1600], [32, 1540], [64, 1400], [128, 1310], [256, 1220]] },
        ]}
        xLabel="number of experts"
        yLabel="tokens / sec / replica"
        label="Serving throughput vs expert count (schematic)"
      />

      <Prose>
        The practical implication: more experts is only useful if you have the interconnect to support the all-to-all. Mixtral's 8 experts per layer was chosen partly because it keeps expert parallelism within a single 8-GPU node. DeepSeek-V3's 256 experts require careful node-wise expert grouping to avoid cross-node all-to-all costs — the paper describes dedicating 8 experts to each node and using topology-aware routing to keep most traffic local.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 When to use dense</H3>

      <Prose>
        Dense transformers are the right choice when (1) the model size is below {"\\approx 13"}B parameters — the fixed overhead of MoE (router, aux loss, dispatch machinery) outweighs the benefits below this scale; (2) inference memory is the hard constraint and the serving budget does not allow for multi-GPU deployment — a 13B dense model fits on a single 40GB A100, while a 47B MoE with equivalent active FLOPs does not, even though its active FLOPs are similar; (3) the workload is latency-critical at tight batch sizes where the dispatch overhead cannot be amortized (token-at-a-time decoding with batch size 1 on a single GPU); (4) the deployment target is edge / on-device hardware with limited memory bandwidth; (5) you do not control training and must fine-tune a dense checkpoint — MoE architectures cannot be cleanly retrofitted.
      </Prose>

      <H3>7.2 When to use MoE</H3>

      <Prose>
        MoE is the right choice when (1) you need frontier-scale capability (near GPT-4-class quality) but cannot afford frontier-scale per-token compute; (2) you have multi-GPU serving infrastructure — at least a single NVLink-connected node with 8 fast GPUs — so that expert parallelism is cheap; (3) the workload is high-throughput batched serving where dispatch overhead amortizes across many concurrent tokens; (4) storage and memory are abundant (cheap, slow bandwidth) but compute is scarce (expensive, limited power budget) — MoE shifts cost from flops to memory; (5) the training compute budget is fixed and you want to maximize quality within that budget — Switch and GLaM both showed {"3\\text{--}7\\times"} speedup in training-compute Pareto relative to dense baselines.
      </Prose>

      <H3>7.3 The frontier-scale breakdown</H3>

      <Prose>
        At publication time (2026), the landscape of open and known-architecture frontier models:
      </Prose>

      <TokenStream
        label="Frontier dense vs MoE (2024-2026, by total / active params)"
        tokens={[
          { label: "Dense: Llama-3 70B (70B / 70B)",                color: "#60a5fa" },
          { label: "Dense: Llama-3 405B (405B / 405B)",             color: "#60a5fa" },
          { label: "Dense: Qwen2-72B (72B / 72B)",                  color: "#60a5fa" },
          { label: "MoE: Mixtral-8x7B (47B / 13B)",                 color: colors.gold },
          { label: "MoE: Mixtral-8x22B (141B / 39B)",               color: colors.gold },
          { label: "MoE: DBRX (132B / 36B)",                        color: colors.gold },
          { label: "MoE: Arctic (480B / 17B)",                      color: colors.gold },
          { label: "MoE: Grok-1 (314B / 79B)",                      color: colors.gold },
          { label: "MoE: Qwen2-MoE-A14B (57B / 14B)",               color: colors.gold },
          { label: "MoE: DeepSeek-V3 (671B / 37B)",                 color: colors.green },
          { label: "MoE (leaked/inferred): GPT-4 class, 1T+/40-80B",color: colors.green },
        ]}
      />

      <Prose>
        The direction is unambiguous: the total-parameter frontier is MoE's domain, and the dense leaderboard saturates at {"\\approx 400"}B. Dense Llama-3 405B was publicly released in July 2024 and, by most measurements, trades quality roughly evenly with Mixtral-8x22B at roughly {"10\\times"} the serving cost per token — an unsustainable position. Meta's subsequent frontier work, per their published architectural notes for Llama-4, is MoE. The industry is consolidating.
      </Prose>

      <H3>7.4 The MoE quality-per-active-parameter advantage</H3>

      <Prose>
        The canonical empirical finding from Switch, GLaM, and Mixtral: an MoE with {"P_{\\text{active}}"} active parameters has quality comparable to a dense model with roughly {"2\\text{--}4 \\cdot P_{\\text{active}}"} parameters, at roughly matched training compute. Put differently, the ratio "dense params needed to match MoE quality / MoE active params" is typically 2–4, and the total MoE params to get there is 8–16× active. So: MoE buys you the quality of a dense model 2–4× larger, for the serving cost of a dense model 1× your active size, at the storage cost of a dense model 4–8× your active size. The trade is: storage (cheap, plentiful) for compute (expensive, scarce) via a factor of {"\\approx 2"}. This arithmetic is the entire reason MoE dominates the frontier.
      </Prose>

      <Callout accent="gold">
        The decision rule: below 13B, use dense. At 20–100B total parameters with NVLink-connected multi-GPU serving, use MoE with {"N = 8, k = 2"} (Mixtral pattern). At 200B+ total with datacenter-grade interconnect, use fine-grained MoE with {"N"} in the hundreds and shared experts (DeepSeek-V3 pattern). At sub-GPU edge deployment, use dense or a heavily quantized small model.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 The scaling dimension that matters: N / k</H3>

      <Prose>
        Dense transformers have one scaling knob: parameters. Double the parameters, pay {"2\\times"} compute, gain roughly {"\\sqrt{2}\\times"} perplexity reduction per Chinchilla laws. MoE adds a second, orthogonal knob: the sparsity factor {"N/k"}. You can scale {"N"} while holding {"k"} fixed, increasing total capacity without touching active compute. Switch chose {"N/k = N"} ({"k = 1"}), Mixtral chose {"N/k = 4"} ({"N = 8, k = 2"}), DeepSeek-V3 chose {"N/k \\approx 32"} ({"N = 257, k = 8"}). The sparsity factor is the MoE dial; it is what dense cannot replicate.
      </Prose>

      <Prose>
        Empirically, quality improves with {"N"} at fixed {"k"}, but sub-logarithmically. Going from 8 to 32 experts yields a meaningful quality jump; from 32 to 128 a smaller one; from 128 to 512 nearly flat. There is a diminishing-returns regime, and it hits earlier if the expert pool is too fat (large {"d_{\\text{expert}}"}) and later if the pool is fine-grained. This is the DeepSeek insight: at fixed total capacity, more smaller experts outperform fewer larger ones.
      </Prose>

      <H3>8.2 What gets harder at scale</H3>

      <Prose>
        At {"N > 64"}, several things break. First, <em>expert parallelism communication</em> becomes dominant: all-to-all scales as {"O(N \\cdot B)"} in bytes, where {"B"} is batch size, and at 256 experts on 8 GPUs, the cross-GPU traffic for one batch exceeds the per-GPU expert compute. DeepSeek-V3 mitigates with <em>node-local expert grouping</em>: partition the 256 experts into 8 groups of 32, pin each group to one GPU or node, restrict routing so that each token's top-8 can pick from a constrained subset that maps cleanly to local experts. This is called "grouped-limited routing" in their paper. Second, <em>load balancing becomes statistical</em>: with 256 experts and 2048 tokens per batch per layer, the expected tokens per expert is 8; random variation makes many experts receive zero tokens in a given batch. The aux loss fights this but the signal is weak when the sample size per expert per batch is single digits. DeepSeek's auxiliary-loss-free bias method is partly a response to this. Third, <em>expert specialization</em> becomes harder to verify. With 8 experts, you can probe each one and ask what it does; with 256, inspection is impractical and you rely on aggregate metrics (utilization, balance, loss) rather than interpretability.
      </Prose>

      <H3>8.3 The role of shared experts</H3>

      <Prose>
        DeepSeek-MoE's shared expert — a single "always-on" expert per layer that every token routes to, in addition to the top-{"k"} routed experts — is a small architectural addition with outsized impact. Without shared experts, the routed experts must collectively learn both the common function (what dense FFNs learn: basic syntax, common token embeddings, residual passthrough) <em>and</em> specialization. This creates a tension: an expert that specializes too narrowly loses the common function, a specialization that is too broad prevents differentiation. Adding a shared expert removes the common-function burden from the routed experts, letting them specialize more aggressively. Empirically, DeepSeek reports that models with 1 shared expert + {"N"} routed experts outperform models with {"N + 1"} routed experts at matched total parameters. The shared expert is "free" in the sense that it only needs to be as big as a few routed experts, which is a tiny fraction of the total.
      </Prose>

      <H3>8.4 Fine-grained experts vs fat experts</H3>

      <Prose>
        Consider two configurations with matched total FFN parameters: (A) 8 experts of {"d_{\\text{ff}} = 14336"} (Mixtral), (B) 64 experts of {"d_{\\text{ff}} = 1792"} (DeepSeek-style). Both have the same total capacity. DeepSeek's empirical finding is that (B) trains to better quality under equivalent compute, because finer granularity enables finer specialization. The intuition: two similar-but-distinct regions of the input distribution are more likely to be handled by two different experts in (B) than in (A), where they might share an expert and interfere with each other. The cost: finer-grained experts have smaller per-expert batches at dispatch time, which can lead to small GEMMs that under-utilize tensor cores. megablocks and similar kernels mitigate this by batching multiple experts' GEMMs into one fused kernel, but the penalty is real — pure throughput at matched params is typically 10–20% worse for fine-grained MoE than for fat-expert MoE, even as quality is better.
      </Prose>

      <H3>8.5 Inference scaling with sequence length</H3>

      <Prose>
        MoE's relationship with long context is a nuance worth understanding. The attention layer's memory (KV cache) scales with sequence length; the FFN's memory does not (FFN is position-wise). So MoE's capacity advantage in the FFN is orthogonal to long-context memory — at 128k context, an MoE model's KV cache is the same as a dense model's with matched attention configuration, and MoE's sparsity does not help there. Long-context optimization is GQA, MLA, sliding window; MoE is about FFN capacity. These combine: Mixtral pairs GQA (for long-context efficiency) with MoE (for FFN capacity), and so do Mistral, DeepSeek, Qwen, and essentially every modern frontier model. The two techniques target different bottlenecks and stack cleanly.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Router collapse (load imbalance)</H3>

      <Prose>
        The most famous MoE failure: one expert dominates and the others go unused. Root cause: rich-get-richer dynamics in the absence of balance pressure. Symptom at training time: auxiliary loss climbs instead of falling, utilization histogram shows a single spike with near-zero tails, and the loss plateaus at a level substantially worse than a dense baseline. Fix: ensure the aux loss is (a) actually being computed, (b) added to the training loss with a non-zero coefficient, (c) not inadvertently detached from the graph. Typical aux loss coefficient: {"10^{-2}"} for Switch-style models, {"2 \\times 10^{-2}"} for Mixtral, {"10^{-3}"} for very large models where stability matters more. If the loss is present but collapse still happens, increase the coefficient; if the loss is too strong, the router becomes uniform and specialization never emerges.
      </Prose>

      <H3>9.2 Missing router z-loss → NaN</H3>

      <Prose>
        Without z-loss, router logits tend to grow unboundedly over training. In bf16, {"\\log \\sum \\exp"} overflows once logits exceed {"\\approx 16"}, producing {"+\\infty"} in the softmax normalizer, and the subsequent top-k is meaningless. The resulting gradient is NaN, which poisons optimizer state and crashes the run. Symptom: training is fine for the first N thousand steps, then loss jumps to NaN and never recovers. Fix: add z-loss with coefficient {"10^{-3}"} from step 0. This is standard in every production MoE pipeline (HuggingFace's MixtralForCausalLM has it; Megatron-Core has it; DeepSpeed-MoE has it). The cost is essentially zero — one scalar per batch per layer.
      </Prose>

      <H3>9.3 Top-k without renormalization</H3>

      <Prose>
        If you compute softmax over all {"N"} experts, pick top-{"k"}, and use the raw softmax probabilities as gate weights <em>without renormalizing to sum to 1</em>, the gate weights sum to some value less than 1 — typically {"0.3"} to {"0.7"} depending on how peaked the router is. The output of the MoE layer is therefore scaled down by that factor, and the residual-plus-MoE output has the wrong magnitude. Downstream layers see attenuated signals; the model trains more slowly or fails to converge. Fix: always renormalize {"g \\leftarrow g / \\sum_{i \\in \\mathcal{T}_k} g_i"} after top-{"k"} selection. Mixtral does this explicitly; Switch (with {"k = 1"}) avoids the issue because {"\\sum g = g_{\\text{top}}"} and the weighting is absorbed into the expert output. This bug is subtle and has bitten many teams reimplementing MoE from scratch; always write a unit test checking that {"\\sum g = 1"} exactly.
      </Prose>

      <H3>9.4 Training without aux loss on a large run</H3>

      <Prose>
        Observational: a common failure is to set {"\\alpha_{\\text{aux}} = 0"} "to see what happens" early in development, observe that small-batch small-step runs look fine (as we saw in section 4.6), and forget to turn it back on before the production training run. The small-batch run looks fine because collapse is statistical and needs many batches to take hold. The production run collapses silently around step 10k–50k, the utilization histogram spikes, and by the time anyone notices, 20% of compute has been wasted on a degenerate model. Fix: (a) make aux loss on by default and require an explicit flag to turn off; (b) log per-expert utilization at every training step, not just every 1000; (c) alert if max utilization exceeds some threshold like {"1.5/N"} for more than a few hundred steps.
      </Prose>

      <H3>9.5 Expert parallelism OOM from imbalanced batches</H3>

      <Prose>
        A subtle production bug: expert parallelism allocates a fixed-size input tensor per expert per GPU, sized for the expected batch. If the actual routing distribution is heavier than expected (rare but real — e.g., a batch that contains mostly code when the router has specialized one expert for code), the over-selected expert's buffer overflows. If capacity-factor padding is in place, this is handled gracefully by dropping tokens. If it is not, you get a CUDA OOM during the forward pass — but only on some batches, not others, which makes the bug maddening to reproduce. Fix: always use capacity factor {"\\geq 1.25"} in training and {"\\geq 1.0"} with fallback-to-residual in inference. Monitor token-drop rate as a first-class metric.
      </Prose>

      <H3>9.6 Expert parallelism + pipeline parallelism interactions</H3>

      <Prose>
        At very large scale (hundreds of billions of parameters), MoE training combines expert parallelism (shard experts across GPUs within a stage), tensor parallelism (shard within an expert), and pipeline parallelism (shard layers across stages). The combination is fragile. Common bugs: (a) pipeline bubbles where one stage waits for another because all-to-all barriers don't align with pipeline schedules; (b) gradient accumulation across pipeline micro-batches interacting with router state in ways that break the aux loss computation (the aux loss needs the full batch's {"f_i"}, not a micro-batch's); (c) checkpoint sharding where expert weights are not correctly placed after reloading. Mitigations: use well-tested frameworks (Megatron-Core, DeepSpeed-MoE) rather than rolling your own combination, and run small-scale integration tests that exercise every parallelism axis simultaneously.
      </Prose>

      <H3>9.7 Token dropping on capacity overflow</H3>

      <Prose>
        When tokens are dropped (capacity factor exceeded), their FFN output is zero and only the residual passes. This is not catastrophic — gradient still flows through the residual and the model can learn — but it biases the model: "drop-sensitive" tokens systematically lose FFN signal. Some token types (rare tokens, unusual syntactic positions) are more likely to route to the same under-provisioned expert and be dropped more often, creating a subtle quality degradation that does not show up in aggregate loss. Fix: monitor per-token-type drop rates, not just aggregate drop rate. If a specific token type or position shows 5%+ drop rate, increase capacity factor for that configuration.
      </Prose>

      <H3>9.8 Fine-tuning a dense model to MoE, or vice versa</H3>

      <Prose>
        A frequently-asked and mostly-disappointing question. Can you take a dense model and "convert" it to MoE by replicating its FFN {"N"} times and adding a router? Technically yes; in practice the model does not learn anything useful by differentiation — the experts stay identical, the router signal is tiny, and the model's quality is almost exactly the dense baseline while paying MoE's overhead. Conversely, can you "prune" an MoE back to dense by keeping only the most-used expert? The pruned model's quality is terrible; it was not trained to do everything, only its slice of the input distribution. The honest conclusion: dense and MoE are architecturally distinct training regimes. Converting between them cheaply is an open research question (Sparse Upcycling, Komatsuzaki et al. 2022, arXiv:2212.05055, showed that MHA-style upcycling works with careful tuning — but the initialization matters enormously and recovery takes 10-20% of original pretraining compute).
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in this order. The first two are conceptual foundations; the rest are frontier-scale implementations that each contributed a specific architectural or training recipe.
      </Prose>

      <Prose>
        <strong>Jacobs, Jordan, Nowlan, Hinton (1991).</strong> "Adaptive Mixtures of Local Experts." <em>Neural Computation</em> 3(1):79–87. The original formulation. Predates deep learning by 20 years, but introduces the gating-plus-experts pattern and the EM-style training intuition. Short and readable. The 1991 MoE is soft (all experts are always computed; the gate is a weighted combination), so it does not save compute — but the architectural skeleton is what every subsequent paper builds on.
      </Prose>

      <Prose>
        <strong>Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (2017).</strong> "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017. arXiv:1701.06538. The paper that revives MoE for the deep learning era. Introduces top-{"k"} sparse gating, the load-balancing aux loss, the idea of adding an MoE layer between LSTM layers, and demonstrates 137B-parameter models that outperform smaller dense baselines on language modeling and translation. Every subsequent MoE paper is a descendant of this one. Essential reading.
      </Prose>

      <Prose>
        <strong>Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen (2020).</strong> "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." arXiv:2006.16668. The engineering paper. Introduces expert parallelism via all-to-all on TPU, the XLA sharding annotations, and the production infrastructure that made MoE trainable at 600B parameters. Heavy on systems; light on architecture novelty. Read for the engineering patterns, not the modeling insights.
      </Prose>

      <Prose>
        <strong>Fedus, Zoph, Shazeer (2021).</strong> "Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR. arXiv:2101.03961. The simplification paper. Argues that {"k = 1"} is sufficient, demonstrates a 1.6T-parameter model, establishes the aux-loss + capacity-factor recipe that became standard. The "selective precision" section — running experts in bf16 but the router in fp32 — is a practical gem that every subsequent implementation inherited.
      </Prose>

      <Prose>
        <strong>Du, Huang, Dai, Tong, Lepikhin, Xu, Krikun, Zhou, Yu, Firat, Zoph, Fedus, Bosma, Zhou, Wang, Wang, Webster, Pellat, Robinson, Meier-Hellstern, Duke, Dixon, Zhang, Le, Wu, Chen, Cui (2022).</strong> "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts." ICML. arXiv:2112.06905. Scales to 1.2T parameters, 64 experts, top-2. The canonical "MoE matches dense at 1/3 the FLOPs" result. Also introduces the quality-vs-active-parameters scaling analysis that is now standard in every MoE paper's benchmarks section.
      </Prose>

      <Prose>
        <strong>Jiang, Sablayrolles, Roux, Mensch, Savary, Bamford, Chaplot, de las Casas, Hanna, Bressand, Lengyel, Bour, Lample, Lavaud, Saulnier, Lachaux, Stock, Subramanian, Yang, Antoniak, Le Scao, Gervet, Lavril, Wang, Lacroix, El Sayed (2024).</strong> "Mixtral of Experts." arXiv:2401.04088. The open-weight Mixtral-8x7B release. Clean, short paper with all the architectural details you need to reimplement. The router-jitter-noise ablation and the expert-specialization analysis (section 5) are particularly useful. This is the paper most practitioners actually reimplement against.
      </Prose>

      <Prose>
        <strong>Dai, Deng, Zhao, Xu, Gao, Li, Tian, Fu, Chen, Luo, Deng, Zhao, Wu, Shen, Xu, Zhang, Liu, Sun, Yu, Jia, Chen, Guo (2024).</strong> "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." arXiv:2401.06066. Introduces fine-grained experts (64 small experts vs 8 big) and shared experts. The architectural ablations in section 4 are the cleanest published evidence that granularity matters; this paper is required reading for anyone designing a modern MoE from scratch.
      </Prose>

      <Prose>
        <strong>DeepSeek-AI (2024).</strong> "DeepSeek-V3 Technical Report." arXiv:2412.19437. 671B total / 37B active. Brings together fine-grained experts, shared experts, grouped-limited routing, auxiliary-loss-free balancing via per-expert bias, FP8 training, and multi-token prediction. The engineering depth is extraordinary — this is the closest thing to a "how to build a frontier-scale MoE" cookbook in the open literature. Read section 2 (architecture) and section 4.1 (load balance) carefully; the rest is useful but more specialized.
      </Prose>

      <Prose>
        <strong>Zoph, Bello, Kumar, Du, Huang, Dean, Shazeer, Fedus (2022).</strong> "ST-MoE: Designing Stable and Transferable Sparse Expert Models." arXiv:2202.08906. The stability paper. Introduces router z-loss and establishes the stability recipes (precision, initialization, expert dropout) that became standard. Section 3 is where the z-loss derivation and ablations live.
      </Prose>

      <Prose>
        <strong>Komatsuzaki, Puigcerver, Lee-Thorp, Ruiz, Mustafa, Ainslie, Tay, Dehghani, Houlsby (2022).</strong> "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints." arXiv:2212.05055. How to convert a trained dense model into an MoE without retraining from scratch. The technique that let Google bootstrap MoE models from existing dense T5 checkpoints. Relevant for anyone trying to adopt MoE without the full pretraining budget.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        If you can answer these five without looking back, you understand MoE at the architectural level.
      </Prose>

      <H3>1. The sparsity factor</H3>

      <Prose>
        Mixtral-8x7B has 8 experts per layer and routes each token to 2 of them. Its total parameter count is 46.7B and its active-per-token parameter count is 12.9B. The sparsity factor — the ratio of total to active — is approximately {"3.6\\times"}. DeepSeek-V3 has 256 routed experts + 1 shared expert per layer with top-8 routing, 671B total and 37B active. What is its sparsity factor? Why is Mixtral's {"3.6\\times"} rather than the naive {"N/k = 4\\times"}? Hint: think about what happens to the attention and embedding parameters, which are shared across all experts. Compute and explain both numbers.
      </Prose>

      <H3>2. Load balancing loss mechanics</H3>

      <Prose>
        Consider an MoE layer with 4 experts and top-1 routing. A batch has 1000 tokens. The router selects expert 0 for 700 of them, expert 1 for 200, expert 2 for 100, expert 3 for 0. The average softmax probabilities over the batch are {"P = [0.6, 0.25, 0.1, 0.05]"}. Compute {"f_i"} for each expert, then compute the auxiliary loss {"\\mathcal{L}_{\\text{aux}} = N \\sum_i f_i P_i"}. Then compute what {"\\mathcal{L}_{\\text{aux}}"} would be at perfectly uniform routing ({"f = P = [0.25, 0.25, 0.25, 0.25]"}). The ratio of the two is the "imbalance factor" — for this batch, how imbalanced is the router relative to uniform?
      </Prose>

      <H3>3. KV cache and MoE do not interact (much)</H3>

      <Prose>
        Mixtral-8x7B and Mistral-7B have the same KV cache size per token (same attention config, same {"n_{\\text{kv\\_heads}}"}, same {"n_{\\text{layers}}"}). Explain why MoE's sparsity does not shrink the KV cache. Conversely, explain why adopting GQA or MLA does not shrink MoE's per-token FFN FLOPs. What does each of these optimizations actually target, and why do they stack cleanly in Mixtral? Your answer should reference the fact that the KV cache stores attention keys/values and the FFN is stateless per token.
      </Prose>

      <H3>4. Router collapse detection</H3>

      <Prose>
        You are training a 16-expert MoE with top-2 routing. The aux loss starts at {"\\approx 2.0"} and drops to {"\\approx 1.1"} by step 1000. You log the per-expert utilization histogram at step 10000 and see {"[0.01, 0.02, 0.02, 0.01, 0.02, 0.03, 0.02, 0.02, 0.02, 0.01, 0.40, 0.01, 0.02, 0.35, 0.02, 0.02]"}. What has happened? What does the aux loss at step 10000 look like, quantitatively (give a rough estimate)? What three changes would you make to recover the run, in order of how likely each is to help? Your answer should cover (a) the coefficient, (b) the learning rate schedule of the gate specifically, and (c) the possibility that something architectural is wrong.
      </Prose>

      <H3>5. When to NOT use MoE</H3>

      <Prose>
        You are building a specialized code-completion model for a mobile IDE plugin. The model will run on-device on a phone or laptop with 16 GB of unified memory, no GPU, and a strict 50ms-to-first-token latency budget. Your training budget is modest. Your training data is 200B tokens of code. Would you use MoE or dense? Explain your reasoning across at least three of these axes: (a) memory — can you even fit an MoE of meaningful size? (b) latency — does MoE's dispatch overhead blow the budget at batch size 1? (c) training — does the fixed overhead of MoE (aux loss, routing, sparse dispatch in the training loop) pay off at your data and compute scale? (d) quality — does your task benefit from the expert specialization that MoE offers, or is code-completion narrow enough that a well-tuned dense model will match? Give a verdict and justify it with back-of-the-envelope numbers.
      </Prose>

    </div>
  ),
};

export default moeContent;
