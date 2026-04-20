import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { Heatmap } from "../../components/viz";

const moeTraining = {
  title: "MoE Training & Expert Load Balancing",
  readTime: "13 min",
  content: () => (
    <div>
      <Prose>
        Mixture-of-Experts is how you get a 600B-parameter model that costs the same to run as a 37B-parameter one. The trick is that only a small subset of parameters is ever active on any given token. At each forward pass, a learned router inspects the token and selects a handful of "experts" — typically two out of sixty-four — to process it. Every other expert sits idle. Compute scales with active parameters, not total, so you get a model whose capacity rivals a massive dense network but whose per-token arithmetic budget resembles something far smaller. Mixtral, DeepSeek-V3, Qwen2-MoE, and GLaM are all built on some flavor of this idea. Training MoE cleanly, however, is harder than training dense, and the reason why is worth understanding in detail before reaching for the architecture.
      </Prose>

      <H2>The architecture, concisely</H2>

      <Prose>
        A standard transformer block contains an attention sublayer followed by a feed-forward network (FFN). MoE replaces that single FFN with <Code>N</Code> parallel experts, each itself a full FFN with its own parameters. A router — a small learned linear layer — processes the token's hidden state and produces a score over all <Code>N</Code> experts. The top-<Code>k</Code> scoring experts are selected (almost always <Code>k=1</Code> or <Code>k=2</Code>), and the token is processed by each of them independently. Their outputs are summed, weighted by the router's scores, and written back into the residual stream.
      </Prose>

      <MathBlock>{"y = \\sum_{i=1}^{N} G(x)_i \\cdot E_i(x), \\quad G(x) = \\text{TopK}(\\text{softmax}(W_r x))"}</MathBlock>

      <Prose>
        Here <Code>W_r</Code> is the router's weight matrix, <Code>E_i</Code> is the <Code>i</Code>-th expert FFN, and <Code>G(x)</Code> is a sparse vector — exactly <Code>k</Code> nonzero entries — that weights the selected experts' contributions. The key observation is that the router produces a score over all <Code>N</Code> experts, but only the top-<Code>k</Code> experts actually compute anything. The remaining <Code>N − k</Code> experts contribute a mathematically present but computationally zero term. Their FLOP cost is zero.
      </Prose>

      <Prose>
        This architecture is appealing in theory. In practice, the routing step introduces a training instability that does not arise in dense networks at all, and getting past it is most of what makes MoE training a distinct engineering problem.
      </Prose>

      <H2>The load-balancing problem</H2>

      <Prose>
        Left to its own devices, training converges toward pathological routing. The router learns quickly that certain experts produce lower loss on certain tokens, and so it routes more and more tokens toward those experts. Those experts receive more gradient signal, update faster, and become better still — a self-reinforcing cycle that eventually collapses nearly all traffic onto a small number of "winning" experts. The rest receive almost no tokens, almost no gradient, and effectively stop learning. The model still has the parameter count of a 600B-parameter architecture, but it is using the effective capacity of something far smaller. Worse, the gradient signal through the router itself grows increasingly concentrated, which creates instability.
      </Prose>

      <Heatmap
        label="token-to-expert routing (untrained → auxiliary loss added)"
        rowLabels={["token 0", "token 1", "token 2", "token 3", "token 4", "token 5", "token 6", "token 7"]}
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

      <Prose>
        This is what collapse looks like. Expert 0 receives between 82% and 95% of every token's routing probability — it is not sharing load, it is monopolizing it. Experts 4 and 6 receive essentially nothing from certain tokens. The model in this state is burning memory to hold seven experts that are barely participating. The fixes that prevent this are what separate a trained MoE from a failed one.
      </Prose>

      <H2>Auxiliary load-balance loss</H2>

      <Prose>
        The standard remedy, introduced by Shazeer et al. in 2017 and refined through GShard (2020) and the Switch Transformer (2021), is an explicit regularization term added to the training objective. The idea is to penalize imbalance directly: if some experts are receiving far more than their share of tokens, increase the loss. The penalty term is designed so that the minimum is achieved when routing is perfectly uniform — every expert sees exactly <Code>1/N</Code> of all tokens.
      </Prose>

      <Prose>
        For each expert <Code>i</Code>, define two quantities over a batch. The first, <Code>f_i</Code>, is the fraction of tokens that selected expert <Code>i</Code> as their top-1 choice — a hard, discrete count. The second, <Code>P_i</Code>, is the mean router probability assigned to expert <Code>i</Code> across all tokens in the batch — a soft, differentiable quantity. The auxiliary loss is their product, summed over all experts and scaled by <Code>N</Code>.
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{aux}} = N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i"}</MathBlock>

      <Prose>
        The product <Code>f_i · P_i</Code> is minimized when both quantities equal <Code>1/N</Code>. Crucially, <Code>f_i</Code> is not differentiable — you cannot take a gradient through a hard argmax — but <Code>P_i</Code> is. The gradient flows through <Code>P_i</Code>, which is sufficient: pushing <Code>P_i</Code> toward uniformity directly reshapes the router's softmax outputs, and <Code>f_i</Code> follows as a consequence. This is the standard workaround for the non-differentiability of the routing decision.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def aux_loss(router_logits, selected_experts, num_experts):
    """
    router_logits: (batch*seq, num_experts)
    selected_experts: (batch*seq, k) indices chosen by TopK
    """
    probs = F.softmax(router_logits, dim=-1)
    # P_i — mean router probability across the batch
    P = probs.mean(dim=0)
    # f_i — fraction of tokens whose TOP-1 is expert i
    ones = F.one_hot(selected_experts[:, 0], num_experts).float()
    f = ones.mean(dim=0)
    return num_experts * (f * P).sum()

# Typical weight: 0.01 — large enough to balance, small enough not to hurt main loss.`}
      </CodeBlock>

      <Prose>
        The auxiliary loss coefficient is a sensitive hyperparameter. Too small and the load imbalance persists; too large and the router stops learning useful specialization, reverting to near-random routing that technically balances load but forfeits the capacity advantage. Most published recipes land in the range of 0.01 to 0.001, with 0.01 being the most common starting point. Monitoring both <Code>f_i</Code> entropy and task loss in parallel during early training is the best way to catch a miscalibrated coefficient before it wastes a long run.
      </Prose>

      <H3>Capacity factor and token dropping</H3>

      <Prose>
        Auxiliary loss is a soft pressure. Even with it, routing will not be perfectly uniform, which creates a practical hardware problem: each expert runs on a fixed-size batch for parallelization efficiency. If more tokens arrive at an expert than its allocated capacity allows, the excess tokens have to go somewhere. In practice they are dropped — they skip the FFN entirely and flow through the layer via the residual connection alone, contributing nothing beyond their pre-existing hidden state.
      </Prose>

      <Prose>
        The capacity for each expert is set by a single factor: <Code>capacity = (tokens_per_batch / num_experts) * capacity_factor</Code>. A capacity factor of 1.0 means exactly equal allocation with zero tolerance for imbalance. Any routing skew above uniform immediately causes drops. Practical values are 1.25 during training and 2.0 at inference. Higher capacity factors waste compute on padding — expert batches are padded to their maximum size regardless of how many tokens actually arrive — but they reduce the frequency of drops, which matter more for inference quality than for training, where stochastic drops act as a mild regularizer. Tuning the capacity factor is one of the first dials turned when a new MoE run shows unusually high token-drop rates.
      </Prose>

      <H2>Expert parallelism</H2>

      <Prose>
        An MoE model holds far more parameters than an equivalently performing dense model. Those parameters have to live somewhere, and in practice they are distributed across GPUs. Expert parallelism is the distribution strategy: each GPU holds a contiguous slice of the expert pool. A model with 64 experts on 8 GPUs assigns 8 experts per GPU, and each expert's FFN weights live only on that GPU.
      </Prose>

      <Prose>
        Routing introduces a communication pattern that has no analog in dense training: the all-to-all. After the router assigns each token to its top-<Code>k</Code> experts, tokens must travel to the GPU holding their assigned expert, be processed there, and travel back. This is an all-to-all exchange — every GPU sends data to every other GPU — and its cost scales with both the number of GPUs and the message volume. In large MoE runs, the all-to-all dominates communication time and often determines what routing topologies and capacity factors are even feasible within a training budget. The engineering tradeoff is direct: more experts means more specialization potential but also more cross-GPU traffic per training step.
      </Prose>

      <Prose>
        Expert parallelism typically runs alongside tensor and pipeline parallelism rather than replacing them. A production MoE training job might use tensor parallelism within each expert, pipeline parallelism across transformer layers, and expert parallelism across the expert dimension — three distinct parallelism axes in simultaneous operation. Getting the combination right requires careful configuration of the all-to-all scheduling to avoid communication bottlenecks.
      </Prose>

      <H3>Shared and routed experts (DeepSeek innovation)</H3>

      <Prose>
        DeepSeek-V2 and DeepSeek-V3 introduce a structural refinement to the standard MoE setup: splitting the expert pool into "shared" experts that activate for every token and "routed" experts that activate only when selected by the router. The shared experts handle patterns that appear everywhere — common function words, basic syntax, general-purpose transformations that every token needs regardless of its content. The routed experts can then specialize without each one needing to independently relearn the universals.
      </Prose>

      <Prose>
        The practical gain is modest but consistent. DeepSeek reports 10–20% better performance at matched active-parameter count compared to vanilla MoE without shared experts. The intuition is that in a pure routed setup, every expert implicitly replicates some fraction of basic capacity — there is no expert so specialized that it can ignore grammar entirely. Sharing that baseline capacity explicitly frees the routed experts to specialize more purely, and the total parameter budget goes further. It also stabilizes load balancing slightly, because the shared expert absorbs a steady load regardless of how the router distributes the rest.
      </Prose>

      <H3>Fine-grained experts</H3>

      <Prose>
        Early MoE designs used a small number of large experts — eight or sixteen experts, each roughly the size of the dense FFN they replaced. Modern MoE uses many small ones. DeepSeek-V3 has 256 routed experts plus 1 shared, each roughly one-sixteenth the size of a standard FFN. Mixtral uses 8 experts but activates 2, keeping each individual expert relatively large. The trend is clearly toward more, smaller experts.
      </Prose>

      <Prose>
        The reason is specialization granularity. With 8 experts, the router can learn coarse distinctions — code versus prose, one language versus another, factual recall versus arithmetic. With 256 experts, it can learn far narrower decompositions: informal English versus formal, Python versus JavaScript, biographical facts versus scientific claims. Whether the router actually learns these distinctions rather than arbitrary hash-like partitions depends heavily on the training data and the load-balancing configuration, but the capacity for fine-grained specialization is present in a way it simply is not with 8 experts. Interpretability researchers have found evidence of real semantic specialization in large fine-grained MoE models — specific experts activating disproportionately on code, on named entities, on numeric tokens — though the picture is noisy and the specialization is never perfectly clean.
      </Prose>

      <H2>When MoE is worth it</H2>

      <Prose>
        The architecture is not universally better than dense. The cases where MoE clearly wins: you have enough training tokens for the router to learn meaningful specialization across N experts; you are latency-bound at inference and need to minimize active-parameter compute; you have memory headroom to hold all N experts even though only k are active per token. The memory cost is the most frequently underestimated constraint. A model that activates 37B parameters per token but holds 600B parameters total requires hardware that can hold 600B parameters — the routing efficiency is a compute win but not a memory win. On a setup where memory is tight and throughput can be sacrificed, a dense 37B-parameter model is simply cheaper to serve.
      </Prose>

      <Prose>
        The cases where MoE hurts: training data is small (the router does not have enough signal to learn useful specialization and may simply converge to arbitrary partitions); available memory is the binding constraint; load balancing remains brittle across the training run, causing repeated collapse. Pre-training MoE models also tends to require more careful learning rate scheduling than dense — the router's gradient dynamics interact with the experts' in ways that can produce instability at standard dense hyperparameters, particularly early in training before the auxiliary loss has had time to take effect.
      </Prose>

      <Callout accent="gold">
        MoE trades static parameters for dynamic compute. Cheap to run, expensive to hold, hard to train well — the three constraints that frame every real decision about using it.
      </Callout>

      <Prose>
        There is no free lunch in the tradeoff. A well-trained MoE model at a given active-parameter budget is consistently better than a dense model at the same active-parameter budget — the specialization is real and the empirical gains are not marginal. But that result is contingent on the training going well: routing staying stable, load balancing holding, all-to-all communication not bottlenecking. None of those are guaranteed, and each requires deliberate engineering rather than inheriting defaults from dense training.
      </Prose>

      <Prose>
        Most of the highest-quality open MoE papers come with honest failure-mode sections: load imbalance, routing instability, all-to-all bandwidth. Anyone starting a serious MoE run reads them first — because the architecture is powerful, but the devil genuinely is in the balancing. The Mixtral paper, the GShard paper, the Switch Transformer ablations, and the DeepSeek-V2 technical report are the four documents that cover the live failure modes most completely. None of them make MoE sound easy. That candor is itself informative.
      </Prose>
    </div>
  ),
};

export default moeTraining;
