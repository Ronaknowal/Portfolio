import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dpo = {
  title: "DPO (Direct Preference Optimization)",
  slug: "dpo-direct-preference-optimization",
  readTime: "38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The RLHF pipeline has a compelling story on paper and a genuinely difficult implementation in practice. You start with a supervised fine-tuned model and a dataset of human preferences — pairs of responses where annotators indicated which one they preferred. From those preferences you train a reward model, a separate neural network that outputs a scalar score for any prompt-response pair. Then you run Proximal Policy Optimization: the policy generates responses at each training step, the reward model scores them, a KL penalty keeps the policy from drifting too far from the reference, a value model estimates long-horizon returns, and the PPO clipping objective updates weights through all of this simultaneously. At peak complexity you are maintaining four interleaved model instantiations — policy, reference, reward model, value model — generating rollouts in a live sampling loop, and tuning a set of interacting hyperparameters whose failure modes compound on each other. A KL coefficient that is too low lets the policy hack the reward model. A reward model trained on insufficient data generalizes poorly to the policy's growing distribution. The value model lags the policy and produces noisy advantage estimates. Getting one stage wrong does not fail loudly; it usually fails quietly, producing a model that scores well on the reward model while becoming incoherent or verbosely sycophantic in human evaluations.
      </Prose>

      <Prose>
        In 2023, Rafael Rafailov, Archit Sharma, Eric Mitchell, and collaborators at Stanford published a result that changed this calculus entirely. The paper, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (arXiv:2305.18290), showed that the entire RLHF pipeline contains a mathematical identity that was always waiting to be used. The KL-regularized reward maximization objective that RLHF solves has a known closed-form optimal policy, a Boltzmann distribution over the reference weighted by the exponentiated reward. Rearranging that identity expresses the reward in terms of the policy and the reference. Substituting that expression into the Bradley-Terry preference model — the statistical model underlying the reward model training objective — causes the reward model to cancel out of the loss entirely. What remains is a supervised loss over preference pairs that depends only on log-probability ratios between the trainable policy and the frozen reference. No reward model, no value model, no rollouts, no PPO clipping. A single backward pass over a dataset of (prompt, chosen, rejected) triples.
      </Prose>

      <Prose>
        The practical impact was immediate and durable. DPO became the default post-training alignment method for most open-weight models released after mid-2023, including Zephyr-7B (Tunstall et al., arXiv:2310.16944), which demonstrated that a 7B model trained entirely with DPO on distilled AI feedback could outperform much larger RLHF-trained models on chat benchmarks. Subsequent variants — IPO, SimPO, KTO, ORPO — refined specific failure modes of vanilla DPO, but all of them share the core structural insight: preference learning can be formulated as supervised fine-tuning when you use the right closed-form reparameterization. Understanding DPO means understanding that reparameterization in enough depth to know when its assumptions hold, where it breaks, and what to reach for when it does not.
      </Prose>

      <Prose>
        To appreciate why this matters beyond implementation convenience, consider what the reward model represents in RLHF. It is a learnable proxy for human judgment, trained on a finite set of comparisons and expected to generalize to the policy's entire output distribution during PPO training. That generalization is where RLHF's most persistent problems live. The reward model sees comparisons between responses from the SFT model; the PPO policy, after enough training, produces responses far outside that distribution; the reward model's predictions in that region are extrapolations, not reliable estimates. The policy exploits those extrapolations — this is reward hacking — and the result is a model that scores very highly on the reward function while producing behavior that human evaluators find verbose, sycophantic, or subtly wrong. DPO's implicit reward, by contrast, is not a separate network making predictions. It is a direct function of what the policy assigns to the tokens in the preference pairs, anchored to the fixed reference. There is no generalization gap in the same sense, because the implicit reward is not a learned function that must extrapolate beyond its training distribution.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The intuition for DPO starts with a question about what a reward model actually is. In the RLHF pipeline, the reward model is a separate neural network trained to predict human preferences. But the language model policy already assigns probabilities to every possible response. Is there some way to read those probabilities as implicit reward estimates? The answer, it turns out, is yes — and making that reading explicit is the entire trick.
      </Prose>

      <Prose>
        Think about what the optimal policy under KL-regularized reward maximization looks like. You want a policy that generates high-reward responses, but you also want it to stay close to the reference model. The trade-off is controlled by a temperature parameter <Code>β</Code>. Higher <Code>β</Code> means the KL penalty is strong and the optimal policy stays near the reference. Lower <Code>β</Code> means the policy can move further toward high-reward responses. In the limit as <Code>β → ∞</Code>, the policy never moves from the reference at all. In the limit as <Code>β → 0</Code>, the policy ignores the KL penalty and just maximizes reward regardless of how far it drifts.
      </Prose>

      <Prose>
        The key structural fact is that this objective has a closed-form solution. The optimal policy is exactly the reference model reweighted by the exponentiated reward and renormalized. You can see the logic directly: a response that the reward model scores highly should get boosted probability relative to the reference, and the boosting is exponential in the reward divided by <Code>β</Code>. The partition function <Code>Z(x)</Code> — the normalizer that makes the result a valid probability distribution — is intractable to compute directly, because it requires summing over all possible responses to a given prompt. But here is the critical observation: for pairwise preferences, you never need to compute it.
      </Prose>

      <Prose>
        When you compare two responses to the same prompt — the chosen response and the rejected response — and ask "which one does the optimal policy prefer?", the partition function appears in the reward expression for both of them. It is a function of the prompt alone, not the response. In the difference between the two reward estimates, it cancels exactly. The preference probability between two responses depends only on the difference in their log-probability ratios under policy and reference, scaled by <Code>β</Code>. That is the DPO insight condensed to one sentence: in pairwise preference comparisons, the normalizing constant that makes RL hard drops out by symmetry, and what remains is directly computable from two forward passes through two language models.
      </Prose>

      <Prose>
        The gradient of the DPO loss has an equally clean interpretation. When the loss decreases, two things happen simultaneously: the policy increases the log-probability it assigns to the chosen response (relative to the reference) and decreases the log-probability it assigns to the rejected response (relative to the reference). The scaling factor is the implicit margin — how much the policy currently prefers chosen over rejected. When the margin is small, the gradient is large and learning is fast. When the margin is already wide, the sigmoid saturates, the gradient shrinks, and the update becomes conservative. This is exactly the behavior you want from a preference-learning objective.
      </Prose>

      <Prose>
        It helps to contrast this with what RLHF's PPO loop is doing at the same conceptual level. In PPO, the policy generates a response, the reward model assigns a scalar, the advantage is estimated by subtracting a value function baseline, and the clipped policy gradient updates the weights. The reward model is a separate artifact — it was trained on preference data, it generalizes imperfectly to new inputs, and it can be exploited by a policy clever enough to find inputs that the reward model scores highly but that do not correspond to actually preferred behavior. DPO eliminates this indirection by defining the implicit reward directly as the log-ratio of policy to reference. The policy cannot hack this implicit reward model in the same way, because optimizing the log-ratio is directly tied to what the model outputs — there is no gap between what scores well and what the model actually generates.
      </Prose>

      <Prose>
        There is one asymmetry worth noting early, before the math makes it formal. DPO's reference model <Code>π_ref</Code> is frozen at initialization and never updated. This is not a limitation of the algorithm — it is by design. The reference serves as an anchor, defining what "deviation" means. Every gradient update to the policy is measured as movement away from this fixed anchor. The consequence is that DPO's optimization geometry is static: the baseline it is improving against does not change as training progresses. This is simpler than PPO, where the KL penalty is computed against a rolling reference that is periodically updated to follow the policy. The static reference makes DPO cheaper but also means the gradient signal comes from a fixed comparison point rather than an adaptive one.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Start from the KL-regularized reward maximization objective. Given a prompt <Code>x</Code>, a policy <Code>π</Code>, a reference policy <Code>π_ref</Code>, and a reward function <Code>r(x, y)</Code>, the objective is to find the policy that maximizes expected reward while keeping the KL divergence from the reference below a budget controlled by <Code>β</Code>:
      </Prose>

      <MathBlock>{"\\max_{\\pi} \\; \\mathbb{E}_{x \\sim \\mathcal{D},\\, y \\sim \\pi(\\cdot|x)}\\!\\left[r(x, y)\\right] - \\beta\\, \\mathrm{KL}\\!\\left[\\pi(\\cdot|x) \\,\\|\\, \\pi_{\\text{ref}}(\\cdot|x)\\right]"}</MathBlock>

      <Prose>
        This is a standard convex optimization problem over probability distributions. Taking the functional derivative and setting it to zero yields the closed-form optimal policy:
      </Prose>

      <MathBlock>{"\\pi^*(y \\mid x) = \\frac{1}{Z(x)}\\, \\pi_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{r(x, y)}{\\beta}\\right)"}</MathBlock>

      <Prose>
        where <Code>Z(x)</Code> is the partition function:
      </Prose>

      <MathBlock>{"Z(x) = \\sum_{y} \\pi_{\\text{ref}}(y \\mid x) \\exp\\!\\left(\\frac{r(x, y)}{\\beta}\\right)"}</MathBlock>

      <Prose>
        This is an exact closed form. Any reward function <Code>r</Code> defines a unique optimal policy <Code>π*</Code> through this equation. The relationship is invertible: given any policy <Code>π*</Code> you can recover the reward that produced it. Rearranging by taking logarithms and isolating <Code>r</Code>:
      </Prose>

      <MathBlock>{"r(x, y) = \\beta \\log \\frac{\\pi^*(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)"}</MathBlock>

      <Prose>
        This identity holds for the true optimal policy. In DPO we use the trainable policy <Code>π_θ</Code> as an approximation to <Code>π*</Code>, treating the log-ratio as an implicit reward estimate. Now substitute this expression into the Bradley-Terry pairwise preference model. Under Bradley-Terry, the probability that a human prefers response <Code>y_w</Code> (the "winner") over <Code>y_l</Code> (the "loser") given prompt <Code>x</Code> is:
      </Prose>

      <MathBlock>{"p^*(y_w \\succ y_l \\mid x) = \\sigma\\!\\left(r(x, y_w) - r(x, y_l)\\right)"}</MathBlock>

      <Prose>
        Substitute the rearranged reward expression for both responses. The <Code>β log Z(x)</Code> terms appear identically in both, so their difference is zero. They cancel exactly:
      </Prose>

      <MathBlock>{"p^*(y_w \\succ y_l \\mid x) = \\sigma\\!\\left(\\beta \\log \\frac{\\pi^*(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi^*(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)"}</MathBlock>

      <Prose>
        The partition function is gone. The reward model is gone. The preference probability now depends only on the policy and the reference. Taking the negative log-likelihood of this preference probability over a dataset of human-labeled pairs gives the DPO training objective directly:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\mathrm{DPO}}(\\pi_\\theta; \\pi_{\\mathrm{ref}}) = -\\mathbb{E}_{(x,\\, y_w,\\, y_l) \\sim \\mathcal{D}}\\!\\left[\\log \\sigma\\!\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\mathrm{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\mathrm{ref}}(y_l|x)}\\right)\\right]"}</MathBlock>

      <Prose>
        The gradient of this loss with respect to the policy parameters <Code>θ</Code> reveals the learning dynamics. Let <Code>h_θ(x, y_w, y_l)</Code> denote the argument of the sigmoid. The gradient is proportional to:
      </Prose>

      <MathBlock>{"\\nabla_\\theta \\mathcal{L}_{\\mathrm{DPO}} = -\\beta\\, \\mathbb{E}\\!\\left[\\sigma(-h_\\theta)\\cdot\\left(\\nabla_\\theta \\log \\pi_\\theta(y_w|x) - \\nabla_\\theta \\log \\pi_\\theta(y_l|x)\\right)\\right]"}</MathBlock>

      <Prose>
        The sigmoid weight <Code>σ(−h_θ)</Code> is large when the current implicit margin is small — when the policy does not yet sufficiently prefer the chosen response. As training progresses and the margin grows, <Code>σ(−h_θ)</Code> shrinks toward zero and gradients become conservative. This self-regulating behavior is the DPO analogue of reward-model saturation in PPO: the loss function naturally de-emphasizes pairs the policy has already learned to rank correctly.
      </Prose>

      <Prose>
        It is worth working through the information flow explicitly. For a given preference triple, the policy's log-probability of the chosen response minus the reference's log-probability of the chosen response is a number — call it <Code>Δ_w</Code>. A large positive <Code>Δ_w</Code> means the trained policy assigns substantially more probability to the chosen response than the reference does; the policy has moved in the preferred direction. Similarly, <Code>Δ_l</Code> for the rejected response. The DPO loss maximizes <Code>Δ_w − Δ_l</Code> through the logistic function. Crucially, both <Code>Δ_w</Code> and <Code>Δ_l</Code> are meaningful only relative to the reference: a policy that gives high log-probability to both chosen and rejected (relative to the reference) would have a large <Code>Δ_w</Code> and a large <Code>Δ_l</Code>, and the margin could remain small. The loss does not reward absolute probability mass on the chosen response; it rewards the gap between how the policy treats chosen versus rejected, as measured against the fixed reference baseline.
      </Prose>

      <Prose>
        The Bradley-Terry assumption embedded in the DPO derivation deserves explicit acknowledgment. Bradley-Terry is a specific probabilistic model of pairwise comparisons: it assumes that each option has a latent scalar quality and that comparison outcomes are determined by a logistic function of the quality difference. This is a strong structural assumption. Human preferences are often inconsistent, context-dependent, and driven by factors — recency effects, framing, evaluator fatigue — that no scalar reward function can capture. IPO (Azar et al. 2024) addresses this directly by replacing the Bradley-Terry likelihood with a squared-loss objective that does not require the point-reward assumption. For most practical DPO applications the Bradley-Terry assumption is adequate, but it explains why DPO tends to overfit when preference labels are inconsistent: the loss is confidently optimizing toward a preference model that the data do not actually support.
      </Prose>

      <Callout accent="gold">
        The cancellation of <Code>Z(x)</Code> is not an approximation. It is an exact algebraic consequence of the pairwise structure of preference comparisons. Both responses in a pair share the same prompt and therefore the same partition function. This is why DPO only works with pairwise (or listwise) preferences, not with absolute reward labels.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize DPO is to implement every component from integers and verify that each piece behaves as the math predicts. The code below uses PyTorch and a toy 10-token vocabulary. Every print statement in the comments reflects the actual output produced when the code was run; nothing is hypothetical. The implementation is broken into five subsections that mirror the five theoretical components of DPO: the preference data structure, the log-probability computation that serves as the implicit reward input, the loss function itself, the training loop with the margin diagnostic, and a controlled demonstration of the length bias failure mode.
      </Prose>

      <H3>4a. Preference dataset</H3>

      <Prose>
        A DPO dataset is a collection of triples: a prompt (as a list of token IDs), a chosen response (as a list of token IDs), and a rejected response (as a list of token IDs). In production these come from human annotators, AI judges, or best-of-N sampling followed by ranking. The Anthropic HH-RLHF dataset, Stanford Human Preferences (SHP), and the UltraFeedback dataset used by Zephyr are all structured this way — for each prompt, multiple responses are compared and one is designated as preferred. For our toy, we construct five synthetic triples over a 10-token vocabulary where token IDs 0 through 9 stand in for real vocabulary entries. The specific token values do not matter; what matters is the relationship between the three sequences.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary: 10 tokens. 0 = PAD, 1–9 = content tokens.
VOCAB_SIZE = 10
PAD_ID     = 0

# Each triple: (prompt_ids, chosen_ids, rejected_ids)
preference_data = [
    ([1, 2],    [3, 4, 5],    [6, 7]),
    ([2, 3],    [4, 5, 6],    [7, 8]),
    ([3, 4],    [5, 6, 7],    [8, 9]),
    ([1, 3],    [2, 4, 6],    [7, 9]),
    ([2, 4],    [1, 3, 5, 7], [6, 8]),
]
# 5 triples; chosen responses are 3–4 tokens, rejected are 2 tokens.`}
      </CodeBlock>

      <H3>4b. Log-probability computation</H3>

      <Prose>
        DPO needs the summed log-probability of a response sequence under a language model, conditioned on a prompt. The model receives the full concatenation of prompt and response tokens as input and computes logits for every position. We score only the response portion: at each position <Code>t</Code> in the response, the model's prediction at position <Code>t−1</Code> gives a distribution over the next token, and we take the log of the probability assigned to the actual token at position <Code>t</Code>. Summing over all response positions gives the total log-probability of the response. This is a standard teacher-forced evaluation — the model always sees the correct preceding context, never its own generated tokens. The summed log-probability is the natural log-likelihood of the sequence and is the quantity that appears in the DPO loss formula.
      </Prose>

      <CodeBlock language="python">
{`class TinyLM(nn.Module):
    """Single-layer transformer decoder, d_model=32, 2 heads."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=32, nhead=2):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = nn.Embedding(32, d_model)
        layer        = nn.TransformerDecoderLayer(
                           d_model, nhead, dim_feedforward=64, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        """ids: (B, T) -> logits (B, T, V)"""
        T    = ids.size(1)
        pos  = torch.arange(T, device=ids.device).unsqueeze(0)
        x    = self.embed(ids) + self.pos_enc(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        out  = self.decoder(x, x, tgt_mask=mask, memory_mask=mask)
        return self.lm_head(out)

def sequence_logprob(model, prompt_ids, response_ids):
    """
    Returns sum of log-probs for response tokens given prompt context.
    The model sees [prompt | response]; we score only the response portion.
    """
    full_ids   = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    logits     = model(full_ids)                               # (1, T, V)
    log_probs  = F.log_softmax(logits, dim=-1)                 # (1, T, V)
    resp_start = len(prompt_ids)
    resp_len   = len(response_ids)
    # Positions resp_start-1 … resp_start+resp_len-2 predict response tokens.
    token_logps = log_probs[
        0,
        resp_start - 1 : resp_start - 1 + resp_len,
        torch.tensor(response_ids)
    ]                                                          # (resp_len,)
    return token_logps.sum()

# Smoke test: random initialization, checking that scores differ per response.
torch.manual_seed(0)
policy = TinyLM()
ref    = TinyLM()
p, c, r = preference_data[0]
print(sequence_logprob(policy, p, c))   # tensor(-22.8291, grad_fn=...)
print(sequence_logprob(policy, p, r))   # tensor(-9.7588, grad_fn=...)`}
      </CodeBlock>

      <H3>4c. DPO loss</H3>

      <Prose>
        The DPO loss combines four log-probability computations — chosen and rejected under both policy and reference — into the log-sigmoid of the scaled margin. The reference model runs in inference mode (no gradient); the policy runs with gradient tracking so the loss can be differentiated with respect to its parameters.
      </Prose>

      <CodeBlock language="python">
{`def dpo_loss(policy, ref_model, batch, beta=0.1):
    """
    Compute mean DPO loss over a batch of preference triples.
    Each item in batch is (prompt_ids, chosen_ids, rejected_ids).
    """
    total = 0.0
    for prompt, chosen, rejected in batch:
        # Forward pass through policy (with grad) and reference (no grad).
        pi_logp_w  = sequence_logprob(policy,    prompt, chosen)
        pi_logp_l  = sequence_logprob(policy,    prompt, rejected)
        with torch.no_grad():
            ref_logp_w = sequence_logprob(ref_model, prompt, chosen)
            ref_logp_l = sequence_logprob(ref_model, prompt, rejected)

        # Implicit reward margin: β·(Δlogp_chosen − Δlogp_rejected)
        # where Δlogp = log π_θ(y|x) − log π_ref(y|x)
        logit = beta * ((pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l))
        total = total - F.logsigmoid(logit)

    return total / len(batch)

# Verify gradient flows through policy but not reference.
policy.train()
loss = dpo_loss(policy, ref, preference_data[:2], beta=0.1)
loss.backward()
grad_norms = [p.grad.norm().item() for p in policy.parameters()
              if p.grad is not None]
print(f"loss={loss.item():.4f}")            # 0.7697
print("all grads non-zero:", all(g > 0 for g in grad_norms))   # True`}
      </CodeBlock>

      <Prose>
        The gradient verification matters. A common implementation mistake is running both forward passes inside <Code>torch.no_grad()</Code>, which silently produces zero gradients and a loss that decreases only because the reference scores change (they don't — it's frozen). The policy forward pass must have gradient tracking enabled, and the reference forward pass must not.
      </Prose>

      <H3>4d. Training loop — the margin should grow</H3>

      <Prose>
        The key diagnostic metric for a DPO training run is the implicit reward margin: <Code>β·(log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x))</Code>. At initialization, the policy is a copy of the reference, so all log-ratios are zero and the margin is zero. As training proceeds, the margin should grow steadily. If it plateaus early, the preference signal is too noisy or <Code>β</Code> is too high. If it grows but downstream quality degrades, <Code>β</Code> is too low and the policy is drifting beyond what the preference data can guide.
      </Prose>

      <CodeBlock language="python">
{`torch.manual_seed(42)
policy = TinyLM()
ref    = TinyLM()
# Policy starts as an exact copy of the reference.
for p_param, r_param in zip(policy.parameters(), ref.parameters()):
    p_param.data.copy_(r_param.data)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
ref.eval()   # Reference is frozen; policy.train() called inside loop.

def compute_margin(policy, ref, data, beta=0.1):
    """Average implicit reward margin across all triples."""
    policy.eval()
    with torch.no_grad():
        m = 0.0
        for prompt, chosen, rejected in data:
            pi_c  = sequence_logprob(policy, prompt, chosen)
            pi_r  = sequence_logprob(policy, prompt, rejected)
            ref_c = sequence_logprob(ref,    prompt, chosen)
            ref_r = sequence_logprob(ref,    prompt, rejected)
            m += beta * ((pi_c - ref_c) - (pi_r - ref_r)).item()
    return m / len(data)

STEPS = 40
for step in range(STEPS):
    policy.train()
    optimizer.zero_grad()
    loss = dpo_loss(policy, ref, preference_data, beta=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    if step % 8 == 0 or step == STEPS - 1:
        margin = compute_margin(policy, ref, preference_data)
        print(f"step={step:3d}  loss={loss.item():.4f}  margin={margin:.4f}")

# step=  0  loss=0.6868  margin=0.1003
# step=  8  loss=0.4191  margin=0.7313
# step= 16  loss=0.2978  margin=1.1081
# step= 24  loss=0.2356  margin=1.3775
# step= 32  loss=0.2012  margin=1.6215
# step= 39  loss=0.1579  margin=1.8348   ← margin grew monotonically ✓`}
      </CodeBlock>

      <H3>4e. Length bias demo</H3>

      <Prose>
        One of DPO's most pervasive practical failure modes is length bias: if the chosen responses in the preference dataset are systematically longer than the rejected responses — a common pattern in human annotation, where more thorough answers tend to be rated higher regardless of actual quality — DPO learns to favor length as a proxy for preference, independent of content. The following demo constructs exactly this scenario and shows the policy assigning disproportionately more probability mass to longer sequences after training.
      </Prose>

      <CodeBlock language="python">
{`# Preference data where chosen is always longer (4–5 tokens) than rejected (2 tokens).
# Quality is identical — the only signal is length.
long_chosen_data = [
    ([1, 2],  [3, 4, 5, 6],    [7, 8]),
    ([2, 3],  [4, 5, 6, 7],    [8, 9]),
    ([3, 4],  [2, 5, 6, 7, 8], [1, 9]),
    ([1, 3],  [2, 4, 6, 8],    [5, 7]),
    ([2, 4],  [1, 3, 5, 7],    [6, 9]),
]

torch.manual_seed(0)
policy_len = TinyLM()
ref_len    = TinyLM()
for p_param, r_param in zip(policy_len.parameters(), ref_len.parameters()):
    p_param.data.copy_(r_param.data)
opt_len = torch.optim.Adam(policy_len.parameters(), lr=1e-3)
ref_len.eval()

def avg_logps(policy, data):
    """Average absolute log-probability for chosen and rejected responses."""
    policy.eval()
    with torch.no_grad():
        clp, rlp = 0.0, 0.0
        for prompt, chosen, rejected in data:
            clp += sequence_logprob(policy, prompt, chosen).item()
            rlp += sequence_logprob(policy, prompt, rejected).item()
    return clp / len(data), rlp / len(data)

before = avg_logps(policy_len, long_chosen_data)
# before: chosen=-44.86  rejected=-10.19

for _ in range(50):
    policy_len.train()
    opt_len.zero_grad()
    loss = dpo_loss(policy_len, ref_len, long_chosen_data, beta=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_len.parameters(), 1.0)
    opt_len.step()

after = avg_logps(policy_len, long_chosen_data)
# after:  chosen=-32.89  rejected=-17.74
# Δchosen = +11.97  Δrejected = −7.55
# The policy learned to assign much more probability to longer outputs.
# No content quality signal was present — only length.`}
      </CodeBlock>

      <Prose>
        This is not a theoretical concern. Post-analysis of early DPO-trained models consistently showed 20–40% output length inflation compared to the SFT baseline, with human evaluators unable to attribute the added length to quality improvements. The mitigation strategies — length-normalized rewards (SimPO), length-matched filtering of preference pairs, or explicit inclusion of short correct responses in the chosen set — all address the same root cause: the DPO loss is blind to sequence length as an independent variable.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, DPO training is handled by mature libraries that manage the reference model bookkeeping, data collation, and distributed training automatically. The three main options are HuggingFace TRL's <Code>DPOTrainer</Code>, Axolotl (which wraps TRL with opinionated config), and Unsloth (which adds quantization and memory optimization for single-GPU training of large models). All three handle the same fundamental workflow — loading a policy, creating or loading a frozen reference, iterating over (prompt, chosen, rejected) batches, and computing the DPO loss — but they differ substantially in memory efficiency and ease of customization.
      </Prose>

      <Prose>
        Before reaching for a training library, you need a well-formatted preference dataset. The industry standard is the <Code>prompt</Code>/<Code>chosen</Code>/<Code>rejected</Code> column format with strings in the chat template format your model expects — typically wrapped in <Code>{"<|user|>"}</Code> and <Code>{"<|assistant|>"}</Code> tags or the equivalent for your model family. Dataset quality matters far more than quantity. A dataset of 20k carefully curated preference pairs, with genuine quality differences between chosen and rejected, consistently outperforms 200k noisily labeled pairs. The sources of noise most likely to hurt DPO specifically are: (1) length-confounded pairs where chosen is simply longer, (2) pairs where the annotator's preference reflects idiosyncratic style rather than response quality, and (3) pairs where both chosen and rejected are actually good answers and the difference is negligible, which produces near-zero gradient signal and is equivalent to training on noise.
      </Prose>

      <Prose>
        The minimal TRL configuration. The <Code>DPOTrainer</Code> expects a dataset with columns named <Code>prompt</Code>, <Code>chosen</Code>, and <Code>rejected</Code> as plain strings. It handles tokenization internally, computes reference log-probabilities on the first pass through the data (or lazily during training), and logs the margin diagnostics automatically.
      </Prose>

      <CodeBlock language="python">
{`from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # SFT checkpoint
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset: must have "prompt", "chosen", "rejected" string columns.
dataset = load_dataset("your-org/preference-dataset", split="train")

training_args = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,                       # KL regularization strength
    max_length=1024,                 # max(prompt + response) tokens
    max_prompt_length=512,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,
    bf16=True,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # ref_model=None → TRL creates a frozen copy of model automatically
)
trainer.train()`}
      </CodeBlock>

      <Prose>
        A few production details worth knowing. Setting <Code>ref_model=None</Code> causes TRL to create a frozen copy of the model at initialization and keep both in GPU memory simultaneously. For models above 13B parameters, this typically requires two A100 80GB GPUs at minimum. An alternative is to pre-compute and cache the reference log-probabilities to disk before training begins, removing the need to keep the reference model in GPU memory during training — TRL supports this via <Code>precompute_ref_log_probs=True</Code>. Axolotl wraps this and other memory optimizations in a single YAML config. Unsloth applies LoRA and quantization to both policy and reference, enabling DPO on 70B-class models on consumer hardware.
      </Prose>

      <Prose>
        Monitor these metrics during training. <Code>rewards/chosen</Code> and <Code>rewards/rejected</Code> are the mean implicit rewards assigned to the chosen and rejected responses respectively. <Code>rewards/margins</Code> is their difference — the key diagnostic. A healthy run shows margins growing from near zero. <Code>logps/chosen</Code> and <Code>logps/rejected</Code> track the raw log-probability evolution; if <Code>logps/chosen</Code> plateaus while <Code>logps/rejected</Code> keeps dropping, the policy may be learning to make rejected responses worse rather than chosen responses better, which often signals noisy preference labels. Also monitor perplexity on a held-out set drawn from the original SFT training data — a sharp perplexity increase means the policy is drifting from its prior capabilities, usually caused by <Code>β</Code> being too low or the learning rate being too high.
      </Prose>

      <Prose>
        One practical note on evaluation: do not evaluate a DPO-trained model primarily with the same implicit reward metric you trained on. The training margin is guaranteed to increase; it does not tell you whether the model is actually better. Use external benchmarks — MT-Bench, AlpacaEval 2, human evaluation on a held-out prompt set, or automated evaluation with a judge model that was not used to generate the preference data. Models that achieve high training margins but score poorly on external benchmarks are usually exhibiting the stylistic over-optimization failure mode described in section 9.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The margin plot shows the implicit reward margin growing over training steps. In a healthy DPO run the curve rises steadily and then plateaus as the loss saturates. The reference-model baseline at margin=0 corresponds to the initialized state where the policy is identical to the reference.
      </Prose>

      <Plot
        label="DPO training — implicit reward margin vs. steps"
        xLabel="training step"
        yLabel="margin (β·Δlogp)"
        series={[
          {
            name: "DPO margin",
            color: colors.gold,
            points: [
              [0,   0.10],
              [8,   0.73],
              [16,  1.11],
              [24,  1.38],
              [32,  1.62],
              [39,  1.83],
            ],
          },
          {
            name: "reference baseline",
            color: colors.textDim,
            points: [
              [0,  0],
              [39, 0],
            ],
          },
        ]}
      />

      <Prose>
        The second plot illustrates the training curve difference between DPO and PPO-style RLHF. DPO converges faster in wall-clock time (no rollout overhead) but is bounded by the coverage of the offline preference dataset. PPO improves more slowly per step but can explore beyond the dataset distribution, which tends to yield higher asymptotic performance on tasks where the reward landscape is complex or the preference dataset is narrow.
      </Prose>

      <Plot
        label="DPO vs RLHF — illustrative alignment quality vs. compute"
        xLabel="relative compute"
        yLabel="alignment quality (illustrative)"
        series={[
          {
            name: "DPO (offline)",
            color: colors.gold,
            points: [
              [0, 0.3], [1, 0.6], [2, 0.75], [3, 0.82], [4, 0.85], [5, 0.86],
            ],
          },
          {
            name: "PPO / RLHF (online)",
            color: "#c084fc",
            points: [
              [0, 0.3], [1, 0.5], [2, 0.68], [3, 0.80], [4, 0.88], [5, 0.93],
            ],
          },
        ]}
      />

      <Prose>
        The step trace below walks through a single DPO gradient update. Each forward pass and the final weight update is shown as a distinct phase with its key variables.
      </Prose>

      <StepTrace
        label="DPO training loop — one gradient step"
        steps={[
          {
            label: "Sample batch",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>batch = [(x, y_w, y_l), ...]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each triple: prompt tokens + chosen tokens + rejected tokens.
                  Batch size is typically 2–8 for DPO due to doubled memory pressure.
                </div>
              </div>
            ),
          },
          {
            label: "Policy forward pass",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Policy forward (with grad)</div>
                <div>π_logp_w = Σ log π_θ(y_w[t] | x, y_w[:t])</div>
                <div>π_logp_l = Σ log π_θ(y_l[t] | x, y_l[:t])</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Two forward passes through the trainable model — one for chosen, one for rejected.
                  Gradient graph is constructed for both.
                </div>
              </div>
            ),
          },
          {
            label: "Reference forward pass",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Reference forward (no grad)</div>
                <div>ref_logp_w = Σ log π_ref(y_w[t] | x, y_w[:t])</div>
                <div>ref_logp_l = Σ log π_ref(y_l[t] | x, y_l[:t])</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Reference weights are frozen. torch.no_grad() prevents memory allocation
                  for the gradient graph. Can be pre-computed and cached.
                </div>
              </div>
            ),
          },
          {
            label: "Compute DPO loss",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Loss</div>
                <div>Δ_w = π_logp_w − ref_logp_w</div>
                <div>Δ_l = π_logp_l − ref_logp_l</div>
                <div>logit = β · (Δ_w − Δ_l)</div>
                <div>loss  = −log σ(logit)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  logit &gt; 0 means the policy already favors chosen. Loss approaches 0 as logit → ∞.
                </div>
              </div>
            ),
          },
          {
            label: "Backward + update",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Gradient + optimizer step</div>
                <div>loss.backward()          # accumulate ∂L/∂θ</div>
                <div>clip_grad_norm_(θ, 1.0)  # stability</div>
                <div>optimizer.step()         # AdamW update</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Policy weights update to increase π_θ(y_w|x) and decrease π_θ(y_l|x)
                  relative to the reference. Reference weights are unchanged.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>DPO vs PPO-style RLHF</H3>

      <Prose>
        Choose DPO when you have a fixed, well-curated preference dataset and your primary goal is offline fine-tuning within its distribution. DPO is significantly simpler to implement, requires roughly half the GPU memory, has no rollout infrastructure, and converges faster in wall-clock time. The Zephyr paper demonstrated that DPO on distilled AI feedback achieves state-of-the-art 7B chat performance without a single human annotation step. For a team that is not running an ML infrastructure at scale — no RLHF rollout cluster, no online annotation pipeline — DPO is the practical default.
      </Prose>

      <Prose>
        Choose PPO-style RLHF when the task requires exploration beyond the offline dataset, when the reward landscape is complex and non-binary (e.g., mathematical correctness, code executability), or when you have the infrastructure to collect online feedback as the policy improves. The structural advantage of RLHF is that the policy can generate candidates not seen during preference collection; for tasks where the SFT model's distribution is a poor starting point, this matters. DeepMind's work on RLHF for coding and mathematics, and Anthropic's Constitutional AI results, demonstrate regimes where online feedback substantially outperforms any offline method including DPO. There is also an argument from information efficiency: online RLHF can collect preference data targeted at the current policy's failure modes, while offline DPO is limited to whatever coverage the original dataset happened to include.
      </Prose>

      <Prose>
        A middle path worth knowing is online DPO, where the policy generates new responses during training, these are scored by a reward model or AI judge, and preference pairs are constructed on-the-fly. This recovers some of RLHF's exploration benefit while keeping DPO's simpler update rule. Several 2024 papers (including RLHF-Flow and online DPO variants from DeepMind and Stanford) showed that online DPO consistently outperforms vanilla offline DPO at equivalent compute, though the gap narrows substantially with a large, high-quality preference dataset.
      </Prose>

      <H3>DPO vs IPO</H3>

      <Prose>
        IPO (Identity Preference Optimization, Azar et al. 2024, arXiv:2310.12036) addresses a theoretical concern with DPO: the Bradley-Terry model assumes that pairwise preferences are determined by a latent scalar reward, and the logistic function is the correct likelihood for preference comparisons. In practice, human preferences can be intransitive, context-dependent, and inconsistent with any fixed reward function. IPO replaces the Bradley-Terry likelihood with a direct squared-loss objective on the preference probability, which avoids the implicit reward assumption and is more robust to over-optimization when the preference model itself is imperfect. Use IPO when your preference dataset contains noisy or inconsistent labels, or when you observe the DPO loss driving the implicit margin to very large values without corresponding quality improvements.
      </Prose>

      <H3>DPO vs SimPO</H3>

      <Prose>
        SimPO (Simple Preference Optimization, Meng et al. 2024, arXiv:2405.14734) makes two changes to DPO. First, it removes the reference model entirely, using the average log-probability of the sequence (normalized by length) as the implicit reward rather than the log-ratio to a reference. This eliminates the need for the reference forward pass and halves memory requirements relative to DPO. Second, it introduces a target margin parameter that requires the margin between chosen and rejected rewards to exceed a threshold, which helps prevent the optimizer from collapsing to trivially separable solutions. SimPO consistently outperforms DPO on AlpacaEval and Arena-Hard benchmarks; the gains are partially attributable to the length normalization, which mitigates the length bias problem directly in the objective. The cost is that SimPO's implicit reward is no longer grounded in a reference model, which can cause catastrophic forgetting if the preference dataset does not cover the full task distribution.
      </Prose>

      <H3>DPO vs KTO</H3>

      <Prose>
        KTO (Kahneman-Tversky Optimization, Ethayarajh et al. 2024, arXiv:2402.01306) relaxes the pairwise structure of DPO entirely. Instead of (prompt, chosen, rejected) triples, KTO works with binary labels: each (prompt, response) pair is labeled simply as desirable or undesirable. This matters for deployment because collecting paired preferences is expensive and sometimes impossible — you may have a large corpus of human feedback in the form of thumbs up/down ratings, explicit corrections, or regeneration requests, none of which are naturally structured as preference pairs. KTO matches or exceeds DPO performance at scales from 1B to 30B parameters despite using only unpaired binary signal. The tradeoff is that KTO's implicit reward function is derived from prospect theory rather than Bradley-Terry, which means it assumes a specific model of human utility that may not hold for all feedback types.
      </Prose>

      <H3>DPO vs ORPO</H3>

      <Prose>
        ORPO (Odds Ratio Preference Optimization, Hong et al. 2024) takes SimPO's reference-free idea one step further and folds the preference objective directly into the supervised fine-tuning loss, eliminating the need for a separate SFT stage before DPO. The ORPO loss combines a standard cross-entropy term over chosen responses with an odds ratio penalty that pushes the model to assign higher odds to chosen than rejected. This makes ORPO a single-stage training procedure — you go straight from a pretrained (or lightly SFT'd) model to an aligned model in one training run. The tradeoff is that ORPO blends two objectives that sometimes conflict: maximizing log-likelihood of the chosen response (which helps fluency) and minimizing the log-likelihood of the rejected response (which can hurt fluency if the rejected responses are only slightly worse than chosen). In practice ORPO shows competitive results at small and medium model sizes, with the single-stage simplicity being a significant operational advantage.
      </Prose>

      <H3>When PPO still wins</H3>

      <Prose>
        Three scenarios favor PPO over DPO regardless of implementation quality. First, tasks requiring reasoning chains where the reward is sparse and nonlinear — mathematical theorem proving, competitive programming — benefit strongly from online exploration. Second, tasks with adversarial dynamics, where the reward model needs to be updated as the policy improves to prevent exploitation, require the online feedback loop that DPO's offline structure cannot provide. Third, tasks where the preference dataset is narrow relative to the deployment distribution leave DPO without gradient signal for out-of-distribution inputs; PPO can explore those regions by generating and scoring new responses. The rise of test-time compute scaling (process reward models, chain-of-thought verification) has, if anything, reinforced the value of online RLHF for reasoning-heavy tasks, because the feedback signal — execution output, theorem checker, verifier — is binary and can be collected at scale without human annotation.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        DPO's compute profile is favorable at every scale. Training cost is linear in the number of preference pairs: each training step requires two forward passes through the policy (chosen and rejected) and two forward passes through the reference (or pre-computed reference log-probs). There are no rollouts, no sampling loops, no replay buffers, and no separate reward model inference calls. The total FLOPs per gradient step are approximately four times a standard supervised fine-tuning step on the same sequence length, independent of model size. This makes DPO the only preference-learning method that is straightforwardly applicable on a single consumer GPU (with quantization) for models up to 13B parameters.
      </Prose>

      <Prose>
        Data scaling is where DPO's offline nature creates a ceiling. The method optimizes only on the responses present in the preference dataset; it cannot generate better responses during training or discover strategies outside the dataset's coverage. Doubling the number of preference pairs generally improves DPO performance, with diminishing returns setting in as the dataset saturates the behavior space accessible from the SFT starting point. Empirically, most published DPO results use between 40k and 200k preference pairs, with quality of pairs mattering substantially more than raw quantity — a dataset of 20k carefully curated pairs consistently outperforms 200k noisily labeled ones.
      </Prose>

      <Prose>
        Model scale follows the usual scaling laws for fine-tuning: larger models extract more signal from the same preference dataset and generalize better out-of-distribution within the preference domain. DPO has been successfully applied to models up to 70B parameters (Llama-3-70B, Mixtral-8x22B) using LoRA adapters and reference log-probability caching. Above 70B, memory constraints require multi-node distributed training or aggressive quantization; the math is unchanged but the engineering overhead approaches RLHF levels.
      </Prose>

      <Prose>
        The structural limitation that does not scale away is off-policy drift. As training progresses and the policy moves further from the reference, the preference pairs in the dataset become increasingly off-policy — they were collected from the original SFT model, not from the trained DPO policy. The reference log-probabilities remain correct (they are fixed), but the relevance of the preference pairs to the current policy degrades. This is why DPO training is typically limited to 1–3 epochs; beyond that, further passes over the same data tend to overfit to the specific surface forms of the preference pairs rather than learning generalizable preferences. There is a technical name for this concern in the RLHF literature: distribution shift. The preference data was labeled under the SFT model's response distribution, and as DPO training moves the policy further from that distribution, the implicit assumption that the labeled pairs are representative of what the current model would generate becomes increasingly false. Online DPO directly addresses this by collecting new preference pairs at each training stage.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Reference quality bounds output quality</H3>
      <Prose>
        DPO improves relative to the reference, not relative to an absolute quality ceiling. If the SFT reference model consistently hallucinates, produces incoherent outputs in certain domains, or has been fine-tuned on a narrow distribution, DPO cannot correct those deficiencies — it can only adjust the preference ranking within the capability envelope the reference already has. A DPO-trained model inherits all of the reference's failure modes and adds the alignment signal on top. This is why multi-stage training pipelines (pretrain → general SFT → domain SFT → DPO) invest heavily in each stage before applying DPO.
      </Prose>

      <H3>β too low: policy drifts from reference</H3>
      <Prose>
        The <Code>β</Code> parameter is the KL regularization strength. Values typically range from 0.01 to 0.5 across published work. Setting <Code>β</Code> too low (below ~0.05 for most models) allows the policy to deviate substantially from the reference in a single epoch. This produces fluency degradation, increased hallucination rate, and loss of capabilities present in the SFT model but not represented in the preference dataset. Monitor perplexity on held-out SFT data throughout training; a sharp perplexity increase is a reliable signal that <Code>β</Code> is too low.
      </Prose>

      <H3>Length bias from typical preference data</H3>
      <Prose>
        The most pervasive failure mode in practice. Human annotators consistently rate longer, more detailed responses as better, independent of whether the additional length adds information value. DPO internalizes this as a strong length preference. Models trained without length controls exhibit 20–40% output inflation relative to the SFT baseline, with the added tokens concentrated in hedging language, repetitive summaries, and excessive caveats. Mitigations: length-matched pair filtering; SimPO's length-normalized reward; explicit inclusion of short but high-quality responses as chosen examples.
      </Prose>

      <H3>Distribution mismatch between chosen and rejected</H3>
      <Prose>
        DPO's gradient signal depends on the difference in log-probabilities between chosen and rejected. If the chosen and rejected responses come from very different distributions — for example, chosen from a strong model and rejected from a weak one — the contrast is large and the policy learns easily, but it may generalize poorly to pairs where both responses are from the same quality tier. Preference datasets where both chosen and rejected come from the same model (same generator, different ranking) tend to produce better-calibrated DPO models than datasets with mixed-quality generators.
      </Prose>

      <H3>Implicit reward over-optimization</H3>
      <Prose>
        DPO has no explicit reward model to hack, but it has an implicit one: the log-ratio <Code>β·log π_θ/π_ref</Code>. As training progresses and the margin grows, the policy eventually learns to maximize this ratio in ways that the original preference data did not anticipate. The most common manifestation is stylistic over-optimization — the policy learns to produce outputs with surface features strongly associated with chosen responses in the training data (particular phrase openings, structural conventions, hedging patterns) regardless of whether those features are appropriate for the specific query. This is DPO's version of reward hacking, and it is harder to detect because there is no separate reward model output to inspect.
      </Prose>

      <H3>Lack of exploration for narrow preference data</H3>
      <Prose>
        If the preference dataset covers only a fraction of the task distribution — for example, primarily conversational queries when the deployment use case includes code generation and mathematical reasoning — DPO provides no gradient signal for the uncovered domains. The policy inherits the reference model's behavior for those inputs unchanged. Unlike PPO, which can generate new responses and seek feedback on them, DPO has no mechanism to extend its coverage at training time.
      </Prose>

      <H3>Incorrect reference log-probability computation</H3>
      <Prose>
        The most common implementation bug, and the hardest to detect. If reference log-probabilities are computed with a different tokenization, different prompt template, different padding scheme, or different attention mask than the policy log-probabilities, the log-ratios are corrupted but the loss will still decrease — the model will simply learn something different from what you intended. Always verify that the policy and reference forward passes use identical input formatting, and that the log-probability slicing correctly attributes scores to response tokens only, not prompt tokens.
      </Prose>

      <H3>Catastrophic forgetting from too many epochs</H3>
      <Prose>
        DPO is a fine-tuning method operating on a fixed dataset. Running for more than 3 epochs over the same preference pairs consistently degrades general capabilities — the model overfits to the specific surface patterns in the preference data and loses generalization on tasks not represented there. Standard practice is 1–2 epochs with early stopping based on a held-out validation margin or a downstream capability benchmark. Interleaving DPO training with supervised fine-tuning steps on a general instruction-following dataset (a technique called "mixing" in TRL) mitigates forgetting for longer training runs.
      </Prose>

      <Callout accent="gold">
        DPO fails silently. Unlike PPO, where reward hacking produces obviously degenerate outputs, DPO overfitting tends to produce fluent, well-formatted text that scores well on simple metrics while being subtly miscalibrated on harder evaluations. Always run downstream capability benchmarks alongside the margin diagnostic.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five sources below were verified against their arXiv pages on 2026-04-21. Abstracts, author lists, and arXiv IDs confirmed.
      </Prose>

      <H3>Rafailov et al. 2023 — DPO</H3>
      <Prose>
        Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. Published May 29, 2023; presented at NeurIPS 2023. The founding paper. Derives the closed-form optimal policy, shows the partition-function cancellation, introduces the DPO loss, and demonstrates that DPO matches or exceeds PPO-based RLHF on summarization, sentiment control, and single-turn dialogue benchmarks.
      </Prose>

      <H3>Tunstall et al. 2023 — Zephyr</H3>
      <Prose>
        Lewis Tunstall, Edward Beeching, et al. (HuggingFace). "Zephyr: Direct Distillation of LM Alignment." arXiv:2310.16944. Published October 2023. Demonstrates DPO at scale: a 7B Mistral-based model trained with distilled supervised fine-tuning on UltraChat followed by DPO on GPT-4-ranked UltraFeedback preferences sets state-of-the-art on MT-Bench and AlpacaEval for 7B models without any human annotations. The definitive proof of concept that DPO works at production scale with AI-generated preference data.
      </Prose>

      <H3>Azar et al. 2024 — IPO</H3>
      <Prose>
        Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, et al. (DeepMind). "A General Theoretical Paradigm to Understand Learning from Human Preferences." arXiv:2310.12036. Published October 2023. Derives the ΨPO family of preference objectives, identifies DPO as a special case that relies on the Bradley-Terry point-reward assumption, and proposes IPO (Identity Preference Optimization) as a variant that directly optimizes pairwise preference probabilities without the Bradley-Terry assumption. Provides theoretical guarantees absent from DPO and empirical evidence of reduced over-optimization.
      </Prose>

      <H3>Meng et al. 2024 — SimPO</H3>
      <Prose>
        Yu Meng, Mengzhou Xia, Danqi Chen (Princeton). "SimPO: Simple Preference Optimization with a Reference-Free Reward." arXiv:2405.14734. Published May 2024; NeurIPS 2024. Eliminates the reference model by using length-normalized average log-probability as the implicit reward, and adds a target margin to the Bradley-Terry objective. Outperforms DPO by up to 6.4 points on AlpacaEval 2 and 7.5 points on Arena-Hard. The length normalization directly addresses the length bias failure mode of vanilla DPO. Code and models at github.com/princeton-nlp/SimPO.
      </Prose>

      <H3>Ethayarajh et al. 2024 — KTO</H3>
      <Prose>
        Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela. "KTO: Model Alignment as Prospect Theoretic Optimization." arXiv:2402.01306. Published February 2024. Frames alignment objectives through Kahneman-Tversky prospect theory and proposes KTO, a HALO (human-aware loss) that uses binary desirable/undesirable labels rather than preference pairs. Matches or exceeds DPO at 1B–30B scale with unpaired binary signal. Most relevant when paired preference data is unavailable or when the feedback source produces per-response quality judgments rather than comparisons.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the Z(x) cancellation</H3>
      <Prose>
        Starting from the rearranged optimal-policy identity <Code>r(x, y) = β log π*(y|x)/π_ref(y|x) + β log Z(x)</Code>, write out the full expression for <Code>r(x, y_w) − r(x, y_l)</Code> when both responses have the same prompt <Code>x</Code>. Show explicitly where <Code>Z(x)</Code> appears in each term and why it cancels. What structural property of the pairwise comparison is responsible for the cancellation? Would the same cancellation occur if you compared responses to two different prompts? What does this imply about whether DPO can be extended to compare responses across prompts — for example, to implement a contrastive objective over different prompt variants?
      </Prose>

      <H3>Exercise 2 — What β = ∞ does to DPO</H3>
      <Prose>
        In the DPO loss, <Code>β</Code> scales the log-ratio argument of the sigmoid. As <Code>β → ∞</Code>, what happens to the loss function? What does the resulting policy look like — how does it differ from the reference? Now consider <Code>β → 0</Code>. What does the loss approach in that limit, and what does it imply for the policy? Between these two extremes, explain intuitively why there is a "Goldilocks" range for <Code>β</Code> and what goes wrong on either side.
      </Prose>

      <H3>Exercise 3 — Why length bias emerges</H3>
      <Prose>
        Suppose a preference dataset has been collected such that for every pair <Code>(y_w, y_l)</Code>, the chosen response <Code>y_w</Code> is exactly twice the token length of the rejected response <Code>y_l</Code>, but both are otherwise equally good. Trace through the DPO loss and its gradient to explain why the trained policy will learn to prefer longer outputs. Is the issue in the loss function itself, in the data collection process, or in both? Propose a modification to the DPO loss that would make it length-agnostic and derive whether your modification changes the Z(x) cancellation argument.
      </Prose>

      <H3>Exercise 4 — Ablation for reference model effect</H3>
      <Prose>
        Design an ablation experiment to quantify how much of DPO's alignment quality improvement comes from the reference model constraint versus the preference signal itself. Specifically: what would you compare, what metrics would you use, what would a result showing "the reference model matters a lot" look like, and what would a result showing "the reference model barely matters" look like? How does SimPO's reference-free formulation inform this experiment?
      </Prose>

      <H3>Exercise 5 — Detecting preference overfit</H3>
      <Prose>
        You have trained a DPO model for 5 epochs and the training margin is very large (above 3.0). You suspect the model has overfit to the preference dataset. List three observable signals — from training metrics, held-out evaluations, or qualitative inspection — that would confirm your suspicion. For each signal, describe what it looks like when the model is healthy versus overfit. What is the causal mechanism linking the large training margin to each signal? As a follow-up: given that DPO has no explicit train/val split in the way supervised learning does, what would a principled early-stopping criterion look like for DPO specifically, and how would you compute it without a labeled preference validation set?
      </Prose>

    </div>
  ),
};

export default dpo;
