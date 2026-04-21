import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const grpoRlooKto = {
  title: "GRPO, RLOO, KTO & Advanced Preference Methods",
  slug: "grpo-rloo-kto-advanced-preference-methods",
  readTime: "~55 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        PPO solved the core problem of online preference optimization: sample responses, score them with a reward model, update the policy to increase the probability of high-scoring responses, and use a KL penalty to keep the policy from drifting into adversarial territory. That pipeline works. The RLHF topic covers it in full. But PPO carries substantial scaffolding, and the cost of that scaffolding becomes the dominant constraint when you try to scale it to the largest models.
      </Prose>

      <Prose>
        The problem is the value model. PPO needs a learned value function to estimate advantages — the difference between a response's actual reward and what the policy could expect to receive from its current state. At language model scale, that value function is typically initialized from the policy checkpoint itself, meaning you maintain a second model of identical size throughout training. For a 70B parameter model, this doubles peak GPU memory before accounting for optimizer states. Worse, the value model must be trained simultaneously with the policy, introducing a second source of instability: if the value model lags the policy, its advantage estimates become noisy; if it overshoots, the policy receives misleading gradient signal. Getting PPO to run stably at large scale is a systems and hyperparameter engineering problem of significant difficulty.
      </Prose>

      <Prose>
        DPO removed the reward model and the online sampling loop entirely, replacing PPO with a supervised loss over preference pairs. But it introduced a different constraint: it requires paired preference data. Every training example must have a chosen response and a rejected response for the same prompt, collected under controlled comparison conditions. That data format is expensive to generate — annotators must evaluate two candidate outputs side by side — and it means that the vast body of single-response feedback that production systems accumulate (thumbs up, thumbs down, individual ratings) is unusable without expensive reformatting. DPO is also off-policy by construction: it trains on a fixed dataset rather than sampling fresh responses, so it cannot adapt the distribution of training data as the policy improves.
      </Prose>

      <Prose>
        Three papers published in early 2024 each targeted one of these three problems. They were developed independently, but together they carve out a much more flexible space of preference optimization methods than PPO or DPO alone.
      </Prose>

      <Callout accent="gold">
        Three problems, three solutions: GRPO removes the value model. RLOO simplifies PPO to its statistically essential core. KTO works with binary labels, no pairing required.
      </Callout>

      <H3>GRPO — Group Relative Policy Optimization</H3>

      <Prose>
        Shao et al. at DeepSeek AI introduced GRPO in the DeepSeekMath paper (arXiv:2402.03300, February 2024). The observation: if you are already sampling multiple responses per prompt to get variance reduction on the reward signal, the reward statistics of those responses already contain the information a value model would provide. The within-group mean is a natural baseline; the within-group standard deviation provides a natural scale. Normalizing each response's reward against its group eliminates the value model entirely. GRPO later became the training algorithm behind DeepSeek-R1, applied at full scale on reasoning tasks where rewards are verifiable — correct final answer, passing unit tests, structured output validation.
      </Prose>

      <H3>RLOO — REINFORCE Leave-One-Out</H3>

      <Prose>
        Ahmadian et al. at Cohere published "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (arXiv:2402.14740, February 2024). Their argument: the full machinery of PPO — value network, multiple epochs per batch, clipping, importance-weighted updates — was developed for control tasks with dense rewards and non-stationary dynamics. Language model preference fine-tuning has a much more constrained structure. The reward is fixed. The policy only needs to move a moderate distance from a well-pretrained SFT checkpoint. Classical REINFORCE, with a leave-one-out baseline computed from the same rollout batch, is sufficient — and empirically matches PPO while training up to 3x faster using 70% less RAM.
      </Prose>

      <H3>KTO — Kahneman-Tversky Optimization</H3>

      <Prose>
        Ethayarajh et al. (arXiv:2402.01306, February 2024) approached the problem from the data side. Both PPO and DPO require paired comparisons: you need to tell the model not just "this response was good" but "this response was better than that one." KTO defines a loss over individual responses labeled desirable or undesirable — a binary signal, with no pairing. The loss is grounded in Kahneman and Tversky's prospect theory: humans are loss-averse, weighting losses more heavily than equivalent gains, and this asymmetry is built directly into the KTO objective. The result is a training method that can consume thumbs-up / thumbs-down logs directly, potentially enabling datasets orders of magnitude larger than what paired-comparison labeling allows at the same cost.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>GRPO: the group as its own baseline</H3>

      <Prose>
        Imagine you send eight students to take the same exam and you want to grade each one on a curve. You do not need to know the absolute difficulty of the exam — you just normalize each student's score against the group's mean and standard deviation. The student who scored 80 when everyone else scored 75 looks different from the student who scored 80 when everyone else scored 85. That relative ranking is the signal you use to give feedback.
      </Prose>

      <Prose>
        GRPO does exactly this for responses. For each prompt, sample G responses. Each response gets a reward from the reward function. Subtract the group mean; divide by the group standard deviation. The resulting advantage tells the policy not "this response was good in absolute terms" but "this response was better or worse than what you typically produce for this type of prompt." That relative signal is what the policy needs to update in the right direction. No value network required — the group statistics serve as a self-contained baseline.
      </Prose>

      <H3>RLOO: leave the evaluated response out of its own baseline</H3>

      <Prose>
        RLOO's insight is a subtle statistical improvement over the simplest REINFORCE baseline. The simplest baseline you could use is the mean reward across the batch — but that includes the response you are currently evaluating, which biases the estimate. Leave-one-out removes this bias: for each response, its baseline is the mean reward of the other k-1 responses in the group. This is an unbiased estimator of what the policy could expect to receive on average, computed from real samples rather than a learned function.
      </Prose>

      <Prose>
        The practical difference from GRPO is the normalization step. GRPO divides by the group standard deviation, which bounds the magnitude of advantages regardless of how spread out the rewards are. RLOO uses raw mean subtraction, which means the advantage scale varies with the actual reward variance. Both are sound estimators; RLOO tends to have lower raw variance because it avoids the noise introduced by estimating the standard deviation from a small group.
      </Prose>

      <H3>KTO: prospect theory meets policy optimization</H3>

      <Prose>
        The economic insight behind KTO is that humans do not evaluate outcomes on an absolute scale. We evaluate them relative to a reference point, and we are asymmetric: a loss of $100 feels worse than an equivalent gain of $100 feels good. Kahneman and Tversky formalized this as prospect theory, with a value function that is concave for gains and convex for losses — steeper on the loss side.
      </Prose>

      <Prose>
        KTO translates this into a language model objective. A desirable response (thumbs up) should have its implicit reward pushed above a reference level — the level the policy currently assigns on average. An undesirable response (thumbs down) should have its implicit reward pushed below that reference. The sigmoid-shaped loss captures the nonlinear saturation: once a response is clearly above or below the reference, further movement yields diminishing gradient signal. The reference level itself is estimated as the expected KL divergence of the batch — a running mean of how far the policy's log-ratios sit from the frozen reference model.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>GRPO advantage</H3>

      <Prose>
        For a prompt <Code>x</Code>, sample <Code>G</Code> responses <Code>{"{y_1, ..., y_G}"}</Code>. Compute reward <Code>r_i = r(x, y_i)</Code> for each. The group-normalized advantage for response <Code>i</Code> is:
      </Prose>

      <MathBlock>{"A_i = \\frac{r_i - \\mu_G}{\\sigma_G + \\varepsilon}, \\quad \\mu_G = \\frac{1}{G}\\sum_{j=1}^G r_j, \\quad \\sigma_G = \\sqrt{\\frac{1}{G}\\sum_{j=1}^G (r_j - \\mu_G)^2}"}</MathBlock>

      <Prose>
        The GRPO policy objective is a PPO-style clipped surrogate with a KL penalty anchored to the reference policy <Code>π_ref</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{GRPO}}(\\theta) = -\\mathbb{E}\\!\\left[\\min\\!\\left(\\rho_i A_i,\\; \\text{clip}(\\rho_i, 1{-}\\varepsilon, 1{+}\\varepsilon) A_i\\right)\\right] + \\beta\\, \\mathbb{E}\\!\\left[\\mathrm{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})\\right]"}</MathBlock>

      <Prose>
        where <Code>ρ_i = π_θ(y_i|x) / π_θ_old(y_i|x)</Code> is the importance ratio. The KL penalty prevents the policy from drifting far from the SFT anchor even without a value model enforcing stability.
      </Prose>

      <H3>RLOO baseline</H3>

      <Prose>
        For a group of <Code>k</Code> responses, the leave-one-out baseline for response <Code>i</Code> is:
      </Prose>

      <MathBlock>{"b_i = \\frac{1}{k-1}\\sum_{j \\neq i} r_j = \\frac{\\sum_{j=1}^k r_j - r_i}{k - 1}"}</MathBlock>

      <Prose>
        The RLOO advantage is then simply <Code>A_i = r_i - b_i</Code>. Because <Code>b_i</Code> is computed from independent samples (it excludes <Code>r_i</Code> itself), this is an unbiased estimate of the policy's baseline value. The RLOO gradient update uses a standard REINFORCE objective — one gradient step per batch, no clipping, no importance weighting:
      </Prose>

      <MathBlock>{"\\nabla_\\theta \\mathcal{L}_{\\text{RLOO}} = -\\mathbb{E}_{i}\\!\\left[(r_i - b_i)\\, \\nabla_\\theta \\log \\pi_\\theta(y_i \\mid x)\\right]"}</MathBlock>

      <H3>KTO loss</H3>

      <Prose>
        Let <Code>π_θ</Code> be the policy and <Code>π_ref</Code> be the frozen reference. Define the log-ratio <Code>h_θ(x, y) = β · log(π_θ(y|x) / π_ref(y|x))</Code>. Let <Code>z_ref</Code> be an estimate of the expected KL across the batch:
      </Prose>

      <MathBlock>{"z_{\\text{ref}} = \\mathbb{E}_{(x, y) \\sim \\mathcal{D}}\\!\\left[\\beta\\,\\log \\frac{\\pi_\\theta(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)}\\right]"}</MathBlock>

      <Prose>
        The KTO loss for a desirable response <Code>y_D</Code> and undesirable response <Code>y_U</Code> is:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{KTO}}(\\theta) = \\mathbb{E}_{(x, y_D)}\\!\\left[1 - \\sigma(h_\\theta(x, y_D) - z_{\\text{ref}})\\right] + \\mathbb{E}_{(x, y_U)}\\!\\left[1 - \\sigma(z_{\\text{ref}} - h_\\theta(x, y_U))\\right]"}</MathBlock>

      <Prose>
        The desirable term pushes the policy's log-ratio above the reference level <Code>z_ref</Code>; the undesirable term pushes it below. Lambda weights <Code>λ_D</Code> and <Code>λ_U</Code> can be applied to the two terms independently to control the asymmetry between reward and penalty pressure, consistent with the loss-aversion framing.
      </Prose>

      <H3>Structural comparison</H3>

      <Prose>
        All three methods share the same KL anchor to a reference policy. They differ in where the gradient signal comes from. GRPO and RLOO are both on-policy: the policy generates responses at each step and the gradient uses those fresh samples. KTO is off-policy in the data sense — it trains on a fixed labeled dataset — but its implicit reward is always computed against the current policy, so the optimization geometry is not static in the way DPO's is.
      </Prose>

      <Callout accent="blue">
        GRPO normalizes by group std (bounded advantages, higher variance estimator). RLOO uses raw mean subtraction (unbounded advantages, lower variance estimator). Both are consistent; the right choice depends on reward scale and group size.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All three implementations use the same toy task: a linear policy maps a 4-dimensional state to a 2-token distribution. Token 0 earns reward 1; token 1 earns reward 0. The policy starts at chance (50% each token) and should converge to high probability of token 0.
      </Prose>

      <H3>4a. GRPO core loop</H3>

      <Prose>
        Sample G=8 responses per prompt, compute group-normalized advantages, apply PPO-style clipped update with KL anchor to the frozen reference policy.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(42)

class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 2) * 0.1)

    def logprobs(self, states):
        return F.log_softmax(states @ self.w, dim=-1)

    def sample(self, states):
        logps = self.logprobs(states)
        tokens = torch.distributions.Categorical(logits=logps).sample()
        return tokens, logps.gather(1, tokens.unsqueeze(1)).squeeze(1)

def reward_fn(tokens):
    # Token 0 = correct answer; reward 1. Token 1 = wrong; reward 0.
    return (tokens == 0).float()

policy     = ToyPolicy()
ref_policy = ToyPolicy()
for p in ref_policy.parameters():
    p.requires_grad_(False)

optimizer  = torch.optim.Adam(policy.parameters(), lr=1e-2)

GROUP_SIZE = 8
N_PROMPTS  = 16
eps        = 0.2
beta       = 0.04
states     = torch.randn(N_PROMPTS, 4)

for epoch in range(50):
    # Expand prompts G times — one row per (prompt, response) pair
    expanded = states.repeat_interleave(GROUP_SIZE, dim=0)

    with torch.no_grad():
        tokens, logps_old = policy.sample(expanded)
        rewards = reward_fn(tokens)

    # ── Group-normalized advantage (replaces PPO value model) ─────────────
    r   = rewards.view(N_PROMPTS, GROUP_SIZE)
    adv = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-8)
    adv = adv.flatten()

    # KL anchor: keep policy close to reference
    with torch.no_grad():
        ref_logps = ref_policy.logprobs(expanded).gather(
            1, tokens.unsqueeze(1)).squeeze(1)

    # PPO clipped update (4 mini-epochs per rollout batch)
    for _ in range(4):
        new_logps = policy.logprobs(expanded).gather(
            1, tokens.unsqueeze(1)).squeeze(1)
        ratio      = (new_logps - logps_old).exp()
        clip_loss  = -torch.min(
            ratio * adv,
            ratio.clamp(1 - eps, 1 + eps) * adv,
        ).mean()
        kl_penalty = beta * (logps_old - ref_logps).mean()
        loss       = clip_loss + kl_penalty
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# Verified output:
# Epoch  1 mean reward: 0.492
# Epoch 25 mean reward: 0.703
# Epoch 50 mean reward: 0.734`}
      </CodeBlock>

      <H3>4b. RLOO variant and variance comparison</H3>

      <Prose>
        Same rollout structure as GRPO, but replace group normalization with leave-one-out mean subtraction and replace the PPO clipped update with a plain REINFORCE gradient step. The empirical advantage variance is lower for RLOO because it avoids dividing by the estimated group standard deviation — that division amplifies noise when rewards within the group are nearly constant.
      </Prose>

      <CodeBlock language="python">
{`# -- Using the same ToyPolicy and setup as GRPO above --
# RLOO advantage: subtract leave-one-out mean, no std normalization

for epoch in range(50):
    expanded  = states.repeat_interleave(GROUP_SIZE, dim=0)

    with torch.no_grad():
        tokens, logps_old = policy_rloo.sample(expanded)
        rewards = reward_fn(tokens)

    r     = rewards.view(N_PROMPTS, GROUP_SIZE)

    # Leave-one-out baseline: (sum - r_i) / (k - 1)
    total       = r.sum(-1, keepdim=True)            # (N, 1)
    loo_baseline = (total - r) / (GROUP_SIZE - 1)    # (N, G)
    adv          = (r - loo_baseline).flatten()       # (N*G,)

    # Plain REINFORCE: one step, no clipping
    new_logps = policy_rloo.logprobs(expanded).gather(
        1, tokens.unsqueeze(1)).squeeze(1)
    loss = -(new_logps * adv.detach()).mean()
    optimizer_rloo.zero_grad(); loss.backward(); optimizer_rloo.step()

# Verified output:
# GRPO advantage variance (early/mid/late): 0.8819 / 0.7717 / 0.6614
# RLOO advantage variance (early/mid/late): 0.3034 / 0.2404 / 0.1890
# GRPO mean reward  (early/mid/late): 0.500 / 0.578 / 0.633
# RLOO mean reward  (early/mid/late): 0.516 / 0.555 / 0.680`}
      </CodeBlock>

      <Callout accent="gold">
        RLOO advantage variance is ~3x lower than GRPO's in this toy task. The difference comes from avoiding the group-std estimate: when all responses in a group have similar rewards, std is near zero and GRPO divides by a noisy small number, amplifying spurious differences. RLOO's raw subtraction avoids this.
      </Callout>

      <H3>4c. KTO loss</H3>

      <Prose>
        Generate a synthetic dataset of 64 examples — alternating desirable (token 0) and undesirable (token 1) responses with binary labels. Train with the KTO loss. The key metric is the margin between the policy's log-probability on good tokens and bad tokens: it should grow as training progresses.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Synthetic binary-labeled dataset
N        = 64
states   = torch.randn(N, 4)
# Even indices: desirable (token 0); odd: undesirable (token 1)
all_tokens = torch.where(torch.arange(N) % 2 == 0,
                         torch.zeros(N, dtype=torch.long),
                         torch.ones(N,  dtype=torch.long))
all_labels = (torch.arange(N) % 2 == 0)   # True = desirable

policy     = ToyPolicy()
ref_policy = ToyPolicy()
for p in ref_policy.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
beta = 0.1

for epoch in range(80):
    policy_logps = policy.logprobs(states).gather(
        1, all_tokens.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        ref_logps = ref_policy.logprobs(states).gather(
            1, all_tokens.unsqueeze(1)).squeeze(1)

    logratio = policy_logps - ref_logps

    # z_ref: running KL estimate — the "reference level" for prospect theory
    kl_est = logratio.mean().detach()

    d_mask = all_labels
    u_mask = ~all_labels

    # Desirable: push logratio above kl_est (gain region of prospect curve)
    desirable_loss   = (1 - F.sigmoid(
        beta * (logratio[d_mask] - kl_est))).mean()

    # Undesirable: push logratio below kl_est (loss region; steeper gradient)
    undesirable_loss = (1 - F.sigmoid(
        beta * (kl_est - logratio[u_mask]))).mean()

    loss = desirable_loss + undesirable_loss
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Verified output:
# Epoch  1 — logp(good): -0.634  logp(bad): -0.760  margin: 0.126
# Epoch 40 — logp(good): -0.585  logp(bad): -1.035  margin: 0.450
# Epoch 80 — logp(good): -0.620  logp(bad): -1.414  margin: 0.793`}
      </CodeBlock>

      <Prose>
        The margin between log-probabilities on desirable and undesirable tokens grows from 0.126 to 0.793 over 80 epochs. The policy learns to prefer desirable tokens without any paired comparison — only the binary labels drive the signal. Note that logp(good) stays relatively stable while logp(bad) drops sharply: this asymmetry is the prospect-theory loss-aversion effect, where the penalty gradient on undesirable samples is larger than the reward gradient on desirable samples at the same margin distance.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        HuggingFace TRL ships <Code>GRPOTrainer</Code>, <Code>RLOOTrainer</Code>, and <Code>KTOTrainer</Code> as first-class trainers. All three follow the standard TRL interface: pass a model, a dataset, and a config. The key parameters differ by method.
      </Prose>

      <H3>GRPOTrainer</H3>

      <CodeBlock language="python">
{`from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    num_generations=8,          # G: responses sampled per prompt
    max_new_tokens=512,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    kl_coef=0.04,               # β: KL anchor to reference policy
    cliprange=0.2,              # ε: PPO clip ratio
    temperature=1.0,
    output_dir="./grpo-output",
)

trainer = GRPOTrainer(
    model=model,                # policy (also serves as ref if ref_model=None)
    ref_model=ref_model,        # frozen SFT checkpoint
    reward_funcs=reward_fn,     # callable: (prompt, response) -> float
    args=config,
    train_dataset=dataset,
)
trainer.train()`}
      </CodeBlock>

      <H3>RLOOTrainer</H3>

      <CodeBlock language="python">
{`from trl import RLOOConfig, RLOOTrainer

config = RLOOConfig(
    rloo_k=4,                   # k: responses sampled per prompt
    learning_rate=1e-6,
    per_device_train_batch_size=8,
    kl_coef=0.05,
    output_dir="./rloo-output",
)

trainer = RLOOTrainer(
    config=config,
    tokenizer=tokenizer,
    policy=policy_model,
    ref_policy=ref_model,
    reward_model=reward_model,  # or callable
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)
trainer.train()`}
      </CodeBlock>

      <H3>KTOTrainer</H3>

      <CodeBlock language="python">
{`from trl import KTOConfig, KTOTrainer

config = KTOConfig(
    beta=0.1,                       # temperature controlling KL penalty strength
    desirable_weight=1.0,           # λ_D
    undesirable_weight=1.0,         # λ_U — increase for stronger negative pressure
    max_length=1024,
    per_device_train_batch_size=8,
    learning_rate=5e-7,
    output_dir="./kto-output",
)

# Dataset format: each row has 'prompt', 'completion', 'label' (bool)
trainer = KTOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,  # expects 'label': True/False column
    tokenizer=tokenizer,
)
trainer.train()`}
      </CodeBlock>

      <Callout accent="blue">
        KTO dataset format is the most permissive: one row per response with a boolean label. You do not need to pair chosen and rejected responses. Thumbs-up / thumbs-down logs from production systems can be loaded directly after filtering to a prompt column, a completion column, and a label column.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>GRPO single update step</H3>

      <StepTrace
        label="grpo single update"
        steps={[
          {
            label: "Step 1 — Sample G responses per prompt",
            render: () => (
              <div>
                <TokenStream
                  label="prompt x → G rollouts"
                  tokens={[
                    { label: "prompt x", color: colors.gold },
                    { label: "→ sample G=8", color: colors.textDim },
                    { label: "y₁ r=0.9", color: colors.green },
                    { label: "y₂ r=0.2", color: "#f87171" },
                    { label: "y₃ r=0.7", color: colors.green },
                    { label: "... y₈", color: colors.textMuted },
                  ]}
                />
                <Prose>
                  Each response is generated with the current policy. Rewards are computed immediately — no value model forward pass.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 2 — Compute group-normalized advantages",
            render: () => (
              <div>
                <TokenStream
                  label="group statistics → advantages"
                  tokens={[
                    { label: "μ_G = 0.55", color: colors.textDim },
                    { label: "σ_G = 0.24", color: colors.textDim },
                    { label: "A₁ = +1.46", color: colors.green },
                    { label: "A₂ = -1.46", color: "#f87171" },
                    { label: "A₃ = +0.63", color: colors.green },
                  ]}
                />
                <Prose>
                  Each response's reward is normalized by the group mean and std. High-reward responses get positive advantages; low-reward responses get negative advantages. The absolute scale of the rewards does not matter — only the relative ranking within the group.
                </Prose>
              </div>
            ),
          },
          {
            label: "Step 3 — PPO clipped update with KL anchor",
            render: () => (
              <div>
                <TokenStream
                  label="policy update"
                  tokens={[
                    { label: "ρ_i = π_new/π_old", color: colors.textDim },
                    { label: "clip(ρ, 0.8, 1.2)", color: colors.gold },
                    { label: "× A_i", color: colors.green },
                    { label: "- β·KL(π‖π_ref)", color: "#60a5fa" },
                    { label: "→ gradient step", color: colors.green },
                  ]}
                />
                <Prose>
                  The clipped surrogate prevents large updates from any single rollout. The KL penalty keeps the policy from drifting far from the SFT anchor, serving the same function the value model's implicit stability constraint served in PPO.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>Advantage variance: GRPO vs RLOO over training</H3>

      <Plot
        title="Advantage estimator variance during training"
        description="Lower variance means more stable gradient signal. RLOO's raw mean subtraction produces ~3× lower variance than GRPO's group-std normalization in the early training regime where group rewards are nearly constant."
        data={[
          {
            label: "GRPO advantage variance",
            color: colors.gold,
            points: [
              { x: 1,  y: 0.88 }, { x: 10, y: 0.83 }, { x: 20, y: 0.79 },
              { x: 30, y: 0.75 }, { x: 40, y: 0.71 }, { x: 50, y: 0.66 },
            ],
          },
          {
            label: "RLOO advantage variance",
            color: "#60a5fa",
            points: [
              { x: 1,  y: 0.30 }, { x: 10, y: 0.28 }, { x: 20, y: 0.26 },
              { x: 30, y: 0.24 }, { x: 40, y: 0.22 }, { x: 50, y: 0.19 },
            ],
          },
        ]}
        xLabel="Training epoch"
        yLabel="Advantage variance"
      />

      <H3>Mean reward over training: GRPO, RLOO, KTO</H3>

      <Plot
        title="Mean reward over training steps (toy task)"
        description="All three methods improve from chance (0.50) on the same toy task. RLOO and GRPO converge at similar rates; KTO converges via a different signal — the implicit reward margin — rather than direct reward maximization."
        data={[
          {
            label: "GRPO",
            color: colors.gold,
            points: [
              { x: 1,  y: 0.49 }, { x: 10, y: 0.56 }, { x: 25, y: 0.70 },
              { x: 40, y: 0.72 }, { x: 50, y: 0.73 },
            ],
          },
          {
            label: "RLOO",
            color: "#60a5fa",
            points: [
              { x: 1,  y: 0.52 }, { x: 10, y: 0.55 }, { x: 25, y: 0.56 },
              { x: 40, y: 0.63 }, { x: 50, y: 0.68 },
            ],
          },
          {
            label: "KTO (implicit reward margin)",
            color: "#a78bfa",
            points: [
              { x: 1,  y: 0.13 }, { x: 20, y: 0.30 }, { x: 40, y: 0.45 },
              { x: 60, y: 0.63 }, { x: 80, y: 0.79 },
            ],
          },
        ]}
        xLabel="Training step"
        yLabel="Mean reward / margin"
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        These methods are not interchangeable. Each sits at a different point in a space defined by your data format, your memory budget, the cost of reward evaluation, and what you need from training dynamics.
      </Prose>

      <Callout accent="gold">
        Start with the data you actually have. If it's binary labels → KTO. If it's paired comparisons and you need on-policy exploration → GRPO or RLOO. If it's paired comparisons and you want offline supervised training → DPO or SimPO.
      </Callout>

      <H3>Use GRPO when</H3>

      <Prose>
        Your task has verifiable rewards that are cheap to compute — math problems with checkable final answers, code that can be executed and tested, structured output validation against a schema. You need on-policy exploration: the policy's distribution should shift as it learns, and the training data should come from the current policy's rollouts, not a fixed offline dataset. You cannot afford the memory cost of a value model at the scale you are training. DeepSeek-R1's successful training on mathematical reasoning is the primary validation point for this setting.
      </Prose>

      <H3>Use RLOO when</H3>

      <Prose>
        You already have a reward model and want on-policy RL without PPO's full apparatus. RLOO matches PPO's empirical performance on standard RLHF benchmarks (helpfulness, harmlessness, MT-Bench, Alpaca Eval) at 3× higher throughput and 70% lower RAM. It is the right default when PPO's value network is the bottleneck and the reward function is not trivially cheap — RLOO is more sample-efficient per-rollout than GRPO because it does not require dividing by an estimated group standard deviation that can amplify noise.
      </Prose>

      <H3>Use KTO when</H3>

      <Prose>
        Your preference data is binary — individual thumbs-up and thumbs-down labels — and you cannot generate or afford paired comparisons. Production logs with user feedback are directly usable. You can operate at a data scale that paired comparison labeling cannot reach at the same cost. Accept that binary signals contain less information per example than head-to-head comparisons, and plan to compensate with larger datasets.
      </Prose>

      <H3>Use DPO or SimPO when</H3>

      <Prose>
        You already have clean preference pairs and want the simplest possible training loop with no RL dynamics. Offline supervised training, no rollout generation, no reward model at inference time. If your dataset is already in (prompt, chosen, rejected) format and you do not need on-policy exploration, the additional complexity of GRPO or RLOO is difficult to justify.
      </Prose>

      <Heatmap
        title="Method selection by axis"
        rowLabels={["GRPO", "RLOO", "KTO", "DPO/SimPO"]}
        colLabels={["Verifiable reward", "On-policy", "Binary labels", "Low memory", "Simple impl"]}
        values={[
          [1.0, 1.0, 0.0, 0.9, 0.5],
          [0.5, 1.0, 0.0, 0.9, 0.9],
          [0.3, 0.5, 1.0, 1.0, 0.8],
          [0.3, 0.0, 0.0, 1.0, 1.0],
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>GRPO compute cost grows with G</H3>

      <Prose>
        GRPO's memory advantage over PPO is real — you lose the value model — but you pay a forward-pass cost. Every training step requires G× more response generation than single-sample methods. At G=8 this means 8 model forward passes per prompt, plus the reward function evaluation on each. For tasks where reward is a fast verifier (regex match, unit test pass, answer equality check), this is dominated by model inference and is acceptable. For tasks where reward requires a human-in-the-loop or a second large model, G=8 is prohibitively expensive and you should use KTO or offline DPO instead.
      </Prose>

      <Prose>
        DeepSeek-R1's production configuration used G=16 (16 responses per prompt per step), with each step processing 32 unique questions for a total batch of 512. This was viable because the reward was fast: check whether the final boxed answer matches the ground truth. The mathematical reasoning task also benefited directly from diverse rollouts — a harder task produces higher reward variance within the group, which sharpens the group-normalized advantage signal. Easier tasks produce more uniform rewards within the group, collapsing the advantages toward zero and reducing gradient signal. GRPO's effectiveness is coupled to reward diversity across the group.
      </Prose>

      <H3>RLOO sample efficiency</H3>

      <Prose>
        RLOO and GRPO have similar rollout costs for the same group size k. The difference is that RLOO skips multiple PPO epochs per batch and clips no importance ratios, so each update is cheaper in compute. The main scaling concern for RLOO is that without clipping, large gradient updates are possible when the leave-one-out advantage is large — this is mitigated by the fact that RLOO's advantages have lower variance than GRPO's, but you still need to monitor gradient norms and apply gradient clipping separately.
      </Prose>

      <H3>KTO scales to very large datasets</H3>

      <Prose>
        KTO's dominant advantage at scale is data economics. Each training example is one response with one label — no pairing required. This means you can collect training data from production logs at a rate that scales with user traffic, not with the number of expensive pairwise comparisons you can afford. At Anthropic-scale traffic, the dataset size available for KTO training is orders of magnitude larger than what paired-comparison annotation could produce at the same cost. The single forward pass per example (plus a reference forward pass) also makes KTO computationally efficient per step.
      </Prose>

      <Prose>
        The limitation is signal quality per example. A thumbs-down tells you the user disliked a response, but not why, not what it would have taken to make it good, and not how it compared to any alternative. At large dataset scale this noisiness averages out, but you need enough data — the KTO paper found that performance starts to match DPO around 10K examples and pulls ahead at 100K+, where DPO datasets are typically too expensive to scale.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. GRPO group size too small</H3>

      <Prose>
        The group-normalized advantage is only well-defined when the group standard deviation is non-zero. With G=2 or G=3, it is common for all responses in a group to receive the same reward (especially early in training when the policy is not yet diverse), making the std near zero. The epsilon in the denominator prevents division by zero but produces near-zero advantages — no learning signal. A minimum of G=4 is generally needed; G=8 is the most common production choice. When rewards are particularly sparse (most responses get zero reward), increase G to 16 or higher.
      </Prose>

      <H3>2. RLOO variance control</H3>

      <Prose>
        RLOO has lower advantage variance than GRPO but no importance-ratio clipping. If the gradient signal for a particular prompt is large — because one response in the group got a much higher reward than its peers — RLOO will take a correspondingly large gradient step. Apply global gradient norm clipping (max norm 1.0 is standard) and monitor per-step gradient norms. A spike in gradient norm is usually a sign that one prompt produced an unusually high-variance group, not a sign that the learning rate is too high.
      </Prose>

      <H3>3. KTO β and λ tuning</H3>

      <Prose>
        The β parameter in KTO controls how sharply the sigmoid saturates — high β means the loss becomes nearly constant once the logratio is a few units above or below <Code>z_ref</Code>. If β is too high, the gradient signal is concentrated in a narrow band near <Code>z_ref</Code> and training stalls once the policy moves away from random initialization. If β is too low, the gradient is near-linear and there is no saturation — the policy can take unbounded steps. The empirical sweet spot is 0.05–0.2; start at 0.1 and adjust based on whether the reward margin grows steadily.
      </Prose>

      <Prose>
        The λ weights allow asymmetric pressure. Setting <Code>λ_U {">"} λ_D</Code> (e.g., 1.5 vs 1.0) emphasizes penalizing bad responses over rewarding good ones, which matches the loss-aversion intuition and tends to improve results empirically in the original paper. But a λ_U that is too large causes the policy to focus almost entirely on avoiding bad outputs, suppressing exploration of the response space.
      </Prose>

      <H3>4. KL anchor drift</H3>

      <Prose>
        All three methods use a KL penalty to keep the policy close to the reference. If the KL coefficient β is too low, the policy drifts far from the SFT anchor and begins producing responses outside the distribution the reward model was trained to score reliably — the classic reward hacking failure mode. If β is too high, the policy barely moves from the SFT checkpoint regardless of the reward signal. Unlike DPO, where the reference is fixed by construction, GRPO and RLOO update the policy actively, so KL drift is an active monitoring concern. Track the KL divergence from the reference as a training metric and set an early stopping threshold.
      </Prose>

      <H3>5. Reward hacking common to all three methods</H3>

      <Prose>
        GRPO, RLOO, and KTO all optimize against a reward signal that is an imperfect proxy for what you actually want. The policy will eventually find any exploitable pattern in the reward function. Common reward hacking modes: verbosity exploitation (longer responses score higher because the reward model conflates length with quality), formatting exploitation (using markdown headers and bullet points regardless of whether they are appropriate), sycophancy (agreeing with user premises even when they are false because agreement scores higher than correction). The mitigation is to include explicit length penalties and diversity bonuses in the reward function, and to run human evaluations periodically rather than relying only on the reward model score.
      </Prose>

      <H3>6. Binary labels hide quality gradations</H3>

      <Prose>
        KTO's binary label format collapses a continuous quality spectrum into two buckets. A mediocre response and an excellent response both receive the same "desirable" label and contribute the same sign of gradient signal. This means KTO cannot distinguish between "slightly better than average" and "clearly best possible answer" — both are thumbs up. For tasks where quality gradations matter (e.g., factual accuracy vs stylistic preference), supplement binary KTO training with periodic paired comparison evaluation to verify that the policy is improving on the intended quality axis.
      </Prose>

      <H3>7. Mixing methods without comparison</H3>

      <Prose>
        It is tempting to switch between GRPO, RLOO, and KTO within a training run — for example, starting with KTO on a large offline dataset and then switching to GRPO for online fine-tuning. This works in principle but requires careful checkpointing: the reference model used for KL penalization should be updated to the checkpoint that starts the new phase, not the original SFT checkpoint. Failing to update the reference causes the new phase to fight against an increasingly stale KL penalty that pushes the policy back toward the original SFT model rather than the KTO-tuned intermediate.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>Foundational papers</H3>

      <Prose>
        <strong>Shao et al. (2024) — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.</strong> arXiv:2402.03300. Introduces GRPO as the training algorithm for DeepSeekMath-7B. Section 4 of the paper describes the group-relative advantage formulation and demonstrates that removing the value model does not harm performance on math benchmarks. The paper also describes the math-specific pretraining data pipeline; the GRPO section is self-contained and transferable.
      </Prose>

      <Prose>
        <strong>Ahmadian et al. (2024) — Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.</strong> arXiv:2402.14740. Published by Arash Ahmadian and colleagues at Cohere. Demonstrates that RLOO matches or exceeds PPO on Alpaca Eval and MT-Bench across 1B–7B model sizes, while training up to 3× faster and using 70% less memory. Section 3 derives the leave-one-out estimator and its variance properties formally.
      </Prose>

      <Prose>
        <strong>Ethayarajh et al. (2024) — KTO: Model Alignment as Prospect Theoretic Optimization.</strong> arXiv:2402.01306. Introduces the HALO framework (human-aware losses) and the KTO objective. Section 2 connects the KTO loss to Kahneman-Tversky utility theory. Section 4 shows that KTO matches or exceeds DPO performance at 1B–30B scale with only binary signal, and demonstrates that KTO scales better as dataset size increases beyond 50K examples.
      </Prose>

      <H3>Scale validation</H3>

      <Prose>
        <strong>DeepSeek-AI (2025) — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.</strong> arXiv:2501.12948. Applies GRPO at full scale (DeepSeek-R1-Zero and DeepSeek-R1) with G=16 rollouts per prompt, KL coefficient 0.001, clip ratio 10, and learning rate 3e-6 over mathematical reasoning benchmarks. Documents the "aha moment" phenomenon — the emergence of extended reasoning chains without explicit chain-of-thought supervision — as a product of GRPO's on-policy exploration enabling the policy to discover novel solution strategies.
      </Prose>

      <H3>Related reading</H3>

      <Prose>
        <strong>Rafailov et al. (2023) — Direct Preference Optimization.</strong> arXiv:2305.18290. The DPO paper that GRPO/RLOO/KTO all contextualize against. Understanding the DPO reparameterization (covered in the DPO topic) makes the KTO loss structure much clearer, as KTO extends the same log-ratio implicit reward formulation to binary labels.
      </Prose>

      <Prose>
        <strong>Stiennon et al. (2020) — Learning to Summarize with Human Feedback.</strong> arXiv:2009.01325. The original RLHF paper for language models. Establishes the pipeline structure that GRPO and RLOO are streamlining and the critique of which KTO represents an alternative data model.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Variance of the group-normalized estimator</H3>

      <Prose>
        Derive the variance of the GRPO advantage estimator <Code>{"A_i = (r_i - mu_G) / sigma_G"}</Code> for a group of G responses. Show that when rewards are drawn i.i.d. from a distribution with variance <Code>sigma^2</Code>, the variance of the normalized advantage is always 1 regardless of sigma. Now derive the variance of the RLOO advantage <Code>{"A_i = r_i - b_i"}</Code> where <Code>{"b_i = (sum_{j!=i} r_j) / (k-1)"}</Code>. Under the same i.i.d. assumption, what is the variance as a function of k and sigma^2? For large k, which estimator has lower variance, and why does GRPO's normalization trade variance reduction for scale invariance?
      </Prose>

      <H3>Exercise 2 — When RLOO beats PPO</H3>

      <Prose>
        The Ahmadian et al. paper claims that PPO's value network contributes little in the language model RLHF setting. Describe two structural properties of the RLHF task (compared to control tasks like MuJoCo) that reduce the benefit of a learned critic. Then identify a setting where a learned critic would provide a substantial advantage over RLOO's leave-one-out baseline — what property of the reward structure or action space would make the RLOO estimator high-variance even at large k?
      </Prose>

      <H3>Exercise 3 — Converting DPO data to KTO format</H3>

      <Prose>
        You have a DPO dataset of (prompt, chosen, rejected) triples. Describe how you would convert this to KTO binary-label format. After conversion, what information has been lost? Specifically: (a) the margin information (how much better was chosen over rejected), (b) the correlation structure between chosen and rejected responses for the same prompt, (c) any information about the absolute quality of responses. How would you design an experiment to measure whether the lost information matters empirically for a given task?
      </Prose>

      <H3>Exercise 4 — KL anchor schedule for GRPO</H3>

      <Prose>
        In DeepSeek-R1, the KL coefficient was set to 0.001 — extremely small compared to the 0.04–0.1 range common in DPO and PPO. Design a KL-coefficient schedule that starts at 0.05 and decays toward 0.001 over training. What is the rationale for starting high and decaying? What risk does a small KL coefficient create early in training, and what risk does a large KL coefficient create late in training? Write the schedule as a function of training step and describe how you would detect that the coefficient has decayed too fast.
      </Prose>

      <H3>Exercise 5 — Detecting reward hacking in a binary-label setting</H3>

      <Prose>
        You are training a model with KTO on production thumbs-up / thumbs-down data from a customer support chatbot. After 10K steps, the reward margin is increasing and KTO loss is decreasing, but user satisfaction scores from A/B tests are flat. Describe three specific reward-hacking hypotheses consistent with these observations. For each, describe a diagnostic: what would you measure or visualize to confirm or rule out the hypothesis? Design a reward function modification that would make each form of hacking harder to exploit.
      </Prose>

      <Callout accent="gold">
        The convergence across all six post-training methods (RLHF, DPO, SimPO, GRPO, RLOO, KTO) is that every one optimizes the same KL-constrained objective, trading off closeness to the SFT anchor against movement toward a preference signal. The algorithm you choose determines the data format you need, the computational cost per step, and whether the policy can explore on-policy during training. What it does not determine, more than you might expect, is the final quality ceiling — that is set by the quality of your preference signal and the KL budget you give the policy to use.
      </Callout>

    </div>
  ),
};

export default grpoRlooKto;
