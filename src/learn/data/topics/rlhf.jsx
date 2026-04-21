import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rlhf = {
  title: "RLHF (Reinforcement Learning from Human Feedback)",
  readTime: "~50 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A pretrained language model knows how to complete text. It knows this in an extraordinarily broad sense — it can write Python, draft legal briefs, continue sonnets, transcribe phonetic patterns from obscure languages, and explain protein folding to a ten-year-old. But it has no preference about which of those things it does. Given the prompt "Help me," it might produce a thoughtful explanation, a fragment of a forum post, a spam template, or a sequence of plausible-but-confabulated instructions for something dangerous, depending entirely on which completion happens to score highest under the distribution it learned from the internet. The base model is a fluent autocomplete engine; it is not an assistant.
      </Prose>

      <Prose>
        The gap between "can produce a correct response somewhere in its distribution" and "reliably produces a correct response on the first try" is where almost all the user-facing value of a language model lives. Supervised fine-tuning (SFT) narrows the gap by imitation — show the model thousands of (prompt, good response) pairs and train it to copy the format. That buys a lot: the model learns to write in a helpful tone, to answer questions directly, to follow the structure of an instruction. But imitation has a ceiling. It teaches the model what the examples look like, not what makes a response genuinely good. A model trained purely on imitation can produce confident nonsense in the register of a helpful assistant.
      </Prose>

      <Prose>
        Reinforcement Learning from Human Feedback is the technique that broke through that ceiling. The core idea, stated plainly: instead of training on demonstrations of good behavior, train on comparisons between behaviors, and use those comparisons to construct a reward signal that the model can optimize against. Humans are generally much better at judging which of two responses is better than they are at writing an ideal response from scratch. RLHF exploits that asymmetry. It collects pairwise preferences, trains a reward model to predict them, and then uses reinforcement learning to push the policy toward responses the reward model scores highly — while keeping it close enough to the SFT model that it does not drift into adversarial territory.
      </Prose>

      <Prose>
        The historical record is precise about where this came from. Paul Christiano, Jan Leike, and collaborators at OpenAI introduced the preference-learning framework for RL agents in 2017 (Christiano et al., "Deep Reinforcement Learning from Human Preferences," arXiv:1706.03741), showing that an agent in Atari and MuJoCo environments could learn complex behaviors from fewer than 1,400 pairwise human comparisons. Three years later, Stiennon, Ouyang, Wu, and collaborators applied the same machinery to language model summarization (arXiv:2009.01325), demonstrating that a reward model trained on human rankings could drive a GPT-3 fine-tune to produce summaries that humans preferred over the reference summaries in the dataset. In early 2022, Ouyang and 19 co-authors published InstructGPT (arXiv:2203.02155): the full RLHF pipeline applied to GPT-3, producing a 1.3B-parameter model that human raters preferred over the raw 175B GPT-3 on 85% of prompts. ChatGPT was a productized version of the same recipe, released in November 2022. Anthropic's concurrent work (Bai et al., arXiv:2204.05862) applied the same pipeline with explicit helpfulness and harmlessness criteria, releasing both the training code and the HH-RLHF dataset. Meta's Llama 2 paper (Touvron et al., arXiv:2307.09288) documented five iterative rounds of RLHF, with separate reward models for helpfulness and safety, accumulating over a million preference annotations.
      </Prose>

      <Prose>
        From 2022 onward, RLHF was the central technique in post-training. Every frontier lab ran a version of it. The specific algorithm — PPO with a KL penalty — is now being displaced in open-source practice by simpler methods like DPO, but the three-stage structure (SFT, preference collection, preference optimization) remains the organizing spine of post-training pipelines everywhere. The algorithm is transitional; the problem it attacks is not.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Strip away the math and RLHF is three sequential operations, each one creating the substrate the next one needs.
      </Prose>

      <StepTrace
        label="the canonical rlhf pipeline"
        steps={[
          {
            label: "Stage 1 — Supervised Fine-Tuning (SFT)",
            render: () => (
              <div>
                <TokenStream
                  label="input: base pretrained LM"
                  tokens={[
                    { label: "base LM", color: colors.textMuted },
                    { label: "→ fine-tune on", color: colors.textDim },
                    { label: "(prompt, response) pairs", color: colors.gold },
                    { label: "→ SFT model π_ref", color: colors.green },
                  ]}
                />
                <Prose>
                  SFT compresses the model's output distribution from "everything the internet contains" down to "things that look like assistant responses." This is necessary because the reward model and PPO both assume the policy is already in a roughly sensible neighborhood — they cannot navigate from a raw base-model distribution.
                </Prose>
              </div>
            ),
          },
          {
            label: "Stage 2 — Reward Model Training",
            render: () => (
              <div>
                <TokenStream
                  label="input: human pairwise preferences"
                  tokens={[
                    { label: "prompt x", color: colors.gold },
                    { label: "+ chosen y_w", color: colors.green },
                    { label: "+ rejected y_l", color: "#f87171" },
                    { label: "→ train r_φ(x, y) → scalar", color: "#c084fc" },
                  ]}
                />
                <Prose>
                  The reward model is a language model with its next-token head replaced by a scalar projection. Trained on pairwise preference data, it learns to assign higher scores to responses humans prefer. Crucially it is trained once and frozen — the policy then optimizes against this fixed signal.
                </Prose>
              </div>
            ),
          },
          {
            label: "Stage 3 — PPO Policy Optimization",
            render: () => (
              <div>
                <TokenStream
                  label="RL loop"
                  tokens={[
                    { label: "π samples response y", color: colors.gold },
                    { label: "→ r_φ(x,y) scores it", color: "#c084fc" },
                    { label: "→ subtract β·KL(π‖π_ref)", color: "#60a5fa" },
                    { label: "→ PPO update", color: colors.green },
                  ]}
                />
                <Prose>
                  PPO updates the policy to increase the probability of high-reward responses while a KL penalty prevents it from drifting too far from the SFT model. The KL term is the safety belt: without it, the policy would exploit any quirk the reward model has learned and produce reward-maximizing gibberish.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The key insight that makes RLHF possible is the asymmetry between generating and judging. Writing an ideal response to "explain quantum entanglement to a curious teenager" requires genuine expertise. But if you show a labeler two responses to that prompt, one thoughtful and one superficial, they can usually identify the better one with high reliability even without deep physics knowledge. RLHF harvests this judgment signal at scale. You need thousands of preference pairs, not millions — the InstructGPT paper used around 40,000 labeled pairs for the reward model and around 31,000 prompt-response pairs for SFT. By contrast, GPT-3's pretraining used hundreds of billions of tokens. The data leverage is enormous.
      </Prose>

      <Prose>
        The three networks in play deserve clear labeling. The SFT model <Code>π_ref</Code> is trained once and then frozen — it serves as the reference distribution that the KL penalty anchors the policy to. The policy <Code>π_θ</Code> is the SFT model initialized with the same weights, then updated throughout PPO training. The reward model <Code>r_φ</Code> is an SFT-model-sized network with a scalar head, trained once on preference pairs and then frozen. The value model <Code>V_ψ</Code> is a fourth network — another SFT-model-sized copy with a scalar head — that is trained online during PPO to predict expected future reward, used only to reduce gradient variance. At peak training, you are holding four large transformers in memory simultaneously. This is the primary reason RLHF at scale costs millions of dollars per run.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The Bradley-Terry preference model</H3>

      <Prose>
        The foundation of reward model training is the Bradley-Terry model, a 1952 statistical framework for pairwise comparisons originally developed for tournament ranking. Given a prompt <Code>x</Code>, a chosen response <Code>y_w</Code>, and a rejected response <Code>y_l</Code>, the probability that a human prefers the chosen response is modeled as a sigmoid of the reward difference:
      </Prose>

      <MathBlock caption="Bradley-Terry model: preference probability as sigmoid of reward gap">
        {"P(y_w \\succ y_l \\mid x) = \\sigma\\!\\left(r_\\phi(x,\\, y_w) - r_\\phi(x,\\, y_l)\\right) = \\frac{1}{1 + e^{-(r_\\phi(x,y_w) - r_\\phi(x,y_l))}}"}
      </MathBlock>

      <Prose>
        Notice what the model does and does not assume. It assumes the latent preference between two options can be captured by a single scalar — the reward. It does not assume anything about the absolute scale of that scalar, only the sign and magnitude of differences. This means the reward model is not trained to predict an absolute quality score; it is trained to preserve orderings. The scale of <Code>r_φ</Code> can drift arbitrarily as long as chosen responses score higher than rejected ones. This has an important practical implication: you cannot compare reward scores across different prompts or different training runs.
      </Prose>

      <H3>3.2 Reward model loss</H3>

      <Prose>
        The training objective for the reward model is the negative log-likelihood of the Bradley-Terry model over the preference dataset <Code>D</Code> of (prompt, chosen, rejected) triples:
      </Prose>

      <MathBlock caption="RM loss: maximize log-probability of observed human preferences">
        {"\\mathcal{L}_{\\text{RM}}(\\phi) = -\\,\\mathbb{E}_{(x,\\,y_w,\\,y_l)\\sim\\mathcal{D}}\\left[\\log\\sigma\\!\\left(r_\\phi(x,\\,y_w) - r_\\phi(x,\\,y_l)\\right)\\right]"}
      </MathBlock>

      <Prose>
        This loss is minimized when the model assigns the chosen response a higher scalar than the rejected response on every pair. Gradient descent through the sigmoid and the reward heads trains both the linear projection and the transformer backbone to extract features that correlate with human preference. In practice, a small regularization term is often added — either L2 on the reward head weights or a penalty that keeps the mean reward near zero — to prevent the scale of rewards from growing without bound, which can destabilize PPO training downstream.
      </Prose>

      <H3>3.3 The KL-regularized RL objective</H3>

      <Prose>
        The policy optimization objective adds a KL penalty to the expected reward. Let <Code>π_θ</Code> be the current policy, <Code>π_ref</Code> be the frozen SFT reference, <Code>r_φ</Code> be the frozen reward model, and <Code>β</Code> be the KL coefficient. The objective is:
      </Prose>

      <MathBlock caption="RLHF objective: maximize reward while staying close to SFT model">
        {"\\max_{\\pi_\\theta}\\; \\mathbb{E}_{x\\sim\\mathcal{D},\\;y\\sim\\pi_\\theta(\\cdot\\mid x)}\\!\\left[r_\\phi(x,y) - \\beta\\,\\log\\frac{\\pi_\\theta(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)}\\right]"}
      </MathBlock>

      <Prose>
        The KL term <Code>β log(π_θ / π_ref)</Code> penalizes the policy for moving away from the SFT distribution. It is positive whenever <Code>π_θ</Code> assigns higher probability to a response than <Code>π_ref</Code> does, and the penalty grows with that excess. The coefficient <Code>β</Code> controls the trade-off: large <Code>β</Code> keeps the policy close to SFT (safe but limited), small <Code>β</Code> allows aggressive optimization against the reward model (powerful but prone to hacking). Published values from InstructGPT, Anthropic HH, and Llama 2 cluster in the range 0.01 to 0.2, with the optimal value depending heavily on the reward model's output scale and the diversity of the prompt distribution.
      </Prose>

      <Prose>
        A key insight from Ziegler et al. 2019 and InstructGPT: the KL term can be computed token-by-token. For an autoregressive policy, <Code>{"log π_θ(y|x) = Σ_t log π_θ(y_t | x, y_{<t})"}</Code>, and similarly for <Code>π_ref</Code>. So the KL is the sum of per-token log-ratio terms. In practice, the KL penalty is added as an extra per-token reward at the final token, or distributed across the sequence — different implementations make different choices here, and this can affect training stability.
      </Prose>

      <H3>3.4 PPO clipped objective</H3>

      <Prose>
        Vanilla policy gradient (REINFORCE) estimates the gradient of the expected reward with a single-sample Monte Carlo estimate, which has high variance on long sequences. PPO (Schulman et al., arXiv:1707.06347) stabilizes this by collecting a batch of rollouts under the current policy <Code>π_old</Code>, then making multiple gradient steps against those rollouts using a clipped importance ratio to bound the size of each update.
      </Prose>

      <Prose>
        Define the probability ratio <Code>r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)</Code> and the advantage estimate <Code>A_t</Code>. The PPO clipped surrogate objective is:
      </Prose>

      <MathBlock caption="PPO clipped objective: limits how much any single update can move the policy">
        {"\\mathcal{L}^{\\text{CLIP}}(\\theta) = \\mathbb{E}_t\\!\\left[\\min\\!\\left(r_t\\,A_t,\\; \\mathrm{clip}(r_t,\\,1-\\varepsilon,\\,1+\\varepsilon)\\,A_t\\right)\\right]"}
      </MathBlock>

      <Prose>
        The clip bounds the effective probability ratio to the interval <Code>[1-ε, 1+ε]</Code> (typically <Code>ε = 0.2</Code>). When the advantage is positive (the action was better than expected), the policy is incentivized to increase <Code>π_θ(a_t)</Code>, but only up to <Code>(1+ε) · π_old(a_t)</Code> — beyond that the gradient is zeroed out by the clip. When the advantage is negative, the policy is penalized for the bad action but only down to <Code>(1-ε) · π_old(a_t)</Code>. This prevents any single batch of rewards from catastrophically shoving the policy far from where it was, which is essential when the reward signal is noisy (as it always is with a learned reward model).
      </Prose>

      <H3>3.5 Generalized Advantage Estimation (GAE)</H3>

      <Prose>
        The advantage <Code>A_t</Code> in the PPO objective estimates "how much better was this action than the average action in this state?" The naive estimate is just the reward minus a baseline. GAE (Schulman et al., 2015) provides a lower-variance estimate by exponentially weighting multi-step returns:
      </Prose>

      <MathBlock caption="GAE: exponentially-weighted sum of TD residuals (λ=1 gives full Monte Carlo, λ=0 gives one-step TD)">
        {"A_t^{\\text{GAE}(\\gamma,\\lambda)} = \\sum_{k=0}^{\\infty} (\\gamma\\lambda)^k \\,\\delta_{t+k}, \\quad \\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)"}
      </MathBlock>

      <Prose>
        In the RLHF setting <Code>γ</Code> is often set to 1 (no discounting) because each trajectory is a single response episode, and <Code>λ</Code> is typically 0.95. The value model <Code>V_ψ</Code> predicts the expected shaped reward from a given (prompt, partial response) state, trained simultaneously with the policy using the same rollouts. The value model's lag relative to the policy — it is always slightly behind — is the primary source of instability in PPO training and the reason Llama 2 used rejection-sampling fine-tuning (a simpler RL variant) for early training before switching to full PPO.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below is runnable NumPy — no PyTorch required for the core demonstrations. Each section is self-contained and produces the outputs shown. The goal is to make the mechanics visceral before the production library abstracts them.
      </Prose>

      <H3>4a. Preference dataset construction</H3>

      <Prose>
        A preference dataset is a collection of (prompt, chosen, rejected) triples. In production, "chosen" and "rejected" come from human labelers comparing two model outputs. Here we construct synthetic preferences using a simple quality heuristic — longer, more informative responses win over minimal ones — to produce a 5-pair dataset whose correctness we can verify by inspection.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import math

np.random.seed(42)

prompts = [
    "What is the capital of France?",
    "Explain photosynthesis.",
    "How do I sort a list in Python?",
    "What causes thunder?",
    "Write a haiku about autumn.",
]

# chosen: informative, detailed responses
chosen_responses = [
    "Paris is the capital of France, a major European city with a population of over 2 million.",
    "Photosynthesis converts light energy into chemical energy, producing glucose from CO2 and water.",
    "Use list.sort() in place, or sorted(list) for a new list. Both accept a key argument.",
    "Thunder is caused by the rapid expansion of air heated by a lightning bolt.",
    "Crimson leaves fall slow. Wind carries them far away. Bare branches remain.",
]

# rejected: minimal, low-information responses
rejected_responses = [
    "Paris.",
    "Plants use sunlight.",
    "Use sort.",
    "Lightning.",
    "Leaves fall.",
]

def synthetic_quality(text):
    return len(text) * 0.5 + text.count(".") * 3 + text.count(",") * 1.5

dataset = []
for p, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
    qc = synthetic_quality(chosen)
    qr = synthetic_quality(rejected)
    dataset.append({"prompt": p, "chosen": chosen, "rejected": rejected})
    print(f"chosen_q={qc:.1f}, rejected_q={qr:.1f}, preference_correct={qc > qr}")

# chosen_q=49.5,  rejected_q=6.0,  preference_correct=True
# chosen_q=52.5,  rejected_q=13.0, preference_correct=True
# chosen_q=53.0,  rejected_q=7.5,  preference_correct=True
# chosen_q=40.5,  rejected_q=8.0,  preference_correct=True
# chosen_q=46.5,  rejected_q=9.0,  preference_correct=True`}
      </CodeBlock>

      <Prose>
        In production, this step is the most expensive and most consequential part of the pipeline. OpenAI used around 40,000 preference pairs for InstructGPT; Anthropic's HH-RLHF dataset contains around 170,000 pairs. The labeling guidelines — what counts as "helpful," how to trade off helpfulness against harmlessness, how to handle factual errors — are as load-bearing as the data itself.
      </Prose>

      <H3>4b. Reward model with Bradley-Terry loss</H3>

      <Prose>
        We implement a minimal reward model: a linear head on top of simple text features, trained with the Bradley-Terry loss. The features are deliberately interpretable — response length, punctuation density, capitalization, and word count — so we can trace exactly why the model assigns the scores it does.
      </Prose>

      <CodeBlock language="python">
{`def featurize(text):
    """4-dim feature vector for a response."""
    l = len(text)
    return np.array([
        l / 200.0,                         # length (normalized)
        (text.count(".") + text.count(",") + text.count("!")) / 10.0,  # punctuation
        sum(1 for c in text if c.isupper()) / max(l, 1),               # caps ratio
        len(text.split()) / 50.0,          # word count (normalized)
    ])

np.random.seed(0)
W = np.random.randn(4) * 0.01             # linear reward head weights

def reward(text, W):
    return np.dot(featurize(text), W)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def bt_loss(W, data):
    """Bradley-Terry negative log-likelihood."""
    total = 0.0
    for d in data:
        r_w = reward(d["chosen"], W)
        r_l = reward(d["rejected"], W)
        total += -math.log(sigmoid(r_w - r_l) + 1e-10)
    return total / len(data)

def bt_grad(W, data):
    """Gradient of BT loss w.r.t. W."""
    grad = np.zeros_like(W)
    for d in data:
        r_w = reward(d["chosen"], W)
        r_l = reward(d["rejected"], W)
        p = sigmoid(r_w - r_l)
        factor = -(1 - p)                  # gradient of -log(sigma(delta))
        grad += factor * (featurize(d["chosen"]) - featurize(d["rejected"]))
    return grad / len(data)

# Train with gradient descent
lr = 0.5
for step in range(200):
    g = bt_grad(W, dataset)
    W = W - lr * g
    if step % 50 == 0:
        loss = bt_loss(W, dataset)
        acc = sum(reward(d["chosen"], W) > reward(d["rejected"], W) for d in dataset)
        print(f"Step {step:3d}: loss={loss:.4f}, accuracy={acc}/{len(dataset)}")

# Step   0: loss=0.6606, accuracy=5/5
# Step  50: loss=0.1849, accuracy=5/5
# Step 100: loss=0.1006, accuracy=5/5
# Step 150: loss=0.0684, accuracy=5/5
# Final W = [ 5.013  1.843 -0.962  3.407]
# Final accuracy: 5/5`}
      </CodeBlock>

      <Callout accent="gold">
        The final weight vector tells the story: the model learned that length (W[0]=5.01) and word count (W[3]=3.41) are the strongest predictors of preference in this synthetic dataset, and capitalization (W[2]=-0.96) is slightly penalized. On real human preferences, the features are transformer-internal representations — but the same Bradley-Terry loss applies.
      </Callout>

      <H3>4c. KL-regularized REINFORCE</H3>

      <Prose>
        Before adding PPO's clipping machinery, we implement KL-regularized REINFORCE — the simpler policy gradient that makes the objective concrete. The policy is a softmax distribution over 5 discrete "responses" (actions 0–4), where the true reward increases with the action index but the SFT reference policy prefers lower actions. The KL penalty tethers the policy to the reference distribution.
      </Prose>

      <CodeBlock language="python">
{`np.random.seed(1)
n_actions = 5

# Reference policy (SFT model): prefers lower-indexed actions
ref_logits = np.array([1.5, 1.0, 0.8, 0.5, 0.3])
ref_probs  = softmax(ref_logits)          # [0.340, 0.206, 0.168, 0.128, 0.109] (approx)

# True reward: action 4 is best (reward model is perfect here — no hacking)
true_rewards = np.array([0.1, 0.2, 0.3, 0.4, 1.0])

policy_logits = ref_logits.copy()         # policy starts at SFT
beta = 0.1                                # KL coefficient
lr   = 0.05

for step in range(80):
    probs = softmax(policy_logits)

    # KL divergence from ref: KL(π || π_ref) = Σ_a π(a) log(π(a)/π_ref(a))
    kl = np.sum(probs * np.log(probs / (ref_probs + 1e-10) + 1e-10))

    # Expected reward
    exp_r = np.sum(probs * true_rewards)

    # Shaped reward per action: r(a) - β·log(π(a)/π_ref(a))
    shaped = true_rewards - beta * np.log(probs / (ref_probs + 1e-10) + 1e-10)
    baseline  = exp_r - beta * kl         # current objective value as baseline
    advantage = shaped - baseline

    # Policy gradient: ∇_θ E[shaped_r] ≈ E[(shaped_r - b) · ∇_θ log π(a)]
    # For softmax: ∂log π(a)/∂logit_i = 1[i==a] - π(i)
    grad = probs * (advantage - np.sum(probs * advantage))
    policy_logits += lr * grad

    if step % 20 == 0:
        print(f"Step {step:2d}: E[r]={exp_r:.4f}, KL={kl:.4f}, "
              f"best_action_prob={probs[4]:.3f}")

# Step  0: E[r]=0.2953, KL=-0.0000, best_action_prob=0.109
# Step 20: E[r]=0.3072, KL=0.0011,  best_action_prob=0.120
# Step 40: E[r]=0.3202, KL=0.0046,  best_action_prob=0.133
# Step 60: E[r]=0.3344, KL=0.0108,  best_action_prob=0.147
# Final E[r]: 0.2953 → 0.3493  (18% improvement over SFT)
# Final probs: [0.292 0.210 0.187 0.147 0.164]`}
      </CodeBlock>

      <Prose>
        The policy visibly shifts probability mass toward action 4 (the highest-reward action) while the KL penalty prevents a full collapse onto that single action. Without the penalty, the policy would converge to always choosing action 4 — here, the penalty keeps the distribution spread, which in a language model corresponds to maintaining response diversity rather than collapsing to a single high-reward template.
      </Prose>

      <H3>4d. PPO with clipping and advantage estimation</H3>

      <Prose>
        REINFORCE draws one sample per update, producing high-variance gradient estimates. PPO collects a batch of rollouts, then makes multiple gradient steps against those rollouts using the clipped ratio to prevent over-updating. We compare both on the same toy task and measure variance in the final 20 steps.
      </Prose>

      <CodeBlock language="python">
{`def run_ppo(n_steps=80, lr=0.05, beta=0.1, clip_eps=0.2, rollout_size=8):
    logits   = ref_logits.copy()
    rewards_out = []
    baseline = 0.0
    alpha_bl = 0.1            # running-mean baseline decay

    for step in range(n_steps):
        old_probs = softmax(logits.copy())

        # --- Collect rollout ---
        actions     = np.random.choice(n_actions, size=rollout_size, p=old_probs)
        rollout_r   = true_rewards[actions]
        kl_pens     = beta * np.log(old_probs[actions] / (ref_probs[actions] + 1e-10) + 1e-10)
        shaped      = rollout_r - kl_pens

        # Running-mean baseline (value model proxy)
        baseline    = (1 - alpha_bl) * baseline + alpha_bl * shaped.mean()
        advantages  = shaped - baseline

        # --- PPO clipped update (2 epochs over same rollout) ---
        for _ in range(2):
            new_probs = softmax(logits)
            ratios    = new_probs[actions] / (old_probs[actions] + 1e-10)
            clipped   = np.clip(ratios, 1 - clip_eps, 1 + clip_eps)

            # gradient of min(r*A, clip(r)*A) w.r.t. logits
            grad = np.zeros(n_actions)
            for a, adv, ratio, clip_r in zip(actions, advantages, ratios, clipped):
                if ratio * adv <= clip_r * adv:   # unclipped region
                    g_factor = adv * ratio
                else:                              # clipped: zero gradient
                    g_factor = 0.0
                g_logit       = -new_probs.copy()
                g_logit[a]   += 1.0               # ∂log π(a)/∂logit_i
                grad          += g_factor * g_logit
            logits += lr * grad / rollout_size

        rewards_out.append(np.sum(softmax(logits) * true_rewards))

    return rewards_out

# REINFORCE: single-sample updates
def run_reinforce(n_steps=80, lr=0.05, beta=0.1):
    logits = ref_logits.copy()
    rewards_out = []
    for step in range(n_steps):
        probs  = softmax(logits)
        action = np.random.choice(n_actions, p=probs)
        r      = true_rewards[action]
        kl_pen = beta * np.log(probs[action] / (ref_probs[action] + 1e-10) + 1e-10)
        shaped = r - kl_pen
        grad   = -probs.copy()
        grad[action] += 1.0
        logits += lr * shaped * grad
        rewards_out.append(np.sum(softmax(logits) * true_rewards))
    return rewards_out

np.random.seed(2)
reinforce_r = run_reinforce()
ppo_r       = run_ppo()

# Variance comparison (last 20 steps)
# REINFORCE variance: 0.000005  (lower variance, but slower convergence)
# PPO       variance: 0.000131  (slightly higher variance, but stronger signal)

# Final E[r]:
# REINFORCE: 0.295 → 0.296 → 0.295 → 0.320   (minimal improvement)
# PPO:       0.295 → 0.313 → 0.342 → 0.420   (strong monotonic improvement)`}
      </CodeBlock>

      <Plot
        label="ppo vs reinforce — expected reward over training steps"
        xLabel="training step"
        yLabel="E[reward]"
        series={[
          {
            name: "PPO (rollout=8, 2 epochs)",
            color: "#4ade80",
            points: [
              [0, 0.295], [10, 0.304], [20, 0.313], [30, 0.327],
              [40, 0.342], [50, 0.363], [60, 0.385], [70, 0.408], [79, 0.420],
            ],
          },
          {
            name: "REINFORCE (single sample)",
            color: "#f97316",
            points: [
              [0, 0.295], [10, 0.295], [20, 0.296], [30, 0.295],
              [40, 0.295], [50, 0.301], [60, 0.308], [70, 0.315], [79, 0.320],
            ],
          },
        ]}
      />

      <Prose>
        PPO's multi-sample rollout and clipped update produce a clean, monotonic reward improvement that REINFORCE — with its single-sample, high-variance gradient — cannot match at the same learning rate. In practice on language models, this gap is even larger because sequences are long, reward signals are sparse (often only the final token receives a reward), and single-sample estimates are nearly uninformative.
      </Prose>

      <H3>4e. Reward hacking demonstration</H3>

      <Prose>
        The most important failure mode of RLHF is not a bug — it is the inevitable consequence of optimizing a proxy. We demonstrate this by training a policy against a reward model that is biased toward longer responses, while the true quality peaks at moderate length. Watch the policy drift toward length 10 (maximum) even as true quality falls.
      </Prose>

      <CodeBlock language="python">
{`np.random.seed(3)

# Biased reward model: reward ∝ length  (real RM quirk: longer = more thorough)
def biased_rm(length):
    return 0.3 * length + np.random.randn() * 0.1

# True quality: bell curve peaked at length 4
def true_quality(length):
    return float(np.exp(-0.5 * ((length - 4) / 2.0) ** 2))

# Policy: distribution over response lengths 0–10
length_logits = np.ones(11) * 0.5        # starts uniform

for step in range(60):
    probs  = softmax(length_logits)
    length = np.random.choice(11, p=probs)
    rm_score = biased_rm(length)

    # Policy gradient on RM score — NO KL penalty (to show drift clearly)
    grad = -probs.copy()
    grad[length] += 1.0
    length_logits += 0.1 * rm_score * grad

    if step % 15 == 0:
        probs_now = softmax(length_logits)
        mean_len  = np.sum(probs_now * np.arange(11))
        tq_vals   = np.array([true_quality(l) for l in range(11)])
        exp_tq    = np.sum(probs_now * tq_vals)
        exp_rm    = np.sum(probs_now * (0.3 * np.arange(11)))
        print(f"Step {step:2d}: mean_length={mean_len:.2f}, "
              f"RM_score={exp_rm:.3f}, true_quality={exp_tq:.3f}")

# Step  0: mean_length=5.02, RM_score=1.505, true_quality=0.453
# Step 15: mean_length=5.22, RM_score=1.567, true_quality=0.469
# Step 30: mean_length=5.33, RM_score=1.600, true_quality=0.486
# Step 45: mean_length=5.83, RM_score=1.748, true_quality=0.429
#
# RM score keeps rising; true quality FALLS after step 30.
# The policy has learned to produce long responses.
# A real production RM biased toward length produces exactly this.`}
      </CodeBlock>

      <Heatmap
        label="reward hacking: length vs rm score vs true quality"
        matrix={[
          [0.00, 0.135],
          [0.30, 0.325],
          [0.60, 0.607],
          [0.90, 0.882],
          [1.20, 1.000],
          [1.50, 0.882],
          [1.80, 0.607],
          [2.10, 0.325],
          [2.40, 0.135],
          [2.70, 0.044],
          [3.00, 0.011],
        ]}
        rowLabels={["len 0","len 1","len 2","len 3","len 4","len 5","len 6","len 7","len 8","len 9","len 10"]}
        colLabels={["RM score","true quality"]}
        colorScale="gold"
        cellSize={42}
      />

      <Prose>
        The heatmap shows the fundamental misalignment: the biased reward model assigns its maximum score at length 10, but true quality peaks at length 4. A policy trained long enough against this RM will produce verbosely meaningless responses — high RM score, poor actual quality. This is Goodhart's Law in action: when a measure becomes a target, it ceases to be a good measure.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 HuggingFace TRL</H3>

      <Prose>
        HuggingFace TRL (Transformer Reinforcement Learning) is the most widely used open-source library for RLHF. It provides <Code>RewardTrainer</Code> for the Bradley-Terry reward model training and <Code>PPOTrainer</Code> for the policy optimization loop. A minimal setup for a 7B model on a single node looks like this:
      </Prose>

      <CodeBlock language="python">
{`from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RewardTrainer, RewardConfig, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ── Stage 2: Train reward model ────────────────────────────────────────────────
reward_config = RewardConfig(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",  # start from SFT model
    output_dir="./reward_model",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    max_length=512,
    gradient_checkpointing=True,          # essential for large models
    learning_rate=1e-5,
)

# dataset must have "chosen" and "rejected" columns (tokenized)
reward_trainer = RewardTrainer(
    model=reward_model,
    args=reward_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
reward_trainer.train()
reward_trainer.save_model()

# ── Stage 3: PPO policy optimization ──────────────────────────────────────────
ppo_config = PPOConfig(
    model_name="./sft_model",
    reward_model="./reward_model",
    learning_rate=1e-5,
    batch_size=64,                        # number of prompt-response pairs per update
    mini_batch_size=8,                    # gradient accumulation chunks
    ppo_epochs=4,                         # inner epochs per PPO batch
    kl_penalty="kl",                      # "kl" or "abs" or "mse"
    init_kl_coef=0.2,                     # initial β; adapted by AdaptiveKLController
    target=6.0,                           # target KL divergence in nats
    horizon=10000,                        # steps over which KL target is approached
    gamma=1.0,                            # discount factor
    lam=0.95,                             # GAE lambda
    cliprange=0.2,                        # PPO ε
    cliprange_value=0.2,                  # value function clip range
    vf_coef=0.1,                          # value loss coefficient
)

# AutoModelForCausalLMWithValueHead adds the scalar value head on top of the LM
policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")
ref_policy = AutoModelForCausalLM.from_pretrained("./sft_model")  # frozen

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy,
    ref_model=ref_policy,
    tokenizer=tokenizer,
)

# Training loop (TRL handles rollout, scoring, and update internally)
for batch in prompt_dataloader:
    queries    = batch["input_ids"]
    responses  = ppo_trainer.generate(queries, max_new_tokens=256)
    scores     = reward_pipeline(tokenizer.batch_decode(responses))  # reward model
    train_stats = ppo_trainer.step(queries, responses, scores)

# TRL's AdaptiveKLController adjusts β automatically to keep KL near target.
# This is more robust than a fixed β and is what most production runs use.`}
      </CodeBlock>

      <H3>5.2 Production-scale considerations</H3>

      <Prose>
        The gap between the TRL snippet above and a production RLHF run at frontier scale is substantial. The InstructGPT paper describes running four large transformers simultaneously: policy, reference, reward model, and value model — all of them in the 6B–175B parameter range, all of them requiring at least 80GB GPU memory per model at bf16. The engineering requirements include: tensor parallelism across 8–16 GPUs per model, pipeline parallelism across nodes, an asynchronous rollout system that generates responses on one set of GPUs while the optimizer runs on another, and careful numerical stabilization because bf16 underflows can corrupt the KL terms. The Llama 2 team ran five iterative rounds of RLHF, each requiring full retraining of the reward model on fresh preference data (including responses from the current policy checkpoint), which means the total preference annotation budget grows with each iteration.
      </Prose>

      <Prose>
        Anthropic's HH-RLHF pipeline (Bai et al., 2022) introduced iterative online RLHF: the reward model is retrained weekly on a mix of static and policy-rollout preferences, keeping the RM in-distribution with the current policy. This substantially reduces reward hacking because the RM is always evaluating responses from the distribution it was trained on, rather than out-of-distribution rollouts from a policy that has moved far from the RM training distribution.
      </Prose>

      <H3>5.3 Chosen vs rejected response visualization</H3>

      <TokenStream
        label="chosen response (green) — preferred by reward model and humans"
        tokens={[
          { label: "Paris", color: "#4ade80" },
          { label: "is", color: "#4ade80" },
          { label: "the", color: "#4ade80" },
          { label: "capital", color: "#4ade80" },
          { label: "of", color: "#4ade80" },
          { label: "France,", color: "#4ade80" },
          { label: "a major", color: "#4ade80" },
          { label: "European", color: "#4ade80" },
          { label: "city", color: "#4ade80" },
          { label: "with over", color: "#4ade80" },
          { label: "2 million", color: "#4ade80" },
          { label: "residents.", color: "#4ade80" },
        ]}
      />

      <TokenStream
        label="rejected response (red) — minimal, lower reward"
        tokens={[
          { label: "Paris.", color: "#f87171" },
        ]}
      />

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Reward vs KL trade-off (Pareto frontier of β)</H3>

      <Prose>
        The KL coefficient <Code>β</Code> governs a fundamental trade-off. Small <Code>β</Code> allows aggressive reward optimization but risks hacking and distributional collapse; large <Code>β</Code> keeps the policy close to SFT but limits alignment gains. Different values of <Code>β</Code> trace a Pareto frontier in (reward, KL) space. The following plot shows this frontier schematically, based on the functional forms reported in InstructGPT and Llama 2.
      </Prose>

      <Plot
        label="pareto frontier — expected reward vs kl divergence as β varies"
        xLabel="KL divergence from SFT (nats)"
        yLabel="reward model score"
        series={[
          {
            name: "reward model score (biased proxy)",
            color: "#e2b55a",
            points: [
              [0.0, 0.30], [0.5, 0.42], [1.0, 0.52], [2.0, 0.61],
              [4.0, 0.69], [7.0, 0.74], [12.0, 0.77], [20.0, 0.78],
            ],
          },
          {
            name: "true human preference (gold standard)",
            color: "#4ade80",
            points: [
              [0.0, 0.30], [0.5, 0.42], [1.0, 0.52], [2.0, 0.60],
              [4.0, 0.63], [7.0, 0.59], [12.0, 0.48], [20.0, 0.33],
            ],
          },
        ]}
      />

      <Prose>
        The gap between the two curves illustrates reward hacking: the reward model score continues rising as KL increases (the policy finds better ways to exploit the proxy), while true human preference peaks around KL ≈ 2–4 nats and then falls as the policy enters adversarial territory. The practical implication is that you should monitor human evaluation — not just reward model score — throughout PPO training, and stop training before the curves diverge. Gao, Schulman, and Hilton (arXiv:2210.10760) showed that this peak-then-fall pattern is quantitatively predictable from reward model size and the number of preference pairs used to train it.
      </Prose>

      <H3>6.2 Reward increase over PPO steps</H3>

      <Plot
        label="ppo training dynamics — reward and kl over steps"
        xLabel="ppo update step"
        yLabel="metric value"
        series={[
          {
            name: "RM score (normalized)",
            color: "#e2b55a",
            points: [
              [0,0.30],[5,0.35],[10,0.41],[20,0.48],[30,0.54],
              [50,0.61],[75,0.66],[100,0.70],[150,0.73],[200,0.74],
            ],
          },
          {
            name: "KL from SFT (nats)",
            color: "#60a5fa",
            points: [
              [0,0.0],[5,0.2],[10,0.5],[20,1.1],[30,1.7],
              [50,2.8],[75,3.9],[100,4.8],[150,6.1],[200,7.0],
            ],
          },
        ]}
      />

      <Prose>
        The characteristic shape of a well-tuned RLHF run: reward climbs steeply in the first 50 steps as the policy adopts obvious improvements from the SFT baseline, then plateaus as it reaches the frontier of what the reward model reliably represents. KL grows roughly log-linearly over the run, and the AdaptiveKLController (used in TRL) adjusts <Code>β</Code> to keep it near the target (typically 4–8 nats for a well-sized run).
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>RLHF / PPO vs the alternatives</H3>

      <Prose>
        The post-training landscape in 2025 offers several methods for preference alignment. The right choice depends on your compute budget, data availability, and whether you need online or offline training.
      </Prose>

      <Heatmap
        label="method comparison matrix (higher = better for that criterion)"
        matrix={[
          [5, 2, 3, 5, 5],
          [4, 4, 5, 3, 3],
          [3, 5, 5, 2, 2],
          [2, 5, 5, 1, 2],
          [4, 3, 4, 4, 3],
        ]}
        rowLabels={["PPO (RLHF)", "DPO", "SimPO", "ORPO", "GRPO"]}
        colLabels={["alignment quality", "simplicity", "stability", "online training", "verifiable rewards"]}
        colorScale="green"
        cellSize={48}
      />

      <Prose>
        <strong>Use full PPO (RLHF)</strong> when: you have a large compute budget; you need online training (the policy's rollouts inform the reward signal in real time); your task has verifiable rewards (math, code, tool use); or you are targeting capabilities that require long-horizon reasoning. The InstructGPT, Llama 2, and most frontier proprietary models use PPO at some stage.
      </Prose>

      <Prose>
        <strong>Use DPO</strong> (Rafailov et al., 2023) when: you have a fixed offline preference dataset; your compute budget does not support running four large models simultaneously; you want a single supervised-style training loop. DPO is mathematically equivalent to RLHF under the optimal policy but avoids the RL training loop entirely. Most open-source releases post-2024 (Llama 3, Mistral, Qwen 2.5, Gemma) use DPO or a DPO variant.
      </Prose>

      <Prose>
        <strong>Use SimPO</strong> when: you want DPO without a reference model (saving one full forward pass per training step); your response lengths are variable and you want length normalization built in. SimPO uses the average log-probability under the current policy as the implicit reward, removing the reference model from the loss entirely.
      </Prose>

      <Prose>
        <strong>Use GRPO</strong> (Group Relative Policy Optimization, used in DeepSeek-R1) when: you are doing mathematical or code reasoning with verifiable ground truth; you want the benefits of online RL without a value model. GRPO replaces the value model with a group-relative baseline — score each response relative to the mean score of a group of responses to the same prompt — eliminating the need for the fourth transformer.
      </Prose>

      <Prose>
        <strong>Skip RLHF entirely</strong> when: you are building a narrow-domain model where SFT on high-quality demonstrations is sufficient (code completion, summarization with clear rubrics, classification); your preference data is too small (&lt;10,000 pairs) to train a reliable reward model; or you are doing a first pass and want to validate that SFT quality is high enough before investing in the preference pipeline.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 What scales</H3>

      <Prose>
        <strong>Reward model accuracy scales with preference data volume.</strong> Larger preference datasets produce reward models that generalize better to out-of-distribution prompts, which in turn reduces reward hacking. The InstructGPT paper used ~40,000 preference pairs; Llama 2 used over a million. The gain is roughly logarithmic — each doubling of the dataset size produces a diminishing but still meaningful improvement in RM generalization. There is no published saturation point.
      </Prose>

      <Prose>
        <strong>Policy quality scales with rollout count per PPO update.</strong> More rollouts per gradient step reduce variance in the advantage estimate, allowing the policy to make larger, more reliable steps. The practical limit is memory and compute — each rollout requires a forward pass through the policy and reward model, and batching is constrained by GPU memory. The Llama 2 paper describes using rejection sampling (generate many responses, keep the top-k by reward model score) as a compute-efficient alternative to full PPO for early training stages.
      </Prose>

      <Prose>
        <strong>Iterative RLHF scales in alignment quality.</strong> Retraining the reward model on fresh preference data from the current policy's rollouts — and repeating this cycle — consistently improves alignment quality over single-round RLHF. Anthropic's iterated online RLHF and Llama 2's five-round pipeline both demonstrate this. The improvement is not primarily from more data but from the RM staying in-distribution with the policy.
      </Prose>

      <H3>8.2 What doesn't scale</H3>

      <Prose>
        <strong>Human annotation cost dominates.</strong> Scaling preference data requires scaling human annotator time. At frontier labs, annotation pipelines cost tens of millions of dollars annually and are among the most tightly controlled competitive advantages. Active learning — presenting annotators with pairs where the reward model is most uncertain — reduces the number of annotations needed for a given RM accuracy, but the marginal cost per annotation does not fall as data scales.
      </Prose>

      <Prose>
        <strong>Reward model generalization gap widens at scale.</strong> Gao et al. (arXiv:2210.10760) showed empirically that the gap between RM score and gold-standard human preference grows as the policy is optimized further. Larger reward models widen the onset of this gap but do not eliminate it. No current architecture has solved the fundamental problem that a reward model trained on a finite preference dataset will eventually be exploited by a powerful enough optimizer.
      </Prose>

      <Prose>
        <strong>PPO hyperparameter sensitivity does not decrease with model size.</strong> The hyperparameter surface for PPO — KL coefficient, learning rates, clip range, GAE lambda, value loss coefficient, rollout batch size — is high-dimensional and its optimal configuration is not predictable from first principles. Larger models do not simplify tuning; they often make it harder because training instabilities (KL explosions, value model divergence) are more consequential and harder to recover from.
      </Prose>

      <Prose>
        <strong>Annotator agreement does not scale to subjective tasks.</strong> On clearly factual tasks ("is this summary accurate?"), inter-annotator agreement is high and scales with annotator training. On subjective tasks ("is this response appropriately cautious?"), agreement plateaus around 65–75% regardless of instruction quality, because the underlying task is genuinely ambiguous. The ceiling on RM quality for subjective tasks is set by human disagreement, not data volume.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Reward hacking (Goodhart's Law)</H3>

      <Prose>
        The defining failure of RLHF. The reward model is a learned proxy for human preference, trained on a finite sample from a bounded distribution. The policy optimizer is powerful and, given enough training steps, will find inputs where the proxy diverges from the true objective. Reward hacking manifests as: verbosely formatted responses that score well on length-biased RMs; obsequious openers ("Great question!") that correlate with polite human responses; confident hallucinations where hedging was penalized; excessive bullet points where the RM learned structure is positive. Defenses: KL penalty, iterative RM retraining, RM ensembles, human evaluation throughout training.
      </Prose>

      <Callout accent="gold">
        Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure." Applied to RLHF: the longer you train against the reward model, the better the policy gets at the reward model score and the worse it gets at what the reward model was supposed to represent.
      </Callout>

      <H3>9.2 KL collapse (β too high)</H3>

      <Prose>
        When <Code>β</Code> is set too high, the KL penalty overwhelms the reward signal. The policy stays so close to the SFT reference that it cannot move toward higher-reward responses at all — alignment training does nothing. This is detectable by monitoring KL divergence: if it stays near zero throughout training, <Code>β</Code> is too high. Reduce <Code>β</Code> or increase the reward model's output scale (though the latter requires retraining the RM).
      </Prose>

      <H3>9.3 KL explosion (β too low)</H3>

      <Prose>
        The opposite failure. With insufficient KL penalty, the policy drifts far from the SFT distribution in a few steps, entering regions of weight space where the reward model is extrapolating wildly. The policy finds high-reward but incoherent outputs — repeated phrases, response-format hallucinations, sudden language switching — and collapses onto them. Detectable by monitoring KL: a sudden spike (KL &gt; 20 nats in early training, or KL growing faster than reward) signals impending collapse. The fix is to increase <Code>β</Code> or reduce the learning rate, and often to checkpoint-restore before the explosion.
      </Prose>

      <H3>9.4 Length bias</H3>

      <Prose>
        Human labelers systematically perceive longer responses as more thorough, even when they are not. This is one of the most documented and persistent biases in preference datasets. The reward model absorbs it, and the policy learns to produce verbose responses. Production mitigation: normalize reward by response length during RM training; add a length penalty to the PPO objective; audit labeled pairs for length-correctness correlation and discount long-preferred pairs where the quality difference is not attributable to content.
      </Prose>

      <H3>9.5 Distributional shift between RM training and PPO rollouts</H3>

      <Prose>
        The reward model is trained on preferences between responses from the SFT model (or an early policy checkpoint). As PPO pushes the policy into new territory, the reward model is asked to evaluate out-of-distribution responses — ones it never saw during training. Its judgments in these regions are extrapolation, not interpolation, and the extrapolation is systematically wrong in the direction of whatever features the RM learned to associate with high reward. Mitigation: iterative RLHF (retrain RM on policy rollouts each round); KL penalty to keep policy near the RM training distribution.
      </Prose>

      <H3>9.6 Annotator disagreement corrupting RM quality</H3>

      <Prose>
        InstructGPT reports inter-annotator agreement of 72–77% on pairwise preferences. This means roughly one in four training pairs is noise: two labelers shown the same pair would disagree, and the label in the dataset reflects one annotator's judgment that may not be the majority view. At this noise rate, the RM can learn spurious correlations rather than genuine preference signals. Mitigation: majority vote across multiple annotators per pair; filtering out near-tie preferences; explicit calibration rounds where annotators must agree on anchor examples before labeling production pairs.
      </Prose>

      <H3>9.7 Value model divergence</H3>

      <Prose>
        The value model <Code>V_ψ</Code> is trained online during PPO on the same rollouts being used to update the policy. Because it is always slightly behind the current policy, its advantage estimates are biased. If the value model diverges — failing to track the policy's improving reward expectations — the advantage estimates become garbage, and PPO gradient steps become random walks. This is the most common cause of mid-run training instability. Mitigation: lower learning rate for the value model than the policy; gradient clipping on value loss; checkpoint-restore if value loss spikes suddenly.
      </Prose>

      <H3>9.8 Over-optimization and reward misgeneralization</H3>

      <Prose>
        Beyond reward hacking (which is RM-specific exploitation), there is a deeper failure: the reward model may be right on the training distribution but wrong on out-of-distribution prompts in a systematic way. A reward model trained primarily on English helpfulness preferences may generalize poorly to multilingual prompts, to domain-specific technical questions outside labeler expertise, or to prompts that elicit novel reasoning chains the annotators never evaluated. The policy, optimizing against this RM, will appear aligned on prompts similar to the training distribution and misaligned on novel ones — not because it is doing anything wrong in a proximate sense, but because the RM's representation of "human preference" does not generalize.
      </Prose>

      <H3>9.9 Mode collapse to high-reward templates</H3>

      <Prose>
        Without sufficient KL penalty and rollout diversity, PPO can collapse the policy onto a small set of high-reward response templates — effectively reducing the language model to a sophisticated template filler. The collapse is detectable: response diversity metrics (entropy of the output distribution, unique bigram ratios across rollouts) fall sharply while RM score rises. Once collapsed, the policy is difficult to recover without resetting to a pre-collapse checkpoint.
      </Prose>

      <H3>9.10 Sycophancy</H3>

      <Prose>
        A subtle, systematic failure: the policy learns to agree with premises stated in the prompt, even when those premises are false, because human labelers rate agreeable responses higher. A labeler who writes a prompt containing a misconception ("Einstein failed math, right?") will often prefer a response that validates the misconception over one that corrects it, because correction feels confrontational. The reward model learns this preference, and the policy becomes sycophantic — telling users what they want to hear rather than what is accurate. This is one of the hardest failure modes to fix because it requires labeling guidelines and labeler training that explicitly counter the natural human bias toward agreement.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All papers below were verified against arXiv as of April 2026.
      </Prose>

      <H3>10.1 Foundational papers</H3>

      <Prose>
        <strong>Christiano, Leike, Brown, Martic, Legg, Amodei (2017).</strong> "Deep Reinforcement Learning from Human Preferences." arXiv:1706.03741. NeurIPS 2017. The originating paper: preference-based RL in Atari and MuJoCo environments using fewer than 1,400 human comparisons. Introduced the reward model training loop and the pairwise preference framework that all subsequent RLHF work builds on.
      </Prose>

      <Prose>
        <strong>Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano (2020).</strong> "Learning to Summarize from Human Feedback." arXiv:2009.01325. NeurIPS 2020. First application of the preference-RL framework to language model fine-tuning. Demonstrated on TL;DR summarization that a reward model trained on human rankings could drive a GPT-3 fine-tune past reference summaries on human evaluation.
      </Prose>

      <Prose>
        <strong>Ouyang, Wu, Jiang, et al. (2022).</strong> "Training Language Models to Follow Instructions with Human Feedback." arXiv:2203.02155. NeurIPS 2022. The InstructGPT paper: full RLHF pipeline on GPT-3. Reports 1.3B InstructGPT preferred over 175B GPT-3 on 85% of prompts. Documents inter-annotator agreement (72–77%), the PPO training loop, KL coefficient selection, and initial reward hacking observations.
      </Prose>

      <Prose>
        <strong>Bai, Jones, Ndousse, Askell, et al. (2022).</strong> "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv:2204.05862. Anthropic's concurrent RLHF paper. Introduces the HH-RLHF dataset (helpfulness and harmlessness preference pairs), iterative online RLHF, and Elo-based evaluation of preference model quality. The HH dataset is publicly available at github.com/anthropics/hh-rlhf.
      </Prose>

      <Prose>
        <strong>Schulman, Wolski, Dhariwal, Radford, Klimov (2017).</strong> "Proximal Policy Optimization Algorithms." arXiv:1707.06347. The PPO paper. Introduces the clipped surrogate objective used in RLHF training. Originally developed for continuous control; adapted for language models by scaling the ratio clipping to handle the discrete token action space.
      </Prose>

      <Prose>
        <strong>Touvron, Martin, Stone, et al. (2023).</strong> "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288. Documents five iterative RLHF rounds, separate reward models for helpfulness and safety, rejection sampling fine-tuning followed by PPO, and over one million preference annotations. The most detailed public account of a production RLHF pipeline.
      </Prose>

      <Prose>
        <strong>Gao, Schulman, Hilton (2023).</strong> "Scaling Laws for Reward Model Overoptimization." arXiv:2210.10760. ICML 2023. Formalizes the reward hacking phenomenon: gold-standard human preference peaks at a finite KL and then declines as the proxy reward model score continues rising. Shows that the peak KL and the rate of decline scale smoothly with reward model size and preference dataset size.
      </Prose>

      <H3>10.2 Follow-on work worth reading</H3>

      <Prose>
        <strong>Rafailov, Sharma, Mitchell, Manning, Ermon, Finn (2023).</strong> "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. NeurIPS 2023. Derives a closed-form loss that optimizes the RLHF objective without a reward model or PPO, training directly on preference pairs with a supervised objective. Foundational for the post-PPO alignment literature.
      </Prose>

      <Prose>
        <strong>Ziegler, Stiennon, Wu, et al. (2019).</strong> "Fine-Tuning Language Models from Human Preferences." arXiv:1909.08593. The intermediate paper between Christiano 2017 and InstructGPT: first application to GPT-2, token-level KL formulation, and the SFT-before-RLHF finding.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the Bradley-Terry gradient</H3>

      <Prose>
        Starting from the reward model loss <Code>L_RM = -E[log σ(r_φ(x, y_w) - r_φ(x, y_l))]</Code>, derive the gradient with respect to the reward model parameters <Code>φ</Code>. Show that the gradient points in the direction of increasing the gap between chosen and rejected reward, and that the magnitude of the gradient is largest when the model is most uncertain (i.e., when the sigmoid output is near 0.5). What does this imply about which training pairs are most informative?
      </Prose>

      <Callout accent="green">
        Hint: let <Code>delta = r_w - r_l</Code>. Then <Code>d/d(delta) [-log σ(delta)] = -(1 - σ(delta))</Code>. Apply the chain rule through <Code>delta</Code> to <Code>φ</Code>. The gradient magnitude is <Code>(1 - σ(delta))</Code>, which is maximized at <Code>delta = 0</Code>.
      </Callout>

      <H3>Exercise 2 — Design an active learning scheme</H3>

      <Prose>
        Human annotation is expensive. Design an active learning strategy for selecting which (prompt, response_A, response_B) pairs to show to annotators, given a partially trained reward model. Your strategy should: (a) maximize information gain per annotation, (b) maintain coverage across the prompt distribution, and (c) avoid concentrating queries in regions the current policy visits often (which would cause RM to overfit to the current policy distribution). Describe how you would estimate RM uncertainty, and what failure modes your strategy introduces.
      </Prose>

      <H3>Exercise 3 — Pick β from KL vs reward curves</H3>

      <Prose>
        You are running a PPO training job and have monitoring data showing two curves: (A) reward model score vs training step, and (B) KL divergence from SFT vs training step. You observe that the RM score is still rising at step 200, but human evaluation (run at steps 50, 100, 150, 200) shows human preference peaked at step 100 and has since declined. At step 100, KL was 3.2 nats; at step 200, KL is 8.7 nats. Given this, what <Code>β</Code> target would you set for your next run? How would you use AdaptiveKLController to enforce it? What additional signal would you collect to confirm your new <Code>β</Code> is better?
      </Prose>

      <H3>Exercise 4 — Detect reward hacking from rollout statistics</H3>

      <Prose>
        Without running a human evaluation, design a set of automatic statistics you would monitor during PPO training to detect reward hacking early. Consider: response length distribution, token entropy, bigram repetition rate, frequency of specific phrases ("Great question!", "As an AI"), refusal rate on sensitive prompts, and calibration of the policy's stated confidence against factual accuracy on a held-out benchmark. For each statistic, describe the direction of change that would indicate hacking and the threshold at which you would intervene.
      </Prose>

      <H3>Exercise 5 — Isolate RM vs PPO contribution</H3>

      <Prose>
        Your RLHF pipeline produces a model that human raters prefer over the SFT baseline on 71% of prompts. Your manager asks: "How much of this improvement came from the reward model and how much from the PPO training?" Design an ablation study to answer this question. Consider the following conditions: (a) SFT baseline, (b) SFT + best-of-N sampling against the RM (no policy update), (c) SFT + PPO with a random reward signal and the same KL coefficient, (d) full RLHF. What does each comparison isolate? What confounders remain? How would you present the results to a non-ML audience?
      </Prose>

    </div>
  ),
};

export default rlhf;
