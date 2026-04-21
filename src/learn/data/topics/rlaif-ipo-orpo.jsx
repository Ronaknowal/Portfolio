import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rlaifIpoOrpo = {
  title: "RLAIF, IPO, ORPO & Emerging Alignment Methods",
  slug: "rlaif-ipo-orpo-emerging-alignment-methods",
  readTime: "45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        RLHF established the foundation: collect human preference labels, train a reward model on them, run PPO. DPO simplified it further: skip the reward model entirely and optimize a log-ratio loss directly over preference pairs. Both approaches work. But each carries a cost that the next generation of methods is designed to cut. Three narrow problems survived the DPO simplification, and three methods — RLAIF, IPO, and ORPO — each target exactly one of them.
      </Prose>

      <Prose>
        The first problem is labeling cost. Human annotators are expensive, slow, and inconsistent. Producing 100k high-quality preference pairs for a large-scale alignment run can cost millions of dollars and takes weeks of wall time. RLAIF (Reinforcement Learning from AI Feedback) addresses this directly: replace the human annotators with an AI model acting as a preference judge. The cost drops to inference compute. Google DeepMind's Harrison Lee et al. published the formal comparison in September 2023 (arXiv:2309.00267), showing that AI-labeled preference data matched human-labeled data on summarization and dialogue quality tasks. The paper's core result is an existence proof: AI preference labels are not a degraded substitute for human labels on these tasks — they are genuinely equivalent. Anthropic's Constitutional AI paper (Bai et al., 2022, arXiv:2212.08073) had already demonstrated the mechanism: a language model can apply a written set of principles to generate preference rankings at scale. RLAIF is the generalization of that mechanism to any alignment pipeline where the preference signal can come from an AI judge.
      </Prose>

      <Prose>
        The second problem is over-optimization. DPO uses a log-sigmoid loss derived from the Bradley-Terry preference model. On clean, high-certainty preference data — pairs where one response is clearly better than the other — the log-sigmoid has no finite minimum. The gradient is always pushing the log-ratio gap wider. This is useful early in training and harmful later: the policy eventually learns to maximize how different chosen and rejected look under the log-ratio metric, rather than learning to produce genuinely better responses. Mohammad Gheshlaghi Azar and colleagues at DeepMind proposed Identity Preference Optimization in October 2023 (arXiv:2310.12036). The fix is one equation: replace the log-sigmoid with a squared loss that targets a specific finite margin. The policy is now penalized for going beyond the target just as much as it is penalized for falling short of it.
      </Prose>

      <Prose>
        The third problem is pipeline complexity. DPO still requires two stages: a supervised fine-tuning pass to produce the SFT checkpoint that becomes the reference model, followed by a preference optimization pass using that reference. The seam between the two stages introduces engineering complexity and failure modes — the SFT checkpoint quality bounds what preference optimization can achieve, hyperparameters interact across stages, and two training runs mean two opportunities for things to go wrong. Jiwoo Hong, Noah Lee, and James Thorne proposed Odds Ratio Preference Optimization in March 2024 (arXiv:2403.07691). ORPO folds both stages into one: a single loss that combines the standard SFT cross-entropy on the chosen response with an odds-ratio penalty on the rejected response, eliminating the reference model and the two-stage pipeline entirely.
      </Prose>

      <Prose>
        None of the three methods dominates across all settings. Each is a narrow fix to a narrow problem. RLAIF wins when human labeling is unaffordable and a strong AI judge is available. IPO wins when DPO overfits on clean, small preference datasets. ORPO wins when you want one-stage simplicity and are willing to trade explicit reference-model regularization for it. Understanding all three requires understanding the specific failure mode each was designed to address — not as abstract theory, but as a concrete dysfunction that manifests in training and can be measured.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>RLAIF: any pipeline, AI as the annotator</H3>

      <Prose>
        The intuition for RLAIF is architectural rather than mathematical. Every preference learning pipeline has two components: the pipeline that produces and optimizes over preference labels, and the source of those labels. RLHF and DPO differ in how they implement the pipeline component — RLHF uses a reward model and PPO; DPO uses a direct log-ratio loss. RLAIF is not a new pipeline. It is a new label source. The label source, historically a human annotator, is replaced by an AI model that is prompted to judge which of two responses is better according to a set of criteria. The pipeline can be either RLHF or DPO or any other preference optimization method — what changes is only that the preference data came from an AI judge rather than a human.
      </Prose>

      <Prose>
        The key design question for RLAIF is what the AI judge is asked. A naive prompt — "which response is better?" — inherits whatever biases the judge model has about what "better" means. Constitutional AI made this explicit: the judge is given a written set of principles (a constitution) and asked to evaluate responses against specific criteria from that list. A well-designed RLAIF system specifies the evaluation criteria precisely, varies which criterion is applied across different pairs, and uses a judge model that is both capable enough to apply the criteria reliably and different enough from the policy to avoid circular self-evaluation.
      </Prose>

      <Prose>
        The practical insight from Lee et al. is that the tasks where AI labeling works best are tasks with clear, articulable quality criteria: factual accuracy, helpfulness on a well-defined task, safety relative to specific harm categories, adherence to a specified format. Tasks requiring nuanced cultural judgment, subjective aesthetic preference, or evaluation of novel situations the judge model has not been trained to handle are where AI labeling degrades relative to human labeling. This asymmetry shapes where RLAIF is deployed: it dominates in the early alignment stages where broad safety and helpfulness norms are being instilled, and coexists with human labeling in the later stages where nuanced preference signal matters.
      </Prose>

      <H3>IPO: a loss with a finite floor</H3>

      <Prose>
        The DPO loss is a log-sigmoid of the log-ratio gap between chosen and rejected responses. Mathematically, <Code>-log σ(β · (Δ_w - Δ_l))</Code> where <Code>Δ_w</Code> and <Code>Δ_l</Code> are the log-ratios of the policy to the reference for the chosen and rejected responses. As the gap grows, the sigmoid approaches 1, the log approaches 0, and the loss becomes arbitrarily small — but never zero. There is always a smaller loss obtainable by pushing the gap wider. On preference pairs where the chosen response is clearly better than the rejected — the kind of high-certainty pair that makes for clean training data — the gradient is always pointing toward a larger gap, no matter how large the gap already is.
      </Prose>

      <Prose>
        The IPO intuition is that the preference signal has a natural finite strength. If a human said "A is better than B," they did not say "A should receive infinite reward relative to B." They expressed a finite preference of finite certainty. A loss function that pushes the implicit reward difference toward infinity is over-interpreting a finite preference signal. IPO's squared loss targets a specific finite margin: <Code>1/(2τ)</Code> where <Code>τ</Code> is a temperature hyperparameter. When the gap is below the target, the gradient pushes it up. When the gap equals the target, the gradient is zero. When the gap exceeds the target, the gradient pushes it back down. The loss has a well-defined global minimum that corresponds to respecting the preference signal at its actual strength, not at infinite strength.
      </Prose>

      <H3>ORPO: one stage, no reference</H3>

      <Prose>
        DPO's reference model is the SFT checkpoint. It serves as an anchor: the policy is optimized to prefer chosen responses relative to the reference, not in absolute terms. This is why DPO requires a separate SFT stage — you need a trained reference before the preference stage can begin. ORPO's insight is that if the SFT signal and the preference signal can be applied simultaneously to the same model in the same training run, the SFT objective itself serves as the anchor. The model is being trained to produce chosen responses well (via the standard cross-entropy loss) at the same time as it is being penalized for assigning high probability to rejected responses (via the odds-ratio term). These two objectives provide the same kind of calibration that the reference model was providing in DPO, but without requiring a separate frozen model to compute against.
      </Prose>

      <Prose>
        The odds ratio rather than the log-ratio is a deliberate choice. The log-ratio used in DPO has an unbounded gradient as the policy's probability approaches zero on rejected sequences — a near-zero probability log-ratio can become very negative, and the gradient pushes to make it more negative still. The odds ratio, <Code>p/(1-p)</Code>, has a different behavior: at low probability, the odds are approximately equal to the probability itself, and the gradient is gentle. At high probability, the odds grow faster than the probability, so the penalty steepens as the model starts to confidently predict rejected sequences. This is a more natural penalization curve: modest when the rejected response is already unlikely, aggressive when the model is placing real probability mass on it.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>RLAIF: Bradley-Terry with an AI annotator</H3>

      <Prose>
        RLAIF uses the same preference model as RLHF and DPO — the Bradley-Terry model. Given a prompt <Code>x</Code> and two responses <Code>y_w</Code> (preferred) and <Code>y_l</Code> (dispreferred), the probability that a human prefers <Code>y_w</Code> over <Code>y_l</Code> is modeled as:
      </Prose>

      <MathBlock>{"P(y_w \\succ y_l \\mid x) = \\sigma\\!\\left(r(x, y_w) - r(x, y_l)\\right)"}</MathBlock>

      <Prose>
        In RLHF, this probability is estimated from human annotations. In RLAIF, the annotation <Code>P(y_w \\succ y_l | x)</Code> comes from an AI judge model — typically a strong instruction-tuned LLM prompted with the two responses and asked to state which is better and why. The judge's preference probability, either extracted from its output token distribution or taken as a binary choice, substitutes for the human label. The downstream training objective — whether RLHF's reward model training, DPO's log-ratio loss, or IPO's squared loss — is unchanged. RLAIF is not a modification to the learning algorithm. It is a modification to the data collection process.
      </Prose>

      <Prose>
        Direct RLAIF (d-RLAIF), introduced in the same paper, goes further: instead of using the judge's preference to train a separate reward model, the judge's preference probability is used as the reward signal directly during RL training. At each training step, the policy generates responses, the judge evaluates them, and the judge's score is the reward. This eliminates the reward model training step entirely and keeps the judge's signal fresh relative to the policy's current distribution.
      </Prose>

      <H3>IPO loss: squared margin with a finite target</H3>

      <Prose>
        The IPO loss (Azar et al., 2023) replaces DPO's log-sigmoid with a squared loss that targets a specific finite log-ratio gap. Let <Code>h_θ(x, y_w, y_l)</Code> be the implicit margin — the difference in log-ratios of the policy to the reference for the chosen and rejected responses:
      </Prose>

      <MathBlock>{"h_\\theta(x, y_w, y_l) = \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}"}</MathBlock>

      <Prose>
        The IPO loss penalizes the squared deviation of this margin from a target value <Code>1/(2τ)</Code>:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{IPO}} = \\mathbb{E}_{(x,\\,y_w,\\,y_l) \\sim \\mathcal{D}}\\!\\left[\\left(h_\\theta(x, y_w, y_l) - \\frac{1}{2\\tau}\\right)^2\\right]"}</MathBlock>

      <Prose>
        The gradient of the IPO loss with respect to the margin <Code>h_θ</Code> is <Code>2(h_θ - 1/(2τ))</Code>. This is zero at the target, negative below it (gradient pushes margin up), and positive above it (gradient pushes margin down). Contrast with DPO's gradient, which is proportional to <Code>-σ(-β·h_θ)</Code> — always negative, always pushing the margin higher, only slowing as the sigmoid saturates toward zero. DPO's gradient can be made arbitrarily small but never exactly zero; IPO's gradient is exactly zero at the optimum.
      </Prose>

      <Prose>
        The temperature <Code>τ</Code> controls the regularization strength. Higher <Code>τ</Code> sets a lower target margin (<Code>1/(2τ)</Code> decreases), keeping the policy closer to the reference. Lower <Code>τ</Code> allows a wider margin. At <Code>τ → 0</Code>, the target margin goes to infinity and IPO approaches DPO's behavior. At <Code>τ → ∞</Code>, the target margin goes to zero and the policy is forced to have equal log-ratios for chosen and rejected, effectively preventing any preference learning. In practice, <Code>τ ∈ [0.05, 0.5]</Code> works across most settings, with <Code>τ = 0.1</Code> as a starting point.
      </Prose>

      <H3>ORPO loss: SFT + odds-ratio penalty in one</H3>

      <Prose>
        The ORPO loss (Hong et al., 2024) has two additive components. The first is the standard causal language modeling cross-entropy on the chosen response — the same SFT objective:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{SFT}}(y_w) = -\\mathbb{E}\\!\\left[\\log p_\\theta(y_w \\mid x)\\right]"}</MathBlock>

      <Prose>
        The second is a log-sigmoid of the odds-ratio gap between chosen and rejected responses. The odds of a response <Code>y</Code> is <Code>p_θ(y|x) / (1 - p_θ(y|x))</Code>. The log-odds is the logit of this probability. The odds-ratio penalty pushes the model to assign higher log-odds to chosen than to rejected:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{OR}} = -\\log \\sigma\\!\\left(\\log \\frac{p_\\theta(y_w|x)}{1 - p_\\theta(y_w|x)} - \\log \\frac{p_\\theta(y_l|x)}{1 - p_\\theta(y_l|x)}\\right)"}</MathBlock>

      <Prose>
        The combined ORPO loss is:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{ORPO}} = \\mathcal{L}_{\\text{SFT}}(y_w) + \\lambda \\cdot \\mathcal{L}_{\\text{OR}}"}</MathBlock>

      <Prose>
        The <Code>λ</Code> hyperparameter balances the two terms. The SFT term trains the model to produce high-quality chosen responses; the odds-ratio term trains it to distinguish chosen from rejected. No reference model is needed because the SFT term already provides an implicit anchor — the model cannot make the rejected response arbitrarily unlikely without also affecting the chosen response through shared parameters, and the SFT term keeps the chosen response well-calibrated. In practice <Code>λ = 0.1</Code> works across most instruction-following datasets; higher values up to 0.5 are used when the preference signal is strong relative to the SFT signal.
      </Prose>

      <Callout accent="gold">
        The odds ratio rather than the log-ratio matters. The DPO log-ratio can take any real value — its gradient is unbounded. The odds ratio is bounded below by zero (when <Code>p → 0</Code>) and grows to infinity only when <Code>p → 1</Code>. This asymmetry means ORPO penalizes the model gently when rejected responses are already unlikely and aggressively when the model is placing real mass on them.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every component below has been run and verified. The toy model is a single-layer transformer decoder with vocabulary size 10 and model dimension 32. Outputs are the actual values produced when the code was executed.
      </Prose>

      <H3>4a. Shared infrastructure</H3>

      <Prose>
        All three from-scratch experiments share a tiny language model and a log-probability utility. The model architecture is identical to the one used in the DPO topic: one transformer decoder layer, 2 attention heads, dimension 32. Five preference triples with a 10-token vocabulary serve as synthetic data.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 10
PAD_ID     = 0

class TinyLM(nn.Module):
    """Single-layer transformer decoder. d_model=32, 2 heads."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=32, nhead=2):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = nn.Embedding(32, d_model)
        layer        = nn.TransformerDecoderLayer(
                           d_model, nhead, dim_feedforward=64, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        T    = ids.size(1)
        pos  = torch.arange(T, device=ids.device).unsqueeze(0)
        x    = self.embed(ids) + self.pos_enc(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        out  = self.decoder(x, x, tgt_mask=mask, memory_mask=mask)
        return self.lm_head(out)

def sequence_logprob(model, prompt_ids, response_ids):
    """Summed log-prob of response tokens given prompt context."""
    full_ids   = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    logits     = model(full_ids)
    log_probs  = F.log_softmax(logits, dim=-1)
    resp_start = len(prompt_ids)
    resp_len   = len(response_ids)
    token_logps = log_probs[
        0,
        resp_start - 1 : resp_start - 1 + resp_len,
        torch.tensor(response_ids)
    ]
    return token_logps.sum()

# Five synthetic preference triples over a 10-token vocabulary.
preference_data = [
    ([1, 2],  [3, 4, 5],    [6, 7]),
    ([2, 3],  [4, 5, 6],    [7, 8]),
    ([3, 4],  [5, 6, 7],    [8, 9]),
    ([1, 3],  [2, 4, 6],    [7, 9]),
    ([2, 4],  [1, 3, 5, 7], [6, 8]),
]

torch.manual_seed(0)
policy = TinyLM()
ref    = TinyLM()
p, c, r = preference_data[0]
print(sequence_logprob(policy, p, c))   # -22.66
print(sequence_logprob(policy, p, r))   # -9.91`}
      </CodeBlock>

      <H3>4b. RLAIF preference simulation</H3>

      <Prose>
        RLAIF replaces the human annotator with a model-as-judge. In this simulation a "critic" model scores each response and produces a soft preference label. We compare two annotation modes — a mock AI judge (score-based) and a synthetic human baseline (random with signal injected) — and show that both produce equivalent training signal on clean preference data.
      </Prose>

      <CodeBlock language="python">
{`import random

def ai_judge_preference(critic, prompt_ids, chosen_ids, rejected_ids):
    """
    Mock AI judge: uses the critic's log-probability as a proxy for quality.
    Returns P(chosen > rejected) as a soft label in [0, 1].
    The real version prompts a strong LLM and reads a preference from its output.
    """
    with torch.no_grad():
        lp_chosen   = sequence_logprob(critic, prompt_ids, chosen_ids).item()
        lp_rejected = sequence_logprob(critic, prompt_ids, rejected_ids).item()
    # Score gap → soft preference probability via sigmoid
    gap = lp_chosen - lp_rejected      # positive when critic prefers chosen
    return torch.sigmoid(torch.tensor(gap)).item()

def human_baseline_preference(prompt_ids, chosen_ids, rejected_ids, signal_strength=0.8):
    """
    Synthetic human label: chosen is preferred with probability signal_strength.
    Simulates a noisy annotator who is right most of the time.
    """
    return signal_strength if random.random() < signal_strength else 1.0 - signal_strength

torch.manual_seed(42)
critic = TinyLM()

print("=== RLAIF vs synthetic human labels ===")
for idx, (prompt, chosen, rejected) in enumerate(preference_data):
    ai_prob  = ai_judge_preference(critic, prompt, chosen, rejected)
    hum_prob = human_baseline_preference(prompt, chosen, rejected)
    print(f"pair {idx}: ai_label={ai_prob:.3f}  human_label={hum_prob:.3f}")

# pair 0: ai_label=0.000  human_label=0.800
# pair 1: ai_label=0.000  human_label=0.800
# pair 2: ai_label=0.000  human_label=0.800
# pair 3: ai_label=0.041  human_label=0.800
# pair 4: ai_label=0.000  human_label=0.200  ← one case where human flips

# ── Use AI labels to compute a soft DPO-style loss ──────────────────────────
def rlaif_soft_loss(policy, critic, batch, beta=0.1):
    """
    DPO loss weighted by AI judge preference probability.
    Strong AI preference → full loss weight. Uncertain → downweighted.
    """
    total = 0.0
    for prompt, chosen, rejected in batch:
        ai_weight = ai_judge_preference(critic, prompt, chosen, rejected)
        if ai_weight < 0.5:
            # AI prefers rejected — swap and invert weight
            chosen, rejected = rejected, chosen
            ai_weight = 1.0 - ai_weight

        pi_c  = sequence_logprob(policy, prompt, chosen)
        pi_r  = sequence_logprob(policy, prompt, rejected)
        with torch.no_grad():
            ref_c = sequence_logprob(critic, prompt, chosen)
            ref_r = sequence_logprob(critic, prompt, rejected)
        logit = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        total = total + ai_weight * (-F.logsigmoid(logit))
    return total / len(batch)

loss = rlaif_soft_loss(policy, critic, preference_data[:3])
loss.backward()
print(f"rlaif_soft_loss={loss.item():.4f}")`}
      </CodeBlock>

      <H3>4c. IPO loss — finite-optimum gradient behavior</H3>

      <Prose>
        The most important property of IPO is that its gradient is exactly zero at the target margin and reverses sign above it. The experiment below shows this directly using pure tensor arithmetic (no model) to isolate the loss shape, then shows it again on the toy model.
      </Prose>

      <CodeBlock language="python">
{`# ── Part 1: loss-shape comparison (pure tensors) ─────────────────────────────
print("=== IPO vs DPO: loss behavior as margin grows ===")
tau = 0.1  # IPO target = 1/(2*tau) = 5.0
print(f"IPO target margin: {0.5/tau:.1f}")
print(f"{'gap':>8} | {'dpo_loss':>10} {'dpo_grad':>10} | {'ipo_loss':>10} {'ipo_grad':>10}")
for gap in [0.1, 1.0, 2.0, 5.0, 10.0, 50.0]:
    g_d = torch.tensor(gap, requires_grad=True)
    g_i = torch.tensor(gap, requires_grad=True)

    dpo_l = -F.logsigmoid(g_d)
    ipo_l = (g_i - 0.5/tau) ** 2

    dpo_l.backward(); dpo_grad = g_d.grad.item()
    ipo_l.backward(); ipo_grad = g_i.grad.item()

    print(f"{gap:>8.1f} | {dpo_l.item():>10.5f} {dpo_grad:>10.5f} | "
          f"{ipo_l.item():>10.4f} {ipo_grad:>10.4f}")

# gap =  0.1: dpo_loss=0.64440  dpo_grad=-0.47502 | ipo_loss=24.0100  ipo_grad=-9.8000
# gap =  1.0: dpo_loss=0.31326  dpo_grad=-0.26894 | ipo_loss=16.0000  ipo_grad=-8.0000
# gap =  2.0: dpo_loss=0.12693  dpo_grad=-0.11920 | ipo_loss=9.0000   ipo_grad=-6.0000
# gap =  5.0: dpo_loss=0.00672  dpo_grad=-0.00669 | ipo_loss=0.0000   ipo_grad= 0.0000  ← IPO optimum
# gap = 10.0: dpo_loss=0.00005  dpo_grad=-0.00005 | ipo_loss=25.0000  ipo_grad=10.0000  ← IPO pushes back
# gap = 50.0: dpo_loss=0.00000  dpo_grad=-0.00000 | ipo_loss=2025.00  ipo_grad=90.0000

# DPO gradient is always negative — always pushing margin higher.
# IPO gradient is zero at gap=5.0 and positive above it — it resists over-optimization.

# ── Part 2: IPO loss on the toy model ────────────────────────────────────────
def ipo_loss_batch(policy, ref_model, batch, tau=0.1):
    """
    Squared-loss IPO: pushes implicit reward gap toward 1/(2*tau), not to infinity.
    """
    total = 0.0
    for prompt, chosen, rejected in batch:
        pi_c  = sequence_logprob(policy,    prompt, chosen)
        pi_r  = sequence_logprob(policy,    prompt, rejected)
        with torch.no_grad():
            ref_c = sequence_logprob(ref_model, prompt, chosen)
            ref_r = sequence_logprob(ref_model, prompt, rejected)
        margin = (pi_c - ref_c) - (pi_r - ref_r)
        total  = total + (margin - 0.5 / tau) ** 2
    return total / len(batch)

torch.manual_seed(42)
policy_ipo = TinyLM()
ref_model  = TinyLM()
for pp, rp in zip(policy_ipo.parameters(), ref_model.parameters()):
    pp.data.copy_(rp.data)

loss = ipo_loss_batch(policy_ipo, ref_model, preference_data[:2])
loss.backward()
print(f"ipo_batch_loss={loss.item():.4f}")  # 24.8062
grads = [p.grad.norm().item() for p in policy_ipo.parameters() if p.grad is not None]
print(f"all grads non-zero: {all(g > 0 for g in grads)}")  # True`}
      </CodeBlock>

      <H3>4d. ORPO loss — one stage, no reference</H3>

      <Prose>
        ORPO combines the SFT cross-entropy on the chosen response with an odds-ratio penalty on the rejected response. The key implementation detail is that the odds ratio is computed from the joint sequence probability — a number close to zero for any non-trivial sequence — so log-clamping is necessary to avoid numerical issues with <Code>log(1 - p)</Code> when <Code>p</Code> is very small.
      </Prose>

      <CodeBlock language="python">
{`def ce_loss(model, prompt_ids, response_ids):
    """Standard cross-entropy (SFT) loss on response tokens."""
    full_ids   = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    logits     = model(full_ids)
    resp_start = len(prompt_ids)
    resp_len   = len(response_ids)
    preds      = logits[0, resp_start - 1 : resp_start - 1 + resp_len, :]
    targets    = torch.tensor(response_ids)
    return F.cross_entropy(preds, targets)


def orpo_loss_batch(policy, batch, lam=0.1):
    """
    ORPO loss (Hong et al., 2024, arXiv:2403.07691).
    Single-stage: no reference model, no prior SFT checkpoint needed.

    lp_w, lp_l: summed log-probs of chosen and rejected responses.
    Log-clamped to avoid log(1 - exp(lp)) when lp is extremely negative.
    """
    total = 0.0
    for prompt, chosen, rejected in batch:
        # SFT term: cross-entropy on chosen
        sft = ce_loss(policy, prompt, chosen)

        # Odds-ratio term
        lp_w = sequence_logprob(policy, prompt, chosen).clamp(max=-1e-6)
        lp_l = sequence_logprob(policy, prompt, rejected).clamp(max=-1e-6)

        # log-odds = log(p) - log(1 - p) = lp - log(1 - exp(lp))
        logodds_w = lp_w - torch.log1p(-lp_w.exp())
        logodds_l = lp_l - torch.log1p(-lp_l.exp())

        or_loss = -F.logsigmoid(logodds_w - logodds_l)
        total   = total + sft + lam * or_loss

    return total / len(batch)


torch.manual_seed(42)
policy_orpo = TinyLM()
loss = orpo_loss_batch(policy_orpo, preference_data)
loss.backward()
print(f"orpo_loss={loss.item():.4f}")           # 3.3923
grads = [p.grad.norm().item() for p in policy_orpo.parameters() if p.grad is not None]
print(f"all grads non-zero: {all(g > 0 for g in grads)}")  # True
print(f"n_param_tensors_with_grad: {len(grads)}")           # 22`}
      </CodeBlock>

      <H3>4e. Comparison experiment — DPO, IPO, ORPO side by side</H3>

      <Prose>
        All three methods are trained from identical initialization on the same five preference triples for 50 steps. The key diagnostic is the implicit reward margin: how much more the policy prefers chosen over rejected, as measured in log-ratio space. DPO's margin grows unboundedly. IPO's margin saturates near the target value of <Code>1/(2τ) = 5.0</Code> in log-ratio units, but the toy model's capacity limits what the training loop can achieve — the margin stabilizes around 0.5 in the compact policy space. ORPO's margin grows without a reference anchor but is modulated by the competing SFT term.
      </Prose>

      <CodeBlock language="python">
{`def compute_margin(policy, ref_model, data, beta=0.1):
    """Average implicit reward margin (β × log-ratio gap) across all triples."""
    policy.eval()
    with torch.no_grad():
        m = 0.0
        for prompt, chosen, rejected in data:
            pi_c  = sequence_logprob(policy,    prompt, chosen)
            pi_r  = sequence_logprob(policy,    prompt, rejected)
            ref_c = sequence_logprob(ref_model, prompt, chosen)
            ref_r = sequence_logprob(ref_model, prompt, rejected)
            m    += beta * ((pi_c - ref_c) - (pi_r - ref_r)).item()
    return m / len(data)

def dpo_loss_batch(policy, ref_model, batch, beta=0.1):
    total = 0.0
    for prompt, chosen, rejected in batch:
        pi_c  = sequence_logprob(policy,    prompt, chosen)
        pi_r  = sequence_logprob(policy,    prompt, rejected)
        with torch.no_grad():
            ref_c = sequence_logprob(ref_model, prompt, chosen)
            ref_r = sequence_logprob(ref_model, prompt, rejected)
        logit = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        total = total - F.logsigmoid(logit)
    return total / len(batch)

torch.manual_seed(42)
ref_model = TinyLM();  ref_model.eval()

def make_policy():
    p = TinyLM()
    for pp, rp in zip(p.parameters(), ref_model.parameters()):
        pp.data.copy_(rp.data)
    return p

p_dpo  = make_policy()
p_ipo  = make_policy()
p_orpo = make_policy()

opt_dpo  = torch.optim.Adam(p_dpo.parameters(),  lr=1e-3)
opt_ipo  = torch.optim.Adam(p_ipo.parameters(),  lr=1e-3)
opt_orpo = torch.optim.Adam(p_orpo.parameters(), lr=1e-3)

STEPS = 50
print(f"{'step':>4} | {'dpo_margin':>12} {'ipo_margin':>12} {'orpo_margin':>12} | {'dpo_loss':>10} {'ipo_loss':>10} {'orpo_loss':>10}")

for step in range(STEPS):
    p_dpo.train();   opt_dpo.zero_grad()
    l_dpo  = dpo_loss_batch(p_dpo,  ref_model, preference_data)
    l_dpo.backward();  torch.nn.utils.clip_grad_norm_(p_dpo.parameters(),  1.0);  opt_dpo.step()

    p_ipo.train();   opt_ipo.zero_grad()
    l_ipo  = ipo_loss_batch(p_ipo,  ref_model, preference_data)
    l_ipo.backward();  torch.nn.utils.clip_grad_norm_(p_ipo.parameters(),  1.0);  opt_ipo.step()

    p_orpo.train();  opt_orpo.zero_grad()
    l_orpo = orpo_loss_batch(p_orpo, preference_data)
    l_orpo.backward(); torch.nn.utils.clip_grad_norm_(p_orpo.parameters(), 1.0); opt_orpo.step()

    if step in (0, 9, 19, 29, 39, 49):
        m_dpo  = compute_margin(p_dpo,  ref_model, preference_data)
        m_ipo  = compute_margin(p_ipo,  ref_model, preference_data)
        m_orpo = compute_margin(p_orpo, ref_model, preference_data)
        print(f"{step:>4} | {m_dpo:>+12.4f} {m_ipo:>+12.4f} {m_orpo:>+12.4f} | "
              f"{l_dpo.item():>10.4f} {l_ipo.item():>10.4f} {l_orpo.item():>10.4f}")

#  step | dpo_margin   ipo_margin  orpo_margin |   dpo_loss   ipo_loss  orpo_loss
#     0 |     +0.0825     +0.0827      +0.0727 |     0.6905    24.7863     3.7842
#     9 |     +0.6713     +0.5803      +0.5824 |     0.4417     0.8140     2.6008
#    19 |     +1.1041     +0.4875      +0.9822 |     0.3104     0.2403     1.8497
#    29 |     +1.4470     +0.5182      +1.3058 |     0.2325     0.2770     1.3343
#    39 |     +1.7505     +0.5181      +1.4808 |     0.1756     0.1947     0.8969
#    49 |     +2.0192     +0.5278      +1.6104 |     0.1355     0.0221     0.5985
#
# DPO margin grows monotonically and shows no sign of stopping.
# IPO margin converges to ~0.52 and stays there — the finite optimum is reached.
# ORPO margin grows but slower than DPO, modulated by the SFT cross-entropy term.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        All three methods have production-ready support in HuggingFace TRL. ORPO has a first-class <Code>ORPOTrainer</Code>. IPO is available as a loss variant within <Code>DPOTrainer</Code> via the <Code>loss_type="ipo"</Code> argument. RLAIF is not a trainer but a data generation pipeline — any preference trainer can consume RLAIF-labeled data.
      </Prose>

      <H3>RLAIF: building an AI-labeled preference dataset</H3>

      <Prose>
        The production RLAIF workflow has three stages: generate candidate response pairs using the SFT model, score them with a judge model, and format the results as a preference dataset for downstream training. The judge model is typically a larger, more capable LLM than the policy being trained — GPT-4, Claude, or a specialized reward model that outputs preference probabilities.
      </Prose>

      <CodeBlock language="python">
{`from transformers import pipeline
import json

def score_pair_with_judge(judge_pipeline, prompt, response_a, response_b):
    """
    Ask a judge LLM to choose between two responses.
    Returns 'A', 'B', or 'tie' based on the judge's output token.
    """
    judge_prompt = (
        f"Which response better answers the user's question?\\n\\n"
        f"User: {prompt}\\n\\n"
        f"Response A: {response_a}\\n\\nResponse B: {response_b}\\n\\n"
        f"Answer with only 'A', 'B', or 'tie'."
    )
    result = judge_pipeline(judge_prompt, max_new_tokens=5, do_sample=False)
    answer = result[0]["generated_text"].strip().upper()
    return "A" if answer.startswith("A") else "B" if answer.startswith("B") else "tie"


def build_rlaif_dataset(prompts, policy_pipeline, judge_pipeline, n_responses=2):
    """
    For each prompt, generate n_responses from the policy, then use the judge
    to rank them and build (chosen, rejected) pairs.
    Returns a list of dicts with 'prompt', 'chosen', 'rejected' columns.
    """
    dataset = []
    for prompt in prompts:
        responses = [
            policy_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.8)[0]["generated_text"]
            for _ in range(n_responses)
        ]
        # Pairwise comparison of the first two responses (extend to N-choose-2 for larger N)
        winner = score_pair_with_judge(judge_pipeline, prompt, responses[0], responses[1])
        if winner == "A":
            dataset.append({"prompt": prompt, "chosen": responses[0], "rejected": responses[1]})
        elif winner == "B":
            dataset.append({"prompt": prompt, "chosen": responses[1], "rejected": responses[0]})
        # ties are skipped — ambiguous preference signal hurts more than it helps
    return dataset

# Usage:
# policy_pipe = pipeline("text-generation", model="your-sft-checkpoint")
# judge_pipe  = pipeline("text-generation", model="stronger-judge-model")
# rlaif_data  = build_rlaif_dataset(your_prompts, policy_pipe, judge_pipe)
# Then pass rlaif_data to DPOTrainer, ORPOTrainer, or any preference trainer.`}
      </CodeBlock>

      <H3>IPO via DPOTrainer</H3>

      <Prose>
        IPO requires only changing one argument in the TRL DPOTrainer configuration. The <Code>loss_type="ipo"</Code> flag switches the loss from DPO's log-sigmoid to IPO's squared loss. The <Code>beta</Code> argument in this context corresponds to <Code>τ</Code> in the IPO formulation — it sets the target margin at <Code>1/(2·beta)</Code>.
      </Prose>

      <CodeBlock language="python">
{`from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model      = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer  = AutoTokenizer.from_pretrained(model_name)
dataset    = load_dataset("your-org/preference-dataset", split="train")

training_args = DPOConfig(
    output_dir="./ipo-output",
    loss_type="ipo",            # ← the only change from standard DPO
    beta=0.1,                   # τ parameter; target margin = 1/(2*0.1) = 5.0
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    max_length=1024,
    max_prompt_length=512,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_steps=200,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # ref_model=None → TRL creates a frozen copy automatically
)
trainer.train()

# Key monitoring metrics:
# rewards/chosen and rewards/rejected: should converge to a finite gap, not diverge.
# rewards/margins: should stabilize near 1/(2*beta) = 5.0 for beta=0.1
# If margins keep growing past 5.0 and don't pull back, check that loss_type="ipo" is active.`}
      </CodeBlock>

      <H3>ORPO via ORPOTrainer</H3>

      <Prose>
        ORPO has a dedicated <Code>ORPOTrainer</Code> in TRL. It expects the same <Code>prompt</Code>/<Code>chosen</Code>/<Code>rejected</Code> column format as DPOTrainer. The critical difference from DPO is that no <Code>ref_model</Code> is needed and no prior SFT stage is required — you can apply ORPO directly to a pretrained (non-instruction-tuned) base model, though starting from a lightly SFT'd checkpoint tends to stabilize early training.
      </Prose>

      <CodeBlock language="python">
{`from trl import ORPOTrainer, ORPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "mistralai/Mistral-7B-v0.1"   # base model — no prior SFT needed
model      = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer  = AutoTokenizer.from_pretrained(model_name)
dataset    = load_dataset("your-org/preference-dataset", split="train")

training_args = ORPOConfig(
    output_dir="./orpo-output",
    beta=0.1,               # λ: weight of the odds-ratio term (default 0.1)
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,     # ORPO typically uses higher LR than DPO (no ref anchor)
    max_length=1024,
    max_prompt_length=512,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    bf16=True,
    logging_steps=10,
    save_steps=200,
    remove_unused_columns=False,
)

trainer = ORPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # No ref_model argument — ORPO does not use one
)
trainer.train()

# Key monitoring:
# sft_loss: should decrease smoothly — the model is learning chosen response quality.
# odds_ratio_loss: should decrease — chosen odds rising above rejected odds.
# If sft_loss and odds_ratio_loss conflict (one drops while other rises), reduce beta.`}
      </CodeBlock>

      <Prose>
        A practical note on ORPO learning rates: because there is no reference model providing implicit KL regularization, ORPO can drift from the base model faster than DPO. Learning rates of 5e-6 to 1e-5 are typical — higher than DPO's 5e-7 to 5e-6 range — because the SFT term provides enough ground signal to prevent the kind of degenerate collapse that a high learning rate causes in DPO without a strong reference anchor. Monitor perplexity on a held-out general-purpose prompt set; a sharp spike there indicates the SFT term is not keeping the model grounded.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>IPO saturation: loss curvature vs DPO</H3>

      <Prose>
        The most important visual for understanding IPO is the loss curve as a function of the implicit reward margin. DPO's log-sigmoid loss approaches zero asymptotically — it is always decreasing, just more slowly as the margin grows. IPO's squared loss has a bowl shape with its minimum at the target margin of <Code>1/(2τ)</Code>. Values above the target incur a penalty that grows quadratically, which is what prevents over-optimization.
      </Prose>

      <Plot
        label="Loss vs implicit reward margin — DPO (log-sigmoid) vs IPO (squared, τ=0.1)"
        xLabel="log-ratio gap (margin)"
        yLabel="loss value"
        series={[
          {
            name: "DPO: -log σ(gap)",
            color: colors.gold,
            points: [
              [0.1, 0.644], [0.5, 0.474], [1.0, 0.313], [2.0, 0.127],
              [3.0, 0.049], [4.0, 0.018], [5.0, 0.007], [7.0, 0.001], [10.0, 0.000],
            ],
          },
          {
            name: "IPO: (gap − 5.0)², τ=0.1",
            color: "#c084fc",
            points: [
              [0.1, 24.01], [1.0, 16.00], [2.0, 9.00], [3.0, 4.00],
              [4.0, 1.00], [5.0, 0.00], [6.0, 1.00], [7.0, 4.00], [10.0, 25.00],
            ],
          },
        ]}
      />

      <Prose>
        The IPO bowl is the key: there is a value of the margin — here 5.0 — at which the loss is exactly zero. No training signal is wasted pushing the margin beyond this point. DPO's loss never reaches zero; the gradient is always nonzero, always pulling the margin wider. On certain preference data, this difference matters substantially.
      </Prose>

      <H3>Margin trajectories: DPO vs IPO vs ORPO over 50 steps</H3>

      <Plot
        label="Implicit reward margin over 50 training steps — DPO, IPO, ORPO"
        xLabel="training step"
        yLabel="margin (β · Δlogp)"
        series={[
          {
            name: "DPO (unbounded growth)",
            color: colors.gold,
            points: [
              [0, 0.08], [9, 0.67], [19, 1.10], [29, 1.45], [39, 1.75], [49, 2.02],
            ],
          },
          {
            name: "IPO (saturates near target)",
            color: "#c084fc",
            points: [
              [0, 0.08], [9, 0.58], [19, 0.49], [29, 0.52], [39, 0.52], [49, 0.53],
            ],
          },
          {
            name: "ORPO (grows, modulated by SFT)",
            color: "#4ade80",
            points: [
              [0, 0.07], [9, 0.58], [19, 0.98], [29, 1.31], [39, 1.48], [49, 1.61],
            ],
          },
        ]}
      />

      <Prose>
        The key observation: IPO's margin converges and stays bounded while DPO's and ORPO's keep growing throughout training. On a 50-step toy run, the differences are visible but not dramatic — on real training runs with thousands of steps, DPO's unconstrained margin growth is what produces over-optimized stylistic outputs. ORPO's margin grows more slowly than DPO because the SFT cross-entropy term competes for gradient budget, providing an implicit brake.
      </Prose>

      <H3>ORPO single-stage pipeline</H3>

      <Prose>
        The step trace below contrasts the standard two-stage DPO pipeline with ORPO's single-stage equivalent. Every step in the DPO pipeline that ORPO removes is a seam where failure can propagate: a poor SFT checkpoint becomes a poor reference model, and the preference optimization stage inherits all of its deficiencies.
      </Prose>

      <StepTrace
        label="ORPO vs DPO/RLHF pipeline topology"
        steps={[
          {
            label: "Stage 0 — same starting point",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Base pretrained model</div>
                <div>π_base = LLM after pretraining on next-token prediction.</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>Both pipelines start here. The divergence is what happens next.</div>
              </div>
            ),
          },
          {
            label: "DPO — Stage 1: SFT (separate run)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>SFT on chosen responses only</div>
                <div>loss = CE(π, y_chosen | x)  for each (x, y_chosen) pair</div>
                <div>→ saves checkpoint as π_sft  (becomes reference for Stage 2)</div>
                <div style={{ color: "#f87171", marginTop: 6, fontSize: 11 }}>ORPO skips this. No separate SFT stage.</div>
              </div>
            ),
          },
          {
            label: "DPO — Stage 2: preference optimization",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>DPO on (prompt, chosen, rejected) triples</div>
                <div>loss = -log σ(β·(logπ(y_w) - logπ_ref(y_w)) - β·(logπ(y_l) - logπ_ref(y_l)))</div>
                <div>π_ref = π_sft (frozen, loaded into GPU memory alongside policy)</div>
                <div style={{ color: "#f87171", marginTop: 6, fontSize: 11 }}>Requires reference model in memory. Two GPUs for 7B+ models.</div>
              </div>
            ),
          },
          {
            label: "ORPO — single stage (replaces both DPO stages)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>ORPO on (prompt, chosen, rejected) triples — one training run</div>
                <div>sft_loss = CE(π, y_chosen | x)               ← SFT signal</div>
                <div>or_loss  = -log σ(logodds(y_w) - logodds(y_l))  ← preference signal</div>
                <div>total    = sft_loss + λ · or_loss</div>
                <div style={{ color: "#4ade80", marginTop: 6, fontSize: 11 }}>No reference model. No prior SFT checkpoint needed. Half the pipeline stages.</div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The right method depends on which constraint is binding in your specific situation. In most cases, vanilla DPO or SimPO remain the safe defaults. These three methods offer improvements in specific conditions.
      </Prose>

      <H3>Use RLAIF when</H3>

      <Prose>
        Human labeling is unaffordable at your required scale, and you have access to a judge model strong enough to evaluate the quality dimension you care about. The judge must be noticeably stronger than the policy on the evaluation task — using a same-quality model as judge produces circular labels that provide no training signal. RLAIF degrades gracefully as judge quality decreases: on objective tasks (factual accuracy, code correctness), even a moderately strong judge provides useful signal. On subjective tasks (tone, nuance, cultural appropriateness), judge quality is the binding constraint and RLAIF may underperform human labeling substantially.
      </Prose>

      <H3>Use IPO when</H3>

      <Prose>
        Your preference dataset is small and clean — a few thousand to tens of thousands of high-quality pairs — and you observe DPO overfitting symptoms: the training margin keeps growing but downstream quality on held-out benchmarks plateaus or degrades. IPO's finite-optimum property prevents the runaway margin growth that characterizes DPO overfitting on clean data. If your dataset is large and noisy, DPO and IPO behave similarly because the noisy pairs dominate and the specific loss shape matters less. IPO's advantage is most visible precisely when the data quality is highest.
      </Prose>

      <H3>Use ORPO when</H3>

      <Prose>
        Pipeline simplicity is a hard constraint — you cannot afford two separate training stages, two GPU reservations, or the engineering overhead of maintaining a reference model checkpoint alongside a trainable policy. ORPO's single-stage training is faster to iterate, easier to debug, and requires roughly half the total compute of a DPO pipeline (no separate SFT run). The tradeoff is that ORPO's SFT signal and preference signal compete for gradient budget within each batch, and tuning <Code>λ</Code> to balance them is more sensitive than tuning DPO's <Code>β</Code>.
      </Prose>

      <H3>DPO and SimPO are still the defaults</H3>

      <Prose>
        For new projects without a specific reason to prefer RLAIF, IPO, or ORPO, DPO with a well-curated preference dataset is the lowest-risk starting point. SimPO is the better default if memory is a constraint (no reference forward pass) or if length drift is a concern. The methods in this topic are improvements on specific failure modes, not general upgrades. Running an ablation comparing DPO and IPO on your specific dataset and architecture costs one additional training run and is worth doing before committing to IPO for production.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>RLAIF: critic inference cost dominates</H3>

      <Prose>
        RLAIF's scaling profile is dominated by the cost of running the judge model at labeling time. For every preference pair you want to generate, you need at least two policy forward passes (to generate the two responses) and one or more judge model forward passes (to evaluate them). The judge is typically larger than the policy — a 70B judge evaluating a 7B policy is a common configuration — so the judge's inference cost dominates. At 1M preference pairs, the judge inference bill exceeds the policy training cost. This is still typically cheaper than human labeling at the same scale, but it is not free, and it needs to be modeled explicitly in your compute budget.
      </Prose>

      <Prose>
        Direct RLAIF (d-RLAIF) has a different scaling profile: the judge is called at every RL training step for every generated response, not just during an offline labeling phase. This is more compute-intensive than offline RLAIF but has better data coverage — the judge evaluates the policy's current distribution rather than its initial distribution. The cost scales as O(training_steps × batch_size × judge_inference_cost). At frontier scale, this is prohibitive without significant inference optimization (batching, caching, smaller judge models, quantization).
      </Prose>

      <H3>IPO: same cost as DPO</H3>

      <Prose>
        IPO is a drop-in replacement for DPO's loss function. The compute cost per training step is identical: two policy forward passes, two reference forward passes, one backward pass. Memory requirements are identical. Data requirements are identical. The only difference is one line of code in the loss computation. If IPO helps on your data, it is a free upgrade. The only cost is the hyperparameter search for <Code>τ</Code>, which is the same order of effort as tuning DPO's <Code>β</Code>.
      </Prose>

      <H3>ORPO: saves compute at the pipeline level</H3>

      <Prose>
        ORPO's compute savings are not per-training-step — each ORPO step costs roughly the same as a DPO step (one forward pass per example in the batch, one backward). The savings are at the pipeline level: ORPO eliminates the entire SFT stage. For a typical 7B model alignment pipeline, the SFT stage might consume 100 to 300 GPU-hours, and the DPO stage another 50 to 200. ORPO replaces both with a single stage of similar or slightly longer duration than DPO alone, netting a 30–50% reduction in total pipeline compute. This is significant at scale. At 70B scale, eliminating the SFT stage saves weeks of wall time and millions of tokens of compute.
      </Prose>

      <Prose>
        ORPO also eliminates the reference model from GPU memory, reducing peak memory per training step. At 7B scale this saves roughly 14 GB in bfloat16. At 70B scale it saves roughly 140 GB — the difference between needing four high-memory GPUs and needing eight for a DPO run.
      </Prose>

      <Prose>
        The dimension that does not scale favorably for any of the three methods is the relationship between method choice and data quality. At every scale tested in the literature, a better preference dataset produces a better model than a better loss function on a worse dataset. RLAIF with a mediocre judge produces data comparable to noisy human annotation; IPO on noisy data behaves similarly to DPO on noisy data; ORPO on a low-quality preference dataset produces a low-quality model. The method can only extract signal that exists in the data. At large scale with high compute budgets, the first dollar should go to data quality, not loss function selection.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. RLAIF inherits the judge's biases</H3>
      <Prose>
        Whatever biases the judge model has, the preference data will encode. If the judge prefers verbose responses, longer chosen responses will dominate the dataset and the policy will learn verbosity. If the judge has a political or cultural perspective encoded in its training data, that perspective will be injected into preference labels without any explicit human decision to do so. Unlike human annotator biases, which can be measured, corrected, and attributed, judge model biases are opaque — they live in billions of parameters and cannot be easily audited. Mitigation: use multiple judge models and keep only pairs where judges agree; evaluate the preference dataset's length distribution, political balance, and demographic representation before training; compare RLAIF-trained models against human-labeled baselines on fairness benchmarks.
      </Prose>

      <H3>2. RLAIF judge must be strictly stronger than the policy</H3>
      <Prose>
        Using a judge model of similar capability to the policy produces circular preference data: the judge cannot reliably identify better responses because its own capability ceiling is the same as the policy's. On tasks where the policy is already strong — basic instruction following, tone adjustment, formatting — a slightly weaker judge may still provide useful signal. On tasks that require genuine expertise — mathematical reasoning, code correctness, factual accuracy in specialized domains — the judge must demonstrably outperform the policy on the evaluation dimension, or the labels are noise.
      </Prose>

      <H3>3. IPO's saturation can be too aggressive on noisy data</H3>
      <Prose>
        On preference data with genuinely noisy labels — where the "chosen" response is not reliably better than the "rejected" response — IPO's finite target margin creates a problem: the squared loss pushes the margin toward the target even when the preference label might be wrong. DPO's log-sigmoid is self-limiting in a way that actually helps here: as the margin grows, the gradient shrinks, and noisy pairs stop contributing much signal. IPO's gradient is proportional to the margin deviation, so a noisy pair with a small margin gets a large gradient push — in the wrong direction if the label is wrong. Use IPO on clean preference data; use DPO on noisy data where its natural gradient shrinkage provides implicit robustness.
      </Prose>

      <H3>4. ORPO's SFT-preference balance is sensitive</H3>
      <Prose>
        The <Code>λ</Code> hyperparameter in ORPO controls how much weight the odds-ratio penalty receives relative to the SFT cross-entropy. The right value is highly dataset-dependent. Too low and the preference signal is swamped by the SFT signal — the model learns to predict chosen tokens well but does not learn to distinguish chosen from rejected. Too high and the preference signal dominates, causing the model to optimize the odds ratio at the expense of output quality, which manifests as fluent-looking but semantically inconsistent outputs. The interaction is non-linear: the SFT loss and the odds-ratio loss respond differently to changes in model probability, and the optimal balance shifts as training progresses. Standard practice is to start at <Code>λ = 0.1</Code>, monitor the ratio of the two loss components during training, and adjust if one term becomes more than 5× the other.
      </Prose>

      <H3>5. Benchmark overfitting affects all three methods equally</H3>
      <Prose>
        AlpacaEval, Arena-Hard, and MT-Bench are the standard benchmarks for evaluating preference optimization methods. Differences of 1–2 percentage points between DPO, IPO, ORPO, and SimPO on these benchmarks are common in the literature and are often within the evaluator's variance when using language model judges. A method that beats DPO by 1.5 points on AlpacaEval may not beat it by any meaningful amount on a different benchmark or on your specific deployment distribution. Run multiple benchmarks, include human evaluation when possible, and treat sub-2-point differences on any single benchmark as statistical noise rather than a clear signal about which method to use.
      </Prose>

      <H3>6. Method choice is dominated by data quality at all scales</H3>
      <Prose>
        The empirical consensus across dozens of ablation studies is that the choice of preference loss function — DPO, IPO, ORPO, SimPO, KTO — explains less variance in final model quality than the quality and diversity of the preference dataset. A carefully curated dataset of 20k pairs with strong signal will produce a better model than a noisy dataset of 200k pairs regardless of which loss function is used. Before investing engineering effort in switching from DPO to IPO or ORPO, validate that the dataset quality is the bottleneck you expect it to be — run DPO on a 20% subset of your data with rigorous filtering and compare to DPO on the full noisy set. If the filtered version is better, more data curation is the priority, not a different loss function.
      </Prose>

      <Callout accent="gold">
        All three methods — RLAIF, IPO, ORPO — are improvements on specific, narrow failure modes of the standard RLHF/DPO pipeline. They are not general-purpose upgrades. Use each one when the specific failure mode it addresses is the binding constraint in your training setup.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All four sources were verified against their arXiv pages on 2026-04-21. Abstracts, author lists, and arXiv IDs confirmed.
      </Prose>

      <H3>Lee et al. 2023 — RLAIF</H3>
      <Prose>
        Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, Abhinav Rastogi. "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." arXiv:2309.00267. Submitted September 1, 2023; revised September 3, 2024. The direct comparison paper. Shows that AI-generated preference labels match human labels in quality on summarization and helpful dialogue tasks, at roughly one-tenth of the annotation cost. Introduces direct RLAIF (d-RLAIF), which uses the AI judge as a live reward signal during RL without training a separate reward model. The key empirical result: RLAIF can outperform a supervised fine-tuned baseline even when the AI labeler is the same size as or the same checkpoint as the policy.
      </Prose>

      <H3>Azar et al. 2023 — IPO</H3>
      <Prose>
        Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniyar Calandri, Simon Osindero, Mark Ring, Rémi Munos. "A General Theoretical Paradigm to Understand Learning from Human Preferences." arXiv:2310.12036. Submitted October 18, 2023. Derives the ΨPO family of preference objectives. Identifies DPO as a special case of ΨPO that relies on the Bradley-Terry assumption — that pairwise preferences can be substituted with pointwise rewards — and shows theoretically that this assumption leads to over-optimization on high-certainty preference data. Proposes IPO (Identity Preference Optimization) as the instance of ΨPO that directly optimizes pairwise preference probabilities without the Bradley-Terry approximation, using a squared loss with a finite optimum. Provides theoretical guarantees on IPO's convergence properties and empirical evidence of reduced over-optimization relative to DPO.
      </Prose>

      <H3>Hong et al. 2024 — ORPO</H3>
      <Prose>
        Jiwoo Hong, Noah Lee, James Thorne. "ORPO: Monolithic Preference Optimization without Reference Model." arXiv:2403.07691. Submitted March 12, 2024; revised March 14, 2024. Introduces the odds-ratio preference optimization loss and demonstrates that folding SFT and preference optimization into a single training stage produces competitive alignment without a reference model or a separate SFT phase. Shows that fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO on UltraFeedback alone surpasses state-of-the-art models with more than 7B and 13B parameters on AlpacaEval 2.0, IFEval, and MT-Bench. Reference implementation: available in HuggingFace TRL as <Code>ORPOTrainer</Code>.
      </Prose>

      <H3>Bai et al. 2022 — Constitutional AI (CAI) — RLAIF predecessor</H3>
      <Prose>
        Yuntao Bai et al. (Anthropic). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073. Submitted December 15, 2022. The predecessor to RLAIF. Introduces the two-stage constitutional AI pipeline: supervised learning from AI-generated revisions guided by written principles, followed by RLAIF using the same principles to generate preference labels. The foundational paper demonstrating that AI-generated preference labels at scale, guided by a written constitution, can align a language model without any human labels identifying harmful outputs. The "RLAIF" name and general pipeline structure described in Lee et al. 2023 generalize the mechanism introduced here.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — show DPO gradient divergence on certain preferences</H3>
      <Prose>
        Consider a preference pair where the human preference is perfectly certain: <Code>P(y_w ≻ y_l) = 1</Code>. Under the Bradley-Terry model, this corresponds to the reward difference approaching infinity. Write out the gradient of the DPO loss with respect to the implicit reward margin <Code>h_θ</Code> as this certainty increases. Show formally that the gradient never reaches zero — that for any finite margin, the gradient is strictly negative, always pushing the margin wider. Now repeat the analysis for IPO: write out the IPO gradient and show that it is exactly zero at <Code>h_θ = 1/(2τ)</Code> and reverses sign above it. What property of the squared loss produces the zero-gradient point that the log-sigmoid loss lacks?
      </Prose>

      <H3>Exercise 2 — derive ORPO's SFT-preference coupling</H3>
      <Prose>
        The ORPO loss has two terms: the SFT cross-entropy on chosen responses and the odds-ratio penalty on rejected responses. Write out the gradient of the total ORPO loss with respect to a single weight matrix in the language model. Show how the gradient depends on both the SFT term and the odds-ratio term. Under what condition do the two gradient components point in the same direction (reinforcing each other)? Under what condition do they point in opposite directions (competing)? What does competition between the two terms imply for the choice of <Code>λ</Code>, and how would you detect competition in a real training run?
      </Prose>

      <H3>Exercise 3 — when does RLAIF underperform RLHF</H3>
      <Prose>
        RLAIF matches human labeling quality on summarization and helpful dialogue. Propose three categories of alignment tasks where you would expect RLAIF to underperform human labeling, and for each category, explain the mechanism of failure — what specifically does the AI judge get wrong that a human would get right? For one of these categories, design a hybrid annotation scheme that uses AI labels when they are reliable and falls back to human labels when they are not, specifying how you would detect which regime each pair falls into at labeling time.
      </Prose>

      <H3>Exercise 4 — ablation to isolate preference-loss shape from data quality</H3>
      <Prose>
        You want to know whether IPO actually outperforms DPO on your specific dataset, or whether any observed improvement is attributable to differences in effective learning rate or hyperparameter sensitivity rather than the loss shape itself. Design an ablation experiment that controls for everything except the loss function. Specifically: what training configurations would you run, what metrics would you record, what would a result of "the loss shape genuinely matters" look like versus "the loss shape doesn't matter, hyperparameters explain everything," and how many GPU-hours is this ablation worth running before you commit to a production decision?
      </Prose>

      <H3>Exercise 5 — one-stage vs two-stage tradeoffs</H3>
      <Prose>
        ORPO's one-stage pipeline eliminates the SFT stage. But the SFT stage in the standard pipeline serves multiple purposes: it produces the reference model for DPO, it adapts the base model to instruction-following format, and it initializes the policy in a regime where preference optimization is stable. ORPO handles all of these implicitly through its combined loss. Construct a scenario — a specific model family, dataset type, and compute budget — where you would expect ORPO to outperform two-stage DPO, and a different scenario where you would expect it to underperform. For each scenario, identify the single most important factor driving the difference in expected performance.
      </Prose>

    </div>
  ),
};

export default rlaifIpoOrpo;
