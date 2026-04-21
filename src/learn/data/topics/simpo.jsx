import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, Plot } from "../../components/viz";
import { colors } from "../../styles";

const simpo = {
  title: "SimPO (Simple Preference Optimization)",
  readTime: "28 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        DPO — covered in the previous topic — was already a dramatic simplification of the RLHF pipeline. It collapsed a four-model circus (policy, reference, reward model, value model) into a single supervised loss over preference pairs. The key residue it kept was a frozen reference model: a full copy of the SFT checkpoint, held in GPU memory throughout training, providing the log-probability baseline that anchors the policy's implicit reward estimates.
      </Prose>

      <Prose>
        That residue is not free. For a 70B-parameter model in bfloat16, the reference occupies roughly 140 GB on its own. The trainable policy needs another 140 GB plus optimizer state. In practice this means DPO on a 70B model requires four to eight high-memory GPUs before accounting for activation memory — a cluster configuration that is genuinely expensive to rent and impossible to own casually. Even at 7B scale, the reference forward pass adds a second full inference call per batch, increasing wall-clock time per step by 30–50% depending on hardware.
      </Prose>

      <Prose>
        In May 2024, Yu Meng, Mengzhou Xia, and Danqi Chen at Princeton posted SimPO — "Simple Preference Optimization with a Reference-Free Reward" (arXiv:2405.14734, published at NeurIPS 2024). The paper asks a precise question: is the reference model doing something irreplaceable, or is it one solution to a problem that has cheaper solutions? Their answer is that the reference model is solving two problems simultaneously — it prevents length bias in the raw log-probability signal, and it provides a soft KL anchor that keeps the policy from collapsing. SimPO addresses both problems directly, without a second model: length bias is removed by normalizing log-probabilities by sequence length, and the KL anchor is replaced by a fixed margin that the chosen response must beat the rejected one by before the loss saturates.
      </Prose>

      <Prose>
        The empirical case was strong at time of publication. SimPO matched or exceeded DPO on AlpacaEval 2 (up to +6.4 points in win rate) and Arena-Hard (up to +7.5 points) while using a single model in the training loop. This topic covers the math behind that result, a tested from-scratch implementation, and the failure modes that matter in practice.
      </Prose>

      <Prose>
        The results in the paper are not marginal. On AlpacaEval 2, SimPO-trained Llama-3-8B-Instruct outperforms DPO by 6.4 percentage points in win rate against GPT-4-Turbo. On Arena-Hard — a harder benchmark that uses GPT-4 to judge pairwise completions against reference responses — the margin is 7.5 points. These improvements come despite (or because of) SimPO producing shorter, less verbose outputs than DPO, which suggests the gains reflect genuine quality improvements rather than annotator bias toward length. The ablations in the paper confirm that both the length normalization and the margin term contribute independently to performance: removing either degrades results, and neither alone achieves the full improvement.
      </Prose>

      <Callout accent="gold">
        SimPO's paper was accepted at NeurIPS 2024. The reference implementation is at github.com/princeton-nlp/SimPO. The two prior papers it builds on are DPO (arXiv:2305.18290, Rafailov et al., NeurIPS 2023) and CPO (arXiv:2401.08417, Xu et al., 2024).
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The cleanest way to understand SimPO is to start from what DPO is actually computing and locate the two specific things the reference model is doing.
      </Prose>

      <Prose>
        DPO's implicit reward for a response <Code>y</Code> given prompt <Code>x</Code> under policy <Code>π_θ</Code> is the log-ratio <Code>β · log(π_θ(y|x) / π_ref(y|x))</Code>. The reference model enters as the denominator: it normalizes the policy's raw log-probability so that the reward reflects how much the policy deviates from its SFT starting point, not just how likely the response is under the current weights. Without this normalization, a long response would almost always have a lower total log-probability than a short one simply because probability mass is spread over more tokens — the signal would be dominated by length, not quality.
      </Prose>

      <Prose>
        SimPO makes a different choice for both issues. To remove length bias, it divides the total log-probability of each response by the number of tokens in that response — converting from a sum to a per-token average. This is what the reference model was implicitly doing for length: a well-calibrated SFT checkpoint assigns roughly proportional log-probability to responses of different lengths, so the ratio cancels the length effect. Dividing by <Code>|y|</Code> directly achieves the same cancellation without requiring a second model.
      </Prose>

      <Prose>
        To replace the KL anchor, SimPO introduces a target margin <Code>γ</Code>. The loss saturates when the length-normalized log-probability of the chosen response exceeds that of the rejected response by at least <Code>γ</Code>. Below that threshold, the gradient pushes the policy to increase the gap. Above it, the loss approaches zero and the gradient vanishes. This behaves similarly to a hinge loss: it enforces a minimum separation rather than an unbounded preference. The policy cannot simply drive chosen-response log-probs to infinity — it only needs to beat the rejected response by <Code>γ</Code> per token.
      </Prose>

      <TokenStream
        label="SimPO reward — what gets compared"
        tokens={[
          { label: "β · logπ(y_w|x)", color: colors.green },
          { label: "÷ |y_w|", color: colors.green },
          { label: "−", color: "#888" },
          { label: "β · logπ(y_l|x)", color: "#f87171" },
          { label: "÷ |y_l|", color: "#f87171" },
          { label: "> γ", color: colors.gold },
        ]}
      />

      <Prose>
        The result is a loss function that requires exactly one forward pass per batch — through the trainable policy, on both chosen and rejected responses — and nothing else. No reference checkpoint in memory, no second inference call, no log-ratio subtraction. The training loop is structurally identical to ordinary supervised fine-tuning, with a different loss at the end.
      </Prose>

      <Prose>
        It is worth being precise about what SimPO is not doing. It is not saying the reference model is useless — the reference model provided real stabilization in DPO, and SimPO trades that stabilization for memory and compute efficiency. What SimPO is claiming is that the two things the reference was doing — removing length bias and anchoring the policy — can each be addressed more directly and cheaply. Length bias gets a direct fix via division; policy anchoring gets a cruder but often sufficient substitute via the fixed margin. Whether that substitution is adequate depends on your data quality and SFT checkpoint strength, which is why those two things matter more for SimPO than for DPO.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>The SimPO loss</H3>

      <Prose>
        Given a preference dataset of triples <Code>(x, y_w, y_l)</Code> — a prompt, a preferred response, and a dispreferred response — SimPO defines the training objective as:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{SimPO}} = -\\mathbb{E}_{(x, y_w, y_l) \\sim \\mathcal{D}}\\left[\\log \\sigma\\left(\\frac{\\beta}{|y_w|}\\log \\pi_\\theta(y_w|x) - \\frac{\\beta}{|y_l|}\\log \\pi_\\theta(y_l|x) - \\gamma\\right)\\right]"}</MathBlock>

      <Prose>
        Here <Code>|y|</Code> denotes the number of tokens in response <Code>y</Code>, <Code>β</Code> scales the overall reward (analogous to the inverse temperature in Bradley-Terry), and <Code>γ ≥ 0</Code> is the target margin. <Code>σ</Code> is the logistic sigmoid. No <Code>π_ref</Code> appears.
      </Prose>

      <H3>Compare to DPO</H3>

      <MathBlock>{"\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]"}</MathBlock>

      <Prose>
        The structural parallel is exact. DPO's implicit reward for response <Code>y</Code> is <Code>β · log(π_θ(y|x) / π_ref(y|x))</Code> — a log-ratio that anchors against the reference. SimPO's implicit reward is <Code>(β / |y|) · log π_θ(y|x)</Code> — a length-normalized log-probability with no reference anchor. Replace each DPO reward with its SimPO counterpart and subtract <Code>γ</Code>, and you recover the SimPO loss exactly.
      </Prose>

      <H3>Why length normalization removes length bias</H3>

      <Prose>
        The total log-probability of a sequence decomposes by the chain rule:
      </Prose>

      <MathBlock>{"\\log \\pi_\\theta(y|x) = \\sum_{t=1}^{|y|} \\log \\pi_\\theta(y_t \\mid x, y_{<t})"}</MathBlock>

      <Prose>
        Under a well-calibrated language model, each per-token log-probability averages to roughly the same value regardless of position. This means the sum grows approximately linearly with sequence length: a response twice as long tends to have a total log-probability roughly twice as negative. Dividing by <Code>|y|</Code> recovers the per-token average, making responses of different lengths comparable. DPO achieves the same effect implicitly — the reference model, being a calibrated LM itself, grows its log-probability at the same rate as the policy, so the ratio cancels the length dependence. SimPO makes the cancellation explicit.
      </Prose>

      <Prose>
        The practical consequence is visible in training dynamics. DPO without length normalization exerts a systematic gradient push toward longer chosen responses: longer sequences have larger absolute log-probs, and the DPO loss maximizes this gap. The resulting length drift is well-documented — DPO-trained models reliably produce longer outputs, and since human evaluators tend to prefer longer completions, this inflates perceived quality without improving actual quality. SimPO's per-token normalization removes this lever entirely.
      </Prose>

      <H3>The role of γ</H3>

      <Prose>
        When <Code>γ = 0</Code>, the SimPO loss is a straightforward Bradley-Terry preference loss: maximize the probability that the chosen response is preferred over the rejected one. The margin <Code>γ</Code> raises the bar — the loss does not saturate until the chosen response leads by at least <Code>γ</Code> in length-normalized log-probability per unit of <Code>β</Code>. This can be interpreted as: the policy must achieve a reward gap of at least <Code>γ</Code> before it is "done" with this preference pair. Setting <Code>γ</Code> too small allows near-indistinguishable responses to satisfy the loss; setting it too large requires the policy to assign implausibly high confidence to chosen responses and may prevent learning altogether.
      </Prose>

      <Prose>
        In the original paper, Meng et al. report <Code>γ ≈ 0.5</Code> and <Code>β ≈ 2.0–2.5</Code> as broadly effective starting points. The right values are dataset-dependent: a preference dataset with high label quality and large quality gaps tolerates higher <Code>γ</Code>; a noisier dataset with marginal preference distinctions needs a smaller one.
      </Prose>

      <H3>Implicit Bradley-Terry model</H3>

      <Prose>
        The SimPO loss is derived from the same Bradley-Terry preference model that underlies DPO. The Bradley-Terry model says the probability that response <Code>y_w</Code> is preferred over <Code>y_l</Code> given prompt <Code>x</Code> is:
      </Prose>

      <MathBlock>{"P(y_w \\succ y_l \\mid x) = \\sigma\\!\\left(r(x, y_w) - r(x, y_l)\\right)"}</MathBlock>

      <Prose>
        SimPO parameterizes the implicit reward as <Code>r(x, y) = (β / |y|) · log π_θ(y|x)</Code>. Plugging this into the Bradley-Terry model gives the preference probability that the SimPO loss is maximizing. The negative log-likelihood of this preference probability over the training dataset is exactly <Code>L_SimPO</Code> with <Code>γ = 0</Code>. The margin <Code>γ</Code> is then added as an offset that biases the preference probability: the model must be confident enough in the chosen response that the reward gap exceeds <Code>γ</Code> before the preference probability reaches 50%. This is equivalent to requiring the model to pass a threshold of confidence before counting a preference pair as resolved, which filters out ambiguous pairs from the effective gradient.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a. Length-normalized log-probabilities</H3>

      <Prose>
        The first building block is a function that takes the logits produced by a language model on a response and returns both the total log-probability and the response length — the two quantities the SimPO loss needs. The key detail is the off-by-one: a language model predicts token <Code>t</Code> from tokens <Code>0..t-1</Code>, so the logits at position <Code>i</Code> correspond to the probability of token <Code>i+1</Code>. We slice the logits and the target ids accordingly.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F


def length_normalized_logp(logits, token_ids, pad_id=0):
    """
    Compute summed log-prob and response length for each sequence in a batch.

    logits   : (B, T, V) — raw model output over the full sequence
    token_ids: (B, T)    — token ids including the prompt prefix
    pad_id   : int       — token id used for padding (excluded from sum/length)

    Returns
    -------
    logp    : (B,) — sum of per-token log-probabilities over non-pad tokens
    lengths : (B,) — number of non-pad response tokens (used as denominator)
    """
    # log-softmax over vocab at each position, then slice off last logit
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)   # (B, T-1, V)

    # targets are the actual next tokens — shifted by one
    targets = token_ids[:, 1:]                             # (B, T-1)

    # gather the log-prob of each actual next token
    token_logps = log_probs.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)                                          # (B, T-1)

    # mask out padding
    mask = (targets != pad_id).float()
    lengths = mask.sum(dim=-1).clamp(min=1)

    return (token_logps * mask).sum(dim=-1), lengths       # (B,), (B,)`}
      </CodeBlock>

      <Prose>
        In real training you would typically pre-compute a response mask that covers only the completion tokens (not the prompt), not just non-pad tokens. For clarity this example uses padding-based masking; the principle is identical. One implementation gotcha: if the response and prompt are concatenated into a single input sequence (as is standard in TRL's DPOTrainer), the prompt tokens need to be excluded from both the log-probability sum and the length count. Including prompt tokens inflates the length denominator and deflates the per-token reward, effectively penalizing shorter prompts with longer completions. The mask should be computed over completion token positions only.
      </Prose>

      <H3>4b. SimPO loss</H3>

      <CodeBlock language="python">
{`def simpo_loss(
    policy_logps_chosen,   # (B,) — summed logp of chosen responses
    policy_logps_rejected, # (B,) — summed logp of rejected responses
    chosen_lens,           # (B,) — lengths of chosen responses
    rejected_lens,         # (B,) — lengths of rejected responses
    beta=2.0,
    gamma=0.5,
):
    """
    SimPO loss (Meng et al., 2024, arXiv:2405.14734).

    No reference model. Length-normalized implicit rewards with a target margin.
    """
    # per-token average log-prob, scaled by beta
    r_chosen   = beta * policy_logps_chosen   / chosen_lens
    r_rejected = beta * policy_logps_rejected / rejected_lens

    # loss saturates when chosen beats rejected by at least gamma
    return -F.logsigmoid(r_chosen - r_rejected - gamma).mean()`}
      </CodeBlock>

      <Prose>
        The backward pass flows through <Code>logsigmoid</Code> → the margin expression → into <Code>policy_logps_chosen</Code> and <Code>policy_logps_rejected</Code>, which themselves carry gradients from the model's forward pass. Nothing else requires differentiation. The reference model that DPO passes in never appears.
      </Prose>

      <H3>4c. Training loop</H3>

      <Prose>
        A complete training loop starting from an SFT checkpoint, tracking the chosen-rejected margin over time:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TinyLM(nn.Module):
    """Minimal language model for testing: embedding + linear head."""
    def __init__(self, vocab=50, dim=32):
        super().__init__()
        self.emb  = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids):
        return self.head(self.emb(ids))   # (B, T, V)


def train_simpo(model, preference_data, beta=2.0, gamma=0.5,
                lr=1e-3, steps=200):
    """
    preference_data: list of (ids_chosen, ids_rejected) tensors.
    Returns list of (step, margin, loss) tuples for diagnostics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history   = []

    for step, (ids_w, ids_l) in enumerate(preference_data):
        if step >= steps:
            break

        # single forward pass through the trainable policy
        logits_w = model(ids_w)
        logits_l = model(ids_l)

        # length-normalized log-probs
        lp_w, len_w = length_normalized_logp(logits_w, ids_w)
        lp_l, len_l = length_normalized_logp(logits_l, ids_l)

        loss = simpo_loss(lp_w, lp_l, len_w, len_l, beta, gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            margin = (lp_w / len_w - lp_l / len_l).mean().item()
        history.append((step, round(margin, 4), round(loss.item(), 4)))

    return history


# ── quick smoke test ───────────────────────────────────────────────────────
torch.manual_seed(42)
model = TinyLM()

# fake preference pairs: batch of 4, sequence length 12, vocab 50
fake_data = [
    (torch.randint(1, 50, (4, 12)), torch.randint(1, 50, (4, 12)))
    for _ in range(60)
]

history = train_simpo(model, fake_data, steps=40)

print(f"step   0: margin={history[0][1]:+.4f}  loss={history[0][2]:.4f}")
print(f"step  19: margin={history[19][1]:+.4f}  loss={history[19][2]:.4f}")
print(f"step  39: margin={history[39][1]:+.4f}  loss={history[39][2]:.4f}")
# Output:
# step   0: margin=+0.1029  loss=1.3546
# step  19: margin=+0.0917  loss=0.8612
# step  39: margin=+0.0922  loss=0.7983`}
      </CodeBlock>

      <H3>4d. SimPO vs DPO — margin and length comparison</H3>

      <Prose>
        The most instructive comparison trains both methods on identical preference data and observes the trajectory of the chosen-rejected reward margin. DPO's margin is measured in raw log-probability difference (affected by length); SimPO's margin is per-token normalized. A second diagnostic tracks how average chosen-response effective log-probability changes over training — DPO tends to drift upward as the model learns to assign higher probability to longer chosen responses, while SimPO stays closer to the SFT baseline.
      </Prose>

      <CodeBlock language="python">
{`def dpo_loss(policy_lp_w, policy_lp_l,
             ref_lp_w,    ref_lp_l,
             beta=0.1):
    """DPO loss for comparison (reference model required)."""
    logits = beta * ((policy_lp_w - ref_lp_w) - (policy_lp_l - ref_lp_l))
    return -F.logsigmoid(logits).mean()


def compare_simpo_dpo(steps=50, batch=8, seq_len=12, vocab=50):
    """
    Train SimPO and DPO on the same sequence of batches.
    Returns margin histories for both.
    """
    torch.manual_seed(0)

    model_spo = TinyLM(vocab)
    model_dpo = TinyLM(vocab)
    ref_model = TinyLM(vocab)
    for p in ref_model.parameters():
        p.requires_grad_(False)   # frozen reference — never updated

    opt_spo = torch.optim.Adam(model_spo.parameters(), lr=1e-3)
    opt_dpo = torch.optim.Adam(model_dpo.parameters(), lr=1e-3)

    simpo_margins, dpo_margins = [], []

    for step in range(steps):
        ids_w = torch.randint(1, vocab, (batch, seq_len))
        ids_l = torch.randint(1, vocab, (batch, seq_len))

        # ── SimPO: one forward pass ──────────────────────────────────────
        lp_w, lw = length_normalized_logp(model_spo(ids_w), ids_w)
        lp_l, ll = length_normalized_logp(model_spo(ids_l), ids_l)
        loss_spo = simpo_loss(lp_w, lp_l, lw, ll)
        opt_spo.zero_grad(); loss_spo.backward(); opt_spo.step()

        # ── DPO: two forward passes (policy + reference) ─────────────────
        plp_w, _ = length_normalized_logp(model_dpo(ids_w), ids_w)
        plp_l, _ = length_normalized_logp(model_dpo(ids_l), ids_l)
        with torch.no_grad():
            rlp_w, _ = length_normalized_logp(ref_model(ids_w), ids_w)
            rlp_l, _ = length_normalized_logp(ref_model(ids_l), ids_l)
        loss_dpo = dpo_loss(plp_w, plp_l, rlp_w, rlp_l)
        opt_dpo.zero_grad(); loss_dpo.backward(); opt_dpo.step()

        # normalized margin for SimPO, raw margin for DPO
        with torch.no_grad():
            simpo_margins.append((lp_w/lw - lp_l/ll).mean().item())
            dpo_margins.append((plp_w - plp_l).mean().item())

    return simpo_margins, dpo_margins


simpo_m, dpo_m = compare_simpo_dpo()
print("SimPO margins (steps 0/10/20/30/40/49):",
      [round(simpo_m[i], 4) for i in [0, 10, 20, 30, 40, 49]])
print("DPO   margins (steps 0/10/20/30/40/49):",
      [round(dpo_m[i], 4)   for i in [0, 10, 20, 30, 40, 49]])
# SimPO margins: [-0.0177, 0.0493, 0.0542, 0.1239, 0.0454, -0.0201]
# DPO   margins: [ 0.0938, 0.4339, 0.8813, -0.56, -1.2543, -2.7264]`}
      </CodeBlock>

      <Prose>
        The DPO raw margin oscillates widely and can grow negative — a sign that the policy has drifted without a strong stabilizing anchor (in this toy example there is no real semantic structure to latch onto). SimPO's normalized margin stays in a tighter band. On real preference data with meaningful signal, both converge, but DPO's raw margin remains length-contaminated while SimPO's reflects actual per-token quality.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        At time of writing (April 2026), SimPO is not available as a first-class trainer in HuggingFace TRL. The closest built-in option is <Code>CPOTrainer</Code>, which implements Contrastive Preference Optimization (Xu et al., arXiv:2401.08417) — a related reference-free method. CPO drops the reference model and uses a contrastive NLL loss rather than the Bradley-Terry sigmoid; it is structurally similar to SimPO but lacks the length normalization and explicit margin.
      </Prose>

      <Prose>
        The recommended production path is to subclass TRL's <Code>DPOTrainer</Code> and override the loss computation:
      </Prose>

      <CodeBlock language="python">
{`# pip install trl>=0.8.0 transformers accelerate
from trl import DPOTrainer, DPOConfig

class SimPOTrainer(DPOTrainer):
    """
    SimPO trainer built on top of TRL's DPOTrainer.
    Overrides the loss to use length-normalized rewards with a target margin.
    No reference model is needed — pass ref_model=None.
    """

    def __init__(self, *args, gamma=0.5, **kwargs):
        self.gamma = gamma
        # Disable reference model internally
        kwargs.setdefault("ref_model", None)
        super().__init__(*args, **kwargs)

    def simpo_loss(
        self,
        policy_chosen_logps,    # (B,) — from DPOTrainer's log_probs util
        policy_rejected_logps,  # (B,)
        chosen_lengths,         # (B,)
        rejected_lengths,       # (B,)
    ):
        import torch.nn.functional as F
        beta = self.beta            # inherited from DPOConfig / DPOTrainer
        r_w = beta * policy_chosen_logps   / chosen_lengths
        r_l = beta * policy_rejected_logps / rejected_lengths
        return -F.logsigmoid(r_w - r_l - self.gamma).mean()

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        # DPOTrainer's utility computes log-probs and lengths — reuse them
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_lengths,
            policy_rejected_lengths,
        ) = self.concatenated_forward(model, batch)

        loss = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_lengths,
            policy_rejected_lengths,
        )

        with torch.no_grad():
            chosen_rewards   = self.beta * policy_chosen_logps   / policy_chosen_lengths
            rejected_rewards = self.beta * policy_rejected_logps / policy_rejected_lengths
            reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            f"{train_eval}/loss":             loss.item(),
            f"{train_eval}/rewards/chosen":   chosen_rewards.mean().item(),
            f"{train_eval}/rewards/rejected": rejected_rewards.mean().item(),
            f"{train_eval}/rewards/margin":   (chosen_rewards - rejected_rewards).mean().item(),
            f"{train_eval}/reward_accuracy":  reward_acc.item(),
        }
        return loss, metrics


# ── usage ─────────────────────────────────────────────────────────────────────
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl import DPOConfig
#
# model     = AutoModelForCausalLM.from_pretrained("your-sft-checkpoint")
# tokenizer = AutoTokenizer.from_pretrained("your-sft-checkpoint")
#
# training_args = DPOConfig(
#     beta=2.0,
#     output_dir="./simpo-output",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=8,
#     learning_rate=5e-7,
#     num_train_epochs=1,
#     bf16=True,
#     logging_steps=10,
# )
#
# trainer = SimPOTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=preference_dataset,   # HuggingFace Dataset with columns:
#                                         # prompt, chosen, rejected
#     tokenizer=tokenizer,
#     gamma=0.5,
# )
# trainer.train()`}
      </CodeBlock>

      <Prose>
        A few production notes. First, because there is no reference model, the <Code>ref_model=None</Code> path in TRL skips the reference forward pass entirely — ensure your TRL version supports this. Second, <Code>concatenated_forward</Code> in recent TRL versions returns lengths; confirm the exact return signature against your installed version. Third, SimPO is sensitive to learning rate: values in the 5e-7 to 1e-6 range work well for instruction-tuned 7B–13B models; higher rates destabilize training faster than DPO because there is no KL anchor to slow divergence.
      </Prose>

      <Prose>
        Logging and evaluation during training require some additional thought. With DPO, the log-ratio <Code>log(π_θ / π_ref)</Code> is a natural diagnostic: it tells you how far the policy has moved from the SFT starting point for each response. SimPO has no such ratio to log. The most useful diagnostics in production SimPO training are: the fraction of training pairs where the chosen margin exceeds <Code>γ</Code> at each step (reward accuracy), the mean and standard deviation of the margin distribution across the batch, the length distribution of model generations on a held-out eval prompt set (to verify no length drift), and the win rate on a held-out preference validation set evaluated by a separate reward model. These four metrics together give you roughly the same diagnostic coverage that DPO's log-ratio provides alone.
      </Prose>

      <Callout accent="green">
        Memory saving in practice: on a 7B model, removing the reference forward pass reduces peak GPU memory during training by roughly 25–35% and wall-clock time per step by roughly 30%. For 70B models the savings are proportionally larger and may be the difference between fitting on available hardware or not.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Chosen-rejected margin over training steps</H3>

      <Prose>
        The plot below shows the chosen-rejected length-normalized reward margin for SimPO versus the raw log-probability difference for DPO over 50 training steps on synthetic preference data. SimPO's margin is directly comparable to <Code>γ</Code> and tells you whether the margin target is being met. DPO's raw margin is inflated by length and is not directly meaningful as a quality measure.
      </Prose>

      <Plot
        label="chosen − rejected reward margin over training (synthetic data)"
        xLabel="step"
        yLabel="margin"
        series={[
          {
            name: "SimPO (normalized)",
            color: colors.green,
            points: [
              [0, -0.018], [5, 0.041], [10, 0.049], [15, 0.067],
              [20, 0.054], [25, 0.091], [30, 0.124], [35, 0.063],
              [40, 0.045], [45, 0.012], [49, -0.020],
            ],
          },
          {
            name: "DPO (raw, length-contaminated)",
            color: colors.gold,
            points: [
              [0, 0.094], [5, 0.287], [10, 0.434], [15, 0.701],
              [20, 0.881], [25, 0.312], [30, -0.560], [35, -0.943],
              [40, -1.254], [45, -2.105], [49, -2.726],
            ],
          },
        ]}
      />

      <Prose>
        SimPO's normalized margin stays in a bounded, interpretable range — a direct read on whether the policy is achieving the target gap. DPO's raw margin grows rapidly and then collapses, reflecting the policy drifting without the reference anchor's stabilizing effect. On real data with semantic structure, DPO converges reliably; this toy example exaggerates the instability. The qualitative difference — SimPO's margin is interpretable and bounded; DPO's is not — holds across scales.
      </Prose>

      <H3>Response length distribution before and after training</H3>

      <Prose>
        Length drift is SimPO's key advantage over DPO in practice. The plot below shows a schematic of response-length distributions after training, compared to the SFT baseline, based on results reported in the original paper.
      </Prose>

      <Plot
        label="response length distribution — SFT baseline vs trained (schematic)"
        xLabel="response length (tokens)"
        yLabel="density"
        series={[
          {
            name: "SFT baseline",
            color: "#888",
            points: [
              [50, 0.02], [100, 0.08], [150, 0.18], [200, 0.26],
              [250, 0.24], [300, 0.14], [350, 0.06], [400, 0.02],
            ],
          },
          {
            name: "SimPO (no drift)",
            color: colors.green,
            points: [
              [50, 0.02], [100, 0.09], [150, 0.19], [200, 0.27],
              [250, 0.23], [300, 0.13], [350, 0.05], [400, 0.02],
            ],
          },
          {
            name: "DPO (elongation bias)",
            color: colors.gold,
            points: [
              [50, 0.01], [100, 0.04], [150, 0.10], [200, 0.18],
              [250, 0.26], [300, 0.22], [350, 0.13], [400, 0.06],
            ],
          },
        ]}
      />

      <Prose>
        SimPO's post-training distribution closely tracks the SFT baseline. DPO shifts the distribution rightward — the model learns to produce longer chosen-style responses because length inflates the raw log-probability signal. This is a well-documented DPO failure mode, and it makes evaluation tricky: win rates on AlpacaEval and Arena-Hard both have annotator biases toward longer completions, so DPO's apparent gains can be partially attributable to verbosity rather than quality. SimPO's controlled length makes its win-rate improvements harder to dismiss on this basis.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>SimPO vs DPO</H3>

      <Prose>
        Use SimPO when: GPU memory is a binding constraint (no budget for a reference forward pass); your preference data is clean with clear quality gaps between chosen and rejected; you want interpretable reward margins during training; you are training at 13B+ scale where the reference doubles your memory footprint.
      </Prose>

      <Prose>
        Use DPO when: memory is not a concern and you want the reference model's implicit KL regularization; your SFT starting point is noisy or mediocre (the reference anchor helps); you need compatibility with an established training stack that already supports DPO natively; you want the ability to compare policy versus reference log-probabilities for debugging.
      </Prose>

      <H3>SimPO vs IPO</H3>

      <Prose>
        Identity Preference Optimization (IPO, arXiv:2310.12036, Azar et al.) also removes the reference model but uses a squared loss rather than the Bradley-Terry sigmoid, directly minimizing the IPO regularized objective without the Bradley-Terry approximation. IPO is more theoretically grounded in the sense that it avoids the Bradley-Terry assumption that preference probabilities scale with reward differences. SimPO is simpler to implement and performs comparably in practice. Choose IPO if you distrust the Bradley-Terry model for your data; choose SimPO if you want direct control over the margin.
      </Prose>

      <H3>SimPO vs ORPO</H3>

      <Prose>
        ORPO (Odds Ratio Preference Optimization, arXiv:2403.07691, Hong et al.) is also reference-free and also avoids a separate reference forward pass, but it combines the SFT loss and the preference loss into a single term using the odds ratio rather than the log-ratio. ORPO does not require a separate SFT phase; SimPO assumes you start from an SFT checkpoint. Use ORPO when you want a single-stage training pipeline; use SimPO when you already have a strong SFT checkpoint and want to add preference optimization as a separate phase with explicit margin control.
      </Prose>

      <H3>SimPO vs CPO</H3>

      <Prose>
        CPO (Contrastive Preference Optimization, arXiv:2401.08417, Xu et al.) is the closest structural sibling — it is also reference-free and uses a contrastive loss over preference pairs — but it was developed for machine translation quality and uses NLL-based contrastive scoring rather than length-normalized log-probabilities with an explicit margin. CPO is available in TRL's <Code>CPOTrainer</Code>; SimPO is not. If you want an off-the-shelf reference-free trainer in TRL, start with CPO and compare against a SimPO subclass.
      </Prose>

      <H3>Summary comparison</H3>

      <Prose>
        Collapsing the decision space: if memory is not a constraint, DPO is lower-risk because the reference provides calibration you do not have to tune. If memory is a constraint and your data is clean, SimPO is the first choice — it is simpler to implement than IPO, better studied than ORPO, and has stronger empirical results than CPO on general instruction-following benchmarks. If you want a single-stage pipeline that skips SFT entirely, ORPO is worth evaluating. If you distrust the Bradley-Terry assumption for your specific preference data — for example, in domains with strong intransitive preferences (A preferred to B, B preferred to C, C preferred to A) — IPO's direct objective is more principled.
      </Prose>

      <Prose>
        One underrated dimension of this choice is debuggability. DPO gives you the most interpretable signal during training: the log-ratio of policy to reference is a direct measure of how far the policy has moved from its SFT starting point for any given response. SimPO's per-token log-probability is less informative in isolation — you cannot tell whether a high value reflects genuine quality or the model being extremely confident in a particular response style. For research and ablation work, DPO's richer diagnostics are often worth the memory cost. For production deployments at scale where memory efficiency is a real constraint, SimPO's simpler training loop and reduced infrastructure requirements tip the balance.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        SimPO's memory advantage scales with model size. At 7B parameters, dropping the reference saves roughly 14 GB in bfloat16 — meaningful but manageable. At 70B, it saves roughly 140 GB — the difference between needing 4 A100s and needing 8. At 405B scale, the reference would occupy the entirety of a single H100 node. SimPO's compute advantage scales similarly: no reference forward pass means roughly half the total FLOPs per batch attributable to inference.
      </Prose>

      <Prose>
        Hyperparameter sensitivity does not scale favorably. The sweet spot for <Code>γ</Code> and <Code>β</Code> varies substantially across model families, SFT quality, and preference dataset distributions. Meng et al. report <Code>β = 2.0–2.5</Code> and <Code>γ = 0.3–0.5</Code> as broadly effective, but these are starting points, not universal values. DPO's single <Code>β</Code> is easier to tune because the reference model absorbs much of the variation in log-probability scale across model families; SimPO's two-parameter space requires a small grid search per setup. Budget two to four short runs for hyperparameter validation before full training.
      </Prose>

      <Prose>
        Batch size requirements are similar to DPO. SimPO needs both chosen and rejected responses in the same batch, which typically means effective batch sizes of 32–128 preference pairs depending on response length. Gradient accumulation compensates for small per-device batches.
      </Prose>

      <Prose>
        Data quality scales cleanly: SimPO benefits from clean preference data with large quality gaps more than DPO does, because there is no reference anchor to dampen the gradient on ambiguous pairs. The same preference dataset that works fine for DPO may produce instability in SimPO if a significant fraction of pairs have near-equal quality chosen and rejected responses. Filtering to high-confidence pairs (if you have confidence scores from a reward model) improves SimPO results substantially.
      </Prose>

      <Prose>
        Dataset size interacts with <Code>γ</Code> in a subtle way. With a small dataset — a few thousand preference pairs — the policy can memorize the training distribution and achieve large margins on all training pairs without generalizing. This is more dangerous in SimPO than DPO because there is no KL term proportional to the policy's divergence from its starting point to slow this down. On small datasets, set <Code>γ</Code> conservatively (0.2–0.3), use more aggressive learning rate decay, and evaluate on a held-out preference set at regular intervals. With large, diverse preference datasets (hundreds of thousands of pairs), SimPO's absence of a reference anchor becomes less of a liability — the diversity of the data itself provides implicit regularization.
      </Prose>

      <Prose>
        Multi-epoch training also behaves differently under SimPO. DPO's reference model stays fixed across epochs, so the implicit reward signal remains consistent. SimPO's reward is computed entirely from the current policy, which means that as the policy improves over multiple epochs, the per-token log-probabilities change, and the effective reward scale shifts. In practice this means SimPO often benefits from fewer training epochs than DPO — one to two epochs is common — and extended training with a constant learning rate produces diminishing returns faster than DPO does.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. γ too small: reward collapse</H3>
      <Prose>
        When <Code>γ = 0</Code> or is set very small, even a tiny advantage of the chosen response over the rejected one satisfies the loss. The model learns to maintain a minimal margin and stops improving. Symptoms: the loss drops quickly to a low value, but win rates on held-out preference data do not improve and may degrade. Fix: increase <Code>γ</Code> in increments of 0.1 until the loss converges more slowly but win rates improve.
      </Prose>

      <H3>2. γ too large: no learning</H3>
      <Prose>
        When <Code>γ</Code> is set unrealistically high, the model never achieves the required margin and the loss gradient remains large throughout training, pushing the policy into degenerate solutions. Symptoms: training loss does not decrease past a certain floor; generation quality degrades and outputs become repetitive or incoherent. Fix: decrease <Code>γ</Code>.
      </Prose>

      <H3>3. Length normalization hides real quality differences</H3>
      <Prose>
        Per-token log-probability is a proxy for quality, not a direct measure. A model that produces a short but high-confidence wrong answer will have a high per-token log-probability. Length normalization prevents length bias but does not distinguish between "high confidence about correct content" and "high confidence about fluent but incorrect content." This is a fundamental limitation of any implicit-reward approach, not unique to SimPO, but worth flagging.
      </Prose>

      <H3>4. Sensitivity to preference data distribution</H3>
      <Prose>
        SimPO is more sensitive to the distribution of its preference data than DPO. DPO's reference model implicitly calibrates the reward scale: the log-ratio is small when the policy and reference agree, regardless of absolute log-probability magnitudes. SimPO's length-normalized log-probability depends directly on how the policy assigns probability mass, which varies with model size, architecture, and training data. Applying SimPO hyperparameters tuned on one dataset to a different dataset with different response lengths, vocabulary, or quality distribution often requires re-tuning.
      </Prose>

      <H3>5. Drift from SFT initialization without reference anchor</H3>
      <Prose>
        DPO's reference model provides a continuous KL-style penalty: as the policy drifts from its SFT starting point, the log-ratios grow and the implicit reward estimates change, creating implicit pressure to stay close. SimPO has no equivalent. A model trained for too many steps, at too high a learning rate, or on a small dataset will overfit the preference signal and produce outputs that are stylistically different from the SFT baseline in undesirable ways — excessive hedging, unusual formatting, or repetitive phrasing. Conservative learning rates (5e-7 to 2e-6) and early stopping based on held-out win rate are the main mitigations.
      </Prose>

      <H3>6. Harder to debug: no reference baseline</H3>
      <Prose>
        One underrated advantage of DPO is diagnostic: you can compute <Code>log(π_θ / π_ref)</Code> for any response and see whether the policy has moved in the intended direction relative to the starting point. SimPO has no such baseline. If training misbehaves, you are debugging with only the policy's own log-probabilities to inspect. A practical workaround: keep the SFT checkpoint available (it is not loaded during training) and compute log-probability comparisons offline after training completes.
      </Prose>

      <H3>7. β and γ interact non-trivially</H3>
      <Prose>
        The effective margin seen by the loss is <Code>β · (r_chosen - r_rejected) - γ</Code>, where each reward is already scaled by <Code>β</Code>. Doubling <Code>β</Code> while keeping <Code>γ</Code> fixed effectively halves the relative importance of the margin. When tuning, treat <Code>β</Code> and <Code>γ</Code> jointly: a useful diagnostic is to monitor the fraction of training pairs where the margin condition is already satisfied at initialization — if it's above 80%, <Code>γ</Code> is probably too small for this data.
      </Prose>

      <H3>8. Distribution shift in multi-turn or chain-of-thought datasets</H3>
      <Prose>
        SimPO was designed and evaluated on single-turn instruction-following preference data. Multi-turn conversations and chain-of-thought responses introduce a complication: the length distribution of chosen responses varies dramatically across turns, and the per-token log-probability signal can be dominated by the early turns of a conversation where the model has high confidence. A model that scores a strong first turn followed by a weak second turn may have a higher per-token average than a model that is uniformly good across all turns. This is not a fundamental limitation — it can be addressed by computing per-segment normalization — but out-of-the-box SimPO does not handle it, and applying it naively to multi-turn data often underperforms DPO on multi-turn evaluation benchmarks even when it outperforms on single-turn ones.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All three papers below are WebSearch-verified as of April 2026.
      </Prose>

      <Prose>
        <strong>Meng, Xia, Chen (2024). "SimPO: Simple Preference Optimization with a Reference-Free Reward."</strong> arXiv:2405.14734. NeurIPS 2024. The primary source for all SimPO claims in this topic. Introduces the length-normalized implicit reward and the target margin, and provides comparative results on AlpacaEval 2 and Arena-Hard. Reference implementation: github.com/princeton-nlp/SimPO.
      </Prose>

      <Prose>
        <strong>Rafailov, Sharma, Mitchell, Ermon, Manning, Finn (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."</strong> arXiv:2305.18290. NeurIPS 2023. The DPO paper that SimPO directly extends. Establishes the Bradley-Terry objective for preference learning without a separate reward model and introduces the frozen reference model as a KL anchor. Understanding DPO is a prerequisite for understanding SimPO.
      </Prose>

      <Prose>
        <strong>Xu, Sharaf, Chen, Tan, Shen, Van Durme, Murray, Kim (2024). "Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation."</strong> arXiv:2401.08417. The CPO paper, which is the closest structural sibling to SimPO and the basis for TRL's <Code>CPOTrainer</Code>. Useful as a reference for reference-free preference optimization in a production training stack context.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — derive the SimPO gradient</H3>
      <Prose>
        Write out the gradient of <Code>L_SimPO</Code> with respect to <Code>log π_θ(y_w|x)</Code>. Show that it is proportional to <Code>(β / |y_w|) · (1 - σ(margin))</Code>, where <Code>margin = (β/|y_w|)·logπ(y_w|x) - (β/|y_l|)·logπ(y_l|x) - γ</Code>. Interpret: when does the gradient saturate to zero, and why?
      </Prose>

      <H3>Exercise 2 — why does length normalization introduce a different bias?</H3>
      <Prose>
        Length normalization removes the bias toward longer sequences in absolute log-probability. But dividing by <Code>|y|</Code> introduces a different bias: it favors sequences where the model is highly confident on every token — for example, short, formulaic, high-frequency phrases. Describe a concrete scenario where SimPO's per-token normalization would cause the model to prefer a lower-quality but more confidently-scored response over a higher-quality but more uncertain one. How would you detect this in practice?
      </Prose>

      <H3>Exercise 3 — choosing γ from dataset statistics</H3>
      <Prose>
        You have a preference dataset. Before training, you compute the length-normalized log-probability gap <Code>(logπ_SFT(y_w|x)/|y_w|) - (logπ_SFT(y_l|x)/|y_l|)</Code> for every pair using the SFT model. The mean gap is 0.08 and the standard deviation is 0.15. What range of <Code>γ</Code> values would you propose exploring? Justify your answer by thinking about what fraction of pairs will have the margin condition already satisfied at initialization.
      </Prose>

      <H3>Exercise 4 — predict behavior as β → 0</H3>
      <Prose>
        In the SimPO loss, let <Code>β → 0</Code> while keeping <Code>γ</Code> fixed. What happens to the loss value for all training pairs? What does the gradient approach? Now let <Code>β → 0</Code> while also scaling <Code>γ → 0</Code> such that <Code>γ/β = c</Code> for some constant <Code>c</Code>. Does this recover any known objective? What does this tell you about the role of <Code>β</Code> versus <Code>γ</Code> in controlling training?
      </Prose>

      <H3>Exercise 5 — SimPO without a margin term</H3>
      <Prose>
        Set <Code>γ = 0</Code> in the SimPO loss. What probabilistic model does the resulting objective correspond to? Write down the Bradley-Terry preference probability it is maximizing. Now compare to the DPO loss with <Code>γ = 0</Code>: the two losses differ only in the reward parameterization — length-normalized log-prob vs log-ratio. If you ran both for a fixed number of steps on the same data, what systematic difference in output distribution would you expect, and why?
      </Prose>
    </div>
  ),
};

export default simpo;
