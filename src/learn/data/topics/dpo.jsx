import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const dpo = {
  title: "DPO (Direct Preference Optimization)",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        The RLHF pipeline covered in the previous topic works, but it asks a lot of you. Train a reward model on preference pairs. Train a value model to estimate long-horizon returns. Run PPO with its four interleaved models, clipping ratios, KL penalties, and a dozen hyperparameters that interact in ways that are difficult to diagnose. A single training run requires keeping multiple model copies in memory simultaneously, and if any part of the chain is miscalibrated — a reward model that overvalues verbosity, a KL coefficient set too low — the output degrades in ways that can be subtle and hard to trace back to a cause.
      </Prose>

      <Prose>
        In 2023, Rafailov, Sharma, Mitchell, and collaborators showed something unexpected: the entire pipeline can be collapsed into a single supervised loss defined directly on preference pairs, with no reward model and no RL at all. The paper is called "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," and the method it describes — DPO — is now the default post-training alignment method for most open-weight models. The key move is a mathematical identity that was always sitting inside the RLHF objective, waiting to be noticed.
      </Prose>

      <H2>The key insight — the optimal policy has a closed form</H2>

      <Prose>
        The RLHF objective asks for a policy that maximizes expected reward while staying close to a reference policy <Code>π_ref</Code>, penalized by a KL divergence term. Written out, you are maximizing <Code>E[r(x, y)] - β · KL(π || π_ref)</Code> over responses <Code>y</Code> given prompt <Code>x</Code>. This is a classic KL-regularized reward maximization problem, and it has a known closed-form solution.
      </Prose>

      <Prose>
        The optimal policy under this objective is a Boltzmann distribution over responses, weighted by the reward and anchored to the reference:
      </Prose>

      <MathBlock>{"\\pi^*(y \\mid x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y \\mid x) \\exp\\left(\\frac{1}{\\beta} r(x, y)\\right)"}</MathBlock>

      <Prose>
        Here <Code>Z(x)</Code> is a partition function — a normalizing constant that sums the unnormalized weights over all possible responses at prompt <Code>x</Code>. In principle you cannot compute it, because the sum is over an infinite set. But you do not need to compute it directly. Instead, rearrange the equation to isolate the reward:
      </Prose>

      <MathBlock>{"r(x, y) = \\beta \\log \\frac{\\pi^*(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)} + \\beta \\log Z(x)"}</MathBlock>

      <Prose>
        This is an exact identity: any reward function that produces <Code>π*</Code> as its optimal policy must have this specific form. Now look at what happens when you plug this expression into the Bradley-Terry model for pairwise preferences — the same preference probability formula from the RLHF topic. Bradley-Terry says the probability that response <Code>y_w</Code> is preferred over <Code>y_l</Code> is <Code>σ(r(x, y_w) - r(x, y_l))</Code>. Substitute the reward expression above into the difference:
      </Prose>

      <Prose>
        The <Code>β log Z(x)</Code> terms are identical for both responses — they depend only on the prompt, not the response — so they cancel. What remains is a preference probability that depends only on the ratio of policy probabilities to reference probabilities. The reward model has vanished from the expression entirely. You never needed to train one.
      </Prose>

      <H2>The DPO loss</H2>

      <Prose>
        Taking the negative log-likelihood of this preference probability over a dataset of pairs <Code>(x, y_w, y_l)</Code> gives the DPO training loss directly:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{DPO}(\\pi_\\theta; \\pi_{\\text{ref}}) = -\\mathbb{E}\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w | x)}{\\pi_{\\text{ref}}(y_w | x)} - \\beta \\log \\frac{\\pi_\\theta(y_l | x)}{\\pi_{\\text{ref}}(y_l | x)}\\right)\\right]"}</MathBlock>

      <Prose>
        Read the argument of the sigmoid as two implicit reward estimates. The first term, <Code>β log(π_θ(y_w|x) / π_ref(y_w|x))</Code>, measures how much the policy prefers the chosen response over the reference. The second term does the same for the rejected response. Minimizing this loss pushes the margin between the two in the right direction: it increases the implicit reward assigned to the chosen response and decreases it for the rejected one, proportionally to how far each currently deviates from the reference. One model, trained on supervised data, with a standard cross-entropy-shaped objective.
      </Prose>

      <Prose>
        There is no sampling during training, no value model, no clipping, no advantage estimation, no reward normalization. The reference model <Code>π_ref</Code> is frozen — you run a forward pass through it to get log-probabilities, but its weights do not update. In practice:
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """
    All inputs are sum-log-probabilities of the response under each model.
    Compare implicit rewards: beta * (policy_logp - ref_logp).
    """
    policy_reward_chosen = beta * (policy_logps_chosen - ref_logps_chosen)
    policy_reward_rejected = beta * (policy_logps_rejected - ref_logps_rejected)
    logits = policy_reward_chosen - policy_reward_rejected
    return -F.logsigmoid(logits).mean()`}
      </CodeBlock>

      <Prose>
        The entire alignment signal fits in eight lines. Computing <Code>policy_logps_chosen</Code> and <Code>ref_logps_chosen</Code> requires two forward passes per batch — one through the trainable policy and one through the frozen reference — but there are no rollouts, no replay buffers, and no separate reward-model inference calls. The training loop looks identical to standard supervised fine-tuning with an extra reference forward pass.
      </Prose>

      <H2>Why this is remarkable</H2>

      <Prose>
        The simplification is not just cosmetic. Three things changed at once when DPO appeared.
      </Prose>

      <Prose>
        First, operational complexity collapsed. PPO-based RLHF requires managing at least four model instantiations simultaneously — the policy being trained, the frozen reference copy, the reward model, and the value model — plus a sampling loop that generates completions at each step and feeds them back into the training gradient. DPO requires two: the trainable policy and the frozen reference. You can DPO-train with a standard Hugging Face <Code>Trainer</Code>, a single GPU node, and the same checkpointing infrastructure you would use for a fine-tuning run. The learning curve for practitioners dropped dramatically.
      </Prose>

      <Prose>
        Second, empirical quality held up. On standard alignment benchmarks — MT-Bench, AlpacaEval, Anthropic HH — DPO-trained models match or exceed PPO-based models when trained on comparable preference datasets and compute budgets. The cases where PPO wins tend to involve large-scale online data collection or difficult instruction-following tasks where the reward model's generalization capacity matters; for the common case of offline fine-tuning on a curated preference dataset, DPO is competitive. Third, and most subtly, DPO removes the reward model as an independent failure surface. PPO-based RLHF has a well-documented failure mode called reward hacking: the policy finds inputs that score highly on the neural-network proxy for human preference without actually being good. DPO has no explicit reward model to hack — the implicit reward is the policy's own log-probability ratio, which is directly tied to what the model outputs. That does not eliminate all modes of exploitation, but it closes the specific exploit of gaming a separate reward network.
      </Prose>

      <H2>The tradeoffs DPO actually makes</H2>

      <Prose>
        Simplicity comes with constraints. Three are worth understanding before deploying DPO.
      </Prose>

      <Prose>
        DPO is an off-policy method. It optimizes on a fixed dataset of preference pairs collected before training begins; the policy does not interact with the environment during training and cannot generate candidates that are not already in the dataset. RLHF's online sampling is a genuine structural advantage here: as the policy improves, PPO can ask for new comparisons on the stronger policy's own outputs, which tends to give better gradient signal on hard tasks and allows the model to discover responses that are better than any example in the original dataset. DPO cannot do this. If your preference dataset is narrow — covering only a fraction of the task distribution — DPO's improvement is bounded by the coverage of your data in a way that PPO's is not.
      </Prose>

      <Prose>
        The second constraint is reference sensitivity. DPO's loss measures improvement relative to <Code>π_ref</Code>. If the reference model is already well-calibrated and broadly capable, DPO has a meaningful signal gradient to climb. If the reference is a poorly-trained SFT model — one that systematically hallucinates, or that has been fine-tuned on a narrow domain — DPO's "improvement" is improvement relative to a bad anchor, and the resulting model may diverge from useful behavior in subtle ways that are hard to catch without comprehensive evaluation.
      </Prose>

      <Callout accent="gold">
        DPO is not "RLHF without the RL." It's offline, on fixed pairs — which is simpler and often enough, but gives up RLHF's exploration.
      </Callout>

      <Prose>
        The third constraint is the regularization strength. The <Code>β</Code> parameter controls how tightly the policy is anchored to the reference — high <Code>β</Code> stays close to <Code>π_ref</Code>, low <Code>β</Code> allows large deviations. In the limit of very small <Code>β</Code>, DPO can drift almost as far from the reference as PPO without KL penalty, which means the coherence and factuality properties of the base model can erode significantly. The implicit KL regularization is real but not as tightly controlled as the explicit KL term in the PPO objective. In practice, <Code>β</Code> needs to be tuned per dataset and per task, and the interaction between <Code>β</Code>, learning rate, and number of epochs is the main source of DPO training instability.
      </Prose>

      <H2>Practical recipe</H2>

      <Prose>
        The standard DPO workflow has four stages. Start from a model that has already been fine-tuned with supervised learning on instruction-response pairs — an SFT checkpoint. DPO on a base pretrained model without SFT first tends to produce unstable or incoherent results, because the base model's distribution is too diffuse to serve as a useful reference anchor.
      </Prose>

      <Prose>
        Gather or synthesize preference pairs. Each example is a triple: a prompt, a chosen response, and a rejected response. The chosen and rejected responses can come from human labelers (expensive, high-quality), from a stronger model acting as a judge (scalable, moderate quality), or from the SFT model itself with the chosen response taken from curated demonstrations and the rejected response sampled at temperature. Libraries like TRL's <Code>DPOTrainer</Code>, Axolotl, and Unsloth provide data formatting utilities and handle the reference model bookkeeping automatically.
      </Prose>

      <Prose>
        Train for 1–3 epochs with <Code>β ≈ 0.1</Code> and learning rate around <Code>5e-7</Code>. Use a small batch size — 8 to 32 examples per gradient step — because DPO gradients can be noisy at scale and the reference forward pass doubles your memory pressure. The primary diagnostic metric to track during training is the reward margin: <Code>policy_reward_chosen - policy_reward_rejected</Code>. A healthy run shows this margin growing steadily from near zero. If it plateaus early, the preference dataset likely has low signal or high noise. If it grows but validation quality degrades, <Code>β</Code> is probably too low and the policy is drifting.
      </Prose>

      <H3>Common failure modes</H3>

      <Prose>
        DPO will train to completion without errors even when it is learning the wrong thing, which makes silent failures more common than with PPO (where reward hacking tends to be obvious). The most pervasive failure mode is length bias. In most human preference datasets, chosen responses are longer than rejected responses — annotators tend to interpret more thorough answers as better ones, regardless of content. DPO internalizes this spurious correlation aggressively: models trained on such data reliably produce longer outputs than their SFT starting points, and the lengthening is not accompanied by proportional quality improvement. Several mitigation strategies exist — filtering pairs to match lengths, using SimPO's length-normalized reward, or explicitly including short-but-correct responses in the chosen set — but none fully eliminates the bias.
      </Prose>

      <Prose>
        A second failure mode is annotation idiosyncrasy. If the preference dataset was labeled by a small team or a single labeler, DPO will learn that labeler's particular style preferences as though they were ground truth. This shows up as unexpectedly strong style biases — overly formal tone, excessive hedging, specific structural conventions — that are hard to explain from the prompt distribution alone. Running DPO on noisily-labeled data amplifies whatever systematic error exists in the labels, because unlike RLHF, there is no reward model step that could be inspected and audited separately from the policy.
      </Prose>

      <Prose>
        Finally: if <Code>β</Code> is set too small relative to the number of training epochs, the policy can diverge from the reference faster than the preference signal can guide it, producing outputs that satisfy the loss function while losing fluency and factual grounding. Monitor perplexity on a held-out set of the SFT training data throughout DPO training. If it climbs sharply, stop and increase <Code>β</Code>.
      </Prose>

      <Prose>
        DPO is the current default for open-weight alignment precisely because it is easy to deploy correctly and hard to deploy catastrophically wrong. The deluge of follow-ups — SimPO, IPO, ORPO, KTO — are refinements on the same core move: preference data, closed-form policy, supervised loss. Each addresses a specific failure mode identified in vanilla DPO: SimPO targets length bias, IPO addresses the overconfidence of the Bradley-Terry assumption, ORPO folds the reference model away entirely, KTO handles unpaired feedback signals. The next topics in this section walk through those variants and the specific regimes where each beats the baseline.
      </Prose>
    </div>
  ),
};

export default dpo;
