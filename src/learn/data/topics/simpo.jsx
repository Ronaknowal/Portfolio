import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const simpo = {
  title: "SimPO (Simple Preference Optimization)",
  readTime: "9 min",
  content: () => (
    <div>
      <Prose>
        DPO replaced RLHF's explicit reward model with the policy itself — a cleaner formulation, covered in the previous topic. SimPO (Meng et al., 2024) goes one step further: it drops the reference model too. The insight is that DPO's reference-model term is there for mathematical elegance, not because training fundamentally requires it. A carefully normalized loss over only the current policy achieves comparable or better results with less memory, simpler code, and one fewer checkpoint to manage.
      </Prose>

      <H2>What DPO needs the reference for</H2>

      <Prose>
        DPO's loss compares log-probability ratios: how much more (or less) likely is each response under the current policy relative to a frozen reference copy of the same model? That ratio stabilizes training — without it, the policy could drive chosen-response log-probs arbitrarily high without bound. The reference acts as a KL-style anchor, keeping the policy from straying too far from its SFT starting point.
      </Prose>

      <Prose>
        The cost is real. The reference model sits in GPU memory alongside the trainable policy for the entire training run. For a 70B-parameter model, that roughly doubles the memory footprint. You need two copies loaded at once — one frozen, one updated — and two forward passes per batch. For large models, this is the single biggest infrastructure constraint of the DPO training loop. SimPO asks whether the anchor is genuinely necessary, or whether the same stability can be bought more cheaply.
      </Prose>

      <H2>The SimPO loss</H2>

      <MathBlock>{"\\mathcal{L}_{SimPO} = -\\mathbb{E}\\left[\\log \\sigma\\left(\\frac{\\beta}{|y_w|}\\log \\pi_\\theta(y_w|x) - \\frac{\\beta}{|y_l|}\\log \\pi_\\theta(y_l|x) - \\gamma\\right)\\right]"}</MathBlock>

      <Prose>
        Read it left to right. For each preference pair, compute the chosen response's total log-probability under the current policy, divide by its length to get a per-token average, and scale by <Code>β</Code>. Do the same for the rejected response. The model is rewarded when the difference between these two quantities exceeds the margin <Code>γ</Code>. No <Code>π_ref</Code> appears anywhere.
      </Prose>

      <Prose>
        The two substitutions — length normalization and a fixed margin — together do the work the reference model was doing. Length normalization prevents long responses from dominating the gradient simply by having more tokens to sum over. The margin <Code>γ</Code> sets a minimum separation the model must achieve before the loss saturates, functioning as a soft regularizer on how confident the policy needs to become. Neither requires a second model in memory.
      </Prose>

      <H2>Why length normalization matters</H2>

      <Prose>
        DPO has a well-documented length bias. When a chosen response is longer than the rejected one, its summed log-probability is a larger number in absolute value — even if the per-token quality is identical. The DPO loss treats that raw sum as the signal, so the gradient pushes the model to produce longer chosen responses. Over many gradient steps this compounds into visible verbosity: models trained with DPO reliably produce longer outputs, and evaluators trained to prefer longer completions reward them for it. The bias is subtle enough that it does not obviously show up as a training pathology, but it shows up clearly in deployment.
      </Prose>

      <Prose>
        Dividing by <Code>|y|</Code> converts the objective from "sum of log-probs" to "average log-prob per token." The loss no longer has a lever for verbosity: a longer response that fills space with filler tokens earns a lower per-token average, not a higher one. Empirically, models trained with SimPO produce outputs whose length distribution is much closer to the SFT baseline, without the upward drift that DPO introduces.
      </Prose>

      <H3>Implementation</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def simpo_loss(policy_logps_chosen, policy_logps_rejected,
               chosen_lens, rejected_lens,
               beta=2.0, gamma=0.5):
    """
    Length-normalized implicit rewards, margin gamma.
    policy_logps_*: summed log-probabilities of each response under the policy.
    chosen_lens, rejected_lens: response lengths in tokens.
    """
    r_chosen = beta * policy_logps_chosen / chosen_lens
    r_rejected = beta * policy_logps_rejected / rejected_lens
    return -F.logsigmoid(r_chosen - r_rejected - gamma).mean()`}
      </CodeBlock>

      <Prose>
        The implementation is intentionally minimal. There is no reference model forward pass, no log-ratio subtraction, and no KL term. The entire training loop reduces to: run one forward pass through the policy on both chosen and rejected responses, collect summed log-probs, normalize by length, compute the margin, and apply the sigmoid loss. On a 7B model, removing the reference forward pass cuts peak memory by roughly 40% and wall-clock time per step by roughly 30%, numbers that scale up at larger model sizes.
      </Prose>

      <H2>Empirical results</H2>

      <Prose>
        Meng et al. evaluated SimPO against DPO and several of its variants on MT-Bench, AlpacaEval 2, and Arena-Hard. Across the board, SimPO matched or exceeded DPO's win rates while using a single model in the training loop. The length-control result was particularly consistent: SimPO outputs were shorter and more concise than DPO outputs, yet scored higher on human-preference metrics — evidence that the DPO length bias was genuinely inflating perceived quality, not producing better responses.
      </Prose>

      <Prose>
        Subsequent analyses have complicated the picture somewhat. SimPO is more sensitive to <Code>β</Code> and <Code>γ</Code> than DPO is to its single <Code>β</Code> hyperparameter. DPO's reference model absorbs some of the hyperparameter sensitivity: it anchors the loss in a way that makes the scale of log-prob differences roughly consistent across model families. Without that anchor, the magnitude of the policy's log-probs depends on the SFT model, the tokenizer, and the response length distribution in the preference dataset, all of which vary. The sweet spots reported across multiple fine-tuning efforts cluster around <Code>γ ≈ 0.3–0.5</Code> and <Code>β ≈ 2.0–2.5</Code>, but those ranges are not universal and benefit from a small hyperparameter sweep on a held-out preference validation set.
      </Prose>

      <H3>What is really being traded</H3>

      <Prose>
        SimPO's removal of the reference model is not free. The reference acts as an implicit anchor on the policy's trajectory through parameter space — without it, the policy can drift more aggressively from its SFT initialization. The margin <Code>γ</Code> partially compensates, but it provides a fixed lower bound on the reward gap rather than a KL-style penalty proportional to how far the policy has moved. A model trained with a bad SFT starting point, or trained for too many steps, or trained with a too-large learning rate, has fewer implicit guards in SimPO than it does in DPO. The practical implication is that SimPO still requires a high-quality SFT checkpoint and conservative learning rates — it reduces infrastructure cost, not the cost of doing the surrounding work carefully.
      </Prose>

      <Callout accent="gold">
        Each successor of DPO — SimPO, IPO, ORPO, KTO — removes or replaces one assumption. They're cheaper; they're also more sensitive. The "right" method is a function of how much you trust your preference data and your SFT starting point.
      </Callout>

      <Prose>
        SimPO is an example of a broader trend: post-training methods are getting simpler, cheaper, and more dataset-dependent. The same move — remove the reference, substitute implicit regularization — shows up in several of the methods covered next. Whether that trade is worthwhile depends on what you have: if your SFT model is strong and your preference data is clean, SimPO's memory and compute savings are real gains. If either input is noisy, the reference model's stabilizing role becomes harder to give up.
      </Prose>
    </div>
  ),
};

export default simpo;
