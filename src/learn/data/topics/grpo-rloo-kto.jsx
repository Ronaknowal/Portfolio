import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const grpoRlooKto = {
  title: "GRPO, RLOO, KTO & Advanced Preference Methods",
  readTime: "13 min",
  content: () => (
    <div>
      <Prose>
        Classical RLHF — covered in the RLHF topic — trains a reward model from preference pairs, then runs PPO to maximize that reward while keeping the policy close to its supervised fine-tuning anchor. PPO carries a lot of machinery: a value network that must be updated alongside the policy, multiple optimization epochs per batch, careful clipping to prevent large updates, and a critic of the same scale as the actor to produce reliable advantage estimates. That machinery is not free. For a 70B language model, the value network is another 70B parameters sitting in GPU memory, updated at every step, used only as an internal scaffolding for variance reduction.
      </Prose>

      <Prose>
        Three methods released in 2024 each discard a different piece of that scaffolding. GRPO removes the value model by sampling groups of responses and using within-group reward statistics as the baseline. RLOO goes back to classical REINFORCE but adds a leave-one-out baseline that sharpens variance reduction without any learned critic. KTO removes the need for paired preferences entirely, training on binary thumbs-up/thumbs-down signals instead of head-to-head comparisons. Each trade-off is specific: you give something up, you get something in return, and the right choice depends on what your training pipeline can actually afford.
      </Prose>

      <H2>GRPO — Group Relative Policy Optimization</H2>

      <Prose>
        GRPO was introduced in DeepSeekMath (2024) and became the training algorithm behind DeepSeek-R1. The insight is simple: if you already need to sample multiple responses to estimate a reward signal — which you do, for variance reduction — you can use the reward statistics of those responses as a baseline instead of training a separate value network to provide one.
      </Prose>

      <Prose>
        For each prompt in the batch, sample <Code>G</Code> responses. Compute a reward for each response. The advantage for response <Code>i</Code> is its reward, normalized by the mean and standard deviation of the group:
      </Prose>

      <MathBlock>{"A_i = \\frac{r_i - \\text{mean}(\\{r_1, \\ldots, r_G\\})}{\\text{std}(\\{r_1, \\ldots, r_G\\})}"}</MathBlock>

      <Prose>
        This is group-relative scoring. A response with a reward of 0.8 looks very different depending on whether its siblings scored 0.2 or 0.9. The normalization centers the advantages around zero and scales them to a unit-variance-like range, which is roughly what a well-trained value network would produce — without any learned parameters. The policy gradient update then proceeds exactly as in PPO, with clipping and a KL penalty relative to the reference policy:
      </Prose>

      <CodeBlock language="python">
{`def grpo_step(policy, ref_policy, reward_fn, prompts, group_size=8, beta=0.04):
    # Sample G responses for each prompt.
    expanded = [p for p in prompts for _ in range(group_size)]
    responses, logps_old = policy.generate_with_logprobs(expanded)
    rewards = torch.tensor([reward_fn(p, r) for p, r in zip(expanded, responses)])

    # Group-normalized advantage — replaces PPO's value model entirely.
    rewards = rewards.view(len(prompts), group_size)
    advantages = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-8)
    advantages = advantages.flatten()

    # KL-regularized policy objective with PPO clipping.
    ref_logps = ref_policy.logprobs_of(expanded, responses)
    kl = logps_old - ref_logps
    for _ in range(ppo_epochs):
        new_logps = policy.logprobs_of(expanded, responses)
        ratio = (new_logps - logps_old).exp()
        policy_loss = -torch.min(
            ratio * advantages,
            ratio.clamp(1 - eps, 1 + eps) * advantages,
        ).mean() + beta * kl.mean()
        policy_loss.backward(); optimizer.step(); optimizer.zero_grad()`}
      </CodeBlock>

      <Prose>
        The win is substantial: eliminating the value model roughly halves peak GPU memory compared to PPO at the same model size, and removes the critic's training dynamics entirely — no value loss to balance, no separate learning rate to tune. The cost is forward-pass compute: you are generating <Code>G</Code> responses per prompt instead of one. In practice, <Code>G</Code> is set between 4 and 16. For tasks with fast verifiable rewards — math problems with a correct final answer, code that can be executed and checked — this trade-off is favorable. For tasks with slow or expensive reward evaluation, the cost of generating eight responses per prompt becomes the bottleneck. DeepSeek-R1's setting was math and reasoning, where rewards are cheap to compute and the value model was the bigger drag on throughput.
      </Prose>

      <H2>RLOO — Reinforce Leave-One-Out</H2>

      <Prose>
        Ahmadian et al. at Cohere (2024) took a step back from PPO in a different direction. Their argument: the entire apparatus of PPO — clipping, multiple epochs, the value network — was developed for control problems with dense reward signals and non-stationary dynamics. Language model fine-tuning has a much more benign structure. The reward function is fixed, the action space is discrete, and the policy only needs to move a small distance from a well-pretrained starting point. In that setting, classical REINFORCE — the simplest policy gradient algorithm — works, provided the baseline is good.
      </Prose>

      <Prose>
        The baseline RLOO uses is leave-one-out: for each response in a group of <Code>k</Code>, its baseline is the mean reward of the other <Code>k-1</Code> responses.
      </Prose>

      <MathBlock>{"A_i = r_i - \\frac{1}{k-1}\\sum_{j \\neq i} r_j"}</MathBlock>

      <Prose>
        The intuition is that each response evaluates itself against its siblings rather than against a learned critic. If the group as a whole got high rewards, the baseline is high, and a response with a slightly-above-average reward contributes only a small positive gradient signal. If the group all scored poorly, even a mediocre response gets a positive signal because it was relatively better. This is almost the same structure as GRPO — both use within-group comparisons — with the key difference that RLOO uses raw mean subtraction rather than normalizing by group standard deviation. RLOO also omits PPO's clipping and multiple epochs per batch: one gradient step per batch, no clipping, no critic, no separate value loss.
      </Prose>

      <Prose>
        The surprising empirical result from the paper: RLOO matches PPO on standard RLHF benchmarks — Alpaca Eval, MT-Bench, the usual preference alignment suites — at substantially lower implementation complexity and computational cost. The value model that PPO invests so heavily in turns out to be doing less than expected when the policy is already a strong pretrained model. For practitioners, RLOO's appeal is that it is the least code: no value network initialization, no critic learning rate schedule, no balancing of the policy loss against a value loss. If your goal is preference alignment on a standard RLHF benchmark and you don't need the exploration properties PPO offers on harder reasoning tasks, RLOO is often the right starting point.
      </Prose>

      <H2>KTO — Kahneman-Tversky Optimization</H2>

      <Prose>
        DPO (covered in the DPO topic) and SimPO both require paired preference data: for each prompt, you need a preferred response and a rejected response, labeled relative to each other. That pairing requirement is expensive. It means you need labelers to see two candidate responses side by side and choose between them, which is both slow and subject to significant inter-annotator disagreement. More practically, it means you cannot use the vast amounts of single-response feedback that production systems accumulate — the thumbs up and thumbs down that users click on individual outputs, with no comparison to an alternative.
      </Prose>

      <Prose>
        Ethayarajh et al. (2024) introduced KTO to work directly with binary feedback. The loss is grounded in prospect theory — specifically Kahneman and Tversky's observation that humans weight losses more heavily than equivalent gains. Desirable and undesirable responses are treated asymmetrically: the loss pushes the policy to increase its implicit reward on good responses and decrease it on bad ones, with a KL anchor to the reference model playing the same role it does in DPO.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def kto_loss(policy_logps, ref_logps, is_desirable,
             beta=0.1, lam_desirable=1.0, lam_undesirable=1.0):
    """
    Binary-feedback preference loss.
    is_desirable: boolean mask of whether each sample is labeled 'good' (True) or 'bad' (False).
    """
    # Implicit reward, KL-based like DPO.
    logratio = policy_logps - ref_logps
    # Kahneman-Tversky shape — sigmoid of margin, applied differently to desirable vs undesirable.
    # Desirable samples: reward above reference → loss low.
    desirable_loss = 1 - F.sigmoid(beta * (logratio - kl_estimate)).mean()
    undesirable_loss = 1 - F.sigmoid(beta * (kl_estimate - logratio)).mean()
    # Note: kl_estimate is a running estimate of expected KL across the batch.
    return lam_desirable * desirable_loss + lam_undesirable * undesirable_loss`}
      </CodeBlock>

      <Prose>
        The structure above captures the key idea: the loss for a desirable sample is low when the policy's log-ratio over the reference is high (meaning the policy has moved toward that response relative to its starting point), and the loss for an undesirable sample is low when the log-ratio is low. The <Code>kl_estimate</Code> term — a running batch average of log-ratios — provides the centering that paired comparisons would otherwise supply. The lambda weights let you apply asymmetric pressure: empirically, weighting undesirable samples more heavily tends to work better in practice, consistent with the loss-aversion framing.
      </Prose>

      <Prose>
        The practical appeal is the data model it enables. Thumbs-up and thumbs-down signals from production logs are directly usable training data. You do not need a labeling pipeline that collects comparisons; you need a logging pipeline that records which responses users found helpful or unhelpful. That data is vastly cheaper to collect, meaning the dataset can be orders of magnitude larger than a paired preference dataset collected at the same cost. The trade-off is signal quality: a thumbs-down on a response tells you less than a head-to-head comparison that the response lost against a specific alternative. But at sufficient dataset scale, quantity compensates for individual signal noisiness.
      </Prose>

      <H2>When to use which</H2>

      <Prose>
        These methods are not interchangeable — each occupies a different point in a space defined by what you can sample, what feedback you can collect, and how much memory you have.
      </Prose>

      <Prose>
        Use <strong>GRPO</strong> when you want PPO's on-policy exploration but cannot afford the value model. It is the natural choice for reasoning-heavy tasks with verifiable rewards — math, code correctness, structured output validation — where generating multiple responses per prompt is acceptable and rewards are cheap to compute. DeepSeek-R1's success validated this setting.
      </Prose>

      <Prose>
        Use <strong>RLOO</strong> when sampling is expensive and you want the simplest algorithm that matches PPO's empirical performance. On standard preference alignment benchmarks — helpfulness, harmlessness, the typical RLHF targets — RLOO performs comparably to PPO at much lower complexity. It is a strong default when you already have a reward model and want to do on-policy RL without the value network overhead.
      </Prose>

      <Prose>
        Use <strong>KTO</strong> when your preference data is binary rather than paired. Production logs with user feedback, A/B test results, human ratings on individual responses — all of this becomes usable. The data collection bar is low enough that it often enables datasets an order of magnitude larger than what paired comparison labeling would allow at the same budget.
      </Prose>

      <Prose>
        Use <strong>DPO or SimPO</strong> — covered in their respective topics — when you already have clean preference pairs and want the simplest possible supervised-style loss with no RL dynamics. If your dataset is already paired and you don't need on-policy exploration, the added complexity of GRPO or RLOO is hard to justify.
      </Prose>

      <H3>The convergence picture</H3>

      <Prose>
        Stepping back across RLHF, DPO, SimPO, GRPO, RLOO, and KTO, a structural pattern becomes visible. Every one of these methods is doing the same thing at a high level: it is adjusting the policy's probability distribution in a direction indicated by a preference signal, subject to a KL constraint that prevents the policy from drifting too far from its SFT anchor. The loss functions look different. The sampling strategies look different. The required data formats look different. But the optimization target is the same: move toward responses the signal says are good, stay close to the reference model, and balance the two with a temperature-like beta parameter.
      </Prose>

      <Callout accent="gold">
        The specific loss matters less than it looks. What matters more: the quality of your preference data and the KL budget you give the policy to drift from its SFT anchor.
      </Callout>

      <Prose>
        The KL budget is the parameter most practitioners underweight. A very small beta keeps the policy tightly constrained to the SFT checkpoint — it will be safe and coherent, but it will not move far enough to exhibit new behaviors. A large beta lets the policy drift — it may develop stronger capabilities on the target task but will start generating text that feels distant from the SFT distribution in ways that are hard to predict. Most papers use beta between 0.01 and 0.1; the right value is more sensitive to task structure than to choice of algorithm. Getting the preference data right and setting beta appropriately will improve results more reliably than switching between any of the methods above.
      </Prose>

      <Prose>
        Three more algorithms down, and the preference optimization landscape is largely covered. The next topic on Constitutional AI approaches the problem from a different angle entirely: rather than collecting human preferences and training on them, it asks what happens when the preference signal itself is generated by a model, using a set of principles rather than human comparisons as the source of reward. The data pipeline changes; the underlying optimization is the same.
      </Prose>
    </div>
  ),
};

export default grpoRlooKto;
