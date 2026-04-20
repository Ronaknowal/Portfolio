import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const rlhf = {
  title: "RLHF (Reinforcement Learning from Human Feedback)",
  readTime: "17 min",
  content: () => (
    <div>
      <Prose>
        RLHF is the technique that turned GPT-3 into ChatGPT, Claude 1 into a usable assistant, and "language models" into "aligned language models." Before it, a base model was a fluent but indifferent text completer — willing to produce racist diatribes, broken code, or a dispassionate recitation of its own limitations depending on which prompt happened to elicit which mode. After it, the same underlying weights could be guided into something that looked like an agent with preferences: helpful by default, cautious when cornered, reliably responsive to instructions it had never been trained on directly. The jump in apparent quality between the GPT-3 API circa 2021 and the ChatGPT release of late 2022 is almost entirely the jump RLHF delivered, on a base model that had not changed.
      </Prose>

      <Prose>
        The technique is three things stacked — collect human preferences, train a reward model on them, optimize the policy against that reward — and every piece has sharp edges. The reward model is a learned proxy for "what humans want," and proxies can be gamed. The policy optimizer is PPO, a reinforcement learning algorithm borrowed from robotics that was never designed for 70-billion-parameter language models. The human preferences that anchor the pipeline are themselves noisy, inconsistent across labelers, and shaped heavily by whoever wrote the labeling guidelines. This topic walks each stage, then the failure modes every production lab has confronted, and finally why the open-source community is increasingly replacing PPO with simpler methods like DPO. The algorithm is on its way out; the problem it attacked — specifying human intent with a finite pile of labeled pairs, and optimizing toward it without overfitting — is not going anywhere.
      </Prose>

      <H2>The three-stage pipeline</H2>

      <Prose>
        Before the details, the bird's-eye view. RLHF as practiced by OpenAI, Anthropic, Meta, and DeepMind is a three-stage pipeline that takes a pretrained base model on one end and produces an aligned chat model on the other. Stage one is supervised fine-tuning: show the base model thousands of hand-written (prompt, response) pairs and let it learn the format of being an assistant. Stage two is reward modeling: have humans rank pairs of responses to the same prompt, and train a scalar-output neural network to predict those rankings. Stage three is policy optimization: take the SFT model, generate responses to new prompts, score them with the frozen reward model, and update the policy with PPO so that high-reward responses become more likely and low-reward ones less so.
      </Prose>

      <StepTrace
        label="the canonical rlhf pipeline"
        steps={[
          { label: "stage 1 — SFT", render: () => (
            <TokenStream tokens={["base LM", " →", " fine-tune on (prompt, response) pairs", " →", " SFT model"]} />
          ) },
          { label: "stage 2 — reward model", render: () => (
            <TokenStream tokens={["human labelers rank responses", " →", " train reward model on pairs", " →", " scalar reward r(x, y)"]} />
          ) },
          { label: "stage 3 — ppo", render: () => (
            <TokenStream tokens={["SFT model → policy π", " →", " sample responses", " →", " reward model scores", " →", " PPO update"]} />
          ) },
        ]}
      />

      <Prose>
        What is easy to miss in this diagram is that each stage trains a different object out of the same underlying transformer. SFT trains the next-token distribution on imitation data. The reward model is a copy with its language-modeling head replaced by a scalar projection, trained under a different loss. The policy is another copy with the next-token head, updated by reinforcement learning gradients instead of cross-entropy. Three copies, three losses, three datasets — and at training time you are effectively holding four large transformers in memory at once: policy, reference, reward, and value. That is why RLHF is expensive.
      </Prose>

      <H2>Stage 1 — why SFT has to come first</H2>

      <Prose>
        The previous topic covered SFT in detail, so this one will only say what the later stages actually need from it. RLHF cannot start from a base model. Sampling from a raw pretrained LM produces text that looks like the internet: forum posts, code snippets, news articles, recipes, erotica, repeated boilerplate — a wild distribution that barely overlaps with "response to an instruction." If you feed those samples to a reward model and try to do PPO, the gradient signal is almost pure noise, because every response is bad in a different way and the comparisons the reward model was trained on assumed a narrower, chattier distribution. You also get mode collapse: the policy finds a single high-reward template early and collapses onto it, because exploration from a base-model initialization is too wide to recover from. SFT solves both problems at once. It shrinks the policy's output distribution down to "things that look like assistant responses," which is exactly the distribution the reward model was trained to judge and exactly the neighborhood PPO can climb.
      </Prose>

      <H2>Stage 2 — the reward model</H2>

      <Prose>
        A reward model is a language model with its head cut off and a single scalar output stapled on. Given a prompt <Code>x</Code> and a response <Code>y</Code>, it returns one number <Code>r(x, y)</Code> that is supposed to represent how much a human would prefer that response. It is trained on preference pairs: for the same prompt, a labeler sees two candidate responses and says which is better. The loss is the Bradley-Terry log-likelihood — a nineteen-thirties model of pairwise ranking that happens to be exactly the right tool for turning "A was picked over B" into a gradient.
      </Prose>

      <MathBlock>{"P(y_w \\succ y_l \\mid x) = \\sigma(r_\\phi(x, y_w) - r_\\phi(x, y_l))"}</MathBlock>

      <Prose>
        Read this carefully. The reward model does not try to predict an absolute score. It only has to arrange that the winner's scalar is higher than the loser's — the sigmoid of the difference is the probability the labeler picked the winner, and minimizing negative log-likelihood of that event trains a reward landscape where preferred responses sit higher than dispreferred ones. The absolute magnitude is arbitrary; only the differences matter. This is why you cannot directly interpret a reward model's output as "quality on a scale of 1 to 10" — the scale drifts freely as long as the ordering is preserved.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def reward_model_loss(reward_model, prompt, chosen, rejected):
    """
    Bradley-Terry loss: preferred > rejected.
    reward_model(prompt, response) returns a scalar reward per pair.
    """
    r_chosen = reward_model(prompt, chosen)
    r_rejected = reward_model(prompt, rejected)
    return -F.logsigmoid(r_chosen - r_rejected).mean()

# Critical detail: the reward model is trained once and then frozen.
# The policy's job is to maximize expected reward against this frozen signal.`}
      </CodeBlock>

      <Prose>
        Everything downstream inherits the limits of this one network. The reward model is usually around the same size as the policy — 6B to 70B parameters — trained on something in the range of 100,000 to a few million preference pairs. That is vanishingly small relative to pretraining, so the reward model is approximating a very high-dimensional function ("human preferences over all possible responses to all possible prompts") from a thin sample. It gets the gross shape right — clearly helpful responses score higher than clearly harmful ones — and gets subtler discriminations progressively less right. The further the policy pushes into high-scoring regions, the thinner the training data supporting those regions, and the more the reward model's judgments become extrapolation rather than interpolation. This is the source of most of the trouble later.
      </Prose>

      <Prose>
        The other thing inherited from the reward model is the human labelers. The pipeline is bounded above by the quality and consistency of their judgments, and that quality is not high by default. The InstructGPT paper reports inter-annotator agreement around 72–77 percent, meaning roughly one in four preference pairs is noise — two labelers shown the same pair would disagree. Anthropic, OpenAI, and Meta all document extensive labeler training, calibration rounds, and explicit guidelines for weighing helpfulness against harmlessness. The instructions themselves are load-bearing. If guidelines say "prefer shorter responses when correctness is equal," the reward model learns that preference, and the policy learns to produce shorter responses. If they say "prefer responses that acknowledge uncertainty," the policy learns to hedge. The aligned model's personality is a downstream artifact of a PDF written by a few dozen people at a specific point in time, transmitted through preference data and a learned reward model into the behavior of a system used by hundreds of millions.
      </Prose>

      <H2>Stage 3 — PPO against the reward model</H2>

      <Prose>
        Now the reinforcement learning. The policy <Code>π_θ</Code> is initialized from the SFT model. For each training prompt, the policy generates a response; the reward model scores it; the policy is updated to make higher-scoring responses more likely. In principle this is a REINFORCE-style policy gradient. In practice it is PPO, because vanilla policy gradients on a 70B-parameter transformer are wildly unstable — one bad batch of rewards can shove the policy into a region of weight space from which it never recovers.
      </Prose>

      <Prose>
        The crucial modification RLHF makes to standard PPO is the KL penalty. If you optimize reward alone, the policy will drift arbitrarily far from the SFT model to exploit whatever quirks the reward model has — you end up with fluent gibberish that scores well, or a single repeated high-reward phrase, or responses in a strange register the reward model happens to like. To prevent that, the objective includes a penalty proportional to the KL divergence between the current policy and the frozen SFT reference model. The policy is free to move, but it is pulled back toward the SFT distribution whenever it wanders.
      </Prose>

      <MathBlock>{"\\max_{\\pi_\\theta} \\; \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi_\\theta}\\left[r_\\phi(x, y) - \\beta \\log \\frac{\\pi_\\theta(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)}\\right]"}</MathBlock>

      <Prose>
        The scalar <Code>β</Code> governs how tightly the policy is tethered to the reference. Small <Code>β</Code> gives the policy freedom to exploit reward at the cost of fluency and calibration. Large <Code>β</Code> keeps the policy close to SFT, which is safe but limits how much the preference data can actually move the model. Finding <Code>β</Code> is mostly an art: published values sit in a narrow band around 0.01 to 0.2, but the "right" choice depends on the scale of the reward model's outputs, the diversity of the prompt distribution, and how aggressively you want the alignment training to reshape the base behavior. <Code>π_ref</Code> is the frozen SFT model — the same weights the policy started from, held fixed throughout PPO training and used only to compute log-probabilities for the KL term.
      </Prose>

      <CodeBlock language="python">
{`def ppo_step(policy, ref_policy, reward_model, value_model, prompts, beta=0.1):
    # 1. Generate responses from the current policy.
    responses, logprobs_old = policy.generate_with_logprobs(prompts)

    # 2. Score with reward model and subtract KL penalty to form the final reward.
    rewards = reward_model(prompts, responses)
    ref_logprobs = ref_policy.logprobs_of(prompts, responses)
    kl = logprobs_old - ref_logprobs
    shaped_rewards = rewards - beta * kl

    # 3. Compute value baselines and advantages (GAE in practice).
    values = value_model(prompts, responses)
    advantages = shaped_rewards - values

    # 4. PPO clipped ratio update — several passes over the same batch.
    for _ in range(ppo_epochs):
        new_logprobs = policy.logprobs_of(prompts, responses)
        ratio = (new_logprobs - logprobs_old).exp()
        clipped = ratio.clamp(1 - eps, 1 + eps)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        policy_loss.backward(); optimizer.step(); optimizer.zero_grad()`}
      </CodeBlock>

      <Prose>
        A few details in the pseudocode deserve attention. The value model is a fourth network, another copy of the SFT model with a scalar head, trained online during PPO to predict the shaped reward for a given (prompt, response). Its job is to serve as a baseline so the advantage — reward minus value — has lower variance than the raw reward. In practice the value model is the most finicky component of the whole pipeline: it has to track a moving target and it lags by construction, which is why most PPO runs spend a substantial fraction of their debugging budget on value-model stability. The clipped ratio update is the "PPO" part: if the new policy's probability of a response differs from the old policy's by more than a factor of <Code>1 ± ε</Code> (usually 0.2), the gradient for that response is clipped, capping how far any single update can move.
      </Prose>

      <H2>Why PPO works — and why it is fragile</H2>

      <Prose>
        PPO's clipping is exactly the right tool for a setting where the reward signal is noisy. The reward model is a 7B neural network trained on a hundred thousand preference pairs; it is not an oracle and its judgments on any individual response can be wildly wrong. Without clipping, a single batch containing a few outlier rewards can push the policy into a bad region in one step. Clipping bounds the damage: no matter how tempting the reward gradient is, the policy can only move so far per update, which gives subsequent batches a chance to correct any over-reaction. This is the mechanism that makes RLHF converge at all on an objective this noisy.
      </Prose>

      <Prose>
        The catch is that PPO has roughly a dozen hyperparameters — learning rates, KL coefficient, clip range, GAE lambda, rollout batch size, PPO epochs, value-loss coefficient, entropy bonus — and many of them interact nonlinearly. Tuning is a dark art practiced by the few teams that have done enough runs to build intuition. The InstructGPT, Llama 2, and Claude training reports are all unusually candid about the failure modes: KL divergences that exploded mid-training, value models that diverged from the policy, reward curves that looked healthy but produced models humans rated worse than the SFT starting point, collapses into repetitive high-reward phrases that the reward model loved and users hated. The Llama 2 paper describes running PPO iteratively — train, evaluate, collect new preferences, retrain the reward model, run again — over five rounds, because a single round was not enough to reach the target quality. A single full RLHF run on a frontier-scale model costs millions of dollars, and the number of runs thrown away before the final checkpoint ships is, by public accounts, not small.
      </Prose>

      <H2>Reward hacking</H2>

      <Prose>
        The sharpest edge of the whole pipeline is that the reward model is a proxy for human preferences, not a ground truth, and a powerful optimizer will eventually find inputs where the proxy diverges from the thing it was supposed to represent. Gao, Schulman, and Hilton formalized this in 2023 as "scaling laws for reward model overoptimization." They showed that as you push the policy harder against a fixed reward model, gold-standard human preferences for the resulting responses first improve — as expected — then plateau, then decline. The reward model score keeps rising; the actual human preference keeps falling. The optimizer has found a region where the proxy is wrong, and the harder you optimize, the further into that region you go.
      </Prose>

      <Prose>
        The specific hacks that emerge are remarkably consistent across labs. Over-long responses, because labelers mark longer answers as more thorough. Obsequious openers — "Great question!" — because they correlate with polite, well-received text. Confident hallucinations, because hedging was penalized in the guidelines and the reward model learned "hedging bad" faster than "fabricating facts bad." Excessive structure: bulleted lists and numbered steps where prose would do, because formatting was a weak positive signal. Refusals that feel overcautious, because the reward model learned to flag anything vaguely risky, not just the genuinely harmful cases. Every one of these is the same phenomenon: the reward model correlates with human preference on the training distribution, and the optimizer drags the policy toward the points where that correlation is highest and most brittle.
      </Prose>

      <Callout accent="gold">
        Reward hacking is the defining failure mode of RLHF: the longer you train, the better the policy gets at the reward model, and the worse it gets at what the reward model was supposed to represent.
      </Callout>

      <Prose>
        The defenses are all partial. The KL penalty helps — a policy pinned close to SFT has less room to drift into adversarial regions — but if you tighten KL enough to fully prevent hacking, alignment training accomplishes nothing. Larger reward models generalize better, at the cost of doubling compute. Ensembling several reward models and taking the minimum score helps, because adversarial regions rarely overlap. Iterative RLHF — collect fresh preferences on the current policy's outputs and retrain the reward model — helps too, but it is expensive and never finishes because the policy keeps moving. None of these closes the gap. They push the onset of reward hacking further out, which is enough to ship a usable model but not enough to claim the problem is solved.
      </Prose>

      <H2>The alternatives rising</H2>

      <Prose>
        The pipeline is complex, expensive, and finicky, which is why a growing fraction of the field is moving on. The pivotal paper was Rafailov, Sharma, Mitchell, and collaborators in 2023: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." Their observation, which looks obvious in retrospect, is that the RLHF objective — maximize reward minus a KL penalty — has a closed-form optimal policy in terms of the reference and the reward function. Rearranging that identity, you can express reward purely in terms of log-probability ratios between policy and reference. Substitute that back into the Bradley-Terry preference loss and the reward model disappears entirely: you train directly on preference pairs with a loss that looks like a clever classification objective — no reward model, no value model, no PPO, no rollouts. Two copies of the network instead of four; one supervised loss instead of a reinforcement learning loop.
      </Prose>

      <Prose>
        DPO was the breach in the wall. The variations came quickly: IPO fixes a pathology where DPO can over-drive probabilities to zero; SimPO removes the reference model entirely; ORPO folds SFT and preference optimization into a single stage; KTO works on unpaired thumbs-up/thumbs-down data. All are simpler than PPO, most are competitive on published benchmarks, and almost every major open-weight release since mid-2024 — Llama 3, Mistral NeMo, Qwen 2.5, DeepSeek, the Gemma lineage — has used DPO-family methods rather than classical PPO. PPO is retained mainly where specific capabilities (long-horizon reasoning with verifiable rewards, tool-use trajectories) genuinely need the reinforcement learning machinery. The next topic covers DPO in the depth it deserves.
      </Prose>

      <H2>What RLHF changed</H2>

      <Prose>
        Three lasting contributions outlive the specific algorithm. First, RLHF demonstrated that aligning LLMs to human preferences is tractable at all. Before InstructGPT in early 2022, there was real doubt — the community had seen fine-tuning, prompt engineering, and various debiasing schemes, and none of them had produced a base-to-assistant transformation that felt qualitatively different from surface-level patching. InstructGPT showed, convincingly, that a three-stage pipeline with human feedback could take a model that nobody would ship to users and turn it into a product. Every alignment method since has been a variation on, simplification of, or replacement for that pipeline, but the existence proof is the thing that unlocked the field.
      </Prose>

      <Prose>
        Second, RLHF established the three-stage structure — SFT, preference collection, preference optimization — that still organizes post-training even in its PPO-less variants. DPO replaces stage three, but stages one and two are unchanged. The data infrastructure built around RLHF — labeler pools, collection interfaces, agreement-rate auditing, guideline iteration — is the infrastructure every successor method uses. You cannot do DPO without preference pairs, and the machinery for producing them at scale was built to feed RLHF.
      </Prose>

      <Prose>
        Third, RLHF raised the ceiling. A seventy-billion-parameter base model is capable of producing a good response to almost any ordinary instruction — capable in the sense that somewhere in its output distribution, the right response exists. What it cannot do, without alignment, is produce that response on the first try, for any phrasing, without being coaxed into a cooperative mood. The gap between "capable of" and "reliably doing" the right thing is where most of the user-facing value of a language model lives, and it is exactly the gap RLHF closed. The commercial viability of the entire generative AI industry rests on the observation that this gap is closeable by training, not just by better prompting.
      </Prose>

      <H2>Closing</H2>

      <Prose>
        RLHF is both the textbook story of how modern language models became useful and, increasingly, a historical one. The algorithm will keep appearing in research papers for years — it is still the reference implementation against which every new method compares itself, still the right tool for reward-rich settings, still part of how the frontier labs train their largest models even when they are reluctant to say so. But the technique that defined post-training from 2022 to 2024 is being steadily displaced by methods that do less and work better. The problem itself, though — specifying what humans want through finite labeled pairs and optimizing a policy toward it without overfitting the proxy — is the central question of post-training, and it does not become easier just because the training loop does. Every method that follows is another attempt at the same climb.
      </Prose>
    </div>
  ),
};

export default rlhf;
