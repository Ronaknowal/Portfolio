import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const dapo = {
  title: "DAPO (Dynamic Adaptive Policy Optimization)",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        DAPO, released by ByteDance Seed in 2025, is not a new algorithm. It is a set of
        targeted repairs to GRPO — the value-model-free policy gradient method that underlies
        much of the open-source reasoning work that followed DeepSeek-R1. The repairs are
        motivated by a specific setting: long-horizon RL on verifiable reasoning tasks, where
        rollouts routinely stretch to thousands of tokens and the assumptions baked into
        standard PPO-style clipping start to degrade silently. DAPO-trained open models reached
        competitive performance on AIME 2024 with a recipe that other labs could reproduce, and
        that reproducibility is part of the point — the contribution is as much documentation
        as it is technique.
      </Prose>

      <H2>What goes wrong in long-rollout RL</H2>

      <Prose>
        Three distinct failure modes accumulate as rollout length grows. The first is entropy
        collapse. As training proceeds on a fixed problem distribution, the policy concentrates
        probability mass on a narrow set of generation patterns. Exploration vanishes. The model
        finds a strategy that scores reliably and stops exploring alternatives — which means
        subsequent RL updates have nothing new to learn from. Rollout diversity, which is what
        makes group-normalized advantage estimates meaningful, slowly disappears.
      </Prose>

      <Prose>
        The second failure mode is length exploitation. GRPO computes advantage by normalizing
        rewards within a group of rollouts for the same prompt. When advantage is nonzero,
        every token in the rollout gets its log-probability adjusted proportionally. A correct
        answer of length 8,000 tokens accumulates a larger total gradient signal than a correct
        answer of 800 tokens — the per-token advantage is the same, but there are ten times as
        many tokens pulling probability up. The model learns that longer responses are
        mechanically more likely to receive larger gradient updates in the correct direction,
        and verbosity becomes a training artifact rather than a learned reasoning style. The
        third failure mode follows directly from length: importance ratios computed at the token
        level can swing far from 1.0 for tokens late in a long chain, particularly early in
        training when the policy is still moving. Standard symmetric clipping addresses this
        imperfectly, and the instability compounds over updates.
      </Prose>

      <H2>Clip-higher — asymmetric PPO clipping</H2>

      <Prose>
        Standard PPO clips the importance ratio <Code>r = π_θ(a|s) / π_ref(a|s)</Code> to a
        symmetric band around 1: tokens whose probability has risen too much or fallen too much
        relative to the reference policy are clipped equally in both directions. The implicit
        assumption is that over-amplifying probability increases and over-amplifying probability
        decreases are equally harmful. On verifiable reasoning tasks, they are not.
      </Prose>

      <Prose>
        Finding a correct reasoning path is rare. When the model stumbles onto a valid chain of
        steps, the training signal should be able to reinforce it substantially — because that
        path may not reappear for thousands of prompts, and its advantage is real. Suppressing
        wrong paths matters, but wrong paths appear constantly; the policy will have many
        opportunities to learn from them. Symmetric clipping penalizes the "increase probability
        of this rare correct path" direction just as aggressively as the "decrease probability
        of this wrong path" direction, which is the wrong tradeoff. DAPO's clip-higher relaxes
        the upper bound:
      </Prose>

      <MathBlock>{"\\text{clip}_{\\text{DAPO}}(r) = \\text{clip}(r,\\; 1 - \\varepsilon_{\\text{low}},\\; 1 + \\varepsilon_{\\text{high}}), \\quad \\varepsilon_{\\text{high}} > \\varepsilon_{\\text{low}}"}</MathBlock>

      <Prose>
        Typical values are <Code>ε_low = 0.2</Code> and <Code>ε_high = 0.28</Code>. The
        asymmetry is small in absolute terms. The effect on maintaining exploration is
        disproportionately large, because entropy collapse is driven primarily by the ceiling
        on how aggressively a correct path can be reinforced — clip-higher raises that ceiling
        without destabilizing the downward direction where the policy is already well-
        constrained by the abundance of wrong-path data.
      </Prose>

      <H2>Dynamic sampling — drop uninformative rollouts</H2>

      <Prose>
        GRPO's group-normalized advantage estimate has a degenerate case that is not
        hypothetical: when every rollout in a group receives the same reward, advantage is
        identically zero for all of them. The group contributes no gradient. The compute spent
        generating and scoring those rollouts is entirely wasted. This occurs frequently on
        hard problems where the model solves none of the rollouts correctly, and equally on
        easy problems where it solves all of them correctly. In both cases, the signal-to-
        compute ratio collapses.
      </Prose>

      <Prose>
        DAPO's dynamic sampling fix is simple: before accepting a group into the batch, check
        that its reward distribution has nonzero variance. If it does not, resample up to a
        fixed number of times. If it still does not, drop the prompt from this batch entirely.
      </Prose>

      <CodeBlock language="python">
{`def dapo_sample(policy, prompt, reward_fn, group_size=8, max_resamples=3):
    """Generate a group that has non-zero reward variance, or fail after N retries."""
    for attempt in range(max_resamples):
        rollouts = [policy.generate(prompt) for _ in range(group_size)]
        rewards = torch.tensor([reward_fn(prompt, r) for r in rollouts])
        if rewards.std() > 0:
            return rollouts, rewards
    # Fall through: all rewards identical; drop this prompt from the batch.
    return None, None`}
      </CodeBlock>

      <Prose>
        The practical effect is a meaningful improvement in training efficiency on hard problem
        sets, where the majority of initial rollouts are wrong. Early in training, a model
        attempting competition-level math will fail on nearly every rollout for a given prompt.
        Without dynamic sampling, the entire group contributes nothing and training stalls.
        With it, those prompts are deprioritized until the policy is capable enough to produce
        variance, and compute is redirected to prompts that are currently at the boundary of
        the model's ability — where the gradient signal is real.
      </Prose>

      <H3>Token-level policy gradient on long rollouts</H3>

      <Prose>
        Standard GRPO assigns one advantage value to an entire rollout and applies it uniformly
        to every token in the response. For a short rollout — 200 tokens, say — this is
        unproblematic. For a 4,000-token chain-of-thought, sharing a single gradient-scaling
        factor across all 4,000 tokens amplifies noise: later tokens in a long sequence
        accumulate importance ratio drift relative to the reference policy, and the uniform
        advantage magnifies that drift. DAPO proposes computing importance ratios and applying
        clipping at the individual token level, with the rollout-level advantage spread across
        tokens rather than repeated identically for each. The change is subtle in description
        but helps with long-rollout stability by preventing any single token's clipped
        importance ratio from dominating the update.
      </Prose>

      <H3>Overlong filtering</H3>

      <Prose>
        A more targeted heuristic from the DAPO paper: rollouts that hit the generation length
        cap without reaching a natural termination — truncated mid-reasoning rather than
        concluded — have their reward zeroed before the advantage is computed. The motivation
        is direct. A truncated rollout that happens to be scored as correct by a verifier
        — because the partial output looks right, or because the answer fragment is present
        even without a full reasoning trace — provides a misleading training signal. It teaches
        the model that hitting the length cap is acceptable, and combined with the length
        exploitation bias described above, this can produce a feedback loop toward
        ever-longer, never-concluding generations. Zeroing the reward for truncated outputs
        breaks that loop at its source.
      </Prose>

      <H2>How this fits with RLVR and GRPO</H2>

      <Prose>
        DAPO is explicitly positioned as a refinement of GRPO within the RLVR framework — the
        setting where rewards come from a verifier checking mathematical or logical correctness
        rather than from a trained reward model. It inherits GRPO's value-model-free structure,
        which eliminates the cost and instability of maintaining a separate critic, and RLVR's
        exact binary rewards, which eliminate the alignment noise introduced by a learned
        reward model. The DAPO contribution sits entirely in the training loop mechanics: clip-
        higher, dynamic sampling, token-level gradients, and overlong filtering are all
        adjustments to how the policy gradient is computed and which rollouts are allowed to
        influence it. The broader GRPO and RLVR papers, which are being covered in companion
        topics, handle the justification for value-model-free training and verifiable rewards
        respectively — DAPO assumes both and asks what goes wrong at scale when rollouts get
        long.
      </Prose>

      <Prose>
        The honest summary is engineering discipline applied to an identified failure regime.
        ByteDance Seed ran GRPO on long reasoning tasks, documented the specific pathologies
        that emerged at scale, and fixed each one with the narrowest intervention that worked.
        That narrowness is what makes the recipe reproducible — each fix is independently
        legible and independently removable, which is not always true of modifications to
        training loops.
      </Prose>

      <Callout accent="gold">
        Long-horizon RL is held together by a dozen small stabilization tricks. DAPO's
        contribution is documenting which tricks actually matter.
      </Callout>

      <Prose>
        These fixes are narrow by design, and their narrowness is also their honest limitation.
        DAPO does not solve the fundamental hardness of long-horizon credit assignment — the
        question of which tokens in a 4,000-token chain-of-thought were actually responsible
        for a correct final answer. It papers over specific pathologies that manifest when you
        run GRPO longer and harder than it was designed for. Progress on reasoning RL in 2025
        is a collection of small discipline-of-engineering wins stacked on top of each other:
        clip-higher, dynamic sampling, overlong filtering, process reward shaping, curriculum
        over problem difficulty. None of these is a fundamental breakthrough. Together, they
        are why open-source models started passing AIME.
      </Prose>

      <Prose>
        DAPO represents the current open-source state of the art for verifiable-reward RL on
        reasoning tasks. The next topic — knowledge distillation — takes a different angle on
        the same problem: instead of teaching a model to reason by having it attempt problems
        and receive correctness signals, copy the reasoning behavior of a stronger model
        directly, trace by trace.
      </Prose>
    </div>
  ),
};

export default dapo;
