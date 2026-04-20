import { Prose, H2, H3, Callout } from "../../components/content";
import { TokenStream, Plot } from "../../components/viz";

const rlForReasoning = {
  title: "RL for Reasoning (DeepSeek-R1 Style)",
  readTime: "13 min",
  content: () => (
    <div>
      <Prose>
        Something genuinely surprising happened in late 2024 and through 2025. Models trained
        with pure RL on verifiable rewards — math competition problems, competitive programming,
        formal proofs — started developing behaviors that look like reasoning. Long internal
        chains of thought. Self-correction mid-derivation. Exploration of alternative
        approaches, then backtracking when those approaches fail. Not because anyone designed
        those behaviors into the training procedure. Because the reward signal made them
        profitable. DeepSeek-R1's technical report called it the "aha moment." OpenAI's
        o-series shows the same pattern. This topic is about what's happening, why it
        constitutes a genuine capability shift, and what we still don't understand.
      </Prose>

      <Prose>
        To be precise about what's surprising: RLHF, DPO, GRPO, and all the preference
        optimization methods covered earlier in this section shape how the model responds —
        its tone, helpfulness, adherence to instructions. They do not reliably change
        how the model thinks. RL on verifiable rewards changed what the model does inside
        the response. That's a different kind of intervention.
      </Prose>

      <H2>The setup</H2>

      <Prose>
        Start with a capable base model — DeepSeek-V3, Qwen-72B, Llama 3 70B, something at
        that capability tier. Skip supervised fine-tuning, or apply only a minimal cold-start
        SFT pass to establish a reasoning format. Then apply GRPO — the group-relative policy
        optimization algorithm described in the GRPO/RLOO/KTO topic — on problems with
        automatic verifiers. Math competition problems verified by answer-matching. Coding
        benchmarks verified by test execution. Formal proofs verified by a type checker.
        The reward is binary: the final answer is correct or it isn't.
      </Prose>

      <Prose>
        The KL anchor to the base model is set to a moderate value — DeepSeek-R1 used β ≈ 0.04
        — loose enough to allow the policy to develop substantially different behavior,
        tight enough to prevent total distribution collapse. GRPO samples eight to sixteen
        responses per problem per training step, computing advantages within each group. As
        covered in the RLVR topic, this group sampling is what makes the sparse reward signal
        tractable: even if only one of eight rollouts solves the problem, the gradient is
        meaningful. Training runs for millions of problems. The compute budget is large. And
        then something unexpected appears in the response traces.
      </Prose>

      <H2>What emerges</H2>

      <Prose>
        The model, without being shown any chain-of-thought demonstrations, starts producing
        longer and longer reasoning traces. Thousands of tokens before the final answer. And
        within those traces, specific patterns emerge that were not in the training data:
        the model pauses, reconsiders, identifies a flaw in its own reasoning, and tries
        again. The vocabulary of self-correction — "wait, let me reconsider," "that can't
        be right," "let me try a different approach" — appears spontaneously. Not as mimicry
        of CoT demonstrations. As a learned strategy for earning higher reward.
      </Prose>

      <TokenStream
        label="a reasoning trace fragment — self-correction in the wild"
        tokens={[
          { label: "Let me compute", color: "#e2b55a" },
          { label: " 37 × 43", color: "#e2b55a" },
          { label: " = 1591.", color: "#e2b55a" },
          { label: " Wait,", color: "#f87171" },
          { label: " let me", color: "#f87171" },
          { label: " verify:", color: "#f87171" },
          { label: " 37 × 40", color: "#4ade80" },
          { label: " = 1480,", color: "#4ade80" },
          { label: " 37 × 3", color: "#4ade80" },
          { label: " = 111,", color: "#4ade80" },
          { label: " total 1591.", color: "#4ade80" },
          { label: " Correct.", color: "#4ade80" },
        ]}
      />

      <Prose>
        Two things to notice in that trace. First, the "Wait, let me verify" moment is not
        a planned behavior — it's what the policy learned to do because it correlates with
        higher accuracy on hard problems. The model is not following a rule that says "always
        check your arithmetic." It discovered that checking arithmetic improves reward, and
        so it checks. Second, the verification is substantive: the model re-derives the
        answer by decomposing 37 × 43 into 37 × 40 + 37 × 3. It isn't repeating the same
        computation. It's choosing a different path to confirm the same result. That's closer
        to structured verification than to confident repetition.
      </Prose>

      <Prose>
        The length of reasoning traces grows systematically with problem difficulty. On
        easy problems the model remains concise. On hard competition problems the traces
        span thousands of tokens, with multiple failed attempts before a successful one.
        This allocation of inference compute is also not designed in — it emerges from the
        reward structure. Harder problems require more exploration to earn reward, so the
        policy learns to explore more on them.
      </Prose>

      <H2>The "aha moment" phenomenon</H2>

      <Prose>
        DeepSeek-R1's technical report documents something even more striking than gradual
        improvement: discontinuities. At certain points during training, the model's approach
        to a class of problems shifts abruptly. Before the transition: greedy short solutions,
        one attempt, commit to the first reasonable answer. After: multi-step exploration,
        explicit backtracking, reasoning traces that acknowledge uncertainty. The transition
        is not gradual. Training loss plots show corresponding inflection points — sustained
        periods of flat progress followed by a jump, then a new plateau.
      </Prose>

      <Prose>
        The mechanism, loosely reconstructed from what's reported: the policy has a threshold
        of reasoning quality below which most problems in some class return zero reward.
        Before that threshold, the gradient on those problems is uninformative — most rollouts
        fail, so GRPO's group normalization produces near-zero advantages. Once the policy
        crosses the threshold — perhaps nudged there by improvement on easier problems —
        a new class of harder problems starts returning non-zero reward. Those problems
        now contribute gradients. The policy improves further. This unlocks gradients on
        yet harder problems. A cascade. Each new tier of solvable problems provides training
        signal that pushes the policy toward the next tier.
      </Prose>

      <Prose>
        This is why curriculum matters more for RLVR than for most training regimes: the
        cascade only starts if the initial problem distribution is hard enough to require
        multi-step reasoning but easy enough to produce non-zero rewards from the starting
        policy. Too easy and the model never develops longer traces. Too hard and training
        stalls. The threshold finding — that reasoning behavior emerges discontinuously
        rather than gradually — is specific to this regime. It's not present in RLHF or
        supervised fine-tuning in the same form.
      </Prose>

      <H2>Test-time compute — where the capability cashes out</H2>

      <Prose>
        The most practically significant consequence of RL-trained reasoning is that the
        model learns to use inference-time tokens productively on hard problems. Given more
        tokens to think, its accuracy on hard math scales sharply. Given fewer, it truncates
        its reasoning and performance drops. The model has learned, through the reward signal,
        that thinking longer is worth it — and this lesson is reflected in its weights.
      </Prose>

      <Plot
        label="accuracy vs. inference tokens on hard math (illustrative)"
        width={520}
        height={240}
        xLabel="log10 inference tokens per problem"
        yLabel="accuracy %"
        series={[
          { name: "RL-trained (R1-style)", points: [[2, 28], [2.5, 42], [3, 58], [3.5, 72], [4, 81], [4.5, 87]] },
          { name: "Base model + CoT prompt", points: [[2, 22], [2.5, 30], [3, 35], [3.5, 38], [4, 40], [4.5, 41]] },
        ]}
      />

      <Prose>
        The plot shows the qualitative shape of the difference. The RL-trained model's
        accuracy-vs-tokens curve on AIME-style benchmarks is approximately linear in
        log inference compute — a relationship that didn't exist in pre-RL models at anything
        like the same slope. The base model with a CoT prompt shows some improvement with
        longer generations, then flattens. The RL-trained model keeps climbing. This is not
        the same model using more words to say the same thing. It's the same model using
        more tokens to explore a larger portion of the reasoning space before committing
        to an answer.
      </Prose>

      <Prose>
        This property — that the same model becomes more accurate with more inference compute
        — is qualitatively different from standard scaling. Larger models are more accurate
        because they have more parameters. The RL-trained model is more accurate because
        it learned to use time. The two dimensions are now independently exploitable: you
        can scale the model or scale inference, and both pay out.
      </Prose>

      <H2>Why this is a real capability jump, not a prompting trick</H2>

      <Prose>
        The natural skeptical read: these models are doing longer chain-of-thought, which
        we've known helps. True, but incomplete. Prompting a base model with "think step
        by step" or providing a few-shot CoT demonstration produces some of the same
        behavior, and it helps. But it plateaus quickly, as the plot above suggests. The
        base model's accuracy flattens even as you allow more tokens, because the model
        doesn't know how to fill the extra space with useful reasoning. It elaborates.
        It restates. It occasionally contradicts itself and keeps going.
      </Prose>

      <Prose>
        The RL-trained model does something different. When it uses extra tokens, it tends
        to use them for exploration: trying an approach, evaluating whether it's working,
        abandoning it, trying another. The backtracking isn't decorative — it's accompanied
        by observable changes in the reasoning direction. The model explores the reasoning
        space in a way that plain prompting cannot induce, because the exploration behavior
        was learned through thousands of training steps where exploratory traces earned
        higher reward than non-exploratory ones. The difference between prompted CoT and
        RL-trained reasoning is closer to the difference between asking someone to show
        their work and training them to actually check it.
      </Prose>

      <Prose>
        Structured search is a reasonable analogy. Not beam search in the generation sense,
        but search in the reasoning sense: the model tries a path, evaluates it against
        implicit criteria, decides whether to continue or backtrack, and terminates when
        it has a solution it trusts. The policy gradient updates during RLVR training
        directly reinforce the search decisions that lead to verified correct answers.
        Over millions of training problems, the model learns which search heuristics work.
      </Prose>

      <H2>What we don't know</H2>

      <Prose>
        The reporting on RL-for-reasoning has been enthusiastic, sometimes to a fault.
        There are genuine open questions worth holding carefully.
      </Prose>

      <Prose>
        First: does the reasoning capability transfer to non-verifiable domains? Early
        evidence from DeepSeek-R1 and similar models says yes — models trained on math
        and code reasoning show improved performance on general benchmarks, writing tasks,
        and instruction following that don't have verifiers. The transfer effect is real
        but muted. The model doesn't become equally capable at reasoning through ambiguous
        policy questions as it is at AIME problems. The RL training sharpens a certain
        style of structured deliberation that benefits some downstream tasks more than others.
        The mechanism isn't fully understood.
      </Prose>

      <Prose>
        Second: is there a ceiling? R1-style training on competition math has produced
        dramatic gains — from mid-tier to near-expert-level performance on AIME-2024 in
        one training run. But it's not clear whether continuing the same procedure will
        keep improving, plateau at human-expert level, or require qualitatively different
        interventions to go further. The cascade dynamic described earlier suggests there
        may be additional jumps available as harder problem classes become solvable —
        but this is extrapolation from a few data points, not a confident prediction.
      </Prose>

      <Prose>
        Third — and most philosophically contested: are these models reasoning in any
        interesting sense, or executing sophisticated pattern-matching that happens to
        perform well on benchmarks? This question depends substantially on what you mean
        by reasoning. The skeptical view holds that the model learned to produce text that
        looks like reasoning because that text pattern correlates with correct answers in
        training, and there's nothing more to it. The less skeptical view points to evidence
        of genuine transfer — the model successfully applies learned reasoning patterns to
        problems structurally unlike anything in its training data — and genuine
        self-correction — the model identifies specific errors in its own intermediate
        steps rather than just regenerating from a different starting point. Both views
        can be held consistently with the experimental evidence. What the experimental
        evidence does not support is the fully dismissive version: that the self-correction
        behaviors are purely cosmetic and don't contribute to accuracy. They do.
      </Prose>

      <Callout accent="gold">
        RL for reasoning is the first post-training regime that reliably produces behaviors
        the training data didn't contain. That alone makes it unlike anything that came before.
      </Callout>

      <H3>What's being built on top</H3>

      <Prose>
        The frontier is moving quickly past math and code. Labs are extending R1-style
        training to multi-step agentic tasks where the "verification" is grounded in
        whether an agent successfully completes a task in a real environment — code agents
        evaluated by whether the repository's test suite passes after their edits, tool-using
        agents evaluated by whether a web form was correctly filled, computer-use agents
        evaluated by whether the screen ends up in a target state. Formal theorem proving
        pipelines use Lean and Isabelle as verifiers and are producing models that can
        find novel proofs of non-trivial results. Each extension follows the same pattern:
        identify a domain with a mechanically checkable notion of success, build the verifier,
        apply GRPO-style training, let the model learn what reasoning looks like in that domain.
        The pattern — RL on verifiable rewards, emergent reasoning traces — generalizes
        wherever a verifier can be constructed.
      </Prose>

      <Prose>
        The generalization pressure runs in both directions. As reasoning models are deployed,
        the interesting failures tend to occur at the boundary of verification — problems
        that are mostly verifiable but require judgment on edge cases, or tasks where the
        verifier is imperfect and the model learns to exploit the imperfection rather than
        solve the problem. These failure modes look structurally like the reward hacking
        in RLHF, applied to a different type of reward signal. The verifier design is
        therefore as important as the training algorithm, and it's actively evolving.
      </Prose>

      <H2>The Post-Training section as a whole</H2>

      <Prose>
        The arc of this section has been about making the base model useful. Supervised
        fine-tuning taught it to follow instructions. RLHF taught it to match human
        preferences. DPO and SimPO simplified the preference optimization loss and removed
        the online RL dynamics. Constitutional AI replaced human labelers with a principled
        model-generated preference signal. RLVR replaced the learned reward model entirely
        with a verifier — exact, stable, unhackable — for tasks where correctness is
        mechanically checkable. And RL for reasoning extended that to problems hard enough
        that the model had to develop new behaviors — longer traces, self-correction,
        structured exploration — to solve them at all.
      </Prose>

      <Prose>
        Each step in that arc addressed a specific limitation of what came before. RLHF's
        reward hacking led to RLVR. RLVR's sparse rewards on hard problems led to GRPO's
        group sampling. GRPO's group sampling on hard enough problems, run long enough,
        produced the aha phenomenon. The next section — Inference Optimization — turns
        to the practical question that follows from all of this: how do you actually serve
        these models in production, at scale, with reasoning traces that may span thousands
        of tokens per query, without spending your margins entirely on compute?
      </Prose>
    </div>
  ),
};

export default rlForReasoning;
