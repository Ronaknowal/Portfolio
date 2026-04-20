import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const prmVsOrm = {
  title: "Process Reward Models (PRM) vs Outcome Reward Models (ORM)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Classical RLHF rewards whole responses. The reward model — covered in the RLHF topic — looks at a completed answer and outputs one scalar. That is an Outcome Reward Model, or ORM: score the outcome, ignore everything that led to it. For open-ended conversation this is reasonable enough. For reasoning tasks — math, coding, multi-step planning — it wastes most of the available signal. You can often see exactly where a proof went wrong, which step introduced a false assumption, which line of code introduced a bug. An ORM cannot use any of that information. It sees only the final answer and decides whether it was right or wrong.
      </Prose>

      <Prose>
        Process Reward Models score each intermediate step. Instead of one scalar at the end of a response, a PRM outputs a score per reasoning step — typically a probability that the step is correct given everything that came before it. The shift from ORM to PRM is a major part of how frontier reasoning models were trained, and understanding the difference clarifies why some alignment techniques that work well on open-ended chat fail quietly on mathematical reasoning.
      </Prose>

      <H2>What an ORM looks like</H2>

      <Prose>
        The architecture is a transformer backbone with a scalar projection head attached at the final token position. Given a prompt and a complete response concatenated together, the model reads the full sequence and produces one number. Training usually follows one of three setups: Bradley-Terry loss on preference pairs (preferred response should score higher than rejected), Elo scoring across a pool of ranked responses, or direct binary supervision for verifiable tasks where "correct answer" and "incorrect answer" are unambiguous ground truth. The last variant is especially common in math and coding benchmarks, where correctness can be checked automatically.
      </Prose>

      <CodeBlock language="python">
{`class OutcomeRewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.hidden_dim, 1)

    def forward(self, prompt, response):
        h = self.backbone(prompt + response)
        return self.head(h[:, -1, :])  # score at the last token only

# Training: (chosen, rejected) pairs under Bradley-Terry, or direct correctness labels.`}
      </CodeBlock>

      <Prose>
        The final-token projection is the key architectural choice. The model reads the entire prompt-plus-response, but only the last token's hidden state is used for scoring. In a causal transformer this position has attended over every preceding token, so the representation is in principle informed by the whole sequence. In practice, the model learns to compress its judgment into that single position during training. The simplicity is the point: one forward pass, one scalar, one gradient. The cost is that every token in the response — including any clearly wrong reasoning steps — receives the same terminal reward signal during RL training.
      </Prose>

      <H2>What a PRM looks like</H2>

      <Prose>
        A PRM shares the same backbone-plus-head architecture, but applies the head after every reasoning step rather than only at the final token. Responses are decomposed into steps — usually separated by newlines, explicit step markers, or a special delimiter token inserted during formatting. For each step, the model reads the accumulated context up to that point and outputs a score, typically passed through a sigmoid to produce a probability that the step is correct. The result is a vector of per-step scores instead of a single terminal scalar.
      </Prose>

      <CodeBlock language="python">
{`class ProcessRewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.hidden_dim, 1)

    def forward(self, prompt, steps):
        """
        steps: list of reasoning steps.
        Returns a score per step.
        """
        scores = []
        for i in range(len(steps)):
            context = prompt + " ".join(steps[:i+1])
            h = self.backbone(context)
            scores.append(torch.sigmoid(self.head(h[:, -1, :])))
        return scores  # (num_steps,)`}
      </CodeBlock>

      <Prose>
        The forward pass above runs the backbone once per step, accumulating context as it goes. In production implementations this is typically batched or handled with KV-cache reuse — the prefix up to step <Code>k</Code> is already computed when evaluating step <Code>k+1</Code>, so incremental inference avoids re-reading the full prompt each time. The step decomposition happens at a formatting layer upstream: the generator is prompted or fine-tuned to produce responses with explicit step structure, and the PRM learns to judge each delimited segment in turn.
      </Prose>

      <H2>Training data — where the asymmetry lives</H2>

      <Prose>
        ORM training data is cheap to collect. For every response, you need exactly one label: correct, incorrect, or a preference rank relative to another response. If the task is verifiable — an equation either balances or it does not — labels can be generated automatically at near-zero cost. Even for subjective tasks, having a human compare two complete responses is fast work: a few seconds per pair, with high inter-annotator agreement on gross quality differences.
      </Prose>

      <Prose>
        PRM training data is expensive. For every response, you need a label per step — and steps within a chain of thought can number anywhere from five to twenty for a typical math problem. OpenAI's "Let's Verify Step by Step" (Lightman et al., 2023) used roughly 800,000 step-level labels gathered from human annotators who rated each reasoning step individually. That is not 800,000 responses; it is 800,000 individual step judgments, with all the annotation infrastructure, labeler training, and quality-control overhead that implies. Math-Shepherd (2024) took a different approach: instead of asking humans to label steps, they generated multiple completions from each partial chain-of-thought and labeled a step as "good" if it led to a correct final answer more often than not across the sampled completions. This is cheaper and easily automated, but the labels are noisy — a step can look good by accident if the model happens to self-correct later, and a genuinely useful step can look bad if the model consistently fails to follow through. The labeling problem is the central practical challenge of PRM training, and the gap between cheap automatic labels and expensive human labels is a gap in signal quality that shows up in downstream benchmark performance.
      </Prose>

      <H2>How PRMs are used</H2>

      <Prose>
        There are two primary deployment patterns. The first is best-of-N with step-level scoring. Sample N complete solutions from the policy at inference time, score each step with the PRM, and pick the response with the best aggregate step score — typically the minimum step score (the weakest-link interpretation) or the product of step scores (a joint probability). This inference-time reranking dramatically outperforms using an ORM for the same best-of-N selection on hard math benchmarks, because the ORM cannot distinguish a response that got the right answer by lucky cancellation of errors from one that reasoned cleanly throughout.
      </Prose>

      <Prose>
        The second pattern is dense reward for RL policy training. Use the PRM as the reward signal during RLHF instead of an ORM. Every step of a generated chain-of-thought receives its own gradient contribution. High-scoring steps get positive reinforcement; low-scoring steps get negative reinforcement, regardless of whether the final answer was correct. This is the more ambitious use and the harder one to get right — it requires the PRM to be well-calibrated enough that per-step gradient signals are informative rather than noisy, which demands the expensive training data described above.
      </Prose>

      <StepTrace
        label="prm vs orm on a chain of thought"
        steps={[
          { label: "orm — final answer only", render: () => (
            <TokenStream tokens={[
              { label: "step 1", color: "#555" },
              { label: "step 2", color: "#555" },
              { label: "step 3", color: "#555" },
              { label: "step 4", color: "#555" },
              { label: "answer", color: "#4ade80" },
            ]} />
          ) },
          { label: "prm — step by step", render: () => (
            <TokenStream tokens={[
              { label: "step 1 ✓", color: "#4ade80" },
              { label: "step 2 ✓", color: "#4ade80" },
              { label: "step 3 ✗", color: "#f87171" },
              { label: "step 4 ✓", color: "#4ade80" },
              { label: "answer ✓", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      <Prose>
        The second panel shows the PRM's structural advantage. Step 3 was wrong even though the final answer happens to be correct — perhaps the error cancelled out, or the model self-corrected implicitly in step 4. An ORM sees a correct answer and rewards the entire response uniformly; it has no mechanism to penalize the flawed step 3 formulation. A PRM-based RL update can push against that specific step's formulation while preserving the steps that worked. Over many training iterations, this selective pressure on intermediate reasoning is why PRM-trained policies tend to produce cleaner chains of thought than ORM-trained ones, even when both achieve comparable final-answer accuracy early in training.
      </Prose>

      <H2>The credit assignment story</H2>

      <Prose>
        The ORM vs PRM distinction is, at its core, a credit assignment problem. In a long chain of reasoning, the model produces many intermediate tokens — setup steps, variable assignments, algebraic manipulations, intermediate conclusions. Determining which of those tokens are responsible for a correct or incorrect final outcome is hard. An ORM punts on the problem entirely: assign the outcome reward uniformly to every token in the response, let the gradient figure it out. This works better than nothing — the policy does eventually learn that certain patterns correlate with correct answers — but the gradient signal is diluted across hundreds of tokens, most of which had no causal relationship to whether the answer was right.
      </Prose>

      <Prose>
        A PRM attempts to solve credit assignment directly by labeling steps. The analog in classical reinforcement learning is the difference between episode-level reward and shaped per-step reward. RL theory has long established that dense, shaped rewards are more sample-efficient than sparse terminal rewards when available — the policy receives informative gradient signal at every step rather than waiting until episode end to learn anything. PRMs are that dense shaping signal applied to language model reasoning. The practical catch is the same as in robotics: designing a good per-step reward is much harder than defining a terminal reward, and a badly designed per-step reward can mislead the policy in ways a terminal reward never would.
      </Prose>

      <H3>The limits of PRMs</H3>

      <Prose>
        Three honest caveats about where PRMs break down or underperform. First, they only work for tasks with natural step boundaries. Mathematical proofs, code with explicit function calls, multi-step planning problems — these decompose cleanly into scorable units. Open-ended writing, summarization, and general conversation do not have obvious "steps" in any well-defined sense, which is why ORM remains the default for most chat alignment work. Trying to impose artificial step structure on prose often produces spurious score boundaries that the policy learns to exploit rather than reason around.
      </Prose>

      <Prose>
        Second, step labels are subjective and noisy even for structured tasks. Was that algebraic rearrangement step correct? It arrived at a valid equation, but it introduced a sign convention that made the next step harder. Annotators disagree on these calls at a rate that matters — inter-annotator agreement on step-level math labels tends to be noticeably lower than on final-answer labels, and the disagreement concentrates on exactly the intermediate steps where precise signal would matter most.
      </Prose>

      <Prose>
        Third, PRMs can be reward-hacked just as ORMs can. The policy learns to produce step sequences that score well according to the PRM's learned heuristics, regardless of whether those steps genuinely advance the proof. If the PRM was trained on human annotations that favored a particular proof style, the policy learns that style. If the PRM's step classifier is fooled by confident-sounding language, the policy learns to sound confident. The reward model is always a proxy, and proxies can always be exploited.
      </Prose>

      <Callout accent="gold">
        The quality of a PRM-trained model is bounded above by the quality of step-level supervision. Cheap automatic labels give cheap improvements; expensive human labels give expensive ones.
      </Callout>

      <Prose>
        The PRM vs ORM choice is one of the clearer levers in post-training: if your task has discrete reasoning steps and you can afford to label them, PRMs give you more signal per sample than an equivalent investment in outcome supervision. The tradeoff is not about architecture — the two models are nearly identical — it is entirely about data collection cost and the fidelity of per-step labels. Upcoming topics on RLVR and RL for Reasoning pick up where this leaves off: what happens when the per-step signal can be generated automatically, without human annotation at all, by replacing step-level human judgment with verifiable program execution or formal proof checking? That is the direction the field is currently moving, and it is where the expensive data collection problem starts to become tractable.
      </Prose>
    </div>
  ),
};

export default prmVsOrm;
