import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { Plot } from "../../components/viz";

const testTimeCompute = {
  title: "Test-Time Compute Scaling",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        Classical scaling laws describe training compute: give a model more parameters, more
        tokens, and more FLOPs during training, and its cross-entropy loss falls as a predictable
        power law. That axis is the one Kaplan and Chinchilla mapped. Test-time compute scaling
        describes a different axis entirely: given a fixed model — one that is already trained,
        whose weights are frozen — the same model gets meaningfully better at hard problems when
        you let it use more tokens at inference. Not a different model, not a bigger checkpoint.
        The same one, spending more time thinking before it answers.
      </Prose>

      <Prose>
        OpenAI's o1, DeepSeek-R1, and their successors are the first generation of models where
        this effect is large, reliable, and a primary product feature. These are not models that
        happen to produce longer outputs. They are models specifically trained — through
        reinforcement learning on verifiable problems — to allocate inference tokens toward
        genuine reasoning, such that each additional token of generation reduces the probability
        of a wrong answer. The training side of how those models are built is covered in the
        RL for Reasoning topic. This topic covers what happens at serving time: the empirical
        curve, the mechanisms behind it, and what it means for how inference stacks are run.
      </Prose>

      <H2>The empirical finding</H2>

      <Prose>
        On hard reasoning benchmarks — AIME 2024, GPQA Diamond, Codeforces competitive
        programming — the same trained reasoning model reaches higher accuracy when allowed to
        generate more tokens per problem. The relationship is approximately log-linear: doubling
        the inference token budget produces a roughly constant additive accuracy gain, so the
        curve looks like a straight line on a log-x axis. For AIME-2024, o1-class models move
        from around 30% accuracy at one thousand tokens per problem to around 80% at thirty
        thousand. That is a 50-point swing from a single dial: how long the model is allowed to
        think.
      </Prose>

      <Prose>
        The critical comparison is against standard pre-RL models. A GPT-4-class model given the
        same token budget shows essentially no such curve. Its accuracy rises from roughly 24% at
        1k tokens to around 37% at 100k tokens — most of that is sampling noise and occasional
        lucky traces, not a reliable reasoning mechanism. The log-linear scaling behavior is not
        a property of transformers in general. It is a property of models trained to use
        inference tokens productively. The RL training procedure is what puts the slope into the
        curve.
      </Prose>

      <Plot
        label="aime accuracy vs. inference tokens per problem (illustrative)"
        width={520}
        height={260}
        xLabel="log10 tokens per problem"
        yLabel="accuracy %"
        series={[
          { name: "o1-class (RL-trained)", points: [[3, 32], [3.5, 52], [4, 68], [4.5, 78], [5, 83]] },
          { name: "GPT-4 class (no RL)", points: [[3, 24], [3.5, 30], [4, 34], [4.5, 36], [5, 37]] },
        ]}
      />

      <H2>Why this is a different scaling axis</H2>

      <Prose>
        Pre-training scaling is capital-intensive and upfront. You pay once, during training, and
        the model's capabilities are fixed into its weights. A model that cost $100M to train
        costs roughly the same to serve regardless of how often it is queried — the training
        expenditure is sunk. Inference scaling is a runtime cost: every query pays per token
        generated, and the token count is variable. Those two economic structures are deeply
        different from a product perspective.
      </Prose>

      <Prose>
        The implication is that test-time compute is controllable in a way that model size is
        not. To get 2× better capabilities from a bigger model, you retrain — a months-long,
        nine-figure commitment that locks in a new capability level for everyone. To get 2×
        better performance on a specific hard query using test-time compute, you set a higher
        token budget — a per-request decision that costs nothing extra for easy queries and pays
        only when the problem warrants it. This makes test-time compute a form of adaptive
        resource allocation that training-time scaling simply cannot provide. A model trained
        with more parameters is always a bigger model, on every query. A reasoning model with a
        tunable token budget is a cheap model on simple queries and an expensive one on hard
        ones — which is exactly what the economics of a general-purpose AI product want.
      </Prose>

      <H2>The techniques that make it work</H2>

      <Prose>
        At least three distinct mechanisms produce the accuracy-vs-tokens curve, and different
        deployed models weight them differently. They are not mutually exclusive — a production
        reasoning model may use all three in combination.
      </Prose>

      <Prose>
        The first is <strong>internal chain-of-thought</strong>. The model generates an extended
        reasoning trace — sometimes tens of thousands of tokens — before producing a final
        answer. That trace is not shown to the user in most deployments, but it is generated and
        billed as output tokens. The reasoning trace lets the model decompose problems, check
        partial results, backtrack when it finds a contradiction, and approach the same
        sub-problem from multiple angles. o1 and R1-class models do this by default. The RL
        training procedure is what makes the reasoning trace useful rather than verbose: it
        rewards models whose longer traces actually produce better final answers on verifiable
        problems.
      </Prose>

      <Prose>
        The second is <strong>best-of-N with a verifier</strong>. Generate N independent
        candidate answers — or N complete reasoning traces — score each with a verifier, and
        return the best-scoring one. The verifier can be a process reward model (PRM) that scores
        reasoning steps, an outcome reward model (ORM) that scores final answers, or an external
        checker (a unit test suite, a symbolic solver, a ground-truth lookup). Best-of-N is
        naturally parallelizable: all N generations run simultaneously on different compute
        slots. This is covered in more detail in the next section.
      </Prose>

      <Prose>
        The third is <strong>tree search over reasoning</strong>. Rather than a linear chain of
        thought, the model explores a tree of reasoning states — expanding promising branches,
        pruning dead ends, using MCTS or beam-search variants to navigate. A process reward model
        scores intermediate states and guides the search. This approach has produced strong
        results in research settings, particularly on math and formal verification tasks, but it
        is not standard in current deployed reasoning models. The infrastructure complexity is
        substantial: managing a branching inference process across a batch of queries is
        qualitatively harder than managing linear generation.
      </Prose>

      <H2>How inference stacks have adapted</H2>

      <Prose>
        Serving a reasoning model is not serving a chat model with a higher <Code>max_tokens</Code>.
        The operational profile is different enough that it requires explicit changes in how serving
        infrastructure is designed and priced.
      </Prose>

      <Prose>
        Hidden reasoning tokens are the first departure. The internal chain-of-thought is
        typically not surfaced to the user — only the final answer is returned. But those tokens
        are generated, occupy GPU memory throughout the generation, and are billed. A user who
        sends a three-sentence math problem may receive a two-sentence answer, but the model
        generated 15,000 tokens of intermediate reasoning that never appeared in the response
        body. Billing APIs for reasoning models have had to introduce explicit distinctions
        between thinking tokens and output tokens precisely because this distinction is
        non-obvious and non-trivial in cost.
      </Prose>

      <Prose>
        Variable per-query latency is the second. A chat model produces its response in one to
        three seconds for typical inputs. A reasoning model may take ten to one hundred and
        twenty seconds. The variance is not just higher in absolute terms — it is higher in ratio,
        because the same model processes both a simple arithmetic check (short reasoning trace)
        and a competition math problem (very long reasoning trace) with the same architecture.
        Serving schedulers designed around low-variance chat workloads need to handle very
        different queue dynamics when a single query can tie up a slot for two minutes.
      </Prose>

      <Prose>
        Output-dominated cost is the third. Standard chat models are roughly balanced between
        input and output cost, or often input-heavy when system prompts are long. Reasoning
        models are heavily output-dominated: the problem statement is short; the reasoning trace
        is the expensive part. This shifts the economics of batching, prefill/decode splitting,
        and KV cache utilization in ways that have required explicit engineering attention in
        production deployments.
      </Prose>

      <CodeBlock language="python">
{`# Serving a reasoning model vs a chat model — different defaults.

# Chat model — typical request
chat_response = model.generate(
    prompt=user_message,
    max_tokens=500,
    temperature=0.7,
)

# Reasoning model — same API, different knobs
reasoning_response = reasoning_model.generate(
    prompt=user_message,
    max_thinking_tokens=32_000,   # budget for internal reasoning
    max_tokens=1_000,             # budget for the final answer
    temperature=0.0,              # reasoning typically uses greedy or near-greedy
)
# Billing: fresh_input * input_rate + (thinking_tokens + output_tokens) * output_rate`}
      </CodeBlock>

      <H3>Best-of-N with verification — the parallel alternative</H3>

      <Prose>
        For verifiable tasks, best-of-N with a verifier approximates reasoning-model quality
        using a standard model rather than an RL-trained one. Sample N complete solutions
        independently, score each with a verifier or process reward model, and return the
        highest-scoring result. In practice, N=32 typically matches or outperforms serial
        chain-of-thought on math benchmarks at roughly the same total token cost. The key
        tradeoff is along the latency-vs-throughput axis rather than the accuracy axis.
      </Prose>

      <Prose>
        Serial chain-of-thought ties up one GPU slot for the full duration of the reasoning
        trace — potentially minutes. Best-of-N runs all N samples in parallel, each on a separate
        slot, and finishes when the slowest sample completes. If 32 slots are available, 32
        samples of 1,000 tokens each finish in roughly the same wall-clock time as one sample of
        1,000 tokens. The latency advantage of best-of-N over long serial reasoning can be
        substantial for latency-sensitive applications. The disadvantage is that best-of-N
        requires a reliable external verifier — a unit test suite, a formal checker, or a
        well-calibrated reward model. Without a verifier that can rank candidate answers
        correctly, sampling 32 solutions gives you 32 chances to pick the wrong one. The
        technique is most powerful exactly where problems are most verifiable: code, math,
        formal logic.
      </Prose>

      <H2>The test-time compute sweet spot</H2>

      <Prose>
        Not every task benefits from more reasoning tokens. Casual conversation gets no lift —
        more thinking time does not make "what's a good pizza topping" a better question to
        answer. Factual lookup that falls within the model's training distribution is already
        answered correctly at one token of deliberation or not at all — the reasoning trace adds
        nothing but latency. Creative writing quality does not improve with extended internal
        deliberation in any reliable way. For most of what general-purpose assistants do across
        a day of queries, test-time compute is waste.
      </Prose>

      <Prose>
        The gains are concentrated on problems with verifiable structure that admit multi-step
        solutions: competition mathematics, programming challenges where the answer can be tested
        against a suite, logical deduction puzzles with a determinate solution, structured
        planning tasks where intermediate steps can be checked for consistency. On those
        problems, the accuracy gains are large, reliable, and reproducible — not quirks of a
        single benchmark. The practical takeaway for product design is that reasoning mode should
        be a routing decision, not a default. The question is whether the query is the kind of
        problem where extended deliberation returns dividends, and for most queries the honest
        answer is no.
      </Prose>

      <Callout accent="gold">
        Test-time compute is a knob, not a default. Use it when the problem has a checkable
        answer or a verifier; don't use it for tasks where the model's first attempt is already
        as good as the 10,000th.
      </Callout>

      <H2>The open questions</H2>

      <Prose>
        Several things about test-time compute scaling remain unsettled, and the honest position
        is to name them rather than paper over them with confident extrapolation.
      </Prose>

      <Prose>
        The first is whether the log-linear accuracy-vs-tokens curve continues indefinitely or
        eventually plateaus. R1-class results on AIME-2024 already sit near the range achieved
        by human competition participants, which suggests that for at least some benchmarks, the
        relevant ceiling is human-expert performance rather than some fundamental model limit.
        Whether the curve keeps climbing beyond that level on genuinely frontier-hard problems —
        open research questions in mathematics, novel software vulnerabilities, scientific
        hypothesis generation — is not known. There is a real possibility that the scaling
        behavior visible in the 1k-to-100k-token regime is specific to problems whose solution
        structure matches the training distribution of verifiable problem types.
      </Prose>

      <Prose>
        The second open question is whether the method transfers to tasks without natural
        verifiers. Current reasoning models are trained and evaluated primarily on tasks where
        correctness is unambiguous. There is some evidence that the learned reasoning behavior
        transfers to harder, more open-ended tasks — that models trained on verifiable math
        reason better on novel tasks generally — but the signal is substantially weaker than
        within the verifiable domain. Whether this is a fundamental limitation or a training
        recipe problem is an active research question.
      </Prose>

      <Prose>
        The third is adaptive budget allocation: given a query, how many inference tokens should
        be spent on it? The current answer in production systems is essentially a fixed ceiling
        — set <Code>max_thinking_tokens</Code> to some large number and let the model stop when
        it decides it is done. But the optimal allocation varies enormously across queries. A
        problem that takes a median human expert two minutes does not need the same token budget
        as one that takes two hours. Building systems that diagnose query difficulty and allocate
        compute accordingly — without solving the query first in order to know how hard it is —
        is itself an open research problem, and one with significant economic implications if it
        is solved well.
      </Prose>

      <Prose>
        Test-time compute scaling is the most interesting inference-side development in AI of
        the past two years. It changes what a deployed model is, both economically and
        operationally: from a fixed-cost artifact whose capabilities are determined at training
        time, to a system whose effective capability is a runtime variable that can be dialed up
        or down per query. The final topic in this section — inference engines and serving
        frameworks — turns to the infrastructure that makes all of this possible at production
        scale.
      </Prose>
    </div>
  ),
};

export default testTimeCompute;
