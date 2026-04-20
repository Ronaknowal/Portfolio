import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const speculativeDecoding = {
  title: "Speculative Decoding",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        The memory-bandwidth bottleneck in autoregressive decode is strange if you sit with it for a moment. A modern GPU has terabytes per second of raw compute capability, and it spends most of decode sitting idle — waiting on HBM reads, fetching all 70 billion parameters from memory so it can emit one token, then doing it again, then again. Compute utilization during decode on a single-user request is routinely under 10%. The chip that could perform quadrillions of operations per second is being paced by a memory bus.
      </Prose>

      <Prose>
        Speculative decoding is a family of techniques that exploit this slack. The core move: use the underutilized compute to verify many candidate tokens per memory pass, rather than producing one guaranteed token per pass. When the candidates are mostly correct, you get 2–3× effective throughput with zero change to the output distribution. No quality tradeoff. The speedup comes entirely from using hardware that was already being wasted.
      </Prose>

      <H2>The core insight</H2>

      <Prose>
        Generating one token with a large model costs roughly the same as generating one token with a large model regardless of what else you do during that pass. The bottleneck is reading the weights — you have to load all of them from HBM to compute any forward pass, and a 70B model has a lot of weights to load. That cost does not scale with the number of tokens you process in parallel. If you process a sequence of K tokens simultaneously, the memory-fetch cost is essentially the same single forward pass; you get K output distributions instead of one.
      </Prose>

      <Prose>
        Now suppose you already have K candidate tokens from some cheap source. Instead of generating one token with the large model, you score all K candidates with a single large-model forward pass — reading the weights once, processing the entire candidate sequence in parallel. If most candidates are correct, you have effectively produced K tokens at the memory cost of one. The acceptance rate of the candidates is what determines the speedup. If 80% of proposed tokens are accepted on average, and you propose 5 at a time, you accept roughly 4 per pass. The large model's memory bandwidth is still the limiter, but now each memory pass yields 4 tokens instead of 1.
      </Prose>

      <H2>The basic algorithm — Leviathan et al. 2022</H2>

      <Prose>
        The original formulation by Leviathan, Kalman, and Matias has three components: a small "draft" model that proposes tokens quickly, a large "target" model that scores them accurately, and a verification procedure with a mathematical guarantee about the output distribution.
      </Prose>

      <Prose>
        Each iteration begins with the draft model autoregressively generating K tokens. This is fast — the draft model is small, often 1–7B parameters — but the K tokens are sequential; each one conditions on the previous. Once the draft sequence is ready, the target model runs a single forward pass over the full context plus all K draft tokens, producing K probability distributions in parallel. Then the verification procedure walks through each position and decides whether to accept or reject.
      </Prose>

      <StepTrace
        label="speculative decoding — one iteration"
        steps={[
          { label: "1. draft model proposes K tokens", render: () => (
            <TokenStream tokens={[
              { label: "The", color: "#e2b55a" },
              { label: " cat", color: "#e2b55a" },
              { label: " sat", color: "#e2b55a" },
              { label: " on", color: "#e2b55a" },
              { label: " the", color: "#e2b55a" },
            ]} />
          ) },
          { label: "2. target model scores all K in parallel", render: () => (
            <TokenStream tokens={[
              { label: "The ✓", color: "#4ade80" },
              { label: " cat ✓", color: "#4ade80" },
              { label: " sat ✓", color: "#4ade80" },
              { label: " on ✓", color: "#4ade80" },
              { label: " the ✗", color: "#f87171" },
            ]} />
          ) },
          { label: "3. accept prefix; resample at rejection", render: () => (
            <TokenStream tokens={[
              { label: "accepted: The cat sat on", color: "#4ade80" },
              { label: " + resampled: ", color: "#555" },
              { label: "mat", color: "#c084fc" },
            ]} />
          ) },
        ]}
      />

      <Prose>
        The acceptance rule is not a heuristic. It is derived from the requirement that the output distribution of the speculative process must be identical to sampling directly from the target model. Tokens where the target assigns higher probability than the draft are always accepted — the target is more confident, so there is no reason to reject. Tokens where the target assigns lower probability are accepted with probability proportional to the ratio. On rejection, the remaining probability mass is renormalized and a new token is sampled from that adjusted distribution.
      </Prose>

      <CodeBlock language="python">
{`def accept_token(draft_prob, target_prob):
    """
    Acceptance rule that guarantees identical output distribution to
    direct sampling from the target model.
    """
    if target_prob >= draft_prob:
        return True  # Always accept
    return random.random() < (target_prob / draft_prob)

def resample_on_rejection(draft_dist, target_dist):
    """
    After rejection, sample from the renormalized difference.
    Guarantees the overall distribution matches target sampling.
    """
    adjusted = torch.clamp(target_dist - draft_dist, min=0)
    adjusted = adjusted / adjusted.sum()
    return torch.multinomial(adjusted, 1)`}
      </CodeBlock>

      <Prose>
        The math is clean: the acceptance-rejection step is exactly the right correction to make the marginal distribution of accepted tokens match the target. After a rejection, the resampled token comes from the probability mass the target assigns above what the draft assigned — the part the draft underestimated. The chain of accepted tokens plus the resampled token at the first rejection point is, in distribution, exactly what you would have gotten from sampling the target directly, one token at a time. No approximation.
      </Prose>

      <H2>Why it works when it works</H2>

      <Prose>
        Speculation wins when the draft model's distribution closely tracks the target's on most tokens. For text that has "easy" stretches — common words, obvious continuations, boilerplate phrasing, code with predictable structure — a 1B draft model agrees with a 70B target model 70–85% of the time. Those agreements cost nothing marginal; the tokens are accepted at the same memory-bandwidth cost as generating zero extra tokens. Only the disagreements require re-spending target-model capacity, and they are handled correctly by the resampling step.
      </Prose>

      <Prose>
        Empirical acceptance rates cluster around task difficulty: roughly 0.7 on natural language chat, 0.55 on code (higher entropy, less predictable token sequences), and 0.4 on reasoning tasks where the draft model genuinely cannot anticipate the target's chain-of-thought. Speedups track this directly — around 2.5× on chat, 2× on code, 1.5× on reasoning. These are not constant; they depend on the specific draft and target pair and on the prompt distribution. But the ordering is consistent: tasks with higher acceptance rates deliver higher speedups, and acceptance rate is approximately what you would predict from the KL divergence between the draft and target distributions.
      </Prose>

      <H2>Draft model options</H2>

      <Prose>
        The original paper assumed a separate small model from the same model family — Llama 3 8B drafting for Llama 3 70B, for instance. That works well: the two models share vocabulary and training distribution, so the draft is well-calibrated to the target's preferences. The downside is operational: you are serving two models simultaneously, managing two sets of KV caches (covered in the KV cache topic), and routing both through the inference stack. For systems that already manage memory carefully, the overhead is non-trivial.
      </Prose>

      <Prose>
        Three other approaches trade draft quality for operational simplicity. N-gram lookup uses a table built from the current context: find the last N tokens, look up what token followed them in the prompt or previous output, propose that. No model. No parameters. Acceptance rates on highly repetitive tasks — code completion with boilerplate, document editing — can hit 30–50%. For those tasks it is the cheapest possible draft source.
      </Prose>

      <Prose>
        Medusa attaches multiple auxiliary prediction heads to the target model itself. The first head predicts the next token; the second predicts the token after that; and so on up to depth four or five. No separate model; the target drafts for itself using its own residual stream as context. Acceptance rates are lower than a well-matched small model, but operational overhead is nearly zero — you only load one set of weights, one KV cache, one model. EAGLE extends this idea by giving the auxiliary heads access to the target's hidden states as additional context, recovering much of the acceptance-rate gap. Both report 2–3× speedups across tasks while keeping the single-model deployment footprint.
      </Prose>

      <H3>Tree speculation</H3>

      <Prose>
        Sequential speculation proposes a single chain of K tokens. Tree speculation proposes a branching tree of possible continuations and verifies all paths in one forward pass. The target model processes every branch in parallel, then the algorithm walks the tree greedily along the path of accepted tokens, stopping at the first rejection in each branch. More tokens are verified per target-model pass — the same memory bandwidth now covers a wider search — and the expected number of accepted tokens per pass increases by another 30–50% over sequential speculation. SpecInfer, Medusa, and EAGLE all support tree speculation; the implementation complexity is higher (batching a tree through the attention mechanism requires careful position encoding and masking), but the throughput payoff is real and most production implementations default to it.
      </Prose>

      <H2>Cost and limits</H2>

      <Prose>
        Speculation is not free. Running a draft model alongside the target consumes additional GPU memory — the draft model's weights, its KV cache, its activation memory. At low concurrency, this is usually acceptable: the target model is memory-bandwidth-bound, so there is genuinely idle compute available for the draft. At high concurrency, the picture changes. With many requests batched together, the target model's compute cores are busy; the memory bandwidth is saturated; there is no slack to exploit. The draft model now adds latency and memory without delivering proportional throughput gains. The speedup degrades gracefully as batch size grows, reaching 1× — no speedup, only overhead — before any negative impact on latency.
      </Prose>

      <Callout accent="gold">
        Speculative decoding is a trick for turning idle GPU compute into effective throughput. At high batch sizes, there is no idle compute to steal — the trick stops working.
      </Callout>

      <Prose>
        The sweet spot is clear from the analysis: interactive chat at low to moderate concurrency, single-user or lightly loaded serving, any workload where the decode GPU is memory-bandwidth-bound rather than compute-bound. That description fits most interactive LLM deployments at the scale of individual users or small teams. It fits poorly for high-throughput batch inference pipelines where maximizing concurrent requests is the goal.
      </Prose>

      <H3>What the math guarantees</H3>

      <Prose>
        The guarantee is worth stating precisely, because it is unusually strong. Well-implemented speculative decoding produces output that is statistically identical to direct sampling from the target model. Not approximately identical. Not identical in expectation. Identical in distribution, provably, by construction of the acceptance-rejection step. A user running speculative decoding gets exactly the outputs they would have gotten without it — the same token probabilities, the same stochastic behavior, the same response to temperature and sampling parameters. This is a rare case where an inference optimization is a strict free lunch on the quality side. The tradeoffs are entirely on the cost side: memory for the draft model, engineering complexity to implement the verification loop, and a speedup that varies with acceptance rate and batch size rather than being a fixed multiplier.
      </Prose>

      <Prose>
        Every major inference engine — vLLM, TensorRT-LLM, SGLang, TGI — ships speculative decoding as a built-in option. It has become table stakes for interactive LLM serving, the obvious first thing to enable when latency under load is a constraint and batch size is modest. The next topic in this track turns to another memory-bandwidth lever: prefix caching, which exploits shared context across concurrent requests rather than shared structure within a single decode sequence.
      </Prose>
    </div>
  ),
};

export default speculativeDecoding;
