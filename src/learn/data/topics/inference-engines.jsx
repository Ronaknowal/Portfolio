import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";

const inferenceEngines = {
  title: "Inference Engines & Serving",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        The techniques this section has covered — KV caching, paged attention, continuous
        batching, speculative decoding, prefix caching, constrained decoding — are not
        individual choices a practitioner assembles from scratch. In production, they arrive
        bundled: implemented, tuned, and enabled by default inside inference engines, the
        serving frameworks that sit between model weights and an HTTP endpoint. Picking a
        technique is mostly picking a framework. This topic is a tour of the current landscape
        and the decision surface among them.
      </Prose>

      <H2>What an inference engine is</H2>

      <Prose>
        A useful mental model: the model is the parameters; the engine is everything else.
        Concretely, an inference engine is responsible for scheduling requests across one or
        more GPUs, managing KV cache memory so that it is never statically over-allocated
        and never fragmented into waste, running continuous batching so that hardware
        utilization stays high across variable-length requests, executing the attention kernel
        (PagedAttention or an equivalent block-sparse variant), handling request admission and
        prioritization under load, running speculative decoding with a draft model, enforcing
        structured output constraints, streaming partial responses over HTTP, and sharding
        the model across multiple devices when a single GPU is insufficient. A production
        serving endpoint is that entire stack, not just the transformer forward pass.
      </Prose>

      <Prose>
        This matters because each of those responsibilities involves non-trivial engineering.
        KV cache memory management alone — paging activations in and out as request lifetimes
        vary, avoiding fragmentation without copying, supporting prefix sharing across requests
        — is a systems problem that took a Berkeley research group a full paper to formalize
        correctly. Continuous batching requires the scheduler to track per-sequence completion
        independently and insert new sequences into a batch mid-iteration, which demands deep
        integration with the attention kernel. None of this is code you want to write once per
        model deployment. It belongs in a shared layer that every deployment reuses.
      </Prose>

      <H2>The major frameworks</H2>

      <Prose>
        Four open-source stacks dominate production LLM serving in 2025, with meaningfully
        different design points.
      </Prose>

      <Prose>
        <strong>vLLM</strong> (UC Berkeley, 2023) is the canonical reference implementation
        of PagedAttention and continuous batching. It is the framework that first demonstrated
        both of those techniques working together at production scale, and it has grown into
        the largest community and broadest model support in the open-source ecosystem. The
        API surface is deliberately close to OpenAI's — dropping it in as a replacement for
        a managed endpoint requires changing a base URL and little else. vLLM is Python-centric
        throughout: approachable to deploy, simple to configure, and the reasonable default
        for any team that has not yet identified a specific reason to deviate.
      </Prose>

      <Prose>
        <strong>SGLang</strong> (LMSys, 2024) builds on the same PagedAttention foundation
        but adds RadixAttention: a prefix caching scheme that shares KV cache blocks not just
        within a single request's system prompt but across the radix tree of all cached
        prefixes, matching much longer shared prefixes than vLLM's block-level approach. The
        practical advantage surfaces in agent and reasoning workloads, where many requests
        share a long common prefix — a system prompt, a tool schema, or a reasoning chain
        preamble — and the savings from avoiding redundant prefill accumulate fast. SGLang
        also adds structured-output primitives at the engine level, tighter than
        post-hoc constraint enforcement. For standard serving it is competitive with vLLM;
        for heavy prefix-sharing workloads it is often measurably faster.
      </Prose>

      <Prose>
        <strong>TensorRT-LLM</strong> (NVIDIA) takes a different approach entirely. Rather than
        a Python-first serving layer built on top of PyTorch, TensorRT-LLM compiles model
        graphs into highly optimized CUDA kernels at model-load time. The result is the highest
        throughput available on NVIDIA hardware for models that are well-supported — typically
        the last 20–30% of tokens-per-second that other frameworks leave on the table. The
        cost is operational: adding a new model architecture requires writing and validating
        a new plugin, upgrades lag model releases, and the compilation step adds friction to
        the deployment pipeline. Enterprise teams that have committed to a single model at scale
        and can absorb the engineering effort frequently choose TensorRT-LLM for that final
        throughput margin. Teams that need to track a fast-moving model landscape usually do not.
      </Prose>

      <Prose>
        <strong>TGI (Text Generation Inference)</strong> (HuggingFace) is the original production
        serving stack for open models. It was the first framework to support continuous batching
        in a widely deployed form, and it remains tightly integrated with the HuggingFace Hub:
        any model on the Hub that lists TGI support can be served with a single container pull.
        On peak throughput benchmarks TGI has fallen slightly behind vLLM, and its prefix
        caching and speculative decoding support is less complete, but for teams already operating
        inside the HuggingFace ecosystem it remains a low-friction choice.
      </Prose>

      <H2>How they differ operationally</H2>

      <Prose>
        The feature matrix has converged substantially. As of 2025, all four stacks implement
        the core techniques this section covered: paged attention, continuous batching,
        speculative decoding, and some form of prefix caching.
      </Prose>

      <CodeBlock>
{`framework       paged-attn  cont-batch  spec-decode  prefix-cache  ease-of-deploy
vLLM                ✓            ✓            ✓            ✓           high
SGLang              ✓            ✓            ✓            ✓ (radix)   high
TensorRT-LLM        ✓            ✓            ✓            ✓           low
TGI                 ✓            ✓            partial      partial     high`}
      </CodeBlock>

      <Prose>
        The differences that remain are increasingly about tail-latency distribution rather
        than feature presence, kernel-level optimizations that only matter at a specific
        hardware and traffic profile, and integration surface — how well the framework fits
        into the rest of a team's infrastructure. The question "does it support speculative
        decoding" is almost never the deciding factor anymore. The question "how does its
        P99 latency behave under bursty traffic with a 4k-token average context" is a much
        harder one, and the answer varies by workload.
      </Prose>

      <H3>Llama.cpp and the edge</H3>

      <Prose>
        Worth separating from the four above because it solves a different problem. Llama.cpp
        and its ecosystem — Ollama for local management, LM Studio for a GUI layer — target
        single-user or edge deployment on consumer hardware. The primary tool is aggressive
        quantization: 4-bit GGUF weights are standard, with 2-bit and 3-bit variants available
        for tighter memory budgets. A 7B model in Q4 fits in 4 GB of RAM and runs at interactive
        speed on a MacBook or a mid-range consumer GPU. The concurrency assumption is low: one
        user, maybe a small team, not thousands of simultaneous requests. None of the
        continuous batching or paged attention logic matters much at that scale.
      </Prose>

      <Prose>
        Llama.cpp is not a competitor to vLLM in the way the four frameworks above are
        competitors to each other. It is an alternative for a categorically different
        problem — running inference locally with no cloud dependency, under a strict memory
        budget, at low concurrency. If that is the problem, it is the right answer.
        If the problem involves serving thousands of requests per second to external users,
        it is not the answer and the comparison is a category error.
      </Prose>

      <H3>Closed-source serving — a brief note</H3>

      <Prose>
        OpenAI, Anthropic, Google, and DeepMind run their own internal serving stacks.
        Public details are sparse by design. From what has been disclosed in papers, blog
        posts, and inferences from latency and pricing data: these systems implement
        disaggregated prefill/decode architectures, where the prefill phase (processing the
        input prompt) and the decode phase (generating tokens one at a time) run on separate,
        specialized hardware pools. They use bespoke speculation models trained specifically
        for the production model, multi-datacenter routing with geographic affinity, and
        caching layers that extend well beyond the per-request prefix caching described in
        the prefix caching topic. The open-source stacks have closed roughly 80% of the gap
        over 2023–2025 — the era when vLLM's release made PagedAttention a commodity. The
        remaining 20% is workload-specific tuning at hyperscale: optimizations that only
        pay off when you are running millions of requests per hour on a specific model,
        with full visibility into the traffic distribution and the hardware. That gap will
        narrow further, but it lives inside frontier labs for now.
      </Prose>

      <H2>Choosing an engine</H2>

      <Prose>
        The decision is mostly about matching the framework's defaults to the workload's
        dominant cost.
      </Prose>

      <Prose>
        For most open-model production serving in 2025, vLLM is the reasonable default. The
        community is large, model support is broad, the deployment story is simple, and the
        performance is strong enough that it will not be the bottleneck in most systems.
        For agent or reasoning workloads with heavy prefix sharing — a system prompt that
        is thousands of tokens long and shared across millions of requests — SGLang's radix
        attention gives a meaningful edge in both latency and memory efficiency, and it
        is worth benchmarking. For pure throughput on NVIDIA hardware at scale, with a stable
        model and an engineering team willing to invest in the deployment pipeline, TensorRT-LLM
        unlocks the last fraction of tokens-per-second that other frameworks cannot reach.
        For edge or single-user deployment, llama.cpp. For teams already inside the
        HuggingFace ecosystem who want the path of least resistance, TGI is still viable —
        though it is losing ground to vLLM on the features that matter most.
      </Prose>

      <Callout accent="gold">
        The best engine is the one whose defaults match your workload and whose feature
        roadmap matches your 12-month needs. All four major stacks are "fast enough"
        for most purposes.
      </Callout>

      <H3>A minimal deployment recipe</H3>

      <Prose>
        The gap between "model weights" and "production endpoint" is smaller than it has
        ever been. For vLLM, one shell command and a handful of client lines:
      </Prose>

      <CodeBlock language="python">
{`# vLLM server — minimal OpenAI-compatible endpoint
# pip install vllm; vllm serve meta-llama/Meta-Llama-3-8B-Instruct

from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Explain RLHF in one paragraph."}],
)
# Out of the box: continuous batching, paged attention, prefix caching enabled.`}
      </CodeBlock>

      <Prose>
        One command, a handful of lines of client code — this is what 10,000+ tokens/second
        of inference looks like in 2025. The techniques covered throughout this section are
        running silently behind that response: the KV cache is paged, the batching is
        continuous, the memory is not over-reserved, the prefix is checked before the
        prefill begins. The practitioner writing the client code does not configure any
        of it explicitly because the engine's defaults are the distillation of several years
        of research into what most workloads need.
      </Prose>

      <H2>What comes next — the system design angle</H2>

      <Prose>
        This section has covered the single-node engine problem: how one serving instance,
        given model weights and a stream of incoming requests, can extract maximum throughput
        at minimum latency. The next section — AI Inference System Design & Architecture —
        covers the distributed problem: how fleets of those instances are orchestrated,
        how routing decisions are made across instances, how prefill and decode are
        disaggregated across specialized hardware pools, how autoscaling responds to traffic
        bursts, and how cost is managed at the organization level when inference is the
        dominant line item on the infrastructure budget. The inference engine is the building
        block. The system is what gets built with it.
      </Prose>

      <Prose>
        Four years ago, serving an LLM at production latency was an open research problem.
        The papers that introduced continuous batching and PagedAttention were describing
        things that did not yet exist in deployable form. Today, those techniques are the
        default configuration of every major open serving framework — enabled on first run,
        not after tuning. The practitioner's job has shifted from implementing them to
        selecting among mature implementations and configuring the margins. That consolidation
        happened faster than most expected, and the same pattern — a research breakthrough
        becoming an opinionated default inside a commodity framework within 18 months — is
        likely to repeat for the distributed inference techniques the next section covers.
        The remaining hard problems in inference are real, but they are increasingly the
        problems of scale and coordination rather than correctness. And correctness, in this
        domain, is now largely solved.
      </Prose>
    </div>
  ),
};

export default inferenceEngines;
