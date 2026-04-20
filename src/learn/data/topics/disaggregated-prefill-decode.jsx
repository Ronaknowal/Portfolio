import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream, Plot } from "../../components/viz";

const disaggregatedPrefillDecode = {
  title: "Disaggregated Prefill & Decode",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Prefill is compute-bound. The GPU processes a prompt of several thousand tokens in a single parallel forward pass — matrix multiplications at full utilization, arithmetic intensity high, memory bandwidth secondary. Decode is memory-bandwidth-bound. One token per step, each reading the full KV cache out of HBM, doing a sliver of arithmetic, then waiting for the next step. These two phases compete for fundamentally different hardware resources, and when they share the same GPU, neither gets what it needs.
      </Prose>

      <Prose>
        The interleaving compounds this. A long prefill does not just occupy compute while running; it blocks every in-progress decode until it finishes. Thirty-two concurrent users chatting away all experience a stutter whenever a new long prompt arrives, because the shared GPU cannot emit their next token until the prefill is done. Disaggregated serving answers this by running prefill and decode on separate GPU pools, each tuned for its own phase. The architectural shift is significant. The latency wins are real, especially at the tail.
      </Prose>

      <H2>The interleaving problem</H2>

      <Prose>
        In a co-located stack — the vLLM default, the configuration most deployments start with — prefill and decode share a single GPU and run in the same iteration loop. When a new request arrives with a 4,000-token prompt, the scheduler processes its prefill in one pass. On a typical A100 at a 70B model, that pass takes roughly 200ms. During those 200ms, every decode iteration stalls. No new tokens are emitted for any in-flight conversation. A serving setup with 32 concurrent chats experiences 32 simultaneous stutters per new high-complexity arrival.
      </Prose>

      <Prose>
        At low concurrency this barely matters — the arrivals are infrequent enough that the interruptions blend into background jitter. At high concurrency, with dozens of long-prompt requests entering per second, the decode stalls begin to dominate p99 latency. The serving stack looks healthy on mean metrics; the tail is what breaks. Users notice not the average experience but the worst one, and in a co-located setup, the worst one is coupled directly to the longest concurrent prefill.
      </Prose>

      <StepTrace
        label="co-located vs disaggregated — a long prefill arrives"
        steps={[
          { label: "t=0 — 8 decodes in flight on 1 gpu", render: () => (
            <TokenStream tokens={[
              { label: "decode 1", color: "#4ade80" },
              { label: "decode 2", color: "#4ade80" },
              { label: "...", color: "#555" },
              { label: "decode 8", color: "#4ade80" },
            ]} />
          ) },
          { label: "co-located: new 4k prefill arrives — stall", render: () => (
            <TokenStream tokens={[
              { label: "prefill (4k)", color: "#c084fc" },
              { label: "8 decodes BLOCKED", color: "#f87171" },
            ]} />
          ) },
          { label: "disaggregated: prefill runs on dedicated gpu", render: () => (
            <TokenStream tokens={[
              { label: "prefill gpu: 4k prefill", color: "#c084fc" },
              { label: "decode gpu: 8 decodes continue", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      <H2>The architecture</H2>

      <Prose>
        Disaggregated serving splits the serving fleet into two pools. Prefill workers are provisioned for compute: high FLOP throughput, full KV write bandwidth, fast tensor cores. Decode workers are provisioned for memory bandwidth: high HBM bandwidth per dollar, ability to hold large KV caches for many concurrent sequences. A request arrives at a router, which dispatches it to an available prefill worker. That worker runs the full prompt through the model, computing and writing out the KV cache for every prompt token. When the prefill is done, the KV blocks are transferred to a selected decode worker. The decode worker picks up from the transferred cache and begins emitting tokens, streaming them back to the user.
      </Prose>

      <Prose>
        The prefill worker is now free to take the next request. The decode worker runs the growing sequence until completion, at which point its KV blocks are freed back into the pool. The two workers never contend for the same GPU — their scheduling problems are independent. A surge of long-prompt requests saturates the prefill pool but does not touch the decode pool. A surge of lengthy conversations that churn through many tokens saturates the decode pool but leaves the prefill pool idle. Each can be scaled independently to match the actual demand pattern of the workload.
      </Prose>

      <H3>The transfer problem</H3>

      <Prose>
        The KV cache transfer is the new critical path. For a 70B model at BF16 with 8 KV heads and a 4,000-token prompt, the cache for a single request runs to roughly 1.3 GB. At 32k tokens it is over 10 GB. Sending that across a standard 100 Gbps Ethernet link takes hundreds of milliseconds — longer than the prefill itself on many prompts — and entirely defeats the purpose of the separation. Three requirements emerge: a fast interconnect (NVLink or InfiniBand between machines in the same pod), block-level incremental transfer (begin shipping completed KV blocks while the prefill worker is still computing later blocks, overlapping transfer with compute), and a compact representation (FP8 or INT8 quantized caches halve or quarter the transfer volume with minimal quality loss).
      </Prose>

      <Prose>
        Production implementations converge on this pattern. Mooncake (Moonshot AI, 2024) uses a disaggregated KV cache store backed by RDMA over InfiniBand between prefill and decode nodes. Splitwise (Microsoft Research, 2024) characterizes the optimal prefill-to-decode ratio at various request mix profiles and demonstrates 2–3× improvement in p99 TTFT. DistServe (Zhong et al., 2024) formalizes the latency model and shows that the break-even point — the concurrency threshold above which disaggregation wins — depends on prompt length distribution and inter-node bandwidth. Below NVLink-class interconnects, the transfer overhead can cancel the scheduling benefit entirely, which is why the pattern does not appear in cloud deployments using plain PCIe or Ethernet between nodes.
      </Prose>

      <CodeBlock language="python">
{`# Simplified disaggregated serving loop

async def handle_request(request):
    # 1. Pick a prefill worker
    prefill_worker = prefill_pool.select()

    # 2. Pick a decode worker with capacity
    decode_worker = decode_pool.select(
        kv_cache_bytes_needed=estimate_kv_size(request)
    )

    # 3. Prefill computes KV cache, streams it to decode worker's pool
    kv_block_handles = await prefill_worker.prefill(
        request.prompt,
        dest_worker=decode_worker.id,
    )

    # 4. Decode picks up from the transferred KV cache
    async for token in decode_worker.decode(kv_block_handles):
        yield token`}
      </CodeBlock>

      <H2>Why this wins — p99 latency</H2>

      <Prose>
        Co-located serving has two competing failure modes. High prefill traffic stalls decode, spiking p99 TTFT and inter-token latency for in-flight chats. High decode load — many long conversations simultaneously — consumes KV cache memory and leaves little headroom for new prefills, increasing queue wait for new arrivals. Each failure mode is a function of the other's traffic; the scheduling problem is entangled. Disaggregation decouples them. The decode pool's p99 is no longer a function of prefill arrival rate. The prefill pool's queue depth is no longer a function of how many tokens are actively being decoded.
      </Prose>

      <Prose>
        Published results from Splitwise, DistServe, and Mooncake consistently report 2–5× improvement in p99 time-to-first-token at the same total GPU count, under mixed workloads with high-variance prompt lengths. Mean TTFT moves less — it was already decent on average — but p95 and p99 fall dramatically. Decode throughput (tokens per second per user mid-stream) is also more stable, because the decode pool no longer gets stalled behind prefills. The gains are not uniformly distributed: short-prompt requests see the least improvement, since their prefills were never the problem. Long-prompt requests see the most; their prefills no longer starve the decode pipeline.
      </Prose>

      <Plot
        label="ttft p99 under mixed load (illustrative)"
        width={520}
        height={240}
        xLabel="concurrent requests"
        yLabel="p99 ttft (ms)"
        series={[
          { name: "co-located", points: [[8, 300], [16, 600], [32, 1400], [64, 3200], [128, 8000]] },
          { name: "disaggregated", points: [[8, 250], [16, 400], [32, 700], [64, 1200], [128, 2100]] },
        ]}
      />

      <H2>The costs</H2>

      <Prose>
        None of this is free. First, pool sizing: disaggregation introduces a new operational problem that does not exist in a co-located stack. How many prefill workers and how many decode workers? The optimal ratio depends on the prompt length distribution, the average output length, and the concurrency mix — all of which vary by time of day and user cohort. Overprovision the prefill pool and you waste compute on idle workers that could have been decode capacity. Overprovision decode and you bottleneck on prefill throughput. Getting the ratio right requires traffic modeling and likely autoscaling, both of which add operational complexity.
      </Prose>

      <Prose>
        Second, the KV transfer latency adds to TTFT for every request, not just the ones that would have been stalled. On short prompts at low concurrency — where co-located serving barely stalls at all — the transfer overhead can make disaggregated serving strictly worse. The pattern has a break-even point, and below it the added plumbing hurts more than the decoupling helps. Third, NVLink-class interconnects are not universal. Between machines connected only by standard Ethernet or PCIe, transfer bandwidth is too low for production usefulness at long contexts. The architectural benefit is conditional on the hardware topology. Finally, engineering complexity is substantial. Most open-source serving stacks — vLLM, SGLang — only added disaggregated-prefill support in 2024, and it is still not the default configuration for either. Debugging a failure in a disaggregated stack requires reasoning about two independent pools, a transfer layer, and a router simultaneously, which raises the operational burden considerably compared to a single co-located process.
      </Prose>

      <H3>Prefill-prefill disaggregation — the newer variant</H3>

      <Prose>
        A separate direction extends the idea into the prefill phase itself. For very long contexts — 32k tokens or more — a single prefill pass can take seconds and constitutes the dominant latency cost. Chunked prefill, already supported in co-located stacks, mitigates this by breaking the prefill into segments interleaved with decode work. A more aggressive variant, prefill-prefill disaggregation, shards the prompt across multiple GPUs: the first GPU handles the initial chunks, the resulting partial KV state is transferred to subsequent GPUs, and the remainder of the sequence is processed in parallel across the pool. This reduces wall-clock TTFT for very long prompts at the cost of inter-GPU transfer overhead for each shard boundary. SGLang research prototypes support this pattern; FastServe (Wu et al.) formalizes the scheduling theory behind it. As context windows push past 100k tokens in production, splitting the prefill work itself becomes the next bottleneck after prefill-decode separation has been achieved.
      </Prose>

      <H2>When is it worth it?</H2>

      <Prose>
        Disaggregation has a break-even point below which it is a net loss. The pattern pays off under a specific combination of conditions: high concurrency (more than 32 sustained simultaneous requests), high-variance prompt lengths (a mix of short and long prompts, not uniform), and strict p99 SLOs where TTFT must stay below 500ms even at the tail. When all three are present, the decoupling of scheduling problems is valuable enough to justify the transfer overhead, the pool-sizing complexity, and the infrastructure requirements.
      </Prose>

      <Prose>
        Outside those conditions — a deployment with mostly short prompts, modest concurrency, or relaxed latency requirements — a well-tuned co-located vLLM with chunked prefill enabled is simpler to operate and delivers comparable mean performance. The distinction matters for capacity planning: moving to disaggregated serving prematurely adds operational burden without corresponding latency benefit, while waiting too long to adopt it means users experience avoidable tail spikes under load.
      </Prose>

      <Callout accent="gold">
        Disaggregation pays for itself when p99 tail latency is expensive — that is, when users leave because a subset of requests feel slow, not because the average is slow.
      </Callout>

      <Prose>
        Disaggregated prefill/decode is the first major post-vLLM architectural shift in LLM serving. It is now standard at hyperscaler scale — OpenAI, Anthropic, and Google all operate some form of separated prefill and decode infrastructure. The open-source equivalent, led by Mooncake and DistServe, is catching up. The next topic in this section covers the three kinds of caches that sit alongside this serving layer: the KV cache pool, the prefix cache, and the emerging semantic cache, and how they interact when disaggregation is in place.
      </Prose>
    </div>
  ),
};

export default disaggregatedPrefillDecode;
