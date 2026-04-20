import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream, Plot } from "../../components/viz";

const continuousBatching = {
  title: "Continuous Batching & PagedAttention",
  readTime: "13 min",
  content: () => (
    <div>
      <Prose>
        Two ideas introduced in vLLM (Kwon et al., 2023) reshaped LLM serving: continuous batching and PagedAttention. Together they increase throughput on a given GPU by roughly 10× compared to the 2022-era status quo, with no change to the model itself. Every production serving stack today — vLLM, TGI, TensorRT-LLM, SGLang — implements them or close equivalents. Neither is a new mathematical technique; both are engineering decisions that borrowed from systems software and applied them to a domain that had been treating GPU memory as if it were unlimited and generation time as if it were fixed. The gap they closed was not between what the model could do and what the hardware allowed, but between what the hardware could do and what the software was actually using.
      </Prose>

      <H2>The problem with static batching</H2>

      <Prose>
        Naive batched inference works like this: collect B prompts, run them through the model in parallel, wait for all B to finish, return the results. The parallelism is real and the throughput improvement over serial processing is substantial — but the approach has a structural flaw that becomes severe on realistic workloads.
      </Prose>

      <Prose>
        Response lengths vary wildly. A user asking for a one-sentence summary gets thirty tokens; a user asking for a code explanation might get five hundred. If you batch those requests together, the batch cannot return until the longest sequence finishes. The GPU computes tokens for that one long request while the three short-request slots sit idle, allocated and warm, contributing nothing. In a heterogeneous workload where some responses are ten tokens and some are five hundred, the average slot utilization during the tail of a batch can fall below thirty percent. The GPU is paying full memory and scheduling cost for every slot, but getting useful work out of fewer than a third of them. The hardware looks busy; the throughput numbers tell a different story.
      </Prose>

      <StepTrace
        label="static batching — gpu idling on mixed-length outputs"
        steps={[
          { label: "t=0 — all B=4 prompts active", render: () => (
            <TokenStream tokens={[
              { label: "req 1 (500)", color: "#4ade80" },
              { label: "req 2 (500)", color: "#4ade80" },
              { label: "req 3 (500)", color: "#4ade80" },
              { label: "req 4 (500)", color: "#4ade80" },
            ]} />
          ) },
          { label: "t=30 — 3 finished, 1 still running", render: () => (
            <TokenStream tokens={[
              { label: "req 1 done", color: "#555" },
              { label: "req 2 done", color: "#555" },
              { label: "req 3 done", color: "#555" },
              { label: "req 4 (500)", color: "#4ade80" },
            ]} />
          ) },
          { label: "t=30..500 — 3 slots wasted", render: () => (
            <TokenStream tokens={[
              { label: "idle", color: "#555" },
              { label: "idle", color: "#555" },
              { label: "idle", color: "#555" },
              { label: "req 4 still going", color: "#4ade80" },
            ]} />
          ) },
        ]}
      />

      <H2>Continuous batching — the fix</H2>

      <Prose>
        Instead of committing to a fixed batch for the duration of generation, the scheduler rebuilds the active batch every iteration. When a sequence finishes — its last token is the end-of-sequence token, or it has hit the user's requested length — its slot immediately fills with a new pending request from the queue. The new request does not wait for the current batch to drain. It joins the very next forward pass. When a new request arrives mid-generation, it enters the next iteration's prefill phase and then streams alongside the already-running decode sequences. The GPU is never waiting for the longest sequence to clear before doing useful work; it is continuously at or near its target batch size.
      </Prose>

      <Prose>
        GPU utilization on realistic mixed workloads climbs from that thirty-percent floor to eighty percent or above. The gains are not evenly distributed: short requests benefit most, because they no longer have to wait behind longer siblings before the next wave of requests can begin. Tail latency for short requests drops substantially. The overall throughput improvement — requests per second at a given quality-of-service target — is typically two to four times in isolation, before PagedAttention adds its own layer on top.
      </Prose>

      <Prose>
        The implementation cost is meaningful but not enormous. The scheduler needs to maintain a queue of pending requests and a set of active sequences, pack them into a unified batch tensor each step — including mixing sequences that are in different positions in their generation — and correctly attribute which KV cache slots belong to which sequence. The batch tensor at each step may contain sequences that have generated one token and sequences that have generated four hundred; the model processes all of them in the same matrix multiply, with padding or careful masking to prevent sequences from attending to each other's tokens. Relative to the throughput gain, the bookkeeping is not expensive. The main engineering challenge is that the scheduler now needs to reason about KV cache capacity — it cannot add a new request to the active set if there is no room for its cache — which is exactly where PagedAttention comes in.
      </Prose>

      <H3>The prefill-decode interaction</H3>

      <Prose>
        A detail that makes continuous batching implementations tricky in practice: prefill and decode are not the same kind of workload, and mixing them in the same batch has subtle performance implications. Prefill is compute-bound. The model processes all prompt tokens in a single forward pass — one large matrix multiplication over the full prompt length, tensor cores spinning at high utilization, arithmetic intensity high. Decode is memory-bandwidth-bound. One new token at a time, each step reading the full KV cache from HBM and doing very little arithmetic against it.
      </Prose>

      <Prose>
        When a long prefill joins a batch of decoding sequences, the prefill computation dominates that iteration. The decoding sequences have to wait for the prefill to complete before their tokens are emitted. A five-hundred-token prefill can block decoding sequences for hundreds of milliseconds — which shows up as a latency spike for users who are mid-conversation and expecting near-realtime streaming. Modern schedulers address this by chunked prefill: rather than processing all prompt tokens in a single pass, the prefill is split into segments of a few hundred tokens each, interleaved across multiple iterations with the ongoing decode work. Tail latency for decoding users stays bounded; prefill throughput is slightly reduced but time-to-first-token for the new request stays acceptable. An alternative at the infrastructure level is disaggregated prefill-decode serving: dedicated GPU clusters handle prefill only, and the resulting KV cache is transferred over the network to decode GPUs that handle streaming. This is covered in the system design section of this track; continuous batching within a single node is the prerequisite mental model.
      </Prose>

      <H2>PagedAttention — the memory fix</H2>

      <Prose>
        Continuous batching maximizes GPU compute utilization. PagedAttention maximizes GPU memory utilization, which is the other side of the same coin — you cannot run more concurrent sequences if you do not have memory to store their KV caches.
      </Prose>

      <Prose>
        Before vLLM, KV caches were contiguous tensors per sequence. When a request arrived, you allocated a slab of GPU memory sized for the maximum possible output length and pinned it to that sequence until generation finished. This created two compounding problems. Internal fragmentation: a request that generates two hundred tokens when you allocated space for two thousand leaves eighteen hundred slots empty but allocated — that memory is gone for the duration of the request. External fragmentation: if you have enough total free memory for a new long request but that memory is scattered across many small gaps between existing allocations, you cannot fit the new contiguous slab and the request has to wait or be evicted. In practice, sixty to eighty percent of GPU KV cache memory was wasted at any given time through these two mechanisms combined. The hardware you paid for was sitting idle, not because there were no requests, but because the memory allocator was making them wait.
      </Prose>

      <Prose>
        PagedAttention borrows the OS paging model and applies it directly. Break the KV cache into fixed-size blocks — sixteen tokens per block is a common choice — and manage a global pool of these blocks. Each sequence gets a page table: a list of block IDs in logical order, mapping logical token positions to physical blocks that can sit anywhere in GPU memory. A sequence that generates five hundred tokens uses thirty-two blocks; when it finishes, those thirty-two blocks return to the pool immediately and any incoming request can claim them. No slab, no contiguous allocation, no fragmentation. The attention kernel is given the page table and gathers KV vectors from discontiguous blocks on every step, making the scattered storage transparent to the rest of the computation.
      </Prose>

      <CodeBlock language="python">
{`# Simplified PagedAttention data structures

BLOCK_SIZE = 16  # tokens per block

class KVBlockPool:
    def __init__(self, num_blocks, num_heads, head_dim, num_layers):
        # One big allocation, indexed by block_id
        self.k_cache = torch.zeros(num_blocks, num_layers, num_heads, BLOCK_SIZE, head_dim)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.free_blocks = set(range(num_blocks))

    def allocate(self) -> int:
        return self.free_blocks.pop()

    def free(self, block_id: int) -> None:
        self.free_blocks.add(block_id)

class Sequence:
    def __init__(self, seq_id, pool):
        self.pool = pool
        self.page_table = []  # list of block_ids, in logical order

    def append_token(self, new_k, new_v):
        if len(self.page_table) * BLOCK_SIZE == self.num_tokens:
            self.page_table.append(self.pool.allocate())
        # ... write new_k, new_v into the correct slot of the last block`}
      </CodeBlock>

      <H3>Copy-on-write for shared prefixes</H3>

      <Prose>
        Because blocks are individually addressable and the KV vectors in a block are a deterministic function of the token IDs and model weights, multiple sequences can safely share the same physical block. If two requests share a system prompt — which in production deployments they almost always do, because the same system prompt is prepended to every call — their KV caches for the shared prefix portion are bit-for-bit identical. With contiguous per-sequence allocation, this observation was academic: sharing a slab across sequences was not architecturally possible. With a paged cache, the scheduler can hash each block of prompt tokens as they arrive, check whether a matching block is already in the pool, and if so simply point the new sequence's page table at the existing physical block rather than allocating and computing a new one.
      </Prose>

      <Prose>
        Write divergence is handled with copy-on-write: the first time a sequence needs to write to a block it shares with another sequence, the allocator makes a private copy for the writing sequence and updates its page table entry. Before that write, both sequences point at the same physical memory. For the common case — a long shared system prompt followed by per-user generation — this means the system-prompt blocks are computed once, cached, and reused across every concurrent session for as long as those blocks stay warm. The memory win for high-volume deployments with hundred-token or thousand-token system prompts is substantial. Prefix caching is covered in depth later in this section; the mechanism is entirely built on paging being in place as the substrate.
      </Prose>

      <H2>The CUDA kernel implications</H2>

      <Prose>
        PagedAttention requires a custom attention kernel. Standard attention implementations — cuBLAS-backed matrix multiplies, FlashAttention, most of what ships in PyTorch — assume that the key and value tensors for a sequence are laid out contiguously in memory. They issue coalesced reads against a dense matrix and depend on that layout for their performance. A paged KV cache is not contiguous. The keys for a five-hundred-token sequence are spread across thirty-two blocks that may be scattered anywhere in the pool, and the attention kernel has to follow a page table to gather them before computing attention scores.
      </Prose>

      <Prose>
        vLLM shipped its own CUDA kernel that does exactly this gather: given a page table, block size, and the physical block pool, it reads keys and values block by block into shared memory and computes the attention scores against the current query. TensorRT-LLM, FlashAttention 2 and 3, and XFormers have since added paged variants, so the ecosystem support is now broad. The kernel overhead relative to standard contiguous attention is small on modern implementations — a few percent, not a meaningful fraction of total step time. The benefit — near-zero memory fragmentation and the ability to serve far more concurrent sequences on the same hardware — overwhelms the kernel cost by orders of magnitude.
      </Prose>

      <H2>What this means for throughput</H2>

      <Prose>
        The combined effect of continuous batching and PagedAttention is best understood through concurrency. What limits the number of requests you can serve simultaneously is, first, KV cache memory (without PagedAttention, fragmentation eats most of it), and second, GPU compute utilization (without continuous batching, idle slots waste most of it). Eliminating both constraints lets the GPU serve a substantially larger active batch at any given moment, and total throughput — tokens per second summed across all concurrent requests — scales nearly linearly with batch size up to the point where genuine hardware limits are hit.
      </Prose>

      <Plot
        label="throughput vs concurrency (illustrative, 13B model, single a100)"
        width={520}
        height={240}
        xLabel="concurrent requests"
        yLabel="tokens/sec (total across batch)"
        series={[
          { name: "static batching", points: [[1, 40], [4, 110], [8, 140], [16, 150], [32, 155]] },
          { name: "+ continuous batching", points: [[1, 40], [4, 140], [8, 220], [16, 300], [32, 340]] },
          { name: "+ pagedattention", points: [[1, 40], [4, 150], [8, 260], [16, 420], [32, 680]] },
        ]}
      />

      <Prose>
        The static batching curve flattens quickly. GPU memory fills with fragmented, partially-used cache slabs; only a handful of sequences can be active; the throughput ceiling is hit at low concurrency. Continuous batching alone raises the ceiling — idle slots are reclaimed and refilled — but KV cache fragmentation still limits how many sequences can be in-flight simultaneously, and the ceiling appears again. Adding PagedAttention breaks the memory bottleneck: the pool is used at near-full efficiency, active batch size grows with concurrency, and throughput keeps climbing until compute or bandwidth genuinely saturates. At 32 concurrent requests on a single A100, the combined system delivers roughly 4× the throughput of continuous batching alone and roughly 4.4× static batching. That ratio holds across a wide range of model sizes and hardware; the exact numbers are illustrative, but the shape is structural.
      </Prose>

      <H2>What still limits you</H2>

      <Prose>
        Continuous batching and PagedAttention remove the avoidable bottlenecks — GPU time wasted waiting for the slowest sequence, GPU memory wasted on fragmented allocations. What remains after they are in place are genuine limits rooted in hardware and physics. Total KV cache capacity still bounds how many sequences can be concurrently active; you can use it efficiently now, but you cannot use more of it than exists. Prefill-decode interaction still bounds tail latency for decoding users when a heavy prefill enters the scheduler; chunked prefill mitigates this but does not eliminate it. The attention kernel's read pattern — streaming the full KV cache from HBM on every decode step — still bounds peak decode throughput as sequence length grows, because HBM bandwidth is finite and decode is bandwidth-limited. At very long contexts, the memory wall becomes a bandwidth wall and the tricks that solve fragmentation do not help.
      </Prose>

      <Callout accent="gold">
        vLLM's real contribution wasn't one clever kernel. It was treating LLM serving as a memory-management problem rather than a compute-management one — and then building the abstractions (pages, page tables, pools) that such a reframing needs.
      </Callout>

      <H2>Closing</H2>

      <Prose>
        Continuous batching and PagedAttention define the modern serving baseline. They are the reason a single A100 can profitably serve dozens of concurrent users at useful context lengths on a 13B model, rather than the handful that was possible in 2022. Every other topic in this section — speculative decoding, prefix caching, disaggregated prefill-decode serving, quantized KV caches — assumes them as a substrate and addresses limits that only become the binding constraint once the avoidable waste they fix is gone. The next topic looks at queueing theory: given this serving stack, how do you predict latency distributions and plan capacity for a production workload where arrival rates are spiky and response lengths are unknowable in advance?
      </Prose>
    </div>
  ),
};

export default continuousBatching;
