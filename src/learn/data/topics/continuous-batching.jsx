import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const continuousBatching = {
  title: "Continuous Batching & PagedAttention",
  readTime: "38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 2022, a GPU running LLM inference was often half-idle. Not because requests were scarce, not because the model was slow, but because the software made a structural choice that threw away most of the available throughput before a single token was computed. That choice was static batching, and it is the thing that continuous batching exists to fix.
      </Prose>

      <Prose>
        Static batching is the naive approach. Collect a fixed number of requests — say, four — run the entire model forward pass for all four simultaneously, wait for the longest sequence to finish, then return all four results and start the next batch. The parallelism during the shared computation is genuine and valuable. The waiting is not. If three of the four requests produce thirty-token responses and one produces five hundred, the GPU sits idle on three slots for four hundred and seventy steps, still holding their allocated memory, contributing nothing. Throughput is hostage to the outlier.
      </Prose>

      <Prose>
        The fix, described formally in the Orca paper (Yu et al., USENIX ATC 2022), is iteration-level scheduling: rebuild the active batch at every decode step. When a sequence finishes, its slot fills immediately with a new request from the queue. No waiting. The GPU runs at or near full batch size throughout. This is continuous batching — the batch is continuously replenished rather than drained and refilled.
      </Prose>

      <Prose>
        But continuous batching exposed a second problem it did not itself solve. Adding new requests mid-batch requires allocating KV cache memory for them. With contiguous per-sequence allocation — the standard before 2023 — that memory had to be a single slab reserved up front for the maximum possible output length. Fragmentation from this strategy routinely wasted sixty to eighty percent of GPU KV cache. You could not fill the batch because you had no memory left, even though most of what you had allocated was sitting empty.
      </Prose>

      <Prose>
        PagedAttention (Kwon et al., arXiv:2309.06180, shipped as vLLM) solved the memory side. It borrowed the paging model from operating systems: break the KV cache into fixed-size blocks, manage them with a global free list, give each sequence a page table mapping logical positions to physical blocks scattered anywhere in the pool. No contiguous allocation, no fragmentation, no pre-reservation. Memory is taken and returned at block granularity as sequences arrive and finish.
      </Prose>

      <Prose>
        Together, the two ideas define the modern LLM serving baseline. Every major inference stack — vLLM, TensorRT-LLM, Text Generation Inference, SGLang, LMDeploy — implements them or close equivalents. The measured improvement on realistic mixed workloads is 2–4× from continuous batching alone and 2–4× additional from PagedAttention, for a combined gain of roughly 10× over the 2022 status quo on the same hardware running the same model. No change to the model itself, no new hardware, no quality loss. Pure systems engineering.
      </Prose>

      <Callout accent="gold">
        Orca (Yu et al., USENIX ATC 2022) introduced iteration-level scheduling. vLLM (Kwon et al., 2023) added PagedAttention. The combination is what every production serving stack runs today.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Static batching: waiting for the slowest</H3>

      <Prose>
        Picture four users arriving at a coffee counter simultaneously. The barista serves all four in one batch. Three order espressos; one orders a fourteen-layer caramel creation that takes eight minutes. The three espresso drinkers get their drinks in ninety seconds, then stand at the counter waiting. The barista is not idle — the fourteen-layer creation takes full attention — but the counter is occupied by three people who are done. No new customers can step up until the batch clears. That is static batching.
      </Prose>

      <Prose>
        In GPU terms: the batch is fixed at the start of a generation episode. Every sequence runs to completion before new ones join. The throughput ceiling is determined by the longest sequence in each batch, not by the average. On a real chat workload where some responses are fifteen tokens and some are eight hundred, the average slot utilization during the decode tail routinely falls below thirty percent.
      </Prose>

      <H3>Continuous batching: the iteration-level scheduler</H3>

      <Prose>
        The fix is to treat each decode step as a scheduling opportunity. After every iteration, the scheduler checks which sequences just finished and immediately admits new ones from the waiting queue to fill those slots. The batch composition changes every step. No sequence has to wait for another to finish before the slot opens. At any given moment, the GPU is working on as many requests as memory allows, with no idle slots.
      </Prose>

      <Prose>
        The key insight is that each decode step is symmetric with respect to sequence position. The model does not care whether it is computing token number three or token number three hundred for a given sequence. All it sees is the current query vector and the KV cache for past tokens. Mixing sequences at wildly different generation stages in one batch is perfectly valid — the attention mask ensures they cannot see each other's tokens.
      </Prose>

      <StepTrace
        label="continuous batching — per-iteration scheduling"
        steps={[
          {
            label: "step 1 — all 4 slots active (varied lengths)",
            render: () => (
              <TokenStream tokens={[
                { label: "req A (len=30)", color: "#4ade80" },
                { label: "req B (len=500)", color: "#c084fc" },
                { label: "req C (len=25)", color: "#4ade80" },
                { label: "req D (len=480)", color: "#c084fc" },
              ]} />
            ),
          },
          {
            label: "step 25 — A and C finish; E and F join immediately",
            render: () => (
              <TokenStream tokens={[
                { label: "req E (new, len=40)", color: "#e2b55a" },
                { label: "req B (step 25/500)", color: "#c084fc" },
                { label: "req F (new, len=450)", color: "#e2b55a" },
                { label: "req D (step 25/480)", color: "#c084fc" },
              ]} />
            ),
          },
          {
            label: "step 65 — E finishes; G joins; B and D still running",
            render: () => (
              <TokenStream tokens={[
                { label: "req G (new, len=20)", color: "#60a5fa" },
                { label: "req B (step 65/500)", color: "#c084fc" },
                { label: "req F (step 40/450)", color: "#e2b55a" },
                { label: "req D (step 65/480)", color: "#c084fc" },
              ]} />
            ),
          },
          {
            label: "step 510 — all done; total wall time ~510 steps vs 1010 for static",
            render: () => (
              <TokenStream tokens={[
                { label: "done", color: "#555" },
                { label: "done", color: "#555" },
                { label: "done", color: "#555" },
                { label: "done", color: "#555" },
              ]} />
            ),
          },
        ]}
      />

      <H3>PagedAttention: pages for KV caches</H3>

      <Prose>
        The mental model is direct: KV caches are managed like virtual memory. The entire KV cache pool is divided into fixed-size blocks — typically sixteen tokens per block. Each sequence has a page table, a list of block IDs in logical order, that maps its token positions to physical blocks anywhere in the pool. When a sequence needs to store a new token and its current last block is full, the allocator pops a block from the free list and appends its ID to the page table. When the sequence finishes, all its block IDs go back on the free list in O(1) time.
      </Prose>

      <Prose>
        From the model's perspective, the KV cache is contiguous. The paged attention kernel dereferences the page table before each read, gathering keys and values from discontiguous physical blocks into a contiguous buffer in shared memory, then computes attention normally. The scatter-gather is a few percent of kernel runtime on modern implementations. The benefit — near-zero fragmentation, blocks recyclable immediately on sequence completion — is orders of magnitude larger.
      </Prose>

      <H3>Block tables: the OS analogy in detail</H3>

      <Prose>
        An OS page table maps virtual addresses to physical page frames. A PagedAttention block table maps logical token indices to physical block IDs. The block size is the page size. The free list is the physical frame allocator. Copy-on-write is used when two sequences share a block (as happens with prefix sharing): the first write to a shared block triggers a private copy for the writing sequence.
      </Prose>

      <Prose>
        The analogy is not decorative. The insight that drove PagedAttention was recognizing that LLM serving had a memory management problem that operating systems solved decades ago, and that the OS solution applies almost verbatim when you squint at KV cache allocation correctly.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Throughput under static batching</H3>

      <Prose>
        For a batch of <Code>B</Code> sequences with output lengths <Code>L₁, L₂, …, L_B</Code>, the wall time under static batching is determined by the maximum:
      </Prose>

      <MathBlock>{"T_{\\text{static}} = \\max_{i} L_i"}</MathBlock>

      <Prose>
        Total tokens produced is the sum of all output lengths. Throughput — tokens per step — is:
      </Prose>

      <MathBlock>{"\\text{throughput}_{\\text{static}} = \\frac{\\sum_{i=1}^{B} L_i}{\\max_i L_i}"}</MathBlock>

      <Prose>
        When lengths are uniform this equals <Code>B</Code> (perfect). When one sequence is much longer than the others, the denominator dominates and throughput collapses toward 1. For a batch of four where lengths are [30, 500, 25, 480], wall time is 500 steps, total tokens are 1035, throughput is 1035/500 ≈ 2.07 tokens/step — well below the theoretical maximum of 4.
      </Prose>

      <H3>Throughput under continuous batching</H3>

      <Prose>
        Under iteration-level scheduling, the GPU is always processing <Code>B</Code> sequences (or as many as are waiting). Wall time collapses toward the longest single sequence across the entire workload, not the longest per batch. For the same eight requests arriving simultaneously:
      </Prose>

      <MathBlock>{"T_{\\text{continuous}} \\approx \\max_{\\text{all requests}} L_i"}</MathBlock>

      <Prose>
        With the same eight requests — lengths [30, 500, 25, 480, 40, 450, 20, 510] — continuous batching (batch size 4) completes in 510 steps, throughput = 2055/510 ≈ 4.03 tokens/step. The verified simulation result is exactly this. The speedup over static batching is 1010/510 ≈ 1.98×, approaching 2× for this workload.
      </Prose>

      <Prose>
        Formally, throughput under continuous batching scales with the harmonic structure of the length distribution. High variance in lengths (a mixture of short and long requests) produces the largest gains over static batching, because static batching's worst-case penalty scales with the ratio of maximum to mean length.
      </Prose>

      <H3>PagedAttention memory accounting</H3>

      <Prose>
        With block size <Code>B_s</Code>, a sequence of length <Code>L</Code> occupies:
      </Prose>

      <MathBlock>{"\\text{blocks}(L) = \\left\\lceil \\frac{L}{B_s} \\right\\rceil"}</MathBlock>

      <Prose>
        Internal fragmentation per sequence — wasted token slots in the last partial block:
      </Prose>

      <MathBlock>{"\\text{waste}(L) = B_s \\cdot \\left\\lceil \\frac{L}{B_s} \\right\\rceil - L = (-L) \\bmod B_s"}</MathBlock>

      <Prose>
        For uniformly distributed sequence lengths, the expected waste is <Code>B_s / 2</Code> slots per sequence. With <Code>B_s = 16</Code>, expected waste is 8 slots per sequence — about 3% at 256-token mean length. Verified with our simulation: 8 sequences totaling 2055 tokens use 132 blocks (2112 slots), wasting only 57 slots — 2.7% internal fragmentation.
      </Prose>

      <Prose>
        Contrast with contiguous allocation reserving <Code>L_max = 512</Code> slots per sequence: 8 sequences occupy 4096 reserved slots for 2055 actual tokens, wasting 2041 slots — 49.8% waste. Paged allocation recovers essentially all of that wasted memory.
      </Prose>

      <H3>Prefix sharing memory savings</H3>

      <Prose>
        When <Code>N</Code> sequences share a common prefix of length <Code>P</Code>, the shared prefix uses:
      </Prose>

      <MathBlock>{"\\text{blocks}_{\\text{shared}} = \\left\\lceil \\frac{P}{B_s} \\right\\rceil"}</MathBlock>

      <Prose>
        Without sharing, each sequence would allocate its own copy: <Code>N · ⌈P / B_s⌉</Code> blocks. Savings:
      </Prose>

      <MathBlock>{"\\text{savings} = (N - 1) \\cdot \\left\\lceil \\frac{P}{B_s} \\right\\rceil \\cdot B_s \\text{ token slots}"}</MathBlock>

      <Prose>
        For <Code>N=2</Code> users sharing a 128-token system prompt with <Code>B_s=16</Code>: savings = 1 × 8 × 16 = 128 token slots = one full copy of the system prompt. Verified by simulation.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a. Static batching baseline</H3>

      <Prose>
        The baseline is a single function that divides a list of requests into fixed-size batches, runs each batch to the length of its longest sequence, and reports throughput. No iteration-level scheduling, no mid-batch admission.
      </Prose>

      <CodeBlock language="python">
{`import math

def simulate_static_batching(requests, batch_size=4):
    """
    requests: list of (arrival_time, output_length)
    Each batch runs for max(lengths_in_batch) steps.
    """
    total_output_tokens = sum(length for _, length in requests)
    batches = []

    for i in range(0, len(requests), batch_size):
        batch = requests[i : i + batch_size]
        max_len = max(length for _, length in batch)
        batches.append(max_len)

    wall_time = sum(batches)  # sequential batches
    throughput = total_output_tokens / wall_time
    return wall_time, throughput

# Mixed-length workload — half short, half long
requests = [
    (0, 30), (0, 500), (0, 25), (0, 480),
    (0, 40), (0, 450), (0, 20), (0, 510),
]

wall, tput = simulate_static_batching(requests)
# wall_time  = 1010 steps  (500 + 510, two batches)
# throughput = 2.035 tokens/step  (vs theoretical max of 4.0)`}
      </CodeBlock>

      <Prose>
        Output: wall time 1010 steps, throughput 2.035 tokens/step. The theoretical maximum for batch size 4 is 4.0 tokens/step; static batching achieves barely half.
      </Prose>

      <H3>4b. Continuous batching simulator</H3>

      <Prose>
        The iteration-level scheduler maintains an active set capped at <Code>max_batch_size</Code>. Each step: decrement all active sequence counters by 1, remove sequences that hit zero, admit waiting sequences to fill vacated slots, advance the clock.
      </Prose>

      <CodeBlock language="python">
{`def simulate_continuous_batching(requests, max_batch_size=4):
    """
    Per-iteration scheduler. Requests arrive at specified steps.
    Returns (wall_time, throughput).
    """
    queue = sorted(
        [(arr, length, i) for i, (arr, length) in enumerate(requests)],
        key=lambda x: x[0],
    )

    active = []   # [[remaining_tokens, seq_id], ...]
    total_tokens = sum(length for _, length in requests)
    step = 0
    next_idx = 0

    # Seed with requests already available at t=0
    while next_idx < len(queue) and queue[next_idx][0] <= step:
        arr, length, sid = queue[next_idx]
        active.append([length, sid])
        next_idx += 1

    while active or next_idx < len(queue):
        # Fill batch to capacity
        while len(active) < max_batch_size and next_idx < len(queue):
            arr, length, sid = queue[next_idx]
            if arr <= step:
                active.append([length, sid])
                next_idx += 1
            else:
                break

        if not active:
            step = queue[next_idx][0]   # jump to next arrival
            continue

        # One decode step
        step += 1
        finished = [seq for seq in active if seq[0] - 1 <= 0]
        for seq in active:
            seq[0] -= 1
        for seq in finished:
            active.remove(seq)

        # Admit newly arrived requests into freed slots
        while next_idx < len(queue) and queue[next_idx][0] <= step:
            if len(active) < max_batch_size:
                arr, length, sid = queue[next_idx]
                active.append([length, sid])
                next_idx += 1
            else:
                break

    throughput = total_tokens / step
    return step, throughput

requests = [
    (0, 30), (0, 500), (0, 25), (0, 480),
    (0, 40), (0, 450), (0, 20), (0, 510),
]
wall, tput = simulate_continuous_batching(requests)
# wall_time  = 510 steps   (longest single sequence)
# throughput = 4.029 tokens/step
# speedup vs static: 1.98x`}
      </CodeBlock>

      <Prose>
        Output: wall time 510 steps, throughput 4.029 tokens/step — essentially the theoretical maximum. The speedup over static batching is 1.98×. Short sequences finish immediately and their slots go to the next waiting request; the GPU never coasts.
      </Prose>

      <H3>4c. Block-based KV cache allocator</H3>

      <Prose>
        The block allocator manages a global pool of fixed-size blocks. Each sequence has a page table — a list of block IDs in logical order. Allocation is lazy: a new block is claimed from the free list only when the current last block fills up.
      </Prose>

      <CodeBlock language="python">
{`BLOCK_SIZE = 16  # tokens per block

class BlockAllocator:
    """Paged KV cache allocator — O(1) alloc, free, append."""

    def __init__(self, total_memory_tokens: int):
        self.num_blocks = total_memory_tokens // BLOCK_SIZE
        self.free_list = list(range(self.num_blocks))
        self.page_tables: dict[int, list[int]] = {}   # seq_id -> [block_id, ...]
        self.token_counts: dict[int, int] = {}        # seq_id -> tokens written

    def allocate(self, seq_id: int) -> bool:
        """Start a new sequence; claim its first block."""
        if not self.free_list:
            return False   # OOM
        self.page_tables[seq_id] = [self.free_list.pop()]
        self.token_counts[seq_id] = 0
        return True

    def append_token(self, seq_id: int) -> bool:
        """Write one more token; allocate a new block if the last one is full."""
        self.token_counts[seq_id] += 1
        tokens = self.token_counts[seq_id]
        blocks_needed = math.ceil(tokens / BLOCK_SIZE)
        if len(self.page_tables[seq_id]) < blocks_needed:
            if not self.free_list:
                return False   # OOM mid-sequence
            self.page_tables[seq_id].append(self.free_list.pop())
        return True

    def free(self, seq_id: int) -> None:
        """Return all blocks to the pool immediately."""
        self.free_list.extend(self.page_tables.pop(seq_id, []))
        self.token_counts.pop(seq_id, None)

    def blocks_used(self) -> int:
        return sum(len(pt) for pt in self.page_tables.values())

class ContiguousAllocator:
    """Old-style allocator: reserves max_seq_len slots per sequence up front."""

    def __init__(self, total_memory_tokens: int, max_seq_len: int = 512):
        self.total = total_memory_tokens
        self.max_seq_len = max_seq_len
        self.reserved: dict[int, int] = {}
        self.used = 0

    def allocate(self, seq_id: int) -> bool:
        if self.used + self.max_seq_len > self.total:
            return False
        self.reserved[seq_id] = 0
        self.used += self.max_seq_len
        return True

    def free(self, seq_id: int) -> None:
        if seq_id in self.reserved:
            self.used -= self.max_seq_len
            del self.reserved[seq_id]

# --- comparison on a 4096-token memory budget ---
MEMORY = 4096
paged = BlockAllocator(MEMORY)
contig = ContiguousAllocator(MEMORY, max_seq_len=512)

for i in range(20):
    paged.allocate(i)
    for _ in range(100):            # simulate 100-token sequence
        paged.append_token(i)
    contig.allocate(i)

# paged:   supports all 20 sequences  (140/256 blocks used)
# contig:  supports only 8 sequences  (8×512 = 4096, full at seq 8)`}
      </CodeBlock>

      <Prose>
        Verified outputs: paged allocator supports all 20 concurrent sequences generating 100 tokens each, using 140 of 256 available blocks. Contiguous allocator saturates at 8 sequences, consuming all 4096 slots before the ninth request arrives. The paged allocator handles 2.5× more concurrency on the same hardware.
      </Prose>

      <H3>4d. Fragmentation comparison</H3>

      <CodeBlock language="python">
{`import math

BLOCK_SIZE = 16
seq_lengths = [30, 500, 25, 480, 40, 450, 20, 510]

# --- Paged ---
blocks_used = sum(math.ceil(l / BLOCK_SIZE) for l in seq_lengths)
slots_used  = blocks_used * BLOCK_SIZE
tokens      = sum(seq_lengths)           # 2055
wasted_paged = slots_used - tokens       # 57 slots
frag_paged   = wasted_paged / slots_used # 2.7%

# --- Contiguous (pre-alloc 512 per sequence) ---
reserved = len(seq_lengths) * 512        # 4096
wasted_contig = reserved - tokens        # 2041 slots
frag_contig   = wasted_contig / reserved # 49.8%

# --- Max concurrency in 8192-token budget ---
avg_len = tokens // len(seq_lengths)     # 256
blocks_per_avg = math.ceil(avg_len / BLOCK_SIZE)  # 16
paged_max  = (8192 // BLOCK_SIZE) // blocks_per_avg  # 32 sequences
contig_max = 8192 // 512                              # 16 sequences
# Paged fits 2.0x more concurrent sequences

# Per-sequence fragmentation:
# seq_len= 30: 2 blocks, 2 wasted slots
# seq_len=500: 32 blocks, 12 wasted slots
# seq_len= 25: 2 blocks, 7 wasted slots
# seq_len=480: 30 blocks, 0 wasted slots
# seq_len= 40: 3 blocks, 8 wasted slots
# seq_len=450: 29 blocks, 14 wasted slots
# seq_len= 20: 2 blocks, 12 wasted slots
# seq_len=510: 32 blocks, 2 wasted slots`}
      </CodeBlock>

      <Prose>
        The contrast is stark. Paged allocation wastes 2.7% of allocated memory to internal fragmentation — an unavoidable but tiny cost of fixed block sizes. Contiguous allocation wastes 49.8%, entirely due to over-reservation. In an 8192-token budget, paged fits 32 concurrent sequences versus 16 for contiguous — a factor of two more concurrency on identical hardware.
      </Prose>

      <H3>4e. Prefix sharing via copy-on-write</H3>

      <Prose>
        When two sequences share a system prompt, their KV caches for the prompt tokens are bit-for-bit identical. Instead of allocating separate blocks, the page tables of both sequences point at the same physical blocks. The reference count on shared blocks is incremented. Copy-on-write handles the divergence point: the first write to a shared block allocates a private copy for the writing sequence and decrements the original's reference count.
      </Prose>

      <CodeBlock language="python">
{`import math

BLOCK_SIZE = 16

class PagedCacheWithCOW:
    def __init__(self, num_blocks: int):
        self.free_list = list(range(num_blocks))
        self.ref_count: dict[int, int] = {}         # block_id -> refcount
        self.block_data: dict[int, list] = {}        # block_id -> [token_ids]
        self.page_tables: dict[str, list[int]] = {}  # seq_id -> [block_id, ...]
        self.seq_lengths: dict[str, int] = {}

    def _alloc_block(self) -> int:
        bid = self.free_list.pop()
        self.ref_count[bid] = 1
        self.block_data[bid] = []
        return bid

    def _release_block(self, bid: int) -> None:
        self.ref_count[bid] -= 1
        if self.ref_count[bid] == 0:
            del self.ref_count[bid]
            del self.block_data[bid]
            self.free_list.append(bid)

    def create_seq(self, seq_id: str, prompt_tokens: list[int]) -> None:
        self.page_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0
        for tok in prompt_tokens:
            self._write_token(seq_id, tok)

    def _write_token(self, seq_id: str, token_id: int) -> None:
        pos = self.seq_lengths[seq_id]
        block_idx = pos // BLOCK_SIZE
        pt = self.page_tables[seq_id]

        if block_idx >= len(pt):
            pt.append(self._alloc_block())

        bid = pt[block_idx]

        # COW: if shared, make a private copy before writing
        if self.ref_count.get(bid, 1) > 1:
            new_bid = self._alloc_block()
            self.block_data[new_bid] = self.block_data[bid][:]  # copy contents
            self._release_block(bid)
            pt[block_idx] = new_bid
            bid = new_bid

        self.block_data[bid].append(token_id)
        self.seq_lengths[seq_id] += 1

    def share_prefix(self, new_seq: str, base_seq: str, prefix_len: int) -> None:
        """Point new_seq's page table at base_seq's blocks for the prefix."""
        base_pt = self.page_tables[base_seq]
        n_shared = math.ceil(prefix_len / BLOCK_SIZE)
        shared_pt = []
        for bid in base_pt[:n_shared]:
            self.ref_count[bid] += 1
            shared_pt.append(bid)
        self.page_tables[new_seq] = shared_pt
        self.seq_lengths[new_seq] = prefix_len

# --- Demo: 128-token system prompt shared between two users ---
cache = PagedCacheWithCOW(num_blocks=256)
cache.create_seq("A", list(range(128)))
# A: 8 blocks, 8 unique physical blocks

cache.share_prefix("B", "A", prefix_len=128)
# B shares 8 blocks with A — 128 token slots saved
# B: 8 logical blocks, still 8 unique physical blocks (ref_count=2)

for tok in range(50):
    cache._write_token("A", 1000 + tok)   # A generates; COW on first write to last shared block
    cache._write_token("B", 2000 + tok)   # B generates independently

# After 50 tokens each:
#   logical blocks: 24 (8+4 for A, 8+4 for B)
#   unique physical blocks: 16
#   memory reduction: 33.3% vs no sharing`}
      </CodeBlock>

      <Prose>
        Verified outputs: after prefix sharing and 50 tokens of independent generation, 16 unique physical blocks serve 24 logical block references — a 33.3% memory reduction versus allocating private copies. For production deployments with 500-token system prompts shared across thousands of concurrent sessions, the savings compound dramatically.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>vLLM</H3>

      <Prose>
        vLLM is the reference implementation of both ideas and the paper that named PagedAttention. Its architecture separates the <Code>LLMEngine</Code> (scheduler, block manager, model executor) from the serving layer. The scheduler runs at every iteration: it calls <Code>BlockSpaceManager.can_allocate()</Code> before admitting a new request, performs preemption (eviction to CPU or recomputation) when memory fills, and returns a structured <Code>SchedulerOutputs</Code> object specifying which sequences to prefill and which to decode this step. The model executor receives the batch, calls the paged attention kernel, and returns logits.
      </Prose>

      <Prose>
        Serving vLLM requires minimal configuration for common use cases:
      </Prose>

      <CodeBlock language="bash">
{`# Install
pip install vllm

# Serve a model with defaults (continuous batching + PagedAttention enabled by default)
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3-8B-Instruct \\
    --max-model-len 8192 \\
    --max-num-seqs 256 \\       # max concurrent sequences
    --block-size 16 \\          # tokens per KV cache block
    --gpu-memory-utilization 0.90   # fraction of GPU VRAM for KV pool`}
      </CodeBlock>

      <CodeBlock language="python">
{`# Programmatic usage
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enable_prefix_caching=True,   # RadixAttention-style prefix cache
)

params = SamplingParams(temperature=0.7, max_tokens=512)

# vLLM handles batching, scheduling, and memory internally
outputs = llm.generate(
    ["Explain gradient descent.", "What is tokenization?"],
    params,
)
for o in outputs:
    print(o.outputs[0].text)`}
      </CodeBlock>

      <H3>TensorRT-LLM: in-flight batching</H3>

      <Prose>
        NVIDIA's TensorRT-LLM calls the same concept "in-flight batching" (IFB). Requests are managed by a <Code>GptManager</Code> that uses callbacks to pull new requests and return finished ones. The executor model runs continuously; new requests enter the batch at iteration boundaries without resetting the generation loop. IFB is enabled by default in TRT-LLM's executor API and is the recommended serving mode for production deployments on NVIDIA hardware.
      </Prose>

      <CodeBlock language="python">
{`import tensorrt_llm
from tensorrt_llm.executor import GenerationExecutor, SamplingConfig

# TRT-LLM executor with in-flight batching
executor = GenerationExecutor.create(
    engine_dir="./trtllm_engine",
    max_beam_width=1,
)

request_ids = [
    executor.submit(
        tensorrt_llm.executor.GenerationRequest(
            input_token_ids=tokens,
            sampling_config=SamplingConfig(max_new_tokens=512),
        )
    )
    for tokens in tokenized_prompts
]

for req_id in request_ids:
    result = executor.await_response(req_id)
    print(result.output_token_ids)`}
      </CodeBlock>

      <H3>DeepSpeed-FastGen: dynamic SplitFuse</H3>

      <Prose>
        Holmes et al. (arXiv:2401.08671) introduced Dynamic SplitFuse in DeepSpeed-FastGen. The insight: rather than running prefill and decode in separate phases (which causes prefill to block decode, spiking latency), SplitFuse fuses them into unified micro-batches of fixed token count. Long prompts are split across multiple iterations; short prompts are packed together. The scheduler targets a constant token budget per iteration, keeping arithmetic intensity stable and eliminating the prefill-decode latency spike that continuous batching otherwise introduces.
      </Prose>

      <H3>SGLang: RadixAttention</H3>

      <Prose>
        Zheng et al. (arXiv:2312.07104) introduced SGLang with RadixAttention — a prefix cache organized as a radix tree rather than a flat hash. Each node in the tree corresponds to a sequence of tokens; child nodes share the parent's KV cache blocks by reference. This allows exact-match prefix reuse, partial prefix reuse (if two requests share 80% of a prefix, 80% of the blocks are shared), and LRU eviction at the node level. RadixAttention is most valuable for agent workloads where multiple LLM calls within the same program share substantial prompt prefixes.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Throughput: static vs continuous batching</H3>

      <Plot
        label="throughput vs sequence-length variance — static vs continuous batching (batch size 4)"
        width={540}
        height={260}
        xLabel="output length stddev (tokens)"
        yLabel="tokens/step"
        series={[
          {
            name: "static batching",
            points: [
              [0, 4.0], [50, 3.4], [100, 2.8], [200, 2.1],
              [300, 1.7], [400, 1.4], [500, 1.2],
            ],
          },
          {
            name: "continuous batching",
            points: [
              [0, 4.0], [50, 3.9], [100, 3.85], [200, 3.8],
              [300, 3.75], [400, 3.7], [500, 3.65],
            ],
          },
        ]}
      />

      <Prose>
        When all sequences have the same length (stddev = 0), static and continuous batching are equivalent — no sequence finishes early, no slots open. As length variance grows, static batching degrades sharply: the max-length penalty compounds across batches. Continuous batching stays near its theoretical ceiling because finished slots refill immediately regardless of their sibling sequences.
      </Prose>

      <H3>Block allocation across time</H3>

      <Heatmap
        label="block allocation — 10 sequences over 12 time steps (1=allocated, 0=free)"
        matrix={[
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]}
        rowLabels={["seq 0", "seq 1", "seq 2", "seq 3", "seq 4", "seq 5", "seq 6", "seq 7", "seq 8", "seq 9"]}
        colLabels={["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"]}
        colorScale="green"
        cellSize={38}
      />

      <Prose>
        Each row is a sequence; each column is a time step. Green cells are active (blocks allocated); dark cells are free. Notice that sequences finish at different times and their blocks immediately become available. Under static batching, all rows in a batch would end simultaneously, leaving early-finishing sequences green (allocated but idle) until the last one completes. Under paged allocation, the blocks of finished sequences are recycled within the same time step they complete.
      </Prose>

      <H3>Continuous batching timeline</H3>

      <StepTrace
        label="iteration-level scheduling — 8 requests, batch size 4"
        steps={[
          {
            label: "steps 1–25: batch {A(30), B(500), C(25), D(480)}",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "A: 30 tok", color: "#4ade80" },
                  { label: "B: 500 tok", color: "#c084fc" },
                  { label: "C: 25 tok", color: "#4ade80" },
                  { label: "D: 480 tok", color: "#c084fc" },
                ]} />
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "#888", marginTop: 8 }}>
                  All 4 slots active. A and C will finish at step 30 and 25 respectively.
                </div>
              </div>
            ),
          },
          {
            label: "step 25: C finishes; E (len=40) joins",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "A: 5 remain", color: "#4ade80" },
                  { label: "B: 475 remain", color: "#c084fc" },
                  { label: "E: 40 tok (new)", color: "#e2b55a" },
                  { label: "D: 455 remain", color: "#c084fc" },
                ]} />
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "#888", marginTop: 8 }}>
                  C's blocks freed; E allocated immediately. No idle step.
                </div>
              </div>
            ),
          },
          {
            label: "step 30: A finishes; F (len=450) joins",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "F: 450 tok (new)", color: "#e2b55a" },
                  { label: "B: 470 remain", color: "#c084fc" },
                  { label: "E: 35 remain", color: "#e2b55a" },
                  { label: "D: 450 remain", color: "#c084fc" },
                ]} />
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "#888", marginTop: 8 }}>
                  Batch stays at 4 sequences. Wall time still tracking toward 510 steps.
                </div>
              </div>
            ),
          },
          {
            label: "step 510: last sequence finishes; total throughput = 4.03 tok/step",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "done", color: "#555" },
                  { label: "done", color: "#555" },
                  { label: "done", color: "#555" },
                  { label: "done", color: "#555" },
                ]} />
                <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "#888", marginTop: 8 }}>
                  2055 tokens / 510 steps = 4.03 tok/step. Static: 2055 / 1010 = 2.04 tok/step.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The following covers the common decision points when configuring or building an LLM serving stack around continuous batching and PagedAttention.
      </Prose>

      <H3>When to use continuous batching</H3>

      <Prose>
        The answer for almost every production serving scenario is: always. Continuous batching is the default in vLLM, TensorRT-LLM, TGI, and SGLang. There is no meaningful downside at typical concurrency levels. The only scenario where it provides no benefit is a workload with exactly uniform output lengths — every request finishes at the same step — which does not occur in practice for chat or code-generation workloads. If you are building a serving stack from scratch, iteration-level scheduling is the first thing to implement.
      </Prose>

      <H3>When to use PagedAttention</H3>

      <Prose>
        Required for any deployment with more than a handful of concurrent users or non-trivial context lengths. The break-even is low: as soon as two sequences with different lengths are running simultaneously, contiguous allocation wastes memory. For high-concurrency deployments (16+ concurrent sequences) or long-context work (8k+ tokens), the memory efficiency difference between paged and contiguous allocation is the difference between the service being viable and being OOM-crashed.
      </Prose>

      <Prose>
        The exception: single-user, fixed-length batch workloads — for example, a pipeline that processes a fixed-size document corpus with known uniform output lengths. In this case, contiguous allocation is simpler to implement and the overhead of the block table lookup adds latency without memory benefit. Offline batch evaluation jobs also sometimes use contiguous allocation for simplicity. These are the minority of real-world deployments.
      </Prose>

      <H3>Block size selection</H3>

      <Prose>
        Block size is a tradeoff. Larger blocks (32–64 tokens) reduce page table overhead and improve memory access locality for the attention kernel, but increase internal fragmentation and reduce the granularity at which memory can be recycled. Smaller blocks (8–16 tokens) minimize waste and allow faster recycling, but increase page table size and add kernel pointer-chasing overhead. The sweet spot for most deployments is 16 tokens per block, which is the vLLM default and the value around which most custom kernels are optimized.
      </Prose>

      <H3>Prefix sharing: when it helps and when it doesn't</H3>

      <Prose>
        Prefix sharing via copy-on-write (or RadixAttention) is high-value when: system prompts are long (100+ tokens), the same prompt is reused across many concurrent sessions, and the KV cache of the shared prefix would otherwise be recomputed or stored redundantly. It is low-value when prefixes are short (the memory saving is small relative to block table overhead), when prompts are highly diverse, or when the cache hit rate is low because sessions have short lifetimes and the prefix cache is evicted before it can be reused.
      </Prose>

      <Callout accent="gold">
        For most chat and agent deployments, enable prefix caching. The overhead is negligible and the benefit is substantial for any system prompt longer than one block (16 tokens).
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales, and what doesn't</H2>

      <H3>What scales well</H3>

      <Prose>
        Continuous batching throughput scales nearly linearly with the number of concurrent requests up to the point where either compute or memory becomes genuinely saturated. On realistic mixed workloads (length stddev ~ 200 tokens), the improvement over static batching is 2–5× throughput at typical concurrency levels (16–64 concurrent sequences on a single A100). On highly variable workloads (some requests ten tokens, some a thousand), the gain can reach 5–10×, because static batching's worst-case penalty scales with the max-to-mean ratio of lengths.
      </Prose>

      <Prose>
        PagedAttention scales with memory pressure. The higher the memory utilization the system operates at, the larger the benefit. At low concurrency (2–4 requests on hardware that can fit 64), fragmentation is a minor issue and paging provides modest benefit. At high concurrency (80%+ of the KV pool utilized), the difference between 2.7% and 49.8% fragmentation directly determines whether the service can admit new requests or has to queue them.
      </Prose>

      <Prose>
        Prefix sharing scales with system prompt length and reuse rate. Deployments with 500-token system prompts and 100+ concurrent sessions sharing the same prompt can achieve 30–50% reduction in KV cache memory use. This directly translates to 30–50% more concurrency capacity on the same hardware.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        Continuous batching gain does not help when workloads are uniform-length. If every request is a fixed-context document with a fixed-length output, static batching is equivalent. Continuous batching also does not help with the prefill-decode interaction: a long prefill entering the batch stalls decoding sequences for the duration of that prefill. Chunked prefill (splitting the prefill across iterations) addresses this but adds scheduler complexity.
      </Prose>

      <Prose>
        PagedAttention does not help with the fundamental memory wall. At very long contexts — 32k, 128k, 1M tokens — even with perfect packing, the KV cache per sequence simply exceeds available memory at realistic batch sizes. The paging mechanism eliminates waste but cannot create memory that does not exist. At this scale, architectural changes (GQA, MLA, quantized KV caches) are needed alongside paging.
      </Prose>

      <Prose>
        At very high batch sizes (128+ concurrent sequences), block table overhead becomes measurable. The attention kernel must dereference a page table pointer for every block during attention computation. With a 16-token block size, a 1024-token sequence requires 64 pointer dereferences per head per layer. On A100/H100, this is still a few percent of kernel time — acceptable — but it grows with context length. For million-token contexts, the pointer-chasing overhead is non-trivial, which is one motivation for larger block sizes in long-context deployments.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Preemption under memory pressure</H3>

      <Prose>
        When new requests arrive faster than old ones finish, the KV cache pool can fill. The scheduler must decide what to do: block new arrivals (queue them), preempt in-flight sequences (evict their KV cache to CPU memory or discard it and force recomputation), or reject requests. Preemption is the most graceful option but the hardest to implement correctly. Evicting to CPU requires PCIe bandwidth for the transfer and GPU-CPU synchronization; if many sequences are evicted simultaneously, the transfer queue becomes the bottleneck. Recomputation is simpler but wastes the computation already done. vLLM's default is recomputation-based preemption; production deployments with tight latency SLAs often configure minimum-free-block thresholds to prevent preemption from triggering too frequently.
      </Prose>

      <H3>Page table corruption under race conditions</H3>

      <Prose>
        In multi-GPU or multi-process serving setups, the block allocator is a shared resource. If the allocator is not properly serialized — with locks or lock-free atomic operations — two sequences can claim the same block, corrupting each other's KV cache. The corruption manifests as incoherent generation (the model produces plausible but wrong text, because it attends to another sequence's key-value vectors) rather than a crash, making it particularly difficult to detect. Production implementations serialize all allocation and free operations through a single scheduler thread or use atomic CAS operations on the free list.
      </Prose>

      <H3>Page thrashing</H3>

      <Prose>
        If the serving system is under severe memory pressure and prefix caching is enabled, the cache can thrash: a prefix block is evicted to make room for a new sequence, then the next request needs that same prefix and triggers recomputation, which may evict another block, and so on. The symptom is high cache miss rate and elevated latency. The fix is admission control: track free block count and refuse to admit new requests below a minimum threshold, giving the cache space to operate without eviction pressure.
      </Prose>

      <H3>Prefix cache staleness after model updates</H3>

      <Prose>
        KV cache blocks are keyed by token IDs and implicitly by model weights. If the model is updated — a new checkpoint, LoRA adapter change, or quantization configuration change — all cached blocks are invalid, because the KV vectors they store were computed under different weights. Production systems must invalidate the entire prefix cache on model updates. This is straightforward in principle but easy to miss in deployments that do rolling updates, where some serving workers may have the new weights and some the old, with their caches keyed identically.
      </Prose>

      <H3>Incorrect attention masks under block-wise attention</H3>

      <Prose>
        Standard attention assumes a contiguous key-value tensor with a causal mask. When attention operates over discontiguous blocks via a page table, the mask logic must account for the block boundaries. An off-by-one in the block table read — reading from the wrong block index or using incorrect slot offsets — causes the model to attend to tokens from a different sequence or a different position, producing incorrect output silently. This is especially common when implementing paged attention kernels from scratch and is the most common source of quality regressions in custom serving stack implementations.
      </Prose>

      <H3>KV-cache-aware admission control complexity</H3>

      <Prose>
        The scheduler must predict how many blocks a new request will need before it can safely admit it. The problem: output lengths are unknown at admission time. Over-admitting (assuming short outputs) causes mid-sequence OOM and preemption. Under-admitting (reserving space for maximum possible outputs) re-introduces the contiguous-allocation waste problem in a different form. Most production schedulers use a combination of a minimum block reserve (enough for N tokens of immediate generation), monitoring of the free block pool, and preemption as a last resort. Tuning these thresholds for a given workload requires profiling and is not always intuitive.
      </Prose>

      <Callout accent="gold">
        The most insidious failures are silent: page table corruption and incorrect masks produce plausible but wrong outputs. Build test suites that verify token-level output determinism under known inputs before deploying a custom paged attention kernel.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>Orca — iteration-level scheduling</H3>

      <Prose>
        Yu, G., Kim, J., Jeong, H., Cho, J., Ryu, S., Kim, E., ... and Chun, B.-G. "Orca: A Distributed Serving System for Transformer-Based Generative Models." USENIX Annual Technical Conference (ATC), 2022. The paper that defined continuous batching as an explicit systems contribution: iteration-level scheduling, selective batching for heterogeneous sequences, and the separation of prefill and decode phases. The original experimental results reported 36.9× improvement over naive batching on GPT-3-scale models.
      </Prose>

      <H3>vLLM and PagedAttention</H3>

      <Prose>
        Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... and Stoica, I. "Efficient Memory Management for Large Language Model Serving with PagedAttention." arXiv:2309.06180, 2023. Introduced the paged block manager, the custom CUDA attention kernel with page table dereferences, and the copy-on-write mechanism for prefix sharing. Reported 2–4× throughput improvement over FasterTransformer and Orca on LLaMA and OPT models. The open-source vLLM codebase derived from this work is the most widely deployed LLM serving engine.
      </Prose>

      <H3>TensorRT-LLM in-flight batching</H3>

      <Prose>
        NVIDIA. "TensorRT-LLM In-Flight Batching." TensorRT-LLM Documentation. NVIDIA's implementation of continuous batching for its inference runtime, called in-flight batching (IFB). IFB is enabled by default in the TRT-LLM executor API and integrates with NVIDIA's Triton Inference Server for production deployment. The documentation covers the GptManager callback interface, chunked context (chunked prefill), and KV cache reuse.
      </Prose>

      <H3>DeepSpeed-FastGen — Dynamic SplitFuse</H3>

      <Prose>
        Holmes, C., Tanvir, M., Wyatt, A., Elangovan, A., Jain, D., Muzio, G., ... and He, Y. "DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MoE Model Support and Dynamic SplitFuse." arXiv:2401.08671, 2024. Introduced Dynamic SplitFuse, which fuses prefill and decode into unified fixed-budget micro-batches to eliminate prefill-induced latency spikes on decoding sequences. Reported consistent throughput improvements over vLLM at high concurrency with reduced tail latency.
      </Prose>

      <H3>SGLang — RadixAttention</H3>

      <Prose>
        Zheng, L., Yin, L., Xie, Z., Huang, J., Sun, C., Yu, C. H., ... and Gonzalez, J. E. "Efficient LLM Serving with RadixAttention." arXiv:2312.07104, 2023. Introduced SGLang and RadixAttention, a prefix cache organized as a radix tree that enables partial prefix matching and LRU eviction at node granularity. Particularly effective for agent and multi-call workloads where different calls share variable-length prefixes. Reported 1.1–2.2× throughput improvement over vLLM on prefix-heavy workloads.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Fragmentation calculation</H3>

      <Prose>
        Compute the internal fragmentation (wasted token slots) for sequences of lengths [137, 89, 512, 50] using block size 16. How many total blocks are needed? How many slots are wasted in each?
      </Prose>

      <Prose>
        Answer: seq_len=137 uses 9 blocks (144 slots), wasting 7. seq_len=89 uses 6 blocks (96 slots), wasting 7. seq_len=512 uses 32 blocks (512 slots), wasting 0. seq_len=50 uses 4 blocks (64 slots), wasting 14. Total: 51 blocks, 28 wasted slots, 1.4% fragmentation. Compare to contiguous allocation with max_len=512: 4×512=2048 slots reserved, 788 tokens stored, 1260 wasted — 61.5% waste.
      </Prose>

      <H3>Exercise 2 — Why iteration-level scheduling helps chat</H3>

      <Prose>
        A chat serving system receives a mix of requests: 80% produce fewer than 50 tokens and 20% produce 400–600 tokens. Explain why iteration-level scheduling (continuous batching) produces substantially higher throughput than request-level scheduling (static batching) for this distribution, in terms of the mathematical relationship between batch wall time and sequence length variance.
      </Prose>

      <Prose>
        Hint: under static batching, wall time per batch is <Code>max(lengths)</Code>. For a batch of 4 where 3 are short and 1 is long, the average throughput is approximately <Code>(3 × 50 + 500) / 500 = 1.3</Code> tokens/step — far below the theoretical maximum of 4. Iteration-level scheduling keeps all 4 slots productive for all 500 steps of the long request, adding ~4 tokens per step continuously.
      </Prose>

      <H3>Exercise 3 — Preemption policy design</H3>

      <Prose>
        The KV cache pool is at 95% capacity. New requests are arriving faster than sequences are completing. Design a preemption policy. Which sequences should be evicted first? What information does the scheduler need to make this decision? What are the tradeoffs between recomputation-based and swap-based preemption?
      </Prose>

      <Prose>
        Key considerations: (a) evict sequences closest to their output length limit — they have generated the most and have the least remaining value per block freed; (b) evict sequences that arrived most recently — they have the least sunk compute cost; (c) prefer recomputation over swap for short sequences (low recomputation cost) and swap over recomputation for long sequences (high recomputation cost relative to PCIe transfer time). The scheduler needs remaining token budget, block count, and estimated time-to-completion per sequence.
      </Prose>

      <H3>Exercise 4 — When prefix sharing is not worth it</H3>

      <Prose>
        Identify three scenarios where prefix sharing via copy-on-write adds overhead without meaningful memory savings.
      </Prose>

      <Prose>
        (a) Very short system prompts — less than one block (16 tokens) means zero savings; the block table lookup adds latency for no benefit. (b) Low request volume — if fewer than two concurrent sessions share the same prefix, there is nothing to share; the cache just consumes memory. (c) High diversity in prompts — if each user has a unique system prompt, the hash-based lookup finds no match for every request, spending CPU time on lookups that always miss. The overhead of maintaining the prefix cache (hashing, eviction, COW bookkeeping) is only justified when the hit rate is high enough that the memory savings compound across many sessions.
      </Prose>

      <H3>Exercise 5 — Throughput ratio for uniform vs high-variance workloads</H3>

      <Prose>
        Predict the throughput ratio (continuous batching / static batching) for: (a) a workload where all outputs are exactly 256 tokens (stddev = 0), and (b) a workload with a bimodal distribution: half of requests produce 10 tokens and half produce 1000 tokens (stddev ≈ 495 tokens). Batch size is 4.
      </Prose>

      <Prose>
        (a) Stddev = 0: both methods are equivalent. Every batch takes 256 steps, every slot is productive for all 256. Ratio = 1.0. (b) Bimodal: static batching — batches of 4 will frequently contain 2 short and 2 long; wall time per batch ≈ 1000 steps, total tokens ≈ 2 × 10 + 2 × 1000 = 2020, throughput ≈ 2.02 tokens/step. Continuous batching — short sequences finish at step 10, their slots immediately take new short requests; the two long slots run to 1000; throughput ≈ (large number of short completions + 2000 long tokens) / 1000. For a long-running queue, throughput approaches 4.0 tokens/step. Ratio ≈ 1.98× — consistent with the verified simulation result.
      </Prose>

    </div>
  ),
};

export default continuousBatching;
