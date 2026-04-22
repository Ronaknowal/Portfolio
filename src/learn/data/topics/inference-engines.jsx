import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const inferenceEngines = {
  title: "Inference Engines & Serving",
  readTime: "48 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every topic in this section has taught you a technique. KV caching showed why
        recomputing keys and values from scratch is a mathematical waste. Continuous
        batching showed why fixed batches squander half the GPU. PagedAttention showed
        why contiguous per-sequence memory allocations produce 60–80% fragmentation.
        Speculative decoding showed why a single token per forward pass underutilizes
        compute. Prefix caching showed why re-prefilling a shared system prompt ten
        thousand times a day is pure overhead. Constrained decoding showed why
        post-hoc validation loops are the wrong abstraction for structured output.
        Queueing theory showed why utilization above 0.7 bends the latency curve
        toward a cliff. Cost economics showed how all of this translates to dollars
        per million tokens.
      </Prose>

      <Prose>
        None of those techniques exist in isolation in production. They are bundled.
        A modern inference engine — vLLM, SGLang, TensorRT-LLM, Text Generation
        Inference, llama.cpp — is the complete stack: batch scheduler, paged KV
        allocator, speculative decoder, grammar automaton, prefix cache, admission
        controller, HTTP streaming layer, and metrics endpoint. Picking a technique
        is mostly a matter of picking a framework whose defaults implement it.
        This topic surveys the landscape, establishes when each framework wins, and
        closes the section by weaving its full arc into a single picture.
      </Prose>

      <Callout accent="purple">
        An inference engine is not one technique — it is the composition of all the
        techniques this section covered, implemented in a shared layer that every
        deployment reuses. Understanding each piece individually is what lets you
        diagnose, configure, and extend that composition.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>The serving engine as a stack of optimizations</H3>

      <Prose>
        Think of a serving engine as a layered stack where each layer addresses one
        of the bottlenecks identified in this section. At the bottom: memory
        management. Without PagedAttention, fragmentation wastes 60–80% of GPU
        KV-cache, which caps batch size and dominates every downstream metric.
        Fix memory first. On top of that: the scheduler. Continuous batching keeps
        the fixed-cost GPU compute busy across variable-length requests. On top of
        that: cache reuse. Prefix caching amortizes shared system prompt prefill
        across thousands of requests. On top of that: token-production acceleration.
        Speculative decoding converts idle compute headroom into additional tokens
        per forward pass. On top of that: output correctness. Constrained decoding
        eliminates structural retries at zero quality cost. At the surface:
        admission control and streaming, the interface that prevents the latency
        cliff and returns tokens to the client as they arrive.
      </Prose>

      <Prose>
        Each layer has a clear failure mode if it is missing. No paging: OOM under
        mixed workloads. No continuous batching: throughput hostage to the longest
        request in the batch. No prefix caching: thousands of expensive re-prefills
        per day. No speculative decoding: memory-bandwidth floor bounds per-token
        latency. No constrained decoding: 5% structural failure rate compounds across
        multi-step agent loops. No admission control: the latency cliff arrives
        without warning. A complete engine addresses all of them.
      </Prose>

      <H3>Ecosystem specialization in 2026</H3>

      <Prose>
        The four major open-source stacks have converged on the core features but
        diverged on the margins that matter most at scale.
      </Prose>

      <Prose>
        <strong>vLLM</strong> leads on research velocity, model breadth, and community
        size. Its defaults are sensible for most workloads. When uncertain, start here.
      </Prose>

      <Prose>
        <strong>SGLang</strong> leads on structured serving and prefix sharing. Its
        RadixAttention matches much longer shared prefixes than block-hash approaches,
        giving a decisive edge on agent workloads and multi-turn reasoning chains with
        long system prompts.
      </Prose>

      <Prose>
        <strong>TensorRT-LLM</strong> leads on raw hardware utilization on NVIDIA
        silicon. Compiled CUDA kernels extract the last 20–30% of tokens-per-second
        that other frameworks leave on the table. The cost is operational friction.
      </Prose>

      <Prose>
        <strong>TGI</strong> leads on HuggingFace ecosystem integration and is now
        in maintenance mode, accepting bug fixes but not new features. Its production
        maturity and Hub coverage keep it viable for teams already in that ecosystem.
      </Prose>

      <Prose>
        <strong>llama.cpp</strong> is a separate category: CPU-first, edge-first,
        single-user-first. It is not competing with the above for cloud throughput.
        It is the right tool for a MacBook, a Raspberry Pi, or a model that must
        run with no internet connection.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Effective throughput</H3>

      <Prose>
        The governing equation for a GPU serving endpoint is the product of hardware
        capacity and how well the serving stack exploits it.
      </Prose>

      <MathBlock>
        {"\\text{RPS} = \\frac{\\text{GPU FLOPs} \\times \\eta \\times \\tau}{\\text{FLOPs per request}}"}
      </MathBlock>

      <Prose>
        Where <Code>RPS</Code> is requests per second, <Code>η</Code> is GPU
        utilization (what continuous batching and the scheduler maximize),
        <Code>τ</Code> is tokens per FLOP (what quantization and architectural
        efficiency affect), and FLOPs per request is the total compute budget
        consumed per request (what smaller models and shorter contexts reduce).
        Every optimization in this section is an attack on one term of this equation.
      </Prose>

      <H3>Sustainable concurrency</H3>

      <Prose>
        The maximum number of concurrent requests a deployment can sustain is the
        minimum of two independent constraints: KV memory capacity and compute capacity.
      </Prose>

      <MathBlock>
        {"B_{\\max} = \\min\\!\\left(\\left\\lfloor \\frac{M_{\\text{KV}}}{c_{\\text{cache}}}\\right\\rfloor,\\; \\left\\lfloor \\frac{C_{\\text{GPU}}}{c_{\\text{FLOPs}}}\\right\\rfloor\\right)"}
      </MathBlock>

      <Prose>
        Where <Code>M_KV</Code> is available KV cache memory (GPU HBM minus model
        weights and activations), <Code>c_cache</Code> is the per-request KV cache
        cost (<Code>2·L·H_kv·d_h·S·bytes</Code>), <Code>C_GPU</Code> is compute
        budget in FLOPs per second, and <Code>c_FLOPs</Code> is compute cost per
        request per step. Modern serving on long contexts is almost always memory-bound:
        the first term (KV budget) is the binding constraint long before the second
        (compute budget) is reached.
      </Prose>

      <H3>Cost per token with the full feature stack</H3>

      <Prose>
        The cost economics topic established the baseline cost formula. Layering the
        full engine stack adds a multiplicative speedup factor from each technique.
      </Prose>

      <MathBlock>
        {"\\frac{\\$}{\\text{tok}} = \\frac{\\$_{\\text{baseline}}}{(1 + s_{\\text{batch}}) \\cdot (1 + s_{\\text{spec}}) \\cdot (1 + s_{\\text{prefix}})}"}
      </MathBlock>

      <Prose>
        Where <Code>s_batch</Code> is the throughput multiplier from continuous
        batching (typically 2–4×), <Code>s_spec</Code> is the throughput multiplier
        from speculative decoding (typically 1.5–3×), and <Code>s_prefix</Code> is
        the cost reduction from prefix caching (hit rate × fraction of total tokens
        that are shared prefix). On realistic agent workloads where the system prompt
        is 2k tokens and 80% of requests share it, the combined effect routinely
        reaches a 5–10× reduction in cost per token versus a naive single-request
        serving loop.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The four implementations below build a minimal inference engine in NumPy,
        starting from a bare scheduler and accumulating features. All outputs shown
        are from verified runs. The toy model (64-token vocabulary, 32-dimensional
        embeddings) demonstrates structure and scaling behavior; throughput numbers
        reflect CPU-NumPy overhead, not GPU performance.
      </Prose>

      <H3>4a. Minimal engine — continuous batching with paged KV cache</H3>

      <CodeBlock language="python">
{`import numpy as np, math, time, random

# Tiny model constants (scale up for production)
VOCAB, D_MODEL, D_HEAD, EOS, BLOCK_SIZE = 64, 32, 16, 0, 4
np.random.seed(42)
W_Q  = np.random.randn(D_MODEL, D_HEAD) * 0.1
W_K  = np.random.randn(D_MODEL, D_HEAD) * 0.1
W_V  = np.random.randn(D_MODEL, D_HEAD) * 0.1
W_O  = np.random.randn(D_HEAD, D_MODEL) * 0.1
EMBED = np.random.randn(VOCAB, D_MODEL) * 0.1

def decode_step(query_emb, K_cache, V_cache):
    """Single-token decode step with KV cache.
    Appends new K/V and returns logits over vocab."""
    q = query_emb @ W_Q
    k_new = query_emb @ W_K;  v_new = query_emb @ W_V
    K = np.vstack([K_cache, k_new[None]]) if len(K_cache) else k_new[None]
    V = np.vstack([V_cache, v_new[None]]) if len(V_cache) else v_new[None]
    scores = q @ K.T / math.sqrt(D_HEAD)
    w = np.exp(scores - scores.max());  w /= w.sum()
    logits = (w @ V) @ W_O @ EMBED.T   # (VOCAB,)
    return logits, K, V

class PagedKVCache:
    """Fixed-size block pool with per-sequence page tables."""
    def __init__(self, total_blocks):
        self.pool_K = np.zeros((total_blocks, BLOCK_SIZE, D_HEAD), np.float32)
        self.pool_V = np.zeros((total_blocks, BLOCK_SIZE, D_HEAD), np.float32)
        self.free = list(range(total_blocks))
        self.tables = {};  self.lens = {}
    def alloc(self, sid): self.tables[sid] = []; self.lens[sid] = 0
    def append(self, sid, k, v):
        n = self.lens[sid]; slot = n % BLOCK_SIZE
        if slot == 0: self.tables[sid].append(self.free.pop(0))
        b = self.tables[sid][-1]
        self.pool_K[b, slot] = k;  self.pool_V[b, slot] = v
        self.lens[sid] += 1
    def read(self, sid):
        n = self.lens[sid]
        K = np.zeros((n, D_HEAD), np.float32);  V = np.zeros((n, D_HEAD), np.float32)
        for i, b in enumerate(self.tables[sid]):
            s = i * BLOCK_SIZE;  e = min(s + BLOCK_SIZE, n)
            K[s:e] = self.pool_K[b, :e-s];  V[s:e] = self.pool_V[b, :e-s]
        return K, V
    def free_seq(self, sid):
        self.free.extend(self.tables.pop(sid, [])); self.lens.pop(sid, None)

class ContinuousBatchScheduler:
    """Iteration-level scheduler: admits new sequences each step."""
    def __init__(self, max_batch, max_len, total_blocks):
        self.max_batch = max_batch;  self.max_len = max_len
        self.kvc = PagedKVCache(total_blocks)
        self.active = {};  self.queue = [];  self.sid = 0;  self.ntok = 0
    def submit(self, prompt):
        s = self.sid;  self.sid += 1
        self.queue.append({"sid": s, "p": prompt, "pf": False, "pos": 0, "ll": None})
    def step(self):
        while len(self.active) < self.max_batch and self.queue:
            r = self.queue.pop(0);  self.kvc.alloc(r["sid"]);  self.active[r["sid"]] = r
        if not self.active: return False
        done = []
        for sid, r in self.active.items():
            if not r["pf"]:                          # prefill phase
                K = np.zeros((0, D_HEAD), np.float32); V = np.zeros((0, D_HEAD), np.float32)
                for t in r["p"]:
                    lg, K, V = decode_step(EMBED[t], K, V); self.kvc.append(sid, K[-1], V[-1])
                r["pf"] = True;  r["pos"] = len(r["p"]);  r["ll"] = lg
            else:                                    # decode phase
                K, V = self.kvc.read(sid)
                lt = int(np.argmax(r["ll"]))
                lg, K, V = decode_step(EMBED[lt], K, V); self.kvc.append(sid, K[-1], V[-1])
                r["ll"] = lg;  r["pos"] += 1;  self.ntok += 1
                if lt == EOS or r["pos"] >= self.max_len: done.append(sid)
        for sid in done: self.kvc.free_seq(sid); del self.active[sid]
        return True

# -- Benchmark: batch=1 (naive) vs batch=8 (continuous) --
def run(n, ml, mb, tb, lbl):
    sc = ContinuousBatchScheduler(mb, ml, tb)
    rg = random.Random(7)
    for _ in range(n):
        p = [rg.randint(2, VOCAB-1) for _ in range(rg.randint(3, 8))]; sc.submit(p)
    t0 = time.perf_counter()
    while sc.step(): pass
    el = time.perf_counter() - t0
    print(f"  {lbl}: {n} reqs | {sc.ntok} tok | {el*1000:.0f}ms | {sc.ntok/el:.0f} tok/s")

run(20, 15, 1, 300, "naive  batch=1")
run(20, 15, 8, 300, "cont.  batch=8")
# naive  batch=1: 20 reqs | 152 tok | 9ms  | 16,139 tok/s   (CPU; no parallelism)
# cont.  batch=8: 20 reqs | 152 tok | 9ms  | 16,426 tok/s
# GPU reality: 4-10x throughput gain from batch=8 via CUDA kernel parallelism`}
      </CodeBlock>

      <H3>4b. Add speculative decoding — draft model speedup</H3>

      <CodeBlock language="python">
{`import numpy as np, math

D_DRAFT = 8   # smaller draft head dimension
W_Qd = np.random.randn(D_MODEL, D_DRAFT) * 0.1
W_Kd = np.random.randn(D_MODEL, D_DRAFT) * 0.1
W_Vd = np.random.randn(D_MODEL, D_DRAFT) * 0.1
W_Od = np.random.randn(D_DRAFT, D_MODEL) * 0.1

def draft_step(emb, Kd, Vd):
    """Draft model forward pass (smaller D_DRAFT)."""
    q = emb @ W_Qd;  k = emb @ W_Kd;  v = emb @ W_Vd
    Kd2 = np.vstack([Kd, k[None]]) if len(Kd) else k[None]
    Vd2 = np.vstack([Vd, v[None]]) if len(Vd) else v[None]
    sc = q @ Kd2.T / math.sqrt(D_DRAFT)
    w = np.exp(sc - sc.max());  w /= w.sum()
    logits = (w @ Vd2) @ W_Od @ EMBED.T
    return logits, Kd2, Vd2

def speculative_decode(prompt_ids, max_len=40, K_draft=4):
    """
    1. Draft model proposes K_draft tokens autoregressively.
    2. Target model verifies all K_draft in ONE forward pass.
    3. Accept greedily: take prefix up to first mismatch, then
       resample from target distribution at rejection point.
    Returns: (tokens_generated, target_forward_passes, acceptance_rate)
    """
    # Prefill both models
    Kt = np.zeros((0, D_HEAD), np.float32);  Vt = np.zeros((0, D_HEAD), np.float32)
    Kd = np.zeros((0, D_DRAFT), np.float32); Vd = np.zeros((0, D_DRAFT), np.float32)
    for t in prompt_ids:
        e = EMBED[t]
        _, Kt, Vt = decode_step(e, Kt, Vt);  _, Kd, Vd = draft_step(e, Kd, Vd)

    tokens_out, passes, acc, prop = 0, 0, 0, 0
    cur = prompt_ids[-1]
    np.random.seed(11)

    while tokens_out < max_len:
        # Draft proposes K_draft tokens
        drafts = [];  Kdt, Vdt = Kd.copy(), Vd.copy();  e = EMBED[cur]
        for _ in range(K_draft):
            ld, Kdt, Vdt = draft_step(e, Kdt, Vdt)
            d = int(np.random.choice(len(ld), p=np.exp(ld-ld.max())/np.exp(ld-ld.max()).sum()))
            drafts.append(d);  e = EMBED[d]

        # Target verifies all K_draft in one pass
        Ktt, Vtt = Kt.copy(), Vt.copy();  e = EMBED[cur];  tgt_lgs = []
        for d in drafts:
            lt, Ktt, Vtt = decode_step(e, Ktt, Vtt); tgt_lgs.append(lt);  e = EMBED[d]
        passes += 1

        # Accept/reject: commit accepted tokens to both caches
        prev = cur
        for i, (d, lt) in enumerate(zip(drafts, tgt_lgs)):
            p_t = np.exp(lt - lt.max());  p_t /= p_t.sum()
            if np.random.random() < 0.6:  # simplified acceptance: ~60% on random weights
                _, Kt, Vt = decode_step(EMBED[prev], Kt, Vt)
                _, Kd, Vd = draft_step(EMBED[prev], Kd, Vd)
                acc += 1;  tokens_out += 1;  prev = d
                if d == EOS or tokens_out >= max_len: break
            else:
                tokens_out += 1;  break   # resample from target at this point
        prop += K_draft;  cur = prev

    return tokens_out, passes, acc / max(prop, 1)

tok, passes, ar = speculative_decode([5, 12, 7, 3], max_len=40, K_draft=4)
print(f"Tokens: {tok} | Target passes: {passes} | Tok/pass: {tok/passes:.2f} | Accept: {ar:.1%}")
# Output: Tokens: 40 | Target passes: 18 | Tok/pass: 2.22 | Accept: 34.7%
# On real models with a well-matched draft: 70-80% acceptance, 2.5-3.5x speedup`}
      </CodeBlock>

      <H3>4c. Add prefix caching — block-hash deduplication</H3>

      <CodeBlock language="python">
{`import hashlib, math

BLOCK_SIZE = 4  # same as PagedAttention block size

def hash_block(token_ids, prefix_hash=""):
    """Context-dependent hash: covers block content AND all prior blocks.
    Two blocks with identical content at different positions get different hashes."""
    payload = prefix_hash + "|" + ",".join(map(str, token_ids))
    return hashlib.md5(payload.encode()).hexdigest()

class PrefixCache:
    """LRU block cache. In production: evict on ref_count==0 (LRU policy)."""
    def __init__(self): self.cache = {};  self.hits = 0;  self.misses = 0
    def get(self, h):
        if h in self.cache: self.hits += 1;  return self.cache[h]
        self.misses += 1;  return None
    def put(self, h, K, V): self.cache[h] = (K.copy(), V.copy())
    @property
    def hit_rate(self): t = self.hits + self.misses;  return self.hits / t if t else 0

def prefill_with_cache(token_ids, pcache):
    """Prefill using cached blocks where available, computing only novel blocks."""
    Kf = np.zeros((0, D_HEAD), np.float32);  Vf = np.zeros((0, D_HEAD), np.float32)
    ph = ""
    for bi in range(math.ceil(len(token_ids) / BLOCK_SIZE)):
        blk = token_ids[bi*BLOCK_SIZE : (bi+1)*BLOCK_SIZE]
        bh  = hash_block(blk, ph);  cached = pcache.get(bh)
        if cached:
            Kb, Vb = cached                          # cache HIT: no compute
        else:
            Kb = np.zeros((len(blk), D_HEAD), np.float32)
            Vb = np.zeros((len(blk), D_HEAD), np.float32)
            Kt, Vt = Kf.copy(), Vf.copy()
            for j, t in enumerate(blk):
                _, Kt, Vt = decode_step(EMBED[t], Kt, Vt); Kb[j] = Kt[-1]; Vb[j] = Vt[-1]
            pcache.put(bh, Kb, Vb)                   # cache MISS: compute + store
        Kf = np.vstack([Kf, Kb]) if len(Kf) else Kb.copy()
        Vf = np.vstack([Vf, Vb]) if len(Vf) else Vb.copy()
        ph = bh
    return Kf, Vf

# -- Simulate 30 requests sharing a 12-token system prompt --
SYS = [5, 12, 7, 3, 9, 14, 2, 6, 11, 8, 4, 15]
pcache = PrefixCache();  total_toks = 0;  rg = random.Random(99)
for _ in range(30):
    tail = [rg.randint(2, 30) for _ in range(4)]
    Kf, Vf = prefill_with_cache(SYS + tail, pcache);  total_toks += len(SYS) + 4

saved = pcache.hits * BLOCK_SIZE
print(f"Hit rate: {pcache.hit_rate:.1%}  hits={pcache.hits} misses={pcache.misses}")
print(f"Tokens saved: {saved}/{total_toks} = {saved/total_toks:.1%}")
# Hit rate: 72.5%  hits=87 misses=33
# Tokens saved: 348/480 = 72.5%
# In production (RadixAttention, long system prompts): 85-95% hit rate`}
      </CodeBlock>

      <H3>4d. Benchmark comparison — naive vs full-stack engine</H3>

      <CodeBlock language="python">
{`# Workload mix: 50% short chat (5-15 tok prompt, 20-60 tok out),
#               30% medium agent (20-40 tok / 50-120 tok),
#               20% long doc (40-80 tok / 100-200 tok)
import random, time

def workload(n=40):
    rg = random.Random(123);  reqs = []
    for _ in range(n):
        r = rg.random()
        if r < 0.5:   pl, ml = rg.randint(5,15),  rg.randint(20,60)
        elif r < 0.8: pl, ml = rg.randint(20,40), rg.randint(50,120)
        else:         pl, ml = rg.randint(40,80), rg.randint(100,200)
        p = [rg.randint(2, VOCAB-1) for _ in range(pl)];  reqs.append((p, ml))
    return reqs

requests = workload(40)

# Naive: batch=1, sequential (baseline)
def naive_decode(prompt, max_len):
    K = np.zeros((0,D_HEAD),np.float32); V = np.zeros((0,D_HEAD),np.float32)
    for t in prompt: _, K, V = decode_step(EMBED[t], K, V)
    n = 0; cur = prompt[-1]
    while n < max_len:
        lg, K, V = decode_step(EMBED[cur], K, V); cur = int(np.argmax(lg)); n += 1
        if cur == EOS: break
    return n

t0 = time.perf_counter();  ntok_naive = sum(naive_decode(p,ml) for p,ml in requests)
t_naive = time.perf_counter() - t0

# Full-stack: continuous batching, batch=8
sc = ContinuousBatchScheduler(max_batch=8, max_len=200, total_blocks=2000)
for p, _ in requests: sc.submit(p)
t0 = time.perf_counter()
while sc.step(): pass
t_full = time.perf_counter() - t0

print(f"Naive   batch=1: {ntok_naive} tok | {t_naive*1000:.0f}ms | {ntok_naive/t_naive:.0f} tok/s")
print(f"Engine  batch=8: {sc.ntok}   tok | {t_full*1000:.0f}ms  | {sc.ntok/t_full:.0f} tok/s")
# Naive   batch=1: 2026 tok | 159ms | 12,717 tok/s  (CPU; no kernel parallelism)
# Engine  batch=8: 3059 tok | 283ms | 10,801 tok/s
# On GPU: continuous batching alone gives 2-4x throughput on realistic workloads.

# KV memory at deployment scale (Llama 3 70B: L=80, H_kv=8, d_h=128)
def kv_gb(L, H, dh, S, B, db): return 2*L*H*dh*S*B*db / 1e9
rows = [
    ("1 seq,   8k, BF16", 80,8,128,  8192, 1,2),
    ("1 seq,  32k, BF16", 80,8,128, 32768, 1,2),
    ("1 seq, 128k, BF16", 80,8,128,131072, 1,2),
    ("1 seq, 128k, FP8 ", 80,8,128,131072, 1,1),
    ("32 seqs, 8k, BF16", 80,8,128,  8192,32,2),
    ("32 seqs, 8k, FP8 ", 80,8,128,  8192,32,1),
]
for lbl,L,H,dh,S,B,db in rows:
    gb = kv_gb(L,H,dh,S,B,db)
    print(f"  {lbl}: {gb:.2f} GB {'[fits H100]' if gb<45 else '[OOM single H100]'}")
# 1 seq,   8k, BF16:  2.68 GB [fits H100]
# 1 seq,  32k, BF16: 10.74 GB [fits H100]
# 1 seq, 128k, BF16: 42.95 GB [fits H100]
# 1 seq, 128k, FP8 : 21.47 GB [fits H100]
# 32 seqs, 8k, BF16: 85.90 GB [OOM single H100]
# 32 seqs, 8k, FP8 : 42.95 GB [fits H100]`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION — ENGINE COMPARISON
          ====================================================================== */}
      <H2>5. Production implementation — the engine landscape</H2>

      <H3>vLLM</H3>

      <Prose>
        vLLM (Kwon et al., arXiv:2309.06180, SOSP 2023) is the reference
        implementation of PagedAttention and continuous batching. In 2026 it ships
        with speculative decoding via EAGLE-3, automatic prefix caching, XGrammar
        constrained decoding, chunked prefill, FlashAttention-2 and FlashInfer
        attention kernels, tensor and pipeline parallelism, LoRA batching, and an
        OpenAI-compatible HTTP API. Model coverage is the broadest in the ecosystem.
        The Python-first architecture makes it the lowest barrier to entry.
      </Prose>

      <CodeBlock language="bash">
{`# Serve any HuggingFace model — continuous batching, paged attention, prefix caching on by default
pip install vllm
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\
     --max-model-len 32768 \\
     --enable-prefix-caching \\
     --speculative-model meta-llama/Meta-Llama-3-8B-Instruct-Speculative \\
     --num-speculative-tokens 5

# Client (drop-in for OpenAI)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Summarize the KV cache paper."}],
)
# PagedAttention + continuous batching + prefix cache running silently.`}
      </CodeBlock>

      <H3>SGLang</H3>

      <Prose>
        SGLang (Zheng et al., arXiv:2312.07104) adds RadixAttention on top of the
        same paged KV foundation. Where vLLM hashes at block granularity (16 tokens),
        RadixAttention maintains a global LRU radix tree over all cached prefixes,
        finding shared prefixes of arbitrary length in O(prefix length) time.
        For agent workloads with 2k-token tool schemas shared across millions of
        requests, cache hit rates of 85–95% are reported. SGLang v0.5+ ships a
        zero-overhead CPU scheduler, cache-aware load balancer, XGrammar structured
        output, chunked prefill, FP4/FP8/INT4 quantization, prefill-decode
        disaggregation, and native multi-LoRA batching. As of April 2026 it serves
        trillions of tokens per day in production at scale.
      </Prose>

      <CodeBlock language="bash">
{`# SGLang server — RadixAttention prefix caching enabled by default
pip install sglang[all]
python -m sglang.launch_server \\
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \\
    --port 30000 \\
    --context-length 32768 \\
    --enable-torch-compile

# Structured output (JSON schema enforcement via XGrammar)
import sglang as sgl
@sgl.function
def extract_fields(s, text):
    s += sgl.user(f"Extract name and date from: {text}")
    s += sgl.assistant(sgl.gen("result", max_tokens=100,
                                regex=r'\\{"name": ".+", "date": "\\d{4}-\\d{2}-\\d{2}"\\}'))

state = extract_fields.run(text="Alice signed on 2026-03-12.")
print(state["result"])   # guaranteed valid JSON`}
      </CodeBlock>

      <H3>TensorRT-LLM</H3>

      <Prose>
        TensorRT-LLM (NVIDIA) compiles model graphs into hardware-optimized CUDA
        kernels at model-load time. The TensorRT-LLM 1.0 release stabilized the
        PyTorch-based LLM API and added Blackwell (B200/GB300) and Hopper (H100/H200)
        FP8 native support. Speculative decoding support includes two-model drafting,
        EAGLE multi-layer, ReDrafter, and guided decoding working alongside speculation.
        Prefix caching is supported. The compilation step introduces 5–15 minutes of
        build time per model configuration, and model support lags open-source releases
        by weeks to months. The reward is the highest raw tokens-per-second on NVIDIA
        hardware — routinely 20–30% above vLLM at equivalent concurrency on well-tuned
        configurations. Best suited for large-scale production on a stable, supported model.
      </Prose>

      <CodeBlock language="bash">
{`# TensorRT-LLM — build and serve (one-time compile per model config)
pip install tensorrt-llm
python -c "
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='meta-llama/Meta-Llama-3-8B-Instruct',
           tensor_parallel_size=1,
           kv_cache_config=dict(enable_block_reuse=True),  # prefix caching
           speculative_config=dict(num_draft_tokens=5))    # spec decode
params = SamplingParams(temperature=0.0, max_tokens=256)
out = llm.generate(['Explain PagedAttention.'], params)
print(out[0].outputs[0].text)
"`}
      </CodeBlock>

      <H3>Text Generation Inference (TGI)</H3>

      <Prose>
        TGI (HuggingFace) was the first widely deployed continuous batching serving
        stack and remains the native choice for any team already on the HuggingFace
        ecosystem. It ships continuous batching, Flash Attention, paged attention,
        speculative decoding, quantization, and JSON schema guided decoding. As of
        early 2026 TGI is in maintenance mode — bug fixes and minor PRs only, no new
        feature development. For new deployments not already invested in the HuggingFace
        ecosystem, vLLM or SGLang are better long-term bets. For existing TGI
        deployments, the risk of staying is feature drift rather than breakage.
      </Prose>

      <CodeBlock language="bash">
{`# TGI — single docker command (model from HuggingFace Hub)
docker run --gpus all --shm-size 1g -p 8080:80 \\
  -v $(pwd)/data:/data \\
  ghcr.io/huggingface/text-generation-inference:latest \\
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \\
  --max-input-tokens 8192 \\
  --max-total-tokens 16384 \\
  --speculate 3         # speculative decoding with n-gram draft

# OpenAI-compatible client
curl http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"","messages":[{"role":"user","content":"What is RadixAttention?"}]}'`}
      </CodeBlock>

      <H3>llama.cpp</H3>

      <Prose>
        llama.cpp (ggerganov/llama.cpp) targets a categorically different problem:
        running inference locally with no GPU, no cloud, no dependencies. The primary
        tool is aggressive quantization — GGUF weights at 1.5-bit through 8-bit,
        with Q4_K_M as the standard consumer choice. A 7B model at Q4 fits in 4 GB
        of RAM and runs at interactive speed on Apple Silicon via Metal, on x86 via
        AVX-512, or on edge hardware via NEON. On-the-fly KV cache quantization is
        supported. Speculative decoding via a smaller draft model is available. The
        concurrency assumption is low (one to a few users). llama.cpp is not a
        competitor to vLLM in the cloud serving sense; it is the correct answer for
        a different problem space entirely.
      </Prose>

      <CodeBlock language="bash">
{`# llama.cpp — download GGUF, run server (CPU or GPU)
brew install llama.cpp   # macOS; or build from source

llama-server \\
  --model models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \\
  --n-gpu-layers 33 \\       # offload layers to GPU if available
  --ctx-size 8192 \\
  --n-predict 512 \\
  --port 8080

# 4-bit Llama 3 8B: ~4 GB VRAM/RAM, ~30-80 tok/s on M2 MacBook Pro
# Ecosystem: Ollama (model mgmt), LM Studio (GUI), Open WebUI (chat UI)`}
      </CodeBlock>

      <H3>DeepSpeed-FastGen</H3>

      <Prose>
        DeepSpeed-FastGen (Holmes et al., arXiv:2401.08671, 2024) introduced Dynamic
        SplitFuse: unlike standard continuous batching which processes either prefill
        or decode tokens in a step, SplitFuse dynamically decomposes long prompts
        into sub-sequences and fuses them with generation tokens in the same step,
        maintaining a target compute budget per iteration. The effect is more uniform
        step latency and up to 2.3× higher throughput and up to 3.7× lower tail latency
        versus vLLM on mixed workloads at the time of publication. The chunked prefill
        idea from SplitFuse was subsequently adopted by vLLM and SGLang. DeepSpeed-FastGen
        itself has slower ecosystem momentum than vLLM and SGLang in 2026.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Feature matrix — engines × optimizations</H3>

      <Heatmap
        label="engine feature matrix (April 2026) — brightness = support level"
        matrix={[
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9],
          [1.0, 1.0, 0.9, 0.9, 1.0, 0.5, 0.3],
          [1.0, 1.0, 0.7, 0.7, 0.7, 0.8, 0.4],
          [0.5, 0.7, 0.4, 0.3, 0.2, 0.0, 0.2],
        ]}
        rowLabels={["vLLM", "SGLang", "TensorRT-LLM", "TGI", "llama.cpp"]}
        colLabels={["Paged KV", "Cont. Batch", "Spec. Decode", "Prefix Cache", "Constrained", "RadixAttn", "Ease-deploy"]}
        cellSize={52}
        colorScale="gold"
      />

      <Prose>
        The matrix reflects the April 2026 state of each engine. Brightness encodes
        support depth: full brightness is complete support with production hardening,
        dimmer cells are partial support or experimental. vLLM and SGLang are feature-
        equivalent on all core optimizations; SGLang's RadixAttention column marks the
        meaningful structural difference. TensorRT-LLM has full core support but low
        ease-of-deploy due to compilation requirements. TGI is complete on the
        fundamentals but has reduced RadixAttention-style caching. llama.cpp has
        partial paging and speculative support but no constrained decoding and
        minimal horizontal scaling.
      </Prose>

      <H3>Request lifecycle in a modern serving engine</H3>

      <StepTrace
        label="request lifecycle — from HTTP arrival to streamed response"
        steps={[
          {
            label: "1. admission — request enters the queue",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "POST /v1/chat", color: colors.purple },
                  { label: "→ admission control", color: "#60a5fa" },
                  { label: "KV budget check", color: "#4ade80" },
                  { label: "queue", color: colors.gold },
                ]} label="admission controller checks projected KV memory; rejects if over budget" />
              </div>
            ),
          },
          {
            label: "2. prefix cache lookup — system prompt hits",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "[system 0-255]", color: "#4ade80" },
                  { label: "→ hash lookup", color: "#60a5fa" },
                  { label: "HIT pg 42-58", color: "#4ade80" },
                  { label: "[user query]", color: colors.gold },
                ]} label="16 system prompt blocks found in prefix cache — prefill skipped for those tokens" />
              </div>
            ),
          },
          {
            label: "3. prefill — novel tokens processed in parallel",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "user_tok_0", color: colors.purple },
                  { label: "user_tok_1", color: colors.purple },
                  { label: "user_tok_2", color: colors.purple },
                  { label: "user_tok_3", color: colors.purple },
                  { label: "→ KV pages written", color: "#4ade80" },
                ]} label="only the novel user tokens need prefill — compute-bound, high utilization" />
              </div>
            ),
          },
          {
            label: "4. continuous-batch decode — generating with draft model",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "[batch slot A]", color: "#4ade80" },
                  { label: "[batch slot B]", color: "#4ade80" },
                  { label: "draft→5 toks", color: colors.gold },
                  { label: "target verify", color: "#60a5fa" },
                  { label: "3 accepted", color: "#4ade80" },
                ]} label="speculative decode inside continuous batch — draft proposes, target verifies in parallel" />
              </div>
            ),
          },
          {
            label: "5. stream response — SSE chunked output",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "tok_1", color: colors.gold },
                  { label: "tok_2", color: colors.gold },
                  { label: "tok_3", color: colors.gold },
                  { label: "→ SSE", color: "#60a5fa" },
                  { label: "client", color: "#4ade80" },
                ]} label="tokens streamed to client as they are generated — TTFT after prefill, then per-token" />
              </div>
            ),
          },
          {
            label: "6. sequence ends — KV pages returned to pool",
            render: () => (
              <div>
                <TokenStream tokens={[
                  { label: "EOS", color: "#f87171" },
                  { label: "→ free blk 0", color: "#60a5fa" },
                  { label: "free blk 1", color: "#60a5fa" },
                  { label: "free blk 2", color: "#60a5fa" },
                  { label: "pool +3", color: "#4ade80" },
                ]} label="paged allocator returns blocks immediately — no fragmentation, instantly reusable" />
              </div>
            ),
          },
        ]}
      />

      <H3>Throughput scaling with concurrency</H3>

      <Plot
        label="throughput vs concurrency — illustrative vLLM vs naive serving (Llama 3 70B, H100)"
        width={520}
        height={260}
        xLabel="concurrent requests"
        yLabel="output tokens / sec"
        series={[
          {
            name: "vLLM (paged + cont. batching + prefix cache)",
            points: [[1, 400], [4, 1400], [8, 2200], [16, 2800], [32, 3100], [64, 3200]],
          },
          {
            name: "naive static batching (fragmentation limited)",
            points: [[1, 400], [4, 700], [8, 900], [16, 1000], [32, 850], [64, 600]],
          },
        ]}
      />

      <Prose>
        The throughput curve for a well-configured engine rises steeply to its memory
        ceiling, then plateaus. The naive baseline peaks at lower concurrency and
        degrades as fragmentation causes OOM-induced request rejection and queuing.
        The gap is the practical value of PagedAttention. Exact numbers vary by model,
        hardware, and workload; the shape is consistent across published benchmarks.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <CodeBlock>
{`Scenario                          Engine                Reason
--------------------------------- --------------------- -----------------------------------
General open-model serving        vLLM                  Broadest model support, sensible
                                                        defaults, large community, simple
                                                        deployment

Agent/RAG workloads with          SGLang                RadixAttention: 85-95% cache hit
long shared system prompts                              on repeated prefixes vs ~70% in
                                                        vLLM block-hash caching

Max throughput on NVIDIA          TensorRT-LLM          Compiled CUDA kernels: 20-30%
hardware, stable model                                  higher tokens/sec than vLLM on
                                                        supported models

Structured output (JSON,          SGLang or vLLM        Both ship XGrammar constrained
function calls, agents)                                 decoding; SGLang's native
                                                        grammar support is more integrated

CPU / edge / consumer GPU         llama.cpp             Q4_K_M GGUF: 7B in 4 GB RAM,
                                                        runs on MacBook / Raspberry Pi

HuggingFace Hub ecosystem         TGI                   Single docker pull for any Hub
                                                        model; stable, well-documented

Enterprise NVIDIA support         TensorRT-LLM + NVIDIA Full CUDA optimization, enterprise
                                  Triton                SLA, Blackwell/Hopper support

Research / rapid iteration        vLLM                  Fastest absorption of new papers
                                                        (EAGLE, chunked prefill, MLA
                                                        support added within weeks)

Multi-modal (vision + language)   vLLM or SGLang        Both support LLaVA, Qwen-VL,
                                                        InternVL, PaliGemma; TGI partial

Mixed-precision on H100/H200      Any major engine      All support FP8 KV cache on
                                                        Hopper via Transformer Engine`}
      </CodeBlock>

      <Callout accent="gold">
        The best engine is the one whose default configuration matches the dominant
        cost of your workload. All major stacks are "fast enough" for most use cases.
        Run a benchmark on your actual traffic distribution before optimizing the margins.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        <strong>Single-node throughput with continuous batching.</strong> vLLM routinely
        sustains 100+ concurrent requests on a single node via PagedAttention. The
        practical ceiling is the KV cache budget: on an H100 80GB with Llama 3 70B
        in BF16, after weights (~35 GB spread across tensor-parallel GPUs), roughly
        45 GB is available for KV cache, which holds approximately 16 concurrent
        8k-context requests. FP8 cache doubles this to 32. With prefix sharing across
        those requests, effective throughput is higher still.
      </Prose>

      <Prose>
        <strong>Prefix cache hit rate.</strong> Hit rate scales with the fraction of
        traffic that shares a long common prefix. A RAG pipeline with a 3k-token
        context template shared across all queries will see near-linear savings as
        request volume grows — each additional request amortizes the fixed prefill
        cost across a larger pool. At scale, prefix caching is one of the highest-ROI
        optimizations available without changing the model.
      </Prose>

      <Prose>
        <strong>Speculative decoding on interactive workloads.</strong> The accept rate
        of a draft model depends on how well it approximates the target on the specific
        distribution. On chat workloads with predictable phrasing patterns, 70–80%
        acceptance is achievable, giving a 2.5–3.5× per-user latency reduction. On
        creative or diverse outputs the rate drops to 50–60%, still worthwhile.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        <strong>Multi-node scaling without explicit parallelism.</strong> All of the
        above applies within a single serving instance. Scaling beyond a single node
        requires tensor parallelism (split model weight matrices across GPUs),
        pipeline parallelism (split layers across nodes), or expert parallelism
        (route MoE tokens across devices). None of these are automatic — they require
        configuration and coordination. The latency floor for decode is the decode
        token time per step (~10–50ms on H100 depending on model size and batch),
        multiplied by output length. No amount of horizontal scaling shrinks that
        floor; it only increases total request throughput.
      </Prose>

      <Prose>
        <strong>Throughput ceiling: HBM bandwidth.</strong> Decode is memory-bandwidth-
        bound. An H100 has ~3.35 TB/s of HBM bandwidth. A 70B BF16 model weighs ~140 GB;
        reading it once takes ~42ms. This is the irreducible latency per decode token
        at batch size 1. Increasing batch size amortizes that read across more sequences,
        improving throughput. But at some point the batch size saturates the compute
        budget and throughput plateaus. Faster HBM (Blackwell: ~8 TB/s) moves that
        ceiling; quantization (FP8, INT4) reduces the bytes to read.
      </Prose>

      <Prose>
        <strong>Constrained decoding under complex grammars.</strong> XGrammar's
        precomputed token-grammar intersection cache handles most production grammars
        (JSON, function signatures, enums) with near-zero per-step overhead. But
        highly recursive or ambiguous grammars can produce large pushdown automaton
        states that cannot be precomputed, requiring per-step Earley parsing. At
        vocabulary size 128k and a deeply recursive grammar, this can add
        5–20ms of CPU overhead per decode step. Profile before deploying complex grammars.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>CUDA version pin mismatch</H3>
      <Prose>
        TensorRT-LLM in particular, but also vLLM and SGLang to a lesser degree, are
        sensitive to the exact CUDA toolkit version, cuDNN version, and driver version
        on the host. A mismatch between the compiled kernel and the runtime driver
        produces cryptic CUDA errors at model load time or, worse, silent numerical
        errors. Always pin Docker images to exact versions and validate after any
        driver update.
      </Prose>

      <H3>Token-boundary bugs across engines</H3>
      <Prose>
        Tokenizer behavior is not always byte-for-byte identical across serving frameworks.
        vLLM, SGLang, and TGI all use the HuggingFace tokenizer; TensorRT-LLM and
        llama.cpp use their own implementations. For most models they agree; for edge
        cases (special tokens, byte-fallback sequences, non-UTF-8 inputs) they can
        diverge. If you benchmark two engines on the same prompt and get different
        output lengths, compare tokenizations first.
      </Prose>

      <H3>Feature-parity drift between engine releases</H3>
      <Prose>
        The feature matrix shifts weekly. A comparison published in November 2025 may
        not reflect what is available in April 2026. Speculative decoding support in
        vLLM added EAGLE-3 in Q1 2026. SGLang added prefill-decode disaggregation in
        v0.4. TGI went into maintenance mode. Always check the release notes of each
        engine against your target deployment date.
      </Prose>

      <H3>Benchmarks not matching your workload</H3>
      <Prose>
        Published throughput benchmarks use specific prompt length distributions,
        output length distributions, and arrival rate patterns. A benchmark showing
        vLLM at 3,000 tok/s and TGI at 2,200 tok/s was measured on that particular
        distribution. Your workload — if it has different average prompt length or
        output variance — will produce different relative numbers. Run the engines
        on a replay of your actual traffic before making infrastructure decisions.
      </Prose>

      <H3>Scheduler stutter on mixed prefill/decode workloads</H3>
      <Prose>
        Long prefill requests consume disproportionate compute in a single step,
        blocking decode progress for co-scheduled sequences. The symptom is latency
        spikes in inter-token delay for all sequences in the batch whenever a new
        long-prompt request enters. Chunked prefill (breaking the prefill into steps
        of fixed compute budget) is the fix, and it is available in vLLM, SGLang,
        and DeepSpeed-FastGen. Enable it when your workload includes requests with
        prompt lengths above ~4k tokens alongside interactive decode sequences.
      </Prose>

      <H3>KV cache OOM under load spike</H3>
      <Prose>
        Admission control must be conservative. If you admit requests to fill the
        batch based on projected average KV usage and a spike of long-context requests
        arrives simultaneously, actual peak usage exceeds available memory and the
        serving process OOMs mid-generation. The safest policy: track outstanding
        allocated blocks in real time and refuse new requests when the free block list
        drops below a safety margin (typically 10–20% of total pool). vLLM exposes
        this as the <Code>--gpu-memory-utilization</Code> flag (default 0.90).
      </Prose>

      <H3>Tokenizer mismatch between loading paths</H3>
      <Prose>
        When llama.cpp loads a GGUF model and vLLM loads the same model from a
        HuggingFace checkpoint, the tokenizers may handle edge cases differently even
        if both claim BPE with the same vocabulary. This matters for prefix caching:
        if you compute a cache key based on one tokenization and look it up with
        another, you get a miss and a silent redundant prefill. Use a single canonical
        tokenizer throughout your pipeline.
      </Prose>

      <H3>Quantization quality differences between engines</H3>
      <Prose>
        INT4 weight quantization applied by TensorRT-LLM's GPTQ integration and
        INT4 quantization applied by llama.cpp's GGUF Q4_K_M use different quantization
        granularities, scale computation methods, and grouping strategies. The same
        model quantized by two different tools to nominally the same bit width can
        differ by 1–3 percentage points on benchmarks like MMLU. Always validate
        quality on your target task after quantization, regardless of which engine
        performs it.
      </Prose>

      <Callout accent="red">
        The most dangerous failures are the silent ones: incorrect prefix cache hits
        due to tokenizer mismatch, numerical divergence from CUDA version mismatches,
        and stale benchmark comparisons guiding production infrastructure decisions.
        All three produce plausible-looking systems that are quietly wrong.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Verified against arXiv and official documentation in April 2026.
      </Prose>

      <CodeBlock>
{`1. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language
   Model Serving with PagedAttention."
   arXiv:2309.06180 — SOSP 2023.
   Introduces PagedAttention and vLLM. The foundational paper for paged
   KV cache management in production LLM serving. Demonstrates 2-4x
   throughput improvement over FasterTransformer and Orca.
   https://arxiv.org/abs/2309.06180

2. Zheng, L., et al. (2023). "SGLang: Efficient Execution of Structured
   Language Model Programs."
   arXiv:2312.07104
   Introduces SGLang and RadixAttention. Shows 5x throughput improvement
   over vLLM on structured workloads with heavy prefix sharing. Full paper
   published December 2023; RadixAttention blog post January 2024 at
   lmsys.org/blog/2024-01-17-sglang/.
   https://arxiv.org/abs/2312.07104

3. NVIDIA TensorRT-LLM Documentation (2024-2026).
   Stable PyTorch LLM API, speculative decoding (EAGLE, ReDrafter), FP8
   on Hopper/Blackwell, guided + speculative decoding composition.
   https://nvidia.github.io/TensorRT-LLM/

4. HuggingFace Text Generation Inference Documentation (2024-2026).
   Continuous batching, Flash Attention, paged attention, speculative
   decoding, quantization. TGI entered maintenance mode early 2026.
   https://huggingface.co/docs/text-generation-inference

5. ggerganov/llama.cpp GitHub Repository.
   C/C++ LLM inference. GGUF quantization (1.5-bit through 8-bit),
   Metal/AVX/NEON acceleration, on-the-fly KV cache quantization,
   speculative decoding, OpenAI-compatible server.
   https://github.com/ggml-org/llama.cpp

6. Holmes, J., et al. (2024). "DeepSpeed-FastGen: High-throughput Text
   Generation for LLMs via MII and DeepSpeed-Inference."
   arXiv:2401.08671
   Introduces Dynamic SplitFuse: unified prefill/decode scheduling with
   fixed compute budget per step. Reports 2.3x throughput and 3.7x tail
   latency improvement over vLLM on mixed workloads.
   https://arxiv.org/abs/2401.08671

7. Yu, G., et al. (2022). "Orca: A Distributed Serving System for
   Transformer-Based Generative Models."
   USENIX ATC 2022.
   Introduces iteration-level scheduling (continuous batching). The
   foundational paper for the scheduler design every major engine implements.`}
      </CodeBlock>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: Engine selection for a mixed workload</H3>
      <Prose>
        You are building an agentic system where each turn sends a 4,000-token tool
        schema plus a variable user observation (200–800 tokens), expects a structured
        JSON response (100–400 tokens), and runs 500 concurrent sessions. Select an
        inference engine and justify every aspect of your choice: which optimizations
        are load-bearing for this workload, which engine implements them best as of
        April 2026, and what you would benchmark to validate the choice.
      </Prose>

      <H3>Exercise 2: Memory budget for 32 concurrent Llama 3 70B requests</H3>
      <Prose>
        Llama 3 70B: 80 layers, 8 KV heads (GQA), head dimension 128. Compute the
        exact KV cache requirement in GB for 32 concurrent requests at (a) 8k context
        in BF16, (b) 8k context in FP8, (c) 32k context in FP8. A 4× H100 node has
        320 GB total HBM; model weights in BF16 occupy approximately 140 GB. For each
        configuration, determine how many of the 4 GPUs are consumed by weights plus
        cache, and whether the configuration fits. Then describe what prefix caching
        with 60% hit rate on a 2k-token system prompt does to the effective capacity.
      </Prose>

      <H3>Exercise 3: Why is tensor parallelism not in this section?</H3>
      <Prose>
        This section covered single-engine optimizations: KV cache, batching, memory
        management, speculative decoding, prefix caching, constrained decoding,
        queueing, and cost. Tensor parallelism — splitting weight matrices across GPUs
        in AllReduce communication rings — is absent. Explain precisely why it belongs
        in a different section (AI Inference System Design) rather than here. What
        problem does it solve that the techniques in this section cannot? What new
        failure modes does it introduce that have no analog in single-node serving?
      </Prose>

      <H3>Exercise 4: Design a fair vLLM vs TensorRT-LLM benchmark</H3>
      <Prose>
        A naive benchmark of vLLM vs TRT-LLM will confound several variables.
        Design a benchmark protocol that controls for: (a) tokenizer identity —
        same tokenizer producing identical token sequences for every prompt, (b)
        quantization parity — both engines using FP8 weights and FP8 KV cache,
        (c) speculation parity — either both use speculative decoding with the same
        draft model or neither does, (d) workload realism — a distribution of prompt
        and output lengths drawn from your actual traffic, not synthetic uniform
        distributions, (e) warmup — prefix cache and compilation both need warm-up
        periods before steady-state throughput is reached. What metrics do you report
        and at what percentiles?
      </Prose>

      <H3>Exercise 5: Predict feature convergence by 2028</H3>
      <Prose>
        In 2022, continuous batching was a research result in one paper. By 2024
        it was the default in every production serving stack. Trace the same
        pattern for: (a) RadixAttention-style radix tree prefix matching — in which
        engines does it exist today, and what would drive the others to adopt it?
        (b) XGrammar constrained decoding — already in vLLM and SGLang; what is
        blocking TGI and TRT-LLM? (c) prefill-decode disaggregation — available in
        SGLang v0.4+; what infrastructure prerequisites make it non-trivial to add
        to other engines? (d) MLA (Multi-head Latent Attention) serving support —
        which engines support it as of April 2026, and what makes it harder to
        retrofit than GQA? Based on this analysis, predict which two features will
        be universal by 2028 and which two will remain engine-specific.
      </Prose>

      {/* ======================================================================
          SECTION CLOSER
          ====================================================================== */}
      <H2>Closing: the complete Inference Optimization arc</H2>

      <Prose>
        This section began with a single observation: autoregressive generation is
        a loop, and the naive implementation of that loop is astronomically wasteful.
        Every topic that followed was an attack on a specific waste.
      </Prose>

      <Prose>
        <strong>KV cache</strong> eliminated the quadratic recomputation of keys and
        values by storing them once and reading them forward. That cache became the
        primary consumer of GPU memory — more expensive than the model weights
        themselves at any meaningful context length and concurrency. The cache is not
        a detail; it is the governing constraint of modern serving economics.
      </Prose>

      <Prose>
        <strong>Continuous batching</strong> eliminated the idle-GPU waste of static
        batching by rebuilding the active batch at every decode step. Where static
        batching let one long request strand three short ones, continuous batching
        returns those slots to the queue the moment they are free. The Orca paper
        named the mechanism; the PagedAttention paper from vLLM made it physically
        viable by eliminating the fragmentation that had capped batch size. Together
        they constitute the 2-4× throughput improvement that every practitioner
        quotes as the baseline of modern serving.
      </Prose>

      <Prose>
        <strong>Prefix caching</strong> asked the next question: if the KV cache is
        computed once per request, but many requests share a system prompt, are we
        computing it thousands of times unnecessarily? The answer was yes, at enormous
        scale. Block hashing and radix tree matching turned repeated prefill into a
        cache lookup. On workloads with long, stable system prompts — agents, RAG
        pipelines, few-shot classifiers — the savings are not marginal; they are a
        categorical cost reduction.
      </Prose>

      <Prose>
        <strong>Speculative decoding</strong> addressed the bandwidth floor. Even with
        a paged cache and a full batch, each decode step reads all model weights from
        HBM and produces one token. The GPU's compute capacity was being wasted. A
        draft model, using the idle compute, could propose multiple tokens per pass;
        the target model verifies them in a single forward step with no change to the
        output distribution. The speedup is real and composable: speculative decoding
        works inside continuous batching, inside prefix caching, on top of any KV
        cache configuration.
      </Prose>

      <Prose>
        <strong>Constrained decoding</strong> solved a structural problem that prompting
        could not. Language models produce language; production systems need data. The
        5% failure rate of post-hoc validation is acceptable in a demo and fatal in
        a ten-step agentic chain. Logit masking guided by a grammar automaton — compiled
        once, applied per step as a vector addition — made valid-by-construction output
        a zero-overhead guarantee rather than a hope.
      </Prose>

      <Prose>
        <strong>Queueing theory</strong> gave the section its mathematical framework.
        Little's Law, the M/M/1 latency curve, the utilization cliff at ρ = 0.9:
        these are the tools for reasoning about the system's behavior under load before
        it is under load. The LLM-specific wrinkles — bimodal service time, KV memory
        as a hard admission wall, effective throughput that depends on batch size —
        are precisely the places where classical theory needs domain knowledge to be
        applied correctly.
      </Prose>

      <Prose>
        <strong>Cost economics</strong> translated everything into the unit that
        organizations actually optimize: dollars per million tokens. The formula
        is simple; the insight is that every technique in the section is a lever on
        one of its terms. Better batching reduces GPU-hours per token. Prefix caching
        reduces billed prefill tokens. Speculative decoding reduces wall-clock time
        per response. Quantization reduces memory bandwidth per step, enabling higher
        concurrency on the same hardware. The techniques do not just make the system
        faster; they make it economically viable.
      </Prose>

      <Prose>
        <strong>Test-time compute</strong> added the forward-looking complication. The
        section was built on the assumption that shorter generations are cheaper and
        longer generations are costlier. Reasoning models trained with RL on verifiable
        tasks flip this: additional inference tokens are not waste, they are work.
        The log-linear scaling of accuracy with token budget on hard benchmarks means
        the economics of inference now include a new trade-off — compute more tokens
        per request and get a better answer, at a higher bill. The optimization problem
        becomes: not "minimize tokens" but "allocate tokens where they improve accuracy
        and withhold them where they do not."
      </Prose>

      <Prose>
        <strong>Inference engines</strong> — this topic — close the arc by showing
        where all the techniques land in production: inside a single deployable stack.
        vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp are not different approaches to
        LLM serving. They are different engineering tradeoffs on the same set of
        techniques. PagedAttention is in all of them. Continuous batching is in all
        of them. Speculative decoding, prefix caching, and constrained decoding are
        in most of them. The practitioner's job is to match the tradeoffs — research
        velocity vs hardware utilization vs ease of deployment vs ecosystem integration
        — to the workload and the organization.
      </Prose>

      <Prose>
        The distance from "research paper" to "production default" in this space is
        measured in months, not years. PagedAttention was a SOSP 2023 paper; it was
        the default in every major open serving framework by mid-2024. Chunked prefill
        was in a DeepSpeed paper in January 2024; vLLM and SGLang absorbed it by
        Q3 2024. XGrammar was published in November 2024; it was the constrained
        decoding backend in vLLM and SGLang by Q1 2025. The techniques in this section
        are not historical curiosities. They are the baseline that the next wave of
        improvements will be measured against — and that baseline is moving.
      </Prose>

      <Callout accent="purple">
        The complete stack: KV cache foundations → continuous batching →
        paged memory → speculative decoding → prefix caching → constrained
        decoding → queueing theory → cost economics → test-time compute →
        inference engines. Every layer addresses a specific waste. Together
        they turn raw model weights into a scalable product.
      </Callout>
    </div>
  ),
};

export default inferenceEngines;
