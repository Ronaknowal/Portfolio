import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const jambaContent = {
  title: "Hybrid SSM-Transformer Architectures (Jamba)",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By the end of 2023 the field had two strong sequence-mixing primitives and a frustrating gap between them. Transformers, with full softmax attention, were the quality leader on every general-purpose language benchmark — MMLU, HellaSwag, GSM8K, BBH — and the entire ecosystem of pretraining recipes, fine-tuning libraries, and inference servers had been built around them. Their cost, however, scaled as <Code>{"O(L^2 \\cdot d)"}</Code> per layer for sequence length <Code>L</Code>, and the KV cache grew as <Code>{"2 \\cdot L \\cdot d \\cdot n_\\text{layers}"}</Code> bytes. At a 32K-token context the KV cache for a 7B Llama-style model alone exceeded 16 GB; at 256K it broke 100 GB. Long context was bottlenecked not by compute but by memory.
      </Prose>

      <Prose>
        Mamba (Gu and Dao, December 2023, arXiv:2312.00752) flipped that picture. A selective state space model with a hardware-aware parallel scan ran in <Code>{"O(L \\cdot d \\cdot N)"}</Code> training time and <Code>{"O(d \\cdot N)"}</Code> per-token inference time, with the KV cache replaced by a fixed-size state of shape <Code>{"[d, N]"}</Code> per layer. At the 7B parameter scale, Mamba matched Llama-2-7B on perplexity and most reasoning benchmarks at far lower inference cost. It was the first SSM that could plausibly serve as a drop-in transformer replacement for general language modeling.
      </Prose>

      <Prose>
        But "matched on perplexity" hid a real problem. As soon as researchers stress-tested Mamba on tasks that demanded <em>exact</em> recall — copying a specific earlier token, in-context learning of arbitrary key-value pairs, multi-needle-in-haystack retrieval — pure Mamba lagged a matched-parameter transformer by 2-5 percentage points. The benchmarks that exposed this gap most cleanly were Phonebook (Jelassi et al. 2024) and the Longbench retrieval suites; on real downstream metrics, MMLU and HumanEval also showed small but consistent deficits. The intuition: a fixed-size <Code>{"[d, N]"}</Code> state is a compressed summary of the past, and when the task requires bit-for-bit retrieval of an arbitrary earlier token, the compression hurts. Attention has the opposite trade-off — its KV cache is a lossless record of every past token, expensive but exact.
      </Prose>

      <Prose>
        The natural question: do you need <em>every</em> layer to have the lossless attention property, or do most layers just need a cheap, expressive sequence-mixer with a few attention layers sprinkled in for the moments when exact lookup matters? In March 2024, AI21 Labs answered "no" with Lieber et al. "Jamba: A Hybrid Transformer-Mamba Language Model" (arXiv:2403.19887). Jamba is a 52B-total / 12B-active parameter model with a 256K-token context window. Its core architecture interleaves Mamba and attention layers in a 1:7 ratio (one attention layer for every seven Mamba layers), with Mixture-of-Experts (MoE) layers added on top of every other block. The result: Llama-2-quality at long context with a fraction of the KV-cache memory, and inference throughput that runs 30% faster than a matched-quality dense transformer.
      </Prose>

      <Prose>
        Jamba was not the only hybrid effort. The same year saw Glorioso et al. "Zamba: A Compact 7B SSM Hybrid Model" (arXiv:2405.16712) from Zyphra, which uses a different layout — a single shared attention block applied to every-Nth token between blocks of Mamba layers — and demonstrates that 7B-class hybrids can hit the dense-transformer Pareto frontier with substantially less training compute. De et al. "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models" (arXiv:2402.19427) from DeepMind, published a month before Jamba, showed that a recurrent-gated-linear (RG-LRU) mixer interleaved with local-window attention beats Llama-2 at 7B with 30% less training data. NVIDIA followed with Zhang et al. "Hymba: A Hybrid-head Architecture for Small Language Models" (arXiv:2411.13676), which fuses SSM and attention <em>within</em> a single layer rather than alternating between layers — more aggressive integration, optimized for sub-2B sizes.
      </Prose>

      <Prose>
        The hybrid design choice is now well established. Production models from AI21 (Jamba 1.5 Mini and Large), Zyphra (Zamba2-7B), IBM (Granite-MoE family), and the open Mistral/Codestral derivatives all use some form of attention-SSM interleaving. The 1:7 ratio Jamba pioneered is not magic — Zamba uses 1:6 with a shared attention block, Hymba uses 1:1 within-layer fusion, Samba (Ren et al. 2025) uses 1:1 across layers — but the principle is universal. <strong>You buy efficiency from the SSM and you buy quality from the attention, and the total is more than either alone.</strong> The hybrid is, as of 2026, the most credible architectural challenger to pure Transformer dominance.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Think of sequence modeling as two distinct problems happening simultaneously: <em>mixing</em> — propagating information between positions — and <em>recall</em> — retrieving an exact earlier token when needed. A pure Mamba excels at the first. Its selective scan moves information across the sequence in linear time, and the SSM state is rich enough to capture local context, syntactic structure, and approximate semantic continuity. Where Mamba struggles is the second problem: when the task says "what was the third token after the SEP marker," Mamba has to have stored that token in its compressed state, and it does so imperfectly. The state is a lossy summary, and lossy summaries fail at exact lookups.
      </Prose>

      <Prose>
        Attention has the opposite profile. Its mixing is excellent because every token can attend to every other, but its cost grows quadratically with sequence length. For 90% of the sequence-mixing operations a language model performs — refining a representation based on local syntax, propagating a phrase-level meaning forward, applying a positional bias — attention is overkill. You do not need lossless lookup of every prior token to predict the next syntactic head; a compressed running summary works fine.
      </Prose>

      <Prose>
        The hybrid insight is to allocate the expensive lossless mechanism only where it is genuinely needed. If most of your layers can be Mamba (cheap, linear, good at mixing) and a few of your layers can be attention (expensive, quadratic, good at exact recall), you get the best of both. This is exactly analogous to a memory hierarchy in computer systems: most accesses go to fast, lossy cache; the few that miss fall through to slow, lossless DRAM. Jamba's 1:7 ratio — one attention every eight blocks — is the architectural equivalent of that hierarchy, and it works for the same reason: most operations do not need the full mechanism, so paying for it everywhere is wasted.
      </Prose>

      <Prose>
        The other intuition that motivates hybrids is the <em>induction head</em> result from mechanistic interpretability (Olsson et al. 2022). Transformers, when trained on language, develop circuits in the second-to-last attention layer that explicitly perform "find the previous occurrence of the current token, then output the token that followed it." This pattern — induction heads — is the substrate for in-context learning. Pure Mambas struggle to form clean induction heads because the SSM's content-mixing is more diffuse than attention's sharp QK lookup. By inserting even one attention layer in a stack of Mambas, you give the network a place to do the QK-style retrieval that induction-head circuits depend on. Empirically, a single attention layer at the right depth recovers most of the transformer's in-context learning capability.
      </Prose>

      <Prose>
        Add MoE on top and the picture sharpens further. MoE layers (Shazeer et al. 2017, Fedus et al. 2022) add capacity without adding per-token compute: a router selects 1 or 2 of <Code>K</Code> experts per token, and only those experts run. Jamba uses MoE at every other block (16 of 32 layers in Jamba-1.5-Large), giving it 52B total parameters but only 12B active per token. The combination — Mamba for cheap mixing, sparse attention for exact recall, MoE for capacity — gives Jamba its quality-cost profile: dense-model quality at long context with sparse-model inference economics.
      </Prose>

      <Callout accent="gold">
        Mental model: Mamba is the workhorse mixer (cheap, linear, lossy memory), attention is the precision tool (expensive, quadratic, lossless lookup), and MoE is the capacity multiplier (sparse, scales parameters without scaling per-token compute). Stack them in a 7:1:every-other ratio and you have Jamba. Change the ratios — 6:1 with shared attn for Zamba, within-layer fusion for Hymba — and you have the rest of the hybrid family.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The hybrid block</H3>

      <Prose>
        A standard transformer block is: <Code>{"x \\leftarrow x + \\text{Attn}(\\text{LN}(x))"}</Code> followed by <Code>{"x \\leftarrow x + \\text{FFN}(\\text{LN}(x))"}</Code>. A pure-Mamba block swaps the attention for a Mamba mixer: <Code>{"x \\leftarrow x + \\text{Mamba}(\\text{LN}(x))"}</Code>, then the same FFN. A hybrid model just decides, layer by layer, which mixer goes in the first position. Jamba's pattern in pseudocode:
      </Prose>

      <MathBlock>
        {"\\text{Block}_i(x) = \\begin{cases} x + \\text{FFN}(\\text{LN}(x + \\text{Attn}(\\text{LN}(x)))) & \\text{if } i \\bmod 8 = 7 \\\\ x + \\text{FFN}(\\text{LN}(x + \\text{Mamba}(\\text{LN}(x)))) & \\text{otherwise} \\end{cases}"}
      </MathBlock>

      <Prose>
        For Jamba-1.5-Large with 32 total layers, this gives 28 Mamba blocks and 4 attention blocks. The placement is uniform — every 8th layer is attention, starting from layer 7 (0-indexed) — but the choice of placement is itself a hyperparameter. Ablations in the Jamba paper show that placing the attention layers at the end of the model (final 1/8) hurts noticeably, and placing them in the middle is best for short tasks; uniform spacing is the practical default and what Jamba ships.
      </Prose>

      <H3>3.2 KV cache only at attention layers</H3>

      <Prose>
        The single most consequential design decision: <Code>{"\\text{kv\\_cache}_i"}</Code> exists only when block <Code>i</Code> is attention. For a Mamba block, the per-token state is the SSM hidden state of fixed shape <Code>{"[d_\\text{inner}, N]"}</Code> where <Code>{"d_\\text{inner} = \\text{expand} \\cdot d_\\text{model}"}</Code> and <Code>{"N"}</Code> is the SSM state dim (16 in Jamba-1, 128 in Jamba-1.5). For an attention block, the per-token KV cache grows by <Code>{"2 \\cdot n_\\text{kv-heads} \\cdot d_\\text{head}"}</Code> entries. Total inference-time memory:
      </Prose>

      <MathBlock>
        {"M_\\text{cache}(L) = n_\\text{attn} \\cdot 2 \\cdot n_\\text{kv-heads} \\cdot d_\\text{head} \\cdot L \\cdot 2_\\text{bytes} + n_\\text{ssm} \\cdot d_\\text{inner} \\cdot N \\cdot 4_\\text{bytes}"}
      </MathBlock>

      <Prose>
        The first term grows linearly in <Code>L</Code> but with the prefactor <Code>{"n_\\text{attn}"}</Code> (4 in Jamba-1.5-Large) instead of <Code>{"n_\\text{layers}"}</Code> (32). The second term is constant in <Code>L</Code>. For <Code>{"L = 256K"}</Code>, the KV cache portion of a Jamba is roughly <Code>{"4 / 32 = 1/8"}</Code> the size of a comparable pure-transformer's KV cache; the SSM state portion is fixed at a few MB. The whole model state at 256K context fits comfortably in single-H100 80 GB memory; an equivalent dense transformer at 256K would not.
      </Prose>

      <H3>3.3 Compute cost per token</H3>

      <Prose>
        Per-layer FLOPs for a forward pass on a sequence of length <Code>L</Code>, model dim <Code>d</Code>, attention heads <Code>{"n_h"}</Code>, head dim <Code>{"d_h"}</Code>, SSM state dim <Code>N</Code>:
      </Prose>

      <MathBlock>
        {"\\text{FLOPs}_\\text{attn-layer} = 4 L d^2 + 2 L^2 d, \\qquad \\text{FLOPs}_\\text{mamba-layer} \\approx 6 L d \\cdot \\text{expand} \\cdot N + 6 L d^2"}
      </MathBlock>

      <Prose>
        At <Code>{"L = 8192"}</Code>, <Code>{"d = 4096"}</Code>, <Code>{"N = 16"}</Code>, expand=2: attention is dominated by its <Code>{"L^2 d"}</Code> term, giving about 0.55 TFLOPs per layer; Mamba is dominated by its <Code>{"L d^2"}</Code> term, giving about 0.4 TFLOPs per layer. Mamba is cheaper, but only by 30% at this length. At <Code>{"L = 65536"}</Code>: attention is 35 TFLOPs per layer, Mamba is 3.2 TFLOPs per layer — Mamba is 11x cheaper. At very long context the linear-vs-quadratic asymptote dominates and the savings explode.
      </Prose>

      <Prose>
        For a Jamba with the 1:7 ratio: total per-token compute is <Code>{"n_\\text{attn} \\cdot \\text{cost}_\\text{attn} + n_\\text{ssm} \\cdot \\text{cost}_\\text{ssm}"}</Code>. With 4 attention and 28 SSM layers at <Code>{"L = 65536"}</Code>: <Code>{"4 \\cdot 35 + 28 \\cdot 3.2 = 230"}</Code> TFLOPs per token. A pure dense transformer of equivalent depth would cost <Code>{"32 \\cdot 35 = 1120"}</Code> TFLOPs per token — roughly 5x more. The hybrid achieves most of attention's quality with 20% of attention's compute at long context.
      </Prose>

      <H3>3.4 MoE accounting</H3>

      <Prose>
        Jamba-1.5-Large adds Mixture-of-Experts to the FFN of every other block, so 16 of 32 blocks have MoE FFNs. Each MoE FFN has <Code>{"K = 16"}</Code> experts, of which <Code>{"k = 2"}</Code> are activated per token by a top-<Code>k</Code> router. Total parameters in an MoE layer: <Code>{"K \\cdot 2 \\cdot d \\cdot d_\\text{ffn}"}</Code>, but compute is only <Code>{"k \\cdot 2 \\cdot d \\cdot d_\\text{ffn}"}</Code>. For <Code>{"d = 4096"}</Code>, <Code>{"d_\\text{ffn} = 14336"}</Code>, <Code>{"K = 16"}</Code>, <Code>{"k = 2"}</Code>: 1.9 billion parameters per MoE FFN, but only 235 million active per token. Stacked across 16 MoE layers the model totals 52B params with 12B active — a 4.3x parameter inflation that costs no extra inference compute and provides the capacity budget that lets Jamba hit Llama-2-70B-class quality at decode speeds closer to a 12B model.
      </Prose>

      <H3>3.5 Discretization-free formulation in Jamba's Mamba blocks</H3>

      <Prose>
        Jamba's Mamba blocks use the standard S6 (Mamba-1) selective SSM. The per-token state update is:
      </Prose>

      <MathBlock>
        {"x_t = \\bar A_t \\odot x_{t-1} + \\bar B_t u_t, \\qquad y_t = C_t^T x_t + D \\odot u_t"}
      </MathBlock>

      <Prose>
        with <Code>{"\\bar A_t = \\exp(\\Delta_t \\odot A)"}</Code>, <Code>{"\\bar B_t = \\Delta_t \\odot B_t"}</Code>, and <Code>{"\\Delta_t, B_t, C_t"}</Code> linear projections of the input <Code>{"u_t"}</Code> (selectivity). The state <Code>{"x_t"}</Code> has shape <Code>{"[d_\\text{inner}, N]"}</Code>. Compared to a vanilla Mamba paper implementation, Jamba's only non-standard tweak is RMSNorm on the residual path before each Mamba block (RMSNorm rather than LayerNorm) — a stability improvement noted in their ablations.
      </Prose>

      <H3>3.6 Attention block: standard but with GQA and RoPE</H3>

      <Prose>
        Jamba's attention layers are conventional: Grouped Query Attention with <Code>{"n_\\text{kv-heads} = 8"}</Code> sharing across <Code>{"n_\\text{heads} = 64"}</Code> query heads (a <Code>{"8 \\times"}</Code> reduction in KV-cache size), Rotary Position Embedding (RoPE) with theta = 10000 for the 4K-trained Jamba-1 and theta scaled up for the 256K Jamba-1.5. There is no positional encoding in the Mamba layers — Mamba's recurrence is intrinsically positional — so RoPE is applied only inside the attention blocks. This asymmetry is fine in practice: positional information injected at every 8th layer is sufficient.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4.1 The hybrid block in PyTorch</H3>

      <Prose>
        We build a minimal Jamba-style hybrid stack: a simplified selective SSM (Mamba-lite, serial scan for clarity), standard causal multi-head attention, and a small MLP. The block alternates between SSM and attention according to a configurable <Code>attn_every</Code> stride. Three matched-depth models — pure SSM, pure attention, and hybrid 1:7 — are then trained on a copy task that requires propagating an arbitrary token across the sequence. This is the canonical demonstration of what attention does and Mamba alone cannot.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(0)

D = 48          # model dim (small for fast CPU run)
N = 8           # SSM state dim
NHEADS = 4
LSEQ = 32
VOCAB = 12

class SimpleSSM(nn.Module):
    """Selective SSM (Mamba-style), serial scan for pedagogy."""
    def __init__(self, d_model, d_state=N):
        super().__init__()
        self.d_model = d_model; self.d_state = d_state
        self.x_proj = nn.Linear(d_model, 2 * d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, u):
        B, L, D = u.shape; N = self.d_state
        x_dbl = self.x_proj(u)
        dt_raw = x_dbl[..., :1]
        Bt = x_dbl[..., 1:1+N]; Ct = x_dbl[..., 1+N:1+2*N]
        delta = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)
        Abar = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, 1, N))
        Bbar = delta.unsqueeze(-1) * Bt.unsqueeze(2)
        h = torch.zeros(B, D, N, device=u.device)
        ys = []
        for t in range(L):
            h = Abar[:, t] * h + Bbar[:, t] * u[:, t].unsqueeze(-1)
            y = (h * Ct[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1) + self.D.view(1, 1, -1) * u
        return self.out_proj(y)


class CausalAttention(nn.Module):
    def __init__(self, d_model, nheads=NHEADS):
        super().__init__()
        self.nheads = nheads; self.d_head = d_model // nheads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nheads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.d_head ** 0.5)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class MLP(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expand * d_model)
        self.fc2 = nn.Linear(expand * d_model, d_model)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """One Jamba-style block: pre-norm + (SSM | Attention) + pre-norm + MLP."""
    def __init__(self, d_model, kind="ssm"):
        super().__init__()
        self.kind = kind
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.mixer = CausalAttention(d_model) if kind == "attn" else SimpleSSM(d_model)
        self.mlp = MLP(d_model)
    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LM(nn.Module):
    """Tiny LM with attn_every: 999 = pure SSM, 1 = pure attn, 8 = Jamba-like."""
    def __init__(self, vocab, n_blocks, attn_every, d_model=D):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, "attn" if (i % attn_every) == (attn_every - 1) else "ssm")
            for i in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))


# Copy task: BOS, prefix(14 tokens), SEP, prefix(14 tokens), score only the recall half.
BOS, SEP, EOS = 0, 1, 2
PREFIX_LEN = 14

def sample_batch(B):
    prefix = torch.randint(3, VOCAB, (B, PREFIX_LEN))
    bos = torch.full((B, 1), BOS, dtype=torch.long)
    sep = torch.full((B, 1), SEP, dtype=torch.long)
    seq = torch.cat([bos, prefix, sep, prefix], dim=1)
    x = seq[:, :-1]
    y = seq[:, 1:].clone()
    y[:, :PREFIX_LEN + 1] = -100   # only score the recall half
    return x, y


def train_model(model, label, steps=400, B=64, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for s in range(steps):
        x, y = sample_batch(B)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 80 == 0 or s == steps - 1:
            preds = logits.argmax(dim=-1)
            mask = (y != -100)
            acc = ((preds == y) & mask).float().sum() / mask.float().sum()
            print(f"  [{label}] step {s:>3} loss={loss.item():.3f} recall_acc={acc.item():.3f}")


# Three matched-depth models
torch.manual_seed(0); pure_ssm  = LM(VOCAB, n_blocks=8, attn_every=999)
torch.manual_seed(0); pure_attn = LM(VOCAB, n_blocks=8, attn_every=1)
torch.manual_seed(0); hybrid    = LM(VOCAB, n_blocks=8, attn_every=8)

print(f"Copy task: prefix len {PREFIX_LEN}, vocab {VOCAB}")
print(f"Pure-SSM  blocks: {[b.kind for b in pure_ssm.blocks]}")
print(f"Pure-Attn blocks: {[b.kind for b in pure_attn.blocks]}")
print(f"Hybrid    blocks: {[b.kind for b in hybrid.blocks]}")
print()

print("Training Pure-SSM:");                 train_model(pure_ssm, "ssm")
print("Training Pure-Attn:");                train_model(pure_attn, "att")
print("Training Hybrid (1:7 attn:ssm):");    train_model(hybrid, "hyb")

# Held-out evaluation
print()
print("FINAL recall accuracy on held-out copies:")
for label, m in [("Pure-SSM", pure_ssm), ("Pure-Attn", pure_attn), ("Hybrid", hybrid)]:
    accs = []
    for _ in range(20):
        x, y = sample_batch(64)
        with torch.no_grad():
            logits = m(x)
        preds = logits.argmax(-1)
        mask = (y != -100)
        accs.append((((preds == y) & mask).float().sum() / mask.float().sum()).item())
    print(f"  {label:>10s}: {sum(accs)/len(accs):.3f}")

# Forward latency
print()
print("Forward latency (CPU, seq=32, batch=8, mean of 5 runs):")
xb = torch.randint(0, VOCAB, (8, 32))
for label, m in [("Pure-SSM", pure_ssm), ("Pure-Attn", pure_attn), ("Hybrid", hybrid)]:
    m.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = m(xb)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            _ = m(xb)
    dt = (time.perf_counter() - t0) / 5 * 1000
    print(f"  {label:>10s}: {dt:6.1f} ms")

# Output:
# Copy task: prefix len 14, vocab 12
# Pure-SSM  blocks: ['ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm']
# Pure-Attn blocks: ['attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn']
# Hybrid    blocks: ['ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'ssm', 'attn']
#
# Training Pure-SSM:
#   [ssm] step   0 loss=26.558 recall_acc=0.102
#   [ssm] step  80 loss=2.245 recall_acc=0.103
#   [ssm] step 160 loss=2.218 recall_acc=0.125
#   [ssm] step 240 loss=2.224 recall_acc=0.106
#   [ssm] step 320 loss=2.240 recall_acc=0.090
#   [ssm] step 399 loss=2.219 recall_acc=0.118
# Training Pure-Attn:
#   [att] step   0 loss=36.156 recall_acc=0.113
#   [att] step  80 loss=2.200 recall_acc=0.123
#   [att] step 160 loss=2.081 recall_acc=0.202
#   [att] step 240 loss=1.856 recall_acc=0.279
#   [att] step 320 loss=1.639 recall_acc=0.350
#   [att] step 399 loss=1.331 recall_acc=0.497
# Training Hybrid (1:7 attn:ssm):
#   [hyb] step   0 loss=26.424 recall_acc=0.109
#   [hyb] step  80 loss=2.139 recall_acc=0.195
#   [hyb] step 160 loss=2.062 recall_acc=0.205
#   [hyb] step 240 loss=1.875 recall_acc=0.272
#   [hyb] step 320 loss=0.380 recall_acc=0.860
#   [hyb] step 399 loss=0.045 recall_acc=0.989
#
# FINAL recall accuracy on held-out copies:
#     Pure-SSM: 0.109
#    Pure-Attn: 0.492
#       Hybrid: 0.982
#
# Forward latency (CPU, seq=32, batch=8, mean of 5 runs):
#     Pure-SSM:   50.4 ms
#    Pure-Attn:    9.1 ms
#       Hybrid:   45.9 ms`}
      </CodeBlock>

      <Prose>
        The result is striking. <strong>Pure-SSM gets 11% recall on the copy task</strong> — essentially chance for a 9-symbol vocabulary (random guessing among 9 non-special tokens is 11.1%). Its compressed state cannot store and re-emit 14 arbitrary tokens. <strong>Pure-Attn reaches 49% in the same training budget</strong>, demonstrating that lossless KV cache enables exact recall but needs many gradient steps to learn the induction-head circuit. <strong>The hybrid (1:7) reaches 98%</strong> — better than pure-attn at the same step budget, because the SSM layers do most of the work and the single attention layer at the end provides the precise lookup. This is the textbook hybrid-architecture story in microcosm.
      </Prose>

      <Prose>
        Forward-pass latency is a different story. At this tiny scale on CPU, pure-attention is fastest (9 ms) because attention's matmul dominates and PyTorch's BMM is much faster than our naive Python scan in <Code>SimpleSSM</Code>. The hybrid is 46 ms, almost as slow as pure-SSM (50 ms), because 7 of its 8 blocks are still serial-Python SSMs. <strong>The latency advantage of hybrids only materializes at long context with a real CUDA scan kernel.</strong> The from-scratch numbers here teach the architectural point — recall capability — not the throughput point.
      </Prose>

      <H3>4.2 KV cache vs SSM state memory accounting</H3>

      <Prose>
        The other from-scratch exercise: compute the inference-time memory of a Jamba-1.5-Large-shaped hybrid versus an equivalent dense transformer at multiple context lengths. This is the calculation that justifies the architecture economically.
      </Prose>

      <CodeBlock language="python">
{`# Compute KV-cache vs SSM-state memory at multiple context lengths.
# KV cache lives ONLY at attention layers; SSM state at every Mamba layer.

n_layers = 32
attn_every = 8
n_attn = n_layers // attn_every     # 4
n_ssm = n_layers - n_attn           # 28
d_model = 4096
n_kv_heads = 8                      # GQA
d_head = 128
d_state = 16
expand = 2
bytes_fp16 = 2
bytes_fp32 = 4

ssm_state_per_layer_bytes = expand * d_model * d_state * bytes_fp32

print(f"Hybrid model: {n_layers} layers, {n_attn} attn + {n_ssm} ssm, GQA n_kv_heads={n_kv_heads}")
print(f"  d_model={d_model}, d_state={d_state}, expand={expand}")
print()
print(f"Per-layer state sizes:")
print(f"  KV cache (per layer, per token): {2 * n_kv_heads * d_head * bytes_fp16} bytes")
print(f"  SSM state (per layer, fixed):    {ssm_state_per_layer_bytes:,} bytes")
print()
print(f"{'context':>10}  {'PureXfmr-KV':>14}  {'Hybrid-KV':>14}  {'Hybrid-SSM':>14}  {'Hybrid-total':>14}")
for L in [1024, 4096, 16384, 65536, 262144]:
    pure_kv = n_layers * 2 * n_kv_heads * d_head * L * bytes_fp16
    hyb_kv  = n_attn   * 2 * n_kv_heads * d_head * L * bytes_fp16
    hyb_state = n_ssm * ssm_state_per_layer_bytes
    hyb_total = hyb_kv + hyb_state
    def gb(b): return f"{b/1e9:>10.2f} GB"
    print(f"{L:>10d}  {gb(pure_kv)}  {gb(hyb_kv)}  {gb(hyb_state)}  {gb(hyb_total)}")

# Output:
# Hybrid model: 32 layers, 4 attn + 28 ssm, GQA n_kv_heads=8
#   d_model=4096, d_state=16, expand=2
#
# Per-layer state sizes:
#   KV cache (per layer, per token): 4096 bytes
#   SSM state (per layer, fixed):    524,288 bytes
#
#    context     PureXfmr-KV       Hybrid-KV      Hybrid-SSM    Hybrid-total
#       1024        0.13 GB        0.02 GB        0.01 GB        0.03 GB
#       4096        0.54 GB        0.07 GB        0.01 GB        0.08 GB
#      16384        2.15 GB        0.27 GB        0.01 GB        0.28 GB
#      65536        8.59 GB        1.07 GB        0.01 GB        1.09 GB
#     262144       34.36 GB        4.29 GB        0.01 GB        4.31 GB`}
      </CodeBlock>

      <Prose>
        At 256K context, a pure-transformer with the same depth and head config would need 34 GB just for KV cache (on top of the 100+ GB for activations during prefill). A Jamba-shaped hybrid needs 4.3 GB for everything cache-related — 8x less. This is the difference between fitting on a single 80GB H100 with room for the model weights and activations versus needing model parallelism across multiple GPUs. At 1M context the gap is even more dramatic; pure transformers are simply infeasible without aggressive paging or sharding, while a hybrid grows linearly with a small constant.
      </Prose>

      <H3>4.3 Where the attention layers should go</H3>

      <Prose>
        A natural follow-up question: in a stack of <Code>N</Code> blocks with <Code>k</Code> attention slots, where should the attention slots live? Three strategies are common: (1) at the end (final <Code>k</Code> blocks), (2) at the start, (3) uniformly spaced. Jamba's ablations (Section 5.2 of the paper) indicate uniform spacing wins on most benchmarks, with a slight edge to placing the first attention layer slightly later in the stack. The intuition: early attention is wasted because there is little prior context to attend to; late attention has nothing to feed forward; uniform attention gives the network multiple "sync points" where induction-head circuits can form. The uniform <Code>{"i \\bmod 8 = 7"}</Code> placement is cheap to implement and works well; do not over-optimize this hyperparameter unless ablations demand it.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 Loading Jamba via HuggingFace</H3>

      <Prose>
        AI21 publishes Jamba weights on HuggingFace under the <Code>ai21labs</Code> organization. The original release was <Code>ai21labs/Jamba-v0.1</Code> in March 2024 (52B total / 12B active, 256K context); production successors are <Code>ai21labs/Jamba-1.5-Mini</Code> (52B / 12B, instruction-tuned) and <Code>ai21labs/Jamba-1.5-Large</Code> (398B / 94B, the flagship). Loading uses the standard <Code>transformers</Code> API with a custom model class registered through <Code>trust_remote_code=True</Code> for older versions; HF transformers 4.40+ has Jamba support natively.
      </Prose>

      <CodeBlock language="python">
{`# pip install transformers>=4.40 mamba-ssm causal-conv1d>=1.2 accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ai21labs/Jamba-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",                    # shard across visible GPUs
    attn_implementation="flash_attention_2",  # FA2 for the attention layers
    use_mamba_kernels=True,                # use selective_scan_cuda for Mamba layers
)

# Generation works exactly like any other HF causal LM
prompt = "Explain how Jamba's hybrid architecture saves memory at 256K context:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))`}
      </CodeBlock>

      <Prose>
        Two flags are non-standard: <Code>use_mamba_kernels=True</Code> requires the <Code>mamba-ssm</Code> and <Code>causal-conv1d</Code> packages to be installed and CUDA-compiled; without them the Mamba layers fall back to a pure-PyTorch implementation that is 10-50x slower (the same trap that catches pure-Mamba users). <Code>attn_implementation="flash_attention_2"</Code> applies FlashAttention only to the attention layers, where it matters; the SSM layers are unaffected. Memory at load is roughly 105 GB (bf16) for Jamba-v0.1 at 52B parameters — fits on two H100-80GB or one H200-141GB. The 1.5-Large at 398B requires 8x H100-80GB.
      </Prose>

      <H3>5.2 Long-context inference with vLLM</H3>

      <Prose>
        vLLM (Kwon et al. 2023) added Jamba support in version 0.5.4 (August 2024). The integration is non-trivial because vLLM's PagedAttention assumes uniform attention layers; Jamba's hybrid pattern required a separate state manager for the Mamba layers. The result: Jamba serves at production throughput with vLLM's standard continuous-batching scheduler.
      </Prose>

      <CodeBlock language="python">
{`# pip install vllm>=0.5.4
from vllm import LLM, SamplingParams

llm = LLM(
    model="ai21labs/Jamba-1.5-Mini",
    dtype="bfloat16",
    max_model_len=262144,       # full 256K context
    enforce_eager=False,         # use the optimized Mamba kernels via torch.compile
    gpu_memory_utilization=0.9,
)

sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400)

prompts = [
    "Summarize this 100,000-token document: " + open("long_doc.txt").read(),
    "What was the main argument in the third paragraph of the document above?",
]
outputs = llm.generate(prompts, sampling)
for o in outputs:
    print(o.outputs[0].text)`}
      </CodeBlock>

      <Prose>
        At 256K context with vLLM, Jamba-1.5-Mini decodes at roughly 30-50 tokens/second on a single H100 (compared to single-digit tokens/sec for an equivalent dense transformer at the same context length, if it would even fit). Time-to-first-token (TTFT) for a 100K prompt is about 6-12 seconds; for a comparable Llama-style transformer it is unfeasible without multi-GPU inference. TGI (Text Generation Inference, HuggingFace) added Jamba support in v2.2; the integration is similar to vLLM's.
      </Prose>

      <H3>5.3 Other production hybrids</H3>

      <Prose>
        <strong>Zamba2-7B</strong> from Zyphra (HuggingFace: <Code>Zyphra/Zamba2-7B</Code>) is a 7.4B-parameter Mamba-2-attention hybrid that uses a single shared attention block applied periodically rather than Jamba's full attention layers. Loading is similar to Jamba but uses <Code>trust_remote_code=True</Code> because Zamba's architecture is not yet in stock transformers. Zamba2-7B beats Llama-3-8B on most reasoning benchmarks at smaller training compute.
      </Prose>

      <Prose>
        <strong>IBM Granite-MoE</strong> family (<Code>ibm-granite/granite-3.0-3b-a800m-instruct</Code> and larger) uses an MoE-Mamba-attention hybrid optimized for enterprise serving. Granite ships with strong tool-use and structured-output capabilities — a focus area different from AI21's general-language emphasis.
      </Prose>

      <Prose>
        <strong>Hymba-1.5B-Base</strong> from NVIDIA (<Code>nvidia/Hymba-1.5B-Base</Code>) implements <em>within-layer</em> hybrid heads — every layer has both attention heads and SSM heads operating in parallel — rather than alternating layers. The architecture is more aggressive integration aimed at sub-2B model sizes where layer-level alternation gives too few of each type. Hymba-1.5B competes with Llama-3-3B on benchmarks at half the parameter count.
      </Prose>

      <H3>5.4 Fine-tuning hybrids</H3>

      <Prose>
        Fine-tuning Jamba uses LoRA in the same way as transformers, but with care: LoRA must be applied to the attention layers' QKV/O projections AND to the Mamba layers' input projections (<Code>x_proj</Code>, <Code>dt_proj</Code>) for full coverage. Applying LoRA to attention layers only — a copy-paste from Llama recipes — leaves 7/8 of the model frozen and gives weak fine-tuning results. The Jamba codebase ships a sample LoRA config that targets both layer types; use it as a starting point. Full-finetuning works but requires the Mamba-specific learning-rate schedule (Mamba layers prefer slightly higher LR than attention layers because their gradient magnitudes differ).
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Three visualizations make the hybrid story concrete: a step-by-step trace of one Jamba block-cycle, a quality-vs-compute Pareto plot comparing pure-Mamba, pure-Transformer, and Jamba, and a memory-vs-context-length plot showing the KV-cache savings.
      </Prose>

      <StepTrace
        label="Jamba block cycle: 7 Mamba blocks then 1 Attention block"
        steps={[
          {
            label: "Block 0 (Mamba): refines local representations",
            render: () => (
              <Prose>
                Tokens enter the first Mamba block. The selective scan runs in <Code>{"O(L \\cdot d \\cdot N)"}</Code> time, propagating local context. The state at each position is an <Code>{"[d_\\text{inner}, N]"}</Code> tensor — for Jamba with <Code>{"d_\\text{inner} = 8192"}</Code>, <Code>{"N = 16"}</Code>, that is 128K floats per layer, half a MB at fp32. No KV cache is built. The output is a residual added back to the input; FFN then refines further.
              </Prose>
            ),
          },
          {
            label: "Blocks 1-6 (Mamba): build a multi-scale summary",
            render: () => (
              <Prose>
                Six more Mamba blocks stack. Each one's state has different learned <Code>{"\\Delta_t"}</Code> profiles, so different blocks specialize in different memory horizons — some hold short-range syntax, others integrate paragraph-level structure. Total compute through these 7 blocks is <Code>{"7 \\cdot O(L d N)"}</Code>, linear in <Code>L</Code>. At <Code>{"L = 32K"}</Code>, the full Mamba portion is roughly equivalent in FLOPs to a single transformer attention layer at the same length.
              </Prose>
            ),
          },
          {
            label: "Block 7 (Attention): sharp lookup, induction heads form here",
            render: () => (
              <Prose>
                The single attention block in this cycle. Its query at position <Code>t</Code> attends over all <Code>L</Code> prior keys, computing exact <Code>{"\\text{softmax}(QK^T / \\sqrt{d_h}) V"}</Code> with GQA (8 KV heads, 64 Q heads) and RoPE positional encoding. The KV cache for THIS layer alone fills with <Code>{"L \\cdot 2 \\cdot 8 \\cdot 128 \\cdot 2"}</Code> bytes = <Code>{"4096 L"}</Code> bytes — at <Code>{"L = 32K"}</Code> that is 128 MB per layer, but only this layer pays it. Induction-head circuits live here.
              </Prose>
            ),
          },
          {
            label: "End of cycle: residual stream carries forward",
            render: () => (
              <Prose>
                After block 7, the residual stream contains contributions from 7 Mamba mixers (compressed lossy summaries) and 1 attention mixer (lossless lookup). The next 7 blocks will be Mamba again, so the attention's contribution is what those Mambas have to keep around in their states. Empirically this pattern — periodic sync via attention, continuous mixing via Mamba — is what gives Jamba its quality. Repeat 4 times for the full Jamba-1.5-Large 32-layer stack.
              </Prose>
            ),
          },
          {
            label: "MoE FFN at every other block (not shown above)",
            render: () => (
              <Prose>
                Half the FFN positions in Jamba-1.5-Large are MoE layers, each with 16 experts of which a top-2 router activates 2 per token. The MoE adds 1.7B parameters per layer (vs 0.2B for a dense FFN at the same hidden dim) but only 0.2B of those are computed per token. Stacked across 16 MoE layers this is the source of Jamba's 52B-total / 12B-active asymmetry. Routing is independent of mixer type — both Mamba blocks and attention blocks can sit beneath an MoE FFN.
              </Prose>
            ),
          },
        ]}
      />

      <Prose>
        The next plot summarizes the published quality-vs-compute landscape for hybrid versus pure architectures, with numbers approximated from the Jamba and Zamba papers and adjusted for typical 7B-class evaluations as of 2026.
      </Prose>

      <Plot
        label="quality (avg of MMLU/HellaSwag/ARC) vs total inference compute, 7B-class models"
        series={[
          {
            name: "Pure Transformer",
            color: "#60a5fa",
            points: [[1.0, 60.0], [2.0, 64.5], [4.0, 68.0], [8.0, 70.5], [16.0, 72.0]],
          },
          {
            name: "Pure Mamba",
            color: "#c084fc",
            points: [[0.4, 56.0], [0.8, 60.5], [1.6, 64.0], [3.2, 66.5], [6.4, 68.0]],
          },
          {
            name: "Hybrid (Jamba/Zamba)",
            color: "#e2b55a",
            points: [[0.6, 60.5], [1.2, 65.5], [2.4, 69.5], [4.8, 71.5], [9.6, 73.0]],
          },
        ]}
        xLabel="relative inference compute"
        yLabel="benchmark avg %"
        width={520}
      />

      <Prose>
        The hybrid curve sits above pure Transformer at every compute budget shown, and substantially above pure Mamba at the high-quality end. Pure Mamba wins on absolute compute efficiency (lowest x-axis values for equivalent y) but caps below transformer quality. Pure Transformer wins at peak quality if compute is unconstrained. The hybrid is Pareto-dominant: it matches or beats Transformer on quality while using 30-50% less inference compute, and it matches or beats Mamba on compute while gaining 3-5 quality points. This is the empirical evidence that motivated the hybrid arms race of 2024-2025.
      </Prose>

      <Plot
        label="KV cache + state memory vs context length (32-layer model, GQA)"
        series={[
          {
            name: "Pure Transformer (32 attn)",
            color: "#60a5fa",
            points: [[1024, 0.13], [4096, 0.54], [16384, 2.15], [65536, 8.59], [262144, 34.36]],
          },
          {
            name: "Hybrid Jamba-style (4 attn + 28 ssm)",
            color: "#e2b55a",
            points: [[1024, 0.03], [4096, 0.08], [16384, 0.28], [65536, 1.09], [262144, 4.31]],
          },
        ]}
        xLabel="context length L (tokens)"
        yLabel="cache memory (GB)"
        width={520}
      />

      <Prose>
        At 256K context the gap is 8x in absolute terms (4.3 GB vs 34 GB). The Jamba hybrid's slope is 8x shallower because only 4 of 32 layers contribute to the KV-cache growth term; the constant-size SSM state (~14 MB total across 28 layers) is invisible at this scale. This plot is the deployment economics argument for hybrids in one image: at long context, hybrids fit in a single 80 GB H100 and pure transformers do not.
      </Prose>

      <Heatmap
        label="layer kind across depth: Jamba-1.5-Large (32 layers, attn_every=8, MoE every 2)"
        matrix={[
          [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2],
          [3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
        ]}
        rowLabels={["Mixer", "FFN"]}
        colLabels={["L0","L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12","L13","L14","L15","L16","L17","L18","L19","L20","L21","L22","L23","L24","L25","L26","L27","L28","L29","L30","L31"]}
        cellSize={20}
        colorScale="gold"
      />

      <Prose>
        The top row shows mixer type per layer: value 1 = Mamba (28 layers), value 2 = Attention (4 layers, at indices 7, 15, 23, 31). The bottom row shows FFN type: value 3 = MoE (every other layer), value 0 = dense FFN. Notice the staggering: MoE FFNs are not specifically aligned with attention mixers — both decisions are independent. This 2D pattern (mixer kind × FFN kind) gives Jamba 4 distinct block types that can appear across its depth.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing among pure transformer, pure Mamba, and hybrid architectures depends on context length, deployment constraints, and quality budget. The 2026 picture is clearer than it was at any earlier point.
      </Prose>

      <H3>7.1 By context length</H3>

      <Prose>
        <strong>Short context ({"<"}4K tokens):</strong> Pure transformer wins. The KV cache is small (under 1 GB), attention's quality advantage is at its largest, and the entire toolchain (Llama, Mistral, etc.) is mature. Hybrids work but their efficiency gains do not yet compensate for the additional implementation complexity at this length.
      </Prose>

      <Prose>
        <strong>Medium context (4K-32K):</strong> Hybrid wins on cost-per-quality. Jamba-1.5-Mini at 32K context decodes faster than Mistral-7B-Instruct at the same length while matching benchmarks. Pure Mamba is competitive but lags by 1-2 quality points. Pure transformer works but pays 2-3x more memory.
      </Prose>

      <Prose>
        <strong>Long context (32K-256K):</strong> Hybrid is the practical choice. Pure transformer requires aggressive KV-cache optimization (sliding window, paged attention, KV-cache compression) to even fit; even with these tricks it loses quality at long context. Pure Mamba fits trivially but its recall capability degrades on multi-needle-in-haystack benchmarks. Hybrid keeps the lossless KV at 1/8 the cost and recovers most of the recall quality.
      </Prose>

      <Prose>
        <strong>Very long context (256K+):</strong> Hybrid or pure Mamba — pure transformer is out. Jamba-1.5-Mini handles 256K natively on a single H100; Jamba-1.5-Large handles 256K on 2x H100 with throughput roughly 3x a similarly-fitted dense transformer. For 1M+ context, pure Mamba (e.g., research-grade Mamba-2 long-context variants) is the only architecture that scales without serious memory pain.
      </Prose>

      <H3>7.2 By deployment constraint</H3>

      <Prose>
        <strong>Frontier serving (long context, cost-sensitive):</strong> Hybrid. Use Jamba 1.5 Mini for 256K context at H100-class throughput, Zamba-2 for 7B-class quality at lower cost.
      </Prose>

      <Prose>
        <strong>Quality-max (research, no cost ceiling):</strong> Pure transformer. Llama-3.1-405B and Claude-class models still lead on the absolute quality benchmarks; hybrids close the gap but do not yet exceed it on MMLU-Pro, GSM8K, or coding evals at the very top.
      </Prose>

      <Prose>
        <strong>Edge / on-device:</strong> Pure Mamba or small hybrid. Constant memory matters more than the last quality point. Hymba-1.5B-Base or small Mamba-2 variants run comfortably on phone-class hardware at 128K context.
      </Prose>

      <Prose>
        <strong>RAG and tool-use:</strong> Hybrid is well-matched. Long-context RAG benefits from cheap KV cache; the periodic attention layers do the document-citation lookups; the Mamba layers handle continuous integration of retrieved chunks. AI21 markets Jamba 1.5 Mini specifically for this use case.
      </Prose>

      <H3>7.3 The 1:7 ratio question</H3>

      <Prose>
        Jamba shipped with 1:7 (one attention per eight blocks). Zamba uses 1:6 with a shared attention block. Samba uses 1:1. Hymba uses within-layer 1:1 fusion. Why 1:7 specifically? Because Jamba's ablations showed the quality-vs-compute Pareto curve flattens around there: going to 1:5 or 1:3 adds compute without meaningful quality gain at 7B-52B scales; going to 1:15 starts to lose quality on retrieval benchmarks. The 1:7 ratio is "good enough on quality, clearly cheap on compute." It is not magic, and at smaller scales (under 1B) within-layer fusion (Hymba style) outperforms layer-level alternation. <strong>For new architectures at 7B+, default to 1:6 or 1:7 layer-level alternation; for sub-2B, consider within-layer fusion.</strong>
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        Jamba-1.5-Large pushes 398B total parameters and 94B active, with a 256K context window, all serving on 8x H100-80GB. It is the most credible non-transformer challenger at frontier scale. The scaling story has four threads.
      </Prose>

      <H3>8.1 Parameter count via MoE</H3>

      <Prose>
        Jamba's MoE FFNs are what make 52B and 398B totals feasible without quadrupling inference compute. Activated parameter ratios across the family: Jamba-v0.1 has 12B/52B = 23%, Jamba-1.5-Large has 94B/398B = 24%. The 4x parameter inflation gives roughly 1.5-2x quality gain in benchmark scores for free at decode time, with the cost paid only at training (where all experts must be loaded into VRAM during forward and backward).
      </Prose>

      <H3>8.2 Context length via SSM dominance</H3>

      <Prose>
        Because 7/8 of the model is Mamba and the SSM state is constant in <Code>L</Code>, doubling the context length only doubles the KV-cache cost (which is already only 1/8 of a comparable transformer's KV cache). Jamba-1.5-Mini scales from 4K to 256K without changing anything but RoPE theta and the training data length distribution. Pure transformers struggle past 32K-128K without specialized long-context training; hybrids extend almost for free.
      </Prose>

      <H3>8.3 Single-GPU 256K context</H3>

      <Prose>
        Jamba-v0.1 (52B/12B, 256K) was the first model to genuinely run a 256K context on a single H100-80GB at production throughput. The entire model state at 256K context fits in 80 GB: ~30 GB for weights (bf16), ~5 GB for KV cache + SSM state, the rest for activations and KV-cache headroom. By comparison, Llama-2-7B at 32K context already needs aggressive KV-cache management on a single H100 to avoid OOM; at 256K it is impossible without multi-GPU sharding.
      </Prose>

      <H3>8.4 The 2024-2025 hybrid wave</H3>

      <Prose>
        Production hybrid models shipped in 2024-2025: Jamba 1.0, Jamba 1.5 Mini, Jamba 1.5 Large (AI21); Zamba-1, Zamba2-7B, Zamba2-2.7B (Zyphra); Granite 3.0 Mamba MoE (IBM); Hymba 1.5B (NVIDIA); Falcon Mamba (TII); Samba (Microsoft Research). All of them adopted some variant of attention-SSM interleaving. The common thread: hybrids have become the default architecture choice for new long-context models, not an exotic experiment. Pure transformers still dominate at frontier short-to-medium context, but every new long-context release in the past 18 months has been a hybrid.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Too few attention layers (loses recall)</H3>

      <Prose>
        Going to 1:15 or 1:20 attention:Mamba ratios saves additional compute but degrades retrieval benchmarks meaningfully. The single attention layer per cycle is what enables induction-head circuits; if those circuits cannot form because attention layers are too sparse, in-context learning suffers. Jamba's ablations show 1:15 loses 2-3 points on Phonebook (a retrieval benchmark) versus 1:7. Diagnose by measuring needle-in-haystack accuracy across your context length; if it degrades significantly between Jamba-style ratio and your custom ratio, you have starved the model of attention.
      </Prose>

      <H3>9.2 Too many attention layers (loses Mamba's efficiency)</H3>

      <Prose>
        Conversely, going to 1:3 or 1:1 ratios pays much more compute and KV-cache memory for marginal quality gain. At 1:1 you have basically an interleaved attention-Mamba architecture (Samba) which is a different beast — it has its own merits at sub-2B scales but does not give you Jamba's economics at 7B+. <strong>If you find yourself adding more attention layers to fix quality, your problem is usually not the ratio; it is the placement, the SSM state dim, or the training distribution.</strong>
      </Prose>

      <H3>9.3 Wrong attention placement</H3>

      <Prose>
        Putting all attention layers at the start (first 4 of 32) or all at the end (last 4 of 32) is consistently worse than uniform spacing. The intuition: attention at the start has nothing to attend to (the residual stream carries little prior context); attention at the end has nowhere to send its output (the FFN compresses and the LM head reads). Uniform spacing gives attention a job to do at multiple depths and lets the residual stream benefit from each attention layer's output. Always uniform-space unless you have ablation evidence for an alternative.
      </Prose>

      <H3>9.4 KV cache placement bookkeeping bugs</H3>

      <Prose>
        Implementations that copy-paste a transformer KV-cache manager and forget that Jamba's KV cache only exists at attention layers will allocate <Code>{"n_\\text{layers}"}</Code> KV-cache slots when only <Code>{"n_\\text{attn}"}</Code> are needed. This wastes 7/8 of cache memory and breaks the long-context economics. Inversely, forgetting that Mamba layers need their own (constant-size) state buffer per request means the SSM state silently resets across batched requests, producing garbled outputs that look like quality regressions but are actually statefulness bugs. Both vLLM and TGI's Jamba implementations had to be rewritten to handle the dual cache types correctly.
      </Prose>

      <H3>9.5 MoE routing across hybrid layers</H3>

      <Prose>
        Jamba's MoE router is per-layer and independent of mixer type. A common bug in custom hybrid implementations: routing decisions trained on attention-block FFNs do not transfer well to Mamba-block FFNs because the residual stream's distribution differs at the two block types. Symptom: expert utilization is uneven (some experts process 30% of tokens, others 2%), training loss plateaus prematurely. Fix: shared router per layer, but train router weights separately for attention-FFN and Mamba-FFN positions. AI21 noted this in their training writeup; ignore it at your peril.
      </Prose>

      <H3>9.6 Fine-tuning recipes are different</H3>

      <Prose>
        Standard transformer fine-tuning recipes (constant LR for LoRA, AdamW with beta2=0.95) do not transfer cleanly to hybrids. Mamba layers prefer slightly higher LR than attention layers — typically 1.5-2x. The recommended LoRA target list must include both attention QKV/O AND Mamba <Code>x_proj</Code>, <Code>dt_proj</Code>, and <Code>out_proj</Code>; omitting the Mamba targets leaves 7/8 of the model frozen and gives weak finetuning quality. Mamba layers also benefit from gradient clipping at 1.0 (vs 0.5 for attention layers) because their gradient magnitudes are 2-3x larger.
      </Prose>

      <H3>9.7 Long-context training data dependence</H3>

      <Prose>
        Jamba's 256K context only works because it was trained with a long-context curriculum: 4K initial pretraining, then progressively extending to 32K, 64K, and 256K with carefully curated long documents. Just inflating the position encoding range at inference time does not work — the SSM's <Code>{"\\Delta_t"}</Code> values were tuned for short-range memory and do not generalize to very long context out of distribution. If you fine-tune a hybrid for long-context use, include long-context samples in the fine-tune data; pure short-context fine-tuning will degrade long-context capability.
      </Prose>

      <H3>9.8 Mamba CUDA kernel availability</H3>

      <Prose>
        Same trap as pure Mamba: if <Code>mamba-ssm</Code> and <Code>causal-conv1d</Code> are not installed and CUDA-compiled correctly, Jamba's Mamba layers fall back to a slow PyTorch reference. Symptom: decode is 10-30x slower than expected; the entire economic case for the hybrid evaporates. Always verify <Code>use_mamba_kernels=True</Code> succeeded and the optimized kernel is in use (the model logs this on first forward pass).
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Read in roughly chronological order. The Mamba and SSD papers are prerequisites; Jamba and the parallel hybrid papers can be read in any order after.
      </Prose>

      <StepTrace
        label="primary literature on hybrid SSM-attention architectures"
        steps={[
          {
            label: "Gu, Dao 2023 — Mamba (arXiv:2312.00752)",
            render: () => (
              <Prose>
                Gu, A., and Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752. Available at arxiv.org/abs/2312.00752. The selective SSM paper that made hybrid architectures possible. You cannot understand Jamba without this paper; in particular, sections 3 (selective SSM formulation) and 4 (hardware-aware kernel) are required reading. The paper also contains the critical observation that Mamba lags transformer on retrieval-heavy benchmarks (section 5), which directly motivates the hybrid design.
              </Prose>
            ),
          },
          {
            label: "Dao, Gu 2024 — Mamba-2 / SSD (arXiv:2405.21060)",
            render: () => (
              <Prose>
                Dao, T., and Gu, A. (2024). "Transformers Are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024. arXiv:2405.21060. Available at arxiv.org/abs/2405.21060. The state space duality paper. Proves that selective SSMs with scalar-identity state transitions are equivalent to masked linear attention. Section 4 introduces the SSD algorithm used in Mamba-2 and (with adaptations) in Zamba-2. Required for understanding why Mamba-2-based hybrids run at near-attention throughput on H100-class hardware.
              </Prose>
            ),
          },
          {
            label: "De et al. 2024 — Griffin (arXiv:2402.19427)",
            render: () => (
              <Prose>
                De, S., Smith, S.L., Fernando, A., Botev, A., Cristian-Muraru, G., Gu, A., Haroun, R., Berrada, L., Chen, Y., Srinivasan, S., Desjardins, G., Doucet, A., Budden, D., Teh, Y.W., Pascanu, R., De Freitas, N., Gulcehre, C. (2024). "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models." arXiv:2402.19427. Available at arxiv.org/abs/2402.19427. DeepMind's hybrid: replaces SSMs with a gated-linear-recurrent (RG-LRU) mixer interleaved with local-window attention. Published a month before Jamba and shows the hybrid principle generalizes beyond Mamba. Section 3 motivates the design; section 5 reports scaling results showing Griffin matches Llama-2-7B with 30% less training data.
              </Prose>
            ),
          },
          {
            label: "Lieber et al. 2024 — Jamba (arXiv:2403.19887)",
            render: () => (
              <Prose>
                Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., Shalev-Shwartz, S., Abend, O., Alon, R., Asida, T., Bergman, A., Glozman, R., Gokhman, M., Manevich, A., Ratner, N., Rozen, N., Shwartz, E., Zusman, M., and Shoham, Y. (2024). "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv:2403.19887. Available at arxiv.org/abs/2403.19887. The defining hybrid paper. Section 3.1 lays out the 1:7 attention:Mamba block pattern and the every-other-block MoE placement. Section 5 has the ablations on attention placement, ratio, and MoE config. Reproduces the long-context throughput and quality benchmarks. The hybrid template every subsequent production model has followed. Read this paper twice.
              </Prose>
            ),
          },
          {
            label: "Glorioso et al. 2024 — Zamba (arXiv:2405.16712)",
            render: () => (
              <Prose>
                Glorioso, P., Anthony, Q., Tokpanov, Y., Whittington, J., Pilault, J., Ibrahim, A., Millidge, B. (2024). "Zamba: A Compact 7B SSM Hybrid Model." arXiv:2405.16712. Available at arxiv.org/abs/2405.16712. Zyphra's hybrid: a single shared attention block applied periodically across a stack of Mamba blocks, using parameter sharing to keep the model compact. 7.4B parameters, beats Llama-2-7B and Mistral-7B on most benchmarks at substantially less training compute. Demonstrates that the hybrid principle works at compact scales without MoE inflation. Sections 3-4 on the shared-attention design are particularly worth reading — it is a clean alternative to Jamba's per-layer attention.
              </Prose>
            ),
          },
          {
            label: "AI21 2024 — Jamba 1.5 (arXiv:2408.12570)",
            render: () => (
              <Prose>
                AI21 Labs (2024). "Jamba 1.5: Hybrid Transformer-Mamba Models at Scale." arXiv:2408.12570. Available at arxiv.org/abs/2408.12570. Production-scale follow-up to Jamba 1.0. Introduces Jamba-1.5-Mini (52B/12B) and Jamba-1.5-Large (398B/94B), both at 256K context. Critical contributions: the ExpertsInt8 quantization scheme that enables 256K context on single-H100 inference for Mini and 8xH100 for Large; instruction tuning recipes for hybrids (different from transformer recipes); and benchmark results showing Jamba 1.5 Large matches Llama-3.1-70B on MMLU and outperforms it on long-context tasks. The deployment-focused paper for hybrid practitioners.
              </Prose>
            ),
          },
          {
            label: "Zhang et al. 2024 — Hymba (arXiv:2411.13676)",
            render: () => (
              <Prose>
                Zhang, X., Shen, Y., Lin, Z., Ren, Y., Yu, B., Wang, S., Sun, F., Zhao, Y., Diao, S., Li, Y., Aithal, S.K., Kuncoro, A., Suhr, A., Ma, X., Nadeem, M., Pavone, M., Molchanov, P., Yin, H. (2024). "Hymba: A Hybrid-head Architecture for Small Language Models." arXiv:2411.13676. Available at arxiv.org/abs/2411.13676. NVIDIA's within-layer hybrid: every layer has parallel attention heads and SSM heads, fused at the output. More aggressive integration than Jamba's layer-level alternation, optimized for sub-2B sizes. Hymba-1.5B outperforms Llama-3-3B at half the parameters. Section 3 details the head fusion; section 4 includes ablations comparing within-layer fusion to between-layer alternation at small scales.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        Five exercises. The first three test the architectural arithmetic; 4 tests intuition for the hybrid trade-off; 5 tests deployment judgment.
      </Prose>

      <H3>Exercise 1 (KV cache accounting)</H3>
      <Prose>
        A Jamba-style model has 32 layers with attention every 8th layer (4 attention layers, 28 Mamba layers), GQA with 8 KV heads, head dim 128, fp16. Compute the KV-cache memory (in GB) at <Code>{"L = 100{,}000"}</Code> tokens. Compare to a 32-layer pure-transformer with the same head config at the same length. What is the ratio?
      </Prose>
      <Callout accent="green">
        <strong>Answer 1.</strong> Per attention layer, per token, KV cache = <Code>{"2 \\cdot 8 \\cdot 128 \\cdot 2 = 4096"}</Code> bytes. Hybrid total = <Code>{"4 \\cdot 4096 \\cdot 100{,}000 = 1{,}638{,}400{,}000"}</Code> bytes = 1.64 GB. Pure-transformer = <Code>{"32 \\cdot 4096 \\cdot 100{,}000 = 13{,}107{,}200{,}000"}</Code> bytes = 13.1 GB. Ratio = <Code>{"32 / 4 = 8 \\times"}</Code>. The hybrid uses 1/8 the KV-cache memory at the same context length. This is exactly the savings ratio that lets Jamba fit 256K context on a single H100 where Llama-style transformers cannot.
      </Callout>

      <H3>Exercise 2 (compute breakdown)</H3>
      <Prose>
        At <Code>{"L = 32{,}768"}</Code>, <Code>{"d = 4096"}</Code>, <Code>{"N = 16"}</Code>, expand=2, with attention layers costing roughly <Code>{"4 L d^2 + 2 L^2 d"}</Code> FLOPs and Mamba layers costing roughly <Code>{"6 L d \\cdot \\text{expand} \\cdot N + 6 L d^2"}</Code> FLOPs, what fraction of total compute does the 4 attention layers consume in a 32-layer Jamba versus the 28 Mamba layers?
      </Prose>
      <Callout accent="green">
        <strong>Answer 2.</strong> Per attention layer: <Code>{"4 \\cdot 32768 \\cdot 4096^2 + 2 \\cdot 32768^2 \\cdot 4096 \\approx 2.2 \\times 10^{12} + 8.8 \\times 10^{12} = 1.1 \\times 10^{13}"}</Code> FLOPs. 4 attn layers = <Code>{"4.4 \\times 10^{13}"}</Code>. Per Mamba layer: <Code>{"6 \\cdot 32768 \\cdot 4096 \\cdot 2 \\cdot 16 + 6 \\cdot 32768 \\cdot 4096^2 \\approx 2.6 \\times 10^{10} + 3.3 \\times 10^{12} = 3.3 \\times 10^{12}"}</Code> FLOPs. 28 Mamba layers = <Code>{"9.2 \\times 10^{13}"}</Code>. Total = <Code>{"1.36 \\times 10^{14}"}</Code> FLOPs. Attention fraction = <Code>{"4.4 / 13.6 \\approx 32\\%"}</Code>. Mamba fraction = <Code>{"9.2 / 13.6 \\approx 68\\%"}</Code>. Even at 32K context, the 4 attention layers do roughly a third of the work — the quadratic term is real and growing. At 8K context the attention fraction drops to about 12%; at 128K it climbs to about 60%. This is why Jamba's compute story is "linear in context for most layers, quadratic only in 4 layers, and that pays off most at moderate-to-long contexts."
      </Callout>

      <H3>Exercise 3 (MoE active vs total parameters)</H3>
      <Prose>
        Jamba-1.5-Large has 32 layers, MoE every other layer (16 MoE layers), each MoE has 16 experts of which 2 are activated per token, and each expert has dense FFN dimension <Code>{"d_\\text{ffn} = 14336"}</Code> with model dim <Code>{"d = 4096"}</Code>. The other 16 layers have dense FFNs (also dim 14336). Compute the total FFN parameters and the per-token active FFN parameters. (Ignore mixer parameters and routers for this exercise.)
      </Prose>
      <Callout accent="green">
        <strong>Answer 3.</strong> One dense FFN: <Code>{"2 \\cdot d \\cdot d_\\text{ffn} = 2 \\cdot 4096 \\cdot 14336 \\approx 1.17 \\times 10^8"}</Code> = 117M params. One MoE FFN: <Code>{"16 \\cdot 117 \\text{M} = 1.88"}</Code> billion params total, <Code>{"2 \\cdot 117 \\text{M} = 234"}</Code> million active per token. Total across 16 dense + 16 MoE: dense FFNs total = <Code>{"16 \\cdot 117 \\text{M} = 1.87"}</Code>B; MoE FFNs total = <Code>{"16 \\cdot 1.88 \\text{B} = 30.1"}</Code>B; sum = 32B FFN parameters. Active per token: dense = 1.87B; MoE = <Code>{"16 \\cdot 234 \\text{M} = 3.74"}</Code>B; total active = 5.6B FFN parameters per token. The MoE inflates the FFN parameter pool 17x while only doubling the per-token active FFN compute. (Adding mixer and embedding parameters brings the totals to roughly 398B / 94B for Jamba-1.5-Large, matching the published numbers.)
      </Callout>

      <H3>Exercise 4 (placement intuition)</H3>
      <Prose>
        You are designing a hybrid with 24 blocks and a budget of 3 attention layers. List two reasonable placement strategies and one bad one. Justify briefly.
      </Prose>
      <Callout accent="green">
        <strong>Answer 4.</strong> <strong>Reasonable A:</strong> uniform spacing at indices 7, 15, 23 (every 8th block). This is the Jamba pattern; gives induction-head circuits multiple opportunities to form across depth. <strong>Reasonable B:</strong> indices 5, 12, 19 (slightly biased toward middle layers). Some ablations suggest middle attention is the most useful; this still maintains roughly uniform spacing while shifting the pattern slightly. <strong>Bad:</strong> indices 0, 1, 2 (all at the start) or 21, 22, 23 (all at the end). Early-only attention has nothing to attend to; late-only attention has nowhere to send its output and the model effectively becomes pure-Mamba for most of the depth. Both fail because they violate the principle that attention layers act as periodic "sync points" for the residual stream.
      </Callout>

      <H3>Exercise 5 (deployment judgment)</H3>
      <Prose>
        You need to deploy a model that handles both 4K-token chat queries and 200K-token document analysis on the same single-H100 server. Quality must be at least Llama-3-8B-class, and you cannot accept a 5x latency hit at long context. Pick an architecture and justify.
      </Prose>
      <Callout accent="green">
        <strong>Answer 5.</strong> Use a hybrid like Jamba-1.5-Mini (52B/12B, 256K context). Justification: (1) Quality at 4K is Llama-3-8B-class on standard benchmarks (Mini's 12B active beats 8B-dense on most evals because of MoE capacity). (2) At 200K context, KV cache fits comfortably in single-H100 memory because hybrid uses 1/8 the KV-cache pages a pure transformer would. (3) Decode latency at 200K stays within 2-3x of decode at 4K because Mamba layers have constant-time per-token cost; pure-transformer at 200K would be 50x+ slower than at 4K. (4) Single deployment serves both workloads — no need to swap models for long-vs-short. Pure Llama-3-8B fails (3) and (4); pure Mamba-7B fails (1) on instruction-following benchmarks; smaller hybrid (Hymba-1.5B) fails (1) on quality.
      </Callout>

    </div>
  ),
};

export default jambaContent;
