"""MoE from-scratch experiments — all stdout captured for the topic file."""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}, torch: {torch.__version__}")

# ---------------------------------------------------------------------------
# 4.1  Top-k gating from scratch
# ---------------------------------------------------------------------------
print("\n=== 4.1  Top-k gating ===")

class TopKGate(nn.Module):
    def __init__(self, d_model, n_experts, k=2):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.W_g = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        # x: [N, d_model]  (N = B*T tokens, flattened)
        logits = self.W_g(x)                        # [N, E]
        top_v, top_i = logits.topk(self.k, dim=-1)  # both [N, k]
        gates = F.softmax(top_v, dim=-1)            # renormalize over top-k
        return gates, top_i, logits

torch.manual_seed(0)
gate = TopKGate(d_model=16, n_experts=8, k=2)
x = torch.randn(6, 16)
g, idx, logits = gate(x)
print("gates (renormalized top-k probs):")
print(g)
print("expert indices (top-k per token):")
print(idx)
print("shape of raw logits:", tuple(logits.shape))
print("row sums of gates (should be 1.0):", g.sum(dim=-1).tolist())

# ---------------------------------------------------------------------------
# 4.2  MoE layer with N=8 experts, k=2
# ---------------------------------------------------------------------------
print("\n=== 4.2  MoE layer ===")

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model=64, d_ff=256, n_experts=8, k=2):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.gate = TopKGate(d_model, n_experts, k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        # x: [B, T, d_model]
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)                # [N, D]
        gates, idx, logits = self.gate(x_flat)      # [N, k], [N, k]

        out = torch.zeros_like(x_flat)
        # dispatch: for each expert, gather the tokens routed to it
        # (the slow but obvious version; production uses fused kernels)
        for e in range(self.n_experts):
            # mask[n] = True if expert e is in top-k of token n
            mask = (idx == e)                       # [N, k] bool
            if not mask.any():
                continue
            token_ids, slot_ids = mask.nonzero(as_tuple=True)  # positions
            token_input = x_flat[token_ids]                    # [M, D]
            weight = gates[token_ids, slot_ids].unsqueeze(-1)  # [M, 1]
            expert_out = self.experts[e](token_input) * weight
            # scatter-add back
            out.index_add_(0, token_ids, expert_out)

        # load balancing aux loss (Switch formulation)
        # f_i = fraction of tokens routed to expert i
        # P_i = average softmax gate prob over expert i across all tokens
        prob_all = F.softmax(logits, dim=-1)        # [N, E]
        P = prob_all.mean(dim=0)                    # [E]
        # fraction of tokens whose top-k includes expert i
        one_hot = F.one_hot(idx, self.n_experts).float().sum(dim=1)  # [N, E]
        f = one_hot.mean(dim=0)                     # [E]
        aux_loss = self.n_experts * (f * P).sum()

        # z-loss (router stability): mean of logsumexp(logits)^2
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        return out.view(B, T, D), aux_loss, z_loss

torch.manual_seed(0)
moe = MoELayer(d_model=64, d_ff=256, n_experts=8, k=2)
x = torch.randn(4, 10, 64)
y, aux, zl = moe(x)
print("input shape :", tuple(x.shape))
print("output shape:", tuple(y.shape))
print(f"aux loss    : {aux.item():.4f}   (uniform target ~ 1.0)")
print(f"z-loss      : {zl.item():.4f}")

# gradient check
y.sum().backward()
gs = [p.grad.abs().mean().item() for p in moe.parameters() if p.grad is not None]
print(f"n params with grad: {len(gs)}  mean |grad|: {sum(gs)/len(gs):.5f}")

# ---------------------------------------------------------------------------
# 4.3  Routing distribution — uniform vs collapsed
# ---------------------------------------------------------------------------
print("\n=== 4.3  Routing distribution ===")

def utilization(moe, x):
    x_flat = x.reshape(-1, x.shape[-1])
    _, idx, _ = moe.gate(x_flat)
    counts = torch.zeros(moe.n_experts)
    for k_slot in range(moe.k):
        counts += torch.bincount(idx[:, k_slot], minlength=moe.n_experts).float()
    return counts / counts.sum()

torch.manual_seed(0)
moe_a = MoELayer(64, 256, 8, 2)
x_big = torch.randn(8, 128, 64)
util = utilization(moe_a, x_big)
print("untrained gate utilization over 1024 tokens:")
print([f"{u:.3f}" for u in util.tolist()])
print(f"entropy: {(-util * (util+1e-9).log()).sum().item():.3f}  (uniform={math.log(8):.3f})")

# simulate a *collapsed* router (one expert dominates)
with torch.no_grad():
    moe_a.gate.W_g.weight.zero_()
    moe_a.gate.W_g.weight[3] = 100.0   # hard-code expert 3 as the winner
util_bad = utilization(moe_a, x_big)
print("collapsed gate utilization:")
print([f"{u:.3f}" for u in util_bad.tolist()])
print(f"entropy: {(-util_bad * (util_bad+1e-9).log()).sum().item():.3f}")

# ---------------------------------------------------------------------------
# 4.4  Dense vs MoE — param count, active FLOPs, small training
# ---------------------------------------------------------------------------
print("\n=== 4.4  Dense vs MoE param/flops comparison ===")

class DenseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

D, FF, N, K = 64, 256, 8, 2
dense = DenseFFN(D, FF)
moe_b = MoELayer(D, FF, N, K)

n_dense = sum(p.numel() for p in dense.parameters())
n_moe   = sum(p.numel() for p in moe_b.parameters())
# active params per token:  K experts + gate
n_expert = sum(p.numel() for p in moe_b.experts[0].parameters())
n_gate   = sum(p.numel() for p in moe_b.gate.parameters())
active_per_tok = K * n_expert + n_gate

print(f"dense params                : {n_dense:,}")
print(f"MoE total params (N=8, k=2) : {n_moe:,}   ({n_moe/n_dense:.1f}x dense)")
print(f"MoE active params per token : {active_per_tok:,}   ({active_per_tok/n_dense:.2f}x dense)")
print(f"MoE capacity / active ratio : {n_moe / active_per_tok:.1f}x")

# ---------------------------------------------------------------------------
# 4.5  Training on a copy task — monitor loss + utilization
# ---------------------------------------------------------------------------
print("\n=== 4.5  Training MoE on copy task ===")

class TinyMoEModel(nn.Module):
    def __init__(self, d_model=64, n_experts=8, k=2, aux_coef=0.01, z_coef=1e-3):
        super().__init__()
        self.moe = MoELayer(d_model, 4*d_model, n_experts, k)
        self.norm = nn.LayerNorm(d_model)
        self.aux_coef = aux_coef
        self.z_coef = z_coef
    def forward(self, x, target):
        y, aux, z = self.moe(self.norm(x))
        mse = F.mse_loss(y, target)
        total = mse + self.aux_coef * aux + self.z_coef * z
        return total, mse.detach(), aux.detach(), z.detach(), y.detach()

torch.manual_seed(0)
model = TinyMoEModel(64, 8, 2)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

# synthetic: target = roll(input, 1).  Model must learn to copy-shift.
B, T, D = 32, 16, 64
x = torch.randn(B, T, D)
tgt = torch.roll(x, 1, dims=1)

util_history = []
losses = []
for step in range(400):
    total, mse, aux, z, y = model(x, tgt)
    opt.zero_grad(); total.backward(); opt.step()
    if step % 50 == 0 or step == 399:
        with torch.no_grad():
            u = utilization(model.moe, x)
        util_history.append(u.tolist())
        losses.append((step, mse.item(), aux.item(), z.item()))
        print(f"step {step:4d}  mse={mse.item():.4f}  aux={aux.item():.4f}  z={z.item():.3f}  "
              f"util_entropy={(-u*(u+1e-9).log()).sum().item():.3f}  max_util={u.max().item():.3f}")

print("\nfinal expert utilization:")
final_u = util_history[-1]
print([f"{u:.3f}" for u in final_u])

# ---------------------------------------------------------------------------
# 4.6  With and without aux loss — collapse demonstration
# ---------------------------------------------------------------------------
print("\n=== 4.6  Aux loss on / off (collapse demo) ===")
for coef in [0.0, 0.01]:
    torch.manual_seed(0)
    m = TinyMoEModel(64, 8, 2, aux_coef=coef, z_coef=1e-3)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    x = torch.randn(32, 16, 64)
    tgt = torch.roll(x, 1, dims=1)
    for _ in range(300):
        total, mse, aux, z, y = m(x, tgt)
        opt.zero_grad(); total.backward(); opt.step()
    with torch.no_grad():
        u = utilization(m.moe, x)
    print(f"aux_coef={coef}  final mse={mse.item():.4f}  "
          f"util_entropy={(-u*(u+1e-9).log()).sum().item():.3f}  "
          f"max_util={u.max().item():.3f}  min_util={u.min().item():.3f}")
    print(f"   utilization: {[f'{v:.2f}' for v in u.tolist()]}")

# ---------------------------------------------------------------------------
# 4.7  Expert-token assignment heatmap data (B*T rows, N cols)
# ---------------------------------------------------------------------------
print("\n=== 4.7  Expert-token assignment batch stats ===")
torch.manual_seed(0)
m = TinyMoEModel(64, 8, 2, aux_coef=0.01)
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
x = torch.randn(32, 16, 64)
tgt = torch.roll(x, 1, dims=1)
for _ in range(400):
    total, *_ = m(x, tgt)
    opt.zero_grad(); total.backward(); opt.step()

# collect 1 batch of routing decisions
with torch.no_grad():
    _, idx, _ = m.moe.gate(x.reshape(-1, 64))
# count expert receipts per token position (averaging over batch)
matrix = torch.zeros(8, 8)   # rows: expert, cols: sequence position
for pos in range(16):
    pos_tokens = torch.arange(32) * 16 + pos    # [B]
    for e in range(8):
        hits = ((idx[pos_tokens] == e).sum()).item()
        matrix[e, pos // 2] += hits
matrix = matrix / matrix.sum(dim=0, keepdim=True).clamp(min=1)
print("expert x position heatmap (normalized per column, rounded):")
for row in matrix.tolist():
    print("  " + " ".join(f"{v:.2f}" for v in row))

# ---------------------------------------------------------------------------
# 4.8  Dense vs MoE loss curves (same active FLOPs)
# ---------------------------------------------------------------------------
print("\n=== 4.8  Dense vs MoE loss at equal active FLOPs ===")
torch.manual_seed(0)

class TinyDense(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ffn = DenseFFN(d, ff)
    def forward(self, x): return self.ffn(self.norm(x))

# Active compute parity: MoE(k=2, d_ff=256) == Dense(d_ff=512) per token
# (2 experts * 256 hidden ≈ one 512-hidden dense FFN for the FFN matmul)
dense_par = TinyDense(64, 512)
moe_par   = TinyMoEModel(64, 8, 2)
opt_d = torch.optim.Adam(dense_par.parameters(), lr=3e-3)
opt_m = torch.optim.Adam(moe_par.parameters(),   lr=3e-3)
B, T = 32, 16
x = torch.randn(B, T, 64)
tgt = torch.roll(x, 1, dims=1)

print(f"dense params   : {sum(p.numel() for p in dense_par.parameters()):,}")
print(f"MoE total      : {sum(p.numel() for p in moe_par.parameters()):,}")
print(f"MoE active/tok : ~{2*sum(p.numel() for p in moe_par.moe.experts[0].parameters()) + sum(p.numel() for p in moe_par.moe.gate.parameters()):,}")

for step in range(0, 401, 50):
    # train up to 'step'
    pass

d_losses, m_losses = [], []
for step in range(401):
    y_d = dense_par(x)
    ld = F.mse_loss(y_d, tgt)
    opt_d.zero_grad(); ld.backward(); opt_d.step()
    total, mse, *_ = moe_par(x, tgt)
    opt_m.zero_grad(); total.backward(); opt_m.step()
    if step % 50 == 0:
        d_losses.append((step, ld.item()))
        m_losses.append((step, mse.item()))
        print(f"step {step:4d}   dense_mse={ld.item():.4f}   moe_mse={mse.item():.4f}")

# ---------------------------------------------------------------------------
# 4.9  Token dropping — capacity factor
# ---------------------------------------------------------------------------
print("\n=== 4.9  Capacity factor and token dropping ===")
# If expert e gets more than capacity = ceil(cf * N_tokens / E) tokens, drop the excess.
def capacity_count(idx, n_experts, cf=1.25, k=2):
    N = idx.shape[0]
    cap = int(math.ceil(cf * N * k / n_experts))
    counts = torch.zeros(n_experts)
    for slot in range(k):
        c = torch.bincount(idx[:, slot], minlength=n_experts)
        counts += c
    dropped = sum(max(0, int(c) - cap) for c in counts)
    return cap, counts.tolist(), dropped

torch.manual_seed(0)
g = TopKGate(64, 8, 2)
x = torch.randn(128, 64)          # 128 tokens
_, idx, _ = g(x)
for cf in [1.0, 1.25, 1.5, 2.0]:
    cap, counts, dropped = capacity_count(idx, 8, cf)
    print(f"cf={cf:.2f}  cap={cap:3d}  counts={[int(c) for c in counts]}  dropped={dropped}")

print("\nAll experiments complete.")
