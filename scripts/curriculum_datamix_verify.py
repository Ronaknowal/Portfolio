"""
From-scratch implementations for the curriculum & data mixing topic.
Every output embedded in the JSX comes from here, verbatim.
"""

import math
import random
from collections import Counter

random.seed(0)

# ------------------------------------------------------------------
# 4a. Mixture sampler
# ------------------------------------------------------------------
def sample_mixture(domains, weights, n, rng=None):
    """Weighted sampling from multiple corpora.

    domains : dict[str, list[str]] — name -> list of documents
    weights : dict[str, float]     — name -> sampling weight (need not sum to 1)
    n       : int                   — total samples to draw
    """
    rng = rng or random
    names = list(domains.keys())
    w = [weights[k] for k in names]
    total = sum(w)
    probs = [x / total for x in w]
    out = []
    for _ in range(n):
        pick = rng.choices(names, weights=probs, k=1)[0]
        out.append((pick, rng.choice(domains[pick])))
    return out


print("=" * 60)
print("4a. Mixture sampler")
print("=" * 60)
domains = {
    "web":   ["w1", "w2", "w3", "w4", "w5"],
    "code":  ["c1", "c2", "c3"],
    "math":  ["m1", "m2"],
}
weights = {"web": 0.6, "code": 0.3, "math": 0.1}
samples = sample_mixture(domains, weights, n=10_000, rng=random.Random(0))
counts = Counter(d for d, _ in samples)
print(f"requested weights: {weights}")
print(f"empirical shares : {{'web': {counts['web']/10000:.3f}, "
      f"'code': {counts['code']/10000:.3f}, "
      f"'math': {counts['math']/10000:.3f}}}")


# ------------------------------------------------------------------
# 4b. Temperature-scaled sampling
# ------------------------------------------------------------------
def temperature_mix(domain_sizes, alpha):
    """Adjust weights by alpha.  p_i = size_i^alpha / sum_j size_j^alpha.
    alpha = 1 → proportional to size (natural frequency)
    alpha = 0 → uniform across domains
    0 < alpha < 1 → upweight the small domains
    """
    raised = {k: v ** alpha for k, v in domain_sizes.items()}
    total = sum(raised.values())
    return {k: v / total for k, v in raised.items()}


print()
print("=" * 60)
print("4b. Temperature-scaled sampling")
print("=" * 60)
sizes = {"en": 1_000_000, "es": 100_000, "hi": 10_000}
for a in (1.0, 0.75, 0.5, 0.3, 0.0):
    w = temperature_mix(sizes, a)
    print(f"alpha={a:.2f}  en={w['en']:.3f}  es={w['es']:.3f}  hi={w['hi']:.3f}")


# ------------------------------------------------------------------
# 4c. DoReMi-style reference-regret reweighting
# ------------------------------------------------------------------
def doremi_step(weights, ref_loss, model_loss, eta=0.3):
    """One step of DoReMi-style reweighting.

    excess_i  = max(0, model_loss_i - ref_loss_i)
    weights   <- renormalise( weights * exp(eta * excess) )
    """
    excess = {k: max(0.0, model_loss[k] - ref_loss[k]) for k in weights}
    unnorm = {k: weights[k] * math.exp(eta * excess[k]) for k in weights}
    Z = sum(unnorm.values())
    return {k: v / Z for k, v in unnorm.items()}


def toy_model_loss(weights, ref_loss, difficulty):
    """Toy model loss: model_loss_i = ref_loss_i + difficulty_i / (0.05 + weights_i).
    Harder domains need much more weight to reduce loss below reference, so at uniform
    weights the excess is largest for the hardest domains - DoReMi upweights them."""
    return {
        k: ref_loss[k] - 2.0 + difficulty[k] / (0.05 + weights[k])
        for k in weights
    }


print()
print("=" * 60)
print("4c. DoReMi-style reweighting (5 domains, 8 iterations)")
print("=" * 60)
domains5 = ["web", "code", "math", "books", "papers"]
ref_loss = {"web": 2.40, "code": 1.80, "math": 2.20, "books": 2.60, "papers": 2.30}
# Per-domain intrinsic difficulty: how much weight it takes to catch up.
# math and code are "harder to learn" per unit data in this toy setup.
difficulty = {"web": 0.30, "code": 0.80, "math": 1.00, "books": 0.40, "papers": 0.50}
w = {k: 1.0 / len(domains5) for k in domains5}
history = [dict(w)]
for it in range(10):
    ml = toy_model_loss(w, ref_loss, difficulty)
    w = doremi_step(w, ref_loss, ml, eta=0.6)
    history.append(dict(w))
print(f"{'iter':<5}" + "".join(f"{d:>10}" for d in domains5))
for i, h in enumerate(history):
    row = f"{i:<5}" + "".join(f"{h[d]:>10.3f}" for d in domains5)
    print(row)


# ------------------------------------------------------------------
# 4d. Cosine + cooldown schedule
# ------------------------------------------------------------------
def schedule(step, total, peak_lr=3e-4, min_lr=3e-5, warmup_frac=0.02, cooldown_frac=0.10):
    """Warmup → cosine → linear cooldown to min_lr."""
    warmup = int(total * warmup_frac)
    cool_start = int(total * (1.0 - cooldown_frac))
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    if step < cool_start:
        # cosine from peak_lr down to ~0.3 * peak_lr at the cooldown boundary
        p = (step - warmup) / (cool_start - warmup)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * p * 0.7))
    # linear cooldown
    p = (step - cool_start) / (total - cool_start)
    lr_at_cool = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * 0.7))
    return lr_at_cool + (min_lr - lr_at_cool) * p


print()
print("=" * 60)
print("4d. LR schedule (sampled)")
print("=" * 60)
total = 10_000
for frac in (0.0, 0.01, 0.02, 0.25, 0.50, 0.75, 0.88, 0.90, 0.95, 0.99, 1.00):
    step = int(frac * total)
    lr = schedule(step, total)
    print(f"frac={frac:.2f}  step={step:<6} lr={lr:.3e}")


# ------------------------------------------------------------------
# 4e. Replay buffer mixer for domain-adaptive pretraining
# ------------------------------------------------------------------
def replay_mix(general, domain, replay_frac, n, rng=None):
    """Sample n items from general+domain with replay_frac from general."""
    rng = rng or random
    out = []
    for _ in range(n):
        if rng.random() < replay_frac:
            out.append(("general", rng.choice(general)))
        else:
            out.append(("domain",  rng.choice(domain)))
    return out


print()
print("=" * 60)
print("4e. Replay buffer — domain-adaptive continued pretraining")
print("=" * 60)
general = [f"g{i}" for i in range(1000)]
domain  = [f"d{i}" for i in range(200)]   # narrow domain corpus
for rf in (0.0, 0.05, 0.10, 0.20, 0.50):
    m = replay_mix(general, domain, rf, n=10_000, rng=random.Random(1))
    shares = Counter(s for s, _ in m)
    print(f"replay_frac={rf:.2f}  general={shares['general']/10000:.3f}  "
          f"domain={shares['domain']/10000:.3f}")


# ------------------------------------------------------------------
# Sanity check: DoReMi convergence is actually sensible
# ------------------------------------------------------------------
print()
print("=" * 60)
print("Post-hoc: DoReMi final weights vs. intrinsic difficulty")
print("=" * 60)
final = history[-1]
for d in domains5:
    print(f"  {d:<8} difficulty={difficulty[d]:.2f}  final_weight={final[d]:.3f}")
print("(harder domains -> more weight: ordering should match.)")
