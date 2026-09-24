// Complete programs. Expected output is fixed after execution under Python 3.12.
export const monteCarloExamples = {
  independent: {
    title: 'Independent contributions and an equal-budget antithetic estimate',
    code: String.raw`import math
import random
import statistics

def estimate(seed, evaluations, paired=False):
    if evaluations < 2 or (paired and evaluations % 2):
        raise ValueError("Use at least two evaluations; pairs need an even count")
    rng = random.Random(seed)
    groups = []
    for _ in range(evaluations // (2 if paired else 1)):
        u = rng.random()
        groups.append((u*u + (1-u)**2) / 2 if paired else u*u)
    mcse = math.sqrt(statistics.variance(groups) / len(groups)) if len(groups) > 1 else None
    return statistics.mean(groups), mcse, len(groups)

for paired in [False, True]:
    result, mcse, groups = estimate(7, 1000, paired)
    print("paired" if paired else "iid", groups, f"mean={result:.6f}", f"MCSE={mcse:.6f}")
print("exact mean", f"{1/3:.6f}")
print("exact variance ratio", (4 / (45 * 1000)) / (1 / (90 * 1000)))`,
    expected: 'iid 1000 mean=0.318963 MCSE=0.009619\npaired 500 mean=0.336163 MCSE=0.003381\nexact mean 0.333333\nexact variance ratio 8.0',
  },
  importance: {
    title: 'Exact importance-weight accounting on three possible states',
    code: String.raw`from fractions import Fraction as F

target = [F(1, 5), F(1, 2), F(3, 10)]
proposal = [F(1, 3)] * 3
h = [0, 1, 4]
weights = [p / q for p, q in zip(target, proposal)]
truth = sum(p * value for p, value in zip(target, h))
weighted_mean = sum(q * w * value for q, w, value in zip(proposal, weights, h))
weighted_variance = sum(q * (w * value - truth)**2 for q, w, value in zip(proposal, weights, h))
print("weights", [str(w) for w in weights])
print("E_target[h]", truth, "E_proposal[w*h]", weighted_mean)
print("variance of one weighted contribution", weighted_variance)
# Self-normalization is a random ratio when the normalizer is unknown.
# With ONE draw its weighted numerator/denominator equals h(draw).
one_draw_ratio_mean = sum(q * value for q, value in zip(proposal, h))
print("one-draw self-normalized expectation", one_draw_ratio_mean)`,
    expected: "weights ['3/5', '3/2', '9/10']\nE_target[h] 17/10 E_proposal[w*h] 17/10\nvariance of one weighted contribution 109/50\none-draw self-normalized expectation 5/3",
  },
  metropolis: {
    title: 'The complete Beta(10,4) Metropolis sampler',
    code: String.raw`import math
import random

rng = random.Random(7)
alpha, beta = 10, 4

def log_target(theta):
    if not 0 < theta < 1:
        return float("-inf")
    return (alpha - 1) * math.log(theta) + (beta - 1) * math.log(1 - theta)

state, accepted, samples = 0.5, 0, []
for step in range(30_000):
    proposal = state + rng.gauss(0, 0.12)
    log_acceptance = log_target(proposal) - log_target(state)
    if math.log(rng.random()) < min(0.0, log_acceptance):
        state = proposal
        accepted += 1
    if step >= 2_000:  # retain draws after warmup, including repeats
        samples.append(state)

print(len(samples), round(accepted / 30_000, 3))
print(round(sum(samples) / len(samples), 3))
print(round(sum(x > 0.70 for x in samples) / len(samples), 3))`,
    expected: '28000 0.699\n0.711\n0.568',
  },
  hamiltonian: {
    title: 'A complete fixed-length HMC chain for a standard normal target',
    code: String.raw`import math
import random
import statistics

def leapfrog(q, r, epsilon):
    r -= epsilon * q / 2  # half kick: grad U(q) = q
    q += epsilon * r      # drift with unit mass
    r -= epsilon * q / 2  # second half kick at the new position
    return q, r

def energy(q, r):
    return (q*q + r*r) / 2

def sample_hmc(seed, iterations=12000, warmup=2000, epsilon=.2, steps=12):
    if not 0 <= warmup < iterations or epsilon <= 0 or steps < 1:
        raise ValueError("Invalid iteration or integration settings")
    rng = random.Random(seed)
    q, samples, accepted = 1.0, [], 0
    for iteration in range(iterations):
        r = rng.gauss(0, 1)  # fresh auxiliary momentum every transition
        old_energy = energy(q, r)
        proposal, momentum = q, r
        for _ in range(steps):
            proposal, momentum = leapfrog(proposal, momentum, epsilon)
        momentum = -momentum  # makes the deterministic proposal an involution
        new_energy = energy(proposal, momentum)
        if math.isfinite(new_energy) and math.log(rng.random()) < min(0, old_energy-new_energy):
            q = proposal
            accepted += 1
        if iteration >= warmup:
            samples.append(q)
    return samples, accepted / iterations

q, r = leapfrog(1, .7, .2)
print("first step", f"q={q:.3f}", f"r={r:.3f}", f"delta_H={energy(q,r)-energy(1,.7):.6f}")
draws, acceptance = sample_hmc(7)
print("draws", len(draws), "acceptance", f"{acceptance:.3f}")
print("mean", f"{statistics.mean(draws):.3f}", "variance", f"{statistics.variance(draws):.3f}")`,
    expected: 'first step q=1.120 r=0.488 delta_H=0.001272\ndraws 10000 acceptance 0.998\nmean -0.004 variance 0.977',
  },
  nuts: {
    title: 'A full, deliberately explicit slice-based NUTS chain',
    code: String.raw`import math
import random
import statistics

# Original simplified Algorithm 2 (Hoffman and Gelman, 2014), unit mass,
# standard normal target. Keep actual candidate lists for teaching clarity.
def energy(state):
    q, r = state
    return (q*q + r*r) / 2

def leapfrog(state, epsilon):
    q, r = state
    r -= epsilon*q/2
    q += epsilon*r
    r -= epsilon*q/2
    return q, r

def no_turn(left, right):
    displacement = right[0] - left[0]
    return displacement*left[1] >= 0 and displacement*right[1] >= 0

def build_tree(start, direction, depth, epsilon, log_slice):
    if depth == 0:
        state = leapfrog(start, direction*epsilon)
        joint_log = -energy(state)
        valid = math.isfinite(joint_log) and joint_log > log_slice - 1000
        candidates = [state] if valid and log_slice <= joint_log else []
        return state, state, candidates, valid
    left, right, candidates, keep = build_tree(start, direction, depth-1, epsilon, log_slice)
    if not keep:  # The paper permits early exit for a rejected subtree.
        return left, right, candidates, False
    endpoint = left if direction == -1 else right
    new_left, new_right, new_candidates, new_keep = build_tree(endpoint, direction, depth-1, epsilon, log_slice)
    if direction == -1:
        left = new_left
    else:
        right = new_right
    return left, right, candidates + new_candidates, new_keep and no_turn(left, right)

def nuts_step(q, rng, epsilon=.25, max_depth=6):
    if epsilon <= 0 or not 1 <= max_depth <= 10:
        raise ValueError("Use a positive step and depth from 1 to 10")
    initial = (q, rng.gauss(0, 1))
    log_slice = -energy(initial) + math.log(rng.random())
    left = right = initial
    candidates = [initial]
    for depth in range(max_depth):
        direction = -1 if rng.random() < .5 else 1
        start = left if direction == -1 else right
        a, b, new_candidates, keep = build_tree(start, direction, depth, epsilon, log_slice)
        if direction == -1:
            left = a
        else:
            right = b
        if keep:
            candidates.extend(new_candidates)
        # Internally turning new subtrees contribute NO candidates.
        # A whole-tree turn after a valid subtree can retain that subtree.
        if not keep or not no_turn(left, right):
            break
    chosen = candidates[int(rng.random()*len(candidates))]
    return chosen[0], len(candidates)

rng = random.Random(7)
q, samples, pool_sizes = 1.0, [], []
for iteration in range(14000):
    q, size = nuts_step(q, rng)
    if iteration >= 2000:
        samples.append(q)
        pool_sizes.append(size)
print("draws", len(samples))
print("mean", f"{statistics.mean(samples):.3f}", "variance", f"{statistics.variance(samples):.3f}")
print("mean candidate count", f"{statistics.mean(pool_sizes):.2f}")`,
    expected: 'draws 12000\nmean 0.014 variance 1.002\nmean candidate count 6.41',
  },
  transformedHmc: {
    title: 'Use HMC on the constrained Beta posterior through log-odds',
    code: String.raw`import math
import random
import statistics

def sigmoid(eta):
    if eta >= 0:
        return 1/(1+math.exp(-eta))
    e = math.exp(eta)
    return e/(1+e)

def softplus(x):
    return max(x,0) + math.log1p(math.exp(-abs(x)))

def potential(eta):
    # Minus log transformed Beta(10,4), including dtheta/deta.
    return 10*softplus(-eta) + 4*softplus(eta)

def gradient(eta):
    return 14*sigmoid(eta) - 10

def leapfrog(eta, momentum, epsilon):
    momentum -= epsilon*gradient(eta)/2
    eta += epsilon*momentum
    momentum -= epsilon*gradient(eta)/2
    return eta, momentum

rng = random.Random(7)
eta, samples, accepted = 0.0, [], 0
for iteration in range(14000):
    r = rng.gauss(0,1)
    old_energy = potential(eta)+r*r/2
    proposal, momentum = eta, r
    for _ in range(12):
        proposal, momentum = leapfrog(proposal,momentum,.3)
    momentum = -momentum
    new_energy = potential(proposal)+momentum*momentum/2
    if math.isfinite(new_energy) and math.log(rng.random()) < min(0,old_energy-new_energy):
        eta = proposal
        accepted += 1
    if iteration >= 2000:
        samples.append(sigmoid(eta))  # transform each retained draw back

print("draws",len(samples),"acceptance",f"{accepted/14000:.3f}")
print("mean theta",f"{statistics.mean(samples):.4f}")
print("P(theta > .70)",f"{statistics.mean(x>.7 for x in samples):.4f}")`,
    expected: 'draws 12000 acceptance 0.993\nmean theta 0.7175\nP(theta > .70) 0.5857',
  },
  precision: {
    title: 'Exact finite-chain precision: thinning under the same transition budget',
    code: String.raw`import math

def exact_error(transitions, rho, thin=1):
    if not -1 < rho < 1 or not 1 <= thin <= transitions:
        raise ValueError("Need abs(rho)<1 and a valid thinning interval")
    n = transitions // thin
    retained_rho = rho**thin
    inflation = 1 + 2*sum((1-lag/n)*retained_rho**lag for lag in range(1, n))
    variance = .25*inflation/n
    return n, math.sqrt(variance), .25/variance

for rho in [.8, 0, -.8]:
    for thin in [1, 2, 5]:
        n, error, equivalent = exact_error(100, rho, thin)
        print(f"rho={rho:+.1f} thin={thin} retained={n} MCSE={error:.4f} equivalent_iid={equivalent:.1f}")`,
    expected: 'rho=+0.8 thin=1 retained=100 MCSE=0.1466 equivalent_iid=11.6\nrho=+0.8 thin=2 retained=50 MCSE=0.1476 equivalent_iid=11.5\nrho=+0.8 thin=5 retained=20 MCSE=0.1542 equivalent_iid=10.5\nrho=+0.0 thin=1 retained=100 MCSE=0.0500 equivalent_iid=100.0\nrho=+0.0 thin=2 retained=50 MCSE=0.0707 equivalent_iid=50.0\nrho=+0.0 thin=5 retained=20 MCSE=0.1118 equivalent_iid=20.0\nrho=-0.8 thin=1 retained=100 MCSE=0.0170 equivalent_iid=861.7\nrho=-0.8 thin=2 retained=50 MCSE=0.1476 equivalent_iid=11.5\nrho=-0.8 thin=5 retained=20 MCSE=0.0810 equivalent_iid=38.1',
  },
  diagnostics: {
    title: 'Four chains: modern rank/folded R-hat and function-specific batch MCSE',
    code: String.raw`import math
import random
import statistics as stats

def chain(seed, start, scale=.12, draws=6000, warmup=1000):
    rng = random.Random(seed)
    def logp(x):
        return 9*math.log(x)+3*math.log1p(-x) if 0 < x < 1 else -math.inf
    x, result = start, []
    for i in range(draws):
        y = x + rng.gauss(0, scale)
        if math.log(rng.random()) < min(0, logp(y)-logp(x)):
            x = y
        if i >= warmup:
            result.append(x)  # repeats remain here
    return result

def split_chains(chains):
    n = len(chains[0]) // 2
    if len(chains) < 2 or n < 2 or any(len(c) != len(chains[0]) for c in chains):
        raise ValueError("Use equal-length chains with at least four draws")
    return [half for c in chains for half in (c[:n], c[-n:])]

def rank_normalize(chains):
    n = len(chains[0])
    flat = [x for c in chains for x in c]
    order = sorted(range(len(flat)), key=lambda i: flat[i])
    ranks = [0.0] * len(flat)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and flat[order[last]] == flat[order[first]]:
            last += 1
        rank = (first + 1 + last) / 2  # average one-based rank for ties
        for j in range(first, last):
            ranks[order[j]] = rank
        first = last
    normal = stats.NormalDist()
    # Corrected paper equation 14: PLUS 1/4 in the denominator.
    z = [normal.inv_cdf((r-3/8)/(len(flat)+1/4)) for r in ranks]
    return [z[i:i+n] for i in range(0, len(z), n)]

def basic_rhat(chains):
    n = len(chains[0])
    within = stats.mean(stats.variance(c) for c in chains)
    between = n*stats.variance([stats.mean(c) for c in chains])
    if within == 0:
        return math.inf if between else math.nan  # no misleading all-constant 1
    return math.sqrt(((n-1)*within/n + between/n) / within)

def rank_folded_rhat(chains):
    split = split_chains(chains)
    median = stats.median(x for c in split for x in c)
    folded = [[abs(x-median) for x in c] for c in split]
    return max(basic_rhat(rank_normalize(split)), basic_rhat(rank_normalize(folded)))

def batch_mcse(chains, function, batch_size=100):
    # Nonoverlapping batch means: an APPROXIMATION requiring batches long
    # relative to dependence, enough batches, and stable chains. No ESS claim.
    variance_of_chain_means = []
    for c in chains:
        if len(c) % batch_size or len(c) // batch_size < 2:
            raise ValueError("Use at least two complete equal-sized batches")
        batches = [stats.mean(function(x) for x in c[i:i+batch_size]) for i in range(0, len(c), batch_size)]
        variance_of_chain_means.append(stats.variance(batches)/len(batches))
    return math.sqrt(sum(variance_of_chain_means)) / len(chains)

chains = [chain(seed, start) for seed, start in zip([11, 22, 33, 44], [.1, .3, .7, .9])]
flat = [x for c in chains for x in c]
print("chains", len(chains), "draws each", len(chains[0]))
print("mean", f"{stats.mean(flat):.4f}", "posterior SD", f"{stats.stdev(flat):.4f}")
print("rank/folded R-hat", f"{rank_folded_rhat(chains):.4f}")
print("batch MCSE mean", f"{batch_mcse(chains, lambda x:x):.4f}")
print("batch MCSE tail", f"{batch_mcse(chains, lambda x:x>.7):.4f}")`,
    expected: 'chains 4 draws each 5000\nmean 0.7147 posterior SD 0.1170\nrank/folded R-hat 1.0020\nbatch MCSE mean 0.0025\nbatch MCSE tail 0.0086',
  },
};
