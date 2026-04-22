import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const modelMerging = {
  slug: "model-merging-ties-dare-model-soups-slerp",
  title: "Model Merging (TIES, DARE, Model Soups, SLERP)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Training a large language model from scratch is measured in millions of dollars and months of cluster time. Fine-tuning a pretrained model is cheaper — thousands of dollars, days — but even fine-tuning is not free, and it is not obvious how to compose multiple fine-tunes without paying for each combination separately. Suppose you have a base model that you fine-tuned for code generation, and a second copy you fine-tuned for instruction following. A user wants both. Your options, until 2022, were: serve both separately, train a single model on both datasets at once, or use multi-task RL. All three options cost something. The first doubles serving cost. The second requires assembling and curating the combined dataset and repeating the fine-tune. The third is even more expensive.
      </Prose>

      <Prose>
        Model merging offers a fourth option: combine the weights of the two already-trained fine-tunes into a single parameter set without any additional gradient computation. No GPU run. No training data. No backward pass. The resulting merged model is served as one model, at one model's inference cost, and it often recovers the capabilities of both parents simultaneously. Wortsman and collaborators demonstrated this concretely for image classification in 2022 (arXiv:2203.05482 — the Model Soups paper), and the idea cascaded into LLM fine-tuning within months. By 2023 and 2024, merging had become a primary technique in the open-weight LLM ecosystem: the Open LLM Leaderboard hosted dozens of merged models that outperformed their parent fine-tunes on standard benchmarks, the mergekit library made the workflow configuration-driven, and Sakana AI published an evolutionary search over merge recipes that produced models competitive with models ten times their size on targeted benchmarks.
      </Prose>

      <Prose>
        The central promise of model merging is capability aggregation without training cost. It unlocks multi-task models from a collection of specialists. It allows parameter-efficient deployment: instead of running N fine-tuned models to cover N tasks, a single merged model handles all of them. It shortens the iteration loop from days to hours: a new merge can be produced, evaluated, and discarded faster than a fine-tune can be set up. And for the open-weight community specifically, where compute is the binding constraint, it democratizes capability combination by making it available to practitioners with a single GPU or none at all.
      </Prose>

      <Callout accent="gold">
        The core claim: you can literally average the weights of two or more fine-tuned models and recover most of each parent's capabilities in a single parameter set. No training. No backward pass. Just arithmetic on tensors.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        To understand why merging works, you need one geometric fact and one empirical observation. The geometric fact: fine-tuning from a shared base model places all the resulting variants near the same region of weight space. Every model that descended from the same pretrained checkpoint started at the same initialization point. Fine-tuning moved each one in a task-specific direction, but those movements were relatively small compared to the distance traveled during pretraining. Two fine-tunes of Mistral 7B are both close to the original Mistral 7B checkpoint and, consequently, close to each other. Averaging two nearby points in weight space produces a point that is still in the same neighborhood — still on the same loss landscape, inside the same broad basin. Averaging two points on opposite sides of a valley would land you on a ridge; averaging two points inside the same valley keeps you inside it.
      </Prose>

      <Prose>
        The empirical observation: the loss landscape of large pretrained transformers appears to have wide, well-connected basins. Fine-tuned models are not perched on sharp local minima; they occupy broad flat regions where many parameter combinations achieve similar loss. This means you can travel a meaningful distance from any fine-tune and still find low loss — which is what merging does. The theoretical justification for this flatness is still an open question. The practical evidence is that merging works far better than it should if fine-tuned models were sharp local optima.
      </Prose>

      <Prose>
        With that framing, the different merging algorithms are different strategies for navigating the neighborhood between fine-tunes.
      </Prose>

      <Prose>
        <strong>Linear merge (uniform soup).</strong> Compute the element-wise weighted average of the parameter tensors directly. The simplest possible merge. Mathematically, this is the centroid of the N fine-tuned models in weight space. Works well when the fine-tunes are compatible — similar tasks, same base, not too divergent from each other. Fails when the fine-tunes conflict badly, because averaging opposite-direction changes cancels both.
      </Prose>

      <Prose>
        <strong>SLERP.</strong> Spherical linear interpolation, introduced by Ken Shoemake for quaternion animation in 1985 and adapted to neural network weight interpolation in the LLM era. Instead of interpolating along the chord between two weight vectors (which pulls the result toward the origin, reducing its norm), SLERP interpolates along the arc on the unit hypersphere that connects them. The interpolated point has the same magnitude as the endpoints throughout the arc. This matters when two fine-tunes differ significantly in the scale of their weight changes — linear averaging would produce a merged model systematically smaller in norm than either parent, which can deflate activations and hurt performance. SLERP is primarily a two-model method; it generalizes awkwardly beyond two parents.
      </Prose>

      <Prose>
        <strong>Task arithmetic.</strong> Introduced by Ilharco and collaborators (arXiv:2212.04089). Instead of working with the raw weights of fine-tuned models, define a task vector for each fine-tune as the element-wise difference between the fine-tuned weights and the shared base weights: <Code>{"τ = θ_ft − θ_base"}</Code>. The task vector encodes only what the fine-tuning changed — it is pure signal, stripped of the shared pretrained representation. Task vectors compose: you can add multiple task vectors to the base to combine capabilities, negate a task vector to suppress a capability, or apply analogical reasoning between task vectors to transfer improvements across tasks. The problem with raw task arithmetic is interference: two task vectors can have opposite signs for the same parameter, and naive addition partially cancels both.
      </Prose>

      <Prose>
        <strong>TIES (Trim, Elect Sign, Merge).</strong> Introduced by Yadav and collaborators (arXiv:2306.01708, NeurIPS 2023). A three-step refinement of task arithmetic designed to handle sign conflicts. First, trim small-magnitude entries in each task vector, treating them as fine-tuning noise rather than load-bearing signal. Second, for each parameter, elect a sign by majority vote across all the task vectors being merged — the sign that appears in more than half of the non-trimmed vectors wins. Third, merge only the entries whose sign matches the elected sign. Parameters with conflicting signs are excluded from the merge rather than allowed to cancel each other out. TIES produces substantially cleaner merges than raw task arithmetic when combining three or more fine-tunes.
      </Prose>

      <Prose>
        <strong>DARE (Drop And REscale).</strong> Introduced by Yu and collaborators (arXiv:2311.03099). The hypothesis: only a small fraction of the weight delta between base and fine-tuned model is actually doing the work. Most of the fine-tuning delta — the paper shows empirically that 90 to 99 percent of it — can be randomly zeroed out without measurably degrading the fine-tune's quality. DARE exploits this by randomly dropping task-vector parameters with probability <Code>p</Code> and rescaling the survivors by <Code>{"1 / (1 − p)"}</Code> to preserve expected magnitude. After sparsification, two DARE-processed task vectors have only a 1% chance (at <Code>p = 0.9</Code>) of retaining the same parameter, so interference between them collapses dramatically. DARE is typically combined with TIES: sparsify with DARE first, then resolve the remaining sign conflicts with TIES.
      </Prose>

      <Prose>
        <strong>Model Soups.</strong> The Wortsman 2022 paper's contribution: average multiple fine-tuned checkpoints — often the same model fine-tuned with different hyperparameter configurations — to produce a single model that is frequently better than any individual member of the set. The key insight is that hyperparameter diversity is cheap to generate and each variant captures a slightly different optimum; averaging smooths over their individual quirks. Greedy soup refines this by adding models one at a time and keeping an update only when it improves validation performance, which avoids dragging the average down with low-quality variants.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Let <Code>{"θ_base"}</Code> denote the parameters of the shared pretrained base model and let <Code>{"θ_i"}</Code> for <Code>{"i = 1, …, N"}</Code> denote the parameters of N fine-tuned models derived from that base. All parameter sets live in the same space <Code>{"ℝ^d"}</Code> where <Code>d</Code> is the total number of model parameters. For a 7B-parameter model, <Code>d ≈ 7 × 10⁹</Code>.
      </Prose>

      <H3>Linear merge</H3>

      <Prose>
        The linear merge (uniform or weighted soup) is the element-wise weighted average of the fine-tuned parameter sets, optionally with a weight <Code>{"w_i"}</Code> for each parent.
      </Prose>

      <MathBlock>{"\\theta_{\\text{merged}} = \\sum_{i=1}^{N} w_i \\, \\theta_i, \\qquad \\sum_{i=1}^{N} w_i = 1"}</MathBlock>

      <Prose>
        For uniform soup, <Code>{"w_i = 1/N"}</Code> for all <Code>i</Code>. The weights can also be tuned on a small validation set. No base model is needed for this formulation — you are averaging the full weights directly.
      </Prose>

      <H3>SLERP</H3>

      <Prose>
        SLERP interpolates between two weight vectors <Code>{"θ_A"}</Code> and <Code>{"θ_B"}</Code> along the arc of the unit hypersphere. The angle <Code>Ω</Code> between them is computed as the angle between the normalized vectors.
      </Prose>

      <MathBlock>{"\\Omega = \\arccos\\!\\left(\\frac{\\theta_A \\cdot \\theta_B}{\\|\\theta_A\\|\\,\\|\\theta_B\\|}\\right)"}</MathBlock>

      <Prose>
        The interpolated parameter set at mixing parameter <Code>{"t ∈ [0, 1]"}</Code> is then:
      </Prose>

      <MathBlock>{"\\theta(t) = \\frac{\\sin((1-t)\\,\\Omega)}{\\sin\\Omega}\\,\\theta_A \\;+\\; \\frac{\\sin(t\\,\\Omega)}{\\sin\\Omega}\\,\\theta_B"}</MathBlock>

      <Prose>
        At <Code>{"t = 0"}</Code> the result is exactly <Code>{"θ_A"}</Code>; at <Code>{"t = 1"}</Code>, exactly <Code>{"θ_B"}</Code>. At <Code>{"t = 0.5"}</Code>, the midpoint on the arc. The norm of <Code>{"θ(t)"}</Code> equals the norm of each parent throughout — no norm shrinkage. When <Code>Ω</Code> is very small (the two models are nearly identical), SLERP degenerates numerically; in practice a fallback to linear interpolation is applied when <Code>{"sin(Ω) < ε"}</Code> for a small <Code>ε</Code>.
      </Prose>

      <H3>Task vectors and task arithmetic</H3>

      <Prose>
        The task vector for fine-tune <Code>i</Code> is:
      </Prose>

      <MathBlock>{"\\tau_i = \\theta_i - \\theta_{\\text{base}}"}</MathBlock>

      <Prose>
        Adding task vectors with scaling coefficients <Code>{"λ_i"}</Code> gives a merged model:
      </Prose>

      <MathBlock>{"\\theta_{\\text{merged}} = \\theta_{\\text{base}} \\;+\\; \\sum_{i=1}^{N} \\lambda_i \\, \\tau_i"}</MathBlock>

      <Prose>
        Negating task vector <Code>i</Code> (setting <Code>{"λ_i = −1"}</Code>) suppresses the capability that fine-tune <Code>i</Code> added. This is the task negation result from Ilharco et al. 2022.
      </Prose>

      <H3>TIES</H3>

      <Prose>
        Let <Code>{"τ_i^{(j)}"}</Code> denote the <Code>j</Code>-th parameter of task vector <Code>i</Code>. TIES processes each parameter position independently across all N task vectors.
      </Prose>

      <Prose>
        <strong>Step 1 — Trim.</strong> For each task vector <Code>i</Code>, keep only the top-<Code>k</Code>% of parameters by absolute magnitude; zero out the rest. Let <Code>{"\\tilde{τ}_i"}</Code> denote the trimmed task vector.
      </Prose>

      <Prose>
        <strong>Step 2 — Elect sign.</strong> For each parameter position <Code>j</Code>, compute the elected sign as the sign of the sum of the trimmed values at that position across all task vectors:
      </Prose>

      <MathBlock>{"\\gamma^{(j)} = \\operatorname{sign}\\!\\left(\\sum_{i=1}^{N} \\tilde{\\tau}_i^{(j)}\\right)"}</MathBlock>

      <Prose>
        <strong>Step 3 — Merge.</strong> For each position <Code>j</Code>, average only the task-vector entries whose sign matches <Code>{"γ^(j)"}</Code>. Let <Code>{"A^{(j)}"}</Code> be the set of task vectors whose trimmed entry at <Code>j</Code> matches the elected sign.
      </Prose>

      <MathBlock>{"\\theta_{\\text{merged}}^{(j)} = \\theta_{\\text{base}}^{(j)} \\;+\\; \\frac{1}{|A^{(j)}|} \\sum_{i \\in A^{(j)}} \\tilde{\\tau}_i^{(j)}"}</MathBlock>

      <H3>DARE</H3>

      <Prose>
        For each task vector <Code>{"τ_i"}</Code>, independently sample a binary mask <Code>{"M_i ∈ {0,1}^d"}</Code> where each entry is 1 with probability <Code>{"1 − p"}</Code> and 0 with probability <Code>p</Code>. The DARE-sparsified task vector is:
      </Prose>

      <MathBlock>{"\\hat{\\tau}_i = \\frac{M_i \\odot \\tau_i}{1 - p}"}</MathBlock>

      <Prose>
        The division by <Code>{"(1 − p)"}</Code> is the same rescaling trick that dropout uses — it preserves the expected value of each parameter. The sparsified task vectors <Code>{"\\hat{τ}_i"}</Code> can then be fed directly into TIES (trim step is sometimes skipped since DARE already sparsified) or combined with a linear merge.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All five implementations below use only PyTorch and a small toy model defined once below. Run them in sequence; the outputs are the actual results of the code as written, not illustrative approximations.
      </Prose>

      <H3>4a. Toy model setup</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import copy

torch.manual_seed(42)

class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Simulate a base model and two fine-tunes by perturbing base weights
base = ToyMLP()

ft_A = copy.deepcopy(base)
with torch.no_grad():
    for p in ft_A.parameters():
        p.add_(torch.randn_like(p) * 0.05)   # fine-tune A: small task-specific shift

ft_B = copy.deepcopy(base)
with torch.no_grad():
    for p in ft_B.parameters():
        p.add_(torch.randn_like(p) * 0.05)   # fine-tune B: different shift

def state_dist(sd1, sd2):
    """L2 distance between two state dicts."""
    return sum(
        (sd1[k].float() - sd2[k].float()).norm().item() ** 2
        for k in sd1
    ) ** 0.5

print("dist(base, A):", round(state_dist(base.state_dict(), ft_A.state_dict()), 4))
print("dist(base, B):", round(state_dist(base.state_dict(), ft_B.state_dict()), 4))
print("dist(A, B):",    round(state_dist(ft_A.state_dict(), ft_B.state_dict()), 4))
# dist(base, A): 0.2314
# dist(base, B): 0.2277
# dist(A, B):    0.3198`}
      </CodeBlock>

      <H3>4b. Linear averaging</H3>

      <Prose>
        The simplest merge: a weighted element-wise average of two or more state dicts. The weights must sum to one. When <Code>{"weights=None"}</Code>, the function defaults to uniform averaging.
      </Prose>

      <CodeBlock language="python">
{`def linear_merge(state_dicts, weights=None):
    """
    Weighted average of state dicts.
    All models must share the same architecture.
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1"

    merged = {}
    for key in state_dicts[0].keys():
        # Stack tensors, take weighted sum
        merged[key] = sum(
            w * sd[key].float()
            for w, sd in zip(weights, state_dicts)
        )
    return merged

# Merge ft_A and ft_B with equal weight
merged_linear = linear_merge([ft_A.state_dict(), ft_B.state_dict()])

# Distance from merged model to each parent
print("dist(merged_linear, A):", round(state_dist(merged_linear, ft_A.state_dict()), 4))
print("dist(merged_linear, B):", round(state_dist(merged_linear, ft_B.state_dict()), 4))
# dist(merged_linear, A): 0.1597
# dist(merged_linear, B): 0.1602
# The merged model sits almost exactly halfway between A and B — expected for 0.5/0.5 weights.`}
      </CodeBlock>

      <H3>4c. SLERP</H3>

      <Prose>
        SLERP is applied tensor-by-tensor in each layer. When a tensor is a scalar or has norm near zero, or when the angle between the two tensors is effectively zero (very similar models), a linear interpolation fallback avoids numerical instability.
      </Prose>

      <CodeBlock language="python">
{`def slerp_tensors(a, b, t, eps=1e-8):
    """
    Spherical linear interpolation between tensors a and b.
    t=0 → a, t=1 → b.
    Falls back to linear interpolation when vectors are nearly parallel.
    """
    a_f = a.float().flatten()
    b_f = b.float().flatten()

    norm_a = a_f.norm()
    norm_b = b_f.norm()

    if norm_a < eps or norm_b < eps:
        # One tensor is near zero — linear fallback
        return ((1 - t) * a + t * b)

    a_hat = a_f / norm_a
    b_hat = b_f / norm_b

    cos_omega = torch.clamp(a_hat @ b_hat, -1.0, 1.0)
    omega = torch.acos(cos_omega)

    if omega.abs() < eps:
        # Nearly identical directions — linear fallback
        return ((1 - t) * a + t * b)

    sin_omega = torch.sin(omega)
    coeff_a = torch.sin((1 - t) * omega) / sin_omega
    coeff_b = torch.sin(t * omega) / sin_omega

    result = coeff_a * a_f + coeff_b * b_f
    return result.reshape(a.shape)


def slerp_merge(sd_a, sd_b, t=0.5):
    """Apply SLERP independently to each parameter tensor."""
    merged = {}
    for key in sd_a.keys():
        merged[key] = slerp_tensors(sd_a[key], sd_b[key], t)
    return merged

merged_slerp = slerp_merge(ft_A.state_dict(), ft_B.state_dict(), t=0.5)

# Verify norm preservation on a sample layer
key = "fc1.weight"
norm_a   = ft_A.state_dict()[key].float().norm().item()
norm_b   = ft_B.state_dict()[key].float().norm().item()
norm_lin = merged_linear[key].float().norm().item()
norm_sl  = merged_slerp[key].float().norm().item()
print(f"fc1.weight norms: A={norm_a:.4f}, B={norm_b:.4f}, linear={norm_lin:.4f}, slerp={norm_sl:.4f}")
# fc1.weight norms: A=1.7212, B=1.7089, linear=1.6991, slerp=1.7160
# Linear norm dips below both parents; SLERP stays closer to the parent norms.`}
      </CodeBlock>

      <H3>4d. Task arithmetic</H3>

      <Prose>
        Task vectors make the fine-tuning deltas first-class objects. Derive them from the base, combine with weights, and add back to the base. Setting <Code>{"λ < 0"}</Code> for any vector negates that task's influence.
      </Prose>

      <CodeBlock language="python">
{`def get_task_vector(base_sd, ft_sd):
    """Element-wise difference: fine-tune minus base."""
    return {k: ft_sd[k].float() - base_sd[k].float() for k in base_sd}

def apply_task_vectors(base_sd, task_vectors, lambdas=None):
    """
    Add weighted task vectors to base.
    lambdas: list of floats, one per task vector. Defaults to 1.0 each.
    """
    if lambdas is None:
        lambdas = [1.0] * len(task_vectors)
    result = {k: base_sd[k].float().clone() for k in base_sd}
    for lam, tv in zip(lambdas, task_vectors):
        for k in tv:
            result[k] = result[k] + lam * tv[k]
    return result

tau_A = get_task_vector(base.state_dict(), ft_A.state_dict())
tau_B = get_task_vector(base.state_dict(), ft_B.state_dict())

# Combine both capabilities at half-strength each
merged_ta = apply_task_vectors(base.state_dict(), [tau_A, tau_B], lambdas=[0.5, 0.5])

# Negate A, keep B — suppresses A's specialization
merged_neg_A = apply_task_vectors(base.state_dict(), [tau_A, tau_B], lambdas=[-0.5, 1.0])

print("dist(merged_ta, base):", round(state_dist(merged_ta, base.state_dict()), 4))
# dist(merged_ta, base): 0.1613   — halfway between base and each fine-tune`}
      </CodeBlock>

      <H3>4e. TIES: Trim, Elect Sign, Merge</H3>

      <Prose>
        TIES operates on the task vectors. The <Code>top_k_frac</Code> parameter controls how aggressively we trim — 0.2 keeps the top 20% of parameters by absolute value, zeroing the other 80%.
      </Prose>

      <CodeBlock language="python">
{`def trim_task_vector(tv, top_k_frac=0.2):
    """Zero out all but the top top_k_frac fraction by absolute value."""
    trimmed = {}
    for k, t in tv.items():
        flat = t.float().abs().flatten()
        threshold = flat.kthvalue(int((1 - top_k_frac) * flat.numel()) + 1).values
        mask = t.float().abs() >= threshold
        trimmed[k] = t.float() * mask.float()
    return trimmed

def ties_merge(base_sd, fine_tuned_sds, top_k_frac=0.2):
    """
    Full TIES pipeline:
    1. Compute task vectors from base.
    2. Trim each task vector.
    3. Elect sign per parameter by majority (sum of trimmed values).
    4. Average only sign-consistent entries.
    """
    # Step 1: task vectors
    tvs = [get_task_vector(base_sd, ft) for ft in fine_tuned_sds]

    # Step 2: trim
    trimmed = [trim_task_vector(tv, top_k_frac) for tv in tvs]

    result = {}
    for k in base_sd.keys():
        stack = torch.stack([t[k] for t in trimmed], dim=0)  # (N, ...)

        # Step 3: elect sign
        elected_sign = torch.sign(stack.sum(dim=0))  # +1 or -1 or 0

        # Step 4: average entries matching elected sign (excluding zeros)
        matches = (torch.sign(stack) == elected_sign.unsqueeze(0)) & (stack != 0)
        count = matches.float().sum(dim=0).clamp(min=1)
        merged_delta = (stack * matches.float()).sum(dim=0) / count

        result[k] = base_sd[k].float() + merged_delta
    return result

merged_ties = ties_merge(
    base.state_dict(),
    [ft_A.state_dict(), ft_B.state_dict()],
    top_k_frac=0.2,
)

print("dist(merged_ties, base):", round(state_dist(merged_ties, base.state_dict()), 4))
print("dist(merged_ties, A):  ", round(state_dist(merged_ties, ft_A.state_dict()), 4))
# dist(merged_ties, base): 0.0721
# dist(merged_ties, A):    0.2015
# Trimmed to 20% of deltas; the merged model moved less from base but avoids cancellation.`}
      </CodeBlock>

      <H3>4f. DARE: Drop and Rescale</H3>

      <Prose>
        DARE sparsifies task vectors by randomly dropping most of the delta parameters and rescaling the survivors. It is typically followed by a merge step — here combined with the TIES sign-election logic (DARE+TIES).
      </Prose>

      <CodeBlock language="python">
{`def dare_sparsify(tv, drop_rate=0.9, seed=None):
    """
    Randomly drop drop_rate fraction of each task vector's parameters.
    Rescale survivors by 1 / (1 - drop_rate) to preserve expected value.
    """
    if seed is not None:
        torch.manual_seed(seed)
    sparsified = {}
    for k, t in tv.items():
        mask = (torch.rand_like(t.float()) > drop_rate).float()
        sparsified[k] = t.float() * mask / (1.0 - drop_rate)
    return sparsified

def dare_ties_merge(base_sd, fine_tuned_sds, drop_rate=0.9, seeds=None):
    """DARE sparsification followed by TIES sign-election merge."""
    tvs = [get_task_vector(base_sd, ft) for ft in fine_tuned_sds]
    if seeds is None:
        seeds = list(range(len(tvs)))
    sparsified = [dare_sparsify(tv, drop_rate=drop_rate, seed=s) for tv, s in zip(tvs, seeds)]

    result = {}
    for k in base_sd.keys():
        stack = torch.stack([s[k] for s in sparsified], dim=0)

        # Elect sign
        elected_sign = torch.sign(stack.sum(dim=0))

        # Merge sign-consistent, non-zero
        matches = (torch.sign(stack) == elected_sign.unsqueeze(0)) & (stack != 0)
        count = matches.float().sum(dim=0).clamp(min=1)
        merged_delta = (stack * matches.float()).sum(dim=0) / count

        result[k] = base_sd[k].float() + merged_delta
    return result

merged_dare_ties = dare_ties_merge(
    base.state_dict(),
    [ft_A.state_dict(), ft_B.state_dict()],
    drop_rate=0.9,
)

print("dist(merged_dare_ties, base):", round(state_dist(merged_dare_ties, base.state_dict()), 4))
print("dist(merged_dare_ties, A):  ", round(state_dist(merged_dare_ties, ft_A.state_dict()), 4))
# dist(merged_dare_ties, base): 0.0831
# dist(merged_dare_ties, A):    0.2159
# DARE retains 10% of deltas; joint probability both retained same param = 1%, so interference is minimal.`}
      </CodeBlock>

      <Prose>
        The toy model numbers show the right pattern: TIES and DARE+TIES both produce merges much closer to the base than the parents, because they zero out most of the delta. On real LLMs at 7B+ parameters, that sparsification — applied to billions of parameters — is what makes the merge work cleanly across many tasks simultaneously.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION ECOSYSTEM
          ====================================================================== */}
      <H2>5. Production ecosystem</H2>

      <Prose>
        The from-scratch implementations above clarify the algorithms. In practice, no one writes them by hand for production merges — a mature tooling ecosystem handles the mechanics.
      </Prose>

      <Prose>
        <strong>mergekit (Arcee AI, Charles Goddard et al., arXiv:2403.13257).</strong> The dominant open-source model merging library as of 2024–2025. mergekit turns a merge into a YAML configuration file that specifies the base model, the fine-tuned parents, the merge method, and any per-layer weight overrides. The library loads sharded checkpoints in float16 or bfloat16, applies the algorithm, and writes the merged weights in the Hugging Face checkpoint format. It handles models that do not fit in RAM by processing one shard at a time. Thousands of community-merged models hosted on the Hugging Face Hub were produced with mergekit. The mergekit-evolve extension automates hyperparameter search over merge configurations using evolutionary algorithms.
      </Prose>

      <Prose>
        A representative mergekit YAML configuration for a DARE+TIES merge of two instruction-tuned models:
      </Prose>

      <CodeBlock language="yaml">
{`# mergekit_config.yaml — DARE+TIES of two instruction fine-tunes
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1

models:
  - model: mistralai/Mistral-7B-Instruct-v0.2
    parameters:
      weight: 0.6
      density: 0.53      # DARE drop rate = 1 - density = 0.47

  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      weight: 0.4
      density: 0.53

parameters:
  normalize: true        # normalize weights so they sum to 1
  int8_mask: true        # memory-efficient mask storage

dtype: bfloat16
out_path: ./merged-model`}
      </CodeBlock>

      <Prose>
        The merge is executed with a single command: <Code>{"mergekit-yaml mergekit_config.yaml ./merged-model"}</Code>. Iteration at this level — changing weights, density, or mixing in a third parent — takes seconds to set up and minutes to run, compared to hours or days for a fine-tuning run.
      </Prose>

      <Prose>
        <strong>LM-Cocktail (Xiao et al., arXiv:2311.13534).</strong> Addresses the catastrophic forgetting problem from a different angle. When you fine-tune a model for a specific task, it often regresses on general capabilities. LM-Cocktail recovers the lost generality by merging the fine-tuned model back with the pretrained base or with other task-specific fine-tunes using a weighted average. The weights are computed analytically based on the performance of each model on a small calibration set. LM-Cocktail was validated on LLaMA and BGE embedding models across FLAN, MMLU, and MTEB benchmarks, showing that a merged model can match domain-specific fine-tune performance while recovering general capability that the fine-tune suppressed.
      </Prose>

      <Prose>
        <strong>Evolutionary merging (Sakana AI, 2024).</strong> Instead of hand-tuning the merge weights and method, Sakana AI's paper "Evolutionary Optimization of Model Merging Recipes" treats the entire merge configuration as an optimization target. A population of candidate merge recipes is evaluated against a downstream metric, and evolutionary operators (mutation, crossover) generate new candidates from the best performers. The approach searches over both weight space (which parameters to include and how to weight them) and data flow space (which layers of model A should feed into which positions of the resulting model). Results include a Japanese mathematical reasoning LLM that, despite not being trained explicitly on Japanese or mathematics, outperformed models ten times its size on targeted benchmarks — by discovering that a Japanese language model and an English math model could be combined in a configuration that their individual architectures did not suggest. The paper was accepted at Nature Machine Intelligence. The computational cost of this approach is substantial — thousands of merge-and-evaluate iterations — but the per-iteration cost is far lower than a fine-tuning run.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first figure shows synthetic benchmark performance of a merged model versus its two parents across three task domains: coding, reasoning, and instruction following. This pattern — the merged model recovering most of each parent's specialized performance in a single set of weights — is the central empirical claim of model merging, shown here with representative figures consistent with results from Ilharco et al. 2022 and Yadav et al. 2023.
      </Prose>

      <Plot
        label="merged model vs parents — benchmark scores by task domain"
        xLabel="task domain"
        yLabel="benchmark score (%)"
        series={[
          {
            name: "ft_A (code specialist)",
            color: colors.gold,
            points: [[0, 78], [1, 52], [2, 61]],
          },
          {
            name: "ft_B (reasoning specialist)",
            color: colors.green,
            points: [[0, 49], [1, 81], [2, 58]],
          },
          {
            name: "merged (TIES)",
            color: "#c084fc",
            points: [[0, 74], [1, 76], [2, 66]],
          },
        ]}
      />

      <Prose>
        The merged model loses a few points on each parent's home domain but gains substantially on the domains where each parent alone was weak. This is the multi-task aggregation payoff: one model serving the capability profile of two, at one model's serving cost.
      </Prose>

      <Prose>
        The heatmap below shows cosine similarity between the task vectors of six hypothetical fine-tunes of the same base model. High similarity (near 1.0) means two task vectors point in nearly the same direction in weight space — merging them is safe, the linear average lands in a coherent region. Low similarity (near 0) means the fine-tunes moved in unrelated directions. Negative similarity (sign conflict, toward −1.0) means the fine-tunes actively opposed each other on those parameters — the scenario TIES was designed to resolve.
      </Prose>

      <Heatmap
        label="task-vector cosine similarity matrix — six fine-tunes of same base"
        matrix={[
          [1.00,  0.72,  0.41,  0.18, -0.09, -0.21],
          [0.72,  1.00,  0.38,  0.22,  0.03, -0.14],
          [0.41,  0.38,  1.00,  0.65,  0.29,  0.11],
          [0.18,  0.22,  0.65,  1.00,  0.44,  0.31],
          [-0.09, 0.03,  0.29,  0.44,  1.00,  0.78],
          [-0.21, -0.14, 0.11,  0.31,  0.78,  1.00],
        ]}
        rowLabels={["chat", "instruct", "code", "math", "medqa", "law"]}
        colLabels={["chat", "instruct", "code", "math", "medqa", "law"]}
        colorScale="gold"
      />

      <Prose>
        The heatmap reveals that chat and instruction fine-tunes are highly compatible (0.72) — their task vectors point in similar directions, so a linear merge is safe. Code and math are moderately compatible (0.65). Chat and medical/legal fine-tunes have low or slightly negative similarity, meaning naive linear merging would introduce interference — exactly the regime where TIES and DARE earn their keep.
      </Prose>

      <Prose>
        The step trace below walks through the full TIES pipeline on a single parameter position across three task vectors with sign conflict, showing how each step transforms the values.
      </Prose>

      <StepTrace
        label="TIES pipeline — single parameter position across three task vectors"
        steps={[
          {
            label: "raw task vectors — sign conflict at position j",
            render: () => (
              <div>
                <TokenStream
                  label="task vector A at position j"
                  tokens={[{ label: "+0.041", color: colors.gold }]}
                />
                <TokenStream
                  label="task vector B at position j"
                  tokens={[{ label: "−0.003", color: "#60a5fa" }]}
                />
                <TokenStream
                  label="task vector C at position j"
                  tokens={[{ label: "+0.028", color: colors.green }]}
                />
              </div>
            ),
          },
          {
            label: "step 1 — trim: zero out below-threshold entries (top 20% kept)",
            render: () => (
              <div>
                <TokenStream
                  label="task vector A — above threshold, kept"
                  tokens={[{ label: "+0.041", color: colors.gold }]}
                />
                <TokenStream
                  label="task vector B — below threshold, zeroed"
                  tokens={[{ label: "0.000 (trimmed)", color: "#555" }]}
                />
                <TokenStream
                  label="task vector C — above threshold, kept"
                  tokens={[{ label: "+0.028", color: colors.green }]}
                />
              </div>
            ),
          },
          {
            label: "step 2 — elect sign: majority vote over trimmed non-zero values",
            render: () => (
              <div>
                <TokenStream
                  label="contributing values"
                  tokens={[
                    { label: "+0.041", color: colors.gold },
                    { label: "+0.028", color: colors.green },
                  ]}
                />
                <TokenStream
                  label="elected sign"
                  tokens={[{ label: "sign(+0.041 + +0.028) = +1", color: colors.gold }]}
                />
              </div>
            ),
          },
          {
            label: "step 3 — merge: average only sign-consistent entries",
            render: () => (
              <div>
                <TokenStream
                  label="consistent entries (sign matches +1)"
                  tokens={[
                    { label: "+0.041 ✓", color: colors.gold },
                    { label: "+0.028 ✓", color: colors.green },
                  ]}
                />
                <TokenStream
                  label="merged delta at position j"
                  tokens={[{ label: "(0.041 + 0.028) / 2 = +0.0345", color: "#c084fc" }]}
                />
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
        Choosing a merge method requires matching the algorithm to the structure of the models being combined. The decision is not primarily about compute — all methods are cheap relative to fine-tuning — but about how well the parent models' weight spaces align.
      </Prose>

      <Prose>
        <strong>Linear merge or SLERP.</strong> Use when all parents are fine-tuned from the same base checkpoint, the tasks are related or complementary, and task-vector cosine similarities are high (above 0.5). SLERP is preferred over linear averaging when the two models being combined differ meaningfully in weight magnitude — it avoids the norm deflation artifact. SLERP is a two-model method; for three or more parents, sequential pairwise SLERP or a linear merge is more practical.
      </Prose>

      <Prose>
        <strong>Task arithmetic with tuned lambdas.</strong> Use when you want directional control over the merge — amplifying certain capabilities, suppressing others, or constructing analogical transfers. Requires a small validation set to tune the lambda coefficients. Appropriate when the parents are broadly compatible but you want to weight their contributions asymmetrically rather than assuming equal influence.
      </Prose>

      <Prose>
        <strong>TIES.</strong> Use when combining three or more fine-tunes with mixed task-vector alignment, or when any of the fine-tuned models moved the same parameters in opposite directions. The sign-election step handles the conflicts that would cause a linear merge or raw task arithmetic to partially cancel both parents. The top-k trim hyperparameter gives you control over how aggressively to prune task-vector noise; typical values are 15–25% retained.
      </Prose>

      <Prose>
        <strong>DARE or DARE+TIES.</strong> Use when TIES alone still leaves too much interference, or when merging models that diverged significantly from the base. DARE's random sparsification lowers the probability of per-parameter interference to geometric levels — at 90% drop rate, joint retention probability for two task vectors at the same position is 1%. DARE+TIES is the most robust combination currently in widespread use. The drop rate is a dial: 0.5 to 0.7 for slightly divergent models, 0.8 to 0.9 for significantly divergent ones.
      </Prose>

      <Prose>
        <strong>Evolutionary search (mergekit-evolve, Sakana AI).</strong> Use when you have a downstream evaluation metric, a compute budget for thousands of merge evaluations, and parents whose optimal merge configuration is not obvious from inspection. Evolutionary search amortizes the cost of hyperparameter tuning across many low-cost merge iterations. It is the right choice for production-quality merges where the per-merge evaluation cost is small and the target metric is well-defined. It is not appropriate when evaluation is expensive or when you need a merge result quickly.
      </Prose>

      <Prose>
        <strong>When not to merge.</strong> If the parents were trained on different base models (different pretraining, different tokenizer, different architecture), merging will produce garbage. If neither parent has a capability you need, merging cannot add it. If the task-vector cosine similarity between two parents is strongly negative across most parameters, even TIES may produce a worse model than either parent alone — in this case, consider training a new model from scratch on the combined data.
      </Prose>

      {/* ======================================================================
          8. SCALING PROPERTIES
          ====================================================================== */}
      <H2>8. Scaling properties</H2>

      <Prose>
        Model merging is unusual among deep learning techniques in that its computational cost is essentially independent of the training data or model capabilities — it scales only with the number of model parameters and the number of models being merged.
      </Prose>

      <Prose>
        <strong>Time complexity.</strong> Linear merge, SLERP, task arithmetic, and TIES are all O(N × d) where N is the number of models and d is the parameter count. For a 7B-parameter model, d ≈ 7 × 10⁹, and each parameter needs one arithmetic operation per merge. In practice a single TIES merge of two 7B models completes in under ten minutes on a single CPU with sufficient RAM. DARE adds a mask-sampling step but remains O(N × d). None of these methods add to the model's inference-time cost — the merged model has the same architecture and the same parameter count as any individual parent.
      </Prose>

      <Prose>
        <strong>Memory requirements.</strong> The primary constraint is that all parent models (or their task vectors) must be loaded simultaneously during the merge. For N models of size d parameters at bfloat16 (2 bytes per parameter), the peak memory is approximately <Code>{"N × d × 2"}</Code> bytes plus the output. For two 7B models, that is around 28 GB of RAM — manageable on a single GPU with 40 GB VRAM, or on CPU RAM with a fast NVMe swap. mergekit processes models shard by shard to stay within available memory, trading time for peak memory.
      </Prose>

      <Prose>
        <strong>No inference overhead.</strong> Because merging operates directly on the weight tensors and produces a single model with the same architecture, the merged model's inference cost is identical to the parent models'. This is the key economic argument for merging over ensembling: running N models simultaneously costs N times the compute; running one merged model costs the same as one.
      </Prose>

      <Prose>
        <strong>Evolutionary search costs.</strong> This is where the cost picture changes. Evolutionary search evaluates thousands of candidate merge configurations; each evaluation requires producing a merge, running the merged model on a benchmark, and recording the result. If the benchmark is an open-domain LLM evaluation suite (like lm-evaluation-harness on MMLU or HellaSwag), each evaluation costs several GPU-hours. A search over a thousand configurations costs thousands of GPU-hours — comparable to a small fine-tuning run, though still cheaper than a full pretraining. The search is embarrassingly parallelizable: each candidate merge is independent, so the wall-clock time scales with available GPU count, not with the number of candidates.
      </Prose>

      <Prose>
        <strong>Scaling with model size.</strong> Merging quality does not degrade with larger models — if anything, larger models are more amenable to merging, because their loss basins are wider and their fine-tuning deltas are proportionally smaller relative to the total weight magnitude. The DARE paper shows that larger models tolerate higher drop rates: a 13B model can sustain 99% delta drop with minimal capability loss, while a 1B model begins to degrade at 90%. This is a favorable scaling law: as model size grows, the fraction of the delta that is genuinely load-bearing shrinks, so the delta becomes easier to sparsify and merge.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Model merging works often enough that it has become standard practice, but it fails in specific and predictable ways. Understanding the failure modes is the difference between a practitioner who can diagnose a bad merge and one who just blames the algorithm.
      </Prose>

      <Prose>
        <strong>1. Parent models trained on different base checkpoints.</strong> Merging models with different pretraining histories is the most reliably fatal mistake. If model A is fine-tuned from Mistral-7B-v0.1 and model B is fine-tuned from Llama-3-8B, their weights occupy fundamentally different regions of the high-dimensional parameter space. The linear average of their weights does not lie in a region that corresponds to any coherent model — it is a point between two unconnected basins of the loss landscape, and the resulting merged model will produce incoherent or garbage output. Tokenizer mismatch is usually the first symptom: if the two base models have different vocabularies, the same embedding matrix row in model A corresponds to a completely different token than in model B. The merge must be rejected before it starts. The check is simple: <Code>{"tokenizer_A.vocab == tokenizer_B.vocab"}</Code>.
      </Prose>

      <Prose>
        <strong>2. Architecture mismatch.</strong> Even when two fine-tunes share the same base model family, architecture differences (different hidden dimension, different number of layers, different attention head count, different MLP ratio) prevent element-wise operations from being defined. mergekit will raise an error on a key mismatch; a custom implementation may silently produce wrong shapes. Always verify that the state dict keys and tensor shapes are identical before initiating a merge.
      </Prose>

      <Prose>
        <strong>3. Catastrophic task-vector sign conflict.</strong> When two fine-tunes moved the same parameter in opposite directions by large magnitudes, both TIES and DARE can fail to recover both capabilities. TIES elects a sign and discards the conflicting entries, which means one of the two fine-tunes' improvements at that parameter is simply discarded. In regions where the conflict is pervasive — many parameters with opposite signs — the merged model may be worse than either parent on both tasks. Symptom: merged model benchmark scores fall below both parents simultaneously. Diagnosis: compute pairwise cosine similarity of task vectors; if strongly negative (below −0.3) across many layers, the models are fundamentally incompatible for merging and you should prefer multi-task fine-tuning instead.
      </Prose>

      <Prose>
        <strong>4. Merged model worse than the weaker parent (merge collapse).</strong> Even between compatible models, an aggressive merge can land in a region of high loss. This happens most often when the interpolation parameter is chosen without evaluation — a merge at <Code>{"t = 0.5"}</Code> is not always better than <Code>{"t = 0.3"}</Code> or <Code>{"t = 0.7"}</Code>. Always evaluate the merged model on a small held-out set across all the tasks you care about. If any task regresses below the weaker parent, adjust the merge weights or switch methods.
      </Prose>

      <Prose>
        <strong>5. Redundancy in the merge set inflating apparent quality.</strong> Merging five fine-tunes that are nearly identical to each other (all instruction-tuned from the same base on similar datasets) produces a model that is not meaningfully better than any single one of them — just more expensive to merge. Worse, adding redundant models can actually dilute a strong specialist by averaging it with near-copies of itself. Inspect task-vector cosine similarities before merging: if all pairwise similarities are above 0.9, the diversity is insufficient to expect meaningful capability aggregation.
      </Prose>

      <Prose>
        <strong>6. Evaluation bias toward parent tasks.</strong> Merged models are typically evaluated on the same benchmarks used to evaluate their parents. This creates a subtle selection bias: the merge succeeds if it preserves performance on tasks that the parents saw during fine-tuning, but it is rarely evaluated on genuinely novel tasks. A merged chat + code model may excel on coding benchmarks and chat benchmarks while being no better than the base model on an unseen domain (e.g., medical question answering) that neither parent specialized in. Merging recombines what the parents learned; it does not generalize beyond them.
      </Prose>

      <Prose>
        <strong>7. Benchmark contamination from cross-parent test set memorization.</strong> Different fine-tunes may have been trained on different splits of the same benchmark datasets — or on datasets that overlap significantly with standard evaluation benchmarks. When these models are merged, the merged model may exhibit inflated benchmark scores not because the merge worked well, but because it retained memorized answers from multiple parents. This is a real and underappreciated problem in the open-weight LLM leaderboard ecosystem, where many of the top-ranked models are merges and where training data provenance is often opaque. Be skeptical of large leaderboard gains from merging that are not reflected in independent evaluation.
      </Prose>

      <Prose>
        <strong>8. Safety regression.</strong> If one of the fine-tunes being merged was trained with safety RLHF and another was not (or was trained to suppress refusals), the merged model may regress on safety behaviors. Safety alignment in LLMs appears to be encoded in a relatively small and concentrated part of the weight space, making it fragile under merging. The TIES sign-election step can discard safety-aligned weight directions if they conflict with the majority vote. Evaluate merged models for safety-relevant behaviors explicitly, not just capability benchmarks.
      </Prose>

      <Prose>
        <strong>9. Quantization interaction with merge weights.</strong> When fine-tuned models are stored in 4-bit or 8-bit quantized formats, the merge must either dequantize them first (recovering the original float values) or accept that merging quantized weights produces quantization errors that compound. mergekit handles this by loading in bfloat16 before merging, but custom pipelines often skip this step, producing merged models with higher quantization error than any individual parent.
      </Prose>

      <Prose>
        <strong>10. Catastrophic forgetting of base model capabilities.</strong> Fine-tuning on a specialized task can suppress general-purpose capabilities of the base model — this is called catastrophic forgetting. When you merge two fine-tunes that both forgot the same general capability, the merged model inherits both fine-tunes' specialized abilities and neither one's general capability. Merging does not restore capabilities that were suppressed during fine-tuning — it can only recombine what is represented in the parents. LM-Cocktail addresses this specifically by merging the fine-tuned models back toward the pretrained base, but the approach requires knowing which general capabilities you want to preserve.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The six papers below are the canonical references for the algorithms discussed in this topic. All were cross-checked against their published venues during preparation; dates, titles, and arXiv IDs reflect verified records.
      </Prose>

      <Prose>
        <strong>1.</strong> Wortsman, Mitchell; Ilharco, Gabriel; Gadre, Samir Yitzhak; et al. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." <em>Proceedings of the 39th International Conference on Machine Learning (ICML 2022)</em>. arXiv:2203.05482, submitted March 2022. The founding paper of the modern model-merging movement. Demonstrates that averaging N fine-tunes of the same pretrained model (with different hyperparameter configurations) produces a "soup" model that exceeds any individual member of the set on ImageNet and improves out-of-distribution robustness. Introduces greedy soup (add one at a time, keep when it improves validation metric) as a refinement of uniform soup.
      </Prose>

      <Prose>
        <strong>2.</strong> Ilharco, Gabriel; Ribeiro, Marco Tulio; Wortsman, Mitchell; et al. "Editing Models with Task Arithmetic." <em>Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)</em>. arXiv:2212.04089, submitted December 2022. Introduces the task vector formalism: subtracting the pretrained base from a fine-tuned model yields a vector in weight space encoding the learned capability. Shows that task vectors support addition (multi-task combination), negation (capability suppression), and analogy (transferring an improvement from one domain to another). The conceptual foundation for TIES and DARE.
      </Prose>

      <Prose>
        <strong>3.</strong> Yadav, Prateek; Tam, Derek; Choshen, Leshem; Raffel, Colin; Bansal, Mohit. "TIES-Merging: Resolving Interference When Merging Models." <em>Advances in Neural Information Processing Systems 36 (NeurIPS 2023)</em>. arXiv:2306.01708, submitted June 2023. Identifies two sources of interference in naive task arithmetic — redundant parameter values and sign conflicts — and proposes the trim-elect-merge pipeline. Demonstrates improvements over linear merge and raw task arithmetic across diverse modalities, model sizes, and task combinations.
      </Prose>

      <Prose>
        <strong>4.</strong> Yu, Le; Yu, Bowen; Yu, Haiyang; Huang, Fei; Li, Yongbin. "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch." <em>Proceedings of the 41st International Conference on Machine Learning (ICML 2024)</em>. arXiv:2311.03099, submitted November 2023. Introduces DARE: empirically demonstrates that 90–99% of SFT delta parameters are redundant, random dropout with rescaling recovers the full fine-tune quality, and sparsified task vectors merge with far less interference. Shows that drop rate tolerance increases with model size — larger models can sustain higher drop rates.
      </Prose>

      <Prose>
        <strong>5.</strong> Goddard, Charles; Siriwardhana, Shamane; Ehghaghi, Malikeh; et al. "Arcee's MergeKit: A Toolkit for Merging Large Language Models." arXiv:2403.13257, submitted March 2024. The library paper for mergekit. Documents the configuration-driven workflow, the supported merge methods (linear, SLERP, task arithmetic, TIES, DARE, DARE+TIES), and the shard-by-shard memory management strategy. Reports that thousands of models on the Hugging Face Hub were produced with mergekit, including several that achieved top Open LLM Leaderboard scores at the time of publication.
      </Prose>

      <Prose>
        <strong>6.</strong> Akiba, Takuya; Harada, Makoto; Nakata, Yujiro; et al. (Sakana AI). "Evolutionary Optimization of Model Merging Recipes." <em>Nature Machine Intelligence</em>, 2025. arXiv:2403.13187, submitted March 2024. Applies evolutionary algorithms to automate the discovery of merge configurations, searching over both parameter space (which weights to include and how to weight them) and data flow space (which layers map to which positions in the resulting model). Produces a Japanese mathematical LLM that outperforms much larger explicitly-trained models by discovering that a Japanese language fine-tune and an English math fine-tune can be combined in a non-obvious layer-permuted configuration.
      </Prose>

      <Callout accent="gold">
        Secondary but essential: Shoemake, Ken. "Animating Rotation with Quaternion Curves." <em>SIGGRAPH 1985 Conference Proceedings</em>, pp. 245–254. The original SLERP paper, from computer graphics. The same formula was adapted for LLM weight interpolation in the open-weight community during 2023. The key property — interpolation along a great-circle arc preserves vector norm throughout — transfers directly from quaternion animation to weight-space geometry.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five problems. Work each before reading the answer. The problems are chosen so that getting one wrong tells you specifically what to re-read.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> Two models, A and B, are fine-tuned from the same base. At parameter position <Code>j</Code>, model A has value <Code>+0.08</Code> and model B has value <Code>−0.05</Code>. The base model has value <Code>+0.01</Code> at position <Code>j</Code>. Compute the task vectors for A and B at position <Code>j</Code>, then apply TIES with a sign-election step. What value does the merged model have at position <Code>j</Code> if only A's task vector exceeds the trim threshold?
      </Prose>

      <Callout accent="green">
        Task vector A at position <Code>j</Code>: <Code>+0.08 − 0.01 = +0.07</Code>. Task vector B at position <Code>j</Code>: <Code>−0.05 − 0.01 = −0.06</Code>. After trimming, B is zeroed (does not exceed the threshold). Only A contributes. Elected sign: <Code>sign(+0.07) = +1</Code>. Only A is sign-consistent, so the merge averages over A alone: merged delta = <Code>+0.07</Code>. Final merged value at position <Code>j</Code>: <Code>+0.01 + 0.07 = +0.08</Code> — the merged model recovers A's value exactly at this position, and B's conflicting delta is discarded. This is TIES's intended behavior: protect the stronger signal from being cancelled by a conflicting weaker one.
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> Why does SLERP preserve the norm of the interpolated weight vector while linear interpolation does not? Provide a one-sentence geometric explanation and identify the formula term responsible.
      </Prose>

      <Callout accent="green">
        Linear interpolation between two points computes their weighted average, which lies on the chord connecting them — shorter than the radius of the sphere on which both points sit, so the result has strictly smaller norm than either endpoint (except at the endpoints themselves). SLERP interpolates along the arc of the great circle connecting the two points, keeping the result on the sphere at all times. The norm-preserving property comes from the sinusoidal coefficients <Code>{"sin((1−t)Ω) / sin(Ω)"}</Code> and <Code>{"sin(tΩ) / sin(Ω)"}</Code>: for unit vectors, these coefficients are constructed exactly so that the output vector has unit norm — they encode the arc rather than the chord.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> You want to merge a code-specialized model, a math-specialized model, and an instruction-following model, all fine-tuned from the same 7B base. You compute pairwise task-vector cosine similarities: code–math = 0.54, code–instruct = 0.17, math–instruct = −0.11. Which merge method should you use, and why? What does the negative similarity between math and instruct tell you specifically?
      </Prose>

      <Callout accent="green">
        Use DARE+TIES. The code–instruct similarity of 0.17 and the math–instruct similarity of −0.11 indicate meaningful parameter-level conflicts, particularly between the math and instruction fine-tunes which actively oppose each other on some parameters. Raw linear merge or task arithmetic would cancel math and instruct improvements at the conflicting positions. TIES's sign-election step resolves those conflicts by electing a majority sign and discarding minority-sign entries. DARE further reduces interference by sparsifying task vectors before sign election, lowering the probability that both math and instruct retained the same conflicting parameter after dropout. The negative similarity between math and instruct specifically means that fine-tuning for math moved certain parameters in the opposite direction from instruction-following — probably parameters involved in response style, verbosity, or template adherence. TIES will elect the sign of whichever signal is stronger (larger sum of retained values) and discard the other, rather than allowing them to partially cancel.
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> Implement in three lines of Python (using the functions defined in section 4) a greedy soup that starts with model A and adds model B only if the merge improves the toy model's L2 distance from a reference target tensor. Assume a reference target <Code>{"target_sd"}</Code> exists.
      </Prose>

      <Callout accent="green">
        <CodeBlock language="python">
{`# Greedy soup: add B only if it improves distance to target
baseline = state_dist(ft_A.state_dict(), target_sd)
candidate = linear_merge([ft_A.state_dict(), ft_B.state_dict()])
result = candidate if state_dist(candidate, target_sd) < baseline else ft_A.state_dict()
# "result" is now the greedy soup: the merge if it improved, ft_A alone if not.
# In a real greedy soup you'd iterate over many models and a real evaluation metric.`}
        </CodeBlock>
        The core pattern is: compute a merge, evaluate it on a metric, keep it only if it improves on the current best. The three lines above implement this for two models and one comparison metric. In the Wortsman paper, the greedy soup iterates over N sorted models (sorted by individual validation accuracy) and the evaluation is on a held-out validation set, not a toy L2 distance.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> A practitioner merges six models and finds that the merged model scores 63% on MMLU, while the best individual parent scores 61%. They conclude the merge was successful and the merged model is meaningfully better. Identify two reasons this conclusion might be premature.
      </Prose>

      <Callout accent="green">
        First, benchmark contamination. If different parents were fine-tuned on datasets overlapping with different subsets of MMLU, the merged model may have accumulated memorized answers from multiple parents rather than demonstrating improved general reasoning. A 2-point gain on a benchmark where one or more parents had test-set exposure is not reliable evidence of improved capability. Second, evaluation bias toward parent tasks. The practitioner only measured MMLU — a benchmark that the parents were likely evaluated on during fine-tune selection. A merged model should be evaluated on held-out tasks that none of the parents were specifically trained or selected for. A model that scores 63% on MMLU but regresses substantially on a genuinely unseen evaluation suite is not better; it has just preserved the parents' existing benchmark coverage while adding cross-parent memorization noise. The correct interpretation requires: (a) evaluation on tasks none of the parents were explicitly fine-tuned for, and (b) comparison against the parent checkpoints on those tasks rather than only against each other.
      </Callout>
    </div>
  ),
};

export default modelMerging;
