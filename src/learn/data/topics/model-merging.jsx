import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const modelMerging = {
  title: "Model Merging (TIES, DARE, Model Soups, SLERP)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Open-weight LLMs come in variants. A base model fine-tuned for chat. The same base model fine-tuned for code. A third copy RL-tuned for reasoning. For a while each was served separately — a 70B-per-variant infrastructure bill, three deployments doing the work one might handle. In 2023–24 it became clear that you can often literally average the weights of similar models and recover most of each variant's capabilities in a single set of weights. No retraining. No additional data. Just arithmetic on parameter files. Model merging is now a surprisingly important technique, especially in the open-weight ecosystem where serving cost is the binding constraint and compute-free weight surgery has real value.
      </Prose>

      <H2>Why this works at all</H2>

      <Prose>
        Two fine-tunes of the same base model live near each other in weight space. Both departed from the same starting point — the pretrained base — and traveled in different directions during fine-tuning: one toward chat behavior, one toward code behavior. The base is a shared reference frame. Averaging them, or doing more sophisticated interpolation, lands somewhere between the two endpoints, inheriting behavior from both parents because both parents were initialized to the same place.
      </Prose>

      <Prose>
        The surprising empirical finding is that merged models often match or exceed their parents on held-out evaluations without any additional training. Whether this "should" work is genuinely an open theoretical question. The loss landscape in large transformer weight space appears to have broad, well-connected basins — fine-tunes of the same base tend to end up in the same basin, which means averaging stays inside it rather than crossing a ridge into high-loss territory. That is a post-hoc rationalization, not a proof. The prediction that it would work came not from theory but from empirical observation that it does, and the field has proceeded from there.
      </Prose>

      <H2>Simple weight averaging — Model Soups</H2>

      <Prose>
        The simplest technique comes from Wortsman and collaborators (2022). Given <Code>N</Code> models fine-tuned from the same base, average their weights element-wise. No task vectors, no sparsification — just a weighted mean of matching parameter tensors. The conditions for it to work are strict but not unusual: all models must share the same architecture and vocabulary, and all must be fine-tuned from the same base checkpoint. When those conditions hold, the average is often better than any individual member of the ensemble, and always cheaper than serving all <Code>N</Code> models separately.
      </Prose>

      <CodeBlock language="python">
{`import torch

def model_soup(state_dicts, weights=None):
    """Average a list of state dicts, optionally with different weights per model."""
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    merged = {}
    for key in state_dicts[0].keys():
        merged[key] = sum(
            w * sd[key] for w, sd in zip(weights, state_dicts)
        )
    return merged

# Works surprisingly well when all parents share the same base.
# Breaks when parents diverged too far (different pre-training, incompatible tokenizers).`}
      </CodeBlock>

      <Prose>
        Greedy soup is a refinement: instead of averaging all models blindly, add them one at a time and keep the update only when it improves a validation metric. This requires a small held-out evaluation set but avoids the case where a low-quality parent drags the average down. Uniform soup is cheaper and often nearly as good when all parents are reasonable fine-tunes of the same base. The failure mode is merging models whose fine-tuning pulled them far enough apart that the average falls into a region of high loss — visually, the average position lands between two basins rather than inside one.
      </Prose>

      <H3>SLERP — smoother interpolation</H3>

      <Prose>
        Linear interpolation between two weight vectors has a geometric quirk: the midpoint of two high-magnitude vectors is pulled toward the origin, because averaging reduces magnitude. When the two parents differ meaningfully in the scale of their weights, the merged model is systematically smaller in norm than either parent, which can deflate activations in a way that hurts performance. Spherical Linear Interpolation (SLERP) addresses this by interpolating along the arc of the unit hypersphere instead of the chord between two points.
      </Prose>

      <MathBlock>{"\\text{slerp}(W_A, W_B, t) = \\frac{\\sin((1-t)\\theta)}{\\sin\\theta} W_A + \\frac{\\sin(t\\theta)}{\\sin\\theta} W_B"}</MathBlock>

      <Prose>
        Here <Code>θ</Code> is the angle between the two weight vectors and <Code>t ∈ [0, 1]</Code> is the interpolation parameter. At <Code>t = 0</Code> the result is <Code>W_A</Code>; at <Code>t = 1</Code>, <Code>W_B</Code>; at <Code>t = 0.5</Code>, the point on the arc equidistant from both. The norm is preserved throughout — the interpolated vector has the same magnitude as the endpoints, not the smaller magnitude of a linear average. SLERP became popular in the open-weight community during the early Llama-2-derived merge wave; many influential merges on HuggingFace in late 2023 and early 2024 used it as their primary interpolation method.
      </Prose>

      <H2>Task arithmetic and TIES</H2>

      <Prose>
        A subtler approach separates the question of what each fine-tune learned from the question of how to combine what they learned. Define the task vector of a fine-tuned model as the element-wise difference between fine-tuned weights and base weights: <Code>τ = θ_ft − θ_base</Code>. The task vector encodes only the changes induced by fine-tuning, stripped of the shared base. Ilharco and collaborators (2022) showed that task vectors compose: adding two task vectors to a base model gives a model with both capabilities; subtracting a task vector removes a capability. You can add the "reasoning" direction and subtract the "verbosity" direction in the same operation.
      </Prose>

      <Prose>
        Task arithmetic works but produces noisy merges when task vectors conflict. The conflict is structural: two fine-tunes may have moved a parameter in opposite directions, and simply adding their deltas results in partial cancellation. TIES (TRIM, ELECT, SIGN & MERGE — Yadav et al. 2023) is a three-step refinement. First, trim: zero out small-magnitude entries in each task vector, treating them as fine-tuning noise rather than signal. Second, elect: for each parameter, resolve sign conflicts by taking a majority vote across the task vectors being merged. Third, merge: average only the surviving, sign-consistent entries. The procedure is more expensive than raw addition but produces measurably cleaner merges, especially when combining three or more task vectors simultaneously. Conflict resolution is the part that matters most — allowing opposite-sign deltas to average out is the primary source of degradation in naive task arithmetic.
      </Prose>

      <H3>DARE — aggressive sparsification</H3>

      <Prose>
        DARE (Yu et al. 2023) pushes the sparsification idea further. The hypothesis: only a small fraction of the weight changes in a typical fine-tune are load-bearing. Most of the delta between base and fine-tuned model is noise — small-magnitude adjustments that the model made incidentally during gradient descent and that contribute nothing to the downstream behavior. If most deltas are noise, you can drop them randomly, rescale the survivors to preserve expected magnitude, and recover nearly identical fine-tune quality from a much sparser task vector.
      </Prose>

      <CodeBlock language="python">
{`def dare_task_vector(base_weights, fine_tuned_weights, drop_rate=0.9):
    """DARE: randomly drop most of the delta, rescale the rest."""
    delta = {k: fine_tuned_weights[k] - base_weights[k] for k in base_weights}
    kept = {}
    for k, d in delta.items():
        mask = (torch.rand_like(d) > drop_rate).float()
        kept[k] = d * mask / (1 - drop_rate)   # rescale survivors
    return kept  # add kept to base + other DARE task vectors for merging`}
      </CodeBlock>

      <Prose>
        With a drop rate of 0.9, only 10% of delta parameters survive. The rescaling by <Code>1 / (1 − p)</Code> restores the expected magnitude of the task vector — the same trick dropout uses during training. The resulting sparse task vector, added back to the base, recovers nearly all of the fine-tune's quality. The payoff in merging: when two sparse task vectors are combined, the probability that both retained the same parameter is only 1% (0.1 × 0.1), so interference between the two fine-tunes is drastically reduced compared to merging dense task vectors. DARE is often combined with TIES — sparsify first, then resolve remaining sign conflicts — for cleaner multi-model merges.
      </Prose>

      <H2>mergekit and the practical ecosystem</H2>

      <Prose>
        Arcee's mergekit library (2023) standardized model-merging recipes into a configuration-driven workflow. A <Code>mergekit.yaml</Code> file describes what to merge and how — which base model, which fine-tunes, which merge method (linear, slerp, ties, dare, ties+dare), and any per-layer or per-module overrides. The library handles the mechanics: loading sharded checkpoints, applying the specified algorithm, and saving the merged weights in the standard format. The result is that merge iteration happens at the config level, not the code level. A practitioner can test ten merge recipes in the time it would take to write one fine-tune training script.
      </Prose>

      <Prose>
        The downstream effect on the open-weight ecosystem has been substantial. HuggingFace now hosts thousands of merged model variants. Many models that appeared in leaderboard rankings in 2024 and 2025 under names like "MegaMerge-70B" or "FusionLlama" were merge artifacts rather than pure fine-tunes. The ability to iterate without retraining accelerated the pace of capability combination in the open-weight community to a degree that would have seemed implausible two years earlier. For hobbyists and researchers without large GPU budgets, merging is often the primary method for producing models with bespoke capability profiles — combine a reasoning-tuned model with a coding-tuned model, dial the interpolation parameter by hand on a small eval set, and publish the result.
      </Prose>

      <H3>The limits</H3>

      <Prose>
        Merging has a quality ceiling that becomes visible quickly. Two mediocre parents merge into a mediocre result; bad directions in weight space compose as readily as good ones. A merge of two models that both hallucinate on factual questions will produce a model that hallucinate on factual questions. Merging cannot add capabilities that are absent from both parents — if neither model can write correct SQL, the merge of the two cannot either. The technique recombines what is latent in the parents; it does not generalize beyond them. And the shared-base assumption is not optional. Merging models trained on different base checkpoints — Llama 2 with Mistral, or Mistral with Qwen — rarely works cleanly, because the weight spaces are not aligned. Different pretraining data, different tokenizers, and different architectural choices mean the basins those models inhabit are not the same basin.
      </Prose>

      <Callout accent="gold">
        Model merging is a way to recombine capabilities, not to invent them. What comes out is a blend of what went in, subject to cancellation — and cancellation is the quiet killer.
      </Callout>

      <Prose>
        Model merging is arguably the cheapest post-training modification available: no GPU training, no dataset curation, hours not weeks of iteration. For open-weight deployments where serving one model is much cheaper than serving many, it is genuinely valuable. The technique sits at a useful intersection — it requires understanding of fine-tuning, weight space geometry, and the capability profiles of the models being combined, but it requires none of the infrastructure or expense of a training run. The next topic in this section returns to the retrieval side of long-context work: hybrid search, which combines dense vector retrieval with sparse keyword matching to improve the quality of what gets retrieved before generation begins.
      </Prose>
    </div>
  ),
};

export default modelMerging;
