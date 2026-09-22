# Mixture-of-Experts Transformers (MoE)

**Explore as you read.** Edit token/router scores, expert values, capacity, grouping and budget dimensions; manipulate supported retained images. Show selected experts, discarded probability mass, overflow/drop routes, recombined outputs, balance terms and active/total resource counts together. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose routing/capacity policies from missing contributions and resource tradeoffs; auxiliary balance does not establish task quality.


Imagine having several useful ways to transform a piece of information, while paying to run only the few that are useful for this particular input. A mixture-of-experts layer makes that choice inside a neural network. It keeps several small networks, gives each input a score for each network, executes the selected ones, and combines their answers.

This separates two resources that usually grow together: **how many parameters the model stores** and **how much expert computation one token uses**. The separation creates an opportunity, and also a practical problem: inputs must reach the right parameters, enough inputs must reach each expert to train it, and their results must return to the right place.

**First pass:** follow §§1–5, the worked cost example in §6, and the actual image experiment in §8; then try practice 1–6. You should be able to trace a token through routing, explain its output and identify where computation is saved. **Deeper pass:** return to router derivatives, alternative assignments, distributed execution and upcycling in §§3, 6–7, 9; finish the remaining practice. The optional full program makes the small experiment reproducible.

The preceding [Vision Transformers](/learn/path/full-curriculum/vision-transformers-vit-deit-swin-dinov2?module=deep-learning-fundamentals) lesson explains turning an image into patch tokens. We use that representation here, but replace a different part of the block: its feed-forward transformation. Review [Transformer Block Architecture](/learn/path/full-curriculum/transformer-block-architecture?module=deep-learning-fundamentals) if residual connections or normalization are unfamiliar; their roles are refreshed below.

## 1. An expert is a network, not a named profession

A feed-forward network, or FFN, transforms a token's feature vector without directly reading other token positions. Attention may already have put contextual information into that vector. Thus an FFN can act on context, even though its own operation is positionwise.

An MoE FFN supplies several alternatives:

\[
x\in\mathbb R^d
\quad\longrightarrow\quad
E_0(x),E_1(x),\ldots,E_{N-1}(x)\in\mathbb R^d.
\]

Each expert has its own learned parameters. Equal architecture does not mean equal weights. A **router** scores these alternatives; a **gate** determines the weights used to combine them. Some writing uses these two names interchangeably, so always inspect both the selection rule and the mixture formula.

A pre-normalized Transformer block can use

\[
u=x+\operatorname{Attention}(\operatorname{LN}(x)),\qquad
x_{\text{next}}=u+\operatorname{MoE}(\operatorname{LN}(u)).
\]

Layer normalization rescales features within a token. The residual additions preserve a direct path around each transformation. Replacing the FFN does not remove attention, change its causal mask, or eliminate its cache.

**Figure 1 — Where the choice happens.** Follow three token lanes through shared attention. At the FFN stage, each lane branches to two highlighted experts and rejoins its original lane. Draw attention edges across tokens separately from expert-assignment edges. Label experts 0–3 rather than “grammar,” “mathematics,” or “vision.”

The label *expert* is historical terminology. Joint training may produce useful differences without giving any branch a clean human interpretation. A routing heatmap shows assignments; proving specialization requires an additional question, such as how an intervention changes predictions on a defined group of inputs. Mixtral's published routing analysis found broadly similar expert distributions across several subject collections, rather than an obvious expert for each subject. [Mixtral, §5](https://arxiv.org/html/2401.04088v1#S5)

This also differs from averaging several independently trained complete models. A typical Transformer MoE routes within many layers, shares much of the surrounding network, and trains router and experts together.

## 2. Score, select, weight, combine

Let a bias-free router produce \(h=W_rx\), where \(W_r\) has shape \(N\times d\). Each component is a real-valued **logit**, a score rather than a probability. A softmax turns scores into positive weights:

\[
p_i=\frac{\exp(h_i)}{\sum_{j=0}^{N-1}\exp(h_j)}.
\]

Top-\(k\) selects the indices of the \(k\) largest scores. The selected index set is \(S(x)\). Selecting by softmax probabilities gives the same ranking as selecting by logits, provided the same tie rule is used.

There are two important combination conventions.

| Convention | Selected expert weight | Sum of selected weights |
| --- | --- | --- |
| Keep full-softmax probabilities | \(g_i=p_i\) for \(i\in S\) | Generally less than 1 when \(k<N\) |
| Normalize among selected experts | \(g_i=\exp(h_i)/\sum_{j\in S}\exp(h_j)\) | Exactly 1 |

Both produce \(y=\sum_{i\in S}g_iE_i(x)\). They are different functions. Naming a layer “top-2 MoE” does not specify which one it computes.

### Follow one actual arithmetic example

Use three expert outputs \(E_0=[2,0]\), \(E_1=[0,3]\), \(E_2=[-1,1]\). Let logits be \(\log[4,2,1]\), meaning \([\log4,\log2,\log1]\). Their full-softmax probabilities are \([4/7,2/7,1/7]\). Select experts 0 and 1.

With selected normalization, their weights are \(2/3\) and \(1/3\):

\[
y=\frac23[2,0]+\frac13[0,3]=[4/3,1].
\]

With full-softmax weights, the same selected experts yield

\[
y_{\text{full}}=\frac47[2,0]+\frac27[0,3]=[8/7,6/7].
\]

Expert 2 is not executed in either case. Its score still affects the full-softmax denominator. Under selected normalization, changing that unselected score has no effect until it crosses the selection boundary.

**Figure 2 — Two normalizers, two answers.** Align the same three score bars, selected indices, weight bars and two-dimensional output arrows. Cross out computation for expert 2, while preserving its arrow into the full-softmax denominator. That remaining arrow is the reason the two cases differ.

A useful algebra check is

\[
\frac{p_i}{\sum_{j\in S}p_j}
=\frac{\exp(h_i)}{\sum_{j\in S}\exp(h_j)}.
\]

Computing full softmax, selecting and renormalizing is mathematically the same selected-normalized operator, including derivatives away from selection boundaries. It does not merely share the same final training optimum. Floating-point implementations may differ slightly.

**Investigation 1 — Change the route, then explain the reunion.** Start with a new four-expert problem. Edit a score or expert output, inspect which paths and output coordinates change, and show immediately the computed contributions. Compare selected normalization with full-softmax weighting. A separate top-1 choice tests whether selecting one branch necessarily prevents the router from learning.

## 3. How a discrete choice learns

Backpropagation cannot ordinarily differentiate the integer identity returned by top-\(k\). But it can differentiate the continuous weights along the currently selected paths. The resulting function is smooth within regions where the selected set stays fixed, with possible discontinuities or derivative changes at route boundaries.

For selected normalization, let \(r=\partial L/\partial y\) be the vector describing how the task loss changes with the mixture output. Holding the selected set fixed,

\[
\frac{\partial L}{\partial h_i}
=g_i\,r^T(E_i-y),\qquad i\in S.
\]

For an unselected logit, this task derivative is zero. To see where the formula comes from, differentiate a selected softmax weight: \(\partial g_\ell/\partial h_i=g_\ell(\mathbf1[\ell=i]-g_i)\). Summing the weighted expert derivatives gives \(\partial y/\partial h_i=g_iE_i-g_i\sum_\ell g_\ell E_\ell=g_i(E_i-y)\). The chain rule then takes its dot product with \(r\). The formula compares an expert's output with the current weighted result, along the direction that matters to the loss. Merely being selected does not guarantee a nonzero derivative.

For example, use the worked outputs and the scalar diagnostic \(L=y_1+y_2\), where these subscripts denote the two coordinates. Experts 0 and 1 have output sums 2 and 3. The score derivatives are \([-2/9,2/9,0]\): increasing the weight on the first reduces this particular scalar. This diagnostic illustrates calculus; it is not a classification objective.

With one selected expert and selected normalization, \(g=1\), so \(y=E_i\) and its task gradient to router scores is zero inside a selection region. This is why normalization matters to the learning mechanism.

Switch uses a different top-1 rule: the selected expert retains its full-softmax probability. Then \(y=p_iE_i\), and changing a score changes the output scale even if the identity remains fixed. For full-softmax masked routing,

\[
\frac{\partial L}{\partial h_j}
=p_j\,r^T\left(\mathbf1[j\in S]E_j-y\right).
\]

An unselected expert's **network** is still not executed, but its **router logit** can receive a task derivative through normalization. [Switch Transformers, §2.1](https://arxiv.org/html/2101.03961v3#S2.SS1)

An expert receives task gradients only from retained assignments that actually use its output. Empty or dropped branches have no such contribution on that step. Whether an optimizer changes an unused parameter also depends on implementation details such as absent versus zero gradients, stored momentum and weight decay. “No task gradient” is a more precise statement than “this weight can never change.”

**Figure 3 — Two backward paths.** Send loss arrows to executed expert parameters and selected mixture weights. Split the router view into selected-normalized and full-softmax cases. An unselected full-softmax score can have a backward arrow even though its expert has no forward execution arrow.

Noise is one way to encourage different selections during training. In the original sparsely gated layer, learned-scale Gaussian noise perturbs scores before top-\(k\). That is one design, not an essential property of every MoE. It also makes it necessary to distinguish noisy training routes from deterministic evaluation routes. [Shazeer et al., §2.1](https://arxiv.org/html/1701.06538v1#S2.SS1)

## 4. Turn selected paths into real computation

Writing a sparse weighted sum is not enough to save work. If code evaluates every expert for every token and multiplies unwanted outputs by zero afterward, it computes a dense reference, not a sparse execution.

A practical sparse path has four steps:

1. **Dispatch:** gather the token vectors assigned to each expert.
2. **Execute:** run each expert on its gathered mini-batch.
3. **Weight:** multiply each returned vector by its selected gate weight.
4. **Combine:** add contributions back into their original token positions.

Keep a token ID and a selected-slot ID. A token routed to two experts must receive both contributions. An indexed assignment that overwrites an earlier contribution silently changes the model.

**Figure 4 — A permutation with a return address.** Five token cards carry their original indices into expert bins. Expert outputs return along arrows labelled with gate weights. Repeated destination IDs meet at an addition sign, not an overwrite symbol. This is the visual explanation of gather and scatter-add.

The full program in §8 performs precisely this operation. A separately computed reference evaluates all four experts and uses a dense gate mask. On the saved models, both outputs and all parameter gradients agree to floating-point precision. The expensive reference is useful for establishing what the sparse code computes.

### Capacity is a policy about assignments

Different experts can receive very different loads. A fixed-capacity implementation reserves \(C\) assignment slots per expert. One common top-\(k\) convention is

\[
C=\left\lceil c\frac{Tk}{N}\right\rceil,
\]

where \(T\) counts tokens in this routing group, \(k\) is assignments per token and \(c\) is a capacity factor. Papers and systems use different conventions, so check whether their formula already includes \(k\).

Consider four tokens, three experts and two assignments per token:

\[
[0,1],\ [0,1],\ [0,2],\ [0,2].
\]

There are eight assignments, with expert loads \([4,2,2]\). At capacity two, expert 0 rejects two assignments. Those two tokens still have their expert-2 routes. **Two dropped assignments do not mean two tokens lost all expert computation.**

If every route for a token is dropped, its MoE contribution is zero in a drop-without-reweighting design. Its residual path still carries the token representation. Dropping one route, renormalizing the survivors, rerouting, padding buffers, and allocating variable-size expert batches are different policies with different outputs and costs.

**Figure 5 — Overflow without losing the whole token.** Show the four token rows and three expert bins. Pattern the two rejected arrows, keep the surviving arrows visible, and put the residual path outside the bins. Count assignments, affected tokens and fully dropped tokens in separate places.

Capacity competition can also make an output depend on batch companions or assignment order. A first-come policy may keep a route when a request runs alone and drop it when another request fills the bin. A causal decoder needs care if future positions influence which earlier assignments survive, especially with global priority selection. A causal attention mask alone does not repair a separate noncausal routing operation.

**Investigation 2 — Fill the bins.** A fresh five-token problem asks you to predict overflow and token outputs at different capacities. Reorder dispatch while keeping token identities fixed. Then switch to dropless dispatch and explain which batch effects disappear.

Dropless systems avoid discarding assignments; they still need memory management and efficient kernels for varying expert loads. MegaBlocks formulates the work with block-sparse operations to avoid the forced choice between dropping routes and padding every expert to a large uniform size. Its reported speed comparisons belong to its measured setup, not every MoE. [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf)

## 5. Train useful routes, not just attractive load bars

If a router repeatedly sends inputs to the same few experts, those experts get more practice. Underused branches may remain weak, reinforcing the initial preference. Meanwhile, overloaded devices can delay the rest of the computation. A balancing objective tries to counter those dynamics while the task objective learns useful predictions.

Define, for one routing group,

\[
q_i=\frac{\text{number of selected assignments to expert }i}{Tk},
\qquad
P_i=\frac1T\sum_t p_{t,i}.
\]

Both vectors sum to one. A commonly used Switch-style extension to top-\(k\) is

\[
L_{\mathrm{bal}}=N\sum_i \operatorname{stopgrad}(q_i)P_i.
\]

The count is treated as fixed for backpropagation; the mean soft probabilities supply the derivative. The uniform assignment/probability case gives 1. If instead counts are divided only by \(T\), their sum is \(k\) and the uniform reference becomes \(k\). These conventions change the numerical loss scale; a printed value of 2 is not automatically evidence of severe imbalance.

Three distinct diagnostics help:

| Quantity | Question |
| --- | --- |
| Assignment counts | How much discrete work reached each expert? |
| Mean soft scores or gate mass | Where did the router place weight? |
| Task loss and intervention effects | Was the resulting computation useful? |

The original sparsely gated work also distinguished importance, based on gate mass, from load, based on how many examples reach an expert. Its coefficient-of-variation penalties are not identical to this bilinear surrogate. [Shazeer et al., §4 and Appendix A](https://arxiv.org/html/1701.06538v1#S4)

### A useful surprise: the balance number is not a distance

For two experts and four top-1 inputs, let three probability rows be \([.51,.49]\) and the fourth be \([.001,.999]\). Then

\[
q=[.75,.25],\quad P=[.38275,.61725],\quad
L_{\mathrm{bal}}=2(.75\cdot.38275+.25\cdot.61725)=.88275.
\]

That is below the uniform reference 1 even though the assignments are unequal. The loss is a training surrogate, not a nonnegative divergence from uniformity with a guaranteed minimum of 1. Inspect actual counts instead of turning this scalar into a “percent balanced” gauge.

**Figure 6 — Counts and probabilities can disagree.** Show all four probability rows, their top-1 arrows, the count histogram and the mean-probability histogram. The two aggregate histograms lean in opposite directions. Keep the explicit dot product underneath.

The group over which balancing is computed matters. A product of microbatch means is generally different from the mean of microbatch products. Balancing each sequence tightly can also restrict useful differences between sequences. Record whether counts and probabilities are per sequence, local group, device or global batch, and how they are aggregated across layers. There is no one scope that every design must use.

### Stabilize logits without confusing stability with balance

ST-MoE introduces a **router z-loss**:

\[
L_z=\frac1T\sum_t\left(\log\sum_i e^{h_{t,i}}\right)^2.
\]

It acts on the log-normalizer, while the task loss and balancing loss address different goals. Compute log-sum-exp stably by subtracting the row maximum before exponentiation and adding it back afterward. Selective higher precision for routing can help numerical behavior, but does not replace an evaluation of the complete training system. [ST-MoE, §3.3–3.4](https://arxiv.org/html/2202.08906v2#S3.SS3)

Adding the same constant \(a\) to every logit leaves softmax probabilities unchanged, while the log-normalizer increases by \(a\). For logits \(\log[.99,.01]\), the z-loss is zero despite highly concentrated routing probabilities. Add 3 to both logits and the probabilities remain \([.99,.01]\), while the z-loss becomes 9. Therefore this penalty is neither entropy maximization nor a promise of equal expert use.

Floating-point formats also differ. BF16 has a broad exponent range resembling FP32, while FP16's range is much narrower. Treating every 16-bit format as if exponentiation overflows at the same input is incorrect. Stable formulas, parameter initialization, optimizer behavior and precision choices all contribute to numerical reliability; a finite small-model run cannot establish large-model stability.

**Investigation 3 — Separate three objectives.** Edit probability rows, assignment preferences and a shared logit offset. Inspect which of selected routes, count balance and z-loss changes, and explain whether that determines task quality. Compare a peaked distribution with zero z-loss and an uneven count distribution with balance loss below 1.

## 6. Count parameters, arithmetic, memory and communication separately

For a bias-free SwiGLU expert with input/output width \(d\) and intermediate width \(m\),

\[
E(x)=W_{\mathrm{down}}\left(
\operatorname{SiLU}(W_{\mathrm{gate}}x)\odot
W_{\mathrm{value}}x\right).
\]

Here \(\operatorname{SiLU}(z)=z/(1+e^{-z})\) acts on each component, and \(\odot\) multiplies corresponding components. One projected branch therefore modulates another before the final projection. There are three matrices: two \(m\times d\) projections and one \(d\times m\) projection. Each expert has \(3dm\) parameters. Evaluating one token requires \(3dm\) matrix multiply-accumulate operations, or MACs, plus elementwise work. Counting one multiplication and one addition separately gives approximately two FLOPs per MAC.

An \(N\)-expert layer stores \(3Ndm\) expert parameters and executes \(3kdm\) expert MACs per token. Its dense linear router adds \(dN\) parameters and roughly \(dN\) MACs, plus selection and normalization. Attention, normalization, residuals, embeddings and the output head remain outside this count.

Take \(d=64,m=128,N=8,k=2\):

- One expert stores 24,576 parameters.
- All eight experts store 196,608 parameters.
- One token executes 49,152 expert MACs.
- The router stores 512 parameters and adds 512 score MACs.

The layer therefore stores four times as many **expert** parameters as it actively uses per token. That does not imply a fourfold whole-model speedup, or a fourfold reduction in the memory needed to load the model.

At two bytes per expert parameter, these expert weights occupy 393,216 bytes, or 0.375 MiB. The router and all other tensors are additional. Training also needs gradients, optimizer states and saved or recomputed activations, depending on the optimizer and sharding strategy. Inference must make every potentially selected expert available, through device memory, distribution or offloading. Active parameters alone do not describe that storage requirement.

**Figure 7 — Two rulers beside the same layer.** One ruler counts all stored expert matrices; the other highlights only the two executed branches. A separate strip lists router and shared-block costs. Labels explicitly distinguish parameters, MACs and bytes.

### Finer experts change the trade-off again

Now replace eight width-128 experts with sixteen width-64 experts, and select four. Total expert parameters and active expert MACs are unchanged:

\[
8(3d128)=16(3d64),\qquad
2(3d128)=4(3d64).
\]

But the router has twice as many scores and each token travels to twice as many expert branches. Finer granularity creates different possible combinations; it is not automatically free.

Adding an always-executed shared expert contributes its own parameters and work. It provides a common computational path, but nothing in the formula guarantees that it learns only “common knowledge.”

### Where the vectors travel

In **expert parallelism**, different devices own different expert parameters. Dispatch sends selected token vectors to their owners; combine returns the outputs. Under a simple model where every selected assignment crosses a device boundary, a forward pass moves approximately

\[
2Tkd\,b
\]

payload bytes: one \(d\)-component input and one \(d\)-component output per assignment, with \(b\) bytes per component. For 32 tokens, two routes, width 64 and two-byte components, that is 16,384 bytes. Metadata, protocol overhead, backward traffic and topology are excluded.

The count depends on active routes, not directly on \(N\). Some routes are local. Several experts on one node can share a transmitted input under an optimized dispatcher. Adding more experts may change placement and traffic patterns even if \(k\) stays fixed.

Data parallelism replicates a model over different examples; tensor parallelism splits a matrix operation; pipeline parallelism splits layers; expert parallelism splits branches. They can be combined. Eight experts do not necessarily require eight GPUs.

Small expert batches, uneven loads, collective startup, matrix dimensions, network bandwidth, memory bandwidth and kernel fusion all influence real latency. Prefill supplies many token assignments at once; decoding may supply relatively few, making occupancy and scheduling especially important. A model with less arithmetic can still be slower in a particular deployment.

**Investigation 4 — Keep one budget fixed and move another.** Change expert count, width, selected count and remote-route fraction. Inspect which resource moves. Compare coarse and fine expert sets with equal active matrix work, then inspect the different router and communication costs. The output is calculated accounting, not a synthetic timing benchmark.

For production execution, [MegaBlocks' implementation](https://github.com/databricks/megablocks) supplies concrete block-sparse and grouped-GEMM pathways. Use an actual backend and benchmark representative batch/sequence shapes before making a hardware choice.

## 7. Several ways to decide who does the work

### Token choice, expert choice and balanced assignment

So far, each token chooses a fixed number of experts. An alternative reverses the selection: each expert chooses a fixed number of tokens from a routing group. A score matrix makes the difference visible.

**Figure 8 — Select across rows or down columns.** In token choice, highlight the largest \(k\) entries in each token row. In expert choice, highlight the largest \(C\) entries in each expert column. Equal column counts can coexist with tokens receiving zero, one or several experts.

Expert choice controls expert batch sizes directly, while token computation becomes variable. A global assignment method instead solves a constrained matching problem, potentially constraining both sides. These methods change the allocation rule; they are not just different regularization coefficients. Autoregressive use must also respect which tokens are available when the choice is made. [Expert Choice author explanation](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)

### Separate selection bias from mixture weight

A useful modern distinction is to keep a load-control bias separate from the weight placed on the selected output. Suppose base affinities are

\[
s=[.8,.7,.6,.1],\qquad b=[0,0,.3,0].
\]

Top-2 of \(s+b\) selects experts 2 and 0. Normalize their **original** affinities, .6 and .8, giving weights \(3/7\) and \(4/7\). Normalizing the biased scores .9 and .8 would compute a different model.

DeepSeek-V3 uses sigmoid affinities, fine-grained routed experts and shared experts. Its load-control biases affect selection, while selected original affinities determine mixture weights. The report also retains a small sequence-wise auxiliary balance term despite the name “auxiliary-loss-free” for its main balancing strategy. [DeepSeek-V3, §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)

The bias can be adjusted after a training step: decrease preference for overloaded experts and increase it for underloaded experts. That update is a control rule based on counts, distinct from differentiating the task loss through selected gate weights. Deployment may fix or manage such routing state differently. A reproduction needs the exact scoring, grouping, scaling and update convention, not merely a model name.

The report also limits the nodes a token may visit to fit its communication design. This is a restriction on feasible assignments, so the independently highest expert scores may not all be eligible. Keep this architecture-specific routing constraint distinct from the unconstrained formula taught in §2.

Historical examples illuminate different decisions: Switch simplifies to one selected branch with a retained probability; Mixtral uses two selected FFNs; expert-choice models control expert occupancy; fine-grained/shared designs change the branch decomposition. Their papers report different data, objectives and hardware. They do not form a universal ranking.

## 8. A real model: which experts process image patches?

A tiny model lets us see routing on actual inputs and inspect every returned vector. Our data come from **Optical Recognition of Handwritten Digits**, collected by E. Alpaydin and C. Kaynak and hosted by UCI under CC BY 4.0. Each record is an 8×8 grid of counts from 0 to 16. These are aggregated bitmap features, not a pen trajectory. [UCI data and attribution](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits)

The experiment uses 500 fitting images, 150 validation images and 300 assessment images, balanced across digits 0–9. Fit and validation come from the original training file; assessment comes from its separate-writer test file. Internal fit/validation writer identities are unavailable. The supplied [CSV](optical-digits.csv) preserves original file and row identifiers; the [provenance](data-provenance.md) records selection and rights. No identical feature vector crosses these roles.

We divide counts by 16, cut each image into sixteen nonoverlapping 2×2 patches and project each patch to 16 features. Learned position vectors preserve patch location. One two-head attention layer allows patches to exchange information. The FFN is either a dense SwiGLU width 32 or four width-16 experts with top-2 selected normalization. The block uses residuals and LayerNorm; normalized patch outputs are averaged before a ten-class linear classifier.

**Figure 9 — Pixels become assignments.** Connect the observed 8×8 grid to its sixteen numbered patches, the attention block, each patch's two expert paths, and the final class scores. Route colors have expert indices and patterns, without giving them semantic names.

The dense FFN and the two active sparse experts have equal matrix MACs per patch. The entire models differ: dense has 3,162 parameters; MoE has 4,762, including its router. The small Python dispatcher prioritizes clarity. This comparison makes no GPU speed claim.

We declared four conditions in advance: dense, and MoE balance coefficients 0, .01 and .1. Every condition uses seeds 17, 41 and 73; each run receives 240 AdamW updates, learning rate .003, weight decay .001, batch size 64 and gradient clipping at norm 1. Validation task cross-entropy selects the checkpoint. The balancing coefficient does not enter checkpoint selection. All twelve outcomes are retained.

| Condition | Seed | Selected update | Assessment CE | Correct / 300 | Lower half zero: correct / 300 |
| --- | --- | --- | --- | --- | --- |
| Dense | 17 | 240 | .51522 | 257 | 94 |
| Dense | 41 | 220 | .68748 | 238 | 92 |
| Dense | 73 | 200 | .74796 | 227 | 120 |
| MoE, balance 0 | 17 | 220 | .67965 | 239 | 36 |
| MoE, balance 0 | 41 | 220 | .53185 | 247 | 103 |
| MoE, balance 0 | 73 | 240 | .62343 | 250 | 62 |
| MoE, balance .01 | 17 | 240 | .64471 | 236 | 61 |
| MoE, balance .01 | 41 | 220 | .50564 | 252 | 87 |
| MoE, balance .01 | 73 | 240 | .65630 | 246 | 62 |
| MoE, balance .1 | 17 | 240 | .56791 | 245 | 68 |
| MoE, balance .1 | 41 | 220 | .54378 | 247 | 96 |
| MoE, balance .1 | 73 | 240 | .70179 | 238 | 76 |

Cross-entropy is the average negative log-probability assigned to the correct class, using natural logarithms; lower is better. A uniform ten-class predictor has CE \(\log10\approx2.30259\) and expected accuracy 10%. The last column is a declared stress input: set the lower four rows to zero, keeping the original class label as the scoring target. It is not another set of naturally observed images.

Seed-to-seed variation is visible, and no balancing coefficient consistently improves every result. The most balanced-looking histogram is not automatically the most accurate classifier. These small runs investigate a mechanism under a fixed budget; they do not estimate a large language model's scaling advantage.

For example, assessment assignment counts for MoE .01, seed 17 are \([2277,2266,2731,2326]\). They sum to \(300\times16\times2=9600\). The same run gets 236/300 correct. MoE without a balance penalty, seed 17, gives \([2128,2342,2637,2493]\) and 239/300 correct. The counts are neither a class distribution nor an expert-quality score.

**Figure 10 — Measured outcomes and their routes.** Show all twelve points or rows, with separate clean/stress marks. A second view shows per-run assignment counts and their total. Selecting a class reveals its actual route counts, without relabelling a frequently used expert as a digit recognizer.

### Change a real input and follow the consequence

Take validation row 1762 of the original training file, an observed zero. With MoE .01, seed 17, its class-0 probability is .951716 and its patch-route counts are \([0,1,15,16]\). Zero the lower half: the predicted class becomes 9, class-0 probability becomes .008668 and routes become \([15,7,4,6]\). Both the representation and the route assignment changed.

Now disable expert 0 on the original image, without renormalizing surviving routes. Nothing changes: that expert was never selected. This is a useful control because a visual intervention should not produce a result when no computational path exists.

A router **temperature** \(\tau>0\) divides every score by \(\tau\) before softmax. Increasing it flattens the selected weights; a positive common divisor preserves score rankings. Doubling the temperature therefore preserves selected identities but changes mixture weights. Here class-0 probability becomes .946955 while the class remains 0 and route counts stay fixed. Unchanged labels do not mean unchanged computation.

**Investigation 5 — Edit a patch, inspect the entire path.** Use a different observed digit, make an actual pixel edit and observe whether routes, mixture outputs and class probabilities change. Compare removing a selected expert with removing an unused one, and compare a temperature edit with a route-changing input edit. The model uses the edited pixels; it does not play back a prewritten response.

Attention in this image model is bidirectional. A changed patch can affect other patch representations before routing. That is legitimate here. A causal language decoder would require causal attention and routing conventions appropriate to its task.

### Reproduce the experiment

The complete [training program](moe_study.py) needs Python, NumPy and PyTorch, with the CSV beside it. From a local environment with those packages installed, run:

~~~bash
python moe_study.py
~~~

It saves all selected weights and the full measured record. The [calculation program](moe_calculations.py) reloads those weights, compares gathered sparse execution against a dense reference including derivatives, and produces the worked/fresh intervention inputs without fitting again. The executed environment was Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu.

The full source is included in the expandable block below so the example remains understandable without finding hidden helper functions.


<details><summary>Complete CPU training and evaluation program</summary>

~~~python
"""Train a small patch Transformer with dense or sparse expert FFNs."""
from pathlib import Path
import csv, copy, hashlib, json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)

class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(16, width, bias=False)
        self.value = nn.Linear(16, width, bias=False)
        self.down = nn.Linear(width, 16, bias=False)

    def forward(self, inputs):
        return self.down(F.silu(self.gate(inputs)) * self.value(inputs))

class DigitTransformer(nn.Module):
    def __init__(self, sparse):
        super().__init__()
        self.sparse = sparse
        self.project = nn.Linear(4, 16)
        self.position = nn.Parameter(torch.randn(16, 16) * .02)
        self.norm_attention = nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48, bias=False)
        self.attention_output = nn.Linear(16, 16, bias=False)
        self.norm_ffn = nn.LayerNorm(16)
        if sparse:
            self.router = nn.Linear(16, 4, bias=False)
            self.experts = nn.ModuleList([SwiGLU(16) for _ in range(4)])
        else:
            self.ffn = SwiGLU(32)
        self.norm_final = nn.LayerNorm(16)
        self.classifier = nn.Linear(16, 10)

    def forward(self, pixels, temperature=1., disabled_expert=None, trace=False):
        batch = pixels.shape[0]
        patches = pixels.reshape(batch, 4, 2, 4, 2).permute(0,1,3,2,4).reshape(batch,16,4)
        hidden = self.project(patches) + self.position
        qkv = self.qkv(self.norm_attention(hidden)).reshape(batch,16,3,2,8)
        query, key, value = qkv.permute(2,0,3,1,4).unbind(0)
        attention = (query @ key.transpose(-1,-2) / np.sqrt(8)).softmax(-1)
        context = (attention @ value).transpose(1,2).reshape(batch,16,16)
        hidden = hidden + self.attention_output(context)
        expert_inputs = self.norm_ffn(hidden).reshape(-1,16)
        auxiliary = hidden.new_tensor(0.)
        details = {}
        if self.sparse:
            scores = self.router(expert_inputs) / temperature
            probabilities = scores.softmax(-1)
            selected_scores, selected = scores.topk(2, dim=-1)
            weights = selected_scores.softmax(-1)
            combined = torch.zeros_like(expert_inputs)
            for expert_id, expert in enumerate(self.experts):
                token_ids, slots = torch.where(selected == expert_id)
                if token_ids.numel() and expert_id != disabled_expert:
                    outputs = expert(expert_inputs[token_ids])
                    combined.index_add_(0, token_ids, outputs * weights[token_ids, slots, None])
            fractions = torch.bincount(selected.flatten(), minlength=4).float() / selected.numel()
            auxiliary = 4 * (fractions.detach() * probabilities.mean(0)).sum()
            details = dict(selected=selected.reshape(batch,16,2), weights=weights.reshape(batch,16,2),
                           probabilities=probabilities.reshape(batch,16,4), counts=torch.bincount(selected.flatten(),minlength=4))
        else:
            combined = self.ffn(expert_inputs)
        hidden = hidden + combined.reshape(batch,16,16)
        logits = self.classifier(self.norm_final(hidden).mean(1))
        if trace:
            details.update(attention=attention, expert_inputs=expert_inputs.reshape(batch,16,16),
                           combined=combined.reshape(batch,16,16), logits=logits)
            return logits, auxiliary, details
        return logits, auxiliary

def load_data():
    rows = list(csv.DictReader((HERE / "optical-digits.csv").open()))
    pixels = torch.tensor([[float(r[f"pixel_{j}"]) / 16 for j in range(64)] for r in rows])
    labels = torch.tensor([int(r["label"]) for r in rows])
    roles = {role: torch.tensor([i for i,r in enumerate(rows) if r["role"] == role])
             for role in ("fit","validation","assessment")}
    return rows, pixels, labels, roles

def evaluate(model, pixels, labels):
    with torch.no_grad():
        logits, auxiliary, details = model(pixels, trace=True)
        predicted = logits.argmax(-1)
        confusion = torch.bincount(labels * 10 + predicted, minlength=100).reshape(10,10)
        result = dict(cross_entropy=F.cross_entropy(logits,labels).item(),
                      correct=(predicted==labels).sum().item(), count=len(labels),
                      confusion=confusion.tolist(), auxiliary=auxiliary.item())
        if model.sparse:
            result["route_counts"] = details["counts"].tolist()
            result["class_route_counts"] = [torch.bincount(details["selected"][labels==c].flatten(),minlength=4).tolist()
                                           for c in range(10)]
    return result

def run_study():
    rows, pixels, labels, roles = load_data()
    fits, results = {}, []
    for condition, coefficient in (("dense",0.),("moe_0",0.),("moe_001",.01),("moe_01",.1)):
        for seed in (17,41,73):
            torch.manual_seed(seed)
            model = DigitTransformer(condition != "dense")
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
            generator = torch.Generator().manual_seed(10000+seed)
            curve, best_loss, selected_step = [], float("inf"), 0
            for step in range(1,241):
                model.train()
                sample = roles["fit"][torch.randint(len(roles["fit"]),(64,),generator=generator)]
                optimizer.zero_grad()
                logits, balance = model(pixels[sample])
                task_loss = F.cross_entropy(logits,labels[sample])
                (task_loss+coefficient*balance).backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step == 1 or step % 20 == 0:
                    model.eval()
                    validation = evaluate(model,pixels[roles["validation"]],labels[roles["validation"]])
                    curve.append(dict(step=step, minibatch_ce_before=task_loss.item(),
                                      minibatch_aux_before=balance.item(), validation_ce_after=validation["cross_entropy"]))
                    if validation["cross_entropy"] < best_loss:
                        best_loss = validation["cross_entropy"]
                        selected_step, best = step, copy.deepcopy(model.state_dict())
            model.load_state_dict(best)
            model.eval()
            key = f"{condition}_{seed}"
            fits[key] = {name: values.tolist() for name,values in best.items()}
            measurements = {role:evaluate(model,pixels[index],labels[index]) for role,index in roles.items()}
            changed = pixels[roles["assessment"]].clone().reshape(-1,8,8)
            changed[:,4:,:] = 0
            measurements["assessment_lower_half_zero"] = evaluate(model,changed.reshape(-1,64),labels[roles["assessment"]])
            results.append(dict(key=key,condition=condition,seed=seed,coefficient=coefficient,
                                parameters=sum(p.numel() for p in model.parameters()),selected_step=selected_step,
                                curve=curve,measurements=measurements))
            print(key, selected_step, measurements["assessment"]["correct"], flush=True)
    (HERE/"fitted-models.json").write_text(json.dumps(fits,separators=(",",":"))+"\n")
    output=dict(python="3.12.14",numpy=np.__version__,torch=torch.__version__,
                data_sha256=hashlib.sha256((HERE/"optical-digits.csv").read_bytes()).hexdigest(),runs=results)
    (HERE/"study-results.json").write_text(json.dumps(output,indent=2)+"\n")

if __name__ == "__main__":
    run_study()

~~~

</details>

### Find the implementation boundary before replacing the dispatcher

The complete `DigitTransformer` above is both the scratch routing implementation and an ordinary PyTorch training model. The router's logits, top-2 choices, selected softmax, per-expert gather and `index_add_` combine are visible. Experts are normal registered `nn.Module` objects, so their parameters enter the optimizer and checkpoint. `moe_calculations.py::dense_reference` evaluates all experts only as an independent oracle: it compares the same output, auxiliary objective and all parameter gradients with sparse dispatch. That dense oracle is not the scalable production path.

The local model is explicitly dropless. The separate `capacity` calculation in `moe_calculations.py` owns the token-order/capacity policy taught in §4: it preserves surviving original gates and can leave a token with no expert update. That is a different operator from renormalizing surviving gates or rerouting overflow. The residual still carries the token. Expert-choice balancing, auxiliary-free selection bias, distributed all-to-all and MegaBlocks are named alternatives, not hidden behavior of this model.

**Take control:** in a copy of the study, replace top-2 by top-1 while keeping the selected-softmax rule. Use a single minibatch and compare task-loss router gradients before changing the auxiliary loss. **Hint:** softmax over one selected logit is exactly one. **Solution:** the task-loss route through that gate has zero local router derivative away from selection boundaries. Full-softmax selected probabilities would be a different policy. Keep the balance derivative visible separately; a nonzero total router gradient does not prove the task gate itself learned. Repeat the sparse/dense value and gradient comparison after the change.

For larger expert counts, the four-expert loop's repeated selection scan becomes a real cost. Group the T×k selected assignments by expert once, gather each contiguous group, and scatter-add back with the original token/slot indices; sorting takes O(Tk log(Tk)) or a counting/bucket strategy exploits bounded integer expert IDs. Expert work depends on assigned tokens, while router scoring still touches all E experts. A maintained grouped-GEMM/distributed backend changes the execution path, not these routing equations. Its performance and overflow contract must be measured on the intended workload; neither the classroom oracle nor the package name is evidence of a speedup.

## 9. Reuse an existing model, and choose the right granularity

### Sparse upcycling

Suppose you already trained a dense Transformer. You can copy one of its FFNs into several expert branches, add a new router, and continue training. This is **sparse upcycling**. It reuses learned parameters instead of requiring every MoE to start from random weights. The original study explores both vision and language models and treats the architectural transition itself as something to evaluate. [Sparse Upcycling](https://arxiv.org/pdf/2212.05055)

There is a revealing exact case. If every copied expert computes the same \(F(x)\), selected weights sum to one, and no route is dropped, then

\[
\sum_{i\in S}g_iF(x)=F(x).
\]

The initial MoE replacement preserves that FFN's output. Its router task derivative is initially zero because every \(E_i-y\) is zero. Once different routed examples produce different expert updates, the branches can separate and the router can have a useful continuous comparison. If all updates also remain identical, symmetry persists.

Full-softmax masked weights, route dropping, an added shared expert or altered normalization can break the equality immediately. Thus “copy the expert” is not a complete preservation argument.

**Figure 11 — Copying preserves a function under stated conditions.** Show a dense FFN copied into three identical branches, two selected weights summing to one, and equal outputs before/after the replacement. A second panel drops one route without reweighting and shows why equality no longer follows.

Fine-tuning an MoE also involves its route distribution. A narrower new dataset may activate different branches from pretraining, and experts have different amounts of relevant evidence. Inspect task performance and routing together. Freezing experts, training only adapters, changing the router, or training all parameters are experimental choices. There is no universal rule that the dense model's fine-tuning settings will be optimal.

### From many experts to a smaller deployment

Distillation trains a smaller student to reproduce useful behavior of a larger teacher, for example by matching output probabilities as well as labels. Simply deleting several experts from a trained token-routed MoE changes its function and is not automatically equivalent to distillation.

A more unusual application changes routing granularity before deployment. In task-level MoE for multilingual translation, routes depend on a task identity such as a language pair. A known task can then use its own subset of experts, making extraction of a smaller task-specific subnetwork possible. That differs from discovering after training that a token-routed model sometimes visits many branches. The benefit comes with less token-specific routing freedom. [TaskMoE author explanation](https://research.google/blog/learning-to-route-by-task-for-efficient-inference/)

This gives a practical design question: does the decision need to vary per token, patch, sentence, task or request? Finer decisions can adapt more locally; coarser decisions can simplify scheduling and deployment. Match that choice to what information is available and what variation the task needs.

For images, a patch is a natural routing unit because different regions can require different transformations; our small study makes this explicit. For multilingual tasks, a shared model may distribute learned capacity across varied data, but a language label is not guaranteed to emerge as an expert identity. MoE is also compatible with other token mixers: later [Jamba](/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals) combines state-space/attention choices with expert FFNs. The two decisions answer different questions.

## 10. Diagnose the symptom at its actual source

| Observation | Inspect first | Why |
| --- | --- | --- |
| Few experts receive most assignments | Counts by layer/group over time, gate probabilities, task metrics | Collapse, data composition and useful uneven specialization are different possibilities |
| Output changes when batch companions change | Capacity, routing priority, group construction, stochastic evaluation | Independent token scores do not imply independent capacity competition |
| All top-1 router task gradients are zero | Selected normalization versus retained full-softmax weight | A normalized single gate is exactly one |
| An unselected router logit has a derivative | Full-softmax denominator and auxiliary objectives | Sparse expert execution does not mean every unselected router score is detached |
| Sparse code matches shape but not values | Gather IDs, slot weights and additive return operation | Multiple routes can overwrite or be returned to the wrong token |
| Balanced load but weak predictions | Task loss, input representation, expert outputs and interventions | A count histogram does not measure usefulness |
| NaNs in routing | Stable log-sum-exp/softmax, precision, initialization and optimizer states | “16-bit” is not one numerical format; z-loss is not a universal guarantee |
| Little speed gain despite low active parameter count | Router, expert batch shapes, dispatch, communication, memory and shared blocks | Saved arithmetic is only one component of execution |
| Fine-tuned model loses quality | Validation distribution, routing drift, per-expert exposure and training protocol | A narrowed dataset can change which branches receive training |
| A disabled expert seems to have no effect | Whether it was selected and the actual probability/vector delta | An unused branch is an expected null; unchanged class alone is insufficient |

The three books to keep are the mathematical function, the training evidence and the execution cost. A claim about one does not automatically settle the others.

## 11. Practice with new inputs

Work out the core routing and dispatch questions before opening the solutions. The later questions ask you to transfer those mechanisms to training and deployment decisions.

### 1. Compute both mixtures

There are three expert scalar outputs \([1,5,-2]\), logits \(\log[2,5,3]\) and \(k=2\). Find the selected experts and outputs under selected normalization and full-softmax weighting.

<details><summary>Hint</summary>
Selection uses the masses 2, 5 and 3; the two normalizers are 8 and 10.
</details>

<details><summary>Solution</summary>
Select experts 1 and 2. Selected normalization gives \(5/8\cdot5+3/8\cdot(-2)=19/8=2.375\). Full-softmax weighting gives \(5/10\cdot5+3/10\cdot(-2)=1.9\). The selected identities agree, but their weights sum to 1 versus .8.
</details>

### 2. A zero router gradient

For question 1, compute the derivative of the scalar output with respect to each logit under selected normalization. Then set \(k=1\) and explain what changes.

<details><summary>Solution</summary>
For selected indices, \(dy/dh_i=g_i(E_i-y)\). The derivatives are \([0,105/64,-105/64]\). With one selected expert, the normalized gate is one and the derivative to every router score is zero away from selection boundaries. The selected expert's own parameters can still receive task gradients. Full-softmax top-1 would be a different calculation.
</details>

### 3. Count the losses of capacity

Six tokens each select experts \([0,1]\). There are three experts and \(k=2\). Under \(C=\lceil cTk/N\rceil\), set \(c=1\), use token-order dispatch, and drop excess assignments without reweighting. How many assignments and entire token routes are lost?

<details><summary>Solution</summary>
Capacity is \(\lceil12/3\rceil=4\) per expert. Experts 0 and 1 each receive six assignments, so each drops two: four assignments total. The last two tokens lose both routes; two tokens therefore have zero MoE contribution. Expert 2 remains empty despite unused capacity. Every token still has its residual path. Raising capacity to six eliminates drops without changing the initial selected identities.
</details>

### 4. Find the scatter bug

A token has two expert outputs, \([2,1]\) and \([-1,3]\), with weights .25 and .75. Code first writes the weighted first output into the token row, then assigns the weighted second output to the same row. What should the row contain, and what does the buggy code contain?

<details><summary>Solution</summary>
It should contain \(.25[2,1]+.75[-1,3]=[-.25,2.5]\). Overwrite leaves only \([-.75,2.25]\). Correct combination must add both contributions at the destination index.
</details>

### 5. A resource budget

Use \(d=32,m=96,N=6,k=2\), no biases. Calculate total expert parameters, active expert MACs per token and router parameters. Then halve expert width, double expert count and double selected count.

<details><summary>Solution</summary>
One expert has \(3\cdot32\cdot96=9216\) parameters. Total expert parameters are 55,296; active expert MACs are 18,432; router parameters are 192. After changing to \(m=48,N=12,k=4\), total expert parameters and active expert MACs stay fixed, router parameters become 384, and assignment count per token doubles. Equal matrix work does not imply equal dispatch cost.
</details>

### 6. Read evidence without inventing specialization

A model routes most patches of digit 8 to expert 2. What additional investigation would help establish whether expert 2 contributes particularly to recognizing 8?

<details><summary>Solution</summary>
Compare controlled expert-output interventions on a held-out set of 8s and other digits, retaining actual probability/loss changes and assignment frequencies. Use the same intervention policy and account for whether the expert was selected. Compare against other experts and a null intervention; keep the original model as baseline. A larger effect can support a functional association under that intervention, but the route histogram alone cannot establish a self-contained “8 expert,” and an intervention can move the model outside its trained distribution.
</details>

### 7. Explain a stable probability distribution with a changing loss

Let logits be \(\log[.2,.3,.5]\). Add \(-2\) to all three. Which probabilities, selected indices and z-loss change?

<details><summary>Solution</summary>
Probabilities and selected indices do not change. Initially log-sum-exp is zero; after the shift it is −2, so the z-loss becomes 4. The common shift changes the normalizer's absolute scale, not relative preferences.
</details>

### 8. Can one scalar certify balance?

Someone reports \(L_{\mathrm{bal}}<1\) under the normalized \(q,P\) convention and concludes the router is “more balanced than uniform.” Explain the flaw and name the measurements you would request.

<details><summary>Solution</summary>
The bilinear surrogate is not a distance from uniformity; §5 gives an explicit .88275 counterexample with counts 3:1. Request actual pre/post-capacity counts, group scope, soft probabilities, gate mass, dropped assignments and task metrics over time. A different convention can also multiply the loss scale by \(k\).
</details>

### 9. Correct a biased mixture

Affinities are \([.4,.8,.7]\), biases are \([.5,0,0]\), and two experts are selected by affinity plus bias but weighted by normalized original affinities. Expert scalar outputs are \([3,-1,2]\). Compute the correct output.

<details><summary>Solution</summary>
Biased scores are \([.9,.8,.7]\), selecting experts 0 and 1. Original selected affinities .4 and .8 normalize to \(1/3,2/3\). Output is \(1/3\cdot3+2/3\cdot(-1)=1/3\). Weighting by .9 and .8 instead would give \(19/17\), a different operator.
</details>

### 10. Upcycle without changing the function

Three copied experts all return \([2,-3]\). Explain when their selected mixture equals the old dense FFN, and give two changes that break the argument.

<details><summary>Solution</summary>
It equals the dense output whenever the retained weights sum to one. Selected normalization with no drops satisfies this regardless of which identical experts are chosen. Dropping a route without renormalization or retaining full-softmax weights whose selected sum is below one breaks the equality. Adding another always-on expert also changes the output unless the rest of the architecture is adjusted.
</details>

### 11. Audit an autoregressive proposal

A decoder chooses each expert's highest-scoring tokens across an entire teacher-forced sequence. Its attention is causal. Is that sufficient to guarantee the earlier outputs do not depend on later input tokens?

<details><summary>Solution</summary>
No. A later token can compete for an expert slot and displace an earlier token. The earlier expert output can therefore depend on a future competitor through allocation, even though attention itself is causal. Specify an allocation using only available information, or adopt a routing rule consistent between training and generation; test future edits against every earlier output. The appropriate fix depends on the intended routing design.
</details>

### 12. Plan a meaningful comparison

You can run two systems: one has twice the experts, the other doubles every expert's width. Design a comparison that would help choose a deployment, rather than producing a misleading “parameters versus accuracy” plot.

<details><summary>Solution</summary>
Specify the same task/data roles, validation selection, training budget definition, activated branches and shared architecture. Report total parameters, active work, measured task quality and measured latency/throughput at representative batch/sequence shapes. Include memory and dispatch/communication costs, all planned seeds/configurations, numerical precision and backend versions. Distinguish training efficiency from inference efficiency; reserve an assessment set and avoid choosing the displayed configuration from its outcomes. There is no required winner.
</details>

You are ready to move on when you can explain a token's exact weighted route, distinguish router and expert derivatives, reason about capacity at the assignment level, and read both resource counts and actual model results. Model-family names become useful shorthand only after those mechanisms are clear.

## 12. References and another way to learn it

- **Original sparse layer:** [Shazeer et al., Outrageously Large Neural Networks](https://arxiv.org/html/1701.06538v1). Read §2 for noisy selected gates, §3 for expert batch and communication constraints, and §4/Appendix A for importance versus load. Its historical experiments are examples, not current hardware advice.
- **One selected expert:** [Switch Transformers](https://arxiv.org/html/2101.03961v3). §2 connects top-1 probability weighting, capacity and training choices; §5 discusses parallel dimensions. Keep the gate-normalization convention visible when comparing it with this lesson's selected-normalized model.
- **Stability and transfer:** [ST-MoE](https://arxiv.org/html/2202.08906v2). A useful deeper route through stability, precision, fine-tuning, routing design and empirical specialization. Its z-loss addresses a different quantity from load balance.
- **A concrete decoder architecture:** [Mixtral](https://arxiv.org/html/2401.04088v1). The architecture and routing analysis give a compact example of sparse FFNs inside a shared Transformer; subject-specific expert labels need evidence.
- **Fine-grained and shared experts:** [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2). Read §2.1.2 closely for affinity, selection bias, selected weights, sequence balance and node restrictions. Follow its exact version rather than replacing the formula with a generic softmax sketch.
- **A different selection direction:** [Expert Choice author article](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/). The row-versus-column visual makes assignment constraints easier to see. Treat its performance results as measurements of its reported experiments.
- **Reuse existing weights:** [Sparse Upcycling](https://arxiv.org/pdf/2212.05055). §§2–3 explain copying FFNs and adding routing; the transition has its own training choices. Useful after understanding the equality test in §9.
- **A less obvious deployment application:** [Task-level routing for translation](https://research.google/blog/learning-to-route-by-task-for-efficient-inference/). Its training-versus-extracted-subnetwork diagrams explain why choosing routes per task can alter serving requirements.
- **From equations to kernels:** [MegaBlocks paper](https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf) and [official implementation](https://github.com/databricks/megablocks). Read after the gather/scatter and capacity example. Kernel capabilities and recommendations depend on version and hardware.
- **Lecture/video route:** [Stanford CS336, 2025 MoE lecture materials](https://cs336.stanford.edu/spring2025/#schedule) and the official course's [2026 recordings playlist](https://www.youtube.com/watch?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&v=JuoVZkPBiKk). The 2026 schedule places attention alternatives and MoE in lecture 4. This route expects prior Transformer and mathematical fluency. The course listings were verified; the recordings were not watched for this manuscript.

The next topic in this module is [Interleaved / Cross-Attention Architectures](/learn/path/full-curriculum/interleaved-cross-attention-architectures?module=deep-learning-fundamentals). It asks where information is read from and how different streams connect. Here, the router chose which transformation to apply; next, the architecture chooses how representations meet.
