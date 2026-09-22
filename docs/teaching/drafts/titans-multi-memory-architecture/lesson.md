# Titans: a memory that learns while a sequence arrives

**Explore as you read.** Edit key/query/value cards, rate, momentum, decay, request tokens and chunk size; step bounded writes or continue/reset request state. Show weight/update state, residual and gradient terms, current query output, gated topology and anchor/current-gradient comparison. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose write timing, state isolation and chunk semantics from the outputs they can affect, rather than from the word memory alone.


Imagine reading a long maintenance log. You need the last few entries to understand what is happening now, an impression of older recurring faults, and general knowledge about how maintenance reports are written. Keeping every entry immediately accessible costs space. Compressing everything into one small summary risks losing a detail you will need later.

Titans explores a combination: attention over recent context, a small neural network whose weights change as it processes the sequence, and learned vectors that carry information shared across sequences. The unusual part is the second one. Reading this memory means running a neural network. Writing to it means taking a gradient step on that network.

**First-pass route.** Start with §§ 1–4 and the associative-memory investigation: follow two writes by hand before reading the code. Then use the three-branch example in § 5, run the real-data program in § 7, and attempt practice 1–4. Return for the outer derivative in § 6, chunk parallelization in § 8, state accounting in § 9 and the research-reading branch in § 10. Expect roughly 40–50 minutes of core reading, with a separate 45–75 minutes for the first exercises and programs. The deeper branches can be another sitting.

This is an architecture lesson, so a little neural-network background helps. A **weight** is a number used by a model; a **loss** measures an error; a **gradient** tells us how changing the weights changes that loss. A **query** asks for a representation, a **key** identifies an association, and a **value** is the representation associated with that key. We will make those roles concrete before using large tensors. The previous [Jamba lesson](/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals) combines attention and recurrence. Here, the recurrent state becomes the parameters of a learner.

## 1. Three places information can live

Think of a workbench, an adjustable prediction rule, and a shared instruction card. The workbench keeps recent items available; the rule changes with experience; the instruction card begins the same for each new job. The analogy describes roles. It does not establish that these components implement human memory or keep reliable copies of arbitrary facts.

| Component | What is stored? | What changes during one sequence? | How is it used? |
| --- | --- | --- | --- |
| Recent attention context | Representations of recent positions, often cached as projected keys and values | Positions enter and leave the permitted window | A query forms a weighted combination of accessible values |
| Contextual neural memory | Fast weights and their update state | Gradient-based writes change the weights and momentum | A query passes through the current memory network |
| Persistent memory | Learned input-independent vectors | Their shared parameter values stay fixed at inference | The core can attend to these prefix representations |

**Follow one sequence.** The six-entry strip has a bracket under its last two entries: the recent window. Earlier observations lead into a small weight grid. A separate prefix row enters from above with no incoming arrow from this session. When the seventh entry arrives, the bracket moves, the grid can change, and the prefix parameter row remains the same. These three paths show the different lifetimes of the stored information.

An attention cache also stores transformed representations, not a literal database of original text. Its advantage is that separate positions remain addressable inside its permitted context. Neural memory compresses associations into shared parameters: a later write can alter more than one earlier answer. Its storage can stay bounded while sequence length grows; its information capacity remains finite.

Persistent vectors are not the same as the initial fast weights. A memory can start from nonzero, learned or otherwise specified initial weights even when no persistent prefix is present. Removing the prefix therefore does not logically imply that the model starts with no useful prior. What a trained prefix learns must be assessed through the task, not inferred from its name.

## 2. Write a rule instead of appending a record

Start with a memory that accepts a two-number key and returns one number:

\[
M_w(k)=w_1 k_1+w_2 k_2.
\]

The key might represent two features of an event. For the arithmetic example, the numbers are deliberately constructed. Our first desired association is key \((1,0)\) → value 2. Starting from \(w=(0,0)\), the memory answers 0. Define the residual as prediction minus desired value, so it is −2.

We use **half-squared error** throughout the calculations:

\[
\ell(w;k,v)=\tfrac12(M_w(k)-v)^2,
\qquad g=\nabla_w\ell=(M_w(k)-v)k.
\]

The one-half cancels the factor 2 when differentiating a square. The main Titans objective writes squared error without the half; its appendix uses the half convention. To obtain the same step when switching to the unhalved loss, halve the learning rate. Loss reduction and learning rate belong together.

Here \(g=(-2,0)\). A gradient step with rate \(\theta=0.5\) gives

\[
w_{\mathrm{new}}=w-\theta g=(0,0)-0.5(-2,0)=(1,0).
\]

Querying \((1,0)\) again now returns 1. One update moved the answer toward 2. It did not create a perfect database entry.

**The write direction.** The key arrow points along the first coordinate axis. Each key coordinate connects to the weight it changes. Beside the before/after grid, the answer marker moves from 0 to 1 on a ruler whose target is 2. The zero second coordinate explains why the second weight stays 0.

### The second write reveals interference

Keep the same weights \((1,0)\), rate 0.5 and a new target 4. First use the key \((0,1)\). The prediction is 0, residual −4, gradient \((0,-4)\), and new weights \((1,2)\). The original query \((1,0)\) still returns 1.

Now restart from \((1,0)\) and use \((1,1)\) as the second key. The prediction is 1, residual −3, gradient \((-3,-3)\), and new weights \((2.5,1.5)\). The original query now returns 2.5. The new association changed an old answer because the keys share a direction in parameter space.

| Second key | New weights | Answer to original key | Interpretation |
| --- | --- | --- | --- |
| \((0,1)\) | \((1,2)\) | 1 | Orthogonal direction leaves this old read unchanged |
| \((1,1)\) | \((2.5,1.5)\) | 2.5 | Correlated direction changes the old read |

You can predict this without recalculating every weight. Let \(q\) be the old query and \(r=M_w(k)-v\) the new residual. For one linear update without decay or momentum,

\[
\Delta M(q)=-\theta r\,k^\top q.
\]

The change is governed by the overlap \(k^\top q\). This is the same old-read difference seen in the table, reached by an algebraic route. Orthogonal keys give zero overlap; other keys may reinforce or disrupt an old answer. In a nonlinear memory, local parameter sensitivities replace this simple key-overlap calculation.

**Investigation — which answer does the write disturb?** Change the key, value or write rate and watch the old query's output and its signed change update immediately. Change the second key's coordinates and target, then make the write. Watch the key arrow, weight cells and old-query answer together. Try a key perpendicular to your chosen query as a control case. Explain the sign using the residual and the dot product, then choose a different query that gives the opposite direction of change.

## 3. Error, gradient, momentum and forgetting are different quantities

With many output coordinates, write \(M_\phi(k)\in\mathbb R^{d_v}\), where \(\phi\) contains all memory weights. The residual is \(r=M_\phi(k)-v\). If \(J\) is the matrix of derivatives of the memory output with respect to its parameters, the half-squared-loss gradient is

\[
g=J^\top r.
\]

A large residual can produce a large gradient, but the parameter sensitivity matters. In our bias-free linear memory, the key \((0,0)\) gives zero gradient even if the target is 10 and the loss is 50: multiplying any weights by that key still gives 0. Changing those weights cannot fix this example. A bias or different key representation changes the situation.

The Titans paper motivates writes through “surprise.” For implementation, keep the following measurements separate: loss, parameter-gradient norm, momentum, actual parameter change and later task performance. None is a direct measurement of a fact's semantic importance. In particular, a factor 4 difference in squared loss implies a factor 2 difference in residual norm; it does not generally imply a factor 2 gradient norm, even for a fixed Jacobian, because residual directions can differ.

### Add a memory of recent updates

Let \(S\) have the same shapes as the memory weights. One update is

\[
g_t=\nabla_{\phi}\ell(\phi_{t-1};k_t,v_t),\qquad
S_t=\eta_t S_{t-1}-\theta_t g_t,
\]
\[
\phi_t=(1-\alpha_t)\phi_{t-1}+S_t.
\]

Read the equations in this order: compute today's gradient at the old weights; retain a fraction of the previous update; add today's gradient step; shrink the old weights; add the resulting update. \(\theta_t\) controls the new gradient, \(\eta_t\) retains momentum, and \(\alpha_t\) controls direct shrinkage. Titans makes these quantities data-dependent through learned mappings. Fixed values in our calculations let us isolate the mechanism.

For one weight, take \(w=1\), old \(S=0.2\), \(g=-2\), rate 0.1, momentum retention 0.5 and decay 0.1. Then \(S'=0.5(0.2)-0.1(-2)=0.3\), and \(w'=0.9(1)+0.3=1.2\). The weight changed by 0.2, while the momentum is 0.3. Decay accounts for the difference.

**Add the actual vectors.** Separate arrows represent retained momentum, the new gradient step and weight shrinkage. Their signed sum is the change in the weight grid. Inspect the components separately before interpreting the combined change.

If the new gradient is zero, existing momentum can still move the weights. If \(\alpha=1\), the retained old-weight term vanishes, but \(S_t\) can remain nonzero. A full sequence reset therefore restores the specified initial weights and clears momentum, recent context, convolution history and position/chunk state. Setting one forget gate to 1 is not a complete reset.

With no writes and no momentum, repeated constant decay gives \(\phi_t=(1-\alpha)^t\phi_0\). For \(\alpha=0.01\), half the initial amplitude remains after about 68.97 steps. This is a decay calculation, not a measured lifetime for stored facts: continued writes and nonlinear readouts change actual recall.

### When can a simple write diverge?

**Deeper branch.** For one fixed key in linear memory, with no momentum or decay, the residual after a step is

\[
r'=r(1-\theta\|k\|^2).
\]

Thus repeated updates reduce the residual magnitude when \(0<\theta\|k\|^2<2\). At 1, this particular association is fitted in one step. At 2, the residual alternates sign without shrinking. Above 2, its magnitude grows. This explains why key scale and learning rate must be considered together. It also explains why the zero-key example cannot improve.

A nonlinear network, changing keys and momentum add further dependencies. Decay is neither necessary for every stable sequence nor sufficient to rescue every unstable one. Monitor finite values, gradients and updates; inspect their cause before changing a rate. Clipping is an explicit modification to the update rule, with a specified norm and threshold, rather than a universal default supplied by the architecture's name.

## 4. Make the memory nonlinear

A two-layer memory can map a key through hidden features:

\[
M_\phi(k)=W_2\,\operatorname{SiLU}(W_1k+b_1)+b_2.
\]

Here \(W_1\) has shape \(H\times d_k\), \(b_1\) has \(H\) entries, \(W_2\) has shape \(d_v\times H\), and \(b_2\) has \(d_v\) entries. SiLU maps a scalar \(a\) to \(a\sigma(a)\), where \(\sigma\) is the logistic sigmoid. The intermediate \(H\)-vector contains nonlinear combinations of the key coordinates. We can still compute a residual, backpropagate it and update every parameter.

Why add depth? A linear rule cannot express every relationship between keys and values. For example, a single affine function cannot map both \((1,1)\) and \((-1,-1)\) to 1 while mapping \((1,-1)\) and \((-1,1)\) to −1. Adding the two positive-case equations fixes twice the bias at 2; adding the negative-case equations fixes it at −2, a contradiction. Nonlinear hidden features can distinguish these patterns.

More expressive functions are useful, but there is no general law saying “H hidden units store H independent facts.” Capacity also depends on precision, key geometry, objectives, training and the tolerated retrieval error. Repeatedly rehearsing the same examples and processing each example once are different experiments.

### A complete memory cell you can inspect

The supplied [neural_memory.py](neural_memory.py) contains the forward function, initialization, parameter copying and this update. It uses a scalar value per key; a batch averages the half-squared scalar losses. Each request owns its parameter and momentum tuples. The update returns new tensors instead of silently changing a shared global model.

```python
import torch
from neural_memory import initialize_memory, read_memory, write_memory

parameters = initialize_memory(seed=3, input_size=2, hidden_size=3)
momentum = tuple(torch.zeros_like(weight) for weight in parameters)
key = torch.tensor([1.0, -0.5], dtype=torch.float64)
value = torch.tensor(0.8, dtype=torch.float64)

parameters, momentum, loss, gradients = write_memory(
    parameters, momentum, key, value,
    rate=0.1, retention=0.0, decay=0.0,
    differentiable=True,
)
query = torch.tensor([-0.25, 0.75], dtype=torch.float64)
retrieved = read_memory(parameters, query)
```

The query differs from the write key. This matters: making the training association easier does not mean every possible query becomes more useful. The next section gives attention another path to the information it needs; § 6 explains why the outer task trains the system to make the paths useful together.

## 5. Connect memory to attention

Titans describes three wiring choices. The distinction is where a memory read enters the computation, not a fixed mapping from task names to a universally best variant. The paper's equations omit some residual and normalization detail, so these diagrams describe their main data paths.

In a full model, the input representation \(x_t\in\mathbb R^{d_{in}}\) supplies learned views. With our column-vector convention, \(k_t=W_Kx_t\), \(v_t=W_Vx_t\), and \(q_t=W_Qx_t\). The projection shapes are \(d_k\times d_{in}\), \(d_v\times d_{in}\), and \(d_k\times d_{in}\), respectively. The write learns to associate the key view with the value view; the query view chooses what to read. The projection weights belong to the outer training loop, while the contextual memory changes within a sequence. Input-conditioned rate and retention mappings can also learn which updates are useful.

The paper's fuller blocks use residual connections, normalized queries/keys, nonlinear projections and short depthwise convolutions. A causal convolution mixes a channel's recent positions with learned coefficients, giving a key some local history before the memory sees it. That history is another piece of sequence state to carry across a boundary. Normalizing a nonzero key controls its length, which connects directly to the key-scale effect derived in § 3. Our unnormalized linear investigation keeps the zero key valid so you can inspect that boundary explicitly.

| Variant | Main path | What attention receives |
| --- | --- | --- |
| Memory as Context, MAC | Read historical memory for a segment → append retrieved context and persistent prefix → attention → update memory from the resulting representation → gated output | The segment plus retrieved historical representations and persistent prefix |
| Memory as Gate, MAG | Persistent prefix and sequence feed a local attention branch and a neural-memory branch → combine their outputs through a nonlinear gate | Prefix and accessible recent input representations |
| Memory as Layer, MAL | Prefix and input → memory layer → attention layer | The memory layer's transformed sequence |

**Three different junctions.** MAC supplies extra rows to attention. MAG has two parallel lanes meeting at a gate. MAL has two blocks in series, memory first. In each, the state arrow crosses to the next segment or position. Follow where the read enters: attention input, a parallel output junction, or an earlier layer.

In MAC, current input queries the historical state before the segment update. The attention result then supplies information for writing. In the paper's schematic, the final output also uses a read from the updated memory. A write followed by a read is consequently not automatically a causality error. The question is which observations were available to produce that write.

For autoregressive prediction at position t, every dependency must originate in tokens already observed at that position. A retrieved vector depends on its query as well as on the old memory. Moving vectors queried by future tokens into an apparently “historical” prefix does not make them safe. Similarly, masking attention cannot repair a memory read that already includes a future write. A faithful implementation must spell out segment retrieval, token availability and prefix masks, then verify that changing a future token leaves earlier outputs unchanged. Treat the paper's segment-level notation as a design description, not a complete indexing specification.

### Walk a small gated block all the way through

Our executable example specializes the MAG topology so every number is inspectable. Inputs are two-vectors. Query, key and value projections are identities. Local attention includes the current token and one previous token, plus an optional persistent vector \(p=(1,0)\). The fast memory is a two-by-two linear matrix, initially zero, with rate 0.5, momentum retention 0.5 and no decay. It writes the observed association \(x_t\to x_t\), then reads at query \(x_t\). Finally,

\[
o_t=a_t\odot\tanh(m_t),
\]

where \(a_t\) is the attention result, \(m_t\) the memory read, and \(\odot\) multiplies corresponding coordinates. This gate can attenuate or reverse a coordinate; it is not a convex probability mixture. These are declared teaching choices, not claimed trained-model defaults. We omit residuals, learned projections, convolution and additional normalization here so the three paths remain visible.

At the first token \((1,0)\), the prefix and current token are identical. Attention returns \((1,0)\). The write sets the first diagonal memory weight to 0.5, so the read is \((0.5,0)\). Output: \((\tanh(0.5),0)\approx(0.462117,0)\).

At the second token \((0,1)\), attention sees \((1,0),(1,0),(0,1)\). Dot each row with the query and divide by the square root of its dimension, giving scores \((0,0,1/\sqrt2)\). Softmax exponentiates the scores and divides by their sum: the resulting weights are approximately \((0.248255,0.248255,0.503490)\). Their weighted sum of the three rows gives \(a=(0.496510,0.503490)\). The memory becomes

\[
W=\begin{bmatrix}0.75&0\\0&0.5\end{bmatrix}.
\]

The first diagonal changed from 0.5 to 0.75 because of retained momentum, even though the new key points along the second axis. The new memory read is \((0,0.5)\), and the output is approximately \((0,0.232671)\).

Remove the prefix, replaying from the same initial state. At the second token the memory is unchanged, but attention has only two rows. Its second coordinate is now 0.669762, and the gated output's second coordinate becomes 0.309508. This intervention reveals the prefix's route through attention. At the first token, removing that duplicate prefix leaves the output unchanged: a useful control case.

**Investigation — what belongs to one request?** Edit a token, record which earlier outputs should stay fixed, and replay the two lanes. Continue a sequence across a saved boundary, then compare with a fresh sequence that starts at the same suffix. Finally, run a separate request and verify that it cannot change the fresh request's result. Track fast weights, momentum and the two-token window together. The supplied [memory_mechanisms.py](memory_mechanisms.py) implements the complete specialization, and [mechanism-results.json](mechanism-results.json) retains its actual arrays.

## 6. Two learning loops, with two different jobs

**Deeper branch.** The inner loop writes associations into fast weights for the current sequence. The outer loop learns the projections, gates, initial conditions and other model parameters that make those writes useful for the final task. A language-model outer objective can still be next-token cross-entropy. The inner target is a learned representation derived from available input, not the withheld answer token that the model is supposed to predict.

**A gradient through a gradient.** Follow old fast weights → inner gradient → new fast weights → query output → outer task loss along the time axis. Slow parameters feed the key, value and rate from above. The outer-gradient arrow travels backward through this whole path. In the inference view, slow parameters stay fixed while new fast-weight states continue to be created.

You can understand this with one scalar. Let the initial weight be 0. Write key 2 → value 1 with a learnable rate \(\theta\). The inner gradient is −2, so \(w'=2\theta\). Query at 3 and compare to outer target 2:

\[
\hat y=6\theta,\qquad L_{\text{outer}}=\tfrac12(6\theta-2)^2,
\qquad \frac{dL_{\text{outer}}}{d\theta}=6(6\theta-2).
\]

At \(\theta=0.25\), the new weight is 0.5, prediction 1.5, outer loss 0.125 and outer derivative −3. Gradient descent on the rate would increase it locally, making this subsequent query more accurate. The outer objective has taught something about how to write.

For a general parameter \(\psi\), differentiation through \(\phi'=\phi-\theta\nabla_\phi\ell(\phi;\psi)\) involves how the inner gradient changes with \(\psi\). With many steps it also includes how previous fast weights affect later writes. Automatic differentiation can evaluate these paths without materializing a dense Hessian. The [TTT paper's inner/outer discussion](https://arxiv.org/html/2407.04620v1#S2.SS2) is a useful companion.

In PyTorch, `autograd.grad(..., create_graph=True)` retains a graph for differentiating the computed gradient. For the nonlinear example in § 4, use outer target −0.3 at its query and half-squared outer loss. The resulting outer rate derivative agrees with a centered finite difference: −0.00249327930752 versus −0.00249327930749 at rate 0.1. That is the same differentiation idea as the hand scalar calculation, now through a SiLU network with a different query.

At inference we only need the current inner write, not an outer derivative through the entire session. The memory cell's ordinary mode detaches the new weights and momentum after each write, then enables gradients on the new weights for the next local write. Detachment at this point bounds retained autograd history. Using that mode during purported full outer training would remove paths from the objective and change what is being optimized.

`eval()` concerns module behavior such as dropout; it is separate from disabling automatic differentiation. A surrounding `no_grad()` would prevent the inner loss from recording the reverse-mode computation required for a write. Keep gradient recording active for that computation, even while avoiding a session-long outer graph. The [PyTorch gradient API](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html) and [no_grad semantics](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html) document the distinction.

## 7. A real stream: forecasting daily bike rentals

The logbook question becomes concrete with [UCI's Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset): 731 daily totals from Washington, DC's Capital Bikeshare system in 2011–2012. Hadi Fanaee-T supplied this dataset for studying rentals and their relation to conditions and events. We use the daily observations, not the larger hourly file. The retained [CSV](bike-sharing-daily.csv), [provider description](source-description.txt) and [provenance](data-provenance.md) make the exercise usable offline; UCI licenses the data under CC BY 4.0.

Our question is: **after training a small prediction rule on 2011, does allowing it to update after each newly observed day improve its 2012 forecasts?** This isolates online neural adaptation on a real stream. The experiment uses observed count targets after they arrive, rather than Titans' learned latent self-supervised targets. It trains an ordinary initial predictor rather than a complete end-to-end Titans language model. The three-branch computation and outer-learning mechanism were exposed separately above so you can identify exactly which part this study exercises.

### Specify when a number becomes available

Assume a day's total is available at that day's end. Before tomorrow arrives, use the preceding seven totals and the weekday of the day being forecast. No future weather observations or components of tomorrow's total enter the input.

**Predict, observe, write.** At the boundary between two dates, seven already observed counts enter a key. The forecast is recorded before the target is revealed. When that target arrives, a separate downward arrow leads into the loss and weight update. The updated state feeds the following date. A reporting-window boundary changes the chart label, not the information-availability rule.

The count mean and population standard deviation are fitted on 2011 only: approximately 3405.762 and 1376.864. Standardize counts using those constants. A key contains seven past standardized counts, followed by sine and cosine of the target weekday; normalize the resulting nine-vector to unit length. The memory has shape 9→8→1 with SiLU and biases. Its output is a standardized count, transformed back to rentals for scoring. This preprocessing is a declared compact teaching representation, not a claim that it is an optimal forecasting feature set.

Fit 358 examples from 2011: the first seven days supply history and targets begin on day 8. Use full-batch Adam with rate 0.01 for 1000 updates. Repeat seeds 3, 7, 19 to show initialization variation. Make two copies of each fitted model: one stays frozen, while the other uses rate 0.005, momentum retention 0.5 and decay 0.0001 after each observed 2012 target. Those choices, the seeds and the simple baselines were fixed in the [experiment protocol](experiment-protocol.md) before running this comparison.

Record a forecast first. Observe the day's target second. Update the adaptive model third. Report the first 183 forecasts as a development window and the following 183 as assessment. No settings were selected from either report, and the adaptive state carries forward across their boundary. The first window ends on 1 July 2012 because 2012 is a leap year; the assessment begins on 2 July.

### One actual forecast and write

For seed 3, both copies forecast 2609.616 rentals for 1 January 2012. They agree because adaptation has not yet made a write. The observed total is 2294, so the count error is 315.616. Dividing by the fitted scale gives the standardized residual; its half-square is 0.02627285. The gradient norm is 0.67017273 and the parameter-change norm is 0.00352720. Those last two are measured in parameter space, not rentals.

For 2 July 2012, the frozen copy forecasts 4707.926 and the adaptive copy 5944.152. The actual count is 6227. The adaptive model has incorporated earlier 2012 outcomes. This is the same state-carrying principle as the continued gated sequence in § 5, now applied to dated observations and measured forecast error.

### Run and inspect the full experiment

Save the supplied files together. Python, NumPy and PyTorch are required. The recorded run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu, float64 with one CPU thread. These are tested versions, not a requirement to install the latest release.

```sh
python -B memory_mechanisms.py
python -B rental_memory_study.py
python -B check_author_packet.py
```

The complete [study program](rental_memory_study.py) includes data loading, feature construction, fitting, replay, scoring and result export. Its central replay operation follows the three moments above:

```python
prediction = float(read_memory(weights, key).detach())
next_weights, next_momentum, loss, gradients = write_memory(
    weights, momentum, key, observed_value,
    rate=0.005, retention=0.5, decay=0.0001,
)
weights, momentum = next_weights, next_momentum
```

Here `observed_value` is supplied only after saving `prediction`. The complete program retains every date, forecast, residual, gradient norm and update norm in [rental-results.json](rental-results.json), along with initial/final weights and environment identity. The short excerpt shows the timing; use the full linked program to run it.

Mean absolute error, or **MAE**, averages the absolute difference between forecast and observation. Its units here are rentals per day. **RMSE** is the square root of the mean squared error; squaring gives large misses more influence. The complete result table retains both. For a readable first comparison, here is MAE, rounded to two decimals:

| Procedure | Development: 1 Jan–1 Jul | Assessment: 2 Jul–31 Dec |
| --- | ---: | ---: |
| Previous day's count | 896.54 | 843.81 |
| Count seven days earlier | 1092.26 | 1128.53 |
| Frozen network, seed 3 | 1334.04 | 1767.05 |
| Adaptive network, seed 3 | 925.34 | 823.54 |
| Frozen network, seed 7 | 1342.21 | 1781.81 |
| Adaptive network, seed 7 | 1007.78 | 867.34 |
| Frozen network, seed 19 | 1392.46 | 1813.87 |
| Adaptive network, seed 19 | 875.92 | 805.50 |

All three adaptive copies improve over their matched frozen networks. Yet the simple previous-day baseline beats two adaptive runs in development and one in assessment. The practical conclusion is to keep that baseline in the comparison. The extra computation provides an observed benefit over freezing this predictor, but initialization and the comparator affect whether the added complexity is useful.

**Measured figure — paired outcomes.** Each seed has two connected MAE dots, frozen and adaptive, and the horizontal line marks the previous-day baseline. Separate panels show development and assessment in the same units. The dated forecast view connects these aggregate errors to actual observations and both seed 3 predictions. Look for the cases where an adaptive dot remains above the simple baseline.

Try inspecting a date with a large error and its following write. A high error could reflect a poor representation, a real change, or an unusual event. The count alone cannot tell you which explanation caused it. State an additional measurement you would need before declaring that the model has detected a particular event.

### A one-line timing error can manufacture perfect performance

For a constructed scalar memory, let \(w=0\), key 1, observed target 4 and rate 1, with no momentum or decay. The forecast before observing the target is 0, an error of −4. After writing, the weight becomes 4. Evaluating against the same target at that point gives zero error. That measures fitting an already revealed target, not forecasting it.

The author checks perturb a later daily count and confirm that earlier forecasts stay unchanged. The first affected forecast comes after the changed count becomes available. This is a direct check of the dependency rule, rather than trusting a label such as “causal” or “test set.”

### Build the write rule, then choose what stays differentiable

[memory_mechanisms.py](memory_mechanisms.py) owns the explicit linear residual/outer-product gradient, momentum and decay in `linear_write`, plus the small complete gated memory/attention composition in `gated_sequence`. [neural_memory.py](neural_memory.py) is the ordinary research implementation for a nonlinear memory: `read_memory` evaluates its two-layer network from explicit parameter tensors and `write_memory` obtains the write-loss derivatives with `torch.autograd.grad`. It returns new parameter and momentum tuples rather than silently mutating a globally shared model. Autograd is the reused derivative engine; the new mechanism is the loss-driven persistent memory update.

The `differentiable` switch is a learning decision. With `True`, `create_graph=True` retains the derivative graph through a write so an outer objective can learn a write rate or initializer. With `False`, the newly returned state is detached and made a new leaf for the next write; this bounds the retained history during ordinary online replay but removes earlier-write meta-gradients. `rental_memory_study.py::replay` chooses that causal online path, preserving the forecast-before-observation timeline. Both paths implement complete writes; they optimize different derivative contracts.

No one-call Titans package is necessary for this route. Standard tensors, functional layer evaluation, autograd and the explicit state tuple are ordinary tools for research on changing fast weights. The linear update's outer product costs O(d_key d_value), and a nonlinear write costs a memory-network forward/backward plus parameter-sized momentum/state. Differentiating through many writes also retains their graphs; constant-sized *carried values* do not imply constant training-memory cost.

**Take control.** In `memory_mechanisms.py::nonlinear_outer`, compare the derivative of the outer loss with respect to the write rate using central differences and `write_memory(..., differentiable=True)`. Then detach the new state and inspect which derivative disappears. Keep the initial weights, key, target and query fixed.

<details><summary>Hint and reasoned solution</summary>

Central differences must rebuild the same initial state for rates η+ε and η−ε. The differentiable path includes how changing η changes the written weights and the later query output. Detaching makes that written value an independent leaf, so the outer loss has no graph path to η through the write; asking for it may return unused/None or raise unless unused inputs are explicitly permitted. That does not mean the numerical function is insensitive to η. It means the chosen differentiation contract discarded that dependence. Restore `differentiable=True` for the meta-learning question, and use the detached mode for the intentionally bounded replay question. Compare a smaller ε to diagnose cancellation rather than assuming the smallest possible ε is best.

</details>

## 8. Parallelize a declared update rule

**Deeper branch.** Each online step can depend on the previous fast weights twice: directly in the weight recurrence and inside the gradient calculation. The second dependency is expensive for a nonlinear memory. Calculating a chunk's gradients at one shared starting state removes that dependency inside the chunk. Prefix sums or scans can then combine the precomputed updates.

This changes the update convention. Take a scalar memory \(M_w(1)=w\), initial weight 0, target sequence 1 then 2, rate 0.5, and no momentum or decay:

| Method | First gradient → weight | Second gradient → weight |
| --- | --- | --- |
| Recompute at the current weights | −1 → 0.5 | \(0.5-2=-1.5\) → 1.25 |
| Both gradients at the chunk anchor 0 | −1 → 0.5 | \(0-2=-2\) → 1.5 |

The difference is where the second gradient is evaluated. Both calculations use a running weight state; one uses a stale anchor when determining the descent direction. Chunk size 1 recovers the sequential rule, and a zero rate makes both leave the weight unchanged.

**Investigation — move the gradient anchor.** Edit the two targets and learning rate, inspect the ordering of the final weights, then compare the two dependency diagrams. Change chunk size from 2 to 1 and explain which edge is restored. Equal outputs on a special fixture do not make the update rules identical: create a target pair that exposes their difference.

With fixed gradient inputs \(u_t\), momentum obeys an affine recurrence \(S_t=\eta_t S_{t-1}-\theta_t u_t\). Two transformations \(F_1(s)=a_1s+b_1\) and \(F_2(s)=a_2s+b_2\) compose as

\[
F_2(F_1(s))=(a_2a_1)s+(a_2b_1+b_2).
\]

Function composition is associative, which permits a scan organized as a tree rather than a strictly serial chain. That identity explains the parallel opportunity. It does not make the original nonlinear gradient evaluations independent. The [Titans parallelization section](https://arxiv.org/html/2501.00663v1#S3.SS2) and [TTT mini-batch derivation](https://arxiv.org/html/2407.04620v1#S2.SS4) discuss this distinction.

Likewise, accumulating gradients over examples at fixed ordinary model weights is not the same operation as repeatedly changing those weights between examples. The next lesson makes that distinction operational in a conventional training loop, including uneven batch sizes and the last partial update.

## 9. Count the state you actually retain

**Deeper branch.** Separate shared model parameters, state for each active sequence, temporary workspaces and the graph retained for outer training. Their scaling can differ.

Let a bias-free memory have dimensions \(d\to H\to d\). It contains \(2dH\) fast weights. If both weights and momentum use four-byte float32, their combined storage is \(2(2dH)4\) bytes per memory instance. With \(d=2048\) and \(H=512\), that is 16 MiB per layer. With 24 such layers and 4 independent sequences, it becomes 1.5 GiB.

For an attention cache with batch \(B\), attention layers \(L_a\), retained positions \(T_r\), KV heads \(h_{kv}\), head dimension \(d_h\) and \(b\) bytes per value,

\[
\text{KV bytes}=2B L_a T_r h_{kv}d_h b.
\]

The leading 2 accounts for keys and values. Use the number of KV heads, not automatically the number of query heads. A 2048-position window with batch 4, 24 attention layers, 8 KV heads, head dimension 128 and two-byte elements uses 0.75 GiB. Retaining 131072 positions under the same assumptions instead uses 48 GiB. These are exact payload calculations for a hypothetical configuration, not the memory footprint of a named released model.

**Separate storage bands.** The 1.5 GiB fast-weight/momentum band sits beside the 0.75 GiB local-KV band. Shared base weights, persistent parameters and workspace are separate categories with unspecified sizes. Increasing batch duplicates sequence state; increasing the window changes only the KV term in this calculation. The two calculated bands total 2.25 GiB; a complete device requirement also includes those other categories.

For fixed dimensions and a fixed window, a forward read and local write cost a bounded amount per position; total work grows with sequence length. A two-layer memory read has work proportional to \(dH\), and its gradient adds work on the same dimensions. Local attention adds a window-dependent term. Projection, feedforward, normalization and other architecture costs still exist. An outer training pass can retain activations or reconstruct them, so its peak memory does not follow solely from the inference-state formula.

Wall-clock latency also depends on chunking, tensor layout, batch size, precision, kernels, transfers and synchronization. Multiplying FLOPs by a peak hardware specification is not a measured runtime. Benchmark the actual causal update convention, context distribution and serving workload, including quality at the required retrieval distances.

## 10. Read the research without turning it into a promise

**Deeper branch.** The canonical source is [Behrouz, Zhong and Mirrokni, *Titans: Learning to Memorize at Test Time*](https://arxiv.org/html/2501.00663v1). Its arXiv submission is 31 December 2024; Google's publication entry labels it 2025. The paper studies memory design, integration, language modeling, retrieval/reasoning, forecasting, DNA modeling, efficiency and ablations. It is useful to read those as separate questions.

For example, the paper's Table 2 reports single-needle retrieval at 2K, 4K, 8K and 16K. Its longer-context BABILong results concern a different benchmark and include different fine-tuning/few-shot settings. Combining them into a smooth invented “accuracy up to 2M” curve would conceal those distinctions. A model that processes a long input has not thereby demonstrated that every fact in it remains recoverable.

Depth and component ablations ask whether changes help under the reported setup. They can motivate a new experiment, while the nonlinear-memory proof above explains a representational possibility independently of those scores. The paper also makes a theoretical expressivity claim about state tracking; this lesson does not supply its complexity-theory proof. It should not be recast as a guarantee that a trained model solves every difficult reasoning task.

The authors' [December 2025 Titans/MIRAS overview](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) introduces a useful research lens: choose a memory structure, its learning objective, its retention rule and its update algorithm. This gives you a way to classify an experiment. Replacing squared loss with a robust loss changes what errors drive writes; changing retention alters what survives; changing the memory network alters the functions it can represent. Those are distinct interventions. The overview is a conceptual companion; the precise equations and experimental conditions should be checked in the underlying papers.

### A less obvious application: a changing visual stream

The TTT research line also explores adaptation for vision. In [Sun and colleagues' 2020 project](https://yueatsprograms.github.io/ttt/home.html), the visible test input supplies a self-supervised task such as predicting an applied image rotation. Shared features are updated using that auxiliary task before the main prediction, without revealing the main class label. It demonstrates why “learning at test time” need not mean looking at a test answer.

Map the roles carefully: the observed image provides the auxiliary target; feature weights adapt; the classification head then predicts the withheld class. For an online stream, carrying state forward additionally assumes that consecutive inputs have a relationship worth exploiting. A sudden camera change could make that assumption less helpful. The natural investigation is to compare reset and carry-forward policies under a declared sequence of shifts, with the same starting model and without using class labels for adaptation.

That application differs from our supervised rental replay and from a Titans key/value memory. Putting them next to each other helps isolate a general design question: **what target is available under the task protocol at the moment a model updates, and how does improving that target help the task we care about?**

### Choose an approach by the work it must do

For exact document provenance, an external retrieval system can retain text, source identifiers and dates. A fast-weight memory instead returns a learned compressed representation. For ordered streaming patterns, an adaptive state can incorporate each observation as it arrives. Recent attention helps address individual accessible positions. These approaches can be combined; “ordered input” is not a reason that retrieval is inherently impossible.

Before choosing a design, define the question, target availability, necessary evidence, tolerated error, reset boundary, expected sequence lengths and resource budget. Then compare an existing simple baseline with the proposed method. An architecture diagram alone cannot determine which released system satisfies those requirements. This lesson does not infer unpublished product internals or make current checkpoint-availability claims from a research paper.

## 11. Diagnose the mechanism you actually ran

| Observation | Useful next comparison | What it can distinguish |
| --- | --- | --- |
| High loss, tiny gradient | Inspect keys and parameter sensitivities; try the zero-key control | A mismatch can be large in output space yet hard to affect through the chosen parameters |
| Weights change despite zero new gradient | Log old/new momentum and decay separately | Carrying an earlier update versus a new gradient-driven write |
| Inner loss improves, task score worsens | Hold the sequence fixed and compare query/task outputs before and after writes | Fitting associations versus improving the outer task |
| Reading in chunks changes outputs | Compare complete state carry and the gradient-anchor convention | Lost state versus a different algorithm |
| A new request depends on an earlier one | Compare isolated initial states with deliberately shared state | An unintended cross-request dependency |
| Larger magnitudes or nonfinite values | Inspect the first offending input, residual, gradient and update | Scale, update, data or arithmetic problems that deserve different repairs |
| A forecast becomes perfect after adaptation | Check when the target first entered the computation | Forecasting versus scoring a fitted target |

Do not jump from a symptom to a unique cause. For example, reducing a rate may prevent an immediate numerical failure while leaving a target-leakage bug intact. A useful debugging record names the hypothesis, one controlled change, what was held fixed and the observation that would distinguish explanations.

## 12. Practice: carry the idea to a changed case

### 1. An old association moves

Start from \(w=(2,-1)\), write key \((1,2)\) → value 3 using half-squared loss, rate 0.1 and no momentum/decay. Compute the new weights and the change in the answer to query \((2,1)\). Verify the change once by direct readout and once by the overlap identity.

<details><summary>Hint</summary>

Compute the prediction at the write key first. The query is a separate vector; use it only when calculating the old and new reads or the overlap.

</details>
<details><summary>Solution</summary>

The write prediction is \(2-2=0\), residual −3 and gradient \((-3,-6)\). New weights are \((2.3,-0.4)\). The query read changes from \(4-1=3\) to \(4.6-0.4=4.2\), an increase of 1.2. The overlap is \((1,2)^\top(2,1)=4\), so \(-0.1(-3)(4)=1.2\) gives the same change. The key tells the memory what to fit; the query exposes a consequence elsewhere.

</details>

### 2. Is decay a reset?

For a scalar weight, let \(w=3\), old momentum 0.4, new gradient 0, momentum retention 0.5 and \(\alpha=1\). What is the new weight? Specify a true fresh-sequence reset when the chosen initial weight is 0.7.

<details><summary>Hint</summary>

Calculate momentum before applying the weight recurrence. Distinguish the zero vector from the configured initial state.

</details>
<details><summary>Solution</summary>

New momentum is 0.2; new weight is \(0\cdot3+0.2=0.2\). A fresh sequence restores the weight to 0.7, momentum to 0, and clears its local context and other sequence state. No claim about the content being forgotten follows solely from a single gate value.

</details>

### 3. Change the chunk rule

Use initial scalar weight 1, keys 1, targets 3 then −1, rate 0.25 and no momentum/decay. Calculate the final weight with current-state gradients and with both gradients at the starting anchor. What happens if the chunk size is 1?

<details><summary>Hint</summary>

The first step agrees. For the second gradient, explicitly write the weight at which the residual is evaluated.

</details>
<details><summary>Solution</summary>

First gradient is −2, giving 1.5. Sequentially the next gradient is \(1.5-(-1)=2.5\), giving \(1.5-0.25(2.5)=0.875\). With anchor 1, the next gradient is 2, giving 1.0. Chunk size 1 refreshes the anchor before each gradient and yields 0.875. Both methods received the same observations; their descent directions differed.

</details>

### 4. Judge the rental result

Use the assessment results to answer: did adaptation help the seed 7 network, and would you choose it over the previous-day baseline on MAE alone? Design a new comparison that could justify a more expensive predictor without reusing the same assessment interval for unlimited model selection.

<details><summary>Hint</summary>

Those are two different comparisons. State the decision criterion before proposing a tuning experiment.

</details>
<details><summary>Solution</summary>

Seed 7's adaptive MAE 867.34 improves greatly over its frozen MAE 1781.81, but remains above the previous-day baseline 843.81. On these assessment MAEs alone, select the simple baseline. A follow-up could predeclare a later dated holdout, choose features/rates on earlier rolling development intervals, and compare the locked procedure's MAE, large-error behavior and compute cost on the later interval. The retained data end in 2012, so a genuinely later real holdout would require additional data or an explicitly redesigned study; it cannot be invented by relabeling the already inspected results.

</details>

### 5. Learn the write rate

Start with scalar weight 0. Write key 1 → value 2 with rate \(\theta\), then query at 2 with outer target 3. Derive the outer loss and its rate derivative. Evaluate them at \(\theta=0.5\).

<details><summary>Hint</summary>

Express the new weight as a function of the rate before substituting the number. Otherwise you can accidentally discard the dependency you need to differentiate.

</details>
<details><summary>Solution</summary>

Inner gradient −2 gives \(w'=2\theta\); query output is \(4\theta\). Outer loss is \(\tfrac12(4\theta-3)^2\), derivative \(4(4\theta-3)\). At 0.5, prediction 2, loss 0.5 and derivative −4. A small increase in the write rate decreases this outer loss locally. A program that detaches the updated weight before computing the outer objective would lose this rate path.

</details>

### 6. A serving-state calculation

A hypothetical bias-free memory has dimensions 512→128→512. Its weights and momentum are float32. There are 12 memory layers and 8 independent requests. Calculate their combined payload, then name at least three missing categories before declaring whether the system fits on a device.

<details><summary>Hint</summary>

Count two weight matrices, a momentum copy, bytes per entry, layers and requests. Divide by \(2^{20}\) for MiB.

</details>
<details><summary>Solution</summary>

Each layer/request has \(2(512)(128)=131072\) weights. Weights plus momentum take \(131072\times2\times4=1048576\) bytes, exactly 1 MiB. Across 12 layers and 8 requests this is 96 MiB. Missing categories include shared base-model weights, attention K/V, convolution and position/chunk state, temporary workspace, and training activations if doing an outer training pass. Dtypes can differ between categories, so use the actual ones in a complete calculation.

</details>

### 7. Explain a delayed reversal

Momentum is at its steady state −0.5 under constant gradient 1, rate 0.05 and retention 0.9. The gradient becomes −1. After how many steps does momentum first become positive? Why is that different from reaching its new steady state?

<details><summary>Hint</summary>

The new steady state is 0.5. Solve the recurrence for the distance from that value.

</details>
<details><summary>Solution</summary>

After n new steps, \(S_n=0.5-0.9^n\). It becomes positive when \(0.9^n<0.5\), first at n=7. The distance to the new steady state is \(0.9^n\); it approaches zero asymptotically and requires a declared tolerance for an approximate stopping time. A timescale such as \(1/(1-0.9)=10\) is not the exact sign-reversal answer.

</details>

### 8. Repair a plausible architecture explanation

Someone says: “We computed every token's memory read from the initial weights, then wrote the whole sequence afterward. Our attention is causal, so the memory has learned the earlier tokens for every output.” Identify the error and propose a controlled check. Then explain the different error in reading every position from the final post-sequence weights.

<details><summary>Hint</summary>

Ask which exact memory version produced the read at position t, independently of the attention mask.

</details>
<details><summary>Solution</summary>

All initial-state reads miss the adaptation from earlier tokens in this call. Reads must use the state prescribed by the declared online or chunk update rule. Compare a whole-call run with a continuation run that carries identical state across a boundary; they should agree when the convention agrees. Using the final post-sequence weights for earlier positions creates the opposite problem: those states can contain future writes. Perturb a future input and check earlier outputs. The attention mask cannot repair either memory-path error.

</details>

## Ready to move on?

Explain where the three memories store information; perform a write and predict a changed query; distinguish loss, gradient and actual update; identify the inner and outer targets; and justify when a target becomes available. Then reproduce one controlled state comparison and interpret a real result against its simple baseline. These are stronger signs of understanding than remembering the three architecture abbreviations.

Continue to [Mini-Batches, Training Loops & Gradient Accumulation](/learn/path/full-curriculum/mini-batches-training-loops-gradient-accumulation?module=deep-learning-fundamentals). It follows this topic in the module and turns the distinction between computing gradients and applying updates into a complete practical loop. The following diagnostics lesson then teaches how to test that loop's behavior.

## References and another way to learn it

- [Behrouz, Zhong and Mirrokni: Titans](https://arxiv.org/html/2501.00663v1). Primary paper; begin with §§ 3.1 and 4 after the hand example, then read § 3.2 and Appendix C for update conventions. The actual agenda and those sections were read, together with the experiment sections. Mathematical notation sometimes compresses implementation details; use the timing distinctions in this lesson when translating it to code.
- [Sun and colleagues: Learning to (Learn at Test Time)](https://arxiv.org/html/2407.04620v1). Primary paper with a helpful alternative inner/outer-loop explanation and mini-batch derivation. Read §§ 2.1–2.4 after § 6 here. The relevant mechanism, training-view and chunk sections were inspected; this packet does not claim to reproduce the language-model experiments.
- [Google Research: Titans + MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/). An illustrated article by the researchers, dated 4 December 2025, for revisiting the high-level design choices. Its conceptual sections were reviewed. Read the precise paper equations for the update rather than interpreting every everyday “surprise” example literally.
- [Sun and colleagues' 2020 TTT project and companion explanation](https://yueatsprograms.github.io/ttt/home.html), with its [ICML talk](https://www.youtube.com/watch?v=NbuWxmMco30). A video alternative about test-time self-supervision and distribution shifts, not a Titans tutorial. The project's substantive introduction/method and official talk link were inspected; the video was not watched, and no timestamps are asserted. Useful after the target-availability discussion in § 7.
- [PyTorch autograd.grad](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html) and [no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html). Versioned API references for implementing the two graph-lifetime modes in § 6. Relevant function semantics were read; the supplied programs were actually executed in the environment recorded with their outputs.
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Hadi Fanaee-T, 2013, [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894). Dataset and descriptive context for the real application. Use the supplied daily CSV and provenance to reproduce this lesson's exact data input.

Research and author calculations were checked on 13 September 2026. The interactive investigations and inline illustrations described here are specified for the subsequent website implementation; the offline programs and recorded results are already supplied.
