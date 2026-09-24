# State Space Models: S4 and the Mamba Family

**Explore as you read.** Edit tiny system coefficients, impulse inputs, selective writes, distraction sequence, chunk boundaries and supported real trajectories. Show impulse response, carried state, input-conditioned updates, SSD matrix entries and chunk equivalence live. Stepping exposes current recurrence arithmetic. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to decide which information needs selection or state carry and distinguish a mathematically equal scan from a different update rule.


A hand moves through a curved path. You receive one coordinate pair at a time. Keeping only the newest pair loses the path; keeping every pair forever makes memory grow with the recording. A state-space sequence model maintains a third kind of representation: a fixed collection of numbers that changes as measurements arrive.

Those numbers are its **state**. Their update rule determines what fades, what accumulates, what oscillates and what survives a distraction. Learning chooses the update and readout parameters so that this evolving summary helps a task.

In [Long-Context Sequence Models](/learn/path/full-curriculum/long-context-sequence-models-transformer-xl-griffin-perceiver?module=deep-learning-fundamentals), we compared a retained attention cache, recurrent memory and latent compression. Here we open the recurrent mechanism. The later [RWKV lesson](/learn/path/full-curriculum/rwkv-linear-attention-models?module=deep-learning-fundamentals) will construct a different recurrent memory from weighted key–value accumulation.

**A route through the lesson.** First calculate a small state update. Then see when the same computation can be expressed as a convolution, why useful memory needs several modes, how Mamba makes the dynamics depend on the input, and how Mamba-2 groups the calculation into matrix operations. Finally train two modest sequence classifiers on real movement data and interpret their limitations. The detailed HiPPO/S4 derivation, S5 and Mamba-3 are optional branches after the core calculation.

You should finish able to choose and explain a sequence operator, check its recurrence against another computation, train a small model under a defensible protocol, and separate a memory-count argument from an empirical performance claim.

## 1. A state is a summary with a rule for changing it

Imagine a temperature sensor whose reading jumps around. Let uₜ be its current reading, and let hₜ be a smoothed estimate:

hₜ = 0.8 hₜ₋₁ + 0.2 uₜ.

Starting at h₋₁=0, inputs 5,0,0 produce states 1,0.8,0.64. The first reading remains influential, but its contribution decays. A state value is not necessarily a stored measurement; it can mix many previous measurements.

A single average is insufficient to distinguish a steady motion from an oscillation. We therefore use a vector of state coordinates, with several update rates and interactions. A linear discrete-time state-space layer is

hₜ = Āhₜ₋₁ + B̄uₜ,  
yₜ = Chₜ + Duₜ.

Here h has N coordinates, u has H input channels and y has J output channels. Ā is N×N, B̄ is N×H, C is J×N and D is J×H. Multiplying by B̄ writes the input into memory; Ā advances existing memory; C reads it; D provides a direct path from the current input.

Throughout this lesson, **the state is updated before it is read**. The initial state is h₋₁. Other references use an output-before-update convention; their first kernel tap can differ by one index without the underlying idea being different.

**Inline figure: four paths through one time step.** Show the old-state arrow passing through Ā, the current-input arrow through B̄, their sum forming hₜ, then C producing the memory contribution. A separate D arrow joins only at the output. Put dimensions beside every arrow. This picture should make it impossible to count the direct path twice.

For a scalar example, take Ā=.5, B̄=1, C=2 and D=.25. From initial state 1 and input 3, the new state is 3.5 and the output is 7.75. The two output contributions are 7 from memory and .75 directly from the input.

A **linear time-invariant** operator, abbreviated LTI, has fixed matrices and obeys linearity with zero initial state: processing a sum of inputs equals the sum of the processed inputs. “Time-invariant” means the same input pattern is treated by the same rule wherever it appears, subject to the sequence boundary. Input-dependent matrices generally break linearity and prevent one fixed convolution kernel from representing the map. A nonlinear operator can still obey time-shift symmetry.

State-space modeling is a broader term than this linear layer: nonlinear physical models and nonlinear state transitions also belong to the family. S4 and Mamba use particular structured linear-in-state updates inside learned nonlinear networks.

### What this buys, and what it does not

A recurrent evaluation retains N state numbers, however long the preceding stream was. That does not mean it remembers every detail exactly. Distinct histories can yield the same finite representation. If a future task asks for an arbitrary old token verbatim, a compressed state can face a different tradeoff from an attention cache that explicitly retains token representations.

The question is therefore: **which information should the state preserve for the task?** S4 supplies useful structured memory modes. Mamba allows the current content to affect what is written, retained and read.

## 2. From continuous change to sampled updates

A physical process can evolve between measurements. A continuous linear system writes

dh(t)/dt = Ah(t) + Bu(t),  
y(t) = Ch(t) + Du(t).

A is now a rate-of-change matrix. If time is measured in seconds, its decay rates have inverse-second units. Ā instead describes one discrete step. Confusing these two matrices can silently change an entire model.

For constant input over an interval of length Δ, exact integration gives

Ā = exp(ΔA),  
B̄ = integral from 0 to Δ of exp(sA)B ds.

Thus hₜ=Āhₜ₋₁+B̄uₜ describes the state at the end of that held-input interval. This is **zero-order hold**, or ZOH: the input is held constant while the state evolves. Matrix exp is a matrix function, not elementwise exponentiation except for a diagonal matrix.

For the scalar decay A=−1 and B=1,

Ā=e^(−Δ),  B̄=1−e^(−Δ).

At Δ=ln 2, the old state and held input each receive weight .5. A longer interval permits more decay and more approach toward the held input. Changing Δ changes both terms, not just the forget factor.

### A singular matrix is a normal case

One sometimes sees B̄=A⁻¹(exp(ΔA)−I)B. It is valid when A is invertible. An integrator has A=0, however, and is perfectly meaningful:

dh/dt=u,  so hₜ=hₜ₋₁+Δuₜ.

With Δ=.5, initial state 3 and inputs 2,−1, the states are 4 and 3.5. No inverse is needed.

The downloadable program obtains Ā and B̄ together from the block exponential

exp(Δ [[A,B],[0,0]]) = [[Ā,B̄],[0,I]].

This works for the integrator and for coupled multidimensional systems. For very small scalar or diagonal arguments, expm1(z)=exp(z)−1 avoids subtracting two nearly equal floating-point numbers.

### S4's original choice: the bilinear transform

The original S4 formulation uses the **bilinear**, or trapezoidal, discretization:

Ā = (I−ΔA/2)⁻¹(I+ΔA/2),  
B̄ = (I−ΔA/2)⁻¹ ΔB.

In code, solve these linear systems instead of explicitly forming inverses. ZOH is exact for a held input; the bilinear rule is a different discretization with useful stability properties. For A=−1, B=1 and Δ=1, ZOH gives Ā≈.367879 and B̄≈.632121; bilinear gives 1/3 and 2/3. Both are legitimate choices. They are not numerically identical.

For continuous dynamics whose eigenvalues lie in the left half-plane, the bilinear map places the corresponding discrete eigenvalues inside the unit disk. It does not follow that every arbitrary learned parameterization or complete nonlinear network is automatically numerically well behaved.

**Inline figure: continuous decay and sampled points.** Draw e^(−t) for an initial impulse-free state 1, show exact ZOH samples and bilinear samples on the same time axis, and identify Δ. Use computed values, not a hand-drawn “stable versus unstable” claim. Beside it, show the A=0 integrator as accumulation rather than decay.

The distinction matters for sensors with known sampling intervals. For token models, Δ is usually a learned internal quantity; there is no reason to call it elapsed physical time. Some modern sequence layers parameterize the discrete recurrence directly. Continuous-time language is one useful construction, not a requirement that every token model simulate a physical differential equation. [S4, §2.2](https://arxiv.org/pdf/2111.00396).

## 3. One operator, three ways to compute it

Expand a fixed-matrix recurrence from zero initial state:

h₀=B̄u₀,  
h₁=ĀB̄u₀+B̄u₁,  
h₂=Ā²B̄u₀+ĀB̄u₁+B̄u₂.

After applying C, the contribution of an input depends only on how many steps ago it arrived. Define the kernel taps

Kₗ = C Āˡ B̄,  for l=0,1,2,...

Then

yₜ = sum over j=0…t of Kₜ₋ⱼ uⱼ + Duₜ.

This is a **causal convolution**: each output combines the present and past using lag-dependent weights. With a nonzero initial state, add C Ā^(t+1) h₋₁. Omitting that term changes the problem.

The continuous-time analogue is equally precise. With initial state h(0), the output is

y(t)=C exp(At)h(0) + integral from 0 to t of C exp(A(t−s))B u(s) ds + Du(t).

The strictly proper impulse response is g(t)=C exp(At)B. If you choose instead to include Dδ(t) in the impulse-response distribution, its convolution already supplies the direct path; do not add Du a second time.

### A calculation you can reconcile by hand

Use two independent continuous modes with A=diag(−1,−2), B=[1,1]ᵀ, C=[1,−.5], D=.25 and Δ=ln 2. ZOH yields

Ā=diag(.5,.25), B̄=[.5,.375]ᵀ.

The first four kernel taps are

[.3125, .203125, .11328125, .0595703125].

For inputs [2,0,1,0] and zero initial state, the output is

[1.125, .40625, .7890625, .322265625].

Check the third output: 2×.11328125 + 1×.3125 + .25×1 = .7890625. Its three terms are the old input's remaining contribution, the current input's memory contribution and direct feedthrough.

**Inline figure: an impulse ledger.** Show each input as a row whose contribution spreads rightward according to the kernel. A column sum produces one output. Align it with the recurrent state trace so the learner can match “one compact state” to “many historical contributions.”

The same result can be calculated in three ways:

| Evaluation | Main operation | Useful setting |
|---|---|---|
| Recurrence | Update and retain state one step at a time | Streaming or autoregressive decoding |
| Direct convolution | Sum lagged input contributions | Small transparent calculations |
| FFT convolution | Transform input and kernel, multiply, inverse-transform | Full fixed-kernel sequences |

The FFT computes circular convolution unless padding is correct. With T input samples and K retained taps, use a transform length at least T+K−1, then retain the causal outputs needed. Padding to T and hoping for the best can wrap a future tail into the beginning.

A dense Ā costs O(N²) per recurrent step. A diagonal Ā costs O(N). Computing every diagonal kernel tap naively costs O(NT), followed by roughly O(T log T) FFT work. S4's special kernel-generation algorithm addresses the kernel construction too. Saying “the model uses an FFT” does not account for every operation.

### Investigation: change a system, then reconcile its computations

Choose an input pulse or edit its signed sample heights; adjust a decay rate, initial state or direct path. Before running, record which outputs you expect to change and whether recurrent and convolutional results should agree.

After running, inspect the state plane, kernel and contribution ledger. Try the singular integrator and the feedthrough-only setting C=0,D=1. Then change only the final input and inspect earlier outputs. Explain every difference by a legal information path.

The comparison must include the initial-state response in the convolutional view. A disagreement caused by leaving it out is a useful diagnosis, not evidence that recurrence and convolution are different models.

## 4. Designing a memory with more than one timescale

A mode with discrete decay .99 retains information much longer than one with decay .2. After k empty updates, a contribution is multiplied by aᵏ. For 0<a<1, its half-life is ln(.5)/ln(a) steps. Half-life describes one mode's attenuation, not the guaranteed recall span of an entire trained network.

Several real decays provide several smoothing rates. Oscillatory modes add sensitivity to patterns that alternate or repeat. Consider

A = [[−.2,−2],[2,−.2]].

With no input, its state rotates at 2 radians per time unit while its radius decays as e^(−.2t). At Δ=.25, each step rotates by .5 radians and shrinks the radius by e^(−.05). The state does not merely get “older”; its direction changes.

**Inline figure: a shrinking spiral synchronized with two coordinate traces.** Mark each discrete sample on the spiral and plot its horizontal and vertical coordinates below. This turns a complex eigenvalue into an observable rotating pair of real numbers.

### A small diagonal state-space layer

A diagonal model can use complex-valued modes aₙ=−exp(αₙ)+iωₙ. The negative real part provides decaying continuous modes; ω controls rotation. For a real input and a real output, include conjugate pairs. One stored half of each pair contributes

2 Re(cₙ hₙ)

to the readout. The factor 2 and real part are part of the model, not cosmetic plotting choices.

Our small training program stores four complex modes per channel and their implicit conjugates. It learns decay, frequency, step size, complex readout and a direct skip; it fixes the continuous B to 1 and calculates its discrete B̄ exactly. It evaluates the temporal operation both recurrently and through a generated convolution kernel.

This is an **S4D-style teaching layer** with an explicitly chosen linear-frequency initialization. It is not a full implementation of S4's diagonal-plus-low-rank kernel algorithm. The distinction lets us teach and test a complete useful model without silently attaching the wrong name to it. [S4D, §§3–4](https://arxiv.org/pdf/2206.11893).

### Optional deeper branch: why HiPPO influenced S4

Suppose we want to summarize a function's entire observed history by polynomial coefficients. On the interval [0,t], use the normalized measure ds/t and an orthonormal polynomial basis. The first two normalized basis functions are 1 and √3(2s/t−1).

A coefficient is the weighted inner product cₙ=integral from 0 to t of f(s)pₙ(s) ds/t. Multiplying the history by a basis function and averaging extracts how strongly that shape is present. For f(s)=s, integrate s/t for the first coefficient and s√3(2s/t−1)/t for the second. The results are

c₀=t/2,  c₁=t√3/6.

At t=2, the coefficients are 1 and √3/3. Reconstructing c₀+c₁√3(s−1) gives s exactly because a linear function lies in the span of these two basis functions.

The first coefficient represents a level; the second represents a trend. Higher-degree coefficients retain more detailed shape. This is a concrete meaning of “memory basis.”

**Inline figure: history, basis functions and reconstruction.** Overlay f(s) with its one-coefficient and two-coefficient reconstructions, while separate small axes show the two basis functions and their weighted contributions. Label the measure and interval. Do not present a generic matrix heatmap as sufficient explanation of the projection.

HiPPO derives online coefficient updates for particular history-weighting measures. For the scaled Legendre construction, the exact growing-interval dynamics have the form

dc/dt = −A₊c/t + B₊f(t)/t,

where, with indices n,k starting at zero,

(A₊)ₙₖ = √((2n+1)(2k+1)) if n>k;  
(A₊)ₙₙ = n+1;  
(A₊)ₙₖ = 0 if n<k;  
(B₊)ₙ = √(2n+1).

The 1/t factors matter. A fixed LTI S4 initialization inspired by this matrix is not literally the same as the time-varying projection over an ever-growing interval. “Optimal” in the projection result means optimal approximation in the specified basis and weighted squared-error criterion, not universally optimal memory for every task. [HiPPO, §§2–3](https://arxiv.org/pdf/2008.07669).

S4 uses a structured starting point and a numerically useful representation. Let A=−A₊ and pₙ=√(n+.5). Then A+ppᵀ is a normal matrix; in this real case its symmetric part is −I/2. A normal matrix has an orthonormal eigenbasis. After a unitary change of basis, the original matrix can therefore be written as a diagonal matrix minus a low-rank correction.

This is **diagonal plus low rank**, or DPLR. It avoids treating a poorly conditioned eigenvector decomposition of the original nonnormal triangular matrix as numerically harmless.

The remaining computational idea is to evaluate the kernel's generating function at suitable frequency points. For T taps, define K_T(z)=sum from l=0 to T−1 of K_l z^l. The finite geometric-series identity gives

K_T(z)=C {I−(zĀ)^T} (I−zĀ)⁻¹ B̄

where the inverse exists. This is a scalar-valued function for one input/output channel; evaluating it at Fourier points gives the transform of the finite tap sequence. The factor I−(zĀ)^T is the finite-length correction.

For a continuous DPLR matrix A=Λ−pq*, let R₀(s)=diag(1/(s−λₙ)). The star denotes conjugate transpose. Woodbury gives

(sI−A)⁻¹ = R₀−R₀p(1+q*R₀p)⁻¹q*R₀.

The apparently large inverse is reduced to diagonal operations and a scalar correction in this rank-one case. Terms such as q*R₀p are sums of weighted 1/(s−λₙ) factors, explaining the Cauchy structure. S4 combines this idea with its discretization to evaluate the finite kernel efficiently. Resolvents of a diagonal matrix are easy; the Woodbury identity handles the low-rank correction. The resulting sums have a Cauchy-like structure, which specialized algorithms exploit, and a transform recovers the time-domain taps. That is the reason for S4's mathematical machinery: retain useful structured dynamics while making a long kernel practical to calculate.

An advanced implementation should follow the paper's finite-length correction, discretization and stability conventions together. Copying only its A matrix into a naïve dense recurrence does not reproduce the kernel algorithm. S4D investigates which benefits survive a diagonal simplification and carefully chosen initialization; its approximation results do not say that a small finite diagonal model equals the full HiPPO system exactly. [S4, §§3.1–3.4](https://arxiv.org/pdf/2111.00396).

## 5. Mamba: let content change the write, retention and read

An LTI operator can copy a fixed delay perfectly. A three-state shift register can store the current input, the previous input and the input before that. Reading the third coordinate sends [2,5,−1,7,0] to [0,0,2,5,−1]. Its convolution kernel is a single pulse at lag 2.

The harder task is to retain a marked item while an unpredictable number of irrelevant items arrive. A fixed temporal kernel applies the same lag weights regardless of which item was marked. A nonlinear deep S4 network is more than one LTI operator, but input-dependent selection gives the temporal update itself a direct way to respond.

A simple selective update is

hₜ=(1−gₜ)hₜ₋₁+gₜuₜ,  0≤gₜ≤1.

Use inputs [4,9,−7,6] with gates [.99,.01,.01,.99]. The states are

[3.96,4.0104,3.900296,5.97900296].

The first and last items substantially replace the state; the middle items have little effect. With a constant gate .5, the states are [2,5.5,−.75,2.625]. These gates are supplied teaching controls. A trained network must learn how to compute useful gates from available inputs.

The scalar gate has an exact state-space connection. For A=−1, B=1 and Δₜ=softplus(zₜ), exact ZOH gives

exp(−Δₜ)=1−sigmoid(zₜ),  
B̄ₜ=1−exp(−Δₜ)=sigmoid(zₜ).

Thus gₜ=sigmoid(zₜ). Softplus is log(1+exp(z)); sigmoid is 1/(1+exp(−z)). Substitute these definitions to verify the identity. A large Δ both erases more old state and increases the new write in this scalar construction.

**Inline figure: annotated write and forget bands.** Above an editable signed input sequence, show g and 1−g as complementary bands. Below, decompose the next state into retained old contribution and incoming contribution. An “important” badge alone cannot explain which quantity a gate changes.

### Mamba's selective state-space operator

For a batch of B sequences of length T with D internal channels and N state coordinates per channel, a common Mamba-1 organization uses:

| Quantity | Shape | Role |
|---|---|---|
| Input u | B×T×D | Features being processed |
| A | D×N, diagonal within each channel's state | Learned base decay rates |
| Δ | B×T×D | Input-dependent step parameters |
| Bₜ and Cₜ | B×T×N | Input-dependent write/read vectors, shared across channels in this organization |
| State | B×D×N | Memory retained during recurrent evaluation |

The original gating derivation uses exact ZOH. The reference implementation's selective scan uses exp(ΔA) for decay and ΔB for input injection:

hₜ,d,n = exp(Δₜ,d A_d,n) hₜ₋₁,d,n + Δₜ,d Bₜ,n uₜ,d,  
yₜ,d = sum over n of Cₜ,n hₜ,d,n + D_d uₜ,d.

That injection is a specific parameterization; it is not generally equal to exact ZOH B̄. For scalar A=−1, B=1 and Δ=1, exact ZOH injection is about .632121, while ΔB is 1. The difference is not necessarily small. Learn and implement the chosen operator consistently. [Mamba, §3.5 and appendix C](https://arxiv.org/pdf/2312.00752); [official selective-scan reference](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py).

Input-dependent B controls what is written, Δ controls the dynamics, and C controls what is read. The state is linear in its previous value when the current input-dependent parameters are fixed, but the full input-to-output map is generally nonlinear.

### The operator is not the entire block

A Mamba-1 block projects its input into an expanded feature branch and a gate branch. The feature branch passes through a short causal depthwise convolution and an activation, then supplies the selective operator. Its result is multiplied by an activated gate and projected back to the model width; residual connections and normalization organize the stack. The gate commonly uses SiLU(x)=x·sigmoid(x), so it is an activated multiplicative branch rather than a probability distribution.

The short convolution gives nearby positions a local interaction before parameter selection. The outside gate is different from Δ inside the recurrence. A diagram should draw both and label their equations, rather than call every multiplication “the forget gate.”

**Inline figure: a shape-aware Mamba block.** Show the branch split, local causal convolution, B/C/Δ projections, selective state update, separate SiLU gate, output projection and residual. A tooltip or side note should distinguish the full block from the simplified mixer used in our small experiment.

Because the coefficients vary with content, one fixed global convolution kernel no longer describes all inputs. Recurrence is still available. Moreover, the affine maps h↦a⊙h+b compose associatively:

(a₂,b₂) after (a₁,b₁) = (a₂⊙a₁, a₂⊙b₁+b₂).

Their coefficients can be computed from the input before scanning. A parallel prefix scan can therefore evaluate the sequence with logarithmic dependency depth, while a work-efficient implementation keeps total arithmetic proportional to sequence length for fixed state dimensions. Parallel does not mean every state can ignore earlier inputs; it means the same dependencies can be grouped.

The practical Mamba algorithm also fuses operations and recomputes selected intermediates in the backward pass to reduce memory traffic. It need not materialize the entire B×T×D×N state trajectory in device memory. Kernel details, precision and shapes determine the actual speed; a Python loop will not inherit fused-kernel throughput.

### Investigation: build a selective memory challenge

Create a signed sequence, mark the items that should replace memory, and edit gaps and distractors. Compare the constant gate and your input-dependent schedule live. The separate old-state and new-write contributions show which change improved retention and which suppressed a distraction.

Now make every input zero and start at zero. Can changing the gates alone create a nonzero state in this update? Then restore the signal and close the write gate while leaving decay active in a more general two-coefficient recurrence. Explain why “stop writing” and “stop forgetting” are distinct interventions.

## 6. Mamba-2 and state-space duality

Mamba-2 makes a specific restriction that enables a different computation. Within one head, the transition is a scalar aₜ times the identity. Let the state Sₜ have shape N×P, let bₜ,cₜ each have N coordinates, and let vₜ have P coordinates:

Sₜ = aₜ Sₜ₋₁ + bₜvₜᵀ,  
yₜ = cₜᵀSₜ.

The outer product bₜvₜᵀ writes an N×P matrix. Different heads can have different dynamics. Sharing a scalar decay inside a head restricts the operator compared with allowing an independent decay for every state coordinate, but the resulting structure is computationally useful.

Unroll from S₋₁=0:

yᵢ = sum over j≤i of (cᵢᵀbⱼ) Lᵢⱼ vⱼ,

where Lᵢⱼ is the product aⱼ₊₁aⱼ₊₂…aᵢ, and Lᵢᵢ=1. An empty product is 1 because the input written at i has not yet undergone a later decay.

Stack the c and b vectors as rows of C and B. The sequence operator is

Y = ((CBᵀ) ⊙ L)V.

The symbol ⊙ means elementwise multiplication. This is an exact equality with the recurrence just defined. It resembles attention: c is query-like, b key-like and v value-like, with a causal structured mask.

It is **not ordinary row-softmax attention**. The coefficients can be negative, need not sum to one, and contain no softmax normalization. The later [Self-Attention & Multi-Head Attention lesson](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) develops that different operator. The SSD connection is precise without saying every transformer block is the same recurrent model.

### A four-step matrix memory

Take N=P=2 and

| Step | a | b | c | v |
|---|---:|---|---|---|
| 0 | .5 | [1,0] | [1,0] | [2,1] |
| 1 | .5 | [0,1] | [1,1] | [3,−1] |
| 2 | .25 | [1,1] | [0,1] | [1,2] |
| 3 | .8 | [1,−1] | [1,2] | [−2,1] |

At step 0, S₀=[[2,1],[0,0]] and y₀=[2,1]. At step 1,

S₁=.5S₀ + [[0,0],[3,−1]] = [[1,.5],[3,−1]],

so y₁=[4,−.5]. Continuing gives y₂=[1.75,1.75] and y₃=[5.8,3.5]. The program independently calculates the recurrent, full matrix and chunked forms and checks that all agree.

**Inline figure: signed influence matrix beside the state.** Each row i shows which earlier values affect output i. Mark future entries as structurally absent, use a zero-centered scale for signed coefficients, and keep the decay matrix L visually distinct from the content factor CBᵀ. An attention-style “probability heatmap” would mislabel these numbers.

### Why grouping into chunks helps

Partition the sequence into chunks of q positions. An output has two sources: inputs inside its own chunk and memory arriving from earlier chunks.

The SSD calculation makes that decomposition explicit:

1. Compute each chunk's local outputs as if its incoming state were zero.
2. Compute the final state each chunk's own inputs would write.
3. Pass states between chunks using each chunk's total decay and own written state.
4. Read the incoming state at every position within the chunk and add that contribution to its local outputs.

For the example with q=2, the second chunk's local outputs are [1,2] and [4.4,3.8]. Its incoming-memory contributions are [.75,−.25] and [1.4,−.3]. Their sums recover [1.75,1.75] and [5.8,3.5].

**Inline figure: a block matrix with carried state bridges.** Highlight diagonal blocks for local interactions and low-rank factorizations for earlier-chunk effects. Synchronize a four-stage storyboard with the numeric local, carried and summed outputs. The boundary should visibly transmit information.

Dense matrix operations within bounded chunks can use hardware designed for matrix multiplication. A work-efficient recurrence or scan carries information between chunks. The creator's short explanatory implementation materializes a dense matrix even between chunks; its simplicity should not be mistaken for the asymptotic behavior of the optimized scan. Our transparent reference passes the chunk states serially and supports a final short chunk. [SSD/Mamba-2, §§5–7](https://arxiv.org/pdf/2405.21060); [creator's algorithm walkthrough](https://tridao.me/blog/2024/mamba2-part3-algorithm/).

Mamba-2 also changes the surrounding architecture, including parallel production of several SSM inputs and normalization/head organization. SSD is the mathematical operator and algorithmic framework; a full Mamba-2 network includes these additional choices.

### Investigation: can you change the chunking without changing the answer?

Edit individual entries in b,c or v, set a decay, and predict which outputs will change. Then choose chunk size 1,2,3 or the entire sequence. Compare the matrix view and the state bridges.

Setting a₂=0 erases the incoming state just before step 2; it does not erase the new step-2 write. Changing only v₃ must leave outputs 0–2 unchanged. Setting every b to zero with zero initial state produces zero output even when c and v vary. Explain these observations from the recurrence before relying on a green equality indicator.

## 7. Make the state useful by learning from real movement

A clever recurrence is not yet a classifier. We need an input representation, a prediction target, a loss, trainable parameters and an evaluation procedure.

We will classify **Libras movement trajectories**. The UCI dataset contains 360 recordings in 15 movement categories. Each record has 45 two-dimensional hand coordinates and a category. Examples include curved swing, circle, horizontal straight line and vertical zigzag. These are normalized trajectory coordinates derived from videos, not calibrated physical positions or complete sign-language conversations. [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement).

**Inline figure: three recorded paths, with time visible.** Draw actual source trajectories on equal-scale x/y axes. Number a few points and attach a start marker and directed segments. A curve without temporal order hides a quantity that sequence models can use.

The downloadable source has 30 repeated feature rows with consistent labels. We retain the first occurrence of each exact trajectory before partitioning, leaving 330 unique rows. Otherwise an identical input could land on both sides of the evaluation boundary.

A fixed classwise shuffle produces 220 fitting, 50 validation and 60 assessment trajectories. Every assessment class has four rows. The exact one-based source IDs are saved in the results. Performer and recording-session identifiers are absent, so this row-level protocol cannot establish performance on new performers or sessions.

### The complete prediction pipeline

Each trajectory is a 45×2 array. Convert a coordinate x in [0,1] to 2x−1. This fixed transformation uses no estimated corpus statistics. A linear projection turns each coordinate pair into 16 features.

Two residual blocks each perform

z ← z + W_out GELU(TemporalMixer(LayerNorm(z))).

Here W_out denotes the learned affine projection, including its bias. GELU is the smooth activation xΦ(x), where Φ is the standard-normal cumulative distribution function. Layer normalization and the output projection operate within each time step. The temporal mixer is what carries information across time. Average the resulting 45 feature vectors, then apply a 16-to-15 linear classifier.

The final 15 numbers are **logits**. Softmax converts them into model probabilities. For true class k, the loss is −log pₖ. A high probability for the wrong class incurs a large loss; a correct top-ranked class can still have a mediocre probability and a nonzero loss.

The recurrence parameters learn through the same computational graph as the projections. For an elementary readout example, hold state h=[1,2], target r=3 and C=[.5,.5]. The output is 1.5. With loss .5(Ch−r)², the gradient with respect to C is (Ch−r)h=[−1.5,−3]. A gradient step of .1 changes C to [.65,.8], output to 2.25 and loss from 1.125 to .28125. Backpropagation through an entire sequence extends this chain to writes, decays and earlier inputs.

**Inline figure: from one measured path to one loss.** Show 45×2 coordinates → 45×16 projected features → two temporal blocks → 16 pooled features → 15 logits → the probability assigned to the labeled category. Put the update arrows back to the actual trainable quantities, not to a mythical “memory quality” knob.

### The two temporal mixers we actually train

The first is the diagonal complex-mode layer from §4: four stored complex modes per channel, exact held-input discretization, real output from conjugate pairs, and fixed parameters across the sequence. It has 1,487 trainable parameters in the complete classifier.

The second is a simplified selective mixer with eight real state coordinates per channel. It projects the current normalized features into B,C and Δ, uses negative learned A, exp(ΔA) decay and ΔB injection, and evaluates a serial reference recurrence. Its complete classifier has 2,287 parameters.

The selective experiment omits the full Mamba block's local convolution and separate multiplicative gate. It isolates a trainable selective temporal mixer inside the same small residual scaffold. Labeling it a reproduced Mamba checkpoint would overstate what was implemented.

An ordered linear baseline flattens all 45 coordinate pairs into 90 features and fits regularized multinomial logistic regression with C=1. It has 1,365 fitted coefficients and intercepts. Unlike a mean-coordinate baseline, it can use the trajectory's order directly.

For each neural mixer we predeclare seeds 17 and 41, train 100 full-batch epochs with Adam at learning rate .003, and choose the epoch with the lowest validation cross-entropy. Assessment labels are not used to choose epochs. Both seeds are reported; they were not searched until a preferred architecture won.

### Run the small study

Keep these files together: [trajectory_state_models.py](trajectory_state_models.py), [movement_libras.data](movement_libras.data), and [movement_libras.names](movement_libras.names). The [provenance record](data-provenance.md) supplies attribution, license, transformations and row roles. The program is complete; its inputs are the retained local files and it makes no network request.

In a Python environment with NumPy, PyTorch and scikit-learn:

```text
python -m pip install numpy torch scikit-learn
python trajectory_state_models.py
```

The author run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1, two CPU threads and deterministic algorithms. Installation is for your chosen environment; a GPU extension is unnecessary for this teaching program. Training writes trajectory-results.json and trajectory-state-fits.npz beside the program.

The core selective step in that complete file is:

```python
# u: batch × time × width
# B and C: batch × time × state_size
# delta: batch × time × width; A: width × state_size
state = torch.zeros(
    (len(u), self.width, self.state_size),
    device=u.device, dtype=u.dtype,
)
outputs = []
for t in range(u.shape[1]):
    decay = torch.exp(delta[:, t, :, None] * A)
    write = (
        delta[:, t, :, None]
        * B[:, t, None, :]
        * u[:, t, :, None]
    )
    state = decay * state + write
    outputs.append(
        (state * C[:, t, None, :]).sum(-1)
        + self.skip * u[:, t]
    )
y = torch.stack(outputs, dim=1)
```

The singleton dimensions make the broadcasts explicit. Every batch member has its own state; B and C share their state vectors across width in this chosen parameterization. A fresh forward call starts a fresh sequence. The full file computes the coefficient projections, trains all parameters, selects validation checkpoints and records confusion matrices.

To reproduce the mathematical checks separately, place [state_space_mechanisms.py](state_space_mechanisms.py) beside the lesson files, install NumPy and SciPy, and run:

```text
python state_space_mechanisms.py
```

It writes mechanism-results.json with recurrence/direct/FFT agreement, the singular and initial-state cases, structured-memory calculations, SSD chunk decompositions and the advanced fixtures. These are computed mechanisms, not timing benchmarks.

### What happened in the recorded run

| Model | Seed | Selected epoch | Fitting errors / 220 | Validation errors / 50 | Assessment errors / 60 |
|---|---:|---:|---:|---:|---:|
| Ordered logistic baseline | — | — | 35 | 17 | 22 |
| Diagonal mixer | 17 | 100 | 59 | 25 | 30 |
| Diagonal mixer | 41 | 100 | 44 | 19 | 28 |
| Selective mixer | 17 | 58 | 65 | 25 | 30 |
| Selective mixer | 41 | 68 | 74 | 31 | 28 |

The ordered baseline made fewer assessment errors than either small neural model here. Both temporal mixers learned useful information, but neither outcome establishes an advantage from selectivity in this small protocol. The two kinds have different parameter counts and inductive biases; this is not a matched large-scale architecture comparison.

Validation cross-entropy and classification error need not choose the same epoch. The two selective runs were selected by cross-entropy, not by a retrospective choice of the most attractive table row. For the diagonal runs, the best validation epoch was the final allowed epoch; that invites a future training-budget study, but does not justify silently extending this one after seeing assessment results.

The saved result file includes all epoch losses and 15×15 confusion matrices. With four assessment rows per class, one changed prediction moves that class's recall by .25. Treat fine-grained per-class differences accordingly.

The diagonal classifier's FFT and recurrent evaluations differed by at most about 4.8×10⁻⁶ in logits in the author run. That checks two evaluations of the same fitted model. The selective classifier was evaluated using its serial reference; no fused GPU scan was run.

### Investigation: which part of a path matters to this fitted model?

Open validation source row 7, a curved-swing trajectory. Choose either seed-17 model, edit one coordinate or reverse the temporal order, and record whether you expect the class or its probability to change. The display should connect the path edit to a timeline of internal responses and the 15-class readout.

A path can keep its general shape while changing its traversal order. Reversal is therefore a substantive input change, not a harmless plotting transformation. Conversely, resetting an edit must restore the same logits. For the diagonal model, switching between its recurrent and convolutional evaluation should agree within numerical tolerance.

The per-step mixer is causal, but the final classifier averages all 45 representations. Altering the end of a trajectory can change the final class without causing any earlier mixer output to change. Keep those two questions separate when interpreting the display.

## 8. Practical choices, resource counts and failures worth diagnosing

For real work, decide what information is available when the output is required. Complete-record classification can use an entire record; online anomaly detection or next-token prediction cannot use future observations. Bidirectional processing is a task decision, not an automatic property of the word “SSM.”

Several applications become clearer through the mechanism:

* **Continuous signals:** a bank of decaying and oscillating modes can represent temporal patterns in audio or instrument recordings. Sampling interval, resampling and frequency units matter; a learned discrete step is not a substitute for recording the sensor's actual clock.
* **Event streams:** content-dependent writes can react differently to an event and a redundant update. The event representation must make the relevant distinction observable; selectivity cannot infer an unavailable marker by magic.
* **Autoregressive generation:** each layer can carry its own bounded state between new tokens. Prompt processing and one-token decoding use different computational regimes.
* **Irregular observations:** a model with a justified continuous construction can change its transition according to the time gap. Once gaps vary, a single lag-only convolution kernel generally no longer applies.
* **Mixed retrieval and compression:** a hybrid can retain selective recurrent summaries and occasional explicit attention. The later [hybrid SSM–Transformer lesson](/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals) examines that choice.

### Count what is actually retained

For a simple bank of real recurrent states, one sequence needs LDNs bytes: L layers, D channels, N state coordinates and s bytes per coordinate. With L=12,D=64,N=16,float32, that is 49,152 bytes, or 48 KiB.

A conventional full multi-head attention cache with total key width D and total value width D uses 2LTDs bytes. At L=12,T=4096,D=64,float16, that is 12,582,912 bytes, or 12 MiB.

These counts describe specified state arrays. They exclude parameters, batch multiplication, training activations, temporary buffers and a Mamba block's short-convolution cache. Grouped-query attention changes the cache widths; complex states change the bytes per stored coordinate. The equations should be adjusted to the actual architecture.

A smaller retained state is not a measured end-to-end speedup. Benchmark prompt processing and decode separately, report hardware, dtype, batch, dimensions, lengths, warm-up and synchronization, and compare implementations under the same task and quality target. A chart with invented time curves cannot establish a result.

### A compact diagnostic guide

| Symptom | First question or check |
|---|---|
| Convolution and recurrence disagree at the beginning | Same output index, initial state, direct path and kernel taps? |
| Changing the final input changes earlier causal outputs | Circular FFT wraparound, incorrect mask, or whole-record preprocessing? |
| Padding changes a recurrent answer | Did padded steps still decay or write to the state? A zero input is not automatically a no-op. |
| A new record depends on the previous record | Were all layer states and local convolution buffers reset? |
| Forward values become nonfinite | Inspect step parameterization, state magnitudes, dtype and kernel arithmetic before attributing it to “long memory.” |
| Good fitting accuracy, weak held-out results | Recheck role boundaries, duplicates, task size and inductive bias before making the model larger. |
| A model forgets despite the write gate being closed | Is old-state decay still active? |
| Results change with the evaluation algorithm | Compare a high-precision tiny reference, then isolate numerical order or implementation errors. |

Negative continuous decay rates and positive Δ give discrete magnitudes below one for these diagonal modes. Finite precision can still round a decay to 1 or underflow a very small contribution. Input writes, readout weights and nonlinear blocks can also amplify values. There is no universal “safe sequence length” for a dtype.

For products of many decays, dividing two cumulative products can create 0/0 after underflow. Working with log decays helps, but subtracting two large cumulative log sums can lose a small local difference. Stable segment-sum implementations accumulate the relevant local sums directly. The creator's SSD walkthrough explains why the form of an equivalent formula can matter numerically.

A recurrent deployment also needs a decision about gradients across chunk boundaries. Carrying a detached state preserves its forward value but stops gradient flow into earlier chunks. It is truncated training, not full backpropagation through the entire past.

The [official Mamba repository](https://github.com/state-spaces/mamba) contains full blocks and hardware-specific implementations. As inspected on 13 September 2026, its installation options distinguish the core package from optional compiled scan support. Follow the documented environment and selected revision when reproducing a kernel. The small CPU programs here do not claim to validate those kernels or a pretrained language model. A base language-model checkpoint is also a different artifact from an instruction-tuned assistant.

## Connect the recurrence to the maintained scan and complete block

The scratch owners are explicit. [state_space_mechanisms.py](state_space_mechanisms.py) implements held-input/bilinear discretization, zero/nonzero-state recurrence, kernel generation, FFT convolution and SSD state/matrix/chunk calculations. [trajectory_state_models.py](trajectory_state_models.py) implements the trainable diagonal and selective mixers and their full fitting loop. Matrix exponential and linear solves reuse the earlier [ODE](/learn/path/full-curriculum/ordinary-differential-equations-linear-systems) and [Matrix Decompositions](/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu) mechanisms; the new owned operation is how these coefficients become sequence state updates.

The new [state_space_library_bridge.py](state_space_library_bridge.py) supplies the ordinary Mamba package route. First it reuses the exact local `SelectiveMixer` weights and inputs, computes B,C,Δ and A once, and calls `selective_scan_fn`. The local model uses `[batch,time,width]`; the scan API uses `[batch,width,time]`. Variable B/C become `[batch,state,time]`. Both implement `exp(ΔA)` retention and the stated `ΔBu` injection, and share D's direct path. Since Δ has already passed softplus, `delta_softplus=False` avoids applying it twice. There is no output gate in this comparison, so z is omitted. It compares output and input/parameter gradients under one fixed upstream tensor.

That mapping matters: feeding the exact held-input integral from §2 into this scan would define a different operator. Also, the current API's optional last-state output does not propagate its gradient through the fused backward. A loss on the output sequence and a loss on only that returned cache are not interchangeable training contracts. The [maintained scan source](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py) was inspected22September2026 for these conventions.

The second part of the program constructs both ordinary `Mamba` and `Mamba2` blocks, uses a complete loss→backward→clip→AdamW step, then runs evaluation. These include projections and other block operations absent from our isolated recurrence. Accordingly, the example demonstrates normal package use without pretending its random complete-block output equals the small classifier. Read [the official installation and usage contract](https://github.com/state-spaces/mamba) before choosing a build: supported accelerator/compiler/kernel combinations matter. The supplied program deliberately requires a compatible CUDA installation and reports failure when unavailable; it does not silently replace a missing fused kernel with a purported measured GPU result. This optional example is written and source checked, **not executed on GPU in this preparation**.

For daily development, start from the exact CPU mechanisms and use the maintained fused scan after matching values and gradients on small controlled cases. The recurrent form carries O(BDN) state for batch B, width D and state N; the training reference's stored history can be larger. The local SSD matrix visualization is intentionally quadratic for inspection. The chunk algorithm avoids a sequence-wide dense matrix and handles a trailing partial chunk, but the current CPU teaching code is not a hardware-throughput claim. Full original S4 DPLR kernel engineering is a deeper specialized implementation, while this page completely supplies its declared diagonal layer, selective recurrence and SSD mechanisms.

**Changed-code task:** add an initial matrix state to `ssd_chunked` and compare against `ssd_recurrent(..., initial=...)` for length7 and chunk sizes1,3,8.

<details><summary>Hint</summary>The first carry must be the supplied state; every chunk's initial contribution multiplies that incoming carry by its within-chunk cumulative decay.</details>

<details><summary>Solution and success criteria</summary>Add an `initial=None` argument, initialize carry with a copied input matrix when provided and retain zero initialization otherwise. Keep `initial_part = cumprod(a)[:,None] * (C @ carry)` and the boundary update `carry = product * carry + own_final`. The shape must be `[state_size,value_width]`. Every chosen chunking should agree with the sequential recurrence, including the final one-position chunk at size3. Zero write does not imply zero output when the supplied initial state is nonzero. Compare that null separately to avoid incorrectly erasing useful memory.</details>

## 9. Optional extensions: S5 and Mamba-3

### S5: one multi-input, multi-output state

A bank of H independent single-input state systems might store HN state coordinates and then mix their outputs. S5 instead develops a multi-input, multi-output system with one P-dimensional state:

hₜ=Āhₜ₋₁+B̄uₜ,  yₜ=Chₜ+Duₜ,

where B̄ is P×H and C is H×P. Inputs write into a shared state through learned projections. A suitable diagonal parameterization and associative scan provide the computation.

This distinction is about the shape and sharing of memory, not the invention of recurrence or parallel scan. S5 also connects initialization to the normal HiPPO representation. When intervals vary, discretization can account for them step by step; the scan can still compose the resulting affine maps. [S5, §§3.1–3.4](https://arxiv.org/pdf/2208.04933).

### Mamba-3: three changes to inspect separately

Mamba-3, described in a March 2026 paper, extends this family through discretization, rotating state dynamics and richer input/output writes. The following mechanisms explain what changed without treating a new publication date as a performance guarantee. [Mamba-3, §3](https://arxiv.org/pdf/2603.15569).

**Two endpoints in the write.** An exponential-trapezoidal construction can use both the previous and current input contribution:

hₜ=αₜhₜ₋₁+βₜBₜ₋₁xₜ₋₁+γₜBₜxₜ,

with αₜ=exp(ΔₜAₜ), βₜ=(1−λₜ)Δₜαₜ and γₜ=λₜΔₜ. In the scalar demonstration, previous state 1, previous input 2, current input 6, B=1, α=.5, Δ=1 and λ=.5 give .5+.5+3=4. Setting λ=1 gives .5+6=6.5. Both are exact values of the stated discrete rule.

At λ=.5 the construction is an exponential trapezoidal rule under its assumptions; a freely learned λ does not automatically retain a second-order numerical approximation guarantee. The paper specifies regularity and λ=.5+O(Δ) for that claim. At a fresh sequence boundary, the previous-input term also needs an explicit initialization.

**Rotation as state tracking.** A real two-dimensional state can be rotated by

R(θ)=[[cos θ,−sin θ],[sin θ,cos θ]].

Starting at [1,0], apply a π rotation for every input bit 1 and no rotation for bit 0. Bits [1,0,1,1] produce odd/even parity [1,1,0,1]. Purely positive scalar forgetting with zero input injection cannot flip a state's sign this way.

Complex modes are a compact representation of pairs of real coordinates with rotation and decay. This does not say every system with real-valued matrices lacks rotation: the earlier 2×2 oscillator is a real matrix too. The relevant distinction is the permitted transition structure.

Mamba-3 rewrites accumulated rotations as data-dependent rotations of the write/read coordinates, connecting to rotary-position ideas. Unlike ordinary fixed-frequency positional RoPE, these rotations depend on the sequence. Its full recurrence combines the rotated coordinates with the two-endpoint write.

**Higher-rank writes and reads.** In a head with N×P state, an outer-product write b vᵀ has rank at most one. Replace b∈Rᴺ and v∈Rᴾ with B∈R^(N×R) and X∈R^(P×R); the write BXᵀ can have rank up to R. A C∈R^(N×R) read produces CᵀS with shape R×P before subsequent combination.

For small R relative to N and P, more arithmetic can reuse the same retained N×P state. Whether this improves actual latency depends on memory traffic, tensor shapes and kernels. The paper's parameter-sharing scheme controls projection growth; simply multiplying every projection width by R is not its whole design.

**Inline figure: endpoint interpolation, rotating pair, rank-R write.** Use three separate panels with their own entities. Link the first to the discretization comparison, the second to the shrinking spiral, and the third to SSD's outer-product write. A single generic architecture rectangle would conceal what each innovation changes.

The complete architecture also adjusts normalization, learned B/C biases and the local convolution arrangement. Its experiments report particular language-model and state-tracking settings; they do not establish that these mechanisms beat every alternative on all continuous signals, hardware or deployment tasks.

## 10. Practice, explain and transfer

These exercises change the examples. Work out a prediction before opening a hint or solution.

### 1. Recover both output paths

Ā=.4, B̄=2, C=−1, D=.5, initial state h₋₁=3 and inputs [1,−2]. Find both states and outputs. Then predict the outputs if C is set to zero.

<details><summary>Hint</summary>

Update the state first. Compute Ch and Du separately before adding them.

</details>
<details><summary>Solution</summary>

h₀=.4×3+2=3.2 and y₀=−3.2+.5=−2.7. Next h₁=.4×3.2−4=−2.72 and y₁=2.72−1=1.72. With C=0, outputs are simply .5u=[.5,−1], regardless of the evolving state. This is a useful direct-feedthrough check.

</details>

### 2. A missing initial condition

For Ā=.5,B̄=1,C=1,D=0, input [0,0,0] and h₋₁=8, a convolution-only implementation returns three zeros. Give the correct outputs and identify the missing term.

<details><summary>Hint</summary>

Zero input does not imply zero state. Apply Ā once before the first read.

</details>
<details><summary>Solution</summary>

The outputs are [4,2,1]. The missing term is C Ā^(t+1)h₋₁. A zero-input test becomes a zero-output null only when the initial state and any biases/direct effects permit it.

</details>

### 3. Can a linear system remember a fixed delay?

Construct a four-state system that returns the input from three steps earlier. Apply it to [3,−1,4,2,8]. Why does this not solve the general “retain the most recent marked item across arbitrary gaps” task?

<details><summary>Hint</summary>

Write into the first coordinate and shift every coordinate to the next one. Read the last coordinate.

</details>
<details><summary>Solution</summary>

Use Ā with ones on its first subdiagonal and zeros elsewhere, B̄=[1,0,0,0]ᵀ, C=[0,0,0,1], D=0 and zero initial state. The result is [0,0,0,3,−1]. The delay is always three steps. A marker-dependent gap is not a fixed lag; the relevant selection must be supplied by a suitable nonlinear or input-dependent mechanism.

</details>

### 4. An exact gate and a different injection

Let A=−1,B=1, Δ=ln 4, h_previous=2 and u=10. Find the exact held-input update. Then calculate the update using ΔB injection. Are the answers the same?

<details><summary>Hint</summary>

exp(−ln 4)=1/4. Exact ZOH writes (1−1/4)u.

</details>
<details><summary>Solution</summary>

Exact ZOH gives .25×2+.75×10=8. The ΔB rule gives .5+10 ln 4≈14.362944. It is a different discrete operator here. The exact scalar gate is .75; calling the other injection “approximately exact” without considering Δ and A would conceal a substantial difference.

</details>

### 5. Read a two-step SSD state

Start at zero. Let a₀=.3,a₁=.2; b₀=[1,2], b₁=[−1,1]; c₀=[0,1],c₁=[2,1]; and scalar values v₀=3,v₁=4. Compute the two outputs by recurrence and by the influence coefficients.

<details><summary>Hint</summary>

At step 1, the old input's coefficient is .2(c₁ᵀb₀). The current input's coefficient is c₁ᵀb₁.

</details>
<details><summary>Solution</summary>

S₀=[3,6]ᵀ, so y₀=6. S₁=.2[3,6]ᵀ+4[−1,1]ᵀ=[−3.4,5.2]ᵀ and y₁=−1.6. The matrix calculation gives .2×4×3+(−1)×4=2.4−4=−1.6. A negative coefficient is valid; these are not softmax probabilities.

</details>

### 6. Choose a useful experiment before seeing its answer

The diagonal trajectory models reach their lowest validation loss at the final allowed epoch. Propose a follow-up that tests whether training budget is limiting, without repeatedly using the existing assessment set to select settings.

<details><summary>Hint</summary>

Separate model-selection evidence from final assessment. State budgets and seeds before the comparison.

</details>
<details><summary>Solution</summary>

Predeclare several training budgets and seeds, select among them using fitting/validation data, and reserve an untouched assessment source or a properly designed outer evaluation for the final decision. Keep preprocessing and duplicate grouping identical. Reusing the already-inspected assessment results as feedback would make them development evidence; acknowledge that change instead of calling each new result an untouched test. Longer training might help or overfit, so the protocol should allow either outcome.

</details>

### 7. Count a deployment state

A real-state model has 20 layers, width 128, state size 32 and two-byte state coordinates, processing batch 3. Count its recurrent state bytes and MiB. Name two memory costs excluded from this calculation.

<details><summary>Hint</summary>

Multiply batch, layers, width, state size and bytes. One MiB is 1,048,576 bytes.

</details>
<details><summary>Solution</summary>

3×20×128×32×2=491,520 bytes=.46875 MiB. Parameters and temporary activations are excluded; a short-convolution cache or allocator workspace are other possible omissions. A complex64 state would use eight bytes per stored complex coordinate, not two.

</details>

### 8. Trace a changing oscillator

With state [1,0], no injection, no decay and rotation π/2 at each step, give the next four states. Explain why a positive scalar decay times the identity cannot produce this trajectory from the same initial state.

<details><summary>Hint</summary>

A quarter-turn maps [x,y] to [−y,x].

</details>
<details><summary>Solution</summary>

The states are [0,1],[−1,0],[0,−1],[1,0]. Positive scalar multiplication preserves the vector's direction and cannot generate those quarter-turns. A real 2×2 rotation matrix can; complex notation is an equivalent compact representation, not a requirement to abandon real arithmetic.

</details>

### 9. Improve the real-data task for a stronger claim

You want to claim the movement classifier works on previously unseen people. Does the existing random row split answer that question? Specify the metadata and split you would need.

<details><summary>Hint</summary>

The unit of generalization should determine which records stay together.

</details>
<details><summary>Solution</summary>

No. We need performer identifiers and a protocol that keeps all recordings from an assessment performer outside fitting and model selection. Session identity may also matter, depending on the claim. Exact duplicate grouping remains necessary but does not replace person-level grouping. Since the retained dataset lacks those IDs, this stronger claim cannot be recovered merely by changing the random seed.

</details>

You are ready to continue when you can explain the write/retain/read paths, derive a kernel with correct initial conditions, recognize when content dependence removes fixed convolution, and reconcile one SSD chunk boundary. Continue to [RWKV & Linear Attention Models](/learn/path/full-curriculum/rwkv-linear-attention-models?module=deep-learning-fundamentals), which builds a recurrent state through another weighted-memory construction.

## References and other ways to learn

Choose a route based on the part you want to understand more deeply.

* **Continuous systems and the full structured kernel:** [Gu, Goel and Ré, S4](https://arxiv.org/pdf/2111.00396). Read §2 for conventions and discretization, then §3 for the computational reason behind normal-plus-low-rank structure. Follow our two-mode example first; the kernel proof is a deeper branch.
* **What memory coefficients mean:** [Gu and colleagues, HiPPO](https://arxiv.org/pdf/2008.07669). §§2–3 start from approximation under a measure and derive online updates. Read with the polynomial-reconstruction figure beside you; keep the 1/t factors in the scaled Legendre equation.
* **A more accessible diagonal implementation route:** [Gu and colleagues, S4D](https://arxiv.org/pdf/2206.11893). §3 separates discretization, kernel computation and real/complex choices; §4 explains why initialization is more than merely choosing stable eigenvalues.
* **Shared-state MIMO and scans:** [Smith, Warrington and Linderman, S5](https://arxiv.org/pdf/2208.04933), §3. Compare the P-dimensional shared state with a bank of independent channel states.
* **Selection and the actual block:** [Gu and Dao, Mamba](https://arxiv.org/pdf/2312.00752), §§3.1–3.6 and appendix C. The fixed-spacing versus selective-copy distinction is especially useful; compare the mathematical gate derivation with the separately linked reference scan.
* **Duality from two directions:** [Dao and Gu, SSD/Mamba-2](https://arxiv.org/pdf/2405.21060), §§5–7, and the creator's [model article](https://tridao.me/blog/2024/mamba2-part1-model/) and [algorithm article](https://tridao.me/blog/2024/mamba2-part3-algorithm/). The articles explain state shape and the four chunk steps with code. The algorithm article also discusses why its shortest pedagogical interchunk implementation is not the optimized work-efficient scan.
* **A spoken alternative with a transcript:** [Albert Gu's conversation on state-space models](https://www.cognitiverevolution.ai/the-state-space-model-revolution-with-albert-gu/) includes an embedded video, chapter list and transcript. The state discussion around 30:59 and training-versus-inference discussion around 39:05–49:20 complement §§1,3 and 8; the Mamba-2 comparison follows. The lesson author read the relevant transcript and verified the host page, rather than claiming to have watched the recording. Treat its 2024 outlook as historical context.
* **The current family extension:** [Mamba-3](https://arxiv.org/pdf/2603.15569), §§3.1–3.4. Read each new recurrence ingredient separately, then inspect the experimental conditions before interpreting the paper's reported gains.
* **Implementation source:** [state-spaces/mamba](https://github.com/state-spaces/mamba) and its [selective scan reference](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py). Use the current environment instructions for full kernels; the lesson's CPU reference remains a separate, reproducible teaching artifact.
* **Data and reproducible results:** [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), our [data provenance](data-provenance.md), [recorded training results](trajectory-results.json), [mechanism calculations](mechanism-results.json) and [complete training program](trajectory_state_models.py). These let you inspect the actual row roles, errors and calculations behind the local examples.
