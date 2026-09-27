# State Space Models: S4 and the Mamba Family

A temperature sensor reports a deviation of 5, then 0, then 0. At the third reading, the newest value is zero—but a disturbance happened only two readings ago. How could a system retain evidence of that disturbance without saving an ever-growing list?

A **state-space model** keeps a small set of numbers and updates them whenever new data arrives. We will begin with one number you can calculate by hand, then build toward S4's structured memories and Mamba's content-dependent updates. Later, the same ideas become a complete trainable classifier for measured hand movements.

In [Long-Context Sequence Models](/learn/path/full-curriculum/long-context-sequence-models-transformer-xl-griffin-perceiver?module=deep-learning-fundamentals), we compared explicit caches, recurrent memories and latent summaries. This lesson opens the recurrent memory: what do its numbers represent, what changes them, and what gets lost? You need basic matrix products and the idea of a recurrent update. Sampling, complex numbers and the attention connection are introduced locally.

**First pass: build and inspect before specializing.** Read §1 and move the retention control. In §2, follow the held-input example; leave the expandable numerical derivation for later. In §3, follow one input's fading trail and open the system laboratory. Read §4 through the rotating pair; the HiPPO/S4 derivation is a separate return visit. In §5, compare marked events with distractors. In §6, follow the two-by-two state before opening the SSD workshop. Then use §7's real trajectories and training program to connect a useful memory to a learned task.

For an implementation pass, run the mechanism program in §7, work through the native/library bridge after §8, and attempt the changed-code exercise. Return to the deeper §2/§4 branches and §9 when you want to derive S4 or investigate newer family members. The practice section changes the examples so you can check whether you can transfer the ideas. All live controls display their result immediately.

The central question remains the same throughout: **what should survive into the next step, and how can we compute that update well?**

## 1. Give the next reading a useful memory

For this teaching example, readings are deviations from a reference temperature in arbitrary units. A zero is an actual reading at the reference level; it does not mean “no observation.” We start with a memory value of zero.

Several simple approaches are possible. Keeping only the latest reading uses little memory but immediately loses the disturbance. Keeping all readings preserves the history but its storage grows. Keeping a running sum and count is also compact and gives the average of the whole stream; however, a long early history can make that average slow to respond to a recent change. Different tasks can justify each choice.

Here we want a summary that reacts to new readings while allowing older evidence to fade. Try this rule in words: **keep 80% of the previous summary, then add 20% of the new reading**. These percentages are an illustrative design choice, not parameters fitted to a measured sensor.

**Inline figure: retain and write before notation.** Vary retained fraction a from 0 to 1. Show observed [5,0,0], selected update decomposed into retained and incoming contribution, proportional bars and the exact trace. Include latest-only/all-history-average comparisons and the a=1 zero-initial boundary. Every valid edit updates directly.

At the default setting, the first update is 0×.8 + 5×.2 = 1. The next is 1×.8 + 0×.2 = .8; the third is .8×.8 + 0×.2 = .64. Read down the trace: the disturbance's contribution persists even though each later reading is zero. Move retention toward zero to favor the latest measurement, or toward one to retain an existing summary longer. At exactly one, this particular rule stops accepting new information; starting at zero then stays zero.

Now give the quantities names. Let uₜ mean the input at step t, hₜ the summary **after** processing it, and hₜ₋₁ the summary from the preceding step. The short subscript t is just an index in the ordered readings. Our rule becomes

hₜ = 0.8 hₜ₋₁ + 0.2 uₜ.

The summary h is called the **state**. A state is not necessarily a stored measurement: .64 mixes past observations according to the update rule. Reusing the previous state in the next update is a **recurrence**. This is already a one-coordinate linear state-space model.

### From one summary to several useful summaries

One summary might track a fast change; another might retain a slower trend. A **vector** simply collects these summaries. A **matrix** specifies how the old summaries and current inputs contribute to the new ones. We will inspect fast/slow and rotating memories in §4. First, keep four jobs separate: advance the old state, write the input, read the new state, and optionally send some input directly to the output.

For a scalar example, keep half of an existing state 1 and add a new input 3. The new state is .5+3=3.5. Read twice the state to get 7; a separate direct path adds .25×3=.75, producing 7.75. Trace both sums in the picture before reading the matrix notation.

**Inline figure: four paths through one time step.** Use the worked .5,1,2,.25 coefficients and explicit 1→3.5→7.75 states; distinguish retained contribution, write, read and direct path. Keep dimensions as a second annotation.

The same four jobs for vectors are

hₜ = Āhₜ₋₁ + B̄uₜ,  
yₜ = Chₜ + Duₜ.

Here h has N coordinates, u has H input channels and y has J output channels. Ā is N×N, B̄ is N×H, C is J×N and D is J×H. Multiplying by B̄ writes the input into memory; Ā advances existing memory; C reads it; D provides a direct path from the current input.

Throughout this lesson, **the state is updated before it is read**. The initial state is h₋₁. Other references use an output-before-update convention; their first kernel tap can differ by one index without the underlying idea being different.

In the picture, the input contributes both to the new state and through the optional direct path. Those are distinct routes. Changing the readout C changes the reported output without changing the state update itself; changing Ā changes what the memory carries onward.

A useful special case applies the same coefficients at every step. Doubling the input then doubles the zero-initial-state output, and adding two inputs adds their outputs. This is **linearity**. Together with using the same rule at each position, it gives a **linear time-invariant** operator, abbreviated LTI. “Time-invariant” means the same input pattern is treated by the same rule wherever it appears, subject to the sequence boundary. Input-dependent matrices generally break linearity and prevent one fixed convolution kernel from representing the map. A nonlinear operator can still obey time-shift symmetry.

State-space modeling is a broader term than this linear layer: nonlinear physical models and nonlinear state transitions also belong to the family. S4 and Mamba use particular structured linear-in-state updates inside learned nonlinear networks.

### What this buys, and what it does not

A recurrent evaluation retains N state numbers, however long the preceding stream was. That does not mean it remembers every detail exactly. Distinct histories can yield the same finite representation. If a future task asks for an arbitrary old token verbatim, a compressed state can face a different tradeoff from an attention cache that explicitly retains token representations.

The question is therefore: **which information should the state preserve for the task?** S4 supplies useful structured memory modes. Mamba allows the current content to affect what is written, retained and read.

## 2. From continuous change to sampled updates

Suppose our memory moves gradually toward a reading held steady between observations. Waiting longer gives it more time to approach that reading. This raises a practical question: if measurements are a different time apart, should we still keep the same .8 of the previous state? A continuous-time construction gives a consistent way to answer.

Take a system that approaches the held reading at rate 1 per time unit. Starting at zero and holding the reading at 1 for one time unit gives a state of about .6321. Starting at 1 and holding the input at zero gives about .3679. These two contributions add to 1: the same interval determines both how much old memory remains and how much new input is written.

**Inline figure: continuous decay and sampled points.** Put the zero-input decay and held-one approach side by side. Continuous curves and exact sample dots are distinct from the bilinear approximation; retain the singular integrator example.

Look first at the solid curve and its dots. They are the evolving state and the times at which we inspect it. The dashed samples use a different numerical rule, explained in the deeper branch below. “Continuous” means a state defined between readings; “discrete” means the sequence of states at the chosen reading times.

### Name the continuous rule and its sampled version

A continuous linear system writes

dh(t)/dt = Ah(t) + Bu(t),  
y(t) = Ch(t) + Du(t).

A is now a rate-of-change matrix. If time is measured in seconds, its decay rates have inverse-second units. Ā instead describes one discrete step. Confusing these two matrices can silently change an entire model.

For constant input over an interval of length Δ, exact integration gives

Ā = exp(ΔA),  
B̄ = integral from 0 to Δ of exp(sA)B ds.

Thus hₜ=Āhₜ₋₁+B̄uₜ describes the state at the end of that held-input interval. This is **zero-order hold**, or ZOH: the input is held constant while the state evolves. Matrix exp is a matrix function, not elementwise exponentiation except for a diagonal matrix.

For the scalar decay A=−1 and B=1,

Ā=e^(−Δ),  B̄=1−e^(−Δ).

At Δ=ln 2, the old state and held input each receive weight .5. A longer interval permits more decay and more approach toward the held input. Changing Δ changes both terms, not just the forget factor. At this half-life interval, old state 2 and held input 10 give .5×2+.5×10=6: halfway from 2 toward 10. With A=−1, the .8/.2 rule from §1 corresponds to Δ=−ln(.8), about .2231 time units. The sensor example and the differential equation have now met at the same update.

**Keep for the first pass:** A is a rate, Ā is retention over a chosen interval, and B̄ accounts for what arrives during that interval. For sensors, Δ can be elapsed physical time. For token models, it is usually a learned internal parameter; tokens do not come with a physical clock implied by this notation.

<details>
<summary>Deeper implementation: singular systems and S4’s bilinear discretization</summary>

This branch explains how to compute the coefficients robustly and why two legitimate discretizations produce different answers. You can continue to §3 once the rate-versus-step distinction is clear.

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

The distinction matters for sensors with known sampling intervals. For token models, Δ is usually a learned internal quantity; there is no reason to call it elapsed physical time. Some modern sequence layers parameterize the discrete recurrence directly. Continuous-time language is one useful construction, not a requirement that every token model simulate a physical differential equation. [S4, §2.2](https://arxiv.org/pdf/2111.00396).

</details>

## 3. One operator, three ways to compute it

We can watch the state change one step at a time. Can we instead calculate the final answer by tracing what each input contributes? For a fixed linear rule, both views describe the same calculation.

Use an even smaller update: keep half of the old state and add the entire new input; read the state directly. Starting from zero, inputs [2,0,1,0] produce [2,1,1.5,.75]. The first input leaves a trail [2,1,.5,.25]. The input 1, arriving at step 2, starts a new trail [1,.5] there. At step 2, .5 from the old input plus 1 from the new input gives 1.5.

**Inline figure: overlapping impulse trails.** Draw birth-time rows for [2,0,1,0] with bars proportional to contributions under a=.5, B=C=1, D=0. Select an output column and sum it. Separately show the unit pulse’s weights [1,.5,.25,.125]. Future cells are absent, not measured zeros.

Move between columns. Every row carries the same fading shape, shifted to the input's arrival time and scaled by that input. An isolated unit input is called an **impulse**. Its output trail is the **impulse response**; the discrete weights are also called **kernel taps**. Summing the overlapping trails is convolution. Nothing has been learned or approximated in changing this view.

### Derive the trail weights from the state rule

Expand a fixed-matrix recurrence from zero initial state:

h₀=B̄u₀,  
h₁=ĀB̄u₀+B̄u₁,  
h₂=Ā²B̄u₀+ĀB̄u₁+B̄u₂.

After applying C, the contribution of an input depends only on how many steps ago it arrived. Define the kernel taps

Kₗ = C Āˡ B̄,  for l=0,1,2,...

Then

yₜ = sum over j=0…t of Kₜ₋ⱼ uⱼ + Duₜ.

This is a **causal convolution**: “causal” means that output t uses only input t and earlier inputs, and a **lag** is how many steps have elapsed since an input arrived. In the little picture Ā=.5 and B̄=C=1, so the taps are exactly 1,.5,.25,.125. In a larger state, the product C Āˡ B̄ performs the same write → l advances → read journey.

With a nonzero initial state, add C Ā^(t+1) h₋₁. That is the trace already in memory before the new sequence begins. For initial state 8, retention .5 and no new inputs, outputs are 4,2,1—not zero. Omitting that term changes the problem.

<details>
<summary>Deeper connection: the continuous impulse response</summary>

The discrete picture sums one contribution per input step. Its continuous counterpart integrates contributions over the times when they arrive. With initial state h(0), the output is

y(t)=C exp(At)h(0) + integral from 0 to t of C exp(A(t−s))B u(s) ds + Du(t).

The strictly proper impulse response is g(t)=C exp(At)B. If you choose instead to include Dδ(t) in the impulse-response distribution, its convolution already supplies the direct path; do not add Du a second time.

</details>

### A calculation you can reconcile by hand

The next example keeps the same input [2,0,1,0] and the same contribution accounting, but adds a second memory coordinate and a direct input path. The extra coordinate changes the kernel; it does not change what convolution means. Use two independent continuous modes with A=diag(−1,−2), B=[1,1]ᵀ, C=[1,−.5], D=.25 and Δ=ln 2. ZOH yields

Ā=diag(.5,.25), B̄=[.5,.375]ᵀ.

The first four kernel taps are

[.3125, .203125, .11328125, .0595703125].

For inputs [2,0,1,0] and zero initial state, the output is

[1.125, .40625, .7890625, .322265625].

Check the third output: 2×.11328125 + 1×.3125 + .25×1 = .7890625. Its three terms are the old input's remaining contribution, the current input's memory contribution and direct feedthrough.

**Inline figure: an impulse ledger.** Show each input as a row whose contribution spreads rightward according to the kernel. A column sum produces one output. Align it with the recurrent state trace so the learner can match “one compact state” to “many historical contributions.”

In the ledger, read a row to follow one input through time; read a column to reconstruct one output. The separate state trace is the compact computation of those same column sums. This connection is why we can choose a convenient evaluation method without choosing a different learned model.

The same result can be calculated in three ways:

| Evaluation | Main operation | Useful setting |
|---|---|---|
| Recurrence | Update and retain state one step at a time | Streaming or autoregressive decoding |
| Direct convolution | Sum lagged input contributions | Small transparent calculations |
| FFT convolution | Transform input and kernel, multiply, inverse-transform | Full fixed-kernel sequences |

The **fast Fourier transform (FFT)** changes from a sequence of values to frequency coordinates, where convolution becomes multiplication, then changes back. You can use the independently checked program without deriving the transform here; [Fourier methods](/learn/path/full-curriculum/complex-numbers-fourier-laplace-transforms?module=math-foundations) develop that machinery. The FFT computes circular convolution unless padding is correct. With T input samples and K retained taps, use a transform length at least T+K−1, then retain the causal outputs needed. Padding to T and hoping for the best can wrap a future tail into the beginning.

A dense Ā costs O(N²) per recurrent step. A diagonal Ā costs O(N). Computing every diagonal kernel tap naively costs O(NT), followed by roughly O(T log T) FFT work. S4's special kernel-generation algorithm addresses the kernel construction too. Saying “the model uses an FFT” does not account for every operation.

### Investigation: change a system, then reconcile its computations

Start by changing only the last input sample and watch which output rows change. Earlier outputs should remain fixed. Then change the initial state: now its contribution can reach every output. Finally try C=0,D=1, which removes the memory readout and returns the inputs directly. Each edit updates the state, kernel, contribution ledger and all three evaluations immediately.

Inspect the state plane, kernel and contribution ledger. Try the singular integrator and the feedthrough-only setting C=0,D=1. Then change only the final input and inspect earlier outputs. Explain every difference by a legal information path.

The comparison must include the initial-state response in the convolutional view. A disagreement caused by leaving it out is a useful diagnosis, not evidence that recurrence and convolution are different models.

## 4. Designing a memory with more than one timescale

Our first memory mixed everything into one number. Suppose a signal contains a sudden disturbance on top of a slowly drifting background. A fast-changing summary can follow the disturbance; a slow-changing summary can keep the background. Keeping both gives a later readout more useful information than forcing one smoothing rate to do both jobs.

Call one independently evolving pattern a **mode**. For now, a mode is simply one state coordinate that repeatedly multiplies its retained contribution by the same number. Feed three such coordinates the same unit impulse and look at their trails before comparing formulas.

**Inline figure: several memory clocks.** Use the existing computed .2,.8,.99 impulse traces, close-up and half-life table. Teach one mode’s attenuation, not a neural recall guarantee.

Follow the .2 curve: almost nothing remains after a few updates. The .99 curve changes little over that same interval. These are different views of the same past event, available simultaneously to the readout. A learned readout can add or subtract them to respond to changes rather than only to a level.

A mode with discrete decay .99 retains information much longer than one with decay .2. After k empty updates, a contribution is multiplied by aᵏ. For 0<a<1, its half-life is ln(.5)/ln(a) steps. Half-life describes one mode's attenuation, not the guaranteed recall span of an entire trained network.

Several real decays provide several smoothing rates. But decay alone moves a positive contribution toward zero without changing its sign. Repeated back-and-forth motion calls for a different pattern: keep two coordinates that rotate together. As the direction changes, either coordinate can switch sign; shrinking the pair gradually forgets an old oscillation.

**Inline figure: a shrinking spiral synchronized with two coordinate traces.** Show the equal-scale plane and both coordinate traces before the matrix. Mark the start and direction; explain that both plots represent the same pair.

Read the spiral from its initial point (1,0), moving counterclockwise inward. The horizontal and vertical traces are the two coordinates of that moving point. A **phase** is its position around the rotation; a **frequency** specifies how quickly the angle advances. Oscillatory modes can respond differently to patterns with different repetition rates. One continuous rule producing the pictured pair is

A = [[−.2,−2],[2,−.2]].

With no input, its state rotates at 2 radians per time unit while its radius decays as e^(−.2t). At Δ=.25, each step rotates by .5 radians and shrinks the radius by e^(−.05). The state does not merely get “older”; its direction changes.

### A small diagonal state-space layer

The rotating pair can be stored either as two real numbers or as one complex number x+iy. Multiplication by a complex coefficient rotates and scales that pair; it is compact notation for an operation we have already drawn. A **diagonal** collection evolves each mode separately, then combines their readouts. A diagonal model can use complex-valued modes aₙ=−exp(αₙ)+iωₙ. The negative real part provides decaying continuous modes; ω controls rotation. For a real input and a real output, include conjugate pairs. One stored half of each pair contributes

2 Re(cₙ hₙ)

to the readout. The factor 2 and real part are part of the model, not cosmetic plotting choices.

Our small training program stores four complex modes per channel and their implicit conjugates. It learns decay, frequency, step size, complex readout and a direct skip; it fixes the continuous B to 1 and calculates its discrete B̄ exactly. It evaluates the temporal operation both recurrently and through a generated convolution kernel.

This is an **S4D-style teaching layer** with an explicitly chosen linear-frequency initialization. It is not a full implementation of S4's diagonal-plus-low-rank kernel algorithm. The distinction lets us teach and test a complete useful model without silently attaching the wrong name to it. [S4D, §§3–4](https://arxiv.org/pdf/2206.11893).

### Return visit: why S4 chooses structured memories

The core takeaway is that a trainable bank of fading and rotating patterns can summarize different parts of a signal's history. You can now continue to §5, where the next input changes the memory rule itself. The branch below answers a harder design question: can the state coordinates be chosen to reconstruct the shape of the past, and can that structured system still be computed efficiently?

<details>
<summary>Deeper derivation: polynomial memory, HiPPO and S4’s structured kernel</summary>

### Give a memory coordinate a precise meaning

Imagine retaining a level and a trend instead of retaining raw samples. Those two numbers reconstruct any straight-line history exactly, while more complicated histories need additional shapes. HiPPO turns this reconstruction question into an online update. The remaining derivation makes the approximation criterion explicit; it is not needed to operate the core labs.

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

### Preserve that memory structure without expensive dense work

The projection tells us what to remember. S4 also has to make its long kernel practical to compute. A completely dense state update would mix every coordinate with every other coordinate. The key is to represent the desired dynamics as an easy part plus a small correction, retaining both rather than discarding the correction.

S4 uses a structured starting point and a numerically useful representation. Let A=−A₊ and pₙ=√(n+.5). Then A+ppᵀ is a normal matrix; in this real case its symmetric part is −I/2. A normal matrix has an orthonormal eigenbasis. After a unitary change of basis, the original matrix can therefore be written as a diagonal matrix minus a low-rank correction.

This is **diagonal plus low rank**, or DPLR. It avoids treating a poorly conditioned eigenvector decomposition of the original nonnormal triangular matrix as numerically harmless.

The remaining computational idea is to evaluate the kernel's generating function at suitable frequency points. For T taps, define K_T(z)=sum from l=0 to T−1 of K_l z^l. The finite geometric-series identity gives

K_T(z)=C {I−(zĀ)^T} (I−zĀ)⁻¹ B̄

where the inverse exists. This is a scalar-valued function for one input/output channel; evaluating it at Fourier points gives the transform of the finite tap sequence. The factor I−(zĀ)^T is the finite-length correction.

For a continuous DPLR matrix A=Λ−pq*, let R₀(s)=diag(1/(s−λₙ)). The star denotes conjugate transpose. Woodbury gives

(sI−A)⁻¹ = R₀−R₀p(1+q*R₀p)⁻¹q*R₀.

The apparently large inverse is reduced to diagonal operations and a scalar correction in this rank-one case. Terms such as q*R₀p are sums of weighted 1/(s−λₙ) factors, explaining the Cauchy structure. S4 combines this idea with its discretization to evaluate the finite kernel efficiently. Resolvents of a diagonal matrix are easy; the Woodbury identity handles the low-rank correction. The resulting sums have a Cauchy-like structure, which specialized algorithms exploit, and a transform recovers the time-domain taps. That is the reason for S4's mathematical machinery: retain useful structured dynamics while making a long kernel practical to calculate.

An advanced implementation should follow the paper's finite-length correction, discretization and stability conventions together. Copying only its A matrix into a naïve dense recurrence does not reproduce the kernel algorithm. S4D investigates which benefits survive a diagonal simplification and carefully chosen initialization; its approximation results do not say that a small finite diagonal model equals the full HiPPO system exactly. [S4, §§3.1–3.4](https://arxiv.org/pdf/2111.00396).

</details>

## 5. Keep the event, ignore the distraction: Mamba’s selection

A stream of device messages contains occasional calibration values mixed with routine status updates. Our teaching task is to retain the most recent marked calibration value. The marker is part of the available input. Averaging every message would let routine updates contaminate the retained value; retrieving whatever arrived two steps ago would work only when the gaps happened to match.

**Inline figure: marked event versus fixed age.** On [4,9,−7,6], mark the first and last positions. Compare exact supplied marker-gated memory [4,4,4,6], two-step delay [0,0,4,9], and half-retention [2,5.5,−.75,2.625]. Each rule answers a different question. Marker gates are programmed here, not learned.

Read across the middle two events. Their ages increase, but the identity of the last marked value stays 4. The requirement is about **which content matters**, not just how old it is. This is an original toy task illustrating selection, not a claim about a deployed calibration system.

### Why a delay and a selective memory solve different problems

An LTI operator can copy a fixed delay perfectly. A three-state shift register can store the current input, the previous input and the input before that. Reading the third coordinate sends [2,5,−1,7,0] to [0,0,2,5,−1]. Its convolution kernel is a single pulse at lag 2.

The harder task is to retain a marked item while an unpredictable number of irrelevant items arrive. A fixed temporal kernel applies the same lag weights regardless of which item was marked. A nonlinear deep S4 network is more than one LTI operator, but input-dependent selection gives the temporal update itself a direct way to respond.

We can solve the marked-value toy task with a switch: on a marked event, replace memory; on a distraction, keep memory. A **gate** generalizes that switch to a value between zero and one. A gate near one emphasizes the new write; a gate near zero preserves the old state in this coupled rule.

A simple selective update is

hₜ=(1−gₜ)hₜ₋₁+gₜuₜ,  0≤gₜ≤1.

Use inputs [4,9,−7,6] with gates [.99,.01,.01,.99]. The states are

[3.96,4.0104,3.900296,5.97900296].

The first and last items substantially replace the state; the middle items have little effect. With a constant gate .5, the states are [2,5.5,−.75,2.625]. These gates are supplied teaching controls. A trained network must learn how to compute useful gates from available inputs.

Before continuing, open the selective-memory lab below and change a middle distractor while its gate is almost closed. Then increase that gate and watch the new-write contribution enter the state. This is what “selection” changes. The lab also lets you separate writing from retention: closing a write does not preserve old memory if decay remains active.

The scalar gate has an exact state-space connection. For A=−1, B=1 and Δₜ=softplus(zₜ), exact ZOH gives

exp(−Δₜ)=1−sigmoid(zₜ),  
B̄ₜ=1−exp(−Δₜ)=sigmoid(zₜ).

Thus gₜ=sigmoid(zₜ). Softplus is log(1+exp(z)); sigmoid is 1/(1+exp(−z)). Substitute these definitions to verify the identity. A large Δ both erases more old state and increases the new write in this scalar construction.

**Inline figure: annotated write and forget bands.** Above an editable signed input sequence, show g and 1−g as complementary bands. Below, decompose the next state into retained old contribution and incoming contribution. An “important” badge alone cannot explain which quantity a gate changes.

### Turn the hand-chosen gate into learned coefficients

The supplied gates make the mechanism visible. A neural model must produce useful coefficients from the features it actually sees. In Mamba-1, learned projections of the current features control the write vector B, the read vector C and a positive step parameter Δ. The base decay rates A are learned parameters shared across positions. Thus the input can change both which information enters memory and which part is exposed as output.

Here **channel** means one feature coordinate in a representation, and **batch** means several independent sequences processed together. Each sequence has its own memory. For a batch of B sequences of length T with D internal channels and N state coordinates per channel, a common Mamba-1 organization uses:

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

We have described the temporal memory operation. A usable neural block also prepares features, mixes nearby positions and routes the result back into the network. Keep that distinction in mind when comparing a short recurrence with a library class.

A Mamba-1 block projects its input into an expanded feature branch and a gate branch. The feature branch passes through a short causal depthwise convolution and an activation, then supplies the selective operator. Its result is multiplied by an activated gate and projected back to the model width; residual connections and normalization organize the stack. The gate commonly uses SiLU(x)=x·sigmoid(x), so it is an activated multiplicative branch rather than a probability distribution.

The short convolution gives nearby positions a local interaction before parameter selection. The outside gate is different from Δ inside the recurrence. A diagram should draw both and label their equations, rather than call every multiplication “the forget gate.”

**Inline figure: a shape-aware Mamba block.** Show the branch split, local causal convolution, B/C/Δ projections, selective state update, separate SiLU gate, output projection and residual. A tooltip or side note should distinguish the full block from the simplified mixer used in our small experiment.

The local convolution mixes a short neighborhood; the selective recurrence carries information beyond it; the outside gate scales what the block returns. Follow the two branches to their joining point in the diagram. The gate on the right is not another name for Δ on the left.

### Compute a dependent chain without one long serial loop

Because the coefficients vary with content, one fixed global convolution kernel no longer describes all inputs. Recurrence is still available. Moreover, the affine maps h↦a⊙h+b compose associatively:

(a₂,b₂) after (a₁,b₁) = (a₂⊙a₁, a₂⊙b₁+b₂).

For example, first apply h→.5h+1, then h→.2h+3. Substituting the first into the second gives h→.1h+3.2. The pair can be summarized by two coefficients before knowing the incoming h. Combining such summaries in a tree is a **parallel prefix scan**: each prefix still includes exactly the earlier updates it needs.

Their coefficients can be computed from the input before scanning. A parallel prefix scan can therefore evaluate the sequence with logarithmic dependency depth, while a work-efficient implementation keeps total arithmetic proportional to sequence length for fixed state dimensions. Parallel does not mean every state can ignore earlier inputs; it means the same dependencies can be grouped.

The practical Mamba algorithm also fuses operations and recomputes selected intermediates in the backward pass to reduce memory traffic. It need not materialize the entire B×T×D×N state trajectory in device memory. Kernel details, precision and shapes determine the actual speed; a Python loop will not inherit fused-kernel throughput.

### Investigation: build a selective memory challenge

Create a signed sequence, mark the items that should replace memory, and edit gaps and distractors. Compare the constant gate and your input-dependent schedule live. The separate old-state and new-write contributions show which change improved retention and which suppressed a distraction.

Now make every input zero and start at zero. Can changing the gates alone create a nonzero state in this update? Then restore the signal and close the write gate while leaving decay active in a more general two-coefficient recurrence. Explain why “stop writing” and “stop forgetting” are distinct interventions.

## 6. Mamba-2 and state-space duality

Mamba-1 lets many state coordinates decay at different rates. Mamba-2's core operator shares one decay across a group of coordinates. Why accept that restriction? It exposes another way to calculate the same outputs, using matrix operations that modern hardware handles well. We first need to see exactly what is shared.

Picture a small memory with two rows and two columns. A new two-number value is [3,−1]. A write vector [0,1] says to add none of that value to the first row and one copy to the second. Multiplying every write-vector entry by every value entry produces [[0,0],[3,−1]]. This is an **outer product**: a column of write weights times a row of values.

Suppose the old memory is [[2,1],[0,0]]. Keep half of **every entry**, add that new write, and get [[1,.5],[3,−1]]. To read it, use weights [1,1]: add the two rows and obtain [4,−.5]. We have just computed an SSD step without needing the attention analogy.

**Inline figure: one matrix memory update.** Show old matrix, uniform half-retention, outer-product write with its row multipliers, summed state and weighted-row read. Use the exact second step of the four-step example; label axes.

Follow the second row: it was zero, so it comes entirely from the new write. The first row comes entirely from carried memory. Changing the read weights would change the output while leaving this stored matrix unchanged.

### Follow four concrete updates

Call the memory matrix S and its output vector y. Now give that same two-by-two memory four updates. The table supplies each decay a, write weights b, read weights c and two-number value v:

| Step | a | b | c | v |
|---|---:|---|---|---|
| 0 | .5 | [1,0] | [1,0] | [2,1] |
| 1 | .5 | [0,1] | [1,1] | [3,−1] |
| 2 | .25 | [1,1] | [0,1] | [1,2] |
| 3 | .8 | [1,−1] | [1,2] | [−2,1] |

At step 0, S₀=[[2,1],[0,0]] and y₀=[2,1]. At step 1,

S₁=.5S₀ + [[0,0],[3,−1]] = [[1,.5],[3,−1]],

so y₁=[4,−.5]. Continuing gives y₂=[1.75,1.75] and y₃=[5.8,3.5]. The program independently calculates the recurrent, full matrix and chunked forms and checks that all agree.

### Name the operator we just calculated

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

**Inline figure: signed influence matrix beside the state.** Each row i shows which earlier values affect output i. Mark future entries as structurally absent, use a zero-centered scale for signed coefficients, and keep the decay matrix L visually distinct from the content factor CBᵀ. An attention-style “probability heatmap” would mislabel these numbers.

### Why grouping into chunks helps

A full influence matrix makes every input-to-output contribution visible, but it becomes large for a long sequence. A one-step recurrence avoids that matrix but has a sequential dependency. Chunking combines the two: do the detailed work inside short stretches and pass only the summary needed between them. This changes how we compute the answer, not which past inputs may affect it.

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

First change only v₃ and observe that outputs 0–2 remain fixed. Then change the chunk size while holding the input and coefficients fixed: the outputs should remain the same, although the local-versus-incoming decomposition changes. Finally edit b or c to see how writing differs from reading. Compare the matrix view and the state bridges.

Setting a₂=0 erases the incoming state just before step 2; it does not erase the new step-2 write. Changing only v₃ must leave outputs 0–2 unchanged. Setting every b to zero with zero initial state produces zero output even when c and v vary. Explain these observations from the recurrence using both the numerical equality and its mathematical explanation.

## 7. Make the state useful by learning from real movement

So far, the numbers in the examples were chosen to expose a mechanism. We now ask the system to learn a useful summary from data. This closes the loop with the opening question: can a fixed collection of changing numbers retain enough of a movement's history to recognize its category?

A clever recurrence is not yet a classifier. We need an input representation, a prediction target, a loss, trainable parameters and an evaluation procedure.

We will classify **Libras movement trajectories**. The UCI dataset contains 360 recordings in 15 movement categories. Each record has 45 two-dimensional hand coordinates and a category. Examples include curved swing, circle, horizontal straight line and vertical zigzag. These are normalized trajectory coordinates derived from videos, not calibrated physical positions or complete sign-language conversations. [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement).

**Inline figure: three recorded paths, with time visible.** Draw actual source trajectories on equal-scale x/y axes. Number a few points and attach a start marker and directed segments. A curve without temporal order hides a quantity that sequence models can use.

Look at the start and end markers before judging the shape. Two recordings can visit similar positions in different orders. Our input therefore keeps all 45 positions in their recorded sequence; it is not just an unordered cloud of dots.

The downloadable source has 30 repeated feature rows with consistent labels. We retain the first occurrence of each exact trajectory before partitioning, leaving 330 unique rows. Otherwise an identical input could land on both sides of the evaluation boundary.

A fixed classwise shuffle produces 220 fitting, 50 validation and 60 assessment trajectories. Every assessment class has four rows. The exact one-based source IDs are saved in the results. Performer and recording-session identifiers are absent, so this row-level protocol cannot establish performance on new performers or sessions.

### The complete prediction pipeline

The classifier must turn an ordered path into one category. It first represents each position with more features, updates those features using temporal memory, summarizes the resulting sequence, and reads a category score. Keep the data shape beside each operation so that “the model learns a memory” becomes an inspectable computation.

Each trajectory is a 45×2 array. Convert a coordinate x in [0,1] to 2x−1. This fixed transformation uses no estimated corpus statistics. A linear projection turns each coordinate pair into 16 features.

Two residual blocks each perform

z ← z + W_out GELU(TemporalMixer(LayerNorm(z))).

Here W_out denotes the learned affine projection, including its bias. GELU is the smooth activation xΦ(x), where Φ is the standard-normal cumulative distribution function. Layer normalization and the output projection operate within each time step. The temporal mixer is what carries information across time. Average the resulting 45 feature vectors, then apply a 16-to-15 linear classifier.

The final 15 numbers are **logits**. Softmax converts them into model probabilities. For true class k, the loss is −log pₖ. A high probability for the wrong class incurs a large loss; a correct top-ranked class can still have a mediocre probability and a nonzero loss.

The recurrence parameters learn through the same computational graph as the projections. For an elementary readout example, hold state h=[1,2], target r=3 and C=[.5,.5]. The output is 1.5. With loss .5(Ch−r)², the gradient with respect to C is (Ch−r)h=[−1.5,−3]. A gradient step of .1 changes C to [.65,.8], output to 2.25 and loss from 1.125 to .28125. Backpropagation through an entire sequence extends this chain to writes, decays and earlier inputs.

**Inline figure: from one measured path to one loss.** Show 45×2 coordinates → 45×16 projected features → two temporal blocks → 16 pooled features → 15 logits → the probability assigned to the labeled category. Put the update arrows back to the actual trainable quantities, not to a mythical “memory quality” knob.

To connect the final stage of the diagram to gradient descent, trace the loss back through the operations. A poor category probability changes the final classifier weights; the gradient also reaches the temporal mixer and tells its write, read and decay parameters which changes would reduce that loss. The little readout calculation above isolates one part of this chain. The complete program lets automatic differentiation carry the chain through all 45 updates.

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

Read the program in three passes. First follow data roles and the model's forward path to locate the state update. Next find the loss, backward call, optimizer step and validation-based checkpoint selection. Finally inspect the recurrent/FFT check and saved row IDs. This makes the code an experiment you can modify, not a long block to copy without knowing its boundaries.

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

Read the loop from top to bottom: `decay` computes what fraction of each old coordinate survives; `write` computes what the current input adds; `state` combines those two terms; the last multiplication by C reads the new state and the skip term adds the direct path. It is the write/retain/read diagram from §1 with learned, input-dependent coefficients.

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

Open validation source row 7, a curved-swing trajectory. First move one selected point with its visible drag handle or numeric fields and watch the internal-response trace and category probabilities update. Then reset and reverse the temporal order. Use one intervention at a time so you can connect an output change to an actual input change. The browser evaluates the fitted model; the probabilities are computed, not an animation chosen to suggest success.

A path can keep its general shape while changing its traversal order. Reversal is therefore a substantive input change, not a harmless plotting transformation. Conversely, resetting an edit must restore the same logits. The native program independently checks the diagonal model’s recurrent and FFT evaluations. The browser uses the recurrent evaluation and displays its current output.

The per-step mixer is causal, but the final classifier averages all 45 representations. Altering the end of a trajectory can change the final class without causing any earlier mixer output to change. Keep those two questions separate when interpreting the display.

## 8. Practical choices, resource counts and failures worth diagnosing

We have now seen three different questions: what memory rule represents the task, how to compute that rule, and whether a trained system works on held-out data. Keep them separate when choosing an architecture. A compact state alone does not answer all three.

For real work, decide what information is available when the output is required. Complete-record classification can use an entire record; online anomaly detection or next-token prediction cannot use future observations. Bidirectional processing is a task decision, not an automatic property of the word “SSM.”

Several applications become clearer through the mechanism:

- **Continuous signals:** a bank of decaying and oscillating modes can represent temporal patterns in audio or instrument recordings. Sampling interval, resampling and frequency units matter; a learned discrete step is not a substitute for recording the sensor's actual clock.
- **Event streams:** content-dependent writes can react differently to an event and a redundant update. The event representation must make the relevant distinction observable; selectivity cannot infer an unavailable marker by magic.
- **Autoregressive generation:** each layer can carry its own bounded state between new tokens. Prompt processing and one-token decoding use different computational regimes.
- **Irregular observations:** a model with a justified continuous construction can change its transition according to the time gap. Once gaps vary, a single lag-only convolution kernel generally no longer applies.
- **Mixed retrieval and compression:** a hybrid can retain selective recurrent summaries and occasional explicit attention. The later [hybrid SSM–Transformer lesson](/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals) examines that choice.

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

## Implementation pass: from scratch to the maintained scan and block

There are three useful levels of control. The tiny NumPy mechanisms expose every state update and serve as numerical references. The trainable PyTorch classifiers add differentiation and an actual fitting/evaluation protocol. The maintained package supplies optimized scans and full architecture blocks. Choose the lowest level needed to inspect or change the mechanism, then verify equivalence before replacing that piece with a faster one.

The scratch owners are explicit. [state_space_mechanisms.py](state_space_mechanisms.py) implements held-input/bilinear discretization, zero/nonzero-state recurrence, kernel generation, FFT convolution and SSD state/matrix/chunk calculations. [trajectory_state_models.py](trajectory_state_models.py) implements the trainable diagonal and selective mixers and their full fitting loop. Matrix exponential and linear solves reuse the earlier [ODE](/learn/path/full-curriculum/ordinary-differential-equations-linear-systems) and [Matrix Decompositions](/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu) mechanisms; the new owned operation is how these coefficients become sequence state updates.

The [state_space_library_bridge.py](state_space_library_bridge.py) supplies the ordinary Mamba package route. First it reuses the exact local `SelectiveMixer` weights and inputs, computes B,C,Δ and A once, and calls `selective_scan_fn`. The local model uses `[batch,time,width]`; the scan API uses `[batch,width,time]`. Variable B/C become `[batch,state,time]`. Both implement `exp(ΔA)` retention and the stated `ΔBu` injection, and share D's direct path. Since Δ has already passed softplus, `delta_softplus=False` avoids applying it twice. There is no output gate in this comparison, so z is omitted. It compares output and input/parameter gradients under one fixed upstream tensor.

That mapping matters: feeding the exact held-input integral from §2 into this scan would define a different operator. Also, the current API's optional last-state output does not propagate its gradient through the fused backward. A loss on the output sequence and a loss on only that returned cache are not interchangeable training contracts. The [maintained scan source](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py) was inspected on 22 September 2026 for these conventions.

The second part of the program constructs both ordinary `Mamba` and `Mamba2` blocks, uses a complete loss→backward→clip→AdamW step, then runs evaluation. These include projections and other block operations absent from our isolated recurrence. Accordingly, the example demonstrates normal package use without pretending its random complete-block output equals the small classifier. Read [the official installation and usage contract](https://github.com/state-spaces/mamba) before choosing a build: supported accelerator/compiler/kernel combinations matter. The supplied program deliberately requires a compatible CUDA installation and reports failure when unavailable; it does not silently replace a missing fused kernel with a purported measured GPU result. This optional example is written and source checked, **not executed on GPU in this lesson’s verification**.

For daily development, start from the exact CPU mechanisms and use the maintained fused scan after matching values and gradients on small controlled cases. The recurrent form carries O(BDN) state for batch B, width D and state N; the training reference's stored history can be larger. The local SSD matrix visualization is intentionally quadratic for inspection. The chunk algorithm avoids a sequence-wide dense matrix and handles a trailing partial chunk, but the current CPU teaching code is not a hardware-throughput claim. Full original S4 DPLR kernel engineering is a deeper specialized implementation, while this page completely supplies its declared diagonal layer, selective recurrence and SSD mechanisms.

**Changed-code task:** add an initial matrix state to `ssd_chunked` and compare against `ssd_recurrent(..., initial=...)` for length 7 and chunk sizes 1, 3 and 8.

<details>
<summary>Hint</summary>

The first carry must be the supplied state; every chunk's initial contribution multiplies that incoming carry by its within-chunk cumulative decay.

</details>

<details>
<summary>Solution and success criteria</summary>

Add an `initial=None` argument, initialize carry with a copied input matrix when provided and retain zero initialization otherwise. Keep `initial_part = cumprod(a)[:,None] * (C @ carry)` and the boundary update `carry = product * carry + own_final`. The shape must be `[state_size,value_width]`. Every chosen chunking should agree with the sequential recurrence, including the final one-position chunk at size3. Zero write does not imply zero output when the supplied initial state is nonzero. Compare that null separately to avoid incorrectly erasing useful memory.

</details>

## 9. Optional extensions: S5 and Mamba-3

This return visit changes one design choice at a time. S5 changes which input channels share a state. Mamba-3 changes the write approximation, the allowed rotation and the number of independent writes. Relate each change to the earlier picture before comparing model names.

### S5: one multi-input, multi-output state

A bank of H independent single-input state systems might store HN state coordinates and then mix their outputs. S5 instead develops a multi-input, multi-output system with one P-dimensional state:

hₜ=Āhₜ₋₁+B̄uₜ,  yₜ=Chₜ+Duₜ,

where B̄ is P×H and C is H×P. Inputs write into a shared state through learned projections. A suitable diagonal parameterization and associative scan provide the computation.

This distinction is about the shape and sharing of memory, not the invention of recurrence or parallel scan. S5 also connects initialization to the normal HiPPO representation. When intervals vary, discretization can account for them step by step; the scan can still compose the resulting affine maps. [S5, §§3.1–3.4](https://arxiv.org/pdf/2208.04933).

### Mamba-3: three changes to inspect separately

Mamba-3, described in a March 2026 paper, extends this family through discretization, rotating state dynamics and richer input/output writes. The following mechanisms explain what changed without treating a new publication date as a performance guarantee. [Mamba-3, §3](https://arxiv.org/pdf/2603.15569).

**Two endpoints in the write.** In §2 we held one input over an interval. If the input changes across the interval, both its previous and current contributions can inform a different integration rule. This gives a reason for adding a second write term before adding its symbols. An exponential-trapezoidal construction can use both the previous and current input contribution:

hₜ=αₜhₜ₋₁+βₜBₜ₋₁xₜ₋₁+γₜBₜxₜ,

with αₜ=exp(ΔₜAₜ), βₜ=(1−λₜ)Δₜαₜ and γₜ=λₜΔₜ. In the scalar demonstration, previous state 1, previous input 2, current input 6, B=1, α=.5, Δ=1 and λ=.5 give .5+.5+3=4. Setting λ=1 gives .5+6=6.5. Both are exact values of the stated discrete rule.

At λ=.5 the construction is an exponential trapezoidal rule under its assumptions; a freely learned λ does not automatically retain a second-order numerical approximation guarantee. The paper specifies regularity and λ=.5+O(Δ) for that claim. At a fresh sequence boundary, the previous-input term also needs an explicit initialization.

**Rotation as state tracking.** A real two-dimensional state can be rotated by

R(θ)=[[cos θ,−sin θ],[sin θ,cos θ]].

Starting at [1,0], apply a π rotation for every input bit 1 and no rotation for bit 0. Bits [1,0,1,1] produce odd/even parity [1,1,0,1]. Purely positive scalar forgetting with zero input injection cannot flip a state's sign this way.

Complex modes are a compact representation of pairs of real coordinates with rotation and decay. This does not say every system with real-valued matrices lacks rotation: the earlier 2×2 oscillator is a real matrix too. The relevant distinction is the permitted transition structure.

Mamba-3 rewrites accumulated rotations as data-dependent rotations of the write/read coordinates, connecting to rotary-position ideas. Unlike ordinary fixed-frequency positional RoPE, these rotations depend on the sequence. Its full recurrence combines the rotated coordinates with the two-endpoint write.

**Higher-rank writes and reads.** In §6, one outer product wrote copies of one value vector into the state rows. Several such writes can add independent directions during the same step. In a head with N×P state, an outer-product write b vᵀ has rank at most one. Replace b∈Rᴺ and v∈Rᴾ with B∈R^(N×R) and X∈R^(P×R); the write BXᵀ can have rank up to R. A C∈R^(N×R) read produces CᵀS with shape R×P before subsequent combination.

For small R relative to N and P, more arithmetic can reuse the same retained N×P state. Whether this improves actual latency depends on memory traffic, tensor shapes and kernels. The paper's parameter-sharing scheme controls projection growth; simply multiplying every projection width by R is not its whole design.

**Inline figure: endpoint interpolation, rotating pair, rank-R write.** Use three separate panels with their own entities. Link the first to the discretization comparison, the second to the shrinking spiral, and the third to SSD's outer-product write. A single generic architecture rectangle would conceal what each innovation changes.

The complete architecture also adjusts normalization, learned B/C biases and the local convolution arrangement. Its experiments report particular language-model and state-tracking settings; they do not establish that these mechanisms beat every alternative on all continuous signals, hardware or deployment tasks.

## 10. Practice, explain and transfer

These exercises change the examples. Work through the arithmetic or design decision, then use the hints and explained solutions to diagnose the step you missed. The live labs remain available for free exploration.

### 1. Recover both output paths

Ā=.4, B̄=2, C=−1, D=.5, initial state h₋₁=3 and inputs [1,−2]. Find both states and outputs. Then predict the outputs if C is set to zero.

<details>
<summary>Hint</summary>

Update the state first. Compute Ch and Du separately before adding them.

</details>
<details>
<summary>Solution</summary>

h₀=.4×3+2=3.2 and y₀=−3.2+.5=−2.7. Next h₁=.4×3.2−4=−2.72 and y₁=2.72−1=1.72. With C=0, outputs are simply .5u=[.5,−1], regardless of the evolving state. This is a useful direct-feedthrough check.

</details>

### 2. A missing initial condition

For Ā=.5,B̄=1,C=1,D=0, input [0,0,0] and h₋₁=8, a convolution-only implementation returns three zeros. Give the correct outputs and identify the missing term.

<details>
<summary>Hint</summary>

Zero input does not imply zero state. Apply Ā once before the first read.

</details>
<details>
<summary>Solution</summary>

The outputs are [4,2,1]. The missing term is C Ā^(t+1)h₋₁. A zero-input test becomes a zero-output null only when the initial state and any biases/direct effects permit it.

</details>

### 3. Can a linear system remember a fixed delay?

Construct a four-state system that returns the input from three steps earlier. Apply it to [3,−1,4,2,8]. Why does this not solve the general “retain the most recent marked item across arbitrary gaps” task?

<details>
<summary>Hint</summary>

Write into the first coordinate and shift every coordinate to the next one. Read the last coordinate.

</details>
<details>
<summary>Solution</summary>

Use Ā with ones on its first subdiagonal and zeros elsewhere, B̄=[1,0,0,0]ᵀ, C=[0,0,0,1], D=0 and zero initial state. The result is [0,0,0,3,−1]. The delay is always three steps. A marker-dependent gap is not a fixed lag; the relevant selection must be supplied by a suitable nonlinear or input-dependent mechanism.

</details>

### 4. An exact gate and a different injection

Let A=−1,B=1, Δ=ln 4, h_previous=2 and u=10. Find the exact held-input update. Then calculate the update using ΔB injection. Are the answers the same?

<details>
<summary>Hint</summary>

exp(−ln 4)=1/4. Exact ZOH writes (1−1/4)u.

</details>
<details>
<summary>Solution</summary>

Exact ZOH gives .25×2+.75×10=8. The ΔB rule gives .5+10 ln 4≈14.362944. It is a different discrete operator here. The exact scalar gate is .75; calling the other injection “approximately exact” without considering Δ and A would conceal a substantial difference.

</details>

### 5. Read a two-step SSD state

Start at zero. Let a₀=.3,a₁=.2; b₀=[1,2], b₁=[−1,1]; c₀=[0,1],c₁=[2,1]; and scalar values v₀=3,v₁=4. Compute the two outputs by recurrence and by the influence coefficients.

<details>
<summary>Hint</summary>

At step 1, the old input's coefficient is .2(c₁ᵀb₀). The current input's coefficient is c₁ᵀb₁.

</details>
<details>
<summary>Solution</summary>

S₀=[3,6]ᵀ, so y₀=6. S₁=.2[3,6]ᵀ+4[−1,1]ᵀ=[−3.4,5.2]ᵀ and y₁=−1.6. The matrix calculation gives .2×4×3+(−1)×4=2.4−4=−1.6. A negative coefficient is valid; these are not softmax probabilities.

</details>

### 6. Choose a useful experiment before seeing its answer

The diagonal trajectory models reach their lowest validation loss at the final allowed epoch. Propose a follow-up that tests whether training budget is limiting, without repeatedly using the existing assessment set to select settings.

<details>
<summary>Hint</summary>

Separate model-selection evidence from final assessment. State budgets and seeds before the comparison.

</details>
<details>
<summary>Solution</summary>

Predeclare several training budgets and seeds, select among them using fitting/validation data, and reserve an untouched assessment source or a properly designed outer evaluation for the final decision. Keep preprocessing and duplicate grouping identical. Reusing the already-inspected assessment results as feedback would make them development evidence; acknowledge that change instead of calling each new result an untouched test. Longer training might help or overfit, so the protocol should allow either outcome.

</details>

### 7. Count a deployment state

A real-state model has 20 layers, width 128, state size 32 and two-byte state coordinates, processing batch 3. Count its recurrent state bytes and MiB. Name two memory costs excluded from this calculation.

<details>
<summary>Hint</summary>

Multiply batch, layers, width, state size and bytes. One MiB is 1,048,576 bytes.

</details>
<details>
<summary>Solution</summary>

3×20×128×32×2=491,520 bytes=.46875 MiB. Parameters and temporary activations are excluded; a short-convolution cache or allocator workspace are other possible omissions. A complex64 state would use eight bytes per stored complex coordinate, not two.

</details>

### 8. Trace a changing oscillator

With state [1,0], no injection, no decay and rotation π/2 at each step, give the next four states. Explain why a positive scalar decay times the identity cannot produce this trajectory from the same initial state.

<details>
<summary>Hint</summary>

A quarter-turn maps [x,y] to [−y,x].

</details>
<details>
<summary>Solution</summary>

The states are [0,1],[−1,0],[0,−1],[1,0]. Positive scalar multiplication preserves the vector's direction and cannot generate those quarter-turns. A real 2×2 rotation matrix can; complex notation is an equivalent compact representation, not a requirement to abandon real arithmetic.

</details>

### 9. Improve the real-data task for a stronger claim

You want to claim the movement classifier works on previously unseen people. Does the existing random row split answer that question? Specify the metadata and split you would need.

<details>
<summary>Hint</summary>

The unit of generalization should determine which records stay together.

</details>
<details>
<summary>Solution</summary>

No. We need performer identifiers and a protocol that keeps all recordings from an assessment performer outside fitting and model selection. Session identity may also matter, depending on the claim. Exact duplicate grouping remains necessary but does not replace person-level grouping. Since the retained dataset lacks those IDs, this stronger claim cannot be recovered merely by changing the random seed.

</details>

You are ready to continue when you can explain the write/retain/read paths, derive a kernel with correct initial conditions, recognize when content dependence removes fixed convolution, and reconcile one SSD chunk boundary. Continue to [RWKV & Linear Attention Models](/learn/path/full-curriculum/rwkv-linear-attention-models?module=deep-learning-fundamentals), which builds a recurrent state through another weighted-memory construction.

## References and other ways to learn

Choose a route based on the part you want to understand more deeply. These deepen the explanations; none is required to follow the worked examples on this page.

- **A literate S4 implementation:** [Rush and Karamcheti, The Annotated S4](https://srush.github.io/annotated-s4/). Its recurrent/convolutional correspondence and separate advanced implementation branch are useful after §3. This lesson's revision reviewed the tutorial's published source and its opening implementation route; it did not run the JAX/Flax project. Our own CPU programs and their recorded environments remain the reproducible route here.

- **Continuous systems and the full structured kernel:** [Gu, Goel and Ré, S4](https://arxiv.org/pdf/2111.00396). Read §2 for conventions and discretization, then §3 for the computational reason behind normal-plus-low-rank structure. Follow our two-mode example first; the kernel proof is a deeper branch.
- **What memory coefficients mean:** [Gu and colleagues, HiPPO](https://arxiv.org/pdf/2008.07669). §§2–3 start from approximation under a measure and derive online updates. Read with the polynomial-reconstruction figure beside you; keep the 1/t factors in the scaled Legendre equation.
- **A more accessible diagonal implementation route:** [Gu and colleagues, S4D](https://arxiv.org/pdf/2206.11893). §3 separates discretization, kernel computation and real/complex choices; §4 explains why initialization is more than merely choosing stable eigenvalues.
- **Shared-state MIMO and scans:** [Smith, Warrington and Linderman, S5](https://arxiv.org/pdf/2208.04933), §3. Compare the P-dimensional shared state with a bank of independent channel states.
- **Selection and the actual block:** [Gu and Dao, Mamba](https://arxiv.org/pdf/2312.00752), §§3.1–3.6 and appendix C. The fixed-spacing versus selective-copy distinction is especially useful; compare the mathematical gate derivation with the separately linked reference scan.
- **Duality from two directions:** [Dao and Gu, SSD/Mamba-2](https://arxiv.org/pdf/2405.21060), §§5–7, and the creator's [model article](https://tridao.me/blog/2024/mamba2-part1-model/) and [algorithm article](https://tridao.me/blog/2024/mamba2-part3-algorithm/). The articles explain state shape and the four chunk steps with code. The algorithm article also discusses why its shortest pedagogical interchunk implementation is not the optimized work-efficient scan.
- **A spoken alternative with a transcript:** [Albert Gu's conversation on state-space models](https://www.cognitiverevolution.ai/the-state-space-model-revolution-with-albert-gu/) includes an embedded video, chapter list and transcript. The state discussion around 30:59 and training-versus-inference discussion around 39:05–49:20 complement §§1,3 and 8; the Mamba-2 comparison follows. The lesson author read the relevant transcript and verified the host page, rather than claiming to have watched the recording. Treat its 2024 outlook as historical context.
- **The current family extension:** [Mamba-3](https://arxiv.org/pdf/2603.15569), §§3.1–3.4. Read each new recurrence ingredient separately, then inspect the experimental conditions before interpreting the paper's reported gains.
- **Implementation source:** [state-spaces/mamba](https://github.com/state-spaces/mamba) and its [selective scan reference](https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/selective_scan_interface.py). Use the current environment instructions for full kernels; the lesson's CPU reference remains a separate, reproducible teaching artifact.
- **Data and reproducible results:** [UCI Libras Movement](https://archive.ics.uci.edu/dataset/181/libras%2Bmovement), our [data provenance](data-provenance.md), [recorded training results](trajectory-results.json), [mechanism calculations](mechanism-results.json) and [complete training program](trajectory_state_models.py). These let you inspect the actual row roles, errors and calculations behind the local examples.
