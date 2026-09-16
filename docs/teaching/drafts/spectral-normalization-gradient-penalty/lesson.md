# Spectral Normalization & Gradient Penalty

A useful learning signal must respond to a meaningful change in the input. If it reacts enormously to an almost invisible change, optimization can become erratic. If it barely reacts to anything, it cannot tell another model how to improve. This lesson studies two ways to shape that sensitivity: rescale the transformations inside a network, or penalize its measured input gradients.

The preceding [Boltzmann Machines & Restricted Boltzmann Machines](/learn/path/full-curriculum/boltzmann-machines-restricted-boltzmann-machines-rbm?module=deep-learning-fundamentals) lesson assigned probability through energy and a normalizing constant. Here a generator produces samples directly, and a second network supplies a learning signal by comparing generated and recorded examples. That second network makes sensitivity a practical concern.

**First pass:** read §§1–6 and the experiment interpretation, then attempt practice 1–6. You need vector lengths, matrix multiplication and the idea that a derivative measures change; these are refreshed below. The normalization derivative, robustness certificate, GroupSort and continuous-time connection are deeper branches. You can inspect the recorded experiment before running its complete program.

By the end, you should be able to explain what each method controls, implement the two training updates without reversing their signs, recognize a misleading “1-Lipschitz” claim, and assess generated samples separately from critic regularization.

## 1. A critic that gives directions

Imagine a generator producing two measurements of a handwritten digit: average ink on its left half and on its right half. Its input is a short random vector \(z\); its output \(G_\theta(z)\) is a proposed measurement pair. The parameters \(\theta\) determine which pairs it tends to produce. A **critic** \(f_\phi(x)\) gives a scalar score to a pair \(x\). Its parameters \(\phi\) are trained to give higher average scores to recorded pairs than to generated pairs.

The generator then changes its parameters to raise the critic's scores on its generated pairs. It does not need a target pair for each random input. It needs a direction through the differentiable chain

\[
\theta\longrightarrow G_\theta(z)\longrightarrow f_\phi(G_\theta(z)).
\]

**Visual: two update paths.** Recorded and generated points enter the same score surface. A critic update changes the surface while the generated points are held fixed. A generator update moves the points on a temporarily fixed surface. Label the parameters that may change on each path; the arrows carry derivatives as well as values.

For the Wasserstein form used here, gradient-descent software minimizes

\[
L_D=\mathbb E[f_\phi(G_\theta(z))]-\mathbb E[f_\phi(x)]+R(\phi),
\qquad L_G=-\mathbb E[f_\phi(G_\theta(z))].
\]

\(R\) is a critic regularizer when one is used. The minus sign in \(L_G\) matters: minimizing a negative score raises the score. During a critic update, generated inputs are detached from the generator's derivative graph. During a generator update, freeze the critic's parameters but retain derivatives with respect to its input. Detaching the critic's output would remove the generator's learning signal.

Take a one-dimensional example: recorded data are at zero, the generator emits \(\theta=2\), and the fixed critic is \(f(x)=-x\). Recorded score is zero and generated score is −2. The generator loss is \(\theta\), its derivative is one, and a step of size .1 moves the output to 1.9, toward the data. Reversing the loss sign moves it to 2.1.

Why constrain the critic at all? If any score difference is useful, multiplying all scores by a million appears better to its maximization objective. A limit on sensitivity gives score differences a meaningful scale.

### Transport distance supplies that scale

The **Wasserstein-1 distance** asks for the least cost of moving probability mass from one distribution to another. Moving mass \(a\) through distance \(d\) costs \(ad\). For point masses at zero and \(\theta\), the answer is \(|\theta|\): the distance decreases continuously as the generated point approaches the recorded point.

Under the appropriate transport assumptions, the same quantity can be written as the largest recorded-minus-generated score difference over all 1-Lipschitz critics. On Euclidean space, finite first moments ensure the usual distance is finite. Continuity with respect to generator parameters needs corresponding regularity; it is not a statement about every arbitrary parameterized distribution. A finite trained network searches a restricted family for a limited number of updates, so its observed score difference is not automatically the exact transport distance. [WGAN, §2–3](https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf)

For comparison, the ideal Jensen–Shannon divergence between two different point masses stays at \(\log 2\), then becomes zero when they coincide. That explains one difficulty with a particular idealized objective. It does not prove that every practical GAN has zero generator gradient: finite critics and the widely used non-saturating generator loss change the argument. [WGAN-GP, §2.1–2.2](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)

## 2. What a sensitivity limit says

A function is **\(L\)-Lipschitz**, for specified input and output norms on a specified domain, when

\[
\|f(x)-f(y)\|\le L\|x-y\|\quad\text{for every allowed }x,y.
\]

If the input moves .02 units and \(L=3\), the output moves at most .06 units. The bound need not be attained. A constant function is 1-Lipschitz as well as 0-Lipschitz: “1-Lipschitz” means an upper bound of one, not that every slope is exactly one.

For a continuously differentiable scalar function on a convex region, a gradient norm bounded by \(L\) everywhere gives the corresponding Lipschitz bound. Integrate the directional derivative along the segment from \(x\) to \(y\). Each small output change is bounded by \(L\) times the small input movement, so the accumulated change has the same bound. The same reasoning applies to ordinary continuous piecewise-linear networks by integrating along their pieces, including across corners. Checking a few sample gradients, however, does not establish the premise everywhere.

Corners are allowed. \(|x|\) and ReLU are 1-Lipschitz even though their ordinary derivative does not exist at zero. The concern is bounded change, not whether a graph has a sharp-looking corner.

**Visual: slope envelope.** Draw two points on an editable piecewise-linear curve and the cones of allowable output differences. A second view marks sampled derivatives as isolated observations. Moving a knot outside the sampled region can violate the global bound without changing any observed gradient. This prevents the picture from treating samples as a proof.

### From layers to a whole network

For Euclidean norms, an affine layer \(Wx+b\) has Lipschitz constant \(\|W\|_2\), the matrix's largest singular value. The bias cancels in differences. For composition, multiply valid layer bounds. ReLU and tanh have bounds one; sigmoid has bound one-quarter; leaky ReLU with negative slope \(\alpha\) has bound \(\max(1,|\alpha|)\). GELU does not have a global bound of one.

This gives a useful ledger for a computation graph:

| Construction | Valid bound when the component bounds apply |
| --- | --- |
| Composition \(g(f(x))\) | \(L_gL_f\) |
| Sum \(f(x)+g(x)\) | \(L_f+L_g\) |
| Residual block \(x+g(x)\) | \(1+L_g\) |
| Concatenation \((f(x),g(x))\), Euclidean output | \(\sqrt{L_f^2+L_g^2}\) |
| Scalar multiplication \(af(x)\) | \(|a|L_f\) |

A residual branch with bound .5 can produce a block with bound 1.5; the example \(g(x)=.5x\) attains it. LayerNorm, learned gains, pooling and attention also belong in this accounting. Normalizing only dense weights does not certify every other operation in the graph.

Bounds can be loose. Compose \(A=\operatorname{diag}(3,1/3)\) with \(B=\operatorname{diag}(1/3,3)\). The product-of-norms bound is nine, while \(BA=I\) has norm one. The direction stretched by the first map is contracted by the second. A small sampled derivative and a large valid upper bound therefore need not contradict each other.

## 3. Spectral normalization: control the strongest stretch

Feed every unit-length vector in two dimensions through \(W=\operatorname{diag}(3,1)\). The unit circle becomes an ellipse with semi-axes three and one. A vector along the first axis is stretched threefold; one along the second is unchanged. The **spectral norm** is the largest stretch:

\[
\sigma_1(W)=\max_{\|v\|_2=1}\|Wv\|_2.
\]

For nonzero \(W\), exact unit spectral normalization uses

\[
\overline W=W/\sigma_1(W).
\]

Our ellipse now has semi-axes one and one-third. All singular values are divided by the same number. This does not make the map orthogonal, force every singular value to one, or change its rank. A target scale \(c>0\) instead uses \(cW/\sigma_1(W)\).

**Visual: circle, ellipse and spectrum.** Keep the same input vector visible before and after rescaling. Show the largest stretch beside all singular values. A separate Frobenius bar has height \(\sqrt{3^2+1^2}=\sqrt{10}\); it is a valid upper bound on the spectral norm, not an average of stretches. The exact values come from the matrices, not a stylized spectrum.

Normalizing to norm one can enlarge a matrix whose norm is already below one. If the intended operation is only to cap the norm, use \(W/\max(1,\sigma_1(W))\). Singular-value clipping is another operation: decompose \(W=U\Sigma V^T\), cap individual singular values, then reconstruct. Entrywise weight clipping changes matrix coefficients directly and has yet another effect. Its threshold does not set a network's Lipschitz constant to that same threshold.

The original spectral-normalization method uses a cheaper estimate of the strongest stretch during training. Its paper also studies other GAN losses, so spectral normalization is not tied exclusively to the Wasserstein objective. [Miyato et al., §2 and Appendix A](https://arxiv.org/pdf/1802.05957)

### Power iteration finds a direction as well as a number

For a nonzero left-side vector \(u\), repeat

\[
v\leftarrow\frac{W^Tu}{\|W^Tu\|_2},\qquad
u\leftarrow\frac{Wv}{\|Wv\|_2},\qquad
\widehat\sigma=u^TWv.
\]

Multiplication amplifies components associated with larger singular values. Repeated normalization prevents the vector itself from growing without bound. With a suitable starting component, the process approaches a leading singular direction. Convergence depends on the spectral gap; a starting vector exactly orthogonal to the leading subspace can miss it.

For \(\operatorname{diag}(3,1)\), start with \(u=(1,1)/\sqrt2\). One round produces \(\widehat\sigma=2.863564\). Dividing by that estimate leaves a true norm of \(3/2.863564=1.047645\), slightly above one. Starting with \(u=(0,1)\) instead yields an estimate of one forever in exact arithmetic, leaving true normalized norm three. The numerical trace makes both cases explicit.

Training commonly retains the previous vectors because weights often move incrementally. A cached vector can be useful; it is not a universal accuracy guarantee. Near-equal leading singular values slow convergence, and a changed matrix can invalidate a previously good direction. For a dense \(m\times n\) matrix, one round costs \(O(mn)\) arithmetic with \(O(m+n)\) vector storage beyond the weights. Calling the arithmetic \(O(m+n)\) confuses storage with work.

**Investigation: hide the strongest direction.** Edit a 2×2 matrix, choose an initial direction, and predict whether one iteration leaves a norm at most one. Compare its estimate with the exact small-matrix singular value. Rotate the initial vector slightly away from an uninformative direction and inspect the trajectory. A null comparison scales a nonzero matrix by a positive constant: exact unit normalization yields the same effective matrix.

### Deeper: why the normalization stays in the derivative graph

Let \(H=\partial L/\partial\overline W\), and assume a unique positive leading singular value with unit singular vectors \(u,v\). Since \(d\sigma_1=\langle uv^T,dW\rangle\), differentiating the quotient gives

\[
\frac{\partial L}{\partial W}
=\frac{1}{\sigma_1}\left(H-\langle H,\overline W\rangle uv^T\right).
\]

The second term accounts for how changing \(W\) also changes its scale. Treating the entire denominator as a detached constant loses that term. A practical approximation estimates \(u,v\) without differentiating through their iterative search, but computes \(u^TWv\) with \(W\) still in the graph. At a repeated leading singular value, the usual unique-vector derivative needs nonsmooth treatment; the displayed formula assumes uniqueness.

The companion calculation checks the derivative for \(W=[[2,1],[0,1]]\) against central differences, with maximum discrepancy below \(7\times10^{-12}\). The derivation explains the term; numerical agreement is a useful check on this particular implementation, not a proof for all matrices.

## 4. A convolution is larger than its stored kernel

A kernel is reused at many spatial locations. Flattening its stored coefficients into a matrix does not generally produce the matrix that maps an entire image to its entire output.

For input \((x_1,x_2,x_3)\) and valid stride-one kernel \([1,1]\),

\[
y=(x_1+x_2,x_2+x_3),\qquad
A=\begin{bmatrix}1&1&0\\0&1&1\end{bmatrix}.
\]

The stored row kernel has norm \(\sqrt2\). The full operator has norm \(\sqrt3\), because the shared middle input contributes to both outputs. Dividing the kernel by \(\sqrt2\) therefore leaves a full-operator norm of \(\sqrt{3/2}\approx1.224745\).

Now use four inputs and stride two: the two windows do not overlap. The full operator consists of two disjoint copies of that row kernel; dividing by \(\sqrt2\) gives norm one. A four-position circular stride-one convolution has full norm two before normalization, leaving \(\sqrt2\) after the same kernel rescaling. Padding, stride, spatial size and overlap are part of the operator definition.

**Visual: overlapping stencils become a matrix.** Selecting the middle input highlights both output contributions and the corresponding matrix column. Compare the valid, disjoint and circular operators using their actual singular values. Keep “stored kernel norm” and “full spatial operator norm” as separate labeled quantities.

This does not make kernel spectral normalization useless. It changes how strongly weights can act and is widely studied as a regularizer. It does mean that a certificate about the whole convolution requires an appropriate operator bound or computation. Fourier-based exact results for circular convolutions have their own boundary assumptions; they cannot silently be applied to every zero-padded convolution. [Sedghi et al., operator analysis](https://arxiv.org/pdf/1805.10408)

## 5. Gradient penalty: measure the function where it is sampled

The WGAN gradient penalty draws a recorded input \(x\), a generated input \(\widetilde x\), and \(\epsilon\sim U[0,1]\), then forms

\[
\widehat x=\epsilon x+(1-\epsilon)\widetilde x,\qquad
R_{GP}=\lambda\mathbb E\left[(\|\nabla_{\widehat x}f(\widehat x)\|_2-1)^2\right].
\]

It asks a direct question about the complete critic: how sensitive is its score at this interpolated input? The derivative is with respect to input coordinates, not the critic's parameters. Training then differentiates the penalty with respect to parameters, which requires a derivative graph through that first derivative.

Why target one? Under the transport theorem's conditions, an optimal critic has unit directional slope along relevant transport segments. Actual WGAN-GP samples random recorded/generated pairs, not a solved optimal transport coupling. The method uses that theory as motivation for a practical sampled regularizer. Its finite samples and soft penalty do not impose a global hard constraint. [Gulrajani et al., Proposition 1 and §4](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)

### Work through a penalty update

For \(f_w(x)=w^Tx\), the input gradient is simply \(w\). Let \(w=(3,4)\), whose Euclidean norm is five, and let \(\lambda=2\). The penalty is \(2(5-1)^2=32\). For nonzero \(w\), its parameter derivative is

\[
\nabla_wR=2\lambda(\|w\|_2-1)\frac{w}{\|w\|_2}=(9.6,12.8).
\]

A penalty-only descent step of size .1 gives \(w'=(2.04,2.72)\), norm 3.4 and penalty 11.52. It moved toward norm one without jumping directly there. In GAN training, the adversarial-loss gradient is added to this gradient; the result need not decrease the penalty every step.

Two-sided target-one penalty, an upper-bound penalty and a zero-centered penalty prefer different functions:

| Gradient norm \(r\) | \((r-1)^2\) | \(\max(0,r-1)^2\) | \(r^2\) |
| --- | ---: | ---: | ---: |
| 0 | 1 | 0 | 0 |
| .5 | .25 | 0 | .25 |
| 1 | 0 | 0 | 1 |
| 2 | 1 | 1 | 4 |

The first column penalizes a constant function even though it satisfies the 1-Lipschitz upper bound. The second does not penalize slopes below one. The third favors a zero gradient at the sampled locations. Those are different objectives, not alternative spellings of the same constraint.

### R1 and R2 change both the target and sampling location

R1 uses \(\frac\gamma2\mathbb E_{x\sim p_{data}}\|\nabla_x f(x)\|^2\); R2 uses the corresponding expectation on generated inputs. They are zero-centered penalties at different distributions. Moving WGAN-GP's target-one penalty onto real inputs alone does not turn it into R1. Their convergence analysis establishes local results under explicit assumptions near an appropriate equilibrium, not unconditional convergence of every large GAN. [Mescheder et al., §4](https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf)

Lazy application every \(k\) updates can multiply the regularizer by \(k\) to preserve its expected contribution under that sampling schedule. This does not make the finite optimizer trajectory identical, and optimizer adjustments may be needed. Choose and document the procedure rather than treating a popular interval as a theorem.

### A penalty can miss a steep region completely

Consider \(f(x)=x+4\operatorname{ReLU}(x-1)\). At sampled points −.5, 0 and .5, the derivative is one and the target-one penalty is zero. At \(x=2\), the derivative is five and the unweighted penalty is sixteen. The global Lipschitz constant is five.

**Investigation: move the unobserved kink.** Edit the kink location, extra slope and actual probe coordinates. Predict the penalty before revealing it. Compare moving the kink outside all probes with moving a probe into the steep region. A second view overlays target-one, one-sided and zero-centered penalties so that a change in target can be separated from a change in sampled region.

### Per-example gradients need per-example functions

The usual code obtains gradients of the sum of batch scores. If each score depends only on its own input, this yields each example's input gradient. Batch-dependent operations can break that interpretation.

For two scalar inputs, define \(f_1=(x_1-x_2)/2\), \(f_2=(x_2-x_1)/2\). Each score has self-derivative .5, but the gradient of \(f_1+f_2\) is zero. Batch centering has coupled the examples, and summing scores cancels their derivatives. This small Jacobian explains why ordinary training-mode BatchNorm is problematic in the standard WGAN-GP calculation. Per-example LayerNorm avoids that specific batch coupling, but its own sensitivity still depends on its formula, gain and epsilon.

## 6. Implement the mechanism without losing its derivatives

For dense layers, PyTorch's parametrization API attaches spectral normalization to the weight. Training-mode weight access updates estimated singular vectors; evaluation mode freezes those iterations. Thus the number of accesses is part of the procedure, not simply the number of optimizer steps. A shared forward for recorded and generated inputs is easy to reason about. [PyTorch 2.14 spectral normalization](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.spectral_norm.html)

The complete experiment below uses the maintained API instead of a custom wrapper. A custom implementation must correctly handle device/dtype buffers and multiple forwards before backward; mutating cached vectors that autograd still needs can invalidate the graph. Do not replace the standard implementation with a shorter wrapper merely to reduce displayed lines.

For gradient penalty, retain the returned tensor until \(L_D\) is differentiated. Converting it to a Python number with `.item()` is appropriate for logging after detaching, but not for the optimized loss. Use `create_graph=True` for the input derivative, detach generator-produced samples in the critic phase, and flatten all non-batch input dimensions when taking each gradient norm. Interpolation needs one scalar per example broadcast over that example's coordinates.

There are numerical choices too. A zero matrix has no nonzero singular direction and cannot be divided by its norm; a tiny estimate requires an explicit policy. The small investigation defines the zero matrix's normalized result as zero and reports “direction undefined.” The training program uses ordinary nonzero initialization and checks actual finite outputs. Reduced precision can affect both norm estimation and the higher-order derivative calculation; establish an FP32 reference and verify the intended mixed-precision path before using it. No unmeasured universal overhead percentage is needed to explain that extra derivatives cost work.

At export, evaluation mode plus `remove_parametrizations(layer, "weight", leave_parametrized=True)` retains the current effective weight. Removing the parametrization need not restore the raw unnormalized weight. The companion example verifies identical outputs before and after this operation on a small dense layer. A frozen approximate norm remains approximate; exporting it does not upgrade it into a certificate. [PyTorch removal contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.remove_parametrizations.html)

## 7. A complete experiment on recorded digit measurements

The question is modest: **how do three declared critic-regularization procedures behave while learning the distribution of two real ink measurements?** This is small enough to plot every recorded point and inspect a learned score surface.

The [offline CSV](digits-400.csv) contains 400 real 8×8 optical digit images, selected as the first 40 examples of each digit from the scikit-learn copy of the UCI dataset. Each integer pixel is between zero and sixteen. For each half-image, sum its 32 pixels and divide by \(32\times16=512\). The two resulting coordinates are average normalized ink on the left and right. Class labels are not used for training or splitting. [Dataset source and attribution](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

Different images sometimes give the same measurement pair. Keep these equal profiles in the same role, retaining their frequency. A seeded split of 394 unique profile groups produces **241 fitting images, 79 development images and 80 assessment images**. No coordinate standardization is learned from assessment data. The [provenance record](data-provenance.md) retains every source ID, group, split rule, dependency version and data hash. Writer-level independence is not established by this extract.

The generator has widths 2→24→24→2, ReLU hidden activations and sigmoid outputs, with 722 parameters. The critic has widths 2→24→24→1, leaky-ReLU slope .2 and an unrestricted scalar output, with 697 parameters. There is no BatchNorm. Compare entrywise clipping at .1, target-one gradient penalty with \(\lambda=10\), and one-iteration spectral normalization on each dense critic weight. All three use the same Wasserstein-style losses.

For each method, seeds 11, 29 and 47 give paired raw initializations and data/latent draws. Train for 600 generator updates with three critic updates per generator update, batch size 64 and Adam rate .001, betas \((0,.9)\). Gradient-penalty interpolation has a separate random stream so it does not alter the common data draws. These choices were fixed before the run. Development measurements are recorded at declared steps; they do not select a checkpoint. All nine final models are retained, including unfavorable outcomes.

### Measure a distributional discrepancy independently of critic loss

Draw the same 256 latent vectors for every final generator. Project generated and recorded pairs onto 64 equally spaced unit directions with angles \(j\pi/64\). In each one-dimensional projection, compute the empirical Wasserstein-1 distance, then average. Sorting and integrating the empirical cumulative-distribution difference gives each one-dimensional value, including when sample counts differ.

This is a **finite directional average of empirical W1**, not exact two-dimensional W1, FID or a log likelihood. Its units are normalized ink coordinates. It can miss differences between the chosen projections and is subject to finite-sample variation. A simple baseline samples 256 fitting profiles with replacement using a fixed seed; it needs no neural training. Its assessment discrepancy is **.010039**.

| Procedure | Seed 11 | Seed 29 | Seed 47 |
| --- | ---: | ---: | ---: |
| Entrywise clipping | .047078 | .031269 | .035174 |
| Gradient penalty | .150081 | .277581 | .318414 |
| Spectral normalization | .018989 | .013203 | .016699 |

Lower means closer under this metric. Spectral normalization is best among these nine neural runs, but the empirical-resampling baseline is better still. The experiment therefore does not show that a neural generator is necessary for this task. Nor does one fixed budget settle which regularizer is best after suitable tuning on a different problem.

The critic measurements answer another question:

| Seed 11 critic | Maximum gradient on 80 assessment points | Maximum on a 41×41 input grid | Product of exact effective matrix norms |
| --- | ---: | ---: | ---: |
| Clipping | .008889 | .018682 | .193269 |
| Gradient penalty | 1.006960 | 1.088335 | 3.471948 |
| Spectral normalization | .007496 | .237637 | 1.000043 |

The spectral product is slightly above one because training used an estimate. Its much smaller observed gradients illustrate a loose product bound. The gradient-penalty model's individual assessment gradient norms range from .737725 to 1.006960: their maximum is near the target, while its generated profiles remain poor. Better local sensitivity behavior is not equivalent to better generated data.

**Visual investigation: points, surface and derivatives.** Overlay recorded profiles and actual generated points on equal-scale ink axes. Keep the critic surface and its gradient arrows in a separate aligned panel. Selecting a latent vector reveals its generated profile; editing a coordinate moves the output through the saved complete generator. Compare model seeds and training checkpoints without smoothing away observed reversals. The baseline belongs on the same discrepancy chart.

For a null that tests the function rather than its label, swap the two latent coordinates and simultaneously swap the two columns of the first generator weight matrix. Every generated output remains unchanged. Swapping only the input coordinates generally changes the output. The saved models verify this distinction; it is a change of coordinate naming versus a change of input to a fixed function.

### Run the declared experiment

Save the CSV beside the program. The author run used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and PyTorch 2.14.0+cpu with one CPU thread. Install compatible packages in an isolated environment, then run `python critic-regularization-study.py`. The program reads the real extract, creates the roles, fits all nine models, measures them, and writes the weights and results. No dataset download or hidden training loop is required.

The [complete program](critic-regularization-study.py) is printed below. It intentionally uses small explicit training steps and saves the evidence needed to understand the results. The [separate sensitivity calculations](sensitivity-calculations.py) reproduce the exact examples and check frozen-model inference without fitting again.

```python
"""Declared small WGAN comparison on real two-coordinate digit measurements."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
STEPS, CRITIC_STEPS, BATCH = 600, 3, 64
RATE, PENALTY, CLIP = .001, 10., .1
SEEDS = (11,29,47)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2,24),nn.Linear(24,24),nn.Linear(24,2)])

    def forward(self, z):
        for layer in self.layers[:-1]:
            z = torch.relu(layer(z))
        return torch.sigmoid(self.layers[-1](z))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2,24),nn.Linear(24,24),nn.Linear(24,1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.leaky_relu(layer(x),negative_slope=.2)
        return self.layers[-1](x).squeeze(-1)


def gradient_penalty(critic, real, fake, rng):
    shape = (len(real),)+(1,)*(real.ndim-1)
    mixing = torch.rand(shape,generator=rng,dtype=real.dtype,device=real.device)
    points = (mixing*real.detach()+(1-mixing)*fake.detach()).requires_grad_(True)
    values = critic(points)
    gradients = torch.autograd.grad(values.sum(),points,create_graph=True)[0]
    norms = gradients.flatten(1).norm(dim=1)
    return ((norms-1)**2).mean()


def projected_distance(left, right):
    # A fixed finite directional average, not exact multivariate Wasserstein or FID.
    angles = np.arange(64)*np.pi/64
    directions = np.c_[np.cos(angles),np.sin(angles)]
    a,b = np.asarray(left)@directions.T, np.asarray(right)@directions.T
    return float(np.mean([wasserstein_distance(a[:,i],b[:,i]) for i in range(64)]))


def effective_layers(model):
    return [dict(weight=l.weight.detach().double().tolist(),bias=l.bias.detach().double().tolist())
            for l in model.layers]


def main():
    records = list(csv.DictReader((ROOT/'digits-400.csv').open(encoding='utf-8')))
    pixels = np.array([[int(row[f'pixel_{j}']) for j in range(64)] for row in records])
    ids = np.array([int(row['source_id']) for row in records])
    image = pixels.reshape(-1,8,8)
    left = image[:,:,:4].sum((1,2)); right = image[:,:,4:].sum((1,2))
    values = np.c_[left,right]/512.  # each half:32 pixels, known max16
    # Equal two-coordinate measurements stay in one role; their frequency is retained.
    grouped = {}
    for row,pair in enumerate(zip(left,right)):
        grouped.setdefault(tuple(int(v) for v in pair),[]).append(row)
    keys = list(grouped)
    order = np.random.default_rng(91).permutation(len(keys))
    first,second = int(.6*len(keys)),int(.8*len(keys))
    group_roles = dict(fit=order[:first],development=order[first:second],assessment=order[second:])
    roles = {name: np.array([i for k in groups for i in grouped[keys[k]]])
             for name,groups in group_roles.items()}
    data = torch.tensor(values,dtype=torch.float32)
    fit = data[roles['fit']]
    evaluation_z = torch.randn(256,2,generator=torch.Generator().manual_seed(2026))
    bootstrap = values[np.random.default_rng(2026).choice(roles['fit'],256,replace=True)]
    baseline = {name: projected_distance(bootstrap,values[rows]) for name,rows in roles.items()}
    results = []
    for method in ('clipping','gradient-penalty','spectral-normalization'):
        for seed in SEEDS:
            torch.manual_seed(seed)
            generator,critic = Generator(),Critic()
            if method=='spectral-normalization':
                for layer in critic.layers:
                    spectral_norm(layer,n_power_iterations=1)
            if method=='clipping':
                with torch.no_grad():
                    for parameter in critic.parameters():
                        parameter.clamp_(-CLIP,CLIP)
            opt_g = torch.optim.Adam(generator.parameters(),lr=RATE,betas=(0.,.9))
            opt_d = torch.optim.Adam(critic.parameters(),lr=RATE,betas=(0.,.9))
            draws = torch.Generator().manual_seed(seed+1000)
            gp_draws = torch.Generator().manual_seed(seed+2000)
            history = []
            for step in range(1,STEPS+1):
                critic.train()
                critic.requires_grad_(True)
                for _ in range(CRITIC_STEPS):
                    real = fit[torch.randint(len(fit),(BATCH,),generator=draws)]
                    with torch.no_grad():
                        fake = generator(torch.randn(BATCH,2,generator=draws))
                    # One joined forward: both groups use the same current SN weight.
                    scores = critic(torch.cat((real,fake)))
                    score_real,score_fake = scores[:BATCH],scores[BATCH:]
                    penalty = gradient_penalty(critic,real,fake,gp_draws) if method=='gradient-penalty' else scores.new_zeros(())
                    critic_loss = score_fake.mean()-score_real.mean()+PENALTY*penalty
                    opt_d.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    opt_d.step()
                    if method=='clipping':
                        with torch.no_grad():
                            for parameter in critic.parameters():
                                parameter.clamp_(-CLIP,CLIP)
                # Freeze critic parameters and spectral-vector updates, not input derivatives.
                critic.eval()
                critic.requires_grad_(False)
                generated = generator(torch.randn(BATCH,2,generator=draws))
                generator_loss = -critic(generated).mean()
                opt_g.zero_grad(set_to_none=True)
                generator_loss.backward()
                opt_g.step()
                if step in (1,100,300,600):
                    with torch.no_grad():
                        generated_eval = generator(evaluation_z).numpy()
                    history.append(dict(step=step,critic_loss=float(critic_loss.detach()),
                                        generator_loss=float(generator_loss.detach()),
                                        raw_gp=float(penalty.detach()),
                                        development_projected_w1=projected_distance(generated_eval,values[roles['development']]),
                                        generated=generated_eval.tolist()))
            critic.eval()
            with torch.no_grad():
                generated = generator(evaluation_z)
                critic_values = critic(data)
                singular_values = [torch.linalg.svdvals(l.weight).tolist() for l in critic.layers]
            probe = data[roles['assessment']].clone().requires_grad_(True)
            grad = torch.autograd.grad(critic(probe).sum(),probe)[0]
            grad_norm = grad.norm(dim=1)
            grid_values = torch.linspace(0,1,41)
            grid = torch.cartesian_prod(grid_values,grid_values).requires_grad_(True)
            grid_score = critic(grid)
            grid_grad = torch.autograd.grad(grid_score.sum(),grid)[0]
            # Save a fresh latent-input edit for the later frozen-model investigation.
            latent_pair = torch.tensor([[-.7,.4],[-.7,1.1]])
            with torch.no_grad():
                edited_outputs = generator(latent_pair)
            result = dict(method=method,seed=seed,history=history,
                          generator_parameters=sum(p.numel() for p in generator.parameters()),
                          critic_parameters=sum(p.numel() for p in critic.parameters()),
                          metrics={name: projected_distance(generated.numpy(),values[rows]) for name,rows in roles.items()},
                          generated=generated.tolist(),generator_layers=effective_layers(generator),
                          critic_layers=effective_layers(critic),critic_values=critic_values.tolist(),
                          critic_singular_values=singular_values,
                          matrix_product_bound=float(np.prod([v[0] for v in singular_values])),
                          assessment_gradient_norm=grad_norm.tolist(),
                          assessment_max_gradient=float(grad_norm.max()),
                          grid_coordinates=grid.detach().tolist(),grid_score=grid_score.detach().tolist(),
                          grid_gradient=grid_grad.detach().tolist(),grid_max_gradient=float(grid_grad.norm(dim=1).max()),
                          latent_intervention=dict(inputs=latent_pair.tolist(),outputs=edited_outputs.tolist()))
            results.append(result)
            print(method,seed,'projected W1',round(result['metrics']['assessment'],6),
                  'sample gradient max',round(result['assessment_max_gradient'],6),
                  'matrix product',round(result['matrix_product_bound'],6),flush=True)
    output = dict(protocol=dict(torch=torch.__version__,numpy=np.__version__,steps=STEPS,critic_steps=CRITIC_STEPS,
                               batch=BATCH,rate=RATE,penalty=PENALTY,clip=CLIP,seeds=SEEDS,rows=len(data),
                               unique_profile_groups=len(keys),roles={k:ids[v].tolist() for k,v in roles.items()},
                               collision_groups=[ids[g].tolist() for g in grouped.values() if len(g)>1],
                               csv_sha256=hashlib.sha256((ROOT/'digits-400.csv').read_bytes()).hexdigest()),
                  measurements=values.tolist(),source_ids=ids.tolist(),evaluation_latents=evaluation_z.tolist(),
                  bootstrap=dict(generated=bootstrap.tolist(),metrics=baseline),fits=results)
    (ROOT/'calculated-inputs.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('roles',{k:len(v) for k,v in roles.items()},'unique groups',len(keys),'baseline',baseline,flush=True)


if __name__=='__main__':
    main()
```

## 8. Useful connections beyond this GAN

### A robustness margin needs the right output difference

A classifier chooses the largest logit. Suppose its two logits at \(x=(2,0)\) are produced by \(F(x)=(x_1,x_2)\). The winning gap is two. The vector-valued map has Euclidean Lipschitz constant one, but the **difference** \(F_1-F_2\) has constant \(\sqrt2\). Any perturbation of norm strictly below \(2/\sqrt2=\sqrt2\) preserves the positive gap. Perturbation \((-1,1)\), whose norm is exactly \(\sqrt2\), reaches a tie.

More generally, if a joint logit vector has bound \(L\), a pairwise logit difference has bound at most \(\sqrt2L\). A positive gap \(m\) therefore gives the sufficient radius \(m/(\sqrt2L)\) for that competitor. If each logit separately has bound \(L\), the direct sum bound is \(2L\). For multiple competitors, take the smallest valid radius. Include input preprocessing and the domain in the bound. A power-iteration estimate or a sampled gradient maximum alone is insufficient evidence for this certificate.

This is a useful application of the same geometry: the matrix stretch becomes a bound on a decision change. It is not a claim that adding spectral normalization alone makes a classifier robust to every perturbation.

### Bounded slope can still limit expressive power

ReLU can remove a component's derivative entirely on its inactive side. A network constrained at every layer may lose useful gradient magnitude through repeated such operations. **GroupSort** instead sorts small groups of activations; within a region where their order is fixed, it permutes components and preserves their Euclidean length. Sorting supplies nonlinearity without discarding that local derivative magnitude.

For two values \((a,b)\), GroupSort returns \((\min(a,b),\max(a,b))\). Their order can change as inputs change, producing a nonlinear piecewise-defined function. This helps explain why the activation and the norm constraint must be designed together. The original universal-approximation theorem uses specified mixed norms: a first-layer \(p\to\infty\) bound and subsequent infinity-norm bounds. It does not establish that any spectrally normalized ReLU network, or every Euclidean GroupSort construction, approximates every Lipschitz function. [Anil et al., architecture and Theorem 3](https://proceedings.mlr.press/v97/anil19a/anil19a.pdf)

### Continuous-time sensitivity is about trajectories too

In an ODE \(dh/dt=f(h,t)\), a Lipschitz bound in \(h\), together with appropriate continuity conditions, helps establish uniqueness and bound how two trajectories separate. A typical bound is \(\|h(t)-\widetilde h(t)\|\le e^{Lt}\|h(0)-\widetilde h(0)\|\). An upper bound on growth is not a promise of contraction. The simple field \(f(h)=Lh\) has bounded slope but unbounded values as \(|h|\) grows and exponentially separating trajectories when \(L>0\).

The later [Neural ODE & Continuous-Depth Models](/learn/path/full-curriculum/neural-ode-continuous-depth-models?module=deep-learning-fundamentals) lesson develops that connection and the separate numerical-solver questions. A layer norm bound alone does not prescribe a safe solver step size.

### Diagnose the observed failure before changing methods

If a critic has non-finite values, first inspect input scales, actual norms and derivative paths. If a penalty stays near its target but samples remain poor, inspect the generated distribution, capacity, update balance and training trajectory. If a tight product bound removes too much sensitivity, consider whether the architecture and desired constraint fit the task. Combining spectral normalization and a sampled penalty can be meaningful because they act through different mechanisms.

Compare equal objectives and report changed choices. Critic-loss signs and offsets, penalty terms and output scales make raw losses across different setups difficult to compare. For a binary discriminator, a summed real/fake BCE near \(2\log2\) is the value obtained by .5 predictions, not proof of perfect separation. A near-zero Wasserstein-style critic difference can indicate indistinguishable distributions or an uninformative critic. Use independent data-space or task-appropriate evaluation to tell these possibilities apart.

## 9. Practice and transfer

Attempt each question before opening its help. The early questions check mechanisms; later ones ask you to diagnose a system.

### 1. A changed matrix

For \(W=\operatorname{diag}(4,2)\), find its spectral norm, Frobenius norm and exact unit-normalized singular values. Then normalize \(.2I\): does it shrink?

<details><summary>Hint</summary>
For a diagonal matrix with nonnegative entries, the entries are its singular values. Exact unit normalization divides by the largest.
</details>
<details><summary>Solution</summary>
The norms are 4 and √20; normalized singular values are 1 and .5. The matrix .2I becomes I, so it grows. A cap-only operation leaves .2I unchanged. Normalization to a boundary and projection into a bounded set are different operations.
</details>

### 2. A direction the estimator cannot see

For \(W=\operatorname{diag}(2,5)\), start power iteration at \(u=(1,0)\). What estimate persists? What is the true norm after dividing by it? Would adding a small second component change the long-run behavior?

<details><summary>Hint</summary>
Track which coordinates matrix multiplication can create from a zero coordinate.
</details>
<details><summary>Solution</summary>
The estimate stays 2 and the resulting true norm is 2.5. In exact arithmetic the leading direction has no component to amplify. A nonzero second component allows repeated multiplication to amplify that direction relative to the first, eventually approaching estimate 5. The number of steps depends on the initial component and spectral gap.
</details>

### 3. A valid bound through a residual path

Two consecutive linear layers have bounds .8 and .6, with ReLU between them. They form a residual branch added to the input. Give a valid block bound. Does a learned multiplier of 2 outside the block preserve it?

<details><summary>Hint</summary>
First compose the branch, then account for addition, then the multiplier.
</details>
<details><summary>Solution</summary>
The branch bound is .48; the residual block bound is 1.48; the scaled block bound is 2.96. These are upper bounds, not assertions that some input pair must attain them. A claim of .48 for the whole residual block omitted the identity path.
</details>

### 4. A penalty that sees the wrong region

Let \(f(x)=x+2\operatorname{ReLU}(x-2)\). Sample only 0, 1 and 1.5. What is the unweighted target-one gradient penalty? What changes if the last probe moves to 3? What is the global Lipschitz constant?

<details><summary>Hint</summary>
The extra slope begins only after the kink; average the three squared deviations.
</details>
<details><summary>Solution</summary>
The first penalty is 0. With probes 0, 1, 3 the slopes are 1, 1, 3, so the mean penalty is 4/3. The global bound is 3. Keeping all probes below 2 leaves the violation unobserved; increasing the penalty coefficient cannot penalize a region that this sample never measures.
</details>

### 5. Repair the optimization graph

A training loop calculates `penalty = gradient_penalty(...).item()`, adds it to the critic loss, and detaches `critic(generator(z))` during the generator update. Explain both failures and their repairs.

<details><summary>Hint</summary>
Distinguish a logged number from a differentiable tensor, and frozen parameters from frozen inputs.
</details>
<details><summary>Solution</summary>
The Python scalar carries no parameter derivative, so that penalty cannot regularize the critic. Keep its tensor in the loss; detach only a separate value for logging. Detaching the generator's scored output removes its learning path. Freeze critic parameters while preserving its derivative with respect to the generated input. Detach generated samples only when they serve as fixed inputs for the critic phase.
</details>

### 6. Interpret the real experiment

The seed 11 gradient-penalty critic has assessment gradient maximum about 1.007, yet generated-profile discrepancy is .150081. The empirical-resampling baseline gives .010039. What conclusion is supported, and what comparison is still missing?

<details><summary>Hint</summary>
The three numbers measure two different questions; one setting per procedure does not isolate its best possible performance.
</details>
<details><summary>Solution</summary>
The largest measured gradient is near the requested target, but the resulting generator fits these profiles poorly under the declared directional metric. The simple baseline is better in this experiment. That does not refute gradient penalty in general or prove spectral normalization universally superior. A broader comparison would predeclare a development-based tuning budget, keep assessment separate, compare several seeds and evaluate the actual intended data representation. Producing realistic digit images is a different task from producing two ink averages.
</details>

### 7. Convolution changes when windows overlap

For the four-input stride-two kernel \([2,2]\), calculate the full operator norm and the norm after normalizing the stored kernel. Why is the answer different for a valid stride-one application to three inputs?

<details><summary>Hint</summary>
Write the full matrices. Scaling a matrix by 2 scales every singular value by 2.
</details>
<details><summary>Solution</summary>
The disjoint operator is [[2,2,0,0],[0,0,2,2]], with norm 2√2. The stored kernel has that same norm, so normalization gives full norm 1. The overlapping three-input operator has norm 2√3; normalization by 2√2 leaves √(3/2). Weight sharing and overlap, not the number of stored coefficients alone, determine the full map.
</details>

### 8. A robustness calculation with units

A two-logit network has a verified joint Euclidean Lipschitz upper bound 2 on normalized inputs and a winning logit gap .8. Give a sufficient perturbation radius in normalized-input units. What else is needed before stating a radius in raw sensor units?

<details><summary>Hint</summary>
Use a bound for the difference of two coordinates, then compose preprocessing.
</details>
<details><summary>Solution</summary>
The sufficient strict radius is .8/(2√2)≈.282843. Raw-unit interpretation needs the normalization map and its operator bound, along with the domain on which the network bound applies. If coordinate scaling is anisotropic, one scalar conversion may be overly conservative; state the input norm and transform explicitly. The bound must be verified, not merely a sampled gradient maximum.
</details>

### 9. Plan a useful follow-up

Your generated profiles form a narrow curve while the recorded profiles occupy a broader region. Propose a next experiment that distinguishes limited generator capacity, insufficient training and an evaluation artifact without choosing settings using assessment results.

<details><summary>Hint</summary>
Separate interventions and keep the observation unit, roles and metric definitions fixed.
</details>
<details><summary>Solution</summary>
Keep the existing split and select a bounded development-only comparison: first longer training at fixed architecture, then a changed generator width or latent dimension with an explicit compute budget. Use paired seeds where possible and retain all outcomes. Inspect actual point clouds and add a predeclared complementary discrepancy or coverage measure; do not rename critic loss as quality. Choose using development evidence, then evaluate the selected procedure on untouched assessment data. Existing assessment results have already been inspected, so a strong new confirmatory claim would need fresh assessment data rather than pretending this set is unseen again.
</details>

**Ready to continue:** you can trace critic versus generator derivatives, distinguish exact norms from estimates and sampled measurements, calculate one normalization and one penalty update, and explain an unfavorable result without changing the question after seeing it. The next topic in the module is [Modern Hopfield Networks](/learn/path/full-curriculum/modern-hopfield-networks?module=deep-learning-fundamentals), which returns to energy-based retrieval and connects it to attention. It remains the next lesson even if publication timing differs.

## References and another way to learn

- [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/pdf/1802.05957), Miyato et al. Start with §2 and Appendix A for the method; Appendix F develops its derivative. Original 2018 experiments describe their own settings, not current hardware rankings.
- [Improved Training of Wasserstein GANs](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf), Gulrajani et al. Read the sampling rule, Algorithm 1 and no-BatchNorm explanation in §4 after working the penalty example.
- [Wasserstein GAN](https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf), Arjovsky et al. The point-mass example and §2–3 connect distance, continuity, critic constraints and the generator sign.
- [Which Training Methods for GANs Do Actually Converge?](https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf), Mescheder et al. Advanced reading for zero-centered penalties and the assumptions behind local convergence results.
- [The Singular Values of Convolutional Layers](https://arxiv.org/pdf/1805.10408), Sedghi et al. Follow the full-operator viewpoint and check circular-boundary assumptions before reusing a formula.
- [Sorting Out Lipschitz Function Approximation](https://proceedings.mlr.press/v97/anil19a/anil19a.pdf), Anil et al. Explains gradient-norm preservation and a precisely stated approximation theorem; compare its norm choices with the Euclidean examples here.
- [PyTorch spectral-normalization API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrizations.spectral_norm.html) and [parametrization removal](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.parametrize.remove_parametrizations.html). Versioned implementation references for the executed code; inspect train/eval behavior when changing versions.
- [Stanford CS236 GAN notes](https://deepgenerativemodels.github.io/notes/gan/). A shorter alternate introduction to the generator/discriminator game and sample-based evaluation. Its broad introductory simplifications should be read alongside the explicit assumptions in this lesson.
- [Build Basic GANs, DeepLearning.AI](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans). An alternate video-and-exercise route with introductory GANs and Wasserstein/gradient-penalty material; suitable after basic PyTorch. The course and its listed curriculum were checked, not all videos watched. Access to graded material may require enrollment; this lesson is self-contained. The course is also linked from [Stanford CS236G's schedule](https://cs236g.stanford.edu/).
- [Offline study and provenance](data-provenance.md). Download the real extract and complete programs to reproduce this lesson's own measurements; these are not published benchmark results.
