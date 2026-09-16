# Boltzmann Machines & Restricted Boltzmann Machines (RBM)

Imagine learning what plausible handwritten digits look like without being told which digit each image represents. A model could assign a score to every possible image, then make images with better scores more probable. It could also use the visible half of an image to reason about the missing half.

A **Boltzmann machine** does this with interacting random variables. A **restricted Boltzmann machine**, or RBM, removes particular connections so that some otherwise difficult calculations become simple. We will build a model with only three binary switches, calculate every probability, and then train a small model on real digit images. The small model is deliberately chosen so that we can check its approximate training methods against exact answers.

The previous [Graph Transformers & Geometric Deep Learning lesson](/learn/path/full-curriculum/graph-transformers-geometric-deep-learning?module=deep-learning-fundamentals) studied how architecture encodes relationships and symmetries. Here a graph has another job: it describes interactions in a probability distribution. A connecting line is not an attention weight or a causal claim.

**First pass:** sections 1–5 establish the mechanism, sections 6–7 carry out and interpret the experiment, and practice 1–5 checks your understanding. Section 8 and the later problems develop deeper connections. You need weighted sums, elementary probability and the idea of a gradient. We introduce the needed conditional probabilities, expectations and normalizing constants locally; no statistical physics background is assumed.

## 1. From interacting switches to a probability distribution

A binary variable can be zero or one. For an image, it might represent whether a particular pixel is dark enough to count as ink. A configuration is the full list of those zeros and ones. We want some configurations to be common, others rare, while leaving room for uncertainty.

Assign a real number $E(s)$ to a configuration $s$. We call this its **energy**, with lower energy meaning greater preference. In this lesson energy is a dimensionless model score, not a measured number of joules. Turn it into a probability using

$$p(s)=\frac{e^{-E(s)}}{Z},\qquad Z=\sum_{s'}e^{-E(s')}.$$

The sum visits every allowed configuration. $Z$, the **partition function**, makes all probabilities add to one. If two energies are 0 and $-\log 3$, their unnormalized weights are 1 and 3, so their probabilities are $1/4$ and $3/4$. An energy difference becomes a probability ratio.

Adding 100 to every energy multiplies every weight by the same $e^{-100}$. It changes $Z$ by that factor and leaves all probabilities unchanged. Thus an isolated energy value cannot establish how likely something is. Changing one state's energy relative to the others can.

For a general binary Boltzmann machine, one possible parameterization is

$$E(s)=-\sum_i c_i s_i-\sum_{i<j}J_{ij}s_is_j.$$

The bias $c_i$ favors switch $i$ being on. A positive interaction $J_{ij}$ favors the two switches being on together; a negative one discourages that joint event. Connections are undirected and counted once. A general Boltzmann machine may have sparse connections; it does not have to connect every pair.

For two visible switches with zero biases and $J_{12}=\log3$, the four configurations $00,01,10,11$ have unnormalized weights $1,1,1,3$. Therefore $p(11)=1/2$, while each marginal on-probability is $2/3$. Independence would predict $4/9$, which is different. **A Boltzmann machine can model dependence even without hidden variables** when visible-to-visible interactions are present.

We will instead use hidden variables to create useful dependencies while imposing a simpler graph.

## 2. What the restriction buys us

An RBM has observed **visible** variables $v_1,\ldots,v_D$ and unobserved **hidden** variables $h_1,\ldots,h_H$. Hidden variables can combine evidence from multiple pixels. They are learned latent features, not automatically digit labels or human-interpretable concepts.

Connections run between the two groups. There are no visible-to-visible or hidden-to-hidden connections. This is a **bipartite** graph. Our Bernoulli–Bernoulli RBM has binary variables on both sides and energy

$$E(v,h)=-a^Tv-b^Th-v^TWh.$$

$W$ is $D\times H$, $a$ contains $D$ visible biases, and $b$ contains $H$ hidden biases. These shapes are also the storage convention in the complete program. Some libraries transpose $W$; the equation, not a variable name, determines which axis means what.

**Visual: two rows of switches and an energy ledger.** Selecting a complete state highlights only the bias terms for on-switches and the interaction terms whose endpoints are both on. Beside the graph, a table adds those contributions. The graph has no arrows: we will use the same learned interaction in both directions.

### Condition on one layer; the other separates

Fix the visible switches. Everything involving a particular hidden switch $h_j$ becomes

$$-h_j\left(b_j+\sum_i v_iW_{ij}\right).$$

Write the expression in parentheses as $z_j$. Hidden state zero contributes weight 1; hidden state one contributes $e^{z_j}$. Normalizing these two alternatives gives

$$p(h_j=1\mid v)=\frac{e^{z_j}}{1+e^{z_j}}=\sigma(z_j).$$

The function $\sigma(z)=1/(1+e^{-z})$ is the logistic sigmoid. There is no term coupling two hidden switches after $v$ is fixed, so all their conditional probabilities factorize:

$$p(h\mid v)=\prod_j p(h_j\mid v),\qquad
p(v_i=1\mid h)=\sigma\!\left(a_i+\sum_jW_{ij}h_j\right).$$

We can therefore sample an entire hidden layer in parallel given the visible layer, then an entire visible layer given the hidden layer. Sampling a Bernoulli variable with probability 0.7 means drawing a fresh uniform number $u\in[0,1)$ and setting the state to one when $u<0.7$. The number 0.7 is a probability, not a possible state of that binary variable.

**Conditionally independent does not mean marginally independent.** Once we average over an unknown hidden layer, visible switches can become dependent. Think of two lamps driven by an unobserved common switch. Learning that one is on changes your belief about the common switch, which changes your expectation of the other lamp. This is a probabilistic analogy, not a statement that an undirected RBM identifies causes.

### A three-switch model you can completely inspect

Use two visible switches and one hidden switch. Set both visible biases and the hidden bias to zero, and both weights to $\log3$.

| Visible state | Weight with $h=0$ | Weight with $h=1$ | Sum over hidden state | Visible probability |
| --- | ---: | ---: | ---: | ---: |
| 00 | 1 | 1 | 2 | 0.1 |
| 01 | 1 | 3 | 4 | 0.2 |
| 10 | 1 | 3 | 4 | 0.2 |
| 11 | 1 | 9 | 10 | 0.5 |

All eight joint-state weights sum to 20. For visible state 11, $p(h=1\mid11)=9/10$. For 10 or 01 it is $3/4$, and for 00 it is $1/2$. The hidden state's marginal probability is $(1+3+3+9)/20=0.8$.

Given $h=0$, each visible switch has on-probability 0.5. Given $h=1$, it has on-probability 0.75. Consequently each visible marginal is $0.2(0.5)+0.8(0.75)=0.7$. But $p(11)=0.5\ne0.7^2$: the visible variables are dependent despite having no direct edge.

**Pause:** if both weights become zero while all biases remain zero, does the hidden switch still create visible dependence?

<details><summary>Reveal the reasoning</summary>

No. All eight joint configurations have equal weight. Every visible state has probability $1/4$, and both visible switches are independent fair Bernoulli variables. A hidden unit with no interaction cannot communicate evidence.

</details>

## 3. Free energy, exact normalization and what remains difficult

Observed data contain $v$, not $h$. We must add the probability of every hidden explanation, not choose the single best explanation. Define **free energy** by

$$e^{-F(v)}=\sum_h e^{-E(v,h)}.$$

For this RBM the sum factors:

$$e^{-F(v)}=e^{a^Tv}\prod_j\left(1+e^{b_j+v^TW_{:,j}}\right),$$

so

$$F(v)=-a^Tv-\sum_j\operatorname{softplus}\left(b_j+v^TW_{:,j}\right),
\quad\operatorname{softplus}(z)=\log(1+e^z).$$

Each factor adds the two possible states of one hidden switch. Multiplying those sums accounts for every hidden combination without enumerating them individually. In our example $e^{-F(11)}=10$ and $e^{-F(10)}=4$; the probability ratio is $10/4=2.5$ within this one model.

The product is also a way to see why several hidden features can jointly constrain a pattern. This connection is discussed as a product of experts in [Hinton's original technical report](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf). It differs from choosing one expert in a mixture; the RBM factors all contribute to the same visible configuration.

Computing $F(v)$ is inexpensive, but a normalized log probability still needs

$$\log p(v)=-F(v)-\log Z.$$

For $D$ binary visible variables, there are $2^D$ visible configurations. This is why exact normalization becomes difficult in many useful RBMs. The restriction simplifies conditionals and marginalizes one layer efficiently; it does not make every global sum cheap.

### Enumerate the smaller layer

There is a valuable exception. Instead of enumerating visible vectors, sum them out and enumerate hidden vectors:

$$Z=\sum_{h\in\{0,1\}^H}e^{b^Th}
\prod_{i=1}^{D}\left(1+e^{a_i+(Wh)_i}\right).$$

Our experiment has 64 visible variables but only eight hidden variables. There are $2^8=256$ hidden configurations. For each one, a 64-term product represents the sum over all visible configurations. We can compute an exact normalizer and exact model expectations without visiting $2^{64}$ images.

“Exact” here describes the finite sum being evaluated; ordinary floating-point rounding remains. Eight hidden units were chosen to make the comparison inspectable, not because they are a universal best size. Raising $H$ from 8 to 24 multiplies the enumeration count by $2^{16}=65,536$.

Use stable logarithmic calculations. Computing `log(1 + exp(z))` directly can overflow even when its mathematical answer is finite. `numpy.logaddexp(0, z)` evaluates softplus stably, and `scipy.special.logsumexp` handles sums of exponentials in log space. Merely replacing `log` with `log1p` does not fix overflow in an already computed `exp(z)`.

## 4. Learning: which co-occurrences need more probability?

We want observed examples to receive more probability. For one weight, differentiating the log probability gives

$$\frac{\partial\log p(v)}{\partial W_{ij}}
=v_i\,p(h_j=1\mid v)-\mathbb E_{p(v',h')}[v'_ih'_j].$$

The first term asks how often that visible/hidden pair is on together when the visible state is supplied by the data. The second asks how often the pair is on together under the model's own distribution. Training increases a weight when the data require more of that co-occurrence than the model currently supplies.

For a minibatch, average the first term across examples. The corresponding bias gradients are

$$\nabla_a\log p=\mathbb E_{\rm data}[v]-\mathbb E_{\rm model}[v],
\qquad\nabla_b\log p=\mathbb E_{\rm data}[p(h=1\mid v)]-\mathbb E_{\rm model}[h].$$

The names **positive phase** and **negative phase** refer to these two expectations. They do not mean that positive examples are labeled correct and negative examples are labeled incorrect. An unsupervised RBM needs no class label in this objective.

Why subtract the model term? Increasing a weight changes probabilities across the entire state space. The derivative of $\log Z$ accounts for that competition. If we only lowered energies of data examples, we could also lower many unwanted states and mistake growing unnormalized scores for improving probability.

### One fully checked update

Suppose the training observation is 11 in our three-switch model. Each positive weight statistic is $1(0.9)=0.9$. Each model statistic is

$$p(h=1)\,p(v_i=1\mid h=1)=0.8(0.75)=0.6.$$

Both weight gradients are 0.3. Both visible-bias gradients are $1-0.7=0.3$, and the hidden-bias gradient is $0.9-0.8=0.1$.

With learning rate 0.1, update **all parameters from the same old state**:

| Parameter | Before | After |
| --- | --- | --- |
| Each interaction | $\log3$ | $\log3+0.03$ |
| Each visible bias | 0 | 0.03 |
| Hidden bias | 0 | 0.01 |

Recomputing the complete distribution changes $\log p(11)$ from −0.69314718 to −0.65690173. The observation has become more probable. We checked every analytic gradient against scalar central differences; the largest discrepancy was below $2\times10^{-11}$. A bigger learning rate would not automatically preserve improvement.

**Visual: two co-occurrence ledgers.** Display the data expectation and the model expectation beside each edge, then subtract them. A learner edits the observed state or a model weight, predicts the direction of the next update, and recomputes both columns. The model column must not remain frozen after a parameter edit.

### Do we need to sample the positive hidden variables?

No: for a fixed binary data vector, $v_i p(h_j=1\mid v)$ is the exact conditional expectation. Using the probability reduces avoidable sampling noise. This is different from pretending all uncertain variables can be replaced by their means throughout a nonlinear chain. We will test that distinction next.

## 5. Gibbs sampling, contrastive divergence and persistence

For larger RBMs, exact model expectations can be impractical. A **Gibbs sampler** alternates two easy draws:

$$v^{(0)}\ \longrightarrow\ h^{(0)}\sim p(h\mid v^{(0)})
\ \longrightarrow\ v^{(1)}\sim p(v\mid h^{(0)})\ \longrightarrow\cdots.$$

This sequence is a Markov chain: its next state depends on its present state. With finite parameters, the binary RBM's conditional probabilities are strictly between zero and one; its alternating chain can eventually reach every state and has the model distribution as its stationary distribution. How quickly it approaches that distribution is the **mixing** question. A sampler may spend many steps near one region even though it is mathematically able to reach another.

Neither these conditional draws nor a within-model probability ratio requires $Z$. Difficult normalization and slow mixing are related computational concerns, not the same operation. Introducing a temperature changes the distribution; ordinary sampling at our fixed temperature one does not require an annealing schedule.

### CD-$k$: start near a data example

**Contrastive divergence**, CD-$k$, starts the chain at data, takes $k$ full hidden/visible transitions, and uses the resulting visible states for an approximate negative statistic. After the final visible draw, calculate the hidden probabilities again for that statistic. Treat the sampled state as fixed when forming this update; this is not backpropagation through discrete draws.

We can see the approximation without Monte Carlo noise in the three-switch model. Start at 11. The hidden draw is one with probability 0.9. If it is zero, the four visible outcomes each have probability 0.25. If it is one, their probabilities are $[1,3,3,9]/16$. Combining the cases gives

$$q_1=[0.08125,\ 0.19375,\ 0.19375,\ 0.53125].$$

Compare that with the stationary model distribution $[0.1,0.2,0.2,0.5]$. The chain retains too much probability on its initial state after one transition. Its expected negative weight statistic is 0.6234375, so expected CD-1 gives a weight update direction of $0.9-0.6234375=0.2765625$, instead of the exact 0.3.

| Full transitions | Probability of 11 | Expected weight gradient | Distance from stationary distribution* |
| --- | ---: | ---: | ---: |
| 0 | 1 | 0 | 0.5 |
| 1 | 0.53125 | 0.2765625 | 0.03125 |
| 2 | 0.5029296875 | 0.2978027344 | 0.0029296875 |
| 3 | 0.5002746582 | 0.2997940063 | 0.0002746582 |

*Total variation distance is half the sum of absolute differences between matching state probabilities. These entries come from multiplying an exactly enumerated four-state transition matrix, with parameters held fixed. They are not measured frequencies of four individual samples, or a general promise that three steps suffice.

**Lab: probability flow through four states.** Follow the full mass distribution as well as a single seeded particle. Change the starting state or interaction strengths, predict the next mass movement, then step. Separate “chain step” from “parameter update”; confusing the two hides what the approximation actually does.

Finite-step CD is a biased approximation to the likelihood gradient. Its usual update also omits a term arising from the parameter dependence of the reconstructed distribution in the proposed CD objective. It need not be the gradient of any scalar objective in general; [Sutskever and Tieleman, 2010](https://proceedings.mlr.press/v9/sutskever10a.html) analyze that distinction. The successful tiny example above does not establish global convergence.

### PCD: keep the model's particles alive

**Persistent contrastive divergence**, also called stochastic maximum likelihood in this setting, retains a collection of sampled states across parameter updates. Each minibatch supplies fresh positive statistics, while these persistent particles take more Gibbs steps for the negative statistics. They are not reset to the current data minibatch. The hope is that a chain already near the old model remains useful as parameters move. The method and its empirical motivation are described in [Tieleman's 2008 paper](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Tieleman.2008.pdf).

Persistence does not magically produce independent equilibrium samples. A rapidly changing model, strongly separated regions or too few particles can still produce poor estimates. CD-$k$ trades additional transitions for cost; PCD changes initialization and reuse. Neither is universally better for every budget, initialization and evaluation measure.

### Why a mean is not a Gibbs state

After our first transition, each visible mean is 0.725. Feeding $[0.725,0.725]$ through the hidden sigmoid gives approximately 0.831036. But averaging the hidden sigmoid across the actual four-state distribution gives 0.809375. The sigmoid of a mean is not generally the mean of a sigmoid.

Using visible probabilities in reconstruction can be a useful explicitly labeled heuristic. It is not an exact transition of this binary Gibbs chain. Hinton's [practical guide](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) discusses probability/state choices and practical monitoring; read those choices in the context of the intended objective. Our training sampler draws binary visible states, while our reconstruction diagnostic deliberately uses a deterministic mean-to-mean pass.

## 6. A real experiment whose likelihood we can actually calculate

We use real handwritten digit images from [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), by E. Alpaydin and C. Kaynak, distributed under CC BY 4.0. This is the small 8×8 dataset available through scikit-learn, not MNIST. The retained CSV contains the first 40 occurrences of each digit in that loader, with original one-based row identifiers and 64 pixel values from 0 to 16.

The task is **unsupervised probability modeling of binary images**. Turn a pixel on when its value is at least eight. This deterministic transformation discards grayscale information and defines what the model's likelihood means. A likelihood for these binary images is not a likelihood for the original grayscale measurements.

After binarization, source 228 duplicates source 12, and source 300 duplicates source 274. Keep the first occurrence before splitting. The remaining 398 distinct binary images are divided within digit using seed 91: eight per class for development, eight for assessment, and the remaining 238 for fitting. Labels are used only for this stratification; they never enter the RBM loss. The source IDs and transformations are in [data-provenance.md](data-provenance.md).

Writer identifiers are unavailable, so this is not a writer-independent evaluation. These images have also appeared in other lessons. Treat this as a reproducible classroom experiment, not a newly untouched benchmark or a comparison with published UCI scores.

### Protocol before results

Compare a 64-parameter independent-pixel Bernoulli baseline with three training procedures for the same 64-visible, eight-hidden RBM, which has 584 parameters. The baseline's pixel probabilities are $(n_{\rm on}+0.5)/(n_{\rm fit}+1)$. This small symmetric smoothing avoids infinite logits at always-off pixels. Initialize RBM visible biases to those logits, hidden biases to zero, and interactions to small independent normal values with standard deviation 0.01.

Run exact likelihood-gradient ascent, sampled CD-1, and sampled PCD-1 for seeds 11, 29 and 47. Each uses 300 epochs, batch size 64 and fixed step size 0.05, without momentum, weight decay, dropout or checkpoint selection. Corresponding seeds share initial parameters and minibatch orders. PCD keeps 64 particles; the final positive minibatch has 46 examples, so positive and negative means are normalized separately. No tuning used development or assessment outcomes.

We evaluate exact **negative log likelihood**, NLL, in natural-log units, or **nats per image**. Lower is better. A model assigning probability $p$ to an image incurs $-\log p$. The average concerns the actual binary images, not whether a nearest-looking reconstruction was produced.

| Method | Seed 11 assessment NLL | Seed 29 | Seed 47 |
| --- | ---: | ---: | ---: |
| Independent pixels | 24.311674 | Same fixed model | Same fixed model |
| Exact gradient | 20.496861 | 20.655667 | 20.531889 |
| CD-1 | 20.556534 | 20.682717 | 20.575317 |
| PCD-1 | 20.534554 | 20.751721 | 20.568239 |

All nine fitted RBMs improved on the independent baseline in this run. Exact-gradient training obtained the lowest assessment NLL for each matched seed, but CD-1 and PCD-1 trade places. Do not turn three seeds on 80 assessment images into a universal ranking. Nor does an exact gradient guarantee a global optimum or better generalization for every initialization.

For exact-gradient seed 11, fit/development/assessment NLLs are 19.791466, 20.650057 and 20.496861. The gap is a reason to monitor generalization, not evidence that every higher-capacity model necessarily fails. Full histories, per-image NLLs and all final weights are retained in [calculated-inputs.json](calculated-inputs.json).

### Complete runnable study

Save the following as [rbm-study.py](rbm-study.py), next to [digits-400.csv](digits-400.csv), and run `python rbm-study.py`. It needs Python, NumPy and SciPy; the recorded execution used Python 3.12.14 and NumPy 2.3.5 on CPU. The program defines the model, preprocessing, roles, all nine fits, exact checks and completion calculations. It does not download a dataset or a pretrained model. Matrix products use batch rows, $W$ stores visible-by-hidden weights, and updates are ascent because we maximize log probability.

~~~python
"""Small Bernoulli RBMs with exact normalization; author evidence, CPU only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parent
SEEDS = (11, 29, 47)
EPOCHS, BATCH, RATE = 300, 64, 0.05


def bits(count):
    return np.array(list(itertools.product((0., 1.), repeat=count)))


def free_energy(v, w, a, b):
    return -v @ a - np.logaddexp(0, v @ w + b).sum(axis=-1)


def hidden_distribution(w, a, b):
    h = bits(len(b))
    logits = a + h @ w.T
    log_mass = h @ b + np.logaddexp(0, logits).sum(axis=1)
    log_z = logsumexp(log_mass)
    return h, np.exp(log_mass-log_z), expit(logits), float(log_z)


def exact_negative(w, a, b):
    h, probability, pv, log_z = hidden_distribution(w, a, b)
    return pv.T @ (probability[:, None]*h), probability @ pv, probability @ h, log_z


def positive(v, w, b):
    ph = expit(v @ w + b)
    return v.T @ ph / len(v), v.mean(axis=0), ph.mean(axis=0)


def gibbs(v, w, a, b, rng):
    h = (rng.random((len(v), len(b))) < expit(v @ w+b)).astype(float)
    return (rng.random(v.shape) < expit(h @ w.T+a)).astype(float)


def conditional_missing(v, observed, w, a, b):
    h = bits(len(b))
    logits = a+h@w.T
    # Observed entries contribute their chosen states; missing entries are summed out.
    log_mass = h@b + logits[:, observed]@v[observed]
    log_mass += np.logaddexp(0, logits[:, ~observed]).sum(axis=1)
    ph = np.exp(log_mass-logsumexp(log_mass))
    result = v.copy()
    result[~observed] = ph @ expit(logits[:, ~observed])
    return result


def metrics(v, w, a, b):
    log_z = hidden_distribution(w, a, b)[3]
    nll = free_energy(v, w, a, b)+log_z
    # Mean -> mean reconstruction is a deterministic diagnostic, NOT a Gibbs draw.
    recon = expit(expit(v@w+b)@w.T+a)
    return dict(nll_nats_per_image=float(nll.mean()),
                mean_reconstruction_mse=float(np.mean((recon-v)**2)),
                per_image_nll=nll.tolist(), log_z=log_z)


def exact_checks():
    v = bits(2)
    w, a, b = np.full((2, 1), np.log(3.)), np.zeros(2), np.zeros(1)
    h, ph, pv, log_z = hidden_distribution(w, a, b)
    p = np.exp(-free_energy(v, w, a, b)-log_z)
    assert np.allclose(p, [.1, .2, .2, .5])
    joint = np.exp(v@a[:, None]+h@b + v@w@h.T)
    assert abs(joint.sum()-np.exp(log_z)) < 1e-12
    vh, vm, hm, _ = exact_negative(w, a, b)
    pos = positive(v[3:], w, b)
    grad = [pos[0]-vh, pos[1]-vm, pos[2]-hm]
    # Independent scalar central differences, every parameter in the tiny model.
    defects = []
    def objective(ww, aa, bb):
        return float(-free_energy(v[3], ww, aa, bb)-hidden_distribution(ww, aa, bb)[3])
    for group, theta in enumerate((w, a, b)):
        for index in np.ndindex(theta.shape):
            plus, minus = [x.copy() for x in (w,a,b)], [x.copy() for x in (w,a,b)]
            plus[group][index] += 1e-5
            minus[group][index] -= 1e-5
            defects.append(abs((objective(*plus)-objective(*minus))/2e-5-grad[group][index]))
    # Enumerate all alternating Gibbs paths, not sampled transition frequencies.
    visible_given_hidden = np.prod(pv[:, None, :]**v[None, :, :]
                                  *(1-pv[:, None, :])**(1-v[None, :, :]), axis=2)
    posterior = expit(v@w+b).ravel()
    transition = np.c_[1-posterior, posterior]@visible_given_hidden
    assert np.max(abs(p@transition-p)) < 1e-12
    q = np.array([0.,0.,0.,1.])
    traces = []
    for step in range(11):
        negative_vh = v.T@(q[:,None]*expit(v@w+b))
        traces.append(dict(step=step, distribution=q.tolist(),
                           weight_gradient=(pos[0]-negative_vh).ravel().tolist(),
                           total_variation=float(abs(q-p).sum()/2)))
        q = q@transition
    updated = [x+.1*g for x,g in zip((w,a,b),grad)]
    # Distribution ratios are invariant to adding a constant to all energies.
    shifted = np.exp(-free_energy(v,w,a,b)-100-logsumexp(-free_energy(v,w,a,b)-100))
    assert np.max(abs(shifted-p)) < 1e-12
    missing = conditional_missing(np.array([1.,0.]), np.array([True,False]), w,a,b)
    assert abs(missing[1]-5/7) < 1e-12
    # Mean-visible replacement differs from averaging the nonlinear hidden response.
    exact_hidden_after_one = traces[1]['distribution']@posterior
    mean_visible = np.array([.725,.725])
    mean_hidden = float(expit(mean_visible@w+b)[0])
    sharp_w, sharp_a, sharp_b = np.full((2,1),20.), np.full(2,-10.), np.array([-20.])
    sharp = metrics(v[3:],sharp_w,sharp_a,sharp_b)
    independent = metrics(v[3:],np.zeros((2,1)),np.full(2,np.log(9.)),np.zeros(1))
    return dict(visible_states=v.tolist(), hidden_states=h.tolist(), joint_mass=joint.tolist(),
                visible_probability=p.tolist(), hidden_probability=ph.tolist(),
                log_z=log_z, z=float(np.exp(log_z)), model_vh=vh.tolist(),
                model_v=vm.tolist(), model_h=hm.tolist(), gradient=[x.tolist() for x in grad],
                finite_difference_max_error=max(defects), transition=transition.tolist(), traces=traces,
                updated_parameters=[x.tolist() for x in updated],
                original_log_probability=objective(w,a,b), updated_log_probability=objective(*updated),
                conditional_second_given_first=missing[1],
                expected_hidden_after_one=exact_hidden_after_one, mean_replacement_hidden=mean_hidden,
                reconstruction_counterexample=dict(sharp_correlated=sharp,independent=independent))


def investigation_checks(output):
    w, a, b = np.full((2,1),np.log(3.)), np.zeros(2), np.zeros(1)
    changed_a = np.array([np.log(2.),0.])
    changed_probability = np.exp(-free_energy(bits(2),w,changed_a,b)
                                 -hidden_distribution(w,changed_a,b)[3])
    pos = positive(np.array([[1.,0.]]),w,b)
    neg = exact_negative(w,a,b)
    changed_gradients = [(p-n).tolist() for p,n in zip(pos,neg)]
    hidden_on = float(expit(2*np.log(3.)+np.log(2.)))
    after_one_11 = (1-hidden_on)*.25+hidden_on*.75**2
    zero_input = np.array([[0.,0.]])
    sharp = metrics(zero_input,np.full((2,1),20.),np.full(2,-10.),np.array([-20.]))
    independent = metrics(zero_input,np.zeros((2,1)),np.full(2,np.log(9.)),np.zeros(1))
    fit = output['fits'][0]
    ww, aa, bb = np.array(fit['weights']), np.array(fit['visible_bias']), np.array(fit['hidden_bias'])
    rows = list(csv.DictReader((ROOT/'digits-400.csv').open(encoding='utf-8')))
    row = next(r for r in rows if int(r['source_id'])==fit['intervention']['source_id'])
    v = np.array([int(row[f'pixel_{j}'])>=8 for j in range(64)],dtype=float)
    observed = np.array([j%8<4 for j in range(64)])
    base = conditional_missing(v,observed,ww,aa,bb)
    changed = v.copy(); changed[18]=1-changed[18]
    edited = conditional_missing(changed,observed,ww,aa,bb)
    return dict(bias_edit_probability=changed_probability.tolist(),observation10_gradients=changed_gradients,
                hidden_bias_ln2_next_probability11=after_one_11,
                input00_comparison=dict(sharp=sharp,independent=independent),
                fresh_completion=dict(source_id=int(row['source_id']),observed_index=18,
                                      original=base.tolist(),edited=edited.tolist(),
                                      max_missing_change=float(np.max(abs(edited[~observed]-base[~observed])))))


def main():
    records = list(csv.DictReader((ROOT/'digits-400.csv').open(encoding='utf-8')))
    retained, seen, dropped = [], {}, []
    for row in records:
        signature = tuple(int(row[f'pixel_{j}']) >= 8 for j in range(64))
        if signature in seen:
            dropped.append(dict(source_id=int(row['source_id']), retained_id=seen[signature]))
        else:
            seen[signature] = int(row['source_id'])
            retained.append((int(row['source_id']), int(row['digit']), signature))
    ids = np.array([r[0] for r in retained])
    labels = np.array([r[1] for r in retained])
    x = np.array([r[2] for r in retained], dtype=float)
    split_rng = np.random.default_rng(91)
    role = dict(fit=[], development=[], assessment=[])
    for digit in range(10):
        rows = split_rng.permutation(np.flatnonzero(labels==digit))
        role['fit'].extend(rows[:-16])
        role['development'].extend(rows[-16:-8])
        role['assessment'].extend(rows[-8:])
    role = {name: np.array(rows) for name,rows in role.items()}
    fit = x[role['fit']]
    mean = (fit.sum(axis=0)+.5)/(len(fit)+1)  # symmetric beta(.5,.5) posterior means
    initial_a = np.log(mean)-np.log1p(-mean)
    baseline = {name: float(-(x[rows]*np.log(mean)+(1-x[rows])*np.log1p(-mean)).sum(1).mean())
                for name,rows in role.items()}
    observed = np.array([j%8 < 4 for j in range(64)])  # fixed left half, all assessment images
    results = []
    for method in ('exact', 'cd1', 'pcd1'):
        for seed in SEEDS:
            w = np.random.default_rng(seed).normal(0,.01,(64,8))
            a, b = initial_a.copy(), np.zeros(8)
            order_rng, sample_rng = np.random.default_rng(seed+1000), np.random.default_rng(seed+2000)
            particles = (sample_rng.random((BATCH,64)) < mean).astype(float)
            history = []
            for epoch in range(1,EPOCHS+1):
                order = order_rng.permutation(len(fit))
                for start in range(0,len(fit),BATCH):
                    data = fit[order[start:start+BATCH]]
                    pos = positive(data,w,b)
                    if method == 'exact':
                        neg = exact_negative(w,a,b)[:3]
                    else:
                        negative_data = gibbs(data if method=='cd1' else particles,w,a,b,sample_rng)
                        neg = positive(negative_data,w,b)
                        if method == 'pcd1':
                            particles = negative_data
                    # Simultaneous ascent using statistics from the OLD parameter state.
                    w += RATE*(pos[0]-neg[0])
                    a += RATE*(pos[1]-neg[1])
                    b += RATE*(pos[2]-neg[2])
                if epoch in (1,10,50,100,300):
                    history.append(dict(epoch=epoch, **{name: metrics(x[rows],w,a,b)
                                                        for name,rows in role.items() if name!='assessment'}))
            evaluated = {name: metrics(x[rows],w,a,b) for name,rows in role.items()}
            test = x[role['assessment']]
            completion = np.array([conditional_missing(v,observed,w,a,b) for v in test])
            h, hp, pv, _ = hidden_distribution(w,a,b)
            draw_rng = np.random.default_rng(seed+3000)
            hidden_draws = draw_rng.choice(len(h),size=16,p=hp)
            exact_samples = (draw_rng.random((16,64)) < pv[hidden_draws]).astype(int)
            # First assessment source only: retain changed-evidence and ignored-missing-value controls.
            input_case = test[0].copy()
            changed = input_case.copy(); changed[27] = 1-changed[27]  # observed column 3
            null = input_case.copy(); null[28] = 1-null[28]  # unobserved column 4
            base = completion[0]
            edited = conditional_missing(changed,observed,w,a,b)
            null_output = conditional_missing(null,observed,w,a,b)
            assert np.max(abs(null_output-base)) < 1e-12
            result = dict(method=method,seed=seed,parameters=584,history=history,metrics=evaluated,
                          weights=w.tolist(),visible_bias=a.tolist(),hidden_bias=b.tolist(),
                          assessment_completion=completion.tolist(),
                          completion_mse=float(np.mean((completion[:,~observed]-test[:,~observed])**2)),
                          completion_correct=int(np.sum((completion[:,~observed]>=.5)==test[:,~observed])),
                          exact_samples=exact_samples.tolist(),hidden_sample_states=h[hidden_draws].tolist(),
                          intervention=dict(source_id=int(ids[role['assessment'][0]]),observed_index=27,
                                            original=base.tolist(), edited=edited.tolist(),
                                            max_missing_change=float(np.max(abs(edited[~observed]-base[~observed]))),
                                            missing_placeholder_null_max=float(np.max(abs(null_output-base)))))
            results.append(result)
            print(method,seed, 'NLL',round(evaluated['assessment']['nll_nats_per_image'],6),
                  'reconstruction',round(evaluated['assessment']['mean_reconstruction_mse'],6),
                  'completion',round(result['completion_mse'],6),flush=True)
    output = dict(protocol=dict(threshold='pixel >= 8',raw_rows=len(records),unique_binary=len(x),
                               dropped_duplicates=dropped,seeds=SEEDS,epochs=EPOCHS,batch=BATCH,rate=RATE,
                               hidden_units=8,split_seed=91,roles={k:ids[v].tolist() for k,v in role.items()},
                               csv_sha256=hashlib.sha256((ROOT/'digits-400.csv').read_bytes()).hexdigest()),
                  baseline=dict(nll_nats_per_image=baseline,pixel_probability=mean.tolist(),
                                completion_mse=float(np.mean((mean[~observed]-x[role['assessment']][:,~observed])**2)),
                                completion_correct=int(np.sum((mean[~observed]>=.5)==x[role['assessment']][:,~observed]))),
                  exact=exact_checks(),fits=results)
    output['investigations'] = investigation_checks(output)
    (ROOT/'calculated-inputs.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
~~~

The program writes its computed evidence beside itself. To reproduce the report, retain that output and the data provenance. For a new experiment, choose the split, hyperparameter search and final assessment procedure before comparing results; changing code after seeing this assessment set does not restore its independence.

## 7. Reconstructing, sampling and completing are different tasks

### A convincing reconstruction can hide a poor distribution

Our deterministic reconstruction computes $\hat v=\sigma(a+W\sigma(b+W^Tv))$, with vector orientation adjusted in code. It asks how closely a two-stage mean pass returns the input. This is useful to inspect, but its mean squared error is not NLL.

A constructed two-pixel counterexample makes the distinction sharp. Model A has both interactions 20, visible biases −10 and hidden bias −20. It puts almost half its probability on 00 and half on 11. Starting from 11, the deterministic reconstruction is almost exactly 11: per-pixel MSE is about $2.06\times10^{-9}$. Nevertheless NLL for 11 is about 0.693238.

Model B has no interactions and both visible on-probabilities 0.9. Its reconstruction of 11 is $[0.9,0.9]$, with the larger MSE 0.01. But it assigns probability 0.81 to 11, giving the **better** NLL 0.210721. Reconstruction measures the return journey near an input; likelihood measures how the model divides its full probability budget.

**Visual: two reconstruction arrows beside four probability bars.** The learner predicts which model wins on each measure before revealing them. Both panels use the actual model distributions. An attractive-looking arrow must not stand in for the missing normalizer.

Similarly, comparing free energies from different model versions requires their different normalizers. A constant downward energy shift could make every raw score look better without changing any probability. Under the same fixed model, a data-versus-development mean free-energy gap does equal the corresponding NLL gap because the one $\log Z$ cancels. That useful diagnostic does not license comparing raw free energy across epochs as if it were normalized likelihood.

### Generate without showing the model an input

Ordinary Gibbs generation runs a chain and must consider burn-in and mixing. Our eight-hidden-unit model has an additional option: enumerate the exact hidden marginal, draw one of its 256 states, and independently draw visible pixels conditional on it. Repeating that process produces independent model samples, apart from pseudorandom-number implementation details, without a burn-in chain.

The saved 16 samples for every fitted model use this exact-enumeration route. They are not reconstructions of selected assessment images. Show all 16 in fixed order, including odd-looking ones. Eight hidden bits support at most 256 hidden configurations, but that does **not** limit the generator to 256 visible images: each hidden configuration produces a full Bernoulli distribution over pixels.

### Complete missing pixels without treating missing as black

Suppose the left half of an image is observed and the right half is unknown. “Unknown” is not a measured zero. Let $O$ contain observed pixel indices and $M$ the missing ones. For each hidden state, its log weight given the observed values is

$$b^Th+\sum_{i\in O}v_i\big(a_i+(Wh)_i\big)
+\sum_{i\in M}\operatorname{softplus}\big(a_i+(Wh)_i\big).$$

Normalize those 256 weights to obtain $p(h\mid v_O)$. A missing pixel's conditional on-probability is then

$$p(v_i=1\mid v_O)=\sum_h p(h\mid v_O)\sigma\big(a_i+(Wh)_i\big).$$

The first sum fixes observed states; the softplus sum marginalizes unknown ones. It is the same reasoning that produced $Z$, now with some variables held fixed. This exact calculation is practical because the hidden layer is small. With a larger hidden layer, a clamped Gibbs chain is an approximate alternative: after each visible draw restore the observed pixels, while allowing missing pixels to change. Do not continually clamp the missing pixels to their initialization.

For the three-switch example, observing $v_1=1$ gives $p(v_2=1\mid v_1=1)=0.5/(0.2+0.5)=5/7$. Filling the missing value with zero and treating it as observed would give a different hidden posterior. A placeholder is a display decision, not evidence.

We applied the fixed left-half mask to every assessment image. There are $80\times32=2,560$ missing pixel targets. Exact-gradient seed 11 achieved conditional-probability MSE 0.117704 versus the independent baseline's 0.132506; thresholding at 0.5 gives 2,112 versus 2,028 correct missing pixels. These are pixel-level results, not digit-classification accuracy. Other seeds and methods are retained, including PCD-1 seed 47's MSE 0.116945. A model can do well at this conditional task without winning on full-image NLL.

**Lab: inspect an actual completion.** Display the observed binary pixels, a distinct missing-data mask, the conditional-probability image and the withheld true pixels in separate roles. On assessment source 186, toggle observed pixel 27 and predict whether the right-half probabilities will change. For exact-gradient seed 11 the largest missing-pixel change is approximately 0.0596011. Toggling only the placeholder at missing pixel 28 changes no conditional probability. This null case checks that the model is using the mask correctly.

The conditional mean can look blurry because several plausible completions disagree. It is not necessarily a plausible joint sample. To see whole alternatives, sample a hidden state from the observed-data posterior and then all missing pixels conditional on it, preserving the observed pixels exactly. Uncertainty is part of the answer.

## 8. Broader uses and deeper boundaries

### Features, labels and ratings

The vector $p(h=1\mid v)$ provides learned features that can feed a classifier. To evaluate this use, fit the RBM and any preprocessing only on permitted training inputs, then fit the classifier using training labels and assess on held-out data. That is a separate supervised experiment; our unsupervised NLL table supplies no classification result.

A different construction includes a categorical label as an additional visible variable. Evaluate the joint free energy for each possible label and normalize over those alternatives to obtain $p(y\mid v)$. The one global $Z$ cancels. In contrast, separate class-specific RBMs have separate $Z_c$ values: simply choosing the lowest raw free energy can be wrong. You must account for normalization and class priors or use a properly fitted discriminative calibration.

Movie ratings provide a historically important application beyond images. The [2007 collaborative-filtering paper](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf) used categorical rating units and shared parameters across user-specific models containing their observed movies. Its treatment of absent ratings is a family of models with tied parameters, not equivalent to inserting zeros or exactly marginalizing every unrated movie in a single fixed model. The useful lesson is to specify both the rating distribution and what “missing” means before choosing the learning rule. This is historical methodology, not a claim that an RBM is today's best recommender.

### Binary is a modeling choice

Bernoulli variables describe binary events. One-hot categorical variables need probabilities that sum to one within each category group. Continuous measurements may use Gaussian visible variables; count data need a suitable count distribution. Changing the support requires changing the energy and conditional distributions consistently. Dividing a real value into $[0,1]$ does not by itself turn its likelihood into a Bernoulli probability for that real value.

For a simple fixed-unit-variance Gaussian-visible, binary-hidden model,

$$E(v,h)=\tfrac12\|v-a\|^2-b^Th-v^TWh,$$

completing the square gives $v\mid h\sim\mathcal N(a+Wh,I)$, while $p(h_j=1\mid v)=\sigma(b_j+v^TW_{:,j})$. The quadratic term prevents arbitrarily large visible values from obtaining unbounded preference for a fixed hidden state. If both layers are continuous, interaction strength and the full quadratic form must support a finite normalizer. An ordinary neural activation substituted into a sampler is not enough to define a valid probability model.

### Stacking does not erase the model definition

An RBM has one bipartite undirected layer pair. A **deep belief network** uses an undirected top pair with directed conditional layers below it; the original [2006 DBN paper](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf) develops layerwise initialization and subsequent learning. A **deep Boltzmann machine** keeps its multilayer interactions undirected and has harder hidden inference, as described in [Salakhutdinov and Hinton, 2009](https://proceedings.mlr.press/v5/salakhutdinov09a.html).

RBMs can initialize other networks, but a stack of independently trained RBMs is not automatically the product of their joint distributions with correct normalization. A deterministic encoder fine-tuned for reconstruction becomes an autoencoder with its own objective. A classifier fine-tuned for labels is evaluated on conditional prediction. Draw the final generative graph and write its probability factorization before transferring a statement about one model to another.

These models helped develop representation-learning ideas. Their history is useful without declaring that unsupervised pretraining always helps, or that energy-based modeling has disappeared. Modern architectures, data regimes and objectives require their own evidence.

### Beyond exact enumeration: AIS and other objectives

For a large RBM, **annealed importance sampling**, AIS, can estimate a partition-function ratio by traversing intermediate distributions between a tractable base and the target. Each stage contributes a ratio of unnormalized weights, and its transition must preserve the intended intermediate distribution. The [2008 quantitative analysis paper](https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf) applies this approach to RBMs and DBNs.

Under the required support, initialization and transition assumptions, the average importance weight estimates the normalizer ratio without bias. Taking its logarithm is a nonlinear operation; a log estimate is generally biased and a particular run is not a certified lower or upper bound. Report repeats, weight variability, schedules and assumptions. Our small example avoids this estimation issue by summing exactly; it should not teach an AIS estimate as an exact NLL curve.

Other training criteria answer different questions. Pseudo-likelihood uses conditionals such as $p(v_i\mid v_{-i})$ and can avoid the global normalizer. Current [scikit-learn BernoulliRBM documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html) identifies its training as PCD/SML and its `score_samples` result as a random-bit pseudo-likelihood estimate, not exact log likelihood. Check that metric contract before plotting a library “score” beside our NLL.

Likewise an autoencoder reconstruction loss, a variational lower bound and a normalized likelihood are distinct quantities. A comparison becomes meaningful only after specifying the data representation, objective, evaluation measure and computational budget.

## 9. Diagnose failures by the quantity that failed

| Observation | What it may mean | Next useful check |
| --- | --- | --- |
| Reconstruction improves; likelihood does not | Local return paths improved without the desired probability allocation | Calculate exact NLL if feasible; otherwise use a justified normalizer estimate and inspect samples |
| A persistent chain stays near one image | Mixing may be poor; correlated particles can give misleading negative statistics | Compare several initial states, autocorrelation and region occupancy; inspect update size |
| Raw free energies fall each epoch | Parameters or offsets changed, not necessarily normalized probability | Include the same version's $\log Z$ |
| Hidden probabilities are nearly all zero or one | Bias/weight scale, saturation or data mismatch may be limiting useful features | Inspect shared-scale weights, per-unit activation distributions and actual gradient increments |
| A grayscale input is silently accepted by a binary sampler | API numeric acceptance may not match the stated likelihood | Declare deterministic or stochastic binarization, or choose another visible distribution |
| Filling a missing placeholder changes the answer | Missing values were accidentally treated as evidence | Verify mask handling against exact tiny conditional calculations |
| Training works only after using assessment outcomes to tune it | Reported assessment is no longer independent | Use a fresh final evaluation protocol; keep all attempted outcomes disclosed |

Large weights can make sampling difficult without necessarily producing NaNs. Numerically stable softplus avoids one arithmetic failure; it does not cure poor mixing or an inappropriate learning rate. Momentum, decay and sparsity penalties can be investigated, but each changes the update and requires its own declared choice and validation. Add one justified change at a time so you can tell what caused the result.

## 10. Practice: make the probability bookkeeping explicit

### 1. Change one preference

In the three-switch model, increase only $a_1$ from zero to $\log2$. What are the four visible probabilities? Does every state become twice as likely?

<details><summary>Hint</summary>

Only configurations with $v_1=1$ acquire the extra factor two. Recompute the normalizer after applying it.

</details>
<details><summary>Worked solution</summary>

The unnormalized visible weights in order $00,01,10,11$ become $2,4,8,20$. Their sum is 34, so probabilities are $1/17,2/17,4/17,10/17$. Ratios within the $v_1=1$ group stay the same, but all normalized probabilities must reflect the new total. A global factor would cancel; this was a state-dependent factor.

</details>

### 2. Change the training observation

Use the original three-switch parameters but observe 10 instead of 11. Calculate both weight gradients and all three bias gradients. State which interactions increase.

<details><summary>Hint</summary>

The hidden on-probability is $3/4$. The model expectations are unchanged until you update the model.

</details>
<details><summary>Worked solution</summary>

Positive weight statistics are $[0.75,0]$ and negative statistics remain $[0.6,0.6]$, giving gradients $[0.15,-0.6]$. Visible-bias gradients are $[1,0]-[0.7,0.7]=[0.3,-0.7]$. The hidden-bias gradient is $0.75-0.8=-0.05$. Only the first interaction increases under positive-step gradient ascent. “Positive phase” does not mean that every parameter moves upward.

</details>

### 3. Exact sampling without $Z$ in the transition

Starting from visible 10, use uniform draw 0.8 for the hidden switch, then draws 0.3 and 0.6 for the two visible switches. What is the next visible state? Is that one sample evidence that the stationary probability of that state is one?

<details><summary>Hint</summary>

Compare each draw with its current conditional probability. A sampled hidden zero changes the visible conditionals.

</details>
<details><summary>Worked solution</summary>

The hidden on-probability is 0.75, so draw 0.8 gives hidden zero. Both visible on-probabilities are then 0.5. Draws 0.3 and 0.6 produce 10. One transition supplies one random outcome; it neither estimates an entire distribution accurately nor establishes mixing. Repeating a state can occur in a perfectly valid chain.

</details>

### 4. A different missing observation

Now observe $v_1=0$ in the three-switch model and leave $v_2$ missing. Find $p(v_2=1\mid v_1=0)$. Compare it with the case $v_1=1$ and with the unconditional marginal.

<details><summary>Hint</summary>

Restrict the visible probability table to the states compatible with the observation, then normalize that smaller table.

</details>
<details><summary>Worked solution</summary>

The compatible states 00 and 01 have probabilities 0.1 and 0.2, so the answer is $0.2/0.3=2/3$. Observing one gives $5/7$; the unconditional answer is 0.7. The hidden variable induces dependence, so evidence changes the conditional answer. An unknown value is neither of the two observed cases.

</details>

### 5. Choose the correct metric

A report claims model A is the better density model because its reconstruction MSE is smaller. For one assessment image, model A assigns probability 0.2 and model B assigns probability 0.3. Which has better NLL on this image? What additional evidence would you request for a useful generator?

<details><summary>Hint</summary>

Take the negative logarithm. Then separate scoring known observations from sampling plausible and diverse new ones.

</details>
<details><summary>Worked solution</summary>

The per-image NLLs are $-\log0.2\approx1.6094$ and $-\log0.3\approx1.2040$, so B is better on that measure. This comparison addresses that one image; assess the full held-out set before making a general performance claim. Request the normalization method, data split and representation, complete sampling protocol, unselected samples and relevant diversity/coverage evaluation. MSE alone settles none of those questions.

</details>

### 6. A resource decision

You have 20 visible binary units and 40 hidden ones. Which layer should you enumerate for an exact normalizer? How does the answer change for 100 visible units and 12 hidden ones?

<details><summary>Hint</summary>

Either layer can be summed out analytically when the other is fixed. Enumerate the smaller binary state space.

</details>
<details><summary>Worked solution</summary>

Enumerate the $2^{20}=1,048,576$ visible configurations in the first case, computing their free energies, rather than $2^{40}$ hidden configurations. Enumerate $2^{12}=4,096$ hidden configurations in the second case. Memory can be bounded by summing in chunks with a stable log accumulator, but the total enumeration work remains exponential in the enumerated layer size.

</details>

### 7. An implementation diagnosis

A PCD trainer resets its particles to the current data minibatch at every iteration, replaces sampled visible states by their means, and reports `score_samples` as exact likelihood. Identify three separate issues and specify a repair for each.

<details><summary>Hint</summary>

Ask where the negative chain starts, what state space its transitions occupy, and what quantity the scoring API returns.

</details>
<details><summary>Worked solution</summary>

Resetting to data removes persistence; retain chain state across updates or rename and implement an intended CD method. Mean-visible replacements change the Bernoulli Gibbs transition; draw binary states for the stated sampler, or explicitly name and evaluate the heuristic. Scikit-learn's RBM score is a pseudo-likelihood estimate; report that name and use an exact normalizer or documented estimator for normalized likelihood. These are independent mistakes, so repairing only one does not fix the experiment.

</details>

### 8. Extend the real study without leaking information

Design a comparison of 8 versus 12 hidden units for missing-pixel completion. State what stays fixed, what is selected using development data, and what must remain unavailable during fitting and selection.

<details><summary>Hint</summary>

The hidden-state sums are still manageable, but the larger model changes capacity and cost. The withheld pixel values are targets for evaluation, not completion inputs.

</details>
<details><summary>Worked solution</summary>

Keep the binary preprocessing, duplicate handling, source-group split, mask definition and evaluation metrics fixed. Predeclare learning-rate/epoch candidates and seeds for each size, recording their different parameter and enumeration costs. Fit on the fitting role; select configurations from development NLL or a declared development completion metric. Freeze the choice before evaluating a fresh assessment role. Keep assessment labels and missing pixel values out of model fitting, conditioning and selection. Since this lesson already exposes its assessment outcomes, a stronger new empirical claim needs a new untouched evaluation protocol. Report all declared seeds and include the independent-pixel baseline.

</details>

## 11. References and another way to learn

Use [Hinton's creator-hosted 2012 lecture collection](https://www.cs.toronto.edu/~hinton/coursera_lectures.html) for a spoken route: 11e introduces probability modeling, 12a–12d develop learning and RBMs, 12e covers ratings, and 14a–14e connect features, fine-tuning and real-valued data. The page links the individual recordings; the historical methods should be read with their stated context. Lecture titles and links were verified, without claiming a full viewing of every recording.

For a practical written route, read the [RBM guide](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) after doing the three-switch arithmetic. Its sections on statistics, monitoring, initialization and visible-unit choices help distinguish training decisions. For the reason finite-step CD needs care, follow the [convergence paper](https://proceedings.mlr.press/v9/sutskever10a.html); for persistence, follow [Tieleman's original method](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Tieleman.2008.pdf). The [AIS analysis](https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf) is the next step when exact enumeration becomes too large.

The next topic in this module is [Spectral Normalization & Gradient Penalty](/learn/path/full-curriculum/spectral-normalization-gradient-penalty?module=deep-learning-fundamentals). It asks how to control the sensitivity of learned functions. It is a new optimization/regularization question, not a continuation of the Gibbs sampler. Later [Modern Hopfield Networks](/learn/path/full-curriculum/modern-hopfield-networks?module=deep-learning-fundamentals) returns to energy and memory with a different state-update mechanism.
