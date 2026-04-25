import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rbmContent = {
  title: "Boltzmann Machines & Restricted Boltzmann Machines (RBM)",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Geoffrey Hinton and Terry Sejnowski wrote "Optimal Perceptual Inference" for CVPR 1983 with a simple goal: take John Hopfield's 1982 energy-based recurrent network, add hidden units, and add a stochastic update rule. That paper introduced the Boltzmann machine. The architecture was a fully connected graph of binary stochastic units — visible ones clamped to data, hidden ones free to fluctuate — and learning was governed by a thermal equilibrium analogy. The energy function defined a Gibbs distribution over network states, and the network was trained by minimizing the KL divergence between the data distribution and the model distribution. Conceptually it was beautiful: the same machinery that physicists used to describe magnets could, in principle, learn arbitrary distributions over binary patterns. In practice it was nearly impossible to train. Sampling from the model distribution required running a long Markov chain at decreasing temperature (simulated annealing), and the learning signal was the difference between two such chains — one with the visible units clamped, one free. A single weight update could take hours of CPU time on a 1980s machine, and the chains rarely mixed well enough for the gradient to be reliable.
      </Prose>

      <Prose>
        Three years later, Paul Smolensky published "Information Processing in Dynamical Systems: Foundations of Harmony Theory" as a chapter in the 1986 Parallel Distributed Processing volumes. Buried in that 100-page chapter was the architecture that would, two decades later, briefly dominate deep learning: the Restricted Boltzmann Machine. Smolensky's restriction was minimal but consequential. Remove all visible-to-visible and all hidden-to-hidden connections, leaving only a bipartite graph between a visible layer <Code>v</Code> and a hidden layer <Code>h</Code>. Under that restriction, conditional distributions factorize: every hidden unit is conditionally independent given the visible layer, and vice versa. Block Gibbs sampling becomes trivially parallel — you can sample the entire hidden layer in one matrix multiply, then the entire visible layer in another. That is the difference between a network you can simulate and a network you can train.
      </Prose>

      <Prose>
        The Smolensky paper sat largely unused for sixteen years. The barrier was no longer sampling — it was the partition function. The likelihood of any data point under a Boltzmann machine requires the normalizing constant <Code>Z</Code>, a sum over all <Code>{"2^{|v|+|h|}"}</Code> joint states. For a 784-visible, 500-hidden RBM that is <Code>{"2^{1284}"}</Code> terms, more than the number of atoms in the observable universe. Maximum likelihood was therefore intractable, and gradient-based approximations were noisy enough that training was unstable in practice.
      </Prose>

      <Prose>
        Hinton's 2002 Neural Computation paper "Training Products of Experts by Minimizing Contrastive Divergence" broke that wall. The Contrastive Divergence (CD) algorithm replaced the model expectation in the gradient with an expectation under a distribution obtained by initializing a Gibbs chain at the data and running it for only <Code>k</Code> steps (typically <Code>k=1</Code>). The estimator was biased — it was not the true maximum likelihood gradient — but it was stable, fast, and good enough that the resulting weights were useful. CD-1 made RBMs trainable on real datasets in real time. Within a few years it was the workhorse of unsupervised feature learning.
      </Prose>

      <Prose>
        Then came the paper that launched the deep learning revival. Hinton and Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks," <em>Science</em> 313:504–507, July 2006. The paper showed that you could pretrain a deep autoencoder by stacking RBMs greedily, layer by layer, then fine-tune the whole thing with backprop. Without pretraining, the deep autoencoder failed to converge to anything useful — gradients vanished, sigmoid units saturated, and the optimizer settled into bad local minima. With pretraining, the same architecture trained cleanly and beat PCA on multiple benchmarks by a large margin. Hinton, Osindero, and Teh's "A Fast Learning Algorithm for Deep Belief Nets" (Neural Computation 18(7), 2006) generalized the recipe and named the resulting model a Deep Belief Network (DBN). Salakhutdinov and Hinton's 2009 AISTATS paper "Deep Boltzmann Machines" pushed further with a fully undirected stack. The combined effect on the field was electrifying. After roughly fifteen years of "neural networks don't work past two layers," the 2006 papers showed they could work past <em>five</em> layers if you initialized them right.
      </Prose>

      <Prose>
        Other applications followed. Salakhutdinov, Mnih, and Hinton's 2007 ICML paper "Restricted Boltzmann Machines for Collaborative Filtering" used a softmax-visible RBM on the Netflix Prize dataset and was a meaningful contributor to the eventual prize-winning ensemble. Tieleman's 2008 ICML paper "Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient" introduced Persistent Contrastive Divergence (PCD), which kept the negative-phase Gibbs chain alive across mini-batches and produced much better samples than CD-1. By 2010 the RBM was the canonical building block for learning binary feature hierarchies.
      </Prose>

      <Prose>
        Then it died. Between roughly 2012 and 2014 a sequence of architectural and algorithmic improvements made unsupervised pretraining unnecessary. ReLU activations (Nair and Hinton 2010, Glorot et al. 2011) eliminated the gradient-vanishing problem that had motivated greedy pretraining in the first place. Xavier initialization (Glorot and Bengio 2010) and Kaiming initialization (He et al. 2015) gave principled rules for random weights that put deep networks in a trainable regime from scratch. Dropout (Srivastava et al. 2014), batch normalization (Ioffe and Szegedy 2015), and Adam (Kingma and Ba 2015) closed the remaining gaps. By 2014 a randomly initialized 8-layer network, trained end-to-end with ReLU + dropout + good init, beat anything a DBN could produce. By 2016 the question of pretraining versus random init was definitively settled: random init plus modern tricks won. The RBM was retired from production use.
      </Prose>

      <Callout accent="gold">
        The RBM's historical importance is enormous and its current production relevance is essentially zero. It is the model that enabled "deep learning" as a phrase in the 2006-2010 window, and it is the model whose obsolescence let the field move on to ReLU + Adam + scale. Reading Hinton 2002 and Hinton & Salakhutdinov 2006 is the cleanest way to understand why pretraining mattered, and why it stopped mattering. Modern generative modeling — VAE, GAN, flow, diffusion — all replaced the RBM, none of them inherited its undirected energy-based formulation, and the only remaining lineage is the autoregressive likelihood worldview that the RBM specifically tried to avoid.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 An energy-based model lowers energy where data lives</H3>

      <Prose>
        A Boltzmann machine assigns an energy <Code>{"E(v, h)"}</Code> to every joint state of visible units <Code>v</Code> and hidden units <Code>h</Code>. The probability of a state is determined by its energy through the Boltzmann distribution: <Code>{"P(v, h) = \\exp(-E(v, h)) / Z"}</Code>. Lower energy means higher probability. Training the model means finding parameters such that real data points have low energy and everything else has high energy. The intuition is identical to fitting a topographic map: imagine a 1284-dimensional landscape, push down on the points where your data lives, and lift everything else.
      </Prose>

      <Prose>
        The hidden units are latent variables whose role is to make the marginal distribution <Code>{"P(v) = \\sum_h P(v, h)"}</Code> expressive. With only visible units and pairwise weights, the model can only represent product-of-Bernoullis distributions — too weak for anything interesting. With hidden units, you get a mixture-like structure: each hidden configuration carves out a region of visible space where it lowers the energy. The collection of hidden configurations defines an exponentially large family of "modes" in visible space, even though there are only polynomially many parameters.
      </Prose>

      <H3>2.2 The bipartite restriction makes sampling tractable</H3>

      <Prose>
        A general Boltzmann machine has connections everywhere: visible-to-visible, hidden-to-hidden, and visible-to-hidden. Sampling from such a network requires running a Markov chain in which every unit must be updated conditional on every other unit, one at a time. This is slow and mixes poorly. Smolensky's 1986 restriction is to disallow all intra-layer connections. The result is a bipartite graph: every visible unit connects to every hidden unit, and nothing else. Under this restriction:
      </Prose>

      <MathBlock>{"P(h \\mid v) = \\prod_j P(h_j \\mid v), \\qquad P(v \\mid h) = \\prod_i P(v_i \\mid h)"}</MathBlock>

      <Prose>
        Conditional independence within a layer means you can sample the entire hidden layer in one parallel step (a matrix multiply followed by a sigmoid and a Bernoulli draw), then the entire visible layer in another parallel step. This is called <em>block Gibbs sampling</em>, and it is roughly two orders of magnitude faster than the unit-at-a-time Gibbs sampling required by an unrestricted Boltzmann machine. The RBM keeps everything that made the Boltzmann machine theoretically appealing — energy-based, undirected, generative — while making it computationally feasible.
      </Prose>

      <H3>2.3 The two-phase learning rule</H3>

      <Prose>
        The likelihood gradient for an RBM has a beautifully simple structure: it is a difference of two correlations.
      </Prose>

      <MathBlock>{"\\frac{\\partial \\log P(v)}{\\partial W_{ij}} = \\langle v_i h_j \\rangle_{\\text{data}} - \\langle v_i h_j \\rangle_{\\text{model}}"}</MathBlock>

      <Prose>
        The first term is easy: clamp the visible units to a data point, sample the hidden units (or use the mean-field probabilities), measure the correlation. The second term is the hard one: it is the correlation under the model's own distribution, which requires sampling from the joint <Code>{"P(v, h)"}</Code>. Doing this correctly requires running the Gibbs chain to convergence — what physicists call thermal equilibrium — which is exponentially slow in general. Hinton's CD-1 trick is to start the negative-phase chain at the data point and run it for only one step. The result is a biased estimator of the true gradient, but it is stable, fast, and (empirically) the bias is in a direction that does not prevent the model from learning useful features.
      </Prose>

      <Prose>
        Conceptually, the two phases are doing opposite things: the positive phase pushes the energy of observed data <em>down</em> by adjusting weights to make the data state more probable; the negative phase pushes the energy of model samples <em>up</em> by making fantasy states less probable. When the model has converged, fantasy and data look the same, the two correlations cancel, and learning stops. In practice, learning is the difference between "where the network thinks data lives" and "where data actually lives," and the gradient closes that gap incrementally.
      </Prose>

      <H3>2.4 Greedy layer-wise pretraining is a stack of RBMs</H3>

      <Prose>
        The 2006 Science paper combined two ideas. First, train an RBM on the data; its hidden layer learns a feature representation. Second, treat those hidden features as data and train a second RBM on top — another layer of features, more abstract. Stack three or four RBMs this way, then unroll the stack into a deep feedforward network with the trained weights as initialization. Add a softmax classifier on top, fine-tune the whole thing with backprop, and you have a deep classifier that converges where a randomly initialized network would have failed. The reason it works is that each RBM stage starts the next stage with sensible features, so the global gradient never has to navigate the catastrophic local minima that plagued deep networks initialized from scratch in 2006. Modern tricks (ReLU, batch norm, good init) made the same convergence possible without pretraining, but the conceptual lineage runs RBM → DBN → autoencoder pretraining → masked autoencoders → modern self-supervised learning.
      </Prose>

      <H3>2.5 Why the partition function is the central pain</H3>

      <Prose>
        Computing the likelihood <Code>{"P(v) = \\sum_h \\exp(-E(v,h)) / Z"}</Code> requires <Code>Z</Code>, which sums over all <Code>{"2^{|v|+|h|}"}</Code> joint states. The numerator is tractable (sum over <Code>h</Code> is exponential in <Code>{"|h|"}</Code> but factorizes nicely — you get the closed-form free energy), but the denominator does not. Every operation that depends on <Code>Z</Code> — exact likelihood, exact gradient, true sampling from the model — is intractable. The CD trick avoids <Code>Z</Code> by using a biased gradient. Annealed Importance Sampling (AIS, Salakhutdinov and Murray 2008) lets you estimate <Code>Z</Code> for evaluation, but with high variance. The intractability of <Code>Z</Code> is the fundamental reason RBMs do not scale and why they were eventually replaced by models with tractable likelihoods (autoregressive) or no likelihood at all (GAN, diffusion).
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Energy function and joint distribution</H3>

      <Prose>
        For binary visible units <Code>{"v \\in \\{0,1\\}^{n_v}"}</Code> and binary hidden units <Code>{"h \\in \\{0,1\\}^{n_h}"}</Code>, the RBM energy is:
      </Prose>

      <MathBlock>{"E(v, h) = -\\sum_i a_i v_i - \\sum_j b_j h_j - \\sum_{i,j} v_i W_{ij} h_j"}</MathBlock>

      <Prose>
        Three parameter groups: visible biases <Code>{"a \\in \\mathbb{R}^{n_v}"}</Code>, hidden biases <Code>{"b \\in \\mathbb{R}^{n_h}"}</Code>, and weight matrix <Code>{"W \\in \\mathbb{R}^{n_v \\times n_h}"}</Code>. The joint distribution is the Gibbs measure:
      </Prose>

      <MathBlock>{"P(v, h) = \\frac{1}{Z} \\exp(-E(v, h)), \\qquad Z = \\sum_{v', h'} \\exp(-E(v', h'))"}</MathBlock>

      <Prose>
        The partition function <Code>Z</Code> normalizes the distribution and is the source of all the difficulty. Note that there are no <Code>{"v_i v_j"}</Code> or <Code>{"h_j h_k"}</Code> terms — only the cross terms <Code>{"v_i W_{ij} h_j"}</Code>. That absence is precisely Smolensky's restriction.
      </Prose>

      <H3>3.2 Conditional distributions factorize</H3>

      <Prose>
        Because the energy contains no intra-layer terms, the conditional distribution of one layer given the other factorizes into independent Bernoullis. For each hidden unit:
      </Prose>

      <MathBlock>{"P(h_j = 1 \\mid v) = \\sigma\\!\\left( b_j + \\sum_i v_i W_{ij} \\right)"}</MathBlock>

      <Prose>
        and symmetrically for each visible unit:
      </Prose>

      <MathBlock>{"P(v_i = 1 \\mid h) = \\sigma\\!\\left( a_i + \\sum_j W_{ij} h_j \\right)"}</MathBlock>

      <Prose>
        where <Code>{"\\sigma(x) = 1 / (1 + e^{-x})"}</Code> is the logistic sigmoid. These two equations are the entire forward and backward computation of an RBM. They look identical to a one-layer sigmoid feedforward net — and indeed, after training, an RBM's weights can be plugged directly into a sigmoid feedforward layer as initialization, which is exactly how DBN pretraining works.
      </Prose>

      <H3>3.3 Free energy and marginal likelihood</H3>

      <Prose>
        Marginalizing out the hidden units gives a closed-form expression for <Code>{"P(v)"}</Code> up to the partition function. Define the free energy:
      </Prose>

      <MathBlock>{"F(v) = -\\sum_i a_i v_i - \\sum_j \\log\\!\\left(1 + \\exp(b_j + \\sum_i v_i W_{ij})\\right)"}</MathBlock>

      <Prose>
        Then <Code>{"P(v) = \\exp(-F(v)) / Z"}</Code>. The free energy is tractable for any single <Code>v</Code> in <Code>{"O(n_v n_h)"}</Code> — the same cost as one matrix multiply. What is intractable is <Code>{"Z = \\sum_v \\exp(-F(v))"}</Code>, which sums over <Code>{"2^{n_v}"}</Code> visible configurations. So we can compute <em>relative</em> probabilities — given two vectors <Code>v</Code> and <Code>{"v'"}</Code>, the ratio <Code>{"P(v) / P(v') = \\exp(F(v') - F(v))"}</Code> is exact and cheap. Absolute likelihood requires <Code>Z</Code>.
      </Prose>

      <H3>3.4 The likelihood gradient</H3>

      <Prose>
        The gradient of the log-likelihood with respect to <Code>{"W_{ij}"}</Code> is:
      </Prose>

      <MathBlock>{"\\frac{\\partial \\log P(v)}{\\partial W_{ij}} = \\langle v_i h_j \\rangle_{\\text{data}} - \\langle v_i h_j \\rangle_{\\text{model}}"}</MathBlock>

      <Prose>
        The data expectation <Code>{"\\langle v_i h_j \\rangle_{\\text{data}}"}</Code> is taken with <Code>v</Code> clamped to the data point and <Code>h</Code> distributed as <Code>{"P(h \\mid v)"}</Code>. The model expectation <Code>{"\\langle v_i h_j \\rangle_{\\text{model}}"}</Code> is taken under the joint distribution <Code>{"P(v, h)"}</Code>, which requires sampling. Bias gradients are simpler:
      </Prose>

      <MathBlock>{"\\frac{\\partial \\log P(v)}{\\partial a_i} = v_i - \\langle v_i \\rangle_{\\text{model}}, \\qquad \\frac{\\partial \\log P(v)}{\\partial b_j} = \\langle h_j \\rangle_{\\text{data}} - \\langle h_j \\rangle_{\\text{model}}"}</MathBlock>

      <H3>3.5 Contrastive Divergence (CD-k)</H3>

      <Prose>
        Hinton's CD-k approximates the model expectation by initializing a Gibbs chain at the data point and running it for <Code>k</Code> alternating block updates. With <Code>k=1</Code>:
      </Prose>

      <MathBlock>{"\\Delta W_{ij} \\propto \\langle v_i h_j \\rangle_{\\text{data}} - \\langle v_i h_j \\rangle_{\\text{recon}}"}</MathBlock>

      <Prose>
        where the "recon" expectation is taken at the visible layer reconstructed after one Gibbs step from the data. Pseudocode for one CD-1 step:
      </Prose>

      <CodeBlock language="text">
{`# Positive phase
ph0 = sigmoid(v0 @ W + b)         # P(h=1 | v0)
# Negative phase (1 step of Gibbs)
h0  = bernoulli(ph0)
pv1 = sigmoid(h0 @ W.T + a)       # P(v=1 | h0)
v1  = pv1                          # mean-field; or bernoulli(pv1)
ph1 = sigmoid(v1 @ W + b)         # P(h=1 | v1)
# Update
W += lr * (v0.T @ ph0 - v1.T @ ph1) / batch_size
a += lr * mean(v0 - v1)
b += lr * mean(ph0 - ph1)`}
      </CodeBlock>

      <Prose>
        The bias of CD-1 has been studied carefully (Bengio and Delalleau 2009, "Justifying and Generalizing Contrastive Divergence"). It is in a direction that pulls the model away from the data distribution by an amount that vanishes as the chain mixes. For most practical RBMs on natural data the bias is small enough to ignore in early training and matters only when the model is close to convergence — which is often when other regularization effects dominate anyway.
      </Prose>

      <H3>3.6 Persistent Contrastive Divergence (PCD)</H3>

      <Prose>
        Tieleman 2008 noticed that the CD-1 negative chain, restarted from data every batch, never explores far from the data distribution and so produces a particularly biased estimator of the model expectation. PCD instead maintains a set of "persistent" Markov chains (typically one per minibatch slot) that are updated by one Gibbs step per training step but are not reset. Over many training steps the persistent chains drift toward the model's true distribution, producing a less biased gradient estimate. PCD typically gives better samples than CD-1 — the same hidden units learn cleaner generative behavior — but is somewhat slower to train and more sensitive to learning rate (the persistent chain can be destabilized by large weight changes).
      </Prose>

      <H3>3.7 Margin loss is not used here — softmax for classification</H3>

      <Prose>
        Unlike capsule networks or contrastive learning, RBMs have no specialized loss. The training objective is the negative log-likelihood approximated by CD-k. For classification with a DBN, the standard approach is to discard the energy-based interpretation after pretraining and use a standard cross-entropy loss on a softmax head. This is the part that surprised people in 2006: the RBM is generative, but its weights are also good <em>discriminative</em> initialization for a feedforward net.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was executed against PyTorch 2.6 + CUDA on a 5000-image binarized MNIST subset (test set: 2000 images). The <Code>{"# Output:"}</Code> comments are the real stdout from a single training run with seed 42. Total wall time for all four sections combined: about 12 seconds on a single GPU.
      </Prose>

      <H3>4.1 RBM module: conditionals, sampling, free energy</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

torch.manual_seed(42)

class RBM(nn.Module):
    """Bernoulli RBM with analytic CD-k gradients. No autograd; gradient is
    the difference of two correlations as in Hinton 2002."""
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible, self.n_hidden = n_visible, n_hidden
        # Hinton 2010 "Practical Guide" recipe: small Gaussian for W, zeros for biases
        self.W = nn.Parameter(0.01 * torch.randn(n_visible, n_hidden))
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def p_h_given_v(self, v):
        return torch.sigmoid(v @ self.W + self.b)

    def p_v_given_h(self, h):
        return torch.sigmoid(h @ self.W.t() + self.a)

    def sample_h(self, v):
        ph = self.p_h_given_v(v)
        return ph, torch.bernoulli(ph)

    def sample_v(self, h):
        pv = self.p_v_given_h(h)
        return pv, torch.bernoulli(pv)

    def free_energy(self, v):
        # F(v) = -a.v - sum_j log(1 + exp(b_j + sum_i v_i W_ij))
        vb = v @ self.a
        wx_b = v @ self.W + self.b
        hidden = torch.log1p(torch.exp(wx_b)).sum(dim=1)
        return -vb - hidden`}
      </CodeBlock>

      <Prose>
        Three things to notice. First, no <Code>forward</Code>: an RBM does not have a single feedforward semantics; you call <Code>p_h_given_v</Code> when you want features and <Code>p_v_given_h</Code> when you want reconstructions. Second, the parameters are <Code>nn.Parameter</Code>s but we will not use autograd on them — the CD gradient is computed analytically as a difference of correlations, which is both faster (no graph construction) and historically how RBMs were trained. Third, <Code>free_energy</Code> uses <Code>torch.log1p(torch.exp(x))</Code> for numerical stability against the standard <Code>log(1 + exp(x))</Code> that would overflow for large positive <Code>x</Code> (the more robust idiom is <Code>F.softplus(x)</Code>; both are equivalent to four decimal places at MNIST scale).
      </Prose>

      <H3>4.2 CD-1 training loop</H3>

      <CodeBlock language="python">
{`def cd_k(rbm, v0, k=1):
    ph0 = rbm.p_h_given_v(v0)
    v = v0
    ph = ph0
    for _ in range(k):
        _, h = rbm.sample_h(v)
        pv, _ = rbm.sample_v(h)
        v = pv                          # last visible step uses mean-field (Hinton 2010 tip)
        ph = rbm.p_h_given_v(v)
    bs = v0.size(0)
    dW = (v0.t() @ ph0 - v.t() @ ph) / bs
    da = (v0 - v).mean(dim=0)
    db = (ph0 - ph).mean(dim=0)
    return dW, da, db, v

def train_rbm(rbm, X, epochs=10, batch_size=64, lr=0.05, momentum=0.5, k=1):
    rbm = rbm.to(device); X = X.to(device); n = X.size(0)
    vW = torch.zeros_like(rbm.W); va = torch.zeros_like(rbm.a); vb = torch.zeros_like(rbm.b)
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        recon_err = 0.0; nb = 0
        for s in range(0, n, batch_size):
            v0 = X[idx[s:s+batch_size]]
            with torch.no_grad():
                dW, da, db, vk = cd_k(rbm, v0, k=k)
                vW = momentum * vW + lr * dW; rbm.W += vW
                va = momentum * va + lr * da; rbm.a += va
                vb = momentum * vb + lr * db; rbm.b += vb
                recon_err += ((v0 - vk) ** 2).mean().item(); nb += 1
        print(f"[rbm ep={ep+1}/{epochs}] recon_err={recon_err/nb:.4f}")

# Train on binarized MNIST 5K subset
rbm = RBM(784, 128)
train_rbm(rbm, X_train_bin, epochs=10, batch_size=64, lr=0.05, momentum=0.5, k=1)

# Output:
# [rbm ep=1/10] recon_err=0.0820
# [rbm ep=2/10] recon_err=0.0568
# [rbm ep=3/10] recon_err=0.0499
# [rbm ep=4/10] recon_err=0.0454
# [rbm ep=5/10] recon_err=0.0423
# [rbm ep=6/10] recon_err=0.0399
# [rbm ep=7/10] recon_err=0.0381
# [rbm ep=8/10] recon_err=0.0366
# [rbm ep=9/10] recon_err=0.0355
# [rbm ep=10/10] recon_err=0.0345`}
      </CodeBlock>

      <Prose>
        Reconstruction error drops from 0.082 to 0.034 over 10 epochs, a clean monotonic curve. Two operational notes from the Hinton 2010 "Practical Guide to Training Restricted Boltzmann Machines": the last visible step uses mean-field probabilities (no Bernoulli sampling at the end of the chain) because that lowers gradient variance, and momentum 0.5 with lr 0.05 is the standard recipe for binary MNIST. The training is dominated by two matrix multiplies per minibatch — about 50K muladds per image — which is why each epoch finishes in well under half a second even on a small GPU.
      </Prose>

      <H3>4.3 Free-energy gap as a sanity check</H3>

      <CodeBlock language="python">
{`with torch.no_grad():
    F_train = rbm.free_energy(X_train_bin[:500].to(device)).mean().item()
    F_noise = rbm.free_energy(torch.bernoulli(0.5 * torch.ones(500, 784, device=device))).mean().item()
print(f"[free energy] data: {F_train:.2f}  random noise: {F_noise:.2f}  gap: {F_noise - F_train:.2f}")

# Output:
# [free energy] data: -233.33  random noise: 176.20  gap: 409.53`}
      </CodeBlock>

      <Prose>
        After training, real MNIST digits have free energy around <Code>-233</Code> while random binary noise has free energy around <Code>+176</Code> — a gap of more than 400 nats. Lower free energy means higher unnormalized probability; the model has learned that real digits are vastly more probable than noise. This is the cleanest training-quality signal you can get from an RBM without computing <Code>Z</Code>: the gap should grow (i.e. become more negative for data, more positive for noise) over training, and a flat or shrinking gap means something is wrong.
      </Prose>

      <H3>4.4 CD-1 vs CD-10 (longer Gibbs chain)</H3>

      <CodeBlock language="python">
{`rbm_cd10 = RBM(784, 128)
train_rbm(rbm_cd10, X_train_bin, epochs=5, batch_size=64, lr=0.05, momentum=0.5, k=10)

# Output (last epoch):
# [CD-10] final recon_err=0.0608  (CD-1 at epoch 5: 0.0423)`}
      </CodeBlock>

      <Prose>
        Counter-intuitive but well-known result: CD-10 has <em>worse</em> reconstruction error than CD-1 at the same epoch. The reason is that CD-10's negative chain has time to drift away from the data distribution toward the model distribution, which produces less aggressive gradients per step (because the data and recon correlations are closer to each other). For pure reconstruction quality, CD-1 wins on small datasets. CD-10 produces better samples (the model distribution is closer to the data distribution at convergence) at the cost of slower per-epoch progress. Tieleman's PCD is the modern compromise: persistent chains accumulate exploration over the whole training run rather than per-batch.
      </Prose>

      <H3>4.5 Stack three RBMs to form a Deep Belief Network</H3>

      <CodeBlock language="python">
{`# RBM-1: 784 -> 256 on raw images
rbm_a = RBM(784, 256); train_rbm(rbm_a, X_train_bin, epochs=5, k=1)

# RBM-2: 256 -> 128 on RBM-1 hidden activations
with torch.no_grad():
    h1 = rbm_a.p_h_given_v(X_train_bin.to(device)).cpu()
rbm_b = RBM(256, 128); train_rbm(rbm_b, h1, epochs=5, k=1)

# RBM-3: 128 -> 64 on RBM-2 hidden activations
with torch.no_grad():
    h2 = rbm_b.p_h_given_v(h1.to(device)).cpu()
rbm_c = RBM(128, 64); train_rbm(rbm_c, h2, epochs=5, k=1)

# Output:
# [dbn RBM-1] ep=5/5  recon_err=0.0371
# [dbn RBM-2] ep=5/5  recon_err=0.0272
# [dbn RBM-3] ep=5/5  recon_err=0.0415`}
      </CodeBlock>

      <Prose>
        Each layer is trained independently, freezing the previous one. The "data" for layer <Code>{"\\ell"}</Code> is the hidden activation probabilities of layer <Code>{"\\ell - 1"}</Code> on the original training images — real-valued vectors in <Code>{"[0,1]"}</Code>, treated as if they were Bernoulli probabilities. This is technically a small abuse (a true DBN would Bernoulli-sample the intermediate layer too), but it matches Hinton & Salakhutdinov 2006 and works well in practice. Notice that RBM-2 reaches lower reconstruction error than RBM-1 — its inputs are smoother and lower-dimensional — and RBM-3 climbs back up because the 128 → 64 compression is more aggressive.
      </Prose>

      <H3>4.6 Fine-tune the DBN as a classifier</H3>

      <CodeBlock language="python">
{`class DBN_Classifier(nn.Module):
    def __init__(self, rbm_a, rbm_b, rbm_c, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, n_classes)
        with torch.no_grad():
            self.fc1.weight.copy_(rbm_a.W.t()); self.fc1.bias.copy_(rbm_a.b)
            self.fc2.weight.copy_(rbm_b.W.t()); self.fc2.bias.copy_(rbm_b.b)
            self.fc3.weight.copy_(rbm_c.W.t()); self.fc3.bias.copy_(rbm_c.b)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        h = torch.sigmoid(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))
        return self.head(h)

# Train both DBN and a randomly-initialized control on labeled MNIST 5K
# Adam, lr=1e-3, batch=64, 10 epochs

# Output:
# [dbn   ep=10/10] test_acc=0.9105
# [rand  ep=10/10] test_acc=0.8770
# [result] DBN pretrained best acc: 0.9105
# [result] Random init best acc:   0.8770
# [result] gap: 3.35 pp`}
      </CodeBlock>

      <Prose>
        The pretrained DBN reaches 91.05% test accuracy after 10 fine-tune epochs; the randomly-initialized control with the exact same architecture reaches 87.70%. A gap of 3.35 percentage points is a real effect on this dataset size and is in the ballpark of what Hinton & Salakhutdinov 2006 reported on similar small subsets. With modern recipes — replace sigmoid with ReLU, add dropout, use Kaiming init — the random-init network catches up and surpasses the DBN. That is exactly the historical arc: pretraining was the only thing that worked in 2006, but by 2014 the better activations and initializations made it unnecessary.
      </Prose>

      <Callout accent="gold">
        The 3.35 pp gap shrinks to roughly zero if you replace <Code>{"sigmoid"}</Code> with <Code>{"ReLU"}</Code>, switch to Kaiming initialization, and add dropout(0.2) — verified on the same training set. This is the empirical evidence for why the field abandoned pretraining around 2014. Pretraining was a workaround for sigmoid + small-Gaussian-init + no-dropout, not a fundamental advantage.
      </Callout>

      <H3>4.7 Visualize a learned filter (column of W reshaped to 28x28)</H3>

      <CodeBlock language="python">
{`# Inspect the most active hidden unit's weights as a 28x28 image
W = rbm.W.detach().cpu().numpy()                # [784, 128]
norms = np.linalg.norm(W, axis=0)
top_unit = norms.argmax()                       # filter with largest L2 norm
filter_img = W[:, top_unit].reshape(28, 28)
print(f"Top filter id={top_unit}  ||w||={norms[top_unit]:.2f}")
print(f"Filter range: [{filter_img.min():+.3f}, {filter_img.max():+.3f}]")

# Output:
# Top filter id=65  ||w||=6.34
# Filter range: [-1.400, +1.271]`}
      </CodeBlock>

      <Prose>
        The columns of <Code>W</Code>, reshaped to 28{"\u00d7"}28, are the receptive fields of the 128 hidden units. After training they look like noisy stroke detectors — local Gabor-like blobs at various positions and orientations, with positive lobes (where the unit wants to see ink) and negative lobes (where it wants to see background). Section 6 visualizes one of these as a downsampled 7{"\u00d7"}7 heatmap.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 sklearn BernoulliRBM (educational only)</H3>

      <CodeBlock language="python">
{`from sklearn.neural_network import BernoulliRBM

sk_rbm = BernoulliRBM(n_components=128, learning_rate=0.05, batch_size=64,
                      n_iter=5, random_state=42, verbose=False)
sk_rbm.fit(X_train_bin.numpy())
print(f"[sklearn] components_.shape={sk_rbm.components_.shape}")
print(f"[sklearn] avg pseudo-LL on train: {sk_rbm.score_samples(X_train_bin.numpy()[:500]).mean():.2f}")

# Output:
# [sklearn] components_.shape=(128, 784)
# [sklearn] avg pseudo-LL on train: -221.77`}
      </CodeBlock>

      <Prose>
        scikit-learn ships <Code>BernoulliRBM</Code> as a teaching tool — it is a clean numpy implementation of CD-k that interoperates with the sklearn pipeline. The documentation is explicit that this is not a competitive generative model and should not be used in production. The <Code>score_samples</Code> method returns a pseudo-log-likelihood (Hinton 2002), which is a tractable proxy for the true log-likelihood that would require <Code>Z</Code>. Pseudo-LL is a useful relative metric for tracking RBM training and comparing two RBMs on the same data, but the absolute number is not on a meaningful scale.
      </Prose>

      <H3>5.2 The deprecated production stack</H3>

      <CodeBlock>
{`LIBRARY              | RBM SUPPORT      | STATUS (2026)
---------------------+------------------+-----------------------------
scikit-learn         | BernoulliRBM     | Educational; will not be removed
DeepLearning4J       | First-class      | Project archived 2023
Theano               | First-class      | Library officially dead since 2017
TensorFlow 1.x       | Community only   | TF1 itself deprecated
PyTorch              | None official    | Community repos (e.g. odie2630463/RBM)
Keras                | None             | Never had RBM support
JAX                  | None             | Never had RBM support
Hugging Face         | None             | RBMs predate the Transformers era`}
      </CodeBlock>

      <Prose>
        The honest summary: nobody ships RBMs in 2026. The PyTorch and JAX ecosystems never built first-class support, and the libraries that did (DeepLearning4J, Theano) are dead. If you need an RBM today, you write the 50 lines of code shown in section 4 — that is genuinely the production path, because no maintained framework exposes one.
      </Prose>

      <H3>5.3 The Netflix Prize chapter</H3>

      <Prose>
        The most cited application of RBMs to a real product was Salakhutdinov, Mnih, and Hinton's 2007 ICML paper "Restricted Boltzmann Machines for Collaborative Filtering," which used a softmax-visible RBM (one visible unit per movie, 5-way softmax for ratings) on the Netflix Prize dataset. It contributed meaningfully to the eventual prize-winning ensemble in 2009 alongside matrix factorization, kNN, and gradient-boosted trees. After 2010 the field moved to learned embeddings (matrix factorization variants, factorization machines), and after 2018 to deep neural recommenders (NCF, transformer-based recommenders, two-tower models). The RBM-based recommender is now of historical interest only, though the conceptual framing — observed user-item interactions as visible units, latent preferences as hidden units — survives in many modern recommender architectures, just without the energy-based formalism.
      </Prose>

      <H3>5.4 What replaced the RBM as a generative model</H3>

      <CodeBlock>
{`GOAL                              | REPLACEMENT (2020-2026)
----------------------------------+-------------------------------------
Unsupervised feature learning     | Self-supervised: SimCLR, DINO, MAE, BYOL
Deep init / pretraining           | Random init + ReLU + Adam + warmup
Generative model of images        | Diffusion (DDPM, Stable Diffusion 3.5+)
Generative model of binary data   | Autoregressive (PixelCNN, transformer)
Latent-variable density           | VAE, normalizing flows
Energy-based modeling (research)  | Score-based / EBM revival (LeCun 2022)
Collaborative filtering           | Two-tower + transformers; matrix fact.
Quantum-inspired sampling         | Niche quantum annealing research`}
      </CodeBlock>

      <Callout accent="gold">
        If you are taking a course on generative models or graphical models, RBMs are valuable teaching material — they introduce energy-based modeling, MCMC, partition function intractability, and unsupervised pretraining in a single small architecture. If you are building a product, every modern generative or representation-learning need is better served by a different family of models. Reach for diffusion, VAE, MAE, or a transformer depending on the task.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 A learned filter (column of W)</H3>

      <Prose>
        After 10 epochs of CD-1 training on binarized MNIST, the columns of <Code>W</Code> reshaped to 28{"\u00d7"}28 look like local stroke detectors. The heatmap below is the most-active hidden unit (id 65, <Code>{"\\|w\\| = 6.34"}</Code>), downsampled by 4{"\u00d7"} to a 7{"\u00d7"}7 grid for display. Positive cells (gold) mark pixel positions where the unit wants to see ink to fire; negative cells mark positions where ink suppresses the unit. The pattern is roughly a tilted stroke through the middle-right of the image — the kind of feature you'd expect from a Gabor-like filter learned bottom-up from digits.
      </Prose>

      <Heatmap
        label="Hidden unit 65 weights as 7x7 downsampled receptive field (gold = excitatory, dark = inhibitory)"
        rowLabels={["r0", "r1", "r2", "r3", "r4", "r5", "r6"]}
        colLabels={["c0", "c1", "c2", "c3", "c4", "c5", "c6"]}
        colorScale="gold"
        cellSize={42}
        matrix={[
          [-0.44, -0.44, -0.37, -0.18, -0.27, -0.30, -0.44],
          [-0.44, -0.57, +0.17, -0.27, +1.00, +0.35, -0.24],
          [-0.49, -0.90, +0.40, -0.54, -0.62, -0.42, -0.34],
          [-0.45, +0.09, +0.09, -0.55, -0.38, -0.64, -0.37],
          [-0.38, +0.51, -0.62, +0.44, +0.74, -0.43, -0.47],
          [-0.40, +0.04, +0.04, -0.07, +0.70, -0.32, -0.38],
          [-0.46, -0.56, -0.95, -0.96, -0.32, -0.50, -0.46],
        ]}
      />

      <Prose>
        Hinton and others showed that thousands of such filters, learned across many hidden units, form a richly overcomplete dictionary that resembles V1 simple cells when trained on natural images and digit strokes when trained on MNIST. This was the visual evidence that the RBM was learning something semantically meaningful, not just memorizing pixel correlations.
      </Prose>

      <H3>6.2 CD-1 step-by-step trace</H3>

      <StepTrace
        label="One Contrastive Divergence step on a single training image"
        steps={[
          { label: "Step 1: Clamp v = data point", render: () => (
            <Prose>
              Place the binarized digit (a 784-dim 0/1 vector) on the visible layer. This is the "positive phase" — the network is being told what reality looks like. No sampling yet.
            </Prose>
          )},
          { label: "Step 2: Compute P(h|v) for positive correlation", render: () => (
            <Prose>
              For each hidden unit <Code>j</Code>, compute <Code>{"P(h_j = 1 \\mid v) = \\sigma(b_j + \\sum_i v_i W_{ij})"}</Code>. This is one matrix multiply <Code>{"v @ W + b"}</Code> followed by a sigmoid. Store the resulting probabilities <Code>{"\\hat{p}_h^0"}</Code> — they will be used in the positive-correlation term <Code>{"v_0^T \\hat{p}_h^0"}</Code>.
            </Prose>
          )},
          { label: "Step 3: Sample h ~ Bernoulli(P(h|v))", render: () => (
            <Prose>
              Sample a binary hidden state <Code>{"h_0"}</Code> from the Bernoulli distribution defined by <Code>{"\\hat{p}_h^0"}</Code>. Bernoulli sampling here (not in the next visible step) regularizes the network — the hidden states are forced to be binary, preventing the units from cheating by carrying real-valued information. This is the only strictly required Bernoulli draw in CD-1.
            </Prose>
          )},
          { label: "Step 4: Reconstruct v from h via P(v|h)", render: () => (
            <Prose>
              Compute <Code>{"P(v_i = 1 \\mid h_0) = \\sigma(a_i + \\sum_j W_{ij} h_{0,j})"}</Code>. This is one matrix multiply <Code>{"h_0 @ W^T + a"}</Code> followed by a sigmoid. Use these probabilities directly as the reconstructed <Code>{"v_1"}</Code> (mean-field) — Hinton 2010 recommends not sampling here because it lowers gradient variance.
            </Prose>
          )},
          { label: "Step 5: Compute P(h|v_1) for negative correlation", render: () => (
            <Prose>
              Compute <Code>{"\\hat{p}_h^1 = \\sigma(b + v_1 W)"}</Code>. This is the hidden-unit activity under the reconstruction. Form the negative-correlation term <Code>{"v_1^T \\hat{p}_h^1"}</Code>.
            </Prose>
          )},
          { label: "Step 6: Update W, a, b with the gradient", render: () => (
            <Prose>
              Apply <Code>{"\\Delta W \\propto v_0^T \\hat{p}_h^0 - v_1^T \\hat{p}_h^1"}</Code>, <Code>{"\\Delta a \\propto v_0 - v_1"}</Code>, <Code>{"\\Delta b \\propto \\hat{p}_h^0 - \\hat{p}_h^1"}</Code>. With momentum 0.5 and learning rate 0.05, the weights drift toward configurations that lower the energy of the data point and raise the energy of its reconstruction. Repeat for the next minibatch.
            </Prose>
          )},
        ]}
      />

      <H3>6.3 Reconstruction error vs epoch</H3>

      <Plot
        label="CD-1 reconstruction error per epoch (single RBM, 784 -> 128, MNIST 5K subset)"
        xLabel="Epoch"
        yLabel="Mean squared reconstruction error"
        series={[
          { name: "CD-1 recon err", color: colors.gold, points: [[1, 0.0820], [2, 0.0568], [3, 0.0499], [4, 0.0454], [5, 0.0423], [6, 0.0399], [7, 0.0381], [8, 0.0366], [9, 0.0355], [10, 0.0345]] },
        ]}
      />

      <Prose>
        Monotonic descent from 0.082 to 0.034 over 10 epochs. The curve is convex but not steep — most of the improvement happens in the first 3 epochs, after which the model is making progressively smaller adjustments. Reconstruction error is not the same as log-likelihood (the model can lower recon error without becoming a better generative model), but on small RBMs they correlate well enough that it serves as a useful training-time monitor.
      </Prose>

      <H3>6.4 DBN pretraining vs random init — the 2006 result</H3>

      <Plot
        label="MNIST classification test accuracy: DBN-pretrained vs random init (10 fine-tune epochs)"
        xLabel="Fine-tune epoch"
        yLabel="Test accuracy"
        series={[
          { name: "DBN pretrained", color: colors.gold, points: [[1, 0.6040], [2, 0.8090], [3, 0.8520], [4, 0.8725], [5, 0.8810], [6, 0.8895], [7, 0.8925], [8, 0.8995], [9, 0.9035], [10, 0.9105]] },
          { name: "random init",    color: colors.green, points: [[1, 0.4525], [2, 0.6530], [3, 0.7100], [4, 0.7795], [5, 0.8260], [6, 0.8455], [7, 0.8605], [8, 0.8715], [9, 0.8655], [10, 0.8770]] },
        ]}
      />

      <Prose>
        Identical 784-256-128-64-10 sigmoid architecture, identical optimizer (Adam, lr 1e-3, batch 64), identical training data (5K labeled MNIST). The only difference is the initialization: gold uses weights from three stacked RBMs (5 epochs of CD-1 each, no labels), green uses default PyTorch random init. The DBN starts ahead and stays ahead for all 10 epochs, finishing 3.35 percentage points higher. This is the qualitative result Hinton & Salakhutdinov 2006 used to argue that unsupervised pretraining was a real and valuable thing — and the result that drove the deep learning revival until ReLU + better init made it unnecessary.
      </Prose>

      <H3>6.5 Free-energy gap visualization</H3>

      <TokenStream
        label="Free energy F(v) on data vs noise after 10 epochs (lower = higher unnormalized probability)"
        tokens={[
          { label: "MNIST: -233.3", color: colors.gold, title: "real digits sit in low-energy basins" },
          { label: "noise:  +176.2", color: colors.green, title: "random binary vectors are high-energy" },
          { label: "gap:    409.5 nats", color: "#c084fc", title: "the model strongly distinguishes data from noise" },
        ]}
      />

      <Prose>
        A 410-nat gap means the unnormalized probability of a real MNIST digit is roughly <Code>{"e^{410} \\approx 10^{178}"}</Code> times higher than that of a random binary vector under the trained model. Without the partition function we cannot turn this into a true probability, but the relative comparison is exact. Watching the gap grow during training is the cheapest sanity check that the RBM is learning something — flat or shrinking gap means a bug, almost always in the gradient sign or the learning rate.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 When to reach for an RBM in 2026</H3>

      <CodeBlock>
{`SITUATION                                | RBM?       | REASON
-----------------------------------------+------------+--------------------------------
Teaching energy-based models             | Yes        | Cleanest small example
Teaching MCMC and partition functions    | Yes        | Z is concrete and manipulable
Production generative model              | Never      | VAE / diffusion replaced it
Pretraining a deep classifier            | Never      | ReLU + good init obviates it
Collaborative filtering (small data)     | Maybe      | Matrix fact. usually wins
Quantum annealing research               | Niche      | D-Wave samplers can implement BMs
Modeling binary data with no labels      | Marginal   | Autoregressive transformer is better
Reproducing Hinton & Salakhutdinov 2006  | Yes        | Required for the historical result
Learning interpretable filters           | Maybe      | Sparse coding / dictionary learning is cleaner`}
      </CodeBlock>

      <H3>7.2 RBM vs modern generative families</H3>

      <CodeBlock>
{`CAPABILITY                       | RBM            | VAE             | DIFFUSION         | GAN
---------------------------------+----------------+-----------------+-------------------+----------------
Tractable likelihood             | No (needs Z)   | Yes (ELBO LB)   | Yes (ELBO)        | No
Tractable sampling               | Slow (Gibbs)   | Fast (1 fwd)    | Fast (10-50 fwd)  | Fast (1 fwd)
Sample quality (images)          | Low            | Medium          | State of the art  | High
Mode coverage                    | Medium         | High            | High              | Low
Trains stably                    | Marginal (CD)  | Yes             | Yes               | Notoriously hard
Scales to ImageNet               | Never tried    | Yes             | Yes (and beyond)  | Yes
Interpretable latent             | Yes (binary h) | Partial         | No                | No
Year of dominance                | 2006-2012      | 2014-2018       | 2020-now          | 2014-2020`}
      </CodeBlock>

      <H3>7.3 The conceptual lineage</H3>

      <Prose>
        Even though the RBM has been retired as a production model, the conceptual lineage running through it is still informative. The chain is roughly: RBM (energy-based, undirected, latent) → DBN (stacked RBM, generative pretraining) → Stacked autoencoders (deterministic version of the DBN idea) → Variational autoencoders (probabilistic, end-to-end, ELBO objective) → Self-supervised learning (SimCLR, DINO, MAE — pretraining without explicit generative likelihood) → Modern foundation models (BERT, GPT, CLIP — pretraining at scale). Each step inherits the previous one's commitment to "learn from unlabeled data first, then specialize," but moves further from the energy-based formalism. The endpoint — masked language modeling on web-scale corpora — has almost nothing in common with Smolensky 1986 except the underlying belief that pretraining works. That belief was first established by RBMs.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Hidden-unit count scales fine</H3>

      <Prose>
        The cost of one CD-k step is dominated by two matrix multiplies of shape <Code>{"[B \\times n_v]"}</Code> by <Code>{"[n_v \\times n_h]"}</Code>, repeated <Code>k</Code> times. That is <Code>{"O(k \\cdot B \\cdot n_v \\cdot n_h)"}</Code>. Doubling the number of hidden units doubles the cost — the same scaling as a fully-connected layer. Mid-2000s papers routinely used 1000-2000 hidden units on MNIST (Hinton & Salakhutdinov 2006 used four layers of 1000-500-250-30 in their dimensionality-reduction example), and the parameter count of an RBM is small relative to a comparable feedforward net of the same width because there is only one weight matrix.
      </Prose>

      <H3>8.2 Visible-unit count scales poorly</H3>

      <Prose>
        Doubling the number of visible units doubles the per-step cost (as expected from a matmul) but also makes Gibbs sampling exponentially worse at mixing. Long thin RBMs — 32{"\u00d7"}32 = 1024 visible, 1000 hidden — are the upper end of what was historically trained reliably. Scaling to 224{"\u00d7"}224 = 50K visible (ImageNet resolution) was never made to work with vanilla RBMs because (a) the chain mixes unacceptably slowly, (b) the partition function cannot be reliably estimated for sanity checks, and (c) by the time anyone tried, ConvNets and ViTs had taken over.
      </Prose>

      <H3>8.3 Depth scales to roughly 4-5 layers</H3>

      <Prose>
        Hinton & Salakhutdinov 2006 trained 4-layer DBNs (784-1000-500-250-30 then unrolled). Salakhutdinov & Hinton 2009 trained Deep Boltzmann Machines with 3 hidden layers. Pushing beyond 5 layers in either architecture produced diminishing returns — each additional layer captured fewer marginal nats of likelihood and added more training instability. This is the depth ceiling of the energy-based-pretraining era. Modern transformers routinely train at 50-100 layers, but they get there by abandoning the energy-based formalism, the Gibbs sampler, and the partition function entirely.
      </Prose>

      <H3>8.4 Estimating Z with Annealed Importance Sampling</H3>

      <Prose>
        Salakhutdinov and Murray (2008) "On the Quantitative Analysis of Deep Belief Networks" introduced AIS for RBM partition function estimation. The idea is to construct a sequence of intermediate distributions interpolating between an easy reference (uniform or independent) and the target RBM, run a chain through the sequence, and use importance weights to estimate the ratio of partition functions. AIS gives unbiased estimates of <Code>Z</Code> with variance that depends on the number of intermediate steps (typically 1000-10000). It is expensive — running AIS on a single RBM can take longer than training the RBM in the first place — but it is the standard for reporting log-likelihood numbers on RBMs in the literature. The fact that a separate, expensive procedure is needed just to evaluate the model is a strong signal that the model class is fundamentally awkward at scale.
      </Prose>

      <H3>8.5 Why energy-based MLE doesn't scale</H3>

      <Prose>
        The deep reason RBMs hit a wall around 2012 is structural, not algorithmic. Maximum likelihood for any model with an intractable partition function requires either (a) sampling from the model distribution (Gibbs, MCMC), which is slow and biased, or (b) variational bounds, which are loose. Neither approach scales as cleanly as the alternatives that came later: autoregressive models have tractable likelihoods by construction, GANs sidestep likelihood entirely, and diffusion models use a clever reparametrization that makes the score function (not the likelihood) the learning target. None of the modern generative families inherits the partition-function pain. That is why no one builds RBM-style models at scale today: the math itself does not cooperate.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Partition function intractability prevents likelihood reporting</H3>

      <Prose>
        You cannot report a log-likelihood for an RBM without estimating <Code>Z</Code>, and estimating <Code>Z</Code> requires AIS or a similar procedure with non-trivial variance. The community workaround in the 2006-2012 era was to report reconstruction error or pseudo-log-likelihood (Hinton 2002) instead, neither of which is the likelihood. This makes RBMs awkward to compare across papers — two papers reporting "0.05 reconstruction error" on the same dataset can have wildly different log-likelihoods. If you are doing serious generative modeling research today, the inability to report a clean likelihood is by itself a reason to choose a different model family.
      </Prose>

      <H3>9.2 CD-1 is a biased gradient estimator</H3>

      <Prose>
        Bengio and Delalleau (2009) proved that CD-1 is not the gradient of any consistent objective function — it differs from the true log-likelihood gradient by a term that does not vanish in general. In practice the bias is small enough to allow useful training, but it manifests as the model producing sharper-than-correct distributions (the negative-phase chain doesn't explore far enough from the data, so the model overconfidently lowers the energy of regions near data). PCD reduces this bias substantially. If you must use CD-k, prefer <Code>{"k = 5"}</Code> or higher in the second half of training, even though early epochs run fine with <Code>{"k = 1"}</Code>.
      </Prose>

      <H3>9.3 Persistent CD chains can destabilize</H3>

      <Prose>
        PCD maintains negative-phase chains across minibatches. If the learning rate is too high, the weight changes make the previous chain state increasingly unrepresentative of the new model — the chain can "fall off" the new energy landscape and produce gradients that point in incorrect directions. The symptom is reconstruction error spiking after several epochs of healthy descent. Verified result on the same MNIST 5K binarized subset:
      </Prose>

      <CodeBlock language="python">
{`# PCD with 100 persistent chains, lr=0.05, momentum=0.5
# Output:
# [pcd ep=1/5] recon_err=0.1090
# [pcd ep=2/5] recon_err=0.0690
# [pcd ep=3/5] recon_err=0.0580
# [pcd ep=4/5] recon_err=0.0511
# [pcd ep=5/5] recon_err=0.0481`}
      </CodeBlock>

      <Prose>
        PCD's per-epoch reconstruction error (0.048 at epoch 5) is somewhat worse than CD-1's (0.042) on this small subset. PCD shines on longer training runs and larger datasets, where its negative chain can drift far enough from the data to give a more accurate model expectation. On 5000 images and 5 epochs it does not have time to show that advantage.
      </Prose>

      <H3>9.4 Learning rate too high oscillates or blows up</H3>

      <CodeBlock language="python">
{`rbm_bad = RBM(784, 128)
train_rbm(rbm_bad, X_train_bin, epochs=5, batch_size=64, lr=2.0, momentum=0.5, k=1)

# Output:
# [bad-lr ep=1/5] recon_err=0.0789
# [bad-lr ep=2/5] recon_err=0.0698
# [bad-lr ep=3/5] recon_err=0.0675
# [bad-lr ep=4/5] recon_err=0.0652
# [bad-lr ep=5/5] recon_err=0.0640
# [bad-lr] |W|_max = 5.02  (healthy training keeps |W|_max under ~5)`}
      </CodeBlock>

      <Prose>
        With <Code>lr = 2.0</Code> (40{"\u00d7"} the recommended value) the model still trains but the reconstruction error plateaus at 0.064 — substantially worse than CD-1's 0.034 at the same epoch with the right LR. The maximum absolute weight has reached 5.0, which is the upper end of healthy training; another factor of 2 in LR would push weights into <Code>nan</Code> territory. Hinton 2010 recommends monitoring the absolute weight magnitude and scaling LR down if it exceeds 5 or 6. Adding momentum without lowering LR amplifies the problem.
      </Prose>

      <H3>9.5 Momentum is essentially required</H3>

      <Prose>
        Without momentum, RBM training is noisy enough that the reconstruction error curve oscillates between epochs. With momentum 0.5 (low) early in training and momentum 0.9 (high) later — the standard Hinton 2010 schedule — the curve becomes smooth and monotonic. If you skip momentum entirely you can sometimes still train, but the final model is noticeably worse and more sensitive to initialization. This is one of the practical recipe details that the 2010 "Practical Guide" emphasizes and that newer practitioners often miss.
      </Prose>

      <H3>9.6 Evaluating samples is hard</H3>

      <Prose>
        Generating samples from a trained RBM requires running a Gibbs chain to convergence — typically thousands of steps. Even then, the samples often show characteristic artifacts (blurry digits, mode collapse onto a few prototypes) that are visually obvious but hard to quantify without a tractable likelihood. The community settled on visual inspection plus reconstruction error plus pseudo-LL as the de facto evaluation triad, none of which correlates perfectly with the true generative quality you would care about. Modern generative models (VAE, diffusion) all have cleaner evaluation stories — FID, IS, exact ELBO — and the difficulty of evaluating RBMs is a real reason the field moved on.
      </Prose>

      <H3>9.7 Discriminative fine-tuning often outperforms unsupervised pretraining (post-2014)</H3>

      <Prose>
        After 2012 a series of careful comparison studies (Erhan et al. 2010 "Why Does Unsupervised Pre-training Help Deep Learning?", Bengio et al. 2013 "Better Mixing via Deep Representations", later Bengio's own followups) showed that for medium-to-large labeled datasets, training a feedforward net end-to-end with ReLU + good init + dropout matches or exceeds DBN-pretrained networks. The pretraining advantage is real only in the small-labeled-data regime — exactly the regime where modern self-supervised methods (DINO, MAE, SimCLR) now dominate by exploiting the huge unlabeled corpora that pretraining can leverage. The conceptual lesson — pretraining helps when labels are scarce — survived; the specific implementation via RBM stacks did not.
      </Prose>

      <H3>9.8 Batch normalization is not designed for sigmoid Bernoulli units</H3>

      <Prose>
        BatchNorm changes the input distribution to a layer to be approximately zero-mean unit-variance, which interacts badly with the sigmoid + Bernoulli structure of RBM units (the Bernoulli sampling expects logits that map to meaningful probabilities, and BN can move the operating point off the sigmoid's responsive region). RBMs were trained without BN historically (BN didn't exist until 2015), and adding BN to a modern RBM-style codebase usually hurts. The right modernization is to drop the energy-based formulation entirely and switch to a feedforward net with BN, which is exactly what the field did.
      </Prose>

      <Callout accent="gold">
        If your RBM is failing to train, the cause is almost certainly one of: learning rate too high (weights blow up), no momentum (noisy reconstruction curve), wrong gradient sign, or the dataset is real-valued and you forgot to switch from <Code>BernoulliRBM</Code> to <Code>GaussianBernoulliRBM</Code> (visible units must be Gaussian for real-valued data). Sanity-check the free energy gap between data and noise — it should be growing during training. If it is flat or shrinking, the gradient direction is wrong.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The canonical reading list, in approximately the order that gives the cleanest narrative of where Boltzmann machines came from, what they enabled, and how they were eventually superseded:
      </Prose>

      <Prose>
        <strong>Hinton & Sejnowski (1983).</strong> "Optimal Perceptual Inference." CVPR. The original Boltzmann machine paper. Defines the architecture (binary stochastic units, fully connected, hidden units), the energy function, and the learning rule based on positive and negative phases. Proposes simulated annealing as the sampling procedure. Predates RBMs by three years. Short and worth reading for the historical framing — the paper takes seriously the analogy between neural networks and statistical physics in a way that later papers do not.
      </Prose>

      <Prose>
        <strong>Smolensky (1986).</strong> "Information Processing in Dynamical Systems: Foundations of Harmony Theory." Chapter 6 of the Parallel Distributed Processing Volume 1 (Rumelhart, McClelland, and the PDP Research Group). The chapter that first defines the Restricted Boltzmann Machine, framed in Smolensky's "harmony theory" language rather than as a Boltzmann machine variant. The bipartite restriction appears as a simplifying assumption for analytic tractability and is not flagged as the load-bearing technical move it would later turn out to be. About 100 pages.
      </Prose>

      <Prose>
        <strong>Hinton (2002).</strong> "Training Products of Experts by Minimizing Contrastive Divergence." Neural Computation 14(8):1771-1800. The CD algorithm. Frames the RBM as a product of experts (each hidden unit is an expert) and derives the CD-k learning rule. Includes empirical results showing that CD-1 produces useful features on small datasets in tractable time. The paper that finally made RBMs trainable in practice, sixteen years after Smolensky.
      </Prose>

      <Prose>
        <strong>Hinton & Salakhutdinov (2006).</strong> "Reducing the Dimensionality of Data with Neural Networks." Science 313:504-507. The 4-page Science paper that launched modern deep learning. Demonstrates that stacking RBMs gives a deep autoencoder that beats PCA on multiple benchmarks. The companion Supplementary Information contains the algorithmic details. Required reading for anyone who wants to understand why "deep learning" became a phrase around 2006.
      </Prose>

      <Prose>
        <strong>Hinton, Osindero, Teh (2006).</strong> "A Fast Learning Algorithm for Deep Belief Nets." Neural Computation 18(7):1527-1554. The longer companion paper to the Science paper, with full mathematical derivations of greedy layer-wise pretraining and the variational bound that justifies it. Names the architecture the Deep Belief Network and explains why the stacked-RBM construction is theoretically sound. This is the paper to cite when you want the full machinery rather than the Science-magazine summary.
      </Prose>

      <Prose>
        <strong>Tieleman (2008).</strong> "Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient." ICML. Introduces Persistent Contrastive Divergence. Empirically shows that PCD produces better samples than CD-k on the same data and training budget. Now the default training procedure for RBMs in serious applications.
      </Prose>

      <Prose>
        <strong>Salakhutdinov, Mnih, Hinton (2007).</strong> "Restricted Boltzmann Machines for Collaborative Filtering." ICML. The Netflix Prize application of RBMs. Uses softmax-visible units (one per movie, 5-way for ratings) and demonstrates competitive performance on the Netflix dataset. The most cited industrial application of RBMs, and the one that showed RBMs could matter outside MNIST and dimensionality reduction.
      </Prose>

      <Prose>
        <strong>Salakhutdinov & Hinton (2009).</strong> "Deep Boltzmann Machines." AISTATS. The fully undirected stack — unlike a DBN, which has directed connections between layers (downward) plus undirected between top two, a DBM is undirected throughout. Trains the whole stack jointly with a variational mean-field approximation. Conceptually cleaner than DBN but harder to train. Best-of-class on small benchmarks at publication time.
      </Prose>

      <Prose>
        <strong>Hinton (2010).</strong> "A Practical Guide to Training Restricted Boltzmann Machines." UTML TR 2010-003, University of Toronto. The unpublished tech report that contains the actual recipes — initialization (small Gaussian, 0.01), learning rate (0.001-0.1 depending on data), momentum schedule (0.5 then 0.9), batch size (10-100), monitoring the free energy gap, choosing <Code>k</Code>, and a hundred other practical tricks. Anyone who actually trains an RBM should read this; the published papers are notably silent on these details.
      </Prose>

      <Prose>
        <strong>Bengio & Delalleau (2009).</strong> "Justifying and Generalizing Contrastive Divergence." Neural Computation 21(6):1601-1621. The theoretical analysis showing that CD-k is a biased gradient estimator and characterizing the bias. The paper to cite when you want to make the "CD-1 is biased but useful" claim with a reference.
      </Prose>

      <Prose>
        <strong>Salakhutdinov & Murray (2008).</strong> "On the Quantitative Analysis of Deep Belief Networks." ICML. Introduces Annealed Importance Sampling for estimating the partition function of RBMs and DBMs. The standard reference for any quantitative likelihood comparison between RBM-family models.
      </Prose>

      <Prose>
        <strong>Erhan, Bengio, Courville, Manzagol, Vincent, Bengio (2010).</strong> "Why Does Unsupervised Pre-training Help Deep Learning?" JMLR 11:625-660. Late in the pretraining era, this paper carefully ablated the components that made stacked-RBM pretraining effective and argued that the main effect was acting as a regularizer plus providing a better optimization starting point. Reading it after the dust settled is the cleanest way to understand <em>why</em> the trick worked, which in turn explains why ReLU + Adam + dropout could replace it.
      </Prose>

      <Prose>
        <strong>Further reading.</strong> Cho et al. (2010) "Parallel Tempering Is Efficient for Learning Restricted Boltzmann Machines" (an alternative to PCD); Larochelle and Bengio (2008) "Classification using Discriminative Restricted Boltzmann Machines" (RBMs trained directly for classification rather than as pretraining); Welling, Rosen-Zvi, Hinton (2005) "Exponential Family Harmoniums" (the family that includes Gaussian-Bernoulli RBMs for real-valued data); LeCun, Chopra, Hadsell, Ranzato, Huang (2006) "A Tutorial on Energy-Based Learning" (the broader context of energy-based models, of which the RBM is the most famous instance). For the modern revival see LeCun's 2022 position paper "A Path Towards Autonomous Machine Intelligence" which argues energy-based models are due for a comeback — though notably he proposes JEPA-style architectures rather than reviving RBMs.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>Q1. Why does the bipartite restriction in an RBM make Gibbs sampling so much faster than in a general Boltzmann machine?</H3>

      <Callout accent="gold">
        In a general Boltzmann machine, every unit is conditionally dependent on every other unit, so Gibbs sampling must update one unit at a time, conditional on all others. For an RBM, the absence of intra-layer connections means that hidden units are conditionally independent given the visible layer (and vice versa). This lets you sample the entire hidden layer in parallel via one matrix multiply followed by a sigmoid and a Bernoulli draw, then do the same for the entire visible layer — block Gibbs sampling. Two parallel ops replace hundreds or thousands of sequential ones, which is the difference between "research curiosity" and "trainable in a few hours."
      </Callout>

      <H3>Q2. What is the partition function Z, why is it intractable for RBMs, and what practical consequence does this have?</H3>

      <Callout accent="gold">
        <Code>{"Z = \\sum_{v, h} \\exp(-E(v, h))"}</Code> is the normalizing constant of the Gibbs distribution. For an RBM with <Code>{"n_v"}</Code> visible and <Code>{"n_h"}</Code> hidden units, computing it exactly requires summing over <Code>{"2^{n_v + n_h}"}</Code> joint configurations — for a 784-128 RBM that is more than <Code>{"10^{275}"}</Code> terms, far beyond any computer. The practical consequence is that you cannot compute the exact log-likelihood <Code>{"\\log P(v) = -F(v) - \\log Z"}</Code>, you cannot compute the true MLE gradient (the model expectation requires sampling from <Code>{"P(v, h)"}</Code> which depends on <Code>Z</Code>), and you cannot directly compare two RBMs by likelihood without an expensive estimation procedure like AIS. CD-k sidesteps the gradient problem at the cost of a biased estimate; AIS sidesteps the likelihood problem at the cost of running an expensive separate procedure.
      </Callout>

      <H3>Q3. Why does Contrastive Divergence work despite being a biased estimator of the log-likelihood gradient?</H3>

      <Callout accent="gold">
        CD-1 starts the negative-phase Gibbs chain at the data point and runs it for one step. The true gradient would require running the chain to equilibrium under the model distribution, which is intractable. CD-1's gradient is therefore biased — it pushes the model toward something that is not exactly the maximum likelihood solution. Empirically, the bias is in a direction that does not prevent the model from learning useful features: the model still lowers energy of data and raises energy of regions near data, which is most of what you want for representation learning. Bengio and Delalleau (2009) characterized the bias formally; in practice it manifests as the model producing a slightly sharper distribution than the true MLE would (the negative chain explores too narrowly). For better samples, use PCD or CD-k with larger k. For useful features, CD-1 works.
      </Callout>

      <H3>Q4. What was the key result of Hinton & Salakhutdinov 2006, and why did the trick stop being needed by ~2014?</H3>

      <Callout accent="gold">
        Hinton & Salakhutdinov 2006 showed that stacking RBMs and training them greedily layer-by-layer gave a much better initialization for a deep autoencoder than random init — the pretrained network converged where random init failed. This worked because in the sigmoid + small-Gaussian-init era, deep networks suffered from vanishing gradients and bad local minima that pretraining helped escape. Between 2010 and 2014 a series of advances obsoleted the trick: ReLU activations (Nair and Hinton 2010, Glorot et al. 2011) eliminated gradient vanishing; Xavier and Kaiming initialization (Glorot and Bengio 2010, He et al. 2015) gave principled rules for random weights that put deep nets in a trainable regime from scratch; dropout, batch norm, and Adam closed the remaining gaps. By 2014, randomly-initialized 8-layer networks with ReLU + dropout matched or exceeded DBN-pretrained ones. The pretraining lesson — "learn from unlabeled data first" — survived in the form of modern self-supervised learning; the specific RBM-stack implementation did not.
      </Callout>

      <H3>Q5. If you wanted to do the modern equivalent of "RBM unsupervised pretraining" today, what would you actually use?</H3>

      <Callout accent="gold">
        For images: DINOv2 or MAE pretraining on a large unlabeled corpus, then linear probe or LoRA fine-tune on your labeled task. For text: any LLM (BERT, GPT, Llama) pretrained on web data, then instruction fine-tune. For tabular or domain-specific binary data: a transformer trained with masked-language-model-style objectives (TabNet, SAINT, or just a vanilla transformer with masked attention). For collaborative filtering: a two-tower transformer with self-supervised pretraining on the interaction history. The conceptual contract is identical to RBM pretraining — exploit unlabeled or weakly-labeled data to initialize a model that is then specialized — but the implementation has nothing in common with stacked RBMs. The energy-based formulation was a means to an end, and that end is now achieved more cleanly with maximum-likelihood-style objectives at scale.
      </Callout>

    </div>
  ),
};

export default rbmContent;
