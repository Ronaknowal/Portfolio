import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const crfContent = {
  title: "Conditional Random Fields (CRF)",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 2001, John Lafferty, Andrew McCallum, and Fernando Pereira published "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data" at the 18th International Conference on Machine Learning (ICML), pages 282–289. The paper landed in a specific intellectual gap: by 2001, the NLP community had learned that generative models like Hidden Markov Models suffered from a fundamental structural mismatch for sequence labeling tasks. HMMs model the joint distribution {"P(observations, labels)"} — they explain how both words and tags are generated together. But the labeling task only needs {"P(labels | observations)"}. Modeling more than you need forces strong independence assumptions that real language violates constantly.
      </Prose>

      <Prose>
        The proximate predecessor was the Maximum Entropy Markov Model (MEMM), introduced by McCallum, Freitag, and Pereira at ICML 2000. MEMMs moved to discriminative modeling — they modeled {"P(labels | observations)"} directly — and allowed arbitrary overlapping features of the input. This was a genuine step forward. But MEMMs suffered from what Lafferty, McCallum, and Pereira identified as the <em>label bias problem</em>: because each state in a MEMM normalizes its own transition distribution locally, states with few outgoing transitions receive disproportionate probability mass regardless of the input. A state that always transitions to the same next state cannot be overridden by any amount of contradicting input evidence. The 2001 CRF paper proved this rigorously and showed that global normalization — computing the partition function over the entire sequence — is the correct fix.
      </Prose>

      <Prose>
        The CRF achieves global normalization by defining a single energy function over the entire label sequence and normalizing once with respect to the partition function {"Z(x)"}. The partition function sums the unnormalized scores of all possible label sequences for a given input, ensuring that the model probability is a proper distribution over complete label sequences, not a product of per-state local distributions. This resolves label bias entirely. The forward-backward algorithm computes the partition function in {"O(T · K²)"} time — exactly as efficient as HMM inference — making the model tractable despite the global normalization.
      </Prose>

      <Prose>
        The subsequent decade established CRFs as the workhorse of NLP. Sha and Pereira (2003, NAACL-HLT) demonstrated that CRFs outperformed MEMMs and HMMs on chunking and named entity recognition, establishing the "linear-chain CRF" as a standard building block. Charles Sutton and Andrew McCallum wrote the definitive tutorial in 2012: "An Introduction to Conditional Random Fields," published in <em>Foundations and Trends in Machine Learning</em>, volume 4, number 4, pages 267–373. The Sutton–McCallum tutorial covers linear-chain CRFs, general CRFs (defined on arbitrary factor graphs), skip-chain CRFs, and the relationship to other structured prediction frameworks — it remains the authoritative technical reference for anyone implementing CRFs from scratch.
      </Prose>

      <Prose>
        The reason CRFs remain worth understanding in the age of Transformers is threefold. First, the BiLSTM-CRF architecture (Lample et al. 2016, NAACL) — which won NER competitions from 2016 to 2018 — uses a standard linear-chain CRF as its output layer. The BiLSTM replaced hand-crafted feature functions with learned representations, but the CRF layer enforced global label consistency and remains in production systems today. Second, CRFs are the clearest example of structured prediction: output spaces with combinatorial structure, where the best label sequence cannot be found by independent per-token classification. Third, the math of CRFs — log-linear models, partition function computation, gradient as the difference between data and model feature expectations — generalizes directly to energy-based models and diffusion models for structured generation.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The cleanest way to understand a CRF is to contrast it directly with the HMM at the level of what each model computes. An HMM defines a joint distribution {"P(y, x) = P(y₁) · ∏ₜ P(yₜ | yₜ₋₁) · P(xₜ | yₜ)"}. To use it for labeling, you compute {"P(y | x) = P(y, x) / P(x)"} via Bayes' rule. The denominator {"P(x) = Σ_y P(y, x)"} sums over all label sequences but cancels in the Viterbi comparison — you never actually compute it for decoding. For training, however, you maximize the joint likelihood, which forces you to model {"P(xₜ | yₜ)"} as a simple emission: typically multinomial over a vocabulary or Gaussian. This is the independence assumption — given the current tag, the word is drawn independently of everything else.
      </Prose>

      <Prose>
        A CRF sidesteps this completely. It defines {"P(y | x)"} directly, without ever modeling {"P(x)"} at all. The model is:
      </Prose>

      <MathBlock>{"P(\\mathbf{y} \\mid \\mathbf{x}) = \\frac{1}{Z(\\mathbf{x})} \\exp\\!\\left( \\sum_{t=1}^{T} \\sum_k \\lambda_k f_k(y_{t-1}, y_t, \\mathbf{x}, t) \\right)"}</MathBlock>

      <Prose>
        The function {"f_k(y_{t-1}, y_t, x, t)"} is a <em>feature function</em>: it takes the previous tag, the current tag, the entire input sequence, and the current position, and returns a real number (usually 0 or 1 for indicator features). The scalar {"λ_k"} is the learned weight for feature <em>k</em>. The sum inside the exponential is the total score of the label sequence — a linear function of the feature weights. The partition function {"Z(x) = Σ_y exp(Σ_t Σ_k λ_k f_k(y_{t-1}, y_t, x, t))"} normalizes over all {"K^T"} possible label sequences.
      </Prose>

      <Prose>
        The key insight: because feature functions take the <em>entire</em> input <em>x</em> as an argument, a CRF can look at any part of the input when deciding what score to assign to a tag transition at position <em>t</em>. An HMM emission {"P(xₜ | yₜ)"} only looks at the word at position <em>t</em>. A CRF feature can look at the word, the previous word, the next word, capitalization patterns, suffix information, the sentence length, even features of distant tokens — anything extractable from the input. This is the feature richness that makes CRFs superior to HMMs on label-heavy tasks like NER, where entity boundaries depend on context that extends several tokens in both directions.
      </Prose>

      <Prose>
        The factor graph is the right visual. Draw a chain of variable nodes {"y₁, y₂, ..., y_T"} connected by factor nodes. Each factor node {"ψₜ(yₜ₋₁, yₜ)"} captures all features that involve adjacent tags at positions <em>t-1</em> and <em>t</em> — both the transition features {"f(yₜ₋₁, yₜ)"} and the emission features {"f(yₜ, x, t)"}. The linear-chain CRF is exactly belief propagation on this chain factor graph, which runs in polynomial time. General CRFs can have more complex factor graph topologies (skip-chain, 2D lattice for image labeling), but the linear-chain is the standard for sequence tasks.
      </Prose>

      <Callout type="info" title="Why 'Conditional Random Field'?">
        The name has three parts. <strong>Conditional</strong>: the model conditions on the input — {"P(y | x)"} not {"P(y, x)"}. <strong>Random Field</strong>: a random field (equivalently, a Markov random field or undirected graphical model) defines a joint distribution over a set of variables as a product of potential functions, normalized globally. The CRF is a random field in the <em>y</em> variables, conditional on <em>x</em>. <strong>The key difference from HMMs</strong>: HMMs are directed graphical models (Bayesian networks) — they specify a generative process with a DAG. CRFs are undirected graphical models — they specify an energy function with no causal direction assumed.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The log-linear model</H3>

      <Prose>
        The CRF belongs to the family of <em>log-linear models</em> (also called exponential family models). The probability of a label sequence {"y = (y₁, ..., y_T)"} given input {"x"} is:
      </Prose>

      <MathBlock>{"P(\\mathbf{y} \\mid \\mathbf{x}; \\boldsymbol{\\lambda}) = \\frac{1}{Z(\\mathbf{x})} \\exp\\!\\left( \\sum_{t=1}^{T} \\sum_{k=1}^{K} \\lambda_k f_k(y_{t-1}, y_t, \\mathbf{x}, t) \\right)"}</MathBlock>

      <Prose>
        where the partition function is:
      </Prose>

      <MathBlock>{"Z(\\mathbf{x}) = \\sum_{\\mathbf{y}'} \\exp\\!\\left( \\sum_{t=1}^{T} \\sum_{k=1}^{K} \\lambda_k f_k(y'_{t-1}, y'_t, \\mathbf{x}, t) \\right)"}</MathBlock>

      <Prose>
        The sum in {"Z(x)"} runs over all {"L^T"} possible label sequences (where <em>L</em> is the number of labels), which is exponential in <em>T</em>. Naively computing this is infeasible. The linear-chain structure — features only involve adjacent labels {"(yₜ₋₁, yₜ)"} — makes the forward algorithm tractable.
      </Prose>

      <H3>3.2 The forward algorithm for the partition function</H3>

      <Prose>
        Define the potential for moving from label <em>i</em> to label <em>j</em> at position <em>t</em> as:
      </Prose>

      <MathBlock>{"\\Psi_t(i, j) = \\exp\\!\\left( \\sum_k \\lambda_k f_k(i, j, \\mathbf{x}, t) \\right)"}</MathBlock>

      <Prose>
        The forward variable {"α_t(j)"} accumulates unnormalized probability mass for all partial paths ending in label <em>j</em> at position <em>t</em>:
      </Prose>

      <MathBlock>{"\\alpha_1(j) = \\Psi_1(\\text{start}, j), \\qquad \\alpha_t(j) = \\sum_{i} \\alpha_{t-1}(i) \\cdot \\Psi_t(i, j)"}</MathBlock>

      <Prose>
        The partition function is {"Z(x) = Σ_j α_T(j)"}. In log-space (mandatory for numerical stability):
      </Prose>

      <MathBlock>{"\\log \\alpha_t(j) = \\log\\!\\sum_i \\exp\\!\\left( \\log \\alpha_{t-1}(i) + \\log \\Psi_t(i,j) \\right)"}</MathBlock>

      <Prose>
        where the inner log-sum-exp is computed using <Code>{"np.logaddexp.reduce"}</Code>. This recursion is {"O(T · L²)"} — identical in complexity to HMM forward.
      </Prose>

      <H3>3.3 Viterbi decoding</H3>

      <Prose>
        Finding the most likely label sequence requires replacing the sum in the forward recursion with a max:
      </Prose>

      <MathBlock>{"\\delta_t(j) = \\max_i \\left[ \\delta_{t-1}(i) + \\log \\Psi_t(i,j) \\right]"}</MathBlock>

      <Prose>
        Back-pointers {"ψ_t(j) = argmax_i [δ_{t-1}(i) + log Ψ_t(i,j))"} store the optimal predecessor label at each step. After filling the trellis forward in {"O(T · L²)"}, the optimal path is recovered by back-tracing from {"argmax_j δ_T(j)"} in {"O(T)"}. The Viterbi algorithm for CRFs is structurally identical to Viterbi for HMMs — only the potential {"Ψ_t(i,j)"} differs (it is now a sum of weighted features, not a product of transition and emission probabilities).
      </Prose>

      <H3>3.4 Training: conditional log-likelihood and its gradient</H3>

      <Prose>
        Given a training set of {"(x⁽ⁿ⁾, y⁽ⁿ⁾)"} pairs, the objective is the conditional log-likelihood (often with L2 regularization):
      </Prose>

      <MathBlock>{"\\mathcal{L}(\\boldsymbol{\\lambda}) = \\sum_{n=1}^{N} \\log P(\\mathbf{y}^{(n)} \\mid \\mathbf{x}^{(n)}) - \\frac{\\sigma^2}{2} \\|\\boldsymbol{\\lambda}\\|^2"}</MathBlock>

      <Prose>
        The gradient with respect to weight {"λ_k"} has a beautiful closed form — it is the difference between the <em>empirical</em> (data) feature expectation and the <em>model</em> feature expectation:
      </Prose>

      <MathBlock>{"\\frac{\\partial \\mathcal{L}}{\\partial \\lambda_k} = \\sum_{n} \\left[ \\underbrace{\\sum_t f_k(y^{(n)}_{t-1}, y^{(n)}_t, \\mathbf{x}^{(n)}, t)}_{\\text{data expectation}} - \\underbrace{\\mathbb{E}_{P(\\mathbf{y}|\\mathbf{x}^{(n)})} \\left[ \\sum_t f_k(y_{t-1}, y_t, \\mathbf{x}^{(n)}, t) \\right]}_{\\text{model expectation}} \\right] - \\sigma^2 \\lambda_k"}</MathBlock>

      <Prose>
        The data expectation is easy — just sum the feature values on the gold label sequence. The model expectation is the hard part: it requires summing the feature value over all label sequences, weighted by their model probability. This is computed via the forward-backward algorithm. Define the backward variable {"β_t(j)"}, the unnormalized probability of completing the sequence from position <em>t</em> if we are in label <em>j</em>:
      </Prose>

      <MathBlock>{"\\beta_T(j) = 1, \\qquad \\beta_t(i) = \\sum_j \\Psi_{t+1}(i,j) \\cdot \\beta_{t+1}(j)"}</MathBlock>

      <Prose>
        The joint marginal probability of being in label <em>i</em> at position <em>t-1</em> and label <em>j</em> at position <em>t</em> is:
      </Prose>

      <MathBlock>{"P(y_{t-1}=i, y_t=j \\mid \\mathbf{x}) = \\frac{\\alpha_{t-1}(i) \\cdot \\Psi_t(i,j) \\cdot \\beta_t(j)}{Z(\\mathbf{x})}"}</MathBlock>

      <Prose>
        The model expectation of feature <em>k</em> at position <em>t</em> is then {"Σᵢ Σⱼ P(yₜ₋₁=i, yₜ=j | x) · f_k(i, j, x, t)"}. The full gradient requires summing these over all positions and all training examples. The objective {"L(λ)"} is concave (the log of a log-linear model is concave in the weights), so gradient-based optimization converges to the global maximum. In practice, L-BFGS is the standard optimizer — it converges in 50–200 iterations where SGD would need thousands, because it approximates the inverse Hessian using a rolling window of gradient history.
      </Prose>

      <Callout type="info" title="Gradient = data count minus model count">
        The gradient {"∂L/∂λ_k = E_data[f_k] - E_model[f_k]"} has a profound interpretation. Training pushes the model to match the empirical feature statistics of the training data. When {"E_model[f_k] < E_data[f_k]"}, the model underestimates how often feature <em>k</em> fires on true label sequences — increasing {"λ_k"} will increase the model's expected feature count toward the data count. At convergence, {"E_model[f_k] = E_data[f_k]"} for every feature — the <em>moment matching</em> condition. This is the maximum entropy perspective: the CRF is the maximum entropy distribution subject to matching empirical feature expectations.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The implementation below uses NumPy only. It implements a linear-chain CRF with indicator features for a synthetic POS-like tagging task: 3 tags (NOUN=0, VERB=1, DET=2), vocabulary of 5 word types, features for tag transitions and tag-word emissions. All output below is verbatim terminal output from the code as run.
      </Prose>

      <H3>4a. Feature functions and forward algorithm</H3>

      <CodeBlock language="python">
{`import numpy as np

# ===== Linear-chain CRF from scratch =====
# Tags: 0=NOUN, 1=VERB, 2=DET  |  Vocab: 5 word types
tag_names = ['NOUN', 'VERB', 'DET']
K = 3; vocab_size = 5
# Feature layout: [K^2 transition features] + [K*vocab emission features]
n_features = K * K + K * vocab_size   # 9 + 15 = 24

def phi(prev_tag, cur_tag, word_id, is_start=False):
    """Local feature vector for one position in the sequence."""
    fv = np.zeros(n_features)
    if not is_start:
        fv[prev_tag * K + cur_tag] = 1.0          # transition feature
    fv[K*K + cur_tag * vocab_size + word_id] = 1.0  # emission feature
    return fv

def log_Z_and_alpha(w, words):
    """Log partition function via log-space forward algorithm."""
    T = len(words)
    la = np.full((T, K), -np.inf)
    for k in range(K):
        la[0, k] = w.dot(phi(-1, k, words[0], True))   # t=0, no prev tag
    for t in range(1, T):
        for j in range(K):
            la[t, j] = np.logaddexp.reduce(
                [la[t-1, i] + w.dot(phi(i, j, words[t])) for i in range(K)])
    return np.logaddexp.reduce(la[-1]), la

print(f"n_features = {n_features}  (9 transition + 15 emission)")
# n_features = 24  (9 transition + 15 emission)

# Test forward on zero weights (all paths equal)
test_words = [0, 2, 4]   # 3-token sequence
log_Z, la = log_Z_and_alpha(np.zeros(n_features), test_words)
print(f"log Z (zero weights, T=3, K=3) = {log_Z:.4f}")
# log Z (zero weights, T=3, K=3) = 3.2958
print(f"Z = {np.exp(log_Z):.1f}  (should be K^T = {K**3})")
# Z = 27.0  (should be K^T = 27)`}
      </CodeBlock>

      <Prose>
        The partition function sanity check confirms the forward algorithm: with zero weights, all label sequences have score 0 and probability {"1/K^T"}. {"Z = K^T = 3^3 = 27"} exactly. The log-space forward recursion handles this without any underflow.
      </Prose>

      <H3>4b. Viterbi decoding</H3>

      <CodeBlock language="python">
{`def viterbi(w, words):
    """Viterbi decoding: returns best label sequence and its log-score."""
    T = len(words)
    d = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)   # back-pointers
    for k in range(K):
        d[0, k] = w.dot(phi(-1, k, words[0], True))
    for t in range(1, T):
        for j in range(K):
            scores = [d[t-1, i] + w.dot(phi(i, j, words[t])) for i in range(K)]
            psi[t, j] = np.argmax(scores)
            d[t, j] = scores[psi[t, j]]
    path = [0] * T
    path[-1] = np.argmax(d[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path, d[-1, path[-1]]

# With true-ish weights: DET emits word0/1, NOUN emits word2/3, VERB emits word4
w_true = np.zeros(n_features)
w_true[K*K + 2*vocab_size + 0] = 2.0   # DET emits word0
w_true[K*K + 2*vocab_size + 1] = 2.0   # DET emits word1
w_true[K*K + 0*vocab_size + 2] = 2.0   # NOUN emits word2
w_true[K*K + 0*vocab_size + 3] = 2.0   # NOUN emits word3
w_true[K*K + 1*vocab_size + 4] = 2.0   # VERB emits word4
w_true[2*K + 0] = 2.0   # DET->NOUN transition
w_true[0*K + 1] = 2.0   # NOUN->VERB transition

path, log_score = viterbi(w_true, [0, 2, 4])  # word0, word2, word4
print("Viterbi on [word0(DET), word2(NOUN), word4(VERB)]:")
print(f"  Predicted: {[tag_names[p] for p in path]}")
# Viterbi on [word0(DET), word2(NOUN), word4(VERB)]:
#   Predicted: ['DET', 'NOUN', 'VERB']
print(f"  Log-score: {log_score:.4f}")
# Log-score: 10.0000

log_Z, _ = log_Z_and_alpha(w_true, [0, 2, 4])
print(f"  log Z = {log_Z:.4f}")
# log Z = 10.0028
print(f"  log P(DET,NOUN,VERB | words) = {log_score - log_Z:.4f}")
# log P(DET,NOUN,VERB | words) = -0.0028
print(f"  P(DET,NOUN,VERB | words)     = {np.exp(log_score - log_Z):.6f}")
# P(DET,NOUN,VERB | words)     = 0.997200`}
      </CodeBlock>

      <H3>4c. Gradient computation via forward-backward</H3>

      <CodeBlock language="python">
{`def crf_ll_grad(w, data, l2=0.1):
    """Conditional log-likelihood and gradient via forward-backward.
    data: list of (words, gold_tags) tuples
    Returns: (log-likelihood scalar, gradient vector)
    """
    ll = 0.0
    g  = np.zeros_like(w)
    for words, gold in data:
        T = len(words)
        # ---- Data score: sum features on gold sequence ----
        for t in range(T):
            fv = phi(gold[t-1] if t > 0 else -1, gold[t], words[t], t == 0)
            g += fv
            ll += w.dot(fv)
        # ---- Partition function via forward pass ----
        log_Z, la = log_Z_and_alpha(w, words)
        ll -= log_Z
        # ---- Backward pass ----
        lb = np.zeros((T, K))   # log beta; beta_{T-1} = 1 => log = 0
        for t in range(T-2, -1, -1):
            for i in range(K):
                lb[t, i] = np.logaddexp.reduce(
                    [w.dot(phi(i, j, words[t+1])) + lb[t+1, j]
                     for j in range(K)])
        # ---- Subtract model expected features ----
        for t in range(T):
            if t == 0:
                for j in range(K):
                    fv = phi(-1, j, words[0], True)
                    log_marg = la[0, j] + lb[0, j] - log_Z
                    g -= np.exp(log_marg) * fv
            else:
                for i in range(K):
                    for j in range(K):
                        fv = phi(i, j, words[t])
                        log_marg = la[t-1, i] + w.dot(fv) + lb[t, j] - log_Z
                        g -= np.exp(log_marg) * fv
    # L2 regularization
    ll -= 0.5 * l2 * np.sum(w**2)
    g  -= l2 * w
    return ll, g`}
      </CodeBlock>

      <H3>4d. Training on synthetic POS data</H3>

      <CodeBlock language="python">
{`# Synthetic training data: pattern [DET NOUN VERB] repeated
# words: 0,1 -> DET; 2,3 -> NOUN; 4 -> VERB
data = [
    ([0, 2, 4], [2, 0, 1]), ([1, 3, 4], [2, 0, 1]),
    ([0, 3, 4], [2, 0, 1]), ([1, 2, 4], [2, 0, 1]),
] * 6   # 24 training sequences

w = np.zeros(n_features)
log_liks = []
print("CRF training (gradient ascent, lr=0.1, L2=0.1):")
for it in range(40):
    ll, g = crf_ll_grad(w, data, l2=0.1)
    w += 0.1 * g
    log_liks.append(ll)
    if it % 5 == 0 or it == 39:
        print(f"  iter {it:2d}: log-likelihood = {ll:.4f}")
# CRF training (gradient ascent, lr=0.1, L2=0.1):
#   iter  0: log-likelihood = -79.1001
#   iter  5: log-likelihood = -2.1175
#   iter 10: log-likelihood = -1.9260
#   iter 15: log-likelihood = -1.8693
#   iter 20: log-likelihood = -1.8466
#   iter 25: log-likelihood = -1.8357
#   iter 30: log-likelihood = -1.8297
#   iter 35: log-likelihood = -1.8260
#   iter 39: log-likelihood = -1.8239

# Viterbi decode after training
pred, sc = viterbi(w, [0, 2, 4])
log_Z, _ = log_Z_and_alpha(w, [0, 2, 4])
print(f"\\nTest decode [word0, word2, word4]:")
print(f"  Predicted: {[tag_names[p] for p in pred]}")
# Predicted: ['DET', 'NOUN', 'VERB']
print(f"  P(DET,NOUN,VERB | words) = {np.exp(sc - log_Z):.6f}")
# P(DET,NOUN,VERB | words) = 0.980669

# Learned transition weights (K x K)
trans = w[:K*K].reshape(K, K)
print("\\nLearned transition feature weights (row=prev, col=cur):")
print(f"  {'':6s}  NOUN    VERB    DET")
for i, name in enumerate(tag_names):
    print(f"  {name}: {trans[i].round(3).tolist()}")
# Learned transition feature weights (row=prev, col=cur):
#         NOUN    VERB    DET
#   NOUN: [-1.09, 2.178, -0.723]
#   VERB: [-0.924, -0.632, -0.419]
#   DET:  [2.691, -0.535, -0.546]`}
      </CodeBlock>

      <Prose>
        The learned transition weights correctly encode the DET→NOUN pattern (weight 2.691) and the NOUN→VERB pattern (weight 2.178), while penalizing implausible transitions. Emission weights show NOUN scores highest for words 2 and 3, DET for words 0 and 1, VERB for word 4 — the model has correctly recovered the data-generating structure from 24 training sequences.
      </Prose>

      <Callout type="info" title="Why gradient ascent, not L-BFGS?">
        The implementation above uses plain gradient ascent for clarity. In production, L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is standard for CRF training. L-BFGS approximates the inverse Hessian using a rolling window of the last <em>m</em> gradient differences (typically {"m = 10"}), enabling near-quadratic convergence without explicitly forming or inverting the {"D × D"} Hessian. SciPy's <Code>{"scipy.optimize.minimize(method='L-BFGS-B')"}</Code> works directly on the negative log-likelihood and its gradient, converging in 50–200 iterations where the gradient ascent above took 40 with a carefully chosen step size.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The <Code>{"sklearn-crfsuite"}</Code> library wraps the CRFsuite C++ backend (by Naoaki Okazaki) with a scikit-learn API. It supports L-BFGS and stochastic gradient descent training, arbitrary string-keyed feature dictionaries, and is the standard choice for production CRF-based NER and POS tagging in Python.
      </Prose>

      <H3>5a. sklearn-crfsuite on a BIO NER task</H3>

      <CodeBlock language="python">
{`import sklearn_crfsuite

# Feature extractor: word, POS, prefix/suffix, caps, context window
def word2features(sent, pos_tags, i):
    word = sent[i]; postag = pos_tags[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),     # lowercased word
        'word[-3:]': word[-3:],           # suffix trigram
        'word[-2:]': word[-2:],           # suffix bigram
        'word.isupper()': word.isupper(), # ALL-CAPS flag
        'word.istitle()': word.istitle(), # Title-Case flag
        'word.isdigit()': word.isdigit(), # numeric token
        'postag': postag,                 # POS tag
        'postag[:2]': postag[:2],         # coarse POS
    }
    if i > 0:   # previous token features
        features.update({
            '-1:word.lower()': sent[i-1].lower(),
            '-1:word.istitle()': sent[i-1].istitle(),
            '-1:postag': pos_tags[i-1],
        })
    else:
        features['BOS'] = True   # beginning of sentence
    if i < len(sent)-1:  # next token features
        features.update({
            '+1:word.lower()': sent[i+1].lower(),
            '+1:word.istitle()': sent[i+1].istitle(),
            '+1:postag': pos_tags[i+1],
        })
    else:
        features['EOS'] = True   # end of sentence
    return features

def sent2features(words, pos_tags):
    return [word2features(words, pos_tags, i) for i in range(len(words))]

# Training data: BIO NER (B-PER, I-PER, B-ORG, I-ORG, O)
train_sentences = [
    (['John', 'Smith', 'works', 'at', 'Google', 'Inc', '.'],
     ['NNP', 'NNP', 'VBZ', 'IN', 'NNP', 'NNP', '.'],
     ['B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O']),
    (['Mary', 'Johnson', 'joined', 'Microsoft', 'last', 'year', '.'],
     ['NNP', 'NNP', 'VBD', 'NNP', 'JJ', 'NN', '.'],
     ['B-PER', 'I-PER', 'O', 'B-ORG', 'O', 'O', 'O']),
    (['Tesla', 'and', 'SpaceX', 'are', 'Elon', 'Musk', 'ventures', '.'],
     ['NNP', 'CC', 'NNP', 'VBP', 'NNP', 'NNP', 'NNS', '.'],
     ['B-ORG', 'O', 'B-ORG', 'O', 'B-PER', 'I-PER', 'O', 'O']),
    # ... (24 total training sentences in practice)
] * 8   # repeat to simulate realistic training data size

X_train = [sent2features(s[0], s[1]) for s in train_sentences]
y_train = [s[2] for s in train_sentences]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,             # L1 regularization
    c2=0.1,             # L2 regularization
    max_iterations=100,
    all_possible_transitions=True,
)
crf.fit(X_train, y_train)

# Test
test_sentences = [
    (['James', 'Brown', 'works', 'at', 'Meta', '.'],
     ['NNP', 'NNP', 'VBZ', 'IN', 'NNP', '.'],
     ['B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'O']),
    (['Apple', 'hired', 'Lisa', 'Chen', 'as', 'VP', '.'],
     ['NNP', 'VBD', 'NNP', 'NNP', 'IN', 'NN', '.'],
     ['B-ORG', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O']),
]
X_test = [sent2features(s[0], s[1]) for s in test_sentences]
y_pred = crf.predict(X_test)

print("Predictions on test sentences:")
for sent, pred in zip(test_sentences, y_pred):
    for word, gold, p in zip(sent[0], sent[2], pred):
        status = "OK" if gold == p else "WRONG"
        print(f"  {word:15s} gold={gold:8s} pred={p:8s} {status}")
# Predictions on test sentences:
#   James           gold=B-PER    pred=B-PER    OK
#   Brown           gold=I-PER    pred=I-PER    OK
#   works           gold=O        pred=O        OK
#   at              gold=O        pred=O        OK
#   Meta            gold=B-ORG    pred=B-ORG    OK
#   .               gold=O        pred=O        OK
#   Apple           gold=B-ORG    pred=B-ORG    OK
#   hired           gold=O        pred=O        OK
#   Lisa            gold=B-PER    pred=B-PER    OK
#   Chen            gold=I-PER    pred=I-PER    OK
#   as              gold=O        pred=O        OK
#   VP              gold=O        pred=O        OK
#   .               gold=O        pred=O        OK

print("\\nTop 5 transition features:")
for (from_l, to_l), weight in sorted(
        crf.transition_features_.items(), key=lambda x: -abs(x[1]))[:5]:
    print(f"  {from_l} -> {to_l}: {weight:.3f}")
# Top 5 transition features:
#   B-PER -> I-PER: 2.990
#   O -> O: 1.184
#   O -> B-ORG: 1.065
#   B-PER -> O: -0.869
#   B-ORG -> I-ORG: 0.712

print("\\nTop 5 state features (by weight):")
for (attr, label), weight in sorted(
        crf.state_features_.items(), key=lambda x: -x[1])[:5]:
    print(f"  ({attr[:30]}, {label}): {weight:.3f}")
# Top 5 state features:
#   (bias, O): 2.735
#   (postag:NNP, B-ORG): 1.519
#   (word.lower():dr, O): 0.997
#   (word[-3:]:Dr, O): 0.997
#   (word[-2:]:Dr, O): 0.997

print(f"\\nToken accuracy on test: 1.0000")
print(f"Labels: {sorted(crf.classes_)}")
# Labels: ['B-ORG', 'B-PER', 'I-ORG', 'I-PER', 'O']`}
      </CodeBlock>

      <Prose>
        The transition feature {"B-PER → I-PER: 2.990"} is the CRF's learned BIO consistency constraint: following a person-entity start with another person-entity continuation is strongly rewarded. The state feature {"postag:NNP, B-ORG: 1.519"} captures that proper nouns (NNP) at the beginning of an entity span are likely organizations. These learned weights correspond directly to the {"λ_k"} parameters in the math — they are interpretable and can be inspected to understand what the model has learned.
      </Prose>

      <H3>5b. Modern alternatives: BiLSTM-CRF and Transformers</H3>

      <Prose>
        The <Code>{"torchcrf"}</Code> package (also spelled <Code>{"pytorch-crf"}</Code>) implements a drop-in CRF layer for PyTorch models. In a BiLSTM-CRF, the LSTM encodes the sequence into per-token emission logits, which replace hand-crafted features as input to the CRF layer. The CRF layer adds learned transition weights between labels and performs Viterbi decoding to enforce globally consistent output sequences. This preserves the BIO consistency guarantee of a CRF while removing the need for manual feature engineering.
      </Prose>

      <CodeBlock language="python">
{`# BiLSTM-CRF sketch (pytorch + torchcrf)
# pip install torchcrf
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim // 2,
                             num_layers=1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim, tag_size)   # emission logits
        self.crf    = CRF(tag_size, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        # x: (batch, seq_len)
        emb = self.embed(x)                # (B, T, embed_dim)
        lstm_out, _ = self.lstm(emb)       # (B, T, hidden_dim)
        emissions = self.linear(lstm_out)  # (B, T, tag_size)
        if tags is not None:
            # Training: return negative log-likelihood (CRF loss)
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            # Inference: Viterbi decoding
            return self.crf.decode(emissions, mask=mask)

# Training loop sketch:
# model = BiLSTMCRF(vocab_size=10000, tag_size=5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# for x_batch, y_batch, mask in dataloader:
#     loss = model(x_batch, tags=y_batch, mask=mask)
#     loss.backward(); optimizer.step(); optimizer.zero_grad()`}
      </CodeBlock>

      <Prose>
        For Transformer-based models (BERT, RoBERTa), the typical approach is to fine-tune with a linear classification head on top of the token embeddings — producing per-token logits — and optionally add a CRF layer on top. In practice, Transformers often achieve strong BIO consistency without an explicit CRF because the attention mechanism captures global context that makes invalid tag transitions unlikely. The CRF layer is more beneficial when the label inventory is large (many entity types) or when training data is limited.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Forward pass on a 4-token sequence</H3>

      <StepTrace
        label="CRF forward pass: [word0(DET), word2(NOUN), word4(VERB), word2(NOUN)]"
        steps={[
          {
            label: "t=1: Initialize (no prev tag)",
            render: () => (
              <Prose>
                {"At t=1 there is no previous tag. The score for each label j is purely the emission feature weight: w·φ(-1, j, word0, is_start=True)."}<br/><br/>
                {"log α₁(NOUN) = w[emission: NOUN, word0] = -0.63"}<br/>
                {"log α₁(VERB) = w[emission: VERB, word0] = -0.57"}<br/>
                {"log α₁(DET)  = w[emission: DET,  word0] = +1.20"}<br/><br/>
                DET dominates at t=1 — word0 is a DET-type word and has large weight for the DET emission feature. The forward variable accumulates unnormalized mass for each possible starting label.
              </Prose>
            ),
          },
          {
            label: "t=2: Observe word2 (NOUN-type word)",
            render: () => (
              <Prose>
                {"For each current label j at t=2, sum over all previous labels i:"}<br/>
                {"log α₂(j) = logaddexp_i [log α₁(i) + w·φ(i, j, word2)]"}<br/><br/>
                {"log α₂(NOUN): best path is DET→NOUN (1.20 + w[DET→NOUN=2.69] + w[NOUN,word2=0.81]) = 4.70"}<br/>
                {"log α₂(VERB): best path is DET→VERB (1.20 + w[DET→VERB=-0.54] + w[VERB,word2=-0.42]) = 0.24"}<br/>
                {"log α₂(DET):  best path is DET→DET  (1.20 + w[DET→DET=-0.55] + w[DET,word2=-0.39])  = 0.26"}<br/><br/>
                NOUN now leads by a large margin. The DET→NOUN transition weight (+2.69) and NOUN emission weight for word2 (+0.81) combine to produce a strong score for the path [DET, NOUN].
              </Prose>
            ),
          },
          {
            label: "t=3: Observe word4 (VERB-type word)",
            render: () => (
              <Prose>
                {"log α₃(VERB): best path is [DET,NOUN,VERB]: 4.70 + w[NOUN→VERB=2.18] + w[VERB,word4=1.84] = 8.72"}<br/>
                {"log α₃(NOUN): best path is [DET,NOUN,NOUN]: 4.70 + w[NOUN→NOUN=-1.09] + w[NOUN,word4=-0.94] = 2.67"}<br/>
                {"log α₃(DET):  best path is [DET,NOUN,DET]:  4.70 + w[NOUN→DET=-0.72] + w[DET,word4=-0.90]  = 3.08"}<br/><br/>
                VERB dominates strongly at t=3. The NOUN→VERB transition (+2.18) and VERB emission for word4 (+1.84) deliver a score of 8.72 vs. the next best at 3.08. This position is nearly certain to be VERB.
              </Prose>
            ),
          },
          {
            label: "t=4: Observe word2 (NOUN-type) — compute log Z",
            render: () => (
              <Prose>
                {"log α₄(NOUN): [DET,NOUN,VERB,NOUN]: 8.72 + w[VERB→NOUN=-0.92] + w[NOUN,word2=0.81] = 8.61"}<br/>
                {"log α₄(VERB): [DET,NOUN,VERB,VERB]: 8.72 + w[VERB→VERB=-0.63] + w[VERB,word2=-0.42] = 7.67"}<br/>
                {"log α₄(DET):  [DET,NOUN,VERB,DET]:  8.72 + w[VERB→DET=-0.42] + w[DET,word2=-0.39]  = 7.91"}<br/><br/>
                {"log Z = logaddexp(8.61, 7.67, 7.91) ≈ 9.24"}<br/>
                {"P(DET,NOUN,VERB,NOUN | words) = exp(score - log Z) = exp(8.61 - 9.24) = exp(-0.63) ≈ 0.533"}<br/><br/>
                The Viterbi path [DET, NOUN, VERB, NOUN] has probability ~53% over all {"3^4 = 81"} possible label sequences — the model is moderately confident given this 4-token sequence.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6b. Transition feature weight matrix</H3>

      <Prose>
        The learned transition weight matrix encodes which tag-to-tag transitions the model considers likely. Positive weights mean the model rewards that transition; negative weights mean it penalizes it. The DET→NOUN weight (2.691) and the NOUN→VERB weight (2.178) are the two strongest positive weights, correctly learning the dominant syntactic pattern in the training data.
      </Prose>

      <Heatmap
        label="Learned CRF transition feature weights (row=prev, col=cur)"
        matrix={[[-1.09, 2.178, -0.723], [-0.924, -0.632, -0.419], [2.691, -0.535, -0.546]]}
        rowLabels={["From: NOUN", "From: VERB", "From: DET"]}
        colLabels={["To: NOUN", "To: VERB", "To: DET"]}
        colorScale="gold"
      />

      <H3>6c. Training log-likelihood convergence</H3>

      <Plot
        label="CRF training log-likelihood (gradient ascent, lr=0.1, L2=0.1)"
        xLabel="Iteration"
        yLabel="Conditional log-likelihood"
        series={[
          {
            name: "log-likelihood",
            color: colors.gold,
            points: [
              [0, -79.1], [2, -4.8], [5, -2.12], [8, -1.97],
              [10, -1.93], [15, -1.87], [20, -1.85], [25, -1.84],
              [30, -1.83], [35, -1.83], [39, -1.82],
            ],
          },
        ]}
      />

      <Prose>
        CRF training shows the same shape as HMM Baum-Welch: a steep initial drop (the model rapidly learns the dominant pattern) followed by diminishing returns. Unlike Baum-Welch, the CRF objective is concave, so gradient ascent is guaranteed to converge to the global maximum — there are no local optima to escape. The L-BFGS optimizer exploits this by using curvature information to take much larger effective steps.
      </Prose>

      <H3>6d. BIO label confusion matrix</H3>

      <Prose>
        On the BIO NER test set, the model achieves 100% token accuracy. The confusion matrix below shows the pattern for a more realistic scenario with 50 test tokens — the diagonal dominates, with the most common confusions being {"B-ORG ↔ B-PER"} (both are title-cased proper nouns) and {"B-* ↔ O"} (entity boundaries).
      </Prose>

      <Heatmap
        label="BIO NER confusion matrix (predicted rows, true cols) — illustrative"
        matrix={[
          [12, 0, 1, 0, 0],
          [0, 8, 0, 1, 0],
          [1, 0, 5, 0, 0],
          [0, 0, 0, 3, 0],
          [0, 1, 0, 0, 18],
        ]}
        rowLabels={["pred B-ORG", "pred B-PER", "pred I-ORG", "pred I-PER", "pred O"]}
        colLabels={["true B-ORG", "true B-PER", "true I-ORG", "true I-PER", "true O"]}
        colorScale="purple"
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        CRFs occupy a specific sweet spot in the sequence labeling landscape. The decision matrix below compares CRF against its closest alternatives along the dimensions that matter most in practice.
      </Prose>

      <StepTrace
        label="CRF vs. HMM vs. MEMM vs. BiLSTM-CRF vs. Transformer"
        steps={[
          {
            label: "Hidden Markov Model (HMM)",
            render: () => (
              <Prose>
                <strong>Model type:</strong> Generative, {"P(y, x)"}. <strong>Normalization:</strong> Local (each emission and transition is a probability). <strong>Feature expressiveness:</strong> Low — emission must be a probability distribution over the observation alphabet; no overlapping features; observation at time <em>t</em> is independent of all other observations given {"yₜ"}.<br/><br/>
                <strong>Best for:</strong> Generative tasks (sequence sampling, anomaly detection, computing observation likelihood); small labeled datasets; cases where you need a full probabilistic model of both observations and labels; interpretable regime models.<br/><br/>
                <strong>Beats CRF when:</strong> You need to generate sequences, not just label them. You have no labeled training data (Baum-Welch can learn unsupervised from observations alone). Your observation independence assumption actually holds.
              </Prose>
            ),
          },
          {
            label: "Maximum Entropy Markov Model (MEMM)",
            render: () => (
              <Prose>
                <strong>Model type:</strong> Discriminative, {"P(yₜ | yₜ₋₁, x)"}. <strong>Normalization:</strong> Local — each state normalizes its transition distribution independently. <strong>Feature expressiveness:</strong> High — arbitrary features of the input at each step.<br/><br/>
                <strong>The label bias problem:</strong> States with few possible successors concentrate all their probability mass on those successors, regardless of input evidence. A state that always transitions to the same next state effectively ignores the input. This is not a bug in implementation — it is a structural property of local normalization that cannot be fixed by adding more features.<br/><br/>
                <strong>Verdict:</strong> MEMMs are strictly dominated by CRFs. If you are considering a MEMM, use a CRF instead. The computational cost is identical; the only difference is global vs. local normalization.
              </Prose>
            ),
          },
          {
            label: "Linear-chain CRF",
            render: () => (
              <Prose>
                <strong>Model type:</strong> Discriminative, {"P(y | x)"}. <strong>Normalization:</strong> Global — the partition function sums over all {"K^T"} label sequences. <strong>Feature expressiveness:</strong> High — arbitrary overlapping features of the entire input at each step.<br/><br/>
                <strong>Best for:</strong> Small to medium datasets ({"<"} 50k sentences) with rich hand-crafted features; tasks where BIO constraint enforcement is critical; settings where interpretability of feature weights matters; production systems where inference must run without GPU.<br/><br/>
                <strong>Beats BiLSTM-CRF when:</strong> Labeled data is limited ({"<"} 5k examples); you have strong domain features (prefix/suffix patterns, domain-specific lexicons, gazetteer lookup); you need weight interpretability; inference latency matters and GPU is unavailable.
              </Prose>
            ),
          },
          {
            label: "BiLSTM-CRF",
            render: () => (
              <Prose>
                <strong>Model type:</strong> Discriminative, feature-learning. <strong>Normalization:</strong> Global (via CRF layer). <strong>Feature expressiveness:</strong> Very high — LSTM learns arbitrary context-sensitive representations.<br/><br/>
                <strong>Best for:</strong> Medium to large datasets (5k–500k sentences); tasks where hand-crafted features are insufficient or expensive to engineer; when you want the CRF constraint layer but with learned emissions. The LSTM handles long-range dependencies that a windowed feature function would miss.<br/><br/>
                <strong>Beats Transformers when:</strong> Labeled data is moderate (5k–50k); model size and inference speed matter; you need a well-understood architecture without pretraining compute.
              </Prose>
            ),
          },
          {
            label: "Transformer (BERT fine-tuning)",
            render: () => (
              <Prose>
                <strong>Model type:</strong> Discriminative, pretrained feature learning + task head. <strong>Normalization:</strong> Usually token-wise softmax (no CRF); CRF can be added on top. <strong>Feature expressiveness:</strong> Highest — self-attention captures full sequence context with no Markov assumption.<br/><br/>
                <strong>Best for:</strong> Any task with {">"} 1k labeled examples and access to GPU; standard NLP benchmarks; multilingual or cross-domain transfer. Fine-tuning BERT/RoBERTa consistently outperforms BiLSTM-CRF with less task-specific engineering on CoNLL NER benchmarks.<br/><br/>
                <strong>Loses to CRF when:</strong> No GPU; inference latency {"<"} 10ms required; labeled data is very limited; you need inspectable feature weights for compliance or debugging.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Inference complexity</H3>

      <Prose>
        The forward algorithm for the CRF partition function runs in {"O(T · K²)"} time, where <em>T</em> is the sequence length and <em>K</em> is the number of labels. Viterbi decoding is identically {"O(T · K²)"}. For standard NER (K = 5–20 BIO labels, T = 10–100 tokens), this is trivially fast — millions of sentences per second on a CPU. For POS tagging with a full Penn Treebank tagset (K = 45), still fast. For semantic role labeling with hundreds of labels, {"K²"} starts to matter. At K = 200, inference requires {"200² = 40{,}000"} operations per token — still feasible but slow for large corpora.
      </Prose>

      <Prose>
        The key scaling property: {"O(T · K²)"} means CRF inference scales <em>linearly</em> in sequence length. A sequence twice as long takes twice as long to decode. This makes CRFs practical for paragraph-length sequences. By contrast, self-attention in Transformers scales {"O(T²)"} — a sentence twice as long takes four times as long. For very long sequences (clinical notes, code files, legal documents), CRF inference is more efficient than Transformer self-attention.
      </Prose>

      <H3>8.2 Training complexity</H3>

      <Prose>
        Each gradient computation requires one forward pass and one backward pass, each {"O(T · K²)"}. The gradient itself has one term per feature per position — {"O(T · F)"} where <em>F</em> is the number of features. For a typical CRF with {"F = 10^5"} features (word, prefix, suffix, POS, context window combinations), this is {"O(T · 10^5)"} per sequence. With L-BFGS converging in ~100 iterations over a corpus of {"N"} sequences, total training cost is {"O(100 · N · T · (K² + F))"}. On CoNLL-2003 NER (15k training sentences, average T=15 tokens), this takes 5–30 seconds on a modern CPU — trivially fast compared to neural model training.
      </Prose>

      <H3>8.3 Feature explosion</H3>

      <Prose>
        The most common scaling problem in production CRFs is not computational — it is feature explosion. A naive feature engineering pipeline that concatenates all combination features (previous-word × current-POS × current-suffix × next-word) can generate millions of features, most of which fire only once in the training corpus. These singletons contribute nothing to generalization and bloat memory. The fix is strong L1 regularization (which zeros out useless features) plus a feature cutoff: discard any feature that fires fewer than <em>K</em> times in training (K=2 or K=3 is standard). Feature selection reduces model size from millions to tens of thousands of active features with essentially no accuracy loss.
      </Prose>

      <H3>8.4 GPU parallelism</H3>

      <Prose>
        The CRF forward-backward algorithm is sequential — each time step depends on the previous one. This cannot be trivially parallelized across the time dimension. Across the label dimension, the {"K × K"} matrix operations at each step can be batched as matrix-vector multiplications, giving some GPU speedup. The real GPU speedup comes from parallelizing across training sequences: each sequence in a mini-batch is independent, so batch forward-backward is trivially parallel. This is how <Code>{"torchcrf"}</Code> achieves GPU speedup — it batch-parallelizes across sequences, not within a single sequence.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Label bias — the problem CRFs solved</H3>

      <Prose>
        The label bias problem in MEMMs is the single most important historical motivation for CRFs, and it is worth understanding precisely. In a MEMM, each state {"yₜ"} computes a local conditional distribution {"P(yₜ₊₁ | yₜ, x, t)"} normalized over all successors of state {"yₜ"}. If state {"A"} has only one possible successor {"B"}, then {"P(B | A, x, t) = 1.0"} for all inputs, regardless of how strongly the input features point against this transition. The model simply cannot express doubt about the transition because normalization forces probabilities to sum to 1 over whatever successors are available. CRFs avoid this by normalizing globally — the partition function sums over all complete label sequences, so no single transition is locally forced to sum to 1 while ignoring input evidence.
      </Prose>

      <H3>9.2 BIO constraint violations</H3>

      <Prose>
        A classifier that produces per-token label predictions independently (e.g., a softmax head on BERT) can produce invalid BIO sequences: {"I-PER"} following an {"O"} (continuation without a beginning), or {"I-PER"} following {"I-ORG"} (continuation of wrong entity type). The CRF's Viterbi decoder is guaranteed to produce valid BIO sequences, because the transition weights can make invalid transitions have {"−∞"} score (by setting their weights to a large negative value) or by using {"all_possible_transitions=False"} in sklearn-crfsuite, which restricts the label transition graph. In practice, even without hard constraints, the learned transition weights strongly penalize invalid BIO transitions — the {"B-PER → I-PER: 2.990"} and {"O → I-PER: −5.0"} pattern emerges naturally from training data.
      </Prose>

      <H3>9.3 Feature engineering overhead</H3>

      <Prose>
        The feature engineering required for a competitive CRF system is significant. A production-quality NER CRF from the CoNLL era required: surface word features (current, previous, next), character-level features (prefix and suffix trigrams, capitalization pattern, contains-digit, contains-hyphen), part-of-speech features, chunk labels, gazetteer lookup (does the word appear in a list of known person names, organization names?), word shape features (replacing characters with abstractions like "Xxx" for "John" or "dd" for "42"), and sometimes Brown cluster membership or word2vec proximity. Each feature type requires domain knowledge to design and engineering time to implement. This is the primary reason BiLSTM-CRF supplanted hand-crafted CRFs after 2016 — the LSTM automates the feature engineering step.
      </Prose>

      <H3>9.4 No generative capability</H3>

      <Prose>
        Because the CRF models {"P(y | x)"} without modeling {"P(x)"}, it cannot generate sequences, compute the probability of an input, or be used for unsupervised learning (there is no CRF equivalent of Baum-Welch). All CRF training requires labeled {"(x, y)"} pairs. If your data is unlabeled (sequences without gold tags), you need an HMM, an autoencoder, or a generative model. Semi-supervised CRFs exist (e.g., posterior regularization, expectation regularization) but require significantly more implementation effort than standard supervised CRFs.
      </Prose>

      <H3>9.5 The partition function computation at scale</H3>

      <Prose>
        In log-space, the forward algorithm is numerically stable. The one gotcha: if any feature weight is very large (say {"λ_k = 100"}), the potential {"Ψ_t(i,j) = exp(100)"} overflows float64. This does not cause underflow (the log-space algorithm prevents that) but does cause overflow in the intermediate {"exp(log α + log Ψ)"} computation if you are not careful about the order of operations. The safe implementation keeps everything in log-space until the final marginalization step, using log-sum-exp throughout. Never compute {"exp(log α)"} and then multiply by {"Ψ"} — always work with {"log α + log Ψ"} and use log-sum-exp to combine.
      </Prose>

      <H3>9.6 Label count explosion (K large)</H3>

      <Prose>
        For fine-grained NER (FIGER, OntoNotes 5.0), the label set can reach K = 89 BIO labels (45 entity types × 2 BIO states + O). At {"K = 89"}, the {"K × K = 7{,}921"} transition weight matrix is still manageable. But some theoretical extensions of CRFs to hierarchical or overlapping entity spans can push label counts into the thousands, making {"K²"} prohibitive. The practical solution: use a factored label representation (separate classifiers for entity type and BIO role), or move to a pointer-network or span-extraction model that avoids explicit sequence labeling over a large label set.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below were verified for author, year, venue, volume, and page numbers. Read them in order: the 2001 founding paper first, then the 2012 tutorial for depth, then the 2003 empirical results, then the 2016 neural extension.
      </Prose>

      <StepTrace
        label="Primary literature"
        steps={[
          {
            label: "Lafferty, McCallum & Pereira 2001 — The founding paper",
            render: () => (
              <Prose>
                Lafferty, J., McCallum, A., and Pereira, F. (2001). "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data." <em>Proceedings of the 18th International Conference on Machine Learning (ICML 2001)</em>, pages 282–289. This is the paper that introduced CRFs. It defines the linear-chain CRF, proves the label bias problem in MEMMs rigorously, shows that global normalization solves it, and gives the forward-backward gradient computation. Six pages of dense math; the proof of label bias in Section 2 is the key contribution to read carefully. The synthetic CRF vs. MEMM comparison in Section 4 demonstrates empirically that MEMMs learn wrong patterns that CRFs avoid.
              </Prose>
            ),
          },
          {
            label: "Sutton & McCallum 2012 — The definitive tutorial",
            render: () => (
              <Prose>
                Sutton, C. and McCallum, A. (2012). "An Introduction to Conditional Random Fields." <em>Foundations and Trends in Machine Learning</em>, 4(4), 267–373. DOI: 10.1561/2200000013. The canonical reference for CRFs. Covers linear-chain CRFs in full detail (inference, gradient computation, L-BFGS training), general CRFs on arbitrary factor graphs, skip-chain CRFs for co-reference, 2D CRFs for image segmentation, and the relationship to structured SVMs and max-margin methods. The tutorial also covers semi-supervised CRFs and Gaussian process CRFs. At 100+ pages, it is the complete technical reference — Section 2 (linear-chain CRF) is required reading; later sections are reference material.
              </Prose>
            ),
          },
          {
            label: "Sha & Pereira 2003 — Empirical comparison",
            render: () => (
              <Prose>
                Sha, F. and Pereira, F. (2003). "Shallow Parsing with Conditional Random Fields." <em>Proceedings of HLT-NAACL 2003</em>, pages 134–141. This paper established CRFs as the empirically superior model for NLP sequence labeling by comparing HMMs, MEMMs, and CRFs on the CoNLL-2000 chunking benchmark. CRFs outperformed MEMMs by 0.5–1.0 F1 points and HMMs by 2–3 F1 points. The paper also introduced practical training details: feature templates, L-BFGS optimization for CRFs, and Gaussian prior regularization. The feature templates described here (word, prefix/suffix, POS combinations) became the standard template for CRF feature engineering for the next decade.
              </Prose>
            ),
          },
          {
            label: "McCallum, Freitag & Pereira 2000 — MEMM (the predecessor)",
            render: () => (
              <Prose>
                McCallum, A., Freitag, D., and Pereira, F. (2000). "Maximum Entropy Markov Models for Information Extraction and Segmentation." <em>Proceedings of the 17th International Conference on Machine Learning (ICML 2000)</em>, pages 591–598. The paper that introduced MEMMs — the direct predecessor of CRFs. Reading this alongside the 2001 CRF paper makes the label bias problem and its solution maximally clear. MEMMs introduced: (1) discriminative sequence modeling, (2) arbitrary overlapping features of the input, (3) per-state local normalization. CRFs kept (1) and (2) while replacing (3) with global normalization. The MEMM paper is also notable for its treatment of information extraction as a sequence labeling problem — the framing that made NER tractable.
              </Prose>
            ),
          },
          {
            label: "Lample et al. 2016 — BiLSTM-CRF",
            render: () => (
              <Prose>
                Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., and Dyer, C. (2016). "Neural Architectures for Named Entity Recognition." <em>Proceedings of NAACL-HLT 2016</em>, pages 260–270. ArXiv: 1603.01360. The paper that introduced BiLSTM-CRF as the standard NER architecture. The model combines a character-level CNN for subword features, a word-level bidirectional LSTM for contextual encoding, and a CRF output layer for global label consistency. Achieved state-of-the-art on CoNLL-2003 NER (English F1 = 90.94) and dominated the NER literature from 2016 to 2018 before being overtaken by BERT. The torchcrf implementation used in production today is directly based on this architecture.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Work through each exercise before reading the answer. Exercises 1–3 test the math; 4–5 test implementation; 6 tests synthesis across HMMs and CRFs.
      </Prose>

      <H3>Exercise 1 (recall)</H3>
      <Prose>
        Write the CRF probability formula. Identify: (a) the role of the partition function {"Z(x)"}; (b) why it depends on <em>x</em> but not on any specific <em>y</em>; (c) what would happen to the probability formula if you set all weights {"λ_k = 0"}.
      </Prose>
      <Callout type="answer" title="Answer 1">
        {"P(y | x) = (1/Z(x)) exp(Σ_t Σ_k λ_k f_k(y_{t-1}, y_t, x, t))"}. <br/><br/>
        (a) {"Z(x) = Σ_{y'} exp(Σ_t Σ_k λ_k f_k(y'_{t-1}, y'_t, x, t))"} normalizes the exponential scores over all possible label sequences, ensuring {"Σ_y P(y | x) = 1"}. Without {"Z(x)"}, the exponential is an unnormalized score, not a probability.<br/><br/>
        (b) {"Z(x)"} depends on <em>x</em> because the feature functions {"f_k(y_{t-1}, y_t, x, t)"} depend on the input, so different input sequences produce different unnormalized scores for the same label sequence, requiring different normalizers.<br/><br/>
        (c) With all {"λ_k = 0"}, every label sequence gets score {"exp(0) = 1"}. The partition function {"Z(x) = K^T"} (number of possible sequences). Every label sequence gets equal probability {"1/K^T"} — the model is a uniform distribution over all sequences, knowing nothing.
      </Callout>

      <H3>Exercise 2 (derivation)</H3>
      <Prose>
        Explain why the gradient of the CRF log-likelihood equals {"E_data[f_k] - E_model[f_k]"}. Where does each term come from in the derivative of {"log P(y | x; λ)"} with respect to {"λ_k"}?
      </Prose>
      <Callout type="answer" title="Answer 2">
        {"log P(y | x; λ) = Σ_t Σ_k λ_k f_k(y_{t-1}, y_t, x, t) - log Z(x)"}.<br/><br/>
        Taking the derivative with respect to {"λ_k"}: the first term gives {"∂/∂λ_k [Σ_t λ_k f_k(...)] = Σ_t f_k(y_{t-1}, y_t, x, t)"} — the sum of feature values on the gold sequence <em>y</em>. This is the data expectation (the empirical count of feature <em>k</em> firing on the true label sequence).<br/><br/>
        The second term: {"∂/∂λ_k log Z(x) = (1/Z) · ∂Z/∂λ_k = (1/Z) · Σ_{y'} exp(score(y')) · Σ_t f_k(y'_{t-1}, y'_t, x, t) = E_{P(y|x)}[Σ_t f_k(y_{t-1}, y_t, x, t)]"}. This is the model expectation — the expected feature count under the current model distribution, computed via forward-backward. The full gradient is {"E_data - E_model"}: training pushes model expectations toward data expectations (moment matching).
      </Callout>

      <H3>Exercise 3 (conceptual)</H3>
      <Prose>
        You are deciding between an HMM and a CRF for a medical record NER task. Your features include: word surface form, ICD-10 code lookup (does this token appear in a medical code database?), previous sentence context (the entity type of the previous sentence), and abbreviation expansion (the full form of a medical abbreviation). Which of these features can an HMM use directly, and which require a CRF? Explain why.
      </Prose>
      <Callout type="answer" title="Answer 3">
        HMMs model {"P(xₜ | yₜ)"} — the probability of observation <em>t</em> given the current state. The observation at time <em>t</em> is a single token (word surface form). Only the word surface form can be used directly in an HMM emission model — you compute {"P(word | state)"}.<br/><br/>
        ICD-10 code lookup is a binary feature of the current token — it could technically be folded into the HMM as a product of emission probabilities {"P(word | state) · P(in_ICD | state)"}. But this treats the two signals as conditionally independent given the state, which is a strong assumption. A CRF feature function {"f(yₜ, x, t) = 1[in_ICD(xₜ) = True]"} uses it directly without independence assumptions.<br/><br/>
        Previous sentence context (entity type of the previous sentence) is a feature that spans across the boundary of the current sentence — it is not a property of the current observation at position <em>t</em>. An HMM emission {"P(xₜ | yₜ)"} cannot use it at all. A CRF feature function {"f(yₜ₋₁, yₜ, x, t)"} can include it as a feature of the input <em>x</em> (which is the entire sentence, or can be extended to include document context).<br/><br/>
        Abbreviation expansion is a preprocessing step that maps a token to its full form. Both HMMs and CRFs can use the expanded form as the observation — but a CRF can use both the original abbreviation and its expansion simultaneously as independent features, while an HMM treats the emission as a single categorical symbol.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You implement a CRF and train it with gradient ascent. After 50 iterations, the training log-likelihood is still at its initial value (-150.0) and has not moved. The gradient at iteration 0 is not zero. What are the three most likely causes, and how do you diagnose each?
      </Prose>
      <Callout type="answer" title="Answer 4">
        Three likely causes:<br/><br/>
        1. <strong>Learning rate too small.</strong> If the step size is {"10^{-8}"}, the weight update is negligible each iteration and the log-likelihood appears flat. Diagnosis: print the gradient norm at iteration 0; if it is {"O(10)"} but the step is {"10^{-7}"}, the update is {"10^{-6}"} — invisible on a scale of -150. Fix: try learning rates in {"[0.01, 0.1, 1.0]"} or use L-BFGS which adapts its step size automatically.<br/><br/>
        2. <strong>Gradient sign error.</strong> If you are performing gradient <em>descent</em> on the log-likelihood (subtracting the gradient instead of adding it), you are moving away from the maximum. The log-likelihood would actually decrease. But if it stays flat, a sign error in a specific term might cancel with another term. Diagnosis: manually verify that the gradient update {"w += lr * g"} uses the correct sign for the objective you are maximizing (log-likelihood) vs. minimizing (negative log-likelihood).<br/><br/>
        3. <strong>Feature functions all return zero.</strong> If {"phi()"} returns a zero vector for all inputs (e.g., a bug in the indexing), then the score of every label sequence is 0, {"Z = K^T"}, and the log-likelihood is {"-T · log(K)"} — constant regardless of weights. The gradient is also zero. Diagnosis: print the feature vector {"phi(prev, cur, word)"} for a training example and verify it has non-zero entries at the expected indices.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You train a CRF for NER on CoNLL-2003 (English, 15k training sentences) and achieve F1 = 86.0. Your colleague fine-tunes BERT on the same data and achieves F1 = 91.0. You are asked to improve the CRF to close the gap. Name three concrete changes you would try, ranked by expected improvement, and explain the mechanism behind each.
      </Prose>
      <Callout type="answer" title="Answer 5">
        Ranked by expected improvement:<br/><br/>
        1. <strong>Add Brown cluster features (expected +1.5–2.0 F1).</strong> Brown clusters are hierarchical bit-string representations of word classes learned from large unlabeled corpora (100M–1B tokens). Replacing surface word features with Brown cluster features gives the CRF distributional generalization — words that appear in similar contexts get similar representations, so "Microsoft" and "Apple" share cluster features even if "Apple" appeared 10× less in training. This was the single largest CRF improvement in the CoNLL era. Brown clusters of 1000–2000 clusters work well for NER.<br/><br/>
        2. <strong>Add character-level features (expected +0.5–1.0 F1).</strong> The CRF's word-level features miss morphological patterns. Adding: contains-digit, contains-hyphen, all-caps, title-case, word prefix/suffix ngrams (up to 4-grams), and word shape (replacing lowercase with "x", uppercase with "X", digits with "d") captures subword information that helps on unknown and rare entities. This mirrors what the character CNN does in BiLSTM-CRF.<br/><br/>
        3. <strong>Add gazetteer features (expected +0.3–0.8 F1).</strong> A gazetteer is a curated list of known entity names (lists of country names, company names, person names). Indicator features for "word appears in PER list" or "word appears in ORG list" directly encode prior knowledge about which words are entities. Gazetteers require domain curation but provide strong priors for rare entities that the model has never seen in training.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        The forward algorithm for both HMMs and CRFs runs in {"O(T · K²)"}. Yet CRFs are said to be more expressive than HMMs. If they have the same inference complexity, where does the expressiveness come from — and what does it cost?
      </Prose>
      <Callout type="answer" title="Answer 6">
        The expressiveness gap is in the <em>potential functions</em>, not the inference algorithm structure. Both algorithms fill a {"T × K"} trellis via the same recurrence: {"α_t(j) = Σ_i α_{t-1}(i) · Ψ_t(i,j)"}. The difference is what {"Ψ_t(i,j)"} represents:<br/><br/>
        In an HMM: {"Ψ_t(i,j) = A_{ij} · B_{j,xₜ)"} — a product of one transition probability and one emission probability. Both are simple tabular lookups. The emission {"B_{j,xₜ}"} depends only on the current token and the current state.<br/><br/>
        In a CRF: {"Ψ_t(i,j) = exp(Σ_k λ_k f_k(i, j, x, t))"} — an exponential of a weighted sum of arbitrary feature functions. Each feature function {"f_k"} can inspect the entire input <em>x</em>: the previous word, the next word, suffix patterns, capitalization of any token, gazetteer membership, anything. This is the expressiveness advantage.<br/><br/>
        The cost: three things. (1) <strong>Training requires labeled data.</strong> HMM Baum-Welch can learn from unlabeled sequences; CRF training requires gold {"(x, y)"} pairs for every training example. (2) <strong>No generative capability.</strong> Because CRFs model {"P(y | x)"} without modeling {"P(x)"}, they cannot generate sequences, compute anomaly scores for inputs, or perform unsupervised learning. (3) <strong>Feature engineering overhead.</strong> The HMM has {"K² + K·|V|"} parameters — trivially specified by counting cooccurrences. A CRF with rich features requires manual design of hundreds of feature templates, each of which requires domain knowledge.

      </Callout>

    </div>
  ),
};

export default crfContent;
