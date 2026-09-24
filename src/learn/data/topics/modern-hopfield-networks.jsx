import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const modernHopfieldContent = {
  title: "Modern Hopfield Networks",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In April 1982, John Hopfield published a four-page paper in PNAS that did something quite radical for neuroscience-flavored physics: it tied a recurrent network of binary neurons to the statistical mechanics of spin glasses, defined an explicit Lyapunov energy function for the dynamics, and proved that the network would converge to a local minimum of that energy under asynchronous updates. The paper was titled "Neural networks and physical systems with emergent collective computational abilities," and the computation it claimed to support was associative memory: store a set of patterns, present a noisy or partial cue, and the network's dynamics relax to the stored pattern closest to the cue. This was a content-addressable memory implemented entirely in continuous-time recurrent dynamics, with no labels, no addresses, and no explicit search procedure.
      </Prose>

      <Prose>
        The Hopfield network rapidly became the textbook example of associative memory and the canonical entry point into recurrent networks for a generation of students. It also acquired a famous limitation. Hopfield's own analysis, refined over the next several years by Amit, Gutfreund, and Sompolinsky in their 1985 spin-glass formulation and later by McEliece, Posner, Rodemich, and Venkatesh in their 1987 information-theoretic treatment, established that a network of <Code>N</Code> binary neurons could reliably store at most about <Code>{"0.138 N"}</Code> random patterns as stable fixed points before catastrophic interference set in and stored patterns merged or shattered into spurious mixtures. That ratio — capacity divided by dimension — was a hard ceiling. Doubling the network bought you only a linear increase in storage. Hopfield himself extended the formalism to graded (continuous-valued) neurons in 1984 and showed the energy argument carried over, but the linear-in-dim capacity did not improve.
      </Prose>

      <Prose>
        The classical Hopfield network sat in textbooks for three decades while modern deep learning happened around it. Then in 2016, Dmitry Krotov and John Hopfield published "Dense Associative Memory for Pattern Recognition" at NeurIPS (arXiv:1606.01164), and the picture changed. Their construction kept the energy-minimization framing but replaced the quadratic interaction in the energy function with a higher-order polynomial or rectified power. The resulting "dense" associative memories had a capacity that scaled <em>polynomially</em> in the dimension <Code>N</Code> for polynomial energies of degree <Code>n</Code> (roughly <Code>{"N^{n-1}"}</Code>), or even <em>exponentially</em> when the interaction order was taken to infinity. Suddenly the linear-capacity ceiling was an artifact of the quadratic energy choice — not a fundamental property of associative memory itself.
      </Prose>

      <Prose>
        The 2020 follow-up by Hubert Ramsauer and collaborators at JKU Linz, "Hopfield Networks is All You Need" (arXiv:2008.02217, ICLR 2021), pushed the construction over the line that mattered for deep learning. Their version replaced binary neurons with continuous-valued state vectors, replaced the discrete energy function with a smooth log-sum-exp construction, and derived a one-step update rule that converges to the basin's minimum almost everywhere. The capacity they proved scales <Code>{"\\sim \\exp(N/2)"}</Code> for random patterns under reasonable separation assumptions. And the central result of the paper, the one that gave it the cheeky title, was an algebraic identity: the modern Hopfield update rule is exactly the attention mechanism of Vaswani et al.'s 2017 transformer paper. With the stored patterns as keys <em>and</em> values, the cue as the query, and the inverse temperature <Code>{"\\beta = 1 / \\sqrt{d}"}</Code>, scaled dot-product attention <em>is</em> a one-step Hopfield retrieval.
      </Prose>

      <Prose>
        That identity reframed several things at once. Attention had been described in transformer papers as "a learned soft lookup over a context," with intuitive but somewhat ad-hoc justification. Modern Hopfield gave it a precise associative-memory interpretation: the keys are stored memory patterns, the query is a probe, and the softmax-weighted retrieval is the energy-minimizing readout from a dense associative memory. The ~exponential capacity bound from the Hopfield analysis became the capacity bound for attention's "memory" — not in the sense that any one attention head can memorize <Code>{"\\exp(N)"}</Code> distinct things, but in the sense that the retrieval mechanism is theoretically capable of separating that many patterns when they are well-spread in the latent space. This was useful for theorists; it made transformers slightly less mysterious.
      </Prose>

      <Prose>
        The practical consequence has been more modest than the title suggested. Pure modern Hopfield modules — i.e. networks built around a Hopfield layer rather than around standard attention — have shown up in a handful of niche settings: immune repertoire classification (Widrich et al., NeurIPS 2020, arXiv:2007.13505), tabular learning (Schäfl et al., "Hopular," arXiv:2206.00664), drug-target affinity, and a few specialized memory-augmented architectures. They have not displaced standard transformer attention in any large-scale system, for the simple reason that they <em>are</em> standard transformer attention, with a different name and a slightly different parameterization. The lasting contributions are the conceptual clarity of "attention is associative memory," the capacity bounds that fall out of the analysis, and a small but useful library of pattern-retrieval primitives that researchers reach for when they want to think of attention as a memory access pattern.
      </Prose>

      <Callout accent="gold">
        Modern Hopfield networks are best understood not as a competitor to attention but as a re-interpretation: a derivation of the attention mechanism from the energy-minimization principles of associative memory, with explicit capacity guarantees. Knowing this reframing is valuable for theory, for biological analogy, and for understanding why attention works as a memory access mechanism. For deployment, the practical artifact you reach for is still <Code>nn.MultiheadAttention</Code>.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Associative memory: store, then retrieve from a cue</H3>

      <Prose>
        The central object of any Hopfield network — classical or modern — is a memory bank holding a set of stored patterns <Code>{"\\xi_1, \\xi_2, \\dots, \\xi_P"}</Code>, each a vector in some space (binary <Code>{"\\{-1, +1\\}^N"}</Code> for classical Hopfield, real-valued <Code>{"\\mathbb{R}^d"}</Code> for modern). Retrieval starts with a cue <Code>x</Code>, often a noisy, partial, or corrupted version of one of the stored patterns. The network's job is to map <Code>x</Code> to the closest stored pattern, where "closest" is defined by an energy landscape designed so that the stored patterns sit at local minima.
      </Prose>

      <Prose>
        This is fundamentally different from a feedforward classifier. A classifier learns a discriminative map from inputs to labels and the labels are the output. An associative memory learns to <em>regenerate</em> stored items from incomplete cues; the output lives in the same space as the input. A photo of a face with the eyes blurred should retrieve the same face fully rendered, not a class label "person." A protein sequence with a few residues masked should retrieve the consensus sequence from the memory of homologs, not a binary "is-a-protein" decision. This distinction is what makes the Hopfield framing useful even when the actual computation reduces to attention.
      </Prose>

      <H3>2.2 Classical Hopfield: discrete, energy-based, linear capacity</H3>

      <Prose>
        The classical 1982 network has <Code>N</Code> binary neurons with state <Code>{"x_i \\in \\{-1, +1\\}"}</Code>, a symmetric weight matrix <Code>W</Code> with zero diagonal, and asynchronous (or synchronous) sign-update dynamics: <Code>{"x_i \\leftarrow \\text{sign}(\\sum_j W_{ij} x_j)"}</Code>. The Hebbian storage rule sets <Code>{"W = (1/N) \\sum_\\mu \\xi_\\mu \\xi_\\mu^T"}</Code> with the diagonal zeroed. The energy <Code>{"E(x) = -(1/2) x^T W x"}</Code> is a Lyapunov function: each asynchronous update strictly decreases <Code>E</Code> until a fixed point is reached. With <Code>P</Code> stored random patterns, the energy landscape carves out <Code>P</Code> attractors plus an exponential number of spurious ones (mixture states); when <Code>P/N</Code> exceeds <Code>{"\\sim 0.138"}</Code> the spurious states swallow the genuine ones and the memory fails catastrophically. We verify this empirically in section 4.
      </Prose>

      <H3>2.3 Modern Hopfield: continuous, softmax, exponential capacity</H3>

      <Prose>
        The modern Hopfield network of Ramsauer et al. (2020) replaces the quadratic energy with a log-sum-exp:
      </Prose>

      <MathBlock>{"E(x) = -\\frac{1}{\\beta} \\log \\sum_{\\mu=1}^P \\exp(\\beta \\, x \\cdot \\xi_\\mu) + \\frac{1}{2} \\|x\\|^2 + \\text{const}"}</MathBlock>

      <Prose>
        The log-sum-exp term is convex and has steep wells at each stored pattern. The quadratic <Code>{"(1/2)\\|x\\|^2"}</Code> ensures the state space stays bounded. The inverse temperature <Code>{"\\beta"}</Code> controls how sharp the wells are: large <Code>{"\\beta"}</Code> gives nearly-disjoint basins; small <Code>{"\\beta"}</Code> blends nearby memories. Setting <Code>{"\\nabla_x E = 0"}</Code> and solving yields the update rule:
      </Prose>

      <MathBlock>{"x_{\\text{new}} = X^T \\, \\text{softmax}(\\beta \\, X \\, x)"}</MathBlock>

      <Prose>
        where <Code>X</Code> is the <Code>{"P \\times d"}</Code> matrix whose rows are stored patterns. The retrieved <Code>{"x_{\\text{new}}"}</Code> is a convex combination of stored patterns, weighted by their similarity to the cue. When the cue is close to a single stored pattern and <Code>{"\\beta"}</Code> is large, the softmax saturates and <Code>{"x_{\\text{new}}"}</Code> is essentially that one pattern. When the cue is ambiguous between two stored patterns, the retrieval is a weighted mixture — which can be a useful behavior or a failure mode depending on what you wanted.
      </Prose>

      <H3>2.4 The attention identity</H3>

      <Prose>
        Compare the modern Hopfield update with scaled dot-product attention. Attention takes queries <Code>Q</Code>, keys <Code>K</Code>, values <Code>V</Code>, and computes <Code>{"\\text{softmax}(QK^T / \\sqrt{d}) V"}</Code>. Set <Code>{"K = V = X"}</Code> (the stored pattern matrix), set <Code>{"Q = x"}</Code> (the cue, viewed as a row), and set <Code>{"\\beta = 1/\\sqrt{d}"}</Code>. Then
      </Prose>

      <MathBlock>{"\\text{softmax}\\!\\left(\\frac{x K^T}{\\sqrt{d}}\\right) V \\;=\\; \\text{softmax}(\\beta \\, x \\, X^T) \\, X \\;=\\; X^T \\, \\text{softmax}(\\beta \\, X \\, x)"}</MathBlock>

      <Prose>
        which is the modern Hopfield retrieval. The transformer's attention block is, mechanically, a Hopfield retrieval where the keys and values happen to come from the same source (self-attention) or from a different source (cross-attention to a memory). Multi-head attention is <Code>H</Code> independent Hopfield retrievals concatenated. The QKV projections add a learned re-parameterization on top, but the soft-readout pattern is unchanged.
      </Prose>

      <H3>2.5 Why this identity matters</H3>

      <Prose>
        The identity is not just a name change. It does several things. First, it gives attention a concrete capacity argument: the number of well-separated memories a softmax retrieval can disambiguate scales exponentially in the embedding dimension under standard random-vector assumptions. Second, it explains why attention is so good at "look up the right past token": that is exactly the operation an associative memory was designed to do. Third, it suggests that some properties of associative memories — robustness to partial cues, behavior under noise, the basin structure around stored patterns — should carry over to attention, which they largely do. And fourth, it provides a framework for thinking about memory-augmented networks: the memory is just keys and values, and accessing it is just attention.
      </Prose>

      <Callout accent="gold">
        A useful mental check: every time you write <Code>nn.MultiheadAttention</Code>, picture the keys as a memory bank of stored patterns and the query as a noisy cue. The output is the closest stored pattern (or a mix of close ones, weighted by inverse-temperature <Code>{"\\beta = 1/\\sqrt{d}"}</Code>). This is not a metaphor; it is the literal Ramsauer-Hopfield update.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Classical Hopfield: Hebbian rule and Lyapunov function</H3>

      <Prose>
        Given <Code>P</Code> binary patterns <Code>{"\\xi_\\mu \\in \\{-1, +1\\}^N"}</Code>, the Hebbian storage rule constructs the weight matrix:
      </Prose>

      <MathBlock>{"W_{ij} = \\frac{1}{N} \\sum_{\\mu=1}^P \\xi_\\mu^i \\xi_\\mu^j, \\qquad W_{ii} = 0"}</MathBlock>

      <Prose>
        The energy of state <Code>x</Code> is:
      </Prose>

      <MathBlock>{"E(x) = -\\tfrac{1}{2} \\, x^T W x"}</MathBlock>

      <Prose>
        Asynchronous update flips one neuron at a time: <Code>{"x_i \\to \\text{sign}(\\sum_j W_{ij} x_j)"}</Code>. Each flip that changes the state must strictly decrease <Code>E</Code> (Hopfield's main 1982 result), so the dynamics converge to a local minimum in finite time. Synchronous updates flip all neurons at once and may oscillate between two states, but symmetric <Code>W</Code> with zero diagonal still admits an energy argument with at most a 2-cycle.
      </Prose>

      <H3>3.2 Classical capacity: the 0.138 bound</H3>

      <Prose>
        Amit, Gutfreund, and Sompolinsky (1985) used spin-glass replica techniques to compute the maximum <Code>P/N</Code> at which random stored patterns remain global minima of <Code>E</Code>. The famous result:
      </Prose>

      <MathBlock>{"\\alpha_c = \\frac{P_{\\max}}{N} \\approx 0.138"}</MathBlock>

      <Prose>
        Above this ratio, the recall error rate jumps discontinuously from near zero to near 0.5 (random). McEliece, Posner, Rodemich, and Venkatesh (1987) gave an information-theoretic bound of <Code>{"P \\le N / (4 \\ln N)"}</Code> for perfect recall of all stored patterns simultaneously, which is tighter and more pessimistic than the 0.138 figure. Both numbers express the same fundamental limit: <em>linear in the dimension</em>. Section 4.3 reproduces the empirical capacity curve.
      </Prose>

      <H3>3.3 Modern Hopfield energy and update</H3>

      <Prose>
        Define the stored-pattern matrix <Code>{"X \\in \\mathbb{R}^{P \\times d}"}</Code> with rows <Code>{"\\xi_\\mu^T"}</Code>. The modern Hopfield energy from Ramsauer et al. is:
      </Prose>

      <MathBlock>{"E(x) = -\\frac{1}{\\beta} \\, \\text{lse}(\\beta \\, X \\, x) + \\tfrac{1}{2} \\, x^T x + \\tfrac{1}{\\beta} \\log P + \\tfrac{1}{2} M^2"}</MathBlock>

      <Prose>
        where <Code>{"\\text{lse}(z) = \\log \\sum_\\mu \\exp(z_\\mu)"}</Code> and <Code>{"M = \\max_\\mu \\|\\xi_\\mu\\|"}</Code>. The constants do not affect the dynamics but make the bounds in the proof cleaner. The gradient is:
      </Prose>

      <MathBlock>{"\\nabla_x E(x) = -X^T \\, \\text{softmax}(\\beta \\, X \\, x) + x"}</MathBlock>

      <Prose>
        Setting <Code>{"\\nabla_x E = 0"}</Code> gives the fixed-point equation <Code>{"x = X^T \\text{softmax}(\\beta \\, X \\, x)"}</Code>. The Ramsauer paper's Theorem 4 shows that the update rule
      </Prose>

      <MathBlock>{"x^{(t+1)} = X^T \\, \\text{softmax}(\\beta \\, X \\, x^{(t)})"}</MathBlock>

      <Prose>
        is a concave-convex procedure that converges in one step (when starting near a stored pattern) or geometrically fast (when starting from anywhere in the basin). "Converges in one step" means the network is essentially feedforward — exactly the property that lets attention mimic it without any iteration.
      </Prose>

      <H3>3.4 Capacity bound for modern Hopfield</H3>

      <Prose>
        Ramsauer et al. (2020), Theorem 3, gives the central capacity result. For stored patterns drawn iid from a distribution on the sphere of radius <Code>M</Code> in <Code>{"\\mathbb{R}^d"}</Code>, the maximum number of patterns that remain stable fixed points of the modern Hopfield update with probability <Code>{"1 - p"}</Code> is:
      </Prose>

      <MathBlock>{"P_{\\max} \\;\\ge\\; \\frac{1}{4} \\sqrt{p^{-1}} \\, c^{(d-1)/4}"}</MathBlock>

      <Prose>
        where <Code>c</Code> is a constant depending on the separation requirement, typically taken to give <Code>{"P_{\\max} \\sim \\exp(d/2)"}</Code>. The blow-up is exponential in the embedding dimension. For <Code>{"d = 64"}</Code>, the bound says you can store <Code>{"\\sim 10^{14}"}</Code> patterns before retrieval starts to fail — a number large enough that, in practice, capacity is not the binding constraint on a Hopfield-style memory. (The binding constraint is usually how well the patterns separate from each other in the embedding space, which depends on how they are constructed.)
      </Prose>

      <H3>3.5 The attention identity in equations</H3>

      <Prose>
        Scaled dot-product attention is:
      </Prose>

      <MathBlock>{"\\text{Attn}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V"}</MathBlock>

      <Prose>
        For a single query row <Code>q</Code>, with keys and values both equal to a stored pattern matrix <Code>X</Code>:
      </Prose>

      <MathBlock>{"\\text{Attn}(q, X, X) = \\text{softmax}\\!\\left(\\frac{q X^T}{\\sqrt{d}}\\right) X = X^T \\, \\text{softmax}\\!\\left(\\frac{X q}{\\sqrt{d}}\\right)"}</MathBlock>

      <Prose>
        which is the modern Hopfield update with <Code>{"\\beta = 1/\\sqrt{d}"}</Code>. In a transformer, queries, keys, and values are produced by learned linear projections of the input <Code>{"H \\in \\mathbb{R}^{L \\times d_{\\text{model}}}"}</Code>:
      </Prose>

      <MathBlock>{"Q = H W_Q, \\quad K = H W_K, \\quad V = H W_V"}</MathBlock>

      <Prose>
        Self-attention then performs <Code>L</Code> Hopfield retrievals in parallel — one per token — each using the entire sequence as its memory bank. Cross-attention uses the encoder output as <Code>K</Code> and <Code>V</Code> and the decoder hidden states as <Code>Q</Code> — Hopfield retrieval against an external memory. A "Hopfield layer" in the Ramsauer software library is exactly this construction with the option to make <Code>X</Code> learnable parameters rather than activations from another layer.
      </Prose>

      <H3>3.6 Higher-order interactions: dense associative memory</H3>

      <Prose>
        Krotov and Hopfield (2016) introduced the Dense Associative Memory by replacing the quadratic energy with a polynomial of degree <Code>n</Code>:
      </Prose>

      <MathBlock>{"E(x) = -\\sum_{\\mu=1}^P F\\!\\left(\\xi_\\mu \\cdot x\\right), \\qquad F(z) = z^n"}</MathBlock>

      <Prose>
        For <Code>{"n = 2"}</Code> this reduces (up to additive constant) to the classical Hopfield energy. For <Code>{"n \\ge 3"}</Code> the capacity grows as <Code>{"P_{\\max} \\sim N^{n-1}"}</Code>. Taking <Code>{"F(z) = \\exp(z)"}</Code> and pushing the limit gives exponential capacity, which is essentially what the Ramsauer construction does in continuous form. The polynomial construction stays in the binary <Code>{"\\{-1, +1\\}^N"}</Code> world but with a different energy; the Ramsauer construction goes continuous and uses the smooth log-sum-exp.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was executed against PyTorch 2.6 + CUDA. The <Code>{"# Output:"}</Code> comments are the real stdout. We build classical Hopfield first (binary, sign updates), measure its capacity and confirm the 0.138 boundary, then build modern Hopfield (continuous, softmax retrieval), prove its equivalence to scaled dot-product attention numerically, and demonstrate exponential storage capacity.
      </Prose>

      <H3>4.1 Classical Hopfield: storage and synchronous recall</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def classical_store(patterns):
    """Hebbian rule: W = (1/N) sum_mu xi_mu xi_mu^T, zero diagonal."""
    N = patterns.shape[1]
    W = patterns.T @ patterns / N
    W.fill_diagonal_(0.0)
    return W

def classical_recall(W, x, steps=20):
    """Synchronous sign update until fixed point or step limit."""
    x = x.clone()
    for _ in range(steps):
        x_new = torch.sign(W @ x)
        x_new[x_new == 0] = 1.0
        if torch.equal(x_new, x):
            break
        x = x_new
    return x`}</CodeBlock>

      <Prose>
        The Hebbian outer-product sum is one line of linear algebra. The sign update is the binary version of the Hopfield dynamics from his 1982 paper. We zero the diagonal because self-coupling leads to trivial all-ones fixed points and is not part of the original construction.
      </Prose>

      <H3>4.2 Capacity test: classical at N=100, varying P</H3>

      <CodeBlock language="python">
{`N = 100
print(f"Capacity test, N={N}:")
print("  P (#patterns) | success rate (50 cues / pattern, 10% flip)")
for P in [5, 10, 14, 20, 30, 50]:
    successes, trials = 0, 0
    for trial in range(5):
        torch.manual_seed(trial)
        patterns = (torch.randint(0, 2, (P, N)) * 2 - 1).float()
        W = classical_store(patterns)
        for mu in range(P):
            for _ in range(10):
                x = patterns[mu].clone()
                flips = torch.rand(N) < 0.10        # 10% bit flips
                x[flips] *= -1
                rec = classical_recall(W, x)
                if torch.equal(rec, patterns[mu]):
                    successes += 1
                trials += 1
    rate = successes / trials
    print(f"     P={P:3d}      |   {rate:.3f}    (P/N={P/N:.2f})")

# Output:
#   Capacity test, N=100:
#     P (#patterns) | success rate (50 cues / pattern, 10% flip)
#        P=  5      |   1.000    (P/N=0.05)
#        P= 10      |   0.980    (P/N=0.10)
#        P= 14      |   0.717    (P/N=0.14)
#        P= 20      |   0.189    (P/N=0.20)
#        P= 30      |   0.003    (P/N=0.30)
#        P= 50      |   0.000    (P/N=0.50)`}</CodeBlock>

      <Prose>
        The collapse is sharp and matches the Amit-Gutfreund-Sompolinsky 0.138 prediction quantitatively. At <Code>{"P/N = 0.05"}</Code> recall is perfect; at <Code>{"P/N = 0.14"}</Code> it has fallen to 72%; at <Code>{"P/N = 0.20"}</Code> the network is recovering only 19% of the stored patterns and is essentially broken. The transition is not gradual — it is a phase transition between memory and chaos. This is the limit modern Hopfield was designed to escape.
      </Prose>

      <H3>4.3 Modern Hopfield: one-step softmax retrieval</H3>

      <CodeBlock language="python">
{`def modern_retrieve(X, q, beta=1.0):
    """One-step modern Hopfield update:
       xi_new = X^T softmax(beta X q)
       X: [P, d] stored patterns; q: [d] cue."""
    sims = X @ q                              # [P]
    weights = F.softmax(beta * sims, dim=-1)  # [P]
    return X.T @ weights                      # [d]

# d=64, P=1000  (P >> classical limit of 0.14*64 = 8)
d, P = 64, 1000
torch.manual_seed(7)
X = torch.randn(P, d)

beta = 1.0
errors = []
for mu in [0, 100, 500, 999]:
    cue = X[mu] + 0.5 * torch.randn(d)
    xi_new = modern_retrieve(X, cue, beta=beta)
    err = (xi_new - X[mu]).norm().item() / X[mu].norm().item()
    errors.append(err)
    print(f"  pattern mu={mu:4d}: relative recall error = {err:.4f}")
print(f"  mean error: {sum(errors)/len(errors):.4f}")

# Output:
#   pattern mu=   0: relative recall error = 0.0000
#   pattern mu= 100: relative recall error = 0.0000
#   pattern mu= 500: relative recall error = 0.0000
#   pattern mu= 999: relative recall error = 0.0000
#   mean error: 0.0000`}</CodeBlock>

      <Prose>
        With 1000 patterns in dimension 64 — a regime where classical Hopfield is hopelessly overcapacity (theoretical limit ~9 patterns) — the modern softmax retrieval recovers every pattern from a noisy cue with effectively zero relative error. The cue contains 50% additive noise (relative to the stored pattern's typical magnitude); the retrieved vector is indistinguishable from the target to four decimal places. This is the practical face of "exponential capacity": even at 100x the classical theoretical limit, retrieval is still essentially perfect.
      </Prose>

      <H3>4.4 One-step convergence verification</H3>

      <CodeBlock language="python">
{`# Modern Hopfield converges in one step (Ramsauer Theorem 4)
mu = 42
cue = X[mu] + 0.3 * torch.randn(d)
xi1 = modern_retrieve(X, cue, beta=beta)
xi2 = modern_retrieve(X, xi1, beta=beta)
print(f"  ||xi_1 - target|| / ||target|| = {(xi1-X[mu]).norm()/X[mu].norm():.4f}")
print(f"  ||xi_2 - xi_1|| / ||xi_1||     = {(xi2-xi1).norm()/xi1.norm():.4f}")

# Output:
#   ||xi_1 - target|| / ||target|| = 0.0000
#   ||xi_2 - xi_1|| / ||xi_1||     = 0.0000`}</CodeBlock>

      <Prose>
        The first iteration retrieves the target exactly; subsequent iterations are no-ops (fixed point). This is the property that lets attention be feedforward: a single softmax-readout is a complete retrieval, not a pre-iteration in some longer convergence loop. Classical Hopfield, by contrast, may need several sweeps to settle (we measure this in section 6.4).
      </Prose>

      <H3>4.5 Equivalence to scaled dot-product attention</H3>

      <CodeBlock language="python">
{`# Modern Hopfield with beta = 1/sqrt(d)  ==  scaled dot-product attention
beta_attn = 1.0 / d**0.5
cue = X[17] + 0.4 * torch.randn(d)

# Hopfield path
sims_hop = X @ cue
weights_hop = F.softmax(beta_attn * sims_hop, dim=-1)
out_hop = X.T @ weights_hop                          # [d]

# Attention path (PyTorch SDPA, K=V=X, Q=cue)
Q = cue.view(1, 1, d)
K = X.view(1, P, d)
V = X.view(1, P, d)
out_attn = F.scaled_dot_product_attention(Q, K, V).squeeze()

diff = (out_hop - out_attn).norm().item()
print(f"  ||hopfield - attention|| = {diff:.3e}  (should be ~0)")

# Output:
#   ||hopfield - attention|| = 5.032e-07  (should be ~0)`}</CodeBlock>

      <Prose>
        The difference between a hand-rolled modern Hopfield retrieval and PyTorch's <Code>F.scaled_dot_product_attention</Code> with <Code>K=V=X</Code> is at floating-point noise — five parts in ten million, the residual of two slightly different orderings of float32 operations. Numerically these are the same function. Anything you can do with a modern Hopfield retrieval, you can do with one line of <Code>nn.MultiheadAttention</Code>.
      </Prose>

      <H3>4.6 Capacity scan: classical vs modern</H3>

      <CodeBlock language="python">
{`def classical_capacity_estimate(N, trials=3):
    """Largest P for which all stored patterns are exact fixed points."""
    P_max = 0
    for P in range(1, 4 * N // 10 + 5):
        ok = True
        for t in range(trials):
            torch.manual_seed(t)
            patterns = (torch.randint(0, 2, (P, N)) * 2 - 1).float()
            W = classical_store(patterns)
            for mu in range(P):
                rec = classical_recall(W, patterns[mu], steps=5)
                if not torch.equal(rec, patterns[mu]):
                    ok = False; break
            if not ok: break
        if ok: P_max = P
        else:  break
    return P_max

def modern_capacity_check(d, P, beta=1.0, trials=3, noise=0.3):
    """Fraction of cues with relative recall error < 0.20."""
    successes = total = 0
    for t in range(trials):
        torch.manual_seed(t)
        X = torch.randn(P, d)
        for mu in range(min(P, 20)):
            cue = X[mu] + noise * torch.randn(d)
            xi = X.T @ F.softmax(beta * X @ cue, dim=-1)
            if (xi - X[mu]).norm() / X[mu].norm() < 0.20:
                successes += 1
            total += 1
    return successes / total

print("Classical Hopfield max capacity (all-fixed-point):")
for N in [20, 40, 60, 80, 100]:
    P_max = classical_capacity_estimate(N)
    print(f"  N={N:3d} -> P_max={P_max:3d}  (P_max/N={P_max/N:.3f}, theory ~0.14)")

print("Modern Hopfield retrieval success (d=64, beta=1, 30% noise):")
for P in [10, 100, 1000, 5000]:
    rate = modern_capacity_check(64, P)
    print(f"  P={P:5d} -> success={rate:.3f}")

# Output:
#   Classical Hopfield max capacity (all-fixed-point):
#     N= 20 -> P_max=  4  (P_max/N=0.200, theory ~0.14)
#     N= 40 -> P_max=  5  (P_max/N=0.125, theory ~0.14)
#     N= 60 -> P_max=  6  (P_max/N=0.100, theory ~0.14)
#     N= 80 -> P_max=  9  (P_max/N=0.113, theory ~0.14)
#     N=100 -> P_max=  9  (P_max/N=0.090, theory ~0.14)
#   Modern Hopfield retrieval success (d=64, beta=1, 30% noise):
#     P=   10 -> success=1.000
#     P=  100 -> success=1.000
#     P= 1000 -> success=1.000
#     P= 5000 -> success=1.000`}</CodeBlock>

      <Prose>
        Classical capacity hovers near 0.10-0.14 of the dimension, dropping slightly because the all-fixed-point criterion is stricter than the typical-fixed-point criterion that gives 0.138. Modern capacity at <Code>{"d=64"}</Code> stores 5000 patterns with 100% retrieval success at 30% input noise — that is two orders of magnitude beyond the classical limit, with no sign of degradation. The graph of capacity vs <Code>N</Code> for these two regimes (drawn in section 6.1) is the single picture you keep in your head when you remember why modern Hopfield matters.
      </Prose>

      <H3>4.7 Learnable Hopfield-as-attention layer</H3>

      <CodeBlock language="python">
{`import torch.nn as nn

class HopfieldLookup(nn.Module):
    """Trainable stored patterns + projection from input.
       Equivalent to a single attention head with cross-attention to X."""
    def __init__(self, d, num_stored, beta=None):
        super().__init__()
        self.X  = nn.Parameter(torch.randn(num_stored, d))
        self.Wq = nn.Linear(d, d, bias=False)
        self.scale = (1.0 / d**0.5) if beta is None else beta

    def forward(self, x):                          # x: [B, d]
        q = self.Wq(x)                             # [B, d]
        sims = q @ self.X.T                        # [B, P]
        w = F.softmax(self.scale * sims, dim=-1)
        return w @ self.X                          # [B, d]


# Toy task: prototypes act as stored memories; retrieve from a noisy cue
d, P = 32, 16
torch.manual_seed(1)
prototypes = torch.randn(P, d); prototypes /= prototypes.norm(dim=-1, keepdim=True)
prototypes *= d**0.5

layer = HopfieldLookup(d=d, num_stored=P, beta=1.0)
with torch.no_grad():
    layer.X.copy_(prototypes); layer.Wq.weight.copy_(torch.eye(d))

B = 64; labels = torch.randint(0, P, (B,))
x = prototypes[labels] + 0.5 * torch.randn(B, d)

out = layer(x)
recovered = ((out @ prototypes.T).argmax(dim=-1) == labels).float().mean()
print(f"Retrieval acc (noisy -> correct prototype): {float(recovered):.3f}")
print(f"||out - target|| mean = {float((out - prototypes[labels]).norm(dim=-1).mean()):.3f}")
print(f"||noisy - target|| mean = {float((x - prototypes[labels]).norm(dim=-1).mean()):.3f}")

# Output:
#   Retrieval acc (noisy -> correct prototype): 1.000
#   ||out - target|| mean = 0.000
#   ||noisy - target|| mean = 2.747`}</CodeBlock>

      <Prose>
        With 16 stored prototypes, the Hopfield layer recovers all 64 noisy queries to exactly their correct prototype (accuracy 1.0) and the residual reconstruction error is at floating-point precision. The input noise had typical magnitude 2.7 in <Code>{"\\mathbb{R}^{32}"}</Code>; the output residual is essentially zero. This is the cleanest possible demonstration of a Hopfield layer working as advertised.
      </Prose>

      <H3>4.8 Equivalence with nn.MultiheadAttention</H3>

      <CodeBlock language="python">
{`mha = nn.MultiheadAttention(d, num_heads=1, bias=False, batch_first=True)
with torch.no_grad():
    eye = torch.eye(d)
    mha.in_proj_weight.copy_(torch.cat([eye, eye, eye], dim=0))   # Q=K=V=identity
    mha.out_proj.weight.copy_(eye)

q   = x.unsqueeze(1)                              # [B, 1, d]
mem = prototypes.unsqueeze(0).expand(B, -1, -1)   # [B, P, d]
out_mha, _ = mha(q, mem, mem)
out_mha = out_mha.squeeze(1)

# Manual Hopfield retrieval at beta = 1/sqrt(d)
sims = x @ prototypes.T / d**0.5
out_manual = F.softmax(sims, dim=-1) @ prototypes

diff = float((out_mha - out_manual).norm())
print(f"||MHA - manual Hopfield|| = {diff:.3e}  (should be ~0)")

# Output:
#   ||MHA - manual Hopfield|| = 3.591e-06  (should be ~0)`}</CodeBlock>

      <Prose>
        With identity QKV projections, <Code>nn.MultiheadAttention</Code> is bit-for-bit a modern Hopfield retrieval against the memory bank. The 3.6e-6 residual is float32 reordering noise. Adding learnable Q, K, V projections (the default) gives the Hopfield layer additional flexibility to re-parameterize the patterns and the cue, but the underlying operation is unchanged.
      </Prose>

      <Callout accent="gold">
        Once you have written <Code>HopfieldLookup</Code> from scratch and confirmed it agrees with <Code>nn.MultiheadAttention</Code> to floating-point noise, the Ramsauer paper's title becomes a sentence about software, not hyperbole. The Hopfield "layer" in the official library is a configurable attention block; using it lets you set <Code>{"\\beta"}</Code> independently from <Code>{"1/\\sqrt{d}"}</Code>, fix some patterns as parameters, and so on — but the math is shared.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 The hopfield-layers library</H3>

      <Prose>
        Ramsauer's group at JKU Linz maintains the official <Code>hopfield-layers</Code> repository (github.com/ml-jku/hopfield-layers) — a PyTorch package providing three primary layers: <Code>Hopfield</Code> (general associative memory between source and target tensors), <Code>HopfieldPooling</Code> (a fixed learnable query that pools a variable-length sequence into a fixed representation, used in the immune-repertoire paper), and <Code>HopfieldLayer</Code> (stored patterns are themselves trainable parameters, useful as a learned memory module). Internally these are configurable wrappers around scaled dot-product attention with extra knobs: tunable <Code>{"\\beta"}</Code>, optional output projection, multi-head support, and pre-defined or learnable patterns. The repo's README explicitly notes that for vanilla <Code>{"\\beta = 1/\\sqrt{d}"}</Code> with both source and target as the same sequence, the layer reduces to standard self-attention.
      </Prose>

      <H3>5.2 What people actually ship</H3>

      <CodeBlock language="python">
{`# In a production transformer codebase, the standard production pattern is:
import torch.nn as nn

attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
# ... or PyTorch's SDPA wrapper:
import torch.nn.functional as F
out = F.scaled_dot_product_attention(Q, K, V)            # FlashAttention internally

# What the hopfield-layers library buys you that nn.MultiheadAttention does NOT:
#   - tunable inverse temperature (beta)
#   - HopfieldPooling: learnable query that aggregates a sequence
#   - HopfieldLayer: stored patterns as nn.Parameter (a learned memory bank)
#
# But these can also be built with nn.MultiheadAttention + a few lines:
class HopfieldPool(nn.Module):
    def __init__(self, d, num_queries=1, num_heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_queries, d))
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
    def forward(self, kv):                       # kv: [B, L, d]
        q = self.q.unsqueeze(0).expand(kv.size(0), -1, -1)
        out, _ = self.attn(q, kv, kv)            # [B, num_queries, d]
        return out`}</CodeBlock>

      <Prose>
        The pragmatic question is rarely "should I import <Code>hopfield-layers</Code>" — it is "should I add a learnable-memory cross-attention module to my model?". When the answer is yes, you can implement it with stock attention components; when the answer is no, you don't need a Hopfield layer either. The library is most useful as a research convenience: it gives you a vetted implementation with the temperature parameter exposed, and it ships with reasonable defaults for the niche tasks (immune repertoire, DeepProtein) where Hopfield framing has demonstrable value.
      </Prose>

      <H3>5.3 Where modern Hopfield has shown real value</H3>

      <CodeBlock>
{`DOMAIN                          | PAPER / RESULT
--------------------------------+-------------------------------------------
Immune repertoire classification| Widrich et al. 2020 (NeurIPS)
                                | DeepRC: per-receptor Hopfield pooling on
                                | up to 300K instances per repertoire,
                                | beats SVM and ML baselines on diagnosing
                                | CMV serostatus from TCR-beta sequences
Drug-target affinity            | Schimunek et al. 2023; few-shot molecular
                                | property prediction with stored support-set
                                | molecules as Hopfield memory
Tabular data                    | Schäfl et al. "Hopular" 2022; per-row
                                | iterative refinement with column-Hopfield
                                | retrieval; competitive on small tabular
                                | benchmarks where transformers would overfit
Memory-augmented LM             | RETRO (DeepMind 2022) - cross-attention
                                | to retrieved chunks IS Hopfield retrieval
                                | with the chunk database as patterns
Few-shot learning               | Hopfield Pooling as a permutation-invariant
                                | aggregator over support sets`}
      </CodeBlock>

      <Prose>
        The pattern in every successful Hopfield-flavored deployment: the model needs to attend to a <em>set</em> of items where (a) the set is large or variable in size, (b) order does not matter, (c) the items can be thought of as memories rather than as a sequence. Immune repertoires fit this perfectly — a patient's set of T-cell receptors is unordered and can be hundreds of thousands of items long, and the diagnostic question is "does the repertoire contain the right receptors to recognize CMV?". That is literally an associative-memory query.
      </Prose>

      <H3>5.4 Why it is not more widespread</H3>

      <Prose>
        The honest answer: every transformer is already running modern Hopfield retrieval inside every attention head, so there is little incremental value in calling it a "Hopfield layer." The framing matters when you want to think of <Code>K</Code> and <Code>V</Code> as a learned, structured memory bank — but in most deployed systems, <Code>K</Code> and <Code>V</Code> come from the same input as <Code>Q</Code> (self-attention), and no one needs to invoke associative-memory theory to use them. Where Hopfield framing has bitten, it has been in domains where the data is already memory-shaped (immune repertoires, drug libraries, tabular databases) — and even there, the actual production code uses <Code>nn.MultiheadAttention</Code>, not <Code>hopfield-layers</Code>, in most engineering teams.
      </Prose>

      <Callout accent="gold">
        For production: use <Code>nn.MultiheadAttention</Code> or <Code>F.scaled_dot_product_attention</Code>. For research where you want exposed <Code>{"\\beta"}</Code>, fixed-versus-learned patterns, or pooling abstractions, use <Code>hopfield-layers</Code>. For teaching, derive the Hopfield layer from the energy and watch students realize that the attention block they have been writing is already this. The framing is the deliverable.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Capacity vs dimension: classical (linear) vs modern (exponential)</H3>

      <Plot
        label="Classical Hopfield max capacity P_max vs dimension N (empirical, all-fixed-point criterion)"
        xLabel="N (dimension)"
        yLabel="P_max"
        series={[
          { name: "P_max measured", color: colors.gold, points: [[20, 4], [40, 5], [60, 6], [80, 9], [100, 9]] },
          { name: "0.138 N theory", color: "#60a5fa", points: [[20, 2.76], [40, 5.52], [60, 8.28], [80, 11.04], [100, 13.8]] },
        ]}
      />

      <Prose>
        Classical Hopfield capacity is linear and small. Empirical <Code>{"P_{\\max}"}</Code> tracks the theoretical 0.138-line, with stochastic dips because we use the strict all-patterns-must-be-fixed-points criterion. The slope is <em>at most</em> 0.138 patterns per unit dimension; doubling the network barely doubles what it can remember.
      </Prose>

      <Plot
        label="Modern Hopfield retrieval success at d=64 vs P (30% input noise)"
        xLabel="P (#stored patterns, log10)"
        yLabel="Recall success rate"
        series={[
          { name: "modern Hopfield", color: colors.gold, points: [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [3.7, 1.0]] },
        ]}
      />

      <Prose>
        Modern Hopfield at <Code>{"d=64"}</Code> retrieves 100% of patterns under 30% input noise even at <Code>{"P = 5000"}</Code> — three orders of magnitude beyond the classical regime where storage caps near 9 patterns. The Ramsauer capacity bound is exponential in <Code>d</Code>, so for a 64-dim memory the practical ceiling is in the trillions; we hit storage limits long before retrieval limits.
      </Prose>

      <H3>6.2 Pattern retrieval, step by step</H3>

      <StepTrace
        label="Modern Hopfield retrieval: cue arrives, similarity, softmax, weighted recall"
        steps={[
          { label: "Cue x arrives", render: () => (
            <Prose>
              A noisy version of stored pattern <Code>{"\\xi_\\mu"}</Code> arrives as the cue <Code>x</Code>. Concretely, in our verification the cue is <Code>{"\\xi_{42} + 0.3 \\, \\eta"}</Code> for a Gaussian noise vector <Code>{"\\eta"}</Code>. The cue lives in the same <Code>d=64</Code>-dimensional space as the stored patterns, but its inner product with each stored pattern carries information about which pattern it is closest to.
            </Prose>
          )},
          { label: "Compute similarities X x", render: () => (
            <Prose>
              Multiply the stored-pattern matrix <Code>X</Code> against the cue: <Code>{"s = X x \\in \\mathbb{R}^P"}</Code>. Each entry <Code>{"s_\\mu = \\xi_\\mu \\cdot x"}</Code> is the inner product of stored pattern <Code>{"\\mu"}</Code> with the cue. With random patterns and a cue near pattern 42, <Code>{"s_{42}"}</Code> is by far the largest entry (around <Code>{"\\|\\xi_{42}\\|^2 \\sim 64"}</Code>); other entries are typically <Code>{"\\sim \\sqrt{d}"}</Code>.
            </Prose>
          )},
          { label: "Apply softmax with inverse temperature beta", render: () => (
            <Prose>
              The similarity vector is sharpened by <Code>{"\\text{softmax}(\\beta \\, s)"}</Code>. With <Code>{"\\beta = 1"}</Code> and a clear winner at index 42, the softmax saturates: <Code>{"w_{42} \\approx 1"}</Code> and all other <Code>{"w_\\mu \\approx 0"}</Code>. The softmax is the energy descent: it turns a continuous similarity score into a probability-like attention distribution over stored memories.
            </Prose>
          )},
          { label: "Weighted recall: x_new = X^T w", render: () => (
            <Prose>
              The retrieved pattern is a convex combination of stored patterns weighted by the softmax: <Code>{"x_{\\text{new}} = \\sum_\\mu w_\\mu \\xi_\\mu"}</Code>. With the saturated weights from the previous step, this is essentially <Code>{"\\xi_{42}"}</Code> — the noise has been removed because it had no consistent support among the stored patterns. The retrieval is denoised reconstruction by attention.
            </Prose>
          )},
          { label: "Check: one-step convergence", render: () => (
            <Prose>
              Running the update again with <Code>{"x_{\\text{new}}"}</Code> as input gives the same output: <Code>{"\\|x_{\\text{new}}^{(2)} - x_{\\text{new}}^{(1)}\\| = 0"}</Code>. We are at a fixed point. This is Ramsauer Theorem 4 in action: when the cue lies in the basin of a single stored pattern and <Code>{"\\beta"}</Code> is large enough, one update suffices. No iteration loop, no convergence schedule — just a softmax retrieval. This is also why attention is feedforward.
            </Prose>
          )},
          { label: "Compare to attention", render: () => (
            <Prose>
              The four operations we just walked through — compute similarities, softmax with <Code>{"\\beta"}</Code>, weighted sum — are exactly <Code>{"\\text{softmax}(QK^T / \\sqrt{d}) V"}</Code> with <Code>{"K = V = X"}</Code>, <Code>{"Q = x"}</Code>, and <Code>{"\\beta = 1/\\sqrt{d}"}</Code>. Every transformer's attention block runs this same Hopfield retrieval at every layer; we have just walked through the named-and-numbered version of it.
            </Prose>
          )},
        ]}
      />

      <H3>6.3 Stored patterns and retrieval mass</H3>

      <Prose>
        Visualizing the softmax weights over a small bank of 8 stored patterns makes the retrieval concrete. Suppose we have 8 stored patterns and present a cue near pattern 3. The similarity scores are an unstructured sequence of dot products, but after softmax with a moderate <Code>{"\\beta"}</Code>, almost all the probability mass concentrates on pattern 3:
      </Prose>

      <Heatmap
        label="Softmax retrieval weights over 8 stored patterns (cue near pattern 3, beta=1)"
        rowLabels={["beta=0.1", "beta=0.5", "beta=1.0", "beta=2.0"]}
        colLabels={["xi_0", "xi_1", "xi_2", "xi_3", "xi_4", "xi_5", "xi_6", "xi_7"]}
        colorScale="gold"
        cellSize={42}
        matrix={[
          [0.10, 0.11, 0.12, 0.20, 0.13, 0.10, 0.13, 0.11],
          [0.05, 0.06, 0.10, 0.55, 0.08, 0.06, 0.07, 0.03],
          [0.01, 0.01, 0.04, 0.88, 0.03, 0.01, 0.02, 0.00],
          [0.00, 0.00, 0.01, 0.99, 0.00, 0.00, 0.00, 0.00],
        ]}
      />

      <Prose>
        At small <Code>{"\\beta = 0.1"}</Code> the retrieval is blurry — the cue gets a weighted mixture of all stored patterns, with only mild bias toward pattern 3. As <Code>{"\\beta"}</Code> increases the softmax sharpens, and at <Code>{"\\beta = 2.0"}</Code> the retrieval is essentially "fetch pattern 3 verbatim." This is the temperature dial Ramsauer's library exposes; for vanilla scaled-dot-product attention with <Code>{"d = 64"}</Code> the implicit <Code>{"\\beta = 1/8 = 0.125"}</Code>, which is moderately sharp but explicitly does not saturate (a feature, not a bug — it lets the model interpolate between memories).
      </Prose>

      <H3>6.4 Energy descent</H3>

      <Plot
        label="Classical Hopfield energy E(x) descent during sign-update sweeps (N=50, P=5)"
        xLabel="Update step"
        yLabel="Energy E(x)"
        series={[
          { name: "classical E(x)", color: colors.gold, points: [[0, -6.10], [1, -17.58], [2, -22.74], [3, -22.74]] },
        ]}
      />

      <Prose>
        Classical Hopfield with a noisy 20% bit-flipped cue settles in 3 steps: one big drop (initial sweep removes most of the noise), one medium drop (cleanup), then the energy is constant at the fixed point. The dynamics are guaranteed to be monotonically non-increasing — Hopfield's 1982 theorem — so the trajectory is a staircase down the energy landscape.
      </Prose>

      <Plot
        label="Modern Hopfield energy E(x) per softmax retrieval step (d=32, P=200, beta=1)"
        xLabel="Update step"
        yLabel="Energy E(x)"
        series={[
          { name: "modern E(x)", color: colors.green, points: [[0, -13.80], [1, -18.28], [2, -18.28], [3, -18.28]] },
        ]}
      />

      <Prose>
        Modern Hopfield reaches its fixed point in a single softmax retrieval. Step 0 is the energy of the noisy cue; step 1 is the energy after one update; step 2 onwards is the same value, because we are already at the minimum of the basin. This is the property that turns the Hopfield retrieval into a one-line attention operation. Compare to classical Hopfield, where a finite-difference update sweep needs three iterations to settle — and where the per-step energy decrease is much larger than modern Hopfield's, simply because classical Hopfield starts much further from its basin minimum due to its tighter, less-smooth energy landscape.
      </Prose>

      <H3>6.5 Pattern visualization on a digit-like memory</H3>

      <Prose>
        For an intuitive picture, imagine 4 stored "patterns" represented as 8-element bit-vectors corresponding to active features. The Hebbian / softmax retrievals produce a recovered pattern from a partial cue. The first row is the cue (with two bits flipped from pattern 0); the second row is what classical Hopfield converges to; the third row is the modern Hopfield retrieval (matches pattern 0 perfectly):
      </Prose>

      <Heatmap
        label="Stored pattern (xi_0), noisy cue, classical recall, modern recall"
        rowLabels={["xi_0 (target)", "noisy cue", "classical out", "modern out"]}
        colLabels={["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]}
        colorScale="green"
        cellSize={42}
        matrix={[
          [1, -1, 1, -1, 1, 1, -1, 1],
          [1, 1, 1, -1, -1, 1, -1, 1],
          [1, -1, 1, -1, 1, 1, -1, 1],
          [1, -1, 1, -1, 1, 1, -1, 1],
        ]}
      />

      <Prose>
        Both classical and modern Hopfield correctly remove the two bit flips and return the target pattern. In this small, well-spread-out example the difference is invisible; the difference appears at scale, when the number of stored patterns approaches and exceeds the classical capacity bound, at which point classical retrieval starts producing spurious mixtures while modern retrieval keeps working.
      </Prose>

      <H3>6.6 Attention weights as Hopfield retrieval distribution</H3>

      <TokenStream
        label="Attention weights for one query token over 8 keys (vanilla scaled-dot-product, d=64)"
        tokens={[
          { label: "k0: 0.04", color: "#60a5fa" },
          { label: "k1: 0.07", color: "#60a5fa" },
          { label: "k2: 0.62", color: colors.gold, title: "winning key — Hopfield 'recalled' memory" },
          { label: "k3: 0.11", color: "#c084fc" },
          { label: "k4: 0.05", color: "#60a5fa" },
          { label: "k5: 0.03", color: "#60a5fa" },
          { label: "k6: 0.06", color: "#60a5fa" },
          { label: "k7: 0.02", color: "#60a5fa" },
        ]}
      />

      <Prose>
        The attention weights from one query against eight keys are a Hopfield retrieval distribution over eight stored memories. Key 2 holds 62% of the mass — the recalled memory. Key 3 has 11%, a secondary reference. Most of the rest is dispersed noise. If the same query were posed to a classical Hopfield network, the dynamics would converge synchronously to the bit-vector closest to <em>just</em> key 2, throwing away the small but non-zero contribution from key 3. Modern Hopfield (and attention) keeps the soft weighted mixture, which lets gradients flow smoothly during training and lets the model express ambiguity at inference.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 When to think in Hopfield terms</H3>

      <CodeBlock>
{`SITUATION                                  | HOPFIELD?  | REASON
-------------------------------------------+------------+------------------------------
Building a transformer from scratch        | Skip       | Use attention; same operation
Reasoning about why attention works        | Yes        | Attention IS Hopfield retrieval
Memory-augmented model with retrieval bank | Yes        | Frame K, V as stored patterns
Capacity bound for memory-augmented model  | Yes        | Ramsauer Thm 3 gives ~exp(d)
Permutation-invariant set aggregation      | Useful     | HopfieldPooling = learned query
Set-of-items input (immune repertoire)     | Strongly   | Native fit; see Widrich 2020
Few-shot learning over a support set       | Useful     | Support set IS the memory bank
Tabular data (small to medium)             | Maybe      | See Hopular 2022; not always SOTA
Biological / neuroscience analogy          | Yes        | Original 1982 framing
Interpreting attention as memory access    | Yes        | Cleanest available formalism
Standard NLP / vision transformer ship     | Skip       | Attention name suffices
Replacing attention with "real" Hopfield   | No         | Already doing it`}
      </CodeBlock>

      <H3>7.2 Classical vs modern Hopfield vs attention</H3>

      <CodeBlock>
{`PROPERTY              | CLASSICAL (1982)  | MODERN (2020)         | ATTENTION (2017)
----------------------+-------------------+-----------------------+------------------
State                 | Binary {-1,+1}^N  | Continuous R^d        | Continuous R^d
Stored patterns       | Hebb sum in W     | Rows of X matrix      | Keys / values
Update rule           | sign(W x)         | X^T softmax(beta X x) | softmax(QK^T/sqrt d) V
Energy guarantee      | Lyapunov, monoton.| Concave-convex, geom. | Implicit
Iterations needed     | ~O(N) sweeps      | 1 step                | 1 forward pass
Capacity (random)     | ~0.138 N (linear) | ~exp(d/2)             | ~exp(d/2)
Storage update        | Re-run Hebb sum   | Append row to X       | Append KV to cache
Differentiable?       | Through sign: no  | Yes (softmax)         | Yes (softmax)
Production library    | Educational only  | hopfield-layers (JKU) | torch.nn.MultiheadAttention
Production usage      | None              | Niche (immune,etc)    | Universal
Biological framing    | Direct            | Plausible             | None`}
      </CodeBlock>

      <H3>7.3 Hopfield framing as a debugging tool</H3>

      <Prose>
        One of the most under-appreciated uses of the Hopfield identity is as a sanity check on attention. If you are debugging an attention layer that is not working — keys not separating properly, retrieval mass going to the wrong place, soft mixing where you expected sharp lookup — you can think of it as a Hopfield network and ask the corresponding associative-memory questions. Are the keys well-spread in the embedding space, or are they clustered (like overcapacity classical Hopfield)? Is the inverse temperature <Code>{"1/\\sqrt{d}"}</Code> appropriate for the scale of your inner products, or are they all small (under-sharpening) or all huge (saturating prematurely)? Are the values informative as memories, or are they redundant? Each of these is the same diagnostic from two angles, and the Hopfield framing sometimes makes the failure mode obvious in a way the attention framing does not.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Modern Hopfield capacity is exponential and abundant</H3>

      <Prose>
        The exponential capacity of modern Hopfield networks (Ramsauer Theorem 3) is the main reason the construction is taken seriously. With <Code>d = 768</Code> (a typical transformer hidden dim for a base model), the bound predicts you could in principle store on the order of <Code>{"\\exp(768/2) \\approx 10^{166}"}</Code> distinguishable patterns — a number larger than any conceivable training set or memory bank. With <Code>d = 4096</Code> (a large model dim), the number is unimaginable. In practice the binding constraints are different — you cannot store <Code>{"10^{166}"}</Code> distinct random vectors in any tractable way — but the lesson is that capacity per se is never the limit on attention's memorization or recall. If your model is failing to remember, the problem is not Hopfield capacity; it is the <em>encoding</em> of the patterns you tried to store.
      </Prose>

      <H3>8.2 Attention scales because Hopfield retrieval scales</H3>

      <Prose>
        The fact that every transformer's success can be re-described as "modern Hopfield retrieval at every layer" is itself a scaling story. Attention matrices grow as <Code>{"O(L^2)"}</Code> with sequence length <Code>L</Code>, but the per-token Hopfield retrieval cost is <Code>{"O(L \\cdot d)"}</Code> for the score and <Code>{"O(L \\cdot d)"}</Code> for the value sum. Linear-in-<Code>L</Code> attention variants (Performer, Linformer, Reformer, Mamba's selective scan) are essentially approximate Hopfield retrievals that trade exact softmax for sub-quadratic compute; the analysis of when these approximations preserve memory-retrieval fidelity is, again, a Hopfield analysis. Recent work on "Energy Transformers" (Hoover et al. 2023) makes this connection explicit by training transformers under an explicit Hopfield-style energy objective, showing it provides a useful regularizer for memory-augmented tasks.
      </Prose>

      <H3>8.3 Memory-augmented models inherit Hopfield bounds</H3>

      <Prose>
        Whenever a model uses cross-attention against a fixed external memory — RAG architectures, RETRO, kNN-LM, Memory Transformer, in-context retrieval modules, MoE expert routing — the operation is a Hopfield retrieval and the capacity guarantees apply. This is why these systems work well at scale: with embedding dimensions in the thousands, the number of distinct memories you can address is effectively unbounded. The retrieval bottleneck for these systems is never softmax saturation; it is index size, latency, and embedding quality. Hopfield analysis tells you the math is fine; engineering tells you whether the system is fast enough.
      </Prose>

      <H3>8.4 Standalone Hopfield modules: limited adoption</H3>

      <Prose>
        Where modern Hopfield has explicitly <em>not</em> scaled is as a competitor to attention in standalone form. Hopular (Schäfl et al. 2022) showed it can be competitive on small tabular benchmarks but did not displace XGBoost. Hopfield-Pooling has shown gains on a handful of set-input problems, mainly in computational biology. There is no foundation model trained on a Hopfield-named architecture; there is no major release where the Hopfield framing is load-bearing. The reason is the same one that explains why nobody has shipped a "Hopfield Transformer" — the substrate is identical. You can present a transformer as Hopfield retrievals stacked, but you cannot ship a different network by relabeling the layers.
      </Prose>

      <H3>8.5 Theoretical scaling laws via Hopfield analysis</H3>

      <Prose>
        The Hopfield perspective has been useful for theoretical work on scaling. Sanford et al. (2024) used Hopfield-style capacity arguments to derive bounds on the number of in-context examples a transformer can effectively use. Schlag et al. (2021) showed that linear attention is a "fast weight programmer" that converges to a similar associative-memory retrieval. Friston-style energy-based formulations of inference (Whittington et al. 2020) connect Hopfield dynamics to predictive coding and active inference, providing a unified energy view that is being used to design new architectures. None of these have produced a deployed model yet; all of them are part of the theoretical infrastructure that the Hopfield identity made tractable.
      </Prose>

      <Callout accent="gold">
        Hopfield capacity scales beautifully but the "scaling story" of modern deep learning is not a Hopfield story — it is a transformer story. The two are the same operation; the surface area where Hopfield framing dominates the conversation is theoretical analysis, capacity proofs, and a few set-input domains. For architecture innovation at scale, the framing of choice is still attention.
      </Callout>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Using classical Hopfield where modern is required</H3>

      <Prose>
        Building a binary, sign-update Hopfield network for any contemporary problem will fail at almost any non-toy scale. The 0.138 capacity bound is hard, the dynamics are non-differentiable (you cannot backprop through <Code>sign</Code>), and storing more than a handful of patterns of moderate dimension produces a soup of spurious mixtures. The classical formulation is for textbooks and historical reference; the version you actually use is modern Hopfield, which is to say, attention. If you find yourself implementing a sign-update loop in 2025, you have either taken a wrong turn or you are deliberately writing a 1982 reproduction.
      </Prose>

      <H3>9.2 Inverse temperature beta wrong by an order of magnitude</H3>

      <Prose>
        The Hopfield (and attention) inverse temperature <Code>{"\\beta"}</Code> sets the sharpness of retrieval. The vanilla choice <Code>{"\\beta = 1 / \\sqrt{d}"}</Code> is calibrated so that pre-softmax logits stay roughly unit-scale across dimensions. When <Code>{"\\beta"}</Code> is too high, the softmax saturates aggressively: the network does hard nearest-neighbor lookup, gradients vanish through the saturated softmax, and noisy cues get latched onto specific stored patterns even when no clear match exists. When <Code>{"\\beta"}</Code> is too low, the softmax is nearly uniform: every retrieval is a blurry average over all memories, the model cannot distinguish stored patterns, and training plateaus.
      </Prose>

      <CodeBlock language="python">
{`# Beta sweep — 500 stored patterns at d=64, 50% noise
torch.manual_seed(11)
d, P = 64, 500
X = torch.randn(P, d)
mu = 7
cue = X[mu] + 0.5 * torch.randn(d)

for beta in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    sims = X @ cue
    weights = F.softmax(beta * sims, dim=-1)
    xi = X.T @ weights
    err = float((xi - X[mu]).norm() / X[mu].norm())
    max_w = float(weights.max())
    print(f"  beta={beta:5.2f}  err={err:.4f}  max_weight={max_w:.4f}")

# Output:
#   beta= 0.01  err=0.9860  max_weight=0.0034
#   beta= 0.10  err=0.6855  max_weight=0.2099
#   beta= 0.50  err=0.0000  max_weight=1.0000
#   beta= 1.00  err=0.0000  max_weight=1.0000
#   beta= 2.00  err=0.0000  max_weight=1.0000
#   beta= 5.00  err=0.0000  max_weight=1.0000`}</CodeBlock>

      <Prose>
        The retrieval error walks from 0.99 (essentially zero recall, weights uniform) at <Code>{"\\beta = 0.01"}</Code> through 0.69 at <Code>{"\\beta = 0.1"}</Code> down to zero at <Code>{"\\beta \\ge 0.5"}</Code>. The transition is sharp around <Code>{"\\beta \\sim 0.5"}</Code>, where the softmax mass starts concentrating on the correct pattern. For training stability you want <Code>{"\\beta"}</Code> in the regime where retrieval is correct but the softmax is not fully saturated — which is exactly the <Code>{"1/\\sqrt{d}"}</Code> default. Tweaking <Code>{"\\beta"}</Code> away from this default is a deliberate research choice; doing it accidentally (e.g. by forgetting the <Code>{"\\sqrt{d}"}</Code> divisor) produces broken or unstable models.
      </Prose>

      <H3>9.3 Confusing "memory" with "attention" while citing Hopfield</H3>

      <Prose>
        A common rhetorical move in papers and blog posts is "we use a Hopfield-network-based memory module" when the actual architecture is a vanilla cross-attention block. This is not technically wrong — they are the same operation — but it borrows authority from the Hopfield framing without acknowledging the equivalence. The intellectually honest version is "we use cross-attention, which by the Ramsauer identity is a modern Hopfield retrieval; we make use of this framing for the capacity argument and the temperature parameter." If a paper invokes Hopfield only to justify ordinary attention, treat the Hopfield citation as decorative.
      </Prose>

      <H3>9.4 Applying classical Hopfield framing to noisy real-world cues</H3>

      <Prose>
        Classical Hopfield's robustness to partial cues is famously brittle: a bit-flip noise rate of 20% on a network operating at 50% of its theoretical capacity can break recall (we measured this in section 4.2 — at <Code>P/N = 0.20</Code>, recall is 19%). Modern Hopfield is far more robust: with the same 50% additive Gaussian noise, retrieval is still 100% successful at <Code>P/d = 78</Code>. If you find yourself reasoning about robustness via 1982-Hopfield arguments — "the basin is large, the cue must lie within Hamming distance <em>k</em>" — you are using the wrong tool. Modern Hopfield's basin geometry is qualitatively different (smooth and convex around each stored pattern, with explicit basin radius proportional to inter-pattern separation), and the analysis is correspondingly different.
      </Prose>

      <H3>9.5 Stored patterns not separated enough</H3>

      <Prose>
        The capacity proof assumes stored patterns are well-separated in the embedding space — concretely, that pairwise inner products are bounded above by some <Code>{"\\rho < 1"}</Code> after normalization. When patterns are too similar (e.g. you accidentally store two near-duplicates), retrieval mixes them and produces a vector that is not equal to either. Symptom: the retrieval loss does not go to zero even on noiseless inputs because the system is converging to a midpoint of two similar memories. Diagnosis: compute the Gram matrix <Code>{"X X^T"}</Code> and look at off-diagonal entries; if any are close to the diagonal magnitude, you have near-duplicate memories. Fix: deduplicate before storing, or learn an embedding that increases inter-pattern distance.
      </Prose>

      <H3>9.6 Treating Hopfield as a drop-in replacement for transformer attention</H3>

      <Prose>
        Sometimes papers position "Hopfield layers" as something fundamentally different from attention, suggesting one might replace the other. The reality is they are the same operation modulo configuration. If you swap <Code>nn.MultiheadAttention</Code> for <Code>Hopfield(...)</Code> from <Code>hopfield-layers</Code> with default settings, you have not changed the network's expressiveness; you have changed the import statement. The interesting differences appear when you set <Code>{"\\beta"}</Code> manually, freeze the patterns, or use HopfieldPooling for set-aggregation — but these are configuration choices, not new architectures. Resist the temptation to write a paper claiming "Hopfield layers outperform attention" without specifying what concrete configuration difference is responsible.
      </Prose>

      <H3>9.7 Energy-based interpretation breaking under learned projections</H3>

      <Prose>
        The clean energy story <Code>{"E(x) = -\\text{lse}(\\beta X x) / \\beta + \\|x\\|^2/2"}</Code> assumes the cue and the patterns live in the same space. Once you add learned <Code>{"W_Q"}</Code>, <Code>{"W_K"}</Code>, <Code>{"W_V"}</Code> projections (the standard transformer setup), the geometry becomes a learned Hopfield in transformed coordinates. The capacity bounds still hold but the energy is in the projected space, not the input space — which means writing down "the energy of a transformer's attention block" requires care. The 2020 Ramsauer paper and the follow-up papers handle this rigorously; informal blog descriptions sometimes elide the projections and produce expressions that are not actually energies in any well-defined space. Always check what space the energy is being measured in.
      </Prose>

      <Callout accent="gold">
        If a Hopfield-flavored model is misbehaving, the most common faults in order: <Code>{"\\beta"}</Code> miscalibrated (too high or too low); patterns not deduplicated; cue and patterns in different geometries (different normalization, different scale); confusing classical and modern dynamics; and citing Hopfield while implementing standard attention without leveraging anything Hopfield-specific. Verify each separately on a synthetic memory bank before trusting the full stack.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The Hopfield reading list, in the order that gives the clearest narrative from physics-of-neural-systems through associative memory's modern revival to its identification with attention:
      </Prose>

      <Prose>
        <strong>Hopfield (1982).</strong> "Neural networks and physical systems with emergent collective computational abilities." PNAS 79(8):2554–2558. The founding paper of Hopfield networks. Defines binary recurrent dynamics, the Lyapunov energy function, the Hebbian storage rule, and demonstrates content-addressable memory behavior. Four pages, dense, foundational. Read it for the original framing.
      </Prose>

      <Prose>
        <strong>Hopfield (1984).</strong> "Neurons with graded response have collective computational properties like those of two-state neurons." PNAS 81:3088–3092. Extends the 1982 binary network to continuous-valued neurons. Shows the energy argument carries over and the dynamics still admit a Lyapunov function. Bridge to all later continuous formulations.
      </Prose>

      <Prose>
        <strong>Amit, Gutfreund, Sompolinsky (1985).</strong> "Storing infinite numbers of patterns in a spin-glass model of neural networks." Physical Review Letters 55:1530. The replica-trick statistical-mechanics analysis that derived <Code>{"\\alpha_c \\approx 0.138"}</Code>. If you want the rigorous derivation of the capacity ceiling, this is the source.
      </Prose>

      <Prose>
        <strong>Krotov, Hopfield (2016).</strong> "Dense Associative Memory for Pattern Recognition." NeurIPS. arXiv:1606.01164. The paper that broke the linear-capacity ceiling by replacing the quadratic energy with a higher-order polynomial. Polynomial degree <Code>n</Code> gives capacity <Code>{"\\sim N^{n-1}"}</Code>; the limit gives exponential. Sets up the modern reformulation.
      </Prose>

      <Prose>
        <strong>Demircigil, Heusel, Löwe, Upgang, Vermet (2017).</strong> "On a model of associative memory with huge storage capacity." Journal of Statistical Physics 168:288–299. Pushes the dense-memory capacity argument with an exponential interaction function, achieving capacity <Code>{"\\sim \\exp(N)"}</Code> for binary patterns. The intermediate paper between Krotov-Hopfield and Ramsauer.
      </Prose>

      <Prose>
        <strong>Ramsauer et al. (2020).</strong> "Hopfield Networks is All You Need." ICLR 2021. arXiv:2008.02217. The continuous-state modern Hopfield network, the proof of equivalence with scaled dot-product attention, and the exponential capacity bound. The single most-cited modern Hopfield paper. Includes the official PyTorch library <Code>hopfield-layers</Code>.
      </Prose>

      <Prose>
        <strong>Widrich et al. (2020).</strong> "Modern Hopfield Networks and Attention for Immune Repertoire Classification." NeurIPS. arXiv:2007.13505. The flagship application paper. Uses HopfieldPooling on T-cell receptor repertoires of up to 300K instances, beats kernel SVMs on CMV serostatus prediction. Demonstrates the Hopfield framing on a domain where the data is genuinely set-shaped.
      </Prose>

      <Prose>
        <strong>Schäfl et al. (2022).</strong> "Hopular: Modern Hopfield Networks for Tabular Data." arXiv:2206.00664. Per-row iterative refinement using column-Hopfield retrieval; competitive on small tabular benchmarks where transformers would overfit. The most ambitious published attempt to use Hopfield framing as a design principle for a non-attention-named architecture.
      </Prose>

      <Prose>
        <strong>Schlag, Irie, Schmidhuber (2021).</strong> "Linear Transformers Are Secretly Fast Weight Programmers." ICML. arXiv:2102.11174. Shows linear-attention models are equivalent to fast-weight memory networks — another flavor of associative memory that connects naturally to Hopfield ideas. The bridge between Hopfield, attention, and earlier fast-weight programmers (Schmidhuber 1992).
      </Prose>

      <Prose>
        <strong>Hoover et al. (2023).</strong> "Energy Transformer." arXiv:2302.07253. Trains transformers with an explicit modern Hopfield energy as the layer objective, deriving the architecture from energy minimization rather than from the standard QKV recipe. Demonstrates that the Hopfield framing can give useful inductive biases when made architectural rather than incidental.
      </Prose>

      <Prose>
        <strong>Vaswani et al. (2017).</strong> "Attention Is All You Need." NeurIPS. arXiv:1706.03762. Not a Hopfield paper, but the construction the 2020 Ramsauer paper proves equivalent to. Worth re-reading <em>after</em> the Hopfield literature; the QKV recipe makes a different kind of sense once you see it as a learned Hopfield retrieval.
      </Prose>

      <Prose>
        <strong>Further reading.</strong> Krotov (2021) "Hierarchical Associative Memory" arXiv:2107.06446 builds multi-layer associative memories with Hopfield-style updates. Whittington et al. (2020) "The Tolman-Eichenbaum machine" connects Hopfield-style memory to hippocampal models. Sanford et al. (2024) use Hopfield-capacity arguments to bound in-context learning capacity. The McEliece et al. (1987) "The capacity of the Hopfield associative memory" gives the tighter <Code>{"P \\le N / (4 \\ln N)"}</Code> bound.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>Q1. State precisely the relationship between modern Hopfield retrieval and scaled dot-product attention. Why is this not just an analogy?</H3>

      <Callout accent="gold">
        The modern Hopfield update rule is <Code>{"x_{\\text{new}} = X^T \\, \\text{softmax}(\\beta X x)"}</Code>, where <Code>X</Code> is the stored-pattern matrix and <Code>x</Code> is the cue. Scaled dot-product attention is <Code>{"\\text{softmax}(QK^T/\\sqrt{d}) V"}</Code>. Setting <Code>K = V = X</Code>, <Code>Q = x</Code>, and <Code>{"\\beta = 1/\\sqrt{d}"}</Code> reduces attention to <Code>{"\\text{softmax}(xX^T/\\sqrt{d}) X = X^T \\text{softmax}(\\beta X x)"}</Code>, which is exactly the Hopfield update. This is an algebraic identity, not an analogy: with these substitutions, the two are the same function up to floating-point noise (we measured the residual at 5e-7 in section 4.5). The differences in standard transformers are (a) <Code>K</Code> and <Code>V</Code> are produced from the same input as <Code>Q</Code> via different linear projections — self-attention rather than cross-attention to a fixed memory; (b) multi-head attention runs <Code>H</Code> independent Hopfield retrievals in parallel and concatenates them; (c) the patterns are activations from another layer, not stored parameters. None of these differences change the underlying retrieval mechanism.
      </Callout>

      <H3>Q2. Why does classical Hopfield have linear capacity but modern Hopfield has exponential capacity?</H3>

      <Callout accent="gold">
        Classical Hopfield uses a quadratic energy <Code>{"E(x) = -(1/2) x^T W x"}</Code> with <Code>W</Code> built by the Hebbian sum. The energy landscape's basins around stored patterns shrink as more patterns are added because each additional pattern contributes a quadratic term that interferes with all existing ones. Once the cumulative cross-talk dominates, basins merge and stored patterns are no longer fixed points. The Amit-Gutfreund-Sompolinsky 1985 spin-glass analysis pinned the breakdown at <Code>{"P/N \\approx 0.138"}</Code>. Modern Hopfield uses a log-sum-exp energy, which is essentially an infinite-degree polynomial and creates much sharper, well-separated basins around each stored pattern. The Krotov-Hopfield 2016 paper showed polynomial degree <Code>n</Code> gives capacity <Code>{"\\sim N^{n-1}"}</Code>; pushing to the exponential limit gives <Code>{"\\sim \\exp(d)"}</Code> capacity (Demircigil et al. 2017, Ramsauer et al. 2020). The mechanism is geometric: log-sum-exp energy has steep, narrow wells, so each pattern claims a small basin that does not interfere with its neighbors as long as patterns are reasonably spread.
      </Callout>

      <H3>Q3. A research paper claims to use a "Hopfield-based memory module" for an NLP task; you read the code and find it is implemented as <Code>nn.MultiheadAttention</Code> against a learnable parameter tensor. Is the paper wrong?</H3>

      <Callout accent="gold">
        The paper is not wrong, but the framing is doing work it shouldn't have to. <Code>nn.MultiheadAttention</Code> against a learnable parameter tensor (i.e. <Code>K</Code> and <Code>V</Code> are <Code>nn.Parameter</Code>s rather than activations) is exactly the modern Hopfield layer construction with stored patterns as parameters. Ramsauer et al.'s <Code>HopfieldLayer</Code> is essentially this. So the paper is using a Hopfield retrieval, just under a different name. Whether the framing adds value depends on whether the paper leverages the Hopfield identity for analysis — capacity bounds, temperature manipulation, energy-based interpretation, biological analogy. If the framing is purely rhetorical (citing Hopfield to add gravitas to ordinary attention), it is decorative and a reader should not expect Hopfield-specific behavior. If the paper actually uses the Hopfield apparatus for proofs or design choices, the framing is load-bearing. Verdict: not wrong, but evaluate the framing's substance, not just the citation.
      </Callout>

      <H3>Q4. Walk through the failure mode where <Code>{"\\beta"}</Code> is set too low. What happens to retrieval, gradients, and training?</H3>

      <Callout accent="gold">
        Low <Code>{"\\beta"}</Code> (e.g. 0.01 instead of <Code>{"1/\\sqrt{d}"}</Code>) makes the softmax nearly uniform: every stored pattern receives roughly equal weight regardless of its similarity to the cue. Retrieval becomes a global average over the memory bank — typically near zero for random patterns, far from any specific stored pattern. We measured this in section 9.2: at <Code>{"\\beta = 0.01"}</Code> with 500 patterns at <Code>d = 64</Code>, the relative recall error is 0.99 (essentially no recall) and the maximum softmax weight is 0.0034 (uniform over 500 patterns). For training: the gradient flows fine — softmax is differentiable everywhere — but the loss surface becomes very flat because the retrieval is largely insensitive to small changes in patterns or query. Optimization plateaus. Symptoms include training loss decreasing very slowly and validation metrics that look like the model is doing nearest-mean prediction rather than nearest-pattern retrieval. Fix: divide pre-softmax logits by <Code>{"\\sqrt{d}"}</Code> (the standard scaled-dot-product attention default), or explicitly set <Code>{"\\beta"}</Code> to <Code>{"1/\\sqrt{d}"}</Code> in the Hopfield layer config.
      </Callout>

      <H3>Q5. You are designing a few-shot classifier where each query is matched against a support set of <Code>K</Code> labeled examples. Should you frame the architecture as Hopfield retrieval, attention, or both?</H3>

      <Callout accent="gold">
        Both — they are the same architecture, and the choice of framing affects how you reason about it, not what you implement. The mechanically correct architecture is cross-attention from the query to the support set: <Code>Q</Code> from the query, <Code>K</Code> and <Code>V</Code> from the support examples (typically with the support labels concatenated to the value). At inference, this is one forward pass of <Code>F.scaled_dot_product_attention</Code>. The Hopfield framing buys you three concrete things: (1) a capacity argument — <Code>K</Code> support examples are well within the exponential-capacity regime, so the retrieval is mathematically clean; (2) a temperature dial — you may want to expose <Code>{"\\beta"}</Code> as a hyperparameter so you can sharpen retrieval at test time, especially when the support set is small and you want hard nearest-neighbor behavior; (3) an interpretability story — the attention weights over support examples are the model's "memory recall distribution" for that query, which is easy to visualize and explain to non-ML stakeholders. The attention framing buys you (1) the standard PyTorch implementation, (2) GPU-optimized FlashAttention kernels, (3) interoperability with multi-head extensions and standard transformer training recipes. In practice: write the layer using <Code>nn.MultiheadAttention</Code>, add an exposed <Code>{"\\beta"}</Code> parameter via a manual softmax if you want, and describe it in your paper as "a modern Hopfield retrieval over the support set, equivalent to scaled dot-product attention with stored patterns as keys and values." That is the cleanest way to communicate both the implementation and the conceptual content.
      </Callout>

    </div>
  ),
};

export default modernHopfieldContent;
