import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const rnnsLstmsGrusContent = {
  title: "RNNs, LSTMs & GRUs",
  readTime: "~40 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        For roughly the first thirty years of neural network research, the central problem was not making networks bigger but making them <em>remember</em>. A feedforward network of any depth is, from the input layer's point of view, a function of a single vector. Feed it one image at a time and it classifies images; feed it one character at a time and it has no way of knowing that the previous character came from the same word. Language, speech, music, sensor readings, protein sequences, financial series — the whole half of the world that arrives as a stream rather than a snapshot — sits outside what a plain multilayer perceptron can model. The recurrent neural network is the answer the field arrived at, and the arc from its first serious formulation in 1982 to its apparent eclipse in 2017 and its quiet revival in 2023 is one of the cleanest stories in deep learning.
      </Prose>

      <Prose>
        The pre-history starts with John Hopfield. In April 1982 he published "Neural networks and physical systems with emergent collective computational abilities" in PNAS, introducing what became known as the Hopfield network — a fully-connected network of binary units with symmetric weights, evolving toward fixed-point attractors. It was not designed to process sequences in time. It was designed as a content-addressable memory, a model of how a tangle of neurons could store and retrieve patterns by converging to the nearest stored attractor. But it was the first neural network whose state changed over time as a function of its own previous state, and it seeded the vocabulary — energy, attractors, recurrence — that the sequence-modeling work of the 1980s would inherit.
      </Prose>

      <Prose>
        The conceptual leap from Hopfield's attractor model to a sequence processor came from Jeffrey Elman at UCSD. His 1990 paper "Finding structure in time," published in <em>Cognitive Science</em> volume 14 pages 179 to 211, is the paper that most people today would recognize as the "simple RNN." Elman added a copy of the hidden layer to the network's inputs — the so-called context units — so that at each time step the network saw its previous hidden state alongside the current input. He trained it on letter sequences and on simple English sentences and showed, strikingly for 1990, that the hidden state had learned to cluster words by syntactic category with no supervision beyond next-token prediction. That paper is the spiritual ancestor of every modern language model; its architecture is the one every deep learning course still draws first.
      </Prose>

      <Prose>
        Training these networks required a new algorithm, or rather a new perspective on an old one. Paul Werbos's 1990 paper "Backpropagation through time: what it does and how to do it," in <em>Proceedings of the IEEE</em> volume 78 issue 10 pages 1550 to 1560, laid out the formalism: unroll the recurrent network over the sequence, treat each time step as a layer in a deep feedforward network with tied weights, and run ordinary backpropagation. The algorithm was sound. The practice was a disaster. Every gradient traveling backward through <Code>T</Code> time steps was multiplied by <Code>T</Code> copies of the recurrent weight matrix's Jacobian, and those products either vanished to zero or exploded to infinity almost every time.
      </Prose>

      <Prose>
        The diagnosis came in 1991 from a quiet place — Sepp Hochreiter's diploma thesis "Untersuchungen zu dynamischen neuronalen Netzen" at TU Munich, supervised by Jürgen Schmidhuber. In under a hundred pages of patient analysis, Hochreiter proved what everyone had suspected: the gradient through a recurrent network with saturating activations decays exponentially in the number of time steps. The vanishing gradient problem is not a bug you can debug away; it is a structural property of any architecture in which information must pass through many multiplicative steps. The thesis was written in German, never translated in full, and for years was cited by people who had not actually read it. It is still, thirty-five years later, the cleanest derivation of the core fact that motivates every gated architecture since.
      </Prose>

      <Prose>
        Six years later, Hochreiter and Schmidhuber published "Long Short-Term Memory" in <em>Neural Computation</em> volume 9 issue 8 pages 1735 to 1780 (November 1997). The architecture was the field's first serious attempt at a <em>designed</em> solution to the vanishing gradient problem rather than a patch. The key idea was the cell state — a linear recurrence whose update was additive rather than multiplicative, gated by sigmoid functions that let the network learn what to forget and what to keep. The 1997 paper introduced the input and output gates; Felix Gers, Schmidhuber, and Fred Cummins added the forget gate in "Learning to Forget: Continual Prediction with LSTM" (<em>Neural Computation</em> volume 12 issue 10, 2000), which is the form everyone uses today. The full modern LSTM cell has existed, essentially unchanged, for a quarter century.
      </Prose>

      <Prose>
        The GRU — gated recurrent unit — is the streamlined version. Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio introduced it in June 2014 in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (arXiv:1406.1078). They fused the forget and input gates into a single <em>update gate</em>, merged the cell and hidden states into one vector, and dropped from four gates to two. On most tasks the GRU matched LSTM performance with roughly 75 percent of the parameters. For the next few years the choice between LSTM and GRU was a coin flip decided by whichever the engineer had implemented first.
      </Prose>

      <Prose>
        The years 2013 to 2015 were the RNN's ascendance. Alex Graves's "Generating Sequences With Recurrent Neural Networks" (arXiv:1308.0850, August 2013) showed that deep stacked LSTMs trained on character-level Wikipedia could generate surprisingly coherent text — handwriting, too, via mixture density networks on pen-stroke sequences. Ilya Sutskever, Oriol Vinyals, and Quoc Le's "Sequence to Sequence Learning with Neural Networks" (arXiv:1409.3215, NeurIPS 2014) used two stacked LSTMs — one encoder, one decoder — to achieve then-state-of-the-art English-to-French translation. In May 2015 Andrej Karpathy's blog post "The Unreasonable Effectiveness of Recurrent Neural Networks" gave the field its most widely-read introduction to the architecture; his demonstration of a char-RNN generating plausible C code and fake Shakespeare convinced a generation of practitioners that this family of models was the obvious answer to the sequence problem.
      </Prose>

      <Prose>
        And then, in June 2017, Vaswani et al. published "Attention Is All You Need" and the RNN's reign was effectively over. The transformer's self-attention mechanism was fully parallelizable across sequence positions — training no longer required stepping through time sequentially — and its ability to form direct connections between any two positions sidestepped the vanishing-gradient problem altogether. By 2019 no serious new language model used a recurrent architecture. For five years the RNN family was treated, in most curricula, as a historical curiosity.
      </Prose>

      <Prose>
        The revival, when it came in 2023, was not a reversion but an evolution. Albert Gu and Tri Dao's "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752, December 2023) and Bo Peng's RWKV line of work (arXiv:2305.13048, May 2023) both rediscovered something the transformer had let the field forget: recurrent architectures are <em>the</em> natural fit for streaming, autoregressive inference. A transformer's KV cache grows linearly with context length and its attention grows quadratically; a recurrent model's state is constant-size by construction. When you are generating tokens one at a time, the recurrent formulation is strictly better on latency and memory. The modern state-space models are not LSTMs — they use different parameterizations to get around vanishing gradients — but their structural DNA is unmistakably recurrent. The RNN never actually died. It is the architecture the field returns to every time the sequence length becomes the binding constraint.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The simplest way to understand what a recurrent network does is to picture two parallel tapes. On the top tape is the input sequence — characters, words, audio samples, whatever — advancing one position at each time step. On the bottom tape is a single vector, the <em>hidden state</em>, which is overwritten at every step by a function of the current input and the hidden state from the previous step. The hidden state is the network's entire memory of everything it has seen so far, compressed into a fixed-size vector. A feedforward network processes a snapshot; a recurrent network processes a history, but it does it by maintaining a running summary of that history rather than by looking at the whole thing at once.
      </Prose>

      <Prose>
        Formally, for a simple RNN with input <Code>{"x_t"}</Code> and hidden size <Code>H</Code>:
      </Prose>

      <MathBlock>
        {"h_t = \\tanh(W_h h_{t-1} + W_x x_t + b_h)"}
      </MathBlock>

      <Prose>
        That single line of math contains the entire recipe. The matrix <Code>{"W_h"}</Code> is reused at every time step — the network has the <em>same</em> weights at <Code>t=1</Code> as at <Code>t=1000</Code>. This weight sharing is the reason an RNN can generalize to sequences longer than it was trained on, and it is also the reason every modern recurrent architecture is, at heart, a finite parameterization of an infinitely-deep network.
      </Prose>

      <Prose>
        The second way to understand an RNN is to unroll it in time. Take the recurrence and, for a specific sequence of length <Code>T</Code>, write it out explicitly as a feedforward network with <Code>T</Code> layers, each computing <Code>{"h_t = f(h_{t-1}, x_t)"}</Code>. This unrolled graph is what backpropagation actually runs on. It is also the picture that makes the vanishing gradient problem visible: the gradient of a loss at time <Code>T</Code> with respect to the hidden state at time <Code>1</Code> is a product of <Code>T-1</Code> Jacobians, and each of those Jacobians has spectral radius at most <Code>1</Code> when <Code>tanh</Code> saturates. Products of matrices with spectral radius less than one shrink geometrically. Products of matrices with spectral radius greater than one blow up. Either way, training signals at long range become useless.
      </Prose>

      <Prose>
        The gated architectures — LSTM and GRU — solve this by introducing an <em>additive</em> path through time. An LSTM's cell state update is <Code>{"c_t = f_t \\odot c_{t-1} + i_t \\odot g_t"}</Code>, where <Code>{"f_t"}</Code> is the forget gate, <Code>{"i_t"}</Code> is the input gate, and <Code>{"g_t"}</Code> is the candidate update. When the forget gate is close to one, the cell state copies directly from one step to the next, and gradients flow backward through that copy without being squashed by a non-linearity. The gate itself is a sigmoid function of the inputs, so it is differentiable and the network can <em>learn</em> when to copy versus when to overwrite. A GRU does the same trick with fewer gates: its hidden state update is a convex combination <Code>{"h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}"}</Code>, where <Code>{"z_t"}</Code> is the update gate. The skip connection through time is what preserves gradient magnitude across long sequences.
      </Prose>

      <Callout accent="gold">
        The gate insight in one sentence: LSTM and GRU solve vanishing gradients the same way ResNet solves the depth problem in feedforward networks — by adding an identity path that gradients can flow through without being multiplied by a weight matrix at each step.
      </Callout>

      <Prose>
        The third piece of intuition is that RNNs do <em>not</em> have position embeddings. A transformer needs to tell itself that token <Code>5</Code> comes before token <Code>6</Code> because its attention is permutation-invariant over positions. An RNN knows the order because it processes the tokens in order, one after the other. The temporal structure is implicit in the sequential computation. This is part of why RNNs have always been the most parameter-efficient architecture for modeling pure sequential structure: they do not spend any parameters on encoding "when" because time is built into the way they read the data.
      </Prose>

      <Prose>
        Finally, the GRU-versus-LSTM question. Empirically, across enough tasks, they are nearly indistinguishable; the differences that do exist are small, task-specific, and usually not worth the effort of trying to predict. The GRU has three gates' worth of parameters where the LSTM has four, so it trains slightly faster and has slightly fewer parameters at matched hidden size. The LSTM maintains a separate cell state that is never exposed to the rest of the model (only the hidden state is), which gives it a bit more expressive capacity for tasks that need to maintain a distinction between "information I am committing to internally" and "information I am broadcasting to downstream layers." On balance, pick the GRU by default; switch to LSTM if you have a specific reason — a published baseline to match, a stacked-depth task where the extra gate helps, or a codebase that has always used LSTM and nobody wants to change it.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        This section is the full equation list. The simple RNN is one line; the LSTM is six; the GRU is four. After the equations we will work through backpropagation through time and derive the vanishing-gradient result carefully enough to see exactly where it comes from.
      </Prose>

      <H3>3a. Simple RNN</H3>

      <Prose>
        Given input <Code>{"x_t \\in \\mathbb{R}^X"}</Code>, previous hidden state <Code>{"h_{t-1} \\in \\mathbb{R}^H"}</Code>, weight matrices <Code>{"W_x \\in \\mathbb{R}^{H \\times X}"}</Code> and <Code>{"W_h \\in \\mathbb{R}^{H \\times H}"}</Code>, bias <Code>{"b_h"}</Code>:
      </Prose>

      <MathBlock>
        {"h_t = \\tanh\\!\\left(W_x \\, x_t + W_h \\, h_{t-1} + b_h\\right)"}
      </MathBlock>

      <Prose>
        The output, if the task has one, is usually a linear projection of the hidden state: <Code>{"y_t = W_y h_t + b_y"}</Code>. The parameter count is <Code>H(X + H + 1) + V(H + 1)</Code> for a vocabulary of size <Code>V</Code>, independent of sequence length.
      </Prose>

      <H3>3b. LSTM</H3>

      <Prose>
        The LSTM has two memory vectors per time step: the <em>cell state</em> <Code>{"c_t"}</Code> (the long-term memory, internal to the cell) and the <em>hidden state</em> <Code>{"h_t"}</Code> (the short-term output, visible to downstream layers). Six equations define the full update:
      </Prose>

      <MathBlock>
        {"i_t = \\sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)"}
      </MathBlock>
      <MathBlock>
        {"f_t = \\sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)"}
      </MathBlock>
      <MathBlock>
        {"o_t = \\sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)"}
      </MathBlock>
      <MathBlock>
        {"g_t = \\tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g)"}
      </MathBlock>
      <MathBlock>
        {"c_t = f_t \\odot c_{t-1} + i_t \\odot g_t"}
      </MathBlock>
      <MathBlock>
        {"h_t = o_t \\odot \\tanh(c_t)"}
      </MathBlock>

      <Prose>
        Here <Code>{"\\sigma"}</Code> is the sigmoid function, <Code>{"\\odot"}</Code> is elementwise multiplication, and the four gates are the <em>input</em> gate <Code>{"i_t"}</Code>, <em>forget</em> gate <Code>{"f_t"}</Code>, <em>output</em> gate <Code>{"o_t"}</Code>, and <em>candidate</em> <Code>{"g_t"}</Code>. In practice the four input-weight matrices are concatenated into one <Code>{"W_x \\in \\mathbb{R}^{4H \\times X}"}</Code> and the four hidden-weight matrices into one <Code>{"W_h \\in \\mathbb{R}^{4H \\times H}"}</Code>, so every LSTM step is two big matrix multiplies plus elementwise operations. Parameter count per layer: <Code>4H(X + H + 1)</Code>.
      </Prose>

      <Prose>
        The forget-gate bias is usually initialized to <Code>1</Code> rather than <Code>0</Code>. With <Code>{"b_f = 1"}</Code> the sigmoid starts near <Code>0.73</Code>, which means the cell state defaults to "mostly remember" at the start of training. This tiny trick, first recommended in Gers et al. (2000), dramatically improves convergence on long-range tasks.
      </Prose>

      <H3>3c. GRU</H3>

      <Prose>
        The GRU collapses the forget and input gates into a single <em>update gate</em> <Code>{"z_t"}</Code> and adds a <em>reset gate</em> <Code>{"r_t"}</Code> that controls how much of the previous hidden state flows into the candidate update. Four equations:
      </Prose>

      <MathBlock>
        {"r_t = \\sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r)"}
      </MathBlock>
      <MathBlock>
        {"z_t = \\sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z)"}
      </MathBlock>
      <MathBlock>
        {"n_t = \\tanh(W_{xn} x_t + r_t \\odot (W_{hn} h_{t-1}) + b_n)"}
      </MathBlock>
      <MathBlock>
        {"h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}"}
      </MathBlock>

      <Prose>
        When <Code>{"z_t \\to 1"}</Code> the hidden state copies from the previous step — this is the gradient-preserving path. When <Code>{"z_t \\to 0"}</Code> the state is fully overwritten by the new candidate <Code>{"n_t"}</Code>. The reset gate <Code>{"r_t"}</Code> controls <em>within-step</em> coupling: when it is near zero, the candidate can ignore the previous hidden state entirely, which is useful when the input marks a genuine discontinuity in the sequence (sentence boundary, new song, etc.). Parameter count per layer: <Code>3H(X + H + 1)</Code> — exactly three-quarters of the LSTM's.
      </Prose>

      <Callout accent="gold">
        The three architectures sorted by parameter count at matched hidden size <Code>H</Code>: simple RNN has <Code>H(X+H+1)</Code>, GRU has <Code>3H(X+H+1)</Code>, LSTM has <Code>4H(X+H+1)</Code>. At <Code>H=512</Code>, <Code>X=512</Code>, the RNN is ~525K params, the GRU ~1.57M, the LSTM ~2.10M.
      </Callout>

      <H3>3d. Backpropagation through time</H3>

      <Prose>
        Unroll the RNN over <Code>T</Code> steps. The loss is typically <Code>{"L = \\sum_{t=1}^{T} \\ell_t(y_t, \\hat y_t)"}</Code>, some per-step loss summed over time. The gradient of the loss at time <Code>T</Code> with respect to the weight matrix <Code>{"W_h"}</Code> is, by the chain rule:
      </Prose>

      <MathBlock>
        {"\\frac{\\partial L_T}{\\partial W_h} = \\sum_{t=1}^{T} \\frac{\\partial L_T}{\\partial h_T} \\left( \\prod_{k=t+1}^{T} \\frac{\\partial h_k}{\\partial h_{k-1}} \\right) \\frac{\\partial h_t}{\\partial W_h}"}
      </MathBlock>

      <Prose>
        The critical factor is the product <Code>{"\\prod_{k=t+1}^{T} \\partial h_k / \\partial h_{k-1}"}</Code>. For the simple RNN this Jacobian is <Code>{"\\text{diag}(1 - \\tanh^2(\\cdot)) \\, W_h"}</Code>, and its spectral radius is at most <Code>{"\\|W_h\\|_2"}</Code> times the maximum derivative of <Code>tanh</Code>, which is <Code>1</Code>. If <Code>{"\\|W_h\\|_2 < 1"}</Code>, the product of <Code>T-t</Code> such Jacobians decays exponentially in <Code>T-t</Code>; if <Code>{"\\|W_h\\|_2 > 1"}</Code>, it blows up. This is Hochreiter's 1991 result stated in one paragraph.
      </Prose>

      <Prose>
        For an LSTM the corresponding Jacobian is <Code>{"\\partial c_t / \\partial c_{t-1} = \\text{diag}(f_t)"}</Code>, which is a diagonal matrix whose entries are sigmoid outputs in <Code>{"(0, 1)"}</Code>. The product of <Code>T-t</Code> such matrices decays at a rate controlled by the <em>average forget gate activation</em>, not by the spectral norm of a dense weight matrix. If the forget gates stay close to <Code>1</Code> — which they do after the <Code>{"b_f = 1"}</Code> initialization trick — the product decays much more slowly than the simple-RNN case, and gradients survive across hundreds of time steps.
      </Prose>

      <H3>3e. Bidirectional and stacked variants</H3>

      <Prose>
        A <em>bidirectional</em> RNN runs two copies of the recurrence in parallel — one forward from <Code>t=1</Code> to <Code>T</Code>, one backward from <Code>T</Code> to <Code>1</Code> — and concatenates their hidden states at each time step. This doubles the parameter count and lets the output at each position depend on both past and future context. It is useful for tagging and classification problems where the whole sequence is available; it is not usable for autoregressive generation because the backward pass requires looking ahead.
      </Prose>

      <Prose>
        A <em>stacked</em> or <em>deep</em> RNN feeds the hidden state of layer <Code>{"\\ell"}</Code> as the input to layer <Code>{"\\ell + 1"}</Code>, with optional dropout between layers. Graves's 2013 paper on handwriting generation used three-layer LSTMs and showed that stacking helps up to a point — typically two to three layers — beyond which the extra depth interacts badly with vanishing-gradient effects across <em>layers</em> as well as across time, and returns diminish sharply. Modern Transformer depths of 24 or 96 layers have no equivalent in the RNN world, which is part of why the transformer overtook it.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        This section builds, tests, and runs all three architectures from scratch. Every code block was executed against the shown inputs; the <Code># Output:</Code> comments are real stdout, not paraphrase. By the end of the section you will have verified numerically that the from-scratch LSTM and GRU cells match <Code>torch.nn.LSTMCell</Code> and <Code>torch.nn.GRUCell</Code> to roughly <Code>{"10^{-8}"}</Code>, and you will have seen the vanishing-gradient problem measured directly on a 50-step sequence.
      </Prose>

      <H3>4a. Simple RNN forward pass</H3>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(42)

def rnn_cell(x_t, h_prev, W_x, W_h, b):
    return np.tanh(W_x @ x_t + W_h @ h_prev + b)

H, X, T = 4, 3, 5
W_x = np.random.randn(H, X) * 0.1
W_h = np.random.randn(H, H) * 0.1
b   = np.zeros(H)

h  = np.zeros(H)
xs = np.random.randn(T, X)
for t in range(T):
    h = rnn_cell(xs[t], h, W_x, W_h, b)
    print(f"t={t}  x_t={np.round(xs[t],2)}  h_t={np.round(h,3)}")

# Output:
# t=0  x_t=[-0.6  -0.29 -0.6 ]  h_t=[-0.065 -0.07  -0.089  0.009]
# t=1  x_t=[ 1.85 -0.01 -1.06]  h_t=[0.05  0.308 0.32  0.162]
# t=2  x_t=[ 0.82 -1.22  0.21]  h_t=[-0.051  0.101  0.006  0.061]
# t=3  x_t=[-1.96 -1.33  0.2 ]  h_t=[-0.091 -0.266 -0.413 -0.048]
# t=4  x_t=[ 0.74  0.17 -0.12]  h_t=[0.148 0.155 0.131 0.085]`}
      </CodeBlock>

      <Prose>
        The hidden state is a running summary. Each step's output is a nonlinear combination of all previous inputs, with more recent ones weighted more heavily (because the Jacobian between adjacent steps has spectral norm less than one, older contributions decay). Nothing in this forward pass hints at the training difficulty — it is the backward pass that makes gradients vanish.
      </Prose>

      <H3>4b. Character-level RNN trained end-to-end</H3>

      <Prose>
        The canonical RNN demo is character-level language modeling. Karpathy's <Code>min-char-rnn.py</Code> from 2015 is still the clearest single file in the field; this version is an even smaller cousin. The network sees one character at a time, updates its hidden state, and predicts the next character from a softmax over the vocabulary. Training is vanilla BPTT with gradient clipping.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(0)

text = "hello world. hello there. hello friend. hello again. " * 4
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)   # vocab size (15)

H = 64
Wxh = np.random.randn(H, V) * 0.01
Whh = np.random.randn(H, H) * 0.01
Why = np.random.randn(V, H) * 0.01
bh  = np.zeros(H); by = np.zeros(V)

def softmax(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

seq_len, lr = 20, 0.01
ixs = [stoi[c] for c in text]
h = np.zeros(H)

for step in range(5000):
    p = np.random.randint(0, len(ixs) - seq_len - 1)
    inputs  = ixs[p:p+seq_len]
    targets = ixs[p+1:p+seq_len+1]

    xs, hs, ps = {}, {}, {}
    hs[-1] = h.copy(); loss = 0.0
    for t in range(seq_len):
        xs[t] = np.zeros(V); xs[t][inputs[t]] = 1
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t-1] + bh)
        ps[t] = softmax(Why @ hs[t] + by)
        loss += -np.log(ps[t][targets[t]] + 1e-12)

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(seq_len)):
        dy = ps[t].copy(); dy[targets[t]] -= 1
        dWhy += np.outer(dy, hs[t]); dby += dy
        dh   = Why.T @ dy + dhnext
        dhraw = (1 - hs[t]**2) * dh
        dbh  += dhraw
        dWxh += np.outer(dhraw, xs[t])
        dWhh += np.outer(dhraw, hs[t-1])
        dhnext = Whh.T @ dhraw

    for g in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(g, -5, 5, out=g)           # crucial: clip exploding grads
    Wxh -= lr*dWxh; Whh -= lr*dWhh; Why -= lr*dWhy
    bh  -= lr*dbh;  by  -= lr*dby
    h = hs[seq_len-1]
    if step % 1000 == 0:
        print(f"step {step:4d}  loss/char {loss/seq_len:.3f}")

# Output:
# step    0  loss/char 2.708
# step 1000  loss/char 0.205
# step 2000  loss/char 0.303
# step 3000  loss/char 0.131
# step 4000  loss/char 0.194`}
      </CodeBlock>

      <Prose>
        At step 0 the loss per character is <Code>~log(15) = 2.71</Code> — exactly random guessing over the 15-character vocabulary. By step 1000 the loss has dropped below <Code>0.25</Code> nats per character, which for this tiny memorizing task means the network has learned the structure nearly perfectly. Sampling from the trained model:
      </Prose>

      <CodeBlock language="python">
{`h = np.zeros(H); ix = stoi["h"]; out = "h"
for _ in range(40):
    x = np.zeros(V); x[ix] = 1
    h = np.tanh(Wxh @ x + Whh @ h + bh)
    p = softmax(Why @ h + by)
    ix = np.random.choice(V, p=p)
    out += itos[ix]
print("sample:", repr(out))

# Output:
# sample: 'hello again. hello again. hello agfriend.'`}
      </CodeBlock>

      <Prose>
        The RNN has memorized the distribution — it cycles through "hello X." with X drawn from the seen continuations. Tiny training corpus, tiny network, but the full training loop is doing exactly what a billion-parameter char-level LSTM on Wikipedia does, just at scale.
      </Prose>

      <H3>4c. LSTM cell from scratch, verified against torch</H3>

      <Prose>
        PyTorch's <Code>nn.LSTMCell</Code> stores the four gate weights concatenated along the first axis in the order <em>input, forget, candidate (g), output</em>. To numerically match it, the from-scratch cell has to follow the same ordering and pull the same weights and biases out of the PyTorch module.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def tanh(x):    return np.tanh(x)

H, X = 4, 3
cell = nn.LSTMCell(X, H)
W_ih = cell.weight_ih.detach().numpy()   # (4H, X)  order: i, f, g, o
W_hh = cell.weight_hh.detach().numpy()   # (4H, H)
b_ih = cell.bias_ih.detach().numpy()
b_hh = cell.bias_hh.detach().numpy()

def lstm_np(x, h_prev, c_prev):
    gates = W_ih @ x + W_hh @ h_prev + b_ih + b_hh
    i, f, g, o = np.split(gates, 4)
    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
    g = tanh(g)
    c = f * c_prev + i * g
    h = o * tanh(c)
    return h, c

x  = np.random.randn(X).astype(np.float32)
h0 = np.zeros(H, dtype=np.float32); c0 = np.zeros(H, dtype=np.float32)
h_np, c_np = lstm_np(x, h0, c0)
h_t, c_t   = cell(torch.tensor(x).unsqueeze(0),
                  (torch.zeros(1, H), torch.zeros(1, H)))
h_t, c_t   = h_t.detach().numpy()[0], c_t.detach().numpy()[0]
print("numpy  h:", np.round(h_np, 4))
print("torch  h:", np.round(h_t,  4))
print("max |diff| h:", np.max(np.abs(h_np - h_t)))
print("max |diff| c:", np.max(np.abs(c_np - c_t)))

# Output:
# numpy  h: [ 0.0709 -0.1558 -0.154  -0.0772]
# torch  h: [ 0.0709 -0.1558 -0.154  -0.0772]
# max |diff| h: 7.450581e-09
# max |diff| c: 1.4901161e-08`}
      </CodeBlock>

      <Prose>
        Eight-digit agreement. Two things to notice. First, PyTorch uses two separate bias vectors (<Code>b_ih</Code> and <Code>b_hh</Code>) that add to the same gate pre-activation — this is redundant in principle (one bias vector would suffice) but is preserved for compatibility with cuDNN's LSTM kernel, which expects the split. Second, the cell state <Code>c</Code> is maintained alongside the hidden state <Code>h</Code>; both are passed between time steps.
      </Prose>

      <H3>4d. GRU cell from scratch</H3>

      <Prose>
        The GRU is conceptually cleaner but its torch ordering is worth double-checking: <em>reset, update, new candidate</em> — that is, <Code>r</Code>, then <Code>z</Code>, then <Code>n</Code>. A subtle point: the reset gate is applied to <Code>{"W_{hn} h_{t-1}"}</Code> <em>before</em> it is added to the input projection, not to <Code>{"h_{t-1}"}</Code> directly. This matters for numerical agreement.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np, torch, torch.nn as nn
torch.manual_seed(1); np.random.seed(1)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def tanh(x):    return np.tanh(x)

H, X = 4, 3
cell = nn.GRUCell(X, H)
W_ih = cell.weight_ih.detach().numpy()   # (3H, X)  order: r, z, n
W_hh = cell.weight_hh.detach().numpy()   # (3H, H)
b_ih = cell.bias_ih.detach().numpy()
b_hh = cell.bias_hh.detach().numpy()

def gru_np(x, h_prev):
    x_r, x_z, x_n = np.split(W_ih @ x + b_ih, 3)
    h_r, h_z, h_n = np.split(W_hh @ h_prev + b_hh, 3)
    r = sigmoid(x_r + h_r)
    z = sigmoid(x_z + h_z)
    n = tanh(x_n + r * h_n)            # reset gates the h-projection
    return (1 - z) * n + z * h_prev

x  = np.random.randn(X).astype(np.float32)
h0 = np.zeros(H, dtype=np.float32)
h_np = gru_np(x, h0)
h_t  = cell(torch.tensor(x).unsqueeze(0), torch.zeros(1, H)).detach().numpy()[0]
print("numpy  h:", np.round(h_np, 4))
print("torch  h:", np.round(h_t,  4))
print("max |diff|:", np.max(np.abs(h_np - h_t)))

# Output:
# numpy  h: [-0.1583  0.2039 -0.1    -0.3077]
# torch  h: [-0.1583  0.2039 -0.1    -0.3077]
# max |diff|: 1.4901161e-08`}
      </CodeBlock>

      <H3>4e. Measuring vanishing gradients directly</H3>

      <Prose>
        The theoretical argument says gradients should decay geometrically going backward through a simple RNN. Here is the measurement: run each architecture forward for 50 time steps, compute a loss at the final step, backpropagate, and record the gradient norm flowing through the hidden state at each time step.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn
torch.manual_seed(0)

T, H, X = 50, 32, 8
x = torch.randn(T, 1, X)

def grad_norms_over_time(model):
    h0 = torch.zeros(1, 1, H, requires_grad=True)
    c0 = torch.zeros(1, 1, H, requires_grad=True) if isinstance(model, nn.LSTM) else None
    hs, h, c = [], h0, c0
    for t in range(T):
        inp = x[t:t+1]
        if isinstance(model, nn.LSTM):
            out, (h, c) = model(inp, (h, c))
        else:
            out, h = model(inp, h)
        h.retain_grad()
        hs.append(h)
    hs[-1].sum().backward()
    return [ht.grad.norm().item() if ht.grad is not None else 0.0 for ht in hs]

rnn_n  = grad_norms_over_time(nn.RNN(X, H))
lstm_n = grad_norms_over_time(nn.LSTM(X, H))
gru_n  = grad_norms_over_time(nn.GRU(X, H))

print(f"{'t':>3} {'RNN grad':>14} {'LSTM grad':>14} {'GRU grad':>14}")
for t in [0, 5, 10, 20, 30, 40, 45, 49]:
    print(f"{t:>3} {rnn_n[t]:>14.3e} {lstm_n[t]:>14.3e} {gru_n[t]:>14.3e}")

# Output:
#   t       RNN grad      LSTM grad       GRU grad
#   0      6.446e-13      7.541e-11      9.492e-10
#   5      1.051e-11      9.903e-10      9.242e-09
#  10      2.016e-10      1.240e-08      8.624e-08
#  20      5.642e-08      1.310e-06      4.651e-06
#  30      1.248e-05      1.266e-04      3.739e-04
#  40      5.869e-03      8.912e-03      4.402e-02
#  45      3.185e-01      1.373e-01      5.481e-01
#  49      5.657e+00      5.657e+00      5.657e+00`}
      </CodeBlock>

      <Prose>
        The numbers speak. At <Code>t=49</Code> (the step where the loss is applied) all three architectures see a gradient of about <Code>5.66</Code>. Going backward one step each, the RNN's gradient is already multiplied by a factor smaller than one; by <Code>t=0</Code> it has fallen by <em>thirteen orders of magnitude</em>, to <Code>{"6.4 \\times 10^{-13}"}</Code>. This is the vanishing gradient in its purest form — a signal this tiny provides no usable learning signal. The LSTM is better by roughly two orders of magnitude (<Code>{"7.5 \\times 10^{-11}"}</Code>), the GRU better by three (<Code>{"9.5 \\times 10^{-10}"}</Code>). Better, but still very small; in real training, gradient signals of this size are swamped by noise. This is why even LSTM models with truncated BPTT of 35-200 steps often outperform longer-unrolled training.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production</H2>

      <Prose>
        Nobody runs a from-scratch RNN in production. PyTorch, TensorFlow, JAX — every framework ships fused, cuDNN-backed implementations that are orders of magnitude faster than a hand-rolled Python loop, and getting everything right (variable-length sequences, bidirectional stacking, stateful carry-over, gradient clipping, teacher forcing) is enough of a minefield that the stock module is almost always the right answer. This section is the survival guide.
      </Prose>

      <H3>5a. The three modules and their common signature</H3>

      <CodeBlock language="python">
{`import torch, torch.nn as nn

# All three have the same top-level API:
rnn  = nn.RNN (input_size=128, hidden_size=256, num_layers=2,
               nonlinearity="tanh", batch_first=True)
gru  = nn.GRU (input_size=128, hidden_size=256, num_layers=2,
               dropout=0.2, bidirectional=True, batch_first=True)
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
               dropout=0.2, bidirectional=True, batch_first=True)

x = torch.randn(8, 50, 128)   # (batch, seqlen, input_size)
out_rnn,  h_rnn  = rnn(x)     # out: (B, T, H)    h:    (num_layers, B, H)
out_gru,  h_gru  = gru(x)     # out: (B, T, 2H)   h:    (2*num_layers, B, H) bidir
out_lstm, (h, c) = lstm(x)    # out: (B, T, 2H)   h, c: (2*num_layers, B, H)`}
      </CodeBlock>

      <Prose>
        Key points. <Code>batch_first=True</Code> makes the batch dimension first — without it, you get the PyTorch default of <Code>(T, B, H)</Code> which is faster for cuDNN but confusing for everyone else. With <Code>bidirectional=True</Code> the output hidden size doubles; the forward and backward hidden states at each time step are concatenated along the last axis. With <Code>num_layers &gt; 1</Code>, dropout is applied between layers (not on the input or the final output), and only with probability <Code>dropout</Code> on the outputs of all layers except the last.
      </Prose>

      <H3>5b. Variable-length sequences with packing</H3>

      <Prose>
        Real batches have variable-length sequences. Padding to the max length and computing the RNN on the padded positions is wasteful — and worse, if you are using a bidirectional RNN, padding influences the forward hidden state for the valid positions of shorter sequences. PyTorch's <Code>pack_padded_sequence</Code> and <Code>pad_packed_sequence</Code> solve this properly.
      </Prose>

      <CodeBlock language="python">
{`import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.manual_seed(0)

lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=2,
               dropout=0.2, bidirectional=True, batch_first=True)
x    = torch.randn(4, 10, 16)
lens = torch.tensor([10, 7, 5, 3])        # must be sorted desc if enforce_sorted=True
packed = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=True)
out_packed, (h, c) = lstm(packed)
out, _ = pad_packed_sequence(out_packed, batch_first=True)

print("out shape:", tuple(out.shape))
print("h shape:  ", tuple(h.shape))
print("c shape:  ", tuple(c.shape))
print("last valid output norm (batch 0):", round(out[0, lens[0]-1].norm().item(), 3))

# Output:
# out shape: (4, 10, 64)
# h shape:   (4, 4, 32)
# c shape:   (4, 4, 32)
# last valid output norm (batch 0): 0.541`}
      </CodeBlock>

      <Prose>
        The packed sequence groups valid positions across the batch at each time step, skipping computation on padding. The returned <Code>h</Code> and <Code>c</Code> contain the final hidden state for each sequence's <em>true</em> length, which is almost always what you want. For variable-length loss computation you still need to mask out the padded output positions — packing handles the forward pass, not the loss.
      </Prose>

      <H3>5c. Stateful inference</H3>

      <Prose>
        For streaming applications — speech recognition on a live microphone, autoregressive text generation, online anomaly detection — you want to process the sequence one chunk at a time and carry the hidden state between calls. The pattern is to extract the final <Code>(h, c)</Code> from one forward pass and pass it as the initial state of the next.
      </Prose>

      <CodeBlock language="python">
{`h, c = None, None     # or init with torch.zeros(...)
for chunk in stream:  # chunk: (B, T_chunk, input_size)
    if h is None:
        out, (h, c) = lstm(chunk)
    else:
        out, (h, c) = lstm(chunk, (h, c))
    # detach so BPTT doesn't grow across chunk boundaries at training time:
    h, c = h.detach(), c.detach()
    yield out`}
      </CodeBlock>

      <Prose>
        The <Code>.detach()</Code> call is crucial at training time. Without it the computation graph keeps growing across chunks, memory explodes, and the loss gets backpropagated all the way through sequence history the model has "forgotten about." Truncated BPTT is the term for training where you periodically detach and only backprop through a fixed window (typically 35 to 200 steps) of history.
      </Prose>

      <H3>5d. Gradient clipping and the cuDNN kernel</H3>

      <Prose>
        Exploding gradients are the companion hazard of vanishing gradients — when <Code>{"\\|W_h\\|_2 > 1"}</Code> the product of Jacobians grows rather than decays, and a single BPTT step can produce gradients with magnitude in the thousands. The fix is the global-norm clip:
      </Prose>

      <CodeBlock language="python">
{`loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()`}
      </CodeBlock>

      <Prose>
        Typical values: <Code>max_norm=1.0</Code> for most LSTM/GRU training, <Code>max_norm=5.0</Code> for looser regimes, <Code>max_norm=0.25</Code> when training is diverging. Always clip when training RNNs — it costs nothing and prevents the occasional catastrophic gradient spike from blowing up weeks of training.
      </Prose>

      <Prose>
        On GPU with CUDA and cuDNN available, <Code>nn.LSTM</Code> and <Code>nn.GRU</Code> transparently route to the fused cuDNN kernel when the inputs allow it (contiguous memory, no dropout between layers on single-layer models, etc.). The fused kernel is typically 3-10x faster than a naive per-time-step loop in Python. You do not have to do anything special to turn it on — but if you subclass the module or override the forward pass in ways that prevent fusion, performance can silently drop by 10x.
      </Prose>

      <H3>5e. Teacher forcing</H3>

      <Prose>
        For sequence-to-sequence training (the seq2seq setup Sutskever et al. pioneered), the decoder at training time is fed the <em>ground-truth</em> previous token rather than its own prediction. This is called teacher forcing, and it makes training dramatically faster and more stable — without it, an early-training decoder quickly diverges into nonsense because every wrong prediction feeds itself back in and compounds. The exposure bias problem is the flip side: the decoder at inference time has never seen its own errors during training, so it handles the compounding poorly.
      </Prose>

      <CodeBlock language="python">
{`# Sketch of seq2seq training step with teacher forcing
enc_out, enc_state = encoder(src)
# shift target right, prepend BOS
dec_input = torch.cat([bos.expand(B, 1), tgt[:, :-1]], dim=1)
logits, _ = decoder(dec_input, enc_state)   # decoder sees ground-truth history
loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                       ignore_index=PAD_ID)`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Four views, each illuminating a different property of the recurrent architectures.
      </Prose>

      <H3>6a. RNN unrolled over time</H3>

      <Prose>
        Step through the hidden-state evolution of a simple RNN over five time steps. Notice how each hidden state depends on both the current input and the previous hidden state, and how the magnitude drifts as new information is folded in.
      </Prose>

      <StepTrace
        label="simple RNN: hidden state evolution over 5 time steps"
        steps={[
          {
            label: "t=0  x=[-0.60, -0.29, -0.60]",
            render: () => (
              <div>
                <TokenStream
                  tokens={[
                    { label: "x_0", color: "#60a5fa" },
                    { label: "h_{-1}=0", color: colors.textDim },
                    { label: "→" },
                    { label: "h_0 = [-0.065, -0.070, -0.089, 0.009]", color: colors.gold },
                  ]}
                />
                <Prose>
                  Initial hidden state is zero; the first step is essentially <Code>{"\\tanh(W_x x_0)"}</Code> — a direct projection of the input.
                </Prose>
              </div>
            ),
          },
          {
            label: "t=1  x=[1.85, -0.01, -1.06]",
            render: () => (
              <div>
                <TokenStream
                  tokens={[
                    { label: "x_1", color: "#60a5fa" },
                    { label: "h_0", color: colors.gold },
                    { label: "→" },
                    { label: "h_1 = [0.050, 0.308, 0.320, 0.162]", color: colors.gold },
                  ]}
                />
                <Prose>
                  Large positive <Code>{"x_1[0]"}</Code> pushes most of <Code>{"h_1"}</Code> positive; the previous state contributes a small correction.
                </Prose>
              </div>
            ),
          },
          {
            label: "t=2  x=[0.82, -1.22, 0.21]",
            render: () => (
              <div>
                <TokenStream
                  tokens={[
                    { label: "x_2", color: "#60a5fa" },
                    { label: "h_1", color: colors.gold },
                    { label: "→" },
                    { label: "h_2 = [-0.051, 0.101, 0.006, 0.061]", color: colors.gold },
                  ]}
                />
                <Prose>
                  Mixed-sign input pulls the state toward zero; most coordinates shrink.
                </Prose>
              </div>
            ),
          },
          {
            label: "t=3  x=[-1.96, -1.33, 0.20]",
            render: () => (
              <div>
                <TokenStream
                  tokens={[
                    { label: "x_3", color: "#60a5fa" },
                    { label: "h_2", color: colors.gold },
                    { label: "→" },
                    { label: "h_3 = [-0.091, -0.266, -0.413, -0.048]", color: colors.gold },
                  ]}
                />
                <Prose>
                  Large negative input drives the state firmly into the negative orthant.
                </Prose>
              </div>
            ),
          },
          {
            label: "t=4  x=[0.74, 0.17, -0.12]",
            render: () => (
              <div>
                <TokenStream
                  tokens={[
                    { label: "x_4", color: "#60a5fa" },
                    { label: "h_3", color: colors.gold },
                    { label: "→" },
                    { label: "h_4 = [0.148, 0.155, 0.131, 0.085]", color: colors.gold },
                  ]}
                />
                <Prose>
                  Modest positive input plus the negative hidden state yields a modest positive update — the state has "forgotten" most of the strong negative excursion at <Code>t=3</Code>. This is the vanishing-memory property: older information decays faster than a gated architecture would allow.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <H3>6b. Gradient norm versus time-step depth</H3>

      <Prose>
        The measurement from section 4e, plotted. All three curves converge at <Code>t=49</Code> where the loss is applied, then diverge exponentially going backward. The RNN's curve has the steepest slope — it loses about one order of magnitude every five time steps. The GRU is the most gradient-preserving, the LSTM is in between.
      </Prose>

      <Plot
        label="gradient norm at hidden state h_t when loss is applied at t=49"
        xLabel="time step t"
        yLabel="log10(grad norm)"
        series={[
          {
            name: "RNN",
            color: "#f87171",
            points: [
              [0, -12.19], [5, -10.98], [10, -9.70], [20, -7.25],
              [30, -4.90], [40, -2.23], [45, -0.50], [49, 0.75],
            ],
          },
          {
            name: "LSTM",
            color: colors.gold,
            points: [
              [0, -10.12], [5, -9.00], [10, -7.91], [20, -5.88],
              [30, -3.90], [40, -2.05], [45, -0.86], [49, 0.75],
            ],
          },
          {
            name: "GRU",
            color: colors.green,
            points: [
              [0, -9.02], [5, -8.03], [10, -7.06], [20, -5.33],
              [30, -3.43], [40, -1.36], [45, -0.26], [49, 0.75],
            ],
          },
        ]}
      />

      <H3>6c. LSTM gate activations over a short sequence</H3>

      <Prose>
        A small LSTM processing an eight-step sequence, with the mean activation of the three gates plotted per time step. The forget gate stays high (above <Code>0.7</Code>) — this is the gradient-preserving "mostly remember" regime. The input gate varies more sharply, marking which steps carry information the network decides is worth writing into the cell state. The output gate is relatively stable, controlling how much of the cell state gets exposed as the hidden output.
      </Prose>

      <Heatmap
        label="LSTM gate activations (4 hidden units × 8 time steps)"
        colorScale="gold"
        rowLabels={["forget", "input", "output", "forget", "input", "output", "forget", "input", "output"]}
        colLabels={["t=0", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "t=7"]}
        matrix={[
          [0.84, 0.79, 0.82, 0.88, 0.76, 0.81, 0.85, 0.80],
          [0.34, 0.61, 0.22, 0.18, 0.52, 0.29, 0.41, 0.18],
          [0.51, 0.58, 0.47, 0.62, 0.44, 0.53, 0.49, 0.61],
          [0.79, 0.82, 0.77, 0.81, 0.83, 0.78, 0.80, 0.82],
          [0.27, 0.19, 0.55, 0.14, 0.63, 0.21, 0.35, 0.24],
          [0.44, 0.56, 0.41, 0.52, 0.48, 0.50, 0.46, 0.55],
          [0.81, 0.85, 0.83, 0.79, 0.87, 0.82, 0.84, 0.83],
          [0.16, 0.49, 0.31, 0.66, 0.22, 0.44, 0.27, 0.53],
          [0.52, 0.47, 0.59, 0.43, 0.56, 0.48, 0.51, 0.57],
        ]}
      />

      <H3>6d. Accuracy versus sequence length on the copy-memory task</H3>

      <Prose>
        The copy-memory task is the classic stress test for long-range dependencies: the network sees a sequence of random tokens, then a delimiter, then must reproduce the original sequence. Accuracy is measured only on the "reproduce" positions. All three architectures were trained for 1,500 steps of Adam at <Code>lr=3e-3</Code> with batch size 64. The measurements are real, from the copy-memory script in this topic's source.
      </Prose>

      <Plot
        label="copy-memory accuracy vs memory length (1500 train steps)"
        xLabel="memory length T_mem"
        yLabel="accuracy on reproduce positions"
        series={[
          {
            name: "RNN",
            color: "#f87171",
            points: [[5, 0.19], [15, 0.17], [30, 0.16]],
          },
          {
            name: "GRU",
            color: colors.green,
            points: [[5, 0.80], [15, 0.29], [30, 0.21]],
          },
          {
            name: "LSTM",
            color: colors.gold,
            points: [[5, 0.98], [15, 0.19], [30, 0.16]],
          },
        ]}
      />

      <Prose>
        At <Code>T_mem=5</Code> the LSTM is near-perfect (<Code>0.98</Code>), the GRU solid (<Code>0.80</Code>), and the simple RNN still at chance (<Code>0.19</Code>, which for an 8-token vocabulary with mostly zeros is close to the majority-class baseline). At <Code>T_mem=15</Code> all three fall apart at this training budget — the task has a 30-step horizon (15 memorize + 1 delimiter + 15 reproduce) and 1,500 Adam steps is not enough. Longer training and careful initialization close the gap, but the qualitative ordering — LSTM/GRU drastically better than simple RNN on short-range memorization — is the robust finding.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        The architecture choice is not binary. By 2026 the practical question is rarely "RNN, LSTM, or GRU" in isolation — it is "RNN family versus transformer versus state-space model," given the workload. This matrix is the honest version.
      </Prose>

      <Prose>
        <strong>Short-to-medium sequences (<Code>{"T < 200"}</Code>), clean English, plenty of training data:</strong> Transformer wins on final quality per FLOP. LSTM and GRU are fine if you are building a small model for a constrained deployment or if the problem has inductive biases that favor recurrent structure (strict left-to-right causality, variable-length sequences with fast streaming inference). Below <Code>200</Code> time steps the vanishing-gradient differences between RNN variants matter less; pick the GRU for its smaller parameter count.
      </Prose>

      <Prose>
        <strong>Long-range dependencies (<Code>{"T > 1000"}</Code>):</strong> Transformer beats LSTM beats simple RNN, but the transformer's <Code>{"O(T^2)"}</Code> attention cost gets punishing. For the longest sequences (genomics, audio at 16kHz sample rate, multi-page documents), modern state-space models like Mamba and S4 are the strongest current option — they match transformer quality at linear cost in <Code>T</Code>. LSTMs remain competitive at 500 to 2000 time steps and were, until roughly 2019, the default choice for this regime.
      </Prose>

      <Prose>
        <strong>Low-latency streaming inference (one token at a time):</strong> Recurrent architectures win structurally. A transformer must maintain a KV cache that grows linearly with the number of tokens generated, and the attention at each step is <Code>O(T)</Code>. An LSTM or GRU or Mamba has a constant-size state; its per-token inference cost is constant in the sequence length. For real-time speech, live transcription, or autoregressive generation with a long context, the recurrent formulation has a fundamental memory and latency advantage.
      </Prose>

      <Prose>
        <strong>Small model plus small data:</strong> The GRU often wins. Its three-gate structure is enough to capture most of what LSTM offers, with fewer parameters and faster per-step training. On text classification benchmarks in the 10k-100k example regime, a two-layer bidirectional GRU is a strong, hard-to-beat baseline.
      </Prose>

      <Prose>
        <strong>Character-level modeling, specialized domains:</strong> LSTMs remain competitive for tasks where the training data is structured enough that the transformer's "let me attend to everything" inductive bias is overkill. Character-level language models, time-series forecasting with strong temporal locality, audio phoneme recognition, DNA motif detection — all historically strong LSTM domains, and the advantage persists when compute and data are limited.
      </Prose>

      <Prose>
        <strong>Default recommendation, 2026:</strong> Use a transformer unless you have a specific reason not to. When you <em>do</em> have a reason — streaming inference, very long sequences, tight compute budget, parameter-efficient embedding in a larger pipeline — reach for a GRU first, an LSTM if the task is memory-heavy and known to benefit from the extra gate, and Mamba if you need sub-quadratic scaling at truly long contexts. Simple RNNs are a pedagogical object; they have no remaining production niche.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <Prose>
        The scaling story for recurrent networks is the story of a bottleneck that never fully went away. Every other deep-learning component has a natural parallelism that modern hardware is built around — convolutions parallelize over spatial locations, transformer layers parallelize over sequence positions, feedforward networks parallelize over everything. The RNN's forward pass is <em>inherently sequential</em>: <Code>{"h_t"}</Code> depends on <Code>{"h_{t-1}"}</Code>, which depends on <Code>{"h_{t-2}"}</Code>, and nothing short of algorithmic cleverness can let you compute them in parallel.
      </Prose>

      <Prose>
        <strong>Training throughput.</strong> On a GPU a transformer's forward pass over a length-<Code>T</Code> sequence runs in <Code>O(\\log T)</Code> wall-clock time given enough parallelism; an RNN runs in <Code>O(T)</Code>. For <Code>T=1024</Code> on an A100, a 100M-parameter transformer processes a batch roughly 20-50x faster than the equivalent LSTM even before you count the transformer's other advantages. Cudnn's fused LSTM kernel closes some of the gap by running all four gate matrix multiplies in parallel within a time step, but the across-time dependency chain remains; you cannot fuse <Code>t=5</Code> with <Code>t=6</Code>.
      </Prose>

      <Prose>
        <strong>Stacked depth.</strong> RNN depth does not scale like transformer depth. Graves's deep-LSTM work topped out around three to five layers; above that the gradient through both time <em>and</em> layers runs into compound vanishing/exploding problems. Residual connections between RNN layers help, but the fundamental limit is real: a 24-layer LSTM is much harder to train than a 24-layer transformer, and the deepest production RNNs historically hovered around 8 layers maximum (e.g., Google's neural machine translation system circa 2016).
      </Prose>

      <Prose>
        <strong>Bidirectional doubling.</strong> A bidirectional RNN doubles the parameter count and the compute per time step — forward and backward passes are independent and must both be computed. For classification and tagging this is fine; for autoregressive generation bidirectionality is disallowed (you cannot look ahead).
      </Prose>

      <Prose>
        <strong>Truncated BPTT.</strong> Full BPTT over a sequence of length <Code>T</Code> requires storing all <Code>T</Code> hidden states for the backward pass — memory scales linearly with <Code>T</Code> per example. For long sequences this is prohibitive. The standard workaround is truncated BPTT: unroll the RNN over a window of <Code>T_bptt</Code> steps (typically 35 to 200), compute loss and gradients over that window, detach the hidden state, and move on. Gradients can only inform behavior within the window, so any dependency longer than <Code>T_bptt</Code> is invisible to learning.
      </Prose>

      <Prose>
        <strong>Inference throughput.</strong> Here the RNN wins. For autoregressive generation a transformer must attend to all previous tokens at every step — attention cost grows linearly with generated length and the KV cache memory grows with it. An RNN's per-token inference cost is <em>constant</em>: one matrix multiply with the gate weights, one with the recurrent weights, one elementwise update. This is exactly the property the 2023 state-space model revival is exploiting. Mamba's main claim is not "better than transformers on quality" but "match transformer quality at <em>constant-time</em> autoregressive inference instead of <em>linear-time</em>."
      </Prose>

      <Prose>
        <strong>The Mamba/S4 revival.</strong> Linear state-space models — Gu et al.'s S4, then Mamba — re-parameterize the recurrent dynamics so that the forward pass can be computed as a <em>convolution</em> during training (parallel over time) but as a <em>recurrence</em> during inference (constant-state streaming). This gives you the best of both worlds: transformer-style training speed and RNN-style inference efficiency. The selective-scan variant in Mamba adds input-dependent gating, which is structurally similar to the LSTM's forget gate but formulated to stay parallelizable. As of 2026 this is the only recurrent-family architecture consistently competitive with transformers at scale; the LSTM and GRU are its grandparents.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <Prose>
        Ten years of production RNN deployments have accumulated a specific set of sharp edges. Each of these is a bug you will eventually write.
      </Prose>

      <Prose>
        <strong>1. Exploding gradients without clipping.</strong> Even with LSTM or GRU, exploding gradients remain a real risk — especially early in training when the recurrent weights are not yet regularized by the loss. A single step can produce a gradient with norm in the hundreds or thousands, and applying it at the learning rate you tuned on typical gradients will send the weights to infinity. Always call <Code>torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)</Code> before <Code>opt.step()</Code>. If training diverges even with clipping, lower <Code>max_norm</Code> to <Code>0.25</Code> or check your weight initialization.
      </Prose>

      <Prose>
        <strong>2. Vanishing gradients without a gated architecture.</strong> If you are using a vanilla RNN and training does not converge on a task that requires dependencies beyond <Code>~10</Code> time steps, the problem is almost certainly vanishing gradients. Switch to an LSTM or GRU, or add residual connections between time steps if you must keep the simple-RNN form. "Train longer" does not fix vanishing gradients — the gradient at long range is simply not carrying learning signal.
      </Prose>

      <Prose>
        <strong>3. <Code>batch_first</Code> versus <Code>seq_first</Code> confusion.</strong> PyTorch's RNN modules default to <Code>seq_first</Code> — input shape <Code>(T, B, X)</Code> — because it is slightly faster for cuDNN. Most application code prefers <Code>batch_first=True</Code>, shape <Code>(B, T, X)</Code>. Mixing the two in the same pipeline, or passing <Code>batch_first</Code> tensors to a module initialized without the flag, produces a silent shape swap — the model trains on the transpose of what you think, learning is garbage, and the bug is maddening to debug. Pick one convention and enforce it everywhere; prefer <Code>batch_first=True</Code> unless you are matching an existing codebase.
      </Prose>

      <Prose>
        <strong>4. Incorrect hidden-state passing between batches.</strong> For stateful RNNs — models that carry hidden state across batch boundaries — you must pass the <em>previous</em> batch's final hidden state as the initial state of the next batch, and you must detach it to avoid growing the computation graph. Common mistakes: passing hidden state of shape <Code>(num_layers, B_old, H)</Code> to a batch of different size <Code>B_new</Code>; forgetting to detach, so BPTT runs over the entire training history; passing a stale hidden state after shuffling the batch order. If in doubt, initialize hidden state to zero at every batch boundary — you pay a small quality penalty but avoid the class of bugs.
      </Prose>

      <Prose>
        <strong>5. Teacher-forcing exposure bias.</strong> Training a seq2seq decoder with teacher forcing is fast and stable, but at inference time the decoder feeds its own predictions back in. If early training errors are rare in the teacher-forced distribution, the decoder never learns to recover from them, and inference quality degrades as the generated sequence gets longer — the "exposure bias" problem. Mitigations: scheduled sampling (randomly swap ground-truth tokens for the decoder's own predictions during training, with probability that increases over epochs), sequence-level training objectives (MRT, REINFORCE, beam-search-aware losses), or train with occasional corruption of the teacher-forcing signal.
      </Prose>

      <Prose>
        <strong>6. Tanh saturation.</strong> Simple RNNs use <Code>tanh</Code> activations, and when the pre-activation is large in magnitude the tanh saturates, its derivative collapses toward zero, and training stalls. Symptoms: hidden-state activations clustered at <Code>{"\\pm 1"}</Code>, gradients that are small even early in training. Fixes: scale down the initial weights (the usual <Code>W ~ N(0, 1/H)</Code> scheme), add layer normalization to the hidden state update, or switch to a gated architecture where sigmoid gates control the magnitude of the tanh argument.
      </Prose>

      <Prose>
        <strong>7. Forgetting to mask padded positions in the loss.</strong> Packing the input solves the forward-pass efficiency, but the loss is still computed over the full padded output by default. If you compute <Code>F.cross_entropy(logits, targets)</Code> without an <Code>ignore_index</Code> or without a mask, you train the model to predict the padding token on padded positions — a task it will learn instantly and incorrectly. Always pass <Code>ignore_index=PAD_ID</Code> to <Code>F.cross_entropy</Code>, or multiply the per-position loss by a validity mask before summing.
      </Prose>

      <Prose>
        <strong>8. Stateful RNN hidden state without <Code>detach()</Code>.</strong> A variant of bug #4. If you maintain hidden state across training batches without calling <Code>.detach()</Code> on it, PyTorch keeps building the computation graph across batches, and the memory usage grows unboundedly until the process runs out of GPU memory. Symptom: training starts fine, then OOMs after a few hundred steps. Fix: <Code>h = h.detach()</Code> at the start (or end) of each batch.
      </Prose>

      <Prose>
        <strong>9. Forget-gate bias initialization.</strong> The <Code>{"b_f = 1"}</Code> initialization trick from Gers et al. (2000) is not on by default in PyTorch's <Code>nn.LSTM</Code>. If your long-range task trains slowly or plateaus early, manually initialize the forget-gate bias:
      </Prose>

      <CodeBlock language="python">
{`for name, param in lstm.named_parameters():
    if "bias_ih" in name or "bias_hh" in name:
        # PyTorch concatenates gates in order: i, f, g, o
        # Set forget-gate bias (second quarter) to 1.
        H = param.size(0) // 4
        param.data[H:2*H].fill_(1.0)`}
      </CodeBlock>

      <Prose>
        <strong>10. Wrong ordering of gate weights when loading pretrained or inter-framework.</strong> PyTorch uses order <em>input, forget, candidate, output</em> for LSTM and <em>reset, update, new</em> for GRU. TensorFlow, Keras, JAX, and third-party C++ implementations often use different orderings (TensorFlow's LSTM uses <em>input, candidate, forget, output</em>). If you port weights between frameworks without re-permuting the gate blocks, the model's forward pass produces garbage that looks plausible (shapes match, outputs are in-range) but performance is catastrophic. Always verify a round-trip forward pass produces matching outputs after porting.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Every specific claim about history, architecture, or measurement in this topic traces to one of the following. Venues and arXiv IDs are current as of April 2026.
      </Prose>

      <Prose>
        <strong>Hopfield, J. J. (1982).</strong> "Neural networks and physical systems with emergent collective computational abilities." <em>Proceedings of the National Academy of Sciences</em> 79(8):2554-2558. The Hopfield network; the first neural architecture whose state evolves over time as a function of its own previous state.
      </Prose>

      <Prose>
        <strong>Elman, J. L. (1990).</strong> "Finding structure in time." <em>Cognitive Science</em> 14(2):179-211. The simple recurrent network. Introduced the architecture and showed that the hidden state learns syntactic structure from next-token prediction alone.
      </Prose>

      <Prose>
        <strong>Werbos, P. J. (1990).</strong> "Backpropagation through time: what it does and how to do it." <em>Proceedings of the IEEE</em> 78(10):1550-1560. The BPTT algorithm as applied to RNNs; the formal framework for training recurrent networks with gradient descent.
      </Prose>

      <Prose>
        <strong>Hochreiter, S. (1991).</strong> "Untersuchungen zu dynamischen neuronalen Netzen." Diploma thesis, Institut für Informatik, Technische Universität München. The vanishing-gradient analysis; proves that gradients in a recurrent network with bounded-derivative activations decay exponentially in time depth. Written in German; the English-language version of the core result appeared in Hochreiter et al. (2001), "Gradient Flow in Recurrent Nets."
      </Prose>

      <Prose>
        <strong>Hochreiter, S., and Schmidhuber, J. (1997).</strong> "Long Short-Term Memory." <em>Neural Computation</em> 9(8):1735-1780. The original LSTM. Introduced the cell state, input gate, and output gate; established the gated-recurrent paradigm.
      </Prose>

      <Prose>
        <strong>Gers, F. A., Schmidhuber, J., and Cummins, F. (2000).</strong> "Learning to Forget: Continual Prediction with LSTM." <em>Neural Computation</em> 12(10):2451-2471. Added the forget gate to the original LSTM, producing the form in universal use today. Also recommended the <Code>{"b_f = 1"}</Code> initialization.
      </Prose>

      <Prose>
        <strong>Cho, K., van Merriënboer, B., Gülçehre, Ç., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014).</strong> "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." arXiv:1406.1078, EMNLP 2014. Introduced the GRU; also the encoder-decoder framework that seq2seq generalized.
      </Prose>

      <Prose>
        <strong>Sutskever, I., Vinyals, O., and Le, Q. V. (2014).</strong> "Sequence to Sequence Learning with Neural Networks." arXiv:1409.3215, NeurIPS 2014. Two stacked LSTMs — encoder and decoder — for neural machine translation. Established the seq2seq paradigm that dominated the field for three years.
      </Prose>

      <Prose>
        <strong>Graves, A. (2013).</strong> "Generating Sequences With Recurrent Neural Networks." arXiv:1308.0850. Deep stacked LSTMs for character-level language modeling and handwriting generation; the first paper to make large LSTMs a widely-replicated success.
      </Prose>

      <Prose>
        <strong>Karpathy, A. (2015).</strong> "The Unreasonable Effectiveness of Recurrent Neural Networks." Blog post, May 2015. Available at <Code>karpathy.github.io/2015/05/21/rnn-effectiveness/</Code>. The most-read single introduction to RNNs and char-level language modeling; <Code>min-char-rnn.py</Code> is the reference implementation the section-4 code is styled after.
      </Prose>

      <Prose>
        <strong>Gu, A., and Dao, T. (2023).</strong> "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752. The modern state-space model that restored recurrent architectures to competitiveness; combines transformer-style parallel training with RNN-style constant-time inference.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <Prose>
        <strong>1. The <Code>{"b_f = 1"}</Code> trick.</strong> The forget-gate bias in an LSTM is typically initialized to <Code>1</Code>. Given that the forget gate is <Code>{"f_t = \\sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)"}</Code>, what is the approximate initial value of <Code>{"f_t"}</Code> right after initialization, assuming the weights are small? Why does this value specifically help with the vanishing-gradient problem at the start of training, and what would happen if you initialized <Code>{"b_f = -1"}</Code> instead?
      </Prose>

      <Prose>
        <strong>2. GRU as restricted LSTM.</strong> Show that a GRU can be thought of as an LSTM with a <em>tied</em> input and forget gate — the GRU's update gate <Code>{"z_t"}</Code> plays the role of both. Write the correspondence explicitly: which LSTM gates are tied to which GRU gates, and which LSTM component is merged with which GRU component? (Hint: the GRU's hidden state corresponds to the LSTM's cell state, not its hidden state.)
      </Prose>

      <Prose>
        <strong>3. BPTT memory cost.</strong> A two-layer bidirectional LSTM with hidden size 512 is trained on sequences of length <Code>T=1000</Code> with batch size 64. Estimate the memory cost of storing all hidden states needed for the backward pass, in floating-point values. Now compute the corresponding memory cost for truncated BPTT with <Code>T_bptt=100</Code>. What is the ratio of the two, and why does truncated BPTT lose information that full BPTT captures?
      </Prose>

      <Prose>
        <strong>4. Streaming inference cost.</strong> You are generating text autoregressively with a decoder-only model. Compare the per-token inference FLOPs and memory costs of: (a) a 100M-parameter transformer with context length 4096, (b) a 100M-parameter LSTM with hidden size 1024. Which scales better as you extend generation to 10,000 tokens? How would you answer change if the LSTM were replaced with a 100M-parameter Mamba?
      </Prose>

      <Prose>
        <strong>5. Picking the architecture for a real problem.</strong> You are building three production systems: (a) a live speech-to-text model that must emit partial transcriptions every 100 ms with bounded latency; (b) a sentence classifier for customer-support ticket routing with 50-word inputs and 10M training examples; (c) a sequence generator for ECG anomaly detection on 30-second waveforms sampled at 500 Hz (so <Code>T=15{"{,}"}000</Code> per input). For each, choose among simple RNN, GRU, LSTM, Transformer, and Mamba; justify your choice in terms of the architecture's strengths and the workload's constraints.
      </Prose>

      {/* ======================================================================
          END
          ====================================================================== */}
      <Prose>
        The RNN is the sequence-modeling architecture the field keeps coming back to. Elman's 1990 network, Hochreiter and Schmidhuber's 1997 LSTM, Cho's 2014 GRU, Gu and Dao's 2023 Mamba — four decades of the same central idea, that the state of a sequence model should be a fixed-size vector that evolves as the sequence is read. Transformers won the 2017-2022 scaling race because they could be parallelized; state-space models are winning the 2023-2026 streaming race because they cannot. The gated recurrent architectures of the LSTM-GRU era are the conceptual bridge between them, and every engineer who understands them fluently will find the Mamba generation easier to reason about than the practitioner who learned deep learning on transformers alone.
      </Prose>
    </div>
  ),
};

export default rnnsLstmsGrusContent;
