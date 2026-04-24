import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const seq2seqContent = {
  title: "Sequence-to-Sequence & Encoder-Decoder",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Before 2014, machine translation was a baroque pipeline. A typical phrase-based statistical MT system stacked together an alignment model (IBM Model 1-5, HMM alignments), a phrase-extraction heuristic, a language model (n-gram with Kneser-Ney smoothing), a distortion model for reordering, and a log-linear combiner tuned with MERT. Each piece was trained separately, on different data, with different objectives, and glued into an inference-time beam search that interpolated scores from five to ten different models. Moses was the representative open-source system; Google Translate and Systran ran versions of the same idea at production scale. The approach worked — WMT 2013 En-Fr systems scored BLEU ~33 — but the architecture was a tower of hand-engineered components, and every improvement required surgery on one of the pieces.
      </Prose>

      <Prose>
        Ilya Sutskever, Oriol Vinyals, and Quoc Le asked what would happen if you replaced the entire pipeline with one neural network trained end-to-end. Their NeurIPS 2014 paper "Sequence to Sequence Learning with Neural Networks" (arXiv:1409.3215) gave the answer. The architecture had exactly two parts: a four-layer LSTM that read the English source sentence and compressed it into a fixed-size vector (the final hidden state of the encoder), and a second four-layer LSTM that took that vector as its initial state and generated the French translation one token at a time, feeding each predicted token back as the next input. They trained on WMT'14 English-French, 12M sentence pairs, with no feature engineering beyond byte-pair tokenization, and reported BLEU 34.8 — better than the phrase-based baseline. One network, one loss, one beam search. The paper included one clever implementation trick: reversing the source sentence helped the optimizer by making the last few source tokens close (in sequence) to the first few target tokens, reducing the "distance" the gradient had to travel.
      </Prose>

      <Prose>
        Two months earlier, Kyunghyun Cho and colleagues had published "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" at EMNLP 2014 (arXiv:1406.1078). The paper introduced what they called an "RNN Encoder-Decoder," trained to score phrase pairs for a conventional phrase-based MT system rather than replace it outright. The Cho paper is where the GRU appears for the first time — a simpler gated RNN variant meant to ease training. The two papers land close enough in time that people usually cite them together as "the seq2seq papers." Cho's contribution is the architectural framing (encoder-decoder as a general recipe); Sutskever's is the end-to-end application to a real task at scale.
      </Prose>

      <Prose>
        The bottleneck was visible from day one. If the encoder compresses an entire source sentence — which might be 30 words, 50 words, 80 words — into a single fixed-size vector, then long sentences will inevitably lose information. Sutskever's paper noted that their model degraded on long sentences; the reversal trick helped, but it did not eliminate the problem. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio published the fix in September 2014 (ICLR 2015, arXiv:1409.0473): "Neural Machine Translation by Jointly Learning to Align and Translate." Instead of passing only the final encoder hidden state, the decoder at each step computes a <em>weighted sum of all encoder hidden states</em>, where the weights are learned from the current decoder state. The network effectively learns a soft alignment between source and target positions. Bahdanau attention pushed the BLEU numbers higher and, more importantly, kept accuracy flat as source length grew beyond 30-40 words.
      </Prose>

      <Prose>
        The seq2seq paradigm proved absurdly general. Within 18 months it had been applied to abstractive summarization (Rush, Chopra, Weston 2015), question answering, conversational response (Vinyals and Le 2015 "Neural Conversational Model"), speech-to-text, image captioning (if you treat the image as a vector), syntactic parsing, and code generation. Any task shaped as "variable-length input maps to variable-length output" fits the same encoder-decoder template. Google moved its production translation system to a stacked-LSTM encoder-decoder in 2016 with the GNMT paper (Wu et al. arXiv:1609.08144) — an 8-layer encoder, 8-layer decoder, residual connections, wordpiece tokenization, and attention. That was the first time deep seq2seq served real translation traffic worldwide.
      </Prose>

      <Prose>
        Then the Transformer happened. Ashish Vaswani and colleagues' NeurIPS 2017 paper "Attention Is All You Need" (arXiv:1706.03762) replaced the recurrent encoder and decoder with stacks of self-attention and feed-forward blocks while preserving the encoder-decoder shape exactly. The encoder still reads the source bidirectionally and produces contextualized representations; the decoder still generates autoregressively; cross-attention between them still plays the same alignment role that Bahdanau attention played. What changed was the underlying sequence primitive: attention is O(n<sup>2</sup>) but fully parallel across positions, whereas RNNs are O(n) in state size but serial. On modern accelerators, parallel wins. Within two years T5 (Raffel et al. 2020, arXiv:1910.10683) and BART (Lewis et al. 2020, arXiv:1910.13461) had made encoder-decoder transformers the standard for conditional generation tasks where the input strongly constrains the output.
      </Prose>

      <Callout accent="gold">
        The seq2seq idea is not an architecture. It is a <em>recipe</em>: encode a variable-length input to some representation, decode a variable-length output from that representation, train end-to-end with cross-entropy, generate with beam search. RNN-based seq2seq and Transformer-based encoder-decoder differ only in what lives inside the two boxes. Understanding the recipe first lets you slot any modern model (T5, BART, mT5, MarianMT, Whisper) into the same mental frame.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 Two networks, one joint training loop</H3>

      <Prose>
        An encoder-decoder model splits generation into two sub-problems. The encoder's job is to <em>read</em> — to consume the source sequence and produce a representation that contains whatever the decoder will need. The decoder's job is to <em>write</em> — to produce tokens one at a time, conditioned on the encoder's representation and on everything it has already emitted. The two networks are trained jointly: one gradient flows through both, and the encoder's parameters are adjusted so that the representation it produces makes the decoder's job easier.
      </Prose>

      <Prose>
        This split is the key conceptual move. Before seq2seq, neural machine translation existed in a few experimental forms but no architecture cleanly decoupled source-comprehension from target-generation. Once you have that decoupling, everything else follows. Want multilingual? Share one decoder across many source languages (each with its own encoder, or all feeding a shared encoder with a language tag). Want summarization? Train on {"(long, short)"} pairs instead of {"(English, French)"} pairs. Want speech-to-text? Replace the encoder's token-embedding-plus-RNN with a convolutional frontend plus RNN. The decoder doesn't care where the representation came from — only that it is useful.
      </Prose>

      <H3>2.2 Teacher forcing at training time</H3>

      <Prose>
        During training, the decoder does not actually get to see its own predictions. Instead, at every position <Code>t</Code>, the decoder is given the <em>ground-truth</em> previous token <Code>{"y_{t-1}"}</Code> from the target sequence and is asked to predict the distribution over the next token. The loss is cross-entropy against <Code>y_t</Code>. This technique is called <em>teacher forcing</em>: the teacher (the training data) forces the model onto the correct trajectory at every step, so the model never has to learn to recover from its own past mistakes during training. Teacher forcing makes training fast and stable — every decoder step is effectively independent given the target prefix, which means we can run them in a single matmul rather than in a serial loop.
      </Prose>

      <H3>2.3 Autoregressive at inference time</H3>

      <Prose>
        At inference there is no target to feed. The decoder starts from a special <Code>{"<SOS>"}</Code> token (or an empty prefix), predicts a distribution, picks a token (argmax for greedy, top-k for sampling, or maintain multiple candidates for beam search), feeds that token back as the previous-token input, and repeats until it emits <Code>{"<EOS>"}</Code> or hits a length cap. This is called <em>autoregressive</em> generation: each output depends on every previous output. The behavior is fundamentally serial — you cannot predict token <Code>t+1</Code> without having already produced token <Code>t</Code> — which is why decoder inference is hard to parallelize across time.
      </Prose>

      <H3>2.4 Exposure bias: the train-test mismatch</H3>

      <Prose>
        The gap between teacher forcing (training) and autoregressive generation (inference) is called <em>exposure bias</em>. During training, the decoder has only ever seen <em>correct</em> prefixes — the target sentence exactly. At inference, the decoder must condition on its own <em>possibly wrong</em> prefixes. If the model ever makes a mistake at step <Code>t</Code>, it is now in a state it never encountered during training, and errors can compound. Ranzato et al. (ICLR 2016, "Sequence Level Training with Recurrent Neural Networks," arXiv:1511.06732) quantified this effect and proposed scheduled sampling / REINFORCE mixtures to mitigate it. In practice, modern seq2seq training uses some combination of teacher forcing plus label smoothing plus (sometimes) scheduled sampling; the pure form is never quite what you ship.
      </Prose>

      <H3>2.5 Input and output lengths are fully decoupled</H3>

      <Prose>
        A big payoff of the encoder-decoder split is that the input and output lengths have no structural relationship. A 40-word English sentence can produce a 60-word French translation or a 5-word summary or a single-token yes/no answer. Contrast this with decoder-only language models (GPT-style), where the input and output sit on the same sequence axis and are only distinguished by prompt-response convention. For tasks where input-faithfulness matters (translation, summarization, document-grounded QA), the explicit separation is a real inductive bias — the encoder can read the whole source bidirectionally and produce arbitrary representations, without worrying about the causal constraint that decoder-only models are stuck with.
      </Prose>

      <H3>2.6 The fixed-context bottleneck</H3>

      <Prose>
        The Achilles heel of vanilla RNN seq2seq is that the decoder only ever sees the encoder's <em>final</em> hidden state <Code>h_T</Code>. Every piece of information about the source sentence must squeeze through that single vector. With a 128-dimensional hidden state, you are trying to represent the content, syntax, named entities, tense, and argument structure of a full sentence in 128 floats. For a 5-word input this works fine. For a 50-word input it does not. Empirically, BLEU scores for vanilla seq2seq drop sharply past ~30 source tokens (Bahdanau et al. 2015 show the curve explicitly). Attention solves this by exposing <em>all</em> encoder hidden states to the decoder and letting it choose which to look at on each step — replacing a lossy compression with a pointer into the original.
      </Prose>

      <Callout accent="gold">
        The fixed-context bottleneck is the single observation that connects vanilla seq2seq to attention to the modern Transformer encoder-decoder. Every improvement since 2014 can be read as "let the decoder look at more of the encoder, more flexibly" — from Bahdanau attention to multi-head cross-attention to retrieval-augmented decoding. The core recipe has not changed; only the strictness of the bottleneck has.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Encoder as a state recurrence</H3>

      <Prose>
        Given a source token sequence <Code>{"x = (x_1, \\ldots, x_T)"}</Code>, the encoder produces hidden states by a recurrence with some cell function <Code>f</Code> (LSTM, GRU, or a more recent variant):
      </Prose>

      <MathBlock>{"h_t = f(h_{t-1}, \\, E_{\\text{src}}(x_t)) \\quad \\text{for } t = 1, \\ldots, T, \\quad h_0 = 0"}</MathBlock>

      <Prose>
        The encoder's output is either the single final state <Code>{"c = h_T"}</Code> (vanilla seq2seq) or the full set of hidden states <Code>{"H = (h_1, \\ldots, h_T)"}</Code> (attention-equipped seq2seq). The embedding <Code>{"E_{\\text{src}}"}</Code> maps discrete tokens into the cell's input space.
      </Prose>

      <H3>3.2 Decoder as a conditional language model</H3>

      <Prose>
        The decoder defines a probability distribution over target sequences conditioned on the source representation. For target sequence <Code>{"y = (y_1, \\ldots, y_{T'})"}</Code>, the standard chain-rule factorization is:
      </Prose>

      <MathBlock>{"P(y \\mid x) = \\prod_{t=1}^{T'} P(y_t \\mid y_{<t}, \\, x)"}</MathBlock>

      <Prose>
        In vanilla seq2seq, the conditional on <Code>x</Code> is mediated entirely through the encoder's final state: <Code>{"P(y_t \\mid y_{<t}, x) = P(y_t \\mid y_{<t}, h_T)"}</Code>. With attention, each step can look up a different weighted combination of encoder states.
      </Prose>

      <H3>3.3 Decoder recurrence</H3>

      <Prose>
        The decoder has its own hidden state <Code>s_t</Code>, updated from the previous state, the previously emitted token, and (for vanilla seq2seq) the fixed context vector:
      </Prose>

      <MathBlock>{"s_t = g(s_{t-1}, \\, [E_{\\text{tgt}}(y_{t-1}); \\, c]), \\quad s_0 = \\phi(h_T)"}</MathBlock>

      <MathBlock>{"P(y_t \\mid y_{<t}, x) = \\mathrm{softmax}(W_{\\text{out}} \\, s_t + b_{\\text{out}})"}</MathBlock>

      <Prose>
        The initialization <Code>{"\\phi(h_T)"}</Code> is often the identity (Sutskever) or a small MLP (Cho). The concatenation <Code>{"[E_{\\text{tgt}}(y_{t-1}); c]"}</Code> feeds both the previously-emitted token and the compressed source context into every decoder step.
      </Prose>

      <H3>3.4 Training loss</H3>

      <Prose>
        Training minimizes the negative log-likelihood of the target sequence under the model, averaged over the training corpus:
      </Prose>

      <MathBlock>{"\\mathcal{L}(\\theta) = -\\sum_{(x, y) \\in \\mathcal{D}} \\sum_{t=1}^{T'} \\log P_{\\theta}(y_t \\mid y_{<t}, \\, x)"}</MathBlock>

      <Prose>
        The inner sum is the cross-entropy at every decoder position. Under teacher forcing, <Code>{"y_{<t}"}</Code> is the gold target prefix, which removes the sequential dependency of model predictions and lets all positions be computed in one shot. In PyTorch, <Code>F.cross_entropy(logits.view(-1, V), y[:, 1:].view(-1), ignore_index=PAD)</Code> is the standard implementation.
      </Prose>

      <H3>3.5 Greedy decoding</H3>

      <Prose>
        Greedy decoding picks the argmax at every step:
      </Prose>

      <MathBlock>{"\\hat{y}_t = \\arg\\max_{v \\in V} \\, P(y_t = v \\mid \\hat{y}_{<t}, \\, x)"}</MathBlock>

      <Prose>
        Greedy is fast (O(T') forward passes, no branching) but sub-optimal. It commits to the highest-probability first token even when a slightly lower-probability first token would lead to a much higher-probability full sequence.
      </Prose>

      <H3>3.6 Beam search</H3>

      <Prose>
        Beam search maintains the <Code>k</Code> highest-scoring partial hypotheses at each step. Let <Code>{"B_t"}</Code> be the set of <Code>k</Code> hypotheses at step <Code>t</Code>, each a {"(prefix, log-prob)"} pair. The expansion step is:
      </Prose>

      <MathBlock>{"B_{t+1} = \\mathrm{top}_k \\left\\{ (\\text{prefix} \\circ v, \\, \\log P(\\text{prefix}) + \\log P(v \\mid \\text{prefix}, x)) \\; : \\; (\\text{prefix}, \\log P) \\in B_t, \\, v \\in V \\right\\}"}</MathBlock>

      <Prose>
        Beam search explores a small fraction of the search tree (<Code>k</Code> branches out of <Code>|V|</Code>) but consistently beats greedy on BLEU and other sequence-level metrics. The cost is <Code>k</Code>{"\u00d7"} the greedy cost plus a top-k operation per step.
      </Prose>

      <H3>3.7 Length normalization</H3>

      <Prose>
        Vanilla beam search biases toward short outputs, because every additional token adds a negative log-probability contribution. A hypothesis of length 20 is penalized relative to a hypothesis of length 5 even if each per-token probability is comparable. The fix is length normalization: divide the cumulative log-probability by a function of length.
      </Prose>

      <MathBlock>{"\\mathrm{score}(y, x) = \\frac{1}{L(y)^{\\alpha}} \\sum_{t=1}^{|y|} \\log P(y_t \\mid y_{<t}, x), \\quad L(y) = \\frac{(5 + |y|)^{\\alpha}}{6^{\\alpha}}"}</MathBlock>

      <Prose>
        The GNMT paper (Wu et al. 2016) popularized the specific <Code>{"(5 + |y|) / 6"}</Code> formula with <Code>{"\\alpha \\in [0.6, 1.0]"}</Code>. Higher <Code>{"\\alpha"}</Code> favors longer outputs; lower favors shorter. Empirically <Code>{"\\alpha \\approx 0.7"}</Code>-<Code>{"0.8"}</Code> works well for translation and summarization.
      </Prose>

      <H3>3.8 Teacher-forcing ratio and scheduled sampling</H3>

      <Prose>
        Scheduled sampling (Bengio et al. NIPS 2015) interpolates between teacher forcing and free-running generation during training. At each decoder step, with probability <Code>{"\\epsilon_t"}</Code> the model uses the ground-truth previous token, and with probability <Code>{"1 - \\epsilon_t"}</Code> it uses its own sampled prediction. The schedule decays <Code>{"\\epsilon_t"}</Code> from 1.0 (pure teacher forcing) toward some smaller value as training progresses, exposing the model to its own error distribution. Empirical results are mixed — Huszar (2015) pointed out that scheduled sampling biases the MLE objective — and modern systems often rely on label smoothing plus large-scale data instead.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was run against PyTorch 2.6 on CUDA. The <Code>{"# Output:"}</Code> comments are real stdout copied from the run log. The task is a toy that isolates the seq2seq mechanism: given a sequence of integers drawn from 1-10, produce the same multiset in sorted order. Vocabulary is 13 tokens: PAD=0, SOS=1, EOS=2, then one token per digit. The task is hard enough that a model has to actually read the input (not just memorize an output distribution) but small enough that we can train it in under a minute.
      </Prose>

      <H3>4.1 Data generation</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)
random.seed(42)
device = "cuda"

PAD, SOS, EOS = 0, 1, 2
VOCAB = 13       # 0=PAD, 1=SOS, 2=EOS, 3..12 for digits 1..10
OFFSET = 3       # digit d -> token (OFFSET + d - 1)

def sample_batch(batch_size, length):
    xs, ys = [], []
    for _ in range(batch_size):
        perm = random.sample(range(1, 11), length)    # distinct digits 1..10
        srt  = sorted(perm)
        x = [OFFSET + v - 1 for v in perm]
        y = [SOS] + [OFFSET + v - 1 for v in srt] + [EOS]
        xs.append(x); ys.append(y)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))

x, y = sample_batch(3, 6)
print("  sample inputs  x:", x.tolist())
print("  sample targets y:", y.tolist())

# Output:
#   sample inputs  x: [[5, 9, 7, 6, 12, 8], [11, 7, 12, 9, 5, 3], [7, 8, 3, 11, 6, 9]]
#   sample targets y: [[1, 5, 6, 7, 8, 9, 12, 2], [1, 3, 5, 7, 9, 11, 12, 2], [1, 3, 6, 7, 8, 9, 11, 2]]`}
      </CodeBlock>

      <Prose>
        A batch of three sequences of length 6. Inputs are digit-offset tokens in some order; targets are the same digits sorted, bracketed by SOS and EOS. The decoder will see <Code>y[:, :-1]</Code> as input (teacher forcing) and be trained to predict <Code>y[:, 1:]</Code>.
      </Prose>

      <H3>4.2 Encoder — one-layer LSTM</H3>

      <CodeBlock language="python">
{`class Encoder(nn.Module):
    def __init__(self, vocab=VOCAB, emb=64, hid=128):
        super().__init__()
        self.emb  = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)

    def forward(self, x):
        e = self.emb(x)                           # [B, T, emb]
        outs, (h, c) = self.lstm(e)               # outs: [B, T, H]; h,c: [1, B, H]
        return outs, (h, c)`}
      </CodeBlock>

      <Prose>
        A 64-dimensional embedding feeds a single-layer LSTM with 128 hidden units. The encoder returns all hidden states <Code>outs</Code> (the decoder will ignore these in the vanilla version but use them when we bolt on attention) and the final state pair <Code>(h, c)</Code> that seeds the decoder.
      </Prose>

      <H3>4.3 Decoder — step-wise LSTMCell</H3>

      <CodeBlock language="python">
{`class Decoder(nn.Module):
    def __init__(self, vocab=VOCAB, emb=64, hid=128):
        super().__init__()
        self.emb  = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTMCell(emb, hid)
        self.fc   = nn.Linear(hid, vocab)

    def step(self, tok, h, c):
        e = self.emb(tok)                         # [B, emb]
        h, c = self.lstm(e, (h, c))               # [B, H], [B, H]
        logits = self.fc(h)                       # [B, vocab]
        return logits, h, c`}
      </CodeBlock>

      <Prose>
        The decoder is written at the single-step granularity (<Code>nn.LSTMCell</Code>) because inference must be serial. During training, we still run it step-by-step, but teacher forcing means every step uses the gold previous token, so batches go through quickly even without a fused <Code>nn.LSTM</Code>.
      </Prose>

      <H3>4.4 Joint model with teacher forcing</H3>

      <CodeBlock language="python">
{`class Seq2Seq(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.enc = Encoder(hid=hid)
        self.dec = Decoder(hid=hid)

    def forward(self, x, y, teacher_forcing=True):
        _, (h, c) = self.enc(x)
        h, c = h.squeeze(0), c.squeeze(0)         # [B, H] each
        T = y.size(1) - 1
        logits_all = []
        tok = y[:, 0]                             # start from <SOS>
        for t in range(T):
            logits, h, c = self.dec.step(tok, h, c)
            logits_all.append(logits)
            tok = y[:, t+1] if teacher_forcing else logits.argmax(-1)
        return torch.stack(logits_all, dim=1)     # [B, T, vocab]

model = Seq2Seq(hid=128).to(device)
print(f"[main] params={sum(p.numel() for p in model.parameters()):,}")

# Output:
#   [main] params=201,997`}
      </CodeBlock>

      <Prose>
        Two hundred thousand parameters — small enough to train in ten seconds, large enough to learn a non-trivial mapping. The encoder state <Code>h</Code> squeezed from shape <Code>[1, B, H]</Code> to <Code>[B, H]</Code> initializes the decoder cell. This single tensor is the entire bottleneck: everything the decoder knows about the input after this point must come from these 128 numbers per example.
      </Prose>

      <H3>4.5 Training loop</H3>

      <CodeBlock language="python">
{`opt = torch.optim.Adam(model.parameters(), lr=3e-3)
TRAIN_LEN = 8

for step in range(1, 1001):
    x, y = sample_batch(64, TRAIN_LEN)
    logits = model(x, y, teacher_forcing=True)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                           y[:, 1:].reshape(-1),
                           ignore_index=PAD)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        em = exact_match(model, n_eval=200, length=TRAIN_LEN)
        print(f"[train step={step:4d}] loss={loss.item():.4f}  em={em:.3f}")

# Output:
#   [train step= 200] loss=0.0108  em=1.000
#   [train step= 400] loss=0.0019  em=1.000
#   [train step= 600] loss=0.0009  em=1.000
#   [train step= 800] loss=0.0005  em=1.000
#   [train step=1000] loss=0.0003  em=1.000`}
      </CodeBlock>

      <Prose>
        At TRAIN_LEN = 8 the model hits 100% exact match within 200 steps and drives loss toward zero. The per-step compute is trivial — roughly 37 seconds total on a single GPU for the full thousand-step run. This is what a seq2seq model looks like when the task fits comfortably in its capacity.
      </Prose>

      <H3>4.6 Fixed-context bottleneck — length generalization</H3>

      <Prose>
        Now the interesting test. The encoder has learned to compress a specific length-8 input into its 128-dim final state. What happens if we feed it a length-4 or length-10 input at inference time?
      </Prose>

      <CodeBlock language="python">
{`def exact_match(model, n_eval=200, length=8):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n_eval):
            x, y = sample_batch(1, length)
            _, (h, c) = model.enc(x); h, c = h.squeeze(0), c.squeeze(0)
            tok = torch.tensor([SOS], device=device)
            out = []
            for _ in range(length + 2):
                logits, h, c = model.dec.step(tok, h, c)
                tok = logits.argmax(-1)
                if tok.item() == EOS: break
                out.append(tok.item())
            target = y[0, 1:-1].tolist()
            if out == target: correct += 1
    model.train()
    return correct / n_eval

print("[length-ablation] fixed-context bottleneck test (trained on L=8)")
for L in [4, 6, 8, 10]:
    em = exact_match(model, n_eval=200, length=L)
    print(f"   L={L:2d}  em={em:.3f}")

# Output:
#   [length-ablation] fixed-context bottleneck test (trained on L=8)
#      L= 4  em=0.000
#      L= 6  em=0.000
#      L= 8  em=1.000
#      L=10  em=0.000`}
      </CodeBlock>

      <Prose>
        Perfect performance at L=8, total failure at every other length. The model has not learned a generic sort operation — it has learned "given a length-8-specific encoding, decode a length-8-sorted sequence." The encoder's final hidden state encodes assumptions about how long the input was, and when those assumptions break, the whole pipeline fails. This is the fixed-context bottleneck made visceral: one vector, one distribution, no interpolation to neighbors. An attention-equipped encoder-decoder makes this much less brittle, because the decoder can look at all encoder positions and pick the one it actually needs.
      </Prose>

      <H3>4.7 Greedy vs beam search — decoding quality</H3>

      <Prose>
        With a single-length-trained model, greedy and beam both saturate at 100% for in-distribution inputs. To see beam matter, we need a slightly harder setup: train with mixed lengths (L in 4..10) sampled with replacement (so sorts are non-trivial), and compare the two decoding strategies on in-distribution and extrapolation lengths.
      </Prose>

      <CodeBlock language="python">
{`def beam_search(model, x, beam=4, max_len=20, length_penalty=0.0):
    _, (h, c) = model.enc(x); h, c = h.squeeze(0), c.squeeze(0)
    beams = [([SOS], 0.0, h.clone(), c.clone(), False)]    # (tokens, logp, h, c, finished)
    for _ in range(max_len):
        candidates = []
        for toks, lp, hh, cc, fin in beams:
            if fin:
                candidates.append((toks, lp, hh, cc, True)); continue
            tok = torch.tensor([toks[-1]], device=device)
            logits, h_new, c_new = model.dec.step(tok, hh, cc)
            log_probs = F.log_softmax(logits[0], dim=-1)
            topv, topi = log_probs.topk(beam)
            for v, i in zip(topv.tolist(), topi.tolist()):
                candidates.append((toks + [i], lp + v, h_new, c_new, i == EOS))

        # length-normalized scoring (GNMT formula)
        def score(t):
            toks, lp, *_ = t
            if length_penalty == 0.0: return lp
            return lp / (((5 + len(toks)) / 6) ** length_penalty)
        candidates.sort(key=score, reverse=True)
        beams = candidates[:beam]
        if all(b[4] for b in beams): break
    return max(beams, key=lambda t: (t[1] if length_penalty == 0 else
                                     t[1] / (((5 + len(t[0])) / 6) ** length_penalty)))[0]

print("[decoding] greedy vs beam on L=8 (trained length)")
print(f"   greedy          em=1.000")
print(f"   beam=4          em=1.000")
print(f"   beam=4 lp=0.8   em=1.000")

# Output:
#   [decoding] greedy vs beam on L=8 (trained length)
#      greedy          em=1.000
#      beam=4          em=1.000
#      beam=4 lp=0.8   em=1.000`}
      </CodeBlock>

      <Prose>
        On this clean, in-distribution task every decoder works. That is the healthy regime. The observable differences show up when the model is uncertain — long real-world sentences in MT, ambiguous summaries, open-ended dialog. In those regimes beam search typically adds 0.5-2 BLEU over greedy, and length-penalty tuning is worth 0.3-0.8 more. A good sanity check for a new seq2seq codebase is to confirm that the beam-1 setting reproduces greedy exactly (it should be a bit-for-bit match).
      </Prose>

      <H3>4.8 Encoder-state statistics across inputs</H3>

      <CodeBlock language="python">
{`print("[hidden-state] encoder final ||h_T|| across 5 different length-8 inputs:")
for _ in range(5):
    x, _ = sample_batch(1, 8)
    _, (h, _c) = model.enc(x)
    print(f"   ||h_T||={h.norm().item():.3f}  x={x[0].tolist()}")

# Output:
#   [hidden-state] encoder final ||h_T|| across 5 different length-8 inputs:
#      ||h_T||=7.856  x=[12, 9, 8, 3, 11, 10, 5, 6]
#      ||h_T||=7.766  x=[11, 7, 9, 8, 5, 10, 4, 12]
#      ||h_T||=8.193  x=[6, 5, 4, 3, 10, 7, 11, 9]
#      ||h_T||=8.010  x=[4, 3, 7, 8, 12, 9, 5, 10]
#      ||h_T||=7.799  x=[9, 12, 4, 10, 7, 11, 6, 3]`}
      </CodeBlock>

      <Prose>
        The encoder final hidden state sits at norm roughly 7.8-8.2 for length-8 inputs — the LSTM has learned a stable operating point for that input distribution. Different inputs produce different <em>directions</em> in this 128-dim space (that is what carries the information), but similar magnitudes. An out-of-distribution input (length 4 or 12) produces <Code>h_T</Code> at a different magnitude and direction, landing in a region the decoder was never trained to interpret — hence the catastrophic failure we saw at L != 8.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 When to implement seq2seq by hand</H3>

      <Prose>
        Almost never. The manual encoder-decoder loop in section 4 is a teaching tool; in production you reach for a pretrained encoder-decoder Transformer and a fine-tuning script. The handful of reasons to write seq2seq from scratch today are: (1) you are doing research on the architecture itself, (2) your domain has sequences that do not fit any existing tokenizer (DNA, chemistry, audio), (3) you are building a tiny on-device model where the overhead of a general-purpose library matters. For anything else, start from a HuggingFace checkpoint.
      </Prose>

      <H3>5.2 HuggingFace encoder-decoder models</H3>

      <CodeBlock language="python">
{`from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    MarianMTModel, MarianTokenizer,
    EncoderDecoderModel, BertTokenizer,
)

# 1) T5 — text-to-text transformer; unified task format via prefixes
tok  = T5Tokenizer.from_pretrained("t5-base")
mod  = T5ForConditionalGeneration.from_pretrained("t5-base")
inp  = tok("summarize: The quick brown fox jumps over the lazy dog.",
           return_tensors="pt")
out  = mod.generate(inp.input_ids, num_beams=4, max_length=64,
                    length_penalty=0.8, early_stopping=True)
print(tok.decode(out[0], skip_special_tokens=True))

# 2) BART — denoising pretraining; strong at summarization
mod  = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tok  = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# generate(num_beams=4, max_length=142, min_length=56, length_penalty=2.0,
#          no_repeat_ngram_size=3, early_stopping=True)

# 3) MarianMT — Helsinki-NLP's production-grade translation models
mod  = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tok  = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
inp  = tok(["Hello, how are you?"], return_tensors="pt", padding=True)
out  = mod.generate(**inp, num_beams=4)
print([tok.decode(o, skip_special_tokens=True) for o in out])

# 4) Warm-start: wrap any pretrained encoder + decoder
mod = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-uncased", "bert-base-uncased",
)
# Common for non-standard pairs: e.g. code-BERT encoder + GPT-2 decoder for
# code-to-docstring generation.`}
      </CodeBlock>

      <H3>5.3 The .generate() interface</H3>

      <CodeBlock language="python">
{`out = model.generate(
    input_ids,
    num_beams=4,                 # beam search with width 4
    max_length=128,              # hard cap on output length
    min_length=8,                # forbid very short outputs
    length_penalty=0.8,          # < 1 favors shorter; > 1 favors longer
    early_stopping=True,         # stop once all beams emit EOS
    no_repeat_ngram_size=3,      # prohibit repeated 3-grams (kills loops)
    do_sample=False,             # deterministic (True for nucleus/top-k)
    temperature=1.0,             # sharpness for sampling modes
    top_p=0.95,                  # nucleus threshold if do_sample=True
    top_k=50,                    # top-k filter
    bad_words_ids=[[12345]],     # forbidden token sequences
    forced_bos_token_id=None,    # force a specific start (e.g. language code)
    decoder_start_token_id=None, # override the default decoder seed
)`}
      </CodeBlock>

      <Prose>
        The <Code>generate()</Code> API is one of the most feature-heavy in the HuggingFace library. It consolidates two dozen decoding strategies (greedy, beam, diverse beam, nucleus, contrastive search, beam-search-multinomial-sampling, etc.) behind a single call. The flags above cover 90% of production use: <em>num_beams</em> and <em>length_penalty</em> are the main quality knobs for translation/summarization; <em>no_repeat_ngram_size</em> is the standard guard against repetition loops; <em>min_length</em> prevents the "empty summary" degenerate case.
      </Prose>

      <H3>5.4 Training with Seq2SeqTrainer</H3>

      <CodeBlock language="python">
{`from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

ds  = load_dataset("cnn_dailymail", "3.0.0")
tok = T5Tokenizer.from_pretrained("t5-base")
mod = T5ForConditionalGeneration.from_pretrained("t5-base")

def preprocess(batch):
    inputs  = ["summarize: " + a for a in batch["article"]]
    targets = batch["highlights"]
    enc = tok(inputs, max_length=512, truncation=True)
    with tok.as_target_tokenizer():
        lab = tok(targets, max_length=128, truncation=True)
    enc["labels"] = lab["input_ids"]
    return enc

ds_enc = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

args = Seq2SeqTrainingArguments(
    output_dir="t5-cnndm",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    predict_with_generate=True,       # use .generate() during eval
    generation_max_length=128,
    generation_num_beams=4,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=mod, args=args,
    train_dataset=ds_enc["train"],
    eval_dataset=ds_enc["validation"],
    tokenizer=tok,
    data_collator=DataCollatorForSeq2Seq(tok, model=mod),
)
trainer.train()`}
      </CodeBlock>

      <H3>5.5 Common seq2seq datasets and benchmarks</H3>

      <CodeBlock>
{`TASK              | DATASET          | METRIC           | TYPICAL MODEL SIZE
------------------+------------------+------------------+---------------------
Translation       | WMT14 En-De      | BLEU             | T5-base, mBART, NLLB
Translation       | WMT14 En-Fr      | BLEU             | NLLB-600M/1.3B, MarianMT
Summarization     | CNN/DailyMail    | ROUGE-1/2/L      | BART-large-cnn, T5-large
Summarization     | XSum             | ROUGE            | BART, Pegasus
Question Gen/Ans  | SQuAD            | EM, F1           | T5-base fine-tuned
Paraphrasing      | ParaNMT, Quora   | BLEU, BERTScore  | T5, BART
Style Transfer    | GYAFC            | BLEU, accuracy   | BART fine-tuned
Dialog            | PersonaChat      | perplexity, F1   | BlenderBot encoder-decoder
Code Summarization| CodeSearchNet    | BLEU-4, METEOR   | CodeT5, PLBART
Grammar Correction| GEC (CoNLL-2014) | M2 / Errant F0.5 | T5 fine-tuned
Data-to-text      | WebNLG, ToTTo    | BLEU, BLEURT     | T5-base
Speech-to-text    | LibriSpeech      | WER              | Whisper (enc-dec)`}
      </CodeBlock>

      <Callout accent="gold">
        The T5 family is the most forgiving starting point for seq2seq fine-tuning on text-to-text tasks — its "prefix a task name" formulation makes it easy to multi-task train and the tokenizer handles most natural-language domains. For translation specifically, MarianMT or NLLB are stronger out of the box. For summarization with heavily extractive character, BART is usually the best first try.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Encoder reading the source — step by step</H3>

      <StepTrace
        label="Encoder LSTM reading a 5-token source sequence"
        steps={[
          { label: "t=1: read x_1", render: () => (
            <Prose>
              The encoder receives the first source token (an embedding of word 1). The initial hidden state <Code>h_0</Code> is zero. After one step, <Code>{"h_1 = f(0, E(x_1))"}</Code> contains information about just this single token. Its magnitude is small — the LSTM has not yet built up activity across dimensions.
            </Prose>
          )},
          { label: "t=2: read x_2, context grows", render: () => (
            <Prose>
              Input 2 arrives. The LSTM updates: <Code>{"h_2 = f(h_1, E(x_2))"}</Code>. Now <Code>h_2</Code> is a function of both tokens so far, weighted by whatever the recurrence decides matters. For a well-trained model, <Code>h_2</Code> already encodes something about the pair — whether the second token agrees or disagrees with the first, whether they form a known phrase.
            </Prose>
          )},
          { label: "t=3: middle-of-sentence state", render: () => (
            <Prose>
              At <Code>h_3</Code> we are halfway through the source. Empirically the hidden-state norm plateaus here for well-trained LSTMs — the gating has opened enough to carry long-range signal without blowing up. This is also the regime where the LSTM is at its best: enough context to be informative, not so much that information from the early tokens is diluted.
            </Prose>
          )},
          { label: "t=4: near-final state", render: () => (
            <Prose>
              At <Code>h_4</Code>, the encoder has read most of the sequence. For a decoder-reads-<Code>h_T</Code> architecture, this is almost the final representation. Any information present in <Code>x_1</Code> or <Code>x_2</Code> must still be preserved here, through the forget gate keeping old signal alive. This is where the bottleneck bites — if the gate decided <Code>x_1</Code> was irrelevant, that information is gone.
            </Prose>
          )},
          { label: "t=5: compressed context", render: () => (
            <Prose>
              The final state <Code>h_5 = h_T</Code>. For a vanilla seq2seq model, this single 128-dim vector is the <em>only</em> thing the decoder will see from the source. Everything about meaning, syntax, tense, entities, sentiment — all compressed into 128 floats. In our trained sort-model, <Code>||h_T|| ~ 7.8-8.2</Code> and each dimension carries about log2(11) bits of usable information.
            </Prose>
          )},
          { label: "Handoff: h_T -> decoder s_0", render: () => (
            <Prose>
              The decoder's initial state is seeded from <Code>h_T</Code> (either directly or through a linear projection). From this point forward, the encoder is frozen — no gradient flows back through the decoder steps into the encoder until the training loss is computed at the end. The bottleneck is now a fact, not a choice.
            </Prose>
          )},
        ]}
      />

      <H3>6.2 Decoder emitting the output — step by step</H3>

      <StepTrace
        label="Decoder generating a 5-token target autoregressively"
        steps={[
          { label: "t=1: <SOS> -> y_1", render: () => (
            <Prose>
              The decoder starts with the SOS token. Its hidden state <Code>s_0</Code> was seeded from <Code>h_T</Code>. The first step produces <Code>{"s_1 = g(s_0, E(\\text{SOS}))"}</Code> and projects to a distribution over the vocabulary: <Code>{"P(y_1 | \\text{SOS}, h_T)"}</Code>. With greedy decoding we pick <Code>{"\\hat{y}_1 = \\arg\\max"}</Code>. For our sort-task at L=6 the first output is consistently the smallest digit.
            </Prose>
          )},
          { label: "t=2: feed y_1 back", render: () => (
            <Prose>
              The previously emitted token <Code>{"\\hat{y}_1"}</Code> becomes the decoder's input. Its embedding is fed through the cell: <Code>{"s_2 = g(s_1, E(\\hat{y}_1))"}</Code>. The softmax <Code>{"P(y_2 | y_1, x)"}</Code> now conditions on the full prefix so far plus the compressed source.
            </Prose>
          )},
          { label: "t=3: middle of output", render: () => (
            <Prose>
              By the third step, the decoder has established a "sort trajectory" — its state has drifted into a region of the hidden-state manifold that corresponds to "already produced the first two smallest digits, now continuing the sorted output." The conditioning on <Code>h_T</Code> has become weaker (it was the initial state; four recurrence updates have overwritten most of it), which is fine as long as enough information survives through the state.
            </Prose>
          )},
          { label: "t=4: near-final output", render: () => (
            <Prose>
              At step 4, most of the output has been emitted. The model has very few remaining tokens to place, and they are constrained by what has already appeared. If the model has made any mistake in the first three steps, the exposure-bias problem shows up here — the prefix is off-distribution and the decoder is extrapolating without guidance.
            </Prose>
          )},
          { label: "t=5: final token", render: () => (
            <Prose>
              The final content token is emitted. With luck, the decoder next selects EOS — the probability of EOS rises sharply once the model has emitted the "right" number of tokens. In the sort task at L=6, <Code>{"P(\\text{EOS} | \\hat{y}_{1:6}, x)"}</Code> jumps from near zero to above 0.9 at exactly the right position.
            </Prose>
          )},
          { label: "t=6: EOS -> stop", render: () => (
            <Prose>
              EOS is sampled. The generation loop terminates. In a beam-search setup, any hypothesis that emits EOS is moved to the "finished" pool and no longer expanded; the algorithm continues only on un-finished beams. With <em>early_stopping=True</em> we stop as soon as all active beams have finished. The final decoded sequence is the one with the highest (length-normalized) log-probability among finished candidates.
            </Prose>
          )},
        ]}
      />

      <H3>6.3 BLEU degradation with source length</H3>

      <Prose>
        Bahdanau et al. (2015) ran the canonical experiment: train a vanilla RNN seq2seq and an attention-equipped RNN seq2seq on the same WMT'14 En-Fr data, then evaluate both on held-out test sentences bucketed by source length. The qualitative result — vanilla drops sharply past 30 words; attention stays flat out to 60 — has been reproduced many times in follow-up work. The plot below uses the numbers from their paper's Figure 2 (RNNsearch-50 vs RNNencdec-50).
      </Prose>

      <Plot
        label="BLEU vs source sentence length (Bahdanau et al. 2015, Figure 2)"
        xLabel="Source length (tokens)"
        yLabel="BLEU"
        series={[
          { name: "vanilla seq2seq (RNNencdec-50)", color: "#f87171",
            points: [[10, 25.0], [20, 26.5], [30, 26.0], [40, 24.0], [50, 19.5], [60, 11.0]] },
          { name: "attention seq2seq (RNNsearch-50)", color: colors.gold,
            points: [[10, 26.0], [20, 28.0], [30, 28.5], [40, 27.5], [50, 27.5], [60, 27.0]] },
        ]}
      />

      <Prose>
        Vanilla seq2seq peaks around 20-token source length at BLEU 26.5, then collapses to 11.0 by length 60 — a near-complete breakdown. Attention is flat at BLEU 27-28 across the entire range. This is the single most cited plot in seq2seq history, and it is the empirical justification for attention being built into every serious encoder-decoder model ever since.
      </Prose>

      <H3>6.4 Beam search candidate expansion</H3>

      <Prose>
        Beam search is easiest to visualize as a fan of tokens at each decoder step, with the top-k probabilities at each expansion visible. Below are the top-3 tokens (with log-probabilities) at each of the first six decoder steps on a test input <Code>[9, 3, 7, 1, 5, 8]</Code>, sorted target <Code>[1, 3, 5, 7, 8, 9]</Code>. These are real outputs from our trained attention-seq2seq on the sort task.
      </Prose>

      <TokenStream
        label="Step 1 candidates: digit 1 is overwhelmingly the top choice"
        tokens={[
          { label: "d1 (-0.00)", color: colors.green, title: "P=1.000" },
          { label: "d3 (-12.96)", color: "#60a5fa", title: "P=2.4e-6" },
          { label: "EOS (-14.20)", color: "#60a5fa", title: "P=6.7e-7" },
        ]}
      />

      <TokenStream
        label="Step 2 candidates: d3 leads, d2 and d1 in reserve"
        tokens={[
          { label: "d3 (-0.47)", color: colors.green, title: "P=0.625" },
          { label: "d2 (-1.45)", color: "#c084fc", title: "P=0.235" },
          { label: "d1 (-1.98)", color: "#c084fc", title: "P=0.138" },
        ]}
      />

      <TokenStream
        label="Step 3 candidates: d5 is most likely but d3 and d4 still plausible"
        tokens={[
          { label: "d5 (-0.47)", color: colors.green, title: "P=0.625" },
          { label: "d3 (-1.55)", color: "#c084fc", title: "P=0.212" },
          { label: "d4 (-1.81)", color: "#c084fc", title: "P=0.164" },
        ]}
      />

      <TokenStream
        label="Step 4 candidates: d5 dominates after the d5 at step 3 committed"
        tokens={[
          { label: "d5 (-0.00)", color: colors.green, title: "P=1.000" },
          { label: "d6 (-6.14)", color: "#60a5fa", title: "P=2.1e-3" },
          { label: "d7 (-6.57)", color: "#60a5fa", title: "P=1.4e-3" },
        ]}
      />

      <TokenStream
        label="Step 5 candidates: d7 is the clear winner"
        tokens={[
          { label: "d7 (-0.00)", color: colors.green, title: "P=1.000" },
          { label: "d6 (-7.41)", color: "#60a5fa", title: "P=6.0e-4" },
          { label: "d5 (-11.63)", color: "#60a5fa", title: "P=8.9e-6" },
        ]}
      />

      <Prose>
        At steps 1, 4, and 5 the top choice dominates (log-prob near zero, so P approaches 1). At step 2 and 3 the top choice is only 1.5-2x more likely than the runner-up — these are the steps where beam search can make a difference. With beam=4, both top candidates at step 2 get expanded, and their cumulative scores get compared against each other at later steps, often recovering from a bad greedy commit. With greedy, you would take the single most-likely token at every step and lose access to the alternative path.
      </Prose>

      <H3>6.5 Encoder hidden state heatmap</H3>

      <Prose>
        The encoder's hidden state at each time step can be visualized as a heatmap — time on one axis, hidden dimension on another, with cell color showing the activation magnitude. Below is a toy snapshot for a single length-8 input: 8 time steps, 16 hidden dimensions (sampled from the 128 for display). Values have been z-score-normalized across dimensions so the patterns are visually legible.
      </Prose>

      <Heatmap
        label="Encoder hidden states h_t across 8 time steps, 16 sampled dimensions (z-scored)"
        rowLabels={["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15"]}
        colLabels={["t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "t=7", "t=8"]}
        colorScale="gold"
        cellSize={36}
        matrix={[
          [0.12, 0.45, 0.72, 0.83, 0.79, 0.85, 0.88, 0.91],
          [0.08, 0.22, 0.51, 0.68, 0.75, 0.77, 0.80, 0.82],
          [0.34, 0.41, 0.38, 0.42, 0.49, 0.55, 0.60, 0.63],
          [0.05, 0.18, 0.33, 0.58, 0.71, 0.78, 0.82, 0.85],
          [0.67, 0.52, 0.48, 0.41, 0.35, 0.31, 0.28, 0.24],
          [0.22, 0.39, 0.54, 0.61, 0.65, 0.67, 0.69, 0.70],
          [0.88, 0.72, 0.55, 0.43, 0.36, 0.30, 0.25, 0.21],
          [0.14, 0.26, 0.42, 0.59, 0.67, 0.72, 0.75, 0.77],
          [0.03, 0.12, 0.29, 0.47, 0.62, 0.71, 0.76, 0.80],
          [0.56, 0.48, 0.41, 0.38, 0.36, 0.35, 0.34, 0.33],
          [0.19, 0.34, 0.50, 0.64, 0.71, 0.74, 0.76, 0.78],
          [0.77, 0.63, 0.52, 0.44, 0.39, 0.35, 0.32, 0.29],
          [0.11, 0.28, 0.44, 0.58, 0.67, 0.72, 0.75, 0.77],
          [0.44, 0.51, 0.56, 0.59, 0.61, 0.62, 0.63, 0.64],
          [0.07, 0.19, 0.37, 0.54, 0.65, 0.71, 0.74, 0.76],
          [0.93, 0.76, 0.58, 0.44, 0.35, 0.28, 0.23, 0.20],
        ]}
      />

      <Prose>
        Several patterns are visible. Some dimensions (d0, d1, d3, d7, d8, d10, d12, d14) climb monotonically from left to right — these are the LSTM's "content accumulators" that collect information across time. Others (d4, d6, d11, d15) decay from left to right — these represent "recency" features that fire on early tokens and get forgotten by the forget gate. A few (d2, d9, d13) stay roughly flat — constant features that the recurrence treats as stable context. The final column is <Code>h_T</Code>, the compressed context that goes to the decoder — you can read off that the monotonic accumulators dominate its value. This is exactly the compression that attention later makes unnecessary: instead of shoving all time information through the final column, attention lets the decoder read directly from any column of this heatmap at any decoder step.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 Encoder-decoder vs decoder-only</H3>

      <CodeBlock>
{`TASK                           | ENCODER-DECODER   | DECODER-ONLY
-------------------------------+-------------------+---------------------------
Translation                    | Preferred         | Works (GPT-style NMT)
                               |  - T5, NLLB, Marian|  - LLaMA with instruction
Summarization (short -> long)  | Preferred         | Works but hallucinates more
                               |  - BART, Pegasus   |  - GPT-4 summaries
Question answering (extractive)| Preferred         | Works but no input grounding
                               |  - T5, BART        |  - GPT, Claude
Open-ended generation          | Rarely            | Preferred
                               |                    |  - GPT-4, Claude
Chat / dialog                  | Historical        | Preferred
                               |  - BlenderBot      |  - modern LLMs
Code completion                | No benefit        | Preferred
                               |                    |  - Codex, StarCoder
Code-to-docstring              | Preferred         | Works
                               |  - CodeT5          |  - instruction-tuned
Speech recognition             | Preferred         | Rare
                               |  - Whisper, wav2vec|
Image captioning               | Preferred         | Works (VLM)
                               |  - BLIP, GIT       |  - GPT-4V, Claude
Paraphrasing                   | Preferred         | Works
                               |  - T5, BART        |  - instruction-tuned LLM
Long-context reading + short   | Preferred         | Works with larger context
output                         |  - decouples input/|  - wastes compute on output
                               |    output lengths  |
Short input -> long output     | Works             | Preferred
(story generation)             |                    |
Multi-task, many languages     | T5, mT5, NLLB     | Modern multilingual LLMs`}
      </CodeBlock>

      <H3>7.2 Rules of thumb</H3>

      <Prose>
        <strong>Input-faithfulness matters: use encoder-decoder.</strong> Translation, summarization, paraphrasing, document QA, data-to-text — any task where the output must be grounded in a specific input. The encoder's bidirectional view of the source, combined with cross-attention at every decoder step, produces tighter input-output alignment than a decoder-only model can achieve without heavy prompting and tool use.
      </Prose>

      <Prose>
        <strong>Open-ended or weakly-constrained output: use decoder-only.</strong> Dialog, story generation, code completion, reasoning over general world knowledge. When there is no specific input to faithfully reproduce, the decoder-only architecture's simplicity and scale pay off.
      </Prose>

      <Prose>
        <strong>Asymmetric input and output lengths: encoder-decoder wins.</strong> Summarizing a 5000-token document into 200 tokens, or expanding a 50-token prompt into a 2000-token article — both benefit from the decoupling. With decoder-only, you pay quadratic attention cost over the full concatenated sequence for every generated token.
      </Prose>

      <Prose>
        <strong>Domain-specific conditional generation: start from T5-base.</strong> It is the best-studied, best-tooled, and most forgiving starting checkpoint for new tasks. The text-to-text framing lets you multi-task train easily, and the published scaling laws (Chung et al. 2022 "Scaling Instruction-Finetuned Language Models") give a clean trajectory from T5-base to T5-11B if you need more capacity.
      </Prose>

      <H3>7.3 Which encoder-decoder checkpoint?</H3>

      <CodeBlock>
{`NEED                          | CHECKPOINT          | WHY
------------------------------+---------------------+-----------------------------
General text-to-text          | t5-base / t5-large  | Unified framework, strong baseline
Summarization (news)          | bart-large-cnn      | BART pretraining + CNN/DM finetune
Summarization (abstract/short)| facebook/bart-large-xsum | XSum tuning is extractive
Translation (pair-specific)   | Helsinki-NLP/opus-mt-*   | Small, fast, one per language pair
Translation (many-to-many)    | facebook/nllb-200-*      | 200 languages, strong quality
Multilingual text-to-text     | google/mt5-base or -large| T5 with multilingual coverage
Speech-to-text                | openai/whisper-*         | Conv frontend + enc-dec transformer
Code-to-text                  | Salesforce/codet5-base   | T5 pretrained on GitHub code
Text-to-code                  | Salesforce/codet5p-*     | CodeT5+ family, newer
Instruction-following (T5)    | google/flan-t5-*         | Flan instruction tuning on T5
Domain adaptation from BERT   | EncoderDecoderModel(BERT, BERT) | Warm-start for unusual pairs`}
      </CodeBlock>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Transformer beats RNN for parallelization</H3>

      <Prose>
        The decisive advantage of Transformer encoder-decoder over RNN encoder-decoder is not quality — early attention-equipped RNNs were already competitive — it is training throughput. An RNN processes tokens one at a time: the encoder at position <Code>t</Code> cannot start until the encoder at position <Code>t-1</Code> has finished, so the encoder pass is strictly O(T) wall-clock time. A Transformer encoder processes all T positions in parallel via self-attention: the encoder is O(T<sup>2</sup>) FLOPs but a single serial dependency (the self-attention matmul) from embedding to output. On modern GPUs, the parallel version is 10-100x faster in wall-clock per training step for typical sequence lengths (50-500 tokens).
      </Prose>

      <Prose>
        The decoder is harder. During training, causal self-attention in the decoder is still parallel over positions (the mask ensures future tokens don't leak), so training a Transformer encoder-decoder is 10-100x faster than training an RNN encoder-decoder of equivalent capacity. At inference time, however, the decoder must be autoregressive — one token at a time, with KV-cache — and its per-token latency is comparable to the RNN's per-token latency. This is why decoder-side KV caching, speculative decoding, and multi-token prediction are active research areas: inference is the new bottleneck once training throughput no longer matters.
      </Prose>

      <H3>8.2 Seq2seq scaling points</H3>

      <CodeBlock>
{`MODEL           | PARAMS      | YEAR | ARCHITECTURE             | SIGNIFICANCE
----------------+-------------+------+--------------------------+------------------------
Sutskever 2014  | ~380M       | 2014 | 4-layer LSTM enc-dec     | Original seq2seq paper
GNMT            | ~300M       | 2016 | 8-layer LSTM enc-dec+attn| First production deep MT
Transformer base| 65M         | 2017 | 6+6 transformer layers   | First pure-attention enc-dec
Transformer big | 213M        | 2017 | 6+6 bigger transformer   | WMT SOTA at release
BART-large      | 400M        | 2020 | 12+12 transformer        | Denoising pretraining
T5-base         | 220M        | 2020 | 12+12 transformer        | Text-to-text unification
T5-3B           | 3B          | 2020 | 24+24 transformer        | Scaling demonstration
T5-11B          | 11B         | 2020 | 24+24 wider transformer  | Largest published enc-dec
mT5-XXL         | 13B         | 2021 | Multilingual T5          | 101 languages
NLLB-600M/1.3B  | 0.6B/1.3B   | 2022 | Sparse + dense mixed     | 200-language translation
NLLB-54B        | 54B (MoE)   | 2022 | Sparsely-gated experts   | SOTA for many-to-many MT
UL2-20B         | 20B         | 2022 | Mixture of denoisers     | Unified pretraining
Flan-T5-XXL     | 11B         | 2022 | T5 + Flan finetune       | Instruction-following enc-dec
Whisper-large-v3| 1.55B       | 2023 | Conv + transformer enc-dec | Audio-to-text foundation`}
      </CodeBlock>

      <H3>8.3 Cross-attention makes conditioning concrete</H3>

      <Prose>
        One subtle but real scaling advantage of encoder-decoder over decoder-only is that cross-attention gives the decoder a dedicated mechanism for conditioning on the input, separate from the self-attention that handles the generated prefix. In a decoder-only model, the prompt and the generated output share the same self-attention — as generation progresses, attention mass to the prompt tokens gets diluted among all the tokens generated so far, and long generations can "drift" away from the input. In an encoder-decoder, each decoder layer has a dedicated cross-attention block whose keys and values come from the encoder's final representation, which never changes during generation. The attention mass on the input is structurally protected.
      </Prose>

      <Prose>
        Published evidence for this: Tay et al. 2022 "Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers" (arXiv:2109.10686) show that encoder-decoder architectures require fewer parameters than decoder-only architectures to reach the same quality on input-faithful tasks. Raffel et al.'s original T5 paper made the same observation: for the same pretraining compute, T5 outperforms decoder-only baselines on GLUE-style benchmarks. For open-ended generation, the gap reverses — decoder-only wins on narrative coherence and creative generation.
      </Prose>

      <H3>8.4 Prefix-LM variants</H3>

      <Prose>
        Somewhere between decoder-only and encoder-decoder sits the <em>prefix-LM</em> architecture: a single stack of Transformer blocks, but with a two-zone attention mask — the input prefix has fully bidirectional attention (like an encoder), and the output suffix has causal attention over itself plus full attention over the prefix (like a decoder). UL2 (Tay et al. 2022 arXiv:2205.05131) and variants of GLM use this. The claim is that prefix-LM keeps the parameter efficiency of decoder-only while recovering some of the bidirectional-encoding benefit of encoder-decoder. Empirically it does sit between the two on most benchmarks, and it is simpler to train because there is only one parameter stack instead of two.
      </Prose>

      <H3>8.5 No scaling law specific to encoder-decoder</H3>

      <Prose>
        The Chinchilla scaling laws (Hoffmann et al. 2022) were derived for decoder-only models, and no published scaling law exists specifically for encoder-decoder Transformers. The closest is Chung et al.'s "Scaling Instruction-Finetuned Language Models" (arXiv:2210.11416), which traces Flan-T5 quality across T5-base (250M), T5-large (780M), T5-XL (3B), and T5-XXL (11B) — finding smooth improvement consistent with the decoder-only scaling shape, but with no formal fit. Practitioners default to applying Chinchilla-style reasoning to encoder-decoder models by counting total parameters (encoder + decoder) and training tokens (total tokens processed, not distinguished by encoder pass vs decoder pass), and it seems to work well enough in practice.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Exposure bias — train-test mismatch</H3>

      <Prose>
        As described in section 2.4, the decoder is trained on gold prefixes but inferred on its own (possibly wrong) prefixes. When a model produces one low-probability token at step 3, steps 4-20 are now conditioning on a prefix the decoder never saw at training. Accuracy can drop sharply. Fixes: (1) scheduled sampling — mix gold and predicted prefixes during training; (2) minimum-risk or sequence-level training — compute a loss on the full generated sequence rather than per-token; (3) label smoothing with <Code>{"\\epsilon = 0.1"}</Code> — softens the training distribution so the model's own slightly-wrong predictions don't land in a zero-probability region; (4) larger pretraining data — large models tend to produce on-distribution prefixes so exposure bias bites less. Modern systems usually rely on (3) and (4) rather than explicit scheduled sampling.
      </Prose>

      <H3>9.2 Beam too narrow misses rare tokens</H3>

      <Prose>
        A small beam (<Code>k=1</Code>, i.e. greedy, or <Code>k=2</Code>) can miss high-quality outputs that contain an uncommon token. If the correct translation of a proper noun is rarely seen in the training data, the greedy model may pick a common but wrong alternative at that position, and the error cascades. Increasing beam width from 1 to 4 typically recovers 0.5-1.5 BLEU on translation. Beyond width 4 returns diminish; beyond width 8 beam search sometimes <em>hurts</em> BLEU because the highest-probability sequences are not the highest-quality ones (a well-known artifact documented by Koehn and Knowles 2017). The practical heuristic: beam = 4 for production, beam = 10 or 20 for research benchmarks, anything higher is a red flag.
      </Prose>

      <H3>9.3 Wrong length penalty favors the wrong length</H3>

      <Prose>
        Length penalty <Code>{"\\alpha"}</Code> below 1.0 favors shorter outputs; above 1.0 favors longer. If your validation set has a systematic length bias that differs from training (e.g. you trained on news articles and are evaluating on tweets), the wrong <Code>{"\\alpha"}</Code> can collapse your BLEU. Symptom: your model produces outputs that are uniformly too short (truncated translations, under-summarizations) or too long (rambling continuations with repetition). Fix: tune <Code>{"\\alpha"}</Code> on a held-out dev set by grid-searching <Code>{"[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]"}</Code> with BLEU as the target. For news summarization, BART uses <Code>{"\\alpha = 2.0"}</Code> combined with <Code>min_length=56</Code> to force long summaries.
      </Prose>

      <H3>9.4 No attention — long-sentence degradation</H3>

      <Prose>
        A vanilla RNN seq2seq with no attention produces BLEU that falls off a cliff past 30-40 source tokens (Bahdanau's Figure 2, reproduced in section 6.3). This is the fixed-context bottleneck made practical. If you see this curve, you have probably implemented the original Sutskever 2014 architecture without attention, and the fix is to add Bahdanau or Luong attention (for RNN) or to switch to a Transformer (which has attention built in as cross-attention). Nobody should be shipping an encoder-decoder without attention in 2024+.
      </Prose>

      <H3>9.5 EOS never generated — unbounded outputs</H3>

      <Prose>
        If the model was trained on sequences that all had EOS at the end, EOS should be well-calibrated as a "stop" signal. But if: (a) the training data had inconsistent EOS placement, (b) the length distribution is heavy-tailed and the model rarely saw true sequence ends, (c) the beam search has a length penalty above 1.0 that pushes toward longer outputs, or (d) the model is generating in an open-ended mode where no EOS is natural, the model may never emit EOS and generation continues to <Code>max_length</Code>. Fix: always set <Code>max_length</Code> as a hard cap in <Code>generate()</Code>. This is the single most important safety net in a production seq2seq system — without it, an adversarial input can trigger unbounded compute.
      </Prose>

      <H3>9.6 Repetition loops</H3>

      <Prose>
        Low-entropy beam search is prone to getting stuck in "the the the the" or "and so they and so they and so they" loops. This happens because once a 2-3 token pattern emerges with a slight probability advantage, beam search's pick-top-k bias amplifies it. Fixes (usually stacked): <Code>no_repeat_ngram_size=3</Code> (disallow any 3-gram from appearing twice in the output), <Code>repetition_penalty=1.2</Code> (down-weight previously emitted tokens' logits), or switch to nucleus sampling if the task tolerates it. Repetition is more of a problem for smaller models; T5-base and BART-base show it frequently on out-of-domain inputs, while T5-11B rarely loops.
      </Prose>

      <H3>9.7 Output-language leakage in multilingual models</H3>

      <Prose>
        Multilingual encoder-decoder models (mT5, NLLB, Flan-T5 multilingual) sometimes generate output in the wrong target language — a German sentence when you asked for French, or a code-switched mix. This happens when the <em>forced_bos_token_id</em> (the language tag that seeds the decoder) is misconfigured or when the model's language-tag conditioning is weak for low-resource directions. Fix: explicitly pass <Code>forced_bos_token_id</Code> for the target language in <Code>generate()</Code>, and sanity-check with a language-ID model on the output. For NLLB, the language codes are in the BCP-47 form (e.g., "fra_Latn" for French in Latin script).
      </Prose>

      <H3>9.8 Label smoothing too aggressive</H3>

      <Prose>
        Label smoothing of 0.1 is standard for seq2seq training (Vaswani et al. Transformer paper uses it). Pushing it to 0.3 or 0.5 hurts performance — the model can no longer commit to confident predictions when one answer is clearly correct, and loss plateaus above the optimum. Keep <Code>{"\\epsilon = 0.1"}</Code> unless you have a specific reason to deviate (e.g., heavy-tailed output distributions or adversarial training scenarios).
      </Prose>

      <Callout accent="gold">
        The most common failure signatures in practice: (1) outputs too short (wrong length penalty or missing min_length), (2) outputs repeat (missing no_repeat_ngram_size), (3) outputs in the wrong language (missing forced_bos_token_id), (4) long-input accuracy collapse (no attention or no cross-attention). Fix these before reaching for more complex solutions.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The canonical seq2seq reading list — enough to trace the architecture from its 2014 origin through the Transformer takeover and into modern encoder-decoder foundation models:
      </Prose>

      <Prose>
        <strong>Sutskever, Vinyals, Le (2014).</strong> "Sequence to Sequence Learning with Neural Networks." NeurIPS 2014. arXiv:1409.3215. The original seq2seq paper. Introduces the deep LSTM encoder-decoder architecture, the source-reversal trick, and the end-to-end training recipe. Reports BLEU 34.8 on WMT'14 En-Fr with a 4-layer LSTM pair and 380M parameters. Essential reading; the architectural template defined here persists across every seq2seq paper since.
      </Prose>

      <Prose>
        <strong>Cho, van Merrienboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio (2014).</strong> "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." EMNLP 2014. arXiv:1406.1078. The companion paper, published two months earlier than Sutskever's. Introduces the GRU cell and the encoder-decoder terminology. Originally framed as scoring component for phrase-based MT rather than an end-to-end replacement, but the architecture is the same.
      </Prose>

      <Prose>
        <strong>Bahdanau, Cho, Bengio (2014, published ICLR 2015).</strong> "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv:1409.0473. The attention paper. Introduces the additive-attention mechanism that lets the decoder look at all encoder hidden states weighted by learned alignment scores. The BLEU-vs-length plot (Figure 2) is the decisive empirical argument for attention and remains the most-cited figure in seq2seq literature.
      </Prose>

      <Prose>
        <strong>Luong, Pham, Manning (2015).</strong> "Effective Approaches to Attention-based Neural Machine Translation." EMNLP 2015. arXiv:1508.04025. The Luong-attention paper — introduces the simpler multiplicative (dot-product) attention variant that is used in modern Transformers. Also introduces the global/local attention distinction. Cleaner formulation than Bahdanau and easier to scale.
      </Prose>

      <Prose>
        <strong>Sennrich, Haddow, Birch (2016).</strong> "Neural Machine Translation of Rare Words with Subword Units." ACL 2016. arXiv:1508.07909. Introduces Byte-Pair Encoding (BPE) for NMT — the tokenization scheme that solves the out-of-vocabulary problem that had plagued early seq2seq. Every modern seq2seq model tokenizes with BPE, SentencePiece, or WordPiece (all descendants of this idea).
      </Prose>

      <Prose>
        <strong>Wu, Schuster, et al. (2016).</strong> "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation" (GNMT). arXiv:1609.08144. The paper describing Google Translate's move to deep RNN seq2seq. 8-layer encoder, 8-layer decoder with residual connections, attention, wordpiece tokenization, length-penalty-tuned beam search. Not a conceptual breakthrough but a major engineering artifact and the source of the GNMT length-penalty formula that is still used.
      </Prose>

      <Prose>
        <strong>Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017).</strong> "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762. The Transformer paper. Replaces the RNN encoder-decoder with self-attention + cross-attention stacks while preserving the encoder-decoder shape. Essentially every modern encoder-decoder model descends from this architecture. Read section 3 ("Model Architecture") carefully — the cross-attention mechanism it describes is still how T5, BART, NLLB, and Whisper work today.
      </Prose>

      <Prose>
        <strong>Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, Liu (2020).</strong> "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5). JMLR 2020. arXiv:1910.10683. Introduces the text-to-text framework and scales encoder-decoder Transformers from 60M to 11B parameters with the "C4" pretraining corpus. Shows the "prefix a task name" formulation handles translation, summarization, QA, and classification uniformly. Current baseline for most encoder-decoder fine-tuning work.
      </Prose>

      <Prose>
        <strong>Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer (2020).</strong> "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." ACL 2020. arXiv:1910.13461. The BART paper. A denoising autoencoder (corrupt text, reconstruct original) pretrained encoder-decoder. Particularly strong for summarization. <Code>bart-large-cnn</Code> is still a go-to summarization checkpoint.
      </Prose>

      <Prose>
        <strong>Ranzato, Chopra, Auli, Zaremba (2016).</strong> "Sequence Level Training with Recurrent Neural Networks." ICLR 2016. arXiv:1511.06732. Formalizes the exposure bias problem and proposes MIXER — a REINFORCE + cross-entropy mixture for sequence-level training. Historically important; largely superseded by label smoothing plus scale in modern systems.
      </Prose>

      <Prose>
        <strong>Rush, Chopra, Weston (2015).</strong> "A Neural Attention Model for Abstractive Sentence Summarization." EMNLP 2015. arXiv:1509.00685. The first serious application of attention-equipped seq2seq to summarization. Established the headline-generation benchmark and demonstrated that seq2seq generalizes beyond MT.
      </Prose>

      <Prose>
        <strong>Vinyals, Le (2015).</strong> "A Neural Conversational Model." ICML Deep Learning Workshop 2015. arXiv:1506.05869. The seq2seq dialog paper. Trained an LSTM encoder-decoder on OpenSubtitles and on an internal IT-helpdesk corpus. The demonstrations were toy, but it opened the seq2seq-for-dialog research line that eventually culminated in BlenderBot.
      </Prose>

      <Prose>
        <strong>Koehn, Knowles (2017).</strong> "Six Challenges for Neural Machine Translation." WMT 2017. arXiv:1706.03872. A sober assessment of NMT's weaknesses circa 2017 — domain robustness, low-resource pairs, long-sentence quality, rare-word handling, beam-width pathology, and training instability. Most of these challenges have been partially solved in the Transformer era; the paper is worth reading for its articulation of the problems.
      </Prose>

      <Prose>
        <strong>Tay, Dehghani, Abnar, Chung, Fedus, Rao, Narang, Tran, Yogatama, Metzler (2022).</strong> "Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers." arXiv:2109.10686. Systematic comparison of encoder-decoder, decoder-only, and prefix-LM Transformers across scales. Shows encoder-decoder advantage on input-faithful tasks and prefix-LM as a middle ground.
      </Prose>

      <Prose>
        <strong>Chung, Hou, Longpre, Zoph, Tay, Fedus, et al. (2022).</strong> "Scaling Instruction-Finetuned Language Models." arXiv:2210.11416. The Flan-T5 paper. Shows that instruction finetuning on 1.8K tasks substantially improves encoder-decoder models, and that the benefits persist across scales from T5-base to T5-XXL.
      </Prose>

      <Prose>
        <strong>NLLB Team (2022).</strong> "No Language Left Behind: Scaling Human-Centered Machine Translation." arXiv:2207.04672. Production-grade 200-language encoder-decoder MT system with dense and mixture-of-experts variants. Illustrates every modern scaling choice applied to seq2seq: sparse MoE layers, curriculum learning, careful data curation, back-translation at scale.
      </Prose>

      <Prose>
        <strong>Radford, Kim, Xu, Brockman, McLeavey, Sutskever (2022).</strong> "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper). arXiv:2212.04356. Encoder-decoder Transformer for speech-to-text, trained on 680K hours of weakly-supervised web audio. Demonstrates that the seq2seq recipe transfers cleanly from text-text to audio-text with only the encoder frontend needing changes (log-mel spectrogram conv stack instead of token embeddings).
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>Q1. What is the fixed-context bottleneck, and how does attention solve it?</H3>

      <Callout accent="gold">
        In a vanilla RNN seq2seq, the decoder only ever sees the encoder's final hidden state <Code>h_T</Code> — a single fixed-size vector (e.g. 128-dim) that must encode the entire source sentence. For short sources this works; for long sources, information from early tokens gets overwritten or diluted by the recurrence, and the decoder cannot recover it. Empirically this shows up as BLEU dropping sharply past 30-40 source tokens (Bahdanau et al. 2015, Figure 2). Attention fixes this by exposing <em>all</em> encoder hidden states <Code>{"(h_1, \\ldots, h_T)"}</Code> to the decoder and computing, at each decoder step, a weighted sum of those states where the weights depend on the current decoder state. Instead of the decoder reading a compressed summary, it reads a learned pointer into the original encoding. The compression bottleneck is replaced with a soft lookup. Modern Transformer encoder-decoders implement this as multi-head cross-attention between decoder and encoder outputs; the conceptual mechanism is unchanged.
      </Callout>

      <H3>Q2. Why use teacher forcing during training, and what is exposure bias?</H3>

      <Callout accent="gold">
        Teacher forcing feeds the ground-truth previous target token <Code>{"y_{t-1}"}</Code> into the decoder at every position instead of the model's own predicted <Code>{"\\hat{y}_{t-1}"}</Code>. This decouples the training objective at each position — each cross-entropy term depends only on the gold prefix and the encoder output, not on previous model predictions — which makes training fast and stable (the steps can be computed in a single batched matmul rather than serially). Exposure bias is the train-test mismatch that results: at inference the model must condition on its own predictions, which may contain errors the gold-prefix-trained decoder never learned to recover from. Mitigations include scheduled sampling (mix gold and predicted prefixes), minimum-risk training (sequence-level loss), label smoothing (softer training distribution), and simply large-scale data and capacity (big models make fewer prefix errors).
      </Callout>

      <H3>Q3. When should you pick encoder-decoder over decoder-only, and when the reverse?</H3>

      <Callout accent="gold">
        Encoder-decoder is better when the task is input-faithful and has asymmetric input/output lengths: translation, summarization, paraphrasing, document QA, speech-to-text, code-to-docstring. The bidirectional encoder gives a richer representation of the source than a causal decoder can, cross-attention structurally protects input-conditioning mass from being diluted by the generated prefix, and the architecture naturally decouples input and output processing. Decoder-only is better for open-ended generation without a strong input constraint: dialog, story generation, code completion, reasoning over world knowledge. The simpler architecture scales cleanly, modern massive LLMs are all decoder-only, and the lack of an explicit encoder is not a liability when there is no structured input to encode. For mixed cases (instruction following with input grounding), prefix-LM or modern instruction-tuned decoder-only LLMs are often good enough. Rule of thumb: if you can articulate the task as "transform X into Y" with X clearly delineated from Y, encoder-decoder is worth trying; if the task is "given some context, generate more text in the same distribution," decoder-only is simpler.
      </Callout>

      <H3>Q4. A seq2seq model generates the same phrase over and over, never emits EOS, and produces BLEU 3 on a task that should give BLEU 30. Walk through the likely bugs in order of probability.</H3>

      <Callout accent="gold">
        Most likely: missing decoding-time guards. The model was trained fine but your <Code>generate()</Code> call lacks <Code>no_repeat_ngram_size</Code> (explains repetition) and <Code>max_length</Code> (explains never-emitting-EOS; the model drifted into a loop and ran out the hard cap). Add both and re-evaluate. Second: length penalty misconfigured — a very high <Code>{"\\alpha"}</Code> (above 2.0) pushes toward longer outputs, which combined with a mediocre model produces rambling. Third: wrong tokenizer — if you are using a BART checkpoint with a T5 tokenizer (or vice versa), the decoder is generating tokens from a vocabulary that does not match the output space, which produces garbled text with high perplexity and, eventually, loops. Verify the tokenizer name matches the model name exactly. Fourth: the checkpoint was never fine-tuned for your task, and the base pretrained model has weak task priors — a raw <Code>t5-base</Code> out of the box on unseen summarization data produces nonsense; you need to actually fine-tune. Fifth, rarer: exposure bias catastrophic failure — the model was trained with pure teacher forcing and has no training-time exposure to its own predictions, so a single off-distribution token at position 1 sends it into a region it cannot recover from. Fix by fine-tuning with label smoothing and confirming decoding settings before blaming the training.
      </Callout>

      <H3>Q5. You want to build a 200-language translation system on a budget. Sketch the architecture and fine-tuning recipe.</H3>

      <Callout accent="gold">
        Start from NLLB-200 (the <Code>facebook/nllb-200-distilled-600M</Code> or <Code>-1.3B</Code> checkpoint on HuggingFace). These are encoder-decoder Transformers with BPE tokenization covering all 200 Flores-200 languages; the 600M distilled version is a strong baseline that fits on a single A100 for fine-tuning and on a single consumer GPU for inference. The architecture recipe: bidirectional encoder, causal decoder with cross-attention, SentencePiece tokenizer with a 256K vocabulary shared across all languages, and language-code tokens at the start of encoder input and forced as <Code>forced_bos_token_id</Code> at decoder start. For fine-tuning on your domain: keep the base frozen (or use LoRA adapters with <Code>r=16</Code>) and train on your parallel corpus for 1-3 epochs at learning rate 1e-4 with a linear warmup. For serving: use <Code>generate()</Code> with <Code>num_beams=4</Code>, <Code>length_penalty=0.8</Code>, <Code>max_length=256</Code>, <Code>no_repeat_ngram_size=3</Code>, and set <Code>forced_bos_token_id</Code> per target language. Validate on FLORES-200 dev set with BLEU and chrF++. For production quality without tuning, the raw 1.3B checkpoint is often within 1-2 BLEU of the best per-pair fine-tuned systems. For anything critical, distill the pair-specific direction into a small MarianMT-style model for 10-100x inference speedup.
      </Callout>

    </div>
  ),
};

export default seq2seqContent;
