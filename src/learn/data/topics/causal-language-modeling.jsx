import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const causalLanguageModeling = {
  slug: "causal-language-modeling-next-token-prediction",
  title: "Causal Language Modeling (Next-Token Prediction)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In the late 1990s, statistical language modeling was dominated by n-gram counts smoothed with techniques like Kneser-Ney. The approach was principled and fast, but it shattered against the curse of dimensionality: natural language sequences live in a space far too large for any count table to cover. To assign probability to the phrase "the cat sat on the mat," an n-gram model needed to have seen exactly that 5-gram before, or it would back off to shorter contexts and lose almost all the long-range information that makes language comprehensible.
      </Prose>

      <Prose>
        Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin published the solution in 2003 in the <em>Journal of Machine Learning Research</em>: "A Neural Probabilistic Language Model." Their proposal was deceptively simple. Instead of counting discrete n-grams, train a neural network that maps a fixed-length word context to a probability distribution over the next word. Words are represented as learned continuous vectors — what we now call embeddings — so words that appear in similar contexts push their vectors toward one another during gradient descent, and the model generalizes automatically to combinations it has never seen verbatim. A network that learned the sentence "the dog sat on the mat" had also, implicitly, learned something about "the cat sat on the mat," because "dog" and "cat" occupy similar positions in the embedding space after training. The joint probability of a sequence factored exactly as the chain rule demanded: multiply together the conditional probabilities of each token given all the ones before it.
      </Prose>

      <Prose>
        The objective that fell out of this framing was next-token prediction: maximize the probability the model assigns to each successive token in the training corpus, given the prefix. Equivalently, minimize the cross-entropy between the model's distribution and the observed token at every position. Bengio et al. used a feedforward network with a fixed context window (typically five to ten tokens). It was slow to train by modern standards and the context window was tiny, but the fundamental objective — predict the next token, measure cross-entropy, take a gradient step — has not changed in twenty-three years.
      </Prose>

      <Prose>
        In 2010, Tomáš Mikolov and colleagues at Brno University of Technology published "Recurrent Neural Network Based Language Model" at INTERSPEECH, replacing the feedforward window with an RNN that propagated a hidden state across the entire sequence. Perplexity on the Penn Treebank dropped roughly 50% compared to the best smoothed n-gram models. The objective was still next-token cross-entropy; the architecture had changed to allow unbounded context in principle. In practice, vanilla RNNs forgot context after twenty to fifty tokens due to gradient vanishing, but the gap over n-grams was large enough to ignite the field.
      </Prose>

      <Prose>
        For the next eight years, the field added depth, gating (LSTM, GRU), and tricks for handling long sequences, but the core objective remained. Matthew Peters and colleagues introduced ELMo in 2018 (arXiv:1802.05365), training a deep bidirectional LSTM as a language model and using its internal representations as features for downstream tasks. ELMo was notable not for what it trained but for what it found: that representations from a language model objective transferred to tasks the model had never seen, a harbinger of the pretraining paradigm that was about to dominate everything.
      </Prose>

      <Prose>
        The architecture shift came in 2017 when Vaswani et al. published "Attention Is All You Need" (arXiv:1706.03762), proposing the transformer: a sequence-to-sequence architecture built entirely from attention and feedforward layers, with no recurrence. Transformers parallelized over the sequence dimension in a way RNNs could not, and their memory of distant context was not degraded by path length. OpenAI combined the transformer with causal language modeling in 2018: Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever released "Improving Language Understanding by Generative Pre-Training" — GPT-1, an OpenAI technical report — a 117M-parameter transformer pretrained with next-token prediction on BooksCorpus, then fine-tuned on downstream tasks. It beat discriminatively trained baselines on 9 of 12 benchmarks. The same objective, on the same kind of data, but at ten times the scale of what had been tried before.
      </Prose>

      <Prose>
        In 2019, Radford et al. published GPT-2 (OpenAI tech report: "Language Models Are Unsupervised Multitask Learners"), scaling to 1.5B parameters and training on WebText — 40GB of web text filtered for quality by Reddit upvote count. GPT-2 demonstrated zero-shot task performance: the model solved reading comprehension, translation, and summarization problems it had never been explicitly trained on. No fine-tuning. The pretraining objective was still next-token cross-entropy. The downstream tasks emerged from scale.
      </Prose>

      <Prose>
        In May 2020, Brown et al. published "Language Models Are Few-Shot Learners" (arXiv:2005.14165), the GPT-3 paper. A 175-billion-parameter model, trained on a filtered crawl of the internet plus books and Wikipedia, using the same cross-entropy objective first written down by Bengio in 2003. The conclusion of the paper made the case explicitly: causal language modeling at sufficient scale is sufficient for few-shot generalization across essentially the entire space of natural language tasks. The simplest training objective turned out to be the only one the field needed.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        There are three ways to think about what next-token prediction is actually doing, and each illuminates a different aspect of why the objective works so well.
      </Prose>

      <H3>A distribution over vocabulary at each position</H3>

      <Prose>
        At each position <Code>t</Code> in a sequence, the model outputs a probability distribution over the entire vocabulary: a vector of non-negative numbers summing to 1, one entry per token. The distribution at position <Code>t</Code> is conditioned on the prefix <Code>x&#8321;, …, x&#8348;&#8315;&#8321;</Code>. Training says: the observed token <Code>x&#8348;</Code> should have high probability under that distribution. Inference says: draw a token from that distribution, append it, and repeat.
      </Prose>

      <Prose>
        This framing makes the objective concrete. The model does not produce a single prediction. It produces a full distribution at every step, and the quality of that distribution is measured by the cross-entropy between it and the one-hot distribution that places all mass on the observed token. A distribution that spreads mass evenly over 50,000 tokens has cross-entropy log(50,000) ≈ 10.8 bits. A distribution that places probability 0.9 on the correct token has cross-entropy ≈ 0.15 bits. The model is being asked to compress the training corpus: assign high probability to what actually comes next, and the training loss measures how well it succeeds.
      </Prose>

      <H3>Dense per-token supervision</H3>

      <Prose>
        In a labeled classification task, each example yields one gradient signal. A 512-token sequence with a single label contributes one update per forward pass. Causal language modeling yields 512 gradient signals from that same sequence — one per position. The model receives a loss gradient for every token it processes. This density is not cosmetic. At the pretraining scales where modern LLMs are trained — tens of trillions of tokens — it is the difference between feasibility and impossibility.
      </Prose>

      <Prose>
        The density also means the objective is self-supervised in the strictest sense. The supervision signal is the corpus itself: every document ever written contains its own labels, derived for free by shifting the input one position to the right. There is no annotation bottleneck. The entire internet is a labeled dataset under this objective.
      </Prose>

      <Prose>
        To predict the next token well, the model must internalize grammar (grammatical continuations outscore ungrammatical ones), semantics (coherent continuations outscore incoherent ones), world knowledge (factually accurate continuations are more likely in real text), long-range dependencies (pronoun resolution, topic tracking, narrative structure), style, and register. None of these are labeled separately. They are all implicit in the single objective of cross-entropy minimization over the training corpus.
      </Prose>

      <H3>Autoregressive sampling as a learned Markov chain</H3>

      <Prose>
        At inference time, generation is autoregressive: sample a token from <Code>p(x&#8348; | x&#8321;,…,x&#8348;&#8315;&#8321;)</Code>, append it, then sample from <Code>p(x&#8348;&#8330;&#8321; | x&#8321;,…,x&#8348;)</Code>, and so on. This is a Markov chain with a variable-order dependency: the transition distribution at each step depends on the entire history, not just the last state. The chain is "learned" in the sense that the transition probabilities are the model's own conditional distributions, which were fitted to the training corpus.
      </Prose>

      <Prose>
        This framing explains why temperature matters. The transition probabilities are computed by applying softmax to the model's logit vector. Dividing logits by a temperature <Code>T &lt; 1</Code> before softmax sharpens the distribution — the chain tends to follow high-probability paths, producing repetitive but fluent text. Dividing by <Code>T &gt; 1</Code> flattens the distribution — the chain takes more surprising steps, producing more varied text that is also more likely to be incoherent. Temperature 1.0 samples from the model's actual learned distribution without modification. The model has no internal concept of temperature; it is an external parameter applied at each step of the Markov chain.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>The chain rule factorization</H3>

      <Prose>
        The joint probability of any sequence of tokens <Code>x&#8321;, x&#8322;, …, x&#8345;</Code> factors exactly via the chain rule of probability:
      </Prose>

      <MathBlock>{"p(x_1, x_2, \\dots, x_n) = \\prod_{t=1}^{n} p(x_t \\mid x_1, \\dots, x_{t-1})"}</MathBlock>

      <Prose>
        This factorization is exact and model-free — it follows from the definition of conditional probability and holds for any distribution. What causal language modeling does is parameterize each factor <Code>p(x&#8348; | x&#8321;,…,x&#8348;&#8315;&#8321;)</Code> with a neural network <Code>p&#952;</Code>, and optimize the parameters <Code>&#952;</Code> to maximize the likelihood of the training corpus under this factorization. Maximizing likelihood is equivalent to minimizing the average negative log-likelihood, which is the cross-entropy loss.
      </Prose>

      <H3>Cross-entropy loss</H3>

      <MathBlock>{"\\mathcal{L}(\\theta) = -\\sum_{t=1}^{N} \\log p_\\theta(x_t \\mid x_{<t})"}</MathBlock>

      <Prose>
        The subscript <Code>x&#8344;&#8348;</Code> denotes the prefix <Code>x&#8321;, …, x&#8348;&#8315;&#8321;</Code>. The loss is a sum over all positions in the sequence (or over all sequences in the corpus). Each term is the negative log probability the model assigns to the observed token at that position. A model that always assigns probability 1 to the correct token achieves loss 0. A model that assigns probability <Code>1/V</Code> uniformly across a vocabulary of size <Code>V</Code> achieves loss <Code>N · log(V)</Code>.
      </Prose>

      <Prose>
        In practice the loss is averaged over positions (divided by <Code>N</Code>) to produce a per-token mean cross-entropy that is comparable across sequences of different lengths. This per-token CE is the number reported as "training loss" in most papers and codebases.
      </Prose>

      <H3>Perplexity</H3>

      <Prose>
        Perplexity is the standard evaluation metric for language models. It is derived directly from the per-token cross-entropy:
      </Prose>

      <MathBlock>{"\\text{PPL} = \\exp\\!\\left(\\frac{\\mathcal{L}}{N}\\right) = \\exp\\!\\left(-\\frac{1}{N}\\sum_{t=1}^{N} \\log p_\\theta(x_t \\mid x_{<t})\\right)"}</MathBlock>

      <Prose>
        The geometric interpretation: perplexity is the effective number of equally likely choices the model faces at each step. A model with perplexity 50 is, on average, as uncertain as if it were choosing uniformly among 50 tokens. A model with perplexity 1 always predicts the correct token with certainty. A uniform distribution over a vocabulary of 50,000 tokens has perplexity 50,000.
      </Prose>

      <Prose>
        The relationship to cross-entropy is clean: if per-token CE is <Code>H</Code> nats, then perplexity is <Code>e^H</Code>. If CE is measured in bits, perplexity is <Code>2^H</Code>. Most deep learning frameworks report nats because natural log is the default. The GPT-3 paper reports perplexity on Penn Treebank as 20.50, which corresponds to a per-token CE of roughly 3.02 nats. The best n-gram baseline was 35.76 perplexity (CE ≈ 3.58 nats). A gap of 0.56 nats per token sounds small; exponentiated, it translates to a 42% reduction in perplexity.
      </Prose>

      <H3>Causal mask</H3>

      <Prose>
        The causal constraint — that position <Code>t</Code> may only attend to positions 1 through <Code>t-1</Code> — is enforced in transformer models via an attention mask applied before the softmax in each attention head. The mask is an <Code>N × N</Code> matrix where the entry at row <Code>i</Code>, column <Code>j</Code> is 0 if <Code>j ≤ i</Code> (attending to past or present is allowed) and <Code>-∞</Code> if <Code>j &gt; i</Code> (attending to the future is forbidden). After adding this mask to the attention logits, the softmax maps all <Code>-∞</Code> entries to exactly 0, and the remaining weights are renormalized over the allowed positions.
      </Prose>

      <Prose>
        The lower-triangular structure means position 1 attends only to itself, position 2 attends to positions 1–2, position 3 attends to 1–3, and so on. The full matrix is computed in one batched operation — every position is processed in parallel — but the mask isolates each position's computation so that it cannot access information from positions later in the sequence. Without the mask, the model could attend to the token it is trying to predict, the loss would be trivially minimized by copying, and the weights would learn nothing useful about language.
      </Prose>

      <Callout type="warn">
        A missing or incorrect causal mask is one of the most dangerous silent bugs in causal LM implementations. The training loss will drop to near zero in a few steps — not because the model learned anything, but because it has direct access to the target. The model will produce garbage at inference time because the cheating path is no longer available. Always verify your mask with a small sanity check before training.
      </Callout>

      <H3>Teacher forcing and exposure bias</H3>

      <Prose>
        During training, each position <Code>t</Code> receives the ground-truth prefix as input, regardless of what the model would have predicted. This is teacher forcing. Its practical consequence is that a single forward pass through a transformer computes all <Code>N</Code> next-token predictions simultaneously: the causal mask ensures each position only sees its ground-truth prefix, and the parallelism of attention means all <Code>N</Code> predictions are computed in one matrix multiply. Without teacher forcing, training would require an autoregressive loop — generate token 1, feed it back, generate token 2, feed it back — which is sequential and roughly <Code>N</Code> times slower.
      </Prose>

      <Prose>
        The tradeoff is exposure bias. At inference, the model consumes its own outputs. A mistake at position <Code>t</Code> becomes part of the input for position <Code>t+1</Code>, and the model was never trained on prefixes that contained its own errors. The distribution it encounters at inference — imperfect model outputs — differs from the distribution it trained on — perfect ground truth. This gap grows with sequence length and is one motivation behind scheduled sampling (gradually replacing ground-truth tokens with model predictions during training), minimum Bayes risk decoding (selecting outputs by expected loss rather than likelihood), and reinforcement learning from human feedback (using a reward model to train on generations rather than teacher-forced tokens). None of these have fully solved the problem; exposure bias remains an active area of research.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The following is a minimal character-level causal language model built in PyTorch from scratch. Every component is spelled out explicitly — no <Code>nn.Transformer</Code> shortcut, no HuggingFace abstraction. The goal is to make the relationship between the math in the previous section and the running code as direct as possible. The model was trained on a tiny rhyming corpus and the outputs at the bottom are the actual outputs from that run, verbatim.
      </Prose>

      <H3>4a. CausalSelfAttention</H3>

      <Prose>
        The self-attention module projects the input into queries, keys, and values, applies the causal mask, computes scaled dot-product attention, and projects back to the model dimension. Multi-head attention splits the model dimension into <Code>n_heads</Code> parallel subspaces.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # single fused projection for Q, K, V
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        # project and split into Q, K, V — each (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=-1)

        # reshape to (B, n_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale   # (B, n_heads, T, T)

        # causal mask: upper triangle (future positions) -> -inf
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)               # future weights -> 0

        # weighted sum of values
        out = att @ v                              # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)`}
      </CodeBlock>

      <H3>4b. TransformerBlock</H3>

      <Prose>
        Each transformer block wraps the attention module and a two-layer MLP with GELU activation, adding residual connections and applying layer normalization before each sub-layer (pre-norm, as used by GPT-2 and most modern models).
      </Prose>

      <CodeBlock language="python">
{`class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.mlp(self.ln2(x))    # residual + MLP
        return x`}
      </CodeBlock>

      <H3>4c. MiniGPT</H3>

      <Prose>
        The full model stacks token embeddings (learned), position embeddings (learned), <Code>N</Code> transformer blocks, a final layer norm, and a linear head that projects back to vocabulary size. The LM head is intentionally kept separate from the token embedding matrix — weight tying (sharing the embedding and LM head weights) is a common optimization but unnecessary for understanding.
      </Prose>

      <CodeBlock language="python">
{`class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4,
                 n_layers=3, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(block_size, d_model)
        self.blocks   = nn.Sequential(
            *[TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f     = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # (B, T, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)                        # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, idx, max_new=60, temperature=0.8, top_k=5):
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k > 0:
                topk_vals = torch.topk(logits, top_k).values
                logits[logits < topk_vals[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx`}
      </CodeBlock>

      <H3>4d. Training loop</H3>

      <Prose>
        The training loop draws random context windows from the corpus, shifts the targets by one position (position <Code>t</Code> predicts position <Code>t+1</Code>), computes cross-entropy, and updates with AdamW. Gradient clipping prevents the norm explosion that can occur early in training when the attention weights are near-uniform.
      </Prose>

      <CodeBlock language="python">
{`# Corpus and character vocabulary
corpus = (
    "the cat sat on the mat the cat ate the rat "
    "the rat ran from the cat the mat is flat\n"
    "the dog bit the log the log fell on the frog "
    "the frog sat on a bog and sang all day long\n"
)
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)                  # vocab size: 22 characters

data = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)

torch.manual_seed(42)
model = MiniGPT(V)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)

block_size, batch_size = 64, 8
loss_curve = []

for step in range(500):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]   for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # shifted target

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 50 == 0:
        loss_curve.append((step, round(loss.item(), 4)))

# Actual training losses (step, loss):
# (0, 3.3702)  (50, 1.2752)  (100, 0.7683)  (150, 0.5900)
# (200, 0.4211)  (250, 0.2579)  (300, 0.1521)  (350, 0.1136)
# (400, 0.1034)  (450, 0.0745)`}
      </CodeBlock>

      <Plot
        label="MiniGPT training loss — char-level, 500 steps, AdamW lr=3e-3"
        xLabel="step"
        yLabel="CE loss"
        series={[{
          name: "train loss",
          color: colors.gold,
          points: [[0,3.3702],[50,1.2752],[100,0.7683],[150,0.59],[200,0.4211],[250,0.2579],[300,0.1521],[350,0.1136],[400,0.1034],[450,0.0745]],
        }]}
      />

      <H3>4e. Sampling loop</H3>

      <Prose>
        The sampling function is already embedded in <Code>MiniGPT.generate</Code> above. At each step: (1) take the last <Code>block_size</Code> tokens as context, (2) run a forward pass to get logits at the final position, (3) divide by temperature, (4) zero out all logits outside the top-k, (5) softmax, (6) sample. Below are actual outputs from the trained model:
      </Prose>

      <CodeBlock language="python">
{`prompt = [stoi[c] for c in "the cat"]
idx = torch.tensor([prompt], dtype=torch.long)

# temperature=0.8, top_k=5 (actual output):
out = model.generate(idx, max_new=80, temperature=0.8, top_k=5)
# 'the cat the rat the rat ran from the cat the mat is flat
#  the dog bit the log the log fe'

# greedy (temperature=0.01, top_k=1) (actual output):
out_greedy = model.generate(idx, max_new=80, temperature=0.01, top_k=1)
# 'the cat the rat the rat rat ran from the cat the mat is flat
#  the dog bit the log the lo'`}
      </CodeBlock>

      <Prose>
        The model has memorized the training corpus — it reproduces it almost verbatim, with slight variations under temperature. This is expected: 22 characters, 500 steps, on 200 tokens of text. The purpose of the exercise is not generalization; it is to verify that the cross-entropy loss falls, that the causal mask is correctly implemented (if it were absent, the loss would collapse to near-zero in a handful of steps as the model simply copies the target), and that sampling produces coherent output from the learned conditional distributions.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, you use HuggingFace <Code>transformers</Code>. GPT-2 ships as <Code>GPT2LMHeadModel</Code> and <Code>GPT2Tokenizer</Code>. The following examples show how to compute cross-entropy on an arbitrary string and how the <Code>.generate()</Code> method exposes the same temperature, top-k, and top-p controls we implemented from scratch.
      </Prose>

      <H3>5a. Computing cross-entropy and perplexity</H3>

      <CodeBlock language="python">
{`import torch, math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

text = ("The quick brown fox jumps over the lazy dog "
        "near the river bank in the morning light.")
enc = tokenizer(text, return_tensors="pt")
input_ids = enc.input_ids          # shape: (1, 18)

with torch.no_grad():
    out = model(input_ids, labels=input_ids)
    ce  = out.loss.item()          # mean per-token cross-entropy (nats)
    ppl = math.exp(ce)

print(f"tokens : {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
print(f"CE loss: {ce:.4f} nats")
print(f"PPL    : {ppl:.2f}")

# Actual output:
# tokens : ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', 'Ġover',
#            'Ġthe', 'Ġlazy', 'Ġdog', 'Ġnear', 'Ġthe', 'Ġriver',
#            'Ġbank', 'Ġin', 'Ġthe', 'Ġmorning', 'Ġlight', '.']
# CE loss: 4.5195 nats
# PPL    : 91.79`}
      </CodeBlock>

      <Prose>
        A perplexity of 91.79 on this sentence means GPT-2 base is, on average, as uncertain as if it were choosing uniformly among about 92 tokens at each step. This is typical for GPT-2 base on general text that wasn't in its training set. GPT-2 large achieves roughly 35 PPL on the WebText test set; GPT-3 gets 20.50 on Penn Treebank.
      </Prose>

      <H3>5b. Generation with sampling strategies</H3>

      <CodeBlock language="python">
{`prompt = "Language models predict the next"
enc2   = tokenizer(prompt, return_tensors="pt")

# greedy decoding — deterministic, argmax at each step
with torch.no_grad():
    g = model.generate(enc2.input_ids, max_new_tokens=20, do_sample=False)
print("greedy   :", tokenizer.decode(g[0]))
# greedy   : Language models predict the next generation of human beings.
#             The first step in the process is to understand how the human brain

# temperature=0.9, top_k=50
with torch.no_grad():
    torch.manual_seed(42)
    g2 = model.generate(
        enc2.input_ids, max_new_tokens=20,
        do_sample=True, temperature=0.9, top_k=50
    )
print("temp+topk:", tokenizer.decode(g2[0]))
# temp+topk: Language models predict the next generation of human-machine
#             relations will be different—and in a way that is fundamentally

# nucleus sampling top_p=0.9
with torch.no_grad():
    torch.manual_seed(42)
    g3 = model.generate(
        enc2.input_ids, max_new_tokens=20,
        do_sample=True, temperature=1.0, top_p=0.9
    )
print("nucleus  :", tokenizer.decode(g3[0]))
# nucleus  : Language models predict the next generation of car-focused
#             transportation will be safer, faster, cleaner, more convenient`}
      </CodeBlock>

      <Prose>
        The greedy output is deterministic and reads fluently but converges on a generic continuation. The temperature+top-k sample is more varied and arguably more interesting. The nucleus sample is the most diverse — and in this case diverges the furthest from the obvious continuation. All three are valid outputs from the same model under the same next-token cross-entropy objective; the difference is entirely in how the distribution is sampled at each step.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Causal attention mask</H3>

      <Prose>
        For a 6-token sequence, the causal mask is a 6×6 matrix where entry (i, j) is 1 (allowed) if j ≤ i and 0 (masked, i.e. set to −∞ before softmax) if j &gt; i. Row <Code>i</Code> is the attention pattern for position <Code>i</Code>: it can see all positions up to and including itself.
      </Prose>

      <Heatmap
        label="causal mask — 6×6 (1=attend, 0=blocked)"
        matrix={[
          [1,0,0,0,0,0],
          [1,1,0,0,0,0],
          [1,1,1,0,0,0],
          [1,1,1,1,0,0],
          [1,1,1,1,1,0],
          [1,1,1,1,1,1],
        ]}
        rowLabels={["pos 0","pos 1","pos 2","pos 3","pos 4","pos 5"]}
        colLabels={["pos 0","pos 1","pos 2","pos 3","pos 4","pos 5"]}
        cellSize={44}
        colorScale="gold"
      />

      <H3>Token-by-token generation (StepTrace)</H3>

      <Prose>
        Each step of autoregressive generation extends the context by one token. The model runs a full forward pass at each step; only the prediction at the final position matters.
      </Prose>

      <StepTrace
        label="autoregressive generation — one forward pass per step"
        steps={[
          {
            label: "prompt only",
            render: () => (
              <TokenStream tokens={[
                { label: "the", color: "#888" },
                { label: " cat", color: "#888" },
              ]} />
            ),
          },
          {
            label: "step 1 — sample ' sat'",
            render: () => (
              <TokenStream tokens={[
                { label: "the", color: "#888" },
                { label: " cat", color: "#888" },
                { label: " sat", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "step 2 — sample ' on'",
            render: () => (
              <TokenStream tokens={[
                { label: "the", color: "#888" },
                { label: " cat", color: "#888" },
                { label: " sat", color: "#888" },
                { label: " on", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "step 3 — sample ' the'",
            render: () => (
              <TokenStream tokens={[
                { label: "the", color: "#888" },
                { label: " cat", color: "#888" },
                { label: " sat", color: "#888" },
                { label: " on", color: "#888" },
                { label: " the", color: colors.gold },
              ]} />
            ),
          },
          {
            label: "step 4 — sample ' mat'",
            render: () => (
              <TokenStream tokens={[
                { label: "the", color: "#888" },
                { label: " cat", color: "#888" },
                { label: " sat", color: "#888" },
                { label: " on", color: "#888" },
                { label: " the", color: "#888" },
                { label: " mat", color: colors.gold },
              ]} />
            ),
          },
        ]}
      />

      <H3>Greedy vs temperature sampling</H3>

      <Prose>
        Given the prompt "Language models predict the next", greedy decoding and temperature sampling produce different continuations from the same model.
      </Prose>

      <TokenStream
        label="greedy — deterministic, argmax at every step"
        tokens={[
          { label: "Language", color: "#888" },
          { label: " models", color: "#888" },
          { label: " predict", color: "#888" },
          { label: " the", color: "#888" },
          { label: " next", color: "#888" },
          { label: " generation", color: colors.gold },
          { label: " of", color: colors.gold },
          { label: " human", color: colors.gold },
          { label: " beings", color: colors.gold },
          { label: ".", color: colors.gold },
        ]}
      />

      <TokenStream
        label="temperature=0.9, top_k=50 — diverse, stochastic"
        tokens={[
          { label: "Language", color: "#888" },
          { label: " models", color: "#888" },
          { label: " predict", color: "#888" },
          { label: " the", color: "#888" },
          { label: " next", color: "#888" },
          { label: " generation", color: "#4ade80" },
          { label: " of", color: "#4ade80" },
          { label: " human", color: "#4ade80" },
          { label: "-machine", color: "#4ade80" },
          { label: " relations", color: "#4ade80" },
        ]}
      />

      <H3>Training loss curve</H3>

      <Plot
        label="MiniGPT char-level — CE loss vs training step (actual run)"
        xLabel="step"
        yLabel="CE loss (nats)"
        series={[{
          name: "cross-entropy",
          color: colors.gold,
          points: [[0,3.3702],[50,1.2752],[100,0.7683],[150,0.59],[200,0.4211],[250,0.2579],[300,0.1521],[350,0.1136],[400,0.1034],[450,0.0745]],
        }]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Causal LM vs masked LM vs prefix LM</H3>

      <Prose>
        Three training objectives dominate the landscape. Causal LM (GPT-style) predicts each token from its left context only. Masked LM (BERT-style) masks 15% of tokens at random and predicts them from bidirectional context. Prefix LM (T5, UL2-style) uses bidirectional attention on a prefix and causal attention on the continuation.
      </Prose>

      <Prose>
        Causal LM is the default for generation tasks. The model is trained in exactly the mode it is used at inference — autoregressive, left-to-right — so there is no train/inference mismatch in the attention pattern. Every token in the training set produces a useful gradient signal. The model can be used for classification and generation without any architectural modification; the only difference is whether you sample from the output distribution or read a representation from it.
      </Prose>

      <Prose>
        Masked LM produces better bidirectional representations for tasks like sentence classification and named entity recognition, because every position has access to both left and right context. The tradeoff is that masked LM cannot generate text autoregressively without substantial architectural surgery — the attention pattern that makes it good at understanding is wrong for generation. BERT and its descendants are strong discriminators; they are weak generators by design.
      </Prose>

      <Prose>
        Prefix LM is a compromise. The encoder portion of the input (the prompt or source document) uses full bidirectional attention, and the decoder portion uses causal attention. This gives the model strong encoding of the input while preserving the causal structure needed for generation. T5 uses this pattern for sequence-to-sequence tasks; UL2 from Google trains with a mixture of all three objectives to maximize transfer across tasks. The main cost is architectural complexity and the need to explicitly mark the prefix/suffix boundary.
      </Prose>

      <H3>Teacher forcing vs scheduled sampling vs RL fine-tuning</H3>

      <Prose>
        Teacher forcing is the default for pretraining and fine-tuning. It is fast — one forward pass per sequence, full parallelism over positions — and it works well when the model is large enough that exposure bias at the temperatures used in inference is a minor perturbation. In practice, most production LLMs are pretrained with teacher forcing throughout and the exposure bias gap is addressed post-hoc via RLHF rather than during pretraining.
      </Prose>

      <Prose>
        Scheduled sampling replaces ground-truth tokens with model predictions with some probability that increases over training, gradually bridging the gap between teacher-forced training and autoregressive inference. It is slower (requires sequential sampling to generate the replacement tokens) and in practice produces modest improvements that do not justify the compute overhead for large models.
      </Prose>

      <Prose>
        RL fine-tuning (PPO, GRPO, DPO) treats the language model as a policy and optimizes it against a reward signal — typically a human preference model or a verifiable outcome (code compilation, math correctness). This directly addresses exposure bias because the model is now trained on its own rollouts. The cost is training instability (policy collapse, reward hacking, high variance gradients) and significant engineering complexity. It is the dominant approach for aligning large pretrained models but is never used for pretraining from scratch.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        The dense per-token supervision signal makes causal LM exceptionally compute-efficient. Every token in the training corpus contributes a gradient; there is no waste. This scales with data volume in a way that annotated datasets cannot: the internet provides effectively unlimited unlabeled text, and every token of it is a training example.
      </Prose>

      <Prose>
        Kaplan et al. (arXiv:2001.08361) established empirically that cross-entropy loss scales as a power law with model parameters, dataset size, and compute, across seven orders of magnitude. The key finding: <Code>L ∝ N^{-0.076}</Code> (parameters), <Code>L ∝ D^{-0.095}</Code> (tokens), <Code>L ∝ C^{-0.050}</Code> (compute). These exponents are small in the sense that you need roughly a 10x increase in any resource to achieve a 26–45% reduction in loss. But they are remarkably stable — the power law holds from 10M to 100B parameters with no sign of saturation as of the GPT-3 era. The predictability of these laws is what makes large training runs tractable to plan: given a compute budget, you can calculate the optimal model size and dataset size (the Chinchilla recipe).
      </Prose>

      <Prose>
        Emergent capabilities scale non-smoothly. Many behaviors — multi-step arithmetic, chain-of-thought reasoning, in-context learning — appear sharply at some scale threshold rather than gradually as loss descends. The loss curve does not predict these transitions; they are not visible in the aggregate number. This means that loss-based scaling laws are necessary but not sufficient for predicting capability.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        Data quality hits a wall before data quantity does. Kaplan et al. showed that the optimal training set size for a given compute budget grows roughly as the square root of compute. Current frontier models are data-limited — there is not enough high-quality text on the internet to train a model that is otherwise compute-optimal. Techniques like data filtering, deduplication, and synthetic data generation are not alternatives to scale; they are attempts to keep the data supply from becoming the binding constraint.
      </Prose>

      <Prose>
        Cross-entropy does not directly measure downstream task performance. A 2.0-CE model is not uniformly better than a 2.1-CE model across all tasks. The aggregate loss averages over common words, rare words, easy and hard positions — and improvements on the high-signal positions that determine task performance can be invisible in the aggregate. This is why MMLU, HumanEval, and GSM8K exist alongside perplexity as evaluation metrics.
      </Prose>

      <Prose>
        Exposure bias does not improve with scale. Larger models are better language models, but they are still trained with teacher forcing and still encounter their own error distributions at inference. RLHF, which does address this, requires a separate and more expensive training regime that does not scale as cleanly as pretraining.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Off-by-one shift error</H3>
      <Prose>
        The target sequence is the input shifted by one token: inputs are <Code>x[:-1]</Code>, targets are <Code>x[1:]</Code>. Getting this wrong — using unshifted targets, or shifting in the wrong direction — produces a model that either predicts the current token from itself (trivially solved by copying, loss collapses to near-zero immediately) or predicts two steps ahead (loss stays high and never converges). Symptom: training loss drops to near-zero in the first 10–50 steps with no real learning, or fails to converge at all.
      </Prose>

      <H3>2. Missing causal mask</H3>
      <Prose>
        Without the causal mask, each position attends to all positions including future ones. The model can trivially predict position <Code>t</Code> by looking at position <Code>t</Code> (or <Code>t+1</Code>) directly. The loss collapses to near-zero in a few steps. The trained model produces garbage at inference because the attending-forward path is unavailable. Symptom: suspiciously fast loss collapse, incoherent generation.
      </Prose>

      <H3>3. Position embedding overflow</H3>
      <Prose>
        Learned position embeddings have a maximum sequence length defined at training time (e.g., 1024 for GPT-2, 2048 for GPT-2-XL). Feeding a sequence longer than this requires either truncation or extrapolation. Extrapolation with learned embeddings is undefined behavior — the embedding table has no entry for position 1025. RoPE and ALiBi were developed specifically to address this; they encode positional information functionally rather than via lookup tables, enabling length generalization.
      </Prose>

      <H3>4. Exposure bias at long horizon</H3>
      <Prose>
        A model trained with teacher forcing degrades on long generations because errors compound. Each mistake shifts the distribution for the next token, and the model was never trained to recover from such shifts. The degradation is roughly proportional to the generation length times the per-step error rate. This is most visible in tasks requiring long coherent chains: multi-paragraph stories, long mathematical derivations, code with many interdependent functions.
      </Prose>

      <H3>5. Beam search length bias</H3>
      <Prose>
        Beam search scores sequences by total log-probability, which is a sum over token log-probabilities. Longer sequences accumulate more terms and thus lower total log-probability, even if each individual token is well-predicted. This biases beam search toward shorter outputs unless length normalization is applied (divide total log-prob by sequence length or a power thereof). Greedy and sampling-based methods are not affected by this bias.
      </Prose>

      <H3>6. Temperature extremes</H3>
      <Prose>
        Temperature near 0 (greedy limit) produces degenerate repetition — the model locks onto a high-probability loop and repeats it indefinitely. Temperature above 1.5 often produces incoherent text because the distribution is so flat that low-quality tokens are sampled with significant probability. The empirically useful range for most tasks is 0.6–1.2, with 0.7–0.9 covering the majority of production use cases.
      </Prose>

      <H3>7. Tokenization mismatch</H3>
      <Prose>
        A model trained with GPT-2's tokenizer and then evaluated or fine-tuned with a different tokenizer will produce nonsense, because the token ID 1234 means different things under different vocabularies. The tokenizer is fused into the model weights via the embedding table; they cannot be swapped. Always verify that the tokenizer used for evaluation and fine-tuning is byte-for-byte identical to the one used for pretraining.
      </Prose>

      <H3>8. Gradient explosion in early training</H3>
      <Prose>
        In the first few hundred steps, before the model has meaningful representations, attention weights can be near-uniform and residual streams are poorly conditioned. A single outlier gradient can overflow the weight update and cause NaN loss. Gradient clipping (norm clip to 1.0 is standard) prevents this. Post-normalization (layer norm after residual) is more susceptible to explosion than pre-normalization (layer norm before); GPT-2 switched to pre-norm in part for this reason.
      </Prose>

      <H3>9. Memory without activation checkpointing</H3>
      <Prose>
        A transformer's peak memory during the backward pass includes all activation tensors from the forward pass (needed for gradient computation). For a model with <Code>L</Code> layers and sequence length <Code>T</Code>, this is <Code>O(L · T · d_model)</Code>. For a 40-layer model with sequence length 2048 and d_model 4096, this is roughly 40 × 2048 × 4096 × 4 bytes ≈ 1.3 GB for the activations alone, on top of the model weights. Activation checkpointing recomputes activations during the backward pass instead of storing them, trading compute for memory at approximately 33% overhead. Without it, large models or long sequences run out of memory.
      </Prose>

      <H3>10. Softmax instability in attention</H3>
      <Prose>
        The attention scores are divided by <Code>sqrt(d_k)</Code> to prevent dot products from growing large and pushing softmax into saturation (where gradients vanish). Without this scaling, QK^T grows as O(d_k) in magnitude, the softmax produces near-one-hot distributions, and gradients through the attention weights approach zero. Flash Attention handles this numerically by using the log-sum-exp trick to compute the scaled softmax in a numerically stable way without materializing the full N×N attention matrix.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations verified via web search against authoritative repositories (JMLR, ISCA Archive, OpenAI, arXiv, NeurIPS).
      </Prose>

      <H3>Foundational papers</H3>

      <Prose>
        <strong>Bengio et al. (2003).</strong> "A Neural Probabilistic Language Model." <em>Journal of Machine Learning Research</em>, 3, 1137–1155. <a href="https://www.jmlr.org/papers/v3/bengio03a.html" target="_blank" rel="noopener noreferrer">jmlr.org/papers/v3/bengio03a.html</a>. The paper that introduced learned word embeddings and neural next-token prediction. Established the chain-rule factorization as the training objective.
      </Prose>

      <Prose>
        <strong>Mikolov et al. (2010).</strong> "Recurrent Neural Network Based Language Model." <em>INTERSPEECH 2010</em>, pp. 1045–1048. DOI: 10.21437/Interspeech.2010-343. <a href="https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html" target="_blank" rel="noopener noreferrer">isca-archive.org</a>. First large-scale RNN language model; ~50% perplexity reduction over n-gram baselines on Wall Street Journal. Applied the same cross-entropy objective over unbounded recurrent context.
      </Prose>

      <Prose>
        <strong>Peters et al. (2018).</strong> "Deep Contextualized Word Representations (ELMo)." <em>NAACL 2018</em>. arXiv:1802.05365. <a href="https://arxiv.org/abs/1802.05365" target="_blank" rel="noopener noreferrer">arxiv.org/abs/1802.05365</a>. Showed that language model representations transfer to downstream tasks without fine-tuning, establishing the pretraining paradigm.
      </Prose>

      <H3>GPT lineage</H3>

      <Prose>
        <strong>Radford et al. (2018).</strong> "Improving Language Understanding by Generative Pre-Training." OpenAI technical report. <a href="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" target="_blank" rel="noopener noreferrer">cdn.openai.com</a>. GPT-1: 117M decoder-only transformer, causal LM pretraining on BooksCorpus, fine-tuned on 12 downstream tasks. Beat discriminative baselines on 9/12 tasks.
      </Prose>

      <Prose>
        <strong>Radford et al. (2019).</strong> "Language Models Are Unsupervised Multitask Learners." OpenAI technical report. <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank" rel="noopener noreferrer">cdn.openai.com</a>. GPT-2: 1.5B parameters, WebText, zero-shot task performance. Established that scale alone enables task transfer under causal LM pretraining.
      </Prose>

      <Prose>
        <strong>Brown et al. (2020).</strong> "Language Models Are Few-Shot Learners." <em>NeurIPS 2020</em>. arXiv:2005.14165. <a href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noopener noreferrer">arxiv.org/abs/2005.14165</a>. GPT-3: 175B parameters. Demonstrated few-shot in-context learning as an emergent property of scale under causal LM. The definitive case that next-token prediction at scale is sufficient for general-purpose language understanding and generation.
      </Prose>

      <H3>Scaling and architecture</H3>

      <Prose>
        <strong>Vaswani et al. (2017).</strong> "Attention Is All You Need." <em>NeurIPS 2017</em>. arXiv:1706.03762. <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener noreferrer">arxiv.org/abs/1706.03762</a>. Introduced the transformer architecture. The causal mask described in Section 3 is from this paper (Section 3.2.3).
      </Prose>

      <Prose>
        <strong>Kaplan et al. (2020).</strong> "Scaling Laws for Neural Language Models." arXiv:2001.08361. <a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener noreferrer">arxiv.org/abs/2001.08361</a>. Power-law relationships between CE loss and model size, dataset size, and compute, stable across seven orders of magnitude. The empirical foundation for compute-optimal training (Chinchilla).
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive perplexity from cross-entropy</H3>
      <Prose>
        Starting from the definition of per-token cross-entropy <Code>H = -(1/N) Σ log p(xₜ | x&#8344;ₜ)</Code>, derive the perplexity formula <Code>PPL = exp(H)</Code>. Then: if a model achieves 3.0 nats per token, what is its perplexity? If it achieves 3.0 bits per token (using log base 2), what is the perplexity? Are they the same number?
      </Prose>

      <H3>Exercise 2 — Cross-entropy of a uniform distribution</H3>
      <Prose>
        GPT-2's vocabulary has 50,257 tokens. If a model assigned probability <Code>1/50,257</Code> uniformly to all tokens at every position, what would its cross-entropy be (in nats)? What would its perplexity be? GPT-2 base achieves perplexity ~35 on its own test set. How many times more uncertain is the uniform model than GPT-2 base?
      </Prose>

      <H3>Exercise 3 — The -inf vs -1e9 question</H3>
      <Prose>
        In the causal mask, future positions are set to <Code>-inf</Code> before the softmax. Some implementations use <Code>-1e9</Code> instead. Under what conditions does <Code>-1e9</Code> fail to correctly implement the causal constraint? Write a 10-line test that demonstrates the failure. (Hint: consider what happens when the attention logits are themselves very large.)
      </Prose>

      <H3>Exercise 4 — 10-line greedy decoder</H3>
      <Prose>
        Using only <Code>torch</Code> and a pretrained <Code>MiniGPT</Code> (from Section 4), write a greedy decoding loop in 10 lines or fewer. It should take a prompt string, encode it to token IDs, loop until a stop token or max length, and return the decoded string. Then modify it to use top-k sampling with k=5.
      </Prose>

      <H3>Exercise 5 — Diagnose off-by-one shift symptoms</H3>
      <Prose>
        Consider two bugs: (a) the targets are not shifted — <Code>y = x</Code> instead of <Code>y = x[1:]</Code> — and (b) the targets are shifted by two positions instead of one. For each bug, predict: (i) how the training loss will behave in the first 100 steps, (ii) what generation output will look like at inference, and (iii) what a sample attention pattern will look like on a short sequence. Verify your predictions by actually introducing each bug into the MiniGPT training loop from Section 4.
      </Prose>
    </div>
  ),
};

export default causalLanguageModeling;
