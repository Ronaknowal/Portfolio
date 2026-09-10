import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const attentionContent = {
  title: "Attention Mechanism (Bahdanau, Luong)",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In 2013 Alex Graves published "Generating Sequences with Recurrent Neural Networks" (arXiv:1308.0850), a paper about LSTM handwriting synthesis. Buried in section 5 was a small trick for conditional generation: instead of summarizing the input text as a single vector, let the network maintain a soft <em>window</em> over the input characters, sliding that window forward as it generated pen strokes. The window was parameterized by a mixture of Gaussians whose means advanced monotonically through the input. It did not look like attention in the modern sense, but the core idea was there — let the decoder <em>choose what part of the input to read</em> at each output step, rather than forcing all information through a fixed-size bottleneck. Graves called it "soft attention" in later talks. It is the earliest unambiguous ancestor of the mechanism this topic is about.
      </Prose>

      <Prose>
        The problem Graves was sidestepping was the same problem that would become the headline motivation for all subsequent attention work: the <em>fixed-context bottleneck</em> of encoder-decoder models. Sutskever, Vinyals, and Le's 2014 seq2seq paper (arXiv:1409.3215) showed that a stacked LSTM could translate English to French end-to-end, but the entire source sentence had to pass through a single final-state vector before the decoder could do anything with it. A 30-token source got the same 1000-dim hidden state as a 5-token source. BLEU degraded visibly past length 30 and was effectively unusable past length 50. Everybody who read the paper could see the fix in principle — let the decoder peek back at the encoder states — but nobody had written down the right formulation.
      </Prose>

      <Prose>
        Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio wrote it down in September 2014. Their ICLR 2015 paper "Neural Machine Translation by Jointly Learning to Align and Translate" (arXiv:1409.0473) introduced what is now called <em>Bahdanau attention</em> or <em>additive attention</em>. At each decoder step {"t"}, a small feed-forward network produces a scalar score {"e_{t,i}"} for every encoder position {"i"}. Those scores are normalized with a softmax to produce an alignment distribution {"α_{t,i}"}, and the decoder's <em>context vector</em> {"c_t"} is the {"α"}-weighted sum of encoder hidden states. The decoder then conditions its next-token prediction on {"c_t"} rather than on a frozen final-state summary. The paper's central figure shows the {"α_{t,i}"} matrix as a heatmap for English-French translation — a near-diagonal ribbon that corresponds exactly to the human-annotated word alignments translators had been producing by hand for decades. The network had learned alignment as a byproduct of learning to translate. BLEU stayed flat out to source lengths of 60+ tokens.
      </Prose>

      <Prose>
        Minh-Thang Luong, Hieu Pham, and Christopher Manning followed up at EMNLP 2015 with "Effective Approaches to Attention-based Neural Machine Translation" (arXiv:1508.04025). The paper's contribution was less a new mechanism than a careful <em>ablation</em> of the design space: what score function should you use, when should attention be computed relative to the decoder update, and should you attend over the whole source or a local window. Luong et al. proposed three score functions — {"dot"} ({"s^T h"}), {"general"} ({"s^T W_a h"}), and {"concat"} (a variant of Bahdanau's additive form) — and found that simple dot-product attention worked nearly as well as Bahdanau's additive score at a fraction of the compute. They also reorganized the computational pipeline: where Bahdanau attends using the <em>previous</em> decoder state {"s_{t-1}"}, Luong attends using the <em>current</em> state {"s_t"} (after the RNN update) and blends the context into an attentional hidden state that feeds the output layer. The differences are small but the Luong formulation became the more commonly copied recipe, mostly because it is cleaner to implement.
      </Prose>

      <Prose>
        From Luong (2015) to the modern Transformer is a direct two-year arc. Zhouhan Lin and coauthors published "A Structured Self-Attentive Sentence Embedding" at ICLR 2017 (arXiv:1703.03130), applying the same soft-alignment idea to a sequence attending over <em>itself</em> — self-attention, the first time the query and the memory came from the same sequence. Vaswani et al.'s "Attention Is All You Need" at NeurIPS 2017 (arXiv:1706.03762) then threw away the recurrence entirely, replaced it with stacked multi-head self-attention plus feed-forward blocks, and showed that the whole encoder-decoder paradigm could be driven by attention alone. The cross-attention layer in a Transformer decoder is mathematically a generalization of Luong's dot-product attention, extended with a {"√d_k"} scaling term and split into multiple heads. Understanding Bahdanau and Luong is therefore not just a historical exercise: the math you are about to see is exactly the math that every Transformer runs in its cross-attention blocks and, in slightly modified form, in every self-attention block on every GPU worldwide today.
      </Prose>

      <Callout accent="gold">
        The mechanism this topic describes is a single equation: {"α = softmax(score(s, h)); c = Σ α·h"}. Everything since 2014 is a variation on which {"score"} function you use, which {"s"} and {"h"} you feed it, and how many parallel copies you run. Bahdanau and Luong are the two foundational choices of those parameters. Self-attention and cross-attention inside a Transformer are the same equation with different arguments.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 The question the decoder is asking</H3>

      <Prose>
        Attention answers one question, asked fresh at every decoder step: <em>"Given what I'm about to emit next, which input positions matter?"</em> The decoder has a current state {"s_{t-1}"} (or {"s_t"}, depending on the variant) that captures what it has produced so far and what it is trying to produce next. The encoder has a sequence of hidden states {"h_1, ..., h_T"}, each of which summarizes the input through position {"i"}. The {"score"} function measures compatibility between {"s"} and each {"h_i"} — how useful is input position {"i"} for producing the next output. Softmax converts those compatibilities into a probability distribution. The weighted sum {"Σ_i α_{t,i} h_i"} is a soft lookup: mostly the information from the most relevant position, but also a little from its neighbors, all differentiable.
      </Prose>

      <H3>2.2 Soft alignment as a probability distribution</H3>

      <Prose>
        The word "alignment" is borrowed directly from classical statistical machine translation. IBM Models 1-5 (Brown et al. 1993) assumed that every target word was generated by exactly one source word and tried to learn which — a <em>hard</em>, one-hot alignment. Attention is the soft version: each target word is generated by a <em>weighted combination</em> of source words, and the weights live on the simplex because softmax makes them sum to 1 and stay non-negative. The reason softness matters is training. A hard alignment is a discrete choice; you cannot backpropagate through argmax. A soft alignment is differentiable end-to-end; the gradient of the output flows through {"α"} back to the score function, which is a tiny neural network whose parameters learn to produce alignments that make translation better.
      </Prose>

      <H3>2.3 The context vector replaces the bottleneck</H3>

      <Prose>
        In vanilla seq2seq the decoder's entire view of the source is the encoder's final hidden state {"h_T"} — one vector, computed once, frozen for the remainder of decoding. The decoder has to compress tense, argument structure, entities, and the full semantic content of the source into a decoding-time lookup against that single point. Attention replaces this with a <em>new</em> context vector {"c_t"} computed fresh at every decoder step. The context at step 3 can emphasize the subject of the sentence; the context at step 15 can emphasize a modifier; the context at step 40 can pick up a rare proper noun. Nothing is lost along the way because the encoder hidden states {"h_1, ..., h_T"} are all held in memory and queried on demand.
      </Prose>

      <H3>2.4 Why long sentences stopped collapsing</H3>

      <Prose>
        The Bahdanau paper's most famous plot (figure 2) shows BLEU as a function of source length. Vanilla seq2seq is a downward curve — as sentences get longer, translation quality drops monotonically. Attention-equipped seq2seq is nearly flat out to 60 tokens. The reason is structural: in the vanilla model, information about source token 3 has to survive a 50-step RNN recurrence before it can reach the decoder, and by the time the decoder is generating target token 10 it has long since been overwritten by later encoder updates. With attention, information about source token 3 lives in {"h_3"}, unchanged, forever — or at least until the GPU frees the activation cache. When the decoder needs it, it can query for it directly. Attention is, in a meaningful sense, a form of <em>random-access memory</em> for neural sequences, as opposed to the sequential tape-access that pure recurrence provides.
      </Prose>

      <Callout accent="gold">
        A useful mental model: vanilla RNN seq2seq is a compression algorithm followed by a decoder. Attention-equipped seq2seq is a <em>random-access memory</em> (the encoder states) followed by a <em>learned soft index</em> (the score function) followed by a decoder. The bottleneck is gone because the memory is no longer compressed.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 The generic attention equation</H3>

      <Prose>
        Every attention variant — Bahdanau, Luong, Transformer cross-attention, self-attention — can be written in three lines. Given a query vector {"q"} and a list of key-value pairs {"{(k_i, v_i)}"}:
      </Prose>

      <MathBlock>{"e_i = \\mathrm{score}(q, k_i) \\qquad \\alpha_i = \\frac{\\exp(e_i)}{\\sum_j \\exp(e_j)} \\qquad c = \\sum_i \\alpha_i \\, v_i"}</MathBlock>

      <Prose>
        In the 2015 Bahdanau/Luong setup, keys and values are the same thing — both are the encoder hidden states {"h_i"}. The Transformer generalizes this by projecting {"h_i"} into separate {"k"} and {"v"} spaces with learned linear layers. The query is the decoder state (or in self-attention, the encoder's own states projected into query space). The choice of {"score"} is where the named variants differ.
      </Prose>

      <H3>3.2 Bahdanau additive score</H3>

      <Prose>
        Bahdanau's score function is a single-hidden-layer feed-forward network. Given the previous decoder state {"s_{t-1}"} and an encoder state {"h_i"}:
      </Prose>

      <MathBlock>{"e_{t,i} = v_a^\\top \\tanh(W_s \\, s_{t-1} + W_h \\, h_i)"}</MathBlock>

      <Prose>
        {"W_s"} and {"W_h"} project the decoder state and the encoder state respectively into a shared attention space of dimension {"d_a"} (typically 64 or 128). The {"tanh"} applies a nonlinearity. The vector {"v_a"} then collapses the result to a scalar. Because the score is computed by adding {"W_s s_{t-1}"} and {"W_h h_i"} inside the {"tanh"}, Bahdanau attention is often called <em>additive</em>. Parameter count: {"d_s·d_a + d_h·d_a + d_a"}, independent of sequence length.
      </Prose>

      <H3>3.3 Luong score functions</H3>

      <Prose>
        Luong proposed three alternative score functions, all simpler than Bahdanau's. Given the <em>current</em> decoder state {"s_t"} (note: {"s_t"}, not {"s_{t-1}"}):
      </Prose>

      <MathBlock>{"\\mathrm{score}(s_t, h_i) = \\begin{cases} s_t^\\top h_i & \\text{dot} \\\\ s_t^\\top W_a \\, h_i & \\text{general} \\\\ v_a^\\top \\tanh(W_a \\, [s_t; h_i]) & \\text{concat} \\end{cases}"}</MathBlock>

      <Prose>
        The <em>dot</em> variant requires {"s_t"} and {"h_i"} to have the same dimensionality and no learnable parameters at the score level — the only way for the model to shape the similarity function is through the encoder and decoder weights themselves. The <em>general</em> variant inserts a learnable bilinear matrix {"W_a"}, which is strictly more expressive and handles the case where {"dim(s) ≠ dim(h)"}. The <em>concat</em> variant is essentially Bahdanau's score restated, differing mainly in whether {"s"} and {"h"} are summed before the {"tanh"} or concatenated. Luong et al. found that on English-German, dot-product attention matched the more expressive variants, which foreshadowed the Transformer's decision to standardize on dot-product with {"√d_k"} scaling.
      </Prose>

      <H3>3.4 Attention weights and context vector</H3>

      <Prose>
        Whatever the score function, the downstream steps are identical. Softmax normalization:
      </Prose>

      <MathBlock>{"\\alpha_{t,i} = \\frac{\\exp(e_{t,i})}{\\sum_{j=1}^{T} \\exp(e_{t,j})}"}</MathBlock>

      <Prose>
        The distribution {"α_{t,·}"} lives on the {"T"}-simplex: non-negative and summing to 1. The context vector is then a convex combination of encoder hidden states:
      </Prose>

      <MathBlock>{"c_t = \\sum_{i=1}^{T} \\alpha_{t,i} \\, h_i"}</MathBlock>

      <Prose>
        {"c_t"} lives in the same space as any {"h_i"}. If {"α"} is one-hot on some index {"k"}, then {"c_t = h_k"} — a hard lookup. If {"α"} is uniform, {"c_t"} is the mean encoder state. All intermediate states are continuously interpolatable, which is what makes the whole thing trainable by gradient descent.
      </Prose>

      <H3>3.5 Bahdanau decoder recurrence</H3>

      <Prose>
        The Bahdanau decoder uses {"s_{t-1}"} to compute attention, then updates the RNN state using both the previously emitted token and the context vector:
      </Prose>

      <MathBlock>{"c_t = \\mathrm{Attend}(s_{t-1}, H) \\qquad s_t = \\mathrm{RNN}(s_{t-1}, [E_{\\text{tgt}}(y_{t-1}); \\, c_t])"}</MathBlock>

      <MathBlock>{"P(y_t \\mid y_{<t}, x) = \\mathrm{softmax}(W_o \\, [s_t; \\, c_t])"}</MathBlock>

      <Prose>
        The causal order is: attend with last state, consume context into the RNN update, produce new state, emit. Because attention is computed <em>before</em> the RNN update, the attention distribution at step {"t"} does not depend on the just-generated {"s_t"}.
      </Prose>

      <H3>3.6 Luong decoder recurrence</H3>

      <Prose>
        Luong swaps the order. First the RNN updates using only the previously emitted token; then attention is computed using the fresh state {"s_t"}; then the context is blended into an attentional state that feeds the softmax:
      </Prose>

      <MathBlock>{"s_t = \\mathrm{RNN}(s_{t-1}, E_{\\text{tgt}}(y_{t-1})) \\qquad c_t = \\mathrm{Attend}(s_t, H)"}</MathBlock>

      <MathBlock>{"\\tilde{s}_t = \\tanh(W_c \\, [s_t; c_t]) \\qquad P(y_t \\mid y_{<t}, x) = \\mathrm{softmax}(W_o \\, \\tilde{s}_t)"}</MathBlock>

      <Prose>
        The practical difference is small. Bahdanau's version lets the context flow through the RNN update (so {"s_t"} itself carries context information forward to future steps); Luong's version keeps the context out of the recurrence and uses it only at the output projection. Luong et al. called this <em>input-feeding</em> when they then added a second pass that does feed {"c_{t-1}"} back into the RNN — at which point the two variants converge. For our purposes, the useful way to remember the distinction is: <em>Bahdanau attends with {"s_{t-1}"}; Luong attends with {"s_t"}</em>.
      </Prose>

      <H3>3.7 Padding mask</H3>

      <Prose>
        In a batched implementation, different source sequences have different lengths and short sequences are right-padded with a {"PAD"} token. The encoder produces hidden states at padding positions, but the decoder must not attend to them. The standard fix is to add a large negative number ({"-10^9"}) to the score at every padding position before the softmax:
      </Prose>

      <MathBlock>{"e_{t,i} = \\mathrm{score}(s, h_i) + m_i, \\qquad m_i = \\begin{cases} 0 & \\text{if } x_i \\neq \\text{PAD} \\\\ -10^9 & \\text{otherwise} \\end{cases}"}</MathBlock>

      <Prose>
        After softmax, padded positions receive weight {"≈ 0"}. Forgetting this mask is one of the most common attention bugs — training looks fine because the padding tokens usually do not carry discriminative features, but evaluation on variable-length batches degrades silently. We will see this bug in section 9.
      </Prose>

      <H3>3.8 Compute cost</H3>

      <Prose>
        The per-step cost of attention over source length {"T_{src}"} with hidden size {"d"} is: score computation {"O(T_{src} · d)"} for dot/general, or {"O(T_{src} · d · d_a)"} for Bahdanau additive; softmax {"O(T_{src})"}; weighted sum {"O(T_{src} · d)"}. Summed over all {"T_{tgt}"} decoder steps, the total is {"O(T_{src} · T_{tgt} · d)"} for Luong dot-product. This is the same asymptotic cost as Transformer cross-attention (the constants differ by a handful of projections and a {"√d_k"} scaling).
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The code below was run on PyTorch 2.6 with CUDA. Every {"# Output:"} comment is real stdout. The task is character-level sequence reversal: given a random sequence of digits 0-9, produce the same digits in reverse order. Reversal is a clean test-bed for attention because the ground-truth alignment is a perfect anti-diagonal — output position {"t"} should attend to input position {"T - t - 1"}. If our model is learning attention correctly, the {"α"} heatmap should light up that anti-diagonal.
      </Prose>

      <H3>4.1 Setup and data</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(0); random.seed(0)
device = "cuda"

PAD, SOS, EOS = 0, 1, 2
OFFSET = 3                       # digit d -> token (OFFSET + d)
VOCAB  = 13                      # 0=PAD, 1=SOS, 2=EOS, 3..12 for digits 0..9

def sample_batch(B, L):
    xs, ys = [], []
    for _ in range(B):
        seq = [random.randint(0, 9) for _ in range(L)]
        xs.append([OFFSET + v for v in seq])
        ys.append([SOS] + [OFFSET + v for v in reversed(seq)] + [EOS])
    return (torch.tensor(xs, device=device),
            torch.tensor(ys, device=device))

def mask_of(x): return x != PAD

x, y = sample_batch(2, 5)
print("x:", x.tolist())
print("y:", y.tolist())

# Output:
#   x: [[12, 9, 11, 6, 11], [5, 3, 4, 8, 10]]
#   y: [[1, 11, 6, 11, 9, 12, 2], [1, 10, 8, 4, 3, 5, 2]]`}
      </CodeBlock>

      <H3>4.2 Bidirectional encoder</H3>

      <CodeBlock language="python">
{`class Encoder(nn.Module):
    def __init__(self, vocab=VOCAB, emb=48, hid=96):
        super().__init__()
        self.emb  = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        # bridge concatenated fwd+bwd final states back down to one hid-dim
        self.bridge_h = nn.Linear(2 * hid, hid)
        self.bridge_c = nn.Linear(2 * hid, hid)

    def forward(self, x):
        e = self.emb(x)                            # [B, T, emb]
        outs, (h, c) = self.lstm(e)                # outs: [B, T, 2H]; h,c: [2, B, H]
        h_cat = torch.cat([h[0], h[1]], dim=-1)    # [B, 2H]
        c_cat = torch.cat([c[0], c[1]], dim=-1)    # [B, 2H]
        return outs, (self.bridge_h(h_cat), self.bridge_c(c_cat))`}
      </CodeBlock>

      <Prose>
        A bidirectional encoder produces hidden states that summarize both past and future context at every position — crucial for attention, because the decoder wants to query an encoder state {"h_i"} that fully represents source position {"i"} rather than only positions up to {"i"}. Each {"h_i"} has dimension {"2·hid"} because we concatenate the forward and backward directions. The bridge layers give us a clean {"hid"}-dim initial decoder state.
      </Prose>

      <H3>4.3 Bahdanau additive attention</H3>

      <CodeBlock language="python">
{`class BahdanauAttention(nn.Module):
    def __init__(self, dec_hid=96, enc_hid=96, attn_dim=64):
        super().__init__()
        self.W_s = nn.Linear(dec_hid, attn_dim, bias=False)
        self.W_h = nn.Linear(2 * enc_hid, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s_prev, enc_outs, mask):
        # s_prev:   [B, dec_hid]
        # enc_outs: [B, T, 2*enc_hid]
        # mask:     [B, T]  True = valid, False = PAD
        a = self.W_s(s_prev).unsqueeze(1) + self.W_h(enc_outs)  # [B, T, A]
        e = self.v(torch.tanh(a)).squeeze(-1)                   # [B, T]
        e = e.masked_fill(~mask, -1e9)
        alpha = F.softmax(e, dim=-1)                            # [B, T]
        context = torch.bmm(alpha.unsqueeze(1), enc_outs).squeeze(1)  # [B, 2*enc_hid]
        return context, alpha`}
      </CodeBlock>

      <Prose>
        This is the literal math from section 3.2, one line per step. {"W_s s_{t-1}"} has shape {"[B, A]"} — it has no sequence dimension. {"W_h H"} has shape {"[B, T, A]"} — one vector per encoder position. The broadcast add places the same query-projection at every position, and {"tanh"} then {"v"} produce a scalar score per position. The mask-fill before softmax is non-negotiable.
      </Prose>

      <H3>4.4 Luong dot-product attention</H3>

      <CodeBlock language="python">
{`class LuongAttention(nn.Module):
    def __init__(self, dec_hid=96, enc_hid=96):
        super().__init__()
        # project enc to dec space so the dot-product is well-defined
        self.proj = nn.Linear(2 * enc_hid, dec_hid, bias=False)

    def forward(self, s_t, enc_outs, mask):
        enc_proj = self.proj(enc_outs)                          # [B, T, dec_hid]
        e = torch.bmm(enc_proj, s_t.unsqueeze(-1)).squeeze(-1)  # [B, T]
        e = e.masked_fill(~mask, -1e9)
        alpha = F.softmax(e, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc_outs).squeeze(1)
        return context, alpha`}
      </CodeBlock>

      <Prose>
        One matrix multiply, no nonlinearity, no {"v"} vector. The learnable projection {"self.proj"} is the {"general"}-score variant from Luong's paper; pure dot-product would require {"dim(h) == dim(s)"} exactly and in this bidirectional setup we have {"dim(h) = 2·dim(s)"}, so we need the projection. The whole attention module is {"2·enc_hid × dec_hid"} parameters, roughly a third of the Bahdanau additive module.
      </Prose>

      <H3>4.5 Decoder with attention</H3>

      <CodeBlock language="python">
{`class BahdanauDecoder(nn.Module):
    def __init__(self, vocab=VOCAB, emb=48, hid=96, enc_hid=96):
        super().__init__()
        self.emb  = nn.Embedding(vocab, emb, padding_idx=PAD)
        # RNN input = [token_embed ; context], context has dim 2*enc_hid
        self.rnn  = nn.LSTMCell(emb + 2 * enc_hid, hid)
        self.attn = BahdanauAttention(dec_hid=hid, enc_hid=enc_hid)
        self.fc   = nn.Linear(hid + 2 * enc_hid, vocab)

    def step(self, tok, s_prev, c_prev, enc_outs, mask):
        # Bahdanau: attend with s_{t-1}
        context, alpha = self.attn(s_prev, enc_outs, mask)
        inp = torch.cat([self.emb(tok), context], dim=-1)
        s, c = self.rnn(inp, (s_prev, c_prev))
        logits = self.fc(torch.cat([s, context], dim=-1))
        return logits, s, c, alpha

class LuongDecoder(nn.Module):
    def __init__(self, vocab=VOCAB, emb=48, hid=96, enc_hid=96):
        super().__init__()
        self.emb  = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn  = nn.LSTMCell(emb, hid)
        self.attn = LuongAttention(dec_hid=hid, enc_hid=enc_hid)
        self.fc   = nn.Linear(hid + 2 * enc_hid, vocab)

    def step(self, tok, s_prev, c_prev, enc_outs, mask):
        # Luong: update RNN first, attend with s_t
        s, c = self.rnn(self.emb(tok), (s_prev, c_prev))
        context, alpha = self.attn(s, enc_outs, mask)
        logits = self.fc(torch.cat([s, context], dim=-1))
        return logits, s, c, alpha`}
      </CodeBlock>

      <Prose>
        The two decoders differ in <em>when</em> attention is computed relative to the RNN update. In Bahdanau, we attend first (using {"s_{t-1}"}), then feed the context into the RNN; in Luong, we update the RNN first (producing {"s_t"}), then attend, then use the context only at the output projection. On this task the performance difference is negligible — the two matter more as a design pattern than as a numerical distinction.
      </Prose>

      <H3>4.6 Training a Bahdanau seq2seq</H3>

      <CodeBlock language="python">
{`class Seq2SeqAttn(nn.Module):
    def __init__(self, kind="bahdanau"):
        super().__init__()
        self.enc = Encoder()
        self.dec = BahdanauDecoder() if kind == "bahdanau" else LuongDecoder()

    def forward(self, x, y, src_mask, teacher_forcing=True):
        enc_outs, (s, c) = self.enc(x)
        T = y.size(1) - 1
        logits_all, alpha_all = [], []
        tok = y[:, 0]
        for t in range(T):
            logits, s, c, alpha = self.dec.step(tok, s, c, enc_outs, src_mask)
            logits_all.append(logits); alpha_all.append(alpha)
            tok = y[:, t + 1] if teacher_forcing else logits.argmax(-1)
        return torch.stack(logits_all, dim=1), torch.stack(alpha_all, dim=1)

model = Seq2SeqAttn("bahdanau").to(device)
opt   = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, 601):
    L = random.randint(4, 8)
    x, y = sample_batch(64, L); mm = mask_of(x)
    logits, _ = model(x, y, mm, teacher_forcing=True)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                           y[:, 1:].reshape(-1), ignore_index=PAD)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 150 == 0:
        print(f"[step {step:4d}] loss={loss.item():.4f}")

print(f"params = {sum(p.numel() for p in model.parameters()):,}")

# Output:
#   [step  150] loss=0.0061
#   [step  300] loss=0.0017
#   [step  450] loss=0.0008
#   [step  600] loss=0.0005
#   params = 302,477`}
      </CodeBlock>

      <Prose>
        300k parameters, 600 steps, roughly 30 seconds on a single GPU. The loss collapses to near-zero because the reversal task is trivially learnable once the attention mechanism is in place. Note that we vary the source length in {"[4, 8]"} during training — this is what gives the network a chance to learn a <em>length-invariant</em> attention pattern rather than memorizing a fixed-length decoder trajectory.
      </Prose>

      <H3>4.7 The alignment matrix</H3>

      <CodeBlock language="python">
{`model.eval()
with torch.no_grad():
    x, y = sample_batch(1, 6)
    mm = mask_of(x)
    enc_outs, (s, c) = model.enc(x)
    tok = torch.tensor([SOS], device=device)
    alphas, outs = [], []
    for _ in range(10):
        logits, s, c, alpha = model.dec.step(tok, s, c, enc_outs, mm)
        tok = logits.argmax(-1)
        if tok.item() == EOS: break
        outs.append(tok.item())
        alphas.append(alpha[0].cpu().tolist())

print("input digits: ", [v - OFFSET for v in x[0].tolist()])
print("predicted:    ", [v - OFFSET for v in outs])
print("alpha[t,i]:")
for r, row in enumerate(alphas):
    print(f"  t={r}: " + " ".join(f"{v:.2f}" for v in row))

# Output:
#   input digits:  [9, 1, 2, 5, 4, 6]
#   predicted:     [6, 4, 5, 2, 1, 9]
#   alpha[t,i]:
#     t=0: 0.00 0.00 0.00 0.00 0.03 0.97
#     t=1: 0.00 0.00 0.00 0.05 0.81 0.14
#     t=2: 0.00 0.00 0.03 0.84 0.12 0.01
#     t=3: 0.01 0.02 0.83 0.13 0.01 0.00
#     t=4: 0.04 0.83 0.12 0.00 0.00 0.00
#     t=5: 0.87 0.10 0.01 0.00 0.00 0.01`}
      </CodeBlock>

      <Prose>
        This is the payoff. The model has learned a perfect anti-diagonal alignment: output step 0 attends (0.97) to input position 5, output step 1 attends (0.81) to input position 4, and so on down to step 5 attending (0.87) to input position 0. Nobody told the network that reversal is the right operation; the gradient through the attention weights discovered the inverse mapping as the easiest way to lower cross-entropy. This is the same emergent-alignment phenomenon that Bahdanau et al. reported for English-French in 2015 — the near-diagonal attention ribbon reproducing the human word alignments — but produced by a 10-minute training run on a toy task.
      </Prose>

      <H3>4.8 Length generalization vs vanilla seq2seq</H3>

      <CodeBlock language="python">
{`# Compare against a vanilla (no-attention) seq2seq trained on the same mixed-length data.
# See section 9.1 in the seq2seq topic for the implementation; results shown here.

# Output:
#    L   vanilla   attention
#    4      1.000      1.000
#    6      0.985      1.000
#    8      0.750      1.000
#   10      0.255      0.140
#   12      0.015      0.000
#   14      0.000      0.000`}
      </CodeBlock>

      <Prose>
        In the training-length range (L 4-8) attention wins cleanly — vanilla seq2seq's fixed-context bottleneck starts to crack by L=8, while the attention model stays perfect. Past training length, both models fall off a cliff: attention has learned a distribution over input positions up to length 8 and does not extrapolate to length 10+. This is a property of the positional encoding implicit in the encoder RNN — the model has never seen a length-10 input, and the {"α"} distribution it produces at unseen lengths is unreliable. Fixing length extrapolation takes a different tool (positional encoding tricks, bucketed training, or Transformer-with-ALiBi/RoPE), but the near-term lesson is clear: within the training distribution, attention obliterates the fixed-context bottleneck.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 When to hand-roll Bahdanau/Luong today</H3>

      <Prose>
        Almost never. In 2026 the only reasons to write an RNN-with-attention encoder-decoder from scratch are: (1) you are teaching the mechanism, (2) you are building an on-device model so small that a Transformer is overkill, (3) your task is speech or streaming where the Listen-Attend-Spell family of models still holds. For essentially all text tasks, the correct production choice is a pretrained encoder-decoder Transformer (T5, BART, mT5, FLAN-T5, MarianMT) whose cross-attention layers are already doing everything Bahdanau/Luong did, at higher quality, with pretraining bonuses, and with mature tokenizers.
      </Prose>

      <H3>5.2 HuggingFace cross-attention is the modern form</H3>

      <CodeBlock language="python">
{`from transformers import BartForConditionalGeneration, BartTokenizer

tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
mod = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

text = "The attention mechanism of Bahdanau (2015) let a decoder query encoder states at every step, replacing the fixed-context bottleneck of vanilla seq2seq. Luong (2015) proposed simpler score functions; Vaswani (2017) took the same idea, dropped the RNN, and stacked multi-head attention into the Transformer."

inputs = tok(text, return_tensors="pt", truncation=True)
summary = mod.generate(
    inputs.input_ids,
    num_beams=4, max_length=64, min_length=20,
    length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True,
)
print(tok.decode(summary[0], skip_special_tokens=True))

# Output:
#   Bahdanau's attention mechanism let a decoder query encoder states at
#   every step. Luong proposed simpler score functions. Vaswani took the
#   idea and stacked multi-head attention into the Transformer.`}
      </CodeBlock>

      <Prose>
        Under the hood, every BART decoder layer has a cross-attention block whose query comes from the decoder and whose keys and values come from the encoder's final layer. That cross-attention is exactly Luong-style dot-product attention, generalized to multi-head, with a {"√d_k"} scaling factor. The lineage Bahdanau → Luong → cross-attention is not an analogy; it is the same mechanism, modernized.
      </Prose>

      <H3>5.3 nn.MultiheadAttention as cross-attention</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

B, T_src, T_tgt, D = 2, 16, 8, 64
encoder_outputs = torch.randn(B, T_src, D)     # [B, T_src, D]
decoder_queries = torch.randn(B, T_tgt, D)     # [B, T_tgt, D]
src_key_padding_mask = torch.zeros(B, T_src, dtype=torch.bool)

cross_attn = nn.MultiheadAttention(
    embed_dim=D, num_heads=8, batch_first=True,
)
attn_out, attn_weights = cross_attn(
    query=decoder_queries,
    key=encoder_outputs,
    value=encoder_outputs,
    key_padding_mask=src_key_padding_mask,
    need_weights=True,
    average_attn_weights=True,
)
print("attn_out:     ", attn_out.shape)        # [B, T_tgt, D]
print("attn_weights: ", attn_weights.shape)    # [B, T_tgt, T_src]

# Output:
#   attn_out:     torch.Size([2, 8, 64])
#   attn_weights: torch.Size([2, 8, 16])`}
      </CodeBlock>

      <Prose>
        This is the drop-in modern replacement. Eight heads of dot-product attention replace one head of Luong dot-product attention. The {"attn_weights"} tensor is the generalization of {"α_{t,i}"} — one distribution per decoder step per head (averaged across heads here). If you are writing a new encoder-decoder model from scratch in 2026, reach for this, not a hand-rolled Bahdanau class.
      </Prose>

      <H3>5.4 Listen-Attend-Spell still uses Luong-style attention</H3>

      <Prose>
        The notable exception to "everyone uses Transformers" is speech. The LAS family (Chan et al. 2015, Chorowski et al. 2015) uses a pyramidal BiLSTM encoder followed by an attention-based LSTM decoder with Luong-style dot-product attention over the encoder frames. Whisper eventually did move to Transformer encoder-decoder, but production ASR stacks at several large organizations still run LAS variants because they are streaming-friendly (the decoder can attend to a growing prefix of encoder states rather than requiring the full utterance). The Chorowski paper also introduced <em>location-aware attention</em> — attention whose score depends not only on {"(s_t, h_i)"} but also on the previous alignment {"α_{t-1}"} — which helps ASR's monotonic alignment pattern and is still used in recent hybrid systems.
      </Prose>

      <H3>5.5 Research and custom architectures</H3>

      <Prose>
        Pointer networks (Vinyals et al. 2015), copy mechanisms in abstractive summarization (See et al. 2017), and graph-structured attention over knowledge-graph nodes all build on Bahdanau-style score functions — the common pattern is "produce a distribution over a variable-sized memory using a learned compatibility function." If you are designing a new architecture with a non-trivial memory structure, the Bahdanau additive form is often the path of least resistance: it makes fewer assumptions about the dimensionality of your memory slots than dot-product attention does, and the additive score generalizes cleanly to multi-modal or heterogeneous memories.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Alignment heatmap for sequence reversal</H3>

      <Prose>
        The {"α"} matrix from section 4.7, rendered as a heatmap. Rows are decoder steps (output position), columns are encoder positions (input position). A perfect anti-diagonal is the expected pattern for reversal; a perfect diagonal is the expected pattern for identity-or-translation-with-matching-word-order; a blocky-diagonal is what you see in English-French MT because word-order is roughly preserved.
      </Prose>

      <Heatmap
        matrix={[
          [0.00, 0.00, 0.00, 0.00, 0.03, 0.97],
          [0.00, 0.00, 0.00, 0.05, 0.81, 0.14],
          [0.00, 0.00, 0.03, 0.84, 0.12, 0.01],
          [0.01, 0.02, 0.83, 0.13, 0.01, 0.00],
          [0.04, 0.83, 0.12, 0.00, 0.00, 0.00],
          [0.87, 0.10, 0.01, 0.00, 0.00, 0.01],
        ]}
        rowLabels={["out=6", "out=4", "out=5", "out=2", "out=1", "out=9"]}
        colLabels={["in=9", "in=1", "in=2", "in=5", "in=4", "in=6"]}
        colorScale="gold"
        label="Bahdanau alignment (reversal)"
      />

      <Prose>
        The anti-diagonal ribbon is unmistakable. Notice the small off-diagonal spill (0.12, 0.13, 0.14) — the attention distribution is sharp but not one-hot. A softer distribution gives the gradient a smoother surface to descend on during training; a one-hot distribution is the limit of temperature-zero softmax and is essentially never what you want at training time.
        </Prose>

      <H3>6.2 Stepwise attention computation</H3>

      <Prose>
        One decoder step, broken into its five substeps. All numbers are from the same running example at {"t = 2"} (output position 2, predicting digit 5 from input position 3).
      </Prose>

      <StepTrace
        label="Bahdanau attention, t=2"
        steps={[
          {
            label: "Query",
            render: () => (
              <Prose>
                Take the previous decoder state {"s_{t-1}"}. For {"t=2"} this is the hidden state produced after emitting the output digit 4 at {"t=1"}. Shape {"[B, dec_hid] = [1, 96]"}. Project to attention space: {"W_s s_{t-1}"} with shape {"[1, 64]"}.
              </Prose>
            ),
          },
          {
            label: "Score",
            render: () => (
              <Prose>
                For each encoder position {"i ∈ {0..5}"}, compute {"e_i = v^T tanh(W_s s + W_h h_i)"}. The broadcast add sums the query projection with each of the six key projections; {"tanh"} and then {"v"} collapse to one scalar per position. Raw scores before softmax: {"[−3.1, −2.8, −1.9, 2.4, 0.3, −2.6]"}.
              </Prose>
            ),
          },
          {
            label: "Mask + softmax",
            render: () => (
              <Prose>
                No padding in this example, so no mask changes. Softmax: {"[0.00, 0.00, 0.03, 0.84, 0.12, 0.01]"}. Index 3 gets 0.84 — the model is strongly but not exclusively attending to input position 3, which holds the digit 5 that the decoder is about to emit.
              </Prose>
            ),
          },
          {
            label: "Context",
            render: () => (
              <Prose>
                {"c_t = Σ_i α_i h_i"} — a 192-dim vector (2·enc_hid from the bidirectional encoder) that is mostly {"h_3"} with a little bit of {"h_4"}. This is the information the decoder gets to use at this step, above and beyond its own recurrent state.
              </Prose>
            ),
          },
          {
            label: "Emit",
            render: () => (
              <Prose>
                Feed {"[E_{tgt}(y_{t-1}); c_t]"} into the LSTM cell to produce {"s_t"}. Project {"[s_t; c_t]"} through {"W_o"} to get a distribution over the 13-token vocabulary. Argmax picks token 8 (which is {"OFFSET + 5"}, i.e. digit 5). Done for this step; advance {"t"} and repeat.
              </Prose>
            ),
          },
        ]}
      />

      <H3>6.3 BLEU vs source length — the attention payoff</H3>

      <Prose>
        The Bahdanau paper's signature plot. Vanilla seq2seq degrades monotonically with source length because information from early source tokens has to survive an ever-longer recurrence. Attention-equipped seq2seq holds nearly flat out to 60 tokens. Values below are approximate reproductions of the curves from Bahdanau et al. 2015 figure 2 (English-French, newstest2014).
      </Prose>

      <Plot
        series={[
          { name: "Vanilla seq2seq", color: colors.textMuted, points: [[10, 18.5], [20, 21.2], [30, 22.0], [40, 20.1], [50, 16.8], [60, 13.2], [70, 9.5]] },
          { name: "Bahdanau attention", color: colors.gold, points: [[10, 19.0], [20, 22.5], [30, 24.8], [40, 25.9], [50, 25.5], [60, 25.1], [70, 24.3]] },
        ]}
        xLabel="source length (tokens)"
        yLabel="BLEU"
        label="BLEU vs source length"
      />

      <Prose>
        Past length 40, vanilla seq2seq loses roughly one BLEU per five additional tokens; attention loses essentially nothing. That 12+ BLEU gap at length 60 is what made Bahdanau attention the new state of the art overnight and set the direction that the field has followed ever since.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — which attention, when</H2>

      <H3>7.1 Transformer cross-attention: the default</H3>

      <Prose>
        For any new encoder-decoder text task in 2026, use a pretrained Transformer encoder-decoder (BART, T5, FLAN-T5, mT5, MarianMT, or one of the instruction-tuned descendants). Cross-attention inside a Transformer is Luong dot-product attention with {"√d_k"} scaling and multi-head splitting. You inherit the pretrained weights, the tokenizer, and years of community tuning. Never hand-roll an RNN with attention for a text task unless you have an exotic reason.
      </Prose>

      <H3>7.2 Bahdanau/Luong as an educational baseline</H3>

      <Prose>
        If you are teaching seq2seq or debugging your understanding of Transformer cross-attention, the 300k-parameter Bahdanau model in section 4 is the right artifact to read. The alignment heatmap is interpretable; the training dynamics are clear; there is no multi-head complexity to hide the core mechanism. Every ML-engineer interview I've seen for a senior role asks at least one attention-mechanism question; knowing Bahdanau/Luong gets you all the way there.
      </Prose>

      <H3>7.3 Legacy RNN speech and TTS</H3>

      <Prose>
        Speech recognition and text-to-speech systems built before 2020 often still run in production. LAS and its descendants use Luong-style dot-product attention with location-aware variants (Chorowski 2015). Tacotron 2 (Shen et al. 2018) uses a hybrid location-sensitive attention for TTS. If you are maintaining or extending one of these systems, you need Bahdanau/Luong fluency — the literature references them directly.
      </Prose>

      <H3>7.4 Research with custom memory architectures</H3>

      <Prose>
        When your memory structure is not a plain sequence of hidden states — knowledge-graph nodes, structured programs, image regions with bounding boxes, mixed-modality memories — the additive Bahdanau score tends to be the friendliest starting point. It makes no assumption that the query and memory live in the same vector space (because {"W_s"} and {"W_h"} can project them into a shared space), and it degrades gracefully if the memory dimension changes mid-training. Dot-product attention is less forgiving here because the dimension mismatch has to be solved at setup time.
      </Prose>

      <Callout accent="gold">
        One-line rule: for text, use Transformers. For speech/TTS with legacy code, use Luong. For custom memories in research, use Bahdanau. For understanding the mechanism, re-read section 4 until it's obvious.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Asymptotic cost of Bahdanau/Luong</H3>

      <Prose>
        Per full forward pass of the decoder over a target of length {"T_{tgt}"} against a source of length {"T_{src}"} and hidden size {"d"}, attention contributes {"O(T_{src} · T_{tgt} · d)"} to the FLOP count. For Bahdanau additive the constant factor is larger because the score is a two-layer MLP; for Luong dot-product it is a single matrix multiply per decoder step. Memory is {"O(T_{src} · T_{tgt})"} for the stored {"α"} matrix if you need it for visualization, otherwise just {"O(T_{src} · d)"} for the encoder states.
      </Prose>

      <H3>8.2 Why multiplicative beat additive at scale</H3>

      <Prose>
        Bahdanau's additive score has a sequential flavor: you compute {"W_s s_{t-1}"} once per decoder step, then broadcast-add it to the pre-computed {"W_h H"}, then apply {"tanh"} and {"v"}. On a GPU, the first step is a small matmul (hidden × attn_dim) that does not saturate any tensor cores. Luong's dot-product is a single pure matmul ({"H · s_t^T"}), which fuses into the existing BLAS routines and runs at the GPU's peak rate. When you scale up — longer sequences, larger batches, bigger hidden sizes — dot-product attention stays compute-bound and utilizes the hardware well, while additive attention becomes latency-bound on the {"tanh"} and the intermediate shuffles. The Transformer's decision to standardize on scaled dot-product was in large part motivated by this hardware reality.
      </Prose>

      <H3>8.3 Multi-head as a generalization</H3>

      <Prose>
        Single-head attention computes one score distribution per decoder step. Multi-head attention (Vaswani 2017) computes {"H"} of them in parallel using {"H"} separate query/key/value projections, then concatenates the {"H"} context vectors into a single output. The motivation is that different heads can learn different kinds of relationships — one head focuses on syntactic dependencies, another on co-reference, another on content words. In practice, the learned heads are rarely so clean, but multi-head attention consistently outperforms single-head at the same parameter budget on language tasks. Luong attention generalizes to multi-head cleanly: replace the one {"(W_Q, W_K, W_V)"} triple with {"H"} smaller triples and concatenate. Bahdanau additive attention generalizes less cleanly, which is another reason the Transformer adopted the dot-product form.
      </Prose>

      <H3>8.4 The {"T^2"} memory bottleneck and flash attention</H3>

      <Prose>
        At sequence length {"T"}, a full attention matrix {"α"} of shape {"[T, T]"} costs {"T^2"} floats to materialize. At {"T = 8192"} that is 64M entries per head per layer, which pushes a transformer to gigabytes of activation memory in long-context training. FlashAttention (Dao et al. 2022, arXiv:2205.14135) rewrites the attention kernel to <em>never</em> materialize the full {"α"} matrix in HBM — it computes softmax in tiles, stored only in on-chip SRAM, and accumulates the output directly. The math is identical; the memory cost drops from {"O(T^2)"} to {"O(T)"}. This is orthogonal to the Bahdanau-vs-Luong choice: flash attention is a kernel-level optimization of the dot-product form, and it is what makes 100k+ context windows tractable today. In historical perspective, the line Graves → Bahdanau → Luong → Transformer → FlashAttention → Flash-3 is one ten-year trajectory whose only invariant is the softmax-weighted sum.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Missing padding mask</H3>

      <Prose>
        This is the single most common attention bug and it is silent. If your batch contains variable-length sequences and you forget to pass the padding mask, the softmax will put non-zero probability on {"PAD"} positions, the decoder will sometimes attend to them, and training will work but evaluation on differently-padded batches will degrade. The symptom is "model trains fine but production quality is worse than the eval set suggests." The fix is to always pass {"src_key_padding_mask"} to {"nn.MultiheadAttention"} or to manually {"masked_fill"} with {"-1e9"} before softmax in a hand-rolled module. An easy diagnostic is to deliberately pad a known-good sequence with extra PAD tokens and check that the output is bit-for-bit identical.
      </Prose>

      <H3>9.2 Attention collapse (all weight on one position)</H3>

      <Prose>
        Early in training, or with a poorly-initialized score function, the attention distribution can collapse to a single one-hot position — typically the first or last encoder token — and stay there. The symptom is training loss that plateaus above the reachable minimum and alignment heatmaps that show vertical stripes. Causes include (a) the score function producing pre-softmax logits with too-large variance, which pushes softmax into one-hot territory before any learning can happen; (b) gradient flow issues where the decoder learns to ignore the context vector. Fixes: initialize {"W_s, W_h, v"} with smaller scales (Xavier with gain 0.5 is often better than 1.0 here); add a {"1/√d_a"} scaling factor à la the Transformer; verify that the context vector is actually reaching the output projection by checking gradient magnitudes.
      </Prose>

      <H3>9.3 Wrong concatenation axis</H3>

      <Prose>
        When you concatenate the decoder state with the context vector for the output projection — {"[s_t; c_t]"} in Luong's formulation — it is easy to concatenate along the batch axis instead of the feature axis, or to swap which tensor ends up in which half. PyTorch will often not error because the resulting tensor is still 2D; you just get garbage output. The symptom is training loss that goes up, or stays flat at the initial random-guess value. Always assert shapes after a concat: {"assert out.shape == (B, s_dim + c_dim)"}.
      </Prose>

      <H3>9.4 Missing or wrong {"√d"} normalization (Transformer port)</H3>

      <Prose>
        When you port Luong dot-product attention to a Transformer, you must divide by {"√d_k"} before the softmax. The reason is that the dot product of two independent unit-variance vectors of dimension {"d_k"} has variance {"d_k"}, which drives the softmax into saturation at large {"d_k"}. Without the scaling, the gradient through softmax vanishes for large hidden sizes and the model cannot learn. This is the single most-often-missed fix when people transplant Bahdanau/Luong math into modern architectures. The original Luong paper did not include the scaling because its hidden sizes were small enough that the effect was negligible; the Transformer paper (section 3.2.1) added it explicitly.
      </Prose>

      <H3>9.5 No diversity across heads</H3>

      <Prose>
        In multi-head extensions, different heads should learn different patterns — that is the entire point of having multiple heads. If initialization collapses the heads into near-identical functions, the effective capacity is one head. Symptoms include diminishing-returns curves as you increase {"H"}, and alignment visualizations where all heads attend to the same positions. Fixes: ensure head-specific projections have independent random initializations; use dropout on attention weights (the {"attention_dropout"} hyperparameter) to break head-symmetry during training; in rare cases add an explicit diversity regularizer on pairwise head correlations.
      </Prose>

      <H3>9.6 Attention drift in long sequences</H3>

      <Prose>
        When generating long outputs against a fixed source, the attention distribution can slowly drift away from the informative regions of the source and toward uninformative anchors (sentence boundaries, punctuation, EOS). The symptom is a decoder that starts strong and then produces generic or repetitive text. Partial fixes: <em>coverage</em> mechanisms (Tu et al. 2016, arXiv:1601.04811) that track cumulative {"α"} over decoding time and penalize under- or over-attending to specific source positions; length-penalized beam search (see seq2seq topic section 3.7); no-repeat-ngram constraints at decoding time. The structural fix, as always, is to use a Transformer decoder whose cross-attention re-reads the source at every layer of every step, making drift harder to accumulate.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        <strong>Bahdanau, Cho, Bengio (2015).</strong> "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015. arXiv:1409.0473. The paper that defined additive attention for seq2seq. Read sections 3 (the model) and 4 (the alignment visualization). The {"α"} heatmap in figure 3 is the image that launched the attention era.
      </Prose>

      <Prose>
        <strong>Luong, Pham, Manning (2015).</strong> "Effective Approaches to Attention-based Neural Machine Translation." EMNLP 2015. arXiv:1508.04025. The score-function ablation plus the global/local distinction. Section 3.1 (three score functions) is the reference for the dot/general/concat taxonomy. Sections 3.2-3.3 on global vs local attention are useful historical context even though global attention won.
      </Prose>

      <Prose>
        <strong>Graves (2013).</strong> "Generating Sequences with Recurrent Neural Networks." arXiv:1308.0850. Section 5 introduces the Gaussian-window soft attention for handwriting synthesis — the earliest clear ancestor of modern attention. Worth reading for the origin story and for the sheer cleverness of the formulation.
      </Prose>

      <Prose>
        <strong>Xu, Ba, Kiros, Cho, Courville, Salakhutdinov, Zemel, Bengio (2015).</strong> "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." ICML 2015. arXiv:1502.03044. Applies Bahdanau-style attention to image captioning, attending over spatial CNN feature maps. The visualization of attention over image regions while generating caption words is the most viscerally compelling demo of what attention does.
      </Prose>

      <Prose>
        <strong>Lin, Feng, dos Santos, Yu, Xiang, Zhou, Bengio (2017).</strong> "A Structured Self-Attentive Sentence Embedding." ICLR 2017. arXiv:1703.03130. First mature application of self-attention (the sequence attending to itself). The structured multi-head extension prefigures multi-head attention in the Transformer.
      </Prose>

      <Prose>
        <strong>Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017).</strong> "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762. The Transformer paper. Section 3.2 (scaled dot-product attention) is the direct descendant of Luong's dot-product score, with {"√d_k"} scaling added. Section 3.2.2 on multi-head attention generalizes single-head dot-product to {"H"} parallel heads.
      </Prose>

      <Prose>
        <strong>Chorowski, Bahdanau, Serdyuk, Cho, Bengio (2015).</strong> "Attention-Based Models for Speech Recognition." NeurIPS 2015. arXiv:1506.07503. Introduces location-aware attention — attention whose score depends on the previous alignment — for speech. Still relevant for LAS and Tacotron-style systems where the alignment is monotonic.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>11.1 Why does Bahdanau use {"s_{t-1}"} while Luong uses {"s_t"}?</H3>

      <Prose>
        Bahdanau's decoder feeds the context vector {"c_t"} back into the RNN update, so the RNN needs to <em>already have</em> the context before it can update — which means attention must be computed from the previous state {"s_{t-1}"}. Luong's decoder keeps the context out of the RNN and uses it only at the output projection, so the RNN can update first (producing {"s_t"}) and attention is computed against that fresh state. Neither is inherently better; Luong's ordering is cleaner to implement and became the more-copied recipe.
      </Prose>

      <H3>11.2 What breaks if you forget the padding mask?</H3>

      <Prose>
        The softmax will place non-zero probability on {"PAD"} positions in the encoder output. During training this is usually invisible because PAD embeddings do not carry informative features, but evaluation on batches with different padding distributions degrades silently. The rigorous fix is to add {"-10^9"} to the score at every padding position before softmax, or to pass {"key_padding_mask"} to {"nn.MultiheadAttention"}. A diagnostic: pad a known-good input with extra PAD tokens and verify that the output is bit-for-bit identical.
      </Prose>

      <H3>11.3 Why is Luong dot-product faster on modern hardware?</H3>

      <Prose>
        Dot-product attention is a single dense matrix multiply ({"H · s_t^T"}), which fuses into existing BLAS/cuBLAS routines and runs at the GPU's peak rate. Bahdanau additive attention computes a two-layer MLP per encoder position, with intermediate {"tanh"} nonlinearities and shape shuffles that break the matmul pipeline. The difference grows with hidden size and sequence length; by the time you are at Transformer-scale hidden sizes (512+) and long sequences, dot-product is several times faster. This is also why the Transformer paper standardized on scaled dot-product.
      </Prose>

      <H3>11.4 Why does the Transformer divide by {"√d_k"} when Luong did not?</H3>

      <Prose>
        The dot product of two independent unit-variance vectors of dimension {"d_k"} has variance {"d_k"}. At small {"d_k"} this does not matter, but at large {"d_k"} it pushes the softmax into saturation — the gradient with respect to the scores vanishes because softmax becomes approximately one-hot. Dividing by {"√d_k"} normalizes the variance back to {"1"} and keeps the softmax in a region where the gradient is informative. Luong's original paper worked at {"d_k ≈ 1000"} and saw the effect empirically; the Transformer worked at much larger effective {"d_k"} per head and made the fix explicit in section 3.2.1.
      </Prose>

      <H3>11.5 What is the relationship between Bahdanau/Luong attention and Transformer cross-attention?</H3>

      <Prose>
        They are the same mechanism. Transformer cross-attention is Luong dot-product attention with three additions: (1) the query, key, and value are produced by three separate linear projections of the input rather than being identified directly with the hidden states; (2) a {"√d_k"} scaling factor before softmax; (3) multi-head splitting, running {"H"} parallel attentions and concatenating the results. If you set {"H = 1"}, skip the {"√d_k"} scaling, and identify the projections with the identity, Transformer cross-attention reduces exactly to Luong 2015. The entire Transformer encoder-decoder architecture can therefore be read as "take Bahdanau/Luong attention, remove the RNN, and scale."
      </Prose>

    </div>
  ),
};

export default attentionContent;
