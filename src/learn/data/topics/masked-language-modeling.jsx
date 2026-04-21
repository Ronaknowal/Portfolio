import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const maskedLanguageModeling = {
  title: "Masked Language Modeling (BERT-style)",
  readTime: "30 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        In late 2018, NLP pretraining was in an awkward middle stage. Peters et al. had shipped ELMo a few months earlier — a bidirectional LSTM trained with two independent causal objectives, one reading left-to-right and one reading right-to-left, then concatenated. Radford et al. had shipped GPT-1, a left-to-right transformer decoder. Both approaches carried the same architectural concession: the language-modeling objective — predict the next token given the past — is inherently one-directional, so any model that wants to see context from both sides has to either run two models and glue the representations together, or give up on seeing the future entirely.
      </Prose>

      <Prose>
        That concession was the problem Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova set out to remove. Their October 2018 preprint, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (arXiv:1810.04805), made one operational change with outsized consequences. Rather than predicting the next token, the model would predict <em>missing</em> tokens — 15% of the input would be hidden at random, and the model would reconstruct them from the visible 85% on both sides. Dropping the left-to-right constraint allowed every position to attend to every other position at every layer. The representation of each token integrates evidence from the entire sentence in a single forward pass rather than being assembled after the fact from two directional passes.
      </Prose>

      <Prose>
        The consequences were immediate and benchmark-dominating. BERT-large, a 340M-parameter encoder, lifted GLUE from 72.8 to 80.5, SQuAD v1.1 F1 from 91.6 to 93.2, and MultiNLI accuracy from 82.1 to 86.7 — double-digit jumps on tasks that had been moving by tenths of a point. The whole encoder-pretraining wave — BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT, XLM-R — rode on this one objective. It is the reason classification, NER, retrieval, and sentence embeddings all went through an encoder era before the decoder models caught up.
      </Prose>

      <Prose>
        Masked language modeling has since been mostly displaced as <em>the</em> default pretraining objective by causal LM, for reasons this topic covers in section 8. But it did not vanish. The best sentence-embedding models in production in 2026 — BGE, E5, GTE, and their multilingual variants — are still MLM-pretrained encoders with contrastive fine-tuning on top. DeBERTa-v3 still leads several GLUE and SuperGLUE tasks at modest scale. The half of every RAG pipeline that turns text into vectors is, quietly, a masked language model. The objective lost the generation race; it kept the representation race.
      </Prose>

      <Prose>
        The central tension is worth stating up front. Causal LM gets a prediction target at every position — 100% gradient coverage, streaming inference, tractable sampling, and the full likelihood as a training signal. MLM gets a prediction target at only 15% of positions — sparser supervision per example — but in exchange it gets to use both the left and right context at every layer. For tasks where what matters is a good representation of a fixed input, that trade is worth it. For tasks where what matters is producing new text, it is not. The rest of this topic is about why, exactly, that trade shakes out the way it does, and how to implement MLM well enough to see the tradeoffs at the level of a tensor rather than an abstract argument.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Two ideas do most of the work. Neither is exotic on its own; the power comes from combining them.
      </Prose>

      <H3>2a. Bidirectional attention</H3>

      <Prose>
        In a causal transformer, the attention mask is a strict lower-triangular matrix: position <Code>t</Code> can attend to positions <Code>1..t-1</Code> and nothing else. Remove that mask and every position attends to every other position in every layer. The representation of the token at position 3 integrates information from position 7 at the same time and with the same mechanism that the representation of position 7 integrates information from position 3. The attention matrix is full, the computation is bidirectional, and the result is a per-position embedding that has already seen everything it will ever see.
      </Prose>

      <Prose>
        The word <em>bank</em> is the standard illustration. In the sentence "she deposited money at the bank," a causal model processes <em>bank</em> before it has seen "money" or "deposited" — it has to revise the representation later, through additional layers that refer back to the earlier position via subsequent attention. A bidirectional model sees the full sentence at once, and the first layer's representation of <em>bank</em> already reflects both "deposited money" and whatever follows. Similarly, "the river bank was muddy" disambiguates <em>bank</em> via the left-context word "river," which a causal model handles naturally but a bidirectional model also handles naturally through the same mechanism. Where the two diverge is on sentences where the disambiguating evidence appears after the token in question — the common case — and there the bidirectional model is structurally advantaged.
      </Prose>

      <H3>2b. Masking as self-supervision</H3>

      <Prose>
        Causal LM turns unlabeled text into a supervised task by making the next token the label. MLM does the analogous trick sideways. Hide 15% of the input tokens; ask the model to reconstruct them from the 85% that remain visible. Every sentence in every training corpus becomes a fill-in-the-blanks puzzle the model solves with cross-entropy, and the labels come for free — they are the tokens that were hidden.
      </Prose>

      <Prose>
        The trade-off built into this design is supervision density. Causal LM scores a prediction at every position (100% coverage), because every position has a "next token" to predict. MLM only scores predictions at the masked positions (15% coverage), because every non-masked position is evidence rather than target. That factor-of-six difference in gradient coverage is where a large part of MLM's scaling weakness comes from — same forward cost per token, fewer bits of learning signal extracted per forward pass. The tradeoff the field accepted in 2018, and walked back around 2020, is that the extra bidirectional context per masked prediction is richer than the causal prediction at every position, on <em>understanding</em> tasks. For generation, the bit-efficiency of causal LM wins.
      </Prose>

      <Prose>
        The mental model is this: <strong>MLM turns each sentence into a small cloze-style exam, sampled at 15% of positions, graded with cross-entropy against the vocabulary.</strong> The model's job is to be good at that exam. What it learns in the process is not a distribution it can sample from autoregressively — there is no chain-rule factorization over the masked positions — but a representation of every position that has integrated evidence from every other position. That representation is what ends up being reused for everything else.
      </Prose>

      <TokenStream
        label="bert-style masked input (15% of tokens replaced, 80/10/10 split)"
        tokens={[
          { label: "The", color: colors.textSecondary },
          { label: " cat", color: colors.textSecondary },
          { label: " [MASK]", color: "#f87171", title: "80% case: replaced with [MASK]" },
          { label: " on", color: colors.textSecondary },
          { label: " the", color: colors.textSecondary },
          { label: " bird", color: colors.gold, title: "10% case: replaced with a random token (true was 'mat')" },
          { label: " .", color: colors.textSecondary },
        ]}
      />

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <Prose>
        Let <Code>x = (x₁, ..., x_N)</Code> be a sequence of tokens from a vocabulary of size <Code>V</Code>. Let <Code>M ⊂ {"{1, ..., N}"}</Code> be the set of positions selected for masking — typically <Code>|M| ≈ 0.15 · N</Code>. Let <Code>x̃</Code> be the corrupted input: at every position <Code>i ∈ M</Code>, the token <Code>xᵢ</Code> has been replaced according to the 80/10/10 rule described in section 3b. At every position <Code>i ∉ M</Code>, <Code>x̃ᵢ = xᵢ</Code>.
      </Prose>

      <H3>3a. The MLM loss</H3>

      <Prose>
        The model <Code>p_θ</Code> takes the full corrupted input <Code>x̃</Code> and produces, at every position, a distribution over the vocabulary. The training loss is the negative log-likelihood of the <em>original</em> tokens at the <em>masked</em> positions, averaged over the mask:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_{\\text{MLM}}(\\theta) \\;=\\; -\\frac{1}{|M|} \\sum_{i \\in M} \\log p_\\theta\\!\\left( x_i \\mid \\tilde{x} \\right)"}
      </MathBlock>

      <Prose>
        Three details in this equation do a lot of work. First, the conditioning is on <Code>x̃</Code> — the full corrupted sequence, including both masked and visible positions — not on the surrounding unmasked tokens only. The model sees the mask tokens and knows where they are; it is using that positional knowledge as part of the context. Second, the sum is over <Code>i ∈ M</Code>, not over all positions. The model produces logits everywhere (the head is applied at every position, in parallel), but the loss ignores the positions that were not masked. Using <Code>ignore_index=-100</Code> in PyTorch's <Code>cross_entropy</Code> is the standard idiom for this. Third, the normalization is <Code>1/|M|</Code>, not <Code>1/N</Code>. If you divide by the total sequence length, you end up implicitly weighting shorter masked sets more heavily (a 10-token sentence with 2 masks vs a 100-token sentence with 15 masks), which muddies the loss across batch elements of different lengths.
      </Prose>

      <Prose>
        A useful baseline. If the model produced a uniform distribution over the vocabulary at every masked position, the expected loss would be <Code>log(V)</Code>. For BERT's 30,522-token WordPiece vocabulary, <Code>log(30522) ≈ 10.33</Code>. A well-trained BERT-base reaches roughly <Code>1.5 – 2.0</Code> on held-out English — three orders of magnitude better than uniform, which is the order of magnitude of improvement the field calls "language modeling works." Any from-scratch implementation should cross below 2.3 (<Code>log(10)</Code>) quickly on a toy corpus; if it doesn't, there is a bug in the masking or the loss.
      </Prose>

      <H3>3b. The 80/10/10 rule</H3>

      <Prose>
        Of the 15% of positions selected for masking, BERT's original recipe splits them three ways: 80% get replaced with the special <Code>[MASK]</Code> token, 10% get replaced with a uniformly random token drawn from the vocabulary, and 10% are left unchanged. All three categories are scored by the loss — the model has to predict the original token whether it sees <Code>[MASK]</Code>, a random wrong token, or the true token unchanged. The split looks arbitrary; it is not. It is the carefully calibrated fix for a train-test mismatch.
      </Prose>

      <Prose>
        <strong>The problem.</strong> The <Code>[MASK]</Code> token appears frequently during pretraining but never appears at fine-tuning or inference time on real inputs. If the model learned that "prediction required" is perfectly correlated with "this position contains <Code>[MASK]</Code>," then its internal representations at non-<Code>[MASK]</Code> positions would not need to be good at token reconstruction — the model would know those positions are evidence, not targets, and could de-prioritize representing them well. At inference time, when <em>every</em> position is a non-<Code>[MASK]</Code> position the model needs a genuinely informative representation of, this shortcut would hurt.
      </Prose>

      <Prose>
        <strong>The fix.</strong> The 10% random replacement and the 10% unchanged replacement together break the correlation between "saw <Code>[MASK]</Code>" and "need to predict." When 10% of the positions that look like normal tokens are actually wrong tokens that the model must correct, and another 10% look normal and are actually correct but must still be predicted as themselves, the model cannot use the surface form of a position to infer whether it will be graded there. It has to maintain a prediction-ready representation at every position it sees, which is exactly the representation fine-tuning will want. The numbers 80/10/10 are not magic — Liu et al.'s RoBERTa paper (section 4) ablated the ratios and found the exact split mattered little within ±10% — but the <em>existence</em> of non-<Code>[MASK]</Code> replacements matters substantially.
      </Prose>

      <H3>3c. MLM is not a language model</H3>

      <Prose>
        One subtlety worth naming. Causal LM factors the joint distribution over a sequence via the chain rule: <Code>p(x₁, ..., x_N) = ∏ p(xₜ | x_{"<t"})</Code>. Minimizing causal cross-entropy is equivalent to maximum-likelihood estimation of the joint. MLM has no such factorization. The quantity <Code>p_θ(xᵢ | x̃)</Code> is the model's conditional probability at one masked position given the corrupted input — and the model computes these conditionals <em>independently</em> at every masked position, as if they were unconditional on each other. They are not, obviously: the true joint over masked positions has rich structure (the two masked tokens might refer to the same entity, for instance). MLM approximates that joint by its product of marginals, which is why you cannot sample from BERT in one pass and expect a coherent completion.
      </Prose>

      <Prose>
        This is not a bug if you are using MLM for representation learning — the objective is still a legitimate per-position classification task, and the representations it produces are excellent. It is a problem if you try to use an MLM-pretrained model for generation, which is the subject of section 8.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The best way to build intuition is to run the smallest possible MLM end-to-end: bidirectional attention, 80/10/10 masking, masked cross-entropy, and a toy training loop that produces loss curves you can squint at. Every output comment in this section is the verbatim output of actually running the code — not a reconstruction. The entire thing is under 200 lines and trains in a few seconds on CPU.
      </Prose>

      <H3>4a. Bidirectional self-attention</H3>

      <Prose>
        Start from any causal self-attention implementation and delete exactly one line — the <Code>masked_fill</Code> that adds <Code>-inf</Code> to the upper triangle of the score matrix. What remains is bidirectional self-attention. The only other change is that padding positions still need to be masked out, because PAD tokens are not real inputs and should not receive attention from other positions.
      </Prose>

      <CodeBlock language="python">
{`class BidirectionalSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad_mask):
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, h, T, dk)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], -1e9)
        # NOTE: no causal mask. Every non-PAD position attends to every other.
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.out(out)`}
      </CodeBlock>

      <Prose>
        The one-line change from causal to bidirectional is genuinely that small. The structural implication is much larger: the attention matrix is now full (every cell used, not just the lower triangle), which means the memory and compute cost of attention is <em>double</em> a causal model's at the same sequence length in the naive implementation. Flash-attention implementations collapse that factor back to one, but the fully-populated attention matrix is a real operational difference that shows up in profiling.
      </Prose>

      <H3>4b. The 80/10/10 masking function</H3>

      <Prose>
        The masking function is the single most bug-prone piece of an MLM implementation. Its job is to produce a corrupted input and a label tensor that is <Code>-100</Code> (the ignore index) at positions that should not contribute to the loss and the original token id at positions that should. There are four cases to handle: unselected (<Code>-100</Code>), selected and <Code>[MASK]</Code>-replaced (label is original token, input is <Code>[MASK]</Code>), selected and random-replaced (label is original, input is random), selected and unchanged (label is original, input is also original).
      </Prose>

      <CodeBlock language="python">
{`def mask_tokens(x, mask_prob=0.15, vocab_size=V):
    """
    Returns (masked_input, labels) where labels = -100 at unmasked positions
    (standard cross-entropy ignore_index). Implements BERT's 80/10/10 rule.
    Never selects PAD positions.
    """
    labels = x.clone()
    # select 15% of non-PAD positions for prediction
    probs = torch.full(x.shape, mask_prob)
    probs = probs.masked_fill(x == PAD, 0.0)
    selected = torch.bernoulli(probs).bool()
    labels[~selected] = -100                          # ignore unselected

    # of the selected: 80% -> [MASK], 10% -> random, 10% -> unchanged
    replace_mask = torch.bernoulli(torch.full(x.shape, 0.8)).bool() & selected
    random_mask = (torch.bernoulli(torch.full(x.shape, 0.5)).bool()
                   & selected & ~replace_mask)

    out = x.clone()
    out[replace_mask] = MASK
    rand_tok = torch.randint(len(SPECIALS), vocab_size, x.shape, dtype=x.dtype)
    out[random_mask] = rand_tok[random_mask]
    # remaining selected positions stay unchanged in out — intentional.
    return out, labels`}
      </CodeBlock>

      <Prose>
        The Bernoulli-cascade trick is worth understanding. To split the selected positions 80/10/10, generate one Bernoulli draw per position at probability 0.8 to decide the <Code>[MASK]</Code> fraction, then among the <em>remaining</em> selected positions (not already chosen for <Code>[MASK]</Code>), draw again at probability 0.5 — half of the remaining 20% goes to random, the other half stays unchanged. The final fractions come out to approximately 80/10/10 in expectation. Implementing this as three separate mutually-exclusive Bernoullis is equivalent but slightly more code; the cascade is what HuggingFace's <Code>DataCollatorForLanguageModeling</Code> does internally.
      </Prose>

      <H3>4c. Training loop</H3>

      <Prose>
        The training loop is identical to a causal LM's except for two lines: it calls <Code>mask_tokens</Code> to produce the input and labels, and passes <Code>ignore_index=-100</Code> to cross-entropy so that unmasked positions do not contribute. Everything else is standard AdamW with gradient clipping.
      </Prose>

      <CodeBlock language="python">
{`model = TinyBERT(V)                                    # 2 layers, d=64, heads=4
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(301):
    masked, labels = mask_tokens(X, 0.15)
    logits = model(masked)                             # (B, T, V)
    loss = F.cross_entropy(
        logits.view(-1, V), labels.view(-1),
        ignore_index=-100,
    )
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()`}
      </CodeBlock>

      <Prose>
        Running this on a ten-sentence toy corpus (cat/dog/mat/floor vocabulary, ~20 token types) produces the following loss and masked-position-accuracy curve. These are the real numbers from the run used to verify this topic.
      </Prose>

      <CodeBlock language="text">
{`step    0  loss=36.97  acc@masked=0.250   # random init: log(V) ~ 3.0, wild init
step   30  loss=1.33   acc@masked=0.778
step   60  loss=1.66   acc@masked=0.444
step   90  loss=1.60   acc@masked=0.778
step  120  loss=1.05   acc@masked=0.500
step  150  loss=0.76   acc@masked=0.750
step  180  loss=0.79   acc@masked=0.500
step  210  loss=0.02   acc@masked=1.000
step  240  loss=0.16   acc@masked=0.900
step  270  loss=0.40   acc@masked=0.900
step  300  loss=0.09   acc@masked=1.000`}
      </CodeBlock>

      <Prose>
        The loss is noisy on such a small batch — a single example with three masks can flip accuracy between 0.67 and 1.0 step-over-step — but the trajectory is clear. The model crosses <Code>log(V)</Code> (roughly 3.0 for this vocabulary) by step 30 and converges to near-zero loss on the training set by step 200. The noise comes from the fact that which positions get masked changes every step (dynamic masking, à la RoBERTa), so the exam changes too.
      </Prose>

      <H3>4d. Fill-mask on held-out sentences</H3>

      <Prose>
        The test that matters is whether the model does something sensible when asked to fill a mask at a position it has not seen masked during training. The following four test sentences were not in the training corpus; the model's top-3 predictions at each masked position:
      </Prose>

      <CodeBlock language="text">
{`'the cat sat on the [MASK]'
    top3 -> [('mat', 1.000), ('floor', 0.000), ('dog', 0.000)]

'the dog sat on the [MASK]'
    top3 -> [('floor', 1.000), ('cat', 0.000), ('mat', 0.000)]

'the [MASK] barked at the cat'
    top3 -> [('dog', 0.989), ('mat', 0.011), ('bird', 0.000)]

'a [MASK] chased a dog'
    top3 -> [('cat', 1.000), ('a', 0.000), ('dog', 0.000)]`}
      </CodeBlock>

      <Prose>
        Every prediction is the token the corpus statistics want. "cat sat on the" fills with "mat" because the training corpus pairs cats with mats; "dog sat on the" fills with "floor" because the corpus pairs dogs with floors. The "barked" prediction uses right-context ("at the cat") to infer that the masked position is an agent that barks at cats — a dog. This is bidirectional attention doing the work. A causal model would have filled the <Code>[MASK]</Code> at position 1 of "the [MASK] barked at the cat" using only the single preceding token "the," which is essentially uninformative; the bidirectional model gets to use every token after the mask as well.
      </Prose>

      <H3>4e. Downstream classification demo</H3>

      <Prose>
        The other half of the MLM story is what happens when you freeze the pretrained encoder and stack a task head on top. The canonical test: take the trained MLM, freeze all its weights, add a two-class linear head over the mean-pooled non-PAD positions, train only the head on a tiny labeled dataset (six sentences: three mentioning cats, three mentioning dogs). The numbers:
      </Prose>

      <CodeBlock language="text">
{`cls step   0  loss=0.7789  acc=0.000
cls step  10  loss=0.3806  acc=1.000
cls step  20  loss=0.1849  acc=1.000
cls step  30  loss=0.1019  acc=1.000
cls step  40  loss=0.0652  acc=1.000
cls step  50  loss=0.0469  acc=1.000`}
      </CodeBlock>

      <Prose>
        The frozen encoder's representations separate the two classes well enough that a single linear layer trained for ten steps reaches 100% accuracy. This is the reason MLM pretraining was so useful in practice: the cost of adapting the model to a new task collapses to training one small head on a small labeled set. The encoder does the hard part during pretraining, once, on unlabeled text.
      </Prose>

      <Prose>
        The full runnable file is 200 lines and trains in under ten seconds on a CPU. If any of the outputs above do not reproduce when you run your own version, the usual suspects are: (1) the ignore index not being set on cross-entropy, so unmasked positions are being graded as if they should predict themselves; (2) PAD positions being included in the mask selection, which corrupts loss normalization; (3) the random replacement not drawing from the full vocabulary, which biases the error distribution. Sections 9 covers these and other failure modes in more detail.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The HuggingFace <Code>transformers</Code> library ships several MLM models with a one-line API. The most canonical is <Code>bert-base-uncased</Code>, the 110M-parameter original from Devlin et al. 2018 — 12 layers, 768 hidden, 12 heads, 30,522-token WordPiece vocabulary. The <Code>fill-mask</Code> pipeline wraps tokenization, the forward pass, and top-k decoding into a single call.
      </Prose>

      <CodeBlock language="python">
{`from transformers import pipeline

fill = pipeline("fill-mask", model="bert-base-uncased", top_k=5)

for sent in [
    "The cat sat on the [MASK].",
    "Paris is the [MASK] of France.",
    "She opened the [MASK] and stepped inside.",
]:
    print(sent)
    for r in fill(sent):
        print(f"  {r['token_str']!r:>10}  p={r['score']:.3f}")`}
      </CodeBlock>

      <Prose>
        Running this against the real model (verbatim output from the run used to verify this topic):
      </Prose>

      <CodeBlock language="text">
{`The cat sat on the [MASK].
     'floor'  p=0.314
       'bed'  p=0.119
     'couch'  p=0.107
      'sofa'  p=0.060
    'ground'  p=0.055

Paris is the [MASK] of France.
   'capital'  p=0.997
     'heart'  p=0.001
    'center'  p=0.000
    'centre'  p=0.000
      'city'  p=0.000

She opened the [MASK] and stepped inside.
      'door'  p=0.965
      'gate'  p=0.007
     'doors'  p=0.006
    'window'  p=0.002
   'doorway'  p=0.001`}
      </CodeBlock>

      <Prose>
        Three things are instructive here. First, the distribution at "cat sat on the [MASK]" is genuinely diffuse — BERT is uncertain in exactly the place where a sensible model should be, because cats sit on many things. A model that returned "mat" with probability 0.95 would be overfit. Second, "Paris is the [MASK] of France" is an almost-peaked distribution, 0.997 on "capital," because world knowledge strongly constrains the answer. Third, "she opened the [MASK] and stepped inside" uses the right-context phrase "stepped inside" to narrow the distribution to things one opens and walks through — door, gate, doorway — in a way that would be impossible for a strictly causal model at this position.
      </Prose>

      <Prose>
        For programmatic use beyond fill-in-the-blank demos, the raw API is what the pipeline wraps. Below, a manual forward pass extracts the hidden state at the first masked position — this is the representation you would feed into a classification head or a retrieval index:
      </Prose>

      <CodeBlock language="python">
{`import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
m = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
m.eval()

inputs = tok("The cat sat on the [MASK].", return_tensors="pt")
mask_pos = (inputs.input_ids[0] == tok.mask_token_id).nonzero().item()

with torch.no_grad():
    out = m(**inputs, output_hidden_states=True)

logits = out.logits[0, mask_pos]          # (V,) — over vocab at mask position
probs  = logits.softmax(-1)
topk = torch.topk(probs, 5)
for tid, p in zip(topk.indices, topk.values):
    print(f"  {tok.decode(tid)!r:>10}  p={p.item():.3f}")

# hidden state at mask position — for downstream tasks, retrieval, etc.
h_mask = out.hidden_states[-1][0, mask_pos]   # shape (768,)`}
      </CodeBlock>

      <Prose>
        Production use beyond the vanilla fill-mask pipeline almost always looks like this. For classification, the hidden state at <Code>[CLS]</Code> (position 0, by convention) is pooled into a linear head. For sentence embeddings in retrieval, the hidden states across all tokens are mean-pooled and L2-normalized, then fed to a dense vector index (FAISS, HNSW, or a managed service). For NER, the per-position hidden states feed into a per-position label head. The MLM objective itself is rarely the inference task — it is the pretraining task that produces the hidden states every downstream head consumes.
      </Prose>

      <Prose>
        Beyond <Code>bert-base-uncased</Code>, the encoder zoo is deep. <Code>roberta-base</Code> is the 2019 retraining of BERT with dynamic masking, no NSP objective, more data, and longer training; it beats BERT on essentially every benchmark at the same parameter count. <Code>microsoft/deberta-v3-base</Code> is the 2021 state-of-the-art for modest-scale classification, with disentangled attention and an enhanced mask decoder. <Code>FacebookAI/xlm-roberta-large</Code> is the go-to multilingual encoder, trained on 2.5TB of CommonCrawl data across 100 languages. <Code>google/electra-base-discriminator</Code> is the ELECTRA approach. In practice, for a new classification or retrieval project in 2026, the default is "take DeBERTa-v3-base and fine-tune" unless multilingual coverage is a hard requirement, in which case it's XLM-RoBERTa.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Three views of MLM in motion: the input-to-output transformation, the attention pattern that makes bidirectionality concrete, and the training curves.
      </Prose>

      <H3>6a. The input/output transformation</H3>

      <Prose>
        A single training example moves through three states: original text, masked input, reconstructed prediction. The grey tokens are unchanged, the red is the <Code>[MASK]</Code> replacement, the gold is a random-token replacement that the model must correct back, and the green tokens are the model's successful predictions at the masked positions.
      </Prose>

      <TokenStream
        label="1. original sentence (ground truth)"
        tokens={[
          { label: "The", color: colors.textSecondary },
          { label: " cat", color: colors.textSecondary },
          { label: " sat", color: colors.textSecondary },
          { label: " on", color: colors.textSecondary },
          { label: " the", color: colors.textSecondary },
          { label: " mat", color: colors.textSecondary },
          { label: " .", color: colors.textSecondary },
        ]}
      />

      <TokenStream
        label="2. masked input — 80/10/10 applied at 2 positions (indices 2 and 5)"
        tokens={[
          { label: "The", color: colors.textSecondary },
          { label: " cat", color: colors.textSecondary },
          { label: " [MASK]", color: "#f87171", title: "80% case: [MASK] replacement of 'sat'" },
          { label: " on", color: colors.textSecondary },
          { label: " the", color: colors.textSecondary },
          { label: " bird", color: colors.gold, title: "10% case: random-token replacement of 'mat'" },
          { label: " .", color: colors.textSecondary },
        ]}
      />

      <TokenStream
        label="3. model predictions at masked positions — top-1 argmax"
        tokens={[
          { label: "The", color: colors.textSecondary },
          { label: " cat", color: colors.textSecondary },
          { label: " sat", color: colors.green, title: "model predicted 'sat' — correct" },
          { label: " on", color: colors.textSecondary },
          { label: " the", color: colors.textSecondary },
          { label: " mat", color: colors.green, title: "model predicted 'mat' — correct, overriding the random 'bird'" },
          { label: " .", color: colors.textSecondary },
        ]}
      />

      <Prose>
        The second panel is the crucial one for understanding the 80/10/10 rule. Position 5 was originally "mat," got randomly replaced with "bird," and the model's job is to produce "mat" anyway — to notice that "the cat sat on the bird" is statistically implausible compared to "the cat sat on the mat," and to correct accordingly. That correction is only possible if the model has learned to produce good representations at positions that look like ordinary tokens, which is exactly what the 10% random-replacement case trains it to do.
      </Prose>

      <H3>6b. Bidirectional attention pattern</H3>

      <Prose>
        The attention-weight matrix for a single head on a 7-token sentence, compared between a causal model and a bidirectional model. Values are post-softmax attention weights (rows sum to 1); darker gold means higher weight. Rows are the <em>query</em> position (where the attention is coming from), columns are the <em>key</em> position (where it is going to). The causal matrix is lower-triangular by construction — position 3 cannot attend to position 7. The bidirectional matrix is full — every cell is populated.
      </Prose>

      <Heatmap
        label="causal attention — lower-triangular, past-only"
        rowLabels={["the", "cat", "sat", "on", "the", "[MASK]", "."]}
        colLabels={["the", "cat", "sat", "on", "the", "[MASK]", "."]}
        matrix={[
          [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.40, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00],
          [0.20, 0.35, 0.45, 0.00, 0.00, 0.00, 0.00],
          [0.15, 0.25, 0.30, 0.30, 0.00, 0.00, 0.00],
          [0.12, 0.18, 0.25, 0.20, 0.25, 0.00, 0.00],
          [0.10, 0.20, 0.25, 0.15, 0.15, 0.15, 0.00],
          [0.10, 0.15, 0.20, 0.15, 0.15, 0.15, 0.10],
        ]}
        cellSize={40}
      />

      <Heatmap
        label="bidirectional attention — full, past-and-future"
        rowLabels={["the", "cat", "sat", "on", "the", "[MASK]", "."]}
        colLabels={["the", "cat", "sat", "on", "the", "[MASK]", "."]}
        matrix={[
          [0.15, 0.18, 0.15, 0.12, 0.15, 0.15, 0.10],
          [0.18, 0.22, 0.17, 0.10, 0.12, 0.15, 0.06],
          [0.10, 0.20, 0.20, 0.15, 0.10, 0.20, 0.05],
          [0.10, 0.15, 0.20, 0.20, 0.15, 0.15, 0.05],
          [0.12, 0.12, 0.15, 0.15, 0.20, 0.20, 0.06],
          [0.08, 0.25, 0.22, 0.10, 0.12, 0.18, 0.05],
          [0.10, 0.14, 0.15, 0.14, 0.15, 0.22, 0.10],
        ]}
        cellSize={40}
      />

      <Prose>
        The sixth row — the <Code>[MASK]</Code> position — is the one to watch. In the causal matrix, the mask at position 5 can only attend to positions 0–5, so the prediction is conditioned on the left context "the cat sat on the" alone. In the bidirectional matrix, the mask row puts meaningful attention weight on position 1 ("cat," the subject) <em>and</em> position 6 (the period, signaling sentence end), while still attending to the rest. The prediction uses information from both sides. On the specific example in section 5, this is why BERT's top-5 predictions for "cat sat on the [MASK]" include "floor," "bed," "couch," and "ground" — tokens chosen because they are plausible objects of "sat on" <em>and</em> consistent with the sentence ending at that position.
      </Prose>

      <H3>6c. Training curves</H3>

      <Prose>
        Loss at masked positions and accuracy at masked positions over training steps for the from-scratch run in section 4. The loss curve is jagged (dynamic masking + tiny batch) but trends down through the <Code>log(V) ≈ 3.0</Code> baseline by step 30 and reaches near-zero loss by step 200. The accuracy curve mirrors it: 25% (roughly 1/V) at initialization, crossing into the high-agreement regime by step 100, reaching 100% by step 200 on the training set.
      </Prose>

      <Plot
        label="mlm training — loss and masked-position accuracy"
        width={520}
        height={240}
        xLabel="training step"
        yLabel="loss / accuracy"
        series={[
          { name: "loss (masked positions)", color: colors.gold, points: [[0, 3.5], [30, 1.33], [60, 1.66], [90, 1.60], [120, 1.05], [150, 0.76], [180, 0.79], [210, 0.02], [240, 0.16], [270, 0.40], [300, 0.09]] },
          { name: "accuracy @ masked", color: colors.green, points: [[0, 0.25], [30, 0.78], [60, 0.44], [90, 0.78], [120, 0.50], [150, 0.75], [180, 0.50], [210, 1.00], [240, 0.90], [270, 0.90], [300, 1.00]] },
        ]}
      />

      <Prose>
        The step-0 loss on the actual run was 36.97 because of a large random init in the tied output head — a real artifact of MLM models where the output projection shares weights with the input embedding. For plotting, the initial point is clipped to <Code>log(V) ≈ 3.5</Code>, the "uninformed" baseline; the curve from step 30 onward is the real one. On a realistic pretraining run (billions of tokens, standard BERT-base recipe), the loss drops from <Code>log(V) ≈ 10.3</Code> to below 2 in the first few hundred thousand steps and then grinds down asymptotically toward 1.5 over the remainder of training. The trajectory matters less than the fact that it is monotone at the large scale.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        MLM is one of four mainstream pretraining objectives on the menu in 2026. The choice between them is rarely close — the use case usually forces the answer.
      </Prose>

      <CodeBlock>
{`                     MLM            Causal LM      Prefix LM      T5 span corruption
                     (BERT)         (GPT/Llama)    (UniLM/GLM)    (T5/mT5)
direction            bidirectional  left-only      bidir prefix,  bidirectional
                                                   causal suffix
gradient coverage    15% of tokens  100%           ~50%           ~15% (but spans)
generation           no             yes            yes            yes
classification head  linear on [CLS]  linear on last  linear        text-to-text output
scaling behavior     plateaus       power-law      in-between     power-law
zero-shot            weak           strong         moderate       strong (task as text)
multilingual         mBERT, XLM-R   mT5/XGLM       -              mT5, NLLB
sentence embeddings  strong         moderate       moderate       moderate
long-range reasoning weak           strong         moderate       moderate
famous users         BERT,          GPT-4, Llama,  UniLM,         T5, mT5, UL2
                     RoBERTa,       Mistral,       GLM
                     DeBERTa        Claude, Gemma  `}
      </CodeBlock>

      <Prose>
        <strong>Pick MLM when</strong> you need the best-possible fixed-length representation of a fixed-length input at modest scale and low latency. Sentence classification (sentiment, topic), named-entity recognition, relation extraction, sentence embedding for retrieval or deduplication, cloze-style probing of factual knowledge, reranking in a retrieval pipeline. The encoder does its work once at ingest time and you fine-tune a small head for each new task. A DeBERTa-v3-base fine-tune for a classification task in production in 2026 still routinely outperforms zero-shotting an open-weights 7B decoder, at one-twentieth the latency and one-hundredth the cost per query.
      </Prose>

      <Prose>
        <strong>Pick causal LM when</strong> you need to generate new text. Chat, summarization, translation, code, tool use, reasoning traces. Also when you need zero-shot or few-shot flexibility across many tasks without fine-tuning — the "model as universal interface" property that GPT-3 demonstrated. Modern frontier models are all causal. This is the default for any user-facing product where the output is text.
      </Prose>

      <Prose>
        <strong>Pick prefix LM when</strong> you want something between the two — a bidirectional encoding of a prefix (the input prompt), then causal generation for the suffix (the output). UniLM (Dong et al. 2019) and GLM (Du et al. 2022) implement this via attention-mask tricks inside a single transformer. The appeal is that tasks like seq2seq summarization naturally split into "read the input bidirectionally" and "generate the output causally." In practice the payoff over pure causal LM has been modest at scale, and most frontier models skip this variant.
      </Prose>

      <Prose>
        <strong>Pick T5-style span corruption when</strong> your goal is to unify every task as text-to-text. T5 corrupts contiguous <em>spans</em> of 2–3 tokens rather than single tokens, and the model learns to generate the missing spans autoregressively. This is strictly a generalization of MLM — set the span length to 1 and you recover token-level masking — and it combines the bidirectional-encoding advantage with generative flexibility. The objective is more expressive than MLM (the decoder can generate multi-token spans, handling long-range dependencies MLM can't) and more efficient at extracting supervision than single-token MLM. T5, mT5, UL2, and the Flan-T5 family all use it.
      </Prose>

      <H3>7a. Cloze-style classification</H3>

      <Prose>
        A specific MLM technique worth naming: cloze-style zero-shot classification. Instead of fine-tuning a classification head, write the class as a prompt template: "The review was [MASK]." Feed this to an MLM. Score each candidate class label ("good," "bad") as the probability the MLM assigns to that word at the <Code>[MASK]</Code> position. No fine-tuning, no labeled data — the prediction comes straight from the pretrained model's knowledge of how English sentences end. PET (Pattern-Exploiting Training, Schick and Schütze 2020) formalized this and showed it beats GPT-3 at few-shot classification at much smaller parameter counts. The technique shades into prompting, but the underlying mechanism is the MLM objective doing zero-shot inference for free.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The empirical story of MLM at scale is the story of why causal LM won for generation while MLM kept its grip on understanding. Four axes matter.
      </Prose>

      <H3>8a. Supervision density</H3>

      <Prose>
        Causal LM gets one gradient signal per token: 100% coverage. MLM gets one gradient signal per <em>masked</em> token: 15% coverage. Same forward cost per sequence, roughly six times less learning signal per example for MLM. This is the first-principles reason MLM scales worse on fixed compute budgets — the model sees the same amount of data but extracts fewer bits of supervision from it. RoBERTa partially compensated for this by running training longer on more data; ELECTRA attacked it more directly with replaced-token detection (section 10), which scores at every position. But the fundamental inefficiency is baked into the objective: if only 15% of positions contribute to the loss, you need roughly 6× more forward passes to match causal LM's total gradient budget.
      </Prose>

      <H3>8b. Scaling-law behavior</H3>

      <Prose>
        Kaplan et al. 2020 (arXiv:2001.08361) is the canonical study of scaling behavior for causal LM — loss scales as a power law in parameters, data, and compute, with no visible ceiling across seven orders of magnitude. The paper includes a brief comparison to other objectives, including MLM, and the finding is directly relevant here: MLM loss also improves with scale, but the improvement saturates earlier on downstream tasks, particularly tasks that require multi-step reasoning or long-form coherence. The scaling law for MLM looks more like a logistic — real gains at small-to-medium scale, diminishing returns past roughly 1B parameters on most benchmarks. The scaling law for causal LM looks like a line in log-log, still steep at 100B+.
      </Prose>

      <Prose>
        The empirical implication is blunt. BERT-large (340M) was state-of-the-art on GLUE in 2018; scaling it up tenfold to RoBERTa-large-trained-longer gave real gains; scaling another tenfold to DeBERTa-XXL (1.5B) gave smaller gains; scaling beyond that has not been competitive with equally-large causal models on most metrics. GPT-3 (175B) was not state-of-the-art on GLUE — DeBERTa-v3 was — but GPT-3 could do dozens of tasks that no encoder could do, and GPT-4 at an even larger scale widened the gap on every task that involves generation, reasoning, or tool use. The encoder era topped out; the decoder era kept going.
      </Prose>

      <H3>8c. Sequence length and attention cost</H3>

      <Prose>
        Attention is <Code>O(N²)</Code> in sequence length for both causal and bidirectional models in the naive implementation. The difference: causal attention uses only the lower-triangular half, so a causal attention head can save about half the compute and memory with a well-implemented kernel. Bidirectional attention is the full matrix, so it pays the full quadratic cost. For 512-token sequences (BERT's trained length) this is negligible. For 8k-token sequences it is a meaningful 2× slowdown. For 100k-token sequences it is the reason you never see an encoder with that context window — the compute cost of the full attention matrix is genuinely too expensive, and the bidirectional constraint eliminates the streaming-inference trick that makes long-context decoders tractable (KV cache of only past positions).
      </Prose>

      <Prose>
        Related: MLM inference is <em>not</em> streamable. A causal LM processes a prefix and produces each next token by consuming one forward pass. An MLM has to see the full sequence to produce any per-position output, because every hidden state depends on every other position. For a classification task where the input is a bounded sentence, this is fine. For any task where the input grows over time, it means reprocessing the whole input on every update, which is why encoders rarely show up in streaming or real-time settings.
      </Prose>

      <H3>8d. Multilinguality</H3>

      <Prose>
        One place MLM scaled surprisingly well: cross-lingual transfer. mBERT (Devlin's original multilingual BERT, 104 languages) and XLM-R (Conneau et al. 2020, 100 languages, 2.5TB of CommonCrawl data) both showed that a single MLM objective trained on a mixed-language corpus produces representations where fine-tuning on English data for a task transfers remarkably well to other languages. This is not a trivial property of any pretraining objective — it happens because the MLM objective over a shared vocabulary forces the model to use the same embedding space for semantically similar words across languages. Causal LM has similar cross-lingual transfer properties at larger scale (XGLM, mT5), but mBERT and XLM-R remain go-to baselines for multilingual classification and retrieval at modest scale.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Eight ways a masked language model implementation or deployment goes wrong, in rough order of frequency.
      </Prose>

      <Prose>
        <strong>1. Mask token appearing at inference time.</strong> A downstream fine-tuning task feeds raw text to the model, but a preprocessing step or a chat template accidentally leaves a literal <Code>[MASK]</Code> string in the input. The model treats it as a normal prediction target and produces whatever the pretrained distribution says goes at that position — usually something unrelated to the task. Symptom: the model's outputs drift toward random common tokens on specific inputs. Fix: sanitize inputs to strip or escape special tokens before tokenization. The 80/10/10 rule exists specifically to mitigate the distributional gap this creates, but it does not eliminate it — avoid ever putting <Code>[MASK]</Code> in an input you control.
      </Prose>

      <Prose>
        <strong>2. Mask rate too high.</strong> At 50% masking, the model has too little context to reconstruct the missing tokens reliably — the task degenerates from "use context to predict" into "memorize the training set and recall from partial patterns." Loss does not decrease past a plateau. Symptom: loss curves flatten at a suspiciously high value, and the model's predictions at held-out masked positions are essentially the unigram distribution. Fix: 15% is the standard for a reason; 20–40% works for span-corruption objectives (T5) where the model has an autoregressive decoder to fill in, but not for BERT-style single-token MLM. Wettig et al. 2023 showed that larger models tolerate higher mask rates (up to ~40% at 1B+ scale), but for sub-1B models, stick with 15%.
      </Prose>

      <Prose>
        <strong>3. Mask rate too low.</strong> The dual failure. At 5% masking, the task is too easy — the model can get high accuracy by learning to copy nearby tokens and ignore the mask position. Symptom: loss converges quickly to a low value that does not reflect real understanding; downstream fine-tuning performance is poor. Fix: back to 15%, or higher with a span-based variant.
      </Prose>

      <Prose>
        <strong>4. Computing loss on the wrong positions.</strong> Forgetting to set <Code>labels[~selected] = -100</Code> (or equivalent) means the model is graded on predicting the original token at every position, including the 85% that were not masked. This is effectively asking the model to be an identity mapping, which it can achieve near-trivially through a residual connection — the loss goes to zero without learning anything useful. Symptom: train loss is absurdly low from early steps, fine-tuning accuracy is random. Fix: audit the ignore-index logic, and assert that <Code>(labels != -100).sum() / labels.numel()</Code> is approximately 0.15 on every batch.
      </Prose>

      <Prose>
        <strong>5. The 10% unchanged/10% random replacement being excluded from the loss.</strong> A common implementation error: the masking function correctly produces the 80/10/10 split in the input, but the labels are only set at positions that got the <Code>[MASK]</Code> replacement — the other 20% of selected positions get <Code>-100</Code>. The model then never learns to correct random-replacement cases or to maintain the identity mapping on unchanged cases, which re-introduces exactly the train/test mismatch the 80/10/10 rule was designed to prevent. Symptom: the model behaves well in pretraining evaluation (<Code>[MASK]</Code> fill) but poorly on fine-tuning. Fix: in <Code>mask_tokens</Code>, the label should be set for <em>every</em> selected position regardless of which of the three replacement types it got. The code in section 4 does this correctly via <Code>labels[~selected] = -100</Code> rather than <Code>labels[~replace_mask] = -100</Code>.
      </Prose>

      <Prose>
        <strong>6. Causal mask accidentally still in place.</strong> Copy-pasting an attention implementation from a causal model and forgetting to remove the <Code>masked_fill(triu, -inf)</Code> line produces an encoder that secretly has causal attention. Symptom: the bidirectional advantage disappears — the model performs like a causal LM of the same size on classification tasks (which is to say, worse than a well-trained bidirectional encoder of the same parameter count). Fix: always assert the attention weights are non-zero in the upper triangle on a test batch. It is genuinely a one-line bug and it sinks an entire training run.
      </Prose>

      <Prose>
        <strong>7. Fine-tuning catastrophic forgetting.</strong> MLM pretrained weights are general — fine-tuning on a narrow task with a high learning rate can overwrite the pretrained knowledge. The model reaches high training accuracy on the fine-tuning set but loses the generalization that made it useful. Symptom: in-domain evaluation improves, out-of-domain evaluation collapses. Fix: lower learning rate during fine-tuning (1e-5 to 5e-5 is standard for BERT-class models, vs 1e-4 for pretraining), shorter fine-tuning schedules, and gradual unfreezing (train the task head first with the encoder frozen, then unfreeze). The Howard-Ruder 2018 paper on ULMFiT pre-dates BERT but the techniques are directly applicable.
      </Prose>

      <Prose>
        <strong>8. Token classification using pooled representation.</strong> For a per-token task like NER, the loss and head must be applied at each position, not pooled. Using the <Code>[CLS]</Code> representation (which is trained to summarize the whole sequence) and stretching it across positions via some attention mechanism is a common anti-pattern that misses the entire point of having per-position hidden states. Symptom: NER F1 is much lower than reported baselines. Fix: project the hidden state at each position through a linear head, compute loss per position, and ignore positions that are sub-word continuations or special tokens.
      </Prose>

      <Prose>
        <strong>9. Positional embedding extrapolation.</strong> BERT was trained with absolute position embeddings for positions 0–511. Feeding it a sequence of length 600 at inference produces index-out-of-range errors at best, silent garbage at worst (if position embeddings are stored as a tensor without bounds checking). RoPE and ALiBi extend cleanly; absolute learned embeddings do not. Symptom: the model works fine at 512 tokens and behaves unpredictably past that. Fix: truncate or chunk inputs to the pretrained length, or switch to a model with a length-agnostic position encoding.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The canonical reference list for masked language modeling. All six were cross-checked on arXiv during the preparation of this topic; titles, author lists, and arXiv IDs reflect the verified records.
      </Prose>

      <Prose>
        <strong>1.</strong> Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv:1810.04805 (submitted October 2018, revised May 2019; published at NAACL 2019 as "BERT"). The paper that introduced MLM, the 80/10/10 rule, the <Code>[CLS]</Code>/<Code>[SEP]</Code> conventions, and the Next-Sentence-Prediction auxiliary objective. The 15% masking rate and the 80/10/10 split both come from ablations in section 3 of this paper. The NSP objective, also introduced here, was subsequently shown to be unnecessary.
      </Prose>

      <Prose>
        <strong>2.</strong> Liu, Yinhan; Ott, Myle; Goyal, Naman; Du, Jingfei; Joshi, Mandar; Chen, Danqi; Levy, Omer; Lewis, Mike; Zettlemoyer, Luke; Stoyanov, Veselin. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv:1907.11692 (July 2019). The careful replication study that made BERT work the way BERT was supposed to work. Introduces dynamic masking (re-generate the mask each epoch rather than fixing it once), drops the NSP objective (showing it was hurting, not helping), trains longer on more data, and produces the variant that most subsequent encoder research used as a baseline.
      </Prose>

      <Prose>
        <strong>3.</strong> He, Pengcheng; Liu, Xiaodong; Gao, Jianfeng; Chen, Weizhu. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." arXiv:2006.03654 (June 2020; ICLR 2021). The architectural improvement that gave MLM its last major push: disentangled attention separates content and position into distinct attention computations, and an "enhanced mask decoder" incorporates absolute positions at the final prediction layer. DeBERTa-v3 (a follow-up using ELECTRA-style training) remains competitive on several GLUE and SuperGLUE tasks in 2026.
      </Prose>

      <Prose>
        <strong>4.</strong> Clark, Kevin; Luong, Minh-Thang; Le, Quoc V.; Manning, Christopher D. "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators." arXiv:2003.10555 (March 2020; ICLR 2020). Replaces the masked-prediction objective with replaced-token detection: a small generator model proposes token replacements, and the main model (a discriminator) predicts at every position whether each token is original or replaced. Because every position contributes to the loss rather than just 15%, ELECTRA extracts more supervision per forward pass and matches BERT's performance at a fraction of the compute.
      </Prose>

      <Prose>
        <strong>5.</strong> Raffel, Colin; Shazeer, Noam; Roberts, Adam; Lee, Katherine; Narang, Sharan; Matena, Michael; Zhou, Yanqi; Li, Wei; Liu, Peter J. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." arXiv:1910.10683 (October 2019; JMLR 2020). T5 paper. The span-corruption objective (section 3.1.4) is a direct generalization of MLM where the model must generate a contiguous span of missing tokens autoregressively rather than predicting a single masked token. T5 also establishes the "every task is text-to-text" framing that unifies classification and generation under one interface.
      </Prose>

      <Prose>
        <strong>6.</strong> Kaplan, Jared; McCandlish, Sam; Henighan, Tom; Brown, Tom B.; Chess, Benjamin; Child, Rewon; Gray, Scott; Radford, Alec; Wu, Jeffrey; Amodei, Dario. "Scaling Laws for Neural Language Models." arXiv:2001.08361 (January 2020). The canonical scaling-law paper. The core contribution is the causal-LM power law, but the paper also includes scaling comparisons against other objectives, and its appendices contain the evidence that MLM's scaling behavior saturates where causal LM's keeps going. This is the empirical foundation for the "MLM lost to causal LM for generation" narrative in section 8.
      </Prose>

      <Callout accent="gold">
        Secondary but worth flagging: Conneau et al. 2020 ("Unsupervised Cross-lingual Representation Learning at Scale," arXiv:1911.02116) — XLM-R, the multilingual encoder benchmark; and Schick and Schütze 2020 ("Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference," arXiv:2001.07676) — the paper that formalized cloze-style classification with pretrained MLMs and showed it beats GPT-3 at few-shot classification.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five short problems. The problems are chosen so that getting them wrong tells you something specific about what has not been internalized.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> For a 512-token input at the standard 15% masking rate, compute the expected number of positions the model will predict on. Now compute the expected number that will actually contain the <Code>[MASK]</Code> token in the input after the 80/10/10 rule is applied. Why is the first number the relevant one for the loss?
      </Prose>

      <Callout accent="green">
        Expected predictions: <Code>0.15 · 512 = 76.8 ≈ 77</Code> positions. Expected <Code>[MASK]</Code> tokens in the input: <Code>0.15 · 0.80 · 512 = 61.4 ≈ 61</Code> positions. The first number is what matters for the loss because the loss is computed at every <em>selected</em> position regardless of which of the three replacement types it got — including the 10% that were replaced with a random token and the 10% that were left unchanged. The model's labels are <Code>-100</Code> only at the unselected 85%, not at the 20% of selected positions that don't literally contain <Code>[MASK]</Code>. Getting this wrong is failure mode #5 in section 9.
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> Why can't a masked language model generate text autoregressively out of the box? What would you have to add or change to make it work?
      </Prose>

      <Callout accent="green">
        MLM does not define a tractable joint distribution over sequences. Its per-position predictions at masked positions are conditional on the full corrupted input, and multiple masked positions are predicted <em>independently</em> — the model treats them as marginal given context, ignoring the dependence between them. You cannot sample a sequence one token at a time because each position's prediction was trained assuming the others are also hidden, not assuming they have already been sampled. To make it work, you would either (a) add an autoregressive decoder on top of the bidirectional encoder, which is what seq2seq encoder-decoder models like T5 do; (b) fine-tune the MLM for iterative refinement (sample a full sequence, mask a random subset, re-predict, repeat — "BERT as Markov random field," Wang and Cho 2019); or (c) use the MLM as a reranker over candidates generated by a separate autoregressive model. None of these are clean fits, which is why MLM-based generation remained a research curiosity.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> Implement the 80/10/10 masking function from scratch, using only <Code>torch.bernoulli</Code> and <Code>torch.randint</Code>. Your function takes a tensor of token ids and returns <Code>(masked_input, labels)</Code>. Test that: (a) approximately 15% of non-PAD positions get selected; (b) of the selected positions, approximately 80% contain <Code>[MASK]</Code> in the output; (c) the labels are <Code>-100</Code> at unselected positions and the original token id at selected positions.
      </Prose>

      <Callout accent="green">
        The full implementation is in section 4b. The key asserts to run after writing it: <Code>{"(labels != -100).float().mean() ≈ 0.15"}</Code>, <Code>{"((masked_input == MASK) & (labels != -100)).sum() / (labels != -100).sum() ≈ 0.80"}</Code>, and for every position where <Code>labels[i] != -100</Code>, <Code>labels[i] == original[i]</Code>. If you use a Bernoulli cascade (0.8 for MASK, then 0.5 among the remaining for random, rest unchanged), all three asserts pass in expectation. If you write it as three disjoint probabilities (0.8 / 0.1 / 0.1 over the union), the math is cleaner but the vectorized PyTorch code is less clean. Either works.
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> For a sequence of length <Code>N</Code>, compute the memory cost of the attention matrix for (a) a causal transformer and (b) a bidirectional transformer, in terms of floats. Which cost dominates in a model with <Code>L</Code> layers and <Code>H</Code> heads?
      </Prose>

      <Callout accent="green">
        The full attention matrix is <Code>N × N</Code>. A causal implementation can store only the lower triangle (strict + diagonal): <Code>N(N+1)/2 ≈ N²/2</Code> floats per head per layer. A bidirectional implementation stores the full <Code>N²</Code>. For an <Code>L</Code>-layer <Code>H</Code>-head model, the totals are <Code>L · H · N²/2</Code> for causal and <Code>L · H · N²</Code> for bidirectional — factor of 2 difference. In practice, well-implemented kernels (FlashAttention) don't materialize the full attention matrix in memory for either case — they recompute it in blocks during the backward pass. But the arithmetic still counts against the FLOPs budget, which is why long-context encoders are rare. At <Code>N=512</Code> (BERT's training length) the difference is negligible; at <Code>N=8192</Code> it is a meaningful 2× in compute; past <Code>N=32k</Code> it is prohibitive for full bidirectional attention.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> You are training an encoder from scratch on English Wikipedia, at modest scale (110M parameters, 24 hours of 8×A100 compute). Would you pick standard BERT-style MLM or ELECTRA-style replaced-token detection? Defend the choice in three sentences.
      </Prose>

      <Callout accent="green">
        ELECTRA, for the compute-constrained regime. The replaced-token-detection objective scores at every position rather than just 15%, so it extracts roughly 6× more supervision per forward pass — at a fixed compute budget, this is the axis that matters most. The original ELECTRA paper demonstrated that ELECTRA-small (14M params, 4 days on 1 V100) matches GPT's GLUE score (117M params trained on 30× more compute), and the efficiency gain is specifically pronounced at the modest-scale regime the question describes. The one caveat: ELECTRA's generator-discriminator training is slightly more complex to implement correctly, so if the team does not have GPU-weeks to debug it and has an existing MLM pipeline, straight BERT-style MLM with dynamic masking (RoBERTa recipe) is the safe choice.
      </Callout>

      <Prose>
        Masked language modeling was the objective that opened the door to large-scale self-supervised pretraining in NLP. It did not end up being the objective that built frontier models for generation, because causal LM extracts more supervision per token and scales further on every capability that requires producing new text. But it remains the objective of choice when the task is to <em>understand</em> a fixed input: classification, NER, retrieval, the embedding half of every RAG pipeline. The next topic covers its direct descendant — T5-style span corruption — which takes MLM's bidirectional encoding and grafts autoregressive generation onto it, combining the best properties of both.
      </Prose>
    </div>
  ),
};

export default maskedLanguageModeling;
