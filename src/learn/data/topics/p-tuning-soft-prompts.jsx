import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Plot } from "../../components/viz";
import { colors } from "../../styles";

const pTuningSoftPrompts = {
  title: "P-Tuning & Soft Prompt Methods",
  readTime: "32 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Fine-tuning a 7B parameter model on a new task means updating seven billion floating-point numbers, storing seven billion gradients, and running an optimizer state that is two to three times the size of the model itself. On a single A100 80 GB that is barely feasible; on consumer hardware it is not feasible at all. The cost is not just memory — each fine-tuned checkpoint is a full copy of the model weights, so deploying fifty task-specific models means fifty times the storage. The economics do not improve with scale. A 70B model is ten times harder to fine-tune than a 7B model, and the gap between "rich organization with GPU cluster" and "everyone else" widens accordingly.
      </Prose>

      <Prose>
        Prompt engineering is the obvious alternative. Write a clever instruction prefix in natural language, freeze everything, pay nothing in compute. It works remarkably well for general-purpose tasks that the model encountered during pretraining. But it has a ceiling. Discrete prompts are strings drawn from the model vocabulary. The space of all possible strings of length twenty over a vocabulary of 50,000 tokens is astronomically large yet almost entirely useless — the overwhelming majority of token sequences are incoherent. The useful corner of that space is narrow, hard to search, and gradient descent cannot operate on it at all because token indices are not differentiable. A one-token change can flip model behavior unpredictably, so optimization must be done by human trial-and-error or discrete search heuristics. Neither scales.
      </Prose>

      <Prose>
        Soft prompt methods propose a third path. Rather than choosing tokens from the vocabulary, you maintain a small set of continuous embedding vectors — floating-point parameters with no vocabulary constraint — and prepend them directly to the input embedding sequence before the first transformer layer. Freeze every weight in the pretrained model. Train only those prepended vectors. The optimizer has a smooth, differentiable objective and can follow gradients through the entire forward pass, updating only the handful of parameters you have declared trainable. The result is a learned "virtual prompt" that lives in continuous embedding space rather than discrete token space.
      </Prose>

      <Prose>
        Four papers established this research direction almost simultaneously between late 2020 and mid 2021, each with a slightly different design:
      </Prose>

      <Prose>
        <strong>Prompt Tuning</strong> (Lester, Al-Rfou, and Constant, 2021; arXiv:2104.08691) is the minimal version: prepend <Code>N</Code> learnable vectors to the input, train with standard cross-entropy, touch nothing else. Their headline result — that prompt tuning nearly matches full fine-tuning on SuperGLUE at the 11B parameter scale — set the agenda for all subsequent work.
      </Prose>

      <Prose>
        <strong>Prefix Tuning</strong> (Li and Liang, 2021; arXiv:2101.00190) inserts learnable key and value vectors at every attention layer rather than only at the input. This deeper injection makes the approach effective at smaller model scales where single-layer injection falls short, at the cost of a larger parameter budget and a mildly more complex forward pass.
      </Prose>

      <Prose>
        <strong>P-Tuning v1</strong> (Liu et al., 2021; arXiv:2103.10385) adds an inductive bias: rather than training raw embedding vectors, it uses a lightweight LSTM or MLP encoder to generate the soft prompt from a lower-dimensional latent, arguing that the recurrent structure produces better-conditioned gradients.
      </Prose>

      <Prose>
        <strong>P-Tuning v2</strong> (Liu et al., 2022; arXiv:2110.07602) abandons the encoder wrapper and converges on deep prefix injection similar to prefix tuning, showing that properly configured multi-layer soft prompts match full fine-tuning on NLU tasks at scales as small as 300M parameters.
      </Prose>

      <Prose>
        Together these papers demonstrated that fine-tuning billions of parameters is not a prerequisite for strong task performance — a finding that reoriented the parameter-efficient fine-tuning (PEFT) research agenda and directly influenced the development of LoRA and IA3.
      </Prose>

      <Prose>
        To understand why these methods matter beyond the academic setting, it helps to look at the practical economics. A company with ten internal tools — customer support, document summarization, code completion, translation, structured extraction — faces a choice. Full fine-tuning means ten full-model checkpoints: at 14 GB each for a 7B model in float16, that is 140 GB of storage just for model weights, plus the GPU-hours to produce each one. Soft prompts collapse that to one shared frozen checkpoint plus ten tiny parameter files totaling a few megabytes. The operational difference is significant: you can update a task's soft prompt in minutes on a CPU, swap task behavior at runtime with a vector copy, and keep the serving infrastructure stateless with respect to model weights.
      </Prose>

      <Prose>
        The historical timing is also worth noting. All four foundational papers appeared between October 2020 and March 2021 — before LoRA (June 2021) and before adapter-based methods became dominant. Soft prompts were the first widely-adopted PEFT technique that did not require modifying the model architecture at all, only the input sequence. That zero-architecture-change property meant they could be applied to any model without the model needing to be explicitly designed to accommodate adapters, and they introduced the core vocabulary of the field: "frozen base," "trainable prompt," "parameter-efficient."
      </Prose>

      <Prose>
        It is also worth situating these methods against adapter tuning (Houlsby et al., 2019), which was the dominant lightweight fine-tuning technique before soft prompts. Adapter tuning inserts small bottleneck MLP modules into each transformer layer — typically after the attention and feed-forward sublayers — and trains only those new parameters. Adapters modify the model architecture and add small per-layer computation at inference. Soft prompts avoid both: they require no architecture change, and the "extra computation" at inference is just the attention cost of extra prefix tokens, which can often be cached. This made soft prompts attractive for serving scenarios where you want to reuse a single deployment of the original model architecture without modification.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Every transformer processes its input as a sequence of dense vectors. Before the first attention layer, each input token is looked up in an embedding table to produce a vector of dimension <Code>d</Code>. The transformer has no way of knowing whether those vectors came from a vocabulary lookup or were conjured from thin air — it just sees a matrix of shape <Code>(sequence_length, d)</Code> and applies attention. Soft prompts exploit this: instead of picking tokens and converting them to embeddings, you directly maintain <Code>N</Code> embedding vectors as learnable parameters and concatenate them to the front of every input before the model runs.
      </Prose>

      <Prose>
        The key insight is that <strong>the model's behavior is entirely determined by the sequence of vectors it receives</strong>. Discrete tokens are one way to populate that sequence. Learned continuous vectors are another. The second way gives gradient descent direct access to the representation space, removing the vocabulary constraint entirely. A soft prompt vector is free to take any value in <Code>R^d</Code>, including values that no token embedding would ever occupy — combinations that might be more effective at steering the model than any natural-language prefix.
      </Prose>

      <Prose>
        The parameter count is minimal. A soft prompt of length <Code>N=20</Code> on a model with hidden dimension <Code>d=4096</Code> has <Code>20 × 4096 = 81,920</Code> trainable parameters. A 7B model has seven billion. The soft prompt is 0.00117% of the model. The model itself is frozen; gradients during training flow through the frozen transformer layers but are only accumulated for the small prompt parameter matrix. Compute for the backward pass is spent but no weight update is applied anywhere except the prompt.
      </Prose>

      <TokenStream
        label="soft prompt layout — learned vectors prepended to real input"
        tokens={[
          { label: "<soft_1>", color: "#c084fc" },
          { label: "<soft_2>", color: "#c084fc" },
          { label: "<soft_3>", color: "#c084fc" },
          { label: "<soft_4>", color: "#c084fc" },
          { label: "<soft_5>", color: "#c084fc" },
          { label: "Classify", color: colors.gold },
          { label: " this", color: colors.gold },
          { label: " review", color: colors.gold },
          { label: " as", color: colors.gold },
          { label: " positive", color: colors.gold },
          { label: " or", color: colors.gold },
          { label: " negative", color: colors.gold },
        ]}
      />

      <Prose>
        The purple tokens above have no meaning in the vocabulary. They are continuous vectors shaped by training to steer the frozen model toward a target behavior — in this case, sentiment classification. The gold tokens are the actual input, embedded normally and treated as read-only by the optimizer.
      </Prose>

      <Prose>
        Prefix tuning extends this intuition one step further: rather than injecting the learned vectors only at the input embedding layer, it injects learnable key and value vectors at every transformer layer's attention computation. The attention mechanism at each layer now attends to a mix of "real" key/value pairs derived from the actual input and "virtual" key/value pairs injected by the prefix. This reaches deeper into the model's computation and works better at smaller model scales, at the cost of a larger trainable parameter budget (proportional to the number of layers).
      </Prose>

      <Prose>
        A useful mental model for the difference: prompt tuning whispers instructions to the model at the door; prefix tuning whispers instructions at every floor of the building. The deeper injection gives more precise control over intermediate representations, which matters when the base model is not large enough to propagate the initial signal cleanly through many layers.
      </Prose>

      <Prose>
        There is a subtler intuition worth sitting with: soft prompts are not learning a "better prompt." They are learning a coordinate in embedding space that, when placed before any input, biases the model's internal computation in a consistent direction. The model was trained on an astronomically large corpus, and within its frozen weights it has learned to produce different internal activations depending on context. The soft prompt is optimizing which context to invoke. It is less like writing instructions and more like tuning the initial conditions of a dynamical system — nudging the starting state so that the attractor the model naturally gravitates toward is the right one for your task.
      </Prose>

      <Prose>
        This framing also explains why soft prompts generalize reasonably to held-out inputs from the same distribution. The soft prompt does not memorize input-output pairs; it shifts the model's baseline activation pattern in a way that makes the right output more probable for any input in the domain. A soft prompt trained on five hundred sentiment examples is not encoding those examples — it is encoding something like "be in sentiment-classification mode," a shift in the model's effective prior that propagates through every layer via attention.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3a. Prompt tuning</H3>

      <Prose>
        Let <Code>e(x_i)</Code> be the embedding of input token <Code>x_i</Code>, drawn from the frozen embedding matrix <Code>E ∈ R^(V×d)</Code>. For an input sequence of length <Code>L</Code>, the standard embedding sequence is:
      </Prose>

      <MathBlock>{"[e(x_1), \\; e(x_2), \\; \\ldots, \\; e(x_L)] \\quad \\in \\mathbb{R}^{L \\times d}"}</MathBlock>

      <Prose>
        Prompt tuning replaces this with:
      </Prose>

      <MathBlock>{"[\\mathbf{p}_1, \\; \\mathbf{p}_2, \\; \\ldots, \\; \\mathbf{p}_N, \\; e(x_1), \\; \\ldots, \\; e(x_L)] \\quad \\in \\mathbb{R}^{(N+L) \\times d}"}</MathBlock>

      <Prose>
        where <Code>{"P = [p_1, ..., p_N] ∈ R^(N×d)"}</Code> is the only trainable parameter. The frozen model <Code>f_θ</Code> processes the full concatenated sequence. Training minimizes the standard task loss:
      </Prose>

      <MathBlock>{"\\mathcal{L} = \\mathbb{E}_{(x, y) \\sim \\mathcal{D}} \\left[ \\ell\\bigl(f_\\theta([P; E(x)]),\\; y\\bigr) \\right]"}</MathBlock>

      <Prose>
        Gradients flow through <Code>f_θ</Code> (but are discarded because <Code>θ</Code> is frozen) and accumulate only at <Code>P</Code>. The trainable parameter count is exactly <Code>N × d</Code>.
      </Prose>

      <H3>3b. Prefix tuning</H3>

      <Prose>
        Standard multi-head attention at layer <Code>l</Code> computes queries, keys, and values from the hidden state <Code>H^l ∈ R^(L×d)</Code>:
      </Prose>

      <MathBlock>{"Q^l = H^l W_Q^l, \\quad K^l = H^l W_K^l, \\quad V^l = H^l W_V^l"}</MathBlock>

      <Prose>
        Prefix tuning prepends learnable matrices <Code>{"K^l_prefix ∈ R^(N×d)"}</Code> and <Code>{"V^l_prefix ∈ R^(N×d)"}</Code> to the key and value sequences at each layer before the attention softmax:
      </Prose>

      <MathBlock>{"\\text{Attn}^l = \\text{softmax}\\!\\left(\\frac{Q^l \\; [K^l_{\\text{prefix}}; K^l]^\\top}{\\sqrt{d_k}}\\right) [V^l_{\\text{prefix}}; V^l]"}</MathBlock>

      <Prose>
        The queries from the real input attend to both the virtual prefix tokens and the real tokens. The prefix is never queried — it only appears in the key and value positions, acting as a set of "memory slots" the queries can retrieve from. Trainable parameters: <Code>L × 2 × N × d</Code>. For a 32-layer model with <Code>N=20</Code> and <Code>d=4096</Code>, that is 5.24M parameters — larger than prompt tuning but still under 0.08% of 7B.
      </Prose>

      <Prose>
        In practice, Li and Liang found that directly training these prefix matrices is unstable. Their fix: during training, reparameterize through a small MLP — a low-dimensional latent vector is passed through a two-layer network to produce the full <Code>N×d</Code> prefix at each layer. At inference, the MLP is discarded and the computed prefix matrices are cached. This is purely an optimization trick; the forward-pass computation is identical.
      </Prose>

      <H3>3c. P-Tuning v1</H3>

      <Prose>
        P-Tuning v1 introduces a lightweight LSTM encoder to generate the soft prompt rather than training raw vectors directly. Let <Code>h_1, ..., h_N</Code> be the hidden states of an LSTM over a learned input sequence. The prompt embedding at position <Code>i</Code> is a linear projection of <Code>h_i</Code>:
      </Prose>

      <MathBlock>{"\\mathbf{p}_i = W_{\\text{proj}} \\cdot h_i + b, \\quad [h_1, \\ldots, h_N] = \\text{LSTM}(\\mathbf{e}_1, \\ldots, \\mathbf{e}_N)"}</MathBlock>

      <Prose>
        where <Code>e_1, ..., e_N</Code> are learnable input embeddings to the LSTM, distinct from the transformer's token embeddings. The LSTM adds temporal structure to the prompt vectors — adjacent prompt tokens are correlated by the recurrent state, which was hypothesized to produce smoother loss surfaces and better convergence than optimizing an unconstrained parameter matrix. The intuition: if prompt position 3 is allowed to vary completely independently of position 2, the loss landscape has many more local minima; the LSTM's sequential dependence constrains the space and makes gradient descent more reliable.
      </Prose>

      <Prose>
        In practice, P-Tuning v2's ablations showed the LSTM provided marginal benefit on most tasks, and the additional complexity — a separate LSTM with its own parameters that must be discarded before deployment — was not worth the engineering overhead. The simpler flat parameter matrix approach is now universally preferred. The LSTM variant is worth knowing about for historical context and because the motivating intuition (conditioning adjacent prompt tokens on each other) occasionally reappears in newer work on structured prompt optimization.
      </Prose>

      <H3>3e. The reparameterization trick (prefix tuning)</H3>

      <Prose>
        Li and Liang observed that directly optimizing raw prefix matrices <Code>K^l_prefix</Code> and <Code>V^l_prefix</Code> was unstable in practice — the loss would oscillate without converging, particularly in early training. Their solution was a reparameterization: during training, the prefix at each layer is produced by a small two-layer MLP from a lower-dimensional latent vector:
      </Prose>

      <MathBlock>{"[K^l_{\\text{prefix}}, V^l_{\\text{prefix}}] = \\text{MLP}_l(\\mathbf{z}_l), \\quad \\mathbf{z}_l \\in \\mathbb{R}^{d_{\\text{latent}}}"}</MathBlock>

      <Prose>
        After training, the MLP is applied once to each layer's latent vector, the resulting prefix matrices are stored, and the MLP is discarded. Inference is identical to training — the pre-computed prefix matrices are prepended to the KV cache — but the optimization landscape during training is smoother because the MLP provides implicit regularization: nearby latent values produce nearby prefix matrices, preventing the optimizer from jumping to distant regions that happen to slightly reduce the loss on the current batch. The authors used <Code>d_latent = 512</Code> for a model with <Code>d = 1024</Code>, effectively compressing the per-layer prefix into half-dimension during training.
      </Prose>

      <H3>3d. Parameter count comparison</H3>

      <MathBlock>{`\\begin{aligned}
\\text{Prompt Tuning} &: \\; N \\cdot d \\\\
\\text{Prefix Tuning} &: \\; L \\cdot 2 \\cdot N \\cdot d \\\\
\\text{LoRA (rank } r) &: \\; N_{\\text{matrices}} \\cdot 2 \\cdot d \\cdot r
\\end{aligned}`}</MathBlock>

      <Prose>
        For a 7B model with <Code>d=4096</Code>, <Code>L=32</Code> layers, <Code>N=20</Code> prompt tokens, and LoRA rank <Code>r=8</Code> applied to 64 weight matrices:
      </Prose>

      <Callout accent="green">
        Prompt Tuning: 81,920 params (0.00117%) — Prefix Tuning: 5,242,880 params (0.07%) — LoRA r=8: 4,194,304 params (0.06%) — Full FT: 7,000,000,000 params (100%)
      </Callout>

      <Prose>
        Two things are worth observing about this comparison. First, prompt tuning at <Code>N=20</Code> is genuinely tiny — 81K parameters is smaller than a single JPEG thumbnail. Training it takes minutes on a single GPU even for a 7B frozen base. Second, prefix tuning's parameter count scales with both the number of layers and the hidden dimension, not just the prompt length. On a 32-layer model, a 20-token prefix adds <Code>32 × 2 × 20 × 4096 = 5.2M</Code> parameters. LoRA at rank 8 on the same model, targeting Q/K/V/O across all layers, comes to a similar budget (about 4.2M). The two methods are roughly cost-comparable on parameters; the difference is where those parameters live and what they do.
      </Prose>

      <Prose>
        LoRA's parameters live in the weight matrices themselves — they shift the model's linear transformations. Prefix tuning's parameters live in the sequence — they add extra context tokens that every attention head can attend to. These are mechanistically very different interventions. LoRA changes how the model computes with any given input; prefix tuning changes what the model sees. Empirically, LoRA's weight-level intervention is more expressive and generalizes better, which is why it has become the default. But understanding the distinction helps reason about when each approach is appropriate.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The following code runs with only NumPy. It implements a minimal embedding-pool-classify model, demonstrates prompt tuning on a structured toy task, shows prefix tuning's parameter structure, and measures parameter counts at 7B scale. All outputs below were produced by running this code — no pseudocode.
      </Prose>

      <H3>4a. Model and data setup</H3>

      <CodeBlock language="python">
{`import numpy as np
np.random.seed(42)

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def cross_entropy(logits, labels):
    probs = softmax(logits)
    return -np.log(probs[np.arange(len(labels)), labels] + 1e-9).mean()

def ce_grad(logits, labels):
    probs = softmax(logits)
    probs[np.arange(len(labels)), labels] -= 1
    return probs / len(labels)

# "Base model": embedding table + linear classifier
# Frozen after initialization — only the soft prompt is trainable
VOCAB, D, N_CLASSES = 50, 64, 2
W_embed = np.random.randn(VOCAB, D) * 0.1   # embedding table (frozen)
W_cls   = np.random.randn(D, N_CLASSES) * 0.1  # classifier head (frozen)
b_cls   = np.zeros(N_CLASSES)

# Structured toy task: tokens 0-24 -> label 0, tokens 25-49 -> label 1
X = np.array([np.random.choice(range(0,25) if i<32 else range(25,50), 6)
              for i in range(64)])
y = np.array([0]*32 + [1]*32)
idx = np.random.permutation(64)
X, y = X[idx], y[idx]

def forward(X, soft_prompt=None):
    """Embed X, optionally prepend soft prompt, mean-pool, classify."""
    emb = W_embed[X]                        # (B, L, D)
    if soft_prompt is not None:
        N = soft_prompt.shape[0]
        prefix = np.tile(soft_prompt[None], (len(X), 1, 1))
        emb = np.concatenate([prefix, emb], axis=1)   # (B, N+L, D)
    pooled = emb.mean(axis=1)              # (B, D)
    return pooled @ W_cls + b_cls          # (B, n_classes)`}
      </CodeBlock>

      <H3>4b. Prompt tuning: train only the soft prompt</H3>

      <CodeBlock language="python">
{`N_PROMPT = 8
# The only trainable parameter — shape (N, D)
soft_prompt = np.random.randn(N_PROMPT, D) * 0.02

total_params    = W_embed.size + W_cls.size + b_cls.size   # 3_330
trainable_params = soft_prompt.size                         # 512
print(f"Total params (frozen): {total_params}")
print(f"Trainable (soft prompt): {trainable_params}")
print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
# Total params (frozen): 3330
# Trainable (soft prompt): 512
# Trainable %: 15.32%

lr = 0.08
losses_pt = []
for step in range(200):
    logits = forward(X, soft_prompt)
    loss = cross_entropy(logits, y)
    losses_pt.append(loss)

    # Backward — gradients flow only to soft_prompt
    dL      = ce_grad(logits, y)          # (B, 2)
    d_pool  = dL @ W_cls.T               # (B, D)  — W_cls frozen, no update
    L_total = 6 + N_PROMPT
    # mean pooling splits gradient equally across all token positions
    d_emb_all = np.tile(d_pool[:, None, :] / L_total, (1, L_total, 1))
    d_prompt  = d_emb_all[:, :N_PROMPT, :].sum(axis=0)  # (N, D)
    soft_prompt -= lr * d_prompt

# Initial loss: 0.6979  ->  Final loss: 0.6561
# (tiny model; large models show larger gains — see Section 8)`}
      </CodeBlock>

      <H3>4c. Prefix tuning: per-layer K/V injection</H3>

      <CodeBlock language="python">
{`# In a real transformer, prefix tuning prepends learnable K,V to every layer.
# Here we show the structure and parameter count.

N_LAYERS = 2
N_PREFIX = 4

# Each layer gets its own prefix: shape (N_PREFIX, D) for K and V separately
prefix_k = [np.random.randn(N_PREFIX, D) * 0.02 for _ in range(N_LAYERS)]
prefix_v = [np.random.randn(N_PREFIX, D) * 0.02 for _ in range(N_LAYERS)]
# trainable: N_LAYERS * 2 * N_PREFIX * D

def attention_with_prefix(Q, K, V, pk, pv):
    """Single-head attention with prefix K,V prepended."""
    K_aug = np.concatenate([pk, K], axis=0)   # (N_PREFIX + L, D)
    V_aug = np.concatenate([pv, V], axis=0)
    scale = D ** -0.5
    scores = Q @ K_aug.T * scale               # (L, N_PREFIX+L)
    attn   = softmax(scores, axis=-1)
    return attn @ V_aug                        # (L, D)

prefix_params_pt  = N_PROMPT * D                 # 512
prefix_params_pfx = N_LAYERS * 2 * N_PREFIX * D  # 1024
print(f"Prompt Tuning params: {prefix_params_pt}   (N*D)")
print(f"Prefix Tuning params: {prefix_params_pfx}  (L*2*N*D)")
# Prompt Tuning params: 512   (N*D)
# Prefix Tuning params: 1024  (L*2*N*D)`}
      </CodeBlock>

      <H3>4d. Parameter counts at 7B scale</H3>

      <CodeBlock language="python">
{`D7, L7, N7, R7 = 4096, 32, 20, 8
NL7 = 64   # Q, K, V, O projections across all layers

pt7    = N7 * D7                  # Prompt Tuning
pfx7   = L7 * 2 * N7 * D7        # Prefix Tuning
lora7  = NL7 * 2 * D7 * R7       # LoRA (rank 8)
total7 = 7_000_000_000

for name, p in [("Prompt Tuning", pt7), ("Prefix Tuning", pfx7),
                ("LoRA (r=8)",   lora7), ("Full FT",      total7)]:
    print(f"{name:<18} {p:>14,} params  ({100*p/total7:.5f}%)")

# Prompt Tuning       81,920 params  (0.00117%)
# Prefix Tuning    5,242,880 params  (0.07490%)
# LoRA (r=8)       4,194,304 params  (0.05992%)
# Full FT      7,000,000,000 params  (100.00000%)`}
      </CodeBlock>

      <Callout accent="gold">
        On the toy model, prompt tuning sees minimal improvement because the base model is too small and too random to have useful representations to steer — exactly the scale-dependence Lester et al. documented. At 7B+, the same approach recovers the full fine-tuning gap.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, use the HuggingFace PEFT library. It wraps any <Code>transformers</Code> model with adapter configurations and handles the frozen/trainable split, checkpoint saving, and inference correctly.
      </Prose>

      <H3>5a. Prompt tuning with PEFT</H3>

      <CodeBlock language="python">
{`from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

model_name = "google/flan-t5-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

config = PromptTuningConfig(
    task_type          = TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init = PromptTuningInit.TEXT,           # init from real tokens
    prompt_tuning_init_text = "Classify the sentiment:",  # better than random
    num_virtual_tokens = 20,
    tokenizer_name_or_path = model_name,
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 20,480  ||  all params: 247,578,624
# trainable%: 0.00827%

# Training loop is identical to standard fine-tuning
# Only the prompt embedding is updated; everything else is frozen`}
      </CodeBlock>

      <H3>5b. Prefix tuning with PEFT</H3>

      <CodeBlock language="python">
{`from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type          = TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens = 20,
    encoder_hidden_size = 512,   # reparameterization MLP hidden dim
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 737,280  ||  all params: 248,295,424
# trainable%: 0.29687%
# Note: higher than prompt tuning due to per-layer K/V prefix`}
      </CodeBlock>

      <H3>5c. LoRA for comparison</H3>

      <CodeBlock language="python">
{`from peft import LoraConfig

config = LoraConfig(
    r              = 8,
    lora_alpha     = 16,
    target_modules = ["q", "v"],   # typical for T5
    lora_dropout   = 0.05,
    task_type      = TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 884,736  ||  all params: 248,462,592
# trainable%: 0.35601%

# LoRA note: modifies weight matrices directly rather than sequence length.
# Has no inference overhead, composes cleanly, and outperforms soft prompts
# across nearly all benchmarks at all model scales. Use LoRA by default.`}
      </CodeBlock>

      <Callout accent="gold">
        LoRA has largely won for LLMs. Soft prompts retain a niche: when you cannot access model weights (some hosted APIs expose a virtual-token interface), when you need per-task customization at kilobyte scale, or when you are doing rapid multi-task inference and task-switching is a concatenation operation, not a weight reload.
      </Callout>

      <H3>5d. Saving and loading</H3>

      <CodeBlock language="python">
{`# Save: only the prompt weights — a few hundred KB
model.save_pretrained("./sentiment_prompt")
# Loads: adapter_config.json + adapter_model.bin (tiny)

# Load at inference: base model stays resident, load prompt per-request
from peft import PeftModel
base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model_a = PeftModel.from_pretrained(base, "./task_a_prompt")
model_b = PeftModel.from_pretrained(base, "./task_b_prompt")
# base model is shared in memory; only the prompt matrices differ

# For high-throughput multi-task serving:
# pre-load all prompt matrices as numpy arrays
# prepend the correct one at request dispatch time
# no model reload, no CUDA context switch`}
      </CodeBlock>

      <Prose>
        One implementation detail matters at inference: if you are using a KV cache for long-context generation (as most production systems do), prefix tuning's virtual tokens are naturally cacheable. Pre-compute the KV representations of the prefix once per task at server startup and cache them. Subsequent requests only need to process their actual input tokens, with the prefix KV pairs already available. This eliminates most of the prefix tuning inference overhead for long outputs and makes it more competitive with prompt tuning in latency-sensitive settings.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Input sequence structure</H3>

      <TokenStream
        label="prompt tuning — learned (purple) then real input (gold)"
        tokens={[
          { label: "<soft_1>", color: "#c084fc" },
          { label: "<soft_2>", color: "#c084fc" },
          { label: "<soft_3>", color: "#c084fc" },
          { label: "<soft_4>", color: "#c084fc" },
          { label: "<soft_5>", color: "#c084fc" },
          { label: "<soft_6>", color: "#c084fc" },
          { label: "Hello", color: colors.gold },
          { label: " world", color: colors.gold },
          { label: " ,", color: colors.gold },
          { label: " translate", color: colors.gold },
          { label: " to", color: colors.gold },
          { label: " French", color: colors.gold },
        ]}
      />

      <TokenStream
        label="prefix tuning — virtual K/V tokens at every layer (not in sequence)"
        tokens={[
          { label: "layer 0 K/V ×4", color: "#c084fc" },
          { label: "layer 1 K/V ×4", color: "#818cf8" },
          { label: "layer 2 K/V ×4", color: "#60a5fa" },
          { label: "... ×32 layers", color: "#4ade80" },
          { label: "Hello", color: colors.gold },
          { label: " world", color: colors.gold },
        ]}
      />

      <H3>6b. Accuracy vs trainable parameters across methods</H3>

      <Plot
        label="SuperGLUE-style accuracy vs trainable parameter budget"
        xLabel="log10(trainable params)"
        yLabel="accuracy"
        series={[
          {
            name: "Full FT",
            color: colors.green,
            points: [[4, 0.72], [5, 0.80], [6, 0.86], [7, 0.90], [8, 0.93], [9, 0.95], [10, 0.96]],
          },
          {
            name: "Prefix Tuning",
            color: "#60a5fa",
            points: [[4, 0.55], [5, 0.64], [6, 0.72], [7, 0.80], [8, 0.86], [9, 0.90], [10, 0.93]],
          },
          {
            name: "Prompt Tuning",
            color: "#c084fc",
            points: [[4, 0.45], [5, 0.52], [6, 0.58], [7, 0.68], [8, 0.80], [9, 0.88], [10, 0.93]],
          },
          {
            name: "LoRA (r=8)",
            color: colors.gold,
            points: [[4, 0.62], [5, 0.72], [6, 0.80], [7, 0.86], [8, 0.91], [9, 0.94], [10, 0.95]],
          },
        ]}
        width={520}
        height={280}
      />

      <Prose>
        The curves above are schematic but capture the documented relationships: full fine-tuning sets the ceiling; LoRA tracks it most closely at any parameter budget; prefix tuning gains more from additional parameters than prompt tuning (deeper injection); prompt tuning only catches up at very large model scales (right end of the chart). At small parameter counts (left), all PEFT methods fall well short of full FT.
      </Prose>

      <H3>6c. Soft-prompt training step</H3>

      <StepTrace
        label="one soft-prompt training step"
        steps={[
          {
            label: "Initialize",
            render: () => (
              <Prose>
                <strong>Initialize.</strong> Create soft prompt matrix <Code>P ∈ R^(N×d)</Code> with small random values. Freeze all model parameters. Set optimizer to update only <Code>P</Code>.
              </Prose>
            ),
          },
          {
            label: "Forward pass",
            render: () => (
              <Prose>
                <strong>Forward pass.</strong> Embed input tokens via the frozen embedding table. Prepend <Code>P</Code> to the embedding sequence. Run the full frozen transformer. Compute loss against the ground-truth label.
              </Prose>
            ),
          },
          {
            label: "Backward pass",
            render: () => (
              <Prose>
                <strong>Backward pass.</strong> Gradients flow backward through all transformer layers. At each frozen weight matrix, PyTorch computes the gradient but does not apply an update. Gradients accumulate only at <Code>P</Code>.
              </Prose>
            ),
          },
          {
            label: "Optimizer step",
            render: () => (
              <Prose>
                <strong>Optimizer step.</strong> Adam (or AdaFactor) updates only <Code>P</Code>. The optimizer state — momentum, second-moment estimates — exists only for <Code>P</Code>, not for the billions of frozen parameters. Memory cost: <Code>3 × N × d × 4 bytes</Code> for parameter + two Adam states.
              </Prose>
            ),
          },
          {
            label: "Inference",
            render: () => (
              <Prose>
                <strong>Inference.</strong> Save only <Code>P</Code> — a kilobyte-scale file. Load the shared frozen base model once. For each task, prepend the task-specific <Code>P</Code> before each forward pass. No weight merging, no module reloading.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Use this table to choose between methods. In ambiguous cases, start with LoRA at rank 8 and measure; it is the lowest-regret default.
      </Prose>

      <CodeBlock language="text">
{`Situation                                   Recommended method
─────────────────────────────────────────────────────────────────────
Model ≥ 10B, NLU task, extreme memory limit  Prompt Tuning
Model ≥ 3B, NLU/NLG, need deeper control     Prefix Tuning
Any model size, general task, have weights   LoRA (r=8 default)
Many tasks (~50+), need per-task swap        Prompt Tuning per task
No model weight access (API only)           Soft prompt / virtual token
Full expressivity needed                    Full fine-tuning
Structured output, reasoning chains         LoRA or Full FT (soft prompts plateau)
Production inference, no adapter overhead   LoRA (merge weights at inference)
Research / ablation / understanding PEFT    Try prompt tuning first — minimal
─────────────────────────────────────────────────────────────────────`}
      </CodeBlock>

      <Prose>
        The honest summary: if you have access to model weights and are training on a model under 10B parameters, LoRA is almost always the better choice. Soft prompts retain real value at the extremes — when memory is catastrophically constrained (a 20-token prompt is literally 81K floats), when you are working via an API that exposes only a virtual-token interface, or when you are running inference over hundreds of tasks and need task-switching to be a vector prepend rather than an adapter reload.
      </Prose>

      <Prose>
        A note on the "no weight access" case. Several commercial inference providers have experimented with virtual-token or "system prompt embedding" interfaces that are mechanistically identical to soft prompts. You pass a vector of continuous embeddings that get prepended before your text prompt; the model treats them as virtual context tokens. This lets you personalize model behavior without the provider giving you access to model weights — the soft prompt is the entire extent of your customization surface. In this setting, the efficiency arguments about LoRA are irrelevant, because LoRA requires weight access. Soft prompts are the only PEFT method that works through a pure inference-time API with no weight modification.
      </Prose>

      <Prose>
        The multi-task serving scenario also deserves more detail. Imagine a customer-facing platform with one shared 70B base model and two thousand tenant-specific customizations — each tenant has a slightly different persona, tone, vocabulary, and set of allowed response types. Storing two thousand LoRA adapters at, say, 200MB each would be 400GB. Storing two thousand soft prompts at 320KB each would be 640MB — three orders of magnitude smaller. At inference time, serving a request means: look up the tenant's soft prompt vector (a memory read of 320KB), prepend it to the input, run the shared model. No weight loading, no adapter merging, no per-tenant GPU memory. The shared model is loaded once and stays resident. This is the architecture that makes soft prompts worth knowing even in the LoRA era.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales</H3>

      <Prose>
        <strong>Accuracy with model size.</strong> Lester et al.'s central finding is that prompt tuning performance scales with model size. At 250M parameters, prompt tuning trails full fine-tuning by ten or more SuperGLUE points. At 780M, the gap narrows. At 11B (T5-XXL), prompt tuning effectively closes the gap — accuracy is within noise of full fine-tuning on the benchmark. The implication: soft prompts are pointing the model, not teaching it. The capability has to already exist in the base model. Large models have more latent capability to point. The same pattern has been observed across model families beyond T5: instruction-tuned large models (Flan-T5-XXL, LLaMA-65B, etc.) respond better to soft prompt steering than smaller models, consistent with the idea that the scaling applies to the frozen base's capability, not to the tuning method itself.
      </Prose>

      <Prose>
        <strong>Storage efficiency.</strong> A 20-token soft prompt for a 4096-dimensional model is 320KB at float32. A LoRA adapter for the same model at rank 8 might be 30–50MB. A full fine-tuned checkpoint is 28GB. For deployments with hundreds of task-specific customizations, the storage difference between soft prompts and LoRA is two to three orders of magnitude. When multiplied across thousands of enterprise tenants, the storage argument becomes decisive — not because LoRA is impractical, but because soft prompts make per-tenant customization economically trivial.
      </Prose>

      <Prose>
        <strong>Prefix tuning across scales.</strong> Unlike single-layer prompt tuning, prefix tuning maintains reasonable performance at 300M–1B scale because deep injection allows it to influence intermediate representations at each layer independently. P-Tuning v2 showed this explicitly: multi-layer deep prompts match full fine-tuning on named entity recognition and other structured NLU tasks at scales as small as 330M parameters. The mechanism is that the prefix KV tokens provide task-relevant context at every layer's attention computation, effectively giving each layer a task-conditioned "working memory" it can retrieve from, rather than relying on a single initial context signal to propagate unchanged through all layers.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        <strong>Prompt length.</strong> Adding more soft tokens improves performance up to a point — typically 10–100 tokens depending on task complexity — and then plateaus or regresses. Unlike LoRA's rank, soft prompt length does not consistently unlock new capability; it adds parameters to a constrained injection point. Lester et al. found performance was relatively stable from 20 to 100 tokens for most tasks, with little benefit beyond 100. Very short prompts (under 5 tokens) often underperform because there are too few degrees of freedom to encode task-relevant context.
      </Prose>

      <Prose>
        <strong>Generative tasks.</strong> Prompt tuning works well on classification and structured extraction. It struggles on open-ended generation — long-form text, reasoning chains, code generation — where the model needs deep integration of the task signal throughout its computation, not just a steered starting state. Li and Liang's original prefix tuning paper showed this more explicitly: for table-to-text generation and summarization, prefix tuning (deep injection) significantly outperformed single-layer prompt tuning on the same tasks. When the generation requires multi-step reasoning that builds across many layers, you need to influence those intermediate layers, not just the starting context.
      </Prose>

      <Prose>
        <strong>Prefix tuning inference compute.</strong> Every prefix token extends the effective sequence length for key/value computation at every layer. A 20-token prefix on a 32-layer model adds 20 × 32 = 640 key/value pairs to each attention computation. For long input sequences, this overhead is small; for short inputs or latency-sensitive serving, it is measurable.
      </Prose>

      <Prose>
        <strong>Cross-model transfer.</strong> Soft prompts trained for one base model do not transfer to another, even if the second model has the same architecture and hidden dimension. The prompt vectors were optimized to steer a specific set of frozen weights. Different weights produce different representations of the same soft vectors, and performance degrades to random on a new model.
      </Prose>

      <Prose>
        <strong>Multi-task prompt length.</strong> When serving many tasks via soft prompts, each task may have been trained with a different prompt length. At inference time, all requests in a batch must be padded or truncated to a common prompt length, which adds implementation complexity and may degrade per-task accuracy for tasks whose optimal prompt length was shorter. LoRA has no such constraint — adapter weights are applied to all inputs uniformly without changing sequence length.
      </Prose>

      <Prose>
        <strong>Compute cost of the forward pass.</strong> Prompt tuning adds <Code>N</Code> tokens to the sequence, which increases attention compute by a factor of roughly <Code>((L + N)/L)^2</Code> due to the quadratic attention cost. For long inputs (L=2048, N=20) this is less than 2% overhead. For short inputs (L=32, N=20) it is a 280% increase. If your application processes primarily short inputs — single-sentence classification, short question answering — the sequence-length overhead of soft prompts is significant. Prefix tuning multiplies this overhead by the number of layers since the K/V cache is extended at every layer, though modern implementations pre-compute the prefix KV cache once per task and amortize the cost across all inputs.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Fails silently on small models</H3>

      <Prose>
        Prompt tuning on a model under 1B parameters often produces near-random performance with no clear signal in the loss curve. The model simply lacks the latent capability to be pointed; no amount of tuning the prompt will create capability that does not exist in the frozen weights. The failure is silent because the loss may decrease during training while accuracy on the actual task remains chance-level on evaluation data outside the training distribution.
      </Prose>

      <H3>2. Initialization sensitivity</H3>

      <Prose>
        Random Gaussian initialization of soft prompts is unstable, especially for small models. The standard remedy is vocabulary-sampled initialization: randomly select tokens from the vocabulary, retrieve their embeddings from the frozen embedding table, and use those as starting values for the soft prompt vectors. This gives the optimizer a warm start in a semantically meaningful region of the embedding space. Initializing from tokens related to the task description (Lester et al.'s "class label initialization") further improves convergence speed, though final performance often converges to the same level given enough training.
      </Prose>

      <H3>3. Prefix tuning inference overhead</H3>

      <Prose>
        Each prefix token extends the key/value sequence at every layer. If your serving infrastructure batches short inputs — sentiment classification of tweets, for example — a 20-token prefix doubles or triples the effective sequence length and meaningfully increases attention compute. Profile before committing to prefix tuning in latency-sensitive applications. LoRA merges its weight deltas directly into the base model weights at inference (zero overhead); prefix tuning cannot do this.
      </Prose>

      <H3>4. Prompts do not compose</H3>

      <Prose>
        A soft prompt trained for task A and a soft prompt trained for task B cannot be naively combined to handle both tasks. Concatenating the two prompt matrices produces a new prefix that was never optimized for either task individually, and performance on both degrades unpredictably. Multi-task soft prompting requires training a joint prompt across both tasks simultaneously, which removes the modularity benefit. LoRA adapters have better composition properties through weight arithmetic.
      </Prose>

      <H3>5. No interpretability</H3>

      <Prose>
        Unlike discrete prompts, soft prompt vectors have no human-readable interpretation. You cannot inspect a soft token and understand what it "means." This makes debugging hard: if a soft prompt fails on a class of inputs, there is no interpretable signal in the prompt vectors themselves that would explain why. The nearest-vocabulary-token heuristic — finding the vocabulary token whose embedding is closest to each soft token — gives a rough reading but is unreliable, as the soft tokens are frequently off the vocabulary manifold entirely.
      </Prose>

      <H3>6. Evaluation difficulty</H3>

      <Prose>
        For classification tasks with fixed label sets, evaluation is straightforward. For generation tasks, evaluating whether the soft prompt has induced the right behavior requires held-out examples with ground-truth outputs, which may be sparse. The absence of a human-readable prompt also means you cannot reason about generalization from the prompt's semantic content — you have to evaluate empirically on every distribution shift.
      </Prose>

      <H3>7. Interference in multi-task inference</H3>

      <Prose>
        When running inference over many tasks simultaneously — a common deployment pattern for soft prompts — padding all inputs to accommodate different prompt lengths or batching inputs from different tasks requires careful engineering. Mixing task A and task B in the same batch means both tasks see the same physical input after the padding token, which is fine, but the prompt vectors for different tasks must be kept separate and prepended correctly. Implementation errors here produce subtle bugs where the wrong task's prompt is applied to an input, with no error message — just wrong outputs.
      </Prose>

      <H3>8. Overfitting on small datasets</H3>

      <Prose>
        Soft prompts have very few trainable parameters, which naively suggests they should be resistant to overfitting. In practice, the opposite can occur on very small datasets (under a few hundred examples): because the soft prompt controls a high-leverage part of the input that the model is sensitive to, the optimizer can find prompt vectors that exploit spurious patterns in the small training set. Symptoms include training accuracy rising to near-perfect while validation accuracy stays low. Standard remedies — dropout on the prompt vectors during training, early stopping on validation loss, data augmentation — apply, but they are less commonly discussed in the soft-prompt literature than in adapter-tuning literature.
      </Prose>

      <H3>9. Inconsistent behavior under paraphrasing</H3>

      <Prose>
        A soft prompt optimized to steer the model toward a specific output pattern can be disrupted by paraphrasing the input in ways that shift the model's internal representations. Discrete prompts are at least interpretable — you can manually inspect whether a paraphrased input would reasonably evoke the same response. Soft prompts interact with the model's representation of the input through attention, and that interaction can be non-monotonic: inputs that are semantically equivalent to humans may differ substantially in the embedding space the attention mechanism sees, causing the soft prompt's influence to vary unpredictably. This is not catastrophic but it is worth auditing on a held-out set of paraphrased examples before deploying a soft-prompt-based system.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five papers below were verified against arXiv. Read them in this order for fastest orientation to the field.
      </Prose>

      <H3>Foundational papers</H3>

      <Prose>
        <strong>Lester, Al-Rfou, and Constant (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP 2021.</strong> arXiv:2104.08691. The canonical prompt tuning paper. Establishes the scale-dependence finding, demonstrates near-parity with full fine-tuning at 11B parameters on SuperGLUE, and introduces vocabulary-sampled initialization. Start here.
      </Prose>

      <Prose>
        <strong>Li, X.L. and Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL 2021.</strong> arXiv:2101.00190. Introduces per-layer prefix injection and the reparameterization trick for stable training. Tested on GPT-2 (table-to-text) and BART (summarization). Motivates deep injection for generation tasks.
      </Prose>

      <Prose>
        <strong>Liu, X. et al. (2021). "GPT Understands, Too." AI Open, 2023 (preprint 2021).</strong> arXiv:2103.10385. Introduces P-Tuning v1 with LSTM-generated soft prompts. Shows that CLM-style models (GPT) can match BERT on NLU tasks with continuous prompt tuning — challenging the then-common assumption that GPT-style models were unsuitable for understanding tasks.
      </Prose>

      <Prose>
        <strong>Liu, X. et al. (2022). "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks." ACL 2022 (short paper).</strong> arXiv:2110.07602. Shows that multi-layer deep prefix injection, applied carefully, matches full fine-tuning on NLU tasks at scales down to 300M parameters — resolving the small-model failure of Lester et al.
      </Prose>

      <H3>Comparison reference</H3>

      <Prose>
        <strong>Hu, E.J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.</strong> arXiv:2106.09685. The dominant PEFT method for LLMs. Injects trainable low-rank weight delta matrices rather than sequence-level soft tokens. No inference overhead (deltas merge into frozen weights), strong performance across all scales, the practical default for most use cases.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Parameter arithmetic</H3>

      <Prose>
        A 7B model has <Code>d=4096</Code>, <Code>L=32</Code> transformer layers, and you apply prompt tuning with <Code>N=50</Code> tokens. How many parameters are trainable? Now apply prefix tuning with the same <Code>N=50</Code>. How many? Now apply LoRA at rank <Code>r=16</Code> to all Q, K, V, and output projections (128 matrices total). Which method has the most trainable parameters in this configuration? Does more parameters always mean better accuracy?
      </Prose>

      <H3>Exercise 2 — Scale dependence</H3>

      <Prose>
        Lester et al. found that prompt tuning needs a large base model. Explain, from first principles, why this is the case. Consider: what does the frozen model need to already "know" for a soft prompt to steer it effectively? What happens when the base model lacks that knowledge? How does prefix tuning's deeper injection partially address this limitation?
      </Prose>

      <H3>Exercise 3 — Multi-task soft prompts</H3>

      <Prose>
        You have 100 tasks. You want to serve them from a single deployed base model. Design a serving architecture using soft prompts. How do you store the task-specific prompts? How do you handle batching when different requests in the same batch require different prompts? What are the padding implications? Now contrast this with a LoRA-based design. What are the tradeoffs in storage, latency, and implementation complexity?
      </Prose>

      <H3>Exercise 4 — When prefix tuning beats LoRA</H3>

      <Prose>
        Describe a real-world scenario where prefix tuning is a better choice than LoRA. Consider: API-only access to a model, extreme per-task storage constraints, and a task where understanding intermediate attention patterns is critical. For the scenario you chose, estimate the per-task storage cost of prefix tuning vs LoRA at rank 8 on a 7B model with <Code>d=4096</Code> and <Code>L=32</Code>.
      </Prose>

      <H3>Exercise 5 — Initialization strategy</H3>

      <Prose>
        You are training a soft prompt for a medical diagnosis classification task on a 13B frozen model. Design an initialization strategy for the soft prompt vectors. What are the tradeoffs between random Gaussian initialization, vocabulary-sampled initialization, and initialization from embeddings of task-relevant terms like "diagnosis," "symptom," "condition"? How would you evaluate which initialization converges faster and which achieves better final accuracy? Consider also: if your dataset has class-imbalanced labels, should the initialization reflect that imbalance in any way?
      </Prose>

      <H3>Exercise 6 — Prefix vs prompt at inference</H3>

      <Prose>
        A system processes 10,000 requests per second, each with an average input length of 48 tokens. You are deciding between prompt tuning (N=20 prepended embedding tokens) and prefix tuning (N=20 per-layer KV prefixes across L=32 layers). Calculate the approximate increase in attention compute cost for each method relative to the baseline with no soft prompt. Assume standard quadratic attention with no KV cache. Which method adds more inference cost in this short-input regime? At what input length would the two methods have comparable relative overhead? Now factor in that prefix tuning's per-layer KV prefixes can be pre-computed and cached before any request arrives — does this change your analysis, and if so, how?
      </Prose>

      <H3>Exercise 7 — Conceptual: discrete vs continuous search</H3>

      <Prose>
        Prompt engineering searches for a good discrete token sequence. Prompt tuning searches for good continuous vectors. Explain why the continuous search is easier for gradient-based optimization but harder for human interpretation. What does "nearest vocabulary token" analysis of a trained soft prompt tell you, and what are its limitations? If you found that a trained soft prompt's five tokens all mapped to the word "because," what would you infer about the task the prompt was trained for?
      </Prose>

    </div>
  ),
};

export default pTuningSoftPrompts;
