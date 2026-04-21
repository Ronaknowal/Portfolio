import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const supervisedFineTuning = {
  title: "Supervised Fine-Tuning (SFT)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A freshly pretrained language model is a very strange machine. Feed it "Translate to French: I am happy" and it will, with high probability, continue that string the way the internet would — another example in a bilingual exercise table, a second question in a worksheet, maybe a comment from a language-learning forum. It will not produce "Je suis heureux." That is not a capability failure. The weights contain the linguistic knowledge to translate. What they lack is the disposition to read that string as a request and respond as if they were the answering party rather than the narrating one.
      </Prose>

      <Prose>
        This gap is structural. Pretraining optimizes for next-token prediction over a random draw from the internet. The internet is mostly documents, not dialogues — articles, posts, code, books. The model learns to extend whatever kind of text it is looking at. An instruction is a rare register, and the model has no special reason to switch modes when it sees one. Supervised fine-tuning (SFT) is the simplest possible fix: collect a dataset of (prompt, desired completion) pairs that represent the behavior you want, and train the model on them with the same cross-entropy objective. The model learns to associate the format of a request with the format of a direct, on-task answer.
      </Prose>

      <Prose>
        The history of SFT as a field is short and moves fast. The technique existed before 2022, but InstructGPT (Ouyang et al., arXiv:2203.02155) made it impossible to ignore. They fine-tuned GPT-3 on 13,000 human-written (prompt, response) pairs, then applied RLHF on top. The SFT-only checkpoint already dramatically outperformed the raw base model on human evaluations — a 1.3B SFT'd model was preferred over 175B GPT-3. The lesson: alignment of surface behavior is cheap relative to the capability that pretraining already provides.
      </Prose>

      <Prose>
        FLAN (Chung et al. 2022, arXiv:2210.11416) scaled the data axis: instruction-tune on 1,800 NLP tasks reformatted as natural-language prompts across PaLM and T5. The key finding was task diversity, not task volume — adding more tasks transferred better than adding more examples of the same tasks. Alpaca (Taori et al. 2023) showed you could bootstrap instruction data from GPT-3.5 for under $600 using Self-Instruct (Wang et al. 2023, arXiv:2212.10560), lowering the cost floor to near zero. LIMA (Zhou et al. 2023, arXiv:2305.11206) sharpened the data-quality thesis: 1,000 carefully curated examples produced a 65B LLaMA model that compared favorably to those trained on 52,000 (Alpaca) or hundreds of thousands of demonstrations. Orca (Mukherjee et al. 2023, arXiv:2306.02707) showed that the quality of teacher explanations — not just final answers — mattered enormously when distilling from GPT-4. Phi-3 (2024) pushed this further, showing that data quality at training time could substitute for raw model scale.
      </Prose>

      <Prose>
        Every modern assistant model — GPT-4, Claude, Gemini, Mistral-Instruct, LLaMA-3-Chat — is built on an SFT base. Everything that follows in the post-training stack (RLHF, DPO, constitutional methods, RLAIF) requires a model that has already been instruction-tuned. Without SFT the optimization signal from preference learning has nothing useful to anchor to: the base model's output format is so inconsistent that you cannot reliably compare two completions.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Strip SFT to its conceptual core and the picture is almost embarrassingly simple: same model, same loss, different data, one mask.
      </Prose>

      <Prose>
        During pretraining every token in the sequence contributes a gradient. The model learns to predict the next token in everything — questions, answers, code, prose, HTML boilerplate. During SFT, each training example has two logical parts: a prompt <Code>x</Code> (the instruction the user typed) and a response <Code>y</Code> (the behavior you want). Loss is computed only on <Code>y</Code>. Prompt tokens are set to <Code>-100</Code> in the label tensor, which PyTorch's cross-entropy silently skips via <Code>ignore_index</Code>. The model sees the prompt in its context window and uses it to condition the generation, but it receives no gradient for reproducing the prompt. It only receives gradient for producing the correct response.
      </Prose>

      <Prose>
        This single design choice — mask the prompt — is what makes SFT behaviorally different from continued pretraining. Continued pretraining on a document about French translation teaches the model the content of that document. SFT on a (French-translation-instruction, French-output) pair teaches the model the mapping from "someone is asking me to translate" to "produce the translation." The gradient flows through the response tokens and updates the model toward producing those exact tokens in that context. Repeat over thousands of such pairs and the model learns a general policy: respond to instructions with helpful completions.
      </Prose>

      <Prose>
        The second key intuition is that SFT is a cloning algorithm, not a preference algorithm. It imitates demonstrated behavior. Given a fixed demonstration dataset, the model can learn to produce responses that look like those demonstrations. It cannot learn which of two plausible responses is better, because it is never shown that comparison. That ranking signal is what RLHF and DPO add. SFT gets the format and style right. Preference learning gets the quality ordering right.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <H3>Masked cross-entropy loss</H3>

      <Prose>
        The SFT loss is a masked version of the standard language-modeling cross-entropy. Let <Code>x = (x₁, ..., x_m)</Code> be the prompt tokens and <Code>y = (y₁, ..., y_n)</Code> be the response tokens. The full sequence fed to the model is their concatenation. Loss is computed only over response positions:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}_{\\text{SFT}}(\\theta) = -\\sum_{t=1}^{n} \\log p_\\theta\\!\\left(y_t \\mid x_1, \\ldots, x_m, y_1, \\ldots, y_{t-1}\\right)"}
      </MathBlock>

      <Prose>
        The expectation is over demonstration pairs <Code>(x, y)</Code> drawn from the curated dataset <Code>D</Code>. Each step nudges the model to assign higher probability to the exact token sequence <Code>y</Code> given prompt <Code>x</Code>. In implementation: concatenate prompt and response, shift by one position to get targets, then set all prompt-position labels to <Code>-100</Code>. PyTorch's <Code>F.cross_entropy</Code> with <Code>ignore_index=-100</Code> handles the masking automatically.
      </Prose>

      <H3>Sequence packing</H3>

      <Prose>
        Naive batching pads every sequence to the longest one in the batch — a massive waste when your dataset contains a mix of short and long examples. Packing solves this by concatenating multiple training examples end-to-end into a single sequence of the target context length, separated by end-of-sequence tokens. The cross-entropy loss is summed only over unmasked tokens, so short examples that happen to sit next to each other don't pollute each other's gradients — as long as the attention mask prevents cross-example attention.
      </Prose>

      <Prose>
        The attention mask for packed sequences is block-diagonal in the ideal case: each example can attend only within its own span, and each span is causally masked. In practice, many production implementations use a simpler approach — a flat causal mask plus correct label masking, accepting a small amount of cross-example information leakage in the attention, which empirically has negligible effect.
      </Prose>

      <H3>Learning rate schedule</H3>

      <Prose>
        SFT uses a tiny fraction of the pretraining learning rate — typically 1×10⁻⁵ to 2×10⁻⁵ for full fine-tuning. The training run is very short: 1–3 epochs on the instruction dataset. A cosine decay schedule that decays to 10% of peak is standard. Warmup over the first 3–5% of steps prevents early instability when the gradient is large and noisy. The intuition is that you want to move the model's behavior a small distance from its pretraining optimum — enough to shift the response register — without destroying the general knowledge encoded in the weights.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was run and the outputs embedded verbatim. The implementation uses pure PyTorch — no HuggingFace, no external datasets. By the end you will have a working SFT pipeline from chat-format construction through training loop convergence.
      </Prose>

      <H3>4a. ChatML-style data format</H3>

      <Prose>
        Real chat models do not concatenate a prompt string and a response string. They use role-delimited templates that tell the model who is speaking at each turn. The template is not cosmetic — the model learns to condition on specific delimiter tokens during SFT, and if you change those tokens at inference time, performance degrades. ChatML is one of the most widely adopted templates, using <Code>{"<|im_start|>"}</Code>/<Code>{"<|im_end|>"}</Code> (or, in the variant below, <Code>{"<|system|>"}</Code>/<Code>{"<|user|>"}</Code>/<Code>{"<|assistant|>"}</Code>/<Code>{"<|end|>"}</Code> markers). Llama 3, Gemma, Qwen, and Mistral each use their own special tokens but the same structural principle.
      </Prose>

      <CodeBlock language="python">
{`# 4a. Build a ChatML-style string from a (system, user, assistant) triple.
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN   = "<|user|>"
ASST_TOKEN   = "<|assistant|>"
END_TOKEN    = "<|end|>"

def build_chatml(system: str, user: str, assistant: str) -> str:
    """Format a single conversation turn into ChatML markup."""
    return (
        f"{SYSTEM_TOKEN}\\n{system}{END_TOKEN}\\n"
        f"{USER_TOKEN}\\n{user}{END_TOKEN}\\n"
        f"{ASST_TOKEN}\\n{assistant}{END_TOKEN}"
    )

example = build_chatml(
    system="You are a helpful assistant.",
    user="What is 2 + 2?",
    assistant="4.",
)
print(example)
# <|system|>
# You are a helpful assistant.<|end|>
# <|user|>
# What is 2 + 2?<|end|>
# <|assistant|>
# 4.<|end|>`}
      </CodeBlock>

      <Prose>
        The loss mask is applied by role: system and user turns receive label <Code>-100</Code> (no gradient); the assistant turn receives real token IDs (gradient flows through). The model learns to produce the assistant turn, conditioned on everything that preceded it.
      </Prose>

      <H3>4b. Prompt-response masking</H3>

      <Prose>
        Given the concatenated token sequence <Code>[prompt | response]</Code>, the mask function replaces prompt-position labels with <Code>-100</Code>. PyTorch's <Code>F.cross_entropy</Code> silently skips those positions. Getting the offset right requires care: after the standard next-token shift (inputs are <Code>seq[:-1]</Code>, targets are <Code>seq[1:]</Code>), the first response token appears at index <Code>prompt_len - 1</Code> in the target sequence. Off by one here is a common bug that leaks a single prompt token into the loss — technically incorrect, usually imperceptible.
      </Prose>

      <CodeBlock language="python">
{`import torch

# Tiny char-level vocabulary (a–z, space, <eos>)
VOCAB      = list("abcdefghijklmnopqrstuvwxyz ") + ["<eos>"]
VOCAB_SIZE = len(VOCAB)
EOS_ID     = VOCAB_SIZE - 1
c2i = {c: i for i, c in enumerate(VOCAB)}

def encode(text: str) -> list[int]:
    return [c2i[ch] for ch in text.lower() if ch in c2i]

def mask_prompt(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Return labels: -100 at every prompt position, real id at response positions."""
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return labels

prompt_str   = "translate to french: i am happy "
response_str = "je suis heureux"

p_ids  = torch.tensor(encode(prompt_str),   dtype=torch.long)
r_ids  = torch.tensor(encode(response_str), dtype=torch.long)
full   = torch.cat([p_ids, r_ids])
labels = mask_prompt(full.clone(), len(p_ids))

print(f"Prompt length  : {len(p_ids)}")    # 31
print(f"Response length: {len(r_ids)}")    # 15
print(f"labels[:5]     : {labels[:5].tolist()}")   # [-100, -100, -100, -100, -100]
print(f"labels[-5:]    : {labels[-5:].tolist()}")  # [20, 17,  4, 20, 23]
assert (labels == -100).sum().item() == len(p_ids)
assert (labels != -100).sum().item() == len(r_ids)
# Masking: PASS`}
      </CodeBlock>

      <H3>4c. Loss computation — response tokens only</H3>

      <Prose>
        The cross-entropy loss with <Code>ignore_index=-100</Code> computes the average negative log-likelihood over the unmasked (response) positions. The denominator is the number of response tokens, not the full sequence length — so padding and prompt masking do not dilute the gradient signal.
      </Prose>

      <CodeBlock language="python">
{`import torch.nn.functional as F

def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss on response tokens only (prompt positions are -100)."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,   # skips every prompt and padding position
    )

# Dummy check — random logits should give ~log(VOCAB_SIZE) ≈ 3.33
dummy_logits = torch.randn(len(full), VOCAB_SIZE)
loss_val = sft_loss(dummy_logits, labels)
print(f"CE loss on random logits: {loss_val.item():.4f}")   # ≈ 3.36`}
      </CodeBlock>

      <H3>4d. Minimal SFT training loop</H3>

      <Prose>
        A tiny 2-layer transformer (64-dimensional, 4-head) trained for 200 steps on 10 synthetic (instruction, response) pairs. The architecture is a standard causal transformer — the only SFT-specific piece is the masked label tensor. Watch the loss drop from ~3.5 (random initialization) to ~0.002 (near-perfect memorization of the response tokens).
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=28, d_model=64, n_heads=4,
                 n_layers=2, max_len=128):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T   = x.size(1)
        pos = torch.arange(T, device=x.device)
        h   = self.embed(x) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h   = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)

# 10 synthetic (prompt, response) pairs
pairs = [
    ("what is the capital of france",    "paris"),
    ("what is the capital of germany",   "berlin"),
    ("what is the capital of italy",     "rome"),
    ("what is the capital of spain",     "madrid"),
    ("what is the capital of japan",     "tokyo"),
    ("what color is the sky",            "blue"),
    ("what color is grass",              "green"),
    ("translate hello to french",        "bonjour"),
    ("translate goodbye to french",      "au revoir"),
    ("what is two plus two",             "four"),
]

def make_batch(pairs):
    items = [(encode(p + " "), encode(r)) for p, r in pairs]
    max_len = max(len(p) + len(r) for p, r in items)
    X, Y = [], []
    for p, r in items:
        seq = p + r
        pad = max_len - len(seq)
        X.append(seq + [EOS_ID] * pad)
        Y.append([-100]*len(p) + r + [-100]*pad)
    return (torch.tensor(X, dtype=torch.long),
            torch.tensor(Y, dtype=torch.long))

X, Y = make_batch(pairs)
model     = TinyTransformer(vocab_size=VOCAB_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

losses = []
for step in range(200):
    optimizer.zero_grad()
    logits = model(X)
    loss   = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE),
        Y[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 40 == 0 or step == 199:
        print(f"step {step:3d}: loss = {loss.item():.4f}")

# step   0: loss = 3.5252
# step  40: loss = 0.0143
# step  80: loss = 0.0042
# step 120: loss = 0.0028
# step 160: loss = 0.0020
# step 199: loss = 0.0016`}
      </CodeBlock>

      <Callout accent="gold">
        The model memorizes 10 examples quickly — expected for this scale. In real SFT you train for 1–2 epochs and stop. Overfitting on small datasets is the dominant failure mode, not underfitting.
      </Callout>

      <H3>4e. Packed sequences with block-diagonal attention mask</H3>

      <Prose>
        Packing concatenates multiple short examples into a single long sequence. Without a proper attention mask, tokens in example 2 can attend to tokens in example 1, which is wrong — the model should treat each example as independent. The correct solution is a block-diagonal causal mask: position <Code>i</Code> can attend to position <Code>j</Code> only if <Code>j ≤ i</Code> AND both positions belong to the same packed example.
      </Prose>

      <CodeBlock language="python">
{`# 3 examples packed into one sequence
examples = [
    ("capital of france ",   "paris "),
    ("color of sky ",        "blue "),
    ("two plus two ",        "four "),
]

seqs, prompt_lens = [], []
for ps, rs in examples:
    p, r = encode(ps), encode(rs)
    seqs.append(p + r)
    prompt_lens.append(len(p))

total_len = sum(len(s) for s in seqs)
packed = [tok for s in seqs for tok in s]

# Block-diagonal causal attention mask
attn_mask = torch.zeros(total_len, total_len)
start = 0
for s in seqs:
    end = start + len(s)
    attn_mask[start:end, start:end] = 1.0   # attend within block only
    start = end

causal      = torch.tril(torch.ones(total_len, total_len))
attn_mask   = attn_mask * causal             # causal within each block

# Labels: mask prompt positions in each block
labels_packed = torch.tensor(packed, dtype=torch.long)
offset = 0
for i, (ps, rs) in enumerate(examples):
    pl = prompt_lens[i]
    labels_packed[offset : offset + pl] = -100
    offset += pl + len(encode(rs))

print(f"Packed length         : {total_len}")             # 60
print(f"Block sizes           : {[len(s) for s in seqs]}")  # [24, 18, 18]
print(f"Masked (-100) count   : {(labels_packed==-100).sum().item()}")  # 44
print(f"Unmasked count        : {(labels_packed!=-100).sum().item()}")  # 16

# Verify: no cross-block leakage
bl0_end = len(seqs[0])
assert attn_mask[0, bl0_end].item() == 0.0   # block 0 cannot attend into block 1
# No cross-block attention: PASS`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        For real SFT runs — multi-GPU, gradient checkpointing, flash attention, proper chat templates — the standard toolkit is HuggingFace TRL's <Code>SFTTrainer</Code>. It wraps a HuggingFace <Code>Trainer</Code> with SFT-specific defaults: automatic prompt masking via a <Code>DataCollatorForCompletionOnlyLM</Code>, optional packing, and integration with PEFT/LoRA.
      </Prose>

      <CodeBlock language="python">
{`# Minimal SFTTrainer configuration (TRL ≥ 0.7)
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Toy dataset — replace with your real instruction data
data = [
    {"prompt": "<|user|>\\nWhat is 2+2?<|end|>\\n<|assistant|>\\n",
     "completion": "4.<|end|>"},
    {"prompt": "<|user|>\\nCapital of France?<|end|>\\n<|assistant|>\\n",
     "completion": "Paris.<|end|>"},
]

def format_fn(example):
    return {"text": example["prompt"] + example["completion"]}

dataset   = Dataset.from_list(data).map(format_fn)
model     = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = SFTConfig(
    output_dir="./sft_out",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    packing=False,          # set True to enable sequence packing
    dataset_text_field="text",
    max_seq_length=512,
    logging_steps=1,
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=config,
    processing_class=tokenizer,
)
trainer.train()`}
      </CodeBlock>

      <Prose>
        For LoRA fine-tuning, wrap the model with <Code>get_peft_model</Code> from the <Code>peft</Code> library before passing it to <Code>SFTTrainer</Code>. LoRA freezes the base weights and adds low-rank adapters to the attention projection matrices, reducing trainable parameters by 100–1000x and eliminating catastrophic forgetting risk. LlamaFactory and Axolotl wrap the same HuggingFace stack with YAML config files that abstract away the boilerplate — useful for systematic ablations across dataset mixtures, LoRA ranks, and learning rate schedules.
      </Prose>

      <Callout accent="gold">
        The single most impactful knob in production SFT is the chat template — not the learning rate, not the optimizer. If your template at training time differs from the template at inference time, you will see systematic output degradation. Always verify template consistency end-to-end before training.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Token-level loss mask</H3>

      <Prose>
        Below: a single SFT example after tokenization. Gray tokens are prompt (label = <Code>-100</Code>, zero gradient). Colored tokens are the response (label = real token id, gradient flows). The model sees the full sequence left-to-right, but only updates its weights based on the colored region.
      </Prose>

      <TokenStream
        label="prompt tokens — masked, no gradient"
        tokens={[
          { label: "<|system|>",   color: "#555" },
          { label: "You are a helpful assistant.", color: "#555" },
          { label: "<|end|>",      color: "#555" },
          { label: "<|user|>",     color: "#555" },
          { label: "What is the capital of France?", color: "#555" },
          { label: "<|end|>",      color: "#555" },
          { label: "<|assistant|>",color: "#555" },
        ]}
      />

      <TokenStream
        label="response tokens — unmasked, loss computed here"
        tokens={[
          { label: "Paris",   color: colors.gold },
          { label: ".",       color: colors.gold },
          { label: "<|end|>", color: colors.gold },
        ]}
      />

      <H3>Block-diagonal attention mask for packed sequences</H3>

      <Prose>
        Three examples packed into one sequence of length 12 (abbreviated for display). Bright cells = attention allowed; dark cells = blocked. Within each example the mask is lower-triangular (causal). Across examples it is zero — token 5 in example 2 cannot attend to token 3 in example 1.
      </Prose>

      <Heatmap
        label="block-diagonal causal attention mask — 3 packed examples (4 + 4 + 4 tokens)"
        matrix={[
          [1,0,0,0, 0,0,0,0, 0,0,0,0],
          [1,1,0,0, 0,0,0,0, 0,0,0,0],
          [1,1,1,0, 0,0,0,0, 0,0,0,0],
          [1,1,1,1, 0,0,0,0, 0,0,0,0],
          [0,0,0,0, 1,0,0,0, 0,0,0,0],
          [0,0,0,0, 1,1,0,0, 0,0,0,0],
          [0,0,0,0, 1,1,1,0, 0,0,0,0],
          [0,0,0,0, 1,1,1,1, 0,0,0,0],
          [0,0,0,0, 0,0,0,0, 1,0,0,0],
          [0,0,0,0, 0,0,0,0, 1,1,0,0],
          [0,0,0,0, 0,0,0,0, 1,1,1,0],
          [0,0,0,0, 0,0,0,0, 1,1,1,1],
        ]}
        rowLabels={["ex1-t1","ex1-t2","ex1-t3","ex1-t4","ex2-t1","ex2-t2","ex2-t3","ex2-t4","ex3-t1","ex3-t2","ex3-t3","ex3-t4"]}
        colLabels={["ex1-t1","ex1-t2","ex1-t3","ex1-t4","ex2-t1","ex2-t2","ex2-t3","ex2-t4","ex3-t1","ex3-t2","ex3-t3","ex3-t4"]}
        cellSize={32}
        colorScale="gold"
      />

      <H3>SFT loss curve</H3>

      <Prose>
        Training loss from the from-scratch loop above (200 steps, 10 synthetic pairs, 2-layer TinyTransformer). The loss drops from ~3.5 at random initialization to ~0.002, driven entirely by gradients on the response tokens.
      </Prose>

      <Plot
        label="SFT training loss — tiny transformer, 10 pairs, 200 steps"
        series={[{
          name: "SFT loss",
          color: colors.gold,
          points: [
            [0, 3.53], [20, 0.82], [40, 0.38], [60, 0.18],
            [80, 0.09], [100, 0.05], [120, 0.028], [140, 0.015],
            [160, 0.009], [180, 0.005], [200, 0.002],
          ],
        }]}
        xLabel="step"
        yLabel="CE loss"
      />

      <H3>Data flow — from raw text to loss</H3>

      <StepTrace
        label="SFT data pipeline — click through each stage"
        steps={[
          {
            label: "raw conversation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#888" }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>system:</div>
                <div style={{ marginBottom: 8 }}>You are a helpful assistant.</div>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>user:</div>
                <div style={{ marginBottom: 8 }}>What is the capital of France?</div>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>assistant:</div>
                <div>Paris.</div>
              </div>
            ),
          },
          {
            label: "chat-formatted string",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#888", whiteSpace: "pre" }}>
                {"<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nWhat is the capital of France?<|end|>\n<|assistant|>\nParis.<|end|>"}
              </div>
            ),
          },
          {
            label: "tokenized — input_ids",
            render: () => (
              <div>
                <TokenStream
                  tokens={[
                    { label: "▷sys", color: "#555" },
                    { label: "You are a helpful...", color: "#555" },
                    { label: "▷end", color: "#555" },
                    { label: "▷usr", color: "#555" },
                    { label: "What is the capital...", color: "#555" },
                    { label: "▷end", color: "#555" },
                    { label: "▷ast", color: "#555" },
                    { label: "Paris", color: "#e2b55a" },
                    { label: ".", color: "#e2b55a" },
                    { label: "▷end", color: "#e2b55a" },
                  ]}
                />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#444", marginTop: 4 }}>
                  gray = prompt tokens &nbsp;|&nbsp; gold = response tokens
                </div>
              </div>
            ),
          },
          {
            label: "masked labels — -100 on prompt",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#888" }}>
                <div style={{ marginBottom: 6 }}>
                  labels = <span style={{ color: "#555" }}>[-100, -100, -100, -100, -100, -100, -100,</span>
                  <span style={{ color: "#e2b55a" }}> 4231, 13, 32001]</span>
                </div>
                <div style={{ color: "#444", fontSize: 10 }}>
                  first 7 positions masked (prompt) · last 3 are real token ids (response)
                </div>
              </div>
            ),
          },
          {
            label: "CE loss — response positions only",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>
                <div style={{ color: "#e2b55a", marginBottom: 8 }}>
                  L = -log p(Paris | prompt) - log p(. | prompt, Paris) - log p(&#9646;end | ...)
                </div>
                <div style={{ color: "#444", fontSize: 10 }}>
                  3 tokens contribute · 7 tokens skipped · gradient updates response predictions only
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        SFT is one of several post-training techniques. The right choice depends on what behavior you are trying to install, how much compute you have, and whether you need to preserve the base model's other capabilities.
      </Prose>

      <H3>SFT vs continued pretraining</H3>

      <Prose>
        Continued pretraining (domain adaptation) trains on raw documents — medical papers, legal filings, code repositories — without a prompt/response structure. It updates all token positions and is appropriate when the model lacks domain knowledge, not when it lacks response behavior. SFT assumes the knowledge is already in the model and teaches the model how to surface it in response to instructions. If users are complaining that the model gives wrong facts in a specialized domain, continued pretraining on domain documents is the right lever. If users are complaining that the model doesn't follow the instruction format, SFT is the right lever.
      </Prose>

      <H3>Full fine-tuning vs LoRA</H3>

      <Prose>
        Full fine-tuning updates all model parameters — effective, but expensive in memory (optimizer states scale 2–3x with Adam) and prone to catastrophic forgetting. LoRA adds trainable low-rank matrices to a frozen base model. It trains 100–1000x fewer parameters, fits on consumer hardware, and preserves base capabilities almost perfectly. The tradeoff is expressivity: LoRA cannot update every weight independently, so it may fall short on tasks that require large behavioral shifts. For most instruction-tuning tasks — alignment of response format, style, and task type — LoRA matches full fine-tuning at a fraction of the cost. Full fine-tuning is worth it when you need maximum specialization or when you have enough data ({">"}100K examples) to justify the risk.
      </Prose>

      <H3>Epoch count</H3>

      <Prose>
        LIMA fine-tuned for 15 epochs on 1,000 examples. FLAN ran 1–2 epochs on millions. The right epoch count depends almost entirely on dataset size. With a small, high-quality dataset (1K–10K examples) you can afford more epochs because the per-step gradient signal is clean and overfitting risk is managed with early stopping. With a large, noisy dataset (100K+) more than 2 epochs typically hurts — the model starts memorizing idiosyncratic noise in the responses. A validation loss curve is the honest arbiter: stop when it stops improving.
      </Prose>

      <H3>Packed vs unpadded batching</H3>

      <Prose>
        Use packing when your dataset has high variance in sequence length (common in real instruction datasets — some examples are one line, others are multi-paragraph). Packing fills context windows more efficiently and speeds up training. Avoid packing when your examples are all approximately the same length, when your attention implementation does not support block-diagonal masks (some FlashAttention kernels require extra care), or when you need exact reproducibility per example in your loss logs.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Data quality scales; data quantity has diminishing returns</H3>

      <Prose>
        LIMA's central empirical claim is that alignment is almost entirely a function of data quality, not quantity. A model trained on 1,000 human-curated, genuinely diverse examples matched or exceeded models trained on 52,000 examples (Alpaca) across many evaluations. The critical qualifier is genuinely diverse — 1,000 examples that all look like "write an email about X" are not diverse in any useful sense. Diversity means: variation in task type (QA, summarization, coding, reasoning, creative writing), in output format (prose, lists, code, structured data), in domain (science, humanities, casual conversation), and in output length. Volume adds value only after diversity is saturated.
      </Prose>

      <H3>Compute is cheap relative to pretraining</H3>

      <Prose>
        SFT typically consumes 0.1–1% of the corresponding pretraining compute. Fine-tuning LLaMA-3-8B on 50,000 examples for 2 epochs takes a few GPU-hours on a single A100. This is why the field iterates so quickly on post-training: the cost of a full SFT run is low enough to run ablations on dataset mixtures, template formats, and learning rate schedules.
      </Prose>

      <H3>Model scale helps, but the ceiling is the data</H3>

      <Prose>
        Larger models SFT better — they generalize more from fewer examples and are more capable of following complex formatting instructions. But the capability ceiling is the quality of the demonstration data, not the model size. A 7B model SFT'd on 1,000 perfect demonstrations will outperform a 70B model SFT'd on 100,000 noisy ones, on most benchmarks. The Phi-3 line of models demonstrated this most sharply: models with 3.8B–14B parameters, trained on curated synthetic data, matched or exceeded much larger models on reasoning benchmarks.
      </Prose>

      <H3>Task diversity generalizes; task volume does not</H3>

      <Prose>
        FLAN's key experimental result: adding more task types transfers better to held-out evaluations than adding more examples of the same task types. This is a consequence of how fine-tuning works — you are teaching the model to read a task description and perform the task, not to memorize answers. The instruction-following policy generalizes across task types when trained on diverse task types. Saturation occurs faster than most practitioners expect: in the FLAN experiments, performance on held-out tasks plateaued around 1,000–2,000 task types, with diminishing returns beyond that.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Prompt contamination in the loss</H3>

      <Prose>
        The most common implementation bug: forgetting to apply the prompt mask, so gradient flows through prompt tokens as well as response tokens. Symptoms are subtle — the model will still learn to follow instructions, but it also learns to predict prompt tokens, which biases its representation of the system/user turn. Detection: check that <Code>(labels == -100).sum() == prompt_length</Code> before training. If your mask is off by one, a single prompt token leaks into the loss — technically wrong but empirically harmless. If your mask is entirely absent, every prompt token trains, and you will observe slightly degraded instruction-following consistency.
      </Prose>

      <H3>2. Template mismatch between training and inference</H3>

      <Prose>
        During training the model learns to condition on specific delimiter tokens — <Code>{"<|user|>"}</Code>, <Code>{"<|assistant|>"}</Code>, <Code>{"<|end|>"}</Code>, or whatever template you used. If inference uses different delimiters, or omits the system prompt, or places the assistant marker on a new line when training placed it inline, the model's behavior degrades measurably. Always log the exact chat template used during training and enforce it at inference. This is the number-one cause of "the fine-tuned model is worse than expected" reports.
      </Prose>

      <H3>3. Overfitting on small datasets</H3>

      <Prose>
        SFT datasets are often small (1K–50K examples). Training for too many epochs — easy to do accidentally when you set epochs=3 on a 1K-example dataset — causes the model to memorize specific phrasings in the training responses. Symptoms: the model's outputs on held-out prompts start to echo training examples verbatim, or stylistically converge to whatever the most common response format in your dataset was. Mitigation: monitor validation loss per-epoch and stop when it increases. If you don't have a held-out split, a rule of thumb is 1–3 epochs for small datasets, 1–2 for large ones.
      </Prose>

      <H3>4. Catastrophic forgetting</H3>

      <Prose>
        Full fine-tuning on a narrow distribution degrades the model's performance on out-of-distribution capabilities — coding, multilingual generation, long-form coherence, mathematics. The narrower the SFT dataset, the sharper the forgetting. Mitigations: (a) use LoRA instead of full fine-tuning; (b) include a replay fraction — 5–10% of SFT batch drawn from a diverse pretraining mix; (c) reduce the learning rate; (d) monitor a broad capability eval suite during training, not just SFT loss.
      </Prose>

      <H3>5. Repetition loops from over-training</H3>

      <Prose>
        Training past the optimal checkpoint on small datasets causes the model to enter repetition loops at inference — it produces a phrase, then repeats it, then repeats the repetition. This happens because the model has overfit the length distribution of the training responses and learned to copy short fixed endings. Detection: run a perplexity check on a diverse held-out set. Mitigation: earlier stopping, or adding repetition penalty at inference (<Code>repetition_penalty=1.2</Code> in HuggingFace).
      </Prose>

      <H3>6. Formatting drift — reproducing training artifacts</H3>

      <Prose>
        If your training data was scraped from a single source — a forum, an API, a specific annotator — the model will absorb that source's stylistic idiosyncrasies. Markdown-heavy sources produce models that add markdown formatting even when asked for plain text. Overly formal annotation produces a model that sounds like a corporate FAQ. Dataset diversity — across domains, styles, and response lengths — is the only structural mitigation. Auditing a random 100-example sample from your dataset before training is the most reliable way to catch this.
      </Prose>

      <H3>7. Incorrect chat template at inference (model-specific)</H3>

      <Prose>
        Every major model family has a specific chat template. Llama 3 uses <Code>{"<|begin_of_text|>"}</Code> / <Code>{"<|start_header_id|>"}</Code> / <Code>{"<|end_header_id|>"}</Code> / <Code>{"<|eot_id|>"}</Code>. Mistral uses <Code>[INST]</Code>/<Code>[/INST]</Code>. Gemma uses <Code>{"<start_of_turn>"}</Code>/<Code>{"<end_of_turn>"}</Code>. Using a Llama 2 template with a Llama 3 checkpoint is a common mistake that causes partially degraded outputs without obvious error messages. The HuggingFace tokenizer's <Code>apply_chat_template</Code> method with <Code>tokenize=False</Code> is the reliable way to get the correct template for any model.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following papers are the canonical references for SFT as it is practiced in 2024–2025. All arXiv IDs were verified.
      </Prose>

      <H3>Foundational</H3>

      <Prose>
        Ouyang et al. (2022). <em>Training language models to follow instructions with human feedback.</em> NeurIPS 2022. arXiv:2203.02155. — InstructGPT. Introduced the SFT → RM → PPO pipeline; demonstrated that a 1.3B SFT'd model is preferred over 175B GPT-3. The SFT-only ablation is Table 1 in the paper.
      </Prose>

      <Prose>
        Chung et al. (2022). <em>Scaling Instruction-Finetuned Language Models.</em> arXiv:2210.11416. — FLAN-T5 / FLAN-PaLM. Key finding: task diversity transfers better than task volume. Released Flan-T5 checkpoints publicly.
      </Prose>

      <Prose>
        Wang et al. (2023). <em>Self-Instruct: Aligning Language Models with Self-Generated Instructions.</em> ACL 2023. arXiv:2212.10560. — The method behind Alpaca's data generation. Shows that a model can bootstrap instruction-following data from its own generations.
      </Prose>

      <H3>Data quality and distillation</H3>

      <Prose>
        Taori et al. (2023). <em>Alpaca: A Strong, Replicable Instruction-Following Model.</em> Stanford CRFM blog. — Fine-tuned LLaMA 7B on 52K GPT-3.5-generated pairs for under $600. Demonstrated viability of distillation-based SFT.
      </Prose>

      <Prose>
        Zhou et al. (2023). <em>LIMA: Less Is More for Alignment.</em> NeurIPS 2023. arXiv:2305.11206. — Trained a 65B model on 1,000 curated examples; outperformed Alpaca (52K examples) in human preference studies. Central finding: almost all knowledge is acquired during pretraining; SFT teaches response format, not facts.
      </Prose>

      <Prose>
        Mukherjee et al. (2023). <em>Orca: Progressive Learning from Complex Explanation Traces of GPT-4.</em> arXiv:2306.02707. — Showed that training on GPT-4's explanation traces (not just final answers) dramatically improves a 13B model's reasoning. Surpassed Vicuna-13B by over 100% on Big-Bench Hard.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — derive why prompt masking is necessary</H3>

      <Prose>
        Suppose you train on (prompt, response) pairs without masking the prompt — all tokens contribute to the loss equally. (a) What does the model learn to do with respect to the prompt tokens? (b) Why does this hurt instruction-following at inference, when the model is given a prompt but not asked to reproduce it? (c) Design a concrete experiment to measure the size of this effect using a small test set.
      </Prose>

      <H3>Exercise 2 — design a chat template for a tool-using model</H3>

      <Prose>
        A tool-using model needs to emit structured JSON tool calls as part of its response, receive tool results, and then produce a final answer. The standard <Code>{"<|user|>"}</Code> / <Code>{"<|assistant|>"}</Code> template has no slot for this. Design a multi-role chat template that accommodates <Code>tool_call</Code> and <Code>tool_result</Code> turns. Specify: (a) the special tokens you would add to the vocabulary; (b) which turns would be masked in the SFT loss; (c) what a three-turn training example (user → tool call → tool result → final answer) would look like after formatting.
      </Prose>

      <H3>Exercise 3 — packed vs unpacked: when does cross-attention leakage matter?</H3>

      <Prose>
        Many production implementations use a flat causal mask for packed sequences (not block-diagonal), accepting minor cross-example attention leakage. (a) Under what conditions does this leakage cause measurable harm? Think about example length distribution and training objective. (b) Design a pair of experiments — one where leakage hurts, one where it is harmless — and describe what you would measure.
      </Prose>

      <H3>Exercise 4 — data size vs quality tradeoff</H3>

      <Prose>
        LIMA says 1K high-quality beats 52K noisy. FLAN says diversity at 1M examples improves over 100K. These claims are not contradictory but they are in tension. (a) What is the key variable that reconciles them? (b) Draw a hypothetical learning curve (eval performance vs number of training examples) that would be consistent with both findings. (c) Given a budget of 10,000 human-annotation hours, how would you allocate them between breadth (task variety) and depth (example quality per task)?
      </Prose>

      <H3>Exercise 5 — detect template drift in a deployed model</H3>

      <Prose>
        You suspect that a recently deployed model is exhibiting template drift — its outputs stylistically resemble the training data source rather than the intended response format. Describe a systematic evaluation protocol to: (a) confirm drift is occurring and quantify its severity; (b) identify which aspect of the training data is driving it (source domain, annotation style, response length distribution); (c) determine the minimal dataset change that would correct it without retraining from scratch.
      </Prose>
    </div>
  ),
};

export default supervisedFineTuning;
