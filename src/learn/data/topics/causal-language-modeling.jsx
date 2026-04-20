import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream } from "../../components/viz";

const causalLanguageModeling = {
  title: "Causal Language Modeling (Next-Token Prediction)",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        The training objective that built modern LLMs is embarrassingly simple: given some text, predict the next token. Every GPT model, every Llama, every Claude — all of them, at their core, minimize cross-entropy on one thing. This simplicity is not a limitation that the field hasn't gotten around to fixing. It is the design. The entire capability stack — syntax, world knowledge, multi-step reasoning, stylistic register — emerges from optimizing a single scalar loss over next-token predictions, applied to enough text at enough scale. Before surveying architectures or scaling laws or alignment techniques, it is worth sitting with how much weight one unglamorous objective carries.
      </Prose>

      <H2>The mechanics</H2>

      <Prose>
        Given an input sequence <Code>x = (x₁, x₂, …, xₙ)</Code>, the model produces a probability distribution over the vocabulary at each position <Code>t</Code>, conditioned on everything that came before it: <Code>p(xₜ | x₁, …, xₜ₋₁)</Code>. The training loss is the negative log-likelihood of the observed tokens under that distribution, summed across all positions in the sequence.
      </Prose>

      <MathBlock>{"\\mathcal{L} = -\\sum_{t=1}^{N} \\log p_\\theta(x_t \\mid x_{<t})"}</MathBlock>

      <Prose>
        The subscript <Code>{"x_{<t}"}</Code> is the whole causal constraint written in one character. Position <Code>t</Code> may only attend to positions 1 through <Code>t−1</Code>. In a transformer this is enforced by a causal attention mask — an upper-triangular matrix of <Code>−∞</Code> values that zeros out the softmax weights for any future position. The mask is applied before the softmax inside every attention head, in every layer, for every token. Without it, position 3 could attend to position 7, the model could read the answer before making its prediction, the loss would be trivially low, and the weights would learn nothing useful. The mask is not a minor implementation detail. It is what makes the objective well-defined.
      </Prose>

      <Prose>
        Minimizing this loss is equivalent to maximizing the likelihood of the training corpus under the model's factorization of the joint distribution. By the chain rule of probability, the joint probability of any sequence factors exactly as a product of conditional probabilities: <Code>p(x₁, …, xₙ) = p(x₁) · p(x₂|x₁) · … · p(xₙ|x₁,…,xₙ₋₁)</Code>. Causal language modeling maximizes each factor simultaneously — one loss, one gradient, the full joint distribution.
      </Prose>

      <H2>Why it works so well</H2>

      <Prose>
        Two properties of the objective are non-obvious and together explain most of the field's recent history.
      </Prose>

      <Prose>
        The first is signal density. In a standard classification task, you get one gradient signal per labeled example. In causal language modeling, you get one signal per token per example. A 512-token sequence yields 512 gradient contributions in a single forward pass. Training efficiency scales with sequence length rather than working against it — longer documents are more informative, not more wasteful. At the scales where modern LLMs are trained (tens of trillions of tokens), this density is the difference between feasibility and impossibility.
      </Prose>

      <Prose>
        The second is that the task is universal without any explicit labeling. To predict the next token well, the model must internalize grammar, because grammatical continuations score higher than ungrammatical ones. It must model semantics, because semantically coherent continuations outscore incoherent ones. It must accumulate world knowledge, because factually accurate continuations are more likely under any real training corpus. It must handle long-range dependencies, coreference, and discourse structure. It must learn stylistic and tonal patterns. None of these are labeled — the labels are the tokens themselves, derived for free from raw text. The objective is self-supervised in the strictest sense: the supervision signal is present in every piece of text ever written, waiting to be consumed.
      </Prose>

      <H3>Teacher forcing</H3>

      <Prose>
        During training, the model predicts every position in the sequence in parallel. At position <Code>t</Code>, the input is the ground-truth prefix <Code>(x₁, …, xₜ₋₁)</Code> — not the model's own previous predictions. This is teacher forcing. The model is shown the correct history at every step, regardless of what it would have predicted.
      </Prose>

      <Prose>
        The efficiency payoff is substantial. A single forward pass through a transformer processes all <Code>N</Code> prediction problems simultaneously, because the causal mask isolates them from one another while sharing the computation of every layer. Without teacher forcing, training would require an autoregressive loop — generate token 1, feed it back, generate token 2, feed it back — which is sequential by construction and prohibitively slow. Teacher forcing is why transformer pretraining is parallelizable across sequence length as well as across the batch.
      </Prose>

      <Prose>
        The tradeoff is exposure bias. At inference time, the model consumes its own outputs, not ground truth. A mistake at position <Code>t</Code> becomes part of the prefix for position <Code>t+1</Code>, and the distribution the model was trained on never included its own errors. The model has seen perfect prefixes during training and imperfect ones during inference — a gap that grows with sequence length and is one of the motivations behind techniques like scheduled sampling, minimum Bayes risk decoding, and reinforcement learning from human feedback.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def next_token_loss(model, tokens):
    """tokens: (batch, seq). The model outputs logits at each position."""
    logits = model(tokens[:, :-1])           # predict positions 1..N
    targets = tokens[:, 1:]                   # ground truth shifted by one
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

# At inference, we sample one token at a time from the model's own outputs.
# The training/inference asymmetry is known as exposure bias.`}
      </CodeBlock>

      <H3>Autoregressive sampling at inference</H3>

      <Prose>
        Generation reverses the training loop. Given a prompt, the model produces a distribution over the next token, a token is drawn from that distribution, it is appended to the sequence, and the model runs again. One step per token, no parallelism across positions, until a stop condition is reached.
      </Prose>

      <Prose>
        The sampling distribution is the main knob. Greedy decoding always picks the argmax — fast and deterministic, but prone to degenerate repetition. Temperature rescales the logits before softmax: divide by <Code>T &lt; 1</Code> to sharpen the distribution toward the most likely tokens, divide by <Code>T &gt; 1</Code> to flatten it toward uniformity. Top-k sampling restricts the distribution to the <Code>k</Code> highest-probability tokens and renormalizes. Top-p (nucleus) sampling instead keeps the smallest set of tokens whose cumulative probability exceeds <Code>p</Code>, adapting the candidate pool to the shape of each distribution rather than fixing its size. In practice these are stacked: temperature first, then nucleus or top-k. The deep mechanics of decoding strategies — beam search, speculative decoding, contrastive search — belong to the Inference Optimization topic.
      </Prose>

      <TokenStream
        label="autoregressive generation — prefix + sampled tokens"
        tokens={[
          { label: "The", color: "#888" },
          { label: " cat", color: "#888" },
          { label: " sat", color: "#888" },
          { label: " on", color: "#e2b55a" },
          { label: " the", color: "#e2b55a" },
          { label: " mat", color: "#e2b55a" },
          { label: ".", color: "#e2b55a" },
        ]}
      />

      <Prose>
        The grey tokens are the prompt; the gold tokens are sampled continuations. Each gold token was drawn from the model's output distribution conditioned on everything to its left. The model ran four separate forward passes to produce four tokens — one per step, each pass slightly longer than the last.
      </Prose>

      <H2>What next-token prediction does not tell you</H2>

      <Prose>
        Validation loss is a real signal and a partial one. A model that achieves 2.0 cross-entropy on held-out text is better at modeling that text distribution than one at 2.1. But the relationship between loss and capability is neither linear nor monotone in the ways that matter most. Emergent behaviors — long-horizon reasoning, code generation, instruction following, in-context learning — often appear sharply at some compute threshold rather than smoothly as the loss descends. A 2.1-loss model might fail entirely at chain-of-thought reasoning. A 2.0-loss model trained on a slightly different data mix might handle it fluently. The loss numbers are close; the behavioral gap is not.
      </Prose>

      <Prose>
        This is not a critique of the objective. It is a property of the metric. Cross-entropy on next-token prediction averages across every position in every sequence — common words, rare words, easy continuations, hard ones. A model can improve dramatically on the hard, high-signal positions while barely moving the aggregate loss, because those positions are numerically dominated by the easy ones. The loss curve tells you the model is improving. It does not tell you which capabilities are emerging, at what rate, or whether the next 10% compute investment will produce another quiet plateau or a sudden phase transition.
      </Prose>

      <Prose>
        This is exactly the gap that scaling laws address — empirical relationships between compute, data, model size, and downstream performance that give practitioners a more predictive handle on the pretraining arc than the loss curve alone. It is also why pretraining is evaluated not just by perplexity but by a battery of downstream benchmarks. The loss is necessary; it is not sufficient.
      </Prose>

      <Prose>
        There is also a structural blind spot. Cross-entropy rewards calibration, not correctness in any absolute sense. A model that assigns probability 0.6 to a factually wrong token and 0.3 to the right one is penalized less than a model that is confidently wrong. But from the downstream user's perspective, both made an error. Loss does not distinguish between "confidently correct," "uncertain," and "confidently wrong" in the way that matters for deployed systems. Reinforcement learning from human feedback, factual grounding, and chain-of-thought verification are all attempts to close this gap — to add supervision that cross-entropy cannot provide.
      </Prose>

      <Prose>
        Next-token prediction is the unreasonably effective algorithm that carried the field from GPT-2 to GPT-4. The next topic covers its complement — masked language modeling — and why the field mostly left it behind for generation tasks.
      </Prose>
    </div>
  ),
};

export default causalLanguageModeling;
