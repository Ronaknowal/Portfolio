import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream } from "../../components/viz";

const pTuningSoftPrompts = {
  title: "P-Tuning & Soft Prompt Methods",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        Prompts are discrete — sequences of tokens drawn from the vocabulary. That constraint is stranger than it first appears. The space of "good prompts" is tiny relative to the space of all possible token sequences, gradient descent cannot operate on token IDs directly, and a single word swap can produce dramatic swings in model behavior. Engineers compensate with prompt engineering: trial and error, handcrafted templates, sensitivity analysis. The process works well enough, but it leaves capability on the table. If prompting is essentially a search problem, why not search in a better space?
      </Prose>

      <Prose>
        Soft prompts relax the discreteness constraint. Instead of selecting tokens from the vocabulary, the idea is to directly optimize continuous embedding vectors that occupy token slots — letting gradient descent do the search in a smooth, high-dimensional space where standard optimizers are comfortable. P-Tuning (Liu et al., 2021), Prefix Tuning (Li and Liang, 2021), and Prompt Tuning (Lester et al., 2021) are three variations on this theme. They emerged nearly simultaneously and cover slightly different points in the design space.
      </Prose>

      <H2>What a soft prompt is</H2>

      <Prose>
        A normal prompt is a list of token IDs. Before the first transformer block, each ID is looked up in the embedding table to produce a dense vector of dimension <Code>d</Code>. The sequence of those vectors is what the model actually processes. A soft prompt short-circuits the lookup: instead of selecting token IDs and converting them to embeddings, you maintain <Code>k</Code> embedding vectors directly as learnable parameters, prepend them to the real input's embeddings, and train them end-to-end while the rest of the model stays frozen.
      </Prose>

      <Prose>
        Formally, the input to the first transformer block becomes:
      </Prose>

      <MathBlock>{"[\\mathbf{p}_1, \\mathbf{p}_2, \\ldots, \\mathbf{p}_k, E(x_1), \\ldots, E(x_n)]"}</MathBlock>

      <Prose>
        where <Code>{"p_i ∈ ℝᵈ"}</Code> are learned parameters and <Code>E(x)</Code> is the frozen embedding of real token <Code>x</Code>. The soft tokens are not pinned to anything in the vocabulary — they live in the full continuous embedding space and can take values that no discrete token would ever occupy. That freedom is exactly the point.
      </Prose>

      <TokenStream
        label="soft prompt layout — learned vectors then real input"
        tokens={[
          { label: "<soft_1>", color: "#c084fc" },
          { label: "<soft_2>", color: "#c084fc" },
          { label: "<soft_3>", color: "#c084fc" },
          { label: "<soft_4>", color: "#c084fc" },
          { label: "Translate", color: "#e2b55a" },
          { label: " to", color: "#e2b55a" },
          { label: " French:", color: "#e2b55a" },
          { label: " Hello", color: "#e2b55a" },
        ]}
      />

      <Prose>
        The purple tokens in the visualization above do not correspond to any word in the vocabulary. They are continuous vectors that the optimizer shaped, over many gradient steps, to steer the frozen model toward the target behavior. The yellow tokens are the actual input — passed through the embedding table in the usual way and treated as read-only by the optimizer.
      </Prose>

      <H2>Prompt Tuning — the simplest form</H2>

      <Prose>
        Lester, Al-Rfou, and Constant (2021) described the most minimal version: prepend <Code>k</Code> learnable vectors, freeze all model weights, train with the standard cross-entropy loss on labeled examples. The number of trainable parameters is exactly <Code>k × d</Code> — for a 20-token prompt on a 4096-dimensional model, that is about 80,000 floats. For comparison, fine-tuning a 7B model touches seven billion.
      </Prose>

      <Prose>
        The headline finding from that paper is scale-dependent: at 11B parameters (T5-XXL), prompt tuning nearly matched full fine-tuning on the SuperGLUE benchmark. At smaller scales — 250M or 780M parameters — it fell noticeably short. This dependence on model size is the central limitation of the pure prompt-tuning approach. The model has to already be capable of performing the task; the soft prompt is pointing it, not teaching it. A smaller model that lacks the underlying capability cannot be pointed into one.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class PromptTuned(nn.Module):
    def __init__(self, base_model, prompt_length=20, hidden_dim=4096):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters(): p.requires_grad = False

        # Learned soft prompt — the only trainable parameters.
        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_length, hidden_dim) * 0.01
        )

    def forward(self, input_ids):
        embeds = self.base.embed(input_ids)           # (B, L, D)
        soft = self.soft_prompt.unsqueeze(0).expand(embeds.size(0), -1, -1)
        full = torch.cat([soft, embeds], dim=1)
        return self.base.transformer(inputs_embeds=full)`}
      </CodeBlock>

      <Prose>
        The implementation maps cleanly to the concept. The base model's parameters are frozen immediately after initialization. The soft prompt is a single <Code>Parameter</Code> — a matrix of shape <Code>(prompt_length, hidden_dim)</Code> initialized near zero to avoid large perturbations at the start of training. At forward time, the prompt is broadcast across the batch and concatenated with the input embeddings before the transformer body sees anything. Gradient flow touches only <Code>self.soft_prompt</Code>.
      </Prose>

      <H2>Prefix Tuning — reach deeper</H2>

      <Prose>
        Prompt tuning inserts learned vectors at the input embedding layer and relies on the transformer's depth to propagate that signal forward. Li and Liang (2021) argued that this single-layer injection is too indirect for smaller models: by the time the signal has passed through a dozen attention layers, it has been diluted. Their alternative — prefix tuning — inserts learnable vectors into every attention layer, not just the first.
      </Prose>

      <Prose>
        Specifically, prefix tuning prepends trainable key and value vectors to the key and value matrices of every multi-head attention layer. If there are <Code>L</Code> layers and <Code>k</Code> prefix tokens, the trainable parameter count scales as <Code>2 × L × k × d</Code> — two to three orders of magnitude larger than a single-layer soft prompt for typical architectures. That higher parameter count comes with broader coverage: prefix tuning works well even at moderate model sizes where pure prompt tuning struggles.
      </Prose>

      <Prose>
        One practical note from the original paper: training raw prefix vectors directly was unstable. Li and Liang found it necessary to reparameterize — a small MLP maps a lower-dimensional vector to the full prefix at each layer during training, and the MLP is discarded at inference. This is essentially a smoother optimization surface, not a change to the forward-pass computation.
      </Prose>

      <H3>P-Tuning v1 and v2</H3>

      <Prose>
        Liu et al. (2021) introduced P-Tuning v1 at roughly the same time with a slightly different motivation. Rather than prepending raw learned vectors, they used an LSTM or lightweight MLP to encode a sequence of virtual token embeddings, on the hypothesis that the recurrent structure would produce better-conditioned gradients than training independent vectors. The architecture is more complex but the idea is identical: learn a continuous prompt, not a discrete one.
      </Prose>

      <Prose>
        P-Tuning v2 (2022) stepped back from the LSTM framing and converged on something much closer to prefix tuning — deep prompts inserted at every layer, tuned without the encoder wrapper. The paper showed this configuration matching full fine-tuning on NLU tasks at model sizes as small as 300M parameters, which the original prompt tuning paper could not achieve. In current usage, "P-Tuning," "P-Tuning v2," and "prefix tuning" are often treated as the same family: continuous vectors prepended at one or more layers, everything else frozen.
      </Prose>

      <H2>Why this still matters in the LoRA era</H2>

      <Prose>
        LoRA has largely won the parameter-efficient fine-tuning race for LLMs. It is more flexible, performs better across scales, easier to compose with other adaptations, and applicable to any weight matrix in the model. For most fine-tuning tasks on accessible hardware, LoRA is the default choice and there is no strong reason to look elsewhere. Soft prompts do not beat it on quality at equivalent parameter budgets.
      </Prose>

      <Prose>
        There are still narrow but real cases where soft prompts are the right tool. When you want to fine-tune for a large number of small, independent tasks — different clients, different domains, different personas — and load them on demand, a soft prompt for each task is a handful of kilobytes. Swapping task-specific behavior becomes a vector concatenation, not a weight matrix reload. When you have no access to model weights at all, some hosted API endpoints expose a "virtual token" or "tuned prompt" interface that is exactly this mechanism under a different name. And when serving simplicity matters, there is nothing simpler than appending a fixed vector prefix at inference time — no adapter modules, no weight merging, no extra infrastructure.
      </Prose>

      <H3>Limitations</H3>

      <Prose>
        Soft prompts are sensitive to initialization in ways that discrete prompts are not. Random initialization near zero is unstable for small models. Initializing from the embeddings of real tokens — words that are semantically related to the task — reliably produces better results, but requires knowing something about the task before training starts. Vocabulary-sampled initialization is a reasonable middle ground: draw random token embeddings from the existing vocabulary as starting points, rather than drawing from a Gaussian.
      </Prose>

      <Prose>
        Composition is the other structural limitation. If you independently tune a soft prompt for task A and a soft prompt for task B, concatenating them does not generally give you a model that handles both tasks. The two sets of vectors were optimized in isolation and interact unpredictably when combined. This means soft prompts do not scale naturally to multi-task settings the way modular adapters do — each task requires its own dedicated prompt with no sharing of learned structure.
      </Prose>

      <Prose>
        Soft prompts also tend to plateau on generative tasks. They work well for classification, extraction, and structured prediction where the base model's generation capability is mostly irrelevant. Open-ended generation — long-form text, reasoning chains, creative output — benefits more from modifications that reach deeper into the model's residual stream, which is why prefix tuning outperforms pure prompt tuning in generation tasks and why LoRA, which modifies attention and MLP weight matrices directly, outperforms both.
      </Prose>

      <Callout accent="gold">
        Soft prompts work best when the base model already knows how to do the task — you're pointing it, not teaching it.
      </Callout>

      <Prose>
        Soft prompting was an early and influential demonstration that fine-tuning does not require updating billions of parameters — that a small learned perturbation at the input can redirect a frozen model's behavior across a wide range of tasks. That insight reoriented the PEFT research agenda and informed later work on adapters, LoRA, and IA³. The broader landscape of parameter-efficient methods — where each technique stands relative to the others, the trade-offs between trainable parameter count and task coverage, and the current practical defaults — is covered under Model Optimization & Efficiency.
      </Prose>
    </div>
  ),
};

export default pTuningSoftPrompts;
