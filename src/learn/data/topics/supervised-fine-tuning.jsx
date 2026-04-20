import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const supervisedFineTuning = {
  title: "Supervised Fine-Tuning (SFT)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Pre-training gives you a model that can continue any text. It does not give you a model that follows instructions. A raw GPT-3 presented with "Translate to French: I am happy" will, with high probability, continue that string the way the internet would — another example in a language-exercise format, a second line of the worksheet, maybe a bilingual table header. It will not produce "Je suis heureux." That is not a failure of capability. The base model has the linguistic knowledge needed to translate. What it lacks is the disposition to respond as if the string were a request rather than a document to extend.
      </Prose>

      <Prose>
        Supervised fine-tuning is the first step in bridging that gap. Take the base model, collect a dataset of (prompt, desired response) pairs that represent the behavior you want, and train on them. The model learns to associate the format of a request with the format of a direct, on-task answer. Everything that follows in the post-training stack — RLHF, DPO, constitutional methods — builds on top of a model that has already been SFT'd. Without that alignment of surface format, preference learning has nothing useful to optimize over.
      </Prose>

      <H2>The mechanics — it's just more next-token prediction</H2>

      <Prose>
        Nothing structural changes. The architecture is identical to pre-training. The loss function is identical to pre-training — cross-entropy over the next token. What changes is the data and one small but critical detail about which tokens contribute to the loss.
      </Prose>

      <Prose>
        During pre-training, every token in the sequence contributes a gradient. The model learns to predict each word given all preceding words, regardless of whether those words are a question, a news headline, or a Python comment. During SFT, the training examples have two parts: a prompt <Code>x</Code> and a desired completion <Code>y</Code>. Loss is computed only on the completion tokens. The prompt tokens are masked out — they appear in the context window and condition the generation, but they contribute zero gradient. The logic is straightforward: you want the model to learn how to respond, not to learn how to reconstruct the prompt it was given.
      </Prose>

      <MathBlock>{"\\mathcal{L}_{SFT} = -\\mathbb{E}_{(x, y) \\sim \\mathcal{D}} \\left[\\sum_{t} \\log p_\\theta(y_t \\mid x, y_{<t}) \\right]"}</MathBlock>

      <Prose>
        The expectation is over demonstration pairs drawn from the curated dataset <Code>D</Code>. Each gradient step nudges the model toward assigning higher probability to the exact token sequence <Code>y</Code> given prompt <Code>x</Code>. Repeat over enough pairs, and the model learns the mapping from request formats to response formats.
      </Prose>

      <Prose>
        Implementing the mask is mechanical. Concatenate prompt and response, shift by one for the standard next-token target, then zero out the loss at every position that belongs to the prompt.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def sft_loss(model, prompt_ids, response_ids):
    """Compute next-token loss only on response tokens, not the prompt."""
    full = torch.cat([prompt_ids, response_ids], dim=-1)
    logits = model(full[:, :-1])
    targets = full[:, 1:]

    # Build a mask that is 0 on prompt positions, 1 on response positions.
    prompt_len = prompt_ids.size(-1)
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, prompt_len - 1:] = 1.0  # -1 because of the shift

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    return (loss * mask).sum() / mask.sum()`}
      </CodeBlock>

      <Prose>
        The <Code>prompt_len - 1</Code> offset accounts for the one-position shift: after slicing <Code>full[:, :-1]</Code> as inputs and <Code>full[:, 1:]</Code> as targets, the first response token appears at index <Code>prompt_len - 1</Code> in the target sequence. Getting this wrong by one position — a common bug — leaks a single prompt token into the loss, which is usually harmless but is technically incorrect.
      </Prose>

      <H2>The data problem</H2>

      <Prose>
        SFT's capability is upper-bounded by its demonstration data. The model can imitate what it is shown; it cannot exceed it. This makes the construction of the instruction-tuning dataset the most consequential design decision in the whole pipeline — more consequential, in most cases, than the choice of learning rate or the number of fine-tuning steps.
      </Prose>

      <Prose>
        Two lineages of SFT datasets have defined the field. The academic lineage — FLAN, T0, Super-NaturalInstructions — aggregated existing NLP benchmarks and reformatted them as instruction-following tasks. The breadth was their strength: thousands of task types, from sentiment classification to reading comprehension to summarization, all cast as natural language instructions. These datasets demonstrated that diversity of task format was the key ingredient: a model trained on 1,800 diverse tasks generalized better to held-out tasks than a model trained on 100 tasks with more examples each. The distillation lineage — Alpaca, Dolly, Vicuna — bootstrapped from GPT-3.5 and GPT-4 outputs, using stronger models as annotators. These datasets produced more conversational, open-ended demonstrations, but came with the ceiling that the student model cannot surpass the teacher.
      </Prose>

      <Prose>
        The LIMA paper (2023) sharpened this into a concrete claim: 1,000 carefully curated, genuinely diverse examples produced a model that compared favorably to models trained on datasets 1,000 times larger. The key word is genuinely. A thousand examples that all look like variations of "write an email in a professional tone" are not diverse in any meaningful sense. Diversity of task type, domain, format, length, and reasoning style — that is what makes a small dataset punch above its weight. Volume is no substitute for coverage.
      </Prose>

      <H3>Chat templates</H3>

      <Prose>
        Real chat models do not simply concatenate a prompt string and a response string. They format conversations using role-delimited templates that tell the model who is speaking and when a turn begins and ends. The template encodes structural information that the model learns to attend to during SFT — it is not cosmetic. Change the template at inference time and performance degrades, sometimes substantially, because the model has learned to condition on specific token patterns.
      </Prose>

      <CodeBlock>
{`<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>`}
      </CodeBlock>

      <Prose>
        This is ChatML, used by early OpenAI models and adopted widely afterward. The special tokens <Code>{"<|im_start|>"}</Code> and <Code>{"<|im_end|>"}</Code> are added to the tokenizer vocabulary during fine-tuning. The role tag — <Code>system</Code>, <Code>user</Code>, or <Code>assistant</Code> — appears on the same line as the opening delimiter. During training, the loss mask is applied by role: system and user turns are masked out, only assistant turns contribute gradients. The model learns to produce the assistant turn, conditioned on everything that preceded it, without being trained to reproduce the instructions it received. Llama 3 uses a similar convention with different special tokens; Gemma uses its own; the underlying principle is identical across all of them.
      </Prose>

      <H3>Multi-turn training</H3>

      <Prose>
        A single training example often contains a full conversation — multiple exchanges between user and assistant, not just one. Two conventions exist for how to handle the loss across turns. The standard approach trains on all assistant turns in the conversation simultaneously: in a five-turn conversation, all three assistant responses contribute to the loss in that single forward pass. This is compute-efficient and exposes the model to realistic conditioning during training, where early assistant turns form part of the context for later ones.
      </Prose>

      <Prose>
        The alternative — attributed in some Llama 3 runs — trains only on the final assistant turn. The motivation is that multi-turn SFT datasets are often constructed by chaining together independently good responses, and earlier assistant turns may have been written or edited without full awareness of what came after. Training on only the last turn avoids contaminating the model with assistant content that is potentially inconsistent or was produced without the later context in mind. Both approaches are defensible, and the empirical gap between them appears to be small on most benchmarks. The choice matters more when the dataset construction process is known to be imperfect — which is most of the time.
      </Prose>

      <H2>The catastrophic forgetting risk</H2>

      <Prose>
        Pre-training builds a broad, dense capability base: the model has seen enormous amounts of text across countless domains, registers, and languages. SFT introduces a strong distributional pull toward the style and content of the fine-tuning set. If that set is narrow, the model may drift — losing fluency in domains it rarely encounters in fine-tuning, forgetting capabilities that were present in the base model, or developing stylistic tics that the base model never had.
      </Prose>

      <Prose>
        Catastrophic forgetting is not hypothetical. Models fine-tuned heavily on one dialect or task type have been observed to degrade on unrelated benchmarks relative to the base checkpoint. The standard mitigations operate at the data and optimization levels. Including a replay fraction — a small percentage of diverse pre-training data mixed into the SFT batch — counteracts the distributional shift by keeping the gradient anchored to the original data distribution. Lower learning rates reduce the magnitude of each update, so the model moves less far from its initialization. Adapter-based approaches, particularly LoRA, freeze the original weights entirely and learn only low-rank perturbations, which limits how far the model's representations can shift and reduces the total number of parameters in play during fine-tuning. The next topic in this track covers parameter-efficient fine-tuning methods, including LoRA and prefix tuning, in detail.
      </Prose>

      <Callout accent="gold">
        A common failure mode: fine-tune on a narrow instruction dataset, then find that the model has lost coding ability, multilingual fluency, or long-form coherence. Adding even 5–10% diverse replay data to the SFT mix often recovers most of the regression.
      </Callout>

      <H2>SFT as a preference-agnostic baseline</H2>

      <Prose>
        SFT teaches a model what a good response looks like, by imitation. It does not teach the model why one response is better than another when both are plausible. That distinction is easy to miss but consequential.
      </Prose>

      <Prose>
        Given a factual question, SFT trains the model to produce a response that looks like the annotator's answer. If two annotators write two different responses to the same question — one more concise, one more thorough, one hedged and one confident — SFT assigns equal training signal to both, because it sees them each as a positive example to imitate. It has no mechanism for learning that users prefer one over the other, or that one is actually more accurate, or that one is safer in edge cases. This preference gap is structural: SFT is a cloning algorithm. It copies demonstrated behaviors, and it has no native ability to reason about the relative merit of alternatives it was not shown.
      </Prose>

      <Prose>
        This is the boundary where RLHF and DPO begin. Those methods operate not on single demonstrations but on preference pairs — two candidate responses to the same prompt, with a label indicating which one was preferred. They directly optimize the model toward producing responses that rank higher under human judgment, rather than toward producing responses that resemble a specific demonstrator. SFT produces capable models. Preference methods produce useful ones. In practice, every modern assistant is SFT-then-preference, because each stage solves a distinct problem the other cannot.
      </Prose>

      <Prose>
        SFT gets the behaviors roughly into the right shape — the model learns to respond rather than continue, to adopt the right format, to engage with the instruction rather than the document context. What it cannot install is the judgment to distinguish between two responses that both fit the format but differ in quality, safety, or user preference. That ranking signal — the harder, richer information — is what the next topic introduces.
      </Prose>
    </div>
  ),
};

export default supervisedFineTuning;
