import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { TokenStream } from "../../components/viz";

const maskedLanguageModeling = {
  title: "Masked Language Modeling (BERT-style)",
  readTime: "9 min",
  content: () => (
    <div>
      <Prose>
        Before GPT-style autoregression took over, a different objective dominated: randomly mask 15% of tokens and train the model to fill them in. BERT, published by Devlin et al. in 2018, introduced it under the name Masked Language Modeling (MLM), and the entire encoder era — BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT — rode on it. The core intuition was that predicting hidden tokens forces the model to build deep, bidirectional representations of language, because every position in the context is fair game as evidence. That intuition was right, and it produced the best NLP models of 2019 and 2020. It is now mostly a historical footnote for generation, but it remains the right objective for understanding tasks: classification, named-entity recognition, coreference resolution, and retrieval.
      </Prose>

      <H2>The mechanics</H2>

      <Prose>
        During pretraining, BERT selects 15% of input tokens at random. Of those selected positions, 80% are replaced with the special <Code>[MASK]</Code> token, 10% are replaced with a random token drawn from the vocabulary, and 10% are left completely unchanged. The model receives this corrupted sequence and is trained to predict the original token at every masked position — a cross-entropy loss over the vocabulary, but only at the positions that were selected.
      </Prose>

      <TokenStream
        label="bert-style masked input (15% of tokens replaced)"
        tokens={[
          { label: "The", color: "#888" },
          { label: " cat", color: "#888" },
          { label: " [MASK]", color: "#f87171" },
          { label: " on", color: "#888" },
          { label: " the", color: "#888" },
          { label: " [MASK]", color: "#f87171" },
          { label: ".", color: "#888" },
        ]}
      />

      <Prose>
        The 80/10/10 split looks arbitrary, but it is carefully motivated. If you replaced every selected token with <Code>[MASK]</Code>, the model would learn to focus its representations almost entirely on positions that contain <Code>[MASK]</Code> — because that is the only signal that a prediction is required. At fine-tuning time, the model never sees <Code>[MASK]</Code> in real inputs, so there would be a train-test mismatch baked into the representations. The 10% random and 10% unchanged fractions force the model to maintain useful representations for every token, not just the masked ones, because it can never be sure which positions will require prediction. The model has to treat every token as potentially informative.
      </Prose>

      <Prose>
        One consequence worth noting: MLM is not a language model in the strict probabilistic sense. It does not define a tractable joint distribution over sequences that you can sample from. It is a discriminative objective — the model scores token candidates at fixed positions given a full (corrupted) context. This distinction matters for generation, but it is a feature rather than a bug for downstream understanding tasks where you want the best possible contextual representation of every token simultaneously.
      </Prose>

      <H2>Bidirectional attention</H2>

      <Prose>
        The architectural consequence of MLM is that the attention mask is dropped entirely. In a causal language model, each token can only attend to positions to its left — the mask is a strict lower triangle. BERT removes this constraint. Every token attends to every other token, in both directions, at every layer. This is what makes masked LM powerful for representation learning: the embedding of each token integrates information from both the left and right context simultaneously, through every attention head, all the way up the stack.
      </Prose>

      <Prose>
        Consider the word <em>bank</em> in two sentences: "She deposited money at the bank" and "The river bank was muddy." A causal model processing these left-to-right will have built a representation of <em>bank</em> before seeing the disambiguating words — it has to revise its interpretation in later tokens. A bidirectional model sees the full sentence simultaneously and can integrate the disambiguating evidence directly into the representation of <em>bank</em> at the first layer. The resulting embedding genuinely encodes the right sense of the word, which is why BERT-style models dominated NLP benchmarks the moment they appeared. Tasks that require understanding meaning in context — reading comprehension, natural language inference, sentiment — benefit directly from seeing both directions.
      </Prose>

      <Prose>
        The same property that makes bidirectional attention excellent for representations makes it awkward for generation. To generate a sequence autoregressively with bidirectional attention, you would need to predict token <em>i</em> while already attending to tokens at positions <em>i+1, i+2, ...</em> — which obviously requires knowing the future. Techniques exist to work around this (masked self-attention in specific heads, iterative refinement, BERT as a reranker over externally generated candidates), but none of them are clean fits. The architecture is shaped by the objective, and MLM shaped it for understanding.
      </Prose>

      <H2>Why it lost to autoregression for generation</H2>

      <Prose>
        Two pressures pushed the field toward causal LM. The first was scaling behavior. Kaplan et al. (2020) showed that causal LM loss on held-out text decreases as a smooth power law in compute, parameters, and data — a curve that showed no sign of flattening at the scales studied. MLM performance also improved with scale, but it hit diminishing returns earlier, particularly for tasks that required multi-step reasoning or long-form coherence. The capabilities of BERT-large and RoBERTa-large were real, but they did not compound the way GPT-3's capabilities did when the model was made ten times larger. The scaling law for MLM looked more like a logistic curve; the scaling law for causal LM looked like a line in log-log space, still steep, with no obvious ceiling.
      </Prose>

      <Prose>
        The second pressure was the breadth of causal LM as an objective. A model trained on causal LM can be prompted to do classification, summarization, translation, extraction, and generation — all from the same weights, with no fine-tuning, through careful prompting. BERT-style models require a task-specific head and fine-tuning on labeled data for almost every application. The encoder approach was not worse on any individual classification task; it was just not a universal interface. Once GPT-3 demonstrated that a sufficiently large autoregressive model could do classification in-context without any labeled data, the asymmetry became hard to ignore. Causal LM is a strict superset: everything you can do with a bidirectional encoder, you can do autoregressively, and generation comes for free.
      </Prose>

      <Callout accent="gold">
        BERT-style encoders are still state of the art for classification and retrieval at modest scale. They are not in the running for frontier LMs. These two facts coexist comfortably.
      </Callout>

      <H2>Where MLM still matters</H2>

      <Prose>
        Encoders are not dead. The sentence embedding models that power modern retrieval — BGE, E5, GTE, and their multilingual variants — are all fine-tuned encoder transformers. They are trained on MLM plus contrastive objectives (such as contrastive loss on positive-negative sentence pairs) and produce dense vector representations of text that are genuinely useful for semantic search, deduplication, reranking, and retrieval-augmented generation pipelines. Bidirectional context is an advantage here: the embedding of a passage should reflect the meaning of the entire passage, not just a directional prefix. A causal model generates tokens; an encoder compresses meaning. For compression, bidirectionality helps.
      </Prose>

      <Prose>
        DeBERTa-v3 and its descendants remain competitive or state of the art on many classification and natural language understanding benchmarks at modest scale — the GLUE and SuperGLUE leaderboards still list encoder models at the top for tasks like natural language inference and coreference resolution. The combination of disentangled attention (DeBERTa's architectural improvement over BERT) and a larger pretraining corpus has kept these models relevant despite being structurally incapable of generation. For tasks where you need the best possible understanding of a fixed-length input at low latency and low cost, an encoder fine-tuned on your task is often a better engineering choice than prompting a large generative model.
      </Prose>

      <H3>ELECTRA's twist</H3>

      <Prose>
        Worth noting briefly before moving on: ELECTRA (Clark et al., 2020) replaced the masked prediction objective with a more sample-efficient alternative called replaced token detection. A small generator model — trained with standard MLM — proposes token replacements for the input sequence. A discriminator model then receives the full sequence and predicts, at every single token position, whether each token is the original or a plausible but wrong replacement. Because every position contributes to the discriminator's loss — not just the 15% that were masked — ELECTRA extracts roughly four times as much gradient signal per token as BERT. At matched compute, the discriminator learns better representations faster.
      </Prose>

      <Prose>
        ELECTRA produced strong results and is still a reasonable choice for encoder pretraining. It did not unseat MLM as the dominant encoder objective in the short run, because the autoregressive wave was already cresting when the paper appeared — the field's attention had shifted to scaling causal LMs. In retrospect, ELECTRA solved a real efficiency problem in an era when that problem was about to become irrelevant for frontier research, though the insight remains interesting for anyone training modest-scale domain-specific encoders today.
      </Prose>

      <H2>The enduring picture</H2>

      <Prose>
        MLM is the objective that built BERT-era NLP and quietly still powers the retrieval half of most production RAG systems. It never figured out how to generate — not because the idea was wrong, but because generation requires a different kind of loss, and the autoregressive approach turned out to scale better and generalize more broadly. The encoder era was not a detour. It established that large-scale pretraining on unlabeled text could transfer to almost any NLP task, a fact that the decoder era inherited and extended. BERT's masked tokens trained the instinct; the GPT line took that instinct and scaled it until generation became the default interface for everything. The objective that built the first generation of capable language models is not what you reach for when you want the model to write — but it is still what you reach for when you want the model to understand.
      </Prose>
    </div>
  ),
};

export default maskedLanguageModeling;
