import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { Heatmap, Plot, TokenStream } from "../../components/viz";
import { colors } from "../../styles";

const vocabularyMultilingual = {
  title: "Vocabulary Design & Multilingual Tokenization",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Vocabulary size isn't a free parameter. It trades embedding-table memory against sequence length, and it silently penalizes some languages more than others. The knob looks like a hyperparameter; it's actually a political decision about whose language compresses well. Pick a vocabulary of 32,000 tokens and you've implicitly decided that English — and the dozen or so European languages that decompose similarly under BPE — will get short, efficient sequences. Every other script pays the difference in tokens, in latency, and in API spend. That tradeoff is baked in before the model sees its first gradient, and it cannot be fixed by fine-tuning. You live with the vocabulary you train.
      </Prose>

      <H2>The vocab-size knob</H2>

      <Prose>
        The embedding table costs exactly <Code>vocab_size × d_model</Code> parameters. For a model with a 4096-dimensional hidden size, 32,000 tokens means 128 million parameters just for the lookup table. Push the vocabulary to 128,000 tokens and that becomes 512 million. Gemma 2's 256,000-token vocabulary hits roughly one billion parameters in embeddings alone — before a single attention head or feedforward weight is counted. Those parameters aren't free, but they pay a dividend. Every token added to the vocabulary that gets used is a merge that compresses the input: shorter sequences mean fewer attention steps, less KV-cache memory, and more text fitting inside a fixed context window. The decision is a tradeoff between parameter budget and compute budget, and different model families have landed at very different points on that curve.
      </Prose>

      <Prose>
        The generational shift is visible in round numbers. BERT-era models sat at 30,000–32,000 tokens — small enough that the embedding table was negligible relative to 12 stacked transformer blocks. GPT-2 moved to 50,257, GPT-3 stayed at 50,257, both justified by a modest improvement in English compression. Then the frontier jumped. GPT-4 uses roughly 100,000 tokens. Llama 3 ships 128,000. Gemma 2 reaches 256,000. The jump isn't accidental: these models were explicitly designed to handle many languages at once, and the only way to keep fertility reasonable across scripts with radically different character inventories is to give each script enough vocabulary real estate that its words don't shatter into bytes.
      </Prose>

      <Plot
        label="vocab size vs. avg tokens per 1k english words"
        width={480}
        height={220}
        xLabel="vocab size"
        yLabel="tokens / 1k words"
        series={[
          { name: "English", points: [[16000, 1850], [32000, 1520], [50000, 1380], [100000, 1260], [128000, 1230], [200000, 1190], [256000, 1175]] },
        ]}
      />

      <H2>Fertility — the metric that matters</H2>

      <Prose>
        Fertility is the average number of tokens a tokenizer produces per input word. It is the cleanest single number for comparing tokenizer efficiency across languages. An English-trained tokenizer typically hits a fertility around 1.3 on English prose: most common words are single tokens, longer words split into two or three recognizable pieces. Run that same tokenizer over Hindi and the fertility jumps to 3–5. The tokenizer was never shown enough Devanagari to build useful merges for it, so each word fragments into subword pieces that range from meaningful morphemes to accidental byte-level residue. Petrov et al. (2023), "Language Model Tokenizers Introduce Unfairness," documented this gap systematically across dozens of languages and tokenizer families and found that it persists even in models nominally described as multilingual.
      </Prose>

      <Heatmap
        label="tokens per word by language (approx, lower = more efficient)"
        rowLabels={["English", "Spanish", "French", "German", "Portuguese", "Hindi", "Arabic", "Korean", "Japanese", "Chinese"]}
        colLabels={["GPT-4", "Llama 3", "Gemma 2"]}
        matrix={[
          [1.31, 1.28, 1.25],
          [1.54, 1.42, 1.36],
          [1.59, 1.45, 1.38],
          [1.78, 1.62, 1.52],
          [1.61, 1.48, 1.39],
          [4.12, 2.81, 1.98],
          [3.21, 2.44, 1.87],
          [3.02, 2.26, 1.74],
          [2.64, 2.11, 1.68],
          [2.81, 2.05, 1.62],
        ]}
        cellSize={50}
      />

      <Prose>
        The numbers in the heatmap are what you actually pay. GPT-4's column reflects the token counts you see on an OpenAI invoice: Hindi costs 4.12× per word what English does. Gemma 2's column looks better across the board — not because Google found a trick, but because the model was explicitly designed with multilingual coverage in mind and trained with heavy non-English oversampling. A 4× fertility gap is a 4× bill. It is also a 4× context-window tax: the same document in Hindi fills the context window four times as fast as in English, leaving four times less room for retrieved context, tool outputs, or conversation history. Gemma 2's improvement to ~2× for Hindi is real, but it still isn't parity, and it was purchased with a vocabulary that costs one billion parameters in embeddings.
      </Prose>

      <H2>The English tax</H2>

      <Prose>
        The fertility gap translates directly into a regressive tax on non-English speakers. A Hindi speaker using a GPT-4-class API for the same task as an English speaker pays 3–5× more per call. Their context window fills 3–5× faster, so they need to truncate documents more aggressively or upgrade to a longer-context tier at higher cost. Per-token latency is measured in time-to-first-token and tokens-per-second; both are identical in absolute terms, but when you need 400 tokens to express what an English speaker says in 100, the wall-clock experience is 4× slower. This is not an abstraction. For users in India, the Middle East, East Asia — regions where LLM adoption is growing fastest — the token-level cost structure is tilted against them in proportion to how far their script diverges from Latin.
      </Prose>

      <Prose>
        The English tax is structural, not incidental. It emerges from the same choice that made BPE efficient for English in the first place: the merge list was learned from a corpus that was mostly English, so the merges that survived to produce common tokens are the ones that compress English well. The vocabulary wasn't designed to be unfair; it was designed to be efficient, and efficient against that corpus meant English. Every subsequent training run that inherits the same tokenizer inherits the same bias. Changing it requires retraining the tokenizer on a different corpus — and then retraining the model, because the embedding matrix is non-transferable. No amount of multilingual fine-tuning fixes a tokenizer that fragments your language into four tokens per word.
      </Prose>

      <H2>Rebalancing during training</H2>

      <Prose>
        The fertility imbalance comes from the pretraining corpus mix. A web crawl in 2020 was roughly 90% English by token count; Common Crawl improvements have pushed that down, but English still dominates any unfiltered web scrape by a wide margin. The standard fix is temperature-scaled sampling: for each language <Code>l</Code> with natural corpus probability <Code>p_l</Code>, sample instead at rate proportional to <Code>p_l^α</Code>, where <Code>α &lt; 1</Code>. Setting <Code>α = 0.3</Code> aggressively oversamples low-resource languages — a language with 0.1% natural frequency ends up contributing something closer to 1–2% of training tokens. The cost is a modest degradation in English quality, because the model sees proportionally less English. Typical values range from <Code>α = 0.3</Code> for heavily multilingual models to <Code>α = 0.7</Code> for models that want broad coverage without sacrificing English performance. mT5, XLM-R, and Gemma all use variants of this scheme; the exact <Code>α</Code> is usually treated as a tunable hyperparameter found by grid search over downstream evaluation benchmarks.
      </Prose>

      <CodeBlock language="python">
{`from collections import Counter

def fertility(tokenizer, corpus_by_lang):
    """
    tokenizer: any object with .encode(text) -> list of ids
    corpus_by_lang: dict[str, str] mapping language -> sample text
    Returns: dict[str, float] mapping language -> tokens-per-word ratio
    """
    result = {}
    for lang, text in corpus_by_lang.items():
        words = text.split()
        tokens = tokenizer.encode(text)
        result[lang] = len(tokens) / max(1, len(words))
    return result

# Usage — comparing two tokenizers on the same multilingual sample:
samples = {"en": english_text, "hi": hindi_text, "zh": chinese_text}
print("GPT-4:   ", fertility(gpt4_tok, samples))
print("Llama 3: ", fertility(llama3_tok, samples))`}
      </CodeBlock>

      <H3>Script fragmentation — the structural problem</H3>

      <Prose>
        Temperature-scaled sampling improves corpus balance, but it cannot fix a problem that lives one level deeper: some scripts are structurally resistant to subword tokenization regardless of how much training data you throw at them. CJK writing is logographic — each character is a complete unit of meaning, there are no spaces between words, and a vocabulary of 100,000 tokens can include individual CJK characters as single tokens but still cannot know where word boundaries fall without additional segmentation logic. Indic scripts introduce a different problem: Devanagari akshara (syllabic units) do not map one-to-one onto Unicode codepoints. A single perceived letter can span two or three codepoints once combining marks and matras are accounted for, so a BPE tokenizer working on Unicode characters will occasionally slice through the middle of what a reader perceives as a single glyph. Arabic adds directionality and contextual shaping to the mix: the same Unicode character has different rendered forms depending on its position within a word, and a tokenizer that was never exposed to enough Arabic may produce merges that cut across contextually distinct forms without knowing it. None of these problems are fatal — production models do handle all three script families — but they are why higher-vocabulary multilingual models still show elevated fertility for these languages even after rebalancing. The vocabulary ceiling isn't high enough, or the merges weren't learned from enough of the right data, to close the gap fully.
      </Prose>

      <H2>What real frontier models chose</H2>

      <Prose>
        The vocabulary choices of production frontier models are worth reading as a survey of the tradeoffs in practice. Llama 3 uses a 128,000-token SentencePiece BPE vocabulary trained on a diverse Common Crawl slice, the most visible jump Meta made from Llama 2's 32,000. Gemma 2 extends further to 256,000 tokens with SentencePiece and deliberate non-English oversampling, buying down fertility across most scripts at the cost of a large embedding table. Qwen2 lands at 152,000 tokens with explicit Chinese optimization — its merge list was seeded with a much larger share of Chinese text than Llama's, and it shows in per-character fertility for CJK. DeepSeek was designed from the start as a bilingual model — English and Chinese — and its tokenizer reflects that: Chinese words and common Chinese multi-character compounds appear as single tokens in ways that a vocabulary trained on English-heavy data would never produce. Mistral's base model inherited Llama's 32,000-token vocabulary, which, at the scale Mistral operates and the number of languages it's expected to handle, is increasingly a liability — a model pushing frontier performance benchmarks, constrained by a vocabulary designed in the BERT era.
      </Prose>

      <Prose>
        The vocabulary itself doesn't determine a model's multilingual quality — training data, instruction tuning, and RLHF all matter — but it is the hard constraint that those other factors have to work around. A model fine-tuned on multilingual instruction data with a 32,000-token vocabulary is fighting the tokenizer on every non-English forward pass. The frontier has largely converged on the judgment that 100,000 to 256,000 tokens is the right operating range for a model that takes multilinguality seriously. Below that, the fertility gap is too large to tune away. Above that, the embedding table starts dominating parameter counts in ways that aren't justified by the marginal compression improvement.
      </Prose>

      <Prose>
        Three topics in, the arc of text tokenization is mostly closed. BPE produces the merges; byte-level coverage removes the failure modes; vocabulary size and corpus rebalancing determine how fairly those merges distribute across scripts. The remaining inefficiencies — script fragmentation, digit tokenization, code identifiers — are real but bounded. The more interesting question now is what changes when the input isn't text at all: when the model needs to read images, audio, video, or structured sensor data through the same interface it uses for words. The abstraction that made subword tokenization work for language turns out to have a natural generalization, and it cuts in surprising directions.
      </Prose>
    </div>
  ),
};

export default vocabularyMultilingual;
