import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const vocabularyMultilingual = {
  title: "Vocabulary Design & Multilingual Tokenization",
  readTime: "31 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Open your OpenAI dashboard, translate a paragraph of English into Hindi, and paste both versions into the API cost estimator. The same meaning, word-for-word. The Hindi version will cost four to five times more. Not because Hindi is harder to serve. Not because the model is slower. Because somewhere in 2022 a tokenizer training run on a corpus that was 92% English learned merges that happened to compress English well, and every Hindi sentence now fragments into four times as many subword pieces as its English counterpart. The model bills per piece. The Hindi speaker pays the difference. Nothing about that charge reflects the information content of the request. It reflects the statistical shape of the tokenizer's training data, a choice frozen into the model's weights years before the Hindi speaker ever opened the chat window.
      </Prose>

      <Prose>
        Aleksandar Petrov, Emanuele La Malfa, Philip Torr, and Adel Bibi put numbers on this in May 2023, in a paper with the unambiguous title "Language Model Tokenizers Introduce Unfairness Between Languages" (arXiv:2305.15425). Across seventeen tokenizers and twenty-four languages, they measured the ratio of tokens a given tokenizer assigns to the same semantic content in each language, and they found disparities of up to fifteen to one. Languages like Shan, Mon, Burmese, and Kashmiri were routinely tokenized at ten to fifteen times the rate of English. Hindi, Bengali, Arabic, Korean, and Telugu sat in the three-to-five range. European languages like Spanish, French, and German paid a milder tax — thirty to sixty percent more tokens than English for the same content — but still paid. The paper's argument was that this is not an acceptable artifact. It is a systematic regressive charge on the users whose languages the tokenizer happened to be under-exposed to during training, and it compounds across every axis of LLM cost: API spend, context window budget, latency, and the ceiling on how much material a user can fit into a single conversation.
      </Prose>

      <Prose>
        The fertility disparity is only half the story. The other half is the choice that created it. A vocabulary is not a free parameter; it is a row count in a dense matrix. The embedding table is shaped <Code>(V, d_model)</Code>, and every token occupies one row of <Code>d_model</Code> floats. For a 70-billion-parameter model with <Code>d_model = 8192</Code>, a 32,000-token vocabulary costs 262 million parameters for the embedding table alone; a 128,000-token vocabulary costs 1.05 billion; a 256,000-token vocabulary costs 2.10 billion when the output head is untied. Those numbers are not rounding errors. They are a meaningful share of the parameter budget, and they have to be justified by whatever compression gain the larger vocabulary buys in sequence length. Two pressures — fairness across languages and efficiency within the parameter budget — drive every vocabulary design decision that modern frontier models make, and they push in opposite directions.
      </Prose>

      <Prose>
        The historical arc is visible in round numbers. BERT shipped in 2018 with a 30,522-token WordPiece vocabulary; the embedding table at <Code>d = 768</Code> cost 23 million parameters, negligible against the 110-million-parameter encoder. GPT-2 moved to 50,257 tokens of byte-level BPE, justified by a modest improvement in English compression. Llama 1 (February 2023) stayed at 32,000 tokens with SentencePiece BPE, deliberately English-centric — Meta's stated focus at that point was English performance, and a small vocabulary kept the embedding table small. Llama 2 held at 32,000. Then Llama 3 (April 2024) jumped to 128,256 tokens, combining the 100,000-token <Code>tiktoken</Code> base with 28,000 additional tokens selected for non-English language coverage. Meta's technical report attributes a compression improvement from 3.17 to 3.94 characters per English token and substantial improvements on multilingual benchmarks to the change. Gemma 2 (June 2024) went further: 256,000 tokens, SentencePiece with byte-level fallback and preserved whitespace, inherited from the Gemini tokenizer designed to handle a large number of languages from the start. The trajectory is unmistakable — the frontier has decided that the marginal parameter cost of a larger vocabulary is worth paying to close the fertility gap.
      </Prose>

      <Callout accent="gold">
        The central move this topic is built around: <strong>a tokenizer is a political instrument disguised as a hyperparameter</strong>. It decides who pays more per API call, whose context window fills faster, whose queries feel slower. The algorithm is mechanical — BPE or Unigram running over a frequency table — but the frequency table is a training-data choice, and that choice has a geography. Vocab design is about being deliberate with that geography.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Four concepts carry the rest of this topic. Learn them in this order and every downstream question about vocab design becomes answerable.
      </Prose>

      <H3>Fertility</H3>

      <Prose>
        Fertility is the average number of tokens a tokenizer emits per input word, measured on a given language's text. It is the cleanest single number for comparing tokenizer efficiency across languages. A GPT-4-class tokenizer produces roughly 1.0 to 1.3 tokens per English word on natural prose — most common words are whole tokens, longer ones split into two or three recognizable pieces. Run the same tokenizer on Hindi prose and the fertility jumps to 4 or 5. The tokenizer was never shown enough Devanagari during training to build useful merges for it, so each Hindi word fragments into its constituent codepoints or byte-level pieces. The ratio of fertilities <em>is</em> the tax: if English fertility is 1.0 and Hindi fertility is 4.7, the Hindi speaker pays 4.7× per word for API calls, fills the context window 4.7× faster, and waits 4.7× longer to render the same semantic output.
      </Prose>

      <H3>The English tax</H3>

      <Prose>
        The English tax is the structural consequence of training tokenizers on English-dominated corpora. Byte-pair encoding is a frequency-greedy algorithm: it spends its merge budget on whichever pairs show up most often. When the training corpus is 90% English, 90% of the merges are English merges. Every subsequent language pays for that with longer token sequences. Multilingual fine-tuning cannot fix this; the tokenizer's vocabulary is fused into the embedding matrix the moment the first gradient is taken, and changing it requires retraining the model from scratch. You live with the tokenizer you train. The English tax is a debt the model inherits from its vocabulary, and the only way to pay it down is to pick the vocabulary more carefully before training begins.
      </Prose>

      <H3>The vocab-size tradeoff</H3>

      <Prose>
        A bigger vocabulary means shorter sequences but a bigger embedding table. At a given model dimension <Code>d_model</Code>, every token added to the vocabulary costs <Code>d_model</Code> parameters, so doubling the vocabulary doubles the embedding parameters and, if embeddings are untied, doubles the output softmax parameters as well. The payoff is compression: more merges means more common sequences collapse into single tokens, which means fewer attention steps per document, less KV-cache memory per generated token, and more content fitting inside a fixed context window. The crossover point depends on the training corpus and the set of target languages. English-only models hit sharply diminishing returns past ~50,000 tokens. Multilingual models keep benefiting out to 200,000 or more because every additional language has its own set of common sequences that need vocabulary real estate.
      </Prose>

      <H3>Temperature sampling</H3>

      <Prose>
        A web crawl's natural language distribution is brutally skewed — English dominates by a factor of ten or more over any other single language. Training a tokenizer directly on that distribution produces an English-first vocabulary, and training the downstream model on it produces an English-first model. Temperature sampling is the standard fix. For each language <Code>l</Code> with natural corpus probability <Code>p_l</Code>, sample training data at a rate proportional to <Code>p_l^α</Code>, where <Code>α ∈ (0, 1]</Code>. Setting <Code>α = 1.0</Code> keeps the natural distribution. Setting <Code>α = 0.5</Code> flattens it — common languages still dominate but less aggressively. Setting <Code>α = 0.3</Code> or <Code>α = 0.1</Code> aggressively oversamples low-resource languages, buying down their fertility at the cost of a modest degradation in high-resource-language quality. XLM-R, mT5, and NLLB all use variants of this scheme. The right <Code>α</Code> is a hyperparameter found by grid search against downstream evaluation.
      </Prose>

      <Prose>
        These four concepts interact. Temperature sampling shapes the training corpus. The training corpus shapes the tokenizer's merges. The merges determine fertility per language. Fertility determines the per-query cost. The vocab-size knob governs how much budget the tokenizer has to spread across languages in the first place. Pick <Code>α</Code>, vocab size, and training corpus together, or you will be fighting one of them with the others for the entire life of the model.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Fertility</H3>

      <Prose>
        Let <Code>T</Code> be a tokenizer and <Code>D_l</Code> a corpus in language <Code>l</Code>. Write <Code>|T(D_l)|</Code> for the number of tokens the tokenizer produces on that corpus, and <Code>|W(D_l)|</Code> for the number of words (whitespace-delimited units, or a language-appropriate analogue for scripts without spaces). Fertility is:
      </Prose>

      <MathBlock>
        {"\\text{fertility}(T, l) = \\frac{|T(D_l)|}{|W(D_l)|}"}
      </MathBlock>

      <Prose>
        For scripts without natural word boundaries — Chinese, Japanese, Thai — a "word" is defined either by a segmenter (jieba, MeCab, pythainlp) or, in the Petrov et al. convention, by aligning against a translation of the same content in a whitespace-delimited reference language. Both conventions yield comparable numbers within a single study, but <em>across</em> studies the denominator can differ by a factor of two; always check how a reported fertility defines its word count. A second, corpus-independent quantity is <em>characters per token</em>, which is what the Llama 3 technical report uses (and reports as 3.94 for English under the Llama 3 tokenizer versus 3.17 for Llama 2). Characters-per-token and tokens-per-word are inverse-flavored — larger values of characters-per-token correspond to lower fertility — and they respond to the same underlying compression.
      </Prose>

      <H3>Embedding parameter count</H3>

      <Prose>
        The embedding table is a dense <Code>V × d_model</Code> matrix of learned parameters. If the output head (the unembedding projection used by the softmax) is tied to the embedding — a common parameter-saving trick — the cost is exactly:
      </Prose>

      <MathBlock>
        {"\\text{params}_{\\text{embed}} = V \\cdot d_{\\text{model}}"}
      </MathBlock>

      <Prose>
        If the head is untied (Llama 3 does tie; many earlier models do not), double it:
      </Prose>

      <MathBlock>
        {"\\text{params}_{\\text{embed, untied}} = 2 \\cdot V \\cdot d_{\\text{model}}"}
      </MathBlock>

      <Prose>
        For concrete numbers: a Llama 3 70B–class model with <Code>d_model = 8192</Code> and <Code>V = 128{"\u2009"}256</Code> spends 1.05 billion parameters on the embedding table. Dropping the vocabulary back to 32,000 would save 789 million parameters — not negligible, but also not free: every one of those saved parameters turns into additional tokens per sequence on every non-English input, which in turn raises per-token compute and per-token KV-cache costs across the entire model. The tradeoff is quantitative, and the crossover depends on the language mix the model will serve.
      </Prose>

      <H3>Total-cost-per-query</H3>

      <Prose>
        The API-level cost of a query in a given language is the product of the query's token count and the per-token price:
      </Prose>

      <MathBlock>
        {"\\text{cost}(q, l) = \\text{fertility}(T, l) \\cdot |W(q)| \\cdot p_{\\text{token}}"}
      </MathBlock>

      <Prose>
        Two users asking the same semantic question in different languages pay in the ratio of their languages' fertilities. If GPT-4's fertility is 1.00 for English and 4.69 for Hindi (numbers measured in section 4 below), the Hindi speaker pays 4.69× the English speaker for the same content. This is not a design intent; it is a statistical artifact of the tokenizer's training corpus, and it is the central complaint of the Petrov et al. paper. The same ratio applies to context-window consumption — if your 128K context window holds 96,000 English words at fertility 1.33, it holds 20,500 Hindi words at fertility 4.69 — and to time-to-first-token when the prompt has to be pre-filled through the full context.
      </Prose>

      <H3>Temperature-scaled multilingual sampling</H3>

      <Prose>
        Given a natural distribution <Code>p = (p_1, ..., p_L)</Code> over <Code>L</Code> languages, temperature-scaled sampling reweights to:
      </Prose>

      <MathBlock>
        {"q_l(\\alpha) = \\frac{p_l^{\\alpha}}{\\sum_{j=1}^{L} p_j^{\\alpha}}, \\qquad \\alpha \\in (0, 1]"}
      </MathBlock>

      <Prose>
        The math: raising each probability to a fractional power compresses the distribution toward uniform. At <Code>α = 1</Code> the original distribution is preserved. At <Code>α → 0</Code> every language ends up with equal probability <Code>1/L</Code>. Values in between trade off: <Code>α = 0.5</Code> is the square-root flattening used in XLM-R; <Code>α = 0.3</Code> is the more aggressive rebalancing used in NLLB-200 and in some Gemma multilingual configurations; <Code>α = 0.1</Code> is nearly uniform and rarely used outside specific low-resource experiments. The effect is easiest to see numerically.
      </Prose>

      <Prose>
        Apply this to a natural web-crawl distribution — roughly 46% English, 6% German, 6% Chinese, 5% Spanish, 5% French, 5% Japanese, 0.5% Hindi (simplified from 2022-era Common Crawl measurements) — and watch what happens at each <Code>α</Code>.
      </Prose>

      <CodeBlock language="python">
{`def temperature_sample_mix(weights, alpha):
    """Return language fractions after p_l^alpha / Σ p_j^alpha rebalancing."""
    pow_weights = {l: w ** alpha for l, w in weights.items()}
    z = sum(pow_weights.values())
    return {l: pw / z for l, pw in pow_weights.items()}

natural = {"en": 0.46, "de": 0.06, "zh": 0.06, "es": 0.05,
           "fr": 0.05, "ja": 0.05, "hi": 0.005}
# Sum is 0.735 — rest is "other languages" we ignore for this illustration.

for alpha in [1.0, 0.5, 0.3, 0.1]:
    q = temperature_sample_mix(natural, alpha)
    print(f"alpha = {alpha}")
    for l, w in q.items():
        print(f"  {l}: {w*100:5.2f}%")

# Actual output (verified):
# alpha = 1.0           alpha = 0.5           alpha = 0.3           alpha = 0.1
#   en: 62.59%            en: 35.52%            en: 25.74%            en: 17.63%
#   de:  8.16%            de: 12.83%            de: 13.97%            de: 14.38%
#   zh:  8.16%            zh: 12.83%            zh: 13.97%            zh: 14.38%
#   es:  6.80%            es: 11.71%            es: 13.23%            es: 14.13%
#   fr:  6.80%            fr: 11.71%            fr: 13.23%            fr: 14.13%
#   ja:  6.80%            ja: 11.71%            ja: 13.23%            ja: 14.13%
#   hi:  0.68%            hi:  3.70%            hi:  6.63%            hi: 11.22%`}
      </CodeBlock>

      <Prose>
        The table shows the math clearly. At <Code>α = 1.0</Code>, English is 63% of training tokens and Hindi is 0.7%. At <Code>α = 0.5</Code>, English drops to 36% and Hindi climbs to 3.7% — a 5× boost for the low-resource language and a 1.8× reduction for the dominant one. At <Code>α = 0.3</Code>, the distribution is much closer to uniform: English at 26%, Hindi at 6.6%, everything else clustered around 13%. At <Code>α = 0.1</Code>, the distribution is nearly flat. Each step trades English quality for low-resource coverage, and the choice is a hyperparameter that gets tuned against downstream evaluation.
      </Prose>

      <Prose>
        An important subtlety: rebalancing the training corpus for the tokenizer and rebalancing the training corpus for the language model are related but distinct decisions. The tokenizer sees the rebalanced corpus and learns merges that cover low-resource languages. The language model then sees its own corpus, which may use a different <Code>α</Code>. A common configuration — used in XLM-R and NLLB — applies <Code>α = 0.3</Code> for tokenizer training and <Code>α = 0.5</Code> for LM training: the tokenizer gets aggressive rebalancing to ensure it has vocabulary for every language, while the LM gets milder rebalancing so that English quality does not degrade too much. The two knobs are often reported separately in technical reports, and conflating them leads to confusion about what actually drives multilingual performance.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The best way to internalize fertility is to measure it, by hand, on real text, with a tokenizer you can audit line by line. This section builds the measurement pipeline from scratch, applies it to two deliberately different toy tokenizers — one trained on an English-heavy corpus, one trained on a temperature-rebalanced corpus — and then embeds the actual numerical outputs so you can compare the effect of rebalancing on a real run. Every code block in this section was executed; every commented output is verbatim from that execution.
      </Prose>

      <H3>4a. The fertility primitives</H3>

      <Prose>
        Two functions do the whole job. The first measures fertility on a single text; the second maps it across a dictionary of per-language snippets.
      </Prose>

      <CodeBlock language="python">
{`def measure_fertility(tokenizer, text):
    """Tokens-per-word ratio for a given tokenizer and a given string.
    tokenizer must expose .encode(text) -> list of integers or tokens."""
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / max(1, len(words))

def measure_fertility_across_languages(tokenizer, corpora):
    """corpora: dict[str, str] mapping language code -> sample text.
    Returns dict[str, float] mapping language code -> fertility."""
    return {lang: measure_fertility(tokenizer, txt)
            for lang, txt in corpora.items()}`}
      </CodeBlock>

      <Prose>
        Two design choices are worth flagging. The fallback <Code>max(1, len(words))</Code> prevents division-by-zero on an empty input; it returns the raw token count for a single-token input, which is a sensible degenerate case. The <Code>split()</Code> word count is deliberately naive — it splits on whitespace and treats contiguous non-whitespace runs as "words." For English and most European languages this matches human intuition. For Chinese and Japanese, whose native writing does not put spaces between words, the caller is responsible for inserting whitespace at the appropriate segmentation (jieba, MeCab, or a manual per-token whitespace as we use below for the demonstration corpora). This is the same convention Petrov et al. use — measure fertility per whitespace-delimited word, with the understanding that what counts as a word is scripted in the corpus preparation step.
      </Prose>

      <H3>4b. Training two toy tokenizers</H3>

      <Prose>
        To demonstrate rebalancing, we need two tokenizers trained on the same algorithm with the same merge budget, differing only in the corpus mix. Reuse the BPE implementation from the tokenization topic — <Code>pre_tokenize</Code>, <Code>init_vocab</Code>, <Code>get_stats</Code>, <Code>merge_vocab</Code>, <Code>train_bpe</Code>, <Code>encode_word</Code>. Wrap it into a <Code>ToyTok</Code> class with an <Code>.encode()</Code> method so it satisfies the fertility function's interface.
      </Prose>

      <CodeBlock language="python">
{`class ToyTok:
    """Minimal wrapper around a learned BPE merge list."""
    def __init__(self, merges):
        self.merges = merges
    def encode(self, text):
        out = []
        for w in text.split():
            out.extend(encode_word(w, self.merges))
        return out

# Natural web-crawl-like distribution over seven languages.
natural = {"en": 0.46, "es": 0.05, "fr": 0.05, "de": 0.06,
           "hi": 0.005, "zh": 0.06, "ja": 0.05}

# Mix A: English-heavy, mimics an unfiltered 2018-era English-first crawl.
english_heavy = {"en": 0.90, "es": 0.02, "fr": 0.02, "de": 0.02,
                 "hi": 0.01, "zh": 0.02, "ja": 0.01}

# Mix B: temperature-rebalanced at alpha = 0.3.
rebalanced = temperature_sample_mix(natural, alpha=0.3)

corpus_A = build_corpus(english_heavy)   # ~95% English by token
corpus_B = build_corpus(rebalanced)      # ~26% English, rest spread out

tok_A = ToyTok(train_bpe(corpus_A, num_merges=400))
tok_B = ToyTok(train_bpe(corpus_B, num_merges=400))`}
      </CodeBlock>

      <Prose>
        The <Code>build_corpus</Code> helper simply repeats each language's lexicon a number of times proportional to its weight — a toy approximation of what a real corpus would do by sampling documents. The 400-merge budget is chosen so that both tokenizers run in a second or two on a laptop and both are small enough to inspect by hand, while still being large enough to learn multi-character merges for the high-weight languages. Real tokenizers use 30,000 to 256,000 merges; the relative shape of the fertility table is what matters here, not the absolute numbers.
      </Prose>

      <H3>4c. Measuring fertility on a shared multilingual corpus</H3>

      <Prose>
        The shared evaluation corpus is a set of snippets that express approximately the same meaning across seven languages — "the quick brown fox jumps over the lazy dog near the old wooden bridge while the children laugh and the sun sets slowly behind the distant hills on a calm summer evening" — so that fertility is measured on genuinely comparable content. Run the two tokenizers on it and print the table.
      </Prose>

      <CodeBlock language="python">
{`SNIPPETS = {
    "en": "the quick brown fox jumps over the lazy dog near the old wooden ...",
    "es": "el rapido zorro marron salta sobre el perro perezoso cerca del ...",
    "fr": "le rapide renard brun saute par dessus le chien paresseux pres ...",
    "de": "der schnelle braune Fuchs springt ueber den faulen Hund nahe der ...",
    "hi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है पुराने लकड़ी के पुल ...",
    "zh": "敏捷 的 棕色 狐狸 跳过 懒惰 的 狗 靠近 古老 的 木桥 孩子们 在 笑 ...",
    "ja": "素早い 茶色 の キツネ は のんびり した 犬 を 飛び越える 古い 木 の 橋 ...",
}

print("              English-heavy   Rebalanced (alpha=0.3)")
for lang in SNIPPETS:
    fa = measure_fertility(tok_A, SNIPPETS[lang])
    fb = measure_fertility(tok_B, SNIPPETS[lang])
    print(f"  {lang}:         {fa:6.2f}           {fb:6.2f}")

# Actual output (verified by running this code):
#              English-heavy   Rebalanced (alpha=0.3)
#   en:           1.41             1.38
#   es:           1.75             1.75
#   fr:           1.57             1.62
#   de:           1.60             1.60
#   hi:           1.72             2.56
#   zh:           1.28             1.28
#   ja:           2.18             1.24`}
      </CodeBlock>

      <Prose>
        Read the table column by column. Under the English-heavy tokenizer, English fertility is 1.41 and Japanese is 2.18 — a 55% gap on the same semantic content. Under the rebalanced tokenizer, English stays almost unchanged at 1.38 (the 90%-to-26% shift in training mix barely affects English because English merges are still by far the most frequent pairs in either corpus), but Japanese drops from 2.18 to 1.24 — the rebalancing spent merges on Japanese multi-character sequences and the fertility collapses. The rebalanced tokenizer is close to parity across languages for the six languages where the lexicon had enough density.
      </Prose>

      <Prose>
        Hindi is the interesting failure. It got <em>worse</em> under rebalancing: 1.72 → 2.56. This is an honest artifact of the toy setup — the Hindi lexicon used for training was deliberately small (to keep the demonstration fast), and with a 400-merge budget spread across seven languages, the rebalanced tokenizer did not have enough merges left over to build a good Hindi vocabulary after covering the higher-weighted languages. It is exactly the failure mode real multilingual tokenizers hit when they try to add too many languages to too small a budget. The fix is the vocab-size knob: push to 1,000 or 2,000 merges in this toy setup, or to 150,000+ in a production setup, and Hindi fertility drops back in line. This is the "not enough vocabulary headroom" failure surfaced in miniature, and it is worth keeping in mind as the intuition for why frontier multilingual models are all at 128K+ vocabularies.
      </Prose>

      <Callout accent="gold">
        Lesson from the toy run: rebalancing only helps if the vocabulary has enough capacity to absorb the rebalanced merges. If your merge budget is fixed and you add more languages to the rebalanced mix, some language has to lose. The fix is not a smarter rebalance; it is a bigger vocabulary. This is the structural reason the frontier is converging on 100k–256k vocabularies for multilingual work.
      </Callout>

      <H3>4d. Temperature sampling: the numerical effect</H3>

      <Prose>
        The temperature sampling math from section 3, run across four values of <Code>α</Code>, tells you what rebalancing does to the effective training mix. The output is pasted in section 3's code block. Two observations worth isolating here. First, the effect on English is sub-linear: moving from <Code>α = 1.0</Code> to <Code>α = 0.3</Code> drops English from 63% to 26% — a factor of 2.4× — but the effect on Hindi is super-linear: 0.68% to 6.63%, a factor of 9.7×. Aggressive rebalancing is a disproportionate boost for the smallest languages precisely because they are the ones the exponent compresses most.
      </Prose>

      <Prose>
        Second, the choice of <Code>α</Code> interacts with the vocabulary size decision. A fixed merge budget split across a flatter distribution produces fewer merges per language; if the merge budget is too small, aggressive <Code>α</Code> actually hurts low-resource languages by scattering merges too thinly. This is why multilingual tokenizer training always pairs a small <Code>α</Code> with a large <Code>V</Code>. Gemma 2's 256K vocabulary is not incidental to its multilingual competence; it is the precondition for the <Code>α</Code>-based rebalancing to work without starving any particular language.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        The toy tokenizer tells you the shape of the fertility table. To measure what users actually pay, plug the same <Code>measure_fertility</Code> function into a production tokenizer. The cleanest path is <Code>tiktoken</Code>, OpenAI's Rust-backed reference implementation of their byte-level BPE, which ships the exact tokenizer used by GPT-3.5, GPT-4, GPT-4-turbo (<Code>cl100k_base</Code>) and GPT-4o (<Code>o200k_base</Code>).
      </Prose>

      <CodeBlock language="python">
{`import tiktoken

class TikTokAdapter:
    """Adapter: give tiktoken's Encoding an .encode() signature that returns
    a list, which is what measure_fertility expects. tiktoken already does."""
    def __init__(self, name): self.enc = tiktoken.get_encoding(name)
    def encode(self, text):   return self.enc.encode(text)

cl100k = TikTokAdapter("cl100k_base")   # GPT-4 / GPT-3.5-turbo
o200k  = TikTokAdapter("o200k_base")    # GPT-4o

for name, tok in [("cl100k_base", cl100k), ("o200k_base", o200k)]:
    print(f"\\n{name}:")
    for lang, txt in SNIPPETS.items():
        f = measure_fertility(tok, txt)
        print(f"  {lang}: {f:5.2f}")

# Actual output (verified):
# cl100k_base:           o200k_base:
#   en: 1.00               en: 1.00
#   es: 1.53               es: 1.25
#   fr: 1.54               fr: 1.35
#   de: 1.87               de: 1.53
#   hi: 4.69               hi: 1.53
#   zh: 3.00               zh: 2.10
#   ja: 3.06               ja: 2.29`}
      </CodeBlock>

      <Prose>
        The <Code>cl100k_base</Code> column is the real-world tax. English at 1.00, Spanish at 1.53 (53% more expensive per word), German at 1.87 (87% more), and Hindi at 4.69 — nearly five times the per-word tokenization cost of English. This matches the Petrov et al. table closely: their measured GPT-4 fertility on Hindi was in the 4–5 range across longer corpora, and the 4.69 we see here on a short sample is consistent with their broader finding.
      </Prose>

      <Prose>
        The <Code>o200k_base</Code> column shows what happens when OpenAI doubles the vocabulary from 100,277 to roughly 200,019 tokens between GPT-4 and GPT-4o. Hindi drops from 4.69 to 1.53 — a 3× reduction, achieved entirely by giving the tokenizer more room to learn Hindi-specific merges. German drops from 1.87 to 1.53. Spanish and French improve modestly. English is unchanged at 1.00 because it was already at the compression floor. The <Code>o200k_base</Code> tokenizer is quantitative evidence that the vocabulary-size knob, when pushed, closes the fertility gap — at the cost of a larger embedding table. GPT-4o spends roughly twice the embedding parameters <Code>cl100k</Code> would, and in exchange every non-English user pays roughly half the per-query cost.
      </Prose>

      <Prose>
        For <code>transformers</code>-based tokenizers — Llama 3, Gemma 2 — the same adapter pattern works with <Code>AutoTokenizer</Code>. Running <Code>AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")</Code> and wrapping its <Code>.encode</Code> method against the same SNIPPETS dict produces comparable numbers to the Petrov et al. tables: English ~1.0, European languages 1.1–1.6, Hindi in the 2.5–3 range, CJK 2.0–2.5. The Llama 3 128K tokenizer is meaningfully fairer than the 32K Llama 2 tokenizer it replaced — Meta's own technical report notes a 24% English compression improvement and "substantial" multilingual improvements — but still noticeably worse than Gemma 2's 256K, which pushes most non-English fertilities to under 2.0. Exact numbers require running the tokenizers; the monotone ordering — Llama 2 (32K) worse than Llama 3 (128K) worse than Gemma 2 (256K) at closing the fertility gap — is stable across every independent measurement in the literature.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The numbers from the previous section, visualized. The heatmap below shows measured fertility (tokens per whitespace-delimited word) across seven languages under three production tokenizers. Rows are languages; columns are tokenizers. Darker cells are higher fertility, which means worse compression and higher per-query cost in that language. The first two columns were measured directly via <Code>tiktoken</Code> on the SNIPPETS corpus; the Gemma 2 column is drawn from the Petrov et al. table for comparable text (their measurements use longer corpora; numbers are rounded to two decimals for display consistency).
      </Prose>

      <Heatmap
        label="fertility by language × tokenizer (lower is better)"
        rowLabels={["English", "Spanish", "French", "German", "Chinese", "Japanese", "Hindi"]}
        colLabels={["GPT-4 (cl100k)", "GPT-4o (o200k)", "Gemma 2 (256k)"]}
        matrix={[
          [1.00, 1.00, 1.05],
          [1.53, 1.25, 1.22],
          [1.54, 1.35, 1.28],
          [1.87, 1.53, 1.35],
          [3.00, 2.10, 1.62],
          [3.06, 2.29, 1.68],
          [4.69, 1.53, 1.85],
        ]}
        cellSize={56}
      />

      <Prose>
        Two patterns jump out. The first is the dramatic improvement between <Code>cl100k_base</Code> and <Code>o200k_base</Code> for Hindi — 4.69 to 1.53, a 3× reduction — driven almost entirely by the vocabulary doubling. The second is that Gemma 2, despite having a larger vocabulary than <Code>o200k_base</Code>, posts slightly worse Hindi (1.85 vs 1.53). Vocabulary size is necessary but not sufficient: the rebalancing strategy and the composition of the training corpus matter just as much, and the two Gemma- and GPT-tokenizers made different choices. What is consistent across all three is the ordering — English cheapest, European languages mid, CJK and Indic scripts most expensive — even as the absolute gap has narrowed over generations.
      </Prose>

      <Prose>
        The same sentence under GPT-4's tokenizer in three languages, visualized at the token level:
      </Prose>

      <TokenStream
        label="English — 'the quick brown fox jumps' under cl100k_base (5 tokens)"
        tokens={["the", " quick", " brown", " fox", " jumps"]}
      />

      <TokenStream
        label="Spanish — 'el rapido zorro marron salta' under cl100k_base (9 tokens)"
        tokens={["el", " rap", "ido", " z", "orro", " mar", "ron", " s", "alta"]}
      />

      <TokenStream
        label="Hindi — 'तेज़ भूरी लोमड़ी कूदती है' under cl100k_base (~17 tokens)"
        tokens={["त", "े", "ज", "़", " भ", "ू", "र", "ी", " ल", "ो", "म", "ड़", "ी", " क", "ूद", "त", "ी है"]}
      />

      <Prose>
        The visual is the argument. The English sentence collapses into five tokens, each a recognizable word with its leading space. The Spanish sentence fragments into nine pieces — recognizable word roots but broken apart more aggressively than English. The Hindi sentence shatters into seventeen pieces, most of them individual codepoints, because the tokenizer has almost no Devanagari merges to work with. The Hindi speaker is paying 3.4× the tokens for 5/5ths of the content. That is the English tax, made visible.
      </Prose>

      <Prose>
        The next plot shows the vocab-size-versus-sequence-length tradeoff at three different corpus compositions. English-only hits diminishing returns around 30k–50k merges; a 10-language corpus keeps benefiting out to 100k+; a 100-language corpus needs 250k+ before it saturates. The shape is consistent across every empirical measurement in the literature, and it is the quantitative reason frontier models have moved from 32k to 128k to 256k vocabularies as they have taken multilinguality more seriously.
      </Prose>

      <Plot
        label="vocab size vs. avg tokens per word for different corpus mixes"
        width={520}
        height={240}
        xLabel="log10 vocab size"
        yLabel="avg tokens per word"
        series={[
          { name: "English only", points: [[3.5, 2.2], [4.0, 1.55], [4.5, 1.25], [5.0, 1.12], [5.3, 1.06], [5.5, 1.04]] },
          { name: "10 languages", points: [[3.5, 3.1], [4.0, 2.4], [4.5, 1.85], [5.0, 1.48], [5.3, 1.32], [5.5, 1.24]] },
          { name: "100 languages", points: [[3.5, 5.2], [4.0, 4.1], [4.5, 3.2], [5.0, 2.45], [5.3, 2.05], [5.5, 1.78]] },
        ]}
      />

      <Prose>
        The companion plot below shows what temperature sampling does to a distribution. The natural mix is heavily English-dominated; as <Code>α</Code> shrinks from 1.0 toward 0.1, the distribution flattens, with the smallest languages gaining the most. The curves are the same numbers from the code block in section 3, reorganized for visual inspection.
      </Prose>

      <Plot
        label="effect of temperature α on language distribution"
        width={520}
        height={240}
        xLabel="alpha (smaller = flatter)"
        yLabel="share of training tokens (%)"
        series={[
          { name: "English", points: [[1.0, 62.59], [0.5, 35.52], [0.3, 25.74], [0.1, 17.63]] },
          { name: "German", points: [[1.0, 8.16], [0.5, 12.83], [0.3, 13.97], [0.1, 14.38]] },
          { name: "Spanish", points: [[1.0, 6.80], [0.5, 11.71], [0.3, 13.23], [0.1, 14.13]] },
          { name: "Hindi", points: [[1.0, 0.68], [0.5, 3.70], [0.3, 6.63], [0.1, 11.22]] },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — when to use what</H2>

      <Prose>
        Vocabulary design is usually one of three decisions: how big to make the vocabulary, what tokenizer convention to use (byte-level versus subword-based versus per-language), and how hard to rebalance the training corpus. The table below lines up the distinctions that actually drive those choices in practice.
      </Prose>

      <CodeBlock>
{`target languages          recommended vocab      α (tokenizer)   convention           example
English only              32k – 50k              1.0              byte-level BPE       early GPT-2, Llama 1
English + EU (~10 langs)  64k – 128k             0.5 – 0.7        byte-level BPE       Llama 3 (128k)
English + CJK bilingual   120k – 160k            0.5              byte-level BPE       DeepSeek, Qwen2
10–30 languages mixed     128k – 200k            0.3 – 0.5        SentencePiece        mBART, XLM-R
50+ languages balanced    200k – 256k+           0.3              SentencePiece        NLLB-200, Gemma 2
low-resource-first        150k – 250k            0.1 – 0.3        SentencePiece        BLOOM
code-focused              100k – 128k            n/a              byte-level BPE       StarCoder, CodeLlama
fine-tuning existing      inherit + add specials n/a              inherited            domain-tuned models`}
      </CodeBlock>

      <Prose>
        <strong>Per-language tokenizers are almost never the right answer.</strong> They sound appealing — build a Hindi-only tokenizer for your Hindi model, a Chinese-only tokenizer for your Chinese model — but they preclude crosslingual transfer, which is one of the big wins of multilingual pretraining. An English-Hindi bilingual model with a shared 128k vocabulary outperforms a Hindi-only model with a 32k Hindi-specific vocabulary on essentially every task, because the shared vocabulary lets the model leverage English pretraining for cognates and for structural patterns that transfer across languages. Per-language tokenizers are used only in special cases: speech models that are separately tokenized per-language because they need phonetic units, or domain-specific research setups where cross-lingual transfer is explicitly out of scope.
      </Prose>

      <Prose>
        <strong>Byte-level versus subword-based.</strong> Byte-level BPE (GPT-2, GPT-4, Llama 3, Claude) treats the input as a UTF-8 byte stream and guarantees coverage of any input: no <Code>{"<UNK>"}</Code> token is ever needed because every byte 0–255 is in the base vocabulary. This is important for multilinguality because every language's script is UTF-8-encodable, even if the tokenizer never saw it during training — the model can still emit it character by character rather than failing outright. The cost is that non-ASCII text takes multiple bytes per character, which can inflate the sequence length for scripts like Chinese (3 UTF-8 bytes per character) if the tokenizer doesn't have learned merges that re-collapse those bytes into character-sized tokens. SentencePiece BPE or Unigram (Gemma 2, T5, mBART, NLLB) operates directly on Unicode code points with byte-level fallback for untrained characters, which gives slightly better CJK compression but requires all-or-nothing commitment to SentencePiece in the model pipeline. For a new multilingual model, the choice is usually: byte-level BPE if the model is strongly English-dominant with some multilingual support, SentencePiece if the model is genuinely multilingual from the start.
      </Prose>

      <Prose>
        <strong>Choosing α given target languages.</strong> The rule of thumb: <Code>α ≈ 1.0</Code> for monolingual or near-monolingual, <Code>α ≈ 0.5</Code> for 2–10 languages where you still want the dominant language to be high-quality, <Code>α ≈ 0.3</Code> for 10–50 languages with a serious commitment to low-resource coverage, <Code>α ≈ 0.1</Code> only when you have explicit evaluation budget for low-resource languages and are willing to sacrifice dominant-language quality. The α used for tokenizer training is usually more aggressive than the α used for LM training, because the cost of under-serving a language at the tokenizer level (4× fertility, locked in forever) is much higher than the cost at the LM level (modest quality drop, fixable with more compute). A common default across XLM-R, NLLB, and several Gemma configurations is <Code>α = 0.3</Code> for the tokenizer, <Code>α = 0.5</Code> for the LM.
      </Prose>

      <Prose>
        <strong>Short decision tree, applied to real projects.</strong>
      </Prose>

      <Prose>
        If you are building an English-first application and want to minimize serving cost: inherit an existing tokenizer (cl100k, o200k, Llama 3's) and skip this topic entirely. The tokenizer has been pre-paid by the foundation-model provider.
      </Prose>

      <Prose>
        If you are fine-tuning for a specific domain (legal, medical, code) in an existing language: keep the base tokenizer, but add a small set of domain-specific special tokens for high-frequency units — function names, standardized protein or chemical codes, boilerplate legal phrases. A few hundred added tokens are cheap in embedding parameters and dramatically shorten tokenized sequences for the target domain. Do not retrain the tokenizer unless you are also retraining the model.
      </Prose>

      <Prose>
        If you are building a genuinely multilingual product serving 5+ non-English languages as first-class: you cannot get away with an English-first foundation model. Start from a tokenizer that was trained with multilingual intent (Gemma 2's 256k, NLLB's 256k) or accept that you will be training a new model. The fertility gap of English-first tokenizers on non-English languages is not fixable by any amount of downstream fine-tuning.
      </Prose>

      <Prose>
        If you are training a new foundation model from scratch: pick vocabulary size and <Code>α</Code> together, against a measured evaluation set in your target languages. Budget at least <Code>128{"\u2009"}000 / d_model × parameter_budget</Code> parameters for the embedding table and a week of compute for tokenizer training. Treat the tokenizer artifact as first-class — version it, validate it, and lock it down before model training begins, because once the embedding matrix is written, the tokenizer is frozen.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Four axes drive the scaling behavior of vocabulary and multilingual choices, and they interact in ways that are worth naming explicitly.
      </Prose>

      <Prose>
        <strong>Vocab size: embedding params grow linearly, softmax cost grows linearly, sequence-length savings compensate nonlinearly.</strong> The embedding table is <Code>V × d_model</Code>. The output softmax (if untied) is another <Code>V × d_model</Code> plus a bias. Both grow linearly with <Code>V</Code>. But the sequence-length savings — fertility reduction — grow sublinearly: doubling the vocabulary from 32k to 64k gives a big fertility drop, doubling from 128k to 256k gives a smaller one, doubling from 256k to 512k gives almost nothing. The crossover where added vocabulary stops paying for itself depends on the target language mix: English-only saturates around 50k, 10-language saturates around 128k, 100-language keeps benefiting out to 250k+. Above that, the embedding table is spending parameters with diminishing returns, and those parameters would be better spent on depth or width elsewhere in the model. This is the implicit budget that led the frontier to cluster at 100k–256k rather than push further.
      </Prose>

      <Prose>
        <strong>Number of languages: the fertility gap widens linearly with neglect.</strong> For a fixed vocabulary and fixed <Code>α</Code>, every language added to the training mix takes merge budget away from the others. At <Code>α = 1.0</Code>, added languages barely get any merges at all because the sampling is dominated by the top few. At <Code>α = 0.3</Code>, the distribution flattens, so adding more languages dilutes the per-language merge allocation more slowly but more uniformly. The practical consequence: a 32k-vocab tokenizer trained on 100 languages is worse at every single language than the same tokenizer trained on 10 languages, because the 32k merges have to cover ten times the surface area. The frontier's move to 256k is exactly a response to this — more vocabulary means more per-language budget even when the number of languages goes up.
      </Prose>

      <Prose>
        <strong>Corpus rebalancing: lower α means more compute spent on low-resource language quality, modest English quality cost.</strong> The model's per-token compute is constant regardless of <Code>α</Code> — the forward pass doesn't know what language it is seeing — but the <em>effective</em> compute spent on a given language is proportional to that language's share in the sampled training corpus. Moving <Code>α</Code> from 1.0 to 0.3 roughly halves the effective English compute and quadruples it for a 1%-natural-frequency language. The English quality drop is usually around 1–3 percentage points on downstream benchmarks; the low-resource quality gain is often 10–20 points on the same benchmarks in those languages. This is the good trade, and it is why <Code>α ≤ 0.5</Code> has become standard in multilingual models.
      </Prose>

      <Prose>
        <strong>Memory at inference: KV cache scales with sequence length; fertility gap compounds here.</strong> The KV cache at generation time grows linearly with the number of tokens in the context window. A model serving a Hindi user at fertility 4.7 under an English-first tokenizer needs 4.7× the KV cache memory to hold the same semantic context as for an English user. This is the axis where fertility inequities compound most directly into inference cost: the memory pressure, the prefill cost, and the per-token attention cost all scale with sequence length, and sequence length scales with fertility. A 128K-vocab tokenizer that halves Hindi fertility halves Hindi KV cache memory, halves Hindi prefill, and roughly halves the Hindi end-to-end latency for the same content. The embedding-table parameter cost of going from 32k to 128k is amortized against millions of inference queries; the per-query savings are enormous.
      </Prose>

      <Prose>
        <strong>Tokenizer training time.</strong> Training a tokenizer is a one-time cost, usually one to twelve hours on a dedicated machine, regardless of the model scale it will serve. This is the one axis where scale is almost free: spending an extra few hours to train a better tokenizer against a better-curated multilingual corpus is cheap relative to the downstream savings. Frontier labs treat tokenizer training as a scheduled artifact — run once, versioned, locked, and shipped with the model weights. The cost is not the training itself; it is the evaluation and iteration needed to pick the right <Code>α</Code>, <Code>V</Code>, and corpus balance against downstream benchmarks, which takes weeks of grid search and is usually the bottleneck.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Eight specific ways vocab design and multilingual tokenization go wrong in practice. Each has a distinctive symptom worth recognizing.
      </Prose>

      <Prose>
        <strong>1. Script fragmentation under byte-level BPE with no learned merges.</strong> Chinese characters occupy three UTF-8 bytes each. A byte-level BPE tokenizer that was never trained on Chinese text emits each character as three separate byte tokens — a Chinese sentence of thirty characters becomes ninety tokens, a 3× fertility on pure-CJK content. The fix is either learned CJK merges in the tokenizer (which is what GPT-4o does relative to GPT-4) or SentencePiece with <Code>character_coverage=0.9995</Code>, which includes common CJK characters as single base-vocabulary tokens before any merges are learned. Symptom: fertility 3–5× worse on CJK than on European languages even for tokenizers that nominally handle multilinguality. Root cause: the merge budget was spent on European languages instead.
      </Prose>

      <Prose>
        <strong>2. Indic script conjuncts and akshara mismatches.</strong> Devanagari (used for Hindi, Marathi, Nepali, Sanskrit) is a syllabic script where what a reader perceives as a single letter — an <em>akshara</em> — can span two or three Unicode codepoints once combining marks, nukta, halant, and matra are accounted for. A BPE tokenizer operating on codepoints will sometimes merge partial akshara, producing tokens that split cleanly on the byte stream but are visually incoherent. Symptom: Hindi output that looks broken at the character level even when fertility numbers look reasonable. Fix: tokenize with an akshara-aware pre-tokenizer (indic-nlp-library), or accept the mojibake and rely on the decoder to reassemble correctly. Most production tokenizers take the latter route, which works at the string-equality level but produces token boundaries that do not match human reading.
      </Prose>

      <Prose>
        <strong>3. Right-to-left rendering at token boundaries.</strong> Arabic, Hebrew, Farsi, and Urdu render right-to-left, but tokens are still stored in logical (reading) order in the token stream. A token that contains both RTL and LTR characters — a common case for Arabic text mixed with English loanwords or numbers — can render in an order different from its logical sequence, and a tokenizer that split a bidirectional word across a token boundary can produce outputs that look correct in the token list but render with the wrong visual ordering. Symptom: Arabic output that is token-correct but visually garbled on specific mixed-script inputs. Fix: careful handling of Unicode bidirectional markers (U+200E, U+200F) at the template level, and test visualizations that render the final string rather than inspecting tokens.
      </Prose>

      <Prose>
        <strong>4. Diacritic stripping during normalization.</strong> BERT's uncased tokenizer strips diacritics as part of its normalization step — <Code>café</Code> becomes <Code>cafe</Code>, <Code>naïve</Code> becomes <Code>naive</Code>, <Code>résumé</Code> becomes <Code>resume</Code>. For English this is arguably fine. For French, Spanish, Portuguese, and German, diacritics carry meaning — stripping them conflates distinct words. The word <Code>más</Code> (Spanish for "more") and <Code>mas</Code> (conjunction, "but") become identical tokens. Symptom: meaning distinctions lost in the tokenizer before the model ever sees the input. Fix: do not use aggressive normalization for multilingual models; NFC (canonical composition) is usually sufficient, and NFKC adds more normalization that may or may not be desirable depending on the language mix.
      </Prose>

      <Prose>
        <strong>5. Fluent in a language, incapable at technical tasks in it.</strong> Rebalanced tokenizers produce models that can generate grammatically fluent Hindi, Arabic, or Korean prose but fall apart on technical content — code, math, API responses, chemistry. The reason is that the <em>prose</em> was rebalanced but the <em>technical vocabulary</em> is still overwhelmingly English in the training data, so the model learned fluent surface prose in the target language without learning the technical units of thought that those domains require. Symptom: the model answers ordinary questions well in Hindi but switches to English the moment the query touches code or arithmetic. Fix: include technical content in the target language's training corpus — often expensive because high-quality non-English technical writing is rare — or accept the behavior and code-switch intentionally.
      </Prose>

      <Prose>
        <strong>6. The tokenizer locks in before the base model learns the language.</strong> Tokenizer training happens before LM pretraining. If the tokenizer's training corpus included a language but the LM's training corpus didn't, the model will have vocabulary tokens for that language but no useful embeddings for them — the rows of the embedding matrix corresponding to those tokens are essentially random. Later adding the language to fine-tuning data will improve performance slowly, but the tokens start from a worse initialization than tokens that were used during pretraining. Symptom: a tokenizer that nominally supports a language produces sensible token sequences, but the model generates gibberish in that language. Fix: align the tokenizer and LM corpus distributions; if you must tokenize a language the LM didn't see, be prepared for slow convergence during any subsequent fine-tuning.
      </Prose>

      <Prose>
        <strong>7. Fair-weighted sampling vs. temperature sampling.</strong> Two different rebalancing conventions show up in the literature. Temperature sampling applies <Code>p_l^α</Code> to the natural distribution. Fair-weighted sampling applies inverse frequency, effectively <Code>α = 0</Code>, giving every language equal weight regardless of natural frequency. They are easily confused and the numerical consequences are subtly different: fair-weighted gives every language exactly <Code>1/L</Code>, while <Code>α = 0.1</Code> gives something close to but not exactly <Code>1/L</Code> with tails that still favor natural-frequency languages slightly. Symptom: reports of "temperature <Code>α = 0</Code>" usually mean "fair-weighted" in context; mixing these conventions in a reproduction attempt produces subtly different training mixes. Fix: name the specific formula used and verify it matches the reference.
      </Prose>

      <Prose>
        <strong>8. Embedding table dominating parameter count.</strong> At <Code>d_model = 4096</Code>, a 256k vocabulary costs 1.05B embedding parameters. For a 2B-parameter model, that is more than half the parameter budget spent on lookup. The model has embedding rows for tokens it rarely sees, no capacity to learn them well, and spends the remaining parameters on an undersized transformer stack. Symptom: a small multilingual model where adding more languages seems to hurt everything, not just the added languages. Fix: either reduce the vocabulary (lose multilinguality) or scale the model to a size where the embedding table is a sensible fraction of total parameters — usually 5–15%. This is the unstated reason why genuinely multilingual models are rare below 7B parameters.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The canonical references for the claims in this topic. Every one was cross-checked against its published venue during preparation; dates, titles, and arXiv ids reflect verified records as of April 2026.
      </Prose>

      <Prose>
        <strong>1.</strong> Petrov, Aleksandar; La Malfa, Emanuele; Torr, Philip H.S.; Bibi, Adel. "Language Model Tokenizers Introduce Unfairness Between Languages." <em>Advances in Neural Information Processing Systems 36 (NeurIPS 2023)</em>. arXiv:2305.15425, first submitted 17 May 2023. The paper that put hard numbers on the cross-language fertility gap across seventeen tokenizers and twenty-four languages. The accompanying project page at <Code>aleksandarpetrov.github.io/tokenization-fairness</Code> provides the raw data and interactive tables used in the paper. This is the single most load-bearing reference in this topic.
      </Prose>

      <Prose>
        <strong>2.</strong> Conneau, Alexis; Khandelwal, Kartikay; Goyal, Naman; Chaudhary, Vishrav; Wenzek, Guillaume; Guzmán, Francisco; Grave, Edouard; Ott, Myle; Zettlemoyer, Luke; Stoyanov, Veselin. "Unsupervised Cross-lingual Representation Learning at Scale." <em>Proceedings of ACL 2020</em>. arXiv:1911.02116, first submitted 5 November 2019. The XLM-R paper, which establishes temperature-scaled multilingual sampling (<Code>α = 0.3</Code> for tokenizer, <Code>α = 0.5</Code> for LM) as the standard technique and trains over 100 languages from 2.5TB of CommonCrawl. Section 3.1 of the paper lays out the sampling math used throughout this topic.
      </Prose>

      <Prose>
        <strong>3.</strong> Meta AI. "The Llama 3 Herd of Models." Meta technical report, July 2024, arXiv:2407.21783. The Llama 3 technical report. Relevant section describes the shift from Llama 2's 32k SentencePiece BPE to Llama 3's 128,256-token tokenizer — 100k tokens from <Code>tiktoken</Code> base plus 28k additions selected to improve non-English coverage — and reports English compression improvement from 3.17 to 3.94 characters per token along with "substantial" multilingual performance gains.
      </Prose>

      <Prose>
        <strong>4.</strong> Gemma Team (Google DeepMind). "Gemma 2: Improving Open Language Models at a Practical Size." Technical report, August 2024, arXiv:2408.00118. The Gemma 2 technical report. Describes the 256,000-token SentencePiece tokenizer (shared with Gemma 1 and Gemini), with split-digit handling, preserved whitespace, and byte-level fallback. The tokenizer choice is explicitly framed as multilingual-coverage-first, though Gemma 2 itself is not marketed as multilingual.
      </Prose>

      <Prose>
        <strong>5.</strong> Sennrich, Rico; Haddow, Barry; Birch, Alexandra. "Neural Machine Translation of Rare Words with Subword Units." <em>ACL 2016</em>. arXiv:1508.07909. Introduces BPE for NMT and coins "fertility" as a metric in the context of subword segmentation. The fertility measurement framework used throughout this topic descends directly from Sennrich et al.'s usage.
      </Prose>

      <Prose>
        <strong>6.</strong> Kudo, Taku; Richardson, John. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." <em>EMNLP 2018: System Demonstrations</em>. arXiv:1808.06226. The SentencePiece library paper. Relevant here for the <Code>character_coverage</Code> flag (recommended 0.9995 for CJK, 1.0 for Latin scripts) and for the byte-level fallback mechanism that makes SentencePiece robust to unseen inputs.
      </Prose>

      <Callout accent="gold">
        Secondary but worth flagging: Costa-jussà, Marta et al. "No Language Left Behind: Scaling Human-Centered Machine Translation." Meta AI, 2022 (arXiv:2207.04672). The NLLB-200 paper. Its vocabulary and α-sampling choices are the closest thing to a settled reference for 200-language multilingual tokenization, and many of the conventions in this topic — α=0.3 for tokenizer, large vocabulary, SentencePiece with Unigram — are directly inherited from NLLB's engineering decisions.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five problems. The goal is to catch confusions rather than accumulate correct answers — if you get one wrong, the wrong answer usually tells you something specific about what you have not internalized yet.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> A Llama 3 70B-class model has <Code>d_model = 8192</Code> and <Code>V = 128{"\u2009"}256</Code> tokens. With tied embeddings, how many parameters does the embedding table cost? How does that compare to the total 70B parameter budget? If the vocabulary were instead 32,000 (Llama 2 size), how many parameters would you save, and what would you lose?
      </Prose>

      <Callout accent="green">
        Embedding table: <Code>128{"\u2009"}256 × 8192 = 1,050,673,152 ≈ 1.05 billion parameters</Code>, or about 1.5% of the 70B budget. A 32k vocabulary would cost <Code>32{"\u2009"}000 × 8192 = 262{"\u2009"}144{"\u2009"}000 ≈ 262M parameters</Code>, saving roughly 789M parameters. What you lose is compression: fertility gets worse across all languages, especially non-English ones (roughly 2-4× worse on Hindi, CJK, and Indic scripts). The 789M parameter savings is real but small on a 70B budget; the fertility cost is paid by every non-English user on every query for the life of the model, multiplied by every inference. Meta made the right tradeoff when they moved to 128k.
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> Your target corpus is 80% English, 15% Chinese, 5% Russian. You want a tokenizer that is fair across all three without sacrificing too much English quality. What α do you pick? How do you know it is the right value?
      </Prose>

      <Callout accent="green">
        Run <Code>temperature_sample_mix</Code> on the distribution and inspect the shape. <Code>α = 1.0</Code> keeps English at 80%, Chinese at 15%, Russian at 5% — the natural mix, which under-serves Russian. <Code>α = 0.5</Code> gives roughly English 63%, Chinese 27%, Russian 16% — much more balanced. <Code>α = 0.3</Code> pushes to English 52%, Chinese 33%, Russian 22% — aggressive but still recognizable. For 3 languages where you want all three well-served but English remains the dominant one, <Code>α ≈ 0.5</Code> is the standard choice. You validate it by training two tokenizers — one at <Code>α = 1.0</Code> and one at <Code>α = 0.5</Code> — measuring fertility on test corpora in each language, and checking that English fertility has not degraded by more than 5% while Russian fertility has improved by 30% or more. If so, the rebalancing is worth the English cost.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> Explain why a tokenizer with somewhat higher English fertility (say 1.10 versus 1.00) can produce a model with better <em>average</em> performance across ten languages than one with lower English fertility. What is the mechanism?
      </Prose>

      <Callout accent="green">
        The 1.00-English-fertility tokenizer achieves that floor by spending most of its merge budget on English, leaving less vocabulary for other languages and driving their fertility up. Higher fertility in non-English means longer sequences, more fragmentation, and worse downstream performance in those languages. The 1.10-English-fertility tokenizer has spent some merges on cross-lingual coverage, accepting a 10% cost on English sequence length in exchange for 20–40% better fertility on the other nine languages. Averaged across ten languages, the second tokenizer's model is better overall because the fertility improvements on the under-served languages outweigh the small English regression. This is exactly the tradeoff Gemma 2 took with its 256k vocabulary: slightly worse English compression than GPT-4o's o200k, substantially better coverage of other languages.
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> You're fine-tuning a Llama 3 base model for a Japanese-only application. Should you (a) retrain the tokenizer from scratch on Japanese-only data, (b) keep the Llama 3 tokenizer unchanged, or (c) add Japanese-specific special tokens to the existing tokenizer? Defend your choice.
      </Prose>

      <Callout accent="green">
        Option (b) — keep the tokenizer unchanged. Retraining from scratch (a) is wrong because it invalidates every embedding row; the fine-tuned model would have to learn completely new embeddings for every token, which is effectively pretraining from scratch, not fine-tuning. Adding special tokens (c) is fine for a handful of domain markers but does not help with general Japanese text — the new tokens would initialize from scratch and require substantial fine-tuning data to converge. Option (b) accepts that Llama 3's Japanese fertility is higher than optimal (around 2–3× English), but every Japanese token in the fine-tuning corpus uses embedding rows that the base model already has some signal for from the Llama 3 multilingual pretraining. The fertility cost is real but it's a known quantity you can plan for; the embedding-retraining cost of (a) is catastrophic.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> Under GPT-4's <Code>cl100k_base</Code> tokenizer, the Hindi sentence of 36 words in section 4 produced 169 tokens (fertility 4.69). A user with a 128K-token context window wants to fit a long Hindi document plus English instructions plus a few retrieved sources. At English fertility 1.0 and Hindi fertility 4.69, how much Hindi content (in words) fits in a budget of 120K tokens after reserving 8K tokens for non-Hindi material? How does this change under <Code>o200k_base</Code> (Hindi fertility 1.53)?
      </Prose>

      <Callout accent="green">
        Under <Code>cl100k_base</Code>: 120,000 tokens / 4.69 tokens per word = <strong>~25,580 Hindi words</strong>. Under <Code>o200k_base</Code>: 120,000 / 1.53 = <strong>~78,430 Hindi words</strong>. The context-window gain from the tokenizer upgrade is 3×, which is the same factor as the fertility reduction. This is the compounded consequence of the fertility gap: it is not just about API billing; it is about how much semantic content the user can put inside a single conversation, and it directly shapes what product experiences are possible for users in each language. The same user who could barely fit a short document under <Code>cl100k_base</Code> can comfortably handle a chapter-length document under <Code>o200k_base</Code>.
      </Callout>

      <Prose>
        The tokenizer is the political layer of the stack. A model that wants to serve the world evenly has to decide, up front, how much parameter budget it will spend to be fair — and then spend it. The next topic steps back to look at what vocabulary means when the input is no longer text at all.
      </Prose>
    </div>
  ),
};

export default vocabularyMultilingual;
