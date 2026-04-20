import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream } from "../../components/viz";
import BPETrainer from "../../components/BPETrainer";
import { colors } from "../../styles";

const tokenization = {
  title: "Byte-Pair Encoding (BPE), WordPiece, SentencePiece, Unigram",
  readTime: "18 min",
  content: () => (
    <div>
      <Prose>
        A language model cannot read. It consumes integers, and something upstream has to decide which integers stand for which stretches of text. That decision — where one unit ends and the next begins — is quietly one of the most consequential design choices in the entire pipeline. It determines vocabulary size, sequence length, how gracefully the model handles a misspelled word or a rare proper noun, and, for better or worse, how many tokens it takes to say the same sentence in Hindi versus English.
      </Prose>

      <Prose>
        Two obvious extremes both fail. Characters are universal but ruinously fine-grained: a 500-word essay balloons into thousands of steps, attention cost explodes quadratically, and the model has to relearn that <Code>t-h-e</Code> is a word every time. Whole words are compact but brittle: the vocabulary grows without bound, every unseen name or typo becomes an out-of-vocabulary token, and the model has no way to notice that <Code>unbelievably</Code> is just <Code>un</Code> plus <Code>believ</Code> plus <Code>ably</Code>. Subword tokenization carves out the middle ground — frequent words stay whole, rare ones decompose into parts the model has already seen, and the vocabulary is bounded by design.
      </Prose>

      <Prose>
        The practical stakes are easy to underestimate. A tokenizer that averages 4.2 characters per token produces a corpus roughly 30 percent shorter than one that averages 3.2. Thirty percent fewer tokens means thirty percent less attention compute, thirty percent more material fitting inside a fixed context window, and thirty percent lower API cost at inference time. The choice is neither cosmetic nor easily reversible — once a model is trained on a particular token inventory, that inventory is baked into its embedding matrix and its every learned weight. You do not change a production model's tokenizer. You live with it.
      </Prose>

      <H2>From characters to subwords</H2>

      <Prose>
        The same sentence, tokenized three different ways. Notice how the word-level view is short but would shatter the moment it encountered a token it had never seen. The character-level view survives anything but drowns the model in steps. Subword tokenization — the flavor used by nearly every production model today — keeps the common pieces intact and only splits where splitting pays.
      </Prose>

      <TokenStream
        label="word-level"
        tokens={["The", "unbelievably", "fast", "model", "tokenizes", "well", "."]}
      />

      <TokenStream
        label="character-level"
        tokens={["T", "h", "e", " ", "u", "n", "b", "e", "l", "i", "e", "v", "a", "b", "l", "y", " ", "f", "a", "s", "t", " ", "m", "o", "d", "e", "l", " ", "t", "o", "k", "e", "n", "i", "z", "e", "s", " ", "w", "e", "l", "l", "."]}
      />

      <TokenStream
        label="subword (bpe)"
        tokens={["The", " un", "bel", "iev", "ably", " fast", " model", " token", "izes", " well", "."]}
      />

      <Prose>
        The subword row is doing something subtle. <Code>unbelievably</Code> is not in its vocabulary as a single unit, but <Code>un</Code>, <Code>bel</Code>, <Code>iev</Code>, and <Code>ably</Code> all are, so the word comes through as four tokens instead of an unknown. <Code>tokenizes</Code> splits into a stem and <Code>izes</Code>. The leading space on <Code>{" fast"}</Code> is not a rendering quirk — modern BPE tokenizers treat the space as part of the following token, which is how the decoder knows whether to glue pieces together or insert a word boundary.
      </Prose>

      <H2>Byte-Pair Encoding — the mechanics</H2>

      <Prose>
        BPE started life in 1994 as a data-compression trick. Philip Gage described an algorithm that repeatedly replaced the most common pair of adjacent bytes with a new, unused byte — a tiny substitution code that squeezed redundancy out of a file. For two decades it lived in obscure compression papers. Then in 2016, Sennrich, Haddow, and Birch pulled it into neural machine translation as a way to handle rare words, and it became the dominant tokenization algorithm in NLP almost overnight.
      </Prose>

      <Prose>
        The training loop is refreshingly simple. Start by pre-tokenizing the corpus into words, then represent each word as a sequence of individual characters followed by an end-of-word marker like <Code>{"</w>"}</Code>. Count every adjacent character pair across the corpus, weighted by how often its word appears. Find the most frequent pair — say <Code>e s</Code> — and merge it everywhere into a single new symbol <Code>es</Code>. That merge becomes the first entry in the vocabulary. Recount. Merge again. Repeat until the vocabulary hits the target size, typically 30k to 50k for English, 100k+ for multilingual models.
      </Prose>

      <Prose>
        What emerges is not a hand-designed morphology but something that rhymes with one. Frequent prefixes, suffixes, and common words get picked up as single tokens because their internal pairs merge early. Rare words stay decomposed into smaller pieces. The end-of-word marker matters: without it, the merger cannot distinguish <Code>est</Code> at the end of <Code>fastest</Code> from <Code>est</Code> in the middle of <Code>estimate</Code>, and you lose the ability to recognize suffix patterns.
      </Prose>

      <Prose>
        Inference is deliberately asymmetric with training. During training you discover the merges; during inference you apply them. Given a new word, you start with its character decomposition and then sweep through the stored merge list in the exact order they were learned, greedily joining any adjacent pair that matches the current merge rule. The order is load-bearing — applying merges out of sequence produces a different, generally worse segmentation. This ordered-merge-list representation is why a trained BPE tokenizer is compact: a vocabulary of 50,000 tokens is just a list of 50,000 pairs plus the 256 or so base characters. No model. No parameters. A flat file you can ship in a few megabytes.
      </Prose>

      <H2>Hands-on: the BPE trainer</H2>

      <Prose>
        The interactive below implements exactly this loop. Type or paste a small corpus — a paragraph of text is plenty — and step through merges one at a time. At each step you will see the current pair counts, which pair won, and how the vocabulary changes. Watch what happens when the corpus contains repeated endings: <Code>{"est</w>"}</Code>, <Code>{"ing</w>"}</Code>, and <Code>{"ed</w>"}</Code> tend to fall out within the first dozen merges, which is BPE silently discovering English suffix structure from frequency alone.
      </Prose>

      <BPETrainer />

      <Prose>
        A useful experiment: feed it a corpus in a language you do not read. The algorithm has no linguistic knowledge and will still produce reasonable-looking subword units, because statistical regularity in character co-occurrence is a fair proxy for morpheme structure in most languages. This is why BPE generalizes across scripts without any per-language configuration — it is genuinely a universal compression scheme that happens to cut along meaningful seams more often than not.
      </Prose>

      <Prose>
        A second experiment that clarifies a lot: run the same corpus through twice, once with 100 merges and once with 1000. The 100-merge vocabulary will collapse only the most blindingly obvious patterns — common bigrams, repeated short words — and leave most of the text as individual characters. The 1000-merge vocabulary will feel like a real subword tokenizer, with whole common words intact. Somewhere in between is a curve where each additional merge yields diminishing returns in compression but adds a row to the vocabulary. Picking a vocabulary size is picking a point on that curve. Too small and you pay in sequence length; too large and you pay in embedding-table parameters and in tokens that appear so rarely the model cannot learn them.
      </Prose>

      <H3>From-scratch BPE in Python</H3>

      <Prose>
        The entire training algorithm fits in about twenty lines. The <Code>get_pair_stats</Code> function walks the vocabulary, counts adjacent symbol pairs weighted by word frequency, and returns a <Code>Counter</Code>. The <Code>merge_pair</Code> function rewrites the vocabulary so that every instance of the chosen pair is joined into a single symbol. The driver loop calls these in turn, recording each merge as it goes.
      </Prose>

      <CodeBlock language="python">
{`from collections import Counter

def get_pair_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_pair(pair, vocab):
    merged = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        merged[word.replace(bigram, replacement)] = freq
    return merged

def train_bpe(corpus, num_merges):
    vocab = Counter(" ".join(list(w)) + " </w>" for w in corpus.split())
    merges = []
    for _ in range(num_merges):
        stats = get_pair_stats(vocab)
        if not stats: break
        best = max(stats, key=stats.get)
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges, vocab`}
      </CodeBlock>

      <Prose>
        The initial vocabulary spaces characters apart with an explicit <Code>{"</w>"}</Code> so that the merge rewriter — which simply joins adjacent space-separated symbols — can work by string replacement instead of parsing. The returned <Code>merges</Code> list is ordered and is the only thing you need at inference time: to tokenize a new word, split it into characters, then apply each merge in order, greedily joining any pair that matches. That is the whole inference algorithm.
      </Prose>

      <Callout accent="gold">
        Production tokenizers like GPT-2's or Llama's operate on UTF-8 bytes rather than Unicode characters. This byte-level BPE variant guarantees that any possible input — emoji, Chinese, arbitrary binary — has a valid tokenization, because every byte value 0-255 is already in the base vocabulary. There are no out-of-vocabulary failures, ever.
      </Callout>

      <H2>WordPiece — BERT's variant</H2>

      <Prose>
        BERT and its many descendants use WordPiece, an algorithm developed at Google originally for Japanese and Korean speech recognition and later generalized. The training loop looks almost identical to BPE — build a character-level vocabulary, score candidate merges, pick the winner, repeat — but the scoring rule is different, and that difference changes the flavor of the resulting vocabulary.
      </Prose>

      <MathBlock>
        {"\\text{score}(A, B) = \\frac{\\text{freq}(AB)}{\\text{freq}(A) \\cdot \\text{freq}(B)}"}
      </MathBlock>

      <Prose>
        BPE merges the pair that appears most often. WordPiece merges the pair whose joint frequency is highest relative to the product of the individual frequencies — a likelihood-ratio test dressed up as a score. The practical effect: BPE will happily merge <Code>t</Code> and <Code>h</Code> early because both are extremely common, even though the combination <Code>th</Code> is not particularly surprising. WordPiece discounts that merge: <Code>t</Code> and <Code>h</Code> show up constantly on their own, so their co-occurrence does not earn many bits. It prefers pairs whose union is sharply more common than their parts would predict — word stems, recurring morphemes, proper nouns.
      </Prose>

      <Prose>
        WordPiece also marks continuation tokens explicitly. The word <Code>playing</Code> might tokenize as <Code>play</Code> and <Code>##ing</Code>, where the <Code>##</Code> prefix signals "glue this to the previous token with no space." The double hash is a BERT convention, chosen to be unlikely to collide with real text. This makes detokenization unambiguous at the cost of a slightly awkward wire format — and it forces the tokenizer to pre-tokenize on whitespace before running the subword split, which is why WordPiece-based models have historically struggled with languages that do not separate words with spaces.
      </Prose>

      <Prose>
        Empirically, the two algorithms produce vocabularies that overlap heavily — most of the same common words and suffixes fall out either way — but WordPiece tends to favor slightly longer, more morphologically coherent subwords where BPE favors high-frequency character bigrams. Neither is uniformly better. The choice is usually determined by which pretrained model you are starting from rather than by measurable downstream quality.
      </Prose>

      <TokenStream
        label="wordpiece (bert-style)"
        tokens={["the", "un", "##bel", "##iev", "##ably", "fast", "model", "token", "##izes", "well", "."]}
      />

      <H2>Unigram LM — SentencePiece's default</H2>

      <Prose>
        BPE and WordPiece are bottom-up and greedy — they build a vocabulary by stacking merges, and once a merge is made it is never undone. The Unigram language model, introduced by Kudo in 2018, inverts the whole procedure. Start with a deliberately oversized vocabulary — every substring that appears more than a handful of times. Fit a unigram probability distribution over the tokens using Expectation-Maximization, so that each token has a probability of being generated. Then iteratively prune: for each token, estimate how much total corpus likelihood would drop if it were removed, and discard the ones whose removal hurts least. Re-fit. Repeat until the vocabulary reaches the target size.
      </Prose>

      <Prose>
        The payoff is that a Unigram tokenizer is genuinely probabilistic. Given a sentence, there are usually many valid segmentations into vocabulary tokens, each with a probability the model can compute. At inference you can pick the most likely segmentation, or — and this is where it gets interesting — you can sample a segmentation. During training this sampling is called subword regularization: the model sees the same sentence split differently across epochs, which acts as a regularizer and tends to improve robustness to rare or noisy inputs. T5, XLNet, ALBERT, and most multilingual SentencePiece-based models ship Unigram tokenizers for exactly this reason.
      </Prose>

      <Prose>
        A BPE tokenizer is deterministic: one input, one token sequence. A Unigram tokenizer is a distribution: one input, a space of token sequences weighted by probability. That difference is small on paper and surprisingly large in practice, especially for morphologically rich languages where the "right" split is often genuinely ambiguous. Japanese, Finnish, Turkish, and agglutinative languages in general tend to benefit from Unigram's flexibility — there are several plausible ways to decompose a long compound word, and pinning down one as canonical is a loss rather than a simplification.
      </Prose>

      <H2>SentencePiece — the framework, not the algorithm</H2>

      <Prose>
        It is common to see "SentencePiece" listed alongside BPE, WordPiece, and Unigram as if it were a fourth algorithm. It is not. SentencePiece is a library, written by Taku Kudo at Google, that can train either a BPE or a Unigram tokenizer. Its contribution is not a new algorithm but a set of engineering decisions that make tokenization cleaner, faster, and fully reversible.
      </Prose>

      <Prose>
        The central trick: SentencePiece treats whitespace as a regular character. Before training, every space in the input is replaced with the Unicode character <Code>▁</Code> (U+2581, a lower-one-eighth block). From the tokenizer's perspective there is no whitespace at all — just a stream of characters, some of which happen to be <Code>▁</Code>. This eliminates the need for language-specific pre-tokenization rules. English, Chinese, and Thai all flow through the same pipeline, because SentencePiece never had to know what a word was in the first place.
      </Prose>

      <Prose>
        Reversibility falls out for free. A SentencePiece tokenizer can decode any token sequence back to the exact original string, whitespace and all, by concatenating tokens and replacing <Code>▁</Code> with a space. Compare this to classical tokenizers that rely on a pre-tokenization step — splitting on whitespace, lowercasing, stripping punctuation — which discard information that cannot be recovered. For languages like Chinese, Japanese, and Thai that do not separate words with spaces, this is not a luxury; it is the only approach that preserves the input faithfully.
      </Prose>

      <TokenStream
        label="sentencepiece (▁ marks word start)"
        tokens={["▁The", "▁un", "bel", "iev", "ably", "▁fast", "▁model", "▁token", "izes", "▁well", "."]}
      />

      <H2>Side-by-side comparison</H2>

      <Prose>
        Four names, three algorithms, one framework. The table below lines up the distinctions that actually matter when you are choosing a tokenizer or reading someone else's code.
      </Prose>

      <CodeBlock>
{`                BPE              WordPiece        Unigram LM       SentencePiece
origin          Gage 1994        Google 2012      Kudo 2018        library (2018)
direction       bottom-up        bottom-up        top-down         either
criterion       pair frequency   likelihood gain  EM over unigram  n/a (wraps algo)
output          deterministic    deterministic    probabilistic    depends on algo
reversible      usually not      no (## prefix)   yes (via ▁)      yes (via ▁)
famous users    GPT-*, Llama,    BERT, DistilBERT T5, XLNet,       Llama (BPE),
                Mistral          ELECTRA          ALBERT           Gemma (Unigram)`}
      </CodeBlock>

      <Prose>
        The production landscape is less tidy than the table suggests. Llama uses SentencePiece configured for BPE — so it is BPE algorithmically but benefits from SentencePiece's <Code>▁</Code> trick for round-trip faithfulness. GPT-2 through GPT-4 use byte-level BPE, a variant where the base vocabulary is the 256 possible byte values rather than Unicode characters, which gives unconditional coverage of any input at the cost of slightly awkward tokens for non-ASCII text. Gemma uses SentencePiece with Unigram. The "four boxes" picture is a useful mental model; the actual tokenizer you are staring at in a model card is almost always a specific combination of algorithm, framework, and byte-handling convention.
      </Prose>

      <Prose>
        A useful rule of thumb when reading papers or model cards: if the text says "BPE" without further qualification, assume byte-level BPE with a 30k-50k vocabulary. If it says "WordPiece," you are looking at a BERT-descendant with <Code>##</Code> continuations and a 30k vocabulary. If it says "SentencePiece," check the config to see whether the underlying model is BPE or Unigram — the name alone does not tell you. And if a model boasts a 100k+ vocabulary, it is almost certainly multilingual and paying for broader script coverage with a larger embedding matrix.
      </Prose>

      <H2>Training your own — practical notes</H2>

      <Prose>
        The HuggingFace <Code>tokenizers</Code> library has effectively won the ecosystem for training and shipping tokenizers. It is fast (Rust backend), supports all four major algorithms, integrates with the <Code>transformers</Code> library, and serializes to a single JSON file you can distribute alongside model weights. A minimal BPE training run looks like this.
      </Prose>

      <CodeBlock language="python">
{`from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]"])
tokenizer.train(["corpus.txt"], trainer)
tokenizer.encode("unbelievably").tokens
# -> ['un', 'believ', 'ably']`}
      </CodeBlock>

      <Prose>
        In nearly every real project, training your own tokenizer from scratch is the wrong move. A pretrained tokenizer ships with the model weights and any mismatch between the two breaks the model silently and completely — the embedding for token 4827 means what it means because it was trained that way, and swapping in a different token-to-id mapping produces gibberish. The cases where training your own pays off are narrow: domains where the pretrained vocabulary is catastrophically mismatched — source code in a rare language, DNA or protein sequences, chemistry SMILES strings, domain-specific math notation — and you are either training a model from scratch or doing heavy continued pretraining. For fine-tuning on ordinary text, keep the tokenizer you inherited.
      </Prose>

      <H2>Where it stands today — and where it breaks</H2>

      <Prose>
        Subword tokenization has been the default for roughly seven years, and no alternative has cleanly displaced it at scale. It is fast, compresses well, generalizes across languages with a single configuration, and plays nicely with the rest of the transformer stack. The design space has narrowed to minor variations: byte-level vs character-level base vocabularies, BPE vs Unigram scoring, vocabulary sizes from 30k to 250k. Most of the interesting work happens elsewhere in the model now.
      </Prose>

      <Prose>
        The failure modes, though, are real and well-documented. A tokenizer trained primarily on English will segment Devanagari, Arabic, or CJK text into three to five times as many tokens as semantically equivalent English — a silent tax on non-English users measured in both API cost and context window consumption. Emoji, code, and rare Unicode sequences bloat in similar ways. Arithmetic is notoriously awful: the number <Code>1234</Code> might tokenize as <Code>12</Code> and <Code>34</Code>, or as <Code>1</Code> and <Code>234</Code>, depending on what appeared in the training corpus, and the model has to learn digit arithmetic through an interface that scrambles place value. Several recent models mitigate this by pre-splitting numbers digit by digit before tokenization, a hand-engineered patch around a structural flaw.
      </Prose>

      <Prose>
        Character-level prompts fare worst of all. Ask a model to reverse a word, count the letters in <Code>strawberry</Code>, or produce text where every word starts with a certain letter, and the failures often trace directly back to the tokenizer. The model never saw the word as letters; it saw it as two or three opaque chunks. It has to reason about spelling through the thinnest possible statistical interface. Tokenization shapes what the model can cheaply say, and the boundary between "cheap" and "expensive" was drawn by the corpus statistics of whatever text was lying around when the tokenizer was trained. The next topic picks up from exactly this pressure — byte-level and tokenizer-free approaches that try to drop the subword layer entirely and let the model learn its own units from raw bytes.
      </Prose>
    </div>
  ),
};

export default tokenization;
