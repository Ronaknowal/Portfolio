import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Plot } from "../../components/viz";
import BPETrainer from "../../components/BPETrainer";
import { colors } from "../../styles";

const tokenization = {
  title: "Byte-Pair Encoding (BPE), WordPiece, SentencePiece, Unigram",
  readTime: "36 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A language model consumes integers. Somewhere upstream of the first attention head, a piece of software decides that the string <Code>"unbelievably"</Code> is going to be the list <Code>[403, 67919, 13140]</Code>, or <Code>[403, 32, 2021]</Code>, or a single integer if the word was common enough in the training corpus to earn its own row in the embedding matrix. That decision — where one unit ends and the next begins — is quietly one of the most consequential design choices in the entire pipeline. It determines the size of the embedding table, the length of every sequence, how gracefully the model handles a misspelled word, a rare proper noun, or an emoji, and, for better or worse, how many tokens it takes to say the same sentence in Hindi versus English. Once a model is trained against a particular token inventory, that inventory is fused into its weights. You do not change a production model's tokenizer. You live with it.
      </Prose>

      <Prose>
        The algorithm that has dominated this layer for nearly a decade was not invented by an NLP researcher. Philip Gage published it in 1994 in <em>The C Users Journal</em>, under the title "A New Algorithm for Data Compression." The problem he was attacking had nothing to do with language. He wanted a small, fast expansion routine for systems with limited memory, something lighter than LZW that could decompress a file with a few hundred bytes of code. His solution — byte-pair encoding — was to scan the input for the most frequent pair of adjacent bytes, replace every occurrence with a single unused byte, record the substitution in a small table, and repeat until no gain remained. The file plus the table was smaller than the file alone. For two decades it lived quietly in compression papers and hobbyist text encoders.
      </Prose>

      <Prose>
        In 2015, Rico Sennrich, Barry Haddow, and Alexandra Birch posted a preprint that would be published the following year at ACL as "Neural Machine Translation of Rare Words with Subword Units" (arXiv:1508.07909). The problem they were attacking was the brittleness of word-level neural MT: out-of-vocabulary words were handled by a hardcoded <Code>{"<UNK>"}</Code> token, which meant every unseen name, compound, or morphological variant turned into a hole in the translation. They pulled Gage's algorithm out of compression and applied it to tokenization. Scan the corpus for the most common pair of adjacent <em>symbols</em>, where "symbol" starts as a single character and grows as merges accumulate. Record each merge in order. The result is a vocabulary that reads like a rough draft of English morphology — frequent words stay whole, rare words decompose into pieces the model has seen, and the vocabulary is bounded by design.
      </Prose>

      <Prose>
        Before subwords, tokenization had two bad options. Character-level tokenization is universal but ruinously fine-grained: a 500-word essay balloons into thousands of steps, attention cost explodes quadratically, and the model has to relearn at every position that <Code>t</Code>-<Code>h</Code>-<Code>e</Code> is a word. Word-level tokenization is compact but brittle: the vocabulary grows without bound, every unseen name or typo becomes an <Code>{"<UNK>"}</Code>, and the model has no way to notice that morphological relatives share any structure. Take the word <Code>"unbelievably"</Code>. A word-level tokenizer trained on ordinary English text may have seen <Code>"believe"</Code>, <Code>"believing"</Code>, and <Code>"unbelievable"</Code>, but the specific form <Code>"unbelievably"</Code> is rare enough that it will collide with the OOV bucket and vanish. A character-level tokenizer handles it fine but at twelve tokens of cost. A subword tokenizer splits it into <Code>un</Code>, <Code>believ</Code>, <Code>ably</Code> — three tokens that together carry the meaning, and all three of which the model has seen combined with other prefixes and suffixes often enough to have learned generalizable representations.
      </Prose>

      <TokenStream
        label="character-level — universal but long"
        tokens={["u", "n", "b", "e", "l", "i", "e", "v", "a", "b", "l", "y"]}
      />

      <TokenStream
        label="word-level — compact but fragile"
        tokens={[{ label: "<UNK>", color: "#f87171" }]}
      />

      <TokenStream
        label="subword (bpe-style) — the middle path"
        tokens={["un", "believ", "ably"]}
      />

      <Prose>
        The stakes are easier to see in cost terms. A tokenizer that averages 4.2 characters per token produces a corpus roughly thirty percent shorter than one that averages 3.2. Thirty percent fewer tokens means thirty percent less attention compute, thirty percent more material fitting inside a fixed context window, thirty percent lower API cost at inference time. The choice is neither cosmetic nor easily reversible. Sennrich et al. did not invent a new algorithm so much as transplant an old one into a setting where its particular biases — frequency-driven, bottom-up, ordered — happened to produce linguistically reasonable units for free. The rest of this topic is about what that algorithm actually does, how its variants differ, and how to build one yourself well enough to know when to trust it and when not to.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Strip the algorithm of notation and it is almost embarrassingly simple. Start by writing every word as a sequence of individual characters. Sprinkle a small end-of-word marker on the last character of each word so that the algorithm can tell "this piece ends a word" apart from "this piece continues one." Scan the whole corpus and count every adjacent pair of symbols, weighted by how often the word they appear in occurs. Find the pair that shows up most often. Glue every instance of that pair into a single new symbol. That new symbol is now the first entry in your vocabulary. Recount. Repeat. Stop when the vocabulary is as large as you want it.
      </Prose>

      <Prose>
        The end-of-word marker is the one non-obvious piece. Sennrich et al. use <Code>{"</w>"}</Code>; modern byte-level variants use a leading space character as an implicit equivalent. Its job is to keep suffixes distinguishable from mid-word sequences. Without it, the bigram <Code>es</Code> looks identical whether it closes the word <Code>fastest</Code> or sits inside <Code>esteem</Code>. With a marker, the merger sees <Code>e s {"</w>"}</Code> in the first case and <Code>e s</Code> in the second, and it can learn <Code>est{"</w>"}</Code> as a bona fide suffix token without gluing it to words that only happen to contain the same three letters in the middle. You can build a working BPE tokenizer without the marker. It just produces a slightly worse vocabulary, because the statistics that frequency-based merging cares about start to blur.
      </Prose>

      <Prose>
        Why does gluing the most frequent pair produce useful units? Because frequent adjacency is an imperfect but surprisingly honest signal for morpheme-like structure. The pair <Code>t h</Code> shows up constantly in English, so BPE merges it almost immediately. The pair <Code>i n g</Code> merges quickly and then the resulting <Code>ing</Code> merges with a word-ending marker to form <Code>ing{"</w>"}</Code>, the suffix. The pair <Code>e d {"</w>"}</Code> falls out in a similar arc. Past-tense, progressive, comparative suffixes — each of them is a high-frequency sequence of characters ending a word, so each of them is a candidate for early merging, and each of them ends up as a single token. Nothing in the algorithm knows what a morpheme is. It is just that the unit "a sequence of characters that frequently ends a word" and the unit "a morpheme" overlap a lot in practice.
      </Prose>

      <Prose>
        The mental model is this: <strong>BPE is building a vocabulary of high-frequency character sequences, picked greedily</strong>. It is not learning language. It is not parsing morphology. It is running a compression algorithm over text and using the compressed units as a substitute for a hand-designed vocabulary. That substitution works shockingly well for most written languages because written text is extremely redundant and its redundancy is concentrated in exactly the places linguists would expect — common words, common prefixes and suffixes, common roots. When the substitution fails — on code, on arithmetic, on languages whose orthography does not line up well with their morphology, on emoji, on names — it fails in ways that trace directly back to the fact that frequency is not the same thing as meaning. Most of the failure modes in section 9 are variations on this theme.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        For BPE itself, the math is operational rather than probabilistic. Let <Code>V</Code> be the current vocabulary — initially the set of individual characters plus the end-of-word marker. Let <Code>C</Code> be the corpus, represented as a multiset of words, where each word <Code>w</Code> has frequency <Code>freq(w)</Code> and is written as a sequence of symbols <Code>(s₁, s₂, ..., sₙ)</Code> drawn from <Code>V</Code>. The frequency of an adjacent symbol pair <Code>(a, b)</Code> across the corpus is the sum of word frequencies for every word that contains that pair in adjacent positions.
      </Prose>

      <MathBlock>
        {"\\text{freq}(a, b) = \\sum_{w \\in C} \\text{freq}(w) \\cdot \\#\\{i : s_i = a \\land s_{i+1} = b\\}"}
      </MathBlock>

      <Prose>
        The merge rule is a greedy argmax. At each step, pick the pair that maximizes this frequency, add the concatenation <Code>ab</Code> to the vocabulary, and rewrite every occurrence of the pair in the corpus as the single new symbol.
      </Prose>

      <MathBlock>
        {"(a^*, b^*) = \\underset{(a, b)}{\\operatorname{argmax}} \\; \\text{freq}(a, b)"}
      </MathBlock>

      <Prose>
        Two things are worth saying about this choice. First, it is greedy — the algorithm commits to the locally best merge at every step without ever reconsidering. A merge made early can never be undone, even if later statistics would suggest a different partition. This means BPE produces a vocabulary that is locally optimal but not globally optimal in any compression-theoretic sense. A truly optimal vocabulary of the same size would need a combinatorial search that no one runs in practice. Second, the merge list is ordered, and the order is load-bearing at inference. Applying the merges in a different order produces a different, usually worse, segmentation of the same input.
      </Prose>

      <Prose>
        WordPiece keeps the same structure but replaces the scoring rule. Schuster and Nakajima, working on a Japanese and Korean voice search system at Google in 2012 ("Japanese and Korean Voice Search," ICASSP 2012), wanted merges that would be favored when the combined unit carries more information than its parts. Their scoring function is the ratio of the pair's frequency to the product of the individual frequencies.
      </Prose>

      <MathBlock>
        {"\\text{score}(A, B) = \\frac{\\text{freq}(AB)}{\\text{freq}(A) \\cdot \\text{freq}(B)}"}
      </MathBlock>

      <Prose>
        This is the normalized pointwise mutual information between the two symbols, minus a constant. Read it the right way and the intuition is clean. BPE merges whichever pair appears most. WordPiece merges the pair whose co-occurrence is most surprising given how common the pieces already are on their own. The letters <Code>t</Code> and <Code>h</Code> are both extremely common, so their co-occurrence is not surprising — BPE merges them greedily because <Code>th</Code> shows up constantly, while WordPiece discounts that merge because most of the frequency is already accounted for by how common each letter is individually. WordPiece prefers pairs whose joint frequency is sharply higher than independence would predict — word stems, recurring morphemes, proper nouns. In the limit, the two algorithms agree on the obvious high-frequency merges and disagree on the borderline cases. Neither is uniformly better on downstream tasks; the choice is usually settled by which pretrained model you are starting from.
      </Prose>

      <Prose>
        Unigram LM, due to Taku Kudo in 2018 ("Subword Regularization," arXiv:1804.10959), inverts the whole procedure. Assume a unigram language model over subword tokens: the probability of a segmentation <Code>s = (s₁, ..., sₙ)</Code> of a word is the product of the token probabilities.
      </Prose>

      <MathBlock>
        {"P(s) = \\prod_{i=1}^{n} p(s_i), \\qquad s^* = \\underset{s \\in \\mathcal{S}(w)}{\\operatorname{argmax}} \\; P(s)"}
      </MathBlock>

      <Prose>
        Here <Code>𝒮(w)</Code> is the set of all valid segmentations of word <Code>w</Code> using tokens from the current vocabulary. Training proceeds top-down. Start with a deliberately oversized vocabulary, seeded by every substring that appears more than a few times in the corpus. Fit the token probabilities by Expectation-Maximization over the segmentation lattice — the E-step computes, for each token, its expected usage marginalized over all plausible segmentations; the M-step re-estimates <Code>p(sᵢ)</Code> as normalized expected counts. Once the probabilities stabilize, score each token by the drop in total corpus log-likelihood if that token were removed. Discard the bottom fraction of tokens by this score — roughly ten to twenty percent at a time — and refit. Iterate until the vocabulary reaches the target size. The resulting tokenizer is probabilistic in a way BPE and WordPiece are not: given a word, the model can produce the maximum-likelihood segmentation, or it can sample a segmentation proportional to probability, which is the mechanism behind subword regularization at training time.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The best way to understand a tokenizer is to build one. Every piece of code in this section was run against the classic Sennrich-paper corpus and the outputs embedded as comments are the actual outputs, verbatim. No pseudo-code, no dependencies beyond the standard library. By the end of this section you will have a complete <Code>BPETokenizer</Code> class with <Code>train</Code>, <Code>encode</Code>, and <Code>decode</Code> methods, and you will understand why it does what it does at every line.
      </Prose>

      <H3>4a. Pre-tokenization</H3>

      <Prose>
        Pre-tokenization is the step that decides what "a word" is for the purposes of counting. Classical BPE assumes whitespace-separated words; SentencePiece treats the input as one long stream and lets the algorithm decide. The choice matters, because pre-tokenization defines the boundary beyond which no merge is ever allowed to cross. A merge can join characters inside a word into a larger unit, but it cannot join the last character of one word to the first character of the next — that would require the pre-tokenizer to have kept those characters adjacent, and whitespace-based pre-tokenizers deliberately do not. The simplest possible pre-tokenizer is a whitespace split.
      </Prose>

      <CodeBlock language="python">
{`from collections import Counter

def pre_tokenize(text):
    """Split on whitespace, return a Counter of word frequencies."""
    return Counter(text.split())

corpus = ("low low low low low lower lower "
          "newest newest newest newest newest newest "
          "widest widest widest")
print(pre_tokenize(corpus))
# Counter({'newest': 6, 'low': 5, 'widest': 3, 'lower': 2})`}
      </CodeBlock>

      <Prose>
        Real systems go further. They strip punctuation into its own tokens, normalize Unicode (NFC or NFKC), lowercase or preserve case depending on the model, split digits, handle apostrophes and contractions explicitly. The GPT-2 pre-tokenizer is a notorious regex that pre-splits on specific Unicode categories before byte-level BPE takes over. But for understanding the core algorithm, whitespace is enough.
      </Prose>

      <H3>4b. Training</H3>

      <Prose>
        Training has three pieces: representing the corpus as a multiset of symbol sequences, counting adjacent pairs, and rewriting the corpus when a merge is chosen.
      </Prose>

      <CodeBlock language="python">
{`def init_vocab(word_freqs):
    """Represent each word as a tuple of single-char symbols plus </w>."""
    return {tuple(list(w) + ["</w>"]): f for w, f in word_freqs.items()}

def get_stats(vocab):
    """Count adjacent-symbol pairs, weighted by word frequency."""
    pairs = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Replace every adjacent occurrence of 'pair' with its concatenation."""
    a, b = pair
    merged = a + b
    out = {}
    for symbols, freq in vocab.items():
        new = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new.append(merged)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        out[tuple(new)] = freq
    return out

def train_bpe(corpus, num_merges):
    word_freqs = pre_tokenize(corpus)
    vocab = init_vocab(word_freqs)
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        count = pairs[best]
        vocab = merge_vocab(best, vocab)
        merges.append((best, count))
    return merges, vocab`}
      </CodeBlock>

      <Prose>
        The three functions map cleanly onto the math in the previous section. <Code>get_stats</Code> implements the pair-frequency sum. <Code>merge_vocab</Code> implements the rewriting step. <Code>train_bpe</Code> is the outer loop: count, argmax, merge, record. The use of tuples rather than strings for symbol sequences is deliberate: it makes the dictionary keys hashable and sidesteps a category of bugs where a multi-character symbol gets accidentally split back into characters by a string operation.
      </Prose>

      <Prose>
        Run this on the classic corpus — the one Sennrich et al. use in Figure 1 of the 2016 paper — and watch the first merges come out.
      </Prose>

      <CodeBlock language="python">
{`corpus = ("low low low low low lower lower "
          "newest newest newest newest newest newest "
          "widest widest widest")

merges, final = train_bpe(corpus, num_merges=10)
for i, (pair, count) in enumerate(merges, 1):
    print(f"{i:2d}. {pair[0]!r:>8} + {pair[1]!r:<8}  count={count}  -> {pair[0]+pair[1]}")

# Actual output (verified by running this code):
#  1.      'e' + 's'       count=9  -> es
#  2.     'es' + 't'       count=9  -> est
#  3.    'est' + '</w>'    count=9  -> est</w>
#  4.      'l' + 'o'       count=7  -> lo
#  5.     'lo' + 'w'       count=7  -> low
#  6.      'n' + 'e'       count=6  -> ne
#  7.     'ne' + 'w'       count=6  -> new
#  8.    'new' + 'est</w>' count=6  -> newest</w>
#  9.    'low' + '</w>'    count=5  -> low</w>
# 10.      'w' + 'i'       count=3  -> wi`}
      </CodeBlock>

      <Prose>
        The first merge is <Code>e + s</Code> with count nine — the pair appears three times inside <Code>newest</Code> (six occurrences) and three times inside <Code>widest</Code> (three occurrences), total nine. The second and third merges are <Code>es + t</Code> and then <Code>est + {"</w>"}</Code>, which together discover the suffix <Code>est{"</w>"}</Code> as a single token in three steps. Merges four through five do the same for <Code>low</Code>. Merge eight is the one that feels magical: <Code>new + est{"</w>"}</Code> collapses the entire six-occurrence word <Code>newest</Code> into a single token, because at that point <Code>new</Code> and <Code>est{"</w>"}</Code> are both already vocabulary items and their adjacency is the most common pair remaining. BPE has discovered, by pure frequency counting, that <Code>newest</Code> is worth a dedicated token.
      </Prose>

      <H3>4c. Encoding</H3>

      <Prose>
        Encoding — applying a trained tokenizer to new text — is the step most tutorials skip. It is also where most production bugs live. The naive approach is to scan the merge list in order and apply each merge everywhere it matches; this is correct but quadratic in the worst case. A cleaner formulation is to think of the merge list as a priority queue: each merge has a rank equal to its position in the list (earlier merges = higher priority = lower rank). To encode a word, repeatedly find the adjacent pair of symbols with the lowest rank, merge it, and repeat until no adjacent pair is in the merge table.
      </Prose>

      <CodeBlock language="python">
{`def encode_word(word, merges):
    """Apply merges greedily by rank to a single word."""
    symbols = list(word) + ["</w>"]
    rank = {pair: i for i, (pair, _) in enumerate(merges)}

    while True:
        # Find the adjacent pair with the lowest rank (highest priority).
        best_i = -1
        best_rank = float("inf")
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            r = rank.get(pair, float("inf"))
            if r < best_rank:
                best_rank = r
                best_i = i
        if best_i == -1:
            break
        a, b = symbols[best_i], symbols[best_i + 1]
        symbols = symbols[:best_i] + [a + b] + symbols[best_i + 2:]
    return symbols

def encode(text, merges):
    out = []
    for w in text.split():
        out.extend(encode_word(w, merges))
    return out

# Actual output (verified):
# encode_word('low',     merges) -> ['low</w>']
# encode_word('lower',   merges) -> ['low', 'e', 'r', '</w>']
# encode_word('newest',  merges) -> ['newest</w>']
# encode_word('widest',  merges) -> ['wi', 'd', 'est</w>']
# encode_word('slowest', merges) -> ['s', 'low', 'est</w>']
# encode_word('lowest',  merges) -> ['low', 'est</w>']`}
      </CodeBlock>

      <Prose>
        The encoder's output is a clean demonstration of what the merge list can and cannot do. <Code>newest</Code>, which was in the training corpus, collapses to a single token because the merge <Code>new + est{"</w>"}</Code> is in the table. <Code>lowest</Code> was <em>not</em> in the training corpus, but the tokenizer still handles it cleanly: <Code>low</Code> is a known merge, <Code>est{"</w>"}</Code> is a known merge, and nothing in the table says to glue them. The word comes out as <Code>[low, est{"</w>"}]</Code> — compositional generalization falling out of greedy merges, which is the whole point of subword tokenization. <Code>slowest</Code> decomposes similarly: <Code>s</Code> (unmergeable), <Code>low</Code>, <Code>est{"</w>"}</Code>.
      </Prose>

      <Prose>
        An important subtlety: the rank-based encoder does not always produce the same result as a strict left-to-right application of the merge list. When two non-overlapping pairs have the same rank, the choice of which to merge first can differ. In practice this rarely matters, but it is the reason the HuggingFace implementation, the <Code>sentencepiece</Code> implementation, and OpenAI's <Code>tiktoken</Code> all spell out their tie-breaking rules explicitly. If you are training a tokenizer for a model you intend to ship, pick one tie-breaking rule and commit to it everywhere — training, inference, and any downstream tokenizer ports.
      </Prose>

      <H3>4d. Decoding</H3>

      <Prose>
        Decoding is simple but worth making explicit. Each token is either a regular symbol or an end-of-word-marked symbol; the decoder concatenates tokens and inserts a space wherever it sees the marker.
      </Prose>

      <CodeBlock language="python">
{`def decode(tokens):
    words = []
    current = ""
    for t in tokens:
        if t.endswith("</w>"):
            current += t[:-4]
            words.append(current)
            current = ""
        else:
            current += t
    if current:
        words.append(current)
    return " ".join(words)

# decode(['low', 'e', 'r', '</w>', 'newest</w>']) -> 'lower newest'`}
      </CodeBlock>

      <Prose>
        Decoding is lossless in this formulation because every word-ending symbol carries the marker. Byte-level BPE achieves the same property differently — it encodes the leading space as part of the token (<Code>" fast"</Code> rather than <Code>fast</Code>), which lets the decoder reconstruct whitespace exactly without an out-of-band marker. Both schemes work; they differ mostly in how they serialize over the wire.
      </Prose>

      <H3>4e. Putting it together</H3>

      <CodeBlock language="python">
{`class BPETokenizer:
    def __init__(self):
        self.merges = []          # list of ((a, b), count) in order
        self.vocab = set()        # all symbols ever seen

    def train(self, corpus, num_merges):
        self.merges, final = train_bpe(corpus, num_merges)
        for symbols in final:
            self.vocab.update(symbols)
        return self

    def encode(self, text):
        return encode(text, self.merges)

    def decode(self, tokens):
        return decode(tokens)

tok = BPETokenizer().train(corpus, num_merges=10)
assert tok.decode(tok.encode("lower newest")) == "lower newest"
assert tok.encode("newest") == ["newest</w>"]
assert tok.encode("lowest") == ["low", "est</w>"]`}
      </CodeBlock>

      <Prose>
        That is a complete BPE tokenizer in under fifty lines of pure Python. It trains, it encodes, it decodes, it round-trips, and its first ten merges on the Sennrich corpus match what the 2016 paper reports. What it does not do is what production tokenizers spend most of their code on: byte-level handling of arbitrary Unicode, a cache for repeated encodings, special tokens for chat formatting or end-of-sequence, an Aho-Corasick trie to make encoding run in near-linear time on long documents, thread-safe access for multi-process workers, serialization to a format other libraries can load, and graceful handling of inputs that contain bytes the tokenizer has never seen. Those are engineering concerns, and section 5 is about how the HuggingFace, SentencePiece, and <Code>tiktoken</Code> libraries handle them. Before we get there, there are two more from-scratch implementations worth building, because they share so much of this scaffolding that the differences fit on one screen.
      </Prose>

      <H3>4f. WordPiece from scratch</H3>

      <Prose>
        WordPiece is what you get when you keep every structural piece of BPE and swap out exactly one function. The pre-tokenization is the same. The end-of-word marker is the same — or rather, the <em>idea</em> of a boundary marker is the same; BERT chose a different convention, which we will get to in a moment. The corpus representation as a dict of symbol tuples to word frequencies is the same. The outer loop is the same. The rewriting step — replace every adjacent occurrence of the chosen pair with its concatenation — is the same. The only difference, algorithmically, is which pair the loop picks at each step. BPE picks <Code>argmax freq(a, b)</Code>. WordPiece picks <Code>argmax freq(a, b) / (freq(a) · freq(b))</Code>. That is a one-line change, and it produces a measurably different vocabulary on the same corpus.
      </Prose>

      <Prose>
        The math connects back to section 3. Up to a constant, the WordPiece score is the pointwise mutual information between symbols <Code>A</Code> and <Code>B</Code>: log of joint probability divided by the product of marginals. PMI is the right quantity here because the question we care about is not "how often does this pair show up" but "how informative is this pair above and beyond the fact that its pieces are each individually common." The pair <Code>t + h</Code> is very common. Most of that commonness is already captured by how often <Code>t</Code> and <Code>h</Code> appear on their own. A merge that bakes <Code>th</Code> into a single token does not buy you much new information; it just shortens sequences. By contrast, the pair <Code>q + u</Code> is not especially frequent in raw counts, but almost every <Code>q</Code> is followed by <Code>u</Code>, so the PMI is huge. WordPiece spends merges on pairs like this. The edge case where <Code>freq(A) = 1</Code> is informative: the score collapses to <Code>freq(AB) / freq(B)</Code>, which is just "what fraction of <Code>B</Code>'s occurrences are preceded by <Code>A</Code>." When a symbol appears exactly once and it appears right before <Code>B</Code>, the score is high — which is exactly what you want, because that unique pair is a perfect predictor and deserves a dedicated token.
      </Prose>

      <CodeBlock language="python">
{`def get_symbol_freqs(vocab):
    """Count occurrences of each single symbol, weighted by word frequency."""
    freqs = Counter()
    for symbols, freq in vocab.items():
        for s in symbols:
            freqs[s] += freq
    return freqs

def get_wordpiece_stats(vocab):
    """PMI-style score: freq(AB) / (freq(A) * freq(B)) for every adjacent pair."""
    sym_freqs = get_symbol_freqs(vocab)
    pair_freqs = get_stats(vocab)    # reuse the BPE helper
    scores = {}
    for (a, b), ab_freq in pair_freqs.items():
        scores[(a, b)] = ab_freq / (sym_freqs[a] * sym_freqs[b])
    return scores, pair_freqs

def train_wordpiece(corpus, num_merges):
    word_freqs = pre_tokenize(corpus)
    vocab = init_vocab(word_freqs)
    merges = []
    for _ in range(num_merges):
        scores, pair_freqs = get_wordpiece_stats(vocab)
        if not scores:
            break
        best = max(scores, key=scores.get)
        vocab = merge_vocab(best, vocab)   # reuse the BPE rewriter
        merges.append((best, scores[best], pair_freqs[best]))
    return merges, vocab

# Run on the same classic corpus as BPE, print the first five merges.
merges, _ = train_wordpiece(corpus, num_merges=10)
for i, (pair, score, count) in enumerate(merges[:5], 1):
    print(f"{i}. {pair[0]!r} + {pair[1]!r}  score={score:.4f}  freq={count}")

# Actual output (verified by running this code):
# 1. 'i' + 'd'      score=0.3333  freq=3  -> id
# 2. 'l' + 'o'      score=0.1429  freq=7  -> lo
# 3. 's' + 't'      score=0.1111  freq=9  -> st
# 4. 'lo' + 'w'     score=0.0625  freq=7  -> low
# 5. 'w' + 'id'     score=0.1111  freq=3  -> wid`}
      </CodeBlock>

      <Prose>
        Compare this to BPE's first five merges on the same corpus. BPE picked <Code>e + s</Code>, <Code>es + t</Code>, <Code>est + {"</w>"}</Code>, <Code>l + o</Code>, <Code>lo + w</Code> — the most frequent pairs in raw count, which happened to walk down the suffix <Code>est</Code> before ever looking at anything else. WordPiece's first pick is <Code>i + d</Code>, with a PMI score of 0.333. That pair only occurs three times — nothing close to the most frequent — but <Code>i</Code> and <Code>d</Code> each occur only three times in the entire corpus, and every time they appear, they appear together, inside <Code>widest</Code>. That is a perfect co-occurrence, and WordPiece spends its first merge on it. By contrast, <Code>s + t</Code> has the highest joint count (nine) but a lower score because <Code>s</Code> and <Code>t</Code> each show up on their own frequently. BPE and WordPiece agree on some merges (both eventually produce <Code>lo</Code> and <Code>low</Code>) and disagree on others. Roughly, BPE prefers pairs that are common-but-independent; WordPiece prefers pairs that are rare-but-locked-together.
      </Prose>

      <Prose>
        A note on the surface convention. The reference BPE implementation in the previous subsection uses <Code>{"</w>"}</Code> to mark the end of a word. BERT's WordPiece uses the opposite convention: the <em>first</em> piece of a word is bare, and every subsequent piece is prefixed with <Code>##</Code>. The word <Code>unbelievably</Code> under BERT's tokenizer comes out as <Code>['un', '##bel', '##ievably']</Code>. The information content is the same — you can recover word boundaries from either marker scheme — but the continuation-prefix convention has two practical advantages. The first is that it generalizes cleanly to languages without whitespace. In Chinese or Japanese, there is no notion of "end of word" that a training corpus can give you for free, but there is still a notion of "this token attaches to the previous one," which is exactly what <Code>##</Code> expresses. The second is detokenization: to reconstruct a sentence, walk left-to-right and prepend a space to every token that does <em>not</em> start with <Code>##</Code>. No stripping of suffix markers, no ambiguity about what a space means. The algorithm underneath is unchanged; the convention is a serialization choice.
      </Prose>

      <H3>4g. Unigram LM from scratch</H3>

      <Prose>
        Unigram is where things actually change. Kudo 2018 is not a merge algorithm with a different scoring rule. There is no merge loop at all. Instead, the training procedure is top-down: start with a deliberately oversized vocabulary, fit a unigram language model over it with Expectation-Maximization, score each token by how much it contributes to compression, drop the worst ones, and repeat. The mental model is sculpture rather than additive construction. BPE and WordPiece start with characters and glue outward. Unigram starts with a pile of candidate tokens and chisels away until only the useful ones remain.
      </Prose>

      <Prose>
        Build it up in four stages.
      </Prose>

      <Prose>
        <strong>Stage 1 — seed.</strong> Enumerate every substring of every word up to some maximum length, count occurrences weighted by word frequency, and keep the top <Code>N</Code> where <Code>N</Code> is two or three times the target vocabulary size. Always include every single character that appears in the corpus, so that no word is ever un-segmentable. Normalize the counts into a probability distribution — this is the initial unigram model.
      </Prose>

      <Prose>
        <strong>Stage 2 — segment.</strong> Given a vocabulary and its log-probabilities, the best segmentation of a word under the unigram model is the one that maximizes the product of token probabilities, equivalently the one that minimizes the sum of negative log-probabilities. That is a shortest-path problem on the segmentation lattice, and Viterbi solves it in <Code>O(n · L)</Code> time per word of length <Code>n</Code> with maximum token length <Code>L</Code>. The recursion is <Code>cost[i] = min over tokens t ending at i of cost[i - len(t)] - log p(t)</Code>. This is equation 3 of the Kudo paper, written in imperative form.
      </Prose>

      <Prose>
        <strong>Stage 3 — EM.</strong> The E-step computes, for each token, its expected count across the corpus under the current model. The exact expectation requires summing over every valid segmentation, which is a forward-backward computation. In practice the Viterbi approximation is used — treat the MAP segmentation as if it were the full posterior, collect token counts from it, and weight by word frequency. The MAP segmentation is the mode of the distribution over segmentations, and on natural-language corpora with sharp posteriors it is close to the full marginal. The M-step is trivial: re-estimate each token's probability as its expected count divided by the total expected count. Repeat until convergence, which in practice takes a handful of iterations.
      </Prose>

      <Prose>
        <strong>Stage 4 — prune.</strong> After EM has settled, score each token by the drop in total corpus log-likelihood if that token were removed from the vocabulary and the words that used it had to be re-segmented using the remaining tokens. A token that is the only way to cheaply express some common substring will have a large score — removing it forces the corpus to pay extra log-probability elsewhere. A token that is already shadowed by its pieces (say, <Code>lo</Code> when <Code>l</Code> and <Code>o</Code> are both around and the combination rarely segments as one piece anyway) will have a score near zero. Drop the bottom <Code>k</Code> percent — Kudo uses around twenty — and go back to Stage 3. Stop when the vocabulary reaches the target size.
      </Prose>

      <CodeBlock language="python">
{`import math
from collections import Counter

def seed_vocab(corpus, max_len=6, top_n=60):
    """Enumerate substrings up to max_len, keep top_n by weighted count,
    always retain every single character so words stay segmentable."""
    word_freqs = pre_tokenize(corpus)
    counts = Counter()
    for w, f in word_freqs.items():
        n = len(w)
        for i in range(n):
            for j in range(i + 1, min(i + max_len, n) + 1):
                counts[w[i:j]] += f
    chars = {c for w in word_freqs for c in w}
    kept = {t for t, _ in counts.most_common(top_n)} | chars
    total = sum(counts[t] for t in kept)
    return {t: counts[t] / total for t in kept}, word_freqs

def viterbi_segment(word, vocab_logprobs):
    """Best segmentation of word under the unigram model (MAP via DP).
    Returns (tokens, log_probability)."""
    n, INF = len(word), float("inf")
    cost = [INF] * (n + 1); back = [None] * (n + 1); cost[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - 16), i):
            tok = word[j:i]
            if tok in vocab_logprobs:
                c = cost[j] - vocab_logprobs[tok]
                if c < cost[i]:
                    cost[i] = c; back[i] = (j, tok)
    if cost[n] == INF:
        return None, -INF
    segs = []; i = n
    while i > 0:
        j, tok = back[i]; segs.append(tok); i = j
    return list(reversed(segs)), -cost[n]

def em_step(vocab_logprobs, word_freqs):
    """One EM iteration. E-step: Viterbi-approximated expected counts.
    M-step: normalize to a probability distribution."""
    expected = Counter()
    for w, f in word_freqs.items():
        segs, _ = viterbi_segment(w, vocab_logprobs)
        if segs is None: continue
        for tok in segs: expected[tok] += f
    total = sum(expected.values())
    if total == 0: return vocab_logprobs
    return {t: math.log(expected[t] / total) if expected[t] > 0
            else math.log(1e-12) for t in vocab_logprobs}

def corpus_loglik(vocab_logprobs, word_freqs):
    return sum(f * viterbi_segment(w, vocab_logprobs)[1]
               for w, f in word_freqs.items())

def score_token_loss(token, vocab_logprobs, word_freqs):
    """Drop in corpus log-likelihood if token were removed.
    Higher = token is more load-bearing for compression."""
    if token not in vocab_logprobs: return 0.0
    base = corpus_loglik(vocab_logprobs, word_freqs)
    reduced = {t: lp for t, lp in vocab_logprobs.items() if t != token}
    return base - corpus_loglik(reduced, word_freqs)

def train_unigram(corpus, target_vocab_size, prune_fraction=0.2, em_iters=2):
    probs, word_freqs = seed_vocab(corpus, max_len=6,
                                   top_n=max(target_vocab_size * 3, 30))
    logprobs = {t: math.log(p) for t, p in probs.items() if p > 0}
    chars = {c for w in word_freqs for c in w}
    while True:
        for _ in range(em_iters):
            logprobs = em_step(logprobs, word_freqs)
        if len(logprobs) <= target_vocab_size: break
        scores = {t: score_token_loss(t, logprobs, word_freqs)
                  for t in logprobs if t not in chars}
        if not scores: break
        sorted_toks = sorted(scores.items(), key=lambda kv: kv[1])
        n_drop = min(max(1, int(len(sorted_toks) * prune_fraction)),
                     len(logprobs) - target_vocab_size)
        for t, _ in sorted_toks[:n_drop]: del logprobs[t]
        z = sum(math.exp(lp) for lp in logprobs.values())
        logprobs = {t: lp - math.log(z) for t, lp in logprobs.items()}
    return logprobs, word_freqs

# Run on the same classic corpus, target a tiny 15-token vocab.
final, wf = train_unigram(corpus, target_vocab_size=15,
                          prune_fraction=0.2, em_iters=3)
for tok, lp in sorted(final.items(), key=lambda kv: -kv[1]):
    print(f"{tok!r:>10}  p={math.exp(lp):.4f}")

# Actual output (verified by running this code):
#   'newest'  p=0.3750
#      'low'  p=0.3125
#   'widest'  p=0.1875
#    'lower'  p=0.1250
#        'e'  p=0.0000   (retained as a character fallback; prob ~ 1e-12)
#        'r'  p=0.0000
#        'l'  p=0.0000
#        'd'  p=0.0000
#        'o'  p=0.0000
#        't'  p=0.0000
#        's'  p=0.0000
#        'i'  p=0.0000
#        'n'  p=0.0000
#        'w'  p=0.0000
#      'wid'  p=0.0000
#
# Viterbi segmentations under this vocab:
#   'newest'  -> ['newest']           # full word is one token
#   'widest'  -> ['widest']           # full word is one token
#   'lowest'  -> ['low', 'e', 's', 't']  # unseen word falls back to pieces
#   'slowest' -> ['s', 'low', 'e', 's', 't']`}
      </CodeBlock>

      <Prose>
        The final vocabulary is dominated by the four full words in the training corpus, with every single character retained as a fallback token for anything the model has never seen. That is a reasonable outcome for a target size of fifteen on a corpus of four distinct words: the full words themselves are by far the most compressive units, and the character fallbacks exist so that out-of-vocabulary text like <Code>lowest</Code> or <Code>slowest</Code> still has a valid segmentation. Run the same procedure on a realistic corpus with target 32,000 and you get a vocabulary that looks much more like BPE's — full common words, frequent prefixes and suffixes, residual characters — but chosen by likelihood rather than by greedy counting.
      </Prose>

      <Prose>
        Three things to notice about this construction. First, the pruning score is the right quantity. It is not "how often does this token appear" but "how much worse does the corpus compress if I take it away." A token that is a perfect substring of a more compressive token, and whose appearances always get absorbed by the longer one under Viterbi, has a pruning score of zero, and it should be the first to go. That is exactly what drives the vocabulary toward a minimum description length optimum. Second, because the segmentation model is explicit and probabilistic, Unigram gives you subword regularization for free at training time. Instead of always using the MAP segmentation, you can sample a segmentation from the distribution <Code>P(s | w) ∝ ∏ p(sᵢ)</Code> with temperature, which exposes the downstream model to multiple plausible tokenizations of the same word and generally improves robustness — this is the "subword regularization" in the title of Kudo's paper. BPE has no equivalent hook, because it is not a probabilistic model; later work ("BPE Dropout," Provilkov 2020) retrofits a similar effect onto BPE, but it is a less principled mechanism.
      </Prose>

      <Prose>
        Third, the cost. One EM iteration is <Code>O(|corpus| · L_max · |V|)</Code> — every word has to be Viterbi-segmented, and Viterbi over a word of length <Code>n</Code> with <Code>V</Code> candidate tokens costs roughly <Code>n · L_max</Code> if you use a prefix index. One pruning iteration is more expensive: each candidate token's score is a full <Code>corpus_loglik</Code> computation, which is another pass over the corpus, so the pruning step is <Code>O(|V| · |corpus| · n)</Code>. Total training is a handful of EM-plus-prune rounds — usually five to ten before the vocabulary reaches target size. Every iteration is more expensive than a BPE merge step, but there are fewer iterations overall, so the two algorithms end up in the same rough neighborhood for final wall-clock time on a given corpus. In production the SentencePiece C++ implementation uses a smarter prefix index and parallelizes over the corpus; the pure-Python version above is useful as a specification, not as a tool to actually train a ten-gigabyte corpus tokenizer.
      </Prose>

      <Prose>
        With BPE, WordPiece, and Unigram all implemented from scratch, the picture of section 5 changes. If you have read this section end to end, you have written working reference implementations of all three major subword algorithms, and you understand exactly what each one is doing at the level of a counter update and a dictionary rewrite. The library code in section 5 is then not a black box; it is the same algorithm with the Rust-and-C++ engineering concerns — parallelism, memory layout, trie-based encoding, serialization — layered on top. Read the libraries looking for those engineering concerns specifically, and the structure of their APIs stops feeling arbitrary.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In practice, nobody trains a production tokenizer from the Python loop above. Three libraries have effectively partitioned the ecosystem. HuggingFace's <Code>tokenizers</Code> (Rust backend, Python bindings) is the default for transformer training pipelines. Google's <Code>sentencepiece</Code> (C++) is the default when the model was trained with it — T5, mBART, XLNet, Gemma — and is still the cleanest path for multilingual work. OpenAI's <Code>tiktoken</Code> (Rust) is the reference implementation of their byte-level BPE and is used by anyone calling the OpenAI API who wants to count tokens before paying for them. All three are orders of magnitude faster than pure Python on the same corpus.
      </Prose>

      <Prose>
        Training a BPE tokenizer with HuggingFace looks like this.
      </Prose>

      <CodeBlock language="python">
{`from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=20,
    special_tokens=["[UNK]"],
    min_frequency=1,
)
tokenizer.train(["corpus.txt"], trainer)

# Running this on the same classic corpus produces:
# vocab: ['[UNK]','d','dest','e','es','est','ew','i','idest','l','lo',
#         'low','n','new','newest','o','r','s','t','w']
# encode('lower').tokens   -> ['low', 'e', 'r']
# encode('slowest').tokens -> ['s', 'low', 'est']`}
      </CodeBlock>

      <Prose>
        The HuggingFace vocabulary does not quite match the pure-Python vocabulary, because their implementation does not use the <Code>{"</w>"}</Code> marker convention — it handles word boundaries through the pre-tokenizer instead. The encoded output is functionally equivalent: <Code>lower</Code> decomposes into <Code>low</Code> plus <Code>e</Code> plus <Code>r</Code>, exactly as it does in the reference implementation, modulo the marker. This is a useful lesson in how much of a "BPE tokenizer" is the merge algorithm versus how much is the surrounding pre-tokenization, normalization, and boundary-marking convention. Swap any of those and the vocabulary shifts even though the algorithm is unchanged.
      </Prose>

      <Prose>
        What the Rust backend hides is substantial. The training loop is parallelized across CPU cores without the Python GIL. Corpus loading is streaming, so you can train on datasets that do not fit in memory. Merge application at inference uses an Aho-Corasick automaton — a trie with failure links — that matches all vocabulary items against the input in a single linear pass, which is why tokenizing a ten-megabyte document takes milliseconds rather than seconds. Special-token handling, UTF-8 validation, and thread safety are all baked in. The entire tokenizer serializes to a single JSON file that ships next to the model weights on the Hub.
      </Prose>

      <Prose>
        SentencePiece is the cleanest path when you want the algorithm to be language-agnostic. It treats the entire input as a stream of Unicode code points, replaces every whitespace character with the sentinel <Code>▁</Code> (U+2581, a lower-one-eighth block), and then runs either BPE or Unigram over the resulting stream with no separate pre-tokenization step. A minimal training call in the Python bindings looks like this.
      </Prose>

      <CodeBlock language="python">
{`import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="sp",
    vocab_size=20,
    model_type="bpe",        # or 'unigram', 'char', 'word'
    character_coverage=1.0,   # 1.0 for languages with small alphabets
)

sp = spm.SentencePieceProcessor(model_file="sp.model")
sp.encode_as_pieces("lower newest")
# -> ['▁lower', '▁', 'new', 'est']  (example; depends on vocab)
sp.decode(sp.encode_as_pieces("lower newest")) == "lower newest"  # True`}
      </CodeBlock>

      <Prose>
        The <Code>character_coverage</Code> flag deserves a note. For languages with small alphabets — English, most European languages — the default is 1.0, meaning every character in the training corpus is guaranteed a spot in the base vocabulary. For languages with large character sets — Chinese, Japanese, Korean — the recommended value is 0.9995, which drops the least common 0.05 percent of characters into an <Code>{"<unk>"}</Code> bucket to keep the base vocabulary tractable. This is the knob that makes SentencePiece work at multilingual scale without the embedding table exploding.
      </Prose>

      <Prose>
        OpenAI's <Code>tiktoken</Code> takes yet another angle. It ships the trained tokenizers of GPT-2, GPT-3.5, GPT-4, and GPT-4o as named encodings that you load by name rather than train from scratch. Its job is tokenization, not training. The API is intentionally small.
      </Prose>

      <CodeBlock language="python">
{`import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 / GPT-3.5
enc.encode("The unbelievably fast model tokenizes 1234 café emojis.")
# -> [791, 40037, 89234, 5043, 1646, 4037, 4861, 220, 4513, 19, 53050, 100166, 13]
enc.decode([791, 40037, 89234]) == 'The unbelievably'   # round-trips

enc2 = tiktoken.get_encoding("o200k_base")    # GPT-4o
# Same sentence: 14 tokens instead of cl100k_base's 15, because o200k_base
# gives 'unbelievably' its own token.`}
      </CodeBlock>

      <Prose>
        Running both encodings on the same input produces a useful comparison. The <Code>cl100k_base</Code> tokenizer, used by GPT-4 and GPT-3.5-turbo, splits <Code>unbelievably</Code> into <Code>[' unbelie', 'vably']</Code>. The newer <Code>o200k_base</Code> tokenizer, used by GPT-4o, has enough vocabulary headroom to give <Code>unbelievably</Code> its own single token. Neither is wrong; the vocabulary sizes are different (100,277 versus 200,019) and the training corpora were different. The important invariant is that within a single encoding, the mapping is fixed and deterministic — an input always produces the same token sequence, because that is the only way the downstream model can remain valid.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The step trace below walks through the first eight merges of BPE on the corpus <Code>{`"low low low low low lower lower newest newest newest newest newest newest widest widest widest"`}</Code>. Each step shows the corpus as its current symbol sequences, with the pair about to merge highlighted. The counts and segmentations are exactly what the reference implementation in section 4 produces — this is the real trajectory of a real run, not a hand-authored approximation.
      </Prose>

      <StepTrace
        label="bpe training — first eight merges"
        steps={[
          {
            label: "step 0 — initial characters",
            render: () => (
              <div>
                <TokenStream label="low (x5)" tokens={["l", "o", "w", "</w>"]} />
                <TokenStream label="lower (x2)" tokens={["l", "o", "w", "e", "r", "</w>"]} />
                <TokenStream label="newest (x6)" tokens={["n", "e", "w", "e", "s", "t", "</w>"]} />
                <TokenStream label="widest (x3)" tokens={["w", "i", "d", "e", "s", "t", "</w>"]} />
              </div>
            ),
          },
          {
            label: "step 1 — merge 'e + s' (count 9)",
            render: () => (
              <div>
                <TokenStream label="low (x5)" tokens={["l", "o", "w", "</w>"]} />
                <TokenStream label="lower (x2)" tokens={["l", "o", "w", "e", "r", "</w>"]} />
                <TokenStream label="newest (x6)" tokens={["n", "e", "w", { label: "es", color: colors.gold }, "t", "</w>"]} />
                <TokenStream label="widest (x3)" tokens={["w", "i", "d", { label: "es", color: colors.gold }, "t", "</w>"]} />
              </div>
            ),
          },
          {
            label: "step 2 — merge 'es + t' (count 9)",
            render: () => (
              <div>
                <TokenStream label="newest (x6)" tokens={["n", "e", "w", { label: "est", color: colors.gold }, "</w>"]} />
                <TokenStream label="widest (x3)" tokens={["w", "i", "d", { label: "est", color: colors.gold }, "</w>"]} />
              </div>
            ),
          },
          {
            label: "step 3 — merge 'est + </w>' (count 9)",
            render: () => (
              <div>
                <TokenStream label="newest (x6)" tokens={["n", "e", "w", { label: "est</w>", color: colors.green }]} />
                <TokenStream label="widest (x3)" tokens={["w", "i", "d", { label: "est</w>", color: colors.green }]} />
              </div>
            ),
          },
          {
            label: "step 4 — merge 'l + o' (count 7)",
            render: () => (
              <div>
                <TokenStream label="low (x5)" tokens={[{ label: "lo", color: colors.gold }, "w", "</w>"]} />
                <TokenStream label="lower (x2)" tokens={[{ label: "lo", color: colors.gold }, "w", "e", "r", "</w>"]} />
              </div>
            ),
          },
          {
            label: "step 5 — merge 'lo + w' (count 7)",
            render: () => (
              <div>
                <TokenStream label="low (x5)" tokens={[{ label: "low", color: colors.gold }, "</w>"]} />
                <TokenStream label="lower (x2)" tokens={[{ label: "low", color: colors.gold }, "e", "r", "</w>"]} />
              </div>
            ),
          },
          {
            label: "step 6 — merge 'n + e' (count 6)",
            render: () => (
              <div>
                <TokenStream label="newest (x6)" tokens={[{ label: "ne", color: colors.gold }, "w", "est</w>"]} />
              </div>
            ),
          },
          {
            label: "step 7 — merge 'ne + w' (count 6)",
            render: () => (
              <div>
                <TokenStream label="newest (x6)" tokens={[{ label: "new", color: colors.gold }, "est</w>"]} />
              </div>
            ),
          },
          {
            label: "step 8 — merge 'new + est</w>' (count 6)",
            render: () => (
              <div>
                <TokenStream label="newest (x6)" tokens={[{ label: "newest</w>", color: colors.green }]} />
                <Prose>
                  The entire six-occurrence word <Code>newest</Code> is now a single token. From this point on, every instance of <Code>newest</Code> in new text will encode to one integer.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The interactive trainer below lets you run the same algorithm on any corpus of your choosing. The default is the Sennrich example, which produces the trace above. Paste in a paragraph of ordinary English and watch the first dozen merges: the usual suspects — <Code>th</Code>, <Code>ing{"</w>"}</Code>, <Code>ed{"</w>"}</Code>, <Code>{" the"}</Code> — fall out in roughly the order their frequencies would predict. Paste in a paragraph of a language you do not read and the algorithm produces reasonable-looking subword units anyway, because statistical regularity in character co-occurrence is a fair proxy for morpheme structure in most scripts. That is BPE's universality in practice: it is a compression scheme that happens to cut along meaningful seams more often than not.
      </Prose>

      <BPETrainer />

      <Prose>
        The comparison below shows the same sentence — <Code>{`"The unbelievably fast model tokenizes 1234 café emojis like this."`}</Code> — tokenized by four real production systems. The outputs are verbatim from running each tokenizer in Python. Notice how differently they treat the same inputs: the digits <Code>1234</Code>, the accented <Code>café</Code>, the compound word <Code>emojis</Code>, and — most instructively — the leading space before each word.
      </Prose>

      <TokenStream
        label="tiktoken cl100k_base (gpt-4, gpt-3.5) — 15 tokens"
        tokens={["The", " unbelie", "vably", " fast", " model", " token", "izes", " ", "123", "4", " café", " emojis", " like", " this", "."]}
      />

      <TokenStream
        label="tiktoken o200k_base (gpt-4o) — 14 tokens"
        tokens={["The", " unbelievably", " fast", " model", " token", "izes", " ", "123", "4", " café", " emojis", " like", " this", "."]}
      />

      <TokenStream
        label="bert-base-uncased wordpiece — 19 tokens"
        tokens={["the", "un", "##bel", "##ie", "##va", "##bly", "fast", "model", "token", "##izes", "123", "##4", "cafe", "em", "##oj", "##is", "like", "this", "."]}
      />

      <TokenStream
        label="t5-small sentencepiece unigram — 21 tokens"
        tokens={["▁The", "▁unbe", "lie", "v", "ably", "▁fast", "▁model", "▁token", "ize", "s", "▁12", "34", "▁café", "▁", "e", "m", "oji", "s", "▁like", "▁this", "."]}
      />

      <TokenStream
        label="gpt-2 byte-level bpe — 15 tokens (Ġ marks a leading space)"
        tokens={["The", "Ġunbelievably", "Ġfast", "Ġmodel", "Ġtoken", "izes", "Ġ12", "34", "ĠcafÃ©", "Ġem", "oj", "is", "Ġlike", "Ġthis", "."]}
      />

      <Prose>
        Several things jump out. The newest OpenAI tokenizer (<Code>o200k_base</Code>) is the most economical on this input, reflecting both a larger vocabulary and more recent training data that has seen <Code>unbelievably</Code> and <Code>café</Code> as unit tokens. BERT's WordPiece lowercases the input and strips the diacritic from <Code>café</Code> — a lossy transformation that is fine for its intended use but would break generation. T5's Unigram tokenizer fragments <Code>emojis</Code> into five pieces (<Code>▁</Code>, <Code>e</Code>, <Code>m</Code>, <Code>oji</Code>, <Code>s</Code>), which reflects how little it saw the word in its pretraining corpus. GPT-2's byte-level BPE renders the non-ASCII <Code>café</Code> as <Code>cafÃ©</Code>, not because it got the word wrong but because the two UTF-8 bytes of <Code>é</Code> are being displayed through a byte-to-character alphabet that deliberately avoids control characters. Decoded back to text, it reconstructs <Code>café</Code> exactly. The <Code>Ġ</Code> glyph is GPT-2's convention for a leading space — the same role that <Code>▁</Code> plays in SentencePiece, a visible stand-in for whitespace that would otherwise be invisible in the token stream.
      </Prose>

      <Prose>
        The plot below sketches the general shape of the vocabulary-size versus sequence-length trade-off. The exact numbers depend heavily on the corpus, but the shape — sharply diminishing returns past roughly 30,000 merges for monolingual English, softer diminishing returns that keep paying out to 200,000+ for multilingual corpora — is consistent across every measurement in the literature.
      </Prose>

      <Plot
        label="vocabulary size vs. average tokens per word (approximate)"
        width={520}
        height={240}
        xLabel="log10 vocab size"
        yLabel="avg tokens per word"
        series={[
          { name: "English-only corpus", points: [[2, 4.2], [3, 2.4], [3.5, 1.8], [4, 1.4], [4.5, 1.18], [5, 1.08], [5.3, 1.05]] },
          { name: "100-language corpus", points: [[2, 6.5], [3, 4.8], [3.5, 3.8], [4, 2.9], [4.5, 2.2], [5, 1.75], [5.3, 1.55]] },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix — when to use what</H2>

      <Prose>
        The practical question is rarely "which algorithm is best" in the abstract. It is "given a specific project with specific constraints, which of the five or six mainstream configurations should I pick." The table below lines up the distinctions that actually show up in those decisions.
      </Prose>

      <CodeBlock>
{`                    BPE            WordPiece      Unigram LM     SP+BPE         SP+Unigram     byte-level BPE
merge criterion     pair freq      PMI / LR       EM prune       pair freq      EM prune       pair freq
direction           bottom-up      bottom-up      top-down       bottom-up      top-down       bottom-up
vocabulary          closed         closed         closed         closed         closed         open (bytes)
output              deterministic  deterministic  probabilistic  deterministic  probabilistic  deterministic
reversible          no*            no (##)        yes (▁)        yes (▁)        yes (▁)        yes (bytes)
multi-language      weak           weak           strong         strong         strongest      strong
pre-tokenization    required       required       optional       none           none           required
typical vocab       30k–50k        30k            30k–250k       30k–128k       30k–250k       50k–200k
famous users        early GPTs,    BERT,          ALBERT,        Llama 1/2/3,   T5, Gemma,     GPT-2 through
                    NMT systems    DistilBERT,    XLNet          Mistral,       mBART, NLLB     GPT-4o, Claude,
                                   ELECTRA                       DeepSeek                       most modern LMs

* reversible with </w> marker but loses some whitespace fidelity in practice`}
      </CodeBlock>

      <Prose>
        Two notes on the table. First, "famous users" is the single most misleading column — every production tokenizer has a specific byte-handling, normalization, and pre-tokenization pipeline that means "uses BPE" tells you only part of the story. Llama 1, Llama 2, and Llama 3 all use SentencePiece configured for BPE, but their vocabularies are not interchangeable; Llama 3's 128,256-token vocabulary is roughly four times Llama 2's and was retrained from scratch. Claude uses byte-level BPE with its own tokenizer. GPT-4o's <Code>o200k_base</Code> is byte-level BPE at twice the vocabulary of <Code>cl100k_base</Code>. The algorithm is load-bearing but rarely the decisive variable.
      </Prose>

      <Prose>
        Second, the "multi-language" column compresses a genuinely complicated comparison. Raw BPE over a mixed-language corpus tends to over-allocate vocabulary to whichever script dominates the training data; a 30k BPE tokenizer trained on a 90% English, 10% everything-else corpus will produce 3× to 5× longer sequences in the non-English languages than in English. Unigram handles this somewhat better because its EM-based pruning tends to balance token utility across the corpus more evenly. SentencePiece with Unigram plus a high vocabulary size plus careful corpus balancing is what multilingual models like mBART, NLLB, and BLOOM actually ship with, and it is the reason they can be called "multilingual" with a straight face rather than "English with some extras."
      </Prose>

      <Prose>
        A short decision tree, applied to real projects:
      </Prose>

      <Prose>
        <strong>If you are training a new frontier English-dominant model:</strong> byte-level BPE, 100k–200k vocabulary, matched closely to the training data distribution. This is what the frontier labs ship, because it gives unconditional coverage of any UTF-8 input (emoji, code, rare scripts) at the cost of slightly awkward tokenization of non-ASCII text. The open-vocabulary property matters operationally: users will paste anything into a chat interface, and a tokenizer that can never fail on an input is one fewer production incident.
      </Prose>

      <Prose>
        <strong>If you need strong multilingual coverage out of the box:</strong> SentencePiece with Unigram, 150k–250k vocabulary, <Code>character_coverage=0.9995</Code>. The SentencePiece <Code>▁</Code> convention handles Chinese, Japanese, and Thai — languages that do not put spaces between words — without hand-engineered per-language rules. Unigram's top-down pruning balances token utility across scripts better than BPE's bottom-up merging, which otherwise starves low-resource languages.
      </Prose>

      <Prose>
        <strong>If you need deterministic, fast inference-time tokenization with perfect fidelity to an existing model:</strong> whatever tokenizer ships with the model, loaded from its Rust implementation. Do not reimplement it. Do not port it. Load the reference bytes and use them. Small discrepancies between a reimplemented tokenizer and the reference will corrupt every serialized conversation, cache, and fine-tuning dataset you produce and the bug may not surface for weeks.
      </Prose>

      <Prose>
        <strong>If you are fine-tuning on a domain where the base vocabulary is mismatched:</strong> keep the base tokenizer, but consider adding a small set of special tokens for the domain's high-frequency units — function names, protein codes, chemical element abbreviations. A few hundred added tokens are cheap in embedding parameters and can drastically shorten the tokenized sequences for your specific use case. Do <em>not</em> retrain the tokenizer from scratch unless you are also training the model from scratch, because the existing embedding rows are bound to specific integer ids and a new tokenizer will scramble the mapping.
      </Prose>

      <Prose>
        <strong>If you are doing research on token distributions or probing the tokenizer's biases:</strong> Unigram or SentencePiece-Unigram, because it is the only mainstream algorithm that assigns probabilities to segmentations. BPE gives you one answer per input; Unigram gives you a distribution over answers, which you can marginalize, sample from, or treat as a lattice.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The costs of tokenization split along four axes, and they scale very differently. Understanding which axis dominates in your setting is the difference between a tokenizer training run that takes twenty minutes and one that takes two weeks, and between an inference loop that is tokenizer-bound and one that is completely dominated by model compute.
      </Prose>

      <Prose>
        <strong>Corpus size at training time.</strong> The pure Python implementation in section 4 runs in roughly <Code>O(n · m · v)</Code> time, where <Code>n</Code> is the number of symbols in the corpus, <Code>m</Code> is the number of merges to learn, and <Code>v</Code> is the current vocabulary size. On a 1-billion-token training corpus with a 50,000-merge target, that is a lot of arithmetic — the naive loop is effectively unusable past ten million tokens. Production implementations (HuggingFace, SentencePiece) are closer to <Code>O(n · m)</Code> with aggressive caching of per-word segmentations, and they parallelize across CPU cores. On a 1-billion-token corpus those implementations finish in one to three hours on a modern 32-core machine. At 1-trillion-token pretraining scale, even that strains: training a Llama 3–class tokenizer can take six to twelve hours of wall-clock time on a dedicated box, which is why frontier labs treat tokenizer training as a scheduled artifact rather than something you iterate on interactively.
      </Prose>

      <Prose>
        <strong>Vocabulary size.</strong> The embedding table is a dense matrix of shape <Code>(V, d)</Code>, where <Code>V</Code> is the vocabulary and <Code>d</Code> is the model dimension. Doubling the vocabulary doubles the embedding parameters and doubles the output softmax parameters (assuming tied embeddings — if untied, the cost is separate). For a 70-billion-parameter model with <Code>d = 8192</Code>, a 32k vocabulary costs 0.26B parameters; a 256k vocabulary costs 2.1B. That is a meaningful fraction of the budget. During training, the softmax cross-entropy computation is effectively <Code>O(V · d)</Code> per token; at inference with KV-caching the cost is <Code>O(V · d)</Code> per generated token, which becomes non-trivial when <Code>V</Code> is in the hundreds of thousands. Modern models use 100k–256k as the sweet spot: large enough to handle multiple languages and avoid fragmenting common words, small enough not to dominate the parameter budget.
      </Prose>

      <Prose>
        <strong>Sequence length at inference.</strong> Tokenization itself runs in near-linear time with a trie-based matcher — an Aho-Corasick automaton preprocesses the vocabulary into a structure that matches any vocabulary prefix against a stream in a single pass. For a ten-megabyte input this is milliseconds. The naive Python implementation, which rescans the full merge list for every word, is closer to <Code>O(n · m)</Code> and becomes noticeable on long documents. If tokenization ever shows up in a profile, the fix is always the same: switch to the Rust implementation, or cache the tokenized output by input hash.
      </Prose>

      <Prose>
        <strong>Number of distinct scripts.</strong> This is the axis that behaves non-linearly. A tokenizer trained primarily on English will handle Spanish and French well, Russian and Greek tolerably, Hindi and Arabic badly, Chinese and Japanese very badly. The reason is not the algorithm — it is that frequency-based merging puts merges where the frequency is, and the frequency is in English. Non-English text gets left as individual characters or worse, individual bytes, and the sequence length for the same semantic content grows by three to five times. Unigram helps a little because its EM-based pruning is less aggressive about giving vocabulary to the dominant language, but the right fix is corpus balancing — sampling non-English text at a rate proportional to what you want the tokenizer to support rather than at its natural frequency in crawled data. This is its own large topic, covered in the multilingual vocabulary discussion.
      </Prose>

      <Plot
        label="illustrative: tokenizer training wall-clock vs. corpus size"
        width={520}
        height={240}
        xLabel="log10 corpus tokens"
        yLabel="log10 seconds"
        series={[
          { name: "pure Python (this topic)", points: [[6, 1.5], [7, 2.8], [8, 4.2], [9, 5.6]] },
          { name: "HuggingFace (Rust, 32c)", points: [[6, 0.3], [7, 1.2], [8, 2.2], [9, 3.3], [10, 4.4], [11, 5.5]] },
          { name: "SentencePiece (C++, 32c)", points: [[6, 0.5], [7, 1.4], [8, 2.4], [9, 3.5], [10, 4.6], [11, 5.7]] },
        ]}
      />

      <Prose>
        The picture above is illustrative — the actual numbers depend heavily on merge count, vocabulary size, and hardware — but the relative positions are stable. Pure Python is a useful pedagogical tool and a non-starter for any real training corpus. The Rust and C++ implementations are roughly equivalent on CPU-bound throughput, with HuggingFace's implementation having a slight edge on small corpora and SentencePiece's scaling slightly better into the trillion-token regime. Neither is the bottleneck in a modern pretraining pipeline; the bottleneck is the data-loading and shuffling stack around them.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <Prose>
        Ten things that reliably go wrong.
      </Prose>

      <Prose>
        <strong>1. Forgetting the end-of-word marker.</strong> Build a BPE tokenizer without <Code>{"</w>"}</Code> or an equivalent leading-space convention, and the merger cannot distinguish <Code>est</Code> at the end of <Code>fastest</Code> from <Code>est</Code> in the middle of <Code>estimate</Code>. The resulting vocabulary learns fewer suffix tokens, assigns longer tokenizations to common morphologically rich words, and loses five to ten percent of compression on English. Symptom: more tokens per word than a reference implementation. Fix: add the marker, or use a library that has one built in.
      </Prose>

      <Prose>
        <strong>2. Non-deterministic pre-tokenization.</strong> Two runs on the same text produce different tokens because Unicode normalization differs, or because the pre-tokenizer depends on a system locale. This is devastating for any system that compares hashes of tokenized text — deduplication pipelines, caches, pre-computed prompts. Symptom: hash mismatches on text that looks identical. Fix: pin the normalization form (NFC or NFKC) explicitly at the tokenizer boundary, and never rely on implicit platform behavior.
      </Prose>

      <Prose>
        <strong>3. Greedy inference without BPE-dropout.</strong> A deterministic tokenizer applied at both training and inference systematically biases attention toward the specific subword partition that was chosen. For rare words this creates a calibration gap: the model has been exposed to only one of several plausible segmentations, and its confidence reflects that narrow exposure. Symptom: brittle behavior on rare or noisy inputs. Fix: Unigram's subword regularization samples alternative segmentations during training; BPE-dropout (Provilkov et al., 2020) provides the same capability for BPE models. Neither is standard in frontier models, but both help on morphologically rich languages.
      </Prose>

      <Prose>
        <strong>4. Leaking training data through the tokenizer.</strong> Rare but real. A merge learned during tokenizer training can include a substring of a rare name, email address, or phone number that appeared enough times in the corpus to earn a merge. The merge then becomes a single token in the final vocabulary, and anyone with API access can enumerate vocabulary tokens to detect that the name was in the training data. Symptom: low-frequency proper nouns showing up as single tokens in a tokenizer trained on a corpus that should not have contained them. Fix: deduplicate and filter PII from the tokenizer training corpus with the same care as the model training corpus; they are typically the same corpus but the filtering step is sometimes skipped for the tokenizer.
      </Prose>

      <Prose>
        <strong>5. Whitespace handling in chat templates.</strong> <Code>"hello"</Code> and <Code>" hello"</Code> tokenize differently — usually as one token and two tokens respectively, or as two different single tokens. When chat templates concatenate system prompts, role markers, and user messages, an off-by-one space can blow up the tokenization of the next segment and produce subtly wrong model behavior. Symptom: chat model output drifts in quality when a leading or trailing space is added or removed from a template. Fix: canonicalize whitespace at the template boundary, and test the tokenized form of the full prompt, not just the segments in isolation.
      </Prose>

      <Prose>
        <strong>6. Adding tokens post-hoc to a pretrained model.</strong> HuggingFace's <Code>tokenizer.add_tokens([...])</Code> appends new vocabulary items, but their embedding rows are initialized randomly and the model has never seen them. Fine-tuning from there works only if you train the new embeddings long enough that they converge to something meaningful, and in practice it often takes surprisingly much data. Symptom: the model produces garbage whenever the new tokens appear. Fix: either train the new tokens with a careful schedule (freeze everything else for the first few hundred steps, then unfreeze), or initialize the new embedding rows to the average of the tokens they are replacing.
      </Prose>

      <Prose>
        <strong>7. Numbers tokenize badly for arithmetic.</strong> The number <Code>1234</Code> tokenizes as <Code>[1234]</Code>, <Code>[12, 34]</Code>, <Code>[1, 234]</Code>, or <Code>[123, 4]</Code> depending on which substrings happened to be common in the training corpus. The model has to learn digit arithmetic through an interface that scrambles place value. Symptom: arithmetic accuracy drops sharply on numbers with specific digit lengths that tokenize poorly. Fix: several recent models (including LLaMA 3 and some Gemma variants) pre-split numbers into individual digits before BPE. This is a hand-engineered patch, but the cleanest one available, and it produces substantially better arithmetic behavior.
      </Prose>

      <Prose>
        <strong>8. Emoji and CJK fragmenting under English-trained BPE.</strong> Without byte-level fallback, a tokenizer that was not trained on emoji will emit an <Code>{"<UNK>"}</Code> or split the emoji into its constituent Unicode bytes, each of which becomes a token. Chinese and Japanese text fragments similarly. Symptom: three to five times as many tokens for CJK content as for semantically equivalent English, and visible mojibake in outputs. Fix: byte-level BPE (all modern frontier models), or add emoji coverage and CJK balance to the training corpus explicitly, or use SentencePiece with <Code>character_coverage=0.9995</Code>.
      </Prose>

      <Prose>
        <strong>9. Vocabulary-size mismatch with the embedding matrix.</strong> A tokenizer JSON has 32,016 tokens. The model was exported with <Code>vocab_size=32000</Code>. The extra sixteen tokens silently collide with the padding rows of the embedding matrix and produce nonsense on any input that uses them. Symptom: certain inputs produce obviously wrong output, others are fine. Fix: assert <Code>len(tokenizer) == model.config.vocab_size</Code> at load time, every time.
      </Prose>

      <Prose>
        <strong>10. Chat template role tokens missing from the base vocabulary.</strong> A model is fine-tuned with <Code>{"<|system|>"}</Code>, <Code>{"<|user|>"}</Code>, <Code>{"<|assistant|>"}</Code> as role markers. The base tokenizer does not have these as single tokens, so they split into four or five pieces each, the model never learns them as role markers, and chat template formatting subtly breaks. Symptom: the model ignores role boundaries in multi-turn conversations. Fix: add role markers as special tokens <em>before</em> fine-tuning, and ensure the tokenizer treats them as atomic units that never get split.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The canonical reference list for the algorithms discussed above. All six were cross-checked against their published venues during the preparation of this topic; dates, titles, and arXiv ids reflect the verified records.
      </Prose>

      <Prose>
        <strong>1.</strong> Gage, Philip. "A New Algorithm for Data Compression." <em>The C Users Journal</em>, Volume 12, Issue 2, February 1994, pages 23–38. The original byte-pair encoding paper. Published in a compression-focused developer journal, not widely read in NLP until the 2015 Sennrich reintroduction. The algorithm Gage describes is byte-level and aimed at memory-constrained decompression; the modern NLP usage descends directly from it.
      </Prose>

      <Prose>
        <strong>2.</strong> Sennrich, Rico; Haddow, Barry; Birch, Alexandra. "Neural Machine Translation of Rare Words with Subword Units." <em>Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)</em>, Volume 1, pages 1715–1725. arXiv:1508.07909 (first submitted August 2015, final version June 2016). The paper that brought BPE into NLP, with the classic <Code>low / lower / newest / widest</Code> worked example used throughout this topic.
      </Prose>

      <Prose>
        <strong>3.</strong> Schuster, Mike; Nakajima, Kaisuke. "Japanese and Korean Voice Search." <em>2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)</em>, Kyoto, 25–30 March 2012, pages 5149–5152. The original WordPiece paper, framed as a voice search problem for languages without whitespace word boundaries. The likelihood-gain scoring function is the key algorithmic contribution.
      </Prose>

      <Prose>
        <strong>4.</strong> Kudo, Taku. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." <em>Proceedings of ACL 2018</em>, pages 66–75. arXiv:1804.10959. The Unigram LM paper. Introduces both the top-down EM-based algorithm and the idea of sampling segmentations during training as a regularizer.
      </Prose>

      <Prose>
        <strong>5.</strong> Kudo, Taku; Richardson, John. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." <em>Proceedings of EMNLP 2018: System Demonstrations</em>, pages 66–71. arXiv:1808.06226. The library paper. Describes the <Code>▁</Code> whitespace-as-character trick and argues for treating tokenization as an end-to-end, language-agnostic component.
      </Prose>

      <Prose>
        <strong>6.</strong> Radford, Alec; Wu, Jeff; Child, Rewon; Luan, David; Amodei, Dario; Sutskever, Ilya. "Language Models are Unsupervised Multitask Learners." OpenAI technical report, 2019. The GPT-2 paper. The tokenizer described in Section 2.2 is byte-level BPE over UTF-8 — the direct ancestor of the tokenizers used by GPT-3, GPT-4, and most modern autoregressive models. It is the paper to cite for the byte-level variant specifically.
      </Prose>

      <Callout accent="gold">
        Secondary but worth flagging: Provilkov, Ivan; Emelianenko, Dmitrii; Voita, Elena. "BPE-Dropout: Simple and Effective Subword Regularization." <em>ACL 2020</em>. The bridge between BPE and Kudo's subword regularization idea, applied to an already-trained BPE tokenizer by randomly skipping merges at inference.
      </Callout>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Five short problems. Spend ten minutes per problem before peeking at the answer. The point is to catch confusions; the problems are chosen so that doing them wrong tells you something specific about what you have not internalized yet.
      </Prose>

      <Prose>
        <strong>Problem 1.</strong> Given the corpus <Code>{`"ab ab ab ba ba"`}</Code>, perform three BPE merges by hand (with the <Code>{"</w>"}</Code> marker convention). What is the final vocabulary and what are the final segmentations of each word?
      </Prose>

      <Callout accent="green">
        Initial segmentations: <Code>a b {"</w>"}</Code> (×3), <Code>b a {"</w>"}</Code> (×2). Initial pair counts: <Code>(a,b)</Code> = 3, <Code>(b,{"</w>"})</Code> = 3, <Code>(b,a)</Code> = 2, <Code>(a,{"</w>"})</Code> = 2. Ties can break either way; resolving alphabetically, merge 1 is <Code>(a,b)</Code> → <Code>ab</Code>. After merge 1: <Code>ab {"</w>"}</Code> (×3), <Code>b a {"</w>"}</Code> (×2). New pair counts: <Code>(ab,{"</w>"})</Code> = 3, <Code>(b,a)</Code> = 2, <Code>(a,{"</w>"})</Code> = 2. Merge 2: <Code>(ab,{"</w>"})</Code> → <Code>ab{"</w>"}</Code>. After merge 2: <Code>ab{"</w>"}</Code> (×3), <Code>b a {"</w>"}</Code> (×2). Remaining pair counts: <Code>(b,a)</Code> = 2, <Code>(a,{"</w>"})</Code> = 2. Merge 3 (tie-break alphabetically): <Code>(a,{"</w>"})</Code> → <Code>a{"</w>"}</Code>. Final vocabulary: <Code>{"{a, b, </w>, ab, ab</w>, a</w>}"}</Code>. Final segmentations: <Code>ab{"</w>"}</Code> (×3) and <Code>b + a{"</w>"}</Code> (×2).
      </Callout>

      <Prose>
        <strong>Problem 2.</strong> Using the from-scratch implementation in section 4, write a test that asserts <Code>decode(encode(x)) == x</Code> for a list of ten words, half of which were in the training corpus and half of which were not. Which round-trip would fail, and why?
      </Prose>

      <Callout accent="green">
        The round-trip succeeds for any whitespace-separated input as long as every character in the input was in the training corpus. It fails when the input contains a character the tokenizer never saw during training — for the classic corpus, any character outside <Code>{"{d, e, i, l, n, o, r, s, t, w}"}</Code> would produce a <Code>KeyError</Code> in <Code>encode_word</Code> (the implementation in section 4 does not add an <Code>{"<UNK>"}</Code> fallback). This is the single most important thing the production implementations add: byte-level BPE cannot fail on any input because every byte value 0–255 is in the base vocabulary by construction.
      </Callout>

      <Prose>
        <strong>Problem 3.</strong> Why does <Code>BPETokenizer().train(corpus, num_merges=0).encode("hello")</Code> return one token per character? Why is this the <em>correct</em> behavior rather than a bug?
      </Prose>

      <Callout accent="green">
        With zero merges, the learned merge list is empty, so the encoder never finds a pair it knows how to merge. Every input word is returned as its character decomposition plus <Code>{"</w>"}</Code>. This is correct because a BPE tokenizer at <Code>num_merges=0</Code> is definitionally a character-level tokenizer — no merges means no subwords, just the base alphabet. The interesting question is what <Code>num_merges=1</Code> does, which is: merge the single most common adjacent pair in the training corpus. On normal English text that is usually <Code>t + h</Code> or <Code>e + {"</w>"}</Code>, neither of which would affect the tokenization of <Code>"hello"</Code>.
      </Callout>

      <Prose>
        <strong>Problem 4.</strong> Explain why <Code>tiktoken.get_encoding("cl100k_base").encode("hello world")</Code> is not equal to <Code>encode("hello") + encode(" world")</Code>, in terms of what the tokenizer was trained to see.
      </Prose>

      <Callout accent="green">
        Modern byte-level BPE tokenizers fuse the leading space into the token that follows. <Code>"hello world"</Code> is tokenized as <Code>["hello", " world"]</Code> (two tokens), where <Code>" world"</Code> is a single token that includes the leading space. <Code>encode("hello")</Code> alone produces <Code>["hello"]</Code>; <Code>encode(" world")</Code> produces <Code>[" world"]</Code>. Concatenating gives the same result in this specific case — but <Code>encode("world")</Code> (no leading space) produces <Code>["world"]</Code>, a different token from <Code>" world"</Code>. The general lesson is that <Code>encode(a + b) != encode(a) + encode(b)</Code> almost always, because the merges at the boundary depend on the full character context. This is why you cannot safely concatenate pre-tokenized sequences without re-tokenizing the full string, and why chat templates are so sensitive to whitespace.
      </Callout>

      <Prose>
        <strong>Problem 5.</strong> You are training a new language model dominated by Arabic text — say 80% Arabic, 20% English. Should you use BPE, WordPiece, or Unigram? Pick one and defend the choice in three sentences.
      </Prose>

      <Callout accent="green">
        Unigram, inside SentencePiece, with a 150k–200k vocabulary. Unigram's top-down pruning allocates vocabulary more evenly across scripts than BPE's bottom-up merging, which is especially valuable when one script (Arabic's connected-letter forms and rich morphology) has very different statistical properties from the other. SentencePiece's <Code>▁</Code> convention handles the mix of Arabic and English whitespace conventions without language-specific rules, and the large vocabulary prevents the Arabic side from fragmenting into long character sequences that would inflate training and inference cost. BPE would work but would under-allocate to Arabic morphology; WordPiece's PMI scoring has similar biases to BPE in this regime and is rarely used for multilingual models in production.
      </Callout>

      <Prose>
        The tokenizer is the layer of the stack that models the least and matters more than it should. The next topic picks up from this point, looking at what happens when you drop the subword step entirely and let the model learn over raw bytes.
      </Prose>
    </div>
  ),
};

export default tokenization;
