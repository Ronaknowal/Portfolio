import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, ByteGrid, Plot } from "../../components/viz";
import { colors } from "../../styles";

const byteLevelTokenization = {
  title: "Byte-Level Tokenization & Token-Free Models",
  readTime: "32 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Open any production subword tokenizer trained in 2019 or 2020 — the original GPT-2 tokenizer, the BERT tokenizer, the early SentencePiece models that shipped with T5 — and feed it three inputs: the English sentence <Code>"The quick brown fox."</Code>, the same sentence translated into Hindi, and a short Python function. The English sentence costs five or six tokens. The Hindi sentence, which is roughly the same length on the page, can cost thirty. The Python function, depending on whitespace and identifiers, often costs more than its character count because the tokenizer has never seen those identifiers and has no useful merges over them. Feed the same three things as a bare emoji — the globe, <Code>🌍</Code> — and a <em>pre-byte-level</em> character BPE tokenizer would emit an <Code>[UNK]</Code> and lose the input entirely. That failure mode is the pressure that forced the whole field toward bytes.
      </Prose>

      <Prose>
        A subword vocabulary is, at bottom, a frozen belief about what text looks like. A sensible English-trained vocabulary has dedicated tokens for <Code>"ing"</Code>, <Code>" the"</Code>, <Code>"tion"</Code>, and for common proper nouns that happened to appear in the training corpus. It has, crucially, no dedicated tokens for anything that was not in the training corpus — a new emoji, a rare script, a binary blob accidentally fed through the API, the bytes of a JPEG header that someone pasted in. A vocabulary carries the assumption that "what comes in" will resemble "what we saw." When the assumption holds the tokenizer is near-optimal. When it doesn't — and for any sufficiently multilingual, multi-modal, or adversarial application it won't — the tokenizer either fragments the input pathologically or drops it on the floor.
      </Prose>

      <Prose>
        Byte-level tokenization drops the assumption. Every input any computer has ever stored is, at some level, a sequence of bytes. UTF-8 is a lossless, variable-length encoding of Unicode, and every Unicode string resolves, without ambiguity, to a sequence of bytes drawn from a fixed alphabet of 256 values. There is no such thing as an "unseen byte" — all 256 of them are in the vocabulary by construction. There is no such thing as an <Code>[UNK]</Code> token, because there is no way for an input to fail to be encodable. The universality is free. The only question, and it is a real one, is what it costs.
      </Prose>

      <Prose>
        The historical arc of byte-level methods is tighter than most of NLP's threads. In February 2019, Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever published the GPT-2 technical report, "Language Models are Unsupervised Multitask Learners." Buried in section 2.2 of that paper is the sentence that changed the ecosystem: "A byte-level version of BPE only requires a base vocabulary of size 256." They argued directly from the coverage-vs-vocab tradeoff — you want a vocabulary small enough to train efficiently and large enough to give common units their own tokens, and starting from bytes rather than from Unicode codepoints is how you square the circle. GPT-2's tokenizer was byte-level BPE with one engineering twist: a fixed reversible mapping from raw byte values to printable Unicode characters, so the merge strings in <Code>vocab.json</Code> stayed human-readable. Every major byte-level tokenizer since — <Code>tiktoken</Code>'s <Code>cl100k_base</Code> and <Code>o200k_base</Code>, Llama's SentencePiece-with-byte-fallback, Mistral and Qwen and Gemma — is a direct descendant of that decision.
      </Prose>

      <Prose>
        A parallel line of work pushed the idea further. If byte-level BPE is "bytes first, merges on top," could a transformer learn useful segmentation from the loss signal alone, with no tokenizer at all? Jonathan H. Clark and colleagues at Google answered that question in March 2021 with CANINE (arXiv:2103.06874), a character-level encoder with learned downsampling. Two months later Linting Xue et al. published ByT5 (arXiv:2105.13626), a T5 variant trained directly on raw UTF-8 bytes. Yi Tay and collaborators followed with Charformer (arXiv:2106.12672), which made segmentation itself differentiable. The line is still live: MambaByte (Wang et al., arXiv:2401.13660) combines byte-level inputs with the linear-time Mamba state-space model, and for the first time makes the sequence-length cost of byte processing genuinely affordable. This topic is about all of it — the byte-level BPE hybrid that ate production, and the token-free research line that keeps asking whether the hybrid is still the right answer.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        UTF-8 is the quiet miracle that makes all of this possible. Ken Thompson and Rob Pike designed it in 1992 on a placemat at a New Jersey diner, and the properties they chose have aged better than almost anything in systems software. Every Unicode codepoint becomes between one and four bytes. ASCII — the first 128 codepoints — is a single byte with the top bit zero, which means ASCII text is byte-identical to its UTF-8 representation. Latin-extended characters like <Code>é</Code>, <Code>ñ</Code>, and <Code>ü</Code> take two bytes. Greek, Cyrillic, Hebrew, Arabic, and most Indic scripts live in the two-to-three byte range. CJK ideographs take three bytes. Most emoji and supplementary-plane characters take four. The encoding is self-synchronizing — given any byte, you can tell by its top bits whether it is an ASCII byte, a leading byte, or a continuation byte — and it is prefix-free, so decoding is unambiguous.
      </Prose>

      <Prose>
        The shape of the alphabet a byte-level tokenizer sees is, therefore, fixed. The base vocabulary has exactly 256 items, one per byte value. For any Unicode string, any language, any emoji, any rare script, any binary blob pasted in by accident, the encoder runs UTF-8 and what comes out is a sequence of integers in <Code>[0, 256)</Code>. That's it. No training, no merges, no vocabulary file. If you stopped there you would have a working character-level tokenizer with universal coverage — it is what ByT5 uses — and you would pay for it in sequence length.
      </Prose>

      <ByteGrid text="Héllo 🌍 こんにちは" label="utf-8 bytes for a mixed-script string" />

      <Prose>
        Eighteen perceived units, twenty-seven bytes. The word <Code>Hello</Code> is green — ASCII, one byte per letter — until the <Code>é</Code>, which takes two bytes (<Code>c3 a9</Code>). The globe emoji takes four. Each hiragana character takes three. Gold cells are continuation bytes of multi-byte sequences. If your model reads bytes directly, every one of those cells is an attention position, and the model has to learn from the loss signal that <Code>c3 a9</Code> is a single grapheme and that certain four-byte sequences are pictograms.
      </Prose>

      <Prose>
        Byte-level BPE is the trick that buys universality cheaply. Take the 256-byte base vocabulary and run BPE on top of it — the same greedy-merge algorithm covered in the Tokenization topic, but starting from bytes instead of from Unicode characters. Common English words and suffixes, whose UTF-8 byte sequences merge early and often, still end up as single tokens. Common multi-byte sequences — the three bytes of a frequent Chinese character, the two bytes of a frequent accented Latin letter — also merge, because "common" in the training corpus translates directly to "high pair frequency" regardless of what encoding wrote it. What changes is the floor. A byte-level tokenizer cannot fail to represent an input, because even if the input is a four-byte emoji the tokenizer has never seen, the worst it can do is emit four single-byte tokens. There is no <Code>[UNK]</Code>. There is no silent truncation. There is no corner case where a rare Unicode character crashes the decoder.
      </Prose>

      <Prose>
        Token-free models take the same observation and push it further. If the tokenizer is a frozen prior about what chunks of text are worth grouping, why not let the model learn the prior from gradients? ByT5 answers "feed bytes straight in, no merges at all." CANINE answers "feed characters in, learn a downsampling conv stack that collapses them to coarser units inside the model." Charformer answers "make the segmentation itself a differentiable operation." MambaByte answers "skip the tokenizer and switch to an architecture where long sequences are cheap again." Each one is a different reading of the same underlying question: who decides where tokens begin and end — a frozen preprocessor trained before the model sees any gradients, or the model itself?
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The math behind byte-level tokenization is not heavy — most of it is discrete combinatorics over bytes — but three pieces are worth writing down explicitly.
      </Prose>

      <H3>3a. The byte-encoding function</H3>

      <Prose>
        Let <Code>U</Code> be the set of Unicode codepoints and <Code>B = {"{0, 1, ..., 255}"}</Code> be the set of byte values. UTF-8 defines a function:
      </Prose>

      <MathBlock>
        {"\\text{utf8}: U^* \\to B^*, \\qquad \\text{utf8}(c_1, c_2, \\ldots, c_n) = \\text{utf8}(c_1) \\| \\text{utf8}(c_2) \\| \\cdots \\| \\text{utf8}(c_n)"}
      </MathBlock>

      <Prose>
        where <Code>||</Code> is byte-string concatenation and each codepoint <Code>c</Code> maps to between one and four bytes depending on its value. The encoding has an inverse <Code>utf8⁻¹</Code> defined on the image of <Code>utf8</Code>, which is not all of <Code>B*</Code> — there are byte strings that do not correspond to valid UTF-8. Production byte-level tokenizers handle this gracefully by allowing invalid byte sequences at the tokenizer level (every byte is a valid token) and surfacing replacement characters at decode time if the model generates a byte sequence that cannot be UTF-8-decoded.
      </Prose>

      <H3>3b. The bytes-to-unicode remap</H3>

      <Prose>
        GPT-2's tokenizer does not store raw byte values in its merge file. It stores <em>printable Unicode characters</em>, with each of the 256 byte values mapped to a unique, visible character. The reason is mundane but important: the vocab file is JSON, merges are strings, and several byte values (0x00 through 0x1F, 0x7F, 0x80 through 0x9F) are control characters that either print as nothing, break JSON, or collide with whitespace handling inside the tokenizer's own splitter. The mapping is a fixed, reversible shuffle — call it <Code>β: B → Σ</Code>, where <Code>Σ</Code> is a set of 256 printable Unicode characters.
      </Prose>

      <Prose>
        The mapping is constructed explicitly. The "already printable" bytes — ASCII characters <Code>!</Code> (0x21) through <Code>~</Code> (0x7E), plus the Latin-1 supplement ranges <Code>¡</Code> (0xA1) through <Code>¬</Code> (0xAC) and <Code>®</Code> (0xAE) through <Code>ÿ</Code> (0xFF) — map to themselves. That accounts for 188 of the 256 byte values. The remaining 68 bytes (the control characters and whitespace characters that needed remapping) are mapped, in order, to the Unicode codepoints starting at <Code>U+0100</Code>. So byte 0x00 maps to <Code>U+0100</Code> (<Code>Ā</Code>), byte 0x01 to <Code>U+0101</Code>, and so on. The space character (byte 0x20) is in the "needed remapping" bucket and gets mapped to <Code>U+0120</Code> — the now-familiar <Code>Ġ</Code> that appears as a prefix on every word-initial GPT-2 token.
      </Prose>

      <MathBlock>
        {"\\beta(b) = \\begin{cases} \\text{chr}(b) & \\text{if } b \\in P \\\\ \\text{chr}(256 + i) & \\text{if } b \\text{ is the } i\\text{-th byte outside } P \\end{cases}"}
      </MathBlock>

      <Prose>
        where <Code>P</Code> is the set of 188 already-printable byte values. When you see a GPT-2 token like <Code>ĠHello</Code>, the <Code>Ġ</Code> is not a strange letter — it is byte 0x20 (space) viewed through <Code>β</Code>. The algorithm underneath is still byte-level BPE; the surface representation is a presentation choice.
      </Prose>

      <H3>3c. Sequence-length multiplier</H3>

      <Prose>
        The cost of going byte-level is, almost entirely, the sequence-length multiplier. For a given piece of text with <Code>N_char</Code> perceived characters and <Code>N_byte</Code> UTF-8 bytes, the ratio <Code>r = N_byte / N_char</Code> depends on the script. For ASCII, <Code>r = 1</Code>. For French with its occasional accents, <Code>r ≈ 1.05</Code>. For Greek or Cyrillic, <Code>r ≈ 2</Code>. For Chinese, Japanese, Korean, <Code>r = 3</Code> per character. For most emoji, <Code>r = 4</Code>. Under a subword-BPE tokenizer with a reasonable multilingual vocabulary, the same text might compress to <Code>N_tok ≈ N_byte / 3</Code> or better on English. So the ratio of byte-level sequence length to subword-BPE sequence length is roughly <Code>3 ⋅ (N_byte / N_char) / 1</Code> — around 3x for English, 6x for CJK.
      </Prose>

      <Prose>
        That would not matter much if transformer cost were linear in sequence length, but attention is quadratic. For self-attention, per-layer FLOPs scale as <Code>O(L^2 ⋅ d)</Code> where <Code>L</Code> is sequence length and <Code>d</Code> is hidden dimension. A 3x increase in <Code>L</Code> produces a 9x increase in attention compute. The KV cache grows by the same 3x factor. The feedforward cost grows linearly. Total cost scales somewhere between linear and quadratic depending on which component dominates at a given model size. At a minimum, byte-level training on an English corpus costs 3-4x the compute of a subword-BPE training run for the same perplexity target — and that is the argument that has kept byte-level models in research for most of the last five years. It is also why MambaByte matters: state-space models scale linearly in sequence length, so a 3x length increase costs 3x compute, not 9x, and the calculation flips.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block in this section was run against the embedded test inputs, and the comments in each block are the actual outputs, verbatim. Nothing is pseudo-code. By the end of the section you will have a complete byte-level BPE tokenizer — roughly eighty lines of pure Python, depending only on the <Code>regex</Code> module — and you will understand what each piece does and why.
      </Prose>

      <H3>4a. bytes_to_unicode — the printable-byte map</H3>

      <Prose>
        The most-copied function in byte-level tokenization is GPT-2's <Code>bytes_to_unicode</Code>. It constructs the <Code>β</Code> mapping from section 3b. Read it carefully — the structure is unusual.
      </Prose>

      <CodeBlock language="python">
{`def bytes_to_unicode():
    """Deterministic map from byte value (0-255) to a printable Unicode char.
    Structure: the 188 bytes that are already printable map to themselves.
    The remaining 68 bytes map, in order, to codepoints starting at U+0100."""
    bs = (
        list(range(ord("!"), ord("~") + 1))         # 0x21..0x7E
        + list(range(ord("\\u00a1"), ord("\\u00ac") + 1))  # 0xA1..0xAC
        + list(range(ord("\\u00ae"), ord("\\u00ff") + 1))  # 0xAE..0xFF
    )
    cs = bs[:]  # these bytes map to themselves
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)  # remaining bytes map into U+0100..
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

# Actual map samples (verified by running this code):
# byte 0x20 -> 'Ġ'  (codepoint U+0120)   ← space, now visible
# byte 0x21 -> '!'  (codepoint U+0021)   ← already printable
# byte 0x41 -> 'A'  (codepoint U+0041)   ← already printable
# byte 0x7e -> '~'  (codepoint U+007E)   ← already printable
# byte 0xe2 -> 'â'  (codepoint U+00E2)   ← already printable (Latin-1)
# byte 0xff -> 'ÿ'  (codepoint U+00FF)`}
      </CodeBlock>

      <Prose>
        Two things are worth noting. First, the mapping is deterministic — the same Python code produces the same table on every machine — and it is reversible. <Code>BYTE_DECODER</Code> is just the inverted dictionary. Second, the mapping is chosen, not derived. The specific choice of which bytes are "already printable" is a design decision — it avoids control characters, DEL (0x7F), and the Latin-1 soft hyphen (0xAD) because those either break printing or behave inconsistently across terminals. There's no mathematical reason for this particular split; it's a pragmatic choice that has been copied verbatim for six years because changing it would break tokenizer compatibility.
      </Prose>

      <H3>4b. encode_text_to_byte_tokens</H3>

      <CodeBlock language="python">
{`def encode_text_to_byte_tokens(text):
    """UTF-8-encode, then remap each byte through bytes_to_unicode.
    Output is a string of printable characters, one per input byte."""
    return "".join(BYTE_ENCODER[b] for b in text.encode("utf-8"))

def decode_byte_tokens_to_text(byte_str):
    """Reverse: each printable char -> byte, then UTF-8-decode."""
    return bytes(BYTE_DECODER[c] for c in byte_str).decode("utf-8", errors="replace")

# Actual output on 'Héllo 🌍 こんにちは' (verified):
# encoded (27 chars): HÃ©lloĠðŁĮįĠãģĵãĤĵãģ«ãģ¡ãģ¯
# decoded: 'Héllo 🌍 こんにちは'   ← lossless round trip`}
      </CodeBlock>

      <Prose>
        The round trip is lossless because <Code>β</Code> is a bijection. Notice what the encoded string looks like: every character in it is printable and visible in a JSON file or a terminal, but the semantic structure is preserved — each visible character corresponds to exactly one byte of the UTF-8 representation. The globe emoji <Code>🌍</Code>, whose UTF-8 bytes are <Code>[240, 159, 140, 141]</Code>, shows up as <Code>ðŁĮį</Code> — four visible characters because 240 is already printable (maps to <Code>ð</Code>), while 159, 140, and 141 are in the "needed remapping" bucket and land at <Code>U+0141</Code>, <Code>U+012E</Code>, <Code>U+012F</Code>. You can eyeball this and it looks like nonsense. That is the point: the BPE training algorithm doesn't care what the characters mean, only which pairs are adjacent and how often.
      </Prose>

      <H3>4c. train_byte_bpe</H3>

      <Prose>
        Training is the BPE loop from the Tokenization topic, adapted to run over the byte-mapped corpus instead of raw characters. The one thing you need that you didn't need before is a <em>pre-tokenizer</em> — a regex that chops the input stream into "words" before the byte mapping runs. Without it, the merge loop would be free to glue the last byte of one word to the first byte of the next, and you'd end up with vocabulary items like <Code>"d the"</Code> that cross semantic boundaries and generalize badly. GPT-2's pre-tokenizer regex is notorious for being load-bearing.
      </Prose>

      <CodeBlock language="python">
{`import regex as re
from collections import Counter

# GPT-2's pre-tokenizer regex.  It splits on contractions, then greedy
# runs of letters, digits, or punctuation, preserving leading spaces.
GPT2_SPLIT_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"""
)

def get_pair_stats(word_freqs):
    pairs = Counter()
    for symbols, freq in word_freqs.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    a, b = pair
    merged = a + b
    out = {}
    for symbols, freq in word_freqs.items():
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

def train_byte_bpe(corpus, num_merges):
    """Pre-tokenize, map to bytes, then BPE-merge over byte-strings."""
    pieces = GPT2_SPLIT_RE.findall(corpus)
    word_freqs = Counter(encode_text_to_byte_tokens(p) for p in pieces if p)
    word_freqs = {tuple(w): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_stats(word_freqs)
        if not pairs: break
        best = max(pairs, key=pairs.get)
        count = pairs[best]
        word_freqs = merge_pair(best, word_freqs)
        merges.append((best, count))
    return merges, word_freqs`}
      </CodeBlock>

      <Prose>
        The pre-tokenizer regex does four things. The leading alternatives match English contractions — <Code>'s</Code>, <Code>'re</Code>, <Code>'d</Code> — so the apostrophe splits cleanly. The middle alternatives match a run of letters or digits, optionally with a leading space, so <Code>" hello"</Code> pre-tokenizes as one piece and the space is glued to the word. The final alternative handles whitespace-only runs, giving them their own tokens. Everything after pre-tokenization is byte-mapped and then fed into the vanilla BPE merge loop.
      </Prose>

      <Prose>
        Run this on a small mixed corpus and watch what emerges.
      </Prose>

      <CodeBlock language="python">
{`CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the cat sat on the mat. café café café. "
    "def f(x): return x**2. def g(x): return x+1. "
    "Héllo 🌍 こんにちは こんにちは 世界 世界"
) * 20

merges, _ = train_byte_bpe(CORPUS, num_merges=60)
for i, (pair, count) in enumerate(merges[:10], 1):
    print(f"  {i:2d}. {pair[0]!r:>10} + {pair[1]!r:<10} count={count}")

# Actual output (verified by running this code):
#   1.        'ã' + 'ģ'        count=160   ← leading byte of hiragana
#   2.        't' + 'h'        count=80
#   3.       'th' + 'e'        count=80
#   4.        'Ġ' + 'c'        count=80    ← 'Ġ' is space, so this is " c"
#   5.       'Ġc' + 'a'        count=80
#   6.        'Ã' + '©'        count=80    ← bytes of 'é' in UTF-8
#   7.        'Ġ' + 'the'      count=60
#   8.        'Ġ' + 'd'        count=60
#   9.      'Ġca' + 'f'        count=60
#  10.     'Ġcaf' + 'Ã©'       count=60    ← the full token " café"`}
      </CodeBlock>

      <Prose>
        Three things jump out. First, merge 1 is <Code>ã + ģ</Code> — a pair of remapped bytes, not ASCII characters. What this actually represents is the first two bytes of the UTF-8 encoding of hiragana: every <Code>こ</Code>, <Code>ん</Code>, <Code>に</Code>, <Code>ち</Code>, <Code>は</Code> starts with byte 0xE3, and its second byte is almost always in the 0x81–0x83 range. Those two bytes remap to <Code>ã</Code> and <Code>ģ</Code>, and the merge loop discovers their adjacency before it discovers <Code>"th"</Code>. Byte-level BPE handles non-ASCII scripts by treating their byte-level patterns as just another source of frequency — no special case required. Second, merge 6 is <Code>Ã + ©</Code>: the two bytes of the UTF-8 encoding of <Code>é</Code> (<Code>0xC3 0xA9</Code>), which remap to <Code>Ã</Code> and <Code>©</Code>. The tokenizer doesn't know <Code>é</Code> is a letter; it just notices those two remapped bytes adjacent to each other nine times and merges them. Third, merge 10 — <Code>Ġcaf + Ã©</Code> — is the moment the tokenizer discovers that <Code>" café"</Code> is a single word worth a dedicated token. The merge list is, recursively, learning the structure of the corpus from the bottom up.
      </Prose>

      <H3>4d. encode_with_byte_bpe and decode_byte_bpe</H3>

      <CodeBlock language="python">
{`def encode_with_byte_bpe(text, merges):
    """Pre-tokenize, byte-map, apply merges by rank to each piece."""
    rank = {pair: i for i, (pair, _) in enumerate(merges)}
    out = []
    for piece in GPT2_SPLIT_RE.findall(text):
        if not piece: continue
        symbols = list(encode_text_to_byte_tokens(piece))
        while True:
            best_i, best_r = -1, float("inf")
            for i in range(len(symbols) - 1):
                r = rank.get((symbols[i], symbols[i + 1]), float("inf"))
                if r < best_r:
                    best_r, best_i = r, i
            if best_i == -1: break
            a, b = symbols[best_i], symbols[best_i + 1]
            symbols = symbols[:best_i] + [a + b] + symbols[best_i + 2:]
        out.extend(symbols)
    return out

def decode_byte_bpe(tokens):
    """Join, reverse the byte map, UTF-8-decode."""
    return decode_byte_tokens_to_text("".join(tokens))`}
      </CodeBlock>

      <Prose>
        Encoding applies the merge list to a pre-tokenized piece by repeatedly finding the adjacent pair with the lowest rank (earliest merge) and collapsing it, until no adjacent pair is in the merge table. Decoding is the reverse of the whole pipeline: concatenate the tokens to get back the remapped-byte string, reverse the byte map to get actual UTF-8 bytes, then UTF-8-decode. If the merges never cross UTF-8 character boundaries in a way that breaks decoding — which they usually don't, because merges operate on whole bytes and UTF-8 is self-synchronizing — the round trip is lossless.
      </Prose>

      <H3>4e. Round-trip tests on mixed input</H3>

      <CodeBlock language="python">
{`TEST = "Héllo 🌍 こんにちは"

toks = encode_with_byte_bpe(TEST, merges)
print(f"{len(toks)} tokens: {toks}")
print(f"decoded: {decode_byte_bpe(toks)!r}")
assert decode_byte_bpe(toks) == TEST

# Actual output (verified by running this code):
# 11 tokens: ['H', 'Ã©', 'l', 'l', 'o', 'Ġ',
#             'ð', 'Ł', 'Į', 'į', 'ĠãģĵãĤĵãģ«ãģ¡ãģ¯']
# decoded: 'Héllo 🌍 こんにちは'`}
      </CodeBlock>

      <Prose>
        Eleven tokens for "Héllo 🌍 こんにちは" under a tokenizer with only sixty merges. Look at what landed where. <Code>H</Code> is a single ASCII token (merge never happened because the corpus didn't have enough <Code>"H"</Code>-starting words). <Code>Ã©</Code> is the two-byte <Code>é</Code>, merged because the corpus had <Code>café</Code> repeated. <Code>l</Code>, <Code>l</Code>, <Code>o</Code> come out as individual letters because the corpus didn't have enough <Code>"Héllo"</Code> to learn the word (the training set had <Code>"the"</Code>, <Code>"café"</Code>, etc., but not enough of "Héllo"). <Code>Ġ</Code> is the remapped space. Then <Code>ð</Code>, <Code>Ł</Code>, <Code>Į</Code>, <Code>į</Code> are the four remapped bytes of the globe emoji — no merge, because the emoji only appeared once in training and didn't have the frequency to merge. Finally <Code>ĠãģĵãĤĵãģ«ãģ¡ãģ¯</Code> — the entire <Code>" こんにちは"</Code> collapsed into one token, because the corpus had it repeated twenty times and the merge loop was happy to build it up step by step. The tokenizer has "learned" a single-token representation for a specific Japanese word from a tiny mixed corpus, purely by frequency.
      </Prose>

      <Prose>
        Now check a few specific cases.
      </Prose>

      <CodeBlock language="python">
{`for ex in ["hello", "café", "🌍", "中", "def f(x): return x**2"]:
    toks = encode_with_byte_bpe(ex, merges)
    back = decode_byte_bpe(toks)
    print(f"{ex!r:>28} -> {len(toks):2d} tokens {toks}  round-trip={'OK' if back==ex else 'FAIL'}")

# Actual output (verified):
#                    'hello' ->  5 tokens ['h', 'e', 'l', 'l', 'o']  round-trip=OK
#                     'café' ->  4 tokens ['c', 'a', 'f', 'Ã©']  round-trip=OK
#                        '🌍' ->  4 tokens ['ð', 'Ł', 'Į', 'į']  round-trip=OK
#                        '中' ->  3 tokens ['ä', '¸', 'Ń']  round-trip=OK
#    'def f(x): return x**2' -> 12 tokens ['d', 'e', 'f', 'Ġf', '(', 'x', '):',
#                                          'Ġreturn', 'Ġx', '*', '*', '2']  round-trip=OK`}
      </CodeBlock>

      <Prose>
        The Chinese character <Code>中</Code>, which did not appear in the training corpus, still round-trips cleanly. Its UTF-8 bytes (<Code>0xE4 0xB8 0xAD</Code>) remap to <Code>ä ¸ Ń</Code>, and since no merge in the table touches any of those bytes, they come out as three separate tokens. The model would see three integers. The decoder joins them, reverses the remap, and UTF-8-decodes back to <Code>中</Code>. No special case. That is the coverage property, demonstrated on an input the tokenizer has never seen.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In practice nobody trains a production byte-level tokenizer from the Python above. OpenAI's <Code>tiktoken</Code> ships pre-trained encodings by name and is the reference implementation of their byte-level BPE. HuggingFace's <Code>tokenizers</Code> library implements the same algorithm in Rust with configurable pre-tokenizers. SentencePiece's BPE-with-byte-fallback is what Llama and Gemma use — slightly different mechanics but the same coverage guarantee. The API for loading a pre-trained tokenizer and running it looks like this.
      </Prose>

      <CodeBlock language="python">
{`import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4, GPT-3.5-turbo
ids = enc.encode("Héllo 🌍 こんにちは")
pieces = [enc.decode([i]) for i in ids]

# Actual output (verified with tiktoken 0.12.0):
# ids:    [39, 19010, 385, 11410, 234, 235, 220, 90115]
# pieces: ['H', 'él', 'lo', ' �', '�', '�', ' ', 'こんにちは']
# 8 tokens total`}
      </CodeBlock>

      <Prose>
        The <Code>�</Code> characters in the <Code>pieces</Code> list are not errors. Each one is a byte-level token that does not, on its own, constitute a valid UTF-8 sequence — the globe emoji takes four bytes and <Code>cl100k_base</Code> represents it as three tokens, each of which is a partial byte sequence. When you call <Code>enc.decode</Code> on the full list of IDs the bytes concatenate and the globe emoji rematerializes. When you call <Code>enc.decode</Code> on a single ID that happens to be a partial multi-byte character, you get the replacement character. This is a consequence of how byte-level BPE handles multi-byte characters: it doesn't guarantee that merges respect UTF-8 boundaries, and for characters rare in the training set it often doesn't.
      </Prose>

      <Prose>
        Side-by-side on the same input, across four tokenizers, the differences in vocabulary maturity show up clearly.
      </Prose>

      <CodeBlock language="python">
{`for name in ["cl100k_base", "o200k_base", "gpt2", "p50k_base"]:
    enc = tiktoken.get_encoding(name)
    n = len(enc.encode("Héllo 🌍 こんにちは"))
    print(f"{name:>14}: {n} tokens")

# Actual output (verified):
#    cl100k_base:  8 tokens   ← GPT-4 / GPT-3.5-turbo
#     o200k_base:  7 tokens   ← GPT-4o, larger vocab
#           gpt2: 13 tokens   ← original GPT-2
#      p50k_base: 13 tokens   ← GPT-3 (Codex generation)`}
      </CodeBlock>

      <Prose>
        The trajectory is monotonic. GPT-2 and p50k_base (same era, ~50k vocab) both emit thirteen tokens for this mixed-script input. <Code>cl100k_base</Code> (100k vocab, trained on a larger and more multilingual corpus) gets it down to eight, mostly because it has a single token for <Code>こんにちは</Code>. <Code>o200k_base</Code> (200k vocab) gets it to seven by merging <Code>llo</Code> into one token. The improvements come from vocabulary headroom plus better training corpora, not from algorithm changes — the algorithm is the same byte-level BPE from 2019. More vocabulary budget and more multilingual training data buy better compression on non-English text without changing a line of code.
      </Prose>

      <Prose>
        The cross-script cost picture is sharper in a table. Here is the ratio of UTF-8 bytes to <Code>cl100k_base</Code> tokens, measured on identical-meaning sentences in seven scripts.
      </Prose>

      <CodeBlock language="python">
{`# Actual measurements from tiktoken 0.12.0, verified on these sentences:
# english   "The quick brown fox jumps over the lazy dog."
# french    "Le renard brun rapide saute par-dessus le chien paresseux."
# greek     "Η γρήγορη καφέ αλεπού πηδάει πάνω από το τεμπέλικο σκυλί."
# hindi     "तेज़ भूरी लोमड़ी आलसी कुत्ते पर कूदती है।"
# chinese   "敏捷的棕色狐狸跳过了懒狗。"
# japanese  "素早い茶色のキツネは怠け者の犬を飛び越えます。"
# emoji     "🌍🌎🌏🦊🐶🐱🐭🐹🐰🦁"
#
#   lang     bytes  cl100k  o200k  bytes/cl   bytes/o2
#   english    44      10     10     4.40       4.40
#   french     58      18     14     3.22       4.14
#   greek     104      50     26     2.08       4.00
#   hindi     109      45     19     2.42       5.74
#   chinese    39      25     14     1.56       2.79
#   japanese   69      32     23     2.16       3.00
#   emoji      40      30     22     1.33       1.82`}
      </CodeBlock>

      <Prose>
        Two observations. First, even byte-level BPE fragments heavily on low-frequency scripts. <Code>cl100k_base</Code> uses fifty tokens for a Greek sentence that takes 104 bytes — a compression ratio of 2.08, barely better than character-level. Second, the gap between <Code>cl100k_base</Code> and <Code>o200k_base</Code> is small for English (four merges saved over ten) but enormous for Greek (halving the token count) and Hindi (2.4x reduction). Most of what makes <Code>o200k_base</Code> "better" is multilingual vocabulary growth. A byte-level tokenizer can be bad at a language without failing on it — the coverage guarantee holds even when the fertility is terrible.
      </Prose>

      <H3>5a. Llama 3 and SentencePiece byte-fallback</H3>

      <Prose>
        Llama 3's tokenizer is not byte-level BPE in exactly the GPT-2 sense. It uses SentencePiece with a BPE model and a feature called <em>byte fallback</em>. The base vocabulary contains human-chosen characters (all Unicode codepoints that appeared with sufficient frequency in the training corpus) rather than raw bytes, and there are 256 reserved tokens for "byte 0x00" through "byte 0xFF" that fire as a fallback only when the input contains a character not in the base vocabulary. The effect is similar — no <Code>[UNK]</Code>, every input encodes — but the mechanics differ. On pure ASCII or common Latin text the two approaches are nearly equivalent. On rare CJK characters, Llama 3's byte-fallback mode lets a single Unicode character be encoded as three bytes (three tokens), which is exactly what GPT-2's byte-level BPE does implicitly. The distinction matters for tokenizer-editing tools but not for model quality.
      </Prose>

      {/* ======================================================================
          6. TOKEN-FREE MODELS
          ====================================================================== */}
      <H2>6. Token-free models</H2>

      <Prose>
        Byte-level BPE is still BPE — the merge list is still load-bearing, the segmentation is still frozen at training time, and the model inherits whatever biases the tokenizer training corpus had. A distinct line of research asks what happens if the tokenizer is removed entirely and the segmentation decision is pushed into the model's weights. Four papers define the design space.
      </Prose>

      <H3>6a. CANINE — characters in, learned downsampling</H3>

      <Prose>
        CANINE (Clark et al., 2021, arXiv:2103.06874) accepts Unicode codepoints directly rather than subword tokens. The input layer hashes each codepoint into a small embedding using multiple hash functions, which gives it a constant-size vocabulary — no embedding table to tune, no vocab file to ship. The sequence of character embeddings is then fed through a strided convolution that downsamples the sequence by a factor of four, producing a shorter sequence of "soft" subword-like representations that the rest of the transformer stack consumes normally. An upsampling head at the output restores the character-level resolution for token-classification tasks.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class CANINEInput(nn.Module):
    """Hash-embed codepoints, then strided-conv downsample by 4x."""
    def __init__(self, d=768, n_hashes=8, buckets=16384, stride=4):
        super().__init__()
        self.hashes = nn.ModuleList([nn.Embedding(buckets, d // n_hashes)
                                     for _ in range(n_hashes)])
        self.buckets, self.n_hashes = buckets, n_hashes
        # Strided conv collapses every 4 character embeddings into 1.
        self.downsample = nn.Conv1d(d, d, kernel_size=stride, stride=stride)

    def forward(self, codepoints):
        # codepoints: (batch, seq_char) of int codepoint values
        parts = []
        for i, emb in enumerate(self.hashes):
            # Multi-hash the codepoint with a distinct salt per head.
            h = (codepoints * (i + 1) * 2654435761) % self.buckets
            parts.append(emb(h))
        x = torch.cat(parts, dim=-1)              # (batch, seq_char, d)
        x = x.transpose(1, 2)                     # (batch, d, seq_char)
        x = self.downsample(x)                    # (batch, d, seq_char/4)
        return x.transpose(1, 2)                  # (batch, seq_down, d)`}
      </CodeBlock>

      <Prose>
        CANINE's paper reports matching or beating mBERT on TyDi QA with 28% fewer parameters. Its claim to fame is multilingual robustness: no tokenizer means no "English-biased vocab" tax on low-resource scripts. The architectural cost is the fixed downsampling factor — four is hardcoded because that's roughly the compression ratio a subword tokenizer would have given you, but it is a design choice rather than a learned parameter, and models that need finer-grained input (say, noisy OCR) pay for it.
      </Prose>

      <H3>6b. ByT5 — bytes in, standard transformer</H3>

      <Prose>
        ByT5 (Xue et al., 2021, arXiv:2105.13626) is the simplest possible token-free design. Take T5, replace its SentencePiece tokenizer with raw UTF-8 bytes, add a handful of special tokens, and train. The vocabulary is literally the 256 byte values plus a small set of sentinels — 384 embeddings, total. Sequence length per document is roughly four to five times what it would be under mT5's subword tokenizer. ByT5 compensates by redistributing parameters: its encoder is three times as deep as the decoder (rather than balanced), because the encoder pays the quadratic attention cost over long byte sequences and it's the component whose capacity matters most for downstream performance.
      </Prose>

      <Prose>
        The ByT5 paper's most cited claim is noise robustness. If you take a subword-tokenized model and corrupt inputs — drop characters, swap adjacent letters, insert typos — a single character-level change can cascade into a completely different token sequence downstream of the corruption, because BPE segmentation is non-local. ByT5, which sees bytes all the way through, degrades gracefully: a corrupted byte affects one attention position and its immediate context, nothing more. On the XTREME multilingual benchmark, ByT5 outperforms mT5 at matched parameter counts for small models and remains competitive at larger scales. Its cost is the one everyone knew going in: wall-clock training time per token of downstream quality is significantly higher than for a subword-tokenized baseline.
      </Prose>

      <H3>6c. Charformer — learned segmentation via GBST</H3>

      <Prose>
        Charformer (Tay et al., 2021, arXiv:2106.12672) takes a third angle: keep character-level input, but make the <em>segmentation</em> itself a differentiable operation. The core mechanism is the Gradient-Based Subword Tokenization (GBST) block. Given a character sequence of length <Code>L</Code>, GBST enumerates candidate blockings at several stride lengths — 1, 2, 3, 4 — producing four parallel "downsampled" sequences where each element is a mean-pooled character span. A small learned scoring network assigns a soft weight to each blocking at each position, and the output is a position-wise weighted sum of the candidates. The whole thing is differentiable end-to-end, so the model learns, from loss gradients, which blocking is best for each position.
      </Prose>

      <Prose>
        In practice Charformer runs 28-100% faster than a byte-level baseline at matched quality, because the downsampling reduces the sequence length the main transformer sees. The tradeoff is architectural complexity — GBST is a non-trivial module, not a preprocessor — and reduced interpretability: you cannot look at a "token list" and reason about what the model saw, because the model saw a soft superposition of blockings.
      </Prose>

      <H3>6d. MambaByte — bytes without quadratic attention</H3>

      <Prose>
        MambaByte (Wang et al., 2024, arXiv:2401.13660) is the newest entry and possibly the most important, because it changes the core tradeoff. The cost argument against byte-level processing has always assumed a transformer. Mamba — a selective state-space model — scales linearly in sequence length with a fixed-size recurrent state, so tripling the sequence length triples the compute rather than squaring it. MambaByte trains a Mamba model directly on raw UTF-8 bytes and, at matched FLOPs, performs competitively with subword-tokenized transformers on language modeling benchmarks. The paper also introduces a speculative-decoding variant with a tokenized draft model and byte-level verification, yielding 2.6x inference speedup.
      </Prose>

      <Prose>
        The significance is less about MambaByte specifically and more about the fact that the "obvious" verdict — byte-level is too expensive, use subword — depends on attention being quadratic. As linear-time architectures mature, the calculation rebalances. Byte-level is not automatically the right answer even with Mamba, but it stops being automatically the wrong one.
      </Prose>

      {/* ======================================================================
          7. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>7. Visual walkthrough</H2>

      <Prose>
        The byte grid at the top of section 2 already made the per-character cost visible. Here is the same input tokenized by four real production encoders, with each encoder's output shown as its token stream. Token counts are from <Code>tiktoken 0.12.0</Code>, run on the same Python expression.
      </Prose>

      <TokenStream
        label="gpt-2 tokenizer (50k vocab) — 13 tokens"
        tokens={["H", "é", "llo", " �", "�", "�", " �", "�", "ん", "に", "�", "�", "は"]}
      />

      <TokenStream
        label="cl100k_base (100k vocab, gpt-4) — 8 tokens"
        tokens={["H", "él", "lo", " �", "�", "�", " ", "こんにちは"]}
      />

      <TokenStream
        label="o200k_base (200k vocab, gpt-4o) — 7 tokens"
        tokens={["H", "é", "llo", " �", "�", " ", "こんにちは"]}
      />

      <TokenStream
        label="raw utf-8 bytes (byt5 input) — 27 tokens"
        tokens={["0x48", "0xc3", "0xa9", "0x6c", "0x6c", "0x6f", "0x20", "0xf0", "0x9f", "0x8c", "0x8d", "0x20", "0xe3", "0x81", "0x93", "0xe3", "0x82", "0x93", "0xe3", "0x81", "0xab", "0xe3", "0x81", "0xa1", "0xe3", "0x81", "0xaf"]}
      />

      <Prose>
        The progression is the whole argument in one image. GPT-2 fragments the globe emoji into three pieces and the hiragana into six. <Code>cl100k_base</Code> has learned <Code>こんにちは</Code> as a single token but still splits the emoji. <Code>o200k_base</Code> squeezes out one more merge. ByT5, which has no merges at all, sees twenty-seven bytes — and the model has to learn from the loss signal that those twenty-seven bytes correspond to eight perceived characters.
      </Prose>

      <Plot
        label="bytes-per-token ratio across scripts (cl100k_base vs o200k_base)"
        series={[
          {
            name: "cl100k_base",
            points: [[1, 4.40], [2, 3.22], [3, 2.08], [4, 2.42], [5, 1.56], [6, 2.16], [7, 1.33]],
          },
          {
            name: "o200k_base",
            points: [[1, 4.40], [2, 4.14], [3, 4.00], [4, 5.74], [5, 2.79], [6, 3.00], [7, 1.82]],
          },
        ]}
        xLabel="script (1=en 2=fr 3=el 4=hi 5=zh 6=ja 7=emoji)"
        yLabel="bytes per token"
        width={520}
        height={240}
      />

      <Prose>
        Higher is better — it means more bytes of input per token of model cost. English, French, and emoji are roughly where you'd expect. The lines diverge sharply on Greek, Hindi, Chinese, and Japanese: <Code>o200k_base</Code>'s larger vocabulary absorbs multi-byte characters into single tokens much more often. This is the "multilingual tax" in plot form, and it is the main thing improving from <Code>cl100k_base</Code> to <Code>o200k_base</Code>.
      </Prose>

      {/* ======================================================================
          8. DECISION MATRIX
          ====================================================================== */}
      <H2>8. Decision matrix — when to use what</H2>

      <Prose>
        The decision is not between "byte-level" and "subword." It is between three concrete options: byte-level BPE (the de facto default), SentencePiece with byte fallback (the multilingual-leaning variant), and token-free (ByT5 or MambaByte for specific applications). The choice is made by answering four questions in order.
      </Prose>

      <Prose>
        <strong>Is your input primarily English or a few closely related scripts?</strong> If yes, byte-level BPE with a 100k-200k vocabulary is the answer. It gives you byte-level coverage for the long tail of emoji and rare content while compressing your main-line traffic to subword-efficient sequence lengths. Every frontier LLM in production in 2025-2026 fits this profile.
      </Prose>

      <Prose>
        <strong>Is your input aggressively multilingual?</strong> If the training corpus spans dozens of scripts and no single vocabulary treats them all fairly, two paths work: a byte-level BPE with a large vocabulary (250k+) trained on a multilingual-balanced corpus, or SentencePiece-with-byte-fallback trained the same way. The second is what Llama 3 and Gemma ship. For research work where fairness across scripts matters more than compression, ByT5 becomes competitive.
      </Prose>

      <Prose>
        <strong>Is your domain out-of-distribution for any subword tokenizer?</strong> Source code (heavy in tokenizer-weird characters — Unicode identifiers, unusual whitespace, long literals), DNA and protein sequences, chemistry SMILES strings, URLs, structured logs, binary-adjacent data — anywhere the "words" a text tokenizer would have chosen are wrong units to begin with. Pure byte-level, or a domain-specific byte-level BPE trained on the target domain, outperforms shoehorning through a text-trained subword vocabulary.
      </Prose>

      <Prose>
        <strong>Is byte-level robustness a hard requirement?</strong> Safety-critical applications where a silent tokenizer failure mode is unacceptable, audit pipelines that must exactly round-trip input bytes, or systems exposed to adversarial input benefit from the byte-level floor even when the English-only cost calculation argues against it. For these, SentencePiece with byte fallback is usually the right choice — guaranteed coverage, subword efficiency on common text, no special cases.
      </Prose>

      <Callout accent="gold">
        In 2026 the default for new frontier LLMs is byte-level BPE (or byte-fallback SentencePiece) with a 100k-300k vocabulary, trained on a multilingual-balanced corpus. Pure token-free (ByT5-style) is niche outside research and specific robust-to-noise applications. MambaByte is the first compelling case that this might change.
      </Callout>

      {/* ======================================================================
          9. SCALING ANALYSIS
          ====================================================================== */}
      <H2>9. What scales and what doesn't</H2>

      <Prose>
        Byte-level tokenization has a clean scaling story along some axes and a brutal one along others. Pull them apart.
      </Prose>

      <H3>9a. Sequence length</H3>

      <Prose>
        This is the dominant cost axis and the one that gets talked about most. Moving from subword to byte-level multiplies sequence length by 3-5x on English, 5-8x on CJK. Under a quadratic-attention transformer the compute increase is roughly the square of that. At a fixed context-window budget (say 8k model tokens), byte-level halves or thirds the effective characters of input you can fit. For training this shows up as a 3-5x increase in wall-clock time per epoch. For inference it shows up as increased latency per output character and reduced throughput per GPU.
      </Prose>

      <H3>9b. Vocabulary size</H3>

      <Prose>
        Byte-level tokenizers tend to have smaller vocabularies — the GPT-2 tokenizer was 50k, <Code>cl100k_base</Code> is 100k, <Code>o200k_base</Code> is 200k. Subword tokenizers for multilingual models sometimes go larger (Llama 3 is 128k, some multilingual specialists reach 256k+). A smaller vocabulary shrinks the embedding table and output softmax, which saves parameters and reduces the output projection's compute. For an embedding dim of 4096, dropping from a 256k vocab to 100k saves ~640M parameters. This is not nothing, but it's small relative to the transformer body, and the savings are dwarfed by the sequence-length cost.
      </Prose>

      <H3>9c. Training compute</H3>

      <Prose>
        Per-token training compute is roughly independent of whether those tokens are bytes or subwords — a forward pass over a sequence of length L with vocabulary V costs about the same whether V is 256 or 100k, because the dominant cost is the transformer body, not the embedding or output projection. But byte-level training sees 3-5x more tokens per document, so total compute per document scales linearly with that factor, and under quadratic attention actually scales worse. Empirically, ByT5 needs around 4x the training FLOPs of mT5 to reach comparable quality on downstream tasks. This is the reason byte-level pure-transformer models have not replaced subword-BPE transformers in production.
      </Prose>

      <H3>9d. Quality per FLOP</H3>

      <Prose>
        At small-to-medium scale, subword-BPE transformers have a quality-per-FLOP edge over byte-level counterparts for most downstream tasks. The edge narrows as you scale up the model — the fixed per-token cost of the tokenizer shrinks in relative terms, and the robustness benefits of byte-level become more valuable as the data gets noisier. The ByT5 scaling curves cross over the mT5 curves at roughly 3B parameters for noisy-data tasks. At even larger scales, the sequence-length cost keeps byte-level transformers behind subword transformers for clean English text. MambaByte's claim is that switching away from attention changes this curve — at matched FLOPs, byte-level Mamba matches or beats subword transformers, because the FLOPs go further when compute scales linearly rather than quadratically with length.
      </Prose>

      {/* ======================================================================
          10. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>10. Failure modes & gotchas</H2>

      <Prose>
        Byte-level tokenization inherits the coverage guarantee cleanly, but the engineering around it has ten years of accumulated sharp edges. Here are the ones that show up in practice.
      </Prose>

      <Prose>
        <strong>1. The bytes_to_unicode direction confusion.</strong> The most common bug: someone reads the GPT-2 tokenizer code, implements <Code>bytes_to_unicode</Code>, and then uses it in the wrong direction. The function maps <em>byte values</em> to <em>unicode chars</em>, not the other way around. When encoding you apply it forward: UTF-8-encode to bytes, then map each byte through. When decoding you apply its inverse: map each unicode char back to a byte, then UTF-8-decode. Getting the direction wrong produces garbage that still sort of looks right for ASCII (because ASCII bytes map to themselves) but corrupts anything non-ASCII. Debug this by running a round-trip test on a string containing both ASCII and non-ASCII characters before doing anything else.
      </Prose>

      <Prose>
        <strong>2. UTF-8 normalization mismatches.</strong> Unicode has multiple valid representations for "the same" string. The character <Code>é</Code> can be a single codepoint (U+00E9, two bytes in UTF-8) or a decomposition into <Code>e</Code> plus combining acute (U+0065 U+0301, three bytes). NFC (Normalization Form C) composes where possible, NFD decomposes. If your training pipeline normalizes to NFC and your inference pipeline normalizes to NFD (or doesn't normalize at all), the byte sequences the tokenizer sees differ, and the same logical string tokenizes differently at training and inference. UAX #15 defines the normalization forms; pick one, apply it consistently everywhere, document which one.
      </Prose>

      <Prose>
        <strong>3. Merges crossing UTF-8 character boundaries.</strong> Byte-level BPE merges are greedy on byte adjacency, not on character structure. Nothing in the algorithm prevents a merge from joining the last byte of one UTF-8 character to the first byte of the next. Most of the time this is fine — the model learns to use the chimeric token, and the decoder rebuilds the byte sequence correctly. Occasionally it fails to round-trip, because the merged byte sequence is not a valid UTF-8 character prefix. Production tokenizers add replacement-character handling at decode time. If you are implementing this from scratch, UTF-8-decode with <Code>errors="replace"</Code>, not <Code>errors="strict"</Code>.
      </Prose>

      <Prose>
        <strong>4. CJK fragmentation under low-vocab byte-BPE.</strong> Every CJK character takes three UTF-8 bytes. If your tokenizer vocabulary is small and your training corpus was mostly English, CJK characters that didn't appear often enough to earn their own multi-byte merge will fragment into three single-byte tokens each. The measurements earlier in this topic show <Code>cl100k_base</Code> using 25 tokens for a 13-character Chinese sentence — nearly two tokens per character. The fix is vocabulary headroom plus multilingual training data, not an algorithm change.
      </Prose>

      <Prose>
        <strong>5. Emoji ZWJ sequences.</strong> Modern emoji are often not a single codepoint. The family emoji <Code>👨‍👩‍👧‍👦</Code> is seven codepoints joined by Zero Width Joiner (U+200D). The professions (👩‍🔬, 🧑‍🚀, etc.) are similar. UTF-8-encoded, these are 25+ bytes. Byte-level BPE will learn the common ones as merged tokens if they appear enough in training, but the long tail of ZWJ sequences routinely fragments into 10-20 tokens each. For apps that care about emoji (social media, messaging), measure this explicitly on your expected emoji distribution.
      </Prose>

      <Prose>
        <strong>6. Special tokens leaking through byte-level encoding.</strong> Chat models use special tokens like <Code>&lt;|im_start|&gt;</Code>, <Code>&lt;|endoftext|&gt;</Code>, <Code>&lt;|begin_of_text|&gt;</Code>. Byte-level tokenizers handle these by matching them before byte encoding — if the pre-tokenizer doesn't split on special-token boundaries, a user input that happens to contain those literal byte sequences can be tokenized as the special token by accident. The <Code>tiktoken</Code> encode API has explicit <Code>allowed_special</Code> and <Code>disallowed_special</Code> parameters for exactly this reason. Never encode untrusted user input without setting <Code>disallowed_special=()</Code> or similar.
      </Prose>

      <Prose>
        <strong>7. Pre-tokenizer regex mismatch between training and inference.</strong> The GPT-2 regex is specific: it splits on contractions, then greedy letter/digit/punctuation runs. If you train a tokenizer with one regex and encode with a subtly different one, the byte-mapped sequences entering the BPE merge loop differ at word boundaries, and your encodings won't match what training produced. When shipping a custom byte-level tokenizer, serialize the exact regex alongside the merges.
      </Prose>

      <Prose>
        <strong>8. Digit tokenization surprises.</strong> Byte-level BPE treats digits like any other bytes — the number <Code>1234</Code> can tokenize as one token, two, three, or four depending on which digit sequences were frequent in training. Llama 3 and GPT-4o deliberately split digits into single-digit tokens to make arithmetic more learnable. If your tokenizer does not, you can end up with inputs where <Code>1234</Code> is one opaque token and <Code>1235</Code> is three tokens, which scrambles the place-value structure the model needs for arithmetic.
      </Prose>

      <Prose>
        <strong>9. Trailing-whitespace round-trip edge cases.</strong> The Ġ prefix convention handles leading whitespace elegantly. Trailing whitespace (a message that ends with a space or newline) is handled inconsistently across implementations. Some strip it, some preserve it, some collapse multiple trailing whitespace into one token. For applications where trailing whitespace is semantically meaningful — code generation, structured output — test explicitly that <Code>decode(encode(x)) == x</Code> on strings ending with whitespace.
      </Prose>

      <Prose>
        <strong>10. Multi-byte character splits in streaming decode.</strong> When a model generates tokens one at a time, each token is decoded independently and the result is concatenated to the output. If a single token does not correspond to complete UTF-8 characters — which is common for byte-level BPE on non-ASCII content — decoding that token in isolation produces a replacement character. The fix is to buffer bytes across tokens until a complete UTF-8 sequence is available. Every major inference library handles this; if you are writing your own decode loop, don't forget to buffer.
      </Prose>

      {/* ======================================================================
          11. PRIMARY SOURCES
          ====================================================================== */}
      <H2>11. Primary sources</H2>

      <Prose>
        Every claim about a specific paper or spec in this topic traces to one of the following. All six were verified via web search; arXiv IDs and venues are current as of April 2026.
      </Prose>

      <Prose>
        <strong>Radford, Wu, Child, Luan, Amodei, Sutskever (2019).</strong> "Language Models are Unsupervised Multitask Learners." OpenAI technical report, February 2019. Section 2.2 introduces byte-level BPE and the bytes-to-unicode remapping. PDF at <Code>cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf</Code>.
      </Prose>

      <Prose>
        <strong>Clark, Garrette, Turc, Wieting (2021).</strong> "CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation." arXiv:2103.06874, published in TACL 2022. Character-level input with learned downsampling convolutions.
      </Prose>

      <Prose>
        <strong>Xue, Barua, Constant, Al-Rfou, Narang, Kale, Roberts, Raffel (2021).</strong> "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models." arXiv:2105.13626, published in TACL 2022. T5 trained directly on UTF-8 bytes; the reference byte-level transformer.
      </Prose>

      <Prose>
        <strong>Tay, Tran, Ruder, Gupta, Chung, Bahri, Qin, Baumgartner, Yu, Metzler (2021).</strong> "Charformer: Fast Character Transformers via Gradient-based Subword Tokenization." arXiv:2106.12672. Introduces the GBST (Gradient-Based Subword Tokenization) block for learned, differentiable segmentation.
      </Prose>

      <Prose>
        <strong>Wang, Gangavarapu, Yan, Rush (2024).</strong> "MambaByte: Token-free Selective State Space Model." arXiv:2401.13660, CoLM 2024. Byte-level Mamba with speculative-decoding inference optimization.
      </Prose>

      <Prose>
        <strong>Unicode Consortium.</strong> "Unicode Standard Annex #15: Unicode Normalization Forms." Available at <Code>unicode.org/reports/tr15/</Code>. Defines NFC, NFD, NFKC, NFKD and their precise semantics. The normative reference for any text-processing pipeline that cares about round-tripping across Unicode representations.
      </Prose>

      {/* ======================================================================
          12. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>12. Self-check exercises</H2>

      <Prose>
        <strong>1. The bytes_to_unicode necessity.</strong> GPT-2's byte-level BPE maps each of the 256 byte values to a printable Unicode character before running merges. Construct a concrete input string for which writing the naïve byte-level BPE vocab file — one that stores raw bytes directly as strings in JSON — would break. Hint: what happens to byte 0x22 (double quote), byte 0x0A (newline), or byte 0x00 (null) when you put them in a JSON string literal?
      </Prose>

      <Prose>
        <strong>2. Round-trip detector.</strong> Write a function <Code>is_roundtrip_safe(tokens, tokenizer)</Code> that takes a list of token IDs and a tokenizer and returns True if <Code>tokenizer.decode(tokens) == ''.join(tokenizer.decode([t]) for t in tokens)</Code> (i.e., decoding the whole sequence gives the same result as decoding each token individually and concatenating). On a byte-level BPE tokenizer, find a token sequence that round-trips when decoded together but produces replacement characters when decoded individually. What structural property of the tokens makes this happen?
      </Prose>

      <Prose>
        <strong>3. Fertility estimation.</strong> Using only the fact that Greek characters take 2 bytes each in UTF-8 and English characters take 1 byte each, estimate the ratio of tokens-per-character for Greek text under a pure byte-level tokenizer (ByT5). Now suppose <Code>o200k_base</Code>'s fertility on Greek is 0.5 tokens per byte (as measured in section 5). What is its fertility in tokens per Greek character? If a model has an 8k-token context window, approximately how many characters of Greek text fit?
      </Prose>

      <Prose>
        <strong>4. Merge-boundary pathology.</strong> Construct a two-character input where naïve byte-level BPE could, in principle, produce a merged token whose bytes straddle the boundary between the two characters in UTF-8. Does this happen in practice with <Code>cl100k_base</Code>? Why or why not? (Hint: think about what pre-tokenization does.)
      </Prose>

      <Prose>
        <strong>5. When does MambaByte win?</strong> Consider three workloads: (a) English chatbot with 2k-token average conversation length; (b) multilingual document summarization with 50k-character documents; (c) code-completion with 8k-token contexts over a mixed-language codebase. For each, reason about whether byte-level BPE on a transformer, ByT5 on a transformer, or MambaByte on a state-space model is the most likely production choice. Which of your three answers is most sensitive to how much you believe the linear-scaling claims of Mamba at long context?
      </Prose>

      {/* ======================================================================
          END
          ====================================================================== */}
      <Prose>
        Byte-level tokenization solved universality cleanly and in 2019. The subword vocabulary question was never about whether we had the right primitive — we do, it's bytes — but about whether we had the right compression on top of them. The next topic, on dynamic tokenization, is about the layer above: once the floor is byte-level and the hybrid is mature, what does it take to let the tokenizer adapt to the actual distribution of inputs in production rather than a frozen snapshot from training?
      </Prose>
    </div>
  ),
};

export default byteLevelTokenization;
