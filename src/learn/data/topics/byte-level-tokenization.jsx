import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { TokenStream, ByteGrid } from "../../components/viz";
import { colors } from "../../styles";

const byteLevelTokenization = {
  title: "Byte-Level Tokenization & Token-Free Models",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        Subword tokenizers trained on English do something quietly unfair: they charge Devanagari, Arabic, CJK, emoji, and code three to five times as many tokens to say the same thing. The vocabulary was drawn from whatever text was lying around, and the text was mostly English. That is the pressure this topic releases. Instead of arguing about which subword vocabulary is the fairest one to ship, we can ask a sharper question — what if we stop designing a subword vocabulary at all and drop down to the one level that is genuinely universal?
      </Prose>

      <Prose>
        Every string any computer has ever stored already <em>is</em> a sequence of bytes. The vocabulary exists. It has 256 entries. No one has to train it.
      </Prose>

      <H2>UTF-8 as the universal vocabulary</H2>

      <Prose>
        UTF-8 is a variable-length encoding of Unicode designed by Ken Thompson and Rob Pike in 1992. Every Unicode codepoint is expressed as one to four bytes. ASCII — the first 128 codepoints — is a single byte, and the leading bit is zero, so ASCII text is byte-identical to its UTF-8 representation. Latin-extended characters like <Code>é</Code>, <Code>ñ</Code>, <Code>ü</Code> take two bytes. Cyrillic, Greek, Hebrew, Arabic, and most Indic scripts sit in the two-to-three byte range. CJK ideographs take three. Most emoji and the less common planes take four.
      </Prose>

      <Prose>
        The properties that matter for modeling are almost boring in how solid they are. UTF-8 is self-synchronizing — given any byte, you can tell whether it is an ASCII byte, a leading byte, or a continuation byte by looking at its top bits. It is prefix-free and lossless, so encoding and decoding compose cleanly. It is backward-compatible with every piece of ASCII software ever written, which is a big part of why it won the encoding wars of the late nineties. And it is standardized in a way that predates almost every model you have heard of. Any string, in any language, with any emoji, any punctuation, any code block, any filename, resolves without ambiguity to a sequence of bytes drawn from a fixed alphabet of 256.
      </Prose>

      <ByteGrid text="Héllo 🌍 こんにちは" label="utf-8 bytes" />

      <Prose>
        Hover over any cell to see which character a byte belongs to and how many bytes that character took. The green cells are single-byte ASCII. The gold cells are the continuation bytes of multi-byte sequences — a visual reminder that "one character" is not "one unit" as far as the underlying encoding is concerned. If you are designing a model that reads this directly, each of those cells is one attention position.
      </Prose>

      <Prose>
        Eighteen bytes for eight perceived characters. The word <Code>Hello</Code> is green the whole way — ASCII, one byte per letter — until the <Code>é</Code>, which takes two bytes (<Code>c3 a9</Code>). The globe emoji takes four. Each of the three hiragana characters takes three. The model, if it reads bytes, does not see "letters" or "characters" at all. It sees a stream of numbers between 0 and 255, and it has to learn from scratch that <Code>c3 a9</Code> is a single grapheme and that certain four-byte sequences are whole pictograms. That is a real cost, and we will come back to it — but first notice what has already been bought: universality. There is no "[UNK]" token. There is no failure mode where the input arrives and the tokenizer shrugs.
      </Prose>

      <H2>Byte-level BPE — the pragmatic hybrid</H2>

      <Prose>
        Pure byte input sounds elegant but is expensive. In 2019, the GPT-2 paper introduced a compromise that has since quietly taken over production: run BPE, but start from bytes instead of from Unicode characters. The base vocabulary is exactly the 256 byte values, and merges proceed as normal — pair frequencies, greedy joining, ordered merge list. Common English words still end up as single tokens because their byte sequences merge early and often. What changes is the floor. There is no longer any such thing as an out-of-vocabulary input: every byte value is already in the vocabulary, so the worst the tokenizer can do is fall back to single bytes for text it has never seen. The character-level BPE that preceded it had a real failure mode — rare Unicode codepoints that never appeared in training would hit an <Code>[UNK]</Code> token, which was both a silent information loss and an information leak about the training distribution. Byte-level BPE removes the failure mode entirely.
      </Prose>

      <Prose>
        This is what you are looking at when a paper says "BPE" in 2024. OpenAI's <Code>tiktoken</Code> library is byte-level BPE. Llama's tokenizer is SentencePiece configured for BPE with byte fallback, which is effectively the same deal with a different code path. Mistral, Qwen, DeepSeek, and Gemma all ship some flavor of this hybrid. The model still gets subword-sized tokens for common text — which is what makes transformer attention tractable — and inherits byte-level coverage for the long tail. Emoji, rare scripts, pathological Unicode, random binary: all of it tokenizes to something, always.
      </Prose>

      <Prose>
        There is one piece of engineering worth naming because it trips people up. GPT-2's original byte-level BPE does not store raw bytes in the tokenizer's text representation — it remaps the 256 byte values onto a set of printable Unicode characters so that the merge strings in the vocabulary file stay human-readable and JSON-safe. The remap is a fixed, reversible shuffle. When you see a GPT-2 token like <Code>ĠHello</Code>, the <Code>Ġ</Code> is not a strange letter; it is byte value 0x20 (the space character) remapped into the printable range. The algorithm is still byte-level BPE. The surface representation is a presentation choice.
      </Prose>

      <CodeBlock language="python">
{`import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer
ids = enc.encode("Héllo 🌍 こんにちは")
tokens = [enc.decode([i]) for i in ids]

# Common English words resolve to single subword tokens.
# é, 🌍, and Japanese hiragana fall back to multi-byte byte-BPE tokens —
# the tokenizer never saw them in training, but every byte value is in the
# base vocabulary, so nothing goes OOV.`}
      </CodeBlock>

      <H2>Token-free models — the other direction</H2>

      <Prose>
        Byte-level BPE is still BPE. The tokenizer is still trained, the merge list is still load-bearing, and the segmentation is still frozen at training time. A more radical line of research asks what happens if you remove the tokenizer entirely — feed raw bytes or characters straight into the model and let it learn whatever segmentation is useful from the loss signal. The motivation is not just aesthetic. A fixed tokenizer encodes a prior about which chunks of text are worth grouping, and that prior is baked in before the model sees a single gradient. If the prior is wrong for your domain, the model has to work against it for the rest of training. Token-free approaches push that decision into the model's weights, where it can be learned, adjusted, and overridden.
      </Prose>

      <Prose>
        Three Google papers from 2021 staked out the design space. CANINE takes character-level inputs and learns a downsampling stack that collapses them into coarser units inside the model, so that downstream layers see something closer to subword-sized units without a tokenizer having chosen them ahead of time. ByT5 trains T5 directly on UTF-8 bytes with no tokenizer at all — the vocabulary is literally the 256 byte values plus a handful of specials. Charformer introduces a soft, gradient-based block scorer that evaluates multiple candidate segmentations in parallel and lets the model weight them end-to-end, so the segmentation itself becomes a differentiable parameter.
      </Prose>

      <TokenStream
        label="subword bpe"
        tokens={["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]}
      />

      <TokenStream
        label="raw bytes (utf-8)"
        tokens={["T", "h", "e", " ", "q", "u", "i", "c", "k", " ", "b", "r", "o", "w", "n", " ", "f", "o", "x", " ", "j", "u", "m", "p", "s", " ", "o", "v", "e", "r", " ", "t", "h", "e", " ", "l", "a", "z", "y", " ", "d", "o", "g", "."]}
      />

      <Prose>
        Ten tokens against forty-four. These models matter less because they won — they largely did not — and more because they proved the point. A transformer can learn useful segmentation internally if you let it. ByT5 in particular was notable for being robust to noise in a way tokenized models are not: randomly drop characters, swap letters, insert typos, and the byte-level model degrades more gracefully than its subword counterpart, because a single corrupted byte does not cascade into a completely different token sequence the way a single corrupted character can under BPE. The reason these approaches have stayed in research-land is not that they fail on quality. It is that they are expensive.
      </Prose>

      <H2>The cost</H2>

      <Prose>
        Bytes are small. There are more of them. A paragraph that subword-BPE compresses to 200 tokens comfortably takes 600 to 800 bytes in UTF-8 — roughly three to four times the sequence length. For non-English text the ratio gets worse: a Japanese sentence that subword-BPE might encode in 50 tokens can weigh 150 bytes or more, because each CJK codepoint costs three. For a transformer, that is not a linear cost. Self-attention is quadratic in sequence length, so a 3x increase in tokens produces roughly a 9x increase in attention FLOPs per layer per forward pass. The KV cache grows by the same factor. The feedforward layers grow linearly. Training time, inference time, memory, and serving throughput all scale the wrong way — and they scale together, so the bill compounds.
      </Prose>

      <Callout accent="gold">
        A 4k-token subword context becomes a 12–16k-byte context. The price of universality is paid in compute, memory bandwidth, and KV cache — and attention charges you quadratically for it.
      </Callout>

      <H3>MambaByte and the state-space revival</H3>

      <Prose>
        The cost argument assumes a transformer. It is partly an architecture question. State-space models — Mamba, S4, and their descendants — scale linearly in sequence length rather than quadratically, with a fixed-size recurrent state that does not blow up when the input does. Long sequences stop being pathological; they become routine. MambaByte, published in early 2024, trains a Mamba model directly on raw UTF-8 bytes and performs competitively with subword-tokenized transformers at matched compute, despite processing three to four times as many tokens. It does so because the architecture pays a fundamentally different bill. The tokenizer-vs-architecture tradeoff is not fixed; as linear-time architectures mature, the calculation that makes subword BPE "obviously correct" gets less obvious. It is one of the few places in current NLP where a choice that looked settled five years ago is quietly back up for debate.
      </Prose>

      <H2>When byte-level is actually the right call</H2>

      <Prose>
        For an English-heavy production LLM in 2024 and 2025, the answer is still the hybrid — byte-level BPE with a 50k to 150k vocabulary. Pure token-free is niche. But there are three places where the calculation shifts and byte-level is genuinely worth the price. The first is aggressively multilingual models where the training distribution spans dozens of scripts and no single subword vocabulary treats them fairly — byte-level removes the tax at the cost of longer sequences, and for languages the tokenizer would otherwise split into three or four tokens per character, the math can actually favor raw bytes. The second is out-of-distribution domains: source code, DNA and protein sequences, URLs, chemistry SMILES, structured logs — anywhere the "words" the tokenizer would have picked up are the wrong units to begin with, and where preserving byte-exact fidelity matters more than shaving tokens. The third is safety-critical or audit-bound pipelines where a silent OOV failure mode is unacceptable and you would rather pay for coverage than debug a tokenizer-induced mistranslation in production.
      </Prose>

      <Prose>
        For everything else, the hybrid wins. Byte-level BPE gives you byte-level robustness on the tail and subword efficiency on the body. That is the actual default, and it is what almost every frontier model ships with today. The name on the box says "BPE" and the behavior underneath says "bytes first, subwords opportunistically" — a resolution that took the field about a decade to find and that almost no one argues with in production.
      </Prose>

      <Prose>
        If byte-level coverage solves the universality problem so cleanly, the subword vocabulary question should be solved too — but it is not. A byte-fallback BPE tokenizer still has to decide which merges to keep, which languages to weight, where on the compression curve to sit, and how to handle boundary cases like digits, whitespace runs, and code identifiers. Those choices determine whether Hindi costs 2x or 5x more than English, whether the number <Code>1234</Code> tokenizes as four digits or as opaque chunks that scramble place value, and whether your fine-tuning domain tokenizes efficiently at all. Byte fallback is a safety net, not a design. The vocabulary itself is the design, and the next topic is about what actually goes into it.
      </Prose>
    </div>
  ),
};

export default byteLevelTokenization;
