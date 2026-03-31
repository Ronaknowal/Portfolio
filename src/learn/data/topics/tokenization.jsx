import { colors, fonts } from "../../styles";

const tokenization = {
  id: "tokenization",
  title: "Tokenization",
  category: "nlp",
  readTime: "10 min",
  order: 8,
  content: () => (
    <div>
      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8, marginBottom: 20 }}>
        Tokenization is the process of converting raw text into a sequence of tokens — the atomic units that a language model actually processes. The choice of tokenizer directly affects the model's vocabulary size, sequence length, and ability to handle rare or unseen words.
      </p>

      <h3 style={{ fontFamily: fonts.sans, fontSize: 20, fontWeight: 600, color: colors.textPrimary, margin: "24px 0 12px" }}>
        From Characters to Subwords
      </h3>

      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8, marginBottom: 16 }}>
        Character-level tokenization gives a tiny vocabulary but very long sequences. Word-level gives manageable sequences but can't handle unseen words. Subword tokenization (BPE, WordPiece, SentencePiece) hits the sweet spot — common words stay whole, rare words get split into meaningful pieces.
      </p>

      <div style={{
        background: "rgba(0,0,0,0.4)",
        border: `1px solid ${colors.border}`,
        borderRadius: 4,
        padding: 16,
        fontFamily: fonts.mono,
        fontSize: 12,
        color: colors.textSecondary,
        lineHeight: 1.7,
        marginBottom: 16,
        overflowX: "auto",
      }}>
        <span style={{ color: colors.textDim }}># BPE tokenization example</span><br/>
        text = <span style={{ color: colors.green }}>"unbelievably"</span><br/>
        <br/>
        <span style={{ color: colors.textDim }}># Word-level: ["unbelievably"] or OOV!</span><br/>
        <span style={{ color: colors.textDim }}># Character-level: ["u","n","b","e","l","i","e","v","a","b","l","y"]</span><br/>
        <span style={{ color: colors.textDim }}># BPE subword: ["un", "believ", "ably"]</span>
      </div>

      <h3 style={{ fontFamily: fonts.sans, fontSize: 20, fontWeight: 600, color: colors.textPrimary, margin: "24px 0 12px" }}>
        Byte Pair Encoding (BPE)
      </h3>

      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8 }}>
        BPE starts with a character-level vocabulary and iteratively merges the most frequent pair of adjacent tokens. After thousands of merges, you get a vocabulary that efficiently represents common words as single tokens while still being able to handle any input by falling back to smaller units. GPT models use a variant of BPE; BERT uses WordPiece (similar idea, different merge criterion).
      </p>
    </div>
  ),
};

export default tokenization;
