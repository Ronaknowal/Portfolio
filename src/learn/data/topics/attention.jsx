import { colors, fonts } from "../../styles";
import AttentionVisualizer from "../../components/AttentionVisualizer";

const attention = {
  id: "attention",
  title: "Attention Mechanism",
  category: "nlp",
  readTime: "8 min",
  order: 10,
  content: () => (
    <div>
      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8, marginBottom: 20 }}>
        The attention mechanism allows a model to dynamically focus on relevant parts of the input sequence when producing each element of the output. Instead of compressing an entire sequence into a single fixed-size vector, attention computes a weighted sum over all input positions — letting the model "attend" to the information that matters most for each prediction step.
      </p>

      <AttentionVisualizer />

      <h3 style={{ fontFamily: fonts.sans, fontSize: 20, fontWeight: 600, color: colors.textPrimary, margin: "24px 0 12px" }}>
        Scaled Dot-Product Attention
      </h3>

      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8, marginBottom: 16 }}>
        The core computation takes three inputs — Queries (Q), Keys (K), and Values (V). The attention scores are computed as the dot product of queries with keys, divided by the square root of the key dimension for numerical stability, then passed through a softmax to obtain weights over the values.
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
        <span style={{ color: "#c084fc" }}>def</span> <span style={{ color: colors.green }}>scaled_dot_product_attention</span>(Q, K, V):<br/>
        {"    "}d_k = Q.shape[-1]<br/>
        {"    "}scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)<br/>
        {"    "}weights = torch.softmax(scores, dim=-1)<br/>
        {"    "}<span style={{ color: "#c084fc" }}>return</span> torch.matmul(weights, V)
      </div>

      <h3 style={{ fontFamily: fonts.sans, fontSize: 20, fontWeight: 600, color: colors.textPrimary, margin: "24px 0 12px" }}>
        Why It Matters
      </h3>

      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8 }}>
        Before attention, sequence-to-sequence models relied on encoding the entire input into a single context vector — a bottleneck that lost information for long sequences. Attention removes this bottleneck by allowing the decoder to look at all encoder states directly, weighted by relevance.
      </p>
    </div>
  ),
};

export default attention;
