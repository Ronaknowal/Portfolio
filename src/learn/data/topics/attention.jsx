import AttentionVisualizer from "../../components/AttentionVisualizer";
import { Prose, H3, CodeBlock } from "../../components/content";
import { colors } from "../../styles";

const attention = {
  title: "Attention Mechanism (Bahdanau, Luong)",
  readTime: "8 min",
  content: () => (
    <div>
      <Prose>
        The attention mechanism allows a model to dynamically focus on relevant parts of the input sequence when producing each element of the output. Instead of compressing an entire sequence into a single fixed-size vector, attention computes a weighted sum over all input positions — letting the model "attend" to the information that matters most for each prediction step.
      </Prose>

      <AttentionVisualizer />

      <H3>Scaled Dot-Product Attention</H3>

      <Prose>
        The core computation takes three inputs — Queries (Q), Keys (K), and Values (V). The attention scores are computed as the dot product of queries with keys, divided by the square root of the key dimension for numerical stability, then passed through a softmax to obtain weights over the values.
      </Prose>

      <CodeBlock>
        <span style={{ color: "#c084fc" }}>def</span> <span style={{ color: colors.green }}>scaled_dot_product_attention</span>(Q, K, V):{"\n"}
        {"    "}d_k = Q.shape[-1]{"\n"}
        {"    "}scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k){"\n"}
        {"    "}weights = torch.softmax(scores, dim=-1){"\n"}
        {"    "}<span style={{ color: "#c084fc" }}>return</span> torch.matmul(weights, V)
      </CodeBlock>

      <H3>Why It Matters</H3>

      <Prose>
        Before attention, sequence-to-sequence models relied on encoding the entire input into a single context vector — a bottleneck that lost information for long sequences. Attention removes this bottleneck by allowing the decoder to look at all encoder states directly, weighted by relevance.
      </Prose>
    </div>
  ),
};

export default attention;
