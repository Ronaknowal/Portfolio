import { colors, fonts } from "../../styles";
import BackpropVisualizer from "../../components/BackpropVisualizer";

const backprop = {
  title: "Backpropagation & Automatic Differentiation",
  readTime: "12 min",
  content: () => (
    <div>
      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8, marginBottom: 20 }}>
        Backpropagation is the algorithm that makes training deep neural networks practical. It efficiently computes the gradient of the loss function with respect to every weight in the network by applying the chain rule of calculus layer by layer, from the output back to the input.
      </p>

      <BackpropVisualizer />

      <h3 style={{ fontFamily: fonts.sans, fontSize: 20, fontWeight: 600, color: colors.textPrimary, margin: "24px 0 12px" }}>
        The Chain Rule at Scale
      </h3>

      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8, marginBottom: 16 }}>
        Consider a simple network: input x passes through layers f, g, and h to produce output y = h(g(f(x))). The gradient of the loss L with respect to the weights of f requires multiplying the gradients through each intermediate layer — that is the chain rule applied recursively.
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
        <span style={{ color: colors.textDim }}># Forward pass — compute and cache activations</span><br/>
        z1 = W1 @ x + b1<br/>
        a1 = relu(z1)<br/>
        z2 = W2 @ a1 + b2<br/>
        loss = mse(z2, target)<br/>
        <br/>
        <span style={{ color: colors.textDim }}># Backward pass — chain rule from output to input</span><br/>
        dL_dz2 = 2 * (z2 - target) / n<br/>
        dL_dW2 = dL_dz2 @ a1.T<br/>
        dL_da1 = W2.T @ dL_dz2<br/>
        dL_dz1 = dL_da1 * (z1 &gt; 0)  <span style={{ color: colors.textDim }}># relu derivative</span><br/>
        dL_dW1 = dL_dz1 @ x.T
      </div>

      <h3 style={{ fontFamily: fonts.sans, fontSize: 20, fontWeight: 600, color: colors.textPrimary, margin: "24px 0 12px" }}>
        Computational Graph Perspective
      </h3>

      <p style={{ fontFamily: fonts.mono, fontSize: 13, color: colors.textSecondary, lineHeight: 1.8 }}>
        Modern frameworks like PyTorch build a dynamic computational graph during the forward pass. Each operation records its inputs and the local gradient function. During backward(), gradients flow through this graph in reverse topological order — this is automatic differentiation, and backpropagation is its specific application to neural network training.
      </p>
    </div>
  ),
};

export default backprop;
