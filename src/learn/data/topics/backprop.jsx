import BackpropVisualizer from "../../components/BackpropVisualizer";
import { Prose, H3, CodeBlock } from "../../components/content";
import { colors } from "../../styles";

const backprop = {
  title: "Backpropagation & Automatic Differentiation",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        Backpropagation is the algorithm that makes training deep neural networks practical. It efficiently computes the gradient of the loss function with respect to every weight in the network by applying the chain rule of calculus layer by layer, from the output back to the input.
      </Prose>

      <BackpropVisualizer />

      <H3>The Chain Rule at Scale</H3>

      <Prose>
        Consider a simple network: input x passes through layers f, g, and h to produce output y = h(g(f(x))). The gradient of the loss L with respect to the weights of f requires multiplying the gradients through each intermediate layer — that is the chain rule applied recursively.
      </Prose>

      <CodeBlock>
        <span style={{ color: colors.textDim }}># Forward pass — compute and cache activations</span>{"\n"}
        z1 = W1 @ x + b1{"\n"}
        a1 = relu(z1){"\n"}
        z2 = W2 @ a1 + b2{"\n"}
        loss = mse(z2, target){"\n"}
        {"\n"}
        <span style={{ color: colors.textDim }}># Backward pass — chain rule from output to input</span>{"\n"}
        dL_dz2 = 2 * (z2 - target) / n{"\n"}
        dL_dW2 = dL_dz2 @ a1.T{"\n"}
        dL_da1 = W2.T @ dL_dz2{"\n"}
        dL_dz1 = dL_da1 * (z1 &gt; 0)  <span style={{ color: colors.textDim }}># relu derivative</span>{"\n"}
        dL_dW1 = dL_dz1 @ x.T
      </CodeBlock>

      <H3>Computational Graph Perspective</H3>

      <Prose>
        Modern frameworks like PyTorch build a dynamic computational graph during the forward pass. Each operation records its inputs and the local gradient function. During backward(), gradients flow through this graph in reverse topological order — this is automatic differentiation, and backpropagation is its specific application to neural network training.
      </Prose>
    </div>
  ),
};

export default backprop;
