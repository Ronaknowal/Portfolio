import { Callout, Code, CodeBlock, H2, Prose } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";

const content = {
  title: "Topology & Topological Data Analysis (TDA)",
  readTime: "~42 min",
  content: () => <div>
    <H2>1. The question: what shape survives changes of scale?</H2>
    <Prose>
      Geometry measures lengths, angles, and coordinates. Topology asks coarser questions that survive continuous deformation: how many connected pieces are there, are there loops, and are there enclosed voids? Topological data analysis (TDA) uses those questions to study the shape of point clouds, images, graphs, dynamical trajectories, and learned representations—especially when no single cluster count or coordinate view tells the whole story.
    </Prose>
    <Prose>
      TDA is not a magic feature extractor. Its output depends on a distance, a scale construction, sampling density, and noise. Its strength is making multi-scale shape assumptions explicit rather than committing to one clustering radius upfront.
    </Prose>

    <H2>2. From points to a shape: simplicial complexes</H2>
    <Prose>
      A dataset is usually a finite set of points, not a filled surface. TDA builds a combinatorial approximation called a simplicial complex. Vertices represent points, edges join nearby points, filled triangles represent three-way connections, and higher-dimensional simplices extend the idea. The Vietoris-Rips complex at scale epsilon adds a simplex whenever all of its vertices are pairwise within epsilon.
    </Prose>
    <MathBlock>{`\\operatorname{VR}_{\\varepsilon}(X)=\\left\\{\\sigma\\subseteq X: d(x_i,x_j)\\leq\\varepsilon\\;\\text{for all }x_i,x_j\\in\\sigma\\right\\}`}</MathBlock>
    <Prose>
      As epsilon grows, isolated points connect, loops can appear, and eventually filled triangles can close loops. This nested sequence is a filtration. The metric and feature scaling determine what counts as nearby, so standardising features or using a domain-specific distance is a modelling decision, not preprocessing trivia.
    </Prose>

    <H2>3. Homology counts holes by dimension</H2>
    <Prose>
      Homology turns the shape of a complex into algebraic summaries. The Betti number beta_0 counts connected components, beta_1 counts independent loops, and beta_2 counts enclosed voids. A circle has beta_0 = 1 and beta_1 = 1; a filled disk has beta_0 = 1 and beta_1 = 0 because its loop is filled in.
    </Prose>
    <MathBlock>{`\\beta_k=\\operatorname{rank}(H_k), \\qquad (\\beta_0,\\beta_1,\\beta_2)=\\text{components, loops, voids}`}</MathBlock>
    <Prose>
      Homology deliberately ignores many details. Two loops can differ drastically in size, position, and density while contributing the same beta_1. That abstraction is useful when connectivity matters more than exact geometry, but it also means TDA must be paired with the domain question it is meant to answer.
    </Prose>

    <H2>4. Persistent homology tracks features across scales</H2>
    <Prose>
      Instead of selecting one epsilon, persistent homology records when a feature is born and when it dies as the filtration grows. A connected component is born at a point; it dies when it merges with an older component. A loop is born when edges close a cycle and dies when higher-dimensional simplices fill it. The interval between birth and death is its persistence.
    </Prose>
    <MathBlock>{`\\text{persistence}=\\text{death}-\\text{birth}`}</MathBlock>
    <Prose>
      A barcode draws one interval per feature; a persistence diagram plots each feature as a point with coordinates (birth, death). Features far from the diagonal persist for a wide scale range and are often more robust than short-lived features. “Often” matters: a long-lived artifact can still come from sampling gaps, feature scale, or a flawed distance metric.
    </Prose>

    <H2>5. A small connectivity filtration you can inspect</H2>
    <Prose>
      The code below tracks beta_0 for four one-dimensional points. At epsilon 0.5, all points are separate. At 0.7, nearby pairs merge. At 1.5, the two pairs connect into one component. This is the simplest persistent-homology story: components die as the scale crosses the gaps between them.
    </Prose>
    <CodeBlock language="python">{`points = [0.0, 0.6, 2.0, 2.6]

def component_count(epsilon):
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if abs(points[i] - points[j]) <= epsilon:
                union(i, j)
    return len({find(i) for i in range(len(points))})

for epsilon in (0.5, 0.7, 1.5):
    print(epsilon, component_count(epsilon))`}</CodeBlock>
    <CodeBlock language="output">{`0.5 4
0.7 2
1.5 1`}</CodeBlock>
    <Prose>
      Real persistent-homology libraries compute the full filtration efficiently and record loops and higher features. This small union-find example is intentionally limited to components so the scale idea remains visible.
    </Prose>

    <H2>6. Turning persistence into inputs for ML</H2>
    <Prose>
      Persistence diagrams are variable-size sets, so many standard models need a fixed representation. Common choices include persistence images, landscapes, kernels, summary statistics, and learned set encoders. Distances such as bottleneck and Wasserstein distance compare diagrams while respecting birth/death structure. Choose a representation using validation and interpretability needs; flattening a diagram without a plan can discard the very structure TDA found.
    </Prose>
    <Callout accent="green" label="Potential applications">
      TDA can describe loops in sensor or biological trajectories, shape in 3D point clouds, coverage holes in networks, structural changes in time series, and topology of learned embeddings. It is most compelling when a persistent-shape hypothesis is plausible and a simpler baseline cannot express it cleanly.
    </Callout>

    <H2>7. A responsible TDA workflow</H2>
    <Prose>
      Start with the question: components, cycles, or voids of what object would change a decision? Define the data representation and metric, inspect scale distributions, and build a filtration suited to those choices. Use synthetic controls or permutations to see which features arise under a matched null. Test feature stability under resampling, noise, and reasonable preprocessing changes. Compare against non-topological baselines on held-out outcomes. A persistence diagram can reveal structure; it cannot tell you whether that structure is causal, useful, or safe to act on.
    </Prose>
    <Callout label="Practice">
      Sample noisy points from a circle and from a filled disk. Construct a Vietoris-Rips filtration for several scales. Which beta_1 feature should persist longer for the circle? How could uneven sampling or the wrong feature scale create a misleading loop?
    </Callout>
  </div>,
};

export default content;
