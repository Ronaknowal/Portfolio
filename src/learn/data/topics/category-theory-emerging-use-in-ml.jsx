import { Callout, CodeBlock, H2, Prose } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";

const content = {
  title: "Category Theory (Emerging Use in ML)",
  readTime: "~38 min",
  content: () => <div>
    <H2>1. Why an abstract language can help with practical systems</H2>
    <Prose>
      Category theory studies structure-preserving relationships and composition. It deliberately pays less attention to what an object is made of and more attention to how it connects to other objects. That perspective is useful when systems are built from many transformations: data schemas feed feature maps, feature maps feed models, models feed decisions, and each stage must compose without silently changing meaning.
    </Prose>
    <Prose>
      It is an emerging lens for ML, not a replacement for linear algebra, probability, or optimisation. Its immediate value is often conceptual and architectural; its deeper value appears in compositional learning, differentiable programming, probabilistic semantics, type systems, and structured data integration.
    </Prose>

    <H2>2. A category is objects plus composable arrows</H2>
    <Prose>
      A category has objects, morphisms (arrows) between objects, an identity arrow for every object, and a way to compose compatible arrows. Composition is associative: when three transformations can be chained, grouping the chain differently does not change the result. Identity means a no-op transformation changes nothing.
    </Prose>
    <MathBlock>{`f:A\\to B,\\;g:B\\to C \\Rightarrow g\\circ f:A\\to C, \\qquad h\\circ(g\\circ f)=(h\\circ g)\\circ f, \\qquad 1_B\\circ f=f`}</MathBlock>
    <Prose>
      In a simplified programming example, types can be objects and total functions can be morphisms. A preprocessing function from raw records to features and a classifier from features to scores compose into a function from records to scores. Real software has failures, state, randomness, and side effects, so an exact categorical model may need richer constructions; the basic composition law still catches a valuable design intuition.
    </Prose>

    <H2>3. The laws are not decoration: they make refactoring safe</H2>
    <Prose>
      Associativity lets a pipeline be regrouped for caching, parallel planning, testing, or optimisation without changing its meaning. Identity lets optional stages be represented uniformly. The tiny code example checks those laws for ordinary functions; the fact that both groupings return the same result is the computational shadow of category-theoretic composition.
    </Prose>
    <CodeBlock language="python">{`def compose(g, f):
    return lambda x: g(f(x))

def identity(x):
    return x

f = lambda x: x + 1
g = lambda x: 2 * x
h = lambda x: x - 3

left = compose(h, compose(g, f))(4)
right = compose(compose(h, g), f)(4)
print(left, right)
print(compose(identity, f)(4), compose(f, identity)(4))`}</CodeBlock>
    <CodeBlock language="output">{`7 7
5 5`}</CodeBlock>
    <Prose>
      The code is not "doing category theory" in a special library. It is demonstrating the universal property we want from a compositional interface: components can be connected and reorganised predictably.
    </Prose>

    <H2>4. Functors map one kind of structure to another</H2>
    <Prose>
      A functor maps objects and morphisms from one category to another while preserving identities and composition. For a data pipeline, a functor-like mapping might turn each schema into a feature space and each valid schema transformation into the corresponding feature transformation. The important question is whether the mapping respects how transformations compose, rather than only producing plausible outputs one step at a time.
    </Prose>
    <MathBlock>{`F:\\mathcal{C}\\to\\mathcal{D}, \\qquad F(g\\circ f)=F(g)\\circ F(f), \\qquad F(1_A)=1_{F(A)}`}</MathBlock>
    <Prose>
      This helps expose mismatch. If combining two source transformations and then embedding them differs from embedding each and then combining them, the representation may be losing compositional structure. That can be a productive hypothesis in representation learning, but it must be tested empirically rather than assumed from notation.
    </Prose>

    <H2>5. Natural transformations compare whole mappings</H2>
    <Prose>
      A natural transformation is a coherent way to translate between two functors. Instead of offering an arbitrary conversion for each object, it requires compatibility with every morphism. Informally, whether you transform first and then translate or translate first and then transform should agree. This is a rigorous vocabulary for interfaces, model transformations, and semantic-preserving compiler passes.
    </Prose>
    <MathBlock>{`\\eta:F\\Rightarrow G, \\qquad G(f)\\circ\\eta_A=\\eta_B\\circ F(f)`}</MathBlock>
    <Prose>
      The commuting-square condition is stronger than "these two systems happen to agree on one example." It says the conversion behaves consistently across the relationships in the entire source structure.
    </Prose>

    <H2>6. Where the ideas meet ML today</H2>
    <Prose>
      Compositional model design treats modules as maps that can be connected while preserving interfaces. Categorical automatic differentiation and differentiable programming study how derivatives compose through programs. Markov categories and string diagrams give graphical languages for probabilistic processes. Functorial data migration supports schema integration. Monoidal categories describe systems that run in parallel as well as in sequence, which is useful for thinking about tensor operations and process networks.
    </Prose>
    <Callout accent="green" label="Keep the claim calibrated">
      These ideas can clarify modelling and software structure, and they motivate active research. They do not automatically make a model more accurate, train faster, or solve a data-quality problem. Use category theory when the bottleneck is compositional meaning or interface coherence—not as decorative abstraction around an ordinary prediction task.
    </Callout>

    <H2>7. How to learn and apply it without getting lost</H2>
    <Prose>
      Start with functions, composition, identity, products, and simple diagrams. Translate every definition into a familiar pipeline, typed program, or probabilistic process. Then learn functors and natural transformations through examples before moving to monoidal, enriched, or higher categories. In a project, write down objects, arrows, invariants, and composition rules; if doing so reveals an ambiguity or an invalid connection, the abstraction has already earned its keep.
    </Prose>
    <Callout label="Practice">
      Describe an ML inference pipeline as a category-inspired diagram: raw request to validated request to features to score to decision. Which arrows are deterministic, which carry uncertainty or failure, and what identity/compatibility laws would make it safe to insert an optional normalisation stage?
    </Callout>
  </div>,
};

export default content;
