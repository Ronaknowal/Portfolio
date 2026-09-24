import { H2, H3, Prose, Callout } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Checkpoint, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { CompositionLab, SchemaFunctorLab, NaturalityLab, UniversalProductLab, StochasticCopyLab, TangentCompositionLab, ImagePreimageLab, HomologyFunctorFigure, ParallelCompositionFigure, OptionCompositionFigure } from '../../components/lesson-labs/CategoryTheoryLabs.jsx';
import { categoryTheoryExamples as examples } from '../category-theory-examples.js';
function Example({
  name
}) {
  const example = examples[name];
  return <section><Prose><strong>Before running.</strong> {example.question}</Prose><RunnableExample example={example} /><Prose>{example.interpretation}</Prose></section>;
}
function Practice({
  title,
  prompt,
  hint,
  children
}) {
  return <section className="lesson-check"><H3>{title}</H3><Prose>{prompt}</Prose><details><summary>Hint</summary><Prose>{hint}</Prose></details><details><summary>Reasoned solution</summary>{children}</details></section>;
}
export default {
  title: 'Category Theory (Emerging Use in ML)',
  readTime: '~85 min read + 3–4 hours practice',
  content: () => <div className="category-theory-lesson">
    <LessonIntro prerequisites="The core starts with finite sets and functions and introduces its notation locally. Python helps you run the complete examples. The later derivative, probability and homology branches use Matrix Calculus, Bayes' Theorem and the preceding Topology/TDA lesson; review those when needed." sections={[['1-start-with-things-you-can-connect', 'Typed composition'], ['2-name-the-system-and-its-laws', 'Categories and inverses'], ['3-translate-a-whole-system', 'Actual functors'], ['4-compare-two-compatible-translations', 'Naturality'], ['5-specify-an-object-by-what-it-must-do', 'Universal constructions'], ['6-compose-parallel-work-and-possible-failure', 'Parallel and effectful arrows'], ['7-compose-randomness-without-losing-dependence', 'Stochastic maps'], ['8-keep-the-point-when-you-differentiate', 'Differentiation'], ['9-answer-a-forward-question-with-a-backward-filter', 'Adjunctions'], ['10-reconstruct-a-system-from-compatible-probes', 'Yoneda'], ['11-audit-a-complete-data-migration', 'Worked capstone'], ['12-practise-the-reasoning', 'Independent practice']]}>A sensor record says it belongs to the South site. Its device record says North. Both lookups work, but the system disagrees with itself. Category theory gives us a precise way to state which routes should agree, translate a whole system while retaining its rules, and build objects from their required interfaces. Start with that concrete job; the abstract language will grow out of it.</LessonIntro>

    <H2>1. Start with things you can connect</H2>
    <Prose>A <strong>function</strong> takes each allowed input to exactly one output. Its <strong>domain</strong> is the set of allowed inputs; its <strong>codomain</strong> is the declared set in which outputs live. The actual outputs reached form its image. For example, a sensor-to-device lookup has sensors as its domain and devices as its codomain. Two sensors may point to the same device, and a device may have no sensors.</Prose>
    <Prose>Write f:A→B for a function from A to B. If g:B→C follows it, we can make a new function: first apply f, then g. This <strong>composite</strong> is written g∘f. The notation reads right to left because (g∘f)(a)=g(f(a)). A sequence of arrows is read left to right: A → B → C. These are two notations for the same order of work.</Prose>
    <Prose>Types matter. A device-to-site lookup cannot accept a temperature just because both happen to be stored as integers. Declaring objects and allowed arrows makes that distinction explicit. Here a function must return an output for every member of its domain. A possible missing result needs a different declared result type, which we will build later.</Prose>
    <CompositionLab />
    <H3>Why regrouping is safe</H3>
    <Prose>Add h:C→D. We can first package g∘f as one function, or first package h∘g. For an arbitrary a in A, both packages return h(g(f(a))). Because they agree on every a and have the same domain and codomain, they are equal functions. This is <strong>associativity</strong>.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      [h\circ(g\circ f)](a)&=h(g(f(a))),\\
      [(h\circ g)\circ f](a)&=h(g(f(a))).
    \end{aligned}`}</MathBlock>
    <Prose>The <strong>identity</strong> id_A:A→A returns its input unchanged. Thus f∘id_A=f and id_B∘f=f. Notice the two different identity objects: one is attached to the input side, one to the output side.</Prose>
    <Example name="original" />
    <Callout title="Regrouping preserves order">Associativity never exchanges f and g. For f(x)=x+1 and g(x)=2x, g(f(4))=10 while f(g(4))=9. Nor does it justify caching a function that reads changing global state, dropping an effect, or running dependent operations in parallel. First specify a faithful model of those effects and the equality you mean.</Callout>
    <Checkpoint prompt="Two Python lambda objects return the same value for every integer. Must Python's == operator report that the functions are equal?"><Prose>No. Our mathematical equality is pointwise equality with the same declared domain and codomain. Python function objects normally compare by identity. The program tests returned values; the mathematical argument establishes the general equality.</Prose></Checkpoint>

    <H2>2. Name the system and its laws</H2>
    <Prose>A <strong>category</strong> packages objects and arrows that can be connected consistently. An arrow is also called a <strong>morphism</strong>. It has a specified source and target, but it need not be an ordinary function on elements. We need objects; arrows between each pair; an identity arrow for each object; and a chosen composite for every composable pair. The composites must obey the two identity laws and associativity.</Prose>
    <Prose>Hom_C(A,B) means the collection of arrows from A to B in category C. It may be empty. In a <strong>locally small</strong> category each such collection is a set; a <strong>small</strong> category also has a set of all objects and arrows. This distinction avoids treating “the set of every set” as an ordinary set. Our finite examples are small; Set is normally treated as a large, locally small category.</Prose>
    <LessonTable caption="Different objects can share the same composition language" headers={['Category', 'Objects and arrows', 'Why the laws hold']} rows={[['FinSet', 'Finite sets; total functions', 'Function substitution is associative; each set has its identity function.'], ['Mat over the reals', 'Nonnegative dimensions n; an arrow n→m is an m×n matrix', 'Matrix multiplication composes linear maps; the n×n identity acts as identity. Include the usual empty matrices when n=0.'], ['A preorder as a category', 'Its elements; exactly one arrow a→b when a≤b', 'Reflexivity supplies identities and transitivity supplies composites. At most one arrow per pair makes parallel composites equal.'], ['A monoid as a category', 'One object; each monoid element is an arrow from it to itself', 'The operation is associative and has an identity. Nonnegative integer addition is one example.']]} />
    <Prose>A <strong>preorder</strong> is a reflexive, transitive relation. A partial order also has antisymmetry. A <strong>monoid</strong> is a set with an associative binary operation and an identity element. A monoid's one object can have many distinct endoarrows, while a preorder's many objects have at most one arrow between any fixed pair. The words “object” and “arrow” separate these different roles.</Prose>
    <H3>A graph supplies possible steps; a category supplies composites</H3>
    <Prose>Suppose a graph has A→B and B→C but no drawn A→C edge. It is not yet a category with only those arrows: the required composite is missing. Its <strong>free category</strong> instead uses vertices as objects and all finite directed paths as arrows, including an empty path at each vertex. Composition concatenates paths. Empty paths are identities. Different paths remain different arrows unless we explicitly impose equations between them.</Prose>
    <Prose>For a sensor schema, impose directSite = siteOf∘deviceOf. This identifies two parallel paths. The relation must remain valid when composed with compatible paths on either side; taking that compatible equivalence gives a quotient of the path category. A <strong>commutative diagram</strong> is one whose indicated parallel routes have equal composites. Drawing a triangle does not make it commute.</Prose>
    <H3>Equality, inverse and isomorphism</H3>
    <Prose>An arrow f:A→B is an <strong>isomorphism</strong> if some g:B→A satisfies both g∘f=id_A and f∘g=id_B. Then the objects are interchangeable through these maps. The inverse is unique: if g and h are inverses, g=g∘id_B=g∘f∘h=id_A∘h=h. In FinSet this means a bijection, a function that is both one-to-one and onto.</Prose>
    <Prose>One equation is insufficient. Let i:{'{0}'}→{'{0,1}'} include 0, and let r:{'{0,1}'}→{'{0}'} send both values to 0. Then r∘i is identity on {'{0}'}, but i∘r sends 1 to 0. We call i a section and r a retraction. A reversible encoding needs both directions to restore their entire stated domains.</Prose>
    <details><summary>Deeper: cancellation and equivalence of categories</summary>
      <Prose>An arrow m is <strong>monic</strong> if m∘u=m∘v implies u=v for every parallel pair entering its source. An arrow e is <strong>epic</strong> if u∘e=v∘e implies u=v for every parallel pair leaving its target. In Set these mean injective and surjective. They are cancellation properties relative to the category's allowed arrows, so those interpretations cannot be assumed everywhere.</Prose>
      <Prose>In a preorder category every arrow is both monic and epic because there is at most one parallel arrow to compare. Yet 0→1 in the usual order has no inverse. Thus “monic and epic” does not imply isomorphism in every category.</Prose>
      <Prose>An <strong>equivalence of categories</strong> has translations back and forth whose composites are naturally isomorphic to the respective identity translations. This permits different representatives of the same structures. It is weaker than literal equality of object names and composites. We next define those translations and their compatibility.</Prose>
    </details>

    <H2>3. Translate a whole system</H2>
    <Prose>A <strong>functor</strong> F:C→D translates both objects and arrows. Each A becomes F(A); each f:A→B becomes F(f):F(A)→F(B). It must preserve identities and composition. Translating a finished path gives the same arrow as translating its steps and then composing them.</Prose>
    <MathBlock>{String.raw`\begin{aligned}
      F(\mathrm{id}_A)&=\mathrm{id}_{F(A)},\\
      F(g\circ f)&=F(g)\circ F(f).
    \end{aligned}`}</MathBlock>
    <Prose>Here is an actual functor. Let C be our three-object schema category: Sensor, Device and Site, with the three named arrows, their identities, and the direct-site equation. An instance F:C→FinSet assigns sets of rows to the objects and lookup functions to the arrows. Its identities are identity lookups. Its only nontrivial two-step composition must equal the direct lookup.</Prose>
    <SchemaFunctorLab />
    <Example name="schema" />
    <Prose>The source objects are schema roles; their images are sets of row identifiers. The source arrows are schema relationships; their images are actual functions on rows. This distinction makes the functor claim precise. Real databases can have nullable fields, duplicate rows, transactions and other semantics. Our finite-set instance does not automatically model those features.</Prose>
    <H3>A construction you have already programmed: lists</H3>
    <Prose>List sends a set A to all finite lists of A-elements, including the empty list. For f:A→B, List(f) applies f to each entry while keeping order and multiplicity. Its target is List(B). Even if A is finite, its lists have unbounded length, so this is conveniently a functor Set→Set, not an endofunctor on FinSet.</Prose>
    <Prose>Mapping id_A leaves every entry unchanged. At position i, first mapping f then g returns g(f(a_i)), exactly what mapping g∘f returns. Both sides preserve length. This position-by-position proof establishes the two functor laws, including the empty list case.</Prose>
    <Prose>A functor need not preserve all information. A constant functor can send every object to a singleton and every arrow to its identity. It still preserves the laws. Every functor does preserve isomorphisms: if g is inverse to f, then F(g)∘F(f)=F(g∘f)=F(id_A)= the identity on F(A), and the other inverse equation follows likewise.</Prose>
    <details><summary>Deeper: faithful, full and opposite</summary>
      <Prose>A functor is <strong>faithful</strong> if each map Hom_C(A,B)→Hom_D(F(A),F(B)) is injective, and <strong>full</strong> if each is surjective. These say how arrows are retained, not simply whether object labels are unique. A functor can identify objects yet have an injective map on every separate hom-set.</Prose>
      <Prose>The <strong>opposite category</strong> Cᵒᵖ formally reverses arrows and composition order. It does not assert that original arrows have inverse functions. A contravariant functor on C is an ordinary functor from Cᵒᵖ. Precomposing a probe into A with u:X→Y turns a probe Y→A into a probe X→A. We will use that reversal in the Yoneda branch.</Prose>
    </details>
    <H3>Connect to the previous lesson: homology carries maps too</H3>
    <Prose>Topology/TDA introduced cycles modulo boundaries. Over F₂, H_k(K)=Z_k(K)/B_k(K): cycles are chains with zero boundary, and cycles differing by a boundary represent the same class. An inclusion K→L maps each included simplex to itself, giving linear maps on chains in every dimension. More generally a chain map c obeys ∂c=c∂.</Prose>
    <HomologyFunctorFigure />
    <Prose>That equation does two jobs. If ∂z=0, then ∂c(z)=c(∂z)=0, so cycles map to cycles. If z′=z+∂b, then c(z′)=c(z)+∂c(b), so equivalent cycle representatives remain equivalent. Therefore [z]↦[c(z)] defines a well-defined linear map on the quotient. A composite d∘c sends [z] to [d(c(z))], exactly the composite of the induced maps; identity maps preserve classes. That proves the functor laws here.</Prose>
    <Prose>A triangle's three-edge cycle is nonzero before its face is filled. After filling, the same chain is the face's boundary and represents zero. An injective inclusion of spaces need not induce an injective homology map. Betti numbers record dimensions and cannot specify these induced maps. For general continuous maps, singular homology supplies a functor from topological spaces to vector spaces after choosing coefficients; the finite simplicial calculation is our example.</Prose>
    <Example name="homology" />

    <H2>4. Compare two compatible translations</H2>
    <Prose>Suppose F and G both translate C into D. A <strong>natural transformation</strong> η:F⇒G gives one arrow η_A:F(A)→G(A) for every object A. These components must be compatible with every source arrow f:A→B. Translate f using F and then change representation, or change representation first and translate f using G: both routes must agree.</Prose>
    <MathBlock>{String.raw`G(f)\circ\eta_A=\eta_B\circ F(f).`}</MathBlock>
    <Prose>Each side starts at F(A) and ends at G(B). Naming F, G, their categories and all component types is essential. “This conversion feels natural” is not the mathematical condition. Testing one input is not its universal proof.</Prose>
    <Prose>For F=G=List, let η_A reverse positions. For a list of length n, both routes put f(a[n−1−i]) at position i. Thus reversal is natural across all set maps. Every component reverses itself, so it is even a natural isomorphism.</Prose>
    <NaturalityLab />
    <Prose>Sorting is a useful contrast. On the selected ordered sets it defines a list operation, but allowing all underlying functions between those objects breaks compatibility. Sorting [2,0,1] then applying x↦2−x gives [2,1,0]. Mapping first and then sorting gives [0,1,2]. One failed square refutes naturality on that category. Restricting to suitable order-preserving maps changes the allowed arrows and the claim being tested.</Prose>
    <Example name="naturality" />
    <Prose>For database instances F,G:C→Set, a natural transformation assigns row maps at Sensor, Device and Site. The sensor and device row maps must respect deviceOf; likewise the site relationships. Renaming identifiers consistently is an example. Merging sites can also be compatible, while losing information. The capstone calculates a query consequence instead of assuming every result remains unchanged.</Prose>
    <Checkpoint prompt="A list operation agrees with mapping on the empty list and every one-element list. Does that establish naturality?"><Prose>No. Sorting passes those cases but fails on [2,0,1] under the reversing map. The condition quantifies over every permitted map and list. Reversal has a general index proof.</Prose></Checkpoint>

    <H2>5. Specify an object by what it must do</H2>
    <Prose>A downstream task needs both a sensor and a technician. A pair object should let us recover either choice and should contain exactly one representation of each pair. We can state that interface before choosing a memory layout.</Prose>
    <Prose>A <strong>product</strong> of A and B has an object P and projections π_A:P→A and π_B:P→B. For every object Z and arrows a:Z→A, b:Z→B, there must be exactly one arrow u:Z→P with π_A∘u=a and π_B∘u=b. This <strong>universal property</strong> covers every such competing arrangement. Both existence and uniqueness matter.</Prose>
    <UniversalProductLab />
    <Prose>In Set, choose P=A×B with π_A(x,y)=x and π_B(x,y)=y. Given a and b, define u(z)=(a(z),b(z)). It has the required projections, proving existence. Any other v must have first coordinate a(z) and second coordinate b(z) at every z, hence v=u. That proves uniqueness. The forced map is denoted ⟨a,b⟩.</Prose>
    <H3>Products do not require identical names or layouts</H3>
    <Prose>If P and Q both satisfy the product requirement, P's projections give a unique compatible u:P→Q, and Q's give v:Q→P. The composite v∘u has the same P-projections as id_P, so uniqueness forces v∘u=id_P. Similarly u∘v=id_Q. They are isomorphic by the unique projection-preserving map. “Unique up to unique compatible isomorphism” does not mean literally identical representations.</Prose>
    <H3>Three more constructions from arrow direction</H3>
    <Prose>A <strong>terminal object</strong> receives exactly one arrow from every object. In Set any singleton is terminal: there is one place to send each input. An initial object sends exactly one arrow to every object; the empty set does this, since it has no inputs needing choices.</Prose>
    <Prose>A <strong>coproduct</strong> reverses the product pattern. It has injections from A and B, and every pair of arrows A→Z, B→Z combines uniquely into one arrow from the coproduct to Z. In Set it is a tagged disjoint union: (left,a) and (right,b) remain distinct even if a=b. Ordinary union loses that tag and cannot combine rules sending the same shared label to different results. The unique function branches on the tag and applies the corresponding rule.</Prose>
    <Prose>A <strong>pullback</strong> handles pairs that must agree about something. Given a:A→S and b:B→S, form P={'{(x,y): a(x)=b(y)}'}. Its projections obey a∘π_A=b∘π_B. For any r:Z→A and s:Z→B with a∘r=b∘s, the unique mediator is z↦(r(z),s(z)). Compatibility puts the pair in P; its two coordinates ensure uniqueness.</Prose>
    <MathBlock>{String.raw`P=\{(x,y):\ a(x)=b(y)\}.`}</MathBlock>
    <Prose>Take A as sensors, B as technicians and S as sites. The pullback contains the pairs sharing a site. It is a finite-set model of a join on a common key. If a job chooses a sensor and technician at different sites, the compatibility premise fails; it is not a counterexample to the universal property.</Prose>
    <Example name="universal" />
    <details><summary>Deeper: limits and colimits</summary><Prose>A diagram of shape J in C is a functor J→C. A cone from Z supplies compatible arrows from Z to all its objects. A <strong>limit</strong> is a cone through which every cone factors uniquely. Products use a discrete shape with no nonidentity arrows; pullbacks use a cospan A→S←B. Reversing cone arrows gives cocones and <strong>colimits</strong>, including coproducts. Verification means naming the diagram, proving the candidate's cone equations, constructing a mediator for an arbitrary compatible cone, and proving it unique.</Prose></details>

    <H2>6. Compose parallel work and possible failure</H2>
    <Prose>Arrows so far connect end to end. To put independent operations beside each other, a <strong>monoidal category</strong> supplies a tensor ⊗ on objects and arrows, a unit object I, and coherent associativity and unit isomorphisms. In Set, use cartesian product and a singleton unit. The parallel map f×g sends (a,x) to (f(a),g(x)).</Prose>
    <ParallelCompositionFigure />
    <MathBlock>{String.raw`\begin{gathered}(h\otimes k)\circ(f\otimes g)\\=(h\circ f)\otimes(k\circ g).\end{gathered}`}</MathBlock>
    <Prose>This <strong>interchange law</strong> is functoriality of the tensor in its two inputs. In Set both sides return (h(f(a)),k(g(x))). A symmetric monoidal category also has a compatible swap of factors. Actual tuples ((a,b),c) and (a,(b,c)) are not literally equal; an associator rearranges brackets canonically. Coherence requires different valid sequences of structural rearrangements to agree.</Prose>
    <Prose>In vector spaces, the tensor product of dimensions m and n has dimension mn and encodes bilinear structure. It differs from the categorical product/direct sum of two finite-dimensional spaces, with dimension m+n. Also v↦v⊗v is generally not linear: (v+w)⊗(v+w) has cross terms. A picture borrowed from ordinary copying would not automatically be a valid linear arrow.</Prose>
    <H3>Make possible failure part of the interface</H3>
    <Prose>Option(A) contains a distinct None value and Some(a) for every a in A. A function f:A→Option(B) is total: failure is an explicit output. To follow it with g:B→Option(C), ordinary g(f(a)) has the wrong type. Instead propagate None; if f(a)=Some(b), call g(b).</Prose>
    <OptionCompositionFigure />
    <Prose>This is <strong>Kleisli composition</strong> for Option. Its identity arrow is a↦Some(a). Applying it before f returns f(a); applying it after f leaves None or Some(b) unchanged. Associativity follows by cases: if f fails, both groupings return None. If f succeeds but g fails, both return None. If both succeed, both call h on the same payload. These exhaust the possibilities.</Prose>
    <Example name="option" />
    <details><summary>Deeper: unit, flattening and a monad</summary>
      <Prose>Option maps ordinary functions by preserving None and sending Some(a) to Some(f(a)). Its unit η_A wraps a. Flattening μ_A:Option(Option(A))→Option(A) removes one success wrapper: None↦None, Some(None)↦None and Some(Some(a))↦Some(a). These maps are natural and satisfy two unit laws and associativity. Write T=Option in the following equations.</Prose>
      <MathBlock>{String.raw`\begin{aligned}
        \mu_A\circ\eta_{T(A)}&=\mathrm{id},\\
        \mu_A\circ T(\eta_A)&=\mathrm{id},\\
        \mu_A\circ T(\mu_A)&=\mu_A\circ\mu_{T(A)}.
      \end{aligned}`}</MathBlock>
      <Prose>The first two say adding then removing a success layer changes nothing. The third says flattening three layers in either order yields None if any encountered layer is None, otherwise Some(a). This endofunctor with natural unit/flattening and laws is a <strong>monad</strong>. The Kleisli composite is μ_C∘Option(g)∘f. A different effect needs its own declared maps and proofs.</Prose>
    </details>

    <H2>7. Compose randomness without losing dependence</H2>
    <Prose>A noisy sensor need not return the same output on each trial. A finite <strong>stochastic map</strong> P:X→Y assigns a distribution on Y to each x. Use a row per input x and column per output y. Entries P(y|x) are nonnegative and each row sums to one.</Prose>
    <Prose>To compose P:X→Y with Q:Y→Z, sum over the middle state. With this row convention the stored matrix is PQ, while the categorical arrow is Q∘P. A row-distribution p becomes pP. The identity channel outputs its input with probability one.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      (Q\circ P)(z\mid x)\\
      =\sum_yP(y\mid x)Q(z\mid y),\\[4pt]
      \sum_z(Q\circ P)(z\mid x)\\
      =\sum_yP(y\mid x)\underbrace{\sum_zQ(z\mid y)}_{1}\\
      =\sum_yP(y\mid x)=1.
    \end{gathered}`}</MathBlock>
    <Prose>Nonnegativity and normalization survive. Associativity follows by expanding both parenthesizations to the same finite double sum. This gives FinStoch. A deterministic function embeds by putting one 1 per row at its output. That inclusion preserves composition and identities.</Prose>
    <H3>What do parallel stochastic wires mean?</H3>
    <Prose>The tensor combines state sets by cartesian product and independent channels by multiplying conditional probabilities. A state is a channel from a singleton, hence a distribution. Two independent bit states with chance p of 1 give (1−p)², (1−p)p, p(1−p), p² for 00,01,10,11.</Prose>
    <StochasticCopyLab />
    <Prose>Copying one sampled bit instead gives 1−p,0,0,p. Both marginals still have chance p of 1. Since the joint is not uniquely determined by its marginals, this tensor with its usual projections does not satisfy the categorical product property. We found two different mediators from the singleton for the same pair of marginals.</Prose>
    <Prose>Copying is consequently not natural with respect to arbitrary stochastic arrows. A coin arrow followed by deterministic copy creates dependence; copying the trivial input first and running the coin independently in both lanes generally does not. They coincide at p=0 and p=1, the deterministic endpoints. A faithful probabilistic diagram retains this distinction.</Prose>
    <H3>Bayes reversal is not an inverse channel</H3>
    <Prose>With prior p(x) and channel P(y|x), q(y)=Σ_x p(x)P(y|x). For q(y)&gt; 0, the reverse conditional is R(x|y)=p(x)P(y|x)/q(y). It depends on the prior. Sending an input through P and drawing from R generally does not recover that exact input. If q(y)=0, the formula is undefined; choosing an arbitrary row is an extra convention.</Prose>
    <Example name="probability" />
    <Prose>For a concrete two-stage calculation, suppose each binary channel keeps a bit with probability 3/4 and flips it with probability 1/4. Starting from 0, the final output is 0 either through 0→0→0, with probability 9/16, or through 0→1→0, with probability 1/16. Adding these disjoint routes gives 10/16=5/8, the first entry in the composed matrix above. This finite construction helps connect probabilistic components and inspect independence assumptions in random preprocessing. It does not establish that a real sensor obeys the chosen channel; that needs measurement. General Markov categories extend the language, with further care for measurable spaces and conditional existence.</Prose>

    <H2>8. Keep the point when you differentiate</H2>
    <Prose>For f(x)=x+1 followed by g(y)=y², a tiny input change uses f′ at x and g′ at f(x). At x=0, the intermediate point is 1, so the composite derivative is 2. Evaluating g′ at 0 instead gives 0. The output point is part of the derivative interface.</Prose>
    <Prose>For a smooth f:ℝⁿ→ℝᵐ, define its <strong>tangent map</strong> Tf(x,v)=(f(x),Df_xv). Here x is the base point, v is the input tangent direction and Df_x is the Jacobian at x. The result contains both the new point and the new direction.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      Tf(x,v)=(f(x),Df_xv),\\[4pt]
      Tg(Tf(x,v))\\
      =(g(f(x)),Dg_{f(x)}Df_xv)\\
      =T(g\circ f)(x,v).
    \end{gathered}`}</MathBlock>
    <Prose>The last equality is the chain rule. Also T(id)(x,v)=(x,v). Taking ℝⁿ to ℝⁿ×ℝⁿ and smooth arrows to tangent maps therefore preserves composition and identity. Smoothness ensures Tf is again a smooth arrow; a bare differentiability assumption needs care if this is the chosen arrow class.</Prose>
    <TangentCompositionLab />
    <H3>Reverse sensitivities use saved forward points</H3>
    <Prose>An output cotangent w is a linear measurement of output change. Pulling it backward through f gives Df_xᵀw in Euclidean coordinates. The characterization is wᵀ(Df_xv)=(Df_xᵀw)ᵀv: the same scalar change measured on either side. For g∘f the pullback is Df_xᵀ Dg(f(x))ᵀ w, in reverse order. The base point f(x) remains essential.</Prose>
    <Prose>A reverse-mode engine therefore records or recomputes forward intermediate values. “Transpose every derivative and reverse all arrows” omits those dependencies. When a parameter is used along two paths, contributions add; ordinary differentiation of a sum already requires that accumulation.</Prose>
    <Example name="tangent" />
    <Prose>The prediction w(wx+b) has derivative (wx+b)+wx=2wx+b with respect to w. At w=2,x=3,b=1, prediction=14 and half-squared error against 10 is 8. Multiplying the output error 4 by local derivatives gives parameter gradients 52 and 8. The two uses of w explain the sum. The chosen update lowers this loss; composition laws do not prove every positive learning rate gives descent.</Prose>
    <details><summary>Where “backprop as functor” goes further</summary>
      <Prose>Fong, Spivak and Tuyéras formalize parameterized maps with update and request functions. Their stated theorem fixes a positive step and a differentiable scalar loss whose first-argument derivative, viewed as a function of the target for each fixed prediction, is invertible. Half-squared error meets that condition. The request function supplies the backward message needed to compose learners.</Prose>
      <Prose>Our tangent proof does not reproduce every construction in that paper or establish it for arbitrary optimizers and stateful training. Its value here is to specify what composes and why. The <a href="/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals">Backpropagation &amp; Automatic Differentiation</a> lesson develops the computation-graph engine.</Prose>
    </details>

    <H2>9. Answer a forward question with a backward filter</H2>
    <Prose>A function f maps source records to categories. Select source subset S and permitted target subset T. “Are all outputs reached by S inside T?” and “Does every selected record pass the T filter?” are the same question viewed in opposite directions.</Prose>
    <Prose>The <strong>direct image</strong> f(S) contains outputs reached from S. The <strong>preimage</strong> f⁻¹(T) contains every source whose image lies in T. Preimage notation does not require an inverse function: many sources may reach the same target.</Prose>
    <ImagePreimageLab />
    <MathBlock>{String.raw`f(S)\subseteq T\quad\Longleftrightarrow\quad S\subseteq f^{-1}(T).`}</MathBlock>
    <Prose>For the forward direction, choose any x∈S. Its image is in f(S), hence T, so x∈f⁻¹(T). Conversely, choose y∈f(S). Some x∈S has f(x)=y. The assumed inclusion puts x in f⁻¹(T), hence y∈T. Both directions use arbitrary elements and definitions; injectivity is unnecessary.</Prose>
    <Prose>View powersets as categories ordered by subset. Image and preimage are monotone, so they are functors between those preorder categories. The equivalence says image is <strong>left adjoint</strong> to preimage. An <strong>adjunction</strong> pairs translations so certain arrow questions on one side correspond exactly to arrow questions on the other.</Prose>
    <Prose>Set T=f(S) to get S⊆f⁻¹(f(S)): the round trip can add other sources with the same images. Set S=f⁻¹(T) to get f(f⁻¹(T))⊆T: targets without sources may disappear. These are the unit and counit containments here. Adjoints need not be inverses.</Prose>
    <Example name="adjunction" />
    <details><summary>Deeper: the hom-set statement and preservation</summary>
      <Prose>For F:C→D and G:D→C, an adjunction F⊣G is a bijection Hom_D(F(A),B)≅Hom_C(A,G(B)) natural in A and B. Changing A or B by an arrow and then translating must agree with translating first and composing with the corresponding arrow. Unrelated equal cardinalities are insufficient.</Prose>
      <Prose>Our powerset hom-sets contain one element when the inclusion holds and none otherwise. The containment equivalence supplies the bijection; compatibility follows from uniqueness of each existing arrow. In richer categories the actual bijection carries more information.</Prose>
      <Prose>Direct image preserves unions: outputs reached from any member of a family are precisely those reached from its union. Preimage preserves intersections and unions by the same f(x) membership test. Direct image can fail on intersections: if a≠b but f(a)=f(b)=0, the images of {'{a}'} and {'{b}'} overlap although their intersection is empty. General adjoint/limit theorems explain a broader pattern after these concrete mechanisms are understood.</Prose>
    </details>

    <H2>10. Reconstruct a system from compatible probes</H2>
    <Prose>This deeper branch makes “know an object by how other objects map into it” precise. Compatibility is doing real work; a few selected features are not automatically enough.</Prose>
    <Prose>Fix A in a locally small category C. Collect all probes X→A into h_A(X)=Hom_C(X,A). An arrow v:Y→X changes a probe u:X→A into u∘v:Y→A. Thus h_A is a contravariant functor Cᵒᵖ→Set, recording probe sets and how they change under every allowed input arrow.</Prose>
    <Prose>Let F:Cᵒᵖ→Set be another functor and θ:h_A⇒F a natural transformation. Apply the component θ_A to id_A. This gives one a=θ_A(id_A) in F(A). That single element determines every component.</Prose>
    <MathBlock>{String.raw`\begin{gathered}
      a=\theta_A(\mathrm{id}_A),\\
      \theta_X(u)=F(u)(a),\\
      \text{for every }u:X\to A,\\[4pt]
      \mathrm{Nat}(h_A,F)\cong F(A).
    \end{gathered}`}</MathBlock>
    <H3>Why the formula is forced, and why it works</H3>
    <Prose>Naturality for u:X→A compares applying θ_A to id_A and then F(u), with precomposing id_A by u and then applying θ_X. Precomposition produces u, so θ_X(u)=F(u)(a). No freedom remains.</Prose>
    <Prose>Conversely, choose any a∈F(A) and define θ by that formula. For v:Y→X, the required routes give F(v)(F(u)(a)) and F(u∘v)(a). They agree because F is contravariant. Evaluating at id_A returns a because F preserves identity. Reconstructing an existing natural θ returns the same θ by the forced formula. These operations are inverse bijections: the <strong>Yoneda lemma</strong> in contravariant form.</Prose>
    <Prose>If F=h_B, then a is an arrow A→B, and θ sends each u:X→A to a∘u:X→B. Every compatible transformation of all probes is postcomposition with one arrow. If θ is a natural isomorphism, its inverse reconstructs an inverse arrow, so A and B are isomorphic.</Prose>
    <Example name="yoneda" />
    <Prose>The program includes all functions between its three objects, including the empty object and every endomorphism of the two-element set. Its 1,024 candidate families reduce to four compatible ones. A learned embedding generally supplies neither all such probes nor a compatibility proof; Yoneda does not automatically establish that the vector determines the original object.</Prose>
    <Callout title="State a checkable question">A categorical proposal should name its objects, arrows, equality and preserved structure, then explain what the result buys. A research position proposing a common architecture language is not itself a theorem that every architecture works well or a performance comparison.</Callout>

    <H2>11. Audit a complete data migration</H2>
    <Prose>The old system has North and South, each with one sensor and technician. A migration merges the sites into Region and renames sensor and technician identifiers. Establish that the migration respects site lookups, then determine how previously compatible pairs map into the new system.</Prose>
    <Prose>For a sensor s, compare migrating its old site with looking up the site of its new identifier. Do the same for technicians. If both squares commute and old s,t share a site, their new sites equal the migrated old site. Thus the new pair is compatible, giving a well-defined map of pullbacks.</Prose>
    <Prose>The program starts with a bad new sensor record, reports the failing square, repairs it, and constructs the induced map. Finally it asks whether every new compatible pair comes from an old one.</Prose>
    <Example name="capstone" />
    <Prose>There are four new compatible pairs but only two images of old pairs. Before merging, s0 and t1 were at different sites; their new records both belong to Region. The commuting migration gives a valid induced map without making it surjective. Preserving schema equations does not mean preserving every query result bijectively.</Prose>
    <Prose>For your system, write the objects and arrow contracts, list meaningful path equations, and check the translation's component squares. Then identify the additional property a downstream task needs—isomorphism, injectivity or unique factorization—and justify it separately. A passing diagram should answer a stated question.</Prose>

    <H2>12. Practise the reasoning</H2>
    <Prose>Write your answer before opening a hint or solution. The programs above run with the standard library on Python 3.12. Examples help discover a proof, while an arbitrary-input argument establishes its general claim.</Prose>
    <Practice title="1. Separate three kinds of equality" prompt="On integers take f(x)=x−2, g(x)=3x, h(x)=x². Compute both parenthesizations at 4, compare g∘f and f∘g, then state the general law." hint="Trace the values in order, then replace 4 by an arbitrary x."><Prose>The path 4→2→6→36 gives 36 under either grouping. But g(f(4))=6 and f(g(4))=10. One input refutes equality of that pair. Associativity holds generally because both groupings return h(g(f(x))), not because one test produced 36.</Prose></Practice>
    <Practice title="2. Supply a category, not just a drawing" prompt="A graph has A,B and f:A→B. Describe its free category. Add g:B→A: is its free category finite?" hint="Include empty paths and repeated round trips."><Prose>Initially the arrows are id_A,id_B and f with the forced identity composites. Adding g gives arbitrarily long distinct paths, so the free category is infinite. Imposing g∘f=id_A and f∘g=id_B gives a different quotient category.</Prose></Practice>
    <Practice title="3. Diagnose a one-sided inverse" prompt="Let i:{a,b}→{0,1,2} send a↦0,b↦1; let r send 0↦a,1↦b,2↦a. Which inverse equation holds?" hint="Test the element i never reaches."><Prose>r∘i is identity, but i(r(2))=0. No bijection exists between these finite sets of different sizes. Restricting the second set to {'{0,1}'} makes the restricted maps inverse. Ignoring 2 without changing the declared domain does not.</Prose></Practice>
    <Practice title="4. Find the missing functor obligation" prompt="A schema requires c=b∘a. Set X={0,1},Y={u},Z={L,R}; a sends both inputs to u, b(u)=L, and c(0)=L,c(1)=R. Repair the instance." hint="Compare the routes at input 1."><Prose>b(a(1))=L differs from c(1)=R. Set c(1)=L while retaining a and b. Well-typed individual functions let us compose them; functoriality also requires equal schema arrows to have equal interpretations.</Prose></Practice>
    <Practice title="5. Prove a different natural transformation" prompt="Let K be the constant Set→Set functor with value the natural numbers and identity action on arrows. Prove list length is natural List⇒K." hint="State both routes on an arbitrary list, including the empty one."><Prose>Mapping any f preserves the number of positions, so length_B(List(f)(xs))=length_A(xs). Applying id_ℕ after length gives the same result. Empty lists give zero. These components are not isomorphisms: they discard values and order.</Prose></Practice>
    <Practice title="6. Use both halves of universality" prompt="For Z={j0,j1}, let a(j0)=red,a(j1)=blue and b(j0)=large,b(j1)=small. Give the product mediator. Refute ordinary union as a general coproduct." hint="Pair the coordinates; use overlapping sets for the coproduct counterexample."><Prose>The unique mediator sends j0→(red,large),j1→(blue,small). For the coproduct take A=B={'{x}'} and maps into {'{0,1}'} sending x to 0 and1 respectively. No function on their ordinary union can restrict to both. Tagged copies preserve the separate choices.</Prose></Practice>
    <Practice title="7. Check a pullback premise" prompt="Sensors s0,s1 are at N and s2 at S. Technician t0 is at N, t1,t2 at S. List compatible pairs. Can a job choosing s0 and t1 factor through them?" hint="Filter cartesian pairs by equality of sites."><Prose>The pairs are (s0,t0),(s1,t0),(s2,t1),(s2,t2). The proposed job chooses different sites, so its cone is not compatible. The universal property promises a mediator only when that premise holds.</Prose></Practice>
    <Practice title="8. Follow explicit failure" prompt="For nonnegative → reciprocal → add one, predict inputs −4,0,4. Explain why regrouping with a fourth possibly failing stage preserves the result." hint="Find the first failure; otherwise follow the same payloads."><Prose>−4 fails immediately ; 0 fails at reciprocal ; 4 gives Some(5/4). A fourth stage receives 5/4 only after success. Either grouping of three arrows returns None at the first failure, or calls all arrows on the same payloads. Repeated associativity extends this to four.</Prose></Practice>
    <Practice title="9. Compute dependence from wiring" prompt="A bit has chance 1/4 of 1. List the joint probabilities 00,01,10,11 for copying and independent sampling. Compute disagreement." hint="The independent case multiplies the marginals."><Prose>Copying gives [3/4,0,0,1/4]; independent sampling gives [9/16,3/16,3/16,1/16]. Disagreement is 0 versus 3/8. Both marginals are [3/4,1/4].</Prose></Practice>
    <Practice title="10. Retain the derivative base point" prompt="For f(x)=x²,g(y)=y³ at x=2, take input tangent v=−1 and output cotangent w=2. Compute forward and reverse changes and verify the pairing." hint="The intermediate point is 4; evaluate g′ there."><Prose>f′(2)=4,g′(4)=48, so the composite derivative is 192. The tangent goes −1→−4→−192. The cotangent goes backward 2→96→384. The scalar pairing is 2·(−192)=384·(−1)=−384. Evaluating g′ at 2 would use the wrong point.</Prose></Practice>
    <Practice title="11. Work an adjunction with missing targets" prompt="Map a,b to 0 and c to 1 in Y={0,1,2}. For S={a},T={0,2}, compute image, preimage and both round trips." hint="Target 2 has no source; a shares its image with b."><Prose>f(S)={'{0}'} and f⁻¹(T)={'{a,b}'}. The round trip f⁻¹(f(S)) adds b; f(f⁻¹(T))={'{0}'} drops 2. Both adjunction containments hold. The map forgets which of a,b produced 0, and preimage cannot invent a source for 2.</Prose></Practice>
    <Practice title="12. Reconstruct and challenge" prompt="A natural transformation h_A⇒h_B sends id_A to k. What must it do to u:X→A? In the capstone, what if all identifiers were renamed bijectively and sites were never merged?" hint="Use naturality at u; then use injectivity of the site renaming."><Prose>The probe must become k∘u. Naturality forces it. For the capstone, bijective component maps and an injective site map reflect site equality. Their inverses recover a unique old compatible pair from each new pair, so the induced pair map is a bijection. The merge lacked that reflection of equality.</Prose></Practice>
    <Prose><strong>Readiness check.</strong> Put types on every arrow, distinguish finite evidence from universal equations, construct an actual functor and natural transformation, prove a unique factorization, and diagnose an invalid probability or derivative diagram. If an application merely resembles a familiar picture, name the missing data or proof obligation.</Prose>
    <Prose><strong>Next in this module:</strong> <a href="/learn/path/full-curriculum/differential-geometry-riemannian-manifolds?module=math-foundations">Differential Geometry &amp; Riemannian Manifolds</a> develops smooth spaces, tangent directions and metric-dependent measurements. Our tangent map's retained base point is the bridge. Publication status does not change this syllabus sequence.</Prose>
    <Sources alternatives={<p>For a spoken route use <a href="https://ocw.mit.edu/courses/18-s097-applied-category-theory-january-iap-2019/pages/lecture-videos-and-readings/chapter-3-databases-categories-functors-and-co-limits/" target="_blank" rel="noreferrer">MIT 18.S097: databases, categories, functors and (co)limits</a>, taught by David Spivak and Brendan Fong with matched readings. The <a href="https://www.youtube.com/watch?v=UusLtx9fIjs" target="_blank" rel="noreferrer">Topos Institute's opening lecture</a> offers an earlier route through orders. The course map and recording identity were checked; full-video viewing and specific timestamps are not claimed.</p>}>
      <li><a href="https://emilyriehl.github.io/files/context.pdf" target="_blank" rel="noreferrer">Riehl, Category Theory in Context</a> — author-hosted second-edition PDF. Selected sections 1.1,1.3,1.4,2.2,3.1–3.2 and4.1 support the formal definitions and proofs. It assumes mathematical maturity; work the finite examples first. The full book was not reviewed.</li>
      <li><a href="https://dspivak.net/7Sketches.pdf" target="_blank" rel="noreferrer">Fong and Spivak, Seven Sketches in Compositionality</a> — chapter 3 develops schemas and set-valued instances. The inspected author-hosted version is from 2018. Its broader applications complement the original sensor examples here.</li>
      <li><a href="https://arxiv.org/abs/1908.07021" target="_blank" rel="noreferrer">Fritz, A synthetic approach to Markov kernels and statistics</a> — example 2.5 in the inspected 2020 version gives the finite stochastic/tensor conventions. General conditional-independence and sufficient-statistic theory is advanced continuation.</li>
      <li><a href="https://arxiv.org/abs/1711.10455" target="_blank" rel="noreferrer">Fong, Spivak and Tuyéras, Backprop as Functor</a> — the inspected 2019 theorem III.2 and quadratic-loss example specify the extra parameter/update/request structure. Compare its hypotheses with our simpler tangent proof.</li>
      <li><a href="https://arxiv.org/abs/2402.15332" target="_blank" rel="noreferrer">Position: Categorical Deep Learning is an Algebraic Theory of All Architectures</a> — a research perspective, not a performance guarantee or established completeness claim. The listed 2024 v2 abstract and an earlier-version introduction were inspected, not the entire formal development.</li>
    </Sources>
  </div>
};
