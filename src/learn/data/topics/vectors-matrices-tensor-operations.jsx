import { Callout, Code, H2, H3, Prose } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { Checkpoint, LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample';
import { CompositionFigure, LinearMapLab, MatrixProductLab, TensorReductionLab, TensorReindexFigure, VectorCombinationFigure, VectorProjectionLab } from '../../components/lesson-labs/VectorTensorLabs';
import { vectorTensorExamples } from '../vector-tensor-examples.js';
function Practice({
  title,
  children,
  hint,
  solution
}) {
  return <section className="vector-practice">
      <H3>{title}</H3>
      {children}
      <details>
        <summary>Optional hint</summary>
        <Prose>{hint}</Prose>
      </details>
      <details>
        <summary>Worked solution and reasoning</summary>
        {solution}
      </details>
    </section>;
}
export default {
  title: 'Vectors, Matrices & Tensor Operations',
  readTime: '~50 min read + 75 min practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot vectors-lesson">
    <LessonIntro exampleKind="Optional NumPy" prerequisites="Signed arithmetic, coordinate pairs, basic algebra and function substitution. We refresh unit directions, length and summation notation. Python is useful for optional checks; the mathematical route can be followed by hand." sections={[['1-name-the-quantities-before-the-shape', 'Objects & units'], ['2-build-a-vector-from-directions', 'Vector geometry'], ['3-a-dot-product-and-its-geometric-meaning', 'Dot & projection'], ['4-a-matrix-is-also-a-map', 'Matrix maps'], ['5-multiply-matrices-one-output-at-a-time', 'Products'], ['6-compose-maps-and-recognize-lost-information', 'Structure'], ['7-give-every-tensor-axis-a-name', 'Tensor axes'], ['8-rearrange-values-without-losing-their-meaning', 'Reindex & broadcast'], ['9-connect-the-operations-to-a-complete-pipeline', 'Applications'], ['10-practise-explain-and-check-readiness', 'Independent practice']]}>
      A moving point, a parts order and a batch of measurements all need the same habit: identify the quantities, decide which ones combine, and interpret the result. See vector and matrix operations geometrically, calculate them by hand, and keep their meaning intact as data gains more axes.
    </LessonIntro>
    <Prose><strong>First pass:</strong> follow the numbered route and try each prediction. Optional branches extend the core without hiding essential steps. Finish able to explain why an operation is valid, identify its contributing inputs and catch a result whose shape is legal but whose meaning is wrong.</Prose>

    <H2>1. Name the quantities before the shape</H2>
    <Prose>Move two metres east and one metre north. One number cannot describe both parts, so write the ordered pair (2,1). The first coordinate means east, the second north. Swap them and the movement changes. Convert metres to centimetres and the same movement has coordinates (200,100). Numbers need interpretation before their operations become meaningful.</Prose>
    <Prose>A <strong>scalar</strong> is one number. A <strong>vector</strong> is an object we can add and scale; a chosen coordinate system represents it by an ordered list. A <strong>matrix</strong> arranges numbers into rows and columns. A numerical <strong>tensor</strong> is an array indexed by zero or more axes, such as session, time and channel. Mathematical tensor theory places additional transformation rules on the represented objects; here we use the numerical-computing meaning, with that deeper distinction reserved for Tensor Algebra.</Prose>
    <LessonTable caption="Shape describes indexing; names describe the question" headers={['Object', 'Shape and one value', 'Meaning']} rows={[['Scalar', '() · 4', 'One count, temperature or coefficient; its role supplies the unit.'], ['Vector', '(2,) · x[0]=2', 'Two spatial coordinates, or two product counts in one order: different interpretations.'], ['Matrix', '(3,2) · X[1,0]=0', 'Three orders × two product types; order 1 has no small packs.'], ['Three-axis array', '(2,2,3) · T[1,0,2]=8', 'Session 1, time 0, channel 2 in the later fixture.'], ['Image batch', '(8,28,28,1)', 'Eight images × height × width × one channel; order is a convention.']]} />
    <Prose>A shape lists axis lengths; their product counts elements. Shape (2,2,3) contains 12 numbers. A vector in two-dimensional space can live in a <em>one-axis</em> array of length two. Vector-space dimension and array-axis count are different. Array “rank” sometimes means axis count; <strong>matrix rank</strong> will mean independent output directions.</Prose>
    <MathBlock caption="ℝ denotes the real numbers.">{String.raw`\begin{gathered}x\in\mathbb{R}^{2},\quad X\in\mathbb{R}^{3\times2}\\T\in\mathbb{R}^{2\times2\times3}\end{gathered}`}</MathBlock>
    <Prose>Hand mathematics below uses <strong>column vectors</strong>, with coordinates written vertically, and Ax for a map applied to x. A dataset often stores observations as rows, giving the batch formula XW. Match shapes and the side on which the map acts. Mathematical indices commonly begin at 1; code and explorer indices begin at 0.</Prose>
    <Checkpoint prompt="A row contains height in metres and age in years. Is its ordinary Euclidean length a physical distance?"><Prose>No. Metres squared plus years squared does not define squared metres. Scaling features can define a useful numerical comparison, but it is a modeling choice. Our geometric arrows use commensurate spatial coordinates; tables use named data axes.</Prose></Checkpoint>

    <H2>2. Build a vector from directions</H2>
    <Prose>Adding displacements means doing one move and then another. For u=(2,1), v=(−1,2) metres, add components: u+v=(2−1,1+2)=(1,3). A negative horizontal coordinate means west. The second arrow can start at u's tip without changing the displacement it represents.</Prose>
    <VectorCombinationFigure />
    <Prose><strong>Scalar multiplication</strong> scales every component: 2u=(4,2), −u=(−2,−1), 0u=(0,0). A negative multiplier reverses direction. Subtraction u−v adds the reverse of v, giving (3,−1). For positions p and q in one frame, q−p is their displacement. Position depends on the origin; displacement describes a difference.</Prose>
    <MathBlock>{String.raw`\begin{gathered}e_1=\begin{bmatrix}1\\0\end{bmatrix},\quad e_2=\begin{bmatrix}0\\1\end{bmatrix}\\\begin{bmatrix}2\\1\end{bmatrix}=2e_1+e_2\end{gathered}`}</MathBlock>
    <Prose>The unit directions e₁ and e₂ form a <strong>basis</strong>: every plane vector has exactly one pair of coefficients in these directions. Scaling vectors and adding them is a <strong>linear combination</strong>. Changing the basis changes the coordinates without necessarily changing the physical movement.</Prose>
    <Prose>Pythagoras gives length √(2²+1²)=√5≈2.236 metres. This is the Euclidean norm, written with double bars. For d coordinates, square every component, add and take the square root. Σ means add the indexed terms.</Prose>
    <MathBlock>{String.raw`\begin{gathered}\|x\|_2=\sqrt{\sum_{i=1}^{d}x_i^2}\\\operatorname{distance}(p,q)=\|q-p\|_2\end{gathered}`}</MathBlock>
    <Prose>For nonzero x, x/‖x‖₂ has the same direction and unit length. Zero has no direction and cannot be normalized this way. L₁ length sums absolute components; L∞ length takes the largest absolute component. For (3,−4), these are 7 and 4, while Euclidean length is 5. Choose the notion of size that answers your question.</Prose>
    <Practice title="Reconstruct a displacement" hint="Combine coordinates first; only then compute length." solution={<Prose>2u−v=(2,4)−(3,−1)=(−1,5), with length √26 and basis coefficients −1 and 5. In general ‖2u−v‖ is not 2‖u‖−‖v‖: lengths discard directions that matter to addition.</Prose>}>
      <Prose>For the new pair u=(1,2), v=(3,−1), calculate 2u−v, its Euclidean length and standard-basis coefficients. Explain why subtracting lengths is wrong.</Prose>
    </Practice>

    <H2>3. A dot product and its geometric meaning</H2>
    <Prose>A <strong>dot product</strong> multiplies matching components and adds. For v=(3,2), u=(2,1), v·u=3×2+2×1=8. The result is scalar. It can represent a weighted score: each weight changes one feature's contribution. Physically interpreted summands must have compatible output units.</Prose>
    <MathBlock>{String.raw`\begin{aligned}v\cdot u&=\sum_i v_i u_i=v^\top u\\v\cdot u&=\|v\|_2\|u\|_2\cos\theta\end{aligned}`}</MathBlock>
    <Prose>For nonzero real vectors measured along perpendicular unit coordinate axes (an <strong>orthonormal</strong> frame), θ is their angle. Positive, zero and negative dot products correspond to acute, right and obtuse angles. Large values can reflect length rather than close direction: (100,0)·(1,1)=100 at 45°. Divide by both lengths for <strong>cosine similarity</strong>, undefined when either vector is zero.</Prose>
    <Prose><strong>Projection</strong> asks for the part of v along the line through u. Write that point as p=αu. Require the remainder v−αu to be perpendicular to u, so its dot product with u is zero. Expand and solve for α:</Prose>
    <MathBlock>{String.raw`\begin{aligned}(v-\alpha u)\cdot u&=0\\v\cdot u-\alpha(u\cdot u)&=0\end{aligned}`}</MathBlock>
    <MathBlock>{String.raw`\alpha=\frac{v\cdot u}{u\cdot u},\qquad p=\frac{v\cdot u}{u\cdot u}u\quad(u\ne0)`}</MathBlock>
    <Prose>Here u·u=5, α=8/5, p=(3.2,1.6), and v−p=(−0.2,0.4). Check: −0.2×2+0.4×1=0. Coefficient 1.6 is not the projection's length unless u is a unit vector and the coefficient is nonnegative. The signed scalar component along the unit direction is (v·u)/‖u‖₂.</Prose>
    <VectorProjectionLab />
    <details className="vector-practice"><summary>Deeper: why is p the closest point on the line?</summary><Prose>Let r=v−p, with r·u=0. Any other point is p+tu. Its squared distance from v is ‖r−tu‖²=‖r‖²−2t(r·u)+t²‖u‖²=‖r‖²+t²‖u‖². The additional term is nonnegative and, for nonzero u, zero only at t=0. Thus p is uniquely closest. This prepares the least-squares idea.</Prose></details>
    <Prose><strong>Optional Python checks:</strong> every block is complete. Use an environment with NumPy installed, save a block as <Code>vectors_example.py</Code> and run <Code>python vectors_example.py</Code>, or paste it into a fresh notebook cell. Tested with Python 3.12 / NumPy 2.3.5. The browser explorers calculate bounded mathematical examples; they do not run your Python.</Prose>
    <RunnableExample example={vectorTensorExamples.projection}><Prose>Floats approximate many real values. <Code>isclose</Code> and <Code>allclose</Code> allow a tolerance for tiny errors. Their default tolerances suit these fixtures, not every scientific accuracy requirement.</Prose></RunnableExample>
    <Checkpoint prompt="Does v·u=0 prove two nonzero perpendicular directions?"><Prose>No. Zero has dot product zero with everything but no angle. Algebraic orthogonality includes zero vectors; a right-angle drawing requires nonzero directions.</Prose></Checkpoint>

    <H2>4. A matrix is also a map</H2>
    <Prose>A shear moves (x,y) to (x+y,y). Thus (2,1) becomes (3,1), and (0,2) becomes (2,2). The first matrix row calculates the new horizontal coordinate; the second calculates the vertical coordinate.</Prose>
    <MathBlock>{String.raw`\begin{gathered}A=\begin{bmatrix}1&1\\0&1\end{bmatrix}\\A\begin{bmatrix}2\\1\end{bmatrix}=\begin{bmatrix}1(2)+1(1)\\0(2)+1(1)\end{bmatrix}=\begin{bmatrix}3\\1\end{bmatrix}\end{gathered}`}</MathBlock>
    <Prose>A sends e₁ to its first column (1,0), and e₂ to its second column (1,1). Since (2,1)=2e₁+e₂, its output combines twice column one with column two. Row calculations and column combinations are two views of one operation.</Prose>
    <MathBlock>{String.raw`Ax=x_1A_{:,1}+x_2A_{:,2}+\cdots+x_nA_{:,n}`}</MathBlock>
    <Prose>A[:,j] means the entire j-th column. An m×n matrix maps n input coordinates to m output coordinates. A 1×2 matrix can turn two inputs into one measurement; a square matrix is only a special case.</Prose>
    <LinearMapLab />
    <Prose>A map L is <strong>linear</strong> if L(u+v)=L(u)+L(v) and L(cu)=cL(u). Matrix multiplication preserves addition and scaling because every output is a sum of fixed coefficients times inputs. These rules explain why basis images determine all outputs. A linear map sends zero to zero.</Prose>
    <Prose>A translation (x,y)↦(x+3,y) moves zero, so it is not linear. Ax+b is <strong>affine</strong>. Libraries often call an affine neural operation a “linear layer”; inspect its bias before applying the mathematical definition. Its scores are not automatically probabilities.</Prose>
    <Checkpoint prompt="F(x,y)=(x²,y²) fixes zero. Is it linear?"><Prose>No: F(2,0)=(4,0), but 2F(1,0)=(2,0). Scaling fails. Fixing zero is necessary, not sufficient.</Prose></Checkpoint>

    <H2>5. Multiply matrices one output at a time</H2>
    <Prose>Three orders request small and large packs. Small packs require three panels/two bolts; large packs require five panels/four bolts. X has order rows and product columns. W has the same product order in its rows, and resource columns.</Prose>
    <LessonTable caption="Three invented orders" headers={['Order', 'Small packs', 'Large packs']} rows={[[0, 2, 1], [1, 0, 3], [2, 1, 2]]} />
    <LessonTable caption="Resources per pack" headers={['Product', 'Panels', 'Bolts']} rows={[['Small', 3, 2], ['Large', 5, 4]]} />
    <Prose>Order 0's panel count pairs its row [2,1] with the panels column [3,5]: 2×3+1×5=11 panels. Packs×panels/pack supplies panels. Its bolts pair the same row with [2,4]: 2×2+1×4=8. Repeat for every order/resource pair.</Prose>
    <MathBlock>{String.raw`\begin{bmatrix}2&1\\0&3\\1&2\end{bmatrix}\begin{bmatrix}3&2\\5&4\end{bmatrix}=\begin{bmatrix}11&8\\15&12\\13&10\end{bmatrix}`}</MathBlock>
    <MatrixProductLab />
    <Prose>Shape (m,n) times (n,p) gives (m,p). Shared n counts the terms in each sum; it is <strong>contracted</strong>, or summed away. Hold output i,j fixed while k varies.</Prose>
    <MathBlock>{String.raw`(AB)_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}`}</MathBlock>
    <Prose>Size establishes arithmetic compatibility; labels and order establish semantic compatibility. Reverse only W's small/large rows and the product still runs, but computes the wrong resource requirements. Libraries cannot infer names or units.</Prose>
    <LessonTable caption="Different products, different questions" headers={['Operation', 'Example', 'Meaning']} rows={[['Elementwise', '[2,1] * [3,5] = [6,5]', 'Keep contributions separate.'], ['Dot', '[2,1] @ [3,5] = 11', 'Sum matching contributions.'], ['Outer', '[2,1] column × [3,5] row = [[6,10],[3,5]]', 'Every pair, without a shared sum.'], ['Matrix', 'X @ W', 'One dot product per output pair.']]} />
    <Prose>NumPy * is elementwise, possibly broadcasting. @ follows matrix-product rules. A d-by-e outer product is a matrix, not a scalar dot product.</Prose>
    <RunnableExample example={vectorTensorExamples.resources}><Prose>[1,0] adds a spare panel to every order afterward. Summing axis 0 removes orders and totals resources. A Boolean mask selects an order with no small packs. These are distinct questions about the same table.</Prose></RunnableExample>

    <H2>6. Compose maps and recognize lost information</H2>
    <Prose>If B acts first and A second, A(Bx)=(AB)x. Column j of AB is A applied to column j of B, where B sent the j-th unit vector. The rightmost map touches x first. Order matters.</Prose>
    <CompositionFigure />
    <RunnableExample example={vectorTensorExamples.maps}><Prose>Associativity allows regrouping (AB)x as A(Bx) in exact arithmetic with compatible shapes, not swapping A and B. Floating-point regrouping may change rounding. Identity, with diagonal ones and other entries zero, preserves every vector.</Prose></RunnableExample>
    <H3>Transpose and conjugate transpose</H3>
    <Prose>Aᵀ swaps indices: (Aᵀ)ᵢⱼ=Aⱼᵢ, so (m,n) becomes (n,m). It is not generally an inverse. The shear inverse subtracts the vertical contribution; its transpose moves the off-diagonal 1 to another position.</Prose>
    <Prose>For complex entries, ordinary transpose still only swaps indices. The <strong>conjugate transpose</strong> A*=conj(A)ᵀ also changes i to −i in each entry; it belongs to the usual complex inner product. For z=[i], zᵀz=−1 but z*z=1, the usual squared length. NumPy @ never conjugates automatically.</Prose>
    <details className="vector-practice"><summary>Useful structure names for decompositions</summary>
      <LessonTable caption="Properties with a concrete test" headers={['Name', 'Condition', 'Example / consequence']} rows={[['Symmetric real matrix', 'Aᵀ=A', 'Mirrored off-diagonal entries do not alone imply positive xᵀAx.'], ['Orthogonal square matrix', 'QᵀQ=I', 'Unit, mutually perpendicular columns; ‖Qx‖²=xᵀQᵀQx=‖x‖²; Q⁻¹=Qᵀ. A quarter-turn qualifies.'], ['Positive definite real symmetric matrix', 'xᵀAx>0 for every nonzero x', 'diag(2,1) qualifies: 2x₁²+x₂²>0. diag(1,−1) fails at (0,1).'], ['Frobenius norm', 'Square root of sum of squared entries', 'An entrywise size, not rank or a universal distortion measure.']]} /><Prose>The next lesson uses these conditions to choose factorizations; this branch introduces meanings rather than algorithms.</Prose>
    </details>
    <H3>Span, independence, rank and null space</H3>
    <Prose>The <strong>span</strong> of columns is all their linear combinations, exactly the reachable outputs Ax. Columns are <strong>independent</strong> when no nonzero coefficient vector combines them to zero. Otherwise some input information disappears. <strong>Rank</strong> is the dimension of this output span, bounded by both row and column counts.</Prose>
    <Prose>The collapse A=[[1,1],[0,0]] has repeated columns (1,0). All outputs are (x₁+x₂,0), a one-dimensional span: rank one. The <strong>null space</strong> contains inputs sent to zero, here all (t,−t). It means vectors, not missing data. For n inputs, rank plus null-space dimension equals n; here 1+1=2.</Prose>
    <LessonTable caption="Is a requested output reachable and unique?" headers={['Map and b', 'Conclusion', 'Reason']} rows={[['Collapse [[1,1],[0,0]], b=(3,1)', 'No solution', 'Every output has second coordinate zero.'], ['Same collapse, b=(3,0)', 'Infinitely many', 'x=(3−t,t) for every real t.'], ['Shear [[1,1],[0,1]], b=(3,1)', 'Unique x=(2,1)', 'x₂=1, then x₁+1=3.']]} />
    <Prose>A square full-rank matrix has an inverse with A⁻¹A=I. A rank-deficient square matrix does not. Rectangular matrices can still have unique inputs for reachable outputs if columns are independent. Numerical code generally solves directly instead of forming an inverse merely to multiply by b. Decompositions develop solving, approximation and sensitivity next.</Prose>
    <H3>A useful lost direction: reject an equal offset</H3>
    <Prose>Readings 11 and 15 differ by [-1,1]·[11,15]=4. Add the same offset c to both: −(11+c)+(15+c)=4. The unwanted [c,c] lies in this measurement map's null space. The information lost is exactly the nuisance here. Unequal offsets or arbitrary noise do not cancel; the difference cannot recover absolute readings.</Prose>
    <RunnableExample example={vectorTensorExamples.nullspace}><Prose>The last two lines check complex products separately; the earlier calculations are real. This is an invented algebraic scenario, not a validated sensor or clinical claim.</Prose></RunnableExample>

    <H2>7. Give every tensor axis a name</H2>
    <Prose>In shape (2,2,3), axis 0 selects session, axis 1 time and axis 2 channel. Each session is a 2×3 table of time rows/channel columns. T[1,0,2]=8 in our distinct-value fixture, all in one arbitrary unit.</Prose>
    <Prose>A <strong>reduction</strong> combines values and removes an index. For a time mean, fix session and channel, sum two time values and divide by two. Shape becomes (2 sessions,3 channels). Session 0/channel 0 gives (0+3)/2=1.5; session 1/channel 0 gives (6+9)/2=7.5. Sessions remain separate.</Prose>
    <TensorReductionLab />
    <Prose>Averaging session instead gives (0+6)/2=3 at time 0/channel 0. It also returns (2,3), but time/channel survive. Shape equality hides different questions. Negative axis −1 means the last axis; omitting axis in NumPy mean combines all values into one scalar.</Prose>
    <MathBlock>{String.raw`M_{s,c}=\frac{1}{2}\sum_{t=0}^{1}T_{s,t,c}`}</MathBlock>
    <Prose>s,c remain on both sides while t is summed. Mean divides by the count included. Missing values and padding need an explicit policy; ordinary mean cannot infer which entries are real observations.</Prose>
    <RunnableExample example={vectorTensorExamples.reduction}><Prose><Code>keepdims=True</Code> leaves the time axis as length one. Shape (2,1,3) preserves its position for subtraction from (2,2,3). Each session/channel's centered time mean is zero.</Prose></RunnableExample>
    <Checkpoint prompt="Can equal axis lengths reveal whether axis=0 means time or session?"><Prose>No. Numbers index positions in the declared convention. Names and the question determine which axis to combine.</Prose></Checkpoint>

    <H2>8. Rearrange values without losing their meaning</H2>
    <Prose><strong>Transpose</strong> permutes axis roles; <strong>reshape</strong> regroups the indexed sequence, preserving element count. Identical output shapes can contain different arrangements. Follow a distinctive value.</Prose>
    <TensorReindexFigure />
    <Prose>For (2,3,4), <Code>transpose(1,0,2)</Code> gives (3,2,4), moving (s,t,c) to (t,s,c). <Code>moveaxis(T,-1,0)</Code> moves the last axis first, preserving the others' relative order. <Code>T.T</Code> reverses <em>all</em> axes. To transpose only each matrix in a batch, use an explicit permutation or <Code>swapaxes(-1,-2)</Code>.</Prose>
    <Prose>Six elements can reshape to (3,2), not (4,2). One −1 lets NumPy infer a length. <Code>reshape(batch,-1)</Code> preserves batch items only if their values were grouped together in the selected index order. Reshape may return a view or a copy. Transpose/moveaxis normally return views; changes may affect shared data. The NumPy lesson develops storage semantics.</Prose>
    <RunnableExample example={vectorTensorExamples.reindex}><Prose>Shape (3,) is neither a (1,3) row matrix nor a (3,1) column matrix; its transpose stays (3,). None inserts a singleton axis. Stack adds an axis to equal-shaped arrays; concatenate extends an existing one. Selecting columns is a separate operation.</Prose></RunnableExample>
    <H3>Broadcasting aligns from the right</H3>
    <Prose>Align shapes at their right edges. Each pair of lengths must match or contain a one; missing leading dimensions act as one. A singleton supplies the same input along the varying axis. This explains values without requiring literal input copies.</Prose>
    <LessonTable caption="Predict aligned shapes and meaning" headers={['Left', 'Right', 'Result']} rows={[['(4,3) observations × features', '(3,) → (1,3)', '(4,3): feature offsets repeat across observations.'], ['(4,3)', '(4,1)', '(4,3): one observation offset repeats across features.'], ['(4,3)', '(4,)', 'Invalid: final 3 and 4 disagree.'], ['(3,1)', '(3,) → (1,3)', '(3,3): every pair, not three paired results.'], ['(2,2,3) session/time/channel', '(2,1,3)', '(2,2,3): baseline repeats over time only.']]} />
    <Prose>For X=[[2,10],[4,14]], row means are [6,9]. Shape (2,) aligns with columns: column one loses 6, column two loses 9. We wanted whole row one to lose 6 and row two to lose 9. Keep the reduced axis for shape (2,1).</Prose>
    <RunnableExample example={vectorTensorExamples.broadcasting}><Prose>The wrong rows fail to average to zero. Outer addition is useful when every pair is intended. Broadcasting avoids input repetition but can create an output much larger than either input; check element count before large allocations.</Prose></RunnableExample>
    <LessonTable caption="Select an operation by its question" headers={['Question', 'Operation', 'Contract']} rows={[['Select observations', 'Index / slice / mask', 'Scalar indexing removes an axis; length-one slices preserve it. Mask shape must fit.'], ['Collect equal-shaped samples', 'stack', 'Create a new axis.'], ['Append table rows', 'concatenate axis 0', 'Other dimensions match; no new axis.'], ['Divide a batch equally', 'split', 'Integer group count divides the selected length.'], ['Change API axis order', 'transpose / moveaxis', 'Track an indexed value.'], ['Regroup a sequence', 'reshape', 'Same count and meaningful new grouping.']]} />
    <RunnableExample example={vectorTensorExamples.images}><Prose>Marker 843 checks the axis move; equality checks split/recombine. Markers are not image intensities. For colour images, name channel order too: shape cannot distinguish RGB from BGR.</Prose></RunnableExample>

    <H2>9. Connect the operations to a complete pipeline</H2>
    <Prose>The original lesson's model has four observations, three features and two scores. Each column of W contains three weights for one score. XW computes four pairs of weighted sums; b=(0.01,−0.02) adds one bias per output across rows.</Prose>
    <MathBlock>{String.raw`\begin{gathered}(4,3)\ @\ (3,2)\ \longrightarrow\ (4,2)\\Y=XW+b\end{gathered}`}</MathBlock>
    <Prose>For [1,2,3], score 0=1×0.2+2×0.4+3×0.1+0.01=1.31. Score 1=1×(−0.1)+2×0.3+3×0.5−0.02=1.98. These are scores, not probabilities. Different feature units are compatible when weights convert terms to common output units.</Prose>
    <RunnableExample example={vectorTensorExamples.affine}><Prose>Centering separately subtracts each feature's observation mean, making centered column sums zero. Scores use original X; this program does not silently feed transformed inputs to unchanged weights.</Prose></RunnableExample>
    <H3>Leading batch axes preserve separate examples</H3>
    <Prose>Embeddings (batch,token,hidden)=(2,2,3) multiplied by W=(3,2) give (2,2,2). Under NumPy @, final two axes form matrix products; leading axes index batches. Each sequence uses the same W. In general (B,L,H)@(H,V) gives (B,L,V): H disappears, batch/token survive.</Prose>
    <Prose>With two higher-axis operands, batch shapes must broadcast too: (5,1,2,3)@(7,3,4) gives (5,7,2,4). Inner 3 matches and leading (5,1)/(7,) broadcast to (5,7). One-dimensional operands get temporary row/column slots removed afterward: (d,)@(d,) is scalar; (m,d)@(d,) gives (m,). A scalar uses *, not @.</Prose>
    <RunnableExample example={vectorTensorExamples.tokens}><Prose>Fixed shared W distributes through the sum: mean(XW)=mean(X)W. This fails for general nonlinear operations: mean([−1,1]²)=1, but mean([−1,1])²=0. Padded tokens need a validity mask and nonzero valid count; ordinary mean divides by full length.</Prose></RunnableExample>
    <details className="vector-practice"><summary>Deeper connections and numerical boundaries</summary><Prose>General index sums lead to tensor contraction/einsum. Factoring maps leads to decompositions; perpendicular residuals lead to least squares. Track the indices and relationships that survive.</Prose><Prose>Real identities are exact; floats round. Large values may overflow, small differences may lose relative precision, and near-dependent columns make recovery sensitive. Fixed-width integers can overflow too. Choose dtypes and tolerances deliberately. Float32 mean generally keeps a float32 accumulator; requesting float64 can improve accumulation but cannot restore precision already absent from inputs. These small fixtures are not stability benchmarks.</Prose></details>

    <H2>10. Practise, explain and check readiness</H2>
    <Prose>Try these with solutions closed. Supply shape, calculation and interpretation; code can check your reasoning afterward.</Prose>
    <Practice title="Project a new vector and diagnose a sign" hint="Compute v·u/u·u and test the residual's dot product." solution={<Prose>v·u=−1, u·u=2, α=−1/2, p=(−1/2,−1/2), r=(−7/2,7/2), and r·u=0. Negative α means the opposite half of the line. Projection length is √(1/2), not −1/2.</Prose>}>
      <Prose>Project v=(−4,3) onto u=(1,1). Find the residual. Repair the claim “the projection length is negative” and check perpendicularity.</Prose>
    </Practice>
    <Practice title="A changed resource order" hint="One row times the same recipe; add the spare afterward." solution={<Prose>[4,2]W=[22,16], then [23,16] with the spare. Large recipe [6,4] gives [24,16], then [25,16]. Reversing only the original recipe rows gives [26,20]: valid arithmetic with misaligned product labels.</Prose>}>
      <Prose>Compute panels/bolts for four small and two large packs plus one spare panel. Change large packs to six panels/four bolts. Explain a size-compatible label-order error.</Prose>
    </Practice>
    <Practice title="Recover an input, or prove you cannot" hint="Compare the two scalar equations." solution={<Prose>For A=[[1,2],[2,4]], equation two is twice equation one. Output (3,6) allows x=(3−2t,t) for all real t. Output (3,7) requires 6=7: impossible. A(−2,1)=0, so rank is one. Finding a least-squares approximation asks a different question, developed next.</Prose>}>
      <Prose>Solve Ax=(3,6), then Ax=(3,7), for A=[[1,2],[2,4]]. Find a nonzero lost input and explain recovery.</Prose>
    </Practice>
    <Practice title="Repair a measurement pipeline" hint="Remove only time; retain its singleton position for subtraction." solution={<><Prose>Time means are [[1.5,2.5,3.5],[7.5,8.5,9.5]]. Use <Code>baseline=T.mean(axis=1, keepdims=True)</Code>, shape (2,1,3), then T−baseline. Centered time means are zero for each session/channel.</Prose><Prose>Unkept (2,3) also broadcasts to (2,2,3), but its leading 2 aligns with time. Session 0 subtracts [1.5,2.5,3.5] at time 0 and [7.5,8.5,9.5] at time 1. Equal lengths hide the wrong axis meaning.</Prose></>}>
      <Prose>Average the 0…11 tensor over time, then center each session/channel. A colleague subtracts the shape-(2,3) result directly. Why does it run but fail? Repair it and give an output-based check.</Prose>
    </Practice>
    <Practice title="Compare two observations with four references" hint="Keep observation/reference indices separate and sum features." solution={<><Prose><Code>X @ R.T</Code> is shape (2,4), with entry (i,j)=ΣₖXᵢₖRⱼₖ. It yields [[1,0,1,2],[0,2,0,2]]. First observation/last reference gives 1×1+0×1+1×1=2.</Prose><Prose>These are dot scores. For angle comparisons normalize nonzero rows and define a zero-row policy. The final reference has a larger norm, so raw scores are not angle-only. Identify both output axes and demonstrate one entry by hand.</Prose></>}>
      <Prose>For X=[[1,0,1],[0,2,0]], R=[[1,0,0],[0,1,0],[0,0,1],[1,1,1]], compare every observation/reference by dot product. Predict expression, shape and values. What changes for angle similarity?</Prose>
    </Practice>
    <Prose><strong>Ready?</strong> Explain one column geometrically and one product cell algebraically; distinguish dot/outer/elementwise operations; name surviving reduction axes; repair a legal but wrong broadcast; give a map that loses information. Revisit the explorer if any answer relies on guessing.</Prose>
    <Callout accent="green" label="Next in this module"><a href="/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu">Matrix Decompositions (SVD, QR, Cholesky, LU)</a> factors complicated maps into simpler maps to solve systems, find directions and approximate data. Column span, projection and rank supply the starting language.</Callout>
    <Sources alternatives={<div><h4>Another explanation or guided practice</h4><ul>
      <li><a href="https://www.youtube.com/watch?v=kYB8IZa5AuE" target="_blank" rel="noopener noreferrer">3Blue1Brown — Linear transformations and matrices (video)</a>, with the <a href="https://www.3blue1brown.com/lessons/linear-transformations/" target="_blank" rel="noopener noreferrer">creator's text adaptation</a>. Revisit basis images after section 4; elementary coordinates suffice. The companion explanation was reviewed; no whole-video viewing is claimed.</li>
      <li><a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/ax-b-and-the-four-subspaces/the-geometry-of-linear-equations/" target="_blank" rel="noopener noreferrer">MIT OpenCourseWare — The Geometry of Linear Equations</a>. Strang lecture, notes and solved practice after sections 4–6. An introductory university route through row/column interpretations, not a NumPy tutorial.</li>
    </ul></div>}>
      <li><a href="https://textbooks.math.gatech.edu/ila/dot-product.html" target="_blank" rel="noopener noreferrer">Interactive Linear Algebra — Dot Products and Orthogonality</a>. Definitions, geometric reasoning and exercises for real-vector lengths and angles.</li>
      <li><a href="https://textbooks.math.gatech.edu/ila/matrix-multiplication.html" target="_blank" rel="noopener noreferrer">Interactive Linear Algebra — Matrix Multiplication</a>. Row-column products, composition and algebraic caveats.</li>
      <li><a href="https://textbooks.math.gatech.edu/ila/dimension.html" target="_blank" rel="noopener noreferrer">Interactive Linear Algebra — Basis and Dimension</a>. Extend section 6 from visible plane examples to column/null spaces and independent coordinates; some later exercises use row reduction.</li>
      <li><a href="https://numpy.org/doc/stable/reference/generated/numpy.matmul.html" target="_blank" rel="noopener noreferrer">NumPy — matmul</a> and <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html" target="_blank" rel="noopener noreferrer">broadcasting tutorial</a>. Batch/vector rules and shape alignment. Current stable docs checked; examples tested with NumPy 2.3.5.</li>
      <li><a href="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html" target="_blank" rel="noopener noreferrer">reshape</a>, <a href="https://numpy.org/doc/stable/reference/generated/numpy.transpose.html" target="_blank" rel="noopener noreferrer">transpose</a> and <a href="https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html" target="_blank" rel="noopener noreferrer">moveaxis</a>. Index order, axis permutations and view/copy limits.</li>
      <li><a href="https://numpy.org/doc/stable/reference/generated/numpy.mean.html" target="_blank" rel="noopener noreferrer">mean</a>, <a href="https://numpy.org/doc/stable/reference/generated/numpy.stack.html" target="_blank" rel="noopener noreferrer">stack</a> and <a href="https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html" target="_blank" rel="noopener noreferrer">concatenate</a>. Reduction/keepdims, accumulator dtype and creating versus extending axes.</li>
    </Sources>
  </div>
};
