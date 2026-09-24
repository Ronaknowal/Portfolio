import { Callout, Code, H2, H3, Prose } from '../../components/content';
import { MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro, LessonTable, Sources } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { CircleAtlasLab, TangentPatchFigure, MetricDifferentialLab, SphereArcLab, SphereBandFigure, SphereStepLab, PolarConnectionLab, SphereTransportLab, CurvatureComparisonLab, CovariancePathsFigure } from '../../components/lesson-labs/DifferentialGeometryLabs.jsx';
import { differentialGeometryExamples as examples } from '../differential-geometry-examples.js';

function GeometryExample({ example }) {
  return <><Prose><strong>Before running:</strong> {example.question}</Prose>
    <RunnableExample example={example} /><Prose>{example.explanation}</Prose></>;
}
function Practice({ question, hint, children }) {
  return <div className="dg-practice"><Prose><strong>Try independently.</strong> {question}</Prose>
    <details><summary>Hint</summary><Prose>{hint}</Prose></details>
    <details><summary>Show explained solution</summary>{children}</details></div>;
}

export default {
  title: 'Differential Geometry & Riemannian Manifolds',
  readTime: '~90 min read + 3–4 hours investigations and practice',
  hasIntegratedGuide: true,
  content: () => <div className="lesson-pilot differential-geometry-lesson">
    <LessonIntro prerequisites="Vectors, dot products, matrix multiplication and ordinary derivatives are useful. We refresh the particular chain rule, partial derivatives and positive-definite matrices we need. Tensor Algebra, Multivariate Calculus and Topology/TDA provide longer background. The labs need no installation; complete Python examples specify their additional libraries."
      sections={[
        ['1-a-point-is-not-its-coordinates','Points, charts and local coordinates'],
        ['2-allowed-velocities-and-local-measurement','Tangent vectors, differentials and metrics'],
        ['3-steepest-depends-on-the-metric','Gradients and coordinate changes'],
        ['4-measure-paths-and-surface-area','Length, distance, energy and area'],
        ['5-follow-a-geodesic-or-take-a-valid-step','Geodesics, Exp, Log and retraction'],
        ['6-differentiate-a-moving-basis','Connections and covariant derivatives'],
        ['7-carry-an-arrow-and-detect-curvature','Transport and intrinsic curvature'],
        ['8-choose-geometry-for-an-application','Means, covariances, Fisher and diffusion'],
        ['9-complete-a-constrained-optimization','A checked optimization workflow'],
        ['10-practise-the-connections','Changed problems and explained solutions'],
        ['11-continue-and-choose-another-learning-route','Next connections and resources'],
      ]}>Learn how to calculate when valid states live on a curved space: describe a point, find allowed directions, measure movement, differentiate an objective and take a valid step. A circle and a sphere carry the core explanation; later examples show why the chosen geometry matters.</LessonIntro>

    <H2>1. A point is not its coordinates</H2>
    <Prose>A camera points in a direction. If you represent that direction by a three-component vector of length one, the valid states lie on a sphere. Adding an arbitrary vector usually leaves the sphere. A compass has the same issue in one fewer dimension: its possible directions form a circle. You need calculus that respects the set of valid states.</Prose>
    <Prose><strong>Differential geometry</strong> studies smooth spaces through local coordinates, derivatives and geometric quantities. “Local” is essential. A small section of a circle behaves like an interval even though the whole circle does not have the shape of a line. A small patch of a sphere behaves like a two-dimensional region even though the entire sphere cannot be flattened into one distortion-free map.</Prose>
    <Prose>A <strong>manifold</strong> is a space that locally has the topology of an open region of ordinary Euclidean space. Its dimension counts independent local coordinates. The circle is one-dimensional and the sphere's surface is two-dimensional; their surroundings do not determine their intrinsic dimension. We use smooth, finite-dimensional manifolds without boundary unless explicitly stated.</Prose>
    <Prose>A <strong>chart</strong> assigns coordinates to an open patch, reversibly and continuously. For a smooth manifold, coordinate changes between overlapping charts are smooth with smooth inverses. Formally we also require the space to be Hausdorff, so distinct points have disjoint neighborhoods, and second countable, so a countable collection of basic open sets suffices. Those conditions exclude pathological spaces; the circle and sphere satisfy them.</Prose>
    <Prose>To see why several charts can help, describe a circle point by an angle. One chart uses α in (−π,π) and excludes the leftmost point. Another uses β in (0,2π) and excludes the rightmost point. Where both apply, the labels either agree or differ by 2π. These cases lie on different connected pieces of the overlap. There is no discontinuity within either chart's domain.</Prose>
    <CircleAtlasLab />
    <Prose>An <strong>atlas</strong> is a collection of compatible charts covering the space. A chart seam is a failure of that label, not a missing physical state. Longitude at a pole is a similar issue: all longitudes name the same pole. The sphere remains smooth there; that coordinate map stops distinguishing directions.</Prose>
    <Callout label="Local Euclidean structure is a real condition">The crossing point of two lines drawn as an X is not a one-dimensional manifold point. Removing its center from any sufficiently small neighborhood leaves four arms; removing a point from an interval leaves two. A drawing that looks curved or branched does not automatically define a smooth manifold.</Callout>
    <Prose><strong>Run the code:</strong> save any complete block as a .py file and run <Code>python filename.py</Code>. Standard-library examples need only Python 3. Programs using NumPy or SymPy show installation once; the browser labs are separate deterministic JavaScript calculations. Printed decimals are rounded results, not claims of exact real arithmetic.</Prose>
    <GeometryExample example={examples.charts} />

    <H2>2. Allowed velocities and local measurement</H2>
    <Prose>Imagine moving along a valid smooth curve c(t) on the sphere. At a chosen time, c′(t) is your instantaneous velocity. It need not itself be a point on the sphere. The set of all such velocities at a point x is its <strong>tangent space</strong>, written TₓM. It is an ordinary vector space attached to that point.</Prose>
    <Prose>For the unit sphere, differentiate c(t)·c(t)=1. The product rule gives 2c·c′=0. At x, every allowed velocity v therefore obeys x·v=0. These are exactly the vectors perpendicular to the radius: a two-dimensional plane through the vector-space origin. In a picture we often translate that plane to touch x, but translating a velocity arrow is a drawing convention.</Prose>
    <MathBlock>{String.raw`\begin{gathered}S^2=\{x\in\mathbb R^3:x^\top x=1\},\\T_xS^2=\{v:x^\top v=0\}.\end{gathered}`}</MathBlock>
    <Prose>More generally, a regular level set h(x)=0 has tangent directions in the null space of its derivative: Dh(x)v=0. “Regular” means the constraint derivative has full rank. At a rank failure, this calculation alone does not certify a smooth manifold or its dimension.</Prose>
    <H3>A local parameterization turns coordinate changes into velocities</H3>
    <Prose>Let F(q¹,q²) map two coordinates into a surface in three-dimensional space. Its columns ∂₁F and ∂₂F form a tangent basis when the derivative has rank two. If the coordinate velocity is v=(v¹,v²), the physical velocity is Jv, where J is the three-by-two Jacobian with those columns. Superscripts here label components; they are not powers.</Prose>
    <Prose>For a sphere of radius R, take polar angle θ measured down from the north pole and longitude φ. The local formula is F(θ,φ)=R(sinθ cosφ, sinθ sinφ, cosθ). Use 0&lt;θ&lt;π and a longitude interval that avoids its seam. The θ basis has length R; the φ basis has length R sinθ. Equal coordinate changes do not generally mean equal physical travel.</Prose>
    <TangentPatchFigure />
    <Prose>A <strong>differential</strong> describes the first-order effect of a velocity on a function. For a scalar f, dfₓ(v) is the rate of change of f along any curve with initial velocity v. In coordinates it is the row of partial derivatives acting on the velocity column. It is a linear functional, also called a <strong>covector</strong>, rather than a preferred movement direction.</Prose>
    <MathBlock>{String.raw`\begin{aligned}\frac{d}{dt}f(c(t))\big|_{t=0}&=df_x(v),\\df_x(v)&=\sum_i(\partial_i f)v^i.\end{aligned}`}</MathBlock>
    <Prose>For a map H:M→N rather than a scalar, its differential sends TₓM to T_H(x)N. In coordinate charts it is the Jacobian of the coordinate version of H. Differentials compose by the chain rule. This is matrix Jacobian composition from multivariate calculus, with the source and destination tangent spaces made explicit.</Prose>
    <H3>A metric says how to measure a velocity</H3>
    <Prose>A <strong>Riemannian metric</strong> assigns a smoothly varying positive-definite inner product to every tangent space. It tells us the squared length of a velocity and the angle between two velocities at the same point. “Positive definite” means every nonzero velocity has strictly positive squared length. We are not using indefinite spacetime metrics in this lesson.</Prose>
    <MathBlock>{String.raw`\begin{gathered}\langle v,w\rangle_q=v^\top G(q)w,\\\|v\|_q=\sqrt{v^\top G(q)v},\\\cos\angle(v,w)=\frac{v^\top G(q)w}{\|v\|_q\|w\|_q}.\end{gathered}`}</MathBlock>
    <Prose>Here G(q) is the symmetric matrix of metric coefficients in the current coordinate basis. If the surface inherits the ambient dot product, then (Jv)·(Jw)=vᵀ(JᵀJ)w, so G=JᵀJ. This is an <strong>induced metric</strong>. A manifold can have other metrics; an embedding does not force every application to choose the induced one.</Prose>
    <MathBlock>{String.raw`\begin{aligned}ds_{\rm plane}^2&=dr^2+r^2d\theta^2,\\ds_{\rm sphere}^2&=R^2d\theta^2\\&\quad+R^2\sin^2\theta\,d\phi^2.\end{aligned}`}</MathBlock>
    <Prose>The symbol ds² abbreviates squared length of an infinitesimal displacement. On a radius-two sphere at θ=60°, a small longitude change δφ has length approximately √3|δφ|. At a pole, the displayed matrix becomes singular because this coordinate basis collapses; the sphere's metric is nondegenerate in a valid chart.</Prose>

    <H2>3. Steepest depends on the metric</H2>
    <Prose>The differential answers “what change in f does this velocity cause?” To choose the steepest direction, we also need a rule for which velocities count as equally long. The <strong>Riemannian gradient</strong> is the vector whose metric pairing with every v equals df(v). If the differential coefficients form a column a, this requirement is gᵀGv=aᵀv for every v, hence Gg=a.</Prose>
    <MathBlock>{String.raw`\begin{gathered}\langle\operatorname{grad}f,v\rangle=df(v),\\\operatorname{grad}f=G^{-1}a.\end{gathered}`}</MathBlock>
    <Prose>Why is it steepest? The metric Cauchy–Schwarz inequality gives df(v)≤‖grad f‖ for every unit v. Equality occurs when v is the normalized gradient, unless the gradient is zero. Negative gradient gives fastest first-order decrease per metric unit. This is local, not a guarantee about a large step.</Prose>
    <H3>The same space in a sheared coordinate grid</H3>
    <Prose>Use physical coordinates x=u+v and y=v. The coordinate basis vectors are (1,0) and (1,1), so they are neither perpendicular nor equally long. Let f=2x−y. In u,v this is 2u+v: the differential coefficients are (2,1). With the ordinary physical dot product the coordinate metric is [[1,1],[1,2]]. Solving Gg=(2,1) gives g=(3,−1). Transform that vector back: (3−1,−1)=(2,−1), exactly the physical gradient.</Prose>
    <Prose>For v=(1,2), df(v)=2·1+1·2=4. Pairing the gradient through G also gives 4. Treating the coefficient column (2,1) as the gradient would give the wrong metric pairing. This develops the covector/metric distinction from Tensor Algebra into a calculation you can use on each tangent space.</Prose>
    <MetricDifferentialLab />
    <Prose>Changing coordinates and changing the metric are separate operations. Under a coordinate change with physical velocity v=Sũ, the same metric becomes G_new=SᵀG_old S. The differential components become a_new=Sᵀa_old, and gradient components transform as S⁻¹g_old. But increasing the physical cost of y-motion, replacing dx²+dy² by dx²+c²dy², changes the gradient of 2x−y to (2,−1/c²). The objective is unchanged; the definition of equal effort changed.</Prose>
    <GeometryExample example={examples.metric} />
    <H3>Why sphere gradients use a projection</H3>
    <Prose>Suppose f is the restriction of a smooth ambient function to the unit sphere and the metric is induced by the ambient dot product. An ambient gradient a may have a radial part (x·a)x that does no work on any tangent v, because x·v=0. Removing it preserves every tangent directional derivative:</Prose>
    <MathBlock>{String.raw`\operatorname{grad}_{S^2}f(x)=a-(x^\top a)x.`}</MathBlock>
    <Prose>At x=(1,0,0) with a=(1,2,0), the tangent gradient is (0,2,0). For a sphere of radius R, the removed part is (x·a)x/R². This projection depends on the induced metric; it does not replace solving with an arbitrary metric matrix.</Prose>

    <H2>4. Measure paths and surface area</H2>
    <Prose>The metric measures a velocity at one point. To measure a whole path, add up its instantaneous speed. For a piecewise smooth curve q(t), t∈[0,T], its length and fixed-time energy are:</Prose>
    <MathBlock>{String.raw`\begin{aligned}L(q)&=\int_0^T\sqrt{\dot q^\top G(q)\dot q}\,dt,\\E(q)&=\frac12\int_0^T\dot q^\top G(q)\dot q\,dt.\end{aligned}`}</MathBlock>
    <Prose>Length is unchanged by a regular increasing reparameterization: speeding up changes the speed and the time element in compensating ways. Energy is different. Cauchy–Schwarz applied to speed and the constant function 1 gives L²≤2TE, with equality exactly for constant speed, apart from negligible times. For a fixed duration, traveling the same path unevenly costs more energy under this definition.</Prose>
    <Prose>The <strong>Riemannian distance</strong> between two points is the infimum of lengths of connecting paths. It measures permitted travel, not a shortcut through an ambient space. On a round sphere, points subtending a smaller central angle α∈[0,π] have distance Rα. Their ambient chord has length 2R sin(α/2). For perpendicular unit directions these are π/2≈1.571 and √2≈1.414.</Prose>
    <SphereArcLab />
    <Prose>Two distinct non-antipodal sphere points have a unique shortest great-circle segment. The same great circle supplies a longer segment. At antipodes, every semicircle in a plane through the common diameter is shortest: the distance is definite but the shortest direction is not. Distance, chosen path and parameterization answer different questions.</Prose>
    <GeometryExample example={examples.paths} />
    <H3>The metric also tells us how much area a coordinate cell covers</H3>
    <Prose>A tiny coordinate rectangle maps to a tangent parallelogram. Its squared area is the determinant of the two-by-two Gram matrix G: |e₁|²|e₂|²−(e₁·e₂)². A Riemannian volume element in n coordinates is therefore √det(G) times the coordinate volume. On the sphere, it is R²sinθ dθ dφ.</Prose>
    <Prose>In three-dimensional space, the cross product e₁×e₂ is perpendicular to both basis vectors and has magnitude equal to their parallelogram's area. That gives another way to compute the same multiplier, used by <Code>np.cross</Code> in the program. For unit vectors x,y it also gives ‖x×y‖=sinα, where α is their angle.</Prose>
    <Prose>Integrating longitude from 0 to 2π and polar angle from a to b gives band area 2πR²(cos a−cos b). Integrating θ from 0 to π gives full area 4πR². Equal θ-width bands are not equal-area bands because longitude circles shrink near the poles.</Prose>
    <SphereBandFigure />
    <GeometryExample example={examples.area} />
    <Prose>A uniform surface point can be constructed by choosing z/R=cosθ uniformly from [−1,1] and φ uniformly from [0,2π), independently. The change of variable absorbs sinθ. Choosing θ uniformly instead concentrates too much probability per unit area near the poles. This is the Jacobian/reference-measure principle from Measure Theory appearing as a geometric volume element.</Prose>

    <H2>5. Follow a geodesic or take a valid step</H2>
    <Prose>A straight Euclidean path has constant velocity and zero acceleration. A surface path usually needs ambient acceleration to stay on the surface. The geometric replacement for straightness is zero <em>tangent</em> acceleration for an induced surface metric. Such a constant-speed locally straight path is a <strong>geodesic</strong>. Section 6 gives the coordinate version for a general Riemannian metric.</Prose>
    <Prose>On the unit sphere, start at x with tangent velocity v and write ℓ=‖v‖. The curve c(t)=cos(tℓ)x+sin(tℓ)v/ℓ stays unit, begins with velocity v and satisfies c″=−ℓ²c. Its acceleration is purely radial, so its intrinsic acceleration is zero. At ℓ=0 the curve is constant. At time one this defines the sphere's exponential map:</Prose>
    <MathBlock>{String.raw`\begin{aligned}\operatorname{Exp}_x(v)&=\cos\ell\,x\\&\quad+\operatorname{sinc}(\ell)v,\\\operatorname{sinc}(\ell)&=\sin\ell/\ell,\\\operatorname{sinc}(0)&=1.\end{aligned}`}</MathBlock>
    <Prose>The word “exponential” names a geometric map from tangent velocity to endpoint, not an entrywise exponential. On a general manifold Expₓ(v) means following the geodesic with initial velocity v for time one wherever that solution exists. The sphere is complete and permits all tangent velocities; an incomplete manifold can have a restricted domain.</Prose>
    <Prose>The inverse <strong>logarithm map</strong> turns a nearby point y into the starting velocity of its shortest connecting geodesic. On a unit sphere away from the antipode, let α=atan2(‖x×y‖,x·y). Then Logₓ(y)=α(y−cosα x)/sinα, with zero at y=x. It has length α. The code uses an equivalent cross-product expression to reduce cancellation near coincident points and rejects numerically unresolved antipodes.</Prose>
    <Callout label="Local inverse, not a global promise">For tangent length less than π on the sphere, Logₓ(Expₓ(v))=v. After a longer journey, a shortest Log can choose a different velocity to the same endpoint. At the antipode it is nonunique. A geodesic is locally minimizing on sufficiently short segments; it need not minimize over an arbitrary long interval.</Callout>
    <Prose>A useful global existence theorem is Hopf–Rinow: on a connected, complete, finite-dimensional Riemannian manifold, any two points can be joined by a minimizing geodesic. This neither guarantees uniqueness nor makes every geodesic segment minimizing. Completeness means the metric has no missing finite-distance limits; we will see a statistical example where that fails.</Prose>
    <H3>A retraction is a simpler local return map</H3>
    <Prose>Optimization often needs a valid step without solving an exact geodesic. A <strong>retraction</strong> Rₓ(v) satisfies Rₓ(0)=x and has derivative equal to the identity on tangent directions at zero. On a unit sphere, normalize the tangent candidate: Rₓ(v)=(x+v)/‖x+v‖. Since x·v=0, its denominator is √(1+‖v‖²), never zero.</Prose>
    <Prose>Expₓ(v) travels angle ‖v‖. This normalized retraction travels angle atan‖v‖ in the same great-circle direction. Their derivatives agree at a zero step, and this particular retraction agrees to second order; it is not exact for finite steps. Retractions are chosen algorithms, not a new distance definition.</Prose>
    <GeometryExample example={examples.original} />
    <SphereStepLab />
    <GeometryExample example={examples.maps} />
    <Prose>The sphere lab uses a two-dimensional cross-section so the radial/tangent split and endpoint angles are visible. Its start and step controls change the same linear objective throughout. Try a large step: a point can remain perfectly unit-normalized while its objective rises. Step selection still matters after the geometry is correct.</Prose>


    <H2>6. Differentiate a moving basis</H2>
    <Prose>A <strong>vector field</strong> chooses one tangent vector at each point of a region. How should we differentiate it? In Cartesian space we subtract nearby vectors using a common fixed basis. In general coordinates, the basis itself changes, and nearby vectors lie in different tangent spaces. Ignoring that change confuses component variation with physical variation.</Prose>
    <Prose>A <strong>connection</strong> specifies how to differentiate tangent fields in tangent directions. The covariant derivative ∇_X Y is linear in the direction X and obeys the product rule in the field Y. In a coordinate basis ∂ᵢ, define Γᵏᵢⱼ by ∇_∂ᵢ∂ⱼ=ΣₖΓᵏᵢⱼ∂ₖ. Differentiating Y=ΣYʲ∂ⱼ has two parts: changing components and changing basis.</Prose>
    <MathBlock>{String.raw`\begin{aligned}(\nabla_XY)^k&=\sum_iX^i\partial_iY^k\\&\quad+\sum_{i,j}\Gamma^k_{ij}X^iY^j.\end{aligned}`}</MathBlock>
    <Prose>Along a curve q(t), replace X by its velocity. Parallel transport sets this derivative to zero. A geodesic transports its own velocity parallel to itself, giving the coordinate equations:</Prose>
    <MathBlock>{String.raw`\begin{gathered}\frac{dV^k}{dt}+\sum_{i,j}\Gamma^k_{ij}\dot q^iV^j=0,\\\ddot q^k+\sum_{i,j}\Gamma^k_{ij}\dot q^i\dot q^j=0.\end{gathered}`}</MathBlock>
    <H3>The metric selects the Levi-Civita connection</H3>
    <Prose>A connection is extra structure in general. A Riemannian metric selects a unique connection that is <strong>metric-compatible</strong>, so differentiating an inner product follows the expected product rule, and <strong>torsion-free</strong>, so ∇_X Y−∇_Y X equals the commutator [X,Y]. The commutator means the difference between applying directional differentiations in the two orders; coordinate basis fields commute. Thus torsion-free gives Γᵏᵢⱼ=Γᵏⱼᵢ in those bases.</Prose>
    <Prose>We can derive the coefficients. Write Gᵢⱼ=⟨∂ᵢ,∂ⱼ⟩. Metric compatibility says ∂ᵢGⱼₗ=ΣₘΓᵐᵢⱼGₘₗ+ΣₘΓᵐᵢₗGⱼₘ. Add the version with i and j exchanged and subtract the version differentiated in l. Symmetry cancels the unwanted terms and leaves 2ΣₘGₗₘΓᵐᵢⱼ. Multiplying by G⁻¹ yields:</Prose>
    <MathBlock>{String.raw`\begin{aligned}\Gamma^k_{ij}&=\frac12\sum_\ell G^{k\ell}\bigl(\partial_iG_{\ell j}\\&\qquad+\partial_jG_{\ell i}-\partial_\ell G_{ij}\bigr).\end{aligned}`}</MathBlock>
    <Prose>Gᵏˡ denotes an entry of the inverse matrix, not an entrywise reciprocal. The Γ coefficients are <strong>Christoffel symbols</strong>. Their transformation includes derivatives of the changing basis, so they are not the components of a tensor. Nonzero symbols alone are not evidence of intrinsic curvature.</Prose>
    <H3>A straight line in polar coordinates</H3>
    <Prose>For G=diag(1,r²), only ∂rGθθ=2r contributes. The nonzero coefficients are Γʳ_θθ=−r and Γθ_rθ=Γθ_θr=1/r. A straight Cartesian line (t,b), with b&gt;0, has r=√(t²+b²) and θ=atan2(b,t). Direct differentiation gives:</Prose>
    <MathBlock>{String.raw`\begin{gathered}r'=\frac t r,\quad \theta'=-\frac b{r^2},\\r''=\frac{b^2}{r^3},\quad \theta''=\frac{2bt}{r^4},\\r''-r(\theta')^2=0,\\\theta''+\frac{2r'\theta'}r=0.\end{gathered}`}</MathBlock>
    <PolarConnectionLab />
    <GeometryExample example={examples.connection} />
    <Prose>The coordinate accelerations are nonzero, but the connection terms cancel them. This is the same straight, flat-plane motion described through a rotating and rescaling basis. On an embedded surface with the induced metric, projecting the ambient derivative back into the tangent space gives this Levi-Civita derivative.</Prose>
    <H3>Second derivatives need the same correction</H3>
    <Prose>The <strong>Riemannian Hessian</strong> differentiates the gradient covariantly: Hess f[v]=∇_v grad f. As a bilinear form its coordinate entries are ∂ᵢ∂ⱼf−ΣₖΓᵏᵢⱼ∂ₖf. The correction is necessary because a matrix of ordinary second partial derivatives does not itself transform tensorially under nonlinear coordinate changes.</Prose>
    <Prose>For a unit sphere and ambient extension f̄, let Pₓ=I−xxᵀ. Differentiating grad f=Pₓ∇f̄ and projecting gives Hess f[v]=Pₓ(∇²f̄ v)−(x·∇f̄)v for tangent v. The second term accounts for the changing tangent plane. Section 9 uses it to distinguish stationary minima from maxima.</Prose>

    <H2>7. Carry an arrow and detect curvature</H2>
    <Prose>Parallel transport carries an arrow along a specified path while keeping its covariant derivative zero. Metric compatibility preserves its length and inner products with other parallel arrows. It does not guarantee that going around a loop returns the same arrow. Comparing the returned vector with the initial one now makes sense because both lie in the same starting tangent space.</Prose>
    <Prose>This also explains a practical issue in optimization with momentum. A stored momentum vector belongs to the old point's tangent space. Before combining it with the new gradient, move it to the new tangent space along the chosen update route. Parallel transport is one principled choice; algorithms can use other explicitly defined vector transports, which need not preserve lengths. Simply reusing coordinate components ignores the changing basis.</Prose>
    <Prose>Use the unit sphere with N=(0,0,1), A=(1,0,0), B=(0,1,0). Start with arrow (1,0,0) at N. Along the N→A meridian, the direction of travel rotates the arrow to (0,0,−1). Along the equator A→B, this vertical arrow remains constant in ambient coordinates and stays tangent, so its derivative is zero. Along B→N it rotates to (0,1,0). The loop returns to N with a 90° turn.</Prose>
    <SphereTransportLab />
    <Prose>For non-antipodal unit endpoints x,y, parallel transport along their shorter great-circle segment has the closed form V↦V−[(V·y)/(1+x·y)](x+y). It is not a rule for every path between those endpoints. Near antipodes the denominator becomes unreliable and a route choice is necessary; the lab keeps each specified leg away from this case.</Prose>
    <GeometryExample example={examples.transport} />
    <Prose>If B has longitude φ between 0 and π, this northern triangle encloses area R²φ on a radius-R sphere. Its oriented transport turn is φ for N→A→B→N and −φ for the reverse loop, with positive orientation viewed from the outward north normal. The angle remains fixed if R changes, because curvature 1/R² and enclosed area change reciprocally. This is an exact spherical case of the curvature/holonomy relationship, not a claim that every arbitrary surface loop has this elementary formula.</Prose>
    <H3>Intrinsic curvature measures a failure of local comparisons to commute</H3>
    <Prose>Transport first in one direction and then another; compare with reversing that order around a tiny loop. The leading discrepancy is encoded by the <strong>Riemann curvature operator</strong>. We fix the convention below; some books choose its negative. This convention gives a round sphere positive sectional curvature.</Prose>
    <MathBlock>{String.raw`\begin{aligned}R(X,Y)Z&=\nabla_X\nabla_YZ\\&\quad-\nabla_Y\nabla_XZ\\&\quad-\nabla_{[X,Y]}Z.\end{aligned}`}</MathBlock>
    <Prose>The bracket correction removes the effect of noncommuting direction fields themselves. In a coordinate basis [∂u,∂v]=0. For a tangent two-plane spanned by X,Y, its sectional curvature is ⟨R(X,Y)Y,X⟩ divided by ‖X‖²‖Y‖²−⟨X,Y⟩². The denominator is the squared area of their parallelogram. On a surface there is only one tangent plane, so this scalar is its <strong>Gaussian curvature</strong> K.</Prose>
    <H3>Derive a whole family rather than memorize four labels</H3>
    <Prose>Consider a local metric ds²=du²+a(u)²dv² with a(u)&gt;0. The coefficient formula gives Γᵘᵥᵥ=−aa′ and Γᵛᵤᵥ=Γᵛᵥᵤ=a′/a. Compute the u component of R(∂u,∂v)∂v: differentiating −aa′ gives −(a′)²−aa″; the product of connection terms contributes +(a′)². Only −aa″ remains. Dividing by the plane-area factor a² gives:</Prose>
    <MathBlock>{String.raw`K=-\frac{a''(u)}{a(u)}.`}</MathBlock>
    <LessonTable caption="Different metrics tested by the same calculation" headers={['Space / coordinates','a(u)','K']} rows={[
      ['Plane in polar coordinates; u>0','u','0'],['Cylinder of radius R; local axial/angular coordinates','R','0'],
      ['Round sphere; u is meridian distance from pole','R sin(u/R)','1/R²'],
      ['Hyperbolic local chart','R exp(u/R)','−1/R²'],
    ]} />
    <Prose>A cylinder bends in three-dimensional space, but unrolling a small patch preserves intrinsic lengths and angles. Its intrinsic curvature is zero. A sphere cannot be flattened that way. A one-dimensional circle also has zero Riemann curvature because there is no tangent two-plane; its ordinary bending curvature as a plane curve is an <em>extrinsic</em> quantity. Keep those meanings separate.</Prose>
    <GeometryExample example={examples.curvature} />
    <H3>Curvature changes how nearby geodesics spread</H3>
    <Prose>For a constant-curvature surface, infinitesimally neighboring radial geodesics have transverse separation coefficient S(s) satisfying S″+KS=0, S(0)=0 and S′(0)=1. This is the radial <strong>Jacobi equation</strong>. It describes the linearized response to changing the initial direction, not an exact finite separation between arbitrary distant rays.</Prose>
    <Prose>Solving gives S=s in flat geometry, R sin(s/R) on the sphere and R sinh(s/R) in the hyperbolic plane. In a geodesic polar disk before cut points, the metric is ds²+S(s)²dφ². A radius-s circle therefore has circumference 2πS(s); integrating that circumference gives disk area.</Prose>
    <CurvatureComparisonLab />
    <Prose>In higher dimensions, different tangent two-planes can have different sectional curvatures. <strong>Ricci curvature</strong> sums appropriate sectional contributions: for a unit X, Ric(X,X)=Σᵢ⟨R(eᵢ,X)X,eᵢ⟩ over an orthonormal basis. <strong>Scalar curvature</strong> is the trace of Ricci. On a surface, Ric=Kg and scalar curvature is 2K; in dimension n with constant sectional curvature K, the scalar is n(n−1)K. These contractions summarize rather than replace the full curvature tensor.</Prose>

    <H2>8. Choose geometry for an application</H2>
    <H3>Direction retrieval and directional averages</H3>
    <Prose>Suppose embeddings and their query are normalized to unit length. Cosine similarity x·y, chord distance √(2−2x·y) and sphere distance arccos(x·y) produce the same nearest-neighbor ranking because the transforms are monotone on [−1,1]. Their numerical values differ. Without normalization, Euclidean distance also depends on magnitudes. A ranking equivalence does not make losses, averages or optimizers equivalent.</Prose>
    <Prose>A mean can minimize summed squared distances to data. For circle angles 0°,0°,90°, the normalized Euclidean vector average points at atan2(1,2)≈26.565°. Minimizing squared angular distances on their short arc instead gives 30°. The first uses ambient chord loss; the second uses intrinsic geodesic loss. For equal data at 0° and 180°, the vector average is zero and cannot be normalized, while the intrinsic squared-distance loss has two midpoint minimizers, ±90°.</Prose>
    <GeometryExample example={examples.means} />
    <Prose>An intrinsic mean, also called a Fréchet mean for the chosen squared-distance loss, may be nonunique. Curvature, spread and domain matter. A learned embedding does not itself prove that its metric recovers the physical geometry of the data; that is a modeling and validation question in manifold learning.</Prose>
    <Prose>Before choosing a geometric model, distinguish a physical constraint from a convenient representation. State the transformations and distortions the metric should respect, test conditioning near boundaries, and compare the downstream task with a suitable Euclidean baseline. Preserving distances does not establish causal structure or remove bias in the observations. Geometric deep learning also includes discrete graphs and meshes; those objects need not themselves be smooth manifolds merely because geometric methods are useful on them.</Prose>
    <H3>Positive-definite covariance matrices have more than one useful metric</H3>
    <Prose>A symmetric positive-definite matrix A satisfies vᵀAv&gt;0 for every nonzero v. Such matrices form an open convex cone in the vector space of symmetric matrices. With the Frobenius metric tr(UV), that cone is flat and its straight segment is (1−t)A+tB. Positive definiteness alone does not force a curved metric.</Prose>
    <Prose>Two other choices are common. The log-Euclidean metric makes matrix logarithms Euclidean coordinates; its path is exp((1−t)log A+t log B). The <strong>affine-invariant metric</strong> measures symmetric tangent matrices U,V at A using tr(A⁻¹UA⁻¹V). It preserves distance under invertible congruences A↦CACᵀ, such as changing linear coordinates of the measured features. Its path and distance are:</Prose>
    <MathBlock>{String.raw`\begin{gathered}C=A^{-1/2}BA^{-1/2},\\\gamma(t)=A^{1/2}C^tA^{1/2},\\d_{\rm aff}(A,B)=\|\log C\|_F.\end{gathered}`}</MathBlock>
    <Prose>For diagonal positive matrices these formulas act on the positive diagonal entries, so the affine path interpolates geometrically rather than arithmetically. Between diag(1,4) and diag(4,1), its midpoint is diag(2,2); the arithmetic midpoint is diag(2.5,2.5). Log-Euclidean and affine-invariant paths agree for this commuting example, but need not for general matrices.</Prose>
    <CovariancePathsFigure />
    <Prose>A general symmetric matrix function uses A=Q diag(λᵢ)Qᵀ and applies the scalar function to its eigenvalues, then transforms back. Entrywise square roots, powers or logarithms do not implement these formulas. The complete program supports general small SPD matrices; the printed example is diagonal so you can check the result by hand.</Prose>
    <GeometryExample example={examples.spd} />
    <H3>Fisher geometry gives a statistical distance, with a finite-distance boundary</H3>
    <Prose>For a regular identifiable probability model, the Fisher metric is the expected outer product of score derivatives. It is positive definite only when there are no locally unidentifiable directions. Earlier Second-Order Methods derives its optimizer role. Here we examine Bernoulli geometry with 0&lt;p&lt;1.</Prose>
    <Prose>The score is 1/p for an observed 1 and −1/(1−p) for an observed 0. Its squared expectation is p/p²+(1−p)/(1−p)²=1/[p(1−p)]. Hence ds²=dp²/[p(1−p)]. Integrating du/dp=1/√[p(1−p)] gives u=2 asin√p. In u coordinates the metric is du² on the open interval (0,π).</Prose>
    <MathBlock>{String.raw`\begin{gathered}u(p)=2\arcsin\sqrt p,\\d_F(p,q)=|u(p)-u(q)|.\end{gathered}`}</MathBlock>
    <Prose>For a fixed p, the distance to q as q tends to zero approaches u(p), a finite value. The metric coefficient diverges, but its square root is integrable. This regular Bernoulli manifold is not complete. Adding degenerate distributions changes statistical regularity assumptions; it is not a coordinate relabeling.</Prose>
    <Prose>For comparison, the hyperbolic upper half-plane has metric (dx²+dy²)/y² for y&gt;0. Along a vertical geodesic, distance is |log(y₂/y₁)|, so y=0 is infinitely far away. A visible coordinate boundary can have a very different metric meaning. Hyperbolic embeddings may encode hierarchical structure, but this calculation is no claim about predictive performance.</Prose>
    <GeometryExample example={examples.fisher} />
    <H3>Combine the gradient with volume to get a diffusion operator</H3>
    <Prose>We now have a way to measure change and a way to measure volume. The metric divergence of a vector field V is its local outward flow per volume. With coordinate volume density √g, where g=det G, div V=(1/√g)Σᵢ∂ᵢ(√g Vⁱ). The factor counts how much physical volume a coordinate cell represents.</Prose>
    <MathBlock>{String.raw`\begin{aligned}\Delta f&=\operatorname{div}(\operatorname{grad}f)\\&=\frac1{\sqrt g}\sum_{i,j}\partial_i\!\left(\sqrt g\,G^{ij}\partial_jf\right).\end{aligned}`}</MathBlock>
    <Prose>This is the <strong>Laplace–Beltrami operator</strong>, the manifold version of the Euclidean Laplacian. For the round sphere and f=cosθ, the gradient coefficient in θ is −sinθ/R². Multiplying by volume R²sinθ gives −sin²θ; differentiating and dividing by volume gives Δf=−2cosθ/R². Height is an eigenfunction. This Δ convention is nonpositive on a closed manifold; spectral contexts often use −Δ instead.</Prose>
    <GeometryExample example={examples.laplacian} />
    <Prose>The formula connects geometry with heat flow, smoothing and spectral methods: ∂t f=Δf diffuses with respect to the chosen metric and volume. Boundary conditions are additional data on a domain with boundary. Full PDE theory and differential-forms versions of divergence/Stokes belong to longer follow-on treatments.</Prose>

    <H2>9. Complete a constrained optimization</H2>
    <Prose>Assemble a usable workflow: given a symmetric matrix A, minimize f(x)=xᵀAx subject to ‖x‖=1. This Rayleigh quotient selects an extremal direction in quadratic models and eigenspace problems. The ambient gradient is 2Ax. Projecting gives grad f=2(Ax−fx). A zero gradient means Ax=fx, so every eigenvector is stationary, not only the minimum one.</Prose>
    <Prose>For tangent v, the sphere Hessian gives Hess f[v]=2(PₓAv−fv). At a unit eigenvector with eigenvalue λ, a tangent unit eigenvector for another eigenvalue μ gives curvature 2(μ−λ) of the objective. At the smallest eigenvalue these quantities are nonnegative; at the largest they are nonpositive; intermediate eigenvectors can be saddles. This is curvature of the <em>objective along the sphere</em>, distinct from the sphere's own Gaussian curvature.</Prose>
    <Prose>Choose a negative-gradient tangent step and normalize the candidate. Use backtracking: start with step one, halve it until f(new)≤f(old)−10⁻⁴ step ‖grad f‖². This Armijo condition asks for a definite fraction of predicted first-order decrease. Smoothness and a nonzero gradient ensure sufficiently small steps work in exact arithmetic locally. The finite program also has a line-search limit and iteration limit, because finite work and floating-point resolution cannot be replaced by that theorem.</Prose>
    <GeometryExample example={examples.optimize} />
    <Prose>The changed matrix is Q diag(2,5,9)Qᵀ. Its objective reaches 2 at printed precision, but the gradient norm is about 3.05×10⁻⁸, above the requested 10⁻⁸. This run reports <Code>line-search limit</Code>: the required further decrease is too small for reliable floating-point resolution. This is a useful status, not an error to relabel as convergence. A looser justified tolerance can give a stationarity result; stationarity still is not a universal minimum certificate. The first fixture separately prints its 8.641×10⁻⁹ gradient norm and zero feasibility residual at the displayed precision.</Prose>
    <Prose>To check a gradient, choose tangent v and compare [f(Expₓ(hv))−f(Expₓ(−hv))]/(2h) with grad f·v over several moderate h values. The central-difference discrepancy initially falls quadratically, then rounding can dominate. Verification uses this changed-direction check; one agreement at one h would be weaker evidence.</Prose>
    <Callout label="A complete result includes its limits">Report the manifold and metric, objective, starting state, update map, stopping rule, feasibility residual, gradient norm and independent reference or certificate. A plotted path on a sphere is not evidence that the right gradient was used or that a global minimum was found.</Callout>

    <H2>10. Practise the connections</H2>
    <Prose>Work these with explanations hidden. Each changes an input or asks you to transfer the mechanism. Re-running a worked program unchanged is useful revision, but does not answer these tasks.</Prose>
    <Practice question="A point has circle angle 315°. Find both chart labels, their transition and a unit tangent vector. Which label survives at angle 180°?" hint="Subtract 360° to place α in (−180°,180°); differentiate (cos θ,sin θ).">
      <Prose>α=−45°=−π/4 and β=315°=7π/4, so β−α=2π. The point is (√2/2,−√2/2) and an increasing-angle unit tangent is (√2/2,√2/2). Their dot product is zero. At 180° the α chart excludes the point, but β=π is valid.</Prose>
    </Practice>
    <Practice question="At x=(0,3,0) on the radius-three sphere, project a=(2,5,−1) to the induced tangent space. Why is v=(1,0,2) tangent?" hint="Use a−(x·a)x/R²; do not use the unit-sphere denominator by accident.">
      <Prose>x·a=15 and R²=9. The removed vector is (0,5,0), leaving (2,0,−1). Since x·v=0, v is tangent. Both gradients give the same directional derivative a·v=0 in this direction, but the projected gradient also satisfies the constraint for every direction.</Prose>
    </Practice>
    <Practice question="Use x=u+2v, y=v with metric dx²+4dy² and f=2x−y. Find G, differential coefficients and the coordinate/physical gradients." hint="Form Sᵀdiag(1,4)S, then solve Gg=a.">
      <Prose>G=[[1,2],[2,8]], a=(2,3), and g=(2.5,−0.25). Transforming gives (2.5+2·(−0.25),−0.25)=(2,−0.25). The shear changes components; the physical y-cost explains the smaller y-component.</Prose>
    </Practice>
    <Practice question="On a sphere of radius two, compare the full latitude circumference at θ=30° with the equator, and compute the area fraction of the band θ∈[30°,60°]." hint="Use ds=R sinθ dφ along latitude and integrate the area element for the band.">
      <Prose>The latitude length is 2πR sin30°=2π, half the equator's 4π. The band fraction is (cos30°−cos60°)/2=(√3−1)/4≈0.183013. The latitude is generally not a geodesic; the equator is. A short curve in coordinate angle need not be the shortest permitted path between endpoints.</Prose>
    </Practice>
    <Practice question="At x=(1,0,0), follow tangent v=(0,3π/2,0). What does Exp return, what shortest Log returns, and why do they differ?" hint="Evaluate sine/cosine at 3π/2 and choose the shorter arc to that endpoint.">
      <Prose>Expₓ(v)=(0,−1,0). Its shortest Log is (0,−π/2,0), not v. The original geodesic went three quarters of a circle; the shortest route goes one quarter in the other direction. At length π the endpoint is antipodal and the shortest Log nonunique.</Prose>
    </Practice>
    <Practice question="For the straight line (t,2), compute polar coordinate accelerations and their corrections at t=0. Does nonzero radial acceleration establish curvature?" hint="Here r=2, r′=0 and θ′=−1/2.">
      <Prose>r″=b²/r³=1/2 and its correction −r(θ′)²=−1/2. Both θ″ and 2r′θ′/r are zero. Covariant acceleration is zero; the plane is flat. A changing coordinate component is not a measurement of intrinsic curvature.</Prose>
    </Practice>
    <Practice question="Use the northern wedge loop with longitude 60° on a radius-two sphere. Find its area, curvature and final turn, then reverse the loop." hint="The triangle area is R²φ; positive orientation is N→A→B→N.">
      <Prose>φ=π/3, area=4π/3 and K=1/4. Their product is π/3, giving a +60° turn; the reverse gives −60°. Transport along a route followed exactly backward is the inverse map and returns every arrow unchanged.</Prose>
    </Practice>
    <Practice question="For ds²=du²+(1+u²)²dv², compute K and scalar curvature at u=0. Is nonzero a′ necessary for curvature?" hint="Use a=1+u², a″=2 and the two-dimensional scalar=2K relation.">
      <Prose>K=−2/(1+u²), so at zero it is −2 and scalar curvature is −4. Here a′=0 but a″≠0, so curvature remains nonzero. Curvature depends on local metric variation, not one connection coefficient alone.</Prose>
    </Practice>
    <Practice question="Compare the arithmetic and affine-invariant midpoint of A=diag(1,9), B=diag(4,1). Why should the result not be computed by taking entrywise powers of general matrices?" hint="For this commuting pair, interpolate each positive diagonal entry geometrically.">
      <Prose>The arithmetic midpoint is diag(2.5,5); the affine midpoint is diag(2,3). Distance is √[(log4)²+(log(1/9))²]. General matrices require eigenvalue-based matrix functions and surrounding A±1/2 factors; their entries are not independent scalar coordinates for these operations.</Prose>
    </Practice>
    <Practice question="In Bernoulli Fisher geometry, how far is p=1/4 from the boundary p→0? Is the boundary a point of the regular manifold?" hint="Transform to u=2 asin√p.">
      <Prose>The limiting distance is π/3. The regular manifold contains only 0&lt;p&lt;1, so the boundary is missing although its distance is finite. A diverging metric coefficient by itself does not establish infinite distance.</Prose>
    </Practice>
    <Practice question="Modify the optimizer: use the changed rotated matrix, start at its largest eigenvector (0,0,1), then at (1,2,3) with tolerance 10⁻⁶. Predict both outcomes and propose evidence stronger than a small gradient." hint="Stationarity holds at every eigenvector. The independent eigenvalues are 2,5,9.">
      <Prose>The largest-eigenvector start reports stationary with value 9. The mixed start reaches a value near 2 with the looser gradient tolerance in the checked run. Compare its Rayleigh value to 2 and inspect feasibility and gradient residuals. At a stationary eigenvector, Hessian eigenvalues on tangent eigendirections are twice the differences from its eigenvalue; at the maximum they are negative. The included changed program with the tighter default records its finite line-search limitation instead of silently accepting it.</Prose>
    </Practice>
    <Prose><strong>Ready to move on:</strong> you can distinguish point from chart, differential from metric gradient, valid step from minimizing step, connection coefficients from curvature, and stationarity from optimality. You can derive a metric from a parameterization, check a constrained update, explain transport's path dependence, and state the geometry behind a distance or average.</Prose>

    <H2>11. Continue and choose another learning route</H2>
    <Prose>The next entry in this module is <strong>Algebra, Functions, Exponentials &amp; Logarithms</strong>. The current catalogue starts a foundational sequence there; publication status does not change that order. Review <a href="/learn/path/full-curriculum/tensor-algebra-einsum-notation?module=math-foundations">Tensor Algebra</a> for vector/covector components, <a href="/learn/path/full-curriculum/multivariate-calculus-gradients?module=math-foundations">Multivariate Calculus</a> for Jacobians and directional derivatives, and <a href="/learn/path/full-curriculum/topology-topological-data-analysis-tda?module=math-foundations">Topology/TDA</a> for neighborhoods and continuity when those are the actual hurdles.</Prose>
    <Prose>For deeper applications, existing owners include Second-Order Methods for natural-gradient algorithms, t-SNE/UMAP/Manifold Learning for learned representations, Coordinate Frames/Transformations for robot states, and geometric deep learning for architecture choices. These local tools do not establish that every embedding, neural state space or constrained parameter set is automatically a smooth manifold with a uniquely correct metric.</Prose>
    <Sources alternatives={<Prose>For another route, use Boumal's optimization-first course after sections 1–5, Tong's coordinate/tensor explanations alongside sections 2 and 6, or MIT's curves-and-surfaces exercises after section 7. Videos supplement the local derivations; no full-video viewing or exact timestamp endorsement is implied.</Prose>}>
      <li><a href="https://www.nicolasboumal.net/book/">Nicolas Boumal, An Introduction to Optimization on Smooth Manifolds</a> — author-hosted book, exercises and recorded EPFL course links. Selected gradient, sphere, retraction and transport sections were inspected; the course page was checked as an alternate route.</li>
      <li><a href="https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf">Boumal's freely available prepublication text</a> — sections 3.7–3.8, 4.5, 7.2, 10.2–10.3 and 11.7 support geometric and optimization conventions. Section numbers agree with the published book; PDF page numbering differs. Useful for proofs and general SPD formulas after the examples.</li>
      <li><a href="https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S2.html">David Tong, Differential Geometry</a> — charts, tangent vectors and one-forms; selected sections 2.1–2.3 inspected. A physics-oriented written alternative with more formal coordinate notation.</li>
      <li><a href="https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S3.html">Tong, Riemannian Geometry</a> — connections, metric compatibility, curvature and geodesic deviation. Selected sections 3.2–3.3 inspected. The broader notes discuss Lorentzian geometry; this lesson's length and positive-definite assumptions are Riemannian.</li>
      <li><a href="https://ocw.mit.edu/courses/8-962-general-relativity-spring-2020/resources/lecture-7-the-principle-of-equivalence-continued-parallel-transport/">MIT 8.962, Lecture 7: Parallel Transport, Scott Hughes</a> — official video resource and description checked. An optional graduate-physics explanation; expect additional spacetime background.</li>
      <li><a href="https://ocw.mit.edu/courses/18-950-differential-geometry-fall-2008/">MIT 18.950 Differential Geometry, Paul Seidel</a> — official course with written notes and problem sets for curves and surfaces. The course description and available resource types were checked; this is not a claimed video playlist.</li>
    </Sources>
  </div>,
};
