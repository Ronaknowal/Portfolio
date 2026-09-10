# Tensor Algebra & Einsum Notation — design and research

10 September 2026. Implementation and numerical/browser author verification are complete; [the verification record](TENSOR-ALGEBRA-EINSUM-VERIFICATION.md) records actual evidence and limits. Publication predates this review and is not itself completion evidence. Root integration and user acceptance are separate.

## Scope, continuity and decisions

Exact topic-plan command run for `tensor-algebra-einsum-notation`; both Vectors destination notes read. Relevant inbox contains no tensor instruction. Stable title and module position five retained; next is Randomized Linear Algebra. Required skill is indexed vector/matrix arithmetic, refreshed in place. Basis and matrix calculus connections are reading bridges, not new prerequisite cycles.

Original coverage retained: axis roles, matrix multiplication with original A/B values, trace/column sums/outer product, attention's two contractions, shape-valid typo risks, ellipsis, implicit-order risks, contraction costs and batched covariance practice. Original shortcomings: repeated-label rule conflated explicit and implicit mode; attention code omitted data/softmax/scaling; covariance prompt omitted centering/denominator; no independent practice solutions, figures or execution model. Repair these without deleting depth.

Core route: name objects/axes → compute one output entry → explicit output and diagonal rules → batches and ellipses → covariance and complete attention → execution order → numerical/semantic checks. A contained deeper branch derives vector, functional, map and metric transformation laws in a finite-dimensional real space; a library axis alone does not specify these laws. Full tensor fields, manifold calculus and connection coefficients remain outside this lesson.

| Hurdle | Representation and action | Independent practice |
| --- | --- | --- |
| A string hides loops | Workbench shows selected output coordinates, every product and source cell | Derive a new non-square contraction and a reduction |
| Repetition is mistaken for summation | Same matrix's highlighted diagonal followed by optional sum; explicit/implicit comparison | Explain retained diagonal and a once-occurring reduced label |
| Batch pairing mistaken for contraction | Paired versus all-pairs batch diagram, then selectable attention query/mask | Repair valid but wrong batch labels |
| Scores, normalized weights and weighted values confused | Per-key dot products → scaled scores → normalized bars → contributions | Calculate a new weighted result and handle no allowed key |
| Equation mistaken for execution schedule | Two intermediate trees and exact multiply counts under editable dimensions | Find an order whose first intermediate and cost are smaller |
| Components mistaken for object | Same world arrow in changed basis, coordinate/functional/metric checks | Preserve a new linear measurement after shear |

All diagrams use declared invented arrays or exact formulas. Counts are scalar multiplications for conventional dense products, not measured time, NumPy FLOP reports or whole-process peak memory. Workbench is a bounded explicit, one/two-operand interpreter, not arbitrary NumPy execution. Attention is real, finite, tiny, untrained scaled dot-product arithmetic; all-masked rows need an explicit policy and return a teaching error. Basis transformations assume a fixed real vector space and invertible S with new basis columns in old coordinates.

## Research ledger

- NumPy stable `einsum`, Notes/Parameters/Examples, substantive text inspected 10 September 2026; page identifies v2.5, local runtime 2.3.5. Verified explicit output, repeated labels within operands, alphabetical implicit order, ellipsis, writeable views, dtype/casting and optimize default. Singleton semantics are tested locally rather than inferred from the documentation's shorthand that broadcasting is not enabled by default.
- NumPy stable `einsum_path`, Parameters/Notes, substantive text inspected same date. Path indexing, greedy versus combinatorial optimal search, intermediate size limits. No published timing copied or treated as a benchmark for this site.
- Vaswani et al., *Attention Is All You Need*, section3.2.1–3.2.3, original paper HTML inspected: scaled dot products, distinct key/value feature sizes, weighted values and masking. Lesson implements these equations on invented arrays; no training quality claim.
- Independent final review required a piecewise masked-weight equation, not merely an allowed-key denominator; both formula and lab readout now explicitly assign zero to blocked keys. The numerical model and native program already implemented that mask correctly. Final evidence includes the literal counterexample and corrected mobile equation captures.
- MIT8.962 Lecture3 official description and transcript PDF inspected for tensor/dual/contraction meaning: transcript pages1–8 and14–15, including multilinear slots, basis-component distinction, one-forms and invariant pairing. The official MIT OpenCourseWare YouTube result verifies direct recording `https://www.youtube.com/watch?v=H6eR3sG524M`; full recording not watched and no timestamp recommendation invented. Graduate relativity context exceeds this lesson; our finite-dimensional basis example is independently derived and checked. Transcript artifacts and its Lorentz-specific claims are not adopted as general Euclidean formulas.
- BYU ACME Advanced NumPy, Einsteinian Summation through Common Operations inspected as an alternate explanation. Its later unqualified performance claims are not adopted. Recommend only the notation section, with the current NumPy reference supplying API contracts.
- NumPy `cov`, Parameters/Notes inspected for rowvar, centering and denominator; local examples use ddof=1, while unbiasedness is qualified with an iid finite-second-moment sampling model.
- OpenStax University Physics1 section7.2 inspected for classical particle/system kinetic energy and joule units. The application supplies invented masses and Cartesian velocities, not a measured simulation or a relativistic energy model.

## Scoped discovery disposition

The two Vectors notes are implemented: diagonal/reduction distinctions occupy core section3 and its inspector/figure; coordinate meaning becomes section8 with a contained mathematical branch. No title change is needed. A further [Differential Geometry destination note](topic-notes/differential-geometry-riemannian-manifolds.md) records how differential/covector coefficients become gradient vectors through a metric and why chart and ambient projection conventions must remain explicit. No other topic source was rewritten.

## Completion gate

Completed independently executed programs/oracles, visual states, desktop/mobile keyboard and ordinary-reading screenshots are recorded in `TENSOR-ALGEBRA-EINSUM-VERIFICATION.md`. Both origin notes are resolved with implementation/evidence links; the further differential-geometric extension has its own reasoned destination note. No claim of complete tensor analysis or observed beginner study.
