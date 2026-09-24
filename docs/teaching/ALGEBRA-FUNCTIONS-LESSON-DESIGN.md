# Algebra, Functions, Exponentials & Logarithms — design

Status: implemented and author-verified; independent review/integration remain parent-owned. Stable ID `algebra-functions-exponentials-logarithms`, mathematics position 43, newly authored publication. No previous lesson/program/output existed to conserve. Parent owns registration, catalogue and integration. Next route topic is Sets, Logic, Relations & Proof Techniques; no order change. Final checks and exact source identity are recorded in [verification](ALGEBRA-FUNCTIONS-VERIFICATION.md) and its durable evidence packet.

## Learning contract and scope

Retain the title: algebraic language and function structure are the foundations that make exponentials and logarithms usable. No required topic prerequisite. A reader needs counting and ordinary addition; signed quantities, multiplication, fractions, percentages and notation are refreshed locally. Python is optional and introduced before the first program, using Python 3.12+ standard library only. The core cannot require the preceding advanced module topics or the later Sets/Calculus lessons.

The anchor is a small measurement system: an initial quantity, additions or repeated multipliers, a conversion rule, and a threshold decision. Other examples add distinct meaning: calibration order, a square-area ambiguity, a quadratic feasible interval, cumulative retry delays, and floating-point cancellation. These are declared exact toy models, not measured hardware or biological laws.

Beginner finish line: translate a story, evaluate and check equations, read an input/output point and know what a logarithm asks. Intermediate finish line: carry domain restrictions, invert valid branches, choose additive versus multiplicative models, derive a threshold and distinguish its continuous crossing from the first discrete observation. Deeper branches: complete the square, geometric sums, real-exponent extension and continuous-rate convention, and one small stable-computation example. No calculus proof, abstract algebra, full polynomial algorithms or statistical fitting course is promised.

## Exact inventory and ownership review

Ran `node scripts/build-curriculum-inventory.mjs --topic "algebra-functions-exponentials-logarithms"` before design on 2026-09-11 local date. Returned existing scoped brief has no prerequisites and outcomes for units/composition/log inversion. There is no destination note. Read UNASSIGNED inbox: bit-mask/interview proposal belongs to programming/DSA and is not inserted here.

| Idea | Scoped evidence / best owner | Decision |
| --- | --- | --- |
| Signed arithmetic, fractions, distribution, linear equations and inequalities | Current title promises algebra; no required prior lesson | Teach necessary mechanisms and changed checks locally. |
| Quantifiers, equivalence, relations and proofs | Next Sets/Logic44 brief explicitly owns them | Explain reversible equation steps informally here; route examples of implication versus equivalence for next author. |
| Coordinates, slope and basic graphs | Needed here to interpret a function; Geometry45 owns angles, frames and rotations | Teach two labeled axes, pairs and slope locally; leave geometric transformations and trigonometry there. |
| Quadratics and rational exclusions | Needed to explain multi-valued inverse candidates, roots and domain | Teach factoring and completing the square, a real quadratic formula and a hole; no polynomial division catalogue. |
| Derivatives, limits and the rigorous construction of exp | Calculus47 and Real Analysis later briefs own them | Define continuous growth convention with a finite subdivision table, state limit without presenting finite data as proof; route exact limit/derivative bridge. |
| Complex logarithm and fractional powers of negative bases | Complex Numbers/Fourier/Laplace later owns complex conventions | State positive-base real laws here; contrast integer and odd-root cases without importing complex analysis. |
| Cancellation and robust threshold decisions | Conditioning/Stability brief owns broad numerical analysis | One complete `log1p`/`expm1` demonstration and exact integer doubling decision here; broad error analysis later. |
| Entropy and algorithmic log counts | Their current lessons already use log bases/branch counts | Short accessible bit-choice/doubling interpretation, not a new entropy or complexity course. |

## Concept and visual map

| Hurdle | Mechanism and worked anchor | Visible representation | Learner evidence |
| --- | --- | --- | --- |
| A quantity is not just a numeral | 3/4 litre + 1/2 litre uses common eighths/quarters; rates multiply time | Inline fraction strip, signed number-line reflection | Changed fractions and a negative-inequality counterexample |
| Expression versus equation | Fixed 6-unit fee plus 3 per item; substitute, distribute, collect like terms | Paired grouping table and multiline distribution derivation with named coefficient/variable terms | Parenthesis/sign diagnosis and checked units |
| What solving preserves | Subtract the same b, then divide by nonzero a; a=0 gives all/none | Equation ledger with both sides, step/back/reset and exact solution type | Predict zero coefficient before progressing; changed inputs |
| One output for each allowed input | Affine, square and reciprocal rules; excluded zero and repeated output | Linked graph/input/output probe, visible domain/branch selection, square preimages | Read table and distinguish function from one-to-one |
| Composition and inverse | 2x+1 then square versus reverse order; square branch loses sign | Two ordered machine paths with actual intermediate values | Reverse affine calibration, state valid inverse domain |
| Equivalent forms reveal different facts | x²−6x+5=(x−3)²−4=(x−1)(x−5) | Movable parabola, vertex/root markers and an immediately visible completed-square derivation | Changed discriminant/feasible interval; original rational exclusion |
| Multiplicative change | 100+20t versus 100(1.2)^t | Linked graph scale toggle and table of differences/ratios | Predict next value and judge stated model boundary |
| Logs recover an exponent | 2³=8; fractional threshold log(3)/log(1.2) | Log ruler linking quantities 1,2,4,8 to equal exponent gaps; same growth plot changes coordinates | Explain product law and reject log(a+b)=log a+log b |
| Discrete decision versus continuous crossing | First whole doubling reaching a storage target, then sum all retry delays | Step table and visibly distinct cumulative sum | Changed exact integer test and independent capstone |

### Interactive contracts

1. **EquationStepsLab**: initial 3x+6=21. Draft a,b,c restricted to integers from −30 through30; explicit Apply validates all before replacing active state and resets step. Both sides change together when stepping. a=0 resolves to all real x or no solution without dividing by zero. Back/reset preserve the active equation; named presets cover negative, identity and contradiction. Text states operation and solution check. No balance-weight picture implies negative physical masses.
2. **FunctionProbeLab**: initial square at x=2. Input slider/domain selector updates immediately. Sampled curve is the selected formula, with axis labels and actual point; reciprocal split at zero so no connecting asymptote. At excluded input show no output, not zero. Square restricted to x≥0 makes inverse choice visible; annotations separate allowed curve, queried point and alternative preimages. Actual coordinates available as text.
3. **CompositionLab**: input x=2; two orderings of affine and square with intermediate and final values. Affine inverse pipeline operates in reverse order and identifies zero-slope noninvertibility if supported. Controls have exact meaning; no uncontrolled animation.
4. **QuadraticLab**: y=(x−h)²+k, initially h=3,k=−4. Move vertex; same curve is expressed in expanded form. At k<0 two roots, k=0 one repeated root, k>0 no real root; exact root formula remains beside rounded coordinates. Graph is calculated with clipping/labels, not a proof of arbitrary polynomial root count.
5. **GrowthScaleLab**: fixed initial100, additive20 per period; selectable repeated multiplier from positive bounded rates, integer query t0..8. Linear/log coordinate toggle changes axis positions only; table and exact rule unchanged. Target relative factor must be positive and supported; show log crossing and direct substitution, distinguish model interpolation from observed steps. Zero/negative quantities never enter ordinary log axis. Decay allowed and threshold direction explained.
6. **LogRulerLab**: select base2/10/0.5 and positive value by exponent slider; compare linear ratio and signed exponent displacement. Descending base reverses ordering explicitly. Fixed products illustrate sum of exponent positions; units are positive ratios to a named reference. Geometry gives exponent location, not physical length.

Inline forms: signed reflection/fraction pieces; expression groups; quadratic completion/omitted rational point; logarithm product-to-sum derivation. Static/initial representations appear where concepts are introduced. Phone labels remain readable, axes cannot lose units; simple curves fit 280px content while wide tables/code use local keyboard scrolling. All state changes visible in text as well as color.

## Complete examples and independent assessment

Plan standard-library programs for exact fractional measurement/equation classification; validated function composition/inverse; quadratic roots checked in original equation; integer and rational exponent conventions; growth and log inversion; exact doubling/cumulative retries; subdivision toward e; cancellation-aware `log1p` and `expm1`. Keep each file executable alone. Bind exact stored source and stdout to browser assertions. Use optional code to check reasoning, never as an unexplained substitute for hand calculations.

Independent tasks vary numbers and sometimes assumptions: grouped signed arithmetic; fee/integer feasibility; negative inequality; rational cancellation and extraneous square root; restricted composition/inverse; quadratic inequality; percentage undoing; logs with domain rejection; half-life; log-axis interpretation; exact doubling/cumulative budget; capstone comparing a fixed-addition forecast and growth forecast with verification. Hints precede explained solutions, including a tempting wrong route. Every specifically requested numerical modification has a checkable result; open tasks have an example accepted answer.

## Research and review bounds

2026-09-11: opened OpenStax Algebra and Trigonometry 2e sections1.1,3.1,3.4,3.7,6.5 and the indexed6.3 explanation, plus Elementary Algebra2e2.7 for negative inequalities. Reviewed relevant written definitions, domains and worked reasoning; these supply authoritative convention checks, not wording or a copied page structure. Additional polynomial/growth sections will be checked as claims are finalized. Primary Python3.12 math documentation was opened for `log`, `log1p`, `expm1`, domain exceptions. Runtime verification will use local Python3.12.14; no installed-library requirement.

Alternate video: 3Blue1Brown, “Logarithm Fundamentals | Ep.6 Lockdown live math”, direct https://www.youtube.com/watch?v=cEvgcoyZvB4 . Creator/title/date/description were retrieved through indexed official video metadata (2020-05-05); direct open returned cache error. No playback or transcript inspected, no timestamp guidance or detailed video-coverage claim. The annotation will identify the basic log topic and honest review bound. Primary written resources are sufficient for every technical claim.

## Verification plan and closure

Independently verify model grids with exact Fraction equations, polynomial expansion/residuals, Decimal growth/log or direct multiplication, explicit input/output pairs and composition, rejection of NaN/Infinity/empty drafts. Compare program answers to changed inputs and independent enumeration rather than reasserting the same sample. Check original denominator/domain after candidate-generating operations. Model boundary contracts must reject unsupported finite arithmetic rather than display Infinity as a real solution.

Actual browser at1440/390/320 with public Space Grotesk: every control, back/reset/preset, invalid input retention, keyboard sliders/buttons, inverse restriction and reciprocal hole, negative exponent/log-base order, graph/table agreement, all displayed equations and all actual runnable source/output/questions, all practice answers. Open ordinary-reading and meaningful changed-state screenshots; errors and overflow checks supplement actual visual inspection. Parent will independently review the source and run integrated build/route checks; author evidence will not claim that integration or user review happened.

## Implementation decisions and source checks

The equation model accepts integer coefficients from−30 to30 (so the initial right side21 is inside its contract). Probe/composition inputs and log exponents use the actual quarter-step controls; quadratic coordinates use half-steps, and growth rates use the displayed discrete choices. Out-of-grid exported calls are explicitly rejected, including tiny nonzero values that could underflow a square or overflow an inverse-growth time. These guards do not restrict the mathematical function domains taught in prose. The graph is a sampled view of those mathematical rules; viewing windows and arithmetic approximations are identified separately.

The grouping comparison table plus complete distribution derivation proved more direct than an additional operation tree. The completed-square derivation and movable root geometry replace the initially considered square-tile picture. Those are deliberate choices about the explanation, not omitted required mechanisms. Final fractional powers use actual rendered roots and a multiline example instead of ASCII exponent text. Two wide derivations were split for320px reading. Source formatting conserves normalized AST after the intentional copy edits, with escaped JSX text preserved.

| Claim / convention | Primary locator | Reviewed scope and limit |
| --- | --- | --- |
| Arithmetic grouping and distribution | OpenStax Algebra/Trig2e1.1, order/properties headings | Written definitions and sample operations; own fraction/measurement examples. |
| Negative inequality reversal | OpenStax Elementary Algebra2e2.7 | Sign reversal and interval solutions; own reflection and changed inequality. |
| Function/domain/composed-domain/inverse | OpenStax Algebra/Trig2e3.1,3.4 domain-composition,3.7 domain/range | Written definitions, restrictions and inverse correspondence; one-to-one versus function explicitly separated. |
| Completing squares and root count | OpenStax5.1 quadratic forms | Written vertex/expanded form and real roots; own general discriminant derivation checked algebraically and numerically. |
| Positive exponential models and e motivation | OpenStax6.1 exponential functions | Positive-base/growth convention and continuous-compounding context; finite sequence evidence is not called a proof. |
| Log laws, base changes and domains | OpenStax6.3 and6.5 product/quotient/power headings | Written inverse definition and law derivations; negative-input power-law trap resolved with absolute value. |
| Stable small changes | Python3.12 math documentation, expm1/log1p/cbrt entries | Primary API contracts checked; all actual programs executed on3.12.14; higher-precision Decimal reference. |
| Optional visual log resource | Official3Blue1Brown video indexed metadata, cEvgcoyZvB4 | Title/creator/description/date verified; direct-open cache error, no playback or transcript. Learner annotation states review limit. |

All claim research occurred on2026-09-11 local date. No outside wording, figure or program was copied. Destination discoveries were saved immediately to Sets/Logic and Single-Variable Calculus notes; no unrelated body was edited. The Sets author has received the bridge directly. Those destination proposals remain subject to the receiving author's reasoned assessment.
