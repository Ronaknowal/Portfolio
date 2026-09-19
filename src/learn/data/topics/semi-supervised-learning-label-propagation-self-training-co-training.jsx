// Static presentation of the prepared manuscript; regeneration is topic-scoped.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';
import { DataTable, LabelLedgerFigure, SameInputsFigure, HardFlowFigure, EvidenceFigure, PrototypeFlowFigure, PairedViewsFigure, PromotionAuditFigure } from '../../components/lesson-labs/SemiSupervisedFigures.jsx';
import { LabelPropagationLab, SelfTrainingLab, CoTrainingLab } from '../../components/lesson-labs/SemiSupervisedLabs.jsx';
import { semiSupervisedExamples } from '../semi-supervised-examples.js';

const semiSupervisedLesson = {
  title: 'Semi-Supervised Learning (Label Propagation, Self-Training, Co-Training)',
  readTime: '~60 min read + 90 min practice',
  hasIntegratedGuide: true,
  content: () => <div className="ssl-lesson">
<LessonIntro prerequisites="Weighted averages, basic classification and train/development/test separation. The lesson refreshes its matrix and probability notation locally." sections={[["1-what-information-is-actually-available","1. What information is actually available?"],["2-label-propagation-let-neighboring-examples-constrain-one-another","2. Label propagation: let neighboring examples constrain one another"],["3-label-spreading-soften-the-anchors-without-confusing-the-normalization","3. Label spreading: soften the anchors without confusing the normalization"],["4-self-training-make-a-guess-record-its-origin-then-refit","4. Self-training: make a guess, record its origin, then refit"],["5-co-training-let-another-representation-provide-the-label","5. Co-training: let another representation provide the label"],["6-a-real-experiment-where-the-supervised-baseline-wins","6. A real experiment where the supervised baseline wins"],["7-choose-the-method-by-the-information-it-can-use","7. Choose the method by the information it can use"],["8-optional-depth-what-connects-these-methods-to-later-learning-systems","8. Optional depth: what connects these methods to later learning systems?"],["9-practice-produce-a-decision-and-explain-its-evidence","9. Practice: produce a decision and explain its evidence"],["10-another-route-through-the-subject","10. Another route through the subject"]]}>Use a few observed labels without mistaking every model guess for new evidence. Follow sections 1–7, then practice; the deeper connections are an optional second pass.</LessonIntro>
<Prose>{"You have collected hundreds of documents, but a person has classified only a handful. Reading more documents is easy for the computer. Knowing which category each document belongs to is the expensive part. Can the documents without labels still help you build a classifier?"}</Prose>

<Prose>{"Sometimes. The extra documents reveal vocabulary, recurring patterns and neighborhoods. They do not tell you what those patterns mean. Semi-supervised learning uses "}<strong>{"labeled examples together with unlabeled inputs"}</strong>{", connecting the two through an assumption about the task."}</Prose>

<Prose>{"By the end of this lesson, you should be able to trace a label through a graph, distinguish a human label from a model's guess, run a small experiment that can reject semi-supervised learning, and explain when two representations can teach each other."}</Prose>

<Prose>{"The core route is sections 1–7, followed by practice. Section 8 connects these mechanisms to deeper methods; it is an optional second pass. Refreshers on weighted averages, probabilities and matrices appear where they are needed."}</Prose>

<H2>{"1. What information is actually available?"}</H2>

<Prose>{"Write the observed examples as "}<InlineMath>{"L=\\{(x_i,y_i)\\}"}</InlineMath>{" and the unlabeled pool as "}<InlineMath>{"U=\\{x_j\\}"}</InlineMath>{". An input "}<InlineMath>{"x"}</InlineMath>{" might be a feature vector; its label "}<InlineMath>{"y"}</InlineMath>{" is the category to predict. “Unlabeled” means that the learning procedure cannot access its target, even when we retain that target separately to audit a teaching experiment."}</Prose>

<Prose>{"A probabilistic classifier estimates "}<InlineMath>{"p(y=c\\mid x)"}</InlineMath>{", a distribution over the possible classes. In binary classification, estimates "}<InlineMath>{"[0.2,0.8]"}</InlineMath>{" lead to class 1 if we choose the larger entry. The value 0.8 is the model's confidence. It becomes a reliable frequency statement only if the probabilities are appropriately calibrated in the population where we use them."}</Prose>

<Prose>{"There are two distinct destinations for the predictions."}</Prose>

<DataTable caption={"1. What information is actually available?"} headers={[<>{"Task"}</>,<>{"What is available during fitting?"}</>,<>{"What must the method produce?"}</>]} rows={[[<>{"Transductive"}</>,<>{"Labels on part of a fixed collection and inputs for the whole collection"}</>,<>{"Labels for that collection's remaining items"}</>],[<>{"Inductive"}</>,<>{"A training collection with some missing labels"}</>,<>{"A rule that can handle future inputs"}</>]]} />

<Prose>{"A graph algorithm naturally assigns labels to its existing nodes. Using it for a new document requires an extension: connect the new input to the training collection, or train a prediction rule using the graph. The real experiment below uses scikit-learn's inductive prediction operation; it never adds development or test inputs to the training graph."}</Prose>

<LabelLedgerFigure />

<H3>{"Why unlabeled inputs cannot solve the task on their own"}</H3>

<Prose>{"Imagine two dense groups of points, one near "}<InlineMath>{"x=-2"}</InlineMath>{" and one near "}<InlineMath>{"x=2"}</InlineMath>{". One task labels the left group 0 and the right group 1. Another task labels points by whether a hidden inspection found a defect, independently of which group they occupy. The input distribution can be identical in both tasks."}</Prose>

<Prose>{"Learning the groups accurately helps the first task, once labels identify the groups. It does not reveal the hidden inspection result in the second. Extra knowledge of "}<InlineMath>{"p(x)"}</InlineMath>{" is useful only through a relationship with "}<InlineMath>{"p(y\\mid x)"}</InlineMath>{"."}</Prose>

<SameInputsFigure />

<Prose>{"Several relationships are common:"}</Prose>

<ul className="ssl-prose-list"><li>{""}<strong>{"Local smoothness:"}</strong>{" sufficiently similar inputs tend to have similar targets."}</li>
<li>{""}<strong>{"Cluster or low-density separation:"}</strong>{" a good classification boundary tends to avoid dense regions of the input distribution."}</li>
<li>{""}<strong>{"Manifold structure:"}</strong>{" relevant variation lies near a lower-dimensional surface, and the target changes smoothly along that surface."}</li>
<li>{""}<strong>{"Complementary views:"}</strong>{" different representations contain useful evidence about the same target, with sufficiently different mistakes."}</li></ul>

<Prose>{"A curved sheet of data does not automatically imply that every connected part has one class. “The data have a manifold” and “labels are smooth on this manifold” are separate claims."}</Prose>

<Prose>{"The previous Gaussian-process lesson made similarity explicit through a covariance kernel. Here we will first express it through graph edges. In both cases, a beautiful similarity picture can encode the wrong relationship."}</Prose>

<H2>{"2. Label propagation: let neighboring examples constrain one another"}</H2>

<Prose>{"Make each example a node. Connect two nodes when their inputs are similar. A nonnegative edge weight "}<InlineMath>{"w_{ij}"}</InlineMath>{" measures the strength of that relationship. For now the graph is undirected, so "}<InlineMath>{"w_{ij}=w_{ji}"}</InlineMath>{", and there are no self-edges."}</Prose>

<Prose>{"Consider this deliberately small, authored graph:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" A\\; \\xleftrightarrow{\\;1\\;}\\;B\\;\n \\xleftrightarrow{\\;1\\;}\\;C\\;\n \\xleftrightarrow{\\;1\\;}\\;D,\n \\qquad f_A=0,\\quad f_D=1."}</MathBlock></div>

<Prose>{"The endpoints are observed labels. The unknown value "}<InlineMath>{"f_B"}</InlineMath>{" will be a score for class 1. "}<strong>{"Hard clamping"}</strong>{" means that A and D remain fixed while B and C update."}</Prose>

<Prose>{"A weighted average multiplies each neighbor's value by its edge weight, adds the products, then divides by the total incident weight. At equilibrium:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" f_B=\\frac{f_A+f_C}{2}=\\frac{f_C}{2},\n \\qquad\n f_C=\\frac{f_B+f_D}{2}=\\frac{f_B+1}{2}."}</MathBlock></div>

<Prose>{"Substitute the first equation into the second: "}<InlineMath>{"f_C=(f_C/2+1)/2"}</InlineMath>{", hence "}<InlineMath>{"f_C=2/3"}</InlineMath>{" and "}<InlineMath>{"f_B=1/3"}</InlineMath>{". Thresholding at 0.5 assigns B to class 0 and C to class 1. This is a consequence of the edges and boundary labels, not an additional observation about B or C."}</Prose>

<Prose>{"We can also reach the answer by repeated averaging. Update all unknown nodes from the "}<strong>{"previous"}</strong>{" state, then restore the endpoints:"}</Prose>

<DataTable caption={"2. Label propagation: let neighboring examples constrain one another"} headers={[<>{"Update"}</>,<>{"A"}</>,<>{"B"}</>,<>{"C"}</>,<>{"D"}</>]} rows={[[<>{"Initial"}</>,<>{"0"}</>,<>{"0.5000"}</>,<>{"0.5000"}</>,<>{"1"}</>],[<>{"1"}</>,<>{"0"}</>,<>{"0.2500"}</>,<>{"0.7500"}</>,<>{"1"}</>],[<>{"2"}</>,<>{"0"}</>,<>{"0.3750"}</>,<>{"0.6250"}</>,<>{"1"}</>],[<>{"3"}</>,<>{"0"}</>,<>{"0.3125"}</>,<>{"0.6875"}</>,<>{"1"}</>],[<>{"Equilibrium"}</>,<>{"0"}</>,<>{"1/3"}</>,<>{"2/3"}</>,<>{"1"}</>]]} />

<Prose>{"The intermediate values alternate around the answer. A single update is not the solution, and changing update order would produce a different intermediate trace."}</Prose>

<HardFlowFigure />

<H3>{"One shortcut can reverse a prediction"}</H3>

<Prose>{"Add an edge from B directly to D with weight 2. Now B receives two units of influence from the class-1 anchor:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" f_B=\\frac{0+f_C+2}{4},\\qquad f_C=\\frac{f_B+1}{2}."}</MathBlock></div>

<Prose>{"The solution is "}<InlineMath>{"f_B=5/7"}</InlineMath>{" and "}<InlineMath>{"f_C=6/7"}</InlineMath>{". B changes class. Adding unlabeled examples can create similar bridges in a feature graph, so collecting more examples can alter predictions far from their immediate neighbors."}</Prose>

<LabelPropagationLab />

<H3>{"Why this is also an electrical circuit"}</H3>

<Prose>{"Interpret each edge weight as a conductance, the inverse of resistance. Hold A at voltage 0 and D at voltage 1. At an interior node, total incoming current equals total outgoing current:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" \\sum_j w_{ij}(f_i-f_j)=0."}</MathBlock></div>

<Prose>{"Rearranging gives precisely the weighted-average equation. A strong edge resists a large disagreement between its endpoints. This analogy gives a useful diagnostic: a mislabeled boundary node can influence a whole well-connected region."}</Prose>

<Prose>{"There is also a random-walk interpretation. Starting at B, repeatedly choose a neighbor with probability proportional to the edge weight. Stop on reaching a labeled node. On this graph, the chance of hitting the class-1 endpoint first is "}<InlineMath>{"1/3"}</InlineMath>{". This probabilistic interpretation belongs to the specified walk and boundary problem; it does not establish calibration for real-world class labels. These connections are developed in "}<a href={"https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf"}>{"Zhu, Ghahramani and Lafferty's harmonic-function formulation"}</a>{"."}</Prose>

<H3>{"The compact matrix form"}</H3>

<Prose>{"Let "}<InlineMath>{"W"}</InlineMath>{" contain the edge weights, and let "}<InlineMath>{"D"}</InlineMath>{" be diagonal with "}<InlineMath>{"D_{ii}=\\sum_j w_{ij}"}</InlineMath>{". The graph Laplacian is "}<InlineMath>{"L_g=D-W"}</InlineMath>{". It measures disagreement: for a vector "}<InlineMath>{"f"}</InlineMath>{","}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" f^\\top L_g f=\\frac12\\sum_{i,j} w_{ij}(f_i-f_j)^2."}</MathBlock></div>

<Prose>{"The factor "}<InlineMath>{"1/2"}</InlineMath>{" avoids counting both directions of each undirected edge twice. Minimizing this disagreement while keeping labeled values fixed produces the harmonic equations."}</Prose>

<Prose>{"Partition the rows into labeled positions "}<InlineMath>{"\\ell"}</InlineMath>{" and unlabeled positions "}<InlineMath>{"u"}</InlineMath>{":"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" (L_g)_{uu}f_u=W_{u\\ell}y_\\ell."}</MathBlock></div>

<Prose>{"Solve this linear system instead of explicitly forming its inverse. For a finite undirected graph with nonnegative weights, every connected component containing unknown nodes needs a labeled anchor for this boundary solution to be unique. An unanchored component admits an arbitrary constant value; no boundary label chooses that constant."}</Prose>

<Prose>{"For more than two classes, use one score column per class and solve the same problem for each one. A one-hot label for class 2 in a three-class task is "}<InlineMath>{"[0,0,1]"}</InlineMath>{": all its mass goes in the third column."}</Prose>

<H3>{"Build the graph deliberately"}</H3>

<Prose>{"A common edge weight is"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" w_{ij}=\\exp(-\\gamma\\|x_i-x_j\\|^2),\\qquad i\\ne j."}</MathBlock></div>

<Prose>{"Small "}<InlineMath>{"\\gamma"}</InlineMath>{" connects more distant points strongly; large "}<InlineMath>{"\\gamma"}</InlineMath>{" concentrates influence nearby. Feature units matter: a coordinate measured in thousands can dominate another measured in fractions unless the metric or scaling addresses that difference."}</Prose>

<Prose>{"A nearest-neighbor graph retains only local connections. Its construction choices matter too: directed neighbor lists, their undirected union, and their mutual-neighbor intersection produce different graphs. A two-dimensional projection of four-dimensional features is a viewing aid, not proof that an edge is wrong. Inspect neighbors in the representation actually used."}</Prose>

<H2>{"3. Label spreading: soften the anchors without confusing the normalization"}</H2>

<Prose>{"Hard propagation treats the observed labels as fixed boundary conditions. Sometimes an observed label could be wrong, or we want a regularized balance between neighbor agreement and label fidelity."}</Prose>

<Prose>{"Define"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" S=D^{-1/2}WD^{-1/2},\\qquad\n F^{(t+1)}=\\alpha S F^{(t)}+(1-\\alpha)Y,\n \\quad 0\\leq\\alpha<1."}</MathBlock></div>

<Prose>{""}<InlineMath>{"F"}</InlineMath>{" now has one column per class. "}<InlineMath>{"Y"}</InlineMath>{" contains one-hot rows at labeled nodes and "}<strong>{"zero rows at unlabeled nodes"}</strong>{". Those zeros mean no injected label evidence. They are not a 50–50 class estimate."}</Prose>

<Prose>{"Each update mixes propagated evidence with a fresh injection of the observed labels. Labeled rows can change, so this is soft anchoring. The normalized matrix "}<InlineMath>{"S"}</InlineMath>{" is symmetric, but it is "}<strong>{"not a row-stochastic transition matrix"}</strong>{". For our four-node chain, its row sums are approximately "}<InlineMath>{"[0.7071,1.2071,1.2071,0.7071]"}</InlineMath>{". The random-walk matrix is "}<InlineMath>{"P=D^{-1}W"}</InlineMath>{", whose non-isolated rows sum to one."}</Prose>

<Prose>{"At a fixed point:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" (I-\\alpha S)F=(1-\\alpha)Y."}</MathBlock></div>

<Prose>{"For the same endpoints and "}<InlineMath>{"\\alpha=0.8"}</InlineMath>{", the raw class-1 score at B is about 0.149652 and its class-0 score is 0.254409. Normalize that row for a distribution-shaped display:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" \\frac{0.149652}{0.254409+0.149652}\\approx0.370370."}</MathBlock></div>

<Prose>{"This differs from hard propagation's "}<InlineMath>{"1/3"}</InlineMath>{". At A, the normalized class-1 score is about 0.197531 even though its observed label is 0. Soft anchoring has an observable effect."}</Prose>

<EvidenceFigure />

<Prose>{"An unanchored connected component also receives no evidence and has zero scores under this update. Do not divide by a zero row sum or turn the first column of an all-zero tie into a confident label."}</Prose>

<H3>{"What the regularizer asks for"}</H3>

<Prose>{"For this symmetric graph, the fixed point minimizes"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" J(F)=\\alpha\\,\\operatorname{tr}\\!\\left(F^\\top(I-S)F\\right)\n       +(1-\\alpha)\\|F-Y\\|_F^2."}</MathBlock></div>

<Prose>{"The first term penalizes disagreement after degree normalization; the second penalizes departure from the injected evidence. Taking the derivative and setting it to zero gives "}<InlineMath>{"(I-\\alpha S)F=(1-\\alpha)Y"}</InlineMath>{". This connects the update to an optimization problem rather than a visual blending trick. "}<a href={"https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf"}>{"Zhou and colleagues"}</a>{" develop the local-and-global-consistency approach."}</Prose>

<Prose>{"The error obeys "}<InlineMath>{"E^{(t+1)}=\\alpha S E^{(t)}"}</InlineMath>{". Since the eigenvalues of this normalized adjacency lie in "}<InlineMath>{"[-1,1]"}</InlineMath>{", the Euclidean error norm contracts by at most "}<InlineMath>{"\\alpha"}</InlineMath>{" per step. Increasing "}<InlineMath>{"\\alpha"}</InlineMath>{" toward one can slow convergence considerably: "}<InlineMath>{"0.99^{100}\\approx0.366"}</InlineMath>{", and "}<InlineMath>{"0.99^{500}\\approx0.00657"}</InlineMath>{". A worst-case reduction to "}<InlineMath>{"10^{-6}"}</InlineMath>{" needs 1,375 steps at that factor. Use a residual or convergence criterion, not an animation that declares victory after an arbitrary number of frames."}</Prose>

<H3>{"Reproduce the two different solutions"}</H3>

<Prose>{"Install NumPy with "}<code>{"python -m pip install numpy"}</code>{", save this complete program as "}<code>{"graph-solutions.py"}</code>{", and run "}<code>{"python graph-solutions.py"}</code>{". It constructs only the four-node anchored chain, so the inverse square-root degrees are defined and the hard system has a unique solution."}</Prose>

<RunnableExample example={semiSupervisedExamples[0]} /><p className="lesson-note"><a href={semiSupervisedExamples[0].download} download>Download the exact Python program</a></p>

<Prose>{"The outputs are hard B,C = [0.333333, 0.666667] and soft class-1 readout = [0.197531, 0.370370, 0.629630, 0.802469], with evidence at all four nodes. A zero row in the guarded division is storage for an unavailable readout; its evidence flag must remain false. To add isolated nodes, also guard zero degrees and check component anchoring before solving the hard system."}</Prose>

<H2>{"4. Self-training: make a guess, record its origin, then refit"}</H2>

<Prose>{"Self-training does not require an explicit graph. Start with a supervised classifier trained on "}<InlineMath>{"L"}</InlineMath>{". Use it to propose labels for "}<InlineMath>{"U"}</InlineMath>{". Add a selected subset of these pseudo-labels to training, then fit again."}</Prose>

<Prose>{"The loop has six concrete operations:"}</Prose>

<ol className="ssl-prose-list"><li>{"Fit using observed labels and previously accepted pseudo-labels."}</li>
<li>{"Predict class probabilities for the remaining unlabeled inputs."}</li>
<li>{"Accept a prediction only when the selection rule allows it; a common rule is "}<InlineMath>{"\\max_c p(c\\mid x)\\geq\\tau"}</InlineMath>{"."}</li>
<li>{"Store its predicted class, confidence, round and provenance."}</li>
<li>{"Refit with the expanded labeled collection."}</li>
<li>{"Stop when no candidates qualify, the pool is exhausted, the round budget is reached, or a predeclared development criterion chooses an earlier model."}</li></ol>

<Prose>{"The last returned model must include the final accepted batch. Returning the model fitted just before the last promotion silently drops that batch's influence."}</Prose>

<H3>{"Watch a boundary move for the wrong reason"}</H3>

<Prose>{"For a transparent mechanism, use a one-dimensional prototype classifier. Each class prototype "}<InlineMath>{"\\mu_c"}</InlineMath>{" is the average of its currently labeled inputs. Its score is"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" p(c\\mid x)=\n \\frac{\\exp(-(x-\\mu_c)^2)}\n {\\exp(-(x-\\mu_0)^2)+\\exp(-(x-\\mu_1)^2)}."}</MathBlock></div>

<Prose>{"These are deliberately chosen model scores, not calibrated probabilities. With equal distance scales, the decision boundary lies halfway between the two prototypes."}</Prose>

<Prose>{"Start with observed "}<InlineMath>{"(-2,0)"}</InlineMath>{" and "}<InlineMath>{"(2,1)"}</InlineMath>{" and unlabeled inputs "}<InlineMath>{"[-1,0,1,3]"}</InlineMath>{". The boundary starts at zero. With threshold 0.8, the first round accepts "}<InlineMath>{"-1"}</InlineMath>{" as class 0, and 1 and 3 as class 1. The new prototypes are "}<InlineMath>{"-1.5"}</InlineMath>{" and 2, so the boundary moves to 0.25. The input 0 now receives a class-0 score about 0.852 and is accepted next. Final prototypes are "}<InlineMath>{"-1"}</InlineMath>{" and 2, giving boundary 0.5."}</Prose>

<Prose>{"Replace the unlabeled input 3 with 9. The first positive pseudo-label batch pulls the class-1 prototype to 4. After the next promotion, the boundary is 1.5. A query at "}<InlineMath>{"x=1.25"}</InlineMath>{" changes from predicted class 1 to class 0 solely because of that unlabeled point's influence. Whether the change helps depends on the query's real label, which the procedure does not know."}</Prose>

<PrototypeFlowFigure /><SelfTrainingLab />

<Prose>{"This feedback is "}<strong>{"confirmation bias"}</strong>{": an incorrect prediction can enter the training data and help generate more incorrect predictions. Raising the threshold can reduce promotions; it does not certify the ones that remain. A class that is initially harder to recognize can also receive fewer pseudo-labels, amplifying imbalance."}</Prose>

<Prose>{"Possible responses include improving the seed coverage, changing the representation, auditing a random sample of proposed labels, weighting pseudo-labels less strongly, or stopping earlier on labeled development evidence. Each changes a specific part of the loop. None removes the need to measure the result."}</Prose>

<H3>{"Why this is not automatically expectation–maximization"}</H3>

<Prose>{"The earlier mixture-model lesson used EM with a generative model "}<InlineMath>{"p_\\theta(x,y)"}</InlineMath>{". Unlabeled inputs contribute "}<InlineMath>{"\\log\\sum_y p_\\theta(x,y)"}</InlineMath>{" to its likelihood, and an E-step computes latent-label responsibilities under that model."}</Prose>

<Prose>{"For a purely conditional classifier, summing "}<InlineMath>{"p_\\theta(y\\mid x)"}</InlineMath>{" over its classes gives 1. The corresponding unlabeled log term is "}<InlineMath>{"\\log 1=0"}</InlineMath>{"; it cannot train the classifier by itself. Pseudo-labeling adds an extra assumption or objective. A thresholded, hard-label refitting loop is not automatically an EM algorithm and does not inherit EM's likelihood-ascent argument."}</Prose>

<H2>{"5. Co-training: let another representation provide the label"}</H2>

<Prose>{"Suppose one classifier reads a web page's body and another reads the text of incoming links. A page with an unfamiliar body may still receive informative incoming links. One view can label an example whose other view is currently difficult, giving the other learner a new training pair."}</Prose>

<Prose>{"This is the purpose of two views: "}<strong>{"useful evidence that is not merely a duplicate of the same mistake"}</strong>{". Splitting a feature vector in half does not establish that property."}</Prose>

<Prose>{"Here is an authored categorical version that exposes the transfer without hiding it inside a large classifier. In each view, the learner remembers a category-to-class rule if all training labels it has seen for that category agree. It abstains on unseen or contradictory categories."}</Prose>

<DataTable caption={"5. Co-training: let another representation provide the label"} headers={[<>{"Row"}</>,<>{"View 1"}</>,<>{"View 2"}</>,<>{"Initially observed label"}</>]} rows={[[<>{"0"}</>,<>{"red"}</>,<>{"round"}</>,<>{"0"}</>],[<>{"1"}</>,<>{"blue"}</>,<>{"square"}</>,<>{"1"}</>],[<>{"2"}</>,<>{"red"}</>,<>{"triangle"}</>,<>{"unknown"}</>],[<>{"3"}</>,<>{"green"}</>,<>{"triangle"}</>,<>{"unknown"}</>],[<>{"4"}</>,<>{"orange"}</>,<>{"square"}</>,<>{"unknown"}</>],[<>{"5"}</>,<>{"orange"}</>,<>{"hexagon"}</>,<>{"unknown"}</>],[<>{"6"}</>,<>{"red"}</>,<>{"square"}</>,<>{"unknown"}</>]]} />

<Prose>{"Initially, view 1 knows red→0 and blue→1; view 2 knows round→0 and square→1."}</Prose>

<Prose>{"In round 1, view 1 proposes class 0 for row 2. The recipient learns triangle→0 from "}<strong>{"its own feature"}</strong>{" on that row. View 2 proposes class 1 for row 4, teaching view 1 orange→1."}</Prose>

<Prose>{"In round 2, triangle→0 lets view 2 teach green→0 through row 3. Orange→1 lets view 1 teach hexagon→1 through row 5. Some already implied labels are exchanged too; those confirmations are not new independent evidence."}</Prose>

<Prose>{"Row 6 is different: red says 0 while square says 1. Our declared conflict rule is to defer both offers on a conflicting row. The disagreement exposes a failure of the category-consistency assumptions. It does not tell us which view is correct."}</Prose>

<PairedViewsFigure /><CoTrainingLab />

<H3>{"A complete, inspectable co-training procedure"}</H3>

<Prose>{"The companion "}<a href={"/learn-assets/semi-supervised-learning/cotrain-categories.py"}>{"categorical co-training program"}</a>{" implements exactly this exercise with Python's standard library. Save it beside your working files and run:"}</Prose>

<CodeBlock language={"text"}>{"python cotrain-categories.py"}</CodeBlock>

<Prose>{"Its rule learner groups received labels by category, retaining a rule only when the set has one member. During a round, both learners fit first and make all proposals before either receives a new label. Each learner has its own training-label array. An accepted offer is stored only in the recipient's array, using the recipient's representation on that row. Conflicting proposals are deferred; already labeled recipient entries are preserved. Iteration stops on no offers or after eight rounds."}</Prose>

<Prose>{"The printed offers have the form (row, donor view, recipient view, label):"}</Prose>

<CodeBlock language={"text"}>{"round 1 offers [(2, 1, 2, 0), (4, 2, 1, 1)] conflicts [6]\nround 2 offers [(2, 2, 1, 0), (3, 2, 1, 0), (4, 1, 2, 1), (5, 1, 2, 1)] conflicts [6]\nround 3 offers [(3, 1, 2, 0), (5, 2, 1, 1)] conflicts [6]\nround 4 offers [] conflicts [6]"}</CodeBlock>

<details><summary>Inspect the complete categorical co-training program and its recorded output</summary><RunnableExample example={semiSupervisedExamples[1]} /><p><a href={semiSupervisedExamples[1].download} download>Download the complete co-training program</a></p></details>

<Prose>{"The final view-1 rules include green→0 and orange→1; view 2 includes triangle→0 and hexagon→1. Run the duplicate-view case and the unresolved view-1 row indices are "}<InlineMath>{"[3,4,5]"}</InlineMath>{". The exercise teaches the information flow and its failure conditions, not the accuracy of a real document classifier."}</Prose>

<Prose>{"The original "}<a href={"https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf"}>{"Blum–Mitchell co-training work"}</a>{" supplies a theoretical setting with compatible views, conditional independence given the target, a weakly useful starting predictor and noise-learnability conditions. That result is not a universal label-budget guarantee for every practical exchange loop."}</Prose>

<Prose>{"Conditional independence means that, "}<strong>{"within a fixed true class"}</strong>{", observing view 1’s features does not change the distribution of view 2. Merely observing agreement, splitting features, or reporting a low overall error correlation does not establish it. Practical co-training can be investigated when the theorem's assumptions are imperfect, but the experiment must earn its own conclusion."}</Prose>

<Prose>{"For a real system, specify separate preprocessing for the views, a shared target definition, promotion thresholds, conflict handling, class imbalance treatment, stopping criteria and label provenance. Evaluate each view and the chosen combination on the same held-out rows. Never feed view-2 features to a model trained to interpret view-1 coordinates."}</Prose>

<H2>{"6. A real experiment where the supervised baseline wins"}</H2>

<Prose>{"We will use four wavelet-derived measurements from the "}<a href={"https://archive.ics.uci.edu/dataset/267/banknote+authentication"}>{"UCI Banknote Authentication dataset"}</a>{", contributed by Volker Lohweg. The retained features are variance, skewness, curtosis and entropy. The source uses class codes 0 and 1; this exercise keeps those codes rather than guessing their semantic mapping."}</Prose>

<Prose>{"Download the accompanying "}<a href={"/learn-assets/semi-supervised-learning/banknote-subset.csv"}>{"480-row CSV"}</a>{" and keep its "}<a href={"/learn-assets/semi-supervised-learning/data-provenance.md"}>{"provenance record"}</a>{". It is a deterministic subset of the 1,372-row dataset, shared under CC BY 4.0. The first 320 retained rows form the training pool; the next 80 are development data and the final 80 are the locked test set."}</Prose>

<Prose>{"We reveal the first three training-pool labels from each class: six labels, with both classes deliberately represented. The other 314 training labels are unavailable to fitting. This balanced seed selection simulates an oracle; six arbitrary labels would not necessarily cover both classes."}</Prose>

<Prose>{"There are also 80 development and 80 test labels. "}<strong>{"Six training labels does not mean six labels for the whole project."}</strong>{" We keep this evaluation cost visible because tiny-label demonstrations can otherwise hide most of their human supervision in model selection."}</Prose>

<Prose>{"Standardization fits all 320 training inputs and no development or test inputs. Every candidate uses that same label-free representation. Our logistic baseline is consequently a supervised classifier with shared unlabeled preprocessing, a controlled comparison of classifier changes rather than a strict no-unlabeled-input system."}</Prose>

<Prose>{"The candidates are specified before looking at development results:"}</Prose>

<ul className="ssl-prose-list"><li>{"Logistic regression with "}<InlineMath>{"C=1"}</InlineMath>{" on the six observed labels."}</li>
<li>{"A 3-nearest-neighbor classifier on the same labels."}</li>
<li>{"RBF label spreading with "}<InlineMath>{"\\alpha=0.2"}</InlineMath>{" and "}<InlineMath>{"\\gamma\\in\\{0.25,1,4\\}"}</InlineMath>{"."}</li>
<li>{"Logistic self-training with thresholds 0.8 or 0.95 and at most ten rounds."}</li></ul>

<Prose>{"No model sees hidden training truth to decide its promotions. That truth is opened only for the final instructional audit of how the guesses went wrong."}</Prose>

<H3>{"Run the experiment"}</H3>

<Prose>{"Use a Python environment with NumPy and scikit-learn:"}</Prose>

<CodeBlock language={"text"}>{"python -m pip install numpy scikit-learn\npython banknote-experiment.py"}</CodeBlock>

<Prose>{"Save the following complete program as "}<code>{"banknote-experiment.py"}</code>{" beside the CSV. The checked run used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1."}</Prose>

<RunnableExample example={semiSupervisedExamples[2]} /><p className="lesson-note"><a href={semiSupervisedExamples[2].download} download>Download the exact Python program</a></p>

<Prose>{"The development results are:"}</Prose>

<DataTable caption={"Run the experiment"} headers={[<>{"Candidate"}</>,<>{"Correct / 80"}</>,<>{"Accuracy"}</>]} rows={[[<>{"Logistic baseline"}</>,<>{"72"}</>,<>{"0.9000"}</>],[<>{"3-neighbor baseline"}</>,<>{"63"}</>,<>{"0.7875"}</>],[<>{"Spreading, "}<InlineMath>{"\\gamma=0.25"}</InlineMath>{""}</>,<>{"61"}</>,<>{"0.7625"}</>],[<>{"Spreading, "}<InlineMath>{"\\gamma=1"}</InlineMath>{""}</>,<>{"59"}</>,<>{"0.7375"}</>],[<>{"Spreading, "}<InlineMath>{"\\gamma=4"}</InlineMath>{""}</>,<>{"65"}</>,<>{"0.8125"}</>],[<>{"Self-training, "}<InlineMath>{"\\tau=0.8"}</InlineMath>{""}</>,<>{"59"}</>,<>{"0.7375"}</>],[<>{"Self-training, "}<InlineMath>{"\\tau=0.95"}</InlineMath>{""}</>,<>{"72"}</>,<>{"0.9000"}</>]]} />

<Prose>{"The 0.95 self-training run promotes no examples, so it is the logistic baseline again. The predeclared tie rule selects that simpler baseline. It correctly predicts "}<strong>{"72 of the 80 locked test rows"}</strong>{". We do not refit on development labels, preserving the six-label training comparison."}</Prose>

<Prose>{"The 0.8 run tells a more revealing story:"}</Prose>

<DataTable caption={"Run the experiment"} headers={[<>{"Promotion round"}</>,<>{"Newly accepted"}</>,<>{"Wrong in the offline audit"}</>]} rows={[[<>{"1"}</>,<>{"48"}</>,<>{"0"}</>],[<>{"2"}</>,<>{"106"}</>,<>{"12"}</>],[<>{"3"}</>,<>{"60"}</>,<>{"28"}</>],[<>{"4"}</>,<>{"18"}</>,<>{"14"}</>],[<>{"5"}</>,<>{"11"}</>,<>{"10"}</>],[<>{"6"}</>,<>{"6"}</>,<>{"6"}</>],[<>{"7"}</>,<>{"3"}</>,<>{"3"}</>],[<>{"8"}</>,<>{"0"}</>,<>{"0"}</>]]} />

<Prose>{"Its first batch looks excellent. Later guesses become increasingly unreliable as the training labels feed back into the decision boundary. It accepts 252 pseudo-labels, including 73 wrong ones, and leaves 62 training examples unlabeled. Increasing the training set's apparent size has not increased its trustworthy information."}</Prose>

<PromotionAuditFigure />

<Prose>{"The appropriate conclusion is narrow: under this subset, seed policy, representation and candidate set, the measured evidence selects the baseline. It does not establish that SSL always fails on banknotes, or that a differently tuned graph cannot work. The useful action is to retain the baseline and investigate a specific representation or label-coverage hypothesis before launching another experiment."}</Prose>

<H2>{"7. Choose the method by the information it can use"}</H2>

<DataTable caption={"7. Choose the method by the information it can use"} headers={[<>{"Available structure"}</>,<>{"Candidate to investigate"}</>,<>{"First thing to inspect"}</>]} rows={[[<>{"Meaningful pairwise similarities and a fixed collection"}</>,<>{"Graph propagation or spreading"}</>,<>{"Neighbor quality, cross-class shortcuts and unanchored components"}</>],[<>{"A reasonable initial classifier and many relevant inputs"}</>,<>{"Self-training"}</>,<>{"Pseudo-label correctness by class and round, with observed-label comparison"}</>],[<>{"Two representations with complementary evidence"}</>,<>{"Co-training"}</>,<>{"View-specific errors, transfer provenance and conflicts"}</>],[<>{"Label-preserving transformations"}</>,<>{"Consistency-based learning"}</>,<>{"Whether each transformation preserves this task's target"}</>]]} />

<Prose>{"For an annotation team, a graph can expose islands with no labeled anchor and suggest where one additional label might matter. For linked biological entities, neighborhood information can suggest a function, but an interaction edge is not automatically evidence for the same function. For paired audio and visual observations, agreement can be informative, yet common background artifacts may make both views wrong together. These are questions to test, not automatic reasons to adopt SSL."}</Prose>

<H3>{"Keep the compute proportional to the collection"}</H3>

<Prose>{"A dense graph on "}<InlineMath>{"n"}</InlineMath>{" nodes stores "}<InlineMath>{"n^2"}</InlineMath>{" weights. At 100,000 nodes, one float64 weight matrix alone takes "}<InlineMath>{"8\\times10^{10}"}</InlineMath>{" bytes, about 80 GB before scores, copies or overhead. That storage calculation follows from dimensions; it is not a timing benchmark."}</Prose>

<Prose>{"A sparse graph with "}<InlineMath>{"E"}</InlineMath>{" stored directed edges takes roughly "}<InlineMath>{"O(E+n)"}</InlineMath>{" graph storage plus "}<InlineMath>{"O(nC)"}</InlineMath>{" scores for "}<InlineMath>{"C"}</InlineMath>{" classes. The graph multiplication costs "}<InlineMath>{"O(EC)"}</InlineMath>{", and adding the label evidence and updating all scores adds "}<InlineMath>{"O(nC)"}</InlineMath>{". Constructing the nearest-neighbor graph can itself be expensive, especially in high dimensions, and symmetrizing it can increase the number of stored edges. Count indices and row pointers as well as weight values."}</Prose>

<Prose>{"Self-training's cost is the sum of fitting and prediction over its rounds, not one supervised fit. Co-training can require two such fitting sequences. Cache representations that do not change; avoid repeated full graph construction when only a display changes; keep the labeled and pseudo-labeled provenance separate from the numerical arrays."}</Prose>

<H3>{"Put the common failure checks in one place"}</H3>

<Prose>{"Ask whether training and unlabeled examples share the relevant target space; whether all important classes appear among the seeds; whether the graph or augmentation encodes the intended similarity; and whether the development labels are sufficient for the comparison you are making."}</Prose>

<Prose>{"Keep test inputs out of inductive training preprocessing, even when their labels are hidden. If you intentionally use all inputs for a transductive problem, state that protocol and evaluate that task. Do not use hidden benchmark labels for stopping, then describe the loop as unlabeled."}</Prose>

<Prose>{"Class-balanced promotion can be a hypothesis when one class is being ignored. It also imposes a selection policy, and an equal quota may be inappropriate for unequal real prevalence. A high threshold, a nice cluster plot and agreement between models each provide a different kind of evidence; none alone proves correctness."}</Prose>

<Prose>{""}<a href={"https://arxiv.org/pdf/1804.09170"}>{"Oliver and colleagues' evaluation study"}</a>{" is a useful reminder to compare underlying models fairly, account for validation labels and investigate mismatched unlabeled data."}</Prose>

<H2>{"8. Optional depth: what connects these methods to later learning systems?"}</H2>

<H3>{"Generative models and low-density boundaries"}</H3>

<Prose>{"A generative mixture can use all inputs to estimate where components lie, while observed labels help connect components to classes. The benefit depends on whether the model's component assumptions match the task. Better input-density fit does not imply better classification."}</Prose>

<Prose>{"A transductive support-vector method instead searches for labels and a large-margin boundary jointly, encouraging separation in low-density regions. The discrete unknown labels make this a harder optimization problem than ordinary supervised convex SVM fitting. Entropy regularization encourages confident predictions on unlabeled inputs; it needs safeguards because confidently assigning everything to one class can satisfy a confidence objective poorly aligned with the task."}</Prose>

<Prose>{"Manifold regularization combines a supervised loss, a function-complexity penalty and graph disagreement. Schematically:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" \\frac1{|L|}\\sum_{i\\in L}\\ell(y_i,f(x_i))\n +\\lambda_A\\|f\\|_{\\mathcal H}^2\n +\\lambda_I f(X)^\\top L_g f(X)."}</MathBlock></div>

<Prose>{"The first term learns from observed labels, the second controls the prediction rule and the third lets the input geometry constrain it. The kernel space connects back to the GP lesson; the graph term connects to section 2. Constants depend on the formulation, so preserve them when reproducing a specific algorithm."}</Prose>

<H3>{"Deep consistency and teacher–student learning"}</H3>

<Prose>{"In "}<a href={"https://arxiv.org/pdf/2001.07685"}>{"FixMatch"}</a>{", a weakly augmented input supplies a pseudo-label; a strongly augmented version is trained toward it:"}</Prose>

<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{" q=p_\\theta(\\cdot\\mid a_{\\rm weak}(x)),\\quad\n \\hat y=\\arg\\max_c q_c,\\quad\n \\ell_u=\\mathbf1[\\max q\\geq\\tau]\\,\n       [-\\log p_\\theta(\\hat y\\mid a_{\\rm strong}(x))]."}</MathBlock></div>

<Prose>{"Treat the selected target and acceptance decision as fixed for that gradient update. Average over the specified unlabeled batch and combine with supervised loss using a weight "}<InlineMath>{"\\lambda_u"}</InlineMath>{". A horizontal flip may preserve an animal category but alter the interpretation of a character. The transformation is part of the modeling assumption."}</Prose>

<Prose>{""}<a href={"https://arxiv.org/pdf/1911.04252"}>{"Noisy Student"}</a>{" trains a teacher, generates pseudo-labels, then trains a student with input and model noise before optionally repeating the process. The student can have equal or greater capacity, and the noise introduces consistency pressure; a pseudo-label still has an origin and can still be wrong."}</Prose>

<Prose>{"Self-supervised representation learning is related but distinct: its training targets can be constructed from inputs, such as masked pieces or paired views. A later classifier may use that representation with scarce human labels. Compare such a representation baseline when appropriate rather than assuming all useful unlabeled learning must happen through label propagation."}</Prose>

<H3>{"Learning guarantees need the relationship to be stated"}</H3>

<Prose>{"If two possible worlds have the same input distribution but different target rules, unlabeled inputs alone cannot distinguish them. A guarantee must restrict that ambiguity through a hypothesis class, compatibility condition, graph assumption or other stated relationship. The later PAC and generalization lessons formalize what can be inferred from a limited sample; they do not turn “more unlabeled data” into an assumption-free promise."}</Prose>

<H2>{"9. Practice: produce a decision and explain its evidence"}</H2>

<Prose>{"Try each question before opening the hint or solution."}</Prose>

<H3>{"A. Change the graph"}</H3>

<Prose>{"In the A–B–C–D chain, use edge weights A–B=2, B–C=1, C–D=1, with endpoint labels 0 and 1. Compute B and C. Which endpoint gained influence?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The new average is "}<InlineMath>{"f_B=f_C/3"}</InlineMath>{". The other interior equation remains "}<InlineMath>{"f_C=(f_B+1)/2"}</InlineMath>{"."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Substitution gives "}<InlineMath>{"f_C=(f_C/3+1)/2"}</InlineMath>{", so "}<InlineMath>{"f_C=3/5"}</InlineMath>{" and "}<InlineMath>{"f_B=1/5"}</InlineMath>{". The stronger connection to the class-0 endpoint pulls both values downward from "}<InlineMath>{"1/3,2/3"}</InlineMath>{"."}</Prose>

</details>

<H3>{"B. Diagnose a normalization error"}</H3>

<Prose>{"A teammate says every row of "}<InlineMath>{"S=D^{-1/2}WD^{-1/2}"}</InlineMath>{" is a probability distribution and initializes all unknown rows of "}<InlineMath>{"Y"}</InlineMath>{" to "}<InlineMath>{"[0.5,0.5]"}</InlineMath>{". What two meanings have been mixed up?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Check the four-node chain's row sums and distinguish injected evidence from normalized display scores."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{""}<InlineMath>{"S"}</InlineMath>{" is a symmetric normalized adjacency, not generally a transition matrix. "}<InlineMath>{"P=D^{-1}W"}</InlineMath>{" is the row-stochastic walk matrix on non-isolated nodes. Standard spreading uses zero unknown rows in "}<InlineMath>{"Y"}</InlineMath>{"; adding uniform rows injects additional evidence and changes the objective. Row-normalized output is a separate readout, unavailable when a row has zero mass."}</Prose>

</details>

<H3>{"C. Identify what a pseudo-label can move"}</H3>

<Prose>{"Use the prototype classifier with observed "}<InlineMath>{"(-3,0),(3,1)"}</InlineMath>{", unlabeled "}<InlineMath>{"[-2,2,8]"}</InlineMath>{", and threshold 0.8. What is the boundary after the first accepted batch? What must be checked before claiming improvement?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"All three inputs have strong initial distance-based scores. Recompute each class mean using the original example as well as its accepted guesses."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The new means are "}<InlineMath>{"(-3-2)/2=-2.5"}</InlineMath>{" and "}<InlineMath>{"(3+2+8)/3=13/3"}</InlineMath>{". The boundary is "}<InlineMath>{"(-2.5+13/3)/2=11/12"}</InlineMath>{". It moved right from zero. Improvement requires observed-label evaluation on the intended prediction population; the number accepted and their model confidence are insufficient."}</Prose>

</details>

<H3>{"D. Break the information bridge"}</H3>

<Prose>{"In the categorical co-training table, replace row 2 with violet/triangle while leaving all other rows unchanged. Which new rules can still be learned? Which chain is broken?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Neither learner initially recognizes violet or triangle. The orange/square row still has a known second-view category."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Square→1 still teaches orange→1, which teaches hexagon→1. No observed or inferred rule reaches triangle, violet or green, so the class-0 transfer through rows 2 and 3 cannot start. This is an absence of a bridge, not evidence that the unresolved categories belong to class 1."}</Prose>

</details>

<H3>{"E. Read the banknote result"}</H3>

<Prose>{"The 0.8 run's first 48 pseudo-labels are all correct in the offline audit. Why does that not justify continuing to exhaustion? What did the 0.95 result establish?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Later predictions come from a different fitted model, and no acceptance is a legitimate outcome."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Refitting changes the boundary and the remaining pool is not the same collection as the first accepted batch. Later correctness can deteriorate, as the checked counts show. At 0.95 no points passed the rule, so this candidate reproduced the initial classifier. It showed no benefit from pseudo-labeling under that setting; it did not show that all high-confidence pseudo-labels are safe."}</Prose>

</details>

<H3>{"F. Audit the learning budget"}</H3>

<Prose>{"A report says “only six labels,” but uses six seed labels, 80 development labels and 80 test labels. It chooses its graph bandwidth after evaluating all three on the test labels. Rewrite the protocol."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate fitting, selection and final reporting, and count supervision outside the training loop."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Report six fitting labels plus 160 evaluation labels. Choose graph bandwidth and other candidates on development data with a predeclared rule. Evaluate the selected procedure once on the locked test set. If earlier test comparisons have already influenced decisions, that set has served as development data; reserve fresh evaluation data or report the limitation explicitly rather than relabeling the old result."}</Prose>

</details>

<H3>{"G. Test the claimed EM explanation"}</H3>

<Prose>{"Someone adds "}<InlineMath>{"\\sum_{x\\in U}\\log\\sum_c p_\\theta(c\\mid x)"}</InlineMath>{" to logistic regression and says unlabeled examples will improve its parameters. Compute the new term and describe a mechanism that would actually introduce information."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Sum a normalized conditional distribution over every class."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Each inner sum is one, so the added term is zero. A generative model for "}<InlineMath>{"p_\\theta(x,c)"}</InlineMath>{", a graph smoothness penalty, a pseudo-label objective or a label-preserving consistency constraint could introduce an additional relationship. Its assumptions and evaluation must then be stated."}</Prose>

</details>

<H3>{"H. A small independent investigation"}</H3>

<Prose>{"Using the CSV, reserve the same locked test rows. On a new development-only experiment, vary the number of observed training labels while retaining shared preprocessing and the same logistic baseline for each budget. Record how seeds are selected, pseudo-label counts and development accuracy. Form a hypothesis about why the threshold-0.8 run deteriorates before changing the method."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Changing the seed set and the algorithm simultaneously makes attribution difficult. Never use hidden training truth to select individual promotions."}</Prose>

</details>

<details><summary>Solution approach</summary>

<Prose>{"Choose deterministic, documented seed sets and compare supervised and self-trained versions within each set. A useful hypothesis is that a small seed set misrepresents part of a class, making later confident promotions unreliable. More representative observed labels may help; they may also change the threshold at which promotions occur. Report the measured result even if the hypothesis fails. These changed experiments have no preclaimed numeric answer, and the original final-test result does not validate them."}</Prose>

</details>

<H2>{"10. Another route through the subject"}</H2>

<ul className="ssl-prose-list"><li>{""}<a href={"https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf"}>{"Zhu's literature survey, July 2008 version"}</a>{": begin with the FAQ and then sections 3, 4 and 6. It provides a broad classical map and a useful discussion of when assumptions can fail; read it as a historical foundation."}</li>
<li>{""}<a href={"https://www.cs.cmu.edu/~wcohen/10-605/notes/graph-ssl.pdf"}>{"CMU graph semi-supervised learning notes"}</a>{": an alternative mathematical route through graph objectives and propagation. Use it after the four-node calculation if you prefer lecture notes to a research paper."}</li>
<li>{""}<a href={"https://www.youtube.com/watch?v=gnNLjX50F7U"}>{"CMU 10-601 lecture 19 video"}</a>{", with its "}<a href={"https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/19_ssl_03-30-2015.pdf"}>{"lecture slides"}</a>{": a spoken route through transductive SVMs, co-training and graph methods. The co-training and graph portions of the slides are particularly useful after sections 2–5."}</li>
<li>{""}<a href={"https://scikit-learn.org/stable/modules/semi_supervised.html"}>{"scikit-learn's semi-supervised guide"}</a>{" and "}<a href={"https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html"}>{"LabelSpreading API"}</a>{": implementation vocabulary, unlabeled encoding, parameters and the distinction between fitted-pool transduction and new-input prediction."}</li>
<li>{""}<a href={"https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf"}>{"Blum and Mitchell's co-training paper"}</a>{": read the two-view setup before its theorem. It explains why representation assumptions are more demanding than simply having two models."}</li>
<li>{""}<a href={"https://academic.oup.com/mit-press-scholarship-online/book/41571"}>{"Chapelle, Schölkopf and Zien's edited book"}</a>{": the contents map generative, low-density, graph, representation and practical families. This is a deeper reference, with full-text access depending on availability."}</li></ul>

<Prose>{"The next module topic is "}<strong>{"Active Learning"}</strong>{". Instead of allowing a model to supply every new label, it asks which examples a human or other labeling oracle should label next. The graph's unanchored island, co-training's conflict and self-training's uncertain boundary are useful reasons to investigate a query; they are not yet guarantees that a query will be valuable."}</Prose>
  </div>,
};
export default semiSupervisedLesson;
