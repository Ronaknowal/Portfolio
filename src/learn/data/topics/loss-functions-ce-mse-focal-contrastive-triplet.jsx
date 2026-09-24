// Complete prepared manuscript rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { LossUpdateFigure, RegressionInfluenceLab, FocalContributionLab, LossDecisionLab, TripletGeometryLab, InfoNceLab, LossScalingFigure } from '../../components/lesson-labs/LossFunctionsLabs.jsx';
import NeuralProgram from '../../components/lesson-labs/NeuralProgram.jsx';
export default {
  title: 'Loss Functions (CE, MSE, Focal, Contrastive, Triplet)',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson loss-functions-lesson">
  <LessonIntro prerequisites="A model maps inputs to outputs; backpropagation propagates derivatives. Residuals, probabilities and embedding coordinates are introduced locally." sections={[["from-a-prediction-to-an-update","From a prediction to an update"],["regression-what-should-a-typical-answer-mean","Regression: what should “a typical answer” mean?"],["classification-confidence-is-part-of-the-answer","Classification: confidence is part of the answer"],["imbalance-inspect-contributions-before-changing-the-objective","Imbalance: inspect contributions before changing the objective"],["one-real-experiment-recognize-a-nine","One real experiment: recognize a nine"],["when-the-output-is-a-location-pair-and-triplet-losses","When the output is a location: pair and triplet losses"],["infonce-finding-the-correct-candidate-is-classification","InfoNCE: finding the correct candidate is classification"],["practical-reductions-scaling-and-diagnosis","Practical reductions, scaling, and diagnosis"],["build-the-objectives-then-control-the-library","Build the objectives, then control the library"],["practice-change-the-problem-then-explain-the-consequence","Practice: change the problem, then explain the consequence"],["another-way-to-learn-and-the-next-connection","Another way to learn, and the next connection"]]}>Choose what errors should change, trace their gradients, then judge the decisions produced by a real model.</LessonIntro>
  <Prose>{""}<strong>{"Explore as you read."}</strong>{" Move observations, change the loss and focal gamma, drag decision thresholds, edit pair/triplet coordinates and change InfoNCE temperature. Update loss, signed gradients, fitted location, confusion counts, eligible negatives and candidate probabilities together. Keep score-based metrics distinct from threshold decisions. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose an objective or operating threshold from the error tradeoff rather than from a single loss number."}</Prose>

<Prose>{"A learning algorithm needs more than examples of correct answers. It needs a way to say how an imperfect answer should change. Predicting a delivery ten minutes late, assigning a wrong label with 99% confidence, and retrieving the wrong photograph are different failures. A "}<strong>{"loss function"}</strong>{" assigns a numerical penalty to a prediction and its target. Its derivatives tell backpropagation how that penalty responds to changes in the model."}</Prose>

<Prose>{"The preceding lesson explained how to compute those derivatives. Here we choose what to differentiate. A small loss is useful only when the objective represents the behavior we need."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow the prediction-to-update map, work the regression and probability examples, investigate focal loss, then build the pair→triplet→candidate-selection connection. Run the small digit experiment and attempt the practice before its solutions. The likelihood, angular-margin, and mutual-information branches add depth; their derivations are not prerequisites for the first experiment."}</Prose>

<H2>{"From a prediction to an update"}</H2>

<Prose>{"Take a prediction "}<InlineMath>{"\\hat y=3"}</InlineMath>{" for a measured value "}<InlineMath>{"y=5"}</InlineMath>{". Squared error gives "}<InlineMath>{"L=(3-5)^2=4"}</InlineMath>{". Its derivative with respect to the prediction is "}<InlineMath>{"2(\\hat y-y)=-4"}</InlineMath>{": increasing the prediction slightly decreases loss. If "}<InlineMath>{"\\hat y=wx+b"}</InlineMath>{", backpropagation continues with "}<InlineMath>{"\\partial L/\\partial w=-4x"}</InlineMath>{" and "}<InlineMath>{"\\partial L/\\partial b=-4"}</InlineMath>{". The optimizer then decides the step size."}</Prose>

<LossUpdateFigure />

<Prose>{"For "}<InlineMath>{"B"}</InlineMath>{" examples, a common training objective is"}</Prose>

<div className="neural-equation"><MathBlock>{"J(\\theta)=\\frac1B\\sum_{i=1}^{B}L(f_\\theta(x_i),y_i)+\\lambda R(\\theta)."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"f_\\theta"}</InlineMath>{" is the model, "}<InlineMath>{"\\theta"}</InlineMath>{" its parameters, "}<InlineMath>{"R"}</InlineMath>{" a regularization penalty, and "}<InlineMath>{"\\lambda"}</InlineMath>{" its strength. Architecture, data, and sampling also shape what is learned. The loss is one part of that system."}</Prose>

<Prose>{"The quantity reported to users can differ from the training objective. A classifier can minimize differentiable cross-entropy and be assessed using recall, a confusion matrix, and a decision cost. We choose a smooth surrogate because a count of correct labels is flat over most small parameter changes. We must still evaluate the intended outcome."}</Prose>

<H2>{"Regression: what should “a typical answer” mean?"}</H2>

<Prose>{"Let the residual be "}<InlineMath>{"r=\\hat y-y"}</InlineMath>{". Four useful penalties produce different responses:"}</Prose>

<NeuralTable caption={"Regression: what should “a typical answer” mean?"} headers={[<>{"Penalty"}</>,<>{"Per-example formula"}</>,<>{"Derivative with respect to "}<InlineMath>{"\\hat y"}</InlineMath>{", away from corners"}</>,<>{"What it emphasizes"}</>]} rows={[[<>{"Squared error"}</>,<>{""}<InlineMath>{"r^2"}</InlineMath>{""}</>,<>{""}<InlineMath>{"2r"}</InlineMath>{""}</>,<>{"Large numerical errors"}</>],[<>{"Absolute error"}</>,<>{""}<InlineMath>{"|r|"}</InlineMath>{""}</>,<>{""}<InlineMath>{"\\operatorname{sign}(r)"}</InlineMath>{""}</>,<>{"An error's direction, with bounded magnitude"}</>],[<>{"Huber"}</>,<>{""}<InlineMath>{"r^2/2"}</InlineMath>{" if "}<InlineMath>{"|r|\\le\\delta"}</InlineMath>{"; "}<InlineMath>{"\\delta(|r|-\\delta/2)"}</InlineMath>{" otherwise"}</>,<>{""}<InlineMath>{"r"}</InlineMath>{" inside; "}<InlineMath>{"\\delta\\operatorname{sign}(r)"}</InlineMath>{" outside"}</>,<>{"Smooth small-error correction, bounded large-error slope"}</>],[<>{"Quantile, level "}<InlineMath>{"q"}</InlineMath>{""}</>,<>{""}<InlineMath>{"q(y-\\hat y)"}</InlineMath>{" if "}<InlineMath>{"y\\ge\\hat y"}</InlineMath>{"; "}<InlineMath>{"(1-q)(\\hat y-y)"}</InlineMath>{" otherwise"}</>,<>{""}<InlineMath>{"-q"}</InlineMath>{" below the observation; "}<InlineMath>{"1-q"}</InlineMath>{" above it"}</>,<>{"A chosen asymmetric underprediction/overprediction balance"}</>]]} />

<Prose>{"MSE means the "}<strong>{"mean"}</strong>{" of squared errors. At residuals one and ten, squared-error losses are one and one hundred; prediction-gradient magnitudes are two and twenty. The loss ratio and gradient ratio are different."}</Prose>

<Prose>{"Consider a constant predictor for seven measurements: "}<InlineMath>{"0,0,0,0,0,0,10"}</InlineMath>{". MSE is minimized at the arithmetic mean "}<InlineMath>{"10/7\\approx1.429"}</InlineMath>{". MAE is minimized at the median, zero. For Huber with "}<InlineMath>{"\\delta=1"}</InlineMath>{", the optimum is "}<InlineMath>{"1/6"}</InlineMath>{": six small residuals contribute derivative "}<InlineMath>{"6c"}</InlineMath>{", the large residual contributes "}<InlineMath>{"-1"}</InlineMath>{", and "}<InlineMath>{"6c-1=0"}</InlineMath>{"."}</Prose>

<Prose>{""}<strong>{"Investigation — move one measurement:"}</strong>{" change the last measurement from 10 to 100 and watch the three fitted constants. Inspect each point's contribution and compare the retained baseline. MSE's optimum becomes "}<InlineMath>{"100/7"}</InlineMath>{"; the MAE and Huber optima stay at zero and "}<InlineMath>{"1/6"}</InlineMath>{". Replacing all measurements by three is a useful contrast: all three losses agree on three."}</Prose>

<RegressionInfluenceLab />

<Prose>{"This is not permission to delete a troublesome observation. A rare large value can be the event the application must predict. Check the measurement and choose the estimand: the mean for expected total cost, a median for a typical case, or a high quantile for a capacity target. Huber reduces sensitivity to large residuals; it does not decide whether those residuals are mistakes."}</Prose>

<H3>{"Deeper: why squared error estimates a mean"}</H3>

<Prose>{"For a random target "}<InlineMath>{"Y"}</InlineMath>{", conditioning on the available input "}<InlineMath>{"x"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[(Y-c)^2\\mid x]=\\operatorname{Var}(Y\\mid x)+(\\mathbb E[Y\\mid x]-c)^2."}</MathBlock></div>

<Prose>{"The variance term does not depend on "}<InlineMath>{"c"}</InlineMath>{". With finite second moments, the optimal constant is the conditional mean, without any Gaussian assumption. Similarly, an absolute-error optimum is a conditional median; a quantile-loss optimum is a conditional quantile, possibly nonunique for discrete distributions."}</Prose>

<Prose>{"A likelihood interpretation adds an explicit probability model. Under independent Gaussian errors with fixed common variance "}<InlineMath>{"\\sigma^2"}</InlineMath>{", negative log likelihood is a constant plus "}<InlineMath>{"\\sum r_i^2/(2\\sigma^2)"}</InlineMath>{". Its optimizer matches squared error. Fixed-scale Laplace errors give absolute error. If the model also learns a different "}<InlineMath>{"\\sigma(x)"}</InlineMath>{" for each input, the Gaussian objective includes both "}<InlineMath>{"r^2/(2\\sigma(x)^2)"}</InlineMath>{" and "}<InlineMath>{"\\log\\sigma(x)"}</InlineMath>{"; omitting the latter rewards inflating uncertainty indefinitely."}</Prose>

<Prose>{"For a practical nonstandard application, a service can predict the 90th percentile of demand to plan reserve capacity. That prediction is deliberately above the median. A quantile of 0.9 corresponds to nine times the local penalty slope for underprediction as for overprediction. The operational cost ratio, not a desire for uniformly high predictions, motivates that choice."}</Prose>

<H2>{"Classification: confidence is part of the answer"}</H2>

<Prose>{"For a binary label "}<InlineMath>{"y\\in\\{0,1\\}"}</InlineMath>{", the model produces a real "}<strong>{"logit"}</strong>{" "}<InlineMath>{"z"}</InlineMath>{". The sigmoid "}<InlineMath>{"p=1/(1+e^{-z})"}</InlineMath>{" converts it into a number between zero and one. Binary cross-entropy is"}</Prose>

<div className="neural-equation"><MathBlock>{"L=-y\\log p-(1-y)\\log(1-p)."}</MathBlock></div>

<Prose>{"Only one term remains for a hard label. If the correct label is one, assigning probabilities 0.9, 0.5, and 0.1 gives losses approximately 0.1054, 0.6931, and 2.3026. Correct but uncertain and confidently wrong predictions receive different penalties. Natural logarithms give units of "}<strong>{"nats"}</strong>{"."}</Prose>

<Prose>{"For mutually exclusive classes, use one logit per class and"}</Prose>

<div className="neural-equation"><MathBlock>{"p_c=\\frac{e^{z_c}}{\\sum_j e^{z_j}},\\qquad\nL=-\\log p_y=\\operatorname{logsumexp}(z)-z_y."}</MathBlock></div>

<Prose>{"Differentiate: "}<InlineMath>{"\\partial L/\\partial z_c=p_c-\\mathbf1[c=y]"}</InlineMath>{". The correct class receives a negative derivative unless its probability is already one; other classes receive positive derivatives. This is the output gradient used by the previous lesson's engine."}</Prose>

<Prose>{"Cross-entropy is also negative log likelihood for the observed categorical outcome. At a population level, expected cross-entropy decomposes as "}<InlineMath>{"H(q,p)=H(q)+D_{\\mathrm{KL}}(q\\Vert p)"}</InlineMath>{", where "}<InlineMath>{"q"}</InlineMath>{" is the target distribution. The irreducible entropy "}<InlineMath>{"H(q)"}</InlineMath>{" remains even when probabilities are correct. Extra expected coding cost from using the wrong distribution is the KL term. "}<a href={"https://cs231n.github.io/linear-classify/#softmax"}>{"Stanford CS231n's softmax discussion"}</a>{" provides an alternate derivation."}</Prose>

<H3>{"Stable logits and explicit targets"}</H3>

<Prose>{"Avoid calculating a tiny softmax probability and then taking its logarithm. For logits "}<InlineMath>{"[1000,-1000]"}</InlineMath>{" and correct index one, the loss is approximately 2000, not infinity. Compute log-sum-exp with a maximum shift. Binary CE has the stable form "}<InlineMath>{"\\max(z,0)-yz+\\log(1+e^{-|z|})"}</InlineMath>{"."}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch.nn import functional as F\n\nlogits = torch.tensor([[1000., -1000.], [1., 2.]], dtype=torch.float64)\nclass_indices = torch.tensor([1, 0], dtype=torch.long)\nprint(F.cross_entropy(logits, class_indices))  # about 1000.656631\nbinary_logits = torch.tensor([-2., 1.5, .2, -.4])\nbinary_targets = torch.tensor([0., 1., 1., 0.])\nprint(F.binary_cross_entropy_with_logits(binary_logits, binary_targets))"}</CodeBlock>

<Prose>{"These complete API examples are verified during implementation, with their actual outputs available in the downloadable execution record. At the 0.5 threshold, all four binary examples are classified correctly, although their confidences differ."}</Prose>

<Prose>{"Multilabel classification is different from multiclass classification: an image can have both “outdoors” and “vehicle.” Use independent binary targets and logits for those labels when that matches the task, not one softmax that forces exactly one category. Neither output format by itself guarantees calibrated probabilities."}</Prose>

<Prose>{"With a probability target "}<InlineMath>{"t_c"}</InlineMath>{", CE becomes "}<InlineMath>{"-\\sum_ct_c\\log p_c"}</InlineMath>{", with gradient "}<InlineMath>{"p_c-t_c"}</InlineMath>{" for a normalized unweighted target. PyTorch label smoothing uses "}<InlineMath>{"t=(1-\\epsilon)\\,\\text{one-hot}+\\epsilon/C"}</InlineMath>{"; a different convention allocates smoothing only to incorrect classes. State the convention. For "}<InlineMath>{"C=3,\\epsilon=.1"}</InlineMath>{", PyTorch's target is "}<InlineMath>{"[.93333,.03333,.03333]"}</InlineMath>{" when class zero is correct, not "}<InlineMath>{"[.9,.05,.05]"}</InlineMath>{". Smoothing changes the desired probabilities; its effect on measured calibration depends on the model and evaluation. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"}>{"CrossEntropyLoss's API contract"}</a>{" specifies targets, shapes, weights, and reductions."}</Prose>

<H2>{"Imbalance: inspect contributions before changing the objective"}</H2>

<Prose>{"An always-negative classifier scores 90% accuracy on a task with 10% positives. That baseline should trigger inspection of recall and probability quality. It does "}<strong>{"not"}</strong>{" establish that BCE cannot learn rare positives. A constant-only model minimizes BCE by predicting prevalence; a model with informative inputs can do better."}</Prose>

<Prose>{"A positive-term weight "}<InlineMath>{"r"}</InlineMath>{" gives"}</Prose>

<div className="neural-equation"><MathBlock>{"L_r=-r\\,y\\log p-(1-y)\\log(1-p)."}</MathBlock></div>

<Prose>{"For a location with actual positive probability "}<InlineMath>{"\\eta"}</InlineMath>{", minimizing its expected weighted loss yields"}</Prose>

<div className="neural-equation"><MathBlock>{"p^*=\\frac{r\\eta}{r\\eta+1-\\eta}."}</MathBlock></div>

<Prose>{"For "}<InlineMath>{"\\eta=.1,r=9"}</InlineMath>{", this is .5. The output now represents a changed cost balance, not automatically the original probability .1. In the ideal unrestricted population solution, subtracting "}<InlineMath>{"\\log r"}</InlineMath>{" from the fitted logit recovers the unweighted log-odds. Finite-data models need validation; this is not a guaranteed calibration repair."}</Prose>

<Prose>{"The ratio of negative to positive training counts balances aggregate class weights. It is a candidate choice, not a universal optimum. Changing the decision threshold is another intervention that leaves the learned ranking intact. Resampling changes the training distribution and can interact with weighting; combining both without accounting for the changes can unintentionally count the same preference twice."}</Prose>

<H3>{"Focal loss changes emphasis as confidence changes"}</H3>

<Prose>{"Define "}<InlineMath>{"p_t=p"}</InlineMath>{" when "}<InlineMath>{"y=1"}</InlineMath>{", otherwise "}<InlineMath>{"p_t=1-p"}</InlineMath>{". Unweighted focal loss is"}</Prose>

<div className="neural-equation"><MathBlock>{"L_{\\mathrm{focal}}=-(1-p_t)^\\gamma\\log p_t,\\qquad \\gamma\\ge0."}</MathBlock></div>

<Prose>{"At "}<InlineMath>{"\\gamma=0"}</InlineMath>{", this is CE. At "}<InlineMath>{"\\gamma=2,p_t=.9"}</InlineMath>{", the loss is multiplied by .01. But the multiplier also depends on the model. Differentiating the product gives a gradient about .028965 times the CE gradient, "}<strong>{"not .01 times"}</strong>{"."}</Prose>

<Prose>{"Let "}<InlineMath>{"s=2y-1"}</InlineMath>{", so "}<InlineMath>{"p_t=\\sigma(sz)"}</InlineMath>{". Its derivative is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial z}\n=s(1-p_t)^\\gamma\\left[\\gamma p_t\\log p_t-(1-p_t)\\right]."}</MathBlock></div>

<Prose>{"This formula connects the moving multiplier to backpropagation. A class factor "}<InlineMath>{"\\alpha_t"}</InlineMath>{", if used, multiplies both the loss and gradient. With the convention "}<InlineMath>{"\\alpha_t=\\alpha"}</InlineMath>{" for positives and "}<InlineMath>{"1-\\alpha"}</InlineMath>{" for negatives, setting "}<InlineMath>{"\\alpha=1"}</InlineMath>{" discards negatives. To recover ordinary BCE at "}<InlineMath>{"\\gamma=0"}</InlineMath>{", omit class weighting rather than setting that "}<InlineMath>{"\\alpha"}</InlineMath>{" to one."}</Prose>

<Prose>{""}<strong>{"Investigation — who moves a shared bias?"}</strong>{" Use 1000 negatives each with predicted positive probability .01, and one positive with probability .1. For an additive bias shared across the examples, sum each logit's gradient. Under CE the negatives contribute +10 and the positive contributes −.9. Under unweighted focal loss with "}<InlineMath>{"\\gamma=2"}</InlineMath>{", negative contributions shrink to about +.002990 and the positive contributes about −1.102019. The net direction reverses. Display per-example and summed gradients, not just attractive loss curves. These are an intentionally constructed mechanism example, not observed digit probabilities."}</Prose>

<Prose>{"Focal loss was introduced for the many easy background candidates in dense object detection; its paper's preferred hyperparameters are results for that setting. Hard examples may include label errors, so emphasizing them is not a general robustness strategy. "}<a href={"https://arxiv.org/html/1708.02002v2#S3"}>{"Lin et al., §3"}</a>{" explains that distinction. Focal confidence scores also need evaluation as probabilities: theoretical classification consistency does not imply strict propriety as a probability score. "}<a href={"https://arxiv.org/abs/2011.09172"}>{"Charoenphakdee et al."}</a>{" studies this explicitly."}</Prose>

<FocalContributionLab />

<H2>{"One real experiment: recognize a nine"}</H2>

<Prose>{"The downloadable "}<a href={"/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/loss-experiments.py"}>{"loss-experiments.py"}</a>{" and "}<a href={"/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/digits-400.csv"}>{"digits-400.csv"}</a>{" run entirely on a CPU after dependencies are installed. The data contains 40 real 8×8 handwriting images per digit from UCI Optical Recognition of Handwritten Digits. It is not MNIST. "}<a href={"/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/data-provenance.md"}>{"Data provenance"}</a>{" explains the source, license, row selection, and split limitations."}</Prose>

<Prose>{"We ask a binary question: “Is this digit nine?” There are 40 positives and 360 negatives. Split the specimens into 280 training and 120 validation rows, stratifying on the original digit with seed 22. Divide pixels by their documented maximum 16. Fit the same 64-input linear logit using BCE, positive-weight-nine BCE, or unweighted focal loss with "}<InlineMath>{"\\gamma=2"}</InlineMath>{". Each gets Adam at .03 for 400 full-batch updates. Reset the initial seed for every objective; repeat seeds one, two, and three."}</Prose>

<Prose>{"The code does not use validation examples in gradients or choose a winner. This small comparison measures the declared protocol. It has no separate final test and does not establish writer-independent handwriting performance."}</Prose>

<CodeBlock language={"text"}>{"python -m venv .venv\n.venv\\Scripts\\python -m pip install numpy==2.3.5 torch==2.14.0 scikit-learn==1.9.1\n.venv\\Scripts\\python loss-experiments.py"}</CodeBlock>

<Prose>{"On macOS/Linux, use "}<code>{".venv/bin/python"}</code>{". Put the CSV beside the program. Installation needs network access; training uses only the included data. The program writes "}<code>{"calculated-inputs.json"}</code>{", containing split IDs, training traces, validation probabilities, metrics, and mathematical fixtures. The author's run used Python 3.12.14 and torch 2.14.0+cpu."}</Prose>

<Prose>{"Read the implementation in three parts. "}<code>{"binary_focal"}</code>{" computes "}<strong>{"unreduced"}</strong>{" per-example losses from logits, retaining the focusing factor in the differentiation graph. "}<code>{"digit_experiment"}</code>{" performs the fixed split and nine fits. Its evaluation uses the same unweighted metrics across objectives, since raw training loss values from different formulas are not directly comparable."}</Prose>

<NeuralProgram topic="loss" /><p><a href="/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/recorded-output.txt">Recorded full-program output</a> · <a href="/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/snippet-output.txt">API snippet outputs</a></p>

<NeuralTable caption={"One real experiment: recognize a nine"} headers={[<>{"Seed"}</>,<>{"Objective"}</>,<>{"False positives"}</>,<>{"False negatives"}</>,<>{"Average precision"}</>,<>{"Brier score"}</>,<>{"Unweighted log loss"}</>]} rows={[[<>{"1"}</>,<>{"BCE"}</>,<>{"1"}</>,<>{"2"}</>,<>{".979070"}</>,<>{".012382"}</>,<>{".042302"}</>],[<>{"1"}</>,<>{"Positive weight 9"}</>,<>{"1"}</>,<>{"1"}</>,<>{".976389"}</>,<>{".013921"}</>,<>{".047416"}</>],[<>{"1"}</>,<>{"Focal, "}<InlineMath>{"\\gamma=2"}</InlineMath>{""}</>,<>{"1"}</>,<>{"2"}</>,<>{".979070"}</>,<>{".017337"}</>,<>{".085298"}</>],[<>{"2"}</>,<>{"BCE"}</>,<>{"1"}</>,<>{"2"}</>,<>{".979070"}</>,<>{".011961"}</>,<>{".043178"}</>],[<>{"2"}</>,<>{"Positive weight 9"}</>,<>{"1"}</>,<>{"1"}</>,<>{".986645"}</>,<>{".011386"}</>,<>{".042535"}</>],[<>{"2"}</>,<>{"Focal, "}<InlineMath>{"\\gamma=2"}</InlineMath>{""}</>,<>{"1"}</>,<>{"1"}</>,<>{".979070"}</>,<>{".019137"}</>,<>{".097343"}</>],[<>{"3"}</>,<>{"BCE"}</>,<>{"1"}</>,<>{"2"}</>,<>{".979070"}</>,<>{".012168"}</>,<>{".040372"}</>],[<>{"3"}</>,<>{"Positive weight 9"}</>,<>{"1"}</>,<>{"0"}</>,<>{".979070"}</>,<>{".010817"}</>,<>{".042596"}</>],[<>{"3"}</>,<>{"Focal, "}<InlineMath>{"\\gamma=2"}</InlineMath>{""}</>,<>{"1"}</>,<>{"1"}</>,<>{".979070"}</>,<>{".015298"}</>,<>{".073454"}</>]]} />

<Prose>{"These are executed results. There are 108 negatives and 12 positives in validation. At threshold .5, BCE's recall is "}<InlineMath>{"10/12"}</InlineMath>{", with specificity "}<InlineMath>{"107/108"}</InlineMath>{"; its balanced accuracy is the average of those two fractions, about .9120. A single positive changes recall by "}<InlineMath>{"1/12"}</InlineMath>{", so small count changes deserve restraint."}</Prose>

<Prose>{"Average precision summarizes the precision-recall ranking, using recall increments to weight precision; it is not an unspecified trapezoidal PR area. Brier score is mean squared probability error and log loss penalizes confident mistakes strongly. Lower is better for the last two columns; higher is better for average precision. The focal run's similar ranking and worse probability scores show why one metric cannot answer every question. None of these observations proves that a different learning rate, model, or loss variant would behave the same way."}</Prose>

<LossDecisionLab />

<H2>{"When the output is a location: pair and triplet losses"}</H2>

<Prose>{"An "}<strong>{"embedding"}</strong>{" is a vector representing an item. Two recordings of the same machine state or two photographs of the same object may need nearby vectors even when their raw inputs differ. A shared encoder converts each input into coordinates; a similarity loss teaches relationships between those coordinates."}</Prose>

<Prose>{"For a pair, let "}<InlineMath>{"D=\\|a-b\\|_2"}</InlineMath>{", and use "}<InlineMath>{"y=1"}</InlineMath>{" for a matching pair. One explicit contrastive convention is"}</Prose>

<div className="neural-equation"><MathBlock>{"L_{\\mathrm{pair}}=yD^2+(1-y)\\max(0,m-D)^2."}</MathBlock></div>

<Prose>{"Matching items are pulled together; nonmatching items receive a penalty only while they are closer than margin "}<InlineMath>{"m"}</InlineMath>{". Some papers reverse the label convention or include a factor one-half. Convert labels and prefactors before comparing code. The original pair-learning research is "}<a href={"https://yann.lecun.com/exdb/publis/"}>{"Hadsell, Chopra, and LeCun"}</a>{"."}</Prose>

<Prose>{"At "}<InlineMath>{"m=1,D=.2"}</InlineMath>{", matching and nonmatching losses are .04 and .64. At "}<InlineMath>{"D=1.2"}</InlineMath>{", the nonmatching loss is zero. The zero-distance corner deserves care: the Euclidean norm has no unique derivative direction there. A library's zero-gradient convention can leave coincident negative embeddings stuck even though their loss is positive."}</Prose>

<Prose>{"A triplet instead says “this positive should be closer than this negative.” Using "}<strong>{"squared"}</strong>{" distances,"}</Prose>

<div className="neural-equation"><MathBlock>{"L_{\\mathrm{triplet}}=\\max(0,\\|a-p\\|^2-\\|a-n\\|^2+\\alpha)."}</MathBlock></div>

<Prose>{"The margin has squared-coordinate units. For "}<InlineMath>{"a=(0,0),p=(1,0),\\alpha=1"}</InlineMath>{", negatives at "}<InlineMath>{"(.5,0),(1.2,0),(2,0)"}</InlineMath>{" give losses 1.75, .56, and zero. They are hard, semi-hard, and easy respectively. Semi-hard means"}</Prose>

<div className="neural-equation"><MathBlock>{"D_{ap}^2<D_{an}^2<D_{ap}^2+\\alpha."}</MathBlock></div>

<Prose>{"An easy triplet supplies no local gradient; it remains useful as evidence that this particular constraint is satisfied. "}<a href={"https://arxiv.org/pdf/1503.03832"}>{"FaceNet, §3.1–3.2"}</a>{" motivates squared distances and explains the role of triplet selection."}</Prose>

<Prose>{""}<strong>{"Investigation — choose a useful negative:"}</strong>{" move the three candidate points and inspect which is eligible under a stated mining rule. Our program selects the nearest strictly semi-hard candidate, breaking ties by row order; it skips the anchor if none exists. This is a transparent teaching policy, not a universal best miner. A no-candidate result should be displayed as “skipped,” never silently substituted with a zero-loss example."}</Prose>

<TripletGeometryLab />

<Prose>{"Inside the active region, derivatives are "}<InlineMath>{"2(n-p)"}</InlineMath>{" for the anchor, "}<InlineMath>{"2(p-a)"}</InlineMath>{" for the positive, and "}<InlineMath>{"2(a-n)"}</InlineMath>{" for the negative. Take a small joint step and recompute both distances. Describing the gradients as attraction and repulsion is helpful, but a large finite step is not guaranteed to improve all desired distances."}</Prose>

<Prose>{"At "}<InlineMath>{"a=p=n"}</InlineMath>{", squared-triplet loss with positive margin is "}<InlineMath>{"\\alpha"}</InlineMath>{", yet every derivative above is zero. Positive margin makes complete collapse costly but does not make it impossible. Unit normalization prevents the zero vector from being a valid unit vector, but every item can still collapse onto the same nonzero unit vector. Inspect embedding spread, norms, pair labels, active constraints, and retrieval outcomes together."}</Prose>

<H3>{"Match the distance used by the API"}</H3>

<Prose>{"PyTorch "}<code>{"TripletMarginLoss(p=2)"}</code>{" uses unsquared Euclidean distances, with an epsilon convention. It does not directly match the squared FaceNet formula. Our "}<code>{"squared_triplet"}</code>{" implements that formula explicitly; "}<code>{"TripletMarginWithDistanceLoss"}</code>{" can also accept a squared-distance function. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginLoss.html"}>{"The API documentation"}</a>{" makes this distinction reviewable."}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch.nn import functional as F\n\na = torch.tensor([[0., 0.]])\np = torch.tensor([[1., 0.]])\nn = torch.tensor([[1.2, 0.]])\nsquared = ((a-p).square().sum(-1) - (a-n).square().sum(-1) + 1).clamp_min(0)\nunsquared = (torch.linalg.vector_norm(a-p, dim=-1)\n             - torch.linalg.vector_norm(a-n, dim=-1) + 1).clamp_min(0)\nprint(squared, unsquared)  # derived: approximately .56 and .8"}</CodeBlock>

<Prose>{"This changes more than notation. Copying the same numerical margin between distance definitions changes which examples are active."}</Prose>

<H2>{"InfoNCE: finding the correct candidate is classification"}</H2>

<Prose>{"For one query "}<InlineMath>{"q"}</InlineMath>{", suppose there is one designated positive key and "}<InlineMath>{"K"}</InlineMath>{" designated negative keys. Convert their similarities "}<InlineMath>{"s_j"}</InlineMath>{" into logits "}<InlineMath>{"s_j/\\tau"}</InlineMath>{", where temperature "}<InlineMath>{"\\tau>0"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"L=-\\log\\frac{\\exp(s_+/\\tau)}{\\sum_{j=0}^{K}\\exp(s_j/\\tau)}."}</MathBlock></div>

<Prose>{"This is ordinary categorical CE over "}<strong>{"candidate items"}</strong>{", rather than over class names. For cosine similarity, use nonzero vectors normalized to unit length. Normalization is a design choice that makes the geometry angular; general InfoNCE does not mathematically require it. A temperature of one is also a valid choice, not an absent parameter that makes the objective invalid."}</Prose>

<InfoNceLab />

<Prose>{"For "}<InlineMath>{"N=K+1"}</InlineMath>{" equal candidates, loss is "}<InlineMath>{"\\log N"}</InlineMath>{". This is the uniform baseline, not an upper bound. A positive that receives much less probability than "}<InlineMath>{"1/N"}</InlineMath>{" has larger loss."}</Prose>

<Prose>{"The program's "}<code>{"paired_info_nce"}</code>{" uses a "}<InlineMath>{"B\\times B"}</InlineMath>{" score matrix with matching query/key rows as positives. A row's remaining keys are designated negatives. It is one-way paired learning. SimCLR constructs two views of each input, excludes each view's self-comparison, and averages both positive directions among "}<InlineMath>{"2B"}</InlineMath>{" views. Those masks define different candidate sets. "}<a href={"https://arxiv.org/pdf/2002.05709"}>{"SimCLR's method and Algorithm 1"}</a>{" are useful to inspect after this simpler matrix."}</Prose>

<Prose>{"A same-class key may be a "}<strong>{"false negative"}</strong>{" for the intended task. If two candidate keys are identical and only one is designated positive, no scoring function can give that positive more than half their combined probability; even if every other key becomes irrelevant, loss cannot fall below "}<InlineMath>{"\\log2"}</InlineMath>{". This is different from an ID or timestamp shortcut that lets training loss approach zero without learning useful semantics."}</Prose>

<Prose>{"When multiple items really are positive, one option is supervised contrastive learning: average the negative log probability assigned to each positive within the candidate set. Keep self-comparisons out and define what happens when an anchor has no positive. An average of log probabilities differs from taking the log of the summed positive probability. "}<a href={"https://arxiv.org/pdf/2004.11362"}>{"Khosla et al., §3.2"}</a>{" compares these formulations."}</Prose>

<H3>{"Deeper connections and useful boundaries"}</H3>

<Prose>{"For exact nonzero unit vectors, "}<InlineMath>{"\\|a-b\\|^2=2-2a^\\top b"}</InlineMath>{". Therefore squared-triplet constraints can be written with cosine similarities. They do not become identical to CE over candidates: margins, candidate weighting, and gradients still differ. As "}<InlineMath>{"\\tau\\to0"}</InlineMath>{", "}<strong>{""}<InlineMath>{"\\tau L"}</InlineMath>{""}</strong>{" tends to "}<InlineMath>{"\\max_j s_j-s_+"}</InlineMath>{". The unscaled loss can diverge when a negative wins, or retain a log-tie penalty. Temperature is not an explicit triplet margin."}</Prose>

<Prose>{"The CPC derivation connects expected InfoNCE with a mutual-information lower bound "}<InlineMath>{"I\\ge\\log N-L_N"}</InlineMath>{", under its joint-positive and marginal-negative sampling assumptions. "}<InlineMath>{"N"}</InlineMath>{" counts all candidates, including the positive. Increasing "}<InlineMath>{"N"}</InlineMath>{" also changes "}<InlineMath>{"L_N"}</InlineMath>{"; simply adding "}<InlineMath>{"\\log2"}</InlineMath>{" to a claimed bound without reevaluating the loss is unjustified. "}<a href={"https://arxiv.org/pdf/1807.03748"}>{"CPC §2.3"}</a>{" gives the assumptions and density-ratio interpretation. Useful representations should still be assessed on retrieval or downstream prediction."}</Prose>

<Prose>{"An angular-margin classifier such as ArcFace normalizes features and class weights, scales their cosine logits, and modifies the target logit to "}<InlineMath>{"s\\cos(\\theta_y+m)"}</InlineMath>{" during training. This directly shapes angular separation; the scalar "}<InlineMath>{"m"}</InlineMath>{" is an angle, unlike a squared-triplet margin. The purpose is an embedding useful beyond the training class head, not a guarantee that any cosine margin wins. The "}<a href={"https://arxiv.org/pdf/1801.07698v3"}>{"ArcFace paper"}</a>{" is a deeper application, after the pair and candidate geometry are secure."}</Prose>

<H2>{"Practical reductions, scaling, and diagnosis"}</H2>

<Prose>{"Always specify what receives one vote. Averaging per pixel, per sequence, or per example can produce different objectives. With a mask "}<InlineMath>{"m_i"}</InlineMath>{", an explicit masked mean is "}<InlineMath>{"\\sum_i m_i L_i/\\sum_i m_i"}</InlineMath>{", with an intentional policy for an empty mask. A long sequence should not accidentally carry more weight merely because its loss was summed while another term was averaged."}</Prose>

<Prose>{"Class-index weighted PyTorch CE divides a mean by the sum of included target weights; CE with probability targets uses a per-observation mean. Binary "}<code>{"pos_weight"}</code>{" multiplies positive terms but its mean still divides by the number of elements. Our focal loss returns a vector and explicitly averages its elements. These denominators matter when comparing gradients or combining losses."}</Prose>

<Prose>{"Full CE over "}<InlineMath>{"C"}</InlineMath>{" already-computed logits costs order "}<InlineMath>{"BC"}</InlineMath>{"; a dense head mapping "}<InlineMath>{"D"}</InlineMath>{" features into those logits costs order "}<InlineMath>{"BDC"}</InlineMath>{". A contrastive "}<InlineMath>{"B\\times B"}</InlineMath>{" similarity matrix costs order "}<InlineMath>{"B^2D"}</InlineMath>{" and stores "}<InlineMath>{"B^2"}</InlineMath>{" scores. For "}<InlineMath>{"B=1024"}</InlineMath>{", float32 scores alone use 4 MiB; at 4096 they use 64 MiB, before gradients and encoder activations. These are dimensional calculations, not runtime benchmarks."}</Prose>

<LossScalingFigure />

<Prose>{"Batch-hard mining can share a pairwise distance matrix of order "}<InlineMath>{"B^2D"}</InlineMath>{", then use label masks and row reductions. Enumerating every possible dataset triplet is unnecessary. Sampled-softmax and noise-contrastive approaches change how candidate normalization is estimated; hierarchical softmax changes the factorization. Their bias, sampling corrections, and inference behavior need their own treatment before substitution. Ordinary gradient accumulation does not create similarities between separate microbatches; enlarging the negative set requires retaining or gathering the relevant embeddings."}</Prose>

<Prose>{"Use the observed failure to choose the next check:"}</Prose>

<NeuralTable caption={"Practical reductions, scaling, and diagnosis"} headers={[<>{"Observation"}</>,<>{"First useful inspection"}</>,<>{"What it does not prove"}</>]} rows={[[<>{"High accuracy, no positive predictions"}</>,<>{"Class counts, ranking, probabilities, threshold, recall"}</>,<>{"CE cannot learn an imbalanced task"}</>],[<>{"Large loss from a few residuals"}</>,<>{"Units, measurement validity, desired estimand"}</>,<>{"Those observations should be removed"}</>],[<>{"Focal improves recall, worsens log loss"}</>,<>{"Threshold and calibration on validation"}</>,<>{"Focal is uniformly better or worse"}</>],[<>{"Triplet loss stays at the margin"}</>,<>{"Embedding spread, gradients, positive/negative labels"}</>,<>{"A larger margin alone will fix collapse"}</>],[<>{"Candidate loss becomes tiny, retrieval fails"}</>,<>{"Split leakage, shortcuts, candidate identities, gallery protocol"}</>,<>{"Duplicates necessarily explain tiny loss"}</>],[<>{"Combined losses change with batch size"}</>,<>{"Reduction denominators and term gradients"}</>,<>{"Equal displayed loss values give equal influence"}</>]]} />

<H2>{"Build the objectives, then control the library"}</H2>

<Prose>{"The earlier formulas tell you what a loss means. Now implement the computation that connects it to a parameter update. The "}<a href={"/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/loss-mechanisms.py"}>{"complete NumPy program"}</a>{" supplies MSE, MAE, Huber, quantile, stable multiclass CE, binary CE/focal, squared-distance triplet and paired InfoNCE, with "}<strong>{"explicit input derivatives"}</strong>{". NumPy supplies array arithmetic; it does not compute these losses or derivatives for us. PyTorch appears only in the comparison code. The pair-contrastive objective and semi-hard miner already have readable tensor-primitive implementations in "}<a href={"/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/loss-experiments.py"}>{"the earlier experiment"}</a>{"; reuse those rather than create a second owner."}</Prose>

<Prose>{"Read the core of multiclass CE first. A row is one example and a column is one class. Subtract each row's maximum, exponentiate, then divide by that row's sum. These are the softmax probabilities. The negative log-probability at the correct class is the per-example loss. Its logit gradient is the probability vector with one subtracted at the correct class. Averaging the loss means dividing "}<strong>{"every"}</strong>{" gradient by the batch size too."}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\ndef cross_entropy(logits, targets):\n    shifted = logits - logits.max(axis=1, keepdims=True)\n    exponential = np.exp(shifted)\n    partition = exponential.sum(axis=1, keepdims=True)\n    log_probability = shifted - np.log(partition)\n    gradient = exponential / partition\n    rows = np.arange(len(logits))\n    loss = -log_probability[rows, targets].mean()\n    gradient[rows, targets] -= 1\n    return loss, gradient / len(logits)\n\nlogits = np.array([[1., 2., -.5], [-.2, 1., .6]])\nlabels = np.array([0, 2])\nloss, logit_gradient = cross_entropy(logits, labels)\nprint(round(float(loss), 6))  # 1.225170\nprint(np.round(logit_gradient.sum(axis=1), 12))  # [0. 0.]"}</CodeBlock>

<Prose>{"Each gradient row sums to zero because shifting all logits equally changes no probability. This is a useful invariant for finding a wrong class axis or missing normalization. The downloaded version additionally checks the input shape and class-index contract. It supports finite logits whose differences and resulting loss fit the dtype; it does not promise meaningful arithmetic on infinities or values beyond floating-point range."}</Prose>

<Prose>{"For the affine classifier "}<InlineMath>{"Z=XW+b"}</InlineMath>{", the new loss supplies "}<InlineMath>{"G=\\partial L/\\partial Z"}</InlineMath>{". The existing chain rule then gives "}<InlineMath>{"\\partial L/\\partial W=X^\\top G"}</InlineMath>{" and "}<InlineMath>{"\\partial L/\\partial b=\\sum_i G_i"}</InlineMath>{". One SGD step subtracts the learning rate times each gradient. The supplied program computes that update manually, copies the "}<strong>{"same"}</strong>{" parameters into "}<code>{"nn.Linear"}</code>{", runs "}<code>{"F.cross_entropy"}</code>{" and "}<code>{"torch.optim.SGD"}</code>{", and compares the resulting parameters. Our "}<InlineMath>{"W"}</InlineMath>{" is input-by-class; "}<code>{"nn.Linear.weight"}</code>{" is class-by-input, so the copy uses a transpose. This is a controlled implementation comparison, not a comparison between independently initialized training runs. The two-row loss fixture above is separate from the three-row update fixture in the full program."}</Prose>

<NeuralTable caption={"Build the objectives, then control the library"} headers={[<>{"What you implemented"}</>,<>{"Normal library route"}</>,<>{"Setting you must preserve"}</>]} rows={[[<>{"Squared, absolute and Huber residual penalties"}</>,<>{""}<code>{"F.mse_loss"}</code>{", "}<code>{"F.l1_loss"}</code>{", "}<code>{"F.huber_loss"}</code>{""}</>,<>{"Mean versus sum; Huber delta and its half-factor"}</>],[<>{"Stable multiclass log probabilities"}</>,<>{""}<code>{"F.cross_entropy"}</code>{""}</>,<>{"Raw logits, integer labels, class axis; weights/smoothing deliberately absent in this comparison"}</>],[<>{"Stable binary CE; confidence-dependent focal weighting"}</>,<>{""}<code>{"F.binary_cross_entropy_with_logits"}</code>{"; compose the focal term with tensor primitives"}</>,<>{"Target convention, gamma, alpha/positive weighting and differentiating the modulation"}</>],[<>{"Squared-distance triplet hinge"}</>,<>{""}<code>{"TripletMarginWithDistanceLoss"}</code>{" with a squared-distance function"}</>,<>{"Distance, margin, swap and reduction; default unsquared triplet is a different objective"}</>],[<>{"Paired cosine candidate classification"}</>,<>{"Normalize features, matrix multiply, then "}<code>{"F.cross_entropy"}</code>{""}</>,<>{"Temperature, positive index, candidate set and one-way versus symmetric loss"}</>]]} />

<Prose>{"For focal loss the stable implementation keeps both correct and incorrect log-probabilities in log space. That prevents subtracting a rounded probability from one and losing the small tail. For InfoNCE, gradients pass through cosine normalization as well as through CE: treating already-normalized vectors as the original inputs drops a real dependency. This comparison excludes zero vectors, requires representable nonzero norms and intermediates, and sets "}<code>{"F.normalize(..., eps=0)"}</code>{" to match the scratch rule. The API's usual small-norm floor is a different function with a different derivative in that region. At the quantile loss's zero-residual corner, the scratch code chooses subgradient zero; "}<code>{"torch.maximum"}</code>{" can choose another valid subgradient. The quantile gradient comparison deliberately uses nonzero residuals; equal values at a corner do not guarantee identical optimization steps. Reuse "}<a href={"/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals"}>{"Backpropagation's chain-rule engine"}</a>{" to understand composition; this lesson owns the objective, not another autodiff system."}</Prose>

<Prose>{"Save "}<code>{"loss-mechanisms.py"}</code>{", use the environment above, and run "}<code>{"python loss-mechanisms.py"}</code>{". It compares forward values and explicit derivatives, including logits of ±1000, then checks the matched classifier update. The update fixture's mean loss changes from about "}<strong>{"1.442721 to 1.313306"}</strong>{" at learning rate .1. This is one verified step, not a claim that every positive step size decreases every loss. "}<a href={"/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/loss-mechanisms-output.json"}>{"Recorded comparison output"}</a>{" provides the exact values and error magnitudes."}</Prose>

<NeuralProgram topic="loss-mechanisms" title="Read the scratch objectives and matched library checks" />

<Prose>{"The dense CE calculation takes "}<InlineMath>{"O(NC)"}</InlineMath>{" time and storage including its returned gradient; it avoids a separate one-hot target matrix. Paired InfoNCE uses matrix multiplication and an "}<InlineMath>{"N\\times N"}</InlineMath>{" score matrix: time "}<InlineMath>{"O(N^2D)"}</InlineMath>{", score storage "}<InlineMath>{"O(N^2)"}</InlineMath>{". That is appropriate for this exact full-candidate objective at modest batch sizes, not a memory-optimal solution for unlimited batches. Chunked log-sum-exp and recomputed backward blocks can reduce peak score storage while retaining the objective; sampling fewer negatives changes it. Maintained fused loss kernels may use less temporary storage than this inspectable NumPy version. These are algorithmic costs, not measured speed claims."}</Prose>

<Prose>{""}<strong>{"Extension — implement label smoothing without a one-hot matrix."}</strong>{" Starting from the scratch CE, replace the target by "}<InlineMath>{"(1-\\epsilon)"}</InlineMath>{" at the correct class plus "}<InlineMath>{"\\epsilon/C"}</InlineMath>{" everywhere. Preserve the stable log probabilities and compare against "}<code>{"F.cross_entropy(..., label_smoothing=epsilon)"}</code>{" at epsilon 0 and .2. Check the scalar loss, every gradient and the zero row-sum invariant on a different three-class batch. This gives you control over a real training choice instead of merely changing an import."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The uniform part of the loss is the negative mean log-probability over classes, while the correct-class part retains weight "}<InlineMath>{"1-\\epsilon"}</InlineMath>{"."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{""}<strong>{"Solution:"}</strong>{" compute "}<code>{"-(1-epsilon)*log_probability[rows, targets].mean() - epsilon*log_probability.mean()"}</code>{". For the unreduced logit gradient, start with probabilities, subtract "}<code>{"epsilon / C"}</code>{" everywhere and subtract "}<code>{"1-epsilon"}</code>{" at the correct class, then divide by "}<code>{"N"}</code>{". At epsilon zero this is the original code. The mean over classes belongs only to the uniform loss term; the outer mean remains over examples. This extension matches the normalized unweighted targets specified in "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"}>{"PyTorch's CE contract"}</a>{", not a weighted or ignored-label variant."}</Prose>

</details>

<H2>{"Practice: change the problem, then explain the consequence"}</H2>

<section className="neural-practice"><Prose>{""}<strong>{"Different measurement units."}</strong>{" Delivery errors change from minutes to seconds. How do squared loss, absolute loss, and a Huber threshold change if the intended behavior should remain the same?"}</Prose><details><summary>Hint</summary><Prose>{"write "}<InlineMath>{"r'=60r"}</InlineMath>{"."}</Prose></details><details><summary>Worked solution</summary><Prose>{"squared loss multiplies by 3600, absolute by 60. Scale Huber's threshold by 60; its numerical loss then scales by 3600. An optimizer or a combined objective may need corresponding scale adjustments."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Different cost balance."}</strong>{" The actual positive probability is .2 and the positive weight is four. Derive the ideal weighted-BCE output."}</Prose><details><summary>Hint</summary><Prose>{"use expected loss before differentiating."}</Prose></details><details><summary>Worked solution</summary><Prose>{""}<InlineMath>{"p^*=.8/(.8+.8)=.5"}</InlineMath>{". This does not mean the original event has probability .5. A .5 threshold in this ideal weighted space corresponds to an original probability threshold .2."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Different margin."}</strong>{" With "}<InlineMath>{"a=(0,0),p=(1,0),n=(1.2,0)"}</InlineMath>{", change the squared margin from one to .3. Predict the active status and loss."}</Prose><details><summary>Worked solution</summary><Prose>{""}<InlineMath>{"1-1.44+.3=-.14"}</InlineMath>{", so loss and its local gradient are zero. The geometry did not change; the required separation changed."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Different candidates."}</strong>{" Three candidates have equal scores; add a fourth identical candidate. What happens to loss, and can lowering temperature reverse that?"}</Prose><details><summary>Worked solution</summary><Prose>{"it rises from "}<InlineMath>{"\\log3"}</InlineMath>{" to "}<InlineMath>{"\\log4"}</InlineMath>{"; temperature cannot distinguish equal scores. Now move only the designated positive score upward and explain why lowering temperature can help."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Repair the experiment."}</strong>{" A colleague selects the threshold with the lowest error count on the final test, then reports that count as untouched performance. Identify the information consumed and propose a valid continuation."}</Prose><details><summary>Worked solution</summary><Prose>{"test labels selected a modeling decision. Treat that test as development information, freeze the revised protocol using development data, and acquire or reserve another untouched evaluation set if an unbiased final assessment is required."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Run a changed objective."}</strong>{" Add "}<code>{"focal_gamma_0"}</code>{" to the program with no class weighting, resetting seeds exactly as before. Predict its relationship to BCE, then inspect losses, parameters, and validation probabilities with floating-point tolerances."}</Prose><details><summary>Worked solution</summary><Prose>{"the formulas are identical; differences should be limited to implementation/numerical effects. Adding the balanced "}<InlineMath>{"\\alpha=.25"}</InlineMath>{" convention would change the objective and invalidate this null comparison."}</Prose></details></section>

<section className="neural-practice"><Prose>{""}<strong>{"Diagnose a gradient."}</strong>{" Two identical embeddings designated negative have positive pair loss but no useful update under the library's zero-distance convention. Explain why displaying only the loss misses the problem."}</Prose><details><summary>Worked solution</summary><Prose>{"the loss value says the constraint is violated; the norm's derivative direction at coincidence is not defined. Inspect representation initialization, symmetry, nonzero variations, and the actual gradient instead of treating positive loss as proof of movement."}</Prose></details></section>

<H2>{"Another way to learn, and the next connection"}</H2>

<Prose>{"Start with "}<a href={"https://cs231n.github.io/linear-classify/"}>{"Stanford CS231n's linear-classification notes"}</a>{" if a second worked softmax explanation helps; its score→loss→probability diagrams complement the local regression view. "}<a href={"https://www.youtube.com/watch?v=h7iBpEHGVNc"}>{"Stanford Lecture 3: Loss Functions and Optimization"}</a>{" provides a spoken explanation of classification objectives and how optimization uses them. Watch it after the probability example; it does not replace the later focal and metric-learning sections. The official channel description was checked; the video itself was not watched for this packet."}</Prose>

<Prose>{"For paper reading, use focal §3 after the contribution lab, FaceNet §3 after the mining exercise, and SimCLR Algorithm 1 after drawing the candidate mask. Read CPC's information-theory branch only after candidate CE is comfortable. The articles are alternate routes and sources; the local explanation and program stand on their own."}</Prose>

<Prose>{"The next topic is "}<strong>{"Batch, Layer, Group, and RMS Normalization"}</strong>{". We now know how scores are judged. Next we examine the intermediate numbers entering those scores: which collections of activations are rescaled, how that changes dependence between examples, and why training and inference sometimes use different statistics."}</Prose>
  <p><a href="/learn/path/full-curriculum/batch-layer-group-rms-normalization?module=deep-learning-fundamentals">Continue to Batch, Layer, Group and RMS Normalization</a></p>
  </div>
};
