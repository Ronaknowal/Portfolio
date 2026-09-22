// Full prepared revision-3 manuscript, rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { DropoutUpdateLab, DropoutExpectationLab, DropoutGeometryLab, DropoutBranchLab, DropoutDepthLab, DropoutModeLab, DropoutMeasuredLab, DropoutMonteCarloLab, DropoutProgram } from '../../components/lesson-labs/DropoutLabs.jsx';

export default {
  title: 'Dropout, DropPath & Stochastic Depth',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson dropout-lesson">
    <LessonIntro prerequisites="Activations, multiplication, means and a loss gradient. The previous residual-connections lesson explains the direct and correction paths; mask probabilities and tensor axes are introduced here." sections={[["learn-with-some-information-temporarily-missing","Learn with some information temporarily missing"],["1-a-mask-changes-values-then-changes-an-update","1. A mask changes values, then changes an update"],["2-why-divide-by-the-keep-probability","2. Why divide by the keep probability?"],["3-what-gets-hidden-geometry-matters","3. What gets hidden? Geometry matters"],["4-drop-the-correction-while-keeping-the-direct-path","4. Drop the correction while keeping the direct path"],["5-modes-state-and-a-normalization-trap","5. Modes, state and a normalization trap"],["use-the-mask-contract-in-a-library-without-changing-its-meaning","Use the mask contract in a library without changing its meaning"],["6-a-complete-experiment-does-masking-help-these-digits","6. A complete experiment: does masking help these digits?"],["7-optional-several-predictions-from-one-dropout-model","7. Optional: several predictions from one dropout model"],["8-optional-choose-a-noise-pattern-for-a-reason","8. Optional: choose a noise pattern for a reason"],["9-practice-with-changed-inputs","9. Practice with changed inputs"],["10-continue-and-read-another-explanation","10. Continue and read another explanation"]]}>Follow a mask through values, gradients, network geometry and real training evidence.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Edit features/weights, probability, survivor scale, mask grouping, branch position, per-block rates and train/eval mode; inspect retained Monte Carlo prefixes. Update weighted outcome means/variances, gradient routes, call counts, state buffers and saved prediction distributions immediately. Keep the sampled mask fixed while comparing a parameter, with resampling a separate action. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose masking scope and evaluation behavior from their actual effects; distinguish expected active depth from work that was really skipped."}</Prose>

<H2>{"Learn with some information temporarily missing"}</H2>

<Prose>{"Imagine recognizing a handwritten 8. A model might use its upper loop, lower loop, central narrowing and stroke locations. If training makes one combination indispensable, the model may struggle when a new handwriting style changes part of that combination. One possible training intervention is to randomly hide some intermediate values and still ask for the correct digit."}</Prose>

<Prose>{""}<strong>{"Dropout"}</strong>{" does this temporary hiding. The values return on another pass; parameters are not permanently deleted. "}<strong>{"DropPath"}</strong>{", commonly used for a form of "}<strong>{"stochastic depth"}</strong>{", hides an entire learned correction in a residual block. Both modify the training problem. Their usefulness must be checked on examples excluded from fitting."}</Prose>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals"}>{"previous lesson on residual connections"}</a>{" showed that a direct path can preserve a representation while another path changes it. Here we ask what happens when the correction is sometimes absent."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow the two-value example, mask geometry, residual branch calculation, train/evaluation mode distinction and real digit comparison; then try practice 1–5. Monte Carlo uncertainty and specialized noise families are optional deeper branches. You need multiplication, averages, a loss and its gradient; these are refreshed where used."}</Prose>

<H2>{"1. A mask changes values, then changes an update"}</H2>

<Prose>{"Suppose a hidden representation is "}<InlineMath>{"h=[1,2]"}</InlineMath>{". A scalar output uses weights "}<InlineMath>{"w=[1,-0.5]"}</InlineMath>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"\\hat y=w^\\top h=1(1)-0.5(2)=0."}</MathBlock></div>

<Prose>{"Let the target be 1. We use half-squared error "}<InlineMath>{"L=\\tfrac12(\\hat y-1)^2"}</InlineMath>{", so the unmasked loss is 0.5."}</Prose>

<Prose>{"Set the "}<strong>{"drop probability"}</strong>{" to "}<InlineMath>{"p=0.5"}</InlineMath>{". The keep probability is "}<InlineMath>{"q=1-p=0.5"}</InlineMath>{". Independently for each value, sample a bit: 1 means keep, 0 means hide. Such a bit is a "}<strong>{"Bernoulli random variable"}</strong>{". Suppose the sampled mask is "}<InlineMath>{"m=[1,0]"}</InlineMath>{"."}</Prose>

<Prose>{"Modern inverted dropout multiplies by the mask and divides surviving values by the keep probability:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\widetilde h=\\frac{m\\odot h}{q}=[2,0],\\qquad\n\\hat y=w^\\top\\widetilde h=2."}</MathBlock></div>

<Prose>{"The output moved from 0 to 2; it did not become the target. The sampled loss is again 0.5, now with error in the opposite direction."}</Prose>

<Prose>{"Trace the gradient through these actual values:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial w}\n=(\\hat y-1)\\widetilde h=[2,0],\n\\qquad\n\\frac{\\partial L}{\\partial h}\n=(\\hat y-1)w\\odot m/q=[2,0]."}</MathBlock></div>

<Prose>{"A gradient is the local sensitivity of loss to a small change. A gradient-descent step of size 0.1 gives "}<InlineMath>{"w_{\\mathrm{new}}=[0.8,-0.5]"}</InlineMath>{". With this same mask, the new output is 1.6 and loss is 0.18. The second weight receives no contribution from this example through the dropped coordinate."}</Prose>

<Prose>{"That is not a promise that its optimizer value never changes: other examples, other paths, momentum or weight decay can still contribute. A fresh mask belongs to the next forward pass. Backpropagation must use the mask from the forward computation it differentiates."}</Prose>

<DropoutUpdateLab />

<Prose>{""}<strong>{"Try a different mask:"}</strong>{" keep the original weights and change the mask to "}<InlineMath>{"[0,1]"}</InlineMath>{". Follow the changed gradient route: the masked representation becomes "}<InlineMath>{"[0,4]"}</InlineMath>{", output −2, error −3 and weight gradient "}<InlineMath>{"[0,-12]"}</InlineMath>{". The two masks train different dependencies of the same model."}</Prose>

<H2>{"2. Why divide by the keep probability?"}</H2>

<Prose>{"For a fixed value "}<InlineMath>{"h_i"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[\\widetilde h_i\\mid h_i]\n=q(h_i/q)+p(0)=h_i."}</MathBlock></div>

<Prose>{"The expectation is an average over repeated masks, not a statement about every pass. Here are "}<strong>{"all four outcomes"}</strong>{" for "}<InlineMath>{"h=[1,2]"}</InlineMath>{", "}<InlineMath>{"p=0.5"}</InlineMath>{":"}</Prose>

<NeuralTable caption={"2. Why divide by the keep probability?"} headers={[<>{"Mask"}</>,<>{"Probability"}</>,<>{"Masked representation"}</>]} rows={[[<>{"[0,0]"}</>,<>{"0.25"}</>,<>{"[0,0]"}</>],[<>{"[0,1]"}</>,<>{"0.25"}</>,<>{"[0,4]"}</>],[<>{"[1,0]"}</>,<>{"0.25"}</>,<>{"[2,0]"}</>],[<>{"[1,1]"}</>,<>{"0.25"}</>,<>{"[2,4]"}</>]]} />

<Prose>{"Their weighted mean is "}<InlineMath>{"[1,2]"}</InlineMath>{". Their coordinate variances are "}<InlineMath>{"[1,4]"}</InlineMath>{". Generally,"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{Var}(\\widetilde h_i\\mid h_i)=\\frac{p}{1-p}h_i^2."}</MathBlock></div>

<Prose>{"For "}<InlineMath>{"p=0.25"}</InlineMath>{", the four probabilities are "}<InlineMath>{"0.0625,0.1875,0.1875,0.5625"}</InlineMath>{", in the same row order. They are not uniform. The mean stays "}<InlineMath>{"[1,2]"}</InlineMath>{", while variances become "}<InlineMath>{"[1/3,4/3]"}</InlineMath>{". Increasing "}<InlineMath>{"p"}</InlineMath>{" changes both how often information disappears and the amplitude of surviving values."}</Prose>

<Prose>{"At ordinary evaluation, inverted dropout returns "}<InlineMath>{"h"}</InlineMath>{" directly. It applies neither a random mask nor an extra keep-probability multiplier. Historical implementations instead left training survivors unscaled and multiplied by "}<InlineMath>{"q"}</InlineMath>{" at evaluation. Both conventions need internally consistent initialization/training scale; mixing their evaluation rules is an error. See the explicit current "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html"}>{"PyTorch Dropout contract"}</a>{"."}</Prose>

<Prose>{"At "}<InlineMath>{"p=0"}</InlineMath>{", training also becomes identity. At "}<InlineMath>{"p=1"}</InlineMath>{", division by zero is invalid; the supplied implementation explicitly returns zeros during training and identity during evaluation. The expectation-preservation formula applies to "}<InlineMath>{"p<1"}</InlineMath>{"."}</Prose>

<DropoutExpectationLab />

<H3>{"Preserved means do not imply an unchanged network"}</H3>

<Prose>{"Use signed contributions "}<InlineMath>{"[1,-1]"}</InlineMath>{", independent masks and "}<InlineMath>{"p=0.5"}</InlineMath>{", then apply ReLU to their sum. Unmasked, the result is "}<InlineMath>{"\\max(0,1-1)=0"}</InlineMath>{". Across the four equally likely masks, the results are "}<InlineMath>{"0,0,2,0"}</InlineMath>{", whose mean is 0.5:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[\\operatorname{ReLU}(Z)]\\ne\n\\operatorname{ReLU}(\\mathbb E[Z])."}</MathBlock></div>

<Prose>{"Even when a linear output preserves its mean, its expected loss can change. “An ensemble of thinned networks” is a useful interpretation of shared parameters under different masks, not a claim of independently trained models or exact arithmetic averaging by one deterministic nonlinear pass."}</Prose>

<Prose>{"The training objective is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\min_\\theta\\frac1N\\sum_{i=1}^N\n\\mathbb E_m[\\ell(f_\\theta(x_i;m),y_i)]."}</MathBlock></div>

<Prose>{"Each sampled update estimates this noisy objective. The goal is to learn useful predictions under that perturbation, not to make each hidden unit a complete classifier. Overfitting means fitting sample-specific patterns that generalize poorly; it does not require a gap that widens forever, and parameter count alone does not diagnose it."}</Prose>

<H2>{"3. What gets hidden? Geometry matters"}</H2>

<Prose>{"A tensor is an array with named axes. For an image representation "}<InlineMath>{"[B,C,H,W]"}</InlineMath>{", "}<InlineMath>{"B"}</InlineMath>{" indexes examples, "}<InlineMath>{"C"}</InlineMath>{" feature channels, and "}<InlineMath>{"H,W"}</InlineMath>{" spatial positions. A channel might respond to a learned pattern over the image. Convolution will explain how those maps are built in the next lesson."}</Prose>

<NeuralTable caption={"3. What gets hidden? Geometry matters"} headers={[<>{"Operation"}</>,<>{"Independent mask shape"}</>,<>{"What disappears together"}</>]} rows={[[<>{"Element dropout"}</>,<>{"[B,C,H,W]"}</>,<>{"One activation value"}</>],[<>{"Channel dropout"}</>,<>{"[B,C,1,1]"}</>,<>{"A whole feature map for one example"}</>],[<>{"Per-example branch dropout"}</>,<>{"[B,1,1,1]"}</>,<>{"The whole correction for one example"}</>],[<>{"Batchwise branch dropout"}</>,<>{"[1,1,1,1]"}</>,<>{"The correction for every example in that batch"}</>]]} />

<Prose>{"Dimensions of size 1 are "}<strong>{"broadcast"}</strong>{": the same bit is repeated along that axis. This small shape choice defines the intervention. On a sequence shaped "}<InlineMath>{"[B,T,D]"}</InlineMath>{", a branch mask "}<InlineMath>{"[B,1,1]"}</InlineMath>{" is shared across tokens and features for an example. An element mask "}<InlineMath>{"[B,T,D]"}</InlineMath>{" makes separate decisions. Neither shape can be inferred from the word “dropout” alone."}</Prose>

<Prose>{"Take two examples with two "}<InlineMath>{"2\\times2"}</InlineMath>{" channels each, filled with values 1–16 in order. Under channel mask "}<InlineMath>{"[1,0]"}</InlineMath>{" for example 1 and "}<InlineMath>{"[0,1]"}</InlineMath>{" for example 2, with "}<InlineMath>{"p=0.5"}</InlineMath>{", the surviving maps contain "}<InlineMath>{"[[2,4],[6,8]]"}</InlineMath>{" and "}<InlineMath>{"[[26,28],[30,32]]"}</InlineMath>{". The other two maps are entirely zero. Under a branch mask "}<InlineMath>{"[1,0]"}</InlineMath>{", both maps of example 1 survive and both of example 2 disappear."}</Prose>

<Prose>{""}<strong>{"Build the intervention:"}</strong>{" make an entire second channel disappear for example 1 while preserving its first channel and all channels of example 2. Choose the mask axes and enter its bits. Then check that no spatial position inside a channel contradicts another."}</Prose>

<Prose>{"The difference is more than appearance. For fixed values "}<InlineMath>{"[1,2]"}</InlineMath>{" and "}<InlineMath>{"p=0.5"}</InlineMath>{", independent masks give covariance 0; one shared mask gives covariance 2. Shared masking makes values move together. Spatial neighbors can carry redundant evidence, so removing isolated values may leave that evidence nearby. Channel or contiguous-region masking can challenge a different dependency. This motivates comparison, not a rule that element dropout after convolution is always useless. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout2d.html"}>{"PyTorch Dropout2d"}</a>{" explicitly defines the channel operation; use a four-dimensional batched input here because its three-dimensional interpretation has a version-specific warning."}</Prose>

<DropoutGeometryLab />

<H2>{"4. Drop the correction while keeping the direct path"}</H2>

<Prose>{"A residual block computes "}<InlineMath>{"y=x+F(x)"}</InlineMath>{". With inverted branch dropout,"}</Prose>

<div className="neural-equation"><MathBlock>{"y=x+\\frac{m}{q}F(x)."}</MathBlock></div>

<Prose>{"Let "}<InlineMath>{"x=[2,-1]"}</InlineMath>{", "}<InlineMath>{"F(x)=[0.5,1]"}</InlineMath>{", "}<InlineMath>{"q=0.5"}</InlineMath>{"."}</Prose>

<NeuralTable caption={"4. Drop the correction while keeping the direct path"} headers={[<>{"Branch bit"}</>,<>{"Block output"}</>]} rows={[[<>{"0"}</>,<>{"[2,−1]"}</>],[<>{"1"}</>,<>{"[3,1]"}</>],[<>{"Average"}</>,<>{"[2.5,0]"}</>]]} />

<Prose>{"The average equals the unmasked block output for this fixed input. If instead you mask the "}<strong>{"whole sum"}</strong>{", a dropped pass gives "}<InlineMath>{"[0,0]"}</InlineMath>{", removing the direct path too. It is a different architecture."}</Prose>

<Prose>{"For a sampled branch bit, the local derivative is "}<InlineMath>{"I+(m/q)J_F"}</InlineMath>{", where "}<InlineMath>{"J_F"}</InlineMath>{" describes how the correction changes with input. A dropped correction leaves "}<InlineMath>{"I"}</InlineMath>{". A surviving correction can still cancel, shrink or amplify the total derivative, as the preceding residual lesson demonstrated. The preserved path is useful, not an unconditional gradient guarantee."}</Prose>

<DropoutBranchLab />

<Prose>{"Terminology varies. Modern libraries commonly call per-example residual-branch masking "}<strong>{"DropPath"}</strong>{", and also call it stochastic depth. Torchvision's "}<a href={"https://docs.pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html"}>{"stochastic-depth implementation"}</a>{" supports both row and batch modes. The "}<a href={"https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/layers/drop.py"}>{"timm implementation"}</a>{" uses one bit per example and has a keep-scaling option. Read the mask and scaling contract instead of assuming two names imply two incompatible algorithms."}</Prose>

<H3>{"Expected active branches are not measured runtime"}</H3>

<Prose>{"For "}<InlineMath>{"L"}</InlineMath>{" residual blocks with drop probabilities "}<InlineMath>{"p_l"}</InlineMath>{", the expected active count is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[A]=\\sum_{l=1}^L(1-p_l)."}</MathBlock></div>

<Prose>{"The original stochastic-depth schedule corresponds, in our drop-probability notation, to "}<InlineMath>{"p_l=p_{\\max}l/L"}</InlineMath>{", for "}<InlineMath>{"l=1,\\ldots,L"}</InlineMath>{". It gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[A]=L-p_{\\max}(L+1)/2."}</MathBlock></div>

<Prose>{"Four blocks with "}<InlineMath>{"p_{\\max}=0.5"}</InlineMath>{" have rates "}<InlineMath>{"[0.125,0.25,0.375,0.5]"}</InlineMath>{", giving 2.75 expected active blocks. A zero-first schedule "}<InlineMath>{"[0,1/6,1/3,1/2]"}</InlineMath>{" gives 3. These are different conventions, both explicit. For a one-block zero-first schedule, our code uses "}<InlineMath>{"[0]"}</InlineMath>{" rather than dividing by "}<InlineMath>{"L-1=0"}</InlineMath>{"."}</Prose>

<Prose>{"Also count "}<strong>{"blocks"}</strong>{", not every layer within them. Huang et al.'s 110-layer example has 54 residual blocks. Their original convention used unscaled surviving branches in training and survival-scaled branches in evaluation. Their speed results involved actually bypassing computation. "}<a href={"https://arxiv.org/pdf/1603.09382"}>{"Deep Networks with Stochastic Depth, §3"}</a>{"."}</Prose>

<Prose>{"In the expression "}<code>{"mask_values(F(x), ...)"}</code>{", Python has already evaluated "}<code>{"F(x)"}</code>{". Multiplication by zero cannot undo that work. Batchwise conditional execution can avoid a branch if the decision comes first; per-example skipping may need gathering, scattering and different batch-statistic treatment. Unequal block costs, random generation, memory traffic and hardware scheduling also matter. Expected active depth is a structural quantity, not a speedup benchmark."}</Prose>

<DropoutDepthLab />

<H2>{"5. Modes, state and a normalization trap"}</H2>

<Prose>{"In PyTorch, "}<code>{"model.train()"}</code>{" enables modules' training behavior; "}<code>{"model.eval()"}</code>{" selects evaluation behavior. "}<code>{"torch.no_grad()"}</code>{" controls recording gradients. It does "}<strong>{"not"}</strong>{" turn dropout off or stop BatchNorm running-statistic updates."}</Prose>

<Prose>{"Ordinary validation uses both evaluation behavior and no gradient recording. The complete experiment calls these explicitly before measuring either training or validation rows. Measuring training data with dropout enabled and validation data with it disabled mixes two forward procedures and can create a misleading “generalization gap.”"}</Prose>

<Prose>{"BatchNorm stores running means and variances for evaluation. Suppose an activation "}<InlineMath>{"X"}</InlineMath>{" is equally likely to be 1 or 3. Its mean is 2 and variance is 1. With independent inverted dropout at "}<InlineMath>{"q=0.5"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathbb E[\\widetilde X^2]=\\mathbb E[X^2]/q=5/0.5=10,\n\\quad \\operatorname{Var}(\\widetilde X)=10-2^2=6."}</MathBlock></div>

<Prose>{"The mean was preserved; the variance was not. A BatchNorm downstream can learn statistics of this noisier distribution, then see the clean distribution at evaluation. That is the variance-shift mechanism studied by "}<a href={"https://arxiv.org/abs/1801.05134"}>{"Li et al."}</a>{"."}</Prose>

<Prose>{"The exact fixture sends "}<InlineMath>{"[0,2,0,6]"}</InlineMath>{" through BatchNorm with momentum 1. Its training variance uses divisor 4, giving 6; the stored unbiased running variance uses divisor 3, giving 8. Clean evaluation inputs "}<InlineMath>{"[1,3]"}</InlineMath>{" are therefore mapped to approximately "}<InlineMath>{"[-0.353553,0.353553]"}</InlineMath>{". Do not confuse population variance 6 with the stored finite-batch estimate 8."}</Prose>

<Prose>{"Placing masking after a particular BatchNorm avoids directly masking that layer's input, but later normalization layers may still see altered distributions. LayerNorm and GroupNorm do not have the same running-statistic mismatch, yet they are not immune to masking. LayerNorm of "}<InlineMath>{"[1,3]"}</InlineMath>{" is approximately "}<InlineMath>{"[-1,1]"}</InlineMath>{"; after the mask produces "}<InlineMath>{"[2,0]"}</InlineMath>{", it is approximately "}<InlineMath>{"[1,-1]"}</InlineMath>{". The representation reversed."}</Prose>

<Prose>{"For MC dropout, put the model in evaluation mode first, then selectively enable its dropout modules. Keep BatchNorm in evaluation mode. Our state probe verifies that "}<code>{"no_grad()"}</code>{" in training still increments a BatchNorm counter, while selective dropout activation does not. For functional calls, pass "}<code>{"training=self.training"}</code>{" during ordinary operation; a hardcoded "}<code>{"True"}</code>{" deliberately ignores "}<code>{"eval()"}</code>{"."}</Prose>

<DropoutModeLab />

<H2>{"Use the mask contract in a library without changing its meaning"}</H2>

<Prose>{"Read "}<code>{"mask_values"}</code>{" in "}<a href={"/learn-assets/dropout-droppath-stochastic-depth/dropout-experiments.py"}>{"the complete program"}</a>{" before the model. It is the scratch implementation: choose the broadcast shape, draw Bernoulli bits once, multiply, divide by keep probability, and bypass sampling in evaluation. Its array work is O(number of activation values); the random mask contains only as many independent entries as its chosen shape. "}<code>{"fixtures"}</code>{" keeps masks fixed for forward/backward arithmetic, while "}<code>{"DigitModel"}</code>{" shows ordinary "}<code>{"nn.Dropout"}</code>{" training. A stochastic sample is not an implementation-equivalence test just because two final losses look close."}</Prose>

<DropoutProgram title="Read the scratch mask and its axis contract" start="def mask_values" end="def fixtures" />

<Prose>{"The usual interfaces for the four scopes are:"}</Prose>

<CodeBlock language={"python"}>{"import torch\nfrom torch import nn\nfrom torchvision.ops import stochastic_depth\n\ntorch.manual_seed(9)\nfeatures = torch.arange(1., 17.).reshape(2, 2, 2, 2)\nelement = nn.Dropout(p=0.25)\nchannel = nn.Dropout2d(p=0.25)\nprint(element(features).shape, channel(features).shape)\nfor mode in (\"row\", \"batch\"):\n    branch = stochastic_depth(features, p=0.25, mode=mode, training=True)\n    print(mode, branch)\n    torch.testing.assert_close(\n        stochastic_depth(features, p=0.25, mode=mode, training=False), features)\nelement.eval()\nchannel.eval()\ntorch.testing.assert_close(element(features), features)\ntorch.testing.assert_close(channel(features), features)"}</CodeBlock>

<Prose>{"This standalone code needs compatible PyTorch/Torchvision versions. It requests the same mask geometry as the scratch implementation, but does not claim the random masks are identical. In "}<code>{"stochastic_depth"}</code>{", row means one bit per batch member, even when each member contains many tokens or pixels; batch means one bit for the entire supplied tensor. Both mask the supplied "}<strong>{"correction"}</strong>{", so the caller still adds the untouched residual input. The "}<a href={"https://docs.pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html"}>{"maintained implementation"}</a>{" makes that convention visible. Record the installed version when executing this newly prepared example."}</Prose>

<DropoutProgram file="dropout-library-checks.py" title="Read the matched library checks and locked-feature solution" />

<Prose>{""}<strong>{"Independent modification:"}</strong>{" add a "}<code>{"locked_features"}</code>{" case for a sequence "}<code>{"[B,T,D]"}</code>{", with independent bits shaped "}<code>{"[B,1,D]"}</code>{". Let the caller supply a fixed mask for a deterministic comparison. Return identity in eval, zeros at p1 in training, and "}<code>{"values * mask / (1-p)"}</code>{" otherwise. Then differentiate the sum of the output."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The forward bit belongs to a feature/example pair; every time step must use the same bit, including backward."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"For one example with time rows [1,2] and [3,4], p0.5 and mask [1,0], the result is [2,0] and [6,0]. The gradient of their total with respect to the input is [2,0] on both rows. A fresh backward mask or a "}<code>{"[B,T,D]"}</code>{" draw changes the contract. Test identity evaluation, p0 and p1 separately, and only then use random masks during training. A mask factory is a meaningful customization point; the loss, tensor gradients and optimizer can remain ordinary library operations."}</Prose>

</details>

<H2>{"6. A complete experiment: does masking help these digits?"}</H2>

<Prose>{"Download "}<a href={"/learn-assets/dropout-droppath-stochastic-depth/dropout-experiments.py"}>{"dropout-experiments.py"}</a>{", "}<a href={"/learn-assets/dropout-droppath-stochastic-depth/digits-400.csv"}>{"digits-400.csv"}</a>{" and the "}<a href={"/learn-assets/dropout-droppath-stochastic-depth/data-provenance.md"}>{"data provenance"}</a>{" into one directory. The program uses Python, PyTorch, NumPy and scikit-learn; run:"}</Prose>

<CodeBlock language={"sh"}>{"python -m pip install torch numpy scikit-learn\npython dropout-experiments.py"}</CodeBlock>

<Prose>{"The recorded run used Python 3.12.14, PyTorch 2.14.0 CPU, NumPy 2.3.5 and scikit-learn 1.9.1. The dataset contains 400 real "}<InlineMath>{"8\\times8"}</InlineMath>{" UCI digit images, not MNIST: 40 per class. A fixed stratified split uses 280 training and 120 validation examples, seed 22. Pixel values are divided by their known maximum 16. No fitted preprocessing uses validation data, and this small reused teaching split is not an official benchmark or final test."}</Prose>

<DropoutProgram />

<Prose>{"The program contains two controlled comparisons:"}</Prose>

<ul><li>{"An MLP: "}<InlineMath>{"64\\to64\\to64\\to10"}</InlineMath>{", tanh hidden activations, element dropout after each hidden activation, "}<InlineMath>{"p\\in\\{0,0.2,0.5,0.8\\}"}</InlineMath>{"."}</li><li>{"A residual MLP: a "}<InlineMath>{"64\\to64"}</InlineMath>{" tanh stem, four corrections "}<InlineMath>{"F_l(h)=0.5\\tanh(W_lh+b_l)"}</InlineMath>{", and a "}<InlineMath>{"64\\to10"}</InlineMath>{" head. Compare no branch masking with row or batch masking using zero-first schedules ending at 0.2 or 0.5."}</li></ul>

<Prose>{"Each family uses the same initial learned parameters for its masking variants at a given seed. The two families have different parameter counts, 8,970 and 21,450, so comparisons between them are not a matched architecture ablation. Every configuration uses Adam at 0.003 for 400 full-batch updates, with three initialization/mask seeds. No augmentation, weight decay, normalization, early stopping or hidden pretrained dependency is included."}</Prose>

<Prose>{"The mask producer is implemented explicitly for element, channel, row and batch shapes. The model's forward method makes placement visible. Training minimizes cross-entropy of raw logits; reported losses are deterministic evaluation-mode cross-entropy in natural-log units per example. Saved points at steps 0, 1, 25, 100, 200 and 400 are actual measurements, available in "}<a href={"/learn-assets/dropout-droppath-stochastic-depth/calculated-inputs.json"}>{"calculated-inputs.json"}</a>{"."}</Prose>

<Prose>{""}<strong>{"Executed final validation results, seed 1:"}</strong>{""}</Prose>

<NeuralTable caption={"6. A complete experiment: does masking help these digits?"} headers={[<>{"Family"}</>,<>{"Mask configuration"}</>,<>{"CE"}</>,<>{"Correct / 120"}</>]} rows={[[<>{"MLP"}</>,<>{"none"}</>,<>{"0.088034"}</>,<>{"118"}</>],[<>{"MLP"}</>,<>{"element 0.2"}</>,<>{"0.090138"}</>,<>{"118"}</>],[<>{"MLP"}</>,<>{"element 0.5"}</>,<>{"0.111671"}</>,<>{"116"}</>],[<>{"MLP"}</>,<>{"element 0.8"}</>,<>{"0.144736"}</>,<>{"116"}</>],[<>{"Residual"}</>,<>{"none"}</>,<>{"0.136714"}</>,<>{"117"}</>],[<>{"Residual"}</>,<>{"row, endpoint 0.2"}</>,<>{"0.150095"}</>,<>{"117"}</>],[<>{"Residual"}</>,<>{"batch, endpoint 0.2"}</>,<>{"0.146876"}</>,<>{"117"}</>],[<>{"Residual"}</>,<>{"row, endpoint 0.5"}</>,<>{"0.157595"}</>,<>{"117"}</>],[<>{"Residual"}</>,<>{"batch, endpoint 0.5"}</>,<>{"0.167144"}</>,<>{"116"}</>]]} />

<Prose>{"All but the element-0.8 configuration classify all 280 training images correctly; that configuration gets 279. Even high dropout did not force chance-level training accuracy here."}</Prose>

<Prose>{"Across seeds, the MLP's no-dropout validation correct count is 117–118, compared with 118 for all three element-0.2 runs. Seed 2's loss improves from 0.071144 to 0.067003 with 0.2, while seeds 1 and 3 slightly worsen. All recorded residual masking variants have worse final validation CE than their corresponding unmasked residual baseline. These observations support a narrow conclusion: masking is not clearly needed for this setup. They do not establish that a different dataset, architecture, schedule or training budget cannot benefit."}</Prose>

<Prose>{""}<strong>{"Investigate:"}</strong>{" compare the two saved runs with their validation-loss traces visible. Also compare correct counts and training loss. Explain why a smaller training–validation gap alone does not decide the winner. If you change a rate or budget in the program, keep the baseline, retain the new outputs and identify that as another validation experiment. A final generalization claim requires a separate evaluation plan."}</Prose>

<DropoutMeasuredLab />

<H2>{"7. Optional: several predictions from one dropout model"}</H2>

<Prose>{"Keep trained dropout active at inference and repeat a forward pass. This is "}<strong>{"Monte Carlo dropout"}</strong>{". For classification, each pass produces a probability vector "}<InlineMath>{"p^{(t)}"}</InlineMath>{"; average those vectors:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\bar p=\\frac1T\\sum_{t=1}^Tp^{(t)}."}</MathBlock></div>

<Prose>{"Average probabilities, not class IDs. Softmax of average logits is generally a different calculation. The supplied "}<code>{"mc_measure"}</code>{" function uses the seed-1 MLP trained with "}<InlineMath>{"p=0.5"}</InlineMath>{", selected in advance for demonstration, and 100 fresh masks. It sets only "}<code>{"nn.Dropout"}</code>{" modules to training mode and restores evaluation afterward."}</Prose>

<Prose>{"Deterministic evaluation has CE 0.111671, Brier score 0.040718 and 116/120 correct. The actual MC mean has CE 0.113693, Brier score 0.042566 and the same correct count. Brier here is the mean over examples of the "}<strong>{"sum across ten classes"}</strong>{" of squared probability errors. Repeated inference did not improve these scores."}</Prose>

<Prose>{"Probability spread can reveal sensitivity to learned-feature availability. To separate two kinds of ambiguity, define categorical entropy "}<InlineMath>{"H(p)=-\\sum_kp_k\\log p_k"}</InlineMath>{", in nats. Compare entropy of the mean with mean entropy:"}</Prose>

<div className="neural-equation"><MathBlock>{"D=H(\\bar p)-\\frac1T\\sum_tH(p^{(t)})."}</MathBlock></div>

<Prose>{"If two hypothetical passes give "}<InlineMath>{"[0.9,0.1]"}</InlineMath>{" and "}<InlineMath>{"[0.1,0.9]"}</InlineMath>{", the mean is "}<InlineMath>{"[0.5,0.5]"}</InlineMath>{", entropy 0.693147 and disagreement "}<InlineMath>{"D=0.368064"}</InlineMath>{". If both passes instead give "}<InlineMath>{"[0.5,0.5]"}</InlineMath>{", the mean is identical but "}<InlineMath>{"D=0"}</InlineMath>{". The first model's sampled predictions disagree; the second is ambiguous on every pass. This arithmetic is illustrative, separate from the measured digit outputs."}</Prose>

<Prose>{"In the actual run, validation specimen source ID 299 is a digit 1 but the mean predicts 6; predictive entropy is 1.216163 and disagreement 0.493401. Source ID 251 is correctly classified as 4, with entropy 0.061497 and disagreement 0.020239. Those two examples help interpret the quantities, but do not validate a universal error-detection threshold."}</Prose>

<DropoutMonteCarloLab />

<Prose>{""}<a href={"https://proceedings.mlr.press/v48/gal16.html"}>{"Gal and Ghahramani"}</a>{" give an approximate Bayesian interpretation under a specified variational family and prior/objective relationships. Arbitrary masks added to a model trained without them are not automatically posterior samples. Our experiment measures mask-induced prediction variability; it does not claim an exact Bayesian posterior, calibrated uncertainty or guaranteed detection of unfamiliar inputs."}</Prose>

<Prose>{"For regression, spread of sampled prediction means omits observation noise. In a model that explicitly assumes Gaussian observation variance "}<InlineMath>{"\\tau^{-1}"}</InlineMath>{", predictive variance includes that term plus variability of the means; "}<InlineMath>{"\\tau"}</InlineMath>{" is precision, and "}<InlineMath>{"\\tau^{-1}"}</InlineMath>{" is variance. Increasing "}<InlineMath>{"T"}</InlineMath>{" reduces Monte Carlo estimation noise, not model bias or all uncertainty."}</Prose>

<Prose>{"A useful application is selecting examples for labeling: disagreement can suggest where another label might help. Another is routing ambiguous inputs for human review. Both require validating the acquisition/deferral policy on the deployment setting. They are possible uses of these quantities, not safety or coverage certificates."}</Prose>

<H2>{"8. Optional: choose a noise pattern for a reason"}</H2>

<Prose>{"Several related methods answer different questions:"}</Prose>

<ul><li>{""}<strong>{"DropBlock"}</strong>{" hides contiguous regions within feature maps. A "}<InlineMath>{"3\\times3"}</InlineMath>{" blank region interrupts local redundant evidence differently from nine scattered zeros. Overlapping blocks and boundaries mean the seed probability for block centers is not simply the final fraction removed. Read the "}<a href={"https://arxiv.org/abs/1810.12890"}>{"original DropBlock paper"}</a>{" before implementing its sampling and normalization recipe."}</li><li>{""}<strong>{"DropConnect"}</strong>{" masks weights rather than activations. A missing activation removes its contribution to every recipient; missing individual weights can remove different connections to different recipients. "}<a href={"https://proceedings.mlr.press/v28/wan13.html"}>{"Wan et al."}</a>{" develop that distinction."}</li><li>{""}<strong>{"Zoneout"}</strong>{" carries selected previous recurrent-state values forward instead of replacing them with zero. If the old state is 0.7 and a proposed update is 0.2, a preserve decision returns 0.7. It is a memory-preserving intervention across time, not ordinary hidden dropout under another name. "}<a href={"https://arxiv.org/abs/1606.01305"}>{"Zoneout"}</a>{"."}</li><li>{""}<strong>{"Shake-Shake"}</strong>{" uses stochastic affine combinations of parallel branches; "}<strong>{"ShakeDrop"}</strong>{" develops a related residual regularizer with its own stabilization behavior. Their forward/backward recipes require separate study; arbitrary branch noise is not an interchangeable substitute. "}<a href={"https://arxiv.org/abs/1705.07485"}>{"Shake-Shake"}</a>{", "}<a href={"https://arxiv.org/abs/1802.02375"}>{"ShakeDrop"}</a>{"."}</li><li>{""}<strong>{"Gaussian/variational dropout"}</strong>{" extends multiplicative noise and can learn noise parameters. Kingma et al.'s local reparameterization and Molchanov et al.'s sparsification are distinct developments from ordinary fixed-rate MC dropout. Additional parameter cost depends on whether noise parameters are shared or per weight; fixed Bernoulli dropout does not double model parameters. "}<a href={"https://arxiv.org/abs/1506.02557"}>{"Local reparameterization"}</a>{", "}<a href={"https://arxiv.org/abs/1701.05369"}>{"variational sparsification"}</a>{"."}</li></ul>

<Prose>{"Attention probability dropout offers another instructive preview. A normalized row "}<InlineMath>{"[0.25,0.75]"}</InlineMath>{", mask "}<InlineMath>{"[1,0]"}</InlineMath>{" and "}<InlineMath>{"q=0.5"}</InlineMath>{" becomes "}<InlineMath>{"[0.5,0]"}</InlineMath>{", whose sum is 0.5. The operation preserves each weight's expectation, not the row sum on every pass. Renormalizing afterward defines a different operation. The attention lesson will explain the values being mixed; the masking calculation already shows why a sampled result need not be a convex average."}</Prose>

<Prose>{"A practical choice starts with the unmasked baseline, the dependency you want to perturb, and a valid validation procedure. Compare a small set of rates and placement choices. Revisit learning rate or training duration if the noisy objective is difficult to fit. Do not copy an architecture's default as a theorem about your data, or assume massive datasets make memorization impossible."}</Prose>

<H2>{"9. Practice with changed inputs"}</H2>

<H3>{"1. Repair the scaling"}</H3>

<Prose>{"A value 3 survives with probability 0.75. A program multiplies survivors by 0.75. What are its expected output and the correct survivor value?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Distinguish the chance of survival from the value conditional on survival."}</Prose>

</details>

<details>

<summary>Worked solution</summary>

<Prose>{"Its expectation is "}<InlineMath>{"0.75(3\\cdot0.75)=1.6875"}</InlineMath>{". Correct inverted scaling returns "}<InlineMath>{"3/0.75=4"}</InlineMath>{" when kept and 0 otherwise, giving expectation 3."}</Prose>

</details>

<H3>{"2. Follow a different update"}</H3>

<Prose>{"Let "}<InlineMath>{"h=[2,-1]"}</InlineMath>{", "}<InlineMath>{"w=[0.5,1]"}</InlineMath>{", target 0, "}<InlineMath>{"q=0.5"}</InlineMath>{", mask "}<InlineMath>{"[0,1]"}</InlineMath>{", half-squared loss. Find output, weight gradient and weights after an SGD step of 0.1."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Form the masked input before differentiating."}</Prose>

</details>

<details>

<summary>Worked solution</summary>

<Prose>{"Masked input "}<InlineMath>{"[0,-2]"}</InlineMath>{", output −2, loss 2, gradient "}<InlineMath>{"[0,4]"}</InlineMath>{", new weights "}<InlineMath>{"[0.5,0.6]"}</InlineMath>{". With the same mask the new output is −1.2 and loss 0.72."}</Prose>

</details>

<H3>{"3. Design the mask axes"}</H3>

<Prose>{"For "}<InlineMath>{"[B,T,D]=[3,5,4]"}</InlineMath>{", hide a feature consistently over all time positions for each example, but allow different examples to keep different features. What mask shape is appropriate?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"List which axis must share a decision, and which axes need independent decisions."}</Prose>

</details>

<details>

<summary>Worked solution</summary>

<Prose>{""}<InlineMath>{"[3,1,4]"}</InlineMath>{". A "}<InlineMath>{"[3,5,4]"}</InlineMath>{" mask varies over time; "}<InlineMath>{"[3,1,1]"}</InlineMath>{" hides the whole example's branch; "}<InlineMath>{"[1,1,4]"}</InlineMath>{" forces the same feature decisions across examples. Actual recurrent placement needs its own temporal-state reasoning."}</Prose>

</details>

<H3>{"4. Catch two validation bugs"}</H3>

<Prose>{"A model with dropout and BatchNorm is scored inside "}<code>{"no_grad()"}</code>{" after training. A second engineer “fixes” its randomness by resetting the random seed before each prediction."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Separate whether gradients are recorded, whether modules use training behavior, and whether a random draw is repeated."}</Prose>

</details>

<details>

<summary>Worked solution</summary>

<Prose>{""}<code>{"no_grad()"}</code>{" alone leaves training behavior active. Resetting the seed repeats randomness rather than making the intended deterministic predictor; BatchNorm state can still change. Use "}<code>{"eval()"}</code>{" plus no gradient recording for ordinary validation. For an explicitly requested MC procedure, selectively enable dropout and draw fresh masks without modifying BatchNorm state."}</Prose>

</details>

<H3>{"5. Choose from evidence"}</H3>

<Prose>{"Model A scores training CE 0.01 and validation CE 0.20. Model B scores 0.30 on both. Which has the smaller gap, and which has the better observed validation loss?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Compute the two gaps, then compare the validation objective independently of those gaps."}</Prose>

</details>

<details>

<summary>Worked solution</summary>

<Prose>{"B has zero gap; A has lower validation loss. B's small gap is compatible with underfitting. These values are a hypothetical diagnostic, not the digit measurements. The gap alone is not the selection objective."}</Prose>

</details>

<H3>{"6. Count blocks and distinguish conventions"}</H3>

<Prose>{"Six blocks use a zero-first schedule ending at drop probability 0.4. Find the rates and expected active count. A programmer computes every correction before masking. Does your answer predict saved computation?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Write the endpoint schedule using block indices beginning at zero. Then distinguish contributing branches from executed branch functions."}</Prose>

</details>

<details>

<summary>Worked solution</summary>

<Prose>{"Rates "}<InlineMath>{"[0,0.08,0.16,0.24,0.32,0.4]"}</InlineMath>{" sum to 1.2, so expected active count is 4.8. Every correction was computed; 20% fewer active contributions does not imply 20% less computation."}</Prose>

</details>

<H3>{"7. Explain a zero uncertainty score"}</H3>

<Prose>{"Every MC pass assigns probability 0.99 to the same wrong class. What does low mask disagreement establish?"}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"Ask what changes across the sampled predictions and what information about correctness the masks actually provide."}</Prose>

</details>

<details>

<summary>Worked solution</summary>

<Prose>{"The sampled masks agree, not that the prediction is correct or the input familiar. More passes estimate that agreement more precisely. Assess probability quality and any deferral policy against observed outcomes on relevant held-out data."}</Prose>

</details>

<H2>{"10. Continue and read another explanation"}</H2>

<Prose>{"You can now trace a sampled mask through values, gradients and mode changes, distinguish masking units, and interpret an actual validation comparison. Next, "}<a href={"/learn/path/full-curriculum/convolution-pooling-receptive-fields?module=deep-learning-fundamentals"}>{"Convolution, Pooling & Receptive Fields"}</a>{" explains how spatially arranged features are created and combined—the structure that made channel and region masks meaningful here."}</Prose>

<Prose>{"For another learning route, "}<a href={"https://d2l.ai/chapter_multilayer-perceptrons/dropout.html"}>{"Dive into Deep Learning §5.6"}</a>{" offers a small network diagram and a from-scratch/built-in comparison. Its example uses Fashion-MNIST and a different experiment budget. Use it to connect the masked diagram to code, not as a substitute for checking this lesson's outcomes."}</Prose>

<Prose>{"The "}<a href={"https://jmlr.org/papers/v15/srivastava14a.html"}>{"2014 JMLR dropout paper"}</a>{" is the historical reference: §§4–5 formalize model/training, §7 studies rates and model averaging, and §9 explores marginalization. Its symbol "}<InlineMath>{"p"}</InlineMath>{" is a "}<strong>{"keep"}</strong>{" probability; this lesson uses "}<InlineMath>{"p"}</InlineMath>{" for "}<strong>{"drop"}</strong>{" probability. The exact input-dropout squared-loss penalty is taught in the "}<a href={"/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml"}>{"Classical ML regularization lesson"}</a>{"; the deep nonlinear objective here should not be silently replaced by a generic L2 penalty. The paper's RBM and unsupervised-pretraining extensions are further probabilistic-model study, not prerequisites for this route."}</Prose>
  </div>,
};
