import { Prose, H2, H3, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock as SharedMathBlock } from '../../components/content/Math.jsx';
import { LessonTable } from '../../components/lesson-labs/LessonElements.jsx';
import MechanismProgram from '../../components/lesson-labs/MechanismProgram.jsx';
import mechanismProgram from '../transfer-learning-mechanism-program.js';
import { TransferReuseFigure, TransferFreezeLab, TransferPartitionsFigure, TransferProgram, TransferLoraLab, TransferAdapterBudget, TransferEvidenceLab, TransferCheckpointLab, TransferScheduleFigure, programUrl, dataUrl, provenanceUrl } from '../../components/lesson-labs/TransferLearningLabs.jsx';

function MathBlock({children}) { return <div className="transfer-equation" role="region" tabIndex={0} aria-label="Equation; scroll horizontally if needed"><SharedMathBlock>{children}</SharedMathBlock></div>; }

export default {
 title: 'Transfer Learning & Fine-Tuning: Reuse, Adapt, and Verify',
 readTime: '~60 min read + 60–90 min exploration and practice; optional deeper branches ~30 min',
 hasIntegratedGuide: true,
 content: () => <div className="transfer-lesson">
<aside className="transfer-downloads"><h3>Learning route</h3><p>Start with a backbone and a new head, investigate freezing and low-rank updates, then choose from actual measured evidence. Before starting: forward/backward passes, squared error and cross-entropy, simple matrix multiplication, and the preceding normalization lesson.</p><nav aria-label="Transfer learning lesson sections"><ol><li><a href="#transfer-section-1">{"1. What exactly moves from one problem to another?"}</a></li><li><a href="#transfer-section-2">{"2. Choose what the target task may change"}</a></li><li><a href="#transfer-section-3">{"3. A complete small transfer pipeline"}</a></li><li><a href="#transfer-section-4">{"4. Make an update without replacing the whole weight matrix"}</a></li><li><a href="#transfer-section-5">{"5. Change features through a bottleneck adapter"}</a></li><li><a href="#transfer-section-6">{"6. Deeper: derivatives tell you what “frozen” actually preserves"}</a></li><li><a href="#transfer-section-7">{"7. Read the evidence before choosing the method"}</a></li><li><a href="#transfer-section-8">{"8. A usable checkpoint includes meaning, not only tensors"}</a></li><li><a href="#transfer-section-9">{"9. Further choices once the basic comparison is sound"}</a></li><li><a href="#transfer-section-10">{"10. Practice and transfer"}</a></li><li><a href="#transfer-section-11">{"Where to go next"}</a></li><li><a href="#transfer-section-12">{"References and other ways to learn"}</a></li></ol></nav></aside>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Toggle parameter updates, gradient recording and module mode; edit LoRA factors/rate; change parameter budgets over saved validation candidates. Show parameters, buffers, gradients and before/after function values separately. Display eligible candidates and the validation-selected winner live; the one retained test result remains clearly identified as previously observed. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to select an adaptation strategy under a real budget and avoid confusing frozen weights with frozen behavior or repeated inspection with a fresh test."}</Prose>

<Prose>{"A model has learned to recognize handwritten digits 0–4. You now need a model for digits 5–9, with only eight labeled examples of each new digit. Can the first model help?"}</Prose>

<Prose>{"Possibly. Its hidden layers may already respond to useful stroke patterns. They may also discard distinctions that the new task needs. "}<strong>{"Transfer learning means reusing something learned on one problem to help with another. Whether it helps is an experimental question."}</strong>{""}</Prose>

<Prose>{"Here you will replace a prediction head, decide which parts may change, make a low-rank update by hand, and run a complete offline comparison. The experiment includes a transferred model that performs worse than training from scratch. That result is part of the lesson."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow sections 1–5, run or inspect the experiment in section 7, and solve practice 1–3. You should be able to explain what was transferred, what was frozen, and which evidence justified choosing a model. Sections 6 and 9 provide optional depth on low-rank derivatives, schedules, and larger-model methods; practice 4–6 checks that depth. You do not need to know convolution or Transformer attention to complete the first pass."}</Prose>

<div id="transfer-section-1" className="transfer-section-anchor" /><H2>{"1. What exactly moves from one problem to another?"}</H2>

<Prose>{"Write a classifier as two cooperating functions:"}</Prose>

<MathBlock>{"x \\longrightarrow h=g_\\psi(x) \\longrightarrow a=h_\\theta(h)."}</MathBlock>

<Prose>{"The "}<strong>{"backbone"}</strong>{" "}<Math>{"g_\\psi"}</Math>{" turns an input into a feature vector. The "}<strong>{"head"}</strong>{" "}<Math>{"h_\\theta"}</Math>{" turns that vector into output scores, called logits. The weights "}<Math>{"\\psi,\\theta"}</Math>{" determine both functions. Cross-entropy compares those scores with the label; backpropagation computes how trainable weights should change."}</Prose>

<Prose>{"In our digit model, 64 pixel values become 32 hidden values, then 16 features, then five scores. The source scores mean digits 0, 1, 2, 3, 4. The target scores must mean 5, 6, 7, 8, 9. Both heads have five outputs, but their label meanings differ. Keeping the old head just because its shape fits is a semantic error."}</Prose>

<TransferReuseFigure />

<Prose>{""}<strong>{"Pretraining"}</strong>{" is the earlier source learning. "}<strong>{"Fine-tuning"}</strong>{" is further training a pretrained model for the target setting. A "}<strong>{"linear probe"}</strong>{" trains a new linear head while keeping the feature function fixed. Transfer also includes reusing learned features without further backbone training."}</Prose>

<Prose>{"A domain describes inputs and their distribution: for example, scanned handwriting from particular writers and equipment. A task describes the desired output: digit identity, writer identity, or whether a scan is readable. The task can change within a similar domain, or the input distribution can change while the task remains the same. In our experiment the label set changes; the scans come from the same collection. This is not a test of transfer across hospitals, languages, or sensors."}</Prose>

<Prose>{"Picture the source backbone as a learned measuring instrument. Reusing it can save learning useful measurements again. However, a measuring instrument that records only weight cannot recover color, however powerful the new decision rule is. A weak probe may indicate missing target information, but it can also reflect poor optimization, unsuitable regularization, or a nonlinear separation that a linear head cannot express. It does not prove the representation contains no useful information."}</Prose>

<Prose>{"For some image networks, earlier features transfer more broadly than later features. The classic study also identified disruption of features that had learned to work together—"}<strong>{"coadaptation"}</strong>{"—as a reason transfer can fail. Layer position alone does not determine usefulness. "}<a href="https://arxiv.org/abs/1411.1792">{"Yosinski et al., "}<em>{"How transferable are features in deep neural networks?"}</em>{""}</a>{""}</Prose>

<div id="transfer-section-2" className="transfer-section-anchor" /><H2>{"2. Choose what the target task may change"}</H2>

<Prose>{"The choices below impose different restrictions on the function you can learn."}</Prose>

<LessonTable caption="Adaptation strategies and their restrictions" headers={["Strategy","Trainable parts","What it asks"]} rows={[[<>{"Train from scratch"}</>,<>{"Backbone and new head, starting from random weights"}</>,<>{"Can target data support learning these features directly?"}</>],[<>{"Linear probe"}</>,<>{"New linear head"}</>,<>{"Are fixed source features already useful for this decision?"}</>],[<>{"Partial fine-tuning"}</>,<>{"New head and selected backbone layers"}</>,<>{"Can a limited part of the representation adapt enough?"}</>],[<>{"Full fine-tuning"}</>,<>{"New head and all backbone layers"}</>,<>{"Does adapting the entire representation help?"}</>],[<>{"Parameter-efficient fine-tuning, or PEFT"}</>,<>{"Small added parameters, often plus a head"}</>,<>{"Can a restricted update adapt the model using fewer trainable values?"}</>]]} />

<Prose>{"“Full” describes which parameters may change, not whether their learning rates must match. A low learning rate for the pretrained backbone and a higher one for a new head can be a reasonable candidate. The head needs to learn its new label meanings; the backbone already implements a learned function."}</Prose>

<Prose>{"Start with an appropriate simple baseline and a probe when available. If the probe fits training examples poorly, allowing representation changes is a useful next investigation. If it fits training examples well but validation performance is poor, adding trainable capacity might worsen overfitting. Inspect the split, preprocessing, labels and error patterns before interpreting every failure as “not enough fine-tuning.”"}</Prose>

<Prose>{""}<strong>{"Negative transfer"}</strong>{" means reuse harms performance relative to an appropriate target-only comparison under a stated protocol. It is not merely low accuracy. A target-only model might perform even worse. Compare the same target rows and evaluation rule, and describe differences in optimization budgets. One experiment does not establish that a source domain is always harmful."}</Prose>

<H3>{"A freeze has three separate meanings to check"}</H3>

<ol><li>{""}<strong>{"Parameter gradients:"}</strong>{" "}<Code>{"requires_grad_(False)"}</Code>{" prevents accumulation of gradients for those parameters."}</li><li>{""}<strong>{"Optimizer membership:"}</strong>{" an optimizer changes the parameters it owns using their available gradients and update rules. Construct it from the intended trainable parameters and clear stale gradients when changing a freeze policy."}</li><li>{""}<strong>{"Model state:"}</strong>{" training mode can update BatchNorm running statistics or enable dropout even when all weights are frozen."}</li></ol>

<Prose>{"The preceding "}<a href="/learn/path/full-curriculum/batch-layer-group-rms-normalization?module=deep-learning-fundamentals">{"normalization lesson"}</a>{" explained why "}<Code>{"eval()"}</Code>{" and "}<Code>{"no_grad()"}</Code>{" are independent. Evaluation mode changes module behavior. Disabling gradient recording changes autograd. Neither substitutes for a complete freeze policy."}</Prose>

<Prose>{"Our backbone uses linear layers and tanh, so it has no running statistics or dropout. For a frozen pretrained backbone that does have such state, a common policy is to call "}<Code>{"model.train()"}</Code>{" and then "}<Code>{"model.backbone.eval()"}</Code>{" at the start of each training epoch. This deliberately keeps the head in training mode and the backbone in evaluation mode. Adapting BatchNorm statistics is another policy; specify and validate it separately."}</Prose>

<Prose>{"Frozen weights can still transmit gradients to their inputs. If an upstream adapter produces "}<Math>{"u"}</Math>{", and a frozen matrix computes "}<Math>{"Wu"}</Math>{", the adapter needs the derivative through "}<Math>{"W"}</Math>{". Wrapping that whole path in "}<Code>{"no_grad()"}</Code>{" would cut it. When a frozen feature extractor receives ordinary data and only a downstream head is trained, features can instead be computed without a graph, or cached if preprocessing is deterministic."}</Prose>

<TransferFreezeLab />

<div id="transfer-section-3" className="transfer-section-anchor" /><H2>{"3. A complete small transfer pipeline"}</H2>

<Prose>{"We use 400 real 8×8 handwritten scans from the UCI Optical Recognition of Handwritten Digits collection: 40 per digit. The values are integers from 0 through 16. Divide by the known feature-range maximum 16; do not estimate a new scale using the holdout. This collection is distinct from MNIST."}</Prose>

<Prose>{"The downloadable "}<a href={dataUrl} download="digits-400.csv">{"data"}</a>{" include "}<Code>{"source_id"}</Code>{", 64 row-major pixels, and "}<Code>{"digit"}</Code>{". The "}<a href={provenanceUrl} download="data-provenance.md">{"provenance"}</a>{" identifies the extraction and license."}</Prose>

<LessonTable caption="Fixed source and target partitions" headers={["Partition","Digits","Rows per digit","Purpose"]} rows={[[<>{"Source training"}</>,<>{"0–4"}</>,<>{"First 30"}</>,<>{"Learn the initial backbone and source head"}</>],[<>{"Source holdout"}</>,<>{"0–4"}</>,<>{"Last 10"}</>,<>{"Describe retention with the original head"}</>],[<>{"Target training"}</>,<>{"5–9"}</>,<>{"First 8"}</>,<>{"Fit each adaptation candidate"}</>],[<>{"Target validation"}</>,<>{"5–9"}</>,<>{"Next 12"}</>,<>{"Compare candidates"}</>],[<>{"Target test"}</>,<>{"5–9"}</>,<>{"Last 20"}</>,<>{"Report the one selected model"}</>]]} />

<TransferPartitionsFigure />

<Prose>{"These are fixed, disjoint row blocks within the extract, not random writer groups. Writer identifiers are unavailable in this file, so the experiment cannot estimate generalization to unseen writers. Row order can also make partitions differ in difficulty. The small experiment teaches the protocol and mechanisms; it is not an official benchmark score. Its source holdout is used for descriptive retention, never for selecting the target model."}</Prose>

<Prose>{"The complete "}<a href={programUrl} download="transfer-experiments.py">{"CPU program"}</a>{" defines every model, optimizer, split, training loop and calculation. Put it beside the CSV. One setup for a separate learner environment is:"}</Prose>

<div className="transfer-code" role="region" tabIndex={0} aria-label="Setup commands"><CodeBlock language="bash">{"python -m venv .venv\n# Activate .venv using your operating system's activation command.\npython -m pip install torch==2.14.0 numpy==2.3.5\npython transfer-experiments.py"}</CodeBlock></div>

<TransferProgram />

<Prose>{"The recorded run used Python 3.12.14 and PyTorch 2.14.0+cpu. It needs no pretrained download, account or GPU. The program limits PyTorch to one CPU thread, writes "}<Code>{"calculated-inputs.json"}</Code>{" and prints validation results followed by the selected test result. Exact floating-point last digits may vary across environments."}</Prose>

<Prose>{"Follow its data flow before changing settings:"}</Prose>

<ol><li>{"Initialize a 64→32→16 tanh backbone and a five-output source head. Train on the 150 source training rows for 400 full-batch Adam updates, learning rate 0.01."}</li><li>{"Save both the original random backbone and the learned backbone. Initialize one new five-output target head. Every method in that seed receives an identical copy of this head."}</li><li>{"Construct six candidates: scratch, probe, full fine-tuning, discriminative rates, rank-2 LoRA, and a bottleneck-4 adapter. Train each for 300 full-batch updates on the same 40 target rows."}</li><li>{"Compare their final validation cross-entropies. Seed 1 is the predeclared selection seed. Choose its lowest validation loss; exact ties use the listed method order."}</li><li>{"Seeds 2 and 3 provide sensitivity comparisons on validation only. They do not change the selection rule. Freeze the exact selected seed-1 weights, with no refit, and evaluate that model on the 100 target test rows."}</li></ol>

<Prose>{"The head rate is 0.01 throughout. Scratch and full fine-tuning use 0.001 for both backbone layers. Discriminative fine-tuning uses 0.0001 for the lower layer and 0.001 for the upper layer. LoRA factors and adapter weights use 0.01. These are six declared training procedures, not a sweep proving each method has received its optimal settings."}</Prose>

<div id="transfer-section-4" className="transfer-section-anchor" /><H2>{"4. Make an update without replacing the whole weight matrix"}</H2>

<Prose>{"Suppose a pretrained linear layer uses "}<Math>{"W\\in\\mathbb R^{d\\times k}"}</Math>{", with "}<Math>{"k"}</Math>{" inputs and "}<Math>{"d"}</Math>{" outputs. Full fine-tuning can independently change its "}<Math>{"dk"}</Math>{" weights. "}<strong>{"Low-rank adaptation"}</strong>{", or LoRA, adds a product of two smaller matrices:"}</Prose>

<MathBlock>{"y=Wx+sB(Ax),\\quad\nA\\in\\mathbb R^{r\\times k},\\quad\nB\\in\\mathbb R^{d\\times r},\\quad\ns=\\alpha/r."}</MathBlock>

<Prose>{"First "}<Math>{"A"}</Math>{" measures "}<Math>{"r"}</Math>{" combinations of the input. Then "}<Math>{"B"}</Math>{" distributes these measurements across output coordinates. This update has rank at most "}<Math>{"r"}</Math>{": its output changes lie in the span of "}<Math>{"B"}</Math>{"'s columns. The restriction applies to the update "}<Math>{"BA"}</Math>{", not to the pretrained matrix "}<Math>{"W"}</Math>{" or to the entire nonlinear network."}</Prose>

<Prose>{"For a 4×6 matrix, rank 2 requires "}<Math>{"2(6+4)=20"}</Math>{" factor weights instead of 24 unrestricted weights. Rank 3 requires 30 factor weights, so “low rank” does not automatically mean fewer parameters at every small shape. In general the saving requires "}<Math>{"r(d+k)<dk"}</Math>{". Also count biases, trainable heads and other modules."}</Prose>

<Prose>{"The original LoRA method freezes "}<Math>{"W"}</Math>{", initializes one factor randomly and the other to zero, and permits merging a trained update into the base matrix. "}<a href="https://arxiv.org/abs/2106.09685">{"Hu et al., "}<em>{"LoRA"}</em>{""}</a>{" Our program uses Gaussian standard deviation 0.1 for "}<Math>{"A"}</Math>{", zero "}<Math>{"B"}</Math>{", "}<Math>{"r=2"}</Math>{", "}<Math>{"\\alpha=2"}</Math>{", and adapters on both backbone matrices. These are explicit teaching choices."}</Prose>

<Prose>{"The scale "}<Math>{"s"}</Math>{" controls the multiplier, but "}<Math>{"\\alpha/r"}</Math>{" does not make changing rank optimization-neutral. Rank changes the number and initialization of factors, the possible update directions, and their gradients. Keep the data fixed when comparing ranks; do not generate a new task for each rank."}</Prose>

<H3>{"Watch the first update"}</H3>

<Prose>{"Use "}<Math>{"W=I_2"}</Math>{", "}<Math>{"A=[1,-1]"}</Math>{", "}<Math>{"B=[0,0]^T"}</Math>{", "}<Math>{"s=1"}</Math>{", "}<Math>{"x=[2,1]^T"}</Math>{", and target "}<Math>{"[0,0]^T"}</Math>{". The initial output is "}<Math>{"[2,1]^T"}</Math>{", because "}<Math>{"B=0"}</Math>{". With mean squared error over the two outputs, loss is "}<Math>{"2.5"}</Math>{"."}</Prose>

<Prose>{"The scalar bottleneck measurement is "}<Math>{"Ax=1"}</Math>{". The gradient for "}<Math>{"B"}</Math>{" is "}<Math>{"[2,1]^T"}</Math>{", while the gradient for "}<Math>{"A"}</Math>{" is zero. After one SGD step of size 0.1, "}<Math>{"B=[-0.2,-0.1]^T"}</Math>{". The output becomes "}<Math>{"[1.8,0.9]^T"}</Math>{", and the loss is "}<Math>{"2.025"}</Math>{"."}</Prose>

<Prose>{"This is why zero initial update does not have to mean zero learning. One factor starts ready to transmit a useful signal. Setting "}<strong>{"both"}</strong>{" factors to zero gives zero gradients for both in this example. The interactive factor editor asks you to inspect which factor can change while displaying these calculations."}</Prose>
<section id="transfer-code-route">
<H3>Implement the factor update without autograd</H3>
<Prose>The full experiment's <Code>LowRankLinear</Code> and <Code>BottleneckAdapter</Code> already define the adaptable paths using ordinary PyTorch layers and operations. For a closer view of what autograd computes, the next standalone program implements the LoRA forward pass and its three pullbacks in NumPy, then matches them to <Code>nn.Linear</Code>, <Code>nn.Parameter</Code>, <Code>F.linear</Code>, mean MSE and plain SGD. It needs Python 3.12, NumPy 2.3.5 and PyTorch 2.14.0, with no data file or pretrained download. Run <Code>python lora-mechanism-bridge.py</Code>.</Prose>
<Prose>In the reusable <Code>forward_and_gradients</Code> routine, rows are examples: X has shape N-by-k, W is d-by-k, A is r-by-k and B is d-by-r. First form <Code>hidden = X @ A.T</Code>; then add <Code>scale * hidden @ B.T</Code> to the frozen base output. For mean MSE across all N×d output entries, <Code>incoming = 2 * residual / residual.size</Code>. Multiplying this incoming derivative backward gives gradients for B, A and X. Section 6 derives those same paths algebraically; Backpropagation owns the general autodiff engine, so this lesson does not build another one.</Prose>
<Prose>The program reproduces loss 2.5→2.025 with B initially zero. It separately runs both-zero factors, zero bottleneck measurement and zero learning rate: each explains a different unchanged result. A changed two-row case starts with nonzero B, so A's derivative is exercised too; its loss moves from 2.375 to 1.517758789. The frozen base has no gradient buffer or parameter change, while the input gradient still includes the base path. Every comparison copies the same values and uses the same float64 reduction and simultaneous update. The merged and separate forward paths agree to floating-point tolerance.</Prose>
<MechanismProgram {...mechanismProgram} title="Read and run the manual / autograd LoRA comparison" />
<Prose>The factor path costs O(Nkr + Nrd) arithmetic and stores O(Nr) bottleneck values, in addition to O(Nd) outputs, O(Nk) input-gradient storage and the base O(Nkd) matrix product. Trainable factor storage is O(r(k + d)); the frozen O(kd) base still occupies memory. Training keeps the two thin projections rather than constructing a dense BA each step. Merging deliberately materializes that d-by-k update once for compatible inference. The routine supports ordinary dense, finite real batches; the comparison uses a fixed linear base without dropout, quantization, mixed precision or distributed state. Those are separate implementation contracts, not implied by these checks.</Prose>
<Prose><strong>Change the implementation:</strong> duplicate every row and target in the two-row case, then compare the loss, A/B gradients and each copied input gradient. Repeat after replacing mean MSE by summed squared error. Use this to diagnose an accidental batch-size multiplier.</Prose>
<details><summary>Implementation exercise solution</summary><Prose>With a mean reduction, duplicating the batch preserves the loss and A/B gradients. Each copied input row receives half its former derivative because the denominator doubles. With a summed reduction, the objective and parameter gradients double while each individual copied input row keeps its original derivative. The frozen base remains unchanged under either objective because it is excluded from the update.</Prose></details>
</section>

<Prose>{"Once trained, form "}<Math>{"W_{\\text{merged}}=W+sBA"}</Math>{". For a plain linear layer this is algebraically equivalent to the separate paths. Floating-point multiplication orders differ: our float64 hand example differs by about "}<Math>{"1.1\\times10^{-16}"}</Math>{"; the seed-1 float32 digit model differs by about "}<Math>{"1.9\\times10^{-6}"}</Math>{" in validation logits. Compare with a suitable tolerance, not a promise of byte-identical outputs."}</Prose>

<Prose>{"Merging removes the extra low-rank path for that fixed adapter. Keeping factors separate makes switching tasks convenient. Quantized weights, active adapter dropout, or incompatible module types require their own merging contract. Preserve the original checkpoint and configuration either way."}</Prose>

<TransferLoraLab />

<div id="transfer-section-5" className="transfer-section-anchor" /><H2>{"5. Change features through a bottleneck adapter"}</H2>

<Prose>{"A bottleneck adapter changes a feature vector rather than directly parameterizing a weight update:"}</Prose>

<MathBlock>{"h'=h+U\\,\\tanh(Dh+b_D)+b_U,\n\\quad D\\in\\mathbb R^{b\\times d},\\quad U\\in\\mathbb R^{d\\times b}."}</MathBlock>

<Prose>{"The narrow intermediate vector has "}<Math>{"b"}</Math>{" values. The added path learns a correction, and the direct "}<Math>{"h"}</Math>{" path keeps the original features available. This introduces the idea of a skip connection; the "}<a href="/learn/path/full-curriculum/residual-connections-skip-connections?module=deep-learning-fundamentals">{"later residual lesson"}</a>{" develops its gradient and architecture consequences."}</Prose>

<Prose>{"Our "}<Math>{"d=16,b=4"}</Math>{" adapter has "}<Math>{"64+4+64+16=148"}</Math>{" parameters. With the 85-parameter target head, 233 parameters train. We initialize the up-projection weight and bias to zero, making this particular adapter exactly the identity initially. The down-projection starts random. After the up-projection moves, gradients can train the down-projection as well."}</Prose>

<Prose>{"This is a small fully specified implementation of the bottleneck idea, not a reproduction of every detail of the original Transformer adapter architecture. "}<a href="https://arxiv.org/abs/1902.00751">{"Houlsby et al., "}<em>{"Parameter-Efficient Transfer Learning for NLP"}</em>{""}</a>{" The program's "}<Code>{"BottleneckAdapter"}</Code>{" defines the complete forward pass and participates in the same target experiment as the other methods."}</Prose>

<Prose>{"An adapter may contain nonlinear computation and therefore is not generally mergeable into one fixed linear weight. It also changes the active feature function even though the base parameters remain untouched."}</Prose>

<TransferAdapterBudget />

<div id="transfer-section-6" className="transfer-section-anchor" /><H2>{"6. Deeper: derivatives tell you what “frozen” actually preserves"}</H2>

<Prose>{"For a single input let "}<Math>{"\\delta=\\partial L/\\partial y"}</Math>{". LoRA's derivatives are"}</Prose>

<MathBlock>{"\\frac{\\partial L}{\\partial B}=s\\,\\delta(Ax)^T,\\qquad\n\\frac{\\partial L}{\\partial A}=s\\,B^T\\delta x^T,\\qquad\n\\frac{\\partial L}{\\partial x}=W^T\\delta+sA^TB^T\\delta."}</MathBlock>

<Prose>{"The frozen base weight has no optimizer update, but "}<Math>{"W^T\\delta"}</Math>{" still contributes to the input gradient. In our saved fixture, a frozen identity matrix with the same squared loss transmits input gradient "}<Math>{"[2,1]^T"}</Math>{", while its weight has no stored gradient."}</Prose>

<Prose>{"With "}<Math>{"B=0"}</Math>{", the "}<Math>{"A"}</Math>{" gradient vanishes on the first step; the "}<Math>{"B"}</Math>{" gradient need not vanish. This is a local explanation, not a claim that identical factor learning rates are optimal. "}<a href="https://arxiv.org/abs/2402.12354">{"LoRA+"}</a>{" investigates different rates for the two factors using width-scaling arguments and experiments. Its findings do not provide a universal rate ratio for our small network."}</Prose>

<Prose>{""}<strong>{"Freezing parameters preserves their stored values. It does not guarantee preservation of the adapted model's old behavior."}</strong>{" Feed the changed representation into the original source head to measure one form of forgetting. Disabling an unmerged adapter can restore the base function when the same base weights, buffers, preprocessing and original head are restored. Retaining a full pre-fine-tuning checkpoint also lets full fine-tuning be reversed."}</Prose>

<div id="transfer-section-7" className="transfer-section-anchor" /><H2>{"7. Read the evidence before choosing the method"}</H2>

<Prose>{"These are actual results after 300 target updates. Cross-entropy is the mean loss in natural-log units; lower is better. Accuracy is shown as a count so the size of the evidence remains visible."}</Prose>

<LessonTable caption="All seed-1 final candidate measurements" headers={["Seed-1 method","Trainable values","Target train correct / 40","Validation CE","Validation correct / 60","Source holdout correct / 50 after adaptation"]} rows={[[<>{"Scratch"}</>,<>{"2,693"}</>,<>{"40"}</>,<>{"0.123143"}</>,<>{"58"}</>,<>{"Not applicable"}</>],[<>{"Probe"}</>,<>{"85"}</>,<>{"37"}</>,<>{"0.738785"}</>,<>{"44"}</>,<>{"49"}</>],[<>{"Full fine-tuning"}</>,<>{"2,693"}</>,<>{"40"}</>,<>{"0.346375"}</>,<>{"53"}</>,<>{"49"}</>],[<>{"Discriminative rates"}</>,<>{"2,693"}</>,<>{"40"}</>,<>{"0.568471"}</>,<>{"48"}</>,<>{"49"}</>],[<>{"LoRA rank 2"}</>,<>{"373"}</>,<>{"40"}</>,<>{"0.369318"}</>,<>{"53"}</>,<>{"44"}</>],[<>{"Adapter width 4"}</>,<>{"233"}</>,<>{"40"}</>,<>{"3.353089"}</>,<>{"39"}</>,<>{"49"}</>]]} />

<Prose>{"Before target adaptation, the source model got 49/50 source holdout rows correct. The probe preserves that result exactly because its backbone and original head remain unchanged. Its new head still cannot fit all 40 target training examples under the declared training procedure."}</Prose>

<Prose>{"Several methods fit all target training examples, yet their validation results differ greatly. The adapter's high validation loss alongside 39/60 correct indicates that some errors receive especially costly probabilities. A training score alone would hide this."}</Prose>

<Prose>{"Scratch has the lowest seed-1 validation CE, so the predeclared rule selects it. The other seeds also favor scratch under these settings:"}</Prose>

<LessonTable caption="Validation-only seed sensitivity" headers={["Method","Seed 2 validation CE; correct / 60","Seed 3 validation CE; correct / 60"]} rows={[[<>{"Scratch"}</>,<>{"0.158462; 58"}</>,<>{"0.082299; 59"}</>],[<>{"Probe"}</>,<>{"0.781923; 42"}</>,<>{"0.785892; 46"}</>],[<>{"Full"}</>,<>{"0.341880; 54"}</>,<>{"0.362536; 51"}</>],[<>{"Discriminative"}</>,<>{"0.423951; 50"}</>,<>{"0.677043; 49"}</>],[<>{"LoRA rank 2"}</>,<>{"0.417762; 52"}</>,<>{"0.333299; 56"}</>],[<>{"Adapter width 4"}</>,<>{"2.350006; 40"}</>,<>{"3.027634; 35"}</>]]} />

<Prose>{"The selected seed-1 scratch model then gets "}<strong>{"77/100 target test rows correct"}</strong>{", with CE "}<strong>{"0.757101"}</strong>{". This is substantially worse than its validation result. Small fixed row blocks need not represent equally difficult populations, and choosing by validation can favor a candidate on that validation set. These data alone cannot isolate the causes of the gap."}</Prose>

<Prose>{"The correct response is to report the gap and the partition limitations. Trying alternatives on these same test labels would turn the test into more development data. A future study could predeclare a stronger split with writer information and more examples, or compare learning rates and source objectives using development data. It would need new final evidence for a fresh performance claim."}</Prose>

<Prose>{"Forgetting also needs careful measurement. Seed-1 LoRA lowers original-head source accuracy from 49/50 to 44/50, although its base weights are frozen. Full fine-tuning retains 49/50 here. This does not prove full fine-tuning always forgets less; it refutes the claim that a frozen base guarantees no forgetting of the adapted function. Accuracy can also remain unchanged while probabilities move: inspect the recorded source cross-entropies."}</Prose>

<Prose>{"All plotted training trajectories come from the saved steps 0, 1, 10, 100 and 300. A connecting line shows those samples; it is not a record of every intervening update or a benchmark of elapsed time."}</Prose>

<TransferEvidenceLab />

<div id="transfer-section-8" className="transfer-section-anchor" /><H2>{"8. A usable checkpoint includes meaning, not only tensors"}</H2>

<Prose>{"To reproduce a prediction, preserve the architecture, weight values, buffers, preprocessing, output-label order and adaptation configuration. For our target task that includes the 64-pixel ordering, division by 16, hidden widths 32 and 16, tanh, and output labels [5, 6, 7, 8, 9]."}</Prose>

<Prose>{"A LoRA checkpoint additionally needs the base identity, targeted layers, factor rank, "}<Math>{"\\alpha"}</Math>{", scaling convention, biases, and any separately trained head. A file containing only "}<Math>{"A,B"}</Math>{" is insufficient to identify the function. The same issue appears with image transforms and language tokenizers: a compatible tensor shape does not establish the same input or label meaning."}</Prose>

<Prose>{"Our program checks a state-dictionary round trip through ordinary lists with the model configuration held fixed; the selected probabilities match exactly in that run. It is a local replay check, not a complete deployment package. When saving a reusable artifact, save a machine-readable configuration alongside it and a small input/output fixture."}</Prose>

<Prose>{"For a cached probe, an additional practical advantage is possible: compute fixed features once and train multiple heads on those features. This is exact only for the preprocessing and backbone state used to produce the cache. Random image augmentation creates different inputs, and adapting BatchNorm statistics changes the feature function; either makes a stale cache inappropriate. Name cache entries by source row, preprocessing version and backbone checkpoint, not merely “features.”"}</Prose>

<Prose>{"For a service hosting many related tasks, one base plus several small adapters can reduce duplicated stored weights. Each task still needs evaluation, correct routing and compatible preprocessing. Merging one adapter favors a fixed serving path; retaining adapters favors switching. This is a concrete architectural tradeoff rather than a guarantee of lower end-to-end latency."}</Prose>

<TransferCheckpointLab />

<div id="transfer-section-9" className="transfer-section-anchor" /><H2>{"9. Further choices once the basic comparison is sound"}</H2>

<H3>{"Unfreeze gradually; make the schedule explicit"}</H3>

<Prose>{"Partial fine-tuning can begin with the head, then release the upper backbone layer, then lower layers. When releasing a layer, include its parameters in the optimizer and decide whether to preserve existing optimizer state. Rebuilding the optimizer resets its moments unless you deliberately transfer them."}</Prose>

<Prose>{"ULMFiT combined language-model pretraining, adaptation to target-domain text, and classifier fine-tuning. It used layer-dependent learning rates, a rise-and-decay schedule, and gradual unfreezing. Its empirically chosen layer-rate divisor 2.6 belongs to that study, not a law of neural networks. "}<a href="https://aclanthology.org/P18-1031/">{"Howard and Ruder, "}<em>{"ULMFiT"}</em>{", §3"}</a>{""}</Prose>

<Prose>{"For an independently specified teaching schedule, let total duration "}<Math>{"T=100"}</Math>{", peak time "}<Math>{"c=10"}</Math>{", floor fraction "}<Math>{"\\rho=1/32"}</Math>{", and peak rate "}<Math>{"\\eta_{\\max}=0.01"}</Math>{":"}</Prose>

<MathBlock>{"q(t)=\n\\begin{cases}t/c&0\\le t\\le c,\\\\(T-t)/(T-c)&c<t\\le T,\\end{cases}\n\\qquad \\eta(t)=\\eta_{\\max}\\,[\\rho+(1-\\rho)q(t)]."}</MathBlock>

<Prose>{"It starts and ends at 0.0003125 and peaks at 0.01. This triangle is a transparent teaching variant, not a claim to reproduce ULMFiT's printed schedule formula exactly. A scheduler must define whether the rate is sampled before or after each update. Warmup limits early step sizes; it does not guarantee preservation of features or a particular kind of minimum."}</Prose>

<TransferScheduleFigure />

<H3>{"Know which part each efficient method changes"}</H3>

<Prose>{"These are optional entry points. Language-model-specific mechanisms are developed after attention and token representations in the later curriculum."}</Prose>

<LessonTable caption="Parameter-efficient method families" headers={["Method","Distinct mechanism","What to investigate before adopting it"]} rows={[[<>{"LoRA"}</>,<>{"Factorized additive weight updates"}</>,<>{"Rank, target layers, initialization, scaling and task head"}</>],[<>{"DoRA"}</>,<>{"Separates weight magnitude and direction, with low-rank directional adaptation"}</>,<>{"Additional state and the evaluated architecture; not an automatic upgrade"}</>],[<>{"QLoRA"}</>,<>{"Trains adapters through a frozen quantized base"}</>,<>{"Quantization format, dequantization computation, memory overhead and supported hardware"}</>],[<>{"Bottleneck adapters"}</>,<>{"Add small feature transformations"}</>,<>{"Placement, nonlinearity and serving overhead"}</>],[<>{"Soft prompt tuning"}</>,<>{"Learns continuous input vectors while the model is frozen"}</>,<>{"Requires differentiable access to the input representation"}</>],[<>{"Prefix tuning"}</>,<>{"Learns continuous conditioning that later tokens can attend to"}</>,<>{"Layer placement, extra sequence/state cost and task fit"}</>]]} />

<Prose>{"DoRA's magnitude/direction split and QLoRA's quantized-base training solve different problems. QLoRA includes NF4 quantization, quantization of scaling information and paged optimizers; its reported 65-billion-parameter experiment on a 48 GB GPU is a particular setup, not a general memory-fit promise. "}<a href="https://arxiv.org/abs/2402.09353">{"DoRA paper"}</a>{", "}<a href="https://arxiv.org/abs/2305.14314">{"QLoRA paper"}</a>{""}</Prose>

<Prose>{"Soft prompts are learned numerical vectors, not automatically discovered human-readable instructions. A hosted text-generation endpoint does not necessarily expose the gradients or embeddings required to train them. Prefix methods condition internal generation differently from merely adding the same input-vector count. "}<a href="https://arxiv.org/abs/2104.08691">{"Prompt tuning"}</a>{", "}<a href="https://arxiv.org/abs/2101.00190">{"Prefix tuning"}</a>{""}</Prose>

<Prose>{"AdaLoRA allocates an update budget across weight matrices using importance estimates and a singular-value-style parameterization. This differs from pruning whichever raw LoRA factor entries happen to be small. IA³ learns multiplicative activation scales; it restricts changes to selected feature rescalings rather than adding an arbitrary matrix update. Their value depends on which restriction matches the task; a catalogue of method names is not a selection procedure. "}<a href="https://arxiv.org/abs/2303.10512">{"AdaLoRA"}</a>{", "}<a href="https://arxiv.org/abs/2205.05638">{"IA³"}</a>{""}</Prose>

<H3>{"Account for memory in units"}</H3>

<Prose>{"For "}<Math>{"P"}</Math>{" parameters, 16-bit weights alone require "}<Math>{"2P"}</Math>{" bytes. Seven billion such weights require 14 GB in decimal units. Raw 4-bit storage would require "}<Math>{"P/2"}</Math>{" bytes, or 3.5 GB, before scale metadata and other overhead."}</Prose>

<Prose>{"An illustrative trainable-parameter budget with FP32 gradients and two FP32 Adam moments adds "}<Math>{"4+8=12"}</Math>{" bytes per trainable value, excluding weights and any master copy. “Optimizer moments” alone are 8 bytes, not 12. Actual implementations may use different precision, allocation or sharding."}</Prose>

<Prose>{"PEFT can greatly reduce gradients and optimizer-state storage while retaining the base model's weight storage and significant activation memory. Sequence length, batch size, where trainable modules sit, and checkpointing affect the latter. Parameter count is therefore a useful exact calculation, not a measurement of browser speed, GPU throughput or total training memory."}</Prose>

<div id="transfer-section-10" className="transfer-section-anchor" /><H2>{"10. Practice and transfer"}</H2>

<H3>{"1. A head that fits but means the wrong thing"}</H3>

<Prose>{"The old head outputs three scores for [cat, dog, horse]. Your new task is [healthy, scratched, broken], also three classes. A colleague loads the old head unchanged because the dimensions match. What should change, and what must be saved for inference?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"distinguish a vector's length from its meaning."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"normally initialize and train a new three-output head for the new task. Assess whether the backbone is useful; shape compatibility is insufficient. Preserve the new class order, input preprocessing, architecture and corresponding weights. Keeping the old head is a candidate initialization only if deliberately tested, not a completed transfer."}</Prose>

</details>

<H3>{"2. Decide from a new validation table"}</H3>

<Prose>{"All candidates use the same 80 validation rows. The rule was declared as lowest validation CE, with a maximum of 500 trainable parameters. Scratch uses 2,000 parameters and CE 0.30. Probe uses 120 and CE 0.55. LoRA uses 480 and CE 0.42. Adapter uses 360 and CE 0.47. Which is eligible and selected? Can you evaluate all four on the final test to reconsider?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"apply the constraint before minimizing."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"probe, LoRA and adapter are eligible; LoRA wins among them. Scratch's lower loss does not satisfy the declared resource constraint. Evaluate the selected artifact on the final test to report its performance. Using test outcomes to change the choice consumes that holdout for development; it no longer supports the original untouched-test claim."}</Prose>

</details>

<H3>{"3. A freeze that still changes predictions"}</H3>

<Prose>{"A probe's backbone parameters stay bit-for-bit unchanged, but its feature vector for the same original image changes between epochs. Name two state or input mechanisms worth checking. When could feature caching be invalid?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Look beyond trainable parameters: what else is read or updated during a forward pass?"}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"check whether BatchNorm running statistics update in training mode, and whether dropout or random augmentation changes the forward computation. A cache is invalid when its preprocessing or backbone function differs from the active one. "}<Code>{"no_grad()"}</Code>{" alone prevents none of those training-mode behaviors."}</Prose>

</details>

<H3>{"4. Change the LoRA example"}</H3>

<Prose>{"Keep "}<Math>{"W=I_2,A=[1,-1],B=0,s=1"}</Math>{", zero target and mean squared loss, but use "}<Math>{"x=[1,3]^T"}</Math>{". Find the first "}<Math>{"B"}</Math>{" gradient and output after one step of size 0.1. Then choose a nonzero input for which this first update vanishes."}</Prose>

<details><summary>Hint</summary>

<Prose>{"calculate "}<Math>{"Ax"}</Math>{" before any matrix gradient."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{""}<Math>{"Ax=-2"}</Math>{", output gradient is "}<Math>{"[1,3]^T"}</Math>{", so "}<Math>{"\\nabla_B=[-2,-6]^T"}</Math>{". The updated "}<Math>{"B=[0.2,0.6]^T"}</Math>{" adds "}<Math>{"[-0.4,-1.2]^T"}</Math>{", yielding "}<Math>{"[0.6,1.8]^T"}</Math>{". The new loss is 1.8. For "}<Math>{"x=[1,1]^T"}</Math>{", "}<Math>{"Ax=0"}</Math>{" and "}<Math>{"B=0"}</Math>{", so both factor gradients vanish despite positive loss. Changing the input can remove the learning signal without changing the optimizer."}</Prose>

</details>

<H3>{"5. Count, then question the count"}</H3>

<Prose>{"A layer has 1,024 inputs and 256 outputs. Find the LoRA factor count at rank 8, excluding bias, and compare with full weight tuning. With 4-byte gradients and two 4-byte moments, how many bytes do those trainable states need? Does that predict total memory?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Write the shapes of both factors, then count gradients and the two optimizer moments separately."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"full tuning has "}<Math>{"1024\\times256=262{,}144"}</Math>{" weights. LoRA has "}<Math>{"8(1024+256)=10{,}240"}</Math>{", or 3.90625% as many. The stated trainable states require 3,145,728 bytes versus 122,880 bytes. Base weights, any master weights, activations and temporary buffers remain outside that calculation."}</Prose>

</details>

<H3>{"6. Design a rank investigation that can answer its question"}</H3>

<Prose>{"A script creates a fresh random target dataset for ranks 1, 2, 4 and 8, then plots accuracy against rank. Redesign it. What can a flat result establish?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Identify which quantities besides rank change across runs, and which behavior a flat correct-count metric might conceal."}</Prose>

</details>

<details><summary>Worked solution</summary>

<Prose>{"keep target rows, splits, base checkpoint, label mapping, training budget and evaluation rule fixed. State initialization and scaling policies, use controlled seed repetitions, and count head parameters as well. Compare development results; select before using final evidence. Flat accuracy means those settings did not change that discrete metric detectably. Check loss and uncertainty. It does not reveal the true rank of the ideal update or prove higher rank can never help."}</Prose>

</details>

<div id="transfer-section-11" className="transfer-section-anchor" /><H2>{"Where to go next"}</H2>

<Prose>{"On the first-pass route, you are ready when you can explain a backbone/head split, construct a deliberate freeze policy, compare transfer with a target-only baseline, and keep selection separate from final reporting. The deeper route adds factor gradients, memory accounting and controlled adaptation experiments."}</Prose>

<Prose>{"The next module topic is "}<a href="/learn/path/full-curriculum/weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals">{"Weight Initialization: Xavier, Kaiming & μP"}</a>{". Transfer starts from learned weights, but a new head, adapter or scratch baseline still needs an initial state. We will investigate how that state changes signal and gradient behavior before training has learned anything."}</Prose>

<div id="transfer-section-12" className="transfer-section-anchor" /><H2>{"References and other ways to learn"}</H2>

<ul><li>{""}<a href="https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html">{"PyTorch: Transfer Learning for Computer Vision"}</a>{" — an alternate practical route with pretrained ResNet18 and ants/bees images. Read the data transforms, head replacement, optimizer construction and custom-image inference. It requires external weights/data and knowledge of convolution. Its shared training loop puts the entire model in training mode, so its “fixed feature extractor” freezes weights while BatchNorm buffers can still update. Apply the explicit state policy taught here. Tutorial updated January 2025; documentation served as 2.14 in the author review."}</li><li>{""}<a href="https://www.youtube.com/watch?v=_JB0AO7QxSA">{"Stanford CS231n 2017, Lecture 7: Training Neural Networks II"}</a>{" — official course video covering optimization and transfer, useful after the first-pass experiment for another explanation of adapting image models. It predates LoRA and current library APIs. The author verified the official title, description and syllabus association, not the full recording."}</li><li>{""}<a href="https://arxiv.org/abs/1411.1792">{"Yosinski et al."}</a>{" — study of generality, specificity and coadaptation; read the experimental setup before generalizing its layer conclusions."}</li><li>{""}<a href="https://aclanthology.org/P18-1031/">{"ULMFiT, §3"}</a>{" — source for the three-stage language-model adaptation strategy and its schedule/unfreezing choices."}</li><li>{""}<a href="https://arxiv.org/abs/2106.09685">{"LoRA, §4"}</a>{" and "}<a href="https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora">{"PEFT 0.20.0 LoRA documentation"}</a>{" — separate the mathematical mechanism from a library's initialization, target-module and merge behavior. The documentation has many model-specific snippets; the self-contained CPU program here does not require that package."}</li><li>{""}<a href="https://arxiv.org/abs/1902.00751">{"Adapter paper"}</a>{", "}<a href="https://arxiv.org/abs/2402.09353">{"DoRA"}</a>{", "}<a href="https://arxiv.org/abs/2305.14314">{"QLoRA"}</a>{", and "}<a href="https://arxiv.org/abs/2402.12354">{"LoRA+"}</a>{" — optional method families with different restrictions and resource goals. Their reported benchmark improvements are evidence for their settings, not predictions for this lesson's data."}</li><li>{""}<a href="https://arxiv.org/abs/2104.08691">{"Prompt tuning"}</a>{" and "}<a href="https://arxiv.org/abs/2101.00190">{"Prefix tuning"}</a>{" — optional bridges after learning token embeddings and attention."}</li></ul>
</div>
};
