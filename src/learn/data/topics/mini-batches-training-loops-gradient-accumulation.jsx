// Generated from the complete ten-section manuscript, engine bridge and changed practice.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {MinibatchStateLab,MinibatchMassLab,MinibatchNormalizationLab} from '../../components/lesson-labs/MinibatchLoopLabs.jsx';
import {MinibatchClocksFigure,MinibatchStoresFigure,MinibatchCoefficientsFigure,MinibatchTargetSlotsFigure,MinibatchShapeFigure,MinibatchGraphLifetimeFigure,MinibatchNormalizationWorked,MinibatchDropoutFigure,MinibatchVarianceFigure,MinibatchBoundaryFigure,MinibatchClippingFigure,MinibatchDdpFigure,MinibatchSevenFigure} from '../../components/lesson-labs/MinibatchLoopDiagrams.jsx';
import {MinibatchIrisHistory,MinibatchIrisLab,MinibatchProgram,MinibatchDownloads} from '../../components/lesson-labs/MinibatchLoopStudy.jsx';
export default {title:'Mini-batches, Training Loops and Gradient Accumulation',readTime:'~75 min read + investigations and practice',content:()=> <div className="neural-lesson neural-lesson-neutral minibatch-lesson">
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Change examples and their microbatch boundaries, then follow the loss derivatives into a shared gradient buffer and one optimizer update. Compare unequal target weights and physical normalization groups to discover when the same rows no longer define the same computation."}</Prose>

<Prose>{"Your model can process twelve examples at a time, but you want one update to use thirty-two. Can you run three forward/backward passes and obtain the same update as a batch of thirty-two? Yes, for an appropriate computation—but the last eight examples must receive the same per-example influence as the first twenty-four. Calling every microbatch's loss "}<code>{".mean()"}</code>{" and averaging those three numbers gives a different objective."}</Prose>

<Prose>{"This lesson makes a training loop inspectable. You will follow examples, predictions, losses, gradients, parameters and optimizer memory as separate objects, then build an offline flower classifier whose full-batch and accumulated updates agree. The central question is: "}<strong>{"which examples contribute to this update, with what weight, evaluated at which parameters?"}</strong>{""}</Prose>

<Prose opening="route">{""}<strong>{"First pass:"}</strong>{" read §§1–6, run the scalar trace in §4 and the Iris program in §6, and attempt practices 1–4. Use the state and denominator investigations as you encounter them. You should finish able to write a correct accumulation loop and explain its final partial group. §§7–8 are deeper branches on batch-dependent computation, numerical precision and distributed training; return to these before applying accumulation to such models. Practice 5 tests that transfer. Allow roughly 45–60 minutes for the core reading and 45–75 minutes for programs and independent practice."}</Prose>

<Prose>{"You need to read a two-dimensional array, recognize a scalar derivative, and know that a gradient tells an optimizer how a local change affects loss. Review "}<a href={"/learn/topic/numpy-arrays-broadcasting-vectorization"}>{"NumPy: Arrays, Broadcasting & Vectorization"}</a>{" for shapes and reductions, and "}<a href={"/learn/topic/backpropagation-automatic-differentiation"}>{"Backpropagation & Automatic Differentiation"}</a>{" for the chain rule. We refresh the particular derivative needed below. No GPU is required for the programs."}</Prose>

<H2>{"1. A batch groups data; an update changes a model"}</H2>

<Prose>{"An "}<strong>{"example"}</strong>{" is one input and its target. A "}<strong>{"mini-batch"}</strong>{" groups examples for a computation. An "}<strong>{"epoch"}</strong>{" is a pass through a specified training sampling procedure, commonly one shuffled traversal of the training rows. An "}<strong>{"optimizer step"}</strong>{" applies an update to parameters and, when present, optimizer state. A "}<strong>{"microbatch"}</strong>{" is a smaller chunk whose gradient contributes to an update shared with other chunks. The examples combined for that update form its "}<strong>{"effective batch"}</strong>{"."}</Prose>

<Prose>{"For ten examples, microbatch size four, and two microbatches per update, the sequence is:"}</Prose>

<CodeBlock language={"text"}>{"examples:       [a b c d] [e f g h] | [i j]\nmicrobatch:         1         2     |   3\neffective batch: [------- 8 ------]|[-- 2 --]\noptimizer step:                   1          2\nepoch:          [------------ one traversal ------------]"}</CodeBlock>

<MinibatchClocksFigure/>

<Prose>{"With a finite ordinary map-style dataset of size "}<InlineMath>{"N"}</InlineMath>{", microbatch limit "}<InlineMath>{"b"}</InlineMath>{", and "}<InlineMath>{"K"}</InlineMath>{" chunks per update, keeping all rows and flushing at each epoch end gives "}<InlineMath>{"M=\\lceil N/b\\rceil"}</InlineMath>{" microbatches and "}<InlineMath>{"U=\\lceil M/K\\rceil"}</InlineMath>{" updates. Write down the sampling and remainder policy before using this formula. Sampling with replacement can revisit a row and omit another within an epoch; a streaming dataset may instead define an epoch by a fixed number of draws."}</Prose>

<Prose>{"The loader decides which examples arrive and how they are collated into tensors. The loop decides when to update. For example, PyTorch's "}<code>{"DataLoader(..., batch_size=4, shuffle=True, drop_last=False)"}</code>{" provides batches, but does not create accumulation boundaries for you. "}<code>{"drop_last=True"}</code>{" discards a loader's short final batch; it does not repair an incorrectly scaled accumulation group. Keep it a deliberate data policy. "}<a href={"https://docs.pytorch.org/docs/2.14/data.html#loading-batched-and-non-batched-data"}>{"PyTorch data loading reference"}</a>{"."}</Prose>

<Prose>{"Why group examples at all? Matrix operations can reuse data and amortize framework overhead across rows. Meanwhile, averaging gradients makes the update depend less on any one sampled example. Batch size therefore affects memory, computation and the statistical path taken through training. Gradient accumulation primarily changes how an effective batch fits into working memory. It still performs the microbatches' work; it is not a promise of faster training. "}<a href={"https://d2l.ai/chapter_optimization/minibatch-sgd.html"}>{"Dive into Deep Learning, §12.5"}</a>{"."}</Prose>

<Prose>{""}<strong>{"Pause:"}</strong>{" If you change microbatch size but preserve each effective group and its one optimizer step, which clock can change? The number of forward/backward calls can change while the number of examples and updates stays fixed. If instead you step after every newly sized batch, you change the update clock too."}</Prose>

<H2>{"2. Follow one complete update"}</H2>

<Prose>{"Use a single adjustable weight "}<InlineMath>{"w"}</InlineMath>{" and prediction "}<InlineMath>{"\\hat y_i=w x_i"}</InlineMath>{". These three constructed rows keep the arithmetic visible:"}</Prose>

<NeuralTable caption={"2. Follow one complete update"} headers={[<>{"Row"}</>,<>{"Input "}<InlineMath>{"x_i"}</InlineMath>{""}</>,<>{"Target "}<InlineMath>{"y_i"}</InlineMath>{""}</>,<>{"Prediction at "}<InlineMath>{"w=0"}</InlineMath>{""}</>,<>{"Half-squared loss "}<InlineMath>{"\\ell_i=\\frac12(wx_i-y_i)^2"}</InlineMath>{""}</>,<>{"Derivative "}<InlineMath>{"(wx_i-y_i)x_i"}</InlineMath>{""}</>]} rows={[[<>{"a"}</>,<>{"1"}</>,<>{"2"}</>,<>{"0"}</>,<>{"2"}</>,<>{"−2"}</>],[<>{"b"}</>,<>{"2"}</>,<>{"0"}</>,<>{"0"}</>,<>{"0"}</>,<>{"0"}</>],[<>{"c"}</>,<>{"3"}</>,<>{"1"}</>,<>{"0"}</>,<>{"0.5"}</>,<>{"−3"}</>]]} />

<Prose>{"The derivative follows the chain rule: changing "}<InlineMath>{"w"}</InlineMath>{" changes the prediction by "}<InlineMath>{"x_i"}</InlineMath>{" times as much, while changing the prediction changes half-squared loss at rate "}<InlineMath>{"wx_i-y_i"}</InlineMath>{". Multiplying those local effects gives the last column. The half factor cancels the derivative of the square. PyTorch's ordinary MSE loss has no half factor; our scalar examples explicitly include it."}</Prose>

<Prose>{"The objective is the "}<strong>{"mean per example"}</strong>{":"}</Prose>

<div className="neural-equation"><MathBlock>{"L(w)=\\frac{1}{3}\\sum_{i=1}^{3}\\ell_i(w),\\qquad\nL(0)=\\frac{2+0+0.5}{3}=\\frac56,\n\\qquad g=\\frac{-2+0-3}{3}=-\\frac53."}</MathBlock></div>

<Prose>{"A gradient is not a parameter change. Plain stochastic gradient descent (SGD) makes the separate decision "}<InlineMath>{"w_{\\text{new}}=w-\\eta g"}</InlineMath>{". With learning rate "}<InlineMath>{"\\eta=0.1"}</InlineMath>{", the new weight is "}<InlineMath>{"1/6"}</InlineMath>{". Its predictions are now "}<InlineMath>{"[1/6,1/3,1/2]"}</InlineMath>{". On the same three rows the mean loss is "}<InlineMath>{"67/108\\approx0.620370"}</InlineMath>{", down from "}<InlineMath>{"5/6\\approx0.833333"}</InlineMath>{". The update reduced the combined loss even though row b became less accurate. A mean objective negotiates among examples."}</Prose>

<MinibatchStoresFigure/>

<Prose>{"Real networks have many parameters, and their gradients have matching shapes. For a batch with "}<InlineMath>{"B"}</InlineMath>{" rows and "}<InlineMath>{"D"}</InlineMath>{" input features, a linear classifier in PyTorch computes"}</Prose>

<div className="neural-equation"><MathBlock>{"X_{B\\times D}W^\\top_{D\\times C}+b_{C}\\longrightarrow\n\\text{logits}_{B\\times C}."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"C"}</InlineMath>{" is the number of classes; logits are raw scores. One integer target per row has shape "}<InlineMath>{"(B,)"}</InlineMath>{". Cross-entropy combines each row's logits and target into a scalar loss after reduction. "}<code>{"backward()"}</code>{" computes a gradient of shape "}<InlineMath>{"(C,D)"}</InlineMath>{" for "}<code>{"nn.Linear.weight"}</code>{" and "}<InlineMath>{"(C,)"}</InlineMath>{" for its bias. Averaging the loss does not average the parameter dimensions away. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"}>{"CrossEntropyLoss shapes and target conventions"}</a>{"."}</Prose>

<MinibatchShapeFigure/>

<H3>{"Parameters, gradients and optimizer memory have different lifetimes"}</H3>

<Prose>{"The parameter persists across training. A gradient slot holds contributions for the next update. Optimizer state can retain information from earlier updates. For example, a simple momentum convention is"}</Prose>

<div className="neural-equation"><MathBlock>{"v_t=\\mu v_{t-1}+g_t,\\qquad w_{t+1}=w_t-\\eta v_t."}</MathBlock></div>

<Prose>{"The coefficient "}<InlineMath>{"\\mu"}</InlineMath>{" controls memory. Starting with "}<InlineMath>{"v_0=0"}</InlineMath>{", the first update agrees with plain SGD. Suppose the next effective group has gradient "}<InlineMath>{"+1"}</InlineMath>{", with "}<InlineMath>{"\\mu=0.9"}</InlineMath>{". Then "}<InlineMath>{"v=0.9(-5/3)+1=-0.5"}</InlineMath>{" and the weight moves from "}<InlineMath>{"1/6"}</InlineMath>{" to "}<InlineMath>{"13/60\\approx0.216667"}</InlineMath>{". It still moves upward because stored momentum outweighs the current gradient. PyTorch SGD agrees with this convention for our zero dampening, no Nesterov, no weight decay setup. Its broader parameter choices have additional semantics. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html"}>{"SGD reference"}</a>{"."}</Prose>

<Prose>{""}<code>{"optimizer.zero_grad(set_to_none=True)"}</code>{" resets gradient slots, not weights or momentum. "}<code>{"None"}</code>{" means no gradient has been supplied; it is distinct from a tensor of zeros. PyTorch optimizers skip parameters with "}<code>{"grad=None"}</code>{", whereas a zero gradient can still allow an update through existing optimizer state or weight decay. This distinction is useful when a branch uses only some parameters. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html"}>{"zero_grad reference"}</a>{"."}</Prose>

<Prose>{"The normal lifecycle is: "}<strong>{"clear → forward → loss → backward → optional gradient processing → step"}</strong>{". Clearing immediately after the preceding step also works if the first group's gradients start clear. "}<code>{"model.train()"}</code>{" selects training behavior; it does not start this lifecycle or change weights by itself. "}<a href={"https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html"}>{"PyTorch optimization tutorial"}</a>{"."}</Prose>

<H2>{"3. Accumulation adds derivatives at the same parameters"}</H2>

<Prose>{"Return to rows a, b, c and split them into microbatches "}<code>{"[a,b]"}</code>{" and "}<code>{"[c]"}</code>{". Hold "}<InlineMath>{"w=0"}</InlineMath>{" throughout both forward/backward passes. The first chunk contributes the derivative of "}<InlineMath>{"(\\ell_a+\\ell_b)/3"}</InlineMath>{", which is "}<InlineMath>{"-2/3"}</InlineMath>{". The second contributes the derivative of "}<InlineMath>{"\\ell_c/3"}</InlineMath>{", which is "}<InlineMath>{"-1"}</InlineMath>{". Adding them gives "}<InlineMath>{"-5/3"}</InlineMath>{", exactly the derivative of the full objective from §2."}</Prose>

<Prose>{"The mathematical reason is linearity of differentiation. If "}<InlineMath>{"S_j(\\theta)"}</InlineMath>{" is the sum of losses in chunk "}<InlineMath>{"j"}</InlineMath>{" and "}<InlineMath>{"D"}</InlineMath>{" is the total number of contributing examples, then"}</Prose>

<div className="neural-equation"><MathBlock>{"\\nabla_\\theta \\left(\\frac{\\sum_j S_j(\\theta)}{D}\\right)\n=\\sum_j\\nabla_\\theta\\left(\\frac{S_j(\\theta)}{D}\\right)."}</MathBlock></div>

<Prose>{""}<InlineMath>{"\\theta"}</InlineMath>{" denotes all model parameters together. The denominator is fixed by the data in this group, not learned. Each call to "}<code>{".backward()"}</code>{" adds another derivative to the existing gradient slots. Fresh forwards produce fresh computation graphs, so ordinary accumulation does not require "}<code>{"retain_graph=True"}</code>{". After each backward pass, that chunk's saved activations can be released; the gradient tensors survive."}</Prose>

<MinibatchGraphLifetimeFigure/>

<Prose>{"The equality concerns a particular objective evaluated at the same parameters with the same per-example computation. Use it when examples do not interact across chunks, randomness is absent or aligned, and optimizer state advances once after the complete sum. Batch-dependent layers and cross-example losses are examined in §7. Floating-point summation order can cause small differences even when the mathematical update is identical."}</Prose>

<H3>{"Why averaging microbatch means can change the question"}</H3>

<Prose>{"Our first chunk's mean loss has gradient "}<InlineMath>{"(-2+0)/2=-1"}</InlineMath>{". The second chunk's mean has gradient "}<InlineMath>{"-3"}</InlineMath>{". Averaging those means gives "}<InlineMath>{"(-1-3)/2=-2"}</InlineMath>{", leading to weight "}<InlineMath>{"0.2"}</InlineMath>{" instead of "}<InlineMath>{"1/6"}</InlineMath>{"."}</Prose>

<Prose>{"The error is visible before calculus. Under an equal average of chunk means, rows a and b each get coefficient "}<InlineMath>{"1/4"}</InlineMath>{", while c gets "}<InlineMath>{"1/2"}</InlineMath>{". Under the intended example mean, all three coefficients are "}<InlineMath>{"1/3"}</InlineMath>{"."}</Prose>

<MinibatchCoefficientsFigure/>

<Prose>{"If chunk "}<InlineMath>{"j"}</InlineMath>{" contains "}<InlineMath>{"n_j"}</InlineMath>{" examples and its mean loss is "}<InlineMath>{"\\bar L_j"}</InlineMath>{", the correct combination is"}</Prose>

<div className="neural-equation"><MathBlock>{"L=\\sum_j\\frac{n_j}{\\sum_k n_k}\\bar L_j."}</MathBlock></div>

<Prose>{"Dividing each mean by the number of chunks is the special case where all their denominators are equal. Count what the loss averages; do not assume every tensor called a batch contains the same amount of supervision."}</Prose>

<MinibatchStateLab/>

<H2>{"4. Write the loop around the update boundary"}</H2>

<Prose>{"This complete program implements the scalar calculation. Save it as "}<code>{"trace_update.py"}</code>{", or use the supplied file. Run "}<code>{"python -B trace_update.py"}</code>{" with PyTorch installed. The recorded execution used Python 3.12.14 and PyTorch 2.14.0 on CPU in float64."}</Prose>

<CodeBlock language={"python"}>{"import torch\n\ntorch.set_default_dtype(torch.float64)\nx = torch.tensor([1.0, 2.0, 3.0])\ny = torch.tensor([2.0, 0.0, 1.0])\nw = torch.nn.Parameter(torch.tensor(0.0))\noptimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)\noptimizer.zero_grad(set_to_none=True)\nprint(\"start\", f\"w={w.item():.6f}\", \"grad=None\")\nfor rows in ([0, 1], [2]):\n    loss_sum = 0.5 * ((w * x[rows] - y[rows]) ** 2).sum()\n    (loss_sum / len(x)).backward()\n    print(\"backward\", rows, f\"grad={w.grad.item():.6f}\", f\"w={w.item():.6f}\")\noptimizer.step()\nprint(\"step\", f\"w={w.item():.6f}\", f\"momentum={optimizer.state[w]['momentum_buffer'].item():.6f}\")\noptimizer.zero_grad(set_to_none=True)\nprint(\"clear\", \"grad=None\", f\"w={w.item():.6f}\")"}</CodeBlock>

<Prose>{"Executed output, rounded to six decimals:"}</Prose>

<CodeBlock language={"text"}>{"start w=0.000000 grad=None\nbackward [0, 1] grad=-0.666667 w=0.000000\nbackward [2] grad=-1.666667 w=0.000000\nstep w=0.166667 momentum=-1.666667\nclear grad=None w=0.166667"}</CodeBlock>

<Prose>{"The two lines labeled "}<code>{"backward"}</code>{" recover the separate contributions in §3. The final clear preserves both the learned weight and momentum. Logging uses scalar values; accumulating graph-connected losses into a list and calling backward only at the end would retain the chunks' graphs and undermine the activation-memory purpose."}</Prose>

<H3>{"The final partial group is an actual update"}</H3>

<Prose>{"For the ten-row example in §1, the effective groups have sizes eight and two. Divide the first group's summed losses by eight and the last group's summed losses by two. If you keep dividing each microbatch mean by the nominal "}<InlineMath>{"K=2"}</InlineMath>{", the last group contains only one mean and receives half the intended gradient. If you step only when a microbatch index is divisible by two, the final two examples never produce an update."}</Prose>

<Prose>{"A clear implementation first defines an effective group, counts its actual denominator, then iterates through that group's microbatches. You can buffer its indices or input tensors without keeping their forward graphs. The Iris program uses precisely this layout. Another approach accumulates derivatives of unnormalized loss sums and divides each populated gradient by the final denominator before clipping and stepping. That approach is useful when the denominator becomes known while streaming; it needs extra care with mixed precision as described in §8."}</Prose>

<Prose>{"Keeping a small final group is an explicit optimization choice: each completed group produces one update of its own mean loss, so individual examples in a smaller final group receive a larger coefficient in that update. Matching a large-batch reference means matching this same sequence of groups. You can instead drop or carry the tail into the next epoch, but then you have chosen a different data/update schedule and should count it accordingly."}</Prose>

<H2>{"5. The denominator defines the objective"}</H2>

<Prose>{"For a scalar prediction per example, example count was the denominator. Other losses can average over pixels, tokens, output coordinates or target weights. Write the group objective as"}</Prose>

<div className="neural-equation"><MathBlock>{"L=\\frac{\\sum_i a_i m_i\\ell_i}{\\sum_i a_i m_i},"}</MathBlock></div>

<Prose>{"where "}<InlineMath>{"a_i\\ge0"}</InlineMath>{" is a fixed importance weight and "}<InlineMath>{"m_i\\in\\{0,1\\}"}</InlineMath>{" indicates whether item "}<InlineMath>{"i"}</InlineMath>{" contributes. Its numerator and denominator add across chunks. This weighted-mean convention is a declared objective; some library losses implement other normalizations."}</Prose>

<Prose>{"For a concrete unequal case, suppose chunk A has two contributing targets with loss sum 2 and chunk B has six with loss sum 18. Their means are 1 and 3. The target-level mean is "}<InlineMath>{"(2+18)/(2+6)=2.5"}</InlineMath>{", whereas the equal mean of chunk means is 2. The same arithmetic applies to derivatives because both routes differentiate these differently weighted objectives."}</Prose>

<Prose>{"In language modeling, a batch can contain two sequences of different lengths. If one has two eligible next-token targets and the other six, a token mean gives the longer sequence three times the total mass. A sequence mean first averages each sequence's tokens, then weights the two sequences equally. Both can be intentional objectives; they answer different questions. Padding that carries no target must not increase either one's target count. Target shifting and causal attention are developed in "}<a href={"/learn/topic/language-model-batches-attention-masks-loss-alignment"}>{"Language-Model Batches, Attention Masks & Loss Alignment"}</a>{", a later curriculum entry."}</Prose>

<MinibatchTargetSlotsFigure/>

<Prose>{"Common PyTorch conventions need separate attention:"}</Prose>

<NeuralTable caption={"5. The denominator defines the objective"} headers={[<>{"Loss setup"}</>,<>{"What to sum"}</>,<>{"Denominator for the stated mean"}</>]} rows={[[<>{"Unweighted class-index cross-entropy"}</>,<>{"Loss for each eligible target"}</>,<>{"Number of nonignored targets"}</>],[<>{"Class-index cross-entropy with class weights, no label smoothing here"}</>,<>{"Class-weighted target losses"}</>,<>{"Sum of the eligible targets' class weights"}</>],[<>{"Cross-entropy with probability targets and class weights"}</>,<>{"Class-weighted cross-entropy per target distribution"}</>,<>{"Number of target positions; it does not use the preceding class-weight denominator"}</>],[<>{"MSE over a tensor with multiple output coordinates"}</>,<>{"All squared coordinate errors"}</>,<>{"Number of error elements, unless you explicitly construct another reduction"}</>],[<>{"Our half-squared scalar regression"}</>,<>{"Half-squared error per row"}</>,<>{"Number of rows, or declared weight sum"}</>]]} />

<Prose>{"The cross-entropy distinction and "}<code>{"ignore_index"}</code>{" semantics come from the current "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"}>{"loss reference"}</a>{"; MSE's element reduction is documented in "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html"}>{"MSELoss"}</a>{". Use "}<code>{"reduction=\"sum\""}</code>{" and an explicitly computed group denominator when implementing these objectives across uneven chunks. Do not divide once in the loss and again in the gradient unless the second factor is the intended chunk weight."}</Prose>

<Prose>{"If a whole group has no eligible target mass, its mean is undefined. Skip that group's update and its update-based scheduler tick, or reject it as an input-construction error. Avoid evaluating an empty mean and then multiplying NaN by zero. A zero-loss sum from an all-ignored microbatch can contribute zero while other chunks make the group's denominator positive; batch-dependent layers may still change state during that forward."}</Prose>

<MinibatchMassLab/>

<Prose>{"The same accounting governs evaluation logs. Add detached loss numerators and their denominators over the evaluation dataset, then divide once. Averaging batch means during validation reproduces the small-batch weighting error even when no gradients are computed. Accuracies similarly use total correct divided by total eligible targets. A training loss accumulated while the model changes is an online summary across different parameter states; an evaluation pass at epoch end measures one fixed state. Label those differently."}</Prose>

<H2>{"6. A complete offline experiment on measured flowers"}</H2>

<Prose>{"Fisher's Iris data records sepal and petal length and width for three species. The classification question is whether these measurements help distinguish the species. We use a small neural classifier to ask a narrower training-mechanics question: "}<strong>{"can different microbatch partitions produce the same sequence of learned parameters and momentum buffers?"}</strong>{""}</Prose>

<Prose>{"The packet supplies all 150 observations in "}<a href={"/learn-assets/mini-batches-training-loops-gradient-accumulation/iris.csv"}>{"iris.csv"}</a>{". Features are centimeters and class IDs 0, 1, 2 denote setosa, versicolor and virginica. The data is attributed to Fisher through "}<a href={"https://archive.ics.uci.edu/dataset/53/iris"}>{"UCI Iris"}</a>{", CC BY 4.0. This CSV is exported from scikit-learn 1.9.1's bundled, corrected variant; scikit-learn documents two corrections relative to the older UCI copy. Row IDs and a header were added, with original row order preserved. See "}<a href={"/learn-assets/mini-batches-training-loops-gradient-accumulation/data-provenance.md"}>{"data-provenance.md"}</a>{" for the exact version and checksum."}</Prose>

<Prose>{"Before fitting, the program chooses forty rows per species for training and ten per species for validation, using split seed 17. Only training rows determine feature centering and scaling. The validation rows are inspected at epoch ends and are never differentiated. We keep twenty epochs and the stated hyperparameters fixed, with no search or early stopping. A uniform-probability baseline has mean cross-entropy "}<InlineMath>{"\\log 3\\approx1.098612"}</InlineMath>{"; predicting one constant species is correct on ten of thirty validation rows. There is no separate final test set in this compact teaching experiment."}</Prose>

<Prose>{"The network has shapes "}<code>{"B×4 → B×8 → B×3"}</code>{": a linear layer, tanh, then a linear output layer. It has no dropout or batch normalization. Both runs clone the same initial model, use the same precomputed shuffled row orders, and use SGD with learning rate 0.05 and momentum 0.9. Their effective groups are "}<code>{"32,32,32,24"}</code>{" each epoch. One run processes each entire group; the other partitions thirty-two as "}<code>{"12+12+8"}</code>{", and twenty-four as "}<code>{"12+12"}</code>{". Both make four updates per epoch. Float64 makes the equivalence comparison easy to inspect on CPU."}</Prose>

<Prose>{"The full runnable program is "}<a href={"/learn-assets/mini-batches-training-loops-gradient-accumulation/train_iris.py"}>{"train_iris.py"}</a>{". Keep it beside "}<code>{"iris.csv"}</code>{"; it needs NumPy and PyTorch and performs no downloads. Run:"}</Prose>

<MinibatchProgram file="train_iris.py"/>

<CodeBlock language={"text"}>{"python -B train_iris.py"}</CodeBlock>

<Prose>{"Read the following central loop with the complete file open. The full file supplies imports, split, normalization, model, fixed orders, evaluation, both training runs and printed results; this excerpt focuses on the update boundary."}</Prose>

<CodeBlock language={"python"}>{"for start in range(0, len(order), 32):\n    group = order[start:start + 32]\n    optimizer.zero_grad(set_to_none=True)\n    for offset in range(0, len(group), microbatch_size):\n        rows = group[offset:offset + microbatch_size]\n        logits = model(features[rows])\n        loss_sum = F.cross_entropy(logits, targets[rows], reduction=\"sum\")\n        (loss_sum / len(group)).backward()\n    optimizer.step()"}</CodeBlock>

<Prose>{""}<code>{"len(group)"}</code>{" is 24 in the final update, not the nominal 32. No optimizer state changes while a group's chunks are being processed. Consequently, the same aggregated gradient reaches the same optimizer state, which produces the same next state. Repeating that argument explains why equivalence extends beyond the first update."}</Prose>

<Prose>{"Recorded CPU results from the supplied program, rounded as printed:"}</Prose>

<CodeBlock language={"text"}>{"epoch updates train_loss train_correct validation_loss validation_correct\n0 0 1.037300 56/120 1.050925 15/30\n1 4 0.806121 79/120 0.823158 20/30\n5 20 0.340418 101/120 0.324301 27/30\n20 80 0.097719 116/120 0.069435 30/30\nfull_forward_backward_calls 80\naccumulated_forward_backward_calls 220\nmax_parameter_gap 2.914e-16\nmax_momentum_gap 8.327e-17\nuniform_probability_loss 1.098612\nconstant_class_validation_correct 10/30"}</CodeBlock>

<Prose>{"The accumulated model learns useful structure on this split, and its final parameters and momentum agree with the full-group run within roundoff. The evidence for accumulation is the paired state comparison, not the classification score. The final validation score describes thirty particular rows in one split; it is not a species-recognition guarantee. No runtime or memory benchmark was collected."}</Prose>

<MinibatchIrisHistory/><MinibatchIrisLab/>

<H3>{"Evaluation selects behavior and suppresses gradient recording separately"}</H3>

<Prose>{"Before a training epoch, call "}<code>{"model.train()"}</code>{". For validation, call "}<code>{"model.eval()"}</code>{" and evaluate within "}<code>{"torch.no_grad()"}</code>{". The first selects behavior for mode-sensitive layers; the second disables ordinary reverse-mode graph recording for the forward computation. Neither call updates weights. "}<code>{"eval()"}</code>{" alone still permits derivatives; "}<code>{"no_grad()"}</code>{" alone leaves dropout and batch-normalization training behavior active. The supplied model has neither layer, but spelling out both operations makes the loop's intent explicit. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html"}>{"no_grad reference"}</a>{", "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html"}>{"BatchNorm1d reference"}</a>{"."}</Prose>

<Prose>{""}<strong>{"Try a direct comparison:"}</strong>{" change only "}<code>{"train(12)"}</code>{" to "}<code>{"train(7)"}</code>{", run it and inspect the number of forward/backward calls together with the final parameter and momentum agreement. Keep effective groups, initialization, optimizer and orders fixed. Practice 4 gives the closed reasoning and a second, stronger variation."}</Prose>

<H3>{"Own the accumulation rule; reuse the derivative and optimizer engines"}</H3>

<Prose>{"The mechanism here is the boundary between losses, accumulated derivatives and one update. The complete "}<a href={"/learn-assets/mini-batches-training-loops-gradient-accumulation/train_iris.py"}>{"train_iris.py"}</a>{" owns that boundary in "}<code>{"train"}</code>{": define actual effective-group rows, clear once, divide each chunk's summed loss by the same real group count, call backward per chunk, and step once. "}<code>{"trace_update.py"}</code>{" makes the gradient slot, parameter and momentum lifetimes visible. These are ordinary PyTorch loops, with the accumulation algorithm expressed explicitly rather than delegated to an opaque trainer."}</Prose>

<Prose>{"If you want to reopen the two reused engines, the "}<strong>{"implemented"}</strong>{" "}<a href={"/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals"}>{"Backpropagation lesson"}</a>{" supplies its "}<a href={"/learn-assets/backpropagation/teaching-autodiff.py"}>{"scratch differentiation engine"}</a>{" and "}<a href={"/learn-assets/backpropagation/engine-library-bridge.py"}>{"matched engine/library bridge"}</a>{". The "}<strong>{"implemented"}</strong>{" "}<a href={"/learn/path/full-curriculum/gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars?module=math-foundations"}>{"Gradient Descent Variants lesson"}</a>{" supplies "}<a href={"/learn-assets/gradient-variants/optimizer_library_bridge.py"}>{"manual_step and optimizer-state comparisons"}</a>{". Those programs actually implement the mechanisms; a link to a bare library reference would not replace them. This topic need not recopy either engine to teach a new grouping rule."}</Prose>

<Prose>{"The correspondence matters: the scratch sum of per-example derivatives becomes repeated "}<code>{".backward()"}</code>{" additions into "}<code>{".grad"}</code>{"; the scratch momentum array becomes SGD's "}<code>{"momentum_buffer"}</code>{"; the scratch update clock becomes exactly one "}<code>{"optimizer.step()"}</code>{" per effective group. Actual state comparisons in the Iris experiment check parameters "}<strong>{"and"}</strong>{" momentum, so similar accuracy cannot conceal a wrong update schedule. Microbatching reduces simultaneously retained activation graphs, while parameter, gradient and optimizer storage remain. Retaining graph-connected losses until the end would lose that intended memory benefit."}</Prose>

<Prose>{""}<strong>{"Changed-constraint exercise."}</strong>{" Keep groups of 32 and change the physical microbatch size to seven. Inspect the final group of 24 as well, preserving all input rows. State the denominator at every backward call, including the short last physical chunk."}</Prose>

<MinibatchSevenFigure/>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"A 32-row group has physical sizes 7,7,7,7,4, and each summed loss divides by 32. A 24-row group has sizes 7,7,7,3, and each divides by 24. Each group still advances momentum once. With the existing 120 fitting rows, one epoch has groups 32,32,32,24: nineteen physical forward/backward calls and four optimizer updates. The full-group and accumulated parameter/momentum paths should match to floating-point tolerance because the model has no cross-example operation or stochastic forward layer. An equal average of the five or four physical means changes row weights; a step after each physical chunk changes both parameters and momentum between derivatives."}</Prose>

</details>

<H2>{"7. Deeper branch: an effective batch need not be a physical batch"}</H2>

<Prose>{"The arithmetic proof in §3 assumes that partitioning leaves each loss term's computation unchanged. It can fail before gradients are added."}</Prose>

<H3>{"Batch normalization sees the physical forward batch"}</H3>

<Prose>{"Suppose a layer receives scalar activations "}<InlineMath>{"[0,2,10,12]"}</InlineMath>{". Normalization using their full mean 6 and population variance 26 yields approximately"}</Prose>

<div className="neural-equation"><MathBlock>{"[-1.176697,-0.784464,0.784464,1.176697]"}</MathBlock></div>

<Prose>{"using "}<InlineMath>{"\\epsilon=10^{-5}"}</InlineMath>{". If instead "}<code>{"[0,2]"}</code>{" and "}<code>{"[10,12]"}</code>{" are normalized independently, each pair becomes approximately "}<code>{"[-0.999995,+0.999995]"}</code>{". The third example changes sign: its value 10 is above the full-group mean but below its own pair's mean."}</Prose>

<MinibatchNormalizationWorked/>

<Prose>{"For a downstream trainable scale "}<InlineMath>{"\\theta"}</InlineMath>{", predict "}<InlineMath>{"\\theta z_i"}</InlineMath>{" and use half-squared loss against targets "}<InlineMath>{"[0,0,1,1]"}</InlineMath>{". At "}<InlineMath>{"\\theta=1"}</InlineMath>{", the full-group mean gradient is approximately 0.509709, while correctly weighted local-normalization chunks give 0.999990. The denominator is correct in both; the normalized inputs differ. Gradient accumulation has no way to retroactively replace those forward statistics."}</Prose>

<Prose>{"Running statistics also update on forwards. With initial running mean zero and update coefficient 0.1, one full forward stores 0.6. Two local forwards store "}<InlineMath>{"0.9(0.1)+0.1(11)=1.19"}</InlineMath>{". Even when microbatches have identical means and variances, their training outputs can agree while their repeated running-statistic updates differ. PyTorch uses a population variance for the current normalization and an unbiased estimate for its running variance; the example above traces only its running mean. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html"}>{"BatchNorm1d semantics"}</a>{"."}</Prose>

<MinibatchNormalizationLab/>

<Prose>{"Layer normalization over features within each independent example does not couple the example axis in this way. A contrastive loss whose negatives come from other batch rows, a batchwise ranking loss, or any operation that explicitly compares examples can have the same partition problem as batch normalization. Choose an implementation that preserves the needed cross-example information; simply adding gradients is insufficient."}</Prose>

<H3>{"Dropout changes the realized computation"}</H3>

<Prose>{"Training dropout samples a mask and scales surviving activations. At drop probability 1/2, the survivors are multiplied by two. If the full and chunked computations use the same realized mask for each example, an otherwise separable loss still accumulates correctly. Separate calls can consume random numbers differently; setting the same seed does not by itself guarantee those masks align across different call shapes."}</Prose>

<Prose>{"For a constructed check, take "}<InlineMath>{"x=[1,2,3,4]"}</InlineMath>{", targets "}<InlineMath>{"[1,0,1,0]"}</InlineMath>{", prediction "}<InlineMath>{"\\theta z"}</InlineMath>{" at "}<InlineMath>{"\\theta=1"}</InlineMath>{", and the half-squared mean loss. A keep-mask "}<code>{"[1,0,1,0]"}</code>{" produces "}<InlineMath>{"z=[2,0,6,0]"}</InlineMath>{" and gradient 8. The different mask "}<code>{"[0,1,0,1]"}</code>{" produces gradient 20. Partitioning the first fixed masked array preserves gradient 8. Disabling dropout gives another computation with gradient 6.5. These are exact mask calculations, not sampled learning curves. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html"}>{"Dropout reference"}</a>{"."}</Prose>

<MinibatchDropoutFigure/>

<Prose>{"Stochastic equivalence in distribution and equality of one realized update are different claims. The Iris experiment intentionally uses a deterministic, example-separable network so its state comparison isolates the accumulation rule."}</Prose>

<H3>{"A larger effective batch changes gradient variability"}</H3>

<Prose>{"At fixed parameters, suppose independently drawn examples yield one gradient component with variance "}<InlineMath>{"\\sigma^2"}</InlineMath>{". Averaging "}<InlineMath>{"B"}</InlineMath>{" such draws gives variance "}<InlineMath>{"\\sigma^2/B"}</InlineMath>{": the variance of their sum is "}<InlineMath>{"B\\sigma^2"}</InlineMath>{", and division by "}<InlineMath>{"B"}</InlineMath>{" scales variance by "}<InlineMath>{"1/B^2"}</InlineMath>{". Its standard deviation therefore shrinks as "}<InlineMath>{"1/\\sqrt B"}</InlineMath>{". Sampling without replacement from a finite dataset adds a finite-population correction; correlations between examples also change the calculation."}</Prose>

<Prose>{"You can see the finite case exactly using our three row gradients [−2,0,−3]. Uniform single-row sampling has mean −5/3 and variance 14/9. The three equally likely two-row subsets have means [−1,−2.5,−1.5], with the same expectation but variance 7/18. Taking all three rows has no sampling variability at this fixed weight. These are enumerated possibilities, not a fitted learning curve."}</Prose>

<MinibatchVarianceFigure/>

<Prose>{"Repartitioning the same effective group into microbatches leaves that group's intended gradient unchanged. Increasing the effective group itself changes how gradients are averaged and how many updates fit into an epoch. A learning-rate scaling rule is therefore a separate optimization choice; it does not follow from the accumulation identity. This is the statistical side of the batching tradeoff introduced in §1. "}<a href={"https://d2l.ai/chapter_optimization/minibatch-sgd.html#minibatches"}>{"Dive into Deep Learning, §12.5.2"}</a>{"."}</Prose>

<H2>{"8. Deeper branch: operations that belong at the boundary"}</H2>

<H3>{"Clipping and schedules"}</H3>

<Prose>{"Gradient clipping limits the norm of a vector. Because it is nonlinear, clipping each chunk and adding differs from clipping the complete gradient. For scalar contributions 3 and −2.5 and a bound of 1, clipping their sum gives 0.5; adding their separately clipped values gives "}<InlineMath>{"1+(-1)=0"}</InlineMath>{". To match a clipped full-batch update, first complete the correctly normalized gradient and then clip it once before the optimizer step."}</Prose>

<MinibatchClippingFigure/>

<Prose>{"A learning-rate schedule needs a declared clock. An update-based schedule advances after a successful optimizer update, not after each microbatch. An epoch-based schedule advances after the epoch; a metric-driven schedule receives the prescribed validation metric. PyTorch's ordinary scheduler order places the optimizer step first. If four chunks form one update, four scheduler ticks would accelerate an update-based schedule by four. "}<a href={"https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate"}>{"Optimizer and scheduler reference"}</a>{"."}</Prose>

<Prose>{"Regularization has a boundary too. If the objective includes "}<InlineMath>{"\\lambda R(\\theta)"}</InlineMath>{" once per effective update, add its derivative once, or distribute coefficients across chunks that sum to one. Adding the entire regularizer to every chunk's already normalized data loss multiplies its influence. Decoupled optimizer weight decay happens when that optimizer steps; stepping more often changes its application frequency. Detailed optimizer choices belong to the optimization lessons."}</Prose>

<H3>{"Automatic mixed precision"}</H3>

<Prose>{"Mixed precision can compute selected operations at lower precision; loss scaling multiplies the loss and resulting gradients by a scale factor to help represent small gradients. The accumulation contract still begins with the correctly normalized objective."}</Prose>

<Prose>{"For an AMP update group, use this order:"}</Prose>

<ol start={1}><li>{"Clear gradients and know the group's target denominator."}</li><li>{"For each chunk, compute its loss numerator under the appropriate autocast context, divide by the group denominator, and call backward on the scaled result."}</li><li>{"Keep that scale fixed across the whole group. After the last backward, unscale the optimizer's gradients once."}</li><li>{"Clip the complete unscaled gradient if requested, then let the scaler attempt the optimizer step and update its scale."}</li><li>{"Advance an update-based schedule only if the optimizer update actually occurred; clear for the next group."}</li></ol>

<Prose>{"The scaler can skip an update when gradients contain infinities or NaNs. A skipped attempt still consumed data, so the example clock can advance while the successful-update clock does not. If you accumulate unnormalized sums because the denominator arrives late, normalize the complete unscaled gradient before clipping. The ordering is the same reasoning with normalization moved to the boundary. "}<a href={"https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation"}>{"AMP accumulation and clipping examples"}</a>{"."}</Prose>

<MinibatchBoundaryFigure/>

<H3>{"The same denominator problem appears across devices"}</H3>

<Prose>{"Under default distributed data parallelism, replicas synchronize gradients and average across ranks. If each of "}<InlineMath>{"W"}</InlineMath>{" ranks processes "}<InlineMath>{"b"}</InlineMath>{" examples in each of "}<InlineMath>{"K"}</InlineMath>{" chunks, the ordinary effective example count is "}<InlineMath>{"WbK"}</InlineMath>{". The equality depends on all those contributions receiving the intended weight."}</Prose>

<Prose>{"For unequal target masses, let rank "}<InlineMath>{"r"}</InlineMath>{" supply numerator "}<InlineMath>{"S_r"}</InlineMath>{" and mass "}<InlineMath>{"D_r"}</InlineMath>{", with global "}<InlineMath>{"D=\\sum_r D_r"}</InlineMath>{". The desired gradient is "}<InlineMath>{"\\nabla\\sum_r S_r/D"}</InlineMath>{". Default rank averaging of local means instead gives "}<InlineMath>{"\\frac1W\\sum_r\\nabla S_r/D_r"}</InlineMath>{". To recover the global weighted mean under this default averaging convention, each rank differentiates "}<InlineMath>{"W S_r/D"}</InlineMath>{"; averaging cancels "}<InlineMath>{"W"}</InlineMath>{"."}</Prose>

<Prose>{"For two ranks holding our scalar rows "}<code>{"[a,b]"}</code>{" and "}<code>{"[c]"}</code>{", the local means have gradients −1 and −3. Their rank average is −2. Scaling local summed gradients as "}<InlineMath>{"2(-2)/3"}</InlineMath>{" and "}<InlineMath>{"2(-3)/3"}</InlineMath>{" yields a rank average of "}<InlineMath>{"-5/3"}</InlineMath>{", recovering the same objective as §3."}</Prose>

<MinibatchDdpFigure/>

<Prose>{"DDP's "}<code>{"no_sync()"}</code>{" can defer synchronization during early chunks, but it must surround their forwards as well as backwards. Every rank must follow compatible synchronization boundaries and establish the shared denominator; a rank with zero local mass still participates in the distributed protocol when other ranks have data. Custom communication hooks and uneven-rank termination need their own contract. Continue with "}<a href={"/learn/topic/data-parallelism-ddp"}>{"Data Parallelism (DDP)"}</a>{" for that implementation. "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html"}>{"PyTorch DDP reference"}</a>{"."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"9. Practice: change the data and defend the boundary"}</H2>

<Prose>{"Attempt each before opening its hint or solution. Exact arithmetic is welcome; compare executed float64 results with a small tolerance rather than requiring byte identity across libraries."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. Count work without hiding the remainder"}</H3>

<Prose>{"You have 23 training examples, microbatch size five, and three microbatches per update. Keep every example and flush at epoch end. Give the microbatch sizes, effective group sizes, backward-call count and update count for one epoch. What changes if you turn on loader "}<code>{"drop_last"}</code>{"?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"List the actual chunks before grouping them. Dropping a short loader chunk is different from dropping an incomplete accumulation group."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Chunks are 5,5,5,5,3. The effective groups are 15 and 8, giving five backward calls and two updates. Their denominators are 15 and 8. With loader drop-last, the last three examples disappear: chunks become 5,5,5,5 and groups 15 and 5, with four backward calls and two updates if the remaining group is flushed. A loop that steps only on multiples of three would incorrectly leave the final five processed examples unapplied."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Reconstruct a changed scalar update"}</H3>

<Prose>{"Use inputs "}<code>{"[1,2,4]"}</code>{", targets "}<code>{"[0,1,2]"}</code>{", initial "}<InlineMath>{"w=0"}</InlineMath>{", half-squared mean loss, and SGD learning rate 0.1. Partition as a one-row chunk and a two-row chunk. Compute the correct gradient and weight, then the equal average of chunk-mean gradients. Explain which row the naive rule overweights. Next put all rows in one chunk and predict whether the disagreement remains."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The per-row derivative is "}<InlineMath>{"(wx-y)x"}</InlineMath>{". First write one derivative per row, then write the coefficient assigned to each."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Derivatives are 0,−2,−8. The correct gradient is −10/3, so the weight becomes 1/3. Chunk means have gradients 0 and −5; their equal average is −2.5, giving weight 0.25. The one-row chunk receives half the objective's total mass, so its first row gets weight 1/2 instead of 1/3. It happens to have zero gradient, which reduces the other rows' combined influence. In one chunk, its mean is already the full mean, so both rules agree. Agreement on that null case does not validate the uneven-chunk rule."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Pick the objective for variable-length sequences"}</H3>

<Prose>{"Sequence A has three eligible targets with loss sum 6. Sequence B has one eligible target with loss sum 5. Each is padded to length five. Compute the token mean, sequence mean and incorrect padding-count mean. If A's loss numerator has derivative 9 and B's has derivative −1, give the corresponding token-mean and sequence-mean gradients. What should happen if all positions are ignored?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"The ten tensor slots are not ten targets. For the sequence mean, average within each sequence before averaging across sequences."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The token mean is 11/4=2.75. The sequence mean is (6/3+5/1)/2=3.5. Dividing by ten gives 1.1, incorrectly counting padding. The token-mean gradient is (9−1)/4=2. The sequence-mean gradient is (9/3−1)/2=1. Neither legitimate objective can be selected purely by the number of rectangular tensor slots. With no eligible target mass, the mean is undefined and there should be no optimizer or update-scheduler step for that group."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Transfer the Iris loop"}</H3>

<Prose>{"First change only the accumulated microbatch size from 12 to 7, keeping the thirty-two-example effective groups and final twenty-four-example group. Predict the twenty-epoch call count and the relationship between the two models' states. Then change the effective-group size in both runs from 32 to 25 and use microbatch limits 25 and 6. State the new group sizes, denominators, update count, and the state comparison you would use. Do not predict a new classification score from the old one."}</Prose>

<details><summary>Hint</summary>

<Prose>{"A thirty-two-row group needs five chunks of at most seven; a twenty-four-row group needs four. For the second task, partition 120 into groups of at most 25 before splitting those groups into chunks."}</Prose>

</details>

<details><summary>Solution and success criteria</summary>

<Prose>{"The first change gives 3×5+4=19 calls per epoch, or 380 in twenty epochs. There are still 80 updates, with the same effective data and optimizer schedule; float64 parameter and momentum differences should remain close to rounding error. The author additionally executed this change and observed maximum parameter gap 4.441e−16 and momentum gap 5.551e−17."}</Prose>

<Prose>{"With effective limit 25, groups are 25,25,25,25,20, so each epoch has five updates and twenty epochs have 100. The microbatch limit six produces 6+6+6+6+1 for each group of 25 and 6+6+6+2 for the final 20. The denominators are 25 or 20, never the nominal number of chunks. Compare the two new runs' parameters and momentum after the same 100 updates, with identical initialization and row orders. These new runs need not match the old 32-example runs because gradients are evaluated at different intermediate parameter states. This second changed-group experiment is an independent exercise; its classification outcome is not supplied or preselected. Success is a correctly explained and checked paired-state agreement, not beating the earlier score."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. An exact match breaks after adding a layer"}</H3>

<Prose>{"A deterministic classifier's full-group and accumulated updates agreed. You add training-mode batch normalization, preserve the example-weighted denominator, and the gradients now differ. A colleague suggests dividing by one more factor of "}<InlineMath>{"K"}</InlineMath>{". Explain why that is the wrong repair. Propose a discriminating check and explain a case where outputs agree yet some model state still differs. Then place unscale, clipping, optimizer step and an update-based scheduler around two AMP microbatches."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compare the activations used to compute each loss before changing a coefficient. Track forward-updated buffers separately from trainable weights."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Batch normalization changes each example's normalized activation according to the examples in the physical forward. An extra factor changes gradient size without restoring those activations. First compare full versus local means, variances and normalized outputs on the same fixed inputs; then repeat with one frozen shared set of statistics. The frozen computation should be partition-invariant in the otherwise separable model. If both chunks have the same mean and variance, current normalized outputs can agree with the full batch, while two running-statistic updates differ from one. For example, mean 1, coefficient 0.1 and initial running mean 0 gives 0.1 after one update and 0.19 after two."}</Prose>

<Prose>{"For AMP, keep one scale across both normalized chunk losses and their backwards. Unscale once after both; clip the complete unscaled gradient; attempt the optimizer step; update the scaler; advance the update-based scheduler only for an accepted optimizer update. Clearing between the backwards discards the first chunk. Changing the scale between them mixes incompatible gradient units."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--next" data-lesson-ending="next"><H2>{"10. Check readiness and continue"}</H2>

<Prose>{"Without looking back, identify the examples and denominator belonging to one update, explain why backward leaves weights unchanged, distinguish gradient clearing from optimizer memory, and handle a partial final group. Then explain why a correct accumulation sum can still differ from a physical large batch with training-mode batch normalization. If these are clear, you have a useful contract against which to judge a real training run."}</Prose>

<Prose>{"The next topic in this module is "}<a href={"/learn/topic/neural-training-diagnostics-reproducible-experiments"}>{"Neural Training Diagnostics & Reproducible Experiments"}</a>{". It uses this correct-loop contract to isolate faults, design controlled comparisons and decide what training evidence supports. Later "}<a href={"/learn/topic/mixed-precision-training-fp16-bf16-tf32"}>{"Mixed Precision Training (FP16, BF16, TF32)"}</a>{" and "}<a href={"/learn/topic/data-parallelism-ddp"}>{"Data Parallelism (DDP)"}</a>{" develop the specialized execution branches introduced here."}</Prose></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"References & another way to learn it"}</H2>

<Prose>{""}<strong>{"Alternate explanations and practice"}</strong>{""}</Prose>

<ul><li>{""}<a href={"https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html"}>{"PyTorch, Optimizing Model Parameters"}</a>{" — beginner article and runnable tutorial connecting loss, backward and optimizer steps. Read after §2. Its simple batch-mean reporting should be adapted using §5 when batch sizes differ; its FashionMNIST setup downloads data, while this lesson's Iris route is offline."}</li><li>{""}<a href={"https://d2l.ai/chapter_optimization/minibatch-sgd.html"}>{"Zhang, Lipton, Li and Smola, Dive into Deep Learning §12.5: Minibatch Stochastic Gradient Descent"}</a>{" — free textbook chapter linking vectorized computation, gradient variability and mini-batch optimization. Read after §4 for the systems/statistics connection. Its section agenda and selected PyTorch explanations/code were reviewed; hardware timing examples are the book's environment, not measurements from this packet."}</li><li>{""}<a href={"https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html"}>{"PyTorch, Training with PyTorch — video and companion notebook"}</a>{" — a guided alternate walkthrough from datasets to a training/validation loop, suitable after the first scalar trace. The companion article's introduction, data abstractions and training/validation code were reviewed; the embedded video was not watched, and no timestamps are claimed. It assumes the preceding PyTorch video-series material and uses FashionMNIST and TensorBoard. Keep this lesson's explicit accumulation boundary when adapting its one-batch-per-step loop."}</li></ul>

<Prose>{""}<strong>{"Precise API and data references"}</strong>{""}</Prose>

<ul><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"}>{"CrossEntropyLoss"}</a>{", "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html"}>{"MSELoss"}</a>{", and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html"}>{"zero_grad"}</a>{" — inspect the reduction, target-type and missing-gradient semantics when adapting the programs. These annotations refer to the reviewed PyTorch 2.14 snapshot."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html"}>{"BatchNorm1d"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html"}>{"Dropout"}</a>{" — reference contracts behind §7's physical-batch and realized-mask examples."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation"}>{"AMP examples"}</a>{" and "}<a href={"https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync"}>{"DDP"}</a>{" — intermediate execution references after §8. Their CUDA/distributed examples were read, not run in this CPU packet."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/53/iris"}>{"Fisher, Iris, UCI Machine Learning Repository"}</a>{" and "}<a href={"https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html"}>{"scikit-learn load_iris"}</a>{" — attribution, feature meanings and the corrected bundled variant. The packet's provenance record distinguishes its offline export from the older UCI file."}</li></ul>
<MinibatchProgram file="trace_update.py"/><MinibatchDownloads/></section>
</div>};
