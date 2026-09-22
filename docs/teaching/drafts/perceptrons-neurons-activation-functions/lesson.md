# Perceptrons, Neurons & Activation Functions

**Explore as you read.** Edit input coordinates, weights, bias and common scale; move the activation operating point and incoming weight; edit XOR hidden bias and output coefficient. Synchronize contribution bars, boundary distance, hard/smooth outputs, activation value/slope and all four XOR rows. Compare a shared coefficient rescaling with a moved input, and a repaired corner with the remaining corners. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to choose whether the decision boundary, smooth confidence or local sensitivity needs to change; a one-row repair need not solve the whole task.


A handwriting recognizer receives numbers, not the idea of a “7.” It must turn a pattern of pixel intensities into evidence for different digits. A neural network does this with many small calculations: combine some inputs, transform the result, and pass it to other calculations. Training adjusts those combinations.

One such calculation is an **artificial neuron**. The biological name is an analogy; the object we will build is an ordinary mathematical function. Understanding it lets you read a network diagram, construct a model that a single straight boundary cannot express, and judge what changing an activation actually changes.

**First pass:** follow §§1–6, the three short investigations, and practices 1–5. You will build an XOR network by hand train a small digit recognizer. §7 and practices 6–8 are deeper branches: smooth gates, approximation theory and resource accounting. They are useful extensions, not prerequisites for the next lesson. Allow about 55–70 minutes for reading and worked examples, plus 35–50 minutes for practice and the CPU experiment; the advanced branch adds 25–40 minutes.

## 1. A neuron is a weighted question

Suppose two measured inputs are $x_1=2$ and $x_2=-1$. A neuron gives the first input weight 1.5, the second weight−2, and adds an offset−1:

$$
z=w_1 x_1+w_2 x_2+b=(1.5)(2)+(-2)(-1)-1=4.
$$

Each product is a contribution. A positive weight makes a growing input increase the score; a negative weight reverses that relationship. The **bias** $b$ moves the score without requiring an input to change. The weighted sum plus bias is an **affine** calculation. “Linear layer” is the common library name even when a bias is present.

The next operation decides how to use this score:

$$
a=\phi(z).
$$

Here $z$ is the **preactivation**, $\phi$ is the **activation function**, and $a$ is the neuron's output, or **activation**. For a hard yes/no rule, output 1 if $z>0$ and 0 otherwise. For ReLU, output $\max(0,z)$. For sigmoid, output $1/(1+e^{-z})$. The same score 4 therefore becomes 1,4, or approximately 0.982, respectively. None of those is automatically a calibrated probability; interpretation depends on the model, objective and evidence.

[VisualA: two contribution bars feeding an addition node, then an explicitly separate activation curve. Show3+2−1=4 before showing the three possible outputs. The bias enters the sum, not the activation's name.]

### The same calculation has a geometric meaning

With two inputs, the equation $w_1 x_1+w_2 x_2+b=0$ draws a line. It separates positive from nonpositive scores. The weight vector $w$ points perpendicular to that line toward increasing scores. The score is **not generally the distance to the line**. For nonzero $w$, signed Euclidean distance is

$$
d(x)=\frac{w^\top x+b}{\|w\|_2}.
$$

Our weight vector has length $\sqrt{1.5^2+(-2)^2}=2.5$, so distance is 4/2.5=1.6 input-coordinate units. Multiplying every weight and the bias by 2 changes the score to 8 while preserving the same boundary and distance. It leaves the hard decision unchanged, but sigmoid becomes closer to 1. Boundary geometry and output confidence are different properties.

If every weight is 0, the output depends only on the bias. There is then no unique separating line or distance to divide by. In more dimensions, the line becomes a **hyperplane**: the same equation, one fewer dimension than the input space.

**Investigation A — move the evidence.** Change the common coefficient scale and watch the hard decision, signed distance and sigmoid output together. Then move a point or edit one weight or the bias; follow its product contribution and position relative to the boundary. For a changed case, use $x=(1,1)$, $w=(1,-1)$, $b=0$ and move only $x_2$. The live readouts show why the tie and either side receive different decisions.

## 2. A perceptron learns a boundary from mistakes

A **perceptron** couples an affine score with a hard threshold and an update rule. It is a useful first learning algorithm because you can see every correction. Rosenblatt's perceptron work is an early landmark; the modern idea we need is the relation between a mistaken prediction and a changed boundary, rather than a historical claim that one invention explains all later neural networks.

Use labels $y\in\{-1,+1\}$. A positive label should have a positive score and a negative label a negative score. The product $yz$ is positive when their signs agree. We update when $yz\le 0$, including a point exactly on the boundary:

$$
w\leftarrow w+\eta yx,\qquad b\leftarrow b+\eta y.
$$

The positive number $\eta$ is the step size. Why this direction? On that same example, its signed score increases by $\eta(\|x\|^2+1)$ because the bias also moves. Other examples can improve or worsen; the rule is not a promise that every update improves the whole dataset. [Cornell's perceptron notes](https://www.cs.cornell.edu/courses/cs4780/2022sp/notes/LectureNotes06.html) derive this rule and its separable-data guarantee.

Consider AND: an alarm should trigger only if **both** binary indicators are 1. Four rows suffice to define the complete constructed task.

| Inputs | AND label | XOR label |
| --- | ---: | ---: |
| (0,0) | −1 | −1 |
| (0,1) | −1 | +1 |
| (1,0) | −1 | +1 |
| (1,1) | +1 | −1 |

Here XOR means exactly one indicator is 1. The distinction will expose a representational limit.

Run this complete NumPy program. Install NumPy if your environment does not have it: `python -m pip install numpy`. It appends a constant 1 to each row so the bias is simply the last weight.

```python
import numpy as np

points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
augmented = np.column_stack([points, np.ones(len(points))])

def train_perceptron(labels, max_epochs=12):
    coefficients = np.zeros(3)
    for epoch in range(1, max_epochs + 1):
        updates = 0
        for row, label in zip(augmented, labels):
            if label * (row @ coefficients) <= 0:
                coefficients += label * row  # step size = 1
                updates += 1
        predictions = np.where(augmented @ coefficients > 0, 1, -1)
        print(epoch, coefficients.astype(int), updates, predictions)
        if updates == 0:
            break
    return coefficients

print("AND")
and_coefficients = train_perceptron(np.array([-1, -1, -1, 1]))
print("XOR")
xor_coefficients = train_perceptron(np.array([-1, 1, 1, -1]))
```

The tie policy matters: predictions at score 0 are−1, but training still updates on a zero margin for either label. Those are deliberately distinct rules.

The executed AND trace ends with $(w_1,w_2,b)=(3,2,-4)$: the four scores are−4,−2,−1,1. Epoch 8 makes one update; epoch 9 makes none. There are already correct final predictions at epoch 3, but some have zero margin, so training continues. An epoch's update count also includes predictions made using intermediate weights; it is not the number of errors in the final model.

For XOR, each of these 12 passes makes four updates and returns to coefficients 0. The final tie rule predicts−1 for every row: two final errors, not four. More patience cannot make this single boundary solve XOR.

### Why XOR is impossible for one boundary

For $(1,0)$ and $(0,1)$ to be positive, we need

$$
w_1+b>0,\qquad w_2+b>0.
$$

Adding gives $w_1+w_2+2 b>0$. But the two negative examples require $b\le 0$ and $w_1+w_2+b\le 0$, which imply $w_1+w_2+2 b\le b\le 0$. The demands contradict one another.

The perceptron convergence theorem applies when a strict separating boundary exists and inputs are bounded. It does not promise convergence for XOR, noisy labels, or contradictory duplicate examples. It also does not say the final separator has maximum margin.

**Deeper proof checkpoint.** Absorb the bias into augmented inputs, suppose their lengths are at most $R$, and suppose a unit vector $u$ separates them with $y_i u^\top x_i\ge\gamma>0$. With step size 1 and zero initialization, after $M$ updates, $w^\top u\ge M\gamma$, whereas $\|w\|^2\le MR^2$. The latter follows because the cross term at an update is nonpositive. Cauchy–Schwarz gives $M\gamma\le\sqrt M R$, hence $M\le(R/\gamma)^2$. This counts updates under these assumptions, not fixed passes for every implementation.

## 3. Hidden neurons change what the model can express

A **hidden layer** constructs intermediate features. “Hidden” means it is neither the input nor the final output; it does not mean its values are inaccessible.

Let $s=x_1+x_2$, and construct two ReLU features:

$$
h_1=\max(0,s),\qquad h_2=\max(0,s-1),\qquad q=h_1-2 h_2.
$$

| Input | Sum $s$ | $h_1$ | $h_2$ | Output $q$ |
| --- | ---: | ---: | ---: | ---: |
| (0,0) | 0 | 0 | 0 | 0 |
| (0,1) | 1 | 1 | 0 | 1 |
| (1,0) | 1 | 1 | 0 | 1 |
| (1,1) | 2 | 2 | 1 | 0 |

This is exact XOR on the four binary inputs, with 0/1 output labels. The first hidden neuron is a **ramp**, not a Boolean OR gate: at(1,1) it outputs 2. The second ramp starts one unit later. Subtracting twice the delayed ramp cancels the both-active case.

[VisualB: input square → two ramp coordinates → output contributions. Give each of the four examples a persistent symbol in all three views; show the (1,1) row as2−2=0. Between binary corners, this is a continuous piecewise-linear function, not a new Boolean truth table.]

**Investigation B — repair the network.** Start from output weights (1,−1), where the both-active point is wrong. Edit the second output weight while watching all four truth-table rows and their hidden contributions. Next change the second hidden bias from −1 to −0.5. Explore whether any value of that output weight can repair every row: a change that repairs one corner can damage another.

This tiny network demonstrates representation, not successful training: we deliberately chose its weights. Learning those weights from examples is a separate problem.

### Why stacking affine layers alone does not do this

For column-vector notation, two affine layers compose to

$$
W_2(W_1 x+b_1)+b_2=(W_2 W_1)x+(W_2 b_1+b_2).
$$

That is another affine map. More such layers change the parameterization but do not create nonlinear features. This argument is about a chain containing only affine operations. An architecture can contain other nonlinear operations, such as normalization or attention, even if a named activation is removed.

For code, we use **rows as examples**. A batch $X$ with shape $(N,d)$, a layer weight matrix $W$ with shape $(m,d)$, and a bias vector with shape $(m,)$ produce

$$
Z=XW^\top+b,\qquad A=\phi(Z),
$$

with shape $(N,m)$. The same bias vector is added to every row. The activation usually acts separately on each entry. [PyTorch Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) uses this weight orientation. A batch of 5 images with 64 inputs through 32 hidden neurons becomes a 5×32 activation matrix. An arrow in a network diagram represents a weight; a row in a batch is not an extra neuron.

## 4. Activation functions shape values and sensitivities

An activation controls both the forward signal and how sensitive the output is to small changes. The derivative $\phi'(z)$ is the local slope: near the current score, a small change $\Delta z$ produces approximately $\phi'(z)\Delta z$. Backpropagation, the next topic, will combine these local sensitivities throughout a graph.

Three useful starting shapes are:

| Function | Output rule | Local slope | What to notice |
| --- | --- | --- | --- |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z))$ | Bounded between0 and1; flat tails |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$ | Bounded between−1 and1; centered at0 |
| ReLU | $\max(0,z)$ | 0 for $z<0$,1 for $z>0$ | Removes negative scores, keeps positive scores |

At zero, ReLU has a corner and no ordinary derivative. A framework chooses a convention for differentiation there; PyTorch uses 0. The forward function remains well-defined.

At $z=-2$, sigmoid is about 0.1192, tanh−0.9640 and ReLU 0. At $z=2$, their outputs are 0.8808,0.9640 and 2. Sigmoid's maximum slope is 0.25 at 0; tanh's is 1 at 0. A sigmoid at 5 still has slope about 0.00665: small, not mathematically zero.

[VisualC: aligned value and slope panels with the same horizontal $z$ axis and a movable vertical guide. At a ReLU corner mark a hollow derivative endpoint and state the framework convention; do not draw a misleading continuous derivative through zero.]

### Slopes belong to a whole computation

For one scalar neuron $a=\phi(wx+b)$, the sensitivity to its input is

$$
\frac{da}{dx}=\phi'(z)w.
$$

A sigmoid's 0.25 maximum slope does not imply that every network layer shrinks every gradient by 0.25: if $w=4$ and $z=0$, this local product is 1. Conversely, a positive ReLU with $w=0.5$ transmits a factor 0.5 despite its activation slope being 1. Products across many layers can shrink or grow. In vector layers, matrices and their directions matter too.

A ReLU whose preactivation is negative for all examples in the current batch receives zero gradient through this activation for those examples. It may be active on another batch. A unit that stays negative on the entire relevant training distribution can be difficult to recover through that path, but parameter momentum, another loss path, or changes in upstream features can change the situation. Diagnose actual activations and gradients rather than using a universal “too many zeros” percentage.

**Investigation C — explore local sensitivity.** Choose an activation and edit $z$ and the incoming scalar weight. Watch the activation slope and their product together, including its sign and size. Positive ReLU and leaky ReLU agree; move to a negative score to expose their difference. A large activation value is not the same thing as a large derivative.

### Output interpretation is a separate choice

A multiclass model often ends with one **logit**, or unrestricted score, for each class. Softmax turns the vector into nonnegative values summing to 1:

$$
p_k=\frac{e^{z_k-c}}{\sum_j e^{z_j-c}},\qquad c=\max_j z_j.
$$

Subtracting the same constant preserves the result and avoids positive exponential overflow for finite logits. Very negative differences can still underflow numerically. Softmax combines coordinates; it is not a pointwise hidden activation. For logits(1,2,3), the probabilities are approximately(0.0900,0.2447,0.6652). Adding 1000 to every logit preserves them when the stable formula is used.

A binary probability output often uses sigmoid. Independent labels can use separate sigmoids; mutually exclusive multiclass labels commonly use softmax. A real-valued regression target may need an unrestricted output. Choose the output from the target and loss, not because a hidden activation was fashionable. Our classifier below passes raw logits to cross-entropy; that library loss already performs the stable log-probability calculation.

## 5. Train a small network on real handwriting

The previous Classical ML capstone separated fitting, model choice and final reporting. Keep that discipline here. A neural model is another candidate function class; moving to this module does not imply that it will beat a good classical model on every dataset.

Our offline file contains 400 actual digit images,40 per class, from E.Alpaydin and C.Kaynak's **UCI Optical Recognition of Handwritten Digits** dataset. Each 8×8 image has 64 integer values from 0 to 16. These represent counts of active pixels in original 4×4 blocks, not arbitrary intensity values invented for a plot. The file is a balanced selection from scikit-learn's 1797-row copy of the historical UCI test partition. We make a fresh instructional train/validation split within that selection; these results do not reproduce the original writer-separated benchmark or measure transfer to unseen writers. [Dataset source and CC BY 4.0 attribution](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+), [loader description](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

[VisualD: a small grid of actual8×8 examples, with digit label and stable source ID, linked to one flattened64-entry row. Use a0–16 intensity legend. It should make “flatten” visible without implying that the fully connected model knows pixel adjacency.]

The question is narrow: **under the same small training protocol, how much does hidden activation choice change training loss and validation mistakes?** We predeclare six activations and three initialization seeds. For each seed, resetting before model construction gives each activation the same initial affine parameters. We keep the data split, width, optimizer and number of updates fixed. That controls several confounders; it does not tune each activation to its own best configuration.

The loss used here is mean cross-entropy: for each image, take the negative natural logarithm of the probability assigned to its actual digit, then average over images. A correct-label probability of 0.5 contributes about 0.693; a probability of 0.9 contributes about 0.105. The loss rewards confident correct predictions and penalizes confident wrong ones. The library computes these probabilities stably from logits inside the loss.

Training repeats five operations:

1. Compute logits for the training images.
2. Compare logits with the correct labels using a loss.
3. Differentiate that scalar loss with respect to weights and biases.
4. Update those parameters using the optimizer.
5. Clear old gradients before the next calculation.

A **gradient** lists how the loss changes with each parameter. Adam is the optimizer used here; you do not need its internal moment equations to follow the forward model. The next lesson opens the differentiation step, and the Loss Functions lesson explains the objective in detail.

Save the following program as `compare_activations.py` and keep the accompanying download `digits-400.csv` in the same directory. No data download is required when it runs. The author calculation used Python 3.12.14, NumPy 2.3.5, scikit-learn 1.9.1 and PyTorch 2.14.0+cpu. For a fresh CPU environment:

```text
python -m pip install numpy==2.3.5 scikit-learn==1.9.1
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python compare_activations.py
```

```python
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
rows = np.genfromtxt(
    Path(__file__).with_name("digits-400.csv"),
    delimiter=",", names=True
)
features = np.column_stack([
    rows[f"pixel_{index}"] for index in range(64)
]).astype(np.float32) / 16.0
labels = rows["digit"].astype(np.int64)
train_ids, valid_ids = train_test_split(
    np.arange(len(rows)), test_size=120,
    stratify=labels, random_state=22
)
x_train = torch.tensor(features[train_ids])
y_train = torch.tensor(labels[train_ids])
x_valid = torch.tensor(features[valid_ids])
y_valid = torch.tensor(labels[valid_ids])

activations = [
    ("sigmoid", nn.Sigmoid),
    ("tanh", nn.Tanh),
    ("relu", nn.ReLU),
    ("leaky_relu", lambda: nn.LeakyReLU(0.1)),
    ("gelu", nn.GELU),
    ("silu", nn.SiLU),
]
print("activation seed train_loss valid_correct/120")
for seed in (1, 2, 3):
    for name, make_activation in activations:
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(64, 32),
            make_activation(),
            nn.Linear(32, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for step in range(200):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final_loss = F.cross_entropy(model(x_train), y_train).item()
            correct = int((model(x_valid).argmax(dim=1) == y_valid).sum())
        print(name, seed, f"{final_loss:.6f}", f"{correct}/120")
```

Dividing by 16 uses the known measurement range, not a statistic fitted using validation examples. The matrices have shapes 280×64 →280×32 →280×10 during training. Labels have shape 280 and integer class IDs 0–9. The last layer has no sigmoid or softmax because cross-entropy expects logits.

The default linear-layer initialization is held fixed across activation comparisons, rather than tailored to each one. Initialization is a later topic. These activation modules contain no learned parameters; some other activation families, such as PReLU, do.

The executed author calculation, using the same data and update loop, produced:

| Activation | Training loss, seeds1 /2 /3 | Validation correct, seeds1 /2 /3 |
| --- | --- | --- |
| Sigmoid | .017062 / .015584 / .016383 | 118 /118 /117 |
| Tanh | .003791 / .003635 / .003606 | 117 /118 /117 |
| ReLU | .002392 / .003283 / .002958 | 118 /118 /116 |
| Leaky ReLU, slope.1 | .002210 / .001647 / .002250 | 118 /118 /116 |
| GELU, exact | .001848 / .002123 / .002102 | 118 /118 /117 |
| SiLU | .001847 / .002045 / .001948 | 118 /117 /117 |

Training cross-entropy is a mean in natural-log units per example; smaller means a higher geometric mean of the probabilities assigned to true training labels. Validation counts have denominator 120. A one-case difference is 0.833 percentage points. Equivalent program blocks are prepared above; the saved author calculation was executed, while fresh-environment installation and independent program replay remain later checks.

[VisualE: measured training-loss traces at updates0,1,10,50,100,200 with separate seed traces, plus aligned validation-correct dots. A log loss axis can reveal late changes but must retain update0 and label the transform. Use the actual saved inputs; connect sampled points without inventing unmeasured intermediate observations.]

Sigmoid fits this shallow, scaled task well. Lower training loss does not consistently yield fewer validation errors. Several comparisons are exact ties in correct count. Three seeds on one small selected dataset establish neither a universal ranking nor equivalence of functions. If you change width, learning rate, depth, split or number of updates, you have a new experiment to report. These validation results have already been used for comparison; they are not a fresh final test.

Inspect wrong images as well as the table. Are two networks wrong on the same examples? A difference of one in the total could hide multiple repairs and new mistakes, just as in the capstone. The supplied calculated record retains per-example validation predictions to support that paired inspection.

## 6. Choose and diagnose with the mechanism in mind

Start with the role of the layer and the surrounding architecture. ReLU is a simple hidden-layer baseline. Smooth GELU or SiLU variants are worth comparing when the architecture and optimization protocol motivate them. Sigmoid remains useful for probabilities and bounded gates; tanh for centered bounded transformations. A function's historical popularity does not replace a validation experiment.

Before changing the activation, check the evidence that would distinguish causes:

| Observation | What to inspect | Why an activation swap may not fix it |
| --- | --- | --- |
| Loss never changes | Parameter registration, gradients, optimizer step, learning rate | A disconnected parameter cannot learn through any activation |
| Many zeros after ReLU | Preactivations by example and layer, gradient paths | Zero on one batch is not permanent inactivity |
| Tiny gradients | Local slopes, weight scales, depth and loss scaling | Matrix products also control gradient magnitude |
| Exploding values | Inputs, weights, update size, numeric precision | A bounded hidden function does not fix every upstream or output computation |
| Duplicate hidden features remain identical | Incoming and outgoing symmetry at initialization | Identical paths can receive identical updates |
| A loaded model changes after an activation edit | Exact original function, approximation flag and parameters | Same tensor shapes do not imply the same model |

Identical initialization is especially subtle: if hidden units have identical incoming parameters and identical downstream roles, deterministic gradients can keep them identical. Random initialization usually breaks this symmetry; it is not merely a way to make loss start lower. Formal scale choices belong to Weight Initialization.

Changing the activation of a pretrained model changes its computed function. Preserve the expected activation and its approximation settings when reproducing a checkpoint; if adapting them, evaluate the resulting model as a changed model. No universal claim follows that every swap either preserves quality or destroys it.

## 7. Deeper branches: smooth gates, expressive capacity and cost

### Smooth negative values and multiplicative gates

Leaky ReLU keeps slope $\alpha>0$ below zero:

$$
\phi(z)=\begin{cases}z,&z\ge 0\\ \alpha z,&z<0.\end{cases}
$$

At $\alpha=0.1$ and $z=-2$, output is−0.2 and slope 0.1. ELU instead uses $\alpha(e^z-1)$ on its negative branch and $z$ on its positive branch; for $\alpha=1$, it approaches−1 in the negative tail. Saturating negative values and a constant negative slope are different choices.

GELU weights its input by a normal cumulative probability:

$$
\operatorname{GELU}(z)=z\Phi(z),
\qquad
\operatorname{GELU}'(z)=\Phi(z)+z\varphi(z),
$$

where $\Phi$ is the standard-normal CDF and $\varphi$ its density. The CDF is a mathematical weighting rule; actual preactivations need not be normally distributed for the function to be defined. The original motivation relates $z\Phi(z)$ to the expected value of an input-dependent binary mask. The usual implemented GELU is deterministic, not random dropout. [GELU formulation, §2](https://arxiv.org/html/1606.08415v5).

The common approximation is

$$
\frac z 2\left[1+\tanh\left(\sqrt{2/\pi}(z+0.044715 z^3)\right)\right].
$$

Exact and approximate are distinct functions. At 1, their values are about 0.841345 and 0.841192. [PyTorch's GELU documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html) exposes `approximate="none"` and `"tanh"` explicitly.

SiLU is $z\sigma(z)$, also called Swish with fixed scale parameter 1; generalized Swish uses $z\sigma(\beta z)$. Its derivative is $\sigma(z)+z\sigma(z)(1-\sigma(z))$. At−2, SiLU is approximately−0.2384 with derivative−0.0908. A negative input can therefore have a negative local slope. GELU also has a small negative-slope region. Smooth does not mean monotone, and neither function guarantees a nonzero derivative at every point. [SiLU API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html), [Swish paper](https://arxiv.org/abs/1710.05941).

Mish is another smooth option, $z\tanh(\operatorname{softplus}(z))$, where $\operatorname{softplus}(z)=\log(1+e^z)$ should be computed with a stable library function. It illustrates a broader design space; learning its name is less useful than reading its value and derivative curves. It is optional here rather than another candidate silently added to the six-function experiment.

A **gated layer** combines two learned projections. One produces values and the other modulates them. For a row vector $x$, a bias-free SwiGLU block can be written

$$
h=\operatorname{SiLU}(xW_g)\odot(xW_v),\qquad o=hW_o.
$$

The symbol $\odot$ means coordinatewise multiplication. If projected gate inputs are(1,−1) and values are(2,3), the product is approximately(1.4621,−0.8068). These gates are not bounded probabilities. A gate can reverse the sign of a value.

[VisualF: two projection lanes from the same input, a curve only on the gate lane, paired-coordinate multiplication, then output projection. Show both products; a diagram of one activation box would hide the important multiplication.]

A complete small block:

```python
import torch
from torch import nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    def __init__(self, width, hidden_width):
        super().__init__()
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.value = nn.Linear(width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, width, bias=False)

    def forward(self, inputs):
        gated = F.silu(self.gate(inputs)) * self.value(inputs)
        return self.output(gated)

torch.manual_seed(4)
block = SwiGLU(width=6, hidden_width=8)
inputs = torch.zeros(2, 6)
print(tuple(block(inputs).shape))
print(sum(parameter.numel() for parameter in block.parameters()))
print(block(inputs))
```

Expected shape is(2,6), parameter count 144, and every output is 0 because all three projections have no bias and the input is zero. This is a shape/parameter/zero-input check, not a trained model. The excerpt's full replay is deferred; the count and zero result follow directly from its equations.

A plain two-projection feedforward block of input/output width $d$ and hidden width $h$ has $2 dh$ weights, excluding biases. This gated block has $3 dh_g$. An equal weight budget therefore uses $h_g=2 h/3$. With $d=6,h=12,h_g=8$, both have 144 weights. At the same hidden width, gating has 50% more weights. This budget adjustment is explicit in [Shazeer's GLU variants paper, §3.1](https://arxiv.org/html/2002.05202v1); it is not a free improvement at an unchanged parameter count.

### Many simple ramps can make a detailed function

The XOR construction used a sum of ramps. In one dimension, a continuous piecewise-linear function can be written as an initial line plus a new ramp at each slope change:

$$
f(x)=a+m_0 x+\sum_j (m_j-m_{j-1})\operatorname{ReLU}(x-t_j).
$$

Here $t_j$ is a breakpoint and $m_j$ the slope immediately after it. Each ramp contributes nothing before its breakpoint, then changes the total slope. For example,

$$
g(x)=\operatorname{ReLU}(x)-2\operatorname{ReLU}(x-1)+\operatorname{ReLU}(x-2)
$$

makes a triangular pulse:0,0.5,1,0.5,0 at inputs 0,0.5,1,1.5,2. On the binary-input sums 0,1,2, the last ramp is zero and the first two reproduce our XOR output. This connects logic and ordinary function approximation.

[VisualG: three signed ramp contributions and their summed triangle; distinguish negative contribution weight from a negative ReLU output.]

By adding breakpoints, piecewise-linear interpolation can approximate a continuous function on a bounded interval increasingly closely. Broader universal-approximation results concern networks of sufficient width, biases, suitable nonpolynomial activations and approximation on compact domains. “Nonlinear” alone is too weak a condition: polynomial activations at a fixed shallow architecture do not supply this universal family. An existence theorem does not tell us that a chosen small network, dataset, optimizer and finite training budget will find the desired function. [Leshno and colleagues, working-paper statement and threshold condition](https://archive.nyu.edu/handle/2451/14329).

### Count the actual tensor before estimating memory

A hidden activation tensor with batch 8, sequence length 4096 and hidden width 16384 contains

$$
8\cdot 4096\cdot 16384=536{,}870{,}912
$$

elements. At two bytes per element, its raw storage is 1,073,741,824 bytes, exactly 1 GiB. This is one tensor, not the total training footprint. Saved backward intermediates, additional gated branches, parameters, gradients, optimizer state and temporary kernels add storage; fusion or recomputation may avoid retaining some intermediates.

For the same widths, a plain bias-free two-projection block has 134,217,728 weights. Exact equal-budget gated width would be 10922⅔, which is not an integer. Rounding to 11008 as an illustrative multiple of 256 gives 135,266,304 weights, slightly more. Hardware-friendly rounding is a configuration decision, not exact budget equality or a universal performance guarantee. Benchmark the actual shapes and implementation if runtime matters; there is no measured speed ranking in this lesson.

## 8. Practice: explain, compute, diagnose

Try each problem before opening its hint or solution.

### 1. A changed boundary

For $w=(2,-1)$, $b=-1$, classify $x=(1,3)$ with the hard $z>0$ rule. Find signed distance. Then multiply $w$ and $b$ by 3. Which results change?

<details><summary>Hint</summary>
Compute the two contributions before the bias. Distance divides the score by the weight-vector length.
</details>
<details><summary>Solution</summary>
Score is 2−3−1=−2, so the output is 0. Distance is $-2/\sqrt 5\approx-0.8944$. Rescaling gives score−6 and weight length $3\sqrt 5$, so distance and hard output stay unchanged. Sigmoid decreases from about 0.1192 to 0.00247. Rescaling confidence is not moving the boundary.
</details>

### 2. Update the model, then check a different example

Start $w=(0,0),b=0$. Present $x=(2,-1)$ with label+1 and step size 0.5. What changes? What score does the updated model assign to the previously unseen point(0,2)?

<details><summary>Hint</summary>
The current margin is 0, so the rule updates. Include the bias update.
</details>
<details><summary>Solution</summary>
The new weights are(1,−0.5), and bias 0.5. The training point's score is 3; the unseen point's score is−0.5. Improving the presented example is not a claim that every other point becomes positive or correct.
</details>

### 3. Repair a shifted XOR feature

Keep $h_1=\max(0,x_1+x_2)$, but replace $h_2$ by $\max(0,x_1+x_2-0.5)$. Can an output $q=h_1+vh_2$ match all four XOR labels by changing only $v$?

<details><summary>Hint</summary>
Write the equations for sums 1 and 2. They must use the same $v$.
</details>
<details><summary>Solution</summary>
At sum 1, matching 1 requires $1+0.5 v=1$, so $v=0$. At sum 2, matching 0 requires $2+1.5 v=0$, so $v=-4/3$. No single value satisfies both. Repairing one highlighted row is insufficient; change the hidden bias or allow other parameters to change.
</details>

### 4. Follow tensor shapes

A network receives 7 examples with 12 features, uses 5 hidden neurons, and outputs 3 logits. Give both weight shapes, both bias shapes, and the number of learned scalars.

<details><summary>Hint</summary>
Use the PyTorch convention: output features first in a weight matrix.
</details>
<details><summary>Solution</summary>
Weights are 5×12 and 3×5; biases are 5 and 3. Total is 60+5+15+3=83. Intermediate batches are 7×5 and 7×3. The batch size changes computation and activation storage, not the parameter count.
</details>

### 5. Diagnose an activation claim

A colleague says, “ReLU has derivative 1, so a ten-layer positive scalar ReLU chain cannot shrink gradients.” Each affine weight in the chain is 0.5. What is the input sensitivity? What extra fact would you need for a real vector network?

<details><summary>Hint</summary>
Multiply the weight and activation slope at each layer.
</details>
<details><summary>Solution</summary>
Along this all-positive path, the product is $0.5^{10}=1/1024$. The activation contributes 1 each time, but the weights shrink the signal. For a vector network, use the actual weight matrices, activation masks and directions; a single scalar derivative slogan is insufficient.
</details>

### 6. Allocate a gated width

A plain bias-free block uses width 24 and hidden width 60. Find an exactly equal-weight SwiGLU hidden width and both counts.

<details><summary>Hint</summary>
Compare two projections with three projections.
</details>
<details><summary>Solution</summary>
Plain count is $2(24)(60)=2880$. Gated width 40 gives $3(24)(40)=2880$. Keeping hidden width 60 would give 4320, not an equal-budget comparison. Biases or differently shaped projections would require a new count.
</details>

### 7. Build a tent somewhere else

Construct a piecewise-linear pulse that is 0 at 0 and 2, reaches 2 at 1, and stays 0 outside[0,2]. Use three ReLUs and verify at−1,0.5,1,1.5,3.

<details><summary>Hint</summary>
Scale the triangle in §7. Its slope changes at 0,1 and 2.
</details>
<details><summary>Solution</summary>
$2\operatorname{ReLU}(x)-4\operatorname{ReLU}(x-1)+2\operatorname{ReLU}(x-2)$ gives0,1,2,1,0 at those inputs. The last term restores the slope to0 beyond2; omitting it would create a descending line, not a bounded pulse.
</details>

### 8. Extend the experiment without rewriting its conclusion

Change hidden width 32 to 8 for all six activations, keep the split and seeds, and report training losses and paired validation errors. Before running, write which difference you expect and why. What would count as a supported conclusion?

<details><summary>Hint</summary>
The experiment changes capacity and keeps a single learning-rate protocol. It still has no untouched final test.
</details>
<details><summary>Solution and evaluation criteria</summary>
A good report identifies the changed width, all three seeds, all candidates, training loss and validation denominators. It compares which image IDs were repaired or broken, and distinguishes the prediction from the observed result. A supported conclusion is limited to this width, dataset and protocol; it may report ties or a reversed comparison. Reporting only the winning seed, labeling validation as test, or claiming that one activation is always superior fails the task. There is no prewritten accuracy result for this unexecuted extension.
</details>

## 9. References and another way to learn

- [3Blue1Brown: But what is a Neural Network?](https://www.3blue1brown.com/lessons/neural-networks/) — creator-hosted video with a substantial text companion. Use it after §1 to connect handwritten pixels, weighted sums and layers. The text companion was reviewed; the video was not independently watched. Its bounded-neuron imagery is especially natural for sigmoid, while our ReLU outputs can exceed 1.
- [Cornell CS 4780: Perceptron](https://www.cs.cornell.edu/courses/cs4780/2022sp/notes/LectureNotes06.html) — lecture notes with accompanying videos. Use after §2 for bias augmentation and the convergence proof; it is an alternate mathematical route, not required background.
- [Deep Learning, chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html) — textbook route through XOR, hidden units and architecture. The chapter section list was audited for coverage; the full web chapter could not be retrieved during preparation. Start with the XOR section, and save its backpropagation section for the next lesson.
- [GELU paper](https://arxiv.org/html/1606.08415v5) and [GLU Variants Improve Transformer](https://arxiv.org/html/2002.05202v1) — original definitions and particular experiments. Read the formulation and parameter-budget sections before interpreting experimental rankings.
- [PyTorch Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html), [GELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html), and [SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html) — exact shape and function contracts used by the programs.
- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+) — E.Alpaydin and C.Kaynak,1998, CC BY 4.0, DOI 10.24432/C50P49. The accompanying provenance describes our subset and its limitations.

You can now trace a network forward and separate its expressive capacity from how it learns. The next topic, **Backpropagation & Automatic Differentiation**, follows the loss backward through those same operations and explains how a library obtains parameter gradients.
