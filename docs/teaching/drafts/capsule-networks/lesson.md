# Capsule Networks: learning which parts belong together

**Explore as you read.** Edit capsule votes, routing iterations, vector magnitude/direction and supported retained image/latent coordinates. Show coupling rows, vote contributions, squash length/direction, current parent vectors and saved/frozen-model outputs. Step routing to inspect its computation, with all current outputs visible. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to distinguish agreement from activation magnitude, pose changes from class evidence and a model intervention from a new empirical result.


A wheel detector firing twice is not enough to recognize a bicycle. The wheels also need a plausible arrangement relative to a frame. A **capsule network** tries to combine evidence about a part's presence with a vector or matrix describing its properties, then asks whether several parts inspect a compatible whole.

In [ConvNeXt](/learn/path/full-curriculum/convnext-modern-cnn-designs?module=deep-learning-fundamentals), we changed how a convolutional network mixes spatial and channel information. Here the question changes: **can the network decide, for this particular input, which higher-level entity should receive each part's evidence?** We will build that computation, train a small classifier and test what the computation does and does not establish.

**First pass:** follow sections 1–6, run the small offline program or inspect its saved results, then attempt practice 1–5 and 8. You only need vector addition, matrix multiplication and the idea that a loss guides parameter updates. Sections 7–8 and practice 6–7 develop derivatives, matrix routing and engineering tradeoffs; they are a deeper branch, not a condition for understanding the core lesson.

## 1. A capsule is a bundle of properties

Imagine a local image feature represented by the vector \(u=(0.3,0.4)\). Its length is \(0.5\). In a vector-capsule design, length is used as a presence **score** and the remaining variation can carry information useful for describing the feature.

This does not mean coordinate 1 must be “rotation” and coordinate 2 must be “width.” A network can learn mixed, entangled coordinates. Calling a vector a pose vector is an architectural intention; interpreting a coordinate physically requires evidence from controlled input changes or reconstruction experiments.

A usual CNN feature tensor already contains multiple channels at multiple positions. Capsules make a particular grouping and downstream computation explicit:

| Representation | What is stored at one location? | How is it combined later? |
| --- | --- | --- |
| Ordinary feature map | Several scalar channel values | Fixed learned convolutions or other mixing |
| Vector capsule | A group of coordinates, such as an 8-vector | Transform into candidate whole-vectors, then combine by routing |
| Matrix capsule | A pose matrix plus a separate activation scalar | Transform into candidate matrices, then estimate agreement and activation |

The distinction is not “CNNs contain no geometry.” Convolutions retain a spatial grid; channels can encode positional or orientation-sensitive information. Pooling can discard some exact detail, but its effect depends on the operation, boundaries and task. Capsule systems commonly begin with ordinary convolutions.

A useful mental picture is an **arrow**, not a glowing neuron: arrow direction carries a multidimensional state and arrow length carries a bounded score. The zero arrow has no direction. Ten class-capsule lengths need not sum to one, so they are not a softmax distribution or automatically calibrated probabilities.

**Visual — one grid cell, several arrows.** Expand four adjacent channels into one capsule, keeping its grid location visible. Flattening the grid later changes indexing, not the numerical contents. Trace one capsule from its row, column and type to its flattened index.

## 2. From a part to a prediction about a whole

A front wheel and a back wheel should make different predictions about the bicycle's center. The relation between each part and the whole matters.

For child capsule \(i\) and candidate parent \(j\), learn a transformation matrix \(W_{ij}\). The **vote**

\[
\widehat u_{j|i}=W_{ij}u_i
\]

is child \(i\)'s prediction of parent \(j\)'s representation. If \(u_i\) has 4 coordinates and the parent has 8, \(W_{ij}\) has shape \(8\times4\). A child has a different vote for each parent. Comparing its untransformed vector directly with every parent would skip the learned relationship.

Suppose three children send these two-dimensional votes:

| Child | Vote for parent A | Vote for parent B |
| --- | --- | --- |
| 1 | \((2,0)\) | \((0,1)\) |
| 2 | \((2,0)\) | \((0,-1)\) |
| 3 | \((0,1)\) | \((0,2)\) |

This is a constructed arithmetic example, not measured image features. Children 1 and 2 reinforce each other for A but oppose each other for B. Child 3 could support B.

We need a way to combine the votes without fixing every connection strength for every image. Define \(c_{ij}\) as child \(i\)'s fraction assigned to parent \(j\). Initially, with two parents, each row is \((0.5,0.5)\).

The tentative parent inputs are **weighted sums**:

\[
s_j=\sum_i c_{ij}\widehat u_{j|i}.
\]

Initially \(s_A=(2,0.5)\) and \(s_B=(0,1)\). The weights sum to one **across parents for each child**. They generally do not sum to one across children for a parent. Therefore this operation is not a weighted average of the incoming votes.

That distinction matters: duplicating two agreeing children can strengthen the parent input. A routing diagram should show both the incoming arrows and their scalar contribution weights, not only a heatmap.

### Keeping the output length below one

Use the squash function

\[
v=\operatorname{squash}(s)=\frac{r}{1+r^2}s,\qquad r=\|s\|.
\]

Its output length is

\[
\|v\|=\frac{r^2}{1+r^2}.
\]

Small inputs become very short; large inputs approach length one. Direction is preserved whenever \(s\ne0\), and \(\operatorname{squash}(0)=0\). This form avoids an explicit division by the norm at zero.

For parent B, \(r=1\), so \(v_B=(0,0.5)\). For A, \(r=\sqrt{4.25}\), giving \(v_A\approx(0.78535,0.19634)\), length \(0.80952\). A has the stronger initial score.

Do not read the output as “an 80.95% probability of a bicycle.” A bounded range and a probabilistic interpretation are different requirements.

## 3. Routing is a short inference computation

**Routing by agreement** repeatedly revises the connection fractions within one forward pass. This is different from updating model parameters across training examples.

Start a logit \(b_{ij}=0\) for each child–parent pair. A logit is an unconstrained score used by softmax. For each routing step:

1. Compute \(c_{ij}=\exp(b_{ij})/\sum_k\exp(b_{ik})\), normalizing over candidate parents.
2. Sum the weighted votes into \(s_j\), then squash to obtain \(v_j\).
3. If another routing step remains, update \(b_{ij}\leftarrow b_{ij}+\widehat u_{j|i}^{\mathsf T}v_j\).

Subtracting the row maximum before exponentiation gives the same softmax with better numerical stability. The agreement is a **dot product**: both direction and magnitude affect it. It is not cosine similarity unless the operands are explicitly normalized, which would define a different routing rule.

For our first step, child 1's agreements are approximately \(1.5707\) with A and \(0.5\) with B. Child 2 agrees by \(1.5707\) with A and \(-0.5\) with B. Child 3 agrees by \(0.1963\) with A and \(1.0\) with B. Those differences change the next softmax rows.

| Step | Child 1 → A | Child 2 → A | Child 3 → A | A length | B length |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | .5000 | .5000 | .5000 | .8095 | .5000 |
| 2 | .7447 | .8880 | .3092 | .9150 | .6993 |
| 3 | .8996 | .9900 | .1076 | .9346 | .7786 |

Each B fraction is one minus the corresponding A fraction. Both parents can acquire substantial scores because different children support them.

**Investigation — change a vote, then inspect the route.** Edit child 2's A vote from \((2,0)\) to \((-2,0)\); show the result, inspect which parent will have the longer vector after three steps.

<details><summary>Worked changed-vote calculation</summary>

The computed lengths become approximately .6491 for A and .8585 for B. Explain the cancellation, then explain why child 3's reassignment also matters. Try a different edited vote without being given its answer in advance.

</details>

Two useful null cases guard against overinterpreting the animation:

- If every vote is zero, every output stays zero and every coupling stays uniform.
- If every child sends exactly the same vote to both parents, symmetry keeps the two parent outputs and each row's fractions equal. Routing cannot invent evidence to break this symmetry.

Repeated agreement often makes rows increasingly concentrated. It need not change the predicted class, converge to the correct grouping or improve generalization. “Run until certain” is not a valid stopping rule. The number of iterations is part of the model configuration.

### Where does learning happen?

The convolution weights, \(W_{ij}\) and decoder weights are persistent learned parameters. Votes, logits, couplings and parent vectors are input-dependent intermediate values. Our logits restart at zero for each new image and every forward pass; they are not carried from the preceding image.

A finite sequence of matrix products, softmax, sums and squash operations can be differentiated. Backpropagation can flow through all routing steps. Some implementations detach intermediate routing computations; that preserves the forward numbers for a fixed input but changes the gradient used to learn the parameters. It is a deliberate algorithmic choice, not a requirement imposed by routing.

The distinction will return in [RNNs, LSTMs and GRUs](/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals): repetition inside routing refines assignments for one image, whereas recurrence along a sequence updates state as new observations arrive.

## 4. Build the classifier and its objective

Our small model receives one real \(8\times8\) grayscale digit. Here is the complete shape path; \(B\) means batch size.

| Stage | Output shape | Meaning |
| --- | --- | --- |
| Image | \(B\times1\times8\times8\) | Pixel intensities divided by 16 |
| Conv \(3\times3\), padding 1; ReLU | \(B\times32\times8\times8\) | Local scalar features |
| Conv \(3\times3\), stride 2, padding 1 | \(B\times16\times4\times4\) | Four capsule types, four coordinates each |
| Regroup and squash | \(B\times64\times4\) | \(4\cdot4\) locations × 4 types |
| Learned votes | \(B\times64\times10\times8\) | Every child predicts all ten digit classes |
| Routing and squash | \(B\times10\times8\) | One vector per class |
| Vector lengths and argmax | \(B\times10\), then \(B\) | Scores, then predicted digit |

Regrouping must preserve the coordinate group. In the supplied program, the primary channels are ordered by capsule type, then coordinate. We reshape to \(B\times4_{\text{type}}\times4_{\text{coord}}\times4_H\times4_W\), permute to location–type–coordinate order, then flatten the child index. Flattening the original tensor indiscriminately could group coordinates from different locations.

For a true class \(k\), the margin objective encourages its length to reach at least .9 and other lengths to stay at most .1:

\[
L_{\text{margin}}
=\sum_j\left[T_j\max(0,.9-\|v_j\|)^2
+.5(1-T_j)\max(0,\|v_j\|-.1)^2\right],
\]

where \(T_j=1\) for the actual class and 0 otherwise. Average this sum over images. If the true-class length is .7 and one wrong-class length is .3, with all other wrong lengths at most .1, the loss is \(.2^2+.5(.2^2)=.06\).

The loss has a zero-penalty region, rather than continually pushing the true length to one. The factor .5 changes the cost of wrong-class activation. This is not cross-entropy, so do not feed the lengths into a cross-entropy API as though they were unconstrained logits.

### Reconstruction asks the vector to retain useful detail

Mask all class vectors except one, flatten the ten 8-vectors into 80 values, and decode through \(80\to64\to128\to64\), using ReLU between layers and sigmoid on the final pixels. During training, select the **true** class vector for the auxiliary reconstruction objective:

\[
L=L_{\text{margin}}+.0005\sum_{p=1}^{64}(\widehat x_p-x_p)^2.
\]

The squared errors are summed over pixels and averaged over images. On 64 pixels, this is equivalent to adding \(.032\) times pixel-mean MSE. Accidentally using mean MSE with coefficient .0005 makes this term 64 times smaller.

The decoder is encouraged to retain image detail, but a reconstruction objective does not identify which latent axis must represent which physical factor. At classification time no label is needed: predict with the longest vector. For ordinary reconstruction at inference, mask using that predicted class. A reconstruction conditioned on the known true class is a separate diagnostic with extra information; we report it separately.

This design has one class capsule for each digit. It cannot separately represent two different instances of the same digit in those ten slots. Representing an image containing two 3s would require additional instance capacity and an appropriate objective; selecting two different class slots does not solve that case.

## 5. Run a controlled experiment on real digits

The offline [400-image CSV](digits-400.csv) contains optical handwritten digits from the [UCI dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), attributed to E. Alpaydin and C. Kaynak and distributed under CC BY 4.0. Each row has 64 integer intensities in 0–16 and a digit label. This is not MNIST.

We take 40 images per class and make a fixed stratified subdivision: 280 training images and 120 development images, using seed 22. The program checks distinct source IDs and complete pixel vectors before splitting. Writer identifiers are absent, so this does not establish performance on independent writers. The development set is used for the comparisons shown here; there is no untouched final test or deployment claim.

Our baseline is the **same capsule model with one routing step**. Since every coupling is initially \(1/10\), it is a uniform-routing classifier. Compare it with a separately trained three-step model, keeping architecture, initial common weights, minibatch draws, optimizer and update count paired. This isolates an actionable routing choice more directly than comparing unrelated large CNN and capsule systems.

Save [capsule-learning.py](capsule-learning.py) beside the CSV. The complete program includes model definitions, input validation, split, training, evaluation and JSON export. It does not download pretrained weights.

```bash
python -m pip install numpy==2.3.5 scikit-learn==1.9.1 torch==2.14.0
python capsule-learning.py
```

The author run used Python 3.12.14 and PyTorch 2.14.0+cpu, one CPU thread. Choose the CPU package source appropriate to your platform if the general package command offers a different accelerator build. Numerical results can vary with platform or future dependency changes; the saved arrays give an exact reference for this run.

There are six fits: seeds 1, 2 and 3, each with one or three routing steps. Every fit performs 600 Adam updates, learning rate .003, batch size 64 sampled with replacement. The vote matrices start from a normal distribution with standard deviation .1; other layers use PyTorch's defaults. No augmentation, early stopping or checkpoint selection is used. Both variants contain 47,184 parameters, including the decoder.

The central routing function is short enough to inspect. Votes have shape \(B,I,J,D\); the parent axis is 2:

```python
def squash(vectors):
    radius = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return vectors * radius / (1 + radius.square())

def route(votes, iterations=3):
    logits = votes.new_zeros(votes.shape[:-1])
    for step in range(iterations):
        coupling = logits.softmax(dim=2)
        sums = (coupling[..., None] * votes).sum(dim=1)
        output = squash(sums)
        if step < iterations - 1:
            logits = logits + (votes * output[:, None]).sum(dim=-1)
    return output
```

The downloaded program adds optional trace capture and an explicitly selected stop-gradient demonstration; training uses the full derivative. The displayed function is the same default computation, with those teaching options removed for readability.

### What actually happened?

All six models classified all 280 training images correctly at the final update. Development performance was:

| Seed | Routing during training and evaluation | Correct / 120 | Margin loss | Reconstruction MSE using predicted class |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 117 | .04680 | .03126 |
| 1 | 3 | 117 | .02364 | .03334 |
| 2 | 1 | 117 | .04823 | .03103 |
| 2 | 3 | 116 | .03311 | .03401 |
| 3 | 1 | 117 | .04608 | .03325 |
| 3 | 3 | 118 | .02497 | .03552 |

The three-step model's margin loss is lower in every paired run, but classification ties, loses one example or gains one example. Stronger margins do not necessarily change the largest score. The reconstruction term is not a proxy for classification quality either.

A training-mean-image baseline has development pixel MSE .07296. Both decoders improve on that crude reconstruction baseline. Their predicted-mask reconstructions differ from their true-label-conditioned reconstructions: for seed 1, the corresponding MSEs are .03126 versus .03041 for one-step routing and .03334 versus .03293 for three steps. The lower diagnostic error uses information unavailable at ordinary inference.

**Visual — a paired evidence panel.** Show all six runs as paired points for classification count, margin loss and reconstruction MSE, with separate labeled axes. A line joins only the two configurations sharing a seed. No smooth “training accuracy” curve should be invented between the five recorded checkpoints.

### A different question: change routing after fitting

Hold each trained model fixed and evaluate it with different routing counts:

| Seed | Steps used to train | Evaluate with 1 | With 2 | With 3 | With 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 117 | 117 | 117 | 116 |
| 1 | 3 | 118 | 118 | 117 | 117 |
| 2 | 1 | 117 | 117 | 117 | 116 |
| 2 | 3 | 118 | 117 | 116 | 116 |
| 3 | 1 | 117 | 117 | 117 | 118 |
| 3 | 3 | 117 | 118 | 118 | 118 |

All entries are correct counts out of the same 120 development images. Equal counts need not mean identical predictions: the seed-3 one-step model changes one prediction at inference step 2, although its correct count stays 117.

This intervention tests sensitivity of **fixed weights**. It is different from training the model under a new routing configuration. Choosing a preferred inference count after seeing this table consumes the development comparison; it is not an unbiased final evaluation of a new setting.

The result supports a bounded conclusion: three-step routing was not consistently better for this small experiment. It does not establish that routing never helps. A published controlled investigation similarly emphasizes testing routing against uniform alternatives, but uses different architectures, datasets and procedures. [Paik, Kwak and Kim, ACML 2019](https://proceedings.mlr.press/v101/paik19a.html)

## 6. Geometry, reconstruction and a useful failure

A system can recognize an object despite movement without retaining a predictable geometric representation. Conversely, a representation can move predictably while the final classifier still makes errors.

- **Invariance:** \(f(Tx)=f(x)\). The chosen output does not change under transformation \(T\).
- **Equivariance:** \(f(Tx)=\rho(T)f(x)\). The output changes according to a specified corresponding transformation \(\rho(T)\).

For an image classification label, invariance may be desirable for a small translation that preserves the digit. For a position estimate, translation equivariance is usually necessary: the estimated location should move. A rotation can change the semantic label in some tasks, so desired invariances must come from the task contract.

A learned capsule vector does not by itself define \(\rho(T)\). Without that definition and a check of the equality, a higher transformed-image accuracy is evidence about robustness, not a proof of equivariance.

### A shift test with no retraining

Shift every development image one pixel right or down, filling the exposed boundary with zero and discarding pixels that leave the frame. Evaluate with the routing count used for training.

| Seed | Trained steps | Unchanged | One pixel right | One pixel down |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 117 | 69 | 75 |
| 1 | 3 | 117 | 61 | 72 |
| 2 | 1 | 117 | 66 | 72 |
| 2 | 3 | 116 | 60 | 77 |
| 3 | 1 | 117 | 72 | 76 |
| 3 | 3 | 118 | 65 | 78 |

The model is highly sensitive to these shifts. The experiment uses small images, stride 2, location-specific vote matrices and no shift augmentation. Cropping can also remove meaningful strokes; inspect individual transformed inputs rather than assuming every transformation perfectly preserves the label. Zero shift reproduces the unchanged predictions.

This is a useful failure. Routing's internal agreement does not manufacture the missing coverage of transformed data or constrain every preceding layer to respect a symmetry.

**Investigation — edit pixels and inspect the evidence.** Start from a saved real image with its class scores. Change a selected pixel or apply a bounded nonwrapping shift. Observe whether the winning class will change, Show the current computed result and its contributing terms immediately. A changed vector with the same winning class is a meaningful result. Keep the original label as a reference annotation, not an input to the classifier.

### What can a latent-coordinate experiment tell us?

Choose one class vector, keep the decoder and mask fixed, and change one coordinate by \(-.1,0,+.1\). Display the three reconstructions at the same intensity scale. This probes how the **learned decoder** responds to that coordinate near that specimen.

It may alter several image properties at once. Even if a change resembles thickness in one image, call it a local observed effect until it recurs under broader controlled tests. A coordinate edit is not guaranteed to lie on the distribution of vectors the encoder actually produces.

There is also a precise null case: if the decoder mask selects class 4, changing only class 5's vector cannot affect the masked decoder input. Changing a caption's label without changing the actual mask cannot affect any arithmetic.

### An unusual application: separate overlapping objects

Why reconstruct one class at a time? In an image containing two different digits, two class capsules can condition two reconstructions. The auxiliary task asks each selected representation to account for a different component rather than the combined image.

This requires appropriate paired component targets and a multi-object objective; the single-digit model above was not trained for it. A careful experiment must build training composites from training specimens and held-out composites from held-out specimens. Millions of pairings of a smaller source set do not become millions of independent original specimens. The original capsule work explored this direction on overlapping digits. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

The same part-to-whole question can arise in video: frame-level or local motion evidence may support an action occurring over a region and interval. The representation, instance capacity and evaluation unit must then include time. This is a reason to investigate capsules, not evidence that this digit model already solves action localization. The [UCF CVPR tutorial](https://www.crcv.ucf.edu/cvpr2019-tutorial/) includes a separate video-capsule session for that application.

## 7. Deeper mechanics: gradients and explicit coordinate frames

### Squash changes radial and sideways sensitivity differently

Write \(s=rq\), where \(q\) is a unit vector. A small change parallel to \(q\) changes length; a perpendicular change initially changes direction. The squash Jacobian has eigenvalues

\[
\lambda_{\text{radial}}=\frac{2r}{(1+r^2)^2},\qquad
\lambda_{\text{tangent}}=\frac{r}{1+r^2}.
\]

Both approach zero at the origin and at very large radius, at different rates. Consequently, making initial votes arbitrarily tiny or letting summed inputs become huge can reduce useful gradients. The radial output-length curve \(r^2/(1+r^2)\) has its inflection at \(r=1/\sqrt3\), not at 1.

For \(s=(.3,.4)\), \(r=.5\), radial sensitivity is .64 and tangent sensitivity .4. For \(s=(3,4)\), \(r=5\), they are approximately .01479 and .19231. A very long vector is much harder to lengthen than to rotate locally.

The [mechanics program](capsule-mechanics.py) computes the analytical Jacobian and checks it by central differences. At zero the exact derivative is zero; a finite difference has a small step-dependent residual. This is a numerical approximation, not a contradictory derivative.

For the three-step routing fixture, differentiating the full computation agrees with central differences to about \(1.2\times10^{-10}\). Detaching earlier routing calculations gives the same scalar loss but a gradient differing by up to .03752. Both can be coded; only the full derivative matches the stated full forward function's derivative.

### Why explicit matrices can help—and what they do not guarantee

Suppose a part really has a homogeneous 2D coordinate frame

\[
M_i=\begin{bmatrix}1&0&2\\0&1&3\\0&0&1\end{bmatrix}
\]

and its relation to the whole is

\[
W_{ij}=\begin{bmatrix}1&0&-1\\0&1&0\\0&0&1\end{bmatrix}.
\]

Then \(M_iW_{ij}\) predicts a whole located at \((1,3)\). Rotate the entire scene by \(90^\circ\), using

\[
G=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}.
\]

Associativity gives \((GM_i)W_{ij}=G(M_iW_{ij})\), so the predicted whole moves to \((-3,1)\) consistently. The part–whole relation stays fixed while the viewing frame changes.

This is an exact calculation **because** these matrices have an explicitly supplied geometric meaning and transform by left multiplication. A learned encoder must still produce frames with the promised transformation behavior. A generic learned \(4\times4\) array does not automatically become a rigid camera pose.

For vector votes, a linear map needs

\[
W\rho_{\text{in}}(T)=\rho_{\text{out}}(T)W
\]

to preserve a chosen symmetry. Arbitrary \(W\) need not satisfy it. For \(W=\operatorname{diag}(2,1)\), \(u=(1,2)\) and a \(90^\circ\) rotation \(R\), \(WRu=(-4,1)\) while \(RWu=(-2,2)\).

Squash itself commutes with an orthogonal rotation because rotation preserves the norm. It does not commute with arbitrary scaling: \(\operatorname{squash}(2u)\ne2\operatorname{squash}(u)\). Exact symmetry is a whole-computation constraint, not a descriptive name. [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576) develops architectures that explicitly constrain transformations.

### Matrix capsules and EM routing

Matrix capsules separate an activation scalar \(a_i\) from pose matrix \(M_i\). Votes are \(V_{ij}=M_iW_{ij}\). Flatten the entries of a vote into coordinates \(h\) only for the statistics.

Instead of repeatedly adding dot-product agreement, a diagonal-Gaussian routing procedure estimates a mean and variance for the votes supporting each parent. Let \(R_{ij}\) be a child's normalized responsibility and \(q_{ij}=a_iR_{ij}\). Then

\[
n_j=\sum_iq_{ij},\quad
\mu_{jh}=\frac{\sum_iq_{ij}V_{ijh}}{n_j},\quad
\sigma^2_{jh}=\frac{\sum_iq_{ij}(V_{ijh}-\mu_{jh})^2}{n_j}.
\]

The child activation scales its contribution. Each iteration recomputes \(q=aR\); repeatedly multiplying a previously scaled \(q\) by \(a\) would incorrectly suppress children again and again. A parent with zero effective mass has no identified mean. A numerical denominator guard prevents a crash but does not create evidence; report the no-evidence case. A variance floor similarly prevents singular densities while changing the fitted spread.

Estimate a parent activation from a coding-cost expression, then revise responsibilities using a Gaussian log density plus log parent activation, normalized over parents. Our complete constructed demonstration uses

\[
\mathrm{cost}_j=\sum_h n_j(\beta_u+\log\sigma_{jh}),\quad
a_j=\operatorname{sigmoid}\{\lambda(\beta_a-\mathrm{cost}_j)\},
\]

\[
R_{ij}=\operatorname{softmax}_j\left[
\log a_j-\frac12\sum_h\left(\log(2\pi\sigma^2_{jh})
+\frac{(V_{ijh}-\mu_{jh})^2}{\sigma^2_{jh}}\right)\right].
\]

It fixes \(\beta_u=\beta_a=0\), variance floor .01 and inverse temperatures .5, .75 and 1. These are declared illustration settings; a trained matrix-capsule model learns cost parameters and uses a chosen schedule.

For three children with activations \(1,1,.5\), parent-A first coordinates \(0,.2,2\) and parent-B first coordinates \(0,3,3.2\), uniform responsibilities give mass 1.25 per parent. The first means are .48 and 1.84. After three rounds they are approximately .10881 and 2.81927. The second coordinate is zero for every vote, so its variance hits the stated floor. Making the third child inactive makes edits to its votes irrelevant to the estimated means.

The resemblance to [Gaussian-mixture EM](/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml) is useful, but each capsule parent sees a differently transformed version of the children, and parent activations do not sum to one. It is not ordinary maximum-likelihood fitting of one common observed dataset. The matrix-capsule paper discusses the change-of-variables issue when comparing densities in different transformed spaces. [Matrix Capsules with EM Routing](https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf)

## Follow routing all the way into a trainable program

The runnable scratch route is split by purpose, not by missing work. [capsule-mechanics.py](capsule-mechanics.py) owns NumPy votes, stable softmax, squash, routing iterations and the bounded diagonal-EM illustration. [capsule-learning.py](capsule-learning.py) owns the differentiable Torch routing, `TinyCapsules`, margin loss, reconstruction and complete fit. The Torch tensor implementation is the ordinary research route: there is no universal standard capsule layer whose import can replace specifying the routing algorithm. Reusing `einsum`, linear layers and autograd does not hide routing; the code explicitly updates its coupling logits and sums weighted votes.

The paired route compares the same votes and iteration count, not two independently fitted classifiers. `author-checks.py` reconstructs saved encoder/routing/reconstruction outputs through a separate NumPy path. The lesson's gradient branch explains how gradients flow through the iterative computation. Detaching intermediate agreements would change that training algorithm even if its forward result stayed identical. Softmax runs over candidate **parents for each child**; moving that axis changes who competes for responsibility.

Dynamic routing with B examples, I child capsules, J parents, vote width D and R rounds uses O(BIJD·R) routing arithmetic and O(BIJD) vote storage, apart from the learned vote transforms. Parent-batched contractions are appropriate for the small inspected classifier; manufacturing an extra all-pairs child tensor is not. Diagonal EM has a different Gaussian/statistical meaning and stays explicitly a small illustrative calculation, not an implementation claim for the complete Matrix Capsules paper or its convolutional pose system.

**Implementation exercise:** add a positive routing temperature τ by replacing `softmax(logits)` with `softmax(logits/τ)`, leaving the agreement update unscaled. Compare τ0.5,1 and2 with fixed votes and R. Do not divide both the logits and every agreement update unless you mean a different algorithm. The worked two-parent logits[0,2] give shares approximately[0.1192,0.8808] atτ1, [0.0180,0.9820] atτ0.5, and[0.2689,0.7311] atτ2. Check each child row sums to one and that a single candidate parent always receives share one. Trace actual vector outputs as well as shares; sharper assignments do not guarantee better classification. At finite nonzero votes this remains differentiable, while τ must remain strictly positive.

## 8. Architecture costs and alternative routing designs

The classic vector CapsNet uses a larger shape chain than our experiment:

\[
28^2\to256\times20\times20
\to32\text{ types}\times6\times6\times8\text{ coordinates}
\to10\times16.
\]

Both convolutions have \(9\times9\) kernels; the second has stride 2. There are 1,152 primary capsules. With a masked 160-value decoder input and decoder widths 512, 1,024 and 784, the parameter accounting is:

| Component | Parameters |
| --- | ---: |
| First convolution | 20,992 |
| Primary-capsule convolution | 5,308,672 |
| Vote transformations | 1,474,560 |
| Reconstruction decoder | 1,411,344 |
| Total | 8,215,568 |

The primary convolution, not the vote matrices, owns the largest share here. Disabling the decoder removes its parameters. A decoder that receives only the selected 16-vector is another design with different counts; do not mix that count with the masked-160 implementation.

For historical context, the vector-capsule paper reports .25% ordinary MNIST test error for its three-routing-step reconstruction model. Its 99.23% MNIST accuracy belongs to a different, expanded-canvas model used in the affNIST transfer comparison. Those are not interchangeable results. Matrix-capsule experiments also use smallNORB: photographs of physical toy objects under controlled views and lighting, not rendered 3D objects. Its separation of physical training and test instances is material to interpreting generalization. [Vector-capsule experiments](https://arxiv.org/pdf/1710.09829), [matrix-capsule experiments](https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf).

For \(I\) children, \(J\) parents, child dimension \(d\), parent dimension \(D\) and \(r\) routing steps:

- Dense vote parameters and vote products scale with \(IJdD\).
- Stored votes scale with \(BIJD\), where \(B\) is batch size.
- Weighted sums at all \(r\) steps plus agreements at the first \(r-1\) steps require \((2r-1)BIJD\) scalar product-and-accumulate terms, excluding softmax, squash and other operations.

At batch 32, the classic vote tensor alone occupies 23,592,960 bytes in float32, about 22.5 MiB. Backpropagation retains additional state. Arithmetic counts do not predict latency without measuring memory movement, kernels, hardware and backward computation.

Naively expanding the original valid-convolution topology to \(224\times224\) gives a \(104\times104\) primary grid, or 346,112 children. Fully connecting these to 1,000 parents with dimensions 8→16 would require 44,302,336,000 vote parameters. This is a warning about that particular expansion, not a lower bound for every capsule architecture. Local capsule receptive fields and shared type-to-type transforms change the scaling.

Matrix multiplication of a \(4\times4\) pose by a learned \(4\times4\) relation uses 16 learned parameters per relation. An unrestricted linear map of a flattened 16-vector to another 16-vector uses 256. That parameter reduction imposes structure; it is not a free replacement for every arbitrary vector map.

Three alternative directions answer different shortcomings:

| Direction | Mechanism | What to examine |
| --- | --- | --- |
| Diagonal EM routing | Estimate vote clusters, spread and activation separately | Variance floors, negligible mass, log-domain numerics, local sharing |
| Variational-Bayes routing | Maintain approximate uncertainty over mixture parameters and assignments with priors | Prior strength, approximation assumptions, whether variance-collapse behavior improves |
| STAR-Caps | Use learned attentive coefficients and binary routing gates with a straight-through gradient estimator | Discrete forward choices versus surrogate gradients, actual sparse execution and measured cost |

The [AAAI 2020 variational-routing paper](https://ojs.aaai.org/index.php/AAAI/article/view/5785) and [NeurIPS 2019 STAR-Caps paper](https://karim-ahmed.github.io/publications/starcaps.pdf) are distinct algorithms, not extra loop counts for the vector-routing function above. STAR-Caps includes ImageNet experiments, so “capsules have never been tried on ImageNet” is incorrect. Historical results should be read with their architecture, data and training conditions, not used as an undated ranking.

For matrix capsules, **spread loss** is another objective:
\[
L=\sum_{i\ne t}\max(0,m-(a_t-a_i))^2.
\]
It asks the true activation to exceed each wrong activation by a margin \(m\), often increased during training. Unlike the earlier independent thresholds .9 and .1, it penalizes a relative activation gap.

Applications involving geometric structure or overlapping instances can justify capsule experiments. They still need matched baselines, valid splits and a specific failure hypothesis. Neither an attractive reconstruction nor resistance to one attack proves general robustness. A 3D viewpoint change can reveal or hide surfaces; it is not always an invertible 2D image transform.

## 9. Practice: implement, explain and compare

### 1. Repeated evidence is not an average

There are two identical children. Each sends \((1,0)\) to both of two parents. What is each parent's length after one step? Add two more identical children. What changes, and will further routing break the symmetry?

<details><summary>Hint</summary>

Each child splits its own contribution in half. Sum contributions at a parent before applying squash.

</details>
<details><summary>Solution</summary>

Two children give \(s=(1,0)\), so the length is .5. Four give \(s=(2,0)\), so the length is .8. Both parents are identical at every step and each row remains \((.5,.5)\). More evidence changes magnitude; it does not create a reason to prefer either parent.

</details>

### 2. A positive agreement can still lose share

A child currently has logits \((0,0)\). The next agreements are \((1,2)\). Did the first parent gain or lose coupling, even though its agreement was positive?

<details><summary>Hint</summary>

Softmax compares scores within the same row. Compute the first fraction from the updated logits.

</details>
<details><summary>Solution</summary>

Its coupling falls from .5 to \(e^1/(e^1+e^2)=1/(1+e)\approx.26894\). A positive absolute update is not necessarily a relative gain.

</details>

### 3. Change the loss convention correctly

A \(16\times16\) reconstruction uses .0005 times summed squared error. Your API returns mean squared error over pixels. What coefficient preserves the objective? If the true class has length .8 and two wrong classes have lengths .2 and .4, compute the margin loss.

<details><summary>Hint</summary>

There are 256 pixels. The true-class shortfall and the two wrong-class excesses use different weights.

</details>
<details><summary>Solution</summary>

Use \(.0005\cdot256=.128\) times pixel MSE. The margin terms are \((.9-.8)^2+.5(.2-.1)^2+.5(.4-.1)^2=.01+.005+.045=.06\), assuming all other wrong lengths are at most .1.

</details>

### 4. Repair a leaking reconstruction report

A program evaluates classification without labels, but reconstructs every development image using its known true class and labels the result “inference reconstruction.” Rewrite the protocol and specify which numbers to retain.

<details><summary>Hint</summary>

Separate the decision available to the deployed model from an optional diagnostic that conditions on the answer.

</details>
<details><summary>Solution</summary>

Select the longest class capsule to compute ordinary inference reconstruction. Retain the true-label-masked reconstruction under an explicit “label-conditioned diagnostic” label. Report both MSEs and classification errors if the distinction is useful. Neither reconstruction should change classification scores. Do not supply the true label to select a capsule in a claimed label-free system.

</details>

### 5. Design a new routing comparison

You want to know whether routing helps with left-shifted digits. The existing table contains only right and downward shifts. Specify a comparison that does not treat those table rows as a new untouched test, then make and check a prediction with the program.

<details><summary>Hint</summary>

State which weights are fixed, how pixels leaving the image are handled, which labels remain valid, and what data have already influenced your choices.

</details>
<details><summary>Solution</summary>

One valid development investigation holds each of the six models fixed, applies a one-pixel left shift with zero fill, inspects label-preservation failures and compares paired one-step/three-step training configurations. Show the current computed result and its contributing terms immediately. The result is exploratory because the dataset and models have already been studied. A subsequent final claim needs a separately reserved, relevant evaluation set and a frozen protocol. The numeric left-shift result is intentionally not supplied: generate it, retain the changed inputs and explain both counts and disagreement cases.

</details>

### 6. Disprove an equivariance claim

An engineer says any learned linear vote map preserves rotation because it is a matrix. Use \(u=(1,0)\), \(W=\operatorname{diag}(3,1)\) and a \(90^\circ\) rotation to test the claim. What must replace that assertion?

<details><summary>Hint</summary>

Compare transforming before the vote map with transforming afterward.

</details>
<details><summary>Solution</summary>

\(WRu=W(0,1)=(0,1)\), while \(RWu=R(3,0)=(0,3)\). The diagram does not commute. Specify input/output group actions and constrain \(W\rho_{\text{in}}=\rho_{\text{out}}W\); also verify the encoder, nonlinearities, routing and readout preserve the intended transformation contract.

</details>

### 7. A low-activation child and diagonal EM

For one parent, two scalar votes are 0 and 4. Responsibilities are both .5; child activations are 1 and .25. Calculate effective mass, mean and variance before a variance floor. Why is repeatedly multiplying the responsibilities by activation in place wrong?

<details><summary>Hint</summary>

Use effective weights .5 and .125. Variance measures squared distance from the weighted mean.

</details>
<details><summary>Solution</summary>

Mass is .625, mean is \(.125\cdot4/.625=.8\), and variance is \([.5(.8)^2+.125(3.2)^2]/.625=2.56\). Repeated in-place multiplication would turn the second activation factor into .25², .25³ and so on, changing the specified model. Each iteration uses the new normalized \(R\) and multiplies by the unchanged \(a\) once.

</details>

### 8. Explain a reconstruction edit without inventing semantics

Use either saved image and the fixed seed-1 three-step model. Change a selected class-vector coordinate by a value other than the demonstrated ±.1. Predict what changes and what must remain invariant. What evidence would justify naming the coordinate “stroke thickness”?

<details><summary>Hint</summary>

Separate encoder scores, decoder inputs, mask selection and visual interpretation.

</details>
<details><summary>Solution</summary>

A decoder-only edit leaves the original encoder's scores unchanged unless you explicitly recompute a score from the edited vector. Editing a masked-out class leaves reconstruction unchanged; editing the selected class can change it. Record the exact coordinate, delta, mask and image difference. Naming a physical factor requires consistent, controlled changes across relevant images and checks for confounded properties, not one appealing morph.

</details>

**Readiness:** you can follow one image through grouping, voting, routing, scoring and reconstruction; explain why routing normalizes over parents for each child; distinguish input-dependent assignments from trained weights; and design a comparison whose conclusion matches the data. For the deeper branch, derive a squash sensitivity, check a transformation identity and trace one EM update.

## 10. Continue and learn another way

Next in the module is [RNNs, LSTMs and GRUs](/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals). We move from repeated assignment refinement for one image to a state updated over observations in time. The connection is repeated computation; the purpose and state lifetime are different.

Useful references and alternate routes:

- [Dynamic Routing Between Capsules — Sabour, Frosst and Hinton](https://arxiv.org/abs/1710.09829). Read the routing procedure with the child/parent axes beside it, then the reconstruction experiment. Its ordinary MNIST result and the separately trained affNIST-transfer model are different protocols; do not merge their scores.
- [Introduction to Capsules — Sara Sabour's slides](https://www.cs.toronto.edu/~saaraa/CapsuleSlides.pdf). A visual route through coordinate frames, agreement and assignment, especially slides 10–40. Some slides use cosine terminology; the implemented vector-routing agreement in this lesson is the dot product.
- [Capsule Networks for Computer Vision — UCF CVPR 2019 tutorial](https://www.crcv.ucf.edu/cvpr2019-tutorial/). The university index links talks and slides, including Sabour's introduction, a survey, video capsules and segmentation. It is historical research teaching, with separate prerequisites for the application sessions; the full video was not watched for this packet.
- [Matrix Capsules with EM Routing](https://www.cs.toronto.edu/~hinton/absps/EMcapsules.pdf). Follow the pose/activation distinction and algorithm, then Appendix A for why transforming Gaussian votes differ from fitting an ordinary mixture.
- [Capsule Networks Need an Improved Routing Algorithm](https://proceedings.mlr.press/v101/paik19a.html). An alternative reading centered on controlled comparisons and assignment polarization. Its experiments are evidence under those configurations, not a universal impossibility theorem.
- [Capsule Routing via Variational Bayes](https://ojs.aaai.org/index.php/AAAI/article/view/5785). A deeper probabilistic route; read after the local EM bridge and prior Gaussian-mixture material.
- [STAR-Caps](https://karim-ahmed.github.io/publications/starcaps.pdf). Study the distinction between a discrete routing decision and its straight-through training gradient before attempting to reproduce this architecture.
- [Data and calculation provenance](data-provenance.md), [complete learning program](capsule-learning.py), [constructed mechanics program](capsule-mechanics.py) and its [small import helper](capsule_learning_import.py). The first trains the offline model; the second supplies exact routing, geometry, derivative and EM fixtures. No browser lab is required to inspect the underlying arithmetic.
