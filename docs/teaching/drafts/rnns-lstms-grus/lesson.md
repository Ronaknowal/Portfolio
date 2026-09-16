# RNNs, LSTMs & GRUs

A pen stroke is more than a collection of points. The order tells you how the pen travelled between them. A recurrent neural network processes that order by repeatedly updating a small collection of numbers: its current state. An LSTM or GRU changes the update rule so that the network can selectively retain, replace and expose information.

**Your first pass:** follow the pen trajectory and the three-step calculation in sections 1–2; learn the retain/write/read roles in sections 3–4; run or inspect the complete handwriting experiment in section 5; then work through state boundaries and padding in sections 6–7. Finish the core practice. Section 8 opens the deeper derivative and architecture questions when you are ready. You do not need its full Jacobian derivation to understand the next lesson.

The previous [Capsule Networks lesson](/learn/path/full-curriculum/capsule-networks?module=deep-learning-fundamentals) organized parts inside one input. Here we follow observations across positions in a sequence. The distinction matters: routing iterations within a capsule model are not elapsed time, and the hidden state of an RNN is not a routing coefficient.

## 1. Keep a running description

Imagine identifying a handwritten digit from eight successive pen coordinates. After point 1, you know where the trace starts. After point 4, you have evidence about a bend. By point 8, the model must produce one of ten digit labels.

One approach is to concatenate all sixteen coordinates and use an ordinary classifier. That preserves their order: the first pair and last pair occupy different input columns. Another approach summarizes the points by their mean, spread and extremes; it loses the order. A recurrent model offers a third approach: apply the same update rule to each new point and a state summarizing the previous points.

This distinction is useful beyond handwriting. A recurrent state can summarize successive machine measurements, maintain context during a conversation, or track what a controller has observed. The state is learned for the prediction objective. It is not a lossless recording of every past event, and it does not automatically represent a human-readable fact.

**Visual: the same trace, three representations.** The left view numbers the eight pen points and connects consecutive points. The middle shows sixteen ordered input slots. The right shows only mean/spread/extreme statistics. Reverse the points: the geometric point set stays fixed, the arrows and ordered slots change, and the orderless statistics do not. This is a question about available information before it is a question about model accuracy.

Let \(x_t\) be the observation at position \(t\). A state \(h_t\) is a vector produced after observing that input:

\[
h_t=F_\theta(x_t,h_{t-1}).
\]

The same parameters \(\theta\) appear at every position. During one prediction, these weights stay fixed while the state changes. Training changes the weights between optimizer updates. Mixing up these two changes makes recurrent models seem more mysterious than they are.

For our pen example, each \(x_t\) has two coordinates and \(h_t\) will have 32 learned features. A batch of 64 complete traces has shape \([64,8,2]\): specimen, position, coordinate. At one position the state has shape \([64,32]\). Rows belong to different specimens and must not exchange their histories.

There are several useful output arrangements:

| Input and output | Example | Which state is read? |
| --- | --- | --- |
| Many observations → one label | A complete pen trace → digit | A summary after the valid final point |
| One prediction per observation | A sensor prefix → current operating state | Each causal state \(h_t\) |
| Previous symbols → next symbol | A token prefix → a next-token distribution | State after the preceding symbols |
| One sequence → another | A source sentence → a translation | An encoder and a separate decoder |

The first is our complete experiment. The last is the [next lesson](/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals). A recurrent layer is a reusable component, not a complete specification of what a model predicts.

### Position does not necessarily mean seconds

The real data used below contains eight points resampled at approximately equal distances along each completed pen trace. Their positions preserve order but are not eight equally spaced timestamps. An input collected at irregular times may need elapsed-time features as well as its measurements. A recurrence knows which observation came first; it cannot infer an unrecorded time gap.

## 2. Build and train the simplest recurrence

A simple tanh RNN first mixes the new input with the old state, then applies a bounded nonlinear transformation:

\[
a_t=W_xx_t+W_hh_{t-1}+b,\qquad h_t=\tanh(a_t).
\]

\(W_x\) maps input features into state features. \(W_h\) mixes the previous state features with each other. \(b\) shifts the resulting values. The tanh function maps any real number into \((-1,1)\); near zero it is close to its input, but large positive or negative inputs produce values close to the endpoints.

For \(D\) input features and \(H\) state features, \(W_x\) is \(H\times D\), \(W_h\) is \(H\times H\), and \(b\) has \(H\) entries. A classification head converts the final state into scores \(Wh_T+b_{\text{out}}\), then softmax converts scores to class probabilities. The matrices have no time index because they are shared.

The [D2L recurrent-network explanation](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) develops this distinction between a hidden layer and a hidden state. Here we can calculate the whole chain rather than relying on the diagram.

### Three observations, one state number

Use \(x=[0.4,-0.2,0.7]\), input weight \(w_x=0.8\), recurrent weight \(w_h=0.6\), bias \(b=0.1\), and \(h_0=0\). These are deliberately constructed numbers, not fitted handwriting features.

| Position | New input | Input contribution \(0.8x_t\) | Previous-state contribution \(0.6h_{t-1}\) | \(h_t\) after adding bias and tanh |
| --- | --- | --- | --- | --- |
| 1 | 0.4 | 0.32 | 0 | 0.396930 |
| 2 | −0.2 | −0.16 | 0.238158 | 0.176297 |
| 3 | 0.7 | 0.56 | 0.105778 | 0.644468 |

The second input is negative, yet the second state is positive. The state includes both the new observation and a transformed contribution from the past. It is not simply a copy of the current input.

**Investigation: edit an observation, then trace its consequences.** Change the middle input before stepping the recurrence. Predict whether the final state will rise, fall or remain equal, and enter an approximate value. The display then recomputes the input contribution, old-state contribution, preactivation and tanh at each affected position. Changing \(x_2\) cannot alter \(h_1\), because this recurrence only moves forward.

### Learning assigns credit to repeated uses of the same weight

Suppose the desired final state is \(y=0.3\) and the loss is

\[
L=\frac12(h_3-y)^2.
\]

The calculated loss is \(0.059329\). To reduce it, training needs to know how changing each weight would change the final output.

Unroll the recurrence into three copies of the calculation. There are three state values, but still one shared input weight, one shared recurrent weight and one shared bias. **Backpropagation through time (BPTT)** is ordinary chain-rule backpropagation through this unrolled graph.

Start with \(\partial L/\partial h_3=h_3-y=0.344468\). Since the derivative of tanh at \(a_t\) is \(1-h_t^2\), the credit arriving at the preactivation is

\[
\delta_t=\frac{\partial L}{\partial h_t}(1-h_t^2).
\]

Each use contributes \(\delta_tx_t\) to the input-weight gradient, \(\delta_th_{t-1}\) to the recurrent-weight gradient, and \(\delta_t\) to the bias gradient. The previous state receives \(w_h\delta_t\). Move backward and add the contributions from every use:

| Shared parameter | Contribution at step 3 | At step 2 | At step 1 | Total |
| --- | --- | --- | --- | --- |
| \(w_x\) | 0.140978 | −0.023416 | 0.023673 | 0.141234 |
| \(w_h\) | 0.035506 | 0.046474 | 0 | 0.081979 |
| \(b\) | 0.201397 | 0.117082 | 0.059181 | 0.377661 |

With learning rate \(0.1\), subtract \(0.1\) times each gradient. The updated parameters are approximately \((0.785877,0.591802,0.062234)\); recalculating the sequence gives loss \(0.042839\). The retained calculation agrees with automatic differentiation. This single successful step explains the mechanism; it does not establish that any learning rate will improve every step.

If there are losses at several positions, add each position's direct loss gradient to the gradient arriving from the future before propagating backward. The [Backpropagation & Automatic Differentiation lesson](/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals) supplies the general graph perspective.

### Why distant credit can become difficult

For a vector state, the local derivative with respect to the previous state is

\[
J_t=\operatorname{diag}(1-h_t^2)W_h.
\]

Credit from a distant position involves a product of these matrices. Repeated contraction can make an earlier input's effect tiny; repeated expansion along relevant directions can make gradients very large. The actual activations and directions matter. A large recurrent matrix alone does not prove exploding gradients.

For example, a scalar recurrent weight of 2 at preactivation 5 gives local derivative \(2(1-\tanh^2 5)\approx0.000363\), a strong contraction. The tanh saturation overwhelms the weight. Section 8 separates reliable norm bounds from misleading eigenvalue shortcuts. [Pascanu, Mikolov and Bengio](https://proceedings.mlr.press/v28/pascanu13.pdf) analyze temporal gradient products and motivate gradient-norm clipping.

Clipping limits the size of a gradient that is already available. It cannot reconstruct a signal that has vanished. Gated recurrences instead change the pathways through which state and credit travel.

## 3. LSTM: retain, write, then expose

An LSTM carries two vectors. The **cell state** \(c_t\) has an additive update path. The **hidden state** \(h_t\) is a transformed, gated view of that cell and is also used to compute the next update. Calling them “long-term” and “short-term” memory can be a starting analogy, but neither name guarantees a retention duration.

Before looking at the learned gates, consider one coordinate:

\[
c_t=f_tc_{t-1}+i_tg_t,\qquad h_t=o_t\tanh(c_t).
\]

Here \(f_t\) controls retention, \(i_t\) controls writing, \(g_t\) proposes a signed value, and \(o_t\) controls how much of the transformed cell is exposed. Each symbol is one coordinate here; a real layer performs these operations component by component.

If the old cell is \(0.8\), retention \(f=0.9\), input gate \(i=0.2\), candidate \(g=-0.5\), and output gate \(o=0.6\), then:

1. Keep \(0.9\times0.8=0.72\) from the old cell.
2. Write \(0.2\times(-0.5)=-0.10\).
3. Add them to get \(c=0.62\).
4. Expose \(h=0.6\tanh(0.62)\approx0.330677\).

An output gate near zero can hide the cell from the current readout while leaving its stored value present. An input gate near zero can avoid writing a distracting candidate. The forget gate controls the old term; it does not itself decide what new value replaces it.

**Visual: an accounting of one memory coordinate.** Show the old cell as a signed quantity, its retained portion, a separate signed write contribution, their sum, then tanh and the output gate. A generic “four boxes” diagram would hide the central distinction between multiplying a gate and adding two information paths.

### The gates are computed, not hand-selected during prediction

The standard non-peephole LSTM used here computes four affine transforms of the current input and previous hidden state:

\[
\begin{aligned}
i_t&=\sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i),\\
f_t&=\sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f),\\
g_t&=\tanh(W_{xg}x_t+W_{hg}h_{t-1}+b_g),\\
o_t&=\sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o).
\end{aligned}
\]

The sigmoid \(\sigma(a)=1/(1+e^{-a})\) produces a number in \((0,1)\), appropriate for a multiplicative gate. The candidate uses tanh because a proposed write may be positive or negative. There are three sigmoid gates and one candidate transform. Gate values vary across coordinates and positions, even though their learned matrices are shared. In vector equations, \(\odot\) means multiply corresponding coordinates, not matrix multiplication.

The [D2L LSTM walkthrough](https://d2l.ai/chapter_recurrent-modern/lstm.html) is a useful second account of these operations. Modern LSTM notation includes a forget gate; the [1997 LSTM paper](https://www.bioinf.jku.at/publications/older/2604.pdf) introduced an earlier architecture, and [Gers, Schmidhuber and Cummins](https://pubmed.ncbi.nlm.nih.gov/11032042/) introduced adaptive forgetting in 2000. Historical implementations and today's full automatic differentiation are not identical algorithms.

### “Almost one” still compounds

Hold the write contribution at zero and the forget factor constant. Starting at \(c_0=1\), the direct retained value after \(T\) steps is \(f^T\).

| Fixed forget factor | After 10 steps | After 100 steps | Steps to halve the retained value |
| --- | --- | --- | --- |
| 0.5 | 0.000977 | \(7.89\times10^{-31}\) | 1 |
| \(\sigma(1)\approx0.731059\) | 0.043604 | \(2.48\times10^{-14}\) | 2.21 |
| 0.99 | 0.904382 | 0.366032 | 68.97 |
| 0.999 | 0.990045 | 0.904792 | 692.80 |

These are exact-formula illustrations, not measured gradients of trained networks. A positive forget bias can initially favor retention, but bias 1 does not by itself preserve a signal for hundreds of steps. To retain half over 100 fixed-factor steps requires \(f=0.5^{1/100}\approx0.993092\), corresponding to sigmoid preactivation about 4.968. Real gates also depend on inputs and state.

**Investigation: design a memory interval.** Choose the number of steps and the fraction you want retained. Predict an appropriate forget factor, then see the calculated curve and half-life. Next allow a nonzero write at an editable position and observe why the total cell value is no longer simply \(f^T\).

This controllable additive route helps with learning long dependencies. It is not a promise that every LSTM gradient stays constant: gates can close, output tanh can saturate, and the complete state has additional derivative paths.

## 4. GRU: blend old state with a new proposal

A GRU carries one state vector. Define its update gate \(z_t\) as the fraction of old state retained:

\[
h_t=z_t\odot h_{t-1}+(1-z_t)\odot n_t.
\]

If an old coordinate is \(0.8\), its candidate is \(-0.4\), and \(z=0.75\), the new state is \(0.75(0.8)+0.25(-0.4)=0.5\). A large \(z\) means a small replacement in this convention. Some presentations use the complementary convention, so read the equation before interpreting the word “update.”

The reset gate \(r_t\) controls how the old state influences the candidate. The update gate decides how much candidate actually replaces the old state. Resetting the candidate's dependence on history is therefore not the same as clearing the entire carried state.

For the PyTorch variant used in the program:

\[
\begin{aligned}
r_t&=\sigma(W_{xr}x_t+b_{xr}+W_{hr}h_{t-1}+b_{hr}),\\
z_t&=\sigma(W_{xz}x_t+b_{xz}+W_{hz}h_{t-1}+b_{hz}),\\
n_t&=\tanh(W_{xn}x_t+b_{xn}+r_t\odot(W_{hn}h_{t-1}+b_{hn})).
\end{aligned}
\]

The [official GRU documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) specifies this reset-after-recurrent-affine form. The [original encoder–decoder paper](https://arxiv.org/abs/1406.1078) uses a reset-before-matrix form. They are not interchangeable when copying weights between implementations.

To see the difference without an entire network, let
\(h=[1,2]^T\), \(r=[0.2,0.8]^T\),
\(W=\begin{bmatrix}1&2\\3&4\end{bmatrix}\), and recurrent bias \(b=[0.5,-0.5]^T\).
Then
\[
W(r\odot h)+b=[3.9,6.5]^T,\qquad
r\odot(Wh+b)=[1.1,8.4]^T.
\]
Multiplying individual coordinates before mixing them is a different operation from scaling the mixed outputs. The recurrent candidate bias is also inside the reset multiplication in the PyTorch form.

**Visual: two reset placements.** Use the same two input coordinates and matrix edges on both sides. Highlight the exact edges scaled before mixing versus the two outputs scaled afterward. A learner should be able to explain why the numbers differ, not merely notice that two frameworks have different labels.

Matched at the same hidden width, a GRU has three affine groups and a standard LSTM has four. That gives a smaller recurrent parameter count under the same bias convention. It does not establish a universal runtime or accuracy ranking, and a GRU is not literally an LSTM with one gate deleted.

## 5. Recognize real pen trajectories

We now use actual handwriting coordinates instead of constructing a memory task whose answer is built into the setup.

The [UCI Pen-Based Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) was contributed by E. Alpaydin and F. Alimoglu. Each row contains eight ordered \((x,y)\) pairs and a digit label. The original collection separates writers between its training and test files. Our retained extract takes the first 60 specimens per class from the original training file and the first 30 per class from the original test file: 600 training and 300 **development** specimens.

We call the second set development because we inspect multiple architectures, seeds and input changes on it. It is no longer an untouched final test. Writer identifiers are absent from individual rows, so the inherited writer separation relies on the source's documented collection procedure. Exact coordinate-vector checks found no duplicates in either full source file and no duplicates across them.

The coordinates were normalized and resampled using each completed trace. We scale their provided range \(0\ldots100\) to \([-1,1]\) with \(x/50-1\). This is fixed arithmetic, with no learned normalization statistics. Because the source preprocessing uses the completed specimen, these results evaluate **completed-trace classification**, not a live system observing an unfinished pen stroke.

The [offline CSV](pen-trajectories.csv), [provenance](data-provenance.md) and [exact extraction record](data-extraction.json) preserve the source rows, attribution and hashes. No pretrained model or new data download is needed to run the lesson experiment.

### Fix the comparison before fitting

All three recurrent models use one unidirectional layer, input width 2, hidden width 32, and a linear ten-class head reading the final hidden output. They start at zero state for every specimen. Train each for 500 Adam updates at learning rate 0.005, batch size 64 sampled with replacement, and cross-entropy on the final digit. Clip the global gradient norm at 1 before each update. There is no dropout, augmentation, early stopping or best-seed selection.

Seeds 1, 2 and 3 each determine a new initialization. For a given seed, the batch generator supplies the same specimen indices across the three architectures. Their different parameter shapes prevent claiming identical initial weights. Native recurrent biases are set to zero, except for an effective LSTM forget bias of 1.

There are also two fixed logistic-regression baselines, each with scaling fitted only on training rows: one sees orderless coordinate statistics; the other sees all sixteen ordered coordinates. Neither is recurrent. They make the contribution of information representation visible.

### Complete program

Create a Python environment with NumPy, PyTorch and scikit-learn, then save the CSV beside the following file. The author run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1. Run:

```text
python pen-sequence-learning.py
```

The program records every declared run, final probabilities, reversal and point-swap results, and the three seed-1 model weights in `calculated-inputs.json`. It does not choose a winning model using development results.

```python
"""RNN, LSTM and GRU on completed real pen trajectories; no online-input claim."""
from pathlib import Path
import csv
import json
import platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent


def load_data():
    with (ROOT / "pen-trajectories.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    coordinates = np.array([[float(row[f"{axis}{time + 1}"]) for time in range(8)
                             for axis in ("x", "y")] for row in records])
    identifiers = [row["source_id"] for row in records]
    assert len(set(identifiers)) == len(records)
    assert len(np.unique(coordinates, axis=0)) == len(records)
    assert np.isfinite(coordinates).all() and coordinates.min() >= 0 and coordinates.max() <= 100
    labels = np.array([int(row["digit"]) for row in records])
    train = np.array([row["partition"] == "train" for row in records])
    assert train.sum() == 600 and (~train).sum() == 300
    return coordinates.reshape(-1, 8, 2) / 50 - 1, labels, train, identifiers


class PenClassifier(nn.Module):
    def __init__(self, kind, hidden_size=32):
        super().__init__()
        self.kind = kind
        self.hidden_size = hidden_size
        self.recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](
            2, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 10)
        with torch.no_grad():
            self.recurrent.bias_ih_l0.zero_()
            self.recurrent.bias_hh_l0.zero_()
            if kind == "lstm":
                self.recurrent.bias_ih_l0[hidden_size:2 * hidden_size].fill_(1)

    def forward(self, sequences, state=None):
        outputs, final_state = self.recurrent(sequences, state)
        return self.readout(outputs[:, -1]), outputs, final_state


@torch.no_grad()
def assess(model, sequences, labels, details=False):
    model.eval()
    logits, outputs, _ = model(sequences)
    predicted = logits.argmax(1)
    report = {"correct": int((predicted == labels).sum()),
              "cross_entropy": float(F.cross_entropy(logits, labels))}
    if details:
        report.update(predictions=predicted.tolist(), probabilities=logits.softmax(1).tolist())
    return report


def features(sequences, orderless=False):
    if orderless:
        return np.concatenate([sequences.mean(1), sequences.std(1),
                               sequences.min(1), sequences.max(1)], axis=1)
    return sequences.reshape(len(sequences), -1)


def main():
    torch.set_num_threads(1)
    sequences, labels, train, identifiers = load_data()
    x_train = torch.tensor(sequences[train], dtype=torch.float32)
    x_dev = torch.tensor(sequences[~train], dtype=torch.float32)
    y_train = torch.tensor(labels[train], dtype=torch.long)
    y_dev = torch.tensor(labels[~train], dtype=torch.long)
    report = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "torch": torch.__version__, "sklearn": sklearn.__version__},
              "train_ids": np.array(identifiers)[train].tolist(),
              "development_ids": np.array(identifiers)[~train].tolist(),
              "development_labels": labels[~train].tolist(), "baselines": {}, "runs": [], "saved_models": {}}
    for orderless in (True, False):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
        model.fit(features(sequences[train], orderless), labels[train])
        original = model.predict_proba(features(sequences[~train], orderless))
        reversed_prob = model.predict_proba(features(sequences[~train, ::-1], orderless))
        name = "orderless_statistics" if orderless else "ordered_flattened"
        report["baselines"][name] = {"development_correct": int((original.argmax(1) == labels[~train]).sum()),
            "cross_entropy": float(log_loss(labels[~train], original)),
            "train_correct": int((model.predict(features(sequences[train], orderless)) == labels[train]).sum()),
            "reversed_correct": int((reversed_prob.argmax(1) == labels[~train]).sum()),
            "reverse_max_probability_change": float(np.max(np.abs(original - reversed_prob)))}
    for seed in (1, 2, 3):
        for kind in ("rnn", "lstm", "gru"):
            torch.manual_seed(seed)
            model = PenClassifier(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            generator = torch.Generator().manual_seed(100 + seed)
            run = {"seed": seed, "kind": kind, "parameters": sum(p.numel() for p in model.parameters()),
                   "trajectory": [], "clipped_updates": 0, "largest_preclip_gradient_norm": 0.0}
            for step in range(501):
                if step in (0, 1, 100, 300, 500):
                    run["trajectory"].append({"step": step, "train": assess(model, x_train, y_train),
                                               "development": assess(model, x_dev, y_dev)})
                if step == 500:
                    break
                model.train()
                batch = torch.randint(len(x_train), (64,), generator=generator)
                logits, _, _ = model(x_train[batch])
                loss = F.cross_entropy(logits, y_train[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                run["clipped_updates"] += int(norm > 1.0)
                run["largest_preclip_gradient_norm"] = max(run["largest_preclip_gradient_norm"], norm)
                optimizer.step()
            run["final"] = assess(model, x_dev, y_dev, details=True)
            run["reversed"] = assess(model, x_dev.flip(1), y_dev, details=True)
            swapped = x_dev.clone()
            swapped[:, [2, 3]] = swapped[:, [3, 2]]
            run["swapped_points_3_4"] = assess(model, swapped, y_dev, details=True)
            if seed == 1:
                report["saved_models"][kind] = {name: value.detach().tolist() for name, value in model.state_dict().items()}
                with torch.no_grad():
                    _, outputs, _ = model(x_dev[:2])
                run["first_two_examples"] = {"source_ids": np.array(identifiers)[~train][:2].tolist(),
                    "labels": y_dev[:2].tolist(), "coordinates": x_dev[:2].tolist(),
                    "hidden_states": outputs.tolist(),
                    "prefix_readout_probabilities": model.readout(outputs).softmax(-1).detach().tolist()}
            report["runs"].append(run)
            print(f"seed={seed} kind={kind} final={run['trajectory'][-1]}", flush=True)
    (ROOT / "calculated-inputs.json").write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report["baselines"], indent=2))


if __name__ == "__main__":
    main()
```

The forward pass returns a sequence of hidden outputs and a final state. `outputs[:, -1]` is correct here because every specimen has exactly eight valid points. Section 7 explains why this expression becomes wrong for a padded batch. Cross-entropy receives raw scores; softmax is used when reporting probabilities.

### What actually happened

| Model | Trainable parameters | Development correct, seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- | --- | --- |
| Orderless statistics + logistic regression | Baseline feature model | 152/300 | Same fixed baseline | Same fixed baseline |
| Ordered coordinates + logistic regression | Baseline feature model | 262/300 | Same fixed baseline | Same fixed baseline |
| Tanh RNN, 32 state features | 1,482 | 268/300 | 267/300 | 274/300 |
| LSTM, 32 cell/hidden features | 4,938 | 273/300 | 276/300 | 271/300 |
| GRU, 32 state features | 3,786 | 279/300 | 276/300 | 275/300 |

The ordered baseline already gets many examples right. Recurrence is useful to investigate, but it is not necessary merely to make order available. These short traces also do not test whether a model can retain a cue over hundreds of steps.

Seed-1 development cross-entropies are approximately 0.3904 for RNN, 0.2629 for LSTM and 0.2611 for GRU; lower is better. The corresponding training correct counts are 597, 595 and 594 out of 600. Accuracy and cross-entropy measure different aspects: changing a probability can change the loss without changing the most likely digit.

All nine runs remain visible in the retained results. A few additional correct specimens in this small, inspected development set do not settle which architecture will work best on another task. No latency was measured, so these numbers support no hardware-speed claim.

### Reverse the trace while keeping the fitted weights

The orderless model returns the same probabilities to floating-point precision when the eight points are reversed. The ordered baseline falls from 262 to 22 correct out of 300. The seed-1 RNN, LSTM and GRU fall from 268, 273 and 279 to 42, 39 and 37 respectively.

The digit label is retained for this constructed input transformation. The reversal uses the same set of points but changes traversal direction, including start and end. It is an explicit distribution-change probe, not a fresh natural handwriting benchmark. Reversing inputs at evaluation is also different from retraining on reversed inputs.

Swapping only points 3 and 4 gives a milder but still meaningful change: seed-1 correct counts become 262, 239 and 259. The models depend on more than the unconnected point set.

**Investigation: change the actual pen path.** Select one retained specimen, move the x-coordinate of point 3, or swap two neighboring points. Before applying the edit, predict whether the most likely digit will change and whether a selected digit's probability will rise, fall or stay equal. Recompute the whole affected suffix with the selected fixed model. Display the numbered path, selected hidden coordinates and class probabilities together.

The checked edit that adds 0.2 in normalized units to point 3's x-coordinate changes probabilities for the first two retained specimens but does not need to change the predicted digit. That is a useful outcome: a visible input change is not evidence that the classifier must flip its decision.

The hidden-state readout after each intermediate point is a view into the same fitted classifier. It was trained for the final point and receives coordinates preprocessed from the complete trace. Treat those intermediate readouts as a mechanism illustration, not validated early-recognition probabilities.

## 6. State belongs to a stream

If five observations belong to one continuous sequence, processing the first two and carrying their final state into the next three should reproduce processing all five together, assuming the same weights and deterministic evaluation behavior. Splitting a file into chunks does not create a new phenomenon in the data.

Resetting at the boundary is different: it removes the preceding context. Detaching at the boundary is different again: it preserves the numerical state but prevents a later loss from sending gradients through the earlier chunk.

| Boundary operation | Forward state value | Credit to the earlier computation |
| --- | --- | --- |
| Carry state | Preserved | Preserved if graph retained |
| Carry detached state | Preserved | Cut at the boundary |
| Reset state to zero | Replaced | Earlier state no longer used |

For a constructed five-observation GRU fixture, whole-sequence and carried-chunk outputs agree exactly in the author calculation. Resetting changes the last hidden output by up to 0.046305. Detaching preserves the forward outputs exactly, but the later loss's gradients with respect to the first two inputs become zero.

This is **truncated BPTT** when applied during training across chunk boundaries. It bounds how far the current loss directly backpropagates. It does not reset memory or prove that the model cannot learn any dependency longer than a chunk. The carried states and shared parameters can still support indirect learning; the truncated gradient differs from the full-sequence objective's gradient.

**Visual: two arrows at every boundary.** Draw a forward state arrow and a backward credit arrow separately. Carry leaves both connected; detach cuts the backward arrow; reset replaces the incoming state. A single “memory length” slider cannot explain these three operations correctly.

For real independent pen specimens, reset for every specimen. Carrying state from one random specimen into another would make its prediction depend on an unrelated writer's previous digit. For a service managing several ongoing sessions, store state by session identity and gather the corresponding states when forming a batch. Reordering batch rows requires reordering states with their owners.

After correcting an earlier observation, recompute the affected suffix from a valid prior state. A state cached after the old observation no longer represents the corrected prefix. A model-weight update can also invalidate a stored state if exact agreement with a fresh run under the new model is required. Practical training often carries states across parameter updates as an approximation; it should not be described as the exact full-sequence computation.

### Global gradient clipping

Treat all parameter gradients as one long vector \(g\). For threshold \(\tau>0\),

\[
g_{\text{clipped}}=g\min\left(1,\frac{\tau}{\|g\|_2}\right)
\]

for a nonzero finite gradient; leave the zero vector unchanged. The vector \([3,4]\) has norm 5. At threshold 2, global clipping gives \([1.2,1.6]\), preserving its direction. Clipping coordinates individually to \([-2,2]\) gives \([2,2]\), a different direction.

Log the norm before clipping so that persistent oversized updates are visible. Clipping does not fix nonfinite gradients or recover missing long-range credit. In the handwriting runs, clipping was activated 136, 245 and 10 times for seed-1 RNN, LSTM and GRU; those counts describe this optimization setup, not an intrinsic ranking of gate quality.

The complete [recurrent-mechanics program](recurrent-mechanics.py) executes the chunk, detach, padding, gate and independent native-parity calculations without refitting the handwriting models.

## 7. Different lengths, stacks and directions

Most real sequence collections do not give every example the same length. Padding is an arrangement for batching; a padded zero is not automatically “no observation.” Even a zero input can change a recurrent state through its recurrent matrix and bias.

For a right-padded unidirectional recurrence, valid outputs before the padding are unaffected by later padding. But reading the final padded position gives the state after extra updates. A backward recurrence encounters right padding before it reaches valid observations, so even its valid outputs can be contaminated.

Packing tells the native recurrent module each sequence's actual length. Here is a complete independent API example. The short sequences are constructed to teach batching, not a new dataset experiment:

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(8)
sequences = [
    torch.tensor([[.2, -.1], [.8, .3], [-.4, .9], [.7, -.2], [.1, .5]]),
    torch.tensor([[.2, -.1], [.8, .3], [-.4, .9]]),
    torch.tensor([[.2, -.1]]),
]
lengths = torch.tensor([len(sequence) for sequence in sequences])
padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
model = nn.GRU(2, 3, batch_first=True, bidirectional=True)
model.eval()
with torch.no_grad():
    packed = pack_padded_sequence(
        padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, final = model(packed)
    # One layer: direction, specimen, feature.
    summary = torch.cat([final[0], final[1]], dim=1)
    separate = torch.stack([model(sequence[None])[1][:, 0]
                            for sequence in sequences], dim=1)
print(padded.shape)   # torch.Size([3, 5, 2])
print(final.shape)    # torch.Size([2, 3, 3])
print(summary.shape)  # torch.Size([3, 6])
print(torch.allclose(final, separate, atol=1e-6))  # True
```

The summary concatenates the final forward and final backward states. It does not take both halves of the output at the last valid index: at that index, the backward component has only just started reading from the end.

The [PyTorch LSTM API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html) documents the hidden/cell shapes, projections and bidirectional final-state distinction. With `batch_first=True`, inputs and sequence outputs use batch first, but final hidden states still use \([\text{layers}\times\text{directions},B,H]\). An LSTM additionally returns a cell-state tensor. With a projection, hidden width and cell width can differ.

In a stack, layer 2 consumes layer 1's output at each position. This adds depth across layers as well as the recurrence across positions. In a bidirectional stack, that input has both directions' features. Do not simply multiply a one-layer parameter count by the number of layers without checking the next layer's input width.

Bidirectional models may use future observations within a completed input. That can be useful for offline labeling or an encoder that receives a whole source sequence. It is incompatible with claiming a causal output at position \(t\) before later observations exist. Separate the deployment question from a library flag.

If there is a loss at every valid position, mask the padded targets too. For lengths 5, 3 and 1 there are nine valid targets, not fifteen. Dividing by fifteen dilutes the loss and changes its scale as padding changes. Packing inputs and masking output losses solve related but distinct problems.

## 8. Deeper questions and practical model choices

### How much does the model cost?

For one layer, one direction, input width \(D\), hidden width \(H\), and two native bias vectors per affine group:

\[
\begin{aligned}
\text{RNN parameters}&=H(D+H+2),\\
\text{GRU parameters}&=3H(D+H+2),\\
\text{LSTM parameters}&=4H(D+H+2).
\end{aligned}
\]

Add \(K(H+1)\) for a \(K\)-class linear head. With \(D=2,H=32,K=10\), the recurrent counts are 1,152, 3,456 and 4,608, and the head adds 330. A textbook formula with one combined bias uses \(+1\) instead of \(+2\); that convention is not a contradiction.

The ordinary dense recurrent work scales with \(T\) times the per-step input/state matrix work. For a fixed-width unidirectional model, its carried inference state need not grow with prefix length. Full BPTT, however, retains intermediate computations for differentiation and its activation storage normally grows with sequence length. A native fused kernel can be much faster than a Python loop, but the actual runtime depends on shapes, backend and hardware.

The standard tanh/LSTM/GRU hidden-state dependence imposes a sequential forward chain. Input projections and batch members can be processed in parallel. Later [State Space Models](/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals) use more structured state updates that admit other evaluation strategies. Their parallelism should not be attributed to every nonlinear recurrence.

### The full LSTM derivative includes both state vectors

If gate values and \(h_{t-1}\) are held fixed in a non-peephole cell, the derivative of \(c_t\) with respect to \(c_{t-1}\) is \(\operatorname{diag}(f_t)\). That is the useful direct cell path.

For the complete recurrence, the state is \((h,c)\). A change in earlier cell state can also affect earlier hidden output and hence subsequent gates. The full local derivative is a block matrix:

\[
\frac{\partial(h_t,c_t)}{\partial(h_{t-1},c_{t-1})}
=
\begin{bmatrix}
\partial h_t/\partial h_{t-1}&\partial h_t/\partial c_{t-1}\\
\partial c_t/\partial h_{t-1}&\partial c_t/\partial c_{t-1}
\end{bmatrix}.
\]

For the explicitly defined scalar cell in the retained mechanics program, at \((h,c)=(0.2,0.8)\) it is approximately
\[
\begin{bmatrix}0.169441&0.318697\\0.458119&0.746494\end{bmatrix}.
\]
The forget value 0.746494 appears in the lower-right entry. It is not the whole matrix. Products of these complete Jacobians describe total state sensitivities. A plot of \(f^T\) must therefore be labeled a direct-path, fixed-gate illustration rather than a measured full LSTM gradient.

### A matrix's eigenvalues are not the whole temporal story

For tanh, \(\|J_t\|_2\leq\|W_h\|_2\). If every local Jacobian norm is bounded by a common \(q<1\), the product norm is at most \(q^{T-k}\). This is a sufficient contraction condition, not a necessary diagnosis for every trajectory.

Time-varying matrices can behave very differently from repeating one fixed matrix. Consider
\[
A=\begin{bmatrix}0&2\\0&0\end{bmatrix},\quad
B=\begin{bmatrix}0&0\\2&0\end{bmatrix}.
\]
Each has only zero eigenvalues, but \(BA=\operatorname{diag}(0,4)\) amplifies one direction. This constructed counterexample explains why inspecting each step's spectral radius is insufficient for a temporal product. It is not a claim that these are the Jacobians of our fitted pen model.

### Initialization, regularization and variants

Orthogonal recurrent initialization can preserve norms for a linear transformation at initialization, but tanh derivatives and training updates still matter. The [initialization lesson](/learn/path/full-curriculum/weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals) explains the broader variance and singular-value picture.

When setting a PyTorch LSTM forget bias, its two bias vectors add. Setting both forget slices to 1 creates effective bias 2. Our program sets the input-side slice to 1 and the recurrent-side slice to zero. The packed LSTM affine order is input, forget, candidate, output; GRU uses reset, update, candidate.

Native recurrent-module dropout is applied between stacked layers, excluding the last layer. It is not an automatic recurrent-state dropout mechanism, and with one layer there is no intervening layer on which to apply it. A separate input/output mask or a specialized recurrent dropout method needs its own declared behavior. `eval()` changes training-dependent module behavior; `no_grad()` independently disables autograd recording.

Peephole LSTMs let gates inspect cell state; projected LSTMs use a narrower hidden output than their cell width. These are concrete architectural choices, not synonyms for every LSTM. More specialized descendants appear later in the module.

Choose candidate architectures from the evidence and deployment constraints. Fixed-length ordered features can be a strong baseline; a causal recurrent state can serve streaming inputs; temporal convolutions supply local receptive fields; attention supplies direct content-dependent access to other positions. No length threshold makes one architecture automatically correct. Measure quality, memory and latency under the actual input-availability contract.

## 9. Practice: change the problem before checking the answer

### 1. Alter the middle observation

In the scalar RNN from section 2, replace \(x_2=-0.2\) with \(0.2\), leaving all other inputs and weights fixed. Which state values change? Calculate the final value.

<details><summary>Hint</summary>

Reuse the unchanged first state. Recompute the second preactivation, then use the new second state at step 3.
</details>

<details><summary>Solution</summary>

Only \(h_2\) and \(h_3\) change. \(h_2=\tanh(0.16+0.6(0.396930432)+0.1)\), and \(h_3=\tanh(0.56+0.6h_2+0.1)\). The final state rises to approximately 0.733564. The earlier state cannot depend on a later input in this causal recurrence.
</details>

### 2. Retain a negative memory

Let \(c_{\text{old}}=-0.6,f=0.8,i=0.25,g=0.4,o=0.5\). Calculate the retained term, write term, new cell and hidden output. Would changing only \(o\) change the new cell?

<details><summary>Hint</summary>

The output gate is applied after the cell update. Keep the signs of the two contributions separate.
</details>

<details><summary>Solution</summary>

The retained term is −0.48 and the write is +0.10, so \(c=-0.38\) and \(h=0.5\tanh(-0.38)\approx-0.18135\). Changing only the output gate leaves this step's cell value unchanged. It changes \(h\), which can affect gates at later steps.
</details>

### 3. A reset gate is not a reset command

A GRU coordinate has old state 0.8 and update/retain gate 0.9. After closing the candidate's reset gate, its candidate is −0.2. Is its new state zero? What update gate would retain none of the old state?

<details><summary>Hint</summary>

The candidate path and the final blend are separate. Apply the blend after finding the candidate.
</details>

<details><summary>Solution</summary>

The new state is \(0.9(0.8)+0.1(-0.2)=0.7\). Closing the reset gate does not erase the retained term. In our convention \(z=0\) makes the state equal to the candidate, which is still not necessarily zero.
</details>

### 4. Longer retention without a promise

You want a fixed direct cell path to retain 80% after 50 steps with no writes. Derive the forget factor. Explain why setting that bias does not guarantee 80% total gradient retention in a trained LSTM.

<details><summary>Hint</summary>

Solve \(f^{50}=0.8\). Then distinguish the controlled fixed-factor experiment from input-dependent gates and the full \((h,c)\) state.
</details>

<details><summary>Solution</summary>

\(f=0.8^{1/50}\approx0.99555\). A sigmoid preactivation \(\log(f/(1-f))\) produces that factor in isolation. Real preactivations also contain input and hidden-state terms, and full gradients include other paths. The calculation describes a specified direct-path experiment.
</details>

### 5. The batch reordered itself

Two ongoing sessions A and B occupy batch rows 0 and 1. The scheduler next returns rows B, A. An implementation passes its old state tensor unchanged. What is wrong, and what additional event requires more than swapping state rows?

<details><summary>Hint</summary>

Associate every state with the prefix that produced it, not with a permanent batch index.
</details>

<details><summary>Solution</summary>

B receives A's history and A receives B's. Gather states by session identity in the new row order. If a previous observation was corrected, swapping rows is insufficient: recompute the affected state suffix from a valid prefix. Ending a session requires retiring its state before that identifier or slot is reused.
</details>

### 6. Padding changes the wrong result

You batch sequences of lengths 4 and 2 with right padding. You only change the second sequence's padded values. Predict the effect on its valid forward outputs, its padded final hidden state, and its valid backward outputs.

<details><summary>Hint</summary>

Follow which values each direction visits before reaching the valid position.
</details>

<details><summary>Solution</summary>

The valid forward outputs are unchanged. The padded final state can change because it includes the extra updates. Valid backward outputs can change because the reverse recurrence visits padding first. Packing with the true lengths removes those padding updates; reading the correct directional final states then represents the actual sequence.
</details>

### 7. Detach or reset?

A later-chunk loss should use preceding context, but you can retain an autograd graph for only the current chunk. Choose carry, detach-and-carry, or reset. What agreement can you expect with a whole-sequence evaluation before any optimizer update?

<details><summary>Hint</summary>

Ask separately whether the numbers and the derivative connections must survive.
</details>

<details><summary>Solution</summary>

Detach and carry. With identical weights and deterministic behavior, forward outputs match the unbroken recurrence. Gradients through the detached boundary do not match full BPTT. Resetting would change the forward computation too.
</details>

### 8. Design an honest follow-up

You want to claim that a model recognizes a digit before the pen finishes and that it generalizes to new writers. Can you use the current intermediate probabilities as your evidence? Design the missing data and evaluation conditions.

<details><summary>Hint</summary>

Inspect both the time at which preprocessing can be computed and the unit kept separate by the split.
</details>

<details><summary>Solution</summary>

The current complete-trace normalization/resampling uses information unavailable at a live prefix, and the classifier was trained for final-trace output. Collect or retain raw timestamped prefixes, specify causal preprocessing and the prediction time, train the declared prefix objective, keep writers separate, and reserve an untouched writer holdout after development choices. Report accuracy and uncertainty by prefix availability, with an appropriate baseline. The current results motivate this study but do not perform it.
</details>

You are ready for the next topic when you can distinguish weights from state, execute a gated update, explain a shared-weight gradient, and keep state ownership, sequence lengths and available inputs consistent with the task. The optional full-Jacobian analysis can be revisited as you study more specialized recurrent models.

## 10. Continue and learn another way

The next [Sequence-to-Sequence Encoder–Decoder lesson](/learn/path/full-curriculum/sequence-to-sequence-encoder-decoder?module=deep-learning-fundamentals) changes the output from one digit into a sequence. It introduces decoder inputs, teacher forcing, stopping and generated-prefix evaluation. The following [Bahdanau & Luong Attention lesson](/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals) lets a decoder consult source positions instead of relying only on one final summary.

For a different explanation or a deeper reference:

- [Christopher Olah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is an approachable diagram-led walkthrough of cell, gate and output paths. Use it after section 3; read its strong long-memory intuition together with our fixed-gate and full-derivative qualifications.
- [Stanford CS231n Lecture 10: Recurrent Neural Networks](https://www.youtube.com/watch?v=6niqTuYFZLQ), with [official slides](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf), offers a spoken route through recurrence, language modeling, image captioning and gated models. It is a 2017 conceptual lecture, not a current framework installation guide; its captioning/attention branches lead beyond this page.
- [D2L: Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) develops the gradient chain and truncation more formally. Its surrounding chapters also provide text-model implementations; our real pen example offers a different applied route.
- [PyTorch GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) and [LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html) document the exact native gate conventions and tensor shapes used here. Check them when transferring weights or changing directions, layers or projections.
- [Pascanu et al., On the Difficulty of Training Recurrent Neural Networks](https://proceedings.mlr.press/v28/pascanu13.pdf) is the mathematical route to temporal gradient products and clipping. [Hochreiter & Schmidhuber's LSTM paper](https://www.bioinf.jku.at/publications/older/2604.pdf) and [Cho et al.'s encoder–decoder paper](https://arxiv.org/abs/1406.1078) provide historical mechanisms; their original algorithms and experimental claims should be read in their own settings.

The [saved numerical results](calculated-inputs.json) and [mechanics calculations](mechanics-results.json) separate fitted-model evidence from exact constructed examples. They let you inspect the numbers behind the lesson rather than treating an attractive plot as evidence by itself.
