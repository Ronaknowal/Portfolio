# Interleaved / Cross-Attention Architectures

An image can contain a bicycle, a person and a road. “What is the person holding?” and “What surface is the bicycle on?” require different uses of the same image. A useful model needs more than one fixed sentence describing everything: it needs a way for its current question or partially written answer to read relevant information from the image.

**Cross-attention is a learned, question-dependent read from a separate collection of representations.** The reader supplies queries; the collection supplies keys and values. The result contains one updated vector per query, however many items were in the collection. In a visual-language model, that collection might contain image patches. In translation it contains source-language states. The two sides need not be different modalities.

There are several ways to connect a reader and a collection. We will construct the read operation, place it inside different architectures, then train a small model that answers two different questions about handwritten digits. The experiment is deliberately small enough to inspect. Its purpose is to learn architecture and evaluation, not to reproduce a large pretrained assistant.

**First pass:** §§1–5, the worked experiment in §6, and core practice in §9. Return to §7 for detailed cost and cache accounting and §8 for deeper design decisions. You need vector dot products, softmax, residual connections and the idea of fitting a loss. Each is refreshed where used. The preceding [vision-transformer lesson](/learn/path/full-curriculum/vision-transformers-vit-deit-swin-dinov2?module=deep-learning-fundamentals) explains image tokens; [self-attention](/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals) explains the attention machinery in more depth. Allow roughly an hour for the core explanation and another hour for computation and practice; the advanced branches can be a separate sitting.

By the end you should be able to draw which representations can read which others, calculate a rectangular attention output, distinguish compression from fusion, explain how a frozen model can transmit gradients, and choose an experiment that tests whether a system actually uses its visual input.

## 1. A query reads a memory

Imagine three memory slots, each containing two things. Its **key** describes when that slot is relevant. Its **value** is the information contributed if selected. A **query** describes what the reader currently seeks. These descriptions are numeric vectors learned for a task, rather than literal English tags.

For each query:

1. Compare it with every permitted key using a dot product.
2. Scale the scores so increasing the vector width does not automatically make their magnitudes large.
3. Apply softmax across the memory slots, producing nonnegative weights summing to one.
4. Form the weighted sum of their values.

Changing the question can change the weights while the memory remains fixed. Changing a value changes what can be returned even if the selection weights stay unchanged. This separation is useful: relevance and content are different jobs.

**Visual: the rectangular read.** Draw two query rows facing three key/value columns. Selecting a row reveals its three score calculations, normalized weights and three weighted arrows contributing to the output. Keep the query and output on the same horizontal track: three memory slots do not create three output slots.

Here is a completely specified example. All quantities are dimensionless constructed features. Let the key width be two:

\[
Q=\sqrt2\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
K=\begin{bmatrix}\log2&0\\0&\log2\\0&0\end{bmatrix},\qquad
V=\begin{bmatrix}2&0\\0&4\\2&2\end{bmatrix}.
\]

The scores are \(QK^T/\sqrt2\). Their first row is \((\log2,0,0)\), so exponentiation gives \((2,1,1)\). Dividing by four gives weights \((1/2,1/4,1/4)\). The first output is

\[
\tfrac12(2,0)+\tfrac14(0,4)+\tfrac14(2,2)=(1.5,1.5).
\]

The second query instead gives weights \((1/4,1/2,1/4)\) and output \((1,2.5)\). This is the complete selection-and-read mechanism, before adding projections, heads or a language model.

**Predict, then edit.** Suppose the first query cannot read slot three. Record its new two output coordinates before revealing them. Softmax must renormalize the remaining weights: they become \((2/3,1/3,0)\), and the output becomes \((4/3,4/3)\). Now change the forbidden slot’s value to \((100,-100)\). The first output stays unchanged. The second query, for which the slot is still allowed, changes to \((25.5,-23)\). A mask is a structural restriction on access, not simply a visual dimming of a column.

Masking every slot leaves no probability distribution to normalize. A robust implementation needs an explicit convention, such as a valid empty-context token or a skipped residual update. It must not quietly show a row of NaNs or fabricate an alignment.

## 2. Shapes reveal the architecture

Let \(X\in\mathbb R^{T\times d_x}\) be the current reader states and \(M\in\mathbb R^{S\times d_m}\) the memory. A single head computes

\[
Q=XW_Q,\quad K=MW_K,\quad V=MW_V,
\quad A=\operatorname{softmax}_{\text{memory}}(QK^T/\sqrt{d_k}+B),
\quad O=AV.
\]

Here \(W_Q:d_x\to d_k\), \(W_K:d_m\to d_k\), \(W_V:d_m\to d_v\). The additive mask \(B\) is zero for permitted pairs and negative infinity for forbidden pairs. Queries and keys must agree on \(d_k\); input memory width and reader width need not agree. Values can have a different width \(d_v\).

| Quantity | One-head shape | Meaning |
| --- | --- | --- |
| Queries | T × dₖ | T readers asking what to retrieve |
| Keys | S × dₖ | S comparable relevance descriptions |
| Values | S × dᵥ | S candidate contributions |
| Weights | T × S | One distribution over memory per reader |
| Output | T × dᵥ | One retrieved vector per reader |

For a batch of size \(B_s\) and \(H\) heads, weights have shape \(B_s\times H\times T\times S\). We use \(B_s\) for batch size to distinguish it from the mask. Each head has its own learned projections. Concatenating head outputs and applying an output projection returns the reader width, allowing a residual addition to \(X\). Different heads provide different learned read functions; training does not guarantee a neat human-interpretable specialization for each.

Cross-attention leaves the memory unchanged within this operation. Updating it requires another operation. By comparison, self-attention uses the same stream to generate Q, K and V; a multimodal self-attention layer can therefore update visual and text positions together, subject to its mask.

**A useful invariance.** Reordering key/value pairs together does not change the weighted sum if all associated masks and positional information travel with their slots. It only reorders the intermediate columns. Reordering values without their keys changes the associations and generally changes the result. A position encoding deliberately lets the model distinguish “left” and “right”; this invariance does not imply a model ignores physical position.

**Visual: shape assembly.** Build one head from rectangular matrix tiles; join three head outputs into one reader-width tile. Let the learner set T=2 and S=5, then swap them. Require an output-shape prediction before revealing which axis follows the reader. A separate key/value pairing exercise makes a mistaken permutation visible through changed numeric outputs.

## 3. Three choices that are easy to confuse

The word *interleaved* can refer to the input sequence or to where blocks are inserted. Neither meaning uniquely identifies the attention operation.

| Design choice | Question it answers | Example |
| --- | --- | --- |
| Input arrangement | Where do images and text occur in an example? | image A → question A → answer A → image B → question B |
| Fusion operation | How does information cross between representations? | Text reads separate visual K/V, or projected visual tokens join a shared stream |
| Block placement | At which depths does fusion happen? | A new cross-attention block between every few existing language blocks |

An architecture can consume interleaved images and text **and** contain cross-attention. Flamingo is an example. Calling every interleaved input a “self-attention-only architecture” hides the very mechanism we want to compare.

### Put projected image tokens in the language stream

A vision encoder maps an image into S vectors. A linear layer or MLP changes their width to match the language model. Those projected vectors are inserted alongside text embeddings. In a simple prefix arrangement:

\[
[\text{image}_1,\ldots,\text{image}_S,\text{question},\text{answer prefix}].
\]

A causal language-model mask allows each position to read earlier positions and itself. Answer positions can use image and question information without a separate cross-attention layer. Notice two different causality boundaries: an image encoder may already have mixed all patches bidirectionally before its outputs enter the causal language stream. Causality over language-model positions does not undo that preprocessing.

If the image is already known when answering, this is appropriate. If future video frames are unavailable in an online task, encoding a full future clip first can leak future information even when the language-model mask looks causal. Define availability at the data boundary, not merely inside the final attention matrix.

### Keep image features in a separate memory

Text states can instead use a residual update from cross-attention. Their own self-attention still handles linguistic context. A text token may directly read one image, all earlier images, or a retrieved subset, depending on a visual mask.

**Visual: an access map, not a family leaderboard.** For two images and two short question/answer spans, show shared-stream causal attention on the left and text self-attention plus rectangular image reads on the right. The learner marks which image each answer is allowed to use. Highlight actual paths through earlier text states as well as direct image edges. This explains how an earlier image can have an indirect influence even when it is not directly accessible at the current cross-attention block.

### Train only the intended predictions

For answer tokens \(a_1,\ldots,a_R\), a usual conditional objective is

\[
-\sum_{r=1}^{R}\log p(a_r\mid \text{image},\text{question},a_{<r}).
\]

Teacher forcing supplies the earlier correct answer tokens while training. The state predicting \(a_r\) must not contain \(a_r\) itself or future answers. Input tokens and prediction targets are shifted accordingly; the loss mask chooses which target positions count. Ignoring question tokens in the loss does not mean hiding them from attention. The attention mask controls information access; the loss mask controls supervision.

For example, with an answer “red bicycle”, the first supervised state sees the image, question and answer-start marker and predicts “red”. The next sees “red” and predicts “bicycle”. If a diagram puts “red” into the very state being graded for guessing “red”, it teaches copying a label rather than conditional generation.

## 4. Compressing a memory and opening a gate

### Learned queries can build a smaller memory

So far the queries came from a question or current text state. They can also be **learned latent vectors**: a fixed number of trainable slots that read a large input and produce a smaller representation. Subsequent computation operates on those outputs. A resampler therefore changes how many context tokens are passed onward; a width projection alone does not.

Suppose a 16-patch image is read by four learned queries. The read matrix is 4×16, and the resulting four vectors can become a language-model prefix or the memory for later text cross-attention. Those four slots need not correspond to four spatial quadrants. Their meaning depends on the training objective. An attention map over learned slots cannot be labeled as a spatial heatmap without a supported mapping back to image coordinates.

Compression also loses information in some cases. A single uniform averaging slot maps values \((1,3)\) and \((2,2)\) to the same mean, two. A downstream reader seeing only that mean cannot distinguish their spread. More elaborate learned compression may preserve task-relevant distinctions, but its fixed size is not a promise to preserve every detail a future question might need.

The [long-context sequence-model lesson](/learn/path/full-curriculum/long-context-sequence-models-transformer-xl-griffin-perceiver?module=deep-learning-fundamentals) develops Perceiver and Perceiver IO, including repeated input reads and output queries. Here the important connection is that compression and fusion can be composed. Repeated reads must still be counted when estimating work.

**Visual: a compression collision.** Feed the two value sets above through an editable one-slot averaging read. Ask whether a later “which set has larger spread?” query can succeed from the identical stored values. Then expose two separate slots and show which information was restored. Treat this as a precisely defined example of information loss, not a universal accuracy curve against latent count.

### A gate controls the initial residual update

Consider a new adapter \(F_\theta\) inserted into an existing network:

\[
Y=X+\tanh(\alpha)F_\theta(X,M).
\]

With \(\alpha=0\), this inserted operation is exactly the identity in ideal arithmetic: \(Y=X\). For the same inputs, unchanged surrounding parameters and the same evaluation behavior, the whole network initially matches the original network. This gives training a well-defined starting point. It does not guarantee all later language abilities will be preserved.

Why can learning start when the update is zero? For a scalar loss L,

\[
\frac{\partial L}{\partial\theta}=\tanh(\alpha)
\frac{\partial L}{\partial Y}\frac{\partial F_\theta}{\partial\theta},\qquad
\frac{\partial L}{\partial\alpha}=(1-\tanh^2\alpha)
\left\langle\frac{\partial L}{\partial Y},F_\theta\right\rangle.
\]

At zero, the adapter-weight gradient through this branch is zero, but the gate gradient can be nonzero. A gate update can open the route; later steps can change the adapter weights. If the adapter output is also identically zero, this particular route can stall. Other losses or shared routes can change that conclusion, so inspect the actual computation graph.

For a hand calculation, set X=1, F=2 and L=½(Y−3)². At α=0, Y=1 and ∂L/∂α=−4. A gradient step with rate .1 gives α=.4 and Y≈1.7599 before changing F. The loss falls from 2 to about.7690. Setting F=0 instead makes the first gate gradient zero. The distinction is visible without inventing a characteristic 50-step delay for all gated models.

**Visual: residual and gradient routes.** Show the unchanged X rail and a gated F rail. Let learners choose F and α, record a predicted gradient and take one exact step. Highlight parameter gradients separately from output values. The unchanged-output case should be as informative as the changing one.

### Frozen parameters still transmit gradients

A frozen layer has parameters excluded from updates; its input can still require a derivative. If a trainable adapter feeds a frozen language model, backpropagation must pass through the language model to teach the adapter. Wrapping that whole forward pass in `no_grad()` would break the learning route.

Conversely, a frozen vision encoder whose input requires no gradient can often run without recording gradients, or its features can be cached for fixed inputs and preprocessing. Dropout, normalization state, augmentation and parameter version still matter. “Frozen” is a training decision; “evaluation mode” controls certain layer behaviors; “do not record gradients” controls differentiation. They solve different problems.

## 5. Read named models as design choices

These are dated, documented recipes, not assertions about hidden internals of every current assistant.

| Model and primary source | Connection and training idea | What to inspect when adapting it |
| --- | --- | --- |
| [Flamingo, 2022](https://arxiv.org/html/2204.14198) | A resampler makes 64 visual tokens; gated cross-attention/dense blocks are inserted into a frozen language model. Inputs can interleave images and text. Text directly attends to its most recent preceding image; earlier text can carry older context. | Resampler latents also contribute K/V; image-availability masks, gate initialization and insertion frequency are part of the recipe. A latent is not a named image region. |
| [BLIP-2, 2023](https://proceedings.mlr.press/v202/li23q/li23q.pdf) | A Q-Former uses 32 learned 768-wide queries; its vision and text submodules share self-attention, with masks determined by contrastive, matching or generation objectives. A later stage projects query outputs into a frozen language model. | Query–text interaction and vision cross-attention are different routes. The 188M-parameter Q-Former is substantial. OPT-style decoder training and FlanT 5-style encoder/decoder prefix training are different variants. |
| [LLaVA, 2023](https://arxiv.org/pdf/2304.08485) | A linear projection inserts CLIP image features into the language embedding stream. Its first alignment stage updates the projection; instruction tuning then updates projection and language model while keeping vision frozen. | Assistant-answer loss masking, feature choice and the training data are central. Do not silently substitute a later MLP-projector variant for the original linear design. |
| [Idefics2, 2024](https://arxiv.org/pdf/2405.02246) | Uses a fully autoregressive multimodal stream with projection and learned pooling, commonly 64 visual tokens per image. | Its architecture study changes training choices as well as connection patterns. It is not simply a Flamingo-style gated cross-attention model. Small token count does not imply a separate visual-memory architecture. |

The BLIP-2 mask distinction is worth drawing. In its representation stage, contrastive query/text streams are separated, matching permits bidirectional interaction, and generation lets text read queries and earlier text while queries do not read text. If matching and generation use the same unrestricted mask, the latter can access information it is supposed to predict. The full objective definitions and experimental settings belong to the linked paper; a name alone does not specify a training recipe.

**Design exercise.** A frozen language model plus a learned visual prefix is possible: learning can travel through the frozen model to the prefix generator. Therefore “must stay frozen” does not logically force cross-attention. It changes the experiment you need to run. Compare a trained prefix, a trained resampler and an inserted adapter under the actual data, budget and evaluation constraints before adopting a family slogan.

## 6. A complete small image-and-question experiment

We now ask two questions about each real handwritten image: **Which digit is it?** and **Is it odd or even?** Both use the image, but their answer spaces differ. This gives us a transparent test of conditioning on a question without the hidden cost of downloading a pretrained vision encoder or language model.

The [400-row offline CSV](digits-400.csv) comes from UCI Optical Recognition of Handwritten Digits, credited to E. Alpaydin and C. Kaynak and distributed under CC BY 4.0. Each row contains 64 measured block-count intensities from 0 to 16, a digit label and its original one-based scikit-learn source ID. This is not MNIST. The two question templates and parity targets are our constructed teaching task over real observations. See [provenance](data-provenance.md).

Split **images first**, then derive both questions. The program uses 240 fit images,80 development images and 80 assessment images, stratified by digit. Both questions from an image stay in the same split. Otherwise one question could expose the same image during fitting and another could appear as supposedly unseen evidence. Source IDs and duplicate checks are retained. Writer identities are unavailable here, so this is not a new-writer evaluation. Earlier lessons use the same source subset; these results are classroom evidence, not an untouched benchmark.

Represent each question by a learned 24-dimensional vector. Split the 8×8 image into sixteen 2×2 patches; project each four-value patch to 24 dimensions and add a learned positional vector. Three attention heads read the resulting memory from the question. A residual feed-forward block and a12-class head predict digit 0–9 or even/odd classes 10–11. The classifier learns which part of that answer space each question uses.

We compare an ordinary flat image-plus-question MLP, the cross-attention classifier and a version with a zero-initialized scalar gate. All train from scratch. They have different parameter counts and inductive biases. The gated version isolates a mechanism; it is not a frozen pretrained Flamingo model. The flat model is a useful baseline because a more elaborate architecture is not automatically better on 8×8 digits.

Before running, record two expectations: will either question be answerable well without the image, and will a zero gate prevent all learning forever? Each assessment set has eight images per digit. A question-only model guessing one digit and always one parity gets 8+40=48 of 160 questions correct, or 30%. This baseline does not require fitting the images.

The full [program](cross-attention-study.py) reads the CSV, constructs every tensor, trains all nine fits (three models × three seeds), evaluates both question types, deliberately mismatches images and questions, and writes the actual evidence to [calculated-inputs.json](calculated-inputs.json). It uses CPU PyTorch and NumPy, one CPU thread, fixed 160-epoch budgets, AdamW at.003, and no dropout. Development scores are recorded but do not choose epochs or hyperparameters. The displayed assessment is now inspected evidence; a future design decision needs new evaluation data.

The patch construction is important: reshape 8×8 into four row blocks of height 2 and four column blocks of width 2, permute block coordinates together, then flatten each 2×2 patch. A plain reshape to 16×4 would cut horizontal strips instead. The code makes this data geometry explicit before attention.

### Complete executable program

~~~python
"""Reproduce the small query-conditioned digits study and exact attention fixtures.

Run beside digits-400.csv with Python, NumPy and CPU PyTorch. No downloads.
This trains small classifiers from scratch, not pretrained vision-language models.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def attention(query, key, value, allowed=None):
    scores = query @ key.T / math.sqrt(query.shape[-1])
    if allowed is not None:
        if not np.all(allowed.any(axis=-1)):
            raise ValueError("Every query must have at least one allowed memory slot")
        scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value, weights


class QuestionClassifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.query = nn.Embedding(2, 24)
        if mode == "flat":
            self.flat = nn.Sequential(nn.Linear(88, 48), nn.ReLU(), nn.Linear(48, 12))
        else:
            self.patch = nn.Linear(4, 24)
            self.position = nn.Parameter(torch.randn(1, 16, 24) * .02)
            self.read = nn.MultiheadAttention(24, 3, dropout=0, batch_first=True)
            self.feed = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
            self.head = nn.Linear(24, 12)
            if mode == "gated":
                self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, images, questions, capture=False):
        query = self.query(questions)
        if self.mode == "flat":
            return self.flat(torch.cat((images.flatten(1), query), dim=-1)), None
        patches = images.reshape(-1, 4, 2, 4, 2).permute(0, 1, 3, 2, 4).reshape(-1, 16, 4)
        memory = self.patch(patches) + self.position
        query = query[:, None, :]
        update, weights = self.read(query, memory, memory, need_weights=capture, average_attn_weights=False)
        if self.mode == "gated":
            update = self.gate.tanh() * update
        hidden = query + update
        hidden = hidden + self.feed(hidden)
        return self.head(hidden[:, 0]), weights


def main():
    with (HERE / "digits-400.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pixels = np.array([[int(row[f"pixel_{j}"]) for j in range(64)] for row in rows])
    labels = np.array([int(row["digit"]) for row in rows])
    source_ids = [int(row["source_id"]) for row in rows]
    # Preserve image groups when deriving two related questions per image.
    rng = np.random.default_rng(912)
    train, development, assessment = [], [], []
    for digit in range(10):
        order = rng.permutation(np.flatnonzero(labels == digit))
        train.extend(order[:24]); development.extend(order[24:32]); assessment.extend(order[32:])
    split = {"fit": sorted(train), "development": sorted(development), "assessment": sorted(assessment)}
    signatures = {}
    for i, row in enumerate(pixels):
        signatures.setdefault(tuple(row.tolist()), []).append(source_ids[i])
    duplicates = [ids for ids in signatures.values() if len(ids) > 1]
    assert len(rows) == 400 and len(set(source_ids)) == 400 and not duplicates
    images = torch.tensor(np.repeat(pixels.reshape(-1, 8, 8) / 16., 2, axis=0), dtype=torch.float32)
    questions = torch.tensor(np.tile([0, 1], 400))
    targets = torch.tensor(np.column_stack((labels, 10 + labels % 2)).reshape(-1))
    indices = {name: torch.tensor(np.array(ids)[:, None] * 2 + [0, 1]).flatten() for name, ids in split.items()}
    measured = []
    for mode in ("flat", "cross", "gated"):
        for seed in (11, 29, 47):
            torch.manual_seed(seed)
            model = QuestionClassifier(mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            history = []
            for epoch in range(160):
                model.train()
                for batch in indices["fit"][torch.randperm(len(indices["fit"]))].split(64):
                    logits, _ = model(images[batch], questions[batch])
                    loss = nn.functional.cross_entropy(logits, targets[batch])
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                if epoch in (0, 9, 39, 79, 159):
                    model.eval()
                    with torch.no_grad():
                        fit_logits, _ = model(images[indices["fit"]], questions[indices["fit"]])
                        dev_logits, _ = model(images[indices["development"]], questions[indices["development"]])
                        history.append({"epoch": epoch + 1, "fit_ce": nn.functional.cross_entropy(fit_logits, targets[indices["fit"]]).item(), "development_ce": nn.functional.cross_entropy(dev_logits, targets[indices["development"]]).item()})
            model.eval()
            record = {"mode": mode, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "history": history}
            with torch.no_grad():
                # Assessment is displayed evidence, so it is not a future untouched holdout.
                for role in ("development", "assessment"):
                    ids = indices[role]
                    logits, _ = model(images[ids], questions[ids])
                    correct = logits.argmax(-1) == targets[ids]
                    record[role] = {"correct": int(correct.sum()), "total": len(ids), "digit_correct": int(correct[::2].sum()), "parity_correct": int(correct[1::2].sum())}
                ids = indices["assessment"]
                grouped_images = images[ids].reshape(-1, 2, 8, 8)
                cyclic_images = grouped_images.roll(1, 0).reshape(-1, 8, 8)
                logits, _ = model(cyclic_images, questions[ids])
                record["cyclic_image_correct"] = int((logits.argmax(-1) == targets[ids]).sum())
                # Sorted-by-label rows make a one-place roll a weak mismatch test.
                permutation = np.random.default_rng(5129).permutation(len(grouped_images))
                wrong_images = grouped_images[permutation].reshape(-1, 8, 8)
                logits, _ = model(wrong_images, questions[ids])
                record["mismatched_image_correct"] = int((logits.argmax(-1) == targets[ids]).sum())
                if mode != "flat":
                    examples = indices["assessment"][:8]
                    logits, weights = model(images[examples], questions[examples], capture=True)
                    record["examples"] = [{"source_id": source_ids[int(i) // 2], "question": int(questions[i]), "target": int(targets[i]), "prediction": int(logits[j].argmax()), "head_weights": weights[j, :, 0].tolist()} for j, i in enumerate(examples)]
                if mode == "gated":
                    record["final_gate"] = float(model.gate.tanh())
            measured.append(record)
    q = np.array([[1., 0.], [0., 1.]]) * math.sqrt(2)
    k = np.array([[math.log(2), 0.], [0., math.log(2)], [0., 0.]])
    v = np.array([[2., 0.], [0., 4.], [2., 2.]])
    output, weights = attention(q, k, v)
    allowed = np.array([[True, True, False], [True, True, True]])
    masked, masked_weights = attention(q, k, v, allowed)
    changed = v.copy(); changed[2] = [100, -100]
    changed_output, _ = attention(q, k, changed, allowed)
    permutation = [2, 0, 1]
    permuted, _ = attention(q, k[permutation], v[permutation])
    assert np.allclose(output, [[1.5, 1.5], [1., 2.5]])
    assert np.allclose(masked[0], changed_output[0]) and np.allclose(output, permuted)
    assessment_labels = labels[split["assessment"]]
    shuffled_labels = assessment_labels[np.random.default_rng(5129).permutation(80)]
    perturbations = {"shuffle_source_ids": [source_ids[split["assessment"][i]] for i in np.random.default_rng(5129).permutation(80)], "shuffle_digit_matches": int((assessment_labels == shuffled_labels).sum()), "shuffle_parity_matches": int((assessment_labels % 2 == shuffled_labels % 2).sum()), "cyclic_digit_matches": int((assessment_labels == np.roll(assessment_labels, 1)).sum()), "cyclic_parity_matches": int((assessment_labels % 2 == np.roll(assessment_labels, 1) % 2).sum())}
    result = {"evidence": "Exact fixtures plus measured CPU classifier study; not a VLM benchmark", "versions": {"torch": torch.__version__, "numpy": np.__version__}, "splits": {role: [source_ids[i] for i in ids] for role, ids in split.items()}, "duplicates": duplicates, "perturbations": perturbations, "question_only_correct": {"development": 48, "assessment": 48, "total_each": 160}, "measurements": measured, "exact": {"query": q.tolist(), "key": k.tolist(), "value": v.tolist(), "weights": weights.tolist(), "output": output.tolist(), "masked_output": masked.tolist(), "masked_weights": masked_weights.tolist(), "changed_masked_output": changed_output.tolist()}}
    (HERE / "calculated-inputs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"mode": r["mode"], "seed": r["seed"], "assessment": r["assessment"], "mismatch": r["mismatched_image_correct"]} for r in measured], indent=2))


if __name__ == "__main__":
    main()
~~~

### Interpret the evidence

These are the actual final assessment counts from the author’s CPU run. The full result file also preserves development counts, parameter counts and loss traces.

| Model | Seed | Correct / 160 | Digit / 80 | Parity / 80 | Shuffled image / 160 |
| --- | --- | --- | --- | --- | --- |
| flat | 11 | 156 | 79 | 77 | 46 |
| flat | 29 | 156 | 78 | 78 | 47 |
| flat | 47 | 155 | 78 | 77 | 46 |
| cross | 11 | 140 | 67 | 73 | 48 |
| cross | 29 | 140 | 67 | 73 | 45 |
| cross | 47 | 142 | 67 | 75 | 48 |
| gated | 11 | 135 | 63 | 72 | 45 |
| gated | 29 | 140 | 70 | 70 | 47 |
| gated | 47 | 137 | 68 | 69 | 45 |

The flat model performs best in these fixed small fits: 155–156/160 versus 140–142/160 for ordinary cross-attention and 135–140/160 for gated cross-attention. This is useful evidence that learning a more elaborate read mechanism is not automatically helpful for this task and budget. The gated models do learn; a zero starting gate does not block them forever. Their learned tanh gates are -0.413, 0.583, 0.423 for seeds 11, 29 and 47. These values describe these fits, not a recommended universal gate target.

A deliberately weak control teaches another lesson. The assessment images are ordered in groups by digit. Moving each image one place cyclically leaves 70/80 digit labels and 70/80 parity labels unchanged. Flat-model scores remain 138–139/160; this does not establish that the models ignore images. A seeded shuffle is a stronger mismatch here: it preserves only 9/80 digit labels and 40/80 parity labels. The scores in the last column fall accordingly. Check what a perturbation changes before interpreting the model’s response.

The stronger image-mismatch diagnostic shuffles whole assessment images with a fixed seed while keeping the original questions and targets. A fall in performance supports reliance on the image–target relationship in this experiment. It is not a proof of compositional reasoning or an attention heatmap’s faithfulness. Some moved images can share labels, and the perturbation is a deliberate diagnostic rather than a natural deployment distribution.

For selected actual cases the program retains all three heads’16 patch weights, the source ID, question, target and predicted class. Show those weights next to the actual 8×8 image and a4×4 patch grid. No hand-labeled “person region” or “semantic latent” is inferred. A high weight says the value was strongly mixed in this read; residual pathways, value directions and the classifier also affect the answer.

**Investigation: which input matters?** Choose a recorded case, predict whether changing the question will preserve the numeric class, then compare the paired recorded output for that same image. Next compare aggregate correct/count for matched and mismatched images across all seeds. Use the exact small attention editor for arbitrary values; the recorded neural results do not pretend to recompute a trained model for unsupported edits.

## 7. Count the work and the cache you actually keep

This optional branch separates attention pair counts, arithmetic, stored state and elapsed time. They answer different questions.

For T text tokens and S visual tokens, a dense shared stream has \((T+S)^2\) candidate pairs per head before a causal mask. Text self-attention plus a cross-attention read has \(T^2+TS\) pairs for that pair of operations. If T=S, the ratio is 4T²/2T²=2, not 4. But this does not compare full models: the shared stream also updates visual tokens, cross-attention may be inserted less often, and both architectures include projection and feed-forward work.

Counting *permitted causal pairs* instead gives \((T+S)(T+S+1)/2\) for a strictly causal shared stream and \(T(T+1)/2+TS\) for causal text self-attention plus unrestricted visual reads. A dense implementation can still calculate masked pairs; a specialized kernel can avoid some work. Pair counts are not measured latency.

For one cross-attention head, the two matrix products cost approximately \(TSd_k+TSd_v\) multiply-accumulate operations. Projection costs depend on input and output widths. If one multiply and one add are counted as two FLOPs, state that convention. With R resampler reads and D latent self-attention blocks, their pair work scales as \(RMS+DM^2\), where M is latent count, before width factors and other layers. Input length still affects every repeated read.

### Visual features are not the whole cross-attention cache

During autoregressive generation, a prefix token’s projected keys and values are commonly cached at each language layer. With batch size Bₛ, P language layers, S visual prefix positions, total cached key width dₖᵥ per layer, and b bytes per scalar:

\[
C_{\text{visual prefix}}=2B_sPSd_{kv}b.
\]

For a separate visual memory read at J layers, caching each layer’s projected visual K/V instead costs

\[
C_{\text{visual cross}}=2B_sJS'd'_{kv}b,
\]

where S′ may be a compressed latent count. Raw encoder features might also be retained, or discarded after projection if no longer needed. Recomputing visual K/V saves their persistent storage but adds projection work. Cross-attention does not make projected K/V storage vanish.

Take a constructed serving configuration: Bₛ=1, P=32, J=8, S=576, S′=64, dₖᵥ=d′ₖᵥ=1024, b=2. Prefix visual K/V occupies 75,497,472 bytes, or 72 MiB. Cross visual K/V occupies 2,097,152 bytes, or 2 MiB. If both use 576 visual positions, the cross cache instead occupies 18 MiB. The difference combines layer count and token compression, not only attention type. These numbers exclude text K/V, weights, activations, temporary kernels and any raw-feature cache. Different GQA or MLA conventions change dₖᵥ and may require additional positional state; use their earlier lessons rather than assuming model width equals cached width.

**Visual: an explicit memory budget.** Let learners enter P, J, S, S′, key/value width and bytes. A stacked bar labels every declared tensor and its formula. A “cache projected K/V / recompute” choice changes storage and flags the extra operation, without inventing milliseconds. Ask for a predicted byte difference first. Use binary MiB consistently, with raw bytes available.

## 8. Choose a design through its failure cases

Start from the decision the model must make. Reading small text in a scanned form may require preserving local high-resolution features. Answering broad questions about a short clip may tolerate stronger compression. Neither determines an architecture by itself; it identifies the errors and budgets to measure.

| Symptom | First useful check | Why it helps |
| --- | --- | --- |
| Answers hardly change when images change | Matched versus mismatched images, then a question-only baseline | Detects reliance on language priors or question shortcuts |
| Training loss is suspiciously tiny | Trace shifted targets, attention availability and supervised positions | Target leakage can look like rapid learning |
| Long documents exceed memory | Count actual crops/tokens and per-layer cached tensors | Image count alone hides resolution, pooling and layer effects |
| A new adapter never updates | Inspect gate and adapter gradients separately | Zero gate may block adapter gradients while allowing gate learning |
| An adapter receives no gradient through a frozen backbone | Inspect differentiation mode and input gradients | Frozen parameters must still transmit the derivative needed upstream |
| Fine details disappear after compression | Compare task-specific errors at controlled latent budgets | Global accuracy can hide OCR or counting losses |
| A replaced vision encoder breaks answers | Check normalization, selected layer, width and representation compatibility | Equal tensor shape does not imply the same learned feature semantics |
| A model attends to unavailable future frames | Trace the encoder and memory construction, not only final masks | Information can leak before the final attention layer |

A useful application beyond image chat is **query-based output construction**. In a set-prediction detector, learned object queries can request a fixed collection of candidate outputs from image memory; in a structured decoder, location queries can ask for outputs at particular coordinates. The query determines the output slot’s role, while the memory need not have the same number of slots. Learning distinct useful queries and matching predictions to targets are separate design problems, not properties granted by the matrix multiplication.

For video, ask separately how frames are sampled, how time is represented, which frames are available at each prediction, and whether one latent bank summarizes a whole clip or each frame. “Sixteen frames” does not identify a token budget. For document analysis, several crops can represent one page; an apparent small image count can still create a long sequence. For serving repeated questions about a fixed image, caching unchanged visual representations can help, but question-conditioned resampling and model/preprocessing changes invalidate some caches.

Production work should begin with one fully inspected example: exact preprocessing, image-token counts, masks, supervised targets and forward/backward shapes. Then check a representative held-out task set, corruption and mismatch behavior, and measured memory/latency under the intended device and batch. Quantization, batching and kernels deserve their own measured evaluation. None of the tiny classifier scores above identifies a universally best production VLM family.

## 9. Practice, hints and solutions

### 1. A changed rectangular read — core

Use first-query weights (1/2,1/4,1/4), but change the values to (4,0),(0,2),(2,6). Compute the output. Then forbid slot one and recompute from the original scores, not by retaining weights that sum to 1/2.

<details><summary>Hint</summary>
After forbidding the score log 2, the two remaining scores are both zero.
</details>
<details><summary>Solution</summary>
The original output is (2.5,2). After masking slot one, slots two and three each receive 1/2, giving (1,4). Dropping a term without renormalizing would incorrectly give (.5,2).
</details>

### 2. Repair the shape contract — core

A reader has 7 positions of width 48, memory has 11 positions of width 80, and there are 4 heads each with key width 12 and value width 8. State the weight shape per example, concatenated output shape, and final projection needed for a residual addition.

<details><summary>Hint</summary>
The value width determines a head’s output width; the query count determines its length.
</details>
<details><summary>Solution</summary>
Weights 4×7×11; head outputs 7×8; concatenation 7×32; an output projection 32→48 gives 7×48 to add to the reader. Project memory 80→12 for keys and 80→8 for values per head. Nothing requires replacing the 11 memory positions by 7.
</details>

### 3. Two masks, two purposes — core

An answer is “three birds”. Draw the information available when predicting “three” and “birds”. A colleague removes the question from the loss but uses bidirectional attention over the entire answer. Explain the remaining error and repair it.

<details><summary>Hint</summary>
Ask whether a prediction can already read its target or future answer.
</details>
<details><summary>Solution</summary>
The first target uses image, question and answer-start state; the second additionally uses “three”. Excluding question positions from supervision does not hide future answer tokens. Shift inputs/targets and use the appropriate causal answer access mask while keeping image/question context available. Verify the encoder or packed examples do not introduce another leakage route.
</details>

### 4. A gate that can and cannot start — core

Set X=2, F=−1, target 0 and squared loss ½Y². At α=0 find the loss and gate gradient. What happens after one gradient step of size.1? What would change if F=0?

<details><summary>Hint</summary>
At zero, sech²α=1; the gate derivative is residual error times F.
</details>
<details><summary>Solution</summary>
Y=2, loss 2, gate gradient−2. The update gives α=.2, so Y=2−tanh (.2)≈1.8026 and loss≈1.6247. If F=0, the gate gradient is zero and this route does not open on that step. A separate training path could still update F; do not assume one exists.
</details>

### 5. Design a valid image-use test — core

Your model scores 92% on questions derived from images, but the training split was made after creating multiple questions per image. Design the corrected split, two baselines or diagnostics, and the conclusion you would avoid even if the model still scores 92%.

<details><summary>Hint</summary>
The unit that must stay together is the underlying observation, not the question row.
</details>
<details><summary>Solution</summary>
Group all variants of an image before splitting; also group known near-duplicates or shared acquisition units when required. Fit all learned preprocessing only on fitting data. Compare question-only and a simple image-plus-question model; inspect mismatched images as a diagnostic. Report question-type counts and errors. High accuracy on this task does not establish general visual reasoning, new-writer generalization or a faithful attention explanation.
</details>

### 6. A cache comparison — deeper

One request has 256 visual tokens,24 language layers,6 visual-read layers, cached K/V width 512 and 2-byte scalars. Compare visual prefix storage with cross-attention storage when both keep all 256 tokens, and when cross-attention uses 32 latents. Exclude raw features and text caches explicitly.

<details><summary>Hint</summary>
Count both K and V at every layer where they are stored.
</details>
<details><summary>Solution</summary>
Prefix:2×24×256×512×2=12,582,912 bytes=12 MiB. Cross with 256:3 MiB. Cross with 32:.375 MiB. The first ratio comes from layer count; the second adds 8-fold compression. This arithmetic does not say which architecture gives adequate answers or faster wall-clock execution.
</details>

### 7. Can later attention recover a missing detail? — deeper

Two memories contain scalar values (0,4) and (1,3). A compressor stores only their unweighted mean. Construct a question for which the compressed memories are insufficient, and describe one modification that preserves the needed information.

<details><summary>Hint</summary>
Ask for a statistic that differs despite equal means.
</details>
<details><summary>Solution</summary>
“What is the maximum?” distinguishes 4 from 3; both stored means are 2. Any deterministic downstream reader receiving only that identical state must give the same answer. Preserve the two slots, or store a sufficient representation for the specified question, such as maximum as an additional feature. Increasing downstream depth alone cannot reconstruct a distinction erased from its only input.
</details>

### 8. Read a model claim critically — deeper

A report says “our model uses interleaved inputs, so it contains no cross-attention; its gate starts at zero, so it cannot forget language; its heatmap proves which image object caused the answer.” Rewrite those three claims into testable statements.

<details><summary>Hint</summary>
Separate input format, initialization and attribution evidence.
</details>
<details><summary>Solution</summary>
Specify the actual fusion blocks and masks independently of input ordering. Zero initialization makes the inserted residual identity under the stated conditions at initialization; later language retention needs measurement. Attention weights describe mixing in one operation; causal attribution requires additional interventions and analysis of alternative paths. Neither a model name nor a visually sharp row settles those questions.
</details>

## 10. What to read or watch next

- [Flamingo’s primary paper](https://arxiv.org/html/2204.14198): read §2 and AppendixA.1 for visual processing, gated insertion and image masking. Its experiments are historical results under the paper’s training setup. The appendix is particularly useful for seeing how the resampler differs from an ordinary cross-attention block.
- [BLIP-2’s primary paper](https://proceedings.mlr.press/v202/li23q/li23q.pdf): start with §§3.1–3.3 and Figures 2–3 to follow mask choices and the two-stage connection. Use it when the training objective is the confusing part of the architecture.
- [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485): §§4.1–4.2 connect an intentionally simple projection to supervised response tokens. Useful alongside the answer-shift exercise.
- [What matters when building vision-language models?](https://arxiv.org/pdf/2405.02246): §§2–3 distinguish input fusion, pooling, backbone adaptation and evaluation. Read the experimental controls before interpreting an architecture ranking.
- [Samuel Albanie’s Flamingo video digest and slides](https://samuelalbanie.com/digests/2022-05-flamingo/): an alternate spoken/visual route into the architecture. The creator page and its material links were checked; this packet does not claim the full video was watched. Compare the illustrated routes with the paper after completing §3.
- [PyTorch2.14 MultiheadAttention documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html): consult query/key/value shapes, batch conventions, masks and returned head weights when modifying the program. API mask conventions must be checked rather than inferred from the word “mask”.

You are ready to continue when you can follow a query through scores, masking, values, residual update and supervised target, and explain one failure that each step can introduce. The next module topic is [Message Passing & Graph Convolutions](/learn/path/full-curriculum/message-passing-graph-convolutions-gcn-gat-graphsage?module=deep-learning-fundamentals). It extends the idea of gathering information from selected neighbors: the neighbor structure becomes an explicit graph, and the questions become which edges are available, how messages are combined and what should stay unchanged when nodes are relabeled.
