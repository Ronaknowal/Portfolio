# Sequence-to-Sequence Encoder–Decoder: Read an Input, Generate an Output

**Explore as you read.** Edit source/target shifts, bridge weights/rate, tiny probability trees, beam width and supported fitted source/prefix inputs. Show aligned timelines, dependency paths, sequence probabilities and bounded beam candidates live. Keep teacher-forced versus generated inputs explicit at every step. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to distinguish model probability from a decoding decision and identify when a prefix or alignment changes the actual task.


A handwriting classifier reads several pen positions and chooses one digit. Now change the request: read a word and generate its past tense. The input `walk` has four characters; `walked` has six. `eat` becomes `ate`, so copying the input and appending a suffix is not enough. The system must decide both **what comes next** and **when the answer is finished**.

A sequence-to-sequence model learns a mapping from an ordered input to an ordered output. An **encoder–decoder** is one way to build it: an encoder turns the input into a learned representation; a decoder uses that representation to produce the output. Here we will build a small recurrent version, inspect its actual states, and discover why fitting the training examples does not mean it has learned a reusable spelling rule.

**First pass:** follow sections 1–6 to understand the two networks, the shifted training targets, generation and a real experiment. Section7 supplies the complete CPU program. Use section 8 to diagnose it, then try the practice. The deeper branches in section 9 are optional on a first reading; they explain architectural variants, training objectives and production contracts.

You need the idea that a recurrent state is an updated vector and that cross-entropy rewards probability placed on the correct outcome. We refresh both below. [RNNs, LSTMs and GRUs](/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals) provides the full cell equations.

## 1. Two timelines, one conditional task

Imagine a language-learning tool that is given a **lemma**, the dictionary form of a word, and a grammatical request:

| Input | Requested form | Output |
|---|---|---|
| `walk` | past | `walked` |
| `try` | past | `tried` |
| `make` | present participle | `making` |
| `eat` | past | `ate` |

The grammatical request is part of the input. Without it, the same word could legitimately require several answers. A training example therefore includes the source information and the desired output; the network cannot infer an omitted task specification merely from being large.

Our running data example is `lactate + past → lactated`, a real entry in the supplied UniMorph extract. A **token** is one item the network processes. In this lesson characters are tokens, and the grammatical request is a separate token:

`<past> → l → a → c → t → a → t → e → <eos>`

The encoder processes those nine source tokens. The decoder starts a different timeline:

`<bos> → predict l → predict a → … → predict d → predict <eos>`

`<bos>` means “begin generating.” `<eos>` is an actual predicted outcome meaning “end the output.” The source also has its own end marker. The two end markers share an ID in our program, but occupy different sequences. Neither is a letter in the word.

**Visual: two token tracks.** Put the source above the generated answer, with a narrow state bridge between encoder and decoder. Show the source cursor finish before the first decoder step. Do not connect every source character to a same-numbered output character: lengths and useful correspondences can differ.

This is different from labeling every source token. A tagger might assign one label to each word; this decoder is free to output a different number of tokens. Translation, speech transcription with an autoregressive decoder, spelling normalization and generation of structured text can use the same broad input/output contract. Their token choices, valid outputs and error costs differ. A recurrent encoder–decoder is not the only architecture capable of these tasks.

An interesting practical connection is **inflection for language tools**. A dictionary assistant may need hundreds of forms of a lemma. A general learned mapping can share patterns across examples, while a rule system can directly encode predictable changes. Neither approach makes dictionary exceptions disappear. Our experiment keeps the rule system in the comparison rather than assuming the neural network should replace it.

## 2. Give every token and tensor a job

The model cannot multiply the string `"a"` by a matrix. We assign each token an integer ID, then use an **embedding table**: a learned row vector for each ID. Looking up row 17 does not claim that token 17 is “larger” than token 8. The number is an address.

Our vocabulary has 32 entries: 26 lowercase letters, three grammatical-request tokens, and `<pad>`/`<bos>`/`<eos>`. The input contract restricts this small experiment to lowercase English spellings. A general text system must decide how it handles other scripts, case, punctuation, unknown tokens and normalization. Word, subword and byte tokenization are separate choices; encoder–decoder learning does not require BPE.

`<pad>` fills unused cells when examples of different lengths share a batch. It is storage, not a requested prediction. We use two distinct mechanisms:

1. **Source lengths:** packing tells the recurrent encoder which source positions exist. The final state corresponds to the last real source token, including its EOS.
2. **Target mask:** the loss ignores target PAD positions. Output EOS is real and remains in the loss.

The decoder can emit only letters or EOS in this task. Before softmax, the program masks PAD, BOS and request-token logits to negative infinity. Their output probabilities become zero. This output-support rule is applied consistently during training and generation; it is not a late cosmetic cleanup of bad strings.

For a batch of `B` examples, longest source length `S` and longest target length `T`:

| Object | Shape in the program | Meaning |
|---|---|---|
| Source IDs | `B × S` | Request, letters, EOS, then padding |
| Source embeddings | `B × S × 24` | Learned input vectors |
| Encoder final state | `1 × B × 64` | One GRU layer, batch, state coordinates |
| Decoder input IDs | `B × T` | BOS followed by the known target prefix during training |
| Decoder states | `B × T × 64` | One state for each predicted output position |
| Output logits | `B × T × 32` | Scores before softmax; invalid outputs masked |
| Target IDs | `B × T` | Desired letters followed by EOS, then padding |

For target `ate` the alignment is:

| Prediction step | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Decoder receives | BOS | a | t | e |
| Correct output | a | t | e | EOS |

At step 2 the decoder receives `a` because it is supposed to predict the token **after** `a`. Feeding `t` at that same step would reveal the answer being scored. Forgetting the shift can produce an impressively small loss for the wrong task.

**Investigation: repair the token tracks.** On a fresh `care + past → cared` example, arrange the decoder input and output cards, including EOS and padding. Then edit the target to a different constructed string and identify the live comparison position whose input can change. The input cards are actual tokens used by the calculation, not a multiple-choice illustration of a prewritten answer.

## 3. How the encoder and decoder communicate

Let $x_1,\ldots,x_S$ be source token IDs and $E[x_i]$ their embedding vectors. The encoder updates a state:

$$h_i=\operatorname{GRU}_{enc}(E[x_i],h_{i-1}),\qquad h_0=0.$$

Its final state $c=h_S$ is the **context**. In our basic architecture the decoder begins with $s_0=c$:

$$s_t=\operatorname{GRU}_{dec}(E[y_{t-1}],s_{t-1}),\qquad y_0=\mathrm{BOS},$$
$$z_t=W_{out}s_t+b_{out},\qquad p_t=\operatorname{softmax}(z_t).$$

The two GRUs have different learned parameters. “Both use a GRU” means they use the same kind of calculation, not the same weights. The embedding table happens to be shared because this task uses the same letters on both sides; separate source and target embeddings are also possible.

The decoder does not receive the raw spelling again in this version. After initialization, all source influence must travel through its evolving state. If we replace one input's context with another's while keeping the decoder and BOS fixed, it generates from that other context. This is a useful, testable meaning of “the context conditions the output.”

### A small complete forward calculation

To make the numbers inspectable, temporarily replace the 64-coordinate GRUs with scalar tanh updates. This is a constructed mechanism example, not the fitted inflector:

$$h_i=\tanh(0.7x_i+0.4h_{i-1}+0.1),\quad h_0=0,\quad x=[0.2,0.8].$$

The first state is $\tanh(0.24)=0.235496$. The second is $\tanh(0.7(0.8)+0.4(0.235496)+0.1)=0.637647$. That second state becomes the decoder's initial state.

Use decoder update $s_t=\tanh(0.6e_{t-1}+0.5s_{t-1}+0.05)$, with BOS embedding 0.1 and token A embedding 0.4. At each step the two output scores are $[s_t,-s_t]$, for A and EOS respectively.

| Step | Decoder input | New state | P(A) | P(EOS) | Desired output |
|---|---|---:|---:|---:|---|
| 1 | BOS embedding 0.1 | 0.404338 | 0.691827 | 0.308173 | A |
| 2 | A embedding 0.4 | 0.455936 | 0.713383 | 0.286617 | EOS |

The model starts the answer reasonably but assigns too little probability to ending it. A state can look numerically stable while its prediction is wrong. A large norm or a smooth heatmap is not evidence of understanding.

**Visual: state bridge with a numerical expansion.** Keep source states, context, decoder states and probability bars synchronized. Label raw hidden coordinates as coordinates. A fitted coordinate is not automatically a “tense neuron” or a “word-length accumulator.”

## 4. Train the probability of the whole answer

For a particular output $y_1,\ldots,y_T$, including its final EOS, the autoregressive model assigns:

$$P_\theta(y\mid x)=\prod_{t=1}^{T}P_\theta(y_t\mid y_{<t},x).$$

“Autoregressive” means a prediction depends on previous output tokens. The product is the probability of this entire route through the decoder. Taking negative logarithms turns the product into a sum:

$$\mathcal L_{\text{sequence}}=-\sum_{t=1}^{T}\log P_\theta(y_t\mid y_{<t},x).$$

In the scalar example, the correct-token probabilities are 0.691827 and 0.286617. Their negative-log costs are 0.368419 and 1.249609. The mean loss is 0.809014 natural-log units, or **nats**, per target token. EOS contributes most of the error here. Removing EOS from the targets would remove the direct lesson “finish after A.”

Across a padded batch, our objective is the sum of valid-token costs divided by the **number of valid target tokens**:

$$\mathcal L=\frac{\sum_{b,t}m_{bt}[-\log p_{bt}(y_{bt})]}{\sum_{b,t}m_{bt}},\quad
m_{bt}=1\ \text{when the target is not PAD}.$$

This gives equal weight to valid tokens. Averaging each sequence first would give equal weight to sequences and therefore relatively more weight to tokens in short answers. Both can be deliberate objectives; changing the denominator silently changes the training problem.

### Teacher forcing: a known prefix, not the current answer

During training we know the reference output. **Teacher forcing** uses its previous tokens as the decoder inputs while scoring each next token. This directly evaluates the conditional factors in the likelihood above. It is ordinary maximum-likelihood training for this model, not a trick that makes the loss invalid.

The GRU states still depend on earlier states. Providing all known input tokens permits a convenient batched call to `nn.GRU`, but it does not make recurrent time steps mathematically independent. The reference token at step 5 cannot determine state5 without the recurrent history.

At inference the reference answer is unavailable. The decoder must use a chosen or sampled previous output. If it makes a mistake, subsequent states may follow a prefix poorly represented in training. This is a reason to measure complete generated answers as well as teacher-forced loss. It does not imply that every initial error causes an irreversible cascade.

### One gradient reaches both networks

Backpropagation starts at the output losses, passes through decoder states and the initial context, then through the encoder. In the scalar example, hold the decoder fixed and differentiate with respect to the encoder's input weight 0.7. The calculated derivative is −0.00556958. A gradient-descent step of size 0.1 changes it to 0.70055696 and reduces mean loss to 0.80901091. The change is small, but it demonstrates the important dependency: an output loss can teach the encoder how to represent its input.

The supplied calculation checks that derivative against a central finite difference. In the real model, a four-example batch gives a nonzero encoder input-weight gradient norm of 0.169993. Detaching the context gives the same forward computation but removes this decoder-to-encoder gradient path. That would defeat joint learning unless a separate encoder objective were intended.

<details>
<summary>Follow the scalar derivative through the bridge</summary>

For the two-token mean loss, the direct derivative at decoder step 1 is $p_1(A)-1$, and at step 2 it is $p_2(A)$ because EOS is correct there. Step2 also sends credit back through step 1:

$$\frac{\partial\mathcal L}{\partial s_1}
=(p_1(A)-1)+p_2(A)\,0.5(1-s_2^2).$$

Multiply this by $0.5(1-s_1^2)$ to cross the decoder-initial-state edge into context $c$. The encoder input weight is reused at both source positions, giving:

$$\frac{\partial c}{\partial w}
=(1-h_2^2)\left[x_2+0.4(1-h_1^2)x_1\right].$$

The product is −0.00556958. The first term in brackets is the direct contribution at source position 2; the second comes through the earlier encoder state. Both belong to the same parameter.

</details>

Return to [Backpropagation and Automatic Differentiation](/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals) for the full graph mechanics. Here the new point is where the two graphs meet.

## 5. Generate an answer, then search more than one route

**Greedy decoding** chooses the largest next-token probability. It starts from BOS, updates the state, emits one token, and feeds that token into the next step. It stops on EOS or an explicit output limit. Hitting the limit means “generation was capped,” not “the model chose to finish.”

For our fitted seed 1 model, the training input `lactate + past` generates `lactated` and EOS. Each emitted token has a probability conditional on that particular input and generated prefix. Multiplying those probabilities does not produce “probability this spelling is linguistically correct.” It is the model's probability for that route.

### Why the locally best token can lose

Consider this completely specified constructed tree. First choose A with probability 0.60 or B with 0.40. After A, choose EOS with 0.51 or C with 0.49. After B, choose EOS with 0.90 or C with 0.10. After C, EOS is mandatory.

| Complete answer | Product | Probability |
|---|---|---:|
| A, EOS | 0.60 × 0.51 | 0.306 |
| A, C, EOS | 0.60 × 0.49 × 1 | 0.294 |
| B, EOS | 0.40 × 0.90 | 0.360 |
| B, C, EOS | 0.40 × 0.10 × 1 | 0.040 |

Greedy commits to A, then EOS: 0.306. The highest-probability complete answer is B, EOS: 0.360. The first decision can look best until we consider what follows it.

**Beam search** retains several partial answers. With width 2, keep A and B after the first step. Expand each live candidate, add its next-token log probability to its accumulated log score, then retain the two highest-scoring candidates. In this tree those are B,EOS and A,EOS. A finished candidate is retained without appending more tokens.

Each candidate owns its prefix **and its decoder state**. Reusing the state from the wrong candidate makes the next distribution wrong even if the displayed strings look right. A beam is not simply a list of alternative final words from one shared state.

Our program keeps completed and live candidates in the same width-limited list, sorts tied scores by token IDs, and stops when every retained candidate is complete or 16 generated tokens have been reached. At a cap it returns the highest-ranked remaining candidate with an explicit termination flag. Width1 matches the program's greedy algorithm under the same support, tie order and stopping convention.

The search remains approximate: a promising route can be pruned before its good continuation appears. A larger beam also optimizes the model's score more thoroughly, which need not improve the task metric when the model is wrong. The real experiment below measures the effect instead of claiming a guaranteed improvement.

### Length normalization changes the objective

Raw log probability is a sum of nonpositive terms. Extending a particular prefix cannot increase that raw probability. Comparing completed answers of different lengths can therefore require an explicit length policy.

One common score is:

$$\operatorname{score}(y)=\frac{\log P(y\mid x)}
{\left((5+L)/6\right)^\alpha},$$

where our definition of $L$ counts generated tokens including EOS and excludes BOS. This is the length term from the GNMT family of scoring rules; its separate coverage term requires attention and is not used here. The exponent is applied **once**. It is a ranking heuristic, not a normalized probability distribution. [GNMT, section 7](https://arxiv.org/abs/1609.08144).

For a length 3 answer with log probability−1.5 and a length 6 answer with −1.8, raw scoring prefers the shorter answer. At $\alpha=1$, their scores are −1.125 and−0.981818, so the longer answer wins. Dividing a negative number by a larger positive denominator makes it less negative. Calling this universally a penalty against long outputs reverses the actual effect.

The experiment fixes $\alpha=0$. A nonzero setting must be chosen on development data and reported with the length convention. Do not borrow a number from a translation paper as a universal spelling-model setting.

**Investigation: edit a probability tree.** Start with A .55/B .45 and the fresh continuation probabilities. Edit a branch probability and watch the complete winner, its probability and the retained beam candidates update. Each conditional distribution stays normalized. Step the search to inspect both retained and pruned paths; this is an exact small search problem, not a translation benchmark.

## 6. A real experiment: learning a function is harder than remembering pairs

The offline extract comes from the [UniMorph English repository](https://github.com/unimorph/eng), pinned to a specific source revision. It contains 600 lowercase lemma spellings, each with a past, present-participle and third-person-singular-present form. The source includes uncommon and historical spellings; it is a lexicon sample, not the frequency distribution of everyday English. Its data license and attribution travel with the extract.

The intended question is: **can this model inflect a spelling absent from training?** All three requests for one lemma stay together. We also group selected lemmas linked by a shared output form: `worke` and `work` share `worked` and `working`. Treating their strings as unrelated would give a misleadingly clean “no overlap” report. This conservative grouping can merge genuine homographs; it is an operational split policy, not a complete linguistic identity system.

The final split has 451 training lemmas/1,353 examples and 149 development lemmas/447 examples. There is no shared lemma string or target form across these partitions. No unseen final test is claimed: we inspect development outcomes to learn about the system. Future tuning would consume more development information and require a separately protected evaluation for a final performance claim.

The experiment fixes a 24-coordinate embedding, one 64-coordinate GRU encoder, one 64-coordinate GRU decoder and a 32-score output head: 37,408 trainable parameters. Each of three seeds receives 1,200 Adam updates at learning rate 0.003, batches of 64 sampled training rows, global gradient clipping at 1, teacher forcing and the same token/split rules. We show the final predeclared update, not the best-looking development checkpoint.

The comparison includes two non-neural systems. **Copy** returns the lemma unchanged. **Predeclared suffix rules** append or replace endings such as `y→ied` and `e→ing`; the exact rules are in the program. They omit irregular dictionaries and some spelling conditions, including consonant doubling. Their prior knowledge is explicit.

For generated strings, **exact match** requires the reference spelling and natural EOS termination. **Character error rate** is total Levenshtein insertions, deletions and substitutions divided by total reference characters. A rate above 1 is possible when a generator inserts many characters. Teacher-forced NLL is measured separately, including EOS and excluding PAD.

| System | Training exact / 1,353 | Development exact / 447 | Development character error rate | Development teacher-forced NLL |
|---|---:|---:|---:|---:|
| Copy lemma | 3 | 1 | 0.255014 | Not a probabilistic model |
| Predeclared suffix rules | 1206 | 407 | 0.016332 | Not a probabilistic model |
| GRU encoder–decoder, seed 1 | 1328 | 53 | 0.434670 | 1.285388 |
| GRU encoder–decoder, seed 2 | 1323 | 41 | 0.439255 | 1.455556 |
| GRU encoder–decoder, seed 3 | 1287 | 52 | 0.422636 | 1.232162 |

All final neural development generations ended with EOS within the limit. These are actual CPU measurements on the declared grouped split.

The neural system can memorize training pairs yet fail badly on new lemmas. Most of this task's characters should be copied from the input. A fixed context forces the model to learn a representation and a decoder that preserve this detail; the hand-written rules already preserve it by construction. The comparison is informative even though the neural model loses.

Do not conclude that all encoder–decoders are poor inflectors. This experiment has a small training lexicon, a particular architecture, initialization, optimizer and budget. It establishes this result for this protocol. Increasing capacity alone might improve memorization without solving the held-out problem.

Seed 1's development slices are 44/168 exact for lemma lengths 3–5 and 9/279 for lengths 6–8. Longer spellings are harder here, but length is entangled with which words occur, spelling patterns and number of required copy operations. This is not a causal experiment proving that a fixed context has a universal character limit.

The same seed's beam 3 search returns 56/447 exact rather than greedy's 53/447; character error rate changes from 0.434670 to 0.410888. No additional training occurred. Search recovers some better-scoring routes, but it cannot supply missing linguistic knowledge or undo the large generalization gap.

**Visual: actual learning curves and paired outputs.** Plot the recorded update checkpoints, label training versus development and teacher-forced NLL versus generated exact match distinctly. Keep actual denominators visible. The curves are measured from this program; no smooth invented continuation or hardware-speed ranking is needed.

## 7. Run the complete small model

Save `english-inflections.csv` beside `inflection-seq2seq.py`. A CPU Python environment with NumPy and PyTorch is sufficient; no pretrained model or runtime data download is used. The recorded run used Python 3.12.14, NumPy 2.3.5 and PyTorch 2.14.0+cpu. If those packages are absent, create a local virtual environment and install NumPy and the CPU build of PyTorch appropriate to your operating system from its official installer.

Run `python inflection-seq2seq.py`. The program prints checkpoint and final metrics and writes `calculated-inputs.json` with predictions, weights and the protocol. Seeds make the experiment reproducible in the recorded environment; another backend/version may produce small numerical differences.

Read the program in this order: `batch` constructs the shifted tracks; `Inflector` connects the states; `greedy` performs free generation; `beam` owns each candidate state; `summarize` separates exact match, edits and termination; `main` applies the fixed training protocol. `assess` uses no gradients. `model.eval()` chooses evaluation behavior, whereas `no_grad()` suppresses gradient recording.

<details>
<summary>Complete runnable program</summary>

```python
"""Complete CPU character encoder-decoder on a fixed, lemma-disjoint real extract."""
from pathlib import Path
import csv
import json
import platform
import string
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parent
TOKENS = ["<pad>", "<bos>", "<eos>", "<past>", "<participle>", "<third_person>"] + list(string.ascii_lowercase)
INDEX = {token: index for index, token in enumerate(TOKENS)}
PAD, BOS, EOS = 0, 1, 2
ALLOWED = [EOS] + list(range(6, len(TOKENS)))
MAX_OUTPUT = 16  # Generated tokens, including EOS; BOS is not counted.


def load_records():
    with (ROOT/"english-inflections.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    train = [row for row in records if row["partition"] == "train"]
    development = [row for row in records if row["partition"] == "development"]
    assert len(train) == 1353 and len(development) == 447
    assert not set(row["lemma"] for row in train) & set(row["lemma"] for row in development)
    assert not set(row["form"] for row in train) & set(row["form"] for row in development)
    return train, development


def source_ids(lemma, feature):
    if not lemma or any(letter not in string.ascii_lowercase for letter in lemma):
        raise ValueError("Use a nonempty lower-case a-z lemma.")
    return [INDEX[f"<{feature}>"]] + [INDEX[letter] for letter in lemma] + [EOS]


def source_batch(records):
    sources = [source_ids(row["lemma"], row["feature"]) for row in records]
    source = torch.full((len(records), max(map(len, sources))), PAD, dtype=torch.long)
    for index, ids in enumerate(sources):
        source[index, :len(ids)] = torch.tensor(ids)
    return source, torch.tensor(list(map(len, sources)), dtype=torch.long)


def batch(records):
    source, lengths = source_batch(records)
    targets = [[INDEX[letter] for letter in row["form"]] + [EOS] for row in records]
    target = torch.full((len(records), max(map(len, targets))), PAD, dtype=torch.long)
    for index, y in enumerate(targets):
        target[index, :len(y)] = torch.tensor(y)
    decoder_input = torch.cat([torch.full((len(records), 1), BOS), target[:, :-1]], dim=1)
    return source, lengths, decoder_input, target


class Inflector(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(TOKENS), 24, padding_idx=PAD)
        self.encoder = nn.GRU(24, 64, batch_first=True)
        self.decoder = nn.GRU(24, 64, batch_first=True)
        self.readout = nn.Linear(64, len(TOKENS))
        invalid = torch.ones(len(TOKENS), dtype=torch.bool)
        invalid[ALLOWED] = False
        self.register_buffer("invalid_output", invalid)

    def encode(self, source, lengths):
        packed = pack_padded_sequence(self.embedding(source), lengths, batch_first=True, enforce_sorted=False)
        _, final = self.encoder(packed)
        return final

    def output_logits(self, hidden):
        return self.readout(hidden).masked_fill(self.invalid_output, -torch.inf)

    def forward(self, source, lengths, decoder_input):
        state = self.encode(source, lengths)
        hidden, _ = self.decoder(self.embedding(decoder_input), state)
        return self.output_logits(hidden)

    def step(self, previous_token, state):
        hidden, state = self.decoder(self.embedding(previous_token[:, None]), state)
        return self.output_logits(hidden[:, 0]), state


def edit_distance(left, right):
    previous = list(range(len(right)+1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[column]+1, previous[column-1]+(a != b)))
        previous = current
    return previous[-1]


def suffix_baseline(lemma, feature):
    consonant_y = lemma.endswith("y") and lemma[-2] not in "aeiou"
    if feature == "past":
        return lemma[:-1]+"ied" if consonant_y else lemma+"d" if lemma.endswith("e") else lemma+"ed"
    if feature == "participle":
        if lemma.endswith("ie"):
            return lemma[:-2]+"ying"
        return lemma[:-1]+"ing" if lemma.endswith("e") and not lemma.endswith("ee") else lemma+"ing"
    if consonant_y:
        return lemma[:-1]+"ies"
    return lemma+"es" if lemma.endswith(("s", "x", "z", "ch", "sh")) else lemma+"s"


@torch.no_grad()
def greedy(model, records, max_output=MAX_OUTPUT):
    model.eval()
    source, lengths = source_batch(records)
    state = model.encode(source, lengths)
    previous = torch.full((len(records),), BOS, dtype=torch.long)
    finished = torch.zeros(len(records), dtype=torch.bool)
    outputs, traces = [[] for _ in records], [[] for _ in records]
    for _ in range(max_output):
        logits, state = model.step(previous, state)
        log_probabilities = logits.log_softmax(-1)
        next_token = log_probabilities.argmax(-1)
        for index in range(len(records)):
            if not finished[index]:
                token = int(next_token[index])
                outputs[index].append(token)
                traces[index].append(float(log_probabilities[index, token]))
        finished |= next_token == EOS
        previous = torch.where(finished, EOS, next_token)
        if bool(finished.all()):
            break
    return [{"prediction": "".join(TOKENS[token] for token in output if token != EOS),
             "tokens": output, "ended_with_eos": bool(output and output[-1] == EOS),
             "log_probability": sum(logs), "token_log_probabilities": logs}
            for output, logs in zip(outputs, traces)]


@torch.no_grad()
def beam(model, record, width=3, alpha=0., max_output=MAX_OUTPUT):
    model.eval()
    source, lengths = source_batch([record])
    # Candidate owns tokens, recurrent state after consuming its previous input, and raw logP.
    candidates = [([], model.encode(source, lengths), 0.)]
    def score(item):
        ids, _, log_probability = item
        return log_probability / (((5+len(ids))/6)**alpha)
    for _ in range(max_output):
        expanded = []
        for ids, state, log_probability in candidates:
            if ids and ids[-1] == EOS:
                expanded.append((ids, state, log_probability))
                continue
            previous = torch.tensor([ids[-1] if ids else BOS])
            logits, new_state = model.step(previous, state)
            values = logits.log_softmax(-1)[0]
            for token in ALLOWED:
                expanded.append((ids+[token], new_state, log_probability+float(values[token])))
        candidates = sorted(expanded, key=lambda item: (-score(item), item[0]))[:width]
        if all(ids and ids[-1] == EOS for ids, _, _ in candidates):
            break
    ids, _, log_probability = candidates[0]
    return {"prediction": "".join(TOKENS[token] for token in ids if token != EOS), "tokens": ids,
            "ended_with_eos": ids[-1] == EOS, "log_probability": log_probability,
            "ranking_score": score(candidates[0]), "beam_width": width, "alpha": alpha}


def summarize(records, predictions):
    errors = [edit_distance(row["form"], item["prediction"]) for row, item in zip(records, predictions)]
    return {"count": len(records), "exact": sum(error == 0 and item.get("ended_with_eos", True) for error, item in zip(errors, predictions)),
            "character_edits": sum(errors), "reference_characters": sum(len(row["form"]) for row in records),
            "character_error_rate": sum(errors)/sum(len(row["form"]) for row in records),
            "no_eos": sum(not item.get("ended_with_eos", True) for item in predictions)}


@torch.no_grad()
def assess(model, records):
    source, lengths, decoder_input, target = batch(records)
    logits = model(source, lengths, decoder_input)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
    predictions = greedy(model, records)
    return {"teacher_forced_nll": float(loss), **summarize(records, predictions)}, predictions


def main():
    torch.set_num_threads(1)
    train, development = load_records()
    report = {"versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "tokens": TOKENS, "allowed_output_ids": ALLOWED, "protocol": {"seeds": [1, 2, 3], "updates": 1200,
              "batch_size": 64, "adam_lr": .003, "gradient_clip": 1., "max_output": MAX_OUTPUT}, "baselines": {}, "runs": []}
    for partition, records in (("train", train), ("development", development)):
        report["baselines"][partition] = {}
        for name, predictor in (("copy", lambda lemma, feature: lemma), ("predeclared_suffix_rules", suffix_baseline)):
            predictions = [{"prediction": predictor(row["lemma"], row["feature"])} for row in records]
            report["baselines"][partition][name] = summarize(records, predictions)
    for seed in report["protocol"]["seeds"]:
        torch.manual_seed(seed)
        model = Inflector()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        generator = torch.Generator().manual_seed(100+seed)
        checkpoints, clipped = [], 0
        for update in range(1201):
            if update in (0, 100, 400, 800, 1200):
                metrics, _ = assess(model, development)
                checkpoints.append({"update": update, **metrics})
                print(json.dumps({"seed": seed, **checkpoints[-1]}), flush=True)
            if update == 1200:
                break
            model.train()
            chosen = torch.randint(len(train), (64,), generator=generator).tolist()
            source, lengths, decoder_input, target = batch([train[index] for index in chosen])
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, lengths, decoder_input)
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            clipped += float(norm) > 1
            optimizer.step()
        training, _ = assess(model, train)
        metrics, predictions = assess(model, development)
        details = [{"lemma": row["lemma"], "feature": row["feature"], "reference": row["form"], **item} for row, item in zip(development, predictions)]
        run = {"seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "clipped_updates": clipped,
               "checkpoints": checkpoints, "train": training, "development": metrics,
               "development_predictions": details,
               "weights": {key: value.tolist() for key, value in model.state_dict().items()}}
        if seed == 1:
            beam_predictions = [beam(model, row, width=3) for row in development]
            run["beam3_development"] = summarize(development, beam_predictions)
            run["beam3_predictions"] = beam_predictions
            for row, expected in zip(development[:12], predictions[:12]):
                actual = beam(model, row, width=1)
                assert actual["tokens"] == expected["tokens"]
        report["runs"].append(run)
        (ROOT/"calculated-inputs.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"seed": seed, "parameters": run["parameters"], "train": training, "development": metrics,
                          "beam3": run.get("beam3_development")}), flush=True)


if __name__ == "__main__":
    main()
```

</details>

The decoder's previous token is selected with `argmax` only during generation. The training graph uses the known target prefix and differentiable logits. We do not backpropagate through the discrete greedy choices to fit this maximum-likelihood model.

The fixed learned model also supports inference without fitting. After the first run, save this next block beside the program and JSON and run it. It loads seed1's saved parameters, asks only for the lemma and grammatical request, and prints both the string and whether EOS ended it. No reference answer is supplied to generation.

```python
from pathlib import Path
import json
import runpy
import torch

folder = Path(__file__).resolve().parent
api = runpy.run_path(str(folder / "inflection-seq2seq.py"))
saved = json.loads((folder / "calculated-inputs.json").read_text())
model = api["Inflector"]().eval()
weights = saved["runs"][0]["weights"]
model.load_state_dict({
    key: torch.tensor(value, dtype=torch.bool if key == "invalid_output" else torch.float32)
    for key, value in weights.items()
})
query = {"lemma": "lactate", "feature": "past"}
for limit in (3, 16):
    result = api["greedy"](model, [query], max_output=limit)[0]
    print(limit, result["prediction"], result["ended_with_eos"])
```

The recorded output is `3 lac False` followed by `16 lactated True` for the worked training example. The browser investigations will similarly load one saved seed only when needed and perform bounded inference on short inputs, not train 1,200 updates in a page.

## 8. Diagnose the model by separating four questions

**Did we ask and encode the right task?** Inspect the request token, source lengths, target shift, EOS and output vocabulary. Swapping a row's reference form must not change the encoder input. If a source spelling is edited, rerun the encoder; do not keep its old context. A typo or unsupported character needs an explicit input decision.

**Can the network fit the supplied examples?** Low training loss and many exact training outputs show optimization has learned something about those pairs. If even a small training batch cannot be learned, inspect gradients, state detachment, masking, data pairing and parameter updates before adding decoding restrictions.

**Does the learned rule transfer?** Compare the same fixed model on grouped development examples and the simple baseline. Examine actual outputs, not only a scalar average. A grammatically plausible suffix attached to the wrong copied stem is still an error. A reference may itself contain an unusual lexicon entry; inspect provenance before “correcting” it to your expectation.

**Is search failing to find a good route the model already scores well?** Compare greedy and beam under a fixed model and declared scoring rule. An increase in model score with a decrease in exact match is possible. Beam size is an inference choice; training updates are a model change. Mixing them in one unexplained curve hides what caused the result.

**Investigation: change the source and the decoder prefix.** Start from a new development spelling, with the predicted outcome unset. Change one actual source character or its grammatical request, inspect the first-token distribution or generated string relationship, and show immediately the new states. In another branch force one generated character and inspect the following state. The probability distribution that produced the forced character is unchanged; later distributions can change because the character is now an input.

A useful null experiment replaces the context with an exact copy of itself: nothing should change. Replaying an already generated prefix should recover the same continuation. Replacing it with a different source's context should use that context's information. Changing only an on-screen label should affect none of the numbers.

Do not ban repeated characters merely because a model repeats. `letter` and `unsubbed` legitimately contain repeats. Diagnose data and generation first; use a constraint only when the task itself rules out the affected outputs.

## Reuse the cell; implement the encoder–decoder protocol

This lesson's new program is the protocol joining two recurrent computations, not another invention of GRU. [sequence-mechanics.py](sequence-mechanics.py) opens the scalar joint derivative, manual gate trace and exact small probability-tree search. [inflection-seq2seq.py](inflection-seq2seq.py) supplies complete source batching, `Inflector`, training, greedy decoding, beam search and evaluation. The prepared [recurrent cell owner](../rnns-lstms-grus/recurrent-mechanics.py) already maps gate order and biases to `nn.GRU`; until its improved page is published, that exact packet remains the honest prerequisite source.

The ordinary implementation uses embeddings and `nn.GRU` for encoding/decoding, and explicit code for shifting targets, carrying context and deciding when to end. There is no requirement to replace this small research model with a downloaded language-model wrapper. The supplied beam function owns candidate state: token IDs, accumulated log probability, end status and decoder state must travel together. A batched decoder can share the encoder memory, but its beam-specific hidden states cannot be accidentally shared and mutated.

The hand-search tree and trained inflector answer different questions. The tree checks search arithmetic exactly; the trained model checks whether learned conditional distributions support useful outputs. Width1 beam should match greedy under the same tie/termination rule. An ended hypothesis is retained without repeatedly consuming EOS, while a hypothesis that reaches the step cap is reported as capped. Length normalization changes ranking; it is not a harmless numerical rescaling.

**Changed-code task:** add a second source to a batched decoding routine, one ending after2 tokens and another after5. Keep an explicit ended mask and original source IDs. After an example ends, preserve its final sequence and stop assigning it new scored tokens; continue the other example. Test that decoding this batch gives the same two results as separate calls in eval mode. For beam search additionally reorder decoder states with the same parent indices used to gather candidate tokens.

<details><summary>Hint</summary>A batch is a collection of independent sequence states, not one common EOS event.</details>

<details><summary>Solution and success criteria</summary>Initialize one hidden state and ended flag per source. On each step form candidate logits only for active rows, append their chosen tokens, mark newly emitted EOS and gather any beam parents consistently. Already-ended output strings remain unchanged. Compare complete token sequences and log probabilities, including the case where one row is capped and the other genuinely ended. Padding is storage, not another generated token. Correct source-to-state ownership matters more than saving a few Python lines.</details>

## 9. Deeper connections and practical extensions

<details>
<summary>Different context interfaces: GRU, LSTM, stacks and attention</summary>

Our decoder initializes from one GRU state. An LSTM has both hidden state $h$ and cell state $c$; a complete handoff must say what happens to both. With several layers, state has a layer axis. A bidirectional encoder has two directional states per layer. “Pass the final hidden vector” is insufficient when the decoder expects a different shape or a missing cell state.

If encoder and decoder widths differ, a learned projection can map a concatenated encoder representation into the required decoder state. For example, two encoder directions of 64 coordinates can be concatenated into 128 and projected to a 64-coordinate decoder state. An LSTM may need separate projections for hidden and cell states. A deliberate zero initialization with a separately supplied context is another design.

Context can also be concatenated to the decoder input at every step. That repeatedly supplies the same source summary, whereas attention supplies a **different weighted combination of source states** for each decoding step. The next lesson derives that mechanism. The difference is access to information, not a guarantee that the weights perfectly explain language or that generation becomes factual.

The 2014 Sutskever system demonstrated large recurrent encoder–decoder translation with word vocabularies and unknown-word tokens. Its reported 34.81 BLEU result used an ensemble of five models and beam 12. Source reversal shortened some important dependency paths; it did not reverse the target language or eliminate recurrent computation. The paper actually reported good performance on long sentences in that setting, so a universal “fails after 30 words” claim would misrepresent it. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215).

</details>

<details>
<summary>Teacher forcing, scheduled sampling and sequence objectives</summary>

Maximum likelihood scores observed prefixes. Deployment uses generated prefixes. That mismatch can expose weaknesses in an imperfect model, but the likelihood objective remains mathematically coherent. A generated-prefix intervention measures a particular response; it does not prove a universal account of every sequence error.

Scheduled sampling mixes reference and generated previous tokens during training, typically changing the mixing probability over time. The targets can remain the original next tokens even when the prefix has changed. That means it is no longer simply evaluating the original data likelihood. The original proposal reported useful results, while an analysis of the sampling objective showed an inconsistency even in a two-symbol setting: replacing the first symbol independently can encourage prediction of the second marginal rather than the correct conditional relationship. It is not an automatic required upgrade. [Bengio et al.](https://arxiv.org/abs/1506.03099), [Huszár's analysis, section 4](https://arxiv.org/abs/1511.05101).

Sequence-level objectives can optimize a reward or risk attached to the complete answer. They introduce their own estimation, optimization and evaluation questions. Label smoothing changes target distributions at the loss; it does not itself train on the model's wrong prefixes. Keep these mechanisms distinct when interpreting an experiment.

</details>

<details>
<summary>Search costs, stopping and model deployment</summary>

With output vocabulary size $V$, limit $T$ and beam width $K$, exhaustive enumeration has exponentially many possible paths, whereas beam expansion considers roughly $KVT$ token extensions. This count omits the cost of each neural state update, embedding lookup, projection and sorting. It is an algorithmic description, not a measured latency claim.

For raw log scores, extending a particular live path cannot improve its score. A completed candidate that already beats every live prefix cannot be beaten by descendants of those retained prefixes. This does not recover routes already pruned. Length-normalized scores need a compatible bound because their denominator changes; borrowing a raw-score stopping proof would be invalid.

In a deployed model, record the exact tokenizer and vocabulary, source normalization, checkpoint revision, input limit, decoder-start and EOS IDs, padding side, precision/device, beam or sampling settings, score definition, output limit and termination reason. For multilingual models a language token is checkpoint-specific, not a universal string format. Keep model and tokenizer versions paired.

An off-the-shelf `generate` API can manage these steps but does not remove their meaning. For example, current Transformers distinguishes beam stopping based on enough completed candidates, a heuristic, or a stricter search condition. Its length-penalty convention need not be the exact GNMT denominator used above. Inspect the actual configuration and documentation instead of assuming a familiar parameter name has one universal definition. [Transformers generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

Translation evaluation usually needs more than exact string match because multiple translations can be acceptable. BLEU measures a form of reference n-gram agreement with length handling; it is not a probability of truth. Report tokenization and metric configuration and include appropriate human/task checks. For short word forms here, exact match, character edits, accepted-reference policy and termination are easier to interpret. Speech and structured-output tasks need their own units and validity checks.

An encoder–decoder can use source context and still invent unsupported content. A decoder-only model can also condition on an input prefix. Architecture family alone establishes neither faithfulness nor a universal speed ranking.

</details>

## 10. Practice: build, diagnose and change the problem

### 1. Repair a shifted target

For `try + past → tried`, write the six decoder inputs and six target tokens. Which target position is lost if you train only on the five letters?

<details><summary>Hint</summary>

The first input starts generation, and the final output teaches termination. Each other input is the immediately preceding target.

</details>
<details><summary>Solution</summary>

Inputs: BOS,t,r,i,e,d. Targets: t,r,i,e,d,EOS. Omitting the sixth target removes the supervised instruction to stop after `d`. EOS is not padding.

</details>

### 2. Compute a fresh likelihood and loss mask

An answer `go` followed by EOS receives correct-token probabilities 0.8,0.5,0.25. Compute its sequence probability and mean token NLL. Two padded storage positions are appended. Should the valid-token mean change?

<details><summary>Hint</summary>

Multiply probabilities for the route; add their negative natural logarithms for the loss. Count EOS but do not count storage padding.

</details>
<details><summary>Solution</summary>

The probability is 0.1. The total NLL is $-\log(0.1)=2.302585$, so the three-token mean is 0.767528. Correctly ignored padding leaves it unchanged. Dividing the same loss sum by five instead gives 0.460517 and silently changes the scale.

</details>

### 3. Separate a source edit from an answer edit

In teacher-forced training, change only the last character of a reference form. Must the encoder state change? Must the distribution predicting that changed target position change? What about the following decoder step?

<details><summary>Hint</summary>

Trace which array each component reads. A target being scored is not yet the previous token supplied to the decoder.

</details>
<details><summary>Solution</summary>

The encoder state stays fixed because its source is unchanged. The distribution at the edited target position stays fixed if the preceding prefix is unchanged; the correct label and loss can change. The following step receives the edited character as input and can have a different state and distribution.

</details>

### 4. Make greedy lose, then make it win

Use a new tree: first A 0.55/B 0.45; after A choose EOS 0.60/C 0.40; after B choose EOS 0.85/C 0.15; after C emit EOS with probability 1. Enumerate all complete paths and compare greedy with beam 2. Then change only P(EOS|A) to 0.90 and its complement accordingly.

<details><summary>Hint</summary>

The first token's probability alone does not rank complete paths. Keep each conditional row normalized after the edit.

</details>
<details><summary>Solution</summary>

Initially A,EOS=.33; A,C,EOS=.22; B,EOS=.3825; B,C,EOS=.0675. Greedy returns A,EOS, while beam 2 finds B,EOS. After the edit A,EOS=.495 and A,C,EOS=.055, so both return A,EOS. Improving search does not require it to return a different answer on every input.

</details>

### 5. Spot the leaked evaluation unit

A dataset contains `worke→worked` in training and `work→worked` in development. A report says “all lemma strings are distinct, therefore this measures completely new lexical items.” What is wrong, and what did this lesson do?

<details><summary>Hint</summary>

A string identity check is useful but narrower than a claim about linguistic identity. Inspect related variants and shared forms.

</details>
<details><summary>Solution</summary>

Distinct spellings can represent closely related variants and share targets. The report overstates what its check proves. This packet conservatively links selected lemma spellings sharing a target, keeps the connected group in one partition and records the policy. It still does not claim to have solved all linguistic alias detection.

</details>

### 6. Explain a smaller loss but worse product

Suppose model A has lower teacher-forced NLL, model B has better generated exact match, and a rule system beats both on this task. Which should you report? Does the discrepancy mean the NLL calculation is broken?

<details><summary>Hint</summary>

Each measurement asks a different question: probability on known-prefix targets, success of a generation procedure, and utility of a specific alternative.

</details>
<details><summary>Solution</summary>

Report all relevant measurements with the same data split and protocol. Lower NLL can improve average probabilities without changing argmax decisions in the same way, and generated prefixes can differ from reference prefixes. The discrepancy does not itself show a broken loss. Choose according to the deployment requirements and reliable evaluation; the rule system remains a legitimate candidate.

</details>

### 7. Investigate a cap without pretending the answer ended

Use the saved seed 1 model on a development input. Set the generation limit to 3 tokens and compare with 16. Record the emitted tokens, EOS flag and log score. Explain why the three-token prefix can have a higher raw score but be an incomplete answer.

<details><summary>Hint</summary>

A prefix probability sums over possible future continuations; it has not yet paid the probability cost of choosing one of them and ending.

</details>
<details><summary>Solution and expected check</summary>

On `emmove + past` the limit 3 output is `emo` with EOS=false and log score −0.833265. The limit 16 run continues to `emoves` and EOS in this saved model. A prefix can have a larger probability than any single complete extension. The cap flag must remain visible; do not relabel `emo` as a natural completed prediction.

</details>

### 8. Design the next controlled comparison

You want to replace the single context with access to all encoder states. Name what you would keep fixed, what you would measure, and one reason an improvement would not prove that attention alone caused every difference.

<details><summary>Hint</summary>

Think about data groups, token support, training budget, parameter counts, decoding, seeds and what information has already been inspected.

</details>
<details><summary>Solution</summary>

Keep the exact data/split, tokenization, target masking, training and evaluation definitions, decoding convention and declared seeds fixed where possible. Report parameter-count and compute changes, generated exact match, edits, termination and teacher-forced loss. Compare failures and source-length slices without selecting only favorable examples. Adding attention changes parameters and optimization as well as information access; a small controlled example is evidence for its protocol, not a universal causal ranking. The same development set remains development.

</details>

## 11. References, another way to learn, and the next step

- [Dive into Deep Learning 1.0.3: encoder–decoder, seq2seq and beam search](https://d2l.ai/chapter_recurrent-modern/seq2seq.html). A useful second implementation route, especially the shifted-target and masking sections. Read the beam chapter with its exact score convention in view; increasing the denominator of a negative log score does not universally penalize longer outputs.
- [Sutskever, Vinyals and Le 2014](https://arxiv.org/abs/1409.3215). Read section 2 for the original model contract and sections 3.2–3.3 for search and source reversal. Its large translation experiment is historical evidence, not the setup of our small character model.
- [Cho et al. 2014: RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078). A complementary formulation in which source context enters the conditional decoder. Useful after you can trace our simpler initial-state interface.
- [Stanford CS224N 2019, Lecture 8: Translation, Seq2Seq, Attention](https://www.youtube.com/watch?v=XXtpJxZBa2c), with [companion notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf). A lecture-based alternative covering the motivation and the transition to attention. The basic encoder–decoder portions fit this lesson; return to the attention portion after the next one. The resource's age matters for “current standard” statements and framework code.
- [UniMorph schema and data project](https://unimorph.github.io/), [UniMorph 4.0 paper](https://aclanthology.org/2022.lrec-1.89/), and [the pinned English source](https://github.com/unimorph/eng/tree/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b). These explain what the real lexical records mean. The supplied extract retains source row numbers, filtering, grouping and CC BY-SA 3.0 attribution.
- [PyTorch 2.14 CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html). Use this to verify raw-logit input, class indices, `ignore_index` and reduction when adapting the program.

You can now connect source tokens, encoder state, decoder state, next-token probabilities, sequence loss and a complete generation procedure. Next, [Attention Mechanisms: Bahdanau and Luong](/learn/path/full-curriculum/attention-mechanism-bahdanau-luong?module=deep-learning-fundamentals) lets the decoder consult the sequence of encoder states at each output step. We will test that change on this same bounded task rather than assume it solves every failure.
