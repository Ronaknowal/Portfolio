# Hyena & Long Convolution Models

**Explore as you read.** Edit convolution signals/kernels, gate vectors, causal/circular mode, recurrence poles/residues, chunk cut and supported biological-sequence symbols. Show dependency ranges, computed outputs, gated operator entries, retained state and exact frozen-model responses as edits apply. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to diagnose future leakage, choose effective context and distinguish exact recurrence carry from truncating the filter.


A short convolution asks what happened nearby. A long convolution lets a distant event still contribute to what happens now. Hyena adds a further choice: the input controls what gets transmitted through that long filter and how the receiver uses it.

Imagine a sensor that produces a sharp pulse. A filter can make that pulse leave a fading echo. Now imagine a sequence of symbols in which some events are useful and others should be suppressed. A fixed echo pattern cannot make that decision by itself. Gates derived from the symbols supply the missing input dependence. Hyena combines these operations into a sequence mixer that can process a whole sequence with fast convolution algorithms.

The preceding [xLSTM lesson](/learn/path/full-curriculum/xlstm-extended-lstm?module=deep-learning-fundamentals) built a memory state and asked how to read it. Here we start from the **influence of an earlier position on a later position**. This change in viewpoint reveals both the appeal of long convolutions and a subtle problem: a fast whole-sequence computation does not automatically provide cheap one-token-at-a-time generation.

**Your first pass:** follow §§1–7, then try practice 1–6. You should be able to compute a causal convolution, identify circular wraparound, explain a gate-filter-gate block, and interpret the real sequence experiment. §8 develops streaming and recurrence extraction; §9 connects modern variants. Those branches and practice 7–10 provide the deeper route. You need weighted sums, matrix shapes and the idea of training by reducing a loss. We introduce the Fourier and signal-processing vocabulary locally.

## 1. Three questions about a long sequence

Suppose a model sees `A 7 B 2 … B` and should answer `2`. The last symbol must identify which earlier association matters. Contrast that with copying `4 7 2` after exactly three time steps: a fixed delay is sufficient for the latter. Both involve distant information, but only the first requires choosing an address from the content.

Sequence mixers make different tradeoffs among three questions:

| Question | What it means | A concrete diagnostic |
| --- | --- | --- |
| Can information travel far? | An old input has a computational path to a new output. | Place a nonzero input far back and inspect its contribution. |
| Can the input change which information matters? | The mixing weights depend on the sequence being processed. | Change a key or a gate while keeping distance fixed. |
| Can the computation be evaluated efficiently? | The algorithm avoids unnecessary work or storage. | Derive the operations actually performed, then measure a specified implementation. |

Hyena’s design joins long filters, input-dependent gates and fast convolution. These are properties of its mechanism; whether training learns a useful solution is a separate empirical question. [The original paper](https://arxiv.org/html/2302.10866v2) used associative recall and other small controlled tasks to guide its design before studying language and vision.

**Figure H01 — Two kinds of “remember.”** Show a fixed-delay conveyor carrying `4,7,2` into three predetermined output slots beside a key-addressed request with rearrangeable `A→7, B→2` pairs. Highlight that changing pair order changes the relevant distance in the second task. This is a task diagram, not a measured model result.

## 2. A convolution is a ledger of delayed contributions

Let $u_t$ be the input at position $t$, starting at zero. Let $h_r$ be the filter coefficient for a **lag** of $r$ positions. The causal output is

\[
y_t=\sum_{j=0}^{t}h_{t-j}u_j
    =h_0u_t+h_1u_{t-1}+\cdots+h_tu_0.
\]

“Causal” means the output at $t$ uses no input after $t$. “Filter” means the collection of coefficients. The name **impulse response** describes the same object from a useful experiment: put a single unit pulse at $u_0=1$, followed by zeros, and the output is $h_0,h_1,h_2,\ldots$. The filter is the echo of that pulse.

Take $u=[1,2,3,4]$ and $h=[1,0.5,0.25,0.125]$. At position 2, the current input contributes 3, the previous input contributes $0.5\times2=1$, and the oldest input contributes $0.25\times1=0.25$. The total is 4.25.

| Position $t$ | Current-to-oldest contributions | $y_t$ |
| --- | --- | ---: |
| 0 | $1\times1$ | 1 |
| 1 | $1\times2+0.5\times1$ | 2.5 |
| 2 | $1\times3+0.5\times2+0.25\times1$ | 4.25 |
| 3 | $1\times4+0.5\times3+0.25\times2+0.125\times1$ | 6.125 |

**Figure H02 — Echo ribbons.** Put input pulses on a horizontal position axis; each pulse emits a scaled copy of the filter. At output position 2, stack the three signed contributions, with labels `3`, `1`, `0.25`. An aligned table supplies the exact values. A pulse ribbon should not imply a physical time unit: these are sequence positions and dimensionless values.

The equivalent matrix is

\[
\begin{bmatrix}1&0&0&0\\.5&1&0&0\\.25&.5&1&0\\.125&.25&.5&1\end{bmatrix}
\begin{bmatrix}1\\2\\3\\4\end{bmatrix}
=\begin{bmatrix}1\\2.5\\4.25\\6.125\end{bmatrix}.
\]

This is a **lower-triangular Toeplitz matrix**: “lower triangular” encodes causality; “Toeplitz” says each diagonal uses the same lag coefficient. Every pair of positions at distance 2 receives $h_2$. A length-$K$ finite impulse response, or FIR, filter additionally sets $h_r=0$ for $r\ge K$.

**Figure H03 — Distance becomes a diagonal.** Label matrix rows as receiving positions and columns as sending positions. Use one pattern per lag, show the upper triangle as forbidden future access, and trace the lag-2 diagonal back to its filter coefficient. Keep signed values visible when later examples use negative coefficients.

For a length-$L$ filter applied to $L$ inputs, direct causal evaluation has $L(L+1)/2$ coefficient-input products per channel. With many channels and long sequences, that work matters. Yet the matrix contains extensive repetition. The FFT exploits this structure.

## 3. Why an FFT can compute the same sum faster

The discrete Fourier transform represents a finite signal using oscillating basis patterns. Think of changing coordinates: we can describe a sound by its samples over time or by the amplitudes and phases of its frequencies. We have not yet removed information.

For two arrays padded to length $P$, the Fourier convolution theorem gives

\[
\operatorname{IFFT}\left(\operatorname{FFT}(u)\odot\operatorname{FFT}(h)\right).
\]

Here $\odot$ is elementwise multiplication. The forward transforms separate the oscillating components; multiplication applies the filter to those components; the inverse transform reconstructs samples. The FFT is an efficient algorithm for that coordinate change, with work proportional to $P\log P$.

There is a catch. A length-$P$ transform treats indices modulo $P$: values that run past the right edge wrap to the left. It naturally computes **circular convolution**. To recover ordinary linear convolution of lengths $L$ and $K$, choose $P\ge L+K-1$, pad both arrays with zeros, and keep the desired outputs. For causal sequence mixing, keep the first $L$.

**Figure H04 — Make room for the tail.** Show an unpadded ring beside a padded straight strip. For the four-input example, ordinary convolution has seven output slots; using eight FFT slots is convenient. Color wrapped tail contributions separately. Do not draw Fourier bins as if they were token positions.

The wrong $P=4$ computation for our example produces `[4, 3.875, 4.75, 6.125]`. Its first output is

\[
1\times1+0.5\times4+0.25\times3+0.125\times2=4.
\]

That first output already contains the future input 4. Changing only the final input from 4 to 8 changes the wrong first output from 4 to 6. In the correctly padded computation the first three outputs remain `[1,2.5,4.25]`.

### A complete numerical program

Install NumPy in your own environment with `python -m pip install numpy`, save this as `causal_fft.py`, and run `python causal_fft.py`. Both inputs and expected outputs are included.

```python
import numpy as np

def causal_fft(values, kernel):
    length = len(values)
    full_length = length + len(kernel) - 1
    fft_length = 1 << (full_length - 1).bit_length()
    product = np.fft.rfft(values, fft_length) * np.fft.rfft(kernel, fft_length)
    return np.fft.irfft(product, fft_length)[:length]

values = np.array([1., 2., 3., 4.])
kernel = np.array([1., .5, .25, .125])
direct = np.convolve(values, kernel)[:len(values)]
fast = causal_fft(values, kernel)
print(np.round(direct, 6))
print(np.round(fast, 6))
print(np.allclose(direct, fast, atol=1e-12))
```

The arrays both display `[1. 2.5 4.25 6.125]`, followed by `True`. The underlying floating-point arrays need not be bit-for-bit identical. In the saved calculation the first FFT output is `0.9999999999999996`. That numerical roundoff is distinct from the large semantic wraparound error above. Always pass the intended inverse-transform length, especially for odd sizes; make normalization conventions agree. The [PyTorch FFT documentation](https://docs.pytorch.org/docs/main/generated/torch.fft.rfft.html) explains normalization and device/dtype restrictions.

**Investigation HA — Find the future leak.** Start with a fresh editable signal `[2,-1,3,0,1]` and filter `[0.5,1,-0.25]`. Before running, inspect which outputs can change when you edit the final input. Compare direct, correctly padded FFT and intentionally circular computation. Repair the padding, then construct a different signed filter for which the circular result happens to agree at one output. Explain why one matching number does not establish causality. The impulse and identity-filter views help distinguish lag direction from wraparound.

## 4. Gates make the mixing depend on the input

A useful basic Hyena block has three projected streams: $q$, $k$ and $v$, each shaped $L\times D$. Here $D$ is the channel width. These letters resemble attention’s query, key and value names, but the computation below contains no softmax and no pairwise query-key dot product.

For one channel,

\[
z_j=k_jv_j,\qquad r_t=\sum_{j\le t}h_{t-j}z_j,
\qquad y_t=q_tr_t.
\]

The sending gate $k_j$ changes what enters the filter. The receiving gate $q_t$ changes how the result is used. Gates can amplify or reverse sign; they are not automatically probabilities in $[0,1]$. A neural implementation derives them from the input using learned projections and small causal depthwise convolutions. “Depthwise” means each channel has its own short filter rather than mixing channels at that operation. Dense projections before and after it handle channel mixing.

**Figure H05 — The gate-filter-gate signal path.** Three parallel rails begin at the same input: `q`, `k`, `v`. Only `k×v` enters the long filter. The receiver multiplies by `q`. Show the short causal preprocessing separately from the long filter so that the number and role of each convolution remain visible.

Use the earlier $v=[1,2,3,4]$, filter $h$, sending gates $k=[1,0,-1,2]$, and receiving gates $q=[1,2,-1,0.5]$. Then

\[
kv=[1,0,-3,8],\quad h*(kv)=[1,.5,-2.75,6.625],
\quad y=[1,1,2.75,3.3125].
\]

At the last position the contributions before the receiving gate are $0.125+0-1.5+8=6.625$. The final factor 0.5 gives 3.3125. The zero gate suppresses the second value along this path; the negative gate reverses the third.

We can expose every pairwise coefficient without using a dense matrix to run the block:

\[
H(q,k)=\operatorname{diag}(q)\,T_h\,\operatorname{diag}(k),
\qquad H_{tj}=q_t h_{t-j}k_j\quad(j\le t).
\]

**Figure H06 — A coefficient is three factors.** For each selected matrix cell, show receiving gate × distance coefficient × sending gate. One gate edit changes a row or column of this conditional matrix. Use a diverging signed scale, a numeric cell label and a contribution ledger. Row totals need not equal one.

For fixed $q,k,h$, this is linear in $v$. A complete block is nonlinear in its original input because the same input also changes its gates. Consequently the displayed matrix is neither the full input-output Jacobian nor a guaranteed causal explanation of the network’s final prediction. It describes a specific internal path with gates held fixed.

### Why “hierarchy” matters

The general construction alternates filters and gates:

\[
z^{(0)}=v,\qquad z^{(n)}=x^{(n)}\odot(h^{(n)}*z^{(n-1)}),
\quad n=1,\ldots,N.
\]

The associated product is $D_{x^{(N)}}T_{h^{(N)}}\cdots D_{x^{(1)}}T_{h^{(1)}}$. Each additional stage introduces intermediate positions through which information can pass. With two stages,

\[
y_t=q_t\sum_{m\le t}\psi_{t-m}k_m\sum_{j\le m}\varphi_{m-j}v_j,
\quad
H_{tj}=q_t\sum_{m=j}^{t}\psi_{t-m}k_m\varphi_{m-j}.
\]

The sum over $m$ is the new feature: an input can reach the receiver through several intervening gates. Every path still goes forward in sequence order, proving causality if the gate-generating operations are also causal.

**Figure H07 — Paths through an intermediate gate.** Trace two possible routes from source $j$ through intermediate $m$ to receiver $t$, then show the summed coefficient. Compare with the single-filter sandwich; do not present the two matrices as identical.

Terminology differs across descriptions. The original paper’s formal $N$-stage hierarchy counts alternating filter/gate stages. HyenaDNA’s common `order=2` implementation has three projected streams, two gates and one long convolution after their short preprocessing. In this lesson the practical model is explicitly a **one-long-filter Hyena-style block**, stacked twice. Inspect the actual equation or code rather than infer the computation from “order two.” [HyenaDNA’s method](https://arxiv.org/html/2306.15794v2) and [the authors’ standalone code](https://github.com/HazyResearch/hyena-dna/blob/main/standalone_hyenadna.py) show this convention.

**Investigation HB — Build a selective route.** You receive editable values `[2,-1,3,1]`, sending gates `[0.5,1,0,-1]`, receiving gates `[1,-0.5,2,1]`, and a signed filter `[1,-0.25,0.5,0]`. Inspect which outputs respond when the zero sending gate becomes one. Run, inspect the signed matrix and create a different arrangement that suppresses one sender while preserving another. Add an optional preceding filter `[1,-0.5,0,0]` and reason about the new intermediate paths. the retained baseline is recorded before each changed-input result.

## 5. Generate the filter from position

An explicit length-$L$, $D$-channel filter stores $LD$ learned coefficients. An **implicit filter** instead learns a function whose input is lag and whose output is a vector of channel coefficients:

\[
h_r=\gamma_\theta(\operatorname{pos}(r))\odot w(r).
\]

The small neural network $\gamma_\theta$ shares parameters across positions. The window $w(r)$, often an exponential envelope, biases the filter’s range. A position representation may include the lag itself and sine/cosine features at several frequencies. A sine activation in the filter network makes oscillating and sharply varying filters easier to represent than an overly smooth initialization would suggest.

**Figure H08 — A function becomes a strip of coefficients.** Feed lag coordinates `0,1,…,59` into one shared network and place its outputs along channel strips. Select one channel to show its raw network curve, positive decay envelope and product. Label whether these are initialized or fitted coefficients; use the saved actual fit for the latter.

Our experiment fixes a reference length $L_{\rm ref}=60$, sets $s_r=r/59$, and uses

\[
\operatorname{pos}(r)=[s_r;(\sin(2\pi f s_r))_{f\in F};(\cos(2\pi f s_r))_{f\in F}],\quad F=\{1,2,4,8\},
\quad
h_r=W_2\sin(W_1\operatorname{pos}(r)+b_1)+b_2,
\]

followed by multiplication by $\exp[-s_r\operatorname{softplus}(a)]$, one decay rate per channel. The position vector has nine features. The network uses 32 hidden units and 16 outputs. Its learned count is $9\times32+32+32\times16+16+16=864$, including the 16 decay parameters. Producing all 60 coefficient vectors still requires evaluating 60 positions and storing their outputs; fewer learned parameters do not make that work disappear.

### A less obvious causality bug: moving the coordinate system

If a four-token prefix uses coordinates $r/3$, then the same prefix inside an eight-token sequence uses $r/7$, its filter changes when later tokens arrive. Correct FFT padding cannot repair that problem. A causal model must use the same lag interpretation for the same prefix.

For the illustrative function $h_r=e^{-s_r}\cos(2\pi s_r)$, using reference length 8 gives initial coefficients `[1,0.54049,-0.16722,-0.58693]`. Recomputing them with active length 4 gives `[1,-0.35827,-0.25671,0.36788]`. The first four coefficients differ by as much as 0.95481. Our implementation stores a fixed position grid and slices it for shorter inputs.

**Figure H09 — A ruler that must not stretch mid-read.** Align the same first four lag positions under the two coordinate conventions. Show the coefficient change at each shared lag, with a stable-reference ruler above both sequence lengths. A separate dashed region past the trained support marks a new extrapolation question, not a free extension.

A filter function may be mathematically evaluable at longer lags, but those values were not necessarily trained. Extending a position buffer, changing frequency scales and continuing training are separate choices. Some official configurations also learn positional embeddings, adding length-dependent parameters. Read the chosen configuration before claiming that *all* parameters are independent of maximum length.

The exponential window is an inductive bias, not a proof that every fitted coefficient decreases monotonically. The network can oscillate or grow inside the envelope; some implementations add a nonzero envelope floor. A finite filter without decay is perfectly well-defined. The practical question is which initialization and parameterization let the model learn useful filters stably.

## 6. How the block learns

Start with one trainable filter and a transparent loss. For inputs `[1,2]`, filter `[0.5,0.25]`, and desired final output 2, the prediction is $2h_0+h_1=1.25$. Use half squared error:

\[
\mathcal L=\tfrac12(2h_0+h_1-2)^2=0.28125.
\]

The prediction error is $-0.75$. Each coefficient’s gradient is this error multiplied by the input it weights: $\nabla_h\mathcal L=[-1.5,-0.75]$. One gradient step with learning rate 0.1 gives `[0.65,0.325]`, prediction 1.625 and loss 0.0703125. A coefficient grows because doing so moves this prediction toward its target.

**Figure H10 — Follow one gradient to two coefficients.** Trace `loss → prediction error → weighted inputs → coefficient updates`, using the exact values above. The arrow labels distinguish the learning rate from a gate and a filter coefficient.

In a neural Hyena block, backpropagation follows the same chain through receiving gates, convolution, sending gates and the filter-generating network. FFT convolution is differentiable. A direct implementation and a correctly normalized FFT implementation must agree on the operation’s derivatives within numerical tolerance. The saved double-precision probe checks both outputs and gradients against an independently indexed direct sum.

The full sequence block also has residual paths and a per-position feed-forward network. These keep the representation update separate from the long mixing operation. The small classifier below uses LayerNorm before each update, two blocks, a 16-dimensional token embedding and a three-logit head at the final position. Each block’s mixed update is

\[
q\odot\{h*(k\odot v)+b_{\rm skip}\odot(k\odot v)\}.
\]

The learned skip coefficient is another current-position path. It does not make the long filter noncausal. No probability interpretation is assigned to either gate; only the classifier’s final softmax produces class probabilities.

For next-token language modeling, a different head produces vocabulary logits at each position and the loss compares each output with the following token. Shift targets consistently; apply any loss mask to predictions of the intended continuation. Teacher-forced accuracy, where preceding true tokens are supplied, and free-running generation answer different questions. The sequence classifier we now train is supervised classification, so it uses one label for the whole 60-position window.

## 7. A real sequence task: identify a splice boundary

A gene can be transcribed into an RNA molecule containing regions called introns and exons. RNA splicing removes introns and joins retained exons; alternative splicing can produce different mature RNAs from one gene. The DNA itself is not cut up by this RNA-processing operation, and introns need not be biologically useless. The [NHGRI RNA explanation](https://www.genome.gov/about-genomics/educational-resources/fact-sheets/ribonucleic-acid-fact-sheet) gives the biological background.

The historical [UCI Splice-Junction dataset](https://archive.ics.uci.edu/dataset/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences) asks whether the central boundary in a 60-character DNA window is `EI` (exon to intron), `IE` (intron to exon), or `N` (neither). We use those explicit boundary directions. Some original metadata reverses the donor/acceptor names; we do not use the conflicting names to define the labels. The central boundary sits between array indices 29 and 30. Display positions as −30…−1 and +1…+30, with no fictitious middle base at zero.

The input alphabet is `A,C,G,T,D,N,R,S`. The last four symbols describe uncertainty among bases: `D` means A/G/T, `N` any of the four, `R` A/G, `S` C/G. They are not four extra chemical bases. We keep them as separate input symbols rather than invent a measured probability for the alternatives. A sequence symbol `N` is also distinct from the *output class* `N`.

**Figure H11 — A DNA window around a boundary.** Show all 60 positions with the central divider, observed characters, and the three possible class directions. A compact ambiguity key appears where needed. Selecting a base must identify both its array index and its biological relative-position label.

### Separate the data roles before fitting

The source contains 3,190 rows and 3,005 distinct sequence strings. Two identical-input rows, source IDs 1022 and 1969, have conflicting labels; we exclude both from this exercise and disclose the decision. We keep the first occurrence of every other sequence, leaving 3,004 examples. Before splitting, we join source-record prefixes connected by an identical sequence, then keep every resulting group within one data role. This also keeps different windows from each observed source-record prefix together.

| Role | Rows | Source/duplicate-connected groups | Use |
| --- | ---: | ---: | --- |
| Fit | 2,128 | 1,008 | Update parameters. |
| Validation | 460 | 216 | Choose an epoch for each predeclared model. |
| Assessment | 416 | 216 | Report the selected models once. |

Exact source IDs, group assignments and class counts accompany the downloadable data. Distinct record names can still describe homologous biological sequences. This grouping addresses the relationships visible in the file; it does not establish independence across species, gene families or new patients. The assessment is a modest historical-data exercise, not a biological validation study or evidence of million-token performance.

**Figure H12 — Keep connected records on one side.** Draw a source-prefix node joined to its windows, and a duplicate-sequence edge joining two source prefixes. Move the entire connected component into one of the three role lanes. Separately place the two conflicting records in a disclosed exclusion tray.

### Train a model you can inspect

Download [the complete training program](splice_models.py), [the data](splice.data) and [its original metadata](splice.names) into one directory. The program needs Python, NumPy, PyTorch and scikit-learn. In a fresh environment run:

```text
python -m pip install numpy torch scikit-learn
python splice_models.py
```

The retained run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1, with two CPU threads. It trains four predeclared fits: a positional one-hot linear classifier with seed 29, gated sequence models with seeds 29 and 71, and an ungated sequence model with seed 29. All use Adam at learning rate 0.003, batches of 128, 80 epochs and gradient norm clipping at 1. The epoch with smallest validation cross-entropy is retained, with the earliest exact tie. The assessment data does not choose the epoch.

The positional linear classifier is a meaningful baseline: the target boundary is always at the same location, so a coefficient for “G at index 30” can be informative. The ungated model retains its nonlinear feed-forward blocks but sets the two mixing gates to one; it is **not a purely linear network**. Its otherwise retained projections include unused gate branches, so its nominal parameter count should not be read as equal effective capacity.

The training program contains data grouping, encoding, the filter network, FFT convolution, residual blocks, loss, optimizer, epoch selection, confusion matrices and saved weights. This is the core mixing function used there:

```python
def causal_convolution(values, kernel):
    # values: batch, length, channels; kernel: length, channels
    length = values.shape[1]
    size = 1 << (2 * length - 2).bit_length()
    spectrum = torch.fft.rfft(values, n=size, dim=1)
    filter_spectrum = torch.fft.rfft(kernel, n=size, dim=0)
    return torch.fft.irfft(
        spectrum * filter_spectrum[None], n=size, dim=1
    )[:, :length]
```

Padding is along the sequence axis, not the channel axis. The filter broadcasts across the batch. The full source imports `torch`; this excerpt belongs to that runnable program. It produces `splice-results.json` and `splice-fits.npz`, containing role IDs, validation histories, selected weights and logits. You can reproduce the experiment without downloading a large pretrained model.

### Read the actual outcome

| Model | Selected epoch | Validation errors / 460 | Assessment errors / 416 |
| --- | ---: | ---: | ---: |
| Positional linear, seed 29 | 42 | 32 | 30 |
| Gated Hyena-style, seed 29 | 11 | 45 | 40 |
| Ungated sequence model, seed 29 | 25 | 82 | 86 |
| Gated Hyena-style, seed 71 | 11 | 50 | 57 |

The positional baseline makes fewer errors than either small gated fit. That is an informative result: a fixed-location motif task can reward a direct representation. Gating helps relative to this ungated comparison, but changing seed also changes outcomes. These fits are not a universal ranking of linear classifiers, Hyena or other architectures.

**Figure H13 — Learning curves with a decision point.** Plot the actual 80 validation cross-entropies for each predeclared fit, with the selected epoch marked. Use a common vertical axis with an optional detailed view; the assessment outcome is a separate table. Do not synthesize a smooth training curve or display assessment scores at every epoch.

For the gated seed-29 model, the assessment confusion matrix is:

| True \ Predicted | EI | IE | N |
| --- | ---: | ---: | ---: |
| EI | 76 | 3 | 6 |
| IE | 1 | 74 | 8 |
| N | 13 | 9 | 226 |

Rows reveal different errors: 9 of 85 EI examples, 9 of 83 IE examples and 22 of 248 N examples are misclassified. A single overall accuracy hides those denominators. A confidence value alone also does not establish calibration.

### Change a sequence and observe a hypothesis

Validation source row 3 is an EI example whose two bases immediately after the central divider are `GT`. The gated seed-29 fit assigns class probabilities approximately `[0.99284,0.00648,0.00068]` in EI/IE/N order. Replacing those two input bases by `AA`, without refitting, produces `[0.08177,0.28564,0.63259]`: the prediction changes to N.

This establishes that the fitted model is sensitive to that edit. It does not provide a newly measured biological label for the edited sequence. The original label belongs to the observed record. Sequence perturbation is a way to investigate a model’s behavior; a biological claim requires separate evidence.

**Figure H14 — An observed sequence and a model counterfactual.** Align the original and edited 60-character strings, mark exactly the two changed bases and show three aligned probability bars before/after. Keep the original observed label on the original row only. A text table contains the logits and probabilities.

**Investigation HC — Which context changes this model’s answer?** Start from the different validation source row 4. Record whether editing a chosen base or short span will change the leading class, and explain why. Edit any of the 60 characters, run the fixed saved model, and inspect the class outputs. Compare an edit near the boundary with an equally sized distant edit. You can also restrict the learned long filter to lags 0–4 or turn off its gates to investigate the internal route. Those controls modify the fitted computation; they do not retrain an alternative architecture. Restore the exact original to check that its result returns. An all-`N` input is a useful uncertainty probe: this model still strongly predicts class N, about 0.95478, illustrating why absence of informative input need not produce uniform probabilities.

A causal network can classify this *whole observed window* using its final position because that position has access to all 60 bases. It does not forecast a central boundary before the right-hand context arrives. If your application Show current outputs with each valid input change.

## 8. Deeper route: long filters, streaming and compact state

FFT convolution is attractive when the input segment is already available. Autoregressive generation presents a different workload: produce one new token, feed it back, then produce the next. Re-running a full FFT for each token wastes earlier work. But “therefore Hyena must re-run a full FFT” is too strong. There are several computational representations.

| Representation | What is retained | Work for an additional output, per channel | Appropriate question |
| --- | --- | --- | --- |
| Direct FIR, length $K$ | A buffer of the latest $K$ inputs | $O(K)$ | Is the filter short enough for a simple exact streaming implementation? |
| Whole-sequence FFT | A segment and transform workspaces | $O(P\log P)$ for the segment | Are enough input samples available to exploit parallel batch computation? |
| Blocked convolution | Input blocks and overlap/history | Depends on block and filter sizes | Can we trade buffering latency against throughput? |
| $d$-mode recurrent filter | $d$ state values | $O(d)$ | Does this filter have, or admit a good approximation by, a compact recurrence? |

These counts exclude projections, gates and feed-forward layers. A fixed-$K$ buffer is constant in the *total stream duration*, yet can still be large in $K$. An unrestricted implicit filter whose support grows with context does not automatically have a fixed small state.

### Blocks must include the overlapping tail

Split inputs `[1,2,3,4]` into `[1,2]` and `[3,4]`. Convolve each block with the filter, shift the second result by two positions, then **add the overlapping outputs**. This overlap-add construction gives the same linear convolution. If each chunk is filtered independently and only its first two outputs are retained, contributions crossing the chunk boundary vanish.

**Figure H15 — Overlap is information.** Put the full convolutions of the two blocks on separate rows, aligned to their original starts. Shade the overlap region and sum its contributions into the output strip. Contrast with an explicitly crossed-out reset-at-boundary result. When used for an autoregressive model, note that future input blocks cannot be computed before their tokens exist.

### Complete the blocked FFT route before choosing a package

The preceding `overlap_add` oracle uses direct convolution in each block. It makes the overlap ledger clear but does not acquire FFT complexity just because the blocks are small. [blocked_convolution.py](blocked_convolution.py) supplies the corresponding efficient transform route: transform the kernel once, zero-pad each input block to prevent circular wraparound, multiply spectra, inverse-transform, and add the entire valid tail into its correct global positions. It never constructs a dense Toeplitz matrix.

Keep it beside [convolution_mechanisms.py](convolution_mechanisms.py), then run `python blocked_convolution.py` with NumPy 2.3.5 and SciPy 1.18.1. The ordinary comparison uses `scipy.signal.oaconvolve(values, kernel, mode="full")[:len(values)]`. The `same` mode is centered, so replacing the causal slice by `mode="same"` changes alignment. [SciPy's mode and overlap-add contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html).

```python
"""Bounded FFT blocks, a reusable filter spectrum, and the ordinary SciPy route.

NumPy 2.3.5 and SciPy 1.18.1 targets. Real, finite, nonempty 1-D arrays.
The output is the first len(values) samples of linear causal convolution.
"""
import numpy as np
from scipy.signal import oaconvolve
from convolution_mechanisms import direct, overlap_add


def overlap_add_fft(values, kernel, block_size):
    values, kernel = np.asarray(values, dtype=float), np.asarray(kernel, dtype=float)
    if values.ndim != 1 or kernel.ndim != 1 or min(len(values), len(kernel)) == 0:
        raise ValueError("Need two nonempty vectors")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    size = 1 << (block_size + len(kernel)-2).bit_length()
    filter_spectrum = np.fft.rfft(kernel, n=size)
    # All requested output samples must be retained; no all-pairs Toeplitz matrix.
    output = np.zeros(len(values))
    for start in range(0, len(values), block_size):
        block = values[start:start+block_size]
        transformed = np.fft.rfft(block, n=size)
        convolved = np.fft.irfft(transformed*filter_spectrum, n=size)
        count = min(len(block)+len(kernel)-1, len(values)-start)
        output[start:start+count] += convolved[:count]
    return output


def main():
    values = np.array([2., -1., 3., 0., 1., 2., -.5])
    kernel = np.array([.5, 1., -.25, .125])
    expected = direct(values, kernel)
    library = oaconvolve(values, kernel, mode="full")[:len(values)]
    np.testing.assert_allclose(library, expected, atol=1e-12)
    for block_size in (1, 2, 3, 8):
        actual = overlap_add_fft(values, kernel, block_size)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(actual, overlap_add(values, kernel, block_size), atol=1e-12)
        print("block", block_size, "output", actual, "maximum error", abs(actual-expected).max())


if __name__ == "__main__":
    main()
```

For input length N, filter length M, block size B and transform length F≥B+M−1, work is O(F log F + ceil(N/B) F log F). Retained workspace beyond the O(N) returned output is O(F); the transformed kernel is reused. This bound describes this program, not a measured speed victory. Very short kernels can favor direct convolution; unsuitable B can waste work. The authoring probe's seven-sample fixture gives `[1, 1.5, 0, 3.5, −.375, 2.375, 1.5]` at B=1,2,3,8, with maximum observed float64 difference 4.45×10⁻¹⁶ from the direct oracle. No training run or speed measurement was added.

The trainable ordinary workflow remains `splice_models.py::causal_convolution`, `ImplicitFilter` and `SequenceBlock`: Torch FFT operations preserve the gradient path into the filter-generating network and both input-dependent gates. NumPy/SciPy are useful numeric references, not differentiable replacements inside that Torch training graph. The two-gate block is an explicitly scoped Hyena-style operation; StripedHyena's other filters/attention layers and pretrained checkpoints remain distinct family examples.

**Change the contract.** Use a ten-sample signal, a five-tap signed filter and B=3; compare the whole output including the incomplete last block. Then change a future input only.

<details><summary>Hint and reasoned solution</summary>

Choose F≥7, hence F=8 for this power-of-two implementation. The final input block contains one real sample; its convolution still has up to five valid terms, of which only positions inside the requested ten-sample output are returned. Add tails rather than overwrite them. A changed input at index j must leave output indices below j unchanged to rounding precision. A centered `same` slice or an undersized transform can violate that causality. The all-zero kernel is a useful null; B>N must still match the direct oracle. To stream output instead of retaining N samples, emit only a block's finalized prefix and carry the overlapping tail, taking care when M−1 exceeds B.

</details>

### Some long filters are exactly recurrent

For $h_r=a^r$, define $s_t=a s_{t-1}+u_t$, with $s_{-1}=0$. Expanding the recurrence gives $s_t=u_t+a u_{t-1}+a^2u_{t-2}+\cdots$. That is exactly convolution with the exponential filter. The past has been compressed into one number because this particular filter has algebraic structure.

A sum of exponentials uses several modes:

\[
h_r=\sum_{n=1}^{d}R_n\lambda_n^r,\qquad
s_{n,t}=\lambda_n s_{n,t-1}+u_t,\qquad
y_t=\sum_n R_n s_{n,t}.
\]

Each $\lambda_n$ determines a mode’s retention and oscillation; $R_n$ determines its contribution. A negative real pole alternates sign. Complex-conjugate pairs can produce real decaying oscillations. For an indefinitely sustained stable filter, poles inside the unit circle give decaying modes; a finite FIR remains well-defined without that infinite-horizon requirement.

Take $R=[0.6,0.4]$, $\lambda=[0.5,-0.25]$. The first six coefficients are `[1,0.2,0.175,0.06875,0.0390625,0.018359375]`. For inputs `[1,-2,0.5,3,-1,2]`, the output is `[1,-1.8,0.275,2.81875,-0.4109375,2.299609375]`.

**Figure H16 — Two modes, one long echo.** Draw one fading positive mode and one alternating mode; their weighted sum forms the filter. Below, a two-register state machine shows the update and read at each input. The mode curves and state values are exact calculations, not fitted experimental results.

This complete NumPy program computes the recurrent output and independently checks convolution:

```python
import numpy as np

values = np.array([1., -2., .5, 3., -1., 2.])
residues = np.array([.6, .4])
poles = np.array([.5, -.25])
state = np.zeros(2)
outputs = []
for value in values:
    state = poles * state + value
    outputs.append(residues @ state)
kernel = (residues[:, None] * poles[:, None] ** np.arange(len(values))).sum(0)
reference = np.convolve(values, kernel)[:len(values)]
print(np.round(outputs, 9))
print(np.allclose(outputs, reference, atol=1e-12))
```

After the first three inputs, the state is `[-0.25,1.0625]`. Carry that state into the remaining inputs to obtain the same continuation. Resetting it gives the different suffix `[3,-0.4,2.325]`. State must travel across chunks just as convolution tails must travel across blocks.

Inside a gated block, these modes accumulate the **gated value** $k_t v_t$; $q_t$ multiplies the read afterward. Cache the short-convolution history as well. An attention layer elsewhere in a hybrid does not execute these updates on behalf of the convolution layers.

### When the recurrence is an approximation

A general filter network need not output a small sum of exponentials. [Laughing Hyena Distillery](https://proceedings.neurips.cc/paper_files/paper/2023/file/371355cd42caaf83412c3fbef4688979-Paper-Conference.pdf) studies converting trained long filters into compact state-space approximations. The teaching idea is: choose a target state size, fit a recurrent filter to the original coefficients or transfer function, then check both the approximation and the model using it.

An elementary error bound explains what must be controlled. If $e=h-\hat h$ is the filter error, then

\[
|y_t-\hat y_t|
 =\left|\sum_{j\le t}e_{t-j}u_j\right|
 \le\|u\|_\infty\sum_{r\ge0}|e_r|
 =\|u\|_\infty\|e\|_1.
\]

For one fixed-gate sandwich, multiply this bound by $\|q\|_\infty\|k\|_\infty$. Later nonlinear layers and changing internal inputs require additional analysis; a small filter error alone is not a universal guarantee of unchanged predictions.

Truncate our six-coefficient example after lag 1. Its omitted coefficient sum is 0.301171875 and the input’s largest magnitude is 3, so the bound is 0.903515625. The observed maximum output change is 0.499609375. The bound is deliberately conservative and the truncation is deliberately visible. Both filters can run quickly, but they do not compute the same result.

**Figure H17 — Approximation error has units and a route.** Align original and truncated coefficient stems, shade the omitted absolute mass, then show observed output error and the bound on one numeric axis. Label this finite six-position calculation; do not imply that it bounds an unmeasured infinite tail.

For a deeper connection, extend this known analytic two-mode filter through lag 10 and arrange its coefficients into a **Hankel matrix**, whose entries are constant on anti-diagonals: $A_{ij}=h_{i+j}$. For a sum of $d$ exponential modes, $A_{ij}=\sum_n R_n\lambda_n^i\lambda_n^j$, so it is a sum of at most $d$ rank-one matrices. Our 6×6 example has two substantive singular values, about 1.08698 and 0.13949; the others are numerical roundoff below $10^{-16}$. A rapidly decaying Hankel spectrum suggests that a smaller state may approximate a filter. It does not say that every neural filter has low rank or that one finite matrix certifies all future lags. The full system-realization theory refines the indexing, assumptions and minimal-state statement.

**Investigation HD — Preserve the past, or approximate it deliberately.** Use fresh inputs `[2,0,-1,3,1,-0.5]` with the two-mode filter. Observe whether changing a chunk boundary, resetting the state or truncating the filter should change the output. Edit inputs, residues and poles; compare direct convolution, full recurrence and carried chunks. Now choose a shorter finite approximation and compare its observed error with the finite-horizon bound. Design a nonzero example with zero truncation error at a chosen position, and explain why that does not establish equality everywhere.

## 9. Deeper route: what the family adds

### Different ways to specify a filter

The [S4/Mamba lesson](/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals) showed how a time-invariant state-space system yields a convolution kernel. Hyena often starts with a direct neural function of lag. These parameterizations impose different structure. Once a model makes its state transition or input maps depend on the current token, the overall operator need not be one fixed convolution. Mamba’s selectivity and Hyena’s surrounding gates should be described by their actual equations rather than called interchangeable implementations.

Continuous kernel convolution, or CKConv, developed the idea of evaluating a learned kernel function at relative positions. S4 supplied a structured state-space route to long kernels. H3 combined short shifts, longer state-space filters and multiplicative paths to support content-dependent mechanisms. SaShiMi explored multiscale state-space sequence modeling for audio. Hyena belongs to this set of ideas; the architectural differences concern which filters, gates, resolutions and states are used. [CKConv](https://arxiv.org/abs/2102.02611), [H3](https://arxiv.org/abs/2212.14052) and [SaShiMi](https://arxiv.org/abs/2202.09729) are useful primary pointers for those branches.

For ordinary row-softmax attention, $A_{tj}\propto\exp(q_t^\top k_j/\sqrt{d_k})$ over allowed positions. Its coefficients are nonnegative and normalized. Hyena’s signed distance-and-gate coefficients generally compute a different operator. [Sparse and linear attention](/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals) distinguishes changing an operator from finding a different algorithm for the same one.

**Figure H18 — Mechanism map, not a leaderboard.** Compare a lag-only kernel, gate-filter-gate sandwich, input-selective state update and normalized attention. Each row displays the equation and retained object, with no fabricated “quality,” “speed” or “best for everything” score.

### Why genomics is an interesting application

Single DNA-base resolution and long context can both matter: a local change and a distant regulatory region may affect what a model should infer. HyenaDNA pretrained a causal decoder on the human reference genome, using individual nucleotide symbols and special tokens, then adapted it to downstream tasks. Its method includes length warm-up, task-specific supervised adaptation and learned soft prompts. Soft prompts are trainable input vectors; they are not newly observed DNA bases. The paper also studied a separately implemented bidirectional ablation. Its main pretrained architecture was not bidirectional. [HyenaDNA](https://arxiv.org/html/2306.15794v2)

The practical workflow is to distinguish the pretraining task from your labeled downstream question, preserve the chosen tokenizer and positional conventions, train or validate the downstream head, and establish an appropriate biological split. A pretrained backbone does not arrive with a valid classifier for every possible new label. Our 60-base experiment demonstrates the trainable mechanism and model inspection; it does not reproduce the paper’s long-context experiments.

### StripedHyena and StripedHyena 2

StripedHyena’s 2023 release combined attention, gated convolutions and feed-forward layers. Its recurrent convolution representation and its attention KV cache coexist during generation. The [original release article](https://www.together.ai/blog/stripedhyena-7b) describes the architecture and the measured workloads of that release; its historical timing claims are not timings for the small program here.

The 2025 StripedHyena 2 work takes a more specific mixture: short explicit filters for local mixing, medium explicit filters with decay regularization, and long implicit filters expressed using exponential modes, interleaved with attention. The long modal filters permit recurrent evaluation; finite short/medium filters retain bounded input history. Filter sharing across channel groups helps organize hardware-efficient computation. These are deliberate choices of memory range and algorithm, rather than an assertion that every layer should use the longest possible filter. The work supports the Evo 2 genomic model family. [StripedHyena 2 methods](https://arxiv.org/html/2503.01868v1)

**Figure H19 — A striped architecture with different ranges.** Use a repeating operator strip: short, medium, long and occasional attention. Under each type, show its filter representation and decode-state object. It is a conceptual architecture map; the caption must not present it as an exact official layer schedule unless a particular verified configuration is selected.

### Complexity is a guide to what to measure

For $D$ channels and a fixed small number of long-filter stages, convolution mixing has $O(DL\log L)$ work, while dense input/output projections and feed-forward updates add $O(LD^2)$. Filter generation also costs work at every evaluated position. Attention’s pairwise mixing has $O(L^2D)$ work, but optimized exact attention need not materialize an $L\times L$ score array in main device memory. A memory-efficient attention algorithm and a different sequence operator answer different questions.

An operation-count sketch such as $L^2/(L\log_2 L)=L/\log_2 L$ is not a speedup measurement. FFT workspaces, complex arithmetic, dtype, kernel fusion, hardware utilization, batch size and projections all affect wall time. [FlashFFTConv](https://arxiv.org/abs/2311.05908) investigates why FFT-based sequence convolutions need hardware-aware algorithms; later multi-hybrid work also uses blocked direct convolution for selected filter lengths.

**Figure H20 — From an equation to a benchmark protocol.** Separate three columns: asymptotic mixing work, actual tensor/storage inventory, and measured workload. A learner fills the workload card with sequence length, width, batch, precision, implementation/version, device, forward/backward/prefill/decode scope and warm-up. Leave timing cells empty until there is an actual measurement.

When diagnosing a model, distinguish an operator bug from a modeling problem. Padding, lag direction, shifted targets, short-filter causality and moving position coordinates can change the intended computation. A correct model can still underfit, overfit, exploit a shortcut or fail on a new context length. Inspect the data roles, learning curves and changed-input behavior before attributing a result to the architecture family.

The next topic in this module is [Ring Attention & Sequence Parallelism](/learn/path/full-curriculum/ring-attention-sequence-parallelism?module=deep-learning-fundamentals). It changes how attention work is distributed across devices. Keep that distinction: Hyena changes the sequence mixer; Ring Attention distributes an attention computation. Later, [Hybrid SSM–Transformer Architectures](/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals) combines state and attention paths in one model.

## 10. Practice: reason, calculate and transfer

Try each task before opening its hint. The first six establish the core route. The final four develop architecture and streaming reasoning. A successful answer explains which computation or information boundary produced its result.

### 1. A filter with a negative echo

For input `[2,0,-1,3]` and filter `[1,-0.5,0.25]`, compute the first four causal outputs. Which lag is absent at position 1, and why?

<details><summary>Hint</summary>
Write current, one-step-old and two-step-old contributions separately. Inputs before the sequence start are zero.
</details>

<details><summary>Solution</summary>
The outputs are `[2,-1,-0.5,3.5]`. At position 1 there is no two-step-old input, so the lag-2 term contributes zero. At position 3 the contributions are $3+(-0.5)(-1)+0.25(0)=3.5$. A negative filter coefficient can increase the output when the input it multiplies is also negative.
</details>

### 2. Padding is part of the mathematics

A length-6 signal is convolved with a length-4 filter. What is the smallest sufficient transform length? What convenient power-of-two length could you choose? Does keeping only the first six results make an unpadded length-6 transform causal?

<details><summary>Hint</summary>
Count the support of the full linear convolution before deciding what to crop.
</details>

<details><summary>Solution</summary>
The full length is $6+4-1=9$. Any supported FFT length at least 9 suffices; 16 is a convenient power of two. Cropping a circular convolution does not remove tail contributions that have already wrapped into the first six positions. Padding must prevent that aliasing before the transform-domain product is inverted.
</details>

### 3. Change the sender, then the receiver

For $v=[2,1,-1]$, $h=[1,0.5,0.25]$, $k=[1,0,2]$, and $q=[1,-1,0.5]$, calculate the output. If $k_1$ becomes 1, which outputs change? What happens if instead only $q_2$ becomes zero?

<details><summary>Hint</summary>
First compute $kv$; only then convolve and multiply by $q$.
</details>

<details><summary>Solution</summary>
$kv=[2,0,-2]$, the convolution is `[2,1,-1.5]`, and the output is `[2,-1,-0.75]`. Opening $k_1$ gives `[2,-2,-0.5]`: positions 1 and 2 change, while position 0 is unaffected. Setting only $q_2=0$ erases only this block’s final output, giving `[2,-1,0]`. In a full residual network other paths can still carry information.
</details>

### 4. Is the matrix an explanation of everything?

An author plots $H(q(u),k(u))$ and says its entry $H_{tj}$ is the derivative of the final network output with respect to input $u_j$. Identify the missing terms and propose an accurate caption.

<details><summary>Hint</summary>
The input affects more than the value stream. Consider the product rule and later layers.
</details>

<details><summary>Solution</summary>
Changing $u_j$ can change $q$, $k$, $v$, normalization and later updates. The full derivative includes these dependencies and all paths to the final output. A suitable caption is: “Conditional mixing coefficients of this channel with its current gates held fixed.” To study final sensitivity, compute or perturb the full model separately and state what that analysis measures.
</details>

### 5. A classifier has learned something—but what?

A positional linear baseline beats a gated sequence model on a centered splice task. Give two reasonable explanations and one additional evaluation that would answer a different, clearly stated question. Why should you not keep trying variants against the same assessment labels?

<details><summary>Hint</summary>
Use the fixed boundary, optimization and data-role definitions. A larger model is not the only possible change.
</details>

<details><summary>Solution</summary>
The baseline can directly weight each known motif position; the sequence model must learn a useful transformation and may overfit or optimize less effectively with this data budget. A separate, appropriately constructed gene-family-held-out dataset could ask about transfer beyond related biological sequences. Alternatively a newly designed variable-boundary task could ask whether a model locates an unknown site, with labels and input availability defined accordingly. Repeatedly choosing variants based on assessment performance turns that set into validation data, so its original evaluation role is lost.
</details>

### 6. An unchanged prefix should stay unchanged

A model uses a causal FFT and recalculates position features as $r/(L-1)$ for every input length. Appending tokens changes early outputs. Explain the bug and give two checks that distinguish it from ordinary float roundoff.

<details><summary>Hint</summary>
Compare the actual coefficient vectors used for a prefix alone and for that prefix inside a longer sequence.
</details>

<details><summary>Solution</summary>
The lag coordinate changes with active length, so the filter is different even at old lags. Fix a reference coordinate system and slice it consistently. Check that the old filter coefficients agree across prefix lengths, then compare full-network prefix outputs against a direct causal implementation at an appropriate numerical tolerance. Also alter only future token values at a fixed length. This separates a moving-coordinate bug from future-value leakage and small arithmetic differences.
</details>

### 7. A long filter with one state

For $h_r=0.75^r$ and input `[2,-1,0,1]`, compute the recurrent state/output. Continue after the second input using the saved state; compare with resetting it.

<details><summary>Hint</summary>
Apply $s_t=0.75s_{t-1}+u_t$, starting from zero.
</details>

<details><summary>Solution</summary>
The outputs are `[2,0.5,0.375,1.28125]`. The state after the second input is 0.5. Carrying it gives the correct suffix `[0.375,1.28125]`; resetting gives `[0,1]`. A state is not optional bookkeeping: it encodes earlier contributions to later outputs.
</details>

### 8. A finite approximation bound

Over the horizon of interest, an original filter is `[1,0.4,0.2,0.1]` and an approximation is `[1,0.4,0,0]`. Inputs are bounded by magnitude 2. Bound the maximum convolution-output error. How does the bound change for fixed sending gates of magnitude at most 3 and receiving gates at most 0.5?

<details><summary>Hint</summary>
Use the absolute omitted coefficient mass. Keep gates separate from inputs.
</details>

<details><summary>Solution</summary>
The omitted mass is 0.3, so the convolution bound is $2\times0.3=0.6$. The fixed-gate bound is $0.5\times3\times0.6=0.9$. Cancellation can make the observed error smaller. This is a bound for the stated finite filter and fixed path, not for an unmeasured infinite extension or arbitrary downstream network.
</details>

### 9. A copy-task shortcut

An experiment always asks the model to copy each input token exactly five positions later. Construct a filter that solves it without content-dependent gates. Redesign the task to test content-dependent addressing instead.

<details><summary>Hint</summary>
An impulse response can be a pure delay. A key-value query can require different delays in different examples.
</details>

<details><summary>Solution</summary>
Set $h_5=1$ and every other coefficient to zero; then $y_t=u_{t-5}$, channel by channel. Use zero initial history. For addressing, sample new key-value associations per example, vary their order and distances, and query one of the keys at the end. Hold out newly generated examples and compare controlled baselines. Fixed-delay success alone cannot establish the harder capability; conversely this construction does not prove that every ungated deep network must fail the redesigned task.
</details>

### 10. Plan an honest efficiency comparison

You have an FFT sequence mixer and an optimized attention implementation. Design separate experiments for full-sequence training and autoregressive decoding. What changes when the convolution is replaced by a modal approximation?

<details><summary>Hint</summary>
Separate the mathematical operator, the workload and the implementation. Include prediction quality in an approximation study.
</details>

<details><summary>Solution</summary>
Fix model dimensions, input lengths, batch, dtype, device and implementation versions. For training, measure the specified forward/backward/update scope with warm-up and synchronization, recording peak storage and excluding setup consistently. For decoding, separate prompt prefill from incremental steps, specify generated length and cache/state handling, and record latency or throughput with its batch. A modal approximation changes the filter unless exact equivalence is established; report filter error and relevant output/task differences as well as runtime and memory. Do not divide asymptotic expressions and label the result a measured speedup.
</details>

## 11. References and another way to learn

These links supplement the self-contained route above. Read a resource for a particular question rather than treating every linked model as a prerequisite.

- [Hyena Hierarchy — original paper](https://arxiv.org/html/2302.10866v2): the formal hierarchy, synthetic task design, filter parameterization and matrix interpretation. Read §3 after the gate example; Appendix B derives the conditional matrix, and Appendix D explains frequency-rich initialization.
- [Hazy Research’s illustrated Hyena introduction](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena): a more informal creator explanation of why gating and long filters were combined. Its reported comparisons describe the 2023 experiments.
- [ICML 2023 Hyena presentation page](https://icml.cc/virtual/2023/poster/24143): the conference page identifies the paper, authors, poster and a video section. It offers a presentation route alongside the article. The page and identity were verified; the recording was not watched for this manuscript and playback availability may depend on the host.
- [HyenaDNA paper](https://arxiv.org/html/2306.15794v2) and [standalone author implementation](https://github.com/HazyResearch/hyena-dna/blob/main/standalone_hyenadna.py): read method §3 for the practical gate-filter-gate operator, tokenization, length warm-up and adaptation. In code, inspect `fftconv`, `PositionalEmbedding`, `HyenaFilter` and `HyenaOperator` with the actual maximum-length contract.
- [Laughing Hyena Distillery](https://proceedings.neurips.cc/paper_files/paper/2023/file/371355cd42caaf83412c3fbef4688979-Paper-Conference.pdf): the deeper path from a trained convolution to a compact recurrence, including approximation objectives, modal interpolation, Hankel spectra and deployment. Our two-mode example supplies the prerequisite intuition.
- [StripedHyena release](https://www.together.ai/blog/stripedhyena-7b) and [StripedHyena 2 methods](https://arxiv.org/html/2503.01868v1): read the architectural design and storage contracts before the dated benchmark tables. The second paper explains short, medium and long filters plus blocked/context-parallel algorithms.
- [FlashFFTConv](https://arxiv.org/abs/2311.05908): an advanced systems branch about tensor-core use and memory traffic in sequence convolutions. This is a primary pointer, not a reproduced benchmark here.
- [PyTorch real FFT reference](https://docs.pytorch.org/docs/main/generated/torch.fft.rfft.html): normalization, dimensions, padding/trim behavior and supported device/dtype combinations. Consult the documentation matching your installed version.
- [UCI splice data and license](https://archive.ics.uci.edu/dataset/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences), [NHGRI RNA background](https://www.genome.gov/about-genomics/educational-resources/fact-sheets/ribonucleic-acid-fact-sheet), and [this packet’s provenance](data-provenance.md): distinguish biological meaning, historical data and our exact experimental split.
- [Complete experiment](splice_models.py), [mechanism calculations](convolution_mechanisms.py), and [saved-fit inspection program](author_calculations.py): runnable companions for reproducing the numbers and investigating your own inputs. After running the training program, place these companions beside it and run `python convolution_mechanisms.py` or `python author_calculations.py`; the mechanism companion also needs SciPy (`python -m pip install scipy`). They contain the complete procedures, not pseudocode for unreported model training.
