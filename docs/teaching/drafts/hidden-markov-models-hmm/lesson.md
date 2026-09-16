# Hidden Markov Models: Infer the Process Behind a Sequence

A machine changes between operating conditions, but its sensor readings do not announce the condition directly. A low reading is evidence, not a label. Several readings in order can tell you more than any one reading alone.

A **hidden Markov model**, or **HMM**, describes that situation using two connected sequences: a hidden state that evolves, and an observation produced at each step. It lets us ask how likely the observations are, what states could explain them, and how to learn the model's probabilities.

The key picture is a row of hidden-state nodes with observations hanging underneath. The key computation is a **trellis**: all possible states laid out across time, with shared partial calculations replacing an enormous list of complete paths.

**First pass:** work through the model, forward/filtering, backward/smoothing and best-path examples in sections 1–5. Then follow one expected-count update and the real tagging comparison. The implementation, duration models and structured extensions add depth after that foundation. Practice includes separate arithmetic, explanation and changed-input problems.

The preceding [AutoML/NAS lesson](/learn/path/full-curriculum/automl-neural-architecture-search-nas?module=classical-ml) considered how to select a learning procedure. Here we examine one particular model closely. The earlier [GMM lesson](/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml) is a useful connection: a mixture assigns a latent component to each observation; an HMM makes those assignments dependent across time.

## 1. Separate the thing you see from the state you infer

For a small constructed example, imagine receiving a friend's daily activity report while not seeing the weather. Use two hidden states, Rainy and Sunny, and three observations, Walk, Shop and Clean. These are deliberately simplified probabilities, not measured weather data:

| Initial state | Probability |
|---|---:|
| Rainy | 0.6 |
| Sunny | 0.4 |

| Current state → next state | Rainy | Sunny |
|---|---:|---:|
| Rainy | 0.7 | 0.3 |
| Sunny | 0.4 | 0.6 |

| State → reported activity | Walk | Shop | Clean |
|---|---:|---:|---:|
| Rainy | 0.1 | 0.4 | 0.5 |
| Sunny | 0.6 | 0.3 | 0.1 |

Each row is a separate distribution and sums to one. Rainy→Sunny is a transition between states. Rainy→Clean is an emission: an observation conditional on a state. Neither arrow gives the reverse probability. In particular, \(P(\mathrm{Clean}\mid\mathrm{Rainy})=0.5\) does not imply \(P(\mathrm{Rainy}\mid\mathrm{Clean})=0.5\).

Generate a sequence by drawing the initial state, drawing its activity, moving to a new state using the current state's transition row, and repeating. The observer sees the activity row of the story; inference reasons about the hidden row.

The **first-order Markov assumption** says that, given the current hidden state, the next state does not additionally depend on earlier states. A separate **emission assumption** says observations factor independently once the entire state sequence is fixed. Observations can still be correlated marginally: a persistent hidden state can produce a run of similar observations.

This distinction matters in sensor data. Consecutive readings being correlated is not by itself a violation. Correlation that remains after conditioning on the modeled states can reveal a missing dependency, an inadequate state representation, or an unsuitable emission model.

Let \(z_t\) be the hidden state and \(o_t\) the observed symbol at time \(t\), starting at \(t=0\). Write the initial probabilities as \(\pi\), transitions as \(A\), and emission probabilities as \(B\). For \(N\) states and \(M\) symbols, their shapes are \(N\), \(N\times N\), and \(N\times M\).

The probability of one complete hidden path and its observations is

\[
P(z_{0:T-1},o_{0:T-1})
=\pi_{z_0}B_{z_0,o_0}
\prod_{t=1}^{T-1}A_{z_{t-1},z_t}B_{z_t,o_t}.
\]

Read this as “start, emit, transition, emit, transition, emit.” Multiplication follows one possible story; adding over different stories accounts for uncertainty. Parameters are fixed throughout these inference calculations. Learning them is a later operation.

**Picture this:** two horizontally aligned rows, hidden states above and observed activity cards below. Highlight one path and the exact table entry used by every arrow. A timeline makes the number of transitions visible: four observations contain three within-sequence transitions.

## 2. The same observations support different questions

Suppose the reports are **Walk → Shop → Walk → Clean**. Before calculating, choose the question:

| Question | Quantity | Available observations |
|---|---|---|
| How well does the model explain the reports? | \(P(o_{0:T-1})\) | The specified sequence |
| What is the current state after this report? | \(P(z_t\mid o_{0:t})\) | Past and present: **filtering** |
| What was an earlier state, using the later reports too? | \(P(z_t\mid o_{0:T-1})\) | Entire sequence: **smoothing** |
| What is the next state likely to be? | \(P(z_{t+1}\mid o_{0:t})\) | Past and present: **prediction** |
| What single whole path has greatest probability? | \(\arg\max_z P(z\mid o)\) | Entire sequence: **Viterbi decoding** |
| What probabilities should the model use? | Estimate \(\pi,A,B\) | Training sequences: **learning** |

Filtering and smoothing are not interchangeable in a live system. A dashboard operating on Tuesday cannot use a Thursday reading. A retrospective analyst can. A more informed posterior also does not promise a more accurate label on every individual example.

For a fixed observed sequence with positive probability, maximizing \(P(z\mid o)\) is equivalent to maximizing the joint \(P(z,o)\), because all paths share the denominator \(P(o)\). Maximizing \(P(o\mid z)\) alone would omit the path prior and can choose a different answer.

## 3. Forward computation: add the paths without listing them

Four time steps with two possible states each give \(2^4=16\) possible paths. A thousand steps would give \(2^{1000}\). We need to share work.

Define

\[
\alpha_t(j)=P(o_0,\ldots,o_t,z_t=j).
\]

This is **joint probability mass**, not yet a normalized posterior over states. At the first Walk report:

\[
\alpha_0(R)=0.6(0.1)=0.06,\qquad
\alpha_0(S)=0.4(0.6)=0.24.
\]

The observation has probability \(0.06+0.24=0.30\). Dividing by that total gives the filtered belief: 20% Rainy, 80% Sunny.

At the next report, Shop, Rainy can be reached from either earlier state:

\[
\alpha_1(R)
=[0.06(0.7)+0.24(0.4)]\times 0.4=0.0552.
\]

The bracket adds the mass arriving along both arrows; the last factor accounts for the activity at the destination. Similarly,

\[
\alpha_1(S)
=[0.06(0.3)+0.24(0.6)]\times 0.3=0.0486.
\]

The general recurrence repeats this operation:

\[
\alpha_0(j)=\pi_jB_{j,o_0},\qquad
\alpha_t(j)=B_{j,o_t}\sum_i\alpha_{t-1}(i)A_{ij}.
\]

| Time and report | Forward mass Rainy | Forward mass Sunny | Filtered Rainy |
|---|---:|---:|---:|
| 0 Walk | 0.06000000 | 0.24000000 | 0.200000 |
| 1 Shop | 0.05520000 | 0.04860000 | 0.531792 |
| 2 Walk | 0.00580800 | 0.02743200 | 0.174729 |
| 3 Clean | 0.00751920 | 0.00182016 | 0.805109 |

Sum the last two masses:

\[
P(\mathrm{Walk,Shop,Walk,Clean})=0.00933936.
\]

The Rainy mass increases at the last step even though the *total prefix probability* decreases. Contributions have moved between states; not every individual cell must shrink monotonically.

In a trellis, every destination cell has incoming transition-weighted contributions. Use edge widths or labeled amounts for those contributions, then show their sum and destination emission. A single highlight should follow the same arithmetic across the table and drawing.

### Filtering and forecasting without retaining the whole past

Let \(f_t\) be the normalized filtered row vector. First predict the next state:

\[
q_{t+1}=f_tA.
\]

Then, when the new report arrives, multiply \(q_{t+1}\) by its emission column and normalize. After the first Walk, \(f_0=[0.2,0.8]\), so the next-state prediction is

\[
[0.2,0.8]A=[0.46,0.54].
\]

The predicted probability of the next report being Clean is \(0.46(0.5)+0.54(0.1)=0.284\). Forecasting an observation requires both the state transition and the emission distribution. Jumping directly from the current most likely state discards uncertainty.

For a fixed model, the current filtered vector summarizes the observation history needed for the next filtering update. Storing every earlier forward vector is unnecessary if this is the only query.

## 4. Backward computation: later evidence changes an earlier belief

At time 1, filtering slightly favors Rainy: 0.531792. But if we later see Walk and Clean, the smoothed Rainy probability is 0.433828. The later observations changed the earlier conclusion.

Define a backward likelihood:

\[
\beta_t(i)=P(o_{t+1},\ldots,o_{T-1}\mid z_t=i).
\]

At the last time there are no later reports, so \(\beta_{T-1}(i)=1\). Step backward using

\[
\beta_t(i)=\sum_j A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j).
\]

At time 2, the only future report is Clean:

\[
\beta_2(R)=0.7(0.5)+0.3(0.1)=0.38,\qquad
\beta_2(S)=0.4(0.5)+0.6(0.1)=0.26.
\]

Multiply the evidence from the left and right and normalize:

\[
\gamma_t(i)=P(z_t=i\mid o_{0:T-1})
=\frac{\alpha_t(i)\beta_t(i)}{P(o_{0:T-1})}.
\]

| Time | Filtered Rainy | Smoothed Rainy |
|---|---:|---:|
| 0 | 0.200000 | 0.194943 |
| 1 | 0.531792 | 0.433828 |
| 2 | 0.174729 | 0.236316 |
| 3 | 0.805109 | 0.805109 |

The last row agrees because no later observations remain. Smoothing uses the whole sequence, but it does not reveal a verified hidden truth: these are probabilities under the model.

**Investigation: change the future, preserve the past calculation.** Predict what happens to the belief at time 1 if the final Clean card becomes Walk. Commit both a filtering prediction and a smoothing prediction before revealing them. The time-1 filtered value stays 0.531792; the smoothed value changes to 0.397624. The three earlier filtered rows are unchanged. The sequence itself is editable, so the distinction is visible in the objects, not just in two definitions.

A missing report is another useful contrast. At a retained time step with no observation, summing over all possible symbols gives emission likelihood one. The state still transitions. Removing the entire step instead changes elapsed model time and the number of transitions. In this example, replacing Shop with a missing report gives final Rainy probability 0.802780; deleting that step gives 0.795320. Treating missingness this way assumes the fact of missingness itself supplies no additional state evidence.

## 5. Viterbi: keep the best path, not the sum

The forward algorithm combines every path into a cell. Viterbi retains the greatest joint probability among paths ending there:

\[
\delta_t(j)=\max_{z_0,\ldots,z_{t-1}}
P(z_0,\ldots,z_{t-1},z_t=j,o_0,\ldots,o_t).
\]

Replace sum with maximum:

\[
\delta_0(j)=\pi_jB_{j,o_0},\qquad
\delta_t(j)=B_{j,o_t}\max_i[\delta_{t-1}(i)A_{ij}].
\]

Store the maximizing predecessor for each destination. Why can other prefixes be discarded? Two prefixes ending in the same current state have the same possible future factors. Multiplying both by the same nonnegative suffix cannot make the lower-probability prefix strictly better. The current state is the boundary that makes dynamic programming valid.

At the Shop step, the best Rainy prefix comes from Sunny:

\[
\max[0.06(0.7),0.24(0.4)]\times 0.4=0.0384.
\]

The best Sunny prefix has probability 0.0432. At the following Walk step, however, the best **Rainy** predecessor is Rainy:

\[
\max[0.0384(0.7),0.0432(0.4)]\times 0.1
=0.002688.
\]

Selecting the largest cell in the previous column without accounting for the particular transition would get this predecessor wrong.

| Time | Best Rainy prefix | Best Sunny prefix | Rainy predecessor | Sunny predecessor |
|---|---:|---:|---|---|
| 0 | 0.06 | 0.24 | — | — |
| 1 | 0.0384 | 0.0432 | Sunny | Sunny |
| 2 | 0.002688 | 0.015552 | Rainy | Sunny |
| 3 | 0.00311040 | 0.00093312 | Sunny | Sunny |

Start from the largest final cell and follow its stored predecessors backward. The best whole path is **Sunny → Sunny → Sunny → Rainy**.

Its joint probability is 0.0031104. Its posterior probability, conditional on the observed reports, is

\[
\frac{0.0031104}{0.00933936}\approx0.333042.
\]

“Most likely” does not mean “nearly certain”: about two-thirds of posterior mass belongs to other paths collectively. A path-probability bar beside the trellis can show the best path and the remaining mass without confusing joint and conditional probabilities.

### The most probable state at each time can form an impossible path

Pointwise decoding chooses \(\arg\max_i\gamma_t(i)\) separately at each time. It minimizes expected total per-position mistakes when predictions are unconstrained. Viterbi minimizes the chance of getting the entire sequence wrong under a whole-sequence 0–1 loss. They optimize different objectives.

Consider a separate two-step, three-state model whose only observation symbol has probability one in every state. All its information is in these joint path masses:

| First state → second state | A | B | C |
|---|---:|---:|---:|
| A | 0 | 0.20 | 0.20 |
| B | 0.35 | 0 | 0 |
| C | 0.25 | 0 | 0 |

Row sums are the initial probabilities \([0.40,0.35,0.25]\). Divide each nonzero row by its sum to obtain the transition matrix. At the first time, A has the greatest marginal probability, 0.40. At the second, A again leads with 0.60. Pointwise modes therefore return **A→A**, a transition that has probability zero.

Viterbi returns **B→A**, with probability 0.35. Its expected number of correct positions is \(0.35+0.60=0.95\), compared with \(0.40+0.60=1.00\) for the unconstrained but impossible pointwise output. Better expected position count and validity as a joint path are separate properties.

**Investigation: construct and repair a decoded route.** Select the marginal winners, inspect the actual absent edge, and then submit the highest-probability legal path. Change the starting probabilities to \([0.20,0.55,0.25]\) while keeping transitions fixed: both decoders now choose B→A. Relabeling the three state names consistently is a numerical null; changing transition probabilities is not.

If validity and expected position accuracy are both requirements, a constrained minimum-risk decoder can maximize \(\sum_t\gamma_t(z_t)\) over legal paths by another dynamic program. That is still a different objective from Viterbi's product of model factors.

## 6. Learning: replace invisible counts with expected counts

If training states are known, estimate transitions by counting adjacent state pairs and emissions by counting symbols within each state. Sentence or recording boundaries matter: the final state of one recording is not followed by the first state of an unrelated recording.

If states are not labeled, we cannot count one true path. **Baum–Welch** uses expectation-maximization: infer distributions over paths using current parameters, compute expected counts, then fit new probabilities from those counts.

The state posterior \(\gamma_t(i)\) contributes a fractional occupancy count. A transition requires a *joint pair posterior*, not the product of two marginal posteriors:

\[
\xi_t(i,j)=P(z_t=i,z_{t+1}=j\mid o)
=\frac{\alpha_t(i)A_{ij}B_{j,o_{t+1}}\beta_{t+1}(j)}{P(o)}.
\]

For each edge time, \(\sum_{ij}\xi_t(i,j)=1\); its row sums equal \(\gamma_t(i)\) and column sums equal \(\gamma_{t+1}(j)\). These relationships are useful numerical checks and make a count-flow diagram interpretable.

For \(K\) independent sequences with lengths \(T_k\), the updates are

\[
\pi_i^{\mathrm{new}}=\frac1K\sum_k\gamma^{(k)}_0(i),
\qquad
A_{ij}^{\mathrm{new}}=
\frac{\sum_k\sum_{t=0}^{T_k-2}\xi^{(k)}_t(i,j)}
{\sum_k\sum_{t=0}^{T_k-2}\gamma^{(k)}_t(i)},
\]

\[
B_{i,v}^{\mathrm{new}}=
\frac{\sum_k\sum_{t=0}^{T_k-1}\gamma^{(k)}_t(i)\mathbf1[o_t^{(k)}=v]}
{\sum_k\sum_{t=0}^{T_k-1}\gamma^{(k)}_t(i)}.
\]

The transition denominator excludes each sequence's last position because it has no outgoing within-sequence transition. The emission denominator includes it. Initial-state counts are divided by the number of sequences, not the total number of positions. These emission formulas assume every symbol is observed. When missing reports carry no state information and are marginalized out, update emissions using only observed positions in both numerator and denominator; transitions still include the retained time steps.

### One complete update, with a boundary you can move

Treat **Walk→Shop** and **Walk→Clean** as two independent recordings. Under the original parameters, the expected transition counts are

\[
\begin{bmatrix}
0.408329&0.073150\\
0.933322&0.585199
\end{bmatrix}.
\]

They sum to two, because each recording contains one transition. Row-normalizing gives

\[
A^{\mathrm{new}}\approx
\begin{bmatrix}
0.848072&0.151928\\
0.614626&0.385374
\end{bmatrix}.
\]

Expected initial-state counts are \([0.481478,1.518522]\), so \(\pi^{\mathrm{new}}\approx[0.240739,0.759261]\). Expected emission counts are

\[
\begin{bmatrix}
0.481478&0.531792&0.809859\\
1.518522&0.468208&0.190141
\end{bmatrix},
\]

which sum to four. Row-normalizing gives emission rows approximately \([0.264094,0.291692,0.444214]\) and \([0.697571,0.215083,0.087346]\).

The joint training log-likelihood of the two recordings increases from −4.728043 to −3.529108 after this update. Removing the boundary creates a *different* data model, with one start and three transitions, and changes the expected counts. It is not merely a storage optimization.

**Investigation: count what the model could have done.** Edit an activity in either recording or move the recording boundary. Predict the total start, emission and transition counts before revealing fractional flows and their normalized rows. Duplicating the whole two-recording dataset doubles every expected count but leaves the one-step updated probabilities unchanged. That is a useful exact null.

### What EM does and does not guarantee

An exact E-step and matching exact M-step make the observed training log-likelihood nondecreasing in exact arithmetic. The increase can be zero. This property does not promise a global maximum, semantic recovery, better development performance, or a useful state count. Approximate updates, added penalties, changed objectives and finite-precision computations require their own analysis.

The complete program generates twelve independent length-30 sequences from the constructed model, using data seed 71. Three declared starting seeds each receive forty EM updates:

| Starting seed | Initial training log-likelihood | After 40 updates |
|---:|---:|---:|
| 3 | −566.599897 | −390.253507 |
| 7 | −410.236336 | −390.630410 |
| 19 | −459.466552 | −392.284393 |

The generating parameters score −393.562196 on this particular finite sample. A fitted model can exceed that score by adapting to sample variation; higher training likelihood is not proof of recovering the generating parameters. All three full trajectories are retained, including the initial model and the model after each update.

With uniform state initialization, both emission rows remain identical. After the first update they equal the empirical symbol frequencies \([0.333333,0.352778,0.313889]\); subsequent iterations stay at approximately −395.091859. Symmetry has not been broken by simply iterating longer.

State labels can also be permuted without changing observation likelihood: permute \(\pi\), both axes of \(A\), and the rows of \(B\) consistently. That explains index ambiguity. It does not explain every poor fit or justify naming an unsupervised state “Rainy” just because its index is zero. If a state has zero expected occupancy, its unconstrained emission row is unidentified; the program retains that row rather than dividing by zero.


## 7. Make the computation reliable and reproducible

The [complete NumPy program](hmm-experiments.py) implements forward–backward, filtering, Viterbi with backtracking, expected counts, independent-sequence Baum–Welch, scaled forward inference and the real tagging experiment below. Save it beside [the supplied sequence data](ewt-sequences.json), then run:

~~~bash
python hmm-experiments.py
~~~

Python and NumPy are its only computational dependencies. It writes [calculated-inputs.json](calculated-inputs.json), including the actual trellis, posterior rows, count updates, training histories and real predictions. No fitted state or timing curve is invented for display.

| Function | What to follow while reading |
|---|---|
| infer | One forward sum, one backward sum, posterior normalization, then a separate maximum-and-backpointer recurrence |
| forward_scaled | Normalize each filtering row and preserve its predictive evidence factor |
| expected_counts | Add fractional events within each sequence; reset the initial event at every boundary |
| em_step | Normalize expected event counts into probability rows |
| fit_em | Record the objective for the initial parameters and after every complete update |
| real_tagging | Fit category counts on training sentences and compare declared decoders on development sentences |

Start by changing one observation and inspecting one computed row. Understanding that row is more useful than memorizing the whole program.

### Tiny probability or genuinely impossible event?

Repeated multiplication can underflow even when the mathematical probability is positive. In float64, \(0.3^{100}\approx5.154\times10^{-53}\) is still representable. But \(0.01^{400}=10^{-800}\) rounds to zero. Its log probability, \(400\log0.01\approx-1842.068074\), remains manageable.

In log-space, products become sums. Sums of probabilities use log-sum-exp:

\[
\log\sum_i e^{z_i}
=m+\log\sum_i e^{z_i-m},\qquad m=\max_i z_i.
\]

Handle an all-negative-infinity row explicitly: it represents a zero sum. Subtracting negative infinity from itself would create a numerical NaN. A structural zero transition stays negative infinity in log-space. Adding an arbitrary epsilon would change which paths the model permits.

The alternative is normalized scaling. Write \(b_j(o_t)=B_{j,o_t}\). If \(f_{t-1}\) is the filtered distribution, first compute

\[
u_t(j)=b_j(o_t)\sum_i f_{t-1}(i)a_{ij},\qquad
c_t=\sum_j u_t(j),\qquad f_t(j)=u_t(j)/c_t.
\]

At the first observation, time0, use \(\pi_jb_j(o_0)\). Here \(c_t=P(o_t\mid o_{0:t-1})\) for \(t\geq1\), and \(c_0=P(o_0)\), so \(\log P(o)=\sum_t\log c_t\). Our four activities give factors \(0.3,0.346,0.320231,0.280968\), whose product is \(0.00933936\). Some texts define the scale factor as the reciprocal; their final log formula then carries a minus sign. Compare definitions before comparing code.

If every state assigns zero probability to an observed symbol, the observation sequence really is impossible under the model. Neither scaling nor log-space should convert it into a valid posterior. Check whether a hard constraint was intended, or whether a training vocabulary needs a deliberately learned unknown-symbol category. That is a modeling decision.

### Use the library without changing the question

The optional [hmmlearn examples](hmmlearn-examples.py) show a fixed categorical model, independent-sequence fitting and a Gaussian-emission model. They target the [0.3.3 API](https://hmmlearn.readthedocs.io/en/0.3.3/tutorial.html). This dependency was unavailable during authoring, so these complete examples are **not claimed to have been executed**. Phase two must run and verify them before presenting native outputs.

A categorical observation is an integer symbol, stored in a \(T\times1\) array. A multinomial observation is a vector of category counts for one observation; these are different sample spaces. A Gaussian observation is a real-valued feature vector, stored in a \(T\times D\) array. If several independent sequences share one storage array, pass their lengths: a boundary is not an observed transition.

Keep the returned quantity straight:

- Model score: log probability or log density of the observations, summed over hidden paths.
- Viterbi decode: highest-scoring joint path and its joint log score.
- Posterior probabilities: smoothed state marginals for the complete supplied sequence.
- Pointwise posterior decoding: choose a marginal mode at each time, which can violate structural transition constraints.

One version-specific trap matters: in hmmlearn 0.3.3 the MAP decoder returns the sum of the selected marginal probabilities as its score, despite a general return description calling the value a log probability. It is the expected number of correct states for that pointwise decision, not the joint log probability of its path. Inspect the [actual decoder](https://github.com/hmmlearn/hmmlearn/blob/0.3.3/src/hmmlearn/base.py) when interpreting such a result.

Likewise, a convergence monitor can stop because the iteration budget is exhausted. Read the objective history and stopping condition; a boolean flag does not establish a global optimum.

## 8. A real sequence: infer grammatical roles in short sentences

Words make the value and limitations of context visible. “Read the entire article” has a sequence of grammatical roles, even though an isolated word can be ambiguous.

We use an attributed extract of **Universal Dependencies English Web Treebank, release r2.16**: 120 training sentences, 40 development sentences and 40 reserved sentences, containing 1,188, 341 and 370 tokens respectively. These are the first eligible short sentences from each official split, not a representative random sample. The [provenance](data-provenance.md) records the extraction, original sentence IDs and licensing. The reserved split is not scored in this experiment.

For a readable first model, map NOUN and PROPN to **Noun**, VERB and AUX to **Verb**, and all other original UPOS labels to **Other**. Preserve the original labels in the data. Other is a deliberately broad category, so the task is easier and less linguistically complete than full part-of-speech tagging.

Here the training states are labeled. We estimate initial, transition and emission probabilities from actual counts rather than running latent-state EM. The labels are hidden only when predicting a new sentence. This is supervised HMM fitting.

### Fit without learning from the answer sheet

Lowercase training words and retain those occurring at least twice. All remaining words map to a single unknown symbol. This makes 146 emission symbols. Development words do not alter that vocabulary; 140 of the 341 development tokens map to unknown.

For smoothing strength \(\alpha\), use

\[
\hat a_{ij}=\frac{C_{ij}+\alpha}{\sum_kC_{ik}+3\alpha},\qquad
\hat b_i(w)=\frac{C_{iw}+\alpha}{\sum_vC_{iv}+146\alpha}.
\]

Smooth the three initial-state counts similarly. This experiment has no declared impossible tag transitions, so adding pseudocounts to every transition is intentional. In a topology with genuinely forbidden edges, smooth only allowed events and renormalize there.

Compare two decision rules from the same fitted counts:

1. **Lexical baseline:** at each token choose the state maximizing its training token-frequency prior times the state's emission probability. It ignores neighboring tags.
2. **HMM:** use the learned start and transition probabilities to select the Viterbi path for the sentence.

Try only the two declared smoothing strengths, 0.1 and 1.0, yielding two probability fits and four decoder configurations.

| Smoothing | Decoder | Correct tokens / 341 | Entire sentences correct / 40 |
|---:|---|---:|---:|
| 0.1 | Lexical | 266 | 8 |
| 0.1 | HMM | 268 | 9 |
| 1.0 | Lexical | 266 | 8 |
| 1.0 | HMM | 270 | 9 |

The majority-Other baseline gets 216 tokens correct. Development token accuracy selects the HMM with smoothing 1.0 among these candidates; declared tie rules are recorded in the program. These are development findings from a small educational extract, not final held-out estimates or a claim about modern taggers.

Compare the selected HMM with its matching lexical baseline: it repairs ten token decisions and breaks six, for a net gain of four. Context helps some cases and hurts others.

In **“Dear Nina ,”**, the lexical predictions are Noun–Noun–Other; the HMM predicts Other–Noun–Other, matching the coarse reference labels. In **“Read the entire article ; there 's a punchline , too .”**, context changes “article” from a correct Noun into Other. The model's preference for a common transition pattern can override useful lexical evidence. Inspect both examples before celebrating the aggregate gain.

**Real-data investigation.** Switch between the two recorded decoders on a chosen sentence, reveal original and coarse labels, and identify a repair and a new error. Inspect an unknown word beside a known one. For words sharing the unknown category, the model cannot use their distinct spellings as evidence. A future suffix or character feature could help, but it must be defined from training data and evaluated as a new procedure. The next conditional-model lessons explain a more flexible way to use such input features.

A high posterior is confidence under the chosen model. Coarse labels, misspecified independence, unknown-word collapse and limited training data can still make a confident prediction wrong.

## 9. Choose a model that matches how the sequence behaves

### Duration is a hidden assumption you can see

A self-transition lets a state persist. But it imposes a particular duration distribution. If its self-transition probability is \(a\), then, for \(0\leq a<1\),

\[
P(D=d)=a^{d-1}(1-a),\quad d=1,2,\ldots,\qquad E[D]=\frac1{1-a}.
\]

A state must remain for \(d-1\) transitions and then leave. With \(a=0.7\), the mean dwell time is \(3.333\) steps. With \(a=0.95\), it is twenty. The chance of leaving next is still \(1-a\), however long the state has already lasted. That constant hazard is the geometric distribution's memorylessness.

This can suit a simple regime model, but not a process that becomes progressively more likely to end after a characteristic duration. An explicit-duration or hidden semi-Markov model adds a duration model. An absorbing state with \(a=1\) never leaves; do not draw it as a finite-mean geometric curve.

The time step also has meaning. A transition matrix fitted per minute is not automatically a per-second matrix. If \(k\) unobserved equal time steps pass under the same homogeneous model, propagation uses \(A^k\). Keeping a missing observation as an unobserved time step is different from deleting that time step.

### Real-valued measurements: Gaussian emissions

For a sensor vector \(x_t\in\mathbb R^D\), replace the categorical probability with a density:

\[
b_i(x_t)=\mathcal N(x_t;\mu_i,\Sigma_i).
\]

The forward and backward structure is unchanged. A density can exceed one and changes with measurement units; it is not a probability assigned to one exact real-valued point.

With responsibilities \(\gamma_t(i)\), the Gaussian maximum-likelihood M-step is

\[
\mu_i=\frac{\sum_t\gamma_t(i)x_t}{\sum_t\gamma_t(i)},\qquad
\Sigma_i=\frac{\sum_t\gamma_t(i)(x_t-\mu_i)(x_t-\mu_i)^\top}
{\sum_t\gamma_t(i)}.
\]

Sum across independent sequences too. A diagonal covariance models conditional within-time features without off-diagonal covariance; a full covariance permits within-time correlations. Neither change removes the standard HMM's across-time conditional emission factorization. Marginal observations can already be correlated because their hidden states are related.

An autoregressive emission model can instead condition a current observation on previous observations as well as the state. Adding overlapping windows or delta features to an ordinary emission vector may be useful engineering, but does not make those windows conditionally independent. Decide whether the approximation is acceptable for the intended query.

Very small state occupancy and collapsed Gaussian covariances can make likelihood fitting unstable or degenerate. Appropriate covariance constraints, priors, training evidence and development checks matter. Two states with equal means can still have different variances or transition roles; equality of means alone does not prove redundancy.

### How many states?

For a fully free categorical HMM with \(N\) states and \(M\) symbols, the parameter count is

\[
p=(N-1)+N(N-1)+N(M-1).
\]

Each probability row sums to one. Our two-state, three-symbol model has \(1+2+4=7\) free parameters. Structural zeros, tied rows, fixed parameters and other emission families change that count.

Adding states can increase expressive capacity, yet a particular locally optimized fit can have worse likelihood than a smaller model. Compare several starts and the query you care about, using independent sequences or a justified temporal split. AIC and BIC can be useful selection heuristics, but latent models can be nonregular and correlated observations complicate a casual choice of sample size. State the likelihood, free-parameter count and sample-size convention instead of treating a formula as an automatic answer.

For a sensor deployed online, evaluate filtering or forecasting at the actual decision time. A beautiful smoothed reconstruction obtained after the whole recording arrives answers a different operational question.

## 10. Where this idea leads

**Biological sequences.** A profile HMM represents a sequence family with positions that can match, insert or skip. Match states emit aligned residues; insert states emit additional residues; delete states advance the profile without emitting one. A silent delete transition therefore differs from a missing measurement at a real time step. The [HMMER project](https://hmmer.org/) uses profile HMMs for biological sequence analysis; its [historical model guide](https://eddylab.org/software/hmmer/2.3.1/Userguide.pdf) explains this topology. The interesting connection is alignment as inference through an allowed graph, rather than forcing every sequence to have identical length.

**Speech as a sequence of submodels.** Historical speech recognizers combined state models for sound or word segments. Compare observation likelihoods under candidate models, account for class priors, or compose models and search over legal concatenations. The practical questions include feature modeling, duration and boundaries, not simply “run Viterbi.” Rabiner's tutorial develops that application in its historical setting; its hardware timings are not current benchmarks.

**Several hidden causes at once.** A household power signal can reflect several devices whose states evolve separately. A factorial HMM uses multiple hidden chains with shared observations. Independent prior transitions do not make posterior inference independent once those causes explain the same measurement. A joint representation grows rapidly, motivating structured approximations. This is the extension developed by [Ghahramani and Jordan](https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf).

**Continuous hidden states.** Position and velocity are naturally real-valued. A linear Gaussian state-space model replaces discrete state probabilities with Gaussian beliefs and leads to Kalman filtering and smoothing. The common pattern is prediction through a state transition followed by correction from evidence; different assumptions change the representation and computation.

**Conditional sequence prediction.** HMMs model a joint distribution over observations and states. A CRF models the label sequence conditional on an observed input and can use rich features of that input, including future context when the application permits it. A neural encoder can provide scores to a CRF; these ideas are compatible. Parameter count, available labels, inference constraints and the actual task determine whether one model is useful. There is no universal sequence-length or data-count threshold that selects the winner.

The next lesson, [Bayesian Networks & Causal Graphical Models](/learn/path/full-curriculum/bayesian-networks-causal-graphical-models?module=classical-ml), makes the graph and its conditional independences explicit. After it, [Conditional Random Fields](/learn/path/full-curriculum/conditional-random-fields-crf?module=classical-ml) revisits sequence labeling from the conditional perspective.

## 11. Cost follows the allowed edges

For \(T\) time steps and \(N\) states, a dense forward or Viterbi pass evaluates \(O(TN^2)\) transition contributions, plus emission evaluation. Forward combines them with sums; Viterbi uses maxima. A topology with \(E\) allowed edges can reduce the transition work to \(O(TE)\), provided the representation and implementation actually exploit sparsity.

Filtering needs only the current and previous \(N\)-state rows. Retaining a full trellis or Viterbi backpointers costs \(O(TN)\) memory. A full array of pairwise posteriors costs \(O(TN^2)\); EM can instead accumulate sufficient statistics without retaining every pair row. The teaching program deliberately retains the small pair arrays for inspection. It is not a memory-minimal implementation.

For categorical emissions, add expected counts directly into the observed-symbol column. That avoids looping over all \(M\) symbols at every time step; dense emission tables still cost \(O(NM)\). Gaussian diagonal densities cost roughly \(O(TND)\); dense full-covariance densities involve matrix factorizations and roughly \(O(TND^2)\) quadratic-form work after factorization.

A beam limits the active candidates, but work also depends on their outgoing edges and duplicate successors. “Beam width \(K\) means \(O(TK)\)” is incomplete for a dense state graph. Low-rank matrix multiplication can accelerate sums in some models; replacing the sum with a maximum does not preserve the same algebra automatically. Specialized parallel methods and sparse structures are options to measure, not reasons to invent hardware-independent speed ratios.

## 12. Practice: change the evidence, keep the question precise

Try each before opening its hint or solution.

### 1. Forecast another activity

After observing Walk, what is the probability of Shop next? Why not choose the currently most likely state and use its emission row?

<details><summary>Hint</summary>
Propagate the complete filtered distribution [0.2, 0.8] through A before applying the Shop probabilities.
</details>
<details><summary>Solution</summary>
The next-state probabilities are [0.46, 0.54]. Shop has probability 0.46×0.4 + 0.54×0.3 = 0.346. Collapsing the current belief to Sunny would produce next-state probabilities [0.4, 0.6] and a Shop probability of 0.34. It discards uncertainty before prediction.
</details>

### 2. Repair a misleading backpointer

At the third observation, Walk, which previous state gives the best path ending in Rainy? Use the preceding Viterbi scores 0.0384 and 0.0432.

<details><summary>Hint</summary>
Compare scores after multiplying by the appropriate incoming transition, before multiplying by the shared destination emission.
</details>
<details><summary>Solution</summary>
Rainy contributes 0.0384×0.7 = 0.02688; Sunny contributes 0.0432×0.4 = 0.01728. The Rainy predecessor wins even though its previous score is smaller. Multiplying by the Rainy Walk emission gives 0.002688. The final best path need not pass through this cell.
</details>

### 3. A later observation is corrected

The last Clean observation becomes Walk. Should the probability of Rainy immediately after processing the second observation change? Should the probability of Rainy on that same day after seeing the whole corrected recording change?

<details><summary>Hint</summary>
Name which observations each query conditions on.
</details>
<details><summary>Solution</summary>
The filtered value stays 0.531792 because its prefix is unchanged. The smoothed value changes from about 0.433828 to 0.397624. The best complete path changes to all Sunny. These are different queries, not inconsistent answers.
</details>

### 4. A popular state sequence is impossible

In the three-state example, the marginal modes are A then A, but A cannot transition to A. What are that path's joint probability and the Viterbi path?

<details><summary>Hint</summary>
Marginal modes optimize expected position-wise correctness without enforcing path validity.
</details>
<details><summary>Solution</summary>
A→A has probability zero. B→A has joint probability 0.35 and is the Viterbi path. Pointwise modes have expected correct-state count 1.0, versus 0.95 for B→A, but optimize a different loss over a larger decision set. To require legal paths while optimizing position-wise correctness, solve a constrained max-sum problem using marginal rewards.
</details>

### 5. Count independent recordings

Three recordings have lengths 4, 1 and 3. How many expected starts, emissions and within-recording transitions should the E-step counts sum to? What changes if you silently concatenate them?

<details><summary>Hint</summary>
A length-one recording still has a start and emission, but no transition.
</details>
<details><summary>Solution</summary>
There are 3 starts, 8 emissions and (4−1)+(1−1)+(3−1)=5 transitions. Concatenation changes this to 1 start, 8 emissions and 7 transitions. The two extra transitions are invented boundaries. Posterior fractions can change too.
</details>

### 6. A zero that logarithms cannot rescue

Both states emit only symbol 0, but the recording contains symbol 1. A colleague adds a tiny epsilon to every entry. Is this numerical stabilization alone?

<details><summary>Hint</summary>
Distinguish a positive number too small to represent from a genuinely zero probability.
</details>
<details><summary>Solution</summary>
The original event is impossible. Epsilon introduces previously forbidden observations and requires row normalization, creating a different model. If the zeros reflect insufficient data rather than hard constraints, an explicit smoothing model may be sensible; it must be stated and fitted accordingly.
</details>

### 7. Design a five-step mean duration

Choose the self-transition probability for mean duration five. Find the probability of duration exactly three. After ten steps already spent in the state, what is the chance of leaving next?

<details><summary>Hint</summary>
Use the geometric duration and its constant exit probability.
</details>
<details><summary>Solution</summary>
a=0.8. P(D=3)=0.8²×0.2=0.128. The next-step exit probability remains 0.2, conditional on still being in the state. An age-dependent departure process needs a richer duration model.
</details>

### 8. The EM curve decreases

An implementation shows a substantial log-likelihood decrease after an alleged EM iteration. List checks that could explain it before concluding EM's theorem is false.

<details><summary>Hint</summary>
The theorem concerns a matching exact objective and complete update.
</details>
<details><summary>Solution</summary>
Check whether the plotted quantity is log-likelihood or negative log-likelihood; whether scores and parameters refer to the same iteration; whether sequences and boundaries changed; whether probability rows normalize; whether an approximate E/M-step, clipping, prior or penalty changed the objective; and whether zeros or floating arithmetic broke the calculation. Tiny roundoff differences are distinct from substantive decreases.
</details>

### 9. Interpret the real improvement

The HMM repairs ten lexical predictions but breaks six. What is its net improvement? Does this establish that contextual models always help, or that development accuracy is a final test estimate?

<details><summary>Hint</summary>
The unit being counted is a token, and the development set helped select the procedure.
</details>
<details><summary>Solution</summary>
Four additional tokens are correct, from 266 to 270 out of 341. This is a small, selected development comparison on a coarse task. Keep both repair and failure examples, and evaluate the frozen procedure on an appropriate independent set before claiming generalization. Do not add features based on that final set's mistakes and still call it untouched.
</details>

### 10. Equal means, different states

Two Gaussian states have the same mean but different covariances and self-transition probabilities. Must they be merged?

<details><summary>Hint</summary>
A state's role includes its observation distribution and its dynamics.
</details>
<details><summary>Solution</summary>
No. They can distinguish low-variance versus high-variance regimes or short versus persistent episodes. Whether both are useful requires model and task evaluation. Equal means alone establish neither identical distributions nor redundant sequence behavior.
</details>

## References and another way to learn

- **Start with a complete accessible chapter:** [Jurafsky and Martin, SLP Appendix A](https://web.stanford.edu/~jurafsky/slp3/A.pdf). Markov chains, HMMs, forward inference, Viterbi and EM in one progression. The fetched draft is dated 19 August 2026; notation and chapter numbering may change as the book develops.
- **Go deeper into assumptions and implementation:** [Rabiner, A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, 1989](https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf). Especially the three inference/learning problems, scaling, multiple sequences and duration modeling. Historical speech systems supply context, not current performance guidance.
- **Learn by manipulating a small model:** [Jason Eisner's HMM teaching resources](https://www.cs.jhu.edu/~jason/papers/#eisner-2002-tnlp) accompany [An Interactive Spreadsheet for Teaching the Forward-Backward Algorithm](https://aclanthology.org/W02-0102/). The author's bundle offers reading, exercises, spreadsheets and a video demonstration. The paper metadata and resource descriptions were checked; the video was not watched and the spreadsheet was not executed during this draft.
- **Implement the API examples:** [hmmlearn 0.3.3 tutorial](https://hmmlearn.readthedocs.io/en/0.3.3/tutorial.html). Useful for observation shapes, sequence lengths, initialization and decoding. Pair the prose with the versioned API/source when interpreting returned scores.
- **Inspect real annotations:** [Universal Dependencies English EWT](https://universaldependencies.org/treebanks/en_ewt/index.html) and its [r2.16 source and rights notice](https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md). The local extract preserves original labels and IDs alongside the deliberately coarser teaching task.
- **Explore a different application:** [HMMER](https://hmmer.org/) for profile models of biological sequences; [Factorial Hidden Markov Models](https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf) for several interacting hidden explanations. These extend the state structure rather than merely adding more iterations to the same model.

You are ready to continue when you can name the conditioning information in each query, compute one sum and one maximum trellis update, explain a backward correction, preserve sequence boundaries during learning, and recognize when the model's assumptions are doing more work than its data.
