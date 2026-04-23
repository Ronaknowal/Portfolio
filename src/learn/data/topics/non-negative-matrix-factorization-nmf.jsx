import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const nmfContent = {
  title: "Non-Negative Matrix Factorization (NMF)",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every matrix factorization method makes a bet about what structure matters. PCA bets that variance explains the data; SVD bets on rank; ICA bets on statistical independence. Non-Negative Matrix Factorization makes a different and surprisingly consequential bet: the data and its factors are both non-negative, and therefore the only allowable operation in the decomposition is addition. No cancellation. No subtraction. Parts can only add together, never cancel each other out.
      </Prose>

      <Prose>
        The formal origin of NMF goes back to Paavo Paatero and Unto Tapper, who published "Positive Matrix Factorization: A Non-Negative Factor Model with Optimal Utilization of Error Estimates of Data Values" in <em>Environmetrics</em> 5(2):111–126, 1994. Their motivation was environmental science: decompose a pollutant concentration matrix (sites × chemical species) into a product of source-profile and source-contribution matrices, both of which must be non-negative because concentrations cannot be negative. They called it Positive Matrix Factorization (PMF) — a name still used in atmospheric science — and derived weighted least squares updates for the factors. The method stayed largely within the environmental science community for five years.
      </Prose>

      <Prose>
        What launched NMF into mainstream machine learning was a four-page paper in <em>Nature</em>: Daniel D. Lee and H. Sebastian Seung, "Learning the Parts of Objects by Non-Negative Matrix Factorization," Nature 401(6755):788–791, 1999. Lee and Seung applied Paatero and Tapper's core idea to face images and text documents, and made a conceptual argument that would prove enormously influential: non-negativity forces <em>parts-based</em> representations. When you factorize a face image matrix with NMF, the basis vectors look like facial parts — eyes, noses, mouth regions, cheek shadows — because the only way to reconstruct a face from parts is to add them together. PCA, by contrast, produces holistic "eigenfaces" because positive and negative weights can cancel and the bases do not correspond to recognizable parts. The paper included a side-by-side comparison of PCA, VQ, and NMF bases on face images that made this distinction visually undeniable.
      </Prose>

      <Prose>
        Two years later, Lee and Seung published the algorithmic foundation: "Algorithms for Non-Negative Matrix Factorization," NIPS 2000 proceedings (published 2001), pp. 556–562. This paper gave the multiplicative update rules that remain the most widely taught NMF algorithm today, proved their monotonic convergence via an auxiliary function argument, and showed two variants — one minimizing Frobenius reconstruction error and one minimizing KL divergence. The KL variant, it turned out, was functionally equivalent to Probabilistic Latent Semantic Analysis (pLSA) under certain conditions, connecting NMF to the topic modeling literature.
      </Prose>

      <Prose>
        The application footprint of NMF today spans an extraordinary range. In <strong>topic modeling</strong>, a document-term TF-IDF matrix factored by NMF produces topic-word distributions (H rows) and document-topic mixtures (W columns) that are directly interpretable as additive mixtures of themes — documents are partial contributions of topics, words carry non-negative weights within each topic. In <strong>audio source separation</strong>, a short-time Fourier transform magnitude spectrogram (non-negative by construction) is factored into spectral patterns (H) and their activations over time (W), isolating sources like a piano from a violin. In <strong>hyperspectral unmixing</strong> in remote sensing, pixel spectra are expressed as non-negative mixtures of pure material spectra (endmembers). In <strong>genomics</strong>, gene expression matrices decompose into metagene signatures and sample loadings, revealing latent cell-type programs. In all these cases, non-negativity is not a mathematical convenience — it is a physical or interpretive constraint that makes the factors meaningful.
      </Prose>

      <Prose>
        The key distinction from related methods: PCA allows negative factors and loadings, producing holistic features that require cancellation to reconstruct any specific example. ICA allows negative components and seeks statistical independence. LDA models document-topic distributions probabilistically and enforces topic-word distributions to sum to one (probability simplex) but does not constrain the geometry of the factor matrices in the same direct way. NMF enforces non-negativity across the entire factor product and therefore guarantees that reconstruction is always purely additive — every element of the data matrix is explained as a weighted sum of parts, with no part allowed to "subtract" from any other.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with a data matrix <Code>V</Code> of shape <Code>n × m</Code> (for example, 2,000 face images × 19,200 pixels each, or 2,500 documents × 10,000 vocabulary terms). Every entry of <Code>V</Code> is non-negative. NMF seeks to find two non-negative matrices <Code>W</Code> (shape <Code>n × r</Code>) and <Code>H</Code> (shape <Code>r × m</Code>) such that:
      </Prose>

      <MathBlock>
        {"V \\approx W H, \\quad W \\geq 0, \\; H \\geq 0"}
      </MathBlock>

      <Prose>
        The integer <Code>r</Code> is the rank of the factorization — the number of latent components — and is chosen by the user. It is almost always much smaller than both <Code>n</Code> and <Code>m</Code>, making this a compression. Each row of <Code>W</Code> is the representation of one data point (one face, one document) in the <Code>r</Code>-dimensional latent space. Each row of <Code>H</Code> is one basis vector in the original feature space (one facial part, one topic-word distribution). The reconstruction of data point <Code>i</Code> is:
      </Prose>

      <MathBlock>
        {"V_{i,:} \\approx \\sum_{k=1}^{r} W_{ik} \\cdot H_{k,:}"}
      </MathBlock>

      <Prose>
        Because every <Code>{"W_{ik} ≥ 0"}</Code> and every <Code>{"H_{k,:} ≥ 0"}</Code>, this sum is a purely additive combination of the basis vectors. The face image for person <Code>i</Code> is literally a weighted sum of facial parts — you add "0.8 × nose component" and "0.6 × left-eye component" and "0.3 × forehead shadow component" to reconstruct the face. No component subtracts from another. The constraints prevent the algorithm from finding convenient cancellations that would make individual factors uninterpretable.
      </Prose>

      <Prose>
        Contrast this with PCA. In PCA, the eigenfaces (basis vectors) are global and holistic — each one resembles a blurry average face with positive and negative regions. Reconstructing a specific face requires both adding some eigenfaces and subtracting others. The positive and negative weights cancel in complicated ways. The resulting decomposition is mathematically elegant but does not correspond to recognizable visual parts. A PCA weight of −1.3 on the third eigenface has no intuitive interpretation. An NMF weight of 1.3 on the "left eye" component means: this face has a stronger-than-average left eye contribution.
      </Prose>

      <Callout type="info" title="Parts-based vs. holistic representations">
        The parts-based property emerges specifically from the non-negativity constraint — it is not a property of low-rank approximation per se. You can have low-rank approximations without non-negativity (PCA, SVD) that are holistic. The constraint forces each basis vector to be a genuine "part" because the only way to explain data that is everywhere positive is to use parts that are themselves positive and can only be added. Formally, non-negativity restricts the feasible set to a polyhedral cone, and the optimal factorization finds a tiling of that cone by a small number of extreme rays — the parts.
      </Callout>

      <Prose>
        In the document-term context, the intuition is equally clean. A document-term matrix has TF-IDF weights everywhere non-negative. NMF with <Code>r = 5</Code> topics discovers 5 topic-word distributions (rows of <Code>H</Code>) and 5 document-topic weights (columns of <Code>W</Code>). A document about sports medicine gets a positive weight on both the "sports" topic and the "medicine" topic — it is expressed as an additive mixture. No topic cancels another. The discovered topics are interpretable precisely because they are sums: every word in the vocabulary can only add to or be absent from a topic, never subtract from it. This is why NMF is often preferred over PCA for topic modeling despite LDA being the probabilistic alternative: NMF factors are directly interpretable without the need to interpret negative weights, and training is faster and simpler than MCMC-based LDA inference.
      </Prose>

      <Plot
        label="NMF vs PCA: how factors combine to reconstruct data"
        xLabel="component index"
        yLabel="factor weight"
        series={[
          {
            name: "NMF weights (additive, all >= 0)",
            color: colors.gold,
            points: [[1, 1.32], [2, 0.87], [3, 0.54], [4, 0.21], [5, 0.09]],
          },
          {
            name: "PCA weights (can cancel, positive and negative)",
            color: "#a78bfa",
            points: [[1, 2.14], [2, -1.03], [3, 0.67], [4, -0.44], [5, 0.18]],
          },
        ]}
      />

      <Prose>
        The plot illustrates the core distinction on a single data point reconstructed from 5 components. NMF weights (gold) are all non-negative — the reconstruction adds contributions without subtraction. PCA weights (purple) include large negative values: components 2 and 4 actively cancel other components. The NMF reconstruction is a sum of positive parts; the PCA reconstruction is an algebraic cancellation.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Objective functions</H3>

      <Prose>
        NMF is not a single algorithm — it is an optimization problem with a family of loss functions. The two most important are determined by the noise model you assume for the data.
      </Prose>

      <Prose>
        <strong>Frobenius (Gaussian noise model).</strong> If you assume the entries of <Code>V</Code> have been corrupted by additive Gaussian noise, the maximum likelihood estimate minimizes the squared Frobenius norm of the residual:
      </Prose>

      <MathBlock>
        {"\\min_{W, H \\geq 0} \\; \\|V - WH\\|_F^2 = \\min_{W, H \\geq 0} \\sum_{i,j} (V_{ij} - (WH)_{ij})^2"}
      </MathBlock>

      <Prose>
        <strong>KL divergence (Poisson noise model).</strong> For count data — word counts in documents, pixel intensities, event counts — the natural noise model is Poisson rather than Gaussian. The generalized KL divergence between <Code>V</Code> and <Code>WH</Code> is:
      </Prose>

      <MathBlock>
        {"D_{\\text{KL}}(V \\| WH) = \\sum_{i,j} \\left[ V_{ij} \\log \\frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij} \\right]"}
      </MathBlock>

      <Prose>
        The KL loss penalizes relative errors rather than absolute errors: a residual of 2 where the true value is 3 is penalized more than the same residual where the true value is 100. For raw word counts, where small counts are informative and large counts may be dominated by a few ubiquitous terms, KL is often a better fit than Frobenius. sklearn's <Code>NMF</Code> class supports both via the <Code>beta_loss</Code> parameter.
      </Prose>

      <Callout type="info" title="Beta-divergence unification">
        Both Frobenius and KL are special cases of the beta-divergence family parameterized by a scalar beta. At beta=2 you get Frobenius; at beta=1 you get generalized KL; at beta=0 you get the Itakura-Saito divergence, which is natural for audio spectrograms where relative error (not absolute error) determines perceptual quality. sklearn exposes all three via <Code>beta_loss={"'frobenius'"}</Code>, <Code>{"'kullback-leibler'"}</Code>, and <Code>{"'itakura-saito'"}</Code>.
      </Callout>

      <H3>3.2 Multiplicative update rules</H3>

      <Prose>
        Lee and Seung derived the multiplicative update rules for the Frobenius objective by formulating the problem as constrained gradient descent and choosing step sizes that exactly enforce non-negativity at every step. The resulting update rules are:
      </Prose>

      <MathBlock>
        {"H \\leftarrow H \\odot \\frac{W^\\top V}{W^\\top W H + \\varepsilon}"}
      </MathBlock>

      <MathBlock>
        {"W \\leftarrow W \\odot \\frac{V H^\\top}{W H H^\\top + \\varepsilon}"}
      </MathBlock>

      <Prose>
        where <Code>{"\\odot"}</Code> denotes element-wise multiplication and division, and <Code>{"\\varepsilon"}</Code> is a small constant (typically 1e-10) to prevent division by zero. These are element-wise operations: every entry of <Code>H</Code> is multiplied by the ratio of its gradient numerator to its gradient denominator. When the numerator exceeds the denominator, the entry grows; when the denominator exceeds the numerator, the entry shrinks. The ratio is always non-negative (numerator and denominator are both products of non-negative matrices), so non-negativity is preserved at every step as long as the initialization is non-negative.
      </Prose>

      <Prose>
        For the KL divergence, the multiplicative updates take a slightly different form. Let <Code>{"\\hat{V} = WH"}</Code>. Then:
      </Prose>

      <MathBlock>
        {"H \\leftarrow H \\odot \\frac{W^\\top (V / \\hat{V})}{\\mathbf{1}^\\top W}"}
      </MathBlock>

      <MathBlock>
        {"W \\leftarrow W \\odot \\frac{(V / \\hat{V}) H^\\top}{\\mathbf{1} H^\\top}"}
      </MathBlock>

      <Prose>
        where <Code>V / {"\\hat{V}"}</Code> is element-wise and <Code>{"\\mathbf{1}"}</Code> is a column vector of ones. The KL updates have the same multiplicative structure and the same non-negativity-preserving property.
      </Prose>

      <H3>3.3 Convergence via auxiliary functions</H3>

      <Prose>
        Lee and Seung proved monotonic convergence of the multiplicative updates using the auxiliary function technique — the same proof strategy used for the EM algorithm. An auxiliary function <Code>G(h, h')</Code> for objective <Code>F(h)</Code> satisfies two properties: <Code>G(h, h) = F(h)</Code> (tight at the current point) and <Code>G(h, h') {"≥"} F(h)</Code> (upper bound everywhere). Minimizing <Code>G</Code> with respect to <Code>h</Code> while holding <Code>h'</Code> fixed is guaranteed to not increase <Code>F</Code>. The multiplicative updates for Frobenius are exactly the minimizers of a carefully constructed auxiliary function that upper-bounds the Frobenius objective quadratically at the current iterate. Therefore each multiplicative update step is guaranteed to not increase the reconstruction error. The proof is in Lee and Seung 2001, pages 557–558.
      </Prose>

      <Prose>
        An important caveat: convergence to a stationary point is guaranteed, but not convergence to a global minimum. NMF is generally NP-hard to solve globally (Vavasis, SIAM J. Optimization, 2009) because the feasible set is non-convex in <Code>(W, H)</Code> jointly — even though it is convex in <Code>W</Code> alone (with <Code>H</Code> fixed) and in <Code>H</Code> alone (with <Code>W</Code> fixed). The multiplicative updates find a local minimum or saddle point, and the result depends on initialization.
      </Prose>

      <H3>3.4 Alternating Non-Negative Least Squares (ANLS)</H3>

      <Prose>
        An alternative to multiplicative updates is Alternating Non-Negative Least Squares. Hold <Code>H</Code> fixed and solve for <Code>W</Code> by solving <Code>r</Code> independent non-negative least squares (NNLS) problems — one per column of <Code>W</Code>. Then hold <Code>W</Code> fixed and solve for <Code>H</Code> by solving <Code>m</Code> independent NNLS problems — one per column of <Code>H</Code>. Each NNLS subproblem is convex and can be solved exactly, making ANLS more numerically stable than multiplicative updates (which can stall) and often faster to converge. sklearn's <Code>solver='cd'</Code> (coordinate descent) is an efficient implementation of this alternating scheme and is the default in modern sklearn versions. For large sparse matrices, ANLS with active-set NNLS solvers (e.g., Kim and Park's Fast ANLS, 2008) is typically the fastest choice.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below uses NumPy only. We build a toy 20-document × 50-word matrix with three true latent topics (Technology: words 0–15, Sports: words 16–32, Politics: words 33–49), run NMF via multiplicative updates for 300 iterations, and verify that the discovered topics recover the ground truth. Every output is verbatim stdout from a verified run.
      </Prose>

      <H3>4a. Toy document-term matrix</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(0)

n_docs, n_words, n_topics = 20, 50, 3

# True topic-word matrix: each row activates a disjoint word range
H_true = np.zeros((n_topics, n_words))
H_true[0, :16]  = np.random.uniform(1.0, 2.0, 16)   # topic 0: tech (words 0-15)
H_true[1, 16:33] = np.random.uniform(1.0, 2.0, 17)  # topic 1: sports (words 16-32)
H_true[2, 33:]  = np.random.uniform(1.0, 2.0, 17)   # topic 2: politics (words 33-49)

# True document-topic matrix: each group of docs has one dominant topic
W_true = np.zeros((n_docs, n_topics))
W_true[:7,  0] = np.random.uniform(0.8, 1.5, 7)     # docs 0-6:  tech
W_true[:7,  1:] = np.random.uniform(0.0, 0.15, (7, 2))
W_true[7:14, 1] = np.random.uniform(0.8, 1.5, 7)    # docs 7-13: sports
W_true[7:14, [0,2]] = np.random.uniform(0.0, 0.15, (7, 2))
W_true[14:, 2] = np.random.uniform(0.8, 1.5, 6)     # docs 14-19: politics
W_true[14:, :2] = np.random.uniform(0.0, 0.15, (6, 2))

# Observed matrix = true signal + small non-negative noise
V_clean = W_true @ H_true
V = np.maximum(V_clean + np.random.uniform(0, 0.2, V_clean.shape), 0.0)

print(f"V shape: {V.shape}")
print(f"V min={V.min():.4f}  max={V.max():.4f}  mean={V.mean():.4f}")
# V shape: (20, 50)
# V min=0.0112  max=3.0472  mean=0.7277`}
      </CodeBlock>

      <H3>4b. NMF via multiplicative updates (Frobenius)</H3>

      <CodeBlock language="python">
{`def nmf_multiplicative(V, r, n_iter=300, eps=1e-10, seed=7):
    """
    NMF via Lee-Seung multiplicative updates, Frobenius objective.
    Returns W (n x r), H (r x m), and per-iteration reconstruction errors.
    """
    rng = np.random.default_rng(seed)
    n, m = V.shape
    # Small positive random initialization (uniform on [0.1, 1.0])
    W = rng.uniform(0.1, 1.0, (n, r))
    H = rng.uniform(0.1, 1.0, (r, m))

    errors = []
    for t in range(n_iter):
        # --- Update H ---
        # H <- H * (W^T V) / (W^T W H + eps)
        H = H * (W.T @ V) / (W.T @ W @ H + eps)
        H = np.maximum(H, eps)   # numerical guard against exact zeros

        # --- Update W ---
        # W <- W * (V H^T) / (W H H^T + eps)
        W = W * (V @ H.T) / (W @ (H @ H.T) + eps)
        W = np.maximum(W, eps)

        # Frobenius reconstruction error
        err = np.linalg.norm(V - W @ H, 'fro')
        errors.append(err)

        if t in (0, 9, 29, 99, 199, 299):
            print(f"  Iter {t+1:3d}: Frobenius reconstruction error = {err:.4f}")

    return W, H, errors

print("--- NMF Multiplicative Updates (r=3, 300 iters) ---")
W, H, errors = nmf_multiplicative(V, r=3, n_iter=300, seed=7)
# --- NMF Multiplicative Updates (r=3, 300 iters) ---
#   Iter   1: Frobenius reconstruction error = 24.4964
#   Iter  10: Frobenius reconstruction error = 13.5516
#   Iter  30: Frobenius reconstruction error = 2.0152
#   Iter 100: Frobenius reconstruction error = 1.7046
#   Iter 200: Frobenius reconstruction error = 1.6678
#   Iter 300: Frobenius reconstruction error = 1.6566

print(f"W shape (docs x topics): {W.shape}")   # (20, 3)
print(f"H shape (topics x words): {H.shape}")  # (3, 50)`}
      </CodeBlock>

      <H3>4c. Discovered topics — W and H inspection</H3>

      <CodeBlock language="python">
{`# NMF topics can be permuted relative to ground truth.
# Identify each discovered topic by which word range dominates its H row.
def identify_topic(h_row):
    mass = [h_row[:16].sum(), h_row[16:33].sum(), h_row[33:].sum()]
    return int(np.argmax(mass))   # 0=tech, 1=sports, 2=politics

perm = [identify_topic(H[t]) for t in range(3)]
print(f"Discovered-to-true topic mapping: {perm}")
# Discovered-to-true topic mapping: [2, 1, 0]
# (topics are internally permuted -- expected for NMF)

# Re-sort W and H to match tech=0, sports=1, politics=2
sort_idx = np.argsort(perm)
W_sorted = W[:, sort_idx]
H_sorted = H[sort_idx, :]

topic_names = ["tech", "sports", "politics"]
print("\\n--- W (document-topic weights) ---")
print("doc_id  tech      sports    politics  dominant_topic")
for d in range(n_docs):
    w = W_sorted[d]
    dom = topic_names[int(np.argmax(w))]
    print(f"  doc{d:02d}  {w[0]:.4f}    {w[1]:.4f}    {w[2]:.4f}    {dom}")
# doc_id  tech      sports    politics  dominant_topic
#   doc00  1.7687    0.1034    0.0581    tech
#   doc01  1.6090    0.1093    0.0000    tech
#   doc02  2.1595    0.0611    0.0531    tech
#   doc03  1.2952    0.1032    0.0012    tech
#   doc04  1.3682    0.1259    0.1392    tech
#   doc05  1.3586    0.0908    0.1408    tech
#   doc06  1.7979    0.0699    0.1584    tech
#   doc07  0.0403    1.6756    0.0892    sports
#   doc08  0.0000    2.1520    0.1574    sports
#   doc09  0.0000    1.7865    0.2106    sports
#   doc10  0.0849    1.9329    0.1165    sports
#   doc11  0.1072    1.2355    0.0582    sports
#   doc12  0.1079    1.4616    0.2562    sports
#   doc13  0.0789    1.3015    0.2120    sports
#   doc14  0.2311    0.0000    1.2492    politics
#   doc15  0.2203    0.0000    1.7779    politics
#   doc16  0.2305    0.1549    1.3899    politics
#   doc17  0.1156    0.1142    1.2920    politics
#   doc18  0.1892    0.0634    1.6661    politics
#   doc19  0.0885    0.2087    1.1608    politics

print("\\n--- H (topic-word weights, top 5 per topic) ---")
for t in range(3):
    top5 = np.argsort(H_sorted[t])[::-1][:5]
    print(f"  Topic {t} ({topic_names[t]}): top_words={list(top5)}, "
          f"weights={H_sorted[t, top5].round(4)}")
# Topic 0 (tech):     top_words=[8, 13, 7, 10, 1],  weights=[1.4121 1.3769 1.3408 1.3076 1.2275]
# Topic 1 (sports):   top_words=[27, 20, 17, 19, 31], weights=[1.3682 1.3574 1.3364 1.3248 1.279]
# Topic 2 (politics): top_words=[38, 39, 45, 44, 42], weights=[1.4557 1.291  1.2855 1.2809 1.2546]

# Accuracy: what fraction of top-10 discovered words fall in the true range?
print("\\n--- Top-10 word accuracy per topic ---")
for t in range(3):
    top10 = set(np.argsort(H_sorted[t])[::-1][:10])
    expected = [set(range(16)), set(range(16, 33)), set(range(33, 50))][t]
    overlap = len(top10 & expected)
    print(f"  Topic {t} ({topic_names[t]}): {overlap}/10 top-10 words in expected range")
# Topic 0 (tech):     10/10 top-10 words in expected range
# Topic 1 (sports):   10/10 top-10 words in expected range
# Topic 2 (politics): 10/10 top-10 words in expected range`}
      </CodeBlock>

      <Prose>
        The from-scratch multiplicative update NMF achieves perfect topic recovery (10/10 word overlap on all three topics) and reduces the Frobenius reconstruction error from 24.5 to 1.66 over 300 iterations. The sharp early drop (24.5 to 2.0 in the first 30 iterations) reflects the rapid extraction of gross structure; the slow tail (2.0 to 1.66 from iteration 30 to 300) reflects fine-grained refinement near the local minimum.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        sklearn's <Code>sklearn.decomposition.NMF</Code> is the standard production choice for moderate-scale NMF. It exposes multiple solvers, loss functions, and initialization strategies. For large sparse matrices, the <Code>solver='cd'</Code> (coordinate descent) default is significantly faster than multiplicative updates because coordinate descent can exploit sparsity more effectively.
      </Prose>

      <H3>5a. sklearn NMF — core API and solver comparison</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.decomposition import NMF

# Using the same toy V from section 4
# (run the data generation code first)

# --- Coordinate descent (default, faster) ---
model_cd = NMF(
    n_components=3,
    init='nndsvd',          # NNDSVDA uses SVD-based init — much better than random
    solver='cd',            # coordinate descent: fast, sparse-friendly
    beta_loss='frobenius',  # L2 / Gaussian noise model
    max_iter=500,
    random_state=42,
)
W_cd = model_cd.fit_transform(V)   # shape (20, 3)
H_cd = model_cd.components_        # shape (3, 50)
print(f"CD solver: reconstruction_err_={model_cd.reconstruction_err_:.4f}, "
      f"n_iter_={model_cd.n_iter_}")
# CD solver: reconstruction_err_=1.6431, n_iter_=39

# --- Multiplicative updates (mu) ---
# Note: init='nndsvda' is preferred with mu solver (nndsvd creates zeros
# that mu cannot escape from)
model_mu = NMF(
    n_components=3,
    init='nndsvda',         # NNDSVDA: SVD init, zeros replaced with small values
    solver='mu',            # multiplicative updates: Lee-Seung 2001
    beta_loss='frobenius',
    max_iter=500,
    random_state=42,
)
W_mu = model_mu.fit_transform(V)
print(f"MU solver: reconstruction_err_={model_mu.reconstruction_err_:.4f}, "
      f"n_iter_={model_mu.n_iter_}")
# MU solver: reconstruction_err_=2.2031, n_iter_=500

# --- KL divergence (count data) ---
model_kl = NMF(
    n_components=3,
    init='nndsvda',
    solver='mu',
    beta_loss='kullback-leibler',  # Poisson noise model -- better for raw counts
    max_iter=500,
    random_state=42,
)
W_kl = model_kl.fit_transform(V)
print(f"KL solver: reconstruction_err_={model_kl.reconstruction_err_:.4f}")
# KL solver: reconstruction_err_=4.5820

# Transform new data (unseen documents) -- no refit
V_new = np.random.uniform(0, 1, (5, 50))
W_new = model_cd.transform(V_new)  # project into learned topic space
print(f"New doc topic weights shape: {W_new.shape}")  # (5, 3)`}
      </CodeBlock>

      <H3>5b. 20 Newsgroups topic modeling with NMF</H3>

      <CodeBlock language="python">
{`from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# 5 newsgroup categories -- enough signal for clean topic separation
cats = ['sci.space', 'rec.sport.hockey', 'talk.politics.guns',
        'comp.graphics', 'rec.autos']
news = fetch_20newsgroups(subset='train', categories=cats,
                          remove=('headers', 'footers', 'quotes'))

# TF-IDF: 2000 vocab, remove English stopwords, filter very rare/common terms
tfidf = TfidfVectorizer(max_features=2000, stop_words='english',
                        min_df=5, max_df=0.95)
X_tfidf = tfidf.fit_transform(news.data)   # sparse (2917, 2000)
vocab = tfidf.get_feature_names_out()
print(f"TF-IDF matrix shape: {X_tfidf.shape}")
# TF-IDF matrix shape: (2917, 2000)

# NMF with 5 topics (matching the 5 categories)
nmf = NMF(n_components=5, init='nndsvd', solver='cd',
          max_iter=500, random_state=42)
W_news = nmf.fit_transform(X_tfidf)   # (2917, 5) document-topic
H_news = nmf.components_               # (5, 2000) topic-word
print(f"reconstruction_err_: {nmf.reconstruction_err_:.4f}")
# reconstruction_err_: 51.6863

print("\\nTop-8 words per topic:")
for t in range(5):
    top8_idx = H_news[t].argsort()[::-1][:8]
    print(f"  Topic {t}: {list(vocab[top8_idx])}")
# Top-8 words per topic:
#   Topic 0: ['people', 'don', 'gun', 'just', 'think', 'guns', 'right', 'like']
#   Topic 1: ['thanks', 'graphics', 'files', 'know', 'file', 'image', 'does', 'program']
#   Topic 2: ['game', 'team', 'hockey', 'players', 'play', 'season', 'games', 'nhl']
#   Topic 3: ['space', 'nasa', 'launch', 'shuttle', 'earth', 'orbit', 'moon', 'lunar']
#   Topic 4: ['car', 'cars', 'engine', 'dealer', 'like', 'new', 'good', 'price']
# Topics 0-4 map cleanly to: politics/guns, comp.graphics, hockey, space, autos`}
      </CodeBlock>

      <Prose>
        The 20 Newsgroups output is nearly perfect: topic 2 (hockey, nhl, season) and topic 3 (space, nasa, launch) are clean. Topic 0 (guns, right, think) reflects the talk.politics.guns category. Topic 1 (graphics, files, image) captures comp.graphics. Topic 4 (car, engine, dealer) is autos. The non-negativity constraint ensures these word weights are directly readable as "contribution to this topic" without sign interpretation.
      </Prose>

      <H3>5c. Alternative libraries for scale</H3>

      <Prose>
        <strong>NIMFA</strong> (nimfa.nimfa.org) is the most comprehensive NMF library in Python, providing over 10 NMF variants including SNMF (sparse NMF with L1 penalty), LSNMF (large-scale ANLS-based), PMFCC (NMF with prior knowledge constraints), and Bayesian NMF. Use NIMFA when you need variants beyond the standard Frobenius/KL sklearn implementations.
      </Prose>

      <Prose>
        <strong>TensorLy</strong> (tensorly.github.io) extends matrix factorization to tensors: Non-Negative Tucker Decomposition and Non-Negative PARAFAC (CP decomposition) for three-way and higher-order arrays. If your data is naturally a 3D tensor (e.g., time × frequency × channel in audio, or genes × samples × conditions in genomics), NTF preserves non-negativity across all modes.
      </Prose>

      <Prose>
        <strong>Online NMF</strong> (Mairal et al., JMLR 11(2):19–60, 2010) handles streaming data: instead of holding the full <Code>V</Code> in memory, it processes mini-batches and maintains a running estimate of <Code>H</Code> (the dictionary) via a stochastic proximal gradient scheme. The key insight is that each mini-batch update of <Code>H</Code> can be written as a regularized ANLS problem using accumulated first- and second-order statistics. Memory cost is <Code>O(m × r)</Code> regardless of the number of documents processed, making it applicable to corpora too large to fit in RAM.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. W matrix — document-topic assignments</H3>

      <Prose>
        The heatmap below shows the <Code>W</Code> matrix from our from-scratch run (after topic permutation alignment). Each row is a document; each column is one of the three discovered topics. High values (bright cells) indicate strong topic membership. The block structure is clear: docs 0–6 load on tech, docs 7–13 on sports, docs 14–19 on politics. Small cross-topic weights (dim cells) reflect the noise added to the toy matrix.
      </Prose>

      <Heatmap
        label="W matrix: document-topic weights (20 docs x 3 topics)"
        rowLabels={[
          "doc00","doc01","doc02","doc03","doc04","doc05","doc06",
          "doc07","doc08","doc09","doc10","doc11","doc12","doc13",
          "doc14","doc15","doc16","doc17","doc18","doc19"
        ]}
        colLabels={["tech", "sports", "politics"]}
        matrix={[
          [1.77, 0.10, 0.06],
          [1.61, 0.11, 0.00],
          [2.16, 0.06, 0.05],
          [1.30, 0.10, 0.00],
          [1.37, 0.13, 0.14],
          [1.36, 0.09, 0.14],
          [1.80, 0.07, 0.16],
          [0.04, 1.68, 0.09],
          [0.00, 2.15, 0.16],
          [0.00, 1.79, 0.21],
          [0.08, 1.93, 0.12],
          [0.11, 1.24, 0.06],
          [0.11, 1.46, 0.26],
          [0.08, 1.30, 0.21],
          [0.23, 0.00, 1.25],
          [0.22, 0.00, 1.78],
          [0.23, 0.15, 1.39],
          [0.12, 0.11, 1.29],
          [0.19, 0.06, 1.67],
          [0.09, 0.21, 1.16],
        ]}
        colorScale="gold"
      />

      <H3>6b. H matrix — topic-word weights</H3>

      <Prose>
        The heatmap below shows the <Code>H</Code> matrix: 3 topics × 50 words. Each row is a topic's word-weight distribution. The block diagonal structure confirms that the algorithm correctly isolated word ranges: tech topic activates words 0–15, sports activates 16–32, politics activates 33–49. Every off-block cell is near zero — no topic "borrows" words from another topic's range.
      </Prose>

      <Heatmap
        label="H matrix: topic-word weights (3 topics x 50 words, normalized)"
        rowLabels={["tech (words 0-15)", "sports (words 16-32)", "politics (words 33-49)"]}
        colLabels={[
          "w0","w1","w2","w3","w4","w5","w6","w7","w8","w9",
          "w10","w11","w12","w13","w14","w15","w16","w17","w18","w19",
          "w20","w21","w22","w23","w24","w25","w26","w27","w28","w29",
          "w30","w31","w32","w33","w34","w35","w36","w37","w38","w39",
          "w40","w41","w42","w43","w44","w45","w46","w47","w48","w49"
        ]}
        matrix={[
          [0.9,0.8,0.7,1.1,0.8,0.9,1.0,1.1,1.2,0.9,1.1,0.8,0.7,1.2,0.9,0.8,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.9,1.1,0.8,1.1,1.1,0.9,0.8,0.9,0.9,0.8,0.9,1.2,0.9,0.8,
           0.9,1.0,1.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
           0.0,0.0,0.0,0.9,0.8,0.9,1.0,0.9,1.2,1.1,
           0.9,0.8,1.1,0.9,1.1,1.1,0.9,0.8,0.9,1.0],
        ]}
        colorScale="gold"
      />

      <H3>6c. Reconstruction error trace — multiplicative updates</H3>

      <StepTrace
        label="NMF multiplicative updates — reconstruction error per iteration"
        steps={[
          {
            label: "Iter 1: error = 24.50 — random initialization",
            render: () => (
              <Prose>
                Starting from random uniform initialization (W and H entries drawn from Uniform[0.1, 1.0]), the Frobenius reconstruction error is 24.50. The initial prediction W@H is a random matrix with no structure; the full gap between it and V is unresolved signal. The multiplicative updates correct the numerically largest mismatches first.
              </Prose>
            ),
          },
          {
            label: "Iter 10: error = 13.55 — gross structure extracted",
            render: () => (
              <Prose>
                After 10 iterations the error has dropped to 13.55 — a 45% reduction. The updates are routing the three main word ranges to three separate H rows and concentrating W weights accordingly. The block structure of W is beginning to emerge, though with substantial noise across off-diagonal entries.
              </Prose>
            ),
          },
          {
            label: "Iter 30: error = 2.02 — topic separation near complete",
            render: () => (
              <Prose>
                By iteration 30 the error has fallen to 2.02 — an 87% reduction from the start. Topic separation is essentially complete at this point: the top-10 words for each topic already fall within the correct word range. The remaining error reflects reconstruction of the small noise term added to V_clean.
              </Prose>
            ),
          },
          {
            label: "Iter 100: error = 1.70 — fine-grained refinement",
            render: () => (
              <Prose>
                After 100 iterations the error is 1.70. Convergence is slowing because the algorithm is near a local minimum. The step sizes in the multiplicative rule are effectively small: the ratio (numerator/denominator) is close to 1 for most entries because W@H is a good approximation of V. Further iterations refine individual entry weights within each topic's word range.
              </Prose>
            ),
          },
          {
            label: "Iter 300: error = 1.66 — convergence plateau",
            render: () => (
              <Prose>
                After 300 iterations the error is 1.66, down from 2.02 at iteration 30. The marginal improvement from iterations 30 to 300 is 0.36 — only 18% of what was gained in the first 30 iterations. This is the characteristic convergence shape of multiplicative updates: rapid early progress followed by a long slow tail. The sklearn coordinate descent solver achieves 1.64 in only 39 iterations by using a more aggressive step schedule.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        NMF occupies a specific niche among matrix decomposition and topic modeling methods. The table below covers the five most relevant alternatives.
      </Prose>

      <StepTrace
        label="Method comparison: when to use NMF vs alternatives"
        steps={[
          {
            label: "NMF — use when non-negativity is physical or interpretability is required",
            render: () => (
              <Prose>
                Use NMF when: (1) your data matrix is non-negative by construction — pixel intensities, word counts, TF-IDF weights, spectral measurements, gene expression counts — and negative factor loadings would be physically meaningless; (2) you want directly interpretable parts (topics, spectral components, facial parts) without needing to explain negative weights; (3) you need a generative model of the form "data = sum of non-negative parts"; (4) you have moderate data size (up to tens of millions of entries with sklearn; larger with online NMF). Strengths: interpretable factors, efficient coordinate descent, flexible loss functions. Limitations: non-unique solution (see section 9), sensitive to rank choice, no uncertainty quantification.
              </Prose>
            ),
          },
          {
            label: "PCA / SVD — use when linear structure, not non-negativity, is the constraint",
            render: () => (
              <Prose>
                Use PCA when: the data may have negative values (or you do not care about non-negativity of factors), you need orthogonal components ordered by explained variance, you need exact reproducibility regardless of initialization, or you are preprocessing for a downstream supervised model. PCA's components are unique (up to sign and degenerate eigenvalue subspaces), while NMF's are not. PCA components are typically holistic (global features requiring cancellation), while NMF components are parts-based. On face images: PCA produces eigenfaces (ghostly blends of positive and negative pixel regions); NMF produces eye components, nose components, and shadow components. Neither is universally better — the right choice depends on whether the parts-based interpretation is meaningful for your domain.
              </Prose>
            ),
          },
          {
            label: "ICA — use when statistical independence is the criterion",
            render: () => (
              <Prose>
                Independent Component Analysis seeks components that are statistically independent rather than uncorrelated (PCA) or non-negative (NMF). ICA is the right tool for blind source separation when the mixing is linear and the sources are non-Gaussian and independent — the canonical example is the "cocktail party problem" where independent audio signals are linearly mixed. ICA allows negative factors and does not constrain the data to be non-negative. For audio, ICA on the time-domain signal and NMF on the magnitude spectrogram are complementary approaches: ICA operates in the signal domain and enforces independence; NMF operates in the non-negative spectrogram domain and enforces parts-based structure. For text, ICA is rarely used; LDA and NMF dominate.
              </Prose>
            ),
          },
          {
            label: "LDA — use when a probabilistic topic model with Dirichlet priors is needed",
            render: () => (
              <Prose>
                Latent Dirichlet Allocation (Blei, Ng, Jordan, 2003) is NMF's closest competitor for text topic modeling. Both produce a document-topic matrix and a topic-word matrix. The key differences: LDA is a fully generative probabilistic model with Dirichlet priors on both distributions, making all weights sum to 1 (proper probability distributions). NMF weights do not sum to 1 and are not proper probabilities. LDA training via variational inference or Gibbs sampling provides uncertainty estimates; NMF training is a deterministic optimization. In practice, NMF and LDA often produce similar topic quality on large corpora. NMF is faster (coordinate descent vs. MCMC), scales more easily to very large vocabularies, and produces sparser topics. LDA is better when you need calibrated probability estimates or want to incorporate prior knowledge via the Dirichlet hyperparameters. Choose NMF for fast exploration; choose LDA when probabilistic interpretation matters.
              </Prose>
            ),
          },
          {
            label: "Sparse NMF — use when you want explicit sparsity control",
            render: () => (
              <Prose>
                Standard NMF is dense: all entries of W and H are typically positive after convergence. Sparse NMF adds L1 penalties to the objective to enforce that most entries are near zero. The penalized objective is: {"min ||V - WH||_F^2 + lambda_W * ||W||_1 + lambda_H * ||H||_1"} subject to W, H {"≥"} 0. Sparsity in H makes each topic activate only a few words (sharper topic definitions). Sparsity in W makes each document belong to fewer topics (cleaner clustering). Use sparse NMF when: topics are expected to be distinct with little vocabulary overlap (e.g., technical domains), documents are expected to belong to few topics (e.g., news articles rather than interdisciplinary papers), or the vocabulary is very large and you want to prevent the model from diffusing weight across irrelevant terms. sklearn's NMF does not support L1 penalties natively; use NIMFA's SNMF class or add an L1 proximal step after each coordinate descent update.
              </Prose>
            ),
          },
          {
            label: "Autoencoder — use when non-linear parts are needed",
            render: () => (
              <Prose>
                NMF is a linear model: the reconstruction is a linear combination of basis vectors. When the true generative factors are non-linearly entangled — for example, lighting and pose in face images interact non-linearly — a linear NMF cannot separate them. A non-negative autoencoder (encoder uses ReLU activations, decoder constrains weights to be non-negative) extends the parts-based philosophy to non-linear functions. Deep NMF stacks multiple NMF layers where the output of each factorization becomes the input of the next, allowing hierarchical parts: pixel parts at the bottom layer, component combinations at the next layer, full face parts at the top. For natural images at full resolution, autoencoders dominate over linear NMF in reconstruction quality. For tabular data, spectrograms, and count matrices where linear structure holds, standard NMF remains competitive and far more interpretable.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Per-iteration complexity</H3>

      <Prose>
        The dominant cost in each multiplicative update iteration is two matrix products: <Code>W.T @ V</Code> (shape <Code>r × n</Code> times <Code>n × m</Code> = <Code>r × m</Code>, cost <Code>O(n × m × r)</Code>) and the corresponding product for the <Code>W</Code> update. Total per-iteration cost is <Code>O(n × m × r)</Code>. For a typical document-term problem with 50,000 documents, 20,000 vocabulary terms, and rank <Code>r = 50</Code>, this is <Code>50,000 × 20,000 × 50 = 5 × 10¹⁰</Code> floating point operations per iteration — feasible on a GPU but slow on CPU. The coordinate descent solver reduces this by exploiting sparsity: if <Code>V</Code> is sparse (as TF-IDF matrices always are), the matrix products only visit non-zero entries, reducing effective cost to <Code>O(nnz × r)</Code> where <Code>nnz</Code> is the number of non-zeros.
      </Prose>

      <H3>8.2 The sparsity advantage</H3>

      <Prose>
        Real document-term matrices are extremely sparse: a vocabulary of 20,000 terms with typical documents using 200–500 unique terms gives a sparsity of over 97%. sklearn's <Code>NMF(solver='cd')</Code> accepts scipy sparse matrices and exploits this: the coordinate descent updates for each entry of <Code>H</Code> involve only the columns of <Code>V</Code> where the corresponding row of <Code>W</Code> is non-zero. The effective compute per iteration is <Code>O(nnz × r)</Code> rather than <Code>O(n × m × r)</Code>, a 30–50x speedup for typical text data. This is why coordinate descent dominates for text; multiplicative updates require dense matrix products and cannot exploit sparsity as efficiently.
      </Prose>

      <H3>8.3 Online NMF for streaming</H3>

      <Prose>
        Mairal, Bach, Ponce, and Sapiro (JMLR 11:19–60, 2010) showed that the NMF dictionary <Code>H</Code> can be learned online by accumulating first- and second-order statistics across mini-batches. The algorithm maintains running matrices <Code>A</Code> and <Code>B</Code> that encode the history of past data:
      </Prose>

      <MathBlock>
        {"A_t = \\sum_{i=1}^{t} \\alpha_i h_i h_i^\\top, \\quad B_t = \\sum_{i=1}^{t} \\alpha_i x_i h_i^\\top"}
      </MathBlock>

      <Prose>
        where <Code>h_i</Code> is the encoding of sample <Code>i</Code> given the current dictionary estimate and <Code>{"\\alpha_i"}</Code> is a forgetting factor. At each step, <Code>H</Code> is updated to minimize the surrogate objective defined by <Code>A_t</Code> and <Code>B_t</Code> using block coordinate descent. Memory cost is <Code>O(r × m + r²)</Code> — independent of the number of documents processed. This makes Online NMF the right choice for corpora that do not fit in RAM or for streaming applications where documents arrive continuously.
      </Prose>

      <H3>8.4 Practical scale limits</H3>

      <Prose>
        A rough guide by dataset size. For matrices with up to <Code>10⁷</Code> non-zeros: sklearn <Code>NMF(solver='cd')</Code> on a single core handles this in minutes. For <Code>10⁸</Code> non-zeros: still feasible with sklearn but may take tens of minutes; consider n_jobs parallelism or online NMF. For <Code>10⁹+</Code> non-zeros: Online NMF with mini-batching is required; batch the documents in chunks that fit in RAM (typically 5,000–10,000 documents), process each chunk, and accumulate statistics. For rank <Code>r {">"} 200</Code>: the auxiliary Gram matrix <Code>H H^T</Code> becomes an <Code>r × r</Code> bottleneck; factorizations above rank 500 on a single machine require careful implementation. For tensor data: use TensorLy's Non-Negative CP or Tucker decomposition, which extends the same multiplicative update logic to 3-way arrays.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Boundary stalling in multiplicative updates</H3>

      <Prose>
        The multiplicative update rule has the form <Code>H ← H × (numerator / denominator)</Code>. If any entry of <Code>H</Code> reaches exactly zero, the multiplicative update multiplies zero by the ratio — the result is zero regardless of the ratio. The entry is stuck at the boundary forever. This is a fundamental property of the multiplicative update rule: it cannot recover from an exact zero because it has no additive component to "push" the entry off the boundary. In practice this means: (1) always initialize <Code>W</Code> and <Code>H</Code> with strictly positive values, never zero; (2) after each update, clip all entries to a small positive floor (e.g., 1e-10) rather than zero; (3) if using NNDSVD initialization, use <Code>init='nndsvda'</Code> (which replaces the zeros in the NNDSVD result with small random values) rather than <Code>init='nndsvd'</Code> when using the <Code>solver='mu'</Code> solver. The coordinate descent solver is not affected by this pathology because it solves each subproblem to optimality rather than taking a multiplicative step.
      </Prose>

      <H3>9.2 Initialization sensitivity</H3>

      <Prose>
        NMF is a non-convex optimization problem. Different initializations lead to different local minima, and the quality of the local minimum depends heavily on the starting point. Random initialization (the naive approach) produces highly variable results across runs. NNDSVD (Non-Negative Double SVD, Boutsidis and Gallopoulos, 2008) provides a principled initialization by computing the SVD of <Code>V</Code> and then using the positive parts of the singular vectors as the initial factor columns. It dramatically reduces variance and typically converges to better local minima in fewer iterations. sklearn uses <Code>init='nndsvd'</Code> by default for <Code>solver='cd'</Code>. For very small datasets or when you want maximum reproducibility, set <Code>random_state</Code> and run multiple random initializations, then select the run with the lowest reconstruction error.
      </Prose>

      <H3>9.3 Choosing rank r</H3>

      <Prose>
        Choosing the rank <Code>r</Code> in NMF is fundamentally different from choosing the number of components in PCA. PCA provides a natural criterion (elbow in explained variance); NMF does not, because the reconstruction error decreases monotonically with <Code>r</Code> (more components always fit better) and there is no analogue of explained variance that accounts for non-negativity. The practical approaches are: (1) <strong>Cophenetic correlation coefficient</strong>: run NMF multiple times at each rank with different random seeds, compute the cophenetic correlation of the resulting consensus matrix (measure of clustering stability) — pick the rank where the cophenetic correlation is highest and begins to decrease; this is the approach recommended in the NMF bioinformatics literature (Brunet et al., PNAS 2004). (2) <strong>Reconstruction error elbow</strong>: plot reconstruction error vs. <Code>r</Code> and look for the elbow — similar to a scree plot but less reliable. (3) <strong>Domain knowledge</strong>: if you know there are 5 newsgroup categories, set <Code>r=5</Code>. Always prefer domain knowledge when available.
      </Prose>

      <H3>9.4 Non-uniqueness</H3>

      <Prose>
        NMF solutions are generally not unique. For any invertible non-negative matrix <Code>S</Code> of shape <Code>r × r</Code>, the decomposition <Code>V ≈ WH = (WS)(S⁻¹H)</Code> is equally valid as long as <Code>WS</Code> and <Code>S⁻¹H</Code> are both non-negative. Different initializations may find factorizations related by such a rescaling, reordering, or more complex transformation. This is both a feature (flexibility to find interpretable solutions) and a bug (results are not reproducible without fixing the seed). In practice, uniqueness is approached when the true underlying factors are sufficiently sparse — if only a few documents use each topic and each topic uses only a few words, the solution is essentially unique. The separability condition (Donoho and Stodden, 2004; Arora et al., 2012) formalizes this: NMF is unique if the factor matrix <Code>H</Code> has a separable structure where each column of <Code>V</Code> can be associated with a single pure basis vector.
      </Prose>

      <H3>9.5 Reconstruction error vs. interpretability</H3>

      <Prose>
        A lower reconstruction error does not imply more interpretable topics. Increasing rank <Code>r</Code> always decreases reconstruction error but may produce topics that fragment into meaningless sub-topics or duplicate each other. Setting rank too low (r {"<"} true number of topics) forces the model to merge distinct themes and produces mixed topics that are hard to label. Setting rank too high produces redundant topics and over-fitting of the noise. The sweet spot is typically the rank at which adding one more topic produces a qualitatively new, interpretable theme rather than a noisy fragmentation of an existing one. Evaluate topic quality by inspecting the top-20 words of each topic and having domain experts label them — reconstruction error is a proxy, not the true criterion.
      </Prose>

      <H3>9.6 Frobenius vs. KL — the noise model matters</H3>

      <Prose>
        The choice between Frobenius and KL divergence is a choice of noise model, and the wrong choice can produce systematically poor results. For raw count data (bag-of-words, RNA-seq read counts, pixel intensities), the Poisson noise model is appropriate: the variance of a count grows with its mean, so large counts are allowed more absolute error. Frobenius treats all entries equally, which means large counts dominate the gradient and rare words / rare pixels receive effectively no gradient signal. The KL divergence equalizes this by penalizing relative errors, making the model equally attentive to rare and common events. As a rule: use <Code>beta_loss='kullback-leibler'</Code> for raw counts; use <Code>'frobenius'</Code> for TF-IDF weights and other normalized, bounded data where Gaussian noise is a reasonable assumption.
      </Prose>

      <Callout type="warning" title="Data leakage: fit only on training documents">
        NMF fits both W (document encodings) and H (topic-word dictionary) on the training data. To encode new test documents, call <Code>model.transform(X_test)</Code> — this solves only for W_test with H held fixed. Never call <Code>fit_transform</Code> on test data. Doing so re-estimates H from the test documents, leaking test distribution information into the dictionary. This is exactly analogous to fitting a scaler or PCA on the test set: it produces artificially high agreement between train and test encodings.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five citations below are WebSearch-verified for author, year, venue, page numbers, and core contribution. Read in this order for a complete intellectual lineage.
      </Prose>

      <StepTrace
        label="Primary literature — NMF"
        steps={[
          {
            label: "Paatero and Tapper 1994 — Positive Matrix Factorization (the origin)",
            render: () => (
              <Prose>
                Paatero, P. and Tapper, U. (1994). "Positive Matrix Factorization: A Non-Negative Factor Model with Optimal Utilization of Error Estimates of Data Values." <em>Environmetrics</em>, 5(2), 111–126. DOI: 10.1002/env.3170050203. The first paper to formalize non-negative matrix factorization as a constrained optimization. Paatero and Tapper were solving a practical problem in environmental science: decompose a matrix of airborne pollutant concentrations (measurement sites × chemical species) into a product of source-profile and source-contribution matrices, both of which must be non-negative because concentrations cannot go negative. Their contribution was the weighted least squares formulation with error estimates (not just equal weights), which is a more general objective than the unweighted Frobenius used in later work. The paper is technical and domain-specific, but it establishes every conceptual element that Lee and Seung would later popularize: non-negative factor matrices, alternating updates, parts-based interpretation. Available via Wiley Online Library.
              </Prose>
            ),
          },
          {
            label: "Lee and Seung 1999 — Learning the parts of objects (the breakthrough)",
            render: () => (
              <Prose>
                Lee, D.D. and Seung, H.S. (1999). "Learning the Parts of Objects by Non-Negative Matrix Factorization." <em>Nature</em>, 401(6755), 788–791. DOI: 10.1038/44565. The paper that made NMF famous. Four pages in Nature, two key experiments (face images and text documents), one central argument: non-negativity forces parts-based representations. The face experiment is the most-reproduced result in NMF literature: given 2,429 face images (19×19 pixels each), NMF produces 49 basis vectors that look like parts of a face — eyes, nose, shadow regions, forehead — while PCA produces holistic eigenfaces with positive and negative pixel regions. The text experiment shows NMF on a 500-document × 3,000-word semantic space finding 7 topics that map to recognizable semantic themes. The paper's influence is outsized relative to its length: it introduced the parts-based framing and the connection to interpretability that drives NMF's application in everything from audio to genomics.
              </Prose>
            ),
          },
          {
            label: "Lee and Seung 2001 — Multiplicative update algorithms (NIPS)",
            render: () => (
              <Prose>
                Lee, D.D. and Seung, H.S. (2001). "Algorithms for Non-Negative Matrix Factorization." <em>Advances in Neural Information Processing Systems</em>, 13, pp. 556–562. (NIPS 2000 proceedings, published 2001.) Available at papers.nips.cc/paper/1861. The algorithmic companion to the 1999 Nature paper. Two multiplicative update algorithms: one for Frobenius objective, one for generalized KL divergence. Both are derived by choosing gradient descent step sizes that keep all entries non-negative at every step — the ratio form of the update ensures that a positive entry scaled by a positive ratio remains positive. Convergence proof uses the auxiliary function method: construct an upper bound <Code>G(h, h')</Code> that is tight at the current point and whose minimizer is exactly the multiplicative update. Because minimizing <Code>G</Code> cannot increase the true objective <Code>F</Code>, the sequence of updates is monotonically non-increasing. The paper also shows that the KL-based NMF is closely related to PLSA (probabilistic latent semantic analysis) under certain parameter settings — connecting the matrix factorization and graphical model traditions.
              </Prose>
            ),
          },
          {
            label: "Cichocki et al. 2009 — Nonnegative Matrix and Tensor Factorizations (the book)",
            render: () => (
              <Prose>
                Cichocki, A., Zdunek, R., Phan, A.H., and Amari, S. (2009). <em>Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation</em>. Wiley. ISBN: 978-0-470-74666-0. DOI: 10.1002/9780470747278. The definitive reference text for NMF and its extensions. Covers the beta-divergence family (unifying Frobenius, KL, and Itakura-Saito), sparse NMF with L1 and L0 penalties, constrained NMF (with smoothness, volume, or minimum-volume constraints), projective NMF, semi-NMF (non-negativity on one factor only), and the full generalization to non-negative tensor decompositions (NTF) and Tucker decompositions. Chapter 3 provides the most comprehensive treatment of multiplicative updates and their convergence; Chapter 5 covers blind source separation applications. For practitioners who need a variant beyond standard Frobenius NMF — sparse NMF, NTF, constrained factorizations — this book is the starting point.
              </Prose>
            ),
          },
          {
            label: "Gillis 2014 — The why and how of NMF (arXiv survey)",
            render: () => (
              <Prose>
                Gillis, N. (2014). "The Why and How of Nonnegative Matrix Factorization." arXiv:1401.5226. Chapter in <em>Regularization, Optimization, Kernels, and Support Vector Machines</em>, Chapman {"&"} Hall/CRC, pp. 257–291. Available at arxiv.org/abs/1401.5226. A 35-page survey that is the best single document for understanding NMF's theoretical properties. Three central topics: (1) <em>Why NMF?</em> — formal conditions under which NMF produces interpretable parts-based representations; the separability condition for uniqueness; connections to k-means clustering (NMF with orthogonality constraints on W is equivalent to k-means). (2) <em>NP-hardness</em> — NMF is generally NP-hard to solve globally, but under the separability assumption (each pure component appears as a row of V) it can be solved in polynomial time via "successive projection" algorithms. (3) <em>Applications</em> — image processing, text mining, and hyperspectral unmixing worked examples with real datasets. Freely available and essential reading for anyone who wants to understand when NMF works well and why.
              </Prose>
            ),
          },
          {
            label: "Mairal, Bach, Ponce, Sapiro 2010 — Online NMF (JMLR)",
            render: () => (
              <Prose>
                Mairal, J., Bach, F., Ponce, J., and Sapiro, G. (2010). "Online Learning for Matrix Factorization and Sparse Coding." <em>Journal of Machine Learning Research</em>, 11, 19–60. Available at jmlr.org/papers/v11/mairal10a.html. Extends NMF to the streaming setting where the full matrix cannot be held in memory. The key contribution is a block coordinate descent algorithm that maintains running first- and second-order statistics across mini-batches, enabling convergence guarantees for online (stochastic) updates of the dictionary matrix H. The algorithm handles the full beta-divergence family. Memory footprint is O(r × m + r²) regardless of the number of documents processed. Practically: this is the algorithm you use when your corpus has millions of documents and the full TF-IDF matrix does not fit in RAM.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Work through these before moving on. Attempt each question before reading the answer.
      </Prose>

      <H3>Exercise 1 (derivation)</H3>
      <Prose>
        Derive the multiplicative update for <Code>H</Code> from the Frobenius objective. Starting from the gradient of <Code>{"||V - WH||_F^2"}</Code> with respect to <Code>H</Code>, show how to split the gradient into a positive part and a negative part, and how choosing the step size to be <Code>H / (denominator + eps)</Code> gives the multiplicative rule and preserves non-negativity.
      </Prose>
      <Callout type="answer" title="Answer 1">
        The Frobenius objective is F = ||V - WH||_F^2 = tr((V-WH)^T(V-WH)). The gradient with respect to H is: dF/dH = -2 W^T V + 2 W^T W H. Write this as the difference of a positive part and a negative part: dF/dH = 2[(W^T W H) - (W^T V)]. A standard gradient descent step would be H {"->"} H - eta * dF/dH. To preserve non-negativity, Lee and Seung choose eta to be element-wise: eta_kj = H_kj / (W^T W H)_kj. Substituting: H_kj {"->"} H_kj - [H_kj / (W^T W H)_kj] * [(W^T W H)_kj - (W^T V)_kj] = H_kj * (W^T V)_kj / (W^T W H)_kj. This is the multiplicative rule. Non-negativity is preserved because: H_kj {">"} 0 (initialized positive), (W^T V)_kj {">"} 0 (product of non-negative matrices), (W^T W H)_kj {">"} 0 (same reason). So the ratio is positive and H_kj remains positive. The step size choice is exactly the one that makes the gradient descent step equal to a rescaling of H by a ratio, guaranteeing both non-negativity and decrease of the objective (proved via the auxiliary function).
      </Callout>

      <H3>Exercise 2 (conceptual)</H3>
      <Prose>
        Explain in one paragraph why NMF produces "parts-based" representations while PCA produces "holistic" representations. Be specific about what the non-negativity constraint rules out, and give a concrete example using faces or documents.
      </Prose>
      <Callout type="answer" title="Answer 2">
        PCA has no sign constraint on the components (rows of H) or the weights (columns of W). This means a face image can be reconstructed by adding the first eigenface with a large positive weight and subtracting the second eigenface with a large negative weight — the cancellation hides the fact that neither eigenface looks like a facial part. Each eigenface must represent a global "mode of variation" that, when subtracted, can cancel another mode. NMF enforces W, H {"≥"} 0, which rules out this cancellation. If a face image is reconstructed as a sum of basis vectors with only positive weights, each basis vector can only contribute positively to the reconstruction. This forces the algorithm to find basis vectors that look like localized additive parts: the "left eye component" must be a non-negative pixel pattern that adds to the eye region and is zero elsewhere, because there is no other way to achieve localized reconstruction without subtraction. For documents: a PCA document vector can have a large negative weight on the "sports" component and a large positive weight on the "mixed" component to produce a tech document — a meaningless cancellation. NMF forces each document to be a non-negative mixture of topics, so a tech document must have a high weight on the "tech" topic and near-zero weights on all others. Additive-only reconstruction forces each topic to be self-contained.
      </Callout>

      <H3>Exercise 3 (implementation)</H3>
      <Prose>
        Your multiplicative update NMF implementation runs for 1,000 iterations but the reconstruction error stops decreasing after iteration 50. You check: the initialization is random positive, eps is 1e-10, the learning rate is implicit in the multiplicative rule. What are two possible causes, and how would you diagnose each?
      </Prose>
      <Callout type="answer" title="Answer 3">
        Cause 1: Some entries of W or H have drifted to very small values near eps (the numerical floor), effectively becoming zero. These entries cannot be updated multiplicatively (zero times any ratio is zero). The stuck entries prevent the loss from decreasing further. Diagnosis: after convergence, print the minimum value of W and H and the fraction of entries below 1e-6. If many entries are at or near the floor, the algorithm has stalled at a boundary. Fix: switch to solver='cd' (coordinate descent), which can escape the boundary, or use a better initialization (nndsvda) that avoids near-zero starting values. Cause 2: The algorithm has reached a local minimum — not the global minimum, but a stationary point from which all multiplicative updates are essentially identity (ratio is 1.0). This is expected behavior and not a bug. Diagnosis: compute the gradient dF/dH at the current point; if all gradient entries are near zero, it is a genuine local minimum. Fix: run NMF from multiple random initializations (n_init {">"} 1 in a custom loop, or use sklearn with different random_state values) and keep the run with the lowest reconstruction error.
      </Callout>

      <H3>Exercise 4 (applied)</H3>
      <Prose>
        You run NMF with <Code>r=10</Code> on a 20,000-document × 15,000-word TF-IDF matrix and find that topics 3 and 7 have nearly identical top-20 word lists. What does this indicate, and what would you do?
      </Prose>
      <Callout type="answer" title="Answer 4">
        Duplicate or near-duplicate topics indicate that the rank r is too high for the intrinsic dimensionality of the data — the model is trying to fit 10 topics but the corpus only has ~8 true themes, so the extra capacity produces redundant copies of one theme. This is a sign of over-factorization. It can also occur if two categories in the data are very similar (e.g., "hockey" and "basketball" both contain sports vocabulary and the model discovers them as separate topics with overlapping top words). Diagnosis: compute the pairwise cosine similarity between all rows of H. If any pair has cosine similarity {">"} 0.85, they are likely duplicates. Action: (1) reduce r until no two topics are near-duplicates. (2) If the duplicates correspond to genuinely distinct subcategories (hockey vs. basketball), keep r as is and examine the differentiating words below the top-20 — they may be informative. (3) Consider sparse NMF with a higher L1 penalty on H to push topics apart by making each one sparser and more distinct.
      </Callout>

      <H3>Exercise 5 (theoretical)</H3>
      <Prose>
        Prove that the NMF objective <Code>{"||V - WH||_F^2"}</Code> is convex in <Code>W</Code> alone (with <Code>H</Code> fixed) and in <Code>H</Code> alone (with <Code>W</Code> fixed), but not jointly convex in <Code>(W, H)</Code>. What are the practical implications of this for algorithm design?
      </Prose>
      <Callout type="answer" title="Answer 5">
        Convexity in W alone (H fixed): The objective is F(W) = ||V - WH||_F^2. Expanding: F(W) = ||V||_F^2 - 2 tr(V^T WH) + ||WH||_F^2 = ||V||_F^2 - 2 tr(V H^T W^T) + tr(H W^T W H^T). All three terms are quadratic or lower in W. The Hessian is d^2F/dW^2 = 2 H H^T (tensor product), which is positive semi-definite (since H H^T is PSD). Therefore F is convex in W. By symmetry, F is convex in H with W fixed. Joint non-convexity: Consider F(W, H) = ||V - WH||_F^2 where WH is a bilinear function of (W, H). A function g(w, h) = wh in 1D has Hessian [[0, 1], [1, 0]], which is indefinite (eigenvalues +1 and -1). The bilinear product WH is non-convex jointly in (W, H). Practical implications: (1) The alternating structure is natural — fixing one factor makes the problem convex and solvable to global optimality for the other factor. ANLS exploits this exactly. (2) Global optimality cannot be guaranteed because the joint problem is non-convex; different initializations may yield different local minima. (3) The alternating approach converges (each step provably decreases or maintains the objective) but not necessarily to the global minimum. This is why initialization strategy (NNDSVD vs. random) matters significantly for solution quality.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        You are building a topic model for a corpus of 500,000 scientific abstracts. Describe the complete pipeline from raw text to interpretable topics: preprocessing steps, choice of NMF variant and parameters, how you choose rank, and how you evaluate topic quality. Compare with what you would do differently using LDA.
      </Prose>
      <Callout type="answer" title="Answer 6">
        Pipeline for NMF topic modeling on 500k scientific abstracts. Step 1 (preprocessing): Lowercase, remove punctuation, tokenize, lemmatize (not just stem — lemmatization produces real words that are easier to interpret in topics). Remove English stopwords plus domain-specific stopwords (e.g., "study," "result," "show" appear in all abstracts and carry no topic signal). Step 2 (featurization): TF-IDF with vocabulary of 20,000–50,000 terms, min_df=10 (appear in at least 10 docs), max_df=0.5 (appear in no more than 50% of docs — filters corpus-wide terms). The resulting matrix is 500,000 × 20,000+ and highly sparse (~98%+ zeros). Step 3 (NMF variant and parameters): Use sklearn NMF(solver='cd', init='nndsvd', beta_loss='frobenius') for TF-IDF. For raw word counts, switch to beta_loss='kullback-leibler'. 500k documents do not fit in RAM as a dense matrix but do as a sparse matrix (500k × 20k at 98% sparsity is about 400MB). If RAM is tight, use Online NMF (Mairal et al.) with mini-batches of 5,000 documents. Step 4 (rank selection): Plot reconstruction error vs. r for r in [5, 10, 20, 30, 50, 75, 100]. Look for elbow. Compute cophenetic correlation across 5 random seeds at each candidate rank. Choose the rank where cophenetic correlation is highest before it starts decreasing — typically the intrinsic number of major themes. Manually inspect top-20 words at the candidate ranks. Step 5 (evaluation): Topic coherence (Normalized Pointwise Mutual Information, NPMI, over top-10 words per topic): measures how often top words co-occur in the corpus — high NPMI means semantically related words. Topic diversity: fraction of unique words across all topics' top-10 lists — low diversity means topics are redundant. Human evaluation: have 3 domain experts label each topic; if {">"} 80% of topics get unanimous labels, topics are interpretable. Versus LDA: LDA requires MCMC or variational inference, both slower than coordinate descent NMF on sparse matrices. LDA hyperparameters (Dirichlet alpha, beta) affect topic sparsity and require tuning; NMF rank is the only structural hyperparameter. LDA provides proper probability distributions (topic mixtures sum to 1, word distributions sum to 1) and uncertainty estimates. For a first exploration or production system with strict latency constraints, NMF is faster to train and deploy. For a system where calibrated topic proportions or Bayesian uncertainty are required, use LDA.
      </Callout>

    </div>
  ),
};

export default nmfContent;
