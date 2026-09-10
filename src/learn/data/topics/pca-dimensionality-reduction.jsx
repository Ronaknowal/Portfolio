import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const pcaContent = {
  title: "PCA & Dimensionality Reduction",
  readTime: "~45 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        High-dimensional data is almost always a lie. You collect measurements across hundreds of genes, pixels, sensor channels, or survey responses, and you end up with a matrix with more columns than you could ever meaningfully visualize or reason about directly. But that matrix almost never actually occupies the full dimensionality it appears to. The genes are correlated because they share regulatory pathways. The pixels are correlated because adjacent pixels tend to be similar colors. The sensor channels are correlated because the physical process they measure has fewer degrees of freedom than the sensors do. Real-world data, almost without exception, lives near a low-dimensional manifold embedded inside a high-dimensional ambient space. Dimensionality reduction is the project of finding that manifold and working with it directly.
      </Prose>

      <Prose>
        The intellectual origin of PCA is a 1901 paper by Karl Pearson published in the <em>Philosophical Magazine</em>: "On Lines and Planes of Closest Fit to Systems of Points in Space." The problem Pearson was attacking was geometric: given a scatter of points in three-dimensional space, find the line that minimizes the sum of squared perpendicular distances from the points to the line. He called this the "line of closest fit." Then find the plane of closest fit. The generalization to arbitrary dimension followed immediately. What he had invented was a way to compress a cloud of points by replacing it with a lower-dimensional linear subspace that captures as much of the cloud's spread as possible. His paper established the central insight that persists into every modern implementation: the directions of maximum variance are orthogonal, they can be found analytically, and they rank-order the variance they explain.
      </Prose>

      <Prose>
        Pearson's geometric formulation sat mostly dormant in statistics for three decades. In 1933, Harold Hotelling independently derived the same technique from a completely different direction. His paper "Analysis of a Complex of Statistical Variables into Principal Components," published in two parts in the <em>Journal of Educational Psychology</em> (24(6):417–441 and 24(7):498–520), reframed the problem in terms of variance maximization and introduced the term "principal components" that we still use. Hotelling's formulation was explicitly statistical — he showed how to compute the components from a sample covariance matrix and how to interpret each component as a new uncorrelated variable. The combination of Pearson's geometry and Hotelling's statistics gives PCA its dual nature: it is simultaneously a rotation of the coordinate system and a variance decomposition.
      </Prose>

      <Prose>
        The reasons to use dimensionality reduction in practice are multiple and distinct. First, visualization: you cannot plot 100-dimensional data, but you can project it to 2 or 3 dimensions and see whether clusters, gradients, or outliers are present. Second, compression: a dataset with 10,000 features where 3 principal components capture 95% of the variance can be represented at 1/3,000th the storage cost with minimal information loss. Third, noise reduction: if the signal lives in a low-dimensional subspace and the noise is spread across all dimensions, projecting onto the signal subspace discards the noise. Fourth, preprocessing for supervised learning: many algorithms — regularized regression, k-nearest neighbors, kernel methods — degrade gracefully when features are decorrelated and dimensionality is reduced, because the geometry becomes better-conditioned. Fifth, revealing latent structure: the principal components of a term-document matrix are latent topics; the principal components of a gene expression matrix are latent cell-type signatures; the principal components of a sensor array are latent physical modes.
      </Prose>

      <Prose>
        The tension at the heart of every dimensionality reduction method is between faithfulness to the original data and the cost of representation. PCA resolves this tension by making a specific assumption: the structure that matters is linear. The low-dimensional subspace it finds is flat — a hyperplane, not a curve. This assumption works extraordinarily well on many problems and fails conspicuously on others. Understanding exactly when it holds and when it breaks is the main practical lesson of this topic.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Imagine a cloud of points scattered through 2D space. The cloud is elongated — it stretches more in one direction than in the other. PCA asks: what is the direction along which the points are most spread out? That direction is the first principal component. Then it asks: among all directions orthogonal to the first, which one has the most spread? That is the second principal component. Continue until you have as many components as dimensions. The components are orthogonal by construction, and they rank-order the variance they explain: the first component always explains more variance than the second, which explains more than the third, and so on.
      </Prose>

      <Prose>
        The key geometric insight is that "direction of maximum variance" and "axis of best linear fit" are the same thing. Pearson's perpendicular distance minimization and Hotelling's variance maximization are two ways of describing the same hyperplane. The first principal component is the line through the mean of the data along which the orthogonal projections of the data points are most spread out. Equivalently, it is the line that minimizes the total squared perpendicular distances from the data points to the line.
      </Prose>

      <Plot
        label="2D scatter with principal axes overlay"
        xLabel="x₁"
        yLabel="x₂"
        series={[
          {
            name: "data (correlated)",
            color: colors.gold,
            points: [
              [-1.88, 0.37], [2.58, 2.07], [3.24, 2.51], [-3.25, -2.39],
              [1.43, 1.24], [0.34, 1.16], [-1.79, -1.05], [-2.28, -2.21],
              [-1.00, -0.53], [2.04, 0.32], [0.09, 0.67], [2.16, 1.85],
              [0.72, -0.07], [-1.71, -0.06], [-0.56, -0.15], [1.26, -0.25],
              [-0.45, -0.03], [0.09, -0.13], [-0.76, 0.07], [-0.39, -0.63],
              [1.78, 1.71], [0.11, 0.42], [0.97, 0.37], [-0.56, 0.38],
              [2.55, 1.64], [-0.96, -1.51], [-0.78, -0.73], [4.08, 3.87],
              [-2.97, -1.68], [0.40, 1.03],
            ],
          },
          {
            name: "PC1 — max variance axis",
            color: colors.green,
            points: [[2.91, 2.21], [-2.91, -2.21]],
          },
          {
            name: "PC2 — orthogonal axis",
            color: "#a78bfa",
            points: [[-0.80, 1.05], [0.80, -1.05]],
          },
        ]}
      />

      <Prose>
        In the plot above, the gold points are 30 samples drawn from a 2D Gaussian with covariance <Code>{"[[3, 2], [2, 2]]"}</Code> — a strongly correlated distribution. PC1 (green) runs along the direction of maximum spread, capturing 88.5% of the total variance. PC2 (purple) is orthogonal to PC1 and captures the remaining 11.5%. After projecting onto PC1 only, each 2D point becomes a single number — its coordinate along the green axis — and the reconstruction error (mean squared distance back to the original points) is 0.2163 per sample.
      </Prose>

      <Prose>
        The practical consequence is compression with controlled information loss. If the data has strong linear correlations — which most real datasets do — then the first few principal components capture the overwhelming majority of the variance. The remaining components are noise, and discarding them both reduces storage cost and improves downstream model performance by removing spurious dimensions the model might otherwise overfit to.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Mean-centering and the covariance matrix</H3>

      <Prose>
        Let <Code>X</Code> be an <Code>n × d</Code> data matrix (rows are samples, columns are features). The first step is always mean-centering: subtract the column means so that the data has zero mean in every dimension. Let <Code>X&#772;</Code> denote the mean-centered matrix.
      </Prose>

      <MathBlock>
        {"\\bar{X}_{ij} = X_{ij} - \\bar{x}_j, \\quad \\bar{x}_j = \\frac{1}{n}\\sum_{i=1}^n X_{ij}"}
      </MathBlock>

      <Prose>
        The sample covariance matrix is then:
      </Prose>

      <MathBlock>
        {"\\Sigma = \\frac{1}{n-1}\\bar{X}^\\top \\bar{X}"}
      </MathBlock>

      <Prose>
        This is a <Code>d × d</Code> symmetric positive semi-definite matrix. Its <Code>(i, j)</Code> entry is the sample covariance between feature <Code>i</Code> and feature <Code>j</Code>. The diagonal entries are the sample variances of each feature. If features are uncorrelated, <Code>Σ</Code> is diagonal. PCA finds a rotation that makes the transformed features uncorrelated — it diagonalizes the covariance matrix.
      </Prose>

      <Callout type="info" title="Why centering matters">
        Skipping mean-centering is one of the most common PCA implementation bugs. If the data is not centered, the first principal component will point toward the centroid of the data cloud rather than along the direction of maximum variance within the cloud. The variance of a random variable X is E[X²] − (E[X])², and if E[X] ≠ 0, the two terms interfere. Centering sets E[X] = 0, so variance and second moment coincide. Numerically, failing to center means PCA finds components that explain the offset of the mean from zero rather than the shape of the distribution.
      </Callout>

      <H3>3.2 Eigendecomposition and variance maximization</H3>

      <Prose>
        PCA seeks a unit vector <Code>w</Code> (the first principal component direction) that maximizes the variance of the projected data:
      </Prose>

      <MathBlock>
        {"\\max_{w} \\; \\text{Var}(\\bar{X}w) = \\max_{w} \\; w^\\top \\Sigma w \\quad \\text{subject to} \\quad \\|w\\| = 1"}
      </MathBlock>

      <Prose>
        Using the method of Lagrange multipliers, introduce the constraint via a multiplier <Code>λ</Code>:
      </Prose>

      <MathBlock>
        {"\\mathcal{L}(w, \\lambda) = w^\\top \\Sigma w - \\lambda(w^\\top w - 1)"}
      </MathBlock>

      <Prose>
        Taking the gradient and setting it to zero:
      </Prose>

      <MathBlock>
        {"\\nabla_w \\mathcal{L} = 2\\Sigma w - 2\\lambda w = 0 \\quad \\Rightarrow \\quad \\Sigma w = \\lambda w"}
      </MathBlock>

      <Prose>
        This is the eigenvalue equation. The optimal <Code>w</Code> is an eigenvector of <Code>Σ</Code>, and the variance it captures is the corresponding eigenvalue <Code>λ</Code>: substituting back, <Code>w⊤Σw = w⊤λw = λ</Code>. To maximize variance, choose the eigenvector corresponding to the largest eigenvalue. The second principal component is the eigenvector corresponding to the second-largest eigenvalue, constrained to be orthogonal to the first (the eigenvectors of a symmetric matrix are automatically orthogonal). The full eigendecomposition:
      </Prose>

      <MathBlock>
        {"\\Sigma = V \\Lambda V^\\top"}
      </MathBlock>

      <Prose>
        where <Code>V = [v₁ | v₂ | ... | vd]</Code> is the orthonormal matrix of eigenvectors (principal component directions) and <Code>Λ = diag(λ₁, λ₂, ..., λd)</Code> is the diagonal matrix of eigenvalues in descending order. The projection of the mean-centered data onto the top-<Code>k</Code> components is:
      </Prose>

      <MathBlock>
        {"Z = \\bar{X} V_k, \\quad V_k = [v_1 \\; v_2 \\; \\cdots \\; v_k]"}
      </MathBlock>

      <Prose>
        The explained variance ratio of component <Code>j</Code> is <Code>λⱼ / Σλᵢ</Code> — the fraction of total variance captured by that component. The reconstruction of the original data from the top-<Code>k</Code> components is:
      </Prose>

      <MathBlock>
        {"\\hat{X} = Z V_k^\\top = \\bar{X} V_k V_k^\\top"}
      </MathBlock>

      <H3>3.3 Connection to SVD</H3>

      <Prose>
        There is a cleaner, numerically stabler way to compute PCA that avoids forming the covariance matrix explicitly. The Singular Value Decomposition (SVD) of the mean-centered data matrix is:
      </Prose>

      <MathBlock>
        {"\\bar{X} = U \\Sigma_s V^\\top"}
      </MathBlock>

      <Prose>
        where <Code>U</Code> is <Code>n × n</Code> orthogonal (left singular vectors), <Code>Σs</Code> is <Code>n × d</Code> with non-negative diagonal entries (singular values <Code>σ₁ ≥ σ₂ ≥ ... ≥ 0</Code>), and <Code>V</Code> is <Code>d × d</Code> orthogonal (right singular vectors). The connection to eigendecomposition:
      </Prose>

      <MathBlock>
        {"\\frac{1}{n-1}\\bar{X}^\\top \\bar{X} = \\frac{1}{n-1}(U\\Sigma_s V^\\top)^\\top(U\\Sigma_s V^\\top) = V \\frac{\\Sigma_s^2}{n-1} V^\\top"}
      </MathBlock>

      <Prose>
        The right singular vectors (columns of <Code>V</Code>, or rows of <Code>Vᵀ</Code>) are the principal component directions. The eigenvalues of <Code>Σ</Code> are the squared singular values divided by <Code>(n−1)</Code>: <Code>λⱼ = σⱼ² / (n−1)</Code>. In practice, always use SVD rather than explicit eigendecomposition of <Code>ΣXX</Code>. Forming <Code>XᵀX</Code> squares the condition number of the problem, so floating-point errors are amplified before eigendecomposition even starts. SVD avoids this by working directly with <Code>X</Code>.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below uses NumPy only — no scikit-learn. Every output is verbatim terminal output from a verified run. We implement PCA via both eigendecomposition and SVD on the same dataset to confirm they produce identical results, then run on a higher-dimensional dataset to show the scree plot pattern.
      </Prose>

      <H3>4a. PCA via eigendecomposition</H3>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)
n = 100
# Correlated 2D data: covariance [[3,2],[2,2]]
X = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], n)

# Step 1: mean-center
X_c = X - X.mean(axis=0)
print(f"Mean before centering: {X.mean(axis=0).round(4)}")
# Output: Mean before centering: [-0.0481  0.0217]
print(f"Mean after centering:  {X_c.mean(axis=0).round(8)}")
# Output: Mean after centering:  [-0.  0.]

# Step 2: covariance matrix (d x d = 2 x 2 here)
cov = (X_c.T @ X_c) / (n - 1)
print("Covariance matrix Sigma:")
print(cov.round(4))
# Covariance matrix Sigma:
# [[2.2817 1.4008]
#  [1.4008 1.5006]]

# Step 3: eigendecompose (eigh is for symmetric matrices — numerically better)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# eigh returns ascending order — reverse to descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]  # each column is a PC direction

print(f"Eigenvalues: {eigenvalues.round(4)}")
# Output: Eigenvalues: [3.3454 0.4369]
print("Eigenvectors (columns = PC directions):")
print(eigenvectors.round(4))
# Eigenvectors (columns = PC directions):
# [[-0.7964  0.6048]
#  [-0.6048 -0.7964]]

# Explained variance ratio
evr = eigenvalues / eigenvalues.sum()
print(f"Explained variance ratio: {evr.round(4)}")
# Output: Explained variance ratio: [0.8845 0.1155]
print(f"Cumulative:               {np.cumsum(evr).round(4)}")
# Output: Cumulative:               [0.8845 1.    ]

# Step 4: project onto top-1 PC
X_proj = X_c @ eigenvectors[:, :1]   # shape (100, 1)
print(f"Projected shape: {X_proj.shape}")
# Output: Projected shape: (100, 1)

# Reconstruction from 1 PC
X_recon = X_proj @ eigenvectors[:, :1].T
recon_mse = np.mean((X_c - X_recon) ** 2)
print(f"Reconstruction MSE (1 PC): {recon_mse:.4f}")
# Output: Reconstruction MSE (1 PC): 0.2163

# Full reconstruction (2 PCs) — should recover exactly
X_recon_full = (X_c @ eigenvectors) @ eigenvectors.T
print(f"Reconstruction MSE (2 PCs): {np.mean((X_c - X_recon_full)**2):.8f}")
# Output: Reconstruction MSE (2 PCs): 0.00000000`}
      </CodeBlock>

      <H3>4b. PCA via SVD — numerically stabler</H3>

      <CodeBlock language="python">
{`# Same dataset X_c from above

# SVD of mean-centered matrix (never form X^T X explicitly)
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
# U: (100, 2), S: (2,) singular values, Vt: (2, 2) — rows are PC directions

print(f"Singular values: {S.round(4)}")
# Output: Singular values: [18.1987  6.5769]

# Eigenvalues from singular values: lambda_j = sigma_j^2 / (n-1)
eigenvalues_svd = (S ** 2) / (n - 1)
print(f"Eigenvalues (from SVD): {eigenvalues_svd.round(4)}")
# Output: Eigenvalues (from SVD): [3.3454 0.4369]

evr_svd = eigenvalues_svd / eigenvalues_svd.sum()
print(f"Explained variance ratio: {evr_svd.round(4)}")
# Output: Explained variance ratio: [0.8845 0.1155]

# PC directions are rows of Vt (columns of V)
print("PC directions (rows of Vt):")
print(Vt.round(4))
# PC directions (rows of Vt):
# [[ 0.7964  0.6048]
#  [ 0.6048 -0.7964]]
# Note: sign flips relative to eigh are normal — eigenvectors have sign ambiguity.
# The subspace they span is identical.

# Project onto top-1 PC
X_proj_svd = X_c @ Vt[:1].T
X_recon_svd = X_proj_svd @ Vt[:1]
print(f"Reconstruction MSE (SVD, 1 PC): {np.mean((X_c - X_recon_svd)**2):.4f}")
# Output: Reconstruction MSE (SVD, 1 PC): 0.2163
# Identical to eigendecomposition — different path, same answer.`}
      </CodeBlock>

      <H3>4c. Higher-dimensional PCA and scree pattern</H3>

      <CodeBlock language="python">
{`# 10-feature dataset where first 3 dimensions dominate
np.random.seed(42)
n, d = 200, 10
X_raw = np.random.randn(n, d)
# Scale: first 3 dims have high variance, rest are noise
scale = np.array([5.0, 4.0, 3.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
X_high = X_raw * scale
X_hc = X_high - X_high.mean(axis=0)

U, S, Vt = np.linalg.svd(X_hc, full_matrices=False)
ev = (S ** 2) / (n - 1)
evr = ev / ev.sum()
cumevr = np.cumsum(evr)

print("PC  | Expl. Var | Cumulative")
for i in range(d):
    print(f"PC{i+1:02d}|  {evr[i]:.4f}   |  {cumevr[i]:.4f}")
# PC  | Expl. Var | Cumulative
# PC01|  0.4001   |  0.4001
# PC02|  0.3647   |  0.7648
# PC03|  0.1613   |  0.9261
# PC04|  0.0203   |  0.9464
# PC05|  0.0195   |  0.9659
# PC06|  0.0168   |  0.9827
# PC07|  0.0051   |  0.9877
# PC08|  0.0046   |  0.9923
# PC09|  0.0040   |  0.9963
# PC10|  0.0037   |  1.0000

# Top-3 PCs capture 92.6% of variance
X_proj3 = X_hc @ Vt[:3].T
X_recon3 = X_proj3 @ Vt[:3]
mse3 = np.mean((X_hc - X_recon3) ** 2)
print(f"\nReconstruction MSE (3 PCs / 10 features): {mse3:.4f}")
# Output: Reconstruction MSE (3 PCs / 10 features): 0.3859
print(f"Variance explained by top-3: {cumevr[2]:.4f}")
# Output: Variance explained by top-3: 0.9261
print(f"Variance explained by top-5: {cumevr[4]:.4f}")
# Output: Variance explained by top-5: 0.9659`}
      </CodeBlock>

      <Prose>
        The scree plot pattern is the hallmark of structured data: a sharp drop after the first few components (where real signal lives) followed by a flat plateau (noise). On random isotropic Gaussian data, all eigenvalues are equal and the scree plot is flat. The presence of an "elbow" is empirical evidence that the data has low-dimensional structure.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Scikit-learn's <Code>sklearn.decomposition.PCA</Code> is the standard production choice. It handles mean-centering internally, exposes <Code>explained_variance_ratio_</Code> directly, and selects the right SVD algorithm based on dataset shape. The API follows the standard sklearn pattern: <Code>fit</Code>, <Code>transform</Code>, <Code>fit_transform</Code>, <Code>inverse_transform</Code>.
      </Prose>

      <H3>5a. sklearn PCA — core API</H3>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data   # shape (150, 4): sepal len, sepal wid, petal len, petal wid

# Always standardize before PCA when features have different units/scales.
# Iris features are all in cm and similarly scaled — but this is best practice.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Full PCA: all 4 components ---
pca_full = PCA(n_components=4)
pca_full.fit(X_scaled)

print("explained_variance_ratio_:", pca_full.explained_variance_ratio_.round(4))
# Output: explained_variance_ratio_: [0.7296 0.2285 0.0367 0.0052]
print("cumulative:                ", pca_full.explained_variance_ratio_.cumsum().round(4))
# Output: cumulative:                 [0.7296 0.9581 0.9948 1.    ]
print("singular_values_:          ", pca_full.singular_values_.round(4))
# Output: singular_values_:           [20.9231 11.7092  4.6919  1.7627]

# --- Top-2 PCA for visualization ---
pca2 = PCA(n_components=2, svd_solver='full')
X_2d = pca2.fit_transform(X_scaled)
print(f"Shape after 2-component PCA: {X_2d.shape}")
# Output: Shape after 2-component PCA: (150, 2)
print(f"Variance explained by 2 PCs: {pca2.explained_variance_ratio_.sum():.4f}")
# Output: Variance explained by 2 PCs: 0.9581

# Reconstruction error (2 PCs on 4-feature data)
X_recon = pca2.inverse_transform(X_2d)
recon_mse = np.mean((X_scaled - X_recon) ** 2)
print(f"Reconstruction MSE (2 PCs): {recon_mse:.4f}")
# Output: Reconstruction MSE (2 PCs): 0.0419

# Components (loadings): shape (n_components, n_features)
# Each row is a PC direction in original feature space
print("Components (loadings matrix):")
print(pca_full.components_.round(4))
# Components (loadings matrix):
# [[ 0.5211 -0.2693  0.5804  0.5649]
#  [ 0.3774  0.9233  0.0245  0.0669]
#  [ 0.7196 -0.2444 -0.1421 -0.6343]
#  [-0.2613  0.1235  0.8014 -0.5236]]`}
      </CodeBlock>

      <H3>5b. svd_solver options, IncrementalPCA, KernelPCA</H3>

      <CodeBlock language="python">
{`from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

# svd_solver options:
# 'full'        — LAPACK full SVD. Exact. O(n*d^2 + d^3). Best for small d.
# 'randomized'  — Halko-Martinsson-Tropp 2011 randomized SVD. O(n*d*k) time.
#                 Excellent for n >> d and k << d. Default for large datasets.
# 'arpack'      — Lanczos iteration. Good for sparse X.
# 'auto'        — sklearn picks based on shape (default).

# Randomized PCA — same results, much faster on large matrices
pca_rand = PCA(n_components=2, svd_solver='randomized', random_state=42)
pca_rand.fit(X_scaled)
print("Randomized SVD EVR:", pca_rand.explained_variance_ratio_.round(4))
# Output: Randomized SVD EVR: [0.7296 0.2285]

# whiten=True divides each component by sqrt(eigenvalue), making
# the projected features have unit variance. Useful before ICA or clustering.
pca_w = PCA(n_components=2, whiten=True)
X_white = pca_w.fit_transform(X_scaled)
print(f"Whitened variance: {X_white.var(axis=0).round(4)}")
# Output: Whitened variance: [1. 1.]

# --- IncrementalPCA: out-of-core, processes data in chunks ---
# Useful when full dataset doesn't fit in RAM. Streams mini-batches.
ipca = IncrementalPCA(n_components=2, batch_size=30)
X_ipca = ipca.fit_transform(X_scaled)
print("IncrementalPCA EVR:", ipca.explained_variance_ratio_.round(4))
# Output: IncrementalPCA EVR: [0.7291 0.2285]
# Slightly different from full PCA because batch processing introduces
# numerical differences — not a bug.

# --- KernelPCA: nonlinear dimensionality reduction via kernel trick ---
# Maps data to a high-dim feature space then applies PCA.
# kernel options: 'rbf', 'poly', 'sigmoid', 'cosine'
# O(n^2) memory — only feasible for n < ~10k.
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
X_kpca = kpca.fit_transform(X_scaled)
print(f"KernelPCA shape: {X_kpca.shape}")
# Output: KernelPCA shape: (150, 2)
# KernelPCA has no explained_variance_ratio_ by default (no guarantee of ordering
# by variance in the input space after kernel transformation).`}
      </CodeBlock>

      <Callout type="info" title="Choosing n_components">
        Three practical rules. (1) Variance threshold: choose the smallest k such that the cumulative explained variance exceeds a target, typically 95% or 99%. (2) Elbow method: plot explained variance ratio vs. component index (scree plot), look for a kink where the curve flattens. Components before the kink capture signal; after, noise. (3) Downstream task: if PCA is preprocessing for a supervised model, choose k via cross-validation on the downstream metric rather than on explained variance. Sometimes 85% explained variance is enough; sometimes you need 99%.
      </Callout>

      <Prose>
        Beyond sklearn, two modern alternatives deserve mention. Random projections (<Code>sklearn.random_projection.GaussianRandomProjection</Code>) are theoretically motivated by the Johnson-Lindenstrauss lemma: a random linear map from high to low dimension approximately preserves pairwise distances with high probability. They are even faster than randomized PCA (no data-dependent computation) but sacrifice the variance-maximization property — the projected dimensions are not ordered by importance. Autoencoders generalize PCA to nonlinear compression: an encoder network maps inputs to a bottleneck, a decoder reconstructs, and the whole system is trained end-to-end. They capture nonlinear structure that linear PCA cannot, at the cost of requiring labeled-style training, a fixed architecture, and GPU compute.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6a. Scree plot</H3>

      <Plot
        label="Scree plot — variance explained per component (10-feature dataset)"
        xLabel="principal component index"
        yLabel="explained variance ratio"
        series={[
          {
            name: "individual variance",
            color: colors.gold,
            points: [
              [1, 0.4001], [2, 0.3647], [3, 0.1613], [4, 0.0203],
              [5, 0.0195], [6, 0.0168], [7, 0.0051], [8, 0.0046],
              [9, 0.0040], [10, 0.0037],
            ],
          },
          {
            name: "cumulative variance",
            color: colors.green,
            points: [
              [1, 0.4001], [2, 0.7648], [3, 0.9261], [4, 0.9464],
              [5, 0.9659], [6, 0.9827], [7, 0.9877], [8, 0.9923],
              [9, 0.9963], [10, 1.0000],
            ],
          },
        ]}
      />

      <Prose>
        The elbow at component 3 is unmistakable. The first three components together explain 92.6% of total variance; the remaining seven explain 7.4%. This is the scree plot signature of structured data: a sharp descent (the signal), then a flat tail (the noise). The word "scree" comes from geology — it is the loose rock debris that accumulates at the base of a cliff, flat and undifferentiated compared to the cliff face above. The elbow is where the cliff meets the scree.
      </Prose>

      <H3>6b. Loading heatmap (Iris dataset)</H3>

      <Prose>
        The component loadings matrix tells you which original features contribute to each principal component. For the standardized Iris dataset (4 features: sepal length, sepal width, petal length, petal width), the 4 PCs have the following loadings. Strong positive loadings mean the feature contributes positively to that PC's score; strong negative loadings contribute negatively.
      </Prose>

      <Heatmap
        label="PCA loadings — Iris dataset (rows=PCs, cols=features)"
        rowLabels={["PC1 (72.96%)", "PC2 (22.85%)", "PC3 (3.67%)", "PC4 (0.52%)"]}
        colLabels={["sepal len", "sepal wid", "petal len", "petal wid"]}
        matrix={[
          [0.52, -0.27, 0.58, 0.56],
          [0.38, 0.92, 0.02, 0.07],
          [0.72, -0.24, -0.14, -0.63],
          [-0.26, 0.12, 0.80, -0.52],
        ]}
        colorScale="gold"
      />

      <Prose>
        PC1 loads positively on sepal length, petal length, and petal width — it is a "size" component that captures the overall scale of the flower. PC2 loads almost entirely on sepal width — it captures the aspect ratio of the sepals, independent of overall size. This decomposition is not imposed by the algorithm; it emerges from the correlations in the data. The fact that PC1 is a "size" axis and PC2 is a "shape" axis is a biological fact about irises that PCA reveals automatically.
      </Prose>

      <H3>6c. Manual PCA trace — step by step on 5 points</H3>

      <StepTrace
        label="Manual PCA — 5-point 2D toy dataset"
        steps={[
          {
            label: "Step 0 — raw data",
            render: () => (
              <Prose>
                Five points: (2.5, 2.4), (0.5, 0.7), (2.2, 2.9), (1.9, 2.2), (3.1, 3.0). Both features are positively correlated and share similar scale. Before any computation, note that the cloud looks elongated along the diagonal — a strong hint that the first PC will point roughly along x₁ = x₂.
              </Prose>
            ),
          },
          {
            label: "Step 1 — mean-center",
            render: () => (
              <Prose>
                Column means: x̄₁ = 2.04, x̄₂ = 2.24. Subtract from each point. Mean-centered points: (0.46, 0.16), (−1.54, −1.54), (0.16, 0.66), (−0.14, −0.04), (1.06, 0.76). The cloud now sits at the origin. Centering is mandatory — skipping it would make the "direction of maximum variance" point toward the centroid (2.04, 2.24) instead of along the cloud's actual elongation axis.
              </Prose>
            ),
          },
          {
            label: "Step 2 — covariance matrix",
            render: () => (
              <Prose>
                Covariance matrix Σ = X̄ᵀX̄ / (n−1) = X̄ᵀX̄ / 4. Computing the four entries: Σ₁₁ = Var(x₁) = 0.9380, Σ₂₂ = Var(x₂) = 0.8530, Σ₁₂ = Cov(x₁,x₂) = 0.8405. The off-diagonal value 0.8405 is nearly as large as the diagonal entries, confirming strong positive correlation between the two features. A correlation coefficient r = 0.8405 / sqrt(0.9380 × 0.8530) ≈ 0.939.
              </Prose>
            ),
          },
          {
            label: "Step 3 — eigendecompose",
            render: () => (
              <Prose>
                Eigenvalues: λ₁ = 1.7371, λ₂ = 0.0539. PC1 direction: [−0.7247, −0.6890] (normalized). PC2 direction: [0.6890, −0.7247] (orthogonal to PC1). Explained variance ratio: PC1 = 1.7371/1.7910 = 96.99%, PC2 = 3.01%. A single component captures 97% of the variance in a 2D dataset — this is unusually high, but a natural consequence of the near-perfect correlation (r ≈ 0.94) we observed in the covariance matrix.
              </Prose>
            ),
          },
          {
            label: "Step 4 — project onto PC1",
            render: () => (
              <Prose>
                Projection scores (dot product of each centered point with PC1 direction [−0.7247, −0.6890]): Point 1: −0.4436. Point 2: 2.1772. Point 3: −0.5707. Point 4: 0.1290. Point 5: −1.2919. These five numbers are the complete representation of the dataset in 1D. Point 2 (originally at (0.5, 0.7), the "lower-left" outlier) has the largest positive score because it is the furthest from the mean along PC1. The sign is negative for PC1 direction — the algorithm finds directions up to sign, and both [−0.7247, −0.6890] and [0.7247, 0.6890] span the same axis.
              </Prose>
            ),
          },
          {
            label: "Step 5 — reconstruct",
            render: () => (
              <Prose>
                Reconstruction from 1 PC: multiply each score back by the PC1 direction and add the mean. For point 1: score × PC1 + mean = −0.4436 × [−0.7247, −0.6890] + [2.04, 2.24] = [0.321, 0.306] + [2.04, 2.24] = [2.361, 2.546]. Original was (2.5, 2.4) — reconstruction error is small. For point 2 (score 2.1772): reconstructed ≈ (0.42, 0.74), original (0.5, 0.7) — excellent. The 97% explained variance means the reconstruction is nearly perfect everywhere in this dataset.
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
        PCA is not the only dimensionality reduction method, and for many problems it is not the best one. The choice depends on whether the structure is linear, how large the dataset is, whether you need to generalize to new points, and whether your goal is compression or visualization.
      </Prose>

      <StepTrace
        label="dimensionality reduction method comparison"
        steps={[
          {
            label: "Linear PCA",
            render: () => (
              <Prose>
                When to use: data lies near a linear subspace; you need reproducible components with clear loadings; you need to apply the same transformation to new test points (transform is a fixed linear map); you care about reconstruction fidelity; you are preprocessing before a supervised model. Strengths: exact, deterministic, interpretable loadings, fast (randomized SVD scales to millions of rows), generalizes to new data via matrix multiply. Limitations: only finds linear structure — Swiss roll, nested manifolds, ring clusters are invisible to linear PCA. The components are ordered by variance, not by relevance to a downstream task.
              </Prose>
            ),
          },
          {
            label: "Kernel PCA",
            render: () => (
              <Prose>
                When to use: data lies on a nonlinear manifold but you want a kernel-based method with a known similarity function (RBF, polynomial). The kernel trick maps data implicitly to a high-dimensional feature space and applies linear PCA there. Example: concentric rings are not linearly separable but RBF kernel PCA separates them cleanly. Limitations: O(n²) memory for the kernel matrix — impractical above ~10k samples. No direct way to transform new points without storing the training kernel matrix (use KernelPCA(fit_inverse_transform=True) in sklearn for approximate inverse). No explicit loadings — the components live in feature space, not input space.
              </Prose>
            ),
          },
          {
            label: "Random Projections (Johnson-Lindenstrauss)",
            render: () => (
              <Prose>
                When to use: you need to reduce d to k very fast, you do not need the components to be ordered by variance, and approximate distance preservation is sufficient. The JL lemma guarantees that a random k × d matrix approximately preserves all pairwise distances if k = O(log n / ε²). Strengths: O(n × d × k) time, no data-dependent computation, can be applied without seeing the data first (useful for streaming). Limitations: random projections do not maximize explained variance — you cannot choose k by looking at a scree plot; you need a theoretical or empirical bound on the approximation error. sklearn has GaussianRandomProjection and SparseRandomProjection.
              </Prose>
            ),
          },
          {
            label: "t-SNE / UMAP",
            render: () => (
              <Prose>
                When to use: visualization only — you want to see 2D or 3D structure in high-dimensional data and do not need to apply the same map to new points. t-SNE (van der Maaten and Hinton, 2008) minimizes KL divergence between a student-t distribution over pairwise distances in the low-dim embedding and a Gaussian distribution over distances in the original space. UMAP (McInnes et al., 2018) is faster and preserves global structure better. Limitations: both are nonparametric and stochastic — running twice with different seeds gives different embeddings. Neither has a closed-form inverse. They are not suitable for compression or preprocessing before supervised learning. Distances and scales in the 2D embedding are not interpretable. Both are O(n log n) in practice.
              </Prose>
            ),
          },
          {
            label: "Autoencoder",
            render: () => (
              <Prose>
                When to use: very high dimensional data (images, text embeddings) with learnable nonlinear structure; you have enough labeled or unlabeled data to train a neural network; you need a reconstruction objective with high fidelity. An autoencoder is a neural network with a bottleneck: encoder maps input to a low-dim code, decoder reconstructs. Trained end-to-end by minimizing reconstruction loss. Variational autoencoders (VAEs) impose a structured prior on the code space. Strengths: captures highly nonlinear structure, reconstruction can be near-perfect for natural images. Limitations: requires neural network training infrastructure; hyperparameter-sensitive; no interpretable loadings; overfits on small datasets; no guarantee of ordered components by variance.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>8.1 Computational complexity of full PCA</H3>

      <Prose>
        Full PCA via exact SVD costs <Code>O(min(n·d², d·n²))</Code>. For the common case where <Code>n {">"} d</Code> (more samples than features), the bottleneck is forming <Code>XᵀX</Code> at <Code>O(n·d²)</Code> and computing its eigendecomposition at <Code>O(d³)</Code>. The total is <Code>O(n·d² + d³)</Code>. For typical tabular data with <Code>n = 10,000</Code> and <Code>d = 100</Code>, this is trivially fast. At <Code>n = 100,000</Code> and <Code>d = 10,000</Code>, forming <Code>XᵀX</Code> is <Code>10⁵ × 10⁸ = 10¹³</Code> flops — feasible but slow. At <Code>d = 100,000</Code> (gene expression, text), the <Code>d × d</Code> covariance matrix requires <Code>10¹⁰ × 8</Code> bytes = 80 GB of memory for float64. This is the wall where full PCA breaks down.
      </Prose>

      <H3>8.2 Randomized SVD — the scalable path</H3>

      <Prose>
        The Halko-Martinsson-Tropp 2011 algorithm "Finding Structure with Randomness" (SIAM Review 53(2):217–288) provides a randomized SVD that computes an approximate rank-k SVD of an <Code>n × d</Code> matrix in <Code>O(n·d·k)</Code> time and <Code>O((n+d)·k)</Code> memory. The algorithm works in two stages: first, compute a random sketch <Code>Y = X · Ω</Code> where <Code>Ω</Code> is a random Gaussian matrix of shape <Code>d × (k+p)</Code> (p is a small oversampling parameter, typically 10); second, compute the SVD of a much smaller matrix derived from <Code>Y</Code>. For <Code>k = 50</Code> components from a <Code>1,000,000 × 10,000</Code> matrix, randomized SVD is tractable on a single machine; exact SVD is not. sklearn uses this algorithm automatically via <Code>svd_solver='randomized'</Code> and defaults to it when <Code>n_components</Code> is much smaller than <Code>min(n, d)</Code>.
      </Prose>

      <H3>8.3 IncrementalPCA for streaming</H3>

      <Prose>
        When the full dataset does not fit in RAM, IncrementalPCA processes it in chunks using an online version of the PCA algorithm. Each batch updates the running estimates of the principal components without ever holding the full data in memory. The memory footprint is <Code>O(batch_size × d + k × d)</Code> — linear in the batch size and the number of components. The result is slightly less accurate than full PCA (due to the batching approximation) but is the only option for streaming or very large datasets. sklearn's <Code>IncrementalPCA</Code> on the iris dataset with batch_size=30 gives explained_variance_ratio_ = [0.7291, 0.2285] versus the full PCA result of [0.7296, 0.2285] — a difference of 0.05 percentage points.
      </Prose>

      <H3>8.4 Scale comparison — choosing the right method</H3>

      <Prose>
        A rough guide by dataset size. For <Code>n {"<"} 10,000</Code> and <Code>d {"<"} 1,000</Code>: use full PCA (<Code>svd_solver='full'</Code>) — exact and fast. For <Code>n</Code> up to millions, <Code>d {"<"} 10,000</Code>, <Code>k {"<"} d/5</Code>: use randomized PCA (<Code>svd_solver='randomized'</Code>) — near-exact, scales. For <Code>n</Code> too large for RAM: use IncrementalPCA with a reasonable batch size. For Kernel PCA: <Code>O(n²)</Code> memory means hard ceiling around <Code>n ≈ 10,000</Code>. For t-SNE: <Code>O(n²)</Code> naive, <Code>O(n log n)</Code> with Barnes-Hut approximation, practical ceiling around <Code>n ≈ 100,000</Code>. For UMAP: <Code>O(n log n)</Code>, scales to millions. For autoencoders: scales to any size with mini-batch training, GPU memory is the constraint.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Forgetting to mean-center</H3>

      <Prose>
        This is the most common implementation error. If you forget to subtract column means before computing the covariance matrix or running SVD, the first principal component will not be the direction of maximum variance within the cloud of points — it will be the direction from the origin toward the centroid of the cloud. For data where the mean is large relative to the spread, this first "component" is essentially a mean vector, and the remaining components are the actual principal axes. sklearn's PCA handles this automatically, but if you are implementing from scratch or using a custom SVD routine, always center first. The fix is one line: <Code>X -= X.mean(axis=0)</Code>.
      </Prose>

      <H3>9.2 Scale sensitivity — standardize before PCA</H3>

      <Prose>
        PCA maximizes variance. If one feature has a variance of 10,000 (say, income in dollars) and another has a variance of 1 (say, a 0–1 rating), the first principal component will be almost entirely aligned with the income feature — not because income is more informative, but because it is numerically larger. This is almost never what you want. The fix: standardize every feature to zero mean and unit variance (<Code>StandardScaler</Code>) before applying PCA. After standardization, all features contribute on equal footing and the components reflect genuine correlational structure rather than numerical scale. Exception: if features are already on the same scale and their variance differences are meaningful (e.g., gene expression in the same units where high-variance genes are biologically significant), do not standardize — the raw covariance PCA captures the right structure.
      </Prose>

      <H3>9.3 Interpreting PCs as real features</H3>

      <Prose>
        Principal components are linear combinations of original features. They are mathematical constructs that maximize variance, not meaningful entities in the real world. The temptation is to name them ("PC1 is size, PC2 is shape") and treat them as if they were real quantities. This works sometimes — the Iris PC1 being a "size" axis is a genuine biological pattern. But it is coincidental, not guaranteed. In high dimensions, PCs are often uninterpretable combinations of dozens of features with small, similar loadings. Naming them is speculative unless you have independent domain evidence. Never drop original features from your interpretation in favor of PCs without acknowledging that the mapping is lossy and basis-dependent.
      </Prose>

      <H3>9.4 Sign ambiguity</H3>

      <Prose>
        Eigenvectors are defined only up to a sign flip: if <Code>v</Code> is an eigenvector, so is <Code>−v</Code>. Both span the same 1D subspace and the explained variance is identical. sklearn resolves this by choosing the sign that makes the largest-magnitude component positive, but different implementations make different choices. The practical consequence: if you compute PCA twice with different libraries (or different sklearn versions), the sign of some components may be flipped, making the projected coordinates differ by a sign. This does not affect clustering, distance-based methods, or reconstruction quality, but it will break naive comparison of coordinates across runs. Always use the projected data (or reconstructions) for downstream tasks rather than relying on the sign of specific PCs.
      </Prose>

      <H3>9.5 Choosing the number of components</H3>

      <Prose>
        The elbow in a scree plot is often ambiguous in practice — real data rarely has a sharp kink. The 95% variance threshold is a rule of thumb, not a theorem. For supervised learning, the right number of components is the one that maximizes validation performance, which can be very different from the one that maximizes explained variance. A dataset where 5 components capture 95% variance might achieve better downstream classification with 20 components, because the low-variance components still carry discriminative information. Treat <Code>n_components</Code> as a hyperparameter to be tuned via cross-validation when the task is supervised. For visualization, use 2 or 3 components because that is what humans can plot.
      </Prose>

      <H3>9.6 PCA is not invariant to feature scaling</H3>

      <Prose>
        Unlike some methods (SVMs with RBF kernel, cosine similarity), PCA produces different outputs when you scale a feature by a constant. Scaling feature <Code>j</Code> by 2 doubles its variance, which changes the covariance matrix and therefore changes all the eigenvectors — not just the contribution of feature <Code>j</Code>. This is a direct consequence of variance maximization: larger numbers dominate. This is not a bug; it is the intended behavior when features represent physically meaningful quantities with natural scales. But it means that the choice to standardize (or not) is a modeling decision that changes the result in a non-trivial way, and you should make that decision consciously rather than by default.
      </Prose>

      <Callout type="warning" title="PCA before train/test split is a data leak">
        If you fit PCA on the full dataset (train + test combined) and then split, you have allowed test-set information to influence the principal component directions. The correct order is: split first, fit PCA on training data only, then apply the fitted transform to test data. sklearn Pipelines enforce this automatically. Doing it manually, always call fit_transform on training data and transform (not fit_transform) on test data.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All citations below are WebSearch-verified for author, year, venue, and core claims. Read in this order to follow the intellectual lineage.
      </Prose>

      <StepTrace
        label="primary literature"
        steps={[
          {
            label: "Pearson 1901 — On Lines and Planes of Closest Fit",
            render: () => (
              <Prose>
                Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of Points in Space." <em>Philosophical Magazine</em>, Series 6, 2(11), 559–572. The founding paper of PCA, written in geometric language. Pearson asks: given a scatter of points in 3D, what line minimizes the sum of squared perpendicular (orthogonal) distances? He generalizes immediately to planes and then arbitrary dimension. The result is what we now call the first principal component (line of closest fit) and first two principal components (plane of closest fit). No matrices, no eigenvalues — the connection to covariance structure was established by Hotelling 32 years later. A remarkable paper to read: it is short, clear, and the core insight is on the first page.
              </Prose>
            ),
          },
          {
            label: "Hotelling 1933 — Analysis of a Complex of Statistical Variables into Principal Components",
            render: () => (
              <Prose>
                Hotelling, H. (1933). "Analysis of a Complex of Statistical Variables into Principal Components." <em>Journal of Educational Psychology</em>, 24(6), 417–441 and 24(7), 498–520. The statistical reformulation of Pearson's geometry. Hotelling introduces the variance-maximization formulation, derives the eigendecomposition of the sample covariance matrix, coins the term "principal components," and shows how to interpret each component as an uncorrelated linear combination of the original variables. The two-part structure reflects the computational labor required in 1933: computing the eigenvectors of a 10×10 correlation matrix by hand took months of arithmetic by "computers" (human calculators). The psychometric application — finding latent factors in test score matrices — remains a live area of research in education.
              </Prose>
            ),
          },
          {
            label: "Jolliffe 2002 — Principal Component Analysis (2nd ed.)",
            render: () => (
              <Prose>
                Jolliffe, I.T. (2002). <em>Principal Component Analysis</em>, 2nd edition. New York: Springer. ISBN: 978-0-387-95442-4. The definitive reference text on PCA. Covers derivation from multiple perspectives (variance maximization, least-squares, singular value decomposition), properties of the sample PCA estimator, inferential procedures, connections to factor analysis and canonical correlation, applications in atmospheric science and biology, and extensions including sparse PCA and functional PCA. If you need to understand a specific aspect of PCA in depth — computational properties, asymptotic theory, choosing the number of components, connections to other methods — this book has the answer. Chapter 5 on "Graphical Representation of Data Using Principal Components" is particularly useful for understanding scree plots and biplots.
              </Prose>
            ),
          },
          {
            label: "Halko, Martinsson, Tropp 2011 — Finding Structure with Randomness",
            render: () => (
              <Prose>
                Halko, N., Martinsson, P.-G., and Tropp, J.A. (2011). "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." <em>SIAM Review</em>, 53(2), 217–288. DOI: 10.1137/090771806. The paper that made large-scale PCA practical. Provides rigorous probabilistic bounds on the error of the randomized SVD algorithm: with high probability, the approximation error is within a small constant of the optimal rank-k approximation. The key algorithmic idea — form a random sketch Y = XΩ, then compute the SVD of a smaller matrix — is implementable in 20 lines of NumPy and is now the default algorithm in sklearn, Spark MLlib, and every serious ML library. The exposition is detailed and accessible to anyone comfortable with linear algebra and probability.
              </Prose>
            ),
          },
          {
            label: "Shlens 2014 — A Tutorial on Principal Component Analysis",
            render: () => (
              <Prose>
                Shlens, J. (2014). "A Tutorial on Principal Component Analysis." arXiv:1404.1100. A 20-page tutorial that connects all the perspectives: variance maximization, covariance diagonalization, SVD, and geometric interpretation. The derivation via diagonalizing the covariance matrix is particularly clean. This is the best single document for understanding why PCA is both a rotation and a compression — it shows explicitly that the eigenvector basis diagonalizes the covariance matrix, making the transformed features uncorrelated by construction. Freely available on arXiv.
              </Prose>
            ),
          },
          {
            label: "van der Maaten & Hinton 2008 — t-SNE (for comparison)",
            render: () => (
              <Prose>
                van der Maaten, L. and Hinton, G. (2008). "Visualizing Data Using t-SNE." <em>Journal of Machine Learning Research</em>, 9(86), 2579–2605. Essential reading for understanding when PCA is the wrong tool. t-SNE reveals local cluster structure that linear PCA cannot: nested manifolds, clusters within clusters, non-convex structures. The paper is also a good case study in the limitations of linear methods: the MNIST digits embedded by PCA show overlapping clouds; embedded by t-SNE, they form clean separate clusters. Understanding both methods together gives you a principled basis for choosing between them: PCA for global linear structure and compression; t-SNE/UMAP for visualization of local nonlinear structure.
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
        Work through these before moving on. The answers are below each exercise — read the question, attempt it, then check.
      </Prose>

      <H3>Exercise 1 (derivation)</H3>
      <Prose>
        Derive the variance-maximization formulation of PCA. Starting from the objective "maximize <Code>wᵀΣw</Code> subject to <Code>‖w‖ = 1</Code>," use the method of Lagrange multipliers to show that the optimal <Code>w</Code> must be an eigenvector of <Code>Σ</Code>. Why does choosing the top eigenvector maximize variance?
      </Prose>
      <Callout type="answer" title="Answer 1">
        Form the Lagrangian: L(w, λ) = wᵀΣw − λ(wᵀw − 1). Take the gradient and set to zero: ∇_w L = 2Σw − 2λw = 0, so Σw = λw. This is the eigenvalue equation — w must be an eigenvector of Σ. The variance captured is wᵀΣw = wᵀ(λw) = λwᵀw = λ. So the variance equals the eigenvalue. To maximize variance, choose the eigenvector corresponding to the largest eigenvalue λ₁. The second component is found by imposing orthogonality to the first (second Lagrange constraint), which gives the second-largest eigenvector. The eigenvectors of a symmetric positive semi-definite matrix are guaranteed to be orthogonal (they are the columns of an orthogonal matrix V in the eigendecomposition Σ = VΛVᵀ), so the constraint is automatically satisfied.
      </Callout>

      <H3>Exercise 2 (conceptual)</H3>
      <Prose>
        A colleague runs PCA on a dataset without standardizing the features first. The first principal component has a loading of 0.999 on "annual salary" (range $0–$500,000) and near-zero loadings on the other 9 features (age, years experience, etc., all in single-digit or small-integer ranges). They report that "salary almost entirely determines the first PC." What went wrong, and how do you fix it?
      </Prose>
      <Callout type="answer" title="Answer 2">
        PCA maximizes variance. Salary has a variance on the order of (50,000)² = 2.5 × 10⁹ in squared dollars. Age has a variance of perhaps (10)² = 100 in squared years. The salary feature dominates the covariance matrix simply because of its numerical scale, not because it carries more information. The first PC finds the direction of the largest numerical spread, which is essentially the salary axis. The other 9 features are squeezed into the remaining components. Fix: apply StandardScaler to subtract the mean and divide by the standard deviation of each feature before PCA. After standardization, every feature has variance 1, and the principal components reflect genuine correlational structure — which features move together — rather than scale. Rule: always standardize before PCA unless features are already on the same physical scale and their variance differences are scientifically meaningful.
      </Callout>

      <H3>Exercise 3 (implementation)</H3>
      <Prose>
        You have an <Code>n × d</Code> matrix with <Code>n = 1,000,000</Code> rows and <Code>d = 5,000</Code> columns. You want the top-50 principal components. (a) Why does full PCA fail? (b) What sklearn class and parameter do you use instead? (c) What is the approximate time complexity?
      </Prose>
      <Callout type="answer" title="Answer 3">
        (a) Full PCA via exact SVD requires forming the d × d covariance matrix: 5,000 × 5,000 = 25 million entries × 8 bytes = 200 MB (manageable), then decomposing it in O(d³) = O(1.25 × 10¹¹) flops — very slow. More critically, computing the full SVD of X directly (1,000,000 × 5,000) requires storing intermediate matrices of size n × min(n,d), which at n = 10⁶ is infeasible. (b) Use PCA(n_components=50, svd_solver='randomized', random_state=42). The randomized SVD algorithm (Halko et al. 2011) computes an approximate rank-50 SVD directly. (c) Time complexity is O(n · d · k) ≈ O(10⁶ × 5,000 × 50) = O(2.5 × 10¹¹) — similar to full SVD in theory, but with much smaller constant factors because the intermediate sketch matrix is (n × (k+p)) ≈ 10⁶ × 60, which is dense but manageable. Memory footprint is O((n+d) × k) = O(10⁶ + 5,000) × 50 ≈ 200 MB.
      </Callout>

      <H3>Exercise 4 (debugging)</H3>
      <Prose>
        You fit PCA on a training set and then apply it to a test set. The first principal component on the test set has a very different distribution than on the training set, even though the test data was drawn from the same process. You check: the number of components is correct, the transform call is applied correctly. What are the two most likely causes, and how do you diagnose each?
      </Prose>
      <Callout type="answer" title="Answer 4">
        Cause 1: You fit StandardScaler (or another preprocessor) separately on the training and test sets, using different means and standard deviations. If the scaler is fit_transformed on training but fit_transformed again on test (instead of just transformed), the normalization differs between sets, which changes the effective data seen by PCA. Diagnosis: print the scaler's mean_ and scale_ and verify they were estimated on training data only. Fix: use a sklearn Pipeline, which guarantees fit on train and transform on test. Cause 2: Distribution shift — the test data is genuinely from a different distribution (different time period, different source, different population). The PCA components are determined by the training data's covariance structure; if the test distribution has a different covariance, projecting onto training PCs can produce unexpected results. Diagnosis: compare summary statistics (means, standard deviations, pairwise correlations) between train and test. If they differ substantially, you have a distribution shift problem that PCA cannot fix — investigate the data collection process.
      </Callout>

      <H3>Exercise 5 (applied)</H3>
      <Prose>
        You use PCA to reduce a face image dataset from 1,024 dimensions (32×32 pixels) to 50 dimensions, then train a k-NN classifier on the compressed representations. Your test accuracy is 82%. A colleague tries the same pipeline without PCA (k-NN on 1,024 dimensions) and gets 79%. Another colleague skips PCA but uses cosine similarity instead of Euclidean distance in k-NN and gets 83%. (a) Why does PCA improve over raw k-NN with Euclidean distance? (b) Why might cosine similarity on raw features outperform PCA + Euclidean? (c) What does this tell you about choosing dimensionality reduction vs. distance normalization?
      </Prose>
      <Callout type="answer" title="Answer 5">
        (a) PCA improves k-NN because it removes noise dimensions. In 1,024-dimensional space, the Euclidean distance between two images is dominated by pixel-level noise and lighting variation — low-variance dimensions that are not related to identity. Projecting onto 50 PCs retains the high-variance dimensions (global shape, lighting direction, face structure) and discards the noise, so the k-NN distances become more semantically meaningful. This is the "curse of dimensionality" in action: in high dimensions, all pairwise distances concentrate near the same value, degrading k-NN performance. (b) Cosine similarity on raw features is equivalent to L2-normalizing each image (making it a unit vector) before computing Euclidean distance. For face images, overall pixel intensity varies with lighting — two photos of the same person under different lighting are far apart in Euclidean distance but close in cosine similarity. Cosine similarity implicitly removes the overall brightness factor, which is a simpler and cheaper normalization than full PCA. (c) PCA and distance normalization attack different problems: PCA removes noise dimensions; cosine similarity removes global scale. Often the right approach is both — standardize features to remove scale, then apply PCA to remove noisy dimensions. In this case, L2-normalization alone happened to capture the most important invariance (lighting) without the complexity of PCA.
      </Callout>

      <H3>Exercise 6 (synthesis)</H3>
      <Prose>
        Explain why PCA fails on the "Swiss roll" dataset (a 2D manifold coiled in 3D space). What does PCA find, and what would you use instead? Sketch — in words — what the Swiss roll looks like after PCA projection to 2D versus after UMAP projection to 2D.
      </Prose>
      <Callout type="answer" title="Answer 6">
        PCA finds the 2D linear subspace of 3D space that captures the most variance. For the Swiss roll, the maximum-variance 2D projection is roughly the plane containing the "unrolled" roll — it sees the coiled structure and projects it down, collapsing points from different parts of the roll that happen to lie near each other in 3D even though they are far apart along the manifold. The result looks like an ellipse with the entire Swiss roll squashed together: the inner coils and outer coils are interleaved. The 2D PCA projection of a Swiss roll is essentially uninterpretable — the local manifold structure is destroyed. After UMAP projection to 2D: UMAP preserves the local neighborhood structure of the manifold by constructing a graph of nearest neighbors in the original space and optimizing a 2D layout that preserves those neighborhoods. For the Swiss roll, UMAP "unrolls" the manifold and produces a 2D representation that looks like a flat rectangle or strip — a faithful representation of the 2D manifold embedded in 3D. The lesson: PCA finds linear projections; it cannot "see through" a coiled structure to the underlying manifold. Any method that relies on geodesic distance (distance along the manifold surface) rather than Euclidean distance will outperform PCA on manifold data. UMAP, Isomap, LLE, and locally linear embedding all take this approach.
      </Callout>

    </div>
  ),
};

export default pcaContent;
