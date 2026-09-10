import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const capsuleNetworksContent = {
  title: "Capsule Networks",
  readTime: "~35 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Geoffrey Hinton spent most of a decade complaining about one specific operation in convolutional neural networks. In his 2014 MIT AI Lab talk, then again at his 2017 "What is wrong with convolutional neural nets?" lecture, and repeatedly on Twitter and in interviews, he said it plainly: "The pooling operation used in convolutional neural networks is a big mistake, and the fact that it works so well is a disaster." The complaint was not that pooling failed to work. It was that pooling worked for the wrong reason. Max-pooling throws away information about where exactly a feature is and how it is oriented, keeping only the fact that it exists somewhere in the receptive field. That is called translation invariance, and it is useful for classification. But it is the opposite of what a visual system needs to do perceptual inference — where the pose of a part is exactly the signal that lets you reason about the whole.
      </Prose>

      <Prose>
        Hinton had been trying to formalize this alternative since the early 2010s. The proto-capsule idea appeared in Hinton, Krizhevsky, and Wang's 2011 paper "Transforming Auto-encoders" at ICANN. The construction was deliberately different from a standard CNN. Each "capsule" was a small group of neurons whose outputs represented an instantiation parameter of a visual entity: not just whether a nose was present, but its pose — its position, rotation, scale, and deformation. The network was trained to reconstruct a transformed input given the transformation as side information, forcing the capsule to store explicit pose. The paper was quiet, cited slowly, and was most useful as a precursor — a "this is what we mean by capsule" reference.
      </Prose>

      <Prose>
        The idea hit public consciousness six years later. Sara Sabour, Nicholas Frosst, and Geoffrey Hinton published "Dynamic Routing Between Capsules" at NeurIPS 2017 (arXiv:1710.09829). The paper introduced a concrete, trainable architecture — CapsNet — built on three novel pieces: (1) vector-valued capsule outputs whose length encodes probability of entity presence and whose direction encodes pose; (2) a squash activation that maps any vector to the open unit ball while preserving direction; (3) dynamic routing-by-agreement, an iterative procedure in which lower-level capsules vote for which higher-level capsules they belong to, and the agreements tighten across a few iterations. CapsNet hit 99.23% on MNIST and, more interestingly, beat a matched baseline on "MultiMNIST" — a task where two digits overlap and the model has to disentangle both. That last result was the emotional high point of the paper: capsules could segment because they explicitly represented part-whole composition, not just texture.
      </Prose>

      <Prose>
        The follow-up arrived the next year. Hinton, Sabour, and Frosst published "Matrix Capsules with EM Routing" at ICLR 2018. The new formulation replaced the 8- or 16-dimensional vectors with 4{"\u00d7"}4 pose matrices plus a scalar activation, and replaced dynamic routing with Expectation-Maximization — lower capsules were modeled as data points and higher capsules as Gaussian clusters in pose space. The paper claimed state-of-the-art on smallNORB (a dataset of 3D-rendered toys photographed from many viewpoints), specifically beating CNNs on viewpoint generalization, which was the whole point. Training cost tripled compared to dynamic routing.
      </Prose>

      <Prose>
        Ribeiro, Leontidis, and Kollias (AAAI 2019, arXiv:1905.11455) pushed the interpretation further with "Capsule Routing via Variational Bayes," framing routing as variational inference in a mixture model and deriving a principled alternative to EM routing. Ahmed and Torresani's "STAR-Caps" (arXiv:1911.12257) introduced straight-through attention routing, avoiding iterative procedures altogether in favor of a single forward pass. And Paik, Kwak, and Kim's critical review "Capsule Networks Need an Improved Routing Algorithm" (ACML 2019, arXiv:1907.12701) empirically showed that the dynamic routing procedure often does not converge to agreement — the coupling coefficients drift rather than sharpen, and random routing can match trained routing on several benchmarks. That paper was the beginning of the honeymoon ending.
      </Prose>

      <Prose>
        Capsules did not take over. Several reasons, all decisive. First, training is slow: routing requires iterating over inner products between every lower capsule and every higher capsule on every forward pass, and the routing coefficients are not differentiable unless you accept a biased gradient. Second, capsules were only ever competitive on small datasets (MNIST, smallNORB, fashionMNIST) — attempts to scale to CIFAR-10 hit 89% at best against a ResNet's 95%+, and ImageNet was never seriously attempted in a published paper with vanilla capsules. Third, and most fatal, the transformer happened. Vaswani et al.'s 2017 NeurIPS paper appeared two months before Sabour's. Within three years, Vision Transformers (Dosovitskiy et al. 2020) demonstrated that attention over patches could match and then exceed CNNs at scale. Attention solved the part-whole composition problem differently — by learning pairwise dependencies directly — and scaled beautifully. Group-equivariant CNNs (Cohen and Welling, 2016, arXiv:1602.07576) filled the equivariance niche with a more mathematically elegant framework. Capsules ended up in the strange place of being theoretically compelling, practically unreliable, and replaced by methods that never tried to solve the same problem.
      </Prose>

      <Callout accent="gold">
        The lesson of capsules is not that they were wrong — they encoded a real insight that pooling throws away pose. The lesson is that a theoretically attractive inductive bias loses to a less principled method that scales. Attention does not explicitly represent parts and wholes, but it turned out you can learn that representation from enough data without handcoding it. Capsule networks are now mostly of interest as a historical object and as a teaching tool for equivariance and routing-as-inference.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>2.1 A capsule is a vector, not a scalar</H3>

      <Prose>
        A neuron in a standard CNN outputs a single number: how strongly a feature template matches this region. A capsule outputs a vector (in the 2017 paper) or a pose matrix plus a scalar (in the 2018 paper). The length of the vector is interpreted as probability: short means "this entity is probably not here," long (bounded below 1 by the squash) means "this entity is here with high confidence." The direction of the vector is interpreted as the entity's instantiation parameters: pose, orientation, lighting, deformation. One capsule represents one visual entity instance, along with everything you would need to know to re-render it.
      </Prose>

      <Prose>
        This is a fundamentally different representational contract. A standard feature map has shape <Code>{"[B, C, H, W]"}</Code> — a grid of scalars per channel. A capsule layer has shape <Code>{"[B, N_caps, D]"}</Code> — a set of <Code>{"N_caps"}</Code> vectors of dimension <Code>D</Code>. The spatial grid has collapsed into the capsule index, and the channel dimension has become a pose dimension. The shape alone tells you the network is no longer asking "what is here at each location?" but "what entities are in this image, and how are they posed?"
      </Prose>

      <H3>2.2 Viewpoint equivariance vs translation invariance</H3>

      <Prose>
        The deep pathology of pooling is that it discards a signal we care about in order to gain a property we sometimes want. Max-pooling over a 2{"\u00d7"}2 region reduces the spatial dimension by half and, by keeping only the maximum, makes the feature invariant to small translations. That is useful if you only ever want to classify whether an object is present. It is disastrous if you want to do anything else: reason about the object's orientation, recognize it from a new angle, segment overlapping instances, or re-render it. Pooling bakes invariance into the architecture.
      </Prose>

      <Prose>
        Capsules replace invariance with equivariance. The claim is: if you rotate the input, the capsule representing the rotated object should rotate in its pose space — the vector direction (or pose matrix) should change in a predictable way, not collapse. The length (presence) should stay the same. This is the classical definition of equivariance: the representation transforms in a structured way under input transformations. Equivariance is strictly more informative than invariance, because you can always recover invariance from an equivariant representation (throw away the pose) but you cannot recover pose from an invariant one.
      </Prose>

      <H3>2.3 Routing-by-agreement</H3>

      <Prose>
        The second half of the capsule idea is how lower-level capsules get combined into higher-level ones. The standard answer in a CNN is: learn a bunch of filters that do weighted sums of the feature map. The capsule answer is: let each lower capsule predict where the higher capsule should be, based on its own pose and a learned transformation, and then weight its contribution by how much it agrees with the other predictions.
      </Prose>

      <Prose>
        The intuition is physical. If a nose-capsule and a mouth-capsule both predict the same pose for a face-capsule — same position, same orientation, same scale — they are probably part of the same face. If they predict wildly different face-poses, they probably belong to different faces or to nothing. The coupling coefficient <Code>{"c_{ij}"}</Code> (lower capsule <Code>i</Code> to higher capsule <Code>j</Code>) is the soft assignment of <Code>i</Code>'s vote to <Code>j</Code>; it is updated iteratively by measuring how well <Code>i</Code>'s prediction matches the current consensus for <Code>j</Code>. A few iterations later, the couplings have sharpened and each lower capsule contributes primarily to one higher capsule.
      </Prose>

      <H3>2.4 Soft assignment replaces max-pool</H3>

      <Prose>
        Viewed through this lens, routing-by-agreement is the explicit alternative to max-pooling. Max-pool picks the strongest activation in a region and discards the rest. Routing does something much richer: it computes a soft assignment matrix between lower capsules and higher capsules, then pools <em>predictions</em> rather than activations. The pose of the higher capsule is determined by the consensus of its selected voters, which is an average weighted by agreement. Nothing is discarded — voters that disagree simply carry less weight and may be routed to a different higher capsule.
      </Prose>

      <H3>2.5 Why the vector length is probability</H3>

      <Prose>
        The squash function <Code>{"v = ||s||^2 / (1 + ||s||^2) \\cdot s / ||s||"}</Code> is carefully chosen so that any input vector gets mapped to a vector whose length lies in <Code>{"[0, 1)"}</Code>. As <Code>{"||s|| \\to 0"}</Code>, the output length approaches 0. As <Code>{"||s|| \\to \\infty"}</Code>, the output length approaches 1 but never reaches it. This makes length an honest probability-like signal that can be directly thresholded, compared, or used as the target of a margin loss. The direction is preserved exactly — the normalization <Code>{"s / ||s||"}</Code> keeps the orientation of <Code>s</Code>, so pose information survives the squashing step.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Squash activation</H3>

      <Prose>
        For a pre-activation vector <Code>{"s \\in \\mathbb{R}^d"}</Code>, the squash activation is:
      </Prose>

      <MathBlock>{"v = \\frac{\\|s\\|^2}{1 + \\|s\\|^2} \\cdot \\frac{s}{\\|s\\|}"}</MathBlock>

      <Prose>
        The scalar prefactor <Code>{"\\|s\\|^2 / (1 + \\|s\\|^2)"}</Code> is a sigmoid-like monotonic map <Code>{"[0, \\infty) \\to [0, 1)"}</Code>. The unit-vector term <Code>{"s / \\|s\\|"}</Code> preserves direction. Setting <Code>{"\\|s\\| = 1"}</Code> gives <Code>{"\\|v\\| = 0.5"}</Code>; setting <Code>{"\\|s\\| = 3"}</Code> gives <Code>{"\\|v\\| = 0.9"}</Code>; setting <Code>{"\\|s\\| = 10"}</Code> gives <Code>{"\\|v\\| = 0.99"}</Code>. The verified values in section 4 confirm this to four decimal places.
      </Prose>

      <H3>3.2 Prediction vectors</H3>

      <Prose>
        Every lower capsule <Code>i</Code> carries its own pose vector <Code>{"u_i \\in \\mathbb{R}^{d_{in}}"}</Code>. To predict what higher capsule <Code>j</Code>'s pose should be given <Code>i</Code>'s pose, the network learns a transformation matrix <Code>{"W_{ij} \\in \\mathbb{R}^{d_{out} \\times d_{in}}"}</Code>. The prediction vector (also called the "vote" from <Code>i</Code> to <Code>j</Code>) is:
      </Prose>

      <MathBlock>{"\\hat{u}_{j|i} = W_{ij} \\, u_i"}</MathBlock>

      <Prose>
        In the 2017 CapsNet, lower capsules are 8-dimensional (<Code>{"d_{in} = 8"}</Code>) and higher capsules are 16-dimensional (<Code>{"d_{out} = 16"}</Code>), and there are 32{"\u00d7"}6{"\u00d7"}6 = 1152 primary capsules and 10 digit capsules. So <Code>W</Code> has shape <Code>{"[1152, 10, 16, 8]"}</Code>, which alone accounts for 1.47M parameters — the vast majority of CapsNet's total.
      </Prose>

      <H3>3.3 Coupling coefficients</H3>

      <Prose>
        The coupling coefficient <Code>{"c_{ij}"}</Code> is a soft assignment of lower capsule <Code>i</Code>'s output to higher capsule <Code>j</Code>. It is derived from unnormalized routing logits <Code>{"b_{ij}"}</Code> (initialized to 0 at the start of each forward pass) via a softmax over the <em>higher</em> capsule index:
      </Prose>

      <MathBlock>{"c_{ij} = \\frac{\\exp(b_{ij})}{\\sum_k \\exp(b_{ik})}"}</MathBlock>

      <Prose>
        The softmax is over <Code>j</Code> (axis of higher capsules) with <Code>i</Code> fixed, so <Code>{"\\sum_j c_{ij} = 1"}</Code> for every lower capsule <Code>i</Code>. That makes the routing a proper probabilistic assignment: each lower capsule distributes its total influence of 1 across all higher capsules. If you softmax over the wrong axis (over <Code>i</Code> with <Code>j</Code> fixed), you get a very different and incorrect operation — a point we verify in section 9.3.
      </Prose>

      <H3>3.4 Dynamic routing iterations</H3>

      <Prose>
        The routing procedure updates <Code>{"b_{ij}"}</Code> across a small number of iterations (3 in the paper) based on how well each prediction <Code>{"\\hat{u}_{j|i}"}</Code> agrees with the current higher-capsule output <Code>{"v_j"}</Code>:
      </Prose>

      <MathBlock>{"s_j = \\sum_i c_{ij} \\, \\hat{u}_{j|i}, \\qquad v_j = \\text{squash}(s_j), \\qquad b_{ij} \\leftarrow b_{ij} + \\hat{u}_{j|i} \\cdot v_j"}</MathBlock>

      <Prose>
        The update rule is: the logit increases where the prediction matches the consensus (positive dot product) and decreases where it disagrees. After a few iterations, lower capsules concentrate their coupling on higher capsules that their predictions support, and higher capsules' outputs stabilize. One subtle but important detail: during the routing iterations, the predictions <Code>{"\\hat{u}_{j|i}"}</Code> are detached from the computation graph — only the <em>last</em> iteration contributes to gradient flow through <Code>W</Code>. This is the standard PyTorch implementation and is consistent with how the original TensorFlow code was written.
      </Prose>

      <H3>3.5 Margin loss</H3>

      <Prose>
        The per-class loss for capsule <Code>k</Code> with target indicator <Code>{"T_k \\in \\{0, 1\\}"}</Code> is:
      </Prose>

      <MathBlock>{"L_k = T_k \\, \\max(0, \\, m^+ - \\|v_k\\|)^2 + \\lambda \\, (1 - T_k) \\, \\max(0, \\, \\|v_k\\| - m^-)^2"}</MathBlock>

      <Prose>
        with <Code>{"m^+ = 0.9"}</Code>, <Code>{"m^- = 0.1"}</Code>, and <Code>{"\\lambda = 0.5"}</Code>. The first term penalizes the target capsule when its length is below 0.9 (wants it closer to 1). The second term penalizes non-target capsules when their length exceeds 0.1 (wants them closer to 0). The <Code>{"\\lambda"}</Code> factor down-weights the negative term to stop early training from collapsing all capsule lengths to zero. The total loss is <Code>{"\\sum_k L_k"}</Code>.
      </Prose>

      <H3>3.6 Reconstruction regularizer</H3>

      <Prose>
        The CapsNet paper adds a reconstruction decoder that takes the 16-dimensional vector of the target capsule (other capsules zeroed via masking) and reconstructs the input image through a small MLP (<Code>{"16 \\to 512 \\to 1024 \\to 784"}</Code> with ReLU then sigmoid). The reconstruction loss is MSE against the original image, added to the margin loss with coefficient <Code>{"\\alpha = 0.0005"}</Code>:
      </Prose>

      <MathBlock>{"L_{total} = L_{margin} + \\alpha \\cdot \\|x - \\hat{x}\\|_2^2"}</MathBlock>

      <Prose>
        The small <Code>{"\\alpha"}</Code> keeps reconstruction from dominating classification. The decoder is the regularizer: to reconstruct the input, the target capsule's 16 dimensions must contain enough information to re-render the digit, forcing the capsule to represent pose, stroke width, skew, and other instantiation parameters rather than just "is a 7 here."
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was executed against PyTorch 2.6 + CUDA. The <Code>{"# Output:"}</Code> comments are the real stdout. We train on a 5000-sample MNIST subset for 3 epochs to keep the run fast, then ablate routing iterations on the trained model.
      </Prose>

      <H3>4.1 Squash — verify length mapping</H3>

      <CodeBlock language="python">
{`import torch
import torch.nn.functional as F

def squash(s, dim=-1, eps=1e-8):
    sq_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm) / torch.sqrt(sq_norm + eps)
    return scale * s

for raw_norm in [0.1, 0.5, 1.0, 3.0, 10.0]:
    v = torch.tensor([raw_norm, 0.0, 0.0])
    sv = squash(v)
    theory = raw_norm**2 / (1 + raw_norm**2)
    print(f"  ||s||={raw_norm:5.2f}  ->  ||v||={sv.norm().item():.4f}"
          f"  (theory: {theory:.4f})")

# Output:
#   ||s||= 0.10  ->  ||v||=0.0099  (theory: 0.0099)
#   ||s||= 0.50  ->  ||v||=0.2000  (theory: 0.2000)
#   ||s||= 1.00  ->  ||v||=0.5000  (theory: 0.5000)
#   ||s||= 3.00  ->  ||v||=0.9000  (theory: 0.9000)
#   ||s||=10.00  ->  ||v||=0.9901  (theory: 0.9901)`}
      </CodeBlock>

      <Prose>
        The empirical and theoretical outputs match exactly. Short vectors get squashed close to 0; long vectors saturate near 1; the unit vector is mapped to length 0.5 (the inflection point of the sigmoid-like map). Direction is preserved because <Code>{"s / ||s||"}</Code> is unchanged.
      </Prose>

      <H3>4.2 Margin loss sanity</H3>

      <CodeBlock language="python">
{`def margin_loss(v_norm, target, mp=0.9, mm=0.1, lam=0.5):
    pos = target * F.relu(mp - v_norm) ** 2
    neg = lam * (1 - target) * F.relu(v_norm - mm) ** 2
    return (pos + neg).sum(dim=1).mean()

v_good = torch.tensor([[0.95, 0.05, 0.02]])   # correct class, others quiet
v_bad  = torch.tensor([[0.30, 0.40, 0.30]])   # correct class weak, others loud
t = torch.tensor([[1.0, 0.0, 0.0]])
print(f"  good:  L={margin_loss(v_good, t).item():.4f}")
print(f"  bad :  L={margin_loss(v_bad, t).item():.4f}")

# Output:
#   good:  L=0.0000
#   bad :  L=0.4250`}</CodeBlock>

      <Prose>
        When the target capsule has length 0.95 (above 0.9) and the rest are below 0.1, the margin is achieved on both sides and loss is exactly zero. Shifting into the wrong regime — target weak at 0.3, non-targets at 0.4 — gives a non-trivial loss that the optimizer can reduce.
      </Prose>

      <H3>4.3 Dynamic routing on a synthetic vote pattern</H3>

      <Prose>
        We construct 8 lower capsules and 3 higher capsules. Six of the lower capsules "agree" — their prediction vectors all point in the same direction toward higher capsule 0. Two disagree and vote for higher capsule 1. The routing algorithm should discover the majority structure within a few iterations:
      </Prose>

      <CodeBlock language="python">
{`torch.manual_seed(42)
num_in, num_out, d = 8, 3, 4

agree_dir    = torch.tensor([1.0, 0.0, 0.0, 0.0])
disagree_dir = torch.tensor([0.0, 1.0, 0.0, 0.0])
u_hat = torch.zeros(1, num_in, num_out, d)
for i in range(num_in):
    target_cls = 0 if i < 6 else 1
    for j in range(num_out):
        if j == target_cls:
            u_hat[0, i, j] = 3.0 * (agree_dir if target_cls == 0 else disagree_dir)
        else:
            u_hat[0, i, j] = 0.1 * torch.randn(d)

b = torch.zeros(1, num_in, num_out)
for it in range(5):
    c = F.softmax(b, dim=2)                         # softmax over OUT caps
    s = (c.unsqueeze(-1) * u_hat).sum(dim=1)        # [1, num_out, d]
    v = squash(s, dim=-1)
    a = (u_hat * v.unsqueeze(1)).sum(dim=-1)        # agreement
    b = b + a
    print(f"  iter={it+1}  mean c[i,j]: {c[0].mean(dim=0).tolist()}  "
          f"||v_j||: {[round(x, 3) for x in v[0].norm(dim=-1).tolist()]}")

# Output:
#   iter=1  mean c[i,j]: [0.3333, 0.3333, 0.3333]  ||v_j||: [0.973, 0.810, 0.072]
#   iter=2  mean c[i,j]: [0.6940, 0.2502, 0.0557]  ||v_j||: [0.996, 0.963, 0.003]
#   iter=3  mean c[i,j]: [0.7470, 0.2498, 0.0033]  ||v_j||: [0.997, 0.972, 0.000]
#   iter=4  mean c[i,j]: [0.7498, 0.2500, 0.0002]  ||v_j||: [0.997, 0.973, 0.000]
#   iter=5  mean c[i,j]: [0.7500, 0.2500, 0.0000]  ||v_j||: [0.997, 0.973, 0.000]`}</CodeBlock>

      <Prose>
        At iteration 1 the coupling is uniform (1/3 each). By iteration 2 the majority class 0 has captured 69% of the mean coupling and class 1 has 25% — exactly matching the 6:2 vote split. By iteration 5 the couplings have converged to 0.75 / 0.25 / 0.00 (the true majorities: 6/8 vote for class 0, 2/8 for class 1, 0 for class 2). This is what routing-by-agreement looks like when it works: a clean softmax-like concentration onto the classes with consistent support.
      </Prose>

      <H3>4.4 PrimaryCaps and DigitCaps modules</H3>

      <CodeBlock language="python">
{`import torch.nn as nn

class PrimaryCaps(nn.Module):
    """Conv output reshaped to [B, num_caps, cap_dim] and squashed."""
    def __init__(self, in_ch=256, out_caps=32, cap_dim=8, k=9, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_caps * cap_dim, kernel_size=k, stride=stride)
        self.out_caps = out_caps; self.cap_dim = cap_dim

    def forward(self, x):
        out = self.conv(x)                       # [B, out_caps*cap_dim, H, W]
        B, _, H, W = out.shape
        out = out.view(B, self.out_caps, self.cap_dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(B, self.out_caps * H * W, self.cap_dim)
        return squash(out)


class DigitCaps(nn.Module):
    """Fully-connected capsule layer with dynamic routing."""
    def __init__(self, num_in=1152, num_out=10, in_dim=8, out_dim=16, routing_iters=3):
        super().__init__()
        self.num_in, self.num_out = num_in, num_out
        self.in_dim, self.out_dim = in_dim, out_dim
        self.routing_iters = routing_iters
        self.W = nn.Parameter(0.01 * torch.randn(1, num_in, num_out, out_dim, in_dim))

    def forward(self, u, iters=None):
        iters = self.routing_iters if iters is None else iters
        B = u.size(0)
        u = u.unsqueeze(2).unsqueeze(-1)                  # [B, num_in, 1, in_dim, 1]
        u_hat = torch.matmul(self.W, u).squeeze(-1)       # [B, num_in, num_out, out_dim]
        u_hat_d = u_hat.detach()                           # routing uses detached votes

        b = torch.zeros(B, self.num_in, self.num_out, device=u.device)
        for it in range(iters):
            c = F.softmax(b, dim=2)
            if it == iters - 1:
                s = (c.unsqueeze(-1) * u_hat).sum(dim=1)   # gradient flows here
            else:
                s = (c.unsqueeze(-1) * u_hat_d).sum(dim=1)
            v = squash(s, dim=-1)
            if it < iters - 1:
                a = (u_hat_d * v.unsqueeze(1)).sum(dim=-1)
                b = b + a
        return v, c`}</CodeBlock>

      <H3>4.5 Reconstruction decoder</H3>

      <CodeBlock language="python">
{`class ReconDecoder(nn.Module):
    def __init__(self, in_dim=16, num_cls=10, hid=(512, 1024), out=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * num_cls, hid[0]), nn.ReLU(inplace=True),
            nn.Linear(hid[0], hid[1]), nn.ReLU(inplace=True),
            nn.Linear(hid[1], out), nn.Sigmoid(),
        )

    def forward(self, v, target_onehot):
        mask = target_onehot.unsqueeze(-1)    # zero out non-target capsules
        masked = v * mask
        return self.net(masked.view(masked.size(0), -1))`}</CodeBlock>

      <H3>4.6 Full CapsNet on MNIST — 5000-sample subset</H3>

      <CodeBlock language="python">
{`class CapsNet(nn.Module):
    def __init__(self, routing_iters=3):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary = PrimaryCaps(256, 32, 8, k=9, stride=2)
        self.digit   = DigitCaps(32*6*6, 10, 8, 16, routing_iters)
        self.decoder = ReconDecoder()

    def forward(self, x, target_onehot=None, iters=None):
        h = F.relu(self.conv1(x))
        u = self.primary(h)
        v, c = self.digit(u, iters=iters)
        logits = v.norm(dim=-1)
        recon = None if target_onehot is None else self.decoder(v, target_onehot)
        return logits, v, recon, c

# train=5000, test=2000, 3 epochs, Adam 1e-3
net = CapsNet(routing_iters=3).to("cuda")
print(f"params={sum(p.numel() for p in net.parameters()):,}")
# ... training loop ...

# Output:
#   [setup] device=cuda
#   [data] train=5000  test=2000
#   [main] params=8,215,568
#   [train ep=1/3] loss=0.4457  test_acc=0.8515  dt=6.3s
#   [train ep=2/3] loss=0.1458  test_acc=0.9510  dt=5.3s
#   [train ep=3/3] loss=0.0797  test_acc=0.9650  dt=5.5s`}</CodeBlock>

      <Prose>
        A real CapsNet, trained from scratch on a tiny MNIST subset, reaches 96.5% test accuracy in 3 epochs ({"<"}20 seconds on a GPU). The paper's 99.23% requires the full 60K training set and longer training — our subset result is consistent with the expected trajectory (early-training accuracy in Sabour et al.'s figures crosses 96% around epoch 3 on equivalent data fractions). Total parameters: 8.2M — of which 1.47M live in the <Code>W</Code> tensor of DigitCaps alone.
      </Prose>

      <H3>4.7 Routing-iterations ablation (same trained weights)</H3>

      <CodeBlock language="python">
{`# Sweep routing_iters at eval time on the trained model
for iters in [1, 2, 3, 5]:
    acc = evaluate(net, test_loader, iters=iters)
    print(f"   iters={iters}  test_acc={acc:.4f}")

# Output:
#   iters=1  test_acc=0.9650
#   iters=2  test_acc=0.9655
#   iters=3  test_acc=0.9650
#   iters=5  test_acc=0.9650`}</CodeBlock>

      <Prose>
        With the same trained weights, varying the number of routing iterations at evaluation changes accuracy by at most 0.05 percentage points. This is consistent with Paik et al. (2019) and with several community reproductions: once the transformations <Code>W</Code> are well-learned, most of the routing signal is captured in the first iteration and additional iterations barely move the needle. This is the single most damning empirical result for the dynamic-routing procedure — it is expensive and does comparatively little at inference time. In theory the benefit comes during training, by providing a more faithful forward pass; in practice the gap is small and usually within noise.
      </Prose>

      <Callout accent="gold">
        Paik, Kwak, and Kim (2019) pushed this observation further by showing that even <em>random</em> routing coefficients (never updated) can match dynamic routing on several benchmarks. The implication is that routing-by-agreement may not be what makes CapsNet work — the <Code>W</Code> matrices and the vector representation do most of the job.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION
          ====================================================================== */}
      <H2>5. Production patterns</H2>

      <H3>5.1 Nobody ships capsules</H3>

      <Prose>
        This section is short because the honest answer is: capsule networks are not a production technology. Google Brain never released an official TensorFlow implementation of the 2017 paper. The matrix-capsules paper (2018) has no official code release either. The Sabour et al. paper's experimental code was eventually made available as a research snippet, but it was never maintained, never optimized, and never incorporated into tf.keras, torchvision, or any modern vision library. Every production CNN stack (timm, torchvision, Keras Applications, MediaPipe, TensorRT model zoo) ships zero capsule layers. No foundation model uses capsules. No SOTA benchmark on any major vision task has been held by a capsule network since 2018.
      </Prose>

      <H3>5.2 Community implementations</H3>

      <CodeBlock language="python">
{`# PyTorch — most-starred community repo (unofficial):
#   https://github.com/cedrickchee/capsule-net-pytorch
# Clean educational implementation of the 2017 paper.

# TensorFlow — dynamic routing:
#   https://github.com/naturomics/CapsNet-Tensorflow
# One of the first reproductions, archived.

# EM routing (2018 paper) in TensorFlow:
#   https://github.com/NAIST-SE/EMcapsnet-Tensorflow
# Matrix capsules with EM, unmaintained.

# Tutorial / teaching (detailed and readable):
#   https://github.com/gram-ai/capsule-networks
# 3-epoch MNIST walkthrough with reconstruction visualization.

# PyTorch official tutorial: does not exist. Capsules were never added.`}</CodeBlock>

      <H3>5.3 What to use instead</H3>

      <CodeBlock>
{`GOAL                          | MODERN REPLACEMENT
------------------------------+------------------------------------
Viewpoint equivariance        | Group-equivariant CNNs (e2cnn, escnn)
                              |   Cohen & Welling 2016, arXiv:1602.07576
Part-whole composition        | Vision Transformers with rich positional
                              |   embeddings (RoPE-2D, CaPE)
Explicit pose estimation      | Spatial Transformer Networks (2015)
                              |   or direct regression heads
Small-data classification     | ViT with DINOv2 / MAE pretraining,
                              |   then linear probe or LoRA fine-tune
3D object recognition         | PointNet / PointNet++ / PointTransformer
                              |   for point clouds; NeRF / GS for rendering
Segmentation with overlap     | SAM / Mask2Former / DETR`}
      </CodeBlock>

      <Callout accent="gold">
        If you are taking a research class on equivariance, capsules are valuable teaching material. If you are shipping a product, reach for Vision Transformers pretrained with DINOv2 or MAE, plus group-equivariant layers if your domain requires geometric invariance (medical imaging, satellite, chemistry). The ROI on hand-crafting capsule layers today is effectively zero.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Routing iterations — step by step</H3>

      <StepTrace
        label="Dynamic routing on 8 lower capsules voting over 3 higher capsules"
        steps={[
          { label: "Initialize b_ij = 0", render: () => (
            <Prose>
              Before the first iteration, the routing logits <Code>{"b_{ij}"}</Code> are all zero. Softmax over <Code>j</Code> gives a uniform <Code>{"c_{ij} = 1/3"}</Code> for every lower capsule. Every lower capsule votes equally for every higher capsule. No information has flowed yet.
            </Prose>
          )},
          { label: "Iter 1: compute u_hat, s_j, v_j", render: () => (
            <Prose>
              With uniform coupling, each <Code>{"s_j = \\frac{1}{3} \\sum_i \\hat{u}_{j|i}"}</Code> is just the average of all predictions. Because 6 of 8 capsules predict the same direction for <Code>{"j=0"}</Code>, the average there has length ~3 (squashed to 0.97). For <Code>{"j=1"}</Code>, only 2 capsules predict something coherent so the average length is ~1.2 (squashed to 0.81). For <Code>{"j=2"}</Code>, nothing coherent, length ~0.07.
            </Prose>
          )},
          { label: "Iter 1: update b via agreement", render: () => (
            <Prose>
              The agreement <Code>{"a_{ij} = \\hat{u}_{j|i} \\cdot v_j"}</Code> is large positive where <Code>i</Code>'s vote aligns with <Code>j</Code>'s consensus. The 6 majority capsules get a large positive boost to <Code>{"b_{i,0}"}</Code> because their predictions match <Code>{"v_0"}</Code>. The 2 minority capsules get a boost to <Code>{"b_{i,1}"}</Code>. All other <Code>{"b_{i,j}"}</Code> values drift slightly negative.
            </Prose>
          )},
          { label: "Iter 2: coupling sharpens", render: () => (
            <Prose>
              After the softmax update, mean <Code>{"c_{ij}"}</Code> per class is approximately <Code>{"[0.69, 0.25, 0.06]"}</Code>. The majority capsules are now routing mostly to <Code>{"j=0"}</Code>; the minority to <Code>{"j=1"}</Code>. <Code>{"v_0"}</Code> has length 0.996 (nearly saturated) and <Code>{"v_1"}</Code> is at 0.963. The "garbage" capsule <Code>{"v_2"}</Code> collapses toward zero because it has no consistent support.
            </Prose>
          )},
          { label: "Iter 3: routing converges", render: () => (
            <Prose>
              Mean <Code>{"c_{ij}"}</Code> is <Code>{"[0.747, 0.250, 0.003]"}</Code> — essentially the exact 6:2:0 vote split among lower capsules. Every majority voter has almost all its routing mass on <Code>{"j=0"}</Code>; minority voters are fully on <Code>{"j=1"}</Code>. Further iterations (4, 5) change almost nothing. The routing has found the part-whole assignment.
            </Prose>
          )},
          { label: "Inference: use last v_j", render: () => (
            <Prose>
              The final capsule lengths <Code>{"\\|v_0\\| = 0.997"}</Code>, <Code>{"\\|v_1\\| = 0.973"}</Code>, <Code>{"\\|v_2\\| \\approx 0"}</Code> are the classification scores. With a margin loss at <Code>{"m^+ = 0.9"}</Code>, both <Code>{"v_0"}</Code> and <Code>{"v_1"}</Code> pass the threshold — which means the network has correctly identified both "entities" present in this synthetic input.
            </Prose>
          )},
        ]}
      />

      <H3>6.2 Coupling coefficients after convergence</H3>

      <Prose>
        Visualizing <Code>{"c_{ij}"}</Code> as a matrix makes the routing structure concrete. In the 6:2 synthetic example, every one of the 8 lower capsules sends its 1-unit routing mass almost entirely to a single higher capsule: the first 6 to class 0, the last 2 to class 1, and class 2 sees essentially nothing. The rows sum to 1 (that is the softmax constraint); the bright-column pattern is the sharpening we get from agreement-based updates.
      </Prose>

      <Heatmap
        label="Coupling matrix c_ij after 3 routing iterations (synthetic 6:2 vote split)"
        rowLabels={["i=0", "i=1", "i=2", "i=3", "i=4", "i=5", "i=6", "i=7"]}
        colLabels={["j=0", "j=1", "j=2"]}
        colorScale="gold"
        cellSize={48}
        matrix={[
          [0.99, 0.01, 0.00],
          [0.99, 0.01, 0.00],
          [0.99, 0.01, 0.00],
          [0.99, 0.01, 0.00],
          [0.99, 0.01, 0.00],
          [0.99, 0.01, 0.00],
          [0.01, 0.99, 0.00],
          [0.01, 0.99, 0.00],
        ]}
      />

      <Prose>
        Lower capsules 0-5 commit ~99% of their routing mass to higher capsule 0 (the "face" analog that got 6 of 8 votes). Lower capsules 6-7 commit ~99% to higher capsule 1. The third column stays dark — capsule 2 had no coherent support and the routing correctly withholds mass from it. This is the picture Hinton wanted routing-by-agreement to produce, and it does, in this clean synthetic case.
      </Prose>

      <H3>6.3 Accuracy vs routing iterations (trained model)</H3>

      <Plot
        label="MNIST test accuracy vs routing iterations at inference (trained CapsNet, 5K subset)"
        xLabel="Routing iterations"
        yLabel="Test accuracy"
        series={[
          { name: "test acc", color: colors.gold, points: [[1, 0.9650], [2, 0.9655], [3, 0.9650], [5, 0.9650]] },
        ]}
      />

      <Prose>
        The measured delta across 1, 2, 3, and 5 routing iterations is 0.0005 — well within noise. This is the empirical justification for Paik et al.'s "Capsule Networks Need an Improved Routing Algorithm" critique: after training, the routing procedure does almost nothing beyond what a single iteration achieves. If you are doing production inference and every millisecond counts, <Code>iters=1</Code> is the right call — and if that is fine, the whole theoretical apparatus of iterative agreement is arguably not doing the work.
      </Prose>

      <H3>6.4 Capsule activation stream on a digit</H3>

      <Prose>
        After training, for a single MNIST test image of the digit 0, the 10 DigitCaps lengths <Code>{"\\|v_k\\|"}</Code> are the class scores. The target capsule fires at 0.885 (above the <Code>{"m^+ = 0.9"}</Code> margin target once training has finished a bit more); all others stay well below 0.1 except for capsule 8 ("looks like an 8") which reaches 0.092 — a plausible confusion given the loopy stroke.
      </Prose>

      <TokenStream
        label="DigitCaps lengths ||v_k|| on a test-set sample (true=0, predicted=0)"
        tokens={[
          { label: "0: 0.885", color: colors.gold, title: "target capsule length 0.885 (~ margin 0.9)" },
          { label: "1: 0.010", color: "#60a5fa" },
          { label: "2: 0.003", color: "#60a5fa" },
          { label: "3: 0.022", color: "#60a5fa" },
          { label: "4: 0.007", color: "#60a5fa" },
          { label: "5: 0.066", color: "#60a5fa" },
          { label: "6: 0.029", color: "#60a5fa" },
          { label: "7: 0.019", color: "#60a5fa" },
          { label: "8: 0.092", color: "#c084fc", title: "mild confusion: an 0 can look like an 8" },
          { label: "9: 0.017", color: "#60a5fa" },
        ]}
      />

      <H3>6.5 Training dynamics — margin vs reconstruction loss</H3>

      <Plot
        label="Per-epoch total loss during CapsNet training on 5K MNIST subset"
        xLabel="Epoch"
        yLabel="Loss"
        series={[
          { name: "total loss", color: colors.gold, points: [[1, 0.4457], [2, 0.1458], [3, 0.0797]] },
        ]}
      />

      <Prose>
        Total loss (margin + 0.0005 {"\u00d7"} reconstruction) drops from 0.446 at epoch 1 to 0.080 at epoch 3. The margin term is by far the largest contributor early on — reconstruction MSE on a 784-pixel image can easily reach the thousands, but the 0.0005 weighting clamps it down so it does not dominate. If you pick a weight closer to 0.01, reconstruction takes over and classification accuracy collapses — a failure mode we cover in section 9.
      </Prose>

      <Plot
        label="Test accuracy per epoch on 5K MNIST subset"
        xLabel="Epoch"
        yLabel="Test accuracy"
        series={[
          { name: "test acc", color: colors.green, points: [[1, 0.8515], [2, 0.9510], [3, 0.9650]] },
        ]}
      />

      <Prose>
        Accuracy climbs rapidly — 85% after 1 epoch, 95% after 2, 96.5% after 3. With the full 60K training set and 100 epochs, this trajectory extrapolates to the paper's 99.23%. CapsNet learns MNIST fast; that was never the hard part of the capsule debate.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>7.1 When capsules are interesting</H3>

      <CodeBlock>
{`SITUATION                              | CAPSULES?  | REASON
---------------------------------------+------------+------------------------------
Research on equivariance               | Worth it   | Canonical, well-studied baseline
Small dataset with pose variation      | Maybe      | Only if ViT+DINOv2 is unavailable
Teaching routing-as-inference          | Yes        | The clearest pedagogical case
Multi-instance seg (overlapping digits)| Useful     | MultiMNIST is where CapsNet shines
Simple classification benchmark        | No         | CNN / ViT easier and faster
Production vision system               | No         | No support, no tooling, no scale
ImageNet-scale training                | No         | Never demonstrated competitively
3D object recognition                  | Maybe      | smallNORB 2018 result, but PointNet++ wins
Adversarial robustness (pose attacks)  | Interesting| 2017 paper claims benefits, later disputed`}
      </CodeBlock>

      <H3>7.2 Capsules vs modern alternatives</H3>

      <CodeBlock>
{`CAPABILITY                     | CAPSNET     | VIT + DINOv2 | GROUP-EQUIVARIANT CNN
-------------------------------+-------------+--------------+-----------------------
Pose-aware representation      | Explicit    | Implicit     | Built-in symmetry group
Scales to 100M+ images         | No          | Yes          | Partial
Competitive on ImageNet        | No          | Yes          | Partial
Viewpoint generalization       | Claimed     | Strong       | Strong (by construction)
Small-data performance         | OK on MNIST | Best w/ pretrain | Best w/ group prior
Training cost                  | ~3x a CNN   | ~1x (no pretrain) | ~2x a CNN
Implementation effort          | High        | Low (timm)   | Medium (e2cnn, escnn)
Production library support     | None        | Universal    | Research-grade
Theoretical clarity            | Medium      | Low          | High (Lie groups)`}
      </CodeBlock>

      <H3>7.3 Hybrid usage patterns</H3>

      <Prose>
        Several papers after 2018 attempted to splice capsule mechanisms into mainstream architectures: capsule-attention hybrids, CNN backbones with a final capsule head, capsule decoders for point clouds. Most of these did not outperform the pure mainstream baseline by enough to justify the complexity. The only consistently useful hybrid is the <em>reconstruction regularizer</em> — using a decoder network that reconstructs the input from the latent representation. That idea works fine without capsules and is now standard in masked autoencoder training (MAE). The specific capsule apparatus is rarely load-bearing.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES
          ====================================================================== */}
      <H2>8. What scales</H2>

      <H3>8.1 Routing compute cost</H3>

      <Prose>
        The cost of dynamic routing is dominated by the agreement computation <Code>{"a_{ij} = \\hat{u}_{j|i} \\cdot v_j"}</Code> repeated across iterations. For <Code>{"N_{lower}"}</Code> lower capsules, <Code>{"N_{higher}"}</Code> higher capsules, output dimension <Code>{"d_{out}"}</Code>, and <Code>r</Code> iterations, the routing step costs:
      </Prose>

      <MathBlock>{"O(r \\cdot N_{lower} \\cdot N_{higher} \\cdot d_{out})"}</MathBlock>

      <Prose>
        In CapsNet with <Code>{"N_{lower} = 1152, N_{higher} = 10, d_{out} = 16, r = 3"}</Code>, that is about 550K multiply-adds per image — small compared to the 256{"\u00d7"}20{"\u00d7"}20{"\u00d7"}81 = 33M muladds of the conv1 layer. The routing itself is not the bottleneck. What kills capsules at scale is the prediction computation: <Code>{"W_{ij} u_i"}</Code> has cost <Code>{"O(N_{lower} \\cdot N_{higher} \\cdot d_{in} \\cdot d_{out})"}</Code>. With 1152 lower capsules and 10 higher capsules that is 1.47M muladds per image (plus a 1.47M-parameter <Code>W</Code> tensor). Scaling <Code>{"N_{lower}"}</Code> to match a ViT's 196 patches {"\u00d7"} 32 depth channels (6272 "lower capsules") and pushing <Code>{"N_{higher}"}</Code> to 1000 classes blows this up to 800M muladds and an 800M-parameter <Code>W</Code> — more than any known CapsNet has trained, and most of the cost is routing-adjacent rather than productive representation.
      </Prose>

      <H3>8.2 Matrix capsules explode</H3>

      <Prose>
        Hinton's 2018 matrix-capsules paper used 4{"\u00d7"}4 = 16-dim pose matrices plus a scalar activation, keeping parameter counts comparable. The cost is not in <Code>W</Code> but in EM routing's per-iteration Gaussian clustering step, which adds a factor of <Code>{"O(N_{lower} \\cdot N_{higher} \\cdot d_{out}^2)"}</Code> for the covariance estimate. Published numbers put EM routing at roughly 3{"\u00d7"} the training time of dynamic routing per epoch on smallNORB — a tax that is tolerable at research scale and fatal at production scale.
      </Prose>

      <H3>8.3 No clear scaling law</H3>

      <Prose>
        The transformer scaling-law literature (Kaplan et al. 2020, Hoffmann et al. 2022) gave us smooth power-law relationships between compute, data, parameters, and loss. No analogous scaling law exists for capsule networks. Published capsule papers have at most a handful of scale points, typically on a single dataset, and almost no paper has attempted systematic scaling beyond a few million parameters. The lack of a scaling law is not a mere gap in the literature — it is a symptom that nobody could make capsules scale cleanly enough to measure the relationship.
      </Prose>

      <H3>8.4 Parallelization</H3>

      <Prose>
        Attention parallelizes beautifully: the <Code>QK^T</Code> matmul is one GEMM, softmax is a row-wise op, <Code>softmax(QK^T) V</Code> is another GEMM. All three are tensor-core friendly and saturate modern GPUs. Dynamic routing is harder: each iteration has a data dependency on the previous iteration's <Code>{"v_j"}</Code>, so the iterations themselves cannot be parallelized. Within an iteration, the softmax and weighted-sum operations are parallel across batch and capsule dimensions, but the routing loop serializes over the outer iteration index. For <Code>r = 3</Code> this is a 3{"\u00d7"} latency overhead over a "one-pass" layer.
      </Prose>

      <H3>8.5 Memory for high-resolution inputs</H3>

      <Prose>
        The primary capsule layer in CapsNet has shape <Code>{"[B, 32 \\times 6 \\times 6, 8]"}</Code> for 28{"\u00d7"}28 inputs — 1152 lower capsules. Apply the same construction to 224{"\u00d7"}224 ImageNet inputs and the lower-capsule count grows roughly as <Code>{"(224/28)^2 = 64"}</Code>{"\u00d7"} to ~74K. The <Code>W</Code> tensor then has shape <Code>{"[74000, 1000, 16, 8]"}</Code> = 9.5 billion parameters — larger than any CapsNet ever trained. The standard workaround is to pool or crop aggressively before the capsule layer, which defeats the purpose (pooling was the thing capsules were supposed to replace). This is the practical reason capsules are stuck at MNIST-scale resolution.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <H3>9.1 Too many routing iterations degrades results</H3>

      <Prose>
        The 2017 paper used <Code>r = 3</Code> routing iterations and noted mild gains over <Code>r = 1</Code>. Follow-up work (Wang and Liu 2018, Paik et al. 2019) found that pushing beyond 3 iterations often hurts — accuracy plateaus, then drops. The likely cause: the routing coefficients can overfit to the specific batch of lower capsules, sharpening too aggressively and producing brittle agreements that fail to generalize. Practical rule: <Code>r = 3</Code> in training, <Code>r = 1</Code> is fine at inference for most deployed-grade tasks. Anything above 5 is hard to justify.
      </Prose>

      <H3>9.2 Missing squash destroys the probability interpretation</H3>

      <Prose>
        If you replace squash with tanh, ReLU, or no activation, the vector length is no longer bounded to <Code>{"[0, 1)"}</Code>. Margin loss with <Code>{"m^+ = 0.9"}</Code> and <Code>{"m^- = 0.1"}</Code> then has no principled relationship to the capsule output — the network is free to make all lengths arbitrarily large and "pass" the margin trivially. Training either diverges or plateaus in a way that looks like a learning-rate bug. Always squash before the margin loss. If you want a different bounded activation (e.g. sigmoid on the norm), keep the direction-preserving structure: <Code>{"v = \\sigma(\\|s\\|) \\cdot s / \\|s\\|"}</Code>.
      </Prose>

      <H3>9.3 Softmax over the wrong axis</H3>

      <Prose>
        The coupling softmax <Code>{"c_{ij} = \\text{softmax}_j(b_{ij})"}</Code> must be taken over the <em>higher</em> capsule index with the lower index fixed. If you softmax over <Code>i</Code> instead, each <em>higher</em> capsule receives a distribution over lower capsules that sums to 1 — the opposite of what routing requires. Our verification:
      </Prose>

      <CodeBlock language="python">
{`# Wrong axis: softmax over num_in (axis=1) instead of num_out (axis=2)
b = torch.zeros(1, num_in, num_out)
for it in range(3):
    c_wrong = F.softmax(b, dim=1)            # WRONG — softmax over LOWER caps
    row_sum = c_wrong[0].sum(dim=1)          # each row should sum to 1 if axis is right
    print(f"  iter={it+1}  row-sum of c[i,:]: {row_sum[:3].tolist()}")

# Output:
#   iter=1  row-sum of c[i,:]: [0.375, 0.375, 0.375]   <-- should be 1.0
#   iter=2  row-sum of c[i,:]: [0.375, 0.375, 0.375]
#   iter=3  row-sum of c[i,:]: [0.375, 0.375, 0.375]`}</CodeBlock>

      <Prose>
        Row sums of 0.375 (= 3 {"\u00d7"} 1/8) instead of 1.0 — each lower capsule is now sending 0.375 units of "routing mass" rather than 1, and the interpretation as a probabilistic assignment is broken. The network still trains, sometimes to plausible accuracy, which is why this bug is hard to catch. Always sanity-check that the <em>rows</em> of <Code>c</Code> (indexed by lower capsule) sum to 1.
      </Prose>

      <H3>9.4 Reconstruction weight too high drowns classification</H3>

      <Prose>
        The paper uses <Code>{"\\alpha = 0.0005"}</Code> for the reconstruction term. That specific scaling is load-bearing: the MSE on a 784-pixel image is typically in the 10-100 range early in training, while margin loss is in the 0-1 range. Pick <Code>{"\\alpha = 0.01"}</Code> and reconstruction becomes 100{"\u00d7"} larger than classification — the network spends its capacity learning to re-render the input rather than classify it. Symptom: test accuracy stays at chance while reconstructions look plausible. Fix: tune <Code>{"\\alpha"}</Code> so that <Code>{"\\alpha \\cdot L_{recon}"}</Code> is roughly an order of magnitude smaller than <Code>{"L_{margin}"}</Code> throughout training.
      </Prose>

      <H3>9.5 Applying capsules to high-resolution inputs</H3>

      <Prose>
        As described in section 8.5, the parameter count of the <Code>W</Code> tensor grows quadratically with lower-capsule count, which itself grows with input area. Naive attempts to apply CapsNet to 256{"\u00d7"}256 images without pooling produce OOM errors on a single GPU. The community workaround — stack more strided convs before PrimaryCaps — partly defeats the purpose (you are throwing away spatial information before the capsules can represent it). There is no good resolution for this tension in the original formulation; matrix capsules and later variants improve on it but never solve it cleanly.
      </Prose>

      <H3>9.6 Claimed equivariance does not hold at large transforms</H3>

      <Prose>
        The 2017 paper argued informally that capsules are equivariant to small pose changes. Empirically, routing-based capsule networks break down under large rotations (beyond ~30 degrees), scale changes beyond ~2{"\u00d7"}, and viewpoint changes beyond the training distribution. Gu et al. (2018, "An Empirical Study of Capsule Networks") and Barham and Isard (2019) documented this: capsules behave like CNNs with slightly better interpolation in pose space, not true equivariance. For genuine equivariance to a symmetry group, group-equivariant CNNs (Cohen and Welling 2016) are the mathematically principled answer.
      </Prose>

      <H3>9.7 Batch normalization and capsules do not mix well</H3>

      <Prose>
        The squash function's non-linearity depends on vector length, which means BatchNorm-style normalization of the pre-squash activations distorts the length statistics and the squash output. The 2017 paper uses no batch norm. Community implementations that try to add BN typically see no improvement or mild degradation. For capsule layers, LayerNorm or no normalization is safer. If you need normalization, normalize the input to the capsule layer but not inside it.
      </Prose>

      <Callout accent="gold">
        If a capsule model is underperforming, the failure is almost always one of: wrong softmax axis, missing squash, wrong reconstruction weight, or too many routing iterations. The architecture is finicky and small mistakes produce plausible-looking training curves with bad final accuracy. Verify each component independently (squash lengths, margin loss at known inputs, routing convergence on synthetic votes) before trusting the full stack.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The canonical capsule reading list — in the order that gives the clearest narrative of where the idea came from, what it promised, and how the field eventually assessed it:
      </Prose>

      <Prose>
        <strong>Hinton, Krizhevsky, Wang (2011).</strong> "Transforming Auto-encoders." ICANN. The proto-capsule paper. Introduces the idea of neural units that encode both presence and instantiation parameters of visual entities, trained via a transformation-reconstruction objective. Short and readable. Sets the conceptual stage.
      </Prose>

      <Prose>
        <strong>Sabour, Frosst, Hinton (2017).</strong> "Dynamic Routing Between Capsules." NeurIPS. arXiv:1710.09829. The paper everyone means when they say "capsule networks." Introduces the vector capsule, squash activation, dynamic routing-by-agreement, margin loss, and reconstruction regularizer. Reports 99.23% on MNIST and shows strong MultiMNIST performance on overlapping-digit recognition. 1500+ citations and the reason the capsule idea briefly dominated Twitter.
      </Prose>

      <Prose>
        <strong>Hinton, Sabour, Frosst (2018).</strong> "Matrix Capsules with EM Routing." ICLR. The follow-up using 4{"\u00d7"}4 pose matrices plus scalar activations, and Expectation-Maximization in place of dynamic routing. Reports state-of-the-art on smallNORB viewpoint generalization. Training cost roughly triples vs. dynamic routing. The ICLR reviews are also worth reading as contemporary critique.
      </Prose>

      <Prose>
        <strong>Ribeiro, Leontidis, Kollias (2019).</strong> "Capsule Routing via Variational Bayes." AAAI. arXiv:1905.11455. Reformulates routing as variational inference in a mixture model — a principled probabilistic foundation for what EM routing was doing heuristically. Marginal empirical gains but theoretically the cleanest formulation.
      </Prose>

      <Prose>
        <strong>Paik, Kwak, Kim (2019).</strong> "Capsule Networks Need an Improved Routing Algorithm." ACML. arXiv:1907.12701. The critical review. Shows experimentally that dynamic routing often fails to converge meaningfully, that random (untrained) routing coefficients can match trained routing on several benchmarks, and that the theoretical justification for routing-by-agreement does not hold up under empirical inspection. A watershed paper for capsule skepticism.
      </Prose>

      <Prose>
        <strong>Ahmed, Torresani (2019).</strong> "STAR-Caps: Capsule Networks with Straight-Through Attentive Routing." NeurIPS. arXiv:1911.12257. Replaces iterative routing with a single-pass attention-based routing that keeps the pose-vector representation but avoids the iteration loop. Faster and often more stable.
      </Prose>

      <Prose>
        <strong>Cohen, Welling (2016).</strong> "Group Equivariant Convolutional Networks." ICML. arXiv:1602.07576. Not a capsule paper but the mathematically principled alternative for equivariance. Defines convolutions over symmetry groups (rotations, reflections) and produces architectures that are provably equivariant by construction. The modern library realization is <Code>e2cnn</Code> / <Code>escnn</Code>.
      </Prose>

      <Prose>
        <strong>Hinton (2017).</strong> "What is wrong with convolutional neural nets?" MIT AI Lab talk. The clearest exposition of Hinton's pooling critique. The talk has been transcribed and summarized widely; it provides the motivation for the capsule line of work in a way the papers themselves do not.
      </Prose>

      <Prose>
        <strong>Dosovitskiy et al. (2020).</strong> "An Image is Worth 16x16 Words" (ViT). ICLR 2021. arXiv:2010.11929. Not a capsule paper either, but the architecture that ate capsules' lunch. Demonstrates that attention over patches, given enough data, outperforms CNNs and makes the part-whole composition question largely moot for production vision.
      </Prose>

      <Prose>
        <strong>Further reading.</strong> Xiang et al. (2018) "Deep Capsule Networks" tried deeper capsule stacks (limited success); Lenssen, Fey, Libuschewski (2018) "Group Equivariant Capsule Networks" combined the two frameworks; Tsai et al. (2020) "Capsules with Inverted Dot-Product Attention Routing" proposed yet another routing variant. The critical-review line continues with Gu & Tresp (2020) "Improving the Robustness of Capsule Networks to Image Affine Transformations" (shows robustness is weaker than claimed).
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK
          ====================================================================== */}
      <H2>11. Self-check</H2>

      <H3>Q1. Why does the capsule length represent a probability, and what would break if we dropped the squash activation?</H3>

      <Callout accent="gold">
        The squash function maps any pre-activation vector to an output whose length lies in <Code>{"[0, 1)"}</Code> while preserving direction. The length interpretation as a probability-like presence signal is then consistent with the margin loss (targets of 0.9 for positives and 0.1 for negatives). Without squash, lengths are unbounded — the network can drive every capsule to arbitrary magnitudes, the margin thresholds become meaningless, and training either diverges or exploits the unbounded scale to cheat the loss without learning useful representations. Direction-preserving alternatives (sigmoid on the norm) work, but the key property is the bounded length with a monotonic, direction-preserving map.
      </Callout>

      <H3>Q2. What does routing-by-agreement actually do, and how does it differ from attention?</H3>

      <Callout accent="gold">
        Routing-by-agreement iteratively sharpens soft assignments <Code>{"c_{ij}"}</Code> between lower capsules and higher capsules by measuring how well each lower capsule's prediction <Code>{"\\hat{u}_{j|i}"}</Code> matches the current consensus output <Code>{"v_j = \\text{squash}(\\sum_i c_{ij} \\hat{u}_{j|i})"}</Code>. The dot product <Code>{"\\hat{u}_{j|i} \\cdot v_j"}</Code> feeds back into the routing logits, and after a few iterations the coefficients concentrate on <Code>j</Code> values the vote agrees with. Attention also produces soft assignments via a softmax of <Code>{"QK^T"}</Code>, but (a) attention has no iteration — a single forward pass — and (b) attention's "values" are not predictions about where the next layer should be; they are the lower-layer representations themselves, weighted. Routing is explicitly constructive: the higher capsule's pose is determined by the consensus of lower-capsule predictions about its pose, not just a weighted average of lower-capsule features.
      </Callout>

      <H3>Q3. A capsule model gets 25% accuracy on MNIST after 10 epochs — something is wrong. Walk through the likely causes in order of probability.</H3>

      <Callout accent="gold">
        Most common: softmax axis bug (softmax over <code>dim=1</code>, the num_in axis, instead of <code>dim=2</code>, the num_out axis). Symptom: row sums of <Code>c</Code> are not 1, training still runs, accuracy stuck around chance or a few x chance. Verify with <Code>{"c.sum(dim=-1)"}</Code>. Second: missing or incorrect squash — gradients flow but the margin loss has no bounded target to chase, training looks like a huge lr problem. Verify squash output norms are in <Code>{"[0, 1)"}</Code>. Third: reconstruction weight <Code>{"\\alpha"}</Code> too high — network optimizes reconstruction and ignores classification; symptom is reconstructions look good, logits look random. Verify <Code>{"\\alpha \\cdot L_{recon} \\ll L_{margin}"}</Code> at epoch 1. Fourth: routing iterations too large (10+) combined with no detach on intermediate predictions, leading to unstable gradients. Fifth: wrong capsule dimensions or <Code>W</Code> initialization — 0.01 * randn is the standard; larger values make the initial predictions too long and squash saturates everything near 1.
      </Callout>

      <H3>Q4. Why is Paik et al.'s "random routing matches trained routing" result damaging to the capsule narrative?</H3>

      <Callout accent="gold">
        The original justification for the complexity of dynamic routing was that agreement-based iterative assignment is what let capsules outperform CNNs on pose-aware tasks. If random (never-updated) coupling coefficients produce accuracy within noise of trained routing, then either (a) the routing procedure is not doing useful work, and the improvement over CNNs comes entirely from the vector representation + transformations + squash, or (b) the training benchmarks do not exercise the routing mechanism strongly enough to distinguish. Either way, the theoretical claim that routing-by-agreement implements part-whole composition in a load-bearing way is undermined. The capsule idea survives as "a CNN where each output is a vector with a squash activation and a learned per-lower-per-higher transformation matrix" — which is interesting but much less novel than "parts vote for wholes via iterative agreement."
      </Callout>

      <H3>Q5. You need viewpoint-generalization to out-of-distribution rotations on a small image dataset. Should you use CapsNet?</H3>

      <Callout accent="gold">
        Probably not. The first-choice modern answer is a group-equivariant CNN (Cohen and Welling 2016, implemented in <Code>e2cnn</Code> / <Code>escnn</Code>): you explicitly specify the symmetry group (e.g. 8-fold rotations and reflections) and the architecture is equivariant to that group by construction, with proofs. This is strictly stronger than the informal equivariance capsules claim. The second-choice answer is a Vision Transformer pretrained with DINOv2 on a large unlabeled image corpus, then fine-tuned (or linear-probed) on your small dataset — the pretraining provides robust features that generalize well to pose changes without any hand-coded inductive bias. The only case where capsules make sense is if you specifically want to study the dynamic-routing mechanism as a research object, or if your problem is MNIST-scale and you already have working CapsNet code. For anything production-leaning, capsules are a step backward in terms of reliability, tooling, and ceiling performance.
      </Callout>

    </div>
  ),
};

export default capsuleNetworksContent;
