import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, PatchGrid, Heatmap, StepTrace, Plot } from "../../components/viz";
import { colors } from "../../styles";

const multimodalTokenization = {
  title: "Multimodal Tokenization (Visual, Audio, Video)",
  readTime: "42 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Text tokenization is a one-dimensional problem with a universal substrate. Every piece of human writing, in every script, every emoji, every proper noun, every typo, ultimately bottoms out at a sequence of Unicode codepoints, and every Unicode codepoint bottoms out at a sequence of bytes. BPE, WordPiece, SentencePiece, and Unigram argue over how to <em>group</em> those bytes into units a model can learn from, but they never have to argue about what the input is. The input is always a one-dimensional string with well-defined boundaries between symbols. Images, audio, and video have none of those properties. A photograph is a 2D grid of three-channel float values with no natural boundaries. A second of speech is a sequence of twenty-four thousand floating-point samples that carries meaning only in aggregate, distributed across hundreds of frames in ways that no single frame captures. A video adds a third dimension and enormous temporal redundancy. There is no universal substrate, no natural unit, no equivalent of "byte" that generalizes across modalities. Multimodal tokenization is the set of algorithms that invent one.
      </Prose>

      <Prose>
        The problem is not abstract. The modern transformer was designed to consume a sequence of integer token IDs. Its attention mechanism expects <Code>(B, T, D)</Code> inputs, where <Code>T</Code> is a sequence length and each position is the embedding of a discrete symbol. Everything downstream — the cross-entropy loss, autoregressive sampling, beam search, speculative decoding, KV cache reuse, speculative prefill — assumes that assumption. If you want the same transformer that writes text to also describe a photograph, transcribe a recording, or generate a video clip, you have to convert continuous signals into the discrete-token representation the architecture already speaks. There are two families of ways to do this, and the choice between them is the first architectural decision in any multimodal system.
      </Prose>

      <Prose>
        The first family keeps the signal continuous. An image encoder — typically a Vision Transformer or a CLIP-style encoder — maps pixels to a sequence of embedding vectors, and those vectors are projected into the language model's hidden dimension and interleaved with text token embeddings at the input layer. Flamingo, BLIP, LLaVA, and most production multimodal LLMs use this pattern. The appeal is modularity: the image encoder can be trained separately, swapped in without retraining the language model, and reused across tasks. The cost is asymmetry. The model can <em>read</em> images but cannot <em>write</em> them, because there is no discrete vocabulary to sample from on the output side. To generate an image, you bolt on a separate decoder — a diffusion model, a separate VQ-based generator — and the end-to-end unified objective is lost.
      </Prose>

      <Prose>
        The second family discretizes. Each modality gets its own tokenizer that converts the raw signal into a sequence of integer codebook indices, drawn from a fixed-size vocabulary. Once that step is done, images, audio, and video all look identical to the transformer — just more integers. Cross-entropy applies unchanged. Sampling applies unchanged. The transformer can <em>generate</em> images, audio, or interleaved multimodal sequences using the same machinery it uses to generate the next word. This is the approach of DALL·E 1, Parti, MaskGIT, AudioLM, MusicLM, VideoPoet, and — in the purest form — Chameleon, which trains a single transformer over a vocabulary of 65k text tokens and 8k image codebook entries sharing one embedding table. The price is that you must train a good discretizer for each modality, and the discretizer's failures become the model's ceiling.
      </Prose>

      <Prose>
        The historical arc runs from continuous to discrete and back. CLIP in 2021 (Radford et al., arXiv:2103.00020) established that 400 million image-text pairs could train a continuous image encoder that aligned with a text encoder in a shared embedding space — the foundation for every continuous-embedding multimodal model since. ViT in 2020 (Dosovitskiy et al., arXiv:2010.11929) introduced the idea of treating an image as a sequence of patches, borrowing the transformer's sequence-processing machinery without yet making the patches discrete. VQ-VAE in 2017 (van den Oord, Vinyals, Kavukcuoglu, arXiv:1711.00937) introduced the learned discrete codebook — the mechanism that turns a continuous encoder output into an integer index a transformer can consume. VQ-GAN in 2020 (Esser et al., arXiv:2012.09841) added a perceptual-plus-adversarial objective that sharpened reconstructions enough to make high-resolution image generation work. SoundStream in 2021 (Zeghidour et al., arXiv:2107.03312) and EnCodec in 2022 (Défossez et al., arXiv:2210.13438) brought residual vector quantization from the signal-processing literature into deep learning, turning any audio stream into a tokenizable sequence at controllable bitrates. Chameleon in May 2024 (Meta, arXiv:2405.09818) and the Gemini 1.5 report in 2024 (Google, arXiv:2403.05530) demonstrate the payoff: unified mixed-modal models that treat every input modality as tokens in the same sequence. This topic is about the algorithms that make that unification possible.
      </Prose>

      <Prose>
        One thing to notice about this arc: each paper solves a problem the previous paper could not. ViT made images consumable by transformers but kept them continuous, which left generation unsolved. VQ-VAE made them discrete but reconstructed blurry pixels under a pure MSE loss. VQ-GAN fixed reconstruction quality but kept training dynamics fragile. RVQ made discrete audio tractable at useful bitrates. Chameleon and Gemini demonstrated that once every modality is tokens, a single decoder-only transformer can generate all of them. The field did not find a master algorithm; it found a sequence of local fixes that compose into a workable pipeline. Each fix is worth understanding in isolation because real systems combine them in domain-specific ways — you will see VQ-GAN for images, EnCodec for audio, space-time VQ-VAE for video, all routed into one Chameleon-style transformer, in the same production deployment.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Four mental models cover every multimodal tokenizer you are likely to encounter. Internalize them before looking at the math, and the math will read like obvious extensions rather than new inventions.
      </Prose>

      <H3>Patches: a grid is a sequence</H3>

      <Prose>
        The ViT insight is to stop asking "what is a pixel-level unit" and start asking "what is a patch-level unit." Divide the image into a regular grid of non-overlapping squares — typically 16×16 pixels each. Flatten each square into a vector. Project that vector into the model's hidden dimension with a single linear layer. Tag each vector with a learned positional embedding. What comes out is a sequence of vectors the transformer can consume with no architectural change. A 224×224 image with 16×16 patches becomes a sequence of 196 tokens, each a 768-dimensional float vector. The attention mechanism treats these patches exactly the way it treats subword embeddings: query, key, value, softmax, weighted sum. There is no "vision module." There is a tokenizer (the patch projection) and then the ordinary transformer. This is the basic template, and every other visual tokenizer is a refinement of it.
      </Prose>

      <H3>Codebook quantization: snap the continuous to the nearest discrete</H3>

      <Prose>
        Patches solve the sequence problem but not the <em>discrete</em> problem. The patch projection produces continuous float vectors, which is fine for a discriminative model that will pass them through more float-valued layers, but useless for a generative model that needs a discrete vocabulary to sample from. VQ-VAE solves this with a learned codebook: a table of <Code>K</Code> vectors, each the same dimension as an encoder output. At each spatial position, the encoder produces a continuous feature; the tokenizer looks up the nearest codebook entry by L2 distance and emits its integer index. That index is the token. The decoder later looks up the same index in the same codebook to get back a vector, from which it reconstructs pixels. The codebook is learned jointly with the encoder and decoder, so the entries drift to cover the space of actually-observed encoder outputs. A 32×32 feature grid over a 256×256 image becomes a sequence of 1,024 integer codebook indices drawn from a vocabulary of, typically, 8,192.
      </Prose>

      <H3>Residual vector quantization: stack codebooks to hit a bitrate</H3>

      <Prose>
        A single codebook has a bitrate ceiling: with <Code>K</Code> = 8,192, each token carries <Code>log₂(8192)</Code> = 13 bits of information. To get more fidelity you either grow the codebook (which causes training instability and dead codes) or stack multiple codebooks in sequence. Residual vector quantization (RVQ), which SoundStream brought to audio, takes the elegant second option. The first codebook quantizes the feature coarsely; the <em>residual</em> error between the feature and its nearest code is then quantized by a second codebook, capturing finer detail; the residual of that is quantized by a third, and so on. The final reconstruction is the sum of all the codebook outputs. With eight RVQ levels of codebook size 1024, you get the equivalent of a vocabulary of <Code>1024⁸ ≈ 1.2 × 10²⁴</Code>, but each individual codebook is still small and trainable. The price is sequence-length multiplication: each original frame now produces eight tokens, one per level.
      </Prose>

      <H3>Interleaving: once everything is tokens, sequences just concatenate</H3>

      <Prose>
        The final intuition is deceptively simple. Once images, audio, and text are all sequences of integers drawn from a unified vocabulary, the input to a multimodal model is just a longer sequence of integers. Special boundary tokens — <Code>{"<img>"}</Code>, <Code>{"</img>"}</Code>, <Code>{"<audio>"}</Code>, <Code>{"</audio>"}</Code> — mark modality transitions. The transformer learns cross-modal dependencies through the same attention mechanism it uses for in-text coreference: an image token early in the sequence can directly influence a text token two hundred positions later, without any architectural routing module. Chameleon's 65k-token vocabulary and 8k image codebook share one embedding table, and the model learns, from scratch, when to generate text versus image tokens based on context alone. This is what the community means by "native multimodality": the model's architecture does not distinguish modalities at all. The tokenizer does.
      </Prose>

      <Prose>
        Two corollaries of this design are worth stating plainly because they are often missed. First, every architectural improvement to transformers automatically benefits multimodal models too — flash attention, RoPE, MQA, speculative decoding, whatever comes next, all apply without modification. There is no "multimodal flash attention" to invent because there is no separate multimodal architecture. Second, the serving infrastructure built for text LMs — continuous batching, KV cache sharing, prefix caching, paged attention — applies identically to multimodal workloads. An image token in the KV cache is the same shape as a text token in the KV cache, and the same attention kernel reads both. This is the systems-level payoff of the "everything is tokens" design, and it is the reason Chameleon and Gemini serving look, at the infrastructure layer, almost identical to serving a pure text model.
      </Prose>

      <Callout accent="gold">
        Patches turn a 2D grid into a sequence. Codebooks turn continuous features into discrete tokens. RVQ stacks codebooks to hit a bitrate without blowing up codebook size. Interleaving glues it all into one sequence. Those four ideas compose into every modern multimodal tokenizer.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>ViT patch projection</H3>

      <Prose>
        Given an image <Code>x ∈ ℝ^(H×W×C)</Code> and a patch size <Code>P</Code>, the patch projection is a deterministic reshape followed by a learned linear map. Let <Code>N = (H/P)·(W/P)</Code> be the number of patches. The reshape folds each <Code>P×P×C</Code> patch into a single vector of length <Code>P²·C</Code>, producing a matrix of shape <Code>(N, P²·C)</Code>. A learnable weight matrix <Code>E ∈ ℝ^(P²C × d)</Code> then projects each row into the model's hidden dimension <Code>d</Code>.
      </Prose>

      <MathBlock>
        {"z_0 = [x_{\\text{cls}};\\, x_1 E;\\, x_2 E;\\, \\ldots;\\, x_N E] + E_{\\text{pos}}, \\qquad x_i \\in \\mathbb{R}^{P^2 C}"}
      </MathBlock>

      <Prose>
        The optional <Code>x_cls</Code> is a learned class token prepended to the sequence; <Code>E_pos</Code> is a learned or sinusoidal positional embedding added element-wise. From there the sequence <Code>z_0</Code> flows through standard transformer blocks with no further vision-specific operations. The only vision-specific line in the whole ViT forward pass is the reshape + projection above. Everything else is a text transformer.
      </Prose>

      <H3>VQ-VAE: quantization and the three-term loss</H3>

      <Prose>
        Let <Code>E</Code> be the encoder and <Code>D</Code> the decoder. Given an input <Code>x</Code>, the encoder produces a spatial grid of continuous feature vectors <Code>z_e(x)</Code>, with each position a <Code>d</Code>-dimensional vector. A codebook <Code>e ∈ ℝ^(K×d)</Code> holds <Code>K</Code> learnable embeddings. Quantization replaces each feature <Code>z_e(x)_i</Code> with the nearest codebook entry by L2 distance:
      </Prose>

      <MathBlock>
        {"k_i = \\underset{k \\in \\{1, \\ldots, K\\}}{\\operatorname{argmin}} \\; \\|z_e(x)_i - e_k\\|_2, \\qquad z_q(x)_i = e_{k_i}"}
      </MathBlock>

      <Prose>
        The decoder takes <Code>z_q(x)</Code> and reconstructs <Code>x̂ = D(z_q(x))</Code>. The training loss has three terms: reconstruction, codebook, and commitment. The codebook and commitment terms are where the stop-gradient operator <Code>sg[·]</Code> does its work. Gradients flow through <Code>sg[y]</Code> in the forward pass (it is the identity) but are blocked in the backward pass (it is treated as a constant).
      </Prose>

      <MathBlock>
        {"\\mathcal{L} = \\underbrace{\\|x - D(z_q(x))\\|_2^2}_{\\text{reconstruction}} + \\underbrace{\\|\\text{sg}[z_e(x)] - e\\|_2^2}_{\\text{codebook}} + \\beta \\underbrace{\\|z_e(x) - \\text{sg}[e]\\|_2^2}_{\\text{commitment}}"}
      </MathBlock>

      <Prose>
        Read each term for what it does. The reconstruction term trains the encoder and decoder to round-trip the input through the quantization bottleneck. The codebook term moves the codebook entries toward the encoder outputs — the stop-gradient on <Code>z_e(x)</Code> means only <Code>e</Code> gets the gradient, so the codebook entries follow the encoder but the encoder does not chase them. The commitment term pulls the encoder outputs toward the codebook entries — the stop-gradient on <Code>e</Code> means only the encoder gets the gradient, so the encoder commits to a nearest code but the codebook does not drift to meet it. The coefficient <Code>β</Code>, typically 0.25, controls how hard the encoder is pulled toward its assigned code. Without the two stop-gradients, the encoder and codebook chase each other in circles and training collapses. With them, the two stabilize into a mutual agreement where encoder outputs cluster near codebook entries and codebook entries tile the encoder's output manifold.
      </Prose>

      <H3>Straight-through estimator</H3>

      <Prose>
        The argmin in the forward pass is non-differentiable — there is no well-defined gradient of "the index of the nearest codebook entry" with respect to the encoder output. The straight-through estimator (STE) sidesteps this with a trick: during the forward pass, use <Code>z_q(x)</Code>; during the backward pass, pretend the argmin was the identity and let gradients from the decoder flow straight into the encoder as if quantization never happened. In code this is one line:
      </Prose>

      <MathBlock>
        {"z_q^{\\text{forward}}(x) = z_e(x) + \\text{sg}[z_q(x) - z_e(x)]"}
      </MathBlock>

      <Prose>
        The value at runtime is <Code>z_q(x)</Code> (the two <Code>z_e(x)</Code> terms cancel). The gradient at runtime is that of <Code>z_e(x)</Code> (the stop-gradient kills the contribution from the difference). STE is an approximation — it assumes the encoder's gradient direction is still informative after quantization — but it works in practice and every VQ-based tokenizer uses some version of it.
      </Prose>

      <H3>Residual vector quantization</H3>

      <Prose>
        RVQ iterates the VQ step over a stack of <Code>L</Code> codebooks <Code>{"\\{e^{(1)}, \\ldots, e^{(L)}\\}"}</Code>. Initialize the residual as the encoder feature itself. At each level, quantize the current residual to the nearest code in that level's codebook, add that code to the running reconstruction, and subtract it from the residual for the next level.
      </Prose>

      <MathBlock>
        {"r_0 = z_e(x), \\qquad k_i^{(l)} = \\operatorname{argmin}_k \\|r_{l-1} - e^{(l)}_k\\|_2, \\qquad r_l = r_{l-1} - e^{(l)}_{k^{(l)}}"}
      </MathBlock>

      <MathBlock>
        {"\\hat{z}_q(x) = \\sum_{l=1}^{L} e^{(l)}_{k^{(l)}}"}
      </MathBlock>

      <Prose>
        Each layer quantizes what the previous layer missed. The bit budget is additive: <Code>L</Code> layers of codebook size <Code>K</Code> give <Code>L·log₂(K)</Code> bits per frame, and the effective vocabulary is <Code>K^L</Code>. For a real numerical picture of how fast residual error shrinks with added layers, see section 4c.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Four implementations cover the substrate. The ViT patchifier is pure NumPy and fully runnable. The VQ-VAE quantization step is NumPy for the forward pass and the loss terms; the encoder/decoder networks are shown as PyTorch-style reference pseudocode because training a real conv stack in pure NumPy in the margin of an article is a losing proposition. The RVQ simulation is runnable end-to-end in NumPy, and the numbers embedded below were produced by running the exact code shown, so they are verifiable. The interleaved-sequence constructor is trivial but important because it is where most bugs in multimodal models live.
      </Prose>

      <H3>4a. Patchify from scratch</H3>

      <Prose>
        The patchifier is four lines of reshape-and-transpose. The only subtlety is the axis ordering: a naive reshape produces patches whose pixels are in the wrong order, which is why the intermediate transpose is necessary. The easiest way to verify correctness is to round-trip a random image and check bit-exact equality.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def patchify(image, patch_size=16):
    """image: (H, W, C). Returns (N_patches, patch_size*patch_size*C)."""
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    gh, gw = H // patch_size, W // patch_size
    patches = image.reshape(gh, patch_size, gw, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)          # (gh, gw, ps, ps, c)
    return patches.reshape(gh * gw, patch_size * patch_size * C)

def unpatchify(patches, H, W, patch_size=16):
    C = patches.shape[1] // (patch_size * patch_size)
    gh, gw = H // patch_size, W // patch_size
    p = patches.reshape(gh, gw, patch_size, patch_size, C)
    p = p.transpose(0, 2, 1, 3, 4)
    return p.reshape(H, W, C)

np.random.seed(0)
img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
patches = patchify(img, 16)

# Actual output (verified by running this code):
# input image shape : (224, 224, 3)
# patches shape     : (196, 768)
# n_patches         : 196
# patch_dim         : 768
# round-trip exact  : True`}
      </CodeBlock>

      <PatchGrid
        label="ViT-style 14x14 patches on a 224-pixel image"
        src="https://picsum.photos/seed/multimodal/224/224"
        patches={14}
        size={280}
      />

      <Prose>
        The flattened <Code>(196, 768)</Code> tensor is the ViT input before any learned projection. The next step in a real ViT adds a linear layer <Code>E ∈ ℝ^(768×d)</Code> that projects each patch into the model's hidden dimension — <Code>d = 768</Code> for ViT-B, <Code>d = 1024</Code> for ViT-L. For a VQ-VAE, the patchify step is replaced by a conv encoder that produces a smaller spatial grid of higher-dimensional features, but the logical structure is the same: a 2D grid of continuous vectors, ready to be either consumed directly (ViT) or quantized against a codebook (VQ-VAE).
      </Prose>

      <H3>4b. VQ-VAE quantization and the straight-through estimator</H3>

      <Prose>
        The core VQ operation — nearest-code lookup — is a two-line NumPy function. The straight-through trick is one more line. The loss terms are two more lines. The encoder and decoder networks are standard 2D convolutional stacks that would be six lines each in PyTorch; they are shown here as reference pseudocode because our focus is on the quantization, not the conv arithmetic.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def nearest_code(feature, codebook):
    """feature: (N, D). codebook: (K, D). Returns indices of shape (N,)."""
    # Pairwise squared L2: (N, 1, D) - (1, K, D) -> (N, K, D) -> sum over D.
    dists = ((feature[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    return dists.argmin(1)

def vq_step(z_e, codebook, beta=0.25):
    """Forward pass returning z_q, loss terms, and codebook indices."""
    idx = nearest_code(z_e, codebook)
    e   = codebook[idx]                              # nearest entries
    codebook_loss = ((z_e - e) ** 2).mean()          # ||sg[z_e] - e||^2
    commit_loss   = ((z_e - e) ** 2).mean()          # ||z_e - sg[e]||^2
    # Straight-through: forward is e, backward passes z_e's gradient to encoder.
    # In numpy we cannot express the gradient rewrite; in PyTorch it is:
    #     z_q = z_e + (e - z_e).detach()
    z_q = e
    return z_q, codebook_loss, beta * commit_loss, idx

# Tiny demo: 16 feature vectors of dim 8, codebook of size 4.
np.random.seed(0)
z_e = np.random.randn(16, 8).astype(np.float32)
codebook = np.random.randn(4, 8).astype(np.float32)

z_q, cb_l, cm_l, idx = vq_step(z_e, codebook)

# Actual output (verified):
# encoder output    : (16, 8)
# codebook          : (4, 8)
# nearest indices   : [0, 0, 1, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 0, 0]
# code usage        : [10, 3, 0, 3]   <- code 2 is dead
# codebook_loss     : 0.6062
# commit_loss (beta=0.25): 0.1516`}
      </CodeBlock>

      <Prose>
        The code-usage histogram shows the pathology that plagues every real VQ-VAE: some codes are dead. Out of four entries, entry 2 was the nearest neighbor of zero encoder outputs. In a real 8k-entry codebook, without counter-measures, 40-60% of entries routinely end up dead because the codebook update rule only moves entries that are chosen, so an entry that starts far from the encoder's output manifold stays far forever. Production training runs counter this with EMA codebook updates (from the VQ-VAE-2 paper), random restarts that reinitialize dead codes from currently-active encoder outputs, or the "codebook reset" trick where any entry with low running usage is teleported next to a randomly chosen encoder output. The visualization below is a synthetic codebook-usage heatmap showing the typical pattern — a handful of very-frequently-used entries, a long tail of moderately-used ones, and many entries that never fire.
      </Prose>

      <Heatmap
        label="codebook usage counts (synthetic 64-entry codebook, ~40% dead)"
        matrix={[
          [0, 11, 12, 2, 192, 0, 0, 6],
          [0, 0, 552, 1, 29, 11, 286, 4],
          [16, 0, 0, 0, 1, 302, 204, 101],
          [191, 14, 26, 1, 0, 86, 0, 0],
          [0, 0, 0, 0, 1, 51, 915, 140],
          [31, 0, 0, 923, 0, 0, 31, 2155],
          [0, 0, 53, 26, 0, 15, 22, 0],
          [173, 200, 57, 199, 0, 402, 0, 0],
        ]}
        cellSize={34}
        colorScale="gold"
      />

      <Prose>
        Reference pseudocode for the full VQ-VAE forward pass. PyTorch is assumed but not required to run — the important lines are the ones involving <Code>detach()</Code> (stop-gradient) and the sum of three losses.
      </Prose>

      <CodeBlock language="python">
{`# Reference PyTorch pseudocode (not runnable in this environment's numpy-only kernel).
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, d=64, K=512, beta=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, d, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(d, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.codebook = nn.Embedding(K, d)
        self.codebook.weight.data.uniform_(-1/K, 1/K)
        self.beta = beta

    def quantize(self, z_e):
        # z_e: (B, d, H, W) -> flatten spatial dims.
        B, d, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)    # (B*H*W, d)
        dists = (flat.pow(2).sum(1, keepdim=True)
                 - 2 * flat @ self.codebook.weight.t()
                 + self.codebook.weight.pow(2).sum(1))
        idx = dists.argmin(1)                             # (B*H*W,)
        e   = self.codebook(idx).view(B, H, W, d).permute(0, 3, 1, 2)
        # Straight-through: forward is e, backward sends gradient to z_e.
        z_q = z_e + (e - z_e).detach()
        codebook_loss = F.mse_loss(e, z_e.detach())       # sg[z_e]
        commit_loss   = F.mse_loss(z_e, e.detach())       # sg[e]
        return z_q, codebook_loss + self.beta * commit_loss, idx.view(B, H, W)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, idx = self.quantize(z_e)
        x_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(x_hat, x)
        return x_hat, recon_loss + vq_loss, idx`}
      </CodeBlock>

      <Prose>
        Three properties to note about the <Code>detach()</Code> pattern. First, <Code>z_q = z_e + (e - z_e).detach()</Code> gives forward value <Code>e</Code> but backward gradient <Code>∂z_q/∂z_e = I</Code>, which is the straight-through approximation. Second, <Code>codebook_loss = F.mse_loss(e, z_e.detach())</Code> has gradient only through <Code>e</Code> — this updates the codebook toward the encoder. Third, <Code>commit_loss = F.mse_loss(z_e, e.detach())</Code> has gradient only through <Code>z_e</Code> — this updates the encoder toward the codebook. The two loss terms never fight because each has its own grad-consumer; without the <Code>detach</Code> calls, both terms would flow gradients in both directions and training would diverge.
      </Prose>

      <Prose>
        A subtlety worth highlighting: <em>commitment loss is not symmetric with codebook loss under a scalar rescaling</em>. If you drop the codebook loss and scale commitment up by an equivalent amount, the encoder still gets pulled toward the codebook but the codebook stops tracking the encoder, and quality collapses. The two terms do different things. Codebook loss says "codebook entries, go to where the encoder is." Commitment loss says "encoder, go to where the codebook is." You need both, at their proper coefficients, or one side of the pair drifts. In production VQ-VAE implementations, the codebook-side update is often replaced with an EMA on assigned features (no gradient, just running average), which decouples codebook dynamics from the optimizer entirely and fixes a class of instabilities that plague gradient-based codebook updates. The commitment loss stays in the objective either way, because the encoder still needs its pull toward the codebook.
      </Prose>

      <H3>4c. Residual vector quantization from scratch</H3>

      <Prose>
        RVQ is where the numbers get satisfying. The code below simulates 256 audio-frame-like feature vectors of dimension 32, and stacks eight codebooks of size 64 each — a configuration in the ballpark of what SoundStream uses for speech at moderate bitrates. Each codebook is trained with a few iterations of Lloyd's algorithm (k-means) on the residual left by the previous layers, which is the standard initialization for RVQ codebooks. The reconstruction MSE falls by roughly 50% per layer — the classic RVQ "each layer halves the error" rule of thumb.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(2024)

def nearest_code(feature, codebook):
    dists = ((feature[:, None, :] - codebook[None, :, :]) ** 2).sum(-1)
    return dists.argmin(1)

def rvq_encode(feature, codebooks):
    """Iterative residual quantization: quantize, subtract, repeat."""
    residual = feature.copy()
    total = np.zeros_like(feature)
    indices, errs = [], []
    for cb in codebooks:
        idx = nearest_code(residual, cb)
        quant = cb[idx]
        total = total + quant
        residual = residual - quant
        indices.append(idx)
        errs.append(((feature - total) ** 2).mean())
    return indices, total, errs

# 256 frames of dim 32, 8 levels of codebook size 64.
N, D, K, L = 256, 32, 64, 8
feature = np.random.randn(N, D).astype(np.float32)

# Train codebooks layer-by-layer with 20 k-means iterations on residuals.
codebooks, residual = [], feature.copy()
for l in range(L):
    cb = residual[np.random.choice(N, K, replace=False)].copy()
    for _ in range(20):
        a = nearest_code(residual, cb)
        for k in range(K):
            m = residual[a == k]
            if len(m) > 0: cb[k] = m.mean(0)
    codebooks.append(cb)
    residual = residual - cb[nearest_code(residual, cb)]

indices, recon, errs = rvq_encode(feature, codebooks)

# Actual output (verified by running this code):
# layer        mse      rel   bits/frame
# init      1.0215    1.000            0
# 1         0.5729    0.561            6
# 2         0.3161    0.309           12
# 3         0.1725    0.169           18
# 4         0.0925    0.091           24
# 5         0.0496    0.049           30
# 6         0.0261    0.026           36
# 7         0.0141    0.014           42
# 8         0.0070    0.007           48`}
      </CodeBlock>

      <Prose>
        The pattern in the output is the whole point. With one codebook the mean squared error is 56% of the unquantized baseline. With two it is 31%. By level eight it is 0.7% — two orders of magnitude reduction, achieved with eight codebooks of 64 entries each, at a total cost of 48 bits per frame. The equivalent single-codebook approach would require <Code>2⁴⁸ ≈ 2.8 × 10¹⁴</Code> entries, which is unimaginable to train. RVQ is not just a convenience; it is what makes end-to-end neural codecs viable at broadcast-quality bitrates.
      </Prose>

      <StepTrace
        label="RVQ levels progressively refine reconstruction"
        steps={[
          {
            label: "level 1 · mse 0.573 (56% of baseline)",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textMuted, padding: 12 }}>
                r₀ = feature  ·  k⁽¹⁾ = argmin‖r₀ − e⁽¹⁾‖  ·  r₁ = r₀ − e⁽¹⁾_k
                <br />residual energy: <span style={{ color: colors.gold }}>56.1%</span> of input
              </div>
            ),
          },
          {
            label: "level 2 · mse 0.316 (31%)",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textMuted, padding: 12 }}>
                r₁ = residual from level 1  ·  k⁽²⁾ = argmin‖r₁ − e⁽²⁾‖  ·  r₂ = r₁ − e⁽²⁾_k
                <br />residual energy: <span style={{ color: colors.gold }}>30.9%</span> of input
              </div>
            ),
          },
          {
            label: "level 4 · mse 0.093 (9%)",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textMuted, padding: 12 }}>
                Four codebooks of size 64. 24 bits per frame. Residual down to 9% of input energy. Sufficient for intelligible speech at ~3 kbps.
              </div>
            ),
          },
          {
            label: "level 8 · mse 0.007 (0.7%)",
            render: () => (
              <div style={{ fontFamily: "monospace", fontSize: 12, color: colors.textMuted, padding: 12 }}>
                Eight codebooks. 48 bits per frame. Residual energy at 0.7% of input. Near-transparent reconstruction at ~6 kbps (for 75 frames/sec audio).
              </div>
            ),
          },
        ]}
      />

      <Plot
        label="RVQ reconstruction MSE vs number of levels"
        series={[
          {
            name: "relative mse",
            points: [
              [0, 1.0],
              [1, 0.561],
              [2, 0.309],
              [3, 0.169],
              [4, 0.091],
              [5, 0.049],
              [6, 0.026],
              [7, 0.014],
              [8, 0.007],
            ],
          },
        ]}
        xLabel="rvq level"
        yLabel="mse / baseline"
      />

      <H3>4d. Interleaved multimodal sequences</H3>

      <Prose>
        The sequence-construction step is where a multimodal model's input actually gets built. The function is trivial — insert modality-marker tokens before and after each modality's block, concatenate — but it is load-bearing. Every bug in a multimodal training run that manifests as "the model ignores the image" or "the model generates image tokens in text regions" traces back to this function or to the tokenizer that feeds it.
      </Prose>

      <CodeBlock language="python">
{`def build_interleaved(image_tokens, text_tokens,
                      img_open="<img>", img_close="</img>"):
    """Wrap image tokens in boundary markers, append text tokens, return flat list."""
    return [img_open, *image_tokens, img_close, *text_tokens]

image_tokens = [4021, 178, 8832, 1045, 3117, 901]
text_tokens  = ["This", " is", " a", " cat", "."]

seq = build_interleaved(image_tokens, text_tokens)

# Actual output (verified):
# seq length : 13
# sequence   : ['<img>', 4021, 178, 8832, 1045, 3117, 901, '</img>',
#               'This', ' is', ' a', ' cat', '.']

# Two-image example: image + caption + image + caption.
seq2 = (build_interleaved([11, 22, 33], ["cat", "."])
      + build_interleaved([44, 55, 66], ["dog", "."]))
# ['<img>', 11, 22, 33, '</img>', 'cat', '.',
#  '<img>', 44, 55, 66, '</img>', 'dog', '.']`}
      </CodeBlock>

      <TokenStream
        label="interleaved multimodal sequence"
        tokens={[
          { label: "<img>", color: "#c084fc" },
          { label: "vis_4021", color: "#60a5fa" },
          { label: "vis_0178", color: "#60a5fa" },
          { label: "vis_8832", color: "#60a5fa" },
          { label: "vis_1045", color: "#60a5fa" },
          { label: "vis_3117", color: "#60a5fa" },
          { label: "vis_0901", color: "#60a5fa" },
          { label: "</img>", color: "#c084fc" },
          { label: "This", color: "#e2b55a" },
          { label: " is", color: "#e2b55a" },
          { label: " a", color: "#e2b55a" },
          { label: " cat", color: "#e2b55a" },
          { label: ".", color: "#e2b55a" },
        ]}
      />

      <Prose>
        Three implementation details that actually matter. First, the boundary markers must be <em>separate vocabulary entries</em>, not strings that get re-tokenized — otherwise BPE will split <Code>{"<img>"}</Code> into <Code>{"< im g >"}</Code> and the boundary information is lost. Chameleon adds the markers as special tokens before training. Second, the image token IDs must not collide with text token IDs in the shared embedding table. Chameleon solves this by mapping image codebook indices <Code>0..8191</Code> to text-vocab indices <Code>65536..73727</Code>, so the embedding layer sees a single integer range with no overlap. Third, the positional embedding scheme must handle the interleaving — either one monotonic position index across modalities (Chameleon's choice) or 2D positional encoding for image regions and 1D for text regions (Gemini's reported approach, though details are less public).
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Three libraries cover most of the production landscape. HuggingFace <Code>transformers</Code> ships with CLIP's image preprocessor and the Chameleon/Idefics model families. <Code>facebookresearch/encodec</Code> is the reference audio tokenizer. The original <Code>CompVis/taming-transformers</Code> repo provides VQ-GAN checkpoints that are still the default visual tokenizer for many research codebases.
      </Prose>

      <H3>CLIP image preprocessing</H3>

      <CodeBlock language="python">
{`from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

img = Image.open(requests.get("https://images.cocodataset.org/val2017/000000039769.jpg",
                              stream=True).raw)
inputs = processor(text=["a photo of a cat", "a photo of a dog"],
                   images=img, return_tensors="pt", padding=True)

# inputs.pixel_values: (1, 3, 224, 224) — normalized, center-cropped.
# inputs.input_ids:    (2, 10)          — tokenized captions.
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
# probs[0] -> tensor([0.99, 0.01])  <- image correctly matches "cat" caption.`}
      </CodeBlock>

      <Prose>
        CLIP is the continuous-embedding path, not the discrete one. Its "tokenizer" for images is the 32-pixel patch projection inside the ViT encoder; the output is a single 512-dim embedding per image (or 50 patch embeddings per image, depending on which layer you tap). For retrieval, zero-shot classification, or multimodal embedding alignment, this is what you want. For a model that generates images, you need the discrete path below.
      </Prose>

      <H3>VQ-GAN discrete tokenization</H3>

      <CodeBlock language="python">
{`# Reference — requires the CompVis/taming-transformers repo + a pretrained checkpoint.
from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import torch, torchvision.transforms as T

config = OmegaConf.load("configs/vqgan_imagenet_f16_16384.yaml")
model  = VQModel(**config.model.params)
model.load_state_dict(torch.load("vqgan_imagenet_f16_16384.ckpt")["state_dict"], strict=False)
model.eval()

img = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])(pil_image).unsqueeze(0)
with torch.no_grad():
    z_q, _, (_, _, indices) = model.encode(img * 2 - 1)   # normalize to [-1, 1]

# indices: (256,) — a 16x16 grid of codebook indices into a 16384-entry codebook.
# z_q: (1, 256, 16, 16) — the quantized feature map.
# Decode back to pixels:
recon = model.decode(z_q)`}
      </CodeBlock>

      <Prose>
        The <Code>f16</Code> in the config name is the spatial downsampling factor: a 256×256 input becomes a 16×16 feature grid, so one image is 256 tokens. The codebook has 16,384 entries, so each token carries 14 bits. This configuration is what DALL·E-M, Parti-style, and MaskGIT-style generators use as their image representation. At inference, the transformer predicts the 256 token grid autoregressively (or in parallel for non-autoregressive variants), and the VQ-GAN decoder renders the result to pixels.
      </Prose>

      <H3>EnCodec audio tokenization</H3>

      <CodeBlock language="python">
{`from transformers import EncodecModel, AutoProcessor
import torchaudio

model     = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

audio, sr = torchaudio.load("example.wav")
if sr != 24000:
    audio = torchaudio.functional.resample(audio, sr, 24000)

inputs = processor(raw_audio=audio[0].numpy(), sampling_rate=24000, return_tensors="pt")
with torch.no_grad():
    out = model.encode(inputs["input_values"], inputs["padding_mask"])

# out.audio_codes: (1, 1, n_quantizers, n_frames)
# For 24kHz, 75 frames/sec. 8 quantizers at bandwidth=6kbps, 32 at 24kbps.
# Flattened to a single integer stream for LM training: AudioLM / MusicLM approach.

# Decode back to waveform:
recon = model.decode(out.audio_codes, out.audio_scales, inputs["padding_mask"])`}
      </CodeBlock>

      <Prose>
        EnCodec at 24 kHz compresses audio by a factor of 320, producing 75 frames per second of input. Each frame gets quantized by 2 to 32 RVQ levels depending on the target bitrate (1.5 kbps to 24 kbps). For language-model training over audio, the typical setup is 8 levels at 6 kbps, giving 75 × 8 = 600 tokens per second. Ten seconds of audio is 6,000 tokens, which fits comfortably in a 32k context window alongside a text prompt. MusicLM and AudioLM use this exact configuration. For a streaming ASR or TTS system where lower latency matters, you would drop to 4 levels at 3 kbps and accept lower reconstruction quality.
      </Prose>

      <Prose>
        One structural note about how RVQ tokens get consumed by a language model. The RVQ output is a 2D array — <Code>(n_frames, n_levels)</Code> — but language models want a 1D sequence. There are two canonical flattening strategies. The "coarse-first" order emits all of level 1 for every frame, then all of level 2, and so on; this is what AudioLM's coarse/fine split does, and it works well when an upstream "semantic token" pass has already handled long-range structure. The "interleaved per-frame" order emits level 1 through <Code>L</Code> for frame 1, then level 1 through <Code>L</Code> for frame 2, and so on; this keeps related tokens adjacent, which helps attention but produces longer dependencies between layer 1 of frame <Code>t</Code> and layer 1 of frame <Code>t+1</Code>. The choice materially affects downstream model quality, and different papers have reported different winners depending on the domain. If you are designing a new audio LM from scratch, run both and measure.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        Four visualizations trace the full pipeline from raw pixels to interleaved sequence. Follow the same input through all four views: a photograph enters, gets chopped into a patch grid, each patch feature is quantized to a code in a learned codebook, and the resulting integer indices merge with text tokens into a single sequence the transformer can consume end-to-end.
      </Prose>

      <PatchGrid
        label="step 1 — ViT patch grid (14x14 on a 224 image)"
        src="https://picsum.photos/seed/mm-walk/224/224"
        patches={14}
        size={260}
      />

      <Prose>
        The patchifier imposes a regular grid. Each cell becomes one token position. A ViT-B/16 sees 196 patches; a ViT-L/14 sees 256. The choice of patch size is a compute-quality knob we will quantify in section 7.
      </Prose>

      <Heatmap
        label="step 2 — codebook usage distribution (8x8 = 64 codes; dark cells are dead)"
        matrix={[
          [0, 11, 12, 2, 192, 0, 0, 6],
          [0, 0, 552, 1, 29, 11, 286, 4],
          [16, 0, 0, 0, 1, 302, 204, 101],
          [191, 14, 26, 1, 0, 86, 0, 0],
          [0, 0, 0, 0, 1, 51, 915, 140],
          [31, 0, 0, 923, 0, 0, 31, 2155],
          [0, 0, 53, 26, 0, 15, 22, 0],
          [173, 200, 57, 199, 0, 402, 0, 0],
        ]}
        cellSize={32}
        colorScale="purple"
      />

      <Prose>
        The quantizer maps each patch feature to one of <Code>K</Code> learned codes. The usage distribution is never uniform — a handful of codes absorb most assignments, many are dead. The ratio of used-to-dead codes is the single most informative diagnostic of VQ-VAE training health.
      </Prose>

      <Plot
        label="step 3 — RVQ cumulative reconstruction error per level"
        series={[
          {
            name: "mse / baseline",
            points: [
              [0, 1.0], [1, 0.561], [2, 0.309], [3, 0.169],
              [4, 0.091], [5, 0.049], [6, 0.026], [7, 0.014], [8, 0.007],
            ],
          },
        ]}
        xLabel="rvq level"
        yLabel="rel mse"
      />

      <Prose>
        Each RVQ level quantizes the previous level's residual. The error curve is roughly geometric — each layer halves the remaining error under reasonable codebook sizes. The practical decision is where to stop: more levels cost more tokens per second of audio.
      </Prose>

      <TokenStream
        label="step 4 — interleaved output sequence (text + two images + text)"
        tokens={[
          { label: "Here", color: "#e2b55a" },
          { label: " is", color: "#e2b55a" },
          { label: "<img>", color: "#c084fc" },
          { label: "vis_4021", color: "#60a5fa" },
          { label: "vis_0178", color: "#60a5fa" },
          { label: "vis_8832", color: "#60a5fa" },
          { label: "</img>", color: "#c084fc" },
          { label: " and", color: "#e2b55a" },
          { label: "<img>", color: "#c084fc" },
          { label: "vis_1045", color: "#60a5fa" },
          { label: "vis_3117", color: "#60a5fa" },
          { label: "</img>", color: "#c084fc" },
          { label: ".", color: "#e2b55a" },
        ]}
      />

      <Prose>
        The transformer sees one flat sequence of integers. It learns through training which integer ranges mean text and which mean image, and which boundary markers to respect. No special routing, no cross-attention modules, no modality-specific heads.
      </Prose>

      <Prose>
        The four-step pipeline above is the entire logical path from pixels to transformer input, but production systems spread the work across library boundaries that do not always respect the logical structure. CLIP's image preprocessor handles resize + center-crop + mean/std normalization; the ViT patch projection lives inside the model's first layer; a VQ-GAN tokenizer is a separate model with its own checkpoint; the interleaving happens in the multimodal model's tokenizer wrapper, which also handles the special-token insertion. A bug anywhere in this chain — a mismatched normalization constant, a patch-size inconsistency, a special token that gets re-tokenized — manifests as degraded downstream quality with no clear error message. When debugging a multimodal pipeline, the first thing to verify is that the same preprocessor, the same patch size, and the same vocabulary are in use at every stage end-to-end.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Continuous embeddings vs discrete tokens</H3>

      <Prose>
        Continuous embeddings (CLIP, LLaVA, Flamingo, most production vision-language models) win when the model only needs to <em>read</em> images and generate text. Training is simpler because the image encoder is separate and can be trained once and reused. The encoder's embedding space is richer than any finite codebook, so fine visual detail that matters for grounding, OCR, or spatial reasoning is preserved. Discrete tokens (DALL·E, Parti, MaskGIT, Chameleon, Gemini's image-generation path) win when the model needs to <em>generate</em> images. Sampling from a discrete vocabulary plays nicely with cross-entropy, autoregressive decoding, and speculative prefill; sampling from a continuous embedding space requires bolt-on decoders that break the end-to-end training story. If the answer is "both," you either maintain two pipelines (most production systems) or commit to tokens everywhere (Chameleon's bet).
      </Prose>

      <H3>VQ-VAE vs VQ-GAN vs RVQ</H3>

      <Prose>
        VQ-VAE is the simplest and produces the blurriest reconstructions — pixel-space MSE loss is a known underperformer for perceptual quality, and the decoder has no reason to produce sharp edges or plausible textures when averaging over uncertainty scores lower pixel-MSE. VQ-GAN keeps the VQ-VAE skeleton but swaps the pixel-MSE loss for LPIPS perceptual loss plus an adversarial discriminator; the result is dramatically sharper reconstructions at the same codebook size, at the cost of more fragile training (GAN dynamics are always fragile). RVQ is orthogonal to both — it is a way to stack quantization layers regardless of the loss used on top. In practice, VQ-GAN + RVQ together is the recipe used by FSQ-style codecs and modern audio systems; VQ-GAN alone is still the default for image generation at the 256-token-per-image scale; VQ-VAE alone survives only in research codebases that have not been updated.
      </Prose>

      <H3>Patch size (image)</H3>

      <Prose>
        Token count scales as <Code>(H/P)²</Code> where <Code>P</Code> is the patch side. For a 224×224 image, <Code>P</Code>=32 gives 49 tokens, <Code>P</Code>=16 gives 196, <Code>P</Code>=8 gives 784. Each halving of patch size quadruples token count and the attention cost within the image. Larger patches lose fine detail — a 32×32 patch compresses 3,072 float values into one embedding, and small text or faces degrade first. Smaller patches cost more compute and context. Production vision-language models almost all use <Code>P</Code>=14 or <Code>P</Code>=16; research systems exploring high-resolution inputs sometimes drop to <Code>P</Code>=8 on critical image regions only. The right setting is task-dependent: OCR and scene-text tasks need small patches; image classification and retrieval tolerate large ones.
      </Prose>

      <H3>Audio RVQ levels</H3>

      <Prose>
        For a neural codec (compress audio, decompress later, no language model involved) the target is transparency — the reconstructed audio must be indistinguishable from the original at listening tests. EnCodec at 24 kbps uses 32 RVQ levels; at this setting, MUSHRA scores approach transparent. For language-model training over audio tokens, the target is compression — fewer tokens per second means longer possible audio inside a fixed context window. MusicLM trains on 8 levels at 6 kbps: 600 tokens per second of audio, good enough for a language model to learn musical structure while still fitting 30 seconds of audio in an 18k-token budget. AudioLM uses a split design with "coarse" semantic tokens from a separate codec and "fine" acoustic tokens from EnCodec, interleaved. The general rule: drop levels aggressively when the downstream model will regenerate the missing detail from context, keep levels when the tokenizer is the final stage.
      </Prose>

      {/* ======================================================================
          8. SCALING ANALYSIS
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Resolution</H3>

      <Prose>
        Image token count scales quadratically with resolution under a fixed patch size. A 224×224 image at <Code>P</Code>=16 is 196 tokens; a 1024×1024 image at the same <Code>P</Code> is 4,096 tokens. Double the resolution, quadruple the tokens, quadruple the attention cost inside the image region, quadruple the time per step. This is why modern vision models handle high-resolution inputs with tiling (break the image into 224×224 tiles, tokenize each separately, aggregate later) rather than scaling the ViT. Gemini 1.5's reported long-context behavior on multi-image and video inputs relies on tiling plus aggressive spatial compression during encoding. Naively scaling patch-based tokenization to 4k video frames breaks at the context-window level before it breaks at the quality level.
      </Prose>

      <Prose>
        There is a second, less obvious resolution effect: the quality of the patches themselves depends on the relationship between patch size and the scale of structures in the image. A 16-pixel patch covering most of a face in a cropped portrait carries a lot of signal per patch; the same 16-pixel patch covering an eyelash in a high-resolution shot carries almost nothing — it is a few pixels of skin and one or two of hair. Doubling the input resolution without shrinking the patch size hands the model more tokens but each token is less individually informative. Doubling resolution and halving patch size to compensate restores per-token information but multiplies token count by 16. There is no free lunch; high-resolution inputs always cost either sequence length or per-token informativeness, and the tokenizer's job is to pick a balance.
      </Prose>

      <H3>Codebook size</H3>

      <Prose>
        Larger codebooks improve reconstruction quality but scale badly in three ways. First, training stability degrades — the codebook update rule only touches entries that are assigned to at least one encoder output per batch, and with <Code>K</Code> large enough, most entries get zero assignments most of the time. Second, dead-code fraction rises — at <Code>K</Code>=16384 without EMA and restarts, 60%+ of codes routinely end up dead. Third, the argmin lookup cost grows linearly in <Code>K</Code>; at inference, with long sequences and large codebooks, this starts to matter. Chameleon's 8,192-entry codebook is near the sweet spot for image tokenization; pushing to 65k would require EMA updates, aggressive code-reset, and a serious argmin-acceleration trick.
      </Prose>

      <H3>Sequence length</H3>

      <Prose>
        All multimodal tokens compete with text for the context window. A 1024×1024 image at 256 tokens, plus one minute of audio at 600 tokens/sec = 36,000 tokens, plus a text prompt of a few hundred tokens, plus the model's generated response — the arithmetic fills a 100k-token context fast. The production tension is that every additional token of input fidelity costs one token of output budget. Long-context models (Gemini 1.5 at 10M tokens, Claude at 200k, GPT-4 at 128k) push the ceiling but do not change the tradeoff. Downsampling the tokenizer — fewer patches per image, fewer RVQ levels per second of audio — is almost always a better lever than upscaling the model.
      </Prose>

      <H3>Fidelity vs sequence length</H3>

      <Prose>
        The fundamental multimodal tradeoff is not between quality and compute per se, but between quality and tokens. More tokens means more fidelity but slower inference, bigger KV cache, fewer simultaneous modalities in context. Every multimodal tokenizer sits on a Pareto frontier defined by this tradeoff. DALL·E 1 at 1024 tokens per image, MaskGIT at 256, SDXL (continuous VAE latents) at 4096 float vectors — these are all points on the same curve, differing in where they trade off reconstruction quality against sequence budget. The engineering question is never "how many tokens fully represent this image" (the answer is always "more than we can afford"); it is "how few tokens preserve enough for the downstream task," and the answer depends on the task.
      </Prose>

      <Prose>
        A useful sanity-check heuristic for any tokenizer budget decision: write out the token count for a realistic input at the candidate configuration, multiply by the bytes-per-token in the KV cache (typically 2 × n_layers × d_model × 2 bytes for fp16 per token per layer), and check whether the resulting memory fits within your serving hardware's per-request budget. A 70B model at fp16 with 80 layers and 8192 hidden dim spends roughly 2.5 MB of KV cache per token. Ten thousand multimodal tokens per request is 25 GB of KV just for that one request, which is most of an H100. The tokenizer's fidelity setting is, in effect, a memory-budget parameter dressed up as a quality parameter. Cutting image tokens from 1024 to 256 per image is a 4× memory reduction for the image portion of every context, which propagates directly into how many concurrent users a serving node can handle.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes & gotchas</H2>

      <H3>Dead codes</H3>

      <Prose>
        A codebook entry that is never the nearest neighbor of any encoder output never receives a gradient, so it stays wherever it was initialized — usually in the wrong place — and stays dead forever. On standard VQ-VAE training runs without countermeasures, 40-60% of codes typically end up dead. Fixes: EMA codebook updates (codebook entries move as exponential moving averages of assigned features, not via gradient descent, so they stay near the data manifold even with intermittent assignments); random restart (reinitialize any entry with low running usage to a randomly sampled encoder output); k-means-style reinitialization every N training steps. Production codebases like VQGAN use EMA updates by default.
      </Prose>

      <H3>Codebook collapse</H3>

      <Prose>
        The opposite failure: almost all encoder outputs map to the same one or two codes. Usually caused by an encoder that produces near-constant outputs — if the encoder's output has too little variation, nearest-neighbor is dominated by whichever codebook entry happens to be closest to the center of mass, and every feature maps there. Symptoms: nearly-uniform reconstructions, catastrophic KL-like divergence of the codebook entries from the encoder outputs, <Code>>95%</Code> of assignments to one or two codes. Fixes: increase the commitment loss coefficient <Code>β</Code>, add batch normalization to the encoder output, or restart the codebook entirely from a fresh k-means clustering of a batch of encoder outputs.
      </Prose>

      <H3>Posterior-collapse analog</H3>

      <Prose>
        In VAEs with powerful decoders, the decoder learns to ignore the latent and reconstruct from the prior alone. VQ-VAE avoids this in the usual sense — the discrete code is a hard bottleneck — but a softer version shows up: the encoder learns to produce near-identical features at every spatial position, so every position quantizes to the same code. Training loss looks fine because the decoder learns to produce plausible images from a constant token grid. Symptoms: zero spatial variance in the token grid, reconstructions that look like a mean image no matter the input. Fix: add per-position reconstruction terms or auxiliary losses that force the encoder to preserve spatial information.
      </Prose>

      <H3>STE giving wrong-direction gradients</H3>

      <Prose>
        The straight-through estimator assumes that the gradient of the loss with respect to <Code>z_q</Code> is also the right direction for <Code>z_e</Code>. This assumption breaks when the argmin crosses a codebook-cell boundary: the loss wants to push <Code>z_q</Code> toward a different code, but STE pushes <Code>z_e</Code> in the direction of the <em>current</em> code. In practice this manifests as training plateaus where the codebook assignment pattern refuses to change even as the loss stays flat. Fixes: use EMA codebook updates (which sidestep STE for the codebook direction), or use finite-difference-style estimators like rotation trick or Gumbel-softmax with a temperature schedule.
      </Prose>

      <H3>Modality-boundary confusion</H3>

      <Prose>
        The model treats a text token as if it were an image token, or vice versa. Root cause is almost always vocabulary collision or tokenization bugs. If image codebook indices and text token IDs overlap in the shared embedding table, the model has no way to distinguish "token 42 = codebook entry 42" from "token 42 = BPE token 42"; both map to the same embedding. Fix: keep the ranges strictly non-overlapping (text ends at <Code>V_text</Code>, image starts at <Code>V_text</Code>, audio at <Code>V_text + V_image</Code>). Second root cause is boundary tokens that get re-tokenized — if <Code>{"<img>"}</Code> is a string, not a dedicated vocab entry, BPE will shatter it. Fix: add modality markers as special tokens before training, not at inference.
      </Prose>

      <H3>RVQ level imbalance</H3>

      <Prose>
        Early RVQ layers carry most of the signal; later layers carry mostly noise. If the first codebook is too small or badly initialized, it captures too little, and the later codebooks spend their capacity duplicating what a larger first codebook would have covered. If the first codebook is too large, later codebooks find only rounding-error residuals to quantize and end up underutilized. The usual diagnostic is per-layer entropy: compute <Code>H(k^(l)) = -Σ p_i log p_i</Code> over the usage distribution of each level; a healthy RVQ shows entropy near <Code>log K</Code> at every level; an unhealthy one shows entropy collapsing to near-zero at later levels. Fix: initialize each level from k-means on the residual of the previous level (not random), and monitor per-level entropy during training.
      </Prose>

      <H3>Audio tokenizer train-test mismatch</H3>

      <Prose>
        Codecs trained on speech fail catastrophically on music, and vice versa. Speech has a narrow dynamic range, predominantly voiced signal, and strong harmonic structure; music has wide dynamic range, polyphony, percussion, and non-harmonic content. An EnCodec checkpoint trained on LibriSpeech will produce unlistenable artifacts on a piano recording, not because the codec is broken but because its codebook entries were fit to a distribution that does not contain piano. Fix: match the codec's training distribution to the deployment distribution, or use a general-purpose codec (EnCodec's <Code>24khz</Code> variant) trained on mixed audio.
      </Prose>

      <H3>Video temporal redundancy</H3>

      <Prose>
        Naive frame-level tokenization produces <Code>T × N_patches</Code> tokens for a <Code>T</Code>-frame video. For a 5-second clip at 30 fps with 256 tokens per frame, that is 38,400 tokens — 95% of which are duplicates of the previous frame's tokens because adjacent frames are nearly identical. MagViT, MagViT-2, and VideoPoet address this with space-time VQ-VAE: quantize 3D blocks (16 pixels × 16 pixels × 4 frames) rather than 2D patches, with temporal downsampling factors of 4 to 8. A 5-second HD clip becomes a few thousand tokens, not tens of thousands. Without temporal compression, video tokenization is useless for any LM-based generation.
      </Prose>

      <H3>Preprocessor normalizer mismatches</H3>

      <Prose>
        CLIP's image preprocessor normalizes with specific mean/std values (<Code>(0.48145466, 0.4578275, 0.40821073)</Code> mean, <Code>(0.26862954, 0.26130258, 0.27577711)</Code> std). Using different values — the ImageNet defaults, or forgetting to normalize entirely — produces embeddings that are still numerically valid but are offset from the distribution the model trained on. The model then systematically misclassifies, with no error message and no obvious symptom. The same class of bug exists for audio (sample rate mismatches, amplitude normalization differences) and video (frame rate mismatches). Fix: always use the tokenizer's companion preprocessor, never roll your own, and log the preprocessor config at training time for reproducibility.
      </Prose>

      <H3>Emoji and ZWJ in interleaved text</H3>

      <Prose>
        Text tokenizers use byte-level BPE, which handles arbitrary Unicode including emoji with zero-width-joiner (ZWJ) sequences. But multimodal models sometimes mix BPE text tokens with image-codebook tokens in arithmetic ways — adding a fixed offset to image tokens to avoid vocab collision, or concatenating with string-based markers. These arithmetic operations can silently corrupt multi-byte UTF-8 sequences if the code path assumes ASCII. Symptoms: emoji in input prompts come out mangled in the output, family-emoji ZWJ sequences split into their components. Fix: always operate on integer token ID sequences, never on string concatenation, and test the round-trip with a stress input containing every Unicode category.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Every source below was WebSearch-verified against arXiv during the writing of this topic. The identifiers and titles are exact.
      </Prose>

      <Prose>
        <strong>Dosovitskiy et al., 2020.</strong> "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." <Code>arXiv:2010.11929</Code>. Published ICLR 2021. Introduces the Vision Transformer — image as sequence of 16×16 patches, linear projection to the transformer's hidden dimension, learned positional embeddings. The foundational paper for every patch-based visual tokenizer since.
      </Prose>

      <Prose>
        <strong>van den Oord, Vinyals, Kavukcuoglu, 2017.</strong> "Neural Discrete Representation Learning." <Code>arXiv:1711.00937</Code>. Published NeurIPS 2017. Introduces VQ-VAE — the learned codebook, the three-term loss with stop-gradients, the straight-through estimator applied to argmin quantization. The foundational paper for every discrete visual tokenizer.
      </Prose>

      <Prose>
        <strong>Esser, Rombach, Ommer, 2020.</strong> "Taming Transformers for High-Resolution Image Synthesis." <Code>arXiv:2012.09841</Code>. Published CVPR 2021 (oral). Introduces VQ-GAN — adds LPIPS perceptual loss and an adversarial discriminator to VQ-VAE, producing codebooks that reconstruct sharp images at up to megapixel resolution. The de-facto visual tokenizer for most downstream text-to-image generation research.
      </Prose>

      <Prose>
        <strong>Zeghidour et al., 2021.</strong> "SoundStream: An End-to-End Neural Audio Codec." <Code>arXiv:2107.03312</Code>. Google. Introduces end-to-end learned audio codecs with residual vector quantization — multiple codebooks stacked, each quantizing the previous layer's residual. Establishes the 3 kbps to 18 kbps bitrate range with a single model via structured dropout over RVQ levels.
      </Prose>

      <Prose>
        <strong>Défossez et al., 2022.</strong> "High Fidelity Neural Audio Compression." <Code>arXiv:2210.13438</Code>. Meta FAIR. Introduces EnCodec — production-grade neural audio codec at 24 kHz and 48 kHz. Adds a loss balancer for stable training, a lightweight transformer for further compression, and extensive MUSHRA evaluation. Widely used as the audio tokenizer in language-model-based audio generation systems.
      </Prose>

      <Prose>
        <strong>Radford et al., 2021.</strong> "Learning Transferable Visual Models From Natural Language Supervision." <Code>arXiv:2103.00020</Code>. OpenAI. Introduces CLIP — contrastive training over 400 million image-text pairs, jointly learning an image encoder and a text encoder that map to a shared embedding space. The foundational paper for continuous-embedding multimodal models, and the reference point against which every discrete multimodal approach is compared.
      </Prose>

      <Prose>
        <strong>Meta FAIR, 2024.</strong> "Chameleon: Mixed-Modal Early-Fusion Foundation Models." <Code>arXiv:2405.09818</Code>. Introduces a decoder-only transformer trained from scratch on interleaved text and image tokens with a unified vocabulary. The image tokenizer maps 512×512 images to 32×32 grids of 8,192-entry codebook indices. Demonstrates that native multimodality — a single transformer with no modality-specific modules — is competitive with the best specialized models.
      </Prose>

      <Prose>
        <strong>Google, 2024.</strong> "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context." <Code>arXiv:2403.05530</Code>. Technical report for the Gemini 1.5 family. Multimodal MoE with near-perfect recall at 10M tokens; supports interleaved audio, video, text, and image inputs in a single sequence. Tokenization details are less public than Chameleon's but the architecture is another data point for the "everything is tokens" design.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        <strong>1. Image sequence length.</strong> A ViT-style tokenizer is applied to a 1024×1024 RGB image at patch size 16. How many tokens does this produce? What if the patch size drops to 8? What is the attention cost (in FLOPs, ignoring constants) of self-attention within the image region in each case? Answer: 4,096 tokens at P=16, 16,384 at P=8. Attention is quadratic in sequence length, so P=8 is 16× more expensive than P=16 just for the image's self-attention.
      </Prose>

      <Prose>
        <strong>2. Why commitment loss.</strong> Remove the commitment loss term from the VQ-VAE objective (keep reconstruction and codebook loss only). What fails during training? Answer: the encoder has no pressure to produce outputs that lie near codebook entries. It can produce arbitrarily large-magnitude features, and the codebook — pulled toward those features by the codebook-loss term — can chase them indefinitely. Training diverges. The commitment loss anchors the encoder's output magnitude; <Code>β</Code> controls how hard the anchor pulls.
      </Prose>

      <Prose>
        <strong>3. RVQ configuration design.</strong> You need to store 10 seconds of 24 kHz audio in at most 200 tokens. The neural codec downsamples by 320×. How many frames per second of audio? How many RVQ levels can you afford at what codebook size, given the 200-token budget? Answer: 24000/320 = 75 frames per second of input; 750 frames in 10 seconds. 200 tokens / 750 frames is under 1 token per frame — impossible without dropping to subframe resolution. Either increase the downsampling factor (640×, giving 37.5 frames/sec = 375 frames; still more than 200), drop to a different codec design, or accept that 10 seconds in 200 tokens is below any reasonable neural codec's operating point. This is the kind of back-of-envelope check that should be the first thing you do when designing a multimodal context budget.
      </Prose>

      <Prose>
        <strong>4. Dead-code diagnosis.</strong> You trained a VQ-VAE with <Code>K</Code>=8192 and observe that 70% of codebook entries receive zero assignments on the validation set. List three interventions, in order of how much they will cost you to implement. Answer: (1) add EMA codebook updates (5 lines of code, no retraining needed for the interface, just a new optimizer branch); (2) add random-restart for codes with usage below a threshold every N steps (20 lines, compatible with EMA); (3) reduce <Code>K</Code> to a smaller value with a retrained run (most expensive, but in many cases the honest answer — you simply had too many codes for your data scale).
      </Prose>

      <Prose>
        <strong>5. Modality vocabulary layout.</strong> You are training a Chameleon-style model with 65,536 text tokens and an 8,192-entry image codebook. Design the unified vocabulary: what is the total size, which ID range is which, and where do the modality markers <Code>{"<img>"}</Code> and <Code>{"</img>"}</Code> go? Answer: Total = 65,536 + 8,192 + 2 = 73,730. IDs 0..65,535 are text tokens (including whatever special tokens BPE allocates). IDs 65,536..73,727 are image codebook entries (map codebook index <Code>k</Code> to global ID <Code>65536 + k</Code>). IDs 73,728 and 73,729 are <Code>{"<img>"}</Code> and <Code>{"</img>"}</Code>. All three ranges are strictly non-overlapping in the embedding table.
      </Prose>

      <Callout accent="gold">
        The tokenizer is the interface between the world and the model. It determines what the model can perceive, what it can generate, how long its sequences are, how fast its inference runs, and how gracefully it handles inputs outside its training distribution. No amount of model scale recovers information the tokenizer discarded.
      </Callout>
    </div>
  ),
};

export default multimodalTokenization;
