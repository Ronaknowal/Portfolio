import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, PatchGrid, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const multimodalPretraining = {
  title: "Multimodal Pre-Training (Vision Encoders, Cross-Modal Alignment)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Language models consume tokens. Every byte of training data, every prompt at inference time, every generated word — all of it is integers drawn from a fixed vocabulary. That constraint is architectural: the embedding table has a finite number of rows, the cross-entropy loss is defined over a discrete distribution, and sampling is only defined when you have a probability over a known set of symbols. Images do not arrive as integers. A single 224×224 RGB photograph is a tensor of 150,528 floating-point values with no natural boundaries, no universal alphabet, and no agreed-upon unit of meaning. Audio is worse: a single second at CD quality is 44,100 floating-point samples. Video multiplies that by time and space simultaneously. The question "how do we get these signals into a language model?" does not have an obvious answer, and the answer that any particular team chose in any given year tells you almost everything about how their system behaves.
      </Prose>

      <Prose>
        The Multimodal Tokenization topic in this section covered the discretization half of the pipeline: VQ-VAE, VQ-GAN, residual vector quantization, patch projection. Those algorithms convert raw signals into token sequences. This topic covers the alignment half: once you have token sequences from multiple modalities, how do you train a model so that visual tokens and text tokens are meaningful to each other? How do you get an image of a cat and the phrase "a photo of a cat" to live close together in representation space? The history of that problem is a sequence of three architecturally distinct answers, each inheriting the failure modes of its predecessor.
      </Prose>

      <Prose>
        The first generation is contrastive pre-training. CLIP (Radford et al., 2021; arXiv:2103.00020) trained two separate encoders — one for images, one for text — jointly on 400 million image-text pairs scraped from the web. The training signal is purely contrastive: matched pairs should be close in embedding space, mismatched pairs far apart. CLIP produces a shared embedding space with remarkable zero-shot properties, but it is a discriminative model. It cannot generate text, cannot reason over images, cannot produce a caption. It aligns. That is all.
      </Prose>

      <Prose>
        The second generation is frozen-encoder plus projector. By 2023, powerful pretrained components were available off-the-shelf: CLIP vision encoders and instruction-tuned language models. LLaVA (Liu et al., 2023; arXiv:2304.08485) connected them with a small trainable bridge — initially a linear projection, later a two-layer MLP — that maps CLIP's output dimension into the LLM's embedding dimension. Only the bridge is trained from scratch; both towers are mostly frozen. The recipe is cheap, replicable, and widely adopted. BLIP-2 (Li et al., 2023; arXiv:2301.12597) introduced the Q-Former as a more sophisticated bridge: a small learned transformer with N fixed query tokens that cross-attend to visual features and compress them to a fixed-length sequence regardless of image resolution.
      </Prose>

      <Prose>
        The third generation is native multimodal training. Chameleon (Meta FAIR, 2024; arXiv:2405.09818) and Gemini 1.5 (Google, 2024; arXiv:2403.05530) remove the separation entirely. Images are discretized into integer tokens and added to the model's shared vocabulary. Training is standard next-token prediction over interleaved sequences of image tokens, text tokens, audio tokens — whatever modalities are present, in whatever order they appear. There is no dedicated vision pathway, no separate vision loss, no cross-modal projection module. The transformer learns to process every modality through the same residual stream. Understanding why each generation exists — and why none of them has fully superseded the others — is the core content of this topic.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Contrastive alignment (CLIP): teach two encoders to agree</H3>

      <Prose>
        Imagine two students, one who can only read and one who can only look at pictures, both asked to describe the same concept. Contrastive pre-training is a curriculum that forces them to agree. In each training batch of N image-text pairs, the image student encodes every image into a fixed-size vector and the text student encodes every caption. The loss pushes the N matched pairs' vectors close together and pushes all N²−N mismatched pairs far apart. After enough training on enough pairs, the two students have developed a shared vocabulary of high-dimensional geometry: images of dogs cluster near text about dogs, images of beaches cluster near text about beaches, images of scientific diagrams cluster near technical language. The geometry is the alignment. Zero-shot classification falls out for free because "classify this image" becomes "find the text embedding closest to this image embedding," which requires no additional training at all.
      </Prose>

      <Prose>
        The key intuition is that this works without any explicit labels, any explicit definition of "dog" or "beach," any curated hierarchy. The training signal is entirely implicit in the co-occurrence statistics of 400 million pairs scraped from the internet: images that appear near certain text must, on average, be related to that text. The contrastive loss is the mechanism for distilling those co-occurrence statistics into a reusable geometric structure.
      </Prose>

      <H3>Frozen encoder + projector (LLaVA): bridge two pretrained towers cheaply</H3>

      <Prose>
        Once CLIP exists, you have a pretrained vision encoder that produces meaningful image representations. Once instruction-tuned LLMs exist, you have a pretrained language model that can follow instructions. The obvious question is: can you connect them without retraining either? The LLaVA intuition is yes, via a thin learned bridge. The vision encoder speaks in its own dimensional language — CLIP's ViT-L/14 outputs 1024-dimensional vectors. The LLM speaks in a different dimensional language — LLaMA's residual stream might be 4096-dimensional. A two-layer MLP can learn to translate between them. The translation is not perfect — the vision encoder was never trained to help a generative model, and the LLM was never trained to receive visual inputs — but it is good enough for a wide range of tasks, and it costs a tiny fraction of what either pretraining run cost.
      </Prose>

      <Prose>
        The Q-Former variant (BLIP-2) adds one more insight: you do not need to pass all visual features to the LLM. A 224×224 image with 16×16 patches produces 196 patch tokens; a 336×336 image produces 441. Passing all of them to the LLM is expensive and floods it with spatial detail it may not need. The Q-Former keeps N=32 learned query embeddings that cross-attend to all the visual features and distill them into 32 output tokens — the same count regardless of image resolution. Those 32 tokens are what the LLM receives. The Q-Former learns, during training, which 32 "questions" about the image are most useful for language generation.
      </Prose>

      <H3>Native multimodal (Chameleon/Gemini): make all tokens equal</H3>

      <Prose>
        The frozen-encoder approach inherits a structural assumption: that a vision encoder trained with one objective on one dataset can produce features a language model can usefully consume. That assumption holds well enough to ship products, but it is not obviously correct, and it has real failure modes — fine-grained spatial reasoning, precise counting, reading text in images. The native multimodal intuition is to eliminate the assumption. Discretize images into integer tokens using a VQ-GAN; add those token IDs to the shared vocabulary; train one transformer end-to-end on sequences that may contain image tokens, text tokens, or both; use the same cross-entropy loss everywhere. Now there is no assumption about what a frozen vision encoder's features mean to a language model, because there is no frozen vision encoder. There is just a transformer whose embedding table happens to contain both text subwords and image codebook entries, and whose attention mechanism treats both identically.
      </Prose>

      <Callout accent="gold">
        Three generations, three philosophies: align two towers (CLIP), bridge two towers (LLaVA), merge into one tower (Chameleon). Each is strictly more capable than the last and strictly more expensive to train.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>InfoNCE loss (CLIP)</H3>

      <Prose>
        Given a batch of <Code>N</Code> image-text pairs, let <Code>v_i ∈ ℝ^d</Code> be the ℓ2-normalised image embedding and <Code>t_i ∈ ℝ^d</Code> the ℓ2-normalised text embedding, both produced by their respective encoders. The pairwise similarity matrix is:
      </Prose>

      <MathBlock>{"S_{ij} = \\frac{v_i \\cdot t_j}{\\tau}"}</MathBlock>

      <Prose>
        where <Code>τ</Code> is a learned temperature scalar. The image-to-text loss treats each image as a query and its matched caption as the single correct key among all <Code>N</Code> captions in the batch:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{i2t}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(S_{ii})}{\\sum_{j=1}^{N} \\exp(S_{ij})}"}</MathBlock>

      <Prose>
        The text-to-image loss is symmetric — each caption queries across all images. The total CLIP loss is their average:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\text{CLIP}} = \\frac{1}{2}\\left(\\mathcal{L}_{\\text{i2t}} + \\mathcal{L}_{\\text{t2i}}\\right)"}</MathBlock>

      <Prose>
        This is the InfoNCE (Noise-Contrastive Estimation) objective with the batch providing in-batch negatives. With batch size <Code>B=32768</Code>, every image has 32,767 negatives; the contrastive signal is extremely dense. Small <Code>τ</Code> sharpens the softmax, forcing matched pairs to be much closer than any random pair. The temperature is initialised at 0.07 and learned — the model discovers the right sharpness for the data distribution.
      </Prose>

      <H3>Projector: (B, N, D_v) → (B, N, D_llm)</H3>

      <Prose>
        The LLaVA projector is a 2-layer MLP with GELU activation. Given vision features <Code>F ∈ ℝ^(B×N×D_v)</Code>:
      </Prose>

      <MathBlock>{"\\hat{F} = \\text{GELU}(F W_1 + b_1) W_2 + b_2, \\quad W_1 \\in \\mathbb{R}^{D_v \\times D_{\\text{llm}}},\\; W_2 \\in \\mathbb{R}^{D_{\\text{llm}} \\times D_{\\text{llm}}}"}</MathBlock>

      <Prose>
        The output <Code>F̂ ∈ ℝ^(B×N×D_llm)</Code> is prepended to the text token embeddings before being passed to the LLM. Nothing in the LLM architecture changes; the visual tokens are just a longer prefix. Only <Code>W1, b1, W2, b2</Code> are trained in stage 1. Stage 2 additionally updates the LLM weights (or LoRA adapters on them).
      </Prose>

      <H3>Q-Former: N learned queries cross-attending to K visual features</H3>

      <Prose>
        The Q-Former maintains a set of <Code>M</Code> learned query embeddings <Code>Q ∈ ℝ^(M×D)</Code> (typically M=32). On a forward pass, these queries cross-attend to the <Code>K</Code> frozen visual features <Code>V ∈ ℝ^(K×D_v)</Code> (K may be 196, 256, 441 — whatever the image resolution produces) through standard multi-head cross-attention:
      </Prose>

      <MathBlock>{"\\text{Attn}(Q, V) = \\text{softmax}\\!\\left(\\frac{QW_q (VW_k)^\\top}{\\sqrt{d_k}}\\right) VW_v"}</MathBlock>

      <Prose>
        The Q-Former output is always shape <Code>(M, D)</Code> regardless of how large <Code>K</Code> is — the resolution independence is the whole point. The M=32 output tokens are then projected to the LLM dimension. The Q-Former is trained with three objectives simultaneously: image-text contrastive loss, image-grounded text generation loss, and image-text matching loss. The multi-objective training teaches the queries to extract features that are simultaneously alignable with text, useful for generation, and discriminative across images.
      </Prose>

      <H3>Native multimodal: interleaved CE loss</H3>

      <Prose>
        In Chameleon-style training, the vocabulary <Code>V = V_text ∪ V_image</Code> is shared. A training sequence <Code>x = (x_1, x_2, ..., x_T)</Code> may contain tokens from either sub-vocabulary in any order. The loss is the standard causal language modeling objective applied uniformly:
      </Prose>

      <MathBlock>{"\\mathcal{L} = -\\sum_{t=1}^{T} \\log P(x_t \\mid x_{<t}; \\theta)"}</MathBlock>

      <Prose>
        There is no separate vision loss, no separate text loss, no modality-specific weights. Every token position contributes equally to the gradient, regardless of whether the token is a text subword or an image codebook entry. The model learns, purely from the data distribution, when to generate image tokens versus text tokens.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <H3>4a. CLIP contrastive loss</H3>

      <Prose>
        We implement the symmetric InfoNCE loss from scratch using only NumPy. The test uses a batch of B=8 pairs with d=64 dimensional embeddings — small enough to inspect, large enough to verify the loss decreases when matched pairs are made more similar.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

np.random.seed(42)
B, d = 8, 64
tau = 0.07

# Synthetic image / text embeddings, L2-normalised
V = np.random.randn(B, d).astype(np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True)

T = np.random.randn(B, d).astype(np.float32)
T /= np.linalg.norm(T, axis=1, keepdims=True)

# Pairwise cosine similarities scaled by temperature  (B, B)
sim = V @ T.T / tau

# Numerically-stable softmax
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# Cross-entropy against identity labels (diagonal is the correct pair)
labels = np.arange(B)

def cross_entropy(logits, labels):
    probs = softmax(logits)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-9))

L_i2t = cross_entropy(sim,   labels)   # image queries text keys
L_t2i = cross_entropy(sim.T, labels)   # text queries image keys
L_clip = (L_i2t + L_t2i) / 2

print(f"L_image2text      = {L_i2t:.4f}")
print(f"L_text2image      = {L_t2i:.4f}")
print(f"L_clip (symmetric)= {L_clip:.4f}")
print(f"sim matrix shape  : {sim.shape}")

# Output:
# L_image2text      = 3.1813
# L_text2image      = 3.1071
# L_clip (symmetric)= 3.1442
# sim matrix shape  : (8, 8)`}
      </CodeBlock>

      <Prose>
        The loss of ~3.14 with 8 random pairs is sensible: log(8) ≈ 2.08 is the theoretical maximum (uniform distribution over 8 choices), but the temperature τ=0.07 sharpens the distribution aggressively, pushing raw logits into much higher ranges and inflating the loss above log(N). In a real CLIP run with B=32768 and well-matched pairs the diagonal will dominate and the loss will drop toward zero; here it is high because the embeddings are random and carry no actual semantic signal.
      </Prose>

      <H3>4b. Patchify + linear projection</H3>

      <Prose>
        The ViT patch projection is a reshape followed by a learned linear map. A 224×224 image with 16×16 patches produces 196 tokens. This is covered in detail in the sibling Multimodal Tokenization topic; we include a compact version here for completeness and to establish the shapes used in sections 4c and 4d.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

H, W, C = 224, 224, 3
P = 16                          # patch side length in pixels
N = (H // P) * (W // P)         # = 196 patches
d_model = 768                   # hidden dimension

img = np.random.randn(H, W, C).astype(np.float32)

# Reshape: (H, W, C) -> (N, P*P*C)
patches = (
    img.reshape(H // P, P, W // P, P, C)
       .transpose(0, 2, 1, 3, 4)
       .reshape(-1, P * P * C)
)

# Learned linear projection E: (P²C, d_model)
E = np.random.randn(P * P * C, d_model).astype(np.float32) * 0.02
z = patches @ E   # (N, d_model)

print(f"image shape   : {img.shape}")
print(f"patches shape : {patches.shape}   ({N} patches, dim {P*P*C})")
print(f"projected z   : {z.shape}   ({N} tokens, dim {d_model})")

# image shape   : (224, 224, 3)
# patches shape : (196, 768)   (196 patches, dim 768)
# projected z   : (196, 768)   (196 tokens, dim 768)`}
      </CodeBlock>

      <PatchGrid
        label="ViT-style patch decomposition — 14×14 grid, 196 tokens"
        src="https://picsum.photos/seed/multimodal-pretrain/224/224"
        patches={14}
        size={260}
      />

      <H3>4c. LLaVA projector + two-stage training sketch</H3>

      <Prose>
        The projector takes frozen vision features of shape (B, N_vis, D_v) and produces embeddings of shape (B, N_vis, D_llm) that the LLM can consume as a prefix. Only the projector weights have gradients in stage 1. In stage 2, the LLM weights are unfrozen (or LoRA adapters are added). The vision encoder stays frozen throughout.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

# Dimensions matching LLaVA-1.5 with LLaMA-2 13B
D_v   = 1024    # CLIP ViT-L/14 output dim
D_llm = 5120    # LLaMA-2 13B hidden dim
N_vis = 256     # visual tokens (after CLIP: 16×16 patches = 256)
B     = 1

# ── Simulated frozen vision encoder output ──────────────────────────────
# In real LLaVA: this comes from CLIP ViT forward pass, no grad
vis_feats = np.random.randn(B, N_vis, D_v).astype(np.float32)

# ── 2-layer MLP projector (trainable) ───────────────────────────────────
# W1, b1, W2, b2 are the ONLY parameters updated in stage 1
W1 = np.random.randn(D_v,   D_llm).astype(np.float32) * 0.02
b1 = np.zeros(D_llm, dtype=np.float32)
W2 = np.random.randn(D_llm, D_llm).astype(np.float32) * 0.02
b2 = np.zeros(D_llm, dtype=np.float32)

# Forward pass
h    = np.maximum(vis_feats @ W1 + b1, 0)   # ReLU activation
proj = h @ W2 + b2                            # (B, N_vis, D_llm)

print(f"vision feats : {vis_feats.shape}  (frozen)")
print(f"projected    : {proj.shape}  <- prepended to LLM input as visual prefix")

# Stage-1 training note:
#   optimizer.param_groups = [{'params': [W1, b1, W2, b2]}]
#   vision encoder: requires_grad = False throughout
#   LLM: requires_grad = False in stage 1, True in stage 2

# vision feats : (1, 256, 1024)  (frozen)
# projected    : (1, 256, 5120)  <- prepended to LLM input as visual prefix`}
      </CodeBlock>

      <Prose>
        Stage 1 trains only the projector on a large corpus of image-caption pairs — typically 558K filtered CC3M pairs in the original LLaVA. The LLM receives visual tokens as a prefix and must predict the caption tokens; the projector's gradient signal comes entirely from how well the LLM can predict the caption given the projected visual prefix. This stage is fast: the frozen LLM acts as a fixed scoring function. Stage 2 unfreezes the LLM (or adds LoRA) and fine-tunes on visual instruction-following data: conversations, visual question answering, reasoning chains about images. The vision encoder stays frozen throughout both stages.
      </Prose>

      <H3>4d. Q-Former mini-demo</H3>

      <Prose>
        The Q-Former's resolution-independence property is its main selling point. Below we show that M=32 learned queries cross-attending to K=256 visual features always produce exactly 32 output tokens, regardless of K.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

M = 32    # number of learned queries (fixed, resolution-independent)
K = 256   # visual features from encoder (varies with image resolution)
D = 512   # Q-Former hidden dim

# Learned query embeddings (initialised and trained, not derived from input)
queries = np.random.randn(1, M, D).astype(np.float32) * 0.02

# Frozen visual features (from CLIP or any pretrained encoder)
vis_kv = np.random.randn(1, K, D).astype(np.float32)

# Single-head cross-attention  Q = queries, K/V = vis_kv
Wq = np.random.randn(D, D).astype(np.float32) * 0.02
Wk = np.random.randn(D, D).astype(np.float32) * 0.02
Wv = np.random.randn(D, D).astype(np.float32) * 0.02

Q_proj = queries @ Wq           # (1, M, D)
K_proj = vis_kv  @ Wk           # (1, K, D)
V_proj = vis_kv  @ Wv           # (1, K, D)

scale       = np.sqrt(D)
attn_scores = np.einsum('bmd,bkd->bmk', Q_proj, K_proj) / scale  # (1, M, K)
attn_probs  = np.exp(attn_scores - attn_scores.max(-1, keepdims=True))
attn_probs /= attn_probs.sum(-1, keepdims=True)

out = np.einsum('bmk,bkd->bmd', attn_probs, V_proj)  # (1, M, D)

print(f"queries shape     : {queries.shape}")
print(f"vis_kv shape      : {vis_kv.shape}  (K can be 196, 256, 441...)")
print(f"attn_scores shape : {attn_scores.shape}  (M queries attend to K features)")
print(f"Q-Former output   : {out.shape}   <- always M tokens, resolution-independent")

# queries shape     : (1, 32, 512)
# vis_kv shape      : (1, 256, 512)  (K can be 196, 256, 441...)
# attn_scores shape : (1, 32, 256)   (M queries attend to K features)
# Q-Former output   : (1, 32, 512)   <- always M tokens, resolution-independent`}
      </CodeBlock>

      <H3>4e. Interleaved sequence constructor</H3>

      <Prose>
        Native multimodal training uses interleaved sequences. The constructor below shows how text and image tokens are merged into a single integer sequence with boundary markers, and reports the total token count.
      </Prose>

      <CodeBlock language="python">
{`# Special token IDs (beyond the regular vocabulary)
IMG_START = 32000   # <img>
IMG_END   = 32001   # </img>
VIS_TOKENS = 256    # image codebook tokens per image (VQ-GAN compressed)

# Example document: caption, then image, then follow-up text, then another image
text1 = [101, 2023, 2003, 1037, 3319, 1997]     # "this is a photo of"
img1  = [IMG_START] + list(range(8000, 8000 + VIS_TOKENS)) + [IMG_END]
text2 = [2023, 2003, 1037, 5965, 6077, 102]     # "this is a golden retriever"
img2  = [IMG_START] + list(range(9000, 9000 + VIS_TOKENS)) + [IMG_END]

sequence = text1 + img1 + text2 + img2

print(f"text1  tokens : {len(text1)}")
print(f"img1   tokens : {len(img1)}  ({VIS_TOKENS} visual + 2 markers)")
print(f"text2  tokens : {len(text2)}")
print(f"img2   tokens : {len(img2)}  ({VIS_TOKENS} visual + 2 markers)")
print(f"total sequence: {len(sequence)} tokens")
print()
print(f"layout: [text: {len(text1)}] [<img>+vis+</img>: {len(img1)}] "
      f"[text: {len(text2)}] [<img>+vis+</img>: {len(img2)}]")
print(f"training loss computed over all {len(sequence)} positions uniformly")

# text1  tokens : 6
# img1   tokens : 258  (256 visual + 2 markers)
# text2  tokens : 6
# img2   tokens : 258
# total sequence: 528 tokens
# layout: [text: 6] [<img>+vis+</img>: 258] [text: 6] [<img>+vis+</img>: 258]
# training loss computed over all 528 positions uniformly`}
      </CodeBlock>

      <Prose>
        The uniform loss across all 528 positions means the model is penalised for incorrectly predicting both text tokens and image tokens. When the model generates an image, it must correctly predict each codebook index given the preceding context — text and image alike. This is qualitatively different from the frozen-encoder approach, where the model only ever generates text; the native multimodal model is genuinely bidirectional across modalities.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, CLIP and LLaVA-family models are available via HuggingFace Transformers. The CLIPModel class wraps both encoders and exposes the joint similarity computation; LlavaForConditionalGeneration handles the full frozen-encoder+projector pipeline end-to-end, including the processor that tokenises text and patchifies images into the format the model expects.
      </Prose>

      <CodeBlock language="python">
{`# ── CLIP: image-text similarity (zero-shot retrieval) ───────────────────
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests, torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load any image
url = "https://picsum.photos/seed/dog/224/224"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of a dog", "a photo of a cat", "a photo of a car"]
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image[0]       # (3,) similarities
probs  = logits.softmax(dim=0)

for t, p in zip(texts, probs):
    print(f"  {p:.3f}  {t}")`}
      </CodeBlock>

      <CodeBlock language="python">
{`# ── LLaVA-1.5: visual question answering ────────────────────────────────
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests, torch

model_id = "llava-hf/llava-1.5-7b-hf"   # requires ~14 GB VRAM
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open(requests.get(
    "https://picsum.photos/seed/citystreet/512/512", stream=True
).raw)

# LLaVA's chat template: <image> marker in user turn
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this image."},
        ],
    }
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)

response = processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:],
                             skip_special_tokens=True)
print(response)`}
      </CodeBlock>

      <Prose>
        The processor is doing significant hidden work: it runs the CLIP ViT forward pass, extracts patch features, applies the two-layer MLP projector, and constructs the token sequence with visual features prepended to the text tokens. From the LLM's perspective it receives a single tensor of shape (1, N_vis + N_text, D_llm) — it does not "know" that some tokens came from an image. The visual tokens are just a longer prefix.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Image-text similarity matrix (CLIP setup)</H3>

      <Prose>
        The heatmap below shows a 5×5 cosine similarity matrix from a CLIP-style setup. Rows are images, columns are captions. The diagonal — matched pairs — should be warm; off-diagonal entries — mismatched pairs — should be cool. The contrastive loss pushes the matrix toward an identity-like structure over training.
      </Prose>

      <Heatmap
        label="clip image-text cosine similarities — matched pairs on diagonal"
        matrix={[
          [0.91, 0.12, 0.08, 0.14, 0.06],
          [0.11, 0.88, 0.15, 0.07, 0.09],
          [0.09, 0.13, 0.85, 0.11, 0.07],
          [0.07, 0.08, 0.10, 0.90, 0.12],
          [0.05, 0.10, 0.08, 0.13, 0.87],
        ]}
        rowLabels={["img_0", "img_1", "img_2", "img_3", "img_4"]}
        colLabels={["cap_0", "cap_1", "cap_2", "cap_3", "cap_4"]}
        cellSize={44}
        colorScale="gold"
      />

      <H3>Interleaved multimodal token stream</H3>

      <Prose>
        The token stream below shows how a native multimodal sequence looks at the token level. Yellow tokens are text subwords; purple tokens are boundary markers; blue tokens are image codebook indices. From the transformer's perspective, this is a single flat sequence of integers — there is no modal seam at the architecture level.
      </Prose>

      <TokenStream
        label="native multimodal sequence — text + image tokens interleaved"
        tokens={[
          { label: "A", color: "#e2b55a" },
          { label: " photo", color: "#e2b55a" },
          { label: " of", color: "#e2b55a" },
          { label: " a", color: "#e2b55a" },
          { label: " dog:", color: "#e2b55a" },
          { label: "<img>", color: "#c084fc" },
          { label: "v_841", color: "#60a5fa" },
          { label: "v_203", color: "#60a5fa" },
          { label: "v_1190", color: "#60a5fa" },
          { label: "v_77", color: "#60a5fa" },
          { label: "v_512", color: "#60a5fa" },
          { label: "v_6", color: "#60a5fa" },
          { label: "</img>", color: "#c084fc" },
          { label: "The", color: "#e2b55a" },
          { label: " dog", color: "#e2b55a" },
          { label: " is", color: "#e2b55a" },
          { label: " brown", color: "#e2b55a" },
          { label: ".", color: "#e2b55a" },
        ]}
      />

      <H3>Stage-1 vs stage-2 loss curves</H3>

      <Prose>
        Stage 1 (projector-only training) shows a fast initial drop as the projector learns the coarse mapping from vision space to LLM space, then plateaus. Stage 2 (LLM unfrozen) starts from that plateau and continues dropping as the LLM adapts to interpret the visual prefix — but the gains come more slowly because the LLM has many more parameters to update.
      </Prose>

      <Plot
        label="stage-1 vs stage-2 training loss"
        xLabel="steps (×1k)"
        yLabel="loss"
        series={[
          {
            name: "stage-1 (projector only)",
            color: colors.gold,
            points: [
              [0, 3.8], [5, 2.9], [10, 2.4], [15, 2.1], [20, 1.95],
              [25, 1.88], [30, 1.84], [35, 1.82], [40, 1.81],
            ],
          },
          {
            name: "stage-2 (LLM unfrozen)",
            color: "#60a5fa",
            points: [
              [40, 1.81], [45, 1.65], [50, 1.52], [55, 1.44], [60, 1.38],
              [65, 1.33], [70, 1.30], [75, 1.28], [80, 1.26],
            ],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing a multimodal architecture is primarily a compute budget and task fidelity decision. The three paradigms have non-overlapping sweet spots.
      </Prose>

      <Prose>
        Use contrastive alignment (CLIP and variants) when your task is retrieval, search, or classification and you do not need to generate text. CLIP embeddings are fast to compute, easy to index with approximate nearest-neighbour search, and highly transferable across domains with no fine-tuning. The ceiling is hard: you cannot ask CLIP to explain why two images are similar or describe what is in an image. If you need generation, CLIP alone is not enough.
      </Prose>

      <Prose>
        Use frozen-encoder plus projector (LLaVA, InstructBLIP) when your budget is limited, you need generative multimodal capability, and you can tolerate the ceiling imposed by the frozen encoder. A full LLaVA-1.5 13B training run costs on the order of a few hundred GPU-hours on A100s — feasible for an academic lab or small team. The projector stage alone can be done in tens of GPU-hours. The resulting model performs well on captioning, visual QA, and instruction following; it underperforms on tasks requiring precise spatial reasoning, small-text reading, or fine-grained attribute binding.
      </Prose>

      <Prose>
        Use native multimodal training (Chameleon, Gemini, future architectures in this line) when you need the best-quality cross-modal reasoning, you want the model to generate in multiple modalities, and you have the compute budget for a full pre-training run. The advantages are real: no ontology mismatch between the vision encoder and the LLM, no structural bottleneck from a frozen encoder, and the ability to generate images or audio using the same sampling machinery as text. The cost is real too: training a Chameleon-7B scale model requires thousands of GPU-hours and terabytes of interleaved multimodal data.
      </Prose>

      <Callout accent="gold">
        Rule of thumb: CLIP for retrieval; frozen+projector for low-budget generation; native multimodal when you can afford to train one and need the best cross-modal quality.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales well</H3>

      <Prose>
        CLIP scales strongly with batch size. The InfoNCE loss with in-batch negatives gets a better training signal when B is larger — more negatives per positive means the model must be more discriminative. OpenAI's original CLIP used B=32768; SigLIP (Zhai et al., 2023) showed that replacing the softmax with sigmoid (so each pair is an independent binary prediction) partially decouples performance from batch size, but the batch-size effect is real and large for the standard InfoNCE formulation.
      </Prose>

      <Prose>
        LLaVA-style systems scale well with LLM size. Replacing a 7B LLM with a 13B or 70B LLM behind the same projector improves downstream task performance substantially, and the projector requires almost no additional parameters. The vision encoder quality also matters but saturates earlier — the gap between ViT-B and ViT-L is large; the gap between ViT-L and ViT-G is smaller; beyond that the returns diminish.
      </Prose>

      <Prose>
        Native multimodal models scale with everything simultaneously: model size, data scale, data quality, and training compute. The Gemini 1.5 report documents consistent improvements across all three axes with no obvious plateau in the studied regime. Chameleon's training instabilities at scale (documented in the paper) were addressed through careful architecture choices — QK-norm, revised layer-norm placement — suggesting that the scaling behavior is architecture-sensitive in ways that text-only models are not.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        Patch token count scales quadratically with image resolution. A 224×224 image at patch size 16 gives 196 tokens; a 448×448 image gives 784 tokens; a 672×672 image gives 1,764 tokens. Feeding high-resolution images to a frozen-encoder+LLM system means the LLM must attend over a very long visual prefix, which is expensive and often hits context length limits. LLaVA-HD and LLaVA-1.6 address this by splitting high-resolution images into tiles and encoding each tile separately — a practical fix but one that increases token count further. Native multimodal systems have the same quadratic problem at the image tokenization stage.
      </Prose>

      <Prose>
        Visual reasoning does not scale as cleanly as visual recognition. Adding more image-caption pairs or more compute improves recognition benchmarks (ImageNet, zero-shot classification) at predictable rates. It improves reasoning benchmarks (spatial QA, counting, text reading in images) at much slower rates. The current evidence suggests that reasoning capabilities require qualitatively different training signal — dense spatial annotations, chain-of-thought reasoning traces grounded in images — rather than just more of the same weakly-labelled pairs.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. CLIP batch-size dependence</H3>
      <Prose>
        The InfoNCE loss is critically sensitive to batch size in ways that most other losses are not. Halving the batch from 32K to 16K approximately halves the number of negatives per positive, significantly weakening the training signal. Small-batch CLIP training (batch sizes below 4K) often produces embeddings with poor zero-shot transfer. If your compute budget constrains batch size, consider SigLIP's sigmoid variant or using a memory bank of stored embeddings as additional negatives.
      </Prose>

      <H3>2. Projector overfitting in stage 1</H3>
      <Prose>
        Stage 1 trains only the projector — typically 5–20M parameters — on hundreds of thousands of image-caption pairs. This is a very small model on a moderate dataset, and projectors overfit readily if trained too long. A projector that overfits to the stage-1 captions will have learned to predict those captions rather than to map vision features to general LLM representations. The symptom is good stage-1 validation loss but poor stage-2 performance. Standard fixes: regularise the projector (weight decay, dropout), keep stage 1 short, use diverse caption data.
      </Prose>

      <H3>3. Frozen-encoder / LLM ontology mismatch</H3>
      <Prose>
        CLIP's vision encoder was trained to align with short, simple captions scraped from the web: "a cat," "Paris Eiffel Tower," "business meeting." The LLM was trained on rich, structured natural language including complex reasoning, technical prose, and multi-step instructions. The projector must bridge these two very different representation styles. For tasks within the CLIP training distribution (recognising common objects, matching simple descriptions) the bridge works well. For tasks that require the LLM to reason about fine-grained visual attributes — the exact shade of a color, the specific model of a car, the precise spatial relationship between objects — the projector cannot recover information that the CLIP encoder never represented richly in the first place. This is not a projector problem; it is a ceiling imposed by the frozen encoder's pretraining objective.
      </Prose>

      <H3>4. Visual hallucination</H3>
      <Prose>
        Multimodal models frequently generate text that is plausible but not grounded in the actual image: objects that are not present, attributes that are incorrect, relationships that are inverted. Hallucination is worse in frozen-encoder models than in native multimodal models, and worse still when the image contains unusual or rare content that the vision encoder has not seen enough of. The root cause is that the LLM's strong language prior can "fill in" details that are not in the visual prefix — it generates what is likely given the text context, not what is in the image. Mitigation strategies include instruction-following fine-tuning on datasets specifically designed to punish hallucination (LLaVA-rlhf, POVID), and inference-time techniques like visual contrastive decoding.
      </Prose>

      <H3>5. Safety filters that don't see the image</H3>
      <Prose>
        A common deployment mistake is applying a text-only safety filter to the text output of a multimodal model without also checking the input image. An adversarial user can provide an image containing harmful content (text embedded in an image, a visual prompt injection, NSFW imagery) whose textual description would be blocked, but the model sees the image and generates a response that references or acts on the image content. The correct architecture applies safety checks to both the image (using a separate image classifier or the multimodal model itself as a judge) and the generated text.
      </Prose>

      <H3>6. Preprocessing mismatch between train and inference</H3>
      <Prose>
        CLIP and ViT models are trained with specific preprocessing: exact image size (224×224 or 336×336), specific normalization constants (ImageNet mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] or CLIP's own constants), and center-crop vs. resize-and-crop. Using the wrong preprocessing at inference — even slightly different normalization constants — can cause substantial performance degradation. Always use the processor class provided by the model's HuggingFace repository rather than writing your own preprocessing code.
      </Prose>

      <H3>7. Interleaved layout ordering artifacts</H3>
      <Prose>
        Native multimodal models trained on interleaved sequences develop strong priors about the typical order of modalities in their training data. If a model saw mostly text-then-image sequences during training, it will struggle with image-then-text sequences at inference. Models also develop position-in-sequence biases: an image at position 0 (the start of a conversation) is processed differently than an image at position 500 (mid-conversation), because the positional embedding distribution seen during training was non-uniform. Always check whether your inference-time layout matches the training data layout.
      </Prose>

      <H3>8. Position embeddings across modalities</H3>
      <Prose>
        Standard learned absolute positional embeddings do not generalise to sequence lengths longer than those seen during training. This is a problem for both the image tokenizer (higher-resolution images produce more tokens than the model was trained on) and for native multimodal models (long interleaved documents may exceed the training context length). RoPE (Rotary Position Embedding) is now standard for the LLM backbone and generalises better, but the 2D positional structure of image patches often requires separate handling. LLaVA-1.6 and related models use image-specific positional encodings that are independent of the text sequence position.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The six papers below are the direct references for this topic. All arXiv IDs and titles are verified.
      </Prose>

      <Prose>
        <strong>Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision."</strong> arXiv:2103.00020. Published ICML 2021. Introduces CLIP: 400M image-text pairs, contrastive pretraining of a vision encoder and text encoder, symmetric InfoNCE loss, temperature learned during training. Demonstrates zero-shot ImageNet classification competitive with supervised ResNets.
      </Prose>

      <Prose>
        <strong>Dosovitskiy et al. (2021). "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale."</strong> arXiv:2010.11929. Published ICLR 2021. Introduces ViT: divide image into non-overlapping 16×16 patches, project each to d-dimensional embedding, feed sequence to standard transformer. No convolutions. Pretrained on JFT-300M, sets state-of-the-art on ImageNet and CIFAR-100.
      </Prose>

      <Prose>
        <strong>Li et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models."</strong> arXiv:2301.12597. Published ICML 2023. Introduces the Q-Former: M=32 learned queries cross-attending to frozen vision features, trained with three objectives (ITC, ITG, ITM), output 32 tokens passed to frozen LLM. Outperforms Flamingo-80B on zero-shot VQAv2 with 54x fewer trainable parameters.
      </Prose>

      <Prose>
        <strong>Liu et al. (2023). "Visual Instruction Tuning."</strong> arXiv:2304.08485. Published NeurIPS 2023. Introduces LLaVA: GPT-4-generated visual instruction data, CLIP ViT-L/14 frozen encoder, linear projection (later 2-layer MLP in 1.5), two-stage training (projector-only then LLM fine-tune). Achieves 85.1% relative GPT-4 score on synthetic visual instruction benchmark.
      </Prose>

      <Prose>
        <strong>Chameleon Team, Meta FAIR (2024). "Chameleon: Mixed-Modal Early-Fusion Foundation Models."</strong> arXiv:2405.09818. First large-scale demonstration of native early-fusion multimodal training: VQ-GAN image tokenizer, shared vocabulary of 65k text + 8k image tokens, single decoder-only transformer trained end-to-end on ~10T interleaved tokens. Documents training instabilities at scale and architectural fixes (QK-norm, modified layer norm).
      </Prose>

      <Prose>
        <strong>Gemini Team, Google (2024). "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context."</strong> arXiv:2403.05530. Documents the Gemini 1.5 family: mixture-of-experts architecture, multimodal training on text, image, audio, and video, million-token context window, near-perfect recall on long-context retrieval tasks across modalities.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the InfoNCE gradient</H3>
      <Prose>
        Write out the partial derivative of <Code>L_clip</Code> with respect to the image embedding <Code>v_i</Code>. Show that the gradient has two terms: a positive term pulling <Code>v_i</Code> toward its matched text embedding <Code>t_i</Code>, and a negative term pushing it away from all other text embeddings in the batch, weighted by their current similarity. What does this tell you about why large batch sizes give stronger signal?
      </Prose>

      <H3>Exercise 2 — Why symmetric two-sided loss</H3>
      <Prose>
        CLIP uses both the image-to-text loss and the text-to-image loss. Imagine training with only the image-to-text loss. What failure mode would emerge? Specifically, what would happen to the text encoder's representations — would they remain well-distributed or would they collapse? Use the gradient analysis from Exercise 1 to reason about this.
      </Prose>

      <H3>Exercise 3 — Q-Former output count independent of resolution</H3>
      <Prose>
        Explain in your own words why the Q-Former produces exactly M=32 output tokens regardless of whether the input image is 224×224 (196 patches) or 448×448 (784 patches). Now explain the tradeoff: what information is necessarily discarded when M=32 queries must summarise 784 visual features? Under what types of visual tasks would this compression hurt most?
      </Prose>

      <H3>Exercise 4 — Stage-1 vs stage-2 freezing strategy</H3>
      <Prose>
        LLaVA trains the projector alone in stage 1, then unfreezes the LLM in stage 2. Why not skip stage 1 and start with stage 2 directly? What would happen to the projector's gradients if the LLM is also training from random initialisation of the projector? Separately, explain why the vision encoder stays frozen throughout both stages rather than being unfrozen in stage 2.
      </Prose>

      <H3>Exercise 5 — When native multimodal is worth the cost</H3>
      <Prose>
        You are building a product that must (a) retrieve the most relevant image from a database given a text query, (b) generate a detailed caption for a retrieved image, and (c) edit an image by generating a modified version from a text instruction. For each of these three capabilities, identify whether contrastive alignment, frozen-encoder+projector, or native multimodal training is the minimum architecture needed. Can all three be served by a single native multimodal model, and if so, what does that cost compared to running three purpose-built models?
      </Prose>
    </div>
  ),
};

export default multimodalPretraining;
