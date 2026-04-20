import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { PatchGrid, TokenStream, Heatmap } from "../../components/viz";
import { colors } from "../../styles";

const multimodalTokenization = {
  title: "Multimodal Tokenization (Visual, Audio, Video)",
  readTime: "15 min",
  content: () => (
    <div>
      <Prose>
        Three topics in, every example of tokenization has assumed the same thing: the input is text, and the question is how to chop it into subword pieces. That assumption collapses the moment you ask a model to read a photograph, transcribe a recording, or describe a video clip. The input is now a pixel grid, a waveform, or a sequence of frames — continuous, high-dimensional, and carrying no built-in notion of where one "token" ends and the next begins. Getting those signals into a transformer requires solving the same fundamental problem that BPE solved for language, but for signals that the tokenizer literature almost never discusses.
      </Prose>

      <H2>Why multimodal models need discrete units</H2>

      <Prose>
        Two architectural philosophies exist. The first keeps everything continuous: encode the image (or audio, or video) with a learned encoder, project the resulting embeddings into the transformer's hidden dimension, and interleave them with text embeddings from the language side. CLIP does this for images; BLIP, Flamingo, and LLaVA follow the same pattern. It is a clean and practical approach — no new vocabulary, no codebook to train, just an encoder that maps pixels to vectors the transformer already speaks. The architecture is modular and the image embeddings do not need to live in the same space as text tokens.
      </Prose>

      <Prose>
        The second philosophy discretizes. Instead of feeding continuous embeddings, the model first compresses each modality into a sequence of discrete integer indices — tokens in exactly the same sense as subword tokens. Once that step is done, images, audio clips, and video segments all look identical to the transformer: just more integers from a vocabulary. The cross-entropy loss that the model uses for text now applies to image patches and audio frames too. Sampling, beam search, and speculative decoding work unchanged. The transformer can generate images, generate audio, or generate interleaved multimodal sequences using the exact same machinery it uses to generate the next word. This is the approach taken by VQ-VAE, DALL·E 1, Parti, MaskGIT, and — in a form close enough to count — Chameleon and Gemini. The payoff is composability. The price is the need to train a good discretizer for each modality.
      </Prose>

      <H2>Image patches — the ViT trick</H2>

      <Prose>
        The Vision Transformer (ViT), introduced by Dosovitskiy et al. in 2020, established the foundational vocabulary for visual tokenization without actually producing discrete tokens. The idea is elegant: cut the input image into a regular grid of non-overlapping square patches, flatten each patch into a vector, and project each vector into the model's hidden dimension. A 224×224 image with 16×16 patches produces a 14×14 grid — 196 patches, each treated as one token in the attention sequence. Positional embeddings tag each patch with its location in the grid, and from that point forward the transformer has no idea whether it is reading text or image patches. The attention mechanism operates identically.
      </Prose>

      <PatchGrid
        label="ViT-style 14×14 patches on a 224-pixel image"
        src="https://picsum.photos/seed/multimodal/224/224"
        patches={14}
        size={280}
      />

      <Prose>
        The crucial caveat is that ViT patch embeddings are continuous. Flattening a 16×16×3 patch produces a 768-dimensional float vector; there is no codebook, no argmin, no discrete index. The output of the patch-projection layer is real-valued and every possible image maps to a unique trajectory through that space. This is perfectly fine for discriminative models — classification, retrieval, grounding — where the downstream task has a fixed label set anyway. But it is a problem for generative multimodal models. You cannot sample from a continuous embedding distribution the way you sample from a discrete vocabulary. You cannot apply cross-entropy loss to a float vector. The next step — discretization — is what turns the ViT trick into a genuine tokenizer.
      </Prose>

      <H2>VQ-VAE and learned codebooks</H2>

      <Prose>
        Vector Quantization Variational Autoencoders, introduced by van den Oord et al. in 2017, are the machinery behind discrete visual tokenization. The architecture has three parts: an encoder that maps an input image to a spatial grid of continuous feature vectors, a codebook of <Code>K</Code> learned embedding vectors, and a decoder that reconstructs the image from the quantized features. At each spatial position in the feature grid, the encoder produces a vector <Code>z_e(x)</Code>. Quantization replaces it with the nearest codebook entry <Code>e_k</Code>, emitting its index <Code>k</Code>. A 32×32 feature grid over a 256×256 image becomes a sequence of 1,024 discrete integers, each drawn from a vocabulary of size <Code>K</Code> — typically 512 to 8,192. That sequence is the image's token representation.
      </Prose>

      <MathBlock>
        {"\\mathcal{L} = \\|x - \\hat{x}\\|^2 + \\|\\text{sg}[z_e(x)] - e\\|^2 + \\beta \\|z_e(x) - \\text{sg}[e]\\|^2"}
      </MathBlock>

      <Prose>
        The loss has three terms. The first is reconstruction loss — pixel-space MSE between the original image <Code>x</Code> and the decoder's output <Code>x̂</Code>. The second is codebook loss, which pulls the codebook entries toward the encoder's outputs; the stop-gradient operator <Code>sg[·]</Code> prevents gradients from flowing through the encoder here. The third is commitment loss, which pulls the encoder's outputs toward the codebook entries; <Code>β</Code> is a small coefficient, typically 0.25, that controls how hard the encoder commits to its nearest codebook entry. The stop-gradient on <Code>sg[e]</Code> prevents gradients from flowing through the codebook in this term. Together, the two stop-gradients create a stable training dynamic where the encoder learns to produce vectors that land near codebook entries, and the codebook entries learn to cover the space of encoder outputs, without either chasing the other in circles.
      </Prose>

      <Prose>
        The non-differentiable argmin — the nearest-neighbor lookup that maps continuous encoder outputs to discrete codebook indices — breaks the backpropagation chain. Gradients cannot flow through an argmin. The straight-through estimator sidesteps this: during the forward pass, use the quantized code; during the backward pass, pretend the argmin was an identity function and pass gradients directly from the decoder back to the encoder as if quantization had not occurred. This is an approximation, but it works surprisingly well in practice, and it is the standard trick in every VQ-based tokenizer since the original VQ-VAE. Two important successors refine the approach: VQ-GAN (Esser et al., 2021) adds an adversarial discriminator to the reconstruction objective, producing sharper, perceptually more faithful reconstructions at the cost of more complex training dynamics. RQ-VAE (Lee et al., 2022) stacks codebooks residually — the first codebook quantizes the encoder output coarsely, the second quantizes the residual error, and so on — achieving much higher fidelity at the same codebook size by spreading the representational budget across multiple quantization stages instead of one.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np

def patchify(image, patch_size=16):
    """image: (H, W, 3) uint8 array. Returns (N_patches, patch_size*patch_size*3)."""
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = image.reshape(H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)  # (h, w, ps, ps, c)
    return patches.reshape(-1, patch_size * patch_size * C)

# A 224x224 image -> 196 patches of 768 dims each (ViT-B/16).
# VQ-VAE replaces the flatten step with a learned encoder + nearest-codebook lookup,
# so each patch becomes a single integer codebook index instead of a vector.`}
      </CodeBlock>

      <H2>Audio — SoundStream, EnCodec, and residual vector quantization</H2>

      <Prose>
        Audio presents the same compression challenge as images but in one dimension rather than two, and with much higher temporal resolution. A single second of audio at 24 kHz is 24,000 samples. Even at 16 kHz, feeding raw waveform values into a transformer would produce sequences hundreds of times longer than the text sequences the architecture was designed for. SoundStream (Zeghidour et al., 2021) and EnCodec (Défossez et al., 2023) solve this with a convolutional encoder that downsamples the waveform aggressively — typically by a factor of 320 or more — reducing one second of audio to around 75 vectors. Residual Vector Quantization (RVQ) then discretizes those vectors: a first codebook quantizes each frame coarsely, capturing the dominant structure of the signal. The first codebook's residual error is fed to a second codebook, which captures finer structure. The residuals of the second codebook go to a third, and so on, for four to twelve stages depending on the target bitrate.
      </Prose>

      <Prose>
        The output of RVQ is not a single stream of tokens but a set of parallel streams, one per codebook stage. For a model trained to generate audio autoregressively, the typical approach is to flatten these streams into a single interleaved sequence — the "coarse-to-fine" ordering where coarse codebook tokens come first and each fine-detail token follows its parent. This is exactly how AudioLM (Borsos et al., 2022), MusicLM (Agostinelli et al., 2023), and recent generative audio systems like Suno frame audio generation: as a language modeling problem over a vocabulary of audio tokens produced by an EnCodec-style codec. The transformer generating audio is, in every meaningful sense, doing the same thing as a transformer generating text — it is predicting the next integer in a sequence. The meaning of that integer happens to be "the closest codebook entry to this 13-millisecond audio frame" rather than "the subword that continues this sentence," but the mechanism is identical.
      </Prose>

      <H3>Video — space-time patches</H3>

      <Prose>
        Video adds the time dimension on top of the spatial compression problem images already present. Naively extending image tokenization to video — tokenizing each frame independently and concatenating the token sequences — produces unmanageable sequence lengths even for short clips. A five-second video at 24 frames per second is 120 frames; if each frame produces 256 image tokens, that is 30,720 tokens before any temporal compression. MagViT (Yu et al., 2022) and its successor MagViT-2 address this by treating video as a three-dimensional volume — height, width, and time — and applying a space-time VQ-VAE that compresses along all three axes simultaneously. Temporal downsampling factors of 4 to 8 are typical, combined with spatial compression; a five-second HD clip might compress to a few thousand tokens. This is the representational substrate underlying Sora, Veo, and other recent generative video systems. The generative model is a transformer — or a masked generative model like MaskGIT — operating over a vocabulary of space-time patch tokens the way a language model operates over subwords. The difference between generating a sentence and generating a video clip is, at the architecture level, almost entirely in the tokenizer.
      </Prose>

      <H2>Interleaved multimodal sequences</H2>

      <Prose>
        Discretization's payoff becomes clear once you look at what it enables at the sequence level. If images are tokens, audio is tokens, and text is tokens, then a multimodal model's input is just a longer token sequence with several different "languages" interleaved. Special boundary tokens delimit modality transitions — a pair of <Code>&lt;img&gt;</Code> and <Code>&lt;/img&gt;</Code> markers wrapping a block of visual tokens, for instance — and the transformer attends across all of them without any architectural modification. Gemini processes image tokens and text tokens in the same sequence. Chameleon (Meta, 2024) trains a language model jointly over text and image tokens from scratch, using a unified vocabulary where subword tokens and codebook indices share the same embedding table. Fuyu and MM1 follow similar patterns. The practical result is that the model learns cross-modal dependencies through the same attention mechanism it uses for in-text coreference: an image token early in the sequence can directly influence the text token generated two hundred positions later, and the model can learn when and how to route that information without any special cross-attention module.
      </Prose>

      <TokenStream
        label="interleaved multimodal sequence"
        tokens={[
          { label: "<img>", color: "#c084fc" },
          { label: "vis_4021", color: "#60a5fa" },
          { label: "vis_0178", color: "#60a5fa" },
          { label: "vis_8832", color: "#60a5fa" },
          { label: "vis_1045", color: "#60a5fa" },
          { label: "</img>", color: "#c084fc" },
          { label: "This", color: "#e2b55a" },
          { label: " is", color: "#e2b55a" },
          { label: " a", color: "#e2b55a" },
          { label: " cat", color: "#e2b55a" },
          { label: ".", color: "#e2b55a" },
        ]}
      />

      <H3>The fidelity vs. sequence-length tradeoff</H3>

      <Prose>
        Every multimodal tokenizer sits on a curve. On one end: more tokens per image, per second of audio, per frame of video — higher fidelity reconstruction, more information preserved, better downstream quality from the model's perspective. On the other end: fewer tokens, coarser representation, faster inference, smaller KV cache, more media fitting inside a context window. DALL·E 1 used 32×32 = 1,024 tokens per image from a codebook of 8,192 entries. MaskGIT and Parti used similar configurations. Gemini's multimodal tokenization is less publicly documented, but its context efficiency suggests aggressive spatial compression. The engineering question is never "how many tokens fully represent this image" — the answer is always "more than we can afford" — but "how few tokens preserve enough for the downstream task." For a model that generates image captions, coarse visual tokens that preserve semantic content may be sufficient. For a model that generates images, the tokenizer becomes the reconstruction bottleneck and fidelity is everything.
      </Prose>

      <Callout accent="gold">
        Tokenizer choice sets the floor on what a multimodal model can see and the ceiling on how fast it can respond. No amount of model scale recovers information the tokenizer discarded.
      </Callout>

      <Prose>
        The tokenizers in this topic share one property with every text tokenizer in the preceding topics: their vocabulary is fixed at training time. The codebook learned by a VQ-VAE is frozen after training; so is the merge list of a BPE tokenizer, the Unigram probability table, and the SentencePiece model. Every input — image, audio, text — gets mapped to a vocabulary that was decided before the model saw a single downstream task. The final topic in this section asks whether that has to be true. Can tokenization itself adapt to the input, to the task, or to the model's evolving representation?
      </Prose>
    </div>
  ),
};

export default multimodalTokenization;
