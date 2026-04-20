import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { PatchGrid, TokenStream } from "../../components/viz";

const multimodalPretraining = {
  title: "Multimodal Pre-Training (Vision Encoders, Cross-Modal Alignment)",
  readTime: "14 min",
  content: () => (
    <div>
      <Prose>
        The Multimodal Tokenization topic covered the front half of the pipeline: how pixels, audio frames, and video volumes become sequences of discrete tokens. But turning a photograph into 256 integers is only the beginning. Those integers have to mean something to a language model — have to live in the same representational space as the word "cat" before the model can produce a sensible caption, answer a question about the image, or generate text that reasons about what it sees.
      </Prose>

      <Prose>
        Getting visual tokens and text tokens into alignment is the problem that multimodal pre-training solves. The answer has evolved through three recognizable generations. The first — contrastive pre-training, epitomized by CLIP — built a shared embedding space through large-scale paired training. The second — frozen encoder with a trainable projector, the LLaVA recipe — bolted pretrained vision and language towers together at low cost. The third — native multimodal training, as in Gemini and Chameleon — collapses the separation entirely and trains one model end-to-end on interleaved text and image tokens from the start. Each generation inherits the limitations of the one before it. Understanding why tells you most of what you need to know about where the field is today.
      </Prose>

      <H2>CLIP — contrastive pre-training</H2>

      <Prose>
        OpenAI's CLIP (Radford et al., 2021) trained two encoders — a vision encoder and a text encoder — jointly on 400 million image-text pairs scraped from the web. The training signal is contrastive: in each batch of <Code>N</Code> image-text pairs, the model should assign high similarity to the <Code>N</Code> matching pairs and low similarity to the <Code>N² − N</Code> mismatched pairs. Image encoders in CLIP are Vision Transformers — the input image is divided into a grid of non-overlapping patches, each flattened and projected into the transformer's hidden dimension, then processed by standard self-attention. A 224×224 image with 16×16 patches yields 196 patch tokens, plus a learned <Code>[CLS]</Code> token whose final representation is the image embedding.
      </Prose>

      <PatchGrid
        label="clip-style vision encoder — image as patches"
        src="https://picsum.photos/seed/multimodal-pretrain/224/224"
        patches={14}
        size={260}
      />

      <Prose>
        The training objective is the InfoNCE loss, applied symmetrically over both image-to-text and text-to-image directions. Given a batch of <Code>N</Code> pairs, let <Code>v_i</Code> be the <Code>ℓ2</Code>-normalized image embedding and <Code>t_i</Code> the normalized text embedding. The loss for the image side is:
      </Prose>

      <MathBlock>{"\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(\\text{sim}(v_i, t_i) / \\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(v_i, t_j) / \\tau)}"}</MathBlock>

      <Prose>
        The temperature parameter <Code>τ</Code> is learned during training. A small <Code>τ</Code> sharpens the softmax, making the model push matched pairs very close and unmatched pairs very far apart. The symmetric version adds the same loss computed from the text side. With a batch size of 32,768 — each image has 32,767 negatives — the contrastive signal is remarkably strong without any explicit labels.
      </Prose>

      <Prose>
        What CLIP produces is a shared embedding space: images and texts map to vectors in the same high-dimensional manifold, with cosine similarity as a meaningful distance. Zero-shot classification falls out for free. To classify an image into one of <Code>K</Code> categories, encode the image and encode each class label as a text prompt (e.g., "a photo of a dog"), then assign the class whose text embedding is closest to the image embedding. No fine-tuning, no additional parameters — just nearest-neighbor search in the joint space. On ImageNet, the original CLIP ViT-L/14 achieves roughly 76% top-1 accuracy zero-shot, competitive with supervised ResNets trained explicitly on that benchmark.
      </Prose>

      <Prose>
        The limitation of CLIP is that it is a discriminative model. It produces embeddings, not text. Given an image, CLIP cannot describe it, answer questions about it, or reason over it. Extracting natural language from a CLIP vision encoder requires pairing it with a generative language model — which is exactly what the next generation of architectures does.
      </Prose>

      <H2>Frozen encoder + trainable projector — the LLaVA recipe</H2>

      <Prose>
        By 2023, two powerful pretrained components were widely available: CLIP vision encoders and instruction-tuned language models. LLaVA (Liu et al., 2023) connected them with a minimal trainable bridge: a linear projection layer (later upgraded to a two-layer MLP) that maps CLIP's vision feature dimension into the LLM's token embedding dimension. The vision encoder and the LLM are both pretrained and mostly frozen; only the projector is trained from scratch.
      </Prose>

      <Prose>
        In practice, training proceeds in two stages. Stage 1 trains only the projector on a large dataset of image-caption pairs — the LLM sees visual tokens as a prefix and learns to predict the associated caption. Stage 2 unfreezes the LLM (or a low-rank adapter on top of it) and fine-tunes on visual instruction-following data: questions, conversations, and reasoning tasks grounded in images. The vision encoder stays frozen throughout both stages.
      </Prose>

      <CodeBlock language="python">
{`import torch
import torch.nn as nn

class LLaVAStyle(nn.Module):
    def __init__(self, vision_encoder, projector, llm):
        super().__init__()
        self.vision = vision_encoder  # frozen
        self.proj = projector          # trained
        self.llm = llm                 # frozen during stage 1, unfrozen in stage 2

    def forward(self, image, text_tokens):
        # Image: (B, 3, 224, 224) -> 256 visual patch features of dim D_vision
        v = self.vision(image)                       # (B, 256, D_vision)
        v = self.proj(v)                             # (B, 256, D_llm)
        t = self.llm.embed(text_tokens)              # (B, L, D_llm)
        # Concatenate visual tokens as prefix, feed through LLM
        inputs = torch.cat([v, t], dim=1)
        return self.llm.transformer(inputs)

# Stage 1: train only the projector on image-caption pairs (cheap, fast).
# Stage 2: unfreeze LLM, fine-tune on visual instruction-following data.`}
      </CodeBlock>

      <Prose>
        The appeal of this recipe is economic. Training a projector on top of two frozen pretrained towers is orders of magnitude cheaper than training either tower from scratch. LLaVA-1.5 — which upgraded the projector to an MLP and used higher-resolution inputs — achieved competitive performance on visual question answering benchmarks with a total compute budget that a single academic lab could afford. That made the recipe immediately replicable, and it spawned a large family of derivatives: InstructBLIP, Idefics, MiniGPT-4, CogVLM, and many others.
      </Prose>

      <Prose>
        The cost is structural. The vision encoder was pretrained with CLIP's contrastive objective — optimized to produce embeddings that contrast well against text, not embeddings that help a generative model produce detailed descriptions. The LLM was pretrained on text alone. The projector is being asked to paper over a mismatch between two representations that were never designed to meet. For many tasks this works well enough. For tasks that require fine-grained spatial reasoning, precise counting, or reading small text in images, the stitching often shows.
      </Prose>

      <H3>BLIP-2's Q-Former</H3>

      <Prose>
        Between LLaVA's linear projector and fully native multimodal training sits an intermediate approach: BLIP-2's Q-Former (Li et al., 2023). Instead of projecting all <Code>N</Code> vision patch features into the LLM's token space with a simple linear map, the Q-Former inserts a small transformer between the vision encoder and the LLM. It maintains a set of <Code>M</Code> learned query embeddings — typically 32, regardless of how many patches the image produces. These queries cross-attend to the frozen vision features and self-attend among themselves, and only their final representations are passed to the LLM. The Q-Former compresses 256 patch features down to 32 query tokens.
      </Prose>

      <Prose>
        The Q-Former is trained with a three-part objective: image-text contrastive loss (to align visual and text features), image-grounded text generation loss (to make the queries extract features useful for language generation), and image-text matching loss (to distinguish matched from mismatched pairs). This richer training signal means the Q-Former learns to select the visual information that the LLM actually needs, rather than projecting everything and hoping the LLM figures it out. The tradeoff is that 32 tokens is an aggressive compression — fine-grained spatial detail that the projector might have preserved is explicitly discarded.
      </Prose>

      <H2>Native multimodal — Chameleon, Gemini, MM1</H2>

      <Prose>
        The frozen-encoder-plus-projector approach rests on a fundamental assumption: that a vision encoder trained separately, with a different objective, on different data, produces features that a language model can usefully consume after a thin learned bridge. That assumption is convenient but not obviously correct. The natural question is what happens when you remove the assumption entirely — when image tokens and text tokens are trained together, end-to-end, from the beginning.
      </Prose>

      <Prose>
        Chameleon (Meta, 2024) took this path. Images are first discretized by a VQ-GAN into sequences of integer codebook indices — the same approach described in the Multimodal Tokenization topic. Those image tokens are added to the model's vocabulary alongside text subwords, so the combined vocabulary contains both. Training is standard causal language modeling over sequences of arbitrary length that may contain image tokens, text tokens, or both. The model predicts the next token regardless of modality; there is no separate vision loss, no separate language loss, just the same cross-entropy computed uniformly over every position. The result is a model with no dedicated vision pathway at all — the transformer's residual stream carries the same kind of representation whether the current position corresponds to the word "cat" or a discretized image patch from a photo of one.
      </Prose>

      <Prose>
        Gemini 1.5 and MM1 extend this to audio and video. The sequence can interleave tokens from any modality. Training data is a mixture of text-only, image-text, video-text, and audio-text documents sampled according to a carefully tuned schedule, with multimodal data mixed in from early in training rather than grafted on later. The key structural observation is that once every modality is a token sequence, the transformer architecture requires no modification: cross-modal attention is just attention, because there is no distinction between modalities at the architecture level.
      </Prose>

      <TokenStream
        label="native multimodal training sequence"
        tokens={[
          { label: "A", color: "#e2b55a" },
          { label: " photo", color: "#e2b55a" },
          { label: " of", color: "#e2b55a" },
          { label: " a", color: "#e2b55a" },
          { label: " cat:", color: "#e2b55a" },
          { label: "<img>", color: "#c084fc" },
          { label: "vis_841", color: "#60a5fa" },
          { label: "vis_1022", color: "#60a5fa" },
          { label: "vis_2309", color: "#60a5fa" },
          { label: "vis_17", color: "#60a5fa" },
          { label: "</img>", color: "#c084fc" },
          { label: "The", color: "#e2b55a" },
          { label: " cat", color: "#e2b55a" },
          { label: " is", color: "#e2b55a" },
          { label: " gray", color: "#e2b55a" },
          { label: ".", color: "#e2b55a" },
        ]}
      />

      <Prose>
        The practical downside of native multimodal training is compute. CLIP pre-training is expensive once and reusable many times — the same CLIP ViT-L/14 sits inside dozens of deployed models. Training a native multimodal model from scratch requires sustained access to both multimodal data and substantial compute throughout the full training run. For large labs this is feasible; for smaller ones it is not. This is why the frozen-encoder recipes persist: they give smaller labs access to multimodal capability at a fraction of the compute cost, even if the resulting models have structural limitations that native training does not.
      </Prose>

      <H2>Cross-modal data curation</H2>

      <Prose>
        Text pre-training has a data quality problem — web text is noisy, repetitive, and full of low-value boilerplate. Multimodal pre-training has that problem and several additional ones. Image-text pairs scraped from the web are often misaligned in ways that text pairs are not. A text document's title usually describes the document. An image's surrounding HTML does not reliably describe the image: captions are frequently filename metadata ("IMG_4821.jpg"), product listing descriptions that apply to a category rather than the specific item pictured, SEO-optimized keyword strings that mention the image's subject tangentially, or alt-text written for accessibility without visual specificity. The caption for a photo of a golden retriever might say "pets" or "animals" or "dog photography" — each technically correct but almost useless as a training signal for the fine-grained visual features the model needs.
      </Prose>

      <Prose>
        The standard response is synthetic re-captioning. A smaller multimodal model — itself trained on higher-quality data or fine-tuned for dense captioning — is run over the image corpus to generate new, detailed captions that actually describe what is in each image. Llama 3.2 Vision, Pixtral (Mistral), and Molmo (AllenAI) all document heavy re-captioning pipelines in their technical reports. Molmo, notably, built a human annotation pipeline specifically to generate detailed spoken descriptions of images, then trained on those descriptions rather than on any scraped web data. The difference in caption quality translates directly into models that can describe spatial relationships, read text within images, and count objects more reliably than models trained on web-scraped captions.
      </Prose>

      <Prose>
        The compute economics of re-captioning are interesting. Running a vision-language model over tens of millions of images is expensive, but it is a one-time offline cost. The resulting captions can be cached and reused across many training runs. This turns data quality into an asset that compounds — each investment in better captions benefits every future model trained on that corpus. The labs that invested early in high-quality captioning pipelines hold a durable data advantage that is not easy to replicate quickly.
      </Prose>

      <Prose>
        Interleaved multimodal documents — the kind used for native multimodal training — require additional curation beyond paired captions. The training data needs coherent documents where images and text together form a meaningful unit: articles, tutorials, textbooks, scientific papers with figures, web pages with inline images and prose that references them. MMC4 (Zhu et al., 2023) and OBELICS (Laurençon et al., 2023) are the two large publicly available interleaved datasets; both apply significant filtering to remove documents where the image-text association is weak or random. The ratio of text-only to multimodal documents in the training mix matters too — too little text and the model's language capabilities degrade; too much and the visual grounding stays shallow.
      </Prose>

      <H3>Cross-modal alignment — the open problem</H3>

      <Prose>
        Even with native end-to-end training and carefully curated data, multimodal models exhibit a consistent pattern: they recognize well and reason poorly. A model that correctly identifies the breed of dog in a photograph, reads the signage in a street scene, and describes the composition of a painting will fail to count how many dogs are in the image, will misread a digit in a phone number, and will confuse left and right when describing the painting's spatial layout. The recognition-reasoning gap is not a training artifact that disappears with more scale — it persists across model sizes and architectures, narrowing slowly but never closing.
      </Prose>

      <Prose>
        The gap has a structural explanation. Visual recognition is essentially retrieval: the model has seen enough (dog, golden retriever, retriever) clusters during training that a new image activates the right representations. Visual reasoning requires operations that language reasoning also requires — counting, spatial ordering, arithmetic, logical composition — but applied to a representation that was never forced to be explicit about its spatial structure. A text transformer that reads "three dogs" has the number three sitting in a token. A vision transformer that encodes an image of three dogs has the information distributed across 196 patch embeddings in a way that the residual stream may or may not have disentangled into a discrete count. The training objective — predict the next token — rewards getting the gist right, not enumerating carefully. And getting the gist of most images requires no counting at all.
      </Prose>

      <Callout accent="gold">
        Multimodal models today recognize the world well. They reason about it poorly. Closing that gap is where the next generation of work lives.
      </Callout>

      <Prose>
        Proposed mitigations range from architectural — building explicit counting modules or spatial reasoning heads — to data-centric — constructing training sets where reasoning, not recognition, is what determines whether the answer is correct. Chain-of-thought prompting applied to visual tasks helps at inference time, and models fine-tuned on visual reasoning datasets (CLEVR, GQA, SpatialBench) improve on those benchmarks. But generalization outside the fine-tuning distribution remains limited. The benchmark scores improve; the underlying capability seems to scale more slowly than text reasoning does with model size.
      </Prose>

      <Prose>
        The alignment problem is also not symmetric. Text-to-image alignment — given a text prompt, does the generated image match? — has a different failure mode than image-to-text alignment — given an image, does the generated text accurately describe it? Models can hallucinate objects that are not in the image, omit objects that are, and confuse attributes across objects. These failures correlate weakly with the model's recognition accuracy, suggesting that the generation pathway has its own failure modes independent of the perception pathway. Measuring and closing both gaps simultaneously is an active area of work.
      </Prose>

      <Prose>
        The architectural story is mostly settled for now: native multimodal training wins where data and compute permit; frozen-encoder recipes remain the pragmatic choice for smaller labs and faster iteration. What is genuinely unsettled is whether the capability gap between recognition and reasoning closes with more scale, with better curated data, with new architectural changes we have not yet seen, or with some combination of all three. The contrastive generation — CLIP through LLaVA — delivered strong perceptual capabilities. The native multimodal generation is delivering stronger grounding and coherence. Neither has yet produced a model that reasons about visual inputs with the same fluency it brings to text. That is the open problem the next generation of multimodal work is organized around.
      </Prose>
    </div>
  ),
};

export default multimodalPretraining;
