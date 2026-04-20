import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";
import { colors } from "../../styles";

const dynamicTokenization = {
  title: "Dynamic Tokenization (ADAT, BoundlessBPE, LiteToken)",
  readTime: "9 min",
  content: () => (
    <div>
      <Prose>
        Every tokenizer this section has covered — BPE, WordPiece, Unigram, byte-level BPE,
        ViT patches, VQ-VAE codebooks, RVQ for audio — shares one property: its vocabulary is
        frozen at training time. Once the model ships, its token inventory is fixed. A sample
        that would segment better under a slightly different tokenizer has no way to ask for
        one. The merge table does not negotiate. The codebook does not grow. The patch grid does
        not rescale. What arrived in the config file is what every input gets, forever.
      </Prose>

      <Prose>
        Dynamic tokenization asks whether that has to be true. Not as an abstract engineering
        curiosity, but as a practical question: given that static vocabularies routinely
        over-fragment certain inputs and under-fragment others, is there a principled way to
        let segmentation adapt — to the domain, to the context window, to the input at hand?
        Three research directions have pushed on this frontier in the last few years, and
        none of them has crossed into production. Understanding why tells you as much about
        the state of serving infrastructure as it does about the state of tokenization.
      </Prose>

      <H2>The limits of static vocabulary</H2>

      <Prose>
        The failure modes of static tokenizers are easiest to see at the extremes. Consider a
        code-heavy sample — say, a repository README full of Python identifiers, decorators,
        and bracket-heavy syntax. A tokenizer trained on English prose will recognize common
        words like <Code>import</Code> or <Code>return</Code> as single tokens, but it will
        fragment identifiers like <Code>compute_attention_scores</Code> into a half-dozen
        pieces, cut <Code>!=</Code> and <Code>{">="}</Code> into individual characters, and
        treat every decorator symbol as a separate token. The tokens that emerge are not
        meaningless, but they are a bad match for the structure of the input. The model has
        to learn to reassemble the pieces that a better-matched vocabulary would have kept
        atomic.
      </Prose>

      <Prose>
        The second failure mode is domain shift at scale. Medical notes, legal filings, and
        chemistry SMILES strings each have a distinct sub-vocabulary: recurring drug names,
        Latin abbreviations, clause templates, molecular descriptors. A study by Kraljevic
        et al. found that a SMILES string representing a common molecule — something like
        <Code>CC(=O)Oc1ccccc1C(=O)O</Code> — tokenizes into roughly three times as many
        tokens under a general-purpose BPE vocabulary as under a chemistry-specific one.
        That 3x factor is a 3x context-window tax. It does not improve with scale. The model
        can become very good at reading over-fragmented chemistry, but it will never be as
        efficient as a model whose tokenizer spoke chemistry to begin with. Static tokenizers
        serve the average case and penalize the tails. The tails, in practice, are entire
        scientific disciplines.
      </Prose>

      <H2>ADAT — learning token boundaries</H2>

      <Prose>
        ADAT, or Adaptive Tokenization, takes the most direct route: treat segmentation as a
        learned function of the input rather than a fixed lookup. Instead of one frozen merge
        table, the tokenizer is backed by a small neural network — typically a shallow
        convolutional or recurrent encoder — that reads the input character by character and
        predicts where boundaries should fall. The boundary decisions are conditioned on local
        context, so the same three-character sequence <Code>ing</Code> might be kept attached
        to <Code>runn</Code> in prose but separated in a code identifier where it functions
        differently.
      </Prose>

      <Prose>
        The training setup is end-to-end: the segmenter sits upstream of the main model, and
        its gradients come through the downstream loss. The model's cross-entropy on the next
        token propagates back through the embedding lookup, through the boundary decisions, and
        into the segmenter's weights. Over the course of training, the segmenter learns
        boundaries that help the model, not boundaries that reflect a corpus frequency table
        computed offline. The distinction is meaningful. BPE merges what co-occurs most. ADAT
        merges what helps the model predict best — which is a subtly different thing.
      </Prose>

      <Prose>
        The costs are real. Tokenization is no longer a flat-file lookup; it now requires a
        forward pass through a neural network for every new input. Batching helps, but the
        latency floor rises. More subtly: if the segmenter uses any stochastic component —
        sampling from a boundary distribution rather than argmax-ing — the same input can
        produce different token sequences across runs. That breaks caching, breaks
        reproducibility, and complicates debugging. Most ADAT implementations in practice
        use deterministic inference (argmax segmentation), accepting the segmenter's
        stochasticity only during training where it acts as regularization. Even so, the
        compute overhead relative to a lookup table is never zero.
      </Prose>

      <H2>BoundlessBPE — vocabulary at inference</H2>

      <Prose>
        BoundlessBPE takes a different cut at the same problem. Rather than replacing the BPE
        algorithm with a neural segmenter, it keeps BPE's bottom-up merge logic but relaxes
        the constraint that the merge list must be finalized before inference. When processing
        a document, the algorithm tracks pair frequencies within the current context window.
        If a novel pair — say, a product code like <Code>TX-4891</Code> that never appeared
        in the training corpus — occurs repeatedly within a long document, BoundlessBPE adds
        a temporary merge for that pair, valid for the duration of the document. The vocabulary
        grows as text is processed and shrinks back when the document ends.
      </Prose>

      <Prose>
        This is particularly useful for long-context models where a single document introduces
        a specialized sub-vocabulary that recurs throughout. A legal contract might use the
        defendant's company name, a specific clause identifier, or a monetary abbreviation
        hundreds of times across a hundred-page document. Under a static vocabulary those
        recurrences are each re-fragmented on arrival. Under BoundlessBPE, after the first
        dozen occurrences, the name gets a temporary single token and the rest of the document
        processes it atomically. The compression savings can be substantial — and unlike
        ADAT's neural overhead, the BoundlessBPE extension is still just pair counting and
        table updates, which are cheap.
      </Prose>

      <Prose>
        The catch is context-dependency. Two users sending the same prompt to the same model
        may receive tokens with different IDs if their surrounding document contexts differ.
        Embedding tables map integer IDs to vectors; if <Code>TX-4891</Code> is token 102,543
        in one context and doesn't exist as a token in another, the model is effectively
        processing a different vocabulary at the two call sites. KV caching across requests
        becomes unsound. Batch inference assumes a shared vocabulary; BoundlessBPE breaks that
        assumption per document. These are tractable engineering problems, but they are not
        small ones.
      </Prose>

      <H3>LiteToken and compression-aware tokenization</H3>

      <Prose>
        LiteToken reframes the problem in the language of information theory. Rather than
        asking "what boundaries does the neural net prefer" or "what new merges appear in this
        document," it asks: given a fixed token budget, what segmentation best preserves
        downstream task performance? Tokenization is formalized as a rate-distortion problem —
        rate being the number of tokens consumed, distortion being the degradation in model
        output quality from coarser segmentation. The optimal tokenizer is the one that sits
        on the Pareto frontier of that curve for the content at hand.
      </Prose>

      <Prose>
        In practice, LiteToken is trained to predict segmentations that maximize compression
        while minimizing a proxy for distortion — typically the perplexity of the downstream
        language model on the segmented output. Different content types land at different
        points on the curve: scientific notation tolerates aggressive compression better than
        ambiguous proper nouns; code identifiers need to stay atomic to preserve meaning;
        common English prose is already near-optimally handled by a good static vocabulary.
        The insight LiteToken makes explicit is that "good tokenization" is not a universal
        property — it is a function of content type and task, and the right operating point
        varies. Static vocabularies implicitly pick one operating point and apply it
        everywhere. LiteToken asks you to pick per input.
      </Prose>

      <Prose>
        Walk through a small corpus below — a learned segmenter redrawing boundaries per domain.
      </Prose>

      <StepTrace
        label="adaptive boundaries shift with content"
        steps={[
          { label: "English prose", render: () => (
            <TokenStream tokens={["The", " quick", " brown", " fox", " jumps", " over"]} />
          ) },
          { label: "Python code", render: () => (
            <TokenStream tokens={["def", " compute", "(", "x", ",", " y", ")", ":", " return"]} />
          ) },
          { label: "Mixed multilingual", render: () => (
            <TokenStream tokens={["Hello", " ", "こんにちは", " ", "世界", " ", "world"]} />
          ) },
          { label: "Chemical notation", render: () => (
            <TokenStream tokens={["C", "(", "=", "O", ")", "OH", " ", "+", " ", "NH3"]} />
          ) },
        ]}
      />

      <Prose>
        A static tokenizer cannot do this — it inherits the merge table it was trained on,
        and every input is cut the same way regardless of domain.
      </Prose>

      <H2>Why this is still mostly research</H2>

      <Prose>
        The research case for adaptive tokenization is real. On domain-specific benchmarks,
        models paired with adaptive tokenizers consistently outperform identical models with
        static vocabularies, particularly in low-resource languages, scientific domains, and
        long-document settings. The results are not marginal. So why has nothing shipped?
      </Prose>

      <Prose>
        Production serving stacks are built around static vocabularies in ways that are
        difficult to overstate. The KV cache stores activations indexed by token position —
        and by implication, by token identity. A cache hit requires not just the same text
        but the same token sequence; adaptive tokenization, where the same text produces
        different token sequences depending on context, invalidates most caching strategies
        outright. Embedding tables are allocated at model-load time for a fixed vocabulary
        size; a vocabulary that grows during inference requires either pre-allocating a maximum
        that is never reached (wasteful) or dynamic memory allocation inside a hot inference
        loop (latency-hostile). Inference schedulers assume tokens are cheap to produce —
        a lookup and a matrix multiply. A neural segmenter in the tokenization stage can
        add 5–20% to total inference latency depending on implementation, a cost that
        accumulates linearly across requests at scale. None of these problems are
        theoretically unsolvable. Together, they constitute a very large implementation
        surface that benefits no paying customer today.
      </Prose>

      <Callout accent="gold">
        The question isn't whether adaptive tokenization works. It's whether the infrastructure
        can afford it.
      </Callout>

      <H2>Where it might go</H2>

      <Prose>
        Two paths forward are plausible, and they are not mutually exclusive. The first is
        specialization: adaptive tokenizers become standard equipment in verticals where the
        domain mismatch is acute enough to justify the engineering cost. Code models that
        need to treat module paths and API names atomically. Scientific LLMs processing
        molecular structure, genomic notation, or mathematical expressions. Personalized
        assistants that encounter recurring proper nouns across a long conversation.
        In each of these cases, the ROI calculation is different from general-purpose chat —
        the domain vocabulary is smaller, the efficiency gains are larger, and the user
        tolerates slightly higher latency in exchange for meaningfully better outputs.
      </Prose>

      <Prose>
        The second path is the one that has won every previous infrastructure debate in
        machine learning: static subword tokenizers remain dominant because "good enough" plus
        infrastructure compatibility beats "better" plus operational complexity. BPE survived
        not because it is theoretically optimal — it is not — but because it is fast, it is
        simple, it serializes to a flat file, and every serving framework knows how to use it.
        The same properties that made it easy to adopt in 2016 make it hard to displace in
        2026. Dynamic tokenization will have to clear a very high bar not just on model
        quality but on deployment simplicity before it finds its way into a production serving
        config.
      </Prose>

      <Prose>
        There is a tendency, when surveying a field, to treat tokenization as the preprocessing
        step you have to get through before the real work begins. This section has argued
        against that. Every topic here has shown tokenization as an architectural decision with
        compound downstream consequences: the vocabulary size shapes embedding cost; the merge
        table determines which languages pay a fertility tax; the patch grid fixes the
        resolution at which an image can be seen; the codebook sets the fidelity ceiling for
        generated audio. The models we build, the languages they serve well, the media they
        can represent, and even the pace at which we can evolve them — all of it is shaped by
        the decision about what, exactly, a token is. That decision has always deserved more
        attention than the field has given it, and the emergence of adaptive methods is the
        first serious sign that the field is starting to agree.
      </Prose>
    </div>
  ),
};

export default dynamicTokenization;
