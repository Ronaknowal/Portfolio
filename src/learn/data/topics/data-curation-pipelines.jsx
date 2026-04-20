import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const dataCurationPipelines = {
  title: "Data Curation Pipelines (Curator Models, Quality Filtering)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Deduplication is the floor. It removes redundancy — the same article scraped twice, boilerplate text duplicated across millions of pages, near-identical documents that would otherwise overweight gradient updates. But a fully deduplicated Common Crawl snapshot is still mostly garbage: ad pages, login walls, auto-generated spam, low-quality machine translations, thin affiliate content, templated product listings with the product names swapped out. Getting to a corpus worth training on requires a second, harder thing — distinguishing documents that are informative from documents that merely exist.
      </Prose>

      <Prose>
        Quality filtering is where the ceiling of a pre-training run gets set. Every frontier model's corpus is the output of a long filtering pipeline, and the quality classifier sitting at the end of that pipeline arguably matters more than any single architectural choice the team made. The evidence for this is now fairly concrete. Sub-10B parameter models trained on rigorously curated data routinely outperform 30B models trained on less-curated data across standard benchmarks. The data is doing work that additional parameters cannot compensate for. Understanding what that work looks like — mechanically, in code, in multi-stage pipelines — is what this topic covers.
      </Prose>

      <H2>What "quality" actually means</H2>

      <Prose>
        There is no single definition of quality for pre-training data. Different labs operationalize it differently, and the choice of proxy shapes the resulting corpus in ways that are not always visible until the model is evaluated.
      </Prose>

      <Prose>
        The most principled proxy is perplexity under a reference language model. If a small model trained on high-quality text — Wikipedia, digitized books, academic papers — assigns low perplexity to a document, that document is coherent and in-distribution relative to curated human writing. Documents with anomalously high perplexity are likely incoherent, machine-generated in a recognizable way, or written in a pattern the reference model never encountered. The limitation is obvious: the reference model's biases become the corpus's biases. Whatever the reference found strange gets filtered out, regardless of whether "strange" means "low quality" or merely "outside the reference distribution."
      </Prose>

      <Prose>
        A second proxy is classifier score: train a small model to distinguish curated text (Wikipedia, books, high-quality news) from average web content, then score every crawled document against that classifier and threshold. This is sometimes called "WebText-style" filtering because OpenAI's WebText dataset for GPT-2 was filtered precisely this way — pages had to be linked from Reddit with enough upvotes to suggest human judgment of quality. The classifier crystallizes that judgment and applies it at scale.
      </Prose>

      <Prose>
        A third family of proxies is heuristic: surface statistics like punctuation density, average word length, fraction of alphabetic characters, presence of certain boilerplate strings, language detection confidence. These are coarser but fast, interpretable, and robust. No single proxy is sufficient. In practice, pipelines combine all three, running cheaper passes first to reduce the volume that more expensive passes must process.
      </Prose>

      <H2>Heuristic filters — the cheap first pass</H2>

      <Prose>
        Heuristic filtering was systematized by the C4 dataset in 2020 and has since become a near-universal first stage. The core idea: documents that fail simple surface checks are almost certainly bad, and checking surfaces is cheap enough to run at corpus scale on CPU. Drop documents whose average sentence length falls outside a plausible range. Drop documents with too many symbols relative to alphabetic characters — a hallmark of scraped navigation menus, JSON blobs, and ad markup. Drop documents where language detection returns low confidence or flags a different language than expected. Drop documents containing the strings <Code>lorem ipsum</Code> or other template markers. Drop documents shorter than fifty words or longer than a hundred thousand.
      </Prose>

      <Prose>
        None of these checks require inference. They run fast, they remove the most obvious junk first, and critically, they reduce the volume that subsequent, more expensive stages have to process. A pipeline that runs heuristics before a neural classifier pays for classifier inference on a much smaller set of documents. The order is not arbitrary — it is a deliberate engineering decision about where to spend compute.
      </Prose>

      <CodeBlock language="python">
{`import re
from langdetect import detect_langs

def heuristic_filter(doc: str) -> bool:
    """Return True if doc passes cheap heuristic checks."""
    words = doc.split()
    if not (50 <= len(words) <= 100_000):
        return False
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    if not (3 <= avg_len <= 10):
        return False
    # Reject if less than 80% of characters are alphanumeric/whitespace
    alnum_ratio = sum(c.isalnum() or c.isspace() for c in doc) / max(len(doc), 1)
    if alnum_ratio < 0.8:
        return False
    # Language detection confidence threshold
    langs = detect_langs(doc[:2000])
    if not langs or langs[0].prob < 0.9:
        return False
    return True`}
      </CodeBlock>

      <Prose>
        The thresholds here are not magic numbers — they are tunable, and different pipelines use different values. The important thing is to measure what each threshold removes before committing to it. An average word length floor of 3.0 removes documents that are mostly punctuation or abbreviations; raising it to 4.5 starts clipping legitimate technical writing where short variable names and acronyms are common. Heuristics encode implicit assumptions about what text looks like, and those assumptions should be tested against random samples from the corpus, not just accepted as defaults.
      </Prose>

      <H2>Classifier-based filtering</H2>

      <Prose>
        After heuristics, the documents that remain are syntactically coherent but may still be semantically thin. This is where a trained quality classifier earns its place. The standard approach, introduced prominently with GPT-3's training pipeline and replicated across open successors, is to train a small binary classifier — often a fastText model or a tiny fine-tuned BERT — on two classes: documents from curated, high-quality sources on one side, and documents sampled from the raw web on the other. The classifier learns to score documents on a continuous axis from "looks like Wikipedia" to "looks like average crawl." Documents below a threshold are discarded.
      </Prose>

      <Prose>
        The key design choice is what you use as the positive class. Using Wikipedia produces a classifier that rewards formal, encyclopedic prose and penalizes conversational or technical writing that happens to be good. The Phi series from Microsoft pushed this further: they found that using "educational content" as the positive class, rather than Wikipedia generically, produced much sharper discrimination between documents that teach something and documents that merely describe. FineWeb-Edu (HuggingFace, 2024) formalized this into a five-point scale — "how educational is this document?" — and showed that training on documents scoring 3 or above on that scale produced measurably better models on reasoning and knowledge benchmarks than training on a Wikipedia-positive classifier's output.
      </Prose>

      <Prose>
        The weakness of classifier-based filtering is that the classifier reflects the biases of whoever defined the positive class. A Wikipedia-positive classifier systematically undervalues domains not well-represented in Wikipedia: programming, legal writing, medical documentation, colloquial speech. A classifier that treats educational value as the criterion undervalues narrative fiction and cultural writing. These are not hypothetical concerns — they show up in the model's downstream behavior, in what it does and does not know, in what registers it handles fluently.
      </Prose>

      <H2>Curator models — LLMs as judges</H2>

      <Prose>
        The newest layer in quality pipelines replaces the human-curated training signal for the classifier with a stronger model's judgment. The loop is straightforward: sample a small fraction of the full corpus — a few hundred thousand documents — and run them through a powerful LLM with a detailed quality rubric. The LLM annotates each document. Those annotations become the training labels for a fast, small classifier that can then score the entire corpus at a fraction of the cost. The LLM provides quality; the small classifier provides scale.
      </Prose>

      <Prose>
        Phi-3 used this approach, training a classifier on quality judgments from a large teacher model. FineWeb-Edu trained its educational-quality scorer on Llama-3-70B annotations of 500,000 documents, using prompts asking the model to rate how valuable the document would be as educational material for a student. Nemotron-CC used a similar technique to produce the Nemotron dataset, demonstrating that the approach generalizes beyond educational content to other quality dimensions. In each case the same structure appears: an expensive teacher that labels a sample, a cheap student that generalizes those labels to billions of documents.
      </Prose>

      <StepTrace
        label="curator-model pipeline — bootstrapping a fast quality classifier"
        steps={[
          { label: "1. sample", render: () => (
            <TokenStream tokens={["500B doc corpus", " →", " sample 500K"]} />
          ) },
          { label: "2. llm annotate", render: () => (
            <TokenStream tokens={["500K docs", " →", " Llama-3-70B rates quality 1-5", " →", " 500K labeled"]} />
          ) },
          { label: "3. train classifier", render: () => (
            <TokenStream tokens={["500K labels", " →", " train 350M-param BERT classifier", " →", " quality score head"]} />
          ) },
          { label: "4. filter corpus", render: () => (
            <TokenStream tokens={["500B docs", " →", " classifier scores all", " →", " keep top 20%", " →", " 100B curated"]} />
          ) },
        ]}
      />

      <Prose>
        The economics of this loop are what make it viable. Running Llama-3-70B inference on 500,000 documents is expensive but tractable — a few hundred GPU-hours at most. Training a 350M-parameter BERT classifier on those 500,000 labels takes an afternoon. Running that classifier over 500 billion documents at batch inference speeds is fast. The expensive LLM touches less than 0.1% of the corpus; its judgment is amortized over everything else through the classifier's generalization. The resulting annotations are noisier than a human expert would produce and noisier than running the LLM on every document — but they are far better than heuristics alone, and they are at-scale.
      </Prose>

      <H3>The FineWeb recipe</H3>

      <Prose>
        HuggingFace's FineWeb (2024) is the most thoroughly documented open curation pipeline to date, and it is worth walking through as a concrete example of how the stages compose. The full pipeline runs in order: URL filtering against a domain reputation list removes known spam and adult-content domains before any document is read. Language identification runs next, keeping only documents with high confidence English detection. Deduplication — URL-level Bloom filtering followed by MinHash near-duplicate removal — eliminates redundancy. C4-style heuristic filters remove documents that fail surface checks. A perplexity filter using a small reference language model removes incoherent or templated text. Finally, the FineWeb-Edu classifier, trained on Llama-3-70B annotations, scores every remaining document for educational value; only documents scoring 3 or above on the 0–5 scale are retained.
      </Prose>

      <Prose>
        Each stage is aggressive. URL filtering discards a meaningful fraction of crawled pages before any processing. Language identification cuts non-English content. Dedup removes between 30 and 70 percent of what remains, depending on the crawl. Heuristics remove another significant slice. By the time the neural classifier runs, the corpus is already a small fraction of the original crawl. The classifier's final pass retains roughly the top 20 percent of its inputs. End to end, the curated FineWeb corpus is approximately one-twentieth of the raw Common Crawl it was derived from — by document count. The per-document cost of having gone through the pipeline is paid once; the benefit compounds across every training step the model takes on that data.
      </Prose>

      <H3>Data-quality scaling laws</H3>

      <Prose>
        Recent empirical work has started to quantify the tradeoff between data quality and data quantity in formal scaling-law terms. Penedo et al. (2024) and Sachdeva et al. (2024) both find that for a fixed compute budget, tightening the quality threshold can outperform simply adding more data. At least on the benchmarks they study, doubling the quality bar is worth more than doubling the token count. The intuition is clean: if the marginal document added by relaxing the quality threshold carries mostly noise, gradient descent wastes steps reinforcing that noise. A smaller corpus of higher-quality documents can produce a model that generalizes better on tasks that require reasoning and factual knowledge.
      </Prose>

      <Prose>
        This is not a stable or universal finding. Some benchmarks reward breadth — coverage of obscure domains, rare language patterns, unusual facts — and broad, lightly filtered corpora do better on them. The quality-quantity tradeoff looks different depending on what you are optimizing for, and it changes with model scale. But the finding has shifted the prevailing practice: serious pre-training teams now treat the quality filtering pipeline as a first-class engineering investment, not an afterthought. Spending compute on curation infrastructure competes favorably with spending it on longer training runs.
      </Prose>

      <H2>What can go wrong</H2>

      <Prose>
        Over-filtering is a real failure mode. Push the quality threshold high enough and the resulting corpus becomes narrowly homogeneous: dense with formal prose, thin on colloquial language, technical jargon, regional dialects, spoken-word transcriptions, and any register that does not resemble the positive class the classifier was trained on. A model trained exclusively on Wikipedia-like text handles encyclopedic queries fluently and struggles with casual conversation, code comments, tweets, and the kind of loose, associative writing that makes up a large fraction of how people actually communicate. The filtering succeeded at its stated goal and failed at the implicit one.
      </Prose>

      <Prose>
        Quality classifiers trained primarily on English text import English-language biases into the global corpus. A document in Hindi or Swahili that would be judged high quality by a native-speaker classifier may score poorly against an English-trained model's notion of quality — not because the document is bad, but because its vocabulary, sentence structure, and stylistic conventions are outside the classifier's training distribution. Applying such a classifier globally narrows the non-English portion of the corpus more aggressively than the English portion, which compounds the fertility imbalances that multilingual tokenization already introduces.
      </Prose>

      <Callout accent="gold">
        A quality classifier is a model of what quality looks like. Filtering makes your corpus smarter about what that model already values — and dumber about everything else.
      </Callout>

      <Prose>
        Curator models carry a related problem. A Llama-3-70B annotator reflects what Llama-3-70B was trained to value: the text patterns that received high reward in its own training process, the stylistic conventions of its positive-class data, the subjects and formats it was reinforced to prefer. Training a quality classifier on its annotations and filtering with that classifier creates a loop where the student model's corpus is shaped by the teacher model's priors. If the teacher undervalues certain kinds of valid writing, those kinds disappear from the student's training data. The bias is not random — it is systematic, in the direction of whatever the teacher already knew.
      </Prose>

      <Prose>
        None of these failure modes disqualify the tools. They argue for auditing: sampling documents at each filter stage and reading them, measuring downstream model behavior across different domains and user populations, and calibrating thresholds against held-out evaluations rather than setting them once and forgetting them. The pipeline is not a one-time setup. It is infrastructure that requires the same maintenance attention as any other production system.
      </Prose>

      <Prose>
        The boring truth about pre-training data is that "better data" is mostly a matter of larger, smarter filter stacks — not secret datasets no one else has access to. Common Crawl is public. Wikipedia is public. The gap between a team that wins at pre-training and a team that does not is usually the gap between their curation infrastructure: how many filter stages they built, how carefully they calibrated each one, how much engineering they spent making the pipeline fast enough to iterate on. The ceiling is set before the first gradient step, and it is set by the data.
      </Prose>
    </div>
  ),
};

export default dataCurationPipelines;
