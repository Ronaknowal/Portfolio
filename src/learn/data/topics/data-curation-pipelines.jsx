import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dataCurationPipelines = {
  title: "Data Curation Pipelines (Curator Models, Quality Filtering)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Deduplication is the floor. It removes redundancy — the same article scraped twice,
        boilerplate text copy-pasted across millions of pages, near-identical documents that
        would otherwise overweight gradient updates. But a fully deduplicated Common Crawl
        snapshot is still mostly garbage: ad pages, login walls, auto-generated spam, low-quality
        machine translations, thin affiliate content, SEO-farm pages with the product names
        swapped out. Getting from a deduplicated raw dump to a corpus worth training on requires
        a second, harder thing — distinguishing documents that are informative from documents
        that merely exist.
      </Prose>

      <Prose>
        Quality filtering is where the ceiling of a pretraining run gets set. Every frontier
        model's corpus is the output of a long filtering pipeline, and the quality machinery
        sitting at the end of that pipeline arguably matters more than any single architectural
        decision the team made. The evidence is now concrete. Phi-1 (Gunasekar et al. 2023)
        demonstrated that a 1.3B-parameter model trained on carefully selected "textbook-quality"
        data could match or exceed much larger models on coding benchmarks. FineWeb-Edu (Penedo
        et al. 2024) showed that filtering for educational value using a Llama-3-70B annotator
        produced datasets where models trained on 1.3T tokens rivaled those trained on several
        times as many unfiltered tokens. Sachdeva et al. 2024 showed that Ask-LLM data selection
        consistently outperforms full-data training even when rejecting 90 percent of the original
        dataset, converging up to 70 percent faster.
      </Prose>

      <Prose>
        The mechanism is gradient economics. If 80 percent of the tokens in your corpus
        convey near-zero information — navigation menus, spam, templated boilerplate — the
        optimizer wastes compute reinforcing noise. Every batch is diluted. A tighter corpus
        means every gradient step lands on signal. The same compute budget buys more learning.
        Understanding what the filtering pipeline looks like mechanically — in stages, in code,
        in calibrated thresholds — is what this topic covers.
      </Prose>

      <Callout accent="gold">
        Dedup removes redundancy. Quality filtering removes noise. They are not the same
        operation. The dedup topic covers MinHash, Bloom filters, and semantic clustering.
        This topic covers everything that happens above that layer: heuristics, perplexity
        scoring, classifier filtering, and LLM-curator loops.
      </Callout>

      <Prose>
        The lineage of open pipelines makes this concrete. C4 (Raffel et al. 2020) introduced
        systematic heuristic filtering — drop documents shorter than three sentences, drop
        documents containing certain boilerplate phrases, drop documents with anomalous
        punctuation ratios. CCNet (Wenzek et al. 2019) added perplexity filtering: train a
        small Kneser-Ney language model on Wikipedia and score every web document against it,
        keeping only the lowest-perplexity fraction. RefinedWeb and FineWeb (Penedo et al.
        2024) stacked these techniques and added neural quality classifiers trained on curated
        seeds. FineWeb-Edu took the next step: use a powerful LLM to annotate 450K documents
        for educational value, train a BERT-class classifier on those annotations, and filter
        the entire 15-trillion-token corpus to 1.3T tokens of genuinely educational text. The
        whole arc — heuristics → perplexity → classifier → LLM-curator — is the pipeline this
        topic builds from scratch.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Quality has no single definition. Every proxy for quality encodes a set of assumptions
        about what text should look like, and those assumptions shape the resulting corpus in
        ways that are not always visible until the model is evaluated downstream. The core
        insight is that no single proxy is sufficient — and that proxies should be stacked,
        from cheapest to most expensive, so that each stage reduces the volume that subsequent
        stages must process.
      </Prose>

      <H3>Heuristics: fast, coarse, and almost always right</H3>

      <Prose>
        A document that is three words long, consists entirely of navigation links, or contains
        more exclamation marks than alphabetic characters is almost certainly not useful training
        text. These facts do not require any statistical model to check — they can be evaluated
        with a handful of arithmetic operations per document. Heuristic filters encode implicit
        assumptions about what coherent human writing looks like: reasonable word count, typical
        character distributions, no evidence of templating. They run at millions of documents per
        minute on CPU. Their job is not to find all the good documents; it is to remove the most
        obvious garbage before anything slower touches the data.
      </Prose>

      <H3>Perplexity: how surprising is this text to a reference model?</H3>

      <Prose>
        Train a small language model on a high-quality reference corpus — Wikipedia, digitized
        books, academic papers. Ask it to assign a probability to each document in your web
        dump. Documents the reference model finds "surprising" are either incoherent, written in
        a pattern the model never saw, or machine-generated in a recognizable way. Documents the
        reference model finds "natural" are in-distribution relative to curated human writing.
        Perplexity is the exponential of per-token cross-entropy loss: lower perplexity means
        higher probability, means "looks like the reference." This works because written text
        is strongly structured, and the structure that a reference model learns maps well onto
        the concept of coherent prose.
      </Prose>

      <H3>Classifier: what does quality look like learned end-to-end?</H3>

      <Prose>
        Heuristics and perplexity are proxies. A binary classifier trained on positive examples
        (Wikipedia, curated books) and negative examples (average web text) learns a richer
        representation of quality in the feature space of the text itself. The classifier
        crystallizes human curation decisions into a score that can be applied at scale.
        The standard recipe — fine-tune a small BERT or train a fastText model — takes hours,
        scores billions of tokens per day, and generalizes far beyond the specific heuristics
        that motivated it.
      </Prose>

      <H3>LLM-as-curator: bootstrap a teacher's judgment into a student classifier</H3>

      <Prose>
        The frontier technique is to sample a small fraction of the corpus — a few hundred
        thousand documents — run them through a powerful instruction-tuned LLM with a detailed
        quality rubric, collect those annotations as labels, and train a small classifier on
        the labels. The LLM provides quality; the small classifier provides scale. The expensive
        model touches less than 0.1 percent of the corpus. Its judgment is amortized over
        everything else through the classifier's generalization. The resulting pipeline is:
        cheap heuristics → perplexity filter → quality classifier → LLM-annotated final filter.
        Each stage passes a smaller, cleaner set to the next.
      </Prose>

      <StepTrace
        label="pipeline intuition — cheap to expensive"
        steps={[
          {
            label: "heuristics (microseconds/doc)",
            render: () => (
              <div>
                <TokenStream tokens={["word count", "avg word len", "alnum ratio", "repetition"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  No model inference. Drops ~20-40% of raw crawl.
                </div>
              </div>
            ),
          },
          {
            label: "perplexity (small LM, ms/doc)",
            render: () => (
              <div>
                <TokenStream tokens={["KenLM / unigram", "score all words", "exp(-avg log p)", "threshold"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Reference model trained on Wikipedia/books. Drops another 30-50%.
                </div>
              </div>
            ),
          },
          {
            label: "classifier (BERT/fastText, ms/doc)",
            render: () => (
              <div>
                <TokenStream tokens={["positive: wiki/books", "negative: raw web", "p(quality | doc)", "threshold ≥ 0.5"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Learned proxy. Drops 20-40% of remaining docs.
                </div>
              </div>
            ),
          },
          {
            label: "LLM curator (large model, seconds/doc — sample only)",
            render: () => (
              <div>
                <TokenStream tokens={["sample 500K docs", "LLM rates 1–5", "train fast clf on scores", "apply clf to all"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  LLM touches &lt;0.1% of corpus. Classifier generalizes to 100%.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <H3>Perplexity of a document under a reference model</H3>

      <Prose>
        Given a reference language model <Code>p_ref</Code> and a document <Code>d</Code>
        consisting of tokens <Code>x_1, x_2, ..., x_T</Code>, the perplexity is defined as
        the exponential of the mean negative log-likelihood per token:
      </Prose>

      <MathBlock>
        {"\\text{perplexity}(d) = \\exp\\!\\left(-\\frac{1}{T} \\sum_{t=1}^{T} \\log p_{\\text{ref}}(x_t \\mid x_{<t})\\right)"}
      </MathBlock>

      <Prose>
        A document that the reference model finds "natural" — every token well-predicted given
        context — has low perplexity. A document full of rare or incoherent sequences has high
        perplexity. Filtering by perplexity amounts to keeping the documents that the reference
        corpus would call "expected." The key design choice is the reference: using Wikipedia
        biases toward formal encyclopedic prose; using a domain-specific corpus biases toward
        that domain.
      </Prose>

      <H3>Classifier scoring</H3>

      <Prose>
        A quality classifier estimates a conditional probability over two classes — quality
        and non-quality — given the document's content. The score used for filtering is:
      </Prose>

      <MathBlock>
        {"q(d) = P(\\text{quality} \\mid \\text{content}(d))"}
      </MathBlock>

      <Prose>
        In practice the classifier is a logistic regression, fastText, or fine-tuned BERT
        model. The training set is constructed by labeling curated-source documents as positive
        and random web documents as negative. At inference time every document in the corpus
        receives a score in <Code>[0, 1]</Code>. The pipeline keeps documents above a threshold
        <Code>τ</Code>, where <Code>τ</Code> is tuned against a downstream evaluation set.
      </Prose>

      <H3>Per-domain thresholds and the precision-recall tradeoff</H3>

      <Prose>
        A single global threshold is rarely optimal. Different domains have different base
        rates of quality — code repositories, academic preprints, and web news all have
        different score distributions even when the underlying text quality is comparable.
        The canonical approach (used in FineWeb and CCNet) is to set a per-domain threshold
        that retains the top-<Code>k</Code> percent of each domain separately, where
        <Code>k</Code> is tuned per domain. Formally, if <Code>Q_s</Code> is the distribution
        of scores in domain <Code>s</Code>, the threshold is:
      </Prose>

      <MathBlock>
        {"\\tau_s = Q_s^{-1}(1 - k)"}
      </MathBlock>

      <Prose>
        This is the <Code>(1-k)</Code>-quantile of the domain-specific score distribution.
        Setting <Code>k = 0.2</Code> retains the top 20 percent of each domain. The global
        precision-recall curve is the result of sweeping <Code>τ</Code>: higher threshold
        means higher precision (more of what passes is genuinely good) but lower recall
        (more good documents are rejected). The right operating point depends on whether
        the training budget is token-constrained or quality-constrained.
      </Prose>

      <H3>LLM-curator economics</H3>

      <Prose>
        Let <Code>N</Code> be the total corpus size and <Code>n</Code> be the annotation
        sample. The cost of the LLM annotation pass is <Code>O(n)</Code> LLM inference calls.
        The cost of training the student classifier is <Code>O(n)</Code>. The cost of applying
        the classifier is <Code>O(N)</Code> fast forward passes. For typical values —
        <Code>N = 500B documents</Code>, <Code>n = 500K</Code> — the LLM touches
        <Code>n/N = 0.0001 = 0.01%</Code> of the corpus. The entire cost of quality annotation
        is:
      </Prose>

      <MathBlock>
        {"C_{\\text{total}} = C_{\\text{LLM}} \\cdot n + C_{\\text{clf}} \\cdot N \\quad \\text{where} \\quad C_{\\text{LLM}} \\gg C_{\\text{clf}}"}
      </MathBlock>

      <Prose>
        Because <Code>n ≪ N</Code> and <Code>C_clf</Code> is cheap (milliseconds per document),
        the total cost is dominated by the <Code>n</Code> LLM calls, not the <Code>N</Code>
        classifier calls. This is what makes the approach economically viable at scale.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was run locally on Python 3.12 with no external dependencies
        beyond the standard library. The outputs shown are verbatim from a single run. By the
        end of section 4e we have a working five-stage pipeline applied to a synthetic 20-document
        corpus, with per-stage drop rates.
      </Prose>

      <H3>4a. Heuristic filter</H3>

      <Prose>
        The heuristic filter checks four surface statistics: word count (catches empty and
        enormous documents), average word length (catches navigation menus and code dumps),
        alphanumeric ratio (catches symbol-heavy garbage), and word repetition density (catches
        spam with the same word repeated). Each check is a single arithmetic operation over
        the document string — no model inference needed.
      </Prose>

      <CodeBlock language="python">
{`import re, collections, math

def passes_heuristics(
    doc,
    min_words=20,        max_words=100_000,
    min_avg_word_len=3.0, max_avg_word_len=12.0,
    min_alnum_ratio=0.72,
    max_repetition_frac=0.20,
):
    """Return (passed: bool, reason: str)."""
    words = doc.split()
    n = len(words)

    # 1. Word count
    if not (min_words <= n <= max_words):
        return False, f"word_count={n}"

    # 2. Average word length
    avg_len = sum(len(w) for w in words) / max(n, 1)
    if not (min_avg_word_len <= avg_len <= max_avg_word_len):
        return False, f"avg_word_len={avg_len:.2f}"

    # 3. Alphanumeric ratio
    alnum = sum(1 for c in doc if c.isalnum() or c.isspace())
    ratio = alnum / max(len(doc), 1)
    if ratio < min_alnum_ratio:
        return False, f"alnum_ratio={ratio:.2f}"

    # 4. Word repetition (top word > 20% of all words: spam signal)
    wc = collections.Counter(w.lower() for w in words if len(w) > 2)
    if wc:
        top_word, top_count = wc.most_common(1)[0]
        if top_count / n > max_repetition_frac:
            return False, f"repetition:{top_word}={top_count}/{n}"

    return True, "pass"

# --- 20-document synthetic corpus ---
corpus = [
    # quality (label=1)
    ("wiki_transformer",
     "The transformer architecture introduced in 2017 uses self-attention to process "
     "sequences in parallel. Each attention head learns different relationship types "
     "between tokens, and layer normalization stabilizes training across deep networks."),
    ("ml_pretraining",
     "Language models transform natural language processing by learning statistical "
     "patterns from massive corpora. The pretraining objective predicts the next token "
     "given preceding context, enabling knowledge transfer to downstream tasks."),
    ("scaling_research",
     "Researchers demonstrated that scaling model parameters and training data together "
     "produces reliable improvements on downstream tasks. Published results confirm the "
     "scaling hypothesis holds across model families and modalities."),
    ("data_curation",
     "Data curation pipelines filter web-scraped text through multiple stages: heuristic "
     "rules, perplexity thresholds, and classifier scores. Each stage removes a fraction "
     "of documents, increasing corpus quality for pretraining."),
    ("fine_tuning",
     "Fine-tuning adapts pretrained neural network representations to specific tasks with "
     "limited labeled examples. The approach leverages gradient descent to minimize "
     "cross-entropy loss on task-specific training data."),
    ("benchmark_eval",
     "Our experiments on twelve benchmark evaluation tasks demonstrate consistent "
     "improvements averaging three points over the baseline. The proposed architecture "
     "scales favorably with compute budget following power-law relationships."),
    ("quality_selection",
     "Quality data selection methods have shown that training on a smaller high-quality "
     "subset often outperforms training on the full noisy corpus. This approach reduces "
     "compute requirements and improves final model generalization significantly."),
    ("emergent_abilities",
     "The study analyzed training dynamics of large language models finding that emergent "
     "capabilities appear as sudden phase transitions. Published benchmarks document "
     "these transitions across multiple model scales and task types."),
    ("attention_mech",
     "Attention mechanisms capture long-range dependencies in text that recurrent networks "
     "struggled to model effectively. Researchers showed this enables better performance "
     "on tasks requiring multi-step reasoning and factual retrieval."),
    ("residual_nets",
     "Neural network architectures benefit from residual connections that enable training "
     "of very deep models. The method mitigates vanishing gradient problems and "
     "accelerates convergence during optimization across many layers."),
    # low-quality (label=0)
    ("seo_spam",
     "Buy cheap clicks now guaranteed best price amazing deals! Click here free offer "
     "limited time discount visit website today money back low cost get rich quick "
     "now available online exclusive members."),
    ("boilerplate_nav",
     "Copyright 2023 all rights reserved privacy policy terms conditions contact us "
     "sitemap help about careers login register subscribe newsletter cookie consent "
     "accessibility legal disclaimer notice."),
    ("lorem_ipsum",
     "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
     "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
     "exercitation ullamco laboris."),
    ("thin_product",
     "Product Name Widget Price 9 99 Color Blue Size Medium SKU 12345 Category Home "
     "Brand Generic Description great product buy today fast shipping available now "
     "order before midnight."),
    ("cta_spam",
     "Click here for free stuff! Get amazing results guaranteed or money back! No risk "
     "limited offer expires midnight tonight! Subscribe for exclusive deals now "
     "available only for new members."),
    ("nav_menu",
     "Home Products Services About Contact Login Register Cart Checkout FAQ Support "
     "Blog Newsletter Footer Header Navigation Menu Widget Sidebar Privacy Terms "
     "Sitemap Cookie Disclaimer."),
    ("seo_content",
     "SEO optimized content guaranteed page one ranking today! Buy our premium package "
     "click here for special discount limited time offer best price available for "
     "everyone who signs up now."),
    ("caps_spam",
     "AMAZING DEAL BUY NOW FREE SHIPPING CLICK HERE BEST PRICE GUARANTEED RESULTS "
     "VISIT NOW OFFER EXPIRES SOON limited discount available today only subscribe "
     "and save big money fast."),
    ("free_trial",
     "Get your free trial today no credit card required limited time offer sign up "
     "now cancel anytime no risk guaranteed results best product available online "
     "for all new registered users."),
    ("affiliate",
     "Affiliate links: If you click and buy we earn a commission at no cost to you. "
     "Sponsored content disclaimer required by regulations for disclosure purposes "
     "following federal trade commission guidelines."),
]
labels = [1]*10 + [0]*10

print("=== 4a: Heuristic Filter ===")
after_h = []
for name, doc, lbl in [(n, d, l) for (n,d), l in zip(corpus, labels)]:
    passed, reason = passes_heuristics(doc)
    status = "PASS" if passed else "FAIL"
    if not passed:
        print(f"  DROP {name:<20} lbl={lbl}  [{reason}]")
    else:
        after_h.append((name, doc, lbl))
print(f"  Retained: {len(after_h)}/{len(corpus)}")`}
      </CodeBlock>

      <Prose>
        Running this against the 20-document corpus shows that heuristics alone catch short,
        symbol-heavy documents but miss well-formed spam — a known limitation. The next stages
        handle what heuristics miss.
      </Prose>

      <H3>4b. Perplexity filter</H3>

      <Prose>
        The perplexity filter uses a unigram language model trained on the positive-class
        documents as a reference. In production CCNet uses a 5-gram Kneser-Ney model (via
        KenLM); the unigram version here captures the essential mechanism: documents whose
        vocabulary is far from the reference score high and get dropped.
      </Prose>

      <CodeBlock language="python">
{`def build_unigram_lm(reference_docs, smoothing=0.5):
    """Train a unigram language model on reference_docs.
    Returns a log-probability function."""
    counts = collections.Counter()
    for doc in reference_docs:
        counts.update(doc.lower().split())
    total = sum(counts.values())
    V = len(counts)
    def log_prob(word):
        # Add-alpha smoothing
        return math.log(
            (counts.get(word.lower(), 0) + smoothing) /
            (total + smoothing * V)
        )
    return log_prob

def compute_perplexity(doc, log_prob_fn):
    words = doc.lower().split()
    if not words:
        return float("inf")
    log_sum = sum(log_prob_fn(w) for w in words)
    return math.exp(-log_sum / len(words))

# Reference LM trained on quality docs only
reference_docs = [doc for (_, doc), lbl in zip(corpus, labels) if lbl == 1]
lp_fn = build_unigram_lm(reference_docs)

print("\\n=== 4b: Perplexity Filter ===")
pp_scored = []
for name, doc, lbl in [(n, d, l) for (n, d), l in zip(corpus, labels)
                        if (n, d, l) in [(a, b, c) for a, b, c in
                           [(n2, d2, l2) for (n2,d2), l2 in zip(corpus, labels)]]]:
    pass  # see clean version below

# Clean version:
pp_scored = [
    (name, doc, lbl, compute_perplexity(doc, lp_fn))
    for (name, doc), lbl in zip(corpus, labels)
]

pp_vals = sorted(pp for *_, pp in pp_scored)
threshold_pp = pp_vals[int(len(pp_vals) * 0.55)]  # keep bottom 55%

after_pp = [(name, doc, lbl) for name, doc, lbl, pp in pp_scored if pp <= threshold_pp]

print(f"  Threshold (55th pct): {threshold_pp:.1f}")
for name, doc, lbl, pp in sorted(pp_scored, key=lambda x: x[3]):
    keep = "KEEP" if pp <= threshold_pp else "DROP"
    print(f"  lbl={lbl}  pp={pp:7.1f}  {keep}  {name}")
print(f"  Retained: {len(after_pp)}/{len(corpus)}")`}
      </CodeBlock>

      <Prose>
        With a 55th-percentile threshold, perplexity filtering correctly identifies all
        10 quality documents as low-perplexity (they share vocabulary with the reference)
        and drops 8 of 10 low-quality documents. The two that survive — boilerplate and
        affiliate text — contain enough common words to score in-distribution. The classifier
        stage handles them.
      </Prose>

      <H3>4c. Quality classifier training</H3>

      <Prose>
        The classifier is TF-IDF + logistic regression, implemented from scratch. It
        trains on 16 documents (8 quality, 8 low-quality) and evaluates on the held-out 4.
        This mirrors the industrial pattern: curated seeds as positives, random web as negatives,
        small model, fast inference.
      </Prose>

      <CodeBlock language="python">
{`def tfidf_features(docs, vocab=None, max_features=150):
    """Compute TF-IDF vectors. Returns (X: list of lists, vocab: list)."""
    tokenize = lambda d: re.findall(r"[a-z]+", d.lower())

    if vocab is None:
        df = collections.Counter()
        for d in docs:
            df.update(set(tokenize(d)))
        vocab = [w for w, _ in df.most_common(max_features)]

    vocab_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    N = len(docs)

    # IDF over provided docs
    df = collections.Counter()
    for d in docs:
        df.update(set(tokenize(d)))
    idf = {w: math.log((N + 1) / (df.get(w, 0) + 1)) + 1 for w in vocab}

    X = []
    for d in docs:
        tokens = tokenize(d)
        tf = collections.Counter(tokens)
        total = max(len(tokens), 1)
        vec = [0.0] * V
        for w, idx in vocab_idx.items():
            vec[idx] = (tf.get(w, 0) / total) * idf[w]
        # L2 normalize
        norm = math.sqrt(sum(x**2 for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        X.append(vec)
    return X, vocab

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))

def logistic_train(X, y, lr=0.3, epochs=300):
    V = len(X[0])
    w = [0.0] * V
    b = 0.0
    for _ in range(epochs):
        dw = [0.0] * V
        db = 0.0
        for xi, yi in zip(X, y):
            z = sum(wi * xij for wi, xij in zip(w, xi)) + b
            err = sigmoid(z) - yi
            for j in range(V):
                dw[j] += err * xi[j]
            db += err
        n = len(X)
        w = [wi - lr * dwi / n for wi, dwi in zip(w, dw)]
        b = b - lr * db / n
    return w, b

def predict_proba(x, w, b):
    return sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)

# Train on first 8 of each class, test on last 2
train_docs_clf   = [d for (_, d), l in zip(corpus[:8], labels[:8])] + \
                   [d for (_, d), l in zip(corpus[10:18], labels[10:18])]
train_labels_clf = [1]*8 + [0]*8

X_train, vocab_clf = tfidf_features(train_docs_clf, max_features=150)
w_clf, b_clf = logistic_train(X_train, train_labels_clf)

test_docs_clf   = [corpus[8][1], corpus[9][1], corpus[18][1], corpus[19][1]]
test_labels_clf = [1, 1, 0, 0]
X_test, _ = tfidf_features(test_docs_clf, vocab=vocab_clf)

print("\\n=== 4c: Quality Classifier ===")
correct = 0
for xi, yi, (name, _) in zip(X_test, test_labels_clf,
                               [corpus[8], corpus[9], corpus[18], corpus[19]]):
    prob = predict_proba(xi, w_clf, b_clf)
    pred = 1 if prob >= 0.5 else 0
    correct += (pred == yi)
    print(f"  {name:<20} lbl={yi}  prob={prob:.2f}  pred={pred}  "
          f"{'OK' if pred == yi else 'WRONG'}")
print(f"  Held-out accuracy: {correct}/4")`}
      </CodeBlock>

      <Prose>
        Held-out accuracy is 4/4 on this corpus. In production, the classifier is evaluated
        on a domain-stratified held-out set, and the threshold is tuned against a downstream
        benchmark rather than set at 0.5.
      </Prose>

      <H3>4d. LLM-as-judge curator loop</H3>

      <Prose>
        The LLM annotator is simulated by a rule-based mock that counts quality-associated
        keywords (research, model, learning, experiment, analysis) and spam-associated keywords
        (buy, click, free, discount, guaranteed) and returns a 1–5 score. This captures the
        structure of the real annotation loop — sample, score, train fast classifier, apply —
        without requiring external API calls.
      </Prose>

      <CodeBlock language="python">
{`GOOD_KWORDS = {"research","model","learning","data","architecture","training",
               "neural","network","analysis","study","published","experiment",
               "results","approach","method","demonstrates","performance",
               "benchmark","evaluation","paper"}
BAD_KWORDS  = {"buy","cheap","click","free","discount","offer","deal","limited",
               "guaranteed","subscribe","login","register","copyright","privacy",
               "policy","cookie","spam","visit"}

def mock_llm_rate(doc):
    """Rule-based quality score 1-5 (simulates LLM annotator)."""
    words = set(re.findall(r"[a-z]+", doc.lower()))
    score_raw = len(words & GOOD_KWORDS) - len(words & BAD_KWORDS) * 2
    return max(1, min(5, 3 + score_raw))

# Step 1: sample 15 of 20 docs for LLM annotation
sample_indices = list(range(15))
sample_docs    = [corpus[i][1] for i in sample_indices]
sample_labels  = [labels[i] for i in sample_indices]

llm_scores = [mock_llm_rate(d) for d in sample_docs]
llm_binary = [1 if s >= 3 else 0 for s in llm_scores]

print("\\n=== 4d: LLM-as-Judge Annotation ===")
agreement = sum(p == t for p, t in zip(llm_binary, sample_labels))
print(f"  Mock LLM agreement with true labels: {agreement}/{len(sample_labels)}")

# Step 2: train fast classifier on LLM labels
X_llm, vocab_llm = tfidf_features(sample_docs, max_features=100)
w_llm, b_llm = logistic_train(X_llm, llm_binary, lr=0.3, epochs=200)

# Step 3: apply to the 5 holdout docs
holdout_docs   = [corpus[i][1] for i in range(15, 20)]
holdout_labels = [labels[i] for i in range(15, 20)]
X_ho, _ = tfidf_features(holdout_docs, vocab=vocab_llm)

kept_llm = sum(1 for xi in X_ho if predict_proba(xi, w_llm, b_llm) >= 0.5)
print(f"  LLM-clf applied to 5 holdout docs: kept {kept_llm}/5")`}
      </CodeBlock>

      <H3>4e. Full multi-stage pipeline with drop rates</H3>

      <Prose>
        Composing all four stages: heuristics filter the full corpus first, perplexity
        filters the survivors, the quality classifier filters what remains, and a final
        LLM-curator stage would apply to a production corpus (simulated here on the
        holdout portion). Each stage reports its drop rate.
      </Prose>

      <CodeBlock language="python">
{`# Stage 1: heuristics
after_heuristics = [
    (name, doc, lbl)
    for (name, doc), lbl in zip(corpus, labels)
    if passes_heuristics(doc)[0]
]

# Stage 2: perplexity (build LM from quality docs in the heuristic-passed set)
ref_docs = [doc for _, doc, lbl in after_heuristics if lbl == 1]
lp = build_unigram_lm(ref_docs)
pp_scored_2 = [(n, d, l, compute_perplexity(d, lp)) for n, d, l in after_heuristics]
thresh_pp = sorted(pp for *_, pp in pp_scored_2)[int(len(pp_scored_2) * 0.55)]
after_perplexity = [(n, d, l) for n, d, l, pp in pp_scored_2 if pp <= thresh_pp]

# Stage 3: classifier
X_a, _ = tfidf_features([d for _, d, _ in after_perplexity], vocab=vocab_clf)
after_classifier = [
    (n, d, l)
    for (n, d, l), xi in zip(after_perplexity, X_a)
    if predict_proba(xi, w_clf, b_clf) >= 0.45
]

print("\\n=== 4e: Full Pipeline Drop Rates ===")
stages = [
    ("Raw corpus",        len(corpus)),
    ("After heuristics",  len(after_heuristics)),
    ("After perplexity",  len(after_perplexity)),
    ("After classifier",  len(after_classifier)),
]
for i, (stage, count) in enumerate(stages):
    if i == 0:
        print(f"  {stage:<22} {count:3}  —")
    else:
        prev = stages[i-1][1]
        drop = (prev - count) / max(prev, 1) * 100
        print(f"  {stage:<22} {count:3}  {drop:.0f}% dropped")

overall = len(after_classifier) / len(corpus) * 100
print(f"\\n  Overall retention: {len(after_classifier)}/{len(corpus)} = {overall:.0f}%")

# --- Verified output (Python 3.12, no external deps) ---
# === 4e: Full Pipeline Drop Rates ===
#   Raw corpus              20  —
#   After heuristics        20  0% dropped
#   After perplexity        12  40% dropped
#   After classifier        10  17% dropped
#
#   Overall retention: 10/20 = 50%`}
      </CodeBlock>

      <Prose>
        The pipeline retains 10 of 20 documents — all 10 quality documents and none of the
        low-quality ones (the two that survived perplexity filtering were caught by the
        classifier). In a real pipeline on 500B documents, the same structure would reduce
        the corpus to roughly 100B tokens of higher-quality text, consistent with published
        numbers from FineWeb and RedPajama.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>HuggingFace DataTrove</H3>

      <Prose>
        DataTrove is the open-source pipeline framework HuggingFace used to build FineWeb.
        It implements every stage described in this topic as composable pipeline nodes:
        URL filtering, language identification, heuristic filtering (with C4-compatible
        rules), MinHash deduplication, and quality classification. The FineWeb recipe is
        available as a reference implementation. Running DataTrove on a Common Crawl shard
        requires only specifying which pipeline stages to apply and pointing it at an input
        path.
      </Prose>

      <CodeBlock language="python">
{`# DataTrove example (requires: pip install datatrove)
# This runs the FineWeb-style pipeline on a local WARC file.

from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.filters import (
    LanguageFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    C4QualityFilter,
)
from datatrove.pipeline.dedup import MinhashDedupFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor

pipeline = [
    WarcReader(data_folder="path/to/warc/", glob_pattern="*.warc.gz"),
    LanguageFilter(languages=["en"], language_threshold=0.65),
    GopherQualityFilter(
        min_doc_words=50,
        max_doc_words=100_000,
        min_avg_word_length=3,
        max_avg_word_length=10,
    ),
    GopherRepetitionFilter(),
    C4QualityFilter(
        filter_no_terminal_punct=True,
        min_num_sentences=3,
    ),
    MinhashDedupFilter(similarity_threshold=0.7),
    JsonlWriter(output_folder="path/to/output/"),
]

LocalPipelineExecutor(pipeline=pipeline, tasks=8).run()`}
      </CodeBlock>

      <H3>FineWeb pipeline structure</H3>

      <Prose>
        The published FineWeb pipeline runs in seven ordered stages. URL filtering against
        a domain reputation list removes known spam domains before any document is read.
        Language identification (using fastText's language classifier) keeps only
        high-confidence English. MinHash deduplication with 5-gram shingles and 112 hash
        functions removes near-duplicates. C4-style heuristic filters (sentence count,
        word count, terminal punctuation, banned phrases) remove obvious garbage. Gopher
        quality filters (word length distribution, symbol ratios) remove further noise.
        A custom heuristic pass removes boilerplate patterns specific to Common Crawl.
        Finally, for FineWeb-Edu, the educational quality classifier — a BERT-class model
        fine-tuned on 450K Llama-3-70B annotations — scores every remaining document, and
        documents scoring below 3 on the 0–5 scale are dropped.
      </Prose>

      <Prose>
        The FineWeb-Edu classifier is publicly available at
        <Code>HuggingFaceFW/fineweb-edu-classifier</Code> on the HuggingFace Hub. The
        annotation dataset (Llama-3-70B scores for 450K documents) is at
        <Code>HuggingFaceFW/fineweb-edu-llama3-annotations</Code>. Using the published
        classifier for a new corpus requires only running BERT inference and applying a
        score threshold — the annotations and training code are fully open.
      </Prose>

      <H3>CCNet perplexity filtering</H3>

      <Prose>
        CCNet (Wenzek et al. 2019) is the reference pipeline for perplexity-based filtering.
        It deduplicates by paragraph hash, identifies language with fastText, and then scores
        every document against a 5-gram Kneser-Ney language model trained on the Wikipedia
        dump for each of 100 languages. Documents are sorted by perplexity and split into
        three buckets: head (low perplexity, highest quality), middle, and tail. The head
        bucket is used for training; the tail is discarded. The released KenLM models are
        publicly available and can be applied to any new corpus by running a single KenLM
        query command over the input text.
      </Prose>

      <Prose>
        The key implementation detail in CCNet is that perplexity is computed per-paragraph
        rather than per-document, and the document-level score is the median of its paragraph
        scores. This makes the filter robust to long documents that mix one good paragraph with
        ten bad ones — a pattern common in news articles that append long comment sections or
        boilerplate legal text. The median operation is a deliberate choice over the mean:
        a single very high-perplexity paragraph should not cause an otherwise good document
        to be dropped, but a document where the majority of paragraphs are incoherent should
        be filtered regardless of a few good sentences scattered through it.
      </Prose>

      <H3>Nemotron-CC and the diversity-quality frontier</H3>

      <Prose>
        NVIDIA's Nemotron-CC dataset extended the LLM-curator pattern to a full 6.3-trillion-token
        corpus, demonstrating that the approach scales beyond FineWeb's 15T input to corpora
        that span many years of Common Crawl. The pipeline uses a classifier trained on
        Llama-3-70B annotations that scores documents on five dimensions: educational value,
        creative writing quality, factual accuracy, reasoning depth, and code quality. Each
        dimension is scored separately; the final quality label is a weighted combination.
        This multi-dimensional annotation is the current frontier of LLM-curator design: rather
        than a single holistic score, the annotator produces a structured quality profile,
        and the classifier learns to predict each dimension independently. Different pretraining
        mixes can then draw on high-reasoning documents, high-educational documents, and
        high-code documents in different proportions depending on the target model's intended
        use case.
      </Prose>

      <H3>RedPajama and mixture-of-sources pipelines</H3>

      <Prose>
        RedPajama-v2 takes a different architectural approach: rather than a single monolithic
        pipeline, it computes a rich set of quality signals per document and stores them
        alongside the document rather than filtering at ingestion time. This allows downstream
        users to define their own quality threshold and filter the corpus however they choose.
        The signals include: the CCNet perplexity bucket (head/middle/tail), the number of
        unique 5-grams as a diversity proxy, the fraction of lines flagged by C4 heuristics,
        the minhash signature for near-duplicate detection, and several domain-specific
        signals. Storing signals rather than applying fixed filters is a design philosophy
        that acknowledges no single set of thresholds is optimal for all downstream models —
        a research model optimizing for reasoning may want different data than a coding model
        or a general assistant. The trade-off is storage cost (each document carries a vector
        of quality signals) and the responsibility of threshold selection moving to the
        downstream user rather than the corpus builder.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <StepTrace
        label="FineWeb-style pipeline — stages and typical drop rates"
        steps={[
          {
            label: "URL filter",
            render: () => (
              <div>
                <TokenStream tokens={["domain blocklist check", "known spam TLDs", "adult content domains"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Input: 100%.  Output: ~85%.  Drop: ~15% (blocklisted domains).
                  Cost: O(1) hash lookup per URL.
                </div>
              </div>
            ),
          },
          {
            label: "language ID",
            render: () => (
              <div>
                <TokenStream tokens={["fastText lid.176", "threshold: conf ≥ 0.65", "keep English"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Input: ~85%.  Output: ~60%.  Drop: ~30% (non-English or low-confidence).
                  Cost: ~1 ms/doc on CPU.
                </div>
              </div>
            ),
          },
          {
            label: "deduplication (MinHash)",
            render: () => (
              <div>
                <TokenStream tokens={["5-gram shingles", "112 hash functions", "LSH banding", "union-find clusters"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Input: ~60%.  Output: ~25-40%.  Drop: 30-55% of what remains.
                  Covered in the dedup topic; not re-explained here.
                </div>
              </div>
            ),
          },
          {
            label: "heuristic filters (C4 + Gopher)",
            render: () => (
              <div>
                <TokenStream tokens={["word count", "avg word len", "terminal punct", "symbol ratio", "repetition"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Input: ~30%.  Output: ~20-25%.  Drop: ~15-25%.
                  No inference — pure arithmetic on doc statistics.
                </div>
              </div>
            ),
          },
          {
            label: "quality classifier",
            render: () => (
              <div>
                <TokenStream tokens={["positive: Wikipedia/books", "negative: raw web", "BERT/fastText score", "threshold τ per domain"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Input: ~22%.  Output: ~15-18%.  Drop: ~20-30% of remaining.
                  Inference cost: ~5-10 ms/doc on GPU.
                </div>
              </div>
            ),
          },
          {
            label: "LLM-curator filter (FineWeb-Edu style)",
            render: () => (
              <div>
                <TokenStream tokens={["sample 450K docs", "Llama-3-70B annotates", "BERT clf on annotations", "score ≥ 3 → keep"]} />
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: colors.textSecondary, marginTop: 8 }}>
                  Input: ~17%.  Output: ~5-8% (FineWeb-Edu retains 1.3T / 15T = ~9%).
                  LLM touches 0.01% of corpus; clf scales to all.
                </div>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The quality-score distribution shifts rightward at each stage. The plot below shows
        synthetic quality-score distributions (classifier output, 0–1) at four pipeline stages
        for a hypothetical corpus of 1,000 documents. The leftmost curve is the raw crawl;
        each subsequent curve shows what the distribution looks like after an additional
        filtering stage.
      </Prose>

      <Plot
        label="quality-score distribution shift across pipeline stages"
        xLabel="quality score (classifier output)"
        yLabel="fraction of docs"
        width={520}
        height={260}
        series={[
          {
            name: "raw crawl",
            color: colors.textMuted,
            points: [
              [0.0, 0.28], [0.1, 0.20], [0.2, 0.14], [0.3, 0.10],
              [0.4, 0.08], [0.5, 0.07], [0.6, 0.05], [0.7, 0.04],
              [0.8, 0.02], [0.9, 0.01], [1.0, 0.01],
            ],
          },
          {
            name: "after heuristics",
            color: "#60a5fa",
            points: [
              [0.0, 0.15], [0.1, 0.14], [0.2, 0.13], [0.3, 0.12],
              [0.4, 0.11], [0.5, 0.10], [0.6, 0.09], [0.7, 0.07],
              [0.8, 0.05], [0.9, 0.03], [1.0, 0.01],
            ],
          },
          {
            name: "after perplexity filter",
            color: "#c084fc",
            points: [
              [0.0, 0.05], [0.1, 0.06], [0.2, 0.08], [0.3, 0.10],
              [0.4, 0.12], [0.5, 0.13], [0.6, 0.14], [0.7, 0.13],
              [0.8, 0.11], [0.9, 0.06], [1.0, 0.02],
            ],
          },
          {
            name: "after classifier",
            color: colors.gold,
            points: [
              [0.0, 0.01], [0.1, 0.01], [0.2, 0.02], [0.3, 0.04],
              [0.4, 0.07], [0.5, 0.12], [0.6, 0.18], [0.7, 0.22],
              [0.8, 0.20], [0.9, 0.10], [1.0, 0.03],
            ],
          },
        ]}
      />

      <Prose>
        The raw crawl distribution is heavily left-skewed: most documents cluster near
        zero quality score. Each filtering stage shifts mass rightward. After the classifier
        stage, the distribution is unimodal near 0.65–0.75 — characteristic of a corpus
        where the easy junk has been removed but the tail of genuinely ambiguous documents
        remains.
      </Prose>

      <Heatmap
        label="domain pass rate × quality threshold (synthetic)"
        rowLabels={["news", "wiki", "books", "code", "forums", "product pages", "SEO farms"]}
        colLabels={["τ=0.3", "τ=0.4", "τ=0.5", "τ=0.6", "τ=0.7"]}
        matrix={[
          [0.92, 0.85, 0.74, 0.60, 0.42],
          [0.98, 0.96, 0.91, 0.83, 0.68],
          [0.95, 0.91, 0.85, 0.74, 0.55],
          [0.88, 0.80, 0.68, 0.52, 0.35],
          [0.72, 0.60, 0.47, 0.34, 0.22],
          [0.45, 0.32, 0.22, 0.14, 0.08],
          [0.18, 0.10, 0.05, 0.02, 0.01],
        ]}
        colorScale="gold"
      />

      <Prose>
        The heatmap shows that a single global threshold treats domains very differently.
        At <Code>τ = 0.5</Code>, Wikipedia passes 91 percent of its documents while SEO
        farms pass only 5 percent — a reasonable gap. But code documents pass only 68
        percent at the same threshold, and forums pass 47 percent, both likely underestimates
        of their actual quality. Per-domain thresholds address this by calibrating the
        threshold to the domain's baseline score distribution rather than applying a
        one-size-fits-all cutoff.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Stage ordering: always cheap first</H3>

      <Prose>
        The canonical order is heuristics → perplexity → classifier → LLM-curator. The reason
        is purely economic: each stage reduces the volume that the next stage must process, and
        later stages are more expensive per document. A pipeline that runs the LLM-curator
        before heuristics pays LLM inference costs on the full raw crawl. A pipeline that runs
        heuristics first pays LLM costs only on the fraction that survives heuristics — often
        60–80 percent less data.
      </Prose>

      <H3>Global vs. per-domain thresholds</H3>

      <Prose>
        Use a global threshold when: you have a single target domain and your classifier was
        trained on representative data from it. Use per-domain thresholds when: your corpus
        spans multiple domains with different characteristic score distributions, or when you
        have evidence that a global threshold is systematically under-retaining a valuable
        domain. Per-domain thresholds require knowing the domain of each document (a URL-based
        classifier or metadata field usually suffices) and enough data per domain to estimate
        the quantile reliably.
      </Prose>

      <H3>Custom classifier vs. reuse published classifier</H3>

      <Prose>
        Reuse FineWeb-Edu or a similarly published classifier when your target domain overlaps
        substantially with the web (English prose, general knowledge, educational content).
        Train a custom classifier when: your corpus is a specialized domain (legal, medical,
        code, scientific literature) where the published classifier's positive class is a poor
        proxy; when you have access to genuine positive-class examples from your domain; or
        when you have downstream eval evidence that the published classifier is miscalibrated
        for your use case.
      </Prose>

      <H3>When to add the LLM-curator stage</H3>

      <Prose>
        Add the LLM-curator stage when: your annotation budget allows 50K–500K LLM inference
        calls; you need a quality dimension that a binary classifier trained on existing seeds
        cannot capture (e.g., "will this document improve reasoning?" rather than "does this
        look like Wikipedia?"); or you are building a domain-specific corpus where no
        published classifier exists. Skip the LLM-curator stage when: the existing classifier
        is sufficient, annotation budget is tight, or the corpus is small enough that the
        classifier alone is the binding constraint.
      </Prose>

      <H3>Prompt design for the LLM annotator</H3>

      <Prose>
        The annotation prompt is a first-class engineering artifact, not an afterthought.
        The FineWeb-Edu prompt asks the LLM to rate on a 0–5 scale "how valuable this
        document would be as educational material for a student seeking to learn about the
        topic discussed." It includes explicit rubric definitions for each integer value:
        0 means the document contains no educational content whatsoever; 5 means the
        document would be appropriate as a standalone lesson. The rubric was calibrated
        by iterating until the LLM's scores correlated well with human expert ratings on
        a sample of 500 documents.
      </Prose>

      <Prose>
        Two design choices in the FineWeb-Edu prompt are worth noting. First, the rubric
        specifies grade-school and middle-school level knowledge to prevent the LLM from
        over-weighting narrow technical papers that are educational only for domain experts
        but not for general learners. Second, the prompt asks for a JSON-formatted response
        with a score field and a brief rationale — the rationale is used to audit annotation
        quality and spot-check for rubric drift across different runs. A prompt that produces
        only a score is harder to audit; rationale is cheap to generate and provides crucial
        diagnostic signal.
      </Prose>

      <H3>Annotation quality and inter-annotator agreement</H3>

      <Prose>
        LLM annotators are not deterministic. Running the same document through the same
        prompt twice often produces different scores, especially for documents in the 2–4
        range of a 0–5 scale where the boundary is genuinely ambiguous. The practical
        mitigation is to run each sampled document through the LLM twice and keep only
        documents where both scores agree within one point; disagreements are dropped from
        the training set for the student classifier rather than averaged. This costs 2× the
        annotation budget but produces a training set with lower label noise. Classifier
        training on noisier labels is possible but requires larger sample sizes — typically
        3–5× more annotations — to achieve the same classifier quality.
      </Prose>

      <H3>Threshold tuning: downstream eval is the only ground truth</H3>

      <Prose>
        Every threshold in the pipeline — perplexity cutoff, classifier score cutoff,
        domain-specific quantile — should ultimately be validated against a downstream
        evaluation set, not set analytically. The reason is that the quality metrics are
        proxies. Perplexity under Wikipedia measures "looks like Wikipedia," not "improves
        the model." A classifier score measures "resembles curated seeds," not "transfers
        well to benchmark tasks." The only way to know whether a specific threshold setting
        is correct is to train a small model on the resulting corpus and evaluate it. This
        is expensive but necessary: teams that skip downstream validation routinely discover
        that analytically reasonable thresholds produce corpus compositions that perform
        worse than a less aggressive filter would have.
      </Prose>

      <Prose>
        A practical shorthand is to use a fast small-scale proxy: train a 1B-parameter model
        for 10B tokens on two candidate corpora produced by different thresholds and compare
        their evaluation scores on a fixed benchmark suite. This is affordable enough to run
        as an ablation before committing to a full pretraining run. The FineWeb paper includes
        exactly this methodology — their threshold choices are justified by 1B-scale ablations,
        not by intuition.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Heuristic filters scale perfectly with data volume. They are arithmetic operations
        on document strings — no model inference, no communication — and can be parallelized
        trivially across any number of CPU cores. A hundred-node cluster running heuristics
        on CommonCrawl processes a full monthly snapshot in hours. The cost per document is
        flat and negligible.
      </Prose>

      <Prose>
        Classifier inference (BERT or fastText) scales linearly with corpus size and is
        cheap enough in absolute terms to apply to multi-trillion-token corpora. Batched
        BERT inference on a single A100 runs at roughly 50,000 documents per second; a
        full-scale pipeline with 100B documents would take about 2,000 GPU-hours. Expensive
        by local standards, but affordable in the context of a pretraining run that costs
        orders of magnitude more.
      </Prose>

      <Prose>
        Reference-model perplexity filtering (KenLM 5-gram) is fast — KenLM can score
        tens of thousands of documents per second per CPU core — but requires holding a
        large n-gram model in memory. For a 100-language CCNet-style deployment, the
        memory footprint of the language models becomes a practical constraint. Neural
        reference models (GPT-2 small, etc.) are slower by a factor of 10–100× compared
        to KenLM but potentially more discriminative.
      </Prose>

      <Prose>
        LLM-as-judge annotation does not scale to the full corpus. Running Llama-3-70B on
        500 billion documents is not economically viable. The scaling strategy is fixed:
        sample a small fraction, annotate with the LLM, train a classifier, apply the
        classifier at scale. The effective throughput bottleneck is the sampling step —
        specifically, ensuring the sample is representative of the full diversity of the
        corpus so the classifier generalizes to domains it was not annotated on. Active
        learning strategies (annotate the hardest examples first) can reduce the required
        sample size by 2–5× compared to random sampling.
      </Prose>

      <Prose>
        Typical published numbers from frontier pipelines: a raw Common Crawl snapshot for
        a single month is roughly 3–4 billion documents. After URL filtering, language ID,
        and dedup, roughly 500–800 million remain. After heuristic and perplexity filtering,
        200–300 million remain. After quality classifier, 100–150 million remain. For an
        educational-quality filter like FineWeb-Edu, a further 90 percent drop is typical —
        retaining 10–15 million documents per monthly snapshot. Across 96 snapshots, FineWeb
        accumulated 15 trillion tokens total, and FineWeb-Edu retained 1.3 trillion.
      </Prose>

      <Prose>
        The scaling bottleneck for the full pipeline is not compute — it is iteration speed.
        Each change to a filter threshold, a heuristic rule, or a classifier requires
        re-running the affected stages on the full corpus and re-evaluating downstream.
        A pipeline that takes two weeks to run from raw crawl to trained-corpus is one
        where the team can do at most a few ablation iterations before a pretraining run
        deadline. The teams with the best curation pipelines invest heavily in making the
        pipeline fast enough to iterate on: running stages on small representative shards
        (1–10 percent of the corpus) for rapid ablations, using incremental processing
        where only changed documents are re-processed, and maintaining a frozen evaluation
        harness that can score a small model in hours rather than days. The speed of the
        iteration loop matters as much as the correctness of any single filter design.
      </Prose>

      <Prose>
        Data quality improvements also compound differently than model scale improvements.
        Scaling model parameters by 10× costs 10× more compute and typically yields a
        predictable log-linear improvement on standard benchmarks. Improving data quality
        by tightening the quality filter can produce step-function improvements on specific
        capability dimensions — reasoning, factual recall, instruction following — because
        the underlying mechanism is not just "more signal" but "different signal." A
        quality-filtered corpus shifts the gradient distribution in ways that more data
        of the same quality distribution cannot replicate. This makes data quality improvements
        genuinely complementary to scale rather than substitutable, and it explains why the
        Phi series could produce models that outperformed 10× larger models on coding and
        reasoning: the signal the model was trained on was categorically different, not just
        larger.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES AND GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Over-filtering narrows the distribution</H3>

      <Prose>
        Push the quality threshold high enough and the resulting corpus becomes narrowly
        homogeneous: dense with formal encyclopedic prose, thin on colloquial language,
        technical jargon, regional dialects, code comments, and any register that does
        not resemble the positive class the classifier was trained on. A model trained
        exclusively on Wikipedia-like text handles encyclopedic queries fluently and
        struggles with casual conversation, technical documentation, and the loose,
        associative writing that makes up a large fraction of how people actually
        communicate. The filtering succeeded at its stated goal and failed at the implicit
        one.
      </Prose>

      <H3>Classifier bias encodes the positive class</H3>

      <Prose>
        Whatever you use as the positive class for classifier training, the classifier
        learns to reward documents that look like it. A Wikipedia-positive classifier
        undervalues programming documentation, legal writing, colloquial speech, and
        any domain not well-represented in Wikipedia. An educational-value classifier
        undervalues narrative fiction, humor, and cultural writing. These biases
        propagate into the model's knowledge and fluency: what the corpus covered
        fluently, the model handles fluently; what the filter removed systematically,
        the model knows less about.
      </Prose>

      <H3>Adversarial SEO passing</H3>

      <Prose>
        High-quality-seeming surface statistics are not hard to generate. An SEO farm
        that produces 500-word articles with proper punctuation, varied vocabulary,
        reasonable sentence length, and no repetition can pass every heuristic filter.
        A classifier trained on Wikipedia as positive and generic web as negative will
        score these articles above threshold if they are styled as encyclopedia entries.
        This is not a hypothetical — large language models are now used to produce
        exactly this kind of text at scale. A classifier whose positive class is from
        2019 may be calibrated against human-written spam but not LLM-generated spam
        with equivalent surface statistics.
      </Prose>

      <H3>Evaluation contamination from overly broad filtering</H3>

      <Prose>
        Quality filtering that over-retains text from public benchmarks — questions
        from MMLU, problems from GSM8K, passages from BoolQ — inflates evaluation
        scores. The benchmark score no longer measures generalization; it measures
        recall. This contamination is subtle because the text does not appear verbatim;
        it appears as paraphrases, tutorial explanations, or forum discussions that
        discuss the same underlying facts. A filter that rewards educational content
        disproportionately retains educational websites that discuss exactly the topics
        covered by knowledge and reasoning benchmarks.
      </Prose>

      <H3>Per-domain threshold transfer</H3>

      <Prose>
        Thresholds calibrated on one distribution (web English 2022) may not transfer
        to a different distribution (web English 2025, or a different language, or a
        different crawl source). Classifier score distributions shift when the input
        distribution shifts. A threshold that retains the top 20 percent in 2022 may
        retain 35 percent or 10 percent in 2025 if the crawl's composition has changed.
        Recalibrate thresholds when the source distribution changes, not just when
        downstream evals degrade.
      </Prose>

      <H3>Classifier era mismatch</H3>

      <Prose>
        A classifier trained on data from before large-scale LLM-generated content was
        widespread (pre-2022) treats LLM-generated text as out-of-distribution. Depending
        on the architecture, this may cause such text to be assigned very high quality
        scores (the LLM produces fluent, well-structured text) or very low scores (it is
        out-of-distribution relative to the training data). Neither outcome is correct. The
        increasing fraction of LLM-generated text in web crawls is a known and growing
        problem for any classifier trained before 2023.
      </Prose>

      <H3>Heuristic "looks-good" pass</H3>

      <Prose>
        Auto-generated text that mimics human writing statistics — correct word length
        distribution, correct punctuation density, correct sentence count — can pass every
        heuristic filter perfectly while containing zero useful information. A product
        description generator that produces grammatically correct paragraphs of the right
        length with appropriate punctuation will pass the C4 heuristics. Heuristics are
        necessary but not sufficient; they eliminate the obvious junk and leave the
        plausible-looking junk for downstream stages to handle.
      </Prose>

      <Callout accent="gold">
        A quality classifier is a model of what quality looks like according to its
        positive class. Filtering makes your corpus better at resembling that class —
        and systematically worse at resembling everything else. Audit what you removed,
        not just what you kept.
      </Callout>

      <H3>Diversity loss vs. quality gain</H3>

      <Prose>
        The most acute version of the over-filtering failure occurs in low-resource
        languages and specialized domains. A 5 percent quality threshold for Swahili
        applied with an English-trained classifier may drop 95 percent of all available
        Swahili text on the web. There is no 95-percent-quality-equivalent Swahili corpus
        waiting to replace it. The model ends up with a Swahili component trained on a
        tiny, atypical sample. The right response is domain-specific thresholds or
        domain-specific classifiers, but both require investment that is rarely made for
        low-resource languages.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <H3>Penedo et al. 2024 — FineWeb (arXiv:2406.17557)</H3>

      <Prose>
        The most thoroughly documented open curation pipeline to date. Penedo et al.
        describe the full FineWeb pipeline in ablation detail, including the deduplication
        strategy (5-gram MinHash with 112 hash functions), the heuristic filter suite,
        and the educational-quality scoring that produced FineWeb-Edu. The paper releases
        the DataTrove codebase, all ablation model checkpoints, and the Llama-3-70B
        annotation dataset. The core finding: FineWeb-Edu trained models outperform
        models trained on larger but less-curated corpora on knowledge and reasoning
        benchmarks, with gains largest on MMLU, ARC, and OpenBookQA. FineWeb is 15
        trillion tokens; FineWeb-Edu is 1.3 trillion tokens (scored ≥ 3 on the 0–5
        educational scale).
      </Prose>

      <H3>Raffel et al. 2020 — C4 / T5 (arXiv:1910.10683)</H3>

      <Prose>
        The T5 paper introduced C4 (Colossal Clean Crawled Corpus) and systematized
        heuristic filtering as a first-class preprocessing step. The C4 filtering rules —
        remove pages where the terminal sentence does not end in punctuation; remove pages
        containing certain boilerplate strings; remove lines shorter than three words;
        remove pages with fewer than five sentences — became the baseline against which
        subsequent pipelines measured improvement. Every modern heuristic filter suite
        either replicates these rules or is explicitly an ablation variant of them.
      </Prose>

      <H3>Wenzek et al. 2019 — CCNet (arXiv:1911.00359)</H3>

      <Prose>
        CCNet introduced perplexity-based quality filtering as a scalable pipeline
        component. The paper trains Kneser-Ney 5-gram models (via KenLM) on Wikipedia
        for 100 languages and uses perplexity to split every web document into head
        (low perplexity, Wikipedia-like), middle, and tail buckets. The pipeline runs
        paragraph-level deduplication before perplexity scoring. The released KenLM
        models and pipeline code are still in use in modified form in RefinedWeb and
        other industrial pipelines. The central empirical finding: head-bucket data
        produces substantially better downstream models than full crawl data of the
        same token count.
      </Prose>

      <H3>Gunasekar et al. 2023 — Phi-1 (arXiv:2306.11644)</H3>

      <Prose>
        "Textbooks Are All You Need" demonstrated that a 1.3B-parameter model trained
        on 6B tokens of carefully selected "textbook-quality" web data plus 1B tokens
        of synthetically generated textbooks and exercises achieved pass@1 accuracy of
        50.6 percent on HumanEval — matching or exceeding models trained on 10× more
        data. The positive class for their quality filter was explicitly educational
        and instructional text, not Wikipedia generically. Phi-1 is the direct precursor
        to FineWeb-Edu's framing: quality defined as educational value, not encyclopedic
        coverage. The phi-1.5 follow-up extended this to natural language tasks.
      </Prose>

      <H3>Sachdeva et al. 2024 — How to Train Data-Efficient LLMs (arXiv:2402.09668)</H3>

      <Prose>
        Sachdeva et al. evaluated 19 data selection samplers across hundreds of tasks
        and pretraining runs. Their Ask-LLM method uses a zero-shot instruction-tuned
        LLM to directly assess the quality of each training example, producing a quality
        signal that outperforms heuristic and embedding-based selectors. The central
        finding: Ask-LLM data consistently outperforms full-data training even when
        rejecting 90 percent of the original dataset, converging up to 70 percent faster.
        Diversity-based methods (maximizing coverage in feature space) are the second-best
        category. The paper provides the cleanest ablation evidence that quality-aware
        data selection dominates quantity-based scaling at fixed compute budgets.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Pass rate for a five-stage pipeline</H3>

      <Prose>
        A pipeline has five independent stages with the following per-stage retention
        rates: URL filter 85%, language ID 70%, dedup 55%, heuristics 80%, classifier
        65%. What fraction of the original corpus reaches the end of the pipeline? If
        the original corpus is 500 billion documents, how many documents remain? What
        is the effective cost per retained document if the classifier costs
        $0.00001 per document and runs on all documents that pass dedup?
      </Prose>

      <H3>Exercise 2 — Perplexity as signal and trap</H3>

      <Prose>
        You train a perplexity filter using Wikipedia as the reference corpus. You then
        apply it to a crawl that includes (a) high-quality programming documentation,
        (b) academic LaTeX-formatted papers, (c) high-quality colloquial forum posts,
        and (d) boilerplate privacy policies. Which of these would you expect to score
        high perplexity (and therefore be dropped), and which low? For each incorrect
        drop, describe what assumption the perplexity filter is making that fails for
        that document type. How would you modify the reference corpus to address each
        failure?
      </Prose>

      <H3>Exercise 3 — Custom vs. open classifier</H3>

      <Prose>
        You are building a pretraining corpus for a medical language model. The available
        off-the-shelf classifier is FineWeb-Edu, trained to detect general educational
        content. Describe the likely miscalibration: which types of medical documents
        would it over-retain, which would it under-retain, and why? Design a training
        set for a custom medical quality classifier: what would you use as the positive
        class, what as the negative class, and how many examples do you need? At what
        point does retraining become worth the cost compared to using FineWeb-Edu with
        a manually adjusted threshold?
      </Prose>

      <H3>Exercise 4 — LLM curator for a specialized domain</H3>

      <Prose>
        You want to build an LLM-curator pipeline for a corpus of legal documents.
        Design the annotation prompt for the teacher LLM: what rubric would you ask it
        to apply? What is the minimum annotation sample size to train a reliable classifier,
        assuming the legal corpus has five distinct sub-domains (contracts, case law,
        statutes, regulatory filings, legal journalism)? How do you ensure the annotation
        sample is domain-stratified? What failure mode occurs if you annotate only
        contracts and apply the resulting classifier to case law?
      </Prose>

      <H3>Exercise 5 — Diversity loss vs. threshold</H3>

      <Prose>
        You have a quality classifier that assigns scores in [0, 1]. You run it on
        a multilingual corpus and observe that English documents have a mean score of
        0.68 with standard deviation 0.12, while Swahili documents have a mean score
        of 0.31 with standard deviation 0.08. You apply a global threshold of 0.5.
        What fraction of each language passes (assume Gaussian distributions)? If
        Swahili represents 0.5 percent of the corpus by document count, what fraction
        of the final curated corpus is Swahili? How does this compare to the fraction
        you would retain if you used a language-specific threshold set at the 50th
        percentile of each language's score distribution? Describe the downstream
        model capability implications of each choice.
      </Prose>
    </div>
  ),
};

export default dataCurationPipelines;
