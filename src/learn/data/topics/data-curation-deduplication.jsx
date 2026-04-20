import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const dataCurationDeduplication = {
  title: "Data Curation & Deduplication (MinHash, Bloom Filters)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Pre-training corpora are large — Common Crawl alone is tens of petabytes of compressed text — and a substantial fraction of those bytes are near-duplicates of each other. Near-duplicates are worse than useless. They bias gradients toward repeated content, silently contaminate evaluation benchmarks, and waste compute on material the model has effectively seen before. Any honest pre-training pipeline spends serious engineering on finding and removing them. This topic is about the two workhorse tools: MinHash with locality-sensitive hashing for near-duplicate detection, and Bloom filters for exact-string deduplication.
      </Prose>

      <H2>Why deduplication matters more than it seems</H2>

      <Prose>
        Three concrete harms make the stakes precise. The first is training signal distortion. When a sentence appears a thousand times in a corpus, it is effectively overweighted by a factor of a thousand in the loss. The model spends gradient steps reinforcing that content at the expense of rarer, often more informative text. Lee et al. (2022) showed that deduplication alone — no architecture changes, no additional data — consistently improves downstream accuracy. The second harm is memorization. Verbatim repetition of training text dramatically increases the probability that a model will reproduce it at inference time. That creates both privacy risk, when the duplicated text contains personal information, and copyright risk, when it contains licensed prose or code. The third harm is evaluation contamination. If any portion of a benchmark shows up in training — even paraphrased — the numbers you report are inflated relative to genuine generalization. Exact string match only catches 1:1 duplicates. The real battle is near-duplicates, where the same content appears with minor wording changes, different formatting, or light rewrites.
      </Prose>

      <Prose>
        The scale of the problem is not intuitive until you measure it. Studies of Common Crawl have found that anywhere from 30 to 70 percent of documents have at least one near-duplicate elsewhere in the same crawl. Boilerplate content — cookie consent banners, legal disclaimers, navigation menus scraped from millions of pages — accounts for a large share of this. News articles reprinted across wire services, Reddit comments mirrored on aggregator sites, StackOverflow answers scraped into Q&A datasets: all of these produce near-duplicate clusters that dominate certain regions of the data distribution. The deduplication pass is not a minor cleanup step. It is often the single biggest reduction in effective corpus size.
      </Prose>

      <H2>MinHash and locality-sensitive hashing</H2>

      <Prose>
        The classic approach to near-duplicate detection is MinHash combined with locality-sensitive hashing. The starting point is representing each document as a set of shingles — overlapping n-grams of words or characters. Two documents are near-duplicates if their shingle sets have high Jaccard similarity. Jaccard is the right measure: it is the size of the intersection divided by the size of the union, bounded between zero and one, and it degrades gracefully as documents diverge.
      </Prose>

      <MathBlock>{"J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}"}</MathBlock>

      <Prose>
        The problem is computational. Computing Jaccard similarity between all document pairs is O(N²) — infeasible for a corpus of billions of documents. MinHash solves this by producing a compact signature for each document that estimates Jaccard from a small number of hash values. The key insight is that if you apply the same hash function to both sets and take the minimum value from each, the probability that those two minima are equal is exactly the Jaccard similarity of the sets. With k independent hash functions you get k min-values per document; the fraction that agree between two signatures is an unbiased estimator of Jaccard.
      </Prose>

      <CodeBlock language="python">
{`import hashlib

def shingles(doc, n=5):
    tokens = doc.split()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def minhash(doc, num_hashes=128):
    """Returns a signature — one min-value per hash function."""
    shings = shingles(doc)
    sig = [float("inf")] * num_hashes
    for sh in shings:
        for i in range(num_hashes):
            # Seed each hash function differently
            h = int(hashlib.md5(f"{i}:{sh}".encode()).hexdigest(), 16)
            sig[i] = min(sig[i], h)
    return sig

def estimated_jaccard(sig_a, sig_b):
    """Fraction of matching min-hashes ≈ Jaccard similarity."""
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)`}
      </CodeBlock>

      <Prose>
        With 128 hash functions, each document compresses to 128 integers regardless of its original length. For a corpus of one billion documents, this is still a large dataset, but it is now in a form where efficient approximate nearest-neighbor lookup becomes feasible.
      </Prose>

      <H3>LSH — the bucketing trick</H3>

      <Prose>
        MinHash signatures reduce documents to compact vectors, but you still need to avoid comparing every pair. Locality-sensitive hashing solves this. Split each 128-value signature into b bands of r rows — say b = 20 bands of r = 5 rows each. For each band, concatenate its r values into a single string and hash that string into a bucket. Two documents that share at least one identical band will hash into the same bucket and become candidates for comparison. Documents that share zero bands are ignored.
      </Prose>

      <Prose>
        The tuning of b and r controls which similarity threshold triggers bucketing. With b = 20 and r = 5, documents with around 80% Jaccard similarity are very likely to share at least one band; documents below 50% are very unlikely to. This creates a near-step-function threshold that you can slide by adjusting the band decomposition. The payoff is severe: instead of O(N²) comparisons, you compare only within-bucket pairs. For typical web corpora, this reduces comparisons by several orders of magnitude and brings the full pairwise near-duplicate check within reach on a cluster of ordinary machines.
      </Prose>

      <H2>Bloom filters for exact-string dedup</H2>

      <Prose>
        For finding exact duplicates — the same URL crawled twice, the same paragraph appearing verbatim, the same document hash appearing in multiple shards — MinHash is overkill. Bloom filters give O(1) membership checks with a configurable false-positive rate and tiny memory footprint. The mechanics are simple: allocate an array of m bits, initialized to zero; choose k independent hash functions. To insert an item, compute all k hashes and set the corresponding bit positions. To check membership, compute all k hashes and test whether every corresponding bit is set. If any bit is zero, the item is definitely not in the set. If all bits are set, the item is probably in the set — with a false-positive probability determined by m, k, and the number of items inserted.
      </Prose>

      <Prose>
        Given a target number of items n and an acceptable false-positive rate p, the optimal m and k are fixed by closed-form expressions. A Bloom filter for one billion URLs at a 0.1% false-positive rate requires roughly 1.8 GB of bits — versus the tens of gigabytes a hash set of the same items would occupy. For URL-level deduplication at web scale, that memory difference is decisive.
      </Prose>

      <CodeBlock language="python">
{`import mmh3
import math

class BloomFilter:
    def __init__(self, n_items, fp_rate=0.001):
        # Optimal size and hash count for given n and fp rate
        self.m = int(-n_items * math.log(fp_rate) / (math.log(2) ** 2))
        self.k = max(1, int((self.m / n_items) * math.log(2)))
        self.bits = bytearray(self.m // 8 + 1)

    def _hashes(self, item):
        for i in range(self.k):
            h = mmh3.hash(item, seed=i) % self.m
            yield h

    def add(self, item):
        for h in self._hashes(item):
            self.bits[h // 8] |= 1 << (h % 8)

    def __contains__(self, item):
        return all(self.bits[h // 8] & (1 << (h % 8)) for h in self._hashes(item))`}
      </CodeBlock>

      <Prose>
        The false-positive behavior has a useful asymmetry for deduplication. A false positive means an item is incorrectly flagged as already seen, causing a genuine unique document to be discarded. A false negative cannot occur — if the filter says an item is new, it is new. Setting the false-positive rate to 0.1% means you lose, at most, 0.1% of genuinely unique documents to spurious filtering. For URL deduplication across a web crawl, that is an acceptable tradeoff against the memory savings.
      </Prose>

      <H3>Semantic deduplication</H3>

      <Prose>
        Both MinHash and Bloom filters operate on lexical surface form. MinHash will correctly identify two articles as near-duplicates if they share most of their words; it will miss a near-duplicate where the same story is told in entirely different words. Semantic deduplication addresses this by embedding documents into a vector space and clustering in that space. SemDeDup, published by researchers at Meta in 2023, demonstrated this approach at scale: embed each document with a small pretrained encoder, cluster embeddings using k-means or approximate nearest-neighbor search, and remove all but the highest-quality representative from each dense cluster.
      </Prose>

      <Prose>
        The computational cost is substantially higher than MinHash — each document requires a forward pass through an encoder, and the resulting embeddings are typically 400 to 768 floats per document. For a billion-document corpus, that is several terabytes of embeddings and a non-trivial clustering problem. But the reduction is qualitatively different. MinHash removes copies; semantic dedup removes versions. The same Wikipedia article written in simple English and standard English, two news reports about the same event with different phrasing, tutorial content covering the same programming concept in different examples — all of these survive MinHash and fall to semantic dedup. The pretraining efficiency gains from this second pass are real, though smaller than the first-pass lexical dedup.
      </Prose>

      <H2>The pipeline in practice</H2>

      <Prose>
        Real curation pipelines — The Pile, C4, RedPajama, FineWeb — combine these tools in a layered sequence. The stages are not interchangeable; each one removes a qualitatively different kind of duplicate, and running them in order is important because earlier stages dramatically reduce the input size for more expensive later stages. URL deduplication is cheap and eliminates the most obvious redundancy immediately. Document-level MinHash and LSH then handle near-duplicate documents regardless of their URL. Paragraph-level or line-level exact dedup cleans up boilerplate that survived document-level dedup because it appeared in otherwise-unique documents. The optional semantic pass at the end operates on whatever remains.
      </Prose>

      <StepTrace
        label="a typical large-scale dedup pipeline"
        steps={[
          { label: "1. url bloom filter", render: () => (
            <TokenStream tokens={["100B docs", " →", " Bloom check", " →", " 60B unique urls"]} />
          ) },
          { label: "2. minhash + lsh", render: () => (
            <TokenStream tokens={["60B docs", " →", " 128-hash signatures", " →", " LSH buckets", " →", " 20B clusters"]} />
          ) },
          { label: "3. cluster keep-one", render: () => (
            <TokenStream tokens={["20B clusters", " →", " keep highest-quality per cluster", " →", " 20B docs"]} />
          ) },
          { label: "4. (optional) semantic", render: () => (
            <TokenStream tokens={["20B docs", " →", " embed + cluster", " →", " 15B semantically-unique"]} />
          ) },
        ]}
      />

      <Prose>
        The numbers in the pipeline diagram are illustrative but not unreasonable. URL-level Bloom filtering alone can eliminate 40% of a raw web crawl — that many URLs are recrawled. MinHash dedup of the remaining documents typically removes another 50 to 70%. The final corpus that actually trains the model is often 10 to 20% of the raw crawl by document count, but a much higher fraction of that corpus is genuinely informative content rather than boilerplate, mirrors, and scraped noise.
      </Prose>

      <Callout accent="gold">
        FineWeb (HuggingFace, 2024) provides a well-documented example of how these stages compose in practice, with ablations showing the effect of each dedup pass on downstream benchmark performance. The gains from URL dedup, MinHash dedup, and quality filtering are each separable and each meaningful.
      </Callout>

      <Prose>
        The tools described here are simple. MinHash is decades-old compression theory. Bloom filters are undergraduate data structures. Neither requires a GPU or a PhD to implement. What makes them consequential is the compounding effect: a better dedup pipeline shows up in downstream accuracy, in how much compute you waste training on redundant content, in how safe the model is from memorizing sensitive training text, and in how trustworthy your evaluation numbers are. This is one of those places where boring data engineering quietly decides the ceiling.
      </Prose>
    </div>
  ),
};

export default dataCurationDeduplication;
