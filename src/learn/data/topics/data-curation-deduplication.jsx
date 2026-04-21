import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const dataCurationDeduplication = {
  title: "Data Curation & Deduplication (MinHash, Bloom Filters)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        A pretraining corpus is not a clean object. The raw Common Crawl snapshot from a single month is measured in petabytes; scraped once a month for a decade, the archive is somewhere past an exabyte. A large fraction of those bytes are repeats — the same page crawled by different URLs, the same article syndicated across wire services, the same boilerplate cookie banner duplicated across ten million otherwise-different sites, the same StackOverflow answer mirrored on a dozen aggregator domains. When you open a fresh web dump and start counting, typical estimates land in the range of thirty to seventy percent of documents having at least one near-duplicate elsewhere. The exact ratio depends on how you define "near," but the shape of the distribution is consistent: a long tail of unique content buried under a short, very heavy head of reprinted, boilerplated, and trivially-rewritten duplicates.
      </Prose>

      <Prose>
        Deduplication is the operation that reshapes that distribution before any token ever reaches a GPU. It matters for three concrete reasons, each of which is measurable and each of which has ended up in the ablation tables of at least one headline pretraining paper. The first is training-signal distortion. If a sentence appears a thousand times in the corpus, the cross-entropy loss over the corpus is effectively a thousand-times-weighted version of that sentence's loss. Gradient steps reinforce that content at the expense of rarer text that is usually more informative per token. The clearest empirical evidence for this comes from Lee et al. 2022, "Deduplicating Training Data Makes Language Models Better" (arXiv:2107.06499). They trained identical models on C4 and on a deduplicated version of C4 and on an order of magnitude less deduplicated data, and showed that models trained on the smaller-but-deduplicated corpus matched or exceeded the fully-duplicated baseline on downstream benchmarks, while requiring fewer training steps and emitting memorized text at roughly one-tenth the rate. The corpus did not need to be bigger. It needed to be cleaner.
      </Prose>

      <Prose>
        The second is memorization and privacy. Verbatim repetition of a string in the training data sharply increases the probability that a language model will regurgitate that exact string at inference time. That is an architectural observation, not a research prediction — the softmax over a vocabulary is overwhelmingly peaked on whatever continuation the model has seen most often, and a sentence that appeared twelve thousand times is hard not to overfit. Every leaked email, every copy-pasted license key, every personally-identifying address pulled from a scraped web page has a duplication count in the raw crawl. Dedup brings those counts back down to one, and one copy is statistically hard for a model to memorize verbatim. The third is evaluation contamination. If any slice of a public benchmark appears in training — even paraphrased, even reformatted into a blog post that happened to include the question — the benchmark score no longer measures generalization; it measures recall. Lee et al. document exactly this: on the datasets they examined, more than four percent of validation set items had overlap with the training set after standard "exact duplicate" filtering. The numbers reported on those benchmarks were inflated by something like that much. Near-duplicates, not exact duplicates, were the source.
      </Prose>

      <Prose>
        The algorithmic tools for this job are older than deep learning and have not fundamentally changed. MinHash was introduced by Andrei Broder in 1997 at AltaVista ("On the Resemblance and Containment of Documents"), where it was used to cluster thirty million crawled documents into three and a half million near-duplicate groups. Locality-sensitive hashing was formalized by Indyk and Motwani a year later ("Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality", STOC 1998). Bloom filters are older still: Burton Bloom published them in the CACM in 1970, motivated by the problem of hyphenation dictionaries too large to fit in core memory. Semantic deduplication, the newest of the four, is a 2023 contribution from Abbas et al. at Meta ("SemDeDup", arXiv:2303.09540), and it is really a repackaging of vector clustering with a specific dedup-y frame. None of this is frontier research. The entire toolkit has been sitting in undergraduate textbooks and web-search systems for decades. What is new is the scale at which it must run and the downstream cost of getting it wrong.
      </Prose>

      <Prose>
        One more framing is useful before the mechanics. There are three levels at which duplication can exist in a corpus, and the dedup pipeline attacks each separately. At the URL level, the same string identifier has been crawled twice and produced two document records — the cheapest possible duplication, caught by a Bloom filter membership check in microseconds. At the document level, two different URLs point to content that is identical or near-identical byte-for-byte — mirror sites, news syndication, bot-generated SEO farms, and the myriad aggregators that scrape other aggregators. And at the semantic level, two documents express the same content with different words — translations, paraphrases, automated rewrites, tutorial explanations that cover the same concept from different angles. Each level requires a different algorithm because each level produces a different kind of similarity signal. A good pipeline treats them as sequential stages, not as interchangeable alternatives.
      </Prose>

      <Callout accent="gold">
        The deduplication pass is not cleanup. On realistic web crawls it removes more content than the quality filter, the language filter, and the profanity filter combined. A billion-document corpus routinely shrinks to under two hundred million documents of genuinely distinct material. Everything downstream — scaling-law fits, ablations, eval numbers, compute budgets — assumes that reduction has happened.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Three pictures cover the whole topic. Each is the mental model for one of the three algorithmic primitives, and each is visualizable without any equations.
      </Prose>

      <H3>MinHash: the minimum of a random permutation</H3>

      <Prose>
        Imagine the universe of all possible shingles — every five-word sequence of English, every five-character sequence of Unicode, however you are chopping documents into pieces. Write those shingles down in some order, and next to each one put a column for each document in your corpus, with a 1 if that document contains the shingle and 0 otherwise. You now have a gigantic sparse 0/1 matrix with one row per shingle and one column per document. The Jaccard similarity between two columns is the number of rows where both are 1 divided by the number of rows where at least one is 1 — the shared shingles over the union.
      </Prose>

      <Prose>
        Now permute the rows at random. Look down column A from top to bottom and record the index of the first 1 you see. Do the same for column B. The probability that those two indices are equal is <em>exactly</em> the Jaccard similarity of the two columns — no approximation, no asymptotics, just a counting argument. The first 1 in each column is a uniformly random element of that column's shingle set; the two sets produce the same minimum iff the first row that appears in either set is in the intersection. Repeat this with a different random permutation a hundred and twenty-eight times and you have a 128-number signature per document. The fraction of signature entries that match between two documents is an unbiased estimator of their Jaccard similarity. The precision of that estimate is set by the number of hashes, not by document length.
      </Prose>

      <Prose>
        In practice you do not actually permute rows — you hash each shingle with a different seed and take the minimum hash value. That is computationally equivalent to permuting rows and reading off the first 1, because a uniform random hash is indistinguishable from a random permutation as far as the minimum statistic is concerned. The name "MinHash" is a compression of "minimum of a hashed set."
      </Prose>

      <H3>LSH banding: turning pairwise comparison into bucket lookup</H3>

      <Prose>
        MinHash gives you a tiny fixed-size signature per document, but computing the similarity between every pair of documents in a billion-document corpus is still a trillion-comparison job you cannot afford. Locality-sensitive hashing is the trick that converts this into hash-table lookups. Chop each 128-number signature into <Code>b</Code> bands of <Code>r</Code> rows each — say <Code>b=32</Code> bands of <Code>r=4</Code> rows. For each band, concatenate its four numbers into a string and use that string as a bucket key. Two documents end up in the same bucket for band <Code>i</Code> iff those four MinHash values agree between them. Documents that share at least one bucket anywhere are candidate near-duplicates, and you do the full similarity check only on those.
      </Prose>

      <Prose>
        The whole operation turns into a threshold. High-Jaccard pairs share many MinHash values, and with many shared MinHash values they are almost certain to agree on <em>some</em> band of four. Low-Jaccard pairs share few MinHash values and are almost certain to disagree on <em>every</em> band. In between, there is a crossover region — called the S-curve — whose steepness and center are controlled by your choice of <Code>b</Code> and <Code>r</Code>. Tuning those two knobs tunes your similarity threshold. The entire payoff is that bucket lookups are O(1), so finding candidate pairs is linear in the number of documents instead of quadratic.
      </Prose>

      <H3>Bloom filter: a coarse lossy set that uses no memory</H3>

      <Prose>
        A Bloom filter is an array of <Code>m</Code> bits, initialized to zero, plus <Code>k</Code> independent hash functions from items to bit positions. To insert an item, hash it <Code>k</Code> ways and flip those <Code>k</Code> bits to 1. To ask whether an item is a member, hash it <Code>k</Code> ways and check whether all <Code>k</Code> bits are 1. If any bit is zero, the item is definitely not in the set; an item that was inserted would have set that bit. If all bits are 1, the item is probably in the set, but bits can be set to 1 by <em>other</em> inserted items, which means you have a controllable false-positive rate and a guaranteed zero false-negative rate.
      </Prose>

      <Prose>
        The key asymmetry is that false positives are controllable and false negatives are impossible. For URL-level deduplication you want exactly that direction of error: you would rather occasionally drop a fresh URL (false positive on "already seen") than accidentally re-ingest a URL you already processed. The cost is decisive. A hash set storing a billion URLs as strings takes tens of gigabytes of memory. A Bloom filter with a one-in-a-thousand false-positive rate stores the same billion URLs in under two gigabytes of bits, and each query is a handful of cache-line-friendly reads.
      </Prose>

      <Callout>
        The mental shortcut is: MinHash tells you how similar, LSH tells you which pairs to even look at, Bloom tells you whether you have ever seen this exact thing before. The three are composed, not compared.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <H3>Jaccard similarity</H3>

      <Prose>
        The entire near-duplicate pipeline rests on Jaccard similarity between shingle sets. For two sets <Code>A</Code> and <Code>B</Code> the definition is the familiar intersection-over-union.
      </Prose>

      <MathBlock>{"J(A, B) \\;=\\; \\frac{|A \\cap B|}{|A \\cup B|}"}</MathBlock>

      <Prose>
        Jaccard is bounded in <Code>[0, 1]</Code>, equal to 1 when the sets are identical and 0 when disjoint. It is the right similarity for shingle sets because it degrades gracefully: inserting a sentence, deleting a paragraph, rephrasing a caption all reduce Jaccard smoothly rather than cliff-jumping to zero. Character-ngram Jaccard is not identical to semantic similarity, which is why semantic dedup exists as a separate pass, but for lexical near-duplicates it is the canonical measure.
      </Prose>

      <H3>MinHash preserves Jaccard in expectation</H3>

      <Prose>
        The MinHash preservation identity is the one non-trivial equation in the topic. Let <Code>h</Code> be a hash function behaving like a uniform random permutation of the shingle universe, and let <Code>{"m(S) = min{h(x) : x ∈ S}"}</Code> be the min-hash of set <Code>S</Code>. Then:
      </Prose>

      <MathBlock>{"\\Pr\\!\\left[\\, m(A) = m(B) \\,\\right] \\;=\\; \\frac{|A \\cap B|}{|A \\cup B|} \\;=\\; J(A, B)"}</MathBlock>

      <Prose>
        The proof is a two-line combinatorial argument. Consider the element <Code>x*</Code> that achieves the minimum hash in <Code>A ∪ B</Code>; since <Code>h</Code> is a uniform random permutation, <Code>x*</Code> is uniformly distributed over <Code>A ∪ B</Code>. The two min-hashes agree iff <Code>x* ∈ A ∩ B</Code>, which happens with probability <Code>|A ∩ B| / |A ∪ B|</Code>. With <Code>k</Code> independent hash functions the estimator is the sample mean of <Code>k</Code> independent Bernoulli random variables, each with parameter <Code>J</Code>, so the estimator has standard deviation <Code>sqrt(J(1-J)/k)</Code>. For <Code>k = 128</Code> and <Code>J = 0.5</Code> that is about 4.4 percent.
      </Prose>

      <H3>LSH banding probability</H3>

      <Prose>
        With <Code>b</Code> bands of <Code>r</Code> rows each (so total signature length <Code>k = br</Code>), two documents with true Jaccard <Code>s</Code> agree on all <Code>r</Code> rows of any single band with probability <Code>s^r</Code>, disagree on at least one row with probability <Code>1 - s^r</Code>, disagree on every one of <Code>b</Code> bands with probability <Code>(1 - s^r)^b</Code>, and therefore agree on at least one band — which is what bucketing requires — with probability:
      </Prose>

      <MathBlock>{"P_\\text{match}(s) \\;=\\; 1 - (1 - s^r)^b"}</MathBlock>

      <Prose>
        This is the S-curve. At <Code>s = 0</Code> the probability is 0; at <Code>s = 1</Code> it is 1; in between there is a sigmoid whose inflection point and sharpness are set by <Code>r</Code> and <Code>b</Code>. Increasing <Code>r</Code> pushes the threshold rightward (harder to match); increasing <Code>b</Code> shifts it leftward (more chances to match). A useful approximate threshold is <Code>s* ≈ (1/b)^(1/r)</Code>, the point where <Code>P_match</Code> first rises above 50 percent. For <Code>(b, r) = (32, 4)</Code> this gives <Code>s* ≈ 0.42</Code>; for <Code>(20, 5)</Code> it is roughly 0.55; for <Code>(10, 10)</Code> around 0.80. The following heatmap shows the full S-curves for three common choices.
      </Prose>

      <Heatmap
        label="LSH band-matching probability vs true Jaccard"
        rowLabels={["s=0.2", "s=0.3", "s=0.4", "s=0.5", "s=0.6", "s=0.7", "s=0.8", "s=0.9"]}
        colLabels={["(32,4)", "(20,5)", "(10,10)"]}
        matrix={[
          [0.050, 0.006, 0.000],
          [0.229, 0.047, 0.000],
          [0.564, 0.186, 0.001],
          [0.873, 0.470, 0.010],
          [0.988, 0.802, 0.059],
          [1.000, 0.975, 0.249],
          [1.000, 1.000, 0.679],
          [1.000, 1.000, 0.986],
        ]}
        colorScale="gold"
      />

      <Prose>
        Read the columns. At <Code>(32, 4)</Code>, a document with true Jaccard 0.5 matches with probability 0.87 — this is a loose threshold, catching down to roughly 0.4. At <Code>(10, 10)</Code>, the same 0.5-similar document matches with probability 0.01, effectively never. The banding is a dial: <Code>(32, 4)</Code> is aggressive and recall-heavy; <Code>(10, 10)</Code> is conservative and precision-heavy. Real pipelines usually pick somewhere around 0.6–0.7 target similarity and tune <Code>(b, r)</Code> accordingly.
      </Prose>

      <H3>Bloom filter false-positive rate</H3>

      <Prose>
        Given <Code>m</Code> bits, <Code>k</Code> hash functions, and <Code>n</Code> inserted items, the probability that any specific bit is still zero after all insertions is:
      </Prose>

      <MathBlock>{"\\Pr[\\text{bit zero}] \\;=\\; \\left(1 - \\tfrac{1}{m}\\right)^{kn} \\;\\approx\\; e^{-kn/m}"}</MathBlock>

      <Prose>
        A membership query returns a false positive iff all <Code>k</Code> bits it checks happen to be 1. Treating those bits as independent (a good approximation for <Code>m ≫ k</Code>):
      </Prose>

      <MathBlock>{"\\text{FP}(m, k, n) \\;=\\; \\left(1 - e^{-kn/m}\\right)^k"}</MathBlock>

      <Prose>
        For a target FP rate <Code>p</Code>, two knobs matter: the bits-per-item ratio <Code>m/n</Code> and the number of hash functions <Code>k</Code>. Taking the derivative of the FP expression with respect to <Code>k</Code> and setting it to zero yields the optimal hash count:
      </Prose>

      <MathBlock>{"k^* \\;=\\; \\frac{m}{n} \\ln 2 \\;\\approx\\; 0.693 \\cdot \\tfrac{m}{n}"}</MathBlock>

      <Prose>
        Substituting back gives <Code>p ≈ (1/2)^(k*)</Code>, which inverts to <Code>m ≈ -n \ln p / (\ln 2)^2 ≈ 1.44 n \log_2(1/p)</Code>. At a target 0.1 percent FP rate this is roughly 14.4 bits per item — under 2 GB for a billion URLs. The plot below shows FP rate as a function of <Code>m/n</Code> at four different <Code>k</Code>; notice that the minimum over <Code>k</Code> at fixed <Code>m/n</Code> is exactly at <Code>k* = 0.693 m/n</Code>.
      </Prose>

      <Plot
        label="Bloom FP rate vs bits-per-item (log-scale effect)"
        xLabel="bits per item (m/n)"
        yLabel="false-positive rate"
        width={520}
        height={260}
        series={[
          { name: "k=4",  points: [[4, 0.160], [6, 0.056], [8, 0.024], [10, 0.012], [12, 0.006], [14, 0.004], [16, 0.002], [20, 0.001]] },
          { name: "k=6",  points: [[4, 0.220], [6, 0.064], [8, 0.022], [10, 0.008], [12, 0.004], [14, 0.002], [16, 0.001], [20, 0.0003]] },
          { name: "k=8",  points: [[4, 0.312], [6, 0.086], [8, 0.025], [10, 0.008], [12, 0.003], [14, 0.001], [16, 0.0006], [20, 0.0001]] },
          { name: "k=10", points: [[4, 0.425], [6, 0.123], [8, 0.034], [10, 0.010], [12, 0.003], [14, 0.001], [16, 0.0005], [20, 0.00009]] },
        ]}
      />

      <Prose>
        The four curves cross each other. At <Code>m/n = 4</Code>, <Code>k=4</Code> is best because too many hashes saturate the filter; at <Code>m/n = 20</Code>, <Code>k=10</Code> wins because there is enough room to afford the extra membership bits. Every real Bloom deployment picks <Code>k</Code> to match its <Code>m/n</Code>, not the other way around.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every piece of Python in this section was run locally on Python 3.12, and the outputs embedded below are copied verbatim from a single session. Dependencies: only the standard library. By the end of the section we will have working shingling, MinHash, LSH banding, a Bloom filter, and a composed pipeline that runs URL-level bloom dedup followed by document-level MinHash+LSH on a synthetic near-duplicate corpus.
      </Prose>

      <H3>4a. Shingles</H3>

      <Prose>
        Shingling turns a document into a set of overlapping n-grams. Token n-grams of length 5 are the standard for English text; character n-grams of length 9 work better for code; short documents often use n=3 because longer shingles leave almost nothing to compare. The core function is a one-liner.
      </Prose>

      <CodeBlock language="python">
{`def shingles(doc, n=5):
    tokens = doc.split()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

doc = "the quick brown fox jumps over the lazy dog by the river"
sh = shingles(doc, n=5)
# verified output:
# doc tokens: 12
# # shingles (n=5): 8
# sample: 'brown fox jumps over the'
#         'fox jumps over the lazy'
#         'jumps over the lazy dog'`}
      </CodeBlock>

      <Prose>
        A document of <Code>L</Code> tokens has <Code>L - n + 1</Code> overlapping n-grams, which for typical web documents of a few hundred tokens gives a few hundred shingles. Sets, not lists, because order does not matter for Jaccard — only whether each shingle is present.
      </Prose>

      <H3>4b. MinHash</H3>

      <Prose>
        One hash function plus one seed gives one min-value. Do that 128 times with different seeds and you have a 128-number signature. In the pedagogical implementation we use MD5 for its convenient fixed-width output; production systems use MurmurHash or xxHash for speed, but MD5 is uniform enough that the Jaccard estimate is accurate. The extra per-seed formatting prefix (<Code>f"{"{"}i{"}"}:{"{"}sh{"}"}"</Code>) is what effectively gives us 128 independent hash functions from one.
      </Prose>

      <CodeBlock language="python">
{`import hashlib

def minhash(doc, num_hashes=128, n=5):
    shings = shingles(doc, n=n)
    sig = [float("inf")] * num_hashes
    for sh in shings:
        for i in range(num_hashes):
            h = int(hashlib.md5(f"{i}:{sh}".encode()).hexdigest(), 16)
            if h < sig[i]:
                sig[i] = h
    return sig

def estimated_jaccard(sig_a, sig_b):
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)

def true_jaccard(a, b, n=5):
    sa, sb = shingles(a, n=n), shingles(b, n=n)
    return len(sa & sb) / max(1, len(sa | sb))`}
      </CodeBlock>

      <Prose>
        The sanity check is to compare true and estimated Jaccard on a pair with known overlap. On two near-duplicate documents (same 20-word frame with a couple of word substitutions) and one unrelated document, running this code produced the following table.
      </Prose>

      <CodeBlock language="python">
{`# doc_a vs doc_b: two near-duplicate documents with 2 word swaps
# doc_a vs doc_c: completely unrelated subject
# Verified output (128 hashes, shingle size varies):

# n=3  J(A,B)_true=0.524  J(A,B)_est=0.602   J(A,C)_true=0.000  J(A,C)_est=0.000
# n=5  J(A,B)_true=0.474  J(A,B)_est=0.438   J(A,C)_true=0.000  J(A,C)_est=0.000`}
      </CodeBlock>

      <Prose>
        The estimator lands within roughly one standard deviation of the true value — the theory predicts <Code>sqrt(J(1-J)/128) ≈ 0.044</Code>, and the observed n=5 deviation is 0.036, the n=3 deviation 0.078 (one-sigma-ish and two-sigma-ish draws, respectively — normal Monte Carlo noise for a single sample). The unrelated pair estimates zero exactly, as expected: when sets are disjoint no hash can produce the same min. With only 32 hashes instead of 128 the estimate would wobble by around 0.09; increase to 256 and the wobble drops to around 0.031. There is a linear memory tradeoff and a <Code>1/sqrt(k)</Code> accuracy gain — typical for Monte Carlo estimators.
      </Prose>

      <H3>4c. LSH banding</H3>

      <Prose>
        LSH takes the 128-value signature, chops it into <Code>b</Code> bands of <Code>r</Code> rows, hashes each band to a bucket, and returns candidate pairs. The whole thing is eighteen lines of Python.
      </Prose>

      <CodeBlock language="python">
{`from collections import defaultdict

def lsh_candidates(sigs, bands=32, rows=4):
    assert bands * rows == len(next(iter(sigs.values())))
    buckets = defaultdict(list)
    for doc_id, sig in sigs.items():
        for b in range(bands):
            band = tuple(sig[b * rows:(b + 1) * rows])
            key = (b, hash(band))         # one bucket namespace per band
            buckets[key].append(doc_id)
    cands = set()
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cands.add(tuple(sorted((ids[i], ids[j]))))
    return cands`}
      </CodeBlock>

      <Prose>
        Two things worth noticing. First, <Code>(b, hash(band))</Code> separates each band into its own bucket namespace — two documents that happen to share values on band 3 do not collide with documents that share the same values on band 7. Second, within-bucket candidate generation is quadratic in bucket size; if a bucket explodes (which happens for boilerplate text whose signature is near-degenerate) you can cap bucket sizes at some large constant without meaningfully hurting recall. Tested on a synthetic corpus with three near-duplicate clusters of five documents each plus two unrelated documents, the LSH pass produced the following candidate pairs.
      </Prose>

      <CodeBlock language="python">
{`# Synthetic corpus: 3 clusters of 5 perturbed duplicates + 2 unrelated docs.
# Verified output (bands=32, rows=4, num_hashes=128, n=3):
# corpus size: 17
# candidate pairs found: 5
#   within-cluster: 5   cross-cluster: 0
# sample: [('a0', 'a1'), ('a0', 'a3'), ('a3', 'a4'), ('b3', 'b4')]`}
      </CodeBlock>

      <Prose>
        Five candidate pairs, every single one of them within a true cluster. Zero cross-cluster false positives; zero singleton outliers bucketed with anyone. In a brute-force comparison the same signal would have required <Code>C(17, 2) = 136</Code> pairwise distance computations. LSH did it with five candidate checks — a roughly 27-fold reduction on seventeen documents, and the reduction grows super-linearly as the corpus grows. Note that two-word perturbations at <Code>n=3</Code> shingles yield Jaccard values near 0.35–0.55, which is right at the S-curve's knee for <Code>(b, r) = (32, 4)</Code>; that is why some intra-cluster pairs do not become candidates. Lowering <Code>r</Code> to 3 or the shingle size to <Code>n=2</Code> would widen recall at the cost of more candidate noise.
      </Prose>

      <H3>4d. Bloom filter</H3>

      <Prose>
        The minimal Bloom filter fits in a class. The arithmetic for <Code>m</Code> and <Code>k</Code> uses the optimal expressions derived in section 3. Two independent hash functions (MD5 and SHA1) are enough because a standard trick — <Code>h_i(x) = h1(x) + i*h2(x) mod m</Code> — gives you <Code>k</Code> approximately-independent hashes from two real ones, avoiding the cost of <Code>k</Code> full cryptographic hashes per operation.
      </Prose>

      <CodeBlock language="python">
{`import hashlib
import math

class BloomFilter:
    def __init__(self, n_items, fp_rate=0.001):
        self.m = int(-n_items * math.log(fp_rate) / (math.log(2) ** 2))
        self.k = max(1, int((self.m / n_items) * math.log(2)))
        self.bits = bytearray(self.m // 8 + 1)
        self.n = 0

    def _hashes(self, item):
        item_bytes = item.encode() if isinstance(item, str) else item
        h1 = int(hashlib.md5(item_bytes).hexdigest(), 16)
        h2 = int(hashlib.sha1(item_bytes).hexdigest(), 16)
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item):
        for h in self._hashes(item):
            self.bits[h // 8] |= 1 << (h % 8)
        self.n += 1

    def __contains__(self, item):
        return all(self.bits[h // 8] & (1 << (h % 8)) for h in self._hashes(item))

    def expected_fp_rate(self):
        return (1 - math.exp(-self.k * self.n / self.m)) ** self.k`}
      </CodeBlock>

      <Prose>
        Sanity check: construct a filter sized for 10,000 items with a 1 percent target FP rate, insert 10,000 random URLs, then query 100,000 held-out URLs that were never inserted. The observed false-positive rate should hit the target within Monte Carlo noise. Running this code produced:
      </Prose>

      <CodeBlock language="python">
{`# Verified output:
# m=95,850 bits   k=6   memory=11.7 KiB
# false negatives on 10000 inserted items: 0
# false positives on 100000 held-out items: 971   (observed FP rate 0.0097)
# theoretical FP rate at n=10K:                    0.0101`}
      </CodeBlock>

      <Prose>
        Observed 0.97 percent, theoretical 1.01 percent — within Monte Carlo noise of the prediction (100K trials at p=0.01 gives a standard error of about 0.001). Zero false negatives, as the theory guarantees: a Bloom filter never forgets something it has seen. Eleven and a half kilobytes of memory for ten thousand items. Scaled to a billion URLs at the same parameters, memory is about 1.2 GB and query cost remains six hash operations and six bit tests per membership check.
      </Prose>

      <H3>4e. Composed pipeline</H3>

      <Prose>
        The three primitives compose in order. URL-level Bloom dedup first — it is the cheapest and removes the largest share of trivial duplicates. Then MinHash+LSH on whatever survives, clustering near-duplicate documents and keeping one representative per cluster. A real pipeline would add line-level exact dedup and a semantic pass after this; the simplified version below shows URL + MinHash+LSH.
      </Prose>

      <CodeBlock language="python">
{`def dedup_pipeline(docs, bloom_fp=0.001, bands=32, rows=4, n_shingle=3):
    # Stage 1: URL-level Bloom dedup.
    url_seen = BloomFilter(n_items=max(len(docs), 1000), fp_rate=bloom_fp)
    stage1 = []
    for url, text in docs:
        if url in url_seen:
            continue
        url_seen.add(url)
        stage1.append((url, text))

    # Stage 2: MinHash + LSH candidate generation + union-find clustering.
    sigs = {url: minhash(text, bands * rows, n=n_shingle) for url, text in stage1}
    cands = lsh_candidates(sigs, bands=bands, rows=rows)
    parent = {url: url for url, _ in stage1}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in cands:
        # Tighten LSH candidate with a signature-level Jaccard check.
        if estimated_jaccard(sigs[a], sigs[b]) >= 0.7:
            union(a, b)

    # Keep one representative per cluster.
    seen_root, stage2 = {}, []
    for url, text in stage1:
        root = find(url)
        if root in seen_root:
            continue
        seen_root[root] = True
        stage2.append((url, text))
    return stage1, stage2`}
      </CodeBlock>

      <Prose>
        Tested on a synthetic 110-document corpus with three clusters of twenty near-duplicates each, thirty URL-duplicate insertions, and twenty unrelated noise documents, the pipeline produced:
      </Prose>

      <CodeBlock language="python">
{`# Verified output:
# raw input: 110 docs
# after URL bloom dedup:     80 docs  (removed 30)
# after MinHash+LSH dedup:   65 docs  (removed 15)
# overall reduction:         110 -> 65  (40.9% removed)`}
      </CodeBlock>

      <Prose>
        URL bloom removed the thirty exact URL duplicates. MinHash+LSH then collapsed fifteen additional near-duplicate documents that survived URL dedup because their URLs were different. Combined reduction: 41 percent of the input. The sixty-five survivors are a clean cover of the three underlying semantic clusters (approximately kept-representatives plus the pairs whose perturbations pushed Jaccard below the <Code>≥ 0.7</Code> cluster threshold) plus the twenty unrelated documents. Tighten the shingle size or widen LSH bands to remove more; loosen to remove fewer. There is no single "right" operating point — only one matched to the perturbation profile of your corpus.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Three libraries carry nearly all of the production weight in open-source pretraining today. None of them require you to reimplement any of the algorithms in section 4, and at the scale of a real crawl you do not want to — the libraries handle on-disk signature storage, sharded LSH, incremental insertion, parallel execution, and compressed bitmaps in ways the fifty-line pedagogical versions above do not.
      </Prose>

      <H3>datasketch — MinHash + LSH in Python</H3>

      <Prose>
        <a href="https://ekzhu.com/datasketch/" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>datasketch</a> is the most battle-tested Python MinHash library. It provides <Code>MinHash</Code>, <Code>MinHashLSH</Code>, and <Code>MinHashLSHForest</Code> (for approximate nearest-neighbor queries rather than just thresholded retrieval). The API maps directly onto the concepts in section 4.
      </Prose>

      <CodeBlock language="python">
{`from datasketch import MinHash, MinHashLSH

def shingles(doc, n=5):
    toks = doc.split()
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}

def to_minhash(doc, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for sh in shingles(doc, n=3):
        m.update(sh.encode("utf-8"))
    return m

lsh = MinHashLSH(threshold=0.4, num_perm=128)
for doc_id, text in corpus.items():
    lsh.insert(doc_id, to_minhash(text))

# Query:
near = lsh.query(to_minhash(query_text))   # list of doc_ids above threshold`}
      </CodeBlock>

      <Prose>
        Running this on six documents containing two near-duplicate pairs returned exactly the expected buckets:
      </Prose>

      <CodeBlock language="python">
{`# production datasketch MinHashLSH, threshold=0.4, num_perm=128
#   a1: near-dups -> ['a2']     # both "Rust is a systems programming..." variants
#   a2: near-dups -> ['a1']
#   a3: near-dups -> []         # perturbed variant fell below threshold
#   b1: near-dups -> ['b2']     # both "Bloom filters allow compact..." variants
#   b2: near-dups -> ['b1']
#   c1: near-dups -> []         # unrelated pasta document, no near-dups`}
      </CodeBlock>

      <Prose>
        The library handles threshold-to-band-parameter translation automatically — you give it a threshold and <Code>num_perm</Code>, and it picks <Code>(b, r)</Code> for you. For explicit control, construct with <Code>params=(bands, rows)</Code> and set threshold to <Code>None</Code>. For billion-document corpora, <Code>datasketch</Code> supports a Redis or Cassandra storage backend so that the LSH index does not need to fit in one machine's RAM.
      </Prose>

      <H3>pybloom-live and rbloom — production Bloom filters</H3>

      <Prose>
        For URL-level deduplication at scale, <Code>pybloom-live</Code> (pure Python) and <Code>rbloom</Code> (Rust-backed, roughly ten times faster) are the standard choices. Both support scalable Bloom filters — a chain of Bloom filters of geometrically increasing size that gracefully handles unknown item counts without saturating.
      </Prose>

      <CodeBlock language="python">
{`from pybloom_live import ScalableBloomFilter

bf = ScalableBloomFilter(initial_capacity=10_000_000,
                         error_rate=0.001,
                         mode=ScalableBloomFilter.SMALL_SET_GROWTH)
for url in url_stream():
    if url in bf:
        continue
    bf.add(url)
    process(url)`}
      </CodeBlock>

      <Prose>
        The scalable variant adds internal filters when the current one approaches capacity; each new filter is twice the size of the previous with a tighter FP rate, and the aggregate FP rate stays bounded. For a web crawl of a priori unknown size, this is the right choice — a fixed-size Bloom filter whose <Code>n</Code> is set too low will silently degrade to effectively zero useful membership information.
      </Prose>

      <H3>DataTrove — HuggingFace's production pipeline</H3>

      <Prose>
        <a href="https://github.com/huggingface/datatrove" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>DataTrove</a> is the pipeline framework HuggingFace built for FineWeb. It provides pre-implemented <Code>MinhashDedupSignature</Code>, <Code>MinhashDedupBuckets</Code>, <Code>MinhashDedupCluster</Code>, and <Code>MinhashDedupFilter</Code> steps that chain into an <Code>LocalPipelineExecutor</Code> or a SLURM-dispatched <Code>SlurmPipelineExecutor</Code> for cluster execution.
      </Prose>

      <CodeBlock language="python">
{`from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupBuckets
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from datatrove.utils.hashing import HashConfig

# Stage 1: compute per-document MinHash signatures.
sig_pipe = [
    JsonlReader("s3://crawl/raw/"),
    MinhashDedupSignature(output_folder="s3://work/signatures/",
                          config=HashConfig(hash_fc="sha1"),
                          n_grams=5, num_buckets=14, hashes_per_bucket=8),
]

# Stages 2-4: bucket -> cluster -> filter. Each runs after the previous.
# In practice these are separate pipelines because they have different
# parallelism characteristics.`}
      </CodeBlock>

      <Prose>
        The FineWeb paper (Penedo et al. 2024, arXiv:2406.17557) reports that this pipeline processed 96 Common Crawl snapshots into a 15-trillion-token clean corpus. Their ablation showed that dedup applied per-snapshot outperformed dedup applied across snapshots — an important detail, because naively deduplicating the whole archive at once removes too much legitimately-repeated content (news stories quoting prior stories, documentation referencing earlier versions) and actually degrades downstream benchmark scores. The scale at which dedup aggressiveness starts to hurt is itself an empirical question, and FineWeb's contribution is partly in documenting where that line is.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The following walkthrough shows the four canonical stages of a large-scale pretraining dedup pipeline, with representative document-count reductions at each stage. The absolute numbers are illustrative (drawn from published FineWeb and C4 ablations) but the stage-by-stage <em>ratios</em> are representative of real crawls.
      </Prose>

      <StepTrace
        label="typical large-scale dedup pipeline"
        steps={[
          {
            label: "Stage 1: URL-level Bloom dedup",
            render: () => (
              <TokenStream tokens={["100B raw docs", " →", " Bloom check", " →", " 60B unique URLs"]} />
            ),
          },
          {
            label: "Stage 2: document-level MinHash",
            render: () => (
              <TokenStream tokens={["60B docs", " →", " 128-hash signatures", " →", " 60B signatures"]} />
            ),
          },
          {
            label: "Stage 3: LSH + cluster keep-one",
            render: () => (
              <TokenStream tokens={["60B signatures", " →", " LSH buckets", " →", " 20B clusters", " →", " 20B representatives"]} />
            ),
          },
          {
            label: "Stage 4: (optional) semantic dedup",
            render: () => (
              <TokenStream tokens={["20B docs", " →", " embed + cluster", " →", " 15B semantically-unique"]} />
            ),
          },
        ]}
      />

      <Prose>
        The per-stage reductions are not equal. Stage 1 is the fattest cut by percentage (40 percent gone) because URL-level duplication is trivially easy to produce — any link shared on social media gets crawled dozens of times from mirrors and aggregators. Stage 3 is the fattest cut by value, because the documents it removes are the ones that would have silently distorted training. Stage 4 is often skipped in smaller pipelines; the marginal win is real but smaller than stages 1–3, and the compute cost is much higher.
      </Prose>

      <Callout accent="gold">
        The per-stage ratios shown are median, not bounds. Sites with heavy boilerplate (news wire sites, legal document repositories, pharmaceutical leaflet dumps) can hit 90 percent reduction at stage 3. Sites with diverse content (academic papers, well-curated forums, personal blogs) may lose only 10 percent at the same stage. Running dedup separately per source is often better than running it corpus-wide.
      </Callout>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Bloom vs MinHash vs exact hash</H3>

      <Prose>
        These three do not substitute for each other; they solve different problems and compose rather than compete.
      </Prose>

      <Heatmap
        label="when to use which"
        rowLabels={["URL dedup", "exact-doc dedup", "near-duplicate", "semantic paraphrase"]}
        colLabels={["exact hash set", "Bloom", "MinHash+LSH", "embed+cluster"]}
        matrix={[
          [0.9, 1.0, 0.2, 0.1],
          [0.9, 0.9, 0.3, 0.1],
          [0.1, 0.1, 1.0, 0.5],
          [0.0, 0.0, 0.3, 1.0],
        ]}
        colorScale="green"
      />

      <Prose>
        The exact hash set is the most accurate (zero false positives, zero false negatives) but uses the most memory. Bloom gives up a tunable sliver of precision for one to two orders of magnitude less memory and is the right choice anywhere the item count is in the billions. MinHash+LSH is required whenever you care about <em>similarity</em> rather than equality — a one-character edit to a document keeps it in the same semantic bucket but changes every exact hash. Embedding-based dedup catches paraphrases and cross-lingual duplicates that are lexically almost disjoint but semantically identical; it is also the slowest and most compute-heavy.
      </Prose>

      <H3>Shingle size</H3>

      <Prose>
        The dominant knob in any MinHash pipeline is the shingle size <Code>n</Code>. The guidance, distilled from a dozen production writeups:
      </Prose>

      <ul style={{ color: colors.textSecondary, fontSize: 14, lineHeight: 1.7, paddingLeft: 22 }}>
        <li><strong>n=5 tokens</strong> — default for English prose and most Western languages. Balances distinctiveness (avoiding over-clumping) against coverage (avoiding under-matching on short documents).</li>
        <li><strong>n=3 tokens</strong> — short documents like tweets, search queries, product titles. With long shingles, short documents have fewer than ten shingles and MinHash estimates become noisy.</li>
        <li><strong>n=9 characters</strong> — code, where "word" is ambiguous and character-level shingles capture stylistic signatures better. Also useful for languages without whitespace word separators (CJK, Thai).</li>
        <li><strong>n=13 characters</strong> — for aggressive plagiarism detection where even paraphrase-light duplication matters. Common in academic integrity tooling rather than pretraining dedup.</li>
      </ul>

      <H3>Number of MinHash functions</H3>

      <Prose>
        More hashes means smaller signature-estimate variance but linearly more memory and compute. The standard range is 128 to 256; FineWeb used 112 (14 buckets × 8 rows), DataTrove defaults to 112, and datasketch defaults to 128. Pushing below 64 gets noisy enough that thresholding becomes unreliable; pushing above 512 wastes compute without measurable gain. The <Code>1/sqrt(k)</Code> error scaling tells you that doubling from 128 to 256 only tightens the estimate by about 40 percent — usually not worth it.
      </Prose>

      <H3>When to add a semantic pass</H3>

      <Prose>
        Semantic dedup is worth adding when (a) the corpus contains substantial cross-lingual duplication (translated articles, multilingual documentation, news syndication across language borders), (b) your training objective is sensitive to paraphrased eval contamination, or (c) you have measured an actual benchmark win from semantic dedup on your dataset. It is not worth adding as a default. The compute cost — a forward pass through an encoder per document, plus a terabyte-scale vector index — is nearly always larger than the compute cost of stages 1–3 combined.
      </Prose>

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>MinHash and LSH scale beautifully</H3>

      <Prose>
        MinHash signature computation is O(<Code>L · k</Code>) per document, where <Code>L</Code> is document length and <Code>k</Code> is hash count. Both factors are constants, and the operation is embarrassingly parallel — no cross-document state at all. A ten-thousand-machine Hadoop cluster can compute signatures for a trillion documents in a few hours. LSH bucketing is O(<Code>b</Code>) per document, where <Code>b</Code> is the number of bands; bucket membership queries are O(1) amortized. The end-to-end pipeline has done billion-document corpora on modest clusters (a few hundred cores) for over a decade. The original Broder 1997 paper reports clustering thirty million documents at AltaVista on 1997-era hardware; modern implementations have advanced by several orders of magnitude without fundamentally changing the algorithm.
      </Prose>

      <Prose>
        The one regime where MinHash+LSH does not scale linearly is near-degenerate signatures. If a single bucket accumulates millions of documents (because all of them happen to be boilerplate whose signature is nearly identical), the within-bucket candidate generation is O(<Code>N²</Code>) for that bucket. Production pipelines cap per-bucket size, either by rejecting over-saturated buckets or by sub-bucketing with additional hashes. Without this safety valve, a single dump of automatically-generated SEO spam can make the pipeline hang.
      </Prose>

      <H3>Bloom filters scale by memory, not by count</H3>

      <Prose>
        The memory of a Bloom filter depends on the target item count and the target FP rate, not on the actual number of items inserted — provided you do not exceed the target. At 1.44 bits per item per decibel of FP rate (in base-10 units of FP), a filter for one billion URLs at 0.1 percent FP takes about 1.67 GB of bits. Inserts and queries are both O(<Code>k</Code>) where <Code>k</Code> is the number of hash functions, typically 6–10. Both operations are cache-friendly: only <Code>k</Code> bit-positions are touched per operation, and those positions are scattered randomly across <Code>m</Code> bits, which is important to get right — consecutive or nearby hash outputs cause cache-line thrashing that kills throughput.
      </Prose>

      <Prose>
        The failure mode at scale is saturation. If you insert twice as many items as the filter was sized for, the FP rate does not degrade gracefully to 2× — it degrades catastrophically. At <Code>n = 2 n_design</Code> for a filter designed at 0.1 percent FP, the observed rate climbs to around 10 percent. Scalable Bloom filters (a chain of filters of geometrically growing size) solve this, at a ~2× memory overhead for a priori-unknown data.
      </Prose>

      <H3>Semantic dedup is the expensive one</H3>

      <Prose>
        Embedding every document is O(<Code>N · L · d</Code>) where <Code>L</Code> is document length and <Code>d</Code> is model cost — a forward pass through a 100M-parameter encoder is on the order of 10<sup>11</sup> FLOPs per 1000-token document. At a billion documents that is 10<sup>20</sup> FLOPs, on the order of a full small-model training run just for embeddings. Clustering the resulting vectors is O(<Code>N · d</Code>) per k-means iteration or O(<Code>N<sup>1.5</sup></Code>) with approximate nearest-neighbor structures. The Abbas et al. SemDeDup paper reports roughly 50 percent corpus reduction on LAION with minimal downstream loss, but the compute cost of the dedup pass is comparable to the compute cost of the downstream training it enables. Whether that is a good trade depends entirely on how many models you are going to train on the deduplicated corpus.
      </Prose>

      <H3>Parallelism is universal</H3>

      <Prose>
        All four stages (Bloom, MinHash, LSH, semantic) are embarrassingly parallel in the data-parallel sense: split the corpus into shards, run the same code on each shard, and combine. Bloom filters merge trivially (OR of bit arrays). MinHash signatures merge trivially (they are already per-document). LSH buckets merge as dictionaries. Only the semantic-clustering step requires genuine coordination, because clustering is inherently non-decomposable. In practice even that step is parallelized by running approximate nearest-neighbor search (HNSW, FAISS) in shards and then merging cluster assignments with a final pass.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes & gotchas</H2>

      <H3>Wrong shingle size</H3>

      <Prose>
        Shingle size too small and every document looks like every other — the shingle set devolves toward a handful of common bigrams that almost any text contains, Jaccard stays high between unrelated documents, and the dedup pass removes content indiscriminately. Shingle size too large and near-duplicates stop matching — a document with a single inserted word has a different n-gram at every position around that word, and for <Code>n=10</Code> you have just knocked out 20 shingles, which is a large fraction of the total for short documents. The safest default for English prose is <Code>n=5</Code> tokens; test with a held-out set of known-duplicate pairs and adjust.
      </Prose>

      <H3>Hash collisions concentrating on common phrases</H3>

      <Prose>
        MD5 and its kin are uniform in expectation, but adversarial or pathologically frequent shingles can still create degenerate signatures. The phrase "the quick brown" appears in a measurable fraction of English text; if your min-hash for some seed happens to be unusually small for "the quick brown", many otherwise-unrelated documents get the same signature value for that seed. This is a constant-factor degradation, not a catastrophic failure, but it accumulates across seeds. The fix is to use hash functions with a cryptographic uniformity guarantee (SHA1, SHA256) at the cost of throughput, or to pre-filter shingles by inverse document frequency so that overly-common n-grams get less weight.
      </Prose>

      <H3>LSH banding misses borderline near-duplicates</H3>

      <Prose>
        The S-curve is sharp but not infinitely sharp. With <Code>(b, r) = (20, 5)</Code> set for a target threshold of 0.5, a document pair at true Jaccard 0.45 matches with probability 0.40 — nearly two-thirds of those pairs are silently missed. If you care about that boundary region, either widen the bands (lower <Code>r</Code>) or run a second LSH pass with a lower threshold. Do not assume the S-curve is a step function.
      </Prose>

      <H3>Bloom saturation degrades cliff-wise</H3>

      <Prose>
        A Bloom filter designed for <Code>n</Code> items and then fed <Code>2n</Code> items has an observed FP rate roughly ten times the theoretical rate at <Code>n</Code> — not two times. For a filter designed at 0.1 percent FP, over-saturation pushes it to something like 10 percent, which is unusable. Monitor the filter's load factor (number of set bits divided by <Code>m</Code>) in production; at 50 percent set bits you are at the design FP rate, at 70 percent you are already three times worse. Scalable Bloom filters eliminate this failure mode at the cost of a memory overhead factor of roughly 2.
      </Prose>

      <H3>URL dedup without normalization</H3>

      <Prose>
        <Code>http://example.com/page</Code> and <Code>https://example.com/page/</Code> and <Code>https://example.com/page?utm_source=twitter</Code> are three different strings with identical content. A Bloom filter run on raw URLs will insert all three. Production pipelines normalize URLs before insertion: lowercase the hostname, strip tracking parameters, canonicalize trailing slashes, remove fragment identifiers. The normalization rule must be documented and consistent — the biggest data bug I have personally seen in a dedup pipeline was one team normalizing and another not, producing a silent doubling of the corpus across pipeline stages.
      </Prose>

      <H3>MinHash with poor hash functions</H3>

      <Prose>
        The Jaccard estimate is unbiased only if the hash function is effectively uniform over the shingle universe. <Code>hash()</Code> in Python is not — its randomization is per-process and not cryptographic, and the output distribution has exploitable structure for certain input classes. Use <Code>hashlib.md5</Code>, <Code>hashlib.sha1</Code>, <Code>mmh3</Code>, or <Code>xxhash</Code>. Never use the Python builtin <Code>hash()</Code> or Java's <Code>String.hashCode()</Code> for cross-run-stable signatures.
      </Prose>

      <H3>Evaluation contamination the dedup pass missed</H3>

      <Prose>
        Lexical dedup removes copies of the training data that are word-for-word close to test items. It does not remove paraphrased eval items — "What is the capital of France?" and "Name France's capital city." have near-zero Jaccard. If evaluation integrity is critical (frontier-model contamination audits, regulatory compliance), a semantic pass or a dedicated eval-decontamination pass is required. Several recent papers (including FineWeb) document measurable benchmark inflation that survived standard dedup.
      </Prose>

      <H3>Semantic dedup removing legitimately diverse content</H3>

      <Prose>
        Embedding-based dedup catches paraphrases, which is the point, but also catches documents that happen to share topic and style. Ten thousand distinct StackOverflow answers about Python list comprehensions can cluster together in embedding space despite being useful in aggregate for a coding model. Semantic dedup with an aggressive threshold can over-reduce — reported cases include embeddings clustering legitimate news coverage of the same event into a single "duplicate" cluster. The fix is to measure downstream benchmark impact rather than trusting compression ratio as the only metric, and to pair semantic dedup with a content-quality filter so that the cluster representative selection uses quality signal rather than random choice.
      </Prose>

      <H3>Deduplicating across crawl snapshots degrades models</H3>

      <Prose>
        This one is empirical and published in the FineWeb ablations. Running MinHash dedup across all 96 Common Crawl snapshots jointly — treating the archive as one corpus — produces worse downstream benchmarks than running MinHash dedup within each snapshot separately. The hypothesis is that documents genuinely re-published month to month (updated documentation, revised news articles, new editions of technical guides) carry useful temporal signal that whole-archive dedup destroys. Per-snapshot dedup is the right default for large multi-snapshot corpora.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Every source below was verified via WebSearch against the canonical ArXiv, ACM, or official repository URL. Where multiple versions exist (preprint vs conference vs journal), the most-cited canonical link is listed.
      </Prose>

      <ul style={{ color: colors.textSecondary, fontSize: 14, lineHeight: 1.7, paddingLeft: 22 }}>
        <li>
          <strong>Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini (ACL 2022)</strong> — "Deduplicating Training Data Makes Language Models Better." Canonical reference for the downstream impact of dedup on language model quality. <a href="https://arxiv.org/abs/2107.06499" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>arXiv:2107.06499</a>.
        </li>
        <li>
          <strong>Broder (1997)</strong> — "On the Resemblance and Containment of Documents." The original MinHash paper, applied to thirty million AltaVista documents. Published in Compression and Complexity of Sequences 1997. <a href="https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>Princeton-hosted PDF</a>.
        </li>
        <li>
          <strong>Indyk & Motwani (STOC 1998)</strong> — "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality." The formalization of locality-sensitive hashing; LSH banding of MinHash signatures is the canonical application. <a href="https://graphics.stanford.edu/courses/cs468-06-fall/Papers/06%20indyk%20motwani%20-%20stoc98.pdf" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>Stanford-hosted PDF</a>.
        </li>
        <li>
          <strong>Bloom (CACM 1970)</strong> — "Space/Time Trade-offs in Hash Coding with Allowable Errors." The original Bloom filter paper, motivated by hyphenation dictionaries too large to fit in memory. <a href="https://dl.acm.org/doi/10.1145/362686.362692" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>ACM Digital Library</a>.
        </li>
        <li>
          <strong>Abbas, Tirumala, Simig, Ganguli, Morcos (2023)</strong> — "SemDeDup: Data-efficient learning at web-scale through semantic deduplication." Introduces embedding-based dedup as a complement to lexical dedup; reports roughly 50 percent corpus reduction on LAION with minimal downstream loss. <a href="https://arxiv.org/abs/2303.09540" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>arXiv:2303.09540</a>.
        </li>
        <li>
          <strong>Penedo et al. (2024)</strong> — "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale." Documents the full pretraining dedup pipeline applied to 96 Common Crawl snapshots, including the per-snapshot-vs-global dedup ablation. <a href="https://arxiv.org/abs/2406.17557" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>arXiv:2406.17557</a>; pipeline code at <a href="https://github.com/huggingface/datatrove" target="_blank" rel="noreferrer" style={{ color: colors.gold }}>huggingface/datatrove</a>.
        </li>
      </ul>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <Prose>
        Solutions follow each question. Try to answer before reading the solution.
      </Prose>

      <H3>Exercise 1 — MinHash signature memory</H3>

      <Prose>
        Given one million documents, each represented by a 128-hash MinHash signature with 64-bit integers, how much memory do the signatures occupy?
      </Prose>

      <Callout>
        <strong>Solution.</strong> 1,000,000 docs × 128 hashes × 8 bytes = 1.024 × 10<sup>9</sup> bytes ≈ 0.95 GB. Fits comfortably in RAM on any modern server. Scaling to one billion documents gives 950 GB — still tractable in a sharded setup, though no longer single-machine. This is why signature-based pipelines scale: the data-reduction factor from raw documents (kilobytes each) to signatures (one kilobyte each) is already three to four orders of magnitude.
      </Callout>

      <H3>Exercise 2 — Bloom filter sizing for one billion URLs</H3>

      <Prose>
        What <Code>m</Code> (bit array size) and <Code>k</Code> (hash count) should a Bloom filter use for one billion URLs at a target 0.1 percent false-positive rate?
      </Prose>

      <Callout>
        <strong>Solution.</strong> Using the formulas <Code>m = -n \ln p / (\ln 2)<sup>2</sup></Code> and <Code>k = (m/n) \ln 2</Code>: <Code>m ≈ -10<sup>9</sup> × \ln(0.001) / (\ln 2)<sup>2</sup> ≈ 1.44 × 10<sup>10</sup></Code> bits ≈ 1.67 GB. <Code>k ≈ (m/n) × 0.693 ≈ 10</Code> hashes. At roughly 14.4 bits per item — independent of the hash-value range or item size — this is a small fraction of the memory that a string hash set holding one billion URLs would need (tens of GB). Query cost is ten bit-tests.
      </Callout>

      <H3>Exercise 3 — Why LSH if MinHash signatures are small?</H3>

      <Prose>
        MinHash compresses a document to 128 numbers. Why do we need LSH bucketing on top of that — can we not just compare signatures pairwise?
      </Prose>

      <Callout>
        <strong>Solution.</strong> Pairwise comparison of <Code>N</Code> signatures is <Code>C(N, 2) = N(N-1)/2</Code> comparisons. For one billion documents that is 5 × 10<sup>17</sup> pairs — infeasible even at one nanosecond per comparison (it would take around fifteen years on a single core). The signature size bounded the per-pair cost but not the pair count. LSH replaces the O(<Code>N<sup>2</sup></Code>) pair enumeration with a hash-table lookup that is O(<Code>N · b</Code>) to build and O(<Code>1</Code>) amortized per candidate lookup, reducing total work by several orders of magnitude. The key insight is that most pairs have near-zero similarity and can be safely pruned without explicit comparison — LSH does that pruning by bucket misses.
      </Callout>

      <H3>Exercise 4 — Jaccard from partial agreement</H3>

      <Prose>
        Two documents are each represented by a 32-hash MinHash signature. Ten of the thirty-two hash values agree between them. What is the estimated Jaccard similarity, and what is the 95 percent confidence interval on the estimate?
      </Prose>

      <Callout>
        <strong>Solution.</strong> Point estimate: <Code>10/32 = 0.3125</Code>. The estimator is a sample mean of 32 Bernoulli trials with unknown parameter <Code>J</Code>; at <Code>J ≈ 0.31</Code> the standard error is <Code>sqrt(J(1-J)/32) ≈ sqrt(0.31 × 0.69 / 32) ≈ 0.082</Code>. A 95 percent CI is approximately <Code>0.31 ± 1.96 × 0.082 = [0.15, 0.47]</Code> — wide, because 32 hashes is too few to estimate similarity precisely. This is why production systems use 128–256 hashes: the standard error scales as <Code>1/sqrt(k)</Code>, so 128 hashes tighten the interval by a factor of 2.
      </Callout>

      <H3>Exercise 5 — When does semantic dedup add value over MinHash?</H3>

      <Prose>
        Under what conditions does running a semantic (embedding-based) dedup pass after MinHash+LSH actually improve downstream model quality, versus being just a compute tax with no upside?
      </Prose>

      <Callout>
        <strong>Solution.</strong> Semantic dedup helps specifically when your corpus contains documents that are lexically dissimilar but semantically near-identical, and when those documents would distort training or eval. Concrete cases: (1) Cross-lingual duplicates — the same Wikipedia article in simple English vs standard English, or news reports of the same event in English vs Spanish. MinHash will not match these; semantic dedup will. (2) Heavy paraphrase — the same technical tutorial written by two authors using different examples. (3) Paraphrased benchmark questions that survived MinHash decontamination. It does <em>not</em> help, and is often actively harmful, when your corpus is composed of diverse documents that share topic and style (e.g., ten thousand distinct but stylistically-similar coding tutorials). Before adding a semantic pass, measure downstream benchmark impact on a held-out set — do not assume more aggressive dedup is always better.
      </Callout>

      <Prose>
        The pattern across all five exercises is the same. Dedup is a set of cheap, old, well-understood algorithms whose value comes from composition and from correct parameter setting. Get the knobs right and a dedup pipeline removes half your corpus while improving downstream benchmark scores. Get them wrong and you either waste memory on an exact hash set, saturate a Bloom filter into uselessness, over-prune legitimate content, or miss the evaluation contamination the pipeline was built to catch. The dial is delicate, the cost of getting it wrong is real, and the difference between a good and a bad dedup pipeline shows up in exactly the places where architecture changes and training recipes do not.
      </Prose>
    </div>
  ),
};

export default dataCurationDeduplication;
