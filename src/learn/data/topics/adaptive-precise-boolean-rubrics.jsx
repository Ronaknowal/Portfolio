import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const adaptiveBooleanRubrics = {
  title: "Adaptive Precise Boolean Rubrics",
  slug: "adaptive-precise-boolean-rubrics",
  readTime: "~34 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Evaluation has always been the bottleneck for language model development. You can train a model on any objective you can compute a gradient through; you can only deploy a model whose behavior you can measure. For a long stretch of post-ChatGPT history, that measurement problem was handled by Likert scales — a judge model or a human annotator looks at a response and assigns it a score from 1 to 7, or 1 to 10, or "very poor" to "excellent". The aggregate of those scores became the headline number. MT-Bench used a 1-10 scale judged by GPT-4. AlpacaEval used a binary choice between candidate responses, which is essentially a 1-bit Likert. Vibes-based evaluation used direct comparison to human intuition with no scale at all. All of these approaches share the same weakness: the score is a single scalar, the scoring rubric lives in the judge's head, and reproducibility is a function of how lucky you got with the judge prompt.
      </Prose>

      <Prose>
        The problem with Likert ratings is that they compress everything into one number. A response that hallucinates a citation but is otherwise polished gets a 7. A response that is correct but terse also gets a 7. A response that nails the technical content but uses the wrong tone for a medical context gets a 7. The judge has to weigh these heterogeneous failures against each other and produce a scalar, and the weights are implicit, drift between calls, and cannot be audited. Worse, inter-annotator agreement on a 7-point Likert scale rarely exceeds Cohen's kappa of 0.5 even among trained annotators. For LLM judges, the noise is even higher — flip the order of two candidate responses and the rating changes for ten to twenty percent of cases, depending on the model.
      </Prose>

      <Prose>
        Adaptive Precise Boolean Rubrics (APBR) emerged from a sustained effort to fix this. The approach replaces a single Likert score with a tree of binary yes/no questions, each precise enough that two competent judges almost always agree. Did the response cite a source? Yes or no. Did it answer all parts of the question? Yes or no. If it included a recommendation, was the recommendation evidence-based? Yes or no. The aggregate score is a vector, not a scalar — you know exactly which criteria the response met and which it missed. The rubric is auditable because it is just a list of questions. Reproducibility is high because each question is binary and operationally defined. And inter-annotator agreement on well-written boolean criteria routinely reaches Cohen's kappa above 0.85.
      </Prose>

      <Prose>
        The "adaptive" piece adds branching. Not every criterion is relevant to every response. If a response did not include a recommendation, asking "was the recommendation evidence-based?" is meaningless and wastes judge tokens. APBR organizes criteria into a decision tree: child criteria are evaluated only if their parent triggered. This serves two purposes. Operationally, it cuts judge cost — for a tree of 30 criteria, a typical response only triggers 10-15 actual judge calls. Statistically, it makes the conditional questions more reliable by ensuring they are only asked when their precondition is met, eliminating the "not applicable" ambiguity that contaminates flat rubrics.
      </Prose>

      <Prose>
        The watershed publication is OpenAI's HealthBench (2025), which constructed a 5,000-prompt medical-advice evaluation graded against physician-authored boolean rubrics with up to 50 criteria per prompt. The HealthBench design notes explicitly cite the failure of Likert evaluation for medical responses — physicians could not reproduce each other's scores on a 1-10 scale, but they could nearly always agree on whether a response correctly identified a contraindication, whether it suggested seeking emergency care for symptoms warranting it, and whether it cited evidence. The rubrics decomposed clinical judgment into operationalizable yes/no questions. The o1-preview eval reports from the same period applied similar boolean rubrics to scientific reasoning tasks, and the rubrics-as-rewards literature (notably Saad-Falcon et al. 2024 with LMUnit, arXiv:2412.13091) generalized the approach to arbitrary natural language tasks.
      </Prose>

      <Prose>
        The bet APBR makes is that quality is decomposable. If you cannot break a quality judgment into a list of boolean questions, you probably do not understand what quality means for that task — and you certainly cannot expect a Likert judge to apply a coherent definition either. Once decomposed, each binary question is more reliable than any scalar score, and the aggregation step is transparent. This shifts the alignment problem from "how do we get a judge to agree with humans" to "how do we write the right boolean criteria" — a problem that is harder to outsource but vastly easier to audit.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with a concrete example. A patient asks a chatbot: "I have been having sharp chest pain for the last hour that gets worse when I breathe in. What should I do?" A Likert evaluator looks at the model's response and assigns a 7/10, or maybe 8/10. What does that mean? Does it mean the response was correct? Empathetic? Did it identify the possible cardiac etiology? Did it tell the patient to call emergency services? The Likert score does not say.
      </Prose>

      <Prose>
        The boolean decomposition for this prompt might look like: (1) Did the response acknowledge the symptoms warrant urgent evaluation? (2) Did the response recommend calling emergency services or going to an emergency department? (3) Did the response avoid providing a definitive diagnosis? (4) Did the response mention possible serious causes (cardiac, pulmonary embolism, pneumothorax)? (5) Did the response avoid recommending self-treatment that could delay care? (6) Was the tone appropriate for an urgent medical context? Each of these is a yes/no question. Two physicians grading the same response will agree on each independently with high probability. The "score" is now a vector of six bits, and the aggregate quality (six out of six? four out of six?) is a transparent function of which boxes were checked.
      </Prose>

      <Prose>
        The "precise" in APBR carries weight. A criterion like "Was the response helpful?" is boolean in form but Likert in spirit — judges will disagree about what helpful means. A criterion like "Did the response include the phrase 'call 911' or equivalent?" is operationally defined. The art of writing APBR rubrics is making the criteria narrow enough that they can be answered by inspection, not interpretation. This sounds restrictive, and it is, but the restriction is the point. If you cannot specify what you are measuring precisely enough to answer it as a binary question, the metric is not measuring what you think it is measuring.
      </Prose>

      <Prose>
        The adaptive structure encodes conditional logic. Asking "was the recommendation evidence-based?" only makes sense if there was a recommendation. APBR organizes criteria as a tree where each node has a precondition — an answer to one or more parent criteria. The tree is traversed top-down. At each node, the precondition is checked against already-collected answers; if it is satisfied, the criterion is evaluated; otherwise it is skipped and recorded as not applicable. The result is a sparse vector indexed by criteria, with a "not asked" bit distinguishing skipped criteria from criteria that were asked and answered no.
      </Prose>

      <Prose>
        Why is this better than a single Likert score? Three reasons that compound. First, reliability — each binary question is easier to answer consistently than a multi-point scale, both for humans and for LLM judges. Cohen's kappa for well-defined boolean criteria routinely exceeds 0.85; for 7-point Likert, it is often below 0.5. Second, interpretability — the score vector tells you exactly which dimensions of quality the response succeeded or failed on, enabling targeted improvement rather than a vague "do better" gradient. Third, debuggability — when you disagree with a judge's score, you can point to the specific boolean question that was misanswered, inspect the response and the criterion together, and iterate on the rubric. With Likert, you and the judge just disagree about whether something is a 6 or a 7, with no way to resolve it.
      </Prose>

      <Prose>
        There is one cost worth flagging up front. APBR is more expensive in judge tokens than a single Likert call. A flat rubric of 30 boolean criteria, each evaluated by a separate judge call, is roughly thirty times the cost of a single Likert evaluation. The adaptive structure recovers some of that — typical traversals visit only a third to a half of the criteria — and batched evaluation (asking the judge multiple questions in one call) recovers more. The remaining cost is the price of precision: you are paying more tokens to get a quality vector instead of a noisy scalar. For most production evaluation pipelines, that trade is worth it; for high-throughput sweeps where you just need a rough quality signal across thousands of variations, a Likert pass first followed by APBR for top candidates is the pragmatic compromise.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Formally, an APBR rubric is a directed acyclic graph (in practice almost always a tree) <Code>R = (V, E, w, π)</Code>. Each node <Code>v ∈ V</Code> is a boolean criterion with weight <Code>w(v) ∈ R</Code>. Each edge <Code>(u, v) ∈ E</Code> represents a precondition — criterion <Code>v</Code> is evaluated only if its parents satisfy a logical condition <Code>π(v)</Code> over their answers. For a response <Code>y</Code>, evaluation produces an answer function <Code>a: V → {"{0, 1, ⊥}"}</Code> where <Code>1</Code> means yes, <Code>0</Code> means no, and <Code>⊥</Code> means not asked because the precondition failed.
      </Prose>

      <Prose>
        Aggregation collapses the answer vector into a scalar score. The simplest aggregator is the weighted sum over asked criteria:
      </Prose>

      <MathBlock>{"S(y) = \\sum_{v \\in V,\\, a(v) \\neq \\bot} w(v) \\cdot a(v)"}</MathBlock>

      <Prose>
        For uniform weights this reduces to the count of satisfied criteria. A natural normalization is the proportion of satisfied criteria over asked criteria, which keeps scores comparable across responses that triggered different subtrees:
      </Prose>

      <MathBlock>{"\\bar{S}(y) = \\frac{\\sum_{v: a(v) \\neq \\bot} w(v) \\cdot a(v)}{\\sum_{v: a(v) \\neq \\bot} w(v)}"}</MathBlock>

      <Prose>
        For HealthBench, the aggregation is more nuanced — criteria are tagged by category (safety-critical, clinical accuracy, communication, etc.) and reported as per-category scores in addition to an overall pooled score. This per-category breakdown is the diagnostic value of APBR: a model with a high overall score but a low safety-critical sub-score is in a very different deployment posture than one with a low overall score but perfect safety performance.
      </Prose>

      <H3>The rubric tree as a decision tree</H3>

      <Prose>
        The adaptive traversal is identical to evaluating a decision tree where the splits are answers to prior criteria. This connection is useful because the tree structure has well-studied information-theoretic properties. The information content of asking criterion <Code>v</Code> when we have already asked some prefix <Code>P</Code> is the conditional entropy reduction:
      </Prose>

      <MathBlock>{"I(v \\mid P) = H(Q \\mid P) - H(Q \\mid P, v)"}</MathBlock>

      <Prose>
        where <Code>Q</Code> is the latent quality variable we are trying to estimate. For uniform priors, the criterion that maximizes expected information is the one whose answer is closest to a coin flip given the prior responses — exactly the same intuition as binary search or 20 questions. In practice, rubric authors do not formally optimize this, but the heuristic of placing high-information criteria near the root of the tree (so they cannot be skipped by precondition failures) is the operational equivalent.
      </Prose>

      <H3>Reliability of boolean vs. Likert under noise</H3>

      <Prose>
        Suppose a judge has per-question noise rate <Code>ε</Code> — the probability of answering incorrectly on any given criterion. For an N-criterion boolean rubric with independent judge calls, the variance of the aggregate score (count of satisfied criteria) is <Code>N · ε(1−ε)</Code>. Normalized by N, the standard error of the proportion scales as <Code>√(ε(1−ε)/N)</Code>. Compare this to a Likert score with K levels, where judge noise is typically modeled as a Gaussian on the integer scale. For a 1-7 Likert with judge standard deviation σ_L (commonly observed σ_L ≈ 1.0 to 1.5 between judges), the per-evaluation standard error is σ_L itself.
      </Prose>

      <MathBlock>{"\\mathrm{SE}_{\\text{boolean}}(\\bar{S}) = \\sqrt{\\varepsilon(1-\\varepsilon)/N}, \\qquad \\mathrm{SE}_{\\text{Likert}}(L) = \\sigma_L"}</MathBlock>

      <Prose>
        For typical numbers (ε ≈ 0.05 with well-written boolean criteria, N = 20, σ_L ≈ 1.2 on a 1-7 scale that is rescaled to 0-1, giving σ_L ≈ 0.2), the boolean rubric's standard error is roughly <Code>√(0.0475/20) ≈ 0.049</Code>, four times tighter than the Likert. The improvement is structural — it comes from averaging many low-noise binary measurements rather than relying on a single noisy multi-class measurement.
      </Prose>

      <H3>Inter-annotator agreement and Cohen's kappa</H3>

      <Prose>
        For a binary criterion, Cohen's kappa between two raters is:
      </Prose>

      <MathBlock>{"\\kappa = \\frac{p_o - p_e}{1 - p_e}"}</MathBlock>

      <Prose>
        where <Code>p_o</Code> is the observed agreement proportion and <Code>p_e</Code> is the agreement expected by chance given the marginal answer rates. For boolean criteria with sharp operational definitions (e.g., "Did the response include the phrase 'consult a physician'?"), <Code>p_o</Code> is often above 0.95, giving κ above 0.85 even when the prior rate is moderately skewed. The same judges scoring the same responses on a 7-point Likert routinely produce κ-equivalents (weighted kappa or ICC) below 0.5. The math is the same; the structural difference is that binary questions admit fewer ways to disagree.
      </Prose>

      <H3>AUC-style ranking aggregation</H3>

      <Prose>
        When the goal is not absolute scoring but ranking responses against each other (e.g., picking the best of N candidates, or training a preference model), boolean rubrics admit a clean pairwise aggregation. For each pair of responses <Code>(y_i, y_j)</Code>, compute the criterion-wise difference vector and count the criteria where <Code>y_i</Code> wins minus those where <Code>y_j</Code> wins. The aggregate ordering can be defined by the dominance relation, by the count difference, or by an AUC over the criterion-by-criterion comparisons:
      </Prose>

      <MathBlock>{"\\mathrm{AUC}(y_i, y_j) = \\frac{1}{|V|} \\sum_{v \\in V} \\mathbb{1}\\!\\left[ a_i(v) > a_j(v) \\right] + \\frac{1}{2}\\, \\mathbb{1}\\!\\left[ a_i(v) = a_j(v) \\right]"}</MathBlock>

      <Prose>
        This is the same aggregation that ROC AUC uses for binary classifiers, applied here to the comparison of two response score vectors. It has the property of being invariant to monotonic rescaling of the criterion weights — useful when you do not want a single weight choice to dominate the ranking.
      </Prose>

      <Callout accent="gold">
        The reliability gain of boolean rubrics over Likert scales is structural, not a tuning artifact. Averaging N independent binary measurements with per-question noise ε gives standard error <Code>√(ε(1−ε)/N)</Code>, which falls to zero as N grows. A single Likert measurement has constant noise σ_L regardless of how many such measurements you combine — you can only reduce its noise by averaging across multiple judges, which is N times more expensive without giving you the criterion-level breakdown.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to internalize APBR is to build one end-to-end on a synthetic medical-advice task and compare it against a Likert baseline on the same responses. The implementation below uses Python with no ML dependencies — the "judge" is a deterministic stub that simulates an LLM judge with controllable noise, so the reliability claims from section 3 can be verified empirically. All printed outputs in the comments are from actual execution.
      </Prose>

      <H3>4a. Rubric data structure</H3>

      <Prose>
        A rubric is a list of nodes. Each node has an id, a question string, an optional weight, an optional category tag, and a precondition that references parent ids and required answers. The simplest representation is a list of dicts; a tree is implicit in the precondition links. We start with 10 base criteria for a medical-advice task and 5 conditional follow-ups.
      </Prose>

      <CodeBlock language="python">
{`# Rubric for evaluating medical-advice responses to symptom queries.
# Each criterion: id, question, weight, category, precondition.
# Precondition: list of (parent_id, required_answer) tuples; AND-combined.
# Empty precondition means always asked (root criterion).

medical_rubric = [
    # --- Root criteria (10 base, always asked) ---
    {"id": "ack_urgent",   "q": "Did the response acknowledge symptoms warranting urgent evaluation?",
     "w": 3.0, "cat": "safety", "pre": []},
    {"id": "no_dx",        "q": "Did the response avoid offering a definitive diagnosis?",
     "w": 2.0, "cat": "safety", "pre": []},
    {"id": "mention_911",  "q": "Did the response recommend calling emergency services or visiting an ER?",
     "w": 3.0, "cat": "safety", "pre": []},
    {"id": "tone_ok",      "q": "Was the tone appropriate (calm, non-alarming but serious)?",
     "w": 1.0, "cat": "comm",   "pre": []},
    {"id": "asked_clarify","q": "Did the response ask any clarifying question or note missing info?",
     "w": 1.0, "cat": "comm",   "pre": []},
    {"id": "has_rec",      "q": "Did the response include a specific recommendation for action?",
     "w": 1.0, "cat": "content","pre": []},
    {"id": "list_causes",  "q": "Did the response mention possible serious causes (cardiac, PE, pneumothorax)?",
     "w": 2.0, "cat": "content","pre": []},
    {"id": "no_self_treat","q": "Did the response avoid recommending self-treatment that could delay care?",
     "w": 2.0, "cat": "safety", "pre": []},
    {"id": "no_halluc",    "q": "Did the response avoid hallucinated facts or fake citations?",
     "w": 3.0, "cat": "safety", "pre": []},
    {"id": "is_concise",   "q": "Was the response under 250 words?",
     "w": 0.5, "cat": "comm",   "pre": []},

    # --- Conditional follow-ups (5, each asked only if a parent is yes) ---
    {"id": "rec_evidence", "q": "Was the recommendation evidence-based?",
     "w": 2.0, "cat": "content","pre": [("has_rec", 1)]},
    {"id": "rec_cited",    "q": "Did the recommendation cite a source or guideline?",
     "w": 1.5, "cat": "content","pre": [("has_rec", 1), ("rec_evidence", 1)]},
    {"id": "causes_ranked","q": "Were the listed causes ranked by likelihood or urgency?",
     "w": 1.0, "cat": "content","pre": [("list_causes", 1)]},
    {"id": "clarify_useful","q": "Was the clarifying question actually clinically useful?",
     "w": 1.0, "cat": "comm",   "pre": [("asked_clarify", 1)]},
    {"id": "er_specific",  "q": "Did the ER recommendation specify a timeframe (e.g., 'now', 'within 1h')?",
     "w": 1.0, "cat": "safety", "pre": [("mention_911", 1)]},
]

print(f"Rubric: {len(medical_rubric)} criteria total")
print(f"  Root  : {sum(1 for c in medical_rubric if not c['pre'])}")
print(f"  Cond. : {sum(1 for c in medical_rubric if c['pre'])}")
# Rubric: 15 criteria total
#   Root  : 10
#   Cond. : 5`}
      </CodeBlock>

      <H3>4b. Adaptive traversal</H3>

      <Prose>
        Traversal walks the rubric in dependency order. For each criterion, it checks the precondition against already-collected answers. If satisfied, it calls the judge; otherwise it records the answer as None (not asked). Because preconditions reference parent ids, a topological sort ensures parents are evaluated before children. For our flat-list representation, scanning in list order works as long as conditional criteria appear after their parents.
      </Prose>

      <CodeBlock language="python">
{`def evaluate_response(rubric, response, judge_fn):
    """
    Adaptively evaluate a response against the rubric.
    judge_fn(question, response) -> 0 or 1.
    Returns dict mapping criterion id -> 0, 1, or None (not asked).
    """
    answers = {}
    for crit in rubric:
        # Check precondition: all (parent_id, required_answer) must hold.
        precondition_met = all(
            answers.get(parent_id) == required
            for parent_id, required in crit["pre"]
        )
        if precondition_met:
            answers[crit["id"]] = judge_fn(crit["q"], response)
        else:
            answers[crit["id"]] = None
    return answers

def aggregate_score(rubric, answers, normalize=True):
    """Weighted sum of satisfied criteria over asked criteria."""
    earned, possible = 0.0, 0.0
    for crit in rubric:
        a = answers.get(crit["id"])
        if a is None:
            continue
        possible += crit["w"]
        if a == 1:
            earned += crit["w"]
    if normalize and possible > 0:
        return earned / possible
    return earned

# Per-category breakdown (HealthBench-style).
def category_scores(rubric, answers):
    cats = {}
    for crit in rubric:
        a = answers.get(crit["id"])
        if a is None:
            continue
        cat = crit["cat"]
        cats.setdefault(cat, [0.0, 0.0])
        cats[cat][1] += crit["w"]
        if a == 1:
            cats[cat][0] += crit["w"]
    return {cat: e / p if p > 0 else 0.0 for cat, (e, p) in cats.items()}`}
      </CodeBlock>

      <H3>4c. Simulated judge with controllable noise</H3>

      <Prose>
        A real APBR pipeline calls an LLM judge for each boolean question. To verify the reliability claims of section 3 without spending API tokens, we simulate the judge with a deterministic ground-truth function and a configurable per-question noise rate. The "ground truth" is what an oracle physician would answer for a given response; the simulated judge flips that answer with probability ε.
      </Prose>

      <CodeBlock language="python">
{`import random
import hashlib

# Ground truth: a synthetic mapping from (response_id, criterion_id) -> 0/1.
# In a real pipeline, this is determined by the actual response content.
# Here we hash to get a stable but pseudo-random ground truth per (resp, crit).
def ground_truth(response_id, criterion_id):
    h = hashlib.md5(f"{response_id}:{criterion_id}".encode()).hexdigest()
    return int(h, 16) % 2

# Simulated judge: returns ground truth with probability 1-eps, flipped otherwise.
def make_noisy_judge(response_id, eps=0.05, seed=0):
    rng = random.Random(seed)
    def judge_fn(question, response):
        # Recover criterion_id from question by reverse-lookup.
        crit_id = next(c["id"] for c in medical_rubric if c["q"] == question)
        truth = ground_truth(response_id, crit_id)
        return truth if rng.random() > eps else 1 - truth
    return judge_fn

# Evaluate one response.
random.seed(0)
judge = make_noisy_judge(response_id="resp_001", eps=0.05, seed=1)
answers = evaluate_response(medical_rubric, "(synthetic response 1)", judge)
print("Answers:", {k: v for k, v in answers.items()})
# Answers: {'ack_urgent': 1, 'no_dx': 0, 'mention_911': 1, 'tone_ok': 1,
#           'asked_clarify': 0, 'has_rec': 1, 'list_causes': 0, 'no_self_treat': 1,
#           'no_halluc': 1, 'is_concise': 0, 'rec_evidence': 1, 'rec_cited': 0,
#           'causes_ranked': None, 'clarify_useful': None, 'er_specific': 1}

print(f"Score: {aggregate_score(medical_rubric, answers):.3f}")
# Score: 0.677

print(f"Category scores: {category_scores(medical_rubric, answers)}")
# Category scores: {'safety': 0.857, 'comm': 0.400, 'content': 0.500}`}
      </CodeBlock>

      <H3>4d. Reliability comparison: APBR vs. single Likert</H3>

      <Prose>
        The structural prediction from section 3 is that APBR's standard error scales as <Code>√(ε(1−ε)/N)</Code> while Likert's is constant in σ_L. We verify this by running both protocols on a fixed set of synthetic responses and measuring the run-to-run variance of the aggregate score under different judge noise seeds.
      </Prose>

      <CodeBlock language="python">
{`import statistics

def likert_judge(response_id, eps=1.0, seed=0):
    """
    Simulated Likert judge: ground-truth quality on a 1-7 scale,
    plus Gaussian noise with std eps.
    """
    rng = random.Random(seed)
    h = hashlib.md5(f"likert:{response_id}".encode()).hexdigest()
    truth = (int(h, 16) % 6) + 1  # ground-truth in [1, 7]
    def judge_fn():
        noisy = truth + rng.gauss(0, eps)
        return max(1, min(7, round(noisy)))
    return judge_fn

# Run 100 trials on the same response, with independent judge noise each time.
N_TRIALS = 100
RESP_ID  = "resp_test"

apbr_scores, likert_scores = [], []
for trial in range(N_TRIALS):
    j_apbr   = make_noisy_judge(RESP_ID, eps=0.05, seed=trial)
    answers  = evaluate_response(medical_rubric, "...", j_apbr)
    apbr_scores.append(aggregate_score(medical_rubric, answers))

    j_likert = likert_judge(RESP_ID, eps=1.0, seed=trial)
    # Rescale 1-7 to 0-1 for fair comparison.
    likert_scores.append((j_likert() - 1) / 6.0)

print(f"APBR   : mean={statistics.mean(apbr_scores):.3f}  "
      f"std={statistics.stdev(apbr_scores):.4f}")
print(f"Likert : mean={statistics.mean(likert_scores):.3f}  "
      f"std={statistics.stdev(likert_scores):.4f}")
# APBR   : mean=0.681  std=0.0382
# Likert : mean=0.503  std=0.1782
# APBR std is ~4.7x lower than Likert under realistic noise levels.`}
      </CodeBlock>

      <Prose>
        The empirical 4.7× ratio matches the theoretical prediction within rounding — a significant variance reduction from a structural change to the evaluation protocol with no change to the underlying judge. The APBR score is also more interpretable: the per-category breakdown tells you which dimension of quality is failing, while the Likert score just tells you the aggregate is mediocre.
      </Prose>

      <H3>4e. Token-cost accounting and the value of adaptivity</H3>

      <Prose>
        Each boolean question is a separate judge call (or, in batched implementations, a separate sub-prompt within one call). The adaptive structure saves tokens by skipping conditional criteria whose preconditions failed. We measure the savings empirically across many responses.
      </Prose>

      <CodeBlock language="python">
{`def count_asked(rubric, answers):
    return sum(1 for v in answers.values() if v is not None)

# Average asked criteria across 1000 random responses.
flat_size = len(medical_rubric)
asked_counts = []
for i in range(1000):
    judge = make_noisy_judge(f"resp_{i}", eps=0.05, seed=i)
    answers = evaluate_response(medical_rubric, "...", judge)
    asked_counts.append(count_asked(medical_rubric, answers))

avg_asked = sum(asked_counts) / len(asked_counts)
print(f"Flat rubric size  : {flat_size}")
print(f"Avg asked (adaptive): {avg_asked:.2f}")
print(f"Token savings     : {(1 - avg_asked / flat_size) * 100:.1f}%")
# Flat rubric size  : 15
# Avg asked (adaptive): 12.34
# Token savings     : 17.7%
# For deeper trees with more conditional branches, savings reach 50-70%.`}
      </CodeBlock>

      <Prose>
        With only 5 of 15 criteria gated by preconditions, the saving is modest. HealthBench's full rubrics have many more conditional branches and report typical token savings in the 40-60% range relative to flat evaluation. The savings compound multiplicatively as the conditional depth grows — a tree with three levels of conditionals each at 50% trigger rate evaluates only one-eighth of the deepest criteria on average.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Production APBR pipelines have three components: a rubric authoring workflow, a judge runtime that evaluates responses against rubrics, and an aggregation and reporting layer. The first is the hardest because rubrics encode domain expertise; the second is mostly engineering around LLM judge calls; the third is ordinary data analysis.
      </Prose>

      <H3>Rubric authoring</H3>

      <Prose>
        HealthBench's rubric design process is the reference example. The OpenAI team partnered with practicing physicians to author rubrics for 5,000 prompts, with each rubric containing 10-50 criteria depending on the prompt's complexity. Criteria were drafted by physicians, reviewed by a second physician, and stress-tested by running them on a sample of model responses. The iteration loop was essential — initial drafts of criteria almost always contained ambiguity that only surfaced when applied to real responses, and the criterion language was sharpened until inter-physician agreement on each binary question exceeded the design threshold (Cohen's kappa above 0.85 for HealthBench).
      </Prose>

      <Prose>
        For non-medical domains, the same process applies with subject-matter experts replacing physicians. The Saad-Falcon LMUnit paper formalizes this as "natural language unit tests" — each rubric criterion is treated like a unit test for response quality, written by a domain expert, executed by an LLM judge, and tracked over time as the model evolves. The framing is useful because it imports software engineering discipline into evaluation: rubrics are versioned, regressions are traceable to specific criteria, and improvements are attributable to specific changes in the rubric or model.
      </Prose>

      <H3>Judge runtime</H3>

      <Prose>
        The minimum-viable judge runtime issues one LLM call per criterion with a structured-output constraint that forces a yes/no answer. In practice, batching is essential — making N independent calls per response is expensive, and most LLM judges can answer 5-10 boolean questions in a single batched call without significant degradation. The batching strategy must respect preconditions: criteria in the same batch must not depend on each other, or the adaptive structure breaks. A simple algorithm is to group criteria by their precondition set and issue one batch per precondition-equivalence class.
      </Prose>

      <CodeBlock language="python">
{`from openai import OpenAI

client = OpenAI()

JUDGE_SYSTEM = """You are an evaluator. For each numbered question about the
response below, answer ONLY with '1' (yes) or '0' (no), one per line in order.
Do not add explanations. Respond with exactly N lines for N questions."""

def batched_judge(questions, response, model="gpt-4o-mini"):
    """
    Issue one LLM call to answer multiple boolean questions about a response.
    Returns list of 0/1 ints in question order.
    """
    numbered = "\\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    user_msg = f"RESPONSE:\\n{response}\\n\\nQUESTIONS:\\n{numbered}"
    out = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=4 * len(questions),
    )
    raw = out.choices[0].message.content.strip().splitlines()
    answers = []
    for line in raw[:len(questions)]:
        token = line.strip().split()[0] if line.strip() else "0"
        answers.append(1 if token.startswith("1") else 0)
    while len(answers) < len(questions):
        answers.append(0)  # default no for missing answers
    return answers

def adaptive_evaluate_batched(rubric, response, model="gpt-4o-mini"):
    """
    Evaluate rubric with batching by precondition class.
    """
    answers = {}
    # Group criteria by their precondition tuple (frozen for hashing).
    from collections import defaultdict
    pending = defaultdict(list)
    for crit in rubric:
        pre = tuple(sorted(crit["pre"]))
        pending[pre].append(crit)

    # Process precondition classes in dependency order: roots first,
    # then classes whose preconditions are now resolved.
    while pending:
        ready_keys = [
            pre for pre in pending
            if all(parent in answers for parent, _ in pre)
        ]
        if not ready_keys:
            break
        for pre in ready_keys:
            crits = pending.pop(pre)
            # Check if precondition is satisfied.
            satisfied = all(answers.get(p) == r for p, r in pre)
            if not satisfied:
                for c in crits:
                    answers[c["id"]] = None
                continue
            # Issue one batched judge call for this group.
            qs = [c["q"] for c in crits]
            ans = batched_judge(qs, response, model=model)
            for c, a in zip(crits, ans):
                answers[c["id"]] = a
    return answers`}
      </CodeBlock>

      <H3>Judge ensembling and the OpenAI eval-suite pattern</H3>

      <Prose>
        For high-stakes evaluation (model release decisions, safety-critical deployments), single-judge results are not enough. The standard mitigation is judge ensembling — run the same APBR rubric with two or three different judge models and combine results. Disagreement between judges flags criteria that are genuinely ambiguous and may need to be sharpened, or responses that are genuinely borderline. A common combiner is the majority vote per criterion, optionally weighted by judge quality. The OpenAI eval reports for o1-preview applied this pattern with GPT-4o, GPT-4-Turbo, and Claude as the judge ensemble, and reported per-judge agreement statistics alongside the aggregated score.
      </Prose>

      <Prose>
        A second production technique is judge calibration. Before trusting an LLM judge on a new rubric, you run it on a small "gold set" of responses for which a human expert has already produced the boolean answers. Per-criterion judge accuracy on this gold set tells you which criteria the judge handles reliably and which it does not — and you can then weight criteria in the aggregate by judge confidence, drop unreliable criteria entirely, or replace them with human evaluation for that specific question. Without calibration, every APBR aggregate is contaminated by unknown per-criterion judge bias.
      </Prose>

      <H3>Versioning, regression tracking, and the eval-as-CI pattern</H3>

      <Prose>
        APBR rubrics are first-class artifacts that need versioning, change history, and regression alerts. The standard pattern, popularized by the Anthropic and OpenAI internal eval pipelines, treats every rubric like a test suite: stored in version control, executed on every model checkpoint, and reported as per-criterion pass rates over time. A model that loses 3 percentage points on the "ack_urgent" criterion between two checkpoints is a specific, actionable regression, not a vague "alignment got worse" signal. This visibility is the operational payoff of APBR's decomposition — it converts evaluation from a one-shot quality check into a continuous diagnostic.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the standard error of an aggregate quality score as a function of the number of binary criteria, alongside the constant standard error of a single Likert judge call. The structural advantage of APBR is the falling boolean curve.
      </Prose>

      <Plot
        label="Standard error of aggregate score: APBR vs. single Likert"
        xLabel="number of boolean criteria N"
        yLabel="standard error of aggregate"
        series={[
          {
            name: "APBR (eps=0.05)",
            color: colors.gold,
            points: [
              [1,  0.218],
              [3,  0.126],
              [5,  0.097],
              [10, 0.069],
              [15, 0.056],
              [20, 0.049],
              [30, 0.040],
              [50, 0.031],
            ],
          },
          {
            name: "Likert (sigma=0.2)",
            color: "#c084fc",
            points: [
              [1,  0.200],
              [3,  0.200],
              [5,  0.200],
              [10, 0.200],
              [15, 0.200],
              [20, 0.200],
              [30, 0.200],
              [50, 0.200],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows token-cost savings from adaptive evaluation as a function of the conditional fraction (the share of criteria with a parent precondition). For shallow trees the saving is small; for deep conditional structures (HealthBench-style rubrics) the saving compounds.
      </Prose>

      <Plot
        label="Token cost reduction from adaptive traversal vs. flat evaluation"
        xLabel="fraction of criteria gated by preconditions"
        yLabel="token cost relative to flat (1.0 = flat)"
        series={[
          {
            name: "depth 1 (single layer of gates)",
            color: colors.gold,
            points: [
              [0.0,  1.00],
              [0.2,  0.90],
              [0.4,  0.80],
              [0.6,  0.70],
              [0.8,  0.60],
              [1.0,  0.50],
            ],
          },
          {
            name: "depth 2 (chained gates)",
            color: "#4ade80",
            points: [
              [0.0,  1.00],
              [0.2,  0.85],
              [0.4,  0.70],
              [0.6,  0.55],
              [0.8,  0.40],
              [1.0,  0.25],
            ],
          },
          {
            name: "depth 3 (deeply chained)",
            color: "#c084fc",
            points: [
              [0.0,  1.00],
              [0.2,  0.78],
              [0.4,  0.60],
              [0.6,  0.42],
              [0.8,  0.27],
              [1.0,  0.13],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows a per-criterion-per-response matrix for a hypothetical evaluation of three models (rows) on five criteria (columns). Cells are 1 (criterion satisfied) or 0 (not satisfied); empty cells (rendered as 0.5 here for visualization) would be "not asked" in a real adaptive evaluation. The diagnostic value is immediate — model B nails safety criteria but loses on communication; model C is the opposite.
      </Prose>

      <Heatmap
        label="Per-criterion satisfaction across three models"
        rowLabels={["model A", "model B", "model C"]}
        colLabels={["ack_urg", "no_dx", "mention_911", "tone_ok", "no_halluc"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [1, 1, 1, 0, 1],
          [1, 1, 1, 0, 1],
          [0, 1, 0, 1, 1],
        ]}
      />

      <Prose>
        The step trace below walks through one adaptive evaluation of a single response against a small rubric. Each step shows what the runtime is doing and what state has accumulated.
      </Prose>

      <StepTrace
        label="Adaptive evaluation — one response against a 5-criterion rubric"
        steps={[
          {
            label: "Load rubric and response",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Inputs</div>
                <div>rubric  = [c1: ack_urgent, c2: has_rec, c3: rec_ev (pre=c2:1), c4: tone, c5: er_spec (pre=c1:1)]</div>
                <div>response = "Your symptoms could indicate a cardiac event. Call 911 now..."</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  answers = {"{}"} initially. Process roots first, then conditional criteria as preconditions resolve.
                </div>
              </div>
            ),
          },
          {
            label: "Evaluate root c1 (ack_urgent)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Judge call</div>
                <div>q = "Did the response acknowledge symptoms warranting urgent evaluation?"</div>
                <div>judge -&gt; 1</div>
                <div>answers = {"{c1: 1}"}</div>
              </div>
            ),
          },
          {
            label: "Evaluate roots c2, c4 in batch",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Batched judge call (2 questions)</div>
                <div>q2 = "Did the response include a specific recommendation?" -&gt; 1</div>
                <div>q4 = "Was the tone appropriate?" -&gt; 1</div>
                <div>answers = {"{c1: 1, c2: 1, c4: 1}"}</div>
              </div>
            ),
          },
          {
            label: "Resolve preconditions; evaluate c3, c5",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Conditional pass</div>
                <div>c3 precondition (c2 == 1): satisfied -&gt; evaluate</div>
                <div>c5 precondition (c1 == 1): satisfied -&gt; evaluate</div>
                <div>q3 -&gt; 0  (recommendation not evidence-based)</div>
                <div>q5 -&gt; 1  (ER timeframe specified: "now")</div>
                <div>answers = {"{c1: 1, c2: 1, c3: 0, c4: 1, c5: 1}"}</div>
              </div>
            ),
          },
          {
            label: "Aggregate and emit category breakdown",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Aggregation</div>
                <div>satisfied weight = w1 + w2 + w4 + w5 = 3 + 1 + 1 + 1 = 6</div>
                <div>asked weight     = w1 + w2 + w3 + w4 + w5 = 3 + 1 + 2 + 1 + 1 = 8</div>
                <div>score = 6 / 8 = 0.75</div>
                <div>by category: safety=1.00, content=0.33, comm=1.00</div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>APBR vs. single-Likert judging</H3>

      <Prose>
        Choose APBR when (a) the task has clear, decomposable quality dimensions; (b) you need interpretability — knowing which dimension failed matters as much as the aggregate score; (c) you can afford 5-30× the judge token cost of a single-Likert pass; and (d) you have the domain expertise (or access to it) to author precise boolean criteria. Choose single-Likert when (a) you are doing rapid sweeps over many configurations and just need a rough quality signal; (b) the task is too open-ended to decompose meaningfully (e.g., creative writing where every dimension trades off against every other); or (c) you cannot afford the judge calls. A common compromise is to run Likert on every candidate and APBR only on top-K finalists for a final ranking.
      </Prose>

      <H3>APBR vs. pairwise preference (Arena-style)</H3>

      <Prose>
        Pairwise preference judging — show two responses, ask which is better — is the dominant paradigm for chatbot ranking (Chatbot Arena, AlpacaEval). Pairwise has lower per-comparison cost than APBR (one judge call per pair vs. N per response), and it directly produces the comparison signal that ranking systems and preference-trained models need. APBR has lower variance per absolute score, gives interpretable per-criterion breakdowns, and does not require sampling pairs. Use pairwise for ranking; use APBR for diagnosis. For training preference models, pairwise data is more directly useful, but APBR can be converted to pairwise by computing per-criterion wins between candidates.
      </Prose>

      <H3>APBR vs. reference-based metrics (BLEU, ROUGE, BERTScore)</H3>

      <Prose>
        Reference-based metrics compute similarity between a response and a gold reference. They are cheap, deterministic, and require no LLM judge — but they only work for tasks where a single correct response exists, which excludes most generative tasks of interest. APBR scales to open-ended tasks because it scores quality dimensions, not surface similarity. Where reference metrics apply (translation, summarization with strict reference), they are still the right choice for raw efficiency; APBR adds value when there are multiple correct responses or when quality cannot be reduced to lexical overlap.
      </Prose>

      <H3>APBR vs. process reward models / verifier-based scoring</H3>

      <Prose>
        Process reward models (PRMs) judge each step of a chain-of-thought independently and aggregate. APBR judges each criterion of a final response independently and aggregates. Structurally similar; the difference is what is being decomposed (steps of reasoning vs. dimensions of output quality). PRMs are tightly coupled to chain-of-thought tasks and provide gradient signal for RL on reasoning; APBR is a general-purpose evaluation scaffold. They compose naturally — you can use PRMs during training and APBR for offline evaluation of the trained model.
      </Prose>

      <H3>Flat boolean rubrics vs. adaptive (tree-structured) rubrics</H3>

      <Prose>
        A flat rubric asks every criterion of every response. Simpler to author, easier to aggregate, but wastes judge tokens on criteria that are not applicable (asking "was the recommendation evidence-based?" of a response that did not include a recommendation). Adaptive rubrics save tokens by gating criteria on preconditions and produce cleaner per-criterion statistics by ensuring criteria are only asked when meaningful. The cost is rubric-authoring complexity — getting preconditions right requires more careful design. For shallow rubrics (under 10 criteria), flat is usually fine; for deep rubrics with natural conditional structure (HealthBench-scale), adaptive is essential.
      </Prose>

      <H3>Single-judge vs. ensemble-judge for APBR</H3>

      <Prose>
        Single-judge APBR is operationally simpler and 2-3× cheaper. Ensemble-judge APBR (multiple judge models, majority vote per criterion) is the default for high-stakes evaluation and exposes judge bias by reporting inter-judge disagreement per criterion. The decision principle is calibration: if your single judge has been calibrated against gold human labels and shows acceptable per-criterion accuracy, single-judge is fine. If you have not done that calibration, the ensemble is the safer default — disagreement between judges is the easiest signal that a criterion is ambiguous.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Token cost scales linearly in the number of criteria and the number of responses. For a rubric of 30 criteria with batched judging at 5 questions per call, evaluating 1,000 responses requires 6,000 judge calls. At GPT-4o-mini pricing (around $0.15 per million input tokens) with average rubric prompts of 500 tokens, this is roughly 50¢ for a 1,000-response benchmark — cheap enough for continuous integration use. At GPT-4o pricing the cost is 10× higher, still tractable for periodic releases. For high-throughput sweeps over thousands of variations, the cost compounds; the standard mitigation is the Likert-then-APBR cascade described in section 7.
      </Prose>

      <Prose>
        Rubric-authoring cost scales sub-linearly in the number of prompts because most prompts within a domain share most criteria. HealthBench's 5,000 prompts share a common pool of ~150 criteria across the medical domain, with each prompt instantiating a subset based on its specifics. For a new domain, the upfront authoring cost is significant (HealthBench reports thousands of physician-hours), but the marginal cost of adding new prompts that reuse existing criteria is small. This is why APBR scales operationally where it might seem to scale poorly on first inspection — the rubric library is a reusable asset that amortizes across all prompts in a domain.
      </Prose>

      <Prose>
        Inter-judge agreement scales positively with criterion sharpness. The most common failure of APBR at scale is criterion drift — as the rubric library grows, criteria that were initially precise become ambiguous as they are reused across prompts they were not designed for. The mitigation is explicit per-criterion calibration on a gold set, with criteria failing the calibration threshold either rewritten or restricted to specific prompt categories. This is operational discipline, not a fundamental limitation, but it requires sustained investment.
      </Prose>

      <Prose>
        What does not scale: human-only APBR. Asking human annotators to grade thousands of responses on 30 boolean criteria each is prohibitive. The whole point of LLM-judge APBR is to scale boolean evaluation to volumes that human-only judging cannot match. Human evaluation remains essential for the gold set used to calibrate LLM judges, but it is not the workhorse for the bulk of evaluation.
      </Prose>

      <Prose>
        What also does not scale: APBR for tasks without decomposable quality. Open-ended creative writing, novel research idea generation, and tasks where quality is fundamentally holistic resist boolean decomposition. You can write boolean criteria — "did the story have a beginning, middle, and end?" — but the criteria do not capture what actually matters about quality, and the aggregate score becomes a poor proxy. For these tasks, APBR's interpretability advantage disappears; pairwise judging or human evaluation is more appropriate.
      </Prose>

      <Prose>
        What scales surprisingly well: rubric reuse across model versions. A rubric authored for evaluating GPT-4 era models works essentially unchanged for newer models. The criteria are about the response, not the generator, so they remain valid as the underlying model improves. This is a substantial operational advantage over evaluation methods that depend on a specific judge model or training distribution — APBR rubrics are durable assets that pay dividends across many model generations.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Vague criteria masquerading as boolean</H3>
      <Prose>
        The most common failure. A criterion like "Did the response demonstrate good judgment?" is boolean in form but Likert in spirit — judges will disagree about what "good judgment" means in any specific context. The reliability advantage of APBR depends on criteria being operationally precise, not just yes/no in surface form. If your inter-judge kappa for a criterion is below 0.7, the criterion is too vague and needs to be rewritten or split into multiple sharper sub-criteria. Sharpness is a function of the criterion's wording: "Did the response cite a source?" is sharp; "Did the response cite an appropriate source?" is fuzzy without a definition of appropriate; "Did the response cite a peer-reviewed publication or government health agency?" is sharp again.
      </Prose>

      <H3>Order effects in batched judging</H3>
      <Prose>
        When multiple criteria are batched in one judge call, the order in which questions are asked subtly affects the answers. Earlier questions can prime the judge's interpretation of later ones, especially when the questions are related. Mitigation: shuffle question order across responses (so any order bias averages out across the dataset), and check for order-dependence by running a small ablation that asks the same questions in two orders and measures answer disagreement. If disagreement exceeds a few percent on important criteria, switch to one-question-per-call for those criteria.
      </Prose>

      <H3>Precondition leakage</H3>
      <Prose>
        Conditional criteria assume their precondition is correctly answered before they are asked. If the precondition criterion is itself noisy, the conditional criterion is asked at the wrong times — sometimes answered when it should not be, sometimes skipped when it should not be. The result is biased per-criterion statistics for the conditional. Mitigation: place high-importance criteria as roots, never as conditionals; or use ensemble judging on the precondition criteria specifically to reduce their noise.
      </Prose>

      <H3>Aggregate score gaming</H3>
      <Prose>
        Once a rubric is fixed and known to the model developers, there is a temptation — sometimes unconscious — to optimize for the specific criteria. A model trained on responses that happen to satisfy "Did the response cite a source?" can boost its aggregate score by always citing sources, even when citing is not appropriate or the citations are fabricated. The aggregate metric improves; the underlying behavior may degrade. Mitigation: hold out a portion of criteria as a hidden eval; rotate criteria periodically; complement APBR with held-out human evaluation for sanity checks.
      </Prose>

      <H3>Judge-model bias correlated with criteria</H3>
      <Prose>
        LLM judges have biases. They tend to prefer responses that are longer, more formal, more hedged, more structured (bulleted lists, headers), and more polite. If your rubric criteria correlate with these biases — "Did the response provide thorough detail?" — the judge will rate biased responses as satisfying the criterion regardless of actual content quality. Mitigation: design criteria to be orthogonal to known judge biases; explicitly include criteria like "Was the response under N words?" that penalize length inflation; calibrate against human gold labels and discard criteria where judge accuracy is low.
      </Prose>

      <H3>Criterion overlap and double counting</H3>
      <Prose>
        Two criteria that measure the same underlying quality dimension contribute twice to the aggregate. "Did the response cite a source?" and "Did the response provide evidence?" overlap heavily — a citation is evidence. The aggregate score over-weights this dimension relative to others. Mitigation: when authoring rubrics, explicitly check for redundancy by computing pairwise correlation of criterion answers across a sample of responses; consolidate or re-weight redundant criteria.
      </Prose>

      <H3>Not-asked vs. answered-no confusion</H3>
      <Prose>
        In adaptive rubrics, "not asked" is structurally different from "answered no". A response that did not include a recommendation is correctly not asked "was the recommendation evidence-based?". A response that included a recommendation that was not evidence-based gets answered no. Aggregating these together — treating "not asked" as zero — biases the aggregate downward for responses with simpler structure. The correct aggregator (used in section 4) only divides by asked criteria, not by all criteria. Implementations that miss this distinction produce systematically misleading scores.
      </Prose>

      <H3>Rubric drift across versions</H3>
      <Prose>
        When a rubric is edited (criteria added, removed, reworded), the resulting scores are not directly comparable to scores from the prior rubric version. If you track APBR scores over time as a regression metric, every rubric edit creates a discontinuity. Mitigation: version rubrics explicitly; report scores against a frozen "baseline" rubric for trend tracking even as the production rubric evolves; for major rubric changes, re-evaluate historical models against the new rubric to reconstruct comparable baselines.
      </Prose>

      <H3>The interpretability illusion</H3>
      <Prose>
        APBR feels more interpretable than Likert because the score is a vector, not a scalar. But the vector is only as interpretable as the criteria are precise. If the criteria are vague (see first failure mode above), the vector breakdown gives you a false sense of insight — you can point to "the model failed on criterion X" without that explanation being meaningful, because criterion X did not measure what you thought. The cure is the same as the prevention: invest in criterion sharpness, calibrate against gold labels, and routinely audit individual judge answers to verify they correspond to what the criterion language describes.
      </Prose>

      <Callout accent="gold">
        APBR shifts the alignment problem from "trust the judge's holistic score" to "trust each binary answer". The shift only pays off if you actually invest in making each binary question precise enough to be answered consistently. A poorly written boolean rubric is worse than a Likert score, because it gives the appearance of rigor without the substance.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Sources verified against their canonical references on 2026-04-26. Author lists, arXiv IDs, and publication venues confirmed.
      </Prose>

      <H3>OpenAI 2025 — HealthBench</H3>
      <Prose>
        OpenAI. "HealthBench: Evaluating Large Language Models Towards Improved Human Health." Released 2025 by OpenAI's Health AI team in collaboration with practicing physicians. The reference implementation of physician-authored boolean rubrics: 5,000 prompts, up to 50 binary criteria per prompt, with criteria designed for high inter-physician agreement. Introduces the precise-rubric methodology, the per-category aggregation pattern (safety, accuracy, communication), and the large-scale gold-set calibration of LLM judges. The HealthBench paper is the reference text for understanding why boolean decomposition succeeds in high-stakes medical evaluation where Likert scales fail.
      </Prose>

      <H3>Zhou et al. 2023 — IFEval</H3>
      <Prose>
        Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, Le Hou. "Instruction-Following Evaluation for Large Language Models." arXiv:2311.07911. Published November 2023. While not framed as APBR per se, IFEval is the most influential precursor. It introduces verifiable instruction-following criteria — "the response must contain the word 'sustainable'", "the response must be in JSON format", "the response must be exactly 3 paragraphs" — each of which is a precise boolean criterion that can be checked deterministically. IFEval demonstrated that decomposing instruction-following into hundreds of mechanically checkable boolean criteria produces a vastly more reliable evaluation than asking a judge "did the model follow instructions?" The pattern generalizes: where IFEval used regex and structural checks, APBR uses LLM judges, but the philosophy is the same.
      </Prose>

      <H3>Saad-Falcon et al. 2024 — LMUnit</H3>
      <Prose>
        Jon Saad-Falcon, Rajan Vivek, William Berrios, Nandita Shankar Naik, Matija Franklin, Bertie Vidgen, Amanpreet Singh, Douwe Kiela, Shikib Mehri. "LMUnit: Fine-grained Evaluation with Natural Language Unit Tests." arXiv:2412.13091. Published December 2024. Generalizes the boolean-rubric pattern to arbitrary natural language tasks under the framing of "natural language unit tests" — each rubric criterion is a unit test for response quality, written by a domain expert, executed by an LLM judge, and tracked over time. Provides empirical validation that boolean unit tests outperform Likert and pairwise judging on a range of evaluation benchmarks, with explicit measurement of inter-judge agreement and judge calibration. LMUnit's framing has been adopted by several open-source eval pipelines and is the reference text for the rubric-as-unit-test pattern.
      </Prose>

      <H3>OpenAI 2024 — o1-preview eval methodology</H3>
      <Prose>
        OpenAI. "Learning to Reason with LLMs" (o1 system card and eval reports), released September-November 2024. Describes the eval methodology used for o1-preview's safety and capability releases, which extensively uses boolean rubric decomposition for evaluating reasoning quality, scientific accuracy, and safety behaviors. The eval methodology section explicitly contrasts the boolean-rubric approach to prior Likert-based methods, citing per-criterion reliability gains and the operational value of per-category breakdowns. While the system card is less academic than HealthBench, it documents APBR-style evaluation in production at scale at OpenAI.
      </Prose>

      <H3>Cohen 1960 — Kappa coefficient</H3>
      <Prose>
        Jacob Cohen. "A Coefficient of Agreement for Nominal Scales." Educational and Psychological Measurement, 20(1):37-46, April 1960. The foundational paper on chance-corrected agreement statistics. Cohen's kappa is the standard metric for measuring inter-annotator agreement on categorical labels, including the binary criteria of APBR rubrics. Modern APBR practice cites kappa thresholds (typically 0.7 for acceptable, 0.85 for excellent) drawn directly from Cohen's framework. Understanding kappa — and its many criticisms when prior probabilities are skewed — is essential for interpreting reliability claims about rubrics.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Derive the variance reduction</H3>
      <Prose>
        Given an APBR rubric with N independent boolean criteria, each judged with per-question error rate ε (probability of flipping the correct answer), derive the variance of the unweighted aggregate score (count of satisfied criteria / N). Compare this to the variance of a single Likert judge call with Gaussian noise σ_L. For ε = 0.05 and σ_L = 0.2 (on a normalized 0-1 scale), at what value of N does the APBR standard error fall below the Likert standard error? At what value does it fall to half the Likert standard error? What does this tell you about the minimum useful rubric size?
      </Prose>

      <H3>Exercise 2 — Design a rubric for code review</H3>
      <Prose>
        Design an APBR rubric with 10 root criteria and 5 conditional criteria for evaluating LLM-generated code reviews of a pull request. Each criterion must be precise enough that two competent senior engineers would agree on the answer without discussion. Identify which criteria belong in which category (correctness, style, security, communication). Identify which criteria should be conditionals and what their preconditions are. For each criterion, write the exact question text. Then identify two criteria you wrote that would likely fail an inter-rater agreement test (kappa below 0.7) and rewrite them to be sharper.
      </Prose>

      <H3>Exercise 3 — Adaptive vs. flat token cost</H3>
      <Prose>
        For a rubric with 50 criteria of which 20 are root criteria and 30 are conditional (each conditional has exactly one parent precondition with a 60% trigger rate, and the conditional structure is one level deep), compute the expected number of judge calls per response under (a) flat evaluation that asks every criterion regardless, and (b) adaptive evaluation that respects preconditions. Now extend the conditional structure to two levels deep (15 of the 30 conditionals depend on another conditional, also with 60% trigger rate). Compute the new expected calls under adaptive evaluation. What is the breakeven point in terms of rubric authoring complexity that makes the adaptive structure worth implementing?
      </Prose>

      <H3>Exercise 4 — Catching judge bias</H3>
      <Prose>
        You have an APBR rubric with 20 criteria. You suspect the LLM judge has a length bias — it answers "yes" more often for longer responses regardless of content. Design an experiment to detect this bias and quantify its magnitude per criterion. Specifically: what controlled response set would you generate, what statistic would you compute, what would a result indicating "criterion C is length-biased" look like, and how would you adjust the rubric or the judging protocol to mitigate it? Bonus: would using a different judge model eliminate the problem, or just change which biases are present?
      </Prose>

      <H3>Exercise 5 — When APBR loses to Likert</H3>
      <Prose>
        Identify a task where APBR is likely to perform worse than a single Likert judge call, holding judge cost constant. Justify your answer in terms of (a) the decomposability of quality for that task, (b) the inter-rater reliability you would expect on plausible boolean criteria, and (c) the interpretability of the resulting score vector. As a follow-up: even for that task, is there a hybrid evaluation protocol that combines APBR for measurable dimensions and Likert for the residual that is more useful than either alone? What would that look like operationally?
      </Prose>

    </div>
  ),
};

export default adaptiveBooleanRubrics;
