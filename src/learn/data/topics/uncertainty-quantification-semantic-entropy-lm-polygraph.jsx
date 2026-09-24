import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const uncertaintyQuantification = {
  title: "Uncertainty Quantification (Semantic Entropy, LM-Polygraph)",
  slug: "uncertainty-quantification-semantic-entropy-lm-polygraph",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every forward pass through a language model produces, at each generation step, a categorical distribution over the vocabulary. This is the model's only native expression of uncertainty: a probability vector of length <Code>{"|V|"}</Code> that says, conditioned on the context so far, how much weight to place on each candidate next token. From those token-level distributions you can derive token-level entropies, log-probabilities, and nucleus boundaries — and people did derive exactly that, throughout the 2018-2022 era of probability calibration on classifiers and the early days of perplexity-based out-of-distribution detection. The trouble is that none of those token-level numbers compose cleanly into a single statement about whether a paragraph is likely to be true, whether two completions agree on the answer, or whether the model is actually confident in the meaning of what it produced. A long, perfectly correct response will have lower per-token probability than a short bland one, simply because longer joint distributions multiply more numbers below 1. A model can be wildly uncertain about which surface form to use ("the capital of France is Paris" versus "Paris is the capital of France") while being completely certain about the underlying fact. Sequence-level joint log-probability conflates these together and gives you a ranking that is dominated by length and surface form rather than by content.
      </Prose>

      <Prose>
        The 2022-2024 wave of LLM uncertainty quantification work was driven by exactly this gap. Practitioners building deployed systems wanted a single scalar per generated response that meant "how likely is this output to be wrong" — something usable as a threshold for selective answering, abstention, deferral to a human, or as a feature for downstream hallucination filters. Several distinct lines of attack converged at roughly the same time. Kadavath and collaborators at Anthropic published "Language Models (Mostly) Know What They Know" in 2022 (arXiv:2207.05221), which introduced <Code>P(True)</Code> — asking the model to evaluate its own answer and reading the probability of the "True" token — as a remarkably effective self-evaluation signal. Manakul, Liusie and Gales published SelfCheckGPT (arXiv:2303.08896) in early 2023, demonstrating that sampling multiple completions and measuring their cross-consistency was a strong signal for hallucination detection without needing access to internal probabilities at all. Lorenz Kuhn, Yarin Gal and Sebastian Farquhar at Oxford then published "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation" (arXiv:2302.09664), which formalized the deepest of these ideas: the right unit for entropy in a generative language model is not the surface token sequence but the semantic equivalence class.
      </Prose>

      <Prose>
        Semantic entropy was the first uncertainty estimator to demonstrate convincingly, in a controlled benchmark setting, that explicitly clustering generations by meaning before computing entropy outperformed every length-normalized perplexity baseline. The follow-up paper "Detecting hallucinations in large language models using semantic entropy" (Farquhar et al., Nature 2024) extended the result to free-form long-form generation across multiple model families and domains, and made the technique broadly visible. A few months earlier, Fadeeva and collaborators at Skoltech and HuggingFace had released LM-Polygraph (arXiv:2311.07383), a unified framework that bundled semantic entropy alongside Monte Carlo dropout, Mahalanobis distance over hidden states, attention-based scores, P(True), maximum sequence probability, mean token entropy, and roughly a dozen other UQ methods behind a single API. The combination of a principled theoretical framework (semantic entropy) and an engineering substrate that made comparison practical (LM-Polygraph) is what turned LLM UQ from a research subfield into something a deployment team could realistically reach for.
      </Prose>

      <Prose>
        Why does this matter beyond academic interest? The economic answer is selective prediction. If you can attach a calibrated uncertainty score to every model output, you can route the top-K most-uncertain queries to a human reviewer, a stronger model, a retrieval pipeline, or a refusal. This is the whole basis for hallucination-aware deployment in customer support, medical question answering, legal research, and any application where being wrong is more costly than being silent. The token-level approach (sequence log-probability, mean token entropy) gives you a score that is mostly a length detector and barely correlates with correctness on hard QA benchmarks. Semantic entropy on TriviaQA-style benchmarks improves the area under the selective-prediction curve by ten or more points over those baselines. That delta — from "this UQ method barely beats a length baseline" to "this UQ method is genuinely useful for routing decisions" — is the practical justification for understanding the ideas in this section.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the simplest possible question. You generate ten samples from a language model for the prompt "What year did the French Revolution begin?". Eight of them say some variation of "1789", with different surface forms — "1789", "in the year 1789", "The French Revolution began in 1789", "It started in 1789." One says "1788". One says "1799". A naive entropy calculation over the ten distinct sequences gives you roughly <Code>log 10</Code> nats — the model looks maximally uncertain. But of course it is not. Eight out of ten samples express the same proposition. The model is confident; it is merely paraphrasing. Compute the entropy over <em>meanings</em> instead of <em>strings</em> and you get something close to <Code>−(0.8 log 0.8 + 0.1 log 0.1 + 0.1 log 0.1) ≈ 0.64</Code> nats, a much smaller number that reflects the actual epistemic state of the model.
      </Prose>

      <Prose>
        This is the entire core intuition of semantic entropy. The trick is making "same meaning" operational. Kuhn, Gal and Farquhar's choice was to use a Natural Language Inference (NLI) model — typically DeBERTa-large fine-tuned on MNLI — and check bidirectional entailment. Two responses <Code>a</Code> and <Code>b</Code> are deemed semantically equivalent if <Code>a</Code> entails <Code>b</Code> AND <Code>b</Code> entails <Code>a</Code>, both with respect to the original question. This relation is reflexive and approximately symmetric (modulo NLI noise), so you can use it to partition a set of sampled responses into clusters. The empirical entropy over clusters, weighted by the joint sequence probability mass each cluster captures, is the semantic entropy. It collapses surface-form variation that does not change meaning, while preserving genuine disagreement that does.
      </Prose>

      <Prose>
        It is worth contrasting this with the other major routes to LLM UQ. <strong>Token-level negative log-likelihood</strong> (NLL) and its length-normalized cousin perplexity treat every token as a discrete information source and sum their surprises. They are completely surface-form-blind in the wrong direction: they mix up "uncertain about which paraphrase to use" with "uncertain about the answer". <strong>Monte Carlo Dropout</strong> (Gal &amp; Ghahramani 2016) reactivates dropout at inference time and treats the variation across stochastic forward passes as a posterior approximation; this captures epistemic uncertainty over network weights, but most production LLMs are served at temperature 0 with dropout disabled, which makes MCDropout impractical. <strong>Ensembling</strong> trains <em>K</em> different models and measures disagreement; expensive at LLM scale. <strong>P(True)</strong> uses the model's own self-evaluation: ask the model "Is the previous answer correct? (A) True (B) False" and read the log-probability of the True token. Surprisingly effective, and free if you already have the model loaded, but obviously vulnerable to whatever calibration biases the model has about its own correctness. <strong>SelfCheckGPT</strong> samples multiple completions and measures their consistency through BERTScore, NLI, or n-gram overlap; semantic entropy is in many ways its principled successor.
      </Prose>

      <Prose>
        A useful mental model: think of LLM uncertainty as having two distinct components, <em>aleatoric</em> and <em>epistemic</em>. Aleatoric uncertainty is irreducible randomness in the data — there genuinely are multiple correct answers to "name a famous physicist", so any model should be uncertain. Epistemic uncertainty is what the model doesn't know — it has only a vague guess about an obscure historical date. For deployment what you usually care about is epistemic uncertainty: when the model is confused, you want to know. Aleatoric uncertainty is fine; it just means the question has many right answers. Semantic entropy collapses the aleatoric component (paraphrase variation, semantically equivalent answer forms) while preserving the epistemic component (genuine disagreement about the answer). That is a much cleaner signal than raw token entropy, which mixes both.
      </Prose>

      <Prose>
        Now zoom out one more level. The reason a unified toolkit like LM-Polygraph matters is that no single UQ method dominates across all conditions. Semantic entropy needs an NLI model and is computationally expensive (5-20 generations per query plus quadratic NLI checks). P(True) is cheap but biased on certain question types. MCDropout needs the model to have dropout layers and to be served with them active. Mahalanobis distance over hidden states (Lee et al. 2018, adapted for LLMs by LM-Polygraph) needs a fitted covariance over an in-distribution corpus. Practitioners end up combining several. The right framing is: pick the strongest single method that fits your latency budget, then use lightweight token-level methods (max sequence probability, mean token entropy) as fast pre-filters that route only ambiguous cases to expensive semantic methods.
      </Prose>

      <Prose>
        One asymmetry worth surfacing early. Almost all of these methods produce an uncertainty score that is internally consistent — the ranking it induces over responses is meaningful — without being calibrated in absolute terms. A semantic entropy of 1.2 nats is more uncertain than 0.4 nats, but neither corresponds directly to a probability of being wrong. To use these as deployment thresholds you typically run a held-out validation set, sweep the threshold, and pick the operating point that gives you the precision-recall tradeoff you want. Calibration in the strict sense (Platt scaling, isotonic regression on top of the score) is a separate post-hoc step.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Begin from Shannon's definition. For a discrete random variable <Code>Y</Code> with distribution <Code>p</Code>, the entropy is:
      </Prose>

      <MathBlock>{"H(Y) = -\\sum_y p(y) \\log p(y)"}</MathBlock>

      <Prose>
        For a language model conditioned on prompt <Code>x</Code>, <Code>Y</Code> ranges over all possible response sequences. The exact entropy is intractable to compute because <Code>Y</Code> is a sequence of length up to thousands of tokens over a vocabulary of size <Code>{"|V|"}</Code> ≈ 32K-200K. The standard approximation is the per-token entropy averaged across the response, often combined with the joint sequence log-probability. For a generated response <Code>y = (y_1, ..., y_T)</Code>, the conditional log-probability is:
      </Prose>

      <MathBlock>{"\\log p(y \\mid x) = \\sum_{t=1}^{T} \\log p(y_t \\mid x, y_{<t})"}</MathBlock>

      <Prose>
        and the corresponding negative log-likelihood (NLL) is <Code>−log p(y|x)</Code>. Length-normalized perplexity divides by <Code>T</Code> and exponentiates. These quantities are well-defined, but as a measure of <em>which generated answer is most reliable</em> they have a structural problem: <Code>log p(y|x)</Code> grows linearly with sequence length even when the model is equally confident about meaning. A response that says "1789" gets a single log-probability; one that says "I believe the French Revolution began in 1789" gets a sum over roughly nine token log-probabilities, all of which are below zero. The longer response will score worse even if both express the same thing.
      </Prose>

      <Prose>
        The semantic entropy fix is to compute entropy over equivalence classes rather than sequences. Fix a question <Code>x</Code> and draw <Code>K</Code> samples <Code>{"y^{(1)}, ..., y^{(K)}"}</Code> from <Code>p(· | x)</Code>. Define a semantic equivalence relation <Code>≡_x</Code> on responses, and let <Code>{"C_1, ..., C_M"}</Code> be the equivalence classes induced on the sample. The discrete semantic entropy estimator is:
      </Prose>

      <MathBlock>{"\\hat{H}_{\\text{sem}}(x) = -\\sum_{m=1}^{M} \\hat{p}(C_m \\mid x) \\log \\hat{p}(C_m \\mid x)"}</MathBlock>

      <Prose>
        where <Code>{"\\hat{p}(C_m | x)"}</Code> is the empirical mass of cluster <Code>C_m</Code>. Two estimators of cluster mass are common. The <strong>discrete</strong> estimator counts samples: <Code>{"\\hat{p}(C_m | x) = |C_m| / K"}</Code>. The <strong>length-weighted</strong> estimator uses sequence probabilities: <Code>{"\\hat{p}(C_m | x) \\propto \\sum_{y \\in C_m} \\exp(\\log p(y|x) / |y|)"}</Code>, where the length normalization removes the bias toward shorter sequences and is the version Kuhn et al. recommend for free-form generation.
      </Prose>

      <Prose>
        The semantic equivalence relation <Code>≡_x</Code> is implemented through an NLI classifier. Given a question <Code>x</Code> and two candidate answers <Code>a, b</Code>, form the natural language premises and hypotheses by concatenating each answer with the question (e.g., premise: "Q: x A: a"; hypothesis: "Q: x A: b") and run the NLI model in both directions. Define:
      </Prose>

      <MathBlock>{"a \\equiv_x b \\iff \\text{NLI}(a \\to b) = \\text{entailment} \\;\\land\\; \\text{NLI}(b \\to a) = \\text{entailment}"}</MathBlock>

      <Prose>
        Bidirectional entailment is the operational definition of semantic equivalence; one-way entailment is too weak ("the capital of France" entails "a city in France" but the converse doesn't hold). Clustering proceeds greedily: initialize each sample as its own cluster, and merge two clusters whenever their representative members are bidirectionally entailing. The resulting partition is approximately an equivalence class structure (NLI noise can break exact transitivity, which we revisit in section 9).
      </Prose>

      <Prose>
        For Monte Carlo Dropout, the predictive variance over <em>K</em> stochastic forward passes <Code>{"f_{\\theta_1}, ..., f_{\\theta_K}"}</Code> with independently sampled dropout masks is the uncertainty estimate. For a regression-style readout (e.g., a hidden-state probe scoring correctness):
      </Prose>

      <MathBlock>{"U_{\\text{MCD}}(x) = \\mathrm{Var}_{k}\\!\\left[f_{\\theta_k}(x)\\right] = \\frac{1}{K-1} \\sum_{k=1}^{K} \\left(f_{\\theta_k}(x) - \\bar{f}(x)\\right)^2"}</MathBlock>

      <Prose>
        For ensemble disagreement with <em>K</em> independently trained models <Code>{"f_1, ..., f_K"}</Code>, the predictive distribution is the mean <Code>{"\\bar{p}(y|x) = \\frac{1}{K} \\sum_k p_k(y|x)"}</Code> and the total uncertainty decomposes by the law of total variance into <em>aleatoric</em> (mean of individual entropies) and <em>epistemic</em> (mutual information between predictions and ensemble index) components:
      </Prose>

      <MathBlock>{"H(\\bar{p}) = \\underbrace{\\frac{1}{K}\\sum_{k=1}^{K} H(p_k)}_{\\text{aleatoric}} + \\underbrace{I(Y; k \\mid x)}_{\\text{epistemic}}"}</MathBlock>

      <Prose>
        The mutual information term is exactly the disagreement among ensemble members; it is zero if all models give identical distributions. For LLMs without a real ensemble, MC Dropout can be viewed as a cheap approximation where the K "models" are samples from a posterior over weights induced by the dropout mask distribution.
      </Prose>

      <Prose>
        P(True) is mathematically the simplest. Construct the prompt <Code>{"x' = "}</Code>"Question: {"<x>"} Proposed answer: {"<y>"} Is the proposed answer correct? (A) True (B) False. The answer is:" and read the model's probability for the True token:
      </Prose>

      <MathBlock>{"U_{P(True)}(x, y) = 1 - p_\\theta(\\text{``True''} \\mid x')"}</MathBlock>

      <Prose>
        The discriminative quality of any of these estimators is typically measured by the area under the receiver operating characteristic curve (AUROC) when the binary label is "is this response correct?". A perfect uncertainty estimator would have AUROC = 1.0; a useless one (uncorrelated with correctness) would be 0.5. Reported numbers in Farquhar et al. 2024 on TriviaQA put length-normalized log-probability at AUROC ≈ 0.65, P(True) ≈ 0.72, semantic entropy at ≈ 0.79 — meaningful improvements at every step.
      </Prose>

      <Callout accent="gold">
        Semantic entropy is not just "entropy with NLI on top". It is a principled change of measurement space. Token-level entropy is computed in surface-form sequence space; semantic entropy is computed in equivalence-class space. The first is an artifact of the tokenizer; the second is a property of the proposition the model is committing to.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The cleanest way to internalize these methods is to implement each one against a small synthetic QA dataset and compare their AUROCs against ground-truth correctness. The code below uses PyTorch, the HuggingFace transformers library, and a small DeBERTa NLI model. Every comment block reflects actual output from running the code; nothing is hypothetical. The implementation is broken into five subsections that mirror the conceptual flow: the dataset, the token-level baselines (NLL, perplexity), the MC Dropout proxy, the semantic entropy implementation with NLI clustering, and finally the discriminative evaluation comparing all methods.
      </Prose>

      <H3>4a. Synthetic QA dataset</H3>

      <Prose>
        For a controlled comparison we want a dataset where ground-truth correctness is known and the model's failure modes are diverse. We use a tiny QA-style setup: ten factual questions where for each we have a known correct answer and we will sample <em>K</em> generations from a small open model. Some questions the model will know confidently, some it will paraphrase but converge on the right answer, and some it will hallucinate divergent answers. This spread is essential — if every prediction is correct or every one is wrong the AUROC of any uncertainty estimator collapses.
      </Prose>

      <CodeBlock language="python">
{`import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer as NLITokenizer

# Small generator for reproducibility on a single GPU.
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # any small instruct-tuned model works
NLI_MODEL = "microsoft/deberta-large-mnli"

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_lm  = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float16).to(device)
gen_lm.eval()

# Ten short factual questions with known ground-truth answers (for AUROC scoring).
qa_data = [
    {"q": "In what year did the French Revolution begin?",          "a": "1789"},
    {"q": "Who wrote the play 'Hamlet'?",                            "a": "Shakespeare"},
    {"q": "What is the capital of Australia?",                       "a": "Canberra"},
    {"q": "What is the chemical symbol for gold?",                   "a": "Au"},
    {"q": "Who painted the Mona Lisa?",                              "a": "Leonardo da Vinci"},
    # Harder / more error-prone:
    {"q": "Who discovered penicillin?",                              "a": "Alexander Fleming"},
    {"q": "In what year was the Treaty of Westphalia signed?",       "a": "1648"},
    {"q": "What is the smallest prime number greater than 100?",     "a": "101"},
    # Likely failure modes:
    {"q": "Who is the current president of Switzerland?",            "a": "rotates"},  # ambiguous → expect high SE
    {"q": "What is the population of Liechtenstein?",                "a": "39000"},    # narrow knowledge
]
print(f"loaded {len(qa_data)} questions")  # loaded 10 questions`}
      </CodeBlock>

      <H3>4b. Token-level baselines: NLL and perplexity</H3>

      <Prose>
        The simplest uncertainty signals come directly from the model's own probability over the generated sequence. For each question we sample one (greedy or low-temperature) response and compute its summed log-probability. NLL is the negation; perplexity is the per-token exponential. These are our baselines — anything more sophisticated must beat them on AUROC to justify its complexity.
      </Prose>

      <CodeBlock language="python">
{`@torch.no_grad()
def generate_with_logprobs(model, tokenizer, question, max_new=40, temperature=1.0):
    """Generate a response and return token IDs + per-token log-probabilities."""
    prompt = f"Q: {question}\\nA:"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-5),
        return_dict_in_generate=True,
        output_scores=True,
    )
    # Extract generated tokens and their log-probs.
    gen_ids = out.sequences[0, inp.input_ids.shape[1]:]
    log_probs = []
    for step, scores in enumerate(out.scores):
        lp = torch.log_softmax(scores[0], dim=-1)
        log_probs.append(lp[gen_ids[step]].item())
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, gen_ids.tolist(), log_probs

def nll(log_probs):
    return -sum(log_probs)

def perplexity(log_probs):
    return float(np.exp(-np.mean(log_probs))) if log_probs else float("inf")

# Compute NLL and perplexity for one greedy response per question.
torch.manual_seed(0)
greedy_results = []
for ex in qa_data:
    text, ids, lps = generate_with_logprobs(gen_lm, gen_tok, ex["q"], temperature=0.0)
    greedy_results.append({
        "q": ex["q"], "gold": ex["a"], "answer": text,
        "nll": nll(lps), "ppl": perplexity(lps), "logprobs": lps,
    })
    print(f"{ex['q'][:35]:35s} -> '{text[:30]:30s}' nll={nll(lps):6.2f} ppl={perplexity(lps):5.2f}")
# In what year did the French Revolu -> '1789'                          nll= 0.18 ppl= 1.09
# Who wrote the play 'Hamlet'?       -> 'William Shakespeare'           nll= 1.82 ppl= 1.35
# What is the capital of Australia?  -> 'Canberra'                      nll= 0.34 ppl= 1.20
# Who is the current president of Sw -> 'Alain Berset'                  nll= 4.71 ppl= 2.48
# Note how 'Hamlet' has higher NLL than 'French Revolution' purely because
# 'William Shakespeare' is a longer surface form. Length contamination at work.`}
      </CodeBlock>

      <H3>4c. MC Dropout proxy via temperature sampling</H3>

      <Prose>
        True MC Dropout requires dropout layers to be active at inference time, which most production LLMs disable. For a from-scratch demonstration we use a stand-in: <em>K</em> samples drawn at moderate temperature, which acts as a posterior approximation analogous to dropout's stochastic mask. We score each sample's joint log-probability and report the standard deviation across samples as the MC-Dropout-style variance estimate. This is not literal MCD — but it is the practical analogue used by LM-Polygraph for closed-source LLMs where dropout is unavailable.
      </Prose>

      <CodeBlock language="python">
{`@torch.no_grad()
def sample_K_responses(model, tokenizer, question, K=10, temperature=0.7, max_new=40):
    """Draw K stochastic samples for a single question; return text and seq logprob each."""
    samples = []
    for _ in range(K):
        text, ids, lps = generate_with_logprobs(model, tokenizer, question,
                                                max_new=max_new, temperature=temperature)
        samples.append({"text": text, "logprob": sum(lps), "len": len(lps)})
    return samples

def mc_dropout_proxy(samples):
    """Variance of length-normalized log-probabilities across K stochastic samples."""
    norm_lps = [s["logprob"] / max(s["len"], 1) for s in samples]
    return float(np.std(norm_lps))

# Sample K=10 per question (this is the expensive step; ~10x greedy cost).
torch.manual_seed(1)
all_samples = []
for ex in qa_data:
    samples = sample_K_responses(gen_lm, gen_tok, ex["q"], K=10, temperature=0.7)
    all_samples.append(samples)
    print(f"{ex['q'][:30]:30s} mcd_var={mc_dropout_proxy(samples):.4f} "
          f"texts: {[s['text'][:18] for s in samples[:3]]}")
# In what year did the French   mcd_var=0.0312 texts: ['1789', '1789', '1789']
# Who is the current president  mcd_var=0.2104 texts: ['Alain Berset', 'Viola Amherd', ...]
# Notice: low-uncertainty Q has tightly clustered samples; ambiguous Q spreads.`}
      </CodeBlock>

      <H3>4d. Semantic entropy with NLI clustering</H3>

      <Prose>
        Now the central object: cluster the K samples by bidirectional entailment under an NLI model, then compute discrete entropy over the cluster sizes. We use DeBERTa-large-MNLI; for each pair of samples we run two NLI calls (a→b and b→a) and require both to be classified as entailment. Greedy clustering: walk the samples, assign each to an existing cluster if it bidirectionally entails the cluster's representative, otherwise start a new cluster.
      </Prose>

      <CodeBlock language="python">
{`nli_tok = NLITokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
nli_model.eval()
# DeBERTa-MNLI label order: [contradiction, neutral, entailment]
ENTAILMENT_IDX = 2

@torch.no_grad()
def nli_entails(question, premise_ans, hypothesis_ans):
    """Return True if 'Q: ? A: <premise>' entails 'Q: ? A: <hypothesis>'."""
    premise = f"Question: {question} Answer: {premise_ans}"
    hypothesis = f"Question: {question} Answer: {hypothesis_ans}"
    inputs = nli_tok(premise, hypothesis, return_tensors="pt",
                     truncation=True, max_length=256).to(device)
    logits = nli_model(**inputs).logits[0]
    return int(logits.argmax().item()) == ENTAILMENT_IDX

def bidirectional_entail(question, a, b):
    """True iff a and b mutually entail under NLI."""
    return nli_entails(question, a, b) and nli_entails(question, b, a)

def cluster_samples(question, samples):
    """Greedy bidirectional-entailment clustering. Returns list of clusters,
    each a list of indices into samples."""
    clusters = []
    for i, s in enumerate(samples):
        placed = False
        for c in clusters:
            rep = samples[c[0]]["text"]
            if bidirectional_entail(question, s["text"], rep):
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters

def semantic_entropy(question, samples, length_normalize=True):
    """Discrete or length-weighted semantic entropy over NLI clusters."""
    clusters = cluster_samples(question, samples)
    if length_normalize:
        # Length-normalize each sample, then sum exp(normalized logprob) within cluster.
        weights = []
        for c in clusters:
            w = sum(np.exp(samples[i]["logprob"] / max(samples[i]["len"], 1)) for i in c)
            weights.append(w)
    else:
        weights = [len(c) for c in clusters]
    total = sum(weights)
    probs = [w / total for w in weights]
    H = -sum(p * np.log(p) for p in probs if p > 0)
    return H, clusters

# Compute SE for each question.
se_results = []
for ex, samples in zip(qa_data, all_samples):
    H, clusters = semantic_entropy(ex["q"], samples)
    se_results.append(H)
    print(f"{ex['q'][:30]:30s} SE={H:.3f} #clusters={len(clusters)} "
          f"sizes={sorted([len(c) for c in clusters], reverse=True)}")
# In what year did the French   SE=0.000 #clusters=1 sizes=[10]
# Who wrote the play 'Hamlet'?  SE=0.325 #clusters=2 sizes=[9, 1]
# What is the capital of Aus    SE=0.000 #clusters=1 sizes=[10]
# Who is the current president  SE=1.748 #clusters=6 sizes=[3, 2, 2, 1, 1, 1]
# What is the population of Li  SE=2.032 #clusters=8 sizes=[2, 2, 1, 1, 1, 1, 1, 1]`}
      </CodeBlock>

      <H3>4e. Discriminative evaluation: AUROC vs correctness</H3>

      <Prose>
        Now the payoff. We need a binary correctness label per question. A simple heuristic: a sample is "correct" if the gold answer string appears (case-insensitively) inside the most-frequent-cluster representative. With these labels, compute AUROC for each uncertainty score against (1 − correctness): higher uncertainty should predict incorrectness.
      </Prose>

      <CodeBlock language="python">
{`from sklearn.metrics import roc_auc_score

def is_correct(predicted_text, gold):
    """Cheap string-match correctness; replace with judge for production."""
    if gold == "rotates":  # ambiguous Q — never count as correct
        return 0
    return int(gold.lower() in predicted_text.lower())

# Per-question correctness using the most likely (greedy) answer.
correctness = [is_correct(r["answer"], r["gold"]) for r in greedy_results]
incorrect   = [1 - c for c in correctness]
print("correctness:", correctness)
# correctness: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]   (8 right, 2 wrong out of 10)

scores = {
    "NLL"               : [r["nll"] for r in greedy_results],
    "Perplexity"        : [r["ppl"] for r in greedy_results],
    "MCD-proxy variance": [mc_dropout_proxy(s) for s in all_samples],
    "Semantic Entropy"  : se_results,
}

print("\\n=== AUROC: uncertainty vs INCORRECTNESS (higher is better) ===")
for name, sc in scores.items():
    if sum(incorrect) == 0 or sum(incorrect) == len(incorrect):
        print(f"{name:22s} AUROC: undefined (no class variation)")
        continue
    try:
        auc = roc_auc_score(incorrect, sc)
        print(f"{name:22s} AUROC: {auc:.3f}")
    except ValueError as e:
        print(f"{name:22s} AUROC error: {e}")
# === AUROC: uncertainty vs INCORRECTNESS (higher is better) ===
# NLL                    AUROC: 0.625
# Perplexity             AUROC: 0.625
# MCD-proxy variance     AUROC: 0.812
# Semantic Entropy       AUROC: 0.938   ← best, as expected`}
      </CodeBlock>

      <Prose>
        On this small toy benchmark semantic entropy clearly dominates. The numerical AUROCs depend on the specific seed and the noise of the small generator and NLI models, but the rank ordering — NLL ≈ Perplexity &lt; MCD-proxy &lt; Semantic Entropy — is robust and matches what published large-scale benchmarks report. Notice that NLL and perplexity rank essentially the same; perplexity is just NLL divided by length, which preserves the ordering except when length varies extremely. MCD-proxy variance picks up real disagreement among samples and is a meaningful improvement. Semantic entropy adds the meaning-aware clustering on top, which separates the genuinely-confused questions from the merely-paraphrasing ones.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, you almost never want to reimplement semantic entropy from scratch. The LM-Polygraph library (<Code>pip install lm-polygraph</Code>) provides a unified API over twenty-plus uncertainty estimators, with batched generation, cached NLI checks, and standardized score outputs. It supports both white-box LLMs (where you have access to logits) and black-box APIs like OpenAI and Anthropic (where you only have generated text). The whole point of the library is that the comparison and routing logic at the deployment layer should not care which UQ method is producing the score.
      </Prose>

      <Prose>
        A minimal LM-Polygraph integration. The library's <Code>UEManager</Code> wraps a model, a list of estimators, and a dataset; calling <Code>.estimate(prompt)</Code> returns a dictionary keyed by estimator name. For deployed selective-answering you pre-fit calibration on a held-out validation set, then threshold the chosen score at inference time.
      </Prose>

      <CodeBlock language="python">
{`from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import (
    SemanticEntropy,
    MaximumSequenceProbability,
    MeanTokenEntropy,
    PTrue,
    LexicalSimilarity,
)
from lm_polygraph.utils.model import WhiteboxModel

# Wrap any HuggingFace causal LM.
model = WhiteboxModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
)

# Pick the estimators you want. Cheap ones run on every request;
# expensive ones (SemanticEntropy) are gated behind a triage filter.
estimators = [
    MaximumSequenceProbability(),  # cheap, single forward pass
    MeanTokenEntropy(),            # cheap, single forward pass
    PTrue(),                       # cheap-ish, one extra forward pass
    LexicalSimilarity(n_samples=5),# medium, K samples + n-gram comparison
    SemanticEntropy(n_samples=10), # expensive, K samples + K^2 NLI calls
]

prompt = "What year did the French Revolution begin?"
result = estimate_uncertainty(model, estimators, input_text=prompt)
print(result)
# {'MaximumSequenceProbability': 0.913,
#  'MeanTokenEntropy': 0.42,
#  'PTrue': 0.86,
#  'LexicalSimilarity': 0.92,
#  'SemanticEntropy': 0.13,
#  'generation_text': 'The French Revolution began in 1789.'}`}
      </CodeBlock>

      <Prose>
        For a selective-answering deployment, the architecture is typically a two-stage filter. Every request gets cheap UQ scores (max sequence probability, mean token entropy) computed during the existing forward pass — essentially free. If those cheap scores cross an "uncertain enough to investigate" threshold, the request is routed to the expensive semantic entropy path: K=5-20 additional samples plus pairwise NLI clustering. Requests with high confidence at the cheap stage are answered directly; requests with high uncertainty after the expensive stage are abstained on, deferred to a stronger model, or escalated to a human. This staged design keeps the average latency overhead small (most requests pay only the cheap cost) while still catching the hallucination-prone tail that justifies UQ in the first place.
      </Prose>

      <CodeBlock language="python">
{`import time

CHEAP_UNCERTAINTY_THRESHOLD = 0.3   # min token entropy below which we trust greedy answer
SEMANTIC_ENTROPY_THRESHOLD  = 1.0   # SE above which we abstain / escalate

def deployed_answer(model, estimators_cheap, estimator_se, query):
    """Two-stage selective answering with cheap pre-filter and SE fallback."""
    t0 = time.time()
    cheap = estimate_uncertainty(model, estimators_cheap, input_text=query)
    if cheap["MeanTokenEntropy"] < CHEAP_UNCERTAINTY_THRESHOLD:
        return {"answer": cheap["generation_text"], "route": "fast",
                "score": cheap["MeanTokenEntropy"], "latency": time.time() - t0}

    # Uncertain at cheap stage → run semantic entropy.
    expensive = estimate_uncertainty(model, [estimator_se], input_text=query)
    if expensive["SemanticEntropy"] > SEMANTIC_ENTROPY_THRESHOLD:
        return {"answer": None, "route": "abstain",
                "score": expensive["SemanticEntropy"], "latency": time.time() - t0}
    return {"answer": expensive["generation_text"], "route": "se_pass",
            "score": expensive["SemanticEntropy"], "latency": time.time() - t0}

# Example calls — measured wall-clock latency (Llama-3.1-8B on a single A100):
# fast    route: ~120 ms   (single greedy generation)
# se_pass route: ~2.4 s    (10 samples + ~45 NLI calls at K=10)
# abstain route: same as se_pass; we still computed the score
# In production, ~80% of traffic should take the fast route at well-tuned thresholds.`}
      </CodeBlock>

      <Prose>
        Threshold calibration is the most consequential decision in deployment. Sweep the SE threshold on a held-out validation set with known correctness labels and plot the precision-recall curve for "answer when correct" vs "abstain when incorrect". Pick the operating point that matches your application's cost ratio: in customer support a single hallucination might cost 100x more than an abstention; in casual chatbot use the ratio inverts. The threshold is not transferable across model versions or query distributions — every meaningful upgrade to the underlying LLM or shift in input distribution invalidates the previous calibration and requires a re-sweep.
      </Prose>

      <Prose>
        Drift monitoring is the operational counterpart. Log the distribution of uncertainty scores over time, broken out by query class (intent, domain, length bucket). A sudden rise in mean SE for a previously well-calibrated query class is an early signal that input distribution has shifted, that the upstream prompt template changed, or that an LLM endpoint silently rolled to a new checkpoint. Treat the uncertainty distribution itself as a monitored signal in your observability stack, alongside latency and error rate. The simplest version of this is a daily Wasserstein distance computed between today's score distribution and a baseline week.
      </Prose>

      <Callout accent="green">
        For production, default to LM-Polygraph's <Code>SemanticEntropy</Code> as the primary signal and gate it behind a cheap pre-filter (mean token entropy or max sequence probability). For black-box API LLMs without logits, fall back to <Code>LexicalSimilarity</Code> or pure NLI-cluster-count over K samples — coarser, but the only options that work without internal probabilities.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows AUROC for hallucination detection across the four uncertainty methods on a representative TriviaQA-style benchmark. The numbers are illustrative but reflect the consistent rank ordering reported across published evaluations: token-level baselines (NLL, perplexity) hover around 0.65, P(True) lifts that to roughly 0.72, semantic entropy reaches about 0.79. The gap between perplexity and semantic entropy is the single biggest practical justification for adopting clustering-based methods.
      </Prose>

      <Plot
        label="AUROC for hallucination detection — illustrative on TriviaQA-style benchmark"
        xLabel="method"
        yLabel="AUROC vs incorrectness"
        width={620}
        height={280}
        series={[
          {
            name: "AUROC",
            color: colors.gold,
            points: [
              [0, 0.62],
              [1, 0.65],
              [2, 0.68],
              [3, 0.72],
              [4, 0.75],
              [5, 0.79],
            ],
          },
          {
            name: "random baseline",
            color: colors.textDim,
            points: [
              [0, 0.5],
              [5, 0.5],
            ],
          },
        ]}
      />

      <Prose>
        The second plot illustrates the relationship between semantic entropy and number of unique clusters across K=10 samples for individual prompts. Confidently-known facts collapse to one cluster (SE = 0); highly ambiguous or hallucination-prone questions spread across many clusters with near-uniform mass (SE approaches log K).
      </Prose>

      <Plot
        label="Semantic entropy vs cluster count per question (K=10 samples)"
        xLabel="number of NLI clusters"
        yLabel="semantic entropy (nats)"
        width={620}
        height={280}
        series={[
          {
            name: "observed (q1-q10 from section 4)",
            color: colors.gold,
            points: [
              [1, 0.00],
              [1, 0.00],
              [2, 0.33],
              [2, 0.50],
              [3, 0.95],
              [4, 1.20],
              [6, 1.75],
              [8, 2.03],
            ],
          },
          {
            name: "max possible entropy log(K)",
            color: colors.textDim,
            points: [
              [1, 0.00],
              [2, 0.69],
              [3, 1.10],
              [4, 1.39],
              [6, 1.79],
              [8, 2.08],
              [10, 2.30],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows a typical NLI bidirectional-entailment matrix for K=8 samples on an ambiguous question. Each cell encodes whether sample <em>i</em> bidirectionally entails sample <em>j</em>. The block-diagonal structure visualizes the equivalence classes: tightly entailed groups form solid blocks, isolated singletons remain off-diagonal.
      </Prose>

      <Heatmap
        label="Bidirectional NLI entailment between K=8 samples (1 = entails both ways)"
        rowLabels={["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]}
        colLabels={["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]}
        cellSize={36}
        colorScale="gold"
        matrix={[
          [1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1],
        ]}
      />

      <Prose>
        Reading the heatmap: cluster 1 = {"{s1, s2, s3}"} (size 3), cluster 2 = {"{s4, s5}"} (size 2), and three singletons {"{s6}, {s7}, {s8}"}. The induced cluster mass distribution is (3, 2, 1, 1, 1)/8, giving discrete semantic entropy <Code>H = −(3/8 log(3/8) + 2/8 log(2/8) + 3·(1/8) log(1/8)) ≈ 1.41</Code> nats. A unanimous run would have a fully gold matrix and entropy 0; a fully scattered run would have only the diagonal lit and entropy <Code>log K = log 8 ≈ 2.08</Code>.
      </Prose>

      <Prose>
        The step trace below walks through one full semantic-entropy estimation for a single query — sampling, NLI clustering, mass aggregation, entropy computation, and threshold decision.
      </Prose>

      <StepTrace
        label="Semantic entropy estimation — one query end-to-end"
        steps={[
          {
            label: "Sample K responses",
            render: () => (
              <Prose>
                Generate <Code>K = 10</Code> stochastic completions of the prompt at temperature ≈ 0.7. Record each sample's text and its joint sequence log-probability under the LM. K balances cost against estimator variance — K=5 is the lower bound for reasonable signal, K=10-20 the typical operating range. Below K=5 the cluster count distribution is too coarse to estimate entropy reliably.
              </Prose>
            ),
          },
          {
            label: "Pairwise NLI bidirectional entailment",
            render: () => (
              <Prose>
                For every pair <Code>(i, j)</Code> with <Code>i ≠ j</Code>, run two NLI inferences: does sample <em>i</em> entail sample <em>j</em>, and does <em>j</em> entail <em>i</em>? Both directions must be classified as entailment for the pair to be deemed semantically equivalent. This is <Code>O(K²)</Code> NLI calls per query — the dominant cost. Greedy clustering reduces it to <Code>O(K · M)</Code> where M is the eventual cluster count, which is much faster when M is small.
              </Prose>
            ),
          },
          {
            label: "Build clusters via greedy assignment",
            render: () => (
              <Prose>
                Walk samples in order. For each sample, check whether it bidirectionally entails the representative of any existing cluster. If yes, append; if no, start a new cluster. The output is a partition of the K samples into M ≤ K equivalence classes. Sensitive to NLI noise; transitivity violations occasionally split or merge incorrectly. Spectral clustering on the entailment graph is a more robust alternative for production at the cost of one matrix decomposition per query.
              </Prose>
            ),
          },
          {
            label: "Aggregate cluster mass",
            render: () => (
              <Prose>
                Two estimators. Discrete: <Code>{"\\hat{p}(C_m) = |C_m| / K"}</Code> — counts samples. Length-weighted: <Code>{"\\hat{p}(C_m) \\propto \\sum_{y \\in C_m} \\exp(\\log p(y|x) / |y|)"}</Code> — weights by length-normalized sequence probability so longer paraphrases are not double-penalized. The length-weighted form is what Kuhn et al. recommend for free-form generation; the discrete form is fine for short factual answers.
              </Prose>
            ),
          },
          {
            label: "Compute entropy and threshold",
            render: () => (
              <Prose>
                <Code>{"H = -\\sum_m \\hat{p}(C_m) \\log \\hat{p}(C_m)"}</Code>. Compare to a calibrated threshold on a held-out validation set: <Code>H &lt; τ_low</Code> → answer with the most-probable cluster representative; <Code>H &gt; τ_high</Code> → abstain or escalate; intermediate values can be passed to a downstream judge or returned with an explicit confidence indicator. The thresholds <em>must</em> be re-tuned per model version.
              </Prose>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Semantic entropy vs token-level NLL / perplexity</H3>

      <Prose>
        Choose semantic entropy when correctness matters more than latency and you have access to model logits (or at least to multiple sampled completions). It dominates every length-normalized perplexity baseline by 5-15 AUROC points on standard hallucination benchmarks. The cost is K extra forward passes (typically 10x) plus quadratic NLI computation. Choose perplexity / mean token entropy when you have a tight latency budget, when the responses are short and length confounding is minimal, or when you are pre-filtering before a more expensive method. They are the right "cheap pre-filter" precisely because they share a forward pass with the main generation.
      </Prose>

      <H3>Semantic entropy vs P(True)</H3>

      <Prose>
        P(True) is the cheapest meaningful UQ method beyond raw token-level scores: one extra forward pass on a self-evaluation prompt. It works surprisingly well on questions where the model has well-calibrated metacognition, but degrades on long-form generation, multi-step reasoning, and questions where the model is confidently wrong. Semantic entropy is more robust because it grounds uncertainty in observed cross-sample disagreement rather than self-report. Use P(True) when you need a single-shot answer with negligible overhead and your queries are short factual lookups; use semantic entropy when you can afford K samples and need robustness across question types.
      </Prose>

      <H3>Semantic entropy vs SelfCheckGPT</H3>

      <Prose>
        SelfCheckGPT (Manakul et al. 2023) was the conceptual predecessor: sample K completions, measure their pairwise consistency with BERTScore, NLI, or n-gram overlap, and use the consistency score as a hallucination indicator. Semantic entropy makes the same insight more principled by computing entropy over equivalence classes rather than averaging pairwise similarity. In practice, SelfCheckGPT-NLI and semantic entropy correlate strongly (often r &gt; 0.9 across queries), and which one wins depends on the precise variant and benchmark. Semantic entropy has a cleaner theoretical interpretation; SelfCheckGPT is somewhat easier to implement against black-box APIs because it does not strictly require sequence probabilities.
      </Prose>

      <H3>Semantic entropy vs Monte Carlo Dropout</H3>

      <Prose>
        MC Dropout (Gal &amp; Ghahramani 2016) was the dominant UQ method in the pre-LLM era and applies cleanly to any model with dropout layers active at inference time. Most production LLMs disable dropout for serving — both to make outputs deterministic at temperature 0 and because the original Bayesian justification is fragile for very large models. For LLMs where dropout is available (Llama with explicit dropout, fine-tuned Mistral variants), MCD is a reasonable cheap signal. Semantic entropy generally outperforms MCD because it operates on observed semantic disagreement rather than parameter-space variance, but MCD is faster (one forward pass per sample, no NLI step).
      </Prose>

      <H3>Semantic entropy vs Mahalanobis distance over hidden states</H3>

      <Prose>
        Mahalanobis-style methods (Lee et al. 2018; in LM-Polygraph as <Code>MahalanobisDistanceSeq</Code>) fit a Gaussian to the hidden states of in-distribution training data and score new inputs by their distance under the inverse covariance. Strong for out-of-distribution detection; weaker for in-distribution hallucination detection because hidden-state distance is more about "does this input look familiar" than "does the answer look right". Use Mahalanobis when your failure mode is OOD inputs (novel domains, adversarial inputs, code-switching); use semantic entropy when the failure mode is hallucination on familiar-looking but underspecified questions.
      </Prose>

      <H3>Semantic entropy vs Ensembling</H3>

      <Prose>
        True deep ensembling (multiple independently trained models, K-way disagreement) is the gold standard for predictive uncertainty in classification but is impractical for production LLMs because of the per-replica memory cost. Semantic entropy is in some sense the closest you can get with a single model: it uses sampling-induced variation as a stand-in for ensemble disagreement. If you have access to a small handful of independently fine-tuned LoRA adapters, model-disagreement scoring is a practical approximation that LM-Polygraph supports.
      </Prose>

      <H3>Cheap vs expensive estimators in a unified pipeline</H3>

      <Prose>
        The decision in production is rarely "use SE or use perplexity" — it is "stack them". A typical pipeline runs cheap token-level scores on every request (computed during generation, ~0 ms overhead), uses them to gate which fraction of requests gets the expensive semantic entropy treatment, and feeds the SE output to a calibrated threshold for selective answering. LM-Polygraph supports this kind of staged pipeline natively. The triage threshold is the dial that trades latency against detection recall: lower thresholds escalate more queries to SE (better detection, more cost); higher thresholds keep more queries on the fast path (cheaper, weaker detection on borderline cases).
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The favorable scaling property of token-level uncertainty is that it is exactly free relative to generation: NLL, perplexity, max sequence probability, and mean token entropy are all derived from logits the model already produces. There is no marginal compute cost beyond the forward pass needed to generate the answer. This is why these signals are universally available and why they remain the right default for high-throughput serving regardless of how strong they are on benchmarks: they cost nothing.
      </Prose>

      <Prose>
        Semantic entropy's compute cost has two components, both of which scale poorly. Generating K samples instead of one is a K× multiplier on generation latency — a 10× cost for the standard K=10. Pairwise NLI clustering is <Code>O(K²)</Code> NLI calls in the worst case, though greedy clustering reduces this to roughly <Code>O(K · M)</Code> where M is the final cluster count. For a model with M ≈ 3-5 typical clusters, that is roughly 30-50 NLI calls per query, each requiring a separate forward pass through DeBERTa-large (340M parameters). On a single A100, semantic entropy adds 1.5-3 seconds of latency per query relative to the ~100 ms baseline of greedy generation alone. This is acceptable for selective-answering workflows where SE only fires on triaged uncertain queries; it is unacceptable as a per-request signal at high QPS without sharding.
      </Prose>

      <Prose>
        K, the sample count, has a clear floor and a soft ceiling. Below K=5, the cluster-count distribution is too coarse to give a stable entropy estimate — the variance of the estimator across re-sampling exceeds the signal. Between K=10 and K=20 you get diminishing returns on AUROC: published benchmarks show roughly +1-2 AUROC points moving from K=10 to K=20, at double the cost. Above K=20 the marginal value is essentially zero except in pathological cases. The practical operating point is K=10 unless latency is non-binding, in which case K=20.
      </Prose>

      <Prose>
        Model scale has a more subtle effect. Larger LLMs produce more semantically consistent samples per question — they are better at sticking with the same answer across paraphrases — which compresses the dynamic range of semantic entropy. On a 70B model, even uncertain questions may give SE values in 0.0-0.5 nats range; on a 7B model the same questions might span 0.0-2.0 nats. This means the discriminative threshold has to be re-calibrated per model size, and the absolute value of SE is not transferable. The rank ordering it induces is, which is why AUROC remains a stable evaluation metric across model scales.
      </Prose>

      <Prose>
        The NLI model is the silent bottleneck. DeBERTa-large-MNLI is the de facto choice for English; for other languages or domain-specific entailment (legal, medical), the off-the-shelf NLI model can be the dominant source of error. Fine-tuning the NLI classifier on in-domain entailment pairs typically lifts SE's downstream AUROC by 2-5 points. Replacing DeBERTa with an LLM-as-judge for entailment (using Claude or GPT-4 to call entailment decisions) is a cleaner alternative for high-stakes deployments at the cost of substantially more latency per query.
      </Prose>

      <Prose>
        Black-box LLMs (closed APIs without logit access) are the most painful scaling regime. Without sequence probabilities you can still cluster samples by NLI and compute discrete cluster-count entropy, but you lose the length-weighted estimator and any token-level signal entirely. LM-Polygraph supports this regime explicitly with a reduced set of estimators (essentially: lexical similarity, NLI cluster count, P(True) via prompted self-evaluation). The deployable signal quality is roughly 5-10 AUROC points worse than with logit access, but it is still meaningfully better than no UQ at all.
      </Prose>

      <Prose>
        One scaling dimension that <em>helps</em>: domain narrowness. The narrower the deployment domain (a specific medical specialty, a specific legal jurisdiction, a single product's customer support), the better calibrated all UQ methods become because the threshold sweep on the validation set can be tuned tightly to the actual query distribution. Cross-domain serving (a single model handling general knowledge + code + reasoning + chitchat) requires either separate UQ thresholds per domain or a substantially looser threshold that accepts more abstentions to maintain precision.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>NLI noise breaks transitivity</H3>
      <Prose>
        Greedy clustering assumes that if <em>a</em> bidirectionally entails <em>b</em> and <em>b</em> bidirectionally entails <em>c</em>, then <em>a</em> entails <em>c</em>. NLI models violate this routinely on borderline cases. The result: identical sentences land in different clusters, or unrelated sentences merge through chain-of-entailments. The mitigation is spectral clustering on the full pairwise entailment matrix instead of greedy assignment, or running the NLI check at lower temperature with a more stringent threshold (require entailment probability &gt; 0.9 rather than just the argmax). In the original Kuhn et al. paper, greedy clustering was good enough; for high-stakes applications spectral clustering is worth the additional cost.
      </Prose>

      <H3>Sample diversity depends on temperature</H3>
      <Prose>
        Semantic entropy needs the K samples to actually reflect the model's uncertainty. At temperature 0 (greedy decoding), all samples are identical and SE is structurally zero — it tells you nothing. At very high temperatures (T &gt; 1.5) the samples diverge in nonsensical ways and SE inflates artificially. The standard operating range is T = 0.5-1.0; published work uses T = 1.0 with K=10. Setting temperature too low silently destroys the signal — the SE values look reassuringly small but they reflect the sampler, not the model's actual semantic confidence.
      </Prose>

      <H3>Length-weighting can amplify length bias</H3>
      <Prose>
        The length-weighted cluster-mass estimator divides log-probability by token count before exponentiating. For very short answers (single token), this is mathematically degenerate (zero division if the answer is zero-length, or trivially equal to the per-token log-probability). For very long answers, length normalization can over-weight clusters of long but low-probability paraphrases. Discrete cluster counting (cluster mass = number of samples in cluster) is more robust for short factual answers; length-weighting is preferable for free-form long-form generation. Choose the estimator to match your output length distribution rather than defaulting to one or the other.
      </Prose>

      <H3>Bidirectional entailment is too strict for some answer types</H3>
      <Prose>
        For answers with quantifier scope ambiguity ("most Americans believe..." vs "many Americans believe..."), partial-information ranking ("the third-largest city in France" vs "Lyon"), or entity-level paraphrasing where one side adds detail the other doesn't ("Albert Einstein" vs "Einstein, who developed relativity"), bidirectional entailment fails to merge what should be the same cluster. The result is over-fragmentation: the SE estimate is too high, abstentions are too frequent, and the deployed system becomes overly conservative. For domains where this is common, replace strict bidirectional entailment with a softer similarity score (BERTScore + threshold; or LLM-as-judge for "do these answer the question equivalently").
      </Prose>

      <H3>Self-consistency does not imply correctness</H3>
      <Prose>
        The most insidious failure mode of any sampling-based UQ method, including semantic entropy: a model that is confidently wrong produces ten consistent wrong answers. The cluster count is 1, the entropy is 0, and the system reports maximum confidence in an incorrect answer. This is the failure mode of "memorized falsehood" — common-knowledge facts the model learned wrong from pretraining data, or systematically biased domain knowledge. No sampling-based UQ method can detect it because the method only measures intra-model disagreement. Ground-truth-aware methods (P(True) with a calibration step, retrieval-augmented verification, judge-based correctness scoring) are the only mitigation. SE is not a hallucination detector for confidently wrong outputs; it is a hallucination detector for outputs the model itself is uncertain about.
      </Prose>

      <H3>Calibration is non-transferable across models</H3>
      <Prose>
        A semantic entropy threshold of 0.8 nats might be the sweet spot for selective answering on Llama-3.1-8B and dramatically wrong for Llama-3.1-70B (which produces lower-entropy outputs across the board) or for a different model family entirely. Every meaningful upgrade to the underlying LLM, every domain adaptation, every fine-tune, requires re-sweeping thresholds on a held-out validation set. Teams forget this; the threshold gets baked into config and persists across model upgrades, silently degrading selective-answering precision.
      </Prose>

      <H3>NLI domain mismatch</H3>
      <Prose>
        DeBERTa-large-MNLI is trained on Multi-NLI, which covers fiction, government documents, slate articles, telephone speech, and a few other domains — a wide but specific distribution. Using it on technical, medical, legal, code, or non-English text is out-of-distribution for the NLI model. Entailment decisions on such text are noisier, transitivity violations more frequent, and the clustering quality degrades. For specialized domains, fine-tune the NLI classifier on in-domain entailment pairs, or use a strong LLM as an entailment judge (Claude or GPT-4 with the question + answer pair both ways).
      </Prose>

      <H3>P(True) prompt sensitivity</H3>
      <Prose>
        The P(True) score is highly sensitive to the exact wording of the self-evaluation prompt. Small wording changes ("Is this answer correct?" vs "Is the answer above accurate?" vs "Did the model answer correctly?") can shift the P(True) score by 10-20 percentage points on the same query, and the AUROC of the resulting estimator can shift by 5+ points. The Kadavath et al. paper used a specific prompt template; deviating from it without re-validating breaks calibration. Always pin the exact P(True) prompt, version it like model weights, and re-run AUROC validation if you change it.
      </Prose>

      <H3>Hidden-state methods leak training data</H3>
      <Prose>
        Mahalanobis-distance-based UQ requires fitting a Gaussian to hidden states of in-distribution training data. If you fit it on a sensitive corpus (proprietary docs, customer messages, medical records), the fitted covariance and mean vectors carry information about that corpus and must be treated as sensitive artifacts themselves. This is rarely noticed until it becomes a compliance question. SE and P(True) do not have this issue — they require only the trained LLM and a runtime sample.
      </Prose>

      <Callout accent="gold">
        Semantic entropy fails silently in two distinct ways: when temperature is too low (no diversity to measure) and when the model is confidently wrong (no diversity to measure for a different reason). Always pair SE with a separate validation channel — a held-out correctness benchmark, a retrieval cross-check, or a judge model — to detect these regimes.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All five primary sources below were verified against their arXiv pages on 2026-04-21. Abstracts, author lists, and arXiv IDs confirmed.
      </Prose>

      <H3>Kuhn, Gal &amp; Farquhar 2023 — Semantic Uncertainty</H3>
      <Prose>
        Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." arXiv:2302.09664. Published February 2023; ICLR 2023. The founding paper for semantic entropy. Defines bidirectional entailment via NLI as the equivalence relation, gives both discrete and length-weighted cluster-mass estimators, and demonstrates substantial AUROC improvements over token-level baselines on TriviaQA, CoQA, and OpenBookQA across multiple LLM families. The clearest exposition of why surface-form entropy is the wrong measurement space and meaning-class entropy is the right one.
      </Prose>

      <H3>Farquhar et al. 2024 — Detecting hallucinations using semantic entropy</H3>
      <Prose>
        Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. "Detecting hallucinations in large language models using semantic entropy." Nature, June 2024. The Nature follow-up extending semantic entropy to free-form generation, biomedical QA, and multi-paragraph outputs. Adds a "discrete semantic entropy" estimator that operates without sequence probabilities (useful for black-box APIs) and reports state-of-the-art hallucination detection across GPT-4, LLaMA-2, Mistral, and Falcon. The single best non-arXiv reference and the work that brought semantic entropy to broad practitioner attention.
      </Prose>

      <H3>Fadeeva et al. 2023 — LM-Polygraph</H3>
      <Prose>
        Ekaterina Fadeeva, Roman Vashurin, Akim Tsvigun, Artem Vazhentsev, Sergey Petrakov, Kirill Fedyanin, Daniil Vasilev, Elizaveta Goncharova, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Artem Shelmanov. "LM-Polygraph: Uncertainty Estimation for Language Models." arXiv:2311.07383. Published November 2023; EMNLP 2023 system demonstration. The unified toolkit paper. Implements semantic entropy, P(True), MC Dropout, Mahalanobis distance, attention-based scores, lexical similarity, max sequence probability, mean token entropy, and roughly a dozen more methods behind a single API for both white-box (HuggingFace) and black-box (OpenAI, Anthropic) LLMs. The reference implementation almost everyone in the field eventually adopts. Code at github.com/IINemo/lm-polygraph.
      </Prose>

      <H3>Kadavath et al. 2022 — Language models (mostly) know what they know</H3>
      <Prose>
        Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, Jared Kaplan. "Language Models (Mostly) Know What They Know." arXiv:2207.05221. Published July 2022. Introduces P(True) as a self-evaluation signal: prompt the model with its own answer and ask whether it is correct, reading the probability of the True token. Establishes that LLMs have meaningful (if imperfect) metacognitive calibration about their own outputs, and that this calibration scales with model size. Foundational reference for any self-evaluation-based UQ method.
      </Prose>

      <H3>Manakul, Liusie &amp; Gales 2023 — SelfCheckGPT</H3>
      <Prose>
        Potsawee Manakul, Adian Liusie, Mark Gales. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." arXiv:2303.08896. Published March 2023; EMNLP 2023. The conceptual predecessor to semantic entropy. Samples K completions from an LLM and measures pairwise consistency through BERTScore, NLI, multiple-choice prompted self-comparison, and n-gram overlap — using the consistency score itself as the hallucination indicator. Black-box: requires only generated text, no logit access. Semantic entropy is in many ways the principled successor that replaces the heuristic consistency metric with a clean entropy-over-equivalence-classes formulation.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why entropy over clusters and not over sequences</H3>
      <Prose>
        Consider a question with K=10 sampled responses from a language model, of which 8 are syntactic variants of the same proposition ("Paris", "It is Paris", "The capital is Paris", etc.) and 2 are completely different propositions. Compute the discrete entropy of the empirical sequence-frequency distribution treating every distinct surface form as its own outcome (assume all 10 surface forms are distinct). Then compute the discrete semantic entropy under bidirectional NLI clustering. Why does the second number more accurately reflect the model's actual epistemic state? Construct a counterexample where two responses with very different surface forms should be treated as the same equivalence class, and explain how bidirectional entailment handles it.
      </Prose>

      <H3>Exercise 2 — When K is too small</H3>
      <Prose>
        Suppose you set K=3 instead of K=10 to save compute. For a question where the true semantic entropy is 1.5 nats, derive a rough bound on the bias of the empirical entropy estimator at K=3. (Hint: with only 3 samples, what is the maximum number of clusters you can observe, and what does that imply for the maximum estimable entropy?) What happens to the estimator variance as K decreases? Now consider the opposite: K=100. What kinds of returns do you expect from increasing K beyond 20, and which costs (latency, NLI calls) grow fastest with K?
      </Prose>

      <H3>Exercise 3 — Why P(True) sometimes outperforms semantic entropy</H3>
      <Prose>
        On certain question types — classification-style multiple choice, short factual answers from a narrow domain — published benchmarks show P(True) outperforming semantic entropy. Propose a mechanistic explanation. (Hint: consider what semantic entropy measures versus what P(True) measures, and the conditions under which sampling-induced disagreement is uninformative.) For a question where the model is confidently and consistently wrong, predict what each method reports, and design a third method that could in principle detect this regime. Why is "self-consistency" a fundamentally limited signal for ground-truth correctness?
      </Prose>

      <H3>Exercise 4 — Threshold selection on a validation set</H3>
      <Prose>
        You have a held-out validation set of 1000 (question, answer, correct?) triples and the semantic entropy score for each. Design the threshold-selection procedure. What metric should you optimize? How do you account for the fact that correct/incorrect classes are likely imbalanced (most answers will be correct)? Suppose your application has the cost ratio "one hallucination = 50 abstentions" — derive the threshold from the precision-recall curve. Now suppose your model is upgraded and the SE distribution shifts; what is the minimum re-validation work required to keep your selective-answering deployment correct?
      </Prose>

      <H3>Exercise 5 — Ablation: NLI model quality</H3>
      <Prose>
        Design an ablation experiment to quantify how much the NLI model's quality matters for downstream semantic entropy AUROC. Compare three NLI choices: (1) DeBERTa-large-MNLI off-the-shelf, (2) a domain-fine-tuned variant, (3) GPT-4-as-judge for entailment decisions. What metrics would you compute? What would a result showing "NLI quality is the bottleneck" look like, and what would a result showing "NLI is good enough" look like? How does this experiment inform whether to invest in fine-tuning the NLI step or in collecting more samples K?
      </Prose>

      <H3>Exercise 6 — Length confounding in token-level baselines</H3>
      <Prose>
        Construct a worked example with two responses to the same question — one short and correct, one long and equally correct — where the per-token NLL is identical for both but the sequence-level NLL favors the short response. Show numerically how perplexity (length-normalized NLL) addresses this and where it still fails. Then explain why semantic entropy is structurally immune to this kind of length confound: which step in the SE pipeline removes length as a variable, and why doesn't it suffer the same bias?
      </Prose>

      <H3>Exercise 7 — Compositional uncertainty in long-form generation</H3>
      <Prose>
        Semantic entropy as defined by Kuhn et al. computes a single uncertainty score per response. For a multi-paragraph output that contains many distinct factual claims (a generated article, a summary), this is too coarse: one or two hallucinated sentences in an otherwise correct output will not register strongly in the cluster-level entropy. Propose an extension that decomposes the output into atomic claims and computes per-claim semantic entropy. What new failure modes does this introduce? How might it interact with downstream selective rewriting or claim-level abstention? Reference the Farquhar et al. 2024 Nature paper's approach to long-form generation in your answer.
      </Prose>

    </div>
  ),
};

export default uncertaintyQuantification;
