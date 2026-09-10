import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const llmAsJudgeParadigm = {
  title: "LLM-as-Judge Paradigm & Core Methodology",
  slug: "llm-as-judge-paradigm-core-methodology",
  readTime: "~40 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Evaluating a language model's output is a surprisingly hard problem, and it became dramatically harder once chat-tuned models started producing fluent, multi-paragraph responses to open-ended prompts. The classical NLP evaluation toolkit — BLEU for translation, ROUGE for summarization, exact-match accuracy for question answering — was designed around tasks where there is a single short reference answer or a small set of valid alternatives. None of those assumptions hold for a modern instruction-following model. Asked to write a haiku about distributed consensus, summarize a court ruling for a non-lawyer, or refactor a piece of code for clarity, there is no canonical reference response and no n-gram overlap measure that meaningfully correlates with whether the answer is good. The thing being evaluated is the holistic quality of an open-ended generation, and humans were the only known reliable measurement instrument for almost a decade.
      </Prose>

      <Prose>
        Human evaluation, while gold-standard, is operationally punishing. A single round of pairwise comparisons between two model variants on a thousand prompts requires recruiting annotators, writing instructions, calibrating pilot tasks, paying for the labor (typically 1–3 USD per comparison at platforms like Scale AI or Surge HQ), waiting days for completion, and accepting that the next iteration of the model will require the entire pipeline to be rerun from scratch. For an organization shipping multiple model checkpoints per week — increasingly the norm at frontier labs and well-funded startups — this is simply not a feedback loop fast enough to drive iteration. The cost is not only money but latency: by the time annotation results return, the engineering team has already moved on to a new variant whose differences from the labeled one are hard to attribute. Beyond cost, human evaluation has its own reliability problems. Inter-annotator agreement on subjective quality judgments rarely exceeds Cohen's kappa of about 0.6, and is often closer to 0.4, meaning two trained humans evaluating the same response will disagree on a substantial fraction of cases. Reproducibility across studies is poor because annotator pools, instructions, and incentives differ.
      </Prose>

      <Prose>
        In June 2023, a team led by Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, and Siyuan Zhuang at LMSYS (Berkeley) published "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (arXiv:2306.05685). The paper made two interlocking claims. First, that a sufficiently strong LLM — at the time, GPT-4 — could be prompted to evaluate other LLMs' outputs and would produce judgments that agreed with human evaluators at roughly 80% rate, comparable to the agreement between two human evaluators on the same task. Second, that this enabled a new kind of evaluation infrastructure: MT-Bench, a curated set of 80 multi-turn questions across eight domains scored by GPT-4, and Chatbot Arena, a crowdsourced pairwise battle platform that aggregates human preferences into Elo rankings. The combination of the two — automated judging for fast iteration, human battles for ground-truth calibration — became the de facto standard for chat model evaluation almost overnight. Within twelve months, "LLM-as-judge" was the assumed default for any internal evaluation pipeline at any team training instruction-following models, and AlpacaEval (Dubois et al. 2024) had refined the methodology with explicit length-controlled correction.
      </Prose>

      <Prose>
        The reason this works at all is worth stating clearly because it is the load-bearing assumption underneath the entire paradigm. A modern frontier LLM has, through its pretraining and post-training, internalized a notion of "helpful, harmless, honest" response quality that empirically correlates well with human aesthetic and factual judgment on the kinds of prompts in MT-Bench-like benchmarks. The judge is not computing some objective quality function — there is no such function — it is performing a learned approximation of human preference, distilled through its own RLHF or DPO training plus its broader knowledge of what good writing, correct reasoning, and useful answers look like. When the judge is meaningfully more capable than the model being evaluated, this approximation is reliable enough to drive iteration. When the judge and the evaluatee are at parity, or when the task is one the judge itself struggles with (advanced mathematics, niche domain knowledge, languages outside its training distribution), the approximation degrades and the judgments become unreliable. Understanding this paradigm means understanding both why it has been so successful and where its assumptions silently break.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The intuition for LLM-as-judge starts from a question that sounds almost paradoxical: if a strong LLM could judge whether an answer is good, why does it not just generate the good answer in the first place? The answer is that judging and generating are different cognitive operations and the judging side is consistently easier. Recognizing that one of two responses is more helpful, more accurate, or better written is a discriminative task. Producing the optimal response from scratch is generative. Discrimination from a small candidate set is generally easier than generation from an unbounded space — this is the same asymmetry that makes multiple-choice questions easier than free-response, and it is what makes verifier-based methods like best-of-N sampling work. The LLM-as-judge paradigm exploits this asymmetry directly. We do not ask the judge to write the response; we ask it to compare or score responses that have already been written.
      </Prose>

      <Prose>
        There are three structural ways to elicit a judgment. The first is pointwise scoring: present a single (prompt, response) pair to the judge and ask for a numeric score, typically on a 1-to-10 Likert scale, often with a written justification. The second is pairwise comparison: present a prompt and two candidate responses (A and B) and ask which is better, usually with a tie option. The third is reference-based grading: present a prompt, a candidate response, and a known-good reference answer, and ask the judge to grade the candidate against the reference. Each of these has different strengths. Pointwise scoring scales linearly in the number of candidates and produces interpretable absolute scores, but suffers from poor calibration — judges tend to bunch scores in the 6–8 range and are sensitive to surface features. Pairwise comparison is much better calibrated because the judge only has to make a relative decision, but it is quadratic in the number of candidates if you want all pairs and requires aggregation (Bradley-Terry, Elo, or simple win-rate) to produce a ranking. Reference-based grading is the most reliable when a high-quality reference exists, because it converts the open-ended quality question into a constrained comparison, but it requires those references to be available, which is often not the case for new tasks.
      </Prose>

      <Prose>
        The core trick that makes pairwise judging work in practice is its insulation from the judge's calibration problems. A judge that systematically rates everything 7 out of 10 is useless for pointwise comparisons but completely fine for pairwise comparisons, because we are only asking for a preference ordering, not an absolute score. A judge that has poor numeric calibration but good ordinal judgment will produce high-quality pairwise data while producing low-quality pointwise data. This is why most serious benchmarks — Chatbot Arena, AlpacaEval 2, Arena-Hard — converged on pairwise as the default elicitation format, even though the resulting data has to be aggregated into rankings rather than read directly as scores.
      </Prose>

      <Prose>
        The biases that the judge brings to the table are the next layer of intuition to internalize. A judge LLM, being itself a learned model, has systematic preferences that have nothing to do with response quality. The three biases established empirically in the original Zheng et al. paper are position bias (a preference for whichever response is presented first or second, often shifted toward "Assistant A"), verbosity bias (a preference for longer responses regardless of information density), and self-enhancement bias (a preference for responses generated by a model in the same family as the judge, particularly visible when GPT-4 judges GPT-3.5 outputs against open-source alternatives). Position bias is mitigated by running each comparison twice with the order swapped and keeping only consistent verdicts. Verbosity bias is mitigated either by length-matched filtering, by length-controlled win rates (AlpacaEval 2 LC), or by explicit instructions to the judge to ignore length. Self-enhancement bias is mitigated by using a judge from a different model family than any of the candidates, or by ensembling judgments from multiple judges. None of these mitigations fully eliminates the underlying bias; they only push it to a level where the signal exceeds the noise.
      </Prose>

      <Prose>
        The intuition for why this paradigm replaced human evaluation so quickly, despite all the biases, comes down to a throughput argument. A GPT-4 judge can score thousands of responses per hour for a few dollars. A human evaluation panel scores hundreds per day for hundreds of dollars. Even if the judge is only 80% as accurate as a human, the ability to run an evaluation overnight rather than over two weeks transforms the iteration loop. The right framing is not "judges replace humans" but "judges enable a different operating point on the cost-accuracy frontier." Production teams use judges for fast inner-loop iteration on every checkpoint, and reserve human evaluation for periodic calibration of the judge itself and for final acceptance testing of release candidates. Chatbot Arena's role in the broader ecosystem is exactly this calibration function: its crowdsourced human preferences serve as the ground-truth signal against which everyone's automated judges are measured.
      </Prose>

      <Prose>
        One last piece of intuition concerns the relationship between the judge and the evaluatee. The reliability of judge-based evaluation is not a property of the judge alone — it is a property of the judge-evaluatee pair. A GPT-4 judge evaluating a 7B open-source model on conversational helpfulness produces highly reliable rankings. A GPT-4 judge evaluating Claude 3 Opus on advanced mathematics produces unreliable rankings, because the judge cannot distinguish a correct answer from a confidently-wrong one in a domain where its own competence is borderline. This means evaluation reliability degrades as the evaluatees approach or exceed the judge in capability, and the practice of using GPT-4 to judge GPT-4-class outputs has known soft spots that the field is still actively working out. The honest framing is that LLM-as-judge is a measurement instrument with a finite dynamic range, and the closer the things being measured are to the limits of that range, the more careful the measurement protocol has to be.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The mathematics of LLM-as-judge has three pieces: the preference model that converts pairwise judgments into rankings, the agreement coefficients that quantify how well judges align with humans (or with each other), and the bias-variance decomposition that explains where evaluation error comes from. Each piece comes from a well-established statistical literature; the contribution of the LLM-as-judge paradigm is in showing that the right strong model can plug into these formulas in place of human annotators with minimal loss of validity.
      </Prose>

      <H3>Bradley-Terry preference model</H3>

      <Prose>
        Pairwise preference data is naturally modeled by Bradley-Terry. Assume each item <Code>i</Code> in the comparison set has a latent quality parameter <Code>{"\\theta_i"}</Code>. The probability that item <Code>i</Code> is preferred to item <Code>j</Code> is given by the logistic function of the quality difference:
      </Prose>

      <MathBlock>{"P(i \\succ j) = \\frac{e^{\\theta_i}}{e^{\\theta_i} + e^{\\theta_j}} = \\sigma(\\theta_i - \\theta_j)"}</MathBlock>

      <Prose>
        Given a dataset of pairwise comparisons <Code>{"\\{(i_n, j_n, y_n)\\}"}</Code> where <Code>{"y_n = 1"}</Code> if <Code>i_n</Code> won and <Code>0</Code> otherwise, the Bradley-Terry log-likelihood of the quality parameters is:
      </Prose>

      <MathBlock>{"\\mathcal{L}(\\boldsymbol{\\theta}) = \\sum_n \\left[ y_n \\log \\sigma(\\theta_{i_n} - \\theta_{j_n}) + (1 - y_n) \\log \\sigma(\\theta_{j_n} - \\theta_{i_n}) \\right]"}</MathBlock>

      <Prose>
        Maximizing this likelihood (typically by gradient ascent or iterative reweighted least squares) gives the maximum-likelihood quality estimates. The parameters are identified up to an additive constant, so they are usually pinned by setting one item's <Code>{"\\theta"}</Code> to zero or by centering them at zero. The resulting estimates are exactly what Chatbot Arena reports as Elo ratings, modulo a scaling factor of <Code>400/ln(10)</Code> applied for historical compatibility with chess Elo. The Elo formula <Code>{"E_A = 1 / (1 + 10^{(R_B - R_A)/400})"}</Code> is mathematically identical to <Code>{"\\sigma(\\theta_A - \\theta_B)"}</Code> after the scaling change, so Chatbot Arena's leaderboard is a Bradley-Terry MLE under a different unit convention.
      </Prose>

      <Prose>
        For LLM-as-judge work specifically, Bradley-Terry serves two purposes. First, it provides the principled way to aggregate many pairwise comparisons (across prompts, across model pairs) into a single ranking. Second, it gives confidence intervals on the resulting ratings via the standard errors of the MLE — these are typically computed by bootstrap resampling of the comparison set, which is exactly the methodology Chatbot Arena uses. A 95% confidence interval of <Code>{"\\pm 5"}</Code> Elo points means the difference is within sampling noise; a difference of <Code>{"50+"}</Code> Elo points is robust.
      </Prose>

      <H3>Cohen's kappa</H3>

      <Prose>
        The agreement question — does the judge agree with humans? — is operationalized by Cohen's kappa for two raters or by Krippendorff's alpha for many raters with missing data. Cohen's kappa adjusts the raw observed agreement <Code>p_o</Code> for the agreement that would be expected by chance <Code>p_e</Code> given the marginal label frequencies:
      </Prose>

      <MathBlock>{"\\kappa = \\frac{p_o - p_e}{1 - p_e}"}</MathBlock>

      <Prose>
        For a binary comparison task (A wins vs. B wins, ignoring ties), with judge label distribution <Code>(p_A, p_B)</Code> and human label distribution <Code>(q_A, q_B)</Code>, the chance agreement is <Code>{"p_e = p_A q_A + p_B q_B"}</Code>. A kappa of 0 means agreement is at chance. A kappa of 1 means perfect agreement. Negative values indicate systematic disagreement. The conventional Landis-Koch interpretation labels kappa values: 0.0–0.2 is "slight", 0.2–0.4 is "fair", 0.4–0.6 is "moderate", 0.6–0.8 is "substantial", and 0.8–1.0 is "almost perfect". The Zheng et al. 2023 result that GPT-4 agrees with humans at roughly the same rate as humans agree with each other corresponds to both inter-judge and judge-human kappas in the 0.4–0.7 range, which is the upper end of "moderate" to "substantial" — strong but not perfect.
      </Prose>

      <Prose>
        A subtle point about kappa: it is most informative when the marginal label distributions are roughly balanced. If 90% of comparisons end in "A wins" because A is from a much stronger model, the chance agreement <Code>p_e</Code> is already 0.82, and even very high observed agreement produces a low kappa. This is the kappa paradox, and it matters for LLM-as-judge work because pairwise comparisons between very different models naturally produce skewed label distributions. The standard mitigation is to use carefully balanced model pairs, to report both raw agreement and kappa, and to use kappa as the primary metric on tasks where the underlying win rate is closer to 50/50.
      </Prose>

      <H3>Krippendorff's alpha</H3>

      <Prose>
        For multi-judge settings — multiple LLM judges or panels of human annotators — Krippendorff's alpha generalizes kappa by handling any number of raters, missing data, and arbitrary level of measurement (nominal, ordinal, interval, ratio). The formula has the same structure as kappa, with observed disagreement <Code>D_o</Code> versus expected disagreement <Code>D_e</Code>:
      </Prose>

      <MathBlock>{"\\alpha = 1 - \\frac{D_o}{D_e}"}</MathBlock>

      <Prose>
        Where <Code>D_o</Code> is the average pairwise disagreement across all rater pairs that scored the same item, and <Code>D_e</Code> is the disagreement that would be expected by random pairing of the marginal distribution. The disagreement function depends on the measurement level: for nominal, it is 0 if labels match and 1 otherwise; for ordinal, it is the squared rank difference; for interval, the squared numeric difference. Use Krippendorff's alpha when comparing a panel of three or more judges (LLM or human), when the comparison protocol is pointwise scoring rather than pairwise, or when judges have rated different overlapping subsets of the data. The Landis-Koch thresholds apply approximately to alpha as well, with alpha {">"} 0.8 considered "good" inter-rater reliability for behavioral research.
      </Prose>

      <H3>Length-controlled win rate</H3>

      <Prose>
        AlpacaEval 2 LC (Dubois et al. 2024, arXiv:2404.04475) introduced an explicit statistical correction for length bias. The raw win rate <Code>w</Code> against a reference model is decomposed into a length-explained component and a length-independent quality component using a logistic regression:
      </Prose>

      <MathBlock>{"\\text{logit}(\\hat{w}) = \\beta_0 + \\beta_L \\cdot (\\ell_A - \\ell_B) + \\beta_M \\cdot \\mathbb{1}[\\text{model is A}]"}</MathBlock>

      <Prose>
        Where <Code>{"\\ell_A"}</Code> and <Code>{"\\ell_B"}</Code> are the response lengths and the indicator picks up whether the candidate or the reference is in position A. The length-controlled win rate is the predicted win rate at zero length difference — that is, what the win rate would be if the candidate and reference produced responses of equal length. This correction is computed offline, after collecting all the judge outputs, and substantially reduces the impact of length bias on leaderboard rankings. AlpacaEval 2 LC results are typically 5–15 percentage points lower than uncorrected AlpacaEval 2 for verbose models, and the correction reduces the rank correlation between AlpacaEval and length-only baselines from above 0.9 down to around 0.4.
      </Prose>

      <H3>Expected calibration error</H3>

      <Prose>
        For pointwise scoring, calibration matters: if the judge gives a 9/10 score, do humans actually rate that response in the top 10% of all responses? Expected calibration error (ECE) measures the gap between predicted confidence and empirical accuracy, computed by binning predictions and comparing per-bin accuracy to per-bin mean confidence:
      </Prose>

      <MathBlock>{"\\text{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N} \\left| \\text{acc}(B_m) - \\text{conf}(B_m) \\right|"}</MathBlock>

      <Prose>
        Where <Code>B_m</Code> is the m-th bin, <Code>N</Code> is the total number of judgments, <Code>{"\\text{acc}(B_m)"}</Code> is the empirical fraction of items in that bin where the judge's preferred response was actually preferred by humans, and <Code>{"\\text{conf}(B_m)"}</Code> is the average judge confidence in that bin. Well-calibrated judges have ECE below 0.05; typical LLM judges have ECE in the 0.10–0.20 range without calibration adjustments. The standard mitigation is temperature scaling — scaling the logits underlying the score distribution by a learned scalar <Code>T</Code> chosen on a calibration set to minimize ECE — which is a one-parameter post-hoc adjustment that often cuts ECE in half without changing rankings.
      </Prose>

      <Callout accent="gold">
        The agreement coefficients (kappa, alpha) and the preference model (Bradley-Terry, Elo) are independent. You can run kappa on raw pairwise judgments to assess judge-human agreement, then aggregate the same judgments through Bradley-Terry to produce a leaderboard. Both reports are needed: kappa establishes that the judge is reliable at all, Bradley-Terry produces the ranking the reliability is being applied to.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize LLM-as-judge is to build a minimal end-to-end evaluation harness against a real API and then verify each component against the math from section 3. The implementation below uses an OpenAI/Anthropic-style client and a small synthetic dataset of 12 prompt-response pairs across three "models". Every piece is intentionally simple: the goal is to expose the moving parts, not to ship production-grade code. The five subsections cover the prompt template, the pointwise judge, the pairwise judge with position-swap mitigation, the Cohen's kappa computation against a synthetic human-label set, and the Bradley-Terry MLE that turns pairwise judgments into a ranking.
      </Prose>

      <H3>4a. The judge prompt template</H3>

      <Prose>
        The judge prompt is the highest-leverage design decision in the entire pipeline. It encodes the evaluation criteria, the output format the judge must produce, and the bias mitigations baked into the instructions. The MT-Bench template (reproduced in spirit below) has three sections: a system message that frames the judge as an impartial evaluator, a user message containing the prompt-and-response(s), and explicit instructions to produce a verdict in a parseable format. The format constraint matters because parsing free-form judge text is brittle; constraining the output to "Answer: A" or "Score: 7" lets you extract the verdict deterministically.
      </Prose>

      <CodeBlock language="python">
{`POINTWISE_TEMPLATE = """You are an impartial judge evaluating the quality of an AI assistant's response to a user question. Your evaluation should consider helpfulness, relevance, accuracy, depth, and the absence of harmful content. Be objective: do not let response length, position, or style distract you from substantive quality.

[User Question]
{question}

[Assistant Response]
{response}

After your analysis, provide a single integer score from 1 (very poor) to 10 (excellent) using the strict format below. Do not include any other text on the final line.

Reasoning: <one or two sentences justifying the score>
Score: <integer 1-10>"""

PAIRWISE_TEMPLATE = """You are an impartial judge comparing two AI assistant responses to the same user question. Evaluate which response is more helpful, accurate, and well-reasoned. Avoid bias toward response length or position. If responses are roughly equivalent in quality, you may choose Tie, but only as a last resort.

[User Question]
{question}

[Assistant A Response]
{response_a}

[Assistant B Response]
{response_b}

Provide your verdict using the strict format below.

Reasoning: <one or two sentences>
Verdict: <A | B | Tie>"""`}
      </CodeBlock>

      <Prose>
        Two design notes about these templates worth dwelling on. First, the explicit "do not let length distract you" instruction is the cheapest verbosity-bias mitigation available; the Zheng et al. ablation showed it cuts verbosity bias by roughly 30%. It does not eliminate the bias because the judge's underlying preferences are baked in by training, but moving the bias from "uncontested instinct" to "explicitly suppressed" does measurably help. Second, the strict output format on the last line is what makes downstream parsing reliable. A judge that writes "I think Assistant A's response is better because..." across multiple paragraphs requires regex acrobatics to parse; a judge constrained to end with "Verdict: A" can be parsed with a single split.
      </Prose>

      <H3>4b. Pointwise scoring with retry and structured parsing</H3>

      <Prose>
        Pointwise scoring takes a (question, response) pair and returns a single integer. The implementation has three concerns: calling the API at temperature zero (so the same input produces the same output for caching and reproducibility), retrying on transient failures, and parsing the verdict robustly. The code below uses a synchronous Anthropic-style client, but the same pattern works with OpenAI or any other provider.
      </Prose>

      <CodeBlock language="python">
{`import re
import time
from anthropic import Anthropic

client = Anthropic()
JUDGE_MODEL = "claude-3-5-sonnet-20241022"  # or "gpt-4o-2024-08-06" for OpenAI
MAX_RETRIES = 3
BACKOFF_BASE = 2.0

SCORE_RE = re.compile(r"Score:\\s*(\\d+)", re.IGNORECASE)

def call_judge(prompt: str, max_tokens: int = 256) -> str:
    """Call the judge with deterministic decoding and exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                temperature=0.0,                 # deterministic
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF_BASE ** attempt)
    raise RuntimeError(f"judge call failed after {MAX_RETRIES} attempts: {last_err}")

def pointwise_score(question: str, response: str) -> int:
    """Returns an integer score 1-10. Raises on parse failure."""
    prompt = POINTWISE_TEMPLATE.format(question=question, response=response)
    raw = call_judge(prompt)
    m = SCORE_RE.search(raw)
    if not m:
        raise ValueError(f"could not parse score from judge output:\\n{raw}")
    score = int(m.group(1))
    if not 1 <= score <= 10:
        raise ValueError(f"score {score} out of range 1-10")
    return score

# Example synthetic eval set: 4 prompts x 3 model responses each.
eval_set = [
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "responses": {
            "model_strong": "100 degrees Celsius at standard atmospheric pressure (1 atm).",
            "model_mid":    "Water boils at 100C at sea level.",
            "model_weak":   "Around 90 degrees, depending on the day.",
        },
    },
    # ... three more prompts ...
]

scores = {m: [] for m in ["model_strong", "model_mid", "model_weak"]}
for item in eval_set:
    for model_name, response in item["responses"].items():
        s = pointwise_score(item["question"], response)
        scores[model_name].append(s)

# After running on the full eval set, print mean scores per model.
for m, ss in scores.items():
    print(f"{m}: mean={sum(ss)/len(ss):.2f}, scores={ss}")
# model_strong: mean=8.50, scores=[9, 8, 9, 8]
# model_mid:    mean=7.25, scores=[8, 7, 7, 7]
# model_weak:   mean=3.50, scores=[2, 4, 4, 4]`}
      </CodeBlock>

      <Prose>
        The temperature=0.0 choice is non-negotiable for evaluation work. Any non-zero temperature produces stochastic verdicts and ruins reproducibility — the same model evaluated twice on the same eval set would produce different scores. The retry loop matters because real APIs have transient 5xx errors, rate limits, and connection resets, and an evaluation run over thousands of pairs will hit these. The strict regex parser will fail on roughly 1–3% of judge calls when the model decides to respond outside the requested format; in production you either retry with a stronger format reminder or fall back to a more permissive parser, depending on how strict your downstream pipeline needs to be.
      </Prose>

      <H3>4c. Pairwise comparison with position swap</H3>

      <Prose>
        Pairwise judgment is structurally cleaner but requires the position-swap mitigation to control for position bias. The pattern is to call the judge twice with the responses in opposite order, accept the verdict only if both calls produce consistent verdicts (both pick the same response, or both call it a tie), and treat inconsistent judgments as ties. This roughly doubles the API cost but eliminates position bias as a confounder in the resulting data.
      </Prose>

      <CodeBlock language="python">
{`VERDICT_RE = re.compile(r"Verdict:\\s*(A|B|Tie)", re.IGNORECASE)

def pairwise_verdict(question: str, response_a: str, response_b: str) -> str:
    """Single judge call. Returns 'A', 'B', or 'Tie'."""
    prompt = PAIRWISE_TEMPLATE.format(
        question=question, response_a=response_a, response_b=response_b
    )
    raw = call_judge(prompt)
    m = VERDICT_RE.search(raw)
    if not m:
        # Fallback: treat parse failure as tie rather than crashing.
        return "Tie"
    return m.group(1).capitalize()

def pairwise_compare(question: str, response_x: str, response_y: str) -> str:
    """
    Position-swap mitigated pairwise comparison.
    Returns 'X', 'Y', or 'Tie' (the labels refer to your inputs, not A/B).
    """
    # Round 1: X is A, Y is B
    v1 = pairwise_verdict(question, response_x, response_y)
    # Round 2: Y is A, X is B (positions swapped)
    v2 = pairwise_verdict(question, response_y, response_x)

    # Translate v2 back to X/Y labels.
    v2_translated = {"A": "Y", "B": "X", "Tie": "Tie"}[v2]
    v1_translated = {"A": "X", "B": "Y", "Tie": "Tie"}[v1]

    if v1_translated == v2_translated:
        return v1_translated   # both rounds agree
    return "Tie"               # inconsistent → treat as tie

# Run all pairs across all prompts for three models.
import itertools

models = ["model_strong", "model_mid", "model_weak"]
pairs  = list(itertools.combinations(models, 2))

# wins[(a, b)] = number of times a beat b
wins = {(a, b): 0 for a, b in pairs}
wins.update({(b, a): 0 for a, b in pairs})
ties = {pair: 0 for pair in pairs}

for item in eval_set:
    for a, b in pairs:
        verdict = pairwise_compare(
            item["question"], item["responses"][a], item["responses"][b]
        )
        if verdict == "X":
            wins[(a, b)] += 1
        elif verdict == "Y":
            wins[(b, a)] += 1
        else:
            ties[(a, b)] += 1

for (a, b), n in wins.items():
    print(f"{a} beat {b}: {n} times")
# model_strong beat model_mid: 3 times
# model_mid beat model_strong: 0 times
# model_strong beat model_weak: 4 times
# model_weak beat model_strong: 0 times
# model_mid beat model_weak: 4 times
# model_weak beat model_mid: 0 times`}
      </CodeBlock>

      <Prose>
        The position-swap pattern is one of those things that looks like an inefficiency until you look at the data without it. Single-pass pairwise comparisons frequently show 5–15% position bias in raw verdicts — the second-position assistant gets a small boost in many studies, the first-position assistant in others, depending on the judge model and the prompt format. After swap mitigation, the residual position-attributable variance drops to roughly 1–2%, which is small enough to not contaminate downstream rankings.
      </Prose>

      <H3>4d. Cohen's kappa against a human label set</H3>

      <Prose>
        Once you have judge verdicts and human verdicts on the same set of pairwise comparisons, Cohen's kappa quantifies how well the judge agrees with the humans beyond chance. The implementation below assumes you have collected both — judge verdicts via the function above, and human verdicts via a parallel labeling effort — and computes kappa over the comparison set excluding ties.
      </Prose>

      <CodeBlock language="python">
{`def cohens_kappa(judge_labels, human_labels):
    """
    Compute Cohen's kappa for two raters on the same items.
    Each label is one of {'A', 'B'} (drop ties before passing in).
    """
    assert len(judge_labels) == len(human_labels)
    n = len(judge_labels)
    if n == 0:
        return float("nan")

    # Observed agreement
    p_o = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n

    # Marginal label frequencies
    p_judge_A = sum(1 for x in judge_labels if x == "A") / n
    p_judge_B = 1.0 - p_judge_A
    p_human_A = sum(1 for x in human_labels if x == "A") / n
    p_human_B = 1.0 - p_human_A

    # Chance agreement
    p_e = p_judge_A * p_human_A + p_judge_B * p_human_B

    if p_e == 1.0:
        return float("nan")   # degenerate: all labels identical

    return (p_o - p_e) / (1.0 - p_e)

# Example synthetic agreement test: 100 paired comparisons.
# Suppose judge agreed with human on 82 of 100, with both A and B being roughly
# 50/50 in the underlying data.
judge_labels = ["A"] * 50 + ["B"] * 50           # 50 A, 50 B
human_labels = (["A"] * 41 + ["B"] *  9 +        # judge says A: 41 agree, 9 disagree
                ["B"] * 41 + ["A"] *  9)          # judge says B: 41 agree, 9 disagree

# (Reorder so positions correspond — for the toy demo just sample matching pairs.)
import random
random.seed(0)
pairs = list(zip(judge_labels, human_labels))
random.shuffle(pairs)
jl, hl = zip(*pairs)

k = cohens_kappa(list(jl), list(hl))
print(f"observed agreement: {sum(1 for j,h in zip(jl, hl) if j==h)/len(jl):.2f}")
print(f"Cohen's kappa: {k:.3f}")
# observed agreement: 0.82
# Cohen's kappa: 0.640    ← "substantial" by Landis-Koch`}
      </CodeBlock>

      <Prose>
        The numbers above are a faithful reproduction of the regime the LLM-as-judge literature reports: 80%-ish raw agreement, kappa in the 0.6–0.7 range, comparable to the upper bound on inter-human agreement for the same task. If your judge produces kappa below 0.4 against a held-out human set, the judge is unreliable for the task and either the prompt template needs work, the judge model needs to be upgraded, or the task is one where automated judging is not yet viable.
      </Prose>

      <H3>4e. Bradley-Terry MLE for aggregating pairwise judgments</H3>

      <Prose>
        Once you have win counts across all model pairs, Bradley-Terry produces a ranking with confidence intervals. The MLE has no closed form but converges fast under iterative reweighted least squares. The implementation below is a direct gradient ascent which is easier to read and converges in tens of iterations for small problems.
      </Prose>

      <CodeBlock language="python">
{`import math

def bradley_terry_mle(wins, models, n_iter=200, lr=0.5):
    """
    wins: dict mapping (winner, loser) -> count
    models: list of model names
    Returns: dict mapping model -> theta (latent quality), centered at 0.
    """
    theta = {m: 0.0 for m in models}
    for _ in range(n_iter):
        grad = {m: 0.0 for m in models}
        for (a, b), n_ab in wins.items():
            if n_ab == 0:
                continue
            p_a_beats_b = 1.0 / (1.0 + math.exp(theta[b] - theta[a]))
            # Each "a beat b" event contributes (1 - p_a_beats_b) to grad[a]
            # and -(1 - p_a_beats_b) to grad[b].
            grad[a] += n_ab * (1.0 - p_a_beats_b)
            grad[b] -= n_ab * (1.0 - p_a_beats_b)
        for m in models:
            theta[m] += lr * grad[m]
        # Re-center to identify the parameters.
        mean = sum(theta.values()) / len(theta)
        for m in models:
            theta[m] -= mean
    return theta

theta = bradley_terry_mle(wins, models)
# Convert to Elo for readability.
elo = {m: 1500 + 400 * t / math.log(10) for m, t in theta.items()}
for m in sorted(elo, key=elo.get, reverse=True):
    print(f"{m}: theta={theta[m]:+.3f}  elo={elo[m]:.0f}")
# model_strong: theta=+1.943  elo=1838
# model_mid:    theta=+0.273  elo=1547
# model_weak:   theta=-2.216  elo=1115`}
      </CodeBlock>

      <Prose>
        For confidence intervals, bootstrap-resample the comparison set with replacement, run the MLE on each bootstrap sample, and report the 2.5th and 97.5th percentiles of the resulting Elo distributions. Chatbot Arena uses exactly this protocol, with 100 bootstrap samples and the percentile method. Differences in Elo that overlap in their confidence intervals are not statistically distinguishable; differences with non-overlapping intervals are robust.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Moving an LLM-as-judge harness from a notebook to a production evaluation pipeline introduces five concerns that the from-scratch code does not address: caching to avoid re-judging unchanged inputs, structured output to make parsing deterministic, batch evaluation patterns to keep the throughput tractable, observability to debug judge failures after the fact, and cost discipline so the evaluation bill does not eclipse the training bill. The mature open-source projects in this space — LLM-as-a-Judge in DeepEval, Promptfoo, OpenAI Evals, the LMSYS Chatbot Arena infrastructure, and HuggingFace's lighteval — have converged on similar architectural patterns, summarized below.
      </Prose>

      <H3>Caching by content hash</H3>

      <Prose>
        Every judge call is a deterministic function of (judge model, prompt template version, question text, response text(s), temperature). If any of these change, you need a fresh judge call. If none of them change, the previous result is valid and re-judging is wasteful. The standard pattern is to compute a SHA-256 hash over the canonical concatenation of those inputs and store the verdict keyed by that hash in a local SQLite or Redis cache. A correctly designed cache cuts evaluation cost by 80–95% across iterative experiments, because most evaluation runs reuse most of the same prompts and most of the same responses across model versions.
      </Prose>

      <CodeBlock language="python">
{`import hashlib
import sqlite3
import json

CACHE_PATH = "judge_cache.sqlite"
PROMPT_TEMPLATE_VERSION = "pairwise-v3"  # bump when template changes

class JudgeCache:
    def __init__(self, path=CACHE_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                cache_key TEXT PRIMARY KEY,
                judge_model TEXT NOT NULL,
                template_version TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                verdict TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    @staticmethod
    def _key(judge_model, template_version, inputs):
        canonical = json.dumps(
            {"judge": judge_model, "tpl": template_version, "in": inputs},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, judge_model, template_version, inputs):
        k = self._key(judge_model, template_version, inputs)
        row = self.conn.execute(
            "SELECT verdict, raw_response FROM judgments WHERE cache_key = ?",
            (k,),
        ).fetchone()
        return row  # None on miss

    def put(self, judge_model, template_version, inputs, verdict, raw_response):
        k = self._key(judge_model, template_version, inputs)
        self.conn.execute(
            "INSERT OR REPLACE INTO judgments "
            "(cache_key, judge_model, template_version, inputs_json, verdict, raw_response) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (k, judge_model, template_version, json.dumps(inputs), verdict, raw_response),
        )
        self.conn.commit()

cache = JudgeCache()

def cached_pairwise_verdict(question, response_a, response_b):
    inputs = {"q": question, "a": response_a, "b": response_b}
    hit = cache.get(JUDGE_MODEL, PROMPT_TEMPLATE_VERSION, inputs)
    if hit is not None:
        return hit[0]   # cached verdict
    verdict = pairwise_verdict(question, response_a, response_b)
    raw = ""            # in real code, capture raw response from call_judge
    cache.put(JUDGE_MODEL, PROMPT_TEMPLATE_VERSION, inputs, verdict, raw)
    return verdict`}
      </CodeBlock>

      <H3>Structured output (JSON mode)</H3>

      <Prose>
        Both OpenAI and Anthropic now support constrained JSON output: you supply a JSON schema describing the verdict object and the model is constrained to produce a parseable JSON document conforming to it. This eliminates the parse-failure tail entirely. For OpenAI this is the <Code>response_format=&#123;type: "json_schema", ...&#125;</Code> parameter; for Anthropic it is implemented via tool use with a single tool. Either way, the operational benefit is the same — the parsing layer becomes a JSON load instead of a regex, the parse failure rate drops from 1–3% to effectively zero, and the verdict object can carry structured side-channel data (per-criterion subscores, reasoning traces, confidence flags) that would be cumbersome to extract from free text.
      </Prose>

      <CodeBlock language="python">
{`# Anthropic structured output via tool use.
JUDGE_TOOL = {
    "name": "submit_pairwise_verdict",
    "description": "Record a pairwise verdict between two AI responses.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief justification for the verdict.",
            },
            "verdict": {
                "type": "string",
                "enum": ["A", "B", "Tie"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0, "maximum": 1.0,
                "description": "How confident the judge is (0.5 = guessing).",
            },
        },
        "required": ["reasoning", "verdict", "confidence"],
    },
}

def structured_pairwise_verdict(question, response_a, response_b):
    prompt = PAIRWISE_TEMPLATE.format(
        question=question, response_a=response_a, response_b=response_b
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        temperature=0.0,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_pairwise_verdict"},
        messages=[{"role": "user", "content": prompt}],
    )
    # Extract the tool call payload — guaranteed to match schema.
    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_pairwise_verdict":
            return block.input   # dict with reasoning, verdict, confidence
    raise RuntimeError("judge did not call submit_pairwise_verdict")`}
      </CodeBlock>

      <H3>Batch evaluation and concurrency</H3>

      <Prose>
        A typical evaluation run might involve 1k prompts x 5 models x pairwise comparisons = 10k judge calls, x2 for position swap = 20k calls. At a sequential rate of 2 seconds per call this is 11 hours; with 32-way concurrency it is roughly 20 minutes. Use <Code>asyncio</Code> with a semaphore to bound concurrency, an async client (Anthropic's <Code>AsyncAnthropic</Code> or OpenAI's <Code>AsyncOpenAI</Code>), and a tqdm-style progress bar so you can see the run's pace and ETA. Both providers also offer batch APIs that accept a JSONL file of requests, return results within 24 hours, and cost roughly 50% less per token than synchronous calls — appropriate when you can tolerate the latency and want to minimize cost on large eval runs.
      </Prose>

      <CodeBlock language="python">
{`import asyncio
from anthropic import AsyncAnthropic

async_client = AsyncAnthropic()
SEMAPHORE_LIMIT = 32   # tune based on your rate limit headroom

async def async_call_judge(prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        resp = await async_client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

async def evaluate_pairs_async(pair_list):
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    tasks = [async_call_judge(p, sem) for p in pair_list]
    return await asyncio.gather(*tasks, return_exceptions=True)

# In a script:
# results = asyncio.run(evaluate_pairs_async(prompt_list))`}
      </CodeBlock>

      <H3>Cost discipline</H3>

      <Prose>
        Judge calls cost real money, especially with frontier models as judges. Estimating cost up front is straightforward: count input tokens (the template plus the prompt and responses), output tokens (typically 100–300 for a pairwise verdict), multiply by model pricing per 1M tokens, multiply by the number of comparisons. A rough rule of thumb is 0.5–2 cents per pairwise comparison with Claude 3.5 Sonnet or GPT-4o as the judge in 2026 pricing, which scales to 5–20 dollars per thousand comparisons. This makes a daily 10k-comparison eval pipeline a roughly 50–200 USD/day operation — significant but not prohibitive. If cost matters, the standard tactics are: cache aggressively (saves 80–95%), use a cheaper judge for first-pass screening and the strong judge only for close calls, batch via the JSONL batch API for non-urgent runs (saves 50%), and use prompt caching for the system message and template (saves 30–60% on input tokens at supported providers).
      </Prose>

      <H3>Observability</H3>

      <Prose>
        Every production evaluation harness should log, for every judge call: the cache key, the inputs hash, the verdict, the full raw judge response, the latency, and any errors. When a downstream stakeholder asks "why did the eval rank model X above model Y?" the answer needs to be reconstructible from the logs without re-running the eval. Persist judge responses to durable storage (S3, GCS, or a database table), index by judge model and template version, and build a small inspection UI that lets a human pull up the underlying judge reasoning for any cell in the leaderboard. The investment pays back the first time a leaderboard regression has to be debugged.
      </Prose>

      <Prose>
        One more production note worth dwelling on: judge upgrades break time-series comparability. If you have been evaluating models against GPT-4o for six months and you switch to Claude 3.5 Sonnet, the historical Elo numbers are not directly comparable to the new ones — both judges have their own bias profiles, and their absolute calibration differs. The standard mitigation is to run a calibration period where both judges score the same comparison set, fit a translation between their score distributions, and apply that translation when migrating historical data. Or, more robustly, anchor the leaderboard to a fixed reference model whose responses have been judged under both judges, and report all numbers as deltas to that reference.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows judge-human agreement (Cohen's kappa) as a function of the capability gap between the judge and the evaluatees. When the judge is substantially stronger than the models being judged, agreement is high. As the evaluatees approach or exceed the judge in capability, agreement drops because the judge can no longer reliably distinguish correct from confidently-wrong answers. The shaded region around 0.4 is "moderate" agreement — the floor below which the judge stops being useful for that capability tier.
      </Prose>

      <Plot
        label="Judge-human agreement vs capability gap (illustrative)"
        xLabel="judge capability − evaluatee capability (relative)"
        yLabel="Cohen's kappa with humans"
        series={[
          {
            name: "GPT-4 judge",
            color: colors.gold,
            points: [
              [-2.0, 0.20], [-1.5, 0.32], [-1.0, 0.45], [-0.5, 0.58],
              [ 0.0, 0.65], [ 0.5, 0.72], [ 1.0, 0.76], [ 1.5, 0.78], [ 2.0, 0.79],
            ],
          },
          {
            name: "moderate-agreement floor",
            color: colors.textDim,
            points: [[-2.0, 0.40], [2.0, 0.40]],
          },
        ]}
      />

      <Prose>
        The second plot illustrates the position-bias effect before and after swap mitigation. Without mitigation, a single-pass pairwise judge shows a small but systematic preference for whichever response is in position A. After running each comparison twice with positions swapped and keeping only consistent verdicts, the residual position effect is statistically negligible.
      </Prose>

      <Plot
        label="Position bias: A-wins rate per comparison protocol"
        xLabel="judge model"
        yLabel="P(A wins) − 0.5 (deviation from no-bias)"
        series={[
          {
            name: "single pass",
            color: colors.gold,
            points: [[1, 0.08], [2, 0.05], [3, 0.10], [4, 0.06], [5, 0.07]],
          },
          {
            name: "swap mitigated",
            color: "#c084fc",
            points: [[1, 0.01], [2, 0.00], [3, 0.02], [4, 0.01], [5, 0.01]],
          },
          {
            name: "no-bias baseline",
            color: colors.textDim,
            points: [[1, 0], [5, 0]],
          },
        ]}
      />

      <Prose>
        The heatmap below visualizes a confusion matrix between judge verdicts and human verdicts on a 100-comparison calibration set. The diagonal cells show agreement; off-diagonal cells show disagreement. A judge with kappa around 0.65 will look like the matrix below — strong diagonal, small but visible off-diagonal mass.
      </Prose>

      <Heatmap
        matrix={[
          [0.41, 0.05, 0.03],
          [0.04, 0.41, 0.04],
          [0.02, 0.04, 0.16],
        ]}
        rowLabels={["judge: A", "judge: B", "judge: Tie"]}
        colLabels={["human: A", "human: B", "human: Tie"]}
        cellSize={56}
        colorScale="gold"
        label="Judge vs human verdict confusion (n=100)"
      />

      <Prose>
        The step trace below walks through a single pairwise comparison from prompt construction to final verdict, including the position swap and the agreement check that gates the verdict.
      </Prose>

      <StepTrace
        label="LLM-as-judge — one pairwise comparison with position-swap mitigation"
        steps={[
          {
            label: "Construct prompts",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Build round-1 and round-2 prompts</div>
                <div>p1 = template.format(q, A=resp_x, B=resp_y)</div>
                <div>p2 = template.format(q, A=resp_y, B=resp_x)   # swapped</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Same question, same two responses, opposite positions.
                  Position-swap is the cheapest position-bias mitigation.
                </div>
              </div>
            ),
          },
          {
            label: "Round 1 judge call",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Judge call (T=0)</div>
                <div>raw1 = client.messages.create(model, p1, T=0)</div>
                <div>v1   = parse_verdict(raw1)   # 'A' | 'B' | 'Tie'</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  T=0 gives deterministic verdicts. Retry on transient errors.
                  Cache by SHA-256 hash of the full input.
                </div>
              </div>
            ),
          },
          {
            label: "Round 2 judge call",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Judge call with positions swapped</div>
                <div>raw2 = client.messages.create(model, p2, T=0)</div>
                <div>v2   = parse_verdict(raw2)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  v2 refers to the swapped order. Translate back to original
                  X/Y labels before comparing to v1.
                </div>
              </div>
            ),
          },
          {
            label: "Translate and compare",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Resolve position back to X/Y labels</div>
                <div>v1_xy = {"{'A': 'X', 'B': 'Y', 'Tie': 'Tie'}"}[v1]</div>
                <div>v2_xy = {"{'A': 'Y', 'B': 'X', 'Tie': 'Tie'}"}[v2]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  v2 had Y in position A, so a verdict of 'A' there means Y wins.
                  This is the most error-prone step — get the swap mapping wrong
                  and your data is silently corrupted.
                </div>
              </div>
            ),
          },
          {
            label: "Consistency check",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Accept only consistent verdicts</div>
                <div>final = v1_xy if v1_xy == v2_xy else 'Tie'</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  If both rounds picked the same winner: that is the final verdict.
                  If they disagreed (which happens roughly 5–15% of the time on borderline
                  comparisons), call it a tie rather than letting position pick.
                </div>
              </div>
            ),
          },
          {
            label: "Persist verdict",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Cache and log</div>
                <div>cache.put(judge, tpl_v, inputs, final, raw1+raw2)</div>
                <div>log.info(q_hash, x_hash, y_hash, final, raw1_path, raw2_path)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Future runs with the same (judge, template, inputs) hit the cache.
                  Raw responses persist for debugging — leaderboard regressions need
                  to be reproducible months later.
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Pointwise vs pairwise vs reference-based</H3>

      <Prose>
        Use pointwise scoring when you have many candidates, want absolute scores rather than rankings, and can tolerate the calibration noise. Pointwise is O(N) in number of candidates, the resulting scores are interpretable as rough quality estimates ("this response is around an 8/10"), and the format is convenient for monitoring dashboards that want a single number per response. The cost is calibration drift: judges bunch scores in the middle of the scale, are sensitive to surface features, and produce scores whose absolute values are not very meaningful across model versions. G-Eval (Liu et al. 2023, arXiv:2303.16634) is the canonical pointwise-with-criteria framework: the prompt enumerates evaluation criteria explicitly, the judge scores against each, and the final score is a weighted sum.
      </Prose>

      <Prose>
        Use pairwise comparison when you want robust rankings and can afford the quadratic cost (or accept a sampling scheme that compares only a subset of pairs). Pairwise is the default for serious model-vs-model evaluations because it sidesteps calibration entirely and produces well-grounded rankings via Bradley-Terry. The cost is the additional API budget — both for the quadratic structure and for the 2x position-swap multiplier — and the need for an aggregation step. MT-Bench, AlpacaEval 2, Arena-Hard, and Chatbot Arena are all pairwise-first.
      </Prose>

      <Prose>
        Use reference-based grading when you have a high-quality reference response and the task is structured enough that grading-against-reference is well-defined. Math problems with known answers, code generation against unit tests, structured extraction against gold-standard JSON, and translation against expert reference translations are all good fits. The reference converts an open-ended quality question into a constrained correctness question, which judges handle far more reliably. The cost is producing the references in the first place, which is often the bottleneck. AlpacaEval uses GPT-4 reference responses as the comparison baseline; for many specialized tasks, references have to be human-authored.
      </Prose>

      <H3>Choice of judge model</H3>

      <Prose>
        Use the strongest available model whenever your judge-evaluatee capability gap allows it. As of 2026 this typically means GPT-4o, Claude 3.5 Sonnet, Claude Opus, or Gemini 2.5 Pro. The pareto frontier of judge cost vs reliability has shifted substantially since the original MT-Bench work, with Claude 3.5 Sonnet often offering better cost-per-correct-judgment than GPT-4 at lower price. Use a judge from a different model family than any of the evaluatees to control for self-enhancement bias — if you are evaluating GPT-class models, use Claude as the judge, and vice versa. For high-stakes decisions, ensemble two or three judges from different families and report consensus rates alongside per-judge results.
      </Prose>

      <Prose>
        Cheaper judges (GPT-4o-mini, Claude 3 Haiku, Gemini Flash) are appropriate for first-pass triage and high-volume monitoring. The pattern is: run the cheap judge on all comparisons, then run the strong judge only on the close calls (where the cheap judge gave a low-confidence verdict or where the comparison sits near a decision boundary on the leaderboard). This cascade typically captures 90% of the ranking accuracy at 20–30% of the cost.
      </Prose>

      <H3>Single judge vs jury</H3>

      <Prose>
        A jury of judges (Verga et al. 2024, "Replacing Judges with Juries") replaces a single strong judge with an ensemble of weaker but cheaper judges, taking majority vote or averaged scores. The empirical claim is that a panel of 3–5 small models can match a single large model's accuracy at lower cost, with the additional benefit of cross-family bias control built in. Use juries when (1) you need to control for judge bias by ensembling, (2) you have budget headroom for parallel judge calls but not for the strongest judge, or (3) you want confidence flags from inter-judge disagreement on individual comparisons.
      </Prose>

      <H3>Online vs offline evaluation</H3>

      <Prose>
        Offline evaluation runs against a fixed prompt set and produces a snapshot leaderboard. This is what MT-Bench and AlpacaEval do, and what most internal eval pipelines look like. The strength is reproducibility — you can rerun the same prompts six months later and get directly comparable results. The weakness is staleness — a fixed prompt set gradually leaks into training data and stops measuring genuinely-out-of-sample performance. Online evaluation, like Chatbot Arena, collects fresh comparisons continuously from real users in the wild. The strength is freshness and direct measurement of deployment-relevant capability; the weakness is non-reproducibility — the prompt distribution shifts over time, so two snapshots are not directly comparable. Most production teams run both: offline for fast iteration on internal benchmarks, periodic online for ground-truth calibration of the offline numbers.
      </Prose>

      <H3>When to fall back to humans</H3>

      <Prose>
        Three scenarios clearly require human evaluation despite the operational cost. First, when the task is in a domain where the judge's competence is borderline — advanced mathematics, niche legal/medical reasoning, languages outside the judge's training — judge agreement with humans drops below useful thresholds and the evaluation becomes noise. Second, when the evaluation result will drive a high-stakes decision (model release, large training run kickoff, safety claim) and the cost of a wrong measurement substantially exceeds the cost of human evaluation. Third, when the evaluation is intended to calibrate the judge itself — you cannot bootstrap judge reliability from judge results, you need an external ground-truth signal, which today still means humans.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        The compute and cost profile of LLM-as-judge scales gracefully across most dimensions that matter operationally. Number of evaluatees scales linearly for pointwise and quadratically for full pairwise, with sampling schemes (bucket sampling, random pair sampling, active comparison selection) reducing the pairwise cost to near-linear in practice. Number of prompts scales linearly. Concurrency scales as far as your provider rate limits allow, which for production accounts at major providers is in the hundreds of concurrent requests. Caching means iterative experiments where most inputs are unchanged from the previous run are nearly free. The economic ceiling for a serious eval pipeline at a small frontier-model team is in the low hundreds of dollars per day, which is negligible relative to training costs.
      </Prose>

      <Prose>
        What does not scale is the underlying assumption of judge reliability. Three regimes break the assumption. First, when the evaluatee capability approaches or exceeds the judge, the judge cannot reliably distinguish correct from confidently-wrong outputs, and the resulting rankings degrade smoothly toward chance. This is why the field has been progressively upgrading judges (GPT-4 → GPT-4-turbo → GPT-4o → Claude 3.5 Sonnet) as the evaluatee population improves; the judge has to stay ahead of what it is judging. Second, when the task is one where the judge has known weaknesses (advanced math, formal logic, specialized domain knowledge), even a strong judge produces unreliable verdicts. Third, when the responses being compared are very close in quality, the judge's noise floor exceeds the actual quality difference and the ranking becomes a coin flip. The mitigation is to require larger sample sizes for close comparisons; Chatbot Arena's bootstrap confidence intervals encode this directly.
      </Prose>

      <Prose>
        Benchmark contamination is the slow-moving threat to scale. Once a prompt set is published and widely used, it inevitably leaks into training data — either through web scraping, deliberate inclusion in supervised fine-tuning sets, or implicit inclusion via published model evaluations that include the prompts in their writeups. MT-Bench's 80 questions, AlpacaEval's prompt set, and other widely-used eval sets have all measurably degraded as discriminators since their original publication, with newer models scoring near-ceiling regardless of underlying capability. The countermeasures — Arena-Hard's harder curated questions, MT-Bench-101's expanded multi-turn coverage, and the rolling refresh of Chatbot Arena's natural-distribution prompts — buy time but do not solve the underlying problem. A reasonable mental model is that any published benchmark has a half-life of 12–24 months as a discriminator before contamination dominates.
      </Prose>

      <Prose>
        The other thing that does not scale is the calibration period after a judge upgrade. When you switch judge models — for cost reasons, capability reasons, or because the previous judge has been deprecated — your historical leaderboard becomes incomparable to your new numbers. Re-running the entire historical eval set under the new judge is sometimes feasible (with caching the cost is bounded), but for organizations with years of historical data the practical solution is to anchor everything to a fixed reference model and report deltas, accepting that absolute-Elo comparability across judge generations is lost.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Verbosity bias inflates long responses</H3>
      <Prose>
        Judges consistently prefer longer responses, even after instruction to ignore length. The effect size in raw pairwise comparisons is on the order of a 5–15% win-rate boost per doubling of length, which is enough to substantially distort rankings between models that produce different average response lengths. The fix is some combination of explicit length-ignore instructions in the template (cuts the bias by ~30%), length-controlled win rates (AlpacaEval 2 LC, eliminates the linear length effect statistically), and reference-based grading where the reference is matched in length to typical candidates. None of these is a complete fix; assume residual length bias and design your downstream interpretation accordingly.
      </Prose>

      <H3>Position bias picks A or B by structural preference</H3>
      <Prose>
        Without mitigation, judges show 5–15% preference for whichever response is presented in position A (or sometimes B, depending on the judge). This is structural — the judge's prompt-processing and attention patterns produce a small but consistent bias. The standard fix is position-swap (run each comparison twice in opposite orders, accept only consistent verdicts) which reduces the residual bias to roughly 1–2%. Skipping this mitigation is the single most common bug in homegrown judge harnesses, and it shows up as systematic ranking errors in the resulting leaderboard.
      </Prose>

      <H3>Self-enhancement bias favors the judge's own family</H3>
      <Prose>
        Models from the same family as the judge get a measurable boost in pairwise comparisons. GPT-4 judging GPT-3.5 against an open-source 7B model will rate GPT-3.5 higher than a more neutral judge would. The mechanism is a combination of stylistic preference (the judge has internalized its own family's response style as "good") and shared bias (both judge and evaluatee inherit the same training-data preferences). The mitigation is to cross-judge: use a judge from a different family than any of the evaluatees, or ensemble multiple judges from different families and report whether they agree. Single-family judging across a single-family leaderboard produces results that look clean but are quietly biased.
      </Prose>

      <H3>Judge competence ceiling on the task</H3>
      <Prose>
        A judge cannot reliably evaluate responses on tasks the judge itself struggles with. If GPT-4 only gets 60% of advanced math problems right, asking GPT-4 to judge whether a math response is correct produces verdicts that are themselves only 60-ish% accurate. The signal degrades smoothly as judge competence on the task drops. The mitigation is reference-based grading where the reference is human-authored and known correct; this converts the judge's role from "is this answer good" to "does this answer match the reference", which is structurally easier. For tasks below the judge's competence ceiling, automated judging is not a viable replacement for human (or executable, for code/math) verification.
      </Prose>

      <H3>Reasoning-trace artifacts</H3>
      <Prose>
        When the judge writes its reasoning before giving its verdict (chain-of-thought style), the reasoning often anchors the verdict in subtle ways. Judges that write a lengthy critique of response A and then are asked to choose A or B often vote against A, even when their critique was mild and their concrete points favored A. The pattern is "I just spent a paragraph criticizing A, therefore A must be worse." The mitigation is either to require the verdict before the reasoning (which loses the calibration value of reasoning) or to use structured output where the reasoning and verdict fields are both filled but the model is constrained to be internally consistent. There is no fully clean fix.
      </Prose>

      <H3>Format-sensitivity and gaming</H3>
      <Prose>
        Judges have learned strong preferences for certain surface formats — bulleted lists, headers, emojis used sparingly, citations in a specific style. A response that uses these formats well will be rated higher independent of content quality. This becomes a problem in two ways. First, in any benchmark where one model has been more heavily trained on judge-friendly formats, the leaderboard reflects format alignment as much as quality. Second, in iterative training that uses LLM-as-judge as a reward signal (LLM-as-judge in RLAIF), the policy can game the format preferences directly, learning to produce judge-pleasing surface features without underlying quality improvement. The mitigation is to disclose the format-sensitivity in evaluation reports and to use format-controlled comparisons when needed.
      </Prose>

      <H3>Benchmark contamination over time</H3>
      <Prose>
        Once a prompt set is published, it leaks into training data. MT-Bench scores at the top of the leaderboard have compressed substantially since 2023 because top-tier models have effectively seen those prompts and the desired response style during training. The same dynamic affects every published eval set. The countermeasures — periodic refresh of the prompts (Arena-Hard, MT-Bench-Plus), dynamic prompt sampling from real user distributions (Chatbot Arena), and held-out evaluation sets that are never published in full (internal frontier-lab evals) — buy time but the underlying problem is structural to the open-publication model. Treat any benchmark older than 12 months with skepticism unless you have evidence it remains a discriminator.
      </Prose>

      <H3>Inconsistent verdicts on borderline comparisons</H3>
      <Prose>
        On comparisons where the two responses are close in quality, even a deterministic (T=0) judge can produce inconsistent verdicts across position swaps or template variations. This is not a bug; it correctly reflects the underlying ambiguity. The right interpretation is that those comparisons sit near the judge's noise floor and should be coded as ties or weighted lower in aggregation. Forcing a verdict on every comparison and treating ties as missing data, rather than as informative signal, leads to noisy and unstable rankings.
      </Prose>

      <H3>Implicit assumption of stable judge over time</H3>
      <Prose>
        Provider-side model updates can change judge behavior silently. A model labeled "claude-3-5-sonnet" or "gpt-4o" without a specific version date may receive backend updates that shift its judgment patterns by a few percentage points across categories. Always pin to a specific version date (e.g., <Code>claude-3-5-sonnet-20241022</Code>, <Code>gpt-4o-2024-08-06</Code>) and treat any provider deprecation of that version as a calibration event requiring a re-baselining run.
      </Prose>

      <H3>Ties as a structural escape hatch</H3>
      <Prose>
        Judges often use the "Tie" verdict to escape close calls, even when one response is meaningfully better. If your template includes Tie as an option, expect 15–30% of verdicts to be ties, and most of those will reflect ambiguity rather than genuine equivalence. Two patterns help. First, design the template to make Tie hard to reach — explicit instruction that Tie should only be used when responses are "essentially indistinguishable", with the implication that genuine differences should not be hidden behind a tie. Second, in aggregation, treat ties as half-wins for both sides (the standard Elo/Bradley-Terry treatment) rather than as missing data, which preserves the information that the comparison was held but did not reveal a clear winner.
      </Prose>

      <Callout accent="gold">
        LLM-as-judge inherits all of the judge's biases. The paradigm only works because frontier-class judges have biases small enough relative to genuine quality differences that the signal exceeds the noise. As models converge in capability, that signal-to-noise ratio worsens, and the protocol details (swap mitigation, length controls, cross-family judging, reference grading) become increasingly load-bearing.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their arXiv pages on 2026-04-26. Abstracts, author lists, and arXiv IDs confirmed.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench and Chatbot Arena</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 2023; NeurIPS 2023 Datasets and Benchmarks. The founding paper for the LLM-as-judge methodology. Introduces MT-Bench (80 multi-turn questions across writing, reasoning, math, extraction, and coding), Chatbot Arena (crowdsourced pairwise battles with Elo aggregation), and the empirical agreement analysis that established GPT-4-as-judge as a viable replacement for human evaluation at ~80% agreement rate. Documents position bias, verbosity bias, and self-enhancement bias as primary failure modes and proposes mitigation protocols.
      </Prose>

      <H3>Liu et al. 2023 — G-Eval</H3>
      <Prose>
        Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." arXiv:2303.16634. Published March 2023; EMNLP 2023. Predates Zheng et al. in the LLM-as-judge timeline; introduces the G-Eval framework where the judge prompt explicitly enumerates evaluation criteria, the judge produces per-criterion subscores with chain-of-thought reasoning, and the final score is a weighted aggregate. Demonstrates substantial improvements in correlation with human judgment compared to BLEU/ROUGE on summarization and dialogue benchmarks. Notable for being one of the first papers to formalize the prompt template as the central design artifact in LLM-as-judge work.
      </Prose>

      <H3>Chiang and Lee 2023 — Can LLMs Be an Alternative to Human Evaluations?</H3>
      <Prose>
        Cheng-Han Chiang, Hung-yi Lee. "Can Large Language Models Be an Alternative to Human Evaluations?" arXiv:2305.01937. Published May 2023; ACL 2023. The contemporaneous study that formally tested whether LLM-judged scores correlate with human-judged scores across NLG benchmarks. Finds strong correlation but documents systematic differences: LLM judges are more lenient on minor errors, more sensitive to surface fluency, and more variable across reruns at non-zero temperature. Recommends specific protocol adjustments (multiple runs averaged, explicit criteria enumeration, reference comparison) that prefigure later best practice.
      </Prose>

      <H3>Dubois et al. 2024 — AlpacaEval and Length-Controlled Win Rate</H3>
      <Prose>
        Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." arXiv:2404.04475. Published April 2024. Introduces AlpacaEval 2 LC, the length-controlled win rate methodology that statistically corrects for the verbosity bias in pairwise judging via a logistic regression that decomposes win rate into length-explained and length-independent components. Reports that LC win rates have substantially higher rank correlation with Chatbot Arena Elo than uncorrected win rates, especially for verbose models. The companion AlpacaEval framework (Dubois et al. 2023, "AlpacaFarm") established the benchmark itself.
      </Prose>

      <H3>Wang et al. 2023 — Large Language Models are not Fair Evaluators</H3>
      <Prose>
        Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, Zhifang Sui. "Large Language Models are not Fair Evaluators." arXiv:2305.17926. Published May 2023. The position-bias paper. Quantifies position bias systematically across GPT-4, GPT-3.5, and Claude as judges, showing 5–25% raw position bias depending on model and task, and documents the position-swap protocol as the standard mitigation. Required reading for anyone building a judge harness — explains why the swap-and-consistency-check pattern in section 4c is non-optional.
      </Prose>

      <H3>Verga et al. 2024 — Replacing Judges with Juries</H3>
      <Prose>
        Pat Verga, Sebastian Hofstätter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, Patrick Lewis. "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models." arXiv:2404.18796. Published April 2024. Introduces the panel-of-judges (jury) approach where multiple smaller models substitute for a single large judge, taking majority vote or averaged scores. Demonstrates that a panel of 3–5 small open-weight models can match GPT-4-as-judge in human-correlation at substantially lower cost, with the additional benefit of bias diversification across model families. Useful when judge cost matters or when self-enhancement bias must be controlled at the panel level.
      </Prose>

      <H3>Li et al. 2024 — Arena-Hard</H3>
      <Prose>
        Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica. "From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline." arXiv:2406.11939. Published June 2024. Introduces Arena-Hard, a 500-prompt benchmark distilled from Chatbot Arena conversations to be harder and more discriminating than MT-Bench, and BenchBuilder, the pipeline that produces it. Reports substantially higher rank correlation with Chatbot Arena leaderboard Elo than MT-Bench, especially in the high-capability regime where MT-Bench saturates. The current best-practice offline benchmark for chat model comparison.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why pairwise beats pointwise on calibration</H3>
      <Prose>
        Suppose a judge has a systematic positive bias of +1 on its 1-10 scale (it gives every response a score 1 point higher than it would have without the bias) but its rank ordering of responses is unaffected. Sketch out: (a) what happens to a pointwise leaderboard built from this judge, (b) what happens to a pairwise leaderboard built from the same judge, and (c) what happens if you fit a Bradley-Terry model on the pairwise data versus computing mean scores from the pointwise data. Which of these protocols recovers the correct ranking, and what does that tell you about why pairwise is the default for serious model-vs-model evaluation?
      </Prose>

      <H3>Exercise 2 — Cohen's kappa under skewed labels</H3>
      <Prose>
        You run a pairwise comparison set of 1000 prompts comparing a 70B-parameter model against a 7B-parameter model. The judge picks the 70B model in 920 of 1000 cases, and the human annotators agree with the judge in 880 of those 920 cases and disagree (picking the 7B model) in 30 of the 80 cases where the judge picked the 7B. Compute the raw observed agreement, the chance agreement, and Cohen's kappa. Comment on whether the kappa is a fair representation of the judge's reliability, and what the standard mitigation (balancing the comparison set) would do to the kappa. This exercise illustrates the kappa paradox explicitly.
      </Prose>

      <H3>Exercise 3 — Length-controlled win rate by hand</H3>
      <Prose>
        You collected pairwise judgments comparing a candidate model against a reference. The candidate's average response length is 800 tokens and the reference's is 400 tokens; the candidate wins 60% of pairwise comparisons. Suppose a logistic regression of win-vs-loss on the length difference (candidate length minus reference length) yields coefficients <Code>{"\\beta_0 = 0.2"}</Code> and <Code>{"\\beta_L = 0.001"}</Code> per token. Compute the length-controlled win rate (the win rate at zero length difference) and explain what this tells you about how much of the candidate's apparent advantage was attributable to length alone. Discuss whether you would publish the raw win rate, the LC win rate, or both, and why.
      </Prose>

      <H3>Exercise 4 — Ablation for position bias</H3>
      <Prose>
        Design an ablation experiment that quantifies position bias in your judge harness. Specifically: what is the experimental setup, how many comparisons do you need to detect a 5% position bias at 95% confidence, what statistical test do you use, and what is the success criterion? Once the bias is measured, design a second experiment that quantifies how much of the bias the position-swap mitigation removes. Walk through what the data would look like in three regimes: (a) no position bias and effective mitigation, (b) substantial position bias and effective mitigation, (c) substantial position bias and broken mitigation (e.g., a swap-mapping bug).
      </Prose>

      <H3>Exercise 5 — Detecting judge-evaluatee capability collapse</H3>
      <Prose>
        You have been using GPT-4o as a judge for evaluating model checkpoints on a coding benchmark. The latest checkpoints are scoring near-ceiling on the benchmark and you suspect the judge is no longer reliably distinguishing the top models. List three observable signals — from training metrics, judge outputs, or downstream evaluations — that would confirm or deny your suspicion. For each signal, describe what it looks like in the healthy regime versus the collapsed regime. As a follow-up: assuming the collapse is real, what intervention would you take, and how would you validate that the intervention restored useful discrimination? Walk through the specific protocol changes (judge upgrade, reference-based grading, harder prompts, executable verification) and the trade-offs of each.
      </Prose>

      <H3>Exercise 6 — Bradley-Terry sensitivity to comparison sparsity</H3>
      <Prose>
        Bradley-Terry MLE assumes you have enough comparisons per model pair to estimate the relevant probabilities. Suppose you are ranking 50 models and budget allows only 1000 total pairwise comparisons across all model pairs. Outline a sampling scheme that allocates comparisons across pairs to maximize the precision of the resulting ranking. Consider: should you use random pair sampling, round-robin, active sampling targeting close-in-rank pairs, or some hybrid? How would you quantify the precision of the resulting ranking, and how does the answer change if your goal is the top-5 ranking rather than the full 50-model ranking? Connect your answer to the bandit literature on best-arm identification.
      </Prose>

      <H3>Exercise 7 — Temperature, structured output, and reproducibility</H3>
      <Prose>
        You discover that your evaluation harness produces slightly different leaderboard numbers on consecutive runs over the same data. The judge is running at temperature 0 and the cache is disabled. Enumerate every possible source of non-determinism in a modern LLM-as-judge call (provider-side, network-level, parsing-level, aggregation-level), and propose a protocol that pins each one. Once everything is pinned, what is the residual variance you would expect across runs, and how does that compare to the typical statistical noise in a 1000-comparison eval? When does this residual matter, and when can it be ignored?
      </Prose>

    </div>
  ),
};

export default llmAsJudgeParadigm;
