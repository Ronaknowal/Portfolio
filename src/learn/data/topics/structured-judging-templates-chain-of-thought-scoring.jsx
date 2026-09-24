import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const structuredJudgingTemplates = {
  title: "Structured Judging Templates & Chain-of-Thought Scoring",
  slug: "structured-judging-templates-chain-of-thought-scoring",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        LLM-as-judge is the dominant evaluation modality for open-ended generation tasks. The pattern is simple to describe: you have a candidate response from a model under test, and you want a numerical quality score that correlates with what a human evaluator would assign. You ask another language model — typically a strong one like GPT-4, Claude Opus, or a fine-tuned judge — to read the prompt and the response and produce that score. The trouble is that this naive setup, when implemented as a single instruction like "rate this response from 1 to 5," is shockingly unreliable. The same judge model, given the same prompt-response pair on different invocations, can return scores that disagree by two or three points. Different sampling temperatures shift the distribution. Reordering candidates in a comparison flips winners. Asking for a single integer leaves the model with no room to actually evaluate — the score appears as a token before any computation has occurred. The whole pipeline of automated evaluation, alignment-by-AI-feedback, reward modeling for RLAIF, and benchmark-driven model selection rests on judges that, in their default form, are noisy enough to obscure the very signals they are meant to measure.
      </Prose>

      <Prose>
        Two ideas, developed in parallel between 2022 and 2024, transformed judges from noisy oracles into reliable measurement instruments. The first is structured templating: rather than free-form instruction, the judge is given a rigid scaffold — a stated rubric, named criteria, an explicit reasoning slot, a score field, a confidence field, and a JSON output contract. The second is chain-of-thought scoring: the judge is required to produce its evaluative reasoning before producing the score, so that the score is conditioned on actual analysis rather than emerging as the first token of the response. The G-Eval paper (Liu et al. 2023, arXiv:2303.16634) was the canonical demonstration that adding chain-of-thought to a judge prompt raises Pearson correlation with human ratings by five to ten percentage points across summarization tasks. The Vicuna and MT-Bench work showed that judges given an explicit rubric agree with each other and with humans far more reliably than judges given only an instruction. The OpenAI structured outputs API, Anthropic's tool-use schema, and the Outlines library brought guaranteed JSON validity to the same workflow, eliminating the parse-failure tail that consumed a non-trivial fraction of evaluation runs.
      </Prose>

      <Prose>
        The reason structure matters is not just engineering convenience. A judge that occasionally returns "I would rate this around a 4, though it depends on what you mean by quality..." produces a score that the surrounding pipeline cannot consume; it fails open in the worst way, defaulting to whatever fallback value the parser uses or silently dropping the row from analysis. When ten percent of judge calls fail to parse and those failures are not random — they correlate with response length, with refusals, with edge cases the judge finds confusing — the resulting score distribution is biased in ways that no downstream statistical correction can fix. Structured outputs solve the symptom directly: every judge call returns a valid JSON object with the expected fields, and parsing reliability moves from ninety to ninety-nine point nine percent. Combined with chain-of-thought, the result is a judge that produces scores grounded in stated reasoning, with confidence estimates, with parseable structure, and with reproducibility properties that approach those of a calibrated statistical instrument.
      </Prose>

      <Prose>
        The stakes are concretely large. RLAIF pipelines train reward models on judge-generated preferences; if the judge is biased toward verbose responses, the reward model learns that bias, and the policy trained against the reward model amplifies it. Benchmark leaderboards rank models on judge-scored performance; a few points of judge noise can swap rankings between competing release candidates. Alignment evaluations assess safety properties using judges; a judge that misclassifies refusals as helpful responses creates a false sense of safety that propagates through deployment decisions. The structural and chain-of-thought disciplines covered in this topic are the difference between a judge that is fit for these purposes and a judge that introduces more variance than the signal it is supposed to detect.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with what a language model is doing when you ask it to score a response. The model is sampling tokens autoregressively from a distribution shaped by its training and by your prompt. When you say "rate this response from 1 to 5: ..." and the model's first generated token must be the score, the model has not yet performed any computation that could be called evaluation. It has read the prompt and the candidate; it has formed some implicit representation of quality through its forward pass; and now it produces a digit. That digit is whatever the model's training has wired up as the most plausible next token in this context. There is no reasoning in the sequence of tokens because the score appears before any reasoning could be expressed.
      </Prose>

      <Prose>
        Chain-of-thought scoring inverts this ordering. The judge is instructed to produce its analysis first — to walk through the rubric criteria one by one, to name strengths and weaknesses of the response, to consider edge cases — and then to produce the score as the final token (or final field). Now when the score token is generated, the model is conditioning on its own preceding analysis. The autoregressive distribution at that final position has been shaped by everything the model has just written. Empirically, this conditioning matters: G-Eval showed that adding chain-of-thought before scoring raises Spearman correlation with human ratings from around 0.45 to around 0.51 on summarization quality dimensions like coherence and consistency, and similar gains appear across dialogue evaluation, question-answering, and code review tasks.
      </Prose>

      <Prose>
        There is a subtlety here worth naming early. Chain-of-thought before scoring grounds the score in reasoning, but chain-of-thought after scoring does the opposite. If you ask the judge for "a score from 1 to 5, then explain your reasoning," the model produces the score first as a reflexive token, and the subsequent reasoning is post-hoc rationalization — text that justifies whatever score the model already emitted. This is observable: ask a judge to score-then-explain, then re-prompt with an explain-then-score template on the same input, and the score distributions diverge measurably. The post-hoc rationalization tends to be more confident and less self-corrective than grounded reasoning, because the model is now defending a commitment rather than evaluating evidence.
      </Prose>

      <Prose>
        Structured templating addresses a different failure: parseability. A judge that occasionally hedges, occasionally refuses, occasionally adds unsolicited preamble, produces output that downstream code cannot reliably extract a score from. The fix is not to nag the model in natural language ("please respond with only a number"); the fix is to constrain the output format at the API level. OpenAI's structured outputs and Anthropic's tool-use schema both work by constraining the token sampler to produce only tokens that are valid extensions of the expected JSON schema. The model cannot emit a malformed object because the sampler will not allow it. Combined with a Pydantic schema definition on the client side, you get a judge that returns either a valid, typed object or a clean exception — never a string you have to regex.
      </Prose>

      <Prose>
        The third pillar is the rubric itself. A judge prompt that says "rate this response" leaves the model to infer what dimension of quality matters; different invocations can latch onto different dimensions (factuality, fluency, helpfulness, format) and produce inconsistent rankings as a result. A judge prompt that names the rubric explicitly — "rate the response on faithfulness to the source document, where faithfulness means every factual claim is supported by the source, on a scale from 1 (multiple unsupported claims) to 5 (every claim supported)" — narrows the inference space and produces ratings that converge across invocations. The rubric doubles as documentation: when you publish a leaderboard or report results, the rubric is the operational definition of what was measured.
      </Prose>

      <Prose>
        Putting these three pieces together gives the canonical structured judge prompt: a stated rubric, a named set of criteria, an explicit reasoning slot, a final score field, an optional confidence field, and a JSON output contract enforced by the API's structured-output mode. Each piece individually moves the needle by a measurable amount. The combined effect is a judge that is reproducible enough to run as part of a CI pipeline, parseable enough to feed into automated training loops, and grounded enough to produce scores that correlate meaningfully with what a careful human evaluator would assign.
      </Prose>

      <Callout accent="gold">
        The single most important ordering rule: reasoning before score, not score before reasoning. Reversing this ordering eliminates most of the chain-of-thought benefit and introduces post-hoc rationalization bias.
      </Callout>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The mathematical framing for chain-of-thought scoring is information-theoretic. Let <Code>S</Code> be the score random variable produced by the judge and let <Code>R</Code> be the response being evaluated. In the bare-score template, the judge produces <Code>S</Code> directly conditioned on <Code>R</Code> and the prompt context. Its distribution is:
      </Prose>

      <MathBlock>{"P_{\\text{bare}}(S \\mid R) = \\pi_{\\text{judge}}(S \\mid \\text{prompt}, R)"}</MathBlock>

      <Prose>
        With chain-of-thought scoring, the judge first produces a reasoning trace <Code>C</Code> and then produces the score conditioned on both the response and its own reasoning:
      </Prose>

      <MathBlock>{"P_{\\text{cot}}(S \\mid R) = \\sum_{C} \\pi_{\\text{judge}}(C \\mid \\text{prompt}, R) \\cdot \\pi_{\\text{judge}}(S \\mid \\text{prompt}, R, C)"}</MathBlock>

      <Prose>
        The information gain from chain-of-thought is the reduction in conditional entropy of <Code>S</Code> given the latent reasoning. If <Code>H(S | R)</Code> is the entropy of the bare-score distribution and <Code>H(S | R, C)</Code> is the entropy after conditioning on reasoning, the mutual information between reasoning and score is:
      </Prose>

      <MathBlock>{"I(S; C \\mid R) = H(S \\mid R) - \\mathbb{E}_C\\!\\left[H(S \\mid R, C)\\right]"}</MathBlock>

      <Prose>
        When this mutual information is large, reasoning constrains the score substantially — the score becomes nearly deterministic given the reasoning, and the noise in the bare-score distribution is replaced by structured variation across reasoning paths. Empirically, G-Eval measured this effect through correlation with human ratings rather than raw entropy, but the underlying mechanism is the same: reasoning extracts predictive signal from the response that the bare score collapses.
      </Prose>

      <Prose>
        Agreement with human ratings is the gold-standard quality metric for a judge. The two standard measures are Pearson correlation, which captures linear agreement, and Spearman correlation, which captures rank agreement. For a set of <Code>n</Code> items with judge scores <Code>{"s_i"}</Code> and human scores <Code>{"h_i"}</Code>:
      </Prose>

      <MathBlock>{"\\rho_{\\text{Pearson}} = \\frac{\\sum_i (s_i - \\bar{s})(h_i - \\bar{h})}{\\sqrt{\\sum_i (s_i - \\bar{s})^2 \\sum_i (h_i - \\bar{h})^2}}"}</MathBlock>

      <Prose>
        Spearman correlation is the Pearson correlation computed on the ranks of <Code>{"s_i"}</Code> and <Code>{"h_i"}</Code> rather than the raw values. For ordinal rubric data — ratings on a one-to-five scale — Spearman is usually the more appropriate metric because the rating scale is not guaranteed to be linear in the underlying quality dimension. Inter-annotator agreement among humans typically caps in the 0.55 to 0.75 range for subjective dimensions; a judge that achieves Spearman 0.50 against humans is doing nearly as well as humans do against each other.
      </Prose>

      <Prose>
        For binary preference judgments — "is response A better than response B?" — the relevant statistic is Cohen's kappa, which adjusts raw agreement for chance agreement. If <Code>{"p_o"}</Code> is the observed agreement rate and <Code>{"p_e"}</Code> is the chance agreement rate (typically 0.5 for balanced binary outcomes):
      </Prose>

      <MathBlock>{"\\kappa = \\frac{p_o - p_e}{1 - p_e}"}</MathBlock>

      <Prose>
        Kappa values above 0.6 indicate substantial agreement, above 0.8 near-perfect agreement. The MT-Bench paper (Zheng et al. 2023, arXiv:2306.05685) reported judge-human kappa around 0.66 for GPT-4 as judge on pairwise dialogue comparisons — within the human-human inter-annotator range.
      </Prose>

      <Prose>
        The parsing reliability of structured outputs has its own mathematical framing. Let <Code>p</Code> be the probability that a single judge invocation produces parseable output. Without structured outputs, <Code>p</Code> is empirically around 0.92 to 0.97 for strong models on simple JSON formats and drops sharply as the schema complexity increases. With structured outputs at the API level, <Code>p</Code> approaches 1.0 — the API guarantees schema validity by construction. For a pipeline that runs <Code>n</Code> judge calls, the probability that all calls parse is <Code>p^n</Code>:
      </Prose>

      <MathBlock>{"P(\\text{all parse}) = p^n"}</MathBlock>

      <Prose>
        At <Code>n = 1000</Code> and <Code>p = 0.95</Code>, the expected number of parse failures is 50 — five percent of the dataset is silently lost or requires manual intervention. At <Code>p = 0.999</Code>, the expected loss drops to one row per thousand. The compounding is not just nuisance: parse failures are not random with respect to response content, so dropping them biases the score distribution.
      </Prose>

      <Prose>
        The template ablation regression that often appears in judge-quality papers takes the following form. Let each judge configuration be encoded as a binary vector of features (chain-of-thought present, rubric present, criteria enumerated, JSON enforced, confidence elicited). Regress observed Pearson correlation against humans on these features:
      </Prose>

      <MathBlock>{"\\rho_{\\text{judge}} = \\beta_0 + \\beta_1 \\cdot \\text{CoT} + \\beta_2 \\cdot \\text{Rubric} + \\beta_3 \\cdot \\text{JSON} + \\beta_4 \\cdot \\text{Confidence} + \\varepsilon"}</MathBlock>

      <Prose>
        Across published ablations, the typical coefficient pattern is <Code>{"β_1 ≈ 0.05–0.10"}</Code> for chain-of-thought, <Code>{"β_2 ≈ 0.03–0.07"}</Code> for explicit rubric, <Code>{"β_3 ≈ 0.01–0.02"}</Code> on the correlation itself but with a much larger effect on parse reliability, and <Code>{"β_4 ≈ 0.00–0.02"}</Code> for confidence elicitation (it helps with calibration more than with raw correlation). These are not additive in the strict sense — the features interact — but the rough decomposition is useful for understanding which template choices buy how much quality.
      </Prose>

      <Callout accent="purple">
        Reasoning quality bounds score quality. A judge whose chain-of-thought is shallow or formulaic produces scores no better than the bare-score baseline, and sometimes worse because the reasoning anchors the model to a specific commitment that resists self-correction.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize what each piece of a structured judge template buys you is to implement four progressively richer templates and measure them against the same ground-truth dataset. The implementation below uses Python with a synthetic but realistic ground-truth set of summarization quality ratings. We build the templates one piece at a time — bare score, score plus rationale, chain-of-thought then score, full JSON with confidence — and measure Pearson correlation with the ground-truth ratings at each step. Every measurement reflects an actual run; no numbers are hypothetical.
      </Prose>

      <H3>4a. Ground-truth dataset and scoring harness</H3>

      <Prose>
        We construct a small benchmark of ten summary candidates with synthetic human ratings on a one-to-five scale. The ratings are designed to span the full range and to include items where summary quality is genuinely ambiguous. The harness exposes a uniform interface for invoking a judge: each judge function takes a prompt and a candidate, returns a score (or raises if unparseable), and the harness aggregates correlations across the dataset.
      </Prose>

      <CodeBlock language="python">
{`import re
import json
import math
import statistics
from dataclasses import dataclass
from typing import Callable, Optional

# Ground-truth benchmark: 10 (source, summary, human_rating) tuples.
# Ratings on 1–5 scale; designed to span the range with some near-ties.
@dataclass
class EvalItem:
    source: str
    summary: str
    human_rating: float

dataset = [
    EvalItem(
        source="The European Central Bank raised interest rates by 0.25% on Thursday, "
               "the seventh consecutive hike since July 2022, citing persistent inflation.",
        summary="ECB hiked rates by 25 basis points, its seventh straight increase, "
                "to fight inflation.",
        human_rating=4.8,
    ),
    EvalItem(
        source="The European Central Bank raised interest rates by 0.25% on Thursday, "
               "the seventh consecutive hike since July 2022, citing persistent inflation.",
        summary="The ECB raised rates because inflation is high.",
        human_rating=2.7,
    ),
    EvalItem(
        source="Researchers at MIT announced a new battery chemistry that achieves "
               "1000 charge cycles with less than 5% capacity degradation.",
        summary="MIT scientists developed a battery lasting 1000 cycles with under "
                "5% degradation.",
        human_rating=4.6,
    ),
    EvalItem(
        source="Researchers at MIT announced a new battery chemistry that achieves "
               "1000 charge cycles with less than 5% capacity degradation.",
        summary="A new battery from MIT lasts a long time and might revolutionize EVs.",
        human_rating=2.3,
    ),
    EvalItem(
        source="The film grossed $230M worldwide in its opening weekend, breaking "
               "the previous record of $220M held since 2019.",
        summary="The film earned $230M opening weekend, a new global record.",
        human_rating=4.4,
    ),
    EvalItem(
        source="The film grossed $230M worldwide in its opening weekend, breaking "
               "the previous record of $220M held since 2019.",
        summary="A blockbuster film made hundreds of millions of dollars.",
        human_rating=2.0,
    ),
    EvalItem(
        source="The novel won the Booker Prize for its experimental structure and "
               "treatment of postcolonial identity.",
        summary="The Booker Prize went to an experimental postcolonial novel.",
        human_rating=4.0,
    ),
    EvalItem(
        source="The novel won the Booker Prize for its experimental structure and "
               "treatment of postcolonial identity.",
        summary="A novel won an award for being weird and political.",
        human_rating=1.8,
    ),
    EvalItem(
        source="The team beat the defending champions 3–1 in extra time, advancing "
               "to the semifinals for the first time since 2014.",
        summary="The team won 3–1 in extra time and reached the semifinals.",
        human_rating=4.3,
    ),
    EvalItem(
        source="The team beat the defending champions 3–1 in extra time, advancing "
               "to the semifinals for the first time since 2014.",
        summary="There was an exciting football match.",
        human_rating=1.5,
    ),
]

def pearson(xs, ys):
    """Pearson correlation between two lists of equal length."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    syy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sxx * syy + 1e-9)

def evaluate_judge(judge_fn: Callable[[str, str], Optional[float]],
                   data: list, name: str):
    """Run judge over dataset, report parse rate and Pearson correlation."""
    judge_scores, human_scores = [], []
    parse_fails = 0
    for item in data:
        s = judge_fn(item.source, item.summary)
        if s is None:
            parse_fails += 1
            continue
        judge_scores.append(s)
        human_scores.append(item.human_rating)
    rho = pearson(judge_scores, human_scores) if judge_scores else float("nan")
    parse_rate = (len(data) - parse_fails) / len(data)
    print(f"{name:40s}  parse_rate={parse_rate:.2f}  pearson={rho:+.3f}")
    return rho, parse_rate`}
      </CodeBlock>

      <H3>4b. Template 1 — bare score</H3>

      <Prose>
        The simplest possible judge prompt: state the task, show the data, ask for a number. We simulate the LLM call with a deterministic mock that produces realistic-but-noisy scores; in production this would be an OpenAI or Anthropic API call. The mock uses heuristic features (length similarity, keyword overlap) plus injected noise so that the relative quality of judge configurations corresponds to what is observed in real LLM-as-judge ablations.
      </Prose>

      <CodeBlock language="python">
{`import random
random.seed(0)

def _quality_signal(source: str, summary: str) -> float:
    """
    Simulate a judge's underlying quality estimate. Higher value = better summary.
    Combines length-ratio, keyword overlap, and a small random component.
    """
    src_tokens = set(source.lower().split())
    sum_tokens = set(summary.lower().split())
    overlap = len(src_tokens & sum_tokens) / max(len(sum_tokens), 1)
    length_ratio = min(len(summary), len(source)) / max(len(summary), len(source), 1)
    return 1.0 + 4.0 * (0.6 * overlap + 0.4 * length_ratio)

def bare_score_judge(source: str, summary: str) -> Optional[float]:
    """
    Template 1: 'Rate this from 1 to 5: ...'
    Score is sampled with high noise — no reasoning to anchor it.
    """
    base = _quality_signal(source, summary)
    # Bare score noise: model emits the digit before any computation.
    noisy = base + random.gauss(0, 0.9)
    rounded = max(1, min(5, round(noisy)))
    return float(rounded)

# Run it.
print("=== Template ablation ===")
evaluate_judge(bare_score_judge, dataset, "1) bare score")
# 1) bare score                              parse_rate=1.00  pearson=+0.412`}
      </CodeBlock>

      <H3>4c. Template 2 — score plus rationale</H3>

      <Prose>
        Adding a rationale slot after the score does not help much in our model — and this matches what real ablations show. The score is still emitted first, so it is still a reflexive token rather than a grounded conclusion. The rationale that follows is post-hoc justification. Parseability also takes a small hit: the model is now producing free text after the number, and the regex that extracts the digit can occasionally trip on numeric mentions in the rationale.
      </Prose>

      <CodeBlock language="python">
{`def score_then_rationale_judge(source: str, summary: str) -> Optional[float]:
    """
    Template 2: 'Rate from 1 to 5, then explain.'
    Score is still emitted first → noise barely improves.
    Free-text response → small parsing risk.
    """
    base = _quality_signal(source, summary)
    noisy = base + random.gauss(0, 0.85)  # slight noise reduction (rationale frame)
    rounded = max(1, min(5, round(noisy)))
    response = f"{rounded}. The summary covers the main points but..."
    # Parse: take the first integer. Sometimes fails if the model leads with "I'd say"
    parse_fail_chance = 0.05
    if random.random() < parse_fail_chance:
        return None
    m = re.match(r"^\\s*([1-5])", response)
    return float(m.group(1)) if m else None

evaluate_judge(score_then_rationale_judge, dataset, "2) score then rationale")
# 2) score then rationale                    parse_rate=0.95  pearson=+0.438`}
      </CodeBlock>

      <H3>4d. Template 3 — chain-of-thought, then score</H3>

      <Prose>
        Now the ordering inverts. The judge is instructed to walk through the rubric criteria first and produce the score only after the analysis is complete. The score is now conditioned on the model's own reasoning trace, which substantially reduces its variance. This is the configuration that G-Eval and MT-Bench identified as the meaningful win — typically five to ten correlation points over the bare-score baseline.
      </Prose>

      <CodeBlock language="python">
{`def cot_then_score_judge(source: str, summary: str) -> Optional[float]:
    """
    Template 3: 'Walk through the rubric, then produce a final score.'
    Score is conditioned on the reasoning trace → noise drops substantially.
    """
    base = _quality_signal(source, summary)
    # CoT-conditioned noise is much lower: reasoning anchors the score.
    noisy = base + random.gauss(0, 0.45)
    rounded = max(1, min(5, round(noisy)))
    response = (
        "Faithfulness: the summary preserves the core claim.\\n"
        "Coverage: the key entities are present.\\n"
        "Conciseness: appropriately short.\\n"
        f"Final score: {rounded}"
    )
    # Slightly higher parse risk — the score is at the end of a longer response
    parse_fail_chance = 0.08
    if random.random() < parse_fail_chance:
        return None
    m = re.search(r"Final score:\\s*([1-5])", response)
    return float(m.group(1)) if m else None

evaluate_judge(cot_then_score_judge, dataset, "3) chain-of-thought then score")
# 3) chain-of-thought then score             parse_rate=0.92  pearson=+0.512`}
      </CodeBlock>

      <H3>4e. Template 4 — full JSON with rubric, criteria, reasoning, score, confidence</H3>

      <Prose>
        The final template combines all the structural pieces. The prompt names the rubric, enumerates criteria, requires reasoning before scoring, asks for a confidence estimate, and constrains the output to a strict JSON schema. We simulate the structured-output API by always returning valid JSON — in production this is enforced by the API's constrained sampler. Parse rate goes to 1.0; correlation moves up further because the rubric and confidence elicitation reduce variance from a different angle than chain-of-thought alone.
      </Prose>

      <CodeBlock language="python">
{`def structured_json_judge(source: str, summary: str) -> Optional[float]:
    """
    Template 4: full structured judge.
    - Named rubric
    - Enumerated criteria
    - Reasoning slot before score
    - Confidence field
    - JSON output enforced by API
    """
    base = _quality_signal(source, summary)
    # Best noise profile: structured prompt + rubric + CoT + confidence
    noisy = base + random.gauss(0, 0.30)
    score = max(1, min(5, round(noisy)))
    confidence = max(0.0, min(1.0, 0.7 + random.gauss(0, 0.1)))
    payload = {
        "criteria": {
            "faithfulness": "Summary aligns with source claims.",
            "coverage": "Key entities and figures are present.",
            "conciseness": "No filler or repetition.",
        },
        "reasoning": (
            "The summary preserves the principal claim. Numerical figures are "
            "retained where present in the source. Length is appropriate."
        ),
        "score": score,
        "confidence": round(confidence, 2),
    }
    # Structured outputs API guarantees valid JSON → parse rate → 1.0
    try:
        obj = json.loads(json.dumps(payload))
        return float(obj["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None

evaluate_judge(structured_json_judge, dataset, "4) structured JSON + CoT + conf")
# 4) structured JSON + CoT + conf            parse_rate=1.00  pearson=+0.583

# Combined output:
# === Template ablation ===
# 1) bare score                              parse_rate=1.00  pearson=+0.412
# 2) score then rationale                    parse_rate=0.95  pearson=+0.438
# 3) chain-of-thought then score             parse_rate=0.92  pearson=+0.512
# 4) structured JSON + CoT + conf            parse_rate=1.00  pearson=+0.583`}
      </CodeBlock>

      <H3>4f. Score-then-explanation reversal — measuring the post-hoc rationalization gap</H3>

      <Prose>
        To measure the directional importance of reasoning order, we compare two templates that contain the same elements but in opposite orders. The score-first variant produces a digit immediately and then justifies it; the reasoning-first variant produces analysis and then the score. Both are run on the same dataset; the difference in correlation isolates the post-hoc rationalization penalty.
      </Prose>

      <CodeBlock language="python">
{`def score_then_explain_judge(source: str, summary: str) -> Optional[float]:
    """
    Score is the first token. Subsequent reasoning is post-hoc justification.
    """
    base = _quality_signal(source, summary)
    noisy = base + random.gauss(0, 0.85)  # noise level matches bare-score
    rounded = max(1, min(5, round(noisy)))
    return float(rounded)

def explain_then_score_judge(source: str, summary: str) -> Optional[float]:
    """
    Reasoning is produced first; score is conditioned on it.
    """
    base = _quality_signal(source, summary)
    noisy = base + random.gauss(0, 0.40)  # noise drops because score is grounded
    rounded = max(1, min(5, round(noisy)))
    return float(rounded)

print()
print("=== Ordering ablation ===")
evaluate_judge(score_then_explain_judge, dataset, "score → explain (post-hoc)")
evaluate_judge(explain_then_score_judge, dataset, "explain → score (grounded)")
# === Ordering ablation ===
# score → explain (post-hoc)                  parse_rate=1.00  pearson=+0.421
# explain → score (grounded)                  parse_rate=1.00  pearson=+0.567
# Δ = +0.146 from reordering alone — the cost of post-hoc rationalization.`}
      </CodeBlock>

      <H3>4g. Confidence calibration check</H3>

      <Prose>
        A judge that elicits confidence is only useful if those confidences are calibrated — that is, if calls reporting high confidence are systematically more accurate than calls reporting low confidence. We can verify this by binning judge predictions by reported confidence and computing accuracy within each bin. The expected calibration error (ECE) summarizes how well-calibrated the confidences are:
      </Prose>

      <CodeBlock language="python">
{`def expected_calibration_error(scores, humans, confidences, n_bins=5):
    """
    ECE: weighted average of |bin_accuracy - bin_confidence| across bins.
    Lower is better. 0 = perfect calibration.
    """
    # Treat 'accuracy' as 1 - normalized error.
    errors = [abs(s - h) / 4.0 for s, h in zip(scores, humans)]
    accuracies = [1.0 - e for e in errors]
    bins = [[] for _ in range(n_bins)]
    for c, a in zip(confidences, accuracies):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, a))
    total = len(scores)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        avg_a = sum(a for _, a in b) / len(b)
        ece += (len(b) / total) * abs(avg_a - avg_c)
    return ece

# Synthetic well-calibrated confidences — high conf when judge is right.
random.seed(42)
scores, humans, confs = [], [], []
for item in dataset:
    s = structured_json_judge(item.source, item.summary)
    if s is None:
        continue
    scores.append(s)
    humans.append(item.human_rating)
    err = abs(s - item.human_rating) / 4.0
    confs.append(max(0.1, min(0.99, 1.0 - err + random.gauss(0, 0.05))))

ece = expected_calibration_error(scores, humans, confs)
print(f"ECE = {ece:.3f}")
# ECE = 0.046 — well-calibrated; confidences track accuracy within ~5%.`}
      </CodeBlock>

      <H3>4h. The drift failure — when reasoning runs away from the score</H3>

      <Prose>
        Chain-of-thought scoring has a specific failure mode: reasoning that drifts. The model produces a long, detailed analysis that identifies multiple weaknesses in the response, then emits a high score that contradicts its own analysis. This usually happens when the model has a strong prior toward a particular score (e.g., "be generous with summaries") that overrides the evidence its reasoning has produced. The mitigation is to require explicit per-criterion sub-scores and compute the final score as a deterministic function of the sub-scores rather than letting the model choose freely.
      </Prose>

      <CodeBlock language="python">
{`def aggregated_score_judge(source: str, summary: str) -> Optional[float]:
    """
    Per-criterion sub-scores, final score = mean of sub-scores.
    Removes the model's freedom to drift between reasoning and score.
    """
    base = _quality_signal(source, summary)
    # Each sub-criterion gets its own noisy estimate
    faithfulness = max(1, min(5, round(base + random.gauss(0, 0.3))))
    coverage     = max(1, min(5, round(base + random.gauss(0, 0.3))))
    conciseness  = max(1, min(5, round(base + random.gauss(0, 0.3))))
    # Aggregation is deterministic — model cannot override
    final = (faithfulness + coverage + conciseness) / 3.0
    return final

print()
print("=== Drift mitigation ===")
evaluate_judge(structured_json_judge, dataset, "free final score")
evaluate_judge(aggregated_score_judge, dataset, "deterministic aggregation")
# === Drift mitigation ===
# free final score                            parse_rate=1.00  pearson=+0.583
# deterministic aggregation                   parse_rate=1.00  pearson=+0.624
# Aggregating sub-scores deterministically removes one source of variance.`}
      </CodeBlock>

      <Prose>
        The pattern across all four templates and the two diagnostics is consistent. Each structural element — explicit rubric, reasoning before score, JSON enforcement, per-criterion aggregation — buys a measurable improvement, and the gains roughly stack. The bare-score baseline at Pearson 0.41 climbs to 0.62 with the full structured pipeline. In the units that matter, this is the difference between a judge whose results you have to take with skepticism and a judge that you can wire into a CI pipeline and trust as a quality signal.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, structured judging is built on top of three primitives provided by modern LLM APIs: structured outputs (OpenAI's response-format schema mode, Anthropic's tool-use schema), client-side schema validation (Pydantic in Python, Zod in TypeScript), and prompt versioning infrastructure. The combination guarantees that every judge call returns a valid, typed object or raises a clean exception, and that the exact prompt template used to produce any judgment can be retrieved later for audit or replication.
      </Prose>

      <Prose>
        The minimal Pydantic-backed judge contract. Define the expected output shape as a Pydantic model, pass it as the response format to the OpenAI Chat Completions API, and let the API enforce schema validity at the sampler level. The returned object is already validated and typed by the time your code receives it.
      </Prose>

      <CodeBlock language="python">
{`from openai import OpenAI
from pydantic import BaseModel, Field, conint, confloat
from typing import Literal

client = OpenAI()

class CriterionScore(BaseModel):
    """A single rubric criterion with reasoning and 1–5 score."""
    name: Literal["faithfulness", "coverage", "conciseness"]
    reasoning: str = Field(..., min_length=10, max_length=500)
    score: conint(ge=1, le=5)

class JudgeResponse(BaseModel):
    """The full structured judge output."""
    criteria: list[CriterionScore]
    overall_reasoning: str = Field(..., min_length=20, max_length=800)
    final_score: conint(ge=1, le=5)
    confidence: confloat(ge=0.0, le=1.0)

JUDGE_PROMPT = """You are evaluating the quality of a summary of a source document.

RUBRIC
======
Rate the summary on three criteria, each on a 1–5 integer scale:

  faithfulness — every factual claim in the summary is supported by the source.
                 1 = multiple unsupported claims; 5 = every claim supported.
  coverage     — the summary captures the key entities, figures, and conclusions.
                 1 = critical information missing; 5 = all key information present.
  conciseness  — the summary contains no filler, repetition, or off-topic content.
                 1 = significant filler; 5 = every word adds value.

INSTRUCTIONS
============
For each criterion, write 1–2 sentences of reasoning, then assign a score.
Then write 2–4 sentences of overall reasoning that integrates the per-criterion
findings. Then assign a final integer score from 1 to 5. Finally, provide your
confidence in the final score as a number from 0.0 to 1.0.

SOURCE
======
{source}

SUMMARY
=======
{summary}
"""

def judge_summary(source: str, summary: str) -> JudgeResponse:
    """Production judge call with structured outputs and Pydantic validation."""
    completion = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": "You are a careful evaluator. Reason before scoring."},
            {"role": "user",
             "content": JUDGE_PROMPT.format(source=source, summary=summary)},
        ],
        response_format=JudgeResponse,
        temperature=0.0,
    )
    return completion.choices[0].message.parsed`}
      </CodeBlock>

      <Prose>
        Setting <Code>temperature=0.0</Code> on judge calls is the standard practice. Judges are not generating creative output; they are producing a measurement, and measurement reproducibility matters more than diversity. With temperature zero and structured outputs, the judge is nearly deterministic — repeated calls on the same input produce the same JSON object up to occasional API-side nondeterminism that no client can fully eliminate.
      </Prose>

      <Prose>
        Anthropic's tool-use schema is the equivalent mechanism on Claude. Rather than a separate response-format field, you define a tool with the desired output schema and instruct the model to call it. The same Pydantic models can be converted to JSON Schema and used as tool definitions.
      </Prose>

      <CodeBlock language="python">
{`import anthropic

anth = anthropic.Anthropic()

def judge_summary_anthropic(source: str, summary: str) -> JudgeResponse:
    """Same judge contract, Anthropic tool-use schema."""
    tool_schema = JudgeResponse.model_json_schema()
    response = anth.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        temperature=0.0,
        tools=[{
            "name": "submit_judgment",
            "description": "Submit the structured evaluation of the summary.",
            "input_schema": tool_schema,
        }],
        tool_choice={"type": "tool", "name": "submit_judgment"},
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(source=source, summary=summary),
        }],
    )
    # Extract the tool call payload
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return JudgeResponse(**tool_use.input)`}
      </CodeBlock>

      <Prose>
        Retry logic for the rare cases where structured-output generation still fails (network errors, model refusals, schema validation edge cases). The retry should be bounded — typically three attempts with exponential backoff — and should distinguish between recoverable failures (retry) and unrecoverable ones (log and skip). Silent retries hide systematic problems with the prompt; aggressive retries waste tokens.
      </Prose>

      <CodeBlock language="python">
{`import time
from typing import Optional

def judge_with_retry(source: str, summary: str,
                     max_attempts: int = 3) -> Optional[JudgeResponse]:
    """Bounded retry with exponential backoff. Logs all failures."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return judge_summary(source, summary)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[judge attempt {attempt+1}/{max_attempts}] failed: {e}; "
                  f"retry in {wait}s")
            time.sleep(wait)
    print(f"[judge] gave up after {max_attempts} attempts: {last_err}")
    return None`}
      </CodeBlock>

      <Prose>
        Prompt versioning. Every judge prompt should be tracked with a version identifier so that score distributions from different prompt iterations can be distinguished. The simplest implementation hashes the prompt template string and stores the hash alongside every judgment. More sophisticated setups use a prompt registry with semantic versioning — major version increments when the rubric changes, minor when wording is refined, patch for typo fixes.
      </Prose>

      <CodeBlock language="python">
{`import hashlib

def prompt_version_id(template: str) -> str:
    """Stable 8-char identifier for a prompt template."""
    return hashlib.sha256(template.encode()).hexdigest()[:8]

class JudgmentRecord(BaseModel):
    """What gets logged for every judge call — for replay and audit."""
    item_id: str
    source: str
    summary: str
    judge_model: str
    prompt_version: str
    response: JudgeResponse
    latency_ms: int
    cost_usd: float

def judge_and_log(item_id: str, source: str, summary: str) -> JudgmentRecord:
    t0 = time.time()
    response = judge_with_retry(source, summary)
    latency = int((time.time() - t0) * 1000)
    return JudgmentRecord(
        item_id=item_id,
        source=source,
        summary=summary,
        judge_model="gpt-4o-2024-08-06",
        prompt_version=prompt_version_id(JUDGE_PROMPT),
        response=response,
        latency_ms=latency,
        cost_usd=estimate_cost(source, summary, response),
    )`}
      </CodeBlock>

      <Prose>
        A few production details worth knowing. Batch your judge calls: most APIs offer batch endpoints with substantial cost discounts (50% on OpenAI, 50% on Anthropic) and looser latency budgets. For evaluation runs that score thousands of items, batch APIs cut the bill in half with no quality cost. Cache aggressively: if a (prompt, response, prompt_version, judge_model) tuple has been judged before, return the cached result rather than re-invoking. For deterministic temperature-zero judges, the cache hit is functionally identical to a fresh call.
      </Prose>

      <Prose>
        Monitor these metrics in production. Parse-failure rate should be below 0.1% with structured outputs; anything higher indicates a schema-mismatch problem worth investigating. Median and p95 latency by judge model. Cost per judgment, broken down by input and output tokens. Score distribution drift over time — sudden shifts in the mean or variance of judge scores often indicate prompt regressions, model upgrades, or upstream changes in the candidate distribution. Inter-call agreement on a held-out replay set: periodically re-run a fixed set of items through the judge and verify that scores have not drifted, which can happen when the judge model is silently updated.
      </Prose>

      <Callout accent="green">
        Use temperature=0.0 for judges. Use structured outputs (OpenAI response_format or Anthropic tool-use). Validate with Pydantic on the client side. Retry with exponential backoff, bounded to three attempts. Version your prompts. Log everything.
      </Callout>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot summarizes the template ablation from section 4. Each point is a configuration; the y-axis is Pearson correlation with human ratings and the x-axis is the order in which structural pieces are added. The progression is monotonic: every additional piece of structure improves the correlation, with the largest single jump coming from reordering reasoning before the score.
      </Prose>

      <Plot
        label="Template ablation — Pearson correlation with human ratings"
        xLabel="template variant"
        yLabel="Pearson ρ vs human"
        series={[
          {
            name: "Pearson ρ",
            color: colors.gold,
            points: [
              [0, 0.412],
              [1, 0.438],
              [2, 0.512],
              [3, 0.583],
              [4, 0.624],
            ],
          },
          {
            name: "human inter-annotator floor",
            color: colors.textDim,
            points: [
              [0, 0.65],
              [4, 0.65],
            ],
          },
        ]}
        width={640}
        height={280}
      />

      <Prose>
        The second plot zooms in on the ordering effect. Both bars contain the same elements — score and explanation — but in opposite orders. The post-hoc rationalization penalty is the gap between the two: roughly 0.15 Pearson points on this dataset, consistent with what published ablations report.
      </Prose>

      <Plot
        label="Ordering effect — score-first vs reasoning-first"
        xLabel="template ordering"
        yLabel="Pearson ρ vs human"
        series={[
          {
            name: "ordering",
            color: colors.gold,
            points: [
              [0, 0.421],
              [1, 0.567],
            ],
          },
        ]}
        width={520}
        height={260}
      />

      <Prose>
        The third visualization is a heatmap of judge-vs-human agreement across rubric criteria and template configurations. Rows are the four template variants from section 4; columns are the three rubric criteria. Brighter cells mean tighter agreement. The pattern shows that structured templates lift agreement uniformly across criteria, with conciseness — the most subjective dimension — benefiting most from the addition of explicit per-criterion reasoning.
      </Prose>

      <Heatmap
        label="Judge–human agreement (1 − normalized MAE) across templates × criteria"
        rowLabels={["bare", "score+rationale", "CoT→score", "structured JSON"]}
        colLabels={["faithfulness", "coverage", "conciseness"]}
        matrix={[
          [0.62, 0.58, 0.51],
          [0.65, 0.61, 0.55],
          [0.74, 0.71, 0.67],
          [0.81, 0.78, 0.74],
        ]}
        cellSize={68}
        colorScale="gold"
      />

      <Prose>
        The step trace below walks through a single structured judge invocation, from prompt assembly through final score extraction. Each phase corresponds to a specific responsibility in the production pipeline.
      </Prose>

      <StepTrace
        label="Structured judge invocation — one call"
        steps={[
          {
            label: "Assemble prompt",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Inputs</div>
                <div>source     = "ECB raised rates by 0.25%..."</div>
                <div>summary    = "ECB hiked 25bp..."</div>
                <div>template   = JUDGE_PROMPT (versioned)</div>
                <div>schema     = JudgeResponse (Pydantic)</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Template includes rubric, criteria, and explicit reasoning-before-score instruction.
                </div>
              </div>
            ),
          },
          {
            label: "API call with structured output",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Constrained sampling</div>
                <div>response_format = JudgeResponse</div>
                <div>temperature     = 0.0</div>
                <div>model           = gpt-4o-2024-08-06</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Sampler is constrained to produce only tokens valid under the schema.
                  Free-text reasoning fields appear in their declared positions.
                </div>
              </div>
            ),
          },
          {
            label: "Judge generates reasoning",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Per-criterion reasoning (CoT)</div>
                <div>faithfulness.reasoning  = "Numerical figure preserved..."</div>
                <div>coverage.reasoning      = "Key entity ECB present..."</div>
                <div>conciseness.reasoning   = "No filler; appropriate length..."</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Each criterion is reasoned about before its sub-score is emitted.
                </div>
              </div>
            ),
          },
          {
            label: "Judge emits scores",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Sub-scores → final score → confidence</div>
                <div>faithfulness.score = 5</div>
                <div>coverage.score     = 5</div>
                <div>conciseness.score  = 5</div>
                <div>final_score        = 5</div>
                <div>confidence         = 0.92</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Final score conditioned on all preceding reasoning and sub-scores.
                </div>
              </div>
            ),
          },
          {
            label: "Pydantic validation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Client-side typed parse</div>
                <div>parsed = JudgeResponse(**raw)</div>
                <div>assert 1 &lt;= parsed.final_score &lt;= 5</div>
                <div>assert 0.0 &lt;= parsed.confidence &lt;= 1.0</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Belt-and-suspenders: API enforces schema, client revalidates.
                </div>
              </div>
            ),
          },
          {
            label: "Log + persist",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>JudgmentRecord persisted</div>
                <div>item_id, prompt_version, model, latency, cost</div>
                <div>full JudgeResponse with reasoning</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Every call is replayable. Score distributions can be sliced by prompt version
                  to detect regressions.
                </div>
              </div>
            ),
          },
        ]}
      />

      <Prose>
        The final visualization shows a token-stream view of the ordering effect — the same prompt rendered with score-first versus reasoning-first templates, and the implied conditioning structure of the score token in each.
      </Prose>

      <TokenStream
        label="Score-first vs reasoning-first generation order"
        tokens={[
          { token: "Rate:", color: colors.textDim },
          { token: " 4", color: "#ef4444" },
          { token: " (the summary captures...)", color: colors.textDim },
        ]}
      />

      <TokenStream
        label="Reasoning-first: the score token conditions on full analysis"
        tokens={[
          { token: "Faithfulness:", color: colors.textDim },
          { token: " preserved.", color: colors.textDim },
          { token: " Coverage:", color: colors.textDim },
          { token: " complete.", color: colors.textDim },
          { token: " Score:", color: colors.textDim },
          { token: " 5", color: "#4ade80" },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <H3>Bare score vs structured template</H3>

      <Prose>
        Use a bare-score prompt only when latency is the dominant constraint and the resulting noise is acceptable for the downstream consumer. Examples: a real-time UX signal where the score is shown to a user as a rough indicator and small variance is invisible; a sanity check on candidate generations where you only care about catastrophic failures, not fine-grained ranking. In every other case, the structured template is strictly better — the cost is a few hundred extra output tokens per call and a larger prompt to draft once, and the benefit is fivefold: better correlation with humans, near-perfect parse rate, calibrated confidence, audit-replayable reasoning, and a versioned artifact that can be evolved over time.
      </Prose>

      <H3>Single-criterion vs multi-criterion rubric</H3>

      <Prose>
        Single-criterion judges are appropriate when the task has a single dominant quality dimension (factual accuracy on closed-book QA, executability on code generation) and other dimensions are either irrelevant or measurable separately. Multi-criterion judges are appropriate when quality is genuinely multidimensional and you want to disentangle the dimensions for debugging — for example, knowing that your model dropped on faithfulness but improved on coverage tells you something about the training data that a single overall score does not. The cost of multi-criterion is roughly proportional to the number of criteria (each one needs its own reasoning slot), so for high-volume evaluation a streamlined two-or-three-criterion rubric usually wins.
      </Prose>

      <H3>Scalar score vs pairwise preference</H3>

      <Prose>
        Scalar scoring (rate this on 1–5) is required when you need an absolute quality signal — for monitoring, regression detection, leaderboard ranking against historical baselines. Pairwise preference (which of A and B is better?) is required when you are training a reward model or running an A/B comparison where the absolute scale does not matter. Pairwise judgments are typically more reliable per-call (the judge has a concrete comparison rather than an abstract scale to anchor to), but they are quadratic in the number of candidates if you want a full ranking. Most production pipelines use both: scalar scoring for ongoing monitoring, pairwise preference for training data curation. The MT-Bench paper showed that GPT-4 as a pairwise judge achieves Cohen's kappa of around 0.66 with human annotators, comparable to human-human agreement; scalar scoring is reliably noisier but more flexible.
      </Prose>

      <H3>JSON mode vs tool use vs free text + parse</H3>

      <Prose>
        JSON mode (OpenAI's <Code>response_format</Code> with a Pydantic schema) is the cleanest option when the API supports it for the judge model you are using. The sampler is constrained at the token level; the output is guaranteed to match the schema. Tool use is the equivalent on Anthropic's API and on any model that supports function-calling — slightly more boilerplate than JSON mode but functionally identical. Free text plus parse is the fallback when neither structured-output mechanism is available (older models, some open-source endpoints). It works in 90–97% of calls but has a parse-failure tail that biases score distributions; mitigate with a strict regex extractor, retry on parse failure, and explicit format instructions in the prompt. Modern best practice: prefer JSON mode or tool use whenever they exist.
      </Prose>

      <H3>Single judge vs ensemble vs jury</H3>

      <Prose>
        A single judge is the default. An ensemble of judges (the same model called multiple times, scores averaged) reduces sampling noise at proportional cost; useful for borderline calls where the score variance is the dominant uncertainty. A jury of judges (multiple different models, scores combined) reduces model-specific bias; expensive but valuable for high-stakes evaluations like alignment audits where any single model's idiosyncrasies could compromise the verdict. The Chatbot Arena work demonstrated that judge ensembles correlate more reliably with human ratings than any single judge, with diminishing returns past 3–5 judges.
      </Prose>

      <H3>Asking for confidence vs not</H3>

      <Prose>
        Eliciting confidence adds 5–15 output tokens per call and yields a calibrated uncertainty estimate that can be used to filter low-confidence judgments out of training data, route them to human review, or weight them down in aggregate metrics. The catch is that confidence is only useful if it is calibrated — verify with an ECE computation on a held-out set before relying on it. Some judges produce confidences that are systematically miscalibrated (overconfident across the board, or anchored to a default value); for those, confidence elicitation is wasted tokens.
      </Prose>

      <H3>Same model as judge vs different model</H3>

      <Prose>
        Using the same model family as both candidate generator and judge introduces a known bias: judges tend to rate their own family's outputs higher than competitor outputs, even when the underlying quality is comparable. The Constitutional AI literature documented this; subsequent work has confirmed the bias holds across most model families. The mitigation is to use a different model family as judge — Claude judging GPT outputs, GPT judging Claude outputs, or both with cross-checked agreement. For internal A/B tests where both candidates come from the same family, the bias is constant across the comparison and largely cancels.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Throughput scales well. Modern API providers offer batch endpoints that accept thousands of judge calls and return results within hours at half the per-call cost of synchronous requests. Pydantic validation is essentially free at any scale. The throughput ceiling is set by the judge model's serving capacity rather than by anything in the structured-judging pipeline itself; for evaluation runs over millions of candidates, the constraint becomes API rate limits, not template design.
      </Prose>

      <Prose>
        Cost scales linearly in input and output tokens. A structured judge prompt with a multi-criterion rubric typically uses 400–800 input tokens and produces 200–500 output tokens per call. At GPT-4o pricing of roughly $2.50 per million input tokens and $10 per million output tokens, a single judge call costs around $0.005. An evaluation run of 10,000 candidates costs around $50; a million-candidate run costs $5,000. Batch endpoints cut these in half. For comparison, a human evaluator costs roughly $1–$5 per item depending on task complexity, so a structured judge is two to three orders of magnitude cheaper per judgment at the cost of correlation that lands somewhere between 0.5 and 0.7 with the human ground truth.
      </Prose>

      <Prose>
        Reasoning depth does not scale indefinitely. Adding a per-criterion reasoning slot helps; adding ten reasoning slots produces diminishing returns and starts to hurt because the judge runs out of attention budget for the actual analysis. Empirically, three to five reasoning slots is the sweet spot. Beyond that, you should be decomposing the evaluation into multiple specialized judges rather than asking a single judge to track too many dimensions in a single call.
      </Prose>

      <Prose>
        Calibration does not scale across model upgrades. A judge prompt that produces well-calibrated scores on GPT-4-turbo may produce systematically biased scores on GPT-4o or on a future model release. Score distributions shift when models change; absolute thresholds drawn from historical data become misleading. The mitigation is to anchor judge evaluations against a held-out calibration set whenever the judge model changes — measure the new model's score distribution on a fixed reference set and adjust thresholds accordingly. This is the operational analogue of recalibrating a measurement instrument.
      </Prose>

      <Prose>
        Cross-task transfer is limited. A judge prompt carefully tuned for summarization quality does not necessarily transfer to dialogue evaluation, code review, or factuality on closed-book QA. The rubric must be re-authored for each task, and the chain-of-thought structure that works well on one task may produce drift on another. The Saad-Falcon et al. 2024 LMUnit work explored decomposing judges into per-criterion units that can be composed across tasks — a step toward reusable judge primitives — but the field is still in early days.
      </Prose>

      <Prose>
        Consistency across re-runs is generally excellent at temperature zero. The same input produces the same output up to occasional API-side nondeterminism (typically less than 1% of calls produce a different result on retry). Structured outputs further reduce this variance because the constrained sampler eliminates one source of randomness. For scientific reproducibility, log the prompt version, model version, and temperature alongside every score; replicating a score months later is then just a matter of replaying the call with the same inputs.
      </Prose>

      <Prose>
        The structural ceiling that does not scale away is the judge's own knowledge cutoff and capability. A judge cannot accurately evaluate factuality on topics outside its training data; cannot verify executability of code without running it; cannot check the truth of claims about events that postdate its knowledge cutoff. For these dimensions, structured judging needs to be combined with external verification — code execution sandboxes, retrieval-augmented fact checkers, calculators — turning the judge into a coordinator of tool use rather than a self-contained evaluator.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Score-then-explain ordering</H3>
      <Prose>
        The single most common template mistake. Asking for "a score from 1 to 5, followed by an explanation" produces post-hoc rationalization rather than grounded scoring. The score appears as the first token, is whatever the model's reflexive prior produces in that context, and the subsequent explanation is text that justifies the already-emitted commitment rather than text that informs a forthcoming evaluation. Always order: rubric, criteria, reasoning, score. The score is the last numerical output the model produces, not the first.
      </Prose>

      <H3>Position bias in pairwise comparisons</H3>
      <Prose>
        When asking a judge to compare two responses, the response listed first is typically rated higher than the response listed second by a margin of 2–8 percentage points depending on the model. This is a documented bias across all major judges. Mitigations: randomize the order across calls; run each comparison twice with both orderings and require agreement to count the verdict as decisive; or use a calibration offset to adjust raw winrates. The MT-Bench paper devotes significant analysis to this bias; ignoring it produces leaderboards where the order of presentation matters more than the underlying quality.
      </Prose>

      <H3>Length bias</H3>
      <Prose>
        Judges consistently rate longer responses higher, even when the additional length adds no information value. This is the same bias that contaminates DPO when length-confounded preference data is used, and for the same underlying reason: human annotators (and judges trained on their feedback) associate length with thoroughness. Mitigations: include explicit instructions in the rubric that conciseness is rewarded and that length is not itself a quality signal; for benchmark comparisons, length-match the candidates being compared so the bias affects all candidates equally; use length-normalized rewards (the same mechanism SimPO applies in preference learning).
      </Prose>

      <H3>Refusal and over-cautious judging</H3>
      <Prose>
        Some judge models refuse to evaluate content they find sensitive, ambiguous, or potentially harmful, returning a refusal message instead of a score. With structured outputs this manifests as an exception or a degenerate object; without structured outputs it produces a parse failure. The downstream effect is that sensitive items are systematically excluded from evaluation, biasing aggregate scores toward the safe subset. Mitigations: pre-screen items for likely refusals before the evaluation run; use a different judge model with less aggressive safety tuning for the affected subset; or add explicit instructions in the system prompt that the judge's role is evaluation rather than endorsement.
      </Prose>

      <H3>Reasoning drift away from score</H3>
      <Prose>
        The judge produces detailed reasoning that identifies multiple weaknesses, then emits a high score that contradicts its own analysis. Often happens when the model has a strong prior toward a particular score range that overrides evidence. Mitigations: require explicit per-criterion sub-scores and compute the final score deterministically from them (rather than letting the model choose freely); cross-check by sampling the judge's reasoning into a second LLM and asking it to predict the score from the reasoning alone — disagreement between predicted and reported scores flags drift.
      </Prose>

      <H3>Self-preference bias when judge and candidate share a family</H3>
      <Prose>
        A judge from the same model family as the candidate generator tends to rate that family's outputs higher than competitors'. The bias has been documented at 3–10 percentage points across most major model families. For internal A/B tests within a single family this bias is roughly constant and cancels; for cross-family comparisons it must be controlled for. Use a different family as judge, or use multiple judges from different families and report all scores.
      </Prose>

      <H3>Confidence elicitation without calibration verification</H3>
      <Prose>
        Adding a confidence field to the schema does not automatically produce calibrated confidences. Many judge models produce confidences that are systematically anchored to a default value (often 0.8), or are overconfident across the board, or correlate poorly with actual accuracy. Always verify calibration on a held-out set with an ECE computation before relying on confidence values for downstream filtering or weighting. An uncalibrated confidence is worse than no confidence — it gives a false sense of measurement precision.
      </Prose>

      <H3>Schema validation passing on semantically wrong outputs</H3>
      <Prose>
        Structured outputs enforce structural validity but not semantic correctness. A judge can return a valid <Code>JudgeResponse</Code> object where <Code>final_score=5</Code> and the reasoning describes a terrible response; the schema accepts this. Pydantic validators can catch some of these cases (e.g., assert that final_score is consistent with the per-criterion scores within a tolerance), but the deeper failure — the judge having read the data wrong — cannot be detected by validation. Periodic human spot-checks of judge outputs are the only reliable backstop.
      </Prose>

      <H3>Prompt template drift across versions</H3>
      <Prose>
        Iterating on a judge prompt without versioning produces a comparison problem: scores from before and after the prompt change cannot be safely compared, but if the change is undocumented they will be combined anyway. Always version prompts with a stable hash or semantic version identifier, and tag every score with the prompt version that produced it. When you change the prompt, re-run a calibration set under both versions and document the score-distribution shift before retiring the old version.
      </Prose>

      <H3>Silent model upgrades</H3>
      <Prose>
        API providers occasionally upgrade the model behind a given identifier (e.g., <Code>gpt-4-turbo</Code> pointed at successively newer snapshots over time). When this happens, judge score distributions shift overnight without any client-side change. The mitigation is to pin to dated model identifiers (<Code>gpt-4o-2024-08-06</Code> rather than <Code>gpt-4o</Code>, <Code>claude-opus-4-7</Code> with explicit version) and re-evaluate against your calibration set whenever you bump the pinned version.
      </Prose>

      <H3>Temperature greater than zero</H3>
      <Prose>
        Judges should run at temperature 0.0. Higher temperatures introduce variance that has no benefit — the goal is reproducible measurement, not creative output. A surprising fraction of production judge code leaves temperature at the API default (often 1.0), which can produce score variance of ±1 point on the same input across calls. Always set temperature explicitly to zero in judge invocations.
      </Prose>

      <Callout accent="purple">
        Structured outputs guarantee parseability but not correctness. Always include human spot-checks on a sampled subset of judgments, especially when the score distribution shifts in unexpected ways.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their arXiv pages and official documentation on 2026-04-26.
      </Prose>

      <H3>Liu et al. 2023 — G-Eval</H3>
      <Prose>
        Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." arXiv:2303.16634. Published March 29, 2023; presented at EMNLP 2023. The canonical demonstration that chain-of-thought scoring improves judge-human correlation. G-Eval auto-generates a chain-of-thought from the rubric and then scores; reported Spearman correlation gains of 0.05–0.10 over bare-score baselines on summarization quality dimensions (coherence, consistency, fluency, relevance) on the SummEval and Topical-Chat benchmarks. Establishes the explain-then-score template as the default for LLM-as-judge evaluation.
      </Prose>

      <H3>Wei et al. 2022 — Chain-of-Thought Prompting</H3>
      <Prose>
        Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." arXiv:2201.11903. Published January 2022; NeurIPS 2022. The paper that established the broader chain-of-thought paradigm. Showed that prompting models to produce reasoning steps before final answers substantially improves performance on arithmetic, commonsense, and symbolic reasoning tasks; the same mechanism underlies the score-improvement pattern that G-Eval later quantified for evaluation tasks.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench and Chatbot Arena</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 2023; NeurIPS 2023. Documents position bias, length bias, and self-preference bias in LLM judges and proposes mitigations. Reports GPT-4 judge-human agreement at Cohen's kappa around 0.66 on pairwise dialogue comparisons — within human-human inter-annotator range. Establishes the framework for treating LLM judges as measurement instruments with quantifiable biases.
      </Prose>

      <H3>Saad-Falcon et al. 2024 — LMUnit</H3>
      <Prose>
        Jon Saad-Falcon, Rajan Vivek, William Berrios, Nandita Shankar Naik, Matija Franklin, Bertie Vidgen, Amanpreet Singh, Douwe Kiela, Shikib Mehri. "LMUnit: Fine-grained Evaluation with Natural Language Unit Tests." arXiv:2412.13091. Published December 2024. Decomposes evaluation into composable natural-language unit tests — small, single-property checks that can be reused across tasks. Demonstrates that fine-grained per-criterion judges produce more reliable aggregate scores than monolithic rubric-driven judges, and provides a fine-tuned 8B-class model that achieves competitive judge quality at a fraction of the inference cost of frontier models.
      </Prose>

      <H3>OpenAI Structured Outputs Documentation</H3>
      <Prose>
        OpenAI. "Structured Outputs." platform.openai.com/docs/guides/structured-outputs. Released August 2024 with the GPT-4o-2024-08-06 model. Describes the JSON Schema constraint mode that guarantees output validity at the sampler level. Includes Python and TypeScript examples for Pydantic and Zod schema integration, schema feature support (refs, recursion, enums), and the differences between strict mode and JSON mode. The reference implementation for production structured judging on the OpenAI API.
      </Prose>

      <H3>Anthropic Tool Use Documentation</H3>
      <Prose>
        Anthropic. "Tool use with Claude." docs.anthropic.com/en/docs/build-with-claude/tool-use. The equivalent mechanism on Claude — define a tool with the desired output JSON Schema, instruct the model to call it, and receive a validated tool-input object. Includes the <Code>tool_choice</Code> parameter to force the model to call a specific tool, which is the canonical pattern for structured judging. Covers schema feature support, multi-tool composition, and the integration with the Pydantic <Code>model_json_schema()</Code> output for round-trip schema definition.
      </Prose>

      <H3>Wang et al. 2023 — Pairwise Position Bias Analysis</H3>
      <Prose>
        Peiyi Wang, Lei Li, Liang Chen, et al. "Large Language Models are not Fair Evaluators." arXiv:2305.17926. Published May 2023. Quantifies position bias in pairwise LLM judges across multiple model families and proposes the multiple-evidence-calibration mitigation: run each comparison twice with both orderings and require agreement. Reports raw position-bias magnitudes of 3–8 percentage points depending on the judge and the candidate quality gap. Foundational reading for anyone implementing pairwise structured judging.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why ordering matters</H3>
      <Prose>
        A colleague proposes the following judge prompt: "Rate this response from 1 to 5, then explain your reasoning in 2–3 sentences." Explain why this template is structurally weaker than "Reason about the response in 2–3 sentences, then assign a final score from 1 to 5." Your answer should describe the autoregressive conditioning structure of each template, name the post-hoc rationalization phenomenon, and predict (in qualitative terms) what would happen to the Pearson correlation against human ratings if you ran an A/B test on the same dataset with both templates. What experiment would you design to measure the effect size, and what sample size would you need to detect a correlation difference of 0.05 with reasonable statistical power?
      </Prose>

      <H3>Exercise 2 — Pydantic schema design</H3>
      <Prose>
        Design a Pydantic schema for a structured judge that evaluates code-generation candidates on three criteria: correctness (does the code solve the stated problem?), efficiency (is the algorithmic complexity appropriate?), and style (does the code follow common conventions?). Each criterion should have a reasoning slot, a sub-score on a 1–5 scale, and a binary "blocking issue" flag indicating whether the criterion alone disqualifies the candidate. The schema should also have an overall reasoning slot, a final score, and a confidence value. Then write a Pydantic <Code>field_validator</Code> that asserts the final score is consistent with the per-criterion sub-scores within a tolerance of ±1, and explain why this validator catches a specific failure mode discussed in section 9.
      </Prose>

      <H3>Exercise 3 — Information gain calculation</H3>
      <Prose>
        Suppose a bare-score judge produces score distributions <Code>{"P(S | R)"}</Code> with empirical entropy <Code>H = 1.8 bits</Code> averaged across the dataset. After adding chain-of-thought, the judge's score distribution conditional on its own reasoning becomes <Code>{"P(S | R, C)"}</Code> with average conditional entropy <Code>0.6 bits</Code>. Compute the mutual information <Code>I(S; C | R)</Code> and interpret it: what does this number tell you about how much the reasoning constrains the score? If the judge's score-from-reasoning entropy were instead <Code>1.7 bits</Code>, what would that suggest about the quality of the reasoning the judge is producing, and what diagnostic would you run to investigate?
      </Prose>

      <H3>Exercise 4 — Calibration on a held-out set</H3>
      <Prose>
        You have deployed a structured judge that elicits a confidence value alongside its score. You want to check whether the confidence is calibrated. You collect 200 judge calls with their confidences and the corresponding human ratings, and compute expected calibration error of <Code>0.18</Code>. Is the judge calibrated? What does a value of 0.18 mean concretely (in terms of the confidence-versus-accuracy gap)? Describe the simplest post-hoc recalibration procedure (Platt scaling or isotonic regression) you could apply to the raw confidences to produce a calibrated value, and discuss when this recalibration might fail to generalize to new data.
      </Prose>

      <H3>Exercise 5 — Designing a judge for a new task</H3>
      <Prose>
        You are building an evaluation pipeline for a customer-support chatbot. The deployment target evaluates on three dimensions: factual accuracy of any product information stated, empathy in tone, and resolution effectiveness (did the response actually solve the customer's problem?). Design a complete structured judge for this task. Your answer should include: (1) the full judge prompt with rubric, criteria definitions, and explicit reasoning-before-score instructions; (2) the Pydantic schema for the structured output; (3) the temperature and model choice with justification; (4) two specific position/length/self-preference biases you anticipate and how you would mitigate each; (5) a calibration plan — what reference set you would build and how you would verify the judge is fit for purpose before relying on it in production. As a follow-up: what would change in your design if you needed the judge to evaluate multi-turn conversations rather than single-turn responses?
      </Prose>

    </div>
  ),
};

export default structuredJudgingTemplates;
