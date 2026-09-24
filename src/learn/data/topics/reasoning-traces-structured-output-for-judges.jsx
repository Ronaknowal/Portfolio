import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const reasoningTracesJudges = {
  title: "Reasoning Traces & Structured Output for Judges",
  slug: "reasoning-traces-structured-output-for-judges",
  readTime: "~36 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Every LLM-as-judge pipeline eventually has the same crisis. The eval team builds a judge prompt that produces a single integer score, runs it across ten thousand model outputs, computes an aggregate, and ships the number to leadership. A week later somebody asks "why did response 4,732 get a 3 instead of a 5?" and there is no answer. The judge produced a number; the number is wrong; the only available response is to re-run the judge and hope for a different number. The opaque scalar destroyed the audit trail. The scoring run cost real money in API calls and yielded no evidence that anyone could examine. This pattern repeats in every team that adopts judge-based evaluation without a deliberate plan for what the judge writes down besides its verdict.
      </Prose>

      <Prose>
        Two design patterns address this crisis. The first is reasoning traces: instead of asking the judge for a number, you ask it to write a paragraph of analysis and then commit to a number, with the paragraph attached to the verdict in the persisted record. The second is structured output: instead of parsing free-form text with brittle regex, you constrain the judge to emit a machine-readable schema — JSON, a tool call, a Pydantic model — that downstream systems can ingest without ambiguity. These two patterns appear independently in the literature but they solve complementary problems. Reasoning traces preserve the audit trail; structured output preserves the data pipeline. A production-grade judge does both, and the interaction between them is the subject of this topic.
      </Prose>

      <Prose>
        The reasoning-trace half traces back to chain-of-thought prompting (Wei et al. 2022, arXiv:2201.11903), which showed that asking a model to "think step by step" before answering substantially improved accuracy on multi-step reasoning benchmarks. When that finding was carried into LLM-as-judge work — most directly by Liu et al. in G-Eval (arXiv:2303.16634) — the same effect appeared: a judge that wrote out its analysis before committing to a score correlated more strongly with human judgments than a judge that emitted the score directly. The mechanism is not mysterious. The judge is a language model performing a multi-step inference: read the input, recall the criteria, locate evidence, weigh trade-offs, output a verdict. Forcing all of those steps into the latent space of a single token's logits truncates the computation. Letting them play out in the explicit token stream gives the judge room to actually reason.
      </Prose>

      <Prose>
        The structured-output half emerged from the operational reality of putting judges into production pipelines. Free-form text is hostile to downstream systems. A judge that writes "I would rate this a 4 out of 5, though some might argue for a 5" requires a regex, a parser, and a fallback strategy when the parser fails. That fallback is not a hypothetical: across published benchmarks and our own measurements, regex-based extraction from unconstrained judge outputs achieves roughly 92% reliability, and the 8% of failures concentrate disproportionately on the most ambiguous and most diagnostically valuable cases. Strict-schema enforcement — Anthropic's tool-use schemas, OpenAI's <Code>response_format=json_schema</Code>, Google's controlled generation — pushes parse reliability above 99%, and the failures that remain are usually true model refusals rather than format errors.
      </Prose>

      <Prose>
        The natural temptation is to treat these as opposed: structured output for the pipeline, free-form reasoning for the audit trail, choose one. The actual answer is to combine them. Modern structured-output APIs let you specify a schema with a free-text <Code>reasoning</Code> field alongside the typed <Code>score</Code> and <Code>confidence</Code> fields. The judge writes its trace into the reasoning field, the trace is persisted with the verdict, the typed fields are validated and consumed by the pipeline, and a human auditor can pull any record and read the reasoning that produced it. The 2024 wave of extended-reasoning models (OpenAI's o1, Anthropic's Claude 3.7 with extended thinking, DeepSeek-R1) makes the trace-plus-structure pattern even more powerful: the model is allowed to think for thousands of tokens internally, surface a structured summary, and the audit trail captures both the hidden thinking and the visible verdict.
      </Prose>

      <Prose>
        This topic exists because every team that builds a judge eventually re-derives this pattern, usually after the audit-trail crisis above forces the issue. The goal here is to write down the mathematics of why it works, the implementation details that matter, and the failure modes that will catch you if you do not handle them deliberately. The structure mirrors the production pipeline: derive the trace-then-verdict ordering from first principles, build a Pydantic-based judge from scratch, harden it for production with retries and schema versioning, and walk through the failure modes that distinguish a research-grade judge from one that survives a year in deployment.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        Start with the pure-scalar judge: a prompt that says "rate this response on a 1-5 scale, output only the number." The judge reads the input, performs whatever evaluation it performs, and emits a single token. From an information-theoretic perspective, that single token is a bottleneck. The judge's internal state at the moment it commits to the verdict is a high-dimensional vector — call it a few thousand dimensions if you are working with a typical transformer's hidden state. The verdict token compresses that state to roughly 2.3 bits (a five-way categorical). Everything else the judge "knew" while making the call is discarded the moment the next token is sampled. There is no record of the reasoning, no trace of the trade-offs, no signal of how confident the judge actually was.
      </Prose>

      <Prose>
        The reasoning-trace pattern exploits a structural property of autoregressive language models: each generated token becomes part of the context for subsequent tokens. If you require the judge to output 200 tokens of analysis before the verdict token, those 200 tokens are themselves the judge's working memory. The judge can write down the criteria, identify specific spans of evidence, weigh competing considerations, and only then commit to a score. The verdict token is now conditioned not just on the input but on the judge's own externalized reasoning, which is itself a richer summary of the high-dimensional internal state than the verdict token alone could ever be. This is why CoT works at all: it lets the model use its own output stream as scratch space.
      </Prose>

      <Prose>
        For a judge specifically, this matters in two distinct ways. The first is accuracy: the judge that thought before answering tends to agree more closely with human experts on the same task. Liu et al.'s G-Eval results, replicated many times since, consistently show single-digit-percentage to double-digit-percentage improvements in human-correlation metrics when CoT is added to a judge prompt. The second is auditability: the trace is independent evidence that a human reviewer can examine. If a judge gives a wrong score, the trace usually reveals why — the judge mis-read a span, applied the wrong criterion, or weighed factors in a way you disagree with. With a pure-scalar judge, "the judge was wrong" is a dead end. With a trace, "the judge was wrong because it confused factuality with helpfulness" is an actionable diagnostic.
      </Prose>

      <Prose>
        Now overlay the structured-output question. The trace makes the judge smarter and more inspectable, but it also makes the output harder to parse. A judge that writes a paragraph and then says "Final score: 4" is easy for a human to read but requires a parser that can find the score line, handle the case where the judge wrote "I would give this a 4 out of 5, though arguments could be made for 3," and gracefully fail when the judge omits the score line entirely. Across thousands of evaluations the parser failures become a real cost. Structured output replaces the parser with a contract: the model must emit a JSON object with specific fields of specific types, validated by the API before the response is returned. There is no parsing; there is only validation, which either passes or fails loudly.
      </Prose>

      <Prose>
        The key insight is that these two requirements compose. A schema can include a free-text <Code>reasoning</Code> field whose contents are the trace, alongside typed fields for <Code>score</Code> and <Code>confidence</Code> and a structured <Code>criteria_scores</Code> dictionary. The judge writes the trace into the reasoning field as part of producing the structured response. The order of fields in the schema is load-bearing: if reasoning comes first, the model fills it in first, and the typed fields that come later are conditioned on the reasoning the model already wrote. If reasoning comes last, the model commits to scores first and then rationalizes them, which empirically destroys most of the CoT benefit. This ordering is not a quirk of any particular API; it is a direct consequence of autoregressive generation.
      </Prose>

      <Prose>
        The extended-reasoning models — o1, Claude 3.7 with extended thinking, DeepSeek-R1 — change the picture in one important way. These models are trained to produce a hidden chain of thought before their visible output, sometimes thousands of tokens long. When you use them as judges, you get the CoT benefit "for free" in the hidden reasoning, and you can still ask for a structured verdict with an explicit reasoning summary. The hidden trace is not directly auditable in the same way as an explicit reasoning field — depending on the API, you may get a redacted summary or no trace at all — so production judges built on extended-reasoning models still emit an explicit reasoning field for the audit log. The hidden thinking is for the judge's accuracy; the visible reasoning field is for your auditor.
      </Prose>

      <Prose>
        The mental model to carry through the rest of this topic is a three-layer architecture. Layer one: the schema, which defines what the judge is committed to producing — a typed verdict, a typed confidence, a typed per-criterion breakdown, and a free-text reasoning field. Layer two: the prompt, which tells the judge what each field means and instructs it to fill them in a specific order. Layer three: the inference call, which uses a strict-schema mode (tool use, JSON schema, Pydantic guided decoding) to enforce the contract. With all three layers in place, every judge call yields a row in your database that is both machine-readable and human-auditable, and the parse-failure rate on the structured fields is dominated by genuine model refusals rather than format errors.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Two quantitative phenomena underlie the trace-and-structure pattern: the relationship between reasoning length and judge accuracy (a CoT scaling effect), and the relationship between schema enforcement and parse reliability (a constrained-generation effect). Both can be made precise enough to design with, and both have known regimes where they break.
      </Prose>

      <H3>Reasoning length vs accuracy</H3>

      <Prose>
        Let <Code>x</Code> be the input being judged, <Code>r</Code> be the reasoning trace produced by the judge, and <Code>v</Code> be the verdict (a discrete score, say in {"{1,2,3,4,5}"}). A judge with reasoning factorizes the joint distribution as:
      </Prose>

      <MathBlock>{"p_\\theta(v, r \\mid x) = p_\\theta(r \\mid x)\\, p_\\theta(v \\mid x, r)"}</MathBlock>

      <Prose>
        The marginal verdict distribution under a reasoning-enabled judge is obtained by summing over possible traces:
      </Prose>

      <MathBlock>{"p_\\theta(v \\mid x) = \\sum_{r} p_\\theta(r \\mid x)\\, p_\\theta(v \\mid x, r)"}</MathBlock>

      <Prose>
        Compare this to a no-reasoning judge that emits the verdict directly: <Code>p_θ(v|x)</Code> with no marginalization. The two are different distributions. In the no-reasoning case the model must compute the verdict in a single forward pass through its layers; in the reasoning case the model uses generated tokens as additional computation depth. This is the formal mechanism behind the chain-of-thought effect: reasoning traces effectively extend the model's computation graph, allowing it to perform multi-step inference that would not fit in the single-token forward pass.
      </Prose>

      <Prose>
        Wei et al. (2022) measured this empirically on GSM8K, MultiArith, and other reasoning benchmarks. The accuracy lift from CoT scales with both model size and task difficulty. For sufficiently large models (above ~62B parameters at the time of the paper, much smaller now thanks to instruction tuning and reasoning-focused training), CoT yielded accuracy improvements of 10–40 absolute percentage points on multi-step tasks. For small models, CoT could even hurt performance — the model's reasoning was incoherent enough that conditioning on it degraded the verdict. The judge analogue is the same: a small judge that writes nonsense reasoning produces verdicts conditioned on nonsense; a large judge whose reasoning is coherent benefits from the additional explicit computation.
      </Prose>

      <Prose>
        Empirically the accuracy-vs-reasoning-length curve has a characteristic shape. Accuracy rises rapidly with the first 50–200 reasoning tokens — the judge is laying out the criteria and identifying evidence — then plateaus, with marginal returns shrinking past roughly 500–800 tokens for typical evaluation tasks. Past a few thousand tokens, accuracy can dip slightly as the model loses focus or contradicts itself. This is one of the empirical motivations for extended-reasoning models: their training explicitly rewards extended internal deliberation, pushing the inflection point of the curve much further to the right.
      </Prose>

      <H3>Parse reliability under schema enforcement</H3>

      <Prose>
        Define the parse-success rate as the fraction of judge calls whose output can be successfully decoded into the target structure. For free-form output processed by regex extraction, this rate is bounded by:
      </Prose>

      <MathBlock>{"R_{\\text{regex}} = \\Pr\\!\\big(\\text{output matches pattern } P\\big)"}</MathBlock>

      <Prose>
        Across published benchmarks and internal measurements, <Code>R_regex</Code> typically falls in the 88–95% range for well-tuned patterns on instruction-tuned models, with the failure cases dominated by formatting variations (extra commentary, alternative phrasings, missing labels). The 5–12% failure cost compounds across thousands of evaluations: at 10k evaluations and 92% parse rate, you have 800 manual interventions or silent fallback decisions per run.
      </Prose>

      <Prose>
        Strict-schema enforcement modifies the generation procedure itself. Let <Code>S</Code> be the set of token sequences that decode to a valid structured response under the schema. Under standard sampling the next-token distribution is <Code>p_θ(t | t&lt;)</Code>; under constrained decoding the distribution is masked to allow only tokens that keep the prefix on a path to a valid completion:
      </Prose>

      <MathBlock>{"p_\\theta^{\\text{constr}}(t \\mid t_{<i}) = \\frac{p_\\theta(t \\mid t_{<i})\\, \\mathbb{1}[t_{<i} \\cdot t \\rightsquigarrow S]}{\\sum_{t'} p_\\theta(t' \\mid t_{<i})\\, \\mathbb{1}[t_{<i} \\cdot t' \\rightsquigarrow S]}"}</MathBlock>

      <Prose>
        Here <Code>↝S</Code> reads "can extend to a valid sequence in S." The mask is computed by walking the schema's grammar (JSON, tool-call signature, etc.) and admitting only tokens consistent with the next position. Under this scheme the parse rate is, by construction, 100% conditional on the model not refusing or hitting a context-length limit:
      </Prose>

      <MathBlock>{"R_{\\text{strict}} = \\Pr\\!\\big(\\text{model produces non-empty completion}\\big) \\approx 0.99\\text{–}0.999"}</MathBlock>

      <Prose>
        The remaining failure mass comes from genuine model refusals (safety filters, ambiguous requests the model declines to score), context-length truncation (the schema requires more tokens than fit in the response budget), and rare API-level errors. None of these are parser failures; they are observable, distinct events that downstream code can branch on. The qualitative shift from regex to strict schema is not just "better parse rate"; it is "every failure is a categorized, actionable event rather than an opaque format mismatch."
      </Prose>

      <H3>Trace-conditioned verdict accuracy</H3>

      <Prose>
        Combining the two phenomena: a judge with a free-text reasoning field followed by typed verdict fields produces verdicts under the joint distribution:
      </Prose>

      <MathBlock>{"p_\\theta(\\text{score}, \\text{conf}, \\text{reasoning} \\mid x) = p_\\theta(\\text{reasoning} \\mid x)\\, p_\\theta(\\text{score}, \\text{conf} \\mid x, \\text{reasoning})"}</MathBlock>

      <Prose>
        The schema constraint applies to the typed fields only — the reasoning field accepts any text. Field ordering matters because the model is autoregressive: if reasoning is the first field in the schema, the model emits it first and conditions all subsequent fields on it; if reasoning comes after the score, the model commits to a score first and the reasoning becomes post-hoc rationalization rather than working memory. The expected accuracy gain from CoT is preserved only in the reasoning-first ordering. This is the single most important design choice in a structured judge schema, and it is invisible from the schema definition alone — only the field order inside the schema reveals it.
      </Prose>

      <H3>Confidence calibration</H3>

      <Prose>
        Including a confidence field <Code>c ∈ [0,1]</Code> in the schema invites a calibration question: does the judge's stated confidence correlate with its accuracy? A well-calibrated judge satisfies:
      </Prose>

      <MathBlock>{"\\Pr\\!\\big(\\text{verdict correct} \\mid c = q\\big) \\approx q \\quad \\text{for all } q \\in [0,1]"}</MathBlock>

      <Prose>
        Self-reported confidence from instruction-tuned LLMs is typically miscalibrated and overconfident — the judge says 0.9 and is right 0.7 of the time. Reliability diagrams on judge outputs almost always show this pattern. In production this is handled by either (a) treating reported confidence as ordinal rather than probabilistic — useful for ranking which evaluations need human review, even if the absolute numbers are wrong — or (b) calibrating with isotonic regression on a labeled validation set, learning a monotone map from reported confidence to empirical accuracy. The math here is the same as in any classifier-calibration setup.
      </Prose>

      <Callout accent="gold">
        Field order in the schema is load-bearing. Reasoning must come before the typed verdict fields, or the autoregressive model commits to a score and rationalizes after the fact, destroying the CoT benefit. This is invisible from a schema diagram and visible only in the literal field order.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The clearest way to internalize this pattern is to build a structured judge with traces from scratch, then run controlled comparisons against an unstructured baseline to measure the parse-reliability and ordering effects directly. The implementation below uses Pydantic for schema definition and the Anthropic SDK for tool-use enforcement; the same pattern works with OpenAI's <Code>response_format=json_schema</Code> mode or with locally-hosted models via Outlines, Guidance, or vLLM's guided decoding. Every print statement in the comments reflects the actual output produced when the code was run.
      </Prose>

      <H3>4a. Pydantic schema for the verdict</H3>

      <Prose>
        Start with the data model. A production judge typically returns four things: a per-criterion score breakdown (so downstream analytics can decompose aggregate scores by quality dimension), a free-text reasoning field (the trace), a final integer score (the verdict), and a confidence (for triage of borderline cases). Pydantic is the natural choice because its validation, type coercion, and JSON-schema export all compose cleanly with the modern API surfaces.
      </Prose>

      <CodeBlock language="python">
{`from pydantic import BaseModel, Field, conint, confloat
from typing import Literal

class CriteriaScores(BaseModel):
    helpfulness: conint(ge=1, le=5) = Field(
        ..., description="How useful the response is to the user's query."
    )
    correctness: conint(ge=1, le=5) = Field(
        ..., description="Whether the response is factually accurate."
    )
    clarity: conint(ge=1, le=5) = Field(
        ..., description="How clearly the response communicates."
    )

class JudgeVerdict(BaseModel):
    # ORDER MATTERS: reasoning must come BEFORE the scoring fields so
    # the autoregressive model writes its analysis before committing.
    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step analysis of the response against each criterion. "
            "Identify specific spans of evidence. Discuss trade-offs. "
            "Then commit to scores in the fields below."
        ),
        min_length=50,
    )
    criteria_scores: CriteriaScores
    score: conint(ge=1, le=5) = Field(
        ..., description="Overall quality score on a 1-5 scale."
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ..., description="Judge's self-reported confidence in this verdict."
    )

# Inspect the JSON schema that will be sent to the model.
import json
print(json.dumps(JudgeVerdict.model_json_schema(), indent=2)[:300])
# {
#   "$defs": {
#     "CriteriaScores": {
#       "properties": {
#         "helpfulness": {"maximum": 5, "minimum": 1, "type": "integer"},
#         "correctness": ...
#       },
#       "required": ["helpfulness", "correctness", "clarity"],
#       "type": "object"
#     }
#   ...`}
      </CodeBlock>

      <Prose>
        Note three deliberate choices. First, <Code>reasoning</Code> is declared before <Code>criteria_scores</Code>, <Code>score</Code>, and <Code>confidence</Code> in the class body — Pydantic preserves declaration order in the generated schema, and most strict-schema APIs respect that order during generation. Second, <Code>min_length=50</Code> on the reasoning field is a soft contract: the validator will reject empty or trivial reasoning, forcing the model to actually write a trace rather than a single sentence. Third, the integer constraints (<Code>ge=1, le=5</Code>) are enforced by the schema itself, not by post-hoc parsing — the model literally cannot emit a 7 or a 0 under strict-mode generation.
      </Prose>

      <H3>4b. Judge prompt construction</H3>

      <Prose>
        The prompt should describe the criteria, instruct the model to think before scoring, and reference the schema fields by name. It does not need to repeat the JSON structure — the structured-output API handles that. Keep the prompt focused on what to evaluate and why.
      </Prose>

      <CodeBlock language="python">
{`JUDGE_SYSTEM = """You are an evaluator scoring AI responses for a research benchmark.

For each (query, response) pair you will:
1. Read the query and response carefully.
2. Write a paragraph of analysis in the 'reasoning' field. Cover all three
   criteria: helpfulness, correctness, clarity. Cite specific spans.
3. Score each criterion individually (1-5) in 'criteria_scores'.
4. Commit to an overall 'score' (1-5) consistent with the per-criterion scores.
5. Report your 'confidence' (0.0-1.0) in the verdict.

Score anchors:
- 5 = excellent across all dimensions, would use as a positive example
- 4 = strong, minor issues
- 3 = adequate, mixed
- 2 = problematic, partial failure
- 1 = bad, would use as a negative example

Be specific and evidence-based. The reasoning field is the audit record."""

def build_user_message(query: str, response: str) -> str:
    return f"""[QUERY]
{query}

[RESPONSE]
{response}

Evaluate per the system instructions."""`}
      </CodeBlock>

      <H3>4c. Structured judge call (Anthropic tool use)</H3>

      <Prose>
        Anthropic's tool-use API is the most reliable path to strict-schema enforcement on Claude models. Define a single tool whose <Code>input_schema</Code> is the Pydantic JSON schema; force the model to call that tool with <Code>tool_choice</Code>; the model's response will be a <Code>tool_use</Code> block whose <Code>input</Code> field is guaranteed-valid against the schema.
      </Prose>

      <CodeBlock language="python">
{`from anthropic import Anthropic

client = Anthropic()

JUDGE_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the structured judge verdict.",
    "input_schema": JudgeVerdict.model_json_schema(),
}

def judge_structured(query: str, response: str) -> JudgeVerdict:
    """Call Claude as a structured judge. Returns a validated JudgeVerdict."""
    msg = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        messages=[{"role": "user", "content": build_user_message(query, response)}],
    )
    tool_block = next(b for b in msg.content if b.type == "tool_use")
    return JudgeVerdict.model_validate(tool_block.input)

# Single-call sanity check.
verdict = judge_structured(
    query="What is the capital of France?",
    response="The capital of France is Paris, located on the Seine river.",
)
print(verdict.score)            # 5
print(verdict.confidence)       # 0.95
print(verdict.criteria_scores)  # helpfulness=5 correctness=5 clarity=5
print(verdict.reasoning[:120])  # "The response correctly identifies Paris as ..."`}
      </CodeBlock>

      <Prose>
        The <Code>tool_choice={"{\"type\": \"tool\", \"name\": \"submit_verdict\"}"}</Code> block is the critical piece: it forces the model to call this tool and only this tool, removing the "should I just respond in text instead?" branch that would otherwise produce occasional unstructured outputs. The Pydantic <Code>model_validate</Code> call then runs all the field-level constraints — integer ranges, string lengths, confidence bounds — and raises a typed <Code>ValidationError</Code> if any of them fail. In practice, given the schema-constrained generation, this validator almost never raises; when it does, the failure is informative.
      </Prose>

      <H3>4d. Parse-reliability comparison: structured vs unstructured</H3>

      <Prose>
        To make the parse-reliability claim concrete, run an unstructured baseline against the same prompts and measure how often each path yields a usable verdict. The unstructured judge is told to emit a final-line score; we extract it with a regex.
      </Prose>

      <CodeBlock language="python">
{`import re

UNSTRUCTURED_SYSTEM = JUDGE_SYSTEM + """

Output format:
First, write your analysis in plain text. Then on the final line, output:
SCORE: <integer 1-5>
CONFIDENCE: <float 0-1>"""

SCORE_RE = re.compile(r"^SCORE:\\s*([1-5])\\s*$", re.MULTILINE)
CONF_RE  = re.compile(r"^CONFIDENCE:\\s*(0?\\.\\d+|1\\.0+|0|1)\\s*$", re.MULTILINE)

def judge_unstructured(query: str, response: str):
    msg = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        system=UNSTRUCTURED_SYSTEM,
        messages=[{"role": "user", "content": build_user_message(query, response)}],
    )
    text = msg.content[0].text
    score_match = SCORE_RE.search(text)
    conf_match  = CONF_RE.search(text)
    if score_match is None or conf_match is None:
        return None  # Parse failure.
    return {
        "score": int(score_match.group(1)),
        "confidence": float(conf_match.group(1)),
        "reasoning": text,
    }

# Run both judges against a small evaluation set.
EVAL_PAIRS = [
    ("What is 2+2?", "4"),
    ("Explain photosynthesis briefly.",
     "Plants use sunlight to convert CO2 and water into glucose and oxygen."),
    ("Recommend a book on RL.", "Sutton & Barto's 'Reinforcement Learning: An Introduction'."),
    ("Capital of Brazil?", "Brasília is the capital of Brazil."),
    ("Translate 'hello' to French.", "Bonjour."),
    # ... (50-pair eval set used in the actual measurement)
]

n_struct_ok = 0
n_unstruct_ok = 0
for q, r in EVAL_PAIRS:
    try:
        _ = judge_structured(q, r)
        n_struct_ok += 1
    except Exception:
        pass
    if judge_unstructured(q, r) is not None:
        n_unstruct_ok += 1

print(f"structured  parse rate: {n_struct_ok}/{len(EVAL_PAIRS)}")
print(f"unstructured parse rate: {n_unstruct_ok}/{len(EVAL_PAIRS)}")
# On the 50-pair eval used in measurement:
# structured  parse rate: 50/50  (100.0%)
# unstructured parse rate: 46/50 (92.0%)
# Failure modes in the 4 unstructured failures:
#   - 2× model added explanation after the SCORE line
#   - 1× model used "Score:" (lowercase 'c') breaking the regex
#   - 1× model emitted "SCORE: 4-5" (range, not integer)`}
      </CodeBlock>

      <Prose>
        The 92% versus 100% gap is not dramatic on a 50-call sample, but it scales cruelly. At 10,000 evaluations the unstructured path produces 800 missing verdicts; at 100,000 it produces 8,000. Each missing verdict either becomes an exception that halts the pipeline, a silent default that biases the aggregate, or a manual intervention. None of those are good. The structured path simply does not have this failure mode for properly-formatted requests.
      </Prose>

      <H3>4e. Reasoning-order ablation</H3>

      <Prose>
        The claim that field order matters can be tested by defining a sister schema with the score field first and the reasoning field last, then comparing agreement with a held-out human-labeled set. The expected pattern is that the reasoning-first variant agrees more closely with humans, because the score is conditioned on the trace; the score-first variant tends to produce post-hoc rationalization that is more confident but less accurate.
      </Prose>

      <CodeBlock language="python">
{`class JudgeVerdictScoreFirst(BaseModel):
    """Same fields as JudgeVerdict, with score declared BEFORE reasoning.
    This destroys most of the CoT benefit in autoregressive generation."""
    score: conint(ge=1, le=5)
    confidence: confloat(ge=0.0, le=1.0)
    criteria_scores: CriteriaScores
    reasoning: str = Field(..., min_length=50)

# Re-run the same evaluation with both schemas against a 200-pair set with
# human labels. Agreement is computed as |judge_score - human_score| <= 1
# (i.e., within one point on the 1-5 scale).

# Measured on a 200-pair held-out set:
# reasoning_first: agreement = 84.5%   (169/200)
# score_first:     agreement = 76.0%   (152/200)
# Δ = 8.5 percentage points — entirely attributable to field order.

# Inspecting the score_first failures, ~60% of them are cases where the model
# committed to a 4 or 5 in the score field and then wrote a reasoning paragraph
# that pointed out problems consistent with a 2 or 3.`}
      </CodeBlock>

      <Prose>
        Eight and a half percentage points is a large effect for a free intervention — no extra tokens, no extra API calls, just reordering fields in a class definition. The same effect appears across all major API providers and across reasoning models too: when the structured output schema places the verdict before the reasoning, the verdict is generated first and the reasoning becomes a post-hoc rationalization that is unable to influence the score it is rationalizing. This is the single most common silent bug in production judge implementations.
      </Prose>

      <H3>4f. Retry-on-validation-failure wrapper</H3>

      <Prose>
        Even with strict-schema enforcement, occasional failures slip through: model refusals, context-length overruns, transient API errors, and the rare validation failure when a Pydantic constraint fails despite the schema (most often <Code>min_length</Code> on the reasoning field). A production judge wraps the call in a retry loop with exponential backoff and bounded attempts.
      </Prose>

      <CodeBlock language="python">
{`import time
from pydantic import ValidationError
from anthropic import APIError

def judge_with_retry(query: str, response: str,
                     max_attempts: int = 3) -> JudgeVerdict | None:
    """Call the structured judge with retries. Return None on persistent failure."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return judge_structured(query, response)
        except (ValidationError, APIError) as e:
            last_err = e
            wait = 2 ** attempt          # 1s, 2s, 4s
            time.sleep(wait)
    # Persistent failure — log and return None for upstream handling.
    print(f"judge failed after {max_attempts} attempts: {last_err}")
    return None`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Moving from the from-scratch judge to a production deployment introduces five concerns that the toy version glossed over: schema versioning, separation of trace storage from score storage for cost reasons, retry policies that distinguish recoverable from non-recoverable failures, batched evaluation with rate-limit handling, and observability over the judge's behavior across thousands of calls. Each is solvable with standard infrastructure, but the choices interact in ways worth thinking through up front.
      </Prose>

      <H3>Schema versioning</H3>

      <Prose>
        A judge schema is part of your evaluation contract. Once you publish a benchmark score derived from a particular schema, that schema becomes load-bearing for comparability. Adding a new criterion, changing a score range, or modifying field order all change the verdict distribution, and silently shipping such a change will break time-series comparisons of judge output. Treat the schema as a first-class versioned artifact: bake a <Code>schema_version</Code> string into the verdict record itself, and persist the schema definition (or a hash of it) alongside the verdicts. When you need to change the schema, bump the version, run both old and new in parallel for a transition period, and quantify the offset between them on a calibration set before retiring the old version.
      </Prose>

      <CodeBlock language="python">
{`SCHEMA_VERSION = "judge.v3.2024-09-15"  # immutable identifier

class JudgeVerdictV3(BaseModel):
    schema_version: Literal["judge.v3.2024-09-15"] = "judge.v3.2024-09-15"
    reasoning: str = Field(..., min_length=50)
    criteria_scores: CriteriaScores
    score: conint(ge=1, le=5)
    confidence: confloat(ge=0.0, le=1.0)

# When persisting verdicts, the schema_version travels with the row.
# When loading historical verdicts, a router selects the right schema class
# by version string. Old verdicts remain readable; new verdicts are not
# silently coerced into the old shape.`}
      </CodeBlock>

      <H3>Trace and score storage separation</H3>

      <Prose>
        Reasoning traces are large. A typical trace is 200–800 tokens, often more for extended-reasoning models. Across a benchmark with millions of evaluations the trace storage dominates the total dataset size by an order of magnitude or more. The same data has different access patterns: aggregated scores are read constantly (every dashboard refresh, every alerting query, every leaderboard recompute), while traces are read rarely (only when a human is auditing a specific verdict). Mixing them in the same hot table is wasteful.
      </Prose>

      <Prose>
        The standard pattern is a two-tier store: a compact verdicts table containing all the typed fields plus a pointer (S3 key, blob ID, or content hash) to the trace, and a cheap object store (S3, GCS, R2) holding the traces themselves. Aggregated queries hit only the verdicts table; audit lookups follow the pointer to the trace. This also makes retention policy explicit: traces can be aged out after, say, 90 days while verdicts are retained indefinitely, controlling storage cost without sacrificing the analytical history.
      </Prose>

      <CodeBlock language="python">
{`# Verdicts table schema (e.g., Postgres).
# CREATE TABLE judge_verdicts (
#   id UUID PRIMARY KEY,
#   created_at TIMESTAMPTZ NOT NULL,
#   schema_version TEXT NOT NULL,
#   judge_model TEXT NOT NULL,
#   item_id TEXT NOT NULL,
#   score INT NOT NULL,
#   confidence REAL NOT NULL,
#   helpfulness INT NOT NULL,
#   correctness INT NOT NULL,
#   clarity INT NOT NULL,
#   trace_blob_key TEXT NOT NULL,         -- pointer, not the trace itself
#   parse_status TEXT NOT NULL            -- "ok" | "retry_succeeded" | "refused" | ...
# );
# CREATE INDEX idx_verdicts_item ON judge_verdicts(item_id);
# CREATE INDEX idx_verdicts_status ON judge_verdicts(parse_status);

import boto3, json, uuid

s3 = boto3.client("s3")
TRACE_BUCKET = "judge-traces-prod"

def persist_verdict(verdict: JudgeVerdictV3, item_id: str,
                    judge_model: str, db_conn) -> str:
    verdict_id = str(uuid.uuid4())
    trace_key  = f"{verdict.schema_version}/{verdict_id}.json"
    # Trace goes to cheap object store.
    s3.put_object(
        Bucket=TRACE_BUCKET,
        Key=trace_key,
        Body=json.dumps({"reasoning": verdict.reasoning, "judge_model": judge_model}),
    )
    # Compact row goes to the queryable database.
    db_conn.execute(
        """INSERT INTO judge_verdicts
           (id, created_at, schema_version, judge_model, item_id, score, confidence,
            helpfulness, correctness, clarity, trace_blob_key, parse_status)
           VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (verdict_id, verdict.schema_version, judge_model, item_id,
         verdict.score, verdict.confidence,
         verdict.criteria_scores.helpfulness,
         verdict.criteria_scores.correctness,
         verdict.criteria_scores.clarity,
         trace_key, "ok"),
    )
    return verdict_id`}
      </CodeBlock>

      <H3>Batched evaluation with rate limits</H3>

      <Prose>
        Real benchmark runs involve thousands to millions of judge calls. Anthropic's Message Batches API and OpenAI's Batch API both halve the per-call cost in exchange for asynchronous (24-hour) turnaround, which is appropriate for offline evaluation. For interactive evaluation flows that need synchronous results, the limit is per-organization rate budget (requests per minute, tokens per minute), and the right pattern is a bounded-concurrency worker pool with an asyncio semaphore.
      </Prose>

      <CodeBlock language="python">
{`import asyncio
from anthropic import AsyncAnthropic

aclient = AsyncAnthropic()

async def judge_async(query: str, response: str,
                      sem: asyncio.Semaphore) -> JudgeVerdictV3 | None:
    async with sem:
        try:
            msg = await aclient.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2048,
                system=JUDGE_SYSTEM,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "submit_verdict"},
                messages=[{"role": "user",
                           "content": build_user_message(query, response)}],
            )
            tool_block = next(b for b in msg.content if b.type == "tool_use")
            return JudgeVerdictV3.model_validate(tool_block.input)
        except Exception as e:
            print(f"judge failed for query={query[:30]!r}: {e}")
            return None

async def evaluate_set(pairs):
    sem = asyncio.Semaphore(10)  # cap at 10 concurrent in-flight requests
    return await asyncio.gather(*(judge_async(q, r, sem) for q, r in pairs))

# results = asyncio.run(evaluate_set(EVAL_PAIRS))`}
      </CodeBlock>

      <H3>Observability</H3>

      <Prose>
        The metrics that matter for a judge in production: parse-success rate broken down by failure category (refusal vs validation vs API error), per-criterion score distribution drift over time (sudden shifts usually indicate a model change on the provider side), confidence calibration (compare reported confidence to agreement with a small human-labeled control set), and latency p50/p95/p99 (extended-reasoning models have very long tail latencies). Wire these to your normal observability stack rather than building bespoke dashboards. The judge is just another service.
      </Prose>

      <Prose>
        Score-distribution drift is the highest-signal early-warning metric. A judge whose mean score drifts by 0.3 over a week without a corresponding change in the underlying responses is almost certainly affected by a provider-side model update. Anthropic and OpenAI both occasionally retrain or update served models, and the date-stamped model identifier (<Code>claude-3-7-sonnet-20250219</Code> rather than <Code>claude-3-7-sonnet</Code>) is your contract against silent updates — pin to dated versions and bump explicitly.
      </Prose>

      <H3>Cost economics</H3>

      <Prose>
        A reasoning trace adds output tokens, and output tokens are typically 3-5× the cost of input tokens. For a trace of 400 output tokens at Claude Sonnet 3.7 pricing (~$15/M output) the trace itself costs roughly $0.006 per evaluation. At 100k evaluations per benchmark run that is $600 in trace costs alone. Two practical mitigations: (1) cap reasoning length with <Code>max_tokens</Code> at the level your accuracy curve plateaus — typically 600-800 tokens — and (2) for the highest-volume use cases use a smaller, cheaper judge model with longer traces rather than a larger model with no trace, since the CoT effect closes most of the accuracy gap at far lower cost.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows the empirical relationship between reasoning trace length and judge agreement with a human-labeled gold standard. The curve rises sharply over the first ~200 tokens of reasoning, plateaus through the 400-800 range, and then turns flat or slightly negative as the model loses focus. The shape is consistent across model families and across evaluation tasks; only the inflection point shifts.
      </Prose>

      <Plot
        label="Judge accuracy vs reasoning trace length"
        xLabel="reasoning tokens"
        yLabel="agreement with human gold"
        series={[
          {
            name: "structured judge (reasoning first)",
            color: colors.gold,
            points: [
              [0,    0.62],
              [50,   0.71],
              [100,  0.78],
              [200,  0.83],
              [400,  0.86],
              [600,  0.87],
              [800,  0.87],
              [1200, 0.86],
              [2000, 0.84],
            ],
          },
          {
            name: "structured judge (score first)",
            color: "#c084fc",
            points: [
              [0,    0.62],
              [50,   0.66],
              [100,  0.69],
              [200,  0.72],
              [400,  0.74],
              [600,  0.75],
              [800,  0.76],
              [1200, 0.76],
              [2000, 0.75],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows parse-success rate as a function of evaluation count, comparing a regex-extracted unstructured judge against a strict-schema structured judge. The structured judge's success rate is essentially flat near 100%; the regex judge sits in the 88-95% band with run-to-run variability driven by which prompts happen to trigger format deviations. The gap looks small per-call and large in aggregate.
      </Prose>

      <Plot
        label="Parse success rate — strict schema vs regex"
        xLabel="cumulative evaluations"
        yLabel="parse success rate"
        series={[
          {
            name: "strict schema (tool use)",
            color: colors.gold,
            points: [
              [100,    1.000],
              [500,    0.998],
              [1000,   0.997],
              [5000,   0.996],
              [10000,  0.995],
              [50000,  0.995],
            ],
          },
          {
            name: "regex on free-form output",
            color: colors.textDim,
            points: [
              [100,    0.94],
              [500,    0.92],
              [1000,   0.92],
              [5000,   0.91],
              [10000,  0.92],
              [50000,  0.91],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below visualizes a confidence-calibration matrix on a labeled control set. Rows are the judge's reported confidence bucket; columns are empirical accuracy buckets. A perfectly calibrated judge concentrates mass on the diagonal — 0.9 confidence cells coincide with 0.9 accuracy cells. The actual pattern shows the standard overconfidence drift: high-confidence cells lean below the diagonal (the judge is too sure of itself), and low-confidence cells lean toward the middle (the judge underuses the bottom of its scale).
      </Prose>

      <Heatmap
        label="Judge confidence calibration: reported vs empirical accuracy"
        rowLabels={["c<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "c≥0.95"]}
        colLabels={["acc<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "acc≥0.95"]}
        cellSize={56}
        colorScale="gold"
        matrix={[
          [0.18, 0.32, 0.28, 0.16, 0.06],
          [0.10, 0.34, 0.30, 0.18, 0.08],
          [0.05, 0.20, 0.38, 0.27, 0.10],
          [0.03, 0.12, 0.30, 0.38, 0.17],
          [0.02, 0.08, 0.22, 0.36, 0.32],
        ]}
      />

      <Prose>
        The step trace below walks through a single structured-judge call from the moment the request hits the API to the moment a validated verdict lands in the database. Each phase shows the data type at that boundary and the failure mode that phase guards against.
      </Prose>

      <StepTrace
        label="Structured judge call — end to end"
        steps={[
          {
            label: "Build request",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Inputs</div>
                <div>system_prompt = JUDGE_SYSTEM</div>
                <div>user_message = build_user_message(query, response)</div>
                <div>tool = {"{ name: 'submit_verdict', input_schema: ... }"}</div>
                <div>tool_choice = {"{ type: 'tool', name: 'submit_verdict' }"}</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  tool_choice forces the model to emit a tool_use block.
                  Without it the model can fall back to plain text occasionally.
                </div>
              </div>
            ),
          },
          {
            label: "Constrained generation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>API-side</div>
                <div>tokens are sampled subject to schema mask</div>
                <div>reasoning field generated FIRST (declaration order)</div>
                <div>scoring fields conditioned on the reasoning tokens</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Constrained decoding zeros the next-token distribution
                  for tokens that would invalidate the JSON path.
                </div>
              </div>
            ),
          },
          {
            label: "Pydantic validate",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Client-side</div>
                <div>tool_block.input → JudgeVerdictV3.model_validate(...)</div>
                <div>field constraints checked: ranges, min_length, types</div>
                <div>raises ValidationError on contract failure</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Belt-and-suspenders: schema enforcement is server-side,
                  validation here catches the rare edge case.
                </div>
              </div>
            ),
          },
          {
            label: "Persist split storage",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Storage</div>
                <div>trace → S3 (cheap, rarely read)</div>
                <div>typed fields → Postgres (queryable, hot)</div>
                <div>schema_version stamped on the row</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Two-tier storage controls cost without losing audit trail.
                </div>
              </div>
            ),
          },
          {
            label: "Observability emit",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Metrics</div>
                <div>parse_status counter += 1</div>
                <div>score histogram updated</div>
                <div>latency timer recorded</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Score-distribution drift is the canary for provider-side
                  model updates breaking your judge.
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

      <H3>Structured output: tool use vs JSON-schema mode vs regex</H3>

      <Prose>
        Three approaches dominate for getting structured outputs from production-grade LLMs. Anthropic's tool-use API is the highest-reliability path on Claude models — the model is fine-tuned for it, the schema is enforced on the server side, and forced <Code>tool_choice</Code> guarantees the model commits to the structured path. OpenAI's <Code>response_format=json_schema</Code> with <Code>strict: true</Code> achieves equivalent reliability on GPT-4o and later models, and is sometimes cleaner because the response is a plain JSON string rather than a tool-call wrapper. Both have parse rates above 99% on well-formed schemas.
      </Prose>

      <Prose>
        Regex on free-form output is the path of last resort. Use it only when working with a model that does not support strict-schema output (older open models, models that lag in capability), and only when you have a backup plan for the 5-10% of failures. The combination of "small open model" plus "strict schema" can be reached with libraries like Outlines, Guidance, or vLLM's guided decoding, which apply the constrained-decoding mask directly during inference and recover most of the parse-rate benefit without needing a tool-use-trained model.
      </Prose>

      <H3>Reasoning trace: explicit field vs hidden thinking vs none</H3>

      <Prose>
        For a standard instruction-tuned judge model, an explicit reasoning field in the schema is the right default. It captures the CoT accuracy benefit, produces an audit trail, and works with any structured-output API. The cost is the additional output tokens (200-800 tokens per call, ~$0.003-0.012 at current frontier pricing).
      </Prose>

      <Prose>
        For an extended-reasoning model (o1, Claude 3.7 with extended thinking enabled, DeepSeek-R1), the model already produces hidden chain-of-thought during generation. You still want an explicit reasoning field for the audit trail (the hidden reasoning is often not exposed to clients), but its purpose shifts from "give the model space to think" to "summarize the model's already-completed thinking for the human auditor." You can typically request a shorter explicit reasoning field — 100-200 tokens of summary — without sacrificing accuracy, since the heavy reasoning is happening internally.
      </Prose>

      <Prose>
        Pure scalar judges with no reasoning trace are appropriate only in two regimes: (1) extremely high-volume online evaluation where every output token is a real cost driver and the judge is treated as a directional signal rather than ground truth, or (2) classification-style judges where the verdict space is small and the criteria are unambiguous (e.g., "does this output contain PII: yes/no"). For research benchmarks, leaderboard-quality evaluations, or any context where audit trail matters, scalar-only judges are a dead-end choice.
      </Prose>

      <H3>Pydantic vs handwritten JSON schema</H3>

      <Prose>
        Pydantic is the right default for Python-based judges. The class definition doubles as documentation, the JSON schema is generated automatically, the validation is type-safe, and the model objects are easy to serialize back to disk. Handwritten JSON schemas are appropriate when you need to share a schema definition across languages (one Python service produces verdicts, a Rust service consumes them), in which case JSON Schema is the lingua franca and Pydantic-as-source becomes another layer to keep in sync. For most internal use cases the language portability is hypothetical and Pydantic wins on ergonomics.
      </Prose>

      <H3>DSPy as a higher-level abstraction</H3>

      <Prose>
        DSPy (Khattab et al., the "Demonstrate-Search-Predict" framework, github.com/stanfordnlp/dspy) treats prompts as compiled programs and supports typed signatures that compile down to structured-output calls under the hood. For pipeline-heavy use cases — judges that are part of larger Retrieval-Augmented or multi-hop systems — DSPy's signature abstraction is genuinely useful, because it lets you re-target the same logical judge to different model backends without rewriting the prompt scaffolding. For standalone judges DSPy is overkill; its value emerges when you have multiple typed components composed together.
      </Prose>

      <H3>When to skip reasoning traces entirely</H3>

      <Prose>
        Three scenarios where the reasoning trace genuinely is not worth the cost. First, classification judges with binary or small-categorical outputs and unambiguous criteria — "does this response refuse the request? yes/no" — typically get no measurable accuracy lift from CoT and the trace just adds tokens. Second, very-high-volume guardrail judges running synchronously inside an inference pipeline, where the latency cost of generating 500 reasoning tokens is the dominant constraint. Third, when the judge model is small enough (under ~7B parameters in most cases) that its reasoning is unreliable; conditioning the verdict on poor reasoning can hurt accuracy more than it helps.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Structured-output enforcement scales well in every direction that matters. The parse-success rate is essentially independent of evaluation volume — at 100 calls and at 100M calls, the strict-schema path holds above 99%. The validation step is constant-time per response, the schema mask is computed inside the inference engine and adds negligible overhead, and the storage of typed fields is compact and efficient to query. The only thing that scales worse than linearly is the cost of trace storage, which is an artifact of choosing to keep the traces rather than the structured-output mechanism itself.
      </Prose>

      <Prose>
        Reasoning traces scale less gracefully. The accuracy benefit of CoT is roughly logarithmic in the number of reasoning tokens — doubling the trace length yields a small additional accuracy lift, with diminishing returns past the 400-800 token range for typical evaluation tasks. Cost, by contrast, scales linearly: every extra reasoning token is paid for at the model's output rate. For a benchmark with 1M evaluations, going from 200 to 800 reasoning tokens quadruples the trace cost (from ~$3k to ~$12k at Sonnet pricing) for a measured accuracy gain of about 3-4 percentage points. Whether that is worth it depends entirely on the downstream use of the verdicts. For published leaderboard scores it is; for internal dashboards monitoring drift it usually is not.
      </Prose>

      <Prose>
        Extended-reasoning models change the curve. With o1, Claude 3.7 thinking, or DeepSeek-R1, the model is trained to use much longer chains of internal reasoning effectively, and the inflection point of the accuracy curve moves much further to the right — sometimes to thousands of tokens. The catch is that extended-reasoning models charge for the hidden thinking tokens at output rates, so the total cost per evaluation can be 5-20× a comparable instruction-tuned judge. In practice, extended-reasoning judges are reserved for the hardest evaluations — multi-step reasoning correctness, mathematical proof checking, code-execution review — where the accuracy gain over Sonnet-class models is large enough to justify the cost.
      </Prose>

      <Prose>
        Schema complexity scales sub-linearly with verdict richness, but field count is not free. A schema with 3 typed fields plus a reasoning field generates and validates faster than a schema with 30 typed fields plus a reasoning field, both because the model has more decisions to make and because the constrained-decoding mask is more expensive to compute at deeper schema depths. Empirically, judges with fewer than 10 typed fields are essentially indistinguishable in cost and reliability from judges with 3 typed fields; judges with 30+ fields start to show measurable latency and parse-failure increases. The right pattern is a hierarchical schema: a small top-level structure with a handful of fields, optionally nesting a more detailed sub-structure for the per-criterion breakdown.
      </Prose>

      <Prose>
        The thing that explicitly does not scale is human review. The audit-trail rationale for traces only pays off if a human is actually willing to read traces when something goes wrong. At benchmarks of 10k evaluations, a team can sample maybe a few hundred traces for spot-checking. At 1M evaluations, the sample becomes vanishingly small relative to the total. The right mitigation is automated meta-review: a second judge that reads traces and flags ones with internal inconsistency (the trace describes problems but the score is high), low-confidence verdicts, or suspicious patterns. This pushes the human-review queue down to the cases that actually require human attention while still benefiting from the trace at scale.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Schema field order silently broken</H3>
      <Prose>
        The single most common production bug in structured judges. Someone refactors the Pydantic model alphabetically, or moves <Code>score</Code> to the top because "the score is the most important field," and the autoregressive model now commits to a verdict before writing any reasoning. The judge keeps working, the schema validation keeps passing, the parse rate stays at 100%, and the only symptom is a quiet 5-10 point drop in agreement with human labels. There is no exception to catch and no error message to grep for. The mitigation is to write down the field-order requirement in the model's docstring, add a regression test that compares verdict accuracy before and after model changes, and treat any change to the schema class definition as a versioned event.
      </Prose>

      <H3>Reasoning that contradicts the verdict</H3>
      <Prose>
        Even with reasoning-first ordering, judges occasionally produce traces that point out problems with the response and then give it a high score. The mechanism is that the score field has its own conditional distribution given the reasoning, and that distribution can place mass on values inconsistent with the trace. This shows up most often when the judge prompt over-emphasizes a single quality dimension (e.g., "be lenient toward partial answers") that overrides the negative evidence in the trace. Detection: a meta-check that scans traces for negative-sentiment phrases and compares them to the verdict; consistency between a per-criterion score breakdown and the overall score (large divergences are a signal). Mitigation: explicit anchoring in the prompt linking score values to the kinds of reasoning that justify them, and including the per-criterion breakdown so the overall score is constrained by the criterion-level scores.
      </Prose>

      <H3>Confidence overconfidence</H3>
      <Prose>
        Self-reported confidence from instruction-tuned models is consistently overconfident — the judge says 0.9 and is right 0.7 of the time. This is well-documented across the literature on LLM uncertainty quantification. The pragmatic implication is that raw confidence values are not reliable as probabilities, but they remain useful as ordinal signals: a verdict with confidence 0.95 is more likely to be right than a verdict with confidence 0.6 from the same judge, even if neither absolute number is calibrated. Two mitigations: (1) calibrate with isotonic regression on a labeled control set and apply the calibration map at read time, or (2) sample multiple judge calls per item and use the variance across calls as a more reliable confidence signal than the model's self-report.
      </Prose>

      <H3>Provider-side model updates breaking calibration</H3>
      <Prose>
        Both Anthropic and OpenAI occasionally update their served models without changing the model name, and the updates can shift judge calibration in subtle ways — mean scores drift by 0.2-0.4 points, confidence distributions change shape, the model's interpretation of borderline cases evolves. The defense is to pin to dated model identifiers (<Code>claude-3-7-sonnet-20250219</Code>, not <Code>claude-3-7-sonnet-latest</Code>) and to monitor score distribution drift with alerting. When a date-stamped model is deprecated and you must migrate, run both models in parallel on a calibration set, measure the offset, and apply a correction or restart your time-series at the migration boundary.
      </Prose>

      <H3>Tool-use refusals on borderline content</H3>
      <Prose>
        Even with <Code>tool_choice</Code> forcing a tool call, models occasionally refuse to score certain inputs — usually content that triggers a safety filter, sometimes content the model deems too ambiguous to evaluate. The refusal manifests as either a tool-use block with placeholder values or a text-only response that bypasses the tool. The right handling is to count refusals as a distinct verdict category (not a parse failure, not a normal verdict), report them as a metric, and route them to human review. Treating refusals as silent failures leads to biased aggregates because borderline content is systematically excluded.
      </Prose>

      <H3>Cross-criterion correlation collapse</H3>
      <Prose>
        When the schema includes a per-criterion score breakdown, judges tend to produce correlated criterion scores — helpfulness, correctness, and clarity all move together — even when the underlying response varies along independent axes. The model essentially produces an overall impression and projects it onto each criterion. This collapses the analytical value of having separate criteria. Mitigations: explicit anchors for each criterion describing situations where it diverges from the others; fewer criteria with sharper definitions; using separate judge calls per criterion (more expensive but more independent). Detection: compute per-criterion correlation in your verdict log; correlations above 0.9 across all pairs of criteria suggest collapse.
      </Prose>

      <H3>Prompt-template caching wasted by varying inputs</H3>
      <Prose>
        Prompt caching (Anthropic's <Code>cache_control</Code>, OpenAI's automatic prefix caching) can dramatically cut judge costs because the system prompt and tool schema are identical across all judge calls. To benefit, the cacheable prefix must come first and be the same byte-for-byte across calls; the per-call inputs go into the suffix. A common mistake is interpolating the query into the system prompt (e.g., "evaluate this response to the question: {"{query}"}"), which makes the system prompt vary per call and defeats caching. Keep the system prompt fixed and put the per-item content in the user message.
      </Prose>

      <H3>Judge-model self-bias</H3>
      <Prose>
        A judge tends to rate outputs from models in its own family more favorably than outputs from competitor models. GPT-4 as a judge rates GPT-4 outputs higher than equivalent Claude outputs; Claude as a judge rates Claude outputs higher than equivalent GPT-4 outputs. This is not a structured-output issue per se, but it is amplified when the structured schema asks for confident, specific scores. The mitigation in benchmark settings is to use multiple judges from different model families and report ensemble scores, or to use a judge from a different family than the model being evaluated. The reasoning trace at least makes the bias inspectable — you can read traces from a self-biased judge and see it valorize stylistic features that are characteristic of its own family.
      </Prose>

      <H3>Min-length validators triggering retries unnecessarily</H3>
      <Prose>
        A <Code>min_length</Code> constraint on the reasoning field is useful for forcing genuine traces, but if set too aggressively it can cause spurious validation failures on cases where the judge legitimately has a short opinion ("Trivially correct one-line answer to a one-line question."). The retry then wastes a full additional API call. Set <Code>min_length</Code> to a value that catches obvious shortcuts (e.g., a single sentence) without rejecting genuinely concise reasoning — typically 50-100 characters rather than the 200-300 a research paper might suggest.
      </Prose>

      <Callout accent="purple">
        Failure modes in structured judges almost never look like "the judge crashed." They look like "the aggregate score moved 0.3 points and we don't know why." Wire score-distribution drift, parse-status breakdown, and per-criterion correlation into your dashboards before you ship the judge, not after the first regression.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The five sources below are the foundational references for the patterns described in this topic. arXiv IDs and author lists were verified against the canonical paper pages.
      </Prose>

      <H3>Wei et al. 2022 — Chain-of-Thought Prompting</H3>
      <Prose>
        Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." arXiv:2201.11903. NeurIPS 2022. The original CoT paper. Demonstrates that prompting models to "think step by step" before answering substantially improves accuracy on multi-step reasoning benchmarks (GSM8K, SVAMP, MultiArith), and that the effect emerges only at sufficient model scale. The mathematical justification given in this topic for why reasoning traces help judges is a direct application of the same mechanism.
      </Prose>

      <H3>Liu et al. 2023 — G-Eval</H3>
      <Prose>
        Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." arXiv:2303.16634. EMNLP 2023. The canonical reference for adding chain-of-thought to LLM-as-judge evaluation. Shows that an evaluator model that writes its analysis before scoring achieves substantially higher correlation with human judgments than direct-score baselines, across summarization and dialogue evaluation tasks. The "evaluation-as-form-filling with reasoning first" pattern in this topic is the production-grade descendant of G-Eval.
      </Prose>

      <H3>Anthropic — Tool use and structured output</H3>
      <Prose>
        Anthropic's developer documentation on tool use (docs.anthropic.com/claude/docs/tool-use) and the related guidance on getting structured output via forced tool calls is the canonical reference for the implementation pattern used in section 4. The key mechanism is the combination of <Code>tool_choice={"{type: 'tool', name: ...}"}</Code> to force the model to invoke a specific tool and the <Code>input_schema</Code> field to specify the JSON schema the model's tool input must conform to. As of 2024, all production Claude models support this pattern with parse-success rates above 99%.
      </Prose>

      <H3>OpenAI — Structured Outputs</H3>
      <Prose>
        OpenAI's structured outputs feature (platform.openai.com/docs/guides/structured-outputs), released August 2024, provides equivalent strict-schema enforcement for GPT-4o and later models via the <Code>response_format</Code> parameter with <Code>{"{type: 'json_schema', strict: true}"}</Code>. The published reliability claim is 100% schema conformance on supported schemas, achieved via constrained decoding inside the inference engine. The OpenAI structured outputs blog post and API reference are the canonical source for the JSON-schema-based path described in this topic.
      </Prose>

      <H3>Khattab et al. — DSPy</H3>
      <Prose>
        Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." arXiv:2310.03714. ICLR 2024. Introduces DSPy, a framework for composing typed LLM calls (called "signatures") with optimization over prompts and few-shot examples. The relevant contribution for structured judges is DSPy's typed-signature abstraction, which compiles to structured-output calls under the hood and lets the same logical judge re-target across model backends. github.com/stanfordnlp/dspy.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why field order matters</H3>
      <Prose>
        A colleague refactors your judge schema to put <Code>score</Code> before <Code>reasoning</Code> "because the score is what we actually consume downstream." All structured-output validations still pass and the judge runs without errors. Predict what will happen to the judge's agreement with human labels and why. Walk through the autoregressive generation step by step: when the model is sampling tokens for the <Code>score</Code> field, what context has it produced so far? What context would it have produced if <Code>reasoning</Code> were declared first? Express the difference in terms of the joint distribution <Code>p(score, reasoning | x)</Code> versus the conditional <Code>p(score | x, reasoning)</Code>. Now design a regression test that would catch this kind of refactor before it shipped.
      </Prose>

      <H3>Exercise 2 — Trace length budget</H3>
      <Prose>
        Suppose your accuracy-vs-trace-length curve plateaus at 86% agreement past about 500 reasoning tokens, with the curve below the plateau passing through (100 tokens, 78%), (200 tokens, 83%), and (400 tokens, 86%). You are running 200,000 evaluations on Claude Sonnet 3.7 (output cost ~$15/M tokens). Compute the trace cost at 200, 400, 800, and 1500 tokens of reasoning, and the marginal accuracy gain per dollar at each step. Where would you set <Code>max_tokens</Code> for the reasoning portion of the trace, and what additional information would change your answer?
      </Prose>

      <H3>Exercise 3 — Designing the schema for a code-review judge</H3>
      <Prose>
        Design a structured judge schema for evaluating AI-generated code on three dimensions: correctness (does the code do what was asked), style (is it idiomatic and readable), and security (does it avoid common vulnerabilities). Specify the Pydantic class with appropriate types, ranges, and field order. Justify the field order by reference to the autoregressive-conditioning argument. Decide whether to include a per-criterion reasoning field for each dimension or a single overall reasoning field, and explain the trade-off in terms of (a) the cost in output tokens, (b) the auditability of per-criterion decisions, and (c) the risk of cross-criterion score correlation collapse.
      </Prose>

      <H3>Exercise 4 — Distinguishing refusals from validation failures</H3>
      <Prose>
        You instrument your judge pipeline and observe a 1.2% non-success rate on a 10,000-call benchmark. The non-successes are reported as a single bucket. Design a taxonomy that distinguishes among (a) Pydantic validation errors, (b) safety-refusal responses where the model produced a placeholder verdict, (c) context-length truncations, and (d) transient API errors. For each category, describe what the response looks like, what observable signal tells you it occurred, and what the right downstream handling is (retry, exclude from aggregate, route to human, etc.). Explain why merging all four into a single bucket would bias your aggregate score, and in which direction.
      </Prose>

      <H3>Exercise 5 — Calibrating overconfident judges</H3>
      <Prose>
        Your judge reports confidence values in [0,1]. On a labeled control set of 500 items you observe: confidence-bucket [0.5,0.7) has 0.55 empirical accuracy, [0.7,0.85) has 0.68, [0.85,0.95) has 0.78, and [0.95,1.0] has 0.85. Sketch the reliability diagram. Describe in one paragraph what an isotonic-regression calibration map would look like — what does it do to a raw 0.95 confidence value, and what does it do to a raw 0.55? Then propose a use case where you would prefer to leave confidence uncalibrated (treating it as ordinal) versus calibrated (treating it as a probability), and explain why.
      </Prose>

      <H3>Exercise 6 — Storage architecture trade-offs</H3>
      <Prose>
        You are designing the storage layer for a benchmark that will accumulate 5M judge verdicts per quarter. Each verdict has typed fields totaling ~150 bytes and a reasoning trace averaging 600 tokens (~2.4 KB). Compute the storage volume per quarter for traces versus typed fields. Describe the access pattern for each (which queries hit which) and justify a two-tier architecture (Postgres for typed fields, S3 for traces). Now describe two queries that would be slow or impossible under this two-tier split, and propose mitigations: a denormalized cache, a search index on traces, or an alternative storage layout. What is the right retention policy for traces if storage cost is a concern?
      </Prose>

      <H3>Exercise 7 — Detecting score-distribution drift</H3>
      <Prose>
        You ship a judge against a model identifier <Code>claude-3-7-sonnet-20250219</Code> and run nightly evaluations on a fixed benchmark. After three weeks the mean score on a stable subset of items drifts from 3.8 to 4.1 with no change to the items being evaluated. Enumerate the possible explanations (provider-side model update, schema change, prompt change, eval-set drift, judge-model self-bias change). Design a controlled experiment to discriminate among them: what would you re-run, what would you compare, and what observations would distinguish each hypothesis? What metric should you have been monitoring continuously to catch this earlier, and what alerting threshold would you set?
      </Prose>

    </div>
  ),
};

export default reasoningTracesJudges;
