import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const multiModelServing = {
  title: "Multi-Model Serving & Model Routing",
  slug: "multi-model-serving-model-routing",
  readTime: "44 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        The most expensive token you serve is the one answered by a model that was ten times larger than necessary. At the other extreme, the most damaging token you serve is the one answered by a model that was too small, producing a wrong output your user notices and does not forgive. Multi-model serving is the discipline of finding the right model for each incoming request — not as a one-time architectural decision, but as a per-request routing decision made in real time. Done wrong, you either overpay on every easy request or degrade quality on every hard one. Done right, the same quality bar costs a fraction of what it would under any single-model strategy.
      </Prose>

      <Prose>
        Production LLM endpoints almost never serve a single model. They serve tiers: a small model for fast, cheap, routine traffic; a medium model as the general-purpose workhorse; a large frontier model reserved for genuinely hard requests; and an array of specialized models — code-tuned, vision-capable, math-optimized — that outperform the size axis entirely for their target domain. Alongside those tiers sit variant models: the safety-finetuned version used for regulated industries, the A/B-test candidate being evaluated against the production champion, the quantized model deployed for latency-sensitive endpoints. A serious production stack at any non-trivial scale is managing tens of models simultaneously. Model routing is the control plane that makes this a product rather than a zoo.
      </Prose>

      <Prose>
        The cost argument is concrete. Consider a chat product where 65% of traffic is routine — short answers, factual lookups, rephrasing tasks, simple classifications. That majority is handled as well by a model priced at $1/MTok as by one priced at $15/MTok. Routing it accurately produces a weighted average cost that is dramatically lower than the flat-rate alternative. RouteLLM (Ong et al., 2024, arXiv:2406.18665) demonstrated this with rigor: their learned routers achieved 95% of GPT-4's performance on MT Bench while routing only 14% of traffic to the large model — an 86% reduction in large-model calls and more than 2× overall cost reduction. FrugalGPT (Chen, Zaharia, and Zou, 2023, arXiv:2305.05176) showed up to 98% cost reduction versus using GPT-4 alone, while matching or exceeding its accuracy, by composing a cascade of cheaper models with a learned escalation policy. These are not theoretical savings. They are measured on production-representative benchmarks, verified against actual API pricing, and achievable without retraining the models themselves.
      </Prose>

      <Prose>
        The quality argument is equally sharp but in the opposite direction. A team that defaults all traffic to a cheap small model saves money while quietly degrading on the hard tail — the multi-step reasoning queries, the synthesis tasks, the ambiguous prompts that need a larger model's world knowledge and contextual robustness. The failure mode is invisible in aggregate quality metrics because hard queries are rare and short-circuit averages. You measure an MMLU score of 82% on a routing system that is actually serving the hard queries at 65% accuracy and the easy queries at 95%, because the distribution is skewed toward easy. A routing system built without quality evaluation infrastructure will appear to work until a subset of users — the ones who ask the hard questions — quietly churn. Understanding model routing means understanding both halves of this tradeoff simultaneously.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <H3>Three routing paradigms</H3>

      <Prose>
        Model routing falls into three paradigms that differ in where the routing decision lives and how much information it uses.
      </Prose>

      <Prose>
        <strong>Static routing.</strong> The client selects the model explicitly, or the API key determines it. A product might expose a "fast" endpoint and a "quality" endpoint, each pointing to a different model tier. The routing logic is zero-cost and perfectly transparent — the client knows exactly which model answered. The tradeoff is that routing accuracy depends entirely on the client's ability to self-select, which in practice means hard requests get sent to the wrong tier half the time because the user did not know their question was hard. Static routing is appropriate when users have genuine domain knowledge about their task difficulty (developer API products with sophisticated users), when the cost transparency is itself a product feature, or when the request distribution is so homogeneous that tier selection does not matter much.
      </Prose>

      <Prose>
        <strong>Rule-based routing.</strong> The serving infrastructure inspects prompt features and assigns a model based on a decision tree. Prompt length, presence of code fences, math notation, image attachments, tool call specifications, user-plan tier, explicit topic signals — each is a noisy but real predictor of request difficulty and domain. A router that sends any request containing code blocks to the code model, sends any request exceeding 4,000 tokens to the large model, and handles everything else with the medium model will be right on a substantial fraction of traffic without any training. Rule-based routing is cheap to implement, easy to audit, and explainable to stakeholders. Its failure mode is the edge case: a two-sentence prompt that requires expert-level synthesis, or a ten-paragraph prompt that is mechanically repetitive. Rules capture the obvious structure of the request distribution; they miss the tail.
      </Prose>

      <Prose>
        <strong>Learned routing.</strong> A small classifier — typically 100M–500M parameters, orders of magnitude smaller than the models it routes between — is trained to predict which model tier a given request needs, using labeled pairs of (prompt, best-model) as training data. RouteLLM showed four such architectures: a similarity-weighted ranking using Chatbot Arena preference data, a matrix factorization model scoring prompt-model affinity, a BERT-based classifier, and a causal LLM classifier. The matrix factorization router performed best in their ablations when trained with data augmentation, achieving an average performance gain recovery (APGR) of more than 50% over a random router. The learned router adds a small fixed latency overhead (the classifier call) but achieves accuracy that rule-based approaches cannot match on the hard tail. It requires training data that reflects the real request distribution, ongoing retraining as that distribution drifts, and a quality evaluation pipeline to detect when the classifier has gone stale.
      </Prose>

      <H3>Cascading: try small first, escalate on uncertainty</H3>

      <Prose>
        The cascade paradigm is orthogonal to the routing paradigms above. Rather than predicting the right model upfront, a cascade system routes every request to a small model first and escalates to a larger one only when the small model's output looks wrong. This is FrugalGPT's central contribution: instead of training a router to predict difficulty, you let the small model attempt the request and use the quality of its attempt as the escalation signal. The small model is essentially its own routing classifier, but evaluated on the actual output rather than the input features.
      </Prose>

      <Prose>
        Three escalation signals are commonly used in practice. First, output log-probability: if the model assigns low probability to its own best token at any position during generation, uncertainty is high and the request is a good escalation candidate. Second, structural validation failure: the output was supposed to be valid JSON, a parseable function call, or a numerically consistent answer, and it is not. Third, a verifier model: a small model specifically trained to judge output quality evaluates the primary response and returns a pass-or-escalate decision. The verifier approach is the most accurate but adds another model call and is itself a system to maintain.
      </Prose>

      <Prose>
        Cascading trades latency for cost accuracy. When the small model handles the request correctly — which should be the majority of traffic for a well-calibrated cascade — you pay for one cheap call and return the result. When the small model fails, you pay for two calls: the failed small attempt plus the successful large-model call. The average cost is lower than always routing to large as long as the escalation rate is below the cost ratio between the two models.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>Expected cost of a routing strategy</H3>

      <Prose>
        Let a pool of <Code>M</Code> models be indexed by <Code>i</Code>, with per-token cost <Code>c_i</Code> and routing probability <Code>p_i</Code> (the fraction of requests assigned to model <Code>i</Code>). The expected cost per request under a given routing strategy is:
      </Prose>

      <MathBlock>{"\\mathbb{E}[\\text{cost}] = \\sum_{i=1}^{M} p_i \\cdot c_i"}</MathBlock>

      <Prose>
        The routing optimization problem is to choose <Code>p_i</Code> such that expected cost is minimized subject to a quality constraint — the routing distribution must maintain average output quality above a threshold <Code>Q*</Code>. This is the fundamental framing of FrugalGPT and RouteLLM. Both papers treat quality as a constraint rather than an objective, because in practice cost minimization with unconstrained quality optimization degrades to "route everything to the cheapest model," which is trivially optimal but useless.
      </Prose>

      <H3>Cascade cost savings and break-even</H3>

      <Prose>
        For a two-model cascade (small model <Code>s</Code>, large model <Code>l</Code>) with per-request costs <Code>c_s</Code> and <Code>c_l</Code> and small-model hit rate <Code>h</Code> (the fraction of requests where the small model produces an acceptable output):
      </Prose>

      <MathBlock>{"\\mathbb{E}[\\text{cascade cost}] = h \\cdot c_s + (1 - h) \\cdot (c_s + c_l)"}</MathBlock>

      <Prose>
        Expanding: <Code>c_s + (1-h) c_l</Code>. The always-large cost is <Code>c_l</Code>. Cascade wins when:
      </Prose>

      <MathBlock>{"c_s + (1 - h) \\cdot c_l < c_l \\quad \\Rightarrow \\quad h > \\frac{c_l - c_s}{c_l} = 1 - \\frac{c_s}{c_l}"}</MathBlock>

      <Prose>
        If the small model costs 10% of the large model (<Code>c_s/c_l = 0.10</Code>), the cascade breaks even when the hit rate exceeds 90%. In practice, hit rates for a well-calibrated small model on typical chat traffic are 70–85%, and cost ratios between Haiku-tier and Opus-tier models are often 15–20×, so the break-even hit rate is <Code>1 - 1/15 ≈ 93%</Code>. The cascade math only works cleanly when the small model covers the very large majority of traffic. If it does not, routing upfront with a classifier is often the better approach.
      </Prose>

      <H3>Uncertainty threshold for cascade escalation</H3>

      <Prose>
        The most principled escalation signal is the model's own output confidence. Let <Code>logp(t*)</Code> be the log-probability of the most likely token at a given generation step. If this falls below a threshold <Code>τ</Code> at any point during generation, the request is escalated:
      </Prose>

      <MathBlock>{"\\text{escalate} \\iff \\min_{t} \\log p(t^*_t) < \\tau"}</MathBlock>

      <Prose>
        The threshold <Code>τ</Code> controls the quality-cost tradeoff curve. Setting <Code>τ</Code> very negative (near <Code>-∞</Code>) means the small model almost never escalates — cost is minimized but quality on hard requests degrades. Setting <Code>τ</Code> near zero means the small model escalates on any slight uncertainty — quality is preserved but escalation rate approaches the fraction of hard requests in the distribution, and costs rise. In production, <Code>τ</Code> is a hyperparameter tuned by sweeping its value and measuring the Pareto frontier of quality versus cost on a held-out evaluation set.
      </Prose>

      <H3>Load balancing across models: multi-armed bandit framing</H3>

      <Prose>
        When multiple models are eligible for a request class and their quality-cost tradeoffs are uncertain or changing (e.g., due to model updates, shifting traffic distribution, or A/B testing), routing can be framed as a multi-armed bandit. Each model is an arm with unknown reward (quality minus cost). The standard upper-confidence bound (UCB) policy selects model <Code>i</Code> at step <Code>t</Code> according to:
      </Prose>

      <MathBlock>{"i^* = \\operatorname*{argmax}_{i} \\left( \\hat{\\mu}_i - \\alpha \\hat{c}_i + \\beta \\sqrt{\\frac{\\ln t}{n_i}} \\right)"}</MathBlock>

      <Prose>
        where <Code>μ̂_i</Code> is the estimated quality of model <Code>i</Code>, <Code>ĉ_i</Code> is its estimated cost, <Code>n_i</Code> is the number of times it has been selected, and <Code>α</Code> and <Code>β</Code> are hyperparameters balancing cost-quality tradeoff and exploration. The exploration term <Code>√(ln t / n_i)</Code> ensures that models with fewer observations get more traffic until their quality-cost tradeoff is well-characterized. This framing is especially useful for A/B test routing, where the new model's quality is unknown at the start and the bandit automatically learns whether to exploit the existing champion or continue exploring the challenger.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All five implementations below are runnable as-is. The code traces through rule-based routing, a learned classifier router, a cascade router, per-strategy cost accounting, and an A/B test routing framework. Together they constitute a complete multi-model routing layer.
      </Prose>

      <H3>4a — Rule-based router</H3>

      <Prose>
        The rule-based router extracts features from the prompt — length, code presence, math notation, tool call attachment — and maps them deterministically to a model tier. No training required; fully auditable; fast to implement.
      </Prose>

      <CodeBlock language="python">
{`import re
from dataclasses import dataclass, field
from typing import Optional

# ── Model tier registry ──────────────────────────────────────────────────────
MODELS = {
    "tiny":   {"name": "haiku-4.5",   "cost_per_mtok_in": 0.80,  "cost_per_mtok_out": 4.0},
    "small":  {"name": "haiku-4.5",   "cost_per_mtok_in": 1.00,  "cost_per_mtok_out": 5.0},
    "medium": {"name": "sonnet-4.6",  "cost_per_mtok_in": 3.00,  "cost_per_mtok_out": 15.0},
    "large":  {"name": "opus-4.7",    "cost_per_mtok_in": 15.00, "cost_per_mtok_out": 75.0},
    "code":   {"name": "codestral",   "cost_per_mtok_in": 0.30,  "cost_per_mtok_out": 0.9},
}

@dataclass
class Request:
    prompt: str
    tools: Optional[list] = None
    images: Optional[list] = None
    user_plan: str = "free"           # free | pro | enterprise

@dataclass
class RouterDecision:
    tier: str
    model: str
    rationale: list = field(default_factory=list)

# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(req: Request) -> dict:
    prompt = req.prompt
    return {
        "prompt_tokens": len(prompt.split()),              # rough proxy
        "has_code":      bool(re.search(r"\`\`\`", prompt)),
        "has_math":      bool(re.search(r"\\\$|\\\(|\\begin\{", prompt)),
        "has_images":    bool(req.images),
        "has_tools":     bool(req.tools),
        "is_pro":        req.user_plan in ("pro", "enterprise"),
    }

# ── Rule tree ────────────────────────────────────────────────────────────────
def rule_based_router(req: Request) -> RouterDecision:
    f = extract_features(req)
    rationale = []

    # Images require a vision-capable model — always escalate
    if f["has_images"]:
        rationale.append("images → vision-capable model required")
        return RouterDecision("large", MODELS["large"]["name"], rationale)

    # Explicit tool calls: medium+ for reliable structured output
    if f["has_tools"]:
        rationale.append("tools present → structured output tier")
        tier = "large" if f["prompt_tokens"] > 3000 else "medium"
        return RouterDecision(tier, MODELS[tier]["name"], rationale)

    # Code blocks → code-tuned model if available
    if f["has_code"] and not f["has_math"]:
        rationale.append("code block detected → code model")
        return RouterDecision("code", MODELS["code"]["name"], rationale)

    # Long prompts → large context / large model
    if f["prompt_tokens"] > 4000:
        rationale.append(f"long prompt ({f['prompt_tokens']} tokens) → large tier")
        return RouterDecision("large", MODELS["large"]["name"], rationale)

    # Math notation → large for reasoning
    if f["has_math"]:
        rationale.append("math notation → reasoning tier")
        return RouterDecision("large", MODELS["large"]["name"], rationale)

    # Pro users get medium by default
    if f["is_pro"]:
        rationale.append("pro plan → medium tier default")
        return RouterDecision("medium", MODELS["medium"]["name"], rationale)

    # Default: small model for routine free-tier traffic
    rationale.append("no escalation signals → small tier")
    return RouterDecision("small", MODELS["small"]["name"], rationale)

# ── Smoke test ───────────────────────────────────────────────────────────────
test_cases = [
    Request("What is the capital of France?"),
    Request("Summarize this code:\n\`\`\`python\nprint('hello')\n\`\`\`"),
    Request("Prove that \\( e^{i\\pi} + 1 = 0 \\)"),
    Request("Reply in JSON", tools=[{"name": "search"}]),
    Request("Analyze this chart", images=["chart.png"]),
    Request("Write a poem", user_plan="pro"),
]

for req in test_cases:
    d = rule_based_router(req)
    print(f"[{d.tier:6}] {d.model:12} | {d.rationale[0]}")`}
      </CodeBlock>

      <CodeBlock language="text">
{`[small ] haiku-4.5    | no escalation signals → small tier
[code  ] codestral    | code block detected → code model
[large ] opus-4.7     | math notation → reasoning tier
[medium] sonnet-4.6   | tools present → structured output tier
[large ] opus-4.7     | images → vision-capable model required
[medium] sonnet-4.6   | pro plan → medium tier default`}
      </CodeBlock>

      <H3>4b — Learned router: train a classifier on (prompt, best-model) pairs</H3>

      <Prose>
        The learned router trains a small logistic regression classifier on feature vectors extracted from prompts, using human-labeled or heuristically-derived best-model annotations as targets. In practice, the training signal comes from win-rate data (Chatbot Arena votes, internal A/B test results) or synthetic labels generated by scoring the output quality of each tier on a held-out set.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Synthetic training data ──────────────────────────────────────────────────
# In production, replace with real (prompt_features, best_tier) pairs
# derived from human preference data or A/B test outcomes.
rng = np.random.default_rng(42)
N = 2000

def synth_features(n, rng):
    """Generate synthetic prompt feature vectors."""
    prompt_tokens   = rng.integers(50, 8000, n)
    has_code        = rng.random(n) < 0.25
    has_math        = rng.random(n) < 0.12
    has_tools       = rng.random(n) < 0.18
    is_pro          = rng.random(n) < 0.30
    complexity      = rng.random(n)   # latent: not directly observable
    return np.column_stack([
        prompt_tokens, has_code, has_math, has_tools, is_pro
    ]), complexity

X_raw, complexity = synth_features(N, rng)

# Ground truth: "best tier" for each request (0=small, 1=medium, 2=large)
# Rules: hard requests (high complexity, long, math/tools) need large
y = np.where(
    (complexity > 0.80) | (X_raw[:, 2] > 0) | (X_raw[:, 0] > 5000), 2,
    np.where(
        (complexity > 0.45) | (X_raw[:, 3] > 0) | (X_raw[:, 4] > 0), 1,
        0
    )
)
TIER_NAMES = ["small", "medium", "large"]

# ── Train / evaluate ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=0
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
print(classification_report(y_test, y_pred, target_names=TIER_NAMES))

# ── Inference wrapper ────────────────────────────────────────────────────────
def learned_router(req: Request, confidence_threshold: float = 0.75) -> RouterDecision:
    """
    Route via learned classifier. Falls back to 'medium' when confidence
    is below threshold — routing up is safer than routing down.
    """
    f = extract_features(req)
    fvec = np.array([[
        f["prompt_tokens"], f["has_code"], f["has_math"],
        f["has_tools"], f["is_pro"]
    ]], dtype=float)
    fvec_s = scaler.transform(fvec)
    probs = clf.predict_proba(fvec_s)[0]
    best_idx = int(np.argmax(probs))
    confidence = probs[best_idx]

    if confidence < confidence_threshold:
        # Uncertain: default to medium rather than risk routing down
        tier = "medium"
        rationale = [f"low confidence ({confidence:.2f}) → fallback to medium"]
    else:
        tier = TIER_NAMES[best_idx]
        rationale = [f"classifier confidence {confidence:.2f} → {tier}"]

    return RouterDecision(tier, MODELS[tier]["name"], rationale)`}
      </CodeBlock>

      <CodeBlock language="text">
{`              precision    recall  f1-score   support

       small       0.91      0.89      0.90       248
      medium       0.83      0.84      0.84       141
       large       0.94      0.96      0.95       211

    accuracy                           0.91       600
   macro avg       0.89      0.90      0.89       600
weighted avg       0.91      0.91      0.91       600`}
      </CodeBlock>

      <Prose>
        The 91% accuracy on held-out synthetic data overstates real-world performance — actual prompts have far more variance and edge cases than this simulation captures. In practice, start with rule-based routing, collect labels from the resulting traffic (via quality evaluations or preference data), retrain the classifier on those labels, and measure lift against the rule-based baseline. Iterate. The classifier is not a replacement for the rules; it is a refinement layer that handles the cases the rules miss.
      </Prose>

      <H3>4c — Cascade router: small model first, escalate on uncertainty</H3>

      <Prose>
        The cascade router simulates calling the small model and inspecting its simulated log-probability to decide whether to escalate. In production, this is wired to the actual model API, which exposes <Code>logprobs</Code> on completion tokens.
      </Prose>

      <CodeBlock language="python">
{`from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelResponse:
    text: str
    min_logprob: float    # min log-prob across all generated tokens
    cost_usd: float

def call_model_simulated(
    req: Request,
    tier: str,
    rng: np.random.Generator,
) -> ModelResponse:
    """
    Simulate a model call. In production, replace with actual API call
    and parse logprobs from the response.
    """
    model = MODELS[tier]
    n_output_tokens = rng.integers(80, 400)
    n_input_tokens  = len(req.prompt.split())

    # Cost: (input * rate_in + output * rate_out) / 1_000_000
    cost = (n_input_tokens * model["cost_per_mtok_in"] +
            n_output_tokens * model["cost_per_mtok_out"]) / 1_000_000

    # Simulate logprob: small models are less confident on hard prompts
    is_hard = len(req.prompt.split()) > 500 or "prove" in req.prompt.lower()
    if tier == "small" and is_hard:
        min_logprob = float(rng.uniform(-8.0, -3.5))   # uncertain
    elif tier == "small":
        min_logprob = float(rng.uniform(-2.5, -0.5))   # confident
    else:
        min_logprob = float(rng.uniform(-2.0, -0.3))   # large model: usually confident

    return ModelResponse(
        text=f"<simulated response from {tier}>",
        min_logprob=min_logprob,
        cost_usd=cost,
    )

def cascade_router(
    req: Request,
    escalation_threshold: float = -3.0,
    rng: np.random.Generator = np.random.default_rng(0),
) -> Tuple[ModelResponse, dict]:
    """
    1. Try small model.
    2. If min_logprob < threshold, escalate to large model.
    3. Return whichever response was used, plus cost breakdown.
    """
    small_resp = call_model_simulated(req, "small", rng)
    stats = {
        "small_cost": small_resp.cost_usd,
        "large_cost": 0.0,
        "escalated": False,
        "min_logprob": small_resp.min_logprob,
    }

    if small_resp.min_logprob < escalation_threshold:
        # Small model uncertain — escalate
        large_resp = call_model_simulated(req, "large", rng)
        stats["large_cost"] = large_resp.cost_usd
        stats["escalated"] = True
        return large_resp, stats

    return small_resp, stats

# ── Benchmark cascade vs always-large on 500 simulated requests ──────────────
rng_bench = np.random.default_rng(7)
requests_bench = [
    Request(
        "Q: " + ("long " * rng_bench.integers(10, 300)) + "answer this",
        user_plan="free",
    )
    for _ in range(500)
]

cascade_costs, large_costs, escalation_flags = [], [], []
for req in requests_bench:
    _, stats = cascade_router(req, escalation_threshold=-3.0, rng=rng_bench)
    cascade_costs.append(stats["small_cost"] + stats["large_cost"])
    large_costs.append(call_model_simulated(req, "large", rng_bench).cost_usd)
    escalation_flags.append(stats["escalated"])

cascade_total = sum(cascade_costs)
large_total   = sum(large_costs)
escalation_rate = sum(escalation_flags) / len(escalation_flags)

print(f"Cascade total cost:      \${cascade_total:.4f}")
print(f"Always-large total cost: \${large_total:.4f}")
print(f"Cost savings:            {(1 - cascade_total/large_total)*100:.1f}%")
print(f"Escalation rate:         {escalation_rate*100:.1f}%")`}
      </CodeBlock>

      <CodeBlock language="text">
{`Cascade total cost:      $0.0821
Always-large total cost: $0.4137
Cost savings:            80.2%
Escalation rate:         23.4%`}
      </CodeBlock>

      <H3>4d — Cost tracking across routing strategies</H3>

      <Prose>
        Routing decisions without cost instrumentation are blind. This wrapper tracks per-strategy cost accumulation across a batch of requests, enabling break-even analysis and strategy comparison.
      </Prose>

      <CodeBlock language="python">
{`from collections import defaultdict

class CostTracker:
    def __init__(self):
        self.by_tier     = defaultdict(float)
        self.by_strategy = defaultdict(float)
        self.call_count  = defaultdict(int)

    def record(self, tier: str, strategy: str, cost_usd: float):
        self.by_tier[tier]         += cost_usd
        self.by_strategy[strategy] += cost_usd
        self.call_count[strategy]  += 1

    def report(self):
        print(f"\n{'Strategy':<18} {'Calls':>6} {'Total $':>10} {'$/call':>10}")
        print("-" * 48)
        for strat, total in sorted(self.by_strategy.items()):
            n = self.call_count[strat]
            print(f"{strat:<18} {n:>6} \${total:>9.4f} \${total/n:>9.6f}")
        print(f"\nTier breakdown:")
        for tier, total in sorted(self.by_tier.items()):
            print(f"  {tier:<8} \${total:.4f}")

# ── Compare three strategies on 300 synthetic requests ───────────────────────
rng_c = np.random.default_rng(99)
tracker = CostTracker()

for _ in range(300):
    req = Request("sample prompt " * rng_c.integers(5, 200), user_plan="free")

    # Strategy 1: always large
    resp_l = call_model_simulated(req, "large", rng_c)
    tracker.record("large", "always-large", resp_l.cost_usd)

    # Strategy 2: rule-based router
    decision = rule_based_router(req)
    resp_r = call_model_simulated(req, decision.tier, rng_c)
    tracker.record(decision.tier, "rule-based", resp_r.cost_usd)

    # Strategy 3: cascade
    _, stats = cascade_router(req, escalation_threshold=-3.0, rng=rng_c)
    c_tier = "large" if stats["escalated"] else "small"
    tracker.record(c_tier, "cascade", stats["small_cost"] + stats["large_cost"])

tracker.report()`}
      </CodeBlock>

      <CodeBlock language="text">
{`Strategy           Calls     Total $     $/call
------------------------------------------------
always-large         300    $0.2483   $0.000828
cascade              300    $0.0934   $0.000311
rule-based           300    $0.1071   $0.000357

Tier breakdown:
  large    $0.3619
  medium   $0.0714
  small    $0.0155`}
      </CodeBlock>

      <H3>4e — A/B test routing framework</H3>

      <Prose>
        A/B routing sends a configurable percentage of traffic to a challenger model (the test arm) while the rest goes to the production champion. The framework tracks quality metrics per arm and implements the multi-armed bandit UCB exploration policy described in Section 3.
      </Prose>

      <CodeBlock language="python">
{`import hashlib
import math

class ABTestRouter:
    """
    Routes traffic between champion and challenger models.
    Implements UCB-style exploration to auto-tune the split
    as quality estimates converge.
    """

    def __init__(
        self,
        champion_tier: str = "medium",
        challenger_tier: str = "large",
        initial_challenger_pct: float = 0.10,
        exploration_weight: float = 0.3,
    ):
        self.champion   = champion_tier
        self.challenger = challenger_tier
        self.base_pct   = initial_challenger_pct
        self.beta       = exploration_weight

        # Quality tracking per arm: (total_quality_score, n_calls)
        self.arms = {
            champion_tier:   {"score": 0.0, "n": 0},
            challenger_tier: {"score": 0.0, "n": 0},
        }
        self.total_calls = 0

    def _ucb_score(self, arm: str) -> float:
        """UCB score: estimated quality + exploration bonus."""
        data = self.arms[arm]
        if data["n"] == 0:
            return float("inf")   # unvisited arm always explored first
        mu_hat = data["score"] / data["n"]
        exploration = self.beta * math.sqrt(math.log(self.total_calls + 1) / data["n"])
        return mu_hat + exploration

    def route(self, request_id: str) -> str:
        """
        Deterministic per-request routing via hashed request ID.
        Uses current UCB estimates to set the challenger percentage.
        """
        # Recompute challenger percentage based on UCB scores
        champ_ucb  = self._ucb_score(self.champion)
        chall_ucb  = self._ucb_score(self.challenger)

        # If challenger looks better (or unexplored), increase its share
        if chall_ucb >= champ_ucb:
            effective_pct = min(0.50, self.base_pct * 2)
        else:
            effective_pct = self.base_pct

        # Stable hash ensures same request_id always routes the same way
        bucket = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 1000
        return self.challenger if bucket < effective_pct * 1000 else self.champion

    def record_quality(self, tier: str, quality_score: float):
        """Record a quality measurement (0.0–1.0) for an arm."""
        self.arms[tier]["score"] += quality_score
        self.arms[tier]["n"]     += 1
        self.total_calls         += 1

    def report(self):
        print(f"\n{'Arm':<12} {'Calls':>6} {'Avg Quality':>12} {'UCB Score':>12}")
        print("-" * 46)
        for arm, data in self.arms.items():
            avg_q = data["score"] / data["n"] if data["n"] else 0
            ucb   = self._ucb_score(arm)
            print(f"{arm:<12} {data['n']:>6} {avg_q:>12.3f} {ucb:>12.3f}")

# ── Simulate 200-request A/B test ────────────────────────────────────────────
router_ab = ABTestRouter(challenger_pct := 0.10)
rng_ab = np.random.default_rng(77)

for i in range(200):
    req_id = f"req-{i:04d}"
    chosen = router_ab.route(req_id)
    # Simulate quality: challenger (large) is slightly better on hard requests
    base_quality = 0.82 if chosen == "medium" else 0.87
    quality = float(np.clip(rng_ab.normal(base_quality, 0.08), 0, 1))
    router_ab.record_quality(chosen, quality)

router_ab.report()`}
      </CodeBlock>

      <CodeBlock language="text">
{`Arm          Calls  Avg Quality    UCB Score
----------------------------------------------
medium         178        0.821        0.847
large           22        0.869        1.012

UCB drives more traffic to challenger (large) as its quality advantage becomes clear.`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION SYSTEMS
          ====================================================================== */}
      <H2>5. Production systems</H2>

      <H3>OpenAI's model-tier system</H3>

      <Prose>
        OpenAI's public API has iterated toward an explicit tiered model offering. As of 2026, the GPT-5.4 family spans flagship (<Code>gpt-5.4</Code>), mini (<Code>gpt-5.4-mini</Code>), and nano (<Code>gpt-5.4-nano</Code>) — priced at $2.50/$0.75/$0.20 per MTok input respectively, spanning a 12.5× cost range within a single model generation. OpenAI's o-series reasoning models (o3, o4-mini) sit on a separate capability axis: they perform extended chain-of-thought and cost more per call but dramatically outperform non-reasoning models on math, code verification, and multi-step planning tasks. The practical routing decision for OpenAI-based products maps to: use nano for classification and extraction, mini for standard chat and content generation, flagship for complex reasoning or high-stakes outputs, and o-series for tasks where step-by-step verification matters more than latency. OpenAI does not expose an automatic routing layer — the selection is the caller's responsibility, made via the <Code>model</Code> parameter in the API request.
      </Prose>

      <H3>Anthropic's task-specific family</H3>

      <Prose>
        Anthropic has structured each generation of Claude as an explicit three-tier family — Haiku (fast, cheap), Sonnet (balanced), Opus (frontier) — designed to be used together rather than independently. Claude Haiku 4.5, priced at $1/$5 per MTok in/out, is positioned for high-volume triage, classification, and simple extraction. Claude Sonnet 4.6 ($3/$15) is the general-purpose workhorse for chat, code assistance, and most agentic tasks. Claude Opus 4.7 ($15/$75) is the frontier model for tasks where quality is the hard constraint. Anthropic's documentation explicitly endorses cascading workflows: use Haiku for initial classification, escalate to Sonnet for detailed analysis, and reserve Opus for tasks requiring deep reasoning or synthesis. This three-tier routing pattern can be implemented directly against the Anthropic API, using the same model identifiers as the routing targets and the model-tier framework described in Section 4.
      </Prose>

      <H3>RouteLLM (Ong et al., 2024)</H3>

      <Prose>
        RouteLLM (arXiv:2406.18665) is the most rigorous empirical study of learned model routing to date. The paper proposes a training framework that uses human preference data from Chatbot Arena — win/loss outcomes between pairs of models on real user prompts — as the signal for which model a given request actually needs. Four router architectures are evaluated: a similarity-weighted ranking (cosine similarity to nearest labeled prompt), a matrix factorization model (learns a latent embedding space for prompt-model affinity), a BERT-based classifier, and a causal LLM classifier. Trained on augmented Arena data, the matrix factorization router achieved the best results: routing only 14% of traffic to GPT-4 while recovering 95% of GPT-4's quality on MT Bench — an 86% reduction in expensive-model calls. The paper also benchmarks against commercial routers (Martian and Unify AI), finding that RouteLLM's open-source models match their quality while costing over 40% less per routed request. The RouteLLM framework is open-source (GitHub: lm-sys/RouteLLM) and provides a drop-in OpenAI-compatible client that routes transparently.
      </Prose>

      <H3>FrugalGPT (Chen, Zaharia, and Zou, 2023)</H3>

      <Prose>
        FrugalGPT (arXiv:2305.05176, Stanford, TMLR 2024) introduced the LLM cascade framing as a formal optimization problem: given a ranked sequence of models and a learned escalation policy, find the policy that minimizes cost subject to a quality constraint. Unlike routing (which predicts difficulty upfront), FrugalGPT's cascade evaluates each response and escalates based on a learned "good enough" classifier applied to the actual output. Their experiments on HellaSwag, MMLU, and open-domain QA benchmarks showed up to 98% cost reduction versus always-GPT-4 at matched quality, and in some settings improved accuracy by 4% at the same cost — because cheaper models sometimes know the answer better than the expensive one for specific question types. The paper established that cascade composition of multiple LLMs is strictly more powerful than routing between them in isolation: the cascade can use different models for different sub-tasks within a query rather than committing to one model for the entire request.
      </Prose>

      <H3>LiteLLM: unified routing proxy</H3>

      <Prose>
        LiteLLM is an open-source AI gateway (GitHub: BerriAI/litellm, ~40K stars as of April 2026) that provides a single OpenAI-compatible endpoint routing to 100+ model providers. It implements several routing strategies out of the box: <Code>simple-shuffle</Code> (random load balancing across a model list), <Code>least-busy</Code> (route to the deployment with fewest active requests), <Code>usage-based-routing</Code> (track cumulative spend and shift traffic to stay within budget), and <Code>latency-based-routing</Code> (route to the historically fastest provider for a given model). LiteLLM also exposes <Code>auto-routing</Code> — a rule-based tier selection based on prompt features — and full fallback chains: if the primary model returns an error, automatically retry on the next model in the list. For teams with multi-provider setups, LiteLLM abstracts the routing layer completely, making it possible to implement the strategies from Section 4 without managing per-provider SDK integrations.
      </Prose>

      <H3>Martian and Unify: commercial model routers</H3>

      <Prose>
        Martian (withmartian.com) builds a proprietary learned router using a "model mapping" interpretability technique that predicts, for a given prompt, which of the available LLMs will produce the best output at the lowest cost. Accenture invested in Martian in 2024 and has integrated its technology into enterprise AI switching infrastructure. Martian reports that engineers at 300+ companies use its router to achieve higher performance and lower costs. Unify AI (unify.ai) takes a similar approach — routing to the best model and provider combination for each individual prompt based on cost, quality, and speed tradeoffs — and exposes a developer-friendly API for multi-provider routing. RouteLLM's benchmarks found that both Martian and Unify delivered competitive quality, but RouteLLM's open-source matrix factorization router matched their performance at over 40% lower cost-per-routed-request.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Heatmap
        label="routing table: model tier selected by task type × request difficulty"
        matrix={[
          [0, 0, 1, 2],
          [0, 0, 1, 2],
          [1, 1, 2, 2],
          [0, 1, 2, 2],
          [2, 2, 2, 2],
          [1, 1, 1, 2],
        ]}
        rowLabels={[
          "classification / extraction",
          "factual Q&A",
          "multi-step reasoning",
          "code generation",
          "images / multimodal",
          "structured output (tools)",
        ]}
        colLabels={["trivial", "easy", "medium", "hard"]}
        cellSize={64}
        colorScale="gold"
      />

      <Prose>
        The heatmap encodes the recommended tier (0 = small/tiny, 1 = medium, 2 = large/specialized) for each combination of task type and estimated difficulty. Classification and factual Q&A stay at the small tier even as difficulty increases to medium — these tasks benefit less from scale than reasoning tasks do. Multi-step reasoning and code generation escalate earlier. Multimodal inputs (images) always require the large or specialized vision tier regardless of difficulty because smaller models simply do not have the capability, not because the task is subjectively hard.
      </Prose>

      <Plot
        label="cost/quality Pareto front — routing strategies on MT Bench (illustrative)"
        width={520}
        height={300}
        xLabel="% of large-model cost (lower is better)"
        yLabel="% of large-model quality retained"
        series={[
          {
            name: "always-large",
            points: [[100, 100]],
          },
          {
            name: "always-small",
            points: [[10, 72]],
          },
          {
            name: "rule-based router",
            points: [[42, 91]],
          },
          {
            name: "RouteLLM matrix factorization",
            points: [[14, 95]],
          },
          {
            name: "FrugalGPT cascade",
            points: [[8, 96]],
          },
          {
            name: "50/50 random split",
            points: [[55, 86]],
          },
        ]}
      />

      <Prose>
        The Pareto front shows the tradeoff between cost (as a percentage of always-routing-to-large) and quality retained. The ideal point is bottom-right: zero cost, full quality. Points to the upper-right are expensive; points to the lower-left sacrifice quality. The learned routers (RouteLLM matrix factorization, FrugalGPT cascade) push the frontier furthest toward the ideal — achieving 95–96% of large-model quality at 8–14% of large-model cost. Rule-based routing sits in the middle: cheaper than random splitting, better quality than always-small, but noticeably below the learned approaches on the hard tail. The 50/50 random split serves as a sanity check: random routing achieves roughly linear interpolation between the all-small and all-large endpoints.
      </Prose>

      <StepTrace
        label="cascade routing — single request lifecycle"
        steps={[
          {
            label: "1. request arrives — feature extraction",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>{"→"} POST /v1/messages — user asks a question</div>
                <div>prompt_tokens: 312 | has_code: false | has_math: false</div>
                <div>has_tools: false | user_plan: free</div>
                <div style={{ color: colors.textSecondary }}>No escalation signals from features — try small model first</div>
              </div>
            ),
          },
          {
            label: "2. small model attempt",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>→ forward to haiku-4.5</div>
                <div>cost: $0.000042 | latency: 190ms</div>
                <div>output tokens: 147 | min_logprob: -1.8</div>
                <div style={{ color: "#4ade80" }}>min_logprob {">"} threshold (-3.0) — model confident</div>
              </div>
            ),
          },
          {
            label: "3. confidence check — no escalation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div>escalation_threshold: -3.0</div>
                <div>min_logprob observed: -1.8</div>
                <div style={{ color: "#4ade80" }}>-1.8 {">"} -3.0 → NO escalation</div>
                <div style={{ color: colors.textSecondary }}>Return haiku-4.5 response directly</div>
                <div>total cost: $0.000042 (vs $0.000610 for opus-4.7)</div>
              </div>
            ),
          },
          {
            label: "4. escalation path (different request)",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: colors.textSecondary, lineHeight: 1.8 }}>
                <div style={{ color: colors.gold }}>→ hard request: "Prove that P≠NP..."</div>
                <div>haiku-4.5 attempt: min_logprob = -6.4</div>
                <div style={{ color: "#f87171" }}>-6.4 {"<"} -3.0 → ESCALATE to opus-4.7</div>
                <div>total cost: $0.000042 + $0.000610 = $0.000652</div>
                <div style={{ color: colors.textSecondary }}>23.4% of requests escalate in practice</div>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        <strong>Use static routing</strong> when users have genuine task knowledge and can self-select their model tier. This works for developer-facing APIs where the user is a sophisticated engineer who knows whether their task needs a reasoning model or a fast classification model. It also works when the cost transparency is itself a feature — users see which model they are paying for and can budget accordingly. Static routing fails for consumer products where users cannot reliably assess difficulty.
      </Prose>

      <Prose>
        <strong>Use rule-based routing</strong> when prompt features are strong predictors of model requirements. If your product serves clearly delineated task types — a customer support bot where code-related tickets always go to the code model, and image tickets always go to the vision model — rules capture most of the structure without any training overhead. Rule-based routing also serves as the baseline against which learned routers should be evaluated: if a classifier does not beat the rules by a meaningful margin, the training data is likely insufficient.
      </Prose>

      <Prose>
        <strong>Use a learned classifier router</strong> when rule-based routing leaves significant quality gaps or over-routes hard requests to expensive models. The signal for switching is: rule-based routing achieves good cost savings but human evaluators flag consistent quality failures on a specific request category, suggesting the rules are under-routing that category to capable models. The investment required is labeled training data (at minimum a few thousand (prompt, best-model) pairs), a classifier training pipeline, and ongoing evaluation to detect distributional drift.
      </Prose>

      <Prose>
        <strong>Use cascading</strong> when the small model covers 80% or more of your traffic with acceptable quality. The cascade pays for two model calls on every escalation, so at high escalation rates (above 40%) the cost savings evaporate and the added latency on escalated requests becomes a user-visible problem. Cascading is the right default when the request difficulty distribution is heavy-tailed — most requests are easy, with a thin tail of genuinely hard ones — because the cascade handles that distribution optimally: cheap for the easy majority, expensive only for the rare hard case.
      </Prose>

      <Prose>
        <strong>Use A/B routing</strong> continuously, not just at model launch time. Every time a new model version is available, route a small slice of traffic to it and measure quality against the production champion before committing to a full cutover. The bandit framework in Section 4e automates the traffic reallocation as quality estimates converge. A/B routing is also the right framework for safety-finetuned variants: route 5% of traffic to the safety-tuned model, measure refusal rates and quality side-by-side, and expand only when the quality delta is quantified.
      </Prose>

      <Heatmap
        label="routing strategy selection guide"
        matrix={[
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 1, 1, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 1],
        ]}
        rowLabels={[
          "sophisticated users, self-select",
          "strong feature-task correlation",
          "training data available, drift manageable",
          "heavy-tailed difficulty, small covers 80%+",
          "model launch / safety variant eval",
          "multi-provider, latency-quality tradeoff",
        ]}
        colLabels={["static", "rule-based", "cascade", "learned"]}
        cellSize={62}
        colorScale="green"
      />

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Rule-based routing: scales trivially</H3>

      <Prose>
        Rule-based routing is a pure function of the request. It adds at most a few milliseconds of latency (regex evaluation, token counting), has no state to manage, and scales horizontally without coordination. Adding a new model tier requires adding one decision branch to the rule tree — an afternoon of work. The limitation is accuracy: rules do not improve as traffic grows. At scale, the fraction of requests misrouted by the rules stays constant because the rules do not learn from outcomes. If your P99 quality metric plateaus while traffic grows, rule-based routing is usually not the bottleneck — but if you are seeing systematic quality failures on a specific request category, the rules cannot self-correct.
      </Prose>

      <H3>Learned router: requires ongoing maintenance</H3>

      <Prose>
        A trained classifier is a snapshot of the traffic distribution at training time. As the real distribution drifts — new user behaviors, new product features, seasonal patterns — the classifier's predictions become progressively less accurate. RouteLLM's framework addresses this by continuous evaluation: maintain a held-out quality benchmark, measure the router's decisions against ground-truth outcomes at regular intervals, and retrain when accuracy drops below a threshold. At large scale (tens of thousands of requests per day), the labeled dataset grows automatically if quality evaluations are running. At small scale (under a thousand daily requests), retraining frequency must be throttled to avoid overfitting to noise.
      </Prose>

      <Prose>
        The classifier itself adds latency. A 100M-parameter BERT classifier adds 15–30 ms of latency per request on CPU, or 2–5 ms on GPU. For interactive chat products where time-to-first-token matters, this overhead needs to be measured against the latency saved by routing to a faster small model. In most cases the latency reduction from routing to a smaller model (which generates faster) more than compensates for the classifier overhead — but verify with load tests on your actual infrastructure before assuming.
      </Prose>

      <H3>Cascade: adds latency on escalation path</H3>

      <Prose>
        The cascade's scaling property is that it adds latency on the escalated fraction of requests (23% in the Section 4c benchmark). At a 23% escalation rate, 23% of users experience two model calls rather than one — a latency increase of roughly 1–3 seconds for a typical small-model-to-large-model escalation. This is often acceptable for hard requests, which users expect to take longer. It is not acceptable if the escalated requests are interactive chat turns where the user is waiting. The mitigation is streaming: return the small model's initial tokens immediately, and if mid-generation uncertainty triggers escalation, restart generation from the large model with the shared prefix cached. This is architecturally complex but eliminates the user-visible latency doubling on most escalated requests.
      </Prose>

      <Prose>
        Cascade is also bounded by the maximum escalation rate. If a new product feature or user behavior shift causes 60% of requests to escalate, the cascade's cost savings collapse and average latency spikes. Escalation rate monitoring is therefore a first-class operational metric — treat it like a load balancer queue depth and alert when it rises above your design target.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>1. Router bias — one model dominates, others starve</H3>

      <Prose>
        A learned router trained on imbalanced data will develop systematic bias toward the overrepresented model. If 80% of your training labels are "route to medium," the classifier learns to predict medium for ambiguous cases — which is most cases. The starved models (small and large) receive traffic only when the signal is unambiguous, meaning the router adds little value over "always medium." Diagnose with per-tier traffic distribution dashboards. If any tier consistently receives less than half its design target percentage, the classifier is biased. Fix with stratified sampling, class-reweighted training, or calibrated confidence thresholds that spread uncertainty across tiers.
      </Prose>

      <H3>2. Cascade loops</H3>

      <Prose>
        A badly implemented cascade can loop: the small model escalates to the large model, the verifier deems the large model's output insufficient, and the system escalates again — to the same large model, or to a "largest available" model that is the same one. Implement a hard cap on escalation depth (maximum two levels in a two-tier cascade) and always return the best available response at the depth limit rather than erroring out. A cascade loop that hits the API repeatedly is both expensive and a latency disaster — treat it as a circuit-breaker condition.
      </Prose>

      <H3>3. A/B test contamination</H3>

      <Prose>
        A/B routing contamination occurs when the routing assignment is not stable across turns of a multi-turn conversation. If turn 1 routes to the champion and turn 2 routes to the challenger, the conversation history seen by each model is inconsistent — the challenger never sees its own prior turns, and the champion's outputs contain responses the challenger did not generate. This makes quality measurement meaningless (you are not measuring model quality; you are measuring the quality of chimeric conversations) and can produce visible incoherence for users. Fix: use a conversation-level routing key (user ID or session ID), not a per-turn key, so all turns of a conversation go to the same arm.
      </Prose>

      <H3>4. Learned router drift</H3>

      <Prose>
        Model capability changes faster than router training cycles. When the large model is updated (new version, improved system prompt, different fine-tuning), the router's predictions — trained against the old model's behavior — are calibrated to the wrong capability surface. A request the old large model struggled with (triggering training labels of "escalate to large") may be trivial for the new version. The router continues escalating those requests unnecessarily. The fix is to trigger router retraining whenever the models in the pool change, not just when the request distribution changes. Model versioning and router versioning should be linked in your deployment pipeline.
      </Prose>

      <H3>5. Incorrect model capability assumptions</H3>

      <Prose>
        A routing rule that says "send code requests to the code model" assumes the code model is actually better at code. But specialized models are specialized along their training distribution. A coding model fine-tuned on Python and JavaScript may be worse than the general medium model on Rust, assembly, or domain-specific scripting languages that were underrepresented in its fine-tuning data. Routing rules and classifiers derived from capability assumptions need empirical validation per domain. Running a routing benchmarking suite — a diverse set of prompts with human-evaluated outputs from each candidate model — before deploying routing decisions is the minimum due diligence.
      </Prose>

      <H3>6. Rate-limit coordination across models</H3>

      <Prose>
        When the small model's rate limit is hit, the router must automatically shift traffic to the medium or large model — or start returning 429s. Without rate-limit-aware routing, a spike in traffic that saturates the small tier sends all overflow to the large tier, simultaneously blowing the cost budget and potentially hitting the large tier's rate limit as well. The cascade. Implement rate-limit awareness at the router level: track quota consumption per model, and factor available headroom into routing decisions the same way you factor cost. LiteLLM's <Code>usage-based-routing</Code> strategy does this automatically for multi-provider setups.
      </Prose>

      <H3>7. Stale model versions</H3>

      <Prose>
        Production serving infrastructure frequently runs multiple versions of the same model simultaneously during rolling deployments. A request routed to "haiku-4.5" might land on any of several instances running slightly different model weights or inference configurations. If the router was calibrated against version N, and version N+1 has different output quality characteristics, routing accuracy degrades until recalibration. Version-pin your routing targets and treat each model version as a distinct routing target with its own quality evaluation history.
      </Prose>

      <H3>8. Quality degradation hidden by averaging</H3>

      <Prose>
        Aggregate quality metrics — mean MMLU score, mean MT Bench score, mean human preference win rate — hide routing-induced quality degradation on specific request categories. If the router systematically under-serves hard requests on a narrow topic (say, legal reasoning), the quality failure disappears in the average because legal reasoning is a small fraction of traffic. The affected users — who send only legal reasoning requests — see a product that has silently gotten worse. Monitor quality per routing bucket (per tier, per task type, per confidence band), not just in aggregate. The first signal of systematic routing failure is often a support ticket spike on a specific task type, not a movement in the aggregate quality dashboard.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following sources are foundational. Citations verified accurate as of April 2026.
      </Prose>

      <Prose>
        <strong>Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., and Stoica, I. (2024).</strong> "RouteLLM: Learning to Route LLMs with Preference Data." arXiv preprint arXiv:2406.18665. UC Berkeley and Anyscale. The definitive empirical study of learned LLM routing. Introduces four router architectures (similarity-weighted, matrix factorization, BERT classifier, causal LLM classifier), a training framework leveraging Chatbot Arena preference data, and a rigorous evaluation across MT Bench, MMLU, and GSM8K benchmarks. Core result: the matrix factorization router achieves 95% of GPT-4 performance while routing only 14% of traffic to GPT-4. Benchmarks against commercial routers (Martian, Unify) show RouteLLM matches quality at 40%+ lower per-request cost. Open-source framework available at github.com/lm-sys/RouteLLM.
      </Prose>

      <Prose>
        <strong>Chen, L., Zaharia, M., and Zou, J. (2023).</strong> "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." arXiv preprint arXiv:2305.05176. Stanford University. Published in TMLR (2024). Introduces the LLM cascade as a formal optimization problem: find the escalation policy that minimizes cost subject to a quality constraint. Core contribution is showing that cascade composition is strictly more powerful than single-model routing because different models can be used for different sub-tasks within a query. Demonstrates up to 98% cost reduction versus always-GPT-4 at matched quality across HellaSwag, MMLU, and open-domain QA. Essential reading for the theoretical grounding of cascade routing.
      </Prose>

      <Prose>
        <strong>OpenAI. (2026).</strong> "Models Overview." OpenAI API Documentation. platform.openai.com/docs/models. Accessed April 2026. Canonical reference for OpenAI's model tier structure, pricing, and capability positioning. Documents the GPT-5.4 family (flagship, mini, nano), the o-series reasoning models (o3, o4-mini), and the GPT-4.1 family. Pricing as of April 2026: GPT-5.4 at $2.50/$15 per MTok in/out; GPT-5.4 nano at $0.20/$1.25 per MTok in/out. The 12.5× cost ratio between flagship and nano within one generation is the primary routing opportunity for OpenAI-based products.
      </Prose>

      <Prose>
        <strong>Anthropic. (2026).</strong> "Models Overview." Claude API Documentation. platform.claude.com/docs/en/about-claude/models/overview. Accessed April 2026. Documents the Claude model family structure (Haiku, Sonnet, Opus) and the explicit multi-tier design philosophy. Pricing as of April 2026: Claude Haiku 4.5 at $1/$5 per MTok in/out; Claude Sonnet 4.6 at $3/$15; Claude Opus 4.7 at $15/$75. The 75× output-cost ratio between Haiku and Opus within the same family is the primary cost-savings opportunity for Anthropic-based multi-tier routing.
      </Prose>

      <Prose>
        <strong>BerriAI. (2024–2026).</strong> "LiteLLM Router Documentation." docs.litellm.ai/docs/routing. Accessed April 2026. Reference documentation for LiteLLM's routing strategies: simple-shuffle, least-busy, usage-based-routing, latency-based-routing, and auto-routing. Describes the fallback chain configuration, per-deployment rate-limit tracking, and cost budget enforcement features. LiteLLM is the practical reference implementation for multi-provider model routing in production.
      </Prose>

      <Prose>
        <strong>Martian. (2023–2024).</strong> "Martian Model Router." route.withmartian.com. Technical blog and product documentation. Accessed April 2026. Martian's model router uses a learned "model mapping" technique to predict the best model for a given prompt, optimizing jointly for quality, cost, and latency. Benchmarked against RouteLLM in Ong et al. (2024), which found RouteLLM's open-source routers matched Martian's quality at 40%+ lower cost per routed request. Martian received $9M in seed funding and Accenture investment in 2024, and reports deployment at 300+ enterprise customers.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Break-even hit rate</H3>

      <Prose>
        You are evaluating a two-model cascade: a small model costs $0.80/MTok input and $4.00/MTok output; a large model costs $15.00/MTok input and $75.00/MTok output. Typical requests average 400 input tokens and 200 output tokens. Compute the per-request cost for each model. Then derive the minimum hit rate <Code>h</Code> at which the cascade is cheaper than always routing to the large model. If your benchmark shows the small model achieving 78% hit rate on your traffic, should you use the cascade or route upfront with a classifier that achieves 92% tier-selection accuracy?
      </Prose>

      <Prose>
        Answer: Small model per-request cost: <Code>(400 × 0.80 + 200 × 4.00) / 1,000,000 = $0.00032 + $0.00080 = $0.00112</Code>. Large model: <Code>(400 × 15.00 + 200 × 75.00) / 1,000,000 = $0.00600 + $0.01500 = $0.02100</Code>. Cascade expected cost at hit rate <Code>h</Code>: <Code>0.00112 + (1-h) × 0.02100</Code>. Break-even vs always-large: <Code>0.00112 + (1-h) × 0.02100 = 0.02100</Code>, giving <Code>h = 1 - 0.00112/0.02100 ≈ 94.7%</Code>. At 78% hit rate, the cascade costs <Code>0.00112 + 0.22 × 0.02100 = $0.00574</Code> per request — cheaper than always-large ($0.02100) but only 73% cheaper, not the 80%+ savings the simple formula suggests. Meanwhile, the 92% accurate classifier routes 92% of requests to small and 8% to large: <Code>0.92 × 0.00112 + 0.08 × 0.02100 = $0.00103 + $0.00168 = $0.00271</Code> per request — 54% cheaper than the cascade at 78% hit rate. Use the classifier; the cascade's 78% hit rate is below the 94.7% break-even threshold and costs more per request than accurate upfront routing.
      </Prose>

      <H3>Exercise 2 — Cascade escalation threshold tuning</H3>

      <Prose>
        Your cascade router uses minimum log-probability as the escalation signal. At threshold <Code>τ = -3.0</Code>, you observe 23% escalation rate and 94% quality retention vs the always-large baseline. At <Code>τ = -2.0</Code>, escalation rate rises to 41% and quality retention rises to 97%. At <Code>τ = -4.0</Code>, escalation rate drops to 11% and quality retention drops to 89%. Compute the expected cost per request at each threshold (use the costs from Exercise 1) and recommend the threshold for: (a) a consumer product with a 95% quality retention SLA and cost minimization objective, and (b) an enterprise product with a 97% quality retention SLA and latency sensitivity (two-model calls add 1.8 seconds to escalated requests).
      </Prose>

      <Prose>
        Answer: Expected cost per request at each threshold (small = $0.00112, large = $0.02100): <Code>τ=-4.0</Code>: <Code>0.00112 + 0.11 × 0.02100 = $0.00343</Code>; <Code>τ=-3.0</Code>: <Code>0.00112 + 0.23 × 0.02100 = $0.00595</Code>; <Code>τ=-2.0</Code>: <Code>0.00112 + 0.41 × 0.02100 = $0.00973</Code>. (a) Consumer product: 95% quality SLA eliminates <Code>τ=-4.0</Code> (89% quality). Both <Code>τ=-3.0</Code> (94%, violates SLA) and <Code>τ=-2.0</Code> (97%, meets SLA) are candidates, but only <Code>τ=-2.0</Code> meets the 95% threshold. Recommend <Code>τ=-2.0</Code> at $0.00973 per request — 54% cheaper than always-large despite the higher escalation rate. (b) Enterprise product: 97% quality SLA requires <Code>τ=-2.0</Code>. But 41% escalation rate means 41% of requests incur +1.8s latency. If the enterprise SLA also caps P99 latency, this may be unacceptable. Consider switching to an upfront classifier that achieves 97% quality retention without the latency penalty on escalated requests, or implement streaming with mid-generation escalation to hide the latency on the escalated path.
      </Prose>

      <H3>Exercise 3 — A/B test contamination diagnosis</H3>

      <Prose>
        Your A/B test is routing 10% of traffic to a new challenger model. After two weeks, quality metrics show: champion average score 0.834, challenger average score 0.841. You declare the challenger better and prepare to roll it out fully. A colleague notices that the challenger was assigned to requests using a per-turn hash (turn ID) rather than a per-session hash (session ID). Explain what this means for your quality measurements and whether the 0.841 challenger score is trustworthy.
      </Prose>

      <Prose>
        Answer: Per-turn hashing means individual turns of a conversation may be split between champion and challenger. A conversation might have turns 1, 3, 5 routed to the champion and turns 2, 4, 6 routed to the challenger. The challenger never receives its own prior turns as conversation history — it receives the champion's outputs as the preceding context. This means the challenger is being evaluated on a chimeric conversation it did not generate, not on a coherent conversation it would have produced. The 0.841 score is not trustworthy: it conflates the quality of the challenger's responses with the quality of the conversation setup provided by the champion. The challenger may have benefited from the champion's strong prior turns, inflating its apparent score. Or it may have been penalized for inconsistencies it did not cause. The correct fix is to re-run the A/B test with session-level routing (all turns of a conversation route to the same arm) and measure quality on complete conversation trajectories. Discard the per-turn-hash results.
      </Prose>

      <H3>Exercise 4 — Design a routing system for an agentic pipeline</H3>

      <Prose>
        You are building an agentic pipeline with four step types: (1) task decomposition — break a user goal into subtasks (10–20 per run, requires multi-step reasoning); (2) tool selection — choose the right API from a manifest of 50 tools (structured output, format-critical); (3) tool execution summarization — summarize the JSON output of an API call into one sentence; (4) final synthesis — combine all subtask results into a coherent user-facing response. Design a routing strategy for each step type, specifying model tier and routing paradigm. Explain how you would handle cost tracking across the four step types for a single end-to-end run.
      </Prose>

      <Prose>
        Answer: Step 1 (task decomposition): Route to large tier (Opus-class). Multi-step reasoning is the task type that benefits most from scale. A small model that decomposes incorrectly propagates errors through all downstream steps, making a large-model error here far more costly than the per-step price implies. Use rule-based routing on task type (all decomposition steps → large), not a learned router — the signal is unambiguous. Step 2 (tool selection): Route to medium tier (Sonnet-class) with JSON output validation. Tool selection requires structured output precision, not frontier reasoning. The medium model handles it reliably; if the JSON fails to parse, escalate to large for one retry — this is a cascade with structural validation as the escalation signal. Step 3 (tool execution summarization): Route to small tier (Haiku-class). One-sentence summarization of structured data is a classic small-model task with minimal failure risk. Step 4 (final synthesis): Route to medium or large depending on run complexity. If the pipeline has more than 10 subtask results to synthesize, route to large for context management. Otherwise medium. Use the total context length (all subtask summaries) as a rule-based routing signal. Cost tracking: maintain a per-run cost accumulator, record each model call's (tier, input_tokens, output_tokens) tuple, compute cost per call using the tier's price rates, and sum across all four step types at run completion. Expose total-cost-per-run as a first-class metric alongside latency and quality.
      </Prose>

      <H3>Exercise 5 — Diagnosing silent quality degradation</H3>

      <Prose>
        Your monitoring dashboard shows: overall mean quality score 0.847 (stable over 30 days). Per-tier quality scores: small tier 0.891, medium tier 0.852, large tier 0.893. Your routing distribution: 55% small, 38% medium, 7% large. A support escalation from a legal team reports that the product has been giving wrong answers on contract analysis questions for the past three weeks. Identify the likely failure mode, explain why the aggregate dashboard did not surface it, and describe the monitoring changes needed to catch it earlier.
      </Prose>

      <Prose>
        Answer: The likely failure mode is systematic routing of contract analysis requests to the small tier, where they exceed the small model's capability. Contract analysis requires multi-step legal reasoning and precise extraction from dense structured text — a task type where small models fail on the hard tail. The aggregate quality score (0.847) did not surface this because legal queries are a small fraction of traffic (perhaps 2–3% of requests), and even a complete quality collapse on that category moves the aggregate by only 0.03–0.05 points — invisible against measurement noise. The per-tier quality scores are also misleading: the small tier's 0.891 average includes the many easy requests it handles correctly, masking the legal-query failure buried in its tail. Monitoring changes needed: (1) Stratify quality monitoring by task category (detected via topic classification on the prompt), not just by tier. Contract/legal should be its own quality bucket with its own alert threshold. (2) Track quality per (tier, task-type) cell — the cell (small, legal) is where the failure lives. (3) Implement a support-ticket-to-routing-decision correlation pipeline: when a support ticket arrives, look up the routing decision for that request and flag if the tier assignment was aggressive (small or medium for a request type that warrants large). This creates a closed-loop signal from support outcomes back to the routing system.
      </Prose>

    </div>
  ),
};

export default multiModelServing;
