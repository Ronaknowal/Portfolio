import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { Plot } from "../../components/viz";

const multiModelServing = {
  title: "Multi-Model Serving & Model Routing",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        The cheapest token served is the one answered by a smaller model. A well-designed multi-model stack routes each incoming request to the smallest model that can handle it well — turning "one expensive model for everything" into "a spectrum of models with traffic routed smartly across them." Done right, this is the single largest cost-reduction lever available to an LLM product. It does not require a better model, a new architecture, or a change to the user experience. It requires understanding your workload and building a routing layer that matches requests to the right tier.
      </Prose>

      <Prose>
        The underlying intuition is straightforward: different requests have different difficulty, and model capability scales with cost. Sending easy requests to expensive models wastes compute. Sending hard requests to cheap models degrades quality. Routing well means identifying which request belongs in which bucket before the model call happens, not after.
      </Prose>

      <H2>Why one model doesn't fit all</H2>

      <Prose>
        A request to summarize a paragraph is handled equally well by a 7B and a 70B model; routing it to the 70B wastes roughly 10× the compute. A request requiring deep multi-step reasoning might need the 70B — or a reasoning-tuned variant — and routing it to a 7B produces a worse product. The quality-cost curve is not flat. It is steeply nonlinear in the hard cases: small models and large models are nearly indistinguishable on easy tasks, and dramatically different on difficult ones.
      </Prose>

      <Prose>
        This matters because most production traffic is not uniformly distributed across difficulty. In a typical chat product, a large fraction of requests are simple: short answers, factual lookups, rephrasing, classification decisions, or extraction from short text. These are well within the capability of a small model. A much smaller fraction are genuinely hard: multi-constraint reasoning, ambiguous instructions, tasks requiring synthesis across long contexts, or high-stakes outputs where errors are costly. Average-case routing — picking one model and applying it to all traffic — either overpays for easy traffic or underserves hard traffic. The gap between those two failure modes is the opportunity that tiered routing captures.
      </Prose>

      <H2>Tiers in practice</H2>

      <Prose>
        A production multi-model stack typically serves three to five tiers. Each tier corresponds to a capability band and a cost band. The tier boundaries are not fixed; they shift as new models are released and per-token prices change. But the structure is stable: a tiny tier for the simplest tasks, a small tier for routine work, a medium tier as the default for general chat, a large tier for hard cases, and specialized models for domain-specific tasks.
      </Prose>

      <Prose>
        Tiny models (1–3B parameters) handle classification, simple extraction, and very short conversational turns — tasks where the output is short and the decision space is narrow. Small models (7–8B) cover the bulk of routine chat, mid-difficulty question-answering, and high-volume batch jobs where cost per request matters more than raw capability. Medium models (30–70B) serve as the workhorse tier: general-purpose chat, content generation, common tool use, and most of what a consumer product delivers. Large frontier models (100B+ parameters or equivalent capability) handle the tail of hard requests — complex reasoning, high-stakes outputs, ambiguous tasks where a smaller model's errors would be noticed. Specialized models — math-tuned, code-tuned, vision-capable — sit alongside this size axis, routed to by task type rather than difficulty alone.
      </Prose>

      <CodeBlock>
{`tier          use cases                         typical cost $/MTok    latency p50
tiny          classification, extraction         0.05 / 0.25           <200ms
small         routine chat, bulk                 0.15 / 0.60           300-600ms
medium        general chat, tool use             1-3 / 10-15           500-1500ms
large         hard reasoning, high-stakes        10-20 / 60-100        1-5s
reasoning     math/code with verification        5-15 / 30-80          3-60s`}
      </CodeBlock>

      <Prose>
        The cost columns are illustrative, calibrated to approximate 2025 commercial pricing. What matters is the ratio: large-tier output tokens cost roughly 150–400× what tiny-tier output tokens cost. If 60% of your traffic can be served at the tiny or small tier without quality loss, the weighted average cost per request falls dramatically — typically 40–70% below the cost of routing everything to the medium or large tier.
      </Prose>

      <H2>Routing approaches — from simplest to smartest</H2>

      <Prose>
        There are three common routing patterns. They differ in where the routing decision lives, how much engineering they require, and how accurate they are.
      </Prose>

      <Prose>
        <strong>Static routing.</strong> The client explicitly chooses the model; the router dispatches. The product exposes model tiers as user-facing options — a "fast" mode and a "quality" mode, or a tiered pricing plan where lower-cost plans access smaller models. The routing logic is trivially simple, and errors in routing are the user's problem rather than the system's. This works well when users have enough context to self-select and when you want the cost structure to be transparent.
      </Prose>

      <Prose>
        <strong>Heuristic routing.</strong> Rules derived from prompt features decide the tier. Prompt length, presence of code blocks, math notation, tool call requirements, user-plan metadata — each of these is a noisy signal about request difficulty. Combining them into a rule set gets you 60–80% accuracy on obvious cases. The failure mode is edge cases: a short prompt that happens to require deep reasoning, or a long prompt that is mechanically repetitive. Heuristic routers degrade gracefully on the easy end and fail quietly on the hard end, which means quality regressions are not always visible in aggregate metrics.
      </Prose>

      <Prose>
        <strong>Classifier routing.</strong> A small model — typically 100M–500M parameters — is trained to predict which tier a given request needs. The classifier runs before the main model call, adds a small fixed latency, and returns a tier assignment with a confidence score. Better accuracy than heuristics; requires investment in curating training data and running ongoing evaluations as the request distribution drifts.
      </Prose>

      <CodeBlock language="python">
{`class TieredRouter:
    def __init__(self, classifier, fallback_tier="medium"):
        self.classifier = classifier  # small model, ~100M params
        self.fallback_tier = fallback_tier

    def route(self, request):
        # Fast classifier decides which tier this request probably needs
        features = self.extract_features(request)
        predicted_tier, confidence = self.classifier.predict(features)

        # If confidence is low, default to a safer (larger) tier
        if confidence < 0.85:
            return self.fallback_tier
        return predicted_tier

    def extract_features(self, request):
        return {
            "prompt_tokens": len(request.prompt),
            "has_code": bool(re.search(r"\`\`\`", request.prompt)),
            "has_math": bool(re.search(r"\\$|\\\\(", request.prompt)),
            "has_tool_call": request.tools is not None,
            "user_tier": request.user_metadata.get("plan"),
        }`}
      </CodeBlock>

      <Prose>
        The confidence threshold in the example deserves attention. When the classifier is uncertain, the right default is usually to route up — to a larger, more capable tier — rather than down. Routing a hard request to a small model creates a visible quality failure. Routing an easy request to a medium model wastes money quietly. For most products, the asymmetry favors conservative fallback.
      </Prose>

      <H2>Cascaded fallback</H2>

      <Prose>
        A different approach to the same problem: instead of predicting difficulty upfront, try the small model first and escalate only when the output looks wrong. This is cascaded fallback, and it works well when the "looks wrong" check is cheap and reliable.
      </Prose>

      <Prose>
        Three common escalation signals. First, low output log-probabilities — if the model's own confidence in its completion is unusually low, it may be struggling with a hard prompt. Second, structural validation failure — the model was supposed to return valid JSON, or fill a specific template, and the output doesn't parse. Third, a separate verifier model — a small model specifically trained to judge output quality, which evaluates the primary model's response and returns a pass or escalate decision.
      </Prose>

      <Prose>
        The tradeoff is explicit: cascading trades latency for cost. On a fallback, you pay for two model calls — the failed small-model attempt plus the successful large-model call. On the majority of requests that the small model handles well, you pay for only one cheap call. Whether the math works depends on your fallback rate. If 20% of requests escalate and the small-tier call costs 10% of the large-tier call, the average cost is approximately <Code>0.8 × small + 0.2 × (small + large) ≈ small + 0.2 × large</Code>. At that ratio, you still save roughly 80% versus routing everything to the large tier. Push the fallback rate above 40-50% and the savings shrink toward zero while latency increases for half your users.
      </Prose>

      <H2>Cost savings in practice</H2>

      <Prose>
        Real deployments consistently report 40–70% cost reductions from tiered routing without user-visible quality loss on standard chat traffic. The savings are not uniform; they depend heavily on workload shape. A product with high request diversity — many different task types, difficulty levels, and user intents — saves more from routing than a narrow API with a homogeneous request distribution. A product where most requests are already in the hard category saves less because there are fewer easy requests to divert to cheaper tiers.
      </Prose>

      <Plot
        label="tiered routing cost impact (illustrative)"
        width={520}
        height={240}
        xLabel="% traffic routed to small/tiny tiers"
        yLabel="% of single-model cost"
        series={[
          { name: "single tier (70B)", points: [[0, 100], [30, 100], [60, 100], [90, 100]] },
          { name: "tiered routing", points: [[0, 100], [30, 75], [60, 48], [90, 23]] },
        ]}
      />

      <Prose>
        The steep reduction in the tiered-routing curve assumes the router is accurate. A router that confidently misclassifies hard requests as easy ones pushes difficult traffic to the small tier and tanks quality — and because individual quality failures are hard to surface in aggregate cost metrics, the degradation can go undetected. Building a routing system without ongoing quality evaluation is the operational version of this mistake. Cost goes down, and no one notices that it also stopped working correctly on the hard tail.
      </Prose>

      <H3>Specialization beyond size</H3>

      <Prose>
        Model size is one routing axis. Task domain is a second, orthogonal axis. A math-tuned model consistently outperforms a general model of the same or larger size on formal mathematical reasoning. A code model wins on code completion, bug detection, and program synthesis. A vision model is the only option for image inputs. Many production stacks end up with ten to twenty specialized models behind a single endpoint, routing by task type in addition to difficulty tier.
      </Prose>

      <Prose>
        The routing logic for specialization is usually simpler than difficulty routing — detecting whether a request involves code or images is more reliable than estimating whether it requires deep reasoning. The operational burden is different: each specialized model needs its own quality evaluation pipeline, its own deployment, and its own update cycle. A math model that improves needs to be validated against the math routing path specifically. Mistakes in the specialization router produce systematic failures on an entire task category, which are more visible but also more damaging than the scattered failures that come from difficulty misclassification.
      </Prose>

      <H2>Operational complexity</H2>

      <Prose>
        The honest tradeoff: multi-model serving adds real complexity. More model weights to manage means more GPU memory allocation, more serving infrastructure, more deployment pipelines, and more checkpoints to update when a better model is released. More eval pipelines means that each tier needs quality benchmarks, regression tests, and monitoring — the number of evaluation surfaces grows with the number of models. Routing quality is itself a system that needs monitoring: if the router drifts as the request distribution changes, routing accuracy decays silently.
      </Prose>

      <Prose>
        There is also a subtler cost: customer-perceived inconsistency. A user whose request gets routed to the medium tier one day and the large tier the next will notice a quality difference. If routing decisions are invisible to users — which they usually are — they cannot predict or control the quality of responses they receive. For most consumer products this is acceptable; for professional tools and API products where consistency is part of the value proposition, it is a real concern.
      </Prose>

      <Prose>
        A single-model endpoint is simpler to operate, simpler to evaluate, and simpler to explain to users. The break-even point is workload-specific. For narrow, high-volume APIs with diverse request difficulty and a mature evaluation infrastructure, multi-tier routing wins decisively. For low-volume exploratory products, early-stage teams, or highly homogeneous workloads, the simplicity of one model often wins — not because the cost savings aren't real, but because the operational overhead isn't worth it yet.
      </Prose>

      <Callout accent="gold">
        Multi-model serving isn't just a cost lever. It's a commitment to evaluating and maintaining a portfolio of models where once you had one.
      </Callout>

      <Prose>
        Routing is how you turn a zoo of models into a product. The next topic covers what happens at the other end of the routing decision — rate limiting, quotas, and fairness across users sharing a model pool.
      </Prose>
    </div>
  ),
};

export default multiModelServing;
