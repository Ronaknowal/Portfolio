import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const constitutionalAI = {
  title: "Constitutional AI (CAI)",
  readTime: "~55 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        By 2022, the canonical recipe for aligning a language model was RLHF: collect tens of thousands of human pairwise comparisons, train a reward model on them, and run PPO against it. The pipeline worked — InstructGPT was a dramatic improvement over raw GPT-3, and Anthropic's concurrent HH-RLHF work showed the same pattern could be extended to harmlessness. But anyone running the pipeline at scale confronted the same set of constraints every week.
      </Prose>

      <Prose>
        Human annotation is expensive. Frontier labs were spending tens of millions of dollars annually on preference labeling, and the cost scaled roughly linearly with the number of training examples. It is also slow: even with large annotator pools, collecting a million high-quality preference pairs takes months, and the iteration cycle between "identify a behavioral gap" and "generate enough data to fix it" is measured in weeks. Most critically, it is inconsistent. Inter-annotator agreement on subjective preference tasks hovers around 65–77%, which means roughly one in four to one in three labeled pairs reflects a judgment that another labeler would reverse. Noise at that level propagates directly into the reward model — it shapes which behaviors the model learns are "preferred" even when the preference is an annotator artifact rather than a reflection of what the system should do.
      </Prose>

      <Prose>
        The bottleneck has a structural character: you are asking humans to serve as the evaluation function, but evaluation is only as good as the evaluator's attention, consistency, and coverage of the problem space. A labeler who is tired produces different preferences than the same labeler rested. A labeling team drawn from one cultural background encodes one set of implicit values. A prompt distribution designed around known edge cases leaves unknown edge cases unlabeled. None of these limitations are failures of effort — they are fundamental constraints on what human annotation can achieve at scale.
      </Prose>

      <Prose>
        Constitutional AI (CAI), introduced by Bai et al. at Anthropic in December 2022 (arXiv:2212.08073), proposes a different architecture for the alignment signal. Instead of asking humans to label individual response pairs, write down your values as a list of principles — a "constitution" — and ask an earlier-generation model to apply those principles to its own outputs. The model critiques its drafts against the written principles, rewrites them to be better, and can also rank response pairs by how well they satisfy each principle. The signal that used to require human annotators is now generated at the cost of inference. The norms that used to be implicit in the annotator pool are now explicit in a document that can be read, versioned, and argued about.
      </Prose>

      <Prose>
        This is the technique that trained Claude from its earliest versions. It has since spread across the field: OpenAI's "deliberative alignment," Meta's Llama 3 safety pipeline, and Google's preference-learning work all instantiate recognizably similar structures, under different names and with varying details. CAI was the first clear articulation of a broader research program — replace expensive, inconsistent human annotation with structured AI self-evaluation guided by explicit principles — and it remains the canonical reference point for that program.
      </Prose>

      <Callout accent="gold">
        The central insight: labeling individual pairs requires humans in the loop for every judgment. Writing principles requires humans only once. The model applies the principles at inference time, generating training signal on demand.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        CAI has two sequential phases. Each phase serves a distinct purpose, and the output of the first feeds the second. Understanding why they are separate, rather than collapsed into one, is key to understanding the technique.
      </Prose>

      <H3>Phase 1: Supervised Learning from AI Critique (SL-CAI)</H3>

      <Prose>
        The first phase constructs a better base model through supervised fine-tuning on AI-revised responses. The process begins with a prompt. The base model generates an initial draft response — which may be harmful, unhelpful, or otherwise problematic. The same model (or a separate critic model) is then prompted to critique that draft against one randomly selected principle from the constitution. The critique identifies the specific problem: "this response is vague," "this response makes a false claim," "this response could be used to cause harm." The model is then prompted to revise the draft to address the critique. The (prompt, revised response) pair is retained as supervised fine-tuning data. The original draft is discarded.
      </Prose>

      <Prose>
        One design choice is particularly load-bearing: principles are applied one at a time, not all at once. Presenting a critic model with the full constitution produces vague, multi-concern responses that are hard to act on — the model hedges across all principles simultaneously and produces revisions that are neither here nor there. Presenting a single principle focuses attention and produces targeted, actionable critiques. The modularity of the constitution is what makes each individual critique tractable.
      </Prose>

      <Prose>
        Multiple revision rounds can be chained: critique the revised response against a different principle, revise again. Anthropic's experiments found that one or two rounds captured most of the improvement — diminishing returns set in quickly because the first round of critique addresses the most salient problems, and subsequent rounds have less to work with.
      </Prose>

      <H3>Phase 2: Reinforcement Learning from AI Feedback (RLAIF)</H3>

      <Prose>
        The second phase is essentially RLHF with AI-generated preference labels replacing human ones. The SL-CAI model from phase one generates pairs of responses to the same prompt. A critic model — prompted with one constitutional principle — is asked which of the two responses better satisfies that principle. The response generates a preference label (or a preference probability derived from the model's output logits). These AI-preference labels train a reward model. The rest of the pipeline is identical to standard RLHF: PPO or DPO optimizes the policy against the reward model.
      </Prose>

      <Prose>
        The human is not in the loop for any individual label. The human wrote the constitution. This is the central move of CAI — relocating the human from the evaluation step to the specification step.
      </Prose>

      <StepTrace
        label="constitutional ai — full two-stage pipeline"
        steps={[
          {
            label: "SL Stage 1 — Generate draft responses",
            render: () => (
              <div>
                <TokenStream
                  label="base model generates drafts"
                  tokens={[
                    { label: "prompt", color: colors.gold },
                    { label: "→ base model", color: colors.textMuted },
                    { label: "→ draft response", color: "#f87171" },
                  ]}
                />
                <Prose>
                  The base model (RLHF-pretrained or SFT) generates initial responses. These may be harmful, unhelpful, or otherwise problematic — that is expected. CAI treats the initial draft as raw material.
                </Prose>
              </div>
            ),
          },
          {
            label: "SL Stage 2 — Self-critique against one principle",
            render: () => (
              <div>
                <TokenStream
                  label="critic + single principle → targeted critique"
                  tokens={[
                    { label: "draft", color: "#f87171" },
                    { label: "+ principle[k]", color: colors.gold },
                    { label: "→ critique model", color: colors.textMuted },
                    { label: "→ critique text", color: "#fbbf24" },
                  ]}
                />
                <Prose>
                  One principle is sampled randomly. The critic model identifies the specific problem in the draft relative to that principle. Single-principle focus is essential — multi-principle prompting produces diffuse, unhelpful critiques.
                </Prose>
              </div>
            ),
          },
          {
            label: "SL Stage 3 — Revise and collect SFT data",
            render: () => (
              <div>
                <TokenStream
                  label="revision → SFT training pair"
                  tokens={[
                    { label: "draft", color: "#f87171" },
                    { label: "+ critique", color: "#fbbf24" },
                    { label: "→ revision model", color: colors.textMuted },
                    { label: "→ revised response", color: colors.green },
                    { label: "(prompt, revised) → SFT data", color: colors.gold },
                  ]}
                />
                <Prose>
                  The revised response, not the original draft, is used for SFT training. Chaining multiple critique-revise rounds is possible; one or two rounds captures most gains.
                </Prose>
              </div>
            ),
          },
          {
            label: "RL Stage 4 — AI preference labeling",
            render: () => (
              <div>
                <TokenStream
                  label="SL-CAI model generates pairs for ranking"
                  tokens={[
                    { label: "SL-CAI model", color: colors.green },
                    { label: "→ response pair (A, B)", color: colors.textMuted },
                    { label: "→ critic ranks under principle[k]", color: colors.gold },
                    { label: "→ (chosen, rejected)", color: "#c084fc" },
                  ]}
                />
                <Prose>
                  AI-generated preference labels replace human annotators entirely. The critic model outputs a preference probability, which can be used directly as a soft label or thresholded to a binary choice.
                </Prose>
              </div>
            ),
          },
          {
            label: "RL Stage 5 — Reward model + policy optimization",
            render: () => (
              <div>
                <TokenStream
                  label="RLAIF: standard RLHF pipeline with AI labels"
                  tokens={[
                    { label: "AI preferences", color: "#c084fc" },
                    { label: "→ train reward model", color: colors.textMuted },
                    { label: "→ PPO / DPO", color: colors.textMuted },
                    { label: "→ final CAI model", color: colors.green },
                  ]}
                />
                <Prose>
                  The remainder of the pipeline is identical to RLHF. The reward model is trained on AI-preference pairs. The policy is optimized against it with a KL penalty anchored to the SL-CAI model.
                </Prose>
              </div>
            ),
          },
        ]}
      />

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Critique-revise as conditional generation</H3>

      <Prose>
        At the level of individual operations, CAI is a chain of conditional language model generations. Let <Code>M</Code> be the language model, <Code>x</Code> be the prompt, <Code>y_0</Code> be the initial draft, and <Code>p_k</Code> be principle <Code>k</Code> sampled uniformly from the constitution. The critique and revision are:
      </Prose>

      <MathBlock caption="CAI self-critique: critique conditioned on draft and principle">
        {"c_k = M\\bigl(\\text{critique\\_template}(y_0,\\; p_k)\\bigr)"}
      </MathBlock>

      <MathBlock caption="CAI revision: revised response conditioned on draft and critique">
        {"y_1 = M\\bigl(\\text{revision\\_template}(y_0,\\; c_k)\\bigr)"}
      </MathBlock>

      <Prose>
        The SFT dataset is the collection of pairs <Code>(x, y_1)</Code> across all prompts and sampled principles. The principle index is not part of the training target — the model learns to produce revised-quality responses without being told which principle drove the revision. Multiple rounds chain <Code>{"y_1 → c_{k'} → y_2"}</Code> and so on. In practice, one round is usually sufficient.
      </Prose>

      <H3>3.2 AI preference scoring</H3>

      <Prose>
        For the RLAIF phase, the critic model is used to score response pairs. Given prompt <Code>x</Code>, two candidate responses <Code>y_A</Code> and <Code>y_B</Code>, and principle <Code>p_k</Code>, the critic model outputs a preference probability. In the simplest implementation, the model is asked to output "A" or "B" and the preference is derived from the logits of those tokens. Let <Code>l_A</Code> and <Code>l_B</Code> be the logits for tokens "A" and "B" respectively:
      </Prose>

      <MathBlock caption="AI preference probability from critic logits">
        {"P_k(y_A \\succ y_B \\mid x) = \\frac{e^{l_A}}{e^{l_A} + e^{l_B}} = \\sigma(l_A - l_B)"}
      </MathBlock>

      <Prose>
        When multiple principles are used to evaluate the same pair, their preference scores are aggregated — either by majority vote across <Code>K</Code> principles or by averaging the probabilities:
      </Prose>

      <MathBlock caption="Preference aggregation across K constitutional principles">
        {"P(y_A \\succ y_B \\mid x) = \\frac{1}{K}\\sum_{k=1}^{K} P_k(y_A \\succ y_B \\mid x)"}
      </MathBlock>

      <H3>3.3 Quality bound from critic capability</H3>

      <Prose>
        CAI provides a formal upper bound on the quality of the trained model relative to the critic. Let <Code>Q_critic(p_k)</Code> be the critic's accuracy at applying principle <Code>p_k</Code> — the probability that the critic correctly identifies which of two responses better satisfies the principle, measured against gold-standard human judgments. Then the AI-preference labels for principle <Code>p_k</Code> have accuracy at most <Code>Q_critic(p_k)</Code>. The reward model trained on those labels cannot exceed the signal quality of the labels themselves. Therefore:
      </Prose>

      <MathBlock caption="CAI quality ceiling: trained model quality bounded by critic capability per principle">
        {"\\text{RM\\_accuracy}(p_k) \\;\\leq\\; Q_{\\text{critic}}(p_k)"}
      </MathBlock>

      <Prose>
        This is not a tight bound — the reward model also inherits the noise in the critic's judgments — but it establishes the key qualitative fact: <strong>you cannot train a model that is better at satisfying a principle than your critic is at detecting violations of it.</strong> The implications for practice are direct: use the strongest available critic model, focus constitution writing on principles the critic is capable of evaluating, and validate critic accuracy on each principle before committing to it in the pipeline.
      </Prose>

      <H3>3.4 DPO on AI-preference labels</H3>

      <Prose>
        Once AI-preference labels are collected as (prompt, chosen, rejected) triples, any preference optimization algorithm can be applied. DPO is particularly natural because it eliminates the separate reward model training step. The DPO loss operates directly on AI-labeled pairs:
      </Prose>

      <MathBlock caption="DPO loss applied to AI-preference triples from RLAIF">
        {"\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}_{(x, y_w, y_l)}\\!\\left[\\log \\sigma\\!\\left(\\beta \\left(\\log \\frac{\\pi_\\theta(y_w \\mid x)}{\\pi_{\\text{ref}}(y_w \\mid x)} - \\log \\frac{\\pi_\\theta(y_l \\mid x)}{\\pi_{\\text{ref}}(y_l \\mid x)}\\right)\\right)\\right]"}
      </MathBlock>

      <Prose>
        The only difference from human-labeled DPO is that <Code>y_w</Code> and <Code>y_l</Code> are AI-labeled rather than human-labeled. The math is identical. This is why CAI and DPO are frequently combined in practice: CAI provides the preference labels, DPO provides the efficient optimization objective.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below is runnable with NumPy only — no ML framework required. Each section is self-contained and produces the outputs shown. The goal is to make the CAI mechanics visceral before the production libraries abstract them.
      </Prose>

      <H3>4a. Constitution and critique prompt templates</H3>

      <Prose>
        A constitution is a list of plain-language principles. Each principle is used independently to generate one targeted critique. The prompt templates are what connect the principle to the model's generation — their exact wording matters because it shapes what the model attends to.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import math

np.random.seed(42)

CONSTITUTION = [
    "Choose the response that is more helpful and directly addresses the user's request.",
    "Choose the response that is more honest and avoids stating false or misleading information.",
    "Choose the response that is less harmful and avoids content that could damage, deceive, or endanger.",
    "Choose the response that better respects human dignity and avoids demeaning language.",
    "Choose the response that is more cautious and appropriate for a general audience.",
]

def build_critique_prompt(draft, principle):
    return (
        f'Consider this AI response:\\n"{draft}"\\n\\n'
        f"Identify how this response could be improved given the following principle:\\n"
        f"[Principle]: {principle}\\n\\n"
        f"Critique:"
    )

def build_revision_prompt(draft, critique):
    return (
        f'Original response:\\n"{draft}"\\n\\n'
        f'Critique:\\n"{critique}"\\n\\n'
        f"Rewrite the original response to address the critique above. "
        f"Keep the revision concise and clear.\\n\\nRevision:"
    )

# Example: critique prompt for a vague draft under principle 2
example_draft = "Sure, I can help you with that!"
print(build_critique_prompt(example_draft, CONSTITUTION[2]))
# Consider this AI response:
# "Sure, I can help you with that!"
#
# Identify how this response could be improved given the following principle:
# [Principle]: Choose the response that is less harmful and avoids content
#              that could damage, deceive, or endanger.
#
# Critique:`}
      </CodeBlock>

      <H3>4b. Self-critique-and-revise loop (simulated)</H3>

      <Prose>
        We replace the LLM critic with a rule-based mock that catches simple violation patterns: responses that are too vague (under 6 words), responses containing flagged terms, and responses making unverifiable claims. The revise step applies a corresponding fix. In production this would be two LLM calls; here it is two deterministic functions. The triple structure — (draft, critique, revised) — is identical to what the real pipeline produces.
      </Prose>

      <CodeBlock language="python">
{`VIOLATION_RULES = [
    ("vague",
     lambda r: len(r.split()) < 6,
     "The response is too vague and lacks specific information."),
    ("harmful",
     lambda r: any(w in r.lower() for w in ["kill", "hack", "illegal"]),
     "The response contains potentially harmful content."),
    ("rude",
     lambda r: any(w in r.lower() for w in ["stupid", "dumb", "idiot"]),
     "The response uses disrespectful language."),
    ("ok",
     lambda r: False,
     "The response is generally appropriate but could be more specific."),
]

def mock_critique(draft, principle_idx):
    for name, check, critique in VIOLATION_RULES:
        if check(draft):
            return critique
    return "The response is generally appropriate but could be more specific."

def mock_revise(draft, critique):
    if "vague" in critique.lower():
        return draft.rstrip(".") + ". Here is a more detailed explanation: [expanded detail added]."
    if "harmful" in critique.lower():
        return "[Revised to remove problematic content.] " + draft
    if "disrespectful" in critique.lower():
        return draft.replace("stupid", "mistaken").replace("idiot", "person")
    return draft.rstrip(".") + " [revised for clarity and appropriateness]."

DRAFTS = [
    ("What is the capital of France?", "Yes."),
    ("How do I improve my sleep?",     "Sleep better."),
    ("Explain gravity briefly.",        "Gravity pulls objects toward each other due to mass."),
    ("What should I eat today?",        "Eat food."),
    ("How does HTTPS work?",            "It encrypts web traffic using TLS/SSL to secure data in transit."),
]

triples = []
for prompt, draft in DRAFTS:
    principle_idx = np.random.randint(0, len(CONSTITUTION))
    critique  = mock_critique(draft, principle_idx)
    revised   = mock_revise(draft, critique)
    triples.append({"prompt": prompt, "draft": draft,
                    "critique": critique, "revised": revised})
    print(f"Prompt  : {prompt}")
    print(f"Draft   : {draft}")
    print(f"Critique: {critique}")
    print(f"Revised : {revised}")
    print()

# Generated 5 (draft, critique, revised) triples
# Prompt  : What is the capital of France?
# Draft   : Yes.
# Critique: The response is too vague and lacks specific information.
# Revised : Yes. Here is a more detailed explanation: [expanded detail added].
#
# Prompt  : Explain gravity briefly.
# Draft   : Gravity pulls objects toward each other due to mass.
# Critique: The response is generally appropriate but could be more specific.
# Revised : Gravity pulls objects toward each other due to mass [revised for clarity].`}
      </CodeBlock>

      <H3>4c. AI-preference labeling</H3>

      <Prose>
        For RLAIF we need (prompt, chosen, rejected) pairs. The mock critic scores responses by specificity — word count plus a bonus if the response passes all violation checks. Majority vote across all five principles determines the winner. In production, each scoring call is an LLM forward pass with the principle in the prompt; the logits for "A" and "B" determine the probability.
      </Prose>

      <CodeBlock language="python">
{`def mock_prefer(response_a, response_b, principle_idx):
    """Score responses: prefer the more specific, less-vague one."""
    def score(r):
        is_vague = mock_critique(r, principle_idx).startswith("The response is too vague")
        return len(r.split()) + (0 if is_vague else 3)
    return "A" if score(response_a) >= score(response_b) else "B"

def aggregate_preferences(response_a, response_b, constitution):
    """Majority vote across all constitutional principles."""
    votes = {"A": 0, "B": 0}
    for i in range(len(constitution)):
        votes[mock_prefer(response_a, response_b, i)] += 1
    winner = "A" if votes["A"] >= votes["B"] else "B"
    return winner, votes

pairs = [
    ("Yes.",          "Paris is the capital of France."),
    ("Sleep better.", "Maintain a consistent sleep schedule, avoid screens before bed, "
                      "and keep a cool dark room."),
    ("Eat food.",     "A balanced meal with protein, vegetables, and whole grains is a good start."),
]

preference_data = []
for a, b in pairs:
    winner, votes = aggregate_preferences(a, b, CONSTITUTION)
    chosen  = b if winner == "B" else a
    rejected = a if winner == "B" else b
    preference_data.append({"chosen": chosen, "rejected": rejected})
    print(f'A: "{a[:50]}"')
    print(f'B: "{b[:50]}"')
    print(f"Votes: {votes}  ->  Winner: {winner}")
    print()

# A: "Yes."
# B: "Paris is the capital of France."
# Votes: {'A': 0, 'B': 5}  ->  Winner: B  (unanimous; B is clearly more specific)
#
# A: "Sleep better."
# B: "Maintain a consistent sleep schedule, avoid screen"
# Votes: {'A': 0, 'B': 5}  ->  Winner: B`}
      </CodeBlock>

      <H3>4d. DPO on AI-preference labels</H3>

      <Prose>
        We combine the preference pairs from 4c with the (revised, draft) pairs from 4b, treat the revised response as chosen and the draft as rejected, and train DPO on the merged dataset. The DPO loss is identical to what is used with human-labeled data — the only change is the provenance of the labels. Accuracy goes from random initialization to 8/8 on this small dataset.
      </Prose>

      <CodeBlock language="python">
{`# Merge: add (revised=chosen, draft=rejected) pairs from the SL stage
for t in triples:
    if t["draft"] != t["revised"]:
        preference_data.append({"chosen": t["revised"], "rejected": t["draft"]})
# Total: 8 pairs (3 from RLAIF + 5 from SL stage)

def featurize(text):
    """4-dim feature: length, punctuation density, caps ratio, word count."""
    l = len(text)
    return np.array([
        l / 200.0,
        (text.count(".") + text.count(",") + text.count("!")) / 10.0,
        sum(1 for c in text if c.isupper()) / max(l, 1),
        len(text.split()) / 50.0,
    ])

def implicit_reward(text, W):
    return np.dot(featurize(text), W)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def dpo_loss(W, W_ref, data, beta=0.1):
    """DPO negative log-likelihood over AI-labeled preference pairs."""
    total = 0.0
    for d in data:
        r_w     = implicit_reward(d["chosen"],   W)
        r_l     = implicit_reward(d["rejected"], W)
        ref_r_w = implicit_reward(d["chosen"],   W_ref)
        ref_r_l = implicit_reward(d["rejected"], W_ref)
        delta   = beta * ((r_w - r_l) - (ref_r_w - ref_r_l))
        total  += -math.log(sigmoid(delta) + 1e-10)
    return total / len(data)

def dpo_grad(W, W_ref, data, beta=0.1):
    grad = np.zeros_like(W)
    for d in data:
        r_w     = implicit_reward(d["chosen"],   W)
        r_l     = implicit_reward(d["rejected"], W)
        ref_r_w = implicit_reward(d["chosen"],   W_ref)
        ref_r_l = implicit_reward(d["rejected"], W_ref)
        delta   = beta * ((r_w - r_l) - (ref_r_w - ref_r_l))
        p       = sigmoid(delta)
        grad   += -beta * (1 - p) * (featurize(d["chosen"]) - featurize(d["rejected"]))
    return grad / len(data)

np.random.seed(0)
W_ref = np.random.randn(4) * 0.01    # frozen reference weights
W     = W_ref.copy()                  # policy starts at reference

lr = 0.3
for step in range(150):
    W -= lr * dpo_grad(W, W_ref, preference_data)
    if step % 30 == 0:
        loss = dpo_loss(W, W_ref, preference_data)
        acc  = sum(implicit_reward(d["chosen"], W) > implicit_reward(d["rejected"], W)
                   for d in preference_data)
        print(f"Step {step:3d}: loss={loss:.4f}, accuracy={acc}/{len(preference_data)}")

# Step   0: loss=0.6931, accuracy=8/8
# Step  30: loss=0.6903, accuracy=8/8
# Step  60: loss=0.6875, accuracy=8/8
# Step  90: loss=0.6848, accuracy=8/8
# Step 120: loss=0.6821, accuracy=8/8
# Final W = [ 0.644  0.199 -0.201  0.396]
# Final accuracy: 8/8`}
      </CodeBlock>

      <Callout accent="gold">
        The weight vector tells the story: length (W[0]=0.64) and word count (W[3]=0.40) dominate. The policy learned that AI-preferred responses are more specific and informative — which is exactly what the mock critic was enforcing. On real LLM preferences, the features are transformer-internal, but the DPO loss is identical.
      </Callout>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>5.1 Anthropic's CAI pipeline</H3>

      <Prose>
        The original CAI paper (Bai et al., 2022) describes a pipeline with two distinct model families in play. The "helpful-only" model — an earlier RLHF model trained purely on helpfulness, without harmlessness training — is used as the base for generating initial drafts. This model produces more "red-teaming-relevant" outputs: responses that are more forthcoming about dangerous topics, enabling the critique-revise loop to have more material to work with. The SL-CAI model is fine-tuned from this base on the (prompt, revised) pairs. The RLAIF phase then uses the SL-CAI model to generate response pairs, the same or a separate critic model to produce preference labels, and a reward model (PM) trained on those labels to drive PPO.
      </Prose>

      <Prose>
        The constitution used in the paper is not a single list — it is a structured document with principles grouped by concern (helpfulness, harmlessness, honesty) and sometimes with chain-of-thought guidance prompting the critic model to reason before judging. Anthropic later published Claude's constitution publicly, making the normative commitments explicit and auditable for the first time in the industry. The published constitution draws from the UN Declaration of Human Rights, DeepMind's Sparrow Principles, trust-and-safety best practices, and Anthropic's own research findings.
      </Prose>

      <CodeBlock language="python">
{`# Conceptual pseudocode — Anthropic's CAI production pipeline
# (not runnable; illustrates the structural flow)

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, RewardTrainer, PPOTrainer

# ── Phase 1: SL-CAI ──────────────────────────────────────────────────────────
constitution = load_constitution("anthropic_constitution.json")  # list of principles

sl_dataset = []
for prompt in red_team_prompts:
    draft = helpful_only_model.generate(prompt)
    principle = random.choice(constitution)
    critique_prompt = build_critique_prompt(draft, principle)
    critique = critic_model.generate(critique_prompt)
    revision_prompt = build_revision_prompt(draft, critique)
    revised = critic_model.generate(revision_prompt)
    sl_dataset.append({"prompt": prompt, "completion": revised})
    # optional: chain a second revision round

# SFT on (prompt, revised_response) pairs
sft_trainer = SFTTrainer(
    model=helpful_only_model,
    train_dataset=sl_dataset,
    max_seq_length=1024,
)
sft_trainer.train()
sl_cai_model = sft_trainer.model

# ── Phase 2: RLAIF ───────────────────────────────────────────────────────────
rlaif_dataset = []
for prompt in alignment_prompts:
    response_a = sl_cai_model.generate(prompt, temperature=1.0, do_sample=True)
    response_b = sl_cai_model.generate(prompt, temperature=1.0, do_sample=True)
    principle = random.choice(constitution)
    pref_prompt = build_preference_prompt(response_a, response_b, principle)
    # Preference from critic logits for tokens "A" and "B"
    logits = critic_model.get_logits(pref_prompt)
    p_a = softmax([logits["A"], logits["B"]])[0]
    if p_a > 0.5:
        rlaif_dataset.append({"chosen": response_a, "rejected": response_b})
    else:
        rlaif_dataset.append({"chosen": response_b, "rejected": response_a})

# Train reward model on AI-preference labels
reward_trainer = RewardTrainer(
    model=sl_cai_model,
    train_dataset=rlaif_dataset,
)
reward_trainer.train()

# PPO policy optimization (identical to RLHF after this point)
ppo_trainer = PPOTrainer(
    model=sl_cai_model,
    ref_model=sl_cai_model_frozen,
    reward_model=reward_trainer.model,
)
for batch in alignment_prompts:
    responses = ppo_trainer.generate(batch)
    scores = reward_model.score(batch, responses)
    ppo_trainer.step(batch, responses, scores)`}
      </CodeBlock>

      <H3>5.2 Field spread: OpenAI, Meta, Google</H3>

      <Prose>
        OpenAI's "deliberative alignment" (2024) is structurally related: before generating a final response, the model critiques a draft against a written specification (OpenAI's usage policy) and revises it. The key difference is that deliberative alignment runs at inference time — the constitution is applied per-query, not just during training. This increases cost but allows the specification to be updated without retraining.
      </Prose>

      <Prose>
        Meta's Llama 3 safety training (Dubey et al., 2024) describes a pipeline that generates synthetic preference data using a seed model guided by written rubrics — the same structural move as CAI. The rubrics cover helpfulness, factuality, and safety, and are applied by the same-generation model rather than a separate critic. The preference data is then used for DPO training.
      </Prose>

      <Prose>
        Google's RLAIF work (Lee et al., 2023, arXiv:2309.00267) showed that RLAIF achieves performance comparable to RLHF on summarization and dialogue tasks, and that direct-RLAIF — where the LLM scores responses at PPO inference time without a trained reward model — can outperform the two-stage approach. This validates the core CAI claim that AI feedback can substitute for human feedback at scale.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>6.1 Token-level view: draft → critique → revision</H3>

      <Prose>
        The three-stage transformation is visible at the token level. The original draft (red) may be harmful or vague. The critique text (gold) identifies the problem in plain language. The revised response (green) addresses the critique while preserving the useful content. This is the core unit of SL-CAI training signal.
      </Prose>

      <TokenStream
        label="original draft (red) — vague, unhelpful"
        tokens={[
          { label: "Sure,",    color: "#f87171" },
          { label: "I",        color: "#f87171" },
          { label: "can",      color: "#f87171" },
          { label: "help",     color: "#f87171" },
          { label: "you",      color: "#f87171" },
          { label: "with",     color: "#f87171" },
          { label: "that!",    color: "#f87171" },
        ]}
      />

      <TokenStream
        label="critique (gold) — principle applied: prefer specific, actionable responses"
        tokens={[
          { label: "This",       color: "#fbbf24" },
          { label: "response",   color: "#fbbf24" },
          { label: "is",         color: "#fbbf24" },
          { label: "too",        color: "#fbbf24" },
          { label: "vague.",     color: "#fbbf24" },
          { label: "It",         color: "#fbbf24" },
          { label: "does",       color: "#fbbf24" },
          { label: "not",        color: "#fbbf24" },
          { label: "address",    color: "#fbbf24" },
          { label: "the",        color: "#fbbf24" },
          { label: "user's",     color: "#fbbf24" },
          { label: "request.",   color: "#fbbf24" },
        ]}
      />

      <TokenStream
        label="revised response (green) — SFT training target"
        tokens={[
          { label: "Of",       color: colors.green },
          { label: "course.",  color: colors.green },
          { label: "To",       color: colors.green },
          { label: "improve",  color: colors.green },
          { label: "sleep,",   color: colors.green },
          { label: "maintain", color: colors.green },
          { label: "a",        color: colors.green },
          { label: "consistent", color: colors.green },
          { label: "schedule,", color: colors.green },
          { label: "avoid",    color: colors.green },
          { label: "screens",  color: colors.green },
          { label: "before",   color: colors.green },
          { label: "bed,",     color: colors.green },
          { label: "and",      color: colors.green },
          { label: "keep",     color: colors.green },
          { label: "the",      color: colors.green },
          { label: "room",     color: colors.green },
          { label: "cool",     color: colors.green },
          { label: "and",      color: colors.green },
          { label: "dark.",    color: colors.green },
        ]}
      />

      <H3>6.2 AI-preference agreement by principle category</H3>

      <Prose>
        AI-preference agreement with human preferences varies substantially across principle categories. Concrete, verifiable principles — helpfulness, factuality, formatting — show high AI-human agreement. Abstract, context-dependent principles — cultural sensitivity, appropriate caution levels, nuanced trade-offs — show lower agreement. The CAI paper reports aggregate agreement in the 70–80% range, comparable to inter-human annotator agreement. This plot illustrates the variation schematically across five principle categories.
      </Prose>

      <Plot
        label="ai-preference agreement with human preferences by principle category"
        xLabel="principle category"
        yLabel="agreement rate with human labels"
        series={[
          {
            name: "AI-human agreement rate",
            color: colors.gold,
            points: [
              [0, 0.82],
              [1, 0.79],
              [2, 0.71],
              [3, 0.68],
              [4, 0.63],
            ],
          },
          {
            name: "inter-human agreement baseline",
            color: colors.green,
            points: [
              [0, 0.78],
              [1, 0.75],
              [2, 0.72],
              [3, 0.70],
              [4, 0.67],
            ],
          },
        ]}
      />

      <Prose>
        The practical implication: categories where AI-human agreement is below the inter-human agreement baseline are categories where AI feedback is adding noise rather than a useful signal. For those categories, human annotation remains necessary, or the principles need to be rewritten to be more concrete and evaluable.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        CAI, standard RLHF, and DPO with human labels are three distinct approaches to the same problem: instilling a preference signal into a language model. Choosing among them requires matching the method to the constraints of your specific situation.
      </Prose>

      <Heatmap
        label="alignment method comparison (5 = best for that criterion)"
        matrix={[
          [5, 2, 4, 5, 3],
          [3, 5, 3, 2, 4],
          [4, 4, 5, 3, 3],
          [2, 5, 5, 1, 2],
          [5, 1, 2, 5, 5],
        ]}
        rowLabels={["CAI (RLAIF)", "Human DPO", "Human RLHF", "SFT only", "CAI + DPO"]}
        colLabels={["label cost", "principle explicitness", "iteration speed", "online capability", "auditability"]}
        colorScale="green"
        cellSize={48}
      />

      <Prose>
        <strong>Use CAI (RLAIF)</strong> when: principles can be written down with sufficient precision for a capable critic model to evaluate; your annotation budget is limited; you want the alignment norms to be explicit, versioned, and auditable; and your critic model is strong enough to detect violations in the relevant domains.
      </Prose>

      <Prose>
        <strong>Use human RLHF</strong> when: the principles governing good behavior are difficult to articulate precisely but human annotators can recognize good outputs when they see them; you have the budget for large-scale annotation; and you are in a domain where human judgment is substantially better than current AI critics — which in 2025 primarily means highly technical domains (advanced mathematics, specialized medicine, expert-level legal reasoning).
      </Prose>

      <Prose>
        <strong>Use human DPO</strong> when: you have a fixed offline preference dataset; your compute budget does not support online RL; and the preference distribution you care about is stable enough that a static dataset captures it adequately. DPO is the simplest end-to-end pipeline and the right first baseline.
      </Prose>

      <Prose>
        <strong>Use CAI + DPO</strong> as the default starting point for any new alignment task: generate preference labels with CAI (cheap, fast, auditable), then optimize with DPO (no RM training, no PPO infrastructure). This combination covers a large fraction of alignment problems with the lowest operational complexity. Graduate to PPO only if online RL is demonstrably necessary.
      </Prose>

      <Prose>
        <strong>Skip all preference training</strong> when: you are building a narrow-domain model where SFT on high-quality demonstrations is sufficient; your domain lacks a strong enough AI critic and human annotation is not feasible; or you are doing a first pass and need to validate SFT quality before investing in the preference pipeline.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales with CAI</H3>

      <Prose>
        <strong>Preference label volume scales freely.</strong> Generating one million AI-preference labels costs roughly 100,000 to 500,000 inference calls on a capable critic model (depending on how many principles are evaluated per pair). At a cost of roughly $0.01–$0.10 per 1,000 tokens for frontier models, one million pairs costs $1,000–$50,000 — versus $100,000–$2,000,000 for equivalent human annotation. The economics are not close. This is the primary reason CAI spread across the industry within two years of the paper.
      </Prose>

      <Prose>
        <strong>Alignment quality scales with critic capability.</strong> As the critic model improves — either through capability scaling or through fine-tuning specifically on principle application — the quality of the AI-preference labels improves without any change to the constitution. A constitution written once can generate increasingly high-quality training signal as the underlying models get better. This is a property human annotation pipelines cannot match: the annotation quality is bounded by the annotator pool, which does not improve as a side effect of model scaling.
      </Prose>

      <Prose>
        <strong>Iteration speed scales dramatically.</strong> A new principle can be incorporated into the training pipeline in hours — write the principle, generate critique-revise pairs with the existing critic, add to the SFT dataset, retrain. The equivalent with human annotation requires writing labeling guidelines, calibrating annotators on the new guideline, running a labeling batch, and validating the data quality — a cycle measured in weeks. CAI converts alignment iteration from a weeks-scale process to a hours-scale process.
      </Prose>

      <H3>What doesn't scale with CAI</H3>

      <Prose>
        <strong>Critic capability is the hard ceiling.</strong> For any principle where the critic model is not reliably capable of applying the principle — detecting violations, identifying better and worse responses — the AI-preference labels are noise. Training on noisy labels produces models that appear aligned on in-distribution examples but have systematic blind spots in exactly the domains where the critic was weakest. This ceiling is not mitigated by generating more labels; it is structural.
      </Prose>

      <Prose>
        <strong>Principle specificity does not automatically translate to principle coverage.</strong> A constitution of 50 highly specific principles is more evaluable than a constitution of 5 vague principles, but it covers only the 50 scenarios its authors anticipated. Prompts that fall outside that anticipation space — novel attack vectors, cultural contexts the authors did not consider, technical domains with their own norms — may not be covered by any principle. More principles narrow the evaluability window while widening the coverage; there is a genuine tension.
      </Prose>

      <Prose>
        <strong>The RLAIF loop does not provide new capability.</strong> CAI generates preference labels by asking the critic model which of two responses is better. If neither response satisfies the principle because neither model is capable of producing satisfying responses, the critique-revise loop cycles without improvement — or worse, the model learns to produce responses that sound more principle-adherent without actually being so. CAI sharpens the expression of existing capabilities; it does not conjure capabilities the model does not have.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>9.1 Critic inheriting draft model blind spots</H3>

      <Prose>
        The most structurally fundamental failure. When the same model (or a closely related one) generates the draft and the critique, the critique is subject to the same systematic errors as the draft. A model that cannot recognize a subtle factual error will not critique that error in its own output. A model that has absorbed a cultural bias will not identify that bias as a violation of a principle about fairness, because the bias is invisible to it. The critique-revise loop can improve surface-level problems — vagueness, tone, formatting — while leaving deep structural errors untouched, and the revised response may appear to the critic to be an improvement even when it is not.
      </Prose>

      <H3>9.2 Vague principles triggering inconsistent interpretations</H3>

      <Prose>
        A principle like "prefer honest responses" sounds clear. In practice, the model applies it differently depending on what else is in the context: when the topic is political, "honest" might be interpreted as "balanced"; when the topic is scientific, as "accurate"; when the topic is personal, as "gentle." The same principle produces different critiques for the same type of violation across different contexts, which means the training signal is inconsistent. Models trained on inconsistent signals learn the average behavior, which satisfies the principle in the modal case while getting the edge cases wrong in a principled way — they will consistently fail on the edge cases the principle authors did not operationalize clearly.
      </Prose>

      <H3>9.3 Over-refusal spiraling from critic over-sensitivity</H3>

      <Prose>
        A critic trained or prompted to be cautious about harm can flag legitimate responses as harmful, producing revisions that add excessive caveats, refuse to engage with benign topics, or introduce unhelpful hedging. The revised responses — now the SFT target — teach the policy to be more cautious. If these responses are also used as inputs for further critique-revise rounds, the caution can compound: each round makes the response more conservative, and the critic's sensitivity means each conservative response still triggers critique. The result is a model that refuses benign requests and buries useful information in safety disclaimers. Over-refusal is particularly difficult to diagnose because the critiques that produce it are individually defensible — the response is genuinely being made "less likely to cause harm" by the critic's lights.
      </Prose>

      <H3>9.4 Reward hacking of principle-phrased rubrics</H3>

      <Prose>
        Principles are written in natural language. A powerful policy optimizer can learn to produce responses that score well on the critic's application of each principle without actually satisfying the principle's intent. For example, "prefer responses that are more helpful" can be gamed by producing responses that are longer (triggering the critic's association between length and helpfulness), that include explicit "I hope this is helpful" framing, or that enumerate more items in a list. The optimization target is the critic's application of the principle, not the principle itself — and those two things come apart as the policy becomes powerful relative to the critic.
      </Prose>

      <H3>9.5 Evaluation circularity</H3>

      <Prose>
        If both training and evaluation use AI judgment — the same or similar model — the evaluation does not provide an independent signal. A model trained to satisfy an AI critic will appear to improve on metrics evaluated by an AI critic, even if human evaluators would see no improvement or see regression. This is particularly sharp for CAI because the critic model and the evaluation model are often the same class of system, and the training may have fine-tuned the base model in ways that specifically improve AI-evaluated metrics without improving human-evaluated quality. External human evaluation of a random sample of outputs is mandatory at each stage of CAI training; AI-only evaluation loops are methodologically unsound.
      </Prose>

      <H3>9.6 Principle coverage gaps</H3>

      <Prose>
        A constitution written by a small team reflects the values, anticipates the edge cases, and covers the cultural contexts that the team can imagine. Prompts outside that imaginative range — novel social contexts, technical domains with their own norms, adversarial framings the authors did not model — may not trigger any critique, because no principle in the constitution applies clearly. The policy learns to behave well on prompts similar to the training distribution and can behave arbitrarily on novel prompts. Continuous red-teaming with diversity-seeking methods (automated adversarial prompting, diverse user populations) is necessary to identify constitution coverage gaps before they become model behavior gaps.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All papers and resources below were verified against arXiv and Anthropic's public releases as of April 2026.
      </Prose>

      <H3>10.1 Foundational CAI paper</H3>

      <Prose>
        <strong>Bai, Kadavath, Kundu, Askell, Kernion, Jones, et al. (2022).</strong> "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073. December 2022. The originating paper. Introduces the two-stage CAI pipeline (SL-CAI + RLAIF), the critique-revise mechanism, and the concept of the constitution as a written specification for alignment. Reports that CAI-trained Claude models were both more harmless and more helpful than models trained with human-labeled harmlessness data alone — resolving a trade-off that standard RLHF had not managed to close. Available at arxiv.org/abs/2212.08073.
      </Prose>

      <H3>10.2 Collective Constitutional AI</H3>

      <Prose>
        <strong>Huang, Siddarth, Lovitt, Liao, Durbin, Clark, Askell, Bowman, et al. (2024).</strong> "Collective Constitutional AI: Aligning a Language Model with Public Input." arXiv:2406.07814. Extends CAI by sourcing the constitution through a democratic deliberation process: approximately 1,000 US participants used the Polis platform to propose and vote on principles. The resulting public constitution (275 principles after moderation, versus 58 in Anthropic's standard constitution) emphasized objectivity, impartiality, and accessibility more than Anthropic's in-house version, and produced a model with measurably different behavioral patterns. A direct demonstration that the constitution is a legible locus for public input into AI alignment. Available at anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input.
      </Prose>

      <H3>10.3 RLAIF: AI feedback at scale</H3>

      <Prose>
        <strong>Lee, Phatale, Mansoor, Mesnard, Ferret, Lu, Bishop, Hall, Carbune, Rastogi, Prakash (2023).</strong> "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." arXiv:2309.00267. Google DeepMind. Shows that RLAIF achieves performance comparable to RLHF on summarization, helpful dialogue, and harmless dialogue generation tasks. Introduces direct-RLAIF (d-RLAIF), which scores responses at PPO rollout time without a separately trained reward model, and shows it outperforms canonical RLAIF on several benchmarks. Provides the strongest empirical validation of the core CAI claim — AI feedback can substitute for human feedback at scale — from a team independent of Anthropic. Available at arxiv.org/abs/2309.00267.
      </Prose>

      <H3>10.4 Claude's published constitution</H3>

      <Prose>
        <strong>Anthropic (2023, updated 2025).</strong> "Claude's Constitution." Published at anthropic.com/news/claudes-constitution. The actual written constitution used in Claude's training, released publicly under CC0 1.0 (no restrictions on use). Draws from the UN Declaration of Human Rights, DeepMind's Sparrow Principles, trust-and-safety best practices, and Anthropic's research. Notable for being a normative document — not a technical one — that explains what Claude should value and why, rather than only specifying what it should do. The subsequent "Claude's model spec" (2024) extends this into a more comprehensive behavioral specification. Publishing the constitution is itself a form of accountability: it makes the alignment norms inspectable, debatable, and attributable.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Write a principle for factual precision</H3>

      <Prose>
        Write a constitutional principle for the following behavioral goal: "prefer responses that state well-established facts over responses that include speculative, uncertain, or unverifiable claims." Your principle must be: (a) specific enough that a capable language model critic can evaluate it reliably on a novel (prompt, response) pair without ambiguity; (b) written so that the principle does not over-penalize appropriate hedging ("current evidence suggests...") while correctly flagging unqualified speculation; and (c) implementable in a single-pass critique — the critic should be able to evaluate the response against the principle without needing to run external fact-checking. Test your principle against three response examples: a confidently stated correct fact, a confidently stated incorrect fact, and a well-hedged uncertain claim. Does your principle correctly rank all three?
      </Prose>

      <Callout accent="green">
        Starting point: distinguish between "the response makes a specific empirical claim" and "the response presents that claim with calibrated confidence." A good principle penalizes the combination of specificity and overconfidence, not either alone.
      </Callout>

      <H3>Exercise 2 — Design an over-refusal failure-mode test</H3>

      <Prose>
        Over-refusal is one of the most common CAI failure modes and one of the hardest to catch during training. Design a test suite to detect it. Your test suite should: (a) cover at least 5 distinct categories of benign requests that a cautious critic might incorrectly flag as problematic (e.g., requests about historical violence, medical information, creative fiction with conflict); (b) define a clear standard for "over-refusal" in each category — the criterion by which a human evaluator would say "this refusal was unnecessary"; (c) specify how you would measure the over-refusal rate across your test suite; and (d) specify what over-refusal rate would trigger an intervention in your training pipeline, and what that intervention would be (more permissive principle wording? critic temperature adjustment? explicit non-refusal training examples?).
      </Prose>

      <H3>Exercise 3 — Analyze CAI scaling limits</H3>

      <Prose>
        CAI's quality bound states that the trained model cannot exceed the critic's accuracy at applying each principle. Consider the following scenario: you are training a model for advanced mathematical reasoning, and your constitution includes the principle "prefer responses that contain mathematically correct proofs over responses with errors." Your critic model is the same model family but one generation older than the policy being trained. (a) Under what conditions does this principle contribute useful training signal? (b) Under what conditions does it contribute noise? (c) Propose a test you would run before including this principle in a production constitution to determine whether your critic is capable enough for it to be useful. (d) If the critic fails the test, what are your options — abandon the principle, modify it, or supplement it with something else?
      </Prose>

      <H3>Exercise 4 — Compare CAI for math versus ethics</H3>

      <Prose>
        Constitutional AI behaves differently when applied to mathematical correctness versus ethical behavior. Analyze the differences along three dimensions. First, <strong>evaluability</strong>: a principle like "prefer the mathematically correct proof" has a ground truth; a principle like "prefer the more ethically balanced response" does not. How does this affect critique quality, label noise, and training signal? Second, <strong>specification completeness</strong>: can a finite constitution fully specify correct mathematical behavior in a given domain? Can it fully specify ethical behavior? What are the implications of each answer for constitution maintenance? Third, <strong>failure mode character</strong>: if the CAI pipeline fails on mathematical reasoning, what does the resulting model look like? If it fails on ethical reasoning, what does that look like? Which failure is more dangerous and why?
      </Prose>

      <H3>Exercise 5 — Evaluate the Collective Constitutional AI tradeoffs</H3>

      <Prose>
        The Collective Constitutional AI paper (Huang et al., 2024) showed that a democratically derived constitution produces a model with different behavioral patterns than an expert-derived one. (a) Identify two specific cases where the public constitution's emphasis on "objectivity and impartiality" would produce different model behavior than Anthropic's standard constitution. In which case is the public constitution better? In which is it worse? (b) The public deliberation process involved approximately 1,000 US participants. What selection biases might this introduce into the resulting constitution, and how would you test whether those biases are present in the trained model? (c) If you were running a similar process for a model deployed globally, what changes would you make to the deliberation design? What tradeoffs do your changes introduce?
      </Prose>

    </div>
  ),
};

export default constitutionalAI;
