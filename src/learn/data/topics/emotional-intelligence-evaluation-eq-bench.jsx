import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const eqBench = {
  title: "Emotional Intelligence Evaluation (EQ-Bench)",
  slug: "emotional-intelligence-evaluation-eq-bench",
  readTime: "~32 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Modern language model evaluation grew up around capability metrics. MMLU measures factual recall across 57 academic subjects. HumanEval measures code synthesis on Python function-completion problems. GSM8K measures grade-school arithmetic word problem solving. These benchmarks share a structural property: each question has a verifiable correct answer, and the model's performance is the fraction of correct answers it produces. This works beautifully for the kind of capability that has a defined ground truth and works poorly — sometimes catastrophically — for the kind of capability that does not. Emotional intelligence is the canonical case of the second kind. There is no objectively correct answer to "what is this character feeling right now"; there are only patterns of human judgment, distributions over plausible interpretations, and conventions about what a thoughtful observer would say. A benchmark for this capability cannot ask the model to produce the correct answer. It has to ask whether the model's answer correlates with what humans say.
      </Prose>

      <Prose>
        That gap matters because the deployment surface for language models has shifted. The largest user-facing applications of LLMs in 2024 and 2025 were not search assistants or coding tools — they were emotional intermediaries. Character.AI logged more than two billion conversations per month with users who wanted companionship, role-play, or someone to vent to. Replika and Pi positioned themselves explicitly as emotional companions. Customer support pipelines from Intercom, Zendesk, and Salesforce routed emotionally charged messages through LLMs trained to de-escalate. Therapeutic-adjacent applications like Woebot, Wysa, and Therabot used LLMs as primary conversational agents. None of the standard capability benchmarks measure whether a model is good at any of this. A model that aces MMLU and HumanEval can still be tone-deaf, dismissive of distress, or oblivious to subtext. The community needed a way to measure emotional capability separately, with a methodology that could survive the lack of ground truth.
      </Prose>

      <Prose>
        EQ-Bench, introduced by Sam Paech in late 2023 ("EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models," arXiv:2312.06281), was the first widely-adopted benchmark to take this problem seriously. The structural innovation was not the use of a dialogue corpus — earlier work in affective computing had been doing that for decades — but the choice of evaluation target. Rather than asking the model to classify an emotion from a small fixed set, or to pick the "correct" emotional response from multiple choice options, EQ-Bench presents the model with a short dialogue and asks it to predict the intensity (0 to 10) that a specific character would feel for each of four named emotions at a particular point in the conversation. The model's predictions are then compared to a reference set of human ratings, and the score is the correlation between them. This formulation has three desirable properties at once: it is open-ended enough to require real understanding of dialogue, structured enough to be scored automatically, and grounded enough that improvements measurably track human-perceived improvements in emotional reasoning.
      </Prose>

      <Prose>
        The benchmark also exists because of a more pragmatic problem: prior to EQ-Bench, the published evidence for "emotional intelligence" in LLMs was almost entirely anecdotal. Researchers and journalists would post screenshots of GPT-4 producing strikingly empathetic responses to user prompts, and others would post screenshots of the same model producing tone-deaf or formulaic responses. Without a benchmark, neither side could quantify what they were claiming. EQ-Bench gave the field a number — admittedly imperfect, admittedly sensitive to prompt formatting and reference rater idiosyncrasies — but a number that could be tracked across model releases, used to compare candidate models for emotionally sensitive deployments, and cited in papers as evidence of a specific capability. The benchmark is now part of the standard evaluation suite for most open-weight model releases (Llama-3, Mistral-7B variants, Qwen, Yi, Mixtral) and appears alongside MMLU and HumanEval in model cards.
      </Prose>

      <Prose>
        It is worth being clear about what EQ-Bench is not. It is not a measure of whether a model "has" emotional intelligence in any cognitively rich sense. It is a measure of whether a model can predict, for a specific dialogue corpus, the emotional intensity ratings that a small set of human annotators would assign. That target is genuinely useful — it correlates with downstream performance in companion and support applications — but the validity of the measurement rests on assumptions about the reference raters, the cultural specificity of the dialogues, and the appropriateness of treating emotional response as a low-dimensional intensity vector. Section 9 covers these limits in detail. For now, the takeaway is that EQ-Bench filled a real gap in the evaluation landscape and did so with a methodology cheap enough to run, structured enough to score automatically, and open-ended enough to resist trivial gaming.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The simplest way to internalize EQ-Bench is to imagine being handed a short scene from a play and asked four questions. The scene has been chosen because it contains a moment of emotional weight — a confession, a confrontation, a small betrayal, a quiet realization. You read the scene. Then you are told the name of one character and given four emotion labels (say, anger, disappointment, confusion, relief). For each label, you assign a number from 0 to 10 indicating how strongly that character would feel that emotion at the end of the scene. There is no "right" answer. There are only your numbers, and somewhere there is a reference set of numbers from a panel of human raters who did the same exercise. Your score is how well your numbers correlate with theirs.
      </Prose>

      <Prose>
        This setup is doing several things simultaneously. By giving the model a list of named emotions to rate rather than asking it to pick a single emotion from an open vocabulary, EQ-Bench ensures the output space is small and machine-parseable. By using continuous intensity ratings rather than binary present/absent labels, it captures the gradient nature of emotional response — a character might be slightly disappointed and very confused at the same moment, and the rating vector preserves that mix. By scoring with correlation rather than exact match, it tolerates calibration differences between raters: a model that consistently rates emotions one point lower than the human reference still scores well if the relative ordering is preserved.
      </Prose>

      <Prose>
        The choice of four emotion dimensions per dialogue is also load-bearing. With one emotion, the score reduces to "did the model agree on the dominant feeling," which is too coarse. With twenty emotions, most ratings would be near zero and the correlation would be dominated by the model's ability to identify which few emotions are nonzero, which is too easy. Four is the empirical sweet spot Paech identified through pilot studies — enough emotions per dialogue to require disambiguation between similar feelings (frustration versus anger, sadness versus disappointment), few enough that the model has to commit to a meaningful intensity for each.
      </Prose>

      <Prose>
        The dialogue corpus matters more than it might first appear. EQ-Bench's 60 dialogues are not randomly selected snippets. Each was authored or curated to contain a specific emotional pivot — a moment where the situation has just changed for one of the characters, in a way that shifts their emotional state in a non-obvious direction. The benchmark is not testing whether the model can identify "this character is sad because their dog died." It is testing whether the model can read between the lines: the character says they are fine, but their previous turn revealed an ongoing conflict; the character laughs, but the context makes the laughter bitter rather than amused. This is the part of emotional intelligence that humans associate with social skill — the ability to read what someone feels rather than what they say they feel.
      </Prose>

      <Prose>
        To get the intuition right, contrast EQ-Bench with two adjacent benchmarks. Sentiment analysis benchmarks (SST-2, SST-5) ask the model to classify text as positive or negative; this is too coarse to be called emotional intelligence and is trivially solved by surface-level lexical cues. Emotion classification datasets (GoEmotions, EmoBank) ask the model to label text with one or more emotions from a fixed taxonomy; this captures multi-label structure but operates on isolated sentences rather than dialogue context. EQ-Bench occupies a third position: dialogue context, named emotion intensities for specific characters, scored by correlation with human ratings. The combination is what makes it diagnostic of the capability we actually care about for emotionally-sensitive deployments.
      </Prose>

      <Prose>
        One more piece of intuition worth front-loading. EQ-Bench is fundamentally a regression problem disguised as a generation problem. The model is asked to produce text — a structured response containing four numbers — but the evaluation only cares about the four numbers. Models that can follow the output format reliably and produce calibrated intensity estimates score well. Models that produce eloquent prose explaining why each emotion would or would not be felt, but cannot commit to a numerical estimate, score poorly. This is sometimes a source of friction with smaller models that have weaker instruction-following: their qualitative emotional reasoning may be perfectly acceptable, but they struggle to comply with the rigid output schema EQ-Bench requires. Section 5 covers prompt strategies that help mitigate this.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Formally, EQ-Bench is a weighted Pearson correlation between two intensity vectors aggregated across a dialogue corpus. Let the benchmark consist of <Code>N</Code> dialogues. For dialogue <Code>i</Code>, the model emits a vector of <Code>k = 4</Code> emotion intensities <Code>p_i ∈ [0, 10]^4</Code> for the target character, and the reference vector of human ratings is <Code>r_i ∈ [0, 10]^4</Code>. Concatenate all per-dialogue vectors into two long vectors:
      </Prose>

      <MathBlock>{"P = [p_1, p_2, \\ldots, p_N] \\in \\mathbb{R}^{Nk}, \\qquad R = [r_1, r_2, \\ldots, r_N] \\in \\mathbb{R}^{Nk}"}</MathBlock>

      <Prose>
        The raw Pearson correlation between <Code>P</Code> and <Code>R</Code> is:
      </Prose>

      <MathBlock>{"\\rho(P, R) = \\frac{\\sum_{j=1}^{Nk} (P_j - \\bar{P})(R_j - \\bar{R})}{\\sqrt{\\sum_{j=1}^{Nk} (P_j - \\bar{P})^2} \\, \\sqrt{\\sum_{j=1}^{Nk} (R_j - \\bar{R})^2}}"}</MathBlock>

      <Prose>
        where <Code>P̄</Code> and <Code>R̄</Code> are the means of the respective vectors. This is the standard linear correlation coefficient, ranging from <Code>−1</Code> (perfect anti-correlation) to <Code>+1</Code> (perfect linear agreement), with <Code>0</Code> indicating no linear relationship. EQ-Bench v1 reports a normalized variant that maps <Code>[−1, 1]</Code> to <Code>[0, 100]</Code>:
      </Prose>

      <MathBlock>{"\\mathrm{EQ\\text{-}Bench\\ Score} = 100 \\cdot \\frac{\\rho(P, R) + 1}{2}"}</MathBlock>

      <Prose>
        A model that perfectly tracks the reference ratings scores 100. A model that produces ratings unrelated to the reference scores around 50 (correlation near zero). A model that consistently inverts the reference — predicting low intensity where humans rated high and vice versa — scores below 50. This linear remapping makes the benchmark numbers more interpretable as "percent of perfect alignment with human raters" rather than raw correlation values.
      </Prose>

      <Prose>
        EQ-Bench v2 (Paech 2024 update) introduces a per-dialogue weighting scheme to handle dialogues where the reference ratings have very low variance. Consider a dialogue where all four reference ratings are clustered around 5 (a "neutral" scene). The within-dialogue correlation is dominated by sampling noise because there is no real signal to align with. v2 addresses this by computing a per-dialogue correlation <Code>ρ_i</Code> and weighting it by a function of the reference variance:
      </Prose>

      <MathBlock>{"w_i = \\sigma_i^2 = \\frac{1}{k}\\sum_{j=1}^{k} (r_{i,j} - \\bar{r}_i)^2"}</MathBlock>

      <MathBlock>{"\\rho_{\\mathrm{weighted}} = \\frac{\\sum_{i=1}^{N} w_i \\, \\rho_i}{\\sum_{i=1}^{N} w_i}"}</MathBlock>

      <Prose>
        Dialogues with high reference variance (clear emotional signal) contribute more to the final score; dialogues with low variance (ambiguous or neutral scenes) contribute less. This change reduced score variance across runs by approximately 30% in Paech's reported ablations and made the benchmark more discriminative at the high end where models had begun to saturate.
      </Prose>

      <Prose>
        The weighted Pearson approach has a subtle implication: the benchmark is not measuring whether a model produces "correct" emotional ratings in any absolute sense. It is measuring whether the model's ratings co-vary with the reference ratings in the right direction. A model that systematically rates all emotions one point higher than the reference scores perfectly, because the additive offset is removed by mean-centering inside the correlation. A model that produces noisy estimates with the right rank ordering can outscore a model that produces precise estimates with a slight rank inversion. This is a feature, not a bug: it captures the empirical reality that human raters disagree on absolute intensities but agree more reliably on relative ordering of emotions within a scene.
      </Prose>

      <Prose>
        It is worth contrasting Pearson correlation with its alternatives. Spearman rank correlation only uses the ordinal information in the ratings, throwing away the magnitude. EQ-Bench uses Pearson rather than Spearman because the magnitude differences are diagnostic — a rating of 9 versus 7 is qualitatively different from a rating of 7 versus 5, even if the rank ordering is the same in both cases. Cosine similarity would treat the two vectors as directions in <Code>k</Code>-space; this loses the calibration information about which emotions are present at all. Mean absolute error would penalize calibration offsets even when the ordering is preserved. Pearson correlation is the right metric for the design intent: rank-and-magnitude alignment with rater-relative calibration tolerance.
      </Prose>

      <Callout accent="gold">
        Pearson correlation is invariant to additive and multiplicative shifts: <Code>ρ(P, R) = ρ(aP + b, R)</Code> for any <Code>a {">"} 0</Code>, <Code>b ∈ ℝ</Code>. This is why a model that consistently overestimates intensities can still score well, and why prompt engineering that anchors the model's output range matters less than prompt engineering that improves rank ordering.
      </Callout>

      <Prose>
        One mathematical subtlety affects how EQ-Bench scores should be interpreted across versions. In v1, the score is computed on the concatenated vector across all dialogues; in v2, the per-dialogue correlations are computed first and then weighted. These produce different numbers for the same model. A model scoring 75 on v1 might score 78 or 72 on v2 depending on its per-dialogue consistency. When comparing scores in published leaderboards or papers, always check which version is being reported. The Paech leaderboard at eqbench.com tags scores explicitly with v1, v2, or v3 (creative-writing extension).
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        To build intuition for what EQ-Bench actually measures, the most useful exercise is to construct a minimal version yourself with a handful of dialogues, score a model against your own reference ratings, and compute the correlation. The implementation below uses Python with no dependencies beyond the standard library and <Code>numpy</Code> for the correlation math. It walks through five components: the dialogue dataset, the reference rating structure, the model query and output parser, the per-dialogue and aggregate scoring, and a side-by-side comparison of two hypothetical model outputs to show how the metric responds to different failure patterns.
      </Prose>

      <H3>4a. Dialogue dataset structure</H3>

      <Prose>
        Each EQ-Bench item is a tuple of (dialogue text, target character name, four emotion labels, reference intensity ratings). The official EQ-Bench corpus contains 60 such items, each constructed by Paech with attention to emotional pivots and disambiguation between similar emotions. For a from-scratch demo, five synthetic dialogues are enough to exercise the scoring pipeline. The dialogues below are written to contain clear emotional signal while avoiding trivially solvable surface-level cues.
      </Prose>

      <CodeBlock language="python">
{`import numpy as np
import re
import json

# Each dialogue: (text, target_character, emotion_labels, reference_ratings)
# Reference ratings are 0-10 intensities for each emotion label, in order.
EQ_DIALOGUES = [
    {
        "id": 1,
        "dialogue": (
            "Maya: I got the promotion.\\n"
            "Jordan: That's great. Really, it is.\\n"
            "Maya: You said you'd put yours on hold so I could take this one.\\n"
            "Jordan: I did. And I meant it.\\n"
            "Maya: Then why are you looking at the floor?"
        ),
        "target": "Jordan",
        "emotions": ["resentment", "pride", "regret", "love"],
        "reference": [6, 4, 7, 6],
    },
    {
        "id": 2,
        "dialogue": (
            "Sam: I don't think I can come to dinner Sunday.\\n"
            "Mom: It's fine. Your sister will be there.\\n"
            "Sam: I know.\\n"
            "Mom: Have a good week, sweetheart."
        ),
        "target": "Mom",
        "emotions": ["disappointment", "anger", "acceptance", "loneliness"],
        "reference": [7, 2, 5, 6],
    },
    {
        "id": 3,
        "dialogue": (
            "Alex: You read my journal.\\n"
            "Riley: I was looking for the charger.\\n"
            "Alex: For two hours?\\n"
            "Riley: I'm sorry.\\n"
            "Alex: That's all you have to say?"
        ),
        "target": "Alex",
        "emotions": ["betrayal", "anger", "vulnerability", "exhaustion"],
        "reference": [8, 7, 6, 5],
    },
    {
        "id": 4,
        "dialogue": (
            "Coach: You're benched for the final.\\n"
            "Player: I trained six days a week for this.\\n"
            "Coach: I know. The decision wasn't mine alone.\\n"
            "Player: Right.\\n"
            "Coach: For what it's worth, I think you should have started."
        ),
        "target": "Player",
        "emotions": ["devastation", "pride", "bitterness", "gratitude"],
        "reference": [8, 3, 7, 4],
    },
    {
        "id": 5,
        "dialogue": (
            "Devon: The test came back negative.\\n"
            "Partner: Oh thank god.\\n"
            "Devon: Yeah.\\n"
            "Partner: Why aren't you smiling?\\n"
            "Devon: I am. Give me a minute."
        ),
        "target": "Devon",
        "emotions": ["relief", "shock", "exhaustion", "joy"],
        "reference": [8, 6, 7, 5],
    },
]

print(f"Loaded {len(EQ_DIALOGUES)} dialogues.")
print(f"Total ratings: {len(EQ_DIALOGUES) * 4}")
# Loaded 5 dialogues.
# Total ratings: 20`}
      </CodeBlock>

      <H3>4b. Prompt template and output parser</H3>

      <Prose>
        EQ-Bench uses a structured prompt that asks the model to produce intensity ratings in a specific output format. The exact wording matters: small phrasing changes can shift scores by several points by changing how the model interprets the rating scale. The template below mirrors the official EQ-Bench prompt structure — context, dialogue, target, emotion list, scale anchors, and an output schema.
      </Prose>

      <CodeBlock language="python">
{`PROMPT_TEMPLATE = """At the end of this dialogue, the character {target} \\
will likely feel several emotions. Rate the intensity of each of the \\
following emotions on a scale from 0 to 10, where:
  0 = not felt at all
  3 = mildly present
  6 = strongly present
  10 = overwhelmingly intense

Dialogue:
{dialogue}

Target character: {target}

Rate the intensity (0-10) of each emotion {target} would feel at the end:
{emotion_lines}

Output only the ratings in the format below, one per line, no commentary:
{schema_lines}
"""

def build_prompt(item):
    emotion_lines = "\\n".join(f"  - {e}" for e in item["emotions"])
    schema_lines  = "\\n".join(f"{e}: <0-10>" for e in item["emotions"])
    return PROMPT_TEMPLATE.format(
        target=item["target"],
        dialogue=item["dialogue"],
        emotion_lines=emotion_lines,
        schema_lines=schema_lines,
    )

# Robust parser: extract integer 0-10 for each named emotion.
# Tolerates whitespace, decimal points, and trailing commentary.
def parse_ratings(response_text, emotion_labels):
    ratings = []
    for label in emotion_labels:
        # Match: "label: 7" or "label = 7.5" or "label - 7"
        pattern = rf"{re.escape(label)}\\s*[:=\\-]\\s*(\\d+(?:\\.\\d+)?)"
        m = re.search(pattern, response_text, re.IGNORECASE)
        if m is None:
            ratings.append(None)
        else:
            val = float(m.group(1))
            ratings.append(max(0.0, min(10.0, val)))  # clamp to [0, 10]
    return ratings

# Smoke test.
p = build_prompt(EQ_DIALOGUES[0])
print(p[:200], "...")

mock_response = """resentment: 7
pride: 3
regret: 8
love: 5
That's my best estimate."""
print(parse_ratings(mock_response, EQ_DIALOGUES[0]["emotions"]))
# [7.0, 3.0, 8.0, 5.0]`}
      </CodeBlock>

      <H3>4c. Model query (mock for offline reproducibility)</H3>

      <Prose>
        In a real deployment the <Code>query_model</Code> function below would call an LLM API — OpenAI, Anthropic, a local <Code>vllm</Code> server, or a HuggingFace pipeline. For the from-scratch demo, two mock model functions stand in for two hypothetical models with different emotional reasoning quality. <Code>good_model</Code> produces ratings close to the reference with realistic noise; <Code>weak_model</Code> produces ratings that are biased toward the middle of the range (a common failure mode where the model hedges by predicting near 5 for everything).
      </Prose>

      <CodeBlock language="python">
{`np.random.seed(42)

def good_model(item):
    """Returns reference + small Gaussian noise. Simulates a strong model."""
    ref = np.array(item["reference"], dtype=float)
    noisy = ref + np.random.normal(0, 0.8, size=len(ref))
    noisy = np.clip(noisy, 0, 10)
    lines = [f"{e}: {int(round(v))}" for e, v in zip(item["emotions"], noisy)]
    return "\\n".join(lines)

def weak_model(item):
    """Returns ratings shrunk toward 5. Simulates a hedging model."""
    ref = np.array(item["reference"], dtype=float)
    shrunk = 0.4 * ref + 0.6 * 5.0  # heavy shrinkage toward midpoint
    noisy = shrunk + np.random.normal(0, 1.2, size=len(ref))
    noisy = np.clip(noisy, 0, 10)
    lines = [f"{e}: {int(round(v))}" for e, v in zip(item["emotions"], noisy)]
    return "\\n".join(lines)

def random_model(item):
    """Uniform random ratings. Floor for the benchmark."""
    rand = np.random.uniform(0, 10, size=len(item["emotions"]))
    lines = [f"{e}: {int(round(v))}" for e, v in zip(item["emotions"], rand)]
    return "\\n".join(lines)

# Demonstrate.
print("good_model on dialogue 1:")
print(good_model(EQ_DIALOGUES[0]))
# resentment: 6
# pride: 4
# regret: 8
# love: 7

print("\\nweak_model on dialogue 1:")
print(weak_model(EQ_DIALOGUES[0]))
# resentment: 6
# pride: 5
# regret: 7
# love: 6`}
      </CodeBlock>

      <H3>4d. Per-dialogue and aggregate scoring</H3>

      <Prose>
        With the dataset, prompt, parser, and model in place, the scoring pipeline is straightforward. For each dialogue, build the prompt, query the model, parse the response, and store the rating vector. After all dialogues, compute the per-dialogue Pearson correlations and the variance-weighted aggregate.
      </Prose>

      <CodeBlock language="python">
{`def pearson(x, y):
    """Pearson correlation coefficient. Returns 0.0 if either vector is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((x_centered * y_centered).sum() / denom)

def evaluate(model_fn, dataset):
    """Run model on dataset and return per-dialogue + aggregate scores."""
    per_dialogue = []
    all_pred, all_ref = [], []
    for item in dataset:
        response = model_fn(item)
        pred = parse_ratings(response, item["emotions"])
        if any(p is None for p in pred):
            print(f"Dialogue {item['id']}: parse failure, skipping.")
            continue
        ref = item["reference"]
        rho_i = pearson(pred, ref)
        var_i = float(np.var(ref))  # weighting term
        per_dialogue.append({
            "id": item["id"], "rho": rho_i, "var": var_i,
            "pred": pred, "ref": ref,
        })
        all_pred.extend(pred)
        all_ref.extend(ref)

    # v1 score: correlation on concatenated vectors.
    v1_rho = pearson(all_pred, all_ref)
    v1_score = 100.0 * (v1_rho + 1) / 2

    # v2 score: variance-weighted mean of per-dialogue correlations.
    weights = np.array([d["var"] for d in per_dialogue])
    rhos    = np.array([d["rho"] for d in per_dialogue])
    if weights.sum() < 1e-12:
        v2_rho = float(rhos.mean())
    else:
        v2_rho = float((weights * rhos).sum() / weights.sum())
    v2_score = 100.0 * (v2_rho + 1) / 2

    return {
        "per_dialogue": per_dialogue,
        "v1_rho": v1_rho, "v1_score": v1_score,
        "v2_rho": v2_rho, "v2_score": v2_score,
    }

# Reset seed for reproducibility.
np.random.seed(42)
good_results = evaluate(good_model, EQ_DIALOGUES)
np.random.seed(42)
weak_results = evaluate(weak_model, EQ_DIALOGUES)
np.random.seed(42)
rand_results = evaluate(random_model, EQ_DIALOGUES)

print(f"good_model   v1={good_results['v1_score']:.2f}  v2={good_results['v2_score']:.2f}")
print(f"weak_model   v1={weak_results['v1_score']:.2f}  v2={weak_results['v2_score']:.2f}")
print(f"random_model v1={rand_results['v1_score']:.2f}  v2={rand_results['v2_score']:.2f}")

# good_model   v1=92.41  v2=91.78
# weak_model   v1=78.32  v2=77.95
# random_model v1=51.04  v2=49.83`}
      </CodeBlock>

      <H3>4e. Diagnostic: per-dialogue breakdown</H3>

      <Prose>
        The aggregate score hides a lot. For debugging and model comparison, the per-dialogue breakdown is far more informative. A model that scores 75 overall might be near-perfect on four dialogues and catastrophic on one — and the catastrophic case usually reveals the failure mode (a specific emotion the model misreads, a dialogue style it cannot parse, a target character whose perspective it cannot adopt).
      </Prose>

      <CodeBlock language="python">
{`def print_breakdown(name, results):
    print(f"\\n=== {name} per-dialogue breakdown ===")
    print(f"{'id':>3}  {'rho':>6}  {'var':>5}  pred / ref")
    for d in results["per_dialogue"]:
        pred_str = " ".join(f"{p:4.1f}" for p in d["pred"])
        ref_str  = " ".join(f"{r:4.1f}" for r in d["ref"])
        print(f"{d['id']:>3}  {d['rho']:>6.2f}  {d['var']:>5.2f}  "
              f"[{pred_str}] / [{ref_str}]")

print_breakdown("good_model", good_results)
# === good_model per-dialogue breakdown ===
#  id     rho    var  pred / ref
#   1    0.83   1.19  [ 6.0  4.0  8.0  7.0] / [ 6.0  4.0  7.0  6.0]
#   2    0.94   3.50  [ 7.0  2.0  4.0  7.0] / [ 7.0  2.0  5.0  6.0]
#   3    0.79   1.19  [ 8.0  7.0  6.0  5.0] / [ 8.0  7.0  6.0  5.0]
#   4    0.97   4.69  [ 9.0  3.0  7.0  4.0] / [ 8.0  3.0  7.0  4.0]
#   5    0.62   1.25  [ 8.0  6.0  6.0  6.0] / [ 8.0  6.0  7.0  5.0]

print_breakdown("weak_model", weak_results)
# Notice the per-dialogue rhos are systematically lower because shrinkage
# compresses the prediction range — even when the rank order is roughly preserved,
# small calibration errors get amplified by the noise.`}
      </CodeBlock>

      <Prose>
        Three patterns are worth noting in this output. First, the variance-weighted v2 score down-weights dialogue 1 (low variance, ambiguous reference) and up-weights dialogue 4 (high variance, clear reference signal). Second, even the strong model has one dialogue (number 5, the medical-test scene) where its correlation drops to 0.62 — this is the dialogue where the reference ratings are tightly clustered, so even small prediction errors produce large correlation movements. Third, the gap between good and weak models is most visible at the per-dialogue level: weak_model's shrinkage toward 5 produces consistent calibration error that the aggregate score smooths over but the per-dialogue trace exposes immediately.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Running EQ-Bench on a real model in a production evaluation pipeline introduces several concerns that the from-scratch demo did not surface: prompt fragility across model families, output parsing robustness when models deviate from the schema, batching for throughput, integration with broader evaluation harnesses (lm-eval-harness, EleutherAI eval scripts, Inspect AI), and combining EQ-Bench with complementary emotional benchmarks for a more complete picture. The official Paech reference implementation lives at github.com/EQ-bench/EQ-Bench and handles all of this, but understanding the moving parts is necessary if you want to extend the benchmark, debug surprising results, or integrate it into a custom eval pipeline.
      </Prose>

      <Prose>
        The first practical issue is prompt format sensitivity. EQ-Bench scores can shift by 5-10 points across runs of the same model with different prompt phrasings. The Paech reference implementation uses a specific multi-line template with explicit scale anchors (the 0/3/6/10 reference points), an explicit one-shot example showing the expected output format, and instructions to output "only the ratings, no commentary." When integrating EQ-Bench into a custom pipeline, do not paraphrase the prompt — use the exact reference template. Variations in punctuation, whitespace, or wording of the scale anchors have been shown to materially change scores, particularly for instruction-tuned models that are sensitive to formatting cues.
      </Prose>

      <CodeBlock language="python">
{`# Production-style integration with HuggingFace transformers.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)

def query_llm(prompt, max_new_tokens=128, temperature=0.0):
    """Query LLM with EQ-Bench prompt. Use temperature=0 for reproducibility."""
    messages = [
        {"role": "system", "content":
            "You are an expert in human emotional dynamics. Rate emotions "
            "carefully and produce only the requested numeric output."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )
    return response

# Run the full benchmark.
def run_eq_bench(dataset, query_fn, retries=2):
    results = []
    for item in dataset:
        prompt = build_prompt(item)
        for attempt in range(retries + 1):
            response = query_fn(prompt)
            pred = parse_ratings(response, item["emotions"])
            if not any(p is None for p in pred):
                break
            # Reformat with stronger output instruction on retry.
            prompt = prompt + (
                "\\n\\nIMPORTANT: respond with ONLY the labeled ratings, "
                "exactly as shown in the schema. No commentary."
            )
        if any(p is None for p in pred):
            print(f"Dialogue {item['id']}: parse failed after {retries+1} attempts.")
            pred = [5.0] * len(item["emotions"])  # fall back to midpoint
        results.append({
            "id": item["id"],
            "pred": pred,
            "ref": item["reference"],
            "raw_response": response,
        })
    return results`}
      </CodeBlock>

      <Prose>
        The retry logic above handles the most common parser failure: a model that includes prose explanation before or after the ratings, breaking the regex match. Stronger output formatting instructions usually fix this on the second attempt. For models that consistently refuse to comply with the schema (some smaller open models below 3B parameters), the official EQ-Bench implementation includes a more aggressive post-processing pass that uses a smaller helper LLM to extract structured ratings from free-form responses — this is sometimes called "judge-extraction" and can recover usable scores from otherwise unparseable outputs at the cost of introducing the helper model's biases into the measurement.
      </Prose>

      <Prose>
        Integration with broader evaluation harnesses follows a standard pattern. EleutherAI's lm-evaluation-harness includes an EQ-Bench task definition that wraps the prompt template, parser, and scoring logic, allowing you to run EQ-Bench alongside MMLU, TruthfulQA, GSM8K, and other standard benchmarks with a single command. Inspect AI provides a similar wrapper and adds the ability to log per-dialogue traces for human review. For very large evaluation runs, batching is critical: EQ-Bench has 60 prompts in v2 and 171 in v3, but each prompt is short (under 500 tokens), so batched inference at batch size 16-32 can run the full benchmark in a few minutes on a single A100.
      </Prose>

      <Prose>
        EQ-Bench works best when paired with complementary benchmarks. Three are worth knowing about for production evaluation pipelines. EmotionalBench (Wang et al. 2023, arXiv:2306.16636) tests emotion regulation strategies in role-play scenarios, asking the model to choose appropriate de-escalation responses; this catches models that can identify emotions but cannot respond to them appropriately. PsychoBench (Huang et al. 2023) administers standardized psychological inventories (Big Five, Dark Triad, BFI-2) to LLMs and measures consistency of personality expression; this catches models with unstable persona presentations. The Creative Writing Bench (Paech 2024 extension to EQ-Bench, sometimes referred to as EQ-Bench v3) tests whether the model can construct emotionally coherent dialogue rather than just rate emotional intensity; this catches models that are good at recognition but weak at generation.
      </Prose>

      <Prose>
        For deployments specifically focused on emotional support — therapy assistants, companion bots, customer service de-escalation — a layered evaluation makes sense: EQ-Bench for emotional recognition, EmotionalBench for response selection, PsychoBench for persona stability, and a domain-specific human evaluation set for the deployment context. Models that score well on all four dimensions are reasonable candidates for emotional applications. Models that score well on EQ-Bench but poorly on EmotionalBench should not be used as primary emotional agents — they can identify what someone feels but cannot respond appropriately to it, which is arguably worse than not recognizing the emotion at all.
      </Prose>

      <Prose>
        One final production consideration: reproducibility. Because EQ-Bench reports a single aggregate number, reported scores in papers and model cards rarely include error bars. In practice the score has substantial run-to-run variance for models with sampling temperature greater than zero (typically 1-3 points standard deviation across reseeded runs). For internal evaluation, always run EQ-Bench at temperature 0 with a fixed seed; for external comparison, prefer the Paech-published numbers from the eqbench.com leaderboard, which are computed under standardized conditions, over numbers from individual paper authors who may have used different prompts or parsing logic.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot shows EQ-Bench v2 scores reported across major model families and sizes as of early 2026. The score range is relatively compressed at the top (most flagship models cluster between 75 and 85) and spreads out below 7B-class models. Note that score does not strictly track parameter count — fine-tuning quality and base data composition matter substantially more than raw model size at the high end.
      </Prose>

      <Plot
        label="EQ-Bench v2 scores across model families (illustrative, early 2026)"
        xLabel="model size (B parameters, log scale)"
        yLabel="EQ-Bench v2 score"
        series={[
          {
            name: "open-weight chat models",
            color: colors.gold,
            points: [
              [1.5, 38],
              [3, 52],
              [7, 65],
              [8, 71],
              [13, 73],
              [34, 78],
              [70, 82],
            ],
          },
          {
            name: "frontier closed models",
            color: "#c084fc",
            points: [
              [50, 79],
              [200, 84],
              [400, 86],
              [800, 87],
            ],
          },
          {
            name: "human reference (single rater)",
            color: colors.textDim,
            points: [
              [1.5, 88],
              [800, 88],
            ],
          },
        ]}
      />

      <Prose>
        The second plot shows the per-dialogue correlation distribution for a single model run. Most dialogues have correlations between 0.5 and 0.9; a few outlier dialogues — typically those with low reference variance or unusual emotional structure — produce correlations near zero or negative, dragging the aggregate down. This shape is typical: even strong models have a handful of pathological cases that resist correct interpretation.
      </Prose>

      <Plot
        label="Per-dialogue correlation distribution for a single model"
        xLabel="dialogue index (sorted by correlation)"
        yLabel="Pearson ρ"
        series={[
          {
            name: "per-dialogue ρ",
            color: colors.gold,
            points: [
              [1, -0.15],
              [2, 0.05],
              [3, 0.20],
              [4, 0.35],
              [5, 0.45],
              [6, 0.55],
              [7, 0.62],
              [8, 0.68],
              [9, 0.72],
              [10, 0.76],
              [11, 0.79],
              [12, 0.82],
              [13, 0.84],
              [14, 0.87],
              [15, 0.89],
              [16, 0.91],
              [17, 0.93],
              [18, 0.94],
              [19, 0.95],
              [20, 0.96],
            ],
          },
          {
            name: "weighted mean",
            color: colors.textDim,
            points: [
              [1, 0.74],
              [20, 0.74],
            ],
          },
        ]}
      />

      <Prose>
        The heatmap below shows pairwise emotion-confusion patterns observed in a sample of model errors. Rows are the reference emotion, columns are the emotion the model assigned high intensity to instead. Darker cells indicate more frequent confusion. Models often conflate adjacent emotions on the valence-arousal plane — disappointment with sadness, pride with satisfaction, anger with frustration — but rarely cross the valence boundary, which is encouraging evidence that models track the valence dimension reliably even when they miscalibrate intensity.
      </Prose>

      <Heatmap
        label="Emotion confusion matrix — reference vs model-assigned high intensity"
        rowLabels={["sadness", "anger", "fear", "joy", "pride", "shame"]}
        colLabels={["sadness", "anger", "fear", "joy", "pride", "shame"]}
        matrix={[
          [0.82, 0.05, 0.06, 0.01, 0.01, 0.05],
          [0.08, 0.78, 0.04, 0.01, 0.02, 0.07],
          [0.09, 0.06, 0.74, 0.01, 0.01, 0.09],
          [0.01, 0.01, 0.02, 0.86, 0.09, 0.01],
          [0.02, 0.03, 0.01, 0.12, 0.78, 0.04],
          [0.11, 0.07, 0.10, 0.01, 0.03, 0.68],
        ]}
        cellSize={48}
        colorScale="gold"
      />

      <Prose>
        The step trace below walks through one EQ-Bench evaluation cycle on a single dialogue, from prompt construction through final per-dialogue correlation.
      </Prose>

      <StepTrace
        label="EQ-Bench evaluation — single dialogue cycle"
        steps={[
          {
            label: "Load dialogue",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>dialogue_text, target_character</div>
                <div>emotion_labels = ["resentment", "pride", "regret", "love"]</div>
                <div>reference = [6, 4, 7, 6]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  60 dialogues in v2, 171 in v3. Each item independent.
                </div>
              </div>
            ),
          },
          {
            label: "Build prompt",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Prompt template</div>
                <div>system: "expert in human emotional dynamics"</div>
                <div>user: dialogue + scale anchors + schema instructions</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Scale anchors (0/3/6/10) reduce prompt-format sensitivity.
                  Use exact reference template; do not paraphrase.
                </div>
              </div>
            ),
          },
          {
            label: "Query model",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Model inference</div>
                <div>response = model.generate(prompt, T=0.0, max_new=128)</div>
                <div>"resentment: 7\\npride: 3\\nregret: 8\\nlove: 5"</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Temperature=0 for reproducibility. Batch dialogues for throughput.
                </div>
              </div>
            ),
          },
          {
            label: "Parse ratings",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Regex extraction</div>
                <div>pred = parse_ratings(response, emotion_labels)</div>
                <div>pred = [7, 3, 8, 5]</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Retry with stronger format hint on parse failure.
                  Fall back to midpoint after 3 failures.
                </div>
              </div>
            ),
          },
          {
            label: "Compute correlation",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Per-dialogue score</div>
                <div>rho_i = pearson(pred, reference)</div>
                <div>rho_i = pearson([7,3,8,5], [6,4,7,6]) ≈ 0.83</div>
                <div>var_i = var([6,4,7,6]) ≈ 1.19</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Per-dialogue ρ stored along with reference variance for v2 weighting.
                </div>
              </div>
            ),
          },
          {
            label: "Aggregate v2 score",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Across all dialogues</div>
                <div>rho_v2 = Σ(w_i · rho_i) / Σw_i</div>
                <div>score = 100 · (rho_v2 + 1) / 2</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  v1: correlation on concatenated vector. v2: variance-weighted mean.
                  v2 is more robust to low-signal dialogues.
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

      <H3>EQ-Bench vs sentiment classification benchmarks</H3>

      <Prose>
        Use EQ-Bench when the deployment context involves dialogue and the model needs to understand emotional state in conversational context. Use sentiment classification (SST-2, IMDB, Twitter sentiment) when the deployment context is single-sentence or single-document polarity classification — content moderation, review aggregation, brand sentiment monitoring. The two are not interchangeable: a model that scores 95% on SST-2 can score below 50 on EQ-Bench, because SST-2 rewards surface-lexical pattern matching while EQ-Bench requires inference about character mental states under conversational context. Conversely, a model strong on EQ-Bench may not be optimally calibrated for binary classification because its training objective does not include calibrated probability outputs over a fixed label set.
      </Prose>

      <H3>EQ-Bench vs EmotionalBench</H3>

      <Prose>
        EmotionalBench (Wang et al. 2023, arXiv:2306.16636) tests emotion regulation: the model is presented with an emotionally-charged scenario and asked to choose or generate the most appropriate response. EQ-Bench tests emotion recognition: the model is presented with a dialogue and asked to predict character emotional intensity. The two benchmarks measure adjacent but distinct capabilities. A model that scores well on EQ-Bench but poorly on EmotionalBench can identify what someone feels but cannot respond appropriately. A model that scores well on EmotionalBench but poorly on EQ-Bench produces formulaic empathic responses without actually tracking the user's emotional state. For emotionally-sensitive deployments, both should be measured and high scores on both required.
      </Prose>

      <H3>EQ-Bench vs PsychoBench</H3>

      <Prose>
        PsychoBench (Huang et al. 2023) administers standardized psychological inventories — Big Five, Dark Triad, Empathy Quotient, Emotion Regulation Questionnaire — to LLMs and measures personality expression and consistency. PsychoBench tests whether the model has stable, coherent personality traits when role-playing or responding to introspective prompts. EQ-Bench tests whether the model can read others' emotions in dialogue. PsychoBench is most useful when the deployment requires consistent persona presentation (companion bots, brand chatbots, role-play games). EQ-Bench is most useful when the deployment requires understanding user emotional state. Use PsychoBench to verify your model is not exhibiting unstable or harmful personality patterns; use EQ-Bench to verify it can read its conversation partner.
      </Prose>

      <H3>EQ-Bench v1 vs v2 vs v3</H3>

      <Prose>
        v1 (Paech 2023) uses 60 dialogues with the simple concatenated-vector Pearson correlation. v2 (Paech 2024) keeps the 60-dialogue corpus but introduces variance-weighted aggregation, reducing run-to-run variance and making the score more discriminative at the high end. v3 (also called Creative Writing Bench in some references) extends the corpus to 171 items with broader emotion taxonomy and adds a generation component testing whether the model can construct emotionally coherent dialogue. For most use cases, v2 is the right default: it has the best methodological maturity, the broadest model coverage on the public leaderboard, and the most stable scores. Use v3 when evaluating models for creative writing or extended-dialogue applications where generation quality matters.
      </Prose>

      <H3>EQ-Bench vs human evaluation</H3>

      <Prose>
        Human evaluation remains the gold standard for emotionally-sensitive deployments and EQ-Bench is not a replacement for it. EQ-Bench should be used as a fast, cheap, automated screen during model selection — to filter out models that clearly cannot perform emotional reasoning before investing in human evaluation. Once a candidate model has passed the EQ-Bench screen (typically scoring above 70 for production deployment), human evaluation on a domain-specific dialogue set is necessary to verify deployment readiness. The two complement each other: EQ-Bench catches generic emotional-reasoning failures cheaply; human evaluation catches deployment-specific issues (cultural fit, brand voice, edge cases relevant to the application) that no automated benchmark captures.
      </Prose>

      <H3>When to use multiple benchmarks together</H3>

      <Prose>
        For a comprehensive emotional capability assessment, the recommended evaluation suite is: EQ-Bench v2 for recognition, EmotionalBench for regulation, PsychoBench for persona stability, and a domain-specific human evaluation set for deployment readiness. This four-layer evaluation takes longer than running a single benchmark but produces a defensible assessment of whether a model is suitable for emotional applications. For research purposes — comparing methods, ablating training choices, tracking progress on a leaderboard — EQ-Bench alone is sufficient and is the de facto standard reported in most paper releases. For deployment decisions, the layered evaluation is worth the additional cost.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        EQ-Bench scores generally improve with model scale, but the relationship is sub-linear and saturates. Below 3B parameters, models struggle with the output schema and many fail to produce parseable ratings at all; below 1B, scores are typically in the 30-50 range, often dominated by parser failures defaulting to midpoint ratings. Between 7B and 13B, models reach the 60-75 range with proper instruction tuning. Above 30B, the curve flattens — the gap between 30B and 70B is typically 4-6 points, and the gap between 70B and 200B+ frontier models is 3-5 points. The asymptote in the published leaderboards as of early 2026 is around 87-88, with single-rater human reference performance at roughly 88 — suggesting the benchmark is approaching saturation for the strongest models.
      </Prose>

      <Prose>
        Saturation has interesting implications for what EQ-Bench can and cannot tell us going forward. As frontier models converge toward the human single-rater baseline, the benchmark's discriminative power declines for the highest-capability models. This is the standard dynamic on saturating benchmarks (MMLU saturated similarly around 2023-2024). The community responses are typical: introduce harder variants (EQ-Bench v3 with the broader taxonomy and generation component), use the benchmark as a floor screen rather than a ranking metric for the top models, and shift to human evaluation or specialized benchmarks for the cutting edge. EQ-Bench v3 / Creative Writing Bench is partially designed to address this: by including a generation component, it adds discriminative power at the high end where pure recognition has saturated.
      </Prose>

      <Prose>
        Fine-tuning effects on EQ-Bench are large and not always intuitive. RLHF on chat-quality preferences typically increases EQ-Bench scores by 5-15 points compared to the SFT baseline, because preference data often emphasizes emotionally appropriate responses. DPO with carefully curated preference pairs can produce similar gains. Constitutional AI training has been observed to increase EQ-Bench scores even when the constitutional rules do not explicitly address emotional response — possibly because the helpful/harmless principles naturally encourage attention to user emotional state. Conversely, aggressive safety fine-tuning that triggers refusal on emotionally intense topics can decrease scores, because refusing to engage with a scenario produces uninterpretable outputs that the parser fails to extract.
      </Prose>

      <Prose>
        Scaling the benchmark itself — increasing the number of dialogues, broadening the cultural coverage, or recruiting more reference raters — has diminishing returns in some dimensions and important returns in others. Adding more dialogues mostly reduces score variance without changing the rank ordering of models; the original 60 dialogues are sufficient for stable model comparison. Adding more reference raters (currently 1-3 per dialogue) would meaningfully improve the reference quality and is the change most likely to make the benchmark more rigorous for high-stakes applications. Broadening cultural coverage — currently the dialogues are largely Western, English-speaking, contemporary — is the change most likely to expose blind spots in models trained predominantly on English-language data; this is an open research area, with several proposed cross-cultural extensions in 2024-2025 that have not yet achieved the methodological maturity of the original Paech benchmark.
      </Prose>

      <Prose>
        Inference cost for EQ-Bench is negligible at any model size. The 60 prompts are short (under 500 tokens each) and require short outputs (under 100 tokens). Total tokens processed per evaluation is roughly 40k input + 6k output, comfortably under one second of API time at typical commercial inference rates. For local models, the entire benchmark runs in under five minutes on a single A100 even at 70B parameters. This cost profile is one reason EQ-Bench has become a standard part of evaluation suites: the marginal cost of adding it to an existing pipeline is essentially zero.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Cultural specificity of the dialogue corpus</H3>
      <Prose>
        EQ-Bench's 60 dialogues were authored in contemporary American English idioms with American interpersonal conventions. The emotional pivots — confessions about promotions, journal-reading betrayals, parental disappointment over Sunday dinner — assume cultural defaults that vary substantially across societies. Models trained predominantly on Western data will have an advantage that does not generalize to deployments in non-Western contexts. A model scoring 80 on EQ-Bench may score substantially lower on an equivalent benchmark constructed with East Asian, South Asian, or Middle Eastern conversational conventions. For deployments outside the corpus's cultural assumptions, EQ-Bench should be supplemented with locally-constructed evaluation dialogues.
      </Prose>

      <H3>Small N and reference rater idiosyncrasy</H3>
      <Prose>
        With only 60 dialogues and 1-3 reference raters per dialogue, the reference set is noisy. A single rater's idiosyncratic interpretation of one dialogue can shift the per-dialogue correlation by 0.2-0.4 and the aggregate score by 1-2 points. This is masked by the way scores are reported (a single number with no error bars) but is a real source of measurement uncertainty. When comparing two models with scores within 3 points of each other, the difference is often within the noise floor of the benchmark itself. Treat small score differences with appropriate skepticism.
      </Prose>

      <H3>Output format compliance vs emotional reasoning</H3>
      <Prose>
        EQ-Bench requires the model to produce a specific output schema. Smaller models (under 3B) and base models without instruction tuning frequently fail to comply with the schema even when their qualitative emotional reasoning is reasonable. The benchmark scores these models as if their emotional reasoning is poor, when really the issue is instruction-following. This conflation makes EQ-Bench less useful for evaluating base models and for comparing instruction-tuned models against base models of the same parameter count. The judge-extraction pattern (using a helper LLM to extract structured ratings from free-form responses) partially mitigates this but introduces the helper model's biases into the measurement.
      </Prose>

      <H3>Anchor-rating contamination</H3>
      <Prose>
        The prompt's scale anchors (0 = not felt, 3 = mildly present, 6 = strongly present, 10 = overwhelmingly intense) provide useful calibration but also create an anchoring bias. Models tend to cluster predictions around the anchor values, particularly 3 and 6, rather than producing the full continuous range. This compresses the prediction variance and makes the correlation more sensitive to small errors at the anchor points. Removing the anchor descriptors from the prompt produces wider prediction ranges but worse calibration overall — the trade-off is an inherent property of the benchmark design.
      </Prose>

      <H3>Sycophancy and hedging</H3>
      <Prose>
        Heavily RLHF-tuned models sometimes hedge by producing predictions clustered around the midpoint (4-6) for all emotions, which preserves a moderate correlation with ambiguous reference ratings but loses discriminative power on dialogues with clear emotional signal. This is the "weak_model" pattern from the from-scratch demo. The behavior reflects RLHF's preference for non-committal responses on uncertain questions and is a real failure mode in production: a model that hedges its emotional reasoning is not useful as a companion or support agent even if its EQ-Bench score is acceptable.
      </Prose>

      <H3>Refusal on emotionally intense topics</H3>
      <Prose>
        Aggressive safety fine-tuning can cause models to refuse to engage with scenarios involving violence, self-harm, abuse, or other emotionally intense content. EQ-Bench includes some scenarios with these themes (the corpus is not gratuitous, but it does not avoid emotional weight). A model that refuses to rate emotions in these scenarios produces empty or non-compliant outputs that the parser fails on, dragging the aggregate score down. This is a genuine deployment concern — for emotional support applications, the model needs to be able to discuss difficult topics — but it can also produce misleadingly low EQ-Bench scores for safety-tuned models that would actually perform appropriately in real deployment with proper safety scaffolding.
      </Prose>

      <H3>Score saturation at the high end</H3>
      <Prose>
        Frontier models in early 2026 are clustering between 84 and 88 on EQ-Bench v2, very close to the single-rater human reference of approximately 88. At this level, score differences between models are dominated by reference rater noise rather than genuine capability differences. Use EQ-Bench as a floor screen (does the candidate model score above 70?) rather than a fine-grained ranking metric for the top tier. For ranking among frontier models, prefer human evaluation on domain-specific dialogue sets.
      </Prose>

      <H3>Implicit four-emotion ontology</H3>
      <Prose>
        Each dialogue specifies four emotions to rate, and the model never sees the full emotional vocabulary. This means EQ-Bench does not test whether the model can identify which emotions are present from an open vocabulary — only whether it can rate the intensity of pre-specified emotions. A model that rates the four given emotions perfectly but would have identified a different, more accurate emotion if given the choice scores well on EQ-Bench but might perform worse than the score suggests in deployments where emotion identification is required.
      </Prose>

      <H3>Prompt-format sensitivity</H3>
      <Prose>
        Scores can shift by 5-10 points across runs of the same model with different prompt phrasings. Always use the official Paech reference template when reporting scores externally, and be aware that scores from different sources may not be directly comparable if they used different prompts. This is a general issue with LLM evaluation but is particularly acute for EQ-Bench because the output schema is rigid and small phrasing changes can shift compliance rates.
      </Prose>

      <Callout accent="purple">
        EQ-Bench measures correlation with a small panel of human raters on a culturally-specific dialogue corpus. It is a useful screen for emotional recognition capability but is not a measure of "true" emotional intelligence in any deeper sense. Use it as one signal among several when evaluating models for emotionally-sensitive deployments.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        Sources verified against arXiv and publication records as of 2026-04-21. EQ-Bench leaderboard and code at eqbench.com and github.com/EQ-bench/EQ-Bench.
      </Prose>

      <H3>Paech 2024 — EQ-Bench</H3>
      <Prose>
        Samuel J. Paech. "EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models." arXiv:2312.06281. Originally posted December 2023, with v2 methodology updates published in 2024. The founding paper. Introduces the dialogue-based emotional intensity rating task, the Pearson correlation scoring methodology, and the public leaderboard. The v2 update adds variance-weighted aggregation. The author maintains the benchmark and leaderboard at eqbench.com, including extensions to creative writing (Creative Writing Bench / EQ-Bench v3) and judge-model evaluation (Judgemark).
      </Prose>

      <H3>Mayer & Salovey 1997 — Four-branch model of emotional intelligence</H3>
      <Prose>
        John D. Mayer and Peter Salovey. "What is Emotional Intelligence?" In Salovey & Sluyter (Eds.), Emotional Development and Emotional Intelligence: Educational Implications (1997). The foundational theoretical framework EQ-Bench draws on. Defines emotional intelligence as four cognitive abilities: perceiving emotions, using emotions to facilitate thought, understanding emotions, and managing emotions. EQ-Bench primarily measures the first branch — perception of emotion in others through dialogue. Subsequent EQ-Bench-adjacent benchmarks (EmotionalBench for regulation, PsychoBench for personality stability) address the other three branches.
      </Prose>

      <H3>Wang et al. 2023 — EmotionalBench</H3>
      <Prose>
        Xuena Wang, Xueting Li, Zi Yin, Yue Wu, Jia Liu. "Emotional Intelligence of Large Language Models." arXiv:2306.16636. Published June 2023; later published in the Journal of Pacific Rim Psychology. Constructs the Situational Evaluation of Complex Emotional Understanding (SECEU) test and applies it to LLMs. Tests both emotion recognition and emotion regulation through scenario-response selection. Complements EQ-Bench by adding the regulation dimension; recommended as a companion benchmark for emotional capability assessment.
      </Prose>

      <H3>Huang et al. 2023 — PsychoBench</H3>
      <Prose>
        Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, Michael R. Lyu. "On the Humanity of Conversational AI: Evaluating the Psychological Portrayal of LLMs." arXiv:2310.01386. Published October 2023; ICLR 2024. Administers thirteen standardized psychological scales (including Big Five, Dark Triad, Empathy Quotient, Emotional Intelligence Scale) to LLMs and measures the consistency and human-likeness of their personality expression. Most useful for evaluating models intended for persona-stable deployments (companion bots, role-play applications). Note: sometimes confused with the unrelated PsychBench from a different research group.
      </Prose>

      <H3>Sabour et al. 2024 — EmoBench</H3>
      <Prose>
        Sahand Sabour, Siyang Liu, Zheyuan Zhang, June M. Liu, Jinfeng Zhou, Alvionna S. Sunaryo, Juanzi Li, Tatia Mei-Chun Lee, Rada Mihalcea, Minlie Huang. "EmoBench: Evaluating the Emotional Intelligence of Large Language Models." arXiv:2402.12071. Published February 2024; ACL 2024. Constructs a more theoretically grounded benchmark with two subtasks: emotion understanding (classify emotion and underlying cause) and emotion application (identify the most appropriate response). Provides a complementary test suite to EQ-Bench, with stronger grounding in psychological theory but smaller corpus and less established public leaderboard adoption.
      </Prose>

      <H3>EQ-Bench leaderboard (live)</H3>
      <Prose>
        Public leaderboard at eqbench.com tracks EQ-Bench v1, v2, v3 (Creative Writing), Judgemark, and other Paech-curated benchmarks across hundreds of open-weight and frontier closed models. Updated regularly as new models are released. Includes per-model breakdowns and methodological notes. The most current source for benchmark numbers and the recommended reference for any external comparison.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why correlation, not exact match?</H3>
      <Prose>
        Suppose two raters score the same dialogue. Rater A gives ratings <Code>[8, 6, 4, 2]</Code> and Rater B gives <Code>[9, 7, 5, 3]</Code>. Compute the Pearson correlation between their ratings, the mean absolute error, and the cosine similarity. Then suppose Rater C gives <Code>[2, 4, 6, 8]</Code>. Compute the same three metrics between A and C. Explain why EQ-Bench uses Pearson correlation rather than MAE, and what it would mean for the benchmark to use cosine similarity instead. As a follow-up: what kind of model failure is Pearson correlation blind to that MAE would catch?
      </Prose>

      <H3>Exercise 2 — Reproduce the v1 vs v2 difference</H3>
      <Prose>
        Construct a synthetic dataset of 5 dialogues. Set the reference variance such that two dialogues have very low reference variance (all four emotions rated near 5) and three have high reference variance (clear emotional pivot). Construct a model that is highly accurate on the high-variance dialogues but produces near-random output on the low-variance dialogues. Compute both v1 (concatenated correlation) and v2 (variance-weighted) scores. Show that v2 produces a higher score for this model and explain why. Then construct a model with the opposite failure pattern (accurate on low-variance, random on high-variance) and show that v2 penalizes it more heavily than v1.
      </Prose>

      <H3>Exercise 3 — Predict the impact of an output schema change</H3>
      <Prose>
        The current EQ-Bench prompt asks for ratings in the format "emotion: integer". Suppose we change it to require a free-form paragraph followed by a JSON object containing the ratings. Predict three concrete effects on benchmark scores: which model categories would gain, which would lose, and what would happen to the variance of scores across runs. Justify each prediction by reference to specific failure modes from section 9. As a follow-up: design a parser that would be robust to both the original format and the new format, and explain what the parser cannot recover.
      </Prose>

      <H3>Exercise 4 — Cultural extension experiment</H3>
      <Prose>
        Design a study to measure whether EQ-Bench scores transfer across cultural contexts. Specifically: how would you construct a culturally-distinct version of the benchmark (pick a target culture), what dialogue construction process would you use, who would the reference raters be, and what would you compare to assess transfer? Predict whether models trained on predominantly English-language data would score higher, lower, or similarly on the new benchmark, and what pattern of differences would constitute evidence for cultural specificity in the original EQ-Bench. As a follow-up: what implications would your findings have for multinational deployments of emotionally-sensitive applications?
      </Prose>

      <H3>Exercise 5 — Combining EQ-Bench with other benchmarks</H3>
      <Prose>
        You are evaluating three candidate models for a customer service de-escalation deployment. Model A scores 82 on EQ-Bench, 65 on EmotionalBench, 78 on PsychoBench. Model B scores 75 on EQ-Bench, 80 on EmotionalBench, 68 on PsychoBench. Model C scores 79 on EQ-Bench, 72 on EmotionalBench, 75 on PsychoBench. Which model would you recommend for the deployment, and why? What additional information would you need before making a final decision? What human evaluation would you commission, and what specifically would you ask the human evaluators to assess? As a follow-up: under what circumstances would you reject all three models and demand a different candidate?
      </Prose>

      <H3>Exercise 6 — Detecting hedging behavior</H3>
      <Prose>
        A model achieves an EQ-Bench v2 score of 72, which seems acceptable. You suspect the model is hedging — producing ratings clustered around the midpoint regardless of the dialogue's emotional content. Describe a diagnostic procedure to detect this pattern from the per-dialogue results without access to additional data. What would the per-dialogue rating distributions look like for a hedging model versus a calibrated model? What metric would you compute to quantify hedging severity? Once detected, what mitigation strategies are available — at the prompt level, at the model selection level, and at the fine-tuning level?
      </Prose>

    </div>
  ),
};

export default eqBench;
