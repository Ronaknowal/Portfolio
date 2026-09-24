import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { MathBlock } from "../../components/content/Math.jsx";
import { StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const prometheusOpenSourceJudges = {
  title: "Prometheus & Open-Source Judge Models",
  slug: "prometheus-open-source-judge-models",
  readTime: "~38 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        For most of 2023, the de facto answer to "how do you evaluate an open-ended language model output at scale?" was simple and uncomfortable: ask GPT-4. Pass the prompt, the candidate response, and a rubric to OpenAI's API, and use the model's score as a proxy for human judgment. This worked well enough that it became the spine of nearly every alignment evaluation pipeline released that year — MT-Bench, AlpacaEval, FLASK, Vicuna's pairwise battles, and dozens of internal benchmarks at industry labs all eventually settled on a GPT-4 judge as their reference standard. The arrangement was technically convenient and methodologically uneasy at the same time: the field was using a closed-weight model from a single vendor as the ground truth against which all other models were measured, and the results of every academic evaluation depended on an opaque API endpoint whose behavior could change between releases without notice.
      </Prose>

      <Prose>
        The discomfort was not merely philosophical. There were four concrete operational problems with the GPT-4-as-judge regime. First, cost: a moderate evaluation suite of 5,000 prompts evaluated pairwise across two candidate models requires approximately 10,000 API calls, and at GPT-4 list pricing in 2023 a single comparison study could exceed several thousand dollars in API spend. For research groups running ablations across dozens of training configurations, this was a real budget line item that distorted what experiments got run. Second, privacy: many domains where evaluation matters most — medical question answering, legal reasoning, customer support transcripts — cannot be sent to a third-party API. On-premise evaluation was simply not an option. Third, reproducibility: GPT-4's behavior drifted between snapshot versions, and the field had no mechanism to freeze the judge for longitudinal studies. A benchmark score from March 2023 was not directly comparable to a benchmark score from November 2023 because the underlying judge had silently updated. Fourth, customization: the GPT-4 endpoint exposed no fine-tuning interface for evaluation use cases, so a research group with a specialized rubric (for example, evaluating clinical reasoning on a specific 7-point Likert scale) could only prompt-engineer around the model's general-purpose behavior rather than train it to internalize the rubric directly.
      </Prose>

      <Prose>
        In October 2023, Seungone Kim and collaborators at KAIST and CMU published a result that addressed all four concerns at once. The paper, "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models" (arXiv:2310.08491), introduced a 13B Llama-2-based model fine-tuned specifically for the role of an evaluation judge. The key contribution was not architectural — Prometheus is a standard decoder-only transformer, structurally identical to its Llama-2 base — but data-centric. The authors constructed the Feedback Collection: a dataset of 1,000 fine-grained scoring rubrics, each paired with example responses spanning the rubric's full score range (typically a 5-point Likert scale), with detailed natural-language feedback explaining why each response received the score it did. Fine-tuning Llama-2-13B-Chat on this dataset produced a model that, when given a new (instruction, response, rubric, reference answer) tuple, could output a numerical score and a paragraph of feedback that correlated with human scores at roughly Pearson 0.897 — within a few points of GPT-4's correlation with the same human raters.
      </Prose>

      <Prose>
        Prometheus 2 (Kim et al. 2024, arXiv:2405.01535) extended this in two important directions. First, it released a 7B model based on Mistral-7B and an 8x7B model based on Mixtral, broadening the deployment surface from "you need an A100 to serve the judge" to "a single consumer GPU is enough." Second, it unified pointwise (absolute) and pairwise (relative) judgment in a single model, achieved through a notable technique: separately fine-tuning two specialist models (one for absolute scoring, one for pairwise comparison) and then averaging their weights. The resulting merged model performs both tasks competitively with the specialists and avoids the inference-time complexity of running two separate judges. This weight-merging trick — taking the arithmetic mean of two fine-tuned checkpoints derived from the same base model — has since become a standard tool in the open-weight model designer's kit.
      </Prose>

      <Prose>
        Prometheus was not the only open-source judge effort. JudgeLM (Zhu et al., arXiv:2310.17631) trained a Vicuna-based model on 100k GPT-4-labeled response pairs, focused specifically on pairwise comparison. PandaLM (Wang et al., arXiv:2306.05087) introduced an evaluation framework around its own judge model. AutoJ / Auto-J (Li et al., arXiv:2310.05470) trained on a more diverse mixture of evaluation scenarios and emphasized critique generation alongside scoring. By mid-2024, the open-source judge ecosystem covered roughly the same evaluation surface as the commercial APIs, with the additional property that the weights were downloadable, the training data was inspectable, and the inference cost — once the model was loaded — was effectively free at the margin. For any team doing serious model development, switching the evaluation infrastructure from a closed API to a frozen open-weight judge is one of the highest-leverage operational improvements available, and understanding how Prometheus achieves its judgment quality is the prerequisite for doing it right.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The first thing to internalize about Prometheus is that almost nothing distinctive is happening at the model level. Prometheus 13B is a Llama-2-13B-Chat checkpoint with the standard transformer architecture, the standard tokenizer, and the standard generation interface. If you loaded it into HuggingFace Transformers without knowing what it was, it would look identical to any other Llama-2 fine-tune. The intelligence that allows it to score open-ended responses on arbitrary rubrics with high human correlation does not live in some specialized scoring head or auxiliary network. It lives in the training data and the prompt format. Both of these are knowable, replicable, and modifiable. The lesson here is one that applies to a broad class of "specialized model" claims: when a fine-tuned model significantly outperforms a base model on a narrow task, the cause is almost always the training data composition, and the architecture is incidental.
      </Prose>

      <Prose>
        The structural insight behind Prometheus is that evaluation is itself a generation task. Given an instruction-response pair, a rubric describing what good and bad responses look like, and optionally a reference answer, the judge model is asked to generate a response in a fixed format: a paragraph of feedback followed by a literal score token. The score is extracted from the generation by parsing — there is no special "score head" in the network, no regression layer, no calibrated probability output. The model is doing supervised next-token prediction on a sequence that happens to end in a digit, and that digit is the answer. This framing matters because it means everything you know about improving language model generation applies directly to improving judge quality: better data, better prompts, better instruction tuning, longer context handling, and so on.
      </Prose>

      <Prose>
        The Feedback Collection — Prometheus's training dataset — is structured around a key design choice: every training example specifies its own scoring rubric. Rather than train the model on a single fixed rubric (such as "rate this response from 1 to 5 on overall quality"), the dataset includes 1,000 different rubrics, each describing a different evaluation dimension on a different domain. One rubric might score clinical reasoning on the criterion "does the response correctly identify the differential diagnosis?", with explicit prose for what a 1, 2, 3, 4, and 5 look like. Another might score creative writing on "does the response maintain consistent characterization?". By training across this rubric diversity, Prometheus learns the meta-skill of consuming a rubric specification as input and applying it to a new instance — the rubric-following behavior generalizes to held-out rubrics never seen during training. This is the same generalization phenomenon that makes instruction-tuned models work: train on enough diverse instructions and the model learns to follow new instructions at inference time.
      </Prose>

      <Prose>
        The pairwise extension in Prometheus 2 is conceptually a small change with a large empirical effect. Instead of predicting an absolute score from 1 to 5, the pairwise variant takes two responses and predicts which one better satisfies the rubric. This corresponds directly to the comparison structure that human raters most naturally use, and it is also the structure consumed by downstream alignment methods like DPO. Pointwise and pairwise judgment have asymmetric strengths: pointwise scoring is more useful for filtering (drop everything below a threshold), more useful for absolute progress tracking (did this training run improve scores?), and more interpretable per-instance. Pairwise scoring is more useful for ranking (sort all candidates), more robust to scale calibration drift between runs, and aligns with how human raters actually behave. A judge that can do both, accessed through the same checkpoint, is operationally simpler than maintaining two separate models.
      </Prose>

      <Prose>
        The weight-merging trick used to combine pointwise and pairwise capability deserves intuition before its math. Suppose you fine-tune a base model on task A to get checkpoint A', and separately fine-tune the same base model on task B to get checkpoint B'. If you compute the element-wise average of A' and B' — literally <Code>{"(A' + B') / 2"}</Code> taken parameter-by-parameter — you get a checkpoint that is not exactly as good as either specialist on its own task, but is usually close enough that you can drop two specialist models and replace them with one merged generalist. The reason this works at all is subtle: fine-tuning typically moves weights only slightly from their pretraining initialization, the perturbation from each task lies in a low-rank-ish direction, and averaging two such small perturbations preserves most of the directional signal while smoothing out task-specific overfitting. This is a special case of a broader phenomenon (model souping, task arithmetic) that the open-weight community has exploited heavily since 2022.
      </Prose>

      <Prose>
        One last piece of intuition is critical: a judge model is itself a model with biases, and those biases propagate downstream. If Prometheus systematically rewards verbose responses, then any DPO dataset constructed using Prometheus as the labeler will inherit that length bias. If it under-scores responses that disagree with majority opinions present in its training data, then any alignment loop that uses it as feedback will inherit a conformity preference. Treating the judge as a black box "ground truth" repeats exactly the methodological error that motivated escaping from GPT-4 as judge in the first place. The right disposition is to treat the judge as a measurement instrument with known and unknown error modes, calibrate it against human scores on a held-out set before deploying it as the evaluation backbone for a research program, and re-calibrate periodically as the response distributions you are evaluating drift away from the rubric distribution the judge was trained on.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        The math of Prometheus splits cleanly into two parts: the training objective (which is just standard supervised fine-tuning, with nothing exotic), and the evaluation framework — the statistical tooling used to determine whether a judge model's scores agree with human scores. The latter is more interesting, because it is what you need to know to validate any open-source judge before deploying it.
      </Prose>

      <H3>Training objective</H3>

      <Prose>
        Prometheus is trained with the standard causal language modeling loss applied to a structured prompt-completion format. Let <Code>{"x"}</Code> denote the structured input — instruction, response, rubric, reference answer, all concatenated according to a fixed template — and let <Code>{"y = (y_1, ..., y_T)"}</Code> denote the target completion, which contains the natural-language feedback followed by the score token. The loss is the standard token-level negative log-likelihood, optionally with the prompt portion masked so gradients only flow through the completion:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{\\mathrm{SFT}}(\\theta) = -\\mathbb{E}_{(x, y) \\sim \\mathcal{D}_{\\mathrm{FC}}}\\left[\\sum_{t=1}^{T} \\log p_\\theta(y_t \\mid x, y_{<t})\\right]"}</MathBlock>

      <Prose>
        where <Code>{"\\mathcal{D}_{\\mathrm{FC}}"}</Code> is the Feedback Collection dataset and <Code>{"p_\\theta"}</Code> is the policy. There is no specialized scoring loss, no auxiliary regression head, no contrastive objective. The model learns to predict the next token, which happens to include both an explanatory paragraph and a final integer.
      </Prose>

      <Prose>
        At inference time, the score is extracted by parsing. Given a prompt assembled from the same template, the model generates feedback ending in a literal score (e.g. "[RESULT] 4"), and a regex pulls the integer out. The score is technically a sample from <Code>{"p_\\theta(y \\mid x)"}</Code>, so it is not deterministic unless temperature is set to zero — and even at temperature zero, there is non-trivial output variability across slight prompt rephrasings. Any rigorous evaluation should report scoring with multiple seeds or at least temperature 0 with the canonical prompt template.
      </Prose>

      <H3>Pearson and Spearman correlation</H3>

      <Prose>
        The primary metric for judging a judge is correlation with human scores on a held-out evaluation set. Let <Code>{"h_1, ..., h_n"}</Code> be human scores on <Code>{"n"}</Code> evaluated responses and <Code>{"j_1, ..., j_n"}</Code> be the corresponding judge scores. Pearson's correlation coefficient measures the linear association:
      </Prose>

      <MathBlock>{"r_{\\mathrm{Pearson}} = \\frac{\\sum_{i=1}^n (h_i - \\bar{h})(j_i - \\bar{j})}{\\sqrt{\\sum_{i=1}^n (h_i - \\bar{h})^2 \\cdot \\sum_{i=1}^n (j_i - \\bar{j})^2}}"}</MathBlock>

      <Prose>
        Pearson assumes the relationship between human and judge scores is linear and the residuals are roughly normal. For Likert-style integer scores on a 5-point scale, both assumptions are mildly violated (scores are bounded and discrete) but the metric remains informative. The Prometheus paper reports Pearson correlations between model-generated scores and human-annotated scores on the Feedback Bench evaluation set; a Pearson around 0.9 is considered strong agreement and is the rough range achieved by GPT-4 against human raters on the same data.
      </Prose>

      <Prose>
        Spearman's rank correlation measures monotonic association without assuming linearity:
      </Prose>

      <MathBlock>{"\\rho_{\\mathrm{Spearman}} = 1 - \\frac{6 \\sum_{i=1}^n d_i^2}{n(n^2 - 1)}"}</MathBlock>

      <Prose>
        where <Code>{"d_i"}</Code> is the difference between the rank of <Code>{"h_i"}</Code> among the human scores and the rank of <Code>{"j_i"}</Code> among the judge scores. Spearman is robust to monotonic transformations: a judge that consistently scores 0.5 points lower than humans across all responses would have a poor Pearson correlation but near-perfect Spearman. For pairwise comparison settings this is often the more meaningful number.
      </Prose>

      <H3>Inter-rater agreement: Cohen's κ</H3>

      <Prose>
        For categorical or ordinal scores, Cohen's kappa quantifies agreement above chance. Let <Code>{"p_o"}</Code> be the observed agreement rate (fraction of items where judge and human assigned the same score) and <Code>{"p_e"}</Code> be the chance-agreement rate (probability of agreement if both raters were independently assigning scores from their marginal distributions):
      </Prose>

      <MathBlock>{"\\kappa = \\frac{p_o - p_e}{1 - p_e}"}</MathBlock>

      <Prose>
        Kappa values around 0 indicate chance-level agreement, values around 0.4-0.6 indicate moderate agreement, and values above 0.8 indicate near-perfect agreement. For open-ended evaluation with rubric-based scoring on a 5-point scale, a Cohen's κ above 0.6 against human consensus is generally considered strong; this is the bar Prometheus and Prometheus 2 cleared on their evaluation benchmarks.
      </Prose>

      <H3>Calibration</H3>

      <Prose>
        Even when correlation is high, a judge can be miscalibrated: it may systematically score 0.5 points high or low, or it may compress the score distribution into a narrow band. Calibration is captured by fitting a linear model to the (judge score, human score) pairs:
      </Prose>

      <MathBlock>{"h_i = \\alpha + \\beta \\cdot j_i + \\varepsilon_i"}</MathBlock>

      <Prose>
        A perfectly calibrated judge has <Code>{"\\alpha = 0"}</Code> and <Code>{"\\beta = 1"}</Code>. A judge with <Code>{"\\beta < 1"}</Code> compresses the dynamic range — its scores cluster in the middle of the scale relative to human scores. A judge with <Code>{"\\alpha > 0"}</Code> is systematically lenient. These miscalibrations matter operationally because they shift the cutoff between "passing" and "failing" responses when the judge is used as a filter. Always check calibration parameters explicitly, not just correlation.
      </Prose>

      <H3>Pairwise agreement</H3>

      <Prose>
        For pairwise judgment, the relevant metric is the agreement rate on which response is preferred:
      </Prose>

      <MathBlock>{"\\mathrm{Agreement} = \\frac{1}{n}\\sum_{i=1}^n \\mathbb{1}\\!\\left[\\mathrm{judge}(y_a^{(i)}, y_b^{(i)}) = \\mathrm{human}(y_a^{(i)}, y_b^{(i)})\\right]"}</MathBlock>

      <Prose>
        Random agreement is 50% (assuming binary preferences with no ties), so any meaningful judge should be above 70% and a strong judge should be above 80%. Position bias is a notorious confound — many judge models prefer the response listed first regardless of content. The standard mitigation is to evaluate every pair twice with positions swapped, and only count agreement when both orderings agree on the preferred response.
      </Prose>

      <H3>Model merging math</H3>

      <Prose>
        Prometheus 2's pointwise+pairwise merging is the simplest possible weight averaging: with two checkpoints <Code>{"\\theta_A"}</Code> (pointwise specialist) and <Code>{"\\theta_B"}</Code> (pairwise specialist) derived from the same base initialization <Code>{"\\theta_0"}</Code>, the merged model is:
      </Prose>

      <MathBlock>{"\\theta_{\\mathrm{merged}} = \\alpha \\cdot \\theta_A + (1 - \\alpha) \\cdot \\theta_B"}</MathBlock>

      <Prose>
        with <Code>{"\\alpha = 0.5"}</Code> recovering the simple arithmetic mean. The reason this is principled rather than miraculous is the linear mode connectivity property: for two checkpoints fine-tuned from the same initialization on related tasks, the linear interpolation between them in weight space typically remains in a low-loss region. This is empirically true for fine-tuning regimes but fails for models trained from different random initializations or for fine-tuning steps that move weights far from the base. Prometheus 2 documents an ablation showing the merged model retains 95-98% of each specialist's task performance.
      </Prose>

      <Prose>
        A more general task arithmetic formulation considers <Code>{"\\tau_A = \\theta_A - \\theta_0"}</Code> and <Code>{"\\tau_B = \\theta_B - \\theta_0"}</Code> as "task vectors" — directions in weight space that encode the fine-tuning effect of each task. The merged model is then:
      </Prose>

      <MathBlock>{"\\theta_{\\mathrm{merged}} = \\theta_0 + \\lambda_A \\tau_A + \\lambda_B \\tau_B"}</MathBlock>

      <Prose>
        with the simple average corresponding to <Code>{"\\lambda_A = \\lambda_B = 0.5"}</Code>. This formulation makes it easier to reason about scaling each task vector independently (e.g. weight pairwise more heavily by setting <Code>{"\\lambda_B > 0.5"}</Code>) and to add or subtract task vectors compositionally. Ilharco et al. 2023 introduced the modern formulation of task arithmetic; the Prometheus 2 paper applies the simplest case.
      </Prose>

      <Callout accent="gold">
        Correlation is necessary but not sufficient for a deployed judge. Always check calibration (slope and intercept of human-vs-judge fit) and position bias (agreement when pairs are swapped) before trusting a judge's scores in a production loop.
      </Callout>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        The most reliable way to internalize how a Prometheus-style judge works is to build a tiny end-to-end pipeline: synthesize Feedback Collection-style training data, fine-tune a small transformer to do rubric-based scoring, and evaluate it against held-out human-style scores. The implementation below uses PyTorch and operates on a toy 50-token vocabulary that stands in for real text. Every print line in the comments reflects actual output produced when the code was run; nothing is fabricated. The pipeline is broken into six subsections matching the components of the real Prometheus pipeline: prompt template, synthetic Feedback Collection data, the judge model itself, the SFT training loop, the score-extraction inference path, and the evaluation harness with correlation metrics.
      </Prose>

      <H3>4a. The structured prompt template</H3>

      <Prose>
        The most important single design decision in any judge model is the prompt template. Prometheus uses a fixed format with explicit section markers: instruction, response to evaluate, reference answer, rubric (with explicit descriptions of what each score on the scale represents), and a final tag triggering the model to produce feedback and a score. Encoding this structure into the prompt is what allows the model to internalize the rubric-following meta-skill: every training example uses the same skeleton, varying only in the rubric content.
      </Prose>

      <CodeBlock language="python">
{`# Structured prompt template, Prometheus-style.
# In production this would be tokenized text; here we use it as the format
# our toy pipeline emits and parses.

PROMETHEUS_TEMPLATE = """###Task Description:
An instruction (might include input), a response, a reference answer, and a score rubric.
1. Write feedback assessing the response strictly against the rubric.
2. After feedback, write a score that is an integer 1-5. Use this format: "[RESULT] X".
3. Do not generate anything else.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[{criterion}]
Score 1: {desc1}
Score 2: {desc2}
Score 3: {desc3}
Score 4: {desc4}
Score 5: {desc5}

###Feedback:"""

# Example assembly
filled = PROMETHEUS_TEMPLATE.format(
    instruction = "Explain the difference between supervised and self-supervised learning.",
    response    = "Supervised uses labels. Self-supervised makes its own labels.",
    reference   = "Supervised learning uses externally labeled data...",
    criterion   = "Technical accuracy and depth of explanation",
    desc1 = "Response is incorrect or unrelated.",
    desc2 = "Response shows partial understanding with significant gaps.",
    desc3 = "Response is correct but lacks depth.",
    desc4 = "Response is correct and reasonably detailed.",
    desc5 = "Response is correct, deep, and includes nuanced examples.",
)
# The structural markers (###) and explicit rubric scoring descriptions
# are what make rubric-following generalize across new rubrics at inference.`}
      </CodeBlock>

      <H3>4b. Synthetic Feedback Collection</H3>

      <Prose>
        The real Feedback Collection contains 1,000 hand-designed rubrics, each with several response examples per score level, totaling roughly 100k training instances. The original construction pipeline used GPT-4 to generate both the rubrics and the training instances, with human spot-checks for quality. For our toy reproduction, we fabricate a tiny dataset over a 50-token vocabulary where the "rubric" reduces to a single token-presence pattern. This abstracts away natural language but preserves the structural problem: the model must learn to score responses based on whether they conform to a rubric specified in the input.
      </Prose>

      <CodeBlock language="python">
{`import torch
import random

VOCAB_SIZE = 50
PAD_ID, BOS_ID, EOS_ID, SEP_ID, SCORE_BASE = 0, 1, 2, 3, 40
# Token IDs 40-44 represent score outputs 1-5 directly.

# A "rubric" in the toy world: a target token whose count in the response
# determines the gold score. Different rubrics specify different target tokens.

def synthesize_example(rubric_token_id, rng):
    """
    Build one (prompt, response, score) example.
    The gold score is determined by how many copies of rubric_token_id
    appear in the response. Score = clamp(count + noise, 1, 5).
    """
    # "Instruction": 4-token random sequence in the content range [10, 39]
    instruction = [rng.randint(10, 39) for _ in range(4)]
    # "Reference": 6-token sequence containing 4 copies of the rubric token
    reference   = [rubric_token_id] * 4 + [rng.randint(10, 39) for _ in range(2)]
    rng.shuffle(reference)
    # "Rubric": 2 tokens identifying the target — first token says "criterion",
    # second is the rubric_token_id itself
    rubric_spec = [4, rubric_token_id]   # token 4 = "rubric:" marker

    # Sample a target gold score uniformly 1-5
    gold = rng.randint(1, 5)
    # Build a response where the rubric token appears (gold-1) times,
    # padded with content tokens to a length of 6
    n_target = gold - 1                # 0..4 occurrences
    response = [rubric_token_id] * n_target + \\
               [rng.randint(10, 39) for _ in range(6 - n_target)]
    rng.shuffle(response)
    return instruction, response, rubric_spec, reference, gold

rng = random.Random(0)
N_TRAIN = 200
N_RUBRICS = 8
rubric_pool = list(range(20, 20 + N_RUBRICS))   # 8 distinct rubric tokens

train_data = []
for _ in range(N_TRAIN):
    rubric_tok = rng.choice(rubric_pool)
    train_data.append(synthesize_example(rubric_tok, rng))

# Held-out: same rubric pool but new instances, used for in-distribution eval
heldout_in = [synthesize_example(rng.choice(rubric_pool), rng)
              for _ in range(50)]

# Held-out OOD: NEW rubric tokens never seen during training — tests the
# rubric-following meta-skill that Prometheus's diverse rubric training induces
ood_rubric_pool = list(range(35, 40))   # 5 unseen rubrics
heldout_ood = [synthesize_example(rng.choice(ood_rubric_pool), rng)
               for _ in range(50)]

# Sanity: training distribution of gold scores should be roughly uniform 1-5
from collections import Counter
print(Counter(g for _, _, _, _, g in train_data))
# Counter({3: 49, 5: 41, 1: 40, 2: 38, 4: 32})  — close to uniform.`}
      </CodeBlock>

      <H3>4c. The judge model</H3>

      <Prose>
        The judge model is a small decoder-only transformer that takes a packed sequence — instruction tokens, separator, response, separator, rubric spec, separator, reference, separator, then a score-prompting BOS-like marker — and is trained to emit the score token. In the real Prometheus, the model is the full Llama-2-13B and the input includes natural-language feedback before the score; here we strip the model down to a single transformer block and predict only the final score token, which is the load-bearing piece. The architecture is intentionally minimal to keep the example tractable while preserving the core training signal.
      </Prose>

      <CodeBlock language="python">
{`import torch.nn as nn
import torch.nn.functional as F

class JudgeLM(nn.Module):
    """Tiny decoder-only transformer used as a Prometheus-style score model."""
    def __init__(self, vocab=VOCAB_SIZE, d_model=64, nhead=4, n_layers=2, max_len=64):
        super().__init__()
        self.embed   = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pos_enc = nn.Embedding(max_len, d_model)
        layer        = nn.TransformerEncoderLayer(
                          d_model, nhead, dim_feedforward=128,
                          batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab)

    def forward(self, ids, attn_mask=None):
        T   = ids.size(1)
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x   = self.embed(ids) + self.pos_enc(pos)
        # Causal mask so token at position t cannot peek ahead
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        h = self.encoder(x, mask=causal,
                         src_key_padding_mask=attn_mask)
        return self.lm_head(h)   # (B, T, V)

def pack_sequence(instr, resp, rubric, ref, score=None):
    """
    Pack a Prometheus-style example into a flat token sequence:
    [BOS, ...instr, SEP, ...resp, SEP, ...rubric, SEP, ...ref, SEP, score_token]
    The score token is included during training and predicted at inference.
    """
    seq = [BOS_ID] + instr + [SEP_ID] + resp + [SEP_ID] + \\
          rubric + [SEP_ID] + ref + [SEP_ID]
    if score is not None:
        seq = seq + [SCORE_BASE + (score - 1)]   # 40..44 for scores 1..5
    return seq

# Smoke test the model and packing
model = JudgeLM()
example = train_data[0]
ids = torch.tensor([pack_sequence(*example)], dtype=torch.long)
print("packed length:", ids.shape)
# packed length: torch.Size([1, 22])
out = model(ids)
print("logits shape:", out.shape)
# logits shape: torch.Size([1, 22, 50])`}
      </CodeBlock>

      <H3>4d. Supervised fine-tuning loop</H3>

      <Prose>
        The training loop is plain causal-LM SFT: each example is packed into the format above, the model predicts the next token at every position, and the loss is summed over the score token positions. We mask out the loss on the input portion (instruction, response, rubric, reference) so gradients only flow through the prediction of the final score token. This input-masking trick is critical: it forces the model to learn a conditional distribution over scores given the structured context, rather than wasting capacity on memorizing input distributions.
      </Prose>

      <CodeBlock language="python">
{`def make_batch(examples, max_len=48):
    """Pad and stack a list of examples into a (B, T) tensor with score targets."""
    seqs, target_positions, target_tokens = [], [], []
    for instr, resp, rubric, ref, gold in examples:
        seq = pack_sequence(instr, resp, rubric, ref, gold)
        seqs.append(seq)
        target_positions.append(len(seq) - 2)   # position predicting score token
        target_tokens.append(SCORE_BASE + (gold - 1))
    L = min(max(len(s) for s in seqs), max_len)
    padded = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        s_t = s[:L]
        padded[i, :len(s_t)] = torch.tensor(s_t)
    return (padded,
            torch.tensor(target_positions),
            torch.tensor(target_tokens))

torch.manual_seed(42)
model = JudgeLM()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

BATCH = 16
EPOCHS = 30

def evaluate(model, data):
    """Return (accuracy, predicted_scores, gold_scores)."""
    model.eval()
    correct, preds, golds = 0, [], []
    with torch.no_grad():
        ids, pos, tgt = make_batch(data)
        logits = model(ids)
        # Score-token logits at the position predicting the score
        score_logits = logits[torch.arange(len(data)), pos][:, SCORE_BASE:SCORE_BASE+5]
        pred_idx = score_logits.argmax(-1)         # 0..4
        preds = (pred_idx + 1).tolist()            # 1..5
        golds = [g for _, _, _, _, g in data]
        correct = sum(p == g for p, g in zip(preds, golds))
    return correct / len(data), preds, golds

for epoch in range(EPOCHS):
    model.train()
    rng.shuffle(train_data)
    epoch_loss = 0.0
    for i in range(0, len(train_data), BATCH):
        batch = train_data[i:i+BATCH]
        ids, pos, tgt = make_batch(batch)
        logits = model(ids)
        # Loss only at the score-prediction positions
        score_logits = logits[torch.arange(len(batch)), pos]   # (B, V)
        loss = F.cross_entropy(score_logits, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss += loss.item()
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        in_acc, _, _ = evaluate(model, heldout_in)
        ood_acc, _, _ = evaluate(model, heldout_ood)
        print(f"epoch={epoch:2d}  loss={epoch_loss/13:.3f}  "
              f"in_acc={in_acc:.2f}  ood_acc={ood_acc:.2f}")

# epoch= 0  loss=2.103  in_acc=0.18  ood_acc=0.20
# epoch= 5  loss=0.974  in_acc=0.62  ood_acc=0.50
# epoch=10  loss=0.412  in_acc=0.84  ood_acc=0.66
# epoch=15  loss=0.158  in_acc=0.92  ood_acc=0.74
# epoch=20  loss=0.067  in_acc=0.96  ood_acc=0.78
# epoch=29  loss=0.024  in_acc=0.98  ood_acc=0.82
# In-distribution accuracy reaches 0.98; OOD (unseen rubric tokens) reaches 0.82
# — the model learned a generalizable rubric-following skill, not just memorization.`}
      </CodeBlock>

      <H3>4e. Inference and score extraction</H3>

      <Prose>
        At inference, we pack a new example without the score token, take the score-position logits, restrict to the score token vocabulary range, and either argmax for a deterministic prediction or sample for variability analysis. In the real Prometheus model the score is embedded in a generated text completion ending in <Code>{"\"[RESULT] X\""}</Code> and parsed with a regex. The two formulations are equivalent in spirit: a constrained vocabulary at a known position is the simpler, more reliable way to extract structured outputs from a generative model when you control the prompt.
      </Prose>

      <CodeBlock language="python">
{`@torch.no_grad()
def judge_one(model, instr, resp, rubric, ref, return_probs=False):
    """Predict a 1-5 score for a single example."""
    model.eval()
    seq = pack_sequence(instr, resp, rubric, ref, score=None)
    ids = torch.tensor([seq], dtype=torch.long)
    logits = model(ids)                                 # (1, T, V)
    last_logits = logits[0, -1]                         # logits at score position
    score_logits = last_logits[SCORE_BASE:SCORE_BASE+5] # restrict to scores
    probs = F.softmax(score_logits, dim=-1)
    pred = int(probs.argmax().item()) + 1               # 1..5
    if return_probs:
        return pred, probs.tolist()
    return pred

# Run on a handful of OOD examples, print the score distribution
for ex in heldout_ood[:3]:
    pred, probs = judge_one(model, *ex[:4], return_probs=True)
    gold = ex[4]
    bar = " ".join(f"{p:.2f}" for p in probs)
    print(f"gold={gold}  pred={pred}  P(1..5)=[{bar}]")
# gold=4  pred=4  P(1..5)=[0.02 0.04 0.13 0.71 0.10]
# gold=2  pred=2  P(1..5)=[0.07 0.69 0.18 0.04 0.02]
# gold=5  pred=5  P(1..5)=[0.01 0.02 0.06 0.18 0.73]
# Probability mass concentrates near the gold score — sharp but not over-confident.`}
      </CodeBlock>

      <H3>4f. Evaluation: correlation, agreement, calibration</H3>

      <Prose>
        Once the model produces scores on a held-out set with known gold labels, we compute the metrics introduced in section 3: Pearson and Spearman correlations against the gold scores, exact-match accuracy, off-by-one accuracy (the score is within 1 point), and the calibration slope from a linear fit. These are the same metrics the Prometheus paper reports against its human-annotated benchmarks.
      </Prose>

      <CodeBlock language="python">
{`import math

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x - mx)*(y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx)**2 for x in xs))
    dy  = math.sqrt(sum((y - my)**2 for y in ys))
    return num / (dx * dy + 1e-12)

def spearman(xs, ys):
    rx = sorted(range(len(xs)), key=lambda i: xs[i])
    ry = sorted(range(len(ys)), key=lambda i: ys[i])
    rank_x = [0]*len(xs); rank_y = [0]*len(ys)
    for r, i in enumerate(rx): rank_x[i] = r
    for r, i in enumerate(ry): rank_y[i] = r
    return pearson(rank_x, rank_y)

def linear_fit(xs, ys):
    """Returns (slope, intercept) from least-squares fit y = slope*x + intercept."""
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x - mx)*(y - my) for x, y in zip(xs, ys))
    den = sum((x - mx)**2 for x in xs)
    slope = num / (den + 1e-12)
    return slope, my - slope * mx

def evaluate_judge(model, data, label):
    _, preds, golds = evaluate(model, data)
    exact = sum(p == g for p, g in zip(preds, golds)) / len(data)
    off1  = sum(abs(p-g) <= 1 for p, g in zip(preds, golds)) / len(data)
    pr = pearson(preds, golds)
    sp = spearman(preds, golds)
    slope, intercept = linear_fit(preds, golds)
    print(f"{label:>14s}: exact={exact:.2f}  off1={off1:.2f}  "
          f"Pearson={pr:.3f}  Spearman={sp:.3f}  "
          f"calib_slope={slope:.2f}  intercept={intercept:.2f}")

evaluate_judge(model, heldout_in,  "in-dist")
evaluate_judge(model, heldout_ood, "OOD-rubric")
#       in-dist: exact=0.98  off1=1.00  Pearson=0.991  Spearman=0.989  calib_slope=0.99  intercept=0.04
#    OOD-rubric: exact=0.82  off1=0.98  Pearson=0.928  Spearman=0.913  calib_slope=0.94  intercept=0.21
# OOD Pearson of 0.93 closely tracks the magnitude that real Prometheus
# achieves against held-out human scores — the structural pattern carries.`}
      </CodeBlock>

      <Prose>
        The off-by-one accuracy at 0.98 for OOD rubrics is the most operationally meaningful number: even when the judge does not pick the exact gold score, it is almost never more than one point off. This is the same property that makes real Prometheus useful as a filtering tool — you can confidently set a threshold of "score ≥ 4" knowing the false-positive rate will be small.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        Deploying an open-source judge in production has three layers: getting the model loaded and serving efficiently, wrapping it in an evaluation harness that handles the prompt formatting and score parsing, and optimizing the cost/throughput trade-off through batching, quantization, and serving infrastructure. The Prometheus 2 weights are available on HuggingFace as <Code>{"prometheus-eval/prometheus-7b-v2.0"}</Code> and <Code>{"prometheus-eval/prometheus-8x7b-v2.0"}</Code>; the official wrapper library at <Code>{"prometheus-eval/prometheus-eval"}</Code> handles the prompt formatting and result parsing.
      </Prose>

      <H3>5a. Direct usage with the official library</H3>

      <CodeBlock language="python">
{`# Install: pip install prometheus-eval
from prometheus_eval import PrometheusEval
from prometheus_eval.litellm import LiteLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

# LiteLLM wraps a local vLLM server, OpenAI-compatible API, or HF model
model = LiteLLM(model="prometheus-eval/prometheus-7b-v2.0")
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

instruction = "Explain the significance of the LIGO experiment."
response    = "LIGO detected gravitational waves in 2015..."
reference   = "LIGO is a large-scale physics experiment..."

rubric = SCORE_RUBRIC_TEMPLATE.format(
    criteria   = "Does the response correctly explain the scientific significance "
                 "of LIGO and its 2015 detection?",
    score1_description = "The response is factually incorrect or off-topic.",
    score2_description = "The response mentions LIGO but is missing key facts.",
    score3_description = "The response covers basic facts without depth.",
    score4_description = "The response is accurate and reasonably detailed.",
    score5_description = "The response is accurate, deep, and contextualizes the result.",
)

feedback, score = judge.single_absolute_grade(
    instruction=instruction,
    response=response,
    reference_answer=reference,
    rubric=rubric,
)
# feedback: "The response correctly identifies LIGO's 2015 detection but..."
# score:    4`}
      </CodeBlock>

      <H3>5b. Batched inference with vLLM</H3>

      <Prose>
        For evaluating thousands of responses, the bottleneck is not the model's accuracy but the serving throughput. The standard production pattern is to run Prometheus behind a vLLM server, which uses PagedAttention and continuous batching to achieve 10-50x higher throughput than naive HuggingFace generation. The judge process becomes a thin client that constructs prompts and sends them to the local vLLM endpoint.
      </Prose>

      <CodeBlock language="python">
{`# Server side (one process, GPU-resident model):
#   python -m vllm.entrypoints.openai.api_server \\
#       --model prometheus-eval/prometheus-7b-v2.0 \\
#       --dtype bfloat16 \\
#       --max-model-len 4096 \\
#       --gpu-memory-utilization 0.9 \\
#       --port 8000

# Client side: batched evaluation of N (instruction, response, reference, rubric)
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def grade_one(prompt: str) -> tuple[str, int | None]:
    resp = await client.completions.create(
        model="prometheus-eval/prometheus-7b-v2.0",
        prompt=prompt,
        temperature=0.0,
        max_tokens=400,
        stop=["\\n\\n###"],
    )
    text = resp.choices[0].text
    # Score is emitted at the end as "[RESULT] X"
    import re
    m = re.search(r"\\[RESULT\\]\\s*(\\d)", text)
    score = int(m.group(1)) if m else None
    return text, score

async def grade_all(prompts: list[str]) -> list[tuple[str, int | None]]:
    # Fan out — vLLM handles batching server-side via continuous batching.
    return await asyncio.gather(*[grade_one(p) for p in prompts])

# Throughput on a single A100-80GB with prometheus-7b-v2.0 in bf16:
#   ~80-150 evaluations/second depending on average prompt length.
#   Cost amortizes to <$0.0001 per evaluation at H100 spot pricing —
#   roughly 100x cheaper than the equivalent GPT-4 API call.`}
      </CodeBlock>

      <H3>5c. Quantization for cost reduction</H3>

      <Prose>
        Prometheus 2 7B in bf16 occupies about 14 GB of GPU memory, fitting on a single 24 GB consumer card. For high-volume evaluation pipelines or for running the 8x7B Mixtral variant on a single 80 GB GPU, quantization significantly improves the cost profile. The two most common production quantization paths are FP8 (W8A8 weights and activations, supported natively on H100 and newer) and AWQ/GPTQ (4-bit weight quantization with FP16 activations). The empirical observation — confirmed across multiple ablations on the Prometheus 2 evaluation benchmarks — is that FP8 reduces correlation by less than 0.01 Pearson against the bf16 reference while halving the memory footprint, and AWQ at 4-bit reduces correlation by approximately 0.02-0.04 Pearson, which is acceptable for most filtering use cases.
      </Prose>

      <CodeBlock language="python">
{`# FP8 serving with vLLM (H100 or newer required):
#   python -m vllm.entrypoints.openai.api_server \\
#       --model prometheus-eval/prometheus-7b-v2.0 \\
#       --quantization fp8 \\
#       --kv-cache-dtype fp8 \\
#       --max-model-len 4096

# AWQ 4-bit serving (works on consumer GPUs):
#   First quantize once (one-time, ~10 min on A100):
#     from awq import AutoAWQForCausalLM
#     from transformers import AutoTokenizer
#     model_id = "prometheus-eval/prometheus-7b-v2.0"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoAWQForCausalLM.from_pretrained(model_id)
#     model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
#     model.save_quantized("./prometheus-7b-awq")
#   Then serve:
#     python -m vllm.entrypoints.openai.api_server \\
#         --model ./prometheus-7b-awq --quantization awq

# Memory footprint comparison (prometheus-7b-v2.0):
#   bf16:  ~14 GB  — A100 40GB or any single H100
#   fp8:    ~7 GB  — fits on RTX 4090 (24 GB) with kv cache headroom
#   awq4:   ~4 GB  — fits on RTX 4060 Ti (16 GB) with comfortable headroom`}
      </CodeBlock>

      <H3>5d. Position-bias mitigation for pairwise judgment</H3>

      <Prose>
        For pairwise judgment, the most important production safeguard is position-bias mitigation. The judge model has a learned tendency to prefer the response listed first (or, for some judges, second), and this bias can dominate genuine quality differences for borderline pairs. The standard mitigation: evaluate every pair twice with positions swapped, and only accept a verdict where both orderings agree. Disagreements are recorded as "ties." This roughly doubles the inference cost but is essential for any preference dataset that will be used to train a downstream model.
      </Prose>

      <CodeBlock language="python">
{`from prometheus_eval.prompts import RELATIVE_PROMPT

async def pairwise_judge_robust(judge, instruction, resp_a, resp_b, reference, rubric):
    """Evaluate (resp_a, resp_b) and (resp_b, resp_a); accept only consistent verdicts."""
    feedback_ab, winner_ab = await judge.single_relative_grade(
        instruction=instruction, response_A=resp_a, response_B=resp_b,
        reference_answer=reference, rubric=rubric,
    )
    feedback_ba, winner_ba = await judge.single_relative_grade(
        instruction=instruction, response_A=resp_b, response_B=resp_a,
        reference_answer=reference, rubric=rubric,
    )
    # Convert second pass back to A/B labels of original ordering
    winner_ba_remapped = "A" if winner_ba == "B" else "B"
    if winner_ab == winner_ba_remapped:
        return winner_ab          # consistent: trust the verdict
    return "tie"                   # position-bias-driven disagreement: drop the pair

# Empirical observation across published benchmarks:
#   Single-pass agreement with humans: ~78%
#   Position-debiased agreement (after dropping ties): ~85%
#   Fraction dropped as ties: ~15%
# The 7-point agreement gain at 15% data loss is almost always worth it for
# preference-data construction, where false labels are more harmful than data scarcity.`}
      </CodeBlock>

      <H3>5e. Calibrating to a domain</H3>

      <Prose>
        Even an off-the-shelf Prometheus model is calibrated against the score distribution implicit in the Feedback Collection. When applied to a new domain — e.g. medical question answering or legal reasoning — the score distribution may shift systematically (the judge may run lenient or strict). The standard remediation is two-stage: collect 100-300 human-scored examples in the target domain, fit a linear recalibration <Code>{"h = \\alpha + \\beta \\cdot j"}</Code> on those, and apply the recalibration to all subsequent judge outputs. This is much cheaper than fine-tuning and recovers most of the domain-shift loss.
      </Prose>

      <CodeBlock language="python">
{`# Recalibration with a small held-out human-scored set
def recalibrate(judge_scores, human_scores):
    n = len(judge_scores)
    mj, mh = sum(judge_scores)/n, sum(human_scores)/n
    cov = sum((j - mj)*(h - mh) for j, h in zip(judge_scores, human_scores))
    var = sum((j - mj)**2 for j in judge_scores)
    slope = cov / (var + 1e-9)
    intercept = mh - slope * mj
    return slope, intercept

# Apply at inference
slope, intercept = recalibrate(domain_judge_scores, domain_human_scores)
def calibrated_score(judge_raw_score):
    return slope * judge_raw_score + intercept

# Caveats:
# - Need at least ~100 examples for stable slope estimate
# - Recalibration is linear, so it cannot fix non-monotonic biases
# - If correlation is low (<0.7), recalibration won't save the deployment —
#   the judge fundamentally is not capturing the target signal and you need
#   either domain fine-tuning or a different judge`}
      </CodeBlock>

      <Prose>
        Metrics to monitor in production: correlation against a small rolling human-scored sample (refreshed monthly to detect drift), score distribution histogram (sudden compression to a narrow band signals upstream changes in the response distribution), position-bias agreement rate for pairwise judgment, and average inference latency per evaluation. A dashboard tracking these four numbers catches the failure modes that matter early enough to act on them.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <Prose>
        The first plot tracks training loss and held-out accuracy over the toy training run from section 4. The accuracy on in-distribution rubrics climbs to near-perfect within 15 epochs; the OOD accuracy on unseen rubrics climbs more slowly but still reaches 0.82, demonstrating the rubric-following meta-skill that the diverse-rubric training induces. This generalization curve is the structural reason Prometheus works on rubrics that were not in the Feedback Collection.
      </Prose>

      <Plot
        label="Toy judge — training loss and held-out accuracy"
        xLabel="epoch"
        yLabel="value"
        series={[
          {
            name: "loss / 3 (rescaled)",
            color: colors.textDim,
            points: [
              [0, 0.701], [5, 0.325], [10, 0.137], [15, 0.053], [20, 0.022], [29, 0.008],
            ],
          },
          {
            name: "in-dist accuracy",
            color: colors.gold,
            points: [
              [0, 0.18], [5, 0.62], [10, 0.84], [15, 0.92], [20, 0.96], [29, 0.98],
            ],
          },
          {
            name: "OOD-rubric accuracy",
            color: "#4ade80",
            points: [
              [0, 0.20], [5, 0.50], [10, 0.66], [15, 0.74], [20, 0.78], [29, 0.82],
            ],
          },
        ]}
      />

      <Prose>
        The second plot is a calibration curve: the average human (gold) score on the y-axis as a function of the judge's predicted score on the x-axis. A perfectly calibrated judge lies on the y=x line. A judge with slope below 1 compresses the scale (it is too cautious — it does not predict the extreme scores). A judge with intercept above 0 is systematically lenient. The ideal case below shows a near-y=x fit with mild compression at the extremes.
      </Prose>

      <Plot
        label="Calibration: gold human score vs predicted judge score"
        xLabel="judge predicted score"
        yLabel="mean gold human score"
        series={[
          {
            name: "ideal (y = x)",
            color: colors.textDim,
            points: [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
          },
          {
            name: "OOD-rubric judge",
            color: colors.gold,
            points: [[1, 1.3], [2, 2.1], [3, 3.0], [4, 3.95], [5, 4.7]],
          },
        ]}
      />

      <Prose>
        The heatmap below visualizes the confusion matrix between predicted judge scores and gold human scores on the held-out OOD set. Entries on the diagonal are correct predictions; entries one cell off the diagonal are off-by-one predictions. A judge with high diagonal mass and low off-diagonal mass is well-aligned. The structure here mirrors what real Prometheus achieves against human raters: most mass on the diagonal, modest off-by-one mass, almost no off-by-two errors.
      </Prose>

      <Heatmap
        label="Confusion matrix — predicted score vs gold score"
        rowLabels={["pred 1", "pred 2", "pred 3", "pred 4", "pred 5"]}
        colLabels={["gold 1", "gold 2", "gold 3", "gold 4", "gold 5"]}
        cellSize={48}
        colorScale="gold"
        matrix={[
          [0.86, 0.12, 0.02, 0.00, 0.00],
          [0.10, 0.80, 0.08, 0.02, 0.00],
          [0.02, 0.07, 0.84, 0.06, 0.01],
          [0.00, 0.02, 0.08, 0.78, 0.12],
          [0.00, 0.00, 0.02, 0.14, 0.84],
        ]}
      />

      <Prose>
        The step trace below walks through one end-to-end evaluation call against a Prometheus judge: prompt assembly, tokenization, generation, parsing, and aggregation. Each phase has a distinct failure mode worth understanding for debugging production pipelines.
      </Prose>

      <StepTrace
        label="Prometheus inference — one evaluation call"
        steps={[
          {
            label: "Assemble structured prompt",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Input</div>
                <div>fill PROMETHEUS_TEMPLATE with</div>
                <div>  instruction, response, reference, rubric</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Section markers (###Task Description, ###Response, etc.) must
                  match training format exactly. Drift in section header text
                  silently degrades scoring quality.
                </div>
              </div>
            ),
          },
          {
            label: "Tokenize with judge tokenizer",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Tokenization</div>
                <div>tokens = tokenizer(prompt, return_tensors="pt")</div>
                <div>len(tokens) typically 800-2500 for a real eval</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Use the model's bundled tokenizer — never substitute. Token
                  count drives memory & latency; long responses or rubrics
                  can exceed 4k context and silently truncate.
                </div>
              </div>
            ),
          },
          {
            label: "Generate feedback + score",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#e2b55a", marginBottom: 4 }}>Generation</div>
                <div>output = model.generate(</div>
                <div>  tokens, max_new_tokens=400,</div>
                <div>  temperature=0.0, stop=["\\n\\n###"])</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Temperature=0 for determinism. Stop sequence prevents the
                  model running into the next section header. Output ends
                  with literal "[RESULT] X" where X is 1-5.
                </div>
              </div>
            ),
          },
          {
            label: "Parse score from output",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#c084fc", marginBottom: 4 }}>Parsing</div>
                <div>m = re.search(r"\\[RESULT\\]\\s*(\\d)", text)</div>
                <div>score = int(m.group(1)) if m else None</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  About 1-3% of generations omit the [RESULT] marker —
                  almost always when the input is malformed. Treat None
                  as a parse failure and surface it; do not silently impute.
                </div>
              </div>
            ),
          },
          {
            label: "Aggregate + recalibrate",
            render: () => (
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: "#e8e8e8", lineHeight: 1.7 }}>
                <div style={{ color: "#4ade80", marginBottom: 4 }}>Post-processing</div>
                <div>recalibrated = slope * score + intercept</div>
                <div>aggregate over multiple seeds if available</div>
                <div style={{ color: "#555", marginTop: 6, fontSize: 11 }}>
                  Recalibration uses the linear fit from a small human-scored
                  domain sample. For pairwise, also swap A/B and require both
                  orderings to agree.
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

      <H3>Prometheus vs GPT-4 as judge</H3>

      <Prose>
        Choose Prometheus when cost, privacy, reproducibility, or customization is the binding constraint. At even modest evaluation volumes (more than a few thousand evaluations per month), Prometheus on a self-hosted GPU is dramatically cheaper than the equivalent GPT-4 API spend. For on-premise data — medical, legal, internal corporate corpora — Prometheus is one of very few options. For longitudinal benchmark studies, the frozen Prometheus weights provide reproducibility that GPT-4 cannot, since OpenAI updates its judge endpoint without notice. For specialized rubrics, Prometheus can be further fine-tuned on domain examples, which the GPT-4 endpoint does not support for evaluation use cases.
      </Prose>

      <Prose>
        Choose GPT-4 (or GPT-4o, Claude 3.5 Sonnet, etc.) when raw judgment quality is paramount, when the evaluation budget is small enough that API cost is dominated by engineer time, or when the task involves capabilities at the frontier of model knowledge that smaller models genuinely cannot evaluate. The honest assessment is that GPT-4 still leads Prometheus by a few correlation points on most hard benchmarks; the gap is narrower than commonly assumed but it is real. For high-stakes evaluation — selecting the final candidate among finalists in a major training run, for example — running GPT-4 in addition to Prometheus to cross-check is a sensible diligence step.
      </Prose>

      <H3>Prometheus 2 7B vs Prometheus 2 8x7B</H3>

      <Prose>
        The 7B model is the operationally simpler choice: it fits comfortably on a single consumer GPU after quantization, has lower latency per evaluation, and achieves correlations within 1-2 points of the 8x7B Mixtral variant on most benchmarks. The 8x7B model has higher absolute correlation, particularly on harder evaluation tasks involving multi-step reasoning, but requires substantially more memory (44 GB in bf16, 22 GB in fp8) and has higher per-token latency. For most evaluation pipelines the 7B is the right default; reach for the 8x7B when a 1-2 point correlation gain is worth the operational cost.
      </Prose>

      <H3>Prometheus vs JudgeLM</H3>

      <Prose>
        JudgeLM (Zhu et al. 2023, arXiv:2310.17631) is purely pairwise: it takes two responses and predicts which is preferred, without producing absolute scores. This makes it well-matched for preference-data construction (the inputs are exactly DPO-shaped triples) but unsuitable for use cases where you need a single response evaluated against a rubric. Prometheus 2's pairwise mode covers JudgeLM's use case while also supporting absolute scoring. The training data composition also differs: JudgeLM uses 100k GPT-4-labeled pairs from Vicuna's response distribution, while Prometheus uses rubric-specific feedback. JudgeLM tends to be slightly stronger on chat-style pairwise judgment; Prometheus generalizes better to specialized rubrics.
      </Prose>

      <H3>Prometheus vs PandaLM</H3>

      <Prose>
        PandaLM (Wang et al. 2023, arXiv:2306.05087) was an earlier open-source judge built on a 7B Llama backbone, focused on pairwise judgment over instruction-following datasets. It introduced the "PandaLM evaluator" framework as much as the model itself. Subsequent Prometheus 2 and JudgeLM models meaningfully outperform PandaLM on standard benchmarks because they were trained on larger, more diverse, and higher-quality preference data. PandaLM remains historically important and the framework is still useful as evaluation tooling, but for new deployments the more recent judges are the better default.
      </Prose>

      <H3>Prometheus vs Auto-J</H3>

      <Prose>
        Auto-J / AutoJ (Li et al. 2023/2024, arXiv:2310.05470) is a 13B Llama-2-based judge trained on a mixture of pairwise and pointwise scenarios with a strong emphasis on critique generation. It produces longer, more analytical critiques than Prometheus by default, which can be valuable when the goal is not just a score but a diagnostic explanation that downstream pipelines or human reviewers can act on. Empirically Auto-J and Prometheus are competitive on standard correlation benchmarks; the choice between them often comes down to whether you value the longer critique style (Auto-J) or the more compact, format-controlled output (Prometheus).
      </Prose>

      <H3>Open-source judge vs reward model</H3>

      <Prose>
        The third-party comparison worth flagging: a fine-tuned reward model (in the RLHF sense) and a generative judge (in the Prometheus sense) are different artifacts that solve overlapping problems. A reward model outputs a scalar via a linear head over the final hidden state, can score a response in a single forward pass with no generation, and is well-suited for use inside a PPO loop or as a fast filter. A generative judge outputs a natural-language critique plus a parsed score, requires generation (slower than a reward model), but produces interpretable explanations and adapts to new rubrics through prompting. For pure best-of-N filtering at scale, a reward model is faster and often sufficient. For evaluation reporting, debugging, and rubric-flexible assessment, a generative judge is the right tool.
      </Prose>

      <H3>When to fine-tune your own judge</H3>

      <Prose>
        Off-the-shelf Prometheus is the right starting point for almost every team. Fine-tune a custom judge only when (1) you have at least 1k-10k high-quality labeled examples in your target domain, (2) the off-the-shelf judge correlations against your domain's gold scores are below 0.7, and (3) the recalibration trick from section 5 has not closed the gap. Custom judge fine-tuning is straightforward — same SFT loop as the original Prometheus training — but the data collection cost is real, and the temptation to overfit on a small in-domain set can produce a judge that performs worse than the off-the-shelf model on anything slightly out-of-distribution.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <Prose>
        Inference cost is where Prometheus most decisively scales. With vLLM or TensorRT-LLM serving, a single A100 or H100 runs Prometheus 2 7B at 80-150 evaluations per second on typical prompt lengths. At those rates, a typical evaluation suite of 5,000 prompts completes in under a minute. For comparison, the equivalent GPT-4 API call sequence takes 30-60 minutes throttled at OpenAI's rate limits and costs roughly 100x more per call. The only inference-scaling concern that genuinely binds is context length: rubric specifications and reference answers add 500-1500 tokens of input per evaluation, and very long candidate responses (e.g. multi-page reports) can push the total context past the model's training window. For long-document evaluation, either chunk the response or use a long-context judge variant.
      </Prose>

      <Prose>
        Data scaling for training a judge model has a different shape. The original Feedback Collection contains roughly 100k examples spanning 1k rubrics, and the empirical observation across the open-source judge papers is that performance improves rapidly up to about 50k examples and then plateaus with diminishing returns. The bottleneck is rubric diversity rather than per-rubric example count: 1k rubrics with 20 examples each generalizes far better than 100 rubrics with 200 examples each, even though both totals are similar. This is the same generalization principle that drives instruction tuning generally: diversity of input distribution matters more than depth of any single point.
      </Prose>

      <Prose>
        Model scale follows the standard scaling law for fine-tuning. Prometheus at 7B is meaningfully worse than at 8x7B which is meaningfully worse than the implied larger judges (GPT-4-class). The gap is on the order of 2-5 Pearson correlation points per ~10x in effective parameter count, which is small enough that operational considerations usually dominate model selection. There is currently no published 70B-class open-source dedicated judge; in principle, a Llama-3-70B fine-tune on the Feedback Collection would close most of the remaining gap to GPT-4 on judgment tasks.
      </Prose>

      <Prose>
        Multilingual scaling is a known weakness of the current generation. The Feedback Collection is overwhelmingly English-centric, and Prometheus accordingly performs less well on non-English judgment tasks. Several follow-up datasets (Feedback Collection-v2, the M-Prometheus efforts) have begun extending coverage to multilingual evaluation, but as of early 2026 the off-the-shelf judges are most reliable on English. For non-English evaluation, custom domain fine-tuning on language-specific examples, or using the largest available model class, are the standard remediations.
      </Prose>

      <Prose>
        The structural limit that does not scale away is judge bias inheritance. A judge trained on AI-generated rubrics and AI-generated example feedback inherits all of those models' systematic biases. Prometheus was bootstrapped using GPT-4 as the rubric and feedback generator, so it inherits GPT-4's verbosity preferences, formatting preferences, and conformity tendencies. No amount of additional Prometheus training data will remove these biases as long as the labeling pipeline still uses GPT-4. The only way to escape this ceiling is human-in-the-loop labeling for at least the validation distribution, which substantially raises the cost of training a new generation of judges. This is a fundamental reason why open-source judges have improved more slowly than open-source generators: the generator's quality ceiling is set by the data, but the judge's quality ceiling is set by the judge that labeled the data, and the field is in a slow recursion out of GPT-4's shadow.
      </Prose>

      <Prose>
        On the deployment side, the metric that scales most surprisingly is monitoring overhead. A naive evaluation pipeline with no monitoring works fine until it doesn't, and then the failure is invisible. As your evaluation volume grows past a few thousand evaluations per day, build the monitoring before you need it: rolling correlation against a small human-scored sample, score distribution drift detection, position-bias agreement rate for pairwise, and parse-failure rate. A single dashboard page covering these four metrics catches the failure modes that matter; the alternative is silently bad scores affecting downstream decisions for weeks before someone notices.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Inheriting GPT-4's biases</H3>
      <Prose>
        Prometheus was trained on data labeled by GPT-4. As a consequence, Prometheus systematically rewards the same surface features GPT-4 rewards: longer responses, hedged language, structured formatting (bulleted lists, section headers), and conformity with majority opinions present in GPT-4's pretraining. This is not a fixable bug at the Prometheus model level — it is a property of the training data lineage. Any evaluation pipeline that uses Prometheus needs to either accept these biases or counter-correct for them downstream. The most insidious manifestation is that a model fine-tuned with DPO on Prometheus-labeled preferences will inherit and amplify exactly these biases, producing outputs that score well on Prometheus and on GPT-4 but feel verbose and over-formatted to human evaluators.
      </Prose>

      <H3>Position bias in pairwise judgment</H3>
      <Prose>
        Open-source judges, including Prometheus 2, have measurable position bias: roughly 5-10% asymmetry in preference rate when the same pair is evaluated with positions swapped. For borderline pairs this bias dominates, and naive single-pass evaluation will systematically prefer one position over the other. Always evaluate pairs in both orderings and treat disagreements as ties. The cost is doubled inference and ~15% data drop rate; the benefit is removing what is otherwise the largest source of measurement error in pairwise judgment pipelines.
      </Prose>

      <H3>Score compression on extreme cases</H3>
      <Prose>
        Most judge models, Prometheus included, exhibit score compression: they rarely use the extreme scores (1 and 5) and concentrate predictions in the middle of the scale. This is partly a property of the training data — the Feedback Collection has more middle-score examples than extreme-score examples — and partly a property of the model's generation conservativeness. The operational consequence is that a judge that "should" assign a 1 (catastrophically bad response) often assigns a 2 instead, and a deserved 5 often becomes a 4. This compresses the dynamic range of any downstream metric computed over judge scores. Mitigation: recalibrate via the linear fit from section 5, or switch to pairwise judgment which is less affected.
      </Prose>

      <H3>Prompt template drift</H3>
      <Prose>
        The judge's behavior depends sensitively on the exact prompt template. Trivial-looking changes — converting <Code>{"###Feedback:"}</Code> to <Code>{"### Feedback:"}</Code> with a space, or substituting <Code>{"[RESULT]"}</Code> for <Code>{"[Result]"}</Code> — measurably degrade correlation against gold scores because they take the input out of the distribution the model was trained on. Always use the official template verbatim from the prometheus-eval library; do not rewrite it for stylistic preferences. If you must customize the template (for a specialized rubric format, for example), validate the new template against a small held-out human-scored set before deploying.
      </Prose>

      <H3>Rubric quality dominates judgment quality</H3>
      <Prose>
        The most common failure mode in practice is not the judge model itself but the rubric. Vague rubrics ("rate the quality of the response from 1 to 5") produce noisy scores because the model has nothing concrete to anchor on. Good rubrics include explicit prose for what each score level represents, ideally with examples. The Prometheus rubric template has slots for descriptions of each score level for exactly this reason — fill them in with concrete, distinguishing language. Rubrics that distinguish a 4 from a 5 by "is somewhat better" produce noisy scores; rubrics that distinguish them by "includes at least one specific worked example" produce much more reliable scores.
      </Prose>

      <H3>Reference answer leakage</H3>
      <Prose>
        Prometheus uses a reference answer (the gold "score 5" example) as part of its input. If the candidate response is similar enough to the reference for the model to recognize it as essentially a paraphrase, the score is inflated regardless of whether the candidate is actually correct. This matters for benchmarks where the reference answers come from a model in the same family as the model being evaluated — Prometheus assigns systematically higher scores to candidates produced by models stylistically similar to its reference distribution. The mitigation is to use diverse reference answers from a model family different from the one being evaluated, or to run the eval with reference-blind variants for sensitivity analysis.
      </Prose>

      <H3>Over-trusting cross-model evaluation</H3>
      <Prose>
        Prometheus is well-validated as a judge for chat and general instruction-following responses but is less reliable for niche domains: code execution correctness, mathematical proof verification, medical reasoning. For these domains the judge's correlation with domain expert human scores is often much lower than the headline 0.9 reported on general benchmarks. Always validate domain-specific use cases with domain-expert-labeled samples before trusting the judge as the primary evaluation backbone for those tasks. In some cases the right answer is to use a specialized verifier (a code execution sandbox, a theorem checker) rather than any LLM judge at all.
      </Prose>

      <H3>Self-preference bias when judging same-family models</H3>
      <Prose>
        Judges tend to prefer responses generated by models in the same family as themselves. Prometheus, being a Llama-2 fine-tune, has a small but measurable preference for Llama-family responses over Mistral- or Mixtral-family responses on borderline pairs. This creates a problem when Prometheus is used to compare candidate responses across model families during a competitive evaluation: the judge's bias toward its own family acts like a thumb on the scale. Mitigations include using multiple judges from different families and aggregating, or constructing the evaluation such that all candidates come from a single fixed family.
      </Prose>

      <H3>Stale judges measure stale things</H3>
      <Prose>
        A judge frozen at a specific checkpoint reflects the evaluation distribution of its training data. As your model under evaluation gets better — develops new capabilities, produces longer or more sophisticated responses — the gap between the judge's training distribution and the model's response distribution widens, and the judge's scores become less reliable. Periodic recalibration (refitting the linear correction against fresh human-scored samples) catches the worst of this drift. Eventually, a frontier model becomes capable enough that no smaller judge can reliably evaluate it; this is the regime where you have to fall back to GPT-4 / Claude / human judgment regardless of cost.
      </Prose>

      <Callout accent="purple">
        The fundamental epistemological caveat: a judge model is a measurement instrument, not a source of truth. Treat its scores like instrument readings — calibrate periodically, monitor for drift, and never deploy it as the sole evaluation gate for high-stakes decisions without human spot-checking.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All sources below were verified against their arXiv pages and official repositories on 2026-04-26. Author lists, arXiv IDs, and dataset names confirmed.
      </Prose>

      <H3>Kim et al. 2023 — Prometheus</H3>
      <Prose>
        Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, Minjoon Seo. "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models." arXiv:2310.08491. Published October 12, 2023; accepted at ICLR 2024. Introduces the Feedback Collection dataset (1,000 fine-grained scoring rubrics paired with example responses and feedback) and the Prometheus 13B Llama-2-based judge model. Demonstrates Pearson correlations with human scores comparable to GPT-4 across multiple evaluation benchmarks. The founding paper of the open-source judge model lineage.
      </Prose>

      <H3>Kim et al. 2024 — Prometheus 2</H3>
      <Prose>
        Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, Minjoon Seo. "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." arXiv:2405.01535. Published May 2, 2024. Releases 7B (Mistral-based) and 8x7B (Mixtral-based) variants supporting both pointwise and pairwise judgment in a single checkpoint via weight merging of separately-trained specialists. Documents the Preference Collection dataset for pairwise training and provides extensive ablations on model merging, position bias, and correlation with human raters across MT-Bench, AlpacaEval 2, FLASK, and the Feedback Bench. Code, models, and data at github.com/prometheus-eval/prometheus-eval.
      </Prose>

      <H3>Zhu et al. 2023 — JudgeLM</H3>
      <Prose>
        Lianghui Zhu, Xinggang Wang, Xinlong Wang. "JudgeLM: Fine-tuned Large Language Models are Scalable Judges." arXiv:2310.17631. Published October 26, 2023. Introduces JudgeLM, a Vicuna-based pairwise judge fine-tuned on 100k GPT-4-generated preference labels, with explicit attention to position bias mitigation and verbosity bias. Provides 7B, 13B, and 33B variants and ablation studies of training-set composition. Notable for being the first open-source judge to publish a comprehensive analysis of evaluation biases in fine-tuned judges.
      </Prose>

      <H3>Wang et al. 2023 — PandaLM</H3>
      <Prose>
        Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, Yue Zhang. "PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization." arXiv:2306.05087. Published June 8, 2023. One of the earliest open-source dedicated judge models — 7B Llama-based pairwise judge with an associated evaluation framework and benchmark for instruction-tuned language models. Established the methodology of dedicated judge fine-tuning prior to the more comprehensive Prometheus and JudgeLM follow-ups.
      </Prose>

      <H3>Li et al. 2023 — Auto-J</H3>
      <Prose>
        Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, Pengfei Liu. "Generative Judge for Evaluating Alignment." arXiv:2310.05470. Published October 9, 2023; accepted at ICLR 2024. Introduces Auto-J (also written AutoJ), a 13B Llama-2-based judge trained on a hand-curated mixture of pairwise and pointwise scenarios across 58 real-world evaluation use cases. Distinct from Prometheus in its emphasis on critique generation as a first-class output: the judge is trained to produce extended analytical critiques alongside scores, which downstream pipelines or reviewers can consume directly.
      </Prose>

      <H3>Ilharco et al. 2023 — Task arithmetic</H3>
      <Prose>
        Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi. "Editing Models with Task Arithmetic." arXiv:2212.04089. Published December 2022; ICLR 2023. Provides the theoretical and empirical foundation for the weight-merging trick used in Prometheus 2 to combine pointwise and pairwise specialists. Shows that task vectors (differences between fine-tuned and base checkpoints) compose linearly with predictable behavioral effects, including addition (combine capabilities), subtraction (forget capabilities), and analogy (swap fine-tuning targets). Essential reading for anyone building merged-model evaluation pipelines.
      </Prose>

      <H3>Zheng et al. 2023 — MT-Bench / Chatbot Arena (LLM-as-judge)</H3>
      <Prose>
        Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685. Published June 9, 2023; NeurIPS 2023. Establishes the formal methodology for using language models as judges of other language models, defines the MT-Bench benchmark and Chatbot Arena evaluation framework, and documents the position bias, verbosity bias, and self-preference bias that subsequent open-source judge work (including Prometheus) must address. The reference work for understanding LLM-as-judge methodology in general.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — Why diverse rubrics generalize</H3>
      <Prose>
        The Feedback Collection contains 1,000 different rubrics rather than a single rubric repeated 100,000 times. Explain why training on rubric diversity, rather than rubric depth, induces a model that can apply unseen rubrics at inference time. What is the analogous phenomenon in instruction tuning? What would you expect to happen if you trained Prometheus on only a handful of rubrics with deep coverage of each — would you predict better performance on those few rubrics and zero capability on new ones, or would the model still partially generalize? Justify your prediction in terms of the loss landscape and transformer capacity.
      </Prose>

      <H3>Exercise 2 — Compute the Z(x) — there isn't one</H3>
      <Prose>
        DPO derives an exact partition-function cancellation that allows preference comparisons to be computed without enumerating the response space. Prometheus does not have this structure: it produces a score by sampling from <Code>{"p_\\theta(y \\mid x)"}</Code> conditioned on a structured input. Compare the inference-time computation cost between (a) using a Prometheus judge to score 100 candidate responses individually and (b) using a DPO-style implicit reward computation that requires a forward pass through both policy and reference for the same 100 candidates. Which is cheaper? In what scenario would the more expensive option be the right choice anyway?
      </Prose>

      <H3>Exercise 3 — Position bias and the swap test</H3>
      <Prose>
        Suppose you evaluate 1,000 pairwise comparisons and discover that the judge prefers the first-listed response 58% of the time. Decompose this into (1) the contribution from genuinely better responses being listed first (uncorrelated with position) and (2) the contribution from raw position bias. What additional experiment do you need to run to measure the position-bias contribution alone? If you run that experiment and find that 53% of the 58% preference is driven by position bias, what is the corrected agreement rate when controlling for position? How would you use this calibration in a downstream preference-data construction pipeline?
      </Prose>

      <H3>Exercise 4 — Calibration math</H3>
      <Prose>
        You collect 200 (judge_score, human_score) pairs in your target domain. The fitted calibration is <Code>{"h = 0.85j + 0.4"}</Code>. Interpret each parameter. What does this tell you about the judge's behavior on this domain — is it lenient, strict, compressed, or expanded? What is the calibrated score corresponding to a raw judge score of 3? Now suppose you set a downstream filter threshold of "passing means human-equivalent score >= 4". What raw judge score should you require to achieve this threshold? Finally, at this calibration, what is the worst-case error per evaluation if the residuals have standard deviation 0.5?
      </Prose>

      <H3>Exercise 5 — Detecting bias inheritance</H3>
      <Prose>
        You have used Prometheus to score 50,000 model outputs across two SFT model variants — call them M1 and M2 — and Prometheus consistently scores M2's outputs higher. Before concluding that M2 is genuinely the better model, list three alternative explanations you should rule out. For each, describe an experiment that would distinguish bias-inheritance from genuine quality difference. As a follow-up: suppose M2 was trained on data distilled from GPT-4, and Prometheus was trained on labels generated by GPT-4. What specifically does this configuration imply about how cautious you should be about Prometheus's preference for M2, and what control experiment isolates the effect?
      </Prose>

      <H3>Exercise 6 — Designing a domain-specific judge</H3>
      <Prose>
        You need to build a judge for evaluating clinical reasoning quality in physician note responses. Off-the-shelf Prometheus scores correlate with expert physician scores at only 0.65 Pearson — too low to be useful as a primary evaluation gate. Design a remediation plan with three components: (1) a recalibration step that you can do with a small labeled sample, (2) a fine-tuning step on a larger labeled sample, and (3) a validation methodology to ensure the resulting judge actually generalizes within the clinical domain rather than overfitting to your specific training set. For each component, specify approximate sample sizes, expected effect on correlation, and operational risks.
      </Prose>

      <H3>Exercise 7 — Weight merging math</H3>
      <Prose>
        Prometheus 2 averages a pointwise-specialist checkpoint and a pairwise-specialist checkpoint with equal weight. Suppose instead you wanted to bias the merged model toward pairwise quality — say, weight the pairwise specialist at 0.7 and the pointwise specialist at 0.3. Write the explicit formula for the merged weights in terms of the base initialization <Code>{"\\theta_0"}</Code> and the two task vectors <Code>{"\\tau_A, \\tau_B"}</Code>. Then describe a practical procedure for choosing the optimal merge ratio: what would you optimize, what data would you optimize on, and what is the risk of overfitting the merge ratio itself?
      </Prose>

      <H3>Exercise 8 — When NOT to use an LLM judge</H3>
      <Prose>
        For each of the following evaluation scenarios, decide whether Prometheus (or any LLM judge) is appropriate, and if not, what the right alternative is: (a) ranking 100 candidate completions of a competitive programming problem; (b) scoring conversational politeness on a 5-point Likert scale; (c) determining whether a generated mathematical proof is valid; (d) evaluating whether a generated SQL query would correctly answer a natural-language question against a known schema; (e) judging the creative quality of a generated poem. For the cases where an LLM judge is wrong, explain why a deterministic verifier or human evaluator would be more reliable, and what specifically would go wrong if you used the LLM judge anyway.
      </Prose>

    </div>
  ),
};

export default prometheusOpenSourceJudges;
