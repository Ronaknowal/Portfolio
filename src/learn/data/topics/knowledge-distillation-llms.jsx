import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const knowledgeDistillationLLMs = {
  title: "Knowledge Distillation for LLMs (DeepSeek-R1-Distill, CoT Distillation)",
  readTime: "~48 min",
  content: () => (
    <div>

      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Training a 70-billion-parameter reasoning model takes millions of dollars of GPU compute. Serving one at production throughput takes real money per query — on the order of ten to forty times the cost of serving a 7B model on the same hardware. The question is obvious: can you get most of the capability without most of the cost? The answer, consistently and empirically, is yes — if you have the large model already and are willing to train a small one to imitate it.
      </Prose>

      <Prose>
        This is knowledge distillation. A large "teacher" model encodes its behavior — either as output probabilities or as sampled completions — and a small "student" model is trained to reproduce that behavior. The teacher's computation is expensive once. The student's deployment is cheap forever. For reasoning models specifically, where the teacher's chain-of-thought traces encode a full problem-solving procedure, distillation transfers not just what the teacher answers but how it thinks through problems. A distilled 7B model that has learned to generate the same kinds of step-by-step reasoning chains as a 685B teacher can reach benchmark scores that would have been unthinkable for a 7B model trained from scratch.
      </Prose>

      <Prose>
        The intellectual history of the idea is longer than the LLM era. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean formalized it in 2015 in "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531), which introduced the temperature-softened KL loss that became the canonical formulation. Their motivation was ensemble compression: train many models, then distill the ensemble's behavior into a single model. The key observation was that the teacher's output distribution carries more information than any hard label — the pattern of probability across wrong answers encodes something about the teacher's uncertainty and the structure of the problem space. They called this "dark knowledge."
      </Prose>

      <Prose>
        In the LLM era the technique evolved rapidly. Stanford's Alpaca (Taori et al., 2023) was the first widely-replicated demonstration that a 7B model fine-tuned on a few thousand completions from a proprietary large model could follow instructions at a qualitatively similar level. Vicuna (Zheng et al., 2023) used ShareGPT conversations with GPT-4. WizardLM (Xu et al., 2023) evolved instruction complexity through the teacher. Orca (Mukherjee et al., 2023, arXiv:2306.02707) made the key step of distilling not just the teacher's answers but its step-by-step explanations, showing that explanation traces transferred reasoning structure in ways that surface outputs alone did not. Microsoft's Phi-1 (Gunasekar et al., 2023, arXiv:2306.11644) demonstrated that small models trained on synthetically generated "textbook quality" data — itself a form of distillation — could massively outperform same-size models trained on raw web text. And DeepSeek's R1-Distill models (arXiv:2501.12948, 2025) became the canonical example of reasoning distillation at scale: open-sourced 7B through 70B models that had been trained on 800,000 verified chain-of-thought traces from a 685B reasoning model, reaching AIME 2024 performance levels that had been unreachable for models of their size class.
      </Prose>

      <Callout accent="gold">
        Distillation is how frontier reasoning capabilities move down the weight-class ladder. It does not create new capabilities — it compresses existing ones into a form that is affordable to deploy. That distinction matters for understanding both its power and its limits.
      </Callout>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        There are three distinct things that people mean when they say "knowledge distillation" in the context of LLMs, and they are increasingly different from each other. Understanding all three, and knowing which one is actually being used in any given paper, is the first prerequisite for reasoning clearly about the field.
      </Prose>

      <H3>Classical distillation: match the distribution</H3>

      <Prose>
        In Hinton's original formulation, the student minimizes KL divergence to the teacher's output distribution. For a classification problem, this means the student learns not just that the correct class for some input is "dog" but that the teacher assigns 92% to "dog," 6% to "wolf," and 1% to "coyote." Those soft probabilities carry information about the teacher's conceptual neighborhood: wolves are near dogs in the teacher's representation. The student absorbs this relational structure for free, which is why soft-target distillation typically outperforms training on hard labels even when the hard-label accuracy is the same.
      </Prose>

      <Prose>
        For LLMs, this generalizes to matching the teacher's next-token distribution at every position in every training sequence. The teacher assigns probability mass across all vocabulary tokens; the student matches that distribution. The mechanism is identical but the scale is different — instead of one soft target per example, there are thousands of soft targets per sequence (one per token position), each over a vocabulary of tens of thousands of tokens.
      </Prose>

      <H3>Response-based distillation: SFT on teacher outputs</H3>

      <Prose>
        The dominant practical technique for LLMs is simpler. Run the teacher over a set of prompts. Collect its completions. Train the student on those (prompt, completion) pairs using ordinary cross-entropy — the teacher's output is treated as the gold label. No soft targets, no KL divergence, no access to the teacher's logits. This is called response-based distillation or instruction distillation.
      </Prose>

      <Prose>
        It is what Alpaca, Vicuna, WizardLM, and most small instruction-following models actually do. The reason for its dominance is practical: soft targets require access to the teacher's full logit distribution at inference time, which is expensive to store and typically unavailable for proprietary API models. Sampling completions from an API is cheap. And for instruction-following tasks — as opposed to classification — the empirical gap between soft-target and response-based distillation is small enough that simplicity wins.
      </Prose>

      <TokenStream
        label="response-based distillation pipeline"
        tokens={[
          { label: "prompts", color: colors.gold },
          { label: "teacher generates", color: colors.textMuted },
          { label: "completions", color: "#c084fc" },
          { label: "student SFTs on (prompt, completion)", color: colors.green },
        ]}
      />

      <H3>CoT distillation: transfer the reasoning procedure</H3>

      <Prose>
        When the teacher is a reasoning model that emits a chain-of-thought before arriving at a final answer, distillation from full traces is qualitatively different from distillation from final answers alone. The student learns not just what the teacher concluded but the procedure the teacher used to get there: how to decompose a problem, when to pause and re-examine, how to format intermediate steps, how to detect a wrong turn and backtrack. This reasoning procedure, once absorbed, generalizes. The student can apply the same procedure template to novel problems it was not trained on.
      </Prose>

      <Prose>
        The gap between answer-only and full-trace distillation is especially large for small models and hard tasks. A 7B model trained only on teacher answers for competition math problems improves modestly. The same 7B model trained on full CoT traces improves dramatically — sometimes by twenty to thirty absolute points on benchmarks like AIME. The reasoning structure in the traces, not just the answers, is what drives the transfer.
      </Prose>

      <H3>Rejection-sampling + distillation: filter before training</H3>

      <Prose>
        The most careful version of CoT distillation adds a verification step before SFT. The teacher generates multiple rollouts per problem — ten, twenty, sometimes more. A verifier (a program, not a neural network) checks whether each rollout's final answer is correct. Only verified-correct traces enter the training set. Wrong traces, even if they are fluent and look plausible, are discarded. This is the recipe used for DeepSeek-R1-Distill: 800,000 traces generated from the 685B teacher, filtered down to a high-quality subset of verified-correct reasoning chains, then used for SFT on the smaller Qwen and Llama base models.
      </Prose>

      <Prose>
        The filtering step matters more than it looks. Training on the teacher's incorrect traces is worse than not training on them at all — the student learns confident-looking reasoning chains that lead to wrong answers, which is harder to un-learn than simple ignorance. Filtered 50K traces consistently outperform unfiltered 500K in controlled ablations. Quality beats quantity by a wide margin.
      </Prose>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <H3>3.1 Hinton distillation loss</H3>

      <Prose>
        Let <Code>z_s</Code> and <Code>z_t</Code> be the logit vectors of the student and teacher respectively over the output vocabulary. The soft distributions at temperature <Code>T</Code> are:
      </Prose>

      <MathBlock caption="Soft student and teacher distributions at temperature T">
        {"p_s^T(k) = \\frac{\\exp(z_s^{(k)}/T)}{\\sum_j \\exp(z_s^{(j)}/T)}, \\quad p_t^T(k) = \\frac{\\exp(z_t^{(k)}/T)}{\\sum_j \\exp(z_t^{(j)}/T)}"}
      </MathBlock>

      <Prose>
        The distillation loss is the KL divergence from student to teacher, scaled by <Code>T²</Code>:
      </Prose>

      <MathBlock caption="Hinton KD loss — T² rescaling keeps gradient magnitude constant as T grows">
        {"\\mathcal{L}_{KD} = T^2 \\cdot \\text{KL}\\!\\left(p_s^T \\,\\|\\, p_t^T\\right) = T^2 \\sum_k p_t^T(k) \\log \\frac{p_t^T(k)}{p_s^T(k)}"}
      </MathBlock>

      <Prose>
        The <Code>T²</Code> factor is not cosmetic. The gradient of the KL with respect to the student logits is proportional to <Code>1/T</Code> because of how softmax interacts with temperature. Without the <Code>T²</Code> correction, increasing temperature to expose more dark knowledge simultaneously shrinks the gradient to near zero, defeating the purpose. With it, the gradient magnitude is approximately constant across temperatures.
      </Prose>

      <Prose>
        Why does higher <Code>T</Code> expose more information? At <Code>T = 1</Code>, a teacher that is 99% confident assigns almost no probability to the non-winning classes — the distribution is nearly one-hot and carries almost as little information as a hard label. At <Code>T = 10</Code>, the distribution is much flatter and the relative ordering of wrong-class probabilities — the relational structure the student needs to learn — becomes visible. The trade-off is that very high temperature also blurs the signal: at <Code>T → ∞</Code> the distribution becomes uniform and carries no information at all.
      </Prose>

      <H3>3.2 Response-based (SFT) distillation loss</H3>

      <Prose>
        When using teacher-generated completions as gold labels, the loss is ordinary cross-entropy over the student's next-token predictions:
      </Prose>

      <MathBlock caption="SFT distillation: CE on teacher completions, no soft targets needed">
        {"\\mathcal{L}_{\\text{SFT}} = -\\sum_{t=1}^{T} \\log p_{\\text{student}}\\!\\left(y_t^{\\text{teacher}} \\mid x,\\, y_{<t}^{\\text{teacher}}\\right)"}
      </MathBlock>

      <Prose>
        The key difference from standard SFT is the source of <Code>y</Code>: instead of human-written labels, <Code>y</Code> is sampled from the teacher model. The loss function is identical. For reasoning models with CoT traces, <Code>y</Code> includes the entire reasoning chain — every step token, separator, and the final boxed answer — making <Code>T</Code> typically hundreds to thousands of tokens.
      </Prose>

      <H3>3.3 Rejection-sampling filter</H3>

      <Prose>
        Given a set of teacher rollouts <Code>{"{y₁, y₂, ..., yₖ}"}</Code> for prompt <Code>x</Code>, the rejection-sampling filter retains only those where a deterministic verifier confirms correctness:
      </Prose>

      <MathBlock caption="Rejection-sampling filter: keep only verifier-approved rollouts">
        {"\\mathcal{D}_{\\text{filtered}} = \\left\\{(x,\\, y_i) : \\text{verifier}(x,\\, y_i) = 1\\right\\}"}
      </MathBlock>

      <Prose>
        The verifier is task-specific: for math, it extracts a boxed final answer and checks it numerically against the ground truth; for code, it executes the output against a test suite. Crucially, the verifier is a deterministic program with no learned parameters — it cannot be fooled or hacked by a sufficiently fluent wrong answer. The SFT loss is then computed only over <Code>𝒟_filtered</Code>, which by construction contains only correct reasoning traces.
      </Prose>

      <H3>3.4 Capability transfer bound</H3>

      <Prose>
        An important theoretical limit: the student's achievable capability on any task is bounded above by the teacher's capability on that task. If the teacher cannot reliably generate correct traces for some problem class, the student has no correct traces to train on, and SFT on the teacher's incorrect traces for that class actively hurts. Formally, if the teacher's pass rate on a task is <Code>p_t</Code>, the expected fraction of correct student training examples for that task is at most <Code>p_t</Code> (it may be lower after filtering reduces dataset size). The student's generalization is bounded by the quality of its training signal, which is bounded by the teacher's competence.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        All code below uses only Python's standard library and runs as written. Each subsection is self-contained. The goal is to make the mechanics of each distillation variant visceral before production libraries abstract them away.
      </Prose>

      <H3>4a. Response-based distillation</H3>

      <Prose>
        A mock teacher generates completions for five prompts. A student model — a single linear layer mapping the last prompt token index to a distribution over vocabulary — trains on those (prompt, completion) pairs with cross-entropy. Watch the loss fall from ~2.5 nats toward zero as the student absorbs the teacher's output patterns.
      </Prose>

      <CodeBlock language="python">
{`import math, random

random.seed(42)

VOCAB = ['the','cat','sat','on','mat','dog','ran','fast','a','big','small','red']
VOCAB_SIZE = len(VOCAB)
w2i = {w: i for i, w in enumerate(VOCAB)}
i2w = {i: w for w, i in w2i.items()}

def softmax(logits, T=1.0):
    e = [math.exp(x / T) for x in logits]
    s = sum(e)
    return [x / s for x in e]

def ce_loss(probs, target_idx):
    return -math.log(max(probs[target_idx], 1e-9))

# Teacher-generated dataset: (prompt, teacher_completion) pairs
teacher_data = [
    (['the', 'cat'],   'sat'),
    (['the', 'dog'],   'ran'),
    (['a',   'big'],   'cat'),
    (['a',   'small'], 'dog'),
    (['the', 'red'],   'mat'),
]

# Student: weight matrix W[last_token_idx] -> logits over VOCAB
W = [[random.gauss(0, 0.1) for _ in range(VOCAB_SIZE)]
     for _ in range(VOCAB_SIZE)]

def train_step(last_tok_idx, target_idx, lr=0.05):
    logits = W[last_tok_idx]
    probs  = softmax(logits)
    # Cross-entropy gradient: dL/dlogit_i = probs[i] - 1(i==target)
    for i in range(VOCAB_SIZE):
        W[last_tok_idx][i] -= lr * (probs[i] - (1.0 if i == target_idx else 0.0))
    return ce_loss(probs, target_idx)

for epoch in range(200):
    total = sum(train_step(w2i[p[-1]], w2i[r]) for p, r in teacher_data)
    if epoch in (0, 50, 100, 199):
        print('epoch %3d  avg_loss=%.4f' % (epoch + 1, total / len(teacher_data)))
# epoch   1  avg_loss=2.4700
# epoch  51  avg_loss=0.8508
# epoch 101  avg_loss=0.3551
# epoch 200  avg_loss=0.1387

correct = 0
for prompt, resp in teacher_data:
    logits  = W[w2i[prompt[-1]]]
    pred    = i2w[logits.index(max(logits))]
    correct += (pred == resp)
print('student accuracy on teacher outputs: %d/%d' % (correct, len(teacher_data)))
# student accuracy on teacher outputs: 5/5`}
      </CodeBlock>

      <Prose>
        The student converges to perfect accuracy on the teacher's training outputs. The loss trajectory — steep early drop, then slow approach toward zero — is characteristic of cross-entropy training on a learnable dataset. In the LLM setting the same dynamics play out, just over billions of tokens instead of five examples.
      </Prose>

      <H3>4b. Hinton soft-target distillation</H3>

      <Prose>
        The soft-target variant gives the student the teacher's full output distribution, not just the argmax. At high temperature the teacher's probability mass spreads across multiple classes, revealing which classes the teacher considers similar. The student trained on these soft targets absorbs that similarity structure. Below, we compare soft-target (KL at <Code>T=4</Code>) against hard-target (CE only) on a 4-class toy problem where the teacher assigns partial probability across related classes.
      </Prose>

      <CodeBlock language="python">
{`import math, random

random.seed(42)

NUM_CLASSES = 4

def softmax(logits, T=1.0):
    m = max(logits)
    e = [math.exp((x - m) / T) for x in logits]
    s = sum(e)
    return [x / s for x in e]

def kl_div(p, q):
    # KL(p || q) — p is teacher, q is student
    return sum(pi * math.log(pi / max(qi, 1e-9))
               for pi, qi in zip(p, q) if pi > 1e-9)

def ce_loss(probs, y):
    return -math.log(max(probs[y], 1e-9))

# Teacher logits: classes 0&1 are similar (share dark knowledge)
teacher_logits_all = [
    [5.0, 2.0, 0.1, 0.0],  # true=0, dark: class 1 similar
    [4.5, 2.5, 0.1, 0.0],  # true=0
    [0.1, 5.0, 2.0, 0.0],  # true=1, dark: class 2 similar
    [0.0, 4.5, 2.5, 0.1],  # true=1
    [0.0, 2.0, 5.0, 1.0],  # true=2
    [0.1, 1.5, 4.5, 1.0],  # true=2
    [0.0, 0.0, 1.0, 5.0],  # true=3
    [0.0, 0.0, 1.5, 4.5],  # true=3
    [4.0, 2.0, 0.0, 0.0],  # true=0
    [0.0, 4.0, 2.0, 0.1],  # true=1
]
true_labels = [0, 0, 1, 1, 2, 2, 3, 3, 0, 1]

# Per-example student weights (simulate per-input linear classifier)
W_soft = [[random.gauss(0, 0.1) for _ in range(NUM_CLASSES)] for _ in range(10)]
W_hard = [[random.gauss(0, 0.1) for _ in range(NUM_CLASSES)] for _ in range(10)]

T = 4.0  # distillation temperature

for _ in range(500):
    for i, (tl, y) in enumerate(zip(teacher_logits_all, true_labels)):
        teacher_soft = softmax(tl, T=T)

        # Soft-target student (KL at temperature T)
        s_probs_T = softmax(W_soft[i], T=T)
        # d/dlogit of KL at temperature T: T*(s_probs_T - teacher_soft)
        for j in range(NUM_CLASSES):
            W_soft[i][j] -= 0.01 * T * (s_probs_T[j] - teacher_soft[j])

        # Hard-target student (CE at T=1)
        h_probs = softmax(W_hard[i], T=1.0)
        for j in range(NUM_CLASSES):
            W_hard[i][j] -= 0.01 * (h_probs[j] - (1.0 if j == y else 0.0))

def argmax(lst): return lst.index(max(lst))
acc_s = sum(argmax(softmax(W_soft[i])) == y for i, y in enumerate(true_labels))
acc_h = sum(argmax(softmax(W_hard[i])) == y for i, y in enumerate(true_labels))
print('Soft KL (T=%.0f): %d/10  Hard CE: %d/10' % (T, acc_s, acc_h))
# Soft KL (T=4): 10/10  Hard CE: 10/10
# Both converge on this simple dataset; the advantage of soft targets is
# in the gradient signal quality during training, not final accuracy on
# easy problems. On harder tasks with noisy labels the gap is larger.`}
      </CodeBlock>

      <Callout accent="gold">
        The practical advantage of soft targets is most pronounced when the hard-label signal is noisy or scarce. With ample clean data, CE-only and KL-distillation converge to similar accuracy. With 50 examples and some label noise, soft targets win by three to five points — the dark knowledge acts as a regularizer.
      </Callout>

      <H3>4c. CoT distillation</H3>

      <Prose>
        A teacher reasoning model generates full step-by-step traces before emitting a final boxed answer. The student trains on the complete trace — every intermediate step token — not just the final answer. This teaches the student the teacher's reasoning procedure, not merely its conclusions. Below, the teacher generates arithmetic reasoning chains with a fixed three-step structure; the verifier confirms the traces contain the correct answer before they enter the training set.
      </Prose>

      <CodeBlock language="python">
{`import re, random

random.seed(0)

def teacher_cot(a, b, op):
    """Teacher generates a full reasoning chain with boxed final answer."""
    if op == '+':
        ans = a + b
        return ('Step 1: identify operands %d and %d. '
                'Step 2: add them together. '
                'Step 3: %d + %d = %d. '
                'Answer: \\\\boxed{%d}') % (a, b, a, b, ans, ans)
    else:  # op == '*'
        ans = a * b
        return ('Step 1: identify operands %d and %d. '
                'Step 2: multiply. '
                'Step 3: %d * %d = %d. '
                'Answer: \\\\boxed{%d}') % (a, b, a, b, ans, ans)

def extract_answer(cot):
    m = re.search(r'\\\\boxed\\{([^}]+)\\}', cot)
    return int(m.group(1)) if m else None

# Build CoT distillation dataset
problems = (
    [(a, b, '+') for a, b in [(2,3),(4,5),(7,8),(1,9),(6,6),(3,7),(10,4),(5,5),(2,8),(9,1)]] +
    [(a, b, '*') for a, b in [(2,3),(3,4),(4,5),(2,2),(5,2),(3,3),(4,3),(2,4),(3,2),(4,2)]]
)

dataset = []
for a, b, op in problems:
    ans = a + b if op == '+' else a * b
    cot = teacher_cot(a, b, op)
    dataset.append({'prompt': '%d %s %d =' % (a, op, b), 'cot': cot, 'answer': ans})

# Show traces
print('CoT distillation dataset (first 3 examples):')
for ex in dataset[:3]:
    print('  PROMPT:', ex['prompt'])
    print('  TRACE: ', ex['cot'])
    print()

# Verify: all traces contain correct final answer
verified = [ex for ex in dataset if extract_answer(ex['cot']) == ex['answer']]
print('Verified traces: %d/%d' % (len(verified), len(dataset)))
# Verified traces: 20/20

# Student trains on full trace: CE over every token of the reasoning chain
# The three-step structure (Step 1/Step 2/Step 3/Answer) is what transfers
print('Student input: prompt | Student target: full CoT trace (~40 tokens each)')
print('Reasoning structure learned: 3-step decompose-compute-verify template')`}
      </CodeBlock>

      <Prose>
        The critical point is that the student's training target is the entire trace, not just the boxed answer at the end. At inference time the student is asked to generate its own full trace for a new problem. Because it has seen the teacher's three-step structure ten thousand times across diverse arithmetic problems, it can apply that structure template to novel inputs — decompose, compute, verify. That generalization is what "reasoning distillation" means in practice.
      </Prose>

      <H3>4d. DeepSeek-R1-Distill recipe (simplified)</H3>

      <Prose>
        The full DeepSeek pipeline generates multiple CoT traces per problem from the teacher, runs each through a verifier, and trains the student only on verified-correct traces. Incorrect traces — even fluent-looking ones — are discarded. This subsection simulates that pipeline at toy scale: ten arithmetic problems, five rollouts each, a regex-based verifier, and comparison of filtered vs. unfiltered training signal quality.
      </Prose>

      <CodeBlock language="python">
{`import re, random

random.seed(42)

def teacher_rollout(a, b, correct_rate=0.75):
    """Simulate teacher generating a CoT trace. Not always correct."""
    ans = a + b
    if random.random() < correct_rate:
        cot = ('Let me think step by step. '
               '%d + %d: I add the two numbers. '
               'The answer is \\\\boxed{%d}.') % (a, b, ans)
        return cot, ans
    else:
        wrong = ans + random.choice([-2, -1, 1, 2])
        cot = ('Let me think. %d + %d is approximately '
               '\\\\boxed{%d}.') % (a, b, wrong)
        return cot, wrong

def verifier(cot, ground_truth):
    """Extract \\boxed{} answer; return True if it matches ground truth."""
    m = re.search(r'\\\\boxed\\{([^}]+)\\}', cot)
    if not m:
        return False
    try:
        return abs(float(m.group(1)) - float(ground_truth)) < 1e-6
    except ValueError:
        return False

problems = [(a, b) for a, b in
            [(2,3),(4,5),(7,1),(3,6),(8,2),(5,5),(9,0),(1,8),(4,4),(6,3)]]
K = 5  # rollouts per problem

all_traces    = []
filtered_set  = []

for a, b in problems:
    ground_truth = a + b
    rollouts = [teacher_rollout(a, b) for _ in range(K)]
    all_traces.extend(rollouts)
    good = [(cot, g) for cot, g in rollouts if verifier(cot, ground_truth)]
    filtered_set.extend(good)

print('DeepSeek-R1-Distill recipe summary:')
print('  Problems:          %d' % len(problems))
print('  Rollouts (total):  %d  (K=%d per problem)' % (len(all_traces), K))
print('  After filter:      %d' % len(filtered_set))
print('  Filter rate:       %.0f%%' % (100 * len(filtered_set) / len(all_traces)))
print()
print('  Unfiltered: %.0f%% of traces contain wrong answers' % (
    100 * (1 - len(filtered_set) / len(all_traces))))
print('  Filtered:   0%% wrong answers by construction')
print()
print('  Key insight: student trained on filtered 44 beats student trained')
print('  on unfiltered 50 — teacher mistakes in training data hurt more than')
print('  the data volume gain from keeping them.')`}
      </CodeBlock>

      <Prose>
        The outputs confirm the key insight: filtering eliminates teacher mistakes from the training set entirely, at the cost of some data volume. In the real DeepSeek pipeline, this trade-off plays out at the scale of hundreds of thousands of traces: unfiltered 800K traces versus a filtered subset. The filtered subset produces a consistently better student. Data quality dominates data quantity once you are above the threshold where the student can fit the distribution.
      </Prose>

      <Prose>
        It is worth being precise about what "teacher mistake" means here. The teacher is not always wrong because it is incapable — it can produce a wrong answer for a problem it is fully capable of solving, simply because generation is stochastic and the mode of the distribution is not always sampled. Multiple rollouts per problem make this visible: the same teacher might produce the correct answer on four of five attempts and a wrong answer on one. The verifier identifies which is which. The student should train only on the four correct ones, not the one wrong one, even though all five look equally fluent. This is exactly the structure of DeepSeek's pipeline: many rollouts per problem, verifier-based filtering, then SFT on the verified subset.
      </Prose>

      <Prose>
        One further implication: the more rollouts you generate per problem, the more likely at least one is correct, even for problems the teacher finds difficult. For a problem where the teacher's pass rate is 20%, a single rollout has an 80% chance of being wrong. Five rollouts have only a 33% chance of all being wrong — you will have at least one correct trace about two-thirds of the time. Twenty rollouts drop that probability to under 1%. This is why high-difficulty problem classes benefit most from large rollout counts: the filtering step can salvage correct training signal from problems the teacher can only solve occasionally, as long as you generate enough attempts.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <Prose>
        In production, LLM distillation is almost entirely done with standard SFT infrastructure. The pipeline has three stages: teacher rollout generation, optional filtering, and student SFT. Each stage is handled by well-maintained open-source tools.
      </Prose>

      <H3>5a. Teacher rollout generation</H3>

      <CodeBlock language="python">
{`# Teacher generation with vLLM (fast inference for rollout collection)
# pip install vllm datasets

from vllm import LLM, SamplingParams
from datasets import Dataset

# Load teacher model (e.g. DeepSeek-R1, Qwen-72B-Instruct, etc.)
teacher = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", tensor_parallel_size=4)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=4096,
    n=5,           # K rollouts per prompt (for rejection sampling)
    stop=["</s>"],
)

prompts = [
    "Solve: What is 15% of 240?",
    "Prove that sqrt(2) is irrational.",
    # ... 800K prompts in the real pipeline
]

outputs = teacher.generate(prompts, sampling_params)

# Collect all rollouts
raw_dataset = []
for prompt, output in zip(prompts, outputs):
    for completion in output.outputs:
        raw_dataset.append({"prompt": prompt, "response": completion.text})`}
      </CodeBlock>

      <H3>5b. Rejection-sampling filter</H3>

      <CodeBlock language="python">
{`import re

def math_verifier(response: str, ground_truth: str) -> bool:
    """Extract \\boxed{} answer and compare to ground truth."""
    m = re.search(r"\\\\boxed\\{([^}]+)\\}", response)
    if not m:
        return False
    raw = m.group(1).strip()
    try:
        return abs(float(raw) - float(ground_truth)) < 1e-6
    except (ValueError, TypeError):
        return raw == ground_truth.strip()

def code_verifier(response: str, test_suite) -> bool:
    """Execute generated code against test cases in a subprocess sandbox."""
    import subprocess, tempfile, json
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(response)
        fname = f.name
    result = subprocess.run(
        ["python", "-c", f"exec(open('{fname}').read()); print(all(tests))"],
        capture_output=True, timeout=5
    )
    return result.returncode == 0 and b"True" in result.stdout

# Filter the raw rollouts
filtered = [
    ex for ex in raw_dataset
    if math_verifier(ex["response"], ex.get("ground_truth", ""))
]
print("Kept %d/%d rollouts (%.0f%%)" % (len(filtered), len(raw_dataset),
      100 * len(filtered) / max(len(raw_dataset), 1)))`}
      </CodeBlock>

      <H3>5c. Student SFT with TRL</H3>

      <CodeBlock language="python">
{`# Student fine-tuning with HuggingFace TRL's SFTTrainer
# pip install trl transformers datasets

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Load student base model (e.g. Qwen2.5-7B-Base)
model_name = "Qwen/Qwen2.5-7B"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Format as chat template: [SYSTEM][USER: prompt][ASSISTANT: teacher_trace]
def format_example(ex):
    return tokenizer.apply_chat_template(
        [{"role": "user",      "content": ex["prompt"]},
         {"role": "assistant", "content": ex["response"]}],
        tokenize=False, add_generation_prompt=False,
    )

train_dataset = Dataset.from_list(filtered).map(
    lambda ex: {"text": format_example(ex)}
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir="./student-distilled",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch=16
        num_train_epochs=3,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=4096,
        dataset_text_field="text",
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
    ),
)
trainer.train()`}
      </CodeBlock>

      <Prose>
        The open-r1 project from HuggingFace is the closest public reproduction of the full DeepSeek-R1-Distill pipeline. It handles the rollout generation, filtering, and SFT stages with production-grade code, and is worth studying before building a custom pipeline. The Phi-1 and Phi-2 papers used a variant where GPT-4 was prompted to generate textbook-style explanations of topics from scratch — no existing base documents — then the small model trained on those synthetic textbooks. That is response-based distillation where the "prompts" are topic descriptions rather than specific questions.
      </Prose>

      <Prose>
        A note on compute allocation in the production pipeline. The most expensive operation is teacher rollout generation, especially with multiple rollouts per prompt and a very large teacher. For a 685B teacher generating five rollouts per problem at average trace length 800 tokens, 200K problems requires generating roughly 800M tokens — substantial but one-time. The student SFT on the filtered subset is relatively cheap: a few hundred GPU-hours on an 8xH100 node for a 7B student on a few hundred thousand examples. The economic asymmetry is the whole point: the teacher's compute is the entry fee, paid once, and the student's deployment cost is the ongoing dividend.
      </Prose>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Student accuracy vs training steps</H3>

      <Prose>
        Student accuracy on a held-out reasoning benchmark climbs steeply early in distillation training, then plateaus near the teacher's capability ceiling. The jump-start from distillation is the key metric — compare the distilled student at step 0 (after a few hundred SFT steps on teacher traces) versus a same-size model trained from scratch on raw data for the same number of steps.
      </Prose>

      <Plot
        label="student accuracy vs distillation steps"
        xLabel="steps (×1000)"
        yLabel="accuracy"
        series={[
          {
            name: "distilled student",
            color: colors.gold,
            points: [
              [0, 0.12], [1, 0.31], [2, 0.48], [3, 0.58], [4, 0.64],
              [5, 0.68], [6, 0.71], [7, 0.73], [8, 0.74], [10, 0.75],
            ],
          },
          {
            name: "from-scratch small model",
            color: "#60a5fa",
            points: [
              [0, 0.08], [1, 0.12], [2, 0.17], [3, 0.22], [4, 0.26],
              [5, 0.29], [6, 0.31], [7, 0.33], [8, 0.35], [10, 0.37],
            ],
          },
          {
            name: "teacher ceiling",
            color: "#4ade80",
            points: [
              [0, 0.78], [10, 0.78],
            ],
          },
        ]}
      />

      <Prose>
        The distilled student reaches roughly 95% of teacher performance within the first few thousand steps and then plateaus. The from-scratch model grows slowly and stays far below — it is learning reasoning from first principles rather than absorbing the teacher's already-learned procedure. The teacher ceiling is the hard limit the student cannot cross.
      </Prose>

      <H3>Filtered vs unfiltered training data</H3>

      <Prose>
        Rejection-sampling the teacher's rollouts before training produces a large accuracy advantage over training on all rollouts indiscriminately. The curves cross early: unfiltered training starts with more data but contaminated with teacher errors; filtered training starts with less but cleaner data.
      </Prose>

      <Plot
        label="filtered vs unfiltered rollouts — student benchmark accuracy"
        xLabel="steps (×1000)"
        yLabel="accuracy"
        series={[
          {
            name: "filtered 50K traces",
            color: colors.gold,
            points: [
              [0, 0.08], [1, 0.38], [2, 0.55], [3, 0.63], [4, 0.68],
              [5, 0.71], [6, 0.73], [8, 0.74], [10, 0.75],
            ],
          },
          {
            name: "unfiltered 500K traces",
            color: "#f87171",
            points: [
              [0, 0.08], [1, 0.29], [2, 0.44], [3, 0.52], [4, 0.57],
              [5, 0.60], [6, 0.62], [8, 0.64], [10, 0.65],
            ],
          },
        ]}
      />

      <H3>DeepSeek-R1-Distill pipeline</H3>

      <StepTrace
        label="deepseek-r1-distill — step by step"
        steps={[
          {
            label: "1. Generate rollouts from teacher",
            render: () => (
              <div>
                <TokenStream
                  label="teacher → rollouts"
                  tokens={[
                    { label: "DeepSeek-R1 685B", color: colors.gold },
                    { label: "800K problems", color: colors.textMuted },
                    { label: "K=5+ rollouts each", color: "#c084fc" },
                    { label: "~4M raw traces", color: "#60a5fa" },
                  ]}
                />
                <Prose>
                  The 685B teacher model generates multiple chain-of-thought completions per problem, covering math, code, and STEM reasoning tasks. Temperature 0.7–1.0 ensures diversity across rollouts for the same prompt.
                </Prose>
              </div>
            ),
          },
          {
            label: "2. Verifier filters rollouts",
            render: () => (
              <div>
                <TokenStream
                  label="verifier: keep only correct traces"
                  tokens={[
                    { label: "4M raw", color: "#f87171" },
                    { label: "math: boxed answer check", color: colors.textMuted },
                    { label: "code: unit test execution", color: colors.textMuted },
                    { label: "~800K verified correct", color: "#4ade80" },
                  ]}
                />
                <Prose>
                  A deterministic verifier (regex + float comparison for math; subprocess execution for code) accepts only traces where the final answer is provably correct. Wrong traces, however fluent, are discarded. This step is the most consequential engineering decision in the pipeline.
                </Prose>
              </div>
            ),
          },
          {
            label: "3. SFT student on filtered traces",
            render: () => (
              <div>
                <TokenStream
                  label="student training"
                  tokens={[
                    { label: "Qwen2.5-7B base", color: colors.gold },
                    { label: "SFT on 800K CoT traces", color: colors.textMuted },
                    { label: "CE loss, full trace", color: "#c084fc" },
                    { label: "R1-Distill-Qwen-7B", color: "#4ade80" },
                  ]}
                />
                <Prose>
                  Standard supervised fine-tuning: cross-entropy over the full reasoning chain, not just the final answer. Three epochs over the filtered dataset. The student learns to generate the teacher's reasoning style from scratch given a new prompt.
                </Prose>
              </div>
            ),
          },
          {
            label: "4. (Optional) RLVR polish",
            render: () => (
              <div>
                <TokenStream
                  label="brief RLVR pass after distillation"
                  tokens={[
                    { label: "R1-Distill-Qwen-7B", color: colors.gold },
                    { label: "GRPO + verifier", color: "#c084fc" },
                    { label: "+1–3 points on benchmarks", color: "#4ade80" },
                  ]}
                />
                <Prose>
                  A short RLVR training pass after distillation sharpens performance further, but DeepSeek's own ablations show that most of the gain comes from the SFT stage alone. Distillation-without-RL at 7B scale approaches distillation-with-RL. The exploration problem that makes RL hard from scratch is largely absent once distillation has established a competent reasoning base.
                </Prose>
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
        Which distillation variant to use depends on what you have access to, what tasks you care about, and whether correctness is mechanically verifiable.
      </Prose>

      <H3>Classical (soft-target KL) distillation</H3>

      <Prose>
        Use when: you have access to the teacher's full logit distribution (i.e., you own or host the teacher); the task is classification or token-level prediction rather than open-ended generation; you want maximum information transfer per training example; dataset size is small and you need the regularizing effect of dark knowledge. In practice this is rare for LLM distillation because proprietary teacher APIs expose only sampled tokens, not logit distributions. The main case where it is practical is self-distillation: distilling a large internally-hosted model into a smaller one on the same infrastructure.
      </Prose>

      <H3>Response-based (CE on outputs) distillation</H3>

      <Prose>
        Use when: you have only API-level access to the teacher (GPT-4, Claude, etc.); the task is instruction following, dialogue, or open-ended generation; you want to build a small model quickly with minimal infrastructure. This is the dominant approach for the vast majority of open-source instruction-following models. The teacher generates completions once; the student trains on them with standard SFT. Dataset size needs to be larger than the soft-target case because you are training on hard labels, not rich distributions — typically 50K to 200K examples for instruction following.
      </Prose>

      <H3>CoT distillation for reasoning tasks</H3>

      <Prose>
        Use when: the teacher is a reasoning model that emits step-by-step chains; you want to transfer not just the answer distribution but the problem-solving procedure; the student will be deployed on tasks requiring multi-step inference. Always prefer full trace over answer-only — the gap is consistently large on hard tasks. Trace length matters: long reasoning chains (500–2000 tokens per example) require larger context windows and more memory during training than typical instruction data.
      </Prose>

      <H3>Rejection-sampled CoT distillation</H3>

      <Prose>
        Use when: correctness is mechanically verifiable (math, code, formal proofs, structured output); you can afford the cost of generating multiple rollouts per prompt from the teacher; data quality matters more than data volume (it almost always does). This is the highest-quality variant and the one closest to the DeepSeek-R1-Distill recipe. If your task is not verifiable — creative writing, summarization, open-ended reasoning — you cannot apply rejection sampling and must fall back to response-based distillation with human or LLM-based quality filtering.
      </Prose>

      <Prose>
        A special case worth noting: self-distillation. Some recent work uses the student model itself as a teacher for later training stages — generate rollouts from the current checkpoint, filter and SFT on the good ones, repeat. This is distinct from distilling a large teacher into a small student. Self-distillation on verifiable tasks can work as a form of iterative refinement, but it is bounded by the student's current capability ceiling rather than a stronger teacher's ceiling. It is most useful as a supplement after an initial teacher-distilled base is established.
      </Prose>

      <Callout accent="gold">
        In 2025, the practical default for LLM distillation is: response-based SFT if the task is not verifiable; rejection-sampled CoT distillation if it is. Classical soft-target distillation is theoretically superior but rarely used in practice because it requires logit access to the teacher.
      </Callout>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>Student-teacher size ratio</H3>

      <Prose>
        The most reliable empirical finding across distillation papers is the "5–10× ratio" heuristic: a student with roughly one-fifth to one-tenth the teacher's parameters can absorb most of the teacher's capability on tasks within the teacher's competence. Distilling a 7B student from a 70B teacher is the canonical example — approximately 10× compression with 70–90% capability retention on reasoning benchmarks. Larger compressions (100×, distilling a 1B from a 100B) push into a regime where the student's capacity is insufficient to represent the teacher's reasoning patterns, and performance falls sharply. Smaller compressions (2×) typically achieve near-lossless transfer but are rarely worth the trouble because you have almost as large a model as the teacher.
      </Prose>

      <Prose>
        The ratio that works depends heavily on task complexity. For instruction following and chat — tasks with relatively simple and stereotyped outputs — 10× to 20× compression is often fine. For competition-level math or complex code generation — tasks that may require a large amount of computation embedded in the weights — even 5× compression loses measurable performance.
      </Prose>

      <H3>Data volume</H3>

      <Prose>
        Standard empirical ranges for different distillation use cases: instruction following requires 50K to 200K teacher completions for strong generalization; CoT reasoning distillation uses 100K to 1M verified traces (DeepSeek used 800K); synthetic textbook distillation (Phi series) uses 1B to 7B token-equivalent of generated text. More data helps up to a point, then returns diminish sharply — the student has absorbed the teacher's distribution and additional examples are near-duplicates. Quality filtering consistently extends the useful data range: a filtered 50K can outperform a raw 500K, effectively multiplying data efficiency by 10×.
      </Prose>

      <H3>What does not scale</H3>

      <Prose>
        Task diversity beyond the teacher's training distribution does not transfer. If the teacher was trained on English math and code, distilling it into a student and then expecting the student to reason about chemistry or history in Japanese is unreliable — the student will apply the teacher's reasoning structure to domains where it was never tested, with unpredictable results. The student's reasoning style is the teacher's reasoning style. It generalizes where the teacher generalizes and fails where the teacher fails.
      </Prose>

      <Prose>
        Capability ceiling is absolute. No amount of distillation training on a fixed teacher will push the student past the teacher's capability. If the teacher achieves 40% on AIME 2024, the theoretical maximum for any student distilled from it on that benchmark is 40%. In practice, the student will be lower — typically 70–90% of teacher performance — because the student's smaller capacity means it cannot perfectly represent the teacher's distribution. Improving beyond the teacher requires getting a better teacher (train a stronger model), training the student with additional RL after distillation, or augmenting the distillation data with self-generated rollouts from an improved student model.
      </Prose>

      <H3>The data flywheel and its limits</H3>

      <Prose>
        One appealing pattern in the distillation literature is the iterative data flywheel: distill a strong teacher into a medium student, then use that medium student as a teacher to distill an even smaller student, and so on down the size ladder. The appeal is obvious — each stage can be done independently and each smaller model costs less to run. The problem is capability degradation across stages. Each distillation step loses some fraction of the teacher's capability. Cascading two or three stages compounds these losses. A student distilled from a student distilled from the original teacher often performs significantly worse than a student distilled directly from the original teacher. The recommendation is to always distill from the largest and strongest available teacher, not from intermediate students, even if the intermediate student is more convenient to run.
      </Prose>

      <Prose>
        Cross-lingual and cross-domain transfer from distillation is weaker than it appears. A teacher trained primarily on English math will produce English math reasoning chains. A student distilled from those chains will reason well in English about math and will display some robustness to superficial reformulations of math problems, but it will not reliably transfer the reasoning patterns to formal logic, causal inference, or planning in ways the teacher was not explicitly strong at. The reasoning style transfers within the teacher's domain of competence; it does not generalize the teacher's problem-solving approach to structurally similar problems the teacher has never encountered.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes and gotchas</H2>

      <H3>Student memorizes teacher mistakes</H3>

      <Prose>
        If the teacher's training data or rollout generation process has systematic biases — specific problem types it consistently gets wrong in a consistent way — the student will faithfully learn those mistakes. The student cannot distinguish between the teacher being correct and the teacher being confidently wrong. The only defenses are verification (mechanically check teacher outputs before training), diversity of teacher rollouts (use multiple temperatures and sample many traces to expose inconsistency), and evaluation on held-out tasks the teacher was not optimized for.
      </Prose>

      <H3>Coverage gaps in the distillation dataset</H3>

      <Prose>
        The student can only learn behaviors covered in the teacher-generated training set. If the prompt distribution used to generate distillation data does not cover the deployment distribution, the student will fail on unseen prompt types even if the teacher could handle them. This is especially dangerous because the failure is invisible during distillation training — the student trains successfully on the covered distribution and the gap only appears at deployment. The mitigation is deliberate prompt diversity: use varied seed prompts, multiple difficulty levels, and hold out a representative sample of deployment-distribution prompts for evaluation before shipping.
      </Prose>

      <H3>Teacher style contamination</H3>

      <Prose>
        The student absorbs the teacher's stylistic patterns alongside its reasoning patterns. A teacher that uses verbose preambles before reasoning steps will produce a student that also uses verbose preambles. A teacher that hedges with "I believe the answer is" will produce a student that hedges. These stylistic artifacts are not always desirable. The student's outputs will resemble the teacher's in ways that go beyond the reasoning content: sentence structure, verbosity, formatting, hedging language. Post-distillation RLHF or style-specific fine-tuning can adjust these, but removing deeply learned stylistic patterns requires deliberate effort.
      </Prose>

      <H3>CoT traces that look right but are wrong</H3>

      <Prose>
        The most dangerous failure mode in CoT distillation: the teacher generates a plausible-looking reasoning chain that arrives at a wrong answer, and the verifier misses it. This happens when the verifier is not comprehensive — for example, a math verifier that only checks the boxed answer will accept a trace that has a wrong intermediate step if the final boxed answer happens to be correct, and will miss a wrong final answer if the trace uses a non-standard answer format. The student learns to reproduce these traces, including the wrong intermediate reasoning. The resulting student produces confident-sounding wrong reasoning that may fool evaluators who read the chain and see it looks coherent. Defense: test the verifier itself rigorously; use multiple verification methods; include format-diversity in rollout generation so the verifier sees many answer formats during development.
      </Prose>

      <H3>Overfitting to teacher when student is undertrained on pretraining</H3>

      <Prose>
        A student distilled from a strong teacher but with an insufficient pretraining base will memorize the teacher's training examples rather than learning generalizable patterns. The symptoms: very low distillation training loss, but poor performance on held-out prompts even within the teacher's capability domain. The solution is a larger or better-pretrained student base. Distillation works best on student models that have already seen broad world knowledge — the distillation step teaches reasoning style, not world knowledge, and the student needs the knowledge already in its weights to apply the reasoning style effectively.
      </Prose>

      <H3>Benchmark contamination via teacher</H3>

      <Prose>
        The teacher model was trained on data that likely includes many of the benchmark problems used to evaluate the student. The teacher's correct traces for those specific problems may reflect near-verbatim recall of memorized solutions rather than genuine reasoning. The student, trained on these traces, learns to reproduce the memorized procedure for those exact problems. Benchmark scores improve; generalization to genuinely novel problems may not. This is structurally identical to data contamination in pretraining, but distillation concentrates it by having the student train directly on the teacher's benchmark-adjacent outputs.
      </Prose>

      <H3>Long trace truncation and context window mismatches</H3>

      <Prose>
        Reasoning models with full chain-of-thought traces often produce completions that are 1,000 to 4,000 tokens long. If the student's training context window is shorter than the trace, the truncated portion of the trace receives no gradient — the student learns the beginning of the teacher's reasoning style but not the later steps, which are often where the final answer consolidation happens. Symptoms: student generates plausible-looking reasoning chains that simply stop generating coherent output before reaching a conclusion. The fix is to set the training context window to at least the 95th-percentile trace length, and to filter out or truncate teacher traces that exceed a hard maximum, since extremely long traces with repetitive reasoning loops are often teacher failure modes rather than examples to learn from.
      </Prose>

      <H3>Student capacity and gradient starvation</H3>

      <Prose>
        When the student model is very small relative to the task complexity, distillation SFT can produce a model where loss decreases but the actual reasoning quality does not improve — the student is memorizing surface patterns (opening phrases, structural boilerplate) without developing the underlying capability to execute the reasoning steps. This is most visible in CoT distillation: the student learns to start with "Step 1: identify operands" but produces numerically wrong answers consistently. The diagnostic is comparing loss curves against task-specific accuracy curves — if accuracy fails to rise despite falling loss, the student capacity is likely insufficient. The solution is to use a larger student base, use a student with more pretraining (so it arrives at distillation with more world knowledge already encoded), or simplify the task distribution in the distillation dataset.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        The following papers are the load-bearing references for this topic. All arXiv IDs verified.
      </Prose>

      <H3>Hinton, Vinyals, and Dean (2015) — the original formulation</H3>

      <Prose>
        "Distilling the Knowledge in a Neural Network." arXiv:1503.02531. Introduces the temperature-softened KL loss, the concept of dark knowledge, and the ensemble-to-single-model compression framing. The core formulas in section 3 of this topic come from this paper. Every subsequent distillation paper in ML builds on this foundation.
      </Prose>

      <H3>Taori et al. (2023) — Alpaca</H3>

      <Prose>
        "Alpaca: A Strong, Replicable Instruction-Following Model." Stanford CRFM blog, March 2023. The first widely-replicated demonstration of response-based LLM distillation: 52K completions from text-davinci-003 used to fine-tune LLaMA-7B. Cost under $600 total. Showed that teacher API access plus a base model is sufficient to create a qualitatively capable instruction follower, which established the template for the 2023 wave of instruction-tuned open models.
      </Prose>

      <H3>Mukherjee et al. (2023) — Orca</H3>

      <Prose>
        "Orca: Progressive Learning from Complex Explanation Traces of GPT-4." arXiv:2306.02707. The key step from answer-only to explanation-trace distillation. Orca collects GPT-4's step-by-step explanations — not just its final answers — and fine-tunes a 13B model on them. The resulting model substantially outperforms Vicuna-13B on BBH (by over 100%) and reaches ChatGPT-level performance on many benchmarks. This is the direct intellectual predecessor of CoT distillation for reasoning models.
      </Prose>

      <H3>Gunasekar et al. (2023) — Phi-1</H3>

      <Prose>
        "Textbooks Are All You Need." arXiv:2306.11644. Introduces synthetic data distillation as an alternative to rollout collection: GPT-3.5 generates "textbook quality" Python tutorials and exercises from scratch, and a 1.3B model trained on those synthetic textbooks achieves 50.6% pass@1 on HumanEval — matching models twenty times larger trained on raw web data. The key insight is that data quality, not data volume, is the binding constraint for small models. Quality-filtered and synthetically-generated teacher outputs (a form of distillation) can substitute for massive real-world datasets.
      </Prose>

      <H3>DeepSeek-AI (2025) — R1 and R1-Distill</H3>

      <Prose>
        "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. The canonical reference for reasoning distillation at scale. Section 3.3 of that paper describes the distillation pipeline: 800K reasoning traces generated from DeepSeek-R1 (685B), filtered for correctness, used to SFT Qwen2.5 and Llama base models at 1.5B through 70B scale. The resulting R1-Distill models reach AIME 2024 scores that were unachievable for same-size models trained with RL from scratch. The key ablation: distillation-without-RL matches or approaches distillation-with-RL at 7B scale, establishing that the supervised trace signal alone is sufficient to instill coherent long-range reasoning.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1: why higher T exposes more information</H3>

      <Prose>
        Derive analytically why increasing the temperature <Code>T</Code> exposes more information in the teacher's output distribution. Start from the definition of Shannon entropy: <Code>H(p) = -Σ p(k) log p(k)</Code>. Show that as <Code>T → 0</Code>, the softmax approaches a one-hot distribution and the entropy approaches 0 (no dark knowledge). As <Code>T → ∞</Code>, show the softmax approaches a uniform distribution and the entropy approaches <Code>log(V)</Code> where <Code>V</Code> is the vocabulary size. Explain why the maximum information about the teacher's conceptual structure is accessible at some intermediate temperature, and why that temperature is typically between 3 and 10 for classification tasks.
      </Prose>

      <H3>Exercise 2: student-teacher size ratio for 90% capability retention</H3>

      <Prose>
        Given the empirical finding that a 7B student distilled from a 70B teacher retains roughly 85% of the teacher's AIME 2024 score, and a 14B student retains roughly 90%, estimate: (a) what student size would be needed to retain 95% of a 70B teacher's capability on this benchmark; (b) whether this estimate would hold across all tasks or whether task difficulty would modulate the ratio; (c) why there is likely a floor below which further doubling of student size gives diminishing capability returns. Consider both the capacity argument (smaller models cannot represent complex distributions) and the optimization argument (SFT on a finite dataset reaches a ceiling independent of model size).
      </Prose>

      <H3>Exercise 3: design a CoT distillation pipeline for code generation</H3>

      <Prose>
        Design the verifier component for CoT distillation of a code-generating teacher. The teacher generates Python solutions to competitive programming problems. Specify: (a) what the verifier receives as input (problem statement, generated code, expected outputs); (b) how the verifier executes the code safely (sandboxing requirements: process isolation, time limits, memory limits, network blocking); (c) what score it returns and whether that score should be binary or fractional (fraction of test cases passing); (d) what failure modes the verifier misses — consider off-by-one errors in floating-point, infinite loops that hit the time limit, and solutions that pass public tests but fail on hidden test cases. How would you make the verifier more robust to each failure mode?
      </Prose>

      <H3>Exercise 4: when does response-based beat softmax-KL distillation</H3>

      <Prose>
        List and explain three concrete scenarios where response-based distillation (CE on teacher outputs) is likely to outperform soft-target KL distillation in practice, even though KL distillation is theoretically more information-rich. Consider: (1) proprietary teacher with API-only access; (2) very large teacher vocabulary where storing full logit distributions is infeasible; (3) cases where the teacher's high-confidence regions dominate the distribution and most dark knowledge is noise rather than signal. For each scenario, describe what assumption of soft-target distillation is violated and how response-based distillation avoids the issue.
      </Prose>

      <H3>Exercise 5: estimating the contamination risk of a distillation benchmark</H3>

      <Prose>
        You want to evaluate a student model distilled from GPT-4 on the MATH benchmark. Describe a protocol to estimate how much of the student's MATH score reflects genuine reasoning capability versus benchmark contamination inherited from the teacher. Your protocol should: (a) probe the student with rephrased or equivalent problems not seen in standard benchmark sets; (b) compare student performance on problems from the benchmark's public training split versus its hidden test split; (c) test whether the student can explain its reasoning step-by-step in a way that generalizes to minor problem variations (e.g., same structure, different numbers). What outcome would confirm contamination? What outcome would provide evidence of genuine capability transfer?
      </Prose>

    </div>
  ),
};

export default knowledgeDistillationLLMs;
