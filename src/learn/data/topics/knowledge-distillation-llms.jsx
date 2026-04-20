import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const knowledgeDistillationLLMs = {
  title: "Knowledge Distillation for LLMs (DeepSeek-R1-Distill, CoT Distillation)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        A 70B-parameter reasoning model costs real money to serve. A 7B model trained to imitate that larger model, on the same reasoning traces, can retain 70–90% of the capability at roughly a tenth the inference cost. This is distillation — and for reasoning models specifically, it has become one of the most practically important post-training techniques. DeepSeek-R1-Distill and its successors are the canonical example.
      </Prose>

      <Prose>
        The idea predates LLMs by a decade. But in the context of large reasoning models, distillation has taken on a particular character: it is primarily about transferring the procedure of reasoning, not just the distribution of answers. A student that has learned how a teacher reasons can generalize in ways that a student trained only on the teacher's conclusions cannot. That distinction — between imitating outputs and imitating process — is what makes CoT distillation qualitatively different from classical compression.
      </Prose>

      <H2>The classical knowledge distillation setup</H2>

      <Prose>
        Hinton, Vinyals, and Dean's 2015 paper framed the idea precisely. The goal is not to train a student model to maximize accuracy on some gold-labeled dataset — it is to train the student to match the teacher's output distribution. For classification, the teacher produces a softmax over class logits. Most of the probability mass lands on the correct class, but a small, structured amount is distributed across wrong classes. A dog image that the teacher assigns 92% probability to "dog" but 6% to "wolf" and 1% to "cat" is telling you something about the teacher's internal concept of dogness — that wolves are similar to dogs, cats are somewhat similar, and fish are irrelevant. That relational structure is "dark knowledge."
      </Prose>

      <Prose>
        The student minimizes KL divergence to the teacher's softened distribution, not cross-entropy to hard labels:
      </Prose>

      <MathBlock>{"\\mathcal{L}_{KD} = T^2 \\cdot \\text{KL}\\left(\\text{softmax}(z_s/T) \\,\\|\\, \\text{softmax}(z_t/T)\\right)"}</MathBlock>

      <Prose>
        The temperature <Code>T</Code> softens both distributions before comparing them. At <Code>T = 1</Code> the teacher's distribution is sharp — most probability on the correct class, trace amounts elsewhere. At <Code>T = 4</Code> or <Code>T = 10</Code>, the distribution spreads out and the relative ordering of wrong answers becomes visible. The factor of <Code>T²</Code> rescales the loss back to a magnitude comparable to ordinary cross-entropy, so the training signal doesn't shrink to noise. For LLMs, the same logic generalizes to matching the teacher's next-token distribution at every position in the sequence — thousands of soft targets per training example instead of one.
      </Prose>

      <H2>For LLMs, distillation is often simpler than that</H2>

      <Prose>
        The standard practical recipe used by the majority of small open-source instruction models does not use KL divergence or soft targets at all. Instead: run the teacher model over a set of prompts, collect its completions, and SFT the student on those (prompt, completion) pairs as ordinary supervised examples. The teacher's output is treated as a gold label. This is sometimes called "response-based distillation" or "instruction distillation," and it is the technique behind Alpaca (bootstrapped from Text-davinci-003), Vicuna (trained on ShareGPT conversations), WizardLM, and most small instruction-following models released in 2023.
      </Prose>

      <Prose>
        Why abandon the principled KL formulation? Practical reasons. Soft targets require access to the teacher's full logit distribution at inference time, which is expensive to store and often unavailable for proprietary models. Sampling completions from an API is cheap. And empirically, for instruction following — as opposed to classification — the gap between soft-target distillation and response distillation is small enough that the simplicity wins.
      </Prose>

      <CodeBlock language="python">
{`def build_distillation_dataset(teacher, prompts, temperature=0.7):
    """Generate a training dataset by sampling completions from the teacher."""
    dataset = []
    for prompt in prompts:
        completion = teacher.generate(prompt, temperature=temperature)
        dataset.append({"prompt": prompt, "response": completion})
    return dataset

# The student is then trained with standard SFT on this dataset.
# Teacher acts as a "ground truth" that's much cheaper to generate than human labels.`}
      </CodeBlock>

      <Prose>
        The temperature at generation time matters. At <Code>temperature=0</Code> the teacher always produces its modal completion — diverse prompts but uniform style. At higher temperatures the dataset covers more of the teacher's distribution, which can expose the student to better reasoning variation or, at high enough temperature, to incoherent outputs. Most practitioners settle somewhere between 0.6 and 0.9 and use light quality filtering to discard obvious failures.
      </Prose>

      <H2>CoT distillation — the reasoning-specific move</H2>

      <Prose>
        When the teacher is a reasoning model that emits chain-of-thought traces before arriving at a final answer, distillation from those traces is dramatically more effective than distillation from final answers alone. This is the finding that runs through Ho et al. (2022), Magister et al. (2023), and the wave of distillation papers that followed. The student is trained to generate the teacher's full reasoning chain — every intermediate step — and then the final answer. It learns not just what the teacher concluded, but the procedure that got there.
      </Prose>

      <Prose>
        The effect is outsized for small models. A 7B model fine-tuned on final-answer pairs from a large teacher will improve measurably. The same 7B model fine-tuned on the teacher's step-by-step reasoning traces will improve several times more on tasks that require multi-step inference. The student learns a reasoning style — how to decompose problems, when to pause and check, how to format intermediate calculations — that it would be extremely unlikely to develop on its own from pretraining signal alone.
      </Prose>

      <Prose>
        There is a ceiling. The student's reasoning ability is bounded above by the teacher's: if the teacher cannot reliably solve a class of problems, the distilled traces for those problems will be wrong or incoherent, and training on them hurts. But within the teacher's competence, distillation compresses the reasoning ability into a model two to three weight classes smaller. DeepSeek-R1-Distill-Qwen-7B reaches AIME 2024 scores that were unthinkable for a 7B model in 2023. That number is the practical argument for the whole approach.
      </Prose>

      <H3>DeepSeek-R1-Distill — the canonical recipe</H3>

      <Prose>
        DeepSeek open-sourced both their full reasoning model and the distillation pipeline that produced smaller variants. The recipe has four stages, though one of them matters much less than you might expect.
      </Prose>

      <StepTrace
        label="deepseek-r1-distill pipeline"
        steps={[
          { label: "1. generate CoT traces", render: () => (
            <TokenStream tokens={["DeepSeek-R1 (685B)", " →", " ~800K reasoning traces on math/code/STEM"]} />
          ) },
          { label: "2. filter for correctness", render: () => (
            <TokenStream tokens={["800K traces", " →", " keep only verified-correct", " →", " ~600K curated"]} />
          ) },
          { label: "3. SFT the student", render: () => (
            <TokenStream tokens={["Qwen 7B/14B/32B base", " →", " SFT on 600K CoT traces", " →", " R1-Distill-Qwen-*"]} />
          ) },
          { label: "4. optional RL polish", render: () => (
            <TokenStream tokens={["R1-Distill-Qwen-7B", " →", " brief RLVR pass", " →", " final release"]} />
          ) },
        ]}
      />

      <Prose>
        The filtering step in stage two is doing more work than it appears. The 685B teacher generates 800K traces, but only a subset are verifiably correct — for math problems, the final numerical answer can be checked; for code, the output can be executed. Keeping only correct traces means the student trains on a clean, high-quality signal. This correctness filter is arguably the most consequential engineering decision in the pipeline. Training on the teacher's incorrect reasoning traces, even though they are often coherent and informative, degrades the student's benchmark performance.
      </Prose>

      <Prose>
        The striking finding from DeepSeek's own ablations: stage four — the RL polish — added relatively little to final benchmark numbers. Most of the gain came from pure SFT on verified CoT traces. Distillation-without-RL at 7B scale matched or approached distillation-with-RL at a fraction of the training cost. This is both encouraging and somewhat counterintuitive — the received wisdom is that RL is what produces coherent long-range reasoning, but at distillation scale, the supervised signal alone appears sufficient to instill it.
      </Prose>

      <H2>Why CoT distillation beats vanilla RL for small models</H2>

      <Prose>
        Running RL from scratch on a small model is, at the bottom, a search problem. The model generates completions, some are correct, some are not, and the policy gradient nudges the model toward the correct ones. For reasoning-heavy tasks like AIME problems or competition-level code, a small model starting from a general pretrained base produces correct solutions rarely enough that the gradient signal is sparse to the point of uselessness. The model explores the wrong region of solution space, gets mostly zero reward, and barely moves. This is not a hyperparameter problem. It is a fundamental exploration challenge that gets harder as the task gets harder and the model gets smaller.
      </Prose>

      <Prose>
        CoT distillation sidesteps the exploration problem entirely. Every training example is a known-correct reasoning trace — not a reward signal from a distant outcome, but a dense, step-by-step supervision at every token. The student never has to discover how to reason about multi-step arithmetic by trial and error; it reads ten thousand examples of the teacher doing it. The learning problem becomes imitation rather than search, and imitation is a problem that supervised learning handles cleanly. DeepSeek's paper made this comparison explicit: RL applied from scratch on the same Qwen base model at 7B scale produced weaker reasoning than distillation from R1, using comparable training compute. The distillation path was both cheaper and more effective.
      </Prose>

      <Prose>
        This does not mean RL is irrelevant for small models. After distillation establishes a baseline reasoning style, even a short RL fine-tuning pass on verifiable tasks can push benchmark numbers further. The practical finding is that the ordering matters: distill first to establish competence, then optionally RL-polish to sharpen it. Skipping distillation and going straight to RL on a naive base model at small scale produces a much weaker result per compute dollar.
      </Prose>

      <H2>The limits</H2>

      <Prose>
        Distillation's ceiling is the teacher. This is obvious in principle but worth spelling out in terms of what it means for capability development. If the teacher cannot reliably solve competition-level geometry, the student will not learn to solve it either — and worse, it may learn confident-looking but wrong reasoning chains for that class of problems, which is harder to detect than simple ignorance. The student inherits not just the teacher's competences but its systematic blind spots.
      </Prose>

      <Prose>
        The contamination concern runs deeper. The teacher was trained on data that likely includes the benchmark problems the student will be evaluated on. The teacher's correct traces for those problems may reflect memorization rather than reasoning. The student, trained on those traces, learns to reproduce the memorized solution procedure. The benchmark score improves; the underlying reasoning ability may not. This is not unique to distillation — pretrained base models have the same issue — but distillation concentrates it, because the student trains directly on the teacher's benchmark-adjacent outputs.
      </Prose>

      <Prose>
        There is also the question of diversity. A distilled model reasons in one style: the teacher's. That style may be excellent for the domains the teacher was strong in and brittle for everything else. A model trained from scratch with RL on novel problem distributions develops whatever reasoning patterns the reward structure elicits — sometimes worse, sometimes qualitatively different in ways that generalize to genuinely new problems. The distilled model is, in a precise sense, a copy. It has the teacher's strengths and the teacher's shape.
      </Prose>

      <Callout accent="gold">
        Distillation trades originality for accessibility. The student reasons exactly like the teacher, at a fraction of the cost — which is both why it's useful and why it can't push the frontier.
      </Callout>

      <Prose>
        Distillation is how frontier capabilities move down the weight-class ladder. It is not how frontier capabilities get created — that requires actual RL on genuinely new problems, the kind of exploration where the model encounters a class of challenge its teacher never handled cleanly, and has to work something out from scratch. That is what the final topic in this section covers.
      </Prose>
    </div>
  ),
};

export default knowledgeDistillationLLMs;
