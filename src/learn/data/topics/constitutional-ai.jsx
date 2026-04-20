import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { StepTrace, TokenStream } from "../../components/viz";

const constitutionalAI = {
  title: "Constitutional AI (CAI)",
  readTime: "12 min",
  content: () => (
    <div>
      <Prose>
        RLHF depends on tens of thousands of human-labeled comparisons. The cost is real, the
        consistency is imperfect, and labelers become a bottleneck. A labeler who is tired, rushed,
        or simply uncertain about what "better" means introduces noise that accumulates across the
        dataset. Constitutional AI (Anthropic, 2022) asks a sharper question: what if the AI
        critiques and improves its own outputs, guided by a written set of principles? The answer
        turned out to matter — CAI is how Claude was trained from its earliest versions, and the
        technique has spread to most frontier alignment pipelines.
      </Prose>

      <H2>The core idea — AI-generated feedback</H2>

      <Prose>
        The central move is to replace human preference labelers with an earlier-generation model
        acting as a critic. Give that critic model a written "constitution" — a list of principles
        like "prefer responses that are helpful," "avoid responses that are harmful," "prefer
        responses that are honest." For each training sample, the critic identifies problems in the
        model's draft response, then rewrites it to be better. The revised responses become
        supervised fine-tuning data. When preferences are needed, the critic also ranks pairs of
        responses against the constitution.
      </Prose>

      <Prose>
        This is a compact description of a significant architectural shift. The training signal
        that used to require human annotators is now generated on demand, at the cost of a model
        inference call. The norms that used to be implicit — whatever preferences the annotator
        pool happened to embody — are now written down explicitly, in plain language, in a document
        you can read and argue about. Both of those changes have practical consequences that go
        beyond the efficiency gain.
      </Prose>

      <H2>The two-stage pipeline</H2>

      <Prose>
        CAI has two distinct phases, run sequentially. The first establishes a better base through
        supervised learning; the second refines preferences through reinforcement.
      </Prose>

      <Prose>
        Phase one is supervised learning from self-critique. Given a prompt, the model generates an
        initial response. The model is then prompted to critique that response against a
        randomly-chosen constitutional principle — one principle at a time, not the entire list at
        once. The critique identifies the specific problem. The model then rewrites the response
        to address it. The (prompt, revised response) pair is kept as SFT training data. Many
        revisions can be chained — critique the revision, rewrite again — though Anthropic's
        experiments found that one or two rounds captured most of the improvement.
      </Prose>

      <Prose>
        Phase two is reinforcement learning from AI feedback, or RLAIF. The supervised model from
        phase one generates pairs of responses to the same prompt. The model is asked to rank the
        two responses against a constitutional principle, producing a preference label. These
        AI-generated labels train a reward model. The rest of the pipeline proceeds identically to
        RLHF — PPO or a similar algorithm optimizes the policy against the reward model. The human
        is not in the loop for any individual label. The human wrote the constitution.
      </Prose>

      <StepTrace
        label="constitutional ai — two-stage pipeline"
        steps={[
          { label: "1. self-critique", render: () => (
            <TokenStream tokens={["base model → draft", " →", " critic + principle", " →", " critique", " →", " rewrite"]} />
          ) },
          { label: "2. sl on revisions", render: () => (
            <TokenStream tokens={["many (prompt, revised)", " →", " fine-tune", " →", " SL-CAI model"]} />
          ) },
          { label: "3. ai preference labels", render: () => (
            <TokenStream tokens={["SL-CAI → pairs", " →", " AI ranks under constitution", " →", " preference data"]} />
          ) },
          { label: "4. rlaif", render: () => (
            <TokenStream tokens={["AI preferences", " →", " reward model", " →", " PPO/DPO", " →", " final CAI model"]} />
          ) },
        ]}
      />

      <H3>What a constitution looks like</H3>

      <Prose>
        Principles are written in plain language, not code. They read more like a style guide or
        an ethics policy than a loss function. Examples from Anthropic's original paper:
      </Prose>

      <CodeBlock>
{`- Please choose the response that is the most helpful, honest, and harmless.
- Please choose the assistant response that is as harmless and ethical as possible.
- Please choose the response that is more supportive and helpful, especially for users
  facing difficult circumstances.
- Please choose the response that is less preachy, obnoxious, or overly-reactive.`}
      </CodeBlock>

      <Prose>
        The critic model is prompted with one principle at a time, per revision. This focuses its
        attention and produces more consistent critiques than presenting the full list simultaneously.
        Presenting a long list invites vague, multi-concern responses; presenting a single principle
        invites a targeted one. This is a subtle but load-bearing design choice — the modularity of
        the constitution is what makes each individual critique tractable.
      </Prose>

      <H2>Why this works</H2>

      <Prose>
        Three reasons, each operating at a different level.
      </Prose>

      <Prose>
        First, models are reasonably good at applying stated principles. Asking an LLM "does this
        response respect principle X" gets decent answers, especially when the principle is concrete
        and the model has been exposed to enough examples of what respecting it looks like. The
        critique quality is far from perfect, but it is good enough to be informative, and
        informative labels — even noisy ones — are useful for training.
      </Prose>

      <Prose>
        Second, the constitution is explicit and auditable, unlike the implicit values encoded in any
        given batch of human labels. Disagreements about alignment become disagreements about the
        constitution, which is a more tractable argument. You can point to a principle, dispute it,
        revise it, and observe what changes in model behavior. You cannot do any of that with the
        latent preferences of an annotator pool. This auditability matters for iteration speed and
        for organizational accountability.
      </Prose>

      <Prose>
        Third, scaling. One well-written constitution can generate millions of preference labels for
        the cost of inference, versus human labeling which caps at roughly what you can afford per
        thousand labels. The economics are not close. As models get stronger, the critique quality
        improves without any change to the constitution, because better models apply principles more
        reliably. The alignment pipeline gets better as a side effect of capability scaling, which
        is a property human-labeling pipelines don't share.
      </Prose>

      <H2>The critical caveat — the critic has to be good enough</H2>

      <Prose>
        CAI works when the critic model is capable enough to notice genuine problems. This
        constraint is sharper than it sounds. On tasks where the model already performs well —
        writing, summarization, tone calibration, adherence to formatting norms — AI critique is
        useful and often excellent. On tasks where the model itself is weak, AI critique just
        replicates the weakness.
      </Prose>

      <Prose>
        A model that cannot distinguish a correct from an incorrect mathematical proof will not
        improve its mathematics through self-critique. It will revise proofs in ways that sound
        more confident, more structured, more principle-adherent — and it will still get the
        underlying reasoning wrong in the same ways. The critique can polish the surface while
        leaving the problem untouched. This shapes what CAI is good for: style, tone, safety,
        refusal behavior, honesty at a level the model can evaluate. It is weaker as a tool for
        pushing raw capability, and essentially useless for correcting errors in domains where the
        model cannot detect those errors.
      </Prose>

      <Callout accent="gold">
        AI feedback compounds what the model can already do well. It does not conjure capability the model doesn't have.
      </Callout>

      <H2>How CAI shows up in practice</H2>

      <Prose>
        Claude's character and refusal patterns are directly shaped by CAI. The particular texture
        of Claude's defaults — the way it pushes back on requests without being preachy, the way
        it acknowledges constraints rather than ignoring them, the consistency of its tone across
        very different prompts — reflects a training process that explicitly optimized for those
        properties using written principles as the signal.
      </Prose>

      <Prose>
        The approach has propagated widely. Meta's Llama 3 training report describes a version of
        the same pipeline: synthetic preference data generated from a seed model, guided by written
        rubrics. OpenAI's "deliberative alignment" (2024) is closely related — a model critiques
        its own responses against a written specification before finalizing the output. Google's
        alignment work references similar structures under different terminology. The specific
        framing varies across labs; the structural move is nearly identical. A written specification,
        a model applying it to its own outputs, the results used as training signal — this is now
        the default pattern, not the experimental one. The name "constitutional AI" belongs to
        Anthropic's paper; the technique belongs to the field.
      </Prose>

      <Prose>
        RLAIF, described in the DPO and RLHF topics, builds directly on this foundation. The
        preference data that DPO trains on can be generated by exactly this mechanism — an AI
        critic ranking response pairs against a written specification. CAI is the upstream of many
        of the techniques in this section of the track.
      </Prose>

      <H3>What the constitution can't do</H3>

      <Prose>
        Constitutions are a blunt instrument for nuanced problems. A principle like "be helpful"
        does not specify what to do when helpfulness conflicts with honesty — when the most
        immediately satisfying answer is not the most accurate one. A principle like "avoid harmful
        responses" does not specify where harm begins, who counts as harmed, or how to weigh harm
        against value in borderline cases. The principles interact, and the interactions are not
        written down.
      </Prose>

      <Prose>
        Labs iterate in response. Constitutions have become longer and more specific, with rules
        that cover edge cases explicitly, carve-outs for particular domains, and priority orderings
        for when principles conflict. This is the right engineering response to the problem. But at
        some point the constitution starts to resemble a detailed policy document rather than a set
        of principles — which raises its own maintenance burden, its own ambiguities, and its own
        question about whether the model is learning the spirit of the document or optimizing
        against its letter. That tension does not have a clean resolution; it just requires ongoing
        attention.
      </Prose>

      <Prose>
        There is also a subtler issue with distributional coverage. A constitution written by a
        small team will reflect the values and anticipate the edge cases that the team can imagine.
        Prompts that fall outside that imaginative range — novel attack vectors, cultural contexts
        the authors didn't consider, technical domains with their own norms — may not be well-served
        by the principles that exist. This is not an argument against having a constitution; it is
        an argument for humility about how complete any constitution actually is.
      </Prose>

      <H2>What comes next</H2>

      <Prose>
        CAI reframed alignment as a question you can write down rather than a preference you can
        only demonstrate. That is a genuine conceptual advance. Before CAI, the alignment signal
        lived entirely in the annotation process — implicit, expensive, hard to inspect. After CAI,
        part of that signal lives in a document that can be read, debated, versioned, and improved
        independently of any training run.
      </Prose>

      <Prose>
        The next several topics — process reward models, RLAIF and its variants, RLVR — keep
        pushing in the same direction: reduce the human bottleneck, increase the structure of the
        signal, make the feedback mechanism legible. CAI is the first clear articulation of that
        research program. The techniques that follow it are elaborations on the same theme: if you
        can write down what you want, you can use the model to help you get it.
      </Prose>
    </div>
  ),
};

export default constitutionalAI;
