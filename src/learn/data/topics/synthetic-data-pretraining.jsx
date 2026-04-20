import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";

const syntheticDataPretraining = {
  title: "Synthetic Data Generation for Pre-Training",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        Pre-training data is running out. Not in the sense that the internet has gone quiet — it
        hasn't — but in the sense that the highest-quality, legally unambiguous, linguistically
        diverse text available for model training is a bounded resource. Common Crawl grows, but
        its signal-to-noise ratio degrades as models get better at telling the difference between
        a thoughtful paragraph and a spun SEO article. Copyrighted books and scientific papers
        sit behind licenses that make large-scale ingestion legally fraught. Non-English text
        remains scarce relative to the fraction of the world that doesn't read English. At the
        scale frontier-class models now operate, the curves that used to look like "more data
        always helps" are starting to bend.
      </Prose>

      <Prose>
        The obvious response: generate the data you don't have. A sufficiently capable model can
        produce text, and inference is cheap enough that generating trillions of tokens is no
        longer a fantasy budget line. The less-obvious question is whether that text actually
        helps a model train — or whether the model ends up consuming a degraded reflection of
        what it already knows, shrinking rather than expanding its understanding of the world.
        Both things can be true simultaneously, in different regimes.
      </Prose>

      <H2>The case for synthetic pre-training data</H2>

      <Prose>
        Three motivations are genuine. They are worth keeping separate because they have different
        evidence bases and different failure modes.
      </Prose>

      <Prose>
        The first is diversity. Real-data corpora are systematically thin in certain domains:
        step-by-step mathematical derivations, explicit reasoning chains, and low-resource
        languages where web text is sparse and uneven. A strong model can be prompted to generate
        thousands of worked calculus problems, formal proofs, or grounded reasoning traces that
        simply don't exist in the volumes needed on the open web. The domain being synthesized
        determines whether this works — math and code lend themselves to it in ways that, say,
        literary fiction does not.
      </Prose>

      <Prose>
        The second is quality. Filtered synthetic data can be more uniformly good than the best
        filtered real data, because you control the generation conditions end-to-end. Prompt a
        strong model to write a clear pedagogical explanation, ask another model to evaluate it,
        discard anything below the threshold. The resulting distribution has fewer outliers than
        a corpus assembled by crawling and filtering the open web, which will always contain
        noise that the filter misses.
      </Prose>

      <Prose>
        The third is scale. LLM inference has become cheap enough that generating large volumes
        of text — tens of billions of tokens, in some pipelines — is economically tractable.
        That wasn't true in 2020. It is now, and the cost continues to fall. Whether cheap
        generation translates into useful training signal is a different question, but the raw
        logistics are no longer the bottleneck.
      </Prose>

      <H2>Textbooks Are All You Need — the phi series</H2>

      <Prose>
        The most influential demonstration of synthetic pre-training data is Microsoft's phi
        series. In 2023, the phi-1 paper argued that a small code model trained on carefully
        curated and synthetically generated "textbook quality" data could punch far above its
        weight. phi-1, at 1.3B parameters, matched models three to five times its size on
        HumanEval — a Python coding benchmark that evaluates functional correctness, not style.
        The difference wasn't architecture or training duration. It was the data.
      </Prose>

      <Prose>
        The recipe: take a strong teacher model, GPT-4 class or close to it, and prompt it to
        generate didactic content — worked examples, explanations that build concept by concept,
        exercises with solutions. Do this at scale across a carefully chosen curriculum of topics.
        Filter aggressively using a quality classifier. Deduplicate against the generated set
        itself. Mix the synthetic data with filtered real code at roughly 10–30% of total
        training tokens, and train the small student on the result. phi-2 and phi-3 extended this
        to general reasoning with similar findings: the data quality story held up at slightly
        larger scales and broader domains.
      </Prose>

      <Prose>
        What the teacher generates looks like this.
      </Prose>

      <CodeBlock language="python">
{`def generate_textbook_chapter(teacher, topic, style="undergraduate"):
    prompt = f"""Write a clear, pedagogical chapter on {topic} at
    the {style} level. Include motivation, a worked example, common
    pitfalls, and exercises with solutions.

    The chapter should read as if from a well-reviewed textbook.
    Do not write the word 'textbook' itself."""
    return teacher.generate(prompt, max_tokens=4096, temperature=0.7)

# Typical pipeline: millions of (topic, style) pairs, filtered by a
# quality classifier, deduplicated against each other, mixed ~10-30%
# of total training tokens alongside filtered web data.`}
      </CodeBlock>

      <Prose>
        The prompt design is not cosmetic. The explicit instruction to "not write the word
        textbook itself" is an attempt to avoid the model producing meta-commentary about what
        it's doing rather than doing it. Small prompt choices at the generation stage compound
        across millions of samples.
      </Prose>

      <H2>Distillation from a stronger teacher</H2>

      <Prose>
        A special case of synthetic data is knowledge distillation at the data level: a strong
        teacher model generates responses to real prompts, and a smaller student trains on those
        responses as if they were ground truth. This is how Alpaca bootstrapped
        instruction-following from Llama 1 paired with GPT-3.5 outputs. It's how WizardLM
        scaled up by having the teacher augment and complicate existing instructions before
        answering them. It's how most small open-weight instruct models in production are made.
      </Prose>

      <Prose>
        This isn't strictly pre-training — it usually lives in the post-training phase, fine-tuning
        a base model on instruction-response pairs. But the line blurs in practice. Several labs
        have experimented with mixing teacher-generated responses into continued pre-training
        corpora, particularly for domains like math where teacher outputs can be verified for
        correctness before inclusion. The pedagogical structure argument from phi applies here
        too: the teacher produces text that models good reasoning explicitly, which may be more
        useful to train on than raw web text that contains the same facts in a less learnable form.
      </Prose>

      <H2>Model collapse — the failure mode</H2>

      <Prose>
        In 2024, Shumailov and colleagues published a paper in Nature demonstrating what they
        called model collapse: a model trained on the outputs of a previous model, which was
        itself trained on synthetic data, degrades in systematic and measurable ways. The tails
        of the distribution shrink. Rare but valid outputs become increasingly unlikely. After
        roughly five to ten generations of "train on previous model's output," the resulting
        model fails on diverse inputs that earlier generations handled fine.
      </Prose>

      <Prose>
        The mechanism is worth understanding precisely, because it's not just about errors
        propagating. Each time the model generates text and that text is used for training, the
        sampling process discards information. The model draws from its learned distribution;
        that distribution is an approximation of the true data distribution; and the approximation
        gets coarser each time it's treated as the source of truth. Low-probability but real
        events — the unusual phrasing, the rare fact, the minority opinion — get underrepresented
        each generation, then further underrepresented, then effectively absent. The vocabulary
        of the model's outputs narrows. The KL divergence between the true distribution and the
        learned one grows with each recursive generation:
      </Prose>

      <MathBlock>{"D_{KL}(p \\mid\\mid p_k) \\geq k \\cdot D_{KL}(p \\mid\\mid p_1)"}</MathBlock>

      <Prose>
        Loosely: each generation of self-training adds at least as much divergence from reality
        as the first generation did. The exact bound depends on assumptions about the generation
        process and model capacity, but the direction is robust across the paper's experimental
        setups. The problem isn't simply that synthetic data can be wrong. It's that synthetic
        data from this model cannot expand what this model knows. It can only rearrange — and
        lossy rearrangement, repeated, converges toward something narrower than what you started
        with.
      </Prose>

      <H3>Mitigation — stay grounded in real data</H3>

      <Prose>
        The working pattern across published work is consistent: always mix synthetic data with
        real data, and never train a generation of model purely on the previous generation's
        outputs. The mixing ratio matters, though no paper has established a universal right
        answer — it depends on domain, synthetic data quality, and how different the synthetic
        distribution is from the real one.
      </Prose>

      <Prose>
        Using a stronger external teacher rather than the model being trained sidesteps the
        worst of model collapse. The teacher's distribution is richer than the student's; the
        student is learning from a source that knows more than it does, rather than from a noisy
        copy of itself. This is the phi setup and the standard distillation setup. The danger
        reappears when the teacher and student approach the same capability level, or when the
        teacher's outputs are themselves heavily synthetic.
      </Prose>

      <Prose>
        Keeping a meaningful fraction of real web data in the mix — even if that web data is
        noisier than the synthetic content on any individual quality metric — preserves the tail
        diversity that synthetic generation tends to erode. Rare languages, unusual phrasings,
        and the long tail of human knowledge live in that noisy web data in ways that even the
        best synthetic pipeline doesn't recover.
      </Prose>

      <H2>Where synthetic is winning</H2>

      <Prose>
        The honest picture, as of now, is that synthetic data works clearly in a few regimes and
        is still speculative in others.
      </Prose>

      <Prose>
        Math is the clearest win. MetaMath, MathInstruct, and their derivatives generate
        large volumes of math solutions — rephrased, augmented, and verified by execution where
        possible — and models trained on them improve measurably on math benchmarks. The key
        is that many math problems have checkable answers, which allows post-hoc filtering of
        synthetic data by correctness rather than purely by style or fluency.
      </Prose>

      <Prose>
        Code has the same property. A generated function either passes its tests or it doesn't.
        Execution-based filtering makes synthetic code data a fundamentally different
        proposition from synthetic prose: you can generate a million candidate solutions and
        keep only the ones that work. The resulting dataset has a quality floor that human-curated
        datasets can't match at the same volume.
      </Prose>

      <Prose>
        Reasoning chains for reinforcement learning have become a central use case, particularly
        for building reward model training data. If you need examples of humans preferring one
        response over another, you can generate candidate responses synthetically and use a
        stronger model to rank them, bootstrapping the preference data that RLHF requires.
        This is technically post-training territory, but the data generation techniques are
        identical.
      </Prose>

      <Prose>
        Low-resource languages are a genuine opportunity. Translating high-quality English
        corpora using strong translation models produces text in Swahili, Tagalog, or
        Welsh at a volume that organic web scraping cannot match. The quality is uneven and
        culturally thin — translation preserves information, not local context — but for
        basic capability coverage, it's better than nothing and sometimes considerably better
        than the noisy low-resource web text available otherwise.
      </Prose>

      <H3>Where it's still open</H3>

      <Prose>
        The open questions are not minor.
      </Prose>

      <Prose>
        Whether synthetic-heavy training reaches the same capability ceiling as real-data-heavy
        training is genuinely unknown at the scale frontier. The phi results are striking, but
        phi models are small — 1B to 14B parameters — and the training budgets involved are
        modest compared to what frontier labs run. Whether the quality advantage of synthetic
        data persists when you're training a 70B model on trillions of tokens is not yet
        publicly answered.
      </Prose>

      <Prose>
        Whether model collapse is a real concern at current scale or only at the extremes
        Shumailov tested is also unsettled. The paper's experimental setups involved fairly
        extreme recursive training — models training on nothing but previous model outputs.
        Real pipelines never do this. But the mechanism is real, and how quickly it bites in
        milder forms of synthetic-heavy training remains an open empirical question.
      </Prose>

      <Prose>
        Contamination detection is becoming genuinely hard. When synthetic data looks like
        real data — and good synthetic data increasingly does — the usual tools for measuring
        benchmark contamination don't work. A model may have trained on synthetically generated
        variants of benchmark problems without any clean way to detect that from outside. As
        the fraction of synthetic data in pre-training corpora grows, the integrity of public
        benchmarks becomes harder to verify.
      </Prose>

      <Callout accent="gold">
        Synthetic data is the current bet for pushing past the data wall. Whether that bet pays
        off — or quietly produces a generation of models with thinner tails — is still being
        settled.
      </Callout>

      <Prose>
        This is the frontier topic of Pre-Training for a reason: no one is certain it works
        at the scales that matter. The evidence from phi and math-specific work is real —
        verifiable correctness filtering is a meaningful tool, and didactic generation quality
        measurably helps small models. The evidence for general-domain synthetic pre-training
        at the frontier scale is thinner. The labs running the largest training runs have not
        published enough detail to know how heavily their data pipelines lean on synthesis,
        or what tradeoffs they've encountered. Watch this space.
      </Prose>

      <Prose>
        This closes the Pre-Training section. Post-training — how base models become useful
        assistants, how instruction-following is instilled, and how reinforcement learning
        shapes the final product — comes next.
      </Prose>
    </div>
  ),
};

export default syntheticDataPretraining;
