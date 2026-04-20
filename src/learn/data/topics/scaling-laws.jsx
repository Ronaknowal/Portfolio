import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { Plot } from "../../components/viz";

const scalingLaws = {
  title: "Scaling Laws (Kaplan, Chinchilla, Beyond)",
  readTime: "18 min",
  content: () => (
    <div>
      <Prose>
        The single most consequential empirical observation in modern AI is also the most boring-sounding one. If you scale up the number of parameters <Code>N</Code>, the number of training tokens <Code>D</Code>, and the amount of compute <Code>C</Code> in roughly the right proportions, cross-entropy loss falls as a smooth power law. Not a noisy trend. Not a shape that only becomes visible in hindsight. A genuine straight line on a log-log plot, stable across six or seven orders of magnitude, predictable well enough that you can forecast a model's loss before you have built the model. That regularity is the reason a frontier lab can commit nine-figure training budgets to a run that does not yet exist and expect the result to land within a tight margin of where the extrapolation said it would.
      </Prose>

      <Prose>
        This is why the industry can say with a straight face that GPT-5 will probably be better than GPT-4 before anyone has trained it, why training-run budgets are denominated in exaflop-days instead of guesses, and why "scale" went from a research hypothesis to an engineering commodity in roughly three years. The rest of this topic is about what the curves actually say, where the early ones were wrong, where the later ones get applied carelessly, and — most usefully — what scaling laws do not predict, which is most of what anyone actually cares about once the loss number comes in.
      </Prose>

      <H2>The Kaplan curves (2020)</H2>

      <Prose>
        The first serious attempt to write these regularities down came from OpenAI. Kaplan, McCandlish, and collaborators trained hundreds of small transformer language models across a wide range of sizes, dataset budgets, and compute budgets, and plotted cross-entropy loss against each of the three axes. What came back was shocking in how clean it was. Fix two of the three variables and the third traces a near-perfect power law across more than six orders of magnitude. Fit a straight line on log-log axes and the residuals are tiny. The loss does not just tend to improve with scale; it improves with a specific, predictable exponent that barely changes as you move around the regime.
      </Prose>

      <MathBlock>{"L(N) \\approx \\left(\\frac{N_c}{N}\\right)^{\\alpha_N}"}</MathBlock>

      <Prose>
        Three findings came out of that paper and each one shaped the next three years of the field. First, loss scales as a power law in each of <Code>N</Code>, <Code>D</Code>, and <Code>C</Code> with exponents somewhere in the 0.05 to 0.10 range — small numbers, but small numbers applied to many orders of magnitude add up to the difference between a model that can barely form sentences and one that can pass the bar exam. Second, the curves are so smooth that extrapolation works: fit on models spanning five orders of magnitude and the prediction for the sixth comes in almost exactly where the fit said it would. Third — and this is the finding that aged poorly — Kaplan concluded that for a fixed compute budget, you should spend most of it on a larger model and proportionally fewer tokens. The optimal tokens-per-parameter ratio fell out of their fits at roughly 1.7. GPT-3, designed in that paradigm, ended up at 175 billion parameters and about 300 billion tokens. A ratio close to 1.7. Kaplan's recommendation, followed to the letter.
      </Prose>

      <Plot
        label="cross-entropy loss vs. training compute (log-log, illustrative)"
        width={520}
        height={260}
        xLabel="log10 compute (petaflop·s·days)"
        yLabel="cross-entropy loss"
        series={[
          { name: "power-law fit", points: [[-1, 4.2], [0, 3.6], [1, 3.1], [2, 2.7], [3, 2.4], [4, 2.15], [5, 1.95], [6, 1.8]] },
        ]}
      />

      <Prose>
        The plot above is the shape that mattered. Every order of magnitude in compute buys roughly a fixed subtraction from the loss — not a fixed fraction, a fixed absolute amount, because power laws on log-log axes are straight lines. Double the compute, shave off a predictable delta. Ten times the compute, ten times the delta. There is no sharp elbow, no saturation, no point where the line bends and scaling stops paying. That absence is the astonishing part. Every other engineering discipline has diminishing returns that kick in visibly somewhere; in language model scaling, the diminishing returns are themselves a power law, which is to say they are the same diminishing returns at every scale.
      </Prose>

      <H2>Chinchilla (2022) — the correction</H2>

      <Prose>
        Two years later, DeepMind came back to the same question with a better experimental protocol. Hoffmann, Borgeaud, Mensch, and collaborators — the paper is usually just called "Chinchilla" after the model it produced — argued that the Kaplan analysis had confounded a subtle variable. Kaplan had varied model size while holding training steps roughly constant, so larger models got proportionally fewer gradient updates per parameter. That made the larger models look better than they should have, because the smaller models in the comparison were being under-trained relative to their capacity. Chinchilla redid the sweep with a cleaner design: for each target compute budget, train many models at different <Code>(N, D)</Code> splits and find the combination that minimizes loss. No assumption about which axis matters more. Let the data say.
      </Prose>

      <Prose>
        The data said something different. For a given compute budget <Code>C ≈ 6 · N · D</Code>, the optimum is not "as many parameters as possible." It is a balanced scaling where <Code>N</Code> and <Code>D</Code> should grow in roughly equal proportion — concretely, about twenty tokens of training data per parameter. GPT-3, with its 1.7 tokens per parameter, was therefore not merely suboptimal but dramatically under-trained. The same compute used to train GPT-3 could have produced a much smaller model trained on far more tokens, and it would have reached a lower loss. Chinchilla-70B, trained on 1.4 trillion tokens, was the proof: at the same compute budget as Gopher-280B it came out smaller, faster, cheaper to serve, and straightforwardly better on every benchmark the two models shared.
      </Prose>

      <MathBlock>{"L(N, D) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}"}</MathBlock>

      <Prose>
        The functional form Chinchilla fit is worth sitting with for a moment. The loss decomposes into three pieces. <Code>E</Code> is an irreducible floor — the entropy of natural language itself, the loss you would get from a perfect model, the thing you cannot beat no matter how much you scale. The <Code>A/N^α</Code> term is the penalty for having too few parameters to represent the distribution. The <Code>B/D^β</Code> term is the penalty for not having seen enough examples to estimate it. The two penalty terms have comparable exponents in the fit, which is the mathematical reason the optimum is balanced: if one term dominated you could trade the other axis for free, but both terms matter at roughly the same rate, so the compute-optimal frontier puts <Code>N</Code> and <Code>D</Code> on roughly equal footing.
      </Prose>

      <Plot
        label="tokens-per-parameter ratio (Kaplan ≈ 1.7, Chinchilla ≈ 20)"
        width={520}
        height={240}
        xLabel="model size (B params)"
        yLabel="optimal training tokens (B)"
        series={[
          { name: "Kaplan 2020 (N-heavy)", points: [[1, 1.7], [10, 17], [70, 119], [280, 476], [1000, 1700]] },
          { name: "Chinchilla 2022 (D-rich)", points: [[1, 20], [10, 200], [70, 1400], [280, 5600], [1000, 20000]] },
        ]}
      />

      <Prose>
        The two frontiers diverge by more than an order of magnitude at the top end. A trillion-parameter model, under Kaplan's recommendation, would be trained on 1.7 trillion tokens; under Chinchilla's, on 20 trillion. That is not a rounding difference. It is the difference between a compute budget spent mostly on parameters and a compute budget spent mostly on data. The practical consequence in 2022 and 2023 was that nearly every major model released after Chinchilla shifted visibly toward the <Code>D</Code>-heavy side. PaLM was trained on more data than Gopher. Llama 1's 7B and 13B were trained on a trillion tokens, well past what Kaplan would have recommended. Llama 2 pushed further. Llama 3's 8B trained on 15 trillion tokens — roughly <em>75 times</em> the Chinchilla optimum — and that is not a mistake. It is the next correction.
      </Prose>

      <H2>Beyond Chinchilla — over-training for inference</H2>

      <Prose>
        Chinchilla's optimum answers a specific question: given a training compute budget, what <Code>(N, D)</Code> minimizes final loss? That question is the right one exactly when training compute is the only cost that matters. For a research paper producing one checkpoint that will be evaluated and shelved, it is. For a frontier lab about to deploy a model to hundreds of millions of users and run inference on it billions of times a day, it is not even close to the right question. Inference cost scales with model size: every additional parameter costs a floating-point multiply on every generated token, forever. Training cost is a one-time expense amortized across the entire lifetime of the deployed model. If you are going to serve enough queries, the economics invert.
      </Prose>

      <Prose>
        The calculation becomes: which model, after training plus expected lifetime inference, minimizes total FLOPs? That objective almost always favors smaller-than-Chinchilla models trained on more-than-Chinchilla data. A 7B model trained on 2 trillion tokens costs more FLOPs at training time than Chinchilla would recommend for 7B, but every served token is cheaper than it would be for a 13B model. Once you serve more than a few billion tokens — a reasonable estimate for a widely deployed model — the cumulative inference savings exceed the extra training cost, and the smaller model wins on total compute. Llama 2's 7B on 2T tokens, Llama 3's 8B on 15T, Mistral 7B, Qwen 2.5's small models, DeepSeek-V3's efficiency-oriented runs — all explicitly over-trained relative to Chinchilla, all justified by this same logic.
      </Prose>

      <Callout accent="gold">
        Chinchilla optimizes training compute. Frontier labs optimize the sum of training and expected inference compute — which almost always favors smaller, longer-trained models.
      </Callout>

      <Prose>
        The secondary effect is even more interesting than the primary one. Over-training also seems to continue lowering loss long after Chinchilla would predict diminishing returns. The <Code>B/D^β</Code> term in the Chinchilla fit keeps paying out — slowly, with an exponent that makes each additional epoch matter less than the last, but without a visible floor in the regime that has been measured. Llama 3's 8B at 15T tokens is still improving meaningfully on quality metrics relative to 8B at 2T. The practical upper bound on how far you can over-train is now set less by scaling laws and more by the finite supply of high-quality training tokens — which is its own problem, and the next correction the field is grappling with.
      </Prose>

      <H2>Emergent abilities — the controversy</H2>

      <Prose>
        Scaling laws describe loss. Users care about capabilities. The relationship between those two things is the messiest part of the whole story. In 2022, Wei and collaborators published "Emergent Abilities of Large Language Models," documenting dozens of tasks — multi-digit arithmetic, word unscrambling, chain-of-thought reasoning on complex problems — where performance stays at near-random levels as model size increases until, at some threshold scale, it jumps sharply to near-human or beyond. The claim was that these capabilities are genuinely discontinuous with scale: not present in small models, present in large ones, with little warning in between. If true, this would be a problem for planning. Loss curves would tell you the model was getting better; capability curves would tell you nothing until a capability suddenly appeared.
      </Prose>

      <Prose>
        Schaeffer, Miranda, and Koyejo pushed back in 2023 with "Are Emergent Abilities of Large Language Models a Mirage?" Their argument: the shape of the emergence curve depends heavily on the metric. If you measure exact-match accuracy on multi-digit arithmetic — correct if every digit is right, wrong otherwise — you get a step function, because small models get most digits wrong and therefore score zero until the point where they get almost all digits right. But if you measure the log-likelihood the model assigns to the correct answer, or a per-digit accuracy, the underlying improvement is smooth and predictable from scaling laws. The discontinuity is in the metric, not in the model. Change the yardstick and the cliff becomes a slope.
      </Prose>

      <Prose>
        The honest synthesis is that both groups are partly right. The smooth scaling law on cross-entropy loss holds. Continuous proxies for capability — log-likelihoods, BLEU, perplexity-weighted accuracy — track that loss smoothly. But the capabilities users actually encounter are discrete: either the model can solve the problem or it cannot, either the code compiles or it does not, either the answer is factually correct or it is not. Those binary outcomes will look like emergence whether or not the underlying competence is emerging, because binary thresholding turns any smooth improvement into a step function somewhere. For planning purposes this matters: you can predict loss from scaling laws with high confidence, and you can predict the aggregate capability frontier with medium confidence, but you cannot reliably predict when a specific discrete capability will cross its specific threshold. That last part remains an empirical fact you only learn by running the model.
      </Prose>

      <H3>Data-constrained scaling</H3>

      <Prose>
        A quieter strand of recent work asks what happens when the data side of the Chinchilla frontier runs out. Muennighoff and collaborators in 2023, "Scaling Data-Constrained Language Models," measured what happens when you have less unique data than Chinchilla would want for your compute budget. The answer is that you can repeat data, but with diminishing returns — roughly up to four epochs of a high-quality corpus gives you most of the value of having four times as much unique data, after which repetition stops helping and eventually hurts. Beyond that, adding more parameters without adding more unique tokens flattens out sharply. The <Code>B/D^β</Code> term in the Chinchilla fit only pays out if <Code>D</Code> is genuinely new data, and "genuinely new" is a finite resource. Estimates of the total available stock of high-quality English text on the public web sit somewhere between 10 and 50 trillion tokens. Frontier training runs in 2024 were already consuming 15T. The arithmetic of the data wall is not subtle.
      </Prose>

      <Prose>
        The three current responses to the data wall are all bets rather than settled engineering. Synthetic data — generate training text with a stronger model and train a smaller one on it — is working better than skeptics expected, especially for reasoning and code; it shows up again in its own topic. Multimodal data — pull signal from images, video, audio, and code execution traces rather than only from written text — expands the effective corpus by orders of magnitude but at the cost of a much messier training signal. And quality curation — aggressive filtering of the existing corpus to extract a smaller but denser distillate — turns out to matter more than anyone expected five years ago, which is part of why data-quality topics live in this track at all.
      </Prose>

      <H3>The practical calculator</H3>

      <Prose>
        What does any of this look like when you are actually planning a training run? The back-of-envelope version is almost insultingly simple. Two approximations do most of the work. First, training FLOPs for a transformer are well approximated by <Code>6 · N · D</Code>, where the factor of six absorbs the cost of a forward pass plus backward pass on every parameter for every token. Second, Chinchilla-optimal wants <Code>D ≈ 20 · N</Code>. Combine those two and you can plug in a compute budget and read out the optimal model size and dataset size directly.
      </Prose>

      <CodeBlock language="python">
{`# Chinchilla compute-optimal sizing.
# Approximation: FLOPs ≈ 6 * N * D, optimal D ≈ 20 * N.

def chinchilla_optimal(flops_budget):
    """Returns (params, tokens) that minimize loss for the given FLOPs."""
    # From FLOPs = 6 * N * D and D = 20 * N: FLOPs = 120 * N^2
    N = (flops_budget / 120) ** 0.5
    D = 20 * N
    return N, D

# GPT-3 was trained on ~3.1e23 FLOPs.
# Chinchilla-optimal for that budget:
N, D = chinchilla_optimal(3.1e23)
print(f"Params: {N/1e9:.1f}B   Tokens: {D/1e9:.0f}B")
# -> Params: 50.8B   Tokens: 1016B
# GPT-3 was 175B on 300B — Chinchilla says it was ~3x too large and ~3x under-trained.`}
      </CodeBlock>

      <Prose>
        Ten lines of arithmetic recapitulate the entire 2022 correction. GPT-3's actual configuration — 175B parameters on 300B tokens — falls on the Kaplan frontier, not the Chinchilla one. The compute-optimal rebalancing at that same FLOPs budget would have produced a roughly 50B model on roughly 1T tokens. The gap between what was built and what the later math says should have been built is exactly the 3x-too-large, 3x-too-few-tokens answer Hoffmann et al. published. That is the value of a closed-form scaling law: it lets you grade old training runs against the frontier without running them again, and it lets you budget new ones without guessing.
      </Prose>

      <Prose>
        For an inference-aware lab, the calculation extends one more step. If you expect to serve <Code>T</Code> tokens in the model's lifetime, total compute is roughly <Code>6 · N · D</Code> for training plus <Code>2 · N · T</Code> for inference (each served token costs about <Code>2 · N</Code> FLOPs in a standard forward pass). Minimize the sum over <Code>(N, D)</Code> subject to an expected loss target rather than a training-FLOPs target, and the optimum migrates visibly to smaller <Code>N</Code> and larger <Code>D</Code> as <Code>T</Code> grows. That is the mathematical form of the "over-train for inference" intuition, and it is the form most frontier deployment decisions are actually being made against in 2024 and 2025, even when the paper does not say so.
      </Prose>

      <H2>What scaling laws DON'T predict</H2>

      <Prose>
        This is where the honesty is due. Scaling laws are a theory of loss, and loss is only loosely coupled to the things anyone actually wants from a language model. The loss curve tells you that the model will be more calibrated, more coherent, more likely to produce high-probability continuations of training-like text. It does not tell you whether the model will refuse a harmful request, whether it will generalize to a domain that was under-represented in pretraining, whether it will behave sensibly under distribution shift, whether it will hallucinate a plausible-sounding but fabricated fact, whether it will reason correctly through a problem that demands actual working memory, whether its answers will be useful to a human user who is not an expert. Every one of those properties has shown up as a weak function of scale or no function of scale at all, and every one of them is what the model is ultimately judged on.
      </Prose>

      <Prose>
        Scaling solves the "loss goes down" problem cleanly and predictably. It does not solve alignment, it does not solve reasoning, it does not solve truthfulness, and it does not solve the gap between a model that produces probable text and a model that produces helpful text. The next section of this track — post-training — is where those problems actually get attacked, with tools like supervised fine-tuning, RLHF, constitutional AI, and the whole apparatus of making a pretrained base model behave like something a human wants to talk to. Scaling is necessary; the rest of the track is about why scaling is also conspicuously not sufficient, and what the field has had to invent to cover the difference.
      </Prose>

      <Prose>
        The scaling laws are boring in the best possible way: reliable, empirical, quantitative, and the kind of thing you can show to a CFO to justify a nine-figure line item. They turned "build a frontier AI" from a research question with uncertain odds into an engineering question with a budget, a timeline, and a forecast. But engineering questions depend on their inputs, and the input to scaling is every choice made about the data, the architecture, the training schedule, the numerical precision, the curriculum, and the objective. Which is why the next several topics — curriculum and data ordering, FP8 training, mixture-of-experts, multimodal pretraining, data-quality filtering, and synthetic data — are all, at the bottom, about one thing: how to get more out of every FLOP that the scaling laws say you are allowed to spend.
      </Prose>
    </div>
  ),
};

export default scalingLaws;
