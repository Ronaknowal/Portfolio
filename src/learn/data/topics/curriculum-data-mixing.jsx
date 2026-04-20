import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { Plot, StepTrace, TokenStream } from "../../components/viz";

const curriculumDataMixing = {
  title: "Curriculum Learning & Data Mixing Strategies",
  readTime: "10 min",
  content: () => (
    <div>
      <Prose>
        Pre-training corpora aren't uniform. They are mixtures: web crawl, books, code, Wikipedia, research papers, math. The proportions aren't obvious, and neither is the order. Curriculum learning and data mixing ask two related questions — what should the model see, and in what order — and the answers turn out to matter more than most of the architectural decisions that get written up in papers.
      </Prose>

      <Prose>
        The question of what to train on is older than neural language models, but it has become sharper as models have grown large enough that you cannot iterate quickly. At GPT-3 scale, a bad corpus mix is a $4M mistake. At Llama 3 scale it is higher. The decisions get made once, baked into hundreds of billions of parameters, and lived with for years. Getting them right is less about insight than about systematic methods for searching a space that is too large to brute-force.
      </Prose>

      <H2>Data mixing — the under-discussed hyperparameter</H2>

      <Prose>
        Every major pre-training corpus is a weighted combination of sources. The Pile, C4, RefinedWeb, RedPajama, FineWeb — they differ in filtering aggressiveness, deduplication strategy, and source selection, but all of them are fundamentally a mixture with explicit or implicit weights. Those weights matter enormously, and the optimal weights are not obvious from first principles.
      </Prose>

      <Prose>
        Too much code in the mix and natural language fluency degrades. Too little and the model cannot write a function. Too much low-quality web text dilutes the signal; too little and coverage of everyday language suffers. The clearest demonstration of this tradeoff across generations is the shift in code and math emphasis. Llama 2 trained with roughly 4.5% code in its corpus. Llama 3 pushed code and math to something closer to 17% combined — a jump that explains most of the measurable gap between the two on programming benchmarks. The architecture barely changed. The mixture did.
      </Prose>

      <Plot
        label="pre-training mix evolution (illustrative %)"
        width={520}
        height={240}
        xLabel="model generation"
        yLabel="% of corpus"
        series={[
          { name: "web", points: [[1, 75], [2, 65], [3, 55], [4, 50]] },
          { name: "code", points: [[1, 4], [2, 8], [3, 15], [4, 17]] },
          { name: "math", points: [[1, 2], [2, 4], [3, 8], [4, 10]] },
          { name: "books+papers", points: [[1, 15], [2, 17], [3, 15], [4, 13]] },
        ]}
      />

      <Prose>
        The trend is consistent across labs: web share contracts, code and math grow, books and papers hold roughly flat. The contraction of web isn't a bet against web text — it is a bet that the marginal token of filtered web prose is less valuable than a marginal token of structured reasoning. At the same time, web text never falls below half the corpus, because breadth of coverage in natural language still matters for everything else the model is expected to do.
      </Prose>

      <H2>Finding a good mix</H2>

      <Prose>
        Two methods have emerged as practical alternatives to intuition-guided manual tuning. The first is proxy-model grid search: train many small models on different mixes, evaluate them on held-out benchmarks, and extrapolate to the full-scale mix. The small models are cheap enough that you can explore tens or hundreds of configurations. The assumption is that relative performance at small scale predicts relative performance at large scale — which is approximately true for coarse mix differences, though it breaks down for subtle ones.
      </Prose>

      <Prose>
        The second family of methods — DoReMi, DoGE, IDoReMi — replaces grid search with a principled optimization. The core idea: train a small reference model on each domain separately, or on a uniform mix, and use the resulting per-domain losses to derive weights for the main training run. Domains where the reference model performs poorly are domains the main model has room to learn from; they get upweighted. Domains where the reference model is already near-optimal get downweighted. The procedure converges to a mix that minimizes regret across domains rather than just minimizing average loss.
      </Prose>

      <CodeBlock language="python">
{`# Simplified DoReMi-style: re-weight domains by a small reference model's per-domain loss.
def optimize_mix(ref_model, domains, target_losses, n_steps=1000):
    weights = {d: 1.0 / len(domains) for d in domains}
    for _ in range(n_steps):
        loss = {d: ref_model.loss(domains[d]) for d in domains}
        # Upweight domains the reference model is bad at (higher loss = more weight)
        regret = {d: max(0, loss[d] - target_losses[d]) for d in domains}
        total = sum(regret.values()) + 1e-8
        weights = {d: (w + regret[d] / total) * 0.5 for d, w in weights.items()}
    return weights`}
      </CodeBlock>

      <Prose>
        The target losses in the snippet encode the goal: how good do we want the model to be at each domain? Setting them from a reference model trained only on that domain approximates "as good as a specialist." Setting them uniformly lower than the reference loss says "beat the reference everywhere equally." The resulting weights are a data mixture, not a training recipe — you then sample from each domain according to those weights for the entire training run.
      </Prose>

      <Callout accent="gold">
        DoReMi (Xie et al. 2023) reported that its derived weights outperformed the baseline mix by an average of 6.9 perplexity points across domains, using only a 280M reference model to set weights for a 8B training run. The proxy cost was roughly 1% of the main run.
      </Callout>

      <H2>Curriculum learning — easy to hard</H2>

      <Prose>
        The original curriculum learning paper (Bengio et al. 2009) made a clean intuitive argument: humans and animals learn better when examples are ordered from simple to complex. A model that sees well-formed, unambiguous examples early builds better representations than one that immediately faces the full noise of a real corpus. The result should be faster convergence and better final quality.
      </Prose>

      <Prose>
        For LLM pre-training at scale, the empirical picture is more complicated. Random shuffling of a massive corpus — the default — works surprisingly well. The sheer scale of data exposure means the model encounters any given concept many times regardless of order, and the signal from "easy" versus "hard" is diluted by the law of large numbers. Several groups have tried staged curricula at scale and found negligible gains over randomized baselines, or gains that did not survive hyperparameter search.
      </Prose>

      <Prose>
        But curriculum consistently helps at two specific places. First, the very end of training: staged cooldowns on high-quality data give measurable benchmark gains, and the effect has been robust enough that it is now standard practice at several labs. Second, continued pre-training: when you take a base model and adapt it to a narrow domain — medicine, law, code — a curriculum that gradually shifts the mix toward the target domain performs better than an abrupt jump. The base model's existing representations are fragile; a gradual curriculum preserves them while building domain knowledge on top.
      </Prose>

      <H3>Staged cooldown</H3>

      <Prose>
        The three-stage schedule has become a visible pattern in recent model releases — Llama 3, Gemma 2, DeepSeek. The bulk of training runs at a constant high learning rate on the general mixture. The final ~5% of tokens trains at a decaying learning rate on a curated, higher-quality subset: deduplicated web, textbooks, long-form reasoning, math, code. Read as a curriculum: the first 95% of tokens build broad, general competence; the last 5% sharpen the model on the distribution it will be evaluated against.
      </Prose>

      <StepTrace
        label="three-stage pretraining schedule"
        steps={[
          { label: "1. warmup", render: () => (
            <TokenStream tokens={["0-2% tokens", " →", " LR linear warm-up", " →", " general mix"]} />
          ) },
          { label: "2. bulk", render: () => (
            <TokenStream tokens={["2-95% tokens", " →", " peak LR → cosine decay", " →", " general mix"]} />
          ) },
          { label: "3. cooldown on curated data", render: () => (
            <TokenStream tokens={["95-100% tokens", " →", " low LR", " →", " high-quality curated mix"]} />
          ) },
        ]}
      />

      <Prose>
        The cooldown's effectiveness depends on the quality delta between the general mix and the curated subset. If the curated data is only marginally better, the gain is marginal. If it is substantially cleaner — fewer near-duplicates, better source filtering, higher reasoning density — the effect is pronounced. Some reports show 3–5 point improvements on reasoning benchmarks from the cooldown alone, without any change to the model architecture or total token count. It is, in that sense, free: the budget was already committed, only the ordering changed.
      </Prose>

      <H3>Data-mixing laws</H3>

      <Prose>
        Recent work has pushed further toward making the mix a derivable quantity rather than a tuned one. Ye et al. (2024) and Liu et al. (2024) independently showed that loss under a mixed corpus can be approximated by a weighted combination of single-domain losses. The functional form is roughly additive: if you know how well a model trained on a given mixture performs on each domain in isolation, you can predict how it will perform under a different mixture without rerunning training.
      </Prose>

      <Prose>
        This is a scaling-law-style result for data mixing rather than compute or parameters. The practical implication: instead of running many small proxy experiments to search the mix space, you run one set of single-domain experiments and then solve analytically for the weights that minimize a target loss function. The approach has not yet displaced proxy-model search in production — the additive assumption breaks down for mixes that are far from any single-domain training run — but it narrows the search space substantially and provides a principled starting point.
      </Prose>

      <H2>Continued and domain-adaptive pretraining</H2>

      <Prose>
        After a base model is trained, you can extend pre-training on a narrower distribution to specialize it. Code Llama started from Llama 2 and continued training on 500B tokens of code. Meditron adapted Llama 2 on medical literature. BloombergGPT trained on a mix of general and financial text from scratch, but the design logic is the same: financial corpora receive a disproportionate share of the token budget to shift the model's distribution toward that domain.
      </Prose>

      <Prose>
        The central risk is catastrophic forgetting. A model that sees only domain data will rapidly overwrite the general capabilities it spent hundreds of billions of tokens acquiring. The question is how aggressively. The standard mitigation is replay: keep 10–20% of the training mix as general-domain data throughout the domain-adaptive run. This is enough to anchor the general distribution and prevent collapse, while still allowing the domain shift to happen. The exact replay fraction is sensitive to the domain gap — adapting to code, which is already well-represented in most base models, requires less replay than adapting to a narrow medical subfield.
      </Prose>

      <Prose>
        A subtler risk is representation collapse at the tail. Even with replay, continued pre-training can degrade performance on rare languages, obscure domains, or low-frequency tasks that appeared only occasionally in the original mix. The base model's knowledge of those areas was marginal to begin with; the domain-adaptive run dilutes it further. Evaluation suites that focus on the target domain will not catch this. You need a broad coverage benchmark run before and after the adaptation to see it.
      </Prose>

      <Prose>
        The mix isn't a glamorous part of the pre-training story. It doesn't have a neat algorithm at its center, it doesn't produce a clean loss curve to display, and the decisions that go into it are partly empirical and partly judgment calls about what domains matter. But ask any frontier lab's pre-training team what drives the year-over-year gains and "we got the mix and the curriculum right" sits near the top. The architecture memos are public. The corpus decisions mostly aren't.
      </Prose>
    </div>
  ),
};

export default curriculumDataMixing;
