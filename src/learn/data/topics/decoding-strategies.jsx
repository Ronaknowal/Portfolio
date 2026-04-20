import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream } from "../../components/viz";

const decodingStrategies = {
  title: "Decoding Strategies (Greedy, Beam, Top-k, Top-p, Temperature)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        At inference time, a language model outputs a distribution over the vocabulary at each step. How you pick the next token from that distribution is the decoding strategy, and the choice matters more than most practitioners realize. The same model can produce precise technical answers or creative flowing text depending on the decoding knobs alone. Change the sampling configuration without touching a single weight and you have, in effect, a different product. Understanding what each knob does — and why the defaults are what they are — is one of the more immediately practical things you can take away from reading about language models.
      </Prose>

      <Prose>
        The model's output at each position is a vector of logits, one per vocabulary entry. A softmax converts those logits into a proper probability distribution. Everything downstream of that softmax is the decoding strategy's domain. The question is: given this distribution, which token do we commit to next?
      </Prose>

      <H2>Greedy decoding</H2>

      <Prose>
        The simplest possible answer is to always pick the highest-probability token. Greedy decoding does exactly this — at each step, take the <Code>argmax</Code> of the logit vector and emit that token. The result is deterministic: given the same prompt and the same model, you always get the same output. That predictability is valuable in certain settings.
      </Prose>

      <Prose>
        Greedy works well when correctness is narrow. Translation into a target language has a relatively small number of acceptable outputs. Structured extraction — pull the date from this invoice, classify this review — has an even smaller acceptable set. Classification and summarization with tight constraints tend to do fine greedy. The model's most probable token is usually close to what you want, and the feedback loop between one forced-correct token and the next prediction is relatively forgiving.
      </Prose>

      <Prose>
        For open-ended generation, greedy fails in a recognizable way. Local maxima compound. Each step picks what looks best given the current context, but committing to a safe, common word at step five narrows the distributional space at step six, which narrows it further at step seven. The result is text that drifts toward the average — repetitive phrases, the same sentence structures, the same hedging vocabulary. Greedy decoders frequently fall into loops, regenerating the same phrase every few tokens until the context window fills, because each repetition makes the repeated phrase slightly more probable in the subsequent position.
      </Prose>

      <CodeBlock language="python">
{`def greedy_decode(model, prompt, max_tokens=100):
    tokens = prompt[:]
    for _ in range(max_tokens):
        logits = model(tokens)[:, -1, :]
        next_token = logits.argmax(dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
        if next_token.item() == EOS_TOKEN: break
    return tokens`}
      </CodeBlock>

      <H2>Beam search</H2>

      <Prose>
        Beam search generalizes greedy by keeping multiple candidate sequences in parallel. Instead of committing to the single best token at each step, you maintain <Code>k</Code> partial sequences — the beams — and extend every one of them by their top candidates, scoring each extension by cumulative log-probability. After extending, you prune back to the top <Code>k</Code> sequences and repeat. When decoding terminates (usually at an end-of-sequence token or a maximum length), you return the beam with the highest total probability.
      </Prose>

      <Prose>
        The practical effect of beam search is that it can recover from locally bad choices. A greedy decoder that picks a mediocre token at step three is stuck with it. A beam search with width five keeps that mediocre branch alive only if it stays competitive; if a different branch that took a lower-probability token at step three ends up scoring much higher by step ten, the better branch wins. This makes beam search attractive in constrained settings — neural machine translation, speech recognition, structured output generation — where a global optimum matters more than diversity.
      </Prose>

      <Prose>
        Why did beam search fall out of favor for modern generative models? Optimizing for highest-cumulative-probability full sequence is not the same as generating interesting text. Beam search is biased toward short, common, safe continuations. It actively penalizes diversity: any beam that takes a creative turn early is competing against beams that stayed on the statistical highway. The resulting output often reads as generic, repetitive, and bland — a failure mode sometimes called the beam search curse. For chat, creative writing, and any application where engagement matters, these properties are actively harmful. Beam search remains the dominant strategy in constrained decoding, retrieval reranking, and agentic settings where a verifier can judge multiple candidates, but for open-ended text, sampling methods have largely replaced it.
      </Prose>

      <TokenStream
        label="beam search — 3 beams after 4 steps"
        tokens={[
          { label: "beam 1:", color: "#555" },
          { label: " The cat sat", color: "#e2b55a" },
          { label: " on the mat", color: "#e2b55a" },
          { label: "beam 2:", color: "#555" },
          { label: " The cat sat", color: "#4ade80" },
          { label: " quietly", color: "#4ade80" },
          { label: "beam 3:", color: "#555" },
          { label: " The cat was", color: "#c084fc" },
          { label: " sleeping", color: "#c084fc" },
        ]}
      />

      <H2>Temperature</H2>

      <Prose>
        Temperature is the single most-used decoding knob. It controls how peaked or flat the probability distribution is before sampling, by scaling the logits before the softmax is applied. A temperature below one sharpens the distribution — high-probability tokens get relatively more probability, low-probability tokens get relatively less. A temperature above one flattens it — the model becomes less sure of itself, and lower-ranked tokens can get sampled more easily.
      </Prose>

      <MathBlock>{"p_T(x_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}"}</MathBlock>

      <Prose>
        At <Code>T = 0</Code>, the softmax becomes an argmax and you recover greedy decoding. At <Code>T = 1</Code>, the logits pass through unchanged — this is the model's actual training distribution, the one it was calibrated against during training. At <Code>T = 0.7</Code>, the most common default for chat APIs, you get a distribution that is sharper than training but still admits sampling from several plausible continuations. Above <Code>T = 1</Code>, the distribution blurs toward uniform, and the model starts generating tokens that are improbable by its own measure.
      </Prose>

      <Prose>
        Practical guidance on temperature is surprisingly stable across tasks. Math, code, and precise extraction: stay near zero. Factual chat and summarization: <Code>0.7</Code> is a reasonable prior. Creative writing: <Code>1.0</Code> or slightly above if the model's calibration holds up there. Temperatures above <Code>1.2</Code> are rarely useful outside highly constrained experiments — the quality degradation from sampling deep into the tail usually outweighs any diversity gains. Temperature alone, however, does not prevent the tail problem. Even at <Code>T = 0.7</Code>, a vocabulary of 100,000 tokens has tens of thousands of entries with non-negligible probability after rescaling. Temperature sharpens the distribution; it does not truncate it. That is what top-k and top-p are for.
      </Prose>

      <H2>Top-k sampling</H2>

      <Prose>
        Top-k sampling keeps only the <Code>k</Code> highest-probability tokens, zeroes out the rest, renormalizes to sum to one, and then samples from that truncated set. With <Code>k = 40</Code> or <Code>k = 50</Code>, the vast majority of implausible tokens are simply excluded from consideration. The model can no longer accidentally emit a completely out-of-distribution token because the token was never in the candidate set to begin with.
      </Prose>

      <Prose>
        Top-k is simple to implement, easy to reason about, and effective at cutting the long tail. Its drawback is that a fixed <Code>k</Code> is wrong in both directions depending on the context. Some positions in a sequence have only three or four plausible next tokens — the article that follows this noun phrase, the closing bracket that ends this code expression. Setting <Code>k = 40</Code> in those positions means sampling from thirty-six tokens that have no business being there. Other positions have genuine distributional spread — the next word of a creative sentence could be any of a hundred reasonable options. Setting <Code>k = 40</Code> there truncates unnecessarily, making the output more predictable than the model's actual uncertainty warrants. Because the right <Code>k</Code> varies step by step with the context, a fixed constant is always a compromise.
      </Prose>

      <CodeBlock language="python">
{`def top_k_sample(logits, k=40, temperature=0.7):
    logits = logits / temperature
    topk_values, topk_indices = torch.topk(logits, k)
    probs = F.softmax(topk_values, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return topk_indices.gather(-1, sampled)`}
      </CodeBlock>

      <H2>Top-p (nucleus) sampling</H2>

      <Prose>
        Top-p sampling, introduced by Holtzman and colleagues in 2019 under the name nucleus sampling, solves top-k's fixed-threshold problem by making the cutoff adaptive. Instead of keeping a fixed number of tokens, keep the smallest set of tokens whose cumulative probability mass reaches the threshold <Code>p</Code>. Sort the vocabulary by probability in descending order, walk down the list, and stop when the running sum crosses <Code>p</Code>. The tokens in that prefix are the nucleus; everything else is excluded.
      </Prose>

      <Prose>
        The key property: the nucleus shrinks and grows with context. When the model is highly confident — say, generating a closing parenthesis in a syntactically forced position — the top one or two tokens might already account for 0.95 of the probability mass, so the nucleus contains one or two tokens. When the model is genuinely uncertain — generating the third word of a creative story with dozens of plausible continuations — many tokens contribute to the nucleus, and sampling is proportionally more exploratory. Top-p adapts to the model's own uncertainty rather than imposing an external fixed width. In practice, <Code>p = 0.9</Code> or <Code>p = 0.95</Code> covers the vast majority of useful cases, and top-p has largely replaced top-k as the primary truncation mechanism in production systems.
      </Prose>

      <CodeBlock language="python">
{`def top_p_sample(logits, p=0.9, temperature=0.7):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens until cumulative probability reaches p
    keep = cumsum <= p
    keep[..., 0] = True  # always keep the most-probable token
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, sampled)`}
      </CodeBlock>

      <H2>Combining knobs in practice</H2>

      <Prose>
        Real inference systems stack multiple filters. The typical order: temperature scaling first, then top-k (acting as a hard floor against catastrophically unlikely tokens), then top-p (the main adaptive mechanism), then sample. The order matters — applying top-p on un-scaled logits gives different results than applying it on temperature-scaled ones, because the rescaling shifts relative probability mass before the cumulative sum is computed.
      </Prose>

      <Prose>
        Additional knobs exist at the margins. Min-p is a recent variant where a token must exceed <Code>p × max_probability</Code> to be kept — the threshold scales with the peak of the distribution, which makes it more robust at low temperatures where absolute probabilities of non-peak tokens collapse. Repetition penalties subtract from (or divide into) logits for tokens already present in the context, directly targeting the looping behavior that plagues greedy and low-temperature outputs. Frequency penalties go further, scaling the penalty by occurrence count rather than just presence.
      </Prose>

      <H2>Task-dependent defaults</H2>

      <Prose>
        There is no universal best configuration. The right decoding parameters depend on what the model is being asked to do, and the gap between the right and wrong configuration is large enough to matter in production.
      </Prose>

      <Prose>
        Mathematical reasoning and code generation reward deterministic or near-deterministic sampling. A proof step is either valid or it is not; sampling from the tail introduces errors that downstream steps amplify. Temperature near zero — greedy or <Code>T = 0.1</Code> — is standard. Structured extraction and classification tasks follow the same logic. Precise factual retrieval and question answering generally benefit from low temperature as well, though not quite as extreme.
      </Prose>

      <Prose>
        General conversational chat sits in a moderate regime. <Code>T = 0.7</Code> with <Code>top_p = 0.9</Code> is the de facto default across most major chat APIs — varied enough to feel natural, not loose enough to incohere. Creative writing shifts upward: <Code>T = 1.0</Code> or slightly above with <Code>top_p = 0.95</Code>, sampling closer to the model's actual learned distribution at the cost of occasional drift. Most APIs now expose per-call overrides, because the gap between optimal settings for code generation and story generation is too wide for any single system default to bridge.
      </Prose>

      <H3>Structural decoding — the frontier</H3>

      <Prose>
        All of the strategies discussed so far are unconstrained — the model samples freely from the vocabulary, and the various filters operate purely on probability. When the required output must conform to a rigid format — valid JSON, well-formed SQL, strings that match a specific regular expression — probability-based filtering alone is insufficient. A model with <Code>top_p = 0.95</Code> can still produce a closing brace where an opening brace was required. Constrained decoding addresses this by directly masking the logits: at each position, any token that would lead to an invalid output according to the formal grammar is zeroed out before sampling. The model can only ever sample tokens that keep the partial output consistent with the required structure.
      </Prose>

      <Prose>
        The technique requires a parser that can report, given the tokens emitted so far, which tokens are valid next. For JSON this is straightforward; for regular expressions it reduces to walking a finite automaton in parallel with the decoding loop. Constrained decoding has moved from a research curiosity to a standard feature in several inference frameworks — Outlines, Guidance, and llama.cpp's grammar sampling among them. The next topic covers it in depth.
      </Prose>

      <Prose>
        Worth noting: beam search is making a quiet comeback in agentic settings. When a model generates candidate solutions and a verifier scores them, the ability to produce diverse candidates and select the best by an external criterion is exactly what beam search provides. The verifier compensates for pure probability scoring's quality-ordering limitation. This pattern appears in AlphaCode 2, OpenAI's o-series, and several open-source reasoning systems that have emerged since 2023.
      </Prose>

      <Callout accent="gold">
        Decoding is not a preprocessing detail — it's a last-mile control surface. The same model feels like a different product depending on how you sample.
      </Callout>

      <Prose>
        The next topic — constrained decoding — picks up where this one ends: what happens when you need the output to obey hard rules, not just follow a probability distribution.
      </Prose>
    </div>
  ),
};

export default decodingStrategies;
