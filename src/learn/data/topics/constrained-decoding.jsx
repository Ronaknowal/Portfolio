import { Prose, H2, H3, Code, CodeBlock, Callout } from "../../components/content";
import { TokenStream } from "../../components/viz";

const constrainedDecoding = {
  title: "Structured Output & Constrained Decoding (Outlines, XGrammar)",
  readTime: "11 min",
  content: () => (
    <div>
      <Prose>
        Most real LLM applications need outputs in a specific format — valid JSON matching a schema, SQL that parses, function calls that match tool signatures. Prompting the model to produce valid JSON and hoping for the best works roughly 95% of the time. That 5% failure rate breaks your production pipeline: a single malformed response that your parser cannot handle either crashes the request or triggers a retry loop, and both outcomes compound under load. Constrained decoding closes the gap to 100% by construction: at every token, a validator masks the logits so only tokens consistent with the grammar are sampleable. The model is physically incapable of producing output that violates the constraint.
      </Prose>

      <H2>The core trick — logit masking</H2>

      <Prose>
        Decoding is a loop. At each step the model produces a logit vector over the entire vocabulary — roughly 32,000 to 128,000 entries — and a sampling strategy (covered in the sibling topic on decoding strategies) draws the next token from the distribution those logits define. Constrained decoding inserts one operation into that loop: before sampling, ask a validator which tokens are legal given the partial output so far, then set the logits of every illegal token to negative infinity. After softmax, illegal tokens have probability zero. The sampler cannot pick them.
      </Prose>

      <CodeBlock language="python">
{`def constrained_decode(model, prompt, validator, max_tokens=200):
    """
    validator.allowed_tokens(partial_output) -> set of token ids that are legal next.
    """
    tokens = prompt[:]
    output = []
    for _ in range(max_tokens):
        logits = model(tokens)[:, -1, :]
        allowed = validator.allowed_tokens(output)
        mask = torch.full_like(logits, float("-inf"))
        mask[:, list(allowed)] = 0.0
        logits = logits + mask                         # illegal tokens -> -inf

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=-1)
        output.append(next_token.item())
        if validator.is_complete(output): break
    return output`}
      </CodeBlock>

      <Prose>
        The loop is deliberately generic. Nothing here is specific to JSON, SQL, or any particular grammar. The complexity lives entirely inside the validator object — specifically inside <Code>allowed_tokens</Code>, which must answer in microseconds for decoding to remain fast. Everything else is standard sampling, unchanged from unconstrained generation.
      </Prose>

      <H2>JSON schema constraints</H2>

      <Prose>
        JSON is the most common constrained-decoding target, and it is a good case to reason through concretely. Given a schema — field names, types, required fields, enum values — at every step in the output you know exactly what is expected next. After the opening <Code>{"{"}</Code>, a field name must follow. After the field name, a colon. After the colon, a value of the declared type. After the value, either a comma (if more fields remain) or a closing <Code>{"}"}</Code>. The grammar is deterministic given the schema and the partial output.
      </Prose>

      <Prose>
        Tools like Outlines, XGrammar, and LMQL compile the JSON schema into a finite-state machine. Each state in the machine corresponds to a position in the partially-built output and carries a list of the token prefixes that are legal next. Transitioning between states is cheap — you update a single pointer as each token is produced. The expensive work happens once, offline, at compile time: building the FSM from the schema and intersecting it with the model's vocabulary to produce a bitmask for each state. At decode time you look up the bitmask for the current state and apply it. The per-step overhead is a single memory read and a bitwise OR.
      </Prose>

      <H3>Outlines — the reference implementation</H3>

      <Prose>
        Outlines, released by dottxt-AI in 2023, was the first production-quality constrained decoder for transformer LLMs. You pass it a regex, a JSON schema, or a Pydantic model class; it compiles the constraint into a masking function and wraps the generation loop. The interface is deliberately thin — you write the schema, the library handles everything else.
      </Prose>

      <CodeBlock language="python">
{`from outlines import models, generate
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    rating: float

model = models.transformers("meta-llama/Llama-3-8B-Instruct")
generator = generate.json(model, Movie)

# Guaranteed to return a valid Movie instance — not a string that might parse.
result: Movie = generator("Recommend me a movie from the 2010s.")`}
      </CodeBlock>

      <Prose>
        The return value is a typed Python object, not a raw string. You do not call <Code>json.loads</Code>, you do not catch <Code>JSONDecodeError</Code>, you do not write a retry loop. The guarantee is structural: Outlines compiles the Pydantic model into an FSM at import time, and the decoding loop can only produce token sequences that the FSM accepts. If the FSM accepts them, the JSON is valid, and Pydantic can construct the object without error.
      </Prose>

      <H3>XGrammar — the efficient compiler</H3>

      <Prose>
        Outlines's original implementation had real overhead at decode time. Computing the allowed token set for each step in Python, against a vocabulary of 32,000+ entries, could take several milliseconds — enough to slow end-to-end decoding by 2 to 5 times on complex grammars. For production serving where latency and throughput are real constraints, that overhead is unacceptable.
      </Prose>

      <Prose>
        XGrammar, released in 2024 by the MLC group, reformulated the validator as a context-free grammar compiler written in C++ with aggressive caching at both the grammar and token levels. The key insight: most of the FSM transitions for a given schema are visited repeatedly across different requests, and the vocabulary intersection — which tokens match which FSM state — can be precomputed and cached. At decode time, the per-step overhead drops to microseconds rather than milliseconds. Modern serving stacks — vLLM, SGLang, and llama.cpp — have integrated XGrammar or equivalent structured-output backends precisely because the overhead is now low enough to enable by default without a throughput penalty.
      </Prose>

      <H2>The tokenization problem</H2>

      <Prose>
        There is a subtle mismatch that catches every new implementation of constrained decoding: grammars are defined over characters, but models emit tokens that span multiple characters. A single token might be <Code>{`", "rating":`}</Code> — a comma, space, opening quote, the string "rating", closing quote, colon, all in one vocabulary entry. The constraint validator needs to reason about whether a multi-character token is consistent with the grammar at the current position, which requires looking ahead across the token's entire character span.
      </Prose>

      <TokenStream
        label="character-level grammar vs. token-level decoding"
        tokens={[
          { label: '{"', color: "#c084fc" },
          { label: "title", color: "#c084fc" },
          { label: '":', color: "#c084fc" },
          { label: ' "', color: "#c084fc" },
          { label: "Inception", color: "#e2b55a" },
          { label: '", "', color: "#c084fc" },
          { label: "year", color: "#c084fc" },
          { label: '": ', color: "#c084fc" },
          { label: "2010", color: "#e2b55a" },
          { label: "}", color: "#c084fc" },
        ]}
      />

      <Prose>
        The purple tokens are structural — their content is fully determined by the grammar, given the schema and position. The gold tokens are content — the model's actual creative contribution, free to vary within the type constraint. A good constrained decoder recognizes the structural tokens and handles them without touching the logit computation: when only one token can legally follow, you can emit it directly and skip the forward pass entirely. This short-circuiting is called "forced token" optimization, and it meaningfully reduces the number of model calls for highly structured outputs.
      </Prose>

      <H2>What constrained decoding cannot fix</H2>

      <Prose>
        Constrained decoding guarantees the output is syntactically valid. The JSON parses. The SQL has the right clause structure. The function call matches the tool signature. What it does not guarantee — and cannot, in principle — is that the output is semantically correct. The JSON's field values can still be hallucinated. The SQL's table references can still be wrong. The function call's arguments can still be nonsensical. Constraints narrow the distribution over token sequences to those that are structurally legal. They do not alter what the model believes, what it has learned, or how accurately it maps a question to an answer.
      </Prose>

      <Callout accent="gold">
        Constrained decoding fixes parseable-output problems. It does not fix correctness problems. The model can still lie — in perfectly valid JSON.
      </Callout>

      <Prose>
        This distinction matters practically. If your pipeline was failing because of parse errors, constrained decoding solves the problem completely. If your pipeline was failing because the model was producing plausible-looking but wrong field values — wrong dates, hallucinated names, mismatched IDs — constrained decoding changes nothing. Diagnosing which failure you have is the prerequisite to knowing whether this technique applies.
      </Prose>

      <H2>Performance considerations</H2>

      <Prose>
        With an efficient compiler like XGrammar or SGLang's structured output backend, grammar-aware decoding adds less than 5% overhead for typical JSON schemas. The forced-token optimization helps further: structural tokens are emitted for free, so the effective cost scales with the number of content tokens, not total output length. For highly structured grammars — programming languages, proof systems, formal specifications with deep nesting — overhead can be larger, but practical measurements on production workloads still come in well under 2× throughput reduction.
      </Prose>

      <Prose>
        The comparison point that matters is not "constrained decoding vs. unconstrained decoding." It is "constrained decoding vs. generate-and-retry on parse failure." In production, a 5% parse failure rate means roughly 1 in 20 requests triggers a second model call. At high request volume, the expected cost of the retry loop — including the latency of failed attempts, the additional model calls, and the occasional cascade where retries also fail — almost always exceeds the fixed overhead of running a constrained decoder on every request. Constrained decoding is cheaper in expectation as soon as your failure rate is non-negligible.
      </Prose>

      <Prose>
        There is also a tail-latency argument. Unconstrained generation with retries has a heavy tail: most requests succeed on the first try, but a small fraction cycle through multiple failures, accumulating latency. Constrained decoding gives a tight latency distribution with no tail from retries. For applications with latency SLAs, that predictability is often worth more than the average-case comparison.
      </Prose>

      <Prose>
        Constrained decoding is one of the few LLM techniques with a genuine "drop-in upgrade" shape: you modify the generation call, you get typed outputs, your error rate on structured-output tasks goes to zero. Lower error rate, no retry overhead, predictable latency. The next topic moves from what a single inference does to how multiple inferences share resources — continuous batching and PagedAttention.
      </Prose>
    </div>
  ),
};

export default constrainedDecoding;
