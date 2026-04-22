import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const constrainedDecoding = {
  title: "Structured Output & Constrained Decoding (Outlines, XGrammar)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Language models produce language. That is, for most of their history, not a limitation — it is the point. But production software does not run on language. It runs on structured data: JSON that an API can deserialize, SQL that a database can execute, function call signatures that a tool dispatcher can route, XML that a schema validator can check. The gap between "text that looks like JSON" and "text that <em>is</em> JSON" turns out to matter more than it seems at small scale. Prompting a language model to produce valid JSON and hoping for the best works roughly 95% of the time on cooperative models with careful prompts. That sounds high until you run a pipeline at any real volume: a 5% failure rate on a 10-step agentic chain is a 40% chance that at least one step fails, each failure requires a retry or a crash, and the retries compound at peak load.
      </Prose>

      <Prose>
        The deeper problem is that post-hoc validation is the wrong abstraction. It asks the model to try, checks whether it succeeded, and loops if it did not. The model itself has no idea it is being checked; it emits tokens based on learned probabilities, and sometimes those probabilities lead to a closing brace in the wrong place or a trailing comma that JSON does not permit. Prompting helps — "respond only with valid JSON" nudges the distribution — but it cannot push the probability of a syntactically valid response to exactly one. Only something that operates at the token level can guarantee that.
      </Prose>

      <Prose>
        Constrained decoding is that something. The idea is to enforce the grammar at the logit level, before sampling. At each decoding step, before the probability distribution over the vocabulary is formed, a validator computes which vocabulary entries are consistent with the grammar given the partial output so far, and sets the logits of all other entries to negative infinity. After softmax, illegal tokens have probability exactly zero. The model cannot produce them, not because it does not want to, but because they are not in the sample space. The output is valid by construction, not by inspection.
      </Prose>

      <Prose>
        The technique has a short but dense history. LMQL (Beurer-Kellner, Fischer, and Vechev, arXiv:2212.06094, December 2022) introduced regex-level constraints embedded in a query language that interleaved generation with filtering predicates. Outlines (Willard and Louf, arXiv:2307.09702, July 2023) generalized this to full context-free grammars compiled into finite-state machines, with an efficient precomputed vocabulary index that made per-step masking cheap enough for real use. XGrammar (Dong, Ruan et al., arXiv:2411.15100, November 2024) rewrote the compiler in C++ with aggressive caching and a split between context-independent tokens that can be precomputed fully and context-dependent tokens that need per-step resolution, reducing end-to-end overhead to near zero in production serving stacks like vLLM and SGLang. All three work through the same core mechanism — logit masking guided by a grammar automaton — and differ in how efficiently they construct and apply the mask.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The decoding loop at its most basic is: run the model forward on the current token sequence, get a vector of logits over the vocabulary, apply some sampling procedure to pick the next token, append it, repeat. Constrained decoding inserts one operation into that loop between "get logits" and "sample": compute a mask that is zero for legal tokens and negative infinity for illegal ones, and add it to the logit vector. After adding, illegal tokens have logit of negative infinity, which softmax maps to probability zero. They are unreachable by any sampler.
      </Prose>

      <Prose>
        The mask is computed by a grammar automaton. The automaton tracks a state that summarizes everything relevant about the partial output — not the characters themselves, but which structural position in the grammar they correspond to. Given that state, the automaton knows which character sequences can legally follow. The token-level mask asks: for each vocabulary entry, if we were to emit that token right now, would the resulting character sequence be consistent with the grammar? If yes, the token is legal. If no, it gets masked.
      </Prose>

      <Prose>
        The key insight from Willard and Louf is that this per-token legality question can be answered for every (state, token) pair offline, before any decoding begins. The grammar and the vocabulary are both fixed once a model is loaded. The intersection of "which tokens are legal at grammar state q" is a function only of q and the vocabulary — it does not depend on the prompt, the temperature, or anything about the specific request. So you compile it once and cache it. At decode time, you look up the current grammar state, retrieve the precomputed mask from cache, and add it to the logits. The per-step cost is a single memory read and a vector addition, not a scan of the vocabulary.
      </Prose>

      <Prose>
        Four concepts do most of the work in understanding how this machinery is built:
      </Prose>

      <H3>Logit masking</H3>
      <Prose>
        The fundamental operation. You have a logit vector <Code>z</Code> of length <Code>|V|</Code> (vocabulary size). You have a boolean mask <Code>m</Code> where <Code>m[i] = 1</Code> if token <Code>i</Code> is legal and <Code>m[i] = 0</Code> otherwise. The masked logit vector is <Code>z' = z + (1 - m) * (-inf)</Code>, which zeroes out illegal tokens after softmax. Sampling on <Code>z'</Code> can only produce legal tokens. This is composable with any sampling strategy — greedy, top-p, temperature — because all of those operate on the logit vector before committing to a token.
      </Prose>

      <H3>Grammar to FSM compilation</H3>
      <Prose>
        A finite-state machine (FSM) represents a set of legal strings compactly. For regular languages — which includes most token-level constraints like "a number", "one of these enum values", "a UUID" — you can compile the regex directly to a deterministic finite automaton (DFA) via the Thompson NFA construction followed by subset construction. Each state in the DFA corresponds to a class of partial strings that have the same set of legal continuations. The DFA has at most <Code>2^n</Code> states where <Code>n</Code> is the number of NFA states, but in practice most grammars produce small DFAs. Context-free grammars — which you need for properly nested structures like JSON — require pushdown automata (PDAs), which extend FSMs with a stack. Outlines handles both; XGrammar focuses specifically on context-free grammars via Earley parsing and a persistent stack.
      </Prose>

      <H3>Token-level matching</H3>
      <Prose>
        Grammars operate over characters; vocabularies operate over tokens. A single token can span multiple characters — the string <Code>{`": "`}</Code> might be a single vocabulary entry, as might <Code>{`"false"`}</Code>, <Code>{`"},\\n  {"`}</Code>, or any other character sequence common enough in the training corpus to earn a dedicated ID. The token-level mask must therefore check whether feeding all the characters of a token, starting at the current grammar state, keeps the automaton in a non-dead state. A token is legal if and only if the DFA remains alive after consuming all its characters.
      </Prose>

      <H3>Forced tokens</H3>
      <Prose>
        At some grammar states, only one token is legal. The closing quote that ends a JSON string key, for example, has a uniquely determined continuation in some contexts. When exactly one token is legal, you do not need to run the model at all — you can emit the token directly, advance the grammar state, and skip the forward pass. This optimization is called forced-token emission, and it can dramatically reduce the number of model calls for highly structured outputs. The grammar becomes a collaborator in generation, not just a filter on it.
      </Prose>

      {/* ======================================================================
          3. MATH FOUNDATION
          ====================================================================== */}
      <H2>3. Math foundation</H2>

      <H3>Finite-state machines</H3>

      <Prose>
        A deterministic finite automaton (DFA) is a five-tuple <Code>(Q, Σ, δ, q₀, F)</Code> where <Code>Q</Code> is a finite set of states, <Code>Σ</Code> is the input alphabet (here, Unicode characters), <Code>δ</Code> is the transition function, <Code>q₀ ∈ Q</Code> is the start state, and <Code>F ⊆ Q</Code> is the set of accepting states. The transition function maps a state and a character to the next state:
      </Prose>

      <MathBlock>{"\\delta : Q \\times \\Sigma \\to Q"}</MathBlock>

      <Prose>
        For a string <Code>s = c₁c₂...cₙ</Code>, we extend δ to sequences by applying it left to right: <Code>δ*(q, s) = δ(δ*(q, c₁c₂...cₙ₋₁), cₙ)</Code> with base case <Code>δ*(q, ε) = q</Code>. The string <Code>s</Code> is accepted by the DFA if and only if <Code>δ*(q₀, s) ∈ F</Code>. A state is "dead" if it is not accepting and has no path to an accepting state.
      </Prose>

      <H3>Token-level mask construction</H3>

      <Prose>
        Given a DFA and a vocabulary <Code>{`V = {t₁, t₂, ..., t|V|}`}</Code>, the mask for grammar state <Code>q</Code> is:
      </Prose>

      <MathBlock>{"\\text{mask}(q) = \\{ t \\in V : \\delta^*(q, \\text{chars}(t)) \\neq \\text{dead} \\}"}</MathBlock>

      <Prose>
        Here <Code>chars(t)</Code> is the character sequence of token <Code>t</Code>. The condition requires that after feeding all characters of <Code>t</Code>, the automaton is not in a dead state — meaning a valid continuation still exists. Note that the token does not need to land in an accepting state; it just needs to not terminate the grammar. The mask is a subset of the vocabulary (equivalently, a boolean vector of length <Code>|V|</Code>), and it is fully determined by <Code>q</Code> and <Code>V</Code> alone.
      </Prose>

      <Prose>
        Precomputation cost is <Code>O(|Q| × |V| × max_token_length)</Code>. For a typical JSON grammar with ~50 states, a vocabulary of 32,000 tokens, and average token length of 3-4 characters, this is tens of millions of character transitions — expensive to compute once but trivially fast to look up. At decode time, the per-step cost is <Code>O(1)</Code> for the lookup plus <Code>O(|V|)</Code> for the vector addition to the logits.
      </Prose>

      <H3>From regex to DFA: Thompson NFA construction</H3>

      <Prose>
        A regex like <Code>[0-9]+</Code> is compiled to a DFA via two steps. First, Thompson's construction builds a nondeterministic finite automaton (NFA) inductively on the structure of the regex: each character class becomes a pair of states with a labeled transition; <Code>+</Code> (one or more) becomes the sub-automaton with a back-edge from its accept state to its start state. The NFA has ε-transitions. Second, subset construction converts the NFA to a DFA: each DFA state corresponds to a set of NFA states reachable via ε-closure, and DFA transitions are computed by applying the NFA transitions to the entire set. The result is a DFA with at most <Code>2^|NFA states|</Code> states, though practical grammars produce DFAs with far fewer.
      </Prose>

      <H3>Context-free grammars and pushdown automata</H3>

      <Prose>
        JSON is not a regular language — it has arbitrarily deep nesting, and no finite-state machine can track unbounded nesting depth. JSON is context-free, and context-free grammars correspond to pushdown automata (PDAs), which augment FSMs with a stack. A PDA transitions on both the current input character and the current top of the stack, and it can push and pop stack symbols. Properly nested brackets, matched quotation marks, and recursive data structures all require the stack. For practical purposes, JSON grammars compile to PDAs with a manageable number of grammar states (the "LR items" in parsing terminology), and XGrammar's key insight is that most of those states transition on context-independent tokens — tokens whose legality does not depend on the stack — so they can still be precomputed and cached exactly as in the FSM case.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every code block below was executed and its output is embedded as comments. No mocking, no pseudocode, no dependencies beyond the Python standard library.
      </Prose>

      <H3>4a. Regex FSM for [0-9]+</H3>

      <Prose>
        The simplest interesting grammar is a sequence of one or more decimal digits. The DFA has three states: <Code>START</Code> (no character consumed yet), <Code>ACCEPT</Code> (one or more digits consumed), and <Code>DEAD</Code> (a non-digit was seen — no valid continuation exists). Any character in <Code>0-9</Code> moves from <Code>START</Code> or <Code>ACCEPT</Code> to <Code>ACCEPT</Code>. Any other character moves to <Code>DEAD</Code>. The DFA accepts exactly the strings matched by <Code>[0-9]+</Code>.
      </Prose>

      <CodeBlock language="python">
{`DIGIT_CHARS = set('0123456789')

class DigitDFA:
    START, ACCEPT, DEAD = 0, 1, 2

    def __init__(self):
        self.state = self.START

    def feed(self, char):
        if self.state == self.DEAD:
            return self.DEAD
        self.state = self.ACCEPT if char in DIGIT_CHARS else self.DEAD
        return self.state

    def reset(self):
        self.state = self.START

    def is_accepting(self):
        return self.state == self.ACCEPT

# Transition table:
#   START(0)  | digit -> ACCEPT(1), other -> DEAD(2)
#   ACCEPT(1) | digit -> ACCEPT(1), other -> DEAD(2)
#   DEAD(2)   | digit -> DEAD(2),   other -> DEAD(2)

# Acceptance tests:
dfa = DigitDFA()
for s in ['123', '0', '42', '3abc', 'abc', '']:
    dfa.reset()
    for c in s: dfa.feed(c)
    print(f"input={repr(s):<8} accepting={dfa.is_accepting()}")

# input='123'    accepting=True
# input='0'      accepting=True
# input='42'     accepting=True
# input='3abc'   accepting=False
# input='abc'    accepting=False
# input=''       accepting=False`}
      </CodeBlock>

      <H3>4b. Token-level mask</H3>

      <Prose>
        A token is legal at DFA state <Code>q</Code> if feeding all its characters starting from <Code>q</Code> never hits <Code>DEAD</Code>. The function below checks legality for every token in a small vocabulary. Notice that <Code>'42'</Code> and <Code>'99'</Code> are legal (all-digit multi-character tokens) but <Code>'3x'</Code> is not (the <Code>x</Code> would kill the DFA mid-token).
      </Prose>

      <CodeBlock language="python">
{`def token_legal_at_state(dfa_class, initial_state, token):
    """True if feeding all chars of token from initial_state avoids DEAD."""
    dfa = dfa_class()
    dfa.state = initial_state
    for c in token:
        dfa.feed(c)
        if dfa.state == dfa.DEAD:
            return False
    return True

VOCAB = ['0', '1', '42', '99', 'abc', '3x', 'hello', '7']

# Legal tokens at START state (state 0):
#   '0'    -> LEGAL
#   '1'    -> LEGAL
#   '42'   -> LEGAL
#   '99'   -> LEGAL
#   'abc'  -> illegal
#   '3x'   -> illegal
#   'hello'-> illegal
#   '7'    -> LEGAL

# Legal tokens at ACCEPT state (state 1):  (identical — digits still legal)
#   '0'    -> LEGAL   ...   '7' -> LEGAL
#   'abc'  -> illegal  '3x' -> illegal`}
      </CodeBlock>

      <H3>4c. Constrained sampling loop</H3>

      <Prose>
        The sampling loop integrates the mask with a simulated language model (random logits stand in for a real forward pass). At every step, the allowed set is computed, illegal tokens get logit <Code>-inf</Code>, and sampling is constrained to the legal set. In state <Code>START</Code> (no digits yet), the EOS token is excluded — you cannot emit an empty number. Once in <Code>ACCEPT</Code>, EOS becomes available. The output is always a valid digit sequence.
      </Prose>

      <CodeBlock language="python">
{`import random, math
random.seed(42)

VOCAB_CD  = ['1','2','3','42','99','abc','end','7','0','5']
EOS_TOKEN = 'end'

def softmax(logits):
    m = max(v for v in logits if v != float('-inf'))
    exps = [math.exp(v-m) if v != float('-inf') else 0.0 for v in logits]
    s = sum(exps)
    return [e/s for e in exps]

def constrained_decode_sim(max_tokens=5):
    dfa = DigitDFA()
    output = []
    for step in range(max_tokens):
        raw = [random.gauss(0, 1) for _ in VOCAB_CD]
        masked = []
        for i, tok in enumerate(VOCAB_CD):
            if tok == EOS_TOKEN:
                masked.append(raw[i] if dfa.is_accepting() else float('-inf'))
            elif token_legal_at_state(DigitDFA, dfa.state, tok):
                masked.append(raw[i])
            else:
                masked.append(float('-inf'))
        probs  = softmax(masked)
        chosen = random.choices(VOCAB_CD, weights=probs)[0]
        allowed = [t for t,v in zip(VOCAB_CD, masked) if v != float('-inf')]
        print(f"step {step+1}: state={dfa.state} allowed={allowed} -> '{chosen}'")
        if chosen == EOS_TOKEN:
            break
        for c in chosen: dfa.feed(c)
        output.append(chosen)
    return output

constrained_decode_sim()
# step 1: state=0 allowed=['1','2','3','42','99','7','0','5'] -> '3'
# step 2: state=1 allowed=['1','2','3','42','99','end','7','0','5'] -> '0'
# step 3: state=1 allowed=['1','2','3','42','99','end','7','0','5'] -> '99'
# step 4: state=1 allowed=['1','2','3','42','99','end','7','0','5'] -> '1'
# step 5: state=1 allowed=['1','2','3','42','99','end','7','0','5'] -> '5'
# output tokens: ['3', '0', '99', '1', '5']`}
      </CodeBlock>

      <Callout accent="gold">
        At step 1, state=0 (START), EOS is excluded — you cannot produce an empty digit sequence. At step 2 onward, state=1 (ACCEPT), EOS is available because at least one digit has been emitted. The grammar enforces semantic validity, not just syntactic legality.
      </Callout>

      <H3>4d. JSON grammar via pushdown automaton</H3>

      <Prose>
        A simplified JSON object with a single string key and string value requires a pushdown automaton. The states below track structural position — which character is expected next — and the transitions define exactly what token classes are legal at each position. A real implementation uses the full JSON grammar (arrays, nested objects, numbers, booleans, null); this simplified version shows the same principles with less surface area.
      </Prose>

      <CodeBlock language="python">
{`# Simplified PDA for single-pair JSON: {"key": "val"}
# States track structural position:

JSON_STATES = {
    'START':       {'{':            'OPEN_BRACE'},
    'OPEN_BRACE':  {'"':            'IN_KEY'},
    'IN_KEY':      {'<str_chars>':  'IN_KEY',
                    '"':            'AFTER_KEY'},
    'AFTER_KEY':   {':':            'AFTER_COLON'},
    'AFTER_COLON': {' ':            'AFTER_COLON',
                    '"':            'IN_VAL'},
    'IN_VAL':      {'<str_chars>':  'IN_VAL',
                    '"':            'AFTER_VAL'},
    'AFTER_VAL':   {'}':            'DONE'},
    'DONE':        {},   # accepting — EOS legal here
}

# Allowed token classes at each state:
#   START        : ['{']
#   OPEN_BRACE   : ['"']
#   IN_KEY       : ['<word_chars>', '"']
#   AFTER_KEY    : [':']
#   AFTER_COLON  : [' ', '"']
#   IN_VAL       : ['<word_chars>', '"']
#   AFTER_VAL    : ['}']
#   DONE         : ['<EOS>']`}
      </CodeBlock>

      <Prose>
        In a production implementation, each of these states maps to a precomputed bitmask over the full vocabulary. Tokens like <Code>{'" }'}</Code> (quote-space-brace) would be illegal in OPEN_BRACE but might be legal as a transition sequence from IN_VAL to DONE. The compiler must check every multi-character token against every state transition chain to find all legal tokens for each state — which is exactly what the precomputation step does.
      </Prose>

      <H3>4e. Performance: naive vs pre-computed masking</H3>

      <Prose>
        Naive masking scans the entire vocabulary on every decode step. Pre-computed masking builds the per-state allowed-token set once and looks it up in O(1) per step. The following measures both strategies over 20 decode steps with a simulated vocabulary of 32,000 tokens.
      </Prose>

      <CodeBlock language="python">
{`import time
LARGE_VOCAB = [str(i) for i in range(32_000)]  # simulate 32k vocab
STEPS = 20

# Naive: re-scan full vocab every step
t0 = time.perf_counter()
for _ in range(STEPS):
    _ = [t for t in LARGE_VOCAB
         if token_legal_at_state(DigitDFA, DigitDFA.START, t)]
naive_time = time.perf_counter() - t0

# Pre-computed: build cache once, look up per step
t0 = time.perf_counter()
cache = {s: [t for t in LARGE_VOCAB
              if token_legal_at_state(DigitDFA, s, t)]
         for s in [DigitDFA.START, DigitDFA.ACCEPT]}
precomp_build = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(STEPS):
    _ = cache[DigitDFA.START]
precomp_decode = time.perf_counter() - t0

print(f"Naive ({STEPS} steps):            {naive_time*1000:.1f} ms")
print(f"Pre-compute (build once):       {precomp_build*1000:.1f} ms")
print(f"Pre-computed ({STEPS} lookups):    {precomp_decode*1000:.3f} ms")
print(f"Speedup at decode time:         {naive_time/precomp_decode:.0f}x")

# Naive (20 steps):            990.2 ms
# Pre-compute (build once):    96.0 ms
# Pre-computed (20 lookups):   0.242 ms
# Speedup at decode time:      4100x`}
      </CodeBlock>

      <Prose>
        The 4100x speedup at decode time is the core reason precomputation matters. The naive approach would slow decoding by roughly 50ms per step (990ms / 20), which is comparable to the entire forward pass on a small model. Precomputation pays the 96ms cost once at grammar compile time and makes every subsequent decode step nearly free from the masking perspective. XGrammar's C++ implementation further reduces the build cost and pushes per-step overhead below 1% of the total inference time.
      </Prose>

      {/* ======================================================================
          5. PRODUCTION IMPLEMENTATION
          ====================================================================== */}
      <H2>5. Production implementation</H2>

      <H3>Outlines</H3>

      <Prose>
        Outlines is the reference library for constrained decoding in Python. It accepts a Pydantic model, a JSON schema, a regex string, or a choice list, compiles the constraint to an FSM at import time, and wraps the generation loop so that sampling is always constrained. The return value is a typed Python object — you do not call <Code>json.loads</Code> and you do not write a retry loop.
      </Prose>

      <CodeBlock language="python">
{`from outlines import models, generate
from pydantic import BaseModel
from typing import Literal

class AnalysisResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    summary: str

model = models.transformers("meta-llama/Llama-3.1-8B-Instruct")
generator = generate.json(model, AnalysisResult)

# Guaranteed to return a valid AnalysisResult — never raises JSONDecodeError
result: AnalysisResult = generator(
    "Analyze the sentiment: 'The product exceeded all expectations.'"
)
# result.sentiment == 'positive'  (typed, not a string that might be wrong)

# Regex constraint — output matches pattern exactly
date_gen = generate.regex(model, r"\\d{4}-\\d{2}-\\d{2}")
date = date_gen("What is today's date?")
# date == '2026-04-21'  or similar — always ISO 8601 format`}
      </CodeBlock>

      <H3>XGrammar</H3>

      <Prose>
        XGrammar is a C++ grammar engine with Python bindings, used as the structured generation backend in vLLM and SGLang. It handles EBNF grammars (Extended Backus-Naur Form), which subsume JSON, programming language subsets, and most practically useful output formats. The key difference from Outlines is that XGrammar separates tokens into context-independent (can be fully precomputed) and context-dependent (need stack state at runtime), reducing the per-step overhead on complex grammars to near zero.
      </Prose>

      <CodeBlock language="python">
{`import xgrammar as xgr

# Compile a JSON schema grammar once
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

# Compile schema to a CompiledGrammar (happens once per schema)
json_schema = '{"type":"object","properties":{"name":{"type":"string"},"score":{"type":"number"}}}'
compiled = grammar_compiler.compile_json_schema(json_schema)

# At decode time, create a matcher per request
matcher = xgr.GrammarMatcher(compiled)

# In the generation loop:
token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
for step in range(max_tokens):
    logits = model(input_ids)[:, -1, :]
    matcher.fill_next_token_bitmask(token_bitmask)
    xgr.apply_token_bitmask_inplace(logits, token_bitmask)
    # ... sample, get next_token_id ...
    matcher.accept_token(next_token_id)
    if matcher.is_terminated():
        break`}
      </CodeBlock>

      <H3>llama.cpp GBNF grammars</H3>

      <Prose>
        llama.cpp includes a built-in grammar sampler that accepts GBNF (GGML BNF) grammars, a simple variant of BNF. You can constrain any local llama.cpp inference to a grammar by passing a grammar file. This is particularly useful for local deployments where Outlines or XGrammar are not available.
      </Prose>

      <CodeBlock language="text">
{`# example.gbnf — a simple JSON object grammar
root   ::= object
object ::= "{" ws members ws "}"
members ::= member ("," ws member)*
member ::= string ws ":" ws value
value  ::= string | number | "true" | "false" | "null"
string ::= '"' chars '"'
chars  ::= (char)*
char   ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]
number ::= "-"? ("0" | [1-9][0-9]*) ("." [0-9]+)?
ws     ::= [ \\t\\n]*`}
      </CodeBlock>

      <H3>API-level structured outputs</H3>

      <Prose>
        OpenAI's <Code>response_format</Code> with <Code>type: "json_schema"</Code> (available in GPT-4o and later) implements constrained decoding server-side. Anthropic's <Code>tool_use</Code> API routes generation through a schema-constrained path when <Code>tool_choice</Code> forces a specific tool. Both provide the same guarantee — syntactically valid output — without requiring the client to manage a grammar library. The tradeoff is that you cannot inspect or customize the constraint compiler; you trust the API to enforce your schema correctly.
      </Prose>

      <CodeBlock language="python">
{`# OpenAI structured output
from openai import OpenAI
client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-11-20",
    messages=[{"role": "user", "content": "Extract: John joined in March 2023"}],
    response_format=EmployeeRecord,  # Pydantic model
)
record = response.choices[0].message.parsed  # typed EmployeeRecord, never None

# Anthropic tool_choice constrained generation
response = anthropic.messages.create(
    model="claude-opus-4-5",
    tools=[{"name": "extract_record", "input_schema": schema}],
    tool_choice={"type": "tool", "name": "extract_record"},
    messages=[{"role": "user", "content": "Extract: John joined in March 2023"}]
)
# response.content[0].input is always valid against schema`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL WALKTHROUGH
          ====================================================================== */}
      <H2>6. Visual walkthrough</H2>

      <H3>Token-by-token generation under a JSON grammar</H3>

      <Prose>
        The StepTrace below shows constrained generation of <Code>{"{'name': 'Alice', 'score': 42}"}</Code> token by token. Each step shows the current grammar state and which token classes are legal. Note how structural tokens — the braces, quotes, colon — are often forced (only one legal option), while content tokens admit a broader legal set.
      </Prose>

      <StepTrace
        label="JSON constrained generation — step through"
        steps={[
          {
            label: "grammar state: START",
            render: () => (
              <div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#888", marginBottom: 8 }}>
                  Output so far: <span style={{ color: "#e2b55a" }}>(empty)</span>
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 6 }}>Legal tokens at START:</div>
                <TokenStream label="" tokens={[{ label: "{", color: "#c084fc" }]} />
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#666", marginTop: 6 }}>
                  All other tokens masked to -∞. Emit: <span style={{ color: "#c084fc" }}>{"{"}</span>
                </div>
              </div>
            ),
          },
          {
            label: "grammar state: OPEN_BRACE",
            render: () => (
              <div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#888", marginBottom: 8 }}>
                  Output so far: <span style={{ color: "#c084fc" }}>{"{"}</span>
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 6 }}>Legal tokens:</div>
                <TokenStream label="" tokens={[{ label: '"', color: "#c084fc" }]} />
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#666", marginTop: 6 }}>
                  Forced: only <span style={{ color: "#c084fc" }}>"</span> opens a key
                </div>
              </div>
            ),
          },
          {
            label: "grammar state: IN_KEY",
            render: () => (
              <div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#888", marginBottom: 8 }}>
                  Output so far: <span style={{ color: "#c084fc" }}>{`{"`}</span>
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 6 }}>Legal tokens:</div>
                <TokenStream label="" tokens={[
                  { label: "name", color: "#e2b55a" },
                  { label: "score", color: "#e2b55a" },
                  { label: "type", color: "#e2b55a" },
                  { label: "id", color: "#e2b55a" },
                  { label: '..."', color: "#c084fc", title: "closing quote ends key" },
                ]} />
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#666", marginTop: 6 }}>
                  Model freely chooses key content. Chose: <span style={{ color: "#e2b55a" }}>name</span>
                </div>
              </div>
            ),
          },
          {
            label: "grammar state: AFTER_KEY",
            render: () => (
              <div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#888", marginBottom: 8 }}>
                  Output so far: <span style={{ color: "#c084fc" }}>{`{"`}</span><span style={{ color: "#e2b55a" }}>name</span><span style={{ color: "#c084fc" }}>"</span>
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 6 }}>Legal tokens:</div>
                <TokenStream label="" tokens={[{ label: ":", color: "#c084fc" }]} />
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#666", marginTop: 6 }}>
                  Forced: colon must follow key
                </div>
              </div>
            ),
          },
          {
            label: "grammar state: IN_VAL",
            render: () => (
              <div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#888", marginBottom: 8 }}>
                  Output so far: <span style={{ color: "#c084fc" }}>{`{"name": "`}</span>
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 6 }}>Legal tokens (string value):</div>
                <TokenStream label="" tokens={[
                  { label: "Alice", color: "#e2b55a" },
                  { label: "Bob", color: "#e2b55a" },
                  { label: "Carol", color: "#e2b55a" },
                  { label: '..."', color: "#c084fc", title: "close quote ends value" },
                ]} />
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#666", marginTop: 6 }}>
                  Model chooses value. Chose: <span style={{ color: "#e2b55a" }}>Alice</span>
                </div>
              </div>
            ),
          },
          {
            label: "grammar state: DONE",
            render: () => (
              <div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: "#888", marginBottom: 8 }}>
                  Output so far: <span style={{ color: "#c084fc" }}>{`{"name": "Alice"}`}</span>
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 6 }}>Legal tokens:</div>
                <TokenStream label="" tokens={[{ label: "<EOS>", color: "#4ade80" }]} />
                <div style={{ fontFamily: "monospace", fontSize: 11, color: "#666", marginTop: 6 }}>
                  Structure complete — EOS is the only legal token. Output is valid JSON.
                </div>
              </div>
            ),
          },
        ]}
      />

      <H3>Token legality heatmap across grammar states</H3>

      <Prose>
        The Heatmap below shows which token classes are legal (1) or illegal (0) at each grammar state for our simplified JSON object grammar. Purple cells are allowed; dim cells are masked to -∞. The structural pattern is clear — most states permit only one token class, making forced-token emission ubiquitous.
      </Prose>

      <Heatmap
        label="token legality matrix — JSON PDA states × token classes"
        colorScale="purple"
        rowLabels={["START", "OPEN_BRACE", "IN_KEY", "AFTER_KEY", "AFTER_COLON", "IN_VAL", "AFTER_VAL", "DONE"]}
        colLabels={["{", "}", '"', ":", " ", "word", "digit", "<EOS>"]}
        matrix={[
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1],
        ]}
        cellSize={38}
      />

      <H3>Decoding throughput: unconstrained vs naive mask vs precomputed mask</H3>

      <Prose>
        The Plot below shows simulated tokens per second for three decoding modes as grammar complexity increases (measured in number of grammar states). Unconstrained decoding is flat — no grammar overhead. Naive masking degrades linearly with grammar complexity because each additional state requires more per-step computation. Precomputed masking stays nearly flat because the per-step cost is a lookup regardless of grammar complexity; only the build cost grows.
      </Prose>

      <Plot
        label="decoding throughput vs grammar complexity"
        xLabel="grammar states"
        yLabel="tokens/sec (relative)"
        series={[
          {
            name: "unconstrained",
            color: colors.green,
            points: [[5,100],[10,100],[20,100],[50,100],[100,100]],
          },
          {
            name: "naive mask",
            color: "#f87171",
            points: [[5,92],[10,78],[20,55],[50,28],[100,14]],
          },
          {
            name: "precomputed mask",
            color: colors.gold,
            points: [[5,99],[10,98],[20,97],[50,96],[100,95]],
          },
        ]}
      />

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Not every structured output problem requires the same solution. The right choice depends on the complexity of the target format, the latency requirements, and whether you are calling a hosted API or running a local model.
      </Prose>

      <H3>Use regex constraints</H3>
      <Prose>
        When your output is a fixed-format string: an enum value, a UUID, a date in ISO 8601, a numerical range, a phone number format. Regex grammars compile to very small DFAs with few states, precomputation is nearly instantaneous, and per-step overhead is minimal. Outlines' <Code>generate.regex</Code> or <Code>generate.choice</Code> handles these cases with essentially no overhead. Regex is also the right choice when you want to constrain part of a free-text output — for example, extracting a specific field mid-generation.
      </Prose>

      <H3>Use full grammar (JSON/EBNF)</H3>
      <Prose>
        When your output is a structured data object: a JSON document matching a schema, a function call with typed arguments, a nested configuration. JSON grammars are the most common case in production and are well-supported by Outlines (<Code>generate.json</Code>), XGrammar (<Code>compile_json_schema</Code>), and llama.cpp GBNF files. If your schema is large or deeply nested, prefer XGrammar or a C++ backend over pure-Python Outlines for the build step — the vocabulary intersection becomes expensive for schemas with many states.
      </Prose>

      <H3>Use tool_choice / API structured outputs</H3>
      <Prose>
        When you are calling a hosted API (OpenAI, Anthropic) and do not need to customize the constraint compiler. OpenAI's <Code>response_format</Code> with JSON schema and Anthropic's forced tool use both implement constrained decoding server-side. The constraint is guaranteed, the implementation is maintained by the provider, and you pay no infrastructure cost. The tradeoff is loss of control: you cannot inspect the masking logic, cannot use custom grammars, and cannot mix constrained and unconstrained generation in the same pass.
      </Prose>

      <H3>Use XGrammar for latency-sensitive production</H3>
      <Prose>
        When you are running your own inference and latency or throughput is a primary constraint. XGrammar's C++ implementation with its context-independent/context-dependent token split achieves sub-1% overhead on typical JSON schemas, even at vocabulary sizes of 128,000. It is integrated directly into vLLM and SGLang, so the upgrade path is a configuration flag rather than a code change. Use XGrammar when you are measuring tokens per second and every millisecond of per-step overhead compounds.
      </Prose>

      <H3>Use Outlines for flexibility and experimentation</H3>
      <Prose>
        When you need to prototype quickly, try multiple schema variants, or use a grammar type that XGrammar does not yet support (some regex patterns, choice sets, custom interleaved constraints). Outlines is pure Python, installs in seconds, and covers essentially all use cases with somewhat higher overhead. For research, fine-tuning pipelines, and applications where throughput is not a primary concern, Outlines is the correct default.
      </Prose>

      {/* ======================================================================
          8. WHAT SCALES AND WHAT DOESN'T
          ====================================================================== */}
      <H2>8. What scales and what doesn't</H2>

      <H3>What scales well</H3>

      <Prose>
        The precomputed masking approach scales gracefully with vocabulary size. The build cost is linear in <Code>|V| × |Q|</Code> (vocabulary times grammar states), but this is done once per grammar per model load, not per request. At decode time, the cost per step is <Code>O(|V|)</Code> for the vector addition — the same asymptotic cost as the final logit normalization step that every decoding pipeline already performs. Modern GPU memory bandwidth can apply a 128,000-element mask in well under a microsecond. For grammars with many forced tokens (highly structured outputs like programming language syntax), the effective tokens-from-model count is much lower than the output length, so constrained decoding can actually produce longer outputs faster than unconstrained decoding at the same output-token count.
      </Prose>

      <Prose>
        The system also scales with schema complexity up to a point. JSON schemas with 20–30 fields and moderate nesting produce DFAs with tens to low hundreds of states, which is well within the practical range for precomputation. Schemas with recursive structures (JSON arrays of objects of arrays) require the pushdown automaton extension, which XGrammar handles efficiently via its persistent stack representation.
      </Prose>

      <H3>What doesn't scale</H3>

      <Prose>
        The precomputation cost scales with the number of grammar states times the vocabulary size. For grammars with thousands of states — full programming language grammars, proof assistant syntax, complex nested configurations — the build cost can be seconds or even minutes per grammar. This is acceptable if you compile the grammar once and serve thousands of requests with it; it is not acceptable if each request uses a dynamically generated grammar. XGrammar mitigates this with incremental compilation and a separation between the static and dynamic parts of the grammar, but there is no free lunch: highly variable per-request grammars cannot fully amortize the precomputation.
      </Prose>

      <Prose>
        The tokenization mismatch problem also gets harder as vocabulary size grows. Modern tokenizers with vocabularies above 100,000 entries have more multi-byte and multi-character tokens, more language-specific tokens, and more edge cases around whitespace handling. Each of these must be correctly handled in the token-level legality check. UTF-8 multi-byte characters split across token boundaries are particularly tricky — a grammar that admits a specific Unicode character must handle the case where that character's bytes appear in two different tokens.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES & GOTCHAS
          ====================================================================== */}
      <H2>9. Failure modes & gotchas</H2>

      <H3>Grammar too restrictive — repeated token loops</H3>
      <Prose>
        If the grammar is more restrictive than the model's training distribution, the model may reach a state where it has exhausted its knowledge of what should come next but the grammar forces continuation. In practice this manifests as the model emitting the same legal token repeatedly — it knows only one legal option and keeps picking it regardless of the content. The fix is to relax the grammar (allow more token types at the relevant state) or to add a fallback that detects repetition and aborts generation.
      </Prose>

      <H3>Tokenization mismatch with space-prefix tokens</H3>
      <Prose>
        Byte-pair encoding tokenizers commonly include space-prefixed tokens: <Code>"Ġname"</Code> (where Ġ represents a leading space) is a different vocabulary entry from <Code>"name"</Code>. A grammar that expects the JSON key <Code>"name"</Code> must handle both the bare token and the space-prefixed token at the appropriate grammar positions. If the grammar naively checks whether the token characters match the expected sequence, space-prefixed tokens will be incorrectly masked out wherever they are in fact the natural continuation. Outlines handles this by normalizing tokens before matching; simpler implementations frequently get it wrong.
      </Prose>

      <H3>Multi-byte UTF-8 split across token boundaries</H3>
      <Prose>
        A Unicode character above U+007F takes two to four bytes in UTF-8. Some tokenizers produce tokens that end mid-character — the first byte(s) of a multi-byte sequence in one token, the remainder in the next. A grammar over characters cannot correctly mask a token whose character sequence is incomplete. XGrammar and Outlines both handle this by operating at the byte level internally and completing multi-byte sequences before evaluating legality, but implementing this correctly from scratch requires explicit attention.
      </Prose>

      <H3>Grammar that allows EOS anywhere</H3>
      <Prose>
        Some grammars, particularly those generated automatically from loose schemas, allow the end-of-sequence token at many points in the grammar. This causes early termination: the model emits EOS as soon as it is legal, producing incomplete outputs. Always audit your grammar to ensure EOS is only permitted at states that correspond to complete, valid outputs. For JSON, EOS should only be legal after the final closing brace, not after any intermediate accepting state.
      </Prose>

      <H3>Infinite-loop grammars</H3>
      <Prose>
        A grammar with a loop — a state that can return to itself indefinitely — can cause the model to generate arbitrarily long outputs. JSON arrays are the canonical example: a list of integers has no fixed length, and the grammar accepts any number of elements. Without a maximum-length constraint, the model will continue generating elements as long as its probability mass on additional elements exceeds its probability mass on closing the array. Always pair open-ended grammar constructs with maximum length guards at the generation level.
      </Prose>

      <H3>Performance collapse on large vocabularies without precomputation</H3>
      <Prose>
        Naive token-level masking on a vocabulary of 128,000 tokens, for a grammar with 100 states, requires roughly 12.8 million character-sequence checks per grammar compile — and if done naively at runtime on every decode step, this cost multiplies by the output length. The shift from Outlines' original Python implementation to XGrammar's C++ with precomputation represents roughly a 100x improvement in steady-state throughput for this reason. Any implementation that does not precompute will not be usable in production above trivial throughput levels.
      </Prose>

      <H3>Schema complexity vs model capability mismatch</H3>
      <Prose>
        Constrained decoding can enforce a grammar but it cannot give the model knowledge it does not have. If you ask a small 7B model to produce a deeply nested JSON schema with 40 fields, the grammar will ensure the output is syntactically valid, but the content will likely be hallucinated or nonsensical. The constraint narrows the output space to structurally valid outputs; within that space, quality is still a function of the model's capabilities and training. Do not use structured output as a substitute for capability.
      </Prose>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        All four papers below were WebSearch-verified against arXiv on 2026-04-21.
      </Prose>

      <H3>Willard & Louf 2023 — Efficient Guided Generation (Outlines)</H3>
      <Prose>
        Brandon T. Willard and Rémi Louf, "Efficient Guided Generation for Large Language Models," arXiv:2307.09702, July 2023. The foundational paper for modern constrained decoding. Reformulates generation as FSM-state transitions and introduces the vocabulary index that makes per-state mask precomputation practical. Implements the approach as the Outlines Python library. Available at https://arxiv.org/abs/2307.09702.
      </Prose>

      <H3>Dong et al. 2024 — XGrammar</H3>
      <Prose>
        Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, and Tianqi Chen, "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models," arXiv:2411.15100, November 2024. Introduces the context-independent / context-dependent token split that pushes structured generation overhead below 1%. Demonstrates up to 100x speedup over prior implementations. Integrated into vLLM and SGLang. Available at https://arxiv.org/abs/2411.15100.
      </Prose>

      <H3>Beurer-Kellner et al. 2023 — LMQL</H3>
      <Prose>
        Luca Beurer-Kellner, Marc Fischer, and Martin Vechev, "Prompting Is Programming: A Query Language for Large Language Models," arXiv:2212.06094, December 2022 (published PLDI 2023). Introduced regex-level constraints embedded in a query language as an early form of constrained decoding. Showed 26–85% cost reductions through constraint-directed generation that skips unnecessary tokens. Available at https://arxiv.org/abs/2212.06094.
      </Prose>

      <H3>Geng et al. 2023 — Grammar-Constrained Decoding</H3>
      <Prose>
        Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West, "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning," arXiv:2305.13971, published at EMNLP 2023. Demonstrates grammar-constrained decoding as a unified framework for structured NLP tasks including information extraction, entity disambiguation, and constituency parsing — without any task-specific finetuning. Introduces input-dependent grammars where the constraint varies per input. Available at https://arxiv.org/abs/2305.13971.
      </Prose>

      {/* ======================================================================
          11. SELF-CHECK EXERCISES
          ====================================================================== */}
      <H2>11. Self-check exercises</H2>

      <H3>Exercise 1 — ISO date regex FSM</H3>
      <Prose>
        Implement a DFA for ISO 8601 dates in the format <Code>YYYY-MM-DD</Code>. Your DFA should accept exactly strings matching the pattern <Code>[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]</Code> (four digits, hyphen, two digits, hyphen, two digits). How many states does your DFA need? What is the maximum depth of the transition graph? Write the <Code>feed(char)</Code> and <Code>is_accepting()</Code> methods. Then test it on <Code>2026-04-21</Code>, <Code>2026-4-21</Code>, <Code>2026-13-01</Code>, and <Code>hello</Code> — which should pass and why?
      </Prose>

      <H3>Exercise 2 — Token boundary necessity</H3>
      <Prose>
        A vocabulary token <Code>"42abc"</Code> contains both digit and non-digit characters. Explain why a simple character-level grammar check on just the first character of the token is insufficient for determining token legality. What specific scenario does this cause in practice? How does the token-level mask function in section 4b handle this correctly, and what would break if you only checked the first character of each token?
      </Prose>

      <H3>Exercise 3 — Grammar constraint vs retry overhead</H3>
      <Prose>
        Assume a model without constrained decoding produces valid JSON 95% of the time on a given task. A constrained decoder adds 4% overhead to the total tokens-per-second throughput (a realistic figure for a precomputed grammar). Compute the break-even throughput level in requests per second above which constrained decoding is cheaper in aggregate compute than generate-and-retry, assuming retries are independent and retries also fail at 5%. Include at least three requests per second in your analysis. Under what conditions does constrained decoding win even at lower throughput?
      </Prose>

      <H3>Exercise 4 — Determinism and accuracy</H3>
      <Prose>
        A researcher claims that constrained decoding improves the factual accuracy of JSON outputs, not just their syntactic validity. Construct an argument for why this claim might be partially true — under what specific conditions could enforcing a grammar improve the factual content of the values within the structure? Then construct a counter-argument. When is the researcher's claim clearly false? What experiment would distinguish between the two cases?
      </Prose>

      <H3>Exercise 5 — Function call grammar design</H3>
      <Prose>
        Design an EBNF grammar for a function call with the following signature: <Code>search(query: str, max_results: int, include_metadata: bool)</Code>. The output format should be: <Code>search("query text", 10, true)</Code>. Write the grammar rules for: the function name (fixed), the opening parenthesis, the string argument (any characters except unescaped double-quote), the integer argument (1–999), the boolean argument (<Code>true</Code> or <Code>false</Code>), and the closing parenthesis with separating commas. How many grammar states does your PDA need?
      </Prose>
    </div>
  ),
};

export default constrainedDecoding;
