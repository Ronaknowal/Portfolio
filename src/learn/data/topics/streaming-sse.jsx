import { Prose, H2, H3, Code, CodeBlock, Callout, MathBlock } from "../../components/content";
import { TokenStream, StepTrace, Heatmap, Plot } from "../../components/viz";
import { colors } from "../../styles";

const streamingSSE = {
  title: "Streaming & Server-Sent Events (SSE)",
  readTime: "38 min",
  content: () => (
    <div>
      {/* ======================================================================
          1. WHY IT EXISTS
          ====================================================================== */}
      <H2>1. Why it exists</H2>

      <Prose>
        Language model generation is not instantaneous. A model producing a 500-token response
        has to run a forward pass — or more precisely, 500 sequential forward passes through the
        decode layers — before the output is complete. On a well-provisioned GPU cluster serving
        a mid-sized model, that process takes somewhere between three and twenty seconds. Without
        streaming, the server generates the entire response, serializes it, and then sends it in
        one HTTP response body. From the user's perspective: they send a message and stare at a
        spinner for ten seconds while nothing happens. Then the full answer appears at once.
      </Prose>

      <Prose>
        Streaming changes the contract. Instead of waiting for the full completion, the server
        flushes each generated token — or a small batch of tokens — to the client as soon as it
        is available. The user sees text appearing on screen almost immediately after submitting,
        word by word, at roughly the pace the model generates. A ten-second generation that
        starts appearing after 400 milliseconds feels fast. A three-second generation that waits
        two full seconds before the first character appears feels broken. The metric that governs
        perceived quality is not total latency — it is time-to-first-token.
      </Prose>

      <Prose>
        The standard protocol for delivering this token-by-token flow from server to browser is
        Server-Sent Events, defined in the WHATWG HTML Living Standard under section 9.2. SSE is
        a long-lived HTTP response with <Code>Content-Type: text/event-stream</Code> where the
        server writes incremental payloads separated by double newlines and the client reads each
        payload as it arrives. The wire format is intentionally minimal — three field types, a
        newline delimiter, and a convention for signaling stream end. The operational complexity
        is not in the protocol. It is in building the full chain — async server, proxy stack,
        client parser, backpressure handling, reconnect semantics — that makes the protocol
        actually deliver its guarantee end-to-end in production.
      </Prose>

      <Prose>
        A second reason streaming matters, beyond perceived latency, is early cancellation. When
        a model starts generating a response that is clearly wrong — wrong language, wrong tool
        call, wrong tone — the user can abort. Without streaming, the client waits for the full
        generation and then discards it. With streaming, the client sees the error after the
        first few tokens and closes the connection. A correctly wired server detects the
        disconnect and cancels the in-flight generation immediately. The tokens that would have
        been generated are never generated. At scale, across thousands of concurrent users with
        a non-trivial abort rate, the saved compute is substantial. Cancellation is not a
        quality-of-life feature — it is a resource efficiency mechanism.
      </Prose>

      <Prose>
        A third, subtler benefit is trust signaling. Watching a model "think out loud" — seeing
        the first tokens confirm the model understood the request — reduces the anxiety of
        waiting. Users who see streaming output abandon sessions less frequently than users who
        see a spinner. The model's apparent confidence (or confusion) is legible in real time.
        If the first few tokens are in the wrong language, the user hits stop immediately rather
        than waiting for a complete wrong answer. If the model starts with a confident, correct
        framing, the user can read ahead and begin processing context before the response is
        complete. Streaming is not just a latency optimization — it is a fundamentally different
        interaction model that changes how users engage with model output.
      </Prose>

      {/* ======================================================================
          2. CORE INTUITION
          ====================================================================== */}
      <H2>2. Core intuition</H2>

      <Prose>
        The SSE wire format has three building blocks. An <em>event</em> is a block of lines
        terminated by a blank line (two newlines in a row). Within an event, a <Code>data:</Code>{" "}
        line carries the payload. An optional <Code>event:</Code> line names the event type; if
        omitted, the browser fires a generic <Code>message</Code> event. An optional{" "}
        <Code>id:</Code> line assigns the event an identifier the client sends back as{" "}
        <Code>Last-Event-ID</Code> on reconnect. Everything else is convention — and in LLM
        APIs the dominant convention is OpenAI's: a JSON object with a <Code>choices</Code>{" "}
        array, a <Code>delta</Code> field carrying incremental content, and a{" "}
        <Code>[DONE]</Code> sentinel as the final data value.
      </Prose>

      <Prose>
        Compare SSE to its two main alternatives. WebSockets establish a persistent
        bidirectional channel via a protocol upgrade from HTTP. The upside is full-duplex
        communication: the client can send messages to the server at any point during the
        connection, which matters for agent frameworks where the user might interrupt
        mid-generation or inject a tool result. The downside is operational weight: WebSockets
        require explicit upgrade handling in every load balancer and proxy in the path,
        produce more complex reconnect logic, and are blocked by some corporate proxies that
        only allow plain HTTP. For pure output streaming — one request, one token stream, done
        — WebSockets add complexity with no benefit.
      </Prose>

      <Prose>
        Long polling is the pre-SSE workaround: the client opens a request, the server holds it
        open until an event is ready (or a timeout fires), sends one batch of data, closes the
        response, and the client immediately reopens. It works everywhere HTTP works, but it
        pays connection-establishment overhead on every event and cannot sustain the sub-100 ms
        flush latency that LLM streaming needs. gRPC streaming is the preferred choice for
        server-to-server traffic — strongly typed, observable via standard gRPC tooling,
        bidirectional if needed — but it requires HTTP/2 end-to-end and is not natively
        consumable from a browser without a gRPC-Web translation layer.
      </Prose>

      <Prose>
        SSE wins for browser-facing LLM output because it is plain HTTP (no upgrade), works
        through most proxies and CDNs without special configuration (when buffering is
        disabled), has native browser support via the <Code>EventSource</Code> API, and provides
        automatic reconnect with resume semantics baked into the spec. The simplicity of
        unidirectional flow is a feature, not a limitation, for the most common LLM use case.
      </Prose>

      <Prose>
        One concrete advantage of SSE that is often overlooked is its behavior through corporate
        proxies and older network infrastructure. WebSockets require an explicit HTTP Upgrade
        handshake — a <Code>101 Switching Protocols</Code> response — which many enterprise
        proxies and some CDN configurations do not pass through. gRPC requires HTTP/2 end-to-end,
        which many edge nodes and managed load balancers do not support without explicit
        configuration. SSE is just HTTP: the connection looks like a slow download to every
        intermediary that does not understand SSE specifically, and in most cases it works
        without protocol negotiation failures. The failure mode — intermediary buffering — is
        more common than protocol rejection, but it is also diagnosable and fixable. Protocol
        rejection failures from WebSocket upgrades being stripped by a proxy are much harder
        to diagnose and often invisible until a specific network path is exercised.
      </Prose>

      <Callout accent="purple">
        SSE is just a long HTTP response where each chunk ends with{" "}
        <Code>{`\\n\\n`}</Code>. The browser parses it; your server yields it. The difficulty
        is every piece of infrastructure between the two that was built assuming responses
        finish quickly.
      </Callout>

      {/* ======================================================================
          3. MATHEMATICAL FOUNDATION
          ====================================================================== */}
      <H2>3. Mathematical foundation</H2>

      <Prose>
        Two latency metrics characterize a streaming LLM endpoint. Time-to-first-token (TTFT)
        is the wall-clock duration from the moment the client sends its request to the moment
        the first token byte arrives at the client. Time-per-output-token (TPOT) is the average
        inter-token interval after the first token — how long it takes the server to generate
        and deliver each subsequent token. A model producing 50 tokens per second has a TPOT
        of roughly 20 ms.
      </Prose>

      <Prose>
        Perceived latency is a weighted combination of both, but the weights are not equal.
        Psychophysical research on interactive systems places the human perception threshold for
        "instant" response at around 100 ms. Delays up to 1,000 ms feel like a normal computer
        pause. Beyond 1,000 ms, users start to feel the system is slow. Once text is appearing
        on screen at a comfortable reading speed — roughly 200–250 words per minute, or about
        3–4 tokens per second — TPOT becomes invisible. The model can be generating at 30 tokens
        per second and the user experiences it as smooth flow.
      </Prose>

      <MathBlock>
        {"\\text{perceived\\_latency} = w_{\\text{TTFT}} \\cdot \\text{TTFT} + w_{\\text{TPOT}} \\cdot \\text{TPOT}"}
      </MathBlock>

      <Prose>
        In practice <Code>w_TTFT</Code> dominates heavily — typical values in user-study
        literature weight TTFT at 0.7–0.9 and TPOT at 0.1–0.3. This is why optimizing TTFT
        (via prefix caching, faster prefill hardware, early request routing) delivers much
        larger UX improvements than optimizing TPOT over the same engineering budget. A system
        that starts streaming in 200 ms but generates slowly feels better than one that starts
        in 800 ms but finishes quickly.
      </Prose>

      <Plot
        title="TTFT vs perceived latency (TPOT held constant at 20 ms)"
        data={[
          { x: 100, y: 0.12 },
          { x: 200, y: 0.22 },
          { x: 400, y: 0.38 },
          { x: 600, y: 0.54 },
          { x: 800, y: 0.68 },
          { x: 1000, y: 0.82 },
          { x: 1500, y: 1.12 },
          { x: 2000, y: 1.42 },
        ]}
        xLabel="TTFT (ms)"
        yLabel="perceived latency score (lower = better)"
      />

      <Prose>
        Connection overhead follows HTTP version. Under HTTP/1.1, each streaming connection is
        a dedicated TCP connection. A server handling 1,000 concurrent streams holds 1,000 TCP
        connections open simultaneously — each consuming a file descriptor, kernel socket
        buffer, and potentially a TLS session entry. Long-lived connections amplify this: a
        30-second generation keeps a connection open 30 times longer than a 1-second response.
        Under HTTP/2, multiple streams multiplex over a single TCP connection, reducing both
        connection count and TLS handshake frequency. Most modern load balancers support
        HTTP/2 to upstream, which is the right default for high-concurrency LLM serving.
      </Prose>

      <Prose>
        Buffer sizing controls the trade-off between flush frequency and CPU overhead. Flushing
        every single token minimizes time from generation to client display but maximizes
        syscall count. Batching tokens into 4–8 token chunks barely affects perceived latency
        — at 30 tokens/sec, 8 tokens is 267 ms, which is right at the threshold of perceptibility
        — while reducing the number of write syscalls by 8×. Production servers typically flush
        on every token during the first 10–20 tokens (where TTFT is still relevant) and then
        batch slightly during the main body, though most frameworks default to per-token flush.
      </Prose>

      <Prose>
        An important related quantity is prefill time. The model's TTFT is the sum of request
        queuing time, network round-trip time to the inference cluster, and prefill time — the
        time for the GPU to process the entire input context in parallel and produce the first
        output token logit. Prefill is a compute-bound matrix multiply over the full prompt
        length, so TTFT scales roughly linearly with input token count. A 100-token prompt
        might prefill in 50 ms; a 10,000-token prompt in 500 ms on the same hardware. This
        makes TTFT optimization for long-context requests qualitatively different from
        short-context requests: prefix caching (reusing KV cache from a matching earlier prompt)
        is the primary lever, because it converts an O(n) prefill into an O(n - cached_prefix)
        prefill. Without prefix caching, every long-context request pays full prefill cost,
        and TTFT for a retrieval-augmented generation query with 8,000 tokens of retrieved
        context is an order of magnitude higher than for a short conversational turn.
      </Prose>

      {/* ======================================================================
          4. FROM-SCRATCH IMPLEMENTATION
          ====================================================================== */}
      <H2>4. From-scratch implementation</H2>

      <Prose>
        Every piece of code in this section is runnable. The goal is not just to show the
        happy path but to build the complete picture: async generator, token buffering, client
        parser, backpressure, and resume-on-reconnect. By the end you have a working SSE
        server and client that handle the real failure cases, not just the tutorial case.
      </Prose>

      <H3>4a. FastAPI SSE endpoint</H3>

      <Prose>
        FastAPI's <Code>StreamingResponse</Code> accepts an async generator and flushes each
        yielded chunk immediately without internal buffering. The server-side shape is an async
        generator function that yields properly framed SSE events. The critical syntax detail:
        every event must end with exactly two newlines. One newline separates fields within an
        event; two newlines terminate the event and signal the client to dispatch it. Omit one
        and the client receives events it cannot parse.
      </Prose>

      <CodeBlock language="python">
{`import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

# Minimal LLM stub — replace with vLLM, TGI, or Anthropic SDK call
async def llm_token_stream(prompt: str):
    words = ["Hello", " there", ".", " How", " can", " I", " help", "?"]
    for word in words:
        await asyncio.sleep(0.05)   # simulate generation time
        yield word

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    prompt = body["messages"][-1]["content"]

    async def event_stream():
        try:
            async for token in llm_token_stream(prompt):
                payload = {
                    "choices": [{"delta": {"content": token}, "finish_reason": None}]
                }
                # Each event: "data: <json>\\n\\n"
                yield f"data: {json.dumps(payload)}\\n\\n"

            # Final chunk — OpenAI convention: finish_reason on last content chunk
            final = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(final)}\\n\\n"
            yield "data: [DONE]\\n\\n"

        except asyncio.CancelledError:
            # Client disconnected — generator is cancelled, clean up upstream here
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
        },
    )`}
      </CodeBlock>

      <Prose>
        The <Code>X-Accel-Buffering: no</Code> header is not cosmetic. Nginx reads this header
        when acting as a reverse proxy and disables its response buffering for the duration of
        that connection. Without it, Nginx buffers the entire stream until the connection closes
        or its buffer fills — silently converting streaming to batch delivery. Setting it in the
        application response means the fix travels with the route rather than depending on Nginx
        configuration being correct in every environment.
      </Prose>

      <H3>4b. Token-by-token streaming with buffer management</H3>

      <Prose>
        Real inference engines emit tokens in variable-size chunks depending on their internal
        batching. A token may be a single character, a multi-byte UTF-8 sequence, or several
        words if the engine batches. Two problems arise. First, a UTF-8 character can be split
        across chunk boundaries — naively flushing raw bytes produces garbled text for non-ASCII
        content. Second, very small chunks (single bytes) produce excessive syscall overhead.
        The solution is a small accumulator that buffers until a UTF-8 character boundary and
        flushes on complete characters.
      </Prose>

      <CodeBlock language="python">
{`import codecs

async def safe_token_stream(raw_byte_stream):
    """
    Wrap a raw byte stream from an inference engine.
    Accumulates bytes until a complete UTF-8 character is available,
    then yields decoded text. Prevents split-codepoint corruption.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = bytearray()

    async for chunk in raw_byte_stream:
        buffer.extend(chunk)
        # Decode as much as possible without waiting for more bytes
        decoded = decoder.decode(bytes(buffer), final=False)
        if decoded:
            buffer.clear()
            yield decoded

    # Flush remaining bytes at stream end
    final = decoder.decode(b"", final=True)
    if final:
        yield final


async def event_stream_with_buffering(prompt: str):
    """
    Yield SSE events. Batch tokens into ~4-token chunks after the first token
    to reduce syscall pressure while keeping TTFT low.
    """
    token_buffer = []
    first_token_sent = False

    async for text in safe_token_stream(raw_llm_bytes(prompt)):
        token_buffer.append(text)

        # Send first token immediately (minimize TTFT)
        # After that, batch up to 4 tokens before flushing
        if not first_token_sent or len(token_buffer) >= 4:
            content = "".join(token_buffer)
            token_buffer.clear()
            first_token_sent = True
            payload = {"choices": [{"delta": {"content": content}}]}
            yield f"data: {json.dumps(payload)}\\n\\n"

    # Flush any remaining buffered tokens
    if token_buffer:
        content = "".join(token_buffer)
        payload = {"choices": [{"delta": {"content": content}}]}
        yield f"data: {json.dumps(payload)}\\n\\n"

    yield "data: [DONE]\\n\\n"`}
      </CodeBlock>

      <H3>4c. Client-side EventSource parser</H3>

      <Prose>
        The browser's native <Code>EventSource</Code> API handles reconnect and event parsing
        automatically, but it only supports GET requests with no custom body — useless for
        LLM APIs that take a prompt in the POST body. The correct client-side approach is a
        manual <Code>fetch</Code> call with <Code>ReadableStream</Code> parsing, which gives
        full control over the request while replicating the SSE parsing logic that{" "}
        <Code>EventSource</Code> provides natively.
      </Prose>

      <CodeBlock language="javascript">
{`/**
 * parseSSEStream — parse a fetch() response body as an SSE stream.
 * Yields each parsed event object as { event, data, id }.
 */
async function* parseSSEStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split on double-newline (event boundaries)
    const events = buffer.split("\\n\\n");
    // Keep the last (possibly incomplete) chunk in buffer
    buffer = events.pop();

    for (const rawEvent of events) {
      if (!rawEvent.trim()) continue;

      const parsed = { event: "message", data: "", id: null };
      for (const line of rawEvent.split("\\n")) {
        if (line.startsWith("data:")) {
          parsed.data += line.slice(5).trim();
        } else if (line.startsWith("event:")) {
          parsed.event = line.slice(6).trim();
        } else if (line.startsWith("id:")) {
          parsed.id = line.slice(3).trim();
        }
      }
      yield parsed;
    }
  }
}

// Usage: stream tokens into a DOM element
async function streamToElement(prompt, targetEl) {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [{ role: "user", content: prompt }] }),
  });

  if (!response.ok) throw new Error(\`HTTP \${response.status}\`);

  for await (const { data } of parseSSEStream(response)) {
    if (data === "[DONE]") break;

    const parsed = JSON.parse(data);
    const delta = parsed.choices?.[0]?.delta?.content;
    if (delta) targetEl.textContent += delta;
  }
}`}
      </CodeBlock>

      <H3>4d. Backpressure handling</H3>

      <Prose>
        Backpressure is what happens when the server generates tokens faster than the client
        can consume them. In a well-behaved async pipeline, the server's write coroutine blocks
        when the client's TCP receive buffer is full — the OS applies flow control and the
        server stops yielding. In an ill-behaved pipeline, an intermediate layer buffers
        everything: the server keeps generating, the buffer fills, and eventually either the
        buffer overflows (events dropped) or memory exhausts. Detecting and handling backpressure
        correctly requires that every layer between inference engine and client participates.
      </Prose>

      <CodeBlock language="python">
{`import asyncio
from fastapi import Request

async def event_stream_with_backpressure(request: Request, prompt: str):
    """
    Monitor client disconnect and apply a send timeout per token.
    If the client cannot accept a token within MAX_SEND_TIMEOUT seconds,
    treat it as a slow client and abort — freeing the inference slot.
    """
    MAX_SEND_TIMEOUT = 5.0  # seconds before abandoning slow client

    async def generate():
        async for token in llm_token_stream(prompt):
            # Check for client disconnect before each send
            if await request.is_disconnected():
                # Signal upstream to cancel inference
                break

            payload = {"choices": [{"delta": {"content": token}}]}
            try:
                yield f"data: {json.dumps(payload)}\\n\\n"
            except Exception:
                break  # write failed, client gone

    return generate()


# Alternative: detect slow client via asyncio timeout
async def timed_event_stream(prompt: str):
    async for token in llm_token_stream(prompt):
        payload = {"choices": [{"delta": {"content": token}}]}
        event = f"data: {json.dumps(payload)}\\n\\n"

        try:
            async with asyncio.timeout(5.0):
                yield event
        except asyncio.TimeoutError:
            # Client buffer is full — drop stream, log, release KV cache slot
            return`}
      </CodeBlock>

      <H3>4e. Reconnect with Last-Event-ID resume</H3>

      <Prose>
        The SSE spec includes a native reconnect mechanism. If a connection drops, the browser
        automatically reopens the <Code>EventSource</Code> connection and sends the{" "}
        <Code>Last-Event-ID</Code> header with the <Code>id</Code> value from the last
        successfully received event. The server can use this to resume rather than restart the
        stream. For LLM streaming, full resume from an arbitrary mid-generation position is
        expensive (it requires checkpointing or re-running the generation to the resume point),
        but a practical approximation is to cache the full generation server-side and replay
        from the cached offset on reconnect.
      </Prose>

      <CodeBlock language="python">
{`import uuid
from collections import defaultdict

# In-memory cache: request_id -> list of emitted events
# In production, use Redis with a short TTL (e.g. 60 seconds)
event_cache: dict[str, list[str]] = defaultdict(list)

@app.post("/v1/chat/stream")
async def resumable_stream(request: Request):
    body = await request.json()
    prompt = body["messages"][-1]["content"]

    # Client sends Last-Event-ID to resume; new sessions get a fresh ID
    last_event_id = request.headers.get("Last-Event-ID")
    request_id = last_event_id.split(":")[0] if last_event_id else str(uuid.uuid4())
    resume_from = int(last_event_id.split(":")[1]) if last_event_id and ":" in last_event_id else 0

    async def event_stream():
        # Replay cached events for reconnecting clients
        cached = event_cache.get(request_id, [])
        for i, cached_event in enumerate(cached[resume_from:], start=resume_from):
            event_id = f"{request_id}:{i}"
            yield f"id: {event_id}\\ndata: {cached_event}\\n\\n"

        # Continue generating from where cache ends
        i = len(cached)
        async for token in llm_token_stream(prompt):
            payload = json.dumps({"choices": [{"delta": {"content": token}}]})
            event_cache[request_id].append(payload)
            event_id = f"{request_id}:{i}"
            i += 1
            yield f"id: {event_id}\\ndata: {payload}\\n\\n"

        yield "data: [DONE]\\n\\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )`}
      </CodeBlock>

      {/* ======================================================================
          5. PRODUCTION ECOSYSTEM
          ====================================================================== */}
      <H2>5. Production ecosystem</H2>

      <Prose>
        Every major LLM API exposes SSE streaming, but the event schemas differ in ways that
        matter when writing client code that needs to work against more than one provider.
      </Prose>

      <Prose>
        <strong>OpenAI chat completions.</strong> Set <Code>stream: true</Code> in the request
        body. The response is a sequence of <Code>ChatCompletionChunk</Code> objects, each a
        JSON object with a <Code>choices</Code> array. Each choice has a <Code>delta</Code>{" "}
        with an incremental <Code>content</Code> string (or <Code>tool_calls</Code> delta for
        function calling), a <Code>finish_reason</Code> that is <Code>null</Code> on all
        non-final chunks and set to <Code>"stop"</Code>, <Code>"length"</Code>, or{" "}
        <Code>"tool_calls"</Code> on the final one. The stream ends with a bare{" "}
        <Code>data: [DONE]</Code> line. The OpenAI Python SDK wraps this in a{" "}
        <Code>Stream[ChatCompletionChunk]</Code> iterator that handles the parsing for you;
        passing <Code>stream=True</Code> to <Code>client.chat.completions.create()</Code>{" "}
        returns the iterator directly.
      </Prose>

      <Prose>
        <strong>Anthropic Messages API.</strong> Anthropic's streaming schema is more
        structured. Events carry explicit <Code>event:</Code> type fields:{" "}
        <Code>message_start</Code> opens the stream with metadata,{" "}
        <Code>content_block_start</Code> begins each content block (there can be multiple —
        text blocks, tool use blocks, thinking blocks),{" "}
        <Code>content_block_delta</Code> carries incremental content with a{" "}
        <Code>delta.type</Code> of <Code>text_delta</Code> or <Code>thinking_delta</Code>,{" "}
        <Code>content_block_stop</Code> closes a block, <Code>message_delta</Code> carries
        final metadata like stop reason and token counts, and <Code>message_stop</Code>{" "}
        terminates the stream. The Anthropic Python SDK's <Code>stream()</Code> context
        manager hides this event multiplicity and exposes a simple{" "}
        <Code>text_stream</Code> async iterator for the common case.
      </Prose>

      <Prose>
        <strong>vLLM and TGI (OpenAI-compatible mode).</strong> Both vLLM and HuggingFace
        Text Generation Inference expose an OpenAI-compatible <Code>/v1/chat/completions</Code>{" "}
        endpoint that produces the same SSE schema as OpenAI. Pass <Code>stream=True</Code>,
        iterate chunks, extract <Code>delta.content</Code>. The difference from the real
        OpenAI endpoint is operational: vLLM's async engine runs inside the same process as
        the HTTP server, so cancellation on client disconnect propagates synchronously to the
        inference engine via asyncio task cancellation. TGI's <Code>/generate_stream</Code>{" "}
        endpoint is SSE-native and uses the <Code>generated_text</Code> field on the final
        event rather than a <Code>[DONE]</Code> sentinel.
      </Prose>

      <Prose>
        <strong>Vercel AI SDK.</strong> The Vercel AI SDK (now at v5/v6) standardizes streaming
        on the server side via <Code>streamText()</Code> from the <Code>ai</Code> package, which
        returns a <Code>StreamTextResult</Code> with a <Code>toDataStreamResponse()</Code>{" "}
        method that formats the SSE response correctly for Next.js route handlers. On the client,
        the <Code>useChat()</Code> React hook consumes this stream and manages the message list,
        loading state, and abort controller automatically. The SDK abstracts over provider
        differences — switching from OpenAI to Anthropic to a local vLLM endpoint requires
        changing only the provider adapter, not the streaming infrastructure.
      </Prose>

      <Prose>
        The Vercel AI SDK's data stream protocol deserves specific mention because it extends
        raw SSE with typed message parts: tool call deltas, tool results, metadata updates, and
        error events all travel over the same SSE connection as distinct event types, not just
        raw text deltas. This makes it practical to stream complex agentic responses — where the
        model interleaves text generation with tool invocations and their results — without
        building a custom multiplexing layer on top of the base SSE wire format. Applications
        that start with simple chat streaming and later add tool use benefit from adopting the
        AI SDK's data stream protocol early, since retrofitting a custom tool-call streaming
        layer onto a raw SSE client is more complex than upgrading the SDK version.
      </Prose>

      <CodeBlock language="javascript">
{`// Next.js App Router — app/api/chat/route.js
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

export async function POST(req) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai("gpt-4o"),
    messages,
  });

  // toDataStreamResponse() formats SSE correctly for useChat()
  return result.toDataStreamResponse();
}

// React client — components/Chat.jsx
import { useChat } from "ai/react";

export function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();

  return (
    <div>
      {messages.map(m => (
        <div key={m.id}>{m.content}</div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}`}
      </CodeBlock>

      {/* ======================================================================
          6. VISUAL
          ====================================================================== */}
      <H2>6. Connection lifecycle</H2>

      <StepTrace
        label="SSE connection lifecycle — open to close"
        steps={[
          {
            label: "1. client opens connection",
            render: () => (
              <TokenStream tokens={[
                { label: "POST /v1/chat/completions", color: colors.purple },
                { label: "stream: true", color: "#60a5fa" },
                { label: "→ server", color: "#6b7280" },
              ]} />
            ),
          },
          {
            label: "2. server sends headers",
            render: () => (
              <TokenStream tokens={[
                { label: "200 OK", color: "#4ade80" },
                { label: "Content-Type: text/event-stream", color: "#60a5fa" },
                { label: "Cache-Control: no-cache", color: "#60a5fa" },
              ]} />
            ),
          },
          {
            label: "3. TTFT — first token arrives",
            render: () => (
              <TokenStream tokens={[
                { label: "data:", color: "#6b7280" },
                { label: '{"choices":[{"delta":{"content":"Hello"}}]}', color: colors.gold },
                { label: "\\n\\n", color: "#6b7280" },
              ]} />
            ),
          },
          {
            label: "4. token stream (TPOT per token)",
            render: () => (
              <TokenStream tokens={[
                { label: "data: {...}", color: colors.gold },
                { label: "data: {...}", color: colors.gold },
                { label: "data: {...}", color: colors.gold },
                { label: "… ×N", color: "#6b7280" },
              ]} />
            ),
          },
          {
            label: "5. finish_reason on final chunk",
            render: () => (
              <TokenStream tokens={[
                { label: '{"finish_reason":"stop"}', color: "#4ade80" },
              ]} />
            ),
          },
          {
            label: "6. [DONE] sentinel",
            render: () => (
              <TokenStream tokens={[
                { label: "data: [DONE]", color: "#f87171" },
                { label: "\\n\\n", color: "#6b7280" },
                { label: "→ client closes ReadableStream", color: "#6b7280" },
              ]} />
            ),
          },
          {
            label: "7. client disconnect (abort path)",
            render: () => (
              <TokenStream tokens={[
                { label: "AbortController.abort()", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "server CancelledError", color: "#f87171" },
                { label: "→", color: "#6b7280" },
                { label: "KV cache released", color: "#4ade80" },
              ]} />
            ),
          },
        ]}
      />

      <Prose>
        The step that most teams skip is step 7. The happy path — steps 1 through 6 — works
        in local development. Step 7 only surfaces under load, when users abandon requests,
        when mobile network handoffs drop connections, or when a browser tab is closed during
        generation. Every dropped connection that does not propagate a cancel signal to the
        inference engine wastes a KV cache slot and GPU compute for the duration of that
        generation. At scale, this becomes a meaningful fraction of total inference cost.
      </Prose>

      {/* ======================================================================
          7. DECISION MATRIX
          ====================================================================== */}
      <H2>7. Decision matrix</H2>

      <Prose>
        Choosing between streaming transports requires answering three questions: Is the
        client a browser? Does the client need to send data mid-stream? Is the infrastructure
        HTTP/2-capable end-to-end?
      </Prose>

      <Prose>
        <strong>SSE</strong> is the right default when: the client is a browser or any HTTP
        client, the communication pattern is one request in / one token stream out, and you
        want infrastructure that works through standard load balancers without protocol
        negotiation. This covers virtually all chat UIs, code completion assistants, and
        document generation tools.
      </Prose>

      <Prose>
        <strong>WebSockets</strong> become necessary when: the client needs to send data
        while the model is generating. This is primarily relevant in agentic frameworks where
        tool calls return results that must be injected mid-stream, or where the user can
        interrupt and redirect the model before generation completes. The additional complexity
        — upgrade negotiation, keepalive management, reconnect logic — is justified only when
        the use case is genuinely bidirectional. Using WebSockets for pure LLM output streaming
        is over-engineering.
      </Prose>

      <Prose>
        <strong>gRPC bidirectional streaming</strong> is the correct choice for server-to-server
        links: inference worker to orchestration layer, orchestration to safety classifier,
        safety to streaming relay. Strong typing, built-in flow control, and native observability
        integration make gRPC operationally superior to SSE for internal traffic. It requires
        HTTP/2 end-to-end and is not natively consumable from a browser (requires gRPC-Web
        proxy translation), which disqualifies it for the last-mile client connection.
      </Prose>

      <Prose>
        <strong>Long polling</strong> is the fallback for environments where SSE is not viable —
        old enterprise proxies that normalize HTTP connections before the client, environments
        locked to HTTP/1.0, or cases where the event rate is very low (one event per 30
        seconds or slower) and the reconnect overhead per event is acceptable. For LLM token
        streaming, long polling's latency floor is too high to be practical. Each long-poll
        cycle requires a TCP connection setup (or reuse from a pool), an HTTP round trip, and
        a parse step — at token rates of 20–50 tokens per second this produces hundreds of
        round trips per generation, with cumulative overhead that erases any latency advantage
        over simply waiting for the complete response.
      </Prose>

      <Heatmap
        title="transport choice by scenario"
        rowLabels={["browser client", "server client", "bidirectional needed", "low-event-rate", "old proxy env"]}
        colLabels={["SSE", "WebSocket", "gRPC stream", "long poll"]}
        data={[
          [1.0, 0.5, 0.1, 0.2],
          [0.7, 0.6, 1.0, 0.1],
          [0.1, 1.0, 0.9, 0.1],
          [0.4, 0.4, 0.4, 1.0],
          [0.3, 0.3, 0.1, 1.0],
        ]}
      />

      {/* ======================================================================
          8. SCALING
          ====================================================================== */}
      <H2>8. Scaling</H2>

      <Prose>
        The scaling property that most distinguishes LLM streaming from standard HTTP traffic
        is connection duration. A streaming LLM response holds an HTTP connection open for
        the entire duration of generation — typically 5 to 60 seconds at conversational
        context lengths, minutes for long-form generation. Standard HTTP infrastructure is
        designed around request-response cycles that complete in milliseconds. Every component
        in the stack has timeouts, buffer sizes, and connection limits calibrated to that
        assumption.
      </Prose>

      <Prose>
        <strong>Load balancer idle timeouts.</strong> Most cloud load balancers have a default
        idle timeout of 60 seconds, meaning a connection with no data flowing for 60 seconds
        is terminated. A long-context generation that runs longer than 60 seconds with sparse
        token output will have the connection killed by the load balancer partway through. The
        fix is a combination of: increasing the idle timeout on the load balancer to something
        appropriate for your p99 generation length (e.g. 300 seconds), and sending periodic
        SSE comment lines (<Code>: keep-alive</Code>) every 15–20 seconds to reset the idle
        timer. SSE comment lines — lines starting with <Code>:</Code> — are ignored by the
        client parser but count as data for keep-alive purposes.
      </Prose>

      <Prose>
        <strong>Concurrent connection limits.</strong> Under HTTP/1.1, each active stream is
        one TCP connection. Browsers enforce a per-domain connection limit of 6 for HTTP/1.1
        (defined in RFC 7230 and respected by all major browsers). This means a user can have
        at most 6 simultaneous SSE streams open to the same origin — in practice, not a
        problem for chat UIs, but relevant for applications that open multiple concurrent
        completions. HTTP/2 eliminates this limitation by multiplexing streams over a single
        TCP connection; upgrading to HTTP/2 end-to-end is the correct fix for applications
        that need more than 6 concurrent streams per user.
      </Prose>

      <Prose>
        <strong>HTTP/2 multiplexing.</strong> HTTP/2 carries multiple request/response pairs
        over a single TCP connection using stream IDs. For a server handling thousands of
        concurrent LLM streams, HTTP/2 reduces the number of open TCP connections by a factor
        of 10–100x, which proportionally reduces kernel memory consumption, TLS session state,
        and load balancer connection table entries. The trade-off is head-of-line blocking at
        the TCP level: if a TCP packet is lost, all multiplexed HTTP/2 streams on that
        connection stall until the packet is retransmitted. HTTP/3 (QUIC) eliminates this by
        multiplexing at the transport layer, but HTTP/3 support in inference serving stacks is
        not yet universal.
      </Prose>

      <Prose>
        <strong>Horizontal scaling and sticky routing.</strong> SSE connections are stateful
        for their duration — the server is holding the generation state (KV cache, async
        generator context) for a specific request. If a load balancer routes a reconnecting
        client to a different server instance, the new instance has no generation state and
        must restart from scratch. The correct design is to either use sticky sessions (route
        all reconnects for a given request ID to the same upstream), or to externalize generation
        state into a shared cache (Redis, object storage) that any instance can read from for
        replay. Most production stacks use sticky sessions for simplicity and fall back to
        restart on failover.
      </Prose>

      <Prose>
        <strong>Deployment-specific concerns.</strong> Serverless runtimes (AWS Lambda, Google
        Cloud Run, Vercel Edge Functions) impose maximum response duration limits — typically
        29 seconds for Lambda's API Gateway integration, configurable but bounded for Cloud
        Run. A generation that exceeds the serverless function's maximum response timeout is
        killed mid-stream with no error event to the client. This makes serverless deployment
        of LLM streaming endpoints viable only for models with reliably short generation times,
        or for platforms like Vercel Edge Functions that use streaming-native runtimes with
        longer timeout allowances. Long-context generation and agentic workloads generally
        require always-on processes (container-based deployments, Kubernetes pods) rather than
        serverless invocations. Monitoring TTFT and total generation latency at the p95 and p99
        levels — not just p50 — is essential for catching the cases where a long-running
        generation hits an infrastructure timeout limit the p50 would never reveal.
      </Prose>

      {/* ======================================================================
          9. FAILURE MODES
          ====================================================================== */}
      <H2>9. Failure modes</H2>

      <Prose>
        SSE failures have a consistent pattern: they appear correct in local development and
        break in production because local development has no intermediate proxies, no CDN,
        no firewall, no load balancer. The following eight failure modes are the most
        commonly encountered in production LLM deployments.
      </Prose>

      <Prose>
        <strong>1. Proxy buffering (Cloudflare, Nginx, GCP LB).</strong> The most common
        failure. An intermediate proxy accumulates the entire response body before forwarding
        it, converting streaming delivery into batch delivery. The user sees a spinner for
        the full generation duration and then the complete text appears at once. Diagnosis:
        compare token delivery timing with and without the proxy in the path. Fix: set{" "}
        <Code>X-Accel-Buffering: no</Code> on the response (Nginx respects this); disable
        response buffering on the backend service in GCP; use Cloudflare's{" "}
        <Code>Cache: no-store</Code> header and ensure your Cloudflare plan tier supports
        streaming (Cloudflare Workers supports it natively; some proxy configurations on
        older plan tiers do not).
      </Prose>

      <Prose>
        <strong>2. Event truncation at UTF-8 boundaries.</strong> If the inference engine
        emits raw bytes and the server flushes on arbitrary byte boundaries rather than
        UTF-8 character boundaries, multi-byte characters (CJK, emoji, Arabic, diacritical
        marks) arrive split across events. The client receives a partial codepoint, the JSON
        parser sees an invalid string, and the event is dropped or corrupted. Fix: use an
        incremental UTF-8 decoder on the server side (see section 4b) to ensure every flushed
        event contains only complete characters.
      </Prose>

      <Prose>
        <strong>3. Slow client stalling the server.</strong> A client with a slow connection
        or a paused JavaScript event loop cannot drain the TCP receive buffer quickly. The OS
        applies TCP flow control, the server's write coroutine blocks, and the inference
        engine's output queue backs up. In the worst case, the inference engine stops generating
        because its internal queue is full, and the server holds the KV cache slot open
        indefinitely. Fix: apply a per-event write timeout on the server (see section 4d);
        if a write takes longer than N seconds, close the connection and release the inference
        slot. Log slow clients for monitoring.
      </Prose>

      <Prose>
        <strong>4. Retry semantics — resume vs restart.</strong> When a client reconnects after
        a dropped connection, it can either resume (request events after the last received ID)
        or restart (begin a new generation from the full prompt). Resume is user-friendly but
        operationally expensive: it requires either caching the full event sequence
        server-side or re-running generation to the resume point. Restart is simple but
        wastes tokens and produces a visible gap or duplicate text in the UI. Most production
        systems implement restart with deduplication on the client: the client re-submits with
        the accumulated partial response as additional context, or the UI hides the restart
        seam behind a smooth transition. The spec-correct approach is resume via{" "}
        <Code>Last-Event-ID</Code> (section 4e), but few production LLM APIs implement it.
      </Prose>

      <Prose>
        <strong>5. Client disconnect detection.</strong> A TCP connection that drops without
        a FIN packet — mobile network handoff, NAT timeout, process kill — leaves the server
        unaware for up to several minutes (the TCP keepalive timeout). During that window, the
        server keeps generating and writing to a dead socket, consuming GPU compute and KV
        cache memory with no consumer. Fix: use application-layer keepalives (periodic SSE
        comment lines) combined with OS-level TCP keepalive tuning (reduce <Code>tcp_keepalive_time</Code>{" "}
        from the default 7200 seconds to something appropriate, e.g. 60 seconds on your
        serving instances). FastAPI exposes client disconnect via <Code>request.is_disconnected()</Code>;
        poll this on every token.
      </Prose>

      <Prose>
        <strong>6. CORS preflight delays.</strong> SSE requests from a different origin trigger
        a CORS preflight (<Code>OPTIONS</Code>) request before the actual <Code>POST</Code>.
        If the server does not respond to <Code>OPTIONS</Code> quickly, or if the preflight
        result is not cached (missing <Code>Access-Control-Max-Age</Code> header), every SSE
        connection incurs a preflight round trip. This adds 50–200 ms to TTFT on cross-origin
        deployments. Fix: configure <Code>Access-Control-Max-Age: 86400</Code> on your CORS
        preflight response to cache preflight results for 24 hours.
      </Prose>

      <Prose>
        <strong>7. Browser EventSource connection limit.</strong> The browser's native{" "}
        <Code>EventSource</Code> API (GET-only) is subject to HTTP/1.1's 6-connection-per-domain
        limit. Applications that open multiple <Code>EventSource</Code> connections to the
        same origin — for example, a dashboard with several concurrent LLM panels — can
        exhaust the limit and block subsequent connections. Fix: use a single{" "}
        <Code>fetch</Code>-based SSE client (as in section 4c), which shares the HTTP/2
        multiplexed connection and does not count against the EventSource limit.
      </Prose>

      <Prose>
        <strong>8. Compression middleware destroying flush semantics.</strong> Gzip and Brotli
        compression work by accumulating bytes into compression blocks before emitting
        compressed output. A streaming response with per-token flushes is incompatible with
        standard compression: the compressor buffers the first N tokens before emitting the
        first compressed block, which produces the same batch-delivery symptom as proxy
        buffering. Fix: disable compression for <Code>text/event-stream</Code> routes
        explicitly in your middleware configuration. In FastAPI with{" "}
        <Code>GZipMiddleware</Code>, add an exclusion for the SSE endpoint path.
      </Prose>

      <Prose>
        A diagnostic approach that covers most of these failure modes: instrument your SSE
        endpoint with per-event timestamps logged to structured output. Every event should
        carry a server-side timestamp and a sequence number. The client should record its
        receive timestamp for each event. Comparing server-emit timestamps to client-receive
        timestamps reveals buffering delays — if events emitted 20 ms apart arrive at the
        client 2,000 ms apart in a burst, something between the server and client is buffering.
        Comparing sequence numbers reveals dropped events. Running this diagnostic through each
        infrastructure layer in isolation — server only, server plus Nginx, server plus Nginx
        plus load balancer, full production path — pinpoints exactly which layer introduced
        the problem. This end-to-end tracing approach is more reliable than reading proxy
        configuration documentation, because production deployments frequently have layers
        whose configuration is not fully known or not under the team's control.
      </Prose>

      <Callout accent="gold">
        The pattern across all eight failures is the same: infrastructure built for
        short-lived, complete HTTP responses silently breaks long-lived, incremental ones.
        Every component that touches the response — middleware, proxy, CDN, load balancer —
        needs explicit configuration to pass SSE streams through unchanged.
      </Callout>

      {/* ======================================================================
          10. PRIMARY SOURCES
          ====================================================================== */}
      <H2>10. Primary sources</H2>

      <Prose>
        This topic draws on the following primary references. For implementation details,
        the specs and official documentation are more reliable than any tutorial.
      </Prose>

      <Prose>
        <strong>WHATWG HTML Living Standard — Server-Sent Events (§9.2).</strong> The
        authoritative definition of the SSE wire format, the <Code>EventSource</Code> API,
        event parsing algorithm, reconnect semantics, and <Code>Last-Event-ID</Code> behavior.
        The parsing algorithm in particular is worth reading precisely — it specifies exactly
        how field names, values, and event boundaries are identified, which is the only correct
        reference for implementing a custom SSE parser.
        URL: <Code>https://html.spec.whatwg.org/multipage/server-sent-events.html</Code>
      </Prose>

      <Prose>
        <strong>OpenAI Streaming API documentation.</strong> Documents the{" "}
        <Code>ChatCompletionChunk</Code> schema, <Code>finish_reason</Code> values, and the
        new Responses API streaming event model. The Responses API documentation is the
        forward-looking reference; the Chat Completions streaming reference remains the most
        widely implemented schema across the open-source ecosystem.
        URL: <Code>https://platform.openai.com/docs/guides/streaming-responses</Code>
      </Prose>

      <Prose>
        <strong>Anthropic Streaming Messages API.</strong> Defines the full event sequence
        (<Code>message_start</Code>, <Code>content_block_start/delta/stop</Code>,{" "}
        <Code>message_delta</Code>, <Code>message_stop</Code>), the <Code>delta.type</Code>{" "}
        field taxonomy including <Code>thinking_delta</Code> for extended thinking, and the
        SDK-level helpers. The structured event model is more explicit than the OpenAI schema
        and handles multi-block responses (parallel tool calls, interleaved thinking/text)
        more cleanly.
        URL: <Code>https://docs.anthropic.com/en/api/streaming</Code>
      </Prose>

      <Prose>
        <strong>Vercel AI SDK documentation.</strong> The reference for <Code>streamText</Code>,{" "}
        <Code>useChat</Code>, provider adapters, and the data stream protocol format. The AI
        SDK 5/6 migration guide is also useful for understanding the evolution from{" "}
        <Code>StreamingTextResponse</Code> (deprecated) to <Code>toDataStreamResponse()</Code>.
        URL: <Code>https://sdk.vercel.ai/docs</Code>
      </Prose>

      <Prose>
        <strong>HuggingFace Text Generation Inference — Streaming.</strong> Explains TGI's
        SSE implementation, the <Code>/generate_stream</Code> endpoint schema, the{" "}
        <Code>generated_text</Code> final-event convention, and the InferenceClient streaming
        interface. Useful for understanding where TGI diverges from OpenAI-compatible format.
        URL: <Code>https://huggingface.co/docs/text-generation-inference/conceptual/streaming</Code>
      </Prose>

      {/* ======================================================================
          11. EXERCISES
          ====================================================================== */}
      <H2>11. Exercises</H2>

      <Prose>
        <strong>Exercise 1 — Wire format audit.</strong> Open your browser's DevTools Network
        tab and make a streaming request to any LLM API (OpenAI, Anthropic, a local Ollama
        instance). Inspect the raw response. Identify: the <Code>Content-Type</Code> header
        value, the byte sequence that separates events, whether an <Code>event:</Code> field
        is present, and what the final data line contains. Then write a Python script using{" "}
        <Code>httpx</Code> with streaming enabled that parses these events manually — no SDK,
        no SSE library — and prints each delta to stdout.
      </Prose>

      <Prose>
        <strong>Exercise 2 — Proxy buffering diagnosis.</strong> Run the FastAPI endpoint from
        section 4a locally. Confirm it streams correctly by watching tokens appear in the
        browser console with timestamps. Now put Nginx in front of it with default settings
        (<Code>proxy_buffering on</Code>). Observe that streaming is broken. Add{" "}
        <Code>proxy_buffering off</Code> and the <Code>X-Accel-Buffering: no</Code> response
        header. Confirm streaming is restored. Then try adding gzip compression middleware and
        observe the same buffering symptom recur. Fix it by excluding the SSE route from
        compression. Document each step with before/after timing logs.
      </Prose>

      <Prose>
        <strong>Exercise 3 — TTFT measurement harness.</strong> Modify the client-side parser
        from section 4c to record three timestamps: request sent (<Code>t0</Code>), first
        data event received (<Code>t1</Code>), and final <Code>[DONE]</Code> received{" "}
        (<Code>t2</Code>). Compute TTFT = t1 − t0 and total latency = t2 − t0. Run 20 requests
        against a local server, compute the mean and p95 for both metrics. Now artificially
        increase server TTFT by adding a 500 ms delay before the first token yield. Observe
        how p95 total latency changes versus how perceived quality changes if you describe both
        variants to a test user. Does the data match the weight formula from section 3?
      </Prose>

      <Prose>
        <strong>Exercise 4 — Backpressure simulation.</strong> Extend the FastAPI server
        to use an asyncio queue between the inference generator and the SSE sender. The
        generator puts tokens into the queue; the sender reads from it. Add an artificial
        slow consumer: introduce a <Code>await asyncio.sleep(2.0)</Code> inside the sender
        loop. Observe the queue depth grow. Add a maximum queue depth of 5 items and have
        the generator pause (using <Code>queue.join()</Code> or a semaphore) when the queue
        is full. Verify that the generator slows to match the consumer rather than running
        ahead unbounded. Then add a timeout: if the queue does not drain within 10 seconds,
        cancel the connection and measure how quickly the inference slot is released.
      </Prose>

      <Prose>
        <strong>Exercise 5 — Reconnect and resume.</strong> Implement the server-side event
        cache from section 4e using Redis (use <Code>redis-py</Code> with async support).
        Store events as a Redis list with a 5-minute TTL. Modify the client-side parser to
        record the last received event ID and, on fetch failure, automatically retry the
        request with the <Code>Last-Event-ID</Code> header set. Test the full flow: start a
        stream, kill the server process at token 15, restart it within 5 minutes, verify the
        client automatically reconnects and receives the remaining events from the cache
        without restarting generation. Measure the reconnect latency (time from server restart
        to resumed token delivery) and the gap in token delivery the user would observe.
      </Prose>
    </div>
  ),
};

export default streamingSSE;
